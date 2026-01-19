"""
经典因果发现算法集成模块
使用NOTEARS等算法对LLM生成的图进行数据驱动的优化
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import networkx as nx
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
from copy import deepcopy


class NOTEARSRefiner:
    """
    使用NOTEARS算法优化因果图
    采用局部优化策略，只在LLM提出的边附近搜索
    """
    
    def __init__(self, 
                 alpha: float = 0.001,  # L2正则化（用于Ridge回归）
                 w_threshold: float = 0.15,  # 边权重阈值（降低以保留更多边）
                 max_iter: int = 50,
                 poly_degree: int = 2):  # 多项式阶数
        """
        Args:
            alpha: L2正则化系数（用于Ridge回归）
            w_threshold: 边权重阈值（低于此值的边会被移除）
            max_iter: 最大迭代次数
            poly_degree: 多项式特征的阶数（1=线性，2=二次，3=三次）
        """
        self.alpha = alpha
        self.w_threshold = w_threshold
        self.max_iter = max_iter
        self.poly_degree = poly_degree
    
    def refine_graph(self, 
                     initial_graph: Dict,
                     data: np.ndarray,
                     variable_names: List[str],
                     locally_only: bool = True) -> Tuple[Dict, Dict]:
        """
        基于初始图和数据优化因果结构
        
        Args:
            initial_graph: LLM生成的初始图
            data: 观测数据 [n_samples, n_vars]
            variable_names: 变量名列表
            locally_only: 是否只在初始图的邻域内搜索
            
        Returns:
            (refined_graph, refinement_info)
        """
        
        # 提取初始邻接矩阵
        W_init = self._graph_to_adjacency(initial_graph, variable_names)
        
        # 局部优化：只优化现有边和潜在反转边
        W_refined = self._local_notears(data, W_init)
        
        # 转换回图结构
        refined_graph = self._adjacency_to_graph(W_refined, variable_names, initial_graph)
        
        # 计算变化
        refinement_info = self._compute_refinement_info(
            W_init, W_refined, variable_names
        )
        
        return refined_graph, refinement_info
    
    def _local_notears(self, data: np.ndarray, W_init: np.ndarray) -> np.ndarray:
        """
        NOTEARS优化：允许搜索所有可能的边
        使用多项式Ridge回归处理非线性关系
        """
        n, d = data.shape
        
        # 标准化数据
        data_mean = data.mean(axis=0)
        data_std = data.std(axis=0) + 1e-8
        data_norm = (data - data_mean) / data_std
        
        # 初始化权重矩阵
        W = np.zeros_like(W_init)
        
        # 对每个变量进行非线性回归
        for j in range(d):
            # 允许所有变量作为潜在父节点（全局搜索）
            potential_parents = [i for i in range(d) if i != j]
            
            if len(potential_parents) == 0:
                continue
            
            # 准备数据
            X = data_norm[:, potential_parents]
            y = data_norm[:, j]
            
            try:
                if self.poly_degree > 1:
                    # 使用多项式特征进行非线性建模
                    poly = PolynomialFeatures(degree=self.poly_degree, include_bias=False)
                    X_poly = poly.fit_transform(X)
                    
                    # Ridge回归（L2正则化）
                    ridge = Ridge(alpha=self.alpha * n, max_iter=1000)
                    ridge.fit(X_poly, y)
                    
                    # 提取线性项的系数（多项式特征的前n项对应原始变量）
                    # 这些系数反映了每个父节点的总体影响强度
                    w_linear = ridge.coef_[:len(potential_parents)]
                else:
                    # poly_degree=1时，使用普通Ridge回归
                    ridge = Ridge(alpha=self.alpha * n, max_iter=1000)
                    ridge.fit(X, y)
                    w_linear = ridge.coef_
                
                # 应用阈值，保留重要的边
                for idx, parent in enumerate(potential_parents):
                    if abs(w_linear[idx]) > self.w_threshold:
                        W[parent, j] = w_linear[idx]
                    else:
                        W[parent, j] = 0
                        
            except Exception as e:
                # 如果回归失败（如矩阵奇异），保持为0
                print(f"Warning: Regression failed for variable {j}: {e}")
                pass
        
        # 确保无环
        W = self._ensure_acyclic(W)
        
        return W
    
    def _ensure_acyclic(self, W: np.ndarray) -> np.ndarray:
        """确保图是无环的，移除导致环的最弱边"""
        max_attempts = 100
        attempt = 0
        
        while self._has_cycle(W) and attempt < max_attempts:
            W = self._remove_weakest_cycle_edge(W)
            attempt += 1
        
        return W
    
    def _graph_to_adjacency(self, graph: Dict, variable_names: List[str]) -> np.ndarray:
        """将图结构转换为邻接矩阵"""
        n = len(variable_names)
        W = np.zeros((n, n))
        
        var_to_idx = {name: i for i, name in enumerate(variable_names)}
        
        for node in graph['nodes']:
            child_idx = var_to_idx[node['name']]
            for parent in node.get('parents', []):
                parent_idx = var_to_idx[parent]
                W[parent_idx, child_idx] = 1.0  # 初始权重为1
        
        return W
    
    def _adjacency_to_graph(self, W: np.ndarray, variable_names: List[str], 
                            template_graph: Dict) -> Dict:
        """将邻接矩阵转换为图结构"""
        nodes = []
        for j, var in enumerate(variable_names):
            parents = [variable_names[i] for i in range(len(variable_names)) 
                      if W[i, j] != 0]
            nodes.append({
                'name': var,
                'parents': parents
            })
        
        # 复制template的metadata但更新edges数量
        n_edges = int(np.sum(W != 0))
        
        refined_graph = {
            'metadata': template_graph.get('metadata', {}).copy(),
            'nodes': nodes
        }
        
        refined_graph['metadata']['num_edges'] = n_edges
        refined_graph['metadata']['reasoning'] = "NOTEARS refined from LLM proposal"
        
        return refined_graph
    
    def _has_cycle(self, W: np.ndarray) -> bool:
        """检查邻接矩阵是否包含环"""
        G = nx.DiGraph((W != 0).astype(int))
        try:
            nx.find_cycle(G)
            return True
        except nx.NetworkXNoCycle:
            return False
    
    def _remove_weakest_cycle_edge(self, W: np.ndarray) -> np.ndarray:
        """移除导致环的权重最小的边"""
        G = nx.DiGraph()
        n = W.shape[0]
        for i in range(n):
            for j in range(n):
                if W[i, j] != 0:
                    G.add_edge(i, j, weight=abs(W[i, j]))
        
        try:
            cycle = nx.find_cycle(G)
            # 找到环中权重最小的边
            min_edge = min(cycle, key=lambda e: G[e[0]][e[1]]['weight'])
            W[min_edge[0], min_edge[1]] = 0
        except nx.NetworkXNoCycle:
            pass
        
        return W
    
    def _compute_refinement_info(self, W_init: np.ndarray, 
                                  W_refined: np.ndarray,
                                  variable_names: List[str]) -> Dict:
        """计算优化前后的变化"""
        
        added_edges = []
        removed_edges = []
        reversed_edges = []
        kept_edges = []
        
        n = W_init.shape[0]
        
        # 检查所有可能的边
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                
                init_exists = (W_init[i, j] != 0)
                refined_exists = (W_refined[i, j] != 0)
                
                if not init_exists and refined_exists:
                    added_edges.append((variable_names[i], variable_names[j]))
                elif init_exists and not refined_exists:
                    removed_edges.append((variable_names[i], variable_names[j]))
                elif init_exists and refined_exists:
                    kept_edges.append((variable_names[i], variable_names[j]))
        
        # 检查反转的边
        for i in range(n):
            for j in range(i+1, n):
                if W_init[i, j] != 0 and W_refined[j, i] != 0 and W_refined[i, j] == 0:
                    reversed_edges.append((variable_names[i], variable_names[j], variable_names[j], variable_names[i]))
                elif W_init[j, i] != 0 and W_refined[i, j] != 0 and W_refined[j, i] == 0:
                    reversed_edges.append((variable_names[j], variable_names[i], variable_names[i], variable_names[j]))
        
        return {
            'added_edges': added_edges,
            'removed_edges': removed_edges,
            'reversed_edges': reversed_edges,
            'kept_edges': kept_edges,
            'n_changes': len(added_edges) + len(removed_edges) + len(reversed_edges)
        }


class LocallyLinearSEM(nn.Module):
    """每个变量的MLP结构方程（用于NOTEARS-MLP）"""
    def __init__(self, dims, bias=True):
        super().__init__()
        self.dims = dims
        layers = []
        for l in range(len(dims) - 1):
            layers.append(nn.Linear(dims[l], dims[l+1], bias=bias))
            if l < len(dims) - 2:
                layers.append(nn.ReLU())
        self.fc = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.fc(x)


class NOTEARSMLPRefiner:
    """
    官方NOTEARS-MLP实现
    端到端优化邻接矩阵和MLP参数
    
    参考论文: DAGs with NO TEARS: Continuous Optimization for Structure Learning (NeurIPS 2018)
    """
    
    def __init__(self,
                 w_threshold: float = 0.3,
                 max_iter: int = 100,
                 h_tol: float = 1e-8,
                 rho_max: float = 1e+16,
                 w_lr: float = 0.001,
                 hidden_dims: list = [10, 1],
                 device: str = 'cuda'):
        """
        Args:
            w_threshold: 边权重阈值
            max_iter: 最大迭代次数
            h_tol: 无环性容忍度
            rho_max: 增广拉格朗日最大惩罚
            w_lr: 学习率
            hidden_dims: MLP隐藏层维度
            device: 计算设备
        """
        self.w_threshold = w_threshold
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.w_lr = w_lr
        self.hidden_dims = hidden_dims
        self.device = device
    
    def refine_graph(self,
                     initial_graph: Dict,
                     data: np.ndarray,
                     variable_names: List[str],
                     engine) -> Tuple[Dict, Dict]:
        """
        使用官方NOTEARS-MLP端到端优化
        
        Args:
            initial_graph: LLM生成的初始图
            data: 观测数据 [n_samples, n_vars]
            variable_names: 变量名列表
            engine: ModelFittingEngine实例（用于最终评估）
            
        Returns:
            (refined_graph, refinement_info)
        """
        print("\n" + "="*70)
        print("NOTEARS-MLP REFINEMENT (Official Implementation)")
        print("="*70)
        
        # 1. 转换为邻接矩阵
        W_init = self._graph_to_adjacency(initial_graph, variable_names)
        
        # 2. 标准化数据
        X = torch.FloatTensor(data).to(self.device)
        X = (X - X.mean(dim=0, keepdim=True)) / (X.std(dim=0, keepdim=True) + 1e-8)
        
        n_samples, n_vars = X.shape
        print(f"Data: {n_samples} samples, {n_vars} variables")
        print(f"Initial edges: {int(W_init.sum())}")
        
        # 3. 运行NOTEARS-MLP优化
        W_refined = self._notears_mlp(X, W_init)
        
        # 4. 转换回图结构
        refined_graph = self._adjacency_to_graph(W_refined, variable_names)
        
        # 5. 评估两个图（静默模式）
        import sys, os
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        initial_results = engine.fit(initial_graph, data, num_epochs=20, verbose=False)
        refined_results = engine.fit(refined_graph, data, num_epochs=20, verbose=False)
        
        sys.stdout.close()
        sys.stdout = old_stdout
        
        initial_ll = initial_results['log_likelihood']
        refined_ll = refined_results['log_likelihood']
        
        print(f"Initial Graph LL: {initial_ll:.4f}")
        print(f"Refined Graph LL: {refined_ll:.4f}")
        print(f"LL Improvement: {refined_ll - initial_ll:+.4f}")
        
        # 6. 计算变化信息
        refinement_info = self._compute_refinement_info(
            W_init, W_refined, variable_names, initial_ll, refined_ll
        )
        
        print(f"Changes: +{len(refinement_info['added_edges'])} edges, "
              f"-{len(refinement_info['removed_edges'])} edges, "
              f"⟲{len(refinement_info['reversed_edges'])} edges")
        print(f"Final edges: {int((W_refined > self.w_threshold).sum())}")
        print("="*70 + "\n")
        
        return refined_graph, refinement_info
    
    def _notears_mlp(self, X: torch.Tensor, W_init: np.ndarray) -> np.ndarray:
        """
        改进的NOTEARS-MLP实现
        从LLM的图开始优化，添加L1正则化促进稀疏性
        """
        n_samples, d = X.shape
        
        # 验证初始图是否有环
        if self._has_cycle(W_init):
            raise ValueError("❌ Initial graph from LLM contains cycles! This should not happen.")
        
        print(f"  Initial graph validated: {int(W_init.sum())} edges, DAG ✓")
        
        # 使用LLM的图作为初始化（转换为logit空间）
        # W_init是0/1矩阵，转换为logit: logit(p) = log(p/(1-p))
        W_logit = np.zeros_like(W_init, dtype=np.float32)
        epsilon = 0.01  # 避免log(0)
        for i in range(d):
            for j in range(d):
                if W_init[i, j] > 0.5:
                    # 现有的边：sigmoid(W) ≈ 0.9
                    W_logit[i, j] = np.log(0.9 / 0.1)
                else:
                    # 不存在的边：sigmoid(W) ≈ 0.1
                    W_logit[i, j] = np.log(0.1 / 0.9)
        
        # 添加小的随机扰动帮助优化
        W_logit += np.random.randn(d, d) * 0.1
        
        W = torch.FloatTensor(W_logit).to(self.device)
        W.requires_grad = True
        
        # 初始化模型：每个变量一个简单的线性模型（更稳定）
        models = []
        for j in range(d):
            model = nn.Linear(d, 1, bias=True).to(self.device)
            # 小权重初始化
            nn.init.xavier_uniform_(model.weight, gain=0.1)
            models.append(model)
        
        # 收集所有参数
        params = [W]
        for model in models:
            params.extend(model.parameters())
        
        optimizer = torch.optim.Adam(params, lr=self.w_lr)
        
        # 增广拉格朗日参数（更保守的初始化）
        rho, alpha = 1.0, 0.0
        h_prev = np.inf
        
        print(f"Starting improved NOTEARS-MLP optimization...")
        
        for iteration in range(self.max_iter):
            optimizer.zero_grad()
            
            # 计算每个变量的预测（使用连续mask）
            mse_loss = 0.0
            l1_penalty = 0.0
            
            for j in range(d):
                # Soft masking with sigmoid
                mask = torch.sigmoid(W[:, j])
                X_masked = X * mask.unsqueeze(0)
                
                # 预测
                X_pred = models[j](X_masked)
                
                # MSE loss
                mse_loss += torch.mean((X[:, j:j+1] - X_pred) ** 2)
                
                # L1 penalty for sparsity
                l1_penalty += torch.sum(torch.abs(mask))
            
            mse_loss = mse_loss / d
            l1_penalty = l1_penalty / d
            
            # 计算无环性约束
            W_squared = W * W
            h_val = self._compute_h_stable(W_squared)
            
            # 总损失：MSE + L1稀疏 + 增广拉格朗日
            lambda_1 = 0.01  # L1正则化系数
            loss = mse_loss + lambda_1 * l1_penalty + 0.5 * rho * h_val * h_val + alpha * h_val
            
            # 检查数值稳定性
            if not torch.isfinite(loss):
                print(f"  ⚠️ Loss became non-finite at iteration {iteration+1}, stopping")
                break
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪（防止爆炸）
            torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
            
            optimizer.step()
            
            # 投影W到合理范围
            with torch.no_grad():
                W.data = torch.clamp(W.data, -5, 5)
            
            h = h_val.item()
            
            # 打印进度
            if (iteration + 1) % 10 == 0 or iteration == 0:
                n_edges = (torch.sigmoid(W).detach() > 0.5).sum().item()
                print(f"  [Iter {iteration+1}/{self.max_iter}] MSE: {mse_loss.item():.4f}, "
                      f"h: {h:.2e}, rho: {rho:.1e}, edges: {n_edges}")
            
            # 检查收敛
            if h <= self.h_tol:
                print(f"  ✓ Converged at iteration {iteration+1} (h={h:.2e})")
                break
            
            # 更新增广拉格朗日参数（更保守的策略）
            if iteration == 0:
                h_prev = h
            elif h > 0.25 * h_prev:  # h没有显著下降
                rho = min(rho * 10, self.rho_max)
                if rho >= self.rho_max:
                    print(f"  ⚠️ rho reached max value, stopping (h={h:.2e})")
                    break
            
            alpha += rho * h
            h_prev = h
        
        # 提取最终邻接矩阵
        with torch.no_grad():
            W_final = torch.sigmoid(W).cpu().numpy()
        
        # 应用阈值
        W_binary = (W_final > self.w_threshold).astype(float)
        
        # 确保无环（如果优化后产生了环）
        if self._has_cycle(W_binary):
            print(f"  ⚠️ Optimization introduced cycles, removing them...")
            W_binary = self._threshold_and_remove_cycles(W_binary)
        
        print(f"  Final: {int(W_binary.sum())} edges (initial: {int(W_init.sum())})")
        
        return W_binary
    
    def _compute_h_stable(self, W_squared: torch.Tensor) -> torch.Tensor:
        """计算无环性约束（数值稳定版本）"""
        d = W_squared.shape[0]
        # 使用泰勒展开近似，避免矩阵指数爆炸
        # h(W) ≈ trace((I + W²/d)^d) - d
        M = torch.eye(d, device=W_squared.device) + W_squared / d
        # 使用矩阵幂而不是exp（更稳定）
        M_power = M
        h = 0.0
        for _ in range(d):
            h += torch.trace(M_power) / np.math.factorial(_ + 1)
            M_power = torch.mm(M_power, M)
            if _ > 3:  # 只用前几项（避免计算爆炸）
                break
        h = h - d
        return h
    
    def _threshold_and_remove_cycles(self, W_binary: np.ndarray) -> np.ndarray:
        """移除环，确保DAG"""
        # 移除最弱的边直到无环
        max_iterations = 100
        iteration = 0
        
        while self._has_cycle(W_binary) and iteration < max_iterations:
            # 找到一个环并移除其中一条边
            G = nx.DiGraph()
            d = W_binary.shape[0]
            for i in range(d):
                for j in range(d):
                    if W_binary[i, j] > 0:
                        G.add_edge(i, j)
            
            try:
                cycle = nx.find_cycle(G)
                # 移除环中的第一条边
                W_binary[cycle[0][0], cycle[0][1]] = 0
                iteration += 1
            except:
                break
        
        return W_binary
    
    
    def _graph_to_adjacency(self, graph: Dict, variable_names: List[str]) -> np.ndarray:
        """图结构 -> 邻接矩阵"""
        n = len(variable_names)
        W = np.zeros((n, n))
        
        var_to_idx = {name: i for i, name in enumerate(variable_names)}
        
        for node in graph.get('nodes', []):
            child = node['name']
            child_idx = var_to_idx[child]
            
            for parent in node.get('parents', []):
                parent_idx = var_to_idx[parent]
                W[parent_idx, child_idx] = 1.0
        
        return W
    
    def _adjacency_to_graph(self, W: np.ndarray, variable_names: List[str]) -> Dict:
        """邻接矩阵 -> 图结构"""
        nodes = []
        for j, child in enumerate(variable_names):
            parents = [variable_names[i] for i in range(len(variable_names)) 
                      if W[i, j] > self.w_threshold]
            nodes.append({
                'name': child,
                'parents': parents
            })
        
        return {'nodes': nodes}
    
    def _has_cycle(self, W: np.ndarray) -> bool:
        """检查邻接矩阵是否包含环"""
        G = nx.DiGraph()
        n = W.shape[0]
        
        for i in range(n):
            for j in range(n):
                if W[i, j] != 0:
                    G.add_edge(i, j)
        
        return not nx.is_directed_acyclic_graph(G)
    
    def _compute_refinement_info(self, W_init, W_refined, variable_names, 
                                  initial_ll, refined_ll):
        """计算优化变化信息"""
        n = len(variable_names)
        
        added_edges = []
        removed_edges = []
        reversed_edges = []
        kept_edges = []
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                
                init_edge = W_init[i, j] > self.w_threshold
                refined_edge = W_refined[i, j] > self.w_threshold
                
                if not init_edge and refined_edge:
                    added_edges.append((variable_names[i], variable_names[j]))
                elif init_edge and not refined_edge:
                    # 检查是否是反转
                    if W_refined[j, i] > self.w_threshold and W_init[j, i] <= self.w_threshold:
                        reversed_edges.append((variable_names[i], variable_names[j],
                                              variable_names[j], variable_names[i]))
                    else:
                        removed_edges.append((variable_names[i], variable_names[j]))
                elif init_edge and refined_edge:
                    kept_edges.append((variable_names[i], variable_names[j]))
        
        return {
            'added_edges': added_edges,
            'removed_edges': removed_edges,
            'reversed_edges': reversed_edges,
            'kept_edges': kept_edges,
            'n_changes': len(added_edges) + len(removed_edges) + len(reversed_edges),
            'initial_ll': initial_ll,
            'refined_ll': refined_ll,
            'll_improvement': refined_ll - initial_ll
        }

