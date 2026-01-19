"""
Model Fitting Module - MLP增强版
使用MLP模型提升因果图拟合度
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List
from collections import defaultdict


class SmallMLP(nn.Module):
    """
    小型 MLP + 正则化，用于建模非线性因果关系
    结构: input_dim -> 10 -> 1
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.Tanh(),  # 使用 Tanh（有界，防止爆炸）
            nn.Linear(10, 1)
        )
    
    def forward(self, x):
        return self.network(x).squeeze(-1)
    
    def l1_penalty(self):
        """L1 正则化（稀疏性）"""
        return sum(p.abs().sum() for p in self.parameters())
    
    def l2_penalty(self):
        """L2 正则化（权重衰减）"""
        return sum((p**2).sum() for p in self.parameters())


class SimpleCausalModel(nn.Module):
    """
    因果模型 - 使用MLP建模非线性关系
    每个变量都建模为高斯分布，方差可学习
    """
    def __init__(self, structured_graph: Dict, data: np.ndarray = None):
        super().__init__()
        self.graph = structured_graph
        self.nodes = {node['name']: node for node in structured_graph['nodes']}
        self.variable_names = [node['name'] for node in structured_graph['nodes']]
        
        # 拓扑排序
        self.topological_order = self._topological_sort()
        
        # 为每个变量创建模型
        self.models = nn.ModuleDict()
        self.log_stds = nn.ParameterDict()  # 可学习的log标准差
        self.root_params = nn.ParameterDict()  # 根节点参数
        
        for var_name in self.topological_order:
            node = self.nodes[var_name]
            parents = node.get('parents', [])
            
            if len(parents) > 0:
                # 有父节点: 使用小 MLP 建模 E[X|Parents]
                self.models[var_name] = SmallMLP(len(parents))
                
                # 可学习的条件方差 (log空间，保证正值)
                if data is not None:
                    # 用数据初始化方差
                    var_idx = self.variable_names.index(var_name)
                    init_std = np.std(data[:, var_idx])
                    self.log_stds[var_name] = nn.Parameter(
                        torch.log(torch.tensor(init_std + 1e-6))
                    )
                else:
                    self.log_stds[var_name] = nn.Parameter(torch.zeros(1))
            else:
                # 无父节点: 只需要均值和方差
                if data is not None:
                    # 用数据初始化
                    var_idx = self.variable_names.index(var_name)
                    init_mean = np.mean(data[:, var_idx])
                    init_std = np.std(data[:, var_idx])
                    self.root_params[var_name + '_mean'] = nn.Parameter(
                        torch.tensor([init_mean], dtype=torch.float32)
                    )
                    self.root_params[var_name + '_log_std'] = nn.Parameter(
                        torch.log(torch.tensor([init_std + 1e-6], dtype=torch.float32))
                    )
                else:
                    self.root_params[var_name + '_mean'] = nn.Parameter(torch.zeros(1))
                    self.root_params[var_name + '_log_std'] = nn.Parameter(torch.zeros(1))
        
        self.var_to_idx = {var: idx for idx, var in enumerate(self.variable_names)}
    
    def _topological_sort(self) -> List[str]:
        """拓扑排序"""
        in_degree = defaultdict(int)
        graph = defaultdict(list)
        
        for node in self.graph['nodes']:
            var_name = node['name']
            parents = node.get('parents', [])
            in_degree[var_name] = len(parents)
            
            for parent in parents:
                graph[parent].append(var_name)
        
        queue = [var for var in self.variable_names if in_degree[var] == 0]
        result = []
        
        while queue:
            var = queue.pop(0)
            result.append(var)
            
            for child in graph[var]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        
        if len(result) != len(self.variable_names):
            raise ValueError("Graph contains cycles!")
        
        return result
    
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        计算对数似然
        
        返回每个样本的log p(x) = Σ_i log p(x_i | parents(x_i))
        """
        batch_size = data.shape[0]
        log_likelihood = torch.zeros(batch_size, device=data.device)
        
        for var_name in self.topological_order:
            var_idx = self.var_to_idx[var_name]
            x = data[:, var_idx]
            
            parents = self.nodes[var_name].get('parents', [])
            
            if len(parents) > 0:
                # 有父节点: 条件高斯 X | Parents ~ N(f_MLP(Parents), σ²)
                parent_indices = [self.var_to_idx[p] for p in parents]
                parent_values = data[:, parent_indices]
                
                # MLP预测均值
                pred_mean = self.models[var_name](parent_values)
                
                # 可学习的标准差
                log_std = self.log_stds[var_name]
                std = torch.exp(log_std) + 1e-6
                
                # 高斯对数似然: log N(x; μ, σ²)
                log_prob = -0.5 * (
                    torch.log(torch.tensor(2 * np.pi, device=data.device)) + 
                    2 * log_std + 
                    ((x - pred_mean) / std) ** 2
                )
                log_likelihood += log_prob
            else:
                # 无父节点: 边缘高斯 X ~ N(μ, σ²)
                mean = self.root_params[var_name + '_mean']
                log_std = self.root_params[var_name + '_log_std']
                std = torch.exp(log_std) + 1e-6
                
                log_prob = -0.5 * (
                    torch.log(torch.tensor(2 * np.pi, device=data.device)) + 
                    2 * log_std + 
                    ((x - mean) / std) ** 2
                )
                log_likelihood += log_prob
        
        return log_likelihood
    
    def compute_bic(self, data: torch.Tensor) -> float:
        """
        计算BIC score: BIC = -2 * log_likelihood + k * log(n)
        
        越小越好
        """
        with torch.no_grad():
            log_likelihood = self.forward(data).mean().item()
            n = data.shape[0]
            k = sum(p.numel() for p in self.parameters())
            
            bic = -2 * n * log_likelihood + k * np.log(n)
            
        return bic


class ModelFittingEngine:
    """模型拟合引擎 - MLP版本"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = None
        self.training_history = []
    
    def fit(
        self,
        structured_graph: Dict,
        data: np.ndarray,
        num_epochs: int = 100,
        learning_rate: float = 0.01,
        verbose: bool = False,
        seed: int = None
    ) -> Dict:
        """
        训练模型并返回拟合度
        
        Returns:
            Dict包含:
            - log_likelihood: 平均对数似然
            - bic: BIC score
            - num_parameters: 参数数量
        """
        
        # 设置随机种子以保证可复现性
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        print(f"\n[Model Fitting] Training MLP model for {num_epochs} epochs...")
        
        # 创建模型（用数据初始化）
        self.model = SimpleCausalModel(structured_graph, data).to(self.device)
        
        # 转换数据
        data_tensor = torch.FloatTensor(data).to(self.device)
        
        # 优化器
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 训练
        self.training_history = []
        for epoch in range(num_epochs):
            log_likelihood = self.model(data_tensor)
            loss = -log_likelihood.mean()  # 最大化对数似然 = 最小化负对数似然
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            self.training_history.append({
                'epoch': epoch,
                'log_likelihood': log_likelihood.mean().item()
            })
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, "
                      f"Log-Likelihood: {log_likelihood.mean().item():.4f}")
        
        # 最终评估
        self.model.eval()
        with torch.no_grad():
            final_likelihood = self.model(data_tensor)
            F_t = final_likelihood.mean().item()
            bic = self.model.compute_bic(data_tensor)
            num_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"[Model Fitting] Final Log-Likelihood F_t = {F_t:.4f}")
        print(f"[Model Fitting] BIC Score = {bic:.4f}")
        print(f"[Model Fitting] Number of Parameters = {num_params}")
        
        results = {
            'log_likelihood': F_t,
            'bic': bic,
            'num_parameters': num_params,
            'training_history': self.training_history
        }
        
        return results
    
    def fit_with_cv(
        self,
        structured_graph: Dict,
        data: np.ndarray,
        num_epochs: int = 100,
        learning_rate: float = 0.01,
        l1_lambda: float = 0.01,
        l2_lambda: float = 0.001,
        k_folds: int = 5,
        verbose: bool = False,
        seed: int = None
    ) -> Dict:
        """
        K-fold Cross-Validation 训练和评估
        
        Args:
            l1_lambda: L1 正则化强度
            l2_lambda: L2 正则化强度
            k_folds: CV 折数
            
        Returns:
            Dict包含:
            - cv_log_likelihood: CV 平均对数似然
            - cv_std: CV 标准差
            - train_log_likelihood: 训练集 LL
            - bic: BIC score
        """
        from sklearn.model_selection import KFold
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        print(f"\n[Model Fitting] K-Fold CV (k={k_folds}) with regularization...")
        print(f"  L1 lambda: {l1_lambda}, L2 lambda: {l2_lambda}")
        
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        cv_scores = []
        
        fold = 0
        for train_idx, val_idx in kf.split(data):
            fold += 1
            
            # 分割数据
            train_data = data[train_idx]
            val_data = data[val_idx]
            
            # 创建模型
            model = SimpleCausalModel(structured_graph, train_data).to(self.device)
            train_tensor = torch.FloatTensor(train_data).to(self.device)
            val_tensor = torch.FloatTensor(val_data).to(self.device)
            
            # 优化器
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
            # 训练
            for epoch in range(num_epochs):
                model.train()
                log_likelihood = model(train_tensor)
                
                # 计算正则化
                l1_penalty = 0
                l2_penalty = 0
                for var_name, mlp in model.models.items():
                    l1_penalty += mlp.l1_penalty()
                    l2_penalty += mlp.l2_penalty()
                
                # 总损失 = 负对数似然 + 正则化
                loss = -log_likelihood.mean() + l1_lambda * l1_penalty + l2_lambda * l2_penalty
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # 验证集评估
            model.eval()
            with torch.no_grad():
                val_likelihood = model(val_tensor)
                val_score = val_likelihood.mean().item()
                cv_scores.append(val_score)
            
            if verbose:
                print(f"  Fold {fold}/{k_folds}: Val LL = {val_score:.4f}")
        
        # CV 结果
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        print(f"[Model Fitting] CV Log-Likelihood: {cv_mean:.4f} ± {cv_std:.4f}")
        
        # 在全数据上重新训练（用于最终模型和 BIC）
        print(f"[Model Fitting] Retraining on full data...")
        self.model = SimpleCausalModel(structured_graph, data).to(self.device)
        data_tensor = torch.FloatTensor(data).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.training_history = []
        for epoch in range(num_epochs):
            log_likelihood = self.model(data_tensor)
            
            # 正则化
            l1_penalty = 0
            l2_penalty = 0
            for var_name, mlp in self.model.models.items():
                l1_penalty += mlp.l1_penalty()
                l2_penalty += mlp.l2_penalty()
            
            loss = -log_likelihood.mean() + l1_lambda * l1_penalty + l2_lambda * l2_penalty
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            self.training_history.append({
                'epoch': epoch,
                'log_likelihood': log_likelihood.mean().item(),
                'l1_penalty': l1_penalty.item(),
                'l2_penalty': l2_penalty.item()
            })
        
        # 最终评估
        self.model.eval()
        with torch.no_grad():
            final_likelihood = self.model(data_tensor)
            train_ll = final_likelihood.mean().item()
            bic = self.model.compute_bic(data_tensor)
            num_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"[Model Fitting] Final Train LL = {train_ll:.4f}")
        print(f"[Model Fitting] CV Score (validation) = {cv_mean:.4f}")
        print(f"[Model Fitting] BIC Score = {bic:.4f}")
        print(f"[Model Fitting] Number of Parameters = {num_params}")
        
        results = {
            'cv_log_likelihood': cv_mean,  # 主要评分指标
            'cv_std': cv_std,
            'train_log_likelihood': train_ll,
            'bic': bic,
            'num_parameters': num_params,
            'training_history': self.training_history
        }
        
        return results