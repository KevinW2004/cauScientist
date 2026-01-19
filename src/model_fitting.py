"""
Model Fitting Module - MLP增强版
使用MLP模型提升因果图拟合度
直接复用 ENCO 的模型和评估方法
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
from collections import defaultdict
import sys
import os

# 导入 ENCO 的模型
sys.path.append(os.path.join(os.path.dirname(__file__), 'methods', 'ENCO'))
from causal_discovery.multivariable_mlp import create_model
from causal_discovery.multivariable_flow import create_continuous_model


class SimpleCausalModel(nn.Module):
    """
    因果模型 - 直接使用 ENCO 的模型
    参考 enco.py:123-131 的模型选择逻辑
    """
    def __init__(self, structured_graph: Dict, data: np.ndarray = None, 
                 is_categorical: bool = None, num_categs: int = None,
                 hidden_dims: List[int] = [64], use_flow_model: bool = False):
        super().__init__()
        self.graph = structured_graph
        self.nodes = {node['name']: node for node in structured_graph['nodes']}
        self.variable_names = [node['name'] for node in structured_graph['nodes']]
        self.num_vars = len(self.variable_names)
        
        # 必须明确提供变量类型，不使用自动推断
        if is_categorical is None:
            raise ValueError("is_categorical must be explicitly provided. Use variable_type parameter in fit().")
        
        self.is_categorical = is_categorical
        
        # 存储每个变量的状态数（用于 BIC 计算）
        self.var_num_states = None
        
        # 如果是分类变量，必须提供类别数（或从数据检测）
        if is_categorical:
            if num_categs is None:
                if data is None:
                    raise ValueError("num_categs must be provided for categorical/discrete variables, or data must be provided to detect it")
                # 从数据中检测类别数
                max_categs = 0
                var_num_states = []
                for i in range(data.shape[1]):
                    unique_vals = len(np.unique(data[:, i]))
                    var_num_states.append(unique_vals)
                    max_categs = max(max_categs, unique_vals)
                num_categs = max_categs
                self.var_num_states = var_num_states  # 存储每个变量的状态数
                print(f"[SimpleCausalModel] Detected num_categs from data: {num_categs}")
                print(f"[SimpleCausalModel] Variable states: {var_num_states}")
        
        # 按照 enco.py:123-131 的方式创建模型
        if self.is_categorical:
            print(f"[Model] Using categorical model with {num_categs} categories")
            self.model = create_model(num_vars=self.num_vars,
                                     num_categs=num_categs,
                                     hidden_dims=hidden_dims)
        else:
            print(f"[Model] Using continuous model (flow={use_flow_model})")
            self.model = create_continuous_model(num_vars=self.num_vars,
                                                hidden_dims=hidden_dims,
                                                use_flow_model=use_flow_model)
        
        self.var_to_idx = {var: idx for idx, var in enumerate(self.variable_names)}
    
    def _graph_to_adj_matrix(self) -> torch.Tensor:
        """将因果图转换为邻接矩阵"""
        adj_matrix = torch.zeros(self.num_vars, self.num_vars)
        for node in self.graph['nodes']:
            child_name = node['name']
            child_idx = self.var_to_idx[child_name]
            parents = node.get('parents', [])
            for parent_name in parents:
                parent_idx = self.var_to_idx[parent_name]
                adj_matrix[parent_idx, child_idx] = 1
        return adj_matrix
    
    def forward(self, data: torch.Tensor, interventions: torch.Tensor = None) -> torch.Tensor:
        """
        计算对数似然
        参考 graph_fitting.py:290-315 的 evaluate_likelihoods 实现
        
        Args:
            data: 数据 [batch_size, n_variables]
            interventions: 干预矩阵 [batch_size, n_variables]，1表示该变量被干预
        
        返回负对数似然（用于最小化）
        """
        # 获取邻接矩阵作为 mask
        adj_matrix = self._graph_to_adj_matrix().to(data.device)
        # Transpose for mask because adj[i,j] means that i->j
        mask_adj_matrix = adj_matrix.T.unsqueeze(0).expand(data.shape[0], -1, -1)
        
        # 调用 ENCO 模型
        preds = self.model(data, mask=mask_adj_matrix)
        
        # 按照 graph_fitting.py:304-312 的方式评估似然
        if data.dtype == torch.long:
            # 离散变量：使用交叉熵
            preds = preds.flatten(0, 1)
            labels = data.clone()
            
            # 对于被干预的变量，设置 label 为 -1（ignore_index）
            if interventions is not None:
                labels = labels.float()  # 转为 float 以便修改
                labels[interventions > 0] = -1
            
            labels = labels.long().reshape(-1)
            nll = F.cross_entropy(preds, labels, reduction='none', ignore_index=-1)
            nll = nll.reshape(*data.shape)
        else:
            # 连续变量：模型直接返回 NLL
            nll = preds
            
            # 对于被干预的变量，mask掉其 NLL
            if interventions is not None:
                nll = nll * (1 - interventions)
        
        # 返回负对数似然（注意：我们返回负的，因为 ENCO 返回的是 NLL）
        # 为了与之前的代码一致（最大化 log likelihood），我们返回 -nll
        return -nll.sum(dim=1)
    
    def compute_bic(self, data: torch.Tensor, interventions: torch.Tensor = None) -> float:
        """
        计算BIC score: BIC = -2 * log_likelihood + k_eff * log(n)
        
        关键：k_eff 基于图结构（边数和节点配置），而非神经网络参数总数
        
        Args:
            data: 数据 [n_samples, n_variables]
            interventions: 干预矩阵 [n_samples, n_variables]
        
        越小越好
        """
        with torch.no_grad():
            log_likelihood = self.forward(data, interventions).mean().item()
            n = data.shape[0]
            
            # 计算有效参数数量：基于图的边数和节点配置
            # 对于离散贝叶斯网络：每个节点的参数数 = (r_i - 1) * q_i
            # 其中 r_i = 该节点的状态数，q_i = 父节点状态数的乘积
            k_eff = 0
            
            if self.is_categorical:
                # 离散变量：计算每个节点的条件概率表（CPT）大小
                for idx, node in enumerate(self.graph['nodes']):
                    parents = node.get('parents', [])
                    num_parents = len(parents)
                    
                    # 获取该节点的状态数
                    if self.var_num_states is not None:
                        r_i = self.var_num_states[idx]  # 该节点的状态数
                    else:
                        # 如果没有存储，使用保守估计
                        r_i = 3
                    
                    # 计算父节点配置数 q_i = Π(r_parent)
                    if num_parents > 0:
                        q_i = 1
                        for parent_name in parents:
                            parent_idx = self.variable_names.index(parent_name)
                            if self.var_num_states is not None:
                                r_parent = self.var_num_states[parent_idx]
                            else:
                                r_parent = 3
                            q_i *= r_parent
                    else:
                        q_i = 1
                    
                    # CPT 参数数量：(r_i - 1) * q_i
                    node_params = (r_i - 1) * q_i
                    k_eff += node_params
            else:
                # 连续变量：每条边贡献固定参数
                # 对于线性模型：每条边贡献 1 个权重参数
                # 每个节点贡献 1 个偏置参数和 1 个方差参数
                for node in self.graph['nodes']:
                    parents = node.get('parents', [])
                    num_parents = len(parents)
                    
                    # 参数：num_parents 个权重 + 1 个偏置 + 1 个方差
                    node_params = num_parents + 2
                    k_eff += node_params
            
            bic = -2 * n * log_likelihood + k_eff * np.log(n)
            
            # 打印调试信息（可选）
            # print(f"[BIC] n={n}, LL={log_likelihood:.4f}, k_eff={k_eff}, BIC={bic:.2f}")
            
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
        interventions: np.ndarray = None,
        variable_type: str = None,
        is_categorical: bool = None,
        num_categs: int = None,
        hidden_dims: List[int] = [64],
        use_flow_model: bool = False,
        num_epochs: int = 100,
        learning_rate: float = 0.01,
        verbose: bool = False,
        seed: int = None
    ) -> Dict:
        """
        训练模型并返回拟合度
        使用 ENCO 的模型
        
        Args:
            structured_graph: 结构化的因果图
            data: 数据 [n_samples, n_variables]
            interventions: 干预矩阵 [n_samples, n_variables]，1表示该变量被干预
            variable_type: 变量类型 ("continuous" 或 "discrete")，优先使用此参数
            is_categorical: 是否为分类变量（None=根据variable_type推断）
            num_categs: 类别数（分类变量时需要）
            hidden_dims: 隐藏层维度
            use_flow_model: 是否使用 flow 模型（仅连续变量）
            num_epochs: 训练轮数
            learning_rate: 学习率
            verbose: 是否打印详细信息
            seed: 随机种子
        
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
        
        # 必须提供 variable_type
        if variable_type is None:
            raise ValueError(
                "variable_type must be provided ('continuous' or 'discrete'). "
                "This should come from dataset.variable_type, which is defined in DOMAIN_VARIABLE_TYPES in data_loader.py"
            )
        
        # 根据 variable_type 设置 is_categorical
        if is_categorical is None:
            is_categorical = (variable_type == "discrete")
            print(f"[Model Fitting] Variable type: {variable_type} -> is_categorical={is_categorical}")
        
        print(f"\n[Model Fitting] Training ENCO-based model for {num_epochs} epochs...")
        
        # 显示是否使用干预信息
        if interventions is not None:
            n_intervened = (interventions.sum(axis=1) > 0).sum()
            n_total = len(interventions)
            print(f"[Model Fitting] Using intervention information: {n_intervened}/{n_total} samples have interventions")
        else:
            print(f"[Model Fitting] No intervention information provided")
        
        # 创建模型（会自动推断数据类型）
        try:
            self.model = SimpleCausalModel(
                structured_graph, 
                data,
                is_categorical=is_categorical,
                num_categs=num_categs,
                hidden_dims=hidden_dims,
                use_flow_model=use_flow_model
            ).to(self.device)
        except Exception as e:
            print(f"\n❌ ERROR: Failed to create model: {e}")
            raise
        
        # 数据预处理：根据模型类型处理数据
        if self.model.is_categorical:
            # 离散变量：转为 long 类型，不标准化
            # 确保数据是整数索引 0, 1, 2, ...
            data_processed = data.copy()
            for i in range(data.shape[1]):
                unique_vals = np.unique(data[:, i])
                if not np.array_equal(unique_vals, np.arange(len(unique_vals))):
                    # 重新映射为 0, 1, 2, ...
                    value_map = {v: idx for idx, v in enumerate(sorted(unique_vals))}
                    data_processed[:, i] = np.array([value_map[v] for v in data[:, i]])
            
            data_tensor = torch.LongTensor(data_processed.astype(int)).to(self.device)
            print(f"[Model Fitting] Using categorical data (no standardization)")
        else:
            # 连续变量：标准化
            data_mean = data.mean(axis=0, keepdims=True)
            data_std = data.std(axis=0, keepdims=True) + 1e-8
            data_normalized = (data - data_mean) / data_std
            data_tensor = torch.FloatTensor(data_normalized).to(self.device)
            print(f"[Model Fitting] Standardized continuous data: mean={data_normalized.mean():.4f}, std={data_normalized.std():.4f}")
        
        # 转换干预矩阵
        interventions_tensor = None
        if interventions is not None:
            interventions_tensor = torch.FloatTensor(interventions).to(self.device)
        
        # 优化器
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 训练
        self.training_history = []
        for epoch in range(num_epochs):
            log_likelihood = self.model(data_tensor, interventions_tensor)
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
            final_likelihood = self.model(data_tensor, interventions_tensor)
            F_t = final_likelihood.mean().item()
            bic = self.model.compute_bic(data_tensor, interventions_tensor)
            num_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"[Model Fitting] Final Log-Likelihood F_t = {F_t:.4f}")
        print(f"[Model Fitting] BIC Score = {bic:.4f}")
        print(f"[Model Fitting] Number of Parameters = {num_params}")
        
        results = {
            'log_likelihood': -bic,
            'bic': bic,
            'num_parameters': num_params,
            'training_history': self.training_history
        }
        
        return results