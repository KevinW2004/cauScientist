"""
Score Functions for Causal Discovery
使用标准的 Linear Gaussian BIC
"""

import numpy as np
from typing import Dict, List
from causallearn.score.LocalScoreFunctionClass import local_score_BIC
from schemas.causal_graph import StructuredGraph


def score_graph_with_bic(
    structured_graph: StructuredGraph,
    data: np.ndarray,
    variable_names: List[str]
) -> Dict:
    """
    使用标准 Linear Gaussian BIC 评分
    
    这是因果发现领域的金标准方法，基于：
    - 线性高斯假设：X_i = sum(β_j * Pa_j) + ε, ε ~ N(0, σ²)
    - BIC 惩罚：自动平衡拟合度和复杂度
    
    Args:
        structured_graph: 图结构（StructuredGraph schema）
        data: 数据 [n_samples, n_variables]
        variable_names: 变量名列表
        
    Returns:
        Dict包含:
        - cv_log_likelihood: BIC score（越大越好）
        - bic: 传统 BIC（越小越好）
        - num_parameters: 参数数量
        - method: 方法名称
    """
    
    print(f"\n[Score Function] Computing Linear Gaussian BIC...")
    
    total_score = 0.0
    num_params = 0
    
    for node in structured_graph.nodes:
        child = node.name
        parents = node.parents
        
        child_idx = variable_names.index(child)
        parent_indices = [variable_names.index(p) for p in parents]
        
        # 使用 causal-learn 的标准 BIC
        # 注意：local_score_BIC 返回的是 score（越大越好）
        # 内部使用最小二乘法拟合线性模型
        local_score = local_score_BIC(data, child_idx, parent_indices)
        
        # 确保是标量（防止返回数组）
        if isinstance(local_score, np.ndarray):
            local_score = float(local_score.item())
        else:
            local_score = float(local_score)
        
        total_score += local_score
        
        # 计算参数数量：|parents| + 1 (intercept) + 1 (variance)
        num_params += len(parents) + 2
    
    print(f"[Score Function] Total BIC Score: {total_score:.4f}")
    print(f"[Score Function] Number of Parameters: {num_params}")
    
    results = {
        'cv_log_likelihood': total_score,  # 用 BIC score 替代（越大越好）
        'bic': -total_score,  # 传统 BIC（越小越好）
        'num_parameters': num_params,
        'method': 'Linear_Gaussian_BIC'
    }
    
    return results


def score_graph_simple(
    structured_graph: StructuredGraph,
    data: np.ndarray,
    variable_names: List[str]
) -> float:
    """
    简化版本，直接返回 score
    
    Returns:
        float: BIC score（越大越好）
    """
    return score_graph_with_bic(structured_graph, data, variable_names)['cv_log_likelihood']

