from typing import Dict
import numpy as np
from schemas.causal_graph import StructuredGraph

def shd_metric(pred, target):
    """
    Calculates the structural hamming distance
    
    Parameters:
    -----------
    pred: ndarray
        The predicted adjacency matrix (n_variables x n_variables)
    target: ndarray
        The true adjacency matrix (n_variables x n_variables)
    
    Returns:
    --------
    shd: int
        Structural Hamming Distance
    """
    true_labels = target
    predictions = pred
    
    diff = true_labels - predictions
    
    # Reversed edges: edges that exist in both but in opposite directions
    rev = (((diff + diff.T) == 0) & (diff != 0)).sum() / 2
    
    # False negatives: edges in true but not in pred (excluding reversed)
    fn = (diff == 1).sum() - rev
    
    # False positives: edges in pred but not in true (excluding reversed)
    fp = (diff == -1).sum() - rev
    
    return int(fn + fp + rev)

def compute_metrics(pipeline, predicted_graph: StructuredGraph) -> Dict:
    """计算评估指标 - 添加SHD"""
    
    if pipeline.dataset is None:
        return {}
    
    # 构建预测的邻接矩阵
    n_vars = len(pipeline.variable_list)
    pred_adj_matrix = np.zeros((n_vars, n_vars), dtype=int)
    
    predicted_edges = set()
    for node in predicted_graph.nodes:
        child = node['name']
        child_idx = pipeline.variable_list.index(child)
        for parent in node.get('parents', []):
            parent_idx = pipeline.variable_list.index(parent)
            predicted_edges.add((parent_idx, child_idx))
            pred_adj_matrix[parent_idx, child_idx] = 1
    
    # 提取真实的边
    true_edges = set()
    for i in range(pipeline.dataset.n_variables):
        for j in range(pipeline.dataset.n_variables):
            if pipeline.dataset.ground_truth_graph[i, j] == 1:
                true_edges.add((i, j))
    
    # 计算基础指标
    true_positive = len(predicted_edges & true_edges)
    false_positive = len(predicted_edges - true_edges)
    false_negative = len(true_edges - predicted_edges)
    true_negative = (pipeline.dataset.n_variables ** 2 - 
                    len(true_edges) - len(predicted_edges) + true_positive)
    
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positive + true_negative) / (pipeline.dataset.n_variables ** 2)
    
    # 计算SHD
    shd = shd_metric(pred_adj_matrix, pipeline.dataset.ground_truth_graph)
    
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "shd": shd  # 新增
    }
