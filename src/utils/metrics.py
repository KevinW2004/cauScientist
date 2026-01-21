import numpy as np
from typing import Dict, List, Tuple, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
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

def compute_precision_recall_f1(pred_adj_matrix: np.ndarray, true_adj_matrix: np.ndarray, threshold: float = 0.5) -> Dict:
    """
    计算precision, recall和F1 score
    
    Parameters:
    -----------
    pred_adj_matrix: ndarray
        预测的邻接矩阵 (n_variables x n_variables)，可以是连续值或二值
    true_adj_matrix: ndarray
        真实的邻接矩阵 (n_variables x n_variables)，二值
    threshold: float
        用于二值化预测矩阵的阈值（如果预测是连续值）
        
    Returns:
    --------
    dict: 包含precision, recall, f1_score等指标
    """
    # 二值化预测矩阵（如果需要）
    pred_binary = (pred_adj_matrix > threshold).astype(int)
    true_binary = true_adj_matrix.astype(int)
    
    n_vars = pred_binary.shape[0]
    
    # 提取预测的边
    predicted_edges = set()
    for i in range(n_vars):
        for j in range(n_vars):
            if pred_binary[i, j] == 1:
                predicted_edges.add((i, j))
    
    # 提取真实的边
    true_edges = set()
    for i in range(n_vars):
        for j in range(n_vars):
            if true_binary[i, j] == 1:
                true_edges.add((i, j))
    
    # 计算基础指标
    true_positive = len(predicted_edges & true_edges)
    false_positive = len(predicted_edges - true_edges)
    false_negative = len(true_edges - predicted_edges)
    true_negative = (n_vars ** 2 - len(true_edges) - len(predicted_edges) + true_positive)
    
    # 计算precision, recall, f1
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positive + true_negative) / (n_vars ** 2)
    
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(accuracy, 4)
    }

def compute_metrics(pipeline, predicted_graph: "StructuredGraph") -> Dict:
    """计算评估指标 - 添加SHD"""
    
    if pipeline.dataset is None:
        return {}
    
    # 构建预测的邻接矩阵
    n_vars = len(pipeline.variable_list)
    pred_adj_matrix = np.zeros((n_vars, n_vars), dtype=int)
    
    predicted_edges = set()
    for node in predicted_graph.nodes:
        child = node.name
        child_idx = pipeline.variable_list.index(child)
        for parent in node.parents:
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

def has_cycle(nodes: List[Dict]) -> Tuple[bool, List[str]]:
    """检查是否有环（DFS算法），并记录环路信息"""
    
    # 构建邻接表
    graph = defaultdict(list)
    all_nodes = set()
    
    for node in nodes:
        node_name = node['name']
        all_nodes.add(node_name)
        
        for parent in node.get('parents', []):
            all_nodes.add(parent)
            graph[parent].append(node_name)
    
    # DFS检测环，并记录环路
    visited = set()
    rec_stack = set()
    cycle_path = []  # 记录环路路径
    found_cycle = False
    
    def dfs(node, path):
        nonlocal found_cycle, cycle_path
        if found_cycle:
            return True
            
        visited.add(node)
        rec_stack.add(node)
        path.append(node)
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                if dfs(neighbor, path):
                    return True
            elif neighbor in rec_stack:
                # 找到环！记录从neighbor到当前节点的路径
                cycle_start_idx = path.index(neighbor)
                cycle_path = path[cycle_start_idx:] + [neighbor]
                found_cycle = True
                return True
        
        rec_stack.remove(node)
        path.pop()
        return False
    
    for node in all_nodes:
        if node not in visited:
            if dfs(node, []):
                return True, cycle_path
    
    return False, cycle_path

def break_cycles(nodes: List[Dict], cycle_path: List[str]) -> List[Dict]:
    """
    打破环（简单策略：找到环路中权重最小的边，移除它， 迭代直到无环）
    """
    print("  Attempting to break cycles by removing edges...")
    has_cycle_result = True
    while has_cycle_result:
        min_weight = float('inf')
        min_edge = None
        for i,node in enumerate(nodes):
            if node['name'] in cycle_path:
                for parent in node['parents']:
                    if parent in cycle_path:
                        if node['weight'][parent] < min_weight:
                            min_weight = node['weight'][parent]
                            min_edge = (parent, i, node['name'])
        assert min_edge is not None, "This should not happen"
        assert nodes[min_edge[1]]['name'] == min_edge[2], "This should not happen"
        nodes[min_edge[1]]['parents'].remove(min_edge[0])
        print(f"  Removed edge: {min_edge[0]} → {min_edge[2]} with weight {min_weight}")
        has_cycle_result, cycle_path = has_cycle(nodes)
    print("  All cycles broken")
    return nodes