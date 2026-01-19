"""
Baseline Reference Module
从预先计算的传统因果发现方法结果中加载参考信息，作为LLM的参考
"""

import numpy as np
import pandas as pd
import os
import glob
from typing import List, Dict, Tuple, Optional
from src.utils.metrics import has_cycle, break_cycles
import warnings
warnings.filterwarnings('ignore')


class BaselineReferenceGenerator:
    """加载和格式化传统方法的参考信息供LLM使用"""
    
    # 定义方法类型：有向方法（输出因果图）vs 无向方法（输出关联性）
    DIRECTED_METHODS = ['notears', 'avici', 'sdcd', 'drbo']  # 输出有向边的方法
    UNDIRECTED_METHODS = ['corr', 'invcov']  # 输出无向关联的方法
    
    METHOD_NAMES = {
        'corr': 'Correlation (Pearson)',
        'invcov': 'Inverse Covariance (Precision Matrix)',
        'notears': 'NOTEARS (Causal Discovery)',
        'avici': 'AVICI (Deep Learning)',
        'sdcd': 'SDCD (Causal Discovery)',
        'drbo': 'DrBO (Bayesian Optimization)'
    }
    
    def __init__(self, methods: List[str] = None, predict_dir: str = None):
        """
        Args:
            methods: 要使用的方法列表，如 ['corr', 'invcov', 'notears']
            predict_dir: 预测结果目录（如 'predict/'）
        """
        assert methods is not None, "methods cannot be None"
        self.methods = methods
        self.predict_dir = predict_dir
        self.data_name_dict = {
            'sachs': 'bnlearn_sachs',
            'alarm': 'bnlearn_alarm',
            'asia': 'bnlearn_asia',
            'child': 'bnlearn_child',
            'cancer': 'bnlearn_cancer',
            'earthquake': 'bnlearn_earthquake',
            'insurance': 'bnlearn_insurance',
            'water': 'bnlearn_water'
        }
    
    def load_predictions_from_file(
        self,
        dataset_name: str,
        method_name: str,
        variable_list: List[str]
    ) -> Optional[np.ndarray]:
        """
        从 predict 目录加载预测结果
        
        Args:
            dataset_name: 数据集名称（如 'sachs', 'child'）
            method_name: 方法名称（如 'corr', 'invcov', 'notears'）
            variable_list: 变量名列表
        
        Returns:
            预测的邻接矩阵，如果找不到则返回 None
        """# /mnt/shared-storage-user/pengbo/created/projects/CDLLM/Test-1213/predict/graphs/sample_0000_bnlearn_alarm_avici_pred_binary.csv
        assert self.predict_dir is not None, "predict_dir cannot be None"
        assert os.path.exists(self.predict_dir), "predict_dir does not exist"
        # 在 graphs 目录中查找对应的预测文件
        graphs_dir = os.path.join(self.predict_dir, "graphs")
        assert os.path.exists(graphs_dir), "graphs_dir does not exist"
        data_alias = self.data_name_dict[dataset_name]
        pred_file = os.path.join(graphs_dir, f"{data_alias}_{method_name}_pred.csv")
        pred_matrix = pd.read_csv(pred_file, header=None).values
            
        # 检查维度
        n_vars = len(variable_list)
        assert pred_matrix.shape == (n_vars, n_vars), f"Matrix shape mismatch: expected {n_vars}x{n_vars}, got {pred_matrix.shape}"
        return pred_matrix
    
    def load_all_predictions(
        self,
        dataset_name: str,
        variable_list: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        加载所有方法的预测结果
        
        Args:
            dataset_name: 数据集名称
            variable_list: 变量名列表
        
        Returns:
            dict mapping method_name -> prediction matrix
        """
        results = {}
        
        for method in self.methods:
            pred_matrix = self.load_predictions_from_file(
                dataset_name, method, variable_list
            )
            if pred_matrix is not None:
                results[method] = pred_matrix.T
        
        return results

def load_baseline_reference_from_predict(
    dataset_name: str,
    variable_list: List[str],
    predict_dir: str = "predict",
    methods: List[str] = None,
    top_k: int = 10,
    threshold: float = 0.5
) -> Optional[str]:
    """
    从 predict 目录加载预先计算的基线参考信息
    
    Args:
        dataset_name: 数据集名称（如 'sachs', 'child'）
        variable_list: 变量名列表
        predict_dir: predict 目录路径
        methods: 要加载的方法列表（如 ['corr', 'invcov']）
        top_k: 显示top-k个关系
        threshold_percentile: 筛选阈值
    
    Returns:
        formatted reference text for LLM, 或 None（如果加载失败）
    """
    print(f"\n{'='*70}")
    print(f"LOADING BASELINE REFERENCE FROM PREDICT DIR")
    print(f"{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"Predict dir: {predict_dir}")
    print(f"Methods: {methods}")
    
    generator = BaselineReferenceGenerator(methods=methods, predict_dir=predict_dir)
    
    # 加载预测结果
    baseline_results = generator.load_all_predictions(dataset_name, variable_list)
    assert len(baseline_results) > 0, "No predictions loaded for dataset '{dataset_name}'"
    structured_graphs = create_structured_graphs_from_baseline_reference(baseline_results, variable_list, threshold)
    return structured_graphs

def create_structured_graphs_from_baseline_reference(
    baseline_results: Dict[str, np.ndarray],
    variable_list: List[str],
    threshold: float = 0.5
) -> Dict:
    """
    从基线参考结果创建结构化图
    """
    structured_graphs = {}
    for method_name, matrix in baseline_results.items():
        nodes = []
        for i in range(len(variable_list)):
            nodes.append({
                'name': variable_list[i],
                'parents': [variable_list[j] for j in range(len(variable_list)) if matrix[i, j] > threshold],
                # Convert numpy types to native Python types for JSON serialization
                'weight': {variable_list[j]: float(matrix[i, j]) for j in range(len(variable_list)) if matrix[i, j] > threshold}
                    
            })
        is_cyclic, cycle_path = has_cycle(nodes)
        if is_cyclic:
            nodes = break_cycles(nodes, cycle_path)
        structured_graphs[method_name] = {
            'nodes': nodes,
            'metadata': {
                "num_variables": len(variable_list),
                # Convert numpy int to Python int
                'num_edges': int(np.sum([len(node['parents']) for node in nodes])),
                'iteration': 0,
            },
            # Convert numpy array to list for JSON serialization
            'weighted_matrix': matrix.tolist(),
        }
    return structured_graphs

if __name__ == "__main__":
    # 测试示例：从 predict 目录加载
    dataset_name = "sachs"
    variable_list = [
        "Raf", "Mek", "Plcg", "PIP2", "PIP3",
        "Erk", "Akt", "PKA", "PKC", "P38", "Jnk"
    ]
    
    # 从 predict 目录加载参考信息
    reference_text = load_baseline_reference_from_predict(
        dataset_name=dataset_name,
        variable_list=variable_list,
        predict_dir="predict",
        methods=['corr', 'invcov'],
        top_k=10,
        threshold_percentile=90
    )
    
    if reference_text:
        print("\n" + "="*70)
        print("LOADED REFERENCE TEXT:")
        print("="*70)
        print(reference_text)
    else:
        print("\n⚠ Failed to load reference text")

