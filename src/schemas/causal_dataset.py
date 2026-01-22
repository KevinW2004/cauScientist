import numpy as np
from typing import List, Tuple, Optional

class CausalDataset:
    """因果发现数据集 (纯观测版本)"""
    
    def __init__(
        self,
        data: np.ndarray,
        ground_truth_graph: np.ndarray,
        variable_names: Optional[List[str]] = None,
        domain_name: str = "unknown",
        variable_type: str = "continuous"
    ):
        """
        Args:
            data: 观测数据矩阵 [n_samples, n_variables]
            ground_truth_graph: 真实的因果图邻接矩阵 [n_variables, n_variables]
            variable_names: 变量名列表 (LLM推理的关键)
            domain_name: 领域名称 (用于Prompt构建)
            variable_type: 变量类型 ("continuous" 或 "discrete")
        """
        self.data = data
        self.ground_truth_graph = ground_truth_graph
        self.domain_name = domain_name
        self.variable_type = variable_type
        
        self.n_samples, self.n_variables = data.shape
        
        # 变量名处理
        if variable_names is None:
            self.variable_names = [f"X{i}" for i in range(self.n_variables)]
        else:
            self.variable_names = variable_names
        
        # 简单校验
        assert len(self.variable_names) == self.n_variables, \
            f"变量名数量 ({len(self.variable_names)}) 与数据维度 ({self.n_variables}) 不匹配"
        assert self.ground_truth_graph.shape == (self.n_variables, self.n_variables), \
            "Ground Truth 矩阵维度错误"

    def get_data(self) -> np.ndarray:
        """获取所有数据 (默认为观测数据)"""
        return self.data
    
    def get_ground_truth_edges(self) -> List[Tuple[str, str]]:
        """获取真实的因果边 (用于评估)"""
        edges = []
        for i in range(self.n_variables):
            for j in range(self.n_variables):
                if self.ground_truth_graph[i, j] == 1:
                    edges.append((self.variable_names[i], self.variable_names[j]))
        return edges
    
    def print_summary(self):
        """打印数据集摘要"""
        print("\n" + "="*70)
        print(f"DATASET SUMMARY: {self.domain_name}")
        print("="*70)
        print(f"Variables ({self.n_variables}):")
        print(f"  {', '.join(self.variable_names)}")
        print(f"\nData Info:")
        print(f"  Shape: {self.data.shape}")
        print(f"  Type:  {self.variable_type}")
        print(f"  Range: [{self.data.min():.2f}, {self.data.max():.2f}]")
        
        print(f"\nGround Truth Graph:")
        print(f"  Total edges: {self.ground_truth_graph.sum()}")
        print("="*70 + "\n")