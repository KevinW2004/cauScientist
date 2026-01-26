from typing import List

from .search_strategy import SearchStrategy
from schemas import StructuredGraph


class LinearStrategy(SearchStrategy):
    """线性搜索策略: 只保留当前最优解"""
    
    def __init__(self, initial_graph: StructuredGraph):
        super().__init__(initial_graph)
        self.previous_graph: StructuredGraph = initial_graph
        self.history: List[StructuredGraph] = [initial_graph]

    def search(self) -> StructuredGraph:
        return self.previous_graph
    
    def update(self, graphs: List[StructuredGraph]):
        new_graph = None
        if len(graphs) == 0:
            pass
        elif len(graphs) == 1:
            if graphs[0].metadata.log_likelihood is None or self.previous_graph.metadata.log_likelihood is None:
                raise ValueError("Log likelihood is required for linear strategy update.")
            if graphs[0].metadata.log_likelihood > self.previous_graph.metadata.log_likelihood:
                new_graph = graphs[0]
        else:
            # 选择 log_likelihood 最高的图
            new_graph = max(graphs + [self.previous_graph], key=lambda g: g.metadata.log_likelihood or float('-inf'))
        if new_graph is not None and new_graph != self.previous_graph:
            self.previous_graph = new_graph
            self.history.append(self.previous_graph)

    def best_graph(self) -> StructuredGraph:
        return self.previous_graph
    
    def mark_as_final(self):
        self.previous_graph.metadata.is_final_graph = True
    
    def save_to_file(self, file_path: str):
        """将搜索器状态保存到文件"""
        import json
        with open(file_path, 'w') as f:
            json.dump([graph.model_dump() for graph in self.history], f, indent=4)