from typing import List

from .search_strategy import SearchStrategy
from schemas import StructuredGraph


class LinearStrategy(SearchStrategy):
    """线性搜索策略实现"""
    
    def __init__(self, initial_graph: StructuredGraph):
        super().__init__(initial_graph)
        self.previous_graph: StructuredGraph = initial_graph
        self.history: List[StructuredGraph] = [initial_graph]

    def search(self) -> StructuredGraph:
        return self.previous_graph
    
    def update(self, graphs: List[StructuredGraph]):
        if len(graphs) == 0:
            return
        elif len(graphs) == 1:
            self.previous_graph = graphs[0]
        else:
            # 选择 log_likelihood 最高的图
            best_graph = max(graphs, key=lambda g: g.metadata.log_likelihood or float('-inf'))
            self.previous_graph = best_graph
        self.history.append(self.previous_graph)

    def best_graph(self) -> StructuredGraph:
        return self.previous_graph
    
    def save_to_file(self, file_path: str):
        """将搜索器状态保存到文件"""
        import json
        with open(file_path, 'w') as f:
            json.dump([graph.model_dump_json() for graph in self.history], f, indent=4)