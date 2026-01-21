from typing import Dict

from .search_strategy import SearchStrategy
from schemas import StructuredGraph


class LinearStrategy(SearchStrategy):
    """线性搜索策略实现"""
    
    def __init__(self):
        super().__init__()
        self.previous_graph: StructuredGraph | None = None

    def search(self, **kwargs) -> Dict:
        """执行线性搜索，返回结果字典"""
        # 线性搜索的具体实现逻辑
        result = {
            "message": "Linear search executed",
            "previous_graph": self.previous_graph
        }
        return result



    