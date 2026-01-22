from schemas import StructuredGraph
from .search_strategy import SearchStrategy
from .mcts_strategy import MCTSStrategy
from .linear_strategy import LinearStrategy

class SearcherFactory:
    """搜索器工厂类"""
    
    @staticmethod
    def create_searcher(strategy_name: str, initial_graph: StructuredGraph) -> SearchStrategy:
        """根据策略名称创建搜索策略实例"""
        if strategy_name == "mcts":
            return MCTSStrategy(initial_graph)
        elif strategy_name == "linear":
            return LinearStrategy(initial_graph)
        else:
            raise ValueError(f"未知的搜索策略: {strategy_name}")