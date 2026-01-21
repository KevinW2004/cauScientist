from .search_strategy import SearchStrategy
from .mcts_strategy import MCTSStrategy
from .linear_strategy import LinearStrategy

class SearcherFactory:
    """搜索器工厂类 - 根据配置创建搜索策略实例"""
    
    @staticmethod
    def create_searcher(strategy_name: str) -> SearchStrategy:
        """根据策略名称创建搜索策略实例"""
        if strategy_name == "MCTS":
            return MCTSStrategy()
        elif strategy_name == "linear":
            return LinearStrategy()
        else:
            raise ValueError(f"未知的搜索策略: {strategy_name}")