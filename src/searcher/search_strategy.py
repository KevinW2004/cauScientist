"""
搜索策略模块 - 封装不同的图搜索方法
包含 Hill Climbing 和 MCTS
"""

from abc import ABC, abstractmethod
from typing import Dict


class SearchStrategy(ABC):
    """搜索策略基类"""
    
    def __init__(self):
        pass
    
    @abstractmethod
    def search(self, **kwargs) -> Dict:
        """执行搜索，返回结果字典"""
        pass

