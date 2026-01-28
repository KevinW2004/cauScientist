from abc import ABC, abstractmethod

from schemas import StructuredGraph


class SearchStrategy(ABC):
    """搜索策略基类"""
    def __init__(self, initial_graph: StructuredGraph):
        pass

    @abstractmethod
    def search(self) -> StructuredGraph:
        """
        执行搜索
        Args:
        Returns:
            搜索到的下一个需要修改图
        """
        pass

    @abstractmethod
    def update(self, graphs: list[StructuredGraph]):
        """
        根据新传入的图结构更新搜索器状态
        Args:
            graphs: 新的候选图结构列表
        Returns:
        """
        pass

    @abstractmethod
    def best_graph(self) -> StructuredGraph:
        """
        返回当前搜索器认为的最佳图结构
        Returns:
            最佳图结构
        """
        pass

    @abstractmethod
    def mark_as_final(self):
        """
        标记上一次 search() 返回的图为不需要再修改的图
        Returns:
        """
        pass

    @abstractmethod
    def save_to_file(self, file_path: str):
        """
        将搜索器状态保存到文件
        Args:
            file_path: 保存路径
        Returns:
        """
        pass
