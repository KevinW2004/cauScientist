# 未来的 MCTS 策略
import math

from utils.metrics import compute_metrics
from schemas import StructuredGraph
from .search_strategy import SearchStrategy



class MCTSNode: # TODO:
    """MCTS 树节点"""
    
    def __init__(self, graph: StructuredGraph, parent: 'MCTSNode | None' = None, 
                 ll: float = float('-inf'), iteration: int = 0):
        self.graph = graph  # 当前图结构
        self.parent = parent  # 父节点
        self.children = []  # 子节点列表
        self.visits = 0  # 访问次数
        self.value = 0.0  # 累计奖励
        self.ll = ll  # 当前图的log-likelihood
        self.iteration = iteration  # 对应的迭代次数
        self.is_fully_expanded = False  # 是否已完全扩展
    
    def is_leaf(self) -> bool:
        """判断是否为叶节点"""
        return len(self.children) == 0
    
    def best_child(self, exploration_weight: float = 1.414) -> 'MCTSNode | None':
        """使用UCB1选择最佳子节点"""
        best_score = float('-inf')
        best_child = None
        
        for child in self.children:
            if child.visits == 0:
                # 优先选择未访问的节点
                return child
            
            # UCB1公式: exploitation + exploration
            exploitation = child.value / child.visits
            exploration = exploration_weight * math.sqrt(
                math.log(self.visits) / child.visits
            )
            ucb_score = exploitation + exploration
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
        
        return best_child
    
    def most_visited_child(self) -> 'MCTSNode | None':
        """返回访问次数最多的子节点（最终选择）"""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.visits)


class MCTSStrategy(SearchStrategy):
    def __init__(self, initial_graph: StructuredGraph):
        super().__init__(initial_graph)
        self.root = MCTSNode(graph=initial_graph, ll=initial_graph.metadata.log_likelihood or float('-inf'), iteration=initial_graph.metadata.iteration)

    def search(self) -> StructuredGraph: # TODO
        return self.root.graph
    
    def update(self, graphs: list[StructuredGraph]): # TODO
        pass

    def best_graph(self) -> StructuredGraph:
        best_node = self.root.most_visited_child()
        if best_node:
            return best_node.graph
        return self.root.graph
    
    def save_to_file(self, file_path: str):
        """将搜索器状态保存到文件"""
        pass

    def mark_as_final(self):
        pass