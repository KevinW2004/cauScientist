from pydantic import BaseModel, Field, ConfigDict, field_serializer
from typing import Literal
import numpy as np

class CausalNode(BaseModel):
    """
    因果图的原始节点表示
    """
    name: str
    parents: list[str] = Field(default_factory=list)

class GraphChange(BaseModel):
    """
    记录单次图结构变化（单条边的操作）
    """
    type: Literal["ADD", "DELETE", "REVERSE"] = Field(..., description="操作类型")
    parent: str = Field(..., description="父节点")
    child: str = Field(..., description="子节点")
    reasoning: str = Field(default="", description="该操作的推理依据")

class GraphMetadata(BaseModel):
    """
    因果图的元数据，生成信息和评分等等
    """
    # --- 基础信息 (生成时写入) ---
    domain: str = Field(..., description="领域名称")
    iteration: int = Field(..., description="当前迭代轮次 (t)")
    num_variables: int
    num_edges: int
    is_final_graph: bool = Field(default=False, description="LLM 是否认为这是一张不再需要修改的最终图")

    # --- 变化记录 ---
    change_history: list[GraphChange] = Field(
        default=[], description="从初始图到当前图的所有历史变化"
    )

    # --- 评分指标 (Score Functions 写入) ---
    log_likelihood: float | None = Field(default=None, description="BIC Score (CV Log Likelihood), 越大越好")
    bic: float | None = Field(default=None, description="Traditional BIC value, 越小越好")
    num_parameters: int | None = None

class StructuredGraph(BaseModel):
    """
    完整的因果图结构表示
    """
    model_config = ConfigDict(arbitrary_types_allowed=True) # 允许 numpy 类型

    metadata: GraphMetadata
    nodes: list[CausalNode]
    adjacency_matrix: np.ndarray | list[list[int]] = Field(..., description="邻接矩阵表示")

    @field_serializer('adjacency_matrix')
    def serialize_adjacency_matrix(self, adjacency_matrix: np.ndarray | list[list[int]], _info):
        """将 numpy 数组序列化为列表"""
        if isinstance(adjacency_matrix, np.ndarray):
            return adjacency_matrix.tolist()
        return adjacency_matrix

    def get_node_map(self) -> dict[str, CausalNode]:
        """获取节点名称到节点对象的映射"""
        return {node.name: node for node in self.nodes}
