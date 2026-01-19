from pydantic import BaseModel, Field, ConfigDict, model_validator, field_serializer
from typing import List, Tuple, Union
import numpy as np
import json


class CausalNode(BaseModel):
    """
    因果图的原始节点表示
    """
    name: str
    parents: List[str] = Field(default_factory=list)

class GraphChanges(BaseModel):
    """
    记录相对于上一轮迭代的图结构变化
    来源: llm_hypothesis.py -> _compute_changes
    """
    # 注意：源码中是用 tuple (parent, child) 存边的，JSON化时通常会变成 list
    added_edges: List[Tuple[str, str]] = Field(default_factory=list, description="新增的边")
    removed_edges: List[Tuple[str, str]] = Field(default_factory=list, description="移除的边")
    num_added: int = 0
    num_removed: int = 0

class GraphMetadata(BaseModel):
    """
    因果图的元数据，生成信息和评分等等
    """
    # --- 基础信息 (生成时写入) ---
    domain: str = Field(..., description="领域名称")
    iteration: int = Field(..., description="当前迭代轮次 (t)")
    num_variables: int
    num_edges: int
    reasoning: str = Field(..., description="LLM 生成该结构的推理文本")
    
    # --- 变化记录 (生成时写入，第一轮可能为 None) ---
    changes: GraphChanges | None = None

    # --- 评分指标 (Score Functions 写入) ---
    log_likelihood: float | None = Field(None, description="BIC Score (CV Log Likelihood)")
    bic: float | None = Field(None, description="Traditional BIC value")
    num_parameters: int | None = None
    method: str | None = Field(None, description="评分方法，如 'Linear_Gaussian_BIC'")

class StructuredGraph(BaseModel):
    """
    完整的因果图结构表示
    """
    model_config = ConfigDict(arbitrary_types_allowed=True) # 允许 numpy 类型

    metadata: GraphMetadata
    nodes: List[CausalNode]
    adjacency_matrix: Union[np.ndarray, List[List[int]]] = Field(..., description="邻接矩阵表示")

    @field_serializer('adjacency_matrix')
    def serialize_adjacency_matrix(self, adjacency_matrix: Union[np.ndarray, List[List[int]]], _info):
        """将 numpy 数组序列化为列表"""
        if isinstance(adjacency_matrix, np.ndarray):
            return adjacency_matrix.tolist()
        return adjacency_matrix

    def get_node_map(self) -> dict[str, CausalNode]:
        """获取节点名称到节点对象的映射"""
        return {node.name: node for node in self.nodes}