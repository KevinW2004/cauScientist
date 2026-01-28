import time

from utils import SingletonMeta
from .vector_db import VectorDB
from .embedding import EmbeddingModel


class MemoryService(metaclass=SingletonMeta):
    """
    智能体长期记忆管理中枢服务-单例模式
    """

    def __init__(self):
        self.vector_db = VectorDB()
        self.embedding = EmbeddingModel()

    # ==== 对外接口 ====
    def save_memory(self, text: str, metadata: dict = {}):
        """保存记忆文本到向量数据库

        :param text: 需要保存的记忆文本
        :param metadata: 关联的元数据字典
        """
        # 1. 切分文本
        text_chunks = self._chunk_text(text)
        vectors = self.embedding.encode(text_chunks)

        # 2. 批量生成 payloads
        payloads = []
        for i, chunk in enumerate(text_chunks):
            payload = {
                "text": chunk,
                "chunk_index": i,
                "original_full_text": text[:30] + "...",
                "timestamp": time.time(),
                **metadata,
            }
            payloads.append(payload)

        self.vector_db.insert_vectors(vectors=vectors.tolist(), payloads=payloads)
        print(f"🧠 Saved {len(text_chunks)} chunks to memory.")

    def retrieve_memories(self, query_text: str, limit: int = 10) -> list[dict]:
        """根据查询文本检索相关记忆

        :param query_text: 查询文本
        :param limit: 最大允许返回的相关记忆数量
        :return: 相关记忆列表，每条记忆为包含文本和元数据的字典
        """
        # 生成查询嵌入, 就一列
        query_vector = self.embedding.encode([query_text])[0]
        # 搜索相关向量
        points = self.vector_db.search_vectors(query_vector=query_vector, limit=limit)
        # 动态调整返回数量
        scores = [point.score for point in points]
        dynamic_limit = self._find_dynamic_limit(scores)
        points = points[:dynamic_limit]
        # 提取格式化结果给 Agent
        results = []
        for point in points:
            payload = point.payload or {}
            result = {
                "text": payload.get("text"),
                "metadata": payload,
                "score": point.score
            }
            results.append(result)
        return results

    # ==== 辅助函数 ====
    def _chunk_text(
        self, text: str, chunk_size: int = 500, overlap: int = 50
    ) -> list[str]:
        """将文本切分为多个块，便于嵌入和存储"""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunks.append(text[start:end])
            start += chunk_size - overlap

        return chunks

    def _find_dynamic_limit(self, scores: list[float]) -> int:
        """
        根据相似度分布找到拐点，动态调整返回的结果数量。
        """
        if len(scores) < 2: return len(scores)
        # 先直接截掉相似度过低的结果
        score_bar = 0.4
        _scores = [s for s in scores if s >= score_bar]
        if len(_scores) < 2: _scores = scores[:2] # 保底
        # 计算相邻分数的相对变化率
        relative_changes = [
            (_scores[i] - _scores[i + 1]) / _scores[i] if abs(_scores[i]) >= 1e-6 else 0
            for i in range(len(_scores) - 1)
        ]
        # 找到变化率大于某个阈值的第一个位置
        threshold = 0.15
        for i, change in enumerate(relative_changes):
            if change > threshold:
                return i + 1
        # 如果没有找到明显的变化点，返回默认值
        return len(_scores)
