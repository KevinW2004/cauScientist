from qdrant_client import QdrantClient
from qdrant_client.http import models
import os
import uuid
import atexit

from utils import SingletonMeta, ConfigManager
from .embedding import EmbeddingModel


class VectorDB(metaclass=SingletonMeta):
    def __init__(self):
        """初始化 Qdrant: 本地嵌入式模式"""
        self.config = ConfigManager()
        self.collection_name: str = self.config.get("rag.qdrant.collection_name")

        storage_path: str = self.config.get("rag.qdrant.path")
        os.makedirs(storage_path, exist_ok=True)

        print(f"💾 Connecting to Qdrant vector database at: {storage_path}...")
        self.client = QdrantClient(path=storage_path)
        self.vector_dimension = EmbeddingModel().dimension

        # init collection
        if not self.client.collection_exists(collection_name=self.collection_name):
            print(f"🆕 Creating Qdrant collection: {self.collection_name}...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_dimension, distance=models.Distance.COSINE
                ),
            )
        else:
            print(f"✅ Qdrant collection '{self.collection_name}' already exists.")
            
        print("✅ Qdrant vector database is ready. \n")
        # 解决运行结束退出的神秘报错
        atexit.register(lambda: self.client.close() if hasattr(self, "client") else None)

    def insert_vector(self, vector: list[float], payload: dict = {}):
        """插入一条向量数据"""
        self.client.upsert(
            collection_name=self.collection_name,
            points=[models.PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)]
        )

    def insert_vectors(self, vectors: list[list[float]], payloads: list[dict]):
        """批量插入向量数据 (High Performance)"""
        # 自动生成对应数量的 UUID
        ids = [str(uuid.uuid4()) for _ in range(len(vectors))]

        points = [
            models.PointStruct(id=id, vector=vector, payload=payload)
            for id, vector, payload in zip(ids, vectors, payloads)
        ]

        self.client.upsert(collection_name=self.collection_name, points=points)

    def search_vectors(self, query_vector: list[float], limit=10) -> list[models.ScoredPoint]:
        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit
        ).points
