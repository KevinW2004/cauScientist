from fastembed import TextEmbedding
import numpy as np

from utils import SingletonMeta, ConfigManager

class EmbeddingModel(metaclass = SingletonMeta): 
    def __init__(self):
        self.config = ConfigManager()
        model_name: str = self.config.get("rag.embedding.model_name")
        cache_dir: str = self.config.get("cache_dir")

        print(f"🚀 Loading FastEmbed model: {model_name}...")
        print(f"📂 Model cache dir: {cache_dir}")

        self.model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def encode(self, texts: list[str]) -> np.ndarray:
        embeddings_generator = self.model.embed(texts)
        return np.array([embedding for embedding in embeddings_generator])

    @property
    def dimension(self) -> int:
        return self.model.embedding_size