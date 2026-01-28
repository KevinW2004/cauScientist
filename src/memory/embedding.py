from fastembed import TextEmbedding
import numpy as np

from utils import SingletonMeta, ConfigManager

class EmbeddingModel(metaclass = SingletonMeta): 
    def __init__(self):
        self.config = ConfigManager()
        model_name: str = self.config.get("embedding.model_name")
        cache_dir: str = self.config.get("cache_dir")

        print(f"🚀 Loading FastEmbed model: {model_name}...")
        print(f"📂 Model cache dir: {cache_dir}")

        self.model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
