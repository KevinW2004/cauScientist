from typing import Dict, Type
from llm_loader import LLMLoader
from .local_llm_loader import LocalLLMLoader
from .openai_llm_loader import OpenAILLMLoader

class LLMLoaderFactory:
    """LLM加载器工厂类"""
    
    # 静态字典映射LLM类型到对应的加载器类
    llm_loader_classes: Dict[str, Type[LLMLoader]] = {
            "openai": OpenAILLMLoader,
            "local": LocalLLMLoader,
        }

    @staticmethod
    def create_llm_loader(llm_type: str) -> LLMLoader:
        """根据LLM类型创建相应的LLM加载器实例
        
        Args:
            llm_type (str): LLM的类型标识符
            
        Returns:
            LLMLoader: 对应类型的LLM加载器实例
            
        Raises:
            ValueError: 如果提供的LLM类型不受支持
            ('local', 'openai')
        """
        
        
        if llm_type not in LLMLoaderFactory.llm_loader_classes:
            raise ValueError(f"Unsupported LLM type: {llm_type}")
        
        llm_loader_class = LLMLoaderFactory.llm_loader_classes[llm_type]
        return llm_loader_class()