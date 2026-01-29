from abc import ABC, abstractmethod
from typing import Any

class LLMLoader(ABC):
    """LLM加载器抽象基类"""
    
    @abstractmethod
    def load_model(self) -> Any:
        """
        加载模型
        Args:
            **kwargs: 模型加载所需的参数
        """
        pass
    
    @abstractmethod
    def generate(
        self, 
        system_prompt: str, 
        user_prompt: str,
        temperature: float = 0.7,
    ) -> str:
        """
        生成回答
        Args:
            system_prompt (str): 系统提示
            user_prompt (str): 用户提示
            temperature (float): 生成文本的温度参数，默认为0.7
        Returns:
            生成的文本
        """
        pass
    

    @abstractmethod
    def get_backend_type(self) -> str:
        """
        获取后端类型
        Returns:
            后端类型字符串
        """
        pass