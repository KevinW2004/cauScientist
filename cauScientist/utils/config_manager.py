"""
从 /config 文件夹中读取 toml 配置，管理项目
单例模式
"""
import toml
import os
from typing import Any

class ConfigManager:
    # 单例实例
    _instance = None
    _initialized = False
    
    def __new__(cls, config_path: str = None):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: str = None):
        """
        初始化配置管理器
        Args:
            config_path: TOML 配置文件路径（如果为None，使用默认路径）
        """
        # 防止重复初始化
        if ConfigManager._initialized:
            return
        
        if config_path is None:
            # 使用默认路径
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "config",
                "default.toml"
            )
        
        self.config_path = config_path
        
        # 检查文件是否存在
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # 加载TOML
        print(f"Loading configuration from: {config_path}")
        self.config = toml.load(config_path)
        print("✓ Configuration loaded successfully")
        
        ConfigManager._initialized = True
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值，支持嵌套访问
        
        Args:
            key_path: 配置键路径，用"."分隔（例如 "llm.type"）
            default: 如果键不存在时的默认值
        
        Returns:
            配置值或默认值
        """
        keys = key_path.split(".")
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value

# test 
if __name__ == "__main__":
    config_manager = ConfigManager()
    llm_type = config_manager.get("llm.type", "default_type")
    print(f"LLM Type: {llm_type}")
    # 输出整个配置
    print(config_manager.config)