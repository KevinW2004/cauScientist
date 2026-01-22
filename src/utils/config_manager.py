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
    _path_prefix = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "config"
    )
    
    def __new__(cls, config_path: str = None):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: str = None):
        """
        初始化配置管理器
        支持配置分层：先加载 配置.toml，再加载 secret.toml（如果存在）进行覆盖
        
        Args:
            config_path: TOML 配置文件路径（如果为None，使用默认路径）
        """
        # 防止重复初始化
        if ConfigManager._initialized:
            return
        
        if config_path is None: config_path = "default.toml"
        config_path = os.path.join(
            ConfigManager._path_prefix, config_path
        )
        # 检查文件是否存在
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # 加载默认配置
        print(f"Loading configuration from: {config_path}")
        self.config = toml.load(config_path)
        print("✓ Configuration loaded successfully")
        
        # 尝试加载 secret.toml（如果存在）
        secret_path = os.path.join(ConfigManager._path_prefix, "secret.toml")
        if os.path.exists(secret_path):
            print(f"Loading secret configuration from: {secret_path}")
            secret_config = toml.load(secret_path)
            self._merge_config(self.config, secret_config)
            print("✓ Secret configuration merged successfully")
        else:
            print("⚠ No secret.toml found. For production and any sensitive credentials "
                  "(such as real API keys), create config/secret.toml to override "
                  "placeholders in default.toml.")
        ConfigManager._initialized = True
    
    def _merge_config(self, base: dict, override: dict) -> None:
        """
        递归合并配置字典，override 中的值会覆盖 base 中的值
        
        Args:
            base: 基础配置字典（会被修改）
            override: 覆盖配置字典
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                # 递归合并嵌套字典
                self._merge_config(base[key], value)
            else:
                # 直接覆盖
                base[key] = value
    
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
    
    def set(self, key_path: str, value: Any) -> None:
        """
        设置配置值，支持嵌套访问（不支持新增键）
        不会修改原始配置文件，只修改内存中的配置
        
        Args:
            key_path: 配置键路径，用"."分隔（例如 "llm.type"）
            value: 要设置的值
        """
        keys = key_path.split(".")
        cfg = self.config

        for key in keys[:-1]:
            if isinstance(cfg, dict) and key in cfg:
                cfg = cfg[key]
            else:
                raise KeyError(f"Parent path not found: {key_path}")
        
        last_key = keys[-1]
        if isinstance(cfg, dict) and last_key in cfg:
            cfg[last_key] = value
        else:
            raise KeyError(f"Key not found: {key_path}")
