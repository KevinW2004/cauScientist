# CauScientist - 具有记忆的科学因果图推断 Agent

> 重构自 https://github.com/pengbo807/CDLLM
目前还是重构阶段，二阶段计划加入蒙特卡洛搜索树和 RAG 向量数据库记忆，后续三阶段加入基于重放的（replay-based）持续学习机制。 

## 安装
```bash
[uv] pip install -r requirements.txt
```

## 代码说明

1. data_loader 中实现数据库的读取和数据类型规定
2. llm_loader 中通过策略模式实现不同 LLM 后端的使用，统一调抽象基类 llm_loader 即可
3. utils 中存放各类指标的计算函数，以及配置管理对象 ConfigManager （它采用单例模式）。
4. ConfigManager 读取的配置文件在 `/config/default.toml`
5. `/cauScientist/main.py` 为程序==主入口==
6. 单次运行最主要的==主流程==在 `/cauScientist/scripts/cma_pipline.py`
