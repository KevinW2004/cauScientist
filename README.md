# CauScientist - 具有记忆的科学因果图推断 Agent

> 重构自 https://github.com/pengbo807/CDLLM
>
> 一阶段：重构代码，使之结构工程化
> 二阶段：加入蒙特卡洛搜索树和 RAG 向量数据库记忆
> 三阶段：加入基于重放的（replay-based）持续学习机制。

## 安装

```bash
[uv] pip install -r requirements.txt
```

## 代码说明

1. data_loader: 实现数据库的读取和数据类型规定
2. llm_loader: 通过策略模式实现不同 LLM 后端的使用，统一调抽象基类 llm_loader 即可
3. utils: 存放各类指标的计算函数，以及配置管理对象 ConfigManager （它采用单例模式）。
4. ConfigManager 读取的配置文件在 `/config/default.toml`
5. `/src/main.py` 为程序 **主入口**
6. 单次运行最主要的 **主流程** 在 `/src/scripts/cma_pipline.py`
7. schemas: 记录各类关键数据类型，目前有因果图相关的数据结构
