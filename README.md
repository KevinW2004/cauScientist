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

## 配置说明

1. 所有的配置都需要放在 `./config` 下，须使用 toml 格式。
2. 不带参数运行 main 时默认使用 `default.toml`，若使用自己创建的 toml，需要在运行时设置命令行参数 `--config="配置文件名"`。
3. 调用网络平台 API 的 api_key 放在 secret.toml 中，此文件默认在 `.gitignore` 中无法被上传。

## 代码说明

1. data_loader: 实现数据库的读取和数据类型规定
2. llm_loader: 通过策略模式实现不同 LLM 后端的使用，统一调抽象基类 llm_loader 即可
3. utils: 存放各类指标的计算函数，以及配置管理对象 ConfigManager （它采用单例模式）。
4. ConfigManager 读取配置文件
5. `/src/main.py` 为程序 **主入口**
6. 单次运行最主要的 **主流程** 在 `/src/scripts/cma_pipline.py`
7. schemas: 记录各类关键数据类型，目前有因果图相关的数据结构
8. reflection：里面用单例模式实现了一个反思管理器，可以在每次评分后生成反思并合并记录（作为短期记忆，在同一任务内每次都全部输入到 prompt)
9. memory: 使用向量数据库（RAG技术）作为长期记忆，使用的技术栈为 qdrant + fastembed，qdrant 底层使用rust构建，性能非常好而且接口简洁现代化，fastembed是qdrant官方配套的 embedding 库，比 sentece transformer 快很多且占用低。
