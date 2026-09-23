# 配置

`common.config`（`src/common/config/`）负责从 `.env` 加载配置，并按域组织配置项。

## 模块

| 模块 | 职责 |
|------|------|
| `common.config.env` | 通用 `.env` 加载（`load_env`） |
| `common.config.errors` | `ConfigError` |
| `common.config.models` | 模型连接配置（`ModelConfig` / `load_models` / `get_model`） |
| `common.config.log_settings` | 日志开关与轮转参数（`LogSettings` / `load_log_settings`） |

新增配置域时，在本包增加模块即可；通用文件读取复用 `load_env`。

## 模型配置约定

使用单个扁平 `.env`（模板：`.env.example`）。

```env
DEFAULT_MODEL=chatgpt

MODEL_<ID>_PROTOCOL=openai-completions
MODEL_<ID>_NAME=gpt-4o-mini
MODEL_<ID>_BASE_URL=https://api.openai.com/v1
MODEL_<ID>_API_KEY=sk-xxx
MODEL_<ID>_TIMEOUT_SECONDS=60
```

日志相关 `LOG_*` 见 [LOGGING.md](LOGGING.md)。

- 模型 id：`MODEL_<ID>_` 中的 `<ID>`（大小写不敏感，内部小写）
- 必填：`NAME`、`BASE_URL`、`API_KEY`；`PROTOCOL` 缺省为 `openai-completions`
- 真实 `.env` 含密钥，不提交仓库（见 `.gitignore`）

## 加载 API

```python
from common.config import load_env, load_models, get_model, load_log_settings, ConfigError
```
