# llm

基于 httpx 自实现的多协议大模型 HTTP 客户端，附带可多轮对话的 CLI；公共配置与日志由 `common` 提供。

## 特性

- 扁平 `.env` 配置（`common.config`）
- 可配置日志：控制台 / 按大小轮转文件（`common.logger`）
- 协议：`openai-completions`（默认）、`openai-responses`、`anthropic-messages`
- 同步 / 异步 × 流式 / 非流式
- 统一客户端抽象（`Message` / `ChatRequest` / `ChatResponse` / `StreamChunk` / `Error`）
- Reasoning 文本事件流（响应侧）
- 不依赖 openai / anthropic 官方 SDK

## 快速开始

```bash
# 安装（需已安装 uv）
uv sync

# 配置
cp .env.example .env
# 编辑 .env，填入 API Key、日志开关等

# 交互对话
uv run llm-chat
uv run llm-chat --model chatgpt --show-reasoning
uv run llm-chat --list-models
uv run llm-chat --no-stream
```

## 配置与日志

- 配置：[docs/common/CONFIG.md](docs/common/CONFIG.md)、[.env.example](.env.example)
- 日志：[docs/common/LOGGING.md](docs/common/LOGGING.md)

真实 `.env` 含密钥，已加入 `.gitignore`，请勿提交。

## 作为库使用

```python
from common.logger import setup_logging
from llm import LLMClient, ChatRequest, Message

setup_logging()
client = LLMClient.from_env("chatgpt")
resp = client.chat(
    ChatRequest(messages=[Message(role="user", content="你好")])
)
print(resp.content)
```

## 开发

```bash
uv sync
uv run pytest
```

## 文档

| 文档 | 说明 |
|------|------|
| [docs/AGENT.md](docs/AGENT.md) | 给 agent 的项目地图 |
| [docs/common/](docs/common/) | 配置、日志等公共基建 |
| [docs/llm/](docs/llm/) | 大模型调用：需求、架构、协议、Session、CLI 等 |
