# rag

基于已有多协议 LLM 客户端的文档 RAG：FAISS 检索、默认 Rerank、多轮会话；公共 HTTP 层在 `src/api`，CLI 经 HTTP 操作。

## 协议

| 能力 | 支持的 PROTOCOL |
|------|-----------------|
| 对话 | `openai-completions`（缺省）、`openai-responses`、`anthropic-messages` |
| Embedding | `openai-embeddings`（缺省，本期唯一） |
| Rerank | `cohere-rerank`（缺省）、`dashscope-text-rerank`（阿里云百炼） |

完整配置说明见 [docs/common/CONFIG.md](docs/common/CONFIG.md)。

## 快速开始

```bash
uv sync
cp .env.example .env
# 填写对话 / embedding / rerank 密钥

uv run rag serve
```

另一终端：

```bash
uv run rag upload ./notes.md
uv run rag chat
```

FAISS 与 Session 随 `serve` 进程；重启需重新上传。

## 文档

| 文档 | 说明 |
|------|------|
| [docs/AGENT.md](docs/AGENT.md) | Agent 地图 |
| [docs/common/](docs/common/) | 配置与日志 |
| [docs/llm/](docs/llm/) | 对话客户端 |
| [docs/rag/](docs/rag/) | RAG 需求、架构、API、CLI |

```bash
uv run pytest
```
