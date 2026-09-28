# Agent 指南

换机器或新会话时，先读本文件，再按需深入 `docs/common/`、`docs/llm/` 与 `docs/rag/`。

## 项目是什么

Python + uv：公共基建（配置、日志）+ 多协议 LLM 客户端 + 文档 RAG（FAISS、Embedding、Rerank、多轮会话）+ 公共 HTTP 层（`src/api`）与 CLI。

## 技术栈

- Python >= 3.11，uv
- httpx、FastAPI、uvicorn、typer、faiss-cpu、numpy、python-docx、pypdf
- 测试：pytest、respx

## 协议一览（本期）

| 能力 | PROTOCOL |
|------|----------|
| 对话 | `openai-completions`（缺省）、`openai-responses`、`anthropic-messages` |
| Embedding | `openai-embeddings`（缺省，唯一） |
| Rerank | `cohere-rerank`（缺省）、`dashscope-text-rerank`（阿里云百炼） |

细节见 `docs/common/CONFIG.md`。

## 目录地图

```text
src/common/     config / logger
src/api/        公共 HTTP：工厂、trace、统一错误、挂载路由
src/llm/        对话客户端与 llm-chat
src/rag/        解析、切分、embedding、rerank、rewrite、FAISS、retrieve、context、session、query、cli
docs/
tests/
.env.example
```

## 必读顺序

1. `docs/common/CONFIG.md`
2. `docs/common/LOGGING.md`
3. `docs/rag/REQUIREMENTS.md`
4. `docs/rag/ARCHITECTURE.md`
5. `docs/rag/API.md` / `docs/rag/CLI.md`
6. 改对话协议时读 `docs/llm/`

## 扩展

- 新配置域：`src/common/config/`
- 新 HTTP 业务：域内 router + `src/api` `include_router`
- 新 embedding/rerank 协议：Adapter + registry
- 新向量后端：实现 `VectorStore` 并注册（本期仅 faiss）
- 新文件类型：parser registry

## 常用命令

```bash
uv sync
uv run pytest
uv run rag serve
uv run rag upload ./notes.md
uv run rag chat
```

## Git

- 远程：`git@github.com:header-files/rag.git`
- 主分支：`main`
- Conventional Commits
