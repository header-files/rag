# 需求说明

## 目标

在已有 LLM Client 上实现完整文档 RAG：解析、切分、FAISS、默认 Rerank、ContextBuilder、多轮 Session、公共 HTTP API 与 CLI。

## 功能需求

1. Embedding / Rerank / RAG / API 配置分离，约束在配置层校验
2. Embedding 协议：`openai-embeddings`（缺省且本期唯一）；Rerank 协议：`cohere-rerank`（缺省）、`dashscope-text-rerank`（阿里云百炼 text-rerank）
3. 向量库仅 FAISS（进程内、不落盘）；软删 + `rebuild()`；有/无 `document_ids` 双路径检索
4. 解析 txt / md / docx / pdf
5. Rerank 默认开启；`RERANK_TOP_N <= RETRIEVE_TOP_K`
6. Query 改写可选；`RAG_QUERY_REWRITE_ENABLED` 默认 `false`；开启后多轮场景先改写再检索
7. 多轮：每轮重检；History 只存用户原话与助手回复；Session 进程内存且并发读写安全
8. `src/api` 公共层挂载各域路由；本期无前端

## 非目标

- memory 向量实现、索引落盘、Redis/OpenSearch
- 精确 token budget、鉴权、多租户、旧 `.doc`
