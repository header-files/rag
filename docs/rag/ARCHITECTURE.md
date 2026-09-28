# 架构

## 核心链路

```text
Ingest: Parser → Chunker → Embedding → FaissVectorStore
Query:  [可选 QueryRewrite] → Embedding → Retriever → ContextBuilder → LLMClient
```

## 协议（本期已实现）

| 能力 | 支持的 PROTOCOL | 缺省 |
|------|-----------------|------|
| 对话 | `openai-completions`、`openai-responses`、`anthropic-messages` | `openai-completions` |
| Embedding | `openai-embeddings` | `openai-embeddings` |
| Rerank | `cohere-rerank`、`dashscope-text-rerank` | `cohere-rerank` |

配置与字段约定见 [docs/common/CONFIG.md](../common/CONFIG.md)。新增协议：在对应包增加 Adapter 并注册到 registry，同时把协议名加入 `common.config` 的合法集合。

| 模块 | 负责 | 不负责 |
|------|------|--------|
| VectorStore | 存储/搜索/软删/rebuild | Prompt |
| QueryRewriter | 多轮问题 → 独立检索查询（可关） | 向量检索 |
| Retriever | Query → Hits（含可选 Rerank） | Session |
| ContextBuilder | 三分开组装 Messages | LLM |
| SessionStore | History；单进程锁安全，不同 session 请求不强制串行 | 检索 |
| QueryService | 编排 | 检索算法细节 |
| `src/api` | HTTP 工厂与中间件 | 业务算法 |

## Query 改写

- 配置：`RAG_QUERY_REWRITE_ENABLED`（默认 `false`）
- 开启且存在会话历史时，先调 LLM 改写，再用改写结果做 Embedding/检索
- 无历史、改写失败或空输出：使用用户原问题
- Context / Session 仍使用用户原文；改写仅影响检索 query

## FAISS

- `IndexFlatIP` + L2 归一化
- 无 `document_ids`：`index.search`，软删空洞用未删除向量补齐
- 有 `document_ids`：允许 faiss_id 上矩阵内积 Top-K（侧车缓存向量）
- 内部：`chunks_by_faiss_id` / `faiss_ids_by_chunk_id` / `faiss_ids_by_document_id`；`ChunkRecord` 含 `faiss_id` 与 `deleted`
- `rebuild()` 第一期不自动触发

## Context

逻辑上 system / history / query / context；协议层映射为现有 Message。History 追加仅为问题原文。`RAG_MAX_CONTEXT_CHARS` 为字符近似上限。

## 公共 API

`src/api.create_app`：trace、统一错误、`/health`、挂载 `rag` router。
