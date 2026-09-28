# HTTP API

入口由 `src/api` 创建。默认 `http://127.0.0.1:8000`。可带 `X-Trace-Id`。

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 状态 |
| POST | `/documents` | multipart `file` |
| GET | `/documents` | 列表 |
| GET/DELETE | `/documents/{id}` | 详情 / 软删 |
| POST | `/sessions` | 创建会话 |
| DELETE | `/sessions/{id}` | 删除会话 |
| POST | `/chat` | 多轮问答 |

`/chat` 体：`message`、`session_id?`、`document_ids?`、`stream?`。  
响应含 `session_id`、`answer`、`citations`。流式事件：`session` / `citations` / `content` / `done` / `error`。

错误：`{"error":{"type","message"}}`。
