# CLI

`uv run rag`

| 命令 | 说明 |
|------|------|
| `serve` | 启动服务（FAISS/Session 在该进程） |
| `upload` / `ls` / `rm` | 文档 |
| `ask` | 单次提问（可 `--session-id`） |
| `chat` | 交互多轮；启动时创建 session |

交互：`/upload` `/ls` `/rm` `/clear`（删旧 session 并新建）`/exit`。  
`--base-url` 默认 `API_BASE_URL`。
