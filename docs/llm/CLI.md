# CLI

入口：`uv run llm-chat`

## 选项

| 选项 | 说明 |
|------|------|
| `--model / -m` | 模型 id |
| `--env-file` | `.env` 路径 |
| `--list-models` | 列出模型 |
| `--no-stream` | 非流式 |
| `--show-reasoning` | 展示 reasoning |
| `--system` | system prompt |

## REPL 命令

| 命令 | 说明 |
|------|------|
| `/exit` `/quit` | 退出 |
| `/clear` | 清空上下文（保留 system） |
| `/model <id>` | 切换模型并清空非 system 历史 |
