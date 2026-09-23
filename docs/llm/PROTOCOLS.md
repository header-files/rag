# 协议说明

## openai-completions

- URL：`{base_url}/chat/completions`（base 已含完整路径则直接用）
- `max_output_tokens` → `max_tokens`
- 流式：SSE `data:`；`[DONE]` 结束
- reasoning：`message/delta.reasoning_content` 或 `reasoning`

## openai-responses

- URL：`{base_url}/responses`
- system → `instructions`；其余 → `input`
- `max_output_tokens` 原样发送
- 流式事件：`response.output_text.delta`、`response.reasoning_text.delta` 等

## anthropic-messages

- URL：`{base}/v1/messages`（base 以 `/v1` 结尾则只拼 `/messages`）
- Header：`x-api-key`、`anthropic-version`
- system 独立字段；`max_output_tokens` → `max_tokens`（缺省 4096）
- thinking 块 → reasoning 事件/字段

## 新增协议

实现 `ProtocolAdapter`，在 `registry.py` 注册，补充测试与本文档。
