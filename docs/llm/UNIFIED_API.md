# 客户端交互协议（统一抽象）

调用方只认本节类型。

## Message

```text
role: "system" | "user" | "assistant"
content: string   # 本期纯文本
```

## ChatRequest

```text
messages: Message[]
temperature: float | null
max_output_tokens: int | null
stop: string[] | null
```

- `null` 字段不发送
- 非 `null` 经 Adapter 映射后透传；上游拒绝则返回统一 Error

## ChatResponse

```text
content: string
reasoning: string | null
finish_reason: FinishReason | null
model: string | null
usage: Usage | null
```

`FinishReason = "stop" | "length" | "content_filter" | null`  
不含厂商 `raw`。

## StreamChunk

```text
type: "reasoning" | "content" | "finish"
delta: string
finish_reason: FinishReason | null
model: string | null
usage: Usage | null
```

- 写入上下文仅用 `content` 拼接结果
- 流失败：已展示 chunk 可保留；不追加 assistant；不发 `finish`

## Error

见 `ERRORS.md`。

## Client API

```text
chat / chat_stream / achat / achat_stream
LLMClient.from_env(model_id?)
```
