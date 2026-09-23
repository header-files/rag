# Session 策略

`ChatSession` 维护内存中的 `messages`。

## 行为

- 发送前追加 user
- **仅成功**后追加 assistant（`content`）
- 流式：收到 `finish` 后追加；中途失败则回滚本次 user，且不追加 assistant
- `clear()`：清除非 system 消息
- `switch_client(..., clear_history=True)`（默认）：切换模型时清空非 system 历史，避免跨协议上下文问题

切模型策略属于 Session，不属于统一客户端协议。
