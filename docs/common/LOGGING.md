# 日志

`common.logger`（`src/common/logger/`）提供统一的日志初始化、上下文（含 `trace_id`）与获取入口。

## 结构

```text
src/common/config/log_settings.py   # LOG_* 配置解析（LogSettings）
src/common/logger/
  context.py                        # bind / traced / log_context（可扩展字段）
  filters.py                        # ContextFilter：注入 LogRecord
  setup.py                          # setup_logging / get_logger
```

## 行为

| 能力 | 说明 |
|------|------|
| 控制台 | `StreamHandler`（stderr），由 `LOG_TO_CONSOLE` 开关 |
| 文件 | `RotatingFileHandler`，由 `LOG_TO_FILE` 开关 |
| 大小限制 | `LOG_MAX_BYTES`；超出后轮转生成 `app.log.1`、`app.log.2`… |
| 保留份数 | `LOG_BACKUP_COUNT` |
| 上下文 | `contextvars` 隔离；默认注入 `trace_id` |

## 配置（`.env`）

```env
LOG_LEVEL=INFO
LOG_TO_CONSOLE=true
LOG_TO_FILE=false
LOG_FILE_PATH=logs/app.log
LOG_MAX_BYTES=10485760
LOG_BACKUP_COUNT=5
LOG_FORMAT=%(asctime)s | %(levelname)s | %(trace_id)s | %(name)s | %(message)s
LOG_ENSURE_FIELDS=trace_id
```

- `LOG_ENSURE_FIELDS`：逗号分隔；这些字段在未绑定时写入占位符 `-`，避免格式化报错。
- 扩展新字段（如 `user_id`）：`bind(user_id=...)`，把 `%(user_id)s` 加进 `LOG_FORMAT`，并可选写入 `LOG_ENSURE_FIELDS`。

## 使用

```python
from common.logger import setup_logging, get_logger, traced, bind, log_context

setup_logging()
log = get_logger(__name__)

with traced():  # 自动生成 trace_id
    log.info("hello")

with log_context(trace_id="manual-id", user_id="42"):
    log.info("with extra context")
```

默认日志行示例：

```text
2026-09-24 15:00:00,000 | INFO | a1b2c3... | common.logger | hello
```

## 业务侧日志约定

| 层 | 级别 | 内容 |
|----|------|------|
| `common.config.models` | INFO | 加载到的模型 id 列表、default、选中模型（不含 api_key） |
| `llm.client` | INFO/ERROR | 一次 chat / stream 的开始、结束、失败 |
| `llm.session` | INFO/ERROR | 一轮对话开始/成功/失败；clear / switch_client |
| `llm.protocols.common` | ERROR | 上游 HTTP 错误摘要；DEBUG 保留 body 截断 |
| `llm.transport.http` | DEBUG | 请求 method/url/status |
| `llm.cli` | INFO/ERROR | 会话启动、切模型、用户可见错误 |

上下文字段（如 `trace_id`）由入口 `traced()` / `bind()` 注入，上述日志自动带上。
