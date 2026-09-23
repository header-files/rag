# 日志

`common.logger`（`src/common/logger/`）提供统一的日志初始化与获取入口。

## 结构

```text
src/common/config/log_settings.py   # LOG_* 配置解析（LogSettings）
src/common/logger/
  setup.py                          # setup_logging / get_logger
```

## 行为

| 能力 | 说明 |
|------|------|
| 控制台 | `StreamHandler`（stderr），由 `LOG_TO_CONSOLE` 开关 |
| 文件 | `RotatingFileHandler`，由 `LOG_TO_FILE` 开关 |
| 大小限制 | `LOG_MAX_BYTES`；超出后轮转生成 `app.log.1`、`app.log.2`…，继续写入新的 `app.log` |
| 保留份数 | `LOG_BACKUP_COUNT` |

## 配置（`.env`）

```env
LOG_LEVEL=INFO
LOG_TO_CONSOLE=true
LOG_TO_FILE=false
LOG_FILE_PATH=logs/app.log
LOG_MAX_BYTES=10485760
LOG_BACKUP_COUNT=5
```

## 使用

```python
from common.logger import setup_logging, get_logger

setup_logging()  # 从 .env 加载 LogSettings
log = get_logger(__name__)
log.info("hello")
```

应用入口（例如 `llm-chat`）应在启动时调用 `setup_logging`。
