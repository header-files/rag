# Agent 指南

换机器或新会话时，先读本文件，再按需深入 `docs/common/` 与 `docs/llm/`。

## 项目是什么

Python + uv 项目：公共基建（配置、日志）+ 多协议 LLM HTTP 客户端（httpx，无官方 SDK）与 CLI 多轮对话。

## 技术栈

- Python >= 3.11，包管理 uv
- httpx、python-dotenv、typer
- 标准库 logging + RotatingFileHandler
- 测试：pytest、pytest-asyncio、respx

## 目录地图

```text
src/common/               # 公共基建
  config/                 # .env 加载、各域配置（含模型、日志等）
  logger/                 # 控制台 + 按大小轮转文件
src/llm/                  # 大模型调用域
  types.py / errors.py / client.py / session.py
  transport/http.py
  protocols/
  cli/main.py             # llm-chat 入口
docs/common/              # 公共基建文档
docs/llm/                 # 大模型调用域文档
docs/AGENT.md
tests/
.env.example
```

## 必读文档顺序

1. `docs/common/CONFIG.md` — 配置约定
2. `docs/common/LOGGING.md` — 日志约定
3. `docs/llm/REQUIREMENTS.md` — 大模型域需求与边界
4. `docs/llm/ARCHITECTURE.md` — 分层与扩展点
5. `docs/llm/UNIFIED_API.md` — 客户端交互协议
6. `docs/llm/ERRORS.md` — 调用错误归一
7. 改协议时：`docs/llm/PROTOCOLS.md`
8. 改 CLI/Session：`docs/llm/SESSION.md`、`docs/llm/CLI.md`

## 扩展指引

- 公共配置：在 `src/common/config/` 增加模块；通用读 env 用 `common.config.env.load_env`
- 日志：调整 `src/common/logger/`（含 `context.py` 扩展字段）或 `common/config/log_settings.py`
- 新的公共能力：在 `src/common/` 下新增子包，并在 `docs/common/` 补文档
- 新的大模型协议：在 `src/llm/protocols/` 新增 Adapter，注册到 `registry.py`，补测试与 `docs/llm/PROTOCOLS.md`

## 常用命令

```bash
uv sync
uv run pytest
uv run llm-chat --list-models
```

## Git

- 远程：`git@github.com:header-files/llm.git`
- 主分支：`main`
- Commit 使用 Conventional Commits（`feat` / `fix` / `docs` / `test` 等）
