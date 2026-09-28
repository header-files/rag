"""CLI client for the RAG HTTP API."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from common.config.api import load_api_settings
from common.config.errors import ConfigError
from common.config.rag import load_rag_settings
from common.logger import get_logger, setup_logging
from rag.cli.client import ApiClientError, RagApiClient

app = typer.Typer(
    add_completion=False,
    pretty_exceptions_show_locals=False,
    help="RAG 命令行客户端：上传文档并通过后端接口多轮问答",
)
log = get_logger(__name__)


def _client(base_url: str | None, env_file: Path | None) -> RagApiClient:
    settings = load_api_settings(env_file)
    resolved = (base_url or settings.base_url).rstrip("/")
    return RagApiClient(resolved, timeout=settings.timeout_seconds)


def _print_api_error(exc: ApiClientError) -> None:
    suffix = f" (status={exc.status_code})" if exc.status_code is not None else ""
    typer.secho(f"请求失败{suffix}: {exc.message}", fg=typer.colors.RED, err=True)


def _print_citations(citations: list) -> None:
    if not citations:
        return
    typer.secho("引用：", fg=typer.colors.CYAN)
    for index, item in enumerate(citations, start=1):
        filename = item.get("filename", "")
        chunk_index = item.get("chunk_index", "")
        score = item.get("score", "")
        typer.secho(
            f"  [{index}] {filename}#{chunk_index} score={score}",
            fg=typer.colors.CYAN,
        )


@app.callback()
def cli() -> None:
    """RAG 命令行客户端。"""


@app.command()
def serve(
    host: Optional[str] = typer.Option(None, "--host", help="监听地址，默认 API_HOST"),
    port: Optional[int] = typer.Option(None, "--port", help="监听端口，默认 API_PORT"),
    env_file: Optional[Path] = typer.Option(None, "--env-file", help=".env 路径"),
) -> None:
    """启动 HTTP 服务。FAISS 与 Session 位于该进程内存中。"""
    import uvicorn

    from api import create_app

    try:
        setup_logging(env_path=env_file)
        api_settings = load_api_settings(env_file)
        load_rag_settings(env_file)
        application = create_app(env_path=env_file)
    except ConfigError as exc:
        typer.secho(f"配置错误: {exc.message}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    bind_host = host or api_settings.host
    bind_port = port or api_settings.port
    log.info("API listen host=%s port=%d", bind_host, bind_port)
    typer.echo(f"RAG API http://{bind_host}:{bind_port}")
    uvicorn.run(application, host=bind_host, port=bind_port, log_config=None)


@app.command()
def upload(
    path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="API 地址"),
    env_file: Optional[Path] = typer.Option(None, "--env-file", help=".env 路径"),
) -> None:
    """上传一个文档（txt、md、docx、pdf）。"""
    setup_logging(env_path=env_file)
    client = _client(base_url, env_file)
    try:
        document = client.upload(path)
    except ApiClientError as exc:
        _print_api_error(exc)
        raise typer.Exit(code=1) from exc
    except OSError as exc:
        typer.secho(f"无法读取文件: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"已上传 {document.get('filename')} id={document.get('id')} "
        f"chunks={document.get('chunk_count')}"
    )


@app.command("ls")
def list_documents(
    base_url: Optional[str] = typer.Option(None, "--base-url", help="API 地址"),
    env_file: Optional[Path] = typer.Option(None, "--env-file", help=".env 路径"),
) -> None:
    """列出已入库文档。"""
    setup_logging(env_path=env_file)
    client = _client(base_url, env_file)
    try:
        documents = client.list_documents()
    except ApiClientError as exc:
        _print_api_error(exc)
        raise typer.Exit(code=1) from exc
    if not documents:
        typer.echo("(空)")
        return
    for document in documents:
        typer.echo(
            f"{document.get('id')}  {document.get('filename')}  "
            f"chunks={document.get('chunk_count')}  {document.get('created_at')}"
        )


@app.command()
def rm(
    document_id: str = typer.Argument(..., help="文档 id"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="API 地址"),
    env_file: Optional[Path] = typer.Option(None, "--env-file", help=".env 路径"),
) -> None:
    """删除一个文档及其向量（软删）。"""
    setup_logging(env_path=env_file)
    client = _client(base_url, env_file)
    try:
        client.delete(document_id)
    except ApiClientError as exc:
        _print_api_error(exc)
        raise typer.Exit(code=1) from exc
    typer.echo(f"已删除 {document_id}")


@app.command()
def ask(
    question: str = typer.Argument(..., help="问题"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="API 地址"),
    env_file: Optional[Path] = typer.Option(None, "--env-file", help=".env 路径"),
    session_id: Optional[str] = typer.Option(None, "--session-id", help="会话 id"),
    document_id: Annotated[
        list[str] | None,
        typer.Option("--document-id", help="限定文档，可重复"),
    ] = None,
) -> None:
    """单次提问（可带 session 延续多轮）。"""
    setup_logging(env_path=env_file)
    client = _client(base_url, env_file)
    try:
        payload = client.ask(
            question,
            session_id=session_id,
            document_ids=document_id,
        )
    except ApiClientError as exc:
        _print_api_error(exc)
        raise typer.Exit(code=1) from exc
    typer.echo(f"session_id={payload.get('session_id')}")
    _print_citations(payload.get("citations") or [])
    typer.echo(payload.get("answer", ""))


@app.command()
def chat(
    base_url: Optional[str] = typer.Option(None, "--base-url", help="API 地址"),
    env_file: Optional[Path] = typer.Option(None, "--env-file", help=".env 路径"),
    document_id: Annotated[
        list[str] | None,
        typer.Option("--document-id", help="限定文档，可重复"),
    ] = None,
) -> None:
    """交互多轮问答。命令：/exit /ls /upload <path> /rm <id> /clear。"""
    setup_logging(env_path=env_file)
    client = _client(base_url, env_file)
    try:
        session_id = client.create_session()
    except ApiClientError as exc:
        _print_api_error(exc)
        raise typer.Exit(code=1) from exc
    log.info(
        "RAG chat client started base_url=%s session_id=%s",
        client.base_url,
        session_id,
    )
    typer.echo(
        f"已连接 {client.base_url} session={session_id}。"
        " 输入问题，或 /exit /ls /upload <path> /rm <id> /clear"
    )
    while True:
        try:
            user_text = typer.prompt("you")
        except typer.Abort:
            typer.echo()
            break
        text = user_text.strip()
        if not text:
            continue
        if text in {"/exit", "/quit"}:
            break
        if text == "/clear":
            try:
                client.delete_session(session_id)
                session_id = client.create_session()
            except ApiClientError as exc:
                _print_api_error(exc)
                continue
            typer.echo(f"(会话已清空，新 session={session_id})")
            continue
        if text == "/ls":
            try:
                documents = client.list_documents()
            except ApiClientError as exc:
                _print_api_error(exc)
                continue
            if not documents:
                typer.echo("(空)")
                continue
            for document in documents:
                typer.echo(f"{document.get('id')}  {document.get('filename')}")
            continue
        if text.startswith("/upload"):
            parts = text.split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                typer.echo("用法: /upload <path>")
                continue
            file_path = Path(parts[1].strip()).expanduser()
            try:
                document = client.upload(file_path)
            except ApiClientError as exc:
                _print_api_error(exc)
                continue
            except OSError as exc:
                typer.secho(f"无法读取文件: {exc}", fg=typer.colors.RED, err=True)
                continue
            typer.echo(f"已上传 {document.get('filename')} id={document.get('id')}")
            continue
        if text.startswith("/rm"):
            parts = text.split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                typer.echo("用法: /rm <id>")
                continue
            try:
                client.delete(parts[1].strip())
            except ApiClientError as exc:
                _print_api_error(exc)
                continue
            typer.echo("已删除")
            continue
        try:
            typer.echo("assistant: ", nl=False)
            citations: list = []
            for event, payload in client.ask_stream(
                text,
                session_id=session_id,
                document_ids=document_id,
            ):
                if event == "session" and payload.get("session_id"):
                    session_id = str(payload["session_id"])
                elif event == "citations":
                    citations = payload.get("citations") or []
                elif event == "content":
                    typer.echo(payload.get("delta", ""), nl=False)
                elif event == "error":
                    typer.echo()
                    typer.secho(
                        f"请求失败: {payload.get('message')}",
                        fg=typer.colors.RED,
                        err=True,
                    )
                    citations = []
                    break
            else:
                typer.echo()
                _print_citations(citations)
        except ApiClientError as exc:
            typer.echo()
            _print_api_error(exc)
        except KeyboardInterrupt:
            typer.echo()
            break


if __name__ == "__main__":
    app()
