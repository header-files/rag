"""CLI entrypoint: interactive multi-turn chat."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from common.config.errors import ConfigError
from common.logger import get_logger, setup_logging, traced
from llm.client import LLMClient
from llm.errors import LLMError
from llm.session import ChatSession

app = typer.Typer(
    add_completion=False,
    pretty_exceptions_show_locals=False,
    help="Multi-protocol LLM chat CLI",
)

log = get_logger(__name__)


def _print_llm_error(exc: LLMError) -> None:
    parts = [f"Error ({exc.type}): {exc.message}"]
    if exc.status_code is not None:
        parts.append(f"status={exc.status_code}")
    if exc.provider_code:
        parts.append(f"code={exc.provider_code}")
    if exc.retryable is not None:
        parts.append(f"retryable={exc.retryable}")
    typer.secho(" | ".join(parts), fg=typer.colors.RED, err=True)
    log.error("LLM error: %s", " | ".join(parts))


def _print_config_error(exc: ConfigError) -> None:
    typer.secho(f"Config error: {exc.message}", fg=typer.colors.RED, err=True)
    log.error("Config error: %s", exc.message)


@app.callback(invoke_without_command=True)
def main(
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model id from .env (MODEL_<ID>_*)"
    ),
    env_file: Optional[Path] = typer.Option(
        None, "--env-file", help="Path to .env (default: ./.env)"
    ),
    list_models: bool = typer.Option(
        False, "--list-models", help="List configured models and exit"
    ),
    no_stream: bool = typer.Option(
        False, "--no-stream", help="Disable streaming responses"
    ),
    show_reasoning: bool = typer.Option(
        False, "--show-reasoning", help="Print reasoning stream/text"
    ),
    system: Optional[str] = typer.Option(
        None, "--system", help="Optional system prompt"
    ),
) -> None:
    """Start an interactive chat session."""
    try:
        setup_logging(env_path=env_file)
        if list_models:
            default_id, models = LLMClient.list_models(env_path=env_file)
            for mid, cfg in sorted(models.items()):
                mark = "*" if mid == default_id else " "
                typer.echo(
                    f"{mark} {mid}: protocol={cfg.protocol} name={cfg.name} "
                    f"base_url={cfg.base_url}"
                )
            return

        client = LLMClient.from_env(model, env_path=env_file)
        session = ChatSession(client, system_prompt=system)
        with traced():
            log.info(
                "Chat session started model_id=%s protocol=%s name=%s",
                client.config.id,
                client.config.protocol,
                client.config.name,
            )
            typer.echo(
                f"Chatting with '{client.config.id}' "
                f"({client.config.protocol} / {client.config.name}). "
                f"Commands: /exit /clear /model <id>"
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
                    session.clear()
                    typer.echo("(context cleared)")
                    continue
                if text.startswith("/model"):
                    parts = text.split(maxsplit=1)
                    if len(parts) < 2 or not parts[1].strip():
                        typer.echo("Usage: /model <id>")
                        continue
                    new_id = parts[1].strip()
                    with traced():
                        try:
                            new_client = LLMClient.from_env(
                                new_id, env_path=env_file
                            )
                        except ConfigError as exc:
                            _print_config_error(exc)
                            continue
                        except LLMError as exc:
                            _print_llm_error(exc)
                            continue
                        session.switch_client(new_client, clear_history=True)
                        log.info("Switched model to %s", new_client.config.id)
                        typer.echo(
                            f"(switched to '{new_client.config.id}', "
                            "context cleared)"
                        )
                    continue

                with traced():
                    try:
                        if no_stream:
                            response = session.send(text)
                            if show_reasoning and response.reasoning:
                                typer.secho(
                                    f"[reasoning]\n{response.reasoning}\n"
                                    "[/reasoning]",
                                    fg=typer.colors.CYAN,
                                )
                            typer.echo(f"assistant: {response.content}")
                        else:
                            typer.echo("assistant: ", nl=False)
                            for chunk in session.send_stream(text):
                                if chunk.type == "reasoning" and show_reasoning:
                                    typer.secho(
                                        chunk.delta, fg=typer.colors.CYAN, nl=False
                                    )
                                elif chunk.type == "content":
                                    typer.echo(chunk.delta, nl=False)
                            typer.echo()
                    except LLMError as exc:
                        _print_llm_error(exc)
                    except KeyboardInterrupt:
                        typer.echo()
                        break
    except ConfigError as exc:
        _print_config_error(exc)
        raise typer.Exit(code=1) from exc
    except LLMError as exc:
        _print_llm_error(exc)
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
