"""Query rewrite for retrieval (optional, default off)."""

from __future__ import annotations

from common.config.rag import DEFAULT_QUERY_REWRITE_PROMPT
from common.logger import get_logger
from llm.client import LLMClient
from llm.types import ChatRequest, Message

log = get_logger(__name__)


class QueryRewriter:
    def __init__(
        self,
        llm: LLMClient,
        *,
        prompt: str = DEFAULT_QUERY_REWRITE_PROMPT,
    ) -> None:
        self.llm = llm
        self.prompt = prompt

    def rewrite(self, query: str, history: list[Message]) -> str:
        """Return a retrieval query; fall back to the original on failure/empty."""
        original = query.strip()
        if not original:
            return original
        if not history:
            log.info("query rewrite skipped reason=no_history")
            return original

        messages = [
            Message(role="system", content=self.prompt),
            *history,
            Message(
                role="user",
                content=(
                    "请根据以上对话历史，改写下面的最新问题为独立检索查询：\n"
                    f"{original}"
                ),
            ),
        ]
        log.info(
            "query rewrite start history_messages=%d query_chars=%d",
            len(history),
            len(original),
        )
        try:
            response = self.llm.chat(ChatRequest(messages=messages))
        except Exception as exc:
            log.error("query rewrite failed error=%s; using original query", exc)
            return original

        rewritten = (response.content or "").strip()
        if not rewritten:
            log.warning("query rewrite empty; using original query")
            return original
        if (
            len(rewritten) >= 2
            and rewritten[0] == rewritten[-1]
            and rewritten[0] in {"'", '"', "“", "”", "「", "」"}
        ):
            rewritten = rewritten[1:-1].strip() or original
        log.info(
            "query rewrite done original_chars=%d rewritten_chars=%d",
            len(original),
            len(rewritten),
        )
        return rewritten
