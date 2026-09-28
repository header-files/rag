"""Assemble History + Context + Query into LLM messages."""

from __future__ import annotations

from llm.types import Message
from rag.types import SearchHit


class ContextBuilder:
    def __init__(self, *, system_prompt: str, max_context_chars: int) -> None:
        self.system_prompt = system_prompt
        self.max_context_chars = max_context_chars

    def build(
        self,
        *,
        query: str,
        history: list[Message],
        hits: list[SearchHit],
    ) -> tuple[list[Message], list[SearchHit]]:
        """Return (messages, used_hits).

        History / Context / Query stay separate conceptually. Because the LLM
        Message model only supports system/user/assistant, Context and Query are
        mapped into one structured user message.
        """
        used = self._trim_hits(hits)
        context_blocks = [
            f"[{index}] 文件：{hit.filename} 片段：{hit.chunk_index}\n{hit.text}"
            for index, hit in enumerate(used, start=1)
        ]
        context_text = "\n\n".join(context_blocks)
        user_content = (
            "## 资料\n"
            f"{context_text}\n\n"
            "## 问题\n"
            f"{query.strip()}"
        )
        messages: list[Message] = [
            Message(role="system", content=self.system_prompt),
            *history,
            Message(role="user", content=user_content),
        ]
        return messages, used

    def _trim_hits(self, hits: list[SearchHit]) -> list[SearchHit]:
        kept: list[SearchHit] = []
        total = 0
        for hit in hits:
            size = len(hit.text)
            if kept and total + size > self.max_context_chars:
                break
            if not kept and size > self.max_context_chars:
                kept.append(
                    SearchHit(
                        chunk_id=hit.chunk_id,
                        document_id=hit.document_id,
                        filename=hit.filename,
                        chunk_index=hit.chunk_index,
                        text=hit.text[: self.max_context_chars],
                        score=hit.score,
                    )
                )
                break
            kept.append(hit)
            total += size
        return kept
