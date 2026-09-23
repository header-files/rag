"""Chat session: maintains conversation context."""

from __future__ import annotations

from llm.client import LLMClient
from llm.types import ChatRequest, ChatResponse, Message, StreamChunk


class ChatSession:
    """In-memory multi-turn session over a unified LLMClient."""

    def __init__(
        self,
        client: LLMClient,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        self.client = client
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.messages: list[Message] = []
        if system_prompt:
            self.messages.append(Message(role="system", content=system_prompt))

    def clear(self) -> None:
        system = [m for m in self.messages if m.role == "system"]
        self.messages = list(system)

    def switch_client(self, client: LLMClient, *, clear_history: bool = True) -> None:
        """Switch underlying model client.

        By default clears non-system history to avoid cross-protocol context issues.
        """
        self.client = client
        if clear_history:
            self.clear()

    def _request(self) -> ChatRequest:
        return ChatRequest(
            messages=list(self.messages),
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            stop=None,
        )

    def send(self, user_text: str) -> ChatResponse:
        self.messages.append(Message(role="user", content=user_text))
        try:
            response = self.client.chat(self._request())
        except Exception:
            # Failed turn: remove the pending user message? Spec says only append
            # assistant on success. Keep user message so retry is possible, or pop?
            # Keeping user message is awkward on retry. Pop user on failure.
            self.messages.pop()
            raise
        self.messages.append(Message(role="assistant", content=response.content))
        return response

    def send_stream(self, user_text: str):
        """Yield StreamChunks; append assistant only after successful finish."""
        self.messages.append(Message(role="user", content=user_text))
        content_parts: list[str] = []
        finished = False
        try:
            for chunk in self.client.chat_stream(self._request()):
                if chunk.type == "content" and chunk.delta:
                    content_parts.append(chunk.delta)
                if chunk.type == "finish":
                    finished = True
                yield chunk
            if finished:
                self.messages.append(
                    Message(role="assistant", content="".join(content_parts))
                )
            else:
                self.messages.pop()
        except Exception:
            self.messages.pop()
            raise
