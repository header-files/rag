"""Protocol adapter registry."""

from __future__ import annotations

from common.config.errors import ConfigError
from common.config.models import ModelConfig, ProtocolId
from llm.protocols.anthropic_messages import AnthropicMessagesAdapter
from llm.protocols.base import ProtocolAdapter
from llm.protocols.openai_completions import OpenAICompletionsAdapter
from llm.protocols.openai_responses import OpenAIResponsesAdapter
from llm.transport.http import HttpTransport

_REGISTRY: dict[ProtocolId, type[ProtocolAdapter]] = {
    "openai-completions": OpenAICompletionsAdapter,
    "openai-responses": OpenAIResponsesAdapter,
    "anthropic-messages": AnthropicMessagesAdapter,
}


def create_adapter(
    config: ModelConfig,
    transport: HttpTransport | None = None,
) -> ProtocolAdapter:
    cls = _REGISTRY.get(config.protocol)
    if cls is None:
        raise ConfigError(
            message=f"No adapter registered for protocol '{config.protocol}'"
        )
    return cls(config, transport=transport)
