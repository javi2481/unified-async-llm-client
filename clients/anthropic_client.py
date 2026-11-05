from collections.abc import AsyncIterator

import anthropic
from anthropic import AsyncAnthropic

from clients.base import BaseLLMClient, LLMClientError
from schemas import ChatMessage, LLMError, ModelConfig, ModelResponse


class AnthropicClient(BaseLLMClient):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.client = AsyncAnthropic(
            api_key=config.api_key.get_secret_value(),
            base_url=config.base_url,
            max_retries=0,  # retries are done by AsyncLLMManager
        )

    def _build_params(self, messages: list[ChatMessage]) -> dict:
        # Anthropic receives the system prompt as a separate parameter.
        system = "\n".join(m.content for m in messages if m.role == "system")
        params = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,  # required in Anthropic
            "messages": [
                {"role": m.role, "content": m.content}
                for m in messages
                if m.role != "system"
            ],
        }
        if system:
            params["system"] = system
        if self.config.temperature is not None:
            # SDK 2.x removed `temperature` from the method signature
            # (current models do not use it). Older models still accept it here.
            params["extra_body"] = {"temperature": self.config.temperature}
        return params

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        try:
            response = await self.client.messages.create(**self._build_params(messages))
        except anthropic.AnthropicError as e:
            raise to_client_error(e) from e

        # The answer is a list of blocks; we keep the text ones.
        text = "".join(block.text for block in response.content if block.type == "text")
        return ModelResponse(content=text, provider="anthropic", model=response.model)

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        try:
            async with self.client.messages.stream(**self._build_params(messages)) as stream:
                async for text in stream.text_stream:
                    yield text
        except anthropic.AnthropicError as e:
            raise to_client_error(e) from e

    async def close(self) -> None:
        await self.client.close()


def to_client_error(e: anthropic.AnthropicError) -> LLMClientError:
    """Translate an Anthropic SDK exception into our common error."""
    if isinstance(e, anthropic.RateLimitError):
        error = LLMError(type="rate_limit", message=str(e), retryable=True)
    elif isinstance(e, anthropic.APIConnectionError):  # also covers timeouts
        error = LLMError(type="connection", message="Could not reach the API", retryable=True)
    elif isinstance(e, anthropic.AuthenticationError):
        error = LLMError(type="authentication", message=str(e))
    else:
        error = LLMError(type="api", message=str(e))
    return LLMClientError(error)
