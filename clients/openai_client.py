from collections.abc import AsyncIterator

import openai
from openai import AsyncOpenAI

from clients.base import BaseLLMClient, LLMClientError
from schemas import ChatMessage, LLMError, ModelConfig, ModelResponse


class OpenAIClient(BaseLLMClient):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.client = AsyncOpenAI(
            api_key=config.api_key.get_secret_value(),
            base_url=config.base_url,
            max_retries=0,  # retries are done by AsyncLLMManager
        )

    def _build_params(self, messages: list[ChatMessage]) -> dict:
        params = {
            "model": self.config.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_completion_tokens": self.config.max_tokens,
        }
        if self.config.temperature is not None:
            params["temperature"] = self.config.temperature
        return params

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        try:
            response = await self.client.chat.completions.create(**self._build_params(messages))
        except openai.OpenAIError as e:
            raise to_client_error(e) from e

        return ModelResponse(
            content=response.choices[0].message.content or "",  # can be None
            provider="openai",
            model=response.model,
        )

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        try:
            stream = await self.client.chat.completions.create(
                **self._build_params(messages), stream=True
            )
            async for chunk in stream:
                # Some chunks come without text (for example, the last one).
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except openai.OpenAIError as e:
            raise to_client_error(e) from e

    async def close(self) -> None:
        await self.client.close()


def to_client_error(e: openai.OpenAIError) -> LLMClientError:
    """Translate an OpenAI SDK exception into our common error."""
    if isinstance(e, openai.RateLimitError):
        error = LLMError(type="rate_limit", message=str(e), retryable=True)
    elif isinstance(e, openai.APIConnectionError):  # also covers timeouts
        error = LLMError(type="connection", message="Could not reach the API", retryable=True)
    elif isinstance(e, openai.AuthenticationError):
        error = LLMError(type="authentication", message=str(e))
    else:
        error = LLMError(type="api", message=str(e))
    return LLMClientError(error)
