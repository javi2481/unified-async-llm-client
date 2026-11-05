from collections.abc import AsyncIterator

import openai
from openai import AsyncOpenAI

from clients.base import BaseLLMClient, LLMClientError
from schemas import ChatMessage, LLMError, ModelConfig, ModelResponse


# Client for OpenAI (or any OpenAI-compatible API) using the async SDK.
# Cliente para OpenAI (o cualquier API compatible) usando el SDK asíncrono.
class OpenAIClient(BaseLLMClient):
    # Create the async OpenAI SDK client with the config values.
    # Crea el cliente asíncrono del SDK de OpenAI con los valores de la config.
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.client = AsyncOpenAI(
            api_key=config.api_key.get_secret_value(),
            base_url=config.base_url,
            max_retries=0,  # los reintentos los hace AsyncLLMManager
        )

    # Convert our messages and config into the format OpenAI expects.
    # Convierte nuestros mensajes y config al formato que espera OpenAI.
    def _build_params(self, messages: list[ChatMessage]) -> dict:
        params = {
            "model": self.config.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_completion_tokens": self.config.max_tokens,
        }
        if self.config.temperature is not None:
            params["temperature"] = self.config.temperature
        return params

    # Ask OpenAI and return the full answer as a ModelResponse.
    # Le pregunta a OpenAI y devuelve la respuesta completa como ModelResponse.
    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        try:
            response = await self.client.chat.completions.create(**self._build_params(messages))
        except openai.OpenAIError as e:
            raise to_client_error(e) from e

        return ModelResponse(
            content=response.choices[0].message.content or "",  # puede venir None
            provider="openai",
            model=response.model,
        )

    # Ask OpenAI with stream=True and yield each text fragment.
    # Le pregunta a OpenAI con stream=True y devuelve (yield) cada fragmento de texto.
    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        try:
            stream = await self.client.chat.completions.create(
                **self._build_params(messages), stream=True
            )
            async for chunk in stream:
                # Algunos chunks vienen sin texto (por ejemplo, el último).
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except openai.OpenAIError as e:
            raise to_client_error(e) from e

    # Close the SDK's HTTP connections.
    # Cierra las conexiones HTTP del SDK.
    async def close(self) -> None:
        await self.client.close()


# Translate an OpenAI SDK exception into our common error.
# Traduce una excepción del SDK de OpenAI a nuestro error común.
def to_client_error(e: openai.OpenAIError) -> LLMClientError:
    if isinstance(e, openai.RateLimitError):
        error = LLMError(type="rate_limit", message=str(e), retryable=True)
    elif isinstance(e, openai.APIConnectionError):  # incluye los timeouts
        error = LLMError(type="connection", message="Could not reach the API", retryable=True)
    elif isinstance(e, openai.AuthenticationError):
        error = LLMError(type="authentication", message=str(e))
    else:
        error = LLMError(type="api", message=str(e))
    return LLMClientError(error)
