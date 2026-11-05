from collections.abc import AsyncIterator

import anthropic
from anthropic import AsyncAnthropic

from clients.base import BaseLLMClient, LLMClientError
from schemas import ChatMessage, LLMError, ModelConfig, ModelResponse


# Client for Anthropic (Claude) using the async SDK.
# Cliente para Anthropic (Claude) usando el SDK asíncrono.
class AnthropicClient(BaseLLMClient):
    # Create the async Anthropic SDK client with the config values.
    # Crea el cliente asíncrono del SDK de Anthropic con los valores de la config.
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.client = AsyncAnthropic(
            api_key=config.api_key.get_secret_value(),
            base_url=config.base_url,
            max_retries=0,  # los reintentos los hace AsyncLLMManager
        )

    # Convert our messages and config into the format Anthropic expects.
    # Convierte nuestros mensajes y config al formato que espera Anthropic.
    def _build_params(self, messages: list[ChatMessage]) -> dict:
        # Anthropic recibe el mensaje "system" como parámetro aparte, no en la lista.
        system = "\n".join(m.content for m in messages if m.role == "system")
        params = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,  # obligatorio en Anthropic
            "messages": [
                {"role": m.role, "content": m.content}
                for m in messages
                if m.role != "system"
            ],
        }
        if system:
            params["system"] = system
        if self.config.temperature is not None:
            # El SDK 2.x sacó `temperature` de la firma (los modelos actuales no la usan).
            # Los modelos anteriores todavía la aceptan si se envía por extra_body.
            params["extra_body"] = {"temperature": self.config.temperature}
        return params

    # Ask Anthropic and return the full answer as a ModelResponse.
    # Le pregunta a Anthropic y devuelve la respuesta completa como ModelResponse.
    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        try:
            response = await self.client.messages.create(**self._build_params(messages))
        except anthropic.AnthropicError as e:
            raise to_client_error(e) from e

        # La respuesta es una lista de bloques; nos quedamos con los de texto.
        text = "".join(block.text for block in response.content if block.type == "text")
        return ModelResponse(content=text, provider="anthropic", model=response.model)

    # Open an Anthropic stream and yield each text fragment.
    # Abre un stream de Anthropic y devuelve (yield) cada fragmento de texto.
    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        try:
            async with self.client.messages.stream(**self._build_params(messages)) as stream:
                async for text in stream.text_stream:
                    yield text
        except anthropic.AnthropicError as e:
            raise to_client_error(e) from e

    # Close the SDK's HTTP connections.
    # Cierra las conexiones HTTP del SDK.
    async def close(self) -> None:
        await self.client.close()


# Translate an Anthropic SDK exception into our common error.
# Traduce una excepción del SDK de Anthropic a nuestro error común.
def to_client_error(e: anthropic.AnthropicError) -> LLMClientError:
    if isinstance(e, anthropic.RateLimitError):
        error = LLMError(type="rate_limit", message=str(e), retryable=True)
    elif isinstance(e, anthropic.APIConnectionError):  # incluye los timeouts
        error = LLMError(type="connection", message="Could not reach the API", retryable=True)
    elif isinstance(e, anthropic.AuthenticationError):
        error = LLMError(type="authentication", message=str(e))
    else:
        error = LLMError(type="api", message=str(e))
    return LLMClientError(error)
