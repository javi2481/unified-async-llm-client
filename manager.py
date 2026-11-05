"""AsyncLLMManager: chooses the provider and handles errors.
AsyncLLMManager: elige el proveedor y maneja los errores. La app solo habla
con el manager, y sus métodos nunca lanzan excepciones: si algo falla, el
error viene dentro de la respuesta.
"""

import asyncio
import os
from collections.abc import AsyncIterator

from clients import AnthropicClient, BaseLLMClient, LLMClientError, OpenAIClient
from schemas import ChatMessage, LLMError, ModelConfig, ModelResponse, StreamChunk

# Registro de proveedores. Para agregar uno: crear su cliente y sumarlo acá.
CLIENTS: dict[str, type[BaseLLMClient]] = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
}


# Entry point of the app: picks the client by provider and never lets errors escape.
# Punto de entrada de la app: elige el cliente según el proveedor y nunca deja escapar errores.
class AsyncLLMManager:
    # Create the client for the configured provider (or use the given one, e.g. in tests).
    # Crea el cliente del proveedor configurado (o usa el que se le pasa, ej. en los tests).
    def __init__(self, config: ModelConfig, client: BaseLLMClient | None = None) -> None:
        self.config = config
        self.client = client or CLIENTS[config.provider](config)

    # Build the manager from the environment variables (.env).
    # Arma el manager a partir de las variables de entorno (.env).
    @classmethod
    def from_env(cls, provider: str | None = None) -> "AsyncLLMManager":
        provider = provider or os.getenv("LLM_PROVIDER", "openai")
        if provider not in CLIENTS:
            raise ValueError(f"Unknown provider '{provider}'. Use: {', '.join(CLIENTS)}")

        prefix = provider.upper()  # OPENAI_... o ANTHROPIC_...
        api_key = os.getenv(f"{prefix}_API_KEY")
        model = os.getenv(f"{prefix}_MODEL")
        if not api_key or not model:
            raise ValueError(f"Set {prefix}_API_KEY and {prefix}_MODEL in the .env file")

        config = ModelConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=os.getenv("LLM_TEMPERATURE") or None,
            max_tokens=os.getenv("LLM_MAX_TOKENS", 512),
            max_retries=os.getenv("LLM_MAX_RETRIES", 2),
            base_url=os.getenv(f"{prefix}_BASE_URL") or None,
        )
        return cls(config)

    # Return the full answer, retrying temporary errors (rate limit, network) after 1s, 2s, 4s...
    # Devuelve la respuesta completa; reintenta errores temporales (rate limit, red) tras 1s, 2s, 4s...
    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        for attempt in range(self.config.max_retries + 1):
            try:
                return await self.client.generate(messages)
            except LLMClientError as e:
                error = e.error
            except Exception as e:  # cualquier otro error: nunca cortar el programa
                error = LLMError(type="unknown", message=str(e))

            is_last_attempt = attempt == self.config.max_retries
            if not error.retryable or is_last_attempt:
                break
            wait = 2**attempt  # espera exponencial: 1s, 2s, 4s...
            print(f"[retry] {error.type} error, retrying in {wait}s...")
            await asyncio.sleep(wait)

        return ModelResponse(provider=self.config.provider, model=self.config.model, error=error)

    # Yield text as it arrives; on failure the last chunk carries the error (no retry, to avoid repeated text).
    # Devuelve el texto a medida que llega; si falla, el último chunk trae el error (sin reintento, para no repetir texto).
    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[StreamChunk]:
        try:
            async for text in self.client.stream(messages):
                yield StreamChunk(text=text)
        except LLMClientError as e:
            yield StreamChunk(error=e.error)
        except Exception as e:
            yield StreamChunk(error=LLMError(type="unknown", message=str(e)))

    # Close the connections of the underlying client.
    # Cierra las conexiones del cliente que se está usando.
    async def close(self) -> None:
        await self.client.close()
