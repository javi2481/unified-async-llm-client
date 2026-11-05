"""AsyncLLMManager: chooses the provider and handles errors.

The app only talks to the manager. Its methods never raise: if something
fails, the error comes inside the response.
"""

import asyncio
import os
from collections.abc import AsyncIterator

from clients import AnthropicClient, BaseLLMClient, LLMClientError, OpenAIClient
from schemas import ChatMessage, LLMError, ModelConfig, ModelResponse, StreamChunk

# To add a provider: create its client and add it here.
CLIENTS: dict[str, type[BaseLLMClient]] = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
}


class AsyncLLMManager:
    def __init__(self, config: ModelConfig, client: BaseLLMClient | None = None) -> None:
        self.config = config
        # `client` lets the tests use a fake client.
        self.client = client or CLIENTS[config.provider](config)

    @classmethod
    def from_env(cls, provider: str | None = None) -> "AsyncLLMManager":
        """Create the manager from the environment variables (.env)."""
        provider = provider or os.getenv("LLM_PROVIDER", "openai")
        if provider not in CLIENTS:
            raise ValueError(f"Unknown provider '{provider}'. Use: {', '.join(CLIENTS)}")

        prefix = provider.upper()  # OPENAI_... or ANTHROPIC_...
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

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        """Return the answer. If it fails, retry the temporary errors
        (rate limit, network) waiting 1s, 2s, 4s... between attempts."""
        for attempt in range(self.config.max_retries + 1):
            try:
                return await self.client.generate(messages)
            except LLMClientError as e:
                error = e.error
            except Exception as e:  # any other error: never crash the program
                error = LLMError(type="unknown", message=str(e))

            is_last_attempt = attempt == self.config.max_retries
            if not error.retryable or is_last_attempt:
                break
            wait = 2**attempt
            print(f"[retry] {error.type} error, retrying in {wait}s...")
            await asyncio.sleep(wait)

        return ModelResponse(provider=self.config.provider, model=self.config.model, error=error)

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[StreamChunk]:
        """Yield the text as it arrives. If it fails, the last chunk has the error.

        The stream is not retried: if it fails halfway, retrying would
        repeat the text the user already saw.
        """
        try:
            async for text in self.client.stream(messages):
                yield StreamChunk(text=text)
        except LLMClientError as e:
            yield StreamChunk(error=e.error)
        except Exception as e:
            yield StreamChunk(error=LLMError(type="unknown", message=str(e)))

    async def close(self) -> None:
        await self.client.close()
