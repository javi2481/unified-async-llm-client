from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from schemas import ChatMessage, LLMError, ModelConfig, ModelResponse


class LLMClientError(Exception):
    """Common error for all providers, so the rest of the app
    does not need to know the exceptions of each SDK."""

    def __init__(self, error: LLMError) -> None:
        super().__init__(error.message)
        self.error = error


class BaseLLMClient(ABC):
    """Contract that every provider client must follow."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abstractmethod
    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        """Return the full answer."""

    @abstractmethod
    def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """Yield the answer in text fragments as they arrive."""

    @abstractmethod
    async def close(self) -> None:
        """Close the HTTP connections of the SDK."""
