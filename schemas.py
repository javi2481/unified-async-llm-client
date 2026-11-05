"""Pydantic models used by every client.

Validating here avoids passing nested dicts around the code.
"""

from typing import Literal

from pydantic import BaseModel, Field, SecretStr

Role = Literal["system", "user", "assistant"]
Provider = Literal["openai", "anthropic"]
ErrorType = Literal["rate_limit", "connection", "authentication", "api", "unknown"]


class ChatMessage(BaseModel):
    role: Role
    content: str = Field(min_length=1)


class ModelConfig(BaseModel):
    provider: Provider
    model: str = Field(min_length=1)
    api_key: SecretStr  # hidden when the config is printed
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int = Field(default=512, gt=0)
    max_retries: int = Field(default=2, ge=0)
    base_url: str | None = None  # optional, for OpenAI-compatible providers


class LLMError(BaseModel):
    type: ErrorType
    message: str
    retryable: bool = False


class ModelResponse(BaseModel):
    content: str = ""
    provider: Provider
    model: str
    error: LLMError | None = None


class StreamChunk(BaseModel):
    """A piece of text from the stream, or an error that ends it."""

    text: str = ""
    error: LLMError | None = None
