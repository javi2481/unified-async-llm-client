"""Pydantic models used by every client.
Modelos de Pydantic que usan todos los clientes: validan los datos y evitan
pasar diccionarios anidados por todo el código.
"""

from typing import Literal

from pydantic import BaseModel, Field, SecretStr

Role = Literal["system", "user", "assistant"]
Provider = Literal["openai", "anthropic"]
ErrorType = Literal["rate_limit", "connection", "authentication", "api", "unknown"]


# One message of the conversation: who says it (role) and what it says.
# Un mensaje de la conversación: quién lo dice (role) y qué dice (content).
class ChatMessage(BaseModel):
    role: Role
    content: str = Field(min_length=1)  # no puede estar vacío


# Model settings, validated before any API call (e.g. temperature 0-2).
# Configuración del modelo, validada antes de llamar a la API (ej. temperatura 0-2).
class ModelConfig(BaseModel):
    provider: Provider
    model: str = Field(min_length=1)
    api_key: SecretStr  # se oculta al imprimir la config
    temperature: float | None = Field(default=None, ge=0, le=2)  # None = no se envía
    max_tokens: int = Field(default=512, gt=0)
    max_retries: int = Field(default=2, ge=0)
    base_url: str | None = None  # opcional, para proveedores compatibles con OpenAI


# Controlled error returned instead of letting an exception crash the program.
# Error controlado que se devuelve en lugar de dejar que una excepción corte el programa.
class LLMError(BaseModel):
    type: ErrorType
    message: str
    retryable: bool = False  # True si vale la pena reintentar (rate limit, red)


# Model answer, the same for every provider. `error` is set if the call failed.
# Respuesta del modelo, igual para todos los proveedores. Si falló, viene en `error`.
class ModelResponse(BaseModel):
    content: str = ""
    provider: Provider
    model: str
    error: LLMError | None = None


# A piece of text from the stream, or the error that ends it.
# Un fragmento de texto del streaming, o el error que lo termina.
class StreamChunk(BaseModel):
    text: str = ""
    error: LLMError | None = None
