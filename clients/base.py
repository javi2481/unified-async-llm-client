from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from schemas import ChatMessage, LLMError, ModelConfig, ModelResponse


# Common error for all providers, so the app does not depend on each SDK's exceptions.
# Error común a todos los proveedores, así la app no depende de las excepciones de cada SDK.
class LLMClientError(Exception):
    # Store the structured error so the manager can read it.
    # Guarda el error estructurado para que el manager pueda leerlo.
    def __init__(self, error: LLMError) -> None:
        super().__init__(error.message)
        self.error = error


# Contract (abstract class) that every provider client must follow.
# Contrato (clase abstracta) que todo cliente de un proveedor tiene que cumplir.
class BaseLLMClient(ABC):
    # Save the model config so the subclasses can use it.
    # Guarda la configuración del modelo para que la usen las subclases.
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    # Send the messages and return the full answer.
    # Envía los mensajes y devuelve la respuesta completa.
    @abstractmethod
    async def generate(self, messages: list[ChatMessage]) -> ModelResponse: ...

    # Send the messages and yield the answer in fragments as they arrive.
    # Envía los mensajes y devuelve la respuesta en fragmentos a medida que llegan.
    @abstractmethod
    def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]: ...

    # Close the SDK's HTTP connections.
    # Cierra las conexiones HTTP del SDK.
    @abstractmethod
    async def close(self) -> None: ...
