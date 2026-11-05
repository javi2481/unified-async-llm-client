"""Manager tests with a fake client: no API keys or internet needed.
Tests del manager con un cliente falso: no necesitan API keys ni internet.
"""

import pytest

from clients import BaseLLMClient, LLMClientError
from manager import AsyncLLMManager
from schemas import ChatMessage, LLMError, ModelConfig, ModelResponse

CONFIG = ModelConfig(provider="openai", model="fake", api_key="k", max_retries=2)
MESSAGES = [ChatMessage(role="user", content="hi")]

RATE_LIMIT = LLMClientError(LLMError(type="rate_limit", message="429", retryable=True))
AUTH_ERROR = LLMClientError(LLMError(type="authentication", message="401"))


# Fake client that returns (or raises) the given results, one per call.
# Cliente falso que devuelve (o lanza) los resultados indicados, uno por llamada.
class FakeClient(BaseLLMClient):
    # Save the list of results and count the calls.
    # Guarda la lista de resultados y cuenta las llamadas.
    def __init__(self, results: list) -> None:
        super().__init__(CONFIG)
        self.results = results
        self.calls = 0

    # Return the next result, or raise it if it is an exception.
    # Devuelve el siguiente resultado, o lo lanza si es una excepción.
    async def generate(self, messages):
        self.calls += 1
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return ModelResponse(content=result, provider="openai", model="fake")

    # Yield two fragments and then raise an error if one was given.
    # Devuelve dos fragmentos y después lanza un error si se le pasó uno.
    async def stream(self, messages):
        for text in ["Hola", " mundo"]:
            yield text
        if self.results:
            raise self.results.pop(0)

    # Nothing to close in the fake client.
    # El cliente falso no tiene nada que cerrar.
    async def close(self):
        pass


# Skip the real waits between retries so the tests run fast.
# Saltea las esperas reales entre reintentos para que los tests sean rápidos.
@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    async def fake_sleep(seconds):
        pass

    monkeypatch.setattr("manager.asyncio.sleep", fake_sleep)


# A rate limit is retried and the second attempt succeeds.
# Un rate limit se reintenta y el segundo intento funciona.
async def test_retries_rate_limit_and_then_succeeds() -> None:
    client = FakeClient([RATE_LIMIT, "ok"])
    response = await AsyncLLMManager(CONFIG, client).generate(MESSAGES)
    assert response.content == "ok"
    assert client.calls == 2


# When all retries fail, the error comes in the response (no crash).
# Si fallan todos los reintentos, el error viene en la respuesta (sin crash).
async def test_returns_error_when_retries_run_out() -> None:
    client = FakeClient([RATE_LIMIT, RATE_LIMIT, RATE_LIMIT])
    response = await AsyncLLMManager(CONFIG, client).generate(MESSAGES)
    assert response.error.type == "rate_limit"
    assert client.calls == 3  # primer intento + 2 reintentos


# An invalid API key is not retried: retrying would not fix it.
# Una API key inválida no se reintenta: reintentar no lo soluciona.
async def test_does_not_retry_invalid_api_key() -> None:
    client = FakeClient([AUTH_ERROR])
    response = await AsyncLLMManager(CONFIG, client).generate(MESSAGES)
    assert response.error.type == "authentication"
    assert client.calls == 1


# An unexpected exception becomes an "unknown" error instead of crashing.
# Una excepción inesperada se convierte en error "unknown" en lugar de cortar el programa.
async def test_unexpected_exception_does_not_crash() -> None:
    client = FakeClient([RuntimeError("boom")])
    response = await AsyncLLMManager(CONFIG, client).generate(MESSAGES)
    assert response.error.type == "unknown"


# The stream delivers the text fragments in order.
# El streaming entrega los fragmentos de texto en orden.
async def test_stream_yields_text_fragments() -> None:
    llm = AsyncLLMManager(CONFIG, FakeClient([]))
    texts = [chunk.text async for chunk in llm.stream(MESSAGES)]
    assert texts == ["Hola", " mundo"]


# If the stream fails halfway, the text already sent is kept and the error comes last.
# Si el streaming falla a mitad, se conserva el texto ya enviado y el error llega al final.
async def test_stream_error_comes_as_last_chunk() -> None:
    llm = AsyncLLMManager(CONFIG, FakeClient([RATE_LIMIT]))
    chunks = [chunk async for chunk in llm.stream(MESSAGES)]
    assert [c.text for c in chunks[:2]] == ["Hola", " mundo"]
    assert chunks[-1].error.type == "rate_limit"


# from_env creates the client of the requested provider.
# from_env crea el cliente del proveedor pedido.
def test_from_env_chooses_provider(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-test")
    llm = AsyncLLMManager.from_env("anthropic")
    assert type(llm.client).__name__ == "AnthropicClient"


# An unknown provider is rejected with a clear error.
# Un proveedor desconocido se rechaza con un error claro.
def test_from_env_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError):
        AsyncLLMManager.from_env("gemini")
