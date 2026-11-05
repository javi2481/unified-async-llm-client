"""Manager tests with a fake client: no API keys or internet needed."""

import pytest

from clients import BaseLLMClient, LLMClientError
from manager import AsyncLLMManager
from schemas import ChatMessage, LLMError, ModelConfig, ModelResponse

CONFIG = ModelConfig(provider="openai", model="fake", api_key="k", max_retries=2)
MESSAGES = [ChatMessage(role="user", content="hi")]

RATE_LIMIT = LLMClientError(LLMError(type="rate_limit", message="429", retryable=True))
AUTH_ERROR = LLMClientError(LLMError(type="authentication", message="401"))


class FakeClient(BaseLLMClient):
    """Returns (or raises) the given results, one per call."""

    def __init__(self, results: list) -> None:
        super().__init__(CONFIG)
        self.results = results
        self.calls = 0

    async def generate(self, messages):
        self.calls += 1
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return ModelResponse(content=result, provider="openai", model="fake")

    async def stream(self, messages):
        for text in ["Hola", " mundo"]:
            yield text
        if self.results:
            raise self.results.pop(0)

    async def close(self):
        pass


@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    """Skip the real waits between retries so the tests are fast."""

    async def fake_sleep(seconds):
        pass

    monkeypatch.setattr("manager.asyncio.sleep", fake_sleep)


async def test_retries_rate_limit_and_then_succeeds() -> None:
    client = FakeClient([RATE_LIMIT, "ok"])
    response = await AsyncLLMManager(CONFIG, client).generate(MESSAGES)
    assert response.content == "ok"
    assert client.calls == 2


async def test_returns_error_when_retries_run_out() -> None:
    client = FakeClient([RATE_LIMIT, RATE_LIMIT, RATE_LIMIT])
    response = await AsyncLLMManager(CONFIG, client).generate(MESSAGES)
    assert response.error.type == "rate_limit"
    assert client.calls == 3  # first attempt + 2 retries


async def test_does_not_retry_invalid_api_key() -> None:
    client = FakeClient([AUTH_ERROR])
    response = await AsyncLLMManager(CONFIG, client).generate(MESSAGES)
    assert response.error.type == "authentication"
    assert client.calls == 1


async def test_unexpected_exception_does_not_crash() -> None:
    client = FakeClient([RuntimeError("boom")])
    response = await AsyncLLMManager(CONFIG, client).generate(MESSAGES)
    assert response.error.type == "unknown"


async def test_stream_yields_text_fragments() -> None:
    llm = AsyncLLMManager(CONFIG, FakeClient([]))
    texts = [chunk.text async for chunk in llm.stream(MESSAGES)]
    assert texts == ["Hola", " mundo"]


async def test_stream_error_comes_as_last_chunk() -> None:
    llm = AsyncLLMManager(CONFIG, FakeClient([RATE_LIMIT]))
    chunks = [chunk async for chunk in llm.stream(MESSAGES)]
    assert [c.text for c in chunks[:2]] == ["Hola", " mundo"]
    assert chunks[-1].error.type == "rate_limit"


def test_from_env_chooses_provider(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-test")
    llm = AsyncLLMManager.from_env("anthropic")
    assert type(llm.client).__name__ == "AnthropicClient"


def test_from_env_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError):
        AsyncLLMManager.from_env("gemini")
