import pytest
from pydantic import ValidationError

from schemas import ChatMessage, ModelConfig


def test_valid_config() -> None:
    config = ModelConfig(provider="openai", model="gpt-test", api_key="sk-secret")
    assert config.max_tokens == 512


@pytest.mark.parametrize("temperature", [-0.1, 2.1])
def test_temperature_must_be_between_0_and_2(temperature: float) -> None:
    with pytest.raises(ValidationError):
        ModelConfig(provider="openai", model="m", api_key="k", temperature=temperature)


def test_max_tokens_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(provider="openai", model="m", api_key="k", max_tokens=0)


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(provider="gemini", model="m", api_key="k")


def test_api_key_is_hidden() -> None:
    config = ModelConfig(provider="openai", model="m", api_key="sk-secret")
    assert "sk-secret" not in str(config)


def test_invalid_role_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatMessage(role="robot", content="hi")
