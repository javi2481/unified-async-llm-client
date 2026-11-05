"""Tests of the Pydantic validations.
Tests de las validaciones de Pydantic.
"""

import pytest
from pydantic import ValidationError

from schemas import ChatMessage, ModelConfig


# A valid config is accepted and uses the default values.
# Una config válida se acepta y usa los valores por defecto.
def test_valid_config() -> None:
    config = ModelConfig(provider="openai", model="gpt-test", api_key="sk-secret")
    assert config.max_tokens == 512


# Temperature outside 0-2 is rejected.
# Una temperatura fuera de 0-2 se rechaza.
@pytest.mark.parametrize("temperature", [-0.1, 2.1])
def test_temperature_must_be_between_0_and_2(temperature: float) -> None:
    with pytest.raises(ValidationError):
        ModelConfig(provider="openai", model="m", api_key="k", temperature=temperature)


# max_tokens must be greater than 0.
# max_tokens tiene que ser mayor que 0.
def test_max_tokens_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(provider="openai", model="m", api_key="k", max_tokens=0)


# Only "openai" and "anthropic" are valid providers.
# Solo "openai" y "anthropic" son proveedores válidos.
def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(provider="gemini", model="m", api_key="k")


# The API key does not appear when the config is printed.
# La API key no aparece al imprimir la config.
def test_api_key_is_hidden() -> None:
    config = ModelConfig(provider="openai", model="m", api_key="sk-secret")
    assert "sk-secret" not in str(config)


# A message with a role other than system/user/assistant is rejected.
# Un mensaje con un role distinto de system/user/assistant se rechaza.
def test_invalid_role_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatMessage(role="robot", content="hi")
