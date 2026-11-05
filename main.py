"""Test script: asks one question in normal mode and in streaming mode.
Script de prueba: hace una pregunta en modo normal y en modo streaming.

Uso:
    python main.py              # usa LLM_PROVIDER del archivo .env
    python main.py anthropic    # elige el proveedor desde la línea de comandos
"""

import asyncio
import sys
import time

from dotenv import load_dotenv

from manager import AsyncLLMManager
from schemas import ChatMessage

MESSAGES = [
    ChatMessage(role="system", content="Respondé en español, en un párrafo corto."),
    ChatMessage(role="user", content="¿Qué es la entropía?"),
]


# Load the .env, create the manager and ask the question in both modes.
# Carga el .env, crea el manager y hace la pregunta en los dos modos.
async def main() -> None:
    load_dotenv()
    provider = sys.argv[1] if len(sys.argv) > 1 else None

    try:
        llm = AsyncLLMManager.from_env(provider)
    except ValueError as e:
        print(f"Configuration error: {e}")
        return

    print(f"Provider: {llm.config.provider} | Model: {llm.config.model}")
    print(f"Question: {MESSAGES[-1].content}")

    # 1) Modo normal: espera la respuesta completa.
    print("\n=== Normal mode ===")
    start = time.perf_counter()
    response = await llm.generate(MESSAGES)
    if response.error:
        print(f"[error] {response.error.type}: {response.error.message}")
    else:
        print(response.content)
        print(f"({time.perf_counter() - start:.2f}s)")

    # 2) Modo streaming: imprime cada fragmento apenas llega.
    print("\n=== Streaming mode ===")
    start = time.perf_counter()
    async for chunk in llm.stream(MESSAGES):
        if chunk.error:
            print(f"\n[error] {chunk.error.type}: {chunk.error.message}")
            break
        print(chunk.text, end="", flush=True)
    print(f"\n({time.perf_counter() - start:.2f}s)")

    await llm.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # para que Windows muestre "¿" y los acentos
    asyncio.run(main())
