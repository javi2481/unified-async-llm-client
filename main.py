"""Test script: asks one question in normal mode and in streaming mode.

Usage:
    python main.py              # uses LLM_PROVIDER from .env
    python main.py anthropic    # choose the provider from the command line
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

    # 1) Normal mode: wait for the full answer.
    print("\n=== Normal mode ===")
    start = time.perf_counter()
    response = await llm.generate(MESSAGES)
    if response.error:
        print(f"[error] {response.error.type}: {response.error.message}")
    else:
        print(response.content)
        print(f"({time.perf_counter() - start:.2f}s)")

    # 2) Streaming mode: print each fragment as it arrives.
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
    sys.stdout.reconfigure(encoding="utf-8")  # so Windows prints "¿" and accents
    asyncio.run(main())
