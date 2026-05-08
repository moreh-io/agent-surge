#!/usr/bin/env python3
"""Quick-start example - use the mock backend to run turns without a server.

Demonstrates the agentsurge :class:`MockBackend` for local testing and CI.
No live vLLM server is required.

This example is part of the API contract and runs in CI.

Usage::

    python examples/01_quick_start.py
"""

from __future__ import annotations

import asyncio

from agentsurge.backends.mock import MockBackend, MockConfig
from agentsurge.types import TurnResult


async def main() -> None:
    # Configure a mock backend - no network, instant responses.
    config = MockConfig(
        output_tokens=42,
        ttft_ms=5.0,
        total_ms=20.0,
    )

    async with MockBackend(config) as backend:
        # Send a single turn
        result: TurnResult = await backend.send_turn(
            session_id="demo-session",
            turn_index=0,
            messages=[{"role": "user", "content": "Explain KV caching."}],
        )

        print(f"ok={result.ok}, ttft={result.ttft_ms:.1f}ms, tokens={result.output_tokens}")
        assert result.ok
        assert result.output_tokens == 42

        # Send a multi-turn conversation
        for turn_idx in range(1, 4):
            result = await backend.send_turn(
                session_id="demo-session",
                turn_index=turn_idx,
                messages=[
                    {"role": "user", "content": "Explain KV caching."},
                    {"role": "assistant", "content": "KV caching stores..."},
                    {"role": "user", "content": f"Follow-up question {turn_idx}"},
                ],
            )
            assert result.ok

        # Inspect the call log
        assert len(backend.call_log) == 4
        print(f"Completed {len(backend.call_log)} turns")

    print("\n✓ quick-start example passed")


if __name__ == "__main__":
    asyncio.run(main())
