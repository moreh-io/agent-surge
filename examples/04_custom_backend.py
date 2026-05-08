#!/usr/bin/env python3
"""Custom backend - define and use a custom backend adapter.

Shows the extension point for plugging in a new backend via
the :class:`BackendBase` ABC.  Subclasses are auto-registered.

This example is part of the API contract and runs in CI.

Usage::

    python examples/04_custom_backend.py
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from agentsurge import BackendBase
from agentsurge.backends import get_backend
from agentsurge.backends.base import BackendConfig
from agentsurge.types import TurnResult


# ── Define a custom backend ──────────────────────────────────────
@dataclass
class EchoConfig(BackendConfig):
    """Configuration for the echo backend."""

    latency_ms: float = 2.0


class EchoBackend(BackendBase):
    """A minimal backend that echoes input stats as output."""

    name = "echo"

    def __init__(self, config: EchoConfig | None = None) -> None:
        super().__init__(config or EchoConfig())
        self.config: EchoConfig
        self.calls: list[dict] = []

    async def __aenter__(self) -> EchoBackend:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    async def send_turn(
        self,
        session_id: str,
        turn_index: int,
        messages: list[dict[str, Any]],
    ) -> TurnResult:
        t0 = time.monotonic()
        await asyncio.sleep(self.config.latency_ms / 1000)
        elapsed = (time.monotonic() - t0) * 1000

        self.calls.append(
            {
                "session_id": session_id,
                "turn_index": turn_index,
                "n_messages": len(messages),
            }
        )

        return TurnResult(
            session_id=session_id,
            turn_index=turn_index,
            ok=True,
            ttft_ms=elapsed,
            total_ms=elapsed,
            output_tokens=len(messages) * 10,
            input_messages=len(messages),
        )

    async def health_check(self) -> bool:
        return True


# ── Verify it's discoverable (auto-registered by __init_subclass__) ──
backend_cls = get_backend("echo")
assert backend_cls is EchoBackend, "registry should return EchoBackend"
print(f"Registered and discovered backend: {backend_cls.name}")


# ── Use it directly ──────────────────────────────────────────────
async def main() -> None:
    async with EchoBackend(EchoConfig(latency_ms=1.0)) as backend:
        result = await backend.send_turn(
            "s1",
            0,
            [{"role": "user", "content": "hello"}],
        )
        assert result.ok
        assert result.output_tokens == 10
        print(
            f"Echo turn: ok={result.ok}, ttft={result.ttft_ms:.1f}ms, tokens={result.output_tokens}"
        )

        # Multi-turn usage
        for i in range(1, 4):
            r = await backend.send_turn(
                "s1",
                i,
                [{"role": "user", "content": f"turn {i}"}],
            )
            assert r.ok

        assert len(backend.calls) == 4
        print(f"Completed {len(backend.calls)} turns via EchoBackend")


asyncio.run(main())
print("\n✓ custom-backend example passed")
