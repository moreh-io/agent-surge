# SPDX-License-Identifier: MIT
"""Mock backend adapter for testing and CI.

This subpackage provides an in-process mock implementation of
:class:`~agentsurge.backends.base.BackendBase` that requires no network access
and returns configurable fake responses.

Features:

* **Synthetic SSE streaming** - generates OpenAI-compatible SSE chunks with
  realistic async timing, parseable by the same SSE parser used by the real
  vLLM backend.
* **KV-cache simulation** - tracks block-level allocation, eviction, and
  usage metrics to simulate GPU memory pressure.

Quick start::

    from agentsurge.backends.mock import MockBackend, MockConfig

    cfg = MockConfig(output_tokens=42, ttft_ms=5.0)
    async with MockBackend(cfg) as backend:
        result = await backend.send_turn("s1", 0, [{"role": "user", "content": "hi"}])

SSE mode::

    cfg = MockConfig(output_tokens=20, ttft_ms=50.0, simulate_sse=True)
    async with MockBackend(cfg) as backend:
        result = await backend.send_turn("s1", 0, [{"role": "user", "content": "hi"}])

KV simulation::

    from agentsurge.backends.mock import MockBackend, MockConfig, KVSimConfig

    cfg = MockConfig(output_tokens=10, kv_sim=KVSimConfig(total_blocks=100))
    async with MockBackend(cfg) as backend:
        result = await backend.send_turn("s1", 0, [{"role": "user", "content": "hi"}])
        snap = backend.kv_snapshot()
"""

from agentsurge.backends.mock._backend import MockBackend, MockConfig
from agentsurge.backends.mock._kv_sim import KVAllocation, KVSimConfig, KVSimulator
from agentsurge.backends.mock._sse import MockSSEStream

__all__ = [
    "MockBackend",
    "MockConfig",
    "MockSSEStream",
    "KVSimConfig",
    "KVSimulator",
    "KVAllocation",
]
