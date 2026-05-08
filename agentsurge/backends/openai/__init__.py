# SPDX-License-Identifier: MIT
"""OpenAI-compatible backend adapter.

This subpackage implements :class:`~agentsurge.backends.base.BackendBase` for
any OpenAI-compatible API server.  Unlike the vLLM backend, it does **not**
assume a ``/metrics`` Prometheus endpoint exists -- it only measures latency
from SSE streaming.

Quick start::

    from agentsurge.backends.openai import OpenAiBackend

    cfg = BackendConfig(base_url="http://localhost:8000", model="my-model")
    async with OpenAiBackend(cfg) as backend:
        result = await backend.send_turn("s1", 0, [{"role": "user", "content": "hi"}])
"""

from __future__ import annotations

from agentsurge.backends.openai._backend import OpenAiBackend

__all__ = ["OpenAiBackend"]
