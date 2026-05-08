"""H2: vLLM responses-API path must reject tools= rather than silently drop them.

The responses-API SSE parser has no handler for `response.function_call_arguments.delta`
or any other responses-API tool-call event types, so passing tools= would silently
produce TurnResult.tool_calls=None even when the model emitted tool calls. Raise
NotImplementedError instead so callers fail loudly.
"""

from __future__ import annotations

import asyncio

import pytest

from agentsurge.backends.vllm import VllmBackend, VllmConfig


def test_responses_tools_raises_not_implemented():
    cfg = VllmConfig(base_url="http://test:8000", model="m", api_type="responses")
    backend = VllmBackend(cfg)
    backend._http = object()  # bypass _ensure_http; we expect to fail before HTTP

    tools = [{"type": "function", "function": {"name": "finish", "parameters": {}}}]

    async def _run():
        await backend.send_turn(
            "s1",
            0,
            [{"role": "user", "content": "hi"}],
            tools=tools,
        )

    with pytest.raises(NotImplementedError, match="tools.*responses"):
        asyncio.run(_run())
