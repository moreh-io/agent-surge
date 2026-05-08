"""M10: MockSSEStream must be able to exercise tool-call and reasoning_content branches.

Until now MockSSEStream emitted only delta.content chunks, so the bug-prone
tool-call accumulation and reasoning_content paths in parse_sse were unreachable
through the mock. Add explicit modes to drive each branch.
"""

from __future__ import annotations

import asyncio

from agentsurge.backends._sse import parse_sse
from agentsurge.backends.mock._sse import MockSSEStream


def test_mock_sse_tool_calls_mode_drives_parser_tool_call_branch():
    stream = MockSSEStream(
        output_tokens=2,
        ttft_ms=0.0,
        inter_token_ms=0.0,
        emit_tool_calls=[
            {
                "id": "call_1",
                "function": {"name": "my_func", "arguments": '{"x": 1}'},
            }
        ],
    )

    async def _run():
        return await parse_sse(stream, api_type="chat")

    r = asyncio.run(_run())
    assert r.tool_calls is not None
    assert len(r.tool_calls) == 1
    assert r.tool_calls[0]["function"]["name"] == "my_func"
    assert r.tool_calls[0]["function"]["arguments"] == '{"x": 1}'


def test_mock_sse_reasoning_mode_drives_parser_reasoning_branch():
    stream = MockSSEStream(
        output_tokens=3,
        ttft_ms=0.0,
        inter_token_ms=0.0,
        emit_reasoning=True,
    )

    async def _run():
        return await parse_sse(stream, api_type="chat")

    r = asyncio.run(_run())
    assert r.reasoning_tokens == 3
