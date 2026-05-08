# SPDX-License-Identifier: MIT
"""[audit:tool#H4] TTFT and r.tokens semantics for tool-call SSE deltas.

The chat-branch increments ``r.tokens`` once per chunk that includes any
tool-call delta. A chunk that delivers fragments for *several* tool calls
(``delta.tool_calls=[a, b, c]``) only adds 1 to ``r.tokens``. We pin two
properties:

1. TTFT must fire on the first tool-call delta even when ``content`` is empty
   (intentional: a tool-only stream is still meaningful first-byte payload).
2. ``r.tokens`` must count fan-out across tool calls inside one chunk so
   that a 3-tool-call delta contributes 3 token-increments, not 1.
"""

import json

import pytest

from agentsurge.backends._sse import parse_sse
from tests.conftest import MockAsyncLineIterator


def _sse_line(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n".encode()


@pytest.mark.asyncio
async def test_ttft_fires_on_tool_only_delta():
    chunk = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_a",
                            "type": "function",
                            "function": {"name": "bash", "arguments": "{}"},
                        }
                    ]
                }
            }
        ]
    }
    lines = [_sse_line(chunk), b"data: [DONE]\n"]
    r = await parse_sse(MockAsyncLineIterator(lines), api_type="chat")
    assert r.ttft is not None, "TTFT must fire on the first tool-call delta"
    assert r.ttft >= 0


@pytest.mark.asyncio
async def test_tokens_count_tool_call_fanout_within_chunk():
    """One chunk delivering 3 tool-call deltas must add 3 to r.tokens.

    Today the code does `r.tokens += 1` regardless of `len(delta.tool_calls)`,
    which under-counts streamed payload for parallel-tool-call models.
    """
    chunk = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_a",
                            "type": "function",
                            "function": {"name": "think", "arguments": "{}"},
                        },
                        {
                            "index": 1,
                            "id": "call_b",
                            "type": "function",
                            "function": {"name": "bash", "arguments": "{}"},
                        },
                        {
                            "index": 2,
                            "id": "call_c",
                            "type": "function",
                            "function": {"name": "finish", "arguments": "{}"},
                        },
                    ]
                }
            }
        ]
    }
    lines = [_sse_line(chunk), b"data: [DONE]\n"]
    r = await parse_sse(MockAsyncLineIterator(lines), api_type="chat")
    assert r.tool_calls is not None and len(r.tool_calls) == 3
    assert r.tokens >= 3, (
        f"r.tokens must count tool-call fan-out within a chunk: got {r.tokens}, "
        f"expected >= 3 for a 3-tool-call delta"
    )
