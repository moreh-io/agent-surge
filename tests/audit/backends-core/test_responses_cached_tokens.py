"""H1: Responses-API SSE parser must surface cached_tokens from input_tokens_details."""

from __future__ import annotations

import asyncio
import json

from agentsurge.backends._sse import parse_sse


class _MockLineIterator:
    def __init__(self, lines: list[bytes]):
        self._lines = lines
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        if self._idx >= len(self._lines):
            raise StopAsyncIteration
        line = self._lines[self._idx]
        self._idx += 1
        return line


def test_responses_api_cached_tokens_extracted():
    """parse_sse(api_type='responses') must read usage.input_tokens_details.cached_tokens."""
    completed = {
        "type": "response.completed",
        "response": {
            "usage": {
                "input_tokens": 100,
                "output_tokens": 5,
                "input_tokens_details": {"cached_tokens": 42},
            }
        },
    }
    lines = [
        b"event: response.completed\n",
        b"data: " + json.dumps(completed).encode() + b"\n",
        b"\n",
    ]

    async def _run():
        return await parse_sse(_MockLineIterator(lines), api_type="responses")

    r = asyncio.run(_run())
    assert r.cached_tokens == 42
