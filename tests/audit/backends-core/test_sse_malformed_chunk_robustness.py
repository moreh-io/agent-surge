"""H4: parse_sse must not abort the entire turn on a single rogue chunk.

Structural failures (KeyError/TypeError/AttributeError) on a single SSE chunk
should bump parse_errors and continue parsing the remainder of the stream
instead of escaping out of parse_sse.
"""

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


def test_chat_malformed_chunk_with_null_choices_does_not_abort():
    """choices=[null] is JSON-valid but choices[0].get raises AttributeError.

    parse_sse must catch this, bump parse_errors, and keep parsing.
    """
    good = {"choices": [{"delta": {"content": "ok"}}]}
    bad = {"choices": [None]}  # AttributeError on .get inside parser
    final = {"choices": [{"delta": {"content": "done"}}], "usage": {"completion_tokens": 7}}
    lines = [
        b"data: " + json.dumps(good).encode() + b"\n",
        b"data: " + json.dumps(bad).encode() + b"\n",
        b"data: " + json.dumps(final).encode() + b"\n",
        b"data: [DONE]\n",
    ]

    async def _run():
        return await parse_sse(_MockLineIterator(lines), api_type="chat")

    r = asyncio.run(_run())
    assert r.parse_errors >= 1
    assert r.usage_tokens == 7  # final usage chunk still seen
    assert r.tokens >= 2  # both well-formed content deltas counted


def test_responses_malformed_completed_does_not_abort():
    """A response.completed event whose `response` is None must not abort.

    The parser must bump parse_errors and continue.
    """
    delta = {"type": "response.output_text.delta", "delta": "x"}
    bad_completed = {"type": "response.completed", "response": None}
    lines = [
        b"data: " + json.dumps(delta).encode() + b"\n",
        b"data: " + json.dumps(bad_completed).encode() + b"\n",
    ]

    async def _run():
        return await parse_sse(_MockLineIterator(lines), api_type="responses")

    r = asyncio.run(_run())
    assert r.parse_errors >= 1
    assert r.tokens == 1
