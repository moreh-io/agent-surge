# SPDX-License-Identifier: MIT
"""[audit:tool#H1] SSE tool_call id/type must be merged from later deltas.

Some servers split the tool-call header across deltas: the first delta carries
only ``index`` + partial ``function.name``; the ``id`` (and ``type``) arrive in
a later delta. The reassembler must accept the late-arriving id/type rather
than fixing them to the first delta's values.
"""

import json

import pytest

from agentsurge.backends._sse import parse_sse
from tests.conftest import MockAsyncLineIterator


def _sse_line(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n".encode()


@pytest.mark.asyncio
async def test_tool_call_id_arrives_in_later_delta():
    # First delta: index + function name only, no id/type. (Some vendors emit
    # the call header in fragments.)
    chunk1 = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {"index": 0, "function": {"name": "bash"}},
                    ]
                }
            }
        ]
    }
    # Later delta: id + type carried for the same index.
    chunk2 = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_late_42",
                            "type": "function",
                        },
                    ]
                }
            }
        ]
    }
    chunk3 = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {"index": 0, "function": {"arguments": '{"command": "ls"}'}},
                    ]
                }
            }
        ]
    }

    lines = [
        _sse_line(chunk1),
        _sse_line(chunk2),
        _sse_line(chunk3),
        b"data: [DONE]\n",
    ]
    r = await parse_sse(MockAsyncLineIterator(lines), api_type="chat")

    assert r.tool_calls is not None
    assert len(r.tool_calls) == 1
    tc = r.tool_calls[0]
    assert tc["id"] == "call_late_42", f"id should be merged from later delta, got {tc['id']!r}"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "bash"
    assert tc["function"]["arguments"] == '{"command": "ls"}'


@pytest.mark.asyncio
async def test_tool_call_type_only_in_later_delta():
    # First delta: id+name only, no type (first-delta default would be "function"
    # which happens to match — guard against a vendor sending non-default type later).
    chunk1 = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_x",
                            "function": {"name": "search_code"},
                        }
                    ]
                }
            }
        ]
    }
    # Later delta carries an explicit type.
    chunk2 = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {"index": 0, "type": "function", "function": {"arguments": "{}"}}
                    ]
                }
            }
        ]
    }
    lines = [_sse_line(chunk1), _sse_line(chunk2), b"data: [DONE]\n"]
    r = await parse_sse(MockAsyncLineIterator(lines), api_type="chat")
    assert r.tool_calls is not None
    tc = r.tool_calls[0]
    assert tc["id"] == "call_x"
    assert tc["type"] == "function"
