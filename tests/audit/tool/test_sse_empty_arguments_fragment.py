# SPDX-License-Identifier: MIT
"""[audit:tool#H2] Empty ``arguments`` fragments must not be dropped.

Some compatible servers send ``{"function": {"arguments": ""}}`` as the only
arguments fragment for a no-arg tool call. The reassembler used a falsy guard
(``if fn.get("arguments")``), which dropped the empty fragment and left the
final ``arguments`` literally empty — which then fails ``json.loads("")``.

Either accept empty fragments verbatim or finalize empty arguments to ``"{}"``.
"""

import json

import pytest

from agentsurge.backends._sse import parse_sse
from tests.conftest import MockAsyncLineIterator


def _sse_line(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n".encode()


@pytest.mark.asyncio
async def test_empty_arguments_fragment_yields_parseable_json():
    # Single delta carrying a no-arg tool call: name=finish, arguments="".
    # The reassembled tool_call must round-trip through validate_tool_calls
    # without a JSON parse error - the downstream contract that motivates
    # this fix is `json.loads(arguments)` succeeding.
    from agentsurge.tool_call import validate_tool_calls

    chunk = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_finish",
                            "type": "function",
                            "function": {"name": "finish", "arguments": ""},
                        }
                    ]
                }
            }
        ]
    }
    lines = [_sse_line(chunk), b"data: [DONE]\n"]
    r = await parse_sse(MockAsyncLineIterator(lines), api_type="chat")
    assert r.tool_calls is not None and len(r.tool_calls) == 1

    # Reconstitute a chat-completions-shape response with the assembled
    # tool_calls and run it through the validator the runner uses.
    response = {
        "choices": [
            {
                "message": {"role": "assistant", "content": None, "tool_calls": r.tool_calls},
                "finish_reason": "tool_calls",
            }
        ]
    }
    v = validate_tool_calls(response, allowed_tool_names={"finish"})
    assert v.has_tool_calls
    assert v.valid, f"validation failed: {v.errors} / {[tc.errors for tc in v.tool_calls]}"
    assert v.tool_calls[0].arguments_parsed == {}


@pytest.mark.asyncio
async def test_split_empty_then_nonempty_arguments_concatenate():
    # Vendor splits arguments across deltas: first fragment is "" (a no-op
    # placeholder), then a real chunk arrives. The "" must not abort the merge.
    delta1 = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_split",
                            "type": "function",
                            "function": {"name": "bash", "arguments": ""},
                        }
                    ]
                }
            }
        ]
    }
    delta2 = {
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
    lines = [_sse_line(delta1), _sse_line(delta2), b"data: [DONE]\n"]
    r = await parse_sse(MockAsyncLineIterator(lines), api_type="chat")
    assert r.tool_calls is not None and len(r.tool_calls) == 1
    parsed = json.loads(r.tool_calls[0]["function"]["arguments"])
    assert parsed == {"command": "ls"}
