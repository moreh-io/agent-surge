# SPDX-License-Identifier: MIT
"""[audit:misc-core#H3] _count_input_tokens must not zero multipart / tool_calls.

OpenAI multipart message content is a list[dict] (each part has a
``type`` and a ``text``). Assistant turns can also carry only
``tool_calls`` and a None-ish content. The previous implementation did
``len(m.get("content") or "")`` which returned 0 for both cases,
silently undercounting input tokens deep into the conversation.
"""

from __future__ import annotations

from agentsurge.runner import _count_input_tokens


def test_multipart_content_counted() -> None:
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello world from multipart"},
                {"type": "text", "text": "another segment of text here"},
            ],
        },
    ]
    n = _count_input_tokens(msgs, tokenizer=None)
    assert n > 0, f"multipart content was zeroed: got {n} tokens"


def test_tool_calls_counted() -> None:
    msgs = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "do_something",
                        "arguments": '{"path": "/tmp/example", "limit": 100}',
                    },
                }
            ],
        },
    ]
    n = _count_input_tokens(msgs, tokenizer=None)
    assert n > 0, f"tool_calls turn was zeroed: got {n} tokens"


def test_plain_string_content_still_works() -> None:
    msgs = [{"role": "user", "content": "hello world"}]
    n = _count_input_tokens(msgs, tokenizer=None)
    assert n == len("hello world") // 4
