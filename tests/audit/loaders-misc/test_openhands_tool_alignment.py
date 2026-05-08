"""H2: OpenHandsLoader must align tool results with prior tool_calls.

OpenHands trajectories occasionally omit ``tool_call_id`` on the
``role:"tool"`` reply (or ship it under ``name``).  Without FIFO/by-id
matching, downstream replay produces messages an OpenAI-compatible chat
endpoint will reject because tool replies do not reference any preceding
``tool_call_id``.
"""

from __future__ import annotations

import json


def test_tool_result_inherits_id_when_missing(tmp_path):
    from agentsurge.loaders.openhands import OpenHandsLoader

    sample = {
        "instance_id": "oh-align-001",
        "messages": [
            {"role": "user", "content": "fix"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_x",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            # tool_call_id deliberately missing on the reply
            {"role": "tool", "content": "file text"},
        ],
    }
    jsonl_file = tmp_path / "data.jsonl"
    jsonl_file.write_text(json.dumps(sample) + "\n")

    turns = list(OpenHandsLoader(local_path=str(jsonl_file)).load())[0].sessions[0].turns

    tool_turn = turns[2]
    assert tool_turn.role == "tool"
    assert tool_turn.tool_call_id == "call_x"
    assert tool_turn.name == "read_file"


def test_tool_result_explicit_id_matches_before_fifo(tmp_path):
    from agentsurge.loaders.openhands import OpenHandsLoader

    sample = {
        "instance_id": "oh-align-002",
        "messages": [
            {"role": "user", "content": "fix"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    },
                    {
                        "id": "call_b",
                        "type": "function",
                        "function": {"name": "execute_bash", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_b", "content": "pytest ok"},
            {"role": "tool", "tool_call_id": "call_a", "content": "file text"},
        ],
    }
    jsonl_file = tmp_path / "data.jsonl"
    jsonl_file.write_text(json.dumps(sample) + "\n")

    turns = list(OpenHandsLoader(local_path=str(jsonl_file)).load())[0].sessions[0].turns

    assert turns[2].tool_call_id == "call_b"
    assert turns[2].name == "execute_bash"
    assert turns[3].tool_call_id == "call_a"
    assert turns[3].name == "read_file"
