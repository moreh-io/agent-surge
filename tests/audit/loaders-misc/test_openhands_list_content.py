"""H1: OpenHandsLoader must extract text from list-shaped chat content blocks.

Anthropic-style and some OpenHands exports ship ``content`` as
``[{"type": "text", "text": "..."}]``.  ``str([...])`` would emit a Python
literal repr, silently corrupting downstream prompts/metrics.
"""

from __future__ import annotations

import json


def test_list_content_blocks_are_joined_to_text(tmp_path):
    from agentsurge.loaders.openhands import OpenHandsLoader

    sample = {
        "instance_id": "oh-list-001",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "text", "text": "world"},
                ],
            },
            {"role": "assistant", "content": "ok"},
        ],
    }
    jsonl_file = tmp_path / "data.jsonl"
    jsonl_file.write_text(json.dumps(sample) + "\n")

    trajs = list(OpenHandsLoader(local_path=str(jsonl_file)).load())
    user_turn = trajs[0].sessions[0].turns[0]

    assert "hello" in user_turn.content
    assert "world" in user_turn.content
    # No Python literal repr smuggled through.
    assert "{'type'" not in user_turn.content
    assert "[{" not in user_turn.content
