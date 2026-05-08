"""H3: SWESmithLoader must stamp supports_multi_turn on session+trajectory.

``TraceReplayGenerator._build_replay`` reads ``supports_multi_turn`` from
session OR trajectory metadata.  Without the flag, follow-up user turns
are silently dropped.  The flag must reflect the loader class attribute
so a future flip (False -> True) is honoured automatically.
"""

from __future__ import annotations

import json


def test_supports_multi_turn_propagated_to_session_and_trajectory(tmp_path):
    from agentsurge.loaders.swe_smith import SWESmithLoader

    sample = {
        "instance_id": "smith-mt",
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
        ],
    }
    jsonl_file = tmp_path / "data.jsonl"
    jsonl_file.write_text(json.dumps(sample) + "\n")

    traj = list(SWESmithLoader(local_path=str(jsonl_file)).load())[0]

    expected = SWESmithLoader.supports_multi_turn  # default class attr (False)
    assert traj.metadata["supports_multi_turn"] is expected
    assert traj.sessions[0].metadata["supports_multi_turn"] is expected


def test_supports_multi_turn_reflects_subclass_override(tmp_path):
    """If a subclass sets supports_multi_turn=True, the flag must follow."""
    from agentsurge.loaders.swe_smith import SWESmithLoader

    class MultiTurnSmith(SWESmithLoader):
        name = "swe-smith-mt-test"
        supports_multi_turn = True

    sample = {
        "instance_id": "smith-mt-2",
        "messages": [{"role": "user", "content": "hi"}],
    }
    jsonl_file = tmp_path / "data.jsonl"
    jsonl_file.write_text(json.dumps(sample) + "\n")

    traj = list(MultiTurnSmith(local_path=str(jsonl_file)).load())[0]
    assert traj.metadata["supports_multi_turn"] is True
    assert traj.sessions[0].metadata["supports_multi_turn"] is True
