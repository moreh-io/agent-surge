"""M9: ABCBenchLoader must surface the unused prompt key when both differ.

Both ``instruction`` and ``prompt`` are excluded from metadata, but only
one is mapped to ``Task.prompt``.  When a sample has both with different
content (legacy + new schema), the unused one is silently dropped.
The fix stores the alternate under ``metadata["alt_prompt"]``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def _hf_mock(samples):
    m = MagicMock()
    m.load_dataset.return_value = iter(samples)
    return m


def test_alt_prompt_preserved_when_both_keys_differ():
    from agentsurge.loaders.abc_bench import ABCBenchLoader

    sample = {
        "task_id": "abc-dual",
        "instruction": "primary instruction text",
        "prompt": "different prompt text",
        "output": "expected",
    }
    with patch.dict("sys.modules", {"datasets": _hf_mock([sample])}):
        tasks = list(ABCBenchLoader().load())

    task = tasks[0]
    # Primary mapping unchanged.
    assert task.prompt == "primary instruction text"
    # Alternate is reachable via metadata so it's not silently lost.
    assert task.metadata.get("alt_prompt") == "different prompt text"


def test_alt_prompt_absent_when_only_one_key_present():
    from agentsurge.loaders.abc_bench import ABCBenchLoader

    sample = {
        "task_id": "abc-single",
        "instruction": "only instruction",
        "output": "expected",
    }
    with patch.dict("sys.modules", {"datasets": _hf_mock([sample])}):
        tasks = list(ABCBenchLoader().load())
    assert "alt_prompt" not in tasks[0].metadata


def test_alt_prompt_not_recorded_when_keys_match():
    from agentsurge.loaders.abc_bench import ABCBenchLoader

    sample = {
        "task_id": "abc-same",
        "instruction": "same text",
        "prompt": "same text",
    }
    with patch.dict("sys.modules", {"datasets": _hf_mock([sample])}):
        tasks = list(ABCBenchLoader().load())
    assert "alt_prompt" not in tasks[0].metadata
