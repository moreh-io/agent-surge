"""H4: Task.reference must be reachable by name in metadata.

Currently ``Task.reference`` is populated by every task loader but
``TaskToTrajectoryConverter`` only forwards ``task.metadata`` downstream,
so the gold answer is unreadable to anything other than the loader test.
The remediation is to mirror ``reference`` into ``metadata["reference"]``
so a future scoring path can fetch it without re-running the loader.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def _hf_mock(samples):
    m = MagicMock()
    m.load_dataset.return_value = iter(samples)
    return m


def test_swe_bench_reference_in_metadata():
    from agentsurge.loaders.swe_bench import SWEBenchLoader

    sample = {
        "instance_id": "django__django-1",
        "problem_statement": "p",
        "patch": "diff --git a/x b/x",
    }
    with patch.dict("sys.modules", {"datasets": _hf_mock([sample])}):
        tasks = list(SWEBenchLoader().load())
    assert tasks[0].metadata.get("reference") == "diff --git a/x b/x"


def test_abc_bench_reference_in_metadata():
    from agentsurge.loaders.abc_bench import ABCBenchLoader

    sample = {
        "task_id": "abc-1",
        "instruction": "do thing",
        "output": "expected",
    }
    with patch.dict("sys.modules", {"datasets": _hf_mock([sample])}):
        tasks = list(ABCBenchLoader().load())
    assert tasks[0].metadata.get("reference") == "expected"
