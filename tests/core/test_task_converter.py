"""Tests for TaskToTrajectoryConverter - task to synthetic trajectory conversion."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agentsurge.generators.task_converter import (
    _INTENT_TOOL_TYPE,
    TaskToTrajectoryConverter,
    _extract_hints,
    _fallback_body,
    _resolve_intent,
)
from agentsurge.types import Task, Trajectory

# ===========================================================================
# Helpers
# ===========================================================================


def _make_task(
    task_id: str = "repo_bug-123",
    prompt: str = "Fix the null pointer in src/main.py line 42.\ndef foo():\n    return None\n"
    * 10,
    metadata: dict | None = None,
) -> Task:
    return Task(
        task_id=task_id,
        prompt=prompt,
        metadata=metadata if metadata is not None else {"repo": "owner/myrepo"},
    )


# ===========================================================================
# _extract_hints
# ===========================================================================


class TestExtractHints:
    def test_extracts_file_from_prompt(self):
        task = _make_task(prompt="Look at src/main.py and fix it")
        hints = _extract_hints(task)
        assert hints["file_hint"] == "src/main.py"

    def test_extracts_repo_from_metadata(self):
        task = _make_task(metadata={"repo": "org/cool-repo"})
        hints = _extract_hints(task)
        assert hints["repo"] == "cool-repo"

    @pytest.mark.parametrize(
        ("task_kwargs", "key", "expected"),
        [
            ({"prompt": "Fix the bug"}, "file_hint", "src/main.py"),
            ({"prompt": ""}, "file_hint", "src/main.py"),
            ({"task_id": "myproject_issue-1", "metadata": {}}, "repo", "myproject"),
            ({"task_id": "standalone", "metadata": {}}, "repo", "project"),
        ],
    )
    def test_fallbacks(self, task_kwargs, key, expected):
        task = _make_task(**task_kwargs)
        assert _extract_hints(task)[key] == expected


# ===========================================================================
# _fallback_body
# ===========================================================================


class TestFallbackBody:
    def test_returns_nonempty(self):
        task = _make_task()
        body = _fallback_body(task, turn_idx=0)
        assert len(body) > 0
        assert "\n" in body
        assert "/" in body.split("\n", 1)[0]

    def test_includes_repo_and_file(self):
        task = _make_task(metadata={"repo": "owner/testrepo"})
        body = _fallback_body(task, turn_idx=0)
        assert "testrepo" in body

    def test_deterministic(self):
        task = _make_task()
        b1 = _fallback_body(task, turn_idx=0)
        b2 = _fallback_body(task, turn_idx=0)
        assert b1 == b2

    def test_different_turns_different_content(self):
        task = _make_task(prompt="x" * 8000)
        b0 = _fallback_body(task, turn_idx=0)
        b1 = _fallback_body(task, turn_idx=1)
        assert isinstance(b0, str) and isinstance(b1, str)
        assert b0 != b1
        b5 = _fallback_body(task, turn_idx=5)
        assert len(b0) > 0 and len(b5) > 0

    def test_short_prompt(self):
        task = _make_task(prompt="short")
        body = _fallback_body(task, turn_idx=0)
        assert "short" in body


# ===========================================================================
# _resolve_intent
# ===========================================================================


class TestResolveIntent:
    def test_known_intent_produces_tool_output(self):
        task = _make_task()
        text = _resolve_intent("read_code", task, turn_idx=0, bank=None)
        assert text.startswith("[Tool output: read_file]")

    def test_unknown_intent_uses_generic_template(self):
        task = _make_task()
        text = _resolve_intent("unknown_action", task, turn_idx=0, bank=None)
        assert text.startswith("[Tool output]")

    def test_deterministic(self):
        task = _make_task()
        t1 = _resolve_intent("read_code", task, turn_idx=0, bank=None)
        t2 = _resolve_intent("read_code", task, turn_idx=0, bank=None)
        assert t1 == t2

    def test_all_intents_resolve(self):
        task = _make_task()
        for intent in _INTENT_TOOL_TYPE:
            text = _resolve_intent(intent, task, turn_idx=0, bank=None)
            assert len(text) > 0


# ===========================================================================
# TaskToTrajectoryConverter.__init__
# ===========================================================================


class TestConverterInit:
    def test_custom_params_affect_output(self):
        """Custom max_prompt_chars truncates prompt; default does not."""
        task = _make_task(prompt="X" * 20000)
        conv_default = TaskToTrajectoryConverter()
        conv_short = TaskToTrajectoryConverter(max_prompt_chars=500)
        t_default = conv_default._build_trajectory(task, "bug-fix")
        t_short = conv_short._build_trajectory(task, "bug-fix")
        default_first = t_default.sessions[0].turns[0].content
        short_first = t_short.sessions[0].turns[0].content
        assert len(short_first) < len(default_first)


# ===========================================================================
# _chunk_prompt
# ===========================================================================


class TestChunkPrompt:
    def test_truncates_to_max_prompt_chars(self):
        conv = TaskToTrajectoryConverter(max_prompt_chars=10)
        chunks = conv._chunk_prompt("a" * 100, 1)
        assert len(chunks[0]) == 10

    def test_zero_chunks_returns_single(self):
        conv = TaskToTrajectoryConverter()
        # n_chunks <= 1 returns [truncated]
        chunks = conv._chunk_prompt("text", 0)
        assert chunks == ["text"]

    def test_chunks_cover_full_text(self):
        conv = TaskToTrajectoryConverter(max_prompt_chars=100)
        text = "x" * 100
        for n in [2, 3, 5]:
            chunks = conv._chunk_prompt(text, n)
            assert len(chunks) == n
            assert "".join(chunks) == text


# ===========================================================================
# from_tasks
# ===========================================================================


class TestFromTasks:
    def test_basic_conversion(self):
        conv = TaskToTrajectoryConverter()
        tasks = [_make_task(task_id=f"task-{i}") for i in range(3)]
        results = conv.from_tasks(iter(tasks), pattern="bug-fix")
        assert len(results) == 3
        assert all(isinstance(r, Trajectory) for r in results)

    def test_limit(self):
        conv = TaskToTrajectoryConverter()
        tasks = [_make_task(task_id=f"task-{i}") for i in range(10)]
        results = conv.from_tasks(iter(tasks), pattern="bug-fix", limit=3)
        assert len(results) == 3

    def test_all_patterns(self):
        conv = TaskToTrajectoryConverter()
        for pattern in TaskToTrajectoryConverter.PATTERNS:
            task = _make_task(task_id=f"task-{pattern}")
            results = conv.from_tasks(iter([task]), pattern=pattern)
            assert len(results) == 1
            traj = results[0]
            assert len(traj.sessions) == 1
            assert traj.sessions[0].n_turns > 0

    def test_metadata_preserved(self):
        conv = TaskToTrajectoryConverter()
        task = _make_task(task_id="t1", metadata={"repo": "org/repo", "extra": "val"})
        results = conv.from_tasks(iter([task]), pattern="bug-fix")
        traj = results[0]
        assert traj.metadata["task_id"] == "t1"
        assert traj.metadata["source"] == "bug-fix"


# ===========================================================================
# _build_trajectory
# ===========================================================================


class TestBuildTrajectory:
    def test_deterministic(self):
        conv = TaskToTrajectoryConverter()
        task = _make_task(task_id="t1")
        t1 = conv._build_trajectory(task, "bug-fix")
        t2 = conv._build_trajectory(task, "bug-fix")
        assert len(t1.sessions[0].turns) == len(t2.sessions[0].turns)
        for a, b in zip(t1.sessions[0].turns, t2.sessions[0].turns, strict=False):
            assert a.role == b.role
            assert a.content == b.content

    def test_session_id_format(self):
        conv = TaskToTrajectoryConverter()
        task = _make_task(task_id="myid")
        traj = conv._build_trajectory(task, "bug-fix")
        assert traj.sessions[0].session_id == "myid_0"

    def test_trajectory_id(self):
        conv = TaskToTrajectoryConverter()
        task = _make_task(task_id="myid")
        traj = conv._build_trajectory(task, "bug-fix")
        assert traj.trajectory_id == "myid"

    def test_turns_alternate_roles(self):
        conv = TaskToTrajectoryConverter()
        task = _make_task()
        traj = conv._build_trajectory(task, "bug-fix")
        turns = traj.sessions[0].turns
        for i, turn in enumerate(turns):
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert turn.role == expected_role

    def test_intent_markers_resolved(self):
        conv = TaskToTrajectoryConverter()
        task = _make_task()
        traj = conv._build_trajectory(task, "bug-fix")
        turns = traj.sessions[0].turns
        for turn in turns:
            assert not turn.content.startswith("@"), f"Unresolved intent: {turn.content[:50]}"

    def test_chunks_substituted(self):
        conv = TaskToTrajectoryConverter()
        task = _make_task(prompt="UNIQUE_MARKER " * 100)
        traj = conv._build_trajectory(task, "bug-fix")
        first_turn = traj.sessions[0].turns[0]
        assert "UNIQUE_MARKER" in first_turn.content

    def test_assistant_responses_padded(self):
        conv = TaskToTrajectoryConverter(min_assistant_tokens=200)
        task = _make_task()
        traj = conv._build_trajectory(task, "bug-fix")
        for turn in traj.sessions[0].turns:
            if turn.role == "assistant":
                # Padded responses should be longer than the template stubs
                assert len(turn.content) > 50


# ===========================================================================
# Retry patterns
# ===========================================================================


class TestRetryPatterns:
    def test_retry_probability_selects_variant(self):
        """Over many tasks, ~60% should get retry patterns (more turns)."""
        conv = TaskToTrajectoryConverter()
        base_turns = len(TaskToTrajectoryConverter.PATTERNS["bug-fix"])
        retry_turns = len(TaskToTrajectoryConverter.RETRY_PATTERNS["bug-fix-retry"])
        assert retry_turns > base_turns  # retry has more turns

        retry_count = 0
        n = 100
        for i in range(n):
            task = _make_task(task_id=f"task-{i}")
            traj = conv._build_trajectory(task, "bug-fix")
            n_turns = traj.sessions[0].n_turns
            if n_turns == retry_turns:
                retry_count += 1

        # Expect ~60% retry, allow wide margin (40-80%)
        assert 30 < retry_count < 85, f"retry_count={retry_count}/100"

    def test_all_retry_patterns_have_base(self):
        for key in TaskToTrajectoryConverter.RETRY_PATTERNS:
            base_key = key.replace("-retry", "")
            assert base_key in TaskToTrajectoryConverter.PATTERNS


# ===========================================================================
# Dataset pattern map
# ===========================================================================


class TestDatasetPatternMap:
    def test_known_datasets(self):
        for _dataset, pattern in TaskToTrajectoryConverter.DATASET_PATTERN_MAP.items():
            assert pattern in TaskToTrajectoryConverter.PATTERNS

    def test_swe_bench_maps_to_bug_fix(self):
        assert TaskToTrajectoryConverter.DATASET_PATTERN_MAP["swe-bench"] == "bug-fix"


# ===========================================================================
# TracePool integration
# ===========================================================================


class TestTracePoolIntegration:
    def test_without_trace_pool_uses_fallback(self):
        conv = TaskToTrajectoryConverter(trace_pool=None)
        task = _make_task()
        traj = conv._build_trajectory(task, "bug-fix")
        # Should still produce valid trajectory
        assert traj.sessions[0].n_turns > 0
        for turn in traj.sessions[0].turns:
            assert len(turn.content) > 0

    def test_trace_pool_sample_assistant_returns_none(self):
        """When sample_assistant returns None, should fall back to padding."""
        mock_pool = MagicMock()
        mock_pool.sample.return_value = "trace content"
        mock_pool.sample_assistant.return_value = None

        conv = TaskToTrajectoryConverter(trace_pool=mock_pool)
        task = _make_task()
        traj = conv._build_trajectory(task, "bug-fix")
        for turn in traj.sessions[0].turns:
            if turn.role == "assistant":
                assert len(turn.content) > 0
