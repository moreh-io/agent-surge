"""Trimmed workload tests: generators, replay, versioning, mixed workload."""

from __future__ import annotations

import json

import pytest

from agentsurge.types import WORKLOAD_SCHEMA_VERSION, ReplaySession

# ===========================================================================
# Helpers
# ===========================================================================


def _make_turn(role="user", content="hello", tool_calls=None, name=None):
    from agentsurge.loaders.base import Turn

    return Turn(role=role, content=content, tool_calls=tool_calls, name=name)


def _make_session(session_id="s1", turns=None):
    from agentsurge.loaders.base import Session

    if turns is None:
        turns = [
            _make_turn("user", "Fix the bug"),
            _make_turn(
                "assistant",
                "Looking.",
                tool_calls=[
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path": "f.py"}'},
                    }
                ],
            ),
            _make_turn("tool", "file content", name="read_file"),
            _make_turn("assistant", "Fixed."),
        ]
    return Session(session_id=session_id, turns=turns, metadata={})


def _make_trajectory(trajectory_id="t1", sessions=None):
    from agentsurge.loaders.base import Trajectory

    return Trajectory(trajectory_id=trajectory_id, sessions=sessions or [_make_session()])


# ===========================================================================
# flatten_tool_calls
# ===========================================================================


class TestFlattenToolCalls:
    def test_tool_role_becomes_user(self):
        from agentsurge.generators.trace_replay import flatten_tool_calls

        flat = flatten_tool_calls([_make_turn("tool", "result", name="read_file")])
        assert flat[0]["role"] == "user"

    def test_assistant_tool_calls_stripped(self):
        from agentsurge.generators.trace_replay import flatten_tool_calls

        flat = flatten_tool_calls([_make_turn("assistant", "Calling.", tool_calls=[{"id": "c1"}])])
        assert "tool_calls" not in flat[0]

    def test_no_tool_roles_after_flatten(self):
        from agentsurge.generators.trace_replay import flatten_tool_calls

        turns = [
            _make_turn("user", "Go"),
            _make_turn("assistant", "Reading.", tool_calls=[{"id": "c1"}]),
            _make_turn("tool", "content", name="read_file"),
            _make_turn("assistant", "Done."),
        ]
        flat = flatten_tool_calls(turns)
        assert all(m["role"] in ("user", "assistant", "system") for m in flat)


# ===========================================================================
# TraceReplayGenerator
# ===========================================================================


class TestTraceReplayGenerator:
    def test_generates_one_session_per_trajectory(self):
        from agentsurge.generators.trace_replay import TraceReplayGenerator

        gen = TraceReplayGenerator()
        sessions = list(
            gen.from_trajectories(iter([_make_trajectory("t1"), _make_trajectory("t2")]))
        )
        assert len(sessions) == 2

    def test_limit_caps_output_count(self):
        from agentsurge.generators.trace_replay import TraceReplayGenerator

        gen = TraceReplayGenerator()
        limited = list(
            gen.from_trajectories(iter([_make_trajectory(f"t{i}") for i in range(5)]), limit=2)
        )
        assert len(limited) == 2

    def test_each_turn_ends_with_user(self):
        from agentsurge.generators.trace_replay import TraceReplayGenerator

        sessions = list(TraceReplayGenerator().from_trajectories(iter([_make_trajectory()])))
        for turn_msgs in sessions[0].turn_messages:
            assert turn_msgs[-1]["role"] == "user"

    def test_flatten_true_removes_tool_calls(self):
        from agentsurge.generators.trace_replay import TraceReplayGenerator

        sessions = list(
            TraceReplayGenerator(flatten_tools=True).from_trajectories(iter([_make_trajectory()]))
        )
        for turn_msgs in sessions[0].turn_messages:
            for msg in turn_msgs:
                assert "tool_calls" not in msg

    def test_flatten_false_preserves_tool_calls(self):
        from agentsurge.generators.trace_replay import TraceReplayGenerator
        from agentsurge.loaders.base import Session, Trajectory

        sess = Session(
            session_id="s1",
            turns=[
                _make_turn("user", "Go"),
                _make_turn(
                    "assistant",
                    "Reading.",
                    tool_calls=[
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"},
                        }
                    ],
                ),
                _make_turn("tool", "data", name="read_file"),
                _make_turn("user", "Next"),
            ],
        )
        sessions = list(
            TraceReplayGenerator(flatten_tools=False).from_trajectories(
                iter([Trajectory(trajectory_id="t1", sessions=[sess])])
            )
        )
        all_msgs = sessions[0].turn_messages[-1]
        assert any("tool_calls" in m for m in all_msgs if m["role"] == "assistant")


# ===========================================================================
# SyntheticGenerator
# ===========================================================================


class TestSyntheticGenerator:
    def test_generate_returns_requested_session_count(self):
        from agentsurge.generators.synthetic import SyntheticGenerator

        gen = SyntheticGenerator(n_turns=3, tokens_per_turn=100)
        sessions = gen.generate(n_sessions=5)
        assert len(sessions) == 5
        for sess in sessions:
            assert sess.n_turns == 3

    def test_deterministic_with_seed(self):
        from agentsurge.generators.synthetic import SyntheticGenerator

        gen = SyntheticGenerator(n_turns=2, tokens_per_turn=100)
        s1 = gen.generate(n_sessions=3, seed=42)
        s2 = gen.generate(n_sessions=3, seed=42)
        assert len(s1) == len(s2)
        for a, b in zip(s1, s2, strict=True):
            assert a.session_id == b.session_id

    def test_zero_sessions(self):
        from agentsurge.generators.synthetic import SyntheticGenerator

        assert SyntheticGenerator().generate(n_sessions=0) == []


# ===========================================================================
# ReplaySession
# ===========================================================================


class TestReplaySession:
    def test_n_turns_property(self):
        sess = ReplaySession(
            session_id="s1",
            turn_messages=[
                [{"role": "user", "content": "Q1"}],
                [
                    {"role": "user", "content": "Q1"},
                    {"role": "assistant", "content": "A1"},
                    {"role": "user", "content": "Q2"},
                ],
            ],
        )
        assert sess.n_turns == 2

    def test_inject_replaces_assistant_content(self):
        session = ReplaySession(
            session_id="s1",
            turn_messages=[
                [{"role": "system", "content": "sys"}, {"role": "user", "content": "q1"}],
                [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "q1"},
                    {"role": "assistant", "content": "placeholder"},
                    {"role": "user", "content": "q2"},
                ],
            ],
        )
        session.inject_response(0, "actual LLM response")
        assistant_msgs = [m for m in session.turn_messages[1] if m["role"] == "assistant"]
        assert assistant_msgs[0]["content"] == "actual LLM response"

    def test_pending_user_messages_custom_value(self):
        sess = ReplaySession(
            session_id="s1",
            turn_messages=[[{"role": "user", "content": "hi"}]],
            pending_user_messages=["follow-up 1", "follow-up 2"],
        )
        assert sess.pending_user_messages == ["follow-up 1", "follow-up 2"]

    def test_pending_user_messages_roundtrip(self):
        original = ReplaySession(
            session_id="s1",
            turn_messages=[[{"role": "user", "content": "hi"}]],
            pending_user_messages=["u2", "u3"],
        )
        d = original.to_dict()
        assert d["pending_user_messages"] == ["u2", "u3"]
        restored = ReplaySession.from_dict(d)
        assert restored.pending_user_messages == ["u2", "u3"]

    def test_pending_user_messages_empty_not_serialized(self):
        """Empty list is omitted from to_dict to keep workload JSON compact."""
        sess = ReplaySession(session_id="s1", turn_messages=[[{"role": "user", "content": "hi"}]])
        d = sess.to_dict()
        assert "pending_user_messages" not in d

    def test_pending_user_messages_from_dict_missing(self):
        """from_dict handles workload JSON produced before this field existed."""
        d = {
            "session_id": "s1",
            "turn_messages": [[{"role": "user", "content": "hi"}]],
        }
        sess = ReplaySession.from_dict(d)
        assert sess.pending_user_messages == []


# ===========================================================================
# TaskToTrajectoryConverter
# ===========================================================================


class TestTaskToTrajectoryConverter:
    def _make_task(self, task_id="task-1"):
        from agentsurge.loaders.base import Task

        return Task(
            task_id=task_id, prompt="Fix the null pointer in views.py", metadata={"repo": "x/y"}
        )

    def test_bug_fix_pattern(self):
        from agentsurge.generators.task_converter import TaskToTrajectoryConverter

        trajs = TaskToTrajectoryConverter().from_tasks(iter([self._make_task()]), "bug-fix")
        assert len(trajs) == 1
        turns = trajs[0].sessions[0].turns
        # Pattern is 6 (linear) or 14 (retry) turns depending on task_id hash
        assert len(turns) in (6, 14)
        for i, t in enumerate(turns):
            assert t.role == ("user" if i % 2 == 0 else "assistant")

    def test_dataset_pattern_map(self):
        from agentsurge.generators.task_converter import TaskToTrajectoryConverter

        m = TaskToTrajectoryConverter.DATASET_PATTERN_MAP
        assert m["swe-bench"] == "bug-fix" and m["abc-bench"] == "api-impl"

    def test_limit_parameter(self):
        from agentsurge.generators.task_converter import TaskToTrajectoryConverter

        tasks = [self._make_task(task_id=f"t-{i}") for i in range(10)]
        assert len(TaskToTrajectoryConverter().from_tasks(iter(tasks), "bug-fix", limit=3)) == 3

    def test_user_messages_vary_across_tasks(self):
        """Non-chunk user turns should differ between tasks (intent resolution)."""
        from agentsurge.generators.task_converter import TaskToTrajectoryConverter
        from agentsurge.loaders.base import Task

        conv = TaskToTrajectoryConverter()
        t1 = Task(
            task_id="django-1",
            prompt="Fix null pointer in views.py",
            metadata={"repo": "django/django"},
        )
        t2 = Task(
            task_id="flask-2",
            prompt="TypeError in flask/app.py when rendering",
            metadata={"repo": "pallets/flask"},
        )
        turns1 = conv.from_tasks(iter([t1]), "bug-fix")[0].sessions[0].turns
        turns2 = conv.from_tasks(iter([t2]), "bug-fix")[0].sessions[0].turns
        # Turn 2 and 4 are intent-resolved - should differ between tasks
        assert turns1[2].content != turns2[2].content
        assert turns1[4].content != turns2[4].content
        # Intent turns should look like tool output
        assert "[Tool output" in turns1[2].content
        assert "[Tool output" in turns1[4].content

    def test_user_messages_deterministic(self):
        """Same task should produce identical user messages across runs."""
        from agentsurge.generators.task_converter import TaskToTrajectoryConverter

        conv = TaskToTrajectoryConverter()
        t = self._make_task()
        a = conv.from_tasks(iter([t]), "bug-fix")[0].sessions[0].turns
        b = conv.from_tasks(iter([t]), "bug-fix")[0].sessions[0].turns
        for i in range(len(a)):
            assert a[i].content == b[i].content


# ===========================================================================
# MixedWorkloadGenerator
# ===========================================================================


class TestMixedWorkloadGenerator:
    def _make_replay(self, sid="s1"):
        return ReplaySession(
            session_id=sid, turn_messages=[[{"role": "user", "content": "hi"}]], metadata={}
        )

    def test_proportional_weights(self):
        from agentsurge.generators.mixed import MixedWorkloadGenerator

        sources = {
            "a": [self._make_replay(f"a-{i}") for i in range(10)],
            "b": [self._make_replay(f"b-{i}") for i in range(10)],
        }
        result = MixedWorkloadGenerator(seed=42).mix(sources, total=10)
        assert len(result) == 10

    def test_source_tagged_in_metadata(self):
        from agentsurge.generators.mixed import MixedWorkloadGenerator

        result = MixedWorkloadGenerator(seed=42).mix({"src1": [self._make_replay("s1")]}, total=1)
        assert result[0].metadata["source"] == "src1"

    def test_empty_sources(self):
        from agentsurge.generators.mixed import MixedWorkloadGenerator

        assert MixedWorkloadGenerator().mix({}) == []


# ===========================================================================
# Variable turn count distribution
# ===========================================================================


def test_geometric_distribution_varies_turns():
    from agentsurge.generators.synthetic import SyntheticGenerator

    sessions = SyntheticGenerator(n_turns=5, turn_distribution="geometric").generate(50)
    turn_counts = [s.n_turns for s in sessions]
    assert len(set(turn_counts)) > 1
    assert all(t >= 1 for t in turn_counts)


# ===========================================================================
# SingleTurnGenerator
# ===========================================================================


class TestSingleTurnGenerator:
    def test_generates_single_turn(self):
        from agentsurge.generators.synthetic import SingleTurnGenerator

        sessions = SingleTurnGenerator(seed=42).generate(5)
        assert len(sessions) == 5
        for s in sessions:
            assert s.n_turns == 1 and s.metadata["single_turn"] is True

    def test_deterministic(self):
        from agentsurge.generators.synthetic import SingleTurnGenerator

        s1 = SingleTurnGenerator(seed=42).generate(10)
        s2 = SingleTurnGenerator(seed=42).generate(10)
        for a, b in zip(s1, s2, strict=False):
            assert a.session_id == b.session_id and a.turn_messages == b.turn_messages


# ===========================================================================
# Workload versioning
# ===========================================================================


def test_workload_schema_version():
    parts = WORKLOAD_SCHEMA_VERSION.split(".")
    assert len(parts) == 2 and all(p.isdigit() for p in parts), (
        f"WORKLOAD_SCHEMA_VERSION must be 'major.minor', got {WORKLOAD_SCHEMA_VERSION!r}"
    )


@pytest.mark.e2e
def test_cmd_generate_produces_versioned_envelope(tmp_path):
    import argparse

    from agentsurge.cli.generate import cmd_generate

    args = argparse.Namespace(
        config=None,
        source=None,
        n_sessions=2,
        synthetic_tokens_per_turn=100,
        n_turns=2,
        output_dir=str(tmp_path),
        seed=42,
    )
    out = cmd_generate(args)
    with open(out) as f:
        data = json.load(f)
    assert data["version"] == WORKLOAD_SCHEMA_VERSION
    assert isinstance(data["sessions"], list)


@pytest.mark.e2e
def test_load_workload_sessions_versioned(tmp_path):
    from agentsurge.io import load_workload_sessions

    payload = {
        "version": "1.0",
        "sessions": [
            {
                "session_id": "s1",
                "turn_messages": [[{"role": "user", "content": "hi"}]],
                "metadata": {},
            }
        ],
    }
    p = tmp_path / "workload.json"
    p.write_text(json.dumps(payload))
    sessions = load_workload_sessions(str(p))
    assert len(sessions) == 1 and sessions[0].session_id == "s1"


@pytest.mark.e2e
def test_load_workload_sessions_legacy_list(tmp_path):
    from agentsurge.io import load_workload_sessions

    payload = [
        {
            "session_id": "legacy1",
            "turn_messages": [[{"role": "user", "content": "hello"}]],
            "metadata": {},
        }
    ]
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps(payload))
    assert len(load_workload_sessions(str(p))) == 1


@pytest.mark.e2e
def test_load_workload_sessions_invalid_format(tmp_path):
    from agentsurge.io import load_workload_sessions

    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"no_sessions_key": True}))
    with pytest.raises(ValueError, match="Unrecognised workload JSON"):
        load_workload_sessions(str(p))


def test_parse_workload_json_envelope():
    from agentsurge.io import _parse_workload_json

    sessions_list = [{"session_id": "b"}]
    assert _parse_workload_json({"version": "1.0", "sessions": sessions_list}) is sessions_list


def test_parse_workload_json_bad_dict():
    from agentsurge.io import _parse_workload_json

    with pytest.raises(ValueError):
        _parse_workload_json({"foo": "bar"})
