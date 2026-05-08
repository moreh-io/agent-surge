"""Tests for individual generator modules.

Covers: registry, BurstPatternGenerator, SteadyPatternGenerator,
MixedWorkloadGenerator, and GeneratorBase ABC behavior.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from agentsurge.generators.base import GeneratorBase
from agentsurge.generators.burst import BurstPatternGenerator, MapReduceSession
from agentsurge.generators.mixed import MixedWorkloadGenerator
from agentsurge.generators.registry import generator_registry
from agentsurge.generators.steady import SteadyPatternGenerator
from agentsurge.types import ReplaySession


def _assert_valid_session(s: ReplaySession):
    """Check that a session has the required structural fields."""
    assert isinstance(s.session_id, str) and s.session_id
    assert isinstance(s.turn_messages, list) and len(s.turn_messages) > 0
    assert isinstance(s.metadata, dict)
    for turn in s.turn_messages:
        assert isinstance(turn, list)
        assert len(turn) >= 1
        for msg in turn:
            assert "role" in msg
            assert "content" in msg


# ===========================================================================
# Registry
# ===========================================================================


class TestRegistry:
    def test_registry_contains_all_generators(self):
        names = generator_registry.list_generators()
        for expected in ("burst-pattern", "steady-pattern", "mixed"):
            assert expected in names, f"{expected!r} missing from registry"

    def test_registry_get_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown generator"):
            generator_registry.get("nonexistent-generator-xyz")

    def test_ensure_builtins_does_not_latch_on_import_failure(self):
        """If a builtin import fails, _builtins_loaded must not latch True —
        otherwise the registry stays permanently partial after a transient
        failure (e.g. a missing optional dep becoming available)."""
        import builtins

        from agentsurge.generators.registry import GeneratorRegistry

        reg = GeneratorRegistry()
        real_import = builtins.__import__

        def fail_burst(name, *a, **kw):
            if name == "agentsurge.generators.burst":
                raise ImportError("simulated failure")
            return real_import(name, *a, **kw)

        builtins.__import__ = fail_burst
        try:
            reg._ensure_builtins()
        finally:
            builtins.__import__ = real_import

        assert reg._builtins_loaded is False


# ===========================================================================
# BurstPatternGenerator
# ===========================================================================


class TestBurstGenerator:
    def test_burst_generates_correct_count(self):
        gen = BurstPatternGenerator(seed=42)
        n_tasks, n_workers = 2, 3
        sessions = gen.map_reduce(n_tasks=n_tasks, n_workers=n_workers)
        expected = n_tasks * (1 + n_workers + 1)  # planner + workers + gather
        assert len(sessions) == expected

    def test_burst_deterministic(self):
        a = BurstPatternGenerator(seed=7).map_reduce(n_tasks=1, n_workers=2)
        b = BurstPatternGenerator(seed=7).map_reduce(n_tasks=1, n_workers=2)
        assert len(a) == len(b)
        for sa, sb in zip(a, b, strict=True):
            assert sa.session_id == sb.session_id
            assert sa.turn_messages == sb.turn_messages

    def test_burst_sessions_have_valid_structure(self):
        gen = BurstPatternGenerator(seed=42)
        sessions = gen.map_reduce(n_tasks=1, n_workers=2)
        for s in sessions:
            _assert_valid_session(s)

    def test_burst_hierarchical_produces_tree(self):
        gen = BurstPatternGenerator(seed=42)
        sessions = gen.hierarchical(n_trees=1, branching_factor=2, depth=1)
        assert len(sessions) == 1 + 2  # root + 2 children
        roles = {s.agent_role for s in sessions}
        assert "root" in roles
        assert "leaf" in roles


# ===========================================================================
# SteadyPatternGenerator
# ===========================================================================


class TestSteadyGenerator:
    def test_steady_generates_correct_count(self):
        gen = SteadyPatternGenerator(seed=42)
        n_pipelines, stages = 2, 3
        sessions = gen.pipeline(n_pipelines=n_pipelines, stages=stages)
        assert len(sessions) == n_pipelines * stages

    def test_steady_deterministic(self):
        a = SteadyPatternGenerator(seed=11).pipeline(n_pipelines=1, stages=3)
        b = SteadyPatternGenerator(seed=11).pipeline(n_pipelines=1, stages=3)
        assert len(a) == len(b)
        for sa, sb in zip(a, b, strict=True):
            assert sa.session_id == sb.session_id
            assert sa.turn_messages == sb.turn_messages

    def test_steady_turn_count(self):
        gen = SteadyPatternGenerator(seed=42)
        sessions = gen.pipeline(n_pipelines=1, stages=4)
        for s in sessions:
            assert s.n_turns == 1
            _assert_valid_session(s)

    def test_steady_decentralized_turn_count(self):
        gen = SteadyPatternGenerator(seed=42)
        n_groups, n_agents, n_rounds = 1, 3, 4
        sessions = gen.decentralized(
            n_groups=n_groups,
            n_agents=n_agents,
            n_rounds=n_rounds,
        )
        assert len(sessions) == n_groups * n_agents
        for s in sessions:
            assert s.n_turns == n_rounds


# ===========================================================================
# MixedWorkloadGenerator
# ===========================================================================


class TestMixedWorkloadGenerator:
    @staticmethod
    def _make_source(prefix: str, count: int) -> list[ReplaySession]:
        return [
            ReplaySession(
                session_id=f"{prefix}_{i}",
                turn_messages=[[{"role": "user", "content": f"{prefix} prompt {i}"}]],
                metadata={"origin": prefix},
            )
            for i in range(count)
        ]

    def test_mixed_generates_sessions(self):
        gen = MixedWorkloadGenerator(seed=42)
        sources = {
            "alpha": self._make_source("alpha", 10),
            "beta": self._make_source("beta", 10),
        }
        result = gen.mix(sources, total=15)
        assert len(result) <= 15
        assert len(result) > 0
        source_labels = {s.metadata["source"] for s in result}
        assert len(source_labels) == 2, "mixed output should contain both sources"

    def test_mixed_deterministic(self):
        sources = {
            "a": self._make_source("a", 10),
            "b": self._make_source("b", 10),
        }
        a = MixedWorkloadGenerator(seed=42).mix(sources, total=10)
        b = MixedWorkloadGenerator(seed=42).mix(sources, total=10)
        assert len(a) == len(b)
        for sa, sb in zip(a, b, strict=True):
            assert sa.session_id == sb.session_id

    def test_mixed_empty_sources(self):
        gen = MixedWorkloadGenerator(seed=42)
        assert gen.mix({}) == []

    def test_mixed_respects_weights(self):
        gen = MixedWorkloadGenerator(seed=42)
        sources = {
            "heavy": self._make_source("heavy", 100),
            "light": self._make_source("light", 100),
        }
        result = gen.mix(sources, weights={"heavy": 0.9, "light": 0.1}, total=20)
        heavy_count = sum(1 for s in result if s.metadata["source"] == "heavy")
        assert heavy_count > 10, f"heavy source (weight 0.9) should dominate, got {heavy_count}/20"

    def test_mix_raises_when_pool_smaller_than_weighted_demand(self):
        """mix() must raise ValueError when weighted demand exceeds pool size (M14)."""
        gen = MixedWorkloadGenerator(seed=42)
        small_pool = self._make_source("g", 2)
        with pytest.raises(ValueError, match="g"):
            gen.mix(sources={"g": small_pool}, total=10)

    def test_mix_preserves_subclass_identity(self):
        """mix() must not strip subclass type or extended fields (H7)."""
        burst = BurstPatternGenerator(seed=0)
        map_reduce_sessions = burst.map_reduce(
            n_tasks=1, n_workers=2, planner_turns=1, worker_turns=1, gather_turns=1
        )
        assert all(isinstance(s, MapReduceSession) for s in map_reduce_sessions)

        gen = MixedWorkloadGenerator(seed=42)
        result = gen.mix(sources={"mr": map_reduce_sessions}, total=2)

        for s in result:
            assert isinstance(s, MapReduceSession), (
                f"Expected MapReduceSession, got {type(s).__name__}"
            )
            assert s.pattern == "map_reduce", "subclass field 'pattern' was stripped"
            assert s.agent_role != "" or True  # noqa: SIM222 - asserts attr exists
            _ = s.agent_role  # raises AttributeError if field was stripped


# ===========================================================================
# M13 – lazy loading of technical_topics.json
# ===========================================================================


class TestTechnicalDocLazyLoad:
    """_technical_doc must not read the JSON file at import time (M13).

    A missing technical_topics.json should not prevent importing unrelated
    generators.  The error must only surface when the functionality is actually
    used.
    """

    def test_missing_json_does_not_break_import(self, monkeypatch):
        """Importing _technical_doc with a missing JSON must not raise at import."""
        mod_name = "agentsurge.generators._technical_doc"

        # Remove cached module so we get a fresh import.
        monkeypatch.delitem(sys.modules, mod_name, raising=False)

        # Patch Path.read_text to simulate a missing file.
        original_read_text = None

        def _raise_fnf(self, *args, **kwargs):
            if "technical_topics.json" in str(self):
                raise FileNotFoundError("simulated missing file")
            if original_read_text is not None:
                return original_read_text(self, *args, **kwargs)
            raise AssertionError("unexpected read_text call")

        from pathlib import Path

        original_read_text = Path.read_text
        monkeypatch.setattr(Path, "read_text", _raise_fnf)

        # After fix: import must succeed.
        mod = importlib.import_module(mod_name)
        assert mod is not None

    def test_missing_json_raises_on_use(self, monkeypatch):
        """Calling _make_technical_doc when the JSON is absent must raise FileNotFoundError."""
        mod_name = "agentsurge.generators._technical_doc"

        # Remove cached module so we get a fresh import.
        monkeypatch.delitem(sys.modules, mod_name, raising=False)

        # Also clear the lazy cache that may be set from a previous test.
        import agentsurge.generators._technical_doc as _td_before

        monkeypatch.setattr(_td_before, "_TOPICS", None, raising=False)

        from pathlib import Path

        original_read_text = Path.read_text

        def _raise_fnf(self, *args, **kwargs):
            if "technical_topics.json" in str(self):
                raise FileNotFoundError("simulated missing file")
            return original_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", _raise_fnf)

        # Re-import after patching Path.read_text.
        monkeypatch.delitem(sys.modules, mod_name, raising=False)
        mod = importlib.import_module(mod_name)

        # Reset the lazy cache on the freshly-imported module.
        if hasattr(mod, "_TOPICS"):
            monkeypatch.setattr(mod, "_TOPICS", None)

        with pytest.raises(FileNotFoundError):
            mod._make_technical_doc(100, seed=0)


# ===========================================================================
# GeneratorBase ABC behavior
# ===========================================================================


class TestGeneratorBase:
    def test_named_subclass_registers(self):
        class _TestOnlyGen(GeneratorBase):
            name = "_test-only-ephemeral"

        assert "_test-only-ephemeral" in generator_registry


# ---------------------------------------------------------------------------
# TraceReplayGenerator: pending_user_messages extraction
# ---------------------------------------------------------------------------


class TestTraceReplayPendingUserMessages:
    def _make_traj(self, turns, *, supports_multi_turn):
        from agentsurge.types import Session, Trajectory

        sess = Session(
            session_id="s1",
            turns=turns,
            metadata={"supports_multi_turn": supports_multi_turn},
        )
        return Trajectory(
            trajectory_id="t1",
            sessions=[sess],
            metadata={"supports_multi_turn": supports_multi_turn},
        )

    def test_capable_source_extracts_follow_ups(self):
        from agentsurge.generators.trace_replay import TraceReplayGenerator
        from agentsurge.types import Turn

        turns = [
            Turn(role="user", content="u1"),
            Turn(role="assistant", content="a1"),
            Turn(role="user", content="u2"),
            Turn(role="assistant", content="a2"),
            Turn(role="user", content="u3"),
        ]
        traj = self._make_traj(turns, supports_multi_turn=True)
        gen = TraceReplayGenerator(flatten_tools=False)
        sessions = list(gen.from_trajectories(iter([traj])))

        assert len(sessions) == 1
        assert sessions[0].pending_user_messages == ["u2", "u3"]

    def test_incapable_source_leaves_pending_empty(self):
        from agentsurge.generators.trace_replay import TraceReplayGenerator
        from agentsurge.types import Turn

        turns = [
            Turn(role="user", content="u1"),
            Turn(role="assistant", content="a1"),
            Turn(role="user", content="u2"),
        ]
        traj = self._make_traj(turns, supports_multi_turn=False)
        gen = TraceReplayGenerator(flatten_tools=False)
        sessions = list(gen.from_trajectories(iter([traj])))

        assert sessions[0].pending_user_messages == []

    def test_single_user_turn_has_empty_queue(self):
        from agentsurge.generators.trace_replay import TraceReplayGenerator
        from agentsurge.types import Turn

        turns = [
            Turn(role="user", content="u1"),
            Turn(role="assistant", content="a1"),
        ]
        traj = self._make_traj(turns, supports_multi_turn=True)
        gen = TraceReplayGenerator(flatten_tools=False)
        sessions = list(gen.from_trajectories(iter([traj])))

        assert sessions[0].pending_user_messages == []

    def test_skips_empty_user_content(self):
        """Empty/blank user content shouldn't be queued."""
        from agentsurge.generators.trace_replay import TraceReplayGenerator
        from agentsurge.types import Turn

        turns = [
            Turn(role="user", content="u1"),
            Turn(role="assistant", content="a1"),
            Turn(role="user", content=""),
            Turn(role="assistant", content="a2"),
            Turn(role="user", content="   "),
            Turn(role="assistant", content="a3"),
            Turn(role="user", content="u4"),
        ]
        traj = self._make_traj(turns, supports_multi_turn=True)
        gen = TraceReplayGenerator(flatten_tools=False)
        sessions = list(gen.from_trajectories(iter([traj])))

        assert sessions[0].pending_user_messages == ["u4"]

    def test_flatten_tools_mode_still_extracts_from_original_turns(self):
        """pending_user_messages uses original Turn objects (not flattened tool
        outputs), so it works regardless of flatten_tools setting."""
        from agentsurge.generators.trace_replay import TraceReplayGenerator
        from agentsurge.types import Turn

        turns = [
            Turn(role="user", content="u1"),
            Turn(role="assistant", content="a1"),
            Turn(role="tool", content="tool-output-1", name="bash"),
            Turn(role="user", content="u2"),
        ]
        traj = self._make_traj(turns, supports_multi_turn=True)
        gen = TraceReplayGenerator(flatten_tools=True)
        sessions = list(gen.from_trajectories(iter([traj])))

        assert sessions[0].pending_user_messages == ["u2"]

    def test_whitespace_padded_content_is_stored_stripped(self):
        """Guard uses stripped content; the appended value must also be stripped."""
        from agentsurge.generators.trace_replay import TraceReplayGenerator
        from agentsurge.types import Turn

        turns = [
            Turn(role="user", content="u1"),
            Turn(role="assistant", content="a1"),
            Turn(role="user", content="  hello  "),
        ]
        traj = self._make_traj(turns, supports_multi_turn=True)
        gen = TraceReplayGenerator(flatten_tools=False)
        sessions = list(gen.from_trajectories(iter([traj])))

        assert sessions[0].pending_user_messages == ["hello"]

    def test_turn_messages_entries_have_independent_dicts(self):
        """turn_messages[i] and turn_messages[j] must not share dict refs —
        mutating a message inside one turn snapshot would otherwise corrupt
        all later snapshots (inject_response patches by position)."""
        from agentsurge.generators.trace_replay import TraceReplayGenerator
        from agentsurge.types import Turn

        turns = [
            Turn(role="user", content="u1"),
            Turn(role="assistant", content="a1"),
            Turn(role="user", content="u2"),
            Turn(role="assistant", content="a2"),
        ]
        traj = self._make_traj(turns, supports_multi_turn=False)
        gen = TraceReplayGenerator(flatten_tools=True)
        sessions = list(gen.from_trajectories(iter([traj])))

        tm = sessions[0].turn_messages
        assert len(tm) >= 2
        # Mutate a dict in turn 1; turn 0's same-position dict must stay untouched.
        shared_idx = 0  # both turns share a prefix from index 0
        original = tm[0][shared_idx]["content"]
        tm[1][shared_idx]["content"] = "MUTATED"
        assert tm[0][shared_idx]["content"] == original, (
            "turn_messages entries share dict refs — mutation leaks across turns"
        )

    def test_tool_calls_not_aliased_across_snapshots(self):
        """tool_calls lists must not be shared across turn snapshots.

        _build_replay used dict(m) shallow copies; the tool_calls list was the
        same object in every snapshot that included the assistant turn.  Verify
        that mutating tool_calls in one snapshot does not corrupt another.
        """
        from agentsurge.generators.trace_replay import TraceReplayGenerator
        from agentsurge.types import Session, Turn

        tool_call = {
            "id": "tc1",
            "type": "function",
            "function": {"name": "bash", "arguments": "{}"},
        }
        # Agent loop: user -> assistant (tool_calls) -> tool -> assistant -> user -> assistant
        # This produces 3 snapshots; the assistant+tool_calls message appears in
        # snapshots 1 and 2, letting us verify the aliasing bug.
        turns = [
            Turn(role="user", content="u1"),
            Turn(role="assistant", content="", tool_calls=[tool_call]),
            Turn(role="tool", content="output", tool_call_id="tc1", name="bash"),
            Turn(role="assistant", content="done"),
            Turn(role="user", content="u2"),
            Turn(role="assistant", content="done2"),
        ]
        sess = Session(session_id="s-tc", turns=turns)
        gen = TraceReplayGenerator(flatten_tools=False)
        replay = gen._build_replay(sess)

        tm = replay.turn_messages
        # Find a snapshot that has the assistant+tool_calls message and is
        # present in at least two different snapshots.
        tc_positions = []
        for snap_idx, snap in enumerate(tm):
            for msg in snap:
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    tc_positions.append((snap_idx, msg["tool_calls"]))
                    break

        assert len(tc_positions) >= 2, (
            f"Expected tool_calls to appear in 2+ snapshots, got {len(tc_positions)}"
        )

        first_tc = tc_positions[0][1]
        second_tc = tc_positions[1][1]

        # They must be different objects (not aliased).
        assert first_tc is not second_tc, (
            "tool_calls list is the same object across snapshots (aliased shallow copy)"
        )

        # Independence check: mutating one must not affect the other.
        first_tc[0]["function"]["name"] = "MUTATED"
        assert second_tc[0]["function"]["name"] == "bash", (
            "Mutating tool_calls in one snapshot corrupted another"
        )

    def test_no_double_feed_with_flatten_tools_and_supports_multi_turn(self):
        """Tool-mode runner seeds messages from turn_messages[0] only and uses
        pending_user_messages for follow-ups. Verify that u2 appears ONLY in
        pending (not in turn_messages[0]), so the runner feeds it exactly once."""
        from agentsurge.generators.trace_replay import TraceReplayGenerator
        from agentsurge.types import Turn

        turns = [
            Turn(role="user", content="u1"),
            Turn(role="assistant", content="a1"),
            Turn(role="tool", content="bash-output", name="bash"),
            Turn(role="user", content="u2"),
            Turn(role="assistant", content="a2"),
        ]
        traj = self._make_traj(turns, supports_multi_turn=True)
        gen = TraceReplayGenerator(flatten_tools=True)
        sessions = list(gen.from_trajectories(iter([traj])))

        seed = sessions[0].turn_messages[0]
        assert all(m.get("content") != "u2" for m in seed), (
            f"u2 leaked into turn_messages[0]: {seed}"
        )


# ===========================================================================
# SyntheticGenerator – metadata replay seed (H8)
# ===========================================================================


class TestSyntheticGeneratorMetadataSeed:
    def test_metadata_stores_replayable_base_seed_and_session_index(self):
        """metadata must store base_seed and session_index, not seed+s."""
        from agentsurge.generators.synthetic import SyntheticGenerator

        gen = SyntheticGenerator(n_turns=1)
        sessions = gen.generate(n_sessions=1, seed=7)
        sess = sessions[0]

        assert "base_seed" in sess.metadata, "metadata missing 'base_seed'"
        assert "session_index" in sess.metadata, "metadata missing 'session_index'"
        assert sess.metadata["base_seed"] == 7
        assert sess.metadata["session_index"] == 0

    def test_metadata_seed_reproduces_first_user_prompt(self):
        """Caller using base_seed + session_index must reproduce the prompt."""
        from agentsurge.generators.synthetic import SyntheticGenerator, _make_synthetic_prompt

        gen = SyntheticGenerator(n_turns=1)
        sessions = gen.generate(n_sessions=1, seed=7)
        sess = sessions[0]

        base_seed = sess.metadata["base_seed"]
        session_index = sess.metadata["session_index"]

        turn = 0
        replay_seed = base_seed * 1_000_000 + session_index * 1_000 + turn
        expected = _make_synthetic_prompt(gen.tokens_per_turn, replay_seed)

        actual = sess.turn_messages[0][1]["content"]
        assert actual == expected, "replayed prompt does not match stored content"
