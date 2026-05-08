"""Behavioral invariant tests for the 5-stage code review framework.

Each test proves a specific contract declared during review. If all these
tests pass, the corresponding behavioral property is verified.

Contracts:
  1. estimated_tokens - math properties (linearity, empty handling, last-turn-only)
  2. inject_response - propagation to ALL subsequent turns
  3. Gamma arrival - monotonicity, non-negativity, statistical mean
  4. _compute_turn_delay - layered priority (tool > think_time)
  5. Fan-out - spawns concurrent tasks, TTFT = min(ok results)
  6. End-to-end - generate → replay → metrics pipeline consistency
"""

from __future__ import annotations

import asyncio
import random
import statistics
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentsurge import BenchmarkConfig, ReplaySession, RunResult
from agentsurge.backends.mock import MockBackend, MockConfig
from agentsurge.runner import BenchmarkRunner
from agentsurge.types import SessionResult, TurnResult

# ===========================================================================
# Contract 1: estimated_tokens math properties
# ===========================================================================


class TestEstimatedTokens:
    def test_empty_turn_messages_returns_zero(self):
        sess = ReplaySession(session_id="s1", turn_messages=[])
        assert sess.estimated_tokens == 0

    def test_uses_last_turn_only(self):
        """Earlier turns should NOT affect the estimate."""
        sess = ReplaySession(
            session_id="s1",
            turn_messages=[
                [{"role": "user", "content": "a" * 1000}],  # turn 0: 1000 chars
                [{"role": "user", "content": "b" * 100}],  # turn 1: 100 chars
            ],
        )
        assert sess.estimated_tokens == 100 // 4  # 25, not (1000+100)//4

    def test_integer_division_rounds_down(self):
        # 3 chars → 0 tokens, 4 → 1, 5 → 1, 7 → 1, 8 → 2
        for chars, expected in [(3, 0), (4, 1), (5, 1), (7, 1), (8, 2)]:
            sess = ReplaySession(
                session_id="s1", turn_messages=[[{"role": "user", "content": "x" * chars}]]
            )
            assert sess.estimated_tokens == expected, f"chars={chars}"

    def test_sums_across_all_messages_in_turn(self):
        """Multiple messages in the final turn should all contribute."""
        sess = ReplaySession(
            session_id="s1",
            turn_messages=[
                [
                    {"role": "system", "content": "a" * 40},
                    {"role": "user", "content": "b" * 40},
                    {"role": "assistant", "content": "c" * 40},
                    {"role": "user", "content": "d" * 40},
                ],
            ],
        )
        assert sess.estimated_tokens == 160 // 4  # 40

    def test_handles_missing_content_key(self):
        """Messages without 'content' should contribute 0 chars."""
        sess = ReplaySession(
            session_id="s1",
            turn_messages=[[{"role": "user"}, {"role": "assistant", "content": "x" * 80}]],
        )
        assert sess.estimated_tokens == 80 // 4  # 20

    def test_linearity(self):
        """Doubling content should roughly double tokens."""
        sess_a = ReplaySession(
            session_id="a", turn_messages=[[{"role": "user", "content": "x" * 400}]]
        )
        sess_b = ReplaySession(
            session_id="b", turn_messages=[[{"role": "user", "content": "x" * 800}]]
        )
        assert sess_b.estimated_tokens == 2 * sess_a.estimated_tokens


# ===========================================================================
# Contract 2: inject_response propagation to ALL subsequent turns
# ===========================================================================


class TestInjectResponsePropagation:
    def _make_5_turn_session(self):
        """5-turn session where each turn accumulates messages."""
        return ReplaySession(
            session_id="s1",
            turn_messages=[
                # Turn 0: [sys, user]
                [{"role": "system", "content": "sys"}, {"role": "user", "content": "q1"}],
                # Turn 1: [sys, user, assistant, user]
                [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "q1"},
                    {"role": "assistant", "content": "placeholder_A1"},
                    {"role": "user", "content": "q2"},
                ],
                # Turn 2: [sys, user, assistant, user, assistant, user]
                [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "q1"},
                    {"role": "assistant", "content": "placeholder_A1"},
                    {"role": "user", "content": "q2"},
                    {"role": "assistant", "content": "placeholder_A2"},
                    {"role": "user", "content": "q3"},
                ],
                # Turn 3: [sys, user, assistant, user, assistant, user, assistant, user]
                [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "q1"},
                    {"role": "assistant", "content": "placeholder_A1"},
                    {"role": "user", "content": "q2"},
                    {"role": "assistant", "content": "placeholder_A2"},
                    {"role": "user", "content": "q3"},
                    {"role": "assistant", "content": "placeholder_A3"},
                    {"role": "user", "content": "q4"},
                ],
                # Turn 4: same + A4 + user
                [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "q1"},
                    {"role": "assistant", "content": "placeholder_A1"},
                    {"role": "user", "content": "q2"},
                    {"role": "assistant", "content": "placeholder_A2"},
                    {"role": "user", "content": "q3"},
                    {"role": "assistant", "content": "placeholder_A3"},
                    {"role": "user", "content": "q4"},
                    {"role": "assistant", "content": "placeholder_A4"},
                    {"role": "user", "content": "q5"},
                ],
            ],
        )

    def test_propagates_to_all_subsequent_turns(self):
        """inject_response(0, ...) must patch turns 1, 2, 3, and 4."""
        session = self._make_5_turn_session()
        session.inject_response(0, "ACTUAL_RESPONSE_1")

        # Turn 0: no assistant → unchanged
        assert all(m["content"] != "ACTUAL_RESPONSE_1" for m in session.turn_messages[0])

        # Turns 1-4: position 2 (the first assistant) should be patched
        for t_idx in [1, 2, 3, 4]:
            assert session.turn_messages[t_idx][2]["content"] == "ACTUAL_RESPONSE_1", (
                f"Turn {t_idx} not patched"
            )

    def test_later_assistants_unaffected(self):
        """inject_response(0, ...) should only patch the assistant at position found in turn 1."""
        session = self._make_5_turn_session()
        session.inject_response(0, "ACTUAL_RESPONSE_1")

        # placeholder_A2, A3, A4 should still be placeholders
        assert session.turn_messages[2][4]["content"] == "placeholder_A2"
        assert session.turn_messages[3][6]["content"] == "placeholder_A3"
        assert session.turn_messages[4][8]["content"] == "placeholder_A4"

    def test_chained_injections(self):
        """Sequential inject_response calls for each turn should all work."""
        session = self._make_5_turn_session()
        session.inject_response(0, "ACTUAL_1")
        session.inject_response(1, "ACTUAL_2")
        session.inject_response(2, "ACTUAL_3")

        # Turn 3 should have ACTUAL_1 at pos 2, ACTUAL_2 at pos 4, ACTUAL_3 at pos 6
        t3 = session.turn_messages[3]
        assert t3[2]["content"] == "ACTUAL_1"
        assert t3[4]["content"] == "ACTUAL_2"
        assert t3[6]["content"] == "ACTUAL_3"

        # Turn 4 same
        t4 = session.turn_messages[4]
        assert t4[2]["content"] == "ACTUAL_1"
        assert t4[4]["content"] == "ACTUAL_2"
        assert t4[6]["content"] == "ACTUAL_3"

    def test_last_turn_injection_is_noop(self):
        """Injecting at the last turn should be a no-op (no subsequent turns)."""
        session = self._make_5_turn_session()
        before = [m["content"] for m in session.turn_messages[4]]
        session.inject_response(4, "SHOULD_NOT_APPEAR")
        after = [m["content"] for m in session.turn_messages[4]]
        assert before == after

    def test_no_assistant_in_next_turn_is_noop(self):
        """If turn_idx+1 has no assistant message, inject_response is a no-op."""
        session = ReplaySession(
            session_id="s1",
            turn_messages=[
                [{"role": "user", "content": "q1"}],
                [{"role": "user", "content": "q1"}, {"role": "user", "content": "q2"}],
            ],
        )
        session.inject_response(0, "should_not_crash")
        # No assistant → nothing to replace → no crash
        assert all(m["role"] == "user" for m in session.turn_messages[1])

    def test_replacement_no_aliasing(self):
        """Each turn gets its own dict copy to prevent mutation aliasing."""
        session = self._make_5_turn_session()
        session.inject_response(0, "ACTUAL")
        # All patched positions should have EQUAL content but DIFFERENT objects
        objs = [session.turn_messages[t][2] for t in [1, 2, 3, 4]]
        assert all(o["content"] == "ACTUAL" for o in objs)
        assert all(o is not objs[0] for o in objs[1:]), (
            "Each turn should get its own dict to prevent mutation aliasing"
        )

    def test_mutation_after_injection_does_not_propagate(self):
        """Mutating one turn's injected response should NOT affect other turns."""
        session = self._make_5_turn_session()
        session.inject_response(0, "ORIGINAL")
        # Mutate turn 1's assistant dict
        session.turn_messages[1][2]["content"] = "MUTATED"
        # Turn 2, 3, 4 should still have "ORIGINAL"
        assert session.turn_messages[2][2]["content"] == "ORIGINAL"
        assert session.turn_messages[3][2]["content"] == "ORIGINAL"
        assert session.turn_messages[4][2]["content"] == "ORIGINAL"


# ===========================================================================
# Contract 3: Gamma arrival distribution properties
# ===========================================================================


class TestGammaArrivalDistribution:
    def _make_runner(self, rate=5.0, cv=2.0, seed=42):
        return BenchmarkRunner(
            BenchmarkConfig(
                vllm_url="http://x",
                model="t",
                arrival_pattern="gamma",
                arrival_rate=rate,
                gamma_cv=cv,
                seed=seed,
            )
        )

    def test_monotonically_increasing(self):
        delays = self._make_runner()._compute_arrival_delays(50)
        for a, b in zip(delays, delays[1:], strict=False):
            assert b > a, "Gamma delays must be strictly increasing"

    def test_non_negative(self):
        delays = self._make_runner()._compute_arrival_delays(50)
        assert all(d >= 0.0 for d in delays)

    def test_correct_length(self):
        for n in [0, 1, 5, 100]:
            assert len(self._make_runner()._compute_arrival_delays(n)) == n

    def test_deterministic_with_same_seed(self):
        d1 = self._make_runner(seed=99)._compute_arrival_delays(20)
        d2 = self._make_runner(seed=99)._compute_arrival_delays(20)
        assert d1 == d2

    def test_different_seeds_differ(self):
        d1 = self._make_runner(seed=1)._compute_arrival_delays(20)
        d2 = self._make_runner(seed=2)._compute_arrival_delays(20)
        assert d1 != d2

    def test_mean_interarrival_approximates_inverse_rate(self):
        """For large n, mean inter-arrival should approximate 1/rate."""
        rate = 5.0
        runner = self._make_runner(rate=rate, cv=2.0, seed=42)
        delays = runner._compute_arrival_delays(2000)
        inter_arrivals = [b - a for a, b in zip(delays, delays[1:], strict=False)]
        mean_ia = statistics.mean(inter_arrivals)
        # Gamma mean = shape * scale = (1/CV^2) * (CV^2/rate) = 1/rate
        expected = 1.0 / rate
        assert abs(mean_ia - expected) / expected < 0.10, (
            f"Mean inter-arrival {mean_ia:.4f} too far from 1/rate={expected:.4f}"
        )

    def test_cv1_resembles_exponential(self):
        """CV=1.0 makes Gamma(1, 1/rate) = Exponential(rate)."""
        rate = 10.0
        runner_gamma = BenchmarkRunner(
            BenchmarkConfig(
                vllm_url="http://x",
                model="t",
                arrival_pattern="gamma",
                arrival_rate=rate,
                gamma_cv=1.0,
                seed=42,
            )
        )
        runner_poisson = BenchmarkRunner(
            BenchmarkConfig(
                vllm_url="http://x",
                model="t",
                arrival_pattern="poisson",
                arrival_rate=rate,
                seed=42,
            )
        )
        # Both should have similar statistical properties (same distribution)
        d_gamma = runner_gamma._compute_arrival_delays(1000)
        d_poisson = runner_poisson._compute_arrival_delays(1000)
        ia_gamma = [b - a for a, b in zip(d_gamma, d_gamma[1:], strict=False)]
        ia_poisson = [b - a for a, b in zip(d_poisson, d_poisson[1:], strict=False)]
        # Means should be close (both ≈ 1/rate)
        assert abs(statistics.mean(ia_gamma) - statistics.mean(ia_poisson)) < 0.02

    def test_high_cv_increases_variance(self):
        """Higher CV should produce more bursty (higher variance) arrivals."""
        rate = 5.0
        low_cv = self._make_runner(rate=rate, cv=1.0, seed=42)._compute_arrival_delays(500)
        high_cv = self._make_runner(rate=rate, cv=3.0, seed=42)._compute_arrival_delays(500)
        ia_low = [b - a for a, b in zip(low_cv, low_cv[1:], strict=False)]
        ia_high = [b - a for a, b in zip(high_cv, high_cv[1:], strict=False)]
        assert statistics.variance(ia_high) > statistics.variance(ia_low)


# ===========================================================================
# Contract 3b: Ramp arrival distribution properties
# ===========================================================================


class TestRampArrivalDistribution:
    def test_monotonically_increasing(self):
        runner = BenchmarkRunner(
            BenchmarkConfig(
                vllm_url="http://x",
                model="t",
                arrival_pattern="ramp",
                arrival_rate=10.0,
                ramp_duration=5.0,
                seed=42,
            )
        )
        delays = runner._compute_arrival_delays(50)
        for a, b in zip(delays, delays[1:], strict=False):
            assert b > a

    def test_non_negative(self):
        runner = BenchmarkRunner(
            BenchmarkConfig(
                vllm_url="http://x",
                model="t",
                arrival_pattern="ramp",
                arrival_rate=10.0,
                ramp_duration=5.0,
                seed=42,
            )
        )
        delays = runner._compute_arrival_delays(50)
        assert all(d >= 0.0 for d in delays)

    def test_ramp_duration_zero_clamped(self):
        """ramp_duration=0 should be clamped to 1.0 (no division by zero)."""
        runner = BenchmarkRunner(
            BenchmarkConfig(
                vllm_url="http://x",
                model="t",
                arrival_pattern="ramp",
                arrival_rate=10.0,
                ramp_duration=0.0,
                seed=42,
            )
        )
        delays = runner._compute_arrival_delays(10)
        assert len(delays) == 10 and all(d >= 0.0 for d in delays)


# ===========================================================================
# Contract 4: _compute_turn_delay layered priority
# ===========================================================================


class TestComputeTurnDelay:
    def _make_session_with_tool_turn(self):
        """3-turn session where turn 1 has tool output.

        _is_tool_turn checks messages added in the PREVIOUS turn, so
        tool output in turn 1 is detected at turn_idx=2.
        """
        return ReplaySession(
            session_id="s1",
            turn_messages=[
                [{"role": "user", "content": "Fix bug"}],
                [
                    {"role": "user", "content": "Fix bug"},
                    {"role": "assistant", "content": "Reading file."},
                    {"role": "user", "content": "[Tool output: read_file] def foo(): pass"},
                ],
                [
                    {"role": "user", "content": "Fix bug"},
                    {"role": "assistant", "content": "Reading file."},
                    {"role": "user", "content": "[Tool output: read_file] def foo(): pass"},
                    {"role": "assistant", "content": "I see."},
                    {"role": "user", "content": "Now fix it."},
                ],
            ],
        )

    def _make_session_human_turn(self):
        """Session where turn 1 is a human (non-tool) turn."""
        return ReplaySession(
            session_id="s1",
            turn_messages=[
                [{"role": "user", "content": "Hello"}],
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there."},
                    {"role": "user", "content": "What about X?"},
                ],
            ],
        )

    def test_no_configs_returns_zero(self):
        """All delay configs off → 0.0."""
        runner = BenchmarkRunner(
            BenchmarkConfig(
                vllm_url="http://x",
                model="t",
                tool_delay=False,
                think_time=None,
            )
        )
        delay = runner._compute_turn_delay(self._make_session_human_turn(), 1, random.Random(42))
        assert delay == 0.0

    def test_think_time_for_human_turn(self):
        """Human turn with think_time → LogNormal sample > 0."""
        runner = BenchmarkRunner(
            BenchmarkConfig(
                vllm_url="http://x",
                model="t",
                think_time=(1.0, 0.5),
                tool_delay=False,
            )
        )
        delay = runner._compute_turn_delay(self._make_session_human_turn(), 1, random.Random(42))
        assert delay > 0.0

    def test_tool_delay_takes_precedence_over_think_time(self):
        """Tool turn with tool_delay=True should NOT use think_time."""
        runner = BenchmarkRunner(
            BenchmarkConfig(
                vllm_url="http://x",
                model="t",
                tool_delay=True,
                think_time=(1.0, 0.5),
            )
        )
        rng = random.Random(42)
        with patch("agentsurge.tool_timing.sample_tool_delay", return_value=99.9) as mock_tool:
            delay = runner._compute_turn_delay(self._make_session_with_tool_turn(), 2, rng)
        assert delay == 99.9, "Tool delay should take precedence"
        mock_tool.assert_called_once()

    def test_think_time_not_used_for_tool_turns(self):
        """Tool turn should skip think_time even when it's configured."""
        runner = BenchmarkRunner(
            BenchmarkConfig(
                vllm_url="http://x",
                model="t",
                tool_delay=False,
                think_time=(1.0, 0.5),
            )
        )
        rng = random.Random(42)
        delay = runner._compute_turn_delay(self._make_session_with_tool_turn(), 2, rng)
        assert delay == 0.0

    def test_deterministic_with_same_rng(self):
        runner = BenchmarkRunner(
            BenchmarkConfig(
                vllm_url="http://x",
                model="t",
                think_time=(1.0, 0.5),
            )
        )
        d1 = runner._compute_turn_delay(self._make_session_human_turn(), 1, random.Random(42))
        d2 = runner._compute_turn_delay(self._make_session_human_turn(), 1, random.Random(42))
        assert d1 == d2


# ===========================================================================
# Contract 5: Fan-out spawns concurrent tasks
# ===========================================================================


@pytest.mark.e2e
class TestFanOutConcurrency:
    def test_fan_out_records_separate_turns(self):
        """fan_out=3 should produce 3 separate TurnResults."""
        runner = BenchmarkRunner(BenchmarkConfig(vllm_url="http://x", model="t"))
        call_log = []

        async def mock_send(http, sid, tidx, msgs, **kwargs):
            call_log.append(sid)
            return TurnResult(
                session_id=sid,
                turn_index=tidx,
                completed=True,
                ttft_ms=50.0,
                total_ms=100.0,
                output_tokens=10,
            )

        runner._send_with_retry = mock_send

        session = ReplaySession(
            session_id="s1", turn_messages=[[{"role": "user", "content": "hi"}]], fan_out=3
        )

        async def _run():
            with patch("asyncio.sleep", new_callable=AsyncMock):
                return await runner._run_session(MagicMock(), asyncio.Semaphore(10), session, 0.0)

        result = asyncio.run(_run())
        # 3 fan-out calls → 3 separate TurnResults
        assert len(call_log) == 3
        assert set(call_log) == {"s1_fan0", "s1_fan1", "s1_fan2"}
        assert len(result.turns) == 3
        assert {t.session_id for t in result.turns} == {"s1_fan0", "s1_fan1", "s1_fan2"}

    def test_fan_out_individual_ttfts_preserved(self):
        """Each fan-out TurnResult should have its own TTFT (not aggregated)."""
        runner = BenchmarkRunner(BenchmarkConfig(vllm_url="http://x", model="t"))

        async def mock_send(http, sid, tidx, msgs, **kwargs):
            fan_idx = int(sid.split("_fan")[-1])
            ttft = [100.0, 50.0, 200.0][fan_idx]
            return TurnResult(
                session_id=sid,
                turn_index=tidx,
                completed=True,
                ttft_ms=ttft,
                total_ms=ttft + 50,
                output_tokens=10,
            )

        runner._send_with_retry = mock_send

        session = ReplaySession(
            session_id="s1", turn_messages=[[{"role": "user", "content": "hi"}]], fan_out=3
        )

        async def _run():
            with patch("asyncio.sleep", new_callable=AsyncMock):
                return await runner._run_session(MagicMock(), asyncio.Semaphore(10), session, 0.0)

        result = asyncio.run(_run())
        ttfts = sorted(t.ttft_ms for t in result.turns)
        assert ttfts == [50.0, 100.0, 200.0]  # individual, not aggregated
        assert all(t.fan_out_count == 3 for t in result.turns)

    def test_fan_out_1_no_fan_call(self):
        """fan_out=1 should use the normal (non-fan) path."""
        runner = BenchmarkRunner(BenchmarkConfig(vllm_url="http://x", model="t"))
        call_log = []

        async def mock_send(http, sid, tidx, msgs, **kwargs):
            call_log.append(sid)
            return TurnResult(session_id=sid, turn_index=tidx, completed=True, ttft_ms=50.0)

        runner._send_with_retry = mock_send

        session = ReplaySession(
            session_id="s1", turn_messages=[[{"role": "user", "content": "hi"}]], fan_out=1
        )

        async def _run():
            with patch("asyncio.sleep", new_callable=AsyncMock):
                return await runner._run_session(MagicMock(), asyncio.Semaphore(10), session, 0.0)

        result = asyncio.run(_run())
        # fan_out=1 → normal path → session_id is "s1" not "s1_fan0"
        assert call_log == ["s1"]
        assert len(result.turns) == 1

    def test_fan_out_all_fail_still_records_all(self):
        """If all fan-out calls fail, all 3 are still recorded as separate TurnResults."""
        runner = BenchmarkRunner(BenchmarkConfig(vllm_url="http://x", model="t"))

        async def mock_send(http, sid, tidx, msgs, **kwargs):
            return TurnResult(session_id=sid, turn_index=tidx, completed=False, error="fail")

        runner._send_with_retry = mock_send

        session = ReplaySession(
            session_id="s1", turn_messages=[[{"role": "user", "content": "hi"}]], fan_out=3
        )

        async def _run():
            with patch("asyncio.sleep", new_callable=AsyncMock):
                return await runner._run_session(MagicMock(), asyncio.Semaphore(10), session, 0.0)

        result = asyncio.run(_run())
        assert len(result.turns) == 3
        assert all(not t.completed for t in result.turns)


# ===========================================================================
# Contract 6: End-to-end pipeline consistency
# ===========================================================================


@pytest.mark.e2e
class TestEndToEndPipeline:
    def test_generate_replay_metrics_consistent(self):
        """Generate sessions → run against mock → verify result structure."""
        from agentsurge.generators.synthetic import SyntheticGenerator

        # Step 1: Generate
        sessions = SyntheticGenerator(n_turns=3, tokens_per_turn=100).generate(
            n_sessions=5, seed=42
        )
        assert len(sessions) == 5

        # Step 2: Run against mock backend
        mock_cfg = MockConfig(ttft_ms=50.0, output_tokens=10)
        backend = MockBackend(mock_cfg)

        async def _run():
            runner = BenchmarkRunner(
                BenchmarkConfig(
                    vllm_url="http://x",
                    model="t",
                    no_metrics=True,
                    seed=42,
                ),
                backend=backend,
            )
            async with backend:
                return await runner.run(sessions)

        result = asyncio.run(_run())

        # Step 3: Verify consistency
        assert isinstance(result, RunResult)
        assert len(result.sessions) == 5
        assert all(s.completed for s in result.sessions), "All sessions should succeed against mock"

        # Every session should have 3 turns
        for s in result.sessions:
            assert s.n_turns == 3, f"Session {s.session_id} has {s.n_turns} turns, expected 3"

        # TTFT values should be populated and positive
        ttft_vals = result.ttft_values()
        assert len(ttft_vals) == 15  # 5 sessions * 3 turns
        assert all(v > 0 for v in ttft_vals)

    def test_result_serialization_round_trip(self):
        """RunResult.to_dict() should produce valid JSON with all key fields."""
        from agentsurge.generators.synthetic import SyntheticGenerator

        sessions = SyntheticGenerator(n_turns=2, tokens_per_turn=50).generate(n_sessions=2, seed=42)
        mock_cfg = MockConfig(ttft_ms=30.0, output_tokens=5)
        backend = MockBackend(mock_cfg)

        async def _run():
            runner = BenchmarkRunner(
                BenchmarkConfig(
                    vllm_url="http://x",
                    model="t",
                    no_metrics=True,
                    seed=42,
                ),
                backend=backend,
            )
            async with backend:
                return await runner.run(sessions)

        result = asyncio.run(_run())
        d = result.to_dict()

        # Key fields must exist
        assert "sessions" in d
        assert "ttft_values" in d
        assert "total_elapsed_s" in d
        assert "config" in d
        assert d["config"]["n_sessions"] == 2

        # Serializable to JSON
        import json

        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_isl_osl_ratio_populated(self):
        """End-to-end: isl_osl_ratio should be computed from actual token counts."""
        from agentsurge.generators.synthetic import SyntheticGenerator

        sessions = SyntheticGenerator(n_turns=2, tokens_per_turn=100).generate(
            n_sessions=3, seed=42
        )
        mock_cfg = MockConfig(ttft_ms=20.0, output_tokens=20)
        backend = MockBackend(mock_cfg)

        async def _run():
            runner = BenchmarkRunner(
                BenchmarkConfig(
                    vllm_url="http://x",
                    model="t",
                    no_metrics=True,
                    seed=42,
                ),
                backend=backend,
            )
            async with backend:
                return await runner.run(sessions)

        result = asyncio.run(_run())
        d = result.to_dict()

        assert "isl_osl_ratio" in d
        expected_ratio = 0.0
        if result.osl_total > 0:
            expected_ratio = round(result.isl_total / result.osl_total, 1)
        assert d["isl_osl_ratio"] == pytest.approx(expected_ratio)


# ===========================================================================
# Contract 8: requests_per_s metric
# ===========================================================================


class TestRequestsPerSecond:
    def test_requests_per_s_in_to_dict(self):
        """RunResult.to_dict() should include requests_per_s."""
        t = TurnResult(
            session_id="s1",
            turn_index=0,
            completed=True,
            ttft_ms=100.0,
            total_ms=200.0,
            output_tokens=50,
        )
        sess = SessionResult(session_id="s1", turns=[t], total_ms=200.0)
        rr = RunResult(sessions=[sess], total_elapsed_s=2.0, config={"model": "t"})
        d = rr.to_dict()
        assert "requests_per_s" in d
        assert d["requests_per_s"] == pytest.approx(0.5)  # 1 turn / 2s

    def test_requests_per_s_zero_elapsed(self):
        """Zero elapsed time → 0.0 (no division by zero)."""
        t = TurnResult(session_id="s1", turn_index=0, completed=True, ttft_ms=100.0)
        sess = SessionResult(session_id="s1", turns=[t])
        rr = RunResult(sessions=[sess], total_elapsed_s=0.0)
        assert rr.requests_per_s == 0.0

    def test_requests_per_s_multi_session(self):
        """Multiple sessions with multiple turns."""
        turns_a = [TurnResult(session_id="a", turn_index=i, completed=True) for i in range(3)]
        turns_b = [TurnResult(session_id="b", turn_index=i, completed=True) for i in range(2)]
        sess_a = SessionResult(session_id="a", turns=turns_a)
        sess_b = SessionResult(session_id="b", turns=turns_b)
        rr = RunResult(sessions=[sess_a, sess_b], total_elapsed_s=5.0)
        assert rr.requests_per_s == pytest.approx(1.0)  # 5 turns / 5s

    def test_requests_per_s_e2e(self):
        """End-to-end: requests_per_s populated from mock backend run."""
        from agentsurge.generators.synthetic import SyntheticGenerator

        sessions = SyntheticGenerator(n_turns=2, tokens_per_turn=50).generate(n_sessions=4, seed=42)
        mock_cfg = MockConfig(ttft_ms=10.0, output_tokens=5)
        backend = MockBackend(mock_cfg)

        async def _run():
            runner = BenchmarkRunner(
                BenchmarkConfig(
                    vllm_url="http://x",
                    model="t",
                    no_metrics=True,
                    seed=42,
                ),
                backend=backend,
            )
            async with backend:
                return await runner.run(sessions)

        result = asyncio.run(_run())
        n_total_turns = sum(s.n_turns for s in result.sessions)
        expected_rps = n_total_turns / result.total_elapsed_s
        assert result.requests_per_s == pytest.approx(expected_rps, rel=0.5)
        d = result.to_dict()
        assert d["requests_per_s"] == pytest.approx(expected_rps, rel=0.5)


# ===========================================================================
# Contract 9: fit_workload_stats extracts correct distributions
# ===========================================================================


class TestFitWorkloadStats:
    def _make_corpus(self, tmp_path, n_sessions=10, turns_per_session=5):
        """Create a minimal corpus file for testing."""
        import json

        sessions = []
        for i in range(n_sessions):
            turn_messages = []
            for t in range(turns_per_session):
                msgs = [{"role": "system", "content": "You are a coding assistant."}]
                for prev_t in range(t):
                    msgs.append({"role": "user", "content": f"question {prev_t} " + "x" * 100})
                    msgs.append({"role": "assistant", "content": f"answer {prev_t} " + "y" * 80})
                msgs.append({"role": "user", "content": f"question {t} " + "x" * 100})
                turn_messages.append(msgs)
            sessions.append({"session_id": f"s{i}", "turn_messages": turn_messages, "metadata": {}})
        corpus = {"version": "1.0", "sessions": sessions}
        p = tmp_path / "corpus.json"
        p.write_text(json.dumps(corpus))
        return str(p)

    def test_basic_extraction(self, tmp_path):
        from agentsurge.generators.trace_pool import TracePool

        corpus = self._make_corpus(tmp_path, n_sessions=20, turns_per_session=5)
        stats = TracePool.fit_workload_stats([corpus])

        assert stats["n_sessions"] == 20
        assert stats["turn_count"]["mean"] == 5.0
        assert stats["turn_count"]["median"] == 5
        assert stats["tokens_per_turn"]["mean"] > 0
        assert "recommended_config" in stats

    def test_recommended_config_has_required_fields(self, tmp_path):
        from agentsurge.generators.trace_pool import TracePool

        corpus = self._make_corpus(tmp_path, n_sessions=10, turns_per_session=7)
        stats = TracePool.fit_workload_stats([corpus])
        rc = stats["recommended_config"]

        assert "n_turns" in rc
        assert "tokens_per_turn" in rc
        assert rc["n_turns"] == 7  # median of constant 7

    def test_prefix_sharing_detected(self, tmp_path):
        """All sessions share the same system prompt → 1 unique prefix."""
        from agentsurge.generators.trace_pool import TracePool

        corpus = self._make_corpus(tmp_path, n_sessions=15)
        stats = TracePool.fit_workload_stats([corpus])

        ps = stats["prefix_sharing"]
        assert ps["unique_prefixes"] == 1
        assert ps["mean_sessions_per_prefix"] == 15.0

    def test_variable_turn_counts(self, tmp_path):
        """Sessions with different turn counts should produce varied stats."""
        import json

        sessions = []
        for i in range(20):
            n_turns = (i % 5) + 1  # 1, 2, 3, 4, 5, 1, 2, ...
            turn_messages = [
                [{"role": "user", "content": f"q{t} " + "a" * 50}] for t in range(n_turns)
            ]
            sessions.append({"session_id": f"s{i}", "turn_messages": turn_messages, "metadata": {}})
        p = tmp_path / "varied.json"
        p.write_text(json.dumps({"version": "1.0", "sessions": sessions}))

        from agentsurge.generators.trace_pool import TracePool

        stats = TracePool.fit_workload_stats([str(p)])
        assert stats["turn_count"]["min"] == 1
        assert stats["turn_count"]["max"] == 5
        assert stats["turn_count"]["mean"] == 3.0

    def test_empty_corpus(self, tmp_path):
        """Empty corpus should return minimal stats without crashing."""
        import json

        p = tmp_path / "empty.json"
        p.write_text(json.dumps({"version": "1.0", "sessions": []}))

        from agentsurge.generators.trace_pool import TracePool

        stats = TracePool.fit_workload_stats([str(p)])
        assert stats["n_sessions"] == 0
        assert "turn_count" not in stats  # not enough data to fit
