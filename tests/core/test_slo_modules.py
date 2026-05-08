"""Tests for SLO modules: _slo_core, _slo_search, _slo_compare, slo CLI dispatch."""

import json
import os
from unittest.mock import AsyncMock, patch

import pytest

from agentsurge.cli._slo_compare import (
    format_interleaved_table,
    load_config_profile,
    save_comparison_json,
)
from agentsurge.cli._slo_core import check_slo, next_n, slo_binary_search
from agentsurge.types.results import SloComparisonResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_probe_result(
    n: int,
    ttft_p95_ms: float = 500.0,
    kv_pct: float = 30.0,
    error_rate: float = 0.0,
) -> dict:
    return {
        "n": n,
        "kv_pct": kv_pct,
        "ttft_p50_ms": ttft_p95_ms * 0.6,
        "ttft_p95_ms": ttft_p95_ms,
        "n_err": round(n * error_rate),
        "n_completed": n - round(n * error_rate),
        "error_rate": error_rate,
        "elapsed_s": 5.0,
    }


# ---------------------------------------------------------------------------
# _slo_core: check_slo / next_n
# ---------------------------------------------------------------------------


class TestCheckSlo:
    def test_ok_when_below_threshold(self):
        r = _make_probe_result(10, ttft_p95_ms=900.0, error_rate=0.0)
        assert check_slo(r, 1000.0) is True

    def test_breach_when_above_threshold(self):
        r = _make_probe_result(10, ttft_p95_ms=1500.0, error_rate=0.0)
        assert check_slo(r, 1000.0) is False

    def test_breach_when_high_error_rate(self):
        r = _make_probe_result(10, ttft_p95_ms=500.0, error_rate=0.06)
        assert check_slo(r, 1000.0) is False

    def test_ok_at_error_boundary(self):
        r = _make_probe_result(10, ttft_p95_ms=500.0, error_rate=0.05)
        assert check_slo(r, 1000.0) is True

    def test_custom_max_error_rate(self):
        r = _make_probe_result(10, ttft_p95_ms=500.0, error_rate=0.15)
        assert check_slo(r, 1000.0, max_error_rate=0.20) is True
        assert check_slo(r, 1000.0, max_error_rate=0.10) is False

    def test_ttft_at_exact_threshold_fails(self):
        r = _make_probe_result(10, ttft_p95_ms=1000.0, error_rate=0.0)
        assert check_slo(r, 1000.0) is False


class TestNextN:
    def test_low_kv_large_step(self):
        result = next_n(50, 20.0)
        assert result == 100  # min(50*2, 50+100) = 100

    def test_mid_kv_medium_step(self):
        result = next_n(50, 45.0)
        assert result == 70  # 50 + 20

    def test_high_kv_small_step(self):
        result = next_n(50, 75.0)
        assert result == 60  # 50 + 10


# ---------------------------------------------------------------------------
# _slo_core: slo_binary_search
# ---------------------------------------------------------------------------


class TestSloBinarySearch:
    @pytest.mark.asyncio
    async def test_finds_boundary(self):
        """Probe passes for n<=100, fails for n>100. Boundary should be near 100."""

        async def probe(n: int) -> dict:
            ok = n <= 100
            return _make_probe_result(
                n,
                ttft_p95_ms=500.0 if ok else 5000.0,
                kv_pct=min(n * 0.8, 99.0),
            )

        result = await slo_binary_search(
            probe,
            slo_ms=1000.0,
            max_n_cap=200,
            start_n=10,
        )
        assert 95 <= result["max_n"] <= 100
        assert result["boundary"] is not None
        assert result["boundary"]["ttft_p95_ms"] < 1000.0
        assert isinstance(result["scan"], list)
        assert len(result["scan"]) > 0

    @pytest.mark.asyncio
    async def test_all_pass_returns_max_cap(self):
        """When every probe passes, max_n should be the highest probed value."""

        async def probe(n: int) -> dict:
            return _make_probe_result(n, ttft_p95_ms=200.0, kv_pct=10.0)

        result = await slo_binary_search(
            probe,
            slo_ms=1000.0,
            max_n_cap=200,
            start_n=10,
        )
        assert result["max_n"] > 0
        assert all(s["ttft_p95_ms"] < 1000.0 for s in result["scan"])

    @pytest.mark.asyncio
    async def test_all_fail_returns_zero(self):
        """When the first probe already breaches, max_n should be 0."""

        async def probe(n: int) -> dict:
            return _make_probe_result(n, ttft_p95_ms=5000.0)

        result = await slo_binary_search(
            probe,
            slo_ms=1000.0,
            max_n_cap=200,
            start_n=10,
        )
        assert result["max_n"] == 0

    @pytest.mark.asyncio
    async def test_hint_skips_scan(self):
        """With a hint, binary search starts in the hinted range."""
        probed_ns: list[int] = []

        async def probe(n: int) -> dict:
            probed_ns.append(n)
            ok = n <= 50
            return _make_probe_result(
                n,
                ttft_p95_ms=400.0 if ok else 4000.0,
                kv_pct=min(n, 99.0),
            )

        result = await slo_binary_search(
            probe,
            slo_ms=1000.0,
            max_n_cap=200,
            start_n=10,
            hint=50,
        )
        assert result["max_n"] == 50
        assert all(n >= 25 for n in probed_ns)

    @pytest.mark.asyncio
    async def test_max_error_rate_propagated(self):
        """Custom max_error_rate flows through to _is_ok inside binary search."""

        async def probe(n: int) -> dict:
            return _make_probe_result(n, ttft_p95_ms=500.0, kv_pct=30.0, error_rate=0.08)

        result_lenient = await slo_binary_search(
            probe,
            slo_ms=5000.0,
            max_n_cap=50,
            start_n=10,
            max_error_rate=0.10,
        )
        assert result_lenient["max_n"] > 0

        result_strict = await slo_binary_search(
            probe,
            slo_ms=5000.0,
            max_n_cap=50,
            start_n=10,
            max_error_rate=0.05,
        )
        assert result_strict["max_n"] == 0

    @pytest.mark.asyncio
    async def test_scan_results_sorted_by_n(self):
        """The scan list in the result should be sorted by n."""

        async def probe(n: int) -> dict:
            ok = n <= 60
            return _make_probe_result(
                n,
                ttft_p95_ms=300.0 if ok else 3000.0,
                kv_pct=min(n * 1.2, 99.0),
            )

        result = await slo_binary_search(
            probe,
            slo_ms=1000.0,
            max_n_cap=150,
            start_n=10,
        )
        ns = [s["n"] for s in result["scan"]]
        assert ns == sorted(ns)


# ---------------------------------------------------------------------------
# _slo_core: slo_probe (mocked runner)
# ---------------------------------------------------------------------------


class TestSloProbe:
    @pytest.mark.asyncio
    async def test_probe_forwards_tokenizer_kwargs(self):
        """slo_probe passes tokenizer_model, trust_remote_code, and ignore_replay_output_length=True to BenchmarkConfig."""
        from agentsurge.types.results import RunResult, SessionResult, TurnResult

        turn = TurnResult(
            session_id="s1",
            turn_index=0,
            completed=True,
            ttft_ms=100.0,
            total_ms=150.0,
            input_tokens=50,
            output_tokens=20,
            interrupted=False,
        )
        sess = SessionResult(session_id="s1", turns=[turn])
        run_result = RunResult(sessions=[sess], total_elapsed_s=1.0)
        run_result.kv_util_peak = 0.1

        mock_runner_instance = AsyncMock()
        mock_runner_instance.run = AsyncMock(return_value=run_result)

        captured_configs: list = []

        def _fake_runner(cfg, **kwargs):
            captured_configs.append(cfg)
            return mock_runner_instance

        with (
            patch("agentsurge.cli._slo_core.BenchmarkRunner", side_effect=_fake_runner),
            patch("agentsurge.cli._slo_core._generate_sessions", return_value=[{"turns": []}]),
        ):
            from agentsurge.cli._slo_core import slo_probe

            await slo_probe(
                vllm_url="http://fake:8000",
                model="m",
                n_sess=1,
                n_turns=1,
                tokens_per_turn=100,
                max_tokens=50,
                tokenizer_model="custom/tok",
                trust_remote_code=True,
            )

        assert len(captured_configs) == 1
        cfg = captured_configs[0]
        assert cfg.tokenizer_model == "custom/tok"
        assert cfg.trust_remote_code is True
        assert cfg.ignore_replay_output_length is True

    @pytest.mark.asyncio
    async def test_probe_defaults_ignore_replay_output_length_true(self):
        """slo_probe always sets ignore_replay_output_length=True even without tokenizer kwargs."""
        from agentsurge.types.results import RunResult, SessionResult, TurnResult

        turn = TurnResult(
            session_id="s1",
            turn_index=0,
            completed=True,
            ttft_ms=100.0,
            total_ms=150.0,
            input_tokens=50,
            output_tokens=20,
            interrupted=False,
        )
        sess = SessionResult(session_id="s1", turns=[turn])
        run_result = RunResult(sessions=[sess], total_elapsed_s=1.0)
        run_result.kv_util_peak = 0.1

        mock_runner_instance = AsyncMock()
        mock_runner_instance.run = AsyncMock(return_value=run_result)

        captured_configs: list = []

        def _fake_runner(cfg, **kwargs):
            captured_configs.append(cfg)
            return mock_runner_instance

        with (
            patch("agentsurge.cli._slo_core.BenchmarkRunner", side_effect=_fake_runner),
            patch("agentsurge.cli._slo_core._generate_sessions", return_value=[{"turns": []}]),
        ):
            from agentsurge.cli._slo_core import slo_probe

            await slo_probe(
                vllm_url="http://fake:8000",
                model="m",
                n_sess=1,
                n_turns=1,
                tokens_per_turn=100,
                max_tokens=50,
            )

        assert len(captured_configs) == 1
        cfg = captured_configs[0]
        assert cfg.ignore_replay_output_length is True

    @pytest.mark.asyncio
    async def test_probe_injects_backend_with_full_config(self):
        """slo_probe forwards API/backend config into the explicit backend path."""
        from agentsurge.backends.vllm import VllmBackend, VllmConfig
        from agentsurge.types.results import RunResult, SessionResult, TurnResult

        turn = TurnResult(
            session_id="s1",
            turn_index=0,
            completed=True,
            ttft_ms=100.0,
            total_ms=150.0,
            input_tokens=50,
            output_tokens=20,
        )
        run_result = RunResult(
            sessions=[SessionResult(session_id="s1", turns=[turn])],
            total_elapsed_s=1.0,
        )
        run_result.kv_util_peak = 0.1

        mock_runner_instance = AsyncMock()
        mock_runner_instance.run = AsyncMock(return_value=run_result)
        captured = {}

        def _fake_runner(cfg, **kwargs):
            captured["cfg"] = cfg
            captured["backend"] = kwargs.get("backend")
            return mock_runner_instance

        with (
            patch("agentsurge.cli._slo_core.BenchmarkRunner", side_effect=_fake_runner),
            patch("agentsurge.cli._slo_core._generate_sessions", return_value=[{"turns": []}]),
        ):
            from agentsurge.cli._slo_core import slo_probe

            await slo_probe(
                vllm_url="http://fake:8000",
                model="m",
                n_sess=1,
                n_turns=1,
                tokens_per_turn=100,
                max_tokens=50,
                extra_body={"top_k": 20},
                api_type="responses",
                backend_name="vllm",
                request_timeout=17,
                stream_idle_timeout=3.0,
                enable_thinking=True,
            )

        backend = captured["backend"]
        assert isinstance(backend, VllmBackend)
        assert isinstance(backend.config, VllmConfig)
        assert backend.config.api_type == "responses"
        assert backend.config.extra_body == {"top_k": 20}
        assert backend.config.request_timeout == 17
        assert backend.config.stream_idle_timeout == 3.0
        assert backend.config.chat_template_kwargs == {"enable_thinking": True}

    @pytest.mark.asyncio
    async def test_probe_none_kv_util_peak_does_not_raise(self):
        """slo_probe must not raise TypeError when kv_util_peak is None."""
        import math

        from agentsurge.types.results import RunResult, SessionResult, TurnResult

        turn = TurnResult(
            session_id="s1",
            turn_index=0,
            completed=True,
            ttft_ms=300.0,
            total_ms=400.0,
            input_tokens=100,
            output_tokens=50,
            interrupted=False,
        )
        sess = SessionResult(session_id="s1", turns=[turn])
        run_result = RunResult(sessions=[sess], total_elapsed_s=1.0)
        run_result.kv_util_peak = None

        mock_runner_instance = AsyncMock()
        mock_runner_instance.run = AsyncMock(return_value=run_result)

        with (
            patch("agentsurge.cli._slo_core.BenchmarkRunner", return_value=mock_runner_instance),
            patch("agentsurge.cli._slo_core._generate_sessions", return_value=[{"turns": []}]),
        ):
            from agentsurge.cli._slo_core import slo_probe

            r = await slo_probe(
                vllm_url="http://fake:8000",
                model="m",
                n_sess=1,
                n_turns=1,
                tokens_per_turn=100,
                max_tokens=50,
            )

        assert math.isnan(r["kv_pct"])


# ---------------------------------------------------------------------------
# _slo_compare: load_config_profile
# ---------------------------------------------------------------------------


class TestLoadConfigProfile:
    def test_load_valid(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "name: test-cfg\nvllm_url: http://localhost:8000\n"
            "model: Qwen/Qwen3.5-27B\ndescription: test\n"
        )
        profile = load_config_profile(str(cfg))
        assert profile.name == "test-cfg"
        assert profile.vllm_url == "http://localhost:8000"
        assert profile.model == "Qwen/Qwen3.5-27B"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config_profile("/nonexistent/path.yaml")

    def test_missing_fields_raises(self, tmp_path):
        cfg = tmp_path / "bad.yaml"
        cfg.write_text("name: only-name\n")
        with pytest.raises(ValueError, match="missing required fields"):
            load_config_profile(str(cfg))


# ---------------------------------------------------------------------------
# _slo_compare: format_interleaved_table
# ---------------------------------------------------------------------------


class TestFormatInterleavedTable:
    def _make_profile_result(self, configs, capacity_ratios=None, pairwise=None):
        return {
            "profile_label": "7Tx2200",
            "slo_ms": 3000.0,
            "workload": {"n_turns": 7, "synthetic_tokens_per_turn": 2200, "max_tokens": 512},
            "configs": configs,
            "capacity_ratios": capacity_ratios or {},
            "pairwise_ratios": pairwise or {},
        }

    def test_two_configs_table(self):
        configs = [
            {"config_name": "A", "max_n": 100, "boundary": {"ttft_p95_ms": 2800.0, "kv_pct": 85.0}},
            {"config_name": "B", "max_n": 150, "boundary": {"ttft_p95_ms": 2900.0, "kv_pct": 90.0}},
        ]
        ratios = {"A": 1.0, "B": 1.5}
        pairwise = {"A_vs_B": 0.667, "B_vs_A": 1.5}
        pr = self._make_profile_result(configs, ratios, pairwise)
        table = format_interleaved_table([pr], 3000.0)

        assert "SLO Target" in table
        assert "7Tx2200" in table
        assert "A" in table
        assert "B" in table
        assert "50% higher" in table

    def test_identical_configs_equal(self):
        configs = [
            {"config_name": "X", "max_n": 80, "boundary": {"ttft_p95_ms": 2500.0, "kv_pct": 70.0}},
            {"config_name": "Y", "max_n": 80, "boundary": {"ttft_p95_ms": 2500.0, "kv_pct": 70.0}},
        ]
        ratios = {"X": 1.0, "Y": 1.0}
        pairwise = {"X_vs_Y": 1.0, "Y_vs_X": 1.0}
        pr = self._make_profile_result(configs, ratios, pairwise)
        table = format_interleaved_table([pr], 3000.0)

        assert "equal capacity" in table


# ---------------------------------------------------------------------------
# _slo_compare: save_comparison_json
# ---------------------------------------------------------------------------


class TestSaveComparisonJson:
    def test_save_interleaved_list(self, tmp_path):
        data = [{"profile_label": "test", "configs": []}]
        path = save_comparison_json(data, str(tmp_path), "t", "20260401")
        assert os.path.exists(path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["mode"] == "interleaved"
        assert loaded["tag"] == "t"

    def test_save_sequential_result(self, tmp_path):
        data = SloComparisonResult(target_slo_ms=3000.0, entries=[{"a": 1}])
        path = save_comparison_json(data, str(tmp_path), "seq", "20260401")
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["mode"] == "sequential"
        assert loaded["target_slo_ms"] == 3000.0


# ---------------------------------------------------------------------------
# _slo_interleaved: no_cooldown forwarding (H6)
# ---------------------------------------------------------------------------


class TestInterleavedNoCooldown:
    """_run_interleaved_slo_search must respect no_cooldown=True."""

    def _make_config_profile(self, name: str = "cfg-a"):
        from agentsurge.types.results import ConfigProfile

        return ConfigProfile(
            name=name,
            vllm_url="http://fake:8000",
            model="fake-model",
            overrides={},
        )

    def _make_probe_result(self, n: int) -> dict:
        return {
            "n": n,
            "kv_pct": 30.0,
            "ttft_p50_ms": 200.0,
            "ttft_p95_ms": 500.0,
            "n_err": 0,
            "n_completed": n,
            "error_rate": 0.0,
            "elapsed_s": 1.0,
        }

    @pytest.mark.asyncio
    async def test_cooldown_called_when_no_cooldown_false(self):
        """When no_cooldown=False (default), _smart_cooldown IS called."""
        cp = self._make_config_profile()
        probe_result = self._make_probe_result(10)

        with (
            patch(
                "agentsurge.cli._slo_interleaved._smart_cooldown",
                new_callable=AsyncMock,
            ) as mock_cooldown,
            patch(
                "agentsurge.cli._slo_interleaved.slo_probe",
                new_callable=AsyncMock,
                return_value=probe_result,
            ),
        ):
            from agentsurge.cli._slo_interleaved import _run_interleaved_slo_search

            await _run_interleaved_slo_search(
                config_profiles=[cp],
                slo_ms=1000.0,
                cooldown=30,
                max_n_cap=20,
                n_turns=1,
                tokens_per_turn=100,
                max_tokens=50,
                profile_label="test",
            )

        mock_cooldown.assert_called()


# ---------------------------------------------------------------------------
# slo.py: CLI dispatch
# ---------------------------------------------------------------------------
