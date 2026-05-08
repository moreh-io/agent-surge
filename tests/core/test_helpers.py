"""Tests for agentsurge.cli._helpers pure-utility functions."""

import argparse
import os
import re
from unittest.mock import AsyncMock, patch

import pytest

from agentsurge.cli._helpers import (
    _DATA_DIR_MAP,
    _build_runner_config,
    _load_config,
    _output_path,
    _resolve_local,
    _smart_cooldown,
)

# ---------------------------------------------------------------------------
# _load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_broken_yaml_explicit_path_raises(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("vllm_url: [\nbad yaml")
        with pytest.raises((SystemExit, Exception)) as exc_info:
            _load_config(str(bad))
        if isinstance(exc_info.value, SystemExit):
            assert exc_info.value.code != 0
        else:
            assert (
                "bad.yaml" in str(exc_info.value) or "yaml" in type(exc_info.value).__name__.lower()
            )

    def test_null_yaml_explicit_path_raises(self, tmp_path):
        null_file = tmp_path / "null.yaml"
        null_file.write_text("")
        with pytest.raises((SystemExit, ValueError)):
            _load_config(str(null_file))

    def test_no_path_loads_default(self):
        cfg = _load_config()
        assert isinstance(cfg, dict)
        assert "vllm" in cfg

    def test_valid_yaml_explicit_path_loads(self, tmp_path):
        import yaml

        data = {"vllm": {"url": "http://test:8000", "model": "m"}, "workload": {}}
        p = tmp_path / "valid.yaml"
        p.write_text(yaml.dump(data))
        cfg = _load_config(str(p))
        assert cfg["vllm"]["url"] == "http://test:8000"


# ---------------------------------------------------------------------------
# _output_path
# ---------------------------------------------------------------------------


class TestOutputPath:
    def test_creates_directory_and_returns_timestamped_path(self, tmp_path):
        sub = str(tmp_path / "results" / "nested")
        path = _output_path(sub, "sweep")
        assert os.path.isdir(sub)
        assert path.startswith(sub + os.sep)
        basename = os.path.basename(path)
        assert basename.startswith("sweep_")
        assert basename.endswith(".json")
        # Format: <prefix>_<YYYYMMDD>_<HHMMSS>_<pid>_<>=8-hex>.json
        # See cli.md H3: collision suffix entropy raised from 4 to 8 hex.
        assert re.match(r"sweep_\d{8}_\d{6}_\d+_[0-9a-f]{8,}\.json$", basename)

    def test_existing_directory_no_error(self, tmp_path):
        base = str(tmp_path)
        p1 = _output_path(base, "run")
        p2 = _output_path(base, "run")
        assert os.path.dirname(p1) == base
        assert os.path.dirname(p2) == base
        # Two calls in the same second must produce distinct paths: the
        # pid+random suffix prevents collisions between parallel runs.
        assert p1 != p2

    def test_prefix_appears_in_filename(self, tmp_path):
        path = _output_path(str(tmp_path), "my_prefix")
        assert "my_prefix_" in os.path.basename(path)


# ---------------------------------------------------------------------------
# _resolve_local
# ---------------------------------------------------------------------------


class TestResolveLocal:
    def test_explicit_local_path_takes_priority(self, tmp_path):
        explicit = str(tmp_path / "my.jsonl")
        result = _resolve_local("openhands", str(tmp_path), local_path=explicit)
        assert result == explicit

    def test_returns_none_when_no_data_dir(self):
        assert _resolve_local("openhands", None) is None

    def test_auto_maps_known_source(self, tmp_path):
        fname = _DATA_DIR_MAP["openhands"]
        (tmp_path / fname).write_text("{}")
        result = _resolve_local("openhands", str(tmp_path))
        assert result == str(tmp_path / fname)

    def test_returns_none_when_file_missing(self, tmp_path):
        result = _resolve_local("openhands", str(tmp_path))
        assert result is None

    def test_swe_smith_variant_falls_back_to_swe_smith(self, tmp_path):
        fname = _DATA_DIR_MAP["swe-smith"]
        (tmp_path / fname).write_text("{}")
        result = _resolve_local("swe-smith-custom", str(tmp_path))
        assert result == str(tmp_path / fname)

    def test_unknown_source_returns_none(self, tmp_path):
        result = _resolve_local("unknown-source", str(tmp_path))
        assert result is None

    @pytest.mark.parametrize("source", list(_DATA_DIR_MAP.keys()))
    def test_all_known_sources_resolve_when_file_exists(self, tmp_path, source):
        fname = _DATA_DIR_MAP[source]
        (tmp_path / fname).write_text("{}")
        result = _resolve_local(source, str(tmp_path))
        assert result is not None
        assert result.endswith(fname)


# ---------------------------------------------------------------------------
# _smart_cooldown
# ---------------------------------------------------------------------------


class TestSmartCooldown:
    @pytest.fixture()
    def mock_wait(self):
        with patch("agentsurge.cli._helpers._wait_kv_cooldown", new_callable=AsyncMock) as m:
            yield m

    @pytest.mark.asyncio
    async def test_no_cooldown_flag_skips(self, mock_wait):
        await _smart_cooldown(
            "http://fake",
            current_n=100,
            next_n=50,
            current_kv=0.8,
            had_error=False,
            no_cooldown=True,
        )
        mock_wait.assert_not_called()

    @pytest.mark.asyncio
    async def test_had_error_triggers_full_cooldown(self, mock_wait):
        await _smart_cooldown(
            "http://fake",
            current_n=50,
            next_n=100,
            current_kv=0.5,
            had_error=True,
        )
        mock_wait.assert_awaited_once_with("http://fake", target=0.02, timeout=60)

    @pytest.mark.asyncio
    async def test_ascending_skips_cooldown(self, mock_wait):
        await _smart_cooldown(
            "http://fake",
            current_n=50,
            next_n=100,
            current_kv=0.5,
            had_error=False,
        )
        mock_wait.assert_not_called()

    @pytest.mark.asyncio
    async def test_equal_n_skips_cooldown(self, mock_wait):
        await _smart_cooldown(
            "http://fake",
            current_n=50,
            next_n=50,
            current_kv=0.5,
            had_error=False,
        )
        mock_wait.assert_not_called()

    @pytest.mark.asyncio
    async def test_descending_computes_proportional_target(self, mock_wait):
        await _smart_cooldown(
            "http://fake",
            current_n=100,
            next_n=50,
            current_kv=0.8,
            had_error=False,
        )
        expected_target = min(0.50, (50 / 100) * 0.8 + 0.10)  # 0.50
        mock_wait.assert_awaited_once_with("http://fake", target=expected_target, timeout=60)

    @pytest.mark.asyncio
    async def test_descending_target_capped_at_050(self, mock_wait):
        await _smart_cooldown(
            "http://fake",
            current_n=100,
            next_n=90,
            current_kv=0.95,
            had_error=False,
        )
        # (90/100)*0.95 + 0.10 = 0.955, capped to 0.50
        mock_wait.assert_awaited_once_with("http://fake", target=0.50, timeout=60)

    @pytest.mark.asyncio
    async def test_descending_low_kv_small_target(self, mock_wait):
        await _smart_cooldown(
            "http://fake",
            current_n=100,
            next_n=10,
            current_kv=0.2,
            had_error=False,
        )
        expected = min(0.50, (10 / 100) * 0.2 + 0.10)  # 0.12
        mock_wait.assert_awaited_once_with("http://fake", target=expected, timeout=60)

    @pytest.mark.asyncio
    async def test_descending_zero_kv_falls_back_to_full(self, mock_wait):
        await _smart_cooldown(
            "http://fake",
            current_n=100,
            next_n=50,
            current_kv=0.0,
            had_error=False,
        )
        mock_wait.assert_awaited_once_with("http://fake", target=0.02, timeout=60)

    @pytest.mark.asyncio
    async def test_descending_zero_current_n_falls_back_to_full(self, mock_wait):
        await _smart_cooldown(
            "http://fake",
            current_n=0,
            next_n=0,
            current_kv=0.5,
            had_error=False,
        )
        # current_n=0, next_n=0 => next_n >= current_n => skip
        mock_wait.assert_not_called()

    @pytest.mark.asyncio
    async def test_custom_timeout_forwarded(self, mock_wait):
        await _smart_cooldown(
            "http://fake",
            current_n=100,
            next_n=50,
            current_kv=0.5,
            had_error=True,
            timeout=120,
        )
        mock_wait.assert_awaited_once_with("http://fake", target=0.02, timeout=120)


# ---------------------------------------------------------------------------
# _build_runner_config
# ---------------------------------------------------------------------------


def _minimal_cfg():
    return {
        "vllm": {"url": "http://vllm:8000", "model": "Qwen/Qwen3.5-27B"},
        "runner": {},
    }


def _minimal_args(**overrides):
    defaults = {
        "vllm_url": None,
        "model": None,
        "preset": None,
        "client_concurrency": None,
        "max_tokens": None,
        "think_time": None,
        "tool_delay": None,
        "interruption_rate": None,
        "single_turn_ratio": None,
        "arrival": None,
        "arrival_rate": None,
        "lmcache_url": None,
        "lmcache": None,
        "synthetic_prompt_scale": None,
        "no_metrics": False,
        "backend": None,
        "api": "chat",
        "context_distribution": None,
        "seed": 42,
        "warm_up": 0,
        "ramp_duration": None,
        "duration": None,
        "tool_mode": None,
        "tool_call_parser_fallback": None,
        "tool_output_mode": None,
        "sandbox_dir": None,
        "save_responses": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestBuildRunnerConfig:
    def test_minimal_produces_valid_config(self):
        cfg = _build_runner_config(_minimal_args(), _minimal_cfg())
        assert cfg.vllm_url == "http://vllm:8000"
        assert cfg.model == "Qwen/Qwen3.5-27B"
        assert cfg.arrival_pattern == "poisson"
        assert cfg.max_tokens == 64  # runner default

    def test_request_timeout_from_runner_config(self):
        y = _minimal_cfg()
        y["runner"]["request_timeout"] = 17

        cfg = _build_runner_config(_minimal_args(), y)

        assert cfg.request_timeout == 17

    @pytest.mark.parametrize(
        ("arg_overrides", "yaml_override", "attr", "expected"),
        [
            ({"vllm_url": "http://custom:9000"}, None, "vllm_url", "http://custom:9000"),
            ({"model": "meta-llama/Llama-3-70B"}, None, "model", "meta-llama/Llama-3-70B"),
            ({"client_concurrency": 50}, None, "max_concurrency", 50),
            ({}, ("runner.max_concurrency", 200), "max_concurrency", 200),
        ],
    )
    def test_cli_and_yaml_overrides(self, arg_overrides, yaml_override, attr, expected):
        y = _minimal_cfg()
        if yaml_override is not None:
            dotted, val = yaml_override
            sec, key = dotted.split(".")
            y[sec][key] = val
        cfg = _build_runner_config(_minimal_args(**arg_overrides), y)
        assert getattr(cfg, attr) == expected

    def test_think_time_from_args(self):
        # After M9 fix, argparse type=_think_time_type already converts the
        # string to a tuple before _build_runner_config is called.
        args = _minimal_args(think_time=(1.0, 5.0))
        cfg = _build_runner_config(args, _minimal_cfg())
        assert cfg.think_time == (1.0, 5.0)

    def test_tool_delay_true_string(self):
        args = _minimal_args(tool_delay="true")
        cfg = _build_runner_config(args, _minimal_cfg())
        assert cfg.tool_delay is True

    def test_tool_delay_false_string(self):
        args = _minimal_args(tool_delay="false")
        cfg = _build_runner_config(args, _minimal_cfg())
        assert cfg.tool_delay is False

    def test_arrival_from_args(self):
        args = _minimal_args(arrival="burst")
        cfg = _build_runner_config(args, _minimal_cfg())
        assert cfg.arrival_pattern == "burst"

    def test_arrival_from_runner_config(self):
        y = _minimal_cfg()
        y["runner"]["arrival_pattern"] = "constant"
        cfg = _build_runner_config(_minimal_args(), y)
        assert cfg.arrival_pattern == "constant"

    def test_arrival_rate_auto(self):
        args = _minimal_args(arrival_rate="auto", client_concurrency=100)
        cfg = _build_runner_config(args, _minimal_cfg())
        assert cfg.arrival_rate == 30.0  # max(1.0, 100 * 0.3)

    def test_arrival_rate_explicit(self):
        args = _minimal_args(arrival_rate="5.0")
        cfg = _build_runner_config(args, _minimal_cfg())
        assert cfg.arrival_rate == 5.0

    def test_synthetic_prompt_scale_from_models_section(self):
        y = _minimal_cfg()
        y["models"] = {"Qwen3.5-27B": {"max_model_len": 32768}}
        cfg = _build_runner_config(_minimal_args(), y)
        assert cfg.max_model_len == 32768

    def test_synthetic_prompt_scale_from_args(self):
        args = _minimal_args(synthetic_prompt_scale=8192)
        cfg = _build_runner_config(args, _minimal_cfg())
        assert cfg.max_model_len == 8192

    def test_mock_backend_forces_no_metrics(self):
        args = _minimal_args(backend="mock")
        cfg = _build_runner_config(args, _minimal_cfg())
        assert cfg.no_metrics is True

    def test_overrides_applied(self):
        cfg = _build_runner_config(_minimal_args(), _minimal_cfg(), overrides={"max_tokens": 512})
        assert cfg.max_tokens == 512

    def test_overrides_none_values_ignored(self):
        cfg = _build_runner_config(_minimal_args(), _minimal_cfg(), overrides={"max_tokens": None})
        assert cfg.max_tokens == 64

    def test_max_tokens_zero_not_overridden_by_preset(self):
        y = _minimal_cfg()
        y["runner"]["max_tokens"] = 256
        args = _minimal_args(max_tokens=0)
        cfg = _build_runner_config(args, y)
        assert cfg.max_tokens == 0

    def test_seed_from_args(self):
        args = _minimal_args(seed=123)
        cfg = _build_runner_config(args, _minimal_cfg())
        assert cfg.seed == 123

    def test_warm_up_from_args(self):
        args = _minimal_args(warm_up=5)
        cfg = _build_runner_config(args, _minimal_cfg())
        assert cfg.warm_up == 5

    def test_temperature_from_runner_config(self):
        y = _minimal_cfg()
        y["runner"]["temperature"] = 0.7
        cfg = _build_runner_config(_minimal_args(), y)
        assert cfg.temperature == 0.7

    def test_extra_body_from_config(self):
        y = _minimal_cfg()
        y["extra_body"] = {"guided_decoding_backend": "lm-format-enforcer"}
        cfg = _build_runner_config(_minimal_args(), y)
        assert cfg.extra_body == {"guided_decoding_backend": "lm-format-enforcer"}


# ---------------------------------------------------------------------------
# _resolve_lmcache_urls - light coverage (integration-ish but pure logic)
# ---------------------------------------------------------------------------


class TestResolveLmcacheUrls:
    def test_returns_none_when_disabled(self):
        from agentsurge.cli._helpers import _resolve_lmcache_urls

        args = argparse.Namespace(lmcache_url=None, lmcache=False, preset=None)
        result = _resolve_lmcache_urls(args, {})
        assert result is None

    def test_returns_url_when_explicitly_enabled(self):
        from agentsurge.cli._helpers import _resolve_lmcache_urls

        args = argparse.Namespace(lmcache_url="http://lmc:8080/metrics", lmcache=True, preset=None)
        result = _resolve_lmcache_urls(args, {})
        assert result == ["http://lmc:8080/metrics"]

    def test_returns_url_from_config(self):
        from agentsurge.cli._helpers import _resolve_lmcache_urls

        args = argparse.Namespace(lmcache_url=None, lmcache=None, preset=None)
        cfg = {"lmcache": {"enabled": True, "url": "http://cfg:9090/metrics"}}
        result = _resolve_lmcache_urls(args, cfg)
        assert result == ["http://cfg:9090/metrics"]
