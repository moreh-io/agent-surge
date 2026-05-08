"""Tests for _load_tokenizer local-path error and HF-id soft-fail behavior."""

from __future__ import annotations

import logging
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

import agentsurge.runner as runner_mod
from agentsurge.runner import BenchmarkRunner, _load_tokenizer
from agentsurge.types.results import BenchmarkConfig


@pytest.fixture(autouse=True)
def clear_tokenizer_cache():
    runner_mod._tokenizer_cache.clear()
    yield
    runner_mod._tokenizer_cache.clear()


def _make_fake_transformers(from_pretrained_side_effect=None, from_pretrained_return=None):
    """Build a minimal fake transformers module with AutoTokenizer."""
    fake_auto = MagicMock()
    if from_pretrained_side_effect is not None:
        fake_auto.from_pretrained.side_effect = from_pretrained_side_effect
    elif from_pretrained_return is not None:
        fake_auto.from_pretrained.return_value = from_pretrained_return

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = fake_auto
    return fake_transformers, fake_auto


class TestLoadTokenizer:
    def test_hf_id_failure_warns_and_returns_none(self, caplog):
        fake_transformers, _ = _make_fake_transformers(
            from_pretrained_side_effect=OSError("model not found")
        )
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            with caplog.at_level(logging.WARNING, logger="agentsurge.runner"):
                result = _load_tokenizer("org/model")

        assert result is None
        assert any("Could not load tokenizer" in r.message for r in caplog.records)

    def test_absolute_path_failure_raises_runtimeerror(self):
        fake_transformers, _ = _make_fake_transformers(
            from_pretrained_side_effect=OSError("no tokenizer files")
        )
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            with pytest.raises(RuntimeError, match=r"(?s)/tmp/nonexistent-model.*--tokenizer"):
                _load_tokenizer("/tmp/nonexistent-model")

    def test_dot_slash_path_failure_raises_runtimeerror(self):
        fake_transformers, _ = _make_fake_transformers(
            from_pretrained_side_effect=OSError("no tokenizer files")
        )
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            with pytest.raises(RuntimeError, match=r"--tokenizer"):
                _load_tokenizer("./models/local")

    def test_dot_dot_slash_path_failure_raises_runtimeerror(self):
        fake_transformers, _ = _make_fake_transformers(
            from_pretrained_side_effect=OSError("no tokenizer files")
        )
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            with pytest.raises(RuntimeError, match=r"--tokenizer"):
                _load_tokenizer("../models/local")

    def test_local_path_failure_does_not_cache(self):
        fake_transformers, _ = _make_fake_transformers(
            from_pretrained_side_effect=OSError("no tokenizer files")
        )
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            with pytest.raises(RuntimeError):
                _load_tokenizer("/tmp/bad-model")

        assert ("/tmp/bad-model", False) not in runner_mod._tokenizer_cache

    def test_local_path_exception_is_chained(self):
        original = OSError("no tokenizer files")
        fake_transformers, _ = _make_fake_transformers(from_pretrained_side_effect=original)
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            with pytest.raises(RuntimeError) as exc_info:
                _load_tokenizer("/tmp/bad-model")

        assert exc_info.value.__cause__ is original


class TestBenchmarkRunnerTokenizerGuard:
    """Regression tests: T1+T2 interaction fix.

    T1 passes ignore_replay_output_length=True from slo_probe.
    T2 raises RuntimeError for local-path tokenizer failures.
    Without the guard, these cancel out: the crash remains.
    """

    def test_runner_skips_tokenizer_when_ignore_replay_output_length(self):
        """BenchmarkRunner must not call _load_tokenizer when ignore_replay_output_length=True.

        Regression: T1+T2 interaction. SLO probes set ignore_replay_output_length=True
        and may pass a local model path with no tokenizer files. The runner must not
        trigger T2's RuntimeError in that case.
        """
        mock_load = MagicMock(side_effect=RuntimeError("should not be called"))
        cfg = BenchmarkConfig(
            vllm_url="http://localhost:8000",
            model="/nonexistent/local/path",
            ignore_replay_output_length=True,
        )
        with patch("agentsurge.runner._load_tokenizer", mock_load):
            runner = BenchmarkRunner(cfg)

        mock_load.assert_not_called()
        assert runner._tokenizer is None

    def test_runner_loads_tokenizer_when_replay_sizing_required(self):
        """Ensure the guard doesn't break the default path (ignore_replay_output_length=False)."""
        sentinel = MagicMock()
        mock_load = MagicMock(return_value=sentinel)
        cfg = BenchmarkConfig(
            vllm_url="http://localhost:8000",
            model="org/some-model",
            ignore_replay_output_length=False,
        )
        with patch("agentsurge.runner._load_tokenizer", mock_load):
            runner = BenchmarkRunner(cfg)

        mock_load.assert_called_once()
        assert runner._tokenizer is sentinel

    def test_runner_skips_tokenizer_when_skip_tokenizer_load_true_and_ignore_replay(self):
        """skip_tokenizer_load=True + ignore_replay_output_length=True: no load, no raise."""
        mock_load = MagicMock(side_effect=RuntimeError("should not be called"))
        cfg = BenchmarkConfig(
            vllm_url="http://localhost:8000",
            model="org/some-model",
            skip_tokenizer_load=True,
            ignore_replay_output_length=True,
        )
        with patch("agentsurge.runner._load_tokenizer", mock_load):
            runner = BenchmarkRunner(cfg)

        mock_load.assert_not_called()
        assert runner._tokenizer is None

    def test_runner_raises_when_skip_tokenizer_load_true_but_replay_sizing_required(self):
        """skip_tokenizer_load=True without ignore_replay_output_length must raise ValueError."""
        mock_load = MagicMock(side_effect=RuntimeError("should not be called"))
        cfg = BenchmarkConfig(
            vllm_url="http://localhost:8000",
            model="org/some-model",
            skip_tokenizer_load=True,
            ignore_replay_output_length=False,
        )
        with patch("agentsurge.runner._load_tokenizer", mock_load):
            with pytest.raises(ValueError, match=r"tokenizer") as exc_info:
                BenchmarkRunner(cfg)

        mock_load.assert_not_called()
        assert "ignore-replay-output-length" in str(exc_info.value)


class TestTokenizerCLIMapping:
    """Unit tests for _tokenizer_kwargs normalization logic."""

    def setup_method(self):
        from agentsurge.cli._helpers import _tokenizer_kwargs

        self._fn = _tokenizer_kwargs

    def test_none_input_maps_to_auto(self):
        result = self._fn(None)
        assert result == {"tokenizer_model": None, "skip_tokenizer_load": False}

    def test_auto_lowercase(self):
        result = self._fn("auto")
        assert result == {"tokenizer_model": None, "skip_tokenizer_load": False}

    def test_auto_uppercase(self):
        result = self._fn("AUTO")
        assert result == {"tokenizer_model": None, "skip_tokenizer_load": False}

    def test_none_string_lowercase(self):
        result = self._fn("none")
        assert result == {"tokenizer_model": None, "skip_tokenizer_load": True}

    def test_none_string_titlecase(self):
        result = self._fn("None")
        assert result == {"tokenizer_model": None, "skip_tokenizer_load": True}

    def test_cli_mapping_none_all_caps(self):
        result = self._fn("NONE")
        assert result == {"tokenizer_model": None, "skip_tokenizer_load": True}

    def test_cli_mapping_none_mixed_case(self):
        result = self._fn("nOnE")
        assert result == {"tokenizer_model": None, "skip_tokenizer_load": True}

    def test_hf_id_preserves_casing(self):
        result = self._fn("org/my-model")
        assert result == {"tokenizer_model": "org/my-model", "skip_tokenizer_load": False}

    def test_local_path_preserves_casing(self):
        result = self._fn("/local/path")
        assert result == {"tokenizer_model": "/local/path", "skip_tokenizer_load": False}
