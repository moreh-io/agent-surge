"""Trimmed CLI tests: config loading, output path, loaders, cmd_generate, cmd_run, structure."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import tomllib
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PYPROJECT = os.path.join(_REPO_ROOT, "pyproject.toml")
_REQUIREMENTS = os.path.join(_REPO_ROOT, "requirements.txt")


def _make_args(**kwargs):
    defaults = {
        "config": None,
        "vllm_url": None,
        "model": None,
        "output_dir": "/tmp/test_agentsurge_output",
        "command": "generate",
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ===========================================================================
# _load_config
# ===========================================================================


class TestLoadConfig:
    def test_no_args_returns_default(self):
        from agentsurge.cli import _load_config

        cfg = _load_config()
        assert isinstance(cfg, dict)
        assert "vllm" in cfg and "workload" in cfg
        assert cfg["vllm"]["url"] == "http://localhost:8000"

    def test_custom_yaml_path(self):
        import yaml

        from agentsurge.cli import _load_config

        data = {
            "vllm": {"url": "http://custom:9000", "model": "m"},
            "workload": {"n_turns": 1},
            "runner": {"max_concurrency": 2},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            assert _load_config(path)["vllm"]["url"] == "http://custom:9000"
        finally:
            os.unlink(path)


# ===========================================================================
# _output_path
# ===========================================================================


def test_output_path():
    from agentsurge.cli import _output_path

    with tempfile.TemporaryDirectory() as tmpdir:
        result = _output_path(tmpdir, "my_prefix")
        assert "my_prefix" in result and result.endswith(".json")
        new_dir = os.path.join(tmpdir, "subdir")
        _output_path(new_dir, "prefix")
        assert os.path.isdir(new_dir)


def test_build_backend_preserves_vllm_runner_config():
    from agentsurge.backends.vllm import VllmBackend, VllmConfig
    from agentsurge.cli.run import _build_backend
    from agentsurge.types import BenchmarkConfig

    args = argparse.Namespace(backend="vllm")
    cfg = BenchmarkConfig(
        vllm_url="http://vllm:8000",
        model="m",
        max_tokens=123,
        temperature=0.2,
        api_type="responses",
        request_timeout=17,
        extra_body={"top_k": 20},
        stream_idle_timeout=3.5,
        enable_thinking=True,
    )

    backend = _build_backend(args, cfg)

    assert isinstance(backend, VllmBackend)
    assert isinstance(backend.config, VllmConfig)
    assert backend.config.api_type == "responses"
    assert backend.config.request_timeout == 17
    assert backend.config.extra_body == {"top_k": 20}
    assert backend.config.stream_idle_timeout == 3.5
    assert backend.config.chat_template_kwargs == {"enable_thinking": True}


def test_build_runner_config_tool_env_and_workspace_dir():
    from agentsurge.cli._helpers import _build_runner_config

    args = _make_args(
        workspace_dir="/tmp/workspaces",
        tool_env="inherit",
        tool_mode="real",
        use_model_reply_in_next_turn="auto",
    )
    cfg = {
        "vllm": {"url": "http://vllm:8000", "model": "m"},
        "runner": {},
    }

    result = _build_runner_config(args, cfg)

    assert result.workspace_dir == "/tmp/workspaces"
    assert result.sandbox_dir == "/tmp/workspaces"
    assert result.tool_env == "inherit"


def test_dataset_config_name_prefers_source_then_workload_file():
    from agentsurge.cli.run import _dataset_config_name

    assert _dataset_config_name(argparse.Namespace(source="openhands")) == "openhands"
    assert (
        _dataset_config_name(argparse.Namespace(source=None, workload_file="/tmp/workload.json"))
        == "workload"
    )
    assert _dataset_config_name(argparse.Namespace(source=None, workload_file=None)) == "synthetic"


def test_requirements_matches_core_pyproject_dependencies():
    with open(_PYPROJECT, "rb") as f:
        pyproject = tomllib.load(f)
    expected = set(pyproject["project"]["dependencies"])

    with open(_REQUIREMENTS) as f:
        requirements = {
            line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")
        }

    assert requirements == expected


def test_hf_extra_declares_pyarrow_for_parquet_export():
    with open(_PYPROJECT, "rb") as f:
        pyproject = tomllib.load(f)
    extras = pyproject["project"]["optional-dependencies"]

    assert any(dep.startswith("pyarrow") for dep in extras["hf"])
    assert any("hf" in dep for dep in extras["all"])


# ===========================================================================
# _get_trace_loader / _get_task_loader
# ===========================================================================


def test_trace_loader_unknown_raises():
    from agentsurge.cli import _get_trace_loader

    with pytest.raises(ValueError, match="Unknown trace source"):
        _get_trace_loader("unknown_source")


def test_trace_loader_openhands():
    from agentsurge.cli import _get_trace_loader

    loader = _get_trace_loader("openhands", local_path="/tmp/data.jsonl")
    assert loader.name == "openhands"
    assert "openhands" in loader.__class__.__name__.lower()


def test_task_loader_unknown_raises():
    from agentsurge.cli import _get_task_loader

    with pytest.raises(ValueError, match="Unknown task source"):
        _get_task_loader("unknown_source")


def test_task_loader_all_sources():
    from agentsurge.cli import _get_task_loader

    for source in ("swe-bench", "abc-bench", "swe-evo", "featurebench"):
        loader = _get_task_loader(source)
        assert loader is not None, f"source={source!r} returned None"
        assert hasattr(loader, "load"), f"source={source!r} loader missing load()"
        canonical = source.replace("-", "").replace("_", "").lower()
        cls_name = loader.__class__.__name__.lower()
        assert canonical in cls_name or source.split("-")[0] in cls_name, (
            f"source={source!r}: class name {cls_name!r} does not reflect source"
        )


# ===========================================================================
# cmd_generate
# ===========================================================================


class TestCmdGenerate:
    def _mock_session(self):
        s = MagicMock()
        s.session_id = "sess-0"
        s.n_turns = 3
        s.turn_messages = [[{"role": "user", "content": "hi"}]]
        s.metadata = {}
        return s

    def test_synthetic_mode_saves_json(self):
        from agentsurge.cli import cmd_generate

        mock_gen = MagicMock()
        mock_gen.generate.return_value = [self._mock_session()]
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_args(
                source=None,
                n_sessions=5,
                n_turns=3,
                synthetic_tokens_per_turn=100,
                output_dir=tmpdir,
            )
            with patch("agentsurge.generators.synthetic.SyntheticGenerator", return_value=mock_gen):
                cmd_generate(args)
            files = [f for f in os.listdir(tmpdir) if f.startswith("workload_")]
            assert len(files) == 1
            with open(os.path.join(tmpdir, files[0])) as f:
                data = json.load(f)
            assert "version" in data and "sessions" in data

    def test_unknown_source_raises(self):
        from agentsurge.cli import cmd_generate

        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_args(
                source="nonexistent",
                n_sessions=5,
                n_turns=None,
                synthetic_tokens_per_turn=None,
                flatten_tools=True,
                max_prompt_chars=16384,
                output_dir=tmpdir,
            )
            with pytest.raises(ValueError, match="Unknown source"):
                cmd_generate(args)


# ===========================================================================
# cmd_run
# ===========================================================================


class TestCmdRun:
    def test_run_parser_tool_mode_defaults_off(self):
        import sys

        import agentsurge.cli._app as app_mod

        captured_args = {}

        def fake_cmd_run(args):
            captured_args["args"] = args

        test_argv = ["agentsurge", "run"]
        with (
            patch.object(sys, "argv", test_argv),
            patch("agentsurge.cli._app.cmd_run", side_effect=fake_cmd_run),
        ):
            app_mod.main()

        assert captured_args["args"].tool_mode == "off"
        assert captured_args["args"].tool_env == "safe"

    def test_run_parser_tool_env_inherit_and_workspace_dir(self):
        import sys

        import agentsurge.cli._app as app_mod

        captured_args = {}

        def fake_cmd_run(args):
            captured_args["args"] = args

        test_argv = [
            "agentsurge",
            "run",
            "--tool-env",
            "inherit",
            "--tool-workspace-dir",
            "/tmp/workspaces",
        ]
        with (
            patch.object(sys, "argv", test_argv),
            patch("agentsurge.cli._app.cmd_run", side_effect=fake_cmd_run),
        ):
            app_mod.main()

        assert captured_args["args"].tool_env == "inherit"
        assert captured_args["args"].workspace_dir == "/tmp/workspaces"

    def test_nonexistent_workload_file_exits(self):
        from agentsurge.cli import cmd_run

        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_args(
                n_sessions=3,
                n_turns=None,
                synthetic_tokens_per_turn=None,
                concurrency=None,
                arrival=None,
                workload_file="/nonexistent/workload.json",
                output_dir=tmpdir,
            )
            with pytest.raises(SystemExit) as exc:
                cmd_run(args)
            assert exc.value.code == 1

    @pytest.mark.parametrize("source", ["featurebench", "swe-evo", "mixed"])
    def test_real_mode_unsupported_source_exits(self, source, capsys):
        from agentsurge.cli import cmd_run

        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_args(
                n_sessions=1,
                n_turns=None,
                synthetic_tokens_per_turn=None,
                concurrency=None,
                arrival=None,
                workload_file=None,
                trace_pool=None,
                tool_mode="real",
                source=source,
                output_dir=tmpdir,
            )
            with pytest.raises(SystemExit) as exc:
                cmd_run(args)
            assert exc.value.code == 1
            captured = capsys.readouterr()
            assert "workspace setup" in captured.err
            assert "sandbox setup" not in captured.err


# ===========================================================================
# CLI structure / entrypoints
# ===========================================================================


# ---------------------------------------------------------------------------
# --multi-turn-only filter
# ---------------------------------------------------------------------------


class TestMultiTurnOnlyFilter:
    def _make_session(self, session_id, pending):
        from agentsurge.types import ReplaySession

        return ReplaySession(
            session_id=session_id,
            turn_messages=[[{"role": "user", "content": "u1"}]],
            pending_user_messages=list(pending),
        )

    def test_filter_keeps_only_multi_turn_sessions(self):
        from agentsurge.cli._helpers import _filter_multi_turn_only

        sessions = [
            self._make_session("single", pending=[]),
            self._make_session("multi", pending=["u2"]),
            self._make_session("multi-2", pending=["u2", "u3"]),
        ]
        kept = _filter_multi_turn_only(sessions)
        assert [s.session_id for s in kept] == ["multi", "multi-2"]

    def test_filter_returns_empty_when_none_match(self):
        from agentsurge.cli._helpers import _filter_multi_turn_only

        sessions = [self._make_session("a", pending=[]), self._make_session("b", pending=[])]
        assert _filter_multi_turn_only(sessions) == []

    def test_generate_sessions_applies_filter_when_enabled(self):
        """--multi-turn-only filter flows through _generate_sessions."""
        from agentsurge.cli import _helpers

        fake_sessions = [
            self._make_session("single", pending=[]),
            self._make_session("multi", pending=["u2"]),
        ]

        # Patch the internal source-dispatch so _generate_sessions returns our fakes
        # regardless of the source. We only want to verify filter application.
        with (
            patch.object(
                _helpers,
                "_get_trace_loader",
                MagicMock(return_value=MagicMock(load=MagicMock(return_value=iter([])))),
            ),
            patch("agentsurge.cli._helpers.TraceReplayGenerator") as mock_gen_cls,
        ):
            mock_gen = mock_gen_cls.return_value
            mock_gen.from_trajectories.return_value = iter(fake_sessions)

            out = _helpers._generate_sessions(
                source="openhands",
                n_sessions=2,
                n_turns=10,
                synthetic_tokens_per_turn=1000,
                multi_turn_only=True,
            )

        assert [s.session_id for s in out] == ["multi"]

    @pytest.mark.e2e
    def test_cli_run_help_lists_multi_turn_only_flag(self):
        """agentsurge run --help mentions --multi-turn-only."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "agentsurge", "run", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # argparse --help exits 0
        assert result.returncode == 0, result.stderr
        assert "--multi-turn-only" in result.stdout

    @pytest.mark.e2e
    def test_cli_sweep_help_lists_multi_turn_only_flag(self):
        """agentsurge sweep --help also mentions --multi-turn-only."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "agentsurge", "sweep", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert "--multi-turn-only" in result.stdout


# ===========================================================================
# --lmcache tri-state semantics (H5)
# ===========================================================================


class TestLmcacheTriState:
    def _run_parser(self):
        from agentsurge.cli._app import _run_args

        return _run_args()

    def test_lmcache_absent_is_none(self):
        """When --lmcache is not passed, args.lmcache must be None (not False)."""
        p = self._run_parser()
        args, _ = p.parse_known_args([])
        assert args.lmcache is None, (
            f"expected None but got {args.lmcache!r}; "
            "auto-detect branch is dead when lmcache is False"
        )

    def test_lmcache_flag_is_true(self):
        """Passing --lmcache must produce True."""
        p = self._run_parser()
        args, _ = p.parse_known_args(["--lmcache"])
        assert args.lmcache is True

    def test_no_lmcache_flag_is_false(self):
        """Passing --no-lmcache must produce False."""
        p = self._run_parser()
        args, _ = p.parse_known_args(["--no-lmcache"])
        assert args.lmcache is False


# ===========================================================================
# M9 — --think-time parse-error handling (finding M9)
# ===========================================================================


class TestThinkTimeArgType:
    """--think-time must exit 2 with a formatted message for bad input."""

    def _build_parser(self):
        import agentsurge.cli._app as app_mod
        from agentsurge.cli._app import main as _main  # noqa: F401

        return app_mod._run_args()

    def test_single_value_exits_2(self, capsys):
        """--think-time 2.0 (missing second value) must exit 2, not IndexError."""
        p = self._build_parser()
        with pytest.raises(SystemExit) as exc:
            p.parse_args(["--think-time", "2.0"])
        assert exc.value.code == 2
        captured = capsys.readouterr()
        assert "think-time" in captured.err.lower() or "think_time" in captured.err.lower()

    def test_non_numeric_exits_2(self, capsys):
        """--think-time foo,bar (non-float) must exit 2, not ValueError."""
        p = self._build_parser()
        with pytest.raises(SystemExit) as exc:
            p.parse_args(["--think-time", "foo,bar"])
        assert exc.value.code == 2
        captured = capsys.readouterr()
        assert "think-time" in captured.err.lower() or "think_time" in captured.err.lower()

    def test_valid_value_parses_correctly(self):
        """--think-time 1.0,3.0 must parse to (1.0, 3.0)."""
        p = self._build_parser()
        args = p.parse_args(["--think-time", "1.0,3.0"])
        assert args.think_time == (1.0, 3.0)


# ===========================================================================
# M10 — --probe-levels parse-error handling (finding M10)
# ===========================================================================


class TestProbeLevelsArgType:
    """--probe-levels must exit 2 with a formatted message for bad input."""

    def _build_sweep_parser(self):
        # Build a minimal parser that mirrors how p_sweep registers --probe-levels.
        # We exercise the real _app.main parser via parse_known_args on the sweep subcommand.
        import argparse

        import agentsurge.cli._app as app_mod

        p = argparse.ArgumentParser()
        sub = p.add_subparsers(dest="command")
        run_parent = app_mod._run_args()
        common = argparse.ArgumentParser(add_help=False)
        common.add_argument("--config", default=None)
        common.add_argument("--vllm-url", default=None)
        common.add_argument("--model", default=None)
        common.add_argument("--output-dir", "-o", default="/tmp/out")
        common.add_argument("--api", default="chat")
        common.add_argument("--backend", default=None)
        common.add_argument("--no-metrics", action="store_true", default=False)
        common.add_argument("--seed", type=int, default=42)
        p_sweep = sub.add_parser("sweep", parents=[common, run_parent])
        # Copy just the --probe-levels arg from _app.main
        p_sweep.add_argument(
            "--probe-levels",
            default="50,100,200",
        )
        return p

    def test_non_integer_probe_level_exits_2(self, capsys):
        """--probe-levels 10,20,foo must exit 2, not ValueError."""

        # Use the real assembled parser via sys.argv patching
        import sys
        from unittest.mock import patch

        test_argv = ["agentsurge", "sweep", "--probe-levels", "10,20,foo"]
        with patch.object(sys, "argv", test_argv):
            import agentsurge.cli._app as app_mod

            # Re-build real parser by calling main() with bad args
            # We need to intercept parse_args, so instead build the full parser
            # the same way main() does but call parse_args on our argv.
            # Easiest: call app_mod.main() directly and let argparse sys.exit.
            with pytest.raises(SystemExit) as exc:
                app_mod.main()
        assert exc.value.code == 2
        captured = capsys.readouterr()
        assert "probe-levels" in captured.err.lower() or "probe_levels" in captured.err.lower()

    def test_valid_probe_levels_parses_correctly(self):
        """--probe-levels 10,20,30 must parse to [10, 20, 30]."""
        import sys
        from unittest.mock import patch

        test_argv = ["agentsurge", "sweep", "--probe-levels", "10,20,30"]
        with patch.object(sys, "argv", test_argv):
            import agentsurge.cli._app as app_mod

            # Intercept parse_args result before func() is called
            captured_args = {}

            def fake_func(args):
                captured_args["args"] = args

            with patch("agentsurge.cli._app.cmd_sweep", side_effect=fake_func):
                app_mod.main()

        assert captured_args["args"].probe_levels == [10, 20, 30]
