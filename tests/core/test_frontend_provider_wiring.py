"""Hybrid wiring tests for codex/claude/opencode frontend providers.

Layer 1 (default CI): monkeypatch asyncio.create_subprocess_exec so the
renderer drives each Provider/Parser pair against a captured real fixture
streamed line-by-line. Verifies _resolve_provider wiring + end-to-end
metric population without invoking any real binary.

Layer 2 (opt-in): @pytest.mark.real_cli tests gated on
AGENTSURGE_E2E_FRONTENDS=1 actually spawn the real codex/claude/opencode
CLI against a tiny prompt. Skipped by default to keep CI hermetic.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentsurge.frontends.runner import FrontendSessionRenderer
from agentsurge.types import (
    BenchmarkConfig,
    FrontendRuntimeSettings,
    ReplaySession,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES = {
    "codex": REPO_ROOT / "tests/core/fixtures/codex/real_simple_session_v0.125.0.jsonl",
    "claude": REPO_ROOT / "tests/core/fixtures/claude/real_simple_session_v2.1.122.jsonl",
    "opencode": REPO_ROOT / "tests/core/fixtures/opencode/real_simple_session_v1.14.29.jsonl",
}


def _make_session(sid: str = "s_wire") -> ReplaySession:
    return ReplaySession(
        session_id=sid,
        turn_messages=[[{"role": "user", "content": "hi"}]],
        metadata={},
    )


def _make_fake_subprocess(fixture_path: Path):
    """Return an async factory that fakes asyncio.create_subprocess_exec.

    The fake process's stdout streams the captured fixture one line per
    readline() call (plus a tiny await between lines so timing metrics
    populate); stderr returns EOF immediately; wait() returns 0.
    """
    raw = fixture_path.read_bytes().splitlines(keepends=True)
    # Strip empty trailing lines so EOF arrives cleanly.
    lines = [ln for ln in raw if ln.strip()]

    async def _factory(*_args, **_kwargs):
        proc = MagicMock()
        proc.pid = 4242
        proc.returncode = None

        stdout_iter = iter(lines)

        async def _readline_stdout():
            try:
                line = next(stdout_iter)
            except StopIteration:
                return b""
            await asyncio.sleep(0.001)
            return line

        async def _readline_stderr():
            return b""

        proc.stdout = MagicMock()
        proc.stdout.readline = AsyncMock(side_effect=_readline_stdout)
        proc.stderr = MagicMock()
        proc.stderr.readline = AsyncMock(side_effect=_readline_stderr)

        async def _wait():
            proc.returncode = 0
            return 0

        proc.wait = AsyncMock(side_effect=_wait)
        return proc

    return _factory


def _build_config(name: str) -> BenchmarkConfig:
    return BenchmarkConfig(
        vllm_url="http://x",
        model="t",
        no_metrics=True,
        frontend=FrontendRuntimeSettings(
            name=name,
            session_timeout_s=10.0,
            keep_artifacts="always",
        ),
    )


@pytest.mark.asyncio
async def test_codex_provider_wired(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        _make_fake_subprocess(FIXTURES["codex"]),
    )
    cfg = _build_config("codex")
    renderer = FrontendSessionRenderer(cfg, tmp_path)
    result = await renderer.run(_make_session("s_codex"))

    fm = result.frontend_metrics
    assert fm is not None
    assert fm.provider == "codex"
    assert fm.process_exit_code == 0
    assert fm.event_count > 0
    assert fm.failure_category is None
    assert fm.provider_usage is not None
    assert fm.provider_usage.get("input_tokens") == 12804


@pytest.mark.asyncio
async def test_claude_provider_wired(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        _make_fake_subprocess(FIXTURES["claude"]),
    )
    cfg = _build_config("claude")
    renderer = FrontendSessionRenderer(cfg, tmp_path)
    result = await renderer.run(_make_session("s_claude"))

    fm = result.frontend_metrics
    assert fm is not None
    assert fm.provider == "claude"
    assert fm.process_exit_code == 0
    assert fm.event_count > 0
    assert fm.failure_category is None
    # Claude's result-event path normalizes usage differently from raw
    # API counters; just assert it populated.
    assert fm.provider_usage is not None


@pytest.mark.asyncio
async def test_opencode_provider_wired(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        _make_fake_subprocess(FIXTURES["opencode"]),
    )
    cfg = _build_config("opencode")
    renderer = FrontendSessionRenderer(cfg, tmp_path)
    result = await renderer.run(_make_session("s_opencode"))

    fm = result.frontend_metrics
    assert fm is not None
    assert fm.provider == "opencode"
    assert fm.process_exit_code == 0
    assert fm.event_count > 0
    assert fm.failure_category is None
    assert fm.provider_usage is not None
    assert fm.provider_usage.get("input_tokens") == 78


# ---------------------------------------------------------------------------
# Layer 2 — opt-in real-binary smoke tests. Skipped unless
# AGENTSURGE_E2E_FRONTENDS=1 is set. Run with: pytest -m real_cli
# ---------------------------------------------------------------------------

_REAL_CLI_SKIP = pytest.mark.skipif(
    os.environ.get("AGENTSURGE_E2E_FRONTENDS") != "1",
    reason="Real CLI tests skipped; set AGENTSURGE_E2E_FRONTENDS=1 to enable.",
)

_REAL_PROMPT = "Print exactly one short sentence. Do not call any tools."


def _real_session(sid: str) -> ReplaySession:
    return ReplaySession(
        session_id=sid,
        turn_messages=[[{"role": "user", "content": _REAL_PROMPT}]],
        metadata={},
    )


def _real_config(name: str, workspace: Path) -> BenchmarkConfig:
    return BenchmarkConfig(
        vllm_url="http://x",
        model="t",
        no_metrics=True,
        frontend=FrontendRuntimeSettings(
            name=name,
            session_timeout_s=60.0,
            keep_artifacts="always",
            workspace_dir=str(workspace),
        ),
    )


@pytest.mark.real_cli
@_REAL_CLI_SKIP
@pytest.mark.asyncio
async def test_codex_real_binary_smoke(tmp_path: Path):
    cfg = _real_config("codex", tmp_path)
    renderer = FrontendSessionRenderer(cfg, tmp_path)
    result = await renderer.run(_real_session("s_codex_real"))
    fm = result.frontend_metrics
    assert fm is not None
    assert fm.process_exit_code == 0
    assert fm.event_count > 0


@pytest.mark.real_cli
@_REAL_CLI_SKIP
@pytest.mark.asyncio
async def test_claude_real_binary_smoke(tmp_path: Path):
    cfg = _real_config("claude", tmp_path)
    renderer = FrontendSessionRenderer(cfg, tmp_path)
    result = await renderer.run(_real_session("s_claude_real"))
    fm = result.frontend_metrics
    assert fm is not None
    assert fm.process_exit_code == 0
    assert fm.event_count > 0


@pytest.mark.real_cli
@_REAL_CLI_SKIP
@pytest.mark.asyncio
async def test_opencode_real_binary_smoke(tmp_path: Path):
    cfg = _real_config("opencode", tmp_path)
    renderer = FrontendSessionRenderer(cfg, tmp_path)
    result = await renderer.run(_real_session("s_opencode_real"))
    fm = result.frontend_metrics
    assert fm is not None
    # opencode signals failure via is_error semantics in events; the
    # process itself should still exit 0 on a successful smoke run.
    assert fm.process_exit_code == 0
    assert fm.event_count > 0
