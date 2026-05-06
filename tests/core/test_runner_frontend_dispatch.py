"""Frontend dispatch wiring tests for BenchmarkRunner.

Task B: verify _dispatch_sessions branches between _run_session (direct)
and _run_session_frontend based on BenchmarkConfig.frontend_name.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agentsurge import BenchmarkConfig, ReplaySession
from agentsurge.frontends.codex import CodexProvider
from agentsurge.frontends.runner import FrontendSessionRenderer
from agentsurge.runner import BenchmarkRunner
from agentsurge.types import SessionResult
from agentsurge.types.results import FrontendRuntimeSettings
from tests.core.test_frontend_provider_wiring import _make_fake_subprocess

# All declared frontend names are wired in _resolve_provider as of Task N.2.
# Tests below that need a non-direct frontend pick "codex" purely so
# frontend_name != "direct" triggers the frontend-dispatch branch; the
# _run_session_frontend method itself is mocked, so the wiring isn't exercised.
_UNWIRED = FrontendRuntimeSettings(name="codex")


def _make_session(sid: str = "s1") -> ReplaySession:
    return ReplaySession(
        session_id=sid,
        turn_messages=[[{"role": "user", "content": "hi"}]],
    )


def test_default_frontend_name_is_direct():
    cfg = BenchmarkConfig(vllm_url="http://x", model="t")
    assert cfg.frontend_name == "direct"


def test_frontend_name_reads_frontend_attr():
    cfg = BenchmarkConfig(vllm_url="http://x", model="t")
    cfg.frontend = _UNWIRED
    assert cfg.frontend_name == "codex"


def test_dispatch_direct_invokes_run_session_only():
    """Direct mode (default) must route through _run_session, never the frontend path."""
    cfg = BenchmarkConfig(vllm_url="http://x", model="t", no_metrics=True)
    runner = BenchmarkRunner(cfg)

    direct_calls: list[str] = []
    frontend_calls: list[str] = []

    async def fake_direct(http, semaphore, session, delay, session_index=0):
        direct_calls.append(session.session_id)
        return SessionResult(session_id=session.session_id, completed=True)

    async def fake_frontend(http, semaphore, session, delay, session_index=0):
        frontend_calls.append(session.session_id)
        return SessionResult(session_id=session.session_id, completed=True)

    runner._run_session = fake_direct  # type: ignore[method-assign]
    runner._run_session_frontend = fake_frontend  # type: ignore[method-assign]

    sessions = [_make_session("s1"), _make_session("s2")]
    results = asyncio.run(
        runner._dispatch_sessions(MagicMock(), asyncio.Semaphore(10), sessions, [0.0, 0.0])
    )
    assert sorted(direct_calls) == ["s1", "s2"]
    assert frontend_calls == []
    assert len(results) == 2


def test_dispatch_tool_mode_real_with_direct_frontend_uses_tool_session():
    """tool_mode='real' + frontend='direct' must route through _run_tool_session,
    not the frontend path. Frontend dispatch must not override tool-mode dispatch
    when the user explicitly stays on the direct frontend.
    """
    cfg = BenchmarkConfig(vllm_url="http://x", model="t", no_metrics=True, tool_mode="real")
    runner = BenchmarkRunner(cfg)

    direct_calls: list[str] = []
    tool_calls: list[str] = []
    frontend_calls: list[str] = []

    async def fake_direct(http, semaphore, session, delay, session_index=0):
        direct_calls.append(session.session_id)
        return SessionResult(session_id=session.session_id, completed=True)

    async def fake_tool(http, semaphore, session, delay, session_index=0):
        tool_calls.append(session.session_id)
        return SessionResult(session_id=session.session_id, completed=True)

    async def fake_frontend(http, semaphore, session, delay, session_index=0):
        frontend_calls.append(session.session_id)
        return SessionResult(session_id=session.session_id, completed=True)

    runner._run_session = fake_direct  # type: ignore[method-assign]
    runner._run_tool_session = fake_tool  # type: ignore[method-assign]
    runner._run_session_frontend = fake_frontend  # type: ignore[method-assign]

    sessions = [_make_session("s1")]
    asyncio.run(runner._dispatch_sessions(MagicMock(), asyncio.Semaphore(10), sessions, [0.0]))
    assert tool_calls == ["s1"]
    assert direct_calls == []
    assert frontend_calls == []


def test_dispatch_frontend_invokes_frontend_path():
    """When frontend_name != 'direct', dispatch must route through _run_session_frontend."""
    cfg = BenchmarkConfig(vllm_url="http://x", model="t", no_metrics=True)
    cfg.frontend = _UNWIRED
    runner = BenchmarkRunner(cfg)

    direct_calls: list[str] = []
    frontend_calls: list[str] = []

    async def fake_direct(http, semaphore, session, delay, session_index=0):
        direct_calls.append(session.session_id)
        return SessionResult(session_id=session.session_id, completed=True)

    async def fake_frontend(http, semaphore, session, delay, session_index=0):
        frontend_calls.append(session.session_id)
        return SessionResult(session_id=session.session_id, completed=True)

    runner._run_session = fake_direct  # type: ignore[method-assign]
    runner._run_session_frontend = fake_frontend  # type: ignore[method-assign]

    sessions = [_make_session("s1")]
    results = asyncio.run(
        runner._dispatch_sessions(MagicMock(), asyncio.Semaphore(10), sessions, [0.0])
    )
    assert direct_calls == []
    assert frontend_calls == ["s1"]
    assert len(results) == 1


def test_unwired_frontend_records_failure_in_session_result(monkeypatch):
    """A NotImplementedError raised by _run_session_frontend must surface as
    a failed SessionResult with metadata['failed']=True (existing capture pattern).

    All declared frontend names are now wired, so we force the failure by
    monkeypatching _resolve_provider to raise — the dispatch-level failure
    capture itself is what this test pins down.
    """
    from agentsurge.frontends.runner import FrontendSessionRenderer

    def _boom(self, name):
        raise NotImplementedError(f"frontend {name!r} not yet wired")

    monkeypatch.setattr(FrontendSessionRenderer, "_resolve_provider", _boom)

    cfg = BenchmarkConfig(vllm_url="http://x", model="t", no_metrics=True)
    cfg.frontend = _UNWIRED
    runner = BenchmarkRunner(cfg)

    session = _make_session("s_fail")
    results = asyncio.run(
        runner._dispatch_sessions(MagicMock(), asyncio.Semaphore(10), [session], [0.0])
    )
    assert len(results) == 1
    failed = results[0]
    assert failed.session_id == "s_fail"
    assert failed.metadata.get("failed") is True
    err = failed.metadata.get("error", "")
    assert "NotImplementedError" in err
    assert "codex" in err


# ---------- Gap-3: extra_env merge order ----------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CLAUDE_FIXTURE = _REPO_ROOT / "tests/core/fixtures/claude/real_simple_session_v2.1.122.jsonl"


def _make_fake_subprocess_from_fixture(fixture_path: Path):
    """Fake asyncio.create_subprocess_exec that streams a fixture file."""
    from unittest.mock import AsyncMock

    raw = fixture_path.read_bytes().splitlines(keepends=True)
    lines = [ln for ln in raw if ln.strip()]

    async def _factory(*_args, **_kwargs):
        proc = MagicMock()
        proc.pid = 9999
        proc.returncode = None
        stdout_iter = iter(lines)

        async def _readline_stdout():
            try:
                line = next(stdout_iter)
            except StopIteration:
                return b""
            await asyncio.sleep(0.001)
            return line

        proc.stdout = MagicMock()
        proc.stdout.readline = AsyncMock(side_effect=_readline_stdout)
        proc.stderr = MagicMock()
        proc.stderr.readline = AsyncMock(return_value=b"")

        async def _wait():
            proc.returncode = 0
            return 0

        proc.wait = AsyncMock(side_effect=_wait)
        return proc

    return _factory


@pytest.mark.asyncio
async def test_renderer_extra_env_overrides_provider_env(tmp_path: Path, monkeypatch):
    """User-supplied --frontend-extra-env values must override provider-
    contributed env vars. Pinning this contract because the comment in
    claude.py:119-120 relies on it ("User-set ANTHROPIC_AUTH_TOKEN ...
    still wins because extra_env merges last in the renderer").

    The merge in runner.py:128-132 is:
        subprocess_env = {**os.environ, **provider_env, **dict(fconfig.extra_env or {})}
    so extra_env must beat provider_env for overlapping keys."""
    captured_envs: list[dict[str, str]] = []
    fake_factory = _make_fake_subprocess_from_fixture(_CLAUDE_FIXTURE)

    async def spy_factory(*args, **kwargs):
        proc = await fake_factory(*args, **kwargs)
        captured_envs.append(dict(kwargs.get("env") or {}))
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spy_factory)

    cfg = BenchmarkConfig(
        vllm_url="http://127.0.0.1:18000",
        model="t",
        no_metrics=True,
        frontend=FrontendRuntimeSettings(
            name="claude",
            session_timeout_s=10.0,
            keep_artifacts="always",
            server_url="http://127.0.0.1:18000",
            extra_env={"ANTHROPIC_BASE_URL": "http://OVERRIDE-WINS"},
        ),
    )
    renderer = FrontendSessionRenderer(cfg, tmp_path)
    await renderer.run(_make_session("s_extra_env"))

    assert captured_envs, "subprocess was never spawned"
    env = captured_envs[-1]
    assert env.get("ANTHROPIC_BASE_URL") == "http://OVERRIDE-WINS", (
        f"extra_env must override provider-contributed ANTHROPIC_BASE_URL; "
        f"got {env.get('ANTHROPIC_BASE_URL')!r}"
    )


_CODEX_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "codex" / "real_simple_session_v0.125.0.jsonl"
)


@pytest.mark.asyncio
async def test_renderer_spawns_request_shim_for_codex(tmp_path: Path, monkeypatch):
    """Codex provider declares requires_request_rewrite=True. The renderer
    must wrap the CLI in RequestShim and pass the shim URL into the
    provider's build_command/build_env so requests funnel through the
    rewrite logic."""
    captured_server_urls: list[str | None] = []

    real_build_command = CodexProvider.build_command

    def spy_build_command(self, artifacts, config):
        captured_server_urls.append(config.server_url)
        return real_build_command(self, artifacts, config)

    monkeypatch.setattr(CodexProvider, "build_command", spy_build_command)

    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        _make_fake_subprocess(_CODEX_FIXTURE),
    )

    cfg = BenchmarkConfig(
        vllm_url="http://x",
        model="t",
        no_metrics=True,
        frontend=FrontendRuntimeSettings(
            name="codex",
            session_timeout_s=10.0,
            keep_artifacts="always",
            server_url="http://localhost:18002/v1",
        ),
    )
    renderer = FrontendSessionRenderer(cfg, tmp_path)
    await renderer.run(_make_session("s_codex_shim"))

    assert captured_server_urls, "build_command never called"
    seen = captured_server_urls[-1]
    assert seen is not None and seen.startswith("http://127.0.0.1:"), (
        f"codex must see the shim URL, got {seen!r}"
    )
    assert seen != "http://localhost:18002/v1", "codex must not see the raw upstream URL"
