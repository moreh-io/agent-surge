"""Frontend dispatch wiring tests for BenchmarkRunner.

Task B: verify _dispatch_sessions branches between _run_session (direct)
and _run_session_frontend based on BenchmarkConfig.frontend_name.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from agentsurge import BenchmarkConfig, ReplaySession
from agentsurge.runner import BenchmarkRunner
from agentsurge.types import SessionResult
from agentsurge.types.results import FrontendRuntimeSettings

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
