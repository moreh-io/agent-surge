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

# "codex" is a legal value for FrontendRuntimeSettings.name but is not yet wired
# in _resolve_provider, so it falls through to NotImplementedError — exactly
# what these dispatch tests need to exercise the unwired-frontend code path.
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


def test_unwired_frontend_records_failure_in_session_result():
    """A NotImplementedError raised by _run_session_frontend must surface as
    a failed SessionResult with metadata['failed']=True (existing capture pattern)."""
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
