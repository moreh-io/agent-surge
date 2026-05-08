"""Tests for agentsurge.callbacks module."""

from __future__ import annotations

import pytest

from agentsurge.callbacks import invoke_on_session, invoke_on_turn
from agentsurge.types import SessionResult, TurnResult


@pytest.fixture
def turn_result():
    return TurnResult(session_id="s1", turn_index=0, completed=True)


@pytest.fixture
def session_result():
    return SessionResult(session_id="s1")


# ---------------------------------------------------------------------------
# invoke_on_turn
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invoke_on_turn_none(turn_result):
    await invoke_on_turn(None, turn_result)


@pytest.mark.asyncio
async def test_invoke_on_turn_sync(turn_result):
    seen = []
    await invoke_on_turn(lambda tr: seen.append(tr), turn_result)
    assert seen == [turn_result]


@pytest.mark.asyncio
async def test_invoke_on_turn_exception_swallowed(turn_result):
    called = []

    def bad_cb(tr):
        called.append(True)
        raise RuntimeError("boom")

    await invoke_on_turn(bad_cb, turn_result)
    assert called


# ---------------------------------------------------------------------------
# invoke_on_session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invoke_on_session_none(session_result):
    await invoke_on_session(None, session_result)


@pytest.mark.asyncio
async def test_invoke_on_session_sync(session_result):
    seen = []
    await invoke_on_session(lambda sr: seen.append(sr), session_result)
    assert seen == [session_result]


@pytest.mark.asyncio
async def test_invoke_on_session_exception_swallowed(session_result):
    called = []

    def bad_cb(sr):
        called.append(True)
        raise ValueError("oops")

    await invoke_on_session(bad_cb, session_result)
    assert called
