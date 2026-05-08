#!/usr/bin/env python3
"""Callbacks - monitor progress with on_turn and on_session hooks.

Demonstrates the callback protocol using the mock backend.  Both sync
and async callables are accepted by the callback system.

This example is part of the API contract and runs in CI.

Usage::

    python examples/03_callbacks.py
"""

from __future__ import annotations

import asyncio

from agentsurge.backends.mock import MockBackend, MockConfig
from agentsurge.callbacks import invoke_on_session, invoke_on_turn
from agentsurge.types import ReplaySession, SessionResult, TurnResult

# ── Collect events via callbacks ─────────────────────────────────
turn_events: list[TurnResult] = []
session_events: list[SessionResult] = []


def on_turn(tr: TurnResult) -> None:
    """Sync callback - called after each turn completes."""
    turn_events.append(tr)


def on_session(sr: SessionResult) -> None:
    """Sync callback - called after each session completes."""
    session_events.append(sr)
    print(f"  session {sr.session_id}: {'OK' if sr.ok else 'FAIL'} ({len(sr.turns)} turns)")


# ── Build sessions and replay with callbacks ─────────────────────
def make_sessions(n: int, n_turns: int) -> list[ReplaySession]:
    sessions = []
    for i in range(n):
        turn_messages = []
        history: list[dict] = []
        for t in range(n_turns):
            history = list(history)
            history.append({"role": "user", "content": f"Turn {t}"})
            turn_messages.append(list(history))
        sessions.append(
            ReplaySession(
                session_id=f"cb-session-{i}",
                turn_messages=turn_messages,
            )
        )
    return sessions


async def main() -> None:
    sessions = make_sessions(n=4, n_turns=3)
    config = MockConfig(output_tokens=15, ttft_ms=3.0)

    async with MockBackend(config) as backend:
        for session in sessions:
            session_turns: list[TurnResult] = []

            for turn_idx, messages in enumerate(session.turn_messages):
                tr = await backend.send_turn(
                    session.session_id,
                    turn_idx,
                    messages,
                )
                session_turns.append(tr)

                # Fire the on_turn callback (just like the real runner does)
                await invoke_on_turn(on_turn, tr)

            # Build a SessionResult and fire the on_session callback
            sr = SessionResult(
                session_id=session.session_id,
                turns=session_turns,
                total_ms=sum(t.total_ms for t in session_turns),
                llm_ms=sum(t.total_ms for t in session_turns),
            )
            await invoke_on_session(on_session, sr)

    # ── Verify callbacks fired ───────────────────────────────────
    # 4 sessions × 3 turns = 12 turn events
    assert len(turn_events) == 12, f"expected 12 turn events, got {len(turn_events)}"
    assert len(session_events) == 4, f"expected 4 session events, got {len(session_events)}"

    # Each turn event has timing info
    for tr in turn_events:
        assert tr.ok
        assert tr.ttft_ms >= 0

    print(f"\nReceived {len(turn_events)} turn events, {len(session_events)} session events")
    print("\n✓ callbacks example passed")


if __name__ == "__main__":
    asyncio.run(main())
