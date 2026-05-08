#!/usr/bin/env python3
"""Custom sessions - build workloads with ReplaySession and validate them.

Shows how to construct :class:`ReplaySession` objects manually, serialize
them, validate the workload schema, and replay through the mock backend.

This example is part of the API contract and runs in CI.

Usage::

    python examples/02_custom_sessions.py
"""

from __future__ import annotations

import asyncio
import json

from agentsurge import (
    WORKLOAD_SCHEMA_VERSION,
    ReplaySession,
    Session,
    Turn,
    validate_workload,
)
from agentsurge.backends.mock import MockBackend, MockConfig
from agentsurge.exceptions import WorkloadValidationError


# ── Build sessions manually ─────────────────────────────────────
def make_session(session_id: str, n_turns: int = 3) -> ReplaySession:
    """Create a session with N turns of growing context."""
    turn_messages: list[list[dict]] = []
    history: list[dict] = []

    for i in range(n_turns):
        history = list(history)  # copy so each snapshot is independent
        history.append({"role": "user", "content": f"Turn {i}: explain quicksort"})
        history.append({"role": "assistant", "content": f"Response for turn {i}."})
        turn_messages.append(list(history))

    return ReplaySession(
        session_id=session_id,
        turn_messages=turn_messages,
        metadata={"source": "example"},
    )


sessions = [make_session(f"session-{i}", n_turns=4) for i in range(6)]

# ── Serialize / round-trip ───────────────────────────────────────
workload = {
    "version": WORKLOAD_SCHEMA_VERSION,
    "sessions": [s.to_dict() for s in sessions],
}

# Validate the workload schema
validate_workload(workload)
print("Workload validation passed ✓")

# Round-trip through JSON
payload = json.dumps(workload)
restored = json.loads(payload)
restored_sessions = [ReplaySession.from_dict(d) for d in restored["sessions"]]
assert len(restored_sessions) == 6
assert all(s.n_turns == 4 for s in restored_sessions)

# Invalid workload raises
try:
    validate_workload({"version": "99.0", "sessions": []})
    raise AssertionError("should have raised")
except WorkloadValidationError as exc:
    print(f"Invalid workload caught: {exc}")


# ── Replay through mock backend ─────────────────────────────────
async def replay() -> None:
    config = MockConfig(output_tokens=20, ttft_ms=2.0)
    async with MockBackend(config) as backend:
        for session in sessions:
            for turn_idx, messages in enumerate(session.turn_messages):
                result = await backend.send_turn(
                    session.session_id,
                    turn_idx,
                    messages,
                )
                assert result.ok, f"{session.session_id} turn {turn_idx} failed"

        total_turns = sum(s.n_turns for s in sessions)
        assert len(backend.call_log) == total_turns
        print(f"Replayed {len(sessions)} sessions, {total_turns} turns total")


asyncio.run(replay())


# ── Core trace types ─────────────────────────────────────────────
turn = Turn(role="user", content="Explain KV caching in LLM serving.")
session = Session(
    session_id="demo-001",
    turns=[
        turn,
        Turn(role="assistant", content="KV caching stores key-value pairs..."),
    ],
    metadata={"source": "example"},
)
assert session.n_turns == 2
print(f"Session {session.session_id}: {session.n_turns} turns")

print("\n✓ custom-sessions example passed")
