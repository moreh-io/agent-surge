# SPDX-License-Identifier: MIT
"""Callback protocols for structured progress reporting during benchmark runs.

Users can pass ``on_turn`` and ``on_session`` callables to :func:`agentsurge.run`
or :func:`agentsurge.arun` to receive structured updates as the benchmark
progresses.  Both synchronous and asynchronous callables are accepted.

Example
-------
>>> from agentsurge import run, TurnResult, SessionResult
>>> turns_seen: list[TurnResult] = []
>>> result = run(
...     preset="mock",
...     on_turn=lambda tr: turns_seen.append(tr),
...     on_session=lambda sr: print(f"Session {sr.session_id} done"),
... )
"""

import inspect
import logging
from collections.abc import Callable
from typing import Any

from agentsurge.types import SessionResult, TurnResult

__all__ = [
    "OnTurnCallback",
    "OnSessionCallback",
    "invoke_on_turn",
    "invoke_on_session",
]

_log = logging.getLogger(__name__)

OnTurnCallback = Callable[[TurnResult], Any]
OnSessionCallback = Callable[[SessionResult], Any]


async def invoke_on_turn(
    callback: OnTurnCallback | None,
    turn_result: TurnResult,
) -> None:
    """Fire the *on_turn* callback, handling sync and async callables.

    Exceptions raised by the callback are logged but **never** propagated -
    a misbehaving callback must not abort the benchmark.
    """
    if callback is None:
        return
    try:
        rv = callback(turn_result)
        if inspect.isawaitable(rv):
            await rv
    except Exception:
        _log.warning(
            "on_turn callback raised for session=%s turn=%d",
            turn_result.session_id,
            turn_result.turn_index,
            exc_info=True,
        )


async def invoke_on_session(
    callback: OnSessionCallback | None,
    session_result: SessionResult,
) -> None:
    """Fire the *on_session* callback, handling sync and async callables.

    Exceptions raised by the callback are logged but **never** propagated -
    a misbehaving callback must not abort the benchmark.
    """
    if callback is None:
        return
    try:
        rv = callback(session_result)
        if inspect.isawaitable(rv):
            await rv
    except Exception:
        _log.warning(
            "on_session callback raised for session=%s",
            session_result.session_id,
            exc_info=True,
        )
