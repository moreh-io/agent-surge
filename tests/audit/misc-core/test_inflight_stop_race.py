# SPDX-License-Identifier: MIT
"""[audit:misc-core#M4] InflightDumpWriter on_turn racing with stop().

Without a lock, ``on_turn`` can read ``_stopped == False``, yield to
the writer thread, ``stop()`` runs, the writer exits, then ``on_turn``
calls ``put_nowait`` against an abandoned queue. The drain-after-
sentinel path doesn't cover that window because the late item was
queued AFTER the sentinel was processed. Those turns silently disappear.

The fix: hold a lock around the read+put in ``on_turn`` and around the
``_stopped`` flip + sentinel push in ``stop()``, so that any ``on_turn``
that observes ``_stopped == False`` is guaranteed to land its
``put_nowait`` BEFORE ``stop()`` flips the flag.
"""

from __future__ import annotations

import threading
from pathlib import Path

from agentsurge._inflight_dump import InflightDumpWriter
from agentsurge.types import TurnResult


def _mk_turn(idx: int) -> TurnResult:
    return TurnResult(
        session_id=f"s{idx}",
        turn_index=idx,
        completed=True,
        ttft_ms=1.0,
        total_ms=2.0,
        output_tokens=1,
        input_messages=1,
        input_tokens=1,
    )


def test_writer_uses_lock_to_serialize_on_turn_and_stop(tmp_path: Path) -> None:
    """Assert the writer exposes a lock guarding the start/stop transition.

    The audit-recommended fix is a ``threading.Lock`` (or RLock) that
    ``on_turn`` and ``stop`` both acquire so the ``_stopped`` read+put
    pair and the flip+sentinel pair are mutually exclusive. We assert
    the existence of such a lock — without it, the TOCTOU window cannot
    be closed.
    """
    writer = InflightDumpWriter(tmp_path / "turns.jsonl", flush_every=1)
    lock = getattr(writer, "_lock", None)
    assert lock is not None and hasattr(lock, "acquire") and hasattr(lock, "release"), (
        "InflightDumpWriter must expose a _lock attribute to serialize "
        "on_turn/stop transitions; otherwise the on_turn-vs-stop TOCTOU "
        "race strands turns (audit M4)."
    )


def test_late_on_turn_after_stop_does_not_strand_in_queue(tmp_path: Path) -> None:
    """Functional check: stop() must leave the queue empty (no stranded items).

    A ``stop()`` followed by an ``on_turn`` from a thread that already
    crossed the guard must not leave items in the queue. With the lock,
    the second on_turn observes ``_stopped=True`` and returns; without
    it, a put can land after stop and never be drained.
    """
    path = tmp_path / "turns.jsonl"
    writer = InflightDumpWriter(path, flush_every=1)
    writer.start()
    writer.on_turn(_mk_turn(0))

    barrier = threading.Event()

    def late_caller() -> None:
        # Wait until stop() has returned, then call on_turn. With the
        # fix, on_turn re-checks _stopped under the lock and bails.
        barrier.wait(timeout=5.0)
        writer.on_turn(_mk_turn(99))

    t = threading.Thread(target=late_caller)
    t.start()

    writer.stop()
    barrier.set()
    t.join(timeout=5.0)
    assert not t.is_alive()

    assert writer._queue.empty(), (
        "after stop() returned, the queue still contains items the "
        "writer thread cannot drain — late on_turn stranded a turn "
        f"(qsize={writer._queue.qsize()})"
    )
