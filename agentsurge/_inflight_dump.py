# SPDX-License-Identifier: MIT
"""Inflight per-turn JSONL dump writer.

Background thread + ``queue.Queue`` so the hot benchmark loop does a cheap
non-blocking ``put_nowait`` and never stalls on disk I/O. Buffered writes
with periodic flush (default every 10 turns) keep fsync latency off the
NFS path. The final consolidated JSON dump is unaffected.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import threading
from pathlib import Path
from typing import Any

from agentsurge.types import TurnResult

__all__ = ["InflightDumpWriter"]

_log = logging.getLogger(__name__)
_SENTINEL: Any = object()


class InflightDumpWriter:
    """Append one JSONL line per completed turn, out-of-band from the hot loop.

    Single shared file, guarded by the internal writer thread (no asyncio
    lock in the hot path — ``queue.Queue`` is already thread-safe). Flush
    to the OS every ``flush_every`` lines and once more on ``stop()``; no
    per-write ``fsync``.
    """

    def __init__(self, path: str | Path, flush_every: int = 10) -> None:
        self._path = Path(path)
        self._flush_every = max(1, int(flush_every))
        self._queue: queue.Queue[Any] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._started = False
        self._stopped = False
        # Serialise on_turn's read+put with stop()'s flip+sentinel push.
        # Without this, on_turn can read _stopped=False, get context-
        # switched, stop() flips and joins the writer, and on_turn then
        # puts onto a queue whose consumer has exited — turn stranded.
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._started:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="inflight-dump-writer", daemon=True)
        self._started = True
        self._thread.start()

    def on_turn(self, turn: TurnResult) -> None:
        # Hold the lock across the _stopped read AND the put_nowait so
        # stop() cannot interleave between them. The encode is cheap;
        # holding the lock for it is fine for the audit M4 guarantee.
        with self._lock:
            if not self._started or self._stopped:
                return
            self._queue.put_nowait(_encode(turn))

    def stop(self) -> None:
        if not self._started:
            return  # no-op; unstarted writer stays in initial state
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            self._queue.put(_SENTINEL)
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _run(self) -> None:
        pending = 0
        try:
            with self._path.open("a", encoding="utf-8") as f:
                while True:
                    item = self._queue.get()
                    if item is _SENTINEL:
                        # Drain any items that raced past the on_turn `_stopped`
                        # check. Without a lock we can't fully eliminate the
                        # window, but draining here shrinks it to the span
                        # between this loop and thread exit.
                        while True:
                            try:
                                late = self._queue.get_nowait()
                            except queue.Empty:
                                break
                            if late is _SENTINEL:
                                continue
                            f.write(late)
                            f.write("\n")
                        f.flush()
                        # On NFS or async-mount disks, flush() only pushes
                        # Python buffers to the kernel page cache; data is
                        # still lost on host crash. fsync() durably commits
                        # so the inflight dump survives the failures it was
                        # built to recover from.
                        with contextlib.suppress(OSError):
                            os.fsync(f.fileno())
                        return
                    f.write(item)
                    f.write("\n")
                    pending += 1
                    if pending >= self._flush_every:
                        f.flush()
                        pending = 0
        except Exception:
            _log.exception("inflight dump writer thread crashed")


def _encode(turn: TurnResult) -> str:
    return json.dumps(
        {
            "session": turn.session_id,
            "turn": turn.turn_index,
            "tokens": turn.output_tokens,
            "ttft_ms": turn.ttft_ms,
            "response_text": turn.response_text,
            "finish_reason": turn.finish_reason,
        },
        ensure_ascii=False,
    )
