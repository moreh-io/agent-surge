# SPDX-License-Identifier: MIT
"""[audit:misc-core#M3] InflightDumpWriter must fsync on stop().

The whole point of the inflight dump is to recover partial results when
the consolidated dump is unavailable. ``flush()`` only pushes Python
buffers to the kernel; on NFS or async-mount disks the data still sits
in the page cache and is lost on host crash. ``stop()`` must call
``os.fsync(fileno())`` on the writer file before returning.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from agentsurge._inflight_dump import InflightDumpWriter
from agentsurge.types import TurnResult


def test_stop_calls_fsync_on_writer_file(tmp_path: Path) -> None:
    path = tmp_path / "turns.jsonl"
    writer = InflightDumpWriter(path, flush_every=1)

    fsync_calls: list[int] = []
    real_fsync = os.fsync

    def spy(fd: int) -> None:
        fsync_calls.append(fd)
        real_fsync(fd)

    with patch("os.fsync", spy):
        writer.start()
        writer.on_turn(
            TurnResult(
                session_id="s",
                turn_index=0,
                completed=True,
                ttft_ms=1.0,
                total_ms=2.0,
                output_tokens=1,
                input_messages=1,
                input_tokens=1,
            )
        )
        writer.stop()

    assert fsync_calls, "stop() never called os.fsync; NFS host crash will lose the dump"
