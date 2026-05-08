# SPDX-License-Identifier: MIT
"""[audit:tool#M2] real_execute_async must not block the event loop on FS ops.

``real_execute_async`` only overrode subprocess-spawning tools (bash,
execute_bash, search_code) and otherwise delegated to the synchronous
``real_execute``. For filesystem tools (read_file, write_file, edit_file,
str_replace_editor) this means a multi-MB ``Path.read_text`` runs synchronously
inside the loop — under high concurrency one slow read stalls every other
session sharing the loop.

We assert that filesystem-tool reads are dispatched off-thread via
``asyncio.to_thread`` (or equivalent), mirroring the recent fixes for
``_setup_sandbox`` and ``_maybe_warmstart`` on master. Concretely: we make
``Path.read_text`` block for ~120ms on a sentinel path, schedule a
``read_file`` tool call concurrently with a fast async sentinel, and assert
the sentinel completes BEFORE the read_file does — proving the read isn't
hogging the loop.
"""

import asyncio
import time

import pytest

from agentsurge.tool_call import ExecutionWorkspace, real_execute_async


@pytest.mark.asyncio
async def test_read_file_does_not_block_event_loop(tmp_path, monkeypatch):
    ws = ExecutionWorkspace(str(tmp_path))
    target = ws.root / "big.txt"
    target.write_text("hello\n")

    # Patch Path.read_text on this specific file to sleep before returning,
    # simulating a slow disk read.
    from pathlib import Path as _Path

    real_read_text = _Path.read_text

    def slow_read_text(self, *a, **kw):
        if self.name == "big.txt":
            time.sleep(0.12)  # 120ms blocking sleep — must NOT pin the loop
        return real_read_text(self, *a, **kw)

    monkeypatch.setattr(_Path, "read_text", slow_read_text)

    sentinel_done_at: list[float] = []

    async def fast_sentinel():
        # Yields to the loop, sleeps 10ms, then records completion time.
        await asyncio.sleep(0.01)
        sentinel_done_at.append(time.monotonic())

    t0 = time.monotonic()
    sentinel_task = asyncio.create_task(fast_sentinel())
    read_task = asyncio.create_task(real_execute_async("read_file", {"path": "big.txt"}, ws))
    out = await read_task
    read_done_at = time.monotonic()
    await sentinel_task

    assert "hello" in out
    sentinel_elapsed = sentinel_done_at[0] - t0
    read_elapsed = read_done_at - t0
    # The sentinel finishes ~10ms in. If real_execute_async pinned the loop
    # on the 120ms slow read, the sentinel would wait at least 120ms before
    # finishing. Assert the sentinel completes well before the read.
    assert sentinel_elapsed < 0.08, (
        f"event loop appears blocked: sentinel finished after "
        f"{sentinel_elapsed * 1000:.0f}ms (expected <80ms); "
        f"read_file finished after {read_elapsed * 1000:.0f}ms"
    )
    assert read_elapsed >= 0.10, (
        f"slow read should still be observable end-to-end "
        f"(>= 100ms); got {read_elapsed * 1000:.0f}ms"
    )


@pytest.mark.asyncio
async def test_write_file_does_not_block_event_loop(tmp_path, monkeypatch):
    ws = ExecutionWorkspace(str(tmp_path))
    from pathlib import Path as _Path

    real_write_text = _Path.write_text

    def slow_write_text(self, *a, **kw):
        if self.name == "out.txt":
            time.sleep(0.12)
        return real_write_text(self, *a, **kw)

    monkeypatch.setattr(_Path, "write_text", slow_write_text)

    sentinel_done_at: list[float] = []

    async def fast_sentinel():
        await asyncio.sleep(0.01)
        sentinel_done_at.append(time.monotonic())

    t0 = time.monotonic()
    sentinel_task = asyncio.create_task(fast_sentinel())
    write_task = asyncio.create_task(
        real_execute_async("write_file", {"path": "out.txt", "content": "x"}, ws)
    )
    await write_task
    await sentinel_task

    sentinel_elapsed = sentinel_done_at[0] - t0
    assert sentinel_elapsed < 0.08, (
        f"event loop appears blocked on write: sentinel finished after "
        f"{sentinel_elapsed * 1000:.0f}ms (expected <80ms)"
    )
