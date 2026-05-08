# SPDX-License-Identifier: MIT
"""[audit:misc-core#M2] save_corpus must write atomically.

A crash, OOM, or KeyboardInterrupt mid-write must not leave the
canonical destination half-written and shadowing a previously good
file. Atomic = write to ``path.tmp`` then ``os.replace(tmp, path)`` so
either the new contents are visible OR the previous good file is still
there.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agentsurge import io as agent_io
from agentsurge.types import ReplaySession


def _mk_session(sid: str) -> ReplaySession:
    return ReplaySession(
        session_id=sid,
        turn_messages=[[{"role": "user", "content": "hi"}]],
        metadata={"source": "test"},
    )


def test_save_corpus_atomic_when_write_crashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Simulate a crash partway through writing the destination.

    Patch ``Path.write_bytes`` and ``Path.write_text`` so that a write
    targeted AT the canonical path raises after partial bytes have been
    flushed. The atomic implementation routes through a sibling .tmp file
    and ``os.replace``, so the canonical path is never directly written
    in the dangerous path. The pre-existing good file must survive.
    """
    target = tmp_path / "corpus.json"
    seed = '{"sessions": [], "manifest": {"sentinel": "old"}}'
    target.write_text(seed, encoding="utf-8")
    original_old = target.read_text(encoding="utf-8")

    # Track which paths get *direct* destructive writes. The atomic path
    # must write to a temp sibling, not to the canonical path itself.
    direct_writes: list[Path] = []

    real_write_bytes = Path.write_bytes
    real_write_text = Path.write_text

    def patched_write_bytes(self: Path, data, *args, **kwargs):  # type: ignore[no-untyped-def]
        if self == target:
            direct_writes.append(self)
            # Simulate partial bytes hitting the canonical path then crash.
            real_write_bytes(self, data[: max(1, len(data) // 4)])
            raise RuntimeError("simulated crash mid-write")
        return real_write_bytes(self, data, *args, **kwargs)

    def patched_write_text(self: Path, data, *args, **kwargs):  # type: ignore[no-untyped-def]
        if self == target:
            direct_writes.append(self)
            real_write_text(self, data[: max(1, len(data) // 4)])
            raise RuntimeError("simulated crash mid-write")
        return real_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_bytes", patched_write_bytes)
    monkeypatch.setattr(Path, "write_text", patched_write_text)

    # The atomic implementation must write to a tmp sibling, so the
    # canonical path is never the direct write target. Either:
    #   (a) save_corpus completes successfully (atomic rename), OR
    #   (b) save_corpus raises but our patched write was never called on
    #       the canonical path, so the prior good content survives.
    raised = False
    try:
        agent_io.save_corpus([_mk_session("s1")], target)
    except RuntimeError:
        raised = True

    if direct_writes:
        pytest.fail(
            f"save_corpus wrote directly to the canonical path "
            f"{[str(p) for p in direct_writes]!r}; expected tmp+rename"
        )

    assert target.exists(), "destination disappeared"
    if raised:
        # If the implementation crashed elsewhere, the previous file
        # must still be intact.
        assert target.read_text(encoding="utf-8") == original_old


def test_save_corpus_atomic_write_uses_unique_temp_siblings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated writes must not reuse one fixed ``path.tmp`` name."""
    target = tmp_path / "corpus.json"
    seen_tmp_names: list[str] = []
    real_replace = os.replace

    def tracking_replace(src, dst):  # type: ignore[no-untyped-def]
        seen_tmp_names.append(Path(src).name)
        return real_replace(src, dst)

    monkeypatch.setattr(agent_io.os, "replace", tracking_replace)

    agent_io.save_corpus([_mk_session("s1")], target)
    agent_io.save_corpus([_mk_session("s2")], target)

    assert len(seen_tmp_names) == 2
    assert len(set(seen_tmp_names)) == 2
    assert all(
        name.startswith(".corpus.json.") and name.endswith(".tmp") for name in seen_tmp_names
    )
