# SPDX-License-Identifier: MIT
"""[audit:misc-core#M7] warmstart MANIFEST.json must be written atomically.

Two concurrent benchmark runs sharing the default workspace cache race
on ``MANIFEST.json``. The previous code did ``open(..., "w"); json.dump``
— if either writer is interrupted between truncate and the final
buffer flush, the manifest is corrupted. Fix: tmp + rename.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from agentsurge import warmstart


def test_manifest_helper_exists() -> None:
    helper = getattr(warmstart, "_write_manifest_atomic", None)
    assert helper is not None and callable(helper), (
        "warmstart must expose _write_manifest_atomic for atomic writes"
    )


def test_manifest_atomic_helper_does_not_corrupt_on_crash(tmp_path: Path) -> None:
    target = tmp_path / "MANIFEST.json"
    target.write_text('{"sentinel": "previous-good"}', encoding="utf-8")
    original = target.read_text(encoding="utf-8")

    real_write_text = Path.write_text
    real_write_bytes = Path.write_bytes
    direct_writes: list[Path] = []

    def patched_write_text(self: Path, data, *a, **k):  # type: ignore[no-untyped-def]
        if self == target:
            direct_writes.append(self)
            real_write_text(self, data[: max(1, len(data) // 4)])
            raise RuntimeError("simulated crash mid-write")
        return real_write_text(self, data, *a, **k)

    def patched_write_bytes(self: Path, data, *a, **k):  # type: ignore[no-untyped-def]
        if self == target:
            direct_writes.append(self)
            real_write_bytes(self, data[: max(1, len(data) // 4)])
            raise RuntimeError("simulated crash mid-write")
        return real_write_bytes(self, data, *a, **k)

    with (
        patch.object(Path, "write_text", patched_write_text),
        patch.object(Path, "write_bytes", patched_write_bytes),
    ):
        with contextlib.suppress(RuntimeError):
            warmstart._write_manifest_atomic(target, {"completed": 1})

    if direct_writes:
        pytest.fail(
            f"MANIFEST written non-atomically — direct write to {[str(p) for p in direct_writes]!r}"
        )

    # If the atomic helper did its job, the canonical path was never
    # opened for direct write. So either:
    #   (a) helper completed — file now contains the new content, OR
    #   (b) helper crashed before os.replace — file still has *original*.
    # Both are valid atomic outcomes. The bug case (truncated/partial)
    # would have target.read_text() be neither original nor a valid JSON
    # dump of {"completed": 1}.
    import json as _json

    text = target.read_text(encoding="utf-8")
    assert text == original or _json.loads(text) == {"completed": 1}, (
        f"MANIFEST left in partial/corrupt state: {text!r}"
    )


def test_manifest_atomic_helper_uses_unique_temp_siblings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "MANIFEST.json"
    seen_tmp_names: list[str] = []
    real_replace = os.replace

    def tracking_replace(src, dst):  # type: ignore[no-untyped-def]
        seen_tmp_names.append(Path(src).name)
        return real_replace(src, dst)

    monkeypatch.setattr(warmstart.os, "replace", tracking_replace)

    warmstart._write_manifest_atomic(target, {"completed": 1})
    warmstart._write_manifest_atomic(target, {"completed": 2})

    assert len(seen_tmp_names) == 2
    assert len(set(seen_tmp_names)) == 2
    assert all(
        name.startswith(".MANIFEST.json.") and name.endswith(".tmp") for name in seen_tmp_names
    )
