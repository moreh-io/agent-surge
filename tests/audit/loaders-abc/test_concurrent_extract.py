"""M6: ensure_abc_bench_tasks must guard concurrent first-run extracts.

Two threads that both miss the marker must not both run extractall.
The current code has no lock, so concurrent calls each invoke
``tarfile.open(...).extractall``, producing partial trees.  The fix is
to wrap the extract+marker in a file lock.
"""

from __future__ import annotations

import tarfile
import threading
from pathlib import Path


def _make_tarball(tar_path: Path) -> None:
    src = tar_path.parent / "_src" / "tasks"
    src.mkdir(parents=True, exist_ok=True)
    (src / "task_demo").mkdir(exist_ok=True)
    (src / "task_demo" / "task.yaml").write_text("name: demo\n")
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(src, arcname="tasks")


def test_concurrent_callers_extract_only_once(tmp_path, monkeypatch):
    from agentsurge.loaders import abc_bench_assets

    src_tar = tmp_path / "tasks.tar.gz"
    _make_tarball(src_tar)
    cache_dir = tmp_path / "cache"

    # ``ensure_abc_bench_tasks`` does ``from huggingface_hub import hf_hub_download``
    # inside the function, so we stub the module in sys.modules.
    import sys
    import types

    fake_hh = types.ModuleType("huggingface_hub")
    fake_hh.hf_hub_download = lambda *args, **kwargs: str(src_tar)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hh)

    extract_count = {"n": 0}
    count_lock = threading.Lock()
    real_open = abc_bench_assets.tarfile.open

    def counting_open(*args, **kwargs):
        with count_lock:
            extract_count["n"] += 1
        return real_open(*args, **kwargs)

    monkeypatch.setattr(abc_bench_assets.tarfile, "open", counting_open)

    # Synchronise thread starts so they all race the marker check.
    barrier = threading.Barrier(4)
    results: dict[int, Path] = {}
    errors: dict[int, BaseException] = {}

    def worker(idx: int) -> None:
        try:
            barrier.wait()
            results[idx] = abc_bench_assets.ensure_abc_bench_tasks(cache_dir)
        except BaseException as exc:  # pragma: no cover - error path
            errors[idx] = exc

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"worker errors: {errors!r}"
    assert len({str(v) for v in results.values()}) == 1
    assert extract_count["n"] <= 1, (
        f"expected <=1 tarfile.open calls under lock, got {extract_count['n']}"
    )
    assert (cache_dir / "tasks" / "task_demo").is_dir()
