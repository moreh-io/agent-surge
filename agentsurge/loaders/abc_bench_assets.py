# SPDX-License-Identifier: MIT
"""ABC-Bench tarball fetch + extraction.

ABC-Bench (``OpenMOSS-Team/ABC-Bench``) ships its task repositories in a
single ``tasks.tar.gz`` asset (~2.6 GB, Git-LFS) rather than referencing
GitHub repos. This module downloads + extracts the tarball on first use
and resolves a ``task_id`` to its extracted directory.

Extracted layout (per task)::

    tasks/
    +-- task_<repo_slug>__<scenario>/
        +-- <repo_name>/          # repo source code
        +-- task.yaml
        +-- run-tests.sh
        +-- docker-compose.yaml
        +-- Dockerfile
        +-- solution.sh
        +-- tests/

Only the ``<repo_name>/`` directory is consumed by the sandbox - docker
execution of ``run-tests.sh`` is out of scope for the LLM serving benchmark.
"""

from __future__ import annotations

import logging
import os
import tarfile
from pathlib import Path

_log = logging.getLogger(__name__)

DEFAULT_ABC_BENCH_DIR = Path(os.path.expanduser("~/.cache/agentsurge/abc_bench"))
_REPO = "OpenMOSS-Team/ABC-Bench"
_TARBALL = "tasks.tar.gz"
_MARKER = ".extracted"

# Files shipped alongside the repo inside each task_<...>/ dir.
_NON_REPO_ENTRIES = frozenset(
    {
        "task.yaml",
        "run-tests.sh",
        "docker-compose.yaml",
        "Dockerfile",
        "solution.sh",
        "tests",
        "README.md",
    }
)


def ensure_abc_bench_tasks(cache_dir: Path | None = None) -> Path:
    """Download and extract ABC-Bench ``tasks.tar.gz``. Return the ``tasks/`` dir.

    Idempotent: skips if already extracted (marker file present).
    The tarball is ~2.6 GB; first call logs a one-time download notice.
    Concurrent callers are serialised by an ``fcntl.flock`` on
    ``cache_dir/.lock`` so only one process performs the extract.
    """
    import fcntl

    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_ABC_BENCH_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    marker = cache_dir / _MARKER
    tasks_dir = cache_dir / "tasks"
    if marker.exists() and tasks_dir.is_dir():
        return tasks_dir

    lock_path = cache_dir / ".lock"
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            # Re-check under the lock: a sibling worker may have completed
            # the extract while we were waiting.
            if marker.exists() and tasks_dir.is_dir():
                return tasks_dir

            try:
                from huggingface_hub import hf_hub_download
            except ImportError as e:
                raise RuntimeError(
                    "huggingface_hub is required for ABC-Bench --tool-mode real. "
                    "Install with: pip install huggingface_hub"
                ) from e

            _log.warning(
                "ABC-Bench: downloading %s (~2.6 GB, one-time) to %s",
                _TARBALL,
                cache_dir,
            )
            tar_path = hf_hub_download(
                _REPO,
                _TARBALL,
                repo_type="dataset",
                cache_dir=str(cache_dir / "_hf_cache"),
            )

            _log.info("ABC-Bench: extracting %s -> %s", tar_path, cache_dir)
            with tarfile.open(tar_path, "r:gz") as tf:
                # filter="data" blocks path traversal and symlink-outside attacks (py>=3.12).
                tf.extractall(cache_dir, filter="data")

            if not tasks_dir.is_dir():
                raise RuntimeError(f"ABC-Bench: expected {tasks_dir} after extraction, not found")
            marker.touch()
            return tasks_dir
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


def get_task_dir(task_id: str, cache_dir: Path | None = None) -> Path:
    """Return the extracted task directory for *task_id*."""
    tasks_dir = ensure_abc_bench_tasks(cache_dir)
    p = tasks_dir / task_id
    if not p.is_dir():
        raise FileNotFoundError(
            f"ABC-Bench task directory missing: {p} (expected under {tasks_dir})"
        )
    return p


def find_repo_dir(task_dir: Path) -> Path | None:
    """Locate the ``<repo_name>/`` subdirectory inside a task directory.

    Heuristic: the first subdirectory whose name is not a known non-repo
    sibling (task.yaml, tests, etc.).
    """
    for child in sorted(task_dir.iterdir()):
        if child.is_dir() and child.name not in _NON_REPO_ENTRIES:
            return child
    return None
