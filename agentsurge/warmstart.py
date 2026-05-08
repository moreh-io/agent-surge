# SPDX-License-Identifier: MIT
"""Warm-start execution workspace preparation with shared clone cache.

Clones each unique GitHub repo once into a cache directory, then copies
to per-session workspace directories and checks out the required commit.
This turns N clones of the same repo into 1 clone + (N-1) fast copies.

Session metadata must follow SWE-bench conventions (instance_id with
``owner__repo`` format, optional ``repo``, ``base_commit``, ``version``
fields).  Only public GitHub repositories are supported.

Intended to be called from the runner's pre-run phase when tool_mode="real"::

    from agentsurge.warmstart import prepare_execution_workspace

    manifest = prepare_execution_workspace(sessions, workspace_dir, cache_dir)
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentsurge.types import ReplaySession

_log = logging.getLogger(__name__)

DEFAULT_WORKSPACE_DIR = Path(os.path.expanduser("~/.cache/agentsurge/working_repo"))
DEFAULT_SANDBOX_DIR = DEFAULT_WORKSPACE_DIR
DEFAULT_CACHE_DIR = Path(os.path.expanduser("~/.cache/agentsurge/repo_cache"))


def _write_manifest_atomic(path: Path, manifest: dict) -> None:
    """Write *manifest* to *path* atomically (tmp + ``os.replace``).

    Two concurrent runs sharing the default workspace dir race on the
    manifest. A naive ``open(..., "w") + json.dump`` can leave the file
    truncated or interleaved on crash. A tmp sibling + atomic rename
    guarantees readers see either the prior good content or the new
    content — never a partial write.
    """
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp = Path(tmp_name)
    payload = json.dumps(manifest, indent=2).encode("utf-8")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            with contextlib.suppress(OSError):
                tmp.unlink()


def _git_head_sha(repo_dir: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        timeout=10,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _extract_repo_info(session: ReplaySession) -> tuple[str, str, str]:
    """Extract (repo_slug, base_commit, version) from a ReplaySession.

    Must stay in sync with ``ExecutionWorkspace.setup_from_instance_id`` in
    tool_call.py - both parse the same instance_id format to derive workspace
    paths.
    """
    meta = session.metadata
    instance_id = meta.get("instance_id", "")
    repo = meta.get("repo") or meta.get("repository") or meta.get("project") or ""
    base_commit = meta.get("base_commit", "")
    version = meta.get("version", "")

    if not repo:
        from agentsurge.tool_call import _parse_owner_repo

        owner, repo_name = _parse_owner_repo(instance_id)
        if owner and repo_name:
            repo = f"{owner}/{repo_name}"
            parts = instance_id.split("__")
            if len(parts) >= 3:
                dot_parts = parts[1].split(".")
                commit_re = re.compile(r"^[0-9a-f]{7,40}$")
                if not base_commit and len(dot_parts) >= 2 and commit_re.fullmatch(dot_parts[1]):
                    base_commit = dot_parts[1]
                if not version and len(dot_parts) >= 3:
                    version = dot_parts[2]

    return repo, base_commit, version


def _workspace_dir_name(instance_id: str, version: str) -> str:
    """Build the workspace subdirectory name, matching ExecutionWorkspace setup."""
    parts = instance_id.split("__")
    if len(parts) == 2 and version:
        owner = parts[0]
        repo_name = re.sub(r"-\d+$", "", parts[1])
        return f"{owner}__{repo_name}__{version}"
    return instance_id


def _repo_cache_key(repo: str) -> str:
    return repo.replace("/", "__")


# ---------------------------------------------------------------------------
# Low-level: clone into cache, copy to session
# ---------------------------------------------------------------------------


def _clone_repo_to_cache(repo: str, cache_dir: Path, timeout: int = 600) -> Path:
    """Clone a repo into cache_dir (full clone). Returns the cached path.

    If the repo is already cached, returns immediately.
    """
    key = _repo_cache_key(repo)
    dest = cache_dir / key

    if dest.exists():
        _log.info("Cache hit: %s", repo)
        return dest

    repo_url = f"https://github.com/{repo}.git"
    _log.info("Cloning %s (full clone, timeout=%ds)", repo_url, timeout)

    proc = subprocess.run(
        ["git", "clone", repo_url, str(dest)],
        timeout=timeout,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        raise RuntimeError(f"git clone failed for {repo}: {proc.stderr[:500]}")

    return dest


def _copy_and_checkout(
    cached_clone: Path,
    session_dir: Path,
    dir_name: str,
    base_commit: str,
) -> str:
    """Copy cached clone into the session workspace and checkout the commit.

    Returns the HEAD SHA after checkout.  If the target already exists
    (session previously prepared), returns its current HEAD without copying.
    """
    workspace = session_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    clone_target = workspace / dir_name

    if clone_target.exists():
        return _git_head_sha(clone_target)

    # symlinks=False: cached clones hold arbitrary GitHub content, so an
    # in-repo symlink like `link -> /etc` would otherwise be materialised
    # into the per-session workspace verbatim and escape the workspace root.
    # ignore_dangling_symlinks=True so repos that ship intentional broken
    # symlinks (common in SWE-bench test fixtures) do not fail the copy.
    shutil.copytree(
        str(cached_clone),
        str(clone_target),
        symlinks=False,
        ignore_dangling_symlinks=True,
    )

    if base_commit:
        proc = subprocess.run(
            ["git", "checkout", base_commit],
            cwd=str(clone_target),
            timeout=60,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            _log.warning(
                "checkout %s failed in %s: %s",
                base_commit,
                clone_target,
                proc.stderr[:200],
            )

    return _git_head_sha(clone_target)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def prepare_execution_workspace(
    sessions: list[ReplaySession],
    workspace_dir: Path | None = None,
    cache_dir: Path | None = None,
    timeout: int = 600,
) -> dict:
    """Prepare warm-start workspace directories for a list of sessions.

    For each unique repo among the sessions, clones it once into *cache_dir*,
    then copies to ``workspace_dir/{session_id}/workspace/{dir_name}`` and
    checks out the session's base_commit.  Sessions whose workspace directory already
    exists are skipped (idempotent).

    Parameters
    ----------
    sessions:
        ReplaySession objects.  Must have ``session_id`` and ``metadata``
        containing at least ``instance_id`` (plus optionally ``repo``,
        ``base_commit``, ``version``).
    workspace_dir:
        Root directory for per-session workspaces.
        Default ``~/.cache/agentsurge/working_repo/``.
    cache_dir:
        Shared clone cache.
        Default ``~/.cache/agentsurge/repo_cache/``.
    timeout:
        Seconds to allow for each initial git clone.

    Returns
    -------
    dict
        Manifest with per-session ``head_sha`` and aggregate stats.
        Also written to ``workspace_dir/MANIFEST.json``.
    """
    if workspace_dir is None:
        workspace_dir = DEFAULT_WORKSPACE_DIR
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    workspace_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    _log.info("Workspace dir: %s", workspace_dir)
    _log.info("Cache dir:   %s", cache_dir)

    repo_to_entries: dict[str, list[tuple[ReplaySession, str, str]]] = defaultdict(list)
    skipped_no_repo: list[str] = []
    abc_bench_sessions: list[ReplaySession] = []

    for s in sessions:
        if s.metadata.get("source") == "abc-bench":
            abc_bench_sessions.append(s)
            continue
        repo, base_commit, version = _extract_repo_info(s)
        if not repo:
            skipped_no_repo.append(s.session_id)
            continue
        instance_id = s.metadata.get("instance_id", s.session_id)
        dir_name = _workspace_dir_name(instance_id, version)
        repo_to_entries[repo].append((s, base_commit, dir_name))

    unique_repos = list(repo_to_entries.keys())
    total_cloneable = sum(len(v) for v in repo_to_entries.values())

    _log.info(
        "warmstart: %d sessions, %d unique repos, %d cloneable, %d skipped (no repo)",
        len(sessions),
        len(unique_repos),
        total_cloneable,
        len(skipped_no_repo),
    )

    ok = 0
    fail = 0
    session_results: list[dict] = []
    t_start = time.monotonic()

    for repo_idx, (repo, entries) in enumerate(sorted(repo_to_entries.items()), 1):
        _log.info("[%d/%d] %s (%d sessions)", repo_idx, len(unique_repos), repo, len(entries))
        t_clone = time.monotonic()

        try:
            cached = _clone_repo_to_cache(repo, cache_dir, timeout=timeout)
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            _log.error("Clone failed for %s: %s", repo, exc)
            for s, base_commit, _dir_name in entries:
                fail += 1
                session_results.append(
                    {
                        "session_id": s.session_id,
                        "instance_id": s.metadata.get("instance_id", ""),
                        "repo": repo,
                        "base_commit": base_commit,
                        "head_sha": "",
                        "status": "clone_failed",
                    }
                )
            continue

        clone_elapsed = time.monotonic() - t_clone
        _log.info("  Cloned %s in %.1fs", repo, clone_elapsed)

        for sess_idx, (s, base_commit, dir_name) in enumerate(entries, 1):
            session_dir = workspace_dir / s.session_id
            t_copy = time.monotonic()

            try:
                sha = _copy_and_checkout(cached, session_dir, dir_name, base_commit)
                ok += 1
                copy_elapsed = time.monotonic() - t_copy
                _log.info(
                    "  [%d/%d] %s -> %s (HEAD=%s) %.1fs",
                    sess_idx,
                    len(entries),
                    s.session_id,
                    dir_name,
                    sha[:8],
                    copy_elapsed,
                )
                session_results.append(
                    {
                        "session_id": s.session_id,
                        "instance_id": s.metadata.get("instance_id", ""),
                        "repo": repo,
                        "base_commit": base_commit,
                        "head_sha": sha,
                        "status": "ok",
                    }
                )
            except Exception as exc:
                fail += 1
                _log.error("  [%d/%d] FAIL %s: %s", sess_idx, len(entries), s.session_id, exc)
                session_results.append(
                    {
                        "session_id": s.session_id,
                        "instance_id": s.metadata.get("instance_id", ""),
                        "repo": repo,
                        "base_commit": base_commit,
                        "head_sha": "",
                        "status": "copy_failed",
                    }
                )

    if abc_bench_sessions:
        _log.info("warmstart: preparing %d ABC-Bench sessions", len(abc_bench_sessions))
        from agentsurge.loaders.abc_bench_assets import find_repo_dir, get_task_dir

        for s in abc_bench_sessions:
            session_dir = workspace_dir / s.session_id
            task_id = s.metadata.get("task_id") or s.metadata.get("instance_id", "")
            try:
                task_dir = get_task_dir(task_id)
                repo_src = find_repo_dir(task_dir)
                if repo_src is None:
                    raise RuntimeError(f"no repo subdirectory in {task_dir}")
                workspace = session_dir / "workspace"
                workspace.mkdir(parents=True, exist_ok=True)
                clone_target = workspace / repo_src.name
                if not clone_target.exists():
                    # symlinks=False + ignore_dangling_symlinks=True for the
                    # same workspace-escape + broken-fixture reasons as above.
                    shutil.copytree(
                        str(repo_src),
                        str(clone_target),
                        symlinks=False,
                        ignore_dangling_symlinks=True,
                    )
                ok += 1
                session_results.append(
                    {
                        "session_id": s.session_id,
                        "instance_id": task_id,
                        "repo": "abc-bench",
                        "base_commit": "",
                        "head_sha": "",
                        "status": "ok",
                    }
                )
            except Exception as exc:
                fail += 1
                _log.error("ABC-Bench copy failed for %s: %s", s.session_id, exc)
                session_results.append(
                    {
                        "session_id": s.session_id,
                        "instance_id": task_id,
                        "repo": "abc-bench",
                        "base_commit": "",
                        "head_sha": "",
                        "status": "copy_failed",
                    }
                )

    for sid in skipped_no_repo:
        session_results.append(
            {
                "session_id": sid,
                "instance_id": "",
                "repo": "",
                "base_commit": "",
                "head_sha": "",
                "status": "skipped_no_repo",
            }
        )

    elapsed = time.monotonic() - t_start

    manifest = {
        "workspace_dir": str(workspace_dir),
        "sandbox_dir": str(workspace_dir),
        "cache_dir": str(cache_dir),
        "n_sessions": len(sessions),
        "unique_repos": len(unique_repos),
        "completed": ok,
        "fail": fail,
        "skipped": len(skipped_no_repo),
        "elapsed_s": round(elapsed, 1),
        "sessions": session_results,
    }

    manifest_path = workspace_dir / "MANIFEST.json"
    _write_manifest_atomic(manifest_path, manifest)

    _log.info(
        "warmstart done in %.1fs: %d ok, %d failed, %d skipped",
        elapsed,
        ok,
        fail,
        len(skipped_no_repo),
    )

    return manifest


def prepare_sandbox(
    sessions: list[ReplaySession],
    sandbox_dir: Path | None = None,
    cache_dir: Path | None = None,
    timeout: int = 600,
) -> dict:
    """Compatibility alias for :func:`prepare_execution_workspace`."""
    return prepare_execution_workspace(
        sessions,
        workspace_dir=sandbox_dir,
        cache_dir=cache_dir,
        timeout=timeout,
    )
