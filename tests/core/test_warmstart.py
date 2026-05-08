"""Tests for agentsurge.warmstart module."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from agentsurge.types import ReplaySession
from agentsurge.warmstart import (
    _copy_and_checkout,
    _extract_repo_info,
    prepare_execution_workspace,
    prepare_sandbox,
)


def _session(session_id: str = "test_1", **meta) -> ReplaySession:
    return ReplaySession(
        session_id=session_id,
        turn_messages=[[{"role": "user", "content": "test"}]],
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# _extract_repo_info
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "meta,expected",
    [
        pytest.param(
            {
                "instance_id": "django__django-12345",
                "repo": "django/django",
                "base_commit": "abc123",
            },
            ("django/django", "abc123", ""),
            id="swe_bench_format",
        ),
        pytest.param(
            {"instance_id": "astropy__astropy.abc12345.v1__extra"},
            ("astropy/astropy", "abc12345", "v1"),
            id="three_part_instance_id",
        ),
        pytest.param(
            {"instance_id": ""},
            ("", "", ""),
            id="no_repo",
        ),
    ],
)
def test_extract_repo_info(meta, expected):
    s = _session(**meta)
    assert _extract_repo_info(s) == expected


# ---------------------------------------------------------------------------
# prepare_execution_workspace
# ---------------------------------------------------------------------------


def test_prepare_execution_workspace_skips_no_repo_sessions(tmp_path):
    workspace_dir = tmp_path / "workspace"
    cache_dir = tmp_path / "cache"

    s = _session(session_id="empty", instance_id="")

    manifest = prepare_execution_workspace([s], workspace_dir, cache_dir)

    assert manifest["workspace_dir"] == str(workspace_dir)
    assert manifest["sandbox_dir"] == str(workspace_dir)
    assert manifest["skipped"] == 1
    assert manifest["completed"] == 0
    assert manifest["sessions"][0]["status"] == "skipped_no_repo"


def test_prepare_execution_workspace_copies_abc_bench_task(tmp_path):
    """ABC-Bench sessions skip git clone and copy from the extracted tarball."""
    workspace_dir = tmp_path / "workspace"
    cache_dir = tmp_path / "cache"
    fake_tasks_root = tmp_path / "abc_tasks"
    task_dir = fake_tasks_root / "task_acme__scenario"
    repo_src = task_dir / "acme_repo"
    repo_src.mkdir(parents=True)
    (repo_src / "main.py").write_text("print('hello')\n")
    (task_dir / "task.yaml").write_text("name: scenario\n")

    s = _session(
        session_id="abc_s1",
        source="abc-bench",
        task_id="task_acme__scenario",
    )

    with (
        patch(
            "agentsurge.loaders.abc_bench_assets.get_task_dir",
            return_value=task_dir,
        ),
    ):
        manifest = prepare_execution_workspace([s], workspace_dir, cache_dir)

    copied_repo = workspace_dir / "abc_s1" / "workspace" / "acme_repo"
    assert (copied_repo / "main.py").exists()
    assert manifest["completed"] == 1
    assert manifest["fail"] == 0
    assert manifest["sessions"][0]["status"] == "ok"
    assert manifest["sessions"][0]["repo"] == "abc-bench"


def test_prepare_execution_workspace_records_clone_failure_without_aborting(tmp_path):
    """If git clone fails for a repo, prepare_execution_workspace must record the
    failure in the manifest (status='clone_failed') instead of crashing
    the run. Silent workspace-setup corruption is worse than a tagged
    failure that the operator can see in MANIFEST.json.
    """
    workspace_dir = tmp_path / "workspace"
    cache_dir = tmp_path / "cache"

    sessions = [
        _session(
            session_id=f"s{i}",
            instance_id=f"broken__repo-{i}",
            repo="broken/repo",
            base_commit="deadbeef",
        )
        for i in range(3)
    ]

    def mock_run(args, **kwargs):
        result = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
        if args[:2] == ["git", "clone"]:
            result.returncode = 128
            result.stdout = ""
            result.stderr = (
                "remote: Repository not found.\nfatal: repository 'broken/repo' not found\n"
            )
        else:
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
        return result

    with patch("agentsurge.warmstart.subprocess.run", side_effect=mock_run):
        manifest = prepare_execution_workspace(sessions, workspace_dir, cache_dir)

    assert manifest["fail"] == 3
    assert manifest["completed"] == 0
    assert all(s["status"] == "clone_failed" for s in manifest["sessions"])
    assert all(s["repo"] == "broken/repo" for s in manifest["sessions"])
    # MANIFEST.json is the operator's ground truth for per-session status.
    manifest_path = workspace_dir / "MANIFEST.json"
    assert manifest_path.exists()
    import json

    on_disk = json.loads(manifest_path.read_text())
    assert on_disk["fail"] == 3


def test_prepare_sandbox_legacy_alias(tmp_path):
    workspace_dir = tmp_path / "legacy_workspace"
    cache_dir = tmp_path / "cache"
    s = _session(session_id="empty", instance_id="")

    manifest = prepare_sandbox([s], workspace_dir, cache_dir)

    assert manifest["workspace_dir"] == str(workspace_dir)
    assert manifest["sandbox_dir"] == str(workspace_dir)


def test_copy_and_checkout_tolerates_broken_symlinks(tmp_path):
    """A cached clone with a broken in-repo symlink must not fail the copy.

    SWE-bench / SWE-smith repos occasionally ship intentional dangling
    symlinks as negative-test fixtures; the security-motivated switch to
    symlinks=False previously made shutil.copytree raise shutil.Error at
    end-of-copy for any such repo.
    """
    cached = tmp_path / "cached_clone"
    cached.mkdir()
    (cached / "README.md").write_text("hi")
    # Broken symlink - target does not exist
    os.symlink("/nonexistent_target_xyz", cached / "broken_link")

    session_dir = tmp_path / "session"
    _copy_and_checkout(cached, session_dir, "repo", base_commit="")

    workspace = session_dir / "workspace" / "repo"
    assert workspace.is_dir()
    assert (workspace / "README.md").read_text() == "hi"
    # Dangling symlink was skipped, not materialised.
    assert not (workspace / "broken_link").exists()
    assert not (workspace / "broken_link").is_symlink()
