"""Multi-session real-CLI smoke tests against a real vLLM endpoint.

Each provider is exercised at concurrency=4, n-sessions=4. The goal is to
prove (a) the wiring established in batch 2 actually works concurrently
without races, (b) per-session state isolation (XDG_DATA_HOME for
OpenCode, CODEX_HOME for Codex, --bare for Claude) does its job, and
(c) the renderer's per-session artifact tree stays distinct.

Tests are env-gated; see conftest.py for required variables.
"""

from __future__ import annotations

from pathlib import Path

import pytest  # noqa: F401

from tests.integration.conftest import _REAL_CLI_SKIP, run_agentsurge


@_REAL_CLI_SKIP
def test_opencode_4_concurrent_smoke(tmp_path: Path, vllm_env: dict[str, str]) -> None:
    """4 concurrent OpenCode sessions must complete without WAL corruption.

    XDG_DATA_HOME isolation (Task 4) gives each session its own SQLite DB
    under <session_dir>/xdg/opencode/. Without that fix the shared
    ~/.local/share/opencode/opencode.db-{shm,wal} corrupts when two or
    more sessions run concurrently.
    """
    workspace = tmp_path / "ws"
    output = tmp_path / "results"
    proc = run_agentsurge(
        frontend="opencode",
        model=f"vllm/{vllm_env['model']}",
        server_url=vllm_env["url"],
        api_key_env_pair=f"OPENAI_API_KEY={vllm_env['api_key']}",
        workspace_dir=workspace,
        output_dir=output,
        n_sessions=4,
        concurrency=4,
        session_timeout=300,
    )
    assert proc.returncode == 0, (
        f"exit={proc.returncode}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    # Headline workload count must agree with --n-sessions, otherwise the
    # CLI dropped sessions (race / corrupt state).
    assert "4 (4 completed)" in proc.stdout, (
        f"expected 4 of 4 sessions complete; stdout was:\n{proc.stdout}"
    )
    # Per-session artifacts must all exist on disk.
    sessions_dir = workspace / "sessions"
    assert sessions_dir.is_dir()
    session_subdirs = sorted(sessions_dir.iterdir())
    assert len(session_subdirs) == 4, (
        f"expected 4 distinct session dirs, got {len(session_subdirs)}: "
        f"{[p.name for p in session_subdirs]}"
    )
    # Each session must have its own isolated XDG dir; this is the
    # regression guard for Task 4. If isolation breaks, all sessions
    # share one DB and the directories will be empty (or only one will
    # have content).
    for sd in session_subdirs:
        assert (sd / "xdg").is_dir(), f"missing xdg/ in {sd}"
        assert (sd / "opencode.json").is_file()
        assert (sd / "stdout.jsonl").stat().st_size > 0, f"empty stdout in {sd}"


@_REAL_CLI_SKIP
def test_claude_4_concurrent_smoke(tmp_path: Path, vllm_env: dict[str, str]) -> None:
    """4 concurrent Claude sessions must complete via vLLM /v1/messages.

    Claude isolation relies on `--bare` (suppresses OAuth/keychain/plugins)
    plus the ANTHROPIC_API_KEY → ANTHROPIC_AUTH_TOKEN mirror in
    ClaudeProvider.build_env so the vLLM endpoint receives a Bearer token
    rather than the x-api-key header it ignores.
    """
    workspace = tmp_path / "ws"
    output = tmp_path / "results"
    proc = run_agentsurge(
        frontend="claude",
        model=vllm_env["model"],
        server_url=vllm_env["url"],
        api_key_env_pair=f"ANTHROPIC_API_KEY={vllm_env['api_key']}",
        workspace_dir=workspace,
        output_dir=output,
        n_sessions=4,
        concurrency=4,
        session_timeout=300,
    )
    assert proc.returncode == 0, (
        f"exit={proc.returncode}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    assert "4 (4 completed)" in proc.stdout, (
        f"expected 4 of 4 sessions complete; stdout was:\n{proc.stdout}"
    )
    sessions_dir = workspace / "sessions"
    session_subdirs = sorted(sessions_dir.iterdir())
    assert len(session_subdirs) == 4
    for sd in session_subdirs:
        assert (sd / "stdout.jsonl").stat().st_size > 0, f"empty stdout in {sd}"
