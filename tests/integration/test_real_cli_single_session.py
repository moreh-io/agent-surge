# SPDX-License-Identifier: MIT
"""Single-session smoke tests against the real codex/claude/opencode CLIs.

These spawn the real binary against a tiny prompt and assert the renderer
collects benchmark-relevant signal: usage tokens, a final-message
timestamp, and on-disk artifacts. They do **not** require a vLLM
endpoint — each CLI uses whatever upstream it's already configured for
(Anthropic API, OpenAI, etc.) via the user's installed credentials.

Gated on ``AGENTSURGE_E2E_FRONTENDS=1`` because they cost real API
tokens and need the binaries installed. Live here in ``tests/integration/``
rather than ``tests/core/`` so a default ``pytest`` run stays hermetic;
opt in via ``pytest -m real_cli`` (see pytest.ini).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agentsurge.frontends.runner import FrontendSessionRenderer
from agentsurge.types import (
    BenchmarkConfig,
    FrontendRuntimeSettings,
    ReplaySession,
)

pytestmark = pytest.mark.real_cli

_REAL_CLI_SKIP = pytest.mark.skipif(
    os.environ.get("AGENTSURGE_E2E_FRONTENDS") != "1",
    reason="Real CLI tests skipped; set AGENTSURGE_E2E_FRONTENDS=1 to enable.",
)

_REAL_PROMPT = "Print exactly one short sentence. Do not call any tools."


def _real_session(sid: str) -> ReplaySession:
    return ReplaySession(
        session_id=sid,
        turn_messages=[[{"role": "user", "content": _REAL_PROMPT}]],
        metadata={},
    )


def _real_config(name: str, workspace: Path) -> BenchmarkConfig:
    return BenchmarkConfig(
        vllm_url="http://x",
        model="t",
        no_metrics=True,
        frontend=FrontendRuntimeSettings(
            name=name,
            session_timeout_s=60.0,
            keep_artifacts="always",
            workspace_dir=str(workspace),
        ),
    )


def _assert_real_smoke_success(fm, sid: str) -> None:
    """Shared assertions for single-session real-CLI smoke runs.

    A bare exit_code==0 + event_count>0 check accepts any garbage CLI output
    as success. These assertions require the run actually produced
    benchmark-relevant signal: usage tokens, a final-message timestamp, and
    on-disk artifacts the user can inspect for debugging.
    """
    assert fm is not None, f"{sid}: no frontend_metrics on result"
    assert fm.process_exit_code == 0, (
        f"{sid}: process_exit_code={fm.process_exit_code}, failure_category={fm.failure_category}"
    )
    assert fm.failure_category is None, f"{sid}: failure_category={fm.failure_category}"
    # Must have at least final-message + one preceding event.
    assert fm.event_count > 1, f"{sid}: event_count={fm.event_count}"
    # Real CLI runs always produce a usage payload with non-zero output.
    # Parser key normalization differs per provider; accept either.
    assert fm.provider_usage is not None, f"{sid}: provider_usage missing"
    out_tokens = fm.provider_usage.get("output_tokens") or fm.provider_usage.get("output") or 0
    assert out_tokens > 0, f"{sid}: output_tokens={out_tokens} in {fm.provider_usage}"
    # Timing must populate or the benchmark is meaningless.
    assert fm.time_to_final_message_ms is not None, f"{sid}: time_to_final_message_ms is None"
    assert fm.time_to_final_message_ms > 0, (
        f"{sid}: time_to_final_message_ms={fm.time_to_final_message_ms}"
    )
    # Artifact directory must exist and contain the captured stream.
    assert fm.artifact_dir is not None, f"{sid}: artifact_dir is None"
    artifact_dir = Path(fm.artifact_dir)
    assert artifact_dir.is_dir(), f"{sid}: artifact_dir missing on disk"
    stdout_file = artifact_dir / "stdout.jsonl"
    assert stdout_file.is_file(), f"{sid}: stdout.jsonl not written"
    assert stdout_file.stat().st_size > 0, f"{sid}: stdout.jsonl is empty"


@_REAL_CLI_SKIP
@pytest.mark.asyncio
async def test_codex_real_binary_smoke(tmp_path: Path):
    cfg = _real_config("codex", tmp_path)
    renderer = FrontendSessionRenderer(cfg, tmp_path)
    result = await renderer.run(_real_session("s_codex_real"))
    _assert_real_smoke_success(result.frontend_metrics, "codex")


@_REAL_CLI_SKIP
@pytest.mark.asyncio
async def test_claude_real_binary_smoke(tmp_path: Path):
    cfg = _real_config("claude", tmp_path)
    renderer = FrontendSessionRenderer(cfg, tmp_path)
    result = await renderer.run(_real_session("s_claude_real"))
    _assert_real_smoke_success(result.frontend_metrics, "claude")


@_REAL_CLI_SKIP
@pytest.mark.asyncio
async def test_opencode_real_binary_smoke(tmp_path: Path):
    cfg = _real_config("opencode", tmp_path)
    renderer = FrontendSessionRenderer(cfg, tmp_path)
    result = await renderer.run(_real_session("s_opencode_real"))
    # OpenCode signals failure via is_error semantics, not exit code, but a
    # successful smoke run still populates usage and artifacts identically.
    _assert_real_smoke_success(result.frontend_metrics, "opencode")
