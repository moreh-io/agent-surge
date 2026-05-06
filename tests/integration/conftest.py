"""Fixtures for multi-session real-CLI smoke tests.

Every test in this directory is opt-in via env vars and never runs in
plain CI:

- AGENTSURGE_E2E_FRONTENDS=1 enables the suite at all.
- AGENTSURGE_E2E_VLLM_URL points at the upstream OpenAI-compatible
  endpoint (e.g. `http://127.0.0.1:18000` with kubectl port-forward).
- AGENTSURGE_E2E_VLLM_API_KEY is the bearer token for that endpoint.
- AGENTSURGE_E2E_VLLM_MODEL is the served-model-name to benchmark.

Gating happens here in conftest.py rather than per-test so the missing-
env failure surfaces once with a clear message instead of three times.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_REQUIRED_ENV = (
    "AGENTSURGE_E2E_FRONTENDS",
    "AGENTSURGE_E2E_VLLM_URL",
    "AGENTSURGE_E2E_VLLM_API_KEY",
    "AGENTSURGE_E2E_VLLM_MODEL",
)


def _missing_env() -> list[str]:
    return [k for k in _REQUIRED_ENV if not os.environ.get(k)]


_REAL_CLI_SKIP = pytest.mark.skipif(
    bool(_missing_env()),
    reason=f"requires env: {', '.join(_REQUIRED_ENV)}",
)


@pytest.fixture
def vllm_env() -> dict[str, str]:
    return {
        "url": os.environ["AGENTSURGE_E2E_VLLM_URL"],
        "api_key": os.environ["AGENTSURGE_E2E_VLLM_API_KEY"],
        "model": os.environ["AGENTSURGE_E2E_VLLM_MODEL"],
    }


def run_agentsurge(
    *,
    frontend: str,
    model: str,
    server_url: str,
    api_key_env_pair: str,
    workspace_dir: Path,
    output_dir: Path,
    n_sessions: int,
    concurrency: int,
    session_timeout: int = 300,
) -> subprocess.CompletedProcess:
    """Run `python -m agentsurge run` with the given frontend config and
    return the completed process. Caller asserts on `.returncode` and
    `.stdout`."""
    cmd = [
        sys.executable,
        "-m",
        "agentsurge",
        "run",
        "--frontend",
        frontend,
        "--frontend-model",
        model,
        "--frontend-server-url",
        server_url,
        "--frontend-extra-env",
        api_key_env_pair,
        "--frontend-workspace-dir",
        str(workspace_dir),
        "--frontend-keep-artifacts",
        "always",
        "--frontend-session-timeout",
        str(session_timeout),
        "--client-concurrency",
        str(concurrency),
        "--n-sessions",
        str(n_sessions),
        "--backend",
        "mock",
        "--no-metrics",
        "--skip-validation",
        "--ignore-replay-output-length",
        "--output-dir",
        str(output_dir),
    ]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=session_timeout * (n_sessions // concurrency + 1) + 60,
        cwd=str(REPO_ROOT),
    )
