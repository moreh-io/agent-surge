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
import socket
import subprocess
import sys
import time
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


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(port: int, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with socket.socket() as s:
            s.settimeout(0.5)
            try:
                s.connect(("127.0.0.1", port))
                return
            except OSError:
                time.sleep(0.1)
    raise RuntimeError(f"port {port} did not become reachable within {timeout_s}s")


@pytest.fixture
def translator_shim(vllm_env):
    """Spawn `agentsurge proxy-translator` pointed at the vLLM endpoint.
    Yields the translator's listen URL. Used by Codex tests so requests
    flow Codex → shim (developer→system rewrite) → vLLM."""
    port = _find_free_port()
    target = vllm_env["url"]
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "agentsurge",
            "proxy-translator",
            "--listen",
            f"127.0.0.1:{port}",
            "--target",
            target,
            "--timeout",
            "300",
        ],
        cwd=str(REPO_ROOT),
    )
    try:
        _wait_for_port(port)
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


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
