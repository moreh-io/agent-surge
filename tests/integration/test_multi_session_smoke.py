"""Multi-session real-CLI smoke tests against a real vLLM endpoint.

Each provider is exercised at concurrency=4, n-sessions=4. The goal is to
prove (a) the wiring established in batch 2 actually works concurrently
without races, (b) per-session state isolation (XDG_DATA_HOME for
OpenCode, CODEX_HOME for Codex, --bare for Claude) does its job, and
(c) the renderer's per-session artifact tree stays distinct.

Tests are env-gated; see conftest.py for required variables.
"""

from __future__ import annotations

from pathlib import Path  # noqa: F401

import pytest  # noqa: F401

from tests.integration.conftest import _REAL_CLI_SKIP, run_agentsurge  # noqa: F401
