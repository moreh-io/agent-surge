# SPDX-License-Identifier: MIT
"""Backward-compatibility shim -- canonical SSE parser moved to backends._sse."""

from agentsurge.backends._sse import (  # noqa: F401
    SSEResult,
    parse_sse,
    stream_sse_ttft,
)
