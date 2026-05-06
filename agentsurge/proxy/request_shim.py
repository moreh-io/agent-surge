# SPDX-License-Identifier: MIT
"""In-process replacement for the standalone `agentsurge proxy-translator`.

Spawns the role-rewriting aiohttp app from ``role_translator.make_app`` on
a random local port inside the harness's own event loop, then yields the
URL the harness should hand to the CLI as ``--frontend-server-url``.
``__aexit__`` shuts the runner down cleanly so the bound port is released
before the next session starts.

Why in-process: the previous design ran the proxy as a subprocess
(``agentsurge proxy-translator``) so the test fixture had to spawn,
poll, and kill it for every multi-session run. Folding the shim into
the harness loop removes one moving part, removes the subprocess
fixture's poll-for-port handshake, and removes the SIGTERM-on-teardown
race that occasionally leaked the upstream connection.
"""

from __future__ import annotations

import socket
from types import TracebackType

from aiohttp import web

from agentsurge.proxy.role_translator import make_app


def _allocate_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        return int(port)


class RequestShim:
    """Async context manager hosting the role-rewrite proxy in-process.

    Enter yields ``http://127.0.0.1:<port>``. Exit gracefully shuts the
    aiohttp runner down. Single-use — re-entering after exit raises
    RuntimeError because the underlying AppRunner is consumed."""

    def __init__(self, *, upstream_base: str, timeout_s: float = 600.0) -> None:
        self._upstream_base = upstream_base
        self._timeout_s = timeout_s
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None

    async def __aenter__(self) -> str:
        if self._runner is not None:
            raise RuntimeError("RequestShim is single-use; allocate a fresh instance")
        app = make_app(upstream_base=self._upstream_base, timeout_s=self._timeout_s)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        port = _allocate_port()
        self._site = web.TCPSite(self._runner, host="127.0.0.1", port=port)
        await self._site.start()
        return f"http://127.0.0.1:{port}"

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._site is not None:
            await self._site.stop()
            self._site = None
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
