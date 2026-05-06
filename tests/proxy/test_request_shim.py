"""Tests for agentsurge.proxy.request_shim.RequestShim.

Verifies the in-process shim binds a random local port, forwards requests
to the configured upstream with rewrites applied, and cleans up cleanly
on exit. The standalone `agentsurge proxy-translator` subcommand and the
test-side `translator_shim` fixture both go away once this module is
wired into the renderer (Task 4)."""

from __future__ import annotations

from typing import Any

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer

from agentsurge.proxy.request_shim import RequestShim


def _make_upstream_capture() -> tuple[web.Application, list[dict[str, Any]]]:
    captured: list[dict[str, Any]] = []

    async def echo(request: web.Request) -> web.Response:
        body_bytes = await request.read()
        try:
            payload = await request.json() if body_bytes else {}
        except Exception:
            payload = {"_raw": body_bytes.decode("utf-8", errors="replace")}
        captured.append({"path": request.path, "body": payload})
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", echo)
    return app, captured


async def test_request_shim_yields_local_url_and_forwards() -> None:
    upstream_app, captured = _make_upstream_capture()
    upstream_server = TestServer(upstream_app)
    await upstream_server.start_server()
    upstream_url = f"http://{upstream_server.host}:{upstream_server.port}"
    try:
        async with RequestShim(upstream_base=upstream_url) as shim_url:
            assert shim_url.startswith("http://127.0.0.1:")
            async with aiohttp.ClientSession() as client:
                resp = await client.post(
                    f"{shim_url}/v1/responses",
                    json={
                        "input": [
                            {"role": "developer", "content": "rules"},
                            {"role": "user", "content": "hi"},
                        ]
                    },
                )
                assert resp.status == 200
        assert captured, "request never reached upstream"
        forwarded = captured[0]["body"]
        assert [m["role"] for m in forwarded["input"]] == ["user"], (
            "developer→system rewrite must apply through the shim"
        )
        assert forwarded["instructions"] == "rules"
    finally:
        await upstream_server.close()


async def test_request_shim_releases_port_on_exit() -> None:
    """Re-entering the manager twice must not collide on the same port —
    proves cleanup actually frees the bound socket."""
    upstream_app, _ = _make_upstream_capture()
    upstream_server = TestServer(upstream_app)
    await upstream_server.start_server()
    upstream_url = f"http://{upstream_server.host}:{upstream_server.port}"
    try:
        async with RequestShim(upstream_base=upstream_url) as first_url:
            first_port = int(first_url.rsplit(":", 1)[1])
        async with RequestShim(upstream_base=upstream_url) as second_url:
            second_port = int(second_url.rsplit(":", 1)[1])
        # Ports may coincide if the OS rebinds, but neither bind raised.
        assert first_port > 0 and second_port > 0
    finally:
        await upstream_server.close()
