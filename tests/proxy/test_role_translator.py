"""Tests for agentsurge.proxy.role_translator.

The translator is a passthrough reverse proxy that only modifies the
JSON body of POST `/v1/responses` requests, rewriting `developer` role
entries inside the `input` array to `system`. Everything else (other
paths, other methods, malformed bodies, response streams) flows through
verbatim.

Uses aiohttp.test_utils.TestServer directly to avoid adding pytest-aiohttp
as a dev dependency. Each test spins up a mock upstream and the proxy in
the same event loop, then drives the proxy with an aiohttp.ClientSession.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer

from agentsurge.proxy.role_translator import make_app


def _make_upstream_capture() -> tuple[web.Application, list[dict[str, Any]]]:
    """Mock upstream that captures every request body it receives."""
    captured: list[dict[str, Any]] = []

    async def echo_handler(request: web.Request) -> web.Response:
        body = await request.read()
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            payload = {"_raw": body.decode("utf-8", errors="replace")}
        captured.append(
            {
                "method": request.method,
                "path": request.path,
                "headers": dict(request.headers),
                "body": payload,
            }
        )
        return web.json_response({"ok": True, "echoed": payload})

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", echo_handler)
    return app, captured


@asynccontextmanager
async def _proxy_under_test():
    """Yield (proxy_base_url, captured_list).

    Spins up a mock upstream and the proxy in the same loop, ensures both
    are torn down regardless of test outcome.
    """
    upstream_app, captured = _make_upstream_capture()
    upstream_server = TestServer(upstream_app)
    await upstream_server.start_server()
    upstream_url = f"http://{upstream_server.host}:{upstream_server.port}"

    proxy_app = make_app(upstream_base=upstream_url, timeout_s=10.0)
    proxy_server = TestServer(proxy_app)
    await proxy_server.start_server()
    proxy_url = f"http://{proxy_server.host}:{proxy_server.port}"

    try:
        yield proxy_url, captured
    finally:
        await proxy_server.close()
        await upstream_server.close()


async def test_responses_developer_role_rewritten_to_system():
    async with _proxy_under_test() as (proxy_url, captured):
        body = {
            "model": "qwen3.6-27b",
            "input": [
                {"role": "developer", "content": "be brief"},
                {"role": "user", "content": "hi"},
            ],
            "max_output_tokens": 16,
        }
        async with aiohttp.ClientSession() as client:
            resp = await client.post(f"{proxy_url}/v1/responses", json=body)
            assert resp.status == 200
        assert len(captured) == 1
        forwarded = captured[0]["body"]
        roles = [m["role"] for m in forwarded["input"]]
        assert roles == ["system", "user"], f"expected developer→system rewrite, got {roles}"
