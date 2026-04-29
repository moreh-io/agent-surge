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


async def test_responses_developer_message_merged_into_instructions():
    """A bare ``developer`` message in input[] is moved into ``instructions``
    so vLLM's Responses-adapter sees exactly one system source. With no
    pre-existing instructions, the developer content becomes the whole
    instructions string."""
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
        assert roles == ["user"], f"developer should be removed from input; got {roles}"
        assert forwarded.get("instructions") == "be brief"


async def test_responses_developer_appended_to_existing_instructions():
    """Codex sends a Responses request with both ``instructions`` (its
    agent prompt) and a ``developer`` message in input. The translator must
    concatenate the developer content onto the existing instructions
    (separated by a blank line) so neither system source is dropped."""
    async with _proxy_under_test() as (proxy_url, captured):
        body = {
            "model": "qwen3.6-27b",
            "instructions": "You are a coding agent.",
            "input": [
                {"role": "developer", "content": "be concise"},
                {"role": "user", "content": "hi"},
            ],
        }
        async with aiohttp.ClientSession() as client:
            resp = await client.post(f"{proxy_url}/v1/responses", json=body)
            assert resp.status == 200
        forwarded = captured[0]["body"]
        assert forwarded["instructions"] == "You are a coding agent.\n\nbe concise"
        assert [m["role"] for m in forwarded["input"]] == ["user"]


async def test_responses_multiple_system_messages_all_merged():
    """Every developer- and system-role message in input[] gets pulled out;
    only user/assistant entries remain. Concrete failure: 2026-04-29
    mi250-069 E2E saw codex sending [developer, user, user] which after
    role-rewrite alone still left system messages in input alongside the
    instructions string, triggering a duplicate-system 400."""
    async with _proxy_under_test() as (proxy_url, captured):
        body = {
            "model": "qwen3.6-27b",
            "instructions": "base",
            "input": [
                {"role": "user", "content": "ctx"},
                {"role": "developer", "content": "rule a"},
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "rule b"},
                {"role": "assistant", "content": "ack"},
            ],
        }
        async with aiohttp.ClientSession() as client:
            resp = await client.post(f"{proxy_url}/v1/responses", json=body)
            assert resp.status == 200
        forwarded = captured[0]["body"]
        assert [m["role"] for m in forwarded["input"]] == ["user", "user", "assistant"]
        # Order: existing instructions, then developer, then system, in input order.
        assert forwarded["instructions"] == "base\n\nrule a\n\nrule b"


async def test_responses_typed_content_parts_extracted():
    """Real Codex sends content as a list of typed parts
    ([{"type": "input_text", "text": "..."}], etc.). The translator must
    flatten those into the merged instructions string just like plain
    string content."""
    async with _proxy_under_test() as (proxy_url, captured):
        body = {
            "model": "qwen3.6-27b",
            "input": [
                {
                    "role": "developer",
                    "content": [
                        {"type": "input_text", "text": "first chunk"},
                        {"type": "text", "text": "second chunk"},
                    ],
                },
                {"role": "user", "content": "hi"},
            ],
        }
        async with aiohttp.ClientSession() as client:
            resp = await client.post(f"{proxy_url}/v1/responses", json=body)
            assert resp.status == 200
        forwarded = captured[0]["body"]
        assert forwarded["instructions"] == "first chunk\n\nsecond chunk"
        assert [m["role"] for m in forwarded["input"]] == ["user"]


async def test_responses_tool_calls_force_disabled():
    """Codex declares 11 tools and tool_choice=auto. Even with the prompt
    nudging "no tools", qwen3.6 still invokes exec_command on coding
    prompts; the second-turn body that codex sends back then contains
    function_call/function_call_output items that vLLM's Responses
    adapter rejects with 215 pydantic errors. The translator forces
    tool_choice="none" and tools=[] so the loop never starts."""
    async with _proxy_under_test() as (proxy_url, captured):
        body = {
            "model": "qwen3.6-27b",
            "input": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "name": "exec_command"}],
            "tool_choice": "auto",
        }
        async with aiohttp.ClientSession() as client:
            resp = await client.post(f"{proxy_url}/v1/responses", json=body)
            assert resp.status == 200
        forwarded = captured[0]["body"]
        assert forwarded["tool_choice"] == "none"
        assert forwarded["tools"] == []


async def test_responses_no_developer_role_unchanged():
    """A Responses request without any developer role must pass through
    byte-equivalently — no spurious mutations."""
    async with _proxy_under_test() as (proxy_url, captured):
        body = {
            "model": "qwen3.6-27b",
            "input": [{"role": "user", "content": "hi"}],
        }
        async with aiohttp.ClientSession() as client:
            resp = await client.post(f"{proxy_url}/v1/responses", json=body)
            assert resp.status == 200
        forwarded_roles = [m["role"] for m in captured[0]["body"]["input"]]
        assert forwarded_roles == ["user"]


async def test_chat_completions_passes_through_unchanged():
    """Only POST /v1/responses gets rewritten; other endpoints are
    untouched even if they happen to contain developer roles."""
    async with _proxy_under_test() as (proxy_url, captured):
        body = {
            "model": "qwen3.6-27b",
            "messages": [{"role": "developer", "content": "x"}],
        }
        async with aiohttp.ClientSession() as client:
            resp = await client.post(f"{proxy_url}/v1/chat/completions", json=body)
            assert resp.status == 200
        msgs = captured[0]["body"]["messages"]
        assert msgs[0]["role"] == "developer"


async def test_get_passes_through():
    async with _proxy_under_test() as (proxy_url, captured):
        async with aiohttp.ClientSession() as client:
            resp = await client.get(f"{proxy_url}/v1/models")
            assert resp.status == 200
        assert captured[0]["method"] == "GET"
        assert captured[0]["path"] == "/v1/models"


async def test_malformed_json_body_passes_through():
    """Garbage in the body should not crash the proxy — upstream surfaces
    the schema error itself."""
    async with _proxy_under_test() as (proxy_url, captured):
        async with aiohttp.ClientSession() as client:
            resp = await client.post(
                f"{proxy_url}/v1/responses",
                data=b"this is not json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 200
        forwarded = captured[0]["body"]
        assert forwarded.get("_raw") == "this is not json"


async def test_authorization_header_forwarded():
    """Auth headers must reach the upstream so the API key/token gets
    delivered. Without this the proxy would force every request to be
    keyless."""
    async with _proxy_under_test() as (proxy_url, captured):
        async with aiohttp.ClientSession() as client:
            resp = await client.post(
                f"{proxy_url}/v1/responses",
                json={"input": [{"role": "user", "content": "x"}]},
                headers={"Authorization": "Bearer test-token-42"},
            )
            assert resp.status == 200
        assert captured[0]["headers"].get("Authorization") == "Bearer test-token-42"
