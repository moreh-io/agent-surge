"""OpenCode CLI provider and JSONL streaming event parser.

OpenCode 1.14.29 `--format json` actually emits JSONL (one event per
line), not a single JSON document as the synthetic fixtures originally
assumed. Real-fixture-validated event types: step_start, text,
tool_use, step_finish, error. The parser maps them to canonical
FrontendEvents in a streaming line-by-line model that mirrors the
Codex parser. See tests/core/fixtures/opencode/real_v1.14.29_meta.json
for the captured ground truth."""

# SPDX-License-Identifier: MIT
from __future__ import annotations

import json

from agentsurge.frontends import events as E
from agentsurge.frontends.base import (
    FrontendConfig,
    FrontendEvent,
    FrontendProviderCapabilities,
    FrontendRunArtifacts,
)
from agentsurge.frontends.render import render_session
from agentsurge.types import ReplaySession

# Real OpenCode 1.14.29 schema (see tests/core/fixtures/opencode/*.jsonl):
#   {"type": "step_start"|"text"|"tool_use"|"step_finish"|"error",
#    "timestamp": int, "sessionID": str, "part": {...}}
# - step_start          -> EVENT_TURN_STARTED
# - text                -> EVENT_ASSISTANT_TEXT_DELTA  (part.text)
# - tool_use            -> EVENT_TOOL_USE_OBSERVED     (signal: RequestShim strips
#                          `tools` from /v1/chat/completions so this type must
#                          never appear in a benchmark run; its presence means the
#                          shim was not in the request path or the strip logic
#                          regressed — the run is contaminated)
# - step_finish         -> EVENT_TURN_COMPLETED;
#                          if part.reason == "stop", also EVENT_USAGE_COMPLETED
#                          (normalized from part.tokens) and EVENT_SESSION_COMPLETED
# - error               -> EVENT_ERROR
# EVENT_SESSION_STARTED is emitted lazily on the first event seen.


class OpenCodeProvider:
    name = "opencode"
    capabilities = FrontendProviderCapabilities(
        name="opencode",
        supports_custom_base_url=True,
        supports_openai_compatible_endpoint=True,
        model_name_format="provider/model",
        auth_sources=("OPENCODE_API_KEY",),
        config_file_strategy="opencode-config",
        proxy_supported=False,
        structured_output_format="json",
        partial_text_events="unknown",
        final_usage_events="yes",
        requires_git_repo=False,
        required_env=(),
        known_unsupported_modes=(),
        requires_request_rewrite=True,
    )

    def render(self, session: ReplaySession, artifacts: FrontendRunArtifacts) -> None:
        render_session(session, artifacts)

    def build_env(self, artifacts: FrontendRunArtifacts, config: FrontendConfig) -> dict[str, str]:
        # XDG_DATA_HOME isolation: opencode's SQLite session DB lives under
        # $XDG_DATA_HOME/opencode/opencode.db. Concurrent sessions sharing
        # the default $HOME corrupt the WAL files (documented in design doc
        # §Real Fixture Notes; reproduced during 2026-04-29 E2E debug). A
        # per-session XDG_DATA_HOME gives every session its own DB.
        xdg_home = artifacts.session_dir / "xdg"
        xdg_home.mkdir(parents=True, exist_ok=True)
        env: dict[str, str] = {"XDG_DATA_HOME": str(xdg_home)}

        if not config.server_url:
            return env

        # OpenCode 1.14.29 doesn't honor an OPENAI_BASE_URL env override; the
        # custom endpoint must be declared via opencode.json in cwd, which
        # the renderer sets to artifacts.session_dir. The provider key is
        # taken from the slash prefix in --frontend-model (e.g. "vllm" in
        # "vllm/qwen3.6-27b").
        model = config.model or ""
        if "/" not in model:
            return env
        provider_key, model_name = model.split("/", 1)
        api_key = config.extra_env.get("OPENAI_API_KEY", "")
        opencode_json = artifacts.session_dir / "opencode.json"
        opencode_json.write_text(
            json.dumps(
                {
                    "$schema": "https://opencode.ai/config.json",
                    "provider": {
                        provider_key: {
                            "npm": "@ai-sdk/openai-compatible",
                            "name": "AgentSurge target",
                            "options": {
                                "baseURL": config.server_url.rstrip("/") + "/v1",
                                "apiKey": api_key,
                            },
                            "models": {model_name: {}},
                        }
                    },
                },
                indent=2,
            )
        )
        return env

    def build_command(self, artifacts: FrontendRunArtifacts, config: FrontendConfig) -> list[str]:
        workspace = str(artifacts.session_dir)
        prompt_path = str(artifacts.prompt_path)
        frontend_model = config.model or ""

        if config.command_template:
            return [
                part.replace("{prompt_path}", prompt_path)
                .replace("{workspace}", workspace)
                .replace("{model}", frontend_model)
                for part in config.command_template
            ]

        # opencode 1.14.x parses `[message..]` greedily — placing the message
        # AFTER `--file` consumes it as a second filename. Order must be:
        # `opencode run "<msg>" --format json --model X --file=<path>`. The
        # `--file=<path>` form (no space) further constrains the yargs array
        # to one value.
        cmd: list[str] = [
            "opencode",
            "run",
            # Strong no-tools nudge: the harness measures CLI throughput, not
            # agentic tool use. Without this opencode loops on bash/edit/grep
            # for synthetic-code prompts until session_timeout fires (50% of
            # 4-concurrent smoke sessions hit 300 s on 2026-04-29 mi250-069).
            "Complete the AgentSurge session described in the attached "
            "file. Respond with text only — do not call any tools, do not "
            "run shell commands, do not read or edit files.",
            "--format",
            "json",
        ]
        if config.model is not None:
            cmd.extend(["--model", config.model])
        cmd.append(f"--file={prompt_path}")
        return cmd


class OpenCodeEventParser:
    provider_name = "opencode"

    def __init__(self) -> None:
        self._buffer: str = ""
        self._session_started_emitted: bool = False
        self._session_id: str | None = None

    def feed_stdout_line(self, line: str, ts_monotonic: float) -> list[FrontendEvent]:
        events: list[FrontendEvent] = []
        self._buffer += line
        while "\n" in self._buffer:
            segment, self._buffer = self._buffer.split("\n", 1)
            segment = segment.rstrip("\r")
            if not segment.strip():
                continue
            try:
                parsed = json.loads(segment)
            except json.JSONDecodeError:
                events.append(
                    FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_PARSER_ERROR, raw=segment)
                )
                continue
            if not isinstance(parsed, dict):
                events.append(
                    FrontendEvent(
                        ts_monotonic=ts_monotonic, kind=E.EVENT_PARSER_UNKNOWN, raw=parsed
                    )
                )
                continue
            events.extend(self._dispatch(parsed, ts_monotonic))
        return events

    def feed_stderr_line(self, line: str, ts_monotonic: float) -> list[FrontendEvent]:
        return []

    def finish(self, exit_code: int, ts_monotonic: float) -> list[FrontendEvent]:
        events: list[FrontendEvent] = []
        if self._buffer.strip():
            events.append(
                FrontendEvent(
                    ts_monotonic=ts_monotonic, kind=E.EVENT_PARSER_ERROR, raw=self._buffer
                )
            )
        self._buffer = ""
        return events

    def _dispatch(self, parsed: dict, ts_monotonic: float) -> list[FrontendEvent]:
        events: list[FrontendEvent] = []
        if not self._session_started_emitted:
            self._session_id = parsed.get("sessionID")
            events.append(
                FrontendEvent(
                    ts_monotonic=ts_monotonic,
                    kind=E.EVENT_SESSION_STARTED,
                    raw={"session_id": self._session_id},
                )
            )
            self._session_started_emitted = True

        kind = parsed.get("type")
        if kind == "step_start":
            events.append(
                FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_TURN_STARTED, raw=parsed)
            )
        elif kind == "text":
            part = parsed.get("part", {}) or {}
            text = part.get("text", "") if isinstance(part, dict) else ""
            events.append(
                FrontendEvent(
                    ts_monotonic=ts_monotonic,
                    kind=E.EVENT_ASSISTANT_TEXT_DELTA,
                    text_delta=text,
                    raw=parsed,
                )
            )
        elif kind == "tool_use":
            events.append(
                FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_TOOL_USE_OBSERVED, raw=parsed)
            )
        elif kind == "step_finish":
            events.append(
                FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_TURN_COMPLETED, raw=parsed)
            )
            part = parsed.get("part", {}) or {}
            reason = part.get("reason") if isinstance(part, dict) else None
            if reason == "stop":
                tokens = part.get("tokens", {}) if isinstance(part, dict) else {}
                if not isinstance(tokens, dict):
                    tokens = {}
                cache = tokens.get("cache", {}) or {}
                if not isinstance(cache, dict):
                    cache = {}
                normalized_usage: dict = {
                    "input_tokens": tokens.get("input", 0),
                    "output_tokens": tokens.get("output", 0),
                }
                if "reasoning" in tokens:
                    normalized_usage["reasoning"] = tokens["reasoning"]
                if "read" in cache or "write" in cache:
                    normalized_usage["cache_read"] = cache.get("read", 0)
                    normalized_usage["cache_write"] = cache.get("write", 0)
                events.append(
                    FrontendEvent(
                        ts_monotonic=ts_monotonic,
                        kind=E.EVENT_USAGE_COMPLETED,
                        usage=normalized_usage,
                        raw=parsed,
                    )
                )
                events.append(
                    FrontendEvent(
                        ts_monotonic=ts_monotonic, kind=E.EVENT_SESSION_COMPLETED, raw=parsed
                    )
                )
        elif kind == "error":
            events.append(FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_ERROR, raw=parsed))
        else:
            # All real-fixture types (step_start, text, tool_use, step_finish, error)
            # dispatch to a named kind above; no additional types route here from
            # real_simple_session_v1.14.29.jsonl. Re-audit if a CLI version bump
            # introduces a new type.
            events.append(
                FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_PARSER_UNKNOWN, raw=parsed)
            )
        return events
