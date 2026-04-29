"""Claude Code CLI provider and stream-json event parser.

The parser is exercised against synthetic fixtures only. Real fixture
capture against an installed Claude Code CLI is required before this
provider is wired into FrontendSessionRenderer."""

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


class ClaudeProvider:
    name = "claude"
    capabilities = FrontendProviderCapabilities(
        name="claude",
        supports_custom_base_url=False,
        supports_openai_compatible_endpoint=False,
        model_name_format="claude-alias-or-full-name",
        auth_sources=("ANTHROPIC_API_KEY",),
        config_file_strategy="claude-code-config",
        proxy_supported=False,
        structured_output_format="stream-json",
        partial_text_events="yes",
        final_usage_events="yes",
        requires_git_repo=False,
        required_env=(),
        known_unsupported_modes=(),
    )

    def render(self, session: ReplaySession, artifacts: FrontendRunArtifacts) -> None:
        render_session(session, artifacts)

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

        cmd: list[str] = [
            "claude",
            "-p",
            "--output-format",
            "stream-json",
            "--include-partial-messages",
        ]
        if config.model is not None:
            cmd.extend(["--model", config.model])
        cmd.append(f"Read {prompt_path} and complete the AgentSurge session described there.")
        return cmd


class ClaudeEventParser:
    provider_name = "claude"

    def __init__(self) -> None:
        self._buffer: str = ""

    def feed_stdout_line(self, line: str, ts_monotonic: float) -> list[FrontendEvent]:
        self._buffer += line
        events: list[FrontendEvent] = []
        while "\n" in self._buffer:
            segment, self._buffer = self._buffer.split("\n", 1)
            segment = segment.rstrip("\r\n")
            if not segment or not segment.strip():
                continue
            try:
                obj = json.loads(segment)
            except json.JSONDecodeError:
                events.append(
                    FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_PARSER_ERROR, raw=segment)
                )
                continue
            if not isinstance(obj, dict):
                events.append(
                    FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_PARSER_UNKNOWN, raw=obj)
                )
                continue
            events.extend(self._dispatch(obj, ts_monotonic))
        return events

    def _dispatch(self, obj: dict[str, object], ts_monotonic: float) -> list[FrontendEvent]:
        type_str = obj.get("type")
        if type_str == "system":
            # Only system.subtype == "init" is the canonical session-init event.
            # Other subtypes (hook_started, hook_response, status, ...) are
            # session-management noise and route to PARSER_UNKNOWN.
            subtype = obj.get("subtype")
            if subtype == "init":
                return [
                    FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_SESSION_STARTED, raw=obj)
                ]
            return [FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_PARSER_UNKNOWN, raw=obj)]
        if type_str == "assistant":
            # Error-path assistant events carry a top-level "error" field; the
            # subsequent result event is the source-of-truth for failures, so
            # treat the assistant event as parser-noise here.
            if obj.get("error"):
                return [
                    FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_PARSER_UNKNOWN, raw=obj)
                ]
            text = self._extract_assistant_text(obj)
            return [
                FrontendEvent(
                    ts_monotonic=ts_monotonic,
                    kind=E.EVENT_ASSISTANT_MESSAGE_COMPLETED,
                    text_delta=text,
                    raw=obj,
                )
            ]
        if type_str == "user":
            return [FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_USER_MESSAGE, raw=obj)]
        if type_str == "stream_event":
            event = obj.get("event")
            if isinstance(event, dict):
                inner_type = event.get("type")
                if inner_type == "content_block_delta":
                    delta = event.get("delta")
                    if isinstance(delta, dict) and delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        text_delta = text if isinstance(text, str) else ""
                        return [
                            FrontendEvent(
                                ts_monotonic=ts_monotonic,
                                kind=E.EVENT_ASSISTANT_TEXT_DELTA,
                                text_delta=text_delta,
                                raw=obj,
                            )
                        ]
                if inner_type == "message_start":
                    return [
                        FrontendEvent(
                            ts_monotonic=ts_monotonic, kind=E.EVENT_MESSAGE_START, raw=obj
                        )
                    ]
            return [FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_PARSER_UNKNOWN, raw=obj)]
        if type_str == "rate_limit_event":
            return [FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_PARSER_UNKNOWN, raw=obj)]
        if type_str == "result":
            # Branch: error results surface as EVENT_ERROR; success results
            # produce usage + session-completed terminal events. Note that
            # real CLI emits subtype="success" even when is_error=True, so we
            # must branch on is_error rather than subtype.
            if obj.get("is_error") is True:
                return [FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_ERROR, raw=obj)]
            usage_raw = obj.get("usage")
            usage: dict[str, int] | None = None
            if isinstance(usage_raw, dict):
                usage = {k: int(v) for k, v in usage_raw.items() if isinstance(v, (int, float))}
            return [
                FrontendEvent(
                    ts_monotonic=ts_monotonic,
                    kind=E.EVENT_USAGE_COMPLETED,
                    usage=usage,
                    raw=obj,
                ),
                FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_SESSION_COMPLETED, raw=obj),
            ]
        return [FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_PARSER_UNKNOWN, raw=obj)]

    def _extract_assistant_text(self, obj: dict[str, object]) -> str:
        message = obj.get("message")
        if not isinstance(message, dict):
            return ""
        content = message.get("content")
        if not isinstance(content, list):
            return ""
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    def feed_stderr_line(self, line: str, ts_monotonic: float) -> list[FrontendEvent]:
        return []

    def finish(self, exit_code: int, ts_monotonic: float) -> list[FrontendEvent]:
        if self._buffer:
            leftover = self._buffer
            self._buffer = ""
            return [
                FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_PARSER_ERROR, raw=leftover)
            ]
        return []
