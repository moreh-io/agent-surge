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
        supports_custom_base_url=True,
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
        requires_request_rewrite=False,
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

        cmd: list[str] = ["claude"]
        # --bare suppresses OAuth/keychain reads so ANTHROPIC_API_KEY env is
        # the sole auth source; without it persisted Claude.ai login can win
        # over the configured server-url target.
        if config.server_url:
            cmd.append("--bare")
        # `--verbose` is required by Claude Code when combining --print with
        # --output-format=stream-json (CLI rejects the combo otherwise).
        cmd.extend(
            [
                "--verbose",
                "-p",
                "--output-format",
                "stream-json",
                "--include-partial-messages",
                # Hard-disable every built-in tool. The harness measures CLI
                # throughput, not agentic tool execution; without this Claude
                # invokes Bash on synthetic-code prompts and loops until the
                # session_timeout (50% of multi-session smoke runs hit 300 s
                # on 2026-04-29 mi250-069). Empty string = "no tools".
                "--allowedTools",
                "",
            ]
        )
        if config.model is not None:
            cmd.extend(["--model", config.model])
        # `--` stops yargs from interpreting subsequent args as flag values.
        # Without it, ``--allowedTools <tools...>`` is variadic and greedily
        # consumes the prompt positional that follows when no later flag
        # (e.g. ``--model``) intervenes — Claude then errors with
        # "Input must be provided either through stdin or as a prompt
        # argument when using --print" (single-session smoke regression
        # caught this on 2026-04-29 when config.model was None).
        cmd.append("--")
        cmd.append(
            f"Read {prompt_path} and complete the AgentSurge session "
            f"described there. Respond with text only — do not call any "
            f"tools, do not run shell commands, do not read or edit files."
        )
        return cmd

    def build_env(self, artifacts: FrontendRunArtifacts, config: FrontendConfig) -> dict[str, str]:
        if not config.server_url:
            return {}
        # Claude Code reads ANTHROPIC_BASE_URL directly; no config file needed.
        # User must supply ANTHROPIC_API_KEY via --frontend-extra-env or shell
        # env. We don't synthesize a key because vLLM-style targets ignore it
        # and real Anthropic-compatible gateways have user-specific auth.
        #
        # Strip a trailing `/v1` because Claude Code always appends
        # `/v1/messages` itself. Users typically pass the same OpenAI-style
        # base URL (`.../v1`) used by Codex/OpenCode, so without this strip
        # requests land at `/v1/v1/messages` and the upstream 404s.
        base_url = config.server_url.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[: -len("/v1")]
        env = {"ANTHROPIC_BASE_URL": base_url}
        # Mirror the user's ANTHROPIC_API_KEY into ANTHROPIC_AUTH_TOKEN: Claude
        # Code 2.1.x sends ANTHROPIC_API_KEY as the x-api-key header, but
        # OpenAI-compatible targets (vLLM `/v1/messages`) only accept
        # `Authorization: Bearer ...`, which Claude populates from
        # ANTHROPIC_AUTH_TOKEN. Without this mirror every request 401's.
        # User-set ANTHROPIC_AUTH_TOKEN (via --frontend-extra-env) still wins
        # because extra_env merges last in the renderer.
        api_key = config.extra_env.get("ANTHROPIC_API_KEY") or config.extra_env.get(
            "ANTHROPIC_AUTH_TOKEN"
        )
        if api_key:
            env["ANTHROPIC_AUTH_TOKEN"] = api_key
        return env


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
            #
            # Real-fixture PARSER_UNKNOWN inventory (real_simple_session_v2.1.122.jsonl):
            #   system/hook_started   — noise: Claude Code plugin/hook lifecycle, no bench signal.
            #   system/hook_response  — noise: plugin hook reply, no bench signal.
            #   system/status         — noise: CLI status advisory (e.g. "Thinking…"), no bench signal.
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
            # Real-fixture PARSER_UNKNOWN inventory for stream_event subtypes:
            #   stream_event/content_block_start — noise: Anthropic streaming protocol bracket
            #                                      opening each text block; no bench signal.
            #   stream_event/content_block_stop  — noise: Anthropic streaming protocol bracket
            #                                      closing each text block; no bench signal.
            #   stream_event/message_delta       — noise: carries stop_reason, but that is
            #                                      redundant with result → EVENT_USAGE_COMPLETED
            #                                      + EVENT_SESSION_COMPLETED already captured.
            #   stream_event/message_stop        — noise: streaming protocol terminator;
            #                                      no bench signal.
            return [FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_PARSER_UNKNOWN, raw=obj)]
        if type_str == "rate_limit_event":
            # noise: Anthropic throttling advisory; useful for ops monitoring
            # but not for throughput benchmarking.
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
