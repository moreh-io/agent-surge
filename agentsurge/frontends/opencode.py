"""OpenCode CLI provider and JSON-document event parser.

OpenCode's `--format json` emits a single JSON document on completion,
not a stream. This parser buffers stdout and parses on finish(). The
parser is exercised against synthetic fixtures only; real fixture
capture against an installed OpenCode CLI is required before this
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

# Schema assumption (synthetic — real fixtures pending Phase 1.5):
#   {
#     "session_id": str,
#     "messages": [{"role": "user|assistant", "content": str}, ...],
#     "usage": {"input_tokens": int, "output_tokens": int} | null,
#     "error": str | null
#   }
# The parser maps this document to canonical FrontendEvents on finish().
# When real OpenCode `--format json` output is captured, this schema and
# the dispatch logic below must be re-validated.


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

        cmd: list[str] = ["opencode", "run", "--format", "json"]
        if config.model is not None:
            cmd.extend(["--model", config.model])
        cmd.extend(
            [
                "--file",
                prompt_path,
                "Complete the AgentSurge session described in the attached file.",
            ]
        )
        return cmd


class OpenCodeEventParser:
    provider_name = "opencode"

    def __init__(self) -> None:
        self._buffer: str = ""

    def feed_stdout_line(self, line: str, ts_monotonic: float) -> list[FrontendEvent]:
        # OpenCode emits a single JSON document at completion, not a stream.
        # Accumulate and defer all parsing to finish().
        self._buffer += line
        return []

    def feed_stderr_line(self, line: str, ts_monotonic: float) -> list[FrontendEvent]:
        return []

    def finish(self, exit_code: int, ts_monotonic: float) -> list[FrontendEvent]:
        if not self._buffer:
            return []

        buffer = self._buffer
        self._buffer = ""

        try:
            parsed = json.loads(buffer)
        except json.JSONDecodeError:
            return [FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_PARSER_ERROR, raw=buffer)]

        if not isinstance(parsed, dict):
            return [
                FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_PARSER_UNKNOWN, raw=parsed)
            ]

        events: list[FrontendEvent] = []
        session_id = parsed.get("session_id")
        events.append(
            FrontendEvent(
                ts_monotonic=ts_monotonic,
                kind=E.EVENT_SESSION_STARTED,
                raw={"session_id": session_id},
            )
        )

        messages = parsed.get("messages")
        if isinstance(messages, list):
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                if msg.get("role") != "assistant":
                    continue
                content = msg.get("content", "")
                text_delta = content if isinstance(content, str) else ""
                events.append(
                    FrontendEvent(
                        ts_monotonic=ts_monotonic,
                        kind=E.EVENT_ASSISTANT_MESSAGE_COMPLETED,
                        text_delta=text_delta,
                        raw=msg,
                    )
                )

        error = parsed.get("error")
        if error:
            events.append(
                FrontendEvent(
                    ts_monotonic=ts_monotonic,
                    kind=E.EVENT_ERROR,
                    raw={"error": error},
                )
            )
            return events

        usage_raw = parsed.get("usage")
        if isinstance(usage_raw, dict):
            usage = {k: int(v) for k, v in usage_raw.items() if isinstance(v, (int, float))}
            events.append(
                FrontendEvent(
                    ts_monotonic=ts_monotonic,
                    kind=E.EVENT_USAGE_COMPLETED,
                    usage=usage,
                    raw=usage_raw,
                )
            )

        events.append(
            FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_SESSION_COMPLETED, raw=parsed)
        )
        return events
