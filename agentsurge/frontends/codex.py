"""Codex CLI provider and JSONL event parser.

The parser is exercised against synthetic fixtures only. Phase 1.5 (real
codex exec stdout capture) lands separately before this is wired into
FrontendSessionRenderer."""

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


class CodexProvider:
    name = "codex"
    capabilities = FrontendProviderCapabilities(
        name="codex",
        supports_custom_base_url=False,
        supports_openai_compatible_endpoint=False,
        model_name_format="codex-model-name",
        auth_sources=("CODEX_API_KEY", "codex-saved-auth"),
        config_file_strategy="codex-config",
        proxy_supported=False,
        structured_output_format="jsonl",
        partial_text_events="unknown",
        final_usage_events="yes",
        requires_git_repo=True,
        required_env=(),
        known_unsupported_modes=(),
    )

    def render(self, session: ReplaySession, artifacts: FrontendRunArtifacts) -> None:
        render_session(session, artifacts)

    def build_command(self, artifacts: FrontendRunArtifacts, config: FrontendConfig) -> list[str]:
        workspace = str(artifacts.session_dir)
        prompt_path = str(artifacts.prompt_path)
        final_message = str(artifacts.session_dir / "final.txt")
        frontend_model = config.model or ""

        if config.command_template:
            return [
                part.replace("{prompt_path}", prompt_path)
                .replace("{workspace}", workspace)
                .replace("{frontend_model}", frontend_model)
                .replace("{final_message}", final_message)
                for part in config.command_template
            ]

        cmd: list[str] = [
            "codex",
            "exec",
            "--json",
            "--ephemeral",
            "--cd",
            workspace,
        ]
        if config.model is not None:
            cmd.extend(["--model", config.model])
        cmd.extend(
            [
                "--output-last-message",
                final_message,
                f"Read {prompt_path} and complete the AgentSurge session described there.",
            ]
        )
        return cmd


class CodexEventParser:
    provider_name = "codex"

    def __init__(self) -> None:
        self._buffer: str = ""

    def feed_stdout_line(self, line: str, ts_monotonic: float) -> list[FrontendEvent]:
        if line.endswith("\n"):
            chunk = line[:-1]
        else:
            chunk = line
        self._buffer += chunk
        if not self._buffer:
            return []
        try:
            obj = json.loads(self._buffer)
        except json.JSONDecodeError:
            return []
        self._buffer = ""
        if not isinstance(obj, dict):
            return [FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_PARSER_UNKNOWN, raw=obj)]
        return [self._dispatch(obj, ts_monotonic)]

    def _dispatch(self, obj: dict[str, object], ts_monotonic: float) -> FrontendEvent:
        type_str = obj.get("type")
        if type_str == "thread.started":
            return FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_SESSION_STARTED, raw=obj)
        if type_str == "turn.started":
            return FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_TURN_STARTED, raw=obj)
        if type_str == "turn.completed":
            return FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_TURN_COMPLETED, raw=obj)
        if type_str == "item.completed":
            item = obj.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message":
                text = item.get("text", "")
                text_delta = text if isinstance(text, str) else ""
                return FrontendEvent(
                    ts_monotonic=ts_monotonic,
                    kind=E.EVENT_ASSISTANT_MESSAGE_COMPLETED,
                    text_delta=text_delta,
                    raw=obj,
                )
            return FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_ITEM_COMPLETED, raw=obj)
        if type_str == "usage":
            usage_raw = obj.get("usage")
            usage: dict[str, int] | None = None
            if isinstance(usage_raw, dict):
                usage = {k: int(v) for k, v in usage_raw.items() if isinstance(v, (int, float))}
            return FrontendEvent(
                ts_monotonic=ts_monotonic,
                kind=E.EVENT_USAGE_COMPLETED,
                usage=usage,
                raw=obj,
            )
        if type_str == "error":
            return FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_ERROR, raw=obj)
        return FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_PARSER_UNKNOWN, raw=obj)

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
