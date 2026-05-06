"""Codex CLI provider and JSONL event parser.

Codex is not yet wired into FrontendSessionRenderer; that requires real
codex stdout fixtures captured against an installed CLI version."""

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

_CODEX_PROVIDER_NAME = "agentsurge_target"

_CODEX_CONFIG_TEMPLATE = """\
model_provider = "{provider}"

[model_providers.{provider}]
name = "AgentSurge target"
base_url = "{base_url}"
env_key = "OPENAI_API_KEY"
wire_api = "responses"
"""


class CodexProvider:
    name = "codex"
    capabilities = FrontendProviderCapabilities(
        name="codex",
        supports_custom_base_url=True,
        supports_openai_compatible_endpoint=True,
        model_name_format="codex-model-name",
        auth_sources=("OPENAI_API_KEY", "CODEX_API_KEY"),
        config_file_strategy="codex-config",
        proxy_supported=False,
        structured_output_format="jsonl",
        partial_text_events="unknown",
        final_usage_events="yes",
        requires_git_repo=True,
        required_env=(),
        known_unsupported_modes=(),
        requires_request_rewrite=True,
    )

    def render(self, session: ReplaySession, artifacts: FrontendRunArtifacts) -> None:
        render_session(session, artifacts)

    def build_env(self, artifacts: FrontendRunArtifacts, config: FrontendConfig) -> dict[str, str]:
        if not config.server_url:
            return {}
        # Codex reads model_provider + base_url from $CODEX_HOME/config.toml
        # (env vars like OPENAI_BASE_URL are ignored). Per-session CODEX_HOME
        # keeps the user's persisted ChatGPT auth from leaking into the run.
        codex_home = artifacts.session_dir / "codex_home"
        codex_home.mkdir(parents=True, exist_ok=True)
        config_path = codex_home / "config.toml"
        # Codex appends "/responses" without a /v1 prefix; the base_url must
        # therefore already end in /v1 for OpenAI-compatible upstreams.
        base = config.server_url.rstrip("/")
        if not base.endswith("/v1"):
            base = base + "/v1"
        config_path.write_text(
            _CODEX_CONFIG_TEMPLATE.format(
                provider=_CODEX_PROVIDER_NAME,
                base_url=base,
            )
        )
        return {"CODEX_HOME": str(codex_home)}

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
            # The renderer creates session_dir as a fresh dir, not a git repo;
            # codex 0.125+ refuses to run in untrusted non-git dirs by default.
            "--skip-git-repo-check",
            "--cd",
            workspace,
        ]
        if config.model is not None:
            cmd.extend(["--model", config.model])
        cmd.extend(
            [
                "--output-last-message",
                final_message,
                # The harness measures CLI throughput, not agentic tool use.
                # Synthetic prompts often look like code; without an explicit
                # no-tools nudge models loop on shell/edit calls until the
                # session_timeout fires. Concrete failure: 2026-04-29 mi250-069
                # 4-concurrent smoke saw 50% of OpenCode/Claude sessions hit
                # 300 s timeouts mid tool-call.
                f"Read {prompt_path} and complete the AgentSurge session "
                f"described there. Respond with text only — do not call any "
                f"tools, do not run shell commands, do not read or edit files.",
            ]
        )
        return cmd


class CodexEventParser:
    provider_name = "codex"

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
        if type_str == "thread.started":
            return [FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_SESSION_STARTED, raw=obj)]
        if type_str == "turn.started":
            return [FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_TURN_STARTED, raw=obj)]
        if type_str == "turn.completed":
            out: list[FrontendEvent] = [
                FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_TURN_COMPLETED, raw=obj)
            ]
            # Codex >=0.125.0 carries usage inline on turn.completed; older/synthetic
            # shape emits a separate {"type":"usage",...} event handled below. Support both.
            usage_raw = obj.get("usage")
            if isinstance(usage_raw, dict):
                usage = {k: int(v) for k, v in usage_raw.items() if isinstance(v, (int, float))}
                out.append(
                    FrontendEvent(
                        ts_monotonic=ts_monotonic,
                        kind=E.EVENT_USAGE_COMPLETED,
                        usage=usage,
                        raw=obj,
                    )
                )
            return out
        if type_str == "turn.failed":
            return [FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_TURN_FAILED, raw=obj)]
        if type_str == "item.completed":
            item = obj.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message":
                text = item.get("text", "")
                text_delta = text if isinstance(text, str) else ""
                return [
                    FrontendEvent(
                        ts_monotonic=ts_monotonic,
                        kind=E.EVENT_ASSISTANT_MESSAGE_COMPLETED,
                        text_delta=text_delta,
                        raw=obj,
                    )
                ]
            return [FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_ITEM_COMPLETED, raw=obj)]
        if type_str == "usage":
            usage_raw = obj.get("usage")
            standalone_usage: dict[str, int] | None = None
            if isinstance(usage_raw, dict):
                standalone_usage = {
                    k: int(v) for k, v in usage_raw.items() if isinstance(v, (int, float))
                }
            return [
                FrontendEvent(
                    ts_monotonic=ts_monotonic,
                    kind=E.EVENT_USAGE_COMPLETED,
                    usage=standalone_usage,
                    raw=obj,
                )
            ]
        if type_str == "error":
            # Codex 0.125.0: obj["message"] is an escaped JSON string with shape
            # {"type":"error","status":N,"error":{"type":"...","message":"..."}}.
            # Pass through verbatim; downstream consumers parse if needed.
            return [FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_ERROR, raw=obj)]
        return [FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_PARSER_UNKNOWN, raw=obj)]

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
