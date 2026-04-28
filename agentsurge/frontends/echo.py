# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import sys

from agentsurge.frontends import events as E
from agentsurge.frontends.base import (
    FrontendConfig,
    FrontendEvent,
    FrontendProviderCapabilities,
    FrontendRunArtifacts,
)
from agentsurge.frontends.render import render_session
from agentsurge.types import ReplaySession

_ECHO_SCRIPT = r"""
import argparse, json, sys, time
p = argparse.ArgumentParser()
p.add_argument("--prompt-path", required=True)
p.add_argument("--mode", default="normal")
a = p.parse_args()
open(a.prompt_path, "r", encoding="utf-8").read()
if a.mode == "slow":
    time.sleep(60)
def emit(o):
    sys.stdout.write(json.dumps(o) + "\n"); sys.stdout.flush()
emit({"type": "session.started"})
emit({"type": "cli.process.started"})
for i in (1, 2, 3):
    emit({"type": "assistant.text.delta", "delta": f"chunk-{i}"})
    time.sleep(0.005)
emit({"type": "assistant.message.completed", "text": "chunk-1chunk-2chunk-3"})
emit({"type": "usage.completed", "usage": {"input_tokens": 10, "output_tokens": 5}})
emit({"type": "session.completed", "exit_code": 0})
sys.exit(0)
"""


_TYPE_TO_KIND = {
    "session.started": E.EVENT_SESSION_STARTED,
    "cli.process.started": E.EVENT_CLI_PROCESS_STARTED,
    "assistant.text.delta": E.EVENT_ASSISTANT_TEXT_DELTA,
    "assistant.message.completed": E.EVENT_ASSISTANT_MESSAGE_COMPLETED,
    "usage.completed": E.EVENT_USAGE_COMPLETED,
    "session.completed": E.EVENT_SESSION_COMPLETED,
}


class EchoProvider:
    name = "echo"
    capabilities = FrontendProviderCapabilities(
        name="echo",
        supports_custom_base_url=False,
        supports_openai_compatible_endpoint=False,
        model_name_format="echo",
        auth_sources=(),
        config_file_strategy="none",
        proxy_supported=False,
        structured_output_format="jsonl",
        partial_text_events="yes",
        final_usage_events="yes",
        requires_git_repo=False,
        required_env=(),
        known_unsupported_modes=(),
    )

    def render(self, session: ReplaySession, artifacts: FrontendRunArtifacts) -> None:
        render_session(session, artifacts)

    def build_command(self, artifacts: FrontendRunArtifacts, config: FrontendConfig) -> list[str]:
        if config.command_template:
            return [
                part.replace("{prompt_path}", str(artifacts.prompt_path))
                for part in config.command_template
            ]
        return [
            sys.executable,
            "-c",
            _ECHO_SCRIPT,
            "--prompt-path",
            str(artifacts.prompt_path),
        ]


class EchoEventParser:
    provider_name = "echo"

    def feed_stdout_line(self, line: str, ts_monotonic: float) -> list[FrontendEvent]:
        stripped = line.strip()
        if not stripped:
            return []
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            return [FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_PARSER_ERROR, raw=line)]
        if not isinstance(obj, dict):
            return [FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_PARSER_UNKNOWN, raw=obj)]
        type_str = obj.get("type")
        kind = _TYPE_TO_KIND.get(type_str) if isinstance(type_str, str) else None
        if kind is None:
            return [FrontendEvent(ts_monotonic=ts_monotonic, kind=E.EVENT_PARSER_UNKNOWN, raw=obj)]
        text_delta = ""
        usage: dict[str, int] | None = None
        if kind == E.EVENT_ASSISTANT_TEXT_DELTA:
            d = obj.get("delta", "")
            text_delta = d if isinstance(d, str) else ""
        elif kind == E.EVENT_ASSISTANT_MESSAGE_COMPLETED:
            t = obj.get("text", "")
            text_delta = t if isinstance(t, str) else ""
        elif kind == E.EVENT_USAGE_COMPLETED:
            u = obj.get("usage")
            if isinstance(u, dict):
                usage = {k: int(v) for k, v in u.items() if isinstance(v, (int, float))}
        return [
            FrontendEvent(
                ts_monotonic=ts_monotonic,
                kind=kind,
                text_delta=text_delta,
                usage=usage,
                raw=obj,
            )
        ]

    def feed_stderr_line(self, line: str, ts_monotonic: float) -> list[FrontendEvent]:
        return []

    def finish(self, exit_code: int, ts_monotonic: float) -> list[FrontendEvent]:
        return []
