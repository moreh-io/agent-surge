# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from agentsurge.types import ReplaySession

if TYPE_CHECKING:
    pass


@dataclass
class FrontendConfig:
    name: str
    command_template: list[str] | None
    workspace_dir: Path
    prompt_mode: Literal["auto", "file", "stdin", "arg"]
    output_format: Literal["auto", "jsonl", "stream-json", "text"]
    model: str | None
    session_timeout_s: float
    extra_env: dict[str, str]


@dataclass
class FrontendRunArtifacts:
    session_dir: Path
    prompt_path: Path
    session_json_path: Path
    stdout_path: Path
    stderr_path: Path
    final_message_path: Path | None


@dataclass
class FrontendEvent:
    ts_monotonic: float
    kind: str
    text_delta: str = ""
    usage: dict[str, int] | None = None
    raw: dict | str | None = None


class FrontendProvider(Protocol):
    name: str

    def render(self, session: ReplaySession, artifacts: FrontendRunArtifacts) -> None: ...

    def build_command(
        self, artifacts: FrontendRunArtifacts, config: FrontendConfig
    ) -> list[str]: ...


class FrontendEventParser(Protocol):
    provider_name: str

    def feed_stdout_line(self, line: str, ts_monotonic: float) -> list[FrontendEvent]: ...

    def feed_stderr_line(self, line: str, ts_monotonic: float) -> list[FrontendEvent]: ...

    def finish(self, exit_code: int, ts_monotonic: float) -> list[FrontendEvent]: ...


@dataclass(frozen=True)
class FrontendProviderCapabilities:
    name: str
    supports_custom_base_url: bool
    supports_openai_compatible_endpoint: bool
    model_name_format: str
    auth_sources: tuple[str, ...]
    config_file_strategy: str
    proxy_supported: bool
    structured_output_format: str | None
    partial_text_events: Literal["yes", "no", "unknown"]
    final_usage_events: Literal["yes", "no", "unknown"]
    requires_git_repo: bool
    required_env: tuple[str, ...]
    known_unsupported_modes: tuple[str, ...]
