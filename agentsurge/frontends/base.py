# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from agentsurge.types import ReplaySession


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
    server_url: str | None = None


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
    raw: dict[str, object] | str | None = None


class FrontendProvider(Protocol):
    name: str
    capabilities: FrontendProviderCapabilities

    def render(self, session: ReplaySession, artifacts: FrontendRunArtifacts) -> None: ...

    def build_command(
        self, artifacts: FrontendRunArtifacts, config: FrontendConfig
    ) -> list[str]: ...

    def build_env(self, artifacts: FrontendRunArtifacts, config: FrontendConfig) -> dict[str, str]:
        """Return env vars to inject into the spawned CLI subprocess.

        May also write provider-specific config files into artifacts.session_dir
        as a side effect (Codex needs ~/.codex/config.toml, OpenCode needs
        ./opencode.json) — the env vars then point the CLI at those files.

        When ``config.server_url`` is set, providers that declare
        ``supports_custom_base_url=True`` translate it into the CLI's native
        base-URL configuration. Providers that don't support it must raise
        ``CapabilityError`` so the run fails loudly instead of silently
        hitting the CLI's persisted public-provider auth.
        """
        ...


class CapabilityError(RuntimeError):
    """Raised when ``--frontend-server-url`` is set on a provider that does
    not support a configurable base URL."""


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
    requires_request_rewrite: bool = False
