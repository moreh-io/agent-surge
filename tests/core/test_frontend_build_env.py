"""Per-provider build_env tests.

Each provider implements build_env(artifacts, config) to translate
``--frontend-server-url`` into the CLI-specific override mechanism (env vars,
config files, or both). These tests lock the contract validated by the
mi250-069 E2E run on 2026-04-29:

- Claude: ANTHROPIC_BASE_URL env, plus ANTHROPIC_AUTH_TOKEN mirrored from
  the user's ANTHROPIC_API_KEY because vLLM's /v1/messages accepts only
  ``Authorization: Bearer ...`` (Claude populates that from
  ANTHROPIC_AUTH_TOKEN, not ANTHROPIC_API_KEY which becomes x-api-key).
- Codex: $CODEX_HOME/config.toml file with model_provider + base_url +
  wire_api=responses; env returns CODEX_HOME path.
- OpenCode: opencode.json file in cwd (= session_dir) with provider config;
  env always carries XDG_DATA_HOME (per-session SQLite isolation) regardless
  of server_url because the WAL race exists in direct mode too.
- Echo: always returns empty.

When server_url is not set, Claude/Codex/Echo return ``{}`` and write no
files; OpenCode still returns ``{"XDG_DATA_HOME": ...}`` to keep its session
DB hermetic.
"""

from __future__ import annotations

import json
from pathlib import Path

from agentsurge.frontends.base import (
    FrontendConfig,
    FrontendRunArtifacts,
)
from agentsurge.frontends.claude import ClaudeProvider
from agentsurge.frontends.codex import CodexProvider
from agentsurge.frontends.echo import EchoProvider
from agentsurge.frontends.opencode import OpenCodeProvider


def _artifacts(session_dir: Path) -> FrontendRunArtifacts:
    session_dir.mkdir(parents=True, exist_ok=True)
    return FrontendRunArtifacts(
        session_dir=session_dir,
        prompt_path=session_dir / "prompt.md",
        session_json_path=session_dir / "session.json",
        stdout_path=session_dir / "stdout.jsonl",
        stderr_path=session_dir / "stderr.log",
        final_message_path=None,
    )


def _config(
    server_url: str | None = None,
    model: str | None = "qwen3.6-27b",
    extra_env: dict[str, str] | None = None,
) -> FrontendConfig:
    return FrontendConfig(
        name="test",
        command_template=None,
        workspace_dir=Path("/tmp/x"),
        prompt_mode="auto",
        output_format="auto",
        model=model,
        session_timeout_s=30.0,
        extra_env=extra_env or {},
        server_url=server_url,
    )


# ---------- Echo ----------


def test_echo_build_env_always_empty(tmp_path: Path) -> None:
    a = _artifacts(tmp_path / "s")
    assert EchoProvider().build_env(a, _config(server_url="http://x")) == {}
    assert EchoProvider().build_env(a, _config(server_url=None)) == {}


# ---------- Claude ----------


def test_claude_build_env_no_server_url_returns_empty(tmp_path: Path) -> None:
    env = ClaudeProvider().build_env(_artifacts(tmp_path / "s"), _config(server_url=None))
    assert env == {}


def test_claude_build_env_sets_base_url(tmp_path: Path) -> None:
    env = ClaudeProvider().build_env(
        _artifacts(tmp_path / "s"),
        _config(server_url="http://127.0.0.1:18000/"),
    )
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:18000"


def test_claude_build_env_mirrors_api_key_into_auth_token(tmp_path: Path) -> None:
    """vLLM /v1/messages only accepts Authorization Bearer (populated from
    ANTHROPIC_AUTH_TOKEN); without this mirror every request 401's."""
    env = ClaudeProvider().build_env(
        _artifacts(tmp_path / "s"),
        _config(
            server_url="http://127.0.0.1:18000",
            extra_env={"ANTHROPIC_API_KEY": "sk-test-vllm-key"},
        ),
    )
    assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-test-vllm-key"


def test_claude_build_env_no_mirror_when_no_key(tmp_path: Path) -> None:
    env = ClaudeProvider().build_env(
        _artifacts(tmp_path / "s"),
        _config(server_url="http://x", extra_env={}),
    )
    assert "ANTHROPIC_AUTH_TOKEN" not in env


def test_claude_capability_supports_custom_base_url() -> None:
    assert ClaudeProvider().capabilities.supports_custom_base_url is True


# ---------- Codex ----------


def test_codex_build_env_no_server_url_returns_empty(tmp_path: Path) -> None:
    env = CodexProvider().build_env(_artifacts(tmp_path / "s"), _config(server_url=None))
    assert env == {}
    assert not (tmp_path / "s" / "codex_home").exists()


def test_codex_build_env_writes_config_toml(tmp_path: Path) -> None:
    session_dir = tmp_path / "s_codex"
    env = CodexProvider().build_env(
        _artifacts(session_dir),
        _config(server_url="http://127.0.0.1:18000"),
    )
    assert env["CODEX_HOME"] == str(session_dir / "codex_home")
    config_path = session_dir / "codex_home" / "config.toml"
    assert config_path.is_file()
    body = config_path.read_text()
    # The provider key must match the config's [model_providers.X] section
    # name; mismatch silently falls back to default ChatGPT auth.
    assert 'model_provider = "agentsurge_target"' in body
    assert "[model_providers.agentsurge_target]" in body
    # /v1 suffix must be present — Codex appends "/responses" without /v1.
    assert 'base_url = "http://127.0.0.1:18000/v1"' in body
    # Codex 0.125+ rejects wire_api=chat; responses is required.
    assert 'wire_api = "responses"' in body


def test_codex_build_env_appends_v1_when_missing(tmp_path: Path) -> None:
    """Users typically pass the bare host URL; provider must add /v1."""
    CodexProvider().build_env(
        _artifacts(tmp_path / "s"),
        _config(server_url="http://server:8000"),
    )
    body = (tmp_path / "s" / "codex_home" / "config.toml").read_text()
    assert 'base_url = "http://server:8000/v1"' in body


def test_codex_build_env_preserves_v1_when_present(tmp_path: Path) -> None:
    CodexProvider().build_env(
        _artifacts(tmp_path / "s"),
        _config(server_url="http://server:8000/v1"),
    )
    body = (tmp_path / "s" / "codex_home" / "config.toml").read_text()
    assert 'base_url = "http://server:8000/v1"' in body
    assert "/v1/v1" not in body


def test_codex_capability_supports_custom_base_url() -> None:
    assert CodexProvider().capabilities.supports_custom_base_url is True


# ---------- OpenCode ----------


def test_opencode_build_env_no_server_url_writes_no_opencode_json(tmp_path: Path) -> None:
    """Without server_url, the provider config file is not written, but
    XDG_DATA_HOME isolation is still applied (see
    test_opencode_build_env_isolates_xdg_data_home_always)."""
    session_dir = tmp_path / "s"
    env = OpenCodeProvider().build_env(_artifacts(session_dir), _config(server_url=None))
    assert "XDG_DATA_HOME" in env
    assert not (session_dir / "opencode.json").exists()


def test_opencode_build_env_no_provider_prefix_in_model_skips_opencode_json(tmp_path: Path) -> None:
    """Without a slash-prefixed model the provider can't know which
    OpenCode provider key to register; skip the config file rather than
    invent a name that won't match --model. XDG_DATA_HOME isolation is
    still applied (see test_opencode_build_env_isolates_xdg_data_home_always)."""
    session_dir = tmp_path / "s"
    env = OpenCodeProvider().build_env(
        _artifacts(session_dir),
        _config(server_url="http://x", model="qwen3.6-27b"),
    )
    assert "XDG_DATA_HOME" in env
    assert not (session_dir / "opencode.json").exists()


def test_opencode_build_env_writes_provider_config(tmp_path: Path) -> None:
    session_dir = tmp_path / "s_opencode"
    OpenCodeProvider().build_env(
        _artifacts(session_dir),
        _config(
            server_url="http://127.0.0.1:18000",
            model="vllm/qwen3.6-27b",
            extra_env={"OPENAI_API_KEY": "sk-test-vllm-key"},
        ),
    )
    config_path = session_dir / "opencode.json"
    assert config_path.is_file()
    cfg = json.loads(config_path.read_text())
    provider = cfg["provider"]["vllm"]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"]["baseURL"] == "http://127.0.0.1:18000/v1"
    assert provider["options"]["apiKey"] == "sk-test-vllm-key"
    assert "qwen3.6-27b" in provider["models"]


def test_opencode_build_env_disables_all_builtin_tools(tmp_path: Path) -> None:
    """opencode.json must declare every built-in tool as ``false``. Without
    this opencode advertises bash/edit/grep/etc. to the model and qwen3.6
    loops on tool calls until session_timeout fires (regression guard for
    the 2026-04-29 mi250-069 4-concurrent OpenCode smoke timeout)."""
    session_dir = tmp_path / "s_no_tools"
    OpenCodeProvider().build_env(
        _artifacts(session_dir),
        _config(server_url="http://x", model="vllm/qwen3.6-27b"),
    )
    cfg = json.loads((session_dir / "opencode.json").read_text())
    tools = cfg["tools"]
    # Every tool name in the declared map must be disabled; the harness has
    # no value to gain from any of them. New tool names should be added to
    # build_env when opencode adds them.
    assert all(v is False for v in tools.values()), f"all tools must be False; got {tools}"
    assert {"bash", "edit", "write", "read", "grep"}.issubset(tools.keys())


def test_opencode_build_env_empty_api_key_when_not_provided(tmp_path: Path) -> None:
    session_dir = tmp_path / "s"
    OpenCodeProvider().build_env(
        _artifacts(session_dir),
        _config(server_url="http://x", model="vllm/test", extra_env={}),
    )
    cfg = json.loads((session_dir / "opencode.json").read_text())
    assert cfg["provider"]["vllm"]["options"]["apiKey"] == ""


def test_opencode_capability_supports_custom_base_url() -> None:
    assert OpenCodeProvider().capabilities.supports_custom_base_url is True


def test_opencode_build_env_isolates_xdg_data_home_always(tmp_path: Path) -> None:
    """OpenCode's SQLite session DB lives at $XDG_DATA_HOME/opencode/opencode.db.
    Concurrent sessions sharing the default $HOME race on the WAL files. Per-
    session XDG_DATA_HOME is always set so this hazard is fixed even when
    server_url is not configured."""
    session_dir = tmp_path / "s_xdg_default"
    env = OpenCodeProvider().build_env(_artifacts(session_dir), _config(server_url=None))
    expected = session_dir / "xdg"
    assert env.get("XDG_DATA_HOME") == str(expected)
    assert expected.is_dir()


def test_opencode_build_env_isolation_also_present_with_server_url(tmp_path: Path) -> None:
    """With server_url set we still need the XDG isolation; the opencode.json
    write must not displace it."""
    session_dir = tmp_path / "s_xdg_server"
    env = OpenCodeProvider().build_env(
        _artifacts(session_dir),
        _config(server_url="http://127.0.0.1:18000", model="vllm/qwen3.6-27b"),
    )
    assert env.get("XDG_DATA_HOME") == str(session_dir / "xdg")
    assert (session_dir / "opencode.json").is_file()


# ---------- Renderer integration ----------


def test_renderer_passes_provider_env_to_subprocess(tmp_path: Path) -> None:
    """The renderer must merge provider.build_env() into the subprocess env
    so the configured base_url actually reaches the spawned CLI. Probe via
    the echo provider's --frontend-extra-env channel: echo emits whatever
    AGENTSURGE_TEST_ENV_PROBE is set to, so we use a custom EchoProvider
    subclass to assert build_env is called and its values land."""
    import asyncio

    from agentsurge.frontends.runner import FrontendSessionRenderer
    from agentsurge.types import (
        BenchmarkConfig,
        FrontendRuntimeSettings,
        ReplaySession,
    )

    class _ProbeEchoProvider(EchoProvider):
        def build_env(
            self, artifacts: FrontendRunArtifacts, config: FrontendConfig
        ) -> dict[str, str]:
            return {"AGENTSURGE_TEST_ENV_PROBE": "from-build-env"}

    cfg = BenchmarkConfig(
        vllm_url="http://x",
        model="t",
        no_metrics=True,
        frontend=FrontendRuntimeSettings(
            name="echo",
            session_timeout_s=10.0,
            keep_artifacts="always",
        ),
    )
    renderer = FrontendSessionRenderer(cfg, tmp_path)
    original_resolve = renderer._resolve_provider

    def patched_resolve(name: str):
        if name == "echo":
            from agentsurge.frontends.echo import EchoEventParser

            return _ProbeEchoProvider(), EchoEventParser()
        return original_resolve(name)

    renderer._resolve_provider = patched_resolve  # type: ignore[method-assign]

    session = ReplaySession(
        session_id="s_env",
        turn_messages=[[{"role": "user", "content": "hi"}]],
        metadata={},
    )
    asyncio.run(renderer.run(session))

    stdout_text = (tmp_path / "s_env" / "stdout.jsonl").read_text()
    assert "from-build-env" in stdout_text
