"""OpenCode JSONL streaming parser tests against real v1.14.29 fixtures."""

from __future__ import annotations

from pathlib import Path

import agentsurge.frontends.events as E
from agentsurge.frontends.base import FrontendConfig, FrontendRunArtifacts
from agentsurge.frontends.events import (
    EVENT_ASSISTANT_TEXT_DELTA,
    EVENT_ERROR,
    EVENT_PARSER_UNKNOWN,
    EVENT_SESSION_COMPLETED,
    EVENT_SESSION_STARTED,
    EVENT_TOOL_USE_OBSERVED,
    EVENT_TURN_COMPLETED,
    EVENT_TURN_STARTED,
    EVENT_USAGE_COMPLETED,
)
from agentsurge.frontends.opencode import OpenCodeEventParser, OpenCodeProvider

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "opencode"


def test_parse_real_simple_session_v1_14_29():
    """Real OpenCode 1.14.29 success-path: JSONL stream with step_start/text/tool_use/step_finish."""
    fixture = FIXTURE_DIR / "real_simple_session_v1.14.29.jsonl"
    parser = OpenCodeEventParser()
    events = []
    for line in fixture.read_text().splitlines():
        if line.strip():
            events.extend(parser.feed_stdout_line(line + "\n", ts_monotonic=0.0))
    events.extend(parser.finish(0, 0.0))

    kinds = [e.kind for e in events]
    # Expected sequence:
    # SESSION_STARTED (from first event), TURN_STARTED, ASSISTANT_TEXT_DELTA,
    # TOOL_USE_OBSERVED (tool_use — contamination signal),
    # TURN_COMPLETED (reason=tool-calls; no usage on this one), TURN_STARTED,
    # ASSISTANT_TEXT_DELTA,
    # TURN_COMPLETED + USAGE_COMPLETED + SESSION_COMPLETED (reason=stop).
    assert kinds[0] == EVENT_SESSION_STARTED
    assert kinds.count(EVENT_TURN_STARTED) == 2
    assert kinds.count(EVENT_ASSISTANT_TEXT_DELTA) == 2
    assert EVENT_TOOL_USE_OBSERVED in kinds  # tool_use → contamination signal
    assert EVENT_PARSER_UNKNOWN not in kinds  # tool_use must NOT be silently dropped
    assert kinds.count(EVENT_TURN_COMPLETED) == 2
    assert kinds.count(EVENT_USAGE_COMPLETED) == 1  # only on final step_finish
    assert kinds.count(EVENT_SESSION_COMPLETED) == 1

    usage_event = next(e for e in events if e.kind == EVENT_USAGE_COMPLETED)
    assert usage_event.usage["input_tokens"] == 78
    assert usage_event.usage["output_tokens"] == 25
    assert usage_event.usage["cache_read"] == 13328
    assert usage_event.usage["cache_write"] == 0


def test_parse_real_error_session_v1_14_29():
    """Real OpenCode 1.14.29 error-path: single 'error' event line."""
    fixture = FIXTURE_DIR / "real_error_session_v1.14.29.jsonl"
    parser = OpenCodeEventParser()
    events = []
    for line in fixture.read_text().splitlines():
        if line.strip():
            events.extend(parser.feed_stdout_line(line + "\n", ts_monotonic=0.0))
    events.extend(parser.finish(0, 0.0))  # opencode exits 0 even on error

    kinds = [e.kind for e in events]
    assert kinds == [EVENT_SESSION_STARTED, EVENT_ERROR]
    error_event = events[1]
    assert "no-such-model-xyz" in error_event.raw["error"]["data"]["message"]


def _make_artifacts(tmp_path: Path) -> FrontendRunArtifacts:
    session_dir = tmp_path / "s0"
    session_dir.mkdir(parents=True, exist_ok=True)
    return FrontendRunArtifacts(
        session_dir=session_dir,
        prompt_path=session_dir / "prompt.md",
        session_json_path=session_dir / "session.json",
        stdout_path=session_dir / "stdout.jsonl",
        stderr_path=session_dir / "stderr.log",
        final_message_path=None,
    )


def _make_config(model: str | None) -> FrontendConfig:
    return FrontendConfig(
        name="opencode",
        command_template=None,
        workspace_dir=Path("/tmp"),
        prompt_mode="auto",
        output_format="jsonl",
        model=model,
        session_timeout_s=60.0,
        extra_env={},
    )


def test_opencode_provider_build_command_default(tmp_path: Path):
    artifacts = _make_artifacts(tmp_path)
    config = _make_config("anthropic/claude-3-5-sonnet")
    cmd = OpenCodeProvider().build_command(artifacts, config)
    prompt_path = str(artifacts.prompt_path)
    # Message must precede flags and --file uses `=` syntax — yargs greedily
    # consumes `--file <path> <message>` as a two-element file array.
    assert cmd == [
        "opencode",
        "run",
        "Complete the AgentSurge session described in the attached "
        "file. Respond with text only — do not call any tools, do not "
        "run shell commands, do not read or edit files.",
        "--format",
        "json",
        "--model",
        "anthropic/claude-3-5-sonnet",
        f"--file={prompt_path}",
    ]


def test_opencode_provider_build_command_no_model(tmp_path: Path):
    artifacts = _make_artifacts(tmp_path)
    config = _make_config(None)
    cmd = OpenCodeProvider().build_command(artifacts, config)
    prompt_path = str(artifacts.prompt_path)
    assert "--model" not in cmd
    assert cmd == [
        "opencode",
        "run",
        "Complete the AgentSurge session described in the attached "
        "file. Respond with text only — do not call any tools, do not "
        "run shell commands, do not read or edit files.",
        "--format",
        "json",
        f"--file={prompt_path}",
    ]


def test_opencode_parser_routes_tool_use_to_observed_kind():
    """OpenCode emits a `tool_use` JSONL event when it invokes a built-in
    tool (bash/edit/grep/...). In benchmark mode the in-process
    RequestShim strips `tools` from /v1/chat/completions so this should
    never fire; if it does, the run is contaminated and downstream
    metrics are not a clean throughput measurement. Pin the routing so a
    refactor of OpenCodeEventParser._dispatch can't silently regress to
    PARSER_UNKNOWN."""
    parser = OpenCodeEventParser()
    raw = (
        '{"type":"tool_use","timestamp":1777420919152,'
        '"sessionID":"ses_229752c43ffe3C5ZhTXEto9hCH",'
        '"part":{"type":"tool","tool":"bash","callID":"call_function_0zce5eii9tsw_1",'
        '"state":{"status":"completed","input":{"command":"echo \\"Hello, world!\\"","description":"Print one short sentence"},'
        '"output":"Hello, world!\\n","metadata":{"output":"Hello, world!\\n","exit":0,'
        '"description":"Print one short sentence","truncated":false},'
        '"title":"Print one short sentence","time":{"start":1777420919095,"end":1777420919120}},'
        '"id":"prt_dd68ae0990013w32BKZcCDsdSd",'
        '"sessionID":"ses_229752c43ffe3C5ZhTXEto9hCH",'
        '"messageID":"msg_dd68ad4c9001v2t8YU6R4aCWVM"}}\n'
    )
    events = parser.feed_stdout_line(raw, ts_monotonic=0.0)
    assert len(events) == 2  # SESSION_STARTED (lazy) + TOOL_USE_OBSERVED
    tool_use_event = events[1]
    assert tool_use_event.kind == E.EVENT_TOOL_USE_OBSERVED
