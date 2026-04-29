"""Claude stream-json parser tests against synthetic fixtures."""

from __future__ import annotations

import time
from pathlib import Path

from agentsurge.frontends import events as E
from agentsurge.frontends.base import FrontendConfig, FrontendRunArtifacts
from agentsurge.frontends.claude import ClaudeEventParser, ClaudeProvider

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "claude"


def _ts() -> float:
    return time.monotonic()


def _feed_file(parser: ClaudeEventParser, path: Path) -> list:
    events = []
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        events.extend(parser.feed_stdout_line(line, _ts()))
    return events


def test_parse_simple_session():
    parser = ClaudeEventParser()
    events = _feed_file(parser, FIXTURE_DIR / "simple_session.jsonl")
    kinds = [ev.kind for ev in events]
    assert kinds == [
        E.EVENT_SESSION_STARTED,
        E.EVENT_ASSISTANT_TEXT_DELTA,
        E.EVENT_ASSISTANT_TEXT_DELTA,
        E.EVENT_ASSISTANT_MESSAGE_COMPLETED,
        E.EVENT_USAGE_COMPLETED,
        E.EVENT_SESSION_COMPLETED,
    ]
    assert events[1].text_delta == "hi "
    assert events[2].text_delta == "there"
    assert events[3].text_delta == "hi there"
    assert events[4].usage == {"input_tokens": 10, "output_tokens": 3}


def test_parse_error_event():
    parser = ClaudeEventParser()
    events = _feed_file(parser, FIXTURE_DIR / "error_session.jsonl")
    kinds = [ev.kind for ev in events]
    assert kinds == [E.EVENT_SESSION_STARTED, E.EVENT_ERROR]
    raw = events[1].raw
    assert isinstance(raw, dict)
    assert raw["error"] == "rate limited"


def test_parse_partial_line_buffering():
    parser = ClaudeEventParser()
    events = []
    for line in (
        (FIXTURE_DIR / "simple_session.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)
    ):
        midpoint = len(line) // 2
        first = line[:midpoint]
        second = line[midpoint:]
        events.extend(parser.feed_stdout_line(first, _ts()))
        events.extend(parser.feed_stdout_line(second, _ts()))
    kinds = [ev.kind for ev in events]
    assert kinds == [
        E.EVENT_SESSION_STARTED,
        E.EVENT_ASSISTANT_TEXT_DELTA,
        E.EVENT_ASSISTANT_TEXT_DELTA,
        E.EVENT_ASSISTANT_MESSAGE_COMPLETED,
        E.EVENT_USAGE_COMPLETED,
        E.EVENT_SESSION_COMPLETED,
    ]


def test_parse_multiple_lines_in_single_call():
    parser = ClaudeEventParser()
    payload = (
        '{"type":"system","subtype":"init"}\n'
        '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"hello"}]}}\n'
    )
    events = parser.feed_stdout_line(payload, ts_monotonic=0.0)
    assert [e.kind for e in events] == [
        E.EVENT_SESSION_STARTED,
        E.EVENT_ASSISTANT_MESSAGE_COMPLETED,
    ]
    assert events[1].text_delta == "hello"


def test_parse_garbage_line_emits_parser_error_and_recovers():
    parser = ClaudeEventParser()
    payload = 'not json\n{"type":"system","subtype":"init"}\n'
    events = parser.feed_stdout_line(payload, ts_monotonic=0.0)
    kinds = [e.kind for e in events]
    assert kinds == [E.EVENT_PARSER_ERROR, E.EVENT_SESSION_STARTED]


def test_parse_unknown_kind():
    parser = ClaudeEventParser()
    events = parser.feed_stdout_line('{"type":"weird"}\n', _ts())
    assert len(events) == 1
    assert events[0].kind == E.EVENT_PARSER_UNKNOWN
    assert isinstance(events[0].raw, dict)
    assert events[0].raw["type"] == "weird"


def test_parse_truncated_final_line():
    parser = ClaudeEventParser()
    events = parser.feed_stdout_line('{"type":"system', _ts())
    assert events == []
    final_events = parser.finish(0, _ts())
    assert len(final_events) == 1
    assert final_events[0].kind == E.EVENT_PARSER_ERROR
    raw = final_events[0].raw
    assert isinstance(raw, str)
    assert "system" in raw


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
        name="claude",
        command_template=None,
        workspace_dir=Path("/tmp"),
        prompt_mode="auto",
        output_format="stream-json",
        model=model,
        session_timeout_s=60.0,
        extra_env={},
    )


_NO_TOOLS_MESSAGE_SUFFIX = (
    " Respond with text only — do not call any tools, do not run shell "
    "commands, do not read or edit files."
)


def test_claude_provider_build_command_default(tmp_path: Path):
    artifacts = _make_artifacts(tmp_path)
    config = _make_config(None)
    cmd = ClaudeProvider().build_command(artifacts, config)
    prompt_path = str(artifacts.prompt_path)
    assert cmd == [
        "claude",
        "--verbose",
        "-p",
        "--output-format",
        "stream-json",
        "--include-partial-messages",
        "--allowedTools",
        "",
        "--",
        f"Read {prompt_path} and complete the AgentSurge session "
        f"described there.{_NO_TOOLS_MESSAGE_SUFFIX}",
    ]


def test_claude_provider_build_command_with_model(tmp_path: Path):
    artifacts = _make_artifacts(tmp_path)
    config = _make_config("claude-opus-4-7")
    cmd = ClaudeProvider().build_command(artifacts, config)
    prompt_path = str(artifacts.prompt_path)
    assert cmd == [
        "claude",
        "--verbose",
        "-p",
        "--output-format",
        "stream-json",
        "--include-partial-messages",
        "--allowedTools",
        "",
        "--model",
        "claude-opus-4-7",
        "--",
        f"Read {prompt_path} and complete the AgentSurge session "
        f"described there.{_NO_TOOLS_MESSAGE_SUFFIX}",
    ]


def test_parse_real_simple_session_v2_1_122():
    """Real Claude Code 2.1.122 success-path fixture parses cleanly.

    The capture includes hook events (which we route to PARSER_UNKNOWN), an init system
    event, several stream_event envelopes (message_start, content_block_*,
    message_delta, message_stop), a full assistant message, a rate_limit_event,
    and a final result event.
    """
    fixture = Path(__file__).parent / "fixtures" / "claude" / "real_simple_session_v2.1.122.jsonl"
    parser = ClaudeEventParser()
    events = []
    for line in fixture.read_text().splitlines():
        if line.strip():
            events.extend(parser.feed_stdout_line(line + "\n", ts_monotonic=0.0))
    events.extend(parser.finish(0, 0.0))

    kinds = [e.kind for e in events]
    assert E.EVENT_SESSION_STARTED in kinds
    assert E.EVENT_MESSAGE_START in kinds
    assert kinds.count(E.EVENT_ASSISTANT_TEXT_DELTA) == 2
    assert E.EVENT_ASSISTANT_MESSAGE_COMPLETED in kinds
    assert E.EVENT_USAGE_COMPLETED in kinds
    assert E.EVENT_SESSION_COMPLETED in kinds
    assert E.EVENT_PARSER_ERROR not in kinds


def test_parse_real_error_session_v2_1_122():
    """Real Claude Code 2.1.122 error-path fixture: result.is_error=True surfaces as EVENT_ERROR."""
    fixture = Path(__file__).parent / "fixtures" / "claude" / "real_error_session_v2.1.122.jsonl"
    parser = ClaudeEventParser()
    events = []
    for line in fixture.read_text().splitlines():
        if line.strip():
            events.extend(parser.feed_stdout_line(line + "\n", ts_monotonic=0.0))
    events.extend(parser.finish(1, 0.0))

    kinds = [e.kind for e in events]
    assert E.EVENT_ERROR in kinds
    assert E.EVENT_USAGE_COMPLETED not in kinds
    assert E.EVENT_SESSION_COMPLETED not in kinds
    error_event = next(e for e in events if e.kind == E.EVENT_ERROR)
    assert isinstance(error_event.raw, dict)
    assert error_event.raw.get("api_error_status") == 404


def test_finish_drains_buffer():
    parser = ClaudeEventParser()
    parser.feed_stdout_line('{"type":"sys', 0.0)
    final_events = parser.finish(0, 0.0)
    assert len(final_events) == 1
    assert final_events[0].kind == E.EVENT_PARSER_ERROR
    assert isinstance(final_events[0].raw, str)
    assert "sys" in final_events[0].raw
