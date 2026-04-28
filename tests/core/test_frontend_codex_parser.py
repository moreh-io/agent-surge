"""Codex JSONL parser tests against synthetic fixtures."""

from __future__ import annotations

import time
from pathlib import Path

from agentsurge.frontends import events as E
from agentsurge.frontends.base import FrontendConfig, FrontendRunArtifacts
from agentsurge.frontends.codex import CodexEventParser, CodexProvider

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "codex"


def _ts() -> float:
    return time.monotonic()


def _feed_file(parser: CodexEventParser, path: Path) -> list:
    events = []
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        events.extend(parser.feed_stdout_line(line, _ts()))
    return events


def test_parse_simple_session():
    parser = CodexEventParser()
    events = _feed_file(parser, FIXTURE_DIR / "simple_session.jsonl")
    kinds = [ev.kind for ev in events]
    assert kinds == [
        E.EVENT_SESSION_STARTED,
        E.EVENT_TURN_STARTED,
        E.EVENT_ASSISTANT_MESSAGE_COMPLETED,
        E.EVENT_TURN_COMPLETED,
        E.EVENT_USAGE_COMPLETED,
    ]
    msg_event = events[2]
    assert msg_event.text_delta == "hi"
    usage_event = events[4]
    assert usage_event.usage == {"input_tokens": 10, "output_tokens": 2}


def test_parse_error_event():
    parser = CodexEventParser()
    events = _feed_file(parser, FIXTURE_DIR / "error_session.jsonl")
    assert len(events) == 2
    assert events[0].kind == E.EVENT_SESSION_STARTED
    assert events[1].kind == E.EVENT_ERROR
    assert isinstance(events[1].raw, dict)
    assert events[1].raw["message"] == "model overloaded"


def test_parse_partial_line_buffering():
    parser = CodexEventParser()
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
        E.EVENT_TURN_STARTED,
        E.EVENT_ASSISTANT_MESSAGE_COMPLETED,
        E.EVENT_TURN_COMPLETED,
        E.EVENT_USAGE_COMPLETED,
    ]


def test_parse_multiple_lines_in_single_call():
    """A single feed_stdout_line call may contain multiple complete lines."""
    parser = CodexEventParser()
    payload = (
        '{"type":"thread.started","thread_id":"t1"}\n'
        '{"type":"item.completed","item":{"type":"agent_message","text":"hello"}}\n'
    )
    events = parser.feed_stdout_line(payload, ts_monotonic=0.0)
    assert [e.kind for e in events] == [
        E.EVENT_SESSION_STARTED,
        E.EVENT_ASSISTANT_MESSAGE_COMPLETED,
    ]
    assert events[1].text_delta == "hello"


def test_parse_garbage_line_emits_parser_error_and_recovers():
    """A complete line that's not JSON emits PARSER_ERROR; subsequent lines parse normally."""
    parser = CodexEventParser()
    payload = 'not json at all\n{"type":"thread.started","thread_id":"t1"}\n'
    events = parser.feed_stdout_line(payload, ts_monotonic=0.0)
    kinds = [e.kind for e in events]
    # Order matters: recovery must happen AFTER the error, not before it.
    assert kinds == [E.EVENT_PARSER_ERROR, E.EVENT_SESSION_STARTED]


def test_parse_unknown_kind():
    parser = CodexEventParser()
    events = parser.feed_stdout_line('{"type":"weird","data":42}\n', _ts())
    assert len(events) == 1
    assert events[0].kind == E.EVENT_PARSER_UNKNOWN
    assert isinstance(events[0].raw, dict)
    assert events[0].raw["type"] == "weird"


def test_parse_truncated_final_line():
    parser = CodexEventParser()
    events = parser.feed_stdout_line('{"type":"thread.start', _ts())
    assert events == []
    final_events = parser.finish(0, _ts())
    assert len(final_events) == 1
    assert final_events[0].kind == E.EVENT_PARSER_ERROR
    raw = final_events[0].raw
    assert isinstance(raw, str)
    assert "thread.start" in raw


def test_parse_real_simple_session_v0_125_0():
    """Real codex 0.125.0 success-path fixture parses cleanly with inline usage on turn.completed."""
    fixture = Path(__file__).parent / "fixtures" / "codex" / "real_simple_session_v0.125.0.jsonl"
    parser = CodexEventParser()
    events = []
    for line in fixture.read_text().splitlines():
        events.extend(parser.feed_stdout_line(line + "\n", ts_monotonic=0.0))
    events.extend(parser.finish(0, 0.0))

    kinds = [e.kind for e in events]
    # Real CLI emits: thread.started, turn.started, item.completed (agent_message),
    # turn.completed (with inline usage -> emits both turn.completed AND usage events)
    assert kinds == [
        E.EVENT_SESSION_STARTED,
        E.EVENT_TURN_STARTED,
        E.EVENT_ASSISTANT_MESSAGE_COMPLETED,
        E.EVENT_TURN_COMPLETED,
        E.EVENT_USAGE_COMPLETED,
    ]
    # assistant text
    assert events[2].text_delta == "I will not call any tools."
    # usage carries the full token shape including cached_input_tokens and reasoning_output_tokens
    usage_event = events[4]
    assert usage_event.usage["input_tokens"] == 12804
    assert usage_event.usage["cached_input_tokens"] == 10624
    assert usage_event.usage["output_tokens"] == 11
    assert usage_event.usage["reasoning_output_tokens"] == 0


def test_parse_real_error_session_v0_125_0():
    """Real codex 0.125.0 error-path fixture parses cleanly: error + turn.failed both surface."""
    fixture = Path(__file__).parent / "fixtures" / "codex" / "real_error_session_v0.125.0.jsonl"
    parser = CodexEventParser()
    events = []
    for line in fixture.read_text().splitlines():
        events.extend(parser.feed_stdout_line(line + "\n", ts_monotonic=0.0))
    events.extend(parser.finish(1, 0.0))

    kinds = [e.kind for e in events]
    assert kinds == [
        E.EVENT_SESSION_STARTED,
        E.EVENT_TURN_STARTED,
        E.EVENT_ERROR,
        E.EVENT_TURN_FAILED,
    ]
    # error message is escaped JSON; parser preserves verbatim
    error_event = events[2]
    assert "NO_SUCH_MODEL_xyz" in error_event.raw.get("message", "")


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
        name="codex",
        command_template=None,
        workspace_dir=Path("/tmp"),
        prompt_mode="auto",
        output_format="jsonl",
        model=model,
        session_timeout_s=60.0,
        extra_env={},
    )


def test_codex_provider_build_command_default(tmp_path: Path):
    artifacts = _make_artifacts(tmp_path)
    config = _make_config("gpt-5.2-codex")
    cmd = CodexProvider().build_command(artifacts, config)
    workspace = str(artifacts.session_dir)
    final_message = str(artifacts.session_dir / "final.txt")
    prompt_path = str(artifacts.prompt_path)
    assert cmd == [
        "codex",
        "exec",
        "--json",
        "--ephemeral",
        "--cd",
        workspace,
        "--model",
        "gpt-5.2-codex",
        "--output-last-message",
        final_message,
        f"Read {prompt_path} and complete the AgentSurge session described there.",
    ]


def test_codex_provider_build_command_no_model(tmp_path: Path):
    artifacts = _make_artifacts(tmp_path)
    config = _make_config(None)
    cmd = CodexProvider().build_command(artifacts, config)
    workspace = str(artifacts.session_dir)
    final_message = str(artifacts.session_dir / "final.txt")
    prompt_path = str(artifacts.prompt_path)
    assert cmd == [
        "codex",
        "exec",
        "--json",
        "--ephemeral",
        "--cd",
        workspace,
        "--output-last-message",
        final_message,
        f"Read {prompt_path} and complete the AgentSurge session described there.",
    ]
