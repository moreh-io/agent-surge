"""OpenCode JSON-document parser tests against synthetic fixtures."""

from __future__ import annotations

import json
import time
from pathlib import Path

from agentsurge.frontends import events as E
from agentsurge.frontends.base import FrontendConfig, FrontendRunArtifacts
from agentsurge.frontends.opencode import OpenCodeEventParser, OpenCodeProvider

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "opencode"


def _ts() -> float:
    return time.monotonic()


def _feed_file(parser: OpenCodeEventParser, path: Path) -> list:
    events = []
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        events.extend(parser.feed_stdout_line(line, _ts()))
    return events


def test_parse_simple_session():
    parser = OpenCodeEventParser()
    streaming_events = _feed_file(parser, FIXTURE_DIR / "simple_session.json")
    assert streaming_events == []
    events = parser.finish(0, 1.0)
    kinds = [ev.kind for ev in events]
    assert kinds == [
        E.EVENT_SESSION_STARTED,
        E.EVENT_ASSISTANT_MESSAGE_COMPLETED,
        E.EVENT_USAGE_COMPLETED,
        E.EVENT_SESSION_COMPLETED,
    ]
    assert events[1].text_delta == "hello there"
    assert events[2].usage == {"input_tokens": 10, "output_tokens": 3}


def test_parse_error_session():
    parser = OpenCodeEventParser()
    _feed_file(parser, FIXTURE_DIR / "error_session.json")
    events = parser.finish(0, 1.0)
    kinds = [ev.kind for ev in events]
    assert kinds == [E.EVENT_SESSION_STARTED, E.EVENT_ERROR]
    raw = events[1].raw
    assert isinstance(raw, dict)
    assert raw["error"] == "rate limited by upstream provider"


def test_parse_no_streaming_events_during_feed():
    parser = OpenCodeEventParser()
    text = (FIXTURE_DIR / "simple_session.json").read_text(encoding="utf-8")
    for line in text.splitlines(keepends=True):
        result = parser.feed_stdout_line(line, _ts())
        assert result == []
    events = parser.finish(0, 1.0)
    assert len(events) > 0


def test_parse_malformed_json_emits_parser_error_on_finish():
    parser = OpenCodeEventParser()
    _feed_file(parser, FIXTURE_DIR / "malformed.json")
    events = parser.finish(0, 1.0)
    assert len(events) == 1
    assert events[0].kind == E.EVENT_PARSER_ERROR
    raw = events[0].raw
    assert isinstance(raw, str)
    assert "oc-bad" in raw


def test_parse_empty_finish():
    parser = OpenCodeEventParser()
    events = parser.finish(0, 0.0)
    assert events == []


def test_parse_no_assistant_messages():
    parser = OpenCodeEventParser()
    payload = json.dumps(
        {
            "session_id": "oc-empty",
            "messages": [],
            "usage": {"input_tokens": 5, "output_tokens": 0},
            "error": None,
        }
    )
    parser.feed_stdout_line(payload, 0.0)
    events = parser.finish(0, 1.0)
    kinds = [ev.kind for ev in events]
    assert kinds == [
        E.EVENT_SESSION_STARTED,
        E.EVENT_USAGE_COMPLETED,
        E.EVENT_SESSION_COMPLETED,
    ]


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
    assert cmd == [
        "opencode",
        "run",
        "--format",
        "json",
        "--model",
        "anthropic/claude-3-5-sonnet",
        "--file",
        prompt_path,
        "Complete the AgentSurge session described in the attached file.",
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
        "--format",
        "json",
        "--file",
        prompt_path,
        "Complete the AgentSurge session described in the attached file.",
    ]
