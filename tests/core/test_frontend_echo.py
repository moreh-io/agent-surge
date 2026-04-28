"""Echo frontend provider, parser, and FrontendSessionRenderer tests."""

from __future__ import annotations

import asyncio
import json
import signal
import time
from pathlib import Path

import pytest

from agentsurge.frontends import events as E
from agentsurge.frontends.echo import EchoEventParser, EchoProvider
from agentsurge.frontends.runner import FrontendSessionRenderer
from agentsurge.runner import BenchmarkRunner
from agentsurge.types import (
    BenchmarkConfig,
    FrontendRuntimeSettings,
    ReplaySession,
)


def _ts() -> float:
    return time.monotonic()


def test_echo_parser_canonical_events():
    parser = EchoEventParser()
    lines_and_kinds = [
        ('{"type":"session.started"}\n', E.EVENT_SESSION_STARTED),
        ('{"type":"cli.process.started"}\n', E.EVENT_CLI_PROCESS_STARTED),
        (
            '{"type":"assistant.text.delta","delta":"chunk-1"}\n',
            E.EVENT_ASSISTANT_TEXT_DELTA,
        ),
        (
            '{"type":"assistant.message.completed","text":"chunk-1chunk-2"}\n',
            E.EVENT_ASSISTANT_MESSAGE_COMPLETED,
        ),
        (
            '{"type":"usage.completed","usage":{"input_tokens":10,"output_tokens":5}}\n',
            E.EVENT_USAGE_COMPLETED,
        ),
        ('{"type":"session.completed","exit_code":0}\n', E.EVENT_SESSION_COMPLETED),
    ]
    seen_kinds: list[str] = []
    for line, _expected in lines_and_kinds:
        events = parser.feed_stdout_line(line, _ts())
        assert len(events) == 1
        seen_kinds.append(events[0].kind)
    assert seen_kinds == [k for _, k in lines_and_kinds]

    delta_events = parser.feed_stdout_line(
        '{"type":"assistant.text.delta","delta":"hello"}\n', _ts()
    )
    assert delta_events[0].text_delta == "hello"

    usage_events = parser.feed_stdout_line(
        '{"type":"usage.completed","usage":{"input_tokens":3,"output_tokens":4}}\n',
        _ts(),
    )
    assert usage_events[0].usage == {"input_tokens": 3, "output_tokens": 4}


def test_echo_parser_malformed_does_not_raise():
    parser = EchoEventParser()
    events = parser.feed_stdout_line("not json\n", _ts())
    assert len(events) == 1
    assert events[0].kind == E.EVENT_PARSER_ERROR


def test_echo_parser_unknown_kind():
    parser = EchoEventParser()
    events = parser.feed_stdout_line('{"type":"weird"}\n', _ts())
    assert len(events) == 1
    assert events[0].kind == E.EVENT_PARSER_UNKNOWN


def _make_session(sid: str = "s0") -> ReplaySession:
    return ReplaySession(
        session_id=sid,
        turn_messages=[[{"role": "user", "content": "hi"}]],
        metadata={},
    )


@pytest.mark.asyncio
async def test_frontend_session_renderer_end_to_end(tmp_path: Path):
    cfg = BenchmarkConfig(
        vllm_url="http://x",
        model="t",
        no_metrics=True,
        frontend=FrontendRuntimeSettings(name="echo", session_timeout_s=10.0),
    )
    renderer = FrontendSessionRenderer(cfg, tmp_path)
    session = _make_session("s_e2e")
    result = await renderer.run(session)

    assert result.session_id == "s_e2e"
    assert len(result.turns) == 1
    fm = result.frontend_metrics
    assert fm is not None
    assert fm.streaming_text_available is True
    assert fm.event_count == 8
    assert fm.process_exit_code == 0
    assert fm.failure_category is None
    assert fm.time_to_first_assistant_text_ms is not None
    assert fm.time_to_final_message_ms is not None
    assert fm.time_to_first_assistant_text_ms < fm.time_to_final_message_ms
    assert fm.visible_output_tokens_estimate == 3
    assert fm.provider_usage == {"input_tokens": 10, "output_tokens": 5}
    assert fm.process_wall_ms is not None and fm.process_wall_ms > 0

    sd = tmp_path / "s_e2e"
    assert (sd / "prompt.md").exists()
    assert (sd / "session.json").exists()
    assert (sd / "stdout.jsonl").exists()
    assert (sd / "stderr.log").exists()

    # The raw stdout.jsonl emits provider-specific "type" strings; map to
    # canonical kinds and assert the full ordered sequence.
    type_to_kind = {
        "session.started": E.EVENT_SESSION_STARTED,
        "cli.process.started": E.EVENT_CLI_PROCESS_STARTED,
        "assistant.text.delta": E.EVENT_ASSISTANT_TEXT_DELTA,
        "assistant.message.completed": E.EVENT_ASSISTANT_MESSAGE_COMPLETED,
        "usage.completed": E.EVENT_USAGE_COMPLETED,
        "session.completed": E.EVENT_SESSION_COMPLETED,
    }
    raw_kinds: list[str] = []
    for line in (sd / "stdout.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        raw_kinds.append(type_to_kind[obj["type"]])
    assert raw_kinds == [
        E.EVENT_SESSION_STARTED,
        E.EVENT_CLI_PROCESS_STARTED,
        E.EVENT_ASSISTANT_TEXT_DELTA,
        E.EVENT_ASSISTANT_TEXT_DELTA,
        E.EVENT_ASSISTANT_TEXT_DELTA,
        E.EVENT_ASSISTANT_MESSAGE_COMPLETED,
        E.EVENT_USAGE_COMPLETED,
        E.EVENT_SESSION_COMPLETED,
    ]
    assert "hi" in (sd / "prompt.md").read_text()


@pytest.mark.asyncio
async def test_frontend_session_renderer_timeout(tmp_path: Path, monkeypatch):
    """Slow-mode echo subprocess sleeps 60s; renderer must abort within timeout."""
    cfg = BenchmarkConfig(
        vllm_url="http://x",
        model="t",
        no_metrics=True,
        frontend=FrontendRuntimeSettings(name="echo", session_timeout_s=0.5),
    )
    original_build = EchoProvider.build_command

    def slow_build(self, artifacts, config):
        cmd = original_build(self, artifacts, config)
        return cmd + ["--mode", "slow"]

    monkeypatch.setattr(EchoProvider, "build_command", slow_build)

    renderer = FrontendSessionRenderer(cfg, tmp_path)
    session = _make_session("s_timeout")
    t0 = time.monotonic()
    result = await renderer.run(session)
    elapsed = time.monotonic() - t0

    # Renderer waits 0.5s after SIGTERM before SIGKILL, then up to 2s for wait,
    # so 3.0s is the safe ceiling on shared CI; tightening below would flake.
    assert elapsed < 3.0
    fm = result.frontend_metrics
    assert fm is not None
    assert fm.failure_category == "timeout"
    # Process must actually have died via signal (negative returncode → signal).
    assert fm.process_signal in (signal.SIGTERM.value, signal.SIGKILL.value)


@pytest.mark.asyncio
async def test_extra_env_reaches_subprocess(tmp_path: Path):
    """FrontendRuntimeSettings.extra_env must propagate into the subprocess."""
    cfg = BenchmarkConfig(
        vllm_url="http://x",
        model="t",
        no_metrics=True,
        frontend=FrontendRuntimeSettings(
            name="echo",
            session_timeout_s=10.0,
            extra_env=(("AGENTSURGE_TEST_ENV_PROBE", "secret-marker-42"),),
        ),
    )
    renderer = FrontendSessionRenderer(cfg, tmp_path / "ws")
    session = _make_session("env_check")
    await renderer.run(session)

    stdout_jsonl = (tmp_path / "ws" / "env_check" / "stdout.jsonl").read_text()
    assert "secret-marker-42" in stdout_jsonl


@pytest.mark.asyncio
async def test_runner_dispatches_to_echo(tmp_path: Path):
    cfg = BenchmarkConfig(
        vllm_url="http://x",
        model="t",
        no_metrics=True,
        max_concurrency=2,
        frontend=FrontendRuntimeSettings(
            name="echo",
            session_timeout_s=10.0,
            workspace_dir=str(tmp_path / "ws"),
        ),
    )
    runner = BenchmarkRunner(cfg)

    sessions = [_make_session("a"), _make_session("b")]
    semaphore = asyncio.Semaphore(2)
    results = await runner._dispatch_sessions(None, semaphore, sessions, [0.0, 0.0])

    assert len(results) == 2
    for r in results:
        assert r.frontend_metrics is not None
        assert r.frontend_metrics.process_exit_code == 0
        assert r.frontend_metrics.streaming_text_available is True

    # Each session must have a distinct artifact dir; collapsing both to the
    # same dir would clobber prompt/stdout files but still pass the loop above.
    artifact_dirs = {r.frontend_metrics.artifact_dir for r in results}
    assert len(artifact_dirs) == 2
    for path in artifact_dirs:
        assert path is not None
        assert (Path(path) / "prompt.md").exists()
