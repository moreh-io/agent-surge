# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
import time
from contextlib import AsyncExitStack
from dataclasses import replace
from pathlib import Path

from agentsurge.frontends import events as E
from agentsurge.frontends.base import (
    FrontendConfig,
    FrontendEventParser,
    FrontendProvider,
    FrontendRunArtifacts,
)
from agentsurge.frontends.claude import ClaudeEventParser, ClaudeProvider
from agentsurge.frontends.codex import CodexEventParser, CodexProvider
from agentsurge.frontends.echo import EchoEventParser, EchoProvider
from agentsurge.frontends.opencode import OpenCodeEventParser, OpenCodeProvider
from agentsurge.proxy.request_shim import RequestShim
from agentsurge.types import (
    BenchmarkConfig,
    FrontendMetrics,
    ReplaySession,
    SessionResult,
    TurnResult,
)

_DEFAULT_MAX_LOG_BYTES = 10 * 1024 * 1024


def _build_frontend_config(cfg: BenchmarkConfig, session_dir: Path) -> FrontendConfig:
    fr = cfg.frontend
    if fr is None:
        return FrontendConfig(
            name="direct",
            command_template=None,
            workspace_dir=session_dir,
            prompt_mode="auto",
            output_format="auto",
            model=None,
            session_timeout_s=7200.0,
            extra_env={},
            server_url=None,
        )
    cmd_template: list[str] | None = None
    if fr.command_template:
        cmd_template = fr.command_template.split()
    return FrontendConfig(
        name=fr.name,
        command_template=cmd_template,
        workspace_dir=session_dir,
        prompt_mode=fr.prompt_mode,
        output_format=fr.output_format,
        model=fr.model,
        session_timeout_s=fr.session_timeout_s,
        extra_env=dict(fr.extra_env),
        server_url=fr.server_url,
    )


class FrontendSessionRenderer:
    MAX_LOG_BYTES = _DEFAULT_MAX_LOG_BYTES

    def __init__(self, config: BenchmarkConfig, run_workspace: Path) -> None:
        self.config = config
        self.run_workspace = Path(run_workspace)

    def _resolve_provider(self, name: str) -> tuple[FrontendProvider, FrontendEventParser]:
        if name == "echo":
            return EchoProvider(), EchoEventParser()
        if name == "codex":
            return CodexProvider(), CodexEventParser()
        if name == "claude":
            return ClaudeProvider(), ClaudeEventParser()
        if name == "opencode":
            return OpenCodeProvider(), OpenCodeEventParser()
        raise NotImplementedError(f"frontend {name!r} not yet wired")

    def _apply_keep_artifacts_policy(self, result: SessionResult, session_dir: Path) -> None:
        """Honor frontend.keep_artifacts: delete session_dir per policy.

        Mutates result.frontend_metrics.artifact_dir to None when the dir is
        actually removed, so downstream consumers don't dereference it.
        """
        fr = self.config.frontend
        # Tests may build the renderer without going through the CLI; default
        # to "always" so direct callers see no surprise deletion.
        policy = fr.keep_artifacts if fr is not None else "always"
        fm = result.frontend_metrics
        if fm is None:
            return
        success = fm.process_exit_code == 0 and fm.failure_category is None
        if policy == "always":
            return
        if policy == "failed" and not success:
            return
        # policy == "never", or policy == "failed" with success
        shutil.rmtree(session_dir, ignore_errors=True)
        fm.artifact_dir = None

    async def run(self, session: ReplaySession, *, session_index: int = 0) -> SessionResult:
        name = self.config.frontend_name
        provider, parser = self._resolve_provider(name)

        session_dir = (self.run_workspace / session.session_id).resolve()
        session_dir.mkdir(parents=True, exist_ok=True)
        artifacts = FrontendRunArtifacts(
            session_dir=session_dir,
            prompt_path=session_dir / "prompt.md",
            session_json_path=session_dir / "session.json",
            stdout_path=session_dir / "stdout.jsonl",
            stderr_path=session_dir / "stderr.log",
            final_message_path=None,
        )
        provider.render(session, artifacts)

        fconfig = _build_frontend_config(self.config, session_dir)

        async with AsyncExitStack() as stack:
            if provider.capabilities.requires_request_rewrite and fconfig.server_url:
                shim_url = await stack.enter_async_context(
                    RequestShim(
                        upstream_base=fconfig.server_url,
                        timeout_s=fconfig.session_timeout_s,
                    )
                )
                fconfig = replace(fconfig, server_url=shim_url)

            cmd = provider.build_command(artifacts, fconfig)
            timeout_s = fconfig.session_timeout_s
            # Provider env contributions (e.g. CODEX_HOME, ANTHROPIC_BASE_URL)
            # take precedence over inherited os.environ so a stale persisted CLI
            # auth doesn't override the configured benchmark target. User's
            # --frontend-extra-env wins last so explicit overrides still work.
            provider_env = provider.build_env(artifacts, fconfig)
            subprocess_env = {
                **os.environ,
                **provider_env,
                **dict(fconfig.extra_env or {}),
            }

            first_event_t: float | None = None
            first_assistant_text_t: float | None = None
            last_assistant_text_t: float | None = None
            final_message_t: float | None = None
            usage_dict: dict[str, int] | None = None
            event_count = 0
            delta_count = 0
            failure_category: str | None = None
            tool_use_observed = False

            stdout_f = artifacts.stdout_path.open("wb")
            try:
                stderr_f = artifacts.stderr_path.open("wb")
                try:
                    process_spawn_t0 = time.monotonic()
                    try:
                        proc = await asyncio.create_subprocess_exec(
                            *cmd,
                            stdin=asyncio.subprocess.DEVNULL,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            cwd=str(session_dir),
                            env=subprocess_env,
                            preexec_fn=os.setsid,
                            # asyncio's StreamReader defaults to a 64 KB line
                            # limit. Real CLIs emit single JSONL events that
                            # easily exceed it (codex echoes its 45 KB merged
                            # instructions back inside thread.started; claude
                            # ships large stream_event blobs). Without this
                            # bump readline() raises "Separator is found, but
                            # chunk is longer than limit" mid-session — 3 of 4
                            # codex sessions failed this way on 2026-04-29.
                            limit=10 * 1024 * 1024,
                        )
                    except (FileNotFoundError, PermissionError) as exc:
                        # Spawn failed before we have a process; surface as a
                        # categorized failure rather than letting the caller see
                        # an unhandled OSError. Keep the message in stderr.log so
                        # operators can diagnose without relying on logs.
                        stderr_f.write(f"spawn_error: {exc}\n".encode())
                        stderr_f.flush()
                        stderr_f.close()
                        stdout_f.close()
                        process_exit_t = time.monotonic()
                        process_wall_ms = (process_exit_t - process_spawn_t0) * 1000.0
                        fm = FrontendMetrics(
                            provider=name,
                            process_exit_code=None,
                            process_signal=None,
                            process_wall_ms=process_wall_ms,
                            process_startup_to_first_event_ms=None,
                            streaming_text_available=False,
                            time_to_first_assistant_text_ms=None,
                            time_to_final_message_ms=None,
                            frontend_ttft_ms=None,
                            visible_text_tpot_estimate_ms=None,
                            visible_output_tokens_estimate=None,
                            provider_usage=None,
                            event_count=0,
                            artifact_dir=str(session_dir),
                            failure_category="spawn_error",
                        )
                        turn = TurnResult(
                            session_id=session.session_id,
                            turn_index=0,
                            completed=False,
                            total_ms=process_wall_ms,
                            output_tokens=0,
                            input_tokens=0,
                            ttft_ms=0.0,
                            wall_ttft_ms=0.0,
                        )
                        spawn_result = SessionResult(
                            session_id=session.session_id,
                            turns=[turn],
                            total_ms=process_wall_ms,
                            expected_turns=1,
                            start_time=0.0,
                            end_time=(process_exit_t - process_spawn_t0),
                            frontend_metrics=fm,
                            metadata={
                                **dict(session.metadata),
                                "frontend": name,
                                "artifact_dir": str(session_dir),
                                "failed": True,
                            },
                        )
                        self._apply_keep_artifacts_policy(spawn_result, session_dir)
                        return spawn_result
                except BaseException:
                    stderr_f.close()
                    raise
            except BaseException:
                stdout_f.close()
                raise

            max_log_bytes = type(self).MAX_LOG_BYTES
            stdout_bytes_written = 0
            stdout_truncated = False
            stderr_bytes_written = 0
            stderr_truncated = False

            async def _consume_stdout() -> None:
                nonlocal first_event_t, first_assistant_text_t, last_assistant_text_t
                nonlocal final_message_t, usage_dict, event_count, delta_count
                nonlocal stdout_bytes_written, stdout_truncated, tool_use_observed
                assert proc.stdout is not None
                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        return
                    if stdout_bytes_written < max_log_bytes:
                        stdout_f.write(line)
                        stdout_f.flush()
                        stdout_bytes_written += len(line)
                        if stdout_bytes_written >= max_log_bytes and not stdout_truncated:
                            stdout_truncated = True
                            marker = (
                                '{"type":"meta.log_truncated","note":"stdout exceeded '
                                f'{max_log_bytes} bytes; further output suppressed"}}\n'
                            ).encode()
                            stdout_f.write(marker)
                            stdout_f.flush()
                    now = time.monotonic()
                    try:
                        decoded = line.decode("utf-8", errors="replace")
                    except Exception:
                        decoded = ""
                    events = parser.feed_stdout_line(decoded, now)
                    for ev in events:
                        event_count += 1
                        if first_event_t is None:
                            first_event_t = now
                        if ev.kind == E.EVENT_ASSISTANT_TEXT_DELTA:
                            delta_count += 1
                            if first_assistant_text_t is None:
                                first_assistant_text_t = now
                            last_assistant_text_t = now
                        elif ev.kind in (
                            E.EVENT_ASSISTANT_MESSAGE_COMPLETED,
                            E.EVENT_SESSION_COMPLETED,
                        ):
                            if final_message_t is None:
                                final_message_t = now
                        elif ev.kind == E.EVENT_USAGE_COMPLETED and ev.usage is not None:
                            usage_dict = dict(ev.usage)
                        elif ev.kind == E.EVENT_TOOL_USE_OBSERVED:
                            tool_use_observed = True

            async def _consume_stderr() -> None:
                nonlocal stderr_bytes_written, stderr_truncated
                assert proc.stderr is not None
                while True:
                    line = await proc.stderr.readline()
                    if not line:
                        return
                    if stderr_bytes_written < max_log_bytes:
                        stderr_f.write(line)
                        stderr_f.flush()
                        stderr_bytes_written += len(line)
                        if stderr_bytes_written >= max_log_bytes and not stderr_truncated:
                            stderr_truncated = True
                            marker = (
                                f"[agentsurge: stderr exceeded {max_log_bytes} "
                                "bytes; further output suppressed]\n"
                            ).encode()
                            stderr_f.write(marker)
                            stderr_f.flush()

            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        _consume_stdout(),
                        _consume_stderr(),
                        proc.wait(),
                    ),
                    timeout=timeout_s,
                )
            except TimeoutError:
                failure_category = "timeout"
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGTERM)
                await asyncio.sleep(0.5)
                if proc.returncode is None:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(proc.pid, signal.SIGKILL)
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
            finally:
                stdout_f.close()
                stderr_f.close()

            process_exit_t = time.monotonic()
            returncode = proc.returncode
            process_signal: int | None = None
            if returncode is not None and returncode < 0:
                process_signal = -returncode

            # Drain any parser-buffered state. Parsers may emit terminal events
            # here (e.g. codex's truncated-line EVENT_PARSER_ERROR); count them
            # in event_count so the metric reflects everything observed.
            finish_events = parser.finish(
                returncode if returncode is not None else 0, process_exit_t
            )
            for ev in finish_events:
                event_count += 1
                if first_event_t is None:
                    first_event_t = ev.ts_monotonic

            if failure_category is None and returncode is not None and returncode != 0:
                failure_category = "nonzero_exit"
            if failure_category is None and tool_use_observed:
                failure_category = "tool_use_observed"

            process_wall_ms = (process_exit_t - process_spawn_t0) * 1000.0

            startup_to_first_event_ms: float | None = None
            if first_event_t is not None:
                startup_to_first_event_ms = (first_event_t - process_spawn_t0) * 1000.0

            ttfat_ms: float | None = None
            if first_assistant_text_t is not None:
                ttfat_ms = (first_assistant_text_t - process_spawn_t0) * 1000.0

            ttfm_ms: float | None = None
            if final_message_t is not None:
                ttfm_ms = (final_message_t - process_spawn_t0) * 1000.0

            streaming_text_available = first_assistant_text_t is not None

            # Heuristic: 1 delta == 1 visible token. Coarse for echo; canonical
            # parsers should compute tokens from actual text where feasible.
            visible_tokens = delta_count if delta_count > 0 else None

            tpot_ms: float | None = None
            if (
                streaming_text_available
                and last_assistant_text_t is not None
                and first_assistant_text_t is not None
                and last_assistant_text_t > first_assistant_text_t
                and delta_count >= 2
            ):
                denom = max(delta_count - 1, 1)
                tpot_ms = (last_assistant_text_t - first_assistant_text_t) * 1000.0 / denom

            frontend_metrics = FrontendMetrics(
                provider=name,
                process_exit_code=returncode,
                process_signal=process_signal,
                process_wall_ms=process_wall_ms,
                process_startup_to_first_event_ms=startup_to_first_event_ms,
                streaming_text_available=streaming_text_available,
                time_to_first_assistant_text_ms=ttfat_ms,
                time_to_final_message_ms=ttfm_ms,
                frontend_ttft_ms=ttfat_ms if streaming_text_available else None,
                visible_text_tpot_estimate_ms=tpot_ms,
                visible_output_tokens_estimate=visible_tokens,
                provider_usage=usage_dict,
                event_count=event_count,
                artifact_dir=str(session_dir),
                failure_category=failure_category,
            )

            completed = returncode == 0 and failure_category is None
            input_tokens = 0
            output_tokens = visible_tokens or 0
            if usage_dict is not None:
                input_tokens = int(usage_dict.get("input_tokens", 0))
                ut = usage_dict.get("output_tokens")
                if isinstance(ut, int) and ut > 0:
                    output_tokens = ut

            turn = TurnResult(
                session_id=session.session_id,
                turn_index=0,
                completed=completed,
                total_ms=process_wall_ms,
                output_tokens=output_tokens,
                input_tokens=input_tokens,
                ttft_ms=ttfat_ms or 0.0,
                wall_ttft_ms=ttfat_ms or 0.0,
            )

            result = SessionResult(
                session_id=session.session_id,
                turns=[turn],
                total_ms=process_wall_ms,
                expected_turns=1,
                start_time=0.0,
                end_time=(process_exit_t - process_spawn_t0),
                frontend_metrics=frontend_metrics,
                metadata={
                    **dict(session.metadata),
                    "frontend": name,
                    "artifact_dir": str(session_dir),
                },
            )
            self._apply_keep_artifacts_policy(result, session_dir)
            return result
