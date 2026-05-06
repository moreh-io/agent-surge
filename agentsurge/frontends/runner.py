# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import IO

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


@dataclass(slots=True)
class _IoResult:
    first_event_t: float | None = None
    first_assistant_text_t: float | None = None
    last_assistant_text_t: float | None = None
    final_message_t: float | None = None
    usage_dict: dict[str, int] | None = None
    event_count: int = 0
    delta_count: int = 0
    tool_use_observed: bool = False
    failure_category: str | None = None
    stdout_bytes_written: int = 0
    stdout_truncated: bool = False
    stderr_bytes_written: int = 0
    stderr_truncated: bool = False


class _SpawnFailed(Exception):
    """Internal-only marker so run() can branch to _handle_spawn_error."""

    def __init__(self, cause: BaseException) -> None:
        super().__init__(str(cause))
        self.cause = cause


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

    @staticmethod
    def _should_delete_artifacts(policy: str, fm: FrontendMetrics) -> bool:
        success = fm.process_exit_code == 0 and fm.failure_category is None
        if policy == "always":
            return False
        return not (policy == "failed" and not success)

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
        if self._should_delete_artifacts(policy, fm):
            # policy == "never", or policy == "failed" with success
            shutil.rmtree(session_dir, ignore_errors=True)
            fm.artifact_dir = None

    def _setup_session_dir(self, session: ReplaySession) -> tuple[Path, FrontendRunArtifacts]:
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
        return session_dir, artifacts

    def _build_subprocess_env(
        self,
        provider: FrontendProvider,
        fconfig: FrontendConfig,
        artifacts: FrontendRunArtifacts,
    ) -> dict[str, str]:
        provider_env = provider.build_env(artifacts, fconfig)
        return {
            **os.environ,
            **provider_env,
            **dict(fconfig.extra_env or {}),
        }

    async def _spawn(
        self,
        cmd: list[str],
        subprocess_env: dict[str, str],
        artifacts: FrontendRunArtifacts,
    ) -> tuple[asyncio.subprocess.Process, IO[bytes], IO[bytes]]:
        stdout_f: IO[bytes] = artifacts.stdout_path.open("wb")
        try:
            stderr_f: IO[bytes] = artifacts.stderr_path.open("wb")
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(artifacts.session_dir),
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
                stderr_f.close()
                stdout_f.close()
                raise _SpawnFailed(exc) from exc
            except BaseException:
                stderr_f.close()
                raise
        except BaseException:
            if not stdout_f.closed:
                stdout_f.close()
            raise
        return proc, stdout_f, stderr_f

    def _handle_spawn_error(
        self,
        name: str,
        exc: BaseException,
        session: ReplaySession,
        session_dir: Path,
        process_spawn_t0: float,
    ) -> SessionResult:
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

    async def _consume_io(
        self,
        proc: asyncio.subprocess.Process,
        parser: FrontendEventParser,
        stdout_f: IO[bytes],
        stderr_f: IO[bytes],
        timeout_s: float,
    ) -> _IoResult:
        io = _IoResult()
        max_log_bytes = type(self).MAX_LOG_BYTES

        async def _consume_stdout() -> None:
            assert proc.stdout is not None
            while True:
                line = await proc.stdout.readline()
                if not line:
                    return
                if io.stdout_bytes_written < max_log_bytes:
                    stdout_f.write(line)
                    stdout_f.flush()
                    io.stdout_bytes_written += len(line)
                    if io.stdout_bytes_written >= max_log_bytes and not io.stdout_truncated:
                        io.stdout_truncated = True
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
                    io.event_count += 1
                    if io.first_event_t is None:
                        io.first_event_t = now
                    if ev.kind == E.EVENT_ASSISTANT_TEXT_DELTA:
                        io.delta_count += 1
                        if io.first_assistant_text_t is None:
                            io.first_assistant_text_t = now
                        io.last_assistant_text_t = now
                    elif ev.kind in (
                        E.EVENT_ASSISTANT_MESSAGE_COMPLETED,
                        E.EVENT_SESSION_COMPLETED,
                    ):
                        if io.final_message_t is None:
                            io.final_message_t = now
                    elif ev.kind == E.EVENT_USAGE_COMPLETED and ev.usage is not None:
                        io.usage_dict = dict(ev.usage)
                    elif ev.kind == E.EVENT_TOOL_USE_OBSERVED:
                        io.tool_use_observed = True

        async def _consume_stderr() -> None:
            assert proc.stderr is not None
            while True:
                line = await proc.stderr.readline()
                if not line:
                    return
                if io.stderr_bytes_written < max_log_bytes:
                    stderr_f.write(line)
                    stderr_f.flush()
                    io.stderr_bytes_written += len(line)
                    if io.stderr_bytes_written >= max_log_bytes and not io.stderr_truncated:
                        io.stderr_truncated = True
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
            io.failure_category = "timeout"
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
            await asyncio.sleep(0.5)
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(proc.wait(), timeout=2.0)

        return io

    def _drain_parser_finish(
        self,
        parser: FrontendEventParser,
        returncode: int | None,
        process_exit_t: float,
        io: _IoResult,
    ) -> None:
        finish_events = parser.finish(returncode if returncode is not None else 0, process_exit_t)
        for ev in finish_events:
            io.event_count += 1
            if io.first_event_t is None:
                io.first_event_t = ev.ts_monotonic

    @staticmethod
    def _compute_tpot_ms(io: _IoResult) -> float | None:
        if (
            io.first_assistant_text_t is not None
            and io.last_assistant_text_t is not None
            and io.last_assistant_text_t > io.first_assistant_text_t
            and io.delta_count >= 2
        ):
            denom = max(io.delta_count - 1, 1)
            return (io.last_assistant_text_t - io.first_assistant_text_t) * 1000.0 / denom
        return None

    @staticmethod
    def _ms_since(t: float | None, origin: float) -> float | None:
        return (t - origin) * 1000.0 if t is not None else None

    def _assemble_metrics(
        self,
        name: str,
        returncode: int | None,
        process_signal: int | None,
        process_spawn_t0: float,
        process_exit_t: float,
        io: _IoResult,
        session_dir: Path,
    ) -> FrontendMetrics:
        process_wall_ms = (process_exit_t - process_spawn_t0) * 1000.0
        startup_to_first_event_ms = self._ms_since(io.first_event_t, process_spawn_t0)
        ttfat_ms = self._ms_since(io.first_assistant_text_t, process_spawn_t0)
        ttfm_ms = self._ms_since(io.final_message_t, process_spawn_t0)
        streaming_text_available = io.first_assistant_text_t is not None
        visible_tokens = io.delta_count if io.delta_count > 0 else None
        tpot_ms = self._compute_tpot_ms(io)

        return FrontendMetrics(
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
            provider_usage=io.usage_dict,
            event_count=io.event_count,
            artifact_dir=str(session_dir),
            failure_category=io.failure_category,
        )

    @staticmethod
    def _resolve_token_counts(io: _IoResult, fm: FrontendMetrics) -> tuple[int, int]:
        input_tokens = 0
        output_tokens = fm.visible_output_tokens_estimate or 0
        if io.usage_dict is not None:
            input_tokens = int(io.usage_dict.get("input_tokens", 0))
            ut = io.usage_dict.get("output_tokens")
            if isinstance(ut, int) and ut > 0:
                output_tokens = ut
        return input_tokens, output_tokens

    def _assemble_session_result(
        self,
        session: ReplaySession,
        name: str,
        fm: FrontendMetrics,
        io: _IoResult,
        returncode: int | None,
        process_wall_ms: float,
        process_exit_t: float,
        process_spawn_t0: float,
        session_dir: Path,
    ) -> SessionResult:
        completed = returncode == 0 and io.failure_category is None
        input_tokens, output_tokens = self._resolve_token_counts(io, fm)
        ttfat_ms = fm.time_to_first_assistant_text_ms
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

        return SessionResult(
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
            },
        )

    @staticmethod
    def _finalize_failure_category(io: _IoResult, returncode: int | None) -> None:
        if io.failure_category is None and returncode not in (0, None):
            io.failure_category = "nonzero_exit"
        if io.failure_category is None and io.tool_use_observed:
            io.failure_category = "tool_use_observed"

    async def _run_subprocess(
        self,
        session: ReplaySession,
        name: str,
        provider: FrontendProvider,
        parser: FrontendEventParser,
        artifacts: FrontendRunArtifacts,
        fconfig: FrontendConfig,
        session_dir: Path,
    ) -> SessionResult:
        cmd = provider.build_command(artifacts, fconfig)
        subprocess_env = self._build_subprocess_env(provider, fconfig, artifacts)
        process_spawn_t0 = time.monotonic()
        try:
            proc, stdout_f, stderr_f = await self._spawn(cmd, subprocess_env, artifacts)
        except _SpawnFailed as e:
            return self._handle_spawn_error(name, e.cause, session, session_dir, process_spawn_t0)

        try:
            io = await self._consume_io(proc, parser, stdout_f, stderr_f, fconfig.session_timeout_s)
        finally:
            stdout_f.close()
            stderr_f.close()

        process_exit_t = time.monotonic()
        self._drain_parser_finish(parser, proc.returncode, process_exit_t, io)
        process_signal = (
            -proc.returncode if proc.returncode is not None and proc.returncode < 0 else None
        )
        self._finalize_failure_category(io, proc.returncode)
        process_wall_ms = (process_exit_t - process_spawn_t0) * 1000.0
        fm = self._assemble_metrics(
            name, proc.returncode, process_signal, process_spawn_t0, process_exit_t, io, session_dir
        )
        result = self._assemble_session_result(
            session,
            name,
            fm,
            io,
            proc.returncode,
            process_wall_ms,
            process_exit_t,
            process_spawn_t0,
            session_dir,
        )
        self._apply_keep_artifacts_policy(result, session_dir)
        return result

    async def run(self, session: ReplaySession, *, session_index: int = 0) -> SessionResult:
        name = self.config.frontend_name
        provider, parser = self._resolve_provider(name)
        session_dir, artifacts = self._setup_session_dir(session)
        provider.render(session, artifacts)
        fconfig = _build_frontend_config(self.config, session_dir)

        async with AsyncExitStack() as stack:
            if provider.capabilities.requires_request_rewrite and fconfig.server_url:
                shim_url = await stack.enter_async_context(
                    RequestShim(
                        upstream_base=fconfig.server_url, timeout_s=fconfig.session_timeout_s
                    )
                )
                fconfig = replace(fconfig, server_url=shim_url)

            return await self._run_subprocess(
                session, name, provider, parser, artifacts, fconfig, session_dir
            )
