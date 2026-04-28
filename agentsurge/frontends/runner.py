# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import time
from pathlib import Path

from agentsurge.frontends import events as E
from agentsurge.frontends.base import (
    FrontendConfig,
    FrontendEventParser,
    FrontendProvider,
    FrontendRunArtifacts,
)
from agentsurge.frontends.echo import EchoEventParser, EchoProvider
from agentsurge.types import (
    BenchmarkConfig,
    FrontendMetrics,
    ReplaySession,
    SessionResult,
    TurnResult,
)


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
    )


class FrontendSessionRenderer:
    def __init__(self, config: BenchmarkConfig, run_workspace: Path) -> None:
        self.config = config
        self.run_workspace = Path(run_workspace)

    def _resolve_provider(self, name: str) -> tuple[FrontendProvider, FrontendEventParser]:
        if name == "echo":
            return EchoProvider(), EchoEventParser()
        raise NotImplementedError(f"frontend {name!r} not yet wired")

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
        cmd = provider.build_command(artifacts, fconfig)
        timeout_s = fconfig.session_timeout_s

        first_event_t: float | None = None
        first_assistant_text_t: float | None = None
        last_assistant_text_t: float | None = None
        final_message_t: float | None = None
        usage_dict: dict[str, int] | None = None
        event_count = 0
        delta_count = 0
        failure_category: str | None = None

        stdout_f = artifacts.stdout_path.open("wb")
        try:
            stderr_f = artifacts.stderr_path.open("wb")
            try:
                process_spawn_t0 = time.monotonic()
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(session_dir),
                    preexec_fn=os.setsid,
                )
            except BaseException:
                stderr_f.close()
                raise
        except BaseException:
            stdout_f.close()
            raise

        async def _consume_stdout() -> None:
            nonlocal first_event_t, first_assistant_text_t, last_assistant_text_t
            nonlocal final_message_t, usage_dict, event_count, delta_count
            assert proc.stdout is not None
            while True:
                line = await proc.stdout.readline()
                if not line:
                    return
                stdout_f.write(line)
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

        async def _consume_stderr() -> None:
            assert proc.stderr is not None
            while True:
                line = await proc.stderr.readline()
                if not line:
                    return
                stderr_f.write(line)
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

        return SessionResult(
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
