# SPDX-License-Identifier: MIT
"""cmd_run - benchmark run subcommand."""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import cast

from agentsurge._inflight_dump import InflightDumpWriter
from agentsurge.backends import BackendBase, BackendConfig, get_backend
from agentsurge.cli._helpers import (
    _build_runner_config,
    _load_config,
    _output_path,
    _validate_numeric_args,
)
from agentsurge.generators.synthetic import SyntheticGenerator
from agentsurge.io import load_workload_sessions, validate_workload
from agentsurge.runner import BenchmarkRunner


def _start_inflight_writer(config, base_path: str) -> InflightDumpWriter | None:
    if not getattr(config, "inflight_dump", False):
        return None
    w = InflightDumpWriter(str(Path(base_path).with_suffix("")) + "_turns.jsonl")
    w.start()
    return w


_log = logging.getLogger(__name__)


def _backend_preflight(url: str, timeout: float = 5.0) -> bool:
    """Quick liveness check for the inference backend before runner.run().

    cli.md M5: prior to this, an unreachable URL produced ConnectionRefused
    mid-run after warmup tasks were already spawned. We hit /v1/models then
    fall back to the bare URL; either 2xx/3xx/4xx response (i.e. the host
    answered) is treated as reachable. 5xx and connection errors fail.
    """
    import urllib.error
    import urllib.request

    if not url:
        return False
    candidates = [url.rstrip("/") + "/v1/models", url]
    for candidate in candidates:
        try:
            req = urllib.request.Request(candidate, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return bool(200 <= resp.status < 500)
        except urllib.error.HTTPError as e:
            # Host answered with an error status — still reachable.
            if 200 <= e.code < 500:
                return True
            continue
        except (urllib.error.URLError, TimeoutError, OSError):
            continue
    return False


def _build_backend(args: argparse.Namespace, config) -> BackendBase | None:
    """Construct a backend instance from --backend, or None for the default HTTP path."""
    backend_name = getattr(args, "backend", None)
    if not backend_name:
        return None
    backend_cls = get_backend(backend_name)
    if backend_name == "mock":
        from agentsurge.backends.mock import MockConfig

        return backend_cls(MockConfig())
    backend_kwargs = dict(
        base_url=config.vllm_url,
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        request_timeout=config.request_timeout,
        extra_body=config.extra_body,
        stream_idle_timeout=config.stream_idle_timeout,
    )
    if backend_name == "vllm":
        from agentsurge.backends.vllm import VllmConfig

        chat_template_kwargs = {"enable_thinking": True} if config.enable_thinking else None
        return backend_cls(
            VllmConfig(
                **backend_kwargs,
                api_type=config.api_type,
                chat_template_kwargs=chat_template_kwargs,
            )
        )
    return backend_cls(
        BackendConfig(
            **backend_kwargs,
        )
    )


def _dataset_config_name(args: argparse.Namespace) -> str:
    source = getattr(args, "source", None)
    if source:
        return str(source)
    workload_file = getattr(args, "workload_file", None)
    if workload_file:
        return Path(workload_file).stem or "workload-file"
    return "synthetic"


def cmd_run(args: argparse.Namespace) -> None:
    """Run benchmark against vLLM endpoint."""

    _validate_numeric_args(args)

    if getattr(args, "no_stream", False):
        # cli.md M4: --no-stream collapses TTFT == E2E and TPOT goes to ~0;
        # warn loudly so cross-run aggregations don't average meaningless TPOT.
        print(
            "warning: --no-stream is set: non-streaming mode disables TPOT measurement "
            "(TTFT will equal E2E; TPOT ~ 0)",
            file=sys.stderr,
        )

    if getattr(args, "workload_file", None) and getattr(args, "trace_pool", None):
        print("Error: --workload-file and --trace-pool are mutually exclusive", file=sys.stderr)
        raise SystemExit(1)

    cfg = _load_config(args.config)

    # cli.md M5: preflight check the backend URL once cheap structural
    # validation has passed. Skipped for mock backend; skipped if the
    # caller hasn't supplied a URL (will fail later in _build_runner_config
    # via CLIConfigError).
    def _maybe_preflight() -> None:
        backend_name = getattr(args, "backend", None)
        if backend_name in ("mock",):
            return
        url = getattr(args, "vllm_url", None) or (cfg.get("vllm") or {}).get("url")
        if not url:
            return
        if not _backend_preflight(url, timeout=5.0):
            print(
                f"error: backend at {url} is unreachable (preflight failed). "
                f"Check --vllm-url / vllm.url in config.",
                file=sys.stderr,
            )
            raise SystemExit(3)

    # ── Tool-calling mode ──
    tool_mode = getattr(args, "tool_mode", None)
    if tool_mode and tool_mode != "off":
        from agentsurge.tool_call import SANITY_TASKS, SYSTEM_PROMPT
        from agentsurge.types import ReplaySession

        tool_max_turns = getattr(args, "tool_max_turns", None) or 15

        def _ensure_tool_mode_max_turns(sessions: list[ReplaySession]) -> None:
            for s in sessions:
                if "max_turns" not in s.metadata:
                    s.metadata["max_turns"] = tool_max_turns

        n_sessions = args.n_sessions

        # --workload-file: use workload sessions as tool-mode prompts
        if getattr(args, "workload_file", None):
            sessions = load_workload_sessions(args.workload_file)
            _ensure_tool_mode_max_turns(sessions)
            print(f"Using workload sessions as tool-mode prompts: {len(sessions)} sessions")
        elif getattr(args, "source", None):
            _REAL_MODE_UNSUPPORTED = {"featurebench", "swe-evo", "mixed"}
            if tool_mode == "real" and args.source in _REAL_MODE_UNSUPPORTED:
                print(
                    f"Error: --source {args.source} does not support --tool-mode real "
                    f"(no repo metadata for workspace setup). "
                    f"Use --tool-mode replay or --tool-mode off.",
                    file=sys.stderr,
                )
                raise SystemExit(1)

            from agentsurge.cli._helpers import _generate_sessions

            sessions = _generate_sessions(
                source=args.source,
                n_sessions=n_sessions,
                n_turns=args.n_turns or cfg["workload"].get("n_turns", 5),
                synthetic_tokens_per_turn=args.synthetic_tokens_per_turn
                or cfg["workload"].get("synthetic_tokens_per_turn", 2700),
                local_path=getattr(args, "local_path", None),
                data_dir=getattr(args, "data_dir", None),
                flatten_tools=(tool_mode not in ("real", "replay")),
                multi_turn_only=getattr(args, "multi_turn_only", False),
                prefix_overlap_fraction=getattr(args, "prefix_overlap_fraction", 0.0),
                trace_pool_path=getattr(args, "trace_pool", None),
            )
            _ensure_tool_mode_max_turns(sessions)
        else:
            sessions = []
            for i in range(n_sessions):
                task_text = SANITY_TASKS[i % len(SANITY_TASKS)]
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": task_text},
                ]
                sessions.append(
                    ReplaySession(
                        session_id=f"tool_{i:03d}",
                        turn_messages=[messages],
                        metadata={"max_turns": tool_max_turns},
                    )
                )

        config = _build_runner_config(args, cfg)
        backend = _build_backend(args, config)
        out = _output_path(args.output_dir, f"run_tool_{tool_mode}")
        inflight = _start_inflight_writer(config, out)
        runner = BenchmarkRunner(
            config, backend=backend, on_turn=(inflight.on_turn if inflight else None)
        )

        mode_label = f"tool-mode={tool_mode}"
        print(
            f"Running {len(sessions)} sessions ({mode_label}, "
            f"concurrency={config.max_concurrency})..."
        )
        _maybe_preflight()
        try:
            result = asyncio.run(runner.run(sessions))
        except BaseException as exc:
            # cli.md H2: SIGTERM/SIGINT-safe partial marker.
            try:
                with open(out, "w") as f:
                    json.dump(
                        {
                            "partial": True,
                            "reason": type(exc).__name__,
                            "n_sessions_started": len(sessions),
                        },
                        f,
                        indent=2,
                        default=str,
                    )
                _log.warning("partial results marker written to %s on signal", out)
            except Exception:  # noqa: BLE001
                _log.exception("failed to flush partial-results marker on signal")
            if inflight is not None:
                inflight.stop()
            raise
        finally:
            if inflight is not None:
                inflight.stop()

        with open(out, "w") as f:
            json.dump(
                result.to_dict(save_responses=config.save_responses), f, indent=2, default=str
            )

        saved_files = [out]

        if getattr(args, "make_dataset", False):
            from agentsurge.hf import run_result_to_parquet, workload_to_parquet

            hf_dir = out.replace(".json", "_hf")
            model_name = args.model or config.model
            config_name = _dataset_config_name(args)
            run_result_to_parquet(
                result,
                os.path.join(hf_dir, "results"),
                model_name=model_name,
                config_name=config_name,
            )
            workload_to_parquet(
                sessions,
                os.path.join(hf_dir, "workload"),
                model_name=model_name,
                config_name=config_name,
            )
            saved_files.append(hf_dir + "/")

        from agentsurge.cli._display import print_run_summary

        model_name = args.model or config.model
        print_run_summary(result, model=os.path.basename(model_name), saved_files=saved_files)

        return

    # ── Replay mode (default) ──
    if args.workload_file:
        if args.workload_file.startswith("hf://"):
            from agentsurge.hf import download_workload_from_hf

            sessions = download_workload_from_hf(args.workload_file)
        elif not os.path.exists(args.workload_file):
            _log.error("workload file not found: %s", args.workload_file)
            sys.exit(1)
        else:
            if not getattr(args, "skip_validation", False):
                with open(args.workload_file) as _f:
                    _raw = json.load(_f)
                validate_workload(_raw)
            sessions = load_workload_sessions(args.workload_file)
    elif getattr(args, "source", None):
        from agentsurge.cli._helpers import _generate_sessions

        sessions = _generate_sessions(
            source=args.source,
            n_sessions=args.n_sessions,
            n_turns=args.n_turns or cfg["workload"].get("n_turns", 5),
            synthetic_tokens_per_turn=args.synthetic_tokens_per_turn
            or cfg["workload"].get("synthetic_tokens_per_turn", 2700),
            local_path=getattr(args, "local_path", None),
            data_dir=getattr(args, "data_dir", None),
            prefix_overlap_fraction=getattr(args, "prefix_overlap_fraction", 0.0),
            trace_pool_path=getattr(args, "trace_pool", None),
            multi_turn_only=getattr(args, "multi_turn_only", False),
        )
        print(f"Generated {len(sessions)} sessions from source={args.source}")
    else:
        gen = SyntheticGenerator(
            tokens_per_turn=args.synthetic_tokens_per_turn
            or cfg["workload"].get("synthetic_tokens_per_turn", 2700),
            n_turns=args.n_turns or cfg["workload"].get("n_turns", 5),
            prefix_overlap_fraction=getattr(args, "prefix_overlap_fraction", 0.0) or 0.0,
        )
        sessions = gen.generate(n_sessions=args.n_sessions)

    config = _build_runner_config(args, cfg)
    backend = _build_backend(args, config)
    base_path = _output_path(args.output_dir, "run")
    inflight = _start_inflight_writer(config, base_path)
    runner = BenchmarkRunner(
        config, backend=backend, on_turn=(inflight.on_turn if inflight else None)
    )

    preset_label = f", preset={args.preset}" if getattr(args, "preset", None) else ""
    print(
        f"Running {len(sessions)} sessions (concurrency={config.max_concurrency}, "
        f"arrival={config.arrival_pattern}{preset_label})..."
    )
    # Echo the resolved BenchmarkConfig so users can see what the preset +
    # YAML + CLI merge actually produced, without having to re-read their
    # flags after the run lands.
    print(
        "Effective config: "
        f"model={config.model!r} "
        f"tool_mode={config.tool_mode} "
        f"arrival={config.arrival_pattern} "
        f"arrival_rate={config.arrival_rate} "
        f"max_tokens={config.max_tokens} "
        f"max_model_len={config.max_model_len} "
        f"temperature={config.temperature} "
        f"save_responses={config.save_responses} "
        f"output={base_path}"
    )
    _maybe_preflight()
    # cli.md H2: on SIGTERM / SIGINT, flush a partial-marker JSON so callers
    # (k8s eviction, slurm preempt) get a record instead of an empty
    # output-dir. The marker is enough for the partial-result contract;
    # cmd_run also writes the JSON normally on the success path below.
    try:
        result = asyncio.run(runner.run(sessions))
    except BaseException as exc:
        try:
            with open(base_path, "w") as f:
                json.dump(
                    {
                        "partial": True,
                        "reason": type(exc).__name__,
                        "n_sessions_started": len(sessions),
                    },
                    f,
                    indent=2,
                    default=str,
                )
            _log.warning("partial results marker written to %s on signal", base_path)
        except Exception:  # noqa: BLE001
            _log.exception("failed to flush partial-results marker on signal")
        if inflight is not None:
            inflight.stop()
        raise
    finally:
        if inflight is not None:
            inflight.stop()

    _sessions = cast(list, result.sessions)
    n_completed = sum(1 for s in _sessions if s.completed)
    if n_completed == 0 and _sessions:
        with open(base_path, "w") as f:
            json.dump(result.to_dict(save_responses=config.save_responses), f, indent=2)
        from agentsurge.cli._display import print_run_summary

        model_name = args.model or config.model
        print_run_summary(result, model=os.path.basename(model_name), saved_files=[base_path])
        sys.exit(1)

    # Warn if output tokens are suspiciously low (model may be stopping early)
    completed_turns = [t for s in _sessions for t in s.turns if t.completed]
    if completed_turns:
        median_out = sorted(t.output_tokens for t in completed_turns)[len(completed_turns) // 2]
        if median_out <= 4 and config.max_tokens > 16:
            _log.warning(
                "median output_tokens=%d - model may be stopping "
                "immediately (EOS). Check prompt format or try a different model.",
                median_out,
            )

    # Always write JSON (needed by --validate and other post-processing)
    with open(base_path, "w") as f:
        json.dump(result.to_dict(save_responses=config.save_responses), f, indent=2)

    # Always write CSVs for quick inspection
    turns_path = base_path.replace(".json", "_turns.csv")
    summary_path = base_path.replace(".json", "_summary.csv")
    result.to_turns_csv(turns_path)
    result.to_summary_csv(summary_path)

    saved_files = [base_path, turns_path, summary_path]

    # Session timeline figure
    try:
        from agentsurge.cli._timeline import save_session_timeline

        timeline_path = base_path.replace(".json", "_timeline.png")
        if save_session_timeline(result, timeline_path):
            saved_files.append(timeline_path)
    except ImportError:
        pass

    # --make-dataset: produce local HF-compatible parquet files plus metadata
    if getattr(args, "make_dataset", False):
        from agentsurge.hf import run_result_to_parquet, workload_to_parquet

        hf_dir = base_path.replace(".json", "_hf")
        model_name = args.model or config.model
        config_name = _dataset_config_name(args)
        run_result_to_parquet(
            result,
            os.path.join(hf_dir, "results"),
            model_name=model_name,
            config_name=config_name,
        )
        workload_to_parquet(
            sessions,
            os.path.join(hf_dir, "workload"),
            model_name=model_name,
            config_name=config_name,
        )
        saved_files.append(hf_dir + "/")

    # Rich summary (falls back to plain text if rich is not installed)
    from agentsurge.cli._display import print_run_summary

    model_name = args.model or config.model
    print_run_summary(result, model=os.path.basename(model_name), saved_files=saved_files)
