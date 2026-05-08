# SPDX-License-Identifier: MIT
"""Shared CLI utility helpers.

This module contains pure-utility functions used by multiple CLI subcommands.
They are separated here to avoid mixing library-level utilities with the
argument-parsing assembly code in ``agentsurge.cli._app``.

All helpers are re-exported from ``agentsurge.cli`` for backward compatibility.
"""

import argparse
import asyncio
import logging
import os
import time
import urllib.request
import uuid
from datetime import datetime
from typing import Any, cast

import yaml

from agentsurge.generators.mixed import MixedWorkloadGenerator
from agentsurge.generators.synthetic import SyntheticGenerator
from agentsurge.generators.task_converter import TaskToTrajectoryConverter
from agentsurge.generators.trace_pool import TracePool
from agentsurge.generators.trace_replay import TraceReplayGenerator
from agentsurge.loaders import get_loader, loader_registry
from agentsurge.metrics import fetch_metrics
from agentsurge.preset import LMCACHE_AUTO_PRESETS, preset_to_kwargs, resolve_preset_config
from agentsurge.runner import BenchmarkConfig
from agentsurge.types.results import _resolve_use_model_reply

_log = logging.getLogger(__name__)


class CLIConfigError(ValueError):
    """Raised when a required CLI config key is missing or empty.

    cli.md H4: previously bare KeyError dumped a traceback; this typed
    exception lets the CLI catch & format it.
    """


def _validate_numeric_args(args: argparse.Namespace) -> None:
    """Validate numeric CLI args. cli.md M2.

    Rejects:
      - duration < 0
      - arrival_rate <= 0 (unless 'auto' from preset/config)
      - n_sessions <= 0
      - max_tokens <= 0
      - ramp_duration > duration when both > 0
    """
    errors: list[str] = []

    duration = getattr(args, "duration", None)
    if duration is not None and duration < 0:
        errors.append(f"--duration must be >= 0; got {duration}")

    ramp = getattr(args, "ramp_duration", None)
    if ramp is not None and ramp < 0:
        errors.append(f"--ramp-duration must be >= 0; got {ramp}")
    if ramp is not None and duration is not None and ramp > 0 and duration > 0 and ramp > duration:
        errors.append(f"--ramp-duration ({ramp}) must be <= --duration ({duration})")

    rate = getattr(args, "arrival_rate", None)
    if rate is not None and not isinstance(rate, str) and rate <= 0:
        errors.append(f"--arrival-rate must be > 0; got {rate}")

    n_sessions = getattr(args, "n_sessions", None)
    if n_sessions is not None and n_sessions <= 0:
        errors.append(f"--n-sessions must be > 0; got {n_sessions}")

    max_tokens = getattr(args, "max_tokens", None)
    if max_tokens is not None and max_tokens <= 0:
        errors.append(f"--max-tokens must be > 0; got {max_tokens}")

    if errors:
        raise CLIConfigError("; ".join(errors))


def _tokenizer_kwargs(raw: str | None) -> dict:
    """Normalize the --tokenizer CLI value into BenchmarkConfig keyword args.

    - None or 'auto' (case-insensitive) → tokenizer_model=None, skip_tokenizer_load=False
    - 'none' (case-insensitive)         → tokenizer_model=None, skip_tokenizer_load=True
    - anything else                     → tokenizer_model=raw,  skip_tokenizer_load=False
    """
    value = (raw or "auto").strip().lower()
    if value == "auto":
        return {"tokenizer_model": None, "skip_tokenizer_load": False}
    if value == "none":
        return {"tokenizer_model": None, "skip_tokenizer_load": True}
    return {"tokenizer_model": raw, "skip_tokenizer_load": False}


def _load_config(path: str | None = None) -> dict:
    """Load config from YAML file or URL, falling back to bundled default.

    Resolution order:
      1. Explicit path argument (local file or http(s) URL)
      2. AGENTSURGE_CONFIG environment variable (local file or URL)
      3. Bundled configs/default.yaml
    """
    import sys

    explicit_path = path
    env_path = os.environ.get("AGENTSURGE_CONFIG")
    path = path or env_path
    cfg = None
    if path:
        if path.startswith(("http://", "https://")):
            with urllib.request.urlopen(path, timeout=10) as resp:
                cfg = yaml.safe_load(resp.read())
        elif os.path.exists(path):
            with open(path) as f:
                cfg = yaml.safe_load(f)
        elif explicit_path:
            # Explicit -c pointing to missing file is loud.
            raise FileNotFoundError(f"Config file not found: {path}")
        else:
            # cli.md M1: env-var path missing → soft fallback to default.
            _log.warning(
                "AGENTSURGE_CONFIG=%s does not exist; falling back to bundled default",
                path,
            )
            path = None  # so the empty-yaml branch doesn't fire
        if cfg is None and explicit_path:
            sys.exit(
                f"error: config file '{path}' is empty or contains only null — "
                "cannot fall back to default when a path was explicitly provided."
            )
    if cfg is None:
        default = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "configs",
            "default.yaml",
        )
        with open(default) as f:
            cfg = yaml.safe_load(f)
    # Migrate legacy 'saturator' key to 'workload'
    if "saturator" in cfg and "workload" not in cfg:
        cfg["workload"] = cfg.pop("saturator")
    result: dict = cfg
    return result


def _resolve_lmcache_urls(
    args: argparse.Namespace, cfg: dict, auto_presets: tuple = ()
) -> list[str] | None:
    """Resolve LMCache metrics endpoint URLs from args + config."""
    urls = []
    lmcache_cfg = cfg.get("lmcache", {})
    cli_url = getattr(args, "lmcache_url", None)
    enabled = getattr(args, "lmcache", None)
    if enabled is None:
        enabled = (
            cli_url is not None
            or getattr(args, "preset", None) in auto_presets
            or lmcache_cfg.get("enabled", False)
        )
    url = cli_url or lmcache_cfg.get("url")
    if enabled and url:
        urls.append(url)
    return urls or None


def _build_runner_config(
    args: argparse.Namespace, cfg: dict, *, overrides: dict | None = None
) -> BenchmarkConfig:
    """Build a BenchmarkConfig from CLI args + YAML config + preset.

    Consolidates the repeated preset/config resolution pattern used by
    cmd_run, cmd_sweep, etc.
    """
    vllm = cfg.get("vllm") or {}
    run_cfg = cfg.get("runner", {})

    vllm_url = getattr(args, "vllm_url", None) or vllm.get("url")
    model_path = getattr(args, "model", None) or vllm.get("model")
    if not vllm_url:
        raise CLIConfigError(
            "missing vllm.url: pass --vllm-url or set 'vllm.url' in your config YAML"
        )
    if not model_path:
        raise CLIConfigError(
            "missing vllm.model: pass --model or set 'vllm.model' in your config YAML"
        )

    preset_name = getattr(args, "preset", None)
    preset_cfg: dict = {}
    if preset_name:
        raw_preset = resolve_preset_config(preset_name, yaml_cfg=cfg)
        preset_cfg = preset_to_kwargs(raw_preset)

    think_time = preset_cfg.get("think_time")
    if think_time:
        think_time = tuple(think_time)
    if getattr(args, "think_time", None) is not None:
        think_time = args.think_time

    tool_delay = preset_cfg.get("tool_delay", False)
    if getattr(args, "tool_delay", None) is not None:
        tool_delay = args.tool_delay.lower() not in ("false", "0", "no")

    tool_mode_resolved = getattr(args, "tool_mode", None) or "off"
    # Default derives from tool_mode: replay keeps recorded messages for
    # reproducibility; real/off use the actual model output.
    _umr = getattr(args, "use_model_reply_in_next_turn", None)
    explicit_umr: bool | None = None
    if _umr is not None and _umr.lower() != "auto":
        explicit_umr = _umr.lower() == "true"
    use_model_reply = _resolve_use_model_reply(explicit_umr, tool_mode_resolved)

    interruption_rate = preset_cfg.get("interruption_rate", 0.0)
    if getattr(args, "interruption_rate", None) is not None:
        interruption_rate = args.interruption_rate

    single_turn_ratio = preset_cfg.get("single_turn_ratio", 0.0)
    if getattr(args, "single_turn_ratio", None) is not None:
        single_turn_ratio = args.single_turn_ratio

    arrival = (
        getattr(args, "arrival", None)
        or preset_cfg.get("arrival_pattern")
        or run_cfg.get("arrival_pattern", "poisson")
    )
    gamma_cv = preset_cfg.get("gamma_cv", run_cfg.get("gamma_cv", 1.2))

    arrival_rate = preset_cfg.get("arrival_rate", run_cfg.get("arrival_rate", 10.0))
    if getattr(args, "arrival_rate", None) is not None:
        arrival_rate = args.arrival_rate
    if arrival_rate == "auto" or (isinstance(arrival_rate, str) and arrival_rate.lower() == "auto"):
        concurrency = getattr(args, "client_concurrency", None) or run_cfg.get(
            "max_concurrency", 100
        )
        arrival_rate = max(1.0, concurrency * 0.3)
    else:
        arrival_rate = float(arrival_rate)

    extra_metrics_urls = _resolve_lmcache_urls(args, cfg, auto_presets=LMCACHE_AUTO_PRESETS)

    max_model_len = getattr(args, "synthetic_prompt_scale", None)
    if max_model_len is None:
        models_cfg = cfg.get("models", {})
        model_key = os.path.basename(model_path)
        max_model_len = models_cfg.get(model_key, {}).get("max_model_len", 16384)

    no_metrics = getattr(args, "no_metrics", False)
    backend_name = getattr(args, "backend", None)
    if backend_name == "mock":
        no_metrics = True
    elif backend_name == "openai" and not no_metrics:
        if not extra_metrics_urls and not getattr(args, "vllm_url", None):
            no_metrics = True

    kwargs = dict(
        vllm_url=vllm_url,
        model=model_path,
        max_concurrency=getattr(args, "client_concurrency", None)
        or run_cfg.get("max_concurrency", 100),
        max_tokens=(
            _mt
            if (_mt := getattr(args, "max_tokens", None)) is not None
            else preset_cfg.get("max_tokens", run_cfg.get("max_tokens", 64))
        ),
        temperature=run_cfg.get("temperature", 0.3),
        arrival_pattern=arrival,
        arrival_rate=arrival_rate,
        api_type=getattr(args, "api", "chat"),
        think_time=think_time,
        tool_delay=tool_delay,
        max_retries=(
            _r
            if (_r := getattr(args, "max_retries", None)) is not None
            else preset_cfg.get("max_retries", 0)
        ),
        use_model_reply_in_next_turn=use_model_reply,
        tool_output_mode=(
            getattr(args, "tool_output_mode", None)
            or preset_cfg.get("tool_output_mode", "recorded")
        ),
        preset=preset_name,
        gamma_cv=gamma_cv,
        interruption_rate=interruption_rate,
        single_turn_ratio=single_turn_ratio,
        max_model_len=max_model_len,
        context_distribution=getattr(args, "context_distribution", None) or "mixed",
        seed=getattr(args, "seed", 42),
        extra_metrics_urls=extra_metrics_urls or [],
        no_metrics=no_metrics,
        request_timeout=run_cfg.get("request_timeout", 7200),
        warm_up=getattr(args, "warm_up", 0) or 0,
        ramp_duration=getattr(args, "ramp_duration", None) or 0.0,
        duration=getattr(args, "duration", None) or 0.0,
        tool_mode=getattr(args, "tool_mode", None) or "off",
        tool_call_parser_fallback=getattr(args, "tool_call_parser_fallback", None) or "off",
        workspace_dir=getattr(args, "workspace_dir", None) or getattr(args, "sandbox_dir", None),
        sandbox_dir=getattr(args, "sandbox_dir", None),
        tool_env=getattr(args, "tool_env", None) or "safe",
        save_responses=getattr(args, "save_responses", False),
        no_stream=getattr(args, "no_stream", False),
        ignore_replay_output_length=getattr(args, "ignore_replay_output_length", False),
        thinking_budget=getattr(args, "thinking_budget", 8192),
        enable_thinking=getattr(args, "enable_thinking", False),
        **_tokenizer_kwargs(getattr(args, "tokenizer", None)),
        trust_remote_code=getattr(args, "trust_remote_code", False),
        sanitize_truncated_tool_calls=getattr(args, "sanitize_truncated_tool_calls", False),
        continue_turn_on=(getattr(args, "continue_turn_on", None) or "never"),
        max_session_time=getattr(args, "max_session_time", 0.0) or 0.0,
        auto_finish_on_exhaust=getattr(args, "auto_finish_on_exhaust", False),
        max_retry_turns=getattr(args, "max_retry_turns", 0) or 0,
        max_budget_usd=getattr(args, "max_budget_usd", 0.0) or 0.0,
        input_price_per_mtok=getattr(args, "input_price_per_mtok", 0.0) or 0.0,
        output_price_per_mtok=getattr(args, "output_price_per_mtok", 0.0) or 0.0,
        stream_idle_timeout=getattr(args, "stream_idle_timeout", 0.0) or 0.0,
        retry_profile=getattr(args, "retry_profile", None) or "default",
        inflight_dump=bool(getattr(args, "enable_inflight_dump", False)),
        **(
            {"continue_prompt": args.continue_prompt}
            if getattr(args, "continue_prompt", None)
            else {}
        ),
        extra_body=cfg.get("extra_body"),
    )

    if overrides:
        kwargs.update({k: v for k, v in overrides.items() if v is not None})

    return BenchmarkConfig(**kwargs)


def _collision_suffix() -> str:
    """Return a short pid+random token unique across parallel invocations.

    Appended to timestamps anywhere an output filename is constructed, so
    two CLI calls (run / sweep / slo) launched in the same wall-second do
    not collide on the same target path - critical for the sibling
    ``<prefix>_<ts>_turns.jsonl`` the inflight writer appends to.

    cli.md H3: the random tail is 8 hex chars (32 bits), reducing parallel
    same-second collision probability from 1/65k to 1/4B.
    """
    return f"{os.getpid()}_{uuid.uuid4().hex[:8]}"


def _output_path(base_dir: str, prefix: str) -> str:
    """Generate a timestamped output path with a collision suffix.

    cli.md H3: re-rolls the suffix up to 8 times if the candidate already
    exists, so parallel runs never silently overwrite each other's
    sibling artefacts (`_turns.csv`, `_summary.csv`, `_timeline.png`,
    `_hf/`). Raises FileExistsError on truly pathological collisions
    (e.g. monkeypatched uuid).
    """
    os.makedirs(base_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for _ in range(8):
        candidate = os.path.join(base_dir, f"{prefix}_{ts}_{_collision_suffix()}.json")
        if not os.path.exists(candidate):
            return candidate
    raise FileExistsError(
        f"could not allocate non-colliding output path under {base_dir!r} "
        f"with prefix {prefix!r} (8 attempts exhausted; suffix is deterministic?)"
    )


def _get_trace_loader(source: str, local_path: str | None = None, split: str | None = None) -> Any:
    """Get a TraceLoader instance by name.

    Delegates class lookup to :func:`agentsurge.loaders.get_loader` so that
    third-party loaders registered at runtime are also discoverable.

    Parameters
    ----------
    source:
        Loader name or alias (e.g. ``"openhands"``, ``"swe-smith"``,
        ``"swe-smith-tool"``).
    local_path:
        Path to a local JSONL file.  When provided the loader skips the
        HuggingFace download.
    split:
        For ``"swe-smith"`` only - which sub-split to load
        (``"tool"``, ``"xml"``, ``"ticks"``).  If *source* already encodes
        the split (e.g. ``"swe-smith-xml"``) this argument is ignored.
    """
    try:
        cls = get_loader(source)
    except KeyError:
        available = ", ".join(loader_registry.list_trace())
        raise ValueError(f"Unknown trace source: {source!r}. Available: {available}") from None

    # SWESmithLoader requires a split argument - extract it from the source
    # name (e.g. "swe-smith-tool") or fall back to the explicit *split* kwarg.
    if source.startswith("swe-smith"):
        if source.count("-") >= 2:
            s_split = source.split("-", 2)[2]
        else:
            s_split = split or "tool"
        return cls(split=s_split, local_path=local_path)  # type: ignore[call-arg]

    return cls(local_path=local_path)  # type: ignore[call-arg]


def _get_task_loader(source: str, local_path: str | None = None) -> Any:
    """Get a TaskLoader instance by name.

    Delegates class lookup to :func:`agentsurge.loaders.get_loader` so that
    third-party loaders registered at runtime are also discoverable.

    Parameters
    ----------
    source:
        Loader name or alias (e.g. ``"swe-bench-verified"``, ``"swe-bench"``,
        ``"abc-bench"``).
    local_path:
        Path to a local JSONL file.  When provided the loader skips the
        HuggingFace download.
    """
    try:
        cls = get_loader(source)
    except KeyError:
        available = ", ".join(loader_registry.list_task())
        raise ValueError(f"Unknown task source: {source!r}. Available: {available}") from None

    return cls(local_path=local_path)  # type: ignore[call-arg]


_DATA_DIR_MAP = {
    "openhands": "openhands_trajectories.jsonl",
    "swe-smith": "swe_smith_tool.jsonl",  # default
    "swe-smith-tool": "swe_smith_tool.jsonl",
    "swe-smith-xml": "swe_smith_xml.jsonl",
    "swe-smith-ticks": "swe_smith_ticks.jsonl",
    "swe-bench-verified": "swe_bench_verified.jsonl",
    "abc-bench": "abc_bench.jsonl",
    "swe-evo": "swe_evo.jsonl",
    "featurebench": "featurebench.jsonl",
}


def _resolve_local(source: str, data_dir: str | None, local_path: str | None = None) -> str | None:
    """Resolve local path for a source: explicit local_path > data_dir auto-map."""
    if local_path:
        return local_path
    if not data_dir:
        return None

    filename = _DATA_DIR_MAP.get(source)
    if not filename and source.startswith("swe-smith"):
        filename = _DATA_DIR_MAP.get("swe-smith")

    if filename:
        candidate = os.path.join(data_dir, filename)
        if os.path.exists(candidate):
            return candidate
    return None


def _filter_multi_turn_only(sessions: list) -> list:
    """Return only sessions that have at least one pending follow-up user message.

    Used to implement --multi-turn-only. A session with empty
    ``pending_user_messages`` is a single-turn session (no follow-ups from
    the source trace), and is dropped.
    """
    return [s for s in sessions if getattr(s, "pending_user_messages", None)]


def _generate_sessions(
    source: str | None,
    n_sessions: int,
    n_turns: int,
    synthetic_tokens_per_turn: int,
    local_path: str | None = None,
    data_dir: str | None = None,
    prefix_overlap_fraction: float = 0.0,
    flatten_tools: bool = True,
    trace_pool_path: str | None = None,
    multi_turn_only: bool = False,
) -> list:
    """Generate workload sessions for sweep/run from various sources."""
    pool = None
    if trace_pool_path:
        pool = TracePool.from_json(trace_pool_path)

    if source in ("openhands",) or (source and source.startswith("swe-smith")):
        lp = _resolve_local(source, data_dir, local_path)
        loader = _get_trace_loader(source, lp)
        gen = TraceReplayGenerator(flatten_tools=flatten_tools)
        sessions = list(gen.from_trajectories(loader.load(limit=n_sessions * 3), limit=n_sessions))
    elif source == "mixed":
        gen = TraceReplayGenerator(flatten_tools=flatten_tools)
        converter = TaskToTrajectoryConverter(trace_pool=pool)
        mixer = MixedWorkloadGenerator()
        per_source = max(1, n_sessions // 6)
        all_sources = {}
        for src in ("openhands", "swe-smith"):
            try:
                lp = _resolve_local(src, data_dir, local_path)
                loader = _get_trace_loader(src, lp)
                sessions = list(
                    gen.from_trajectories(loader.load(limit=per_source), limit=per_source)
                )
                if sessions:
                    all_sources[src] = sessions
            except Exception as e:
                print(f"  [warn] {src}: {e}")
        for src in ("swe-bench-verified", "abc-bench", "swe-evo", "featurebench"):
            try:
                lp = _resolve_local(src, data_dir)
                task_loader = _get_task_loader(src, local_path=lp)
                pattern = TaskToTrajectoryConverter.DATASET_PATTERN_MAP.get(
                    task_loader.name, "bug-fix"
                )
                tasks = task_loader.load(limit=per_source)
                trajs = converter.from_tasks(tasks, pattern, limit=per_source)
                sessions = list(gen.from_trajectories(iter(trajs), limit=per_source))
                if sessions:
                    all_sources[src] = sessions
            except Exception as e:
                print(f"  [warn] {src}: {e}")
        sessions = mixer.mix(all_sources, total=n_sessions)
    elif source in ("swe-bench-verified", "abc-bench", "swe-evo", "featurebench"):
        lp = _resolve_local(source, data_dir, local_path)
        task_loader = _get_task_loader(source, local_path=lp)
        pattern = TaskToTrajectoryConverter.DATASET_PATTERN_MAP.get(task_loader.name, "bug-fix")
        converter = TaskToTrajectoryConverter(trace_pool=pool)
        tasks = task_loader.load(limit=n_sessions)
        trajs = converter.from_tasks(tasks, pattern, limit=n_sessions)
        gen = TraceReplayGenerator(flatten_tools=flatten_tools)
        sessions = list(gen.from_trajectories(iter(trajs), limit=n_sessions))
    else:
        synth_gen = SyntheticGenerator(
            tokens_per_turn=synthetic_tokens_per_turn,
            n_turns=n_turns,
            prefix_overlap_fraction=prefix_overlap_fraction,
            trace_pool=pool,
        )
        sessions = synth_gen.generate(n_sessions)

    if multi_turn_only:
        sessions = _filter_multi_turn_only(sessions)

    return sessions


async def _wait_kv_cooldown(vllm_url: str, target: float = 0.02, timeout: float = 60) -> None:
    """Wait until KV cache usage drops below target."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        snap = await fetch_metrics(vllm_url)
        if snap is None:
            await asyncio.sleep(2)
            continue
        if cast(float, snap.kv_cache_usage_perc) <= target:
            return
        await asyncio.sleep(2)


async def _smart_cooldown(
    vllm_url: str,
    *,
    current_n: int,
    next_n: int,
    current_kv: float,
    had_error: bool,
    no_cooldown: bool = False,
    timeout: float = 60,
) -> None:
    """Adaptive cooldown between sweep probes.

    Decides whether and how long to wait for KV cache to drain based on
    the direction of the sweep and the outcome of the previous probe.

    Rules:
      1. ``no_cooldown=True``: skip entirely (fast exploratory sweeps).
      2. ``had_error=True``: full cooldown to KV < 2% to reset server state
         after an error.
      3. ``next_n > current_n`` (ascending): skip cooldown - the next probe
         will consume equal or more KV anyway.
      4. ``next_n < current_n`` (descending, e.g. binary search):
         wait until KV drops below an estimated target for the next probe
         plus a 10% margin, capped at 0.50 so we never wait for very-low KV.
         Formula: ``target = min(0.50, (next_n / current_n) * current_kv + 0.10)``

    Parameters
    ----------
    vllm_url:
        vLLM server URL for metrics polling.
    current_n:
        Number of sessions in the probe that just finished.
    next_n:
        Number of sessions for the upcoming probe.
    current_kv:
        KV cache usage (0.0-1.0) observed in the probe that just finished.
    had_error:
        Whether the previous probe had errors (error_rate > 0).
    no_cooldown:
        If True, skip cooldown unconditionally (``--no-cooldown`` flag).
    timeout:
        Maximum seconds to wait before giving up.
    """
    if no_cooldown:
        return

    # After errors: full cooldown to reset server state
    if had_error:
        await _wait_kv_cooldown(vllm_url, target=0.02, timeout=timeout)
        return

    # Ascending direction: next probe uses more sessions, skip cooldown
    if next_n >= current_n:
        return

    # Descending direction (binary search): wait for proportional KV target
    if current_n > 0 and current_kv > 0:
        ratio = next_n / current_n
        target_kv = ratio * current_kv + 0.10
        target_kv = min(target_kv, 0.50)
    else:
        # Fallback: full cooldown
        target_kv = 0.02

    await _wait_kv_cooldown(vllm_url, target=target_kv, timeout=timeout)
