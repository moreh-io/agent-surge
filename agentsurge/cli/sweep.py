# SPDX-License-Identifier: MIT
"""cmd_sweep - multi-level Poisson sweep subcommand.

Also dispatches to SLO binary search (--slo-ms) mode, which was
previously a separate subcommand.
"""

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import cast

import numpy as np

from agentsurge.cli._helpers import (
    _build_runner_config,
    _collision_suffix,
    _generate_sessions,
    _load_config,
    _resolve_lmcache_urls,
    _smart_cooldown,
    _validate_numeric_args,
)
from agentsurge.runner import BenchmarkRunner

_log = logging.getLogger(__name__)

# Hard cap on per-level client concurrency. Above ~1200 in-flight requests
# the local client (asyncio + aiohttp) is the bottleneck, not the server,
# so probing higher just floods the client side. The cap is announced via
# `_log.warning` whenever a probe level exceeds it.
MAX_LEVEL_CONCURRENCY = 1200


def _probe_levels_type(s: str) -> list[int]:
    """Argparse type= for --probe-levels N,N,N."""
    parts = s.split(",")
    try:
        return [int(x) for x in parts]
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            f"--probe-levels must be comma-separated integers (e.g. '50,100,200'); got {s!r}"
        ) from err


def cmd_sweep(args: argparse.Namespace) -> None:
    """Run multi-level Poisson sweep with tier metrics collection.

    Modes:
    - Default: fixed-level sweep across --concurrency levels.
    - ``--slo-ms``: binary search for max N meeting SLO target (absorbed from 'slo').

    Supports --source (synthetic/mixed/openhands) and --preset for realistic workloads.
    Integrates with workload logic: stops at KV saturation or SLO breach.
    """
    _validate_numeric_args(args)
    slo_mode = getattr(args, "slo_ms", None) is not None

    hint_val = getattr(args, "hint", None)
    if hint_val is not None:
        if hint_val <= 0:
            print("Error: --hint must be a positive integer.", file=sys.stderr)
            raise SystemExit(1)
        if not slo_mode:
            print(
                "Warning: --hint is only used with --slo-ms. Ignoring.",
                file=sys.stderr,
            )

    if slo_mode:
        from agentsurge.cli.slo import cmd_slo

        cmd_slo(args)
        return

    cfg = _load_config(args.config)
    vllm = cfg.get("vllm") or {}

    vllm_url = args.vllm_url or vllm.get("url")
    model_path = args.model or vllm.get("model")
    if not vllm_url or not model_path:
        from agentsurge.cli._helpers import CLIConfigError

        missing = "vllm.url" if not vllm_url else "vllm.model"
        flag = "--vllm-url" if not vllm_url else "--model"
        raise CLIConfigError(
            f"missing {missing}: pass {flag} or set '{missing}' in your config YAML"
        )
    levels = args.probe_levels
    source = getattr(args, "source", None)
    tool_mode = getattr(args, "tool_mode", None) or "off"
    _REAL_MODE_UNSUPPORTED = {"featurebench", "swe-evo", "mixed"}
    if tool_mode == "real" and source in _REAL_MODE_UNSUPPORTED:
        print(
            f"Error: --source {source} does not support --tool-mode real "
            f"(no repo metadata for workspace setup). "
            f"Use --tool-mode replay or --tool-mode off.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    extra_metrics_urls = _resolve_lmcache_urls(args, cfg)
    base_config = _build_runner_config(
        args,
        cfg,
        overrides={
            "arrival_pattern": getattr(args, "arrival", None) or None,  # let preset decide
        },
    )
    preset_name = getattr(args, "preset", None)

    n_turns = args.n_turns or cfg["workload"].get("n_turns", 3)
    tokens_per_turn = args.synthetic_tokens_per_turn or cfg["workload"].get(
        "synthetic_tokens_per_turn", 2700
    )
    cooldown = args.cooldown
    no_cooldown = getattr(args, "no_cooldown", False)
    if no_cooldown:
        print(
            "  [warn] --no-cooldown: KV state may carry over between probe levels, results may be noisy"
        )
    max_error_rate = args.max_error_rate
    ttft_threshold = getattr(args, "ttft_threshold", None)

    source_label = source or "synthetic"
    print(f"=== Sweep: {model_path} ===")
    print(f"URL: {vllm_url}")
    print(f"Source: {source_label}" + (f" | Preset: {preset_name}" if preset_name else ""))
    _lmcache_url = getattr(args, "lmcache_url", None) or cfg.get("lmcache", {}).get("url")
    print(f"LMCache: {'on (' + _lmcache_url + ')' if extra_metrics_urls else 'off'}")
    if not source:
        spr_info = getattr(args, "prefix_overlap_fraction", 0.0) or 0.0
        spr_str = f" | prefix_overlap={spr_info:.0%}" if spr_info > 0 else ""
        print(f"Workload: {n_turns}T x {tokens_per_turn}tok{spr_str}")
    print(f"Levels: {levels}")
    if ttft_threshold:
        print(f"TTFT threshold: {ttft_threshold}ms (stops on breach)")
    print()

    async def _run() -> list[dict]:
        results = []
        prev_kv = 0.0
        prev_n = 0
        prev_had_error = False
        for idx, n_sess in enumerate(levels):
            # Smart cooldown: adapt wait strategy based on direction and errors
            if idx > 0:
                next_n = n_sess  # the probe we're about to run
                await _smart_cooldown(
                    vllm_url,
                    current_n=prev_n,
                    next_n=next_n,
                    current_kv=prev_kv,
                    had_error=prev_had_error,
                    no_cooldown=no_cooldown,
                    timeout=cooldown,
                )
            rate = max(10, n_sess // 5)
            if getattr(args, "arrival_rate", None) is not None:
                rate = args.arrival_rate
            print(f"  n={n_sess}, arrival_rate={rate} req/s ...")

            data_dir = getattr(args, "data_dir", None)
            spr = getattr(args, "prefix_overlap_fraction", 0.0) or 0.0
            sessions = _generate_sessions(
                source,
                n_sess,
                n_turns,
                tokens_per_turn,
                data_dir=data_dir,
                prefix_overlap_fraction=spr,
                trace_pool_path=getattr(args, "trace_pool", None),
                multi_turn_only=getattr(args, "multi_turn_only", False),
            )

            level_config = dataclasses.replace(
                base_config,
                max_concurrency=min(n_sess, 1200),
                arrival_rate=rate,
            )
            from agentsurge.cli.run import _build_backend

            backend = _build_backend(args, level_config)
            runner = BenchmarkRunner(level_config, backend=backend)

            t0 = time.monotonic()
            result = await runner.run(sessions)
            elapsed = time.monotonic() - t0

            ttft_vals = result.ttft_values()
            ttft_p50 = float(np.percentile(ttft_vals, 50)) if ttft_vals else 0.0
            ttft_p95 = float(np.percentile(ttft_vals, 95)) if ttft_vals else 0.0
            ttft_p99 = float(np.percentile(ttft_vals, 99)) if ttft_vals else 0.0
            _sessions = cast(list, result.sessions)
            n_completed = sum(1 for s in _sessions if s.completed)
            n_err = len(_sessions) - n_completed
            err_rate = n_err / max(len(_sessions), 1)

            point = {
                "n_sessions": n_sess,
                "arrival_rate": rate,
                "kv_util_peak": result.kv_util_peak,
                "ttft_p50_ms": round(ttft_p50, 1),
                "ttft_p95_ms": round(ttft_p95, 1),
                "ttft_p99_ms": round(ttft_p99, 1),
                "n_completed": n_completed,
                "n_err": n_err,
                "error_rate": round(err_rate, 4),
                "elapsed_s": round(elapsed, 1),
                "prefix_cache_hit_rate": round(result.prefix_cache_hit_rate, 4),
                "lmcache_v1_deltas": result.lmcache_v1_deltas,
            }
            results.append(point)

            _kv_util_peak = cast(float, result.kv_util_peak)
            prev_kv = _kv_util_peak
            prev_n = n_sess
            prev_had_error = n_err > 0

            _degraded = level_config.no_metrics
            lmc = cast("dict | None", result.lmcache_v1_deltas)
            hit_str = f"lmc_hit={lmc.get('hit_rate', 0):.2f}" if lmc else "lmc=off"
            kv_str = "--" if _degraded else f"{_kv_util_peak * 100:.1f}%"
            pfx_str = "--" if _degraded else f"{result.prefix_cache_hit_rate:.2f}"
            print(
                f"    kv_util={kv_str} | ttft p50={ttft_p50:.0f} p95={ttft_p95:.0f}ms | "
                f"completed={n_completed} err={n_err} | pfx={pfx_str} {hit_str if not _degraded else ''} | {elapsed:.0f}s"
            )

            if err_rate > max_error_rate:
                print(f"    >>> Error rate > {max_error_rate * 100:.0f}%, stopping sweep")
                break

            if ttft_threshold and ttft_p95 > ttft_threshold:
                print(
                    f"    >>> TTFT p95 ({ttft_p95:.0f}ms) > threshold ({ttft_threshold}ms), saturated"
                )
                break

        return results

    results = asyncio.run(_run())

    _degraded = base_config.no_metrics
    print(f"\n{'=' * 90}")
    if _degraded:
        hdr = f"{'n':>6}  {'kv%':>6}  {'p50ms':>7}  {'p95ms':>7}  {'p99ms':>7}  {'err%':>6}"
    else:
        hdr = f"{'n':>6}  {'kv%':>6}  {'p50ms':>7}  {'p95ms':>7}  {'p99ms':>7}  {'err%':>6}  {'pfx_hit':>8}  {'lmc_hit':>8}"
    print(hdr)
    print("-" * len(hdr))
    for p in results:
        if _degraded:
            print(
                f"{p['n_sessions']:>6}  {'--':>6}  {p['ttft_p50_ms']:>7.0f}  "
                f"{p['ttft_p95_ms']:>7.0f}  {p['ttft_p99_ms']:>7.0f}  "
                f"{p['error_rate'] * 100:>5.1f}%"
            )
        else:
            lmc = p.get("lmcache_v1_deltas") or {}
            kv_pct = p["kv_util_peak"] * 100 if p["kv_util_peak"] is not None else float("nan")
            print(
                f"{p['n_sessions']:>6}  {kv_pct:>5.1f}%  {p['ttft_p50_ms']:>7.0f}  "
                f"{p['ttft_p95_ms']:>7.0f}  {p['ttft_p99_ms']:>7.0f}  "
                f"{p['error_rate'] * 100:>5.1f}%  {p['prefix_cache_hit_rate']:>8.3f}  "
                f"{lmc.get('hit_rate', float('nan')):>8.3f}"
            )
    print(f"{'=' * 90}")

    if ttft_threshold and results:
        sat_idx = next(
            (i for i, p in enumerate(results) if p["ttft_p95_ms"] > ttft_threshold), None
        )
        if sat_idx is not None and sat_idx > 0:
            last_ok = results[sat_idx - 1]
            print(
                f"\nSaturation point: ~{last_ok['n_sessions']} sessions "
                f"(KV util peak={last_ok['kv_util_peak'] * 100:.1f}%, TTFT p95={last_ok['ttft_p95_ms']:.0f}ms)"
            )
        elif sat_idx == 0:
            print(f"\nSaturation: Already saturated at first level ({results[0]['n_sessions']})")

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir, f"sweep_{ts}_{_collision_suffix()}.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "vllm_url": vllm_url,
                "model": model_path,
                "source": source_label,
                "preset": preset_name,
                "lmcache_url": _lmcache_url if extra_metrics_urls else None,
                "workload": {"n_turns": n_turns, "synthetic_tokens_per_turn": tokens_per_turn},
                "ttft_threshold": ttft_threshold,
                "results": results,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"Saved: {out_path}")

    # Exit 2 on SLO/threshold breach so CI can gate. See cli.md H1.
    if results:
        if ttft_threshold and any(p["ttft_p95_ms"] > ttft_threshold for p in results):
            sys.exit(2)
        if any(p["error_rate"] > max_error_rate for p in results):
            sys.exit(2)
