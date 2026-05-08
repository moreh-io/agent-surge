# SPDX-License-Identifier: MIT
"""Shared SLO probe and binary-search primitives.

Provides two building blocks that eliminate duplicated probe + bisect logic
across ``_slo_search.py`` and ``_slo_interleaved.py``:

- :func:`slo_probe`         -- run *n_sess* sessions and return TTFT stats.
- :func:`slo_binary_search` -- exponential scan + bisect to find the SLO
                               boundary, parameterised on a caller-supplied
                               probe coroutine.
"""

import sys
import time
from argparse import Namespace
from collections.abc import Awaitable, Callable
from typing import cast

import numpy as np

from agentsurge.backends import BackendBase
from agentsurge.cli._helpers import _generate_sessions
from agentsurge.runner import BenchmarkConfig, BenchmarkRunner


async def slo_probe(
    *,
    vllm_url: str,
    model: str,
    n_sess: int,
    n_turns: int,
    tokens_per_turn: int,
    max_tokens: int,
    extra_body: dict | None = None,
    tokenizer_model: str | None = None,
    trust_remote_code: bool = False,
    api_type: str = "chat",
    backend_name: str | None = None,
    request_timeout: int = 7200,
    stream_idle_timeout: float = 0.0,
    enable_thinking: bool = False,
) -> dict:
    """Run *n_sess* sessions against a vLLM endpoint and return TTFT stats.

    This is the single canonical implementation of the "probe" pattern used
    by both the single-config and interleaved SLO searches.

    Parameters
    ----------
    vllm_url, model:
        Server endpoint and model identifier.
    n_sess:
        Number of sessions to launch.
    n_turns, tokens_per_turn, max_tokens:
        Workload shape.
    extra_body:
        Optional extra_body dict forwarded to ``BenchmarkConfig``.
    tokenizer_model:
        Optional HF tokenizer ID or local path forwarded to ``BenchmarkConfig``.
        When ``None``, the runner falls back to ``model``.
    trust_remote_code:
        Forwarded to ``BenchmarkConfig`` to enable ``trust_remote_code`` when
        loading the tokenizer.

    Returns
    -------
    dict
        Keys: ``n``, ``kv_pct``, ``ttft_p50_ms``, ``ttft_p95_ms``,
        ``n_completed``, ``n_err``, ``error_rate``, ``elapsed_s``.
    """
    sessions = _generate_sessions(None, n_sess, n_turns, tokens_per_turn)
    rate = max(10, n_sess // 5)
    cfg_kwargs: dict = dict(
        vllm_url=vllm_url,
        model=model,
        max_concurrency=min(n_sess, 1200),
        arrival_pattern="poisson",
        arrival_rate=rate,
        max_tokens=max_tokens,
        ignore_replay_output_length=True,
        tokenizer_model=tokenizer_model,
        trust_remote_code=trust_remote_code,
        api_type=api_type,
        request_timeout=request_timeout,
        stream_idle_timeout=stream_idle_timeout,
        enable_thinking=enable_thinking,
    )
    if extra_body is not None:
        cfg_kwargs["extra_body"] = extra_body
    config = BenchmarkConfig(**cfg_kwargs)
    backend: BackendBase | None = None
    if backend_name:
        from agentsurge.cli.run import _build_backend

        backend = _build_backend(Namespace(backend=backend_name), config)
    runner = BenchmarkRunner(config, backend=backend)
    t0 = time.monotonic()
    result = await runner.run(sessions)
    elapsed = time.monotonic() - t0

    ttft_vals = result.ttft_values()
    ttft_p50 = float(np.percentile(ttft_vals, 50)) if ttft_vals else 0.0
    ttft_p95 = float(np.percentile(ttft_vals, 95)) if ttft_vals else 0.0
    _sessions = cast(list, result.sessions)
    n_completed = sum(1 for s in _sessions if s.completed)
    n_err = len(_sessions) - n_completed
    err_rate = n_err / max(len(_sessions), 1)
    kv_util_peak = result.backend_metrics.get("kv_util_peak")

    return {
        "n": n_sess,
        "kv_pct": round(float(kv_util_peak) * 100, 2)
        if isinstance(kv_util_peak, (int, float))
        else float("nan"),
        "ttft_p50_ms": round(ttft_p50, 1),
        "ttft_p95_ms": round(ttft_p95, 1),
        "n_completed": n_completed,
        "n_err": n_err,
        "error_rate": round(err_rate, 4),
        "elapsed_s": round(elapsed, 1),
    }


def check_slo(r: dict, slo_ms: float, max_error_rate: float = 0.05) -> bool:
    """Return True when a probe result meets the SLO target."""
    # Default 5%: vLLM benchmark standard; override with --max-error-rate
    return bool(r["ttft_p95_ms"] < slo_ms and r["error_rate"] <= max_error_rate)


def next_n(current_n: int, kv_pct: float) -> int:
    """Compute the next concurrency level for the exponential scan.

    Uses a KV-aware adaptive step: large jumps when KV is low (far from
    saturation), small steps when KV is high (approaching the cliff).
    """
    if kv_pct < 30:
        return min(current_n * 2, current_n + 100)
    elif kv_pct < 60:
        return current_n + 20
    else:
        return current_n + 10


ProbeCallable = Callable[[int], Awaitable[dict]]


async def slo_binary_search(
    probe_fn: ProbeCallable,
    *,
    slo_ms: float,
    max_n_cap: int,
    start_n: int = 10,
    hint: int | None = None,
    label: str = "",
    max_error_rate: float = 0.05,
) -> dict:
    """Exponential scan + bisect to find the SLO boundary.

    Parameters
    ----------
    probe_fn:
        ``async def probe_fn(n_sess: int) -> dict`` -- must return the
        standard probe-result dict (keys ``n``, ``kv_pct``, ``ttft_p95_ms``,
        ``error_rate``, ...).  The caller is responsible for cooldown logic
        and any per-config bookkeeping.
    slo_ms:
        SLO target -- max acceptable p95 TTFT in ms.
    max_n_cap:
        Upper bound for the session count search.
    start_n:
        Initial concurrency level for the exponential scan (default 10).
    hint:
        Warm-start hint -- approximate spike N from a prior run.  When
        provided, skips the full exponential scan and starts binary search
        in the range ``[max(5, hint*0.5), hint*2.0]``.
    label:
        Human-readable label printed in progress output.

    Returns
    -------
    dict
        ``{"max_n": int, "boundary": dict|None, "scan": list[dict]}``
        where *max_n* is the largest N meeting SLO, *boundary* is the
        probe result at that N, and *scan* is the full list of probes
        sorted by N.
    """
    scan_results: list[dict] = []

    def _is_ok(r: dict) -> bool:
        return check_slo(r, slo_ms, max_error_rate)

    def _status_label(r: dict) -> str:
        if _is_ok(r):
            return "OK"
        if r["ttft_p95_ms"] >= slo_ms:
            return "TIMEOUT"
        return "ERROR"

    def _print_probe(n: int, r: dict) -> None:
        status = _status_label(r)
        print(
            f"  n={n:>4}: kv={r['kv_pct']:>5.1f}% p95={r['ttft_p95_ms']:>8.1f}ms "
            f"err={r['error_rate'] * 100:.1f}% [{status}]"
        )
        sys.stdout.flush()

    if hint is not None:
        lo = max(5, round(hint * 0.5 / 5) * 5)
        hi = min(max_n_cap, round(hint * 2.0 / 5) * 5)
        hi = max(hi, lo + 5)  # ensure range is non-empty
        print(f"\n--- Hint={hint}: skipping scan, bisecting [{lo}, {hi}] ---")

        # Validate lo endpoint
        r_lo = await probe_fn(lo)
        scan_results.append(r_lo)
        _print_probe(lo, r_lo)

        if not _is_ok(r_lo):
            print("  >> SLO breached at hint lo. Max N = 0")
            return {"max_n": 0, "boundary": r_lo, "scan": scan_results}

        # Validate hi endpoint
        r_hi = await probe_fn(hi)
        scan_results.append(r_hi)
        _print_probe(hi, r_hi)

        if _is_ok(r_hi):
            print(f"  [hint] hi={hi} still OK, expanding upward...")
            lo = hi
            n = next_n(hi, r_hi["kv_pct"])
            while n <= max_n_cap:
                r = await probe_fn(n)
                scan_results.append(r)
                _print_probe(n, r)
                if not _is_ok(r):
                    hi = n
                    break
                lo = n
                n = next_n(n, r["kv_pct"])
            else:
                hi = n
            hi = min(hi, max_n_cap)
        else:
            pass
    else:
        print("\n--- Phase 1: Exponential scan ---")
        lo, hi = 0, 0
        n = start_n

        while n <= max_n_cap:
            r = await probe_fn(n)
            scan_results.append(r)
            _print_probe(n, r)

            if not _is_ok(r):
                hi = n
                break
            lo = n
            n = next_n(n, r["kv_pct"])
        else:
            hi = n
        hi = min(hi, max_n_cap)

    if lo == 0 and hi <= start_n:
        print(f"  >> SLO breached at n={start_n}. Max N = 0")
        return {
            "max_n": 0,
            "boundary": scan_results[0] if scan_results else None,
            "scan": scan_results,
        }

    print(f"\n--- Phase 2: Binary search [{lo}, {hi}] ---")
    best_n, best_r = lo, None
    for sr in scan_results:
        if _is_ok(sr) and sr["n"] >= best_n:
            best_n, best_r = sr["n"], sr

    while hi - lo > 5:
        mid = ((lo + hi) // 2 // 5) * 5
        mid = max(lo + 5, mid)
        if mid >= hi:
            break

        r = await probe_fn(mid)
        scan_results.append(r)
        _print_probe(mid, r)

        if _is_ok(r):
            lo = mid
            if mid > best_n:
                best_n, best_r = mid, r
        else:
            hi = mid

    print(f"\n  >> Max N meeting SLO: {best_n}")
    if best_r:
        print(f"     kv={best_r['kv_pct']:.1f}% p95={best_r['ttft_p95_ms']:.1f}ms")

    return {"max_n": best_n, "boundary": best_r, "scan": sorted(scan_results, key=lambda x: x["n"])}
