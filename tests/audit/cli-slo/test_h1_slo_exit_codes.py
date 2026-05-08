"""H1: SLO breach must exit with non-zero distinct code.

Audit: cli.md H1 — `_cmd_slo_single` and `cmd_sweep --ttft-threshold` print
results but never propagate breach into `sys.exit`. CI cannot gate on SLO
violation. Pin a distinct exit code for SLO violation.

Convention pinned by this fix:
  0   = success / SLO met
  1   = config / argument error (argparse + SystemExit(1))
  2   = SLO violation (was-0)
  3   = transport error (reserved)
  130 = SIGINT
"""

from __future__ import annotations

import argparse

import pytest


def _slo_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        config=None,
        vllm_url="http://x",
        model="m",
        slo_ms=2000.0,
        cooldown=0,
        no_cooldown=True,
        max_n=1,
        hint=None,
        profiles=None,
        tag="t",
        max_error_rate=0.05,
        configs=None,
        tokenizer="auto",
        trust_remote_code=False,
        api="chat",
        backend=None,
        stream_idle_timeout=0.0,
        enable_thinking=False,
        output_dir=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_cmd_slo_exits_2_when_max_n_zero(tmp_path, monkeypatch):
    """SLO breach at smallest probe → exit code 2 (was 0)."""
    from agentsurge.cli import slo as slo_mod

    args = _slo_args(output_dir=str(tmp_path))

    async def _fake_search(**kwargs):
        # max_n == 0 means even the smallest probe breached the SLO
        return [{"profile": "p", "max_n": 0, "boundary": {}}]

    monkeypatch.setattr(slo_mod, "_run_slo_search", _fake_search)
    monkeypatch.setattr(
        slo_mod, "_load_config", lambda p: {"vllm": {"url": "http://x", "model": "m"}}
    )

    with pytest.raises(SystemExit) as exc:
        slo_mod.cmd_slo(args)
    assert exc.value.code == 2


def test_cmd_slo_exits_0_when_all_pass(tmp_path, monkeypatch):
    """SLO met (max_n > 0) → exit code 0 (still success)."""
    from agentsurge.cli import slo as slo_mod

    args = _slo_args(output_dir=str(tmp_path))

    async def _fake_search(**kwargs):
        return [{"profile": "p", "max_n": 50, "boundary": {"ttft_p95_ms": 1000}}]

    monkeypatch.setattr(slo_mod, "_run_slo_search", _fake_search)
    monkeypatch.setattr(
        slo_mod, "_load_config", lambda p: {"vllm": {"url": "http://x", "model": "m"}}
    )

    # Should NOT raise — exit 0 is implicit (function returns)
    slo_mod.cmd_slo(args)


def test_cmd_sweep_exits_2_on_ttft_threshold_breach(tmp_path, monkeypatch):
    """sweep --ttft-threshold breach → exit code 2."""
    from agentsurge.cli import sweep as sweep_mod

    args = argparse.Namespace(
        config=None,
        vllm_url="http://x",
        model="m",
        output_dir=str(tmp_path),
        probe_levels=[10],
        source=None,
        tool_mode="off",
        no_cooldown=True,
        cooldown=0,
        max_error_rate=0.05,
        ttft_threshold=100.0,
        slo_ms=None,
        hint=None,
        n_turns=1,
        synthetic_tokens_per_turn=10,
        arrival_rate=None,
        arrival=None,
        preset=None,
        max_n=10,
        max_tokens=10,
        backend="mock",
        api="chat",
        no_metrics=True,
        seed=42,
        prefix_overlap_fraction=0.0,
        data_dir=None,
        trace_pool=None,
        multi_turn_only=False,
        client_concurrency=2,
        warm_up=0,
        ramp_duration=None,
        duration=None,
        tokenizer="auto",
        trust_remote_code=False,
        save_responses=False,
        no_stream=False,
        ignore_replay_output_length=True,
        thinking_budget=8192,
        enable_thinking=False,
        sanitize_truncated_tool_calls=False,
        continue_turn_on="never",
        continue_prompt=None,
        max_session_time=0.0,
        auto_finish_on_exhaust=False,
        max_retry_turns=0,
        max_budget_usd=0.0,
        input_price_per_mtok=0.0,
        output_price_per_mtok=0.0,
        stream_idle_timeout=0.0,
        retry_profile="default",
        enable_inflight_dump=False,
        lmcache=None,
        lmcache_url=None,
        think_time=None,
        interruption_rate=None,
        single_turn_ratio=None,
        tool_delay=None,
        tool_output_mode=None,
        synthetic_prompt_scale=None,
        context_distribution=None,
        skip_validation=True,
        workspace_dir=None,
        sandbox_dir=None,
        tool_env="safe",
        tool_call_parser_fallback="off",
        max_retries=0,
        use_model_reply_in_next_turn="auto",
        configs=None,
        profiles=None,
        tag=None,
    )

    # Fake runner result: TTFT p95 above threshold
    class _FakeSession:
        completed = True
        turns: list = []

    class _FakeResult:
        kv_util_peak = 0.5
        prefix_cache_hit_rate = 0.0
        lmcache_v1_deltas = None
        sessions = [_FakeSession()]

        def ttft_values(self):
            return [500.0, 500.0, 500.0]  # > 100ms threshold

    class _FakeRunner:
        def __init__(self, *a, **kw):
            pass

        async def run(self, sessions):
            return _FakeResult()

    monkeypatch.setattr(sweep_mod, "BenchmarkRunner", _FakeRunner)
    monkeypatch.setattr(sweep_mod, "_generate_sessions", lambda *a, **kw: [object()])
    monkeypatch.setattr(
        sweep_mod,
        "_load_config",
        lambda p: {
            "vllm": {"url": "http://x", "model": "m"},
            "workload": {"n_turns": 1, "synthetic_tokens_per_turn": 10},
            "runner": {"max_concurrency": 10},
        },
    )

    with pytest.raises(SystemExit) as exc:
        sweep_mod.cmd_sweep(args)
    assert exc.value.code == 2
