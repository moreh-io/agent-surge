"""M5: No backend preflight; fails mid-run on unreachable URL.

Audit: cli.md M5 — When --backend is omitted, `_build_backend` returns
None and the runner discovers connection failures hundreds of ms in.
Pin: an unreachable URL must produce a fast, dedicated error path
(transport error, exit 3) before sessions are dispatched.

Mock backend bypasses the preflight (no URL); only HTTP path is
checked. We test by mocking the preflight helper.
"""

from __future__ import annotations


def test_backend_preflight_function_exists():
    """Pin the contract: there must be a `_backend_preflight` helper."""
    from agentsurge.cli import run as run_mod

    assert hasattr(run_mod, "_backend_preflight"), (
        "Expected _backend_preflight helper to be defined in cli.run (see cli.md M5)."
    )


def test_backend_preflight_returns_false_on_connection_refused(monkeypatch):
    """The preflight must return False on connection failure (no traceback)."""
    from agentsurge.cli import run as run_mod

    # Use a port that is almost certainly closed.
    ok = run_mod._backend_preflight("http://127.0.0.1:1", timeout=0.5)
    assert ok is False


def test_cmd_run_exits_3_on_preflight_failure(tmp_path, monkeypatch, capsys):
    """When the preflight fails, cmd_run exits 3 (transport error)."""
    import argparse

    from agentsurge.cli import run as run_mod

    monkeypatch.setattr(run_mod, "_backend_preflight", lambda url, timeout=5.0: False)
    monkeypatch.setattr(
        run_mod,
        "_load_config",
        lambda p: {
            "vllm": {"url": "http://unreachable:9999", "model": "m"},
            "workload": {"n_turns": 1, "synthetic_tokens_per_turn": 16},
            "runner": {},
        },
    )

    args = argparse.Namespace(
        config=None,
        vllm_url=None,
        model=None,
        backend=None,  # default HTTP path triggers preflight
        output_dir=str(tmp_path),
        n_sessions=1,
        n_turns=1,
        synthetic_tokens_per_turn=16,
        no_metrics=False,
        seed=42,
        no_stream=False,
        tool_mode="off",
        workload_file=None,
        trace_pool=None,
        source=None,
        preset=None,
        api="chat",
        max_tokens=16,
        arrival_rate=None,
        arrival=None,
        client_concurrency=2,
        warm_up=0,
        ramp_duration=None,
        duration=None,
        tokenizer="none",
        ignore_replay_output_length=True,
        trust_remote_code=False,
        save_responses=False,
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
        local_path=None,
        data_dir=None,
        prefix_overlap_fraction=0.0,
        multi_turn_only=False,
        make_dataset=False,
        tool_max_turns=None,
    )

    import pytest

    with pytest.raises(SystemExit) as exc:
        run_mod.cmd_run(args)
    assert exc.value.code == 3, f"expected exit 3 on preflight failure, got {exc.value.code}"
    err = (capsys.readouterr().err + capsys.readouterr().out).lower()
    assert "unreachable" in err or "preflight" in err or "connect" in err
