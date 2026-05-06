"""Self-report parity regression guards.

The harness reports output_tokens from two independent sources for every
session: the CLI's terminal usage event (real tokenizer count) and the
harness's own EVENT_ASSISTANT_TEXT_DELTA count (one delta ≈ one token,
coarse heuristic). The canonical TurnResult.output_tokens is derived
from the usage event when present, falling back to the heuristic.

These tests pin the cross-check so a refactor of FrontendSessionRenderer's
metric assembly cannot silently break the parity contract — a regression
here would corrupt every downstream throughput chart without surfacing
as a test failure elsewhere.

Both tests reuse fixture-driven subprocess fakery from
test_frontend_provider_wiring.py to avoid recapturing real CLI sessions
for parity-only assertions."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agentsurge.frontends.runner import FrontendSessionRenderer

# Reuse helpers from the wiring tests — pytest does not enforce the
# single-underscore convention, so private-prefixed imports are fine here.
from tests.core.test_frontend_provider_wiring import (
    FIXTURES,
    _build_config,
    _make_fake_subprocess,
    _make_session,
)

# opencode fixture (v1.14.29) predates the usage-event capture and does not
# carry output_tokens in its result event; skip it rather than silently
# weakening the assertion.
_FRONTENDS_WITH_OUTPUT_TOKENS = [
    "codex",
    "claude",
    pytest.param(
        "opencode",
        marks=pytest.mark.skip(
            reason="opencode fixture v1.14.29 lacks output_tokens in usage event"
        ),
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("frontend_name", _FRONTENDS_WITH_OUTPUT_TOKENS)
async def test_turn_output_tokens_matches_provider_usage(
    tmp_path: Path, monkeypatch, frontend_name: str
) -> None:
    """When the CLI emits a terminal usage event with output_tokens, the
    canonical TurnResult.output_tokens must equal that self-report. The
    runner reaches this state by overwriting its delta-count estimate with
    the usage event value (runner.py:408-413). Pin the parity so a
    refactor that drops the overwrite silently regresses every benchmark
    output-token report to the heuristic."""
    fixture = FIXTURES[frontend_name]
    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        _make_fake_subprocess(fixture),
    )
    cfg = _build_config(frontend_name)
    renderer = FrontendSessionRenderer(cfg, tmp_path)
    result = await renderer.run(_make_session(f"s_{frontend_name}_parity"))

    fm = result.frontend_metrics
    assert fm is not None
    assert fm.provider_usage is not None, f"{frontend_name} fixture must carry usage"
    self_reported = fm.provider_usage.get("output_tokens")
    assert isinstance(self_reported, int) and self_reported > 0, (
        f"{frontend_name} provider_usage missing positive output_tokens: {fm.provider_usage}"
    )
    assert result.turns[0].output_tokens == self_reported, (
        f"{frontend_name}: TurnResult.output_tokens={result.turns[0].output_tokens} "
        f"diverged from provider_usage.output_tokens={self_reported}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("frontend_name", _FRONTENDS_WITH_OUTPUT_TOKENS)
async def test_provider_usage_at_least_delta_count(
    tmp_path: Path, monkeypatch, frontend_name: str
) -> None:
    """The CLI's output_tokens self-report (real tokenizer count) must be
    >= the harness's delta-count heuristic (one stream event ≈ one token).
    A heuristic exceeding the tokenizer count would mean we are over-
    counting deltas (e.g. counting non-text events as text), which would
    silently inflate visible-token throughput in older runs that lack a
    usage event."""
    fixture = FIXTURES[frontend_name]
    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        _make_fake_subprocess(fixture),
    )
    cfg = _build_config(frontend_name)
    renderer = FrontendSessionRenderer(cfg, tmp_path)
    result = await renderer.run(_make_session(f"s_{frontend_name}_bound"))

    fm = result.frontend_metrics
    assert fm is not None and fm.provider_usage is not None
    self_reported = fm.provider_usage.get("output_tokens")
    assert isinstance(self_reported, int) and self_reported > 0
    delta_count = fm.visible_output_tokens_estimate
    if delta_count is None:
        # Provider didn't emit streaming text deltas in the fixture (e.g.
        # codex's reasoning-only fixture). Bound is trivially satisfied.
        return
    assert self_reported >= delta_count, (
        f"{frontend_name}: provider_usage.output_tokens={self_reported} is "
        f"smaller than visible_output_tokens_estimate={delta_count} — heuristic "
        f"is over-counting deltas relative to the tokenizer"
    )
