"""H3: Sibling artefacts can race-overwrite on collision suffix collide.

Audit: cli.md H3 — `_collision_suffix` returns 4 hex chars (16 bits). Two
parallel runs landing on the same wall-second + colliding hash overwrite
each other's `_turns.csv`, `_summary.csv`, `_timeline.png`, `_hf/`. Fix
options: (a) raise entropy, or (b) check for existence before write.

This test pins option (a): collision suffix must have ≥ 8 hex chars of
entropy, AND `_output_path` must refuse to overwrite an existing target.
"""

from __future__ import annotations

import os
from unittest.mock import patch


def test_collision_suffix_has_min_8_hex_entropy():
    """Bump uuid suffix from 4 to ≥ 8 hex chars (≥ 32 bits)."""
    from agentsurge.cli._helpers import _collision_suffix

    s = _collision_suffix()
    # Form: "<pid>_<hex>"; require the hex tail to be at least 8 chars.
    parts = s.rsplit("_", 1)
    assert len(parts) == 2, f"unexpected suffix format: {s!r}"
    hex_tail = parts[1]
    assert len(hex_tail) >= 8, (
        f"collision suffix entropy too low (got {len(hex_tail)} hex chars; need >= 8). "
        f"See cli.md H3."
    )
    # And the hex tail must be valid hex.
    int(hex_tail, 16)


def test_output_path_refuses_overwrite_on_collision(tmp_path):
    """When the candidate path exists, _output_path must not silently overwrite.
    It either re-rolls the suffix or raises FileExistsError.
    """
    from agentsurge.cli._helpers import _output_path

    # Force a deterministic suffix collision by patching uuid + datetime.
    with (
        patch("agentsurge.cli._helpers._collision_suffix", return_value="999_aaaaaaaa"),
        patch("agentsurge.cli._helpers.datetime") as fake_dt,
    ):

        class _FakeDT:
            @staticmethod
            def now():
                class _N:
                    @staticmethod
                    def strftime(fmt):
                        return "20260427_120000"

                return _N()

        fake_dt.now = _FakeDT.now

        first = _output_path(str(tmp_path), "run")
        # Touch the target so the next call would overwrite.
        with open(first, "w") as f:
            f.write("{}")

        # Even with the same patched suffix/timestamp, _output_path must
        # not return a path that exists already.
        try:
            second = _output_path(str(tmp_path), "run")
        except FileExistsError:
            # Acceptable: bail loudly rather than overwrite.
            return

    assert second != first or not os.path.exists(first), (
        "second _output_path call returned a path that overwrites the first"
    )
