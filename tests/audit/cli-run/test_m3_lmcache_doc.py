"""M3: --lmcache tri-state under-documented.

Audit: cli.md M3 — `--lmcache` is `store_true, default=None` so omitting
both flags is also auto-enable when --lmcache-url is set or preset is in
LMCACHE_AUTO_PRESETS. The help text doesn't say so.

Pin: --lmcache help text must mention auto-enable conditions; the
`--no-lmcache` help must indicate it overrides auto-enable too.
"""

from __future__ import annotations

import argparse


def _build_parser():
    """Build the same parser main() builds, but skip dispatch."""
    from agentsurge.cli._app import _run_args

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", "-c", default=None)
    p = argparse.ArgumentParser(parents=[common, _run_args()])
    return p


def test_lmcache_help_mentions_tri_state():
    """The --lmcache help string must mention that omitting it can auto-enable."""
    parser = _build_parser()
    actions = {tuple(a.option_strings): a for a in parser._actions}
    lmcache_action = actions.get(("--lmcache",))
    assert lmcache_action is not None, "missing --lmcache flag"
    help_text = (lmcache_action.help or "").lower()
    # Must explicitly mention all three triggers: omit-both / --lmcache-url / preset
    assert "auto" in help_text, f"--lmcache help must mention auto-enable behaviour: {help_text!r}"
    # Must name the URL-set trigger
    assert "lmcache-url" in help_text or "--lmcache-url" in help_text, (
        f"--lmcache help must mention the --lmcache-url auto-enable trigger: {help_text!r}"
    )
    assert "preset" in help_text, (
        f"--lmcache help must mention preset auto-enable trigger: {help_text!r}"
    )


def test_no_lmcache_help_states_override():
    """The --no-lmcache help must indicate it overrides auto-enable."""
    parser = _build_parser()
    actions = {tuple(a.option_strings): a for a in parser._actions}
    no_lmcache = actions.get(("--no-lmcache",))
    assert no_lmcache is not None
    help_text = (no_lmcache.help or "").lower()
    assert "override" in help_text or "auto" in help_text, (
        f"--no-lmcache help must clarify it overrides auto-enable: {help_text!r}"
    )
