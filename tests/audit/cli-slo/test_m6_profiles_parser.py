"""M6: `_parse_profiles` raises ValueError on bad input.

Audit: cli.md M6 — `_parse_profiles` does `int(parts[0])` without try/except,
so non-numeric tokens dump a traceback. Also `len==1` is silently rejected
with a generic message that doesn't name the offender clearly.
"""

from __future__ import annotations

import argparse

import pytest


def _args(profiles):
    return argparse.Namespace(profiles=profiles)


def test_non_numeric_profile_token_exits_clean(capsys):
    """`foo,1500` must produce a clean SystemExit, not a ValueError traceback."""
    from agentsurge.cli.slo import _parse_profiles

    with pytest.raises(SystemExit):
        _parse_profiles(_args(["foo,1500"]))
    out = capsys.readouterr()
    msg = (out.err + out.out).lower()
    assert "foo" in msg or "invalid" in msg, f"error must name the bad input: {msg!r}"


def test_single_field_profile_exits_clean(capsys):
    """`5` (only one field) must exit cleanly with the offending spec named."""
    from agentsurge.cli.slo import _parse_profiles

    with pytest.raises(SystemExit):
        _parse_profiles(_args(["5"]))
    out = capsys.readouterr()
    msg = (out.err + out.out).lower()
    assert "5" in msg or "invalid" in msg


def test_empty_profile_exits_clean():
    """Empty string profile must exit cleanly."""
    from agentsurge.cli.slo import _parse_profiles

    with pytest.raises(SystemExit):
        _parse_profiles(_args([""]))


def test_too_many_fields_exits_clean():
    """`1,2,3,4` (4 fields) must exit cleanly."""
    from agentsurge.cli.slo import _parse_profiles

    with pytest.raises(SystemExit):
        _parse_profiles(_args(["1,2,3,4"]))


def test_valid_profile_still_parses():
    """Sanity: a valid profile still parses correctly."""
    from agentsurge.cli.slo import _parse_profiles

    profs = _parse_profiles(_args(["3,1500,256"]))
    assert profs == [(3, 1500, 256, "3T1500x256")]
