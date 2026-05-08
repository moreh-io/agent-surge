"""Tests for agentsurge.cli.slo - SLO boundary binary search."""

import pytest


class TestParseProfiles:
    def test_default_profiles(self):
        """No --profiles flag returns default 5 profiles."""
        import argparse

        from agentsurge.cli.slo import _parse_profiles

        args = argparse.Namespace(profiles=None)
        profiles = _parse_profiles(args)
        assert len(profiles) == 5
        # Each is (turns, tokens, max_tokens, label)
        for p in profiles:
            assert len(p) == 4
            assert isinstance(p[0], int)
            assert isinstance(p[3], str)

    def test_custom_two_part(self):
        """Two-part spec 'turns,tokens' defaults max_tokens to 256."""
        import argparse

        from agentsurge.cli.slo import _parse_profiles

        args = argparse.Namespace(profiles=["5,1000"])
        profiles = _parse_profiles(args)
        assert len(profiles) == 1
        turns, tokens, max_tokens, label = profiles[0]
        assert turns == 5
        assert tokens == 1000
        assert max_tokens == 256
        assert "5" in label and "1000" in label

    def test_custom_three_part(self):
        """Three-part spec 'turns,tokens,max_tokens'."""
        import argparse

        from agentsurge.cli.slo import _parse_profiles

        args = argparse.Namespace(profiles=["10,2000,512"])
        profiles = _parse_profiles(args)
        turns, tokens, max_tokens, label = profiles[0]
        assert turns == 10
        assert tokens == 2000
        assert max_tokens == 512

    def test_invalid_spec(self):
        """Single-part spec should raise SystemExit."""
        import argparse

        from agentsurge.cli.slo import _parse_profiles

        args = argparse.Namespace(profiles=["invalid"])
        with pytest.raises((SystemExit, ValueError)):
            _parse_profiles(args)

    def test_multiple_profiles(self):
        """Multiple profile specs are all parsed."""
        import argparse

        from agentsurge.cli.slo import _parse_profiles

        args = argparse.Namespace(profiles=["3,500", "7,2200,512"])
        profiles = _parse_profiles(args)
        assert len(profiles) == 2
