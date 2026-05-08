"""Trimmed preset tests: resolve, alias, list, single-source-of-truth."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentsurge.preset import (
    PRESET_ALIASES,
    PRESET_NAMES,
    list_presets,
    preset_to_kwargs,
    resolve_alias,
    resolve_preset,
    resolve_preset_config,
)

ALL_PRESET_NAMES = [
    "stress-default",
    "stress-prefix-cache",
    "stress-lmcache",
    "simulate-default",
    "simulate-production",
    "simulate-openhands",
    "simulate-swe-agent",
    "agent-light",
    "agent-heavy",
    "measure-agent-light",
    "measure-agent-heavy",
    "measure",
]


# ===========================================================================
# Constants and aliases
# ===========================================================================


def test_preset_names_complete():
    for name in ALL_PRESET_NAMES:
        assert name in PRESET_NAMES, f"{name!r} missing from PRESET_NAMES"


def test_preset_aliases():
    assert PRESET_ALIASES["measure"] == "measure-agent-heavy"
    assert resolve_alias("measure") == "measure-agent-heavy"
    assert resolve_alias("stress-default") == "stress-default"


# ===========================================================================
# resolve_preset_config
# ===========================================================================


def test_resolve_preset_config_returns_dict():
    raw = resolve_preset_config("stress-default")
    assert isinstance(raw, dict)
    assert "max_tokens" in raw
    assert "arrival_pattern" in raw
    assert raw["max_tokens"] == 256
    assert raw["arrival_pattern"] == "burst"


def test_resolve_preset_config_unknown_raises():
    with pytest.raises(ValueError, match="Unknown preset"):
        resolve_preset_config("nonexistent-preset")


def test_resolve_preset_config_preloaded_yaml():
    cfg = {"presets": {"my-preset": {"max_tokens": 512, "arrival_pattern": "poisson"}}}
    raw = resolve_preset_config("my-preset", yaml_cfg=cfg)
    assert raw["max_tokens"] == 512


# ===========================================================================
# preset_to_kwargs
# ===========================================================================


def test_think_time_list_converted_to_tuple():
    kwargs = preset_to_kwargs({"think_time": [2.0, 1.0]})
    assert kwargs["think_time"] == (2.0, 1.0) and isinstance(kwargs["think_time"], tuple)


def test_known_fields_passed_through():
    kwargs = preset_to_kwargs({"max_tokens": 4096})
    assert kwargs["max_tokens"] == 4096
    assert preset_to_kwargs({}) == {}


def test_unknown_fields_ignored():
    assert preset_to_kwargs({"unknown_field": "ignored"}) == {}


# ===========================================================================
# resolve_preset (all presets in one loop)
# ===========================================================================


EXPECTED = {
    "stress-default": {"arrival_pattern": "burst"},
    "stress-prefix-cache": {"arrival_pattern": "burst"},
    "stress-lmcache": {"arrival_pattern": "poisson", "single_turn_ratio": 0.3},
    "simulate-default": {"arrival_pattern": "gamma", "tool_delay": True},
    "simulate-production": {"arrival_pattern": "gamma", "tool_delay": False},
    "simulate-openhands": {"arrival_pattern": "gamma", "tool_delay": True},
    "simulate-swe-agent": {"arrival_pattern": "gamma", "tool_delay": True, "max_tokens": 512},
    "agent-light": {"max_tokens": 4096},
    "agent-heavy": {"max_tokens": 16384},
    "measure-agent-light": {"max_tokens": 32768},
    "measure-agent-heavy": {"max_tokens": 32768},
    "measure": {"max_tokens": 32768},
}


@pytest.mark.parametrize("preset_name", ALL_PRESET_NAMES)
def test_all_presets_resolve(preset_name):
    result = resolve_preset(preset_name)
    assert result["preset"] == preset_name or result["preset"] in PRESET_ALIASES.values()
    assert "max_tokens" in result, f"{preset_name}: missing max_tokens"
    assert result["max_tokens"] > 0, f"{preset_name}: max_tokens must be positive"
    for key, expected_val in EXPECTED.get(preset_name, {}).items():
        assert result[key] == expected_val, (
            f"{preset_name}: {key}={result[key]!r}, expected {expected_val!r}"
        )


def test_measure_preset_core_fields():
    r = resolve_preset("measure")
    assert isinstance(r["think_time"], tuple) and len(r["think_time"]) == 2
    assert r["max_tokens"] > 256
    assert r["max_tokens"] != 256


def test_preset_overrides():
    r = resolve_preset("measure", overrides={"max_tokens": 4096})
    assert r["max_tokens"] == 4096


# ===========================================================================
# list_presets
# ===========================================================================


def test_list_presets():
    names = list_presets()
    for canonical in ("stress-default", "measure-agent-heavy", "agent-light", "measure"):
        assert canonical in names
    assert len(names) == len(PRESET_NAMES)


# ===========================================================================
# Configs layout
# ===========================================================================


def configs_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "agentsurge" / "configs"


def test_configs_layout():
    """Packaging test: verify configs/ directory ships the expected layout."""
    root = configs_root()
    assert root.is_dir()
    assert (root / "default.yaml").is_file()
    assert (root / "examples").is_dir()
    entries = {p.name for p in root.iterdir()}
    unexpected = entries - {"default.yaml", "examples"}
    assert not unexpected, f"Unexpected entries in configs/: {unexpected}"
