# SPDX-License-Identifier: MIT
"""Single source of truth for agentsurge preset resolution.

Loads preset definitions from configs/default.yaml and returns resolved
BenchmarkConfig-compatible kwargs.  Used by :meth:`BenchmarkConfig.from_yaml`,
the CLI, and standalone scripts to ensure consistent semantics.

Usage::

    from agentsurge.preset import resolve_preset
    kwargs = resolve_preset("measure")
    config = BenchmarkConfig(vllm_url=url, model=model, **kwargs)

Constants::

    from agentsurge.preset import PRESET_ALIASES, PRESET_NAMES
"""

from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG = Path(__file__).parent / "configs" / "default.yaml"

#: Alias map: short name → canonical preset name in YAML.
PRESET_ALIASES: dict[str, str] = {
    "measure": "measure-agent-heavy",
}

#: All user-visible preset names (including aliases).  This is the
#: authoritative list used for CLI ``--preset`` choices.
PRESET_NAMES: list[str] = [
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
    "measure",  # alias for measure-agent-heavy
]

#: Fields that are mapped from a preset YAML dict to BenchmarkConfig kwargs.
_PRESET_FIELDS: list[str] = [
    "think_time",
    "tool_delay",
    "max_retries",
    "max_tokens",
    "arrival_pattern",
    "interruption_rate",
    "single_turn_ratio",
    "arrival_rate",
    "gamma_cv",
]

#: Presets that auto-enable LMCache metrics collection.
LMCACHE_AUTO_PRESETS: tuple[str, ...] = ("stress-lmcache",)


def resolve_alias(preset_name: str) -> str:
    """Resolve a preset alias to its canonical name.

    Returns *preset_name* unchanged if it is not an alias.
    """
    return PRESET_ALIASES.get(preset_name, preset_name)


def resolve_preset_config(
    preset_name: str,
    *,
    yaml_cfg: dict | None = None,
    config_path: str | Path | None = None,
) -> dict:
    """Look up a preset in the YAML config and return its raw dict.

    Parameters
    ----------
    preset_name:
        User-facing preset name (may be an alias).
    yaml_cfg:
        Pre-loaded YAML config dict.  When provided, *config_path* is ignored.
    config_path:
        Path to a agentsurge YAML config file.  Falls back to the bundled
        ``configs/default.yaml``.

    Returns
    -------
    dict
        Raw preset dict from the YAML ``presets`` section (never empty -
        raises on unknown names).

    Raises
    ------
    ValueError
        If *preset_name* (after alias resolution) is not found in the config.
    FileNotFoundError
        If the config file does not exist.
    """
    resolved_name = resolve_alias(preset_name)

    if yaml_cfg is None:
        cfg_path = Path(config_path) if config_path else _DEFAULT_CONFIG
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        with open(cfg_path) as f:
            yaml_cfg = yaml.safe_load(f) or {}

    presets = yaml_cfg.get("presets", {})

    if resolved_name not in presets:
        available = ", ".join(sorted(presets.keys()))
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")

    return dict(presets[resolved_name])


def preset_to_kwargs(preset_dict: dict) -> dict:
    """Convert a raw preset dict to BenchmarkConfig-compatible kwargs.

    Handles type coercions (e.g. ``think_time`` list → tuple).  Only
    keys present in the preset are included in the output.

    Parameters
    ----------
    preset_dict:
        Raw preset dict (as returned by :func:`resolve_preset_config`).

    Returns
    -------
    dict
        Kwargs suitable for :class:`~agentsurge.types.BenchmarkConfig`.
    """
    kwargs: dict[str, Any] = {}

    for key in _PRESET_FIELDS:
        if key not in preset_dict:
            continue
        value = preset_dict[key]
        # think_time: YAML list → Python tuple
        if key == "think_time" and isinstance(value, list):
            value = tuple(value)
        kwargs[key] = value

    return kwargs


def resolve_preset(
    preset_name: str,
    config_path: str | Path | None = None,
    overrides: dict | None = None,
) -> dict:
    """Resolve a preset name to BenchmarkConfig-compatible kwargs.

    This is the main entry point for scripts.  Loads the preset from
    ``configs/default.yaml`` (or *config_path*), maps fields to
    :class:`~agentsurge.types.BenchmarkConfig` kwargs, then applies
    caller-supplied *overrides* on top.

    Returns a dict suitable for
    ``BenchmarkConfig(**resolve_preset(...), vllm_url=..., model=...)``.

    Parameters
    ----------
    preset_name:
        User-facing preset name (e.g. ``"measure"``).
    config_path:
        Optional path to a agentsurge YAML config file.
    overrides:
        Dict of caller overrides.  ``None``-valued entries are skipped.

    Raises
    ------
    ValueError
        If *preset_name* is not a known preset.
    FileNotFoundError
        If the config file does not exist.
    """
    raw = resolve_preset_config(preset_name, config_path=config_path)
    kwargs = preset_to_kwargs(raw)

    # Preserve the user-facing preset name (not the resolved alias)
    kwargs["preset"] = preset_name

    # Apply caller overrides (e.g. max_tokens, arrival_rate from CLI)
    if overrides:
        kwargs.update({k: v for k, v in overrides.items() if v is not None})

    return kwargs


def list_presets(
    config_path: str | Path | None = None,
) -> list[str]:
    """Return the list of available preset names from the YAML config.

    The returned list includes both canonical names and aliases.
    """
    cfg_path = Path(config_path) if config_path else _DEFAULT_CONFIG
    if not cfg_path.exists():
        return list(PRESET_NAMES)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}

    canonical = sorted(cfg.get("presets", {}).keys())
    # Add aliases that aren't already canonical names
    all_names = list(canonical)
    for alias in PRESET_ALIASES:
        if alias not in all_names:
            all_names.append(alias)
    return all_names
