# SPDX-License-Identifier: MIT
"""H3: BenchmarkConfig.from_yaml must wire every dataclass field.

Regression test: when a new field is added to BenchmarkConfig, from_yaml
must pick it up. Today inflight_dump is silently dropped.
"""

import dataclasses

from agentsurge.types.results import BenchmarkConfig

# Fields that have YAML-side custom keys / handling. Setting these as plain
# top-level keys is not the contract — we exclude them and verify the rest
# round-trip.
_SPECIAL_FIELDS = {
    "preset",  # set via from_yaml(preset=...) argument, not yaml key
    "extra_metrics_urls",  # populated from cfg["lmcache"]
    "extra_body",  # only via override
    "vllm_url",  # falls back to cfg["vllm"]["url"]
    "model",  # falls back to cfg["vllm"]["model"]
    "max_model_len",  # per-model override path
    "think_time",  # list → tuple conversion
    "arrival_rate",  # "auto" handling
    "sandbox_dir",  # legacy alias for workspace_dir
    "workspace_dir",  # tested separately via alias logic
    "continue_prompt",  # only emitted if present in yaml
}


def _yaml_value_for(field: dataclasses.Field) -> object:
    """Pick a non-default value compatible with the field type."""
    t = (
        field.type
        if isinstance(field.type, str)
        else getattr(field.type, "__name__", str(field.type))
    )
    name = field.name

    # Choose discriminated values for known constrained string fields.
    constrained = {
        "arrival_pattern": "constant",
        "api_type": "responses",
        "tool_mode": "real",
        "tool_call_parser_fallback": "deepseek_text",
        "tool_env": "inherit",
        "continue_turn_on": "text-only",
        "retry_profile": "codex",
        "tool_output_mode": "placeholder",
        "context_distribution": "uniform",
        "tool_call_parser": "qwen3_xml",
        "tokenizer_model": "some/model",
    }
    if name in constrained:
        return constrained[name]

    if "bool" in t:
        # Use True if default is False, False if default is True.
        return not bool(field.default)
    if "int" in t and "float" not in t:
        return 17
    if "float" in t:
        return 1.25
    if "str" in t:
        return "test_value"
    return None


def test_every_benchmarkconfig_field_is_wired_through_from_yaml(tmp_path):
    """Build a YAML config that sets every non-special field, load it via
    from_yaml, and assert every field on the resulting object matches."""
    import yaml

    fields = [f for f in dataclasses.fields(BenchmarkConfig) if f.name not in _SPECIAL_FIELDS]

    runner_cfg: dict = {}
    for f in fields:
        v = _yaml_value_for(f)
        if v is None:
            continue
        runner_cfg[f.name] = v

    # Required non-special: provide vllm + model so cfg constructs.
    yaml_doc = {
        "vllm": {"url": "http://test:8000", "model": "test/model"},
        "runner": runner_cfg,
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(yaml_doc))

    cfg = BenchmarkConfig.from_yaml(str(p))

    missing = []
    for f in fields:
        expected = _yaml_value_for(f)
        if expected is None:
            continue
        actual = getattr(cfg, f.name)
        if actual != expected:
            missing.append((f.name, expected, actual))

    assert not missing, "Fields not wired through BenchmarkConfig.from_yaml:\n" + "\n".join(
        f"  {n}: expected {e!r}, got {a!r}" for n, e, a in missing
    )


def test_inflight_dump_specifically_round_trips(tmp_path):
    """Spot-check inflight_dump (the field flagged in the audit)."""
    import yaml

    yaml_doc = {
        "vllm": {"url": "http://x:8000", "model": "m"},
        "runner": {"inflight_dump": True},
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(yaml_doc))
    cfg = BenchmarkConfig.from_yaml(str(p))
    assert cfg.inflight_dump is True
