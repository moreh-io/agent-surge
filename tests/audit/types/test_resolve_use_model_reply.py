# SPDX-License-Identifier: MIT
"""M3: _resolve_use_model_reply must run for both from_yaml and direct
construction so prod and unit-test paths agree.

Today, BenchmarkConfig(tool_mode="real") leaves use_model_reply_in_next_turn
at the dataclass default (False), but BenchmarkConfig.from_yaml(...) with
the same tool_mode resolves it to True. The two construction paths must
produce identical runtime semantics.
"""

from agentsurge.types.results import BenchmarkConfig


def test_direct_construction_matches_from_yaml_for_tool_mode_real(tmp_path):
    import yaml

    yaml_doc = {
        "vllm": {"url": "http://x:8000", "model": "m"},
        "runner": {"tool_mode": "real"},
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(yaml_doc))
    cfg_yaml = BenchmarkConfig.from_yaml(str(p))

    cfg_direct = BenchmarkConfig(vllm_url="http://x:8000", model="m", tool_mode="real")

    assert cfg_direct.use_model_reply_in_next_turn == cfg_yaml.use_model_reply_in_next_turn, (
        "Direct construction and from_yaml must agree on "
        "use_model_reply_in_next_turn for the same tool_mode."
    )


def test_direct_construction_replay_resolves_to_false():
    cfg = BenchmarkConfig(vllm_url="http://x", model="m", tool_mode="replay")
    # Per _resolve_use_model_reply: tool_mode == "replay" → False default
    assert cfg.use_model_reply_in_next_turn is False


def test_direct_construction_real_resolves_to_true():
    cfg = BenchmarkConfig(vllm_url="http://x", model="m", tool_mode="real")
    assert cfg.use_model_reply_in_next_turn is True


def test_explicit_override_is_respected():
    """An explicit override (non-None) wins over the resolver default."""
    cfg = BenchmarkConfig(
        vllm_url="http://x",
        model="m",
        tool_mode="real",
        use_model_reply_in_next_turn=False,
    )
    assert cfg.use_model_reply_in_next_turn is False
