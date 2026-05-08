"""Trimmed loader tests: per-loader logic, registry, auto-discover."""

from __future__ import annotations

import importlib
import json
from unittest.mock import MagicMock, patch

import pytest

from agentsurge.loaders.registry import loader_registry

# ===========================================================================
# Mock samples
# ===========================================================================

MOCK_SAMPLE = {
    "instance_id": "test-001",
    "messages": [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": "Fix the bug in utils.py"},
        {
            "role": "assistant",
            "content": "I'll look at the file.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "utils.py"}'},
                }
            ],
        },
        {
            "role": "tool",
            "content": "def foo():\n    return 1",
            "name": "read_file",
            "tool_call_id": "call_1",
        },
        {"role": "assistant", "content": "Found the issue. Fixing now."},
        {"role": "user", "content": "Apply the fix."},
    ],
    "repo": "owner/repo",
}

MOCK_SWE_SAMPLE = {
    "instance_id": "django__django-11111",
    "problem_statement": "Fix the null pointer in views.py",
    "repo": "django/django",
    "base_commit": "abc123",
    "patch": "diff --git a/views.py ...",
    "test_patch": "def test_foo(): ...",
    "FAIL_TO_PASS": ["test_foo"],
    "PASS_TO_PASS": ["test_bar"],
}

MOCK_ABC_SAMPLE = {
    "id": "abc-001",
    "instruction": "Implement a REST API endpoint for user login.",
    "output": "def login(request): ...",
    "category": "api",
    "tags": ["rest", "auth"],
}

MOCK_SWE_EVO_SAMPLE = {
    "instance_id": "evo-001",
    "problem_statement": "Migrate auth to OAuth2.",
    "patch": "diff --git a/auth.py ...",
    "repo": "org/project",
}

MOCK_FEATURE_SAMPLE = {
    "instance_id": "feat-001",
    "problem_statement": "Add dark mode toggle.",
    "patch": "diff --git a/settings.py ...",
    "repo": "org/app",
}

MOCK_SWE_SMITH_SAMPLE = {
    "instance_id": "smith-001",
    "messages": [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": "Fix the bug"},
        {
            "role": "assistant",
            "content": "Analyzing.",
            "thought": "Need to read file",
            "action": "read",
            "tool_calls": [
                {
                    "id": "tc1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
                }
            ],
        },
        {"role": "tool", "content": "def foo(): pass", "name": "read_file"},
        {"role": "assistant", "content": "Fixed.", "agent": "coder", "message_type": "response"},
    ],
    "repo": "test/repo",
}


def _hf_mock(samples):
    m = MagicMock()
    m.load_dataset.return_value = iter(samples)
    return m


# ===========================================================================
# All HF task loaders: single parametrized test
# ===========================================================================

TASK_LOADER_PARAMS = [
    (
        "agentsurge.loaders.swe_bench",
        "SWEBenchLoader",
        MOCK_SWE_SAMPLE,
        "swe",
        "null pointer",
        "diff --git",
    ),
    (
        "agentsurge.loaders.abc_bench",
        "ABCBenchLoader",
        MOCK_ABC_SAMPLE,
        "abc-bench",
        "REST API",
        "login",
    ),
    (
        "agentsurge.loaders.swe_evo",
        "SWEEvoLoader",
        MOCK_SWE_EVO_SAMPLE,
        "swe-evo",
        "OAuth2",
        "diff",
    ),
    (
        "agentsurge.loaders.featurebench",
        "FeatureBenchLoader",
        MOCK_FEATURE_SAMPLE,
        "featurebench",
        "dark mode",
        "diff",
    ),
]


@pytest.mark.parametrize(
    "module,cls_name,sample,expected_name,prompt_sub,ref_sub",
    TASK_LOADER_PARAMS,
    ids=["swe", "abc", "swe-evo", "featurebench"],
)
def test_task_loader_load_and_content(module, cls_name, sample, expected_name, prompt_sub, ref_sub):
    """Each task loader loads one task with correct name, prompt, and reference."""
    mod = importlib.import_module(module)
    cls = getattr(mod, cls_name)
    with patch.dict("sys.modules", {"datasets": _hf_mock([sample])}):
        tasks = list(cls().load())
    assert len(tasks) == 1, f"{cls_name}: expected 1 task"
    assert prompt_sub in tasks[0].prompt, f"{cls_name}: prompt missing '{prompt_sub}'"
    assert ref_sub in tasks[0].reference, f"{cls_name}: reference missing '{ref_sub}'"
    assert expected_name in cls().name.lower(), f"{cls_name}: name mismatch"
    if "instance_id" in sample:
        assert tasks[0].metadata.get("instance_id") == sample["instance_id"]


# ===========================================================================
# OpenHandsLoader
# ===========================================================================


class TestOpenHandsLoader:
    def test_local_path_loads_jsonl(self, tmp_path):
        from agentsurge.loaders.openhands import OpenHandsLoader

        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text(
            json.dumps(MOCK_SAMPLE)
            + "\n"
            + json.dumps({**MOCK_SAMPLE, "instance_id": "test-002"})
            + "\n"
        )
        loader = OpenHandsLoader(local_path=str(jsonl_file))
        trajs = list(loader.load())
        assert len(trajs) == 2
        assert trajs[0].trajectory_id == "test-001"

    def test_messages_parsed_with_tool_calls(self, tmp_path):
        from agentsurge.loaders.openhands import OpenHandsLoader

        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text(json.dumps(MOCK_SAMPLE) + "\n")
        loader = OpenHandsLoader(local_path=str(jsonl_file))
        trajs = list(loader.load())
        turns = trajs[0].sessions[0].turns
        roles = [t.role for t in turns]
        assert "user" in roles and "assistant" in roles
        assert any(t.tool_calls for t in turns)

    def test_limit_parameter(self, tmp_path):
        from agentsurge.loaders.openhands import OpenHandsLoader

        jsonl_file = tmp_path / "data.jsonl"
        lines = "\n".join(json.dumps({**MOCK_SAMPLE, "instance_id": f"test-{i}"}) for i in range(5))
        jsonl_file.write_text(lines + "\n")
        loader = OpenHandsLoader(local_path=str(jsonl_file))
        assert len(list(loader.load(limit=2))) == 2


# ===========================================================================
# SWESmithLoader
# ===========================================================================


class TestSWESmithLoader:
    def test_local_path_loads_and_parses(self, tmp_path):
        from agentsurge.loaders.swe_smith import SWESmithLoader

        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text(json.dumps(MOCK_SWE_SMITH_SAMPLE) + "\n")
        loader = SWESmithLoader(local_path=str(jsonl_file))
        trajs = list(loader.load())
        assert len(trajs) == 1
        turns = trajs[0].sessions[0].turns
        assert len(turns) == 5
        # Tool calls preserved
        tc_turns = [t for t in turns if t.tool_calls]
        assert len(tc_turns) == 1
        # Metadata from extra fields
        assert turns[2].metadata["thought"] == "Need to read file"

    def test_list_content_extracted(self, tmp_path):
        from agentsurge.loaders.swe_smith import SWESmithLoader

        sample = {
            "instance_id": "smith-002",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {"type": "text", "text": "world"},
                    ],
                },
                {"role": "assistant", "content": "ok"},
            ],
        }
        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text(json.dumps(sample) + "\n")
        trajs = list(SWESmithLoader(local_path=str(jsonl_file)).load())
        assert "hello" in trajs[0].sessions[0].turns[0].content

    def test_invalid_messages_handled(self, tmp_path):
        """String messages, invalid JSON, unknown roles, non-dict entries all handled."""
        from agentsurge.loaders.swe_smith import SWESmithLoader

        for sample, expected_turns in [
            ({"instance_id": "a", "messages": "not valid json"}, 0),
            (
                {
                    "instance_id": "b",
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {"role": "unknown_role", "content": "skip"},
                        {"role": "assistant", "content": "bye"},
                    ],
                },
                2,
            ),
            (
                {
                    "instance_id": "c",
                    "messages": ["just a string", {"role": "user", "content": "valid"}],
                },
                1,
            ),
        ]:
            jsonl_file = tmp_path / f"data_{sample['instance_id']}.jsonl"
            jsonl_file.write_text(json.dumps(sample) + "\n")
            trajs = list(SWESmithLoader(local_path=str(jsonl_file)).load())
            assert len(trajs[0].sessions[0].turns) == expected_turns, (
                f"sample {sample['instance_id']}: expected {expected_turns} turns"
            )

    def test_split_in_name(self):
        from agentsurge.loaders.swe_smith import SWESmithLoader

        assert SWESmithLoader.name == "swe-smith"
        assert SWESmithLoader(split="xml").display_name == "swe-smith-xml"

    def test_derives_repo_metadata_from_compact_instance_id(self, tmp_path):
        from agentsurge.loaders.swe_smith import SWESmithLoader

        sample = {
            "instance_id": "django-money__django-money.835c1ab8.func_pm_ctrl_shuffle__viqnyl9u",
            "messages": [{"role": "user", "content": "hi"}],
        }
        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text(json.dumps(sample) + "\n")
        traj = list(SWESmithLoader(local_path=str(jsonl_file)).load())[0]
        meta = traj.sessions[0].metadata
        assert meta["repo"] == "django-money/django-money"
        assert meta["base_commit"] == "835c1ab8"

    def test_tool_result_matches_explicit_id_before_fifo_name(self, tmp_path):
        from agentsurge.loaders.swe_smith import SWESmithLoader

        sample = {
            "instance_id": "smith-tools",
            "messages": [
                {"role": "user", "content": "fix"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_a",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"},
                        },
                        {
                            "id": "call_b",
                            "type": "function",
                            "function": {"name": "execute_bash", "arguments": "{}"},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "call_b", "content": "pytest ok"},
                {"role": "tool", "tool_call_id": "call_a", "content": "file text"},
            ],
        }
        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text(json.dumps(sample) + "\n")

        turns = list(SWESmithLoader(local_path=str(jsonl_file)).load())[0].sessions[0].turns

        assert turns[2].tool_call_id == "call_b"
        assert turns[2].name == "execute_bash"
        assert turns[3].tool_call_id == "call_a"
        assert turns[3].name == "read_file"


# ===========================================================================
# ABCBenchLoader unique test
# ===========================================================================


def test_abc_bench_metadata_excludes_mapped_fields():
    from agentsurge.loaders.abc_bench import ABCBenchLoader

    with patch.dict("sys.modules", {"datasets": _hf_mock([MOCK_ABC_SAMPLE])}):
        tasks = list(ABCBenchLoader().load())
    assert "instruction" not in tasks[0].metadata
    assert "category" in tasks[0].metadata
    assert tasks[0].metadata["source"] == "abc-bench"
    # task_id is preserved so the sandbox route can resolve the extracted dir.
    assert tasks[0].metadata["task_id"] == "abc-001"
    assert tasks[0].task_id == "abc-001"


def test_abc_bench_loader_uses_real_task_id_key():
    """Real HF schema uses ``task_id``; loader must pick it over the fallback index."""
    from agentsurge.loaders.abc_bench import ABCBenchLoader

    sample = {
        "task_id": "task_15dkatz_official_joke_api__metadata",
        "tags": ["JavaScript"],
        "category": "Specialized",
        "instruction": "...",
    }
    with patch.dict("sys.modules", {"datasets": _hf_mock([sample])}):
        tasks = list(ABCBenchLoader().load())
    assert tasks[0].task_id == "task_15dkatz_official_joke_api__metadata"
    assert tasks[0].metadata["task_id"] == "task_15dkatz_official_joke_api__metadata"


# ===========================================================================
# Registry: discover and get_loader
# ===========================================================================


def test_discover_includes_all_builtin_loaders():
    names = loader_registry.discover()
    assert isinstance(names, list)
    assert names == sorted(names)
    expected_set = {
        "openhands",
        "swe-smith",
        "swe-bench-verified",
        "abc-bench",
        "swe-evo",
        "featurebench",
    }
    assert expected_set.issubset(set(names)), f"Missing builtins: {expected_set - set(names)}"


def test_get_loader_returns_class():
    from agentsurge.loaders import get_loader

    for name in ("openhands", "swe-smith", "swe-bench-verified"):
        cls = get_loader(name)
        assert cls is not None, f"get_loader({name!r}) returned None"
        assert hasattr(cls, "load"), f"{name}: loader class missing load method"
        assert hasattr(cls, "name"), f"{name}: loader class missing 'name' attribute"
        assert callable(cls.load), f"{name}: load is not callable"


def test_get_loader_unknown_raises():
    from agentsurge.loaders import get_loader

    with pytest.raises((KeyError, ValueError)):
        get_loader("nonexistent-loader-xyz")


# ---------------------------------------------------------------------------
# supports_multi_turn capability flag
# ---------------------------------------------------------------------------


class TestMultiTurnCapability:
    def test_openhands_session_metadata_tags_capability(self):
        from agentsurge.loaders.openhands import OpenHandsLoader

        sample = {
            "instance_id": "tid-1",
            "trajectory": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "second"},
            ],
        }
        loader = OpenHandsLoader(local_path="/tmp/does-not-exist.jsonl")
        traj = loader._parse_sample(sample, 0)

        assert traj.metadata["supports_multi_turn"] is True
        assert traj.sessions[0].metadata["supports_multi_turn"] is True


# ===========================================================================
# SWESmithLoader.download uses self.split (H9)
# ===========================================================================


# ===========================================================================
# H10: parse errors must not be masked as HF auth failures
# ===========================================================================


def test_base_hf_task_loader_parse_error_not_masked_as_auth():
    """_make_task KeyError must propagate natively, not as RuntimeError."""
    from agentsurge.loaders.swe_bench import SWEBenchLoader

    malformed = {"instance_id": "bad-002"}

    mock_datasets = MagicMock()
    mock_datasets.load_dataset.return_value = iter([malformed])

    loader = SWEBenchLoader()

    with patch.dict("sys.modules", {"datasets": mock_datasets}):
        with patch.object(
            SWEBenchLoader,
            "_make_task",
            side_effect=KeyError("problem_statement"),
        ):
            with pytest.raises(KeyError):
                list(loader._load_hf())


# ===========================================================================
# M18: blank-line crash in _load_local JSONL iteration
# ===========================================================================


def test_swe_smith_loader_skips_blank_lines(tmp_path):
    """SWESmithLoader._load_local must not crash on blank separator lines."""
    from agentsurge.loaders.swe_smith import SWESmithLoader

    jsonl_file = tmp_path / "blank.jsonl"
    jsonl_file.write_text(
        json.dumps(MOCK_SWE_SMITH_SAMPLE) + "\n"
        "\n" + json.dumps({**MOCK_SWE_SMITH_SAMPLE, "instance_id": "smith-002"}) + "\n"
    )
    trajs = list(SWESmithLoader(local_path=str(jsonl_file)).load())
    assert len(trajs) == 2
    assert trajs[0].trajectory_id == "smith-001"
    assert trajs[1].trajectory_id == "smith-002"


def test_base_hf_task_loader_skips_blank_lines(tmp_path):
    """BaseHFTaskLoader._load_local must not crash on blank separator lines."""
    from agentsurge.loaders.swe_bench import SWEBenchLoader

    jsonl_file = tmp_path / "blank.jsonl"
    jsonl_file.write_text(
        json.dumps(MOCK_SWE_SAMPLE) + "\n"
        "\n" + json.dumps({**MOCK_SWE_SAMPLE, "instance_id": "django__django-22222"}) + "\n"
    )
    tasks = list(SWEBenchLoader(local_path=str(jsonl_file)).load())
    assert len(tasks) == 2
    assert tasks[0].task_id == "django__django-11111"
    assert tasks[1].task_id == "django__django-22222"


# ===========================================================================
# M19: SWESmithLoader._load_hf must wrap load_dataset with HF auth hint
# ===========================================================================


def test_swe_smith_load_hf_auth_error_raises_runtime_with_hint():
    """load_dataset auth/connection error must re-raise as RuntimeError with login hint."""
    from agentsurge.loaders.swe_smith import SWESmithLoader

    mock_datasets = MagicMock()
    mock_datasets.load_dataset.side_effect = ValueError("401 Unauthorized")

    loader = SWESmithLoader()
    with patch.dict("sys.modules", {"datasets": mock_datasets}):
        with pytest.raises(RuntimeError) as exc_info:
            list(loader._load_hf())

    assert "huggingface-cli login" in str(exc_info.value)


def test_swe_smith_load_hf_parse_error_not_masked_as_auth():
    """_parse_sample error must propagate natively, not as RuntimeError (narrow wrap)."""
    from agentsurge.loaders.swe_smith import SWESmithLoader

    mock_datasets = MagicMock()
    mock_datasets.load_dataset.return_value = iter([MOCK_SWE_SMITH_SAMPLE])

    loader = SWESmithLoader()
    with patch.dict("sys.modules", {"datasets": mock_datasets}):
        with patch.object(
            SWESmithLoader,
            "_parse_sample",
            side_effect=KeyError("messages"),
        ):
            with pytest.raises(KeyError):
                list(loader._load_hf())
