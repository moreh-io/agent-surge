"""Tests for agentsurge.tool_call module."""

from __future__ import annotations

import json

import pytest

from agentsurge.tool_call import (
    CODING_TOOLS,
    TOOL_NAMES,
    ExecutionWorkspace,
    Sandbox,
    ToolCallCheck,
    TurnValidation,
    _looks_like_garbage,
    build_tool_response_messages,
    build_tool_response_messages_async,
    real_execute,
    real_execute_async,
    synthesize_tool_calls_from_text,
    validate_tool_calls,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_coding_tools_structure():
    assert len(CODING_TOOLS) >= 5
    for tool in CODING_TOOLS:
        assert tool["type"] == "function"
        func = tool["function"]
        assert "name" in func
        assert "parameters" in func


def test_tool_names_matches_coding_tools():
    names = {t["function"]["name"] for t in CODING_TOOLS}
    assert names == TOOL_NAMES


# ---------------------------------------------------------------------------
# ToolCallCheck / TurnValidation dataclasses
# ---------------------------------------------------------------------------


def test_turn_validation_invalid_with_errors():
    tv = TurnValidation(turn_index=0, errors=["something wrong"])
    assert tv.valid is False


def test_turn_validation_invalid_tool_call():
    tc = ToolCallCheck(valid=False, errors=["bad"])
    tv = TurnValidation(turn_index=0, has_tool_calls=True, tool_calls=[tc])
    assert tv.valid is False


def test_turn_validation_is_finished_text_only():
    tv = TurnValidation(turn_index=0, has_content=True, has_tool_calls=False)
    assert tv.is_finished is True


def test_turn_validation_is_finished_with_finish_tool():
    tc = ToolCallCheck(valid=True, name="finish")
    tv = TurnValidation(turn_index=0, has_tool_calls=True, tool_calls=[tc])
    assert tv.is_finished is True


def test_turn_validation_not_finished():
    tc = ToolCallCheck(valid=True, name="bash")
    tv = TurnValidation(turn_index=0, has_tool_calls=True, tool_calls=[tc])
    assert tv.is_finished is False


# ---------------------------------------------------------------------------
# validate_tool_calls
# ---------------------------------------------------------------------------


def _make_response(tool_calls=None, content=None, finish_reason="stop"):
    msg = {}
    if content is not None:
        msg["content"] = content
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {"choices": [{"finish_reason": finish_reason, "message": msg}]}


def test_validate_no_choices():
    v = validate_tool_calls({}, turn_index=0)
    assert not v.valid
    assert "no 'choices'" in v.errors[0]


def test_validate_text_only():
    resp = _make_response(content="Hello there")
    v = validate_tool_calls(resp)
    assert v.valid
    assert v.has_content
    assert not v.has_tool_calls


def test_validate_no_content_no_tools():
    resp = _make_response()
    v = validate_tool_calls(resp)
    assert not v.valid
    assert "neither tool_calls nor content" in v.errors[0]


def test_synthesize_tool_calls_from_run_tag_prefers_execute_bash():
    checks = synthesize_tool_calls_from_text(
        "I'll inspect it. <run>ls -la /tmp</run>",
        allowed_tool_names={"execute_bash", "finish"},
        mode="deepseek_text",
    )
    assert len(checks) == 1
    assert checks[0].name == "execute_bash"
    assert checks[0].arguments_parsed == {"command": "ls -la /tmp", "timeout": 30}


def test_synthesize_tool_calls_from_bash_fence_uses_bash_when_needed():
    checks = synthesize_tool_calls_from_text(
        "```bash\npwd && ls\n```",
        allowed_tool_names={"bash", "finish"},
        mode="deepseek_text",
    )
    assert len(checks) == 1
    assert checks[0].name == "bash"
    assert checks[0].arguments_parsed == {"command": "pwd && ls"}


def test_validate_valid_tool_call():
    tc = {
        "id": "call_123",
        "type": "function",
        "function": {
            "name": "bash",
            "arguments": json.dumps({"command": "ls"}),
        },
    }
    resp = _make_response(tool_calls=[tc])
    v = validate_tool_calls(resp)
    assert v.valid
    assert v.has_tool_calls
    assert len(v.tool_calls) == 1
    assert v.tool_calls[0].name == "bash"
    assert v.tool_calls[0].arguments_parsed == {"command": "ls"}


def test_validate_unknown_tool():
    tc = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "unknown_tool", "arguments": "{}"},
    }
    resp = _make_response(tool_calls=[tc])
    v = validate_tool_calls(resp)
    assert not v.tool_calls[0].valid
    assert any("unknown tool" in e for e in v.tool_calls[0].errors)


def test_validate_bad_json_arguments():
    tc = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "bash", "arguments": "not json at all"},
    }
    resp = _make_response(tool_calls=[tc])
    v = validate_tool_calls(resp)
    assert not v.tool_calls[0].valid
    assert any("JSON parse error" in e for e in v.tool_calls[0].errors)


def test_validate_bad_json_sanitize_recovers():
    """With sanitize_truncated=True, truncated JSON is recovered as empty dict."""
    tc = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "bash", "arguments": '{"command": "ls -la /tmp'},
    }
    resp = _make_response(tool_calls=[tc])
    v = validate_tool_calls(resp, sanitize_truncated=True)
    assert v.tool_calls[0].valid is True
    assert v.tool_calls[0].arguments_parsed == {}
    assert v.tool_calls[0].arguments_raw == "{}"
    assert not any("JSON parse error" in e for e in v.tool_calls[0].errors)


def test_validate_bad_json_sanitize_off_reports_error():
    """With sanitize_truncated=False (default), truncated JSON is an error."""
    tc = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "bash", "arguments": '{"command": "ls -la /tmp'},
    }
    resp = _make_response(tool_calls=[tc])
    v = validate_tool_calls(resp, sanitize_truncated=False)
    assert not v.tool_calls[0].valid
    assert any("JSON parse error" in e for e in v.tool_calls[0].errors)


def test_validate_sanitize_works_for_any_tool():
    """Sanitize should work for any tool, not just think."""
    for tool_name in ["think", "execute_bash", "str_replace_editor", "bash"]:
        tc = {
            "id": "call_1",
            "type": "function",
            "function": {"name": tool_name, "arguments": '{"key": "truncated value'},
        }
        resp = _make_response(tool_calls=[tc])
        v = validate_tool_calls(resp, sanitize_truncated=True)
        assert v.tool_calls[0].valid is True, f"Failed valid for {tool_name}"
        assert v.tool_calls[0].arguments_parsed == {}, f"Failed args for {tool_name}"


def test_validate_missing_id():
    tc = {
        "type": "function",
        "function": {"name": "bash", "arguments": "{}"},
    }
    resp = _make_response(tool_calls=[tc])
    v = validate_tool_calls(resp)
    assert any("missing 'id'" in e for e in v.tool_calls[0].errors)
    assert not v.tool_calls[0].valid


def test_validate_wrong_type():
    tc = {
        "id": "call_1",
        "type": "not_function",
        "function": {"name": "bash", "arguments": "{}"},
    }
    resp = _make_response(tool_calls=[tc])
    v = validate_tool_calls(resp)
    assert any("type=" in e for e in v.tool_calls[0].errors)
    assert not v.tool_calls[0].valid


def test_validate_arguments_as_dict():
    tc = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "bash", "arguments": {"command": "ls"}},
    }
    resp = _make_response(tool_calls=[tc])
    v = validate_tool_calls(resp)
    assert v.tool_calls[0].arguments_parsed == {"command": "ls"}


def test_validate_openhands_tool_name_with_allowed_set():
    tc = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "execute_bash", "arguments": {"command": "ls"}},
    }
    resp = _make_response(tool_calls=[tc], finish_reason="tool_calls")
    v = validate_tool_calls(resp, allowed_tool_names={"execute_bash", "finish"})
    assert v.valid
    assert v.tool_calls[0].name == "execute_bash"


# ---------------------------------------------------------------------------
# _looks_like_garbage
# ---------------------------------------------------------------------------


def test_garbage_short_text():
    assert _looks_like_garbage("short") is False


def test_garbage_repeated_pattern():
    text = "abcdefghij" * 100
    assert _looks_like_garbage(text) is True


def test_garbage_low_unique_ratio():
    text = "a" * 600
    assert _looks_like_garbage(text) is True


def test_garbage_normal_text():
    text = "This is a perfectly normal string with varied characters and numbers 12345."
    assert _looks_like_garbage(text) is False


# ---------------------------------------------------------------------------
# Sandbox + real_execute
# ---------------------------------------------------------------------------


@pytest.fixture
def sandbox(tmp_path):
    return ExecutionWorkspace(tmp_path / "workspace")


@pytest.mark.e2e
def test_execution_workspace_creates_root(tmp_path):
    sb = ExecutionWorkspace(tmp_path / "new_workspace")
    assert sb.root.exists()


@pytest.mark.e2e
def test_sandbox_alias_is_preserved(tmp_path):
    sb = Sandbox(tmp_path / "legacy_sandbox")
    assert isinstance(sb, ExecutionWorkspace)


@pytest.mark.e2e
def test_execution_workspace_safe_path_blocks_traversal(sandbox):
    with pytest.raises(ValueError, match="traversal"):
        sandbox._safe_path("../../etc/passwd")


@pytest.mark.e2e
def test_execution_workspace_setup_and_list_files(sandbox):
    sandbox.setup_files({"src/main.py": "print('hi')", "README.md": "# Hello"})
    files = sandbox.list_files()
    assert "src/main.py" in files
    assert "README.md" in files


@pytest.mark.e2e
def test_execution_workspace_run_command(sandbox):
    out = sandbox.run_command("echo hello")
    assert out.strip() == "hello"


@pytest.mark.e2e
def test_execution_workspace_run_command_timeout(sandbox):
    out = sandbox.run_command("sleep 10", timeout=1)
    assert "TIMEOUT" in out


def test_execution_workspace_safe_env_does_not_inherit_parent_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTSURGE_SECRET_TOKEN", "secret")
    sb = ExecutionWorkspace(tmp_path / "safe", tool_env="safe")

    out = sb.run_command('printf "%s\\n%s\\n" "$HOME" "${AGENTSURGE_SECRET_TOKEN-unset}"')

    lines = out.splitlines()
    assert lines[0] == str(sb.root)
    assert lines[1] == "unset"


def test_execution_workspace_inherit_env_escape_hatch(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTSURGE_SECRET_TOKEN", "secret")
    sb = ExecutionWorkspace(tmp_path / "inherit", tool_env="inherit")

    out = sb.run_command('printf "%s\\n%s\\n" "$HOME" "$AGENTSURGE_SECRET_TOKEN"')

    lines = out.splitlines()
    assert lines[0] == str(sb.root)
    assert lines[1] == "secret"


@pytest.mark.asyncio
async def test_execution_workspace_async_safe_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTSURGE_SECRET_TOKEN", "secret")
    sb = ExecutionWorkspace(tmp_path / "safe_async", tool_env="safe")

    out = await sb.run_command_async('printf "%s\\n" "${AGENTSURGE_SECRET_TOKEN-unset}"')

    assert out.strip() == "unset"


def test_real_execute_bash(sandbox):
    out = real_execute("bash", {"command": "echo test123"}, sandbox)
    assert out.strip() == "test123"


@pytest.mark.asyncio
async def test_real_execute_async_execute_bash(sandbox):
    out = await real_execute_async("execute_bash", {"command": "echo test123"}, sandbox)
    assert out.strip() == "test123"


def test_real_execute_write_and_read(sandbox):
    real_execute("write_file", {"path": "hello.txt", "content": "world"}, sandbox)
    out = real_execute("read_file", {"path": "hello.txt"}, sandbox)
    assert out == "world"


def test_real_execute_read_nonexistent(sandbox):
    out = real_execute("read_file", {"path": "nope.txt"}, sandbox)
    assert "not found" in out.lower()


def test_real_execute_read_line_range(sandbox):
    real_execute("write_file", {"path": "lines.txt", "content": "a\nb\nc\nd\n"}, sandbox)
    out = real_execute("read_file", {"path": "lines.txt", "start_line": 2, "end_line": 3}, sandbox)
    assert out == "b\nc\n"


def test_real_execute_edit_file(sandbox):
    real_execute("write_file", {"path": "f.txt", "content": "foo bar baz"}, sandbox)
    out = real_execute(
        "edit_file", {"path": "f.txt", "old_string": "bar", "new_string": "qux"}, sandbox
    )
    assert "Edited" in out
    content = real_execute("read_file", {"path": "f.txt"}, sandbox)
    assert content == "foo qux baz"


def test_real_execute_edit_not_found(sandbox):
    real_execute("write_file", {"path": "f.txt", "content": "hello"}, sandbox)
    out = real_execute(
        "edit_file", {"path": "f.txt", "old_string": "xyz", "new_string": "abc"}, sandbox
    )
    assert "not found" in out.lower()


def test_real_execute_search_code(sandbox):
    sandbox.setup_files({"a.py": "import json\n", "b.py": "import os\n"})
    out = real_execute("search_code", {"pattern": "json"}, sandbox)
    assert "json" in out
    assert "a.py" in out


def test_real_execute_finish(sandbox):
    out = real_execute("finish", {"summary": "done"}, sandbox)
    assert out == "Task finished: done\n"


@pytest.mark.asyncio
async def test_real_execute_async_str_replace_editor(sandbox):
    sandbox.setup_files({"src/main.py": "hello\nworld\n"})
    viewed = await real_execute_async(
        "str_replace_editor",
        {"command": "view", "path": "src/main.py", "view_range": [1, -1]},
        sandbox,
    )
    assert "hello" in viewed
    edited = await real_execute_async(
        "str_replace_editor",
        {
            "command": "str_replace",
            "path": "src/main.py",
            "old_str": "world",
            "new_str": "agent",
        },
        sandbox,
    )
    assert "replaced" in edited.lower()
    assert "agent" in sandbox._safe_path("src/main.py").read_text()


def test_real_execute_unknown(sandbox):
    out = real_execute("nonexistent", {}, sandbox)
    assert "Unknown tool" in out


def test_real_execute_path_traversal(sandbox):
    out = real_execute("read_file", {"path": "../../etc/passwd"}, sandbox)
    assert "ERROR" in out


# ---------------------------------------------------------------------------
# build_tool_response_messages
# ---------------------------------------------------------------------------


def test_build_tool_response_messages_invalid_tc():
    tc = ToolCallCheck(
        valid=False,
        tool_call_id="call_bad",
        name="bash",
        errors=["bad args"],
    )
    msgs = build_tool_response_messages({"role": "assistant"}, [tc])
    assert "ERROR" in msgs[1]["content"]


def test_build_tool_response_messages_real(sandbox):
    sandbox.setup_files({"test.txt": "hello"})
    tc = ToolCallCheck(
        valid=True,
        tool_call_id="call_r",
        name="read_file",
        arguments_parsed={"path": "test.txt"},
    )
    msgs = build_tool_response_messages(
        {"role": "assistant"},
        [tc],
        sandbox=sandbox,
    )
    assert msgs[1]["content"] == "hello"


@pytest.mark.asyncio
async def test_build_tool_response_messages_async_openhands_tool(sandbox):
    tc = ToolCallCheck(
        valid=True,
        tool_call_id="call_exec",
        name="execute_bash",
        arguments_parsed={"command": "echo hello"},
    )
    msgs = await build_tool_response_messages_async(
        {"role": "assistant"},
        [tc],
        sandbox=sandbox,
    )
    assert msgs[1]["content"].strip() == "hello"


# ---------------------------------------------------------------------------
# Bug fix: repo URL derivation from metadata["repo"]
# ---------------------------------------------------------------------------


@pytest.mark.e2e
def test_setup_from_instance_id_url_derivation(sandbox, monkeypatch):
    """repo='owner/repo' (not a URL) should be converted to a GitHub URL."""
    captured_args = {}

    def fake_run(args, **kwargs):
        captured_args["clone"] = args
        result = type("R", (), {"returncode": 1, "stderr": "fake", "stdout": ""})()
        return result

    monkeypatch.setattr("agentsurge.tool_call.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="git clone failed"):
        sandbox.setup_from_instance_id(
            "tobymao__sqlglot-2295",
            metadata={"repo": "tobymao/sqlglot"},
        )
    clone_cmd = captured_args["clone"]
    url = [a for a in clone_cmd if a.startswith("http")][0]
    assert url == "https://github.com/tobymao/sqlglot.git"


@pytest.mark.e2e
def test_setup_from_instance_id_derives_compact_swe_smith_repo_and_commit(sandbox, monkeypatch):
    captured_args = {}

    def fake_run(args, **kwargs):
        if args[:2] == ["git", "clone"]:
            captured_args["clone"] = args
            return type("R", (), {"returncode": 1, "stderr": "fake", "stdout": ""})()
        return type("R", (), {"returncode": 0, "stderr": "", "stdout": ""})()

    monkeypatch.setattr("agentsurge.tool_call.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="git clone failed"):
        sandbox.setup_from_instance_id(
            "django-money__django-money.835c1ab8.func_pm_ctrl_shuffle__viqnyl9u"
        )

    clone_cmd = captured_args["clone"]
    assert any(a == "https://github.com/django-money/django-money.git" for a in clone_cmd)


# ---------------------------------------------------------------------------
# ABC-Bench sandbox route (source="abc-bench")
# ---------------------------------------------------------------------------


def test_setup_from_instance_id_abc_bench_uses_extracted_tarball(sandbox, tmp_path, monkeypatch):
    """source=abc-bench must skip git clone and copy from the extracted task dir."""
    task_dir = tmp_path / "task_acme__scenario"
    repo_src = task_dir / "acme_repo"
    repo_src.mkdir(parents=True)
    (repo_src / "main.py").write_text("print('hi')\n")
    (task_dir / "task.yaml").write_text("name: scenario\n")

    # Guard: git clone must NOT be invoked for abc-bench.
    def boom_run(*args, **kwargs):
        raise AssertionError("git clone was invoked for abc-bench source")

    monkeypatch.setattr("agentsurge.tool_call.subprocess.run", boom_run)
    monkeypatch.setattr(
        "agentsurge.loaders.abc_bench_assets.get_task_dir",
        lambda task_id, cache_dir=None: task_dir,
    )

    sandbox.setup_from_instance_id(
        "task_acme__scenario",
        metadata={"source": "abc-bench", "task_id": "task_acme__scenario"},
    )

    copied = sandbox.root / "workspace" / "acme_repo" / "main.py"
    assert copied.exists()
    assert copied.read_text() == "print('hi')\n"


def test_setup_from_instance_id_abc_bench_requires_task_id(sandbox):
    with pytest.raises(RuntimeError, match="metadata\\['task_id'\\]"):
        sandbox.setup_from_instance_id("ignored", metadata={"source": "abc-bench"})


# ---------------------------------------------------------------------------
# Bug fix: /workspace/ path rewriting in _safe_path and run_command
# ---------------------------------------------------------------------------


def test_safe_path_workspace_rewrite(sandbox):
    """Absolute /workspace/ paths should be rewritten to sandbox."""
    ws = sandbox.root / "workspace" / "proj"
    ws.mkdir(parents=True)
    (ws / "f.txt").write_text("ok")
    resolved = sandbox._safe_path("/workspace/proj/f.txt")
    assert resolved == sandbox.root / "workspace" / "proj" / "f.txt"
    assert resolved.read_text() == "ok"


def test_safe_path_relative_still_works(sandbox):
    """Relative paths should still resolve inside the sandbox."""
    sandbox.setup_files({"a.txt": "hello"})
    resolved = sandbox._safe_path("a.txt")
    assert resolved == sandbox.root / "a.txt"


def test_run_command_workspace_rewrite(sandbox):
    """bash commands with /workspace/ should execute in sandbox workspace."""
    ws = sandbox.root / "workspace" / "myrepo"
    ws.mkdir(parents=True)
    (ws / "hello.txt").write_text("world")
    out = sandbox.run_command("cat /workspace/myrepo/hello.txt")
    assert "world" in out


def test_run_command_workspace_cd(sandbox):
    """cd /workspace/... in a bash command should work via rewrite."""
    ws = sandbox.root / "workspace" / "proj"
    ws.mkdir(parents=True)
    (ws / "test.py").write_text("print('hi')")
    out = sandbox.run_command("cd /workspace/proj && ls")
    assert "test.py" in out


# ---------------------------------------------------------------------------
# str_replace_editor: create, insert, undo_edit, error cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_str_replace_editor_create(sandbox):
    out = await real_execute_async(
        "str_replace_editor",
        {"command": "create", "path": "new.txt", "file_text": "hello"},
        sandbox,
    )
    assert "created" in out.lower()
    assert sandbox._safe_path("new.txt").read_text() == "hello"


@pytest.mark.asyncio
async def test_str_replace_editor_insert(sandbox):
    sandbox.setup_files({"ins.txt": "line1\nline2\n"})
    out = await real_execute_async(
        "str_replace_editor",
        {"command": "insert", "path": "ins.txt", "insert_line": 1, "new_str": "inserted\n"},
        sandbox,
    )
    assert "inserted" in out.lower()
    content = sandbox._safe_path("ins.txt").read_text()
    lines = content.splitlines()
    assert lines == ["line1", "inserted", "line2"]


@pytest.mark.asyncio
async def test_str_replace_editor_undo_edit(sandbox):
    sandbox.setup_files({"u.txt": "data"})
    out = await real_execute_async(
        "str_replace_editor",
        {"command": "undo_edit", "path": "u.txt"},
        sandbox,
    )
    assert "not supported" in out.lower()


@pytest.mark.asyncio
async def test_str_replace_editor_bad_view_range(sandbox):
    sandbox.setup_files({"v.txt": "aaa\nbbb\n"})
    out = await real_execute_async(
        "str_replace_editor",
        {"command": "view", "path": "v.txt", "view_range": ["abc", "def"]},
        sandbox,
    )
    assert "aaa" in out
    assert "bbb" in out


@pytest.mark.asyncio
async def test_str_replace_editor_bad_insert_line(sandbox):
    sandbox.setup_files({"b.txt": "content\n"})
    out = await real_execute_async(
        "str_replace_editor",
        {"command": "insert", "path": "b.txt", "insert_line": "not_a_number", "new_str": "x"},
        sandbox,
    )
    assert "ERROR" in out
    assert "integer" in out


@pytest.mark.asyncio
async def test_str_replace_editor_missing_path(sandbox):
    out = await real_execute_async(
        "str_replace_editor",
        {"command": "view", "path": ""},
        sandbox,
    )
    assert "ERROR" in out


@pytest.mark.asyncio
async def test_str_replace_editor_view_directory(sandbox):
    sandbox.setup_files({"mydir/a.txt": "x", "mydir/b.txt": "y"})
    out = await real_execute_async(
        "str_replace_editor",
        {"command": "view", "path": "mydir"},
        sandbox,
    )
    assert "a.txt" in out
    assert "b.txt" in out


# ---------------------------------------------------------------------------
# synthesize_tool_calls_from_text - hermes_xml mode
# ---------------------------------------------------------------------------


def test_synthesize_hermes_xml_single_tool_call():
    checks = synthesize_tool_calls_from_text(
        '<tool_call>{"name": "execute_bash", "arguments": {"command": "ls"}}</tool_call>',
        allowed_tool_names={"execute_bash", "finish"},
        mode="hermes_xml",
    )
    assert len(checks) == 1
    assert checks[0].name == "execute_bash"
    assert checks[0].arguments_parsed == {"command": "ls"}


def test_synthesize_hermes_xml_unknown_tool_skipped():
    checks = synthesize_tool_calls_from_text(
        '<tool_call>{"name": "unknown_tool", "arguments": {}}</tool_call>',
        allowed_tool_names={"bash", "finish"},
        mode="hermes_xml",
    )
    assert len(checks) == 0


def test_synthesize_hermes_xml_malformed_json():
    checks = synthesize_tool_calls_from_text(
        "<tool_call>not valid json</tool_call>",
        mode="hermes_xml",
    )
    assert len(checks) == 0


def test_synthesize_hermes_xml_multiple_calls():
    text = (
        '<tool_call>{"name": "bash", "arguments": {"command": "ls"}}</tool_call>'
        " some text "
        '<tool_call>{"name": "finish", "arguments": {"summary": "done"}}</tool_call>'
    )
    checks = synthesize_tool_calls_from_text(text, mode="hermes_xml")
    assert len(checks) == 2
    assert checks[0].name == "bash"
    assert checks[1].name == "finish"


def test_synthesize_hermes_xml_arguments_as_string():
    checks = synthesize_tool_calls_from_text(
        '<tool_call>{"name": "bash", "arguments": "{\\"command\\": \\"pwd\\"}"}</tool_call>',
        mode="hermes_xml",
    )
    assert len(checks) == 1
    assert checks[0].arguments_parsed == {"command": "pwd"}
