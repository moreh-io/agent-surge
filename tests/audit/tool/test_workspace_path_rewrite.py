# SPDX-License-Identifier: MIT
"""[audit:tool#H5] ``/workspace/`` rewrite must not mutate quoted literal data.

Both ``run_command`` and ``run_command_async`` did
``cmd = cmd.replace("/workspace/", f"{workspace_dir}/")`` unconditionally.
A command like ``echo '/workspace/foo'`` therefore had its quoted *data*
mutated — fine for path resolution but corrupting for grep patterns, sed
scripts, or any literal string the model emits that happens to contain the
sentinel.

The fix limits rewriting to *path-shaped* occurrences (preceded by a token
boundary like start, whitespace, ``=`` or a quote) — the substring inside
``"echo '/workspace/foo'"`` should resolve into the workspace dir as a path
target, but a substring inside ``"grep -r '/workspace/' ."`` (a grep
pattern, not a path argument) should NOT.

We pin the conservative property: when the literal sentinel appears inside
double-quoted *content* (a clear "this is data, not a path" marker), it
must round-trip through ``run_command`` unchanged.
"""

import asyncio

import pytest

from agentsurge.tool_call import ExecutionWorkspace


@pytest.fixture
def workspace(tmp_path):
    ws = ExecutionWorkspace(str(tmp_path))
    # Force the workspace_dir to exist so the rewrite branch is active.
    (ws.root / "workspace").mkdir(exist_ok=True)
    return ws


def test_run_command_does_not_mutate_double_quoted_literal(workspace):
    # Echoing the literal sentinel inside double quotes should print it back
    # verbatim. The rewrite must not touch quoted data: the printed line
    # should exactly equal the literal string the model emitted.
    out = workspace.run_command('echo "/workspace/literal"')
    assert out.strip() == "/workspace/literal", (
        f"literal /workspace/literal in double-quoted echo was mutated; got: {out!r}"
    )


def test_run_command_does_not_mutate_single_quoted_literal(workspace):
    out = workspace.run_command("echo '/workspace/literal'")
    assert out.strip() == "/workspace/literal", (
        f"literal /workspace/literal in single-quoted echo was mutated; got: {out!r}"
    )


def test_run_command_async_does_not_mutate_quoted_literal(workspace):
    async def _run():
        return await workspace.run_command_async('echo "/workspace/literal"')

    out = asyncio.run(_run())
    assert out.strip() == "/workspace/literal", (
        f"async run_command mutated quoted literal; got: {out!r}"
    )


def test_run_command_still_resolves_unquoted_path(workspace):
    """Sanity: unquoted /workspace/ paths *should* still resolve to the
    workspace dir so OpenHands traces continue to work. Create a file under
    the workspace dir and confirm cat /workspace/<name> reads it.
    """
    target = workspace.root / "workspace" / "hello.txt"
    target.write_text("hi-from-workspace\n")
    out = workspace.run_command("cat /workspace/hello.txt")
    assert "hi-from-workspace" in out, f"unquoted /workspace/ path failed to resolve: {out!r}"
