# SPDX-License-Identifier: MIT
"""[audit:tool#M1] Synthesized fallback tool_call_id must be deterministic.

``synthesize_tool_calls_from_text`` minted ``f"fallback_{uuid.uuid4().hex[:24]}"``
for the synthesized ``tool_call_id``. UUID4 is non-deterministic even for
identical input + seed, so replay-mode benchmark runs produce different
``tool_call_id`` strings each invocation. Downstream artefacts (inflight
dumps, JSON outputs) diff on every run despite identical inputs.

The fix derives the id from a stable hash of (mode, name, arguments_raw,
occurrence_index) so identical inputs yield identical ids.
"""

from agentsurge.tool_call import synthesize_tool_calls_from_text


def test_deepseek_fallback_id_stable_across_invocations():
    text = "<run>echo hello</run>"
    a = synthesize_tool_calls_from_text(text, mode="deepseek_text")
    b = synthesize_tool_calls_from_text(text, mode="deepseek_text")
    assert len(a) == 1 and len(b) == 1
    assert a[0].tool_call_id == b[0].tool_call_id, (
        f"deepseek_text fallback id should be deterministic; "
        f"got {a[0].tool_call_id!r} != {b[0].tool_call_id!r}"
    )


def test_hermes_fallback_id_stable_across_invocations():
    text = '<tool_call>{"name": "bash", "arguments": {"command": "ls"}}</tool_call>'
    a = synthesize_tool_calls_from_text(text, mode="hermes_xml")
    b = synthesize_tool_calls_from_text(text, mode="hermes_xml")
    assert len(a) == 1 and len(b) == 1
    assert a[0].tool_call_id == b[0].tool_call_id, (
        f"hermes_xml fallback id should be deterministic; "
        f"got {a[0].tool_call_id!r} != {b[0].tool_call_id!r}"
    )


def test_hermes_multi_call_ids_unique_within_one_synthesis():
    """Two calls in the same text must still get distinct ids (the second
    occurrence of the same name+args must not collide with the first)."""
    text = (
        '<tool_call>{"name": "bash", "arguments": {"command": "ls"}}</tool_call>'
        '<tool_call>{"name": "bash", "arguments": {"command": "ls"}}</tool_call>'
    )
    out = synthesize_tool_calls_from_text(text, mode="hermes_xml")
    assert len(out) == 2
    assert out[0].tool_call_id != out[1].tool_call_id, (
        "consecutive same-name same-args calls must still get unique ids"
    )


def test_fallback_id_format_prefix():
    """Keep the `fallback_` prefix so existing log-grep / debug tools work."""
    text = "<run>echo hello</run>"
    out = synthesize_tool_calls_from_text(text, mode="deepseek_text")
    assert out[0].tool_call_id.startswith("fallback_")
