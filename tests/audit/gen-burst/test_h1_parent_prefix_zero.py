"""Audit H1: parent_prefix_fraction=0.0 should mean zero inheritance.

The hierarchical() docstring says "Fraction (0.0-1.0) of parent's accumulated
context that children inherit. Siblings share this prefix, then diverge".
With fraction=0.0 the child's first turn must NOT begin with any parent message.
"""

from __future__ import annotations

from agentsurge.generators.burst import BurstPatternGenerator


def _first_turn(child) -> list[dict]:
    return child.turn_messages[0]


def test_parent_prefix_fraction_zero_yields_no_inherited_messages():
    gen = BurstPatternGenerator(seed=7)
    sessions = gen.hierarchical(
        n_trees=1,
        branching_factor=2,
        depth=1,
        turns_per_agent=1,
        tokens_per_turn=64,
        root_prefix_chars=2_000,
        parent_prefix_fraction=0.0,
    )
    # sessions[0] is root; sessions[1:] are children at depth 1.
    root = sessions[0]
    children = [s for s in sessions if s.tree_level == 1]
    assert children, "expected at least one child"

    root_first_turn_messages = root.turn_messages[-1]
    root_msg_contents = {(m.get("role"), m.get("content")) for m in root_first_turn_messages}

    for child in children:
        child_msgs = _first_turn(child)
        # With fraction=0.0 child must not start with any parent message.
        # Existing bug: floor of max(1, ...) forces inheriting parent[0]
        # (typically the system prompt), so first child message will match a
        # root message.
        first = child_msgs[0]
        assert (first.get("role"), first.get("content")) not in root_msg_contents, (
            "parent_prefix_fraction=0.0 must not inherit any parent messages; "
            f"child first message matched a parent message (role={first.get('role')})"
        )
