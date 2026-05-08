# SPDX-License-Identifier: MIT
"""H4: Turn.tool_calls (list[dict]) vs TurnResult.tool_calls (list[str]) collision.

These two fields share a name but hold incompatibly-typed values. The contract
must be documented so consumers cannot conflate them, and the asymmetry must
be observable from the public API.
"""

import dataclasses

from agentsurge.types.results import TurnResult
from agentsurge.types.trace import Turn


def test_turn_tool_calls_is_list_of_dict():
    """Turn (input/trace side) uses list[dict] — full payloads."""
    f = next(f for f in dataclasses.fields(Turn) if f.name == "tool_calls")
    t = f.type if isinstance(f.type, str) else str(f.type)
    assert "dict" in t, f"Turn.tool_calls must be list[dict]; got {t!r}"


def test_turn_result_tool_calls_is_list_of_str():
    """TurnResult (output/runner side) uses list[str] — names only."""
    f = next(f for f in dataclasses.fields(TurnResult) if f.name == "tool_calls")
    t = f.type if isinstance(f.type, str) else str(f.type)
    assert "str" in t, f"TurnResult.tool_calls must be list[str]; got {t!r}"


def test_turn_result_documents_tool_calls_asymmetry():
    """TurnResult must carry a docstring/comment that calls out the collision
    so future readers don't conflate Turn.tool_calls and TurnResult.tool_calls.

    Either the class docstring mentions the asymmetry, or the source of the
    field carries an explanatory comment.
    """
    import inspect

    src = inspect.getsource(TurnResult)
    haystack = src.lower()
    # Accept any of: "tool_calls_detail" + comment naming Turn,
    # explicit docstring callout, or a comment explaining "names only".
    has_asymmetry_doc = (
        ("names only" in haystack and "tool_calls_detail" in haystack and "turn." in haystack)
        or ("differs from turn.tool_calls" in haystack)
        or ("asymmetric" in haystack and "tool_calls" in haystack)
    )
    assert has_asymmetry_doc, (
        "TurnResult must document the tool_calls asymmetry against Turn.tool_calls "
        "(see audit types#H4)."
    )
