"""M10: SWESmithLoader split validation + alias factory.

``SWESmithLoader(split="tools")`` (typo) currently silently propagates
into ``load_dataset`` and surfaces only as a runtime "split not found".
Aliases ``swe-smith-{tool,xml,ticks}`` must construct a loader pinned to
the corresponding split rather than always defaulting to ``tool``.
"""

from __future__ import annotations

import pytest


def test_invalid_split_rejected_at_construction():
    from agentsurge.loaders.swe_smith import SWESmithLoader

    with pytest.raises(ValueError):
        SWESmithLoader(split="tools")  # typo


def test_valid_splits_accepted():
    from agentsurge.loaders.swe_smith import SWESmithLoader

    for split in ("tool", "xml", "ticks"):
        loader = SWESmithLoader(split=split)
        assert loader.split == split


def test_alias_xml_carries_split_through_factory():
    """``get_loader("swe-smith-xml")()`` must yield a loader with split='xml'."""
    from agentsurge.loaders import get_loader

    cls_or_factory = get_loader("swe-smith-xml")
    instance = cls_or_factory()
    assert getattr(instance, "split", None) == "xml"


def test_alias_ticks_carries_split_through_factory():
    from agentsurge.loaders import get_loader

    cls_or_factory = get_loader("swe-smith-ticks")
    instance = cls_or_factory()
    assert getattr(instance, "split", None) == "ticks"


def test_alias_tool_explicit():
    from agentsurge.loaders import get_loader

    cls_or_factory = get_loader("swe-smith-tool")
    instance = cls_or_factory()
    assert getattr(instance, "split", None) == "tool"
