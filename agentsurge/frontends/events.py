# SPDX-License-Identifier: MIT
from __future__ import annotations

from agentsurge.frontends.base import FrontendEvent

EVENT_SESSION_STARTED = "session.started"
EVENT_CLI_PROCESS_STARTED = "cli.process.started"
EVENT_MESSAGE_START = "message.start"
EVENT_ASSISTANT_TEXT_DELTA = "assistant.text.delta"
EVENT_ASSISTANT_MESSAGE_COMPLETED = "assistant.message.completed"
EVENT_USAGE_COMPLETED = "usage.completed"
EVENT_SESSION_COMPLETED = "session.completed"
EVENT_PARSER_ERROR = "parser.error"
EVENT_PARSER_UNKNOWN = "parser.unknown"
EVENT_TURN_STARTED = "turn.started"
EVENT_TURN_COMPLETED = "turn.completed"
EVENT_TURN_FAILED = "turn.failed"
EVENT_ITEM_COMPLETED = "item.completed"
EVENT_USER_MESSAGE = "user.message"
EVENT_ERROR = "error"
EVENT_TOOL_USE_OBSERVED = "tool_use.observed"


def event_to_dict(event: FrontendEvent) -> dict[str, object]:
    d: dict[str, object] = {
        "ts_monotonic": event.ts_monotonic,
        "kind": event.kind,
        "text_delta": event.text_delta,
    }
    if event.usage is not None:
        d["usage"] = event.usage
    if event.raw is not None:
        d["raw"] = event.raw
    return d
