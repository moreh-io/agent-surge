# SPDX-License-Identifier: MIT
from __future__ import annotations

import json

from agentsurge.frontends.base import FrontendRunArtifacts
from agentsurge.types import ReplaySession


def render_session(session: ReplaySession, artifacts: FrontendRunArtifacts) -> None:
    artifacts.session_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [f"# Session {session.session_id}", ""]
    for k, v in session.metadata.items():
        lines.append(f"{k}: {v}")
    lines.append("")

    for i, turn_msgs in enumerate(session.turn_messages):
        lines.append(f"## Turn {i}")
        for msg in turn_msgs:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                text = " ".join(part.get("text", "") for part in content if isinstance(part, dict))
            else:
                text = str(content) if content else ""
            lines.append(f"{role}: {text}")
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    lines.append(f"  tool: {tc}")
        lines.append("")

    artifacts.prompt_path.write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "session_id": session.session_id,
        "metadata": session.metadata,
        "turn_messages": session.turn_messages,
    }
    artifacts.session_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
