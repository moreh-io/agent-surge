# SPDX-License-Identifier: MIT
"""`agentsurge proxy-translator` CLI subcommand.

Starts a long-running role-translator reverse proxy in the foreground. The
process exits on SIGINT/SIGTERM after the aiohttp graceful shutdown
completes. Used by Codex frontend benchmark runs to bridge restrictive
chat templates (Qwen 3.6 etc.) that don't accept the OpenAI Responses
`developer` role.

Subparser registration lives in `_app.py` (it follows the inline
`sub.add_parser(...)` + `set_defaults(func=...)` pattern used by every
other subcommand in this codebase). This module only owns `cmd_proxy_translator`.
"""

from __future__ import annotations

import argparse

from aiohttp import web

from agentsurge.proxy.role_translator import make_app


def cmd_proxy_translator(args: argparse.Namespace) -> None:
    host, port_str = args.listen.rsplit(":", 1)
    port = int(port_str)
    app = make_app(upstream_base=args.target, timeout_s=args.timeout)
    print(
        f"agentsurge proxy-translator listening on {host}:{port} → {args.target} "
        f"(timeout={args.timeout}s)"
    )
    web.run_app(app, host=host, port=port, print=None)
