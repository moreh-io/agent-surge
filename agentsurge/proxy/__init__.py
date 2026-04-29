# SPDX-License-Identifier: MIT
"""HTTP proxies that sit between agent CLIs and OpenAI-compatible upstreams.

`role_translator` rewrites Codex's Responses-API `developer` role messages
into `system` so models whose chat templates only know the standard OpenAI
roles still accept the request.
"""
