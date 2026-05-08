"""M6: HttpSessionMixin must apply a finite default connection limit.

aiohttp.TCPConnector(limit=0) means "no limit" -- a runner with thousands of
asyncio tasks can open thousands of sockets concurrently, leading to fd /
ephemeral-port exhaustion.  Default to a finite limit and expose it on
BackendConfig so users can override.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from agentsurge.backends.base import BackendConfig
from agentsurge.backends.openai import OpenAiBackend


def test_connection_limit_default_is_finite():
    cfg = BackendConfig()
    # The new field must exist and default to a finite positive value.
    assert hasattr(cfg, "connection_limit")
    assert cfg.connection_limit > 0


def test_tcp_connector_uses_configured_limit():
    cfg = BackendConfig(base_url="http://test:8000", model="m", connection_limit=128)
    backend = OpenAiBackend(cfg)

    captured = {}

    def _capture_connector(*args, **kwargs):
        captured.update(kwargs)
        return MagicMock()

    async def _run():
        with patch("agentsurge.backends._http.aiohttp") as mock_aiohttp:
            mock_aiohttp.TCPConnector.side_effect = _capture_connector
            mock_session = MagicMock()
            mock_session.close = AsyncMock()
            mock_aiohttp.ClientSession.return_value = mock_session
            async with backend:
                pass

    asyncio.run(_run())
    assert captured.get("limit") == 128
