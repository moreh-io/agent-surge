# SPDX-License-Identifier: MIT
"""vLLM backend adapter - the default agentsurge inference backend.

This subpackage implements :class:`~agentsurge.backends.base.BackendBase` for
vLLM-compatible OpenAI API servers, supporting both chat-completions and
responses-API streaming endpoints.

Quick start::

    from agentsurge.backends.vllm import VllmBackend, VllmConfig

    cfg = VllmConfig(base_url="http://localhost:8000", model="my-model")
    async with VllmBackend(cfg) as backend:
        result = await backend.send_turn("s1", 0, [{"role": "user", "content": "hi"}])
"""

from agentsurge.backends.vllm._backend import VllmBackend, VllmConfig

__all__ = ["VllmBackend", "VllmConfig"]
