# SPDX-License-Identifier: MIT
"""AgentSurge exception hierarchy.

All agentsurge-specific exceptions inherit from :class:`AgentSurgeError`, making it
easy to catch any library error with a single ``except AgentSurgeError`` clause.

Hierarchy
---------
AgentSurgeError
├── ConfigError            – Invalid or missing configuration
├── AgentSurgeConnectionError     – Backend / server connectivity failures
├── WorkloadValidationError – Malformed workload definitions
└── BackendError           – Runtime errors from inference backends
"""


class AgentSurgeError(Exception):
    """Base exception for all agentsurge errors.

    Parameters
    ----------
    message : str
        Human-readable description of the error.
    detail : str | None
        Optional extended information (e.g. full traceback from a backend).
    """

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        self.message = message
        self.detail = detail
        super().__init__(message)

    def __str__(self) -> str:
        if self.detail:
            return f"{self.message} [{self.detail}]"
        return self.message

    def __repr__(self) -> str:
        cls = type(self).__name__
        if self.detail:
            return f"{cls}({self.message!r}, detail={self.detail!r})"
        return f"{cls}({self.message!r})"


class ConfigError(AgentSurgeError):
    """Raised when configuration is invalid, missing, or malformed.

    Examples: unknown preset name, missing required config key, invalid YAML.
    """


class AgentSurgeConnectionError(AgentSurgeError):
    """Raised when a connection to a backend or server fails.

    Examples: unreachable vLLM endpoint, timeout during health check,
    refused connection to metrics server.

    Parameters
    ----------
    message : str
        Human-readable description.
    endpoint : str | None
        The URL or address that could not be reached.
    detail : str | None
        Optional extended information.
    """

    def __init__(
        self,
        message: str,
        *,
        endpoint: str | None = None,
        detail: str | None = None,
    ) -> None:
        self.endpoint = endpoint
        super().__init__(message, detail=detail)


class WorkloadValidationError(AgentSurgeError):
    """Raised when a workload definition fails validation.

    Examples: negative token count, empty session list, invalid arrival
    pattern, context length exceeding model maximum.
    """


class BackendError(AgentSurgeError):
    """Raised when the inference backend encounters a runtime error.

    Examples: OOM on the GPU, model not loaded, unexpected response format
    from the serving engine.

    Parameters
    ----------
    message : str
        Human-readable description.
    backend : str | None
        Name of the backend that failed (e.g. ``"vllm"``, ``"mock"``).
    detail : str | None
        Optional extended information.
    """

    def __init__(
        self,
        message: str,
        *,
        backend: str | None = None,
        detail: str | None = None,
    ) -> None:
        self.backend = backend
        super().__init__(message, detail=detail)
