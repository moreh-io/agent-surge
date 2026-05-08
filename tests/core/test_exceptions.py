"""Tests for agentsurge.exceptions - custom exception hierarchy."""

from __future__ import annotations

from agentsurge.exceptions import (
    AgentSurgeConnectionError,
    AgentSurgeError,
    BackendError,
    ConfigError,
    WorkloadValidationError,
)


class TestExceptionHierarchy:
    def test_all_exceptions_inherit_from_agentsurge_error(self):
        for cls in (ConfigError, AgentSurgeConnectionError, WorkloadValidationError, BackendError):
            assert issubclass(cls, AgentSurgeError)


class TestConfigError:
    def test_message_preserved(self):
        err = ConfigError("missing key: model")
        assert err.message == "missing key: model"
        assert str(err) == "missing key: model"

    def test_detail_included_in_str(self):
        err = ConfigError("bad yaml", detail="line 5: unexpected indent")
        assert "bad yaml" in str(err)
        assert "line 5" in str(err)

    def test_repr_without_detail(self):
        err = ConfigError("oops")
        assert repr(err) == "ConfigError('oops')"

    def test_repr_with_detail(self):
        err = ConfigError("oops", detail="ctx")
        assert "detail=" in repr(err)


class TestWorkloadValidationError:
    def test_message_and_detail(self):
        err = WorkloadValidationError("negative token count", detail="input_tokens=-5")
        assert err.message == "negative token count"
        assert err.detail == "input_tokens=-5"


class TestAgentSurgeConnectionError:
    def test_endpoint_attribute(self):
        err = AgentSurgeConnectionError(
            "connection refused", endpoint="http://10.0.0.1:8000", detail="ECONNREFUSED"
        )
        assert err.endpoint == "http://10.0.0.1:8000"
        assert err.message == "connection refused"
        assert err.detail == "ECONNREFUSED"

    def test_endpoint_defaults_to_none(self):
        err = AgentSurgeConnectionError("timeout")
        assert err.endpoint is None


class TestBackendError:
    def test_backend_attribute(self):
        err = BackendError("model not loaded", backend="vllm", detail="404")
        assert err.backend == "vllm"
        assert err.message == "model not loaded"
        assert err.detail == "404"

    def test_backend_defaults_to_none(self):
        err = BackendError("unknown error")
        assert err.backend is None
