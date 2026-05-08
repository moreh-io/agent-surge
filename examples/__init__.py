"""Example scripts demonstrating agentsurge usage.

These examples serve as API contract - they exercise the public surface
of agentsurge and are expected to run in CI against the mock backend.

Examples
--------
01_quick_start.py
    MockBackend basics: send turns, inspect results, call log.
02_custom_sessions.py
    Build ReplaySession objects, serialize/validate workloads, replay.
03_callbacks.py
    on_turn / on_session callback hooks for progress monitoring.
04_custom_backend.py
    Define a custom backend (auto-registered via BackendBase subclass).
05_run_api.py
    agentsurge.run() programmatic API (mocked for CI, live optional).
"""
