"""Shared fail-closed classification for delegation runtime exit reasons."""

from __future__ import annotations


_FATAL_DELEGATION_EXIT_REASONS = frozenset(
    {
        "all_retries_exhausted_no_response",
        "code_skew_detected",
        "empty_response_exhausted",
        "ollama_runtime_context_too_small",
        "unknown",
    }
)

_FATAL_DELEGATION_EXIT_PREFIXES = (
    "code_skew_attribute_error(",
    "error_near_max_iterations(",
    "local_processing_error(",
)

_PARTIAL_DELEGATION_EXIT_REASONS = frozenset(
    {
        "max_iterations",
        "partial_stream_recovery",
    }
)

_PARTIAL_DELEGATION_EXIT_PREFIXES = ("max_iterations_reached(",)


def is_fatal_delegation_exit_reason(value: object) -> bool:
    """Return whether *value* proves a fatal child-runtime exit."""
    reason = str(value or "").strip()
    return reason in _FATAL_DELEGATION_EXIT_REASONS or reason.startswith(
        _FATAL_DELEGATION_EXIT_PREFIXES
    )


def is_partial_delegation_exit_reason(value: object) -> bool:
    """Return whether *value* proves incomplete but potentially usable work."""
    reason = str(value or "").strip()
    return reason in _PARTIAL_DELEGATION_EXIT_REASONS or reason.startswith(
        _PARTIAL_DELEGATION_EXIT_PREFIXES
    )