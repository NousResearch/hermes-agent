"""Unit tests for shared MCP profile namespace enforcement."""

from unittest.mock import patch

from tools.mcp_profile_scope import resolve_profile_scope, scope_tool_arguments


PROFILE_CONFIG = {
    "profile_scope": {
        "mode": "profile",
        "namespace_prefix": "hermes/profile",
        "shared_namespaces": ["hermes/shared"],
    }
}


def test_private_search_injects_active_profile_namespace():
    with patch("hermes_cli.profiles.get_active_profile_name", return_value="job-scout"):
        args, error = scope_tool_arguments(PROFILE_CONFIG, "sm_search_witnessed", {"query": "roles"})

    assert error is None
    assert args is not None
    assert args == {"query": "roles", "namespaces": ["hermes/profile/job-scout"]}


def test_private_search_rejects_namespace_spoofing():
    with patch("hermes_cli.profiles.get_active_profile_name", return_value="job-scout"):
        args, error = scope_tool_arguments(
            PROFILE_CONFIG,
            "sm_search",
            {"query": "roles", "namespaces": ["hermes/profile/public"]},
        )

    assert args is None
    assert error and "private MCP view" in error


def test_private_fact_write_is_forced_to_own_namespace_when_omitted():
    with patch("hermes_cli.profiles.get_active_profile_name", return_value="job-scout"):
        args, error = scope_tool_arguments(
            PROFILE_CONFIG,
            "sm_add_fact",
            {"content": "candidate prefers remote roles"},
        )

    assert error is None
    assert args is not None
    assert args["namespace"] == "hermes/profile/job-scout"


def test_private_direct_reads_are_blocked():
    with patch("hermes_cli.profiles.get_active_profile_name", return_value="job-scout"):
        args, error = scope_tool_arguments(
            PROFILE_CONFIG,
            "sm_get_fact",
            {"fact_id": "fact-1"},
        )

    assert args is None
    assert error and "private profile view" in error


def test_shared_view_is_read_only_and_restricted_to_shared_namespaces():
    config = {
        "profile_scope": {
            "mode": "shared",
            "shared_namespaces": ["hermes/shared"],
        }
    }
    with patch("hermes_cli.profiles.get_active_profile_name", return_value="job-scout"):
        args, error = scope_tool_arguments(config, "sm_search", {"query": "handoff"})
        write_args, write_error = scope_tool_arguments(config, "sm_add_fact", {"content": "x"})

    assert error is None
    assert args is not None
    assert args["namespaces"] == ["hermes/shared"]
    assert write_args is None
    assert write_error and "read-only" in write_error


def test_cross_profile_view_requires_explicit_profile_namespaces():
    config = {
        "profile_scope": {
            "mode": "cross_profile",
            "namespace_prefix": "hermes/profile",
        }
    }
    with patch("hermes_cli.profiles.get_active_profile_name", return_value="job-scout"):
        missing, missing_error = scope_tool_arguments(config, "sm_search", {"query": "handoff"})
        explicit, explicit_error = scope_tool_arguments(
            config,
            "sm_search",
            {"query": "handoff", "namespaces": ["hermes/profile/public"]},
        )

    assert missing is None
    assert missing_error and "explicit namespaces" in missing_error
    assert explicit_error is None
    assert explicit["namespaces"] == ["hermes/profile/public"]


def test_present_but_invalid_scope_fails_closed():
    with patch("hermes_cli.profiles.get_active_profile_name", return_value="job-scout"):
        try:
            resolve_profile_scope({"profile_scope": {"mode": "unexpected"}}, "job-scout")
        except ValueError as error:
            assert "profile_scope.mode" in str(error)
        else:
            raise AssertionError("invalid profile scope was accepted")
