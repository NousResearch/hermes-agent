"""Regression tests for capability-gated MCP utility schema registration.

Background
==========
For every connected MCP server, hermes-agent used to register four "utility"
tool schemas (``mcp_<server>_list_resources``, ``read_resource``,
``list_prompts``, ``get_prompt``) regardless of whether the server actually
advertises those capabilities. The old gate used ``hasattr(server.session,
method)`` which always returned True because ``mcp.ClientSession`` defines
all four methods on the class — independent of what the remote server
supports.

Tools-only servers like ``@upstash/context7-mcp`` advertise
``{\"tools\": {\"listChanged\": true}}`` in their ``initialize`` response —
no ``prompts`` or ``resources`` keys — and they return JSON-RPC
``-32601 Method not found`` for ``prompts/list``, ``prompts/get``,
``resources/list``, ``resources/read``. The model would try the stubs,
get the error, and incorrectly conclude the MCP server was broken.

The fix captures the ``InitializeResult`` from
``await session.initialize()`` into ``MCPServerTask.initialize_result``
and gates utility schema registration on the advertised
``capabilities.resources`` / ``capabilities.prompts`` sub-objects. See
#18051 for the reporter's repro (Context7) and analysis.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _make_init_result(*, resources: bool, prompts: bool):
    """Build a fake ``InitializeResult`` whose ``capabilities`` sub-object
    matches a server that advertises exactly the given capability set.

    MCP spec shape: ``capabilities.resources`` / ``capabilities.prompts``
    are non-None iff the server implements the corresponding request
    family. We mirror that with ``SimpleNamespace`` because the real SDK
    models are pydantic and we don't want the test to couple to pydantic
    versioning.
    """
    caps_attrs: dict = {"tools": SimpleNamespace(listChanged=True)}
    caps_attrs["resources"] = SimpleNamespace(listChanged=True) if resources else None
    caps_attrs["prompts"] = SimpleNamespace(listChanged=True) if prompts else None
    return SimpleNamespace(capabilities=SimpleNamespace(**caps_attrs))


def _make_fake_server(*, initialize_result):
    """Build a stand-in ``MCPServerTask`` that exposes just the fields
    ``_select_utility_schemas`` inspects: ``name``, ``session``,
    ``initialize_result``.

    A plain ``MCPServerTask`` uses ``__slots__`` and needs an asyncio
    loop for the ``Event``/``Lock`` init — overkill for unit scope.
    """
    server = MagicMock()
    server.name = "test-server"
    # session must satisfy the legacy ``hasattr`` fallback too
    server.session = MagicMock(
        spec=["list_resources", "read_resource", "list_prompts", "get_prompt"]
    )
    server.initialize_result = initialize_result
    return server


def _handler_keys(selected):
    return {entry["handler_key"] for entry in selected}


class TestCapabilityGatedRegistration:
    def test_tools_only_server_gets_no_utility_schemas(self):
        """Context7-shaped server (tools only, no prompts / resources) should
        get zero utility stubs registered — this is the exact scenario
        from the #18051 bug report."""
        from tools.mcp_tool import _select_utility_schemas

        server = _make_fake_server(
            initialize_result=_make_init_result(resources=False, prompts=False)
        )
        selected = _select_utility_schemas("context7", server, {})
        assert _handler_keys(selected) == set(), (
            f"tools-only server should have zero utility stubs, got "
            f"{_handler_keys(selected)}"
        )

    def test_resources_only_server_gets_resource_stubs_only(self):
        from tools.mcp_tool import _select_utility_schemas

        server = _make_fake_server(
            initialize_result=_make_init_result(resources=True, prompts=False)
        )
        selected = _select_utility_schemas("res-only", server, {})
        assert _handler_keys(selected) == {"list_resources", "read_resource"}

    def test_prompts_only_server_gets_prompt_stubs_only(self):
        from tools.mcp_tool import _select_utility_schemas

        server = _make_fake_server(
            initialize_result=_make_init_result(resources=False, prompts=True)
        )
        selected = _select_utility_schemas("prompt-only", server, {})
        assert _handler_keys(selected) == {"list_prompts", "get_prompt"}

    def test_fully_capable_server_gets_all_four_stubs(self):
        from tools.mcp_tool import _select_utility_schemas

        server = _make_fake_server(
            initialize_result=_make_init_result(resources=True, prompts=True)
        )
        selected = _select_utility_schemas("full", server, {})
        assert _handler_keys(selected) == {
            "list_resources", "read_resource", "list_prompts", "get_prompt",
        }


class TestConfigFilterStillApplies:
    """Per-server config flags ``tools.resources: false`` / ``tools.prompts: false``
    must continue to override even when the server DOES advertise the capability."""

    def test_config_disables_resources_even_when_advertised(self):
        from tools.mcp_tool import _select_utility_schemas

        server = _make_fake_server(
            initialize_result=_make_init_result(resources=True, prompts=True)
        )
        selected = _select_utility_schemas(
            "full-but-filtered",
            server,
            {"tools": {"resources": False}},
        )
        assert _handler_keys(selected) == {"list_prompts", "get_prompt"}

    def test_config_disables_prompts_even_when_advertised(self):
        from tools.mcp_tool import _select_utility_schemas

        server = _make_fake_server(
            initialize_result=_make_init_result(resources=True, prompts=True)
        )
        selected = _select_utility_schemas(
            "full-but-filtered",
            server,
            {"tools": {"prompts": False}},
        )
        assert _handler_keys(selected) == {"list_resources", "read_resource"}


class TestLegacyFallback:
    """When ``initialize_result`` is missing (older test fixtures or code
    paths that haven't captured it yet), fall back to the legacy hasattr
    check so pre-existing tests and servers keep working."""

    def test_no_initialize_result_falls_back_to_hasattr_check(self):
        from tools.mcp_tool import _select_utility_schemas

        server = _make_fake_server(initialize_result=None)
        # With the legacy fallback, session.spec includes all four methods,
        # so all four stubs should register (old behavior).
        selected = _select_utility_schemas("legacy", server, {})
        assert _handler_keys(selected) == {
            "list_resources", "read_resource", "list_prompts", "get_prompt",
        }

    def test_no_initialize_result_respects_session_spec(self):
        """Legacy fallback still filters by ``hasattr(session, method)``, so
        a session whose spec lacks a method is correctly skipped."""
        from tools.mcp_tool import _select_utility_schemas

        server = _make_fake_server(initialize_result=None)
        # Override session to a spec that only has list_resources
        server.session = MagicMock(spec=["list_resources"])
        selected = _select_utility_schemas("legacy-partial", server, {})
        assert _handler_keys(selected) == {"list_resources"}
