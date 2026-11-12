"""The execute_code schema must only advertise client-callable search helpers."""

import pytest

from hermes_constants import get_hermes_home
from tools.code_execution_tool import SANDBOX_ALLOWED_TOOLS, build_execute_code_schema


@pytest.mark.parametrize("enabled", [None, {"web_search", "web_extract", "terminal"}])
def test_native_search_schema_directs_search_to_server_side_tool(enabled):
    (get_hermes_home() / "config.yaml").write_text(
        "web:\n  backend: firecrawl\n  search_backend: openai-native\n",
        encoding="utf-8",
    )
    original_tools = set(SANDBOX_ALLOWED_TOOLS if enabled is None else enabled)

    description = build_execute_code_schema(enabled, mode="project")["description"]

    assert "web_search(query" not in description
    assert "server-side web_search tool" in description
    assert "Call that tool directly" in description
    assert "hermes_tools.web_search cannot execute native search" in description
    assert "web_extract(urls" in description
    assert "terminal(command" in description
    assert (SANDBOX_ALLOWED_TOOLS if enabled is None else enabled) == original_tools


@pytest.mark.parametrize("backend", ["firecrawl", "exa", "ddgs"])
@pytest.mark.parametrize("enabled", [None, {"web_search", "web_extract", "terminal"}])
def test_client_search_schema_keeps_python_helper(backend, enabled):
    (get_hermes_home() / "config.yaml").write_text(
        f"web:\n  search_backend: {backend}\n",
        encoding="utf-8",
    )

    description = build_execute_code_schema(enabled, mode="project")["description"]

    assert "web_search(query" in description
    assert "server-side web_search tool" not in description
    assert "cannot execute native search" not in description
