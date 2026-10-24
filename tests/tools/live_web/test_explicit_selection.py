"""Resolve actual bundled web providers without dispatching a service request."""

from contextlib import contextmanager
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from agent.secret_scope import reset_secret_scope, set_secret_scope
from agent import web_search_registry as registry
from hermes_cli.config import atomic_config_write
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import web_tools


@contextmanager
def web_home(home, web):
    home.mkdir()
    home_token = set_hermes_home_override(home)
    scope_token = set_secret_scope({}, profile_home=str(home))
    try:
        atomic_config_write(home / "config.yaml", {"web": web})
        web_tools._ensure_web_plugins_loaded()
        yield
    finally:
        reset_secret_scope(scope_token)
        reset_hermes_home_override(home_token)


@pytest.mark.parametrize("capability", ["search", "extract"])
def test_unknown_explicit_route_stops_before_dispatch(tmp_path, capability):
    absent = "uninstalled-web-provider"
    with web_home(tmp_path / capability, {f"{capability}_backend": absent, "keyless_rescue": False}):
        assert registry.get_provider(absent) is None
        resolver = registry.get_active_search_provider if capability == "search" else registry.get_active_extract_provider
        assert resolver() is None
        if capability == "search":
            result = json.loads(web_tools.web_search_tool("Hermes package management"))
        else:
            provider, error = web_tools._resolve_extract_provider(absent)
            assert provider is None
            result = json.loads(error)
        assert result["success"] is False and absent in result["error"]


def test_managed_route_keeps_the_actual_capable_bundled_providers(tmp_path):
    with web_home(tmp_path / "managed", {"backend": "nous"}):
        search = registry.get_active_search_provider()
        extract = registry.get_active_extract_provider()
        assert search is registry.get_provider("perplexity") and search.supports_search()
        assert extract is registry.get_provider("firecrawl") and extract.supports_extract()
        assert web_tools._get_search_backend() == search.name
        assert web_tools._get_extract_backend() == extract.name
