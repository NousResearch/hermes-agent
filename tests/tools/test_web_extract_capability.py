"""Extraction must reach a capable provider and only advertise a usable route (H4 / #98618)."""

import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("web,available", [({}, True), ({"keyless_fallback": False}, False),
                                         ({"extract_backend": "ddgs"}, False)])
def test_extract_admission_and_dispatch_use_the_same_capability(monkeypatch, public_dns, web, available):
    from hermes_constants import get_hermes_home
    (Path(get_hermes_home()) / "config.yaml").write_text(json.dumps({"web": web}), encoding="utf-8")

    from tools import web_tools
    from tools.registry import registry, invalidate_check_fn_cache
    from plugins.web import keyless_mcp
    import requests

    # Keep real provider discovery, config, selection and dispatch. Only the external MCP transport is stubbed.
    calls = []
    def post(url, **kwargs):
        calls.append((url, kwargs["json"]["params"]))
        response = requests.Response()
        response.status_code = 200
        response.headers["Content-Type"] = "application/json"
        response._content = json.dumps({"result": {"content": [
            {"type": "text", "text": "Example Domain: document content from the extractor."}
        ]}}).encode()
        return response

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: True)
    monkeypatch.setattr(keyless_mcp, "_ring_cursor", 0)
    web_tools._ensure_web_plugins_loaded()
    invalidate_check_fn_cache()
    names = {d["function"]["name"] for d in registry.get_definitions({"web_search", "web_extract"})}
    assert "web_search" in names
    assert ("web_extract" in names) is available
    result = json.loads(registry.dispatch("web_extract", {"urls": ["https://example.com"]}))
    if available:
        assert "Example Domain" in result["results"][0]["content"]
        assert calls and calls[0][0] == keyless_mcp.EXA_MCP_URL
        assert web_tools._get_search_backend() == "ddgs"
    else:
        assert not calls
        if "extract_backend" in web:
            assert "search-only" in result["error"]


def test_autodetect_skips_provider_with_broken_capability_probe(monkeypatch):
    from plugins.web._common import BaseWebSearchProvider
    from tools import web_tools

    class BrokenProvider(BaseWebSearchProvider):
        NAME = "broken-extractor"

        def is_available(self):
            return True

        def supports_extract(self):
            raise RuntimeError("provider not initialized")

    provider = BrokenProvider()
    monkeypatch.setattr(web_tools, "_has_env", lambda name: False)
    monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
    monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: False)
    monkeypatch.setattr(web_tools, "_list_registered_web_providers", lambda: [provider])
    monkeypatch.setattr(web_tools, "_registered_web_provider", lambda name: provider)
    assert web_tools._autodetect_backend("extract") is None
