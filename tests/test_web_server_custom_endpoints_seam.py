"""Seam-identity + aggressive tests for the custom-endpoints extraction (R3-C1).

``hermes_cli/web_routers/custom_endpoints.py`` holds the dashboard's
custom OpenAI-compatible provider endpoint CRUD + validate family, moved
out of ``hermes_cli/web_server.py`` (god-file slice R3-C1, epic #78791).

The seam-identity tests pin the regression this extraction is meant to
prevent: ``web_server`` must resolve every moved name to the *same object*
the router module defines.  The aggressive tests then exercise the failure
modes the endpoint surface must survive: missing entries, bad API keys,
env-ref display, and model-id parsing edge cases.
"""

from fastapi.testclient import TestClient

from hermes_cli import web_server as ws
from hermes_cli.web_routers import custom_endpoints as c

MOVED_NAMES = (
    "_parse_model_ids",
    "_custom_endpoint_id",
    "_models_from_custom_endpoint_entry",
    "_api_key_display",
    "_config_api_key_is_env_ref",
    "_custom_endpoint_response",
    "_detach_main_model_from_provider",
    "_write_custom_endpoint",
    "list_custom_endpoints",
    "upsert_custom_endpoint",
    "activate_custom_endpoint",
    "delete_custom_endpoint",
    "validate_custom_endpoint",
)


def test_moved_names_are_seam_identical():
    # ``is``-identity: web_server must resolve each moved name to the very
    # same object the router module defines — no redefinition allowed.
    for name in MOVED_NAMES:
        assert getattr(ws, name, None) is getattr(c, name, None), name


def test_custom_endpoint_routes_registered():
    paths = [rt.path for rt in ws.app.routes if "custom-endpoints" in getattr(rt, "path", "")]
    assert "/api/providers/custom-endpoints" in paths
    assert "/api/providers/custom-endpoints/validate" in paths
    assert "/api/providers/custom-endpoints/{endpoint_id}/activate" in paths


def _fake_resp(status_ok=True, payload=None):
    class _R:
        def __init__(self):
            self.is_success = status_ok
            self._payload = payload
        def json(self):
            return self._payload
    return _R()


def test_parse_model_ids_openai_shape():
    resp = _fake_resp(payload={"data": [{"id": "gpt-5.6"}, {"id": "deepseek-v4-flash"}]})
    assert c._parse_model_ids(resp) == ["gpt-5.6", "deepseek-v4-flash"]


def test_parse_model_ids_bare_list_shape():
    resp = _fake_resp(payload={"data": ["m1", "m2"]})
    assert c._parse_model_ids(resp) == ["m1", "m2"]


def test_parse_model_ids_empty():
    assert c._parse_model_ids(_fake_resp(payload={})) == []
    assert c._parse_model_ids(_fake_resp(payload={"data": []})) == []


def test_parse_model_ids_http_failure():
    assert c._parse_model_ids(_fake_resp(status_ok=False)) == []


def test_config_api_key_is_env_ref_true(monkeypatch):
    monkeypatch.setattr(c, "read_raw_config",
                        lambda: {"providers": {"myep": {"api_key": "${MY_KEY}"}}})
    assert c._config_api_key_is_env_ref("myep") is True


def test_config_api_key_is_env_ref_false(monkeypatch):
    monkeypatch.setattr(c, "read_raw_config",
                        lambda: {"providers": {"myep": {"api_key": "sk-plain"}}})
    assert c._config_api_key_is_env_ref("myep") is False


def test_config_api_key_is_env_ref_missing(monkeypatch):
    monkeypatch.setattr(c, "read_raw_config", lambda: {"providers": {}})
    assert c._config_api_key_is_env_ref("nope") is False


def test_api_key_display_plaintext_redacted():
    # Plaintext key: (has_key=True, redacted preview) — never the full secret.
    ok, shown = c._api_key_display({"api_key": "sk-abcdefghijklmnop"})
    assert ok is True
    assert shown != "sk-abcdefghijklmnop"


def test_api_key_display_env_ref():
    # key_env entries render as ${VAR}, key stays in .env.
    ok, shown = c._api_key_display({"key_env": "MY_KEY"})
    assert ok is True
    assert shown == "${MY_KEY}"


def test_api_key_display_none():
    assert c._api_key_display({}) == (False, None)


def test_custom_endpoint_id_normalizes():
    ident = c._custom_endpoint_id("My Endpoint")
    assert isinstance(ident, str)
    assert len(ident) > 0
    assert c._custom_endpoint_id("", "fallback-id") == "fallback-id"
