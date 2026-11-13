"""browser_vault on a plugin browser with no CDP endpoint (Camofox): the registered
``BrowserPageBackend`` holding the task's page carries every vault script, and the secret path never
lets a page or backend error echo the credential."""

import json
from unittest.mock import patch

import pytest

from agent.browser_page_backend import BrowserPageBackend
from agent.vault_store import VaultStore

SECRET = "s3cret-pw"
CONTROLS = [
    {"autocomplete": "email", "formIndex": 0, "index": 0, "label": "", "name": "email", "type": "email"},
    {"autocomplete": "current-password", "formIndex": 0, "index": 1, "label": "", "name": "pw", "type": "password"},
]


class FakePage(BrowserPageBackend):
    name = "fakepage"

    def __init__(self, url):
        self.url, self.calls, self.fail_secret = url, [], False

    def has_page(self, task_id):
        return True

    def evaluate(self, task_id, expression, *, secret):
        self.calls.append((expression, secret))
        if secret and self.fail_secret:
            raise RuntimeError(f"page.evaluate failed: {expression}")
        if "location.href" in expression:
            return self.url
        if SECRET in expression:
            return json.dumps({"filled": 1})
        return json.dumps(CONTROLS)


@pytest.fixture()
def page(tmp_path, monkeypatch):
    from agent import browser_page_registry
    from tools import browser_vault_tool

    def cdp_used(*args, **kwargs):
        raise AssertionError("CDP supervisor used while a page backend holds the task's page")

    monkeypatch.setattr(browser_vault_tool, "_ensure_supervisor", cdp_used)
    fake = FakePage("https://example.com/login")
    browser_page_registry.register_provider(fake)
    store = VaultStore(base_dir=tmp_path / "vault")
    meta = store.add_item(kind="login", label="Example", origin="https://example.com",
                          secret={"identifier_type": "email", "identifier": "a@example.com", "password": SECRET})
    with patch("agent.vault_store.get_vault_store", return_value=store):
        yield fake, meta
    browser_page_registry._reset_for_tests()


def test_fill_runs_on_the_page_backend_and_refuses_another_origin(page):
    from tools import browser_vault_tool

    fake, meta = page
    raw = browser_vault_tool.browser_vault_fill(meta.id)
    out = json.loads(raw)
    assert out["success"] is True and out["filled_fields"] == 1 and SECRET not in raw
    secret_calls = [expr for expr, secret in fake.calls if SECRET in expr]
    assert len(secret_calls) == 1 and all(secret for expr, secret in fake.calls if SECRET in expr)

    fake.calls.clear()
    fake.url = "https://evil.example/login"
    out = json.loads(browser_vault_tool.browser_vault_fill(meta.id))
    assert out["error_type"] == "origin_mismatch"
    assert not any(SECRET in expr for expr, _ in fake.calls)


@pytest.mark.parametrize("probe_result", [False, None, [], "false"])
def test_plugin_otp_refuses_before_vault_lookup(page, monkeypatch, probe_result):
    from tools import browser_vault_tool as tool
    fake, meta = page
    monkeypatch.setattr(fake, "evaluate", lambda *args, **kwargs: probe_result)
    def unexpected(*args):
        raise AssertionError("ineligible plugin page must not resolve saved login")
    monkeypatch.setattr(tool, "_saved_login", unexpected)
    out = json.loads(tool.browser_vault_enter_code(meta.id))
    assert out["error_type"] == "no_code_field"


def test_plugin_otp_probe_accepts_before_inspection(page, monkeypatch):
    from tools import browser_vault_tool as tool
    fake, meta = page
    calls = []
    def evaluate(task_id, expression, *, secret):
        calls.append((expression, secret))
        if expression == tool._TAB_PROBES["otp"]:
            return True
        if "location.href" in expression:
            return fake.url
        return json.dumps([])
    monkeypatch.setattr(fake, "evaluate", evaluate)
    monkeypatch.setattr(tool, "_saved_login", lambda handle: (None, None))
    out = json.loads(tool.browser_vault_enter_code())
    assert out["error_type"] == "no_code_field"  # inspection revalidates before any code prompt
    assert calls[0] == (tool._TAB_PROBES["otp"], False)
    assert len(calls) == 3


def test_secret_failure_never_echoes_the_script(page):
    from tools import browser_vault_tool

    fake, meta = page
    fake.fail_secret = True
    raw = browser_vault_tool.browser_vault_fill(meta.id)
    out = json.loads(raw)
    assert out["success"] is False and SECRET not in raw
