"""A supplied login handle authorizes TOTP injection only on its bound origin."""

import json
from dataclasses import replace

import pytest


@pytest.fixture
def browser_vault(tmp_path, monkeypatch):
    from agent import vault_store
    from agent.vault_backends import base, unlock
    from agent.vault_backends.local import LocalLoginBackend
    from tools import browser_vault_tool
    from tools.registry import registry

    store = vault_store.VaultStore(base_dir=tmp_path / "vault")
    item = store.add_item("login", "Example", {
        "identifier_type": "username", "identifier": "fake-user", "password": "fake-password",
        "otp_secret": "JBSWY3DPEHPK3PXP",
    }, origin="https://accounts.example.test")
    backend = LocalLoginBackend()
    state = {"page": item.origin, "fills": [], "resolved": [], "asked": [], "focused": [],
             "fill_result": {"filled": 1}}
    resolve = store.resolve_secret

    def resolve_secret(handle):
        state["resolved"].append(handle)
        return resolve(handle)

    def evaluate(_task_id, expression):
        controls = [{"index": 0, "type": "text", "name": "otp", "autocomplete": "one-time-code"}]
        return {"success": True, "result": json.dumps(controls) if "querySelectorAll" in expression else state["page"]}

    def fill(_task_id, expression):
        state["fills"].append(expression)
        return {"success": True, "result": json.dumps(state["fill_result"])}

    monkeypatch.setattr(vault_store, "get_vault_store", lambda: store)
    monkeypatch.setattr(vault_store, "totp_now", lambda _seed: "123456")
    monkeypatch.setattr(store, "resolve_secret", resolve_secret)
    monkeypatch.setattr(base, "enabled_backends", lambda: [backend])
    monkeypatch.setattr(browser_vault_tool, "_focus_bound_origin",
                        lambda *args: state["focused"].append(args))
    monkeypatch.setattr(browser_vault_tool, "_eval_js", evaluate)
    monkeypatch.setattr(browser_vault_tool, "_eval_js_secret", fill)
    monkeypatch.setattr(unlock, "can_prompt_here", lambda: True)
    unlock.set_code_prompt_callback(lambda site, _hint: state["asked"].append(site) or "654321")

    def dispatch(handle):
        return registry.dispatch("browser_vault_enter_code", {"handle": handle}, task_id="fake-browser-task")

    yield item, backend, state, dispatch
    unlock.set_code_prompt_callback(None)


@pytest.mark.parametrize("attack", ["host", "scheme", "port", "subdomain", "missing", "unbound", "wrong_kind", "unknown_backend"])
def test_supplied_handle_is_rejected_before_secret_resolution_or_prompt(browser_vault, monkeypatch, attack):
    item, backend, state, dispatch = browser_vault
    pages = {"host": "https://attacker.example.test", "scheme": "http://accounts.example.test",
             "port": "https://accounts.example.test:8443", "subdomain": "https://sub.accounts.example.test"}
    state["page"] = pages.get(attack, item.origin)
    if attack in {"missing", "unbound", "wrong_kind"}:
        meta = None if attack == "missing" else replace(item, **({"origin": None} if attack == "unbound" else {"kind": "payment"}))
        monkeypatch.setattr(backend, "get_meta", lambda _handle: meta)
    raw = dispatch("unavailable:fake-handle" if attack == "unknown_backend" else item.id)

    assert json.loads(raw)["success"] is False
    assert state["resolved"] == [], "An origin refusal must happen before the TOTP seed is read"
    assert state["fills"] == []
    assert state["asked"] == [], "An invalid supplied handle must not fall through to a user code prompt"
    assert "123456" not in raw and "654321" not in raw


@pytest.mark.parametrize("flow", ["saved", "handleless", "no_seed", "navigation"])
def test_bound_and_user_code_flows_preserve_final_origin_fence(browser_vault, monkeypatch, flow):
    item, backend, state, dispatch = browser_vault
    if flow == "no_seed":
        monkeypatch.setattr(backend, "resolve_otp", lambda _handle: None)
    if flow == "navigation":
        state["fill_result"] = {"refused": "origin_changed"}
    raw = dispatch("" if flow == "handleless" else item.id)
    result = json.loads(raw)

    assert result["success"] is (flow != "navigation")
    if flow == "navigation":
        assert result["error_type"] == "origin_changed"
    else:
        assert result["source"] == ("user" if flow in {"handleless", "no_seed"} else "local")
    assert len(state["fills"]) == 1
    assert item.origin in state["fills"][0]
    assert "123456" not in raw and "654321" not in raw
    assert bool(state["asked"]) is (flow in {"handleless", "no_seed"})
    if flow != "handleless":
        assert state["focused"] == [("fake-browser-task", item.origin, "otp")]
