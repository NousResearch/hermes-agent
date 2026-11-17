"""Native vault delivery against real Chromium JavaScript-masked widgets.

All credentials and pages are synthetic. The real registry, encrypted vault and
CDP supervisor run; no credential provider or external website is contacted.
"""

import json
import shutil
import time

import pytest
from playwright.sync_api import sync_playwright

from agent.vault_store import VaultStore
from tools import browser_vault_tool
from tools.browser_supervisor import CDPSupervisor, SUPERVISOR_REGISTRY
from tools.registry import registry


SECRET = "synthetic-vault-password-8173"
ORIGIN = "https://vault-widget.test"


@pytest.fixture
def native_widget(tmp_path, monkeypatch):
    store = VaultStore(base_dir=tmp_path / "vault")
    meta = store.add_item(
        kind="login",
        label="Synthetic widget",
        origin=ORIGIN,
        secret={
            "identifier_type": "username",
            "identifier": "synthetic-user",
            "password": SECRET,
            "origin": ORIGIN,
        },
    )
    resolved = []
    original_resolve = store.resolve_secret

    def resolve(handle):
        resolved.append(handle)
        return original_resolve(handle)

    monkeypatch.setattr(store, "resolve_secret", resolve)
    monkeypatch.setattr("agent.vault_store.get_vault_store", lambda: store)
    with sync_playwright() as playwright:
        profile = tmp_path / "chromium"
        context = playwright.chromium.launch_persistent_context(
            str(profile),
            headless=True,
            args=["--remote-debugging-port=0"],
        )
        port_file = profile / "DevToolsActivePort"
        deadline = time.monotonic() + 10
        while not port_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        port, endpoint = port_file.read_text().splitlines()[:2]
        page = context.pages[0]
        page.route(
            ORIGIN + "/**",
            lambda route: route.fulfill(
                body="<!doctype html><title>Synthetic vault widget</title><body></body>",
                content_type="text/html",
            ),
        )
        page.goto(ORIGIN + "/login")
        supervisor = CDPSupervisor(
            "synthetic-widget", "ws://127.0.0.1:" + port + endpoint
        )
        supervisor.start()
        supervisor._synthetic_contexts = []
        original_cdp = supervisor._cdp

        async def instrument_observers(method, params=None, **kwargs):
            response = await original_cdp(method, params, **kwargs)
            if method == "Page.createIsolatedWorld":
                context_id = response["result"]["executionContextId"]
                supervisor._synthetic_contexts.append(
                    (
                        context_id,
                        kwargs["session_id"],
                    )
                )
                await original_cdp(
                    "Runtime.evaluate",
                    {
                        "contextId": context_id,
                        "expression": """(() => {
                      const NativeObserver=MutationObserver; globalThis.liveObservers=0;
                      globalThis.MutationObserver=class extends NativeObserver {
                        observe(...args){super.observe(...args);if(!this.live){this.live=true;liveObservers++;}}
                        disconnect(){super.disconnect();if(this.live){this.live=false;liveObservers--;}}
                      };
                    })()""",
                    },
                    **kwargs,
                )
            return response

        monkeypatch.setattr(supervisor, "_cdp", instrument_observers)
        monkeypatch.setattr(SUPERVISOR_REGISTRY, "get", lambda task_id: supervisor)
        monkeypatch.setattr(
            browser_vault_tool, "_ensure_supervisor", lambda task_id: supervisor
        )
        try:
            yield page, meta, resolved, supervisor
        finally:
            supervisor.stop()
            context.close()
            shutil.rmtree(profile)


def active_observers(supervisor):
    from tools.browser_supervisor import _schedule

    counts = []
    for context_id, session_id in supervisor._synthetic_contexts:
        try:
            response = _schedule(
                supervisor._cdp(
                    "Runtime.evaluate",
                    {
                        "contextId": context_id,
                        "expression": "globalThis.liveObservers",
                        "returnByValue": True,
                    },
                    session_id=session_id,
                ),
                supervisor._loop,
                timeout=5,
            )
            counts.append(response["result"]["result"].get("value", 0))
        except Exception:
            counts.append(
                0
            )  # A destroyed execution context no longer owns an observer.
    return counts


def widget(page, kind):
    attributes = {
        "supported": "",
        "plain": "",
        "unmasked": "",
        "prefilled": 'value="already-present"',
        "readonly": "readonly",
        "hidden": 'style="display:none"',
        "new": 'autocomplete="new-password"',
        "otp": 'autocomplete="one-time-code"',
        "cleanup_failure": "",
        "duplicate": "",
        "standard_and_text": 'autocomplete="current-password"',
        "reactive_class": "",
        "duplicate_wrapper": "",
        "disabled": "disabled",
        "css_only": 'style="-webkit-text-security:disc"',
        "missing_form": "",
        "probe_rename": "",
        "probe_throw": "",
        "stalled_frames": "",
        "main_world_tamper": "",
        "prepare_return_timeout": "",
        "prepare_running_timeout": "",
        "short_limit": 'maxlength="8"',
        "zero_limit": 'maxlength="0"',
    }
    duplicate = (
        '<input name="otherPassword" aria-label="Password" type="text">'
        if kind == "duplicate"
        else ""
    )
    if kind == "standard_and_text":
        duplicate = '<input id="standard" name="password" type="password">'
    page.set_content(
        '<base href="'
        + ORIGIN
        + '/base/"><form id="login" action="session"><label for="password">Current password</label>'
        '<input id="password" name="userPassword" type="text" '
        + attributes[kind]
        + ">"
        + duplicate
        + "<button disabled>Continue</button></form>"
    )
    if kind == "duplicate_wrapper":
        page.evaluate("""() => {
          const el=document.querySelector('input');const wrapper=document.createElement('div');
          wrapper.id=el.id;el.replaceWith(wrapper);wrapper.append(el);
        }""")
    if kind == "missing_form":
        page.evaluate("""() => {
          const form=document.querySelector('form');const div=document.createElement('div');
          div.append(...form.childNodes);form.replaceWith(div);
        }""")
    page.evaluate(
        """kind => {
      window.syntheticState = ''; window.syntheticSubmissions = 0;
      document.querySelector('form')?.addEventListener('submit', e => {
        e.preventDefault(); window.syntheticSubmissions++;
      });
      const input = document.querySelector('input');
      input.addEventListener('input', () => {
        if (window.syntheticReveal) return;
        if (window.syntheticThrow && input.value.startsWith('synthetic-vault-')) throw new Error(input.value);
        if (kind === 'probe_rename' && input.value) input.name = 'changedName';
        if (kind === 'probe_throw' && input.value) throw new Error('Synthetic probe failure');
        if (kind === 'reactive_class') {
          input.classList.add('dirty'); input.classList.toggle('has-value', !!input.value);
        }
        if (kind === 'plain' || kind === 'unmasked' || kind === 'css_only') return;
        if (kind === 'cleanup_failure' && !input.value) {
          input.value = '●'; return;
        }
        window.syntheticState = input.value;
        input.value = '●'.repeat(input.value.length);
      });
    }""",
        kind,
    )


@pytest.mark.parametrize(
    "kind",
    [
        "supported",
        "plain",
        "unmasked",
        "prefilled",
        "readonly",
        "hidden",
        "new",
        "otp",
        "cleanup_failure",
        "duplicate",
        "standard_and_text",
        "reactive_class",
        "duplicate_wrapper",
        "disabled",
        "css_only",
        "missing_form",
        "probe_rename",
        "probe_throw",
        "stalled_frames",
        "main_world_tamper",
        "prepare_return_timeout",
        "prepare_running_timeout",
        "short_limit",
        "zero_limit",
    ],
)
def test_native_fill_requires_masking_before_secret_resolution(
    native_widget, kind, monkeypatch
):
    page, meta, resolved, supervisor = native_widget
    widget(page, kind)
    late_preparations = []
    if kind in {"prepare_return_timeout", "prepare_running_timeout"}:
        from tools import browser_supervisor
        from agent.async_utils import safe_schedule_threadsafe

        schedule = browser_supervisor._schedule
        prepare = supervisor.prepare_masked_password
        armed = []

        def timeout_schedule(coro, loop, **kwargs):
            if armed:
                armed.clear()
                if kind == "prepare_return_timeout":
                    schedule(coro, loop, **kwargs)
                else:
                    late_preparations.append(safe_schedule_threadsafe(coro, loop))
                raise TimeoutError("Synthetic lost preparation response")
            return schedule(coro, loop, **kwargs)

        def timeout_preparation(*args, **kwargs):
            armed.append(True)
            return prepare(*args, **kwargs)

        monkeypatch.setattr(browser_supervisor, "_schedule", timeout_schedule)
        monkeypatch.setattr(supervisor, "prepare_masked_password", timeout_preparation)
    if kind == "main_world_tamper":
        page.evaluate("""() => {
          document.querySelectorAll = () => [];
          Object.defineProperty = () => { throw new Error('Main-world override'); };
          Object.getOwnPropertyDescriptor = () => { throw new Error('Main-world override'); };
          window.MutationObserver = function(){return {observe(){},takeRecords(){return []},disconnect(){}}};
        }""")
    if kind == "stalled_frames":
        cdp = supervisor._cdp

        async def without_animation_frames(method, params=None, **kwargs):
            if method == "Runtime.evaluate" and params and "contextId" in params:
                params = {
                    **params,
                    "expression": "(globalThis.requestAnimationFrame=()=>0,"
                    + params["expression"]
                    + ")",
                }
            return await cdp(method, params, **kwargs)

        monkeypatch.setattr(supervisor, "_cdp", without_animation_frames)
    result = registry.dispatch(
        "browser_vault",
        {"action": "fill", "handle": meta.id},
        task_id="synthetic-widget",
    )
    out = json.loads(result) if isinstance(result, str) else result
    for future in late_preparations:
        future.result(timeout=15)
    assert SECRET not in json.dumps(out)
    assert page.evaluate("window.syntheticSubmissions") == 0
    if kind in {
        "supported",
        "standard_and_text",
        "reactive_class",
        "duplicate_wrapper",
        "main_world_tamper",
    }:
        assert out.get("success") is True, out
        assert out["filled_fields"] == 1
        assert len(resolved) == 1
        if kind == "standard_and_text":
            assert page.locator("#standard").input_value() == SECRET
            assert page.locator("#password").input_value() == ""
            assert page.evaluate("window.syntheticState") == ""
        else:
            assert page.evaluate("s => window.syntheticState === s", SECRET)
            assert page.locator("input#password").input_value() == "●" * len(SECRET)
    else:
        assert out.get("success") is False, out
        assert len(resolved) == (1 if kind == "short_limit" else 0)
        assert page.evaluate("s => window.syntheticState !== s", SECRET)
        if kind in {"short_limit", "zero_limit"}:
            assert page.locator("input").input_value() == ""
        if kind in {"probe_rename", "probe_throw", "stalled_frames"}:
            assert page.locator("input").input_value() == ""
    assert not supervisor._vault_widget_proofs
    assert all(count == 0 for count in active_observers(supervisor))


@pytest.mark.parametrize(
    "mutation",
    [
        "replace",
        "move_form",
        "rename",
        "hide",
        "change_and_revert",
        "navigate",
        "focus_other_tab",
        "focus_replace",
        "reveal_after_probe",
        "throw_after_probe",
        "base_href",
        "tab_switch_at_delivery",
        "provider_exception",
        "redaction_exception",
        "delivery_exception",
        "replay",
    ],
)
def test_native_proof_is_context_bound_and_single_use(
    native_widget, monkeypatch, mutation
):
    page, meta, resolved, supervisor = native_widget
    widget(page, "supported")
    if mutation == "provider_exception":
        from agent.vault_store import get_vault_store

        def failed_provider(handle):
            resolved.append(handle)
            raise RuntimeError(SECRET)

        monkeypatch.setattr(get_vault_store(), "resolve_secret", failed_provider)
    elif mutation == "redaction_exception":

        def failed_redaction(value):
            raise RuntimeError(SECRET)

        monkeypatch.setattr(
            "agent.redact.register_vault_redaction_value", failed_redaction
        )
    elif mutation == "delivery_exception":
        cdp = supervisor._cdp

        async def failed_private_delivery(method, params=None, **kwargs):
            if (
                method == "Runtime.callFunctionOn"
                and params
                and params.get("arguments")
            ):
                raise RuntimeError(SECRET)
            return await cdp(method, params, **kwargs)

        monkeypatch.setattr(supervisor, "_cdp", failed_private_delivery)
    check = supervisor.check_masked_password
    fill = supervisor.fill_masked_password
    captured = []
    scripts = {
        "replace": "const el=document.querySelector('#password');el.replaceWith(el.cloneNode(true));",
        "move_form": "const form=document.createElement('form');document.body.append(form);form.append(document.querySelector('#password'));",
        "rename": "document.querySelector('#password').name='anotherPassword';",
        "hide": "document.querySelector('#password').style.display='none';",
        "change_and_revert": "const el=document.querySelector('#password');el.type='password';el.type='text';",
        "base_href": "document.querySelector('base').href='https://different.test/base/';",
    }

    def checked(proof, **kwargs):
        if mutation in scripts:
            page.evaluate("() => {" + scripts[mutation] + "}")
        elif mutation == "navigate":
            page.goto(ORIGIN + "/different")
        elif mutation == "focus_other_tab":
            other = page.context.new_page()
            other.route(
                ORIGIN + "/**",
                lambda route: route.fulfill(body="<title>Other synthetic tab</title>"),
            )
            other.goto(ORIGIN + "/other")
            assert supervisor.focus_page(
                ORIGIN, accept="location.pathname === '/other'"
            ).get("ok")
        return check(proof, **kwargs)

    def delivered(proof, password, **kwargs):
        captured.append(proof)
        if mutation == "focus_replace":
            page.evaluate("""() => {
              const el=document.querySelector('#password');el.blur();
              el.addEventListener('focus', () => el.replaceWith(el.cloneNode(true)), {once:true});
            }""")
        elif mutation == "reveal_after_probe":
            page.evaluate("window.syntheticReveal = true")
        elif mutation == "throw_after_probe":
            page.evaluate("window.syntheticThrow = true")
        return fill(proof, password, **kwargs)

    if mutation == "tab_switch_at_delivery":
        from tools import browser_supervisor

        schedule = browser_supervisor._schedule
        armed = []
        ordinary_delivery = delivered

        def switch_before_scheduling(coro, loop, **kwargs):
            if armed:
                armed.clear()
                other = page.context.new_page()
                other.route(
                    ORIGIN + "/**",
                    lambda route: route.fulfill(body="<title>Delivery race</title>"),
                )
                other.goto(ORIGIN + "/race")
                assert supervisor.focus_page(
                    ORIGIN, accept="location.pathname === '/race'"
                ).get("ok")
            return schedule(coro, loop, **kwargs)

        def armed_delivery(proof, password, **kwargs):
            armed.append(True)
            return ordinary_delivery(proof, password, **kwargs)

        monkeypatch.setattr(browser_supervisor, "_schedule", switch_before_scheduling)
        delivered = armed_delivery

    monkeypatch.setattr(supervisor, "check_masked_password", checked)
    monkeypatch.setattr(supervisor, "fill_masked_password", delivered)
    raw = registry.dispatch(
        "browser_vault",
        {"action": "fill", "handle": meta.id},
        task_id="synthetic-widget",
    )
    out = json.loads(raw) if isinstance(raw, str) else raw
    assert SECRET not in json.dumps(out)
    if mutation == "replay":
        assert out.get("success") is True, out
        assert fill(captured[0], SECRET).get("ok") is False
        assert len(resolved) == 1
    else:
        assert out.get("success") is False, out
        assert len(resolved) == (
            1
            if mutation
            in {
                "focus_replace",
                "reveal_after_probe",
                "throw_after_probe",
                "tab_switch_at_delivery",
                "provider_exception",
                "redaction_exception",
                "delivery_exception",
            }
            else 0
        )
        assert page.locator("input").evaluate_all(
            "(els,s) => els.every(el => el.value !== s)", SECRET
        )
    assert not supervisor._vault_widget_proofs
    assert all(count == 0 for count in active_observers(supervisor))


def test_masked_page_backend_never_falls_through_to_another_browser(monkeypatch):
    from tools.browser_vault_masked_password import fill_masked_password

    monkeypatch.setattr(browser_vault_tool, "_page_backend", lambda task_id: object())

    def unrelated_browser(*args, **kwargs):
        pytest.fail("A page backend must never fall through to a CDP browser")

    monkeypatch.setattr(browser_vault_tool, "_ensure_supervisor", unrelated_browser)
    out = json.loads(fill_masked_password("synthetic-widget", None, "opaque", ORIGIN))
    assert out["success"] is False
    assert out["error_type"] == "unsupported_password_widget"


def test_combined_masked_fill_is_explicitly_unsupported(native_widget):
    page, meta, resolved, _ = native_widget
    widget(page, "supported")
    out = json.loads(
        browser_vault_tool.browser_vault_fill(
            meta.id, task_id="synthetic-widget", with_identifier=True
        )
    )
    assert out["success"] is False
    assert out["error_type"] == "unsupported_combined_masked_fill"
    assert not resolved
    assert page.locator("#password").input_value() == ""
