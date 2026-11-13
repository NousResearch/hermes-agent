"""The code prompt receives the page chosen for the fill, not its surface session (#152483).

The registered page backend executes the real inspection/fill scripts in Node with a minimal DOM.
Only that browser transport and the human's answer are substitutes; dispatch and origin/nonce
checks use production code against the runner's isolated home.
"""

import json
import subprocess

import pytest

from agent import browser_page_registry, redact
from agent.browser_page_backend import BrowserPageBackend
from agent.vault_backends import unlock
from tools import browser_vault_tool  # registers browser_vault
from tools.registry import registry


_DOM = r"""
const fs = require('node:fs');
const operations = JSON.parse(fs.readFileSync(0, 'utf8'));
let input;
globalThis.window = {location: {}};
globalThis.HTMLInputElement = class {
  constructor() {
    this.tagName = 'INPUT'; this.type = 'text'; this.autocomplete = 'one-time-code';
    this.name = 'otp'; this.id = ''; this.value = ''; this.attrs = {};
  }
  getClientRects() { return [{}]; }
  getAttribute(name) { return this.attrs[name] || null; }
  setAttribute(name, value) { this.attrs[name] = value; }
  removeAttribute(name) { delete this.attrs[name]; }
  focus() {}
  dispatchEvent() {}
};
globalThis.getComputedStyle = () => ({display: 'block', visibility: 'visible'});
globalThis.InputEvent = globalThis.Event = class {};
globalThis.document = {
  querySelectorAll(selector) {
    if (selector === '*' || selector === '[data-hermes-vault-slot]') return [input];
    const match = selector.match(/^\[data-hermes-vault-slot="(.*)"\]$/);
    return match && input.attrs['data-hermes-vault-slot'] === match[1] ? [input] : [];
  },
};
let result;
for (const operation of operations) {
  if ('navigate' in operation) {
    const url = new URL(operation.navigate);
    window.location = {href: url.href, origin: url.origin};
    input = new HTMLInputElement();
  } else {
    result = eval(operation.expression);
  }
}
console.log(JSON.stringify({result, value: input.value}));
"""


class CodePage(BrowserPageBackend):
    name = "code-prompt-test"

    def __init__(self, task_id, url):
        self.name = f"code-prompt-test-{task_id}"
        self.task_id = task_id
        self.operations = [{"navigate": url}]
        self.value = ""
        self.secret_calls = 0

    def has_page(self, task_id):
        return task_id == self.task_id

    def navigate(self, url):
        self.operations.append({"navigate": url})

    def evaluate(self, task_id, expression, *, secret):
        assert task_id == self.task_id
        self.secret_calls += int(secret)
        self.operations.append({"expression": expression})
        completed = subprocess.run(
            ["node", "-e", _DOM], input=json.dumps(self.operations),
            capture_output=True, text=True, check=True, timeout=10,
        )
        output = json.loads(completed.stdout)
        self.value = output["value"]
        return output["result"]


@pytest.fixture()
def page():
    previous = (unlock.get_unlock_prompt_callback(), unlock.get_code_prompt_callback(),
                unlock.get_current_session_id())
    backend = CodePage("fill-task", "https://login.example/verify")
    browser_page_registry.register_provider(backend)
    unlock.set_current_session_id("surface-session")
    unlock.set_unlock_prompt_callback(lambda _backend, _label: "")
    try:
        yield backend
    finally:
        browser_page_registry.restore_registration(backend.name, backend, None)
        unlock.set_unlock_prompt_callback(previous[0])
        unlock.set_code_prompt_callback(previous[1])
        unlock.set_current_session_id(previous[2])
        redact.clear_vault_redaction_values()


def _enter_code():
    return json.loads(registry.dispatch("browser_vault", {"action": "enter_code"}, task_id="fill-task"))


@pytest.mark.parametrize("origin", ["http://login.example", "https://login.example"])
@pytest.mark.parametrize("callback_kind", ["keyword", "kwargs"])
def test_prompt_receives_the_origin_selected_for_the_fill(page, origin, callback_kind):
    page.navigate(origin + "/verify")
    # The UI's session can point at the same host over a different scheme.
    surface = CodePage("surface-session", ("https" if origin.startswith("http:") else "http")
                       + "://login.example/other")
    browser_page_registry.register_provider(surface)
    asked = []

    def prompt(site, hint, *, origin):
        asked.append((site, hint, origin))
        return "246 810"

    def prompt_kwargs(site, hint, **kwargs):
        return prompt(site, hint, **kwargs)

    unlock.set_code_prompt_callback(prompt if callback_kind == "keyword" else prompt_kwargs)
    try:
        result = _enter_code()
    finally:
        browser_page_registry.restore_registration(surface.name, surface, None)
    assert result["success"] is True, result
    assert result["origin"] == origin
    assert asked == [("login.example", "", origin)]
    assert page.value == "246810" and surface.secret_calls == 0


@pytest.mark.parametrize("callback_kind", ["legacy", "opaque", "raises"])
def test_legacy_callback_is_called_once_and_body_errors_are_not_retried(page, callback_kind):
    asked = []

    def legacy(site, hint):
        asked.append((site, hint))
        return "246810"

    class OpaquePrompt:
        __signature__ = "not an inspectable signature"

        def __call__(self, site, hint):
            return legacy(site, hint)

    def raises(site, hint, *, origin=None):
        asked.append((site, hint))
        raise TypeError("synthetic prompt failure")

    unlock.set_code_prompt_callback({"legacy": legacy, "opaque": OpaquePrompt(), "raises": raises}[callback_kind])
    result = _enter_code()
    assert asked == [("login.example", "")]
    if callback_kind == "raises":
        assert "synthetic prompt failure" in result["error"]
        assert page.secret_calls == 0
    else:
        assert result["success"] is True, result
        assert page.value == "246810"


@pytest.mark.parametrize("next_url", ["http://login.example/verify", "https://login.example/new-document"])
def test_navigation_during_prompt_preserves_the_origin_and_nonce_guards(page, next_url):
    asked = []

    def prompt(site, hint, *, origin):
        asked.append(origin)
        page.navigate(next_url)
        return "246810"

    unlock.set_code_prompt_callback(prompt)
    result = _enter_code()
    assert asked == ["https://login.example"]
    assert result["success"] is False, result
    if next_url.startswith("http:"):
        assert result["error_type"] == "origin_changed"
    else:
        assert result["filled_fields"] == 0
    assert page.secret_calls == 1 and page.value == ""


def test_a_missing_page_origin_does_not_ask_the_surface_to_infer_one(page):
    page.navigate("about:blank")
    asked = []
    unlock.set_code_prompt_callback(lambda site, hint, **kwargs: asked.append(kwargs) or "246810")
    result = _enter_code()
    assert result["success"] is False, result
    assert asked == [] and page.secret_calls == 0
