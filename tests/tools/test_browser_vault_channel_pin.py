"""A vault operation keeps to the page channel its origin was checked on.

The page a vault script runs on is either a plugin page backend (Camofox, the Desktop preview) or the
task's CDP supervisor, chosen per script. A fill or ``enter_code`` checks the origin on one of them and
then runs for a while: ``enter_code`` waits for the user to type the code. If the plugin's page goes away
in that window, the next script would go over CDP to whatever tab the supervisor holds; if a plugin page
appears (or another plugin takes the task), a script would go to it and ignore the page that was
checked. Either way the secret must not be sent: the operation is refused with ``supervisor_required``.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from unittest.mock import patch

import pytest

from agent.vault_store import VaultStore

PASSWORD = "s3cret-pw"
CODE = "246810"
_INSPECTION = 'data-hermes-vault-slot", nonce'
_CONTROLS = {
    "fill": [{"autocomplete": "current-password", "formIndex": 0, "index": 1, "label": "", "name": "pw",
              "type": "password"}],
    "enter_code": [{"autocomplete": "one-time-code", "formIndex": 0, "index": 0, "label": "Code", "name": "otp",
                    "type": "text"}],
}


def _answer(expression: str, controls: list[dict]) -> str:
    if "location.href" in expression:
        return "https://example.com/login"
    if "unstamped" in expression:
        return json.dumps({"unstamped": True})
    if "reverted" in expression:
        return json.dumps({"landed": 1, "reverted": 0, "missing": []})
    if "filled" in expression:
        return json.dumps({"filled": 1})
    return json.dumps(controls)


def _carries_secret(expression: str) -> bool:
    return PASSWORD in expression or CODE in expression


class _Plugin:
    """A plugin page backend that records every script it is sent."""

    def __init__(self, name: str, channels: _Channels):
        self.name, self.channels, self.seen = name, channels, []

    def evaluate(self, task_id, expression, *, secret=False):
        self.seen.append(expression)
        return self.channels.after(expression)


class _Channels:
    """Two plugin pages and a CDP supervisor for one task (JS write path; trusted typing with
    ``typing``). ``current`` names the channel ``_page_backend`` reports: ``plugin``, ``plugin2`` or
    ``cdp`` (no plugin page). ``flip()`` moves it to ``flip_to``; with ``flip_on`` set it moves right
    after answering the first script that contains that text."""

    def __init__(self, controls: list[dict], *, current: str, flip_to: str = "", typing: bool = False):
        self.controls, self.current, self.sid = controls, current, "S1"
        self.flip_to = flip_to or ("cdp" if current != "cdp" else "plugin")
        self.flip_on = ""
        self.plugins = {"plugin": _Plugin("fakeplugin", self), "plugin2": _Plugin("otherplugin", self)}
        self.on_cdp: list[tuple[str, str | None]] = []
        self.typed: list[str] = []
        if typing:
            self.insert_text = self._insert_text

    def page(self):
        return self.plugins.get(self.current)

    def flip(self) -> None:
        self.current = self.flip_to

    def _insert_text(self, text, *, session_id=None):
        self.typed.append(text)
        return {"ok": True}

    # CDP supervisor
    def current_page_session(self):
        return self.sid

    def focus_page(self, origin, accept=None):
        return {"ok": True, "url": origin + "/login", "session_id": self.sid}

    def evaluate_runtime(self, expression, *, session_id=None, **kwargs):
        self.on_cdp.append((expression, session_id))
        if session_id is not None and session_id != self.sid:
            return {"ok": False, "error": "the page session changed since it was validated"}
        return {"ok": True, "result": self.after(expression)}

    def after(self, expression: str) -> str:
        out = _answer(expression, self.controls)
        if self.flip_on and self.flip_on in expression:
            self.flip_on = ""
            self.flip()
        return out

    def scripts_on(self, channel: str) -> list[str]:
        if channel == "cdp":
            return [e for e, _ in self.on_cdp]
        return list(self.plugins[channel].seen)

    def secret_scripts(self) -> list[str]:
        return [e for c in ("plugin", "plugin2", "cdp") for e in self.scripts_on(c) if _carries_secret(e)]


def _run(action: str, channels: _Channels, *, flip_at: str, tmp_path) -> dict:
    """Run ``action``; the channel moves at ``flip_at``: ``check`` (right after the origin check),
    ``prompt`` (enter_code: while the user reads the code; fill: once its fields were found),
    ``write`` (right after the fill's write, the first script carrying the password) or ``""`` (never)."""
    from tools import browser_vault_tool

    store = VaultStore(base_dir=tmp_path / "vault")
    check_name = "_bound_page_origin" if action == "fill" else "_current_page_check"
    real_check = getattr(browser_vault_tool, check_name)

    def check(*args, **kwargs):
        out = real_check(*args, **kwargs)
        if flip_at == "check":
            channels.flip()  # a plugin tab closed or opened between the origin check and the fill
        return out

    def prompt(site, hint):
        if flip_at == "prompt":
            channels.flip()  # the plugin tab closed, or one opened, while the user read the code
        return CODE

    with ExitStack() as stack:
        stack.enter_context(patch("agent.vault_store.get_vault_store", return_value=store))
        stack.enter_context(patch.object(browser_vault_tool, "_page_backend", side_effect=lambda t: channels.page()))
        stack.enter_context(patch.object(browser_vault_tool, "_ensure_supervisor", return_value=channels))
        stack.enter_context(patch.object(browser_vault_tool, "_bot_desktop_browser_session", return_value=False))
        stack.enter_context(patch.object(browser_vault_tool, check_name, side_effect=check))
        stack.enter_context(patch("agent.vault_backends.unlock.get_code_prompt_callback", return_value=prompt))
        stack.enter_context(patch("agent.vault_backends.unlock.can_prompt_here", return_value=True))
        if action == "enter_code":
            return json.loads(browser_vault_tool.browser_vault_enter_code(task_id="t"))
        meta = store.add_item(kind="login", label="Example", origin="https://example.com",
                              secret={"identifier_type": "email", "identifier": "u@example.com",
                                      "password": PASSWORD, "origin": "https://example.com"})
        channels.flip_on = {"prompt": _INSPECTION, "write": PASSWORD}.get(flip_at, "")
        return json.loads(browser_vault_tool.browser_vault_fill(meta.id, task_id="t", with_identifier=False))


_MOVES = [("plugin", "cdp"), ("cdp", "plugin"), ("plugin", "plugin2")]


@pytest.mark.parametrize("typing", [False, True], ids=["js", "typing"])
@pytest.mark.parametrize("flip_at", ["check", "prompt"])
@pytest.mark.parametrize("action", ["fill", "enter_code"])
@pytest.mark.parametrize("checked_on,moved_to", _MOVES, ids=["-".join(m) for m in _MOVES])
def test_a_channel_change_after_the_check_refuses_without_sending_the_secret(
        checked_on, moved_to, action, flip_at, typing, tmp_path):
    channels = _Channels(_CONTROLS[action], current=checked_on, flip_to=moved_to, typing=typing)
    out = _run(action, channels, flip_at=flip_at, tmp_path=tmp_path)
    assert out["success"] is False and out["error_type"] == "supervisor_required"
    assert channels.secret_scripts() == [] and channels.typed == []
    assert PASSWORD not in json.dumps(out) and CODE not in json.dumps(out)


@pytest.mark.parametrize("checked_on,moved_to", _MOVES, ids=["-".join(m) for m in _MOVES])
def test_no_script_after_the_write_follows_a_channel_change(checked_on, moved_to, tmp_path):
    """The password reached the checked page, then the channel changed. Neither the read-back nor
    the cleanup goes to the new channel, and the fill is not reported as landed."""
    channels = _Channels(_CONTROLS["fill"], current=checked_on, flip_to=moved_to)
    out = _run("fill", channels, flip_at="write", tmp_path=tmp_path)
    assert out["success"] is False
    assert channels.scripts_on(moved_to) == []
    assert [e for e in channels.scripts_on(checked_on) if _carries_secret(e)] != []


@pytest.mark.parametrize("action", ["fill", "enter_code"])
@pytest.mark.parametrize("checked_on", ["plugin", "cdp"])
def test_without_a_channel_change_the_secret_goes_to_the_checked_channel(action, checked_on, tmp_path):
    channels = _Channels(_CONTROLS[action], current=checked_on)
    out = _run(action, channels, flip_at="", tmp_path=tmp_path)
    assert out["success"] is True
    on_plugin = [e for e in channels.scripts_on("plugin") if _carries_secret(e)]
    on_cdp = [sid for e, sid in channels.on_cdp if _carries_secret(e)]
    assert (bool(on_plugin), bool(on_cdp)) == ((True, False) if checked_on == "plugin" else (False, True))
    assert set(on_cdp) <= {"S1"}


class _Reconnecting(_Channels):
    """A CDP supervisor with no session while it reconnects: it re-attaches to ``S2`` during the first
    script it is sent, and is unreachable once (the attach failed) when the operation starts."""

    def __init__(self, controls: list[dict]):
        super().__init__(controls, current="cdp")
        self.sid = None

    def evaluate_runtime(self, expression, *, session_id=None, **kwargs):
        if self.sid is None:
            self.sid = "S2"
        return super().evaluate_runtime(expression, session_id=session_id, **kwargs)


def test_an_origin_read_while_the_supervisor_has_no_session_is_no_check(tmp_path):
    """A CDP origin check that read no session cannot pin one: the code must not be sent with
    ``session_id=None``, which means whatever tab the supervisor holds when it is sent."""
    from tools import browser_vault_tool

    channels = _Reconnecting(_CONTROLS["enter_code"])
    reachable = iter([channels, channels, channels, None])

    with ExitStack() as stack:
        stack.enter_context(patch.object(browser_vault_tool, "_page_backend", return_value=None))
        stack.enter_context(patch.object(browser_vault_tool, "_ensure_supervisor",
                                         side_effect=lambda t: next(reachable, channels)))
        stack.enter_context(patch.object(browser_vault_tool, "_bot_desktop_browser_session", return_value=False))
        stack.enter_context(patch("agent.vault_backends.unlock.get_code_prompt_callback",
                                  return_value=lambda site, hint: CODE))
        stack.enter_context(patch("agent.vault_backends.unlock.can_prompt_here", return_value=True))
        out = json.loads(browser_vault_tool.browser_vault_enter_code(task_id="t"))

    assert out["success"] is False, out
    assert channels.secret_scripts() == []
    assert all(session is not None for _, session in channels.on_cdp[1:]), channels.on_cdp
