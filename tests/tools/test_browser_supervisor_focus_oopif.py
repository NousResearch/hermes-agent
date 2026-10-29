"""focus_page: a login/OTP form inside a cross-origin iframe (OOPIF) is its own CDP target.

Regression: a step-up prompt rendered by an embedded identity provider lives in an OOPIF; focus_page only
looked at page targets, so enter_code reported no_code_field while the code boxes sat on screen.
"""
import asyncio
import threading

from tools.browser_supervisor import CDPSupervisor

PROBE = "HAS_FORM"
TARGETS = [
    {"targetId": "P1", "type": "page", "url": "https://app.example.com/settings/members"},
    {"targetId": "F1", "type": "iframe", "url": "https://login.idp.example/v1/bundle/web/x"},
    {"targetId": "F2", "type": "iframe", "url": "https://captcha.example.net/anchor"},
]


def _supervisor(has_form: set[str]):
    sup = CDPSupervisor.__new__(CDPSupervisor)
    sup._state_lock = threading.Lock()
    sup._page_session_id = "orig"
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    sup._loop = loop
    attached = []

    async def cdp(method, params=None, *, session_id=None, timeout=10.0):
        if method == "Target.getTargets":
            return {"result": {"targetInfos": TARGETS}}
        if method == "Target.attachToTarget":
            attached.append(params["targetId"])
            return {"result": {"sessionId": "s-" + params["targetId"]}}
        if method == "Runtime.evaluate":
            return {"result": {"result": {"value": session_id[2:] in has_form}}}
        return {"result": {}}

    async def noop(*a, **k):
        return None

    sup._cdp = cdp
    sup._enable_page_domains = noop
    sup._install_dialog_bridge = noop
    return sup, attached, loop


def _focus(has_form: set[str], origin: str, accept=PROBE):
    sup, attached, loop = _supervisor(has_form)
    try:
        out = sup.focus_page(origin, accept=accept)
    finally:
        loop.call_soon_threadsafe(loop.stop)
    return out, sup, attached


def test_focus_lands_in_the_oopif_that_holds_the_form():
    out, sup, attached = _focus({"F1"}, "")
    assert out == {"ok": True, "url": TARGETS[1]["url"]}
    assert sup._page_session_id == "s-F1"
    assert attached[0] == "P1"  # pages are always tried before frames


def test_a_page_with_the_form_still_wins_over_frames():
    out, _, attached = _focus({"P1", "F1"}, "")
    assert out["url"] == TARGETS[0]["url"] and attached == ["P1"]


def test_frames_stay_origin_bound():
    out, sup, attached = _focus({"F1", "F2"}, "https://captcha.example.net")
    assert out["url"] == TARGETS[2]["url"] and attached == ["F2"] and sup._page_session_id == "s-F2"


def test_frames_are_never_picked_without_an_accept_probe():
    out, sup, attached = _focus(set(), "https://login.idp.example", accept=None)
    assert out["ok"] is False and attached == [] and sup._page_session_id == "orig"
