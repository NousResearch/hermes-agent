"""Cross-profile / cross-session async-delegation delivery fences.

Desktop app-global remote mode hosts many profiles in one process and shares
one completion_queue. Completions must re-enter only the commissioning tab in
the commissioning profile — never an idle foreign tab (including other
profiles).
"""

from __future__ import annotations

import tui_gateway.server as server


def _sess(sid_key="k", profile="", home="", finalized=False, agent_session_id=None):
    agent = type("A", (), {"session_id": agent_session_id or sid_key})()
    return {
        "session_key": sid_key,
        "profile": profile,
        "profile_home": home,
        "_finalized": finalized,
        "agent": agent,
    }


def test_strict_origin_blocks_key_only_match():
    evt = {
        "type": "async_delegation",
        "origin_ui_session_id": "origin-tab",
        "session_key": "shared-looking-key",
    }
    foreign = _sess("shared-looking-key")
    assert server._session_owns_notification_event("foreign-tab", foreign, evt) is False
    assert server._session_owns_notification_event("origin-tab", foreign, evt) is True


def test_profile_fence_blocks_cross_profile_origin_match():
    evt = {
        "type": "async_delegation",
        "origin_ui_session_id": "tab1",
        "session_key": "sess1",
        "origin_profile": "highbeam",
        "origin_hermes_home": "/Users/x/.hermes/profiles/highbeam",
    }
    default_sess = _sess("sess1", profile="default", home="/Users/x/.hermes")
    # Even with matching origin_ui id, wrong profile must not own.
    assert server._session_owns_notification_event("tab1", default_sess, evt) is False


def test_same_profile_origin_owns():
    evt = {
        "type": "async_delegation",
        "origin_ui_session_id": "tab1",
        "session_key": "sess1",
        "origin_profile": "highbeam",
        "origin_hermes_home": "/Users/x/.hermes/profiles/highbeam",
    }
    hb = _sess("sess1", profile="highbeam", home="/Users/x/.hermes/profiles/highbeam")
    assert server._session_owns_notification_event("tab1", hb, evt) is True


def test_foreign_profile_is_elsewhere_when_owner_live(monkeypatch):
    evt = {
        "type": "async_delegation",
        "origin_ui_session_id": "hb-tab",
        "session_key": "hb-sess",
        "origin_profile": "highbeam",
        "origin_hermes_home": "/Users/x/.hermes/profiles/highbeam",
    }
    hb = _sess("hb-sess", profile="highbeam", home="/Users/x/.hermes/profiles/highbeam")
    other = _sess("other", profile="default", home="/Users/x/.hermes")
    monkeypatch.setattr(
        server,
        "_sessions",
        {"hb-tab": hb, "other-tab": other},
        raising=False,
    )
    # Patch lock to a simple no-op context manager
    class _L:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(server, "_sessions_lock", _L(), raising=False)
    assert server._notification_event_belongs_elsewhere("other-tab", other, evt) is True
    assert server._notification_event_belongs_elsewhere("hb-tab", hb, evt) is False


def test_legacy_unstamped_event_still_key_matches():
    evt = {
        "type": "async_delegation",
        "origin_ui_session_id": "",
        "session_key": "sess1",
    }
    sess = _sess("sess1")
    assert server._session_owns_notification_event("tabX", sess, evt) is True
