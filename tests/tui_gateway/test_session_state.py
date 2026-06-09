"""SessionState (Phase 3 Step 1) — it must behave EXACTLY like a dict so the
gateway's ~58 session call-sites are unaffected, while the typed accessors stay
in sync with the underlying items."""

import threading

from tui_gateway.session_state import SessionState


def test_is_a_dict_with_exact_dict_semantics():
    s = SessionState({"history": [1], "running": False})
    assert isinstance(s, dict)
    # membership / get / setdefault / pop behave as a plain dict
    assert "history" in s
    assert "missing" not in s
    assert s.get("missing") is None
    assert s.get("missing", "fallback") == "fallback"
    assert s.setdefault("running", True) is False  # existing key untouched
    assert s.setdefault("cols", 80) == 80  # new key inserted
    assert s.pop("cols") == 80
    assert s.pop("nope", "dflt") == "dflt"
    assert dict(s) == {"history": [1], "running": False}


def test_absent_keys_stay_absent():
    # The whole safety argument: a key that was never set is genuinely absent,
    # so `.get()` returns the consumer's default (not a pre-populated sentinel).
    s = SessionState({"history": []})
    assert "agent_ready" not in s
    assert s.get("agent_ready") is None
    assert "personality" not in s
    assert s.get("personality", "cfg-default") == "cfg-default"


def test_accessors_mirror_the_underlying_items():
    lock = threading.Lock()
    s = SessionState(
        {
            "history": [{"role": "user"}],
            "history_lock": lock,
            "history_version": 3,
            "running": False,
            "inflight_turn": None,
            "agent": "AGENT",
            "session_key": "k1",
            "slash_worker": None,
            "transport": "T",
            "cwd": "/tmp",
            "cols": 100,
            "last_active": 1.0,
        }
    )
    assert s.history == s["history"]
    assert s.history_lock is lock
    assert s.history_version == 3
    assert s.running is False
    assert s.agent == "AGENT"
    assert s.session_key == "k1"
    assert s.cwd == "/tmp"
    assert s.cols == 100


def test_setters_write_through_to_items():
    s = SessionState({"running": False, "history": [], "history_version": 0})
    s.running = True
    s.history = [1, 2]
    s.history_version = 5
    assert s["running"] is True
    assert s["history"] == [1, 2]
    assert s["history_version"] == 5
