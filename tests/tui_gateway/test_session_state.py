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
            "explicit_cwd": True,
            "attached_images": ["/tmp/a.png"],
            "image_counter": 2,
            "edit_snapshots": {"f.py": "snap"},
            "pending_title": "my title",
            "show_reasoning": True,
            "tool_progress_mode": "verbose",
            "tool_started_at": {"tu_1": 1.0},
            "personality": "pirate",
            "model_override": {"model": "m1"},
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
    assert s.explicit_cwd is True
    assert s.attached_images == ["/tmp/a.png"]
    assert s.image_counter == 2
    assert s.edit_snapshots == {"f.py": "snap"}
    assert s.pending_title == "my title"
    assert s.show_reasoning is True
    assert s.tool_progress_mode == "verbose"
    assert s.tool_started_at == {"tu_1": 1.0}
    assert s.personality == "pirate"
    assert s.model_override == {"model": "m1"}


def test_setters_write_through_to_items():
    s = SessionState({"running": False, "history": [], "history_version": 0})
    s.running = True
    s.history = [1, 2]
    s.history_version = 5
    assert s["running"] is True
    assert s["history"] == [1, 2]
    assert s["history_version"] == 5
    s.explicit_cwd = True
    s.attached_images = ["/tmp/b.png"]
    s.image_counter = 7
    s.edit_snapshots = {"g.py": "snap2"}
    s.pending_title = None
    s.show_reasoning = False
    s.tool_progress_mode = "concise"
    s.tool_started_at = {}
    s.personality = "butler"
    s.model_override = {"model": "m2", "provider": "p2"}
    assert s["explicit_cwd"] is True
    assert s["attached_images"] == ["/tmp/b.png"]
    assert s["image_counter"] == 7
    assert s["edit_snapshots"] == {"g.py": "snap2"}
    assert s["pending_title"] is None
    assert s["show_reasoning"] is False
    assert s["tool_progress_mode"] == "concise"
    assert s["tool_started_at"] == {}
    assert s["personality"] == "butler"
    assert s["model_override"] == {"model": "m2", "provider": "p2"}


# ── lock dances (Phase 3 Step 2) ────────────────────────────────────────


def _locked_session(**extra):
    return SessionState(
        {
            "history": [],
            "history_lock": threading.Lock(),
            "history_version": 0,
            "running": False,
            "inflight_turn": None,
            "last_active": 0.0,
            **extra,
        }
    )


def test_snapshot_history_returns_shallow_copy_and_version():
    msgs = [{"role": "user", "content": "hi"}]
    s = _locked_session(history=msgs, history_version=7)
    snap, version = s.snapshot_history()
    assert snap == msgs
    assert snap is not msgs  # a copy — mutating it must not touch the session
    snap.append({"role": "assistant", "content": "x"})
    assert s["history"] == [{"role": "user", "content": "hi"}]
    assert version == 7
    # mirrors the server's `.get()` defaults for absent keys
    bare = SessionState({"history_lock": threading.Lock()})
    assert bare.snapshot_history() == ([], 0)


def test_commit_compaction_rejects_on_version_mismatch():
    s = _locked_session(history=[1, 2, 3, 4], history_version=3)
    snap, version = s.snapshot_history()
    # someone mutated mid-compress
    s["history_version"] = 4
    assert s.commit_compaction([1], expected_version=version) is False
    assert s["history"] == [1, 2, 3, 4]  # compressed result dropped
    assert s["history_version"] == 4  # version untouched by the reject


def test_commit_compaction_commits_on_match_and_bumps_version():
    s = _locked_session(history=[1, 2, 3, 4], history_version=3)
    snap, version = s.snapshot_history()
    assert s.commit_compaction([1], expected_version=version) is True
    assert s["history"] == [1]
    assert s["history_version"] == 4  # bumped past the snapshot


def test_begin_turn_claims_idle_session_and_touches_last_active():
    s = _locked_session()
    assert s.begin_turn() is True
    assert s["running"] is True
    assert s["last_active"] > 0.0


def test_begin_turn_returns_false_when_running():
    s = _locked_session(running=True, last_active=1.0)
    assert s.begin_turn() is False
    assert s["running"] is True
    assert s["last_active"] == 1.0  # busy reject does not touch last_active


def test_end_turn_clears_running_and_inflight():
    s = _locked_session(running=True, inflight_turn={"text": "hi"}, last_active=1.0)
    s.end_turn()
    assert s["running"] is False
    assert s["inflight_turn"] is None
    assert s["last_active"] > 1.0


def test_build_once_returns_true_exactly_once():
    s = SessionState({})
    assert s.build_once() is True
    assert s["agent_build_started"] is True
    assert s.build_once() is False
    assert s.build_once() is False
    # the lazily-created lock is reused, not re-made
    assert isinstance(s["agent_build_lock"], type(threading.Lock()))


def test_build_once_returns_false_when_agent_already_ready():
    ready = threading.Event()
    ready.set()
    s = SessionState({"agent_ready": ready})
    # agent already built (ready set) — nobody should build again
    assert s.build_once() is False
    assert "agent_build_started" not in s
