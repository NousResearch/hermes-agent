"""``inline_images=false`` history reads (#116511): the reachable ``[image]`` projection.

``_history_dict_text(content, *, image_urls)`` has always had both renderings; the API layer
hard-coded ``image_urls=True`` so a remote client re-transmitted every stored attachment
(26+ MiB for one measured conversation) on every ``session.resume``. Both history surfaces now
accept ``inline_images=false`` and route to the placeholder branch:

* ``session.resume`` — every path that carries messages (cold/lazy resume, live reattach);
* ``GET /api/sessions/{id}/messages`` — the REST pages the Desktop hydrates over.

Contract: the SAME stored row inlines its data URI by default and renders ``[image]``
when asked; non-image content is byte-identical either way.
"""

import pytest

import tui_gateway.server as srv
import tui_gateway.methods_session  # noqa: F401  (registers the RPC methods)

DATA_URI = "data:image/png;base64," + "a" * 128
IMAGE_TURN = [
    {"role": "user", "content": [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": DATA_URI}},
    ]},
    {"role": "assistant", "content": "a chart"},
]


def test_coerce_message_text_switches_image_rendering():
    """The projection seam: default inlines the URI; ``image_urls=False`` renders ``[image]``."""
    coerce, history_to_messages = srv._coerce_message_text, srv._history_to_messages  # server.py-bound
    content = IMAGE_TURN[0]["content"]
    assert DATA_URI in coerce(content)
    assert DATA_URI not in coerce(content, image_urls=False)
    assert "[image]" in coerce(content, image_urls=False)
    # Non-image content is identical either way.
    assert coerce("plain") == coerce("plain", image_urls=False) == "plain"


def test_history_to_messages_honors_the_switch():
    history_to_messages = srv._history_to_messages  # server.py-bound
    default = history_to_messages(IMAGE_TURN)
    assert DATA_URI in default[0]["text"]

    inlined_off = history_to_messages(IMAGE_TURN, image_urls=False)
    assert "[image]" in inlined_off[0]["text"]
    assert DATA_URI not in inlined_off[0]["text"]
    assert inlined_off[1] == default[1] == {"role": "assistant", "text": "a chart"}


def test_resume_cold_carries_the_switch(tmp_path, monkeypatch):
    """A cold ``session.resume`` with ``inline_images=false`` returns the placeholder projection."""
    from hermes_state import SessionDB

    home = tmp_path / ".hermes"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", home / "state.db")
    db = SessionDB(home / "state.db")
    db.create_session("img-chat", source="desktop")
    db.append_messages_batch("img-chat", [{"role": "user", "content": IMAGE_TURN[0]["content"]}])
    try:
        monkeypatch.setattr(srv, "_get_db", lambda: db)
        monkeypatch.setattr(srv, "_enable_gateway_prompts", lambda: None)
        monkeypatch.setattr(srv, "_schedule_agent_build", lambda *a, **k: None)
        monkeypatch.setattr(srv, "_schedule_session_cap_enforcement", lambda *a, **k: None)
        monkeypatch.setattr(srv, "_maybe_schedule_auto_continue", lambda *a, **k: None)
        monkeypatch.setattr(srv, "_stored_session_runtime_overrides", lambda _found: {})
        known = set(srv._sessions)
        try:
            resp = srv._methods["session.resume"](1, {"session_id": "img-chat"})
            assert "error" not in resp, resp
            assert DATA_URI in resp["result"]["messages"][0]["text"]

            resp = srv._methods["session.resume"](1, {"session_id": "img-chat", "inline_images": False})
            assert "error" not in resp, resp
            assert "[image]" in resp["result"]["messages"][0]["text"]
            assert DATA_URI not in resp["result"]["messages"][0]["text"]
        finally:
            for sid in [s for s in srv._sessions if s not in known]:
                srv._sessions.pop(sid, None)
    finally:
        db.close()


@pytest.mark.parametrize("inline_images", [True, False])
def test_resume_live_reattach_carries_the_switch(tmp_path, monkeypatch, inline_images):
    """The live-session fast path projects through ``_live_session_payload`` too."""
    from hermes_state import SessionDB

    home = tmp_path / ".hermes"
    home.mkdir(parents=True)
    db = SessionDB(home / "state.db")
    record = {
        # In-memory history (no DB rows for the key): the payload projects from session["history"].
        "history": list(IMAGE_TURN), "display_history_prefix": [], "last_active": 0.0, "running": False,
        "session_key": "live-img", "source": "desktop",
    }
    monkeypatch.setattr(srv, "_get_db", lambda: db)
    monkeypatch.setattr(srv, "_find_live_session_by_key", lambda key, *_a: ("live-sid", record))
    monkeypatch.setattr(srv, "_profile_home", lambda _p: None)
    monkeypatch.setattr(srv, "_reattach_refusal", lambda *_a: None)
    monkeypatch.setattr(srv, "_cancel_ws_orphan_reap", lambda *_a: None)
    monkeypatch.setattr(srv, "_child_run_active", lambda *_a: False)
    known = set(srv._sessions)
    srv._sessions["live-sid"] = record
    try:
        resp = srv._methods["session.resume"](
            1, {"session_id": "live-img", "inline_images": inline_images})
        assert "error" not in resp, resp
        text = resp["result"]["messages"][0]["text"]
        assert (DATA_URI in text) is inline_images
        assert ("[image]" in text) is not inline_images
    finally:
        srv._sessions.pop("live-sid", None)
        for sid in [s for s in srv._sessions if s not in known]:
            srv._sessions.pop(sid, None)
        db.close()
