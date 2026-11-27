"""``prompt.submit`` rejects a blank turn before it costs anything.

A whitespace-only submit is never a real turn, but on the pre-fix path it ran
one: the deferred build constructs a full agent and sends the entire
conversation to the provider to answer nothing, then persists an empty user
row. A reconnect-looping client that fires ``prompt.submit(text="")`` every
cycle therefore burns one full-context API call per cycle.

The one blank submit that IS legitimate is an image-only send — the turn's
content is the attachment — so that path must keep working.
"""

import threading
import types

import pytest

from tui_gateway import server


@pytest.fixture(autouse=True)
def _no_prewarm_timer(monkeypatch):
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a, **k: None)


@pytest.fixture
def submit_probe(monkeypatch):
    """A live session plus counters for every side effect a real turn causes."""
    sid = "blank-submit-sid"
    ready = threading.Event()
    ready.set()
    session = {
        "agent": types.SimpleNamespace(),
        "agent_ready": ready,
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
    }
    server._sessions[sid] = session

    effects = {"slots": 0, "agent_builds": 0, "db_rows": 0, "turns": []}

    def claim_slot(*_a, **_k):
        effects["slots"] += 1

    monkeypatch.setattr(server, "_ensure_active_session_slot", claim_slot)
    monkeypatch.setattr(
        server,
        "_start_agent_build",
        lambda *_a, **_k: effects.__setitem__("agent_builds", effects["agent_builds"] + 1),
    )
    monkeypatch.setattr(
        server,
        "_ensure_session_db_row",
        lambda *_a, **_k: effects.__setitem__("db_rows", effects["db_rows"] + 1),
    )
    monkeypatch.setattr(server, "_persist_branch_seed", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_wait_agent_for_prompt", lambda *_a, **_k: None)
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda _rid, _sid, _session, text, **_kw: effects["turns"].append(text),
    )

    def submit(text, **params):
        response = server.handle_request(
            {
                "id": "req",
                "method": "prompt.submit",
                "params": {"session_id": sid, "text": text, **params},
            }
        )
        # The accepted path runs the turn on a background thread.
        thread = session.get("_run_thread")
        if thread is not None:
            thread.join(timeout=2)
            assert not thread.is_alive()
        return response

    try:
        yield types.SimpleNamespace(sid=sid, session=session, submit=submit, effects=effects)
    finally:
        server._sessions.pop(sid, None)


@pytest.mark.parametrize(
    "text",
    ["", " ", "   \n\t  ", "\n", "\u00a0"],
    ids=["empty", "one-space", "mixed-whitespace", "newline", "nbsp"],
)
def test_a_blank_submit_is_rejected(submit_probe, text):
    response = submit_probe.submit(text)

    assert "result" not in response
    assert response["error"]["code"] == 4033
    assert submit_probe.effects == {"slots": 0, "agent_builds": 0, "db_rows": 0, "turns": []}


def test_a_blank_submit_costs_no_agent_build_no_db_row_and_no_api_call(submit_probe):
    """The whole point: rejection happens before anything expensive."""
    submit_probe.submit("   \n  ")

    assert submit_probe.effects == {"slots": 0, "agent_builds": 0, "db_rows": 0, "turns": []}


def test_a_blank_submit_leaves_the_session_idle(submit_probe):
    """A rejected submit must not latch ``running`` and wedge the session."""
    submit_probe.submit("")

    assert submit_probe.session["running"] is False
    assert submit_probe.session["history"] == []
    assert submit_probe.session["history_version"] == 0


def test_an_image_only_submit_still_runs(submit_probe):
    """Blank text with an attachment is a real turn — the image is the content."""
    submit_probe.session["attached_images"] = ["/tmp/screenshot.png"]

    response = submit_probe.submit("")

    assert response["result"] == {"status": "streaming"}
    assert submit_probe.effects["turns"] == [""]
    assert submit_probe.effects["agent_builds"] == 1
    assert submit_probe.effects["slots"] == 1
    assert submit_probe.effects["db_rows"] == 1


def test_an_ordinary_submit_is_unaffected(submit_probe):
    response = submit_probe.submit("what is the weather")

    assert response["result"] == {"status": "streaming"}
    assert submit_probe.effects["turns"] == ["what is the weather"]
    assert submit_probe.effects["slots"] == 1
    assert submit_probe.effects["agent_builds"] == 1
    assert submit_probe.effects["db_rows"] == 1


def test_text_that_only_looks_blank_after_sanitizing_is_still_rejected(submit_probe):
    """The guard runs on the SANITIZED text, not the raw parameter.

    ``sanitize_user_prompt_text`` strips leaked bracketed-paste wrappers. A
    submit carrying nothing but a wrapper reduces to empty and must be rejected
    on the same terms as a literal empty string — otherwise the whole class
    reopens through a client that leaks escape sequences.
    """
    from hermes_cli.input_sanitize import sanitize_user_prompt_text

    raw = "\x1b[200~\x1b[201~"
    assert not sanitize_user_prompt_text(raw).strip(), "fixture no longer sanitizes to empty"

    response = submit_probe.submit(raw)

    assert response["error"]["code"] == 4033
    assert submit_probe.effects == {"slots": 0, "agent_builds": 0, "db_rows": 0, "turns": []}


def test_the_rejection_code_is_not_reused_by_another_prompt_submit_error(submit_probe):
    """4033 must be distinguishable from the other prompt.submit rejections.

    A client needs to tell "you sent nothing" apart from "that truncation would
    erase the transcript" (4028), "that truncation needs confirming" (4029) and "target message is gone" (4018) to show
    the right message.
    """
    blank = submit_probe.submit("")
    bad_ordinal = server.handle_request(
        {
            "id": "req",
            "method": "prompt.submit",
            "params": {
                "session_id": submit_probe.sid,
                "text": "real text",
                "truncate_before_user_ordinal": -1,
            },
        }
    )

    assert blank["error"]["code"] == 4033
    assert bad_ordinal["error"]["code"] != 4033


@pytest.mark.parametrize("text", ["", " \n\t", "\x1b[200~\x1b[201~"])
@pytest.mark.parametrize("fence,code", [("author", 4124), ("hosted", 4120), ("group", 4122)])
def test_blank_submit_preserves_authorization_precedence(submit_probe, monkeypatch, text, fence, code):
    params = {}
    if fence == "author":
        params["_turn_author"] = {"id": "forged"}
        params["_hosted_task"] = {}
    elif fence == "hosted":
        params["_hosted_task"] = {}
    else:
        from gateway import hosted_rooms

        submit_probe.session["title"] = "Group: hosted-room"
        monkeypatch.setattr(hosted_rooms, "probe_hosted_room", lambda *_a, **_k: True)

    response = submit_probe.submit(text, **params)

    assert response["error"]["code"] == code
    assert submit_probe.effects == {"slots": 0, "agent_builds": 0, "db_rows": 0, "turns": []}
    assert submit_probe.session["running"] is False
