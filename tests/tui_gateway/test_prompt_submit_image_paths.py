"""``prompt.submit.image_paths`` binds images to that submit, never through the shared staging slot.

With the two-step ``image.attach`` → ``prompt.submit`` flow the images live in the session's
``attached_images`` slot until the next turn consumes them, so two sends racing (a second client, or a
retried attach whose receipt was lost) attach an image to the wrong prompt. An explicit list must reach its
own turn and leave the slot alone, and the path check must not run for a caller the session refuses.
"""

import threading
import types

from tui_gateway import server


def _session(**overrides):
    return {
        "agent": types.SimpleNamespace(), "session_key": "image-paths-session", "history": [],
        "history_lock": threading.Lock(), "history_version": 0, "running": True, "transport": None,
        "attached_images": [], **overrides,
    }


def _submit(sid, session, rid, **params):
    server._sessions[sid] = session
    try:
        return server.handle_request({"id": rid, "method": "prompt.submit",
                                      "params": {"session_id": sid, **params}})
    finally:
        server._sessions.pop(sid, None)


def test_busy_submit_queues_its_own_images_and_leaves_the_staging_slot(tmp_path, monkeypatch):
    staged, own = tmp_path / "staged.png", tmp_path / "own.png"
    staged.write_bytes(b"\x89PNG\r\n\x1a\n staged")
    own.write_bytes(b"\x89PNG\r\n\x1a\n own")
    session = _session(attached_images=[str(staged)])
    sid = "image-paths-busy"

    queued = _submit(sid, session, "r1", text="Look at this", queued=True, image_paths=[str(own)])
    assert queued["result"]["status"] == "queued"
    assert session["queued_prompt"]["image_paths"] == [str(own)]
    assert session["attached_images"] == [str(staged)]  # another client's staged image stays for its turn

    missing = _submit(sid, session, "r2", text="And this", queued=True, image_paths=[str(tmp_path / "gone.png")])
    assert missing["error"]["code"] == 4016
    assert session["attached_images"] == [str(staged)]
    assert "queued_prompts" not in session  # a rejected submit queues nothing

    # A caller the session refuses gets that refusal, not a host-path existence answer.
    monkeypatch.setattr(server, "_legacy_group_fence_error", lambda rid, *_a: server._err(rid, 4122, "denied"))
    denied = _submit(sid, session, "r3", text="probe", image_paths=[str(tmp_path / "gone.png")])
    assert denied["error"]["code"] == 4122


def test_idle_submit_hands_only_its_own_images_to_the_turn(tmp_path, monkeypatch):
    staged, own = tmp_path / "staged.png", tmp_path / "own.png"
    staged.write_bytes(b"\x89PNG\r\n\x1a\n staged")
    own.write_bytes(b"\x89PNG\r\n\x1a\n own")
    session = _session(running=False, attached_images=[str(staged)], agent=None, agent_ready=threading.Event())
    frames = []

    class Supervisor:
        def submit_turn(self, frame, *, on_complete=None):
            frames.append(frame)
            return frame["request_id"]

    monkeypatch.setattr(server, "_load_cfg", lambda: {"dashboard": {"turn_isolation": True}})
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg=None: Supervisor())

    response = _submit("image-paths-idle", session, "r4", text="Look at this", image_paths=[str(own)])

    assert response["result"]["status"] == "streaming"
    assert frames[0]["attached_images"] == [str(own)]
    assert session["attached_images"] == [str(staged)]
