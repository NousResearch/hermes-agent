"""Voice completion must not confuse a started queue with delivered text (#118589)."""
import queue
import threading

import pytest

from tui_gateway import server


@pytest.mark.parametrize(
    "deltas, final, outcome, pipeline, enabled, expected, spoken",
    [
        ([], "Budget reply", "budget", True, True, ["Budget reply", None], []),
        (["Normal reply"], "Normal reply", "complete", True, True, ["Normal reply", None], []),
        (["", " \n"], "Budget reply", "budget", True, True, ["", " \n", "Budget reply", None], []),
        ([], "", "budget", True, True, [None], []),
        ([], "Budget reply", "budget", True, False, [None], []),
        ([], "Interrupted", "interrupted", True, True, [None], []),
        ([], "", "exception", True, True, [None], []),
        ([], "Budget reply", "budget", False, True, [], ["Budget reply"]),
        ([], "Budget reply", "budget", False, False, [], []),
    ],
)
def test_turn_speaks_unstreamed_final_without_replaying_stream(
    monkeypatch, deltas, final, outcome, pipeline, enabled, expected, spoken
):
    audio = queue.Queue()
    done = threading.Event()
    spoken_done = threading.Event()
    events, whole_text = [], []

    class Agent:
        def run_conversation(self, prompt, stream_callback=None, **kwargs):
            for delta in deltas:
                stream_callback(delta)
            if outcome == "exception":
                raise RuntimeError("fixture provider failure")
            return {
                "final_response": final, "messages": [], "api_calls": 4,
                "completed": outcome == "complete",
                "interrupted": outcome == "interrupted",
                "reason": "max_iterations_reached" if outcome == "budget" else outcome,
            }

    session = dict(agent=Agent(), session_key="voice-budget-fixture", history=[],
                   history_lock=threading.Lock(), history_version=0, running=False,
                   attached_images=[], image_counter=0, cols=80, slash_worker=None,
                   show_reasoning=False, tool_progress_mode="all")
    monkeypatch.setitem(server._sessions, "voice-budget-fixture", session)
    monkeypatch.setattr(server, "_start_turn_voice", lambda: (audio if pipeline else None, False))
    monkeypatch.setattr(server, "_voice_tts_enabled", lambda: enabled)
    monkeypatch.setattr(server, "make_stream_renderer", lambda cols: None)
    monkeypatch.setattr(server, "render_message", lambda raw, cols: None)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    # Avoid unrelated goal/control/trim work; retain actual turn and voice lifecycle.
    monkeypatch.setattr(server, "_goal_followup_after_turn", lambda *a: None)
    monkeypatch.setattr(server, "_publish_session_control_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(server, "_emit_settled_session_info", lambda *a: None)
    monkeypatch.setattr(server, "_sessions_quiescent", lambda **k: False)
    monkeypatch.setattr(server, "_run_post_turn_followups", lambda *a, **k: done.set())
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: events.append((event, payload)))

    def speak(text):
        whole_text.append(text)
        spoken_done.set()

    monkeypatch.setattr(server, "_speak_text_with_barge", speak)
    assert server._run_prompt_submit("fixture", "voice-budget-fixture", session, "Inspect the file")
    assert done.wait(5), events
    worker = session.get("turn_thread")
    if worker is not None:
        worker.join(timeout=5)
    complete = [p for e, p in events if e == "message.complete"]
    assert complete, events
    if outcome not in {"exception", "interrupted"}:
        assert complete[-1]["text"] == final
    sent = []
    while not audio.empty():
        sent.append(audio.get_nowait())
    assert sent == expected
    if spoken:
        assert spoken_done.wait(5)
    assert whole_text == spoken
