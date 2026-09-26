"""A voice turn whose visible text never entered the TTS pipeline is still spoken.

A turn that ends by iteration budget gets its final text from the toolless
summary call — it is returned unstreamed, so the streaming TTS queue only
ever receives the end sentinel. Without a whole-text fallback a voice user
sees the answer on screen and hears silence.
"""

import threading
import types

import tui_gateway.prompt_turn as prompt_turn
from tui_gateway import server
from tui_gateway.prompt_turn import _TurnRun

# prompt_turn bodies resolve their globals (e.g. _voice_tts_enabled) through the
# server namespace at runtime (method_ctx.bind_module rebinds them there); mirror
# that rebinding so the helper under test sees the same world as production.
prompt_turn.bind_module(vars(prompt_turn), server)
from tui_gateway import methods_voice as _mv  # noqa: E402

_mv.register(server)


def _turn_run(*, tts_queue=object(), tts_fed=False, result=None):
    agent = types.SimpleNamespace()
    st = _TurnRun(agent, None, None, receipt_committed=True)
    st.tts_queue = tts_queue
    st.tts_fed = tts_fed
    st.result = result if result is not None else {}
    return st


class TestUnstreamedReplyIsSpoken:
    def test_budget_summary_is_spoken_through_fallback(self, monkeypatch):
        spoken = []
        started = threading.Event()

        def fake_thread(target=None, args=(), daemon=None, **kw):
            spoken.append(args[0] if args else None)
            started.set()
            return types.SimpleNamespace()

        monkeypatch.setattr(threading, "Thread", fake_thread)
        monkeypatch.setattr(server, "_voice_tts_enabled", lambda: True)

        st = _turn_run(
            tts_queue=object(),
            tts_fed=False,
            result={
                "final_response": "partial answer composed after the budget ran out",
                "response_previewed": False,
            },
        )
        server._after_complete_turn(
            "sid",
            {"pending_title": None},
            st,
            "partial answer composed after the budget ran out",
        )

        assert started.wait(2)
        assert spoken == ["partial answer composed after the budget ran out"]

    def test_streamed_reply_is_not_respoken(self, monkeypatch):
        spoken = []
        started = threading.Event()

        def fake_thread(target=None, args=(), daemon=None, **kw):
            spoken.append(args[0] if args else None)
            started.set()
            return types.SimpleNamespace()

        monkeypatch.setattr(threading, "Thread", fake_thread)
        monkeypatch.setattr(server, "_voice_tts_enabled", lambda: True)

        st = _turn_run(tts_queue=object(), tts_fed=True, result={})
        server._after_complete_turn(
            "sid", {"pending_title": None}, st, "already spoken by the stream"
        )

        assert not started.is_set() or not spoken
        assert spoken == []

    def test_previewed_interim_text_is_not_respoken(self, monkeypatch):
        """Text already surfaced as interim (screen) but marked previewed —
        the turn's answer was shown; do not double-speak the sealed segment."""
        spoken = []

        def fake_thread(target=None, args=(), daemon=None, **kw):
            spoken.append(args[0] if args else None)
            return types.SimpleNamespace()

        monkeypatch.setattr(threading, "Thread", fake_thread)
        monkeypatch.setattr(server, "_voice_tts_enabled", lambda: True)

        st = _turn_run(
            tts_queue=object(), tts_fed=False, result={"response_previewed": True}
        )
        server._after_complete_turn(
            "sid", {"pending_title": None}, st, "shown as interim"
        )

        assert spoken == []
