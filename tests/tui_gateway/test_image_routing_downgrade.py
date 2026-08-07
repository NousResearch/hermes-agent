"""The turn-time image mode decision must report downgrades, not hide them.

A user who forces ``agent.image_input_mode: native`` on a Codex app-server
session (or hits a decision failure) previously got a text description with
zero UI signal that the setting was overridden — the only evidence was a
stderr line in the gateway log (#66829).
"""

from types import SimpleNamespace

import pytest

import agent.image_routing as image_routing
import tui_gateway.server as server
from tui_gateway.server import _build_run_message, _decide_turn_image_mode


def _agent(api_mode: str = "chat_completions") -> SimpleNamespace:
    return SimpleNamespace(provider="openai", model="gpt-5.5", api_mode=api_mode)


class TestDecideTurnImageMode:
    def test_native_decision_is_honored_with_no_reason(self, monkeypatch):
        monkeypatch.setattr(
            image_routing, "decide_image_input_mode", lambda p, m, c, **kw: "native"
        )
        mode, reason = _decide_turn_image_mode(_agent())
        assert mode == "native"
        assert reason is None

    def test_codex_app_server_downgrades_native_with_reason(self, monkeypatch):
        monkeypatch.setattr(
            image_routing, "decide_image_input_mode", lambda p, m, c, **kw: "native"
        )
        mode, reason = _decide_turn_image_mode(_agent(api_mode="codex_app_server"))
        assert mode == "text"
        assert reason is not None and "Codex app-server" in reason

    def test_codex_app_server_text_decision_has_no_spurious_reason(self, monkeypatch):
        # When the decision is already text, nothing was overridden — the
        # status line must not cry wolf.
        monkeypatch.setattr(
            image_routing, "decide_image_input_mode", lambda p, m, c, **kw: "text"
        )
        mode, reason = _decide_turn_image_mode(_agent(api_mode="codex_app_server"))
        assert mode == "text"
        assert reason is None

    def test_decision_failure_falls_back_to_text_with_reason(self, monkeypatch):
        def _boom(p, m, c, **kw):
            raise RuntimeError("metadata backend unreachable")

        monkeypatch.setattr(image_routing, "decide_image_input_mode", _boom)
        mode, reason = _decide_turn_image_mode(_agent())
        assert mode == "text"
        assert reason is not None
        assert "RuntimeError" in reason and "metadata backend unreachable" in reason

    def test_trace_line_names_provider_model_and_both_modes(self, monkeypatch, capsys):
        monkeypatch.setattr(
            image_routing, "decide_image_input_mode", lambda p, m, c, **kw: "native"
        )
        _decide_turn_image_mode(_agent(api_mode="codex_app_server"))
        err = capsys.readouterr().err
        assert "[tui_gateway] image_routing:" in err
        assert "decided=native" in err
        assert "final=text" in err


@pytest.fixture
def emitted(monkeypatch):
    """Capture status.update payloads instead of pushing them to a client."""
    seen: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(
        server, "_emit", lambda event, sid, payload=None: seen.append((event, sid, payload or {}))
    )
    monkeypatch.setattr(
        server, "_enrich_with_attached_images", lambda text, paths: f"{text} <described>"
    )
    return seen


def _notices(emitted):
    return [
        p["text"]
        for event, _sid, p in emitted
        if event == "status.update" and p.get("kind") == "image_routing"
    ]


class TestBuildRunMessageEmitsNotices:
    """Every path that silently degrades an image must announce itself.

    The decision helper is unit-tested above; these drive the emission so a
    regression in the wiring (wrong kind, missing branch, swallowed notice)
    fails here rather than shipping a silent downgrade to the user (#66829).
    """

    IMAGES = ["/tmp/shot.png"]

    def test_no_images_emits_nothing_and_passes_prompt_through(self, emitted):
        assert _build_run_message("s1", _agent(), "hello", []) == "hello"
        assert _notices(emitted) == []

    def test_downgrade_reason_is_announced(self, monkeypatch, emitted):
        monkeypatch.setattr(
            image_routing, "decide_image_input_mode", lambda p, m, c, **kw: "native"
        )
        out = _build_run_message("s1", _agent(api_mode="codex_app_server"), "hi", self.IMAGES)
        assert out == "hi <described>"
        assert len(_notices(emitted)) == 1
        assert "Codex app-server" in _notices(emitted)[0]

    def test_unreadable_image_data_is_announced(self, monkeypatch, emitted):
        monkeypatch.setattr(
            image_routing, "decide_image_input_mode", lambda p, m, c, **kw: "native"
        )
        monkeypatch.setattr(
            image_routing, "build_native_content_parts", lambda t, p: ([{"type": "text"}], p)
        )
        out = _build_run_message("s1", _agent(), "hi", self.IMAGES)
        assert out == "hi <described>"
        assert _notices(emitted) == [
            "⚠ Image sent as a text description — no readable image data for native attachment."
        ]

    def test_native_attach_failure_is_announced_with_type(self, monkeypatch, emitted):
        monkeypatch.setattr(
            image_routing, "decide_image_input_mode", lambda p, m, c, **kw: "native"
        )

        def _boom(text, paths):
            raise OSError("disk gone")

        monkeypatch.setattr(image_routing, "build_native_content_parts", _boom)
        out = _build_run_message("s1", _agent(), "hi", self.IMAGES)
        assert out == "hi <described>"
        assert _notices(emitted) == [
            "⚠ Image sent as a text description — native attachment failed (OSError)."
        ]

    def test_successful_native_attach_stays_silent(self, monkeypatch, emitted):
        parts = [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": "x"}}]
        monkeypatch.setattr(
            image_routing, "decide_image_input_mode", lambda p, m, c, **kw: "native"
        )
        monkeypatch.setattr(
            image_routing, "build_native_content_parts", lambda t, p: (parts, [])
        )
        assert _build_run_message("s1", _agent(), "hi", self.IMAGES) == parts
        assert _notices(emitted) == []

    def test_notice_kind_is_not_process(self, monkeypatch, emitted):
        """kind='process' is dropped by the desktop — that was the bug."""
        monkeypatch.setattr(
            image_routing, "decide_image_input_mode", lambda p, m, c, **kw: "native"
        )
        _build_run_message("s1", _agent(api_mode="codex_app_server"), "hi", self.IMAGES)
        kinds = {p.get("kind") for event, _s, p in emitted if event == "status.update"}
        assert kinds == {"image_routing"}
