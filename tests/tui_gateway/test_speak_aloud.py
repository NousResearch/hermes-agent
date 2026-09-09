"""Apple-native read-aloud for fullscreen TUI mode (`speak.*` RPC).

macOS Speak Selection (Option+Esc) reads AXSelectedText, which terminal
alternate-screen apps never expose — so the TUI speaks via /usr/bin/say
(the same Apple voices) instead of the provider-keyed `voice.tts` path.
"""

import pytest

from tui_gateway import methods_voice as m


def _msgs():
    return [
        {"role": "user", "text": "hi"},
        {"role": "assistant", "text": "first reply"},
        {"role": "assistant", "text": "second reply"},
    ]


def test_resolve_empty_arg_picks_last_assistant_message():
    assert m._say_resolve_text("", _msgs()) == "second reply"


def test_resolve_numeric_arg_picks_nth_assistant_message():
    assert m._say_resolve_text("1", _msgs()) == "first reply"
    assert m._say_resolve_text("2", _msgs()) == "second reply"


def test_resolve_numeric_arg_clamps_out_of_range():
    assert m._say_resolve_text("99", _msgs()) == "second reply"


def test_resolve_literal_text_passes_through():
    assert m._say_resolve_text("hello there", _msgs()) == "hello there"


def test_resolve_empty_history_returns_empty():
    assert m._say_resolve_text("", []) == ""
    assert m._say_resolve_text("", [{"role": "user", "text": "hi"}]) == ""


class _FakePopen:
    instances = []

    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        self.pid = 1000 + len(_FakePopen.instances)
        self.terminated = False
        _FakePopen.instances.append(self)

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return None if not self.terminated else 0


@pytest.mark.macos_only
def test_say_start_speaks_and_stop_silences(monkeypatch):
    _FakePopen.instances.clear()
    monkeypatch.setattr(m.subprocess, "Popen", _FakePopen)
    m._say_stop_all()
    assert m._say_speaking() is False
    pid = m._say_start("hello world")
    assert isinstance(pid, int)
    assert _FakePopen.instances[-1].cmd[-1] == "hello world"
    assert "/say" in _FakePopen.instances[-1].cmd[0] or "say" in _FakePopen.instances[-1].cmd[0]
    assert m._say_speaking() is True
    m._say_stop_all()
    assert m._say_speaking() is False
    assert _FakePopen.instances[-1].terminated is True


@pytest.mark.macos_only
def test_say_start_replaces_current_utterance(monkeypatch):
    _FakePopen.instances.clear()
    monkeypatch.setattr(m.subprocess, "Popen", _FakePopen)
    m._say_stop_all()
    m._say_start("first")
    m._say_start("second")
    assert _FakePopen.instances[0].terminated is True
    assert _FakePopen.instances[-1].cmd[-1] == "second"
    assert m._say_speaking() is True
    m._say_stop_all()


def test_speak_methods_registered_on_server():
    from tui_gateway import server

    for name in ("speak.say", "speak.stop", "speak.status"):
        assert name in server._methods, f"{name} not registered"


def test_speak_status_answers_without_audio():
    from tui_gateway import server

    m._say_stop_all()
    result = server._methods["speak.status"](7, {})
    assert result["result"]["ok"] is True
    assert result["result"]["speaking"] is False


def test_speak_say_rejects_empty_text_without_spawning():
    from tui_gateway import server

    result = server._methods["speak.say"](7, {})
    assert "error" in result


def test_speak_mode_defaults_to_once(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert m._say_get_mode() == "once"


def test_speak_mode_round_trips_to_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert m._say_set_mode("always")[0] == "always"
    assert m._say_get_mode() == "always"
    assert (tmp_path / "speak-aloud.mode").read_text() == "always\n"
    assert m._say_set_mode("once")[0] == "once"
    assert m._say_get_mode() == "once"


def test_speak_mode_rejects_garbage():
    with pytest.raises(ValueError):
        m._say_set_mode("sometimes")


@pytest.mark.macos_only
def test_speak_mode_once_stops_playback(monkeypatch):
    _FakePopen.instances.clear()
    monkeypatch.setattr(m.subprocess, "Popen", _FakePopen)
    m._say_stop_all()
    m._say_start("first")
    assert m._say_speaking() is True
    assert m._say_set_mode("once")[0] == "once"
    assert m._say_speaking() is False
    assert _FakePopen.instances[-1].terminated is True
    m._say_stop_all()


def test_speak_mode_rpc_round_trip(tmp_path, monkeypatch):
    from tui_gateway import server

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    result = server._methods["speak.mode"](11, {"mode": "always"})
    assert result["result"] == {"ok": True, "mode": "always", "stopped": False}
    assert server._methods["speak.status"](12, {})["result"]["mode"] == "always"
    bad = server._methods["speak.mode"](13, {"mode": "sometimes"})
    assert "error" in bad
    server._methods["speak.mode"](14, {"mode": "once"})


def _say_items(text):
    from tui_gateway import methods_complete_helpers as h

    if not hasattr(h, "_item"):
        h._item = lambda text, meta, display=None: {"text": text, "display": display or text, "meta": meta}
    return h._say_completions(text)


def test_say_completions_list_options_on_bare_command():
    texts = [i["text"] for i in _say_items("/say")]
    assert texts == [" stop", " always", " once"]


def test_say_completions_list_options_after_space():
    assert [i["text"] for i in _say_items("/say ")] == ["stop", "always", "once"]


def test_say_completions_filter_by_prefix():
    assert [_say_items("/say a")[0]["text"]] == ["always"]
    assert [_say_items("/say o")[0]["text"]] == ["once"]
    assert [_say_items("/say s")[0]["text"]] == ["stop"]
    assert _say_items("/say x") == []


def test_say_completions_ignore_other_commands():
    assert _say_items("/copy") is None
    assert _say_items("/saying hi") is None
    assert _say_items("say") is None


def test_complete_slash_serves_say_options():
    from tui_gateway import server

    result = server._methods["complete.slash"](31, {"text": "/say "})
    texts = [i["text"] for i in result["result"]["items"]]
    assert texts == ["stop", "always", "once"]
    metas = {i["text"]: i["meta"] for i in result["result"]["items"]}
    assert "future reply" in metas["always"]
