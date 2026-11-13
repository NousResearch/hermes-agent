"""``transform_spoken_text``: the one dispatch every speech path calls before synthesis.

Contract from issue #152238: no plugin leaves the text unchanged; the first ``str`` result wins,
empty string included ("speak nothing"); other results are ignored; a failing plugin falls back
to the unchanged text. Plugins load through real discovery from a temp ``HERMES_HOME``.
Overlapping calls with no session id (desktop read-aloud) each reach the plugin.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

import hermes_cli.plugins as plugins_mod
import hermes_yaml as yaml
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from tools.tts_spoken_text_hook import apply_spoken_text_hook


def _load_plugin(body: str) -> None:
    home = Path(os.environ["HERMES_HOME"])
    plugin_dir = home / "plugins" / "spoken_text_probe"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: spoken_text_probe\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(body, encoding="utf-8")
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["spoken_text_probe"]}}), encoding="utf-8")
    plugins_mod._reset_plugin_managers_for_tests()
    plugins_mod.discover_plugins()


def _speak(text="Fine. It passed.", **overrides):
    kwargs = {"surface": "voice_reply", "platform": "telegram", "session_id": "sess-1", **overrides}
    return apply_spoken_text_hook(text, **kwargs)


def test_no_plugin_leaves_text_unchanged():
    plugins_mod._reset_plugin_managers_for_tests()
    plugins_mod.discover_plugins()

    assert _speak("[sighs] Fine.") == "[sighs] Fine."


def test_plugin_replacement_is_spoken_and_sees_the_call_identity():
    _load_plugin(
        "def register(ctx):\n"
        "    ctx.register_hook('transform_spoken_text', lambda text, surface, platform, session_id,"
        " index, **kw: f'{surface}|{platform}|{session_id}|{index}|{text}')\n")

    assert _speak("Hello.", surface="streaming", index=3) == "streaming|telegram|sess-1|3|Hello."


def test_first_string_wins_in_registration_order_and_non_strings_are_ignored():
    _load_plugin(
        "def register(ctx):\n"
        "    ctx.register_hook('transform_spoken_text', lambda **kw: None)\n"
        "    ctx.register_hook('transform_spoken_text', lambda **kw: {'text': 'dict'})\n"
        "    ctx.register_hook('transform_spoken_text', lambda **kw: 'first')\n"
        "    ctx.register_hook('transform_spoken_text', lambda **kw: 'second')\n")

    assert _speak() == "first"


def test_empty_string_means_speak_nothing():
    _load_plugin(
        "def register(ctx):\n"
        "    ctx.register_hook('transform_spoken_text', lambda **kw: '')\n"
        "    ctx.register_hook('transform_spoken_text', lambda **kw: 'later')\n")

    assert _speak() == ""


def test_raising_plugin_falls_back_to_the_unchanged_text():
    _load_plugin(
        "def _boom(**kw):\n"
        "    raise RuntimeError('plugin bug')\n"
        "def register(ctx):\n"
        "    ctx.register_hook('transform_spoken_text', _boom)\n")

    assert _speak("Still spoken.") == "Still spoken."


def test_raising_plugin_does_not_block_a_later_replacement():
    _load_plugin(
        "def _boom(**kw):\n"
        "    raise RuntimeError('plugin bug')\n"
        "def register(ctx):\n"
        "    ctx.register_hook('transform_spoken_text', _boom)\n"
        "    ctx.register_hook('transform_spoken_text', lambda text, **kw: text.upper())\n")

    assert _speak("quiet.") == "QUIET."


def test_text_the_normalizer_emptied_never_reaches_a_plugin():
    _load_plugin(
        "def register(ctx):\n"
        "    ctx.register_hook('transform_spoken_text', lambda **kw: 'revived')\n")

    assert _speak("") == ""


@pytest.mark.parametrize("session_id, second_spoken", [
    ("", "[cue] second"),   # no session id (desktop read-aloud): every call is its own call
    ("sess-1", "second"),   # one session's speech is one sequence: the overlapping call is skipped
])
def test_overlapping_calls_are_told_apart_when_there_is_no_session_id(monkeypatch, session_id, second_spoken):
    manager = PluginManager()
    manager._discovered = True
    monkeypatch.setattr(plugins_mod, "_plugin_manager", manager)
    monkeypatch.setattr(plugins_mod, "_plugin_managers_by_home", {})
    entered, release, spoken = threading.Event(), threading.Event(), {}

    def cue(text, **_kw):
        if text == "first":
            entered.set()
            release.wait(10)
        return f"[cue] {text}"

    PluginContext(PluginManifest(name="cue-probe", source="user"), manager).register_hook(
        "transform_spoken_text", cue)

    def _read_aloud(text):
        spoken[text] = apply_spoken_text_hook(
            text, surface="read_aloud", platform="desktop", session_id=session_id)

    first = threading.Thread(target=_read_aloud, args=("first",))
    first.start()
    assert entered.wait(10)
    try:
        _read_aloud("second")
    finally:
        release.set()
        first.join(10)

    assert spoken == {"first": "[cue] first", "second": second_spoken}
