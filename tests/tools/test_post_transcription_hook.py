"""Tests for the ``post_transcription`` plugin hook wired into
``tools.transcription_tools.transcribe_audio``.

The symmetric partner of ``pre_transcription``: it fires after a backend
returned a successful, non-empty transcript and may replace that transcript
before it reaches the caller (gateway message enrichment, voice mode,
desktop upload), so an annotation lands in the persisted message text.

Covers the behavior contract:

1. Registration surface — the hook name is accepted by the plugin system.
2. A string return replaces the transcript; ``None`` leaves it unchanged.
3. Two hooks → last-writer-wins in registration order (same composition
   rule as ``pre_transcription``).
4. Non-string and empty/whitespace-only returns are ignored: a plugin may
   annotate a transcript but never erase one.
5. Only ``transcript`` is replaced — every other result field passes
   through untouched.
6. Failed and empty-but-successful transcriptions never fire the hook, so
   the gateway's "empty or inaudible" sentinel path keeps working.
7. A raising callback leaves the transcript unchanged (fail-open).
8. The kwargs contract, including the preprocessed ``file_path`` that is
   still readable while the callback runs.
9. No hook registered → ``invoke_hook`` is never called and the result is
   identical to a control run.

Mirrors the conventions of ``tests/tools/test_pre_transcription_hook.py``:
hook plumbing is faked at the ``hermes_cli.plugins`` boundary and the STT
backend is stubbed, so no model is loaded and no network call is made.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

import hermes_cli.plugins as plugins_mod
from tools import transcription_tools


TRANSCRIPT = "kannst du das bis heute abend fertig machen"
MARKER = "[Voice: stressed, fast] "


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_audio(tmp_path):
    audio = tmp_path / "voice.ogg"
    audio.write_bytes(b"fake audio data")
    return str(audio)


def _fake_hooks(monkeypatch, results):
    """Install fake has_hook/invoke_hook for ``post_transcription``.

    ``transcribe_audio`` fires ``pre_transcription`` on the same dispatch,
    so the fake is scoped by hook name: only ``post_transcription`` calls
    are recorded and served *results*. ``captured`` stays empty when the
    hook under test never fired.
    """
    captured = {}

    def _invoke(hook_name, **kw):
        if hook_name != "post_transcription":
            return []
        captured["hook_name"] = hook_name
        captured["kwargs"] = kw
        return [r(**kw) if callable(r) else r for r in results]

    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: True)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _invoke)
    return captured


def _no_hooks(monkeypatch):
    """No hook registered: has_hook is False and invoke_hook must not fire."""
    def _boom(hook_name, **kw):  # pragma: no cover - the assert is the point
        raise AssertionError(
            "invoke_hook must not be called when has_hook() is False"
        )

    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: False)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _boom)


def _dispatch_ctx(stt_config, provider):
    """Patch config load + provider resolution around transcribe_audio."""
    return (
        patch("tools.transcription_tools._load_stt_config", return_value=stt_config),
        patch("tools.transcription_tools._get_provider", return_value=provider),
    )


def _run(audio, backend_result, *, source=None):
    """Transcribe *audio* with the openai backend stubbed to *backend_result*."""
    backend = MagicMock(return_value=backend_result)
    cfg_patch, prov_patch = _dispatch_ctx({"provider": "openai"}, "openai")
    with cfg_patch, prov_patch, \
         patch("tools.transcription_tools._transcribe_openai", backend):
        return transcription_tools.transcribe_audio(audio, source=source)


def _ok(transcript=TRANSCRIPT, **extra):
    return {"success": True, "transcript": transcript, "provider": "openai", **extra}


# ---------------------------------------------------------------------------
# Hook registration surface
# ---------------------------------------------------------------------------


def test_post_transcription_in_valid_hooks():
    assert "post_transcription" in plugins_mod.VALID_HOOKS


def test_post_transcription_is_registerable_on_a_plugin_context():
    """The registration surface actually accepts the name (not just the set)."""
    manager = plugins_mod.PluginManager()
    manifest = plugins_mod.PluginManifest(name="probe")
    ctx = plugins_mod.PluginContext(manifest, manager)
    ctx.register_hook("post_transcription", lambda **kw: None)

    assert manager.has_hook("post_transcription")


# ---------------------------------------------------------------------------
# Replacement semantics
# ---------------------------------------------------------------------------


class TestReplacementSemantics:
    def test_string_return_replaces_transcript(self, monkeypatch, tmp_path):
        audio = _make_audio(tmp_path)
        _fake_hooks(monkeypatch, [lambda transcript, **kw: MARKER + transcript])

        result = _run(audio, _ok())

        assert result["transcript"] == MARKER + TRANSCRIPT

    def test_none_return_leaves_transcript_unchanged(self, monkeypatch, tmp_path):
        audio = _make_audio(tmp_path)
        _fake_hooks(monkeypatch, [lambda **kw: None])

        result = _run(audio, _ok())

        assert result["transcript"] == TRANSCRIPT

    def test_two_hooks_last_writer_wins(self, monkeypatch, tmp_path):
        """Registration order decides — same composition rule as
        pre_transcription. Both callbacks see the ORIGINAL transcript."""
        audio = _make_audio(tmp_path)
        seen = []

        def _first(transcript, **kw):
            seen.append(transcript)
            return "first"

        def _second(transcript, **kw):
            seen.append(transcript)
            return "second"

        _fake_hooks(monkeypatch, [_first, _second])

        result = _run(audio, _ok())

        assert result["transcript"] == "second"
        assert seen == [TRANSCRIPT, TRANSCRIPT]

    def test_non_string_return_ignored(self, monkeypatch, tmp_path):
        audio = _make_audio(tmp_path)
        _fake_hooks(monkeypatch, [{"transcript": "dict is not a string"}, 42])

        result = _run(audio, _ok())

        assert result["transcript"] == TRANSCRIPT

    def test_empty_return_cannot_erase_transcript(self, monkeypatch, tmp_path):
        """A plugin may annotate a transcript, never erase one: an empty
        replacement would turn real speech into the gateway's inaudible
        sentinel case."""
        audio = _make_audio(tmp_path)
        _fake_hooks(monkeypatch, [lambda **kw: ""])

        result = _run(audio, _ok())

        assert result["transcript"] == TRANSCRIPT

    def test_whitespace_only_return_cannot_erase_transcript(
        self, monkeypatch, tmp_path,
    ):
        audio = _make_audio(tmp_path)
        _fake_hooks(monkeypatch, [lambda **kw: "   \n  "])

        result = _run(audio, _ok())

        assert result["transcript"] == TRANSCRIPT

    def test_replacement_is_not_stripped_or_normalized(
        self, monkeypatch, tmp_path,
    ):
        """Surrounding whitespace decides nothing but emptiness — a returned
        string that survives the empty check is used verbatim, so a plugin
        controls its own separator."""
        audio = _make_audio(tmp_path)
        _fake_hooks(monkeypatch, [lambda **kw: f"[Voice: calm]\n{TRANSCRIPT} "])

        result = _run(audio, _ok())

        assert result["transcript"] == f"[Voice: calm]\n{TRANSCRIPT} "

    def test_ignored_return_after_valid_one_keeps_the_valid_one(
        self, monkeypatch, tmp_path,
    ):
        """Last-writer-wins counts only usable returns — an ignored value from
        a later hook does not undo an earlier valid replacement."""
        audio = _make_audio(tmp_path)
        _fake_hooks(monkeypatch, [lambda **kw: "annotated", lambda **kw: None])

        result = _run(audio, _ok())

        assert result["transcript"] == "annotated"


# ---------------------------------------------------------------------------
# Result envelope
# ---------------------------------------------------------------------------


class TestResultEnvelope:
    def test_only_transcript_is_replaced(self, monkeypatch, tmp_path):
        audio = _make_audio(tmp_path)
        _fake_hooks(monkeypatch, [lambda **kw: "annotated"])

        result = _run(
            audio, _ok(language="de", duration=1.5),
        )

        assert result["transcript"] == "annotated"
        assert result["success"] is True
        assert result["provider"] == "openai"
        assert result["language"] == "de"
        assert result["duration"] == 1.5

    def test_result_keys_unchanged_when_hook_replaces(
        self, monkeypatch, tmp_path,
    ):
        """The envelope keeps exactly the keys the backend produced."""
        audio = _make_audio(tmp_path)
        _fake_hooks(monkeypatch, [lambda **kw: "annotated"])
        backend_result = _ok(language="de")

        result = _run(audio, dict(backend_result))

        assert set(result) == set(backend_result)


# ---------------------------------------------------------------------------
# When the hook must NOT fire
# ---------------------------------------------------------------------------


class TestHookNotFired:
    def test_failed_transcription_does_not_fire(self, monkeypatch, tmp_path):
        audio = _make_audio(tmp_path)
        captured = _fake_hooks(monkeypatch, [lambda **kw: "annotated"])

        result = _run(
            audio,
            {"success": False, "transcript": "", "error": "boom",
             "provider": "openai"},
        )

        assert captured == {}
        assert result["transcript"] == ""
        assert result["error"] == "boom"

    def test_empty_successful_transcript_does_not_fire(
        self, monkeypatch, tmp_path,
    ):
        """The gateway's 'empty or inaudible' sentinel path must keep seeing a
        genuinely empty transcript — a plugin annotation would defeat it."""
        audio = _make_audio(tmp_path)
        captured = _fake_hooks(monkeypatch, [lambda **kw: "annotated"])

        result = _run(audio, _ok(transcript=""))

        assert captured == {}
        assert result["transcript"] == ""

    def test_whitespace_only_transcript_does_not_fire(
        self, monkeypatch, tmp_path,
    ):
        audio = _make_audio(tmp_path)
        captured = _fake_hooks(monkeypatch, [lambda **kw: "annotated"])

        result = _run(audio, _ok(transcript="   \n "))

        assert captured == {}
        assert result["transcript"] == "   \n "


# ---------------------------------------------------------------------------
# Fail-open
# ---------------------------------------------------------------------------


class TestFailOpen:
    def test_raising_callback_leaves_transcript_unchanged(
        self, monkeypatch, tmp_path,
    ):
        audio = _make_audio(tmp_path)

        def _explode(hook_name, **kw):
            raise RuntimeError("plugin blew up")

        monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: True)
        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _explode)

        result = _run(audio, _ok())

        assert result["success"] is True
        assert result["transcript"] == TRANSCRIPT

    def test_unimportable_hook_plumbing_leaves_transcript_unchanged(
        self, monkeypatch, tmp_path,
    ):
        audio = _make_audio(tmp_path)

        def _explode(name):
            raise ImportError("plugins unavailable")

        monkeypatch.setattr("hermes_cli.plugins.has_hook", _explode)

        result = _run(audio, _ok())

        assert result["transcript"] == TRANSCRIPT


# ---------------------------------------------------------------------------
# Kwargs contract
# ---------------------------------------------------------------------------


class TestKwargsContract:
    def test_hook_receives_expected_kwargs(self, monkeypatch, tmp_path):
        audio = _make_audio(tmp_path)
        captured = _fake_hooks(monkeypatch, [])

        _run(audio, _ok(), source="gateway")

        assert captured["hook_name"] == "post_transcription"
        kw = captured["kwargs"]
        assert kw["file_path"] == audio
        assert kw["transcript"] == TRANSCRIPT
        assert kw["provider"] == "openai"
        assert kw["source"] == "gateway"

    def test_audio_file_is_still_readable_inside_the_callback(
        self, monkeypatch, tmp_path,
    ):
        """The hook fires before any temp-file cleanup, so a callback that
        wants to analyse the audio it was told about can still open it."""
        audio = _make_audio(tmp_path)
        observed = {}

        def _read_audio(file_path, transcript, **kw):
            observed["exists"] = os.path.exists(file_path)
            observed["bytes"] = Path(file_path).read_bytes()
            return None

        _fake_hooks(monkeypatch, [_read_audio])

        _run(audio, _ok())

        assert observed["exists"] is True
        assert observed["bytes"] == b"fake audio data"

    def test_pre_and_post_hooks_agree_on_file_path(self, monkeypatch, tmp_path):
        """The start-in-pre / join-in-post pattern depends on both hooks
        naming the same file."""
        audio = _make_audio(tmp_path)
        seen = {}

        def _invoke(hook_name, **kw):
            seen[hook_name] = kw.get("file_path")
            return []

        monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: True)
        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _invoke)

        _run(audio, _ok())

        assert seen["pre_transcription"] == seen["post_transcription"]


# ---------------------------------------------------------------------------
# No-hook path stays identical
# ---------------------------------------------------------------------------


class TestNoHookPath:
    def test_no_hook_result_identical_to_control(self, monkeypatch, tmp_path):
        audio = _make_audio(tmp_path)
        _no_hooks(monkeypatch)  # invoke_hook raises if ever called
        backend_result = _ok(language="de")

        result = _run(audio, dict(backend_result))

        assert result == backend_result


# ---------------------------------------------------------------------------
# End-to-end with a real fixture plugin (real PluginManager mechanics)
# ---------------------------------------------------------------------------


def test_real_fixture_plugin_annotates_the_transcript(monkeypatch, tmp_path):
    """Two callbacks registered by a real plugin through the real discovery
    and dispatch path — last-writer-wins, verified on the returned result."""
    hermes_home = Path(os.environ["HERMES_HOME"])
    plugin_dir = hermes_home / "plugins" / "stt_marker"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: stt_marker\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(
        "def register(ctx):\n"
        '    ctx.register_hook("post_transcription", lambda **kw: "loser")\n'
        '    ctx.register_hook(\n'
        '        "post_transcription",\n'
        '        lambda transcript, **kw: "[Voice: calm] " + transcript,\n'
        "    )\n",
        encoding="utf-8",
    )
    cfg_path = hermes_home / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"plugins": {"enabled": ["stt_marker"]}}),
        encoding="utf-8",
    )

    old_manager = plugins_mod._plugin_manager
    plugins_mod._plugin_manager = plugins_mod.PluginManager()
    try:
        plugins_mod.discover_plugins()

        audio = _make_audio(tmp_path)
        result = _run(audio, _ok())
    finally:
        plugins_mod._plugin_manager = old_manager

    assert result["success"] is True
    assert result["transcript"] == f"[Voice: calm] {TRANSCRIPT}"
