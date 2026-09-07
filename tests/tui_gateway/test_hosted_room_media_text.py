"""Native media staging: a room voice message must reach the member as words.

Every schedule drives the **real installed handlers** of the independently loaded backend from
``test_hosted_room_native_phase1`` — real ``file.attach`` staging into a real profile home, the
real ``HostedRoomServerRPC.stage_attachment`` and the real server-bound profile runtime scope —
with real bytes on disk.  Only the STT provider call is stood in for.

What this can prove: the bytes the member's model is told about are the canonical bytes, the
transcription resolves the *member profile's* config and secrets, and video keeps the native
"content is not inlined" contract.  What it cannot prove: the behaviour of any real STT
provider, which is a separate live gate.
"""

from __future__ import annotations

import io
import wave
from pathlib import Path
from typing import Any

import yaml

import hermes_constants
import tools.transcription_tools as transcription_tools
from agent.secret_scope import current_secret_scope
from gateway import media_text

from tests.tui_gateway.test_hosted_room_native_phase1 import native  # noqa: F401  (fixture)


ROOM_TITLE = "Group: media-room"
SPOKEN = "ship it on friday"


def _wav_bytes(*, frames: int = 800) -> bytes:
    """A real 16-bit PCM WAV container with real samples, built here so nothing is downloaded."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(bytes(frames * 2))
    return buffer.getvalue()


def _manifest(name: str, mime: str, data: bytes, *, index: int = 1) -> dict[str, Any]:
    return {
        "attachment_id": f"att_{index:032x}",
        "kind": "file",
        "name": name,
        "size": len(data),
        "mime": mime,
    }


def _profile(home: Path, name: str, *, stt_enabled: bool, api_key: str) -> Path:
    """One named profile home with its own STT switch and its own (fake) provider credential."""
    root = home / "profiles" / name
    root.mkdir(parents=True)
    (root / "config.yaml").write_text(
        yaml.dump({"stt": {"enabled": stt_enabled}}), encoding="utf-8")
    (root / ".env").write_text(f"GROQ_API_KEY={api_key}\n", encoding="utf-8")
    return root


def _session_for(native, profile: str, title: str = ROOM_TITLE, *, cwd: str | None = None) -> str:
    params = {
        "profile": profile, "title": title, "source": "bot_room", "hidden": True,
        "room_plumbing": True, "follow_profile_config": True, "close_on_disconnect": False}
    if cwd is not None:
        params["cwd"] = cwd
    result = native.srv._methods["session.create"]("rid-create", params)
    assert "error" not in result, result
    return str(result["result"]["session_id"])


def _record_stt(native, transcript: str | None = SPOKEN) -> list[dict[str, Any]]:
    """Stand in for the provider only, recording what scope and bytes it was actually given."""
    seen: list[dict[str, Any]] = []

    def fake_transcribe(path, model=None, source=None):
        seen.append({
            "path": str(path),
            "data": Path(path).read_bytes(),
            "home": str(hermes_constants.get_hermes_home()),
            "secrets": dict(current_secret_scope() or {}),
            "source": source,
        })
        if transcript is None:
            return {"success": False, "error": "provider refused"}
        return {"success": True, "transcript": transcript}

    native.monkeypatch.setattr(transcription_tools, "transcribe_audio", fake_transcribe)
    native.monkeypatch.setattr(
        transcription_tools, "transcribe_audio_local_fallback",
        lambda path, model=None: {"success": False, "error": "no local backend"})
    return seen


def _stage(native, *, profile: str, session_id: str, attachment, data: bytes):
    return native.rpc().stage_attachment(
        profile=profile, session_id=session_id, source="bot_room",
        attachment=attachment, data=data, execution_generation=1)


def test_staged_voice_is_transcribed_from_the_canonical_bytes_on_disk(native):
    """The provider is handed the staged file itself, and its words come back as the note."""
    _profile(native.home, "alpha", stt_enabled=True, api_key="alpha-not-a-real-key")
    session_id = _session_for(native, "alpha")
    audio = _wav_bytes()
    seen = _record_stt(native)

    staged = _stage(
        native, profile="alpha", session_id=session_id,
        attachment=_manifest("voice.wav", "audio/wav", audio), data=audio)

    assert staged["derived_text"] == f'"{SPOKEN}"'
    assert len(seen) == 1
    # Real staging, not a re-encoded copy: the model is told about these exact bytes.
    assert seen[0]["data"] == audio
    assert Path(staged["path"]).read_bytes() == audio
    assert staged["ref_text"].startswith("@file:")
    assert seen[0]["source"] == "gateway"


def test_transcription_resolves_the_member_profile_not_this_backend(native):
    """Opposite switches in two homes: each member's own config and secrets decide.

    ``file.attach`` is a plain handler call whose scope does not outlive its return, so the
    canonical session record has to be rebound for the transcription itself.
    """
    alpha = _profile(native.home, "alpha", stt_enabled=True, api_key="alpha-not-a-real-key")
    beta = _profile(native.home, "beta", stt_enabled=False, api_key="beta-not-a-real-key")
    audio = _wav_bytes()
    seen = _record_stt(native)

    alpha_staged = _stage(
        native, profile="alpha", session_id=_session_for(native, "alpha", "Group: room-a"),
        attachment=_manifest("voice.wav", "audio/wav", audio), data=audio)
    beta_staged = _stage(
        native, profile="beta", session_id=_session_for(native, "beta", "Group: room-b"),
        attachment=_manifest("voice.wav", "audio/wav", audio, index=2), data=audio)

    assert alpha_staged["derived_text"] == f'"{SPOKEN}"'
    # The member that turned STT off is not transcribed at all — not merely dropped afterwards.
    assert beta_staged["derived_text"] == ""
    assert len(seen) == 1
    assert seen[0]["home"] == str(alpha)
    assert seen[0]["secrets"].get("GROQ_API_KEY") == "alpha-not-a-real-key"
    assert str(beta) not in seen[0]["home"]
    # The scope is released with the staging call, not left bound on this thread.
    assert str(hermes_constants.get_hermes_home()) == str(native.home)


def test_video_is_staged_for_the_media_tools_and_never_transcribed(native):
    """Native video keeps its contract: bytes on disk, no inlined content, no STT."""
    _profile(native.home, "alpha", stt_enabled=True, api_key="alpha-not-a-real-key")
    session_id = _session_for(native, "alpha")
    clip = b"\x00\x00\x00\x18ftypmp42" + bytes(64)
    seen = _record_stt(native)

    staged = _stage(
        native, profile="alpha", session_id=session_id,
        attachment=_manifest("clip.mp4", "video/mp4", clip), data=clip)

    assert staged["derived_text"] == ""
    assert seen == []
    assert Path(staged["path"]).read_bytes() == clip
    assert staged["ref_text"].endswith("clip.mp4")


def test_failed_transcription_keeps_the_native_agent_reference(native):
    """A turn carrying a voice message reaches the model either as words or as one marker.

    ``_attachment_ref_path`` is workspace-relative for an attachment inside the session
    workspace and absolute for one outside it; the marker carries whichever reference the
    agent can actually open, and says nothing about providers or setup.
    """
    profile_home = _profile(native.home, "alpha", stt_enabled=True, api_key="alpha-not-a-real-key")
    audio = _wav_bytes()
    _record_stt(native, transcript=None)

    inside = _stage(
        native, profile="alpha",
        session_id=_session_for(native, "alpha", "Group: room-inside", cwd=str(profile_home)),
        attachment=_manifest("voice.wav", "audio/wav", audio), data=audio)
    outside = _stage(
        native, profile="alpha", session_id=_session_for(native, "alpha", "Group: room-outside"),
        attachment=_manifest("voice.wav", "audio/wav", audio, index=2), data=audio)

    for staged in (inside, outside):
        assert staged["derived_text"] == media_text.untranscribed_note(staged["ref_path"])
        assert Path(staged["path"]).read_bytes() == audio
        for advice in ("STT", "provider", "refused", "install", "configure", "API key"):
            assert advice not in staged["derived_text"]

    # Both native reference shapes, taken from the sessions' genuine workspace settings.
    assert not Path(inside["ref_path"]).is_absolute()
    assert Path(outside["ref_path"]).is_absolute()


def test_empty_transcription_is_a_sentinel_the_model_can_act_on(native):
    _profile(native.home, "alpha", stt_enabled=True, api_key="alpha-not-a-real-key")
    session_id = _session_for(native, "alpha")
    audio = _wav_bytes()
    _record_stt(native, transcript="   ")

    staged = _stage(
        native, profile="alpha", session_id=session_id,
        attachment=_manifest("voice.wav", "audio/wav", audio), data=audio)

    assert staged["derived_text"] == media_text.EMPTY_TRANSCRIPT_NOTE


def test_a_voice_upload_without_an_extension_is_decoded_through_its_mime(native):
    """A voice note named by its sender, not by its container, still reaches STT."""
    _profile(native.home, "alpha", stt_enabled=True, api_key="alpha-not-a-real-key")
    session_id = _session_for(native, "alpha")
    audio = _wav_bytes()
    seen = _record_stt(native)

    staged = _stage(
        native, profile="alpha", session_id=session_id,
        attachment=_manifest("voice-note", "audio/wav", audio), data=audio)

    assert staged["derived_text"] == f'"{SPOKEN}"'
    assert seen[0]["path"].endswith(".wav")
    assert seen[0]["data"] == audio
    # The temporary decode copy is not what the model is pointed at, and it does not survive.
    assert seen[0]["path"] != staged["path"]
    assert not Path(seen[0]["path"]).exists()
    assert Path(staged["path"]).read_bytes() == audio
