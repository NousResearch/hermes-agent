from __future__ import annotations

from unittest.mock import patch

from tools.openclaw import vrchat_autonomy as va


def _ready_patches(monkeypatch):
    monkeypatch.setattr(va, "is_vrchat_process_running", lambda: True)
    monkeypatch.setattr(
        va,
        "inspect_vrchat_launch_state",
        lambda **kwargs: {"ok": True, "running": True},
    )
    monkeypatch.setattr(va, "python_osc_available", lambda: True)
    monkeypatch.setattr(va, "probe_harness", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(
        va,
        "find_output_device",
        lambda *args, **kwargs: {"ok": True, "configured": False},
    )


def test_readiness_irodori_skips_voicevox_requirement(monkeypatch):
    _ready_patches(monkeypatch)
    monkeypatch.setattr(
        va,
        "probe_irodori",
        lambda *args, **kwargs: {"ok": True, "url": "http://127.0.0.1:8088"},
    )

    result = va.vrchat_autonomy_readiness(
        tts_backend="irodori",
        require_voice=True,
    )

    assert result["ready"] is True
    assert "VOICEVOX Engine" not in result["missing"]
    assert result["checks"]["irodori"]["ok"] is True
    assert result["checks"]["voicevox"].get("skipped") is True


def test_readiness_voicevox_when_voice_required(monkeypatch):
    _ready_patches(monkeypatch)
    monkeypatch.setattr(
        va,
        "probe_voicevox",
        lambda *args, **kwargs: {"ok": False, "error": "connection_refused"},
    )

    result = va.vrchat_autonomy_readiness(
        tts_backend="voicevox",
        require_voice=True,
    )

    assert result["ready"] is False
    assert "VOICEVOX Engine" in result["missing"]


def test_readiness_move_only_skips_tts(monkeypatch):
    _ready_patches(monkeypatch)
    monkeypatch.setattr(
        va,
        "probe_voicevox",
        lambda *args, **kwargs: {"ok": False, "error": "connection_refused"},
    )

    result = va.vrchat_autonomy_readiness(
        tts_backend="voicevox",
        require_voice=False,
    )

    assert result["ready"] is True
    assert "VOICEVOX Engine" not in result["missing"]


def test_speak_tts_routes_to_irodori():
    with patch.object(va, "_speak_irodori", return_value={"success": True, "backend": "irodori"}) as mocked:
        result = va._speak_tts(
            "こんにちは",
            tts_backend="irodori",
            speaker=8,
            output_device=None,
            irodori_voice="hakua",
            irodori_speed=1.0,
            irodori_base_url=None,
        )
    assert result["success"] is True
    mocked.assert_called_once()


def test_validate_profile_rejects_unknown_tts_backend():
    result = va.validate_autonomy_profile({"mode": "private_test", "tts_backend": "edge-tts"})
    assert result["success"] is False
    assert any("invalid_tts_backend" in err for err in result["errors"])
