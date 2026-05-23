from __future__ import annotations

import json

from tools.openclaw import vrchat_autonomy as autonomy
from tools.openclaw import vrchat_completion_audit as audit_mod
from tools.openclaw import vrchat_preflight


def test_completion_audit_reports_incomplete_live_smoke(monkeypatch, tmp_path):
    monkeypatch.setattr(autonomy, "is_vrchat_process_running", lambda: False)
    monkeypatch.setattr(autonomy, "python_osc_available", lambda: True)
    monkeypatch.setattr(autonomy, "probe_voicevox", lambda url: {"ok": True, "url": url, "version": "test"})
    monkeypatch.setattr(autonomy, "probe_harness", lambda url: {"ok": False, "url": url})
    monkeypatch.setattr(autonomy, "find_output_device", lambda name: {"ok": True, "configured": True})
    monkeypatch.setattr(vrchat_preflight, "vrchat_autonomy_readiness", autonomy.vrchat_autonomy_readiness)
    monkeypatch.setattr(vrchat_preflight, "vrchat_autonomy_heartbeat", autonomy.vrchat_autonomy_heartbeat)
    monkeypatch.setattr(vrchat_preflight, "find_output_device", autonomy.find_output_device)
    monkeypatch.setattr(vrchat_preflight, "probe_voicevox_synthesis", _synthesis_ok)
    monkeypatch.setattr(vrchat_preflight, "check_virtual_cable_route", _virtual_cable_ok)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                **autonomy._default_profile(),
                "enabled": True,
                "mode": "private_test",
                "dry_run": True,
                "allow_voice": True,
                "allow_chatbox": True,
                "audio_output_device": "CABLE Input",
            }
        ),
        encoding="utf-8",
    )

    audit = audit_mod.build_completion_audit(
        profile_path=profile_path,
        audio_output_device="CABLE Input",
    )

    assert audit["success"] is True
    assert audit["complete"] is False
    assert "private_live_smoke_verified" in audit["incomplete_requirements"]
    assert "vrchat_runtime_ready" in audit["incomplete_requirements"]
    assert "vrchat_process_phase" in audit["preflight_summary"]
    assert audit["preflight_summary"]["vrchat_process_diagnostic"]
    assert "voicevox_ok" in audit["preflight_summary"]
    assert audit["preflight_summary"]["voicevox_synthesis_ok"] is True
    assert "dry_run_multimodal_turn" not in audit["incomplete_requirements"]
    assert audit["dry_run_multimodal_turn"]["has_chatbox"] is True
    assert audit["dry_run_multimodal_turn"]["has_voice"] is True
    assert "conversation_dry_run_harness" not in audit["incomplete_requirements"]
    assert audit["conversation_dry_run"]["has_chatbox"] is True
    assert audit["conversation_dry_run"]["has_voice"] is True
    assert "private_smoke_prepare_harness" not in audit["incomplete_requirements"]
    assert audit["private_smoke_prepare"]["code"] == "PRIVATE_SMOKE_BLOCKED"
    assert audit["private_smoke_prepare"]["safety"]["actuation_performed"] is False
    assert "wait_then_private_smoke_harness" not in audit["incomplete_requirements"]
    assert "neuro_action_dry_run_route" not in audit["incomplete_requirements"]
    assert audit["neuro_action_dry_run_route"]["has_chatbox"] is True
    assert audit["neuro_action_dry_run_route"]["has_voice"] is True
    assert audit["safety"]["actuation_performed"] is False


def test_completion_audit_requires_voice_and_chatbox_profile_permissions(monkeypatch, tmp_path):
    monkeypatch.setattr(autonomy, "is_vrchat_process_running", lambda: True)
    monkeypatch.setattr(autonomy, "python_osc_available", lambda: True)
    monkeypatch.setattr(autonomy, "probe_voicevox", lambda url: {"ok": True, "url": url, "version": "test"})
    monkeypatch.setattr(autonomy, "probe_harness", lambda url: {"ok": False, "url": url})
    monkeypatch.setattr(autonomy, "find_output_device", lambda name: {"ok": True, "configured": True})
    monkeypatch.setattr(vrchat_preflight, "vrchat_autonomy_readiness", autonomy.vrchat_autonomy_readiness)
    monkeypatch.setattr(vrchat_preflight, "vrchat_autonomy_heartbeat", autonomy.vrchat_autonomy_heartbeat)
    monkeypatch.setattr(vrchat_preflight, "find_output_device", autonomy.find_output_device)
    monkeypatch.setattr(vrchat_preflight, "probe_voicevox_synthesis", _synthesis_ok)
    monkeypatch.setattr(vrchat_preflight, "check_virtual_cable_route", _virtual_cable_ok)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                **autonomy._default_profile(),
                "enabled": True,
                "mode": "private_test",
                "dry_run": True,
                "allow_voice": False,
                "allow_chatbox": True,
                "audio_output_device": "CABLE Input",
            }
        ),
        encoding="utf-8",
    )

    audit = audit_mod.build_completion_audit(
        profile_path=profile_path,
        audio_output_device="CABLE Input",
    )

    assert "safe_autonomous_actions" in audit["incomplete_requirements"]
    assert "dry_run_multimodal_turn" in audit["incomplete_requirements"]
    assert "neuro_action_dry_run_route" in audit["incomplete_requirements"]
    assert "conversation_dry_run_harness" in audit["incomplete_requirements"]
    assert "private_smoke_prepare_harness" in audit["incomplete_requirements"]
    assert "voice_not_allowed" in _requirement(audit, "safe_autonomous_actions")["blockers"]
    assert "voice_plan_missing" in _requirement(audit, "dry_run_multimodal_turn")["blockers"]
    assert "voice_plan_missing" in _requirement(audit, "neuro_action_dry_run_route")["blockers"]
    assert "voice_plan_missing" in _requirement(audit, "conversation_dry_run_harness")["blockers"]


def test_completion_audit_requires_voicevox_synthesis_probe(monkeypatch, tmp_path):
    monkeypatch.setattr(autonomy, "is_vrchat_process_running", lambda: True)
    monkeypatch.setattr(autonomy, "python_osc_available", lambda: True)
    monkeypatch.setattr(autonomy, "probe_voicevox", lambda url: {"ok": True, "url": url, "version": "test"})
    monkeypatch.setattr(autonomy, "probe_harness", lambda url: {"ok": False, "url": url})
    monkeypatch.setattr(autonomy, "find_output_device", lambda name: {"ok": True, "configured": True})
    monkeypatch.setattr(vrchat_preflight, "vrchat_autonomy_readiness", autonomy.vrchat_autonomy_readiness)
    monkeypatch.setattr(vrchat_preflight, "vrchat_autonomy_heartbeat", autonomy.vrchat_autonomy_heartbeat)
    monkeypatch.setattr(vrchat_preflight, "find_output_device", autonomy.find_output_device)
    monkeypatch.setattr(
        vrchat_preflight,
        "probe_voicevox_synthesis",
        lambda **kwargs: {
            "success": False,
            "included": True,
            "ok": False,
            "wav_header_ok": False,
            "safety": {"speech_played": False, "microphone_recorded": False},
        },
    )
    monkeypatch.setattr(vrchat_preflight, "check_virtual_cable_route", _virtual_cable_ok)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                **autonomy._default_profile(),
                "enabled": True,
                "mode": "private_test",
                "dry_run": True,
                "allow_voice": True,
                "allow_chatbox": True,
                "audio_output_device": "CABLE Input",
            }
        ),
        encoding="utf-8",
    )

    audit = audit_mod.build_completion_audit(
        profile_path=profile_path,
        audio_output_device="CABLE Input",
    )

    assert "voicevox_virtual_cable" in audit["incomplete_requirements"]
    requirement = _requirement(audit, "voicevox_virtual_cable")
    assert "voicevox_synthesis_failed" in requirement["blockers"]
    assert "voicevox_wav_header_invalid" in requirement["blockers"]


def test_completion_audit_writes_output_path(monkeypatch, tmp_path):
    monkeypatch.setattr(autonomy, "is_vrchat_process_running", lambda: False)
    monkeypatch.setattr(autonomy, "python_osc_available", lambda: False)
    monkeypatch.setattr(autonomy, "probe_voicevox", lambda url: {"ok": False, "url": url})
    monkeypatch.setattr(autonomy, "probe_harness", lambda url: {"ok": False, "url": url})
    monkeypatch.setattr(autonomy, "find_output_device", lambda name: {"ok": None, "configured": False})
    monkeypatch.setattr(vrchat_preflight, "vrchat_autonomy_readiness", autonomy.vrchat_autonomy_readiness)
    monkeypatch.setattr(vrchat_preflight, "vrchat_autonomy_heartbeat", autonomy.vrchat_autonomy_heartbeat)
    monkeypatch.setattr(vrchat_preflight, "find_output_device", autonomy.find_output_device)
    monkeypatch.setattr(vrchat_preflight, "probe_voicevox_synthesis", _synthesis_failed)
    monkeypatch.setattr(vrchat_preflight, "check_virtual_cable_route", _virtual_cable_not_configured)
    output_path = tmp_path / "audit.json"

    audit = audit_mod.build_completion_audit(output_path=output_path)

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["output_path"] == str(output_path)
    assert audit["output_path"] == str(output_path)
    assert saved["complete"] is False


def _requirement(audit: dict, item_id: str) -> dict:
    return next(item for item in audit["requirements"] if item["id"] == item_id)


def _synthesis_ok(**kwargs) -> dict:
    return {
        "success": True,
        "included": True,
        "ok": True,
        "size_bytes": 128,
        "wav_header_ok": True,
        "safety": {"speech_played": False, "microphone_recorded": False},
    }


def _synthesis_failed(**kwargs) -> dict:
    return {
        "success": False,
        "included": True,
        "ok": False,
        "error": "ConnectError",
        "safety": {"speech_played": False, "microphone_recorded": False},
    }


def _virtual_cable_ok(**kwargs) -> dict:
    return {
        "success": True,
        "included": True,
        "configured": True,
        "ok": True,
        "output_device": "CABLE Input",
        "microphone_device": "CABLE Output",
        "playback": {"ok": True, "matches": [{"name": "CABLE Input"}]},
        "microphone": {"ok": True, "matches": [{"name": "CABLE Output"}]},
        "safety": {"microphone_recorded": False, "speech_played": False},
    }


def _virtual_cable_not_configured(**kwargs) -> dict:
    return {
        "success": True,
        "included": True,
        "configured": False,
        "ok": None,
        "playback": {"ok": None, "matches": []},
        "microphone": {"ok": None, "matches": []},
        "safety": {"microphone_recorded": False, "speech_played": False},
    }
