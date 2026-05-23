from __future__ import annotations

import json

from tools.openclaw import vrchat_autonomy as autonomy
from tools.openclaw import vrchat_smoke


def _profile(**overrides):
    profile = {
        **autonomy._default_profile(),
        "enabled": True,
        "mode": "private_test",
        "dry_run": True,
        "allow_voice": True,
        "allow_chatbox": True,
        "allowed_avatar_actions": ["wave"],
        "avatar_action_profiles": {"wave": [{"name": "Wave", "value": True}]},
    }
    profile.update(overrides)
    return profile


def _write_profile(tmp_path, **overrides):
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(_profile(**overrides)), encoding="utf-8")
    return profile_path


def _ready(monkeypatch):
    monkeypatch.setattr(autonomy, "is_vrchat_process_running", lambda: True)
    monkeypatch.setattr(autonomy, "python_osc_available", lambda: True)
    monkeypatch.setattr(autonomy, "probe_voicevox", lambda url: {"ok": True, "url": url})
    monkeypatch.setattr(autonomy, "probe_harness", lambda url: {"ok": True, "url": url})
    monkeypatch.setattr(autonomy, "find_output_device", lambda name: {"ok": None, "configured": False})
    monkeypatch.setattr(vrchat_smoke, "vrchat_autonomy_readiness", autonomy.vrchat_autonomy_readiness)


def test_private_smoke_blocks_missing_profile(tmp_path):
    result = vrchat_smoke.run_private_smoke(profile_path=tmp_path / "missing.json")

    assert result["success"] is False
    assert result["code"] == "PROFILE_BLOCKED"
    assert result["turn"] is None
    assert result["safety"]["actuation_performed"] is False


def test_private_smoke_dry_run_plans_without_actuation(monkeypatch, tmp_path):
    _ready(monkeypatch)
    profile_path = _write_profile(tmp_path)

    result = vrchat_smoke.run_private_smoke(
        profile_path=profile_path,
        avatar_action="wave",
    )

    assert result["success"] is True
    assert result["code"] == "DRY_RUN_SMOKE_DONE"
    assert result["turn"]["dry_run"] is True
    assert [action["kind"] for action in result["turn"]["planned_actions"]] == [
        "chatbox",
        "voice",
        "avatar_action",
    ]
    assert result["safety"]["actuation_performed"] is False


def test_prepare_private_smoke_reports_live_blockers_without_actuation(monkeypatch, tmp_path):
    _ready(monkeypatch)
    profile_path = _write_profile(tmp_path, dry_run=True)

    result = vrchat_smoke.prepare_private_smoke(
        profile_path=profile_path,
        live_ack=autonomy.LIVE_ACTUATION_ACK,
    )

    assert result["success"] is True
    assert result["code"] == "PRIVATE_SMOKE_BLOCKED"
    assert result["would_execute_live"] is False
    assert "profile_dry_run_true" in result["live_gate"]["blocked_reasons"]
    assert result["turn"]["dry_run"] is True
    assert result["safety"]["actuation_performed"] is False


def test_prepare_private_smoke_can_be_ready_without_live_execution(monkeypatch, tmp_path):
    _ready(monkeypatch)
    profile_path = _write_profile(
        tmp_path,
        dry_run=False,
        live_actuation_ack=autonomy.LIVE_ACTUATION_ACK,
    )
    calls = []

    def fake_chatbox(text, *, immediate):
        calls.append(("chatbox", text, immediate))
        return {"kind": "chatbox", "attempted": True, "success": True}

    monkeypatch.setattr(autonomy, "_send_chatbox", fake_chatbox)
    monkeypatch.setattr(vrchat_smoke, "plan_autonomy_turn", autonomy.plan_autonomy_turn)

    result = vrchat_smoke.prepare_private_smoke(
        profile_path=profile_path,
        live_ack=autonomy.LIVE_ACTUATION_ACK,
    )

    assert result["success"] is True
    assert result["code"] == "PRIVATE_SMOKE_READY"
    assert result["would_execute_live"] is True
    assert result["live_gate"]["allowed"] is True
    assert result["turn"]["dry_run"] is True
    assert result["safety"]["actuation_performed"] is False
    assert calls == []


def test_private_smoke_live_requires_ack_and_non_dry_profile(monkeypatch, tmp_path):
    _ready(monkeypatch)
    profile_path = _write_profile(tmp_path, dry_run=True)

    result = vrchat_smoke.run_private_smoke(
        profile_path=profile_path,
        live=True,
        live_ack=autonomy.LIVE_ACTUATION_ACK,
    )

    assert result["success"] is True
    assert result["code"] == "DRY_RUN_SMOKE_DONE"
    assert "profile_dry_run_true" in result["live_gate"]["blocked_reasons"]
    assert result["turn"]["dry_run"] is True


def test_private_smoke_live_executes_only_after_all_gates(monkeypatch, tmp_path):
    _ready(monkeypatch)
    profile_path = _write_profile(
        tmp_path,
        dry_run=False,
        live_actuation_ack=autonomy.LIVE_ACTUATION_ACK,
    )
    calls = []

    def fake_chatbox(text, *, immediate):
        calls.append(("chatbox", text, immediate))
        return {"kind": "chatbox", "attempted": True, "success": True}

    def fake_voice(text, *, speaker, output_device):
        calls.append(("voice", text, speaker, output_device))
        return {"kind": "voice", "attempted": True, "success": True}

    monkeypatch.setattr(autonomy, "_send_chatbox", fake_chatbox)
    monkeypatch.setattr(autonomy, "_speak_voicevox", fake_voice)
    monkeypatch.setattr(vrchat_smoke, "plan_autonomy_turn", autonomy.plan_autonomy_turn)

    result = vrchat_smoke.run_private_smoke(
        profile_path=profile_path,
        live=True,
        live_ack=autonomy.LIVE_ACTUATION_ACK,
    )

    assert result["success"] is True
    assert result["code"] == "LIVE_SMOKE_DONE"
    assert result["turn"]["dry_run"] is False
    assert result["safety"]["actuation_performed"] is True
    assert calls[0][0] == "chatbox"
    assert calls[1][0] == "voice"


def test_private_smoke_live_blocks_when_readiness_missing(monkeypatch, tmp_path):
    profile_path = _write_profile(
        tmp_path,
        dry_run=False,
        live_actuation_ack=autonomy.LIVE_ACTUATION_ACK,
    )
    monkeypatch.setattr(
        vrchat_smoke,
        "vrchat_autonomy_readiness",
        lambda **kwargs: {
            "ready": False,
            "missing": ["VRChat.exe"],
            "checks": {},
            "safety": {
                "actuation_performed": False,
                "chatbox_sent": False,
                "speech_played": False,
                "avatar_parameters_written": False,
            },
        },
    )

    result = vrchat_smoke.run_private_smoke(
        profile_path=profile_path,
        live=True,
        live_ack=autonomy.LIVE_ACTUATION_ACK,
    )

    assert result["success"] is True
    assert result["code"] == "DRY_RUN_SMOKE_DONE"
    assert "readiness_not_ready" in result["live_gate"]["blocked_reasons"]
    assert result["turn"]["dry_run"] is True


def test_wait_then_private_smoke_times_out_without_prepare(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(
        vrchat_smoke,
        "wait_for_readiness",
        lambda **kwargs: {
            "success": True,
            "status": "timeout",
            "ready": False,
            "safety": {
                "actuation_performed": False,
                "chatbox_sent": False,
                "speech_played": False,
                "avatar_parameters_written": False,
            },
        },
    )
    monkeypatch.setattr(vrchat_smoke, "prepare_private_smoke", lambda **kwargs: calls.append(kwargs))

    result = vrchat_smoke.wait_then_private_smoke(profile_path=tmp_path / "profile.json")

    assert result["success"] is True
    assert result["code"] == "WAIT_READY_TIMEOUT"
    assert result["prepare"] is None
    assert result["smoke"] is None
    assert result["safety"]["actuation_performed"] is False
    assert calls == []


def test_wait_then_private_smoke_stops_after_prepare_by_default(monkeypatch, tmp_path):
    monkeypatch.setattr(
        vrchat_smoke,
        "wait_for_readiness",
        lambda **kwargs: {
            "success": True,
            "status": "ready",
            "ready": True,
            "safety": {
                "actuation_performed": False,
                "chatbox_sent": False,
                "speech_played": False,
                "avatar_parameters_written": False,
            },
        },
    )
    monkeypatch.setattr(
        vrchat_smoke,
        "prepare_private_smoke",
        lambda **kwargs: {
            "success": True,
            "code": "PRIVATE_SMOKE_READY",
            "would_execute_live": True,
            "turn": {"dry_run": True},
            "safety": {
                "actuation_performed": False,
                "chatbox_sent": False,
                "speech_played": False,
                "avatar_parameters_written": False,
            },
        },
    )

    def fail_live(**kwargs):
        raise AssertionError("live smoke should not run without allow_live_smoke")

    monkeypatch.setattr(vrchat_smoke, "run_private_smoke", fail_live)

    result = vrchat_smoke.wait_then_private_smoke(
        profile_path=tmp_path / "profile.json",
        live_ack=autonomy.LIVE_ACTUATION_ACK,
    )

    assert result["success"] is True
    assert result["code"] == "WAIT_READY_PREPARED"
    assert result["prepare"]["would_execute_live"] is True
    assert result["smoke"] is None
    assert result["dry_run"] is True
    assert result["safety"]["actuation_performed"] is False


def test_wait_then_private_smoke_runs_live_only_with_explicit_gate(monkeypatch, tmp_path):
    monkeypatch.setattr(
        vrchat_smoke,
        "wait_for_readiness",
        lambda **kwargs: {
            "success": True,
            "status": "ready",
            "ready": True,
            "safety": {
                "actuation_performed": False,
                "chatbox_sent": False,
                "speech_played": False,
                "avatar_parameters_written": False,
            },
        },
    )
    monkeypatch.setattr(
        vrchat_smoke,
        "prepare_private_smoke",
        lambda **kwargs: {
            "success": True,
            "code": "PRIVATE_SMOKE_READY",
            "would_execute_live": True,
            "turn": {"dry_run": True},
            "safety": {
                "actuation_performed": False,
                "chatbox_sent": False,
                "speech_played": False,
                "avatar_parameters_written": False,
            },
        },
    )
    monkeypatch.setattr(
        vrchat_smoke,
        "run_private_smoke",
        lambda **kwargs: {
            "success": True,
            "code": "LIVE_SMOKE_DONE",
            "turn": {"dry_run": False},
            "safety": {
                "actuation_performed": True,
                "chatbox_sent": True,
                "speech_played": True,
                "avatar_parameters_written": False,
            },
        },
    )

    result = vrchat_smoke.wait_then_private_smoke(
        profile_path=tmp_path / "profile.json",
        allow_live_smoke=True,
        live_ack=autonomy.LIVE_ACTUATION_ACK,
    )

    assert result["success"] is True
    assert result["code"] == "WAIT_READY_LIVE_SMOKE_DONE"
    assert result["smoke"]["code"] == "LIVE_SMOKE_DONE"
    assert result["dry_run"] is False
    assert result["safety"]["actuation_performed"] is True
    assert result["safety"]["chatbox_sent"] is True
    assert result["safety"]["speech_played"] is True
