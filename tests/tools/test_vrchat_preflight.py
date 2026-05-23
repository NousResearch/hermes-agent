from __future__ import annotations

import json

import pytest

from tools.openclaw import vrchat_autonomy as autonomy
from tools.openclaw import vrchat_preflight


def test_preflight_bundle_is_read_only_and_reports_gate(monkeypatch, tmp_path):
    monkeypatch.setattr(autonomy, "is_vrchat_process_running", lambda: False)
    monkeypatch.setattr(autonomy, "python_osc_available", lambda: True)
    monkeypatch.setattr(autonomy, "probe_voicevox", lambda url: {"ok": False, "url": url})
    monkeypatch.setattr(autonomy, "probe_harness", lambda url: {"ok": False, "url": url})
    monkeypatch.setattr(autonomy, "find_output_device", lambda name: {"ok": None, "configured": False})
    monkeypatch.setattr(vrchat_preflight, "vrchat_autonomy_readiness", autonomy.vrchat_autonomy_readiness)
    monkeypatch.setattr(vrchat_preflight, "vrchat_autonomy_heartbeat", autonomy.vrchat_autonomy_heartbeat)
    monkeypatch.setattr(vrchat_preflight, "find_output_device", autonomy.find_output_device)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps({**autonomy._default_profile(), "enabled": True, "mode": "private_test"}),
        encoding="utf-8",
    )

    bundle = vrchat_preflight.build_preflight_bundle(
        profile_path=profile_path,
        include_audio_devices=False,
    )

    assert bundle["success"] is True
    assert bundle["readiness"]["ready"] is False
    assert "VRChat.exe" in bundle["readiness"]["missing"]
    assert bundle["heartbeat_preview"]["code"] == "HEARTBEAT_OK"
    assert bundle["live_smoke_gate"]["ready_for_live_private_smoke"] is False
    assert "readiness_not_ready" in bundle["live_smoke_gate"]["blocked_reasons"]
    assert bundle["safety"]["actuation_performed"] is False
    assert bundle["safety"]["microphone_recorded"] is False
    assert bundle["safety"]["websocket_opened"] is False
    assert bundle["audio"]["virtual_cable_route"]["configured"] is False
    assert bundle["audio"]["virtual_cable_route"]["safety"]["microphone_recorded"] is False
    assert bundle["voicevox_synthesis"]["included"] is False
    assert bundle["commands"]["prepare_dry_run_profile"][:5] == [
        "py",
        "-3.12",
        "scripts\\vrchat_profile.py",
        "--profile",
        str(profile_path),
    ]
    assert bundle["commands"]["print_live_ack"] == [
        "py",
        "-3.12",
        "scripts\\vrchat_profile.py",
        "--print-live-ack",
    ]
    assert "--arm-live" in bundle["commands"]["arm_live_profile_after_private_verification"]
    assert "--prepare-only" in bundle["commands"]["private_smoke_prepare_live_gate"]
    assert (
        "scripts\\vrchat_wait_then_private_smoke.py"
        in bundle["commands"]["wait_readiness_then_private_smoke_prepare"]
    )
    assert "--allow-live-smoke" not in bundle["commands"]["wait_readiness_then_private_smoke_prepare"]
    assert "--live" in bundle["commands"]["private_smoke_live_after_private_verification"]


def test_list_audio_output_devices_reports_output_channels(monkeypatch):
    fake_devices = [
        {"name": "Microphone", "max_output_channels": 0},
        {"name": "CABLE Input (VB-Audio Virtual Cable)", "max_output_channels": 2},
    ]

    class FakeDefault:
        device = [0, 1]

    class FakeSoundDevice:
        default = FakeDefault()

        @staticmethod
        def query_devices():
            return fake_devices

    monkeypatch.setitem(__import__("sys").modules, "sounddevice", FakeSoundDevice)

    result = vrchat_preflight.list_audio_output_devices()

    assert result["success"] is True
    assert result["devices"] == [
        {
            "index": 1,
            "name": "CABLE Input (VB-Audio Virtual Cable)",
            "max_output_channels": 2,
            "default_output": True,
        }
    ]


def test_virtual_cable_route_checks_playback_and_microphone_without_recording(monkeypatch):
    fake_devices = [
        {"name": "CABLE Output (VB-Audio Virtual Cable)", "max_input_channels": 2, "max_output_channels": 0},
        {"name": "CABLE Input (VB-Audio Virtual Cable)", "max_input_channels": 0, "max_output_channels": 2},
        {"name": "Default Speakers", "max_input_channels": 0, "max_output_channels": 2},
    ]

    class FakeSoundDevice:
        @staticmethod
        def query_devices():
            return fake_devices

    monkeypatch.setitem(__import__("sys").modules, "sounddevice", FakeSoundDevice)

    result = vrchat_preflight.check_virtual_cable_route(output_device="CABLE Input")

    assert result["success"] is True
    assert result["ok"] is True
    assert result["microphone_device"] == "CABLE Output"
    assert result["playback"]["matches"][0]["name"] == "CABLE Input (VB-Audio Virtual Cable)"
    assert result["microphone"]["matches"][0]["name"] == "CABLE Output (VB-Audio Virtual Cable)"
    assert result["safety"]["microphone_recorded"] is False
    assert result["safety"]["speech_played"] is False


def test_voicevox_synthesis_probe_does_not_play_or_record(monkeypatch):
    calls = []

    class FakeResponse:
        def __init__(self, *, payload=None, content=b""):
            self._payload = payload or {}
            self.content = content

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        if url.endswith("/audio_query"):
            return FakeResponse(payload={"query": True})
        return FakeResponse(content=b"RIFF\x24\x00\x00\x00WAVEfmt ")

    monkeypatch.setattr(vrchat_preflight.requests, "post", fake_post)

    result = vrchat_preflight.probe_voicevox_synthesis(
        voicevox_url="http://127.0.0.1:50021",
        text="test",
        speaker=8,
    )

    assert result["success"] is True
    assert result["ok"] is True
    assert result["wav_header_ok"] is True
    assert result["size_bytes"] > 0
    assert result["played_audio"] is False
    assert result["microphone_recorded"] is False
    assert result["safety"]["speech_played"] is False
    assert result["safety"]["microphone_recorded"] is False
    assert calls[0][0].endswith("/audio_query")
    assert calls[1][0].endswith("/synthesis")


def test_preflight_bundle_writes_output_file(monkeypatch, tmp_path):
    monkeypatch.setattr(autonomy, "is_vrchat_process_running", lambda: False)
    monkeypatch.setattr(autonomy, "python_osc_available", lambda: True)
    monkeypatch.setattr(autonomy, "probe_voicevox", lambda url: {"ok": False, "url": url})
    monkeypatch.setattr(autonomy, "probe_harness", lambda url: {"ok": False, "url": url})
    monkeypatch.setattr(autonomy, "find_output_device", lambda name: {"ok": None, "configured": False})
    monkeypatch.setattr(vrchat_preflight, "vrchat_autonomy_readiness", autonomy.vrchat_autonomy_readiness)
    monkeypatch.setattr(vrchat_preflight, "vrchat_autonomy_heartbeat", autonomy.vrchat_autonomy_heartbeat)
    monkeypatch.setattr(vrchat_preflight, "find_output_device", autonomy.find_output_device)
    output_path = tmp_path / "preflight.json"

    bundle = vrchat_preflight.build_preflight_bundle(
        include_audio_devices=False,
        output_path=output_path,
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["readiness"]["ready"] is False
    assert saved["output_path"] == str(output_path)
    assert bundle["output_path"] == str(output_path)


def test_wait_for_readiness_returns_ready_after_poll(monkeypatch):
    bundles = iter(
        [
            {
                "readiness": {
                    "ready": False,
                    "missing": ["VRChat.exe"],
                    "checks": {
                        "vrchat_process": {"phase": "not_detected", "diagnostic": "missing"},
                        "voicevox": {"ok": True, "process": {"phase": "engine_process_running"}},
                        "audio_output_device": {"ok": True},
                    },
                },
                "live_smoke_gate": {"ready_for_live_private_smoke": False, "blocked_reasons": ["readiness_not_ready"]},
            },
            {
                "readiness": {
                    "ready": True,
                    "missing": [],
                    "checks": {
                        "vrchat_process": {"phase": "running", "diagnostic": "ready"},
                        "voicevox": {"ok": True, "process": {"phase": "engine_process_running"}},
                        "audio_output_device": {"ok": True},
                    },
                },
                "live_smoke_gate": {"ready_for_live_private_smoke": False, "blocked_reasons": ["profile_dry_run_true"]},
            },
        ]
    )
    monkeypatch.setattr(vrchat_preflight, "build_preflight_bundle", lambda **kwargs: next(bundles))

    result = vrchat_preflight.wait_for_readiness(
        timeout_sec=5,
        interval_sec=1,
        _sleep=lambda seconds: None,
        _clock=iter([0, 0, 0, 0, 1, 1]).__next__,
    )

    assert result["status"] == "ready"
    assert result["ready"] is True
    assert result["attempts"] == 2
    assert result["final_summary"]["vrchat_process_phase"] == "running"
    assert result["safety"]["actuation_performed"] is False


def test_wait_for_readiness_times_out_without_actuation(monkeypatch):
    monkeypatch.setattr(
        vrchat_preflight,
        "build_preflight_bundle",
        lambda **kwargs: {
            "readiness": {
                "ready": False,
                "missing": ["VRChat.exe", "VOICEVOX Engine"],
                "checks": {
                    "vrchat_process": {"phase": "not_detected", "diagnostic": "missing"},
                    "voicevox": {"ok": False, "process": {"phase": "not_detected"}},
                    "audio_output_device": {"ok": True},
                },
            },
            "live_smoke_gate": {"ready_for_live_private_smoke": False, "blocked_reasons": ["readiness_not_ready"]},
        },
    )

    result = vrchat_preflight.wait_for_readiness(timeout_sec=0, _clock=lambda: 10.0)

    assert result["status"] == "timeout"
    assert result["ready"] is False
    assert result["attempts"] == 1
    assert result["final_summary"]["voicevox_process_phase"] == "not_detected"
    assert result["safety"]["speech_played"] is False


def test_runtime_doctor_reports_operator_mismatch_and_next_actions(monkeypatch):
    monkeypatch.setattr(
        vrchat_preflight,
        "build_preflight_bundle",
        lambda **kwargs: {
            "readiness": {
                "ready": False,
                "missing": ["VRChat.exe", "VOICEVOX Engine"],
                "checks": {
                    "vrchat_process": {
                        "ok": False,
                        "phase": "not_detected",
                        "diagnostic": "VRChat.exe was not detected.",
                    },
                    "voicevox": {
                        "ok": False,
                        "url": "http://127.0.0.1:50021",
                        "process": {"phase": "not_detected"},
                    },
                    "harness": {"ok": False},
                    "audio_output_device": {"ok": True, "configured": True},
                },
            },
            "profile": {"success": True, "path": "profile.json", "profile": {"dry_run": True}},
            "live_smoke_gate": {"ready_for_live_private_smoke": False, "blocked_reasons": ["readiness_not_ready"]},
            "commands": {},
        },
    )
    monkeypatch.setattr(
        vrchat_preflight,
        "_probe_voicevox_candidates",
        lambda primary_url, probe_timeout: [
            {"ok": False, "url": primary_url, "configured_primary": True},
            {"ok": True, "url": "http://127.0.0.1:50031", "configured_primary": False},
        ],
    )
    monkeypatch.setattr(
        vrchat_preflight,
        "_local_port_snapshot",
        lambda ports: {"success": True, "ports_checked": ports, "listeners": []},
    )
    monkeypatch.setattr(
        vrchat_preflight,
        "_relevant_window_snapshot",
        lambda: {
            "success": True,
            "included": True,
            "windows": [
                {"title": "VRChat", "pid": 111, "process": "VRChat.exe", "matched_terms": ["vrchat"]},
                {"title": "VOICEVOX", "pid": 222, "process": "VOICEVOX.exe", "matched_terms": ["voicevox"]},
            ],
        },
    )
    monkeypatch.setattr(
        vrchat_preflight,
        "_runtime_launch_discovery",
        lambda: {
            "success": True,
            "included": True,
            "read_only": True,
            "vrchat": {
                "candidate_count": 1,
                "launch_uri": "steam://rungameid/438100",
                "installs": [
                    {
                        "manifest_found": True,
                        "exe_found": True,
                        "exe": r"C:\Steam\steamapps\common\VRChat\VRChat.exe",
                    }
                ],
            },
            "voicevox": {
                "candidate_count": 1,
                "executables": [{"path": r"C:\Users\me\AppData\Local\Programs\VOICEVOX\VOICEVOX.exe"}],
                "shortcuts": [],
            },
        },
    )
    monkeypatch.setattr(
        vrchat_preflight,
        "_process_visibility_snapshot",
        lambda: {
            "success": True,
            "read_only": True,
            "current": {"pid": 10, "session_id": 1, "is_admin": False},
            "inspected_processes": 25,
            "access_denied": 0,
            "relevant_processes": [],
            "relevant_count": 0,
        },
    )

    result = vrchat_preflight.build_runtime_doctor(
        operator_reported_vrchat=True,
        operator_reported_voicevox=True,
    )

    assert result["success"] is True
    assert result["summary"]["status"] == "operator_mismatch"
    assert "operator_reported_vrchat_but_process_not_visible" in result["summary"]["operator_mismatches"]
    assert "operator_reported_voicevox_but_engine_not_reachable" in result["summary"]["operator_mismatches"]
    assert "voicevox_reachable_on_nonconfigured_url" in result["summary"]["operator_mismatches"]
    assert "vrchat_window_visible_but_expected_process_not_ready" in result["summary"]["operator_mismatches"]
    assert "voicevox_window_visible_but_engine_not_reachable" in result["summary"]["operator_mismatches"]
    assert result["summary"]["voicevox_candidate_urls_ok"] == ["http://127.0.0.1:50031"]
    assert result["summary"]["relevant_window_titles"] == ["VRChat", "VOICEVOX"]
    assert result["summary"]["vrchat_launch_candidates"] == 1
    assert result["summary"]["voicevox_launch_candidates"] == 1
    assert result["summary"]["process_visibility_relevant_count"] == 0
    assert result["summary"]["current_session_id"] == 1
    assert len(result["desktop_windows"]["windows"]) == 2
    assert result["launch_discovery"]["read_only"] is True
    assert result["process_visibility"]["read_only"] is True
    assert any("Set the profile VOICEVOX URL" in action for action in result["next_actions"])
    assert any("steam://rungameid/438100" in action for action in result["next_actions"])
    assert any("Windows session 1" in action for action in result["next_actions"])
    assert result["safety"]["actuation_performed"] is False
    assert result["safety"]["websocket_opened"] is False


def test_runtime_doctor_writes_output_file(monkeypatch, tmp_path):
    monkeypatch.setattr(
        vrchat_preflight,
        "build_preflight_bundle",
        lambda **kwargs: {
            "readiness": {
                "ready": True,
                "missing": [],
                "checks": {
                    "vrchat_process": {"ok": True, "phase": "running"},
                    "voicevox": {"ok": True, "url": "http://127.0.0.1:50021"},
                    "harness": {"ok": False},
                    "audio_output_device": {"ok": True, "configured": True},
                },
            },
            "profile": {"success": True, "path": "profile.json", "profile": {"dry_run": True}},
            "live_smoke_gate": {"ready_for_live_private_smoke": False, "blocked_reasons": ["profile_dry_run_true"]},
            "commands": {},
        },
    )
    monkeypatch.setattr(
        vrchat_preflight,
        "_probe_voicevox_candidates",
        lambda primary_url, probe_timeout: [{"ok": True, "url": primary_url, "configured_primary": True}],
    )
    monkeypatch.setattr(
        vrchat_preflight,
        "_local_port_snapshot",
        lambda ports: {"success": True, "ports_checked": ports, "listeners": []},
    )
    monkeypatch.setattr(
        vrchat_preflight,
        "_relevant_window_snapshot",
        lambda: {"success": True, "included": True, "windows": []},
    )
    monkeypatch.setattr(
        vrchat_preflight,
        "_runtime_launch_discovery",
        lambda: {
            "success": True,
            "included": True,
            "read_only": True,
            "vrchat": {"candidate_count": 0, "installs": []},
            "voicevox": {"candidate_count": 0, "executables": [], "shortcuts": []},
        },
    )
    monkeypatch.setattr(
        vrchat_preflight,
        "_process_visibility_snapshot",
        lambda: {
            "success": True,
            "read_only": True,
            "current": {"pid": 10, "session_id": 1, "is_admin": False},
            "inspected_processes": 25,
            "access_denied": 0,
            "relevant_processes": [{"pid": 99, "name": "VRChat.exe", "matched_terms": ["vrchat"]}],
            "relevant_count": 1,
        },
    )
    output_path = tmp_path / "doctor.json"

    result = vrchat_preflight.build_runtime_doctor(output_path=output_path)

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["output_path"] == str(output_path)
    assert result["output_path"] == str(output_path)
    assert saved["summary"]["ready"] is True
    assert saved["safety"]["speech_played"] is False
    assert saved["summary"]["process_visibility_relevant_count"] == 1


def test_voicevox_launch_candidates_are_bounded(tmp_path):
    root = tmp_path / "VOICEVOX"
    nested = root / "current"
    too_deep = root / "a" / "b" / "c"
    nested.mkdir(parents=True)
    too_deep.mkdir(parents=True)
    exe = nested / "VOICEVOX.exe"
    exe.write_text("", encoding="utf-8")
    (too_deep / "VOICEVOX.exe").write_text("", encoding="utf-8")

    result = vrchat_preflight._voicevox_launch_candidates_from_roots([root], max_depth=1)

    assert result == [{"path": str(exe), "source_root": str(root), "kind": "executable"}]


def test_process_visibility_snapshot_does_not_store_command_lines(monkeypatch):
    class FakeProcess:
        info = {
            "pid": 42,
            "ppid": 1,
            "name": "run.exe",
            "exe": r"C:\VOICEVOX\vv-engine\run.exe",
            "cmdline": ["run.exe", "--token", "secret-like-value", "voicevox"],
            "username": r"DESKTOP\operator",
            "status": "running",
        }

    monkeypatch.setattr(vrchat_preflight.os, "getpid", lambda: 100)
    monkeypatch.setattr(vrchat_preflight.os, "getppid", lambda: 50)
    monkeypatch.setattr(vrchat_preflight.psutil, "process_iter", lambda attrs: [FakeProcess()])
    monkeypatch.setattr(vrchat_preflight, "_process_session_id", lambda pid: 7)
    monkeypatch.setattr(vrchat_preflight, "_current_username", lambda: r"DESKTOP\operator")
    monkeypatch.setattr(vrchat_preflight, "_is_current_process_admin", lambda: False)

    result = vrchat_preflight._process_visibility_snapshot()

    assert result["current"]["session_id"] == 7
    assert result["relevant_count"] == 1
    assert result["relevant_processes"][0]["matched_terms"] == ["voicevox", "vv-engine"]
    assert "cmdline" not in result["relevant_processes"][0]


def test_process_visibility_snapshot_ignores_current_command_hosts(monkeypatch):
    class FakeProcess:
        info = {
            "pid": 42,
            "ppid": 1,
            "name": "python.exe",
            "exe": r"C:\Python\python.exe",
            "cmdline": ["python.exe", r"scripts\vrchat_runtime_doctor.py"],
            "username": r"DESKTOP\operator",
            "status": "running",
        }

    monkeypatch.setattr(vrchat_preflight.os, "getpid", lambda: 100)
    monkeypatch.setattr(vrchat_preflight.os, "getppid", lambda: 50)
    monkeypatch.setattr(vrchat_preflight.psutil, "process_iter", lambda attrs: [FakeProcess()])
    monkeypatch.setattr(vrchat_preflight, "_process_session_id", lambda pid: 7)
    monkeypatch.setattr(vrchat_preflight, "_current_username", lambda: r"DESKTOP\operator")
    monkeypatch.setattr(vrchat_preflight, "_is_current_process_admin", lambda: False)
    monkeypatch.setattr(vrchat_preflight, "_current_process_tree_pids", lambda: {100, 50})

    result = vrchat_preflight._process_visibility_snapshot()

    assert result["relevant_count"] == 0
    assert result["relevant_processes"] == []


def test_wait_for_readiness_then_tick_skips_tick_on_timeout(monkeypatch):
    monkeypatch.setattr(
        vrchat_preflight,
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
                "microphone_recorded": False,
                "websocket_opened": False,
            },
        },
    )
    monkeypatch.setattr(
        vrchat_preflight,
        "vrchat_autonomy_heartbeat_tick",
        lambda **kwargs: pytest.fail("heartbeat tick should not run when readiness timed out"),
    )

    result = vrchat_preflight.wait_for_readiness_then_tick()

    assert result["success"] is True
    assert result["code"] == "WAIT_READY_TIMEOUT"
    assert result["tick"] is None
    assert result["safety"]["actuation_performed"] is False


def test_wait_for_readiness_then_tick_runs_gated_tick_when_ready(monkeypatch, tmp_path):
    profile_path = tmp_path / "profile.json"
    calls = []
    monkeypatch.setattr(
        vrchat_preflight,
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
                "microphone_recorded": False,
                "websocket_opened": False,
            },
        },
    )

    def fake_heartbeat_tick(**kwargs):
        calls.append(kwargs)
        return {
            "success": True,
            "code": "HEARTBEAT_TICK_DONE",
            "dry_run": True,
            "safety": autonomy._safety_flags(),
        }

    monkeypatch.setattr(vrchat_preflight, "vrchat_autonomy_heartbeat_tick", fake_heartbeat_tick)

    result = vrchat_preflight.wait_for_readiness_then_tick(
        profile_path=profile_path,
        observations=[{"source": "operator", "text": "ready"}],
        allow_live_profile=True,
        live_ack=autonomy.LIVE_ACTUATION_ACK,
    )

    assert result["success"] is True
    assert result["code"] == "WAIT_READY_TICK_DONE"
    assert result["dry_run"] is True
    assert calls[0]["profile_path"] == profile_path
    assert calls[0]["tick_when_already_ready"] is True
    assert calls[0]["allow_live_profile"] is True
    assert calls[0]["live_ack"] == autonomy.LIVE_ACTUATION_ACK
    assert calls[0]["observations"][0]["text"] == "ready"
