from __future__ import annotations

import json

from tools.openclaw import vrchat_autonomy as autonomy


class _FakeProcess:
    def __init__(self, **info):
        self.info = info


def test_readiness_is_read_only_and_reports_missing_voicevox(monkeypatch):
    monkeypatch.setattr(autonomy, "is_vrchat_process_running", lambda: True)
    monkeypatch.setattr(autonomy, "python_osc_available", lambda: True)
    monkeypatch.setattr(
        autonomy,
        "probe_voicevox",
        lambda url: {"ok": False, "url": url, "error": "ConnectionError"},
    )
    monkeypatch.setattr(autonomy, "probe_harness", lambda url: {"ok": False, "url": url})
    monkeypatch.setattr(autonomy, "find_output_device", lambda name: {"ok": None, "configured": False})

    status = autonomy.vrchat_autonomy_readiness(require_voice=True)

    assert status["ready"] is False
    assert "VOICEVOX Engine" in status["missing"]
    assert status["safety"] == {
        "actuation_performed": False,
        "chatbox_sent": False,
        "speech_played": False,
        "avatar_parameters_written": False,
    }
    assert status["checks"]["vrchat_process"]["phase"] == "running"


def test_launch_state_reports_vrchat_process(monkeypatch):
    monkeypatch.setattr(
        autonomy.psutil,
        "process_iter",
        lambda attrs: iter(
            [
                _FakeProcess(
                    pid=123,
                    name="VRChat.exe",
                    exe=r"C:\Steam\steamapps\common\VRChat\VRChat.exe",
                    cmdline=[],
                )
            ]
        ),
    )

    state = autonomy.inspect_vrchat_launch_state(exclude_pids=set())

    assert state["ok"] is True
    assert state["phase"] == "running"
    assert state["matched_processes"][0]["name"] == "VRChat.exe"


def test_launch_state_reports_launcher_clues_without_ready(monkeypatch):
    monkeypatch.setattr(
        autonomy.psutil,
        "process_iter",
        lambda attrs: iter(
            [
                _FakeProcess(
                    pid=321,
                    name="start_protected_game.exe",
                    exe=r"C:\Steam\steamapps\common\VRChat\start_protected_game.exe",
                    cmdline=[],
                ),
                _FakeProcess(pid=654, name="steam.exe", exe=r"C:\Program Files (x86)\Steam\steam.exe", cmdline=[]),
            ]
        ),
    )

    state = autonomy.inspect_vrchat_launch_state(exclude_pids=set())

    assert state["ok"] is False
    assert state["phase"] == "launching_or_blocked"
    assert state["steam"]["running"] is True
    assert state["launch_clues"][0]["name"] == "start_protected_game.exe"


def test_voicevox_runtime_state_reports_ui_without_engine(monkeypatch):
    monkeypatch.setattr(
        autonomy.psutil,
        "process_iter",
        lambda attrs: iter(
            [
                _FakeProcess(
                    pid=777,
                    name="VOICEVOX.exe",
                    exe=r"C:\Users\downl\AppData\Local\Programs\VOICEVOX\VOICEVOX.exe",
                    cmdline=[],
                )
            ]
        ),
    )

    state = autonomy.inspect_voicevox_runtime_state(exclude_pids=set())

    assert state["phase"] == "ui_running_no_engine"
    assert state["ui_running"] is True
    assert state["engine_process_running"] is False


def test_voicevox_runtime_state_reports_engine(monkeypatch):
    monkeypatch.setattr(
        autonomy.psutil,
        "process_iter",
        lambda attrs: iter(
            [
                _FakeProcess(
                    pid=888,
                    name="run.exe",
                    exe=r"C:\Users\downl\AppData\Local\Programs\VOICEVOX\vv-engine\run.exe",
                    cmdline=["run.exe", "--port", "50021"],
                )
            ]
        ),
    )

    state = autonomy.inspect_voicevox_runtime_state(exclude_pids=set())

    assert state["phase"] == "engine_process_running"
    assert state["engine_processes"][0]["name"] == "run.exe"


def test_readiness_requires_harness_only_when_requested(monkeypatch):
    monkeypatch.setattr(autonomy, "is_vrchat_process_running", lambda: True)
    monkeypatch.setattr(autonomy, "python_osc_available", lambda: True)
    monkeypatch.setattr(autonomy, "probe_voicevox", lambda url: {"ok": True, "url": url})
    monkeypatch.setattr(autonomy, "probe_harness", lambda url: {"ok": False, "url": url})
    monkeypatch.setattr(autonomy, "find_output_device", lambda name: {"ok": None, "configured": False})

    assert autonomy.vrchat_autonomy_readiness()["ready"] is True

    required = autonomy.vrchat_autonomy_readiness(require_harness=True)
    assert required["ready"] is False
    assert "Hypura harness" in required["missing"]


def test_heartbeat_notifies_on_vrchat_launch_and_then_quiets(monkeypatch, tmp_path):
    monkeypatch.setattr(autonomy, "is_vrchat_process_running", lambda: True)
    monkeypatch.setattr(autonomy, "python_osc_available", lambda: True)
    monkeypatch.setattr(autonomy, "probe_voicevox", lambda url: {"ok": True, "url": url})
    monkeypatch.setattr(autonomy, "probe_harness", lambda url: {"ok": False, "url": url})
    monkeypatch.setattr(autonomy, "find_output_device", lambda name: {"ok": None, "configured": False})
    state_path = tmp_path / "heartbeat.json"

    first = autonomy.vrchat_autonomy_heartbeat(state_path=state_path)
    second = autonomy.vrchat_autonomy_heartbeat(state_path=state_path)

    assert first["notify"] is True
    assert first["code"] == "VRCHAT_LAUNCHED_READY"
    assert second["notify"] is False
    assert second["code"] == "HEARTBEAT_OK"
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["current"]["vrchat_running"] is True


def test_heartbeat_suppresses_notification_when_vrchat_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(autonomy, "is_vrchat_process_running", lambda: False)
    monkeypatch.setattr(autonomy, "python_osc_available", lambda: True)
    monkeypatch.setattr(autonomy, "probe_voicevox", lambda url: {"ok": True, "url": url})
    monkeypatch.setattr(autonomy, "probe_harness", lambda url: {"ok": False, "url": url})
    monkeypatch.setattr(autonomy, "find_output_device", lambda name: {"ok": None, "configured": False})

    result = autonomy.vrchat_autonomy_heartbeat(state_path=tmp_path / "heartbeat.json")

    assert result["notify"] is False
    assert result["code"] == "HEARTBEAT_OK"
    assert result["current"]["vrchat_running"] is False


def test_validate_decision_blocks_raw_osc():
    result = autonomy.validate_agent_decision(
        {"address": "/avatar/parameters/Smile", "args": [True]},
        mode="private_test",
    )

    assert result["success"] is False
    assert any(reason.startswith("raw_osc_not_allowed") for reason in result["blocked_reasons"])


def test_build_decision_request_contains_schema_and_observation_context():
    result = autonomy.build_decision_request(
        observations=[
            {"source": "speechToText", "text": "hello"},
            {"source": "visionObservation", "summary": "User is waving."},
        ],
        mode="private_test",
        allowed_avatar_actions=["wave"],
        avatar_action_descriptions={"wave": "Wave once."},
        allow_voice=True,
        allow_chatbox=True,
        persona="Calm companion.",
    )

    assert result["success"] is True
    request = result["request"]
    assert request["response_format"]["type"] == "json_schema"
    schema = request["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["avatar_action"]["enum"] == ["", "wave"]
    user_payload = json.loads(request["messages"][1]["content"])
    assert user_payload["capabilities"]["mode"] == "private_test"
    assert user_payload["observations"][1]["source"] == "visionObservation"
    assert "User is waving." in user_payload["context"]


def test_parse_agent_decision_text_accepts_fenced_json():
    result = autonomy.parse_agent_decision_text(
        '```json\n{"speak_text": "hi", "chatbox_text": "", "emotion": "happy", '
        '"avatar_action": "", "urgency": "low"}\n```'
    )

    assert result["success"] is True
    assert result["decision"]["speak_text"] == "hi"


def test_run_autonomy_decision_turn_calls_llm_and_returns_dry_run_plan():
    captured = {}

    def fake_llm(request, **kwargs):
        captured["request"] = request
        captured["kwargs"] = kwargs
        return json.dumps(
            {
                "speak_text": "Thanks for waving.",
                "chatbox_text": "Thanks.",
                "emotion": "happy",
                "avatar_action": "wave",
                "urgency": "low",
            }
        )

    result = autonomy.run_autonomy_decision_turn(
        observations=[{"source": "visionObservation", "summary": "The user is waving."}],
        mode="private_test",
        avatar_action_profiles={"wave": [{"name": "Wave", "value": True}]},
        avatar_action_descriptions={"wave": "Wave once."},
        allow_voice=True,
        allow_chatbox=True,
        dry_run=True,
        output_device="CABLE Input",
        llm_call=fake_llm,
    )

    assert result["success"] is True
    assert result["stage"] == "turn_planned"
    assert result["dry_run"] is True
    assert captured["request"]["response_format"]["type"] == "json_schema"
    assert captured["kwargs"]["provider"] is None
    assert result["model_decision"]["decision"]["avatar_action"] == "wave"
    assert [action["kind"] for action in result["turn"]["planned_actions"]] == [
        "chatbox",
        "voice",
        "avatar_action",
    ]
    assert result["safety"]["actuation_performed"] is False


def test_run_autonomy_decision_turn_blocks_invalid_model_json():
    result = autonomy.run_autonomy_decision_turn(
        observations=[{"source": "operator", "text": "private test"}],
        mode="private_test",
        allow_voice=True,
        llm_call=lambda request, **kwargs: "not json",
    )

    assert result["success"] is False
    assert result["stage"] == "decision_parse"
    assert result["model_decision"]["error"] == "invalid_json"
    assert result["safety"]["actuation_performed"] is False


def test_enqueue_observation_persists_normalized_jsonl(tmp_path):
    queue_path = tmp_path / "observations.jsonl"

    result = autonomy.enqueue_observation(
        {"source": "visionObservation", "summary": "The user is waving."},
        queue_path=queue_path,
    )

    assert result["success"] is True
    saved = [json.loads(line) for line in queue_path.read_text(encoding="utf-8").splitlines()]
    assert saved[0]["source"] == "visionObservation"
    assert saved[0]["text"] == "The user is waving."
    assert "received_at" in saved[0]


def test_enqueue_observation_rejects_unknown_source(tmp_path):
    result = autonomy.enqueue_observation(
        {"source": "unknown", "text": "ignored"},
        queue_path=tmp_path / "observations.jsonl",
    )

    assert result["success"] is False
    assert result["queued"] is False
    assert result["rejected"] == [{"index": "0", "reason": "unsupported_source:unknown"}]


def test_loop_tick_disabled_never_reads_model(tmp_path):
    called = False

    def fake_llm(request, **kwargs):
        nonlocal called
        called = True
        return "{}"

    result = autonomy.vrchat_autonomy_loop_tick(
        enabled=False,
        observations=[{"source": "operator", "text": "hello"}],
        loop_state_path=tmp_path / "loop.json",
        llm_call=fake_llm,
    )

    assert result["success"] is True
    assert result["code"] == "LOOP_DISABLED"
    assert called is False


def test_loop_tick_blocks_when_readiness_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(autonomy, "is_vrchat_process_running", lambda: False)
    monkeypatch.setattr(autonomy, "python_osc_available", lambda: True)
    monkeypatch.setattr(autonomy, "probe_voicevox", lambda url: {"ok": True, "url": url})
    monkeypatch.setattr(autonomy, "probe_harness", lambda url: {"ok": True, "url": url})
    monkeypatch.setattr(autonomy, "find_output_device", lambda name: {"ok": None, "configured": False})

    result = autonomy.vrchat_autonomy_loop_tick(
        enabled=True,
        observations=[{"source": "operator", "text": "hello"}],
        loop_state_path=tmp_path / "loop.json",
        llm_call=lambda request, **kwargs: "{}",
    )

    assert result["success"] is False
    assert result["code"] == "READINESS_BLOCKED"
    assert "VRChat.exe" in result["readiness"]["missing"]


def test_loop_tick_consumes_queue_and_runs_dry_turn(monkeypatch, tmp_path):
    monkeypatch.setattr(autonomy, "is_vrchat_process_running", lambda: True)
    monkeypatch.setattr(autonomy, "python_osc_available", lambda: True)
    monkeypatch.setattr(autonomy, "probe_voicevox", lambda url: {"ok": True, "url": url})
    monkeypatch.setattr(autonomy, "probe_harness", lambda url: {"ok": True, "url": url})
    monkeypatch.setattr(autonomy, "find_output_device", lambda name: {"ok": None, "configured": False})

    queue_path = tmp_path / "observations.jsonl"
    state_path = tmp_path / "loop.json"
    autonomy.enqueue_observation(
        {"source": "visionObservation", "summary": "The user is waving."},
        queue_path=queue_path,
    )

    result = autonomy.vrchat_autonomy_loop_tick(
        enabled=True,
        queue_path=queue_path,
        loop_state_path=state_path,
        mode="private_test",
        avatar_action_profiles={"wave": [{"name": "Wave", "value": True}]},
        allow_voice=True,
        allow_chatbox=True,
        dry_run=True,
        llm_call=lambda request, **kwargs: json.dumps(
            {
                "speak_text": "Hello.",
                "chatbox_text": "Hello.",
                "emotion": "happy",
                "avatar_action": "wave",
                "urgency": "low",
            }
        ),
    )

    assert result["success"] is True
    assert result["code"] == "TURN_DONE"
    assert result["queue"]["read"] == 1
    assert result["turn"]["stage"] == "turn_planned"
    assert result["turn"]["safety"]["actuation_performed"] is False
    assert not queue_path.exists()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["last_tick_code"] == "TURN_DONE"
    assert state["last_observation_count"] == 1


def test_loop_tick_rate_limits_after_turn(monkeypatch, tmp_path):
    monkeypatch.setattr(autonomy, "is_vrchat_process_running", lambda: True)
    monkeypatch.setattr(autonomy, "python_osc_available", lambda: True)
    monkeypatch.setattr(autonomy, "probe_voicevox", lambda url: {"ok": True, "url": url})
    monkeypatch.setattr(autonomy, "probe_harness", lambda url: {"ok": True, "url": url})
    monkeypatch.setattr(autonomy, "find_output_device", lambda name: {"ok": None, "configured": False})

    state_path = tmp_path / "loop.json"
    first = autonomy.vrchat_autonomy_loop_tick(
        enabled=True,
        observations=[{"source": "operator", "text": "private test"}],
        loop_state_path=state_path,
        mode="private_test",
        allow_chatbox=True,
        min_turn_interval_sec=0,
        llm_call=lambda request, **kwargs: json.dumps(
            {
                "speak_text": "",
                "chatbox_text": "ok",
                "emotion": "neutral",
                "avatar_action": "",
                "urgency": "low",
            }
        ),
    )

    called = False

    def blocked_llm(request, **kwargs):
        nonlocal called
        called = True
        return "{}"

    second = autonomy.vrchat_autonomy_loop_tick(
        enabled=True,
        observations=[{"source": "operator", "text": "private test"}],
        loop_state_path=state_path,
        min_turn_interval_sec=999,
        llm_call=blocked_llm,
    )

    assert first["code"] == "TURN_DONE"
    assert second["code"] == "RATE_LIMITED"
    assert called is False


def test_loop_tick_preserves_queue_when_llm_call_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(autonomy, "is_vrchat_process_running", lambda: True)
    monkeypatch.setattr(autonomy, "python_osc_available", lambda: True)
    monkeypatch.setattr(autonomy, "probe_voicevox", lambda url: {"ok": True, "url": url})
    monkeypatch.setattr(autonomy, "probe_harness", lambda url: {"ok": True, "url": url})
    monkeypatch.setattr(autonomy, "find_output_device", lambda name: {"ok": None, "configured": False})

    queue_path = tmp_path / "observations.jsonl"
    autonomy.enqueue_observation(
        {"source": "speechToText", "text": "hello"},
        queue_path=queue_path,
    )

    def failing_llm(request, **kwargs):
        raise RuntimeError("temporary model outage")

    result = autonomy.vrchat_autonomy_loop_tick(
        enabled=True,
        queue_path=queue_path,
        loop_state_path=tmp_path / "loop.json",
        mode="private_test",
        allow_chatbox=True,
        llm_call=failing_llm,
    )

    assert result["success"] is False
    assert result["code"] == "TURN_BLOCKED"
    assert result["turn"]["stage"] == "llm_call"
    assert result["queue"]["consumed"] is False
    assert queue_path.exists()
    assert "hello" in queue_path.read_text(encoding="utf-8")


def test_loop_tick_emergency_stop_persists_disabled_state(tmp_path):
    result = autonomy.vrchat_autonomy_loop_tick(
        enabled=True,
        emergency_stop=True,
        loop_state_path=tmp_path / "loop.json",
    )

    assert result["success"] is True
    assert result["code"] == "EMERGENCY_STOPPED"
    assert result["enabled"] is False
    assert result["state"]["enabled"] is False


def test_load_autonomy_profile_defaults_to_missing_disabled(tmp_path):
    result = autonomy.load_autonomy_profile(tmp_path / "missing.json")

    assert result["success"] is False
    assert result["exists"] is False
    assert result["errors"] == ["profile_missing"]
    assert result["profile"]["enabled"] is False
    assert result["profile"]["dry_run"] is True


def test_validate_autonomy_profile_requires_live_ack_when_not_dry_run():
    profile = {
        **autonomy._default_profile(),
        "enabled": True,
        "mode": "private_test",
        "dry_run": False,
        "allow_chatbox": True,
    }

    result = autonomy.validate_autonomy_profile(profile)

    assert result["success"] is False
    assert "live_actuation_ack_required" in result["errors"]


def test_validate_avatar_action_profiles_rejects_raw_osc_and_strings():
    result = autonomy.validate_avatar_action_profiles(
        {
            "wave": [
                {"name": "/avatar/parameters/Wave", "value": True},
                {"name": "Mood", "value": "happy"},
                {"name": "Smile", "value": 1.0, "reset_after_sec": 99},
            ]
        }
    )

    assert result["success"] is False
    assert "avatar_parameter_name_not_allowed:wave:/avatar/parameters/Wave" in result["errors"]
    assert "avatar_parameter_value_type_not_allowed:wave:1" in result["errors"]
    assert "reset_after_sec_out_of_range:wave:2" in result["errors"]


def test_profile_tick_disabled_profile_never_reads_model(tmp_path):
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps({**autonomy._default_profile(), "enabled": False}),
        encoding="utf-8",
    )
    called = False

    def fake_llm(request, **kwargs):
        nonlocal called
        called = True
        return "{}"

    result = autonomy.vrchat_autonomy_profile_tick(
        profile_path=profile_path,
        observations=[{"source": "operator", "text": "hello"}],
        llm_call=fake_llm,
    )

    assert result["success"] is True
    assert result["tick"]["code"] == "LOOP_DISABLED"
    assert called is False


def test_profile_tick_enabled_dry_run_uses_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(autonomy, "is_vrchat_process_running", lambda: True)
    monkeypatch.setattr(autonomy, "python_osc_available", lambda: True)
    monkeypatch.setattr(autonomy, "probe_voicevox", lambda url: {"ok": True, "url": url})
    monkeypatch.setattr(autonomy, "probe_harness", lambda url: {"ok": True, "url": url})
    monkeypatch.setattr(autonomy, "find_output_device", lambda name: {"ok": None, "configured": False})
    profile_path = tmp_path / "profile.json"
    profile = {
        **autonomy._default_profile(),
        "enabled": True,
        "mode": "private_test",
        "allow_voice": True,
        "allow_chatbox": True,
        "dry_run": True,
        "allowed_avatar_actions": ["wave"],
        "avatar_action_profiles": {
            "wave": [{"name": "Wave", "value": True, "reset_after_sec": 0.1}]
        },
    }
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    result = autonomy.vrchat_autonomy_profile_tick(
        profile_path=profile_path,
        observations=[{"source": "visionObservation", "summary": "The user is waving."}],
        llm_call=lambda request, **kwargs: json.dumps(
            {
                "speak_text": "Hello.",
                "chatbox_text": "Hello.",
                "emotion": "happy",
                "avatar_action": "wave",
                "urgency": "low",
            }
        ),
    )

    assert result["success"] is True
    assert result["code"] == "TURN_DONE"
    assert result["profile"]["profile"]["enabled"] is True
    assert result["tick"]["turn"]["stage"] == "turn_planned"
    assert result["safety"]["actuation_performed"] is False


def test_heartbeat_tick_runs_profile_on_ready_launch_event(monkeypatch, tmp_path):
    monkeypatch.setattr(autonomy, "is_vrchat_process_running", lambda: True)
    monkeypatch.setattr(autonomy, "python_osc_available", lambda: True)
    monkeypatch.setattr(autonomy, "probe_voicevox", lambda url: {"ok": True, "url": url})
    monkeypatch.setattr(autonomy, "probe_harness", lambda url: {"ok": True, "url": url})
    monkeypatch.setattr(autonomy, "find_output_device", lambda name: {"ok": None, "configured": False})
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps({**autonomy._default_profile(), "enabled": True, "mode": "private_test"}),
        encoding="utf-8",
    )
    calls = []

    def fake_profile_tick(**kwargs):
        calls.append(kwargs)
        return {
            "success": True,
            "code": "TURN_DONE",
            "tick": {"code": "TURN_DONE"},
            "safety": autonomy._safety_flags(),
        }

    monkeypatch.setattr(autonomy, "vrchat_autonomy_profile_tick", fake_profile_tick)

    result = autonomy.vrchat_autonomy_heartbeat_tick(
        profile_path=profile_path,
        observations=[{"source": "operator", "text": "ready"}],
        persist_heartbeat=False,
    )

    assert result["success"] is True
    assert result["code"] == "HEARTBEAT_TICK_DONE"
    assert result["heartbeat"]["code"] == "VRCHAT_LAUNCHED_READY"
    assert result["tick_reason"] == "VRCHAT_LAUNCHED_READY"
    assert calls[0]["profile_path"] == profile_path
    assert calls[0]["observations"][0]["text"] == "ready"


def test_heartbeat_tick_blocks_live_profile_without_command_ack(monkeypatch, tmp_path):
    monkeypatch.setattr(autonomy, "is_vrchat_process_running", lambda: True)
    monkeypatch.setattr(autonomy, "python_osc_available", lambda: True)
    monkeypatch.setattr(autonomy, "probe_voicevox", lambda url: {"ok": True, "url": url})
    monkeypatch.setattr(autonomy, "probe_harness", lambda url: {"ok": True, "url": url})
    monkeypatch.setattr(autonomy, "find_output_device", lambda name: {"ok": None, "configured": False})
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                **autonomy._default_profile(),
                "enabled": True,
                "mode": "private_test",
                "dry_run": False,
                "live_actuation_ack": autonomy.LIVE_ACTUATION_ACK,
            }
        ),
        encoding="utf-8",
    )

    result = autonomy.vrchat_autonomy_heartbeat_tick(
        profile_path=profile_path,
        persist_heartbeat=False,
    )

    assert result["success"] is False
    assert result["code"] == "HEARTBEAT_PROFILE_BLOCKED"
    assert "allow_live_profile_required" in result["live_gate"]["blocked_reasons"]
    assert "live_ack_required" in result["live_gate"]["blocked_reasons"]
    assert result["tick"] is None


def test_heartbeat_tick_quiets_when_vrchat_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(autonomy, "is_vrchat_process_running", lambda: False)
    monkeypatch.setattr(autonomy, "python_osc_available", lambda: True)
    monkeypatch.setattr(autonomy, "probe_voicevox", lambda url: {"ok": True, "url": url})
    monkeypatch.setattr(autonomy, "probe_harness", lambda url: {"ok": True, "url": url})
    monkeypatch.setattr(autonomy, "find_output_device", lambda name: {"ok": None, "configured": False})
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps({**autonomy._default_profile(), "enabled": True, "mode": "private_test"}),
        encoding="utf-8",
    )

    result = autonomy.vrchat_autonomy_heartbeat_tick(
        profile_path=profile_path,
        persist_heartbeat=False,
    )

    assert result["success"] is True
    assert result["code"] == "HEARTBEAT_NO_TICK"
    assert result["tick_reason"] == "readiness_not_ready"
    assert result["tick"] is None


def test_validate_decision_allows_private_profile_action_with_voice_and_chatbox():
    result = autonomy.validate_agent_decision(
        {
            "speak_text": "hello",
            "chatbox_text": "hello VRChat",
            "emotion": "happy",
            "avatar_action": "wave",
        },
        mode="private_test",
        allowed_avatar_actions=["wave"],
        allow_voice=True,
        allow_chatbox=True,
    )

    assert result["success"] is True
    assert result["actuation_permitted"] is True
    assert result["normalized"]["avatar_action"] == "wave"
    assert result["normalized"]["emotion"] == "happy"


def test_validate_decision_keeps_observe_mode_non_actuating():
    result = autonomy.validate_agent_decision(
        {"speak_text": "hello", "avatar_action": "wave"},
        mode="observe",
        allowed_avatar_actions=["wave"],
        allow_voice=True,
    )

    assert result["success"] is False
    assert "voice_not_enabled" in result["blocked_reasons"]
    assert "avatar_action_not_enabled" in result["blocked_reasons"]
    assert result["actuation_permitted"] is False


def test_validate_decision_enforces_chatbox_limits():
    result = autonomy.validate_agent_decision(
        {"chatbox_text": "x" * (autonomy.CHATBOX_MAX_CHARS + 1)},
        mode="private_test",
        allow_chatbox=True,
    )

    assert result["success"] is False
    assert any(reason.startswith("chatbox_text_too_long") for reason in result["blocked_reasons"])


def test_validate_decision_blocks_public_movement_even_if_enabled():
    result = autonomy.validate_agent_decision(
        {"movement": {"forward": 1.0}},
        mode="public",
        allow_movement=True,
    )

    assert result["success"] is False
    assert any(reason.startswith("movement_not_allowed") for reason in result["blocked_reasons"])


def test_validate_decision_warns_on_empty_non_actuating_decision():
    result = autonomy.validate_agent_decision({}, mode="observe")

    assert result["success"] is True
    assert result["warnings"] == ["empty_decision"]
    assert result["actuation_permitted"] is False


def test_output_device_detection_matches_output_channels(monkeypatch):
    fake_devices = [
        {"name": "Microphone", "max_output_channels": 0},
        {"name": "CABLE Input (VB-Audio Virtual Cable)", "max_output_channels": 2},
    ]

    class FakeSoundDevice:
        @staticmethod
        def query_devices():
            return fake_devices

    monkeypatch.setitem(__import__("sys").modules, "sounddevice", FakeSoundDevice)

    result = autonomy.find_output_device("cable input")

    assert result["ok"] is True
    assert result["matches"][0]["index"] == 1


def test_normalize_observations_accepts_multimodal_context():
    result = autonomy.normalize_observations(
        [
            {"source": "speechToText", "text": "hello"},
            {"source": "visionObservation", "summary": "User is waving."},
            {"source": "streamComment", "content": "Nice move"},
            {"source": "unknown", "text": "ignored"},
        ]
    )

    assert result["success"] is True
    assert [item["source"] for item in result["accepted"]] == [
        "speechToText",
        "visionObservation",
        "streamComment",
    ]
    assert result["rejected"] == [{"index": "3", "reason": "unsupported_source:unknown"}]
    assert "visionObservation: User is waving." in result["context"]


def test_plan_turn_dry_run_never_actuates():
    result = autonomy.plan_autonomy_turn(
        observations=[{"source": "textBox", "text": "hello"}],
        decision={
            "speak_text": "hello",
            "chatbox_text": "hello VRChat",
            "avatar_action": "wave",
        },
        mode="private_test",
        avatar_action_profiles={"wave": [{"name": "Wave", "value": True}]},
        allow_voice=True,
        allow_chatbox=True,
        dry_run=True,
        output_device="CABLE Input",
    )

    assert result["success"] is True
    assert result["dry_run"] is True
    assert [action["kind"] for action in result["planned_actions"]] == [
        "chatbox",
        "voice",
        "avatar_action",
    ]
    assert result["execution_results"] == []
    assert result["safety"]["actuation_performed"] is False


def test_plan_turn_requires_avatar_action_profile():
    result = autonomy.plan_autonomy_turn(
        observations=[],
        decision={"avatar_action": "wave"},
        mode="private_test",
        allowed_avatar_actions=["wave"],
        dry_run=True,
    )

    assert result["success"] is False
    assert "avatar_action_profile_missing:wave" in result["decision"]["blocked_reasons"]


def test_plan_turn_executes_only_after_validation(monkeypatch):
    calls = []

    def fake_chatbox(text, *, immediate):
        calls.append(("chatbox", text, immediate))
        return {"kind": "chatbox", "attempted": True, "success": True}

    def fake_voice(text, *, speaker, output_device):
        calls.append(("voice", text, speaker, output_device))
        return {"kind": "voice", "attempted": True, "success": True}

    def fake_avatar(action_id, parameters):
        calls.append(("avatar", action_id, parameters))
        return {"kind": "avatar_action", "attempted": True, "success": True}

    monkeypatch.setattr(autonomy, "_send_chatbox", fake_chatbox)
    monkeypatch.setattr(autonomy, "_speak_voicevox", fake_voice)
    monkeypatch.setattr(autonomy, "_apply_avatar_action", fake_avatar)

    result = autonomy.plan_autonomy_turn(
        observations=[{"source": "operator", "text": "private test"}],
        decision={"speak_text": "hi", "chatbox_text": "hi", "avatar_action": "wave"},
        mode="private_test",
        avatar_action_profiles={"wave": [{"name": "Wave", "value": True}]},
        allow_voice=True,
        allow_chatbox=True,
        dry_run=False,
        output_device="CABLE Input",
    )

    assert result["success"] is True
    assert result["safety"] == {
        "actuation_performed": True,
        "chatbox_sent": True,
        "speech_played": True,
        "avatar_parameters_written": True,
    }
    assert calls == [
        ("chatbox", "hi", True),
        ("voice", "hi", 8, "CABLE Input"),
        ("avatar", "wave", [{"name": "Wave", "value": True}]),
    ]
