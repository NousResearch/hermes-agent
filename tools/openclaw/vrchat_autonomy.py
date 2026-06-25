"""Safe readiness and decision validation for VRChat autonomy."""

from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import httpx
import psutil

from hermes_constants import get_hermes_home

CHATBOX_MAX_CHARS = 144
CHATBOX_MAX_LINES = 9
MAX_SPEECH_CHARS = 200
MAX_OBSERVATION_CHARS = 1000
DEFAULT_HEARTBEAT_STATE_NAME = "vrchat-autonomy-heartbeat.json"
DEFAULT_LOOP_STATE_NAME = "vrchat-autonomy-loop.json"
DEFAULT_OBSERVATION_QUEUE_NAME = "vrchat-autonomy-observations.jsonl"
DEFAULT_PROFILE_NAME = "vrchat-autonomy-profile.json"
DEFAULT_MIN_TURN_INTERVAL_SEC = 10.0
MAX_LOOP_OBSERVATIONS = 12
LLM_DECISION_TASK = "vrchat_autonomy"
LIVE_ACTUATION_ACK = "I understand this sends OSC and/or audio to VRChat."
HEARTBEAT_READY_TICK_CODES = {"VRCHAT_LAUNCHED_READY", "READINESS_COMPLETE"}

DEFAULT_VOICEVOX_URL = os.environ.get("VOICEVOX_URL", "http://127.0.0.1:50021")
DEFAULT_HARNESS_URL = os.environ.get("HYPURA_HARNESS_URL", "http://127.0.0.1:18794")
DEFAULT_IRODORI_BASE_URL = os.environ.get("IRODORI_TTS_BASE_URL", "http://127.0.0.1:8088")
DEFAULT_IRODORI_VOICE = "hakua"
DEFAULT_IRODORI_SPEED = 1.0
VALID_TTS_BACKENDS = frozenset({"voicevox", "irodori"})
DEFAULT_TTS_BACKEND = "voicevox"
DEFAULT_VRCHAT_PROCESS_NAMES = ("VRChat.exe", "VRChat")
VOICEVOX_UI_PROCESS_NAMES = {"voicevox.exe", "voicevox"}
VOICEVOX_ENGINE_PROCESS_NAMES = {"run.exe", "run"}
VRCHAT_LAUNCH_CLUE_TOKENS = ("vrchat", "start_protected_game", "easyanticheat")
STEAM_PROCESS_NAMES = {"steam.exe", "steam"}
GENERIC_COMMAND_HOST_PROCESS_NAMES = {
    "cmd.exe",
    "powershell.exe",
    "pwsh.exe",
    "py.exe",
    "python.exe",
    "pythonw.exe",
    "ruff.exe",
}

VALID_MODES = {"observe", "private_test", "trusted_instance", "public"}
VALID_URGENCY = {"low", "medium", "high", "critical"}
VALID_OBSERVATION_SOURCES = {
    "textBox",
    "speechToText",
    "visionObservation",
    "streamComment",
    "operator",
    "system",
}
VALID_EMOTIONS = {
    "neutral",
    "happy",
    "joy",
    "sad",
    "angry",
    "surprised",
    "thinking",
    "shy",
}
RAW_OSC_KEYS = {"osc", "raw_osc", "osc_address", "address", "args"}
MOVEMENT_KEYS = {"movement", "move", "look", "look_target", "input_action"}


def is_vrchat_process_running(process_names: tuple[str, ...] = DEFAULT_VRCHAT_PROCESS_NAMES) -> bool:
    """Return True when a local VRChat process is visible."""
    expected = {name.casefold() for name in process_names}
    for proc in psutil.process_iter(["name"]):
        try:
            name = (proc.info.get("name") or "").casefold()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        if name in expected:
            return True
    return False


def inspect_vrchat_launch_state(
    process_names: tuple[str, ...] = DEFAULT_VRCHAT_PROCESS_NAMES,
    *,
    known_running: bool | None = None,
    exclude_pids: set[int] | None = None,
) -> dict[str, Any]:
    """Return read-only VRChat process diagnostics without treating launch clues as ready."""
    expected = {name.casefold() for name in process_names}
    ignored = set(exclude_pids) if exclude_pids is not None else _current_process_tree_pids()
    matched: list[dict[str, Any]] = []
    clues: list[dict[str, Any]] = []
    steam_processes: list[dict[str, Any]] = []

    for proc in psutil.process_iter(["pid", "name", "exe", "cmdline"]):
        try:
            info = dict(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

        pid = _coerce_pid(info.get("pid"))
        if pid in ignored:
            continue

        name = str(info.get("name") or "")
        name_folded = name.casefold()
        exe = str(info.get("exe") or "")
        cmdline = _join_cmdline(info.get("cmdline"))
        clue_cmdline = "" if name_folded in GENERIC_COMMAND_HOST_PROCESS_NAMES else cmdline
        haystack = " ".join(part for part in (name, exe, clue_cmdline) if part).casefold()
        brief = {"pid": pid, "name": name, "path": exe}

        if name_folded in expected:
            matched.append(brief)
            continue
        if name_folded in STEAM_PROCESS_NAMES:
            steam_processes.append(brief)
        if any(token in haystack for token in VRCHAT_LAUNCH_CLUE_TOKENS):
            clues.append({**brief, "matched_tokens": [t for t in VRCHAT_LAUNCH_CLUE_TOKENS if t in haystack]})

    ok = bool(matched) if known_running is None else bool(known_running)
    if ok:
        phase = "running"
        diagnostic = "VRChat process is visible."
    elif clues:
        phase = "launching_or_blocked"
        diagnostic = "VRChat is not ready, but launcher or anti-cheat related processes were detected."
    elif steam_processes:
        phase = "steam_running_no_vrchat"
        diagnostic = "Steam is running, but VRChat.exe is not visible."
    else:
        phase = "not_detected"
        diagnostic = "VRChat.exe was not detected."

    return {
        "ok": ok,
        "phase": phase,
        "expected_processes": list(process_names),
        "matched_processes": matched[:5],
        "launch_clues": clues[:8],
        "steam": {
            "running": bool(steam_processes),
            "processes": steam_processes[:3],
        },
        "diagnostic": diagnostic,
    }


def inspect_voicevox_runtime_state(*, exclude_pids: set[int] | None = None) -> dict[str, Any]:
    """Return read-only VOICEVOX UI and Engine process diagnostics."""
    ignored = set(exclude_pids) if exclude_pids is not None else _current_process_tree_pids()
    ui_processes: list[dict[str, Any]] = []
    engine_processes: list[dict[str, Any]] = []

    for proc in psutil.process_iter(["pid", "name", "exe", "cmdline"]):
        try:
            info = dict(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

        pid = _coerce_pid(info.get("pid"))
        if pid in ignored:
            continue

        name = str(info.get("name") or "")
        name_folded = name.casefold()
        exe = str(info.get("exe") or "")
        cmdline = _join_cmdline(info.get("cmdline"))
        clue_cmdline = "" if name_folded in GENERIC_COMMAND_HOST_PROCESS_NAMES else cmdline
        haystack = " ".join(part for part in (name, exe, clue_cmdline) if part).casefold()
        brief = {"pid": pid, "name": name, "path": exe}

        if "vv-engine" in haystack or (name_folded in VOICEVOX_ENGINE_PROCESS_NAMES and "voicevox" in haystack):
            engine_processes.append(brief)
        elif name_folded in VOICEVOX_UI_PROCESS_NAMES or "voicevox" in haystack:
            ui_processes.append(brief)

    if engine_processes:
        phase = "engine_process_running"
        diagnostic = "VOICEVOX Engine process is visible."
    elif ui_processes:
        phase = "ui_running_no_engine"
        diagnostic = "VOICEVOX UI is visible, but the Engine process was not detected."
    else:
        phase = "not_detected"
        diagnostic = "VOICEVOX UI and Engine processes were not detected."

    return {
        "phase": phase,
        "ui_running": bool(ui_processes),
        "engine_process_running": bool(engine_processes),
        "ui_processes": ui_processes[:5],
        "engine_processes": engine_processes[:5],
        "diagnostic": diagnostic,
    }


def python_osc_available() -> bool:
    return importlib.util.find_spec("pythonosc") is not None


def probe_voicevox(base_url: str = DEFAULT_VOICEVOX_URL, timeout: float = 2.0) -> dict[str, Any]:
    url = base_url.rstrip("/")
    process_state = inspect_voicevox_runtime_state()
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(f"{url}/version")
        if response.status_code == 200:
            return {
                "ok": True,
                "url": url,
                "version": response.text.strip().strip('"'),
                "process": process_state,
            }
        return {
            "ok": False,
            "url": url,
            "error": f"status_{response.status_code}",
            "detail": response.text[:200],
            "process": process_state,
        }
    except Exception as exc:
        return {
            "ok": False,
            "url": url,
            "error": type(exc).__name__,
            "detail": str(exc),
            "process": process_state,
        }


def _normalize_tts_backend(value: Any) -> str:
    backend = (_coerce_text(value) or DEFAULT_TTS_BACKEND).lower()
    return backend if backend in VALID_TTS_BACKENDS else DEFAULT_TTS_BACKEND


def probe_irodori(
    base_url: str | None = None,
    timeout: float = 3.0,
) -> dict[str, Any]:
    """Probe the local Irodori TTS HTTP server."""
    try:
        from plugins.irodori_tts.core import server_health, settings as irodori_settings
    except ImportError:
        return {
            "ok": False,
            "url": (base_url or DEFAULT_IRODORI_BASE_URL).rstrip("/"),
            "error": "irodori_plugin_not_installed",
        }

    url = (_coerce_text(base_url) or irodori_settings().base_url).rstrip("/")
    health = server_health(url, timeout=timeout)
    return {
        "ok": bool(health.get("ok")),
        "url": url,
        **health,
    }


def probe_harness(base_url: str = DEFAULT_HARNESS_URL, timeout: float = 2.0) -> dict[str, Any]:
    url = base_url.rstrip("/")
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(f"{url}/status")
        if response.status_code == 200:
            payload: Any
            try:
                payload = response.json()
            except ValueError:
                payload = response.text[:200]
            return {"ok": True, "url": url, "status": payload}
        return {
            "ok": False,
            "url": url,
            "error": f"status_{response.status_code}",
            "detail": response.text[:200],
        }
    except Exception as exc:
        return {"ok": False, "url": url, "error": type(exc).__name__, "detail": str(exc)}


def find_output_device(device_name: str | None) -> dict[str, Any]:
    """Check for a configured output device without opening or recording audio."""
    if not device_name:
        return {"ok": None, "configured": False}
    try:
        import sounddevice as sd
    except Exception as exc:
        return {
            "ok": False,
            "configured": True,
            "device": device_name,
            "error": "sounddevice_unavailable",
            "detail": str(exc),
        }

    try:
        devices = sd.query_devices()
    except Exception as exc:
        return {
            "ok": False,
            "configured": True,
            "device": device_name,
            "error": "device_query_failed",
            "detail": str(exc),
        }

    needle = device_name.casefold()
    matches: list[dict[str, Any]] = []
    for index, device in enumerate(devices):
        name = str(device.get("name", ""))
        max_output_channels = int(device.get("max_output_channels", 0) or 0)
        if max_output_channels > 0 and needle in name.casefold():
            matches.append(
                {
                    "index": index,
                    "name": name,
                    "max_output_channels": max_output_channels,
                }
            )
    return {
        "ok": bool(matches),
        "configured": True,
        "device": device_name,
        "matches": matches[:5],
        "error": None if matches else "output_device_not_found",
    }


def vrchat_autonomy_readiness(
    *,
    voicevox_url: str = DEFAULT_VOICEVOX_URL,
    harness_url: str = DEFAULT_HARNESS_URL,
    audio_output_device: str | None = None,
    require_harness: bool = False,
    tts_backend: str = DEFAULT_TTS_BACKEND,
    irodori_base_url: str | None = None,
    require_voice: bool = False,
) -> dict[str, Any]:
    """Return read-only readiness for a VRChat/Hermes autonomous avatar loop."""
    backend = _normalize_tts_backend(tts_backend)
    vrchat_running = is_vrchat_process_running()
    checks: dict[str, dict[str, Any]] = {
        "vrchat_process": inspect_vrchat_launch_state(known_running=vrchat_running),
        "python_osc": {"ok": python_osc_available()},
        "harness": probe_harness(harness_url),
        "audio_output_device": find_output_device(audio_output_device),
        "tts_backend": {"ok": True, "backend": backend},
    }
    if backend == "irodori":
        checks["irodori"] = probe_irodori(irodori_base_url)
        checks["voicevox"] = {"ok": True, "skipped": True, "reason": "tts_backend=irodori"}
    else:
        checks["voicevox"] = probe_voicevox(voicevox_url)
        checks["irodori"] = {"ok": True, "skipped": True, "reason": "tts_backend=voicevox"}

    missing: list[str] = []
    if not checks["vrchat_process"]["ok"]:
        missing.append("VRChat.exe")
    if not checks["python_osc"]["ok"]:
        missing.append("python-osc")
    if require_voice:
        if backend == "irodori":
            if not checks["irodori"]["ok"]:
                missing.append("Irodori TTS server")
        elif not checks["voicevox"]["ok"]:
            missing.append("VOICEVOX Engine")
    if require_harness and not checks["harness"]["ok"]:
        missing.append("Hypura harness")
    if checks["audio_output_device"].get("configured") and not checks["audio_output_device"]["ok"]:
        missing.append("configured audio output device")

    ready = not missing
    return {
        "ready": ready,
        "missing": missing,
        "checks": checks,
        "tts_backend": backend,
        "mode_recommendation": "observe" if ready else "readiness_blocked",
        "safety": {
            "actuation_performed": False,
            "chatbox_sent": False,
            "speech_played": False,
            "avatar_parameters_written": False,
        },
    }


def vrchat_autonomy_heartbeat(
    *,
    voicevox_url: str = DEFAULT_VOICEVOX_URL,
    harness_url: str = DEFAULT_HARNESS_URL,
    audio_output_device: str | None = None,
    require_harness: bool = False,
    state_path: str | Path | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Detect readiness state changes without sending OSC or audio."""
    readiness = vrchat_autonomy_readiness(
        voicevox_url=voicevox_url,
        harness_url=harness_url,
        audio_output_device=audio_output_device,
        require_harness=require_harness,
    )
    current = _heartbeat_signature(readiness)
    prior = _read_heartbeat_state(state_path)
    event = _classify_heartbeat_event(prior.get("current") if prior else None, current)

    payload = {
        "success": True,
        "notify": event["notify"],
        "code": event["code"],
        "message": event["message"],
        "current": current,
        "previous": prior.get("current") if prior else None,
        "readiness": readiness,
        "safety": readiness["safety"],
    }
    if persist:
        _write_heartbeat_state(
            {
                "updated_at": int(time.time()),
                "current": current,
                "last_event": event,
            },
            state_path,
        )
    return payload


def build_decision_request(
    *,
    observations: list[dict[str, Any]] | None,
    mode: str = "observe",
    allowed_avatar_actions: list[str] | None = None,
    avatar_action_descriptions: dict[str, str] | None = None,
    allow_voice: bool = False,
    allow_chatbox: bool = False,
    allow_movement: bool = False,
    allow_interrupt: bool = False,
    persona: str = "",
    task: str = "",
) -> dict[str, Any]:
    """Build a structured LLM request for one VRChat autonomy decision."""
    observation_state = normalize_observations(observations)
    mode = (mode or "observe").strip()
    allowed_actions = list(allowed_avatar_actions or [])
    descriptions = avatar_action_descriptions or {}
    capabilities = {
        "mode": mode,
        "allow_voice": allow_voice,
        "allow_chatbox": allow_chatbox,
        "allow_movement": allow_movement,
        "allow_interrupt": allow_interrupt,
        "allowed_avatar_actions": [
            {"id": action_id, "description": descriptions.get(action_id, "")}
            for action_id in allowed_actions
        ],
    }
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "speak_text": {"type": "string", "maxLength": MAX_SPEECH_CHARS},
            "chatbox_text": {"type": "string", "maxLength": CHATBOX_MAX_CHARS},
            "emotion": {"type": "string", "enum": sorted(VALID_EMOTIONS)},
            "avatar_action": {"type": "string", "enum": ["", *allowed_actions]},
            "urgency": {"type": "string", "enum": sorted(VALID_URGENCY)},
        },
        "required": ["speak_text", "chatbox_text", "emotion", "avatar_action", "urgency"],
    }
    system_prompt = (
        "You produce one safe VRChat avatar decision as JSON only. "
        "Do not include raw OSC addresses, tool names, code, credentials, or hidden reasoning. "
        "Use only the provided avatar_action IDs. Keep speech and ChatBox short. "
        "When mode is observe, leave speak_text, chatbox_text, and avatar_action empty."
    )
    if persona:
        system_prompt += f"\nPersona: {persona[:500]}"
    user_prompt = {
        "task": task or "Choose the next safe conversational/avatar response.",
        "capabilities": capabilities,
        "observations": observation_state["accepted"],
        "rejected_observations": observation_state["rejected"],
        "context": observation_state["context"],
        "safety_notes": [
            "Raw OSC is not allowed.",
            "Public mode blocks movement-like actions.",
            "Critical interruption is unavailable unless allow_interrupt is true.",
            "If no safe action is useful, return empty speak_text, chatbox_text, and avatar_action.",
        ],
    }
    return {
        "success": observation_state["success"],
        "request": {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "vrchat_agent_decision",
                    "strict": True,
                    "schema": schema,
                },
            },
        },
        "decision_schema": schema,
        "observations": observation_state,
        "capabilities": capabilities,
    }


def run_autonomy_decision_turn(
    *,
    observations: list[dict[str, Any]] | None,
    mode: str = "observe",
    allowed_avatar_actions: list[str] | None = None,
    avatar_action_descriptions: dict[str, str] | None = None,
    avatar_action_profiles: dict[str, list[dict[str, Any]]] | None = None,
    allow_voice: bool = False,
    allow_chatbox: bool = False,
    allow_movement: bool = False,
    allow_interrupt: bool = False,
    persona: str = "",
    task: str = "",
    dry_run: bool = True,
    output_device: str | int | None = None,
    voicevox_speaker: int = 8,
    chatbox_immediate: bool = True,
    tts_backend: str = DEFAULT_TTS_BACKEND,
    irodori_voice: str = DEFAULT_IRODORI_VOICE,
    irodori_speed: float = DEFAULT_IRODORI_SPEED,
    irodori_base_url: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
    temperature: float | None = 0.0,
    max_tokens: int = 320,
    llm_call: Callable[..., str] | None = None,
) -> dict[str, Any]:
    """Call a configured Hermes auxiliary model for one validated autonomy turn."""
    profiles = avatar_action_profiles or {}
    action_ids = _ordered_unique(
        [*(allowed_avatar_actions or []), *[_coerce_text(key) for key in profiles.keys()]]
    )
    decision_request = build_decision_request(
        observations=observations,
        mode=mode,
        allowed_avatar_actions=action_ids,
        avatar_action_descriptions=avatar_action_descriptions,
        allow_voice=allow_voice,
        allow_chatbox=allow_chatbox,
        allow_movement=allow_movement,
        allow_interrupt=allow_interrupt,
        persona=persona,
        task=task,
    )
    if not decision_request["success"]:
        return {
            "success": False,
            "stage": "decision_request",
            "dry_run": dry_run,
            "observations": decision_request["observations"],
            "decision_request": _decision_request_summary(decision_request),
            "model_decision": None,
            "turn": None,
            "safety": _safety_flags(),
        }

    caller = llm_call or _call_decision_llm
    try:
        raw_decision = caller(
            decision_request["request"],
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        return {
            "success": False,
            "stage": "llm_call",
            "dry_run": dry_run,
            "observations": decision_request["observations"],
            "decision_request": _decision_request_summary(decision_request),
            "model_decision": None,
            "turn": None,
            "safety": _safety_flags(),
            "error": type(exc).__name__,
            "detail": str(exc),
        }

    parsed = parse_agent_decision_text(raw_decision)
    if not parsed["success"]:
        return {
            "success": False,
            "stage": "decision_parse",
            "dry_run": dry_run,
            "observations": decision_request["observations"],
            "decision_request": _decision_request_summary(decision_request),
            "model_decision": parsed,
            "turn": None,
            "safety": _safety_flags(),
        }

    turn = plan_autonomy_turn(
        observations=observations,
        decision=parsed["decision"],
        mode=mode,
        allowed_avatar_actions=action_ids,
        avatar_action_profiles=profiles,
        allow_voice=allow_voice,
        allow_chatbox=allow_chatbox,
        allow_movement=allow_movement,
        allow_interrupt=allow_interrupt,
        dry_run=dry_run,
        output_device=output_device,
        voicevox_speaker=voicevox_speaker,
        chatbox_immediate=chatbox_immediate,
        tts_backend=tts_backend,
        irodori_voice=irodori_voice,
        irodori_speed=irodori_speed,
        irodori_base_url=irodori_base_url,
    )
    stage = "turn_planned" if dry_run else "turn_executed"
    if not turn["success"]:
        stage = "turn_blocked"
    return {
        "success": bool(turn["success"]),
        "stage": stage,
        "dry_run": dry_run,
        "observations": decision_request["observations"],
        "decision_request": _decision_request_summary(decision_request),
        "llm": {
            "attempted": True,
            "task": LLM_DECISION_TASK,
            "response_chars": len(_coerce_text(raw_decision)),
        },
        "model_decision": parsed,
        "turn": turn,
        "safety": turn["safety"],
    }


def enqueue_observation(
    observation: dict[str, Any],
    *,
    queue_path: str | Path | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Append one normalized observation for a later loop tick."""
    normalized = normalize_observations([observation])
    if not normalized["accepted"]:
        return {
            "success": False,
            "queued": False,
            "observation": None,
            "rejected": normalized["rejected"],
        }

    queued = {
        **normalized["accepted"][0],
        "received_at": int(time.time()),
    }
    if persist:
        path = _observation_queue_path(queue_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(queued, ensure_ascii=False) + "\n")

    return {
        "success": True,
        "queued": persist,
        "observation": queued,
        "queue_path": str(_observation_queue_path(queue_path)) if persist else None,
    }


def load_autonomy_profile(
    profile_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load and validate a local VRChat autonomy operator profile."""
    path = _profile_path(profile_path)
    defaults = _default_profile()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {
            "success": False,
            "exists": False,
            "path": str(path),
            "profile": defaults,
            "errors": ["profile_missing"],
            "warnings": [],
        }
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "success": False,
            "exists": path.exists(),
            "path": str(path),
            "profile": defaults,
            "errors": [f"profile_unreadable:{type(exc).__name__}"],
            "warnings": [str(exc)],
        }

    if not isinstance(raw, dict):
        return {
            "success": False,
            "exists": True,
            "path": str(path),
            "profile": defaults,
            "errors": ["profile_must_be_object"],
            "warnings": [],
        }

    profile = _merge_profile(defaults, raw)
    validation = validate_autonomy_profile(profile)
    return {
        "success": validation["success"],
        "exists": True,
        "path": str(path),
        "profile": profile,
        "errors": validation["errors"],
        "warnings": validation["warnings"],
    }


def validate_autonomy_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Validate a local operator profile before a profile-driven loop tick."""
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(profile, dict):
        return {"success": False, "errors": ["profile_must_be_object"], "warnings": []}

    mode = _coerce_text(profile.get("mode") or "observe")
    if mode not in VALID_MODES:
        errors.append(f"invalid_mode:{mode}")

    raw_action_ids = profile.get("allowed_avatar_actions", [])
    if not isinstance(raw_action_ids, list):
        errors.append("allowed_avatar_actions_must_be_list")
        raw_action_ids = []
    action_ids = [_coerce_text(item) for item in raw_action_ids or []]
    action_ids = [item for item in action_ids if item]
    profile_actions = profile.get("avatar_action_profiles", {})
    profile_validation = validate_avatar_action_profiles(profile_actions)
    errors.extend(profile_validation["errors"])
    warnings.extend(profile_validation["warnings"])

    descriptions = profile.get("avatar_action_descriptions") or {}
    if not isinstance(descriptions, dict):
        errors.append("avatar_action_descriptions_must_be_object")
        descriptions = {}
    unknown_descriptions = set(descriptions.keys()) - set(action_ids)
    if unknown_descriptions:
        warnings.append("description_without_allowed_action:" + ",".join(sorted(unknown_descriptions)))

    profile_action_ids = set(profile_actions.keys()) if isinstance(profile_actions, dict) else set()
    missing_allowlist = profile_action_ids - set(action_ids)
    if missing_allowlist:
        errors.append("profile_action_not_allowlisted:" + ",".join(sorted(missing_allowlist)))

    if bool(profile.get("allow_movement")) and mode == "public":
        errors.append("public_mode_movement_not_allowed")

    if not bool(profile.get("dry_run", True)) and profile.get("live_actuation_ack") != LIVE_ACTUATION_ACK:
        errors.append("live_actuation_ack_required")

    if not bool(profile.get("enabled", False)):
        warnings.append("profile_disabled")

    raw_tts = _coerce_text(profile.get("tts_backend"))
    if raw_tts and raw_tts.lower() not in VALID_TTS_BACKENDS:
        errors.append(f"invalid_tts_backend:{raw_tts}")

    return {"success": not errors, "errors": errors, "warnings": warnings}


def validate_avatar_action_profiles(
    profiles: dict[str, list[dict[str, Any]]] | None,
) -> dict[str, Any]:
    """Validate approved avatar action profiles before OSC parameter writes."""
    if profiles is None:
        return {"success": True, "errors": [], "warnings": []}
    if not isinstance(profiles, dict):
        return {"success": False, "errors": ["avatar_action_profiles_must_be_object"], "warnings": []}

    errors: list[str] = []
    warnings: list[str] = []
    for action_id, writes in profiles.items():
        action = _coerce_text(action_id)
        if not action:
            errors.append("avatar_action_id_empty")
            continue
        if "/" in action:
            errors.append(f"avatar_action_id_contains_slash:{action}")
        if not isinstance(writes, list) or not writes:
            errors.append(f"avatar_action_profile_empty:{action}")
            continue
        for index, write in enumerate(writes):
            if not isinstance(write, dict):
                errors.append(f"avatar_action_write_must_be_object:{action}:{index}")
                continue
            name = _coerce_text(write.get("name"))
            if not name:
                errors.append(f"avatar_parameter_name_missing:{action}:{index}")
            elif name.startswith("/") or name.casefold().startswith("osc"):
                errors.append(f"avatar_parameter_name_not_allowed:{action}:{name}")
            if any(key in write for key in RAW_OSC_KEYS):
                errors.append(f"raw_osc_not_allowed_in_profile:{action}:{index}")
            value = write.get("value")
            if not isinstance(value, (bool, int, float)):
                errors.append(f"avatar_parameter_value_type_not_allowed:{action}:{index}")
            if write.get("reset_after_sec") is not None:
                try:
                    reset = float(write.get("reset_after_sec"))
                except (TypeError, ValueError):
                    errors.append(f"reset_after_sec_invalid:{action}:{index}")
                else:
                    if reset < 0 or reset > 10:
                        errors.append(f"reset_after_sec_out_of_range:{action}:{index}")
            reset_value = write.get("reset_value")
            if reset_value is not None and not isinstance(reset_value, (bool, int, float)):
                errors.append(f"reset_value_type_not_allowed:{action}:{index}")
        if len(writes) > 8:
            warnings.append(f"avatar_action_profile_large:{action}:{len(writes)}")
    return {"success": not errors, "errors": errors, "warnings": warnings}


def vrchat_autonomy_profile_tick(
    *,
    profile_path: str | Path | None = None,
    observations: list[dict[str, Any]] | None = None,
    emergency_stop: bool = False,
    llm_call: Callable[..., str] | None = None,
) -> dict[str, Any]:
    """Run one loop tick using a local operator profile."""
    loaded = load_autonomy_profile(profile_path)
    if emergency_stop:
        tick = vrchat_autonomy_loop_tick(
            enabled=True,
            emergency_stop=True,
            llm_call=llm_call,
        )
        return {"success": True, "profile": loaded, "tick": tick}
    if not loaded["success"]:
        return {
            "success": False,
            "code": "PROFILE_BLOCKED",
            "profile": loaded,
            "tick": None,
            "safety": _safety_flags(),
        }

    profile = loaded["profile"]
    tick = vrchat_autonomy_loop_tick(
        enabled=bool(profile.get("enabled", False)),
        observations=observations,
        consume_queue=bool(profile.get("consume_queue", True)),
        max_observations=int(profile.get("max_observations", MAX_LOOP_OBSERVATIONS)),
        min_turn_interval_sec=float(
            profile.get("min_turn_interval_sec", DEFAULT_MIN_TURN_INTERVAL_SEC)
        ),
        voicevox_url=_coerce_text(profile.get("voicevox_url")) or DEFAULT_VOICEVOX_URL,
        harness_url=_coerce_text(profile.get("harness_url")) or DEFAULT_HARNESS_URL,
        audio_output_device=_optional_text(profile.get("audio_output_device")),
        require_harness=bool(profile.get("require_harness", False)),
        mode=_coerce_text(profile.get("mode")) or "observe",
        allowed_avatar_actions=list(profile.get("allowed_avatar_actions") or []),
        avatar_action_descriptions=dict(profile.get("avatar_action_descriptions") or {}),
        avatar_action_profiles=dict(profile.get("avatar_action_profiles") or {}),
        allow_voice=bool(profile.get("allow_voice", False)),
        allow_chatbox=bool(profile.get("allow_chatbox", False)),
        allow_movement=bool(profile.get("allow_movement", False)),
        allow_interrupt=bool(profile.get("allow_interrupt", False)),
        persona=_coerce_text(profile.get("persona")),
        task=_coerce_text(profile.get("task")),
        dry_run=bool(profile.get("dry_run", True)),
        output_device=profile.get("output_device"),
        voicevox_speaker=int(profile.get("voicevox_speaker", 8)),
        chatbox_immediate=bool(profile.get("chatbox_immediate", True)),
        tts_backend=_normalize_tts_backend(profile.get("tts_backend")),
        irodori_voice=_coerce_text(profile.get("irodori_voice")) or DEFAULT_IRODORI_VOICE,
        irodori_speed=float(profile.get("irodori_speed", DEFAULT_IRODORI_SPEED)),
        irodori_base_url=_optional_text(profile.get("irodori_base_url")),
        provider=_optional_text(profile.get("provider")),
        model=_optional_text(profile.get("model")),
        base_url=_optional_text(profile.get("base_url")),
        timeout=profile.get("timeout"),
        temperature=profile.get("temperature", 0.0),
        max_tokens=int(profile.get("max_tokens", 320)),
        llm_call=llm_call,
    )
    return {
        "success": bool(tick.get("success")),
        "code": tick.get("code"),
        "profile": loaded,
        "tick": tick,
        "safety": tick.get("safety", _safety_flags()),
    }


def vrchat_autonomy_heartbeat_tick(
    *,
    profile_path: str | Path | None = None,
    observations: list[dict[str, Any]] | None = None,
    voicevox_url: str = DEFAULT_VOICEVOX_URL,
    harness_url: str = DEFAULT_HARNESS_URL,
    audio_output_device: str | None = None,
    require_harness: bool = False,
    persist_heartbeat: bool = True,
    tick_on_ready_event: bool = True,
    tick_when_already_ready: bool = False,
    force_tick: bool = False,
    allow_live_profile: bool = False,
    live_ack: str = "",
    emergency_stop: bool = False,
    llm_call: Callable[..., str] | None = None,
) -> dict[str, Any]:
    """Run heartbeat launch detection and optionally one profile-driven tick.

    This is the safe bridge for scheduler/heartbeat use: a live profile never
    runs unless the caller explicitly allows it and repeats the live ACK.
    """
    heartbeat = vrchat_autonomy_heartbeat(
        voicevox_url=voicevox_url,
        harness_url=harness_url,
        audio_output_device=audio_output_device,
        require_harness=require_harness,
        persist=persist_heartbeat,
    )
    profile_state = load_autonomy_profile(profile_path)
    profile = profile_state.get("profile", {})

    if emergency_stop:
        tick = vrchat_autonomy_profile_tick(
            profile_path=profile_path,
            emergency_stop=True,
            llm_call=llm_call,
        )
        return {
            "success": bool(tick.get("success")),
            "code": "EMERGENCY_STOPPED",
            "heartbeat": heartbeat,
            "profile": profile_state,
            "tick": tick,
            "tick_reason": "emergency_stop",
            "safety": tick.get("safety", _safety_flags()),
        }

    tick_reason = _heartbeat_tick_reason(
        heartbeat,
        tick_on_ready_event=tick_on_ready_event,
        tick_when_already_ready=tick_when_already_ready,
        force_tick=force_tick,
    )
    if not tick_reason["should_tick"]:
        return {
            "success": True,
            "code": "HEARTBEAT_NO_TICK",
            "heartbeat": heartbeat,
            "profile": profile_state,
            "tick": None,
            "tick_reason": tick_reason["reason"],
            "safety": heartbeat.get("safety", _safety_flags()),
        }

    live_gate = _heartbeat_live_profile_gate(
        profile_state,
        allow_live_profile=allow_live_profile,
        live_ack=live_ack,
    )
    if not live_gate["allowed"]:
        return {
            "success": False,
            "code": "HEARTBEAT_PROFILE_BLOCKED",
            "heartbeat": heartbeat,
            "profile": profile_state,
            "tick": None,
            "tick_reason": tick_reason["reason"],
            "live_gate": live_gate,
            "safety": heartbeat.get("safety", _safety_flags()),
        }

    tick = vrchat_autonomy_profile_tick(
        profile_path=profile_path,
        observations=observations,
        llm_call=llm_call,
    )
    return {
        "success": bool(tick.get("success")),
        "code": "HEARTBEAT_TICK_DONE" if tick.get("success") else "HEARTBEAT_TICK_BLOCKED",
        "heartbeat": heartbeat,
        "profile": profile_state,
        "tick": tick,
        "tick_reason": tick_reason["reason"],
        "live_gate": live_gate,
        "safety": tick.get("safety", _safety_flags()),
        "dry_run": bool(profile.get("dry_run", True)),
    }


def vrchat_autonomy_loop_tick(
    *,
    enabled: bool = False,
    emergency_stop: bool = False,
    observations: list[dict[str, Any]] | None = None,
    queue_path: str | Path | None = None,
    consume_queue: bool = True,
    max_observations: int = MAX_LOOP_OBSERVATIONS,
    loop_state_path: str | Path | None = None,
    min_turn_interval_sec: float = DEFAULT_MIN_TURN_INTERVAL_SEC,
    voicevox_url: str = DEFAULT_VOICEVOX_URL,
    harness_url: str = DEFAULT_HARNESS_URL,
    audio_output_device: str | None = None,
    require_harness: bool = False,
    mode: str = "observe",
    allowed_avatar_actions: list[str] | None = None,
    avatar_action_descriptions: dict[str, str] | None = None,
    avatar_action_profiles: dict[str, list[dict[str, Any]]] | None = None,
    allow_voice: bool = False,
    allow_chatbox: bool = False,
    allow_movement: bool = False,
    allow_interrupt: bool = False,
    persona: str = "",
    task: str = "",
    dry_run: bool = True,
    output_device: str | int | None = None,
    voicevox_speaker: int = 8,
    chatbox_immediate: bool = True,
    tts_backend: str = DEFAULT_TTS_BACKEND,
    irodori_voice: str = DEFAULT_IRODORI_VOICE,
    irodori_speed: float = DEFAULT_IRODORI_SPEED,
    irodori_base_url: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
    temperature: float | None = 0.0,
    max_tokens: int = 320,
    llm_call: Callable[..., str] | None = None,
) -> dict[str, Any]:
    """Run one safe periodic loop tick for queued multimodal observations."""
    now = int(time.time())
    state = _read_loop_state(loop_state_path)

    if emergency_stop:
        stopped = {
            **state,
            "updated_at": now,
            "enabled": False,
            "last_tick_code": "EMERGENCY_STOPPED",
            "last_stop_at": now,
        }
        _write_loop_state(stopped, loop_state_path)
        return _loop_result(
            success=True,
            code="EMERGENCY_STOPPED",
            message="Autonomy loop disabled by emergency stop.",
            enabled=False,
            state=stopped,
        )

    if not enabled:
        idle = {
            **state,
            "updated_at": now,
            "enabled": False,
            "last_tick_code": "LOOP_DISABLED",
        }
        _write_loop_state(idle, loop_state_path)
        return _loop_result(
            success=True,
            code="LOOP_DISABLED",
            message="Autonomy loop is disabled.",
            enabled=False,
            state=idle,
        )

    readiness = vrchat_autonomy_readiness(
        voicevox_url=voicevox_url,
        harness_url=harness_url,
        audio_output_device=audio_output_device,
        require_harness=require_harness,
        tts_backend=tts_backend,
        irodori_base_url=irodori_base_url,
        require_voice=allow_voice,
    )
    if not readiness["ready"]:
        blocked = {
            **state,
            "updated_at": now,
            "enabled": True,
            "last_tick_code": "READINESS_BLOCKED",
            "last_missing": readiness.get("missing", []),
        }
        _write_loop_state(blocked, loop_state_path)
        return _loop_result(
            success=False,
            code="READINESS_BLOCKED",
            message="VRChat autonomy prerequisites are not ready.",
            enabled=True,
            state=blocked,
            readiness=readiness,
        )

    last_turn_at = int(state.get("last_turn_at") or 0)
    min_interval = max(0.0, float(min_turn_interval_sec))
    elapsed = now - last_turn_at if last_turn_at else None
    if elapsed is not None and elapsed < min_interval:
        limited = {
            **state,
            "updated_at": now,
            "enabled": True,
            "last_tick_code": "RATE_LIMITED",
        }
        _write_loop_state(limited, loop_state_path)
        return _loop_result(
            success=True,
            code="RATE_LIMITED",
            message="Turn interval has not elapsed.",
            enabled=True,
            state=limited,
            readiness=readiness,
            rate_limit={"elapsed_sec": elapsed, "min_interval_sec": min_interval},
        )

    queued, queue_meta = _read_observation_queue(
        queue_path,
        max_items=max_observations,
        consume=False,
    )
    combined = [*(observations or []), *queued]
    normalized = normalize_observations(combined)
    if not normalized["accepted"]:
        no_observations = {
            **state,
            "updated_at": now,
            "enabled": True,
            "last_tick_code": "NO_OBSERVATIONS",
        }
        _write_loop_state(no_observations, loop_state_path)
        return _loop_result(
            success=True,
            code="NO_OBSERVATIONS",
            message="No accepted observations are queued for this tick.",
            enabled=True,
            state=no_observations,
            readiness=readiness,
            observations=normalized,
            queue=queue_meta,
        )

    turn = run_autonomy_decision_turn(
        observations=normalized["accepted"],
        mode=mode,
        allowed_avatar_actions=allowed_avatar_actions,
        avatar_action_descriptions=avatar_action_descriptions,
        avatar_action_profiles=avatar_action_profiles,
        allow_voice=allow_voice,
        allow_chatbox=allow_chatbox,
        allow_movement=allow_movement,
        allow_interrupt=allow_interrupt,
        persona=persona,
        task=task,
        dry_run=dry_run,
        output_device=output_device,
        voicevox_speaker=voicevox_speaker,
        chatbox_immediate=chatbox_immediate,
        tts_backend=tts_backend,
        irodori_voice=irodori_voice,
        irodori_speed=irodori_speed,
        irodori_base_url=irodori_base_url,
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        temperature=temperature,
        max_tokens=max_tokens,
        llm_call=llm_call,
    )
    if consume_queue and queued and turn.get("stage") in {
        "turn_planned",
        "turn_executed",
        "turn_blocked",
    }:
        queue_meta.update(_consume_observation_queue(queue_path, count=len(queued)))
    updated = {
        **state,
        "updated_at": now,
        "enabled": True,
        "last_tick_code": "TURN_DONE" if turn["success"] else "TURN_BLOCKED",
        "last_turn_at": now,
        "last_turn_stage": turn.get("stage"),
        "last_safety": turn.get("safety", _safety_flags()),
        "last_observation_count": len(normalized["accepted"]),
    }
    _write_loop_state(updated, loop_state_path)
    return _loop_result(
        success=bool(turn["success"]),
        code=updated["last_tick_code"],
        message="Autonomy loop tick completed.",
        enabled=True,
        state=updated,
        readiness=readiness,
        observations=normalized,
        queue=queue_meta,
        turn=turn,
    )


def parse_agent_decision_text(raw_text: str) -> dict[str, Any]:
    """Parse a model response into a decision object without accepting prose."""
    text = _strip_json_fence(_coerce_text(raw_text))
    if not text:
        return {"success": False, "error": "empty_model_response", "decision": None}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return {
            "success": False,
            "error": "invalid_json",
            "detail": str(exc),
            "text_preview": text[:300],
            "decision": None,
        }
    if not isinstance(payload, dict):
        return {
            "success": False,
            "error": "decision_must_be_object",
            "decision": None,
        }
    return {"success": True, "decision": payload}


def validate_agent_decision(
    decision: dict[str, Any],
    *,
    mode: str = "observe",
    allowed_avatar_actions: list[str] | None = None,
    allow_voice: bool = False,
    allow_chatbox: bool = False,
    allow_movement: bool = False,
    allow_interrupt: bool = False,
) -> dict[str, Any]:
    """Validate and normalize an untrusted model decision before actuation."""
    allowed_actions = set(allowed_avatar_actions or [])
    blocked: list[str] = []
    warnings: list[str] = []
    normalized: dict[str, Any] = {
        "speak_text": "",
        "chatbox_text": "",
        "emotion": "neutral",
        "avatar_action": "",
        "urgency": "low",
    }

    if not isinstance(decision, dict):
        return {
            "success": False,
            "blocked_reasons": ["decision_must_be_object"],
            "warnings": [],
            "normalized": normalized,
            "mode": mode,
        }

    mode = (mode or "observe").strip()
    if mode not in VALID_MODES:
        blocked.append(f"invalid_mode:{mode}")

    raw_keys = sorted(RAW_OSC_KEYS.intersection(decision.keys()))
    if raw_keys:
        blocked.append("raw_osc_not_allowed:" + ",".join(raw_keys))

    movement_keys = sorted(MOVEMENT_KEYS.intersection(decision.keys()))
    if movement_keys and (mode == "public" or not allow_movement):
        blocked.append("movement_not_allowed:" + ",".join(movement_keys))

    speak_text = _coerce_text(decision.get("speak_text") or decision.get("speech"))
    if speak_text:
        if mode == "observe" or not allow_voice:
            blocked.append("voice_not_enabled")
        if len(speak_text) > MAX_SPEECH_CHARS:
            blocked.append(f"speak_text_too_long:{len(speak_text)}>{MAX_SPEECH_CHARS}")
        normalized["speak_text"] = speak_text[:MAX_SPEECH_CHARS]

    chatbox_text = _coerce_text(decision.get("chatbox_text") or decision.get("chatbox"))
    if chatbox_text:
        if mode == "observe" or not allow_chatbox:
            blocked.append("chatbox_not_enabled")
        if len(chatbox_text) > CHATBOX_MAX_CHARS:
            blocked.append(f"chatbox_text_too_long:{len(chatbox_text)}>{CHATBOX_MAX_CHARS}")
        if len(chatbox_text.splitlines()) > CHATBOX_MAX_LINES:
            blocked.append(
                f"chatbox_too_many_lines:{len(chatbox_text.splitlines())}>{CHATBOX_MAX_LINES}"
            )
        normalized["chatbox_text"] = chatbox_text[:CHATBOX_MAX_CHARS]

    emotion = _coerce_text(decision.get("emotion") or "neutral").casefold()
    if emotion not in VALID_EMOTIONS:
        blocked.append(f"unsupported_emotion:{emotion}")
    else:
        normalized["emotion"] = emotion

    urgency = _coerce_text(decision.get("urgency") or "low").casefold()
    if urgency not in VALID_URGENCY:
        blocked.append(f"unsupported_urgency:{urgency}")
    elif urgency == "critical" and not allow_interrupt:
        blocked.append("critical_interrupt_not_enabled")
    else:
        normalized["urgency"] = urgency

    avatar_action = _extract_avatar_action(decision.get("avatar_action"))
    if avatar_action:
        if mode == "observe":
            blocked.append("avatar_action_not_enabled")
        elif avatar_action not in allowed_actions:
            blocked.append(f"avatar_action_not_allowed:{avatar_action}")
        normalized["avatar_action"] = avatar_action

    if not any(normalized.get(key) for key in ("speak_text", "chatbox_text", "avatar_action")):
        warnings.append("empty_decision")

    return {
        "success": not blocked,
        "blocked_reasons": blocked,
        "warnings": warnings,
        "normalized": normalized,
        "mode": mode,
        "actuation_permitted": not blocked and mode != "observe",
    }


def normalize_observations(observations: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Normalize multimodal/conversation observations into bounded text context."""
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []

    if observations is None:
        observations = []
    if not isinstance(observations, list):
        return {
            "success": False,
            "accepted": [],
            "rejected": [{"reason": "observations_must_be_list"}],
            "context": "",
        }

    for index, item in enumerate(observations):
        if not isinstance(item, dict):
            rejected.append({"index": str(index), "reason": "observation_must_be_object"})
            continue
        source = _coerce_text(item.get("source") or item.get("type"))
        if source not in VALID_OBSERVATION_SOURCES:
            rejected.append({"index": str(index), "reason": f"unsupported_source:{source}"})
            continue
        text = _coerce_text(item.get("text") or item.get("summary") or item.get("content"))
        if not text:
            rejected.append({"index": str(index), "reason": "empty_text"})
            continue
        truncated = len(text) > MAX_OBSERVATION_CHARS
        accepted.append(
            {
                "source": source,
                "text": text[:MAX_OBSERVATION_CHARS],
                "trust": _coerce_text(item.get("trust") or "untrusted") or "untrusted",
                "timestamp": _coerce_text(item.get("timestamp")),
                "truncated": truncated,
            }
        )

    context = "\n".join(f"- {obs['source']}: {obs['text']}" for obs in accepted)
    return {
        "success": True,
        "accepted": accepted,
        "rejected": rejected,
        "context": context,
    }


def plan_autonomy_turn(
    *,
    observations: list[dict[str, Any]] | None,
    decision: dict[str, Any],
    mode: str = "observe",
    allowed_avatar_actions: list[str] | None = None,
    avatar_action_profiles: dict[str, list[dict[str, Any]]] | None = None,
    allow_voice: bool = False,
    allow_chatbox: bool = False,
    allow_movement: bool = False,
    allow_interrupt: bool = False,
    dry_run: bool = True,
    output_device: str | int | None = None,
    voicevox_speaker: int = 8,
    chatbox_immediate: bool = True,
    tts_backend: str = DEFAULT_TTS_BACKEND,
    irodori_voice: str = DEFAULT_IRODORI_VOICE,
    irodori_speed: float = DEFAULT_IRODORI_SPEED,
    irodori_base_url: str | None = None,
) -> dict[str, Any]:
    """Validate one autonomous turn and optionally execute permitted actions."""
    observation_state = normalize_observations(observations)
    profiles = avatar_action_profiles or {}
    profile_validation = validate_avatar_action_profiles(profiles)
    if not profile_validation["success"]:
        return {
            "success": False,
            "dry_run": dry_run,
            "observations": observation_state,
            "decision": {
                "success": False,
                "blocked_reasons": profile_validation["errors"],
                "warnings": profile_validation["warnings"],
                "normalized": {},
                "mode": mode,
                "actuation_permitted": False,
            },
            "planned_actions": [],
            "execution_results": [],
            "safety": _safety_flags(),
        }
    profile_action_ids = set(profiles.keys())
    allowed = set(allowed_avatar_actions or []) | profile_action_ids

    validation = validate_agent_decision(
        decision,
        mode=mode,
        allowed_avatar_actions=sorted(allowed),
        allow_voice=allow_voice,
        allow_chatbox=allow_chatbox,
        allow_movement=allow_movement,
        allow_interrupt=allow_interrupt,
    )

    if not validation["success"]:
        return {
            "success": False,
            "dry_run": dry_run,
            "observations": observation_state,
            "decision": validation,
            "planned_actions": [],
            "execution_results": [],
            "safety": _safety_flags(),
        }

    normalized = validation["normalized"]
    planned_actions = _build_planned_actions(
        normalized,
        profiles,
        output_device=output_device,
        voicevox_speaker=voicevox_speaker,
        chatbox_immediate=chatbox_immediate,
        tts_backend=tts_backend,
        irodori_voice=irodori_voice,
        irodori_speed=irodori_speed,
        irodori_base_url=irodori_base_url,
    )

    missing_profile = [
        action
        for action in planned_actions
        if action["kind"] == "avatar_action" and not action.get("parameters")
    ]
    if missing_profile:
        validation = {
            **validation,
            "success": False,
            "blocked_reasons": [
                *validation.get("blocked_reasons", []),
                f"avatar_action_profile_missing:{missing_profile[0]['action_id']}",
            ],
            "actuation_permitted": False,
        }
        return {
            "success": False,
            "dry_run": dry_run,
            "observations": observation_state,
            "decision": validation,
            "planned_actions": planned_actions,
            "execution_results": [],
            "safety": _safety_flags(),
        }

    execution_results: list[dict[str, Any]] = []
    safety = _safety_flags()
    if not dry_run:
        for action in planned_actions:
            result = _execute_action(action)
            execution_results.append(result)
            if result.get("attempted"):
                safety["actuation_performed"] = True
            if action["kind"] == "chatbox" and result.get("attempted"):
                safety["chatbox_sent"] = True
            if action["kind"] == "voice" and result.get("attempted"):
                safety["speech_played"] = True
            if action["kind"] == "avatar_action" and result.get("attempted"):
                safety["avatar_parameters_written"] = True

    return {
        "success": True,
        "dry_run": dry_run,
        "observations": observation_state,
        "decision": validation,
        "planned_actions": planned_actions,
        "execution_results": execution_results,
        "safety": safety,
    }


def _build_planned_actions(
    normalized: dict[str, Any],
    avatar_action_profiles: dict[str, list[dict[str, Any]]],
    *,
    output_device: str | int | None,
    voicevox_speaker: int,
    chatbox_immediate: bool,
    tts_backend: str = DEFAULT_TTS_BACKEND,
    irodori_voice: str = DEFAULT_IRODORI_VOICE,
    irodori_speed: float = DEFAULT_IRODORI_SPEED,
    irodori_base_url: str | None = None,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if normalized.get("chatbox_text"):
        actions.append(
            {
                "kind": "chatbox",
                "text": normalized["chatbox_text"],
                "immediate": chatbox_immediate,
            }
        )
    if normalized.get("speak_text"):
        actions.append(
            {
                "kind": "voice",
                "text": normalized["speak_text"],
                "tts_backend": _normalize_tts_backend(tts_backend),
                "speaker": voicevox_speaker,
                "emotion": normalized.get("emotion", "neutral"),
                "output_device": output_device,
                "irodori_voice": irodori_voice,
                "irodori_speed": irodori_speed,
                "irodori_base_url": irodori_base_url,
            }
        )
    if normalized.get("avatar_action"):
        action_id = normalized["avatar_action"]
        actions.append(
            {
                "kind": "avatar_action",
                "action_id": action_id,
                "parameters": avatar_action_profiles.get(action_id, []),
            }
        )
    return actions


def _execute_action(action: dict[str, Any]) -> dict[str, Any]:
    if action["kind"] == "chatbox":
        return _send_chatbox(action["text"], immediate=bool(action.get("immediate", True)))
    if action["kind"] == "voice":
        return _speak_tts(
            action["text"],
            tts_backend=_normalize_tts_backend(action.get("tts_backend")),
            speaker=int(action.get("speaker", 8)),
            output_device=action.get("output_device"),
            irodori_voice=_coerce_text(action.get("irodori_voice")) or DEFAULT_IRODORI_VOICE,
            irodori_speed=float(action.get("irodori_speed", DEFAULT_IRODORI_SPEED)),
            irodori_base_url=action.get("irodori_base_url"),
        )
    if action["kind"] == "avatar_action":
        return _apply_avatar_action(action["action_id"], list(action.get("parameters") or []))
    return {"success": False, "attempted": False, "error": f"unknown_action_kind:{action['kind']}"}


def _call_decision_llm(
    request: dict[str, Any],
    *,
    provider: str | None,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
    timeout: float | None,
    temperature: float | None,
    max_tokens: int,
) -> str:
    from agent.auxiliary_client import call_llm

    response = call_llm(
        task=LLM_DECISION_TASK,
        provider=provider or None,
        model=model or None,
        base_url=base_url or None,
        api_key=api_key or None,
        messages=list(request.get("messages") or []),
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        extra_body={"response_format": request.get("response_format")},
    )
    return _extract_llm_message_content(response)


def _extract_llm_message_content(response: Any) -> str:
    try:
        message = response.choices[0].message
    except (AttributeError, TypeError, IndexError) as exc:
        raise RuntimeError("LLM response did not include choices[0].message.") from exc
    text = _content_to_text(getattr(message, "content", ""))
    if not text:
        raise RuntimeError("LLM response did not include message content.")
    return text


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                value = block.get("text") or block.get("content")
                if value is not None:
                    parts.append(str(value))
        return "\n".join(part.strip() for part in parts if part and str(part).strip()).strip()
    if content is None:
        return ""
    return str(content).strip()


def _strip_json_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


def _decision_request_summary(decision_request: dict[str, Any]) -> dict[str, Any]:
    return {
        "capabilities": decision_request.get("capabilities", {}),
        "decision_schema": decision_request.get("decision_schema", {}),
        "observations": decision_request.get("observations", {}),
    }


def _profile_path(profile_path: str | Path | None) -> Path:
    if profile_path:
        return Path(profile_path).expanduser()
    return get_hermes_home() / "config" / DEFAULT_PROFILE_NAME


def _default_profile() -> dict[str, Any]:
    return {
        "enabled": False,
        "mode": "observe",
        "dry_run": True,
        "consume_queue": True,
        "max_observations": MAX_LOOP_OBSERVATIONS,
        "min_turn_interval_sec": DEFAULT_MIN_TURN_INTERVAL_SEC,
        "voicevox_url": DEFAULT_VOICEVOX_URL,
        "harness_url": DEFAULT_HARNESS_URL,
        "audio_output_device": "",
        "vrchat_microphone_device": "",
        "require_harness": False,
        "allow_voice": False,
        "allow_chatbox": False,
        "allow_movement": False,
        "allow_interrupt": False,
        "output_device": "",
        "voicevox_speaker": 8,
        "chatbox_immediate": True,
        "tts_backend": DEFAULT_TTS_BACKEND,
        "irodori_voice": DEFAULT_IRODORI_VOICE,
        "irodori_speed": DEFAULT_IRODORI_SPEED,
        "irodori_base_url": "",
        "persona": "",
        "task": "",
        "allowed_avatar_actions": [],
        "avatar_action_descriptions": {},
        "avatar_action_profiles": {},
        "provider": "",
        "model": "",
        "base_url": "",
        "timeout": None,
        "temperature": 0.0,
        "max_tokens": 320,
        "live_actuation_ack": "",
    }


def _merge_profile(defaults: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    profile = dict(defaults)
    for key in defaults:
        if key in raw:
            profile[key] = raw[key]
    return profile


def _optional_text(value: Any) -> str | None:
    text = _coerce_text(value)
    return text or None


def _loop_result(
    *,
    success: bool,
    code: str,
    message: str,
    enabled: bool,
    state: dict[str, Any],
    readiness: dict[str, Any] | None = None,
    observations: dict[str, Any] | None = None,
    queue: dict[str, Any] | None = None,
    rate_limit: dict[str, Any] | None = None,
    turn: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "success": success,
        "code": code,
        "message": message,
        "enabled": enabled,
        "readiness": readiness,
        "observations": observations,
        "queue": queue,
        "rate_limit": rate_limit,
        "turn": turn,
        "state": {
            "updated_at": state.get("updated_at"),
            "enabled": bool(state.get("enabled", enabled)),
            "last_tick_code": state.get("last_tick_code"),
            "last_turn_at": state.get("last_turn_at"),
            "last_turn_stage": state.get("last_turn_stage"),
            "last_safety": state.get("last_safety", _safety_flags()),
        },
        "safety": (turn or {}).get("safety", _safety_flags()),
    }


def _send_chatbox(text: str, *, immediate: bool) -> dict[str, Any]:
    from tools.vrchat_osc_tool import vrchat_chatbox

    result = vrchat_chatbox(text, immediate=immediate)
    return {"kind": "chatbox", "attempted": True, "result": result, "success": bool(result.get("success"))}


def _speak_tts(
    text: str,
    *,
    tts_backend: str,
    speaker: int,
    output_device: str | int | None,
    irodori_voice: str,
    irodori_speed: float,
    irodori_base_url: str | None,
) -> dict[str, Any]:
    backend = _normalize_tts_backend(tts_backend)
    if backend == "irodori":
        return _speak_irodori(
            text,
            voice=irodori_voice,
            speed=irodori_speed,
            irodori_base_url=irodori_base_url,
            output_device=output_device,
        )
    return _speak_voicevox(text, speaker=speaker, output_device=output_device)


def _speak_voicevox(
    text: str,
    *,
    speaker: int,
    output_device: str | int | None,
) -> dict[str, Any]:
    from tools.voicevox_tts_tool import voicevox_speak

    result = voicevox_speak(text, speaker=speaker, blocking=True, output_device=output_device)
    return {
        "kind": "voice",
        "backend": "voicevox",
        "attempted": True,
        "result": result,
        "success": bool(result.get("success")),
    }


def _speak_irodori(
    text: str,
    *,
    voice: str,
    speed: float,
    irodori_base_url: str | None,
    output_device: str | int | None,
) -> dict[str, Any]:
    try:
        from plugins.irodori_tts.core import synthesize_text

        env_backup = os.environ.get("IRODORI_TTS_BASE_URL")
        if irodori_base_url:
            os.environ["IRODORI_TTS_BASE_URL"] = str(irodori_base_url).rstrip("/")
        try:
            synth = synthesize_text(text, voice=voice or None, speed=speed or None)
        finally:
            if env_backup is None:
                os.environ.pop("IRODORI_TTS_BASE_URL", None)
            else:
                os.environ["IRODORI_TTS_BASE_URL"] = env_backup
        wav_path = Path(str(synth.get("file_path") or ""))
        if not wav_path.is_file():
            return {
                "kind": "voice",
                "backend": "irodori",
                "attempted": True,
                "success": False,
                "result": {"success": False, "error": "irodori_wav_missing"},
            }
        wav_bytes = wav_path.read_bytes()
        from tools.voicevox_tts_tool import _play_wav

        play_result = _play_wav(wav_bytes, blocking=True, output_device=output_device)
        return {
            "kind": "voice",
            "backend": "irodori",
            "attempted": True,
            "result": {"synthesis": synth, "playback": play_result},
            "success": bool(play_result.get("success")),
        }
    except Exception as exc:
        return {
            "kind": "voice",
            "backend": "irodori",
            "attempted": True,
            "success": False,
            "result": {"success": False, "error": type(exc).__name__, "detail": str(exc)},
        }


def _apply_avatar_action(action_id: str, parameters: list[dict[str, Any]]) -> dict[str, Any]:
    from tools.vrchat_osc_tool import vrchat_avatar_param

    results: list[dict[str, Any]] = []
    for parameter in parameters:
        name = _coerce_text(parameter.get("name"))
        if not name:
            results.append({"success": False, "error": "parameter_name_missing"})
            continue
        result = vrchat_avatar_param(name, parameter.get("value"))
        results.append(result)
        if result.get("success") and parameter.get("reset_after_sec") is not None:
            reset_after = min(max(float(parameter.get("reset_after_sec") or 0), 0.0), 10.0)
            if reset_after:
                time.sleep(reset_after)
            results.append(vrchat_avatar_param(name, parameter.get("reset_value", False)))
    return {
        "kind": "avatar_action",
        "action_id": action_id,
        "attempted": True,
        "result": results,
        "success": all(bool(item.get("success")) for item in results),
    }


def _safety_flags() -> dict[str, bool]:
    return {
        "actuation_performed": False,
        "chatbox_sent": False,
        "speech_played": False,
        "avatar_parameters_written": False,
    }


def _heartbeat_state_path(state_path: str | Path | None) -> Path:
    if state_path:
        return Path(state_path).expanduser()
    return get_hermes_home() / "state" / DEFAULT_HEARTBEAT_STATE_NAME


def _read_heartbeat_state(state_path: str | Path | None) -> dict[str, Any]:
    path = _heartbeat_state_path(state_path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_heartbeat_state(state: dict[str, Any], state_path: str | Path | None) -> None:
    path = _heartbeat_state_path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _loop_state_path(state_path: str | Path | None) -> Path:
    if state_path:
        return Path(state_path).expanduser()
    return get_hermes_home() / "state" / DEFAULT_LOOP_STATE_NAME


def _read_loop_state(state_path: str | Path | None) -> dict[str, Any]:
    path = _loop_state_path(state_path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_loop_state(state: dict[str, Any], state_path: str | Path | None) -> None:
    path = _loop_state_path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _observation_queue_path(queue_path: str | Path | None) -> Path:
    if queue_path:
        return Path(queue_path).expanduser()
    return get_hermes_home() / "state" / DEFAULT_OBSERVATION_QUEUE_NAME


def _read_observation_queue(
    queue_path: str | Path | None,
    *,
    max_items: int,
    consume: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = _observation_queue_path(queue_path)
    limit = max(0, int(max_items))
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return [], {
            "path": str(path),
            "read": 0,
            "remaining": 0,
            "consumed": False,
            "missing": True,
        }

    parsed: list[dict[str, Any]] = []
    rejected = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            rejected += 1
            continue
        if isinstance(item, dict):
            parsed.append(item)
        else:
            rejected += 1

    selected = parsed[:limit] if limit else []
    remaining = parsed[len(selected):] if consume else parsed
    if consume:
        path.parent.mkdir(parents=True, exist_ok=True)
        if remaining:
            path.write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in remaining),
                encoding="utf-8",
            )
        else:
            try:
                path.unlink()
            except OSError:
                path.write_text("", encoding="utf-8")

    return selected, {
        "path": str(path),
        "read": len(selected),
        "remaining": len(remaining),
        "rejected": rejected,
        "consumed": consume,
        "missing": False,
    }


def _consume_observation_queue(
    queue_path: str | Path | None,
    *,
    count: int,
) -> dict[str, Any]:
    path = _observation_queue_path(queue_path)
    consume_count = max(0, int(count))
    queued, meta = _read_observation_queue(
        path,
        max_items=consume_count,
        consume=True,
    )
    return {
        "consumed": True,
        "consumed_count": len(queued),
        "remaining": meta.get("remaining", 0),
        "missing": meta.get("missing", False),
    }


def _current_process_tree_pids() -> set[int]:
    pids = {os.getpid()}
    try:
        proc = psutil.Process(os.getpid())
        pids.update(parent.pid for parent in proc.parents())
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        pass
    return pids


def _coerce_pid(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _join_cmdline(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(str(part) for part in value if part)
    if value is None:
        return ""
    return str(value)


def _heartbeat_signature(readiness: dict[str, Any]) -> dict[str, Any]:
    checks = readiness.get("checks", {})
    return {
        "vrchat_running": bool(checks.get("vrchat_process", {}).get("ok")),
        "ready": bool(readiness.get("ready")),
        "missing": list(readiness.get("missing") or []),
        "voicevox_ok": bool(checks.get("voicevox", {}).get("ok")),
        "harness_ok": bool(checks.get("harness", {}).get("ok")),
        "audio_output_ok": checks.get("audio_output_device", {}).get("ok"),
    }


def _classify_heartbeat_event(
    prior: dict[str, Any] | None,
    current: dict[str, Any],
) -> dict[str, Any]:
    if not current["vrchat_running"]:
        return {"notify": False, "code": "HEARTBEAT_OK", "message": "VRChat is not running."}
    if not prior or not prior.get("vrchat_running"):
        if current["ready"]:
            return {"notify": True, "code": "VRCHAT_LAUNCHED_READY", "message": "VRChat launched and autonomy prerequisites are ready."}
        missing = ", ".join(current["missing"]) or "unknown component"
        return {"notify": True, "code": "VRCHAT_LAUNCHED_BLOCKED", "message": f"VRChat launched; missing {missing}."}
    if prior.get("missing") != current.get("missing"):
        if current["ready"]:
            return {"notify": True, "code": "READINESS_COMPLETE", "message": "VRChat autonomy readiness is complete."}
        missing = ", ".join(current["missing"]) or "unknown component"
        return {"notify": True, "code": "READINESS_CHANGED", "message": f"VRChat autonomy readiness changed; missing {missing}."}
    return {"notify": False, "code": "HEARTBEAT_OK", "message": "No actionable VRChat readiness change."}


def _heartbeat_tick_reason(
    heartbeat: dict[str, Any],
    *,
    tick_on_ready_event: bool,
    tick_when_already_ready: bool,
    force_tick: bool,
) -> dict[str, Any]:
    if force_tick:
        return {"should_tick": True, "reason": "force_tick"}
    code = _coerce_text(heartbeat.get("code"))
    current = heartbeat.get("current") or {}
    if tick_on_ready_event and heartbeat.get("notify") and code in HEARTBEAT_READY_TICK_CODES:
        return {"should_tick": True, "reason": code}
    if tick_when_already_ready and current.get("ready"):
        return {"should_tick": True, "reason": "already_ready"}
    if not current.get("ready"):
        return {"should_tick": False, "reason": "readiness_not_ready"}
    return {"should_tick": False, "reason": "no_ready_event"}


def _heartbeat_live_profile_gate(
    profile_state: dict[str, Any],
    *,
    allow_live_profile: bool,
    live_ack: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    profile = profile_state.get("profile", {})
    if not profile_state.get("success"):
        reasons.append("profile_invalid")
    if not bool(profile.get("enabled", False)):
        reasons.append("profile_disabled")
    if not bool(profile.get("dry_run", True)):
        if not allow_live_profile:
            reasons.append("allow_live_profile_required")
        if live_ack != LIVE_ACTUATION_ACK:
            reasons.append("live_ack_required")
    return {
        "allowed": not reasons,
        "blocked_reasons": reasons,
        "profile_dry_run": bool(profile.get("dry_run", True)),
        "live_profile_requested": not bool(profile.get("dry_run", True)),
    }


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _extract_avatar_action(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return _coerce_text(value.get("id") or value.get("name") or value.get("action_id"))
    return _coerce_text(value)


def _ordered_unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _coerce_text(value)
        if normalized and normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    return result
