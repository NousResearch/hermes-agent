"""Read-only preflight evidence for VRChat autonomy live smoke tests."""

from __future__ import annotations

import json
import os
import platform
import re
import socket
import time
from pathlib import Path
from typing import Any

import psutil
import requests

from tools.openclaw.neuro_bridge import neuro_sdk_vendor_status
from tools.openclaw.vrchat_autonomy import (
    DEFAULT_HARNESS_URL,
    DEFAULT_VOICEVOX_URL,
    LIVE_ACTUATION_ACK,
    load_autonomy_profile,
    probe_voicevox,
    vrchat_autonomy_heartbeat_tick,
    vrchat_autonomy_heartbeat,
    vrchat_autonomy_readiness,
    find_output_device,
)
from tools.openclaw.vrchat_observations import observation_queue_status

COMMON_VOICEVOX_URLS = (
    DEFAULT_VOICEVOX_URL,
    "http://localhost:50021",
    "http://127.0.0.1:50031",
    "http://localhost:50031",
)
RUNTIME_WINDOW_TERMS = ("vrchat", "voicevox")
RUNTIME_PROCESS_TERMS = (
    "vrchat",
    "voicevox",
    "vv-engine",
    "start_protected_game",
    "easyanticheat",
    "steam",
)
GENERIC_PROCESS_VISIBILITY_HOSTS = {
    "cmd.exe",
    "powershell.exe",
    "pwsh.exe",
    "py.exe",
    "python.exe",
    "pythonw.exe",
    "ruff.exe",
}
STEAM_VRCHAT_APP_ID = "438100"
VOICEVOX_EXE_NAME = "VOICEVOX.exe"
DEFAULT_VOICEVOX_SYNTHESIS_TEXT = "\u30c6\u30b9\u30c8"


def build_preflight_bundle(
    *,
    profile_path: str | Path | None = None,
    voicevox_url: str = DEFAULT_VOICEVOX_URL,
    harness_url: str = DEFAULT_HARNESS_URL,
    audio_output_device: str | None = None,
    require_harness: bool = False,
    queue_path: str | Path | None = None,
    include_audio_devices: bool = True,
    max_audio_devices: int = 20,
    include_voicevox_synthesis: bool = False,
    voicevox_synthesis_text: str = DEFAULT_VOICEVOX_SYNTHESIS_TEXT,
    voicevox_synthesis_speaker: int | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Collect read-only evidence needed before a VRChat live smoke test."""
    profile = load_autonomy_profile(profile_path)
    profile_data = profile.get("profile", {}) if profile.get("success") else {}
    effective_audio_output_device = (
        audio_output_device
        or profile_data.get("audio_output_device")
        or profile_data.get("output_device")
        or None
    )
    vrchat_microphone_device = (
        profile_data.get("vrchat_microphone_device")
        or infer_virtual_cable_microphone_device(effective_audio_output_device)
    )
    voicevox_speaker = _coerce_int(
        voicevox_synthesis_speaker
        if voicevox_synthesis_speaker is not None
        else profile_data.get("voicevox_speaker"),
        default=8,
    )
    readiness = vrchat_autonomy_readiness(
        voicevox_url=voicevox_url,
        harness_url=harness_url,
        audio_output_device=effective_audio_output_device,
        require_harness=require_harness,
    )
    heartbeat = vrchat_autonomy_heartbeat(
        voicevox_url=voicevox_url,
        harness_url=harness_url,
        audio_output_device=effective_audio_output_device,
        require_harness=require_harness,
        persist=False,
    )
    bundle = {
        "success": True,
        "created_at": int(time.time()),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
        },
        "vendor": {
            "neuro_sdk": neuro_sdk_vendor_status(),
        },
        "profile": profile,
        "readiness": readiness,
        "heartbeat_preview": {
            "code": heartbeat.get("code"),
            "notify": heartbeat.get("notify"),
            "message": heartbeat.get("message"),
            "current": heartbeat.get("current"),
        },
        "observations": observation_queue_status(queue_path=queue_path),
        "voicevox_synthesis": probe_voicevox_synthesis(
            voicevox_url=voicevox_url,
            text=voicevox_synthesis_text,
            speaker=voicevox_speaker,
        )
        if include_voicevox_synthesis
        else {
            "success": True,
            "included": False,
            "ok": None,
            "played_audio": False,
            "microphone_recorded": False,
        },
        "audio": {
            "requested_output_device": effective_audio_output_device or "",
            "requested_output_device_match": find_output_device(effective_audio_output_device),
            "requested_vrchat_microphone_device": vrchat_microphone_device or "",
            "virtual_cable_route": check_virtual_cable_route(
                output_device=effective_audio_output_device,
                microphone_device=vrchat_microphone_device,
            ),
            "output_devices": list_audio_output_devices(
                max_devices=max_audio_devices,
            )
            if include_audio_devices
            else {"success": True, "included": False, "devices": []},
        },
        "live_smoke_gate": _live_smoke_gate_summary(profile, readiness),
        "commands": _recommended_commands(
            profile.get("path"),
            effective_audio_output_device,
            vrchat_microphone_device,
        ),
        "safety": {
            "actuation_performed": False,
            "chatbox_sent": False,
            "speech_played": False,
            "avatar_parameters_written": False,
            "microphone_recorded": False,
            "websocket_opened": False,
        },
    }
    if output_path:
        path = Path(output_path).expanduser()
        bundle["output_path"] = str(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    return bundle


def build_runtime_doctor(
    *,
    profile_path: str | Path | None = None,
    voicevox_url: str = DEFAULT_VOICEVOX_URL,
    harness_url: str = DEFAULT_HARNESS_URL,
    audio_output_device: str | None = None,
    require_harness: bool = False,
    queue_path: str | Path | None = None,
    include_audio_devices: bool = True,
    max_audio_devices: int = 20,
    operator_reported_vrchat: bool = False,
    operator_reported_voicevox: bool = False,
    voicevox_probe_timeout: float = 1.0,
    output_path: str | Path | None = None,
    preflight_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Explain read-only runtime readiness blockers and operator mismatches."""
    preflight = preflight_bundle or build_preflight_bundle(
        profile_path=profile_path,
        voicevox_url=voicevox_url,
        harness_url=harness_url,
        audio_output_device=audio_output_device,
        require_harness=require_harness,
        queue_path=queue_path,
        include_audio_devices=include_audio_devices,
        max_audio_devices=max_audio_devices,
    )
    readiness = preflight.get("readiness", {})
    checks = readiness.get("checks", {})
    profile_state = preflight.get("profile", {})
    effective_voicevox_url = (
        str(checks.get("voicevox", {}).get("url") or voicevox_url or DEFAULT_VOICEVOX_URL).rstrip("/")
    )
    voicevox_candidates = _probe_voicevox_candidates(
        effective_voicevox_url,
        probe_timeout=voicevox_probe_timeout,
    )
    ports = _diagnostic_ports(effective_voicevox_url, harness_url)
    network = _local_port_snapshot(ports)
    desktop_windows = _relevant_window_snapshot()
    launch_discovery = _runtime_launch_discovery()
    process_visibility = _process_visibility_snapshot()
    summary = _runtime_doctor_summary(
        preflight,
        voicevox_candidates,
        desktop_windows,
        launch_discovery,
        process_visibility,
        operator_reported_vrchat=operator_reported_vrchat,
        operator_reported_voicevox=operator_reported_voicevox,
    )
    result = {
        "success": True,
        "created_at": int(time.time()),
        "summary": summary,
        "preflight_summary": {
            "ready": bool(readiness.get("ready")),
            "missing": list(readiness.get("missing") or []),
            "vrchat_process": checks.get("vrchat_process", {}),
            "voicevox": checks.get("voicevox", {}),
            "voicevox_synthesis": preflight.get("voicevox_synthesis", {}),
            "harness": checks.get("harness", {}),
            "audio_output_device": checks.get("audio_output_device", {}),
            "virtual_cable_route": preflight.get("audio", {}).get("virtual_cable_route", {}),
            "live_smoke_gate": preflight.get("live_smoke_gate", {}),
            "profile_path": profile_state.get("path"),
        },
        "operator_reported": {
            "vrchat": bool(operator_reported_vrchat),
            "voicevox": bool(operator_reported_voicevox),
        },
        "voicevox_candidates": voicevox_candidates,
        "local_ports": network,
        "desktop_windows": desktop_windows,
        "launch_discovery": launch_discovery,
        "process_visibility": process_visibility,
        "next_actions": _runtime_doctor_next_actions(
            preflight,
            voicevox_candidates,
            desktop_windows,
            launch_discovery,
            process_visibility,
            operator_reported_vrchat=operator_reported_vrchat,
            operator_reported_voicevox=operator_reported_voicevox,
        ),
        "commands": {
            **dict(preflight.get("commands") or {}),
            "runtime_doctor": _runtime_doctor_command(profile_state.get("path"), audio_output_device),
        },
        "safety": {
            "actuation_performed": False,
            "chatbox_sent": False,
            "speech_played": False,
            "avatar_parameters_written": False,
            "microphone_recorded": False,
            "websocket_opened": False,
        },
    }
    if output_path:
        path = Path(output_path).expanduser()
        result["output_path"] = str(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def wait_for_readiness(
    *,
    profile_path: str | Path | None = None,
    voicevox_url: str = DEFAULT_VOICEVOX_URL,
    harness_url: str = DEFAULT_HARNESS_URL,
    audio_output_device: str | None = None,
    require_harness: bool = False,
    queue_path: str | Path | None = None,
    include_audio_devices: bool = False,
    max_audio_devices: int = 20,
    timeout_sec: float = 120.0,
    interval_sec: float = 5.0,
    max_snapshots: int = 25,
    output_path: str | Path | None = None,
    _sleep=time.sleep,
    _clock=time.monotonic,
) -> dict[str, Any]:
    """Poll read-only readiness until ready or timeout."""
    timeout = max(0.0, float(timeout_sec))
    interval = max(0.1, float(interval_sec))
    snapshot_limit = max(1, int(max_snapshots))
    start = _clock()
    deadline = start + timeout
    snapshots: list[dict[str, Any]] = []
    attempts = 0
    final_bundle: dict[str, Any] | None = None

    while True:
        attempts += 1
        final_bundle = build_preflight_bundle(
            profile_path=profile_path,
            voicevox_url=voicevox_url,
            harness_url=harness_url,
            audio_output_device=audio_output_device,
            require_harness=require_harness,
            queue_path=queue_path,
            include_audio_devices=include_audio_devices,
            max_audio_devices=max_audio_devices,
        )
        summary = _readiness_wait_snapshot(final_bundle, attempt=attempts, elapsed_sec=_clock() - start)
        if len(snapshots) < snapshot_limit:
            snapshots.append(summary)
        elif snapshots:
            snapshots[-1] = summary

        if final_bundle.get("readiness", {}).get("ready"):
            status = "ready"
            break

        now = _clock()
        if now >= deadline:
            status = "timeout"
            break
        _sleep(min(interval, max(0.0, deadline - now)))

    final_summary = _readiness_wait_snapshot(final_bundle or {}, attempt=attempts, elapsed_sec=_clock() - start)
    result = {
        "success": True,
        "status": status,
        "ready": bool(final_bundle and final_bundle.get("readiness", {}).get("ready")),
        "attempts": attempts,
        "timeout_sec": timeout,
        "interval_sec": interval,
        "elapsed_sec": round(max(0.0, _clock() - start), 3),
        "snapshots": snapshots,
        "final_summary": final_summary,
        "final_preflight": final_bundle,
        "safety": {
            "actuation_performed": False,
            "chatbox_sent": False,
            "speech_played": False,
            "avatar_parameters_written": False,
            "microphone_recorded": False,
            "websocket_opened": False,
        },
    }
    if output_path:
        path = Path(output_path).expanduser()
        result["output_path"] = str(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def wait_for_readiness_then_tick(
    *,
    profile_path: str | Path | None = None,
    observations: list[dict[str, Any]] | None = None,
    voicevox_url: str = DEFAULT_VOICEVOX_URL,
    harness_url: str = DEFAULT_HARNESS_URL,
    audio_output_device: str | None = None,
    require_harness: bool = False,
    queue_path: str | Path | None = None,
    include_audio_devices: bool = False,
    max_audio_devices: int = 20,
    timeout_sec: float = 120.0,
    interval_sec: float = 5.0,
    max_snapshots: int = 25,
    persist_heartbeat: bool = True,
    allow_live_profile: bool = False,
    live_ack: str = "",
    emergency_stop: bool = False,
    output_path: str | Path | None = None,
    llm_call=None,
    _sleep=time.sleep,
    _clock=time.monotonic,
) -> dict[str, Any]:
    """Wait for readiness, then run one gated heartbeat tick only if ready."""
    wait_result = wait_for_readiness(
        profile_path=profile_path,
        voicevox_url=voicevox_url,
        harness_url=harness_url,
        audio_output_device=audio_output_device,
        require_harness=require_harness,
        queue_path=queue_path,
        include_audio_devices=include_audio_devices,
        max_audio_devices=max_audio_devices,
        timeout_sec=timeout_sec,
        interval_sec=interval_sec,
        max_snapshots=max_snapshots,
        _sleep=_sleep,
        _clock=_clock,
    )

    tick = None
    if wait_result.get("ready") or emergency_stop:
        tick = vrchat_autonomy_heartbeat_tick(
            profile_path=profile_path,
            observations=observations,
            voicevox_url=voicevox_url,
            harness_url=harness_url,
            audio_output_device=audio_output_device,
            require_harness=require_harness,
            persist_heartbeat=persist_heartbeat,
            tick_when_already_ready=True,
            allow_live_profile=allow_live_profile,
            live_ack=live_ack,
            emergency_stop=emergency_stop,
            llm_call=llm_call,
        )

    result = {
        "success": bool(tick.get("success")) if tick is not None else True,
        "code": _wait_tick_code(wait_result, tick),
        "wait": wait_result,
        "tick": tick,
        "ready": bool(wait_result.get("ready")),
        "dry_run": tick.get("dry_run") if isinstance(tick, dict) else None,
        "safety": (tick or wait_result).get("safety", _read_only_safety_flags()),
    }
    if output_path:
        path = Path(output_path).expanduser()
        result["output_path"] = str(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def list_audio_output_devices(*, max_devices: int = 20) -> dict[str, Any]:
    """List output-capable sounddevice devices without opening any stream."""
    try:
        import sounddevice as sd
    except Exception as exc:
        return {
            "success": False,
            "included": True,
            "error": "sounddevice_unavailable",
            "detail": str(exc),
            "devices": [],
        }
    try:
        devices = sd.query_devices()
        default = getattr(sd, "default", None)
        default_pair = list(getattr(default, "device", []) or []) if default is not None else []
    except Exception as exc:
        return {
            "success": False,
            "included": True,
            "error": "device_query_failed",
            "detail": str(exc),
            "devices": [],
        }

    default_output = default_pair[1] if len(default_pair) > 1 else None
    output_devices: list[dict[str, Any]] = []
    for index, device in enumerate(devices):
        max_output_channels = int(device.get("max_output_channels", 0) or 0)
        if max_output_channels <= 0:
            continue
        output_devices.append(
            {
                "index": index,
                "name": str(device.get("name", "")),
                "max_output_channels": max_output_channels,
                "default_output": index == default_output,
            }
        )
        if len(output_devices) >= max(0, int(max_devices)):
            break
    return {
        "success": True,
        "included": True,
        "devices": output_devices,
        "default_output_index": default_output,
    }


def check_virtual_cable_route(
    *,
    output_device: str | None = None,
    microphone_device: str | None = None,
) -> dict[str, Any]:
    """Verify virtual cable playback and microphone-side devices without recording."""
    output = (output_device or "").strip()
    microphone = (microphone_device or infer_virtual_cable_microphone_device(output)).strip()
    configured = bool(output or microphone)
    if not configured:
        return {
            "success": True,
            "included": True,
            "configured": False,
            "ok": None,
            "output_device": "",
            "microphone_device": "",
            "playback": {"ok": None, "matches": []},
            "microphone": {"ok": None, "matches": []},
            "safety": {"microphone_recorded": False, "speech_played": False},
        }

    try:
        import sounddevice as sd
    except Exception as exc:
        return {
            "success": False,
            "included": True,
            "configured": True,
            "ok": False,
            "error": "sounddevice_unavailable",
            "detail": str(exc),
            "output_device": output,
            "microphone_device": microphone,
            "playback": {"ok": False, "matches": []},
            "microphone": {"ok": False, "matches": []},
            "safety": {"microphone_recorded": False, "speech_played": False},
        }

    try:
        devices = sd.query_devices()
    except Exception as exc:
        return {
            "success": False,
            "included": True,
            "configured": True,
            "ok": False,
            "error": "device_query_failed",
            "detail": str(exc),
            "output_device": output,
            "microphone_device": microphone,
            "playback": {"ok": False, "matches": []},
            "microphone": {"ok": False, "matches": []},
            "safety": {"microphone_recorded": False, "speech_played": False},
        }

    playback_matches = _match_audio_devices(devices, output, channel_key="max_output_channels")
    microphone_matches = _match_audio_devices(devices, microphone, channel_key="max_input_channels")
    playback_ok = bool(playback_matches) if output else None
    microphone_ok = bool(microphone_matches) if microphone else None
    ok = bool((playback_ok is not False) and (microphone_ok is not False) and (playback_ok or microphone_ok))
    return {
        "success": True,
        "included": True,
        "configured": True,
        "ok": ok,
        "output_device": output,
        "microphone_device": microphone,
        "playback": {
            "ok": playback_ok,
            "matches": playback_matches[:5],
        },
        "microphone": {
            "ok": microphone_ok,
            "matches": microphone_matches[:5],
        },
        "safety": {"microphone_recorded": False, "speech_played": False},
    }


def probe_voicevox_synthesis(
    *,
    voicevox_url: str = DEFAULT_VOICEVOX_URL,
    text: str = DEFAULT_VOICEVOX_SYNTHESIS_TEXT,
    speaker: int = 8,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Verify VOICEVOX can synthesize a WAV without playback or recording."""
    base = (voicevox_url or DEFAULT_VOICEVOX_URL).rstrip("/")
    sample_text = (text or "").strip()
    safety = {
        "speech_played": False,
        "microphone_recorded": False,
    }
    if not sample_text:
        return {
            "success": False,
            "included": True,
            "ok": False,
            "url": base,
            "speaker": int(speaker),
            "error": "voicevox_synthesis_text_empty",
            "safety": safety,
        }

    try:
        query_response = requests.post(
            f"{base}/audio_query",
            params={"speaker": int(speaker), "text": sample_text},
            timeout=max(0.1, float(timeout)),
        )
        query_response.raise_for_status()
        query = query_response.json()
        synthesis_response = requests.post(
            f"{base}/synthesis",
            params={"speaker": int(speaker)},
            json=query,
            timeout=max(0.1, float(timeout)),
        )
        synthesis_response.raise_for_status()
        wav_bytes = synthesis_response.content or b""
    except requests.exceptions.RequestException as exc:
        return {
            "success": False,
            "included": True,
            "ok": False,
            "url": base,
            "speaker": int(speaker),
            "text_length": len(sample_text),
            "error": exc.__class__.__name__,
            "detail": str(exc),
            "safety": safety,
        }
    except Exception as exc:
        return {
            "success": False,
            "included": True,
            "ok": False,
            "url": base,
            "speaker": int(speaker),
            "text_length": len(sample_text),
            "error": "voicevox_synthesis_failed",
            "detail": str(exc),
            "safety": safety,
        }

    wav_header_ok = wav_bytes[:4] == b"RIFF" and wav_bytes[8:12] == b"WAVE"
    ok = bool(wav_bytes) and wav_header_ok
    return {
        "success": ok,
        "included": True,
        "ok": ok,
        "url": base,
        "speaker": int(speaker),
        "text_length": len(sample_text),
        "size_bytes": len(wav_bytes),
        "wav_header_ok": wav_header_ok,
        "played_audio": False,
        "microphone_recorded": False,
        "safety": safety,
    }


def infer_virtual_cable_microphone_device(output_device: str | None) -> str:
    """Infer the VRChat microphone-side cable name from a playback-side cable name."""
    text = (output_device or "").strip()
    if not text:
        return ""
    lowered = text.casefold()
    if "cable input" in lowered:
        return re.sub("(?i)cable input", "CABLE Output", text, count=1)
    if "cable in" in lowered:
        return re.sub("(?i)cable in", "CABLE Out", text, count=1)
    if "input" in lowered and "cable" in lowered:
        return re.sub("(?i)input", "Output", text, count=1)
    return ""


def _match_audio_devices(devices: Any, needle: str, *, channel_key: str) -> list[dict[str, Any]]:
    query = (needle or "").strip().casefold()
    if not query:
        return []
    matches: list[dict[str, Any]] = []
    for index, device in enumerate(devices):
        name = str(device.get("name", ""))
        channels = int(device.get(channel_key, 0) or 0)
        if channels <= 0 or query not in name.casefold():
            continue
        matches.append(
            {
                "index": index,
                "name": name,
                channel_key: channels,
            }
        )
    return matches


def _coerce_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _wait_tick_code(wait_result: dict[str, Any], tick: dict[str, Any] | None) -> str:
    if tick is not None:
        return "WAIT_READY_TICK_DONE" if tick.get("success") else "WAIT_READY_TICK_BLOCKED"
    if wait_result.get("status") == "timeout":
        return "WAIT_READY_TIMEOUT"
    return "WAIT_READY_NO_TICK"


def _read_only_safety_flags() -> dict[str, bool]:
    return {
        "actuation_performed": False,
        "chatbox_sent": False,
        "speech_played": False,
        "avatar_parameters_written": False,
        "microphone_recorded": False,
        "websocket_opened": False,
    }


def _readiness_wait_snapshot(preflight: dict[str, Any], *, attempt: int, elapsed_sec: float) -> dict[str, Any]:
    readiness = preflight.get("readiness", {})
    checks = readiness.get("checks", {})
    vrchat_process = checks.get("vrchat_process", {})
    voicevox = checks.get("voicevox", {})
    voicevox_process = voicevox.get("process", {})
    audio = checks.get("audio_output_device", {})
    live_gate = preflight.get("live_smoke_gate", {})
    return {
        "attempt": attempt,
        "elapsed_sec": round(max(0.0, elapsed_sec), 3),
        "ready": bool(readiness.get("ready")),
        "missing": list(readiness.get("missing") or []),
        "vrchat_process_phase": vrchat_process.get("phase"),
        "vrchat_process_diagnostic": vrchat_process.get("diagnostic"),
        "voicevox_ok": voicevox.get("ok"),
        "voicevox_process_phase": voicevox_process.get("phase"),
        "voicevox_process_diagnostic": voicevox_process.get("diagnostic"),
        "audio_output_ok": audio.get("ok"),
        "live_smoke_ready": bool(live_gate.get("ready_for_live_private_smoke")),
        "live_smoke_blocked_reasons": list(live_gate.get("blocked_reasons") or []),
    }


def _probe_voicevox_candidates(primary_url: str, *, probe_timeout: float = 1.0) -> list[dict[str, Any]]:
    seen: set[str] = set()
    results: list[dict[str, Any]] = []
    for url in (primary_url, *COMMON_VOICEVOX_URLS):
        normalized = str(url or "").rstrip("/")
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result = dict(probe_voicevox(normalized, timeout=max(0.1, float(probe_timeout))))
        result["configured_primary"] = normalized == str(primary_url or "").rstrip("/")
        results.append(result)
    return results


def _diagnostic_ports(voicevox_url: str, harness_url: str) -> list[int]:
    ports = {9000, 9001, 50021, 50031, 18794}
    for url in (voicevox_url, harness_url):
        port = _port_from_url(url)
        if port:
            ports.add(port)
    return sorted(ports)


def _port_from_url(url: str) -> int | None:
    try:
        from urllib.parse import urlparse

        parsed = urlparse(str(url))
        return parsed.port
    except ValueError:
        return None


def _local_port_snapshot(ports: list[int]) -> dict[str, Any]:
    wanted = {int(port) for port in ports}
    try:
        connections = psutil.net_connections(kind="inet")
    except Exception as exc:
        return {
            "success": False,
            "error": type(exc).__name__,
            "detail": str(exc),
            "ports_checked": sorted(wanted),
            "listeners": [],
        }
    listeners: list[dict[str, Any]] = []
    for conn in connections:
        local_port = _connection_port(conn.laddr)
        if local_port not in wanted:
            continue
        proto = "udp" if conn.type == socket.SOCK_DGRAM else "tcp"
        listeners.append(
            {
                "protocol": proto,
                "local_address": _connection_host(conn.laddr),
                "local_port": local_port,
                "status": str(getattr(conn, "status", "") or ""),
                "pid": conn.pid,
                "process": _process_name(conn.pid),
            }
        )
    listeners.sort(key=lambda item: (item["local_port"], item["protocol"], str(item["pid"])))
    return {
        "success": True,
        "ports_checked": sorted(wanted),
        "listeners": listeners[:50],
    }


def _relevant_window_snapshot(max_windows: int = 25) -> dict[str, Any]:
    """Return visible Windows desktop windows that look relevant to VRChat or VOICEVOX."""
    if platform.system() != "Windows":
        return {"success": True, "included": False, "windows": []}
    try:
        import ctypes
        from ctypes import wintypes
    except Exception as exc:
        return {
            "success": False,
            "included": True,
            "error": type(exc).__name__,
            "detail": str(exc),
            "windows": [],
        }

    user32 = ctypes.windll.user32
    windows: list[dict[str, Any]] = []

    enum_windows_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def callback(hwnd, _lparam):
        if len(windows) >= max(0, int(max_windows)):
            return False
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        title = str(buffer.value or "").strip()
        if not title:
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        process_name = _process_name(int(pid.value))
        haystack = f"{title} {process_name}".casefold()
        matched_terms = [term for term in RUNTIME_WINDOW_TERMS if term in haystack]
        if matched_terms:
            windows.append(
                {
                    "title": title,
                    "pid": int(pid.value),
                    "process": process_name,
                    "matched_terms": matched_terms,
                }
            )
        return True

    try:
        user32.EnumWindows(enum_windows_proc(callback), 0)
    except Exception as exc:
        return {
            "success": False,
            "included": True,
            "error": type(exc).__name__,
            "detail": str(exc),
            "windows": windows,
        }
    return {
        "success": True,
        "included": True,
        "windows": windows,
    }


def _runtime_launch_discovery() -> dict[str, Any]:
    """Find bounded, read-only launch/install clues for runtime prerequisites."""
    vrchat = _vrchat_launch_discovery()
    voicevox = _voicevox_launch_discovery()
    return {
        "success": bool(vrchat.get("success")) and bool(voicevox.get("success")),
        "included": True,
        "read_only": True,
        "platform": platform.system(),
        "vrchat": vrchat,
        "voicevox": voicevox,
    }


def _process_visibility_snapshot(max_processes: int = 25) -> dict[str, Any]:
    """Return bounded process/session diagnostics for runtime visibility mismatches."""
    relevant: list[dict[str, Any]] = []
    access_denied = 0
    inspected = 0
    current_pid = os.getpid()
    current_session = _process_session_id(current_pid)
    ignored_pids = _current_process_tree_pids()
    for proc in psutil.process_iter(["pid", "name", "exe", "cmdline", "username", "status", "ppid"]):
        try:
            info = dict(proc.info)
            inspected += 1
        except psutil.AccessDenied:
            access_denied += 1
            continue
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue

        pid = _coerce_pid(info.get("pid"))
        if pid in ignored_pids:
            continue
        name = str(info.get("name") or "")
        name_folded = name.casefold()
        cmdline = "" if name_folded in GENERIC_PROCESS_VISIBILITY_HOSTS else _join_cmdline(info.get("cmdline"))
        haystack = " ".join(
            str(part)
            for part in (
                name,
                info.get("exe") or "",
                cmdline,
            )
            if part
        ).casefold()
        matched_terms = [term for term in RUNTIME_PROCESS_TERMS if term in haystack]
        if not matched_terms:
            continue
        relevant.append(
            {
                "pid": pid,
                "ppid": _coerce_pid(info.get("ppid")),
                "name": name,
                "path": str(info.get("exe") or ""),
                "username": str(info.get("username") or ""),
                "status": str(info.get("status") or ""),
                "session_id": _process_session_id(pid),
                "matched_terms": matched_terms,
            }
        )
        if len(relevant) >= max(0, int(max_processes)):
            break

    return {
        "success": True,
        "read_only": True,
        "platform": platform.system(),
        "current": {
            "pid": current_pid,
            "ppid": os.getppid(),
            "username": _current_username(),
            "session_id": current_session,
            "is_admin": _is_current_process_admin(),
        },
        "inspected_processes": inspected,
        "access_denied": access_denied,
        "relevant_processes": relevant,
        "relevant_count": len(relevant),
    }


def _process_session_id(pid: int | None) -> int | None:
    if platform.system() != "Windows" or not pid:
        return None
    try:
        import ctypes
        from ctypes import wintypes

        session_id = wintypes.DWORD()
        ok = ctypes.windll.kernel32.ProcessIdToSessionId(int(pid), ctypes.byref(session_id))
        if not ok:
            return None
        return int(session_id.value)
    except Exception:
        return None


def _is_current_process_admin() -> bool | None:
    if platform.system() != "Windows":
        return None
    try:
        import ctypes

        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return None


def _current_username() -> str:
    try:
        return psutil.Process(os.getpid()).username()
    except Exception:
        return ""


def _current_process_tree_pids() -> set[int]:
    pids = {os.getpid()}
    try:
        current = psutil.Process(os.getpid())
        pids.update(proc.pid for proc in current.parents())
        pids.update(proc.pid for proc in current.children(recursive=True))
    except Exception:
        pass
    return pids


def _coerce_pid(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _join_cmdline(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(str(part) for part in value)
    if value is None:
        return ""
    return str(value)


def _vrchat_launch_discovery() -> dict[str, Any]:
    try:
        steam_roots = _steam_root_candidates()
        libraries = _steam_library_roots(steam_roots)
        installs = _vrchat_steam_installs(libraries)
        return {
            "success": True,
            "read_only": True,
            "steam_app_id": STEAM_VRCHAT_APP_ID,
            "launch_uri": f"steam://rungameid/{STEAM_VRCHAT_APP_ID}",
            "steam_roots": [str(path) for path in steam_roots],
            "steam_libraries": [str(path) for path in libraries],
            "installs": installs,
            "candidate_count": len(installs),
        }
    except Exception as exc:
        return {
            "success": False,
            "read_only": True,
            "error": type(exc).__name__,
            "detail": str(exc),
            "steam_app_id": STEAM_VRCHAT_APP_ID,
            "launch_uri": f"steam://rungameid/{STEAM_VRCHAT_APP_ID}",
            "steam_roots": [],
            "steam_libraries": [],
            "installs": [],
            "candidate_count": 0,
        }


def _voicevox_launch_discovery() -> dict[str, Any]:
    try:
        install_roots = _voicevox_install_roots()
        shortcut_roots = _voicevox_shortcut_roots()
        executables = _voicevox_launch_candidates_from_roots(install_roots)
        shortcuts = _voicevox_shortcut_candidates_from_roots(shortcut_roots)
        return {
            "success": True,
            "read_only": True,
            "install_roots": [str(path) for path in install_roots],
            "shortcut_roots": [str(path) for path in shortcut_roots],
            "executables": executables,
            "shortcuts": shortcuts,
            "candidate_count": len(executables) + len(shortcuts),
        }
    except Exception as exc:
        return {
            "success": False,
            "read_only": True,
            "error": type(exc).__name__,
            "detail": str(exc),
            "install_roots": [],
            "shortcut_roots": [],
            "executables": [],
            "shortcuts": [],
            "candidate_count": 0,
        }


def _steam_root_candidates() -> list[Path]:
    candidates: list[Path] = []
    candidates.extend(_steam_registry_paths())
    candidates.extend(
        [
            _env_child("ProgramFiles(x86)", "Steam"),
            _env_child("ProgramFiles", "Steam"),
            Path(r"C:\Program Files (x86)\Steam"),
            Path(r"C:\Program Files\Steam"),
        ]
    )
    return [path for path in _unique_paths(candidates) if _safe_is_dir(path)]


def _steam_registry_paths() -> list[Path]:
    if platform.system() != "Windows":
        return []
    try:
        import winreg
    except Exception:
        return []

    candidates: list[Path] = []
    registry_keys = (
        (winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Valve\Steam"),
    )
    for hive, key_path in registry_keys:
        try:
            with winreg.OpenKey(hive, key_path) as key:
                for value_name in ("SteamPath", "InstallPath"):
                    try:
                        value, _kind = winreg.QueryValueEx(key, value_name)
                    except OSError:
                        continue
                    if value:
                        candidates.append(Path(str(value)))
        except OSError:
            continue
    return candidates


def _steam_library_roots(steam_roots: list[Path]) -> list[Path]:
    libraries: list[Path] = list(steam_roots)
    for root in steam_roots:
        library_vdf = root / "steamapps" / "libraryfolders.vdf"
        for path in _read_vdf_paths(library_vdf):
            libraries.append(path)
    return [path for path in _unique_paths(libraries) if _safe_is_dir(path)]


def _read_vdf_paths(path: Path) -> list[Path]:
    text = _read_text_limited(path)
    if not text:
        return []
    paths: list[Path] = []
    for match in re.finditer(r'"path"\s+"([^"]+)"', text):
        raw = match.group(1).replace("\\\\", "\\")
        if raw:
            paths.append(Path(raw))
    return paths


def _vrchat_steam_installs(libraries: list[Path]) -> list[dict[str, Any]]:
    installs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for library in libraries:
        steamapps = library / "steamapps"
        manifest = steamapps / f"appmanifest_{STEAM_VRCHAT_APP_ID}.acf"
        manifest_found = _safe_is_file(manifest)
        install_dir_name = _read_vdf_value(manifest, "installdir") or "VRChat"
        exe = steamapps / "common" / install_dir_name / "VRChat.exe"
        if not _safe_is_file(exe):
            exe = steamapps / "common" / "VRChat" / "VRChat.exe"
        exe_found = _safe_is_file(exe)
        if not manifest_found and not exe_found:
            continue
        dedupe_key = str(exe).casefold()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        installs.append(
            {
                "library": str(library),
                "manifest": str(manifest),
                "manifest_found": manifest_found,
                "install_dir": str(exe.parent),
                "exe": str(exe),
                "exe_found": exe_found,
            }
        )
    return installs


def _read_vdf_value(path: Path, key: str) -> str:
    text = _read_text_limited(path)
    if not text:
        return ""
    match = re.search(rf'"{re.escape(key)}"\s+"([^"]+)"', text)
    return match.group(1).replace("\\\\", "\\") if match else ""


def _voicevox_install_roots() -> list[Path]:
    candidates = [
        _env_child("LOCALAPPDATA", "Programs", "VOICEVOX"),
        _env_child("LOCALAPPDATA", "VOICEVOX"),
        _env_child("ProgramFiles", "VOICEVOX"),
        _env_child("ProgramFiles(x86)", "VOICEVOX"),
        Path.home() / "AppData" / "Local" / "Programs" / "VOICEVOX",
        Path.home() / "AppData" / "Local" / "VOICEVOX",
    ]
    return [path for path in _unique_paths(candidates) if _safe_is_dir(path)]


def _voicevox_shortcut_roots() -> list[Path]:
    candidates = [
        _env_child("APPDATA", "Microsoft", "Windows", "Start Menu", "Programs"),
        _env_child("PROGRAMDATA", "Microsoft", "Windows", "Start Menu", "Programs"),
    ]
    return [path for path in _unique_paths(candidates) if _safe_is_dir(path)]


def _voicevox_launch_candidates_from_roots(roots: list[Path], max_depth: int = 2) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in roots:
        for path in _bounded_named_files(root, VOICEVOX_EXE_NAME, max_depth=max_depth):
            key = str(path).casefold()
            if key in seen:
                continue
            seen.add(key)
            candidates.append({"path": str(path), "source_root": str(root), "kind": "executable"})
    return candidates


def _voicevox_shortcut_candidates_from_roots(roots: list[Path], max_depth: int = 2) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in roots:
        for path in _bounded_files(root, max_depth=max_depth):
            if path.suffix.casefold() != ".lnk" or "voicevox" not in path.name.casefold():
                continue
            key = str(path).casefold()
            if key in seen:
                continue
            seen.add(key)
            candidates.append({"path": str(path), "source_root": str(root), "kind": "shortcut"})
    return candidates


def _bounded_named_files(root: Path, filename: str, *, max_depth: int) -> list[Path]:
    return [
        path
        for path in _bounded_files(root, max_depth=max_depth)
        if path.name.casefold() == filename.casefold()
    ]


def _bounded_files(root: Path, *, max_depth: int) -> list[Path]:
    files: list[Path] = []
    stack: list[tuple[Path, int]] = [(Path(root), 0)]
    limit = max(0, int(max_depth))
    while stack:
        current, depth = stack.pop()
        if not _safe_is_dir(current):
            continue
        try:
            children = list(current.iterdir())
        except OSError:
            continue
        for child in children:
            if _safe_is_file(child):
                files.append(child)
            elif depth < limit and _safe_is_dir(child):
                stack.append((child, depth + 1))
    return files


def _env_child(env_name: str, *parts: str) -> Path | None:
    value = os.environ.get(env_name)
    if not value:
        return None
    return Path(value).joinpath(*parts)


def _unique_paths(paths: list[Path | None]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        if path is None:
            continue
        try:
            resolved = Path(path).expanduser()
        except TypeError:
            continue
        key = str(resolved).casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(resolved)
    return unique


def _read_text_limited(path: Path, *, max_chars: int = 200_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except OSError:
        return ""


def _safe_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _safe_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _connection_port(address: Any) -> int | None:
    if not address:
        return None
    if hasattr(address, "port"):
        return int(address.port)
    if isinstance(address, tuple) and len(address) >= 2:
        return int(address[1])
    return None


def _connection_host(address: Any) -> str:
    if not address:
        return ""
    if hasattr(address, "ip"):
        return str(address.ip)
    if isinstance(address, tuple) and address:
        return str(address[0])
    return str(address)


def _process_name(pid: int | None) -> str:
    if not pid:
        return ""
    try:
        return psutil.Process(pid).name()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return ""


def _runtime_doctor_summary(
    preflight: dict[str, Any],
    voicevox_candidates: list[dict[str, Any]],
    desktop_windows: dict[str, Any],
    launch_discovery: dict[str, Any],
    process_visibility: dict[str, Any],
    *,
    operator_reported_vrchat: bool,
    operator_reported_voicevox: bool,
) -> dict[str, Any]:
    readiness = preflight.get("readiness", {})
    checks = readiness.get("checks", {})
    vrchat_ok = bool(checks.get("vrchat_process", {}).get("ok"))
    voicevox_ok = bool(checks.get("voicevox", {}).get("ok"))
    candidate_ok = [item for item in voicevox_candidates if item.get("ok")]
    vrchat_windows = _windows_matching(desktop_windows, "vrchat")
    voicevox_windows = _windows_matching(desktop_windows, "voicevox")
    vrchat_launch_candidates = _launch_candidate_count(launch_discovery, "vrchat")
    voicevox_launch_candidates = _launch_candidate_count(launch_discovery, "voicevox")
    mismatches: list[str] = []
    if operator_reported_vrchat and not vrchat_ok:
        mismatches.append("operator_reported_vrchat_but_process_not_visible")
    if operator_reported_voicevox and not voicevox_ok:
        mismatches.append("operator_reported_voicevox_but_engine_not_reachable")
    if candidate_ok and not voicevox_ok:
        mismatches.append("voicevox_reachable_on_nonconfigured_url")
    if vrchat_windows and not vrchat_ok:
        mismatches.append("vrchat_window_visible_but_expected_process_not_ready")
    if voicevox_windows and not voicevox_ok:
        mismatches.append("voicevox_window_visible_but_engine_not_reachable")
    status = "ready" if readiness.get("ready") else "blocked"
    if mismatches:
        status = "operator_mismatch"
    return {
        "status": status,
        "ready": bool(readiness.get("ready")),
        "missing": list(readiness.get("missing") or []),
        "operator_mismatches": mismatches,
        "voicevox_candidate_urls_ok": [str(item.get("url", "")) for item in candidate_ok],
        "relevant_window_titles": [str(window.get("title", "")) for window in [*vrchat_windows, *voicevox_windows]],
        "vrchat_launch_candidates": vrchat_launch_candidates,
        "voicevox_launch_candidates": voicevox_launch_candidates,
        "process_visibility_relevant_count": int(process_visibility.get("relevant_count") or 0),
        "process_visibility_access_denied": int(process_visibility.get("access_denied") or 0),
        "current_session_id": (process_visibility.get("current") or {}).get("session_id"),
    }


def _runtime_doctor_next_actions(
    preflight: dict[str, Any],
    voicevox_candidates: list[dict[str, Any]],
    desktop_windows: dict[str, Any],
    launch_discovery: dict[str, Any],
    process_visibility: dict[str, Any],
    *,
    operator_reported_vrchat: bool,
    operator_reported_voicevox: bool,
) -> list[str]:
    checks = preflight.get("readiness", {}).get("checks", {})
    vrchat = checks.get("vrchat_process", {})
    voicevox = checks.get("voicevox", {})
    voicevox_process = voicevox.get("process", {})
    audio = checks.get("audio_output_device", {})
    live_gate = preflight.get("live_smoke_gate", {})
    vrchat_windows = _windows_matching(desktop_windows, "vrchat")
    voicevox_windows = _windows_matching(desktop_windows, "voicevox")
    vrchat_launch = launch_discovery.get("vrchat", {}) if isinstance(launch_discovery, dict) else {}
    voicevox_launch = launch_discovery.get("voicevox", {}) if isinstance(launch_discovery, dict) else {}
    relevant_processes = process_visibility.get("relevant_processes", []) if isinstance(process_visibility, dict) else []
    current_session = (process_visibility.get("current") or {}).get("session_id") if isinstance(process_visibility, dict) else None
    actions: list[str] = []

    if not vrchat.get("ok"):
        phase = vrchat.get("phase")
        if vrchat_windows:
            actions.append("A VRChat-like window is visible, but VRChat.exe readiness is not satisfied; inspect desktop_windows for the process and wait until the official client is fully loaded.")
            if _launch_candidate_count(launch_discovery, "vrchat"):
                actions.append(_vrchat_launch_action(vrchat_launch))
            if not relevant_processes:
                visibility_action = _process_visibility_action(current_session)
                if visibility_action not in actions:
                    actions.append(visibility_action)
        elif operator_reported_vrchat:
            actions.append(
                "Confirm VRChat is running in this Windows user session and has finished loading past launcher or anti-cheat."
            )
            if _launch_candidate_count(launch_discovery, "vrchat"):
                actions.append(_vrchat_launch_action(vrchat_launch))
        elif _launch_candidate_count(launch_discovery, "vrchat"):
            actions.append(_vrchat_launch_action(vrchat_launch))
        elif phase == "steam_running_no_vrchat":
            actions.append("Start VRChat from Steam and wait until VRChat.exe is visible.")
        elif phase == "launching_or_blocked":
            actions.append("Wait for the launcher or anti-cheat phase to finish, then rerun the runtime doctor.")
        else:
            actions.append("Launch VRChat, enable OSC from the Action Menu, and rerun the runtime doctor.")

    if not voicevox.get("ok"):
        configured_url = str(voicevox.get("url") or DEFAULT_VOICEVOX_URL).rstrip("/")
        alternate_ok = [
            str(item.get("url", "")).rstrip("/")
            for item in voicevox_candidates
            if item.get("ok") and str(item.get("url", "")).rstrip("/") != configured_url
        ]
        if alternate_ok:
            actions.append(f"Set the profile VOICEVOX URL to {alternate_ok[0]} or restart Engine on {configured_url}.")
        elif voicevox_windows:
            actions.append("A VOICEVOX-like window is visible, but Engine /version is not reachable; start or enable the local Engine API.")
        elif operator_reported_voicevox:
            actions.append("Confirm the VOICEVOX Engine API is enabled and reachable at /version from this session.")
            if _launch_candidate_count(launch_discovery, "voicevox"):
                actions.append(_voicevox_launch_action(voicevox_launch))
            if not relevant_processes:
                visibility_action = _process_visibility_action(current_session)
                if visibility_action not in actions:
                    actions.append(visibility_action)
        elif voicevox_process.get("phase") == "ui_running_no_engine":
            actions.append("Start or wait for VOICEVOX Engine; the UI is visible but /version is not reachable.")
        elif _launch_candidate_count(launch_discovery, "voicevox"):
            actions.append(_voicevox_launch_action(voicevox_launch))
        else:
            actions.append("Start VOICEVOX Engine and confirm http://127.0.0.1:50021/version responds.")

    if audio.get("configured") and not audio.get("ok"):
        actions.append("Select or configure an output device matching the profile audio_output_device value.")
    elif not audio.get("configured"):
        actions.append("Configure the profile audio_output_device, for example CABLE Input, before live speech.")

    if preflight.get("readiness", {}).get("ready") and live_gate.get("blocked_reasons"):
        actions.append("Keep dry-run until private verification is complete, then arm the profile with the exact live ACK.")

    if not actions:
        actions.append("Runtime prerequisites look ready; proceed only with the gated private smoke workflow.")
    return actions


def _windows_matching(desktop_windows: dict[str, Any], term: str) -> list[dict[str, Any]]:
    folded = term.casefold()
    return [
        window
        for window in desktop_windows.get("windows", [])
        if folded in [str(item).casefold() for item in window.get("matched_terms", [])]
    ]


def _launch_candidate_count(launch_discovery: dict[str, Any], section: str) -> int:
    if not isinstance(launch_discovery, dict):
        return 0
    details = launch_discovery.get(section, {})
    if not isinstance(details, dict):
        return 0
    try:
        return int(details.get("candidate_count") or 0)
    except (TypeError, ValueError):
        return 0


def _vrchat_launch_action(vrchat_launch: dict[str, Any]) -> str:
    uri = str(vrchat_launch.get("launch_uri") or f"steam://rungameid/{STEAM_VRCHAT_APP_ID}")
    installs = [item for item in vrchat_launch.get("installs", []) if isinstance(item, dict)]
    exe = next((str(item.get("exe")) for item in installs if item.get("exe_found")), "")
    if exe:
        return f"Start VRChat using {uri} or the discovered executable {exe}, then rerun the runtime doctor."
    return f"Start VRChat using {uri}, then wait until VRChat.exe is visible and rerun the runtime doctor."


def _voicevox_launch_action(voicevox_launch: dict[str, Any]) -> str:
    executables = [item for item in voicevox_launch.get("executables", []) if isinstance(item, dict)]
    shortcuts = [item for item in voicevox_launch.get("shortcuts", []) if isinstance(item, dict)]
    candidate = next((str(item.get("path")) for item in [*executables, *shortcuts] if item.get("path")), "")
    if candidate:
        return f"Start VOICEVOX from discovered candidate {candidate}, then confirm /version is reachable."
    return "Start VOICEVOX Engine and confirm http://127.0.0.1:50021/version responds."


def _process_visibility_action(current_session: Any) -> str:
    if current_session is None:
        return "No relevant VRChat/VOICEVOX process is visible to this doctor; check Task Manager Details for VRChat.exe and VOICEVOX.exe in the same Windows user session."
    return f"No relevant VRChat/VOICEVOX process is visible to this doctor in Windows session {current_session}; check Task Manager Details for VRChat.exe and VOICEVOX.exe in that same session."


def _live_smoke_gate_summary(
    profile_state: dict[str, Any],
    readiness: dict[str, Any],
) -> dict[str, Any]:
    profile = profile_state.get("profile", {})
    blockers: list[str] = []
    if not readiness.get("ready"):
        blockers.append("readiness_not_ready")
    if not profile_state.get("success"):
        blockers.append("profile_invalid")
    if not bool(profile.get("enabled", False)):
        blockers.append("profile_disabled")
    if bool(profile.get("dry_run", True)):
        blockers.append("profile_dry_run_true")
    if profile.get("live_actuation_ack") != LIVE_ACTUATION_ACK:
        blockers.append("profile_live_ack_missing")
    return {
        "ready_for_live_private_smoke": not blockers,
        "blocked_reasons": blockers,
        "live_ack": LIVE_ACTUATION_ACK,
    }


def _recommended_commands(
    profile_path: str | None,
    audio_output_device: str | None,
    vrchat_microphone_device: str | None,
) -> dict[str, list[str]]:
    profile_arg = profile_path or "<Hermes home>\\config\\vrchat-autonomy-profile.json"
    audio_args = []
    if audio_output_device:
        audio_args = ["--audio-output-device", audio_output_device]
    microphone_args = []
    if vrchat_microphone_device:
        microphone_args = ["--vrchat-microphone-device", vrchat_microphone_device]
    return {
        "prepare_dry_run_profile": [
            "py",
            "-3.12",
            "scripts\\vrchat_profile.py",
            "--profile",
            profile_arg,
            *audio_args,
            *microphone_args,
        ],
        "read_only_preflight": [
            "py",
            "-3.12",
            "scripts\\vrchat_preflight.py",
            "--profile",
            profile_arg,
            *audio_args,
        ],
        "heartbeat_dry_run": [
            "py",
            "-3.12",
            "scripts\\vrchat_heartbeat_tick.py",
            "--profile",
            profile_arg,
            *audio_args,
        ],
        "wait_readiness": [
            "py",
            "-3.12",
            "scripts\\vrchat_wait_ready.py",
            "--profile",
            profile_arg,
            *audio_args,
            "--timeout-sec",
            "120",
            "--interval-sec",
            "5",
        ],
        "wait_readiness_then_tick": [
            "py",
            "-3.12",
            "scripts\\vrchat_wait_then_tick.py",
            "--profile",
            profile_arg,
            *audio_args,
            "--timeout-sec",
            "120",
            "--interval-sec",
            "5",
        ],
        "runtime_doctor": _runtime_doctor_command(profile_arg, audio_output_device),
        "private_smoke_dry_run": [
            "py",
            "-3.12",
            "scripts\\vrchat_private_smoke.py",
            "--profile",
            profile_arg,
            *audio_args,
        ],
        "private_smoke_prepare_live_gate": [
            "py",
            "-3.12",
            "scripts\\vrchat_private_smoke.py",
            "--prepare-only",
            "--profile",
            profile_arg,
            *audio_args,
            "--live-ack",
            LIVE_ACTUATION_ACK,
        ],
        "wait_readiness_then_private_smoke_prepare": [
            "py",
            "-3.12",
            "scripts\\vrchat_wait_then_private_smoke.py",
            "--profile",
            profile_arg,
            *audio_args,
            "--timeout-sec",
            "120",
            "--interval-sec",
            "5",
            "--live-ack",
            LIVE_ACTUATION_ACK,
        ],
        "print_live_ack": [
            "py",
            "-3.12",
            "scripts\\vrchat_profile.py",
            "--print-live-ack",
        ],
        "arm_live_profile_after_private_verification": [
            "py",
            "-3.12",
            "scripts\\vrchat_profile.py",
            "--profile",
            profile_arg,
            *audio_args,
            "--arm-live",
            "--live-ack",
            LIVE_ACTUATION_ACK,
        ],
        "private_smoke_live_after_private_verification": [
            "py",
            "-3.12",
            "scripts\\vrchat_private_smoke.py",
            "--profile",
            profile_arg,
            *audio_args,
            "--live",
            "--live-ack",
            LIVE_ACTUATION_ACK,
        ],
    }


def _runtime_doctor_command(profile_path: str | None, audio_output_device: str | None) -> list[str]:
    profile_arg = profile_path or "<Hermes home>\\config\\vrchat-autonomy-profile.json"
    audio_args = []
    if audio_output_device:
        audio_args = ["--audio-output-device", audio_output_device]
    return [
        "py",
        "-3.12",
        "scripts\\vrchat_runtime_doctor.py",
        "--profile",
        profile_arg,
        *audio_args,
    ]
