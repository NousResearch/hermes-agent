"""Voice / TTS / wake-word JSON-RPC handlers and their process-global state (one mic, one speaker
per process). Reaches server.py state through ``srv`` (method_ctx.py).
"""

from __future__ import annotations

import contextlib
import threading

from .contracts.base import Payload
from .contracts.events import VoiceStatusPayload, VoiceTranscriptPayload, WakeDetectedPayload
from .contracts.liveness import GatewayCapabilitiesResult, PingParams, PingResult
from .contracts.prompt_voice import (
    VoiceRecordParams, VoiceRecordResult, VoiceToggleParams, VoiceToggleResult, VoiceTtsParams,
    VoiceTtsResult, WakeControlParams, WakeFeedParams, WakeFeedResult, WakePauseResult,
    WakeResumeResult, WakeStartParams, WakeStartResult, WakeStatusParams, WakeStatusResult,
    WakeStopParams, WakeStopResult,
)
from .method_ctx import HandlerRegistry, bind_module
from typing import Optional
import os
import queue
import time
import logging
from tui_gateway.transport import Transport

logger = logging.getLogger("tui_gateway.server")  # siblings log as the gateway facade (operators and caplog filter on it)

_registry = HandlerRegistry()
method = _registry.method


# ── Voice state: HERMES_VOICE / HERMES_VOICE_TTS are runtime-only env flags (never config.yaml)
# so a prior session can't auto-start REC.

_voice_sid_lock = threading.Lock()
_voice_event_sid: str = ""
_voice_wake_owner: Optional[Transport] = None


def _caller_transport():
    return srv.current_transport() or srv._stdio_transport


def _voice_emit(event: str, payload: "Payload | None" = None) -> None:
    """Emit toward the session that most recently turned voice on (empty sid = active session)."""
    with srv._voice_sid_lock:
        sid = srv._voice_event_sid
    srv._emit(event, sid, payload)


def _resume_voice_wake() -> None:
    pass  # published state is written through srv
    with srv._voice_sid_lock:
        owner, srv._voice_wake_owner = srv._voice_wake_owner, None
    if owner is not None:
        srv._wake_resume_if_owner(owner)


def _voice_mode_enabled() -> bool:
    return os.environ.get("HERMES_VOICE", "").strip() == "1"


def _voice_tts_enabled() -> bool:
    return os.environ.get("HERMES_VOICE_TTS", "").strip() == "1"


def _end_voice_chat(*, stop_loop: bool, stop_tts: bool) -> None:
    """Flip voice + TTS off; optionally halt the continuous loop / cut live TTS (best-effort)."""
    os.environ["HERMES_VOICE"] = os.environ["HERMES_VOICE_TTS"] = "0"
    if stop_loop:
        with contextlib.suppress(Exception):
            from hermes_cli.voice import stop_continuous
            stop_continuous()
    if stop_tts:
        with contextlib.suppress(Exception):
            srv._tts_stream_stop(user_barge=False)


def _tts_lease_async(lease: str, active: bool) -> None:
    """Acquire/release a TTS lease off the RPC thread (acquiring warms a local engine; must not
    block the toggle's reply). Best-effort."""
    def _run():
        try:
            from tools.tts_tool_lifecycle import acquire_tts_lease, release_tts_lease
            (acquire_tts_lease if active else release_tts_lease)(lease)
        except Exception as e:
            logger.debug("voice: tts lease %s active=%s failed: %s", lease, active, e)
    threading.Thread(target=_run, name=f"tts-lease-{lease}", daemon=True).start()


def _running_sessions() -> list:
    with srv._sessions_lock:
        return [s for s in srv._sessions.values() if s.get("running")]


def _any_session_running() -> bool:
    """Voice busy-probe: silent captures during a long turn don't count toward the no-speech limit."""
    try:
        return bool(srv._running_sessions())
    except Exception:
        return False


# ── Streaming TTS: one pipeline per process (one speaker); a new turn's pipeline barges in on
# the previous. Token deltas feed a sentence-buffering consumer (stream_tts_to_speaker).

_tts_stream_lock = threading.Lock()
_tts_stream_state: Optional[dict] = None


def _tts_stream_begin() -> Optional[queue.Queue]:
    """Start a per-turn streaming TTS consumer; None when TTS can't stream."""
    if not srv._voice_tts_enabled():
        return None
    try:
        from tools.tts_tool import check_tts_requirements
        from tools.tts_tool_speaker import stream_tts_to_speaker
        if not check_tts_requirements():
            return None
    except Exception:
        return None
    srv._tts_stream_stop()
    text_queue: queue.Queue = queue.Queue()
    stop, done = threading.Event(), threading.Event()
    threading.Thread(target=stream_tts_to_speaker, args=(text_queue, stop, done), daemon=True).start()
    pass  # published state is written through srv
    with srv._tts_stream_lock:
        srv._tts_stream_state = {"stop": stop, "done": done}
    srv._arm_barge_listener_if_enabled()
    return text_queue


def _tts_stream_stop(user_barge: bool = True) -> None:
    """Cut in-flight streaming TTS. *user_barge* latches the interruption for the next turn's
    model note; ``False`` for mode changes (/voice off)."""
    pass  # published state is written through srv
    with srv._tts_stream_lock:
        state, srv._tts_stream_state = srv._tts_stream_state, None
    if state is None:
        return
    if user_barge and not state["done"].is_set():
        import traceback
        from tools.tts_streaming import mark_speech_interrupted
        logger.debug("TTS CUT: _tts_stream_stop(user_barge=True) — new turn or "
                     "interrupt cutting in-flight TTS\n%s", "".join(traceback.format_stack()))
        mark_speech_interrupted()
    state["stop"].set()
    with contextlib.suppress(Exception):
        from tools.voice_mode import stop_playback
        stop_playback()


# ── Full-duplex agent-turn listener: arms at utterance-submit, spans generation AND playback
# (per-playback monitors were deaf during generation and mis-calibrated against speaker bleed),
# disarms when no session runs, no TTS is pending, and no audio flows. _fd_speak_pipelines holds
# (stop, done) pairs of fallback whole-reply speak paths: the listener cuts their private stop
# events too, and keeps listening while any is still speaking.

_fd_listener_lock = threading.Lock()
_fd_listener_active = False
_fd_speak_pipelines: "set[tuple[threading.Event, threading.Event]]" = set()


def _arm_full_duplex_listener() -> None:
    """Arm the process-global full-duplex listener (idempotent — one mic)."""
    pass  # published state is written through srv
    with srv._fd_listener_lock:
        if srv._fd_listener_active:
            return
        srv._fd_listener_active = True
    threading.Thread(target=srv._full_duplex_listener, daemon=True, name="voice-full-duplex").start()


def _arm_barge_listener_if_enabled() -> None:
    """Arm the listener when voice mode is on and ``voice.barge_in`` isn't disabled."""
    if srv._voice_mode_enabled() and srv._voice_cfg_dict().get("barge_in", True):
        srv._arm_full_duplex_listener()


def _fd_tts_pending() -> bool:
    """True while any TTS (streaming pipeline or fallback speak) is unfinished."""
    with srv._tts_stream_lock:
        state = srv._tts_stream_state
    with srv._fd_listener_lock:
        pending = ([state["done"]] if state is not None else []) + [done for _stop, done in srv._fd_speak_pipelines]
    return any(not done.is_set() for done in pending)


def _full_duplex_listener() -> None:
    """Mic live from utterance-submit to turn-complete; a trip transcribes -> ``voice.transcript``."""
    pass  # published state is written through srv
    try:
        from tools.voice_mode import (full_duplex_listen, is_audio_output_active,
                                      transcribe_recording)

        def _should_stop() -> bool:
            return not srv._voice_mode_enabled() or not (
                srv._any_session_running() or srv._fd_tts_pending() or is_audio_output_active())
        tripped = threading.Event()

        def _on_trigger(phase: str) -> None:
            tripped.set()
            srv._fd_trip(phase)
        mult, grace_ms = srv._fd_barge_params(srv._voice_cfg_dict())
        wav_path = full_duplex_listen(_should_stop, is_playing=is_audio_output_active,
                                      on_trigger=_on_trigger, multiplier=mult or None,
                                      grace_ms=grace_ms)
        if not (wav_path and tripped.is_set()):
            return
        try:
            result = transcribe_recording(wav_path)
            if result.get("success") and (result.get("transcript") or "").strip():
                srv._deliver_fd_transcript(result["transcript"].strip())
        finally:
            with contextlib.suppress(OSError):
                os.unlink(wav_path)
    except Exception as e:
        logger.debug("full-duplex listener failed: %s", e)
    finally:
        with srv._fd_listener_lock:
            srv._fd_listener_active = False


def _fd_barge_params(cfg: dict) -> tuple[float, int]:
    """``(threshold multiplier, grace ms)`` from the voice config; malformed -> defaults."""
    def num(conv, key, default):
        try:
            return conv(cfg.get(key, default))
        except (TypeError, ValueError):
            return conv(default)
    mult = num(lambda v: float(v or 0), "barge_in_threshold_multiplier", 0)
    return mult, max(0, num(lambda v: int(float(v) * 1000), "barge_in_grace_seconds", 0.5))


def _fd_trip(phase: str) -> None:
    """Listener tripped: latch the interruption, cut TTS FIRST (so a stale reply can never
    speak), and during generation also interrupt every running turn (the ``agent.interrupt()``
    seam ``session.interrupt`` uses)."""
    from tools.tts_streaming import mark_speech_interrupted
    from tools.voice_mode import stop_playback
    mark_speech_interrupted()
    if phase == "playback":
        logger.debug("TTS CUT: full-duplex listener tripped during playback")
    else:
        logger.debug("full-duplex listener tripped during generation — "
                     "interrupting running turn(s)")
    # Cut streaming TTS, every fallback speak pipeline, and the file player.
    srv._tts_stream_stop(user_barge=True)
    with srv._fd_listener_lock:
        for _stop, _done in srv._fd_speak_pipelines:
            _stop.set()
    stop_playback()
    if phase != "playback":
        try:
            for s in srv._running_sessions():
                agent = s.get("agent")
                if agent is not None and hasattr(agent, "interrupt"):
                    with contextlib.suppress(Exception):
                        agent.interrupt()
        except Exception as e:
            logger.debug("voice interjection interrupt failed: %s", e)
    srv._voice_emit("voice.interrupted")


def _deliver_fd_transcript(text: str) -> None:
    """Emit the captured interjection; a bare stop phrase also ends the voice chat. The stop
    check must never break delivery (stubbed voice_mode in tests, partial installs)."""
    try:
        from tools.voice_mode import is_voice_stop_phrase
        is_stop = is_voice_stop_phrase(text)
    except Exception:
        is_stop = False
    if is_stop:  # turn already interrupted / TTS cut at trip time; now end the chat
        srv._end_voice_chat(stop_loop=True, stop_tts=False)
    srv._voice_emit(
        "voice.transcript", VoiceTranscriptPayload(stop_phrase=True, text=text) if is_stop
        else VoiceTranscriptPayload(text=text))


def _speak_text_with_barge(text: str) -> None:
    """speak_text registered in ``_fd_speak_pipelines`` so the listener can cut it / waits for it."""
    from hermes_cli.voice import speak_text
    stop, done = threading.Event(), threading.Event()
    with srv._fd_listener_lock:
        srv._fd_speak_pipelines.add((stop, done))

    def _speak():
        try:
            speak_text(text, stop)
        except TypeError:  # older wrapper without the stop_event parameter
            speak_text(text)
        finally:
            done.set()
            with srv._fd_listener_lock:
                srv._fd_speak_pipelines.discard((stop, done))
    threading.Thread(target=_speak, daemon=True).start()
    srv._arm_barge_listener_if_enabled()


def _voice_cfg_dict() -> dict:
    """Shape-safe ``voice:`` block (no deep-merged defaults: any YAML shape possible; bad → {}).

    ``_load_cfg()`` does not deep-merge DEFAULT_CONFIG, so both the root AND ``voice`` may be any YAML
    scalar / list / None. A hand-edit like ``voice: true`` or a malformed top-level config that parses to a
    scalar would otherwise break ``.get("…")`` and take every ``voice.*`` branch down with it (Copilot
    round-3..7 review on 19835). Coerce through ``isinstance`` at every level so malformed config falls back
    to an empty dict instead of crashing /voice. See #19835.
    """
    cfg = srv._load_cfg()
    voice_cfg = cfg.get("voice") if isinstance(cfg, dict) else None
    return voice_cfg if isinstance(voice_cfg, dict) else {}


def _voice_cfg_number(value, default):
    """Numeric config value, else *default*; bool excluded (``silence_threshold: true`` ≠ 1)."""
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else default


def _voice_status_payload(**extra) -> dict:
    """``{enabled, record_key, tts, **extra}``: record_key (default ``ctrl+b``) rides every voice.toggle
    branch so a tts toggle never resets a custom binding."""
    record_key = srv._voice_cfg_dict().get("record_key")
    record_key = record_key if isinstance(record_key, str) and record_key else "ctrl+b"
    return {"enabled": srv._voice_mode_enabled(), "record_key": record_key, "tts": srv._voice_tts_enabled(), **extra}


# ── Wake word ("Hey Hermes"): process-global detector (one mic). The first eligible transport
# to call wake.start owns it until stop, disconnect, or stream failure; on detection we emit
# wake.detected and the client opens a session + its own capture. The detector yields the mic
# to voice.record (pause/resume) and to the desktop's browser mic (wake.pause/resume RPCs).
_wake_lock = threading.Lock()
_wake_owner_transport: Optional[Transport] = None
_wake_owner_surface = ""


def _wake_owner_snapshot():
    with srv._wake_lock:
        return srv._wake_owner_transport, srv._wake_owner_surface


def _release_wake_for_transport(transport: Transport) -> bool:
    """Release the wake lease iff ``transport`` is the current gateway owner."""
    pass  # published state is written through srv
    with srv._wake_lock:
        if srv._wake_owner_transport is not transport:
            return False
        srv._wake_owner_transport, srv._wake_owner_surface = None, ""
    try:
        from tools.wake_word import stop_listening
        stop_listening(owner=transport)
    except Exception as e:
        logger.debug("wake stop failed: %s", e)
    return True


def _release_gateway_wake_owner() -> bool:
    owner, _surface = srv._wake_owner_snapshot()
    return owner is not None and srv._release_wake_for_transport(owner)


_wake_resume_retry_lock = threading.Lock()
_wake_resume_retry_active = False


def _wake_resume_if_owner(owner: Transport, *, retry_seconds: float = 15.0,
                          retry_interval: float = 1.0) -> bool:
    """Resume the wake detector for ``owner``, self-healing a busy microphone: reopening right after
    a voice turn can fail while the device is still being released (browser WebRTC tracks release
    async), so an exception retries in a background thread until it sticks, the lease changes hands,
    or ``retry_seconds`` elapses. ``False`` (lease gone / other owner) is final — never retried."""
    from tools.wake_word import resume_listening
    try:
        return resume_listening(owner=owner)
    except Exception as e:
        logger.debug("wake resume failed (will retry): %s", e)
    pass  # published state is written through srv
    with srv._wake_resume_retry_lock:
        if srv._wake_resume_retry_active:
            return False
        srv._wake_resume_retry_active = True

    def _retry() -> None:
        pass  # published state is written through srv
        deadline = time.monotonic() + retry_seconds
        try:
            while time.monotonic() < deadline:
                time.sleep(retry_interval)
                with contextlib.suppress(Exception):
                    if resume_listening(owner=owner):
                        logger.info("wake: detector resumed after retry")
                    return  # False — detector gone or lease moved: stop, don't fight it.
            logger.warning("wake: could not resume detector after voice turn "
                           "(microphone still busy?) — toggle the wake word to re-arm")
        finally:
            with srv._wake_resume_retry_lock:
                srv._wake_resume_retry_active = False
    threading.Thread(target=_retry, daemon=True, name="wake-resume-retry").start()
    return False


def _persist_wake_enabled(enabled: bool) -> bool:
    """Write ``wake_word.enabled``; only for explicit gestures (ear toggle, /wake on|off)."""
    try:
        from cli import save_config_value
        return bool(save_config_value("wake_word.enabled", enabled))
    except Exception as e:
        logger.warning("wake: failed to persist wake_word.enabled=%s: %s", enabled, e)
        return False


def _owner_result(result_type, field: str, ok: bool, **extra):
    """Build an owner-gated wake result with a typed reason."""
    return result_type(**{field: ok, "reason": None if ok else "not_owner", **extra})


def _frame_fields(frame: dict) -> dict:
    return {"sample_rate": frame.get("sample_rate", 16000), "frame_length": frame.get("frame_length", 1280)}


def _wake_probe(cfg: dict, client_capture: bool | None, surface: str) -> tuple[str, dict]:
    """``(capture_mode, requirements)``; capture stamped so the probe matches what would arm.
    Desktop (gui) prefers client capture (Mac mic → wake.feed PCM); CLI/TUI stay local."""
    from tools.wake_word import check_wake_word_requirements, resolve_capture_mode
    prefer_client = surface in ("gui", "desktop") or bool(client_capture)
    capture_mode = resolve_capture_mode(cfg, prefer_client=prefer_client)
    return capture_mode, check_wake_word_requirements({**cfg, "capture": capture_mode})


def _wake_detect_handler(transport, sid: str, phrase: str, new_session: bool):
    """On-detect callback: pause, verify ownership, emit ``wake.detected`` on the owner's transport."""
    def _on_detect() -> None:
        from tools.wake_word import get_last_match, owns_listener, pause_listening
        if not pause_listening(owner=transport) or not owns_listener(transport):
            return
        if srv._transport_is_dead(transport):
            srv._release_wake_for_transport(transport)
            return
        # Multi-phrase engines report WHICH phrase/profile fired; single-phrase engines fall back.
        matched_phrase, matched_profile = get_last_match() or (phrase, "")
        logger.info("wake.detected: emitting to sid=%r (transport=%s, profile=%r)",
                    sid, type(transport).__name__, matched_profile)
        token = srv.bind_transport(transport)
        try:
            srv._emit(
                "wake.detected", sid,
                WakeDetectedPayload(
                    phrase=matched_phrase or phrase, profile=matched_profile or None,
                    start_new_session=new_session))
        finally:
            srv.reset_transport(token)
    return _on_detect


@method("gateway.capabilities")
def _(rid, params: PingParams) -> GatewayCapabilitiesResult | dict:
    """Capabilities are sourced from the enforcing module, never config."""
    from hermes_cli.active_sessions import PER_SESSION_EXCLUSIVE_SUBMIT
    return GatewayCapabilitiesResult(per_session_exclusive_submit=bool(PER_SESSION_EXCLUSIVE_SUBMIT))


@method("ping")
def _(rid, params: PingParams) -> PingResult | dict:
    """Cheapest liveness probe, answered on the WS reader thread."""
    return PingResult(pong=True)


@method("wake.start")
def _(rid, params: WakeStartParams) -> WakeStartResult | dict:
    pass  # published state is written through srv
    surface = (params.surface or "auto").strip().lower()
    transport = srv._caller_transport()

    def refused(reason: str, **extra) -> WakeStartResult:
        return WakeStartResult(started=False, reason=reason, **extra)

    try:
        from tools.wake_word import (
            WakeWordInUse, detector_frame_info, load_wake_word_config, owns_listener,
            start_listening, wake_phrase, wake_surface_enabled)
    except Exception as e:
        return srv._err(rid, 5026, f"wake module unavailable: {e}")
    cfg = load_wake_word_config()
    capture_mode, reqs = srv._wake_probe(cfg, params.client_capture, surface)
    if not reqs["available"]:
        logger.warning("wake.start(%s): not available — %s", surface, reqs.get("hint"))
        return refused("unavailable", hint=reqs.get("hint") or "", capture=capture_mode)
    enabled_persisted = bool(params.persist) and not cfg.get("enabled") and srv._persist_wake_enabled(True)
    if enabled_persisted:
        cfg = {**cfg, "enabled": True}
    if not wake_surface_enabled(surface, cfg):
        reason = "disabled" if not cfg.get("enabled") else "disabled_for_surface"
        logger.info("wake.start(%s): %s (enabled=%s, surface=%s)",
                    surface, reason, cfg.get("enabled"), cfg.get("surface"))
        return refused(reason)
    existing_owner, existing_surface = srv._wake_owner_snapshot()
    if existing_owner is not None and (srv._transport_is_dead(existing_owner) or not owns_listener(existing_owner)):
        srv._release_wake_for_transport(existing_owner)
        existing_owner, existing_surface = None, ""
    if existing_owner is not None and existing_owner is not transport:
        return refused("owned", owner_surface=existing_surface)
    try:
        on_detect = srv._wake_detect_handler(transport, params.session_id or "", wake_phrase(cfg),
                                         bool(cfg.get("start_new_session", True)))
        start_listening(on_detect, owner=transport, config=cfg, external_audio=capture_mode == "client")
    except WakeWordInUse:
        return refused("owned", owner_surface=existing_surface or None)
    except Exception as e:
        logger.warning("wake.start(%s): failed to start listener: %s", surface, e)
        return srv._err(rid, 5026, str(e))
    with srv._wake_lock:
        srv._wake_owner_transport, srv._wake_owner_surface = transport, surface
    frame = detector_frame_info()
    return WakeStartResult(
        started=True, phrase=reqs["phrase"], provider=reqs["provider"], owner_surface=surface,
        enabled_persisted=enabled_persisted, capture=capture_mode, **srv._frame_fields(frame))


@method("wake.stop")
def _(rid, params: WakeStopParams) -> WakeStopResult | dict:
    stopped = srv._release_wake_for_transport(srv._caller_transport())
    disabled_persisted = False
    if params.persist:
        try:
            from tools.wake_word import load_wake_word_config
            currently_enabled = bool(load_wake_word_config().get("enabled"))
        except Exception:
            currently_enabled = True
        disabled_persisted = currently_enabled and srv._persist_wake_enabled(False)
    return srv._owner_result(WakeStopResult, "stopped", stopped, disabled_persisted=disabled_persisted)


@method("wake.pause")
def _(rid, params: WakeControlParams) -> WakePauseResult | dict:
    try:
        from tools.wake_word import pause_listening
        paused = pause_listening(owner=srv._caller_transport())
        logger.info("wake.pause: detector paused=%s", paused)
    except Exception as e:
        logger.debug("wake.pause failed: %s", e)
        paused = False
    return srv._owner_result(WakePauseResult, "paused", paused)


@method("wake.resume")
def _(rid, params: WakeControlParams) -> WakeResumeResult | dict:
    resumed = srv._wake_resume_if_owner(srv._caller_transport())
    logger.info("wake.resume: detector resumed=%s", resumed)
    return srv._owner_result(WakeResumeResult, "resumed", resumed)


@method("wake.status")
def _(rid, params: WakeStatusParams) -> WakeStatusResult | dict:
    try:
        from tools.wake_word import (
            audio_is_silent, detector_frame_info, get_input_device_status, is_listening,
            load_wake_word_config, owns_listener, silent_audio_hint)
        cfg = load_wake_word_config()
        probe_capture, reqs = srv._wake_probe(cfg, params.client_capture, (params.surface or "").strip().lower())
        owner, owner_surface = srv._wake_owner_snapshot()
        owned_by_caller = owns_listener(srv._caller_transport())
        listening = owned_by_caller and is_listening()
        silent = listening and audio_is_silent()
        input_device = get_input_device_status(cfg)
        hint = reqs.get("hint", "")
        if input_device.get("error") and not hint:
            hint = f"Wake-word input device could not be resolved: {input_device['error']}"
        if silent and not hint:
            hint = silent_audio_hint(input_device)
        frame = detector_frame_info()
        if owned_by_caller and (frame.get("external_audio") or listening):
            capture = "client" if frame.get("external_audio") else "local"
        else:
            capture = probe_capture or reqs.get("capture") or str(cfg.get("capture") or "auto")
        return WakeStatusResult(
            listening=listening, owned_by_caller=owned_by_caller,
            owner_surface=owner_surface if owner is not None else None,
            phrase=reqs["phrase"], provider=reqs["provider"],
            configured_surface=str(cfg.get("surface") or "auto"), input_device=input_device,
            available=reqs["available"], hint=hint, enabled=bool(cfg.get("enabled")),
            audio_silent=silent, capture=capture,
            local_input_available=bool(reqs.get("local_input_available")), **srv._frame_fields(frame))
    except Exception as e:
        return srv._err(rid, 5026, str(e))


@method("wake.feed")
def _(rid, params: WakeFeedParams) -> WakeFeedResult | dict:
    raw_b64 = params.pcm or params.pcm_b64 or ""
    if not raw_b64.strip():
        return srv._err(rid, 4001, "wake.feed requires base64 pcm")
    import base64
    try:
        pcm = base64.b64decode(raw_b64, validate=False)
    except Exception as e:
        return srv._err(rid, 4001, f"invalid base64 pcm: {e}")
    if not pcm:
        return WakeFeedResult(fed=False, reason="empty")
    if len(pcm) > 64000:
        return srv._err(rid, 4001, "pcm frame too large")
    if params.sample_rate is not None and params.sample_rate not in (0, 16000):
        return srv._err(rid, 4001, "wake.feed only accepts 16 kHz PCM")
    try:
        from tools.wake_word import feed_audio
        ok = feed_audio(owner=srv._caller_transport(), pcm_int16=pcm)
    except Exception as e:
        logger.debug("wake.feed failed: %s", e)
        return srv._err(rid, 5026, str(e))
    return srv._owner_result(WakeFeedResult, "fed", bool(ok))


def _voice_toggle_status(rid, params: VoiceToggleParams) -> VoiceToggleResult:
    # Mirrors CLI _show_voice_status: STT/TTS availability tells the user WHY voice isn't
    # working; record_key lets the TUI bind and display the shortcut.
    payload = srv._voice_status_payload()
    try:
        from tools.voice_mode import check_voice_requirements
        reqs = check_voice_requirements()
        payload.update({k: bool(reqs.get(k)) for k in ("available", "audio_available", "stt_available")},
                       details=reqs.get("details") or "")
    except Exception as e:
        # Optional transcription deps — /voice status must always answer.
        logger.warning("voice.toggle status: requirements probe failed: %s", e)
    return VoiceToggleResult(**payload)


def _voice_toggle_mode(rid, params: VoiceToggleParams) -> VoiceToggleResult:
    enabled = params.action == "on"
    os.environ["HERMES_VOICE"] = "1" if enabled else "0"
    stop_hint = ""
    if enabled:
        # Spoken-stop hint for the client; sourced from voice.stop_phrases, empty when disabled.
        with contextlib.suppress(Exception):
            from tools.voice_mode_transcript import voice_stop_hint
            stop_hint = voice_stop_hint()
        # Speech output already on → warm the engine now, not on the first reply.
        if srv._voice_tts_enabled():
            srv._tts_lease_async("tui:voice-tts", True)
    else:
        # The continuous loop holds the microphone; tear it down with the mode.
        try:
            from hermes_cli.voice import stop_continuous
            stop_continuous()
        except ImportError:
            pass
        except Exception as e:
            logger.warning("voice: stop_continuous failed during toggle off: %s", e)
        srv._set_voice_tts(False)  # TTS is toggled independently later
    return VoiceToggleResult(**srv._voice_status_payload(stop_hint=stop_hint))


def _set_voice_tts(on: bool) -> None:
    """Flip TTS; off silences live speech. The lease pre-loads the engine (on) / releases it (off)."""
    os.environ["HERMES_VOICE_TTS"] = "1" if on else "0"
    if not on:
        srv._tts_stream_stop(user_barge=False)
    srv._tts_lease_async("tui:voice-tts", on)


def _voice_toggle_tts(rid, params: VoiceToggleParams) -> VoiceToggleResult:
    if not srv._voice_mode_enabled():
        return srv._err(rid, 4014, "enable voice mode first: /voice on")
    srv._set_voice_tts(not srv._voice_tts_enabled())
    return VoiceToggleResult(**srv._voice_status_payload())


_VOICE_TOGGLE_ACTIONS = {
    "status": _voice_toggle_status, "on": _voice_toggle_mode, "off": _voice_toggle_mode,
    "tts": _voice_toggle_tts}


@method("voice.toggle")
def _(rid, params: VoiceToggleParams) -> VoiceToggleResult | dict:
    """CLI parity for ``/voice``: report, flip voice mode, or toggle speech output."""
    handler = srv._VOICE_TOGGLE_ACTIONS.get(params.action)
    if handler is None:
        return srv._err(rid, 4013, f"unknown voice action: {params.action}")
    return handler(rid, params)


# voice.record callbacks: each terminal capture event resumes the wake detector so wake-triggered
# and manual captures coexist.
def _vr_transcript(payload: VoiceTranscriptPayload) -> None:
    srv._voice_emit("voice.transcript", payload)
    srv._resume_voice_wake()


def _vr_on_stop_phrase(text: str) -> None:
    # A spoken bare stop phrase ends voice chat instead of becoming a no-speech timeout.
    srv._end_voice_chat(stop_loop=False, stop_tts=True)
    srv._vr_transcript(VoiceTranscriptPayload(stop_phrase=True, text=text))


def _vr_on_status(state) -> None:
    srv._voice_emit("voice.status", VoiceStatusPayload(state=state))
    if state == "idle":
        srv._resume_voice_wake()


@method("voice.record")
def _(rid, params: VoiceRecordParams) -> VoiceRecordResult | dict:
    action = params.action
    wake_paused = False
    transport = srv._caller_transport()
    wake_owner, _surface = srv._wake_owner_snapshot()
    if wake_owner is not None and wake_owner is not transport:
        return VoiceRecordResult(status="busy", reason="wake_owned")
    try:
        pass  # published state is written through srv
        if action == "start" and not srv._voice_mode_enabled():
            return srv._err(rid, 4015, "voice mode is off — enable with /voice on")
        with srv._voice_sid_lock:
            srv._voice_event_sid = params.session_id or srv._voice_event_sid
        if action == "stop":
            from hermes_cli.voice import stop_continuous
            stop_continuous(force_transcribe=True)
            srv._resume_voice_wake()
            return VoiceRecordResult(status="stopped")
        from hermes_cli.voice import start_continuous
        with contextlib.suppress(Exception):
            from hermes_cli.voice import set_voice_busy_probe
            set_voice_busy_probe(srv._any_session_running)
        voice_cfg = srv._voice_cfg_dict()
        max_rec = srv._voice_cfg_number(voice_cfg.get("max_recording_seconds"), 120.0)
        with contextlib.suppress(Exception):
            from tools.wake_word import pause_listening
            wake_paused = pause_listening(owner=transport)
        if wake_paused:
            with srv._voice_sid_lock:
                srv._voice_wake_owner = transport
        started = start_continuous(
            on_transcript=lambda text: srv._vr_transcript(VoiceTranscriptPayload(text=text)), on_status=srv._vr_on_status,
            on_silent_limit=lambda: srv._vr_transcript(VoiceTranscriptPayload(no_speech_limit=True)),
            silence_threshold=srv._voice_cfg_number(voice_cfg.get("silence_threshold"), 200),
            silence_duration=srv._voice_cfg_number(voice_cfg.get("silence_duration"), 3.0),
            auto_restart=False, max_recording_seconds=max_rec if max_rec > 0 else 0.0,
            on_stop_phrase=srv._vr_on_stop_phrase)
        if started is False:
            srv._resume_voice_wake()
        return VoiceRecordResult(status="busy" if started is False else "recording")
    except Exception as e:
        if wake_paused or action == "stop":
            srv._resume_voice_wake()
        if isinstance(e, ImportError):
            return srv._err(rid, 5025, "voice module not available — install audio dependencies")
        return srv._err(rid, 5025, str(e))


@method("voice.tts")
def _(rid, params: VoiceTtsParams) -> VoiceTtsResult | dict:
    text = params.text
    if not text:
        return srv._err(rid, 4020, "text required")
    try:
        import hermes_cli.voice  # noqa: F401  (a missing module must answer 5026, not die in a thread)
    except Exception as e:
        return srv._err(rid, 5026, "voice module not available" if isinstance(e, ImportError) else str(e))
    threading.Thread(target=srv._speak_text_with_barge, args=(text,), daemon=True).start()
    return VoiceTtsResult(status="speaking")


def register(server) -> None:
    bind_module(globals(), server, skip=("_",))

# Bound last, after every definition, so importing this module first (tests, the gateway process)
# lets server.py's own tail import see a complete module — the same tail-import idiom server.py uses.
from tui_gateway import server as srv  # noqa: E402
