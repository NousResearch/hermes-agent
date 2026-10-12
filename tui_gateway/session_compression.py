"""Live compression: config hot-reload onto a running agent, pending model switch apply, /compress
(CompressionLockHeld when a turn holds the lock), session-key sync after compress. Bodies are rebound
onto server.py's globals (method_ctx.bind_module) and reference them bare."""

from __future__ import annotations

import contextlib

from utils import is_truthy_value

from .method_ctx import bind_module


def _tui_compression_config_signature(cfg: dict | None) -> tuple:
    """Stable snapshot of compression/context keys that must apply next turn: the messaging-gateway
    cache-busting extract plus ``idle_compact_after_seconds``/``tail_mode`` (live-TUI-only keys)."""
    from gateway.run import GatewayRunner
    keys = GatewayRunner._extract_cache_busting_config(cfg)
    picked = {k: v for k, v in keys.items() if k.startswith("compression.") or k == "model.context_length"}
    compression = cfg.get("compression") if isinstance(cfg, dict) and isinstance(cfg.get("compression"), dict) else {}
    picked.update({f"compression.{k}": compression.get(k) for k in ("idle_compact_after_seconds", "tail_mode")})
    return tuple(sorted(picked.items()))


def _compressor_ctor_default(name: str, fallback: Any) -> Any:
    """Default read off ContextCompressor.__init__'s REAL signature, so unset-key restoration uses the
    construction path's derivation instead of a hardcoded copy that could drift.

    Unset restoration must go through the same derivation the construction path uses (#94724 review finding
    on #95980) — pulling the default off ``ContextCompressor.__init__`` itself instead of hardcoding copies
    keeps the two from drifting.
    """
    try:
        import inspect
        from agent.context_compressor import ContextCompressor
        default = inspect.signature(ContextCompressor.__init__).parameters[name].default
        return fallback if default is inspect.Parameter.empty else default
    except Exception:
        return fallback


def _default_threshold_tokens_cap():
    """The cap a fresh agent build installs when the key is absent: DEFAULT_CONFIG's
    ``compression.threshold_tokens``. agent_init reads the MERGED config, so "no key in
    config.yaml" installs that default at construction; key removal here must restore the
    same value, or a live session would diverge from a rebuilt one after the first turn
    (#117093). An explicit ``threshold_tokens: null`` stays ratio-only — the key is present,
    so ``.get`` returns it untouched."""
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    return (DEFAULT_CONFIG.get("compression") or {}).get("threshold_tokens")


def _derived_default_threshold_percent(agent: Any, compression: dict) -> float:
    """Default compaction threshold when ``compression.threshold`` is unset. Mirrors agent_init: ctor
    global default, then per-model resolution (Codex autoraise etc.) via the SAME
    ``_resolve_compression_threshold`` — removing the key restores the model-derived value."""
    try:
        pct = float(_compressor_ctor_default("threshold_percent", 0.50))
    except (TypeError, ValueError):
        pct = 0.50
    try:
        from agent.agent_init import _resolve_compression_threshold
        from agent.auxiliary_client import _compression_threshold_for_model, _is_codex_gpt54_or_gpt55, _is_codex_spark
        model, provider = getattr(agent, "model", "") or "", getattr(agent, "provider", "") or ""
        autoraise_enabled = str(compression.get("codex_gpt55_autoraise", True)).lower() in {"true", "1", "yes"}
        pct, _notice = _resolve_compression_threshold(
            pct, _compression_threshold_for_model(model, provider, allow_codex_gpt55_autoraise=autoraise_enabled),
            model=model, is_codex_autoraise=_is_codex_gpt54_or_gpt55(model, provider) or _is_codex_spark(model, provider),
        )
    except Exception:
        pass
    return pct


# (config key == compressor attr, ctor-default fallback, min_value)
_COMPRESSION_INT_KEYS = (
    ("proactive_prune_tokens", 0, 0),
    ("proactive_prune_min_result_chars", 8000, 0),
    ("proactive_prune_min_reclaim_tokens", 4096, 0),
    ("protect_last_n", 20, 0),
    ("min_tail_user_messages", 1, 1),
)


def _apply_live_compression_config(agent: Any, cfg: dict | None) -> None:
    """Update a live session's compressor in place from config.yaml. Every adopted key has UNSET semantics:
    a removed key restores the normalized default (or model-derived value) through the construction
    path's own derivation — acting only on PRESENT keys would leave stale values active forever.

    Every adopted key has UNSET semantics (#94724 review finding on the merged #95980): removing a key from
    config.yaml restores the normalized default — or the model-derived value — on the next turn, through the
    same derivation the construction path uses (ContextCompressor ctor defaults read off its real signature,
    the Codex threshold autoraise via ``_resolve_compression_threshold``, context-length re-inference via
    the deferred ``get_model_context_length`` resolution).
    """
    cfg = cfg if isinstance(cfg, dict) else {}
    compression = cfg.get("compression") if isinstance(cfg.get("compression"), dict) else {}
    from agent.agent_init import config_context_length_for_runtime, set_config_context_length
    enabled_raw = compression.get("enabled", True)
    agent.compression_enabled = enabled_raw if isinstance(enabled_raw, bool) else str(enabled_raw).lower() in {"true", "1", "yes"}
    agent.codex_responses_native_compaction = is_truthy_value(compression.get("codex_responses_native", False))
    native_threshold_raw = compression.get("codex_responses_compact_threshold", 200_000)
    try:
        if isinstance(native_threshold_raw, bool) or (native_threshold := int(native_threshold_raw)) <= 0:
            raise ValueError
    except (TypeError, ValueError):
        logger.warning("Invalid compression.codex_responses_compact_threshold=%r; using 200000.", native_threshold_raw)
        native_threshold = 200_000
    agent.codex_responses_compact_threshold = native_threshold
    # Absence restores the agent_init/config default (0 = disabled).
    with contextlib.suppress(TypeError, ValueError):
        agent.compression_idle_compact_after_seconds = max(0, int(compression.get("idle_compact_after_seconds", 0) or 0))
    cc = getattr(agent, "context_compressor", None)
    if cc is None:
        return
    # tail_mode: unknown/absent values land on the ctor default ("lean"), matching agent_init.
    default_tail = str(_compressor_ctor_default("tail_mode", "lean"))
    mode = str(compression.get("tail_mode", default_tail) or default_tail).strip().lower()
    cc.tail_mode = mode if mode in ("legacy", "lean") else default_tail
    for key, fallback, min_value in _COMPRESSION_INT_KEYS:
        default = int(_compressor_ctor_default(key, fallback))
        raw = compression.get(key, default)
        with contextlib.suppress(TypeError, ValueError):
            setattr(cc, key, max(min_value, default if raw is None else int(raw)))
    with contextlib.suppress(TypeError, ValueError):
        ratio_raw = compression.get("target_ratio", _compressor_ctor_default("summary_target_ratio", 0.20))
        cc.summary_target_ratio = max(0.10, min(float(ratio_raw), 0.80))
    # Absent or invalid shape (agent_init treats both as empty): stale overrides must stop steering.
    raw_thresholds = compression.get("model_thresholds")
    cc.model_thresholds = {
        str(k): float(v) for k, v in raw_thresholds.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
    } if isinstance(raw_thresholds, dict) else {}
    # threshold: present value wins; absence derives via the agent_init resolution (default + autoraise).
    # resolve_model_threshold returns ``pct`` unchanged when model_thresholds is empty.
    from agent.context_compressor import resolve_model_threshold
    pct: float | None = None
    if "threshold" in compression:
        with contextlib.suppress(TypeError, ValueError):
            pct = float(compression["threshold"])
    if pct is None:
        pct = _derived_default_threshold_percent(agent, compression)
    cc._config_threshold_percent = cc._configured_threshold_percent = pct
    base = cc._base_threshold_percent = resolve_model_threshold(
        getattr(agent, "model", "") or "", cc.model_thresholds, pct, getattr(agent, "provider", "") or "",
    )
    try:
        cc.threshold_percent = cc._effective_threshold_percent(cc.context_length, base)
    except Exception:
        cc.threshold_percent = pct
    # Same scoping rule as construction and the switch path: the pin describes the configured default
    # route, so a session that /model-switched elsewhere must not have it re-applied on a config save
    # (None = absent, invalid, or scoped out).
    new_ctx = config_context_length_for_runtime(agent, cfg)
    if new_ctx is not None:
        # Both cached copies: the compressor's (its own re-resolution) and the agent's
        # (switch/fallback + every display surface). Writing one left the other stale, so the
        # session showed a pinned ceiling while compressing against a different window (#116467).
        set_config_context_length(agent, new_ctx)
        with contextlib.suppress(Exception):
            cc.context_length = new_ctx
    elif getattr(cc, "_config_context_length", None) is not None:
        # model.context_length removed: drop the override and force re-inference from model metadata on
        # next access (construction's deferred resolution); re-applies the small-context floor too.
        set_config_context_length(agent, None)
        cc._resolved_context_length = None
    cc.threshold_tokens_cap = cc._coerce_threshold_tokens_cap(
        compression.get("threshold_tokens", _default_threshold_tokens_cap())
    )
    # Invalidate the cached trigger so the next preflight re-derives from percent/window, then the cap.
    cc._threshold_tokens = cc._tail_token_budget = None


def _sync_agent_compression_with_config(sid: str, session: dict) -> None:
    """Adopt compression.* / model.context_length edits at turn start (messaging gateways rebuild the
    agent on these keys; Desktop/TUI keeps the live compressor, so it must be updated in place).

    Desktop/TUI only synced the model; the live compressor kept the threshold captured at agent creation
    (#95151).
    """
    agent = session.get("agent")
    if agent is None:
        return
    cfg = _load_cfg() or {}
    signature = _tui_compression_config_signature(cfg)
    seen = session.get("config_compression_seen")
    session["config_compression_seen"] = signature
    if signature == seen:
        return
    try:
        _apply_live_compression_config(agent, cfg)
    except Exception as e:
        logger.warning("Could not apply live compression config for %s: %s", sid, e)


def _apply_pending_model_switch(sid: str, session: dict) -> None:
    """Apply a model switch queued (``session["pending_model_switch"]``) while a turn was running. Runs on
    the TURN thread at turn start — nothing in flight — so the in-place swap (client rebuild) is safe. A
    failed switch keeps the current model and never blocks the turn.

    A dropped pick is a warning notice, not an ``error``: clients paint ``error`` as a failed turn
    (Desktop's red retry card under the user's message) while this turn runs normally on the old model.
    The session.info re-sync moves the pill off the pick the session never adopted."""
    pending = session.pop("pending_model_switch", None)
    if not pending or session.get("agent") is None:
        return
    model = pending.get("display_model") or pending["raw"]
    try:
        result = _apply_model_switch(sid, session, pending["raw"], confirm_expensive_model=bool(pending.get("confirm_expensive_model")))
        # Honour the expensive-model confirm: drop the switch rather than spend on a model the user
        # never confirmed.
        if not result.get("confirm_required"):
            return
        detail = result.get("confirm_message") or result.get("warning") or ""
        logger.warning("Queued model switch to %s dropped for session %s: %s", model, sid, detail.split("\n", 1)[0])
        current = getattr(session["agent"], "model", "") or "the current model"
        # One line: the TUI status bar truncates and the Desktop toast collapses newlines. The full
        # guard text comes back in the confirm prompt when the user picks it again.
        text = f"Stayed on {current}: switching to {model} needs confirmation. Pick it again to confirm."
    except Exception as e:
        logger.warning("Queued model switch to %s failed for session %s: %s", model, sid, e)
        text = f"Could not switch model: {e}"
    _emit("notification.show", sid, {
        "text": text.rstrip(), "level": "warn", "kind": "ttl", "ttl_ms": 12000,
        "key": "model_switch.dropped", "id": "model_switch.dropped"})
    _emit_session_info(sid, session)


class CompressionLockHeld(Exception):
    """Raised by _compress_session_history when a concurrent compression_locks row skipped compression."""

    def __init__(self, holder: str | None = None):
        self.holder = holder
        super().__init__(f"Compression lock held: {holder or 'unknown'}")


class CompressionBusy(Exception):
    """Raised by _manual_compress_turn when a turn already holds the session; callers map it to their busy reply."""


@contextlib.contextmanager
def _manual_compress_turn(sid: str, session: dict, *, control_token=None):
    """Hold the session busy (``running``) for a whole manual compaction: snapshot, LLM summary, commit
    and session-key re-anchor. A prompt.submit arriving meanwhile takes the busy path (demoted to queue
    by ``_session_compression_in_flight``) instead of snapshotting a history version the compaction is
    about to bump, which dropped the turn's reply (#133504). The queued prompt drains on release."""
    with session["history_lock"]:
        active_control = session.get("_session_control_inflight")
        if active_control is not None and active_control is not control_token:
            raise CompressionBusy(_PLUGIN_SESSION_CONTROL_BUSY_MESSAGE)
        if session.get("running"):
            raise CompressionBusy(busy_message("compress", bool(session.get("_manual_compress_active"))))
        session["running"] = session["_manual_compress_active"] = True
    try:
        yield
    finally:
        with session["history_lock"]:
            session.pop("_manual_compress_active", None)
            session["running"] = False
        if control_token is not None:
            _release_session_control(session, control_token)
        # The compaction emitted session.info with running=true; close that edge or Desktop latches busy.
        _emit_session_info_for_session(sid, session)
        _drain_queued_prompt("__compress__", sid, session)


def _compress_session_history(
    session: dict, focus_topic: str | None = None, approx_tokens: int | None = None,
    before_messages: list | None = None, history_version: int | None = None,
) -> tuple[int, dict]:
    """Single choke point for all manual-compress routes. ``focus_topic`` is the RAW argument string after
    ``/compress``; the shared core (``agent.conversation_compression_manual``) parses boundary forms
    (``here [N]``, ``up to here``, ``--keep N``) so a partial compress triggers on EVERY route instead of a
    FULL compress focused on the literal text. ``--preview`` returns without touching history."""
    from agent.conversation_compression import finalize_context_engine_compression_notification
    from agent.conversation_compression_manual import (
        AGGRESSIVE_UNSUPPORTED, MIN_MESSAGES, compress_now, parse_compress_args)
    agent = session["agent"]
    # Snapshot under the lock so the LLM-bound compression call does NOT hold history_lock for the
    # request — otherwise prompt.submit etc. block on the dispatcher loop while compaction runs.
    if before_messages is None or history_version is None:
        with session["history_lock"]:
            before_messages, history_version = list(session.get("history", [])), int(session.get("history_version", 0))
    if len(before_messages) < MIN_MESSAGES:
        return 0, _get_usage(agent)
    request = parse_compress_args(focus_topic or "")
    if request.aggressive:
        raise ValueError(AGGRESSIVE_UNSUPPORTED)
    # RPC thread: bind the session cwd, or the boundary prompt rebuild resolves the backend's cwd and
    # persists a prompt every other process then rejects as stale runtime (fresh build, no tools pin).
    tokens = _set_session_context(session.get("session_key") or "", cwd=_session_cwd(session))
    def snapshot_is_current():
        with session["history_lock"]:
            return int(session.get("history_version", 0)) == history_version

    try:
        result = compress_now(agent, before_messages, request, task_id=session.get("session_key") or "default",
                              snapshot_is_current=snapshot_is_current)
    finally:
        _clear_session_context(tokens)
    if result.status == "preview":
        return 0, _get_usage(agent)
    # Lock-skipped: raise so callers surface a clear message instead of "No changes from compression".
    if result.status == "lock_skipped":
        raise CompressionLockHeld(result.lock_holder)
    if result.status != "compressed":
        return 0, _get_usage(agent)
    with session["history_lock"]:
        if int(session.get("history_version", 0)) != history_version:
            # External mutation during compaction — drop the result so we don't clobber concurrent edits.
            finalize_context_engine_compression_notification(agent, committed=False)
            return 0, _get_usage(agent)
        session["history"] = result.after_messages
        session["history_version"] = history_version + 1
    return result.removed, _get_usage(agent)


@contextlib.contextmanager
def _session_history_context(session: dict):
    """Use a session history lock, or a lockless legacy test seam."""
    history_lock = session.get("history_lock")
    if history_lock is None:
        yield
        return
    with history_lock:
        yield


def _sync_session_key_after_compress_legacy(
    sid: str,
    session: dict,
    *,
    clear_pending_title: bool = True,
    restart_slash_worker: bool = True,
    control_token=None,
) -> None:
    """Re-anchor session_key when AIAgent._compress_context rotates session_id."""
    agent = session.get("agent")
    with _session_history_context(session):
        active_control = session.get("_session_control_inflight")
        if active_control is not None and active_control is not control_token:
            logger.warning(
                "Compression rotation skipped while session control is active: sid=%s",
                sid,
            )
            return
    new_session_id = getattr(agent, "session_id", None) or ""
    old_key = session.get("session_key", "") or ""
    if not new_session_id or new_session_id == old_key:
        return

    lease_reanchored = _transfer_active_session_slot(
        sid,
        session,
        new_session_id=new_session_id,
    )
    if not lease_reanchored:
        logger.warning(
            "Compression session lease did not re-anchor: sid=%s old_session_id=%s new_session_id=%s",
            sid,
            old_key,
            new_session_id,
        )
    with _session_history_context(session):
        active_control = session.get("_session_control_inflight")
        if active_control is not None and active_control is not control_token:
            logger.warning("Compression rotation skipped while session control is active: sid=%s", sid)
            return
        session["session_key"] = new_session_id
        session["session_generation"] = int(session.get("session_generation", 0) or 0) + 1
    with contextlib.suppress(Exception):
        from tools import approval
        with contextlib.suppress(Exception):
            approval.unregister_gateway_notify(old_key)
        from tools.approval_yolo import transfer_session_yolo
        transfer_session_yolo(old_key, new_session_id)
        with contextlib.suppress(Exception):
            approval.register_gateway_notify(new_session_id, lambda data: _emit_approval_request(sid, data))
    session["_queued_prompt_generation"] = int(session.get("_queued_prompt_generation", 0)) + 1
    if clear_pending_title:
        session["pending_title"] = None
    if restart_slash_worker:
        try:
            _restart_slash_worker(sid, session)
        except Exception:
            logger.debug('sync session key after compress legacy failed', exc_info=True)


def _reanchor_registry_fields_match(
    identity: _CompressionReanchorIdentity,
    *,
    require_marker: bool = True,
) -> bool:
    """Check canonical fields while ``_sessions_lock`` is held."""
    session = identity.session
    if session.get("_session_control_inflight") is not identity.control_token:
        return False
    if session.get("agent") is not identity.agent:
        return False
    if (
        str(session.get("profile_home")) if session.get("profile_home") is not None else None
    ) != identity.profile_home:
        return False
    current_key = str(session.get("session_key") or "")
    current_generation = int(session.get("session_generation", 0) or 0)
    if current_key not in {identity.old_session_key, identity.new_session_id}:
        return False
    if current_generation != identity.generation and not (
        current_key == identity.new_session_id
        and current_generation > identity.generation
    ):
        return False
    if str(getattr(identity.agent, "session_id", "") or "") != identity.new_session_id:
        return False
    marker = session.get(_DEFERRED_COMPRESSION_ROTATION_KEY)
    if require_marker:
        if marker is not identity.marker or not isinstance(marker, dict):
            return False
        try:
            marker_generation = int(marker.get("generation", -1))
        except (TypeError, ValueError):
            return False
        if (
            marker_generation != identity.generation
            or str(marker.get("agent_session_id") or "") != identity.new_session_id
        ):
            return False
    return True


def _reanchor_identity_matches(identity: _CompressionReanchorIdentity) -> bool:
    """Validate history identity, then registry identity, with no I/O."""
    session = identity.session
    history_lock = session.get("history_lock")
    if history_lock is None:
        return False
    with history_lock:
        if not _reanchor_registry_fields_match(identity):
            return False
    with _sessions_lock:
        return _sessions.get(identity.sid) is session and _reanchor_registry_fields_match(
            identity
        )


def _reanchor_apply_in_memory(
    identity: _CompressionReanchorIdentity,
    *,
    clear_pending_title: bool,
) -> bool:
    """Atomically apply only pure canonical session fields under registry lock."""
    with _sessions_lock:
        if _sessions.get(identity.sid) is not identity.session:
            return False
        if not _reanchor_registry_fields_match(identity):
            return False
        current_key = str(identity.session.get("session_key") or "")
        if current_key == identity.old_session_key:
            identity.session["session_key"] = identity.new_session_id
            identity.session["session_generation"] = int(
                identity.session.get("session_generation", 0) or 0
            ) + 1
            identity.session["_queued_prompt_generation"] = int(
                identity.session.get("_queued_prompt_generation", 0)
            ) + 1
        elif current_key != identity.new_session_id:
            return False
        if clear_pending_title:
            identity.session["pending_title"] = None
        return True


def _reanchor_clear_marker(identity: _CompressionReanchorIdentity) -> bool:
    """Clear a deferred marker only after the final exact identity check."""
    with _sessions_lock:
        if _sessions.get(identity.sid) is not identity.session:
            return False
        if not _reanchor_registry_fields_match(identity):
            return False
        identity.session.pop(_DEFERRED_COMPRESSION_ROTATION_KEY, None)
        return True


def _sync_unlocked_compression(sid, session, *, clear_pending_title, restart_slash_worker,
                               control_token, expected_reanchor):
    """Retain the legacy path only for records outside the live registry."""
    pending_marker = session.get(_DEFERRED_COMPRESSION_ROTATION_KEY)
    active_control_hint = session.get("_session_control_inflight")
    if (
        expected_reanchor is not None
        or pending_marker is not None
        or active_control_hint is not None
        or control_token is not None
    ):
        logger.warning(
            "Compression re-anchor requires history_lock; refusing deferred/controlled sync: sid=%s",
            sid,
        )
        return _COMPRESSION_SYNC_FAILED
    with _sessions_lock:
        if any(candidate is session for candidate in _sessions.values()):
            logger.warning(
                "Compression re-anchor requires history_lock for a live registry session: sid=%s",
                sid,
            )
            return _COMPRESSION_SYNC_FAILED
    old_key = str(session.get("session_key", "") or "")
    _sync_session_key_after_compress_legacy(
        sid,
        session,
        clear_pending_title=clear_pending_title,
        restart_slash_worker=restart_slash_worker,
        control_token=None,
    )
    return (
        _COMPRESSION_SYNC_APPLIED
        if str(session.get("session_key", "") or "") != old_key
        else _COMPRESSION_SYNC_UNCHANGED
    )


def _record_compression_rotation_marker(
    session, *,
    active_token,
    new_session_id: str,
    old_session_key: str,
    generation: int,
    clear_title: bool,
    restart_worker: bool,
) -> None:
    existing = session.get(_DEFERRED_COMPRESSION_ROTATION_KEY)
    existing_generation = (
        int(existing.get("generation", -1))
        if isinstance(existing, dict)
        else -1
    )
    same_marker = bool(
        isinstance(existing, dict)
        and existing.get("control_token") is active_token
        and existing_generation == generation
    )
    if not same_marker:
        session[_DEFERRED_COMPRESSION_ROTATION_KEY] = {
            "agent_session_id": str(new_session_id),
            "old_session_key": str(old_session_key),
            "generation": int(generation),
            "control_token": active_token,
            "clear_pending_title": bool(clear_title),
            "restart_slash_worker": bool(restart_worker),
        }
        return
    existing["agent_session_id"] = str(new_session_id)
    existing["old_session_key"] = str(old_session_key)
    existing["generation"] = int(generation)
    existing["control_token"] = active_token
    existing["clear_pending_title"] = bool(clear_title)
    existing["restart_slash_worker"] = bool(restart_worker)


def _capture_compression_reanchor(sid, session, agent, control_token, expected_reanchor):
    """Read and validate the exact compression continuation while history is locked."""
    if expected_reanchor is not None:
        if (
            expected_reanchor.session is not session
            or expected_reanchor.sid != sid
            or (
                control_token is not None
                and control_token is not expected_reanchor.control_token
            )
        ):
            return None, _COMPRESSION_SYNC_IDENTITY_CHANGED
        control_token = expected_reanchor.control_token
    implicit_token = session.get(_MANUAL_COMPRESSION_CONTROL_KEY)
    if control_token is None and implicit_token is not None:
        control_token = implicit_token
    active_control = session.get("_session_control_inflight")
    live_session_id = str(getattr(agent, "session_id", "") or "")
    old_key = str(session.get("session_key", "") or "")
    generation = int(session.get("session_generation", 0) or 0)
    pending = session.get(_DEFERRED_COMPRESSION_ROTATION_KEY)
    pending_generation = (
        int(pending.get("generation", -1))
        if isinstance(pending, dict)
        else -1
    )
    pending_matches = bool(
        isinstance(pending, dict)
        and pending.get("control_token") is not None
        and pending_generation >= 0
        and pending_generation <= generation
        and str(pending.get("agent_session_id") or "") == live_session_id
    )
    if expected_reanchor is not None:
        if pending is not expected_reanchor.marker or not pending_matches:
            return None, _COMPRESSION_SYNC_IDENTITY_CHANGED
        if active_control is not expected_reanchor.control_token:
            return None, _COMPRESSION_SYNC_IDENTITY_CHANGED
        if live_session_id != expected_reanchor.new_session_id:
            return None, _COMPRESSION_SYNC_IDENTITY_CHANGED
        new_session_id = expected_reanchor.new_session_id
    else:
        new_session_id = live_session_id
    return {
        "control_token": control_token,
        "active_control": active_control,
        "old_key": old_key,
        "generation": generation,
        "pending": pending,
        "pending_generation": pending_generation,
        "pending_matches": pending_matches,
        "new_session_id": new_session_id,
        "expected_reanchor": expected_reanchor,
    }, None


def _reserve_compression_reanchor(sid, session, agent, captured, *, clear_pending_title,
                                  restart_slash_worker):
    """Defer competing control or reserve one token for the prepared continuation."""
    control_token = captured["control_token"]
    active_control = captured["active_control"]
    old_key = captured["old_key"]
    generation = captured["generation"]
    pending = captured["pending"]
    pending_generation = captured["pending_generation"]
    pending_matches = captured["pending_matches"]
    new_session_id = captured["new_session_id"]
    expected_reanchor = captured["expected_reanchor"]
    rotation_needed = bool(new_session_id and new_session_id != old_key)
    retry_pending = bool(pending_matches and new_session_id)

    if expected_reanchor is None and active_control is not None and active_control is not control_token:
        if rotation_needed:
            _record_compression_rotation_marker(
                session,
                active_token=active_control,
                new_session_id=new_session_id,
                old_session_key=old_key,
                generation=generation,
                clear_title=clear_pending_title,
                restart_worker=restart_slash_worker,
            )
            logger.warning(
                "Compression rotation deferred while session control is active: sid=%s",
                sid,
            )
            return None, _COMPRESSION_SYNC_DEFERRED
        if retry_pending:
            logger.warning(
                "Deferred compression rotation still blocked by session control: sid=%s",
                sid,
            )
            return None, _COMPRESSION_SYNC_DEFERRED
        return None, _COMPRESSION_SYNC_UNCHANGED

    if not rotation_needed and not retry_pending:
        return None, (
            _COMPRESSION_SYNC_IDENTITY_CHANGED
            if expected_reanchor is not None
            else _COMPRESSION_SYNC_UNCHANGED
        )

    owns_internal_token = False
    operation_token = control_token
    if expected_reanchor is not None:
        operation_token = expected_reanchor.control_token
        if active_control is not operation_token:
            return None, _COMPRESSION_SYNC_IDENTITY_CHANGED
    elif operation_token is None:
        operation_token = object()
        session["_session_control_inflight"] = operation_token
        owns_internal_token = True
    elif active_control is not operation_token:
        return None, _COMPRESSION_SYNC_FAILED

    old_key_for_updates = old_key
    clear_title_for_updates = clear_pending_title
    restart_worker_for_updates = restart_slash_worker
    if retry_pending:
        old_key_for_updates = str(pending.get("old_session_key") or old_key)
        clear_title_for_updates = bool(
            pending.get("clear_pending_title", clear_pending_title)
        )
        restart_worker_for_updates = bool(
            pending.get("restart_slash_worker", restart_slash_worker)
        )

    if expected_reanchor is None and retry_pending:
        expected_reanchor = _CompressionReanchorIdentity(
            sid=sid,
            session=session,
            generation=pending_generation,
            agent=agent,
            new_session_id=new_session_id,
            control_token=operation_token,
            marker=pending,
            old_session_key=old_key_for_updates,
            profile_home=(
                str(session.get("profile_home"))
                if session.get("profile_home") is not None
                else None
            ),
        )
    return {
        "new_session_id": new_session_id,
        "old_key_for_updates": old_key_for_updates,
        "clear_title_for_updates": clear_title_for_updates,
        "restart_worker_for_updates": restart_worker_for_updates,
        "expected_reanchor": expected_reanchor,
        "operation_token": operation_token,
        "owns_internal_token": owns_internal_token,
    }, None


def _apply_compression_reanchor(sid, session, state):
    """Transfer the lease and approval ownership with identity checks between I/O."""
    new_session_id = state["new_session_id"]
    old_key_for_updates = state["old_key_for_updates"]
    clear_title_for_updates = state["clear_title_for_updates"]
    restart_worker_for_updates = state["restart_worker_for_updates"]
    expected_reanchor = state["expected_reanchor"]
    operation_token = state["operation_token"]
    def guard():
        return expected_reanchor is None or _reanchor_identity_matches(expected_reanchor)

    if not guard():
        return _COMPRESSION_SYNC_IDENTITY_CHANGED
    lease_reanchored = _transfer_active_session_slot(
        sid,
        session,
        new_session_id=new_session_id,
    )
    if not lease_reanchored:
        raise RuntimeError(
            "active session lease could not be transferred to the continuation"
        )

    from tools.approval import (
        disable_session_yolo,
        enable_session_yolo,
        is_session_yolo_enabled,
        register_gateway_notify,
        unregister_gateway_notify,
    )

    if not guard():
        return _COMPRESSION_SYNC_IDENTITY_CHANGED
    unregister_gateway_notify(old_key_for_updates)
    if not guard():
        return _COMPRESSION_SYNC_IDENTITY_CHANGED
    yolo_was_on = is_session_yolo_enabled(old_key_for_updates)
    if yolo_was_on:
        if not guard():
            return _COMPRESSION_SYNC_IDENTITY_CHANGED
        enable_session_yolo(new_session_id)
        if not guard():
            return _COMPRESSION_SYNC_IDENTITY_CHANGED
        disable_session_yolo(old_key_for_updates)
    if not guard():
        return _COMPRESSION_SYNC_IDENTITY_CHANGED
    register_gateway_notify(
        new_session_id,
        lambda data: _emit_approval_request(sid, data),
    )

    if expected_reanchor is not None:
        if not _reanchor_apply_in_memory(
            expected_reanchor,
            clear_pending_title=clear_title_for_updates,
        ):
            return _COMPRESSION_SYNC_IDENTITY_CHANGED
    else:
        with _session_history_context(session):
            if session.get("_session_control_inflight") is not operation_token:
                return _COMPRESSION_SYNC_FAILED
            current_key = str(session.get("session_key") or "")
            if current_key != new_session_id:
                session["session_key"] = new_session_id
                session["session_generation"] = int(
                    session.get("session_generation", 0) or 0
                ) + 1
                session["_queued_prompt_generation"] = int(
                    session.get("_queued_prompt_generation", 0)
                ) + 1
            if clear_title_for_updates:
                session["pending_title"] = None

    if restart_worker_for_updates:
        if not guard():
            return _COMPRESSION_SYNC_IDENTITY_CHANGED
        _restart_slash_worker(sid, session)

    if expected_reanchor is not None:
        if not _reanchor_clear_marker(expected_reanchor):
            return _COMPRESSION_SYNC_IDENTITY_CHANGED
    else:
        with _session_history_context(session):
            if session.get("_session_control_inflight") is not operation_token:
                return _COMPRESSION_SYNC_FAILED
            marker = session.get(_DEFERRED_COMPRESSION_ROTATION_KEY)
            if isinstance(marker, dict) and str(
                marker.get("agent_session_id") or ""
            ) == new_session_id:
                session.pop(_DEFERRED_COMPRESSION_ROTATION_KEY, None)
    return _COMPRESSION_SYNC_APPLIED


def _retain_failed_compression_reanchor(sid, session, state, exc):
    """Retain the exact deferred marker after interrupted re-anchor I/O."""
    new_session_id = state["new_session_id"]
    old_key_for_updates = state["old_key_for_updates"]
    clear_title_for_updates = state["clear_title_for_updates"]
    restart_worker_for_updates = state["restart_worker_for_updates"]
    expected_reanchor = state["expected_reanchor"]
    operation_token = state["operation_token"]
    if expected_reanchor is not None:
        try:
            if not _reanchor_identity_matches(expected_reanchor):
                return _COMPRESSION_SYNC_IDENTITY_CHANGED
        except BaseException:
            logger.debug('retain failed compression reanchor failed', exc_info=True)
            return _COMPRESSION_SYNC_IDENTITY_CHANGED
    with _session_history_context(session):
        marker = session.get(_DEFERRED_COMPRESSION_ROTATION_KEY)
        current_generation = int(session.get("session_generation", 0) or 0)
        if isinstance(marker, dict):
            marker["agent_session_id"] = str(new_session_id)
            marker["old_session_key"] = str(
                marker.get("old_session_key") or old_key_for_updates
            )
            marker["generation"] = current_generation
            marker["clear_pending_title"] = bool(clear_title_for_updates)
            marker["restart_slash_worker"] = bool(restart_worker_for_updates)
        else:
            _record_compression_rotation_marker(
                session,
                active_token=operation_token,
                new_session_id=new_session_id,
                old_session_key=old_key_for_updates,
                generation=current_generation,
                clear_title=clear_title_for_updates,
                restart_worker=restart_worker_for_updates,
            )
    logger.warning(
        "Compression rotation re-anchor failed; deferred marker retained: sid=%s error=%s",
        sid,
        exc,
    )
    return _COMPRESSION_SYNC_FAILED


def _sync_session_key_after_compress(
    sid: str, session: dict, *, clear_pending_title: bool = True,
    restart_slash_worker: bool = True, control_token=None,
    expected_reanchor: _CompressionReanchorIdentity | None = None,
) -> str:
    """Re-anchor a compression continuation with an exact identity barrier."""
    history_lock = session.get("history_lock")
    if history_lock is None:
        return _sync_unlocked_compression(
            sid, session, clear_pending_title=clear_pending_title,
            restart_slash_worker=restart_slash_worker, control_token=control_token,
            expected_reanchor=expected_reanchor)
    agent = session.get("agent")
    if agent is None:
        return (_COMPRESSION_SYNC_IDENTITY_CHANGED if expected_reanchor is not None
                else _COMPRESSION_SYNC_UNCHANGED)
    with history_lock:
        captured, error = _capture_compression_reanchor(
            sid, session, agent, control_token, expected_reanchor)
        if error is not None:
            return error
        state, error = _reserve_compression_reanchor(
            sid, session, agent, captured, clear_pending_title=clear_pending_title,
            restart_slash_worker=restart_slash_worker)
        if error is not None:
            return error
    try:
        return _apply_compression_reanchor(sid, session, state)
    except BaseException as exc:
        logger.debug("Compression re-anchor failed", exc_info=True)
        return _retain_failed_compression_reanchor(sid, session, state, exc)
    finally:
        if state["owns_internal_token"]:
            with history_lock:
                if session.get("_session_control_inflight") is state["operation_token"]:
                    session.pop("_session_control_inflight", None)




def _compress_via_compute_host(rid, params: dict, session: dict) -> dict:
    """``session.compress`` for a turn-isolated session: forward ``/compress`` to the host."""
    sid = str(params.get("session_id") or "")
    focus_topic = _str_param(params, "focus_topic")

    def _on_late_ack(late: dict, _sid=sid) -> None:
        _adopt_late_compute_host_compress_ack(_sid, session, late, route_name="session.compress")
    try:
        ack = _send_compute_host_control(
            sid, route_name="session.compress", command="/compress" + (f" {focus_topic}" if focus_topic else ""),
            # compression.context_total_ceiling_seconds: the host legitimately runs that long.
            wait=True, timeout=_compute_host_compress_wait_seconds(), on_late_ack=_on_late_ack)
    except queue.Empty:
        # Waiter gave up, host still compressing; the late-ack handler adopts the rotated session when it
        # lands. Not an error (a 5019 here reported timeouts that later succeeded).
        return _ok(rid, {"status": "pending", "turn_isolation": True,
                         "message": ("compression still running in the background; "
                                     "the transcript will refresh when it finishes")})
    except Exception as exc:
        return _err(rid, 5019, f"compute-host compress failed: {exc}")
    if (resp := _compute_host_ack_error(rid, ack, 4009, "compute-host compress failed")) is not None:
        return resp
    _apply_compute_host_metadata_mirror(session, ack)
    if isinstance(host_result := ack.get("result"), dict):
        # Host-owned result verbatim (carries `status: aborted` / `summary.aborted`).
        return _ok(rid, {**host_result, "turn_isolation": True})
    host_info = ack.get("session_info") if isinstance(ack.get("session_info"), dict) else {}
    return _ok(rid, {
        "status": "compressed", "turn_isolation": True,
        # `messages` goes top-level for the transcript replacement; don't duplicate it in the ack.
        "host_ack": {key: value for key, value in ack.items() if key != "messages"}, "info": host_info,
        "messages": _history_to_messages(ack.get("messages"), profile_home=session.get("profile_home")) if isinstance(ack.get("messages"), list) else [],
        "usage": host_info.get("usage") if isinstance(host_info.get("usage"), dict) else {}})

def _compress_live(
    rid,
    sid: str,
    session: dict,
    focus_topic: str,
    control_token=None,
) -> dict:
    """In-process ``session.compress``: status pinned "compressing", then the before/after summary + messages."""
    from agent.conversation_compression import finalize_context_engine_compression_notification
    from agent.manual_compression_feedback import summarize_manual_compression
    from agent.model_metadata import estimate_request_tokens_rough
    with session["history_lock"]:
        before_messages = list(session.get("history", []))
        history_version = int(session.get("history_version", 0))
    before_count = len(before_messages)
    _agent = session["agent"]
    _sys_prompt = getattr(_agent, "_cached_system_prompt", "") or ""
    _tools = getattr(_agent, "tools", None) or None

    def _tokens(msgs) -> int:
        # Re-reads prompt + tools each call: _compress_context may have rebuilt the system prompt.
        sys_prompt = getattr(_agent, "_cached_system_prompt", "") or _sys_prompt
        tools = getattr(_agent, "tools", None) or _tools
        return estimate_request_tokens_rough(msgs, system_prompt=sys_prompt, tools=tools) if msgs else 0
    before_tokens = _tokens(before_messages)
    if before_count >= 4:
        focus_suffix = f', focus: "{focus_topic}"' if focus_topic else ""
        _status_update(sid, "compressing",
                       f"⠋ compressing {before_count} messages (~{before_tokens:,} tok){focus_suffix}…")
    try:
        removed, usage = _compress_session_history(
            session, focus_topic, approx_tokens=before_tokens, before_messages=before_messages,
            history_version=history_version)
        with session["history_lock"]:
            messages = list(session.get("history", []))
        after_tokens = _tokens(messages)
        agent = session["agent"]
        _sync_session_key_after_compress(
            sid,
            session,
            control_token=control_token,
        )
        summary = summarize_manual_compression(before_messages, messages, before_tokens, after_tokens,
                                               compression_state=getattr(agent, "context_compressor", None))
        info = _session_info(agent, session)
        _emit("session.info", sid, info)
        finalize_context_engine_compression_notification(agent, committed=True)
        return _ok(rid, {
            "status": "aborted" if summary["aborted"] else "compressed", "removed": removed,
            "before_messages": before_count, "after_messages": len(messages),
            "before_tokens": before_tokens, "after_tokens": after_tokens, "summary": summary,
            "usage": usage, "info": info, "messages": _history_to_messages(messages, profile_home=session.get("profile_home"))})
    finally:
        # Always clear the pinned compressing status (success, no-op, or raise).
        _status_update(sid, "ready")

def _run_session_compression(
    rid,
    params: dict,
    session: dict,
    control_token,
) -> dict:
    """Run session compression after its exact session-control lease is held."""
    if _session_uses_compute_host(session):
        return _compress_via_compute_host(rid, params, session)
    if session.get("running"):
        return _err(
            rid,
            4009,
            "session busy — /interrupt the current turn before /compress",
        )
    from agent.conversation_compression import (
        finalize_context_engine_compression_notification,
    )

    sid = params.get("session_id", "")
    try:
        with _manual_compress_turn(sid, session, control_token=control_token):
            return _compress_live(
                rid, sid, session, _str_param(params, "focus_topic"), control_token=control_token,
            )
    except CompressionBusy as exc:
        return _err(rid, 4009, str(exc))
    except CompressionLockHeld as exc:
        _status_update(sid, "ready")
        from agent.manual_compression_feedback import describe_compression_lock_skip

        return _ok(
            rid,
            {
                "compressed": False,
                "lock_held": True,
                "message": describe_compression_lock_skip(exc.holder),
            },
        )
    except Exception as exc:
        logger.debug('run session compression failed', exc_info=True)
        finalize_context_engine_compression_notification(
            session["agent"],
            committed=False,
        )
        return _err(rid, 5005, str(exc))


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
