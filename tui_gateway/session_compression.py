"""Live compression: config hot-reload onto a running agent, pending model switch apply, /compress
(CompressionLockHeld when a turn holds the lock), session-key sync after compress. Bodies are rebound
onto server.py's globals (method_ctx.bind_module) and reference them bare."""

from __future__ import annotations

import contextlib

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
    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
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
    base = cc._base_threshold_percent = resolve_model_threshold(getattr(agent, "model", "") or "", cc.model_thresholds, pct)
    try:
        cc.threshold_percent = cc._effective_threshold_percent(cc.context_length, base)
    except Exception:
        cc.threshold_percent = pct
    raw_ctx = model_cfg.get("context_length")
    if raw_ctx is not None:
        with contextlib.suppress(TypeError, ValueError):
            if (new_ctx := int(raw_ctx)) > 0:
                cc._config_context_length = new_ctx
                with contextlib.suppress(Exception):
                    cc.context_length = new_ctx
    elif getattr(cc, "_config_context_length", None) is not None:
        # model.context_length removed: drop the override and force re-inference from model metadata on
        # next access (construction's deferred resolution); re-applies the small-context floor too.
        cc._config_context_length = cc._resolved_context_length = None
    cc.threshold_tokens_cap = cc._coerce_threshold_tokens_cap(compression.get("threshold_tokens"))
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
    failed switch keeps the current model and never blocks the turn."""
    pending = session.pop("pending_model_switch", None)
    if not pending or session.get("agent") is None:
        return
    try:
        result = _apply_model_switch(sid, session, pending["raw"], confirm_expensive_model=bool(pending.get("confirm_expensive_model")))
        # Honour the expensive-model confirm: surface the warning and drop the switch rather than spend
        # on a model the user never confirmed.
        if result.get("confirm_required"):
            _emit("error", sid, {"message": result.get("confirm_message") or result.get("warning") or ""})
    except Exception as e:
        _emit("error", sid, {"message": f"Could not switch model: {e}"})


class CompressionLockHeld(Exception):
    """Raised by _compress_session_history when a concurrent compression_locks row skipped compression."""

    def __init__(self, holder: str | None = None):
        self.holder = holder
        super().__init__(f"Compression lock held: {holder or 'unknown'}")


def _compress_session_history(
    session: dict, focus_topic: str | None = None, approx_tokens: int | None = None,
    before_messages: list | None = None, history_version: int | None = None,
) -> tuple[int, dict]:
    """Single choke point for all manual-compress routes. ``focus_topic`` is the RAW argument string after
    ``/compress``, parsed HERE (not per-route) so boundary forms (``here [N]``, ``up to here``, ``--keep N``)
    trigger a partial compress on EVERY route instead of a FULL compress focused on the literal text.

    It is parsed here with :func:`parse_partial_compress_args` so boundary-aware forms (``here [N]``, ``up
    to here``, ``--keep N``) trigger a partial compress — head summarized, most recent ``keep_last``
    exchanges kept verbatim — on EVERY route, mirroring cli.py's ``_manual_compress`` and
    gateway/slash_commands.py (PR #35252).
    """
    from agent.conversation_compression import finalize_context_engine_compression_notification
    from agent.model_metadata import estimate_request_tokens_rough
    from hermes_cli.partial_compress import (
        parse_partial_compress_args, rejoin_compressed_head_and_tail, split_history_for_partial_compress,
    )
    agent = session["agent"]
    # Snapshot under the lock so the LLM-bound compression call does NOT hold history_lock for the
    # request — otherwise prompt.submit etc. block on the dispatcher loop while compaction runs.
    if before_messages is None or history_version is None:
        with session["history_lock"]:
            before_messages, history_version = list(session.get("history", [])), int(session.get("history_version", 0))
    history = before_messages
    if len(history) < 4:
        return 0, _get_usage(agent)
    partial, keep_last, focus_topic = parse_partial_compress_args(focus_topic or "")
    # Only the head is summarized; the last `keep_last` exchanges ride along verbatim. A degenerate
    # split (empty tail) falls back to full compression so the user still gets an action.
    head, tail = split_history_for_partial_compress(history, keep_last) if partial else (history, [])
    if not tail:
        head = history
    if approx_tokens is None:
        # Include system prompt + tool schemas so the figure reflects real request pressure.
        # Include system prompt + tool schemas in the estimate — a transcript-only number understates real
        # request pressure and can even appear to grow after compression because a dense handoff summary
        # replaces many short turns (#6217).
        approx_tokens = estimate_request_tokens_rough(
            history, system_prompt=getattr(agent, "_cached_system_prompt", "") or "", tools=getattr(agent, "tools", None) or None
        )
    # system_message=None: passing the cached prompt (already holding the identity block) would append the
    # identity twice. force=True: manual /compress bypasses the summary-failure cooldown like CLI/gateway.
    # Pass system_message=None so AIAgent._compress_context rebuilds the system prompt cleanly via
    # _build_system_prompt(None). Mirrors the CLI's _manual_compress fix for issue #15281. force=True: every
    # caller of this helper is a manual /compress path (session.compress RPC, slash compress/compact,
    # slash-worker mirror) — auto-compaction runs inside the agent loop, not here.
    try:
        compressed, _ = agent._compress_context(
            head, None, approx_tokens=approx_tokens, focus_topic=focus_topic or None, force=True,
            defer_context_engine_notification=True,
        )
    except Exception:
        finalize_context_engine_compression_notification(agent, committed=False)
        raise
    # Lock-skipped: raise so callers surface a clear message instead of "No changes from compression".
    # Type-pinned (is True / str) because bare truthiness is fooled by MagicMock auto-attrs.
    _lock_skipped = getattr(agent, "_compression_skipped_due_to_lock", None)
    if _lock_skipped is True or isinstance(_lock_skipped, str):
        agent._compression_skipped_due_to_lock = None
        # No boundary committed; discard the pending deferred notification (exactly-once, no-op safe).
        finalize_context_engine_compression_notification(agent, committed=False)
        raise CompressionLockHeld(_lock_skipped if isinstance(_lock_skipped, str) else None)
    if tail:
        compressed = rejoin_compressed_head_and_tail(compressed, tail)
    with session["history_lock"]:
        if int(session.get("history_version", 0)) != history_version:
            # External mutation during compaction — drop the result so we don't clobber concurrent edits.
            finalize_context_engine_compression_notification(agent, committed=False)
            return 0, _get_usage(agent)
        session["history"] = compressed
        session["history_version"] = history_version + 1
    return len(history) - len(compressed), _get_usage(agent)


def _sync_session_key_after_compress(
    sid: str, session: dict, *, clear_pending_title: bool = True, restart_slash_worker: bool = True
) -> None:
    """Re-anchor the gateway-side ``session_key`` when _compress_context rotates ``agent.session_id``;
    otherwise approval routing, slash worker, DB lookups and yolo state keep targeting the ended parent.
    ``clear_pending_title``: True for manual /compress (title belongs to the old session), False for
    post-turn auto-compression. ``restart_slash_worker``: False only when the caller manages the worker."""
    agent = session.get("agent")
    new_session_id = getattr(agent, "session_id", None) or ""
    old_key = session.get("session_key", "") or ""
    if not new_session_id or new_session_id == old_key:
        return
    if not _transfer_active_session_slot(sid, session, new_session_id=new_session_id):
        logger.warning(
            "Compression session lease did not re-anchor: sid=%s old_session_id=%s new_session_id=%s",
            sid, old_key, new_session_id,
        )
    # Even if the approval module fails to import, anchor session_key on the continuation id.
    session["session_key"] = new_session_id
    _publish_session_identity(sid, session)
    with contextlib.suppress(Exception):
        from tools import approval
        with contextlib.suppress(Exception):
            approval.unregister_gateway_notify(old_key)
        with contextlib.suppress(Exception):
            if approval.is_session_yolo_enabled(old_key):
                approval.enable_session_yolo(new_session_id)
                approval.disable_session_yolo(old_key)
        with contextlib.suppress(Exception):
            approval.register_gateway_notify(new_session_id, lambda data: _emit_approval_request(sid, data))
    # Invalidate any in-flight ``_drain_queued_prompt`` claim taken under the pre-rotation key: a raced
    # drain must not dispatch on the continuation (its envelope is restored to the queue).
    session["_queued_prompt_generation"] = int(session.get("_queued_prompt_generation", 0)) + 1
    if clear_pending_title:
        session["pending_title"] = None
    if restart_slash_worker:
        with contextlib.suppress(Exception):
            _restart_slash_worker(sid, session)


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
