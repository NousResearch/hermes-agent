"""Local-engine lifecycle for ``tools.tts_tool``: warm-up / release leases.

Local engines load lazily on first synthesis (dead air on the first spoken reply) and then stay
resident. Every surface that flips speech output on holds a *lease* here (warming the configured
engine); when the last lease is released the local model caches are dropped after a keep-warm
window (``tts.keep_warm_seconds``), so one surface's "off" can't unload a model another surface
still needs and a wake-word loop that re-acquires within the window reuses the loaded model.
Lease names are process-wide and any surface may drop one, but what a lease warmed stays owned by
the profile that acquired it: every profile home that acquired since the last unload keeps its
acquisition context, and the keep-warm window, the user provider's release hook and the secrets
that hook forwards are read in that context, never the releaser's or the timer thread's.
Cloud providers have nothing resident; warming only ensures the SDK imports. Origin seams (``_load_tts_config``, ``_get_provider``) are
resolved through :func:`_origin` per call.
"""

from __future__ import annotations

import logging
import threading
import time
from contextvars import Context, copy_context
from typing import Any, Callable, Dict, List, Optional

from agent.memory_provider import ctx_bound
from hermes_constants import hermes_home_key
from tools import tts_command_provider
from tools.tts_command_provider import (
    BUILTIN_TTS_PROVIDERS, _get_command_tts_timeout, _get_named_provider_config,
    _is_command_provider_config, command_env_passthrough as _command_provider_env_passthrough,
    render_command_template as _render_command_tts_template)
from tools.tts_tool_delivery import _origin
from tools.tts_tool_local import (
    _LOCAL_TTS_MODEL_CACHES, _load_kittentts_model_for_config, _load_piper_voice_for_config)
from tools.tts_tool_plugins import _lookup_plugin_provider

logger = logging.getLogger("tools.tts_tool")

_tts_lease_lock = threading.Lock()
_tts_leases: set = set()
# Profile home key -> its latest acquisition context; claimed (cleared) by the unload that releases it.
_tts_owners: dict[str, Context] = {}
# Monotonic generation per profile home, kept past claim so an outer unload skips a home superseded
# even by a nested acquire+zero-window release that already cleared its new ``_tts_owners`` entry.
_tts_owner_generations: dict[str, int] = {}
# Held across an unload's release hooks instead of the lease lock, so an acquire that registers after
# the unload claimed its owners warms only once those hooks are done. Taken before the lease lock;
# re-entrant so a plugin's release() may itself acquire or release a lease.
_tts_teardown_lock = threading.RLock()

# Pending keep-warm unload; the generation lets a timer that already woke up see it was superseded.
_DEFAULT_KEEP_WARM_SECONDS = 60.0
_make_keep_warm_timer = threading.Timer  # test seam
_keep_warm_timer: Optional[threading.Timer] = None
_keep_warm_generation = 0


def _local_tts_warmers() -> dict[str, Callable[[dict[str, Any]], Any]]:
    """Provider name → loader populating that engine's cache slot (same key synthesis uses)."""
    return {
        "piper": lambda cfg: _load_piper_voice_for_config(cfg)[0],
        "kittentts": lambda cfg: _load_kittentts_model_for_config(cfg)[0]}


def _signal_user_tts_provider(name: str, tts_config: dict[str, Any], hook: str) -> Optional[str]:
    """Forward a lease ``hook`` (``"warm"``/``"release"``) to a user-declared provider; returns the action.
    Command providers run their optional ``<hook>_command`` (same template/env/timeout rules as
    ``command``) on a background thread so a toggle never waits on a model server; plugins get
    :meth:`TTSProvider.warm`/``release``. Best-effort: failures are logged at debug."""
    if not name or name in BUILTIN_TTS_PROVIDERS:
        return None
    cfg = _get_named_provider_config(tts_config, name)
    try:
        if _is_command_provider_config(cfg):
            template = str(cfg.get(f"{hook}_command") or "").strip()
            if not template:
                return None
            command = _render_command_tts_template(template, {
                "voice": str(cfg.get("voice", "")),
                "model": str(cfg.get("model", "")),
                "speed": str(cfg.get("speed", tts_config.get("speed", "")))})

            def _run() -> None:
                try:
                    tts_command_provider.run_command_provider(
                        command, _get_command_tts_timeout(cfg),
                        env_passthrough=_command_provider_env_passthrough(cfg))
                except Exception as exc:
                    logger.debug("[TTS] %s_command for %s failed: %s", hook, name, exc)
            # ctx_bound: env_passthrough resolves through the caller's profile secret scope.
            threading.Thread(target=ctx_bound(_run), name=f"tts-{hook}-{name}", daemon=True).start()
            return hook
        plugin_provider = _lookup_plugin_provider(name)
        if plugin_provider is None:
            return None
        getattr(plugin_provider, hook)()
        return hook
    except Exception as exc:
        logger.debug("[TTS] %s hook for %s failed: %s", hook, name, exc)
        return "error"


def warm_tts_provider(tts_config: Optional[dict[str, Any]] = None, provider: Optional[str] = None) -> dict[str, Any]:
    """Pre-load the configured TTS provider so the next synthesis starts hot (blocking; never raises).
    Local engines fill the same LRU slot synthesis reads (including first-use download); lazily
    installed cloud SDKs are made importable; user-declared providers get their warm hook;
    everything else is ``action: "noop"``. The result carries ``warmed`` / ``action`` / ``error``."""
    if tts_config is None:
        tts_config = _origin()._load_tts_config()
    name = (provider or _origin()._get_provider(tts_config) or "").lower().strip()
    result: dict[str, Any] = {"provider": name, "warmed": False, "action": "noop"}
    warmer = _local_tts_warmers().get(name)
    if warmer is not None:
        cache = _LOCAL_TTS_MODEL_CACHES.get(name, {})
        before, started = len(cache), time.monotonic()
        try:
            warmer(tts_config)
        except Exception as exc:  # engine missing, download failed, bad voice…
            logger.warning("[TTS] warm-up for %s failed: %s", name, exc)
            result.update(action="error", error=str(exc))
            return result
        result.update(
            warmed=True, action="loaded" if len(cache) > before else "cached",
            elapsed_ms=int((time.monotonic() - started) * 1000))
        logger.info("[TTS] warm-up %s: %s in %dms", name, result["action"], result["elapsed_ms"])
        return result
    signalled = _signal_user_tts_provider(name, tts_config, "warm")
    if signalled is not None:
        ok = signalled != "error"
        result.update(warmed=ok, action="warmed" if ok else "error")
        return result
    on_demand = _origin()._SDK_ON_DEMAND.get(name)
    if on_demand is not None:
        feature = on_demand[0]
        try:
            import pm
            if pm.available(feature):
                result.update(warmed=True, action="cached")
            else:
                pm.ensure_import(feature)
                result.update(warmed=True, action="installed")
        except Exception as exc:
            logger.debug("[TTS] SDK warm-up for %s skipped: %s", name, exc)
            result.update(action="error", error=str(exc))
    return result


def _signal_release() -> None:
    """Send the release hook to the user provider configured for the CURRENT profile."""
    tts_config = _origin()._load_tts_config()
    _signal_user_tts_provider(_origin()._get_provider(tts_config), tts_config, "release")


def _drop_local_models(name: str = "") -> int:
    """Clear resident local models (only ``name``'s cache when given); returns how many were dropped.
    Reads no config, so it needs no profile context."""
    released = 0
    for cache_name, cache in _LOCAL_TTS_MODEL_CACHES.items():
        if not name or cache_name == name:
            released += len(cache)
            cache.clear()
    if released:
        logger.info("[TTS] released %d resident local model(s)", released)
    return released


def release_tts_provider(provider: Optional[str] = None) -> dict[str, Any]:
    """Drop resident local models -> ``{"released": <count>}``. With ``provider`` only that engine's
    cache is cleared; otherwise every cache is, and the configured user provider is signalled."""
    name = (provider or "").lower().strip()
    if not name:
        _signal_release()
    return {"released": _drop_local_models(name)}


def _run_as_owner(owner: Context, fn: Callable[[], Any], fallback: Any, what: str) -> Any:
    """``fn()`` in a copy of ``owner`` (a Context already entered by another thread cannot be entered
    again). One owner's failure (an unreadable or malformed ``tts`` config) yields ``fallback``, so it
    never stops the other profiles' windows or releases. Logs the exception type only: the message
    can quote that profile's config. A config-backend outage is a ``SystemExit`` by design, so it is
    named here too."""
    from hermes_cli.config_backend import ConfigBackendUnavailable
    try:
        return owner.copy().run(fn)
    except (Exception, ConfigBackendUnavailable) as exc:  # health: allow BLE001 -- one profile never blocks the others
        logger.warning("[TTS] lease owner %s failed (%s); other owners continue", what, type(exc).__name__)
        return fallback


def _owners_keep_warm_seconds_locked() -> float:
    """The longest window any owner's profile asks for, each read in its own context; an owner whose
    window cannot be read counts as the default window."""
    return max((_run_as_owner(owner, _keep_warm_seconds, _DEFAULT_KEEP_WARM_SECONDS, "keep-warm read")
                for owner in _tts_owners.values()), default=_DEFAULT_KEEP_WARM_SECONDS)


def _keep_warm_seconds() -> float:
    """``tts.keep_warm_seconds``, read per call so each profile's config applies; ``0`` = unload now."""
    raw = _origin()._load_tts_config().get("keep_warm_seconds", _DEFAULT_KEEP_WARM_SECONDS)
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return _DEFAULT_KEEP_WARM_SECONDS


def _cancel_keep_warm_locked() -> None:
    global _keep_warm_timer, _keep_warm_generation
    _keep_warm_generation += 1
    if _keep_warm_timer is not None:
        _keep_warm_timer.cancel()
        _keep_warm_timer = None


def _schedule_release_locked() -> Optional[int]:
    """No holders left: (re)start the owners' keep-warm window. When it is 0, returns the generation
    the caller unloads inline once it has dropped the lease lock (``None`` when a timer started)."""
    global _keep_warm_timer
    _cancel_keep_warm_locked()
    delay = _owners_keep_warm_seconds_locked()
    if delay <= 0:
        return _keep_warm_generation
    # Not context-bound: the unload runs each owner's release in that owner's context.
    timer = _make_keep_warm_timer(delay, _release_after_keep_warm, args=(_keep_warm_generation,))
    timer.daemon = True
    _keep_warm_timer = timer
    timer.start()
    return None


def _release_after_keep_warm(generation: int) -> int:
    """Unload unless a holder came back: claim the owners under the lease lock, then signal each one's
    release (its config, provider hook and secrets) in a copy of its acquisition context, outside that
    lock, and drop the local models once. Returns the count of local models released."""
    global _keep_warm_timer
    with _tts_teardown_lock:
        with _tts_lease_lock:
            if generation != _keep_warm_generation or _tts_leases:
                return 0
            _keep_warm_timer = None
            owners = [(home, owner, _tts_owner_generations.get(home)) for home, owner in _tts_owners.items()]
            _tts_owners.clear()
        for home, owner, owner_gen in owners:
            with _tts_lease_lock:
                if _tts_owner_generations.get(home) != owner_gen:
                    continue
            _run_as_owner(owner, _signal_release, None, "release")
        # Context-free, once: also covers the models an owner whose release failed left resident.
        # Recheck under the lease lock so a reentrant or concurrent acquire (or a newly scheduled
        # keep-warm window) never has its freshly warmed local cache wiped by this stale unload.
        with _tts_lease_lock:
            if generation != _keep_warm_generation or _tts_leases:
                return 0
            return _drop_local_models()


def acquire_tts_lease(lease: str, tts_config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Register ``lease`` (e.g. ``"desktop:read-aloud"``) and warm the provider. Re-acquiring is
    idempotent but still re-warms (cheap on a cache hit; heals a cache cleared elsewhere). An
    acquire inside the keep-warm window cancels the pending unload. The calling profile owns what
    this warms until the unload after the last release."""
    home, owner = hermes_home_key(), copy_context()
    with _tts_lease_lock:
        _cancel_keep_warm_locked()
        _tts_leases.add(lease)
        _tts_owners[home] = owner
        _tts_owner_generations[home] = _keep_warm_generation
        holders = len(_tts_leases)
    with _tts_teardown_lock:  # an unload that already claimed its owners finishes before this warm
        pass
    result = warm_tts_provider(tts_config)
    with _tts_lease_lock:
        # An unload during the load claimed this owner, but the load left the resource up again.
        if home not in _tts_owners:
            _tts_owners[home] = owner
            _tts_owner_generations[home] = _keep_warm_generation
        # Every holder released while the load ran: the window starts now that the model is resident.
        inline = _schedule_release_locked() if not _tts_leases else None
    if inline is not None:
        _release_after_keep_warm(inline)
    return {**result, "leases": holders}


def release_tts_lease(lease: str) -> dict[str, Any]:
    """Drop ``lease``; the last one out unloads resident local models once the keep-warm window
    passes with no new acquire; the window and the release follow the acquiring profiles, not the
    caller. A never-acquired lease is a no-op (still reports the holder count) so surfaces can call
    this unconditionally. ``released`` counts models unloaded inline."""
    with _tts_lease_lock:
        held = lease in _tts_leases
        _tts_leases.discard(lease)
        holders = len(_tts_leases)
        inline = _schedule_release_locked() if held and holders == 0 else None
    released = _release_after_keep_warm(inline) if inline is not None else 0
    return {"leases": holders, "released": released}


def tts_lease_holders() -> list[str]:
    """Snapshot of live lease names (diagnostics / tests)."""
    with _tts_lease_lock:
        return sorted(_tts_leases)


def _reset_tts_leases_for_tests() -> None:
    with _tts_lease_lock:
        _cancel_keep_warm_locked()
        _tts_leases.clear()
        _tts_owners.clear()
        _tts_owner_generations.clear()
