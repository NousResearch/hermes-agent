"""Shared-metrics consent JSON-RPC handlers (the Desktop twin of ``hermes setup``'s Shared Metrics
section). Both opt-ins live in the focused profile's config.yaml exactly where the wizard writes
them — ``telemetry.shared_metrics.enabled`` (local collection) and ``.send`` (daily upload) — so the
CLI and the app can never disagree about the answer. ``shared_metrics.set`` enforces the wizard's
invariant (sending needs collection) and reconciles the consent windows in the local store on every
change, through the same single writer the wizard uses.
Bodies are rebound onto server.py's globals (method_ctx.bind_module) and reference them bare.
"""

import logging

from .method_ctx import HandlerRegistry, bind_module

logger = logging.getLogger(__name__)
_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped


def _shared_metrics_section(cfg) -> dict:
    telemetry = cfg.get("telemetry") if isinstance(cfg, dict) else None
    section = telemetry.get("shared_metrics") if isinstance(telemetry, dict) else None
    return section if isinstance(section, dict) else {}


def _shared_metrics_consent(cfg) -> dict:
    """``decided`` = the user answered somewhere (either key written explicitly); the shipped
    defaults are not an answer, so a profile that never saw the question reads undecided."""
    section = _shared_metrics_section(cfg)
    enabled = section.get("enabled") is True
    return {
        "enabled": enabled,
        "send": enabled and section.get("send") is True,
        "decided": "enabled" in section or "send" in section,
    }


def _shared_metrics_record_setup_completed(cfg) -> None:
    """Desktop has no setup-finish RPC; this first-run answer is the one backend call made once,
    right after onboarding settles. A no-op (inside the events API) unless collection is on."""
    from hermes_cli.observability.shared_metrics_events import record_setup_completed

    model = cfg.get("model") if isinstance(cfg, dict) else None
    provider = model.get("provider") if isinstance(model, dict) else None
    record_setup_completed(surface="desktop", provider=provider if isinstance(provider, str) and provider else None)


@method("shared_metrics.status")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """``{enabled, send, decided}`` for the focused profile. A pure read of config.yaml (no defaults
    merged, so ``decided`` sees only what the user wrote)."""
    try:
        return _ok(rid, _shared_metrics_consent(_load_cfg()))
    except Exception as e:
        return _err(rid, 5095, str(e))


@method("shared_metrics.set")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Write both opt-ins at once and reconcile the consent windows. ``send`` is forced off when
    ``enabled`` is off (the wizard's rule: sending cannot outlive collection, and turning collection
    off withdraws send consent). ``first_run`` marks the Desktop first-run answer, which also records
    the setup-completed metric. Answers the stored ``{enabled, send, decided}``."""
    enabled = params.get("enabled") is True
    send = enabled and params.get("send") is True
    try:
        cfg = _load_cfg_raw()
        telemetry = cfg.get("telemetry")
        if not isinstance(telemetry, dict):
            telemetry = cfg["telemetry"] = {}
        section = telemetry.get("shared_metrics")
        if not isinstance(section, dict):
            section = telemetry["shared_metrics"] = {}
        section["enabled"], section["send"] = enabled, send
        _save_cfg(cfg)
    except Exception as e:
        return _err(rid, 5096, str(e))
    from hermes_cli.setup import _record_send_consent_change
    # Unconditional, like the wizard: a send key already false may still have an open window.
    _record_send_consent_change(enabled=send)
    if params.get("first_run") is True:
        _shared_metrics_record_setup_completed(cfg)
    return _ok(rid, _shared_metrics_consent(cfg))


@method("shared_metrics.slash_command")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """The Desktop and Ink TUI dispatchers call this once per user-typed slash command, locally
    handled ones included; the gateway never counts slash.exec / command.dispatch itself, so
    each command lands exactly once. Always ``{ok: true}`` (the events API never raises)."""
    from hermes_cli.observability.shared_metrics_events import record_slash_command

    record_slash_command(command=str(params.get("command") or ""), surface=_resolve_session_platform())
    return _ok(rid, {"ok": True})


@method("shared_metrics.startup_latency")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """The Ink TUI (gateway ready) and Desktop (backend attached) each report their own launch ->
    ready time once per launch. A Desktop on a URL/cloud backend has no ``HERMES_DESKTOP`` here, so
    the client's declared surface wins over env detection. Always ``{ok: true}``."""
    from hermes_cli.observability.shared_metrics_startup import record_rpc_startup_latency

    record_rpc_startup_latency(
        client_surface=params.get("surface") or _resolve_session_platform(), elapsed_ms=params.get("elapsed_ms"),
    )
    return _ok(rid, {"ok": True})


def register(server) -> None:
    bind_module(globals(), server, skip=("_",))
