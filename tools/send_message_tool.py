# ---- BEGIN LOCAL FIX (Nilo, 2026-09-08): _send_to_platform se perdió en el merge del
# refactor upstream (existe en origin/main pero no en esta rama feature). scheduler_delivery
# la importa para la lane standalone de crons. Port mínimo: Telegram vía _send_telegram del
# módulo senders; resto vía live adapter del gateway (con fallback al registry standalone).
# Parche documentado en skill hermes-local-patches; se pierde con hermes update.

async def _send_to_platform(platform, pconfig, chat_id, message, thread_id=None, media_files=None, force_document=False, args=None):
    """Local re-port of the upstream dispatcher (simplified: telegram + registry/adapter)."""
    from gateway.config import Platform
    from tools.send_message_senders import _send_telegram, _registry_standalone_send, _live_adapter
    platform_name = platform.value if hasattr(platform, "value") else str(platform)
    media_files = media_files or []
    if platform == Platform.TELEGRAM:
        return await _send_telegram(
            pconfig.token, chat_id, message, media_files=media_files, thread_id=thread_id,
            force_document=force_document,
            disable_link_previews=bool(getattr(pconfig, "extra", {}) and pconfig.extra.get("disable_link_previews")))
    adapter = None
    try:
        adapter = _live_adapter(platform)[1]
    except Exception:
        adapter = None
    if adapter is not None and hasattr(adapter, "send_message"):
        return await adapter.send_message(chat_id, message, media_files=media_files, thread_id=thread_id)
    return await _registry_standalone_send(platform_name, pconfig, chat_id, message, thread_id=thread_id)

# ---- BEGIN LOCAL FIX (Nilo, 2026-09-11): cron/scheduler_delivery.py importa
# prepare_send_message_platforms y resolve_send_target de ESTE módulo, pero el refactor upstream
# movió resolve_send_target a tools/send_message_targets.py y prepare_send_message_platforms se
# quedó fuera de esta rama feature. Sin esto, TODA entrega de cron con target explícito
# (platform:chat_id) falla con "cannot import name 'prepare_send_message_platforms'" y el output
# se pierde. Parche reaplicable: ~/workspace/hermes-cron-standalone-send-fix.patch (skill
# hermes-local-patches); se pierde con `hermes update`.
from tools.send_message_targets import resolve_send_target  # noqa: E402  (re-export local)


def prepare_send_message_platforms() -> None:
    """Load enabled standalone plugins before tool schemas/cache keys are built."""
    from hermes_cli.plugins import discover_plugins
    discover_plugins()
# ---- END LOCAL FIX

_PLUGIN_COMPAT_LAZY = {
    'redact_sensitive_text': ('agent.redact', 'redact_sensitive_text'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
