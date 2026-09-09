"""The ``plugins.hermes_bots.hide_bot_chats`` user preference.

Shared contract between the hide seams Bot Mode plumbing writes through
(all backend-side):

* ``session.create`` / ``session.set_hidden`` (tui_gateway/methods_session.py)
* REST ``PATCH /api/sessions/{id}`` (hermes_cli/web_routers/sessions.py)

When the pref is false, a generic hide request is suppressed: normal user
chats must never be hidden. Explicit ``hidden: false`` (un-hide) always
passes through.

Fails toward VISIBLE when the config cannot be read (review note on
#102625): a visible chat is recoverable; a silently hidden one is lost.
"""

import contextlib

_CACHE: dict[str, bool] = {}


def bot_hide_pref() -> bool:
    """Read ``plugins.hermes_bots.hide_bot_chats`` (default True = hide).

    Cached per process after the first read; config edits need a backend
    restart to take effect (same contract as every other config-gated
    behavior). An unreadable config fails toward False (visible) — never
    hide a normal chat because the config was unreadable.
    """
    if "v" in _CACHE:
        return _CACHE["v"]
    value = True  # upstream default: Bot Mode sessions hide (key absent)
    try:
        # Late import: tests monkeypatch hermes_cli.config.load_config_readonly
        # and must intercept this read (patch where production reads).
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly()
        plugins = cfg.get("plugins") if isinstance(cfg, dict) else None
        bots = plugins.get("hermes_bots") if isinstance(plugins, dict) else None
        if isinstance(bots, dict) and "hide_bot_chats" in bots:
            value = bool(bots["hide_bot_chats"])
    except Exception:
        value = False  # fail toward visible: never hide on an unreadable config
    _CACHE["v"] = value
    return value


def effective_hidden_flag(raw_value: bool) -> bool:
    """Resolve a caller's ``hidden`` intent against the user pref.

    ``False`` (show) is always honored. ``True`` (hide) only sticks when the
    user has not opted out of Bot Mode hiding.
    """
    if not raw_value:
        return False
    return bot_hide_pref()
