"""History-reuse budgets for native vision embeds (config section ``vision``).

A native ``vision_analyze`` result bakes the image into conversation history, where it is
re-sent on every later API call. ``vision.embed_target_bytes`` bounds that cost
(how large one embed may be); see #112095 for why 256 KB is a budget, not a constant.
"""
from __future__ import annotations

# 256 KB keeps a 1568px screenshot cheap enough to ride the session (#92699); the clamp keeps one
# setting from turning every later request into a multi-megabyte resend or a useless thumbnail.
_DEFAULT_EMBED_TARGET_BYTES = 256 * 1024
_MIN_EMBED_TARGET_BYTES = 64 * 1024
_MAX_EMBED_TARGET_BYTES = 4 * 1024 * 1024


def _cfg_vision(key: str, default=None):
    """``vision.<key>`` from config.yaml; ``default`` when config is unavailable."""
    try:
        from hermes_cli.config import cfg_get, load_config
        return cfg_get(load_config(), "vision", key, default=default)
    except Exception:
        return default


def resolve_embed_target_bytes() -> int:
    """``vision.embed_target_bytes`` clamped to 64 KiB..4 MiB; the 256 KB default on a bad value."""
    raw = _cfg_vision("embed_target_bytes", default=_DEFAULT_EMBED_TARGET_BYTES)
    try:
        if isinstance(raw, bool):
            raise ValueError("boolean is not a byte budget")
        target = int(raw)
    except (TypeError, ValueError, OverflowError):
        return _DEFAULT_EMBED_TARGET_BYTES
    return min(max(target, _MIN_EMBED_TARGET_BYTES), _MAX_EMBED_TARGET_BYTES)
