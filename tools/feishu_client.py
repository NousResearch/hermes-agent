"""Shared helpers for Feishu/Lark tool clients."""

import logging
import os

logger = logging.getLogger(__name__)


def build_client_from_env(log: logging.Logger | None = None):
    """Build a lark_oapi client from configured app credentials, if present.

    Feishu document tools are primarily used from document-comment events where
    the gateway injects a request-scoped client. The same tools can also be
    exposed in ordinary chat sessions, where no comment-context client exists.
    In that case, fall back to the app credentials already configured for the
    gateway instead of reporting that the client is unavailable.
    """
    app_id = os.getenv("FEISHU_APP_ID") or os.getenv("LARK_APP_ID")
    app_secret = os.getenv("FEISHU_APP_SECRET") or os.getenv("LARK_APP_SECRET")
    if not app_id or not app_secret:
        return None

    try:
        import lark_oapi as lark
    except ImportError:
        return None

    try:
        return (
            lark.Client.builder()
            .app_id(app_id)
            .app_secret(app_secret)
            .log_level(lark.LogLevel.ERROR)
            .build()
        )
    except Exception as exc:  # pragma: no cover - defensive SDK guard
        active_logger = log or logger
        active_logger.warning("Failed to build Feishu client from environment: %s", exc)
        return None
