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
    credential_pairs = (
        (os.getenv("FEISHU_APP_ID"), os.getenv("FEISHU_APP_SECRET")),
        (os.getenv("LARK_APP_ID"), os.getenv("LARK_APP_SECRET")),
    )
    app_id, app_secret = next(
        (
            (app_id, app_secret)
            for app_id, app_secret in credential_pairs
            if app_id and app_secret
        ),
        (None, None),
    )
    if not app_id or not app_secret:
        return None

    try:
        import lark_oapi as lark
        from lark_oapi.core.const import FEISHU_DOMAIN, LARK_DOMAIN
    except ImportError:
        return None

    try:
        domain_name = os.getenv("FEISHU_DOMAIN", "").strip().lower()
        domain = LARK_DOMAIN if domain_name == "lark" else FEISHU_DOMAIN
        return (
            lark.Client.builder()
            .app_id(app_id)
            .app_secret(app_secret)
            .domain(domain)
            .log_level(lark.LogLevel.ERROR)
            .build()
        )
    except Exception as exc:  # pragma: no cover - defensive SDK guard
        active_logger = log or logger
        active_logger.warning("Failed to build Feishu client from environment: %s", exc)
        return None
