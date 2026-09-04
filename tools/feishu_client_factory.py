"""Shared lark-client resolution for the Feishu tools.

Both :mod:`tools.feishu_doc_tool` and :mod:`tools.feishu_drive_tool` need the
same client resolution semantics, so it lives here once instead of being
duplicated (and drifting) across the two modules.

Resolution order:

1. A client injected for the current thread by the Feishu comment-event
   handler (``set_client``). This always wins, so behavior inside the comment
   path is unchanged by the fallback.
2. A client built from the ``FEISHU_APP_ID`` / ``FEISHU_APP_SECRET``
   environment credentials — the same values the Feishu platform adapter
   reads. This lets the doc/drive tools work outside a comment-event context
   (DM conversations, CLI sessions, agent-initiated turns), where previously
   they hard-failed with "Feishu client not available".

The env fallback is deliberately **not** available to ``delegate_task`` child
contexts. Building from app credentials grants process-wide access to the
tenant's documents, and child contexts are the least-privileged execution
tier — mirroring the delegated-child mutation guards used elsewhere in the
tree (``_assert_not_delegated_child_mutation``). A delegated child therefore
keeps the historical "client not available" path rather than gaining
tenant-wide doc/drive authority.
"""

import logging
import os
import threading

logger = logging.getLogger(__name__)

__all__ = ["set_client", "get_client", "clear_client"]

_local = threading.local()


def set_client(client):
    """Store a lark client for the current thread (called by feishu_comment).

    Passing ``None`` clears any cached client — including one built from env
    credentials — so a credential swap mid-process is picked up on the next
    ``get_client()`` call.
    """
    # Injected clients live in their own slot so they are never evicted by the
    # credential-keyed cache below (and so clearing the cache cannot drop an
    # explicitly injected client).
    _local.injected = client
    _local.client = None
    _local.cred_key = None


def clear_client():
    """Drop the thread-local client (injected or env-built)."""
    set_client(None)


def _is_delegated_child() -> bool:
    """Return True while running inside a ``delegate_task`` child context."""
    try:
        from agent.delegation_context import is_delegated_child_context
    except ImportError:
        return False
    try:
        return bool(is_delegated_child_context())
    except Exception:  # pragma: no cover - defensive; never break tool flow
        return False


def get_client():
    """Return the injected client, else one built from env credentials."""
    injected = getattr(_local, "injected", None)
    if injected is not None:
        return injected

    app_id = os.getenv("FEISHU_APP_ID")
    app_secret = os.getenv("FEISHU_APP_SECRET")
    if not app_id or not app_secret:
        return None

    if _is_delegated_child():
        logger.debug(
            "[Feishu] env-credential client fallback skipped: delegate_task "
            "child contexts do not inherit tenant-wide doc/drive access"
        )
        return None

    try:
        from lark_oapi import Client, LogLevel
    except ImportError:
        return None

    # Key the cache on the credential pair so a mid-process credential swap
    # rebuilds instead of reusing a stale client.
    cred_key = (app_id, app_secret)
    cached = getattr(_local, "client", None)
    if cached is not None and getattr(_local, "cred_key", None) == cred_key:
        return cached

    client = (
        Client.builder()
        .app_id(app_id)
        .app_secret(app_secret)
        .log_level(LogLevel.WARNING)
        .build()
    )
    # Distinct marker: when this fallback engages, downstream authorization
    # errors come from app credentials rather than the historical "not
    # available" path, so make the difference visible in logs.
    logger.info(
        "[Feishu] built lark client from FEISHU_APP_ID/FEISHU_APP_SECRET env "
        "credentials (outside comment context); permission failures below are "
        "app-credential authorization errors, not 'client not available'"
    )
    _local.client = client
    _local.cred_key = cred_key
    return client
