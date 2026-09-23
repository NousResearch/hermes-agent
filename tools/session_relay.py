"""Local-only desktop-session relay spike (not exposed as an unauthenticated RPC).

The caller is trusted in-process sender code. The sender's identity is derived from the
home this process is bound to, never from the caller's argument, so no caller can speak
as another profile or as the human. No stored session is resumed or written by the
sender process.
"""
from __future__ import annotations

import re

from hermes_cli.profiles import get_profile_dir, normalize_profile_name, profile_exists, read_profile_meta
from tools.bot_live_delivery import (
    deliver_to_live_owner,
    find_continuing_desktop_owner,
    find_exact_desktop_owner,
    read_delivery_result,
    refuse_unsettled_delivery,
)
from tools.bot_relay import delivery_turn_author


def bound_profile_identity() -> tuple[str, str]:
    """``(canonical name, visible label)`` of the home this process is bound to.

    The root home *is* the ``default`` profile; a named profile home is itself. The label
    comes from that home's ``profile.yaml`` ``display_name`` when set, so a caller cannot
    choose how it is displayed.
    """
    from hermes_constants import get_hermes_home

    home = get_hermes_home().resolve()
    name = "default"
    if home.parent.name == "profiles" and normalize_profile_name(home.name) != "default":
        if profile_exists(home.name) and get_profile_dir(home.name).resolve() == home:
            name = normalize_profile_name(home.name)
    label = str(read_profile_meta(home).get("display_name") or "").strip() or name
    return name, label


def _author_slug(label: str, fallback: str) -> str:
    """Author-id fragment for a label: the id must not carry spaces or path separators."""
    slug = re.sub(r"[^a-z0-9_-]+", "-", str(label).strip().lower()).strip("-")
    return slug or fallback


def deliver_to_desktop_session(profile: str, session_id: str, sender: str, message: str,
                               *, delivery_id: str | None = None) -> dict:
    """Admit one peer-authored turn to an existing named profile's exact desktop owner.

    A queued receipt means durable admission to the same owner observed after
    the write. Claimed is explicitly indeterminate until a terminal callback or
    ``read_desktop_delivery_result`` proves the pinned lease ended and records
    ``ambiguous``. No state is replayed.
    """
    try:
        target = normalize_profile_name(profile)
        sender_id = normalize_profile_name(sender)
    except (TypeError, ValueError):
        raise ValueError("named target profile and sender are required") from None
    if target == "default" or not profile_exists(target):
        raise ValueError("named target profile does not exist")
    if not all(isinstance(value, str) and value.strip() for value in (session_id, message)):
        raise ValueError("session_id, sender and message are required")
    bound_name, label = bound_profile_identity()
    if sender_id != bound_name:
        raise ValueError("sender must be the profile this process is bound to")
    home = get_profile_dir(target).resolve()
    owner = find_exact_desktop_owner(home, session_id)
    if owner is None:
        raise ValueError("exact desktop session has no live owner")
    author = delivery_turn_author(_author_slug(label, bound_name), label)
    text = f"Message from 🤖 {label}:\n{message}"
    admitted = deliver_to_live_owner(home, owner, text, delivery_id=delivery_id, author=author)
    current = find_continuing_desktop_owner(home, owner)
    if current is None and admitted.get("status") in {"queued", "claimed"}:
        return refuse_unsettled_delivery(
            home, admitted["delivery_id"], reason="exact desktop owner changed during admission")
    return admitted


def read_desktop_delivery_result(profile: str, delivery_id: str) -> dict | None:
    """Read a receipt and reconcile a dead claimed owner without replay.

    ``claimed`` remains indeterminate while the exact pinned lease is live. Once
    that lease is absent, it becomes terminal ``ambiguous`` because execution
    may have started before the owner disappeared.
    """
    target = normalize_profile_name(profile)
    if target == "default" or not profile_exists(target):
        raise ValueError("named target profile does not exist")
    home = get_profile_dir(target).resolve()
    record = read_delivery_result(home, delivery_id)
    if record is None or record.get("status") not in {"queued", "claimed"}:
        return record
    pinned = record["owner"]
    current = find_continuing_desktop_owner(home, pinned)
    if current is None:
        return refuse_unsettled_delivery(
            home, delivery_id, reason="exact desktop owner lease ended before a terminal receipt")
    return record
