"""Command parser and dispatcher for /wisdom."""

from __future__ import annotations

from dataclasses import dataclass

from wisdom.config import load_wisdom_config
from wisdom.db import WisdomDB
from wisdom.models import WisdomConfig
from wisdom.render import (
    render_applications,
    render_blocked_secret,
    render_capture,
    render_captures,
    render_error,
    render_help,
    render_interpretation,
    render_not_found,
    render_original,
    render_related_payload,
    render_review_action,
    render_review_payload,
    render_status,
)
from wisdom.service import (
    WisdomServiceContext,
    accept,
    apply,
    archive,
    can_natural_capture as service_can_natural_capture,
    capture,
    dismiss,
    inbox,
    interpret,
    original as get_original_capture,
    related,
    related_payload,
    review,
    review_payload,
    search,
    set_enabled,
    status_snapshot,
)


@dataclass(frozen=True)
class WisdomCommandContext:
    channel: str = "gateway"
    source_kind: str = "command"
    session_key: object | None = None
    message_ref: object | None = None


def handle_wisdom_command(
    raw_args: str,
    *,
    context: WisdomCommandContext | None = None,
    config: WisdomConfig | None = None,
    db: WisdomDB | None = None,
) -> str:
    config = config or load_wisdom_config()
    db = db or WisdomDB(config.db_path)
    db.init()
    context = context or WisdomCommandContext()

    raw_args = (raw_args or "").strip()
    subcommand, arg = _split(raw_args)
    subcommand = subcommand.lower() if subcommand else "help"
    service_context = WisdomServiceContext(
        channel=context.channel,
        source_kind=context.source_kind,
        session_key=context.session_key,
        message_ref=context.message_ref,
    )

    if subcommand in {"help", "-h", "--help"}:
        return render_help()
    if subcommand == "status":
        return render_status(status_snapshot(config=config, db=db))
    if subcommand == "on":
        set_enabled(True, config=config, db=db)
        return "Wisdom is on. Capture mode: explicit."
    if subcommand == "off":
        set_enabled(False, config=config, db=db)
        return "Wisdom is off. Status/help/on still work."

    if not status_snapshot(config=config, db=db).enabled:
        return "Wisdom is off. Use /wisdom on to enable it."

    if subcommand == "capture":
        if not arg:
            return "Usage: /wisdom capture <text>"
        outcome = capture(
            arg,
            context=service_context,
            config=config,
            db=db,
            require_enabled=True,
        )
        if outcome.status == "captured" and outcome.capture:
            return render_capture(outcome.capture)
        if outcome.status == "blocked_secret":
            return render_blocked_secret()
        return outcome.message or render_error()

    if subcommand == "inbox":
        return render_captures("Wisdom inbox", inbox(config=config, db=db))

    if subcommand == "search":
        if not arg:
            return "Usage: /wisdom search <query>"
        return render_captures("Wisdom search", search(arg, config=config, db=db))

    if subcommand == "original":
        capture_id = _parse_id(arg)
        if capture_id is None:
            return "Usage: /wisdom original <id>"
        capture_record = get_original_capture(capture_id, config=config, db=db)
        return render_original(capture_record) if capture_record else render_not_found(capture_id)

    if subcommand == "interpret":
        capture_id = _parse_id(arg)
        if capture_id is None:
            return "Usage: /wisdom interpret <id>"
        if get_original_capture(capture_id, config=config, db=db) is None:
            return render_not_found(capture_id)
        return render_interpretation(interpret(capture_id, config=config, db=db, create=True))

    if subcommand == "apply":
        capture_id = _parse_id(arg)
        if capture_id is None:
            return "Usage: /wisdom apply <id>"
        if get_original_capture(capture_id, config=config, db=db) is None:
            return render_not_found(capture_id)
        return render_applications(capture_id, apply(capture_id, config=config, db=db))

    if subcommand == "archive":
        capture_id = _parse_id(arg)
        if capture_id is None:
            return "Usage: /wisdom archive <id>"
        return f"Archived #{capture_id}." if archive(capture_id, config=config, db=db) else render_not_found(capture_id)

    if subcommand == "accept":
        capture_id = _parse_id(arg)
        if capture_id is None:
            return "Usage: /wisdom accept <id>"
        return render_review_action("accepted", accept(capture_id, config=config, db=db), capture_id)

    if subcommand == "dismiss":
        capture_id = _parse_id(arg)
        if capture_id is None:
            return "Usage: /wisdom dismiss <id>"
        return render_review_action("dismissed", dismiss(capture_id, config=config, db=db), capture_id)

    if subcommand == "related":
        capture_id = _parse_id(arg)
        if capture_id is None:
            return "Usage: /wisdom related <id>"
        if get_original_capture(capture_id, config=config, db=db) is None:
            return render_not_found(capture_id)
        return render_related_payload(
            related_payload(related(capture_id, config=config, db=db), capture_id=capture_id)
        )

    if subcommand == "review":
        category, mode = _parse_review_args(arg)
        data = review(category=category, mode=mode, config=config, db=db)
        return render_review_payload(review_payload(data))

    return render_help()


def can_natural_capture(db: WisdomDB, config: WisdomConfig) -> bool:
    return service_can_natural_capture(config=config, db=db)


def _split(raw_args: str) -> tuple[str, str]:
    if not raw_args:
        return "", ""
    parts = raw_args.split(maxsplit=1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def _parse_id(text: str) -> int | None:
    text = (text or "").strip().lstrip("#")
    if not text:
        return None
    try:
        value = int(text.split()[0])
    except ValueError:
        return None
    return value if value > 0 else None


def _parse_review_args(text: str) -> tuple[str | None, str]:
    token = (text or "").strip().lower().replace("-", "_")
    if token in {"unapplied", "high_potential", "all"}:
        return None, token
    if token in {"business", "investing", "health", "life", "inbox"}:
        return token, "needs_review"
    return None, "needs_review"
