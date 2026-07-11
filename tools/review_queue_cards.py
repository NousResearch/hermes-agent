"""Persistent review-queue cards for Telegram/Obsidian workflows."""

from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from hermes_constants import get_hermes_home
except Exception:  # pragma: no cover
    def get_hermes_home() -> Path:
        return Path(os.getenv("HERMES_HOME", "~/.hermes")).expanduser()


Card = Dict[str, Any]
ButtonRows = List[List[Tuple[str, str]]]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cards (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    thesis TEXT NOT NULL,
    person TEXT,
    url TEXT,
    body TEXT NOT NULL,
    source TEXT,
    status TEXT NOT NULL,
    obsidian_path TEXT,
    telegram_chat_id TEXT,
    telegram_thread_id TEXT,
    telegram_message_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    decided_by TEXT,
    decision_note TEXT
);
"""

_COLUMNS = [
    "id", "kind", "thesis", "person", "url", "body", "source", "status",
    "obsidian_path", "telegram_chat_id", "telegram_thread_id", "telegram_message_id",
    "created_at", "updated_at", "decided_by", "decision_note",
]

_ACTION_TO_STATUS = {
    "a": "accepted",
    "accept": "accepted",
    "d": "denied_pending_note",
    "deny": "denied_pending_note",
    "r": "needs_rescore",
    "rescore": "needs_rescore",
    "s": "skipped",
    "skip": "skipped",

    "e": "elaborate_requested",
    "elaborate": "elaborate_requested",
    "n": "accepted_no_note",
    "no_note": "accepted_no_note",
    "no-note": "accepted_no_note",
    "dn": "deny_note_requested",
    "deny_note": "deny_note_requested",
    "deny-note": "deny_note_requested",
    "nn": "denied",
    "deny_no_note": "denied",
    "deny-no-note": "denied",
}

_STATUS_LABELS = {
    "pending": "Pending",
    "accepted": "Accepted — add note / No note",
    "accepted_pending_note": "Accepted — add note / No note",
    "denied": "Denied — no note",
    "denied_pending_note": "Denied — add reason / No note",
    "deny_note_requested": "Denied — write reason as next message",
    "denied_with_note": "Denied — reason captured",
    "needs_rescore": "Needs rescore",
    "skipped": "Skipped",

    "elaborate_requested": "Accepted — write note as next message",
    "elaborated": "Accepted — note captured",
    "accepted_no_note": "Accepted — no note",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def db_path() -> Path:
    override = os.getenv("HERMES_REVIEW_QUEUE_DB", "").strip()
    if override:
        path = Path(override).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    home = Path(get_hermes_home())
    home.mkdir(parents=True, exist_ok=True)
    return home / "review_queue.db"


def _parse_telegram_target(target: str) -> Tuple[str, str]:
    target = (target or "").strip()
    if target.startswith("telegram:"):
        target = target.split(":", 1)[1]
    parts = target.split(":") if target else []
    if parts and parts[0].lstrip("-").isdigit():
        return parts[0], parts[1] if len(parts) > 1 else ""
    return "", ""


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(db_path())
    con.row_factory = sqlite3.Row
    con.execute(_SCHEMA)
    return con


def _row_to_card(row: sqlite3.Row | None) -> Optional[Card]:
    if row is None:
        return None
    return {k: row[k] for k in _COLUMNS}


def _thesis_root() -> Path:
    return Path(os.getenv("ANTIDOTE_THESIS_ROOT", "/root/obsidian-vault/Antidote/Thesis")).expanduser()


def _safe_thesis_dir(thesis: str) -> Path:
    # Thesis names are existing Obsidian folder labels. Avoid path traversal.
    clean = thesis.replace("/", "-").replace("\\", "-").strip() or "Inbox"
    return _thesis_root() / clean


def _split_theses(thesis: str) -> List[str]:
    """Return per-thesis labels from a card's thesis field.

    Expert/source cards are canonical global people/sources, and one expert can
    be relevant to several theses. The SQLite schema still stores that as one
    text column, so multi-thesis relevance is represented as a comma-separated
    label list.
    """
    parts = [p.strip() for p in re.split(r"\s*,\s*", str(thesis or "")) if p.strip()]
    return parts or ["Inbox"]


def _card_theses(card: Card) -> List[str]:
    return _split_theses(str(card.get("thesis") or ""))


def _review_queue_path(thesis: str) -> Path:
    return _safe_thesis_dir(thesis) / "Review queue.md"


def _thesis_note_path(thesis: str) -> Path:
    return _safe_thesis_dir(thesis) / f"{thesis}.md"


_EXPERT_ACCEPTED_STATUSES = {"accepted", "accepted_no_note", "elaborated"}
_EXPERT_PENDING_STATUSES = {"pending", "needs_rescore", "accepted_pending_note", "elaborate_requested", "denied_pending_note", "deny_note_requested"}
_STARTUP_ACCEPTED_STATUSES = {"accepted", "accepted_no_note", "elaborated"}


def _people_root() -> Path:
    return _thesis_root() / "Expert Seeds" / "People"


def _slugify_person(text: str) -> str:
    text = (text or "").strip().lower()
    text = text.split("/", 1)[0].strip() if "/" in text else text
    text = text.lstrip("@")
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "unknown-source"


def _safe_expert_filename(text: str) -> str:
    """Return a display-name filename for an expert/source note."""
    text = (text or "").strip()
    text = text.split("/", 1)[0].strip() if "/" in text else text
    text = re.sub(r"^@", "", text).strip()
    text = re.sub(r"[\\/:*?\"<>|]+", "-", text)
    text = re.sub(r"\s+", " ", text).strip(" .-")
    return text[:120] or "Unknown source"


def _expert_note_filename(card: Card) -> str:
    """Canonical People filename, preferring human names over handle slugs."""
    body = str(card.get("body") or "")
    for label in ("Wikilink", "People note", "Expert note"):
        m = re.search(rf"^{label}:\s*\[\[[^\]|]*(?:/|^)([^/\]|]+)(?:\|([^\]]+))?\]\]", body, re.IGNORECASE | re.MULTILINE)
        if m:
            return _safe_expert_filename(m.group(2) or m.group(1))
    person = str(card.get("person") or "")
    if person:
        return _safe_expert_filename(person)
    m = re.search(r"Candidate:\s*([^\n]+)", body, re.IGNORECASE)
    if m:
        return _safe_expert_filename(m.group(1))
    return _safe_expert_filename(_expert_note_slug(card))


def _expert_note_slug(card: Card) -> str:
    body = str(card.get("body") or "")
    m = re.search(r"CV note:\s*\[\[[^\]|]*(?:/|^)([^/\]|]+)\|?[^\]]*\]\]", body, re.IGNORECASE)
    if m:
        return _slugify_person(m.group(1))
    url = str(card.get("url") or "")
    m = re.search(r"x\.com/([^/?#]+)", url, re.IGNORECASE)
    if m:
        return _slugify_person(m.group(1))
    person = str(card.get("person") or "")
    m = re.search(r"@([A-Za-z0-9_]+)", person)
    if m:
        return _slugify_person(m.group(1))
    return _slugify_person(person or str(card.get("id") or "expert"))


def _expert_note_target_dir(status: str) -> Path:
    status = (status or "pending").strip().lower()
    root = _people_root()
    if status in _EXPERT_ACCEPTED_STATUSES:
        return root
    if status.startswith("denied") or status == "skipped":
        return root / "Denied"
    return root / "Pending"


def _find_expert_note(slug: str, filename: str = "") -> Path | None:
    root = _people_root()
    candidates = {
        slug,
        slug.replace("-", "_"),
        slug.replace("_", "-"),
    }
    if filename:
        candidates.update({
            filename,
            filename.replace(" ", "-"),
            filename.replace(" ", "_"),
        })
    for base in (root, root / "Pending", root / "Denied"):
        for cand in candidates:
            p = base / f"{cand}.md"
            if p.exists():
                return p
    wanted = {c.lower() + ".md" for c in candidates}
    for p in root.rglob("*.md"):
        if p.name.lower() in wanted:
            return p
    return None


def _update_status_line(text: str, status: str, card_id: str) -> str:
    label = {
        "accepted": "accepted — active expert seed",
        "accepted_no_note": "accepted — active expert seed",
        "elaborated": "accepted — active expert seed",
        "skipped": "denied/skipped — not an active expert seed",
    }.get(status, "denied — not an active expert seed" if status.startswith("denied") else "pending review")
    line = f"- Review status: {label}; card `{card_id}`"
    if re.search(r"^- (?:Review )?Status: .*$", text, flags=re.MULTILINE):
        return re.sub(r"^- (?:Review )?Status: .*$", line, text, count=1, flags=re.MULTILINE)
    parts = text.splitlines()
    if parts and parts[0].startswith("# "):
        parts.insert(1, "")
        parts.insert(2, line)
        return "\n".join(parts).rstrip() + "\n"
    return line + "\n\n" + text.rstrip() + "\n"


def _ensure_expert_seed_link(card: Card, note_path: Path) -> None:
    status = str(card.get("status") or "")
    if status not in _EXPERT_ACCEPTED_STATUSES:
        return
    for thesis in _card_theses(card):
        path = _safe_thesis_dir(thesis) / "Expert seeds.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else f"# Expert seeds — {thesis}\n\n"
        rel = note_path.relative_to(_thesis_root()).with_suffix("").as_posix()
        link = f"[[Antidote/Thesis/{rel}|{card.get('person') or note_path.stem}]]"
        if note_path.stem.lower() in existing.lower() or str(card.get("url") or "") in existing:
            path.write_text(existing, encoding="utf-8")
            continue
        block = (
            "\n\n"
            f"### {link}\n"
            "- Domain: accepted expert/source seed from Review Queue.\n"
            f"- Why this person/source matters: accepted for [[{thesis}]]; use this source's timeline/posts in future acquisition runs.\n"
            f"- Profile: {card.get('url') or ''}\n"
            f"- Review status: accepted; card `{card.get('id')}`.\n"
            "- Query seeds:\n"
            f"  - `from:{_expert_note_slug(card)}`\n"
        )
        path.write_text(existing.rstrip() + block + "\n", encoding="utf-8")


def _apply_expert_decision_to_people_note(card: Card) -> None:
    if str(card.get("kind") or "").lower() != "expert":
        return
    status = str(card.get("status") or "pending").strip().lower()
    slug = _expert_note_slug(card)
    filename = _expert_note_filename(card)
    target_dir = _expert_note_target_dir(status)
    target_dir.mkdir(parents=True, exist_ok=True)
    source = _find_expert_note(slug, filename)
    target = target_dir / f"{filename}.md"
    if source is None:
        related = "\n".join(
            f"  - [[Antidote/Thesis/{thesis}/{thesis}|{thesis}]]"
            for thesis in _card_theses(card)
        )
        target.write_text(
            f"# {card.get('person') or slug}\n\n"
            f"- Profile: {card.get('url') or ''}\n"
            f"- Related theses:\n{related}\n"
            "- Review status: pending review\n\n"
            "## CV-quality profile\n\nPending CV/provenance research.\n\n"
            "## Sources\n\n"
            f"- Review Queue card: `{card.get('id')}`\n",
            encoding="utf-8",
        )
    else:
        if source.resolve() != target.resolve():
            if target.exists():
                target.write_text(source.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
                source.unlink()
            else:
                source.rename(target)
    text = target.read_text(encoding="utf-8", errors="ignore")
    text = _update_status_line(text, status, str(card.get("id") or ""))
    target.write_text(text, encoding="utf-8")
    _ensure_expert_seed_link(card, target)

    rel = target.relative_to(_thesis_root()).with_suffix("").as_posix()
    cv_link = f"CV note: [[Antidote/Thesis/{rel}|{target.stem}]]"
    body = str(card.get("body") or "")
    if re.search(r"^CV note:.*$", body, flags=re.MULTILINE):
        body = re.sub(r"^CV note:.*$", cv_link, body, count=1, flags=re.MULTILINE)
    elif body.strip():
        body = body.rstrip() + "\n" + cv_link
    else:
        body = cv_link

    now = _now()
    with _connect() as con:
        con.execute(
            "UPDATE cards SET obsidian_path = ?, body = ?, updated_at = ? WHERE id = ?",
            (str(target), body, now, str(card.get("id") or "")),
        )


def _startup_root(thesis: str) -> Path:
    return _safe_thesis_dir(thesis) / "Startups"


def _startup_note_target_dir(thesis: str, status: str) -> Path:
    status = (status or "pending").strip().lower()
    root = _startup_root(thesis)
    if status in _STARTUP_ACCEPTED_STATUSES:
        return root
    if status.startswith("denied") or status == "skipped":
        return root / "Denied"
    return root / "Pending"


def _safe_note_filename(text: str) -> str:
    text = re.sub(r"[\\/:*?\"<>|]+", "-", (text or "").strip())
    text = re.sub(r"\s+", " ", text).strip(" .-")
    return text[:120] or "Unknown startup"


def _startup_name(card: Card) -> str:
    fields, _ = _body_fields(str(card.get("body") or ""))
    name = fields.get("startup") or fields.get("company") or str(card.get("person") or "").strip()
    if name:
        return _safe_note_filename(name.lstrip("@"))
    url = str(card.get("url") or "")
    m = re.search(r"https?://(?:www\.)?([^/?#]+)", url, re.IGNORECASE)
    if m:
        host = m.group(1).split(":", 1)[0]
        return _safe_note_filename(host.removesuffix(".com"))
    return _safe_note_filename(str(card.get("id") or "Unknown startup"))


def _find_startup_note(thesis: str, name: str) -> Path | None:
    root = _startup_root(thesis)
    wanted = f"{name}.md".lower()
    for base in (root, root / "Pending", root / "Denied"):
        p = base / f"{name}.md"
        if p.exists():
            return p
    if root.exists():
        for p in root.rglob("*.md"):
            if p.name.lower() == wanted:
                return p
    return None


def _startup_status_label(status: str) -> str:
    status = (status or "pending").strip().lower()
    if status in _STARTUP_ACCEPTED_STATUSES:
        return "accepted — active startup map entry"
    if status.startswith("denied") or status == "skipped":
        return "denied/skipped — not active"
    return "pending review"


def _apply_startup_decision_to_note(card: Card) -> None:
    fields, _ = _body_fields(str(card.get("body") or ""))
    status = str(card.get("status") or "pending")
    for thesis in _card_theses(card):
        name = _startup_name(card)
        target_dir = _startup_note_target_dir(thesis, status)
        target_dir.mkdir(parents=True, exist_ok=True)
        existing = _find_startup_note(thesis, name)
        target = target_dir / f"{name}.md"
        if existing and existing != target:
            target.write_text(existing.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
            try:
                existing.unlink()
            except OSError:
                pass
        website = fields.get("website") or fields.get("profile") or str(card.get("url") or "")
        taxonomy = fields.get("taxonomy fit") or fields.get("taxonomy") or fields.get("category")
        market_map = fields.get("market map")
        founders = fields.get("founders") or fields.get("credentials / provenance")
        backers = fields.get("backers")
        traction = fields.get("traction") or fields.get("stage")
        existing_text = target.read_text(encoding="utf-8", errors="ignore") if target.exists() else ""
        body = str(card.get("body") or "").strip()
        note = (
            f"# {name}\n\n"
            f"- Review status: {_startup_status_label(status)}; card `{card.get('id')}`\n"
            f"- Thesis: [[{thesis}]]\n"
            f"- Website / profile: {website}\n"
            f"- Found via: {card.get('source') or ''}\n"
            f"- Taxonomy fit: {taxonomy or 'TBD'}\n"
            f"- Market map source: {market_map or ''}\n"
            f"- Founders / provenance: {founders or ''}\n"
            f"- Backers: {backers or ''}\n"
            f"- Traction / stage: {traction or ''}\n"
            f"- Decision note: {card.get('decision_note') or ''}\n\n"
            "## Research notes\n\n"
            f"{body}\n\n"
            "## Open questions\n\n"
            "- What exact taxonomy bucket should this startup sit in?\n"
            "- Is it a primary thesis signal, adjacent company, competitor, or reject?\n"
            "- What evidence proves traction: revenue, usage, funding, founder pedigree, customer logos, or technical demo?\n"
        )
        if existing_text and "## Review history" in existing_text:
            note += "\n" + existing_text.split("## Review history", 1)[1].lstrip()
        note += f"\n## Review history\n\n- {_now()} — {status}; decided by {card.get('decided_by') or ''}; note: {card.get('decision_note') or ''}.\n"
        target.write_text(note.rstrip() + "\n", encoding="utf-8")
        _ensure_startup_index_link(thesis, target, card, taxonomy)
        with _connect() as con:
            con.execute(
                "UPDATE cards SET obsidian_path = ?, updated_at = ? WHERE id = ?",
                (str(target), _now(), str(card.get("id") or "")),
            )


def _ensure_startup_index_link(thesis: str, note_path: Path, card: Card, taxonomy: str | None = None) -> None:
    index = _startup_root(thesis) / "Startup map.md"
    index.parent.mkdir(parents=True, exist_ok=True)
    existing = index.read_text(encoding="utf-8", errors="ignore") if index.exists() else f"# Startup map — {thesis}\n\n## Taxonomy\n\n## Startups\n"
    rel = note_path.relative_to(_thesis_root()).with_suffix("").as_posix()
    link = f"[[Antidote/Thesis/{rel}|{note_path.stem}]]"
    if note_path.stem.lower() in existing.lower():
        return
    if "## Startups" not in existing:
        existing = existing.rstrip() + "\n\n## Startups\n"
    line = f"\n- {link} — {taxonomy or 'TBD'}; status `{card.get('status')}`; card `{card.get('id')}`."
    index.write_text(existing.rstrip() + line + "\n", encoding="utf-8")


def create_card(
    *,
    card_id: str,
    kind: str,
    thesis: str,
    body: str,
    person: str = "",
    url: str = "",
    source: str = "",
    target: str = "",
    telegram_chat_id: str = "",
    telegram_thread_id: str = "",
    telegram_message_id: str = "",
    obsidian_path: str = "",
) -> Card:
    """Create or update a persistent review-card row."""
    card_id = str(card_id).strip()
    if not card_id:
        raise ValueError("card_id is required")
    kind = (kind or "evidence").strip().lower()
    if kind not in {"evidence", "expert", "startup", "job"}:
        raise ValueError("kind must be evidence, expert, startup, or job")
    if target and (not telegram_chat_id or not telegram_thread_id):
        parsed_chat_id, parsed_thread_id = _parse_telegram_target(target)
        telegram_chat_id = telegram_chat_id or parsed_chat_id
        telegram_thread_id = telegram_thread_id or parsed_thread_id
    now = _now()
    with _connect() as con:
        existing = get_card(card_id)
        created_at = existing["created_at"] if existing else now
        status = existing["status"] if existing else "pending"
        values = {
            "id": card_id,
            "kind": kind,
            "thesis": thesis.strip() or "Inbox",
            "person": person or "",
            "url": url or "",
            "body": body.strip(),
            "source": source or "",
            "status": status,
            "obsidian_path": obsidian_path or "",
            "telegram_chat_id": str(telegram_chat_id or ""),
            "telegram_thread_id": str(telegram_thread_id or ""),
            "telegram_message_id": str(telegram_message_id or ""),
            "created_at": created_at,
            "updated_at": now,
            "decided_by": existing.get("decided_by", "") if existing else "",
            "decision_note": existing.get("decision_note", "") if existing else "",
        }
        placeholders = ", ".join([":" + c for c in _COLUMNS])
        updates = ", ".join([f"{c}=excluded.{c}" for c in _COLUMNS if c != "id"])
        con.execute(
            f"INSERT INTO cards ({', '.join(_COLUMNS)}) VALUES ({placeholders}) "
            f"ON CONFLICT(id) DO UPDATE SET {updates}",
            values,
        )
    card = get_card(card_id)
    assert card is not None
    return card


def get_card(card_id: str) -> Optional[Card]:
    with _connect() as con:
        row = con.execute("SELECT * FROM cards WHERE id = ?", (str(card_id),)).fetchone()
    return _row_to_card(row)


def set_telegram_message_id(card_id: str, message_id: str) -> Optional[Card]:
    now = _now()
    with _connect() as con:
        con.execute(
            "UPDATE cards SET telegram_message_id = ?, updated_at = ? WHERE id = ?",
            (str(message_id), now, str(card_id)),
        )
    return get_card(card_id)


def resolve_card(card_id: str, action: str, user: str = "", note: str = "") -> Card:
    action_key = (action or "").strip().lower()
    status = _ACTION_TO_STATUS.get(action_key)
    if not status:
        raise ValueError(f"unknown review-card action: {action}")
    now = _now()
    with _connect() as con:
        row = con.execute("SELECT * FROM cards WHERE id = ?", (str(card_id),)).fetchone()
        if row is None:
            raise KeyError(f"review card not found: {card_id}")
        con.execute(
            "UPDATE cards SET status = ?, updated_at = ?, decided_by = ?, decision_note = ? WHERE id = ?",
            (status, now, user or "", note or "", str(card_id)),
        )
    card = get_card(card_id)
    assert card is not None
    _append_obsidian_decision(card)
    return card


def capture_pending_note(
    *,
    telegram_chat_id: str,
    telegram_thread_id: str = "",
    user: str = "",
    note: str,
) -> Optional[Card]:
    """Attach a typed note to the newest review card awaiting a typed note in a topic."""
    note = (note or "").strip()
    if not note:
        return None
    chat_id = str(telegram_chat_id or "")
    thread_id = str(telegram_thread_id or "")
    with _connect() as con:
        row = con.execute(
            """
            SELECT * FROM cards
            WHERE status IN ('elaborate_requested', 'deny_note_requested')
              AND telegram_chat_id = ?
              AND COALESCE(telegram_thread_id, '') = ?
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (chat_id, thread_id),
        ).fetchone()
        if row is None:
            return None
        card_id = row["id"]
        next_status = "denied_with_note" if row["status"] == "deny_note_requested" else "elaborated"
        now = _now()
        con.execute(
            "UPDATE cards SET status = ?, updated_at = ?, decided_by = ?, decision_note = ? WHERE id = ?",
            (next_status, now, user or row["decided_by"] or "", note, card_id),
        )
    card = get_card(card_id)
    if card is not None:
        _append_obsidian_decision(card)
    return card


def status_label(card: Card) -> str:
    return _STATUS_LABELS.get(card.get("status") or "pending", str(card.get("status") or "pending"))


def _display_card_id(card_id: str) -> str:
    """Return a readable identifier while keeping the full ID in callback data."""
    card_id = str(card_id or "").strip()
    if not card_id:
        return ""
    if len(card_id) <= 32:
        return card_id
    # Long slugs make Telegram cards unreadable. Show a stable short handle instead.
    import hashlib

    return f"…{card_id[-12:]} · {hashlib.sha1(card_id.encode()).hexdigest()[:6]}"


def _body_fields(body: str) -> Tuple[Dict[str, str], str]:
    """Parse common `Label: value` card bodies and return remaining free text."""
    fields: Dict[str, str] = {}
    free: List[str] = []
    current_key = ""
    known = {
        "evidence id", "expert id", "startup id", "startup", "company", "person", "post", "profile", "website", "found via",
        "candidate", "rationale", "background/cv", "credentials / provenance",
        "cv note", "recommendation", "why follow", "source", "fit / caveats",
        "caveat", "source strength", "why it matters", "top thesis-adjacent posts",
        "category", "taxonomy", "taxonomy fit", "market map", "founders", "backers",
        "traction", "stage", "competitors", "watch for", "job title", "role", "location",
        "work mode", "alert", "alert settings", "job id", "signals", "fit", "why interesting",
    }
    for raw in str(body or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            norm = key.strip().lower()
            if norm in known:
                current_key = norm
                fields[current_key] = value.strip()
                continue
        if current_key:
            sep = "\n" if current_key in {"top thesis-adjacent posts", "background/cv", "credentials / provenance"} else " "
            fields[current_key] = (fields[current_key] + sep + line).strip()
        else:
            free.append(line)
    return fields, "\n".join(free).strip()


def render_card_text(card: Card) -> str:
    kind = (card.get("kind") or "evidence").strip().lower()
    fields, free_body = _body_fields(str(card.get("body") or ""))
    title = "🧑 Expert seed candidate" if kind == "expert" else "🏢 Startup candidate" if kind == "startup" else "💼 Job opportunity" if kind == "job" else "🧾 Evidence candidate"
    thesis_label = "Theses" if kind == "expert" or len(_card_theses(card)) > 1 else "Thesis"
    lines = [title, f"{thesis_label}: {card.get('thesis', '')}".strip()]

    person = fields.get("person") or str(card.get("person") or "").strip()
    if person:
        lines.append(f"Person: {person}")

    if kind == "job":
        role = fields.get("job title") or fields.get("role") or fields.get("candidate") or free_body
        company = fields.get("company") or person
        location = fields.get("location")
        work_mode = fields.get("work mode")
        alert = fields.get("alert") or str(card.get("source") or "").strip()
        alert_settings = fields.get("alert settings")
        job_id = fields.get("job id")
        signals = fields.get("signals")
        fit = fields.get("fit") or fields.get("why interesting") or fields.get("rationale")
        caveat = fields.get("caveat") or fields.get("fit / caveats")
        link = str(card.get("url") or "").strip()

        if role:
            lines.append(f"Role: {role}")
        if company:
            lines.append(f"Company: {company}")
        if location:
            suffix = f" ({work_mode})" if work_mode else ""
            lines.append(f"Location: {location}{suffix}")
        elif work_mode:
            lines.append(f"Work mode: {work_mode}")
        if link:
            lines.append(f"Link: {link}")
        if job_id:
            lines.append(f"LinkedIn job ID: {job_id}")
        if alert:
            lines.append(f"Alert: {alert}")
        if alert_settings:
            lines.append(f"Alert settings: {alert_settings}")
        if signals:
            lines.append(f"Signals: {signals}")
        if fit:
            lines.extend(["", f"Why interesting: {fit}"])
        if caveat:
            lines.append(f"Caveat: {caveat}")
    elif kind == "expert":
        profile = fields.get("profile") or str(card.get("url") or "").strip()
        found_via = fields.get("found via") or str(card.get("source") or "").strip()
        candidate = fields.get("rationale") or fields.get("candidate") or free_body
        background = fields.get("background/cv") or fields.get("credentials / provenance")
        cv_note = fields.get("cv note")
        recommendation = fields.get("recommendation")
        why_follow = fields.get("why follow")
        top_posts = fields.get("top thesis-adjacent posts")
        source_strength = fields.get("source strength")
        caveat = fields.get("caveat") or fields.get("fit / caveats")

        if profile:
            lines.append(f"Profile: {profile}")
        if found_via and found_via != profile:
            lines.append(f"Found via: {found_via}")
        if cv_note:
            lines.append(f"CV note: {cv_note}")
        if recommendation:
            lines.extend(["", f"Recommendation: {recommendation}"])
        if candidate:
            lines.extend(["", f"Rationale: {candidate}"])
        if background:
            lines.extend(["", "Credentials / provenance:", background])
        if top_posts:
            lines.extend(["", "Top thesis-adjacent posts:", top_posts])
        if why_follow:
            lines.extend(["", f"Watch for: {why_follow}"])
        if source_strength:
            lines.append(f"Source strength: {source_strength}")
        if caveat:
            lines.append(f"Caveat: {caveat}")
    elif kind == "startup":
        startup = fields.get("startup") or fields.get("company") or person
        website = fields.get("website") or fields.get("profile") or str(card.get("url") or "").strip()
        found_via = fields.get("found via") or str(card.get("source") or "").strip()
        candidate = fields.get("rationale") or fields.get("candidate") or free_body
        taxonomy = fields.get("taxonomy fit") or fields.get("taxonomy") or fields.get("category")
        market_map = fields.get("market map")
        founders = fields.get("founders") or fields.get("credentials / provenance")
        backers = fields.get("backers")
        traction = fields.get("traction") or fields.get("stage")
        competitors = fields.get("competitors")
        why = fields.get("why it matters")
        caveat = fields.get("caveat") or fields.get("fit / caveats")
        watch = fields.get("watch for") or fields.get("why follow")

        if startup:
            lines.append(f"Startup: {startup}")
        if website:
            lines.append(f"Website: {website}")
        if found_via and found_via != website:
            lines.append(f"Found via: {found_via}")
        if taxonomy:
            lines.append(f"Taxonomy fit: {taxonomy}")
        if market_map:
            lines.append(f"Market map: {market_map}")
        if candidate:
            lines.extend(["", f"Rationale: {candidate}"])
        if why:
            lines.append(f"Why it matters: {why}")
        if founders:
            lines.extend(["", "Founders / provenance:", founders])
        if backers:
            lines.append(f"Backers: {backers}")
        if traction:
            lines.append(f"Traction / stage: {traction}")
        if competitors:
            lines.append(f"Competitors: {competitors}")
        if watch:
            lines.append(f"Watch for: {watch}")
        if caveat:
            lines.append(f"Caveat: {caveat}")
    else:
        post = fields.get("post") or str(card.get("url") or "").strip()
        found_via = fields.get("found via") or str(card.get("source") or "").strip()
        candidate = fields.get("rationale") or fields.get("candidate") or free_body
        why = fields.get("why it matters")
        caveats = fields.get("fit / caveats")

        if post:
            lines.append(f"Post: {post}")
        if found_via and found_via != post:
            lines.append(f"Found via: {found_via}")
        if candidate:
            lines.extend(["", f"Rationale: {candidate}"])
        if why:
            lines.append(f"Why it matters: {why}")
        if caveats:
            lines.append(f"Caveats: {caveats}")

    # If the body had no explicit Rationale/Candidate field, free_body was already
    # rendered above as the Rationale. Only append free-form leftovers when a
    # structured rationale/candidate field existed and free_body is truly extra.
    if free_body and (fields.get("rationale") or fields.get("candidate")) and kind != "expert":
        lines.extend(["", free_body])
    if card.get("decision_note"):
        lines.extend(["", f"Note: {card['decision_note']}"])

    display_id = _display_card_id(str(card.get("id") or ""))
    footer = f"Status: {status_label(card)}"
    if display_id:
        footer += f" · Card: {display_id}"
    lines.extend(["", footer])
    return "\n".join(line for line in lines if line is not None).strip()


def build_button_rows(card: Card) -> ButtonRows:
    card_id = str(card["id"])
    if (card.get("kind") or "").strip().lower() == "job":
        if card.get("status") in {"pending", "needs_rescore"}:
            return [[("✅ Accept", f"rq:n:{card_id}"), ("⏭ Skip", f"rq:s:{card_id}")]]
        return []
    if card.get("status") in {"accepted", "accepted_pending_note"}:
        return [[("📝 Add accepted note", f"rq:e:{card_id}"), ("✓ No note", f"rq:n:{card_id}")]]
    if card.get("status") == "denied_pending_note":
        return [[("📝 Add denial reason", f"rq:dn:{card_id}"), ("✓ No note", f"rq:nn:{card_id}")]]
    if card.get("status") in {"pending", "needs_rescore"}:
        kind = (card.get("kind") or "").strip().lower()
        if kind in {"evidence", "expert"}:
            return [
                [("✅ Accept", f"rq:a:{card_id}"), ("❌ Deny", f"rq:d:{card_id}")],
                [("🔁 Rescore", f"rq:r:{card_id}")],
            ]
        return [
            [("✅ Accept", f"rq:a:{card_id}"), ("❌ Deny", f"rq:d:{card_id}")],
            [("🔁 Rescore", f"rq:r:{card_id}"), ("⏭ Skip", f"rq:s:{card_id}")],
        ]
    return []


def render_keyboard_rows(card: Card) -> ButtonRows:
    """Backward-compatible/readable alias for review-card button rows."""
    return build_button_rows(card)


def button_rows_for_card(card: Card) -> ButtonRows:
    """Compatibility alias used by tests and older callers."""
    return build_button_rows(card)


def apply_decision_to_obsidian(card: Card, action: str = "") -> None:
    """Compatibility wrapper for appending review decisions to Obsidian."""
    if action:
        card = dict(card)
        card["decision_note"] = (card.get("decision_note") or f"action: {action}")
    _append_obsidian_decision(card)


def callback_label(action: str, card: Card) -> str:
    return status_label(card)


def _launch_job_accept_workflow(card: Card) -> None:
    """Start Antidote's job-accept side effects without blocking Telegram callbacks."""
    status = str(card.get("status") or "").strip().lower()
    if status not in {"accepted", "accepted_no_note", "elaborated"}:
        return
    card_id = str(card.get("id") or "").strip()
    if not card_id:
        return
    script = Path(os.getenv("JOB_ACCEPT_WORKFLOW_SCRIPT", str(get_hermes_home() / "scripts" / "job_accept_workflow.py"))).expanduser()
    if not script.exists():
        return
    log_dir = Path(get_hermes_home()) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "job_accept_workflow.log"
    try:
        log = log_path.open("ab")
        env = os.environ.copy()
        env.setdefault("HERMES_HOME", str(get_hermes_home()))
        import subprocess

        subprocess.Popen(
            [os.getenv("PYTHON", "python"), str(script), card_id],
            cwd="/usr/local/lib/hermes-agent" if Path("/usr/local/lib/hermes-agent").exists() else None,
            env=env,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    except Exception:
        # Callback UX must not fail just because a background side-effect failed.
        return


def _append_obsidian_decision(card: Card) -> None:
    kind = str(card.get("kind") or "").lower()
    if kind == "job":
        _launch_job_accept_workflow(card)
        return
    if kind == "expert":
        _apply_expert_decision_to_people_note(card)
        return
    if kind == "startup":
        _apply_startup_decision_to_note(card)
    for thesis in _card_theses(card):
        _append_obsidian_decision_for_thesis(card, thesis)


def _append_obsidian_decision_for_thesis(card: Card, thesis: str) -> None:
    path = _review_queue_path(thesis)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = _now()
    existing_text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    block = [""]
    if "Telegram review decisions" not in existing_text:
        block.append("## Telegram review decisions")
    block.extend([
        f"### {stamp} — {card['id']} — {card['status']}",
        f"- Type: {card['kind']}",
        f"- Theses: {card.get('thesis') or thesis}",
        f"- Person: {card.get('person') or ''}",
        f"- URL: {card.get('url') or ''}",
        f"- Decided by: {card.get('decided_by') or ''}",
        f"- Decision note: {card.get('decision_note') or ''}",
        f"- Body: {card.get('body') or ''}",
    ])
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(block) + "\n")

    # Evidence review-card decisions belong in the local Review queue log.
    # Do not append "Evidence to elaborate" / free-text elaboration blocks to the
    # thesis page itself: accepted evidence is promoted into the thesis evidence
    # callout by the review/evidence synthesis workflow, while notes stay attached
    # to the review decision for that card.


def list_cards(status: Optional[str] = None) -> List[Card]:
    with _connect() as con:
        if status:
            rows = con.execute("SELECT * FROM cards WHERE status = ? ORDER BY created_at DESC", (status,)).fetchall()
        else:
            rows = con.execute("SELECT * FROM cards ORDER BY created_at DESC").fetchall()
    return [c for c in (_row_to_card(row) for row in rows) if c is not None]
