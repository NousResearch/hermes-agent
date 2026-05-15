"""Plane tools — Hermes action layer for Plane work items.

Plane remains the source of truth. These tools provide explicit read/create/update
operations and an opt-in bridge to Hermes kanban execution tasks.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from hermes_constants import get_hermes_home
from hermes_cli.env_loader import load_hermes_dotenv
from tools.plane_client import PlaneAPIError, PlaneClient
from tools.registry import registry, tool_error

PROJECT_KEY_FALLBACK = "AIFACTORY"
DEFAULT_WORKDIR_BASE = "/home/emeric/AI Factory"


def _ok(**fields: Any) -> str:
    return json.dumps({"success": True, **fields}, ensure_ascii=False)


def _check_plane_requirements() -> bool:
    try:
        load_hermes_dotenv(hermes_home=get_hermes_home())
    except OSError:
        # Missing or unreadable .env is non-fatal: env vars may be exported
        # in the shell. Any other failure (parse error, programmer bug) must
        # surface and is not swallowed here.
        pass
    return all(
        (os.getenv(name) or "").strip()
        for name in ("PLANE_API_KEY", "PLANE_WORKSPACE", "PLANE_PROJECT_ID")
    )


def get_plane_client() -> PlaneClient:
    return _client()


# Compatibility alias for earlier local tests and drafts.
def _client() -> PlaneClient:
    return PlaneClient.from_env()


def _client_workspace(client: PlaneClient) -> str:
    config = getattr(client, "config", None)
    return str(getattr(config, "workspace_slug", None) or getattr(client, "workspace_slug", ""))


def _client_project_id(client: PlaneClient) -> str:
    config = getattr(client, "config", None)
    return str(getattr(config, "project_id", None) or getattr(client, "project_id", ""))


def _project_key(client: PlaneClient, project: Optional[dict[str, Any]] = None) -> str:
    project = project or {}
    key = (
        project.get("identifier")
        or project.get("project_identifier")
        or project.get("key")
        or ""
    )
    if not key and hasattr(client, "get_project_identifier"):
        try:
            key = client.get_project_identifier()
        except Exception:
            key = ""
    return str(key or PROJECT_KEY_FALLBACK).strip().upper()


def _sequence_id(item: dict[str, Any]) -> Optional[int]:
    value = item.get("sequence_id") or item.get("sequence")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_state_payload(client: Optional[PlaneClient], item: dict[str, Any]) -> Optional[dict[str, Any]]:
    state = item.get("state")
    if isinstance(state, dict):
        return state
    raw = str(state or "").strip()
    if not raw or client is None or not hasattr(client, "list_states"):
        return None
    try:
        for candidate in client.list_states():
            candidate_id = str(candidate.get("id") or "").strip()
            if candidate_id == raw:
                return candidate
    except Exception:
        return None
    return None


def _state_name(item: dict[str, Any], client: Optional[PlaneClient] = None) -> str:
    resolved = _resolve_state_payload(client, item)
    if resolved is not None:
        return str(resolved.get("name") or resolved.get("id") or "").strip()
    state = item.get("state")
    if isinstance(state, dict):
        return str(state.get("name") or state.get("id") or "").strip()
    return str(state or "").strip()


def _state_id(item: dict[str, Any], client: Optional[PlaneClient] = None) -> Optional[str]:
    resolved = _resolve_state_payload(client, item)
    if resolved is not None:
        return str(resolved.get("id") or "").strip() or None
    state = item.get("state")
    if isinstance(state, dict):
        return str(state.get("id") or "").strip() or None
    raw = str(state or "").strip()
    return raw or None


def _state_group(state: dict[str, Any]) -> Optional[str]:
    group = state.get("group") or state.get("state_group")
    return str(group).strip() if group is not None and str(group).strip() else None


def _is_cancelled_state(item: dict[str, Any], client: Optional[PlaneClient] = None) -> bool:
    group = None
    resolved = _resolve_state_payload(client, item)
    if isinstance(resolved, dict):
        group = _state_group(resolved)
    state_name = _state_name(item, client)
    return str(group or "").strip().casefold() == "cancelled" or state_name.casefold() == "cancelled"


def _labels(item: dict[str, Any]) -> list[str]:
    raw = item.get("labels") or []
    if not isinstance(raw, list):
        return []
    return [str(x.get("name") if isinstance(x, dict) else x).strip() for x in raw if str(x).strip()]


def _assignees(item: dict[str, Any]) -> list[str]:
    raw = item.get("assignees") or []
    if not isinstance(raw, list):
        return []
    out = []
    for assignee in raw:
        if isinstance(assignee, dict):
            name = assignee.get("display_name") or assignee.get("name") or assignee.get("email") or assignee.get("id")
        else:
            name = assignee
        if str(name or "").strip():
            out.append(str(name).strip())
    return out


def _summarize_user(user: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": user.get("id"),
        "email": user.get("email"),
        "display_name": user.get("display_name") or user.get("name") or user.get("first_name"),
    }


def _summarize_project(project: dict[str, Any], client: PlaneClient) -> dict[str, Any]:
    return {
        "id": project.get("id") or _client_project_id(client),
        "name": project.get("name"),
        "identifier": _project_key(client, project),
    }


def _plane_url(client: PlaneClient, item: dict[str, Any], project: Optional[dict[str, Any]] = None) -> str:
    seq = _sequence_id(item)
    key = _project_key(client, project)
    workspace = _client_workspace(client)
    project_id = _client_project_id(client)
    if seq is not None:
        return f"https://app.plane.so/{workspace}/projects/{project_id}/issues/{key}-{seq}"
    wid = item.get("id") or ""
    return f"https://app.plane.so/{workspace}/projects/{project_id}/issues/{wid}"


def _strip_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    return " ".join(html.unescape(text).split())


def _compact_project(project: dict[str, Any]) -> dict[str, Any]:
    """Compact project projection returned by read tools by default."""
    return {
        "id": project.get("id"),
        "name": project.get("name"),
        "identifier": project.get("identifier") or project.get("project_identifier") or project.get("key"),
    }


def _compact_state(state: dict[str, Any], count: int = 0) -> dict[str, Any]:
    """Compact state projection returned by read tools by default."""
    return {
        "id": state.get("id"),
        "name": state.get("name") or state.get("id"),
        "group": _state_group(state),
        "count": count,
    }


def _summarize_item(client: PlaneClient, item: dict[str, Any], project: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Compact work-item projection returned by read tools by default.

    Contract: {id, sequence_id, readable_id, name, state_name, state_id,
    priority, labels, assignees_names, url}. Full Plane payloads are only
    returned by read tools when verbose=True.
    """
    seq = _sequence_id(item)
    key = _project_key(client, project)
    readable_id = f"{key}-{seq}" if seq is not None else None
    return {
        "id": item.get("id"),
        "sequence_id": seq,
        "readable_id": readable_id,
        "name": item.get("name") or item.get("title"),
        "priority": item.get("priority"),
        "state_name": _state_name(item, client),
        "state_id": _state_id(item, client),
        "labels": _labels(item),
        "assignees_names": _assignees(item),
        "url": _plane_url(client, item, project),
    }


def _enrich_item(client: PlaneClient, item: dict[str, Any], project: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    enriched = dict(item)
    enriched.update(_summarize_item(client, item, project))
    return enriched


def _matches_filter(value: Optional[str], candidates: Iterable[str]) -> bool:
    if value is None or str(value).strip() == "":
        return True
    needle = str(value).strip().casefold()
    return any(needle in str(candidate).casefold() for candidate in candidates)


def _filter_items(client: Optional[PlaneClient], items: list[dict[str, Any]], args: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in items:
        if not _matches_filter(args.get("state"), [_state_name(item, client), _state_id(item, client) or ""]):
            continue
        if not _matches_filter(args.get("label"), _labels(item)):
            continue
        if not _matches_filter(args.get("assignee"), _assignees(item)):
            continue
        if args.get("priority") and str(item.get("priority") or "").casefold() != str(args["priority"]).casefold():
            continue
        query = args.get("query")
        if query:
            text = " ".join(
                str(item.get(field) or "")
                for field in ("name", "title", "description_html", "description_stripped")
            )
            if str(query).casefold() not in text.casefold():
                continue
        out.append(item)
    return out


def _resolve_item(client: PlaneClient, *, work_item_id: Optional[str] = None, sequence_id: Optional[Any] = None) -> dict[str, Any]:
    if work_item_id:
        try:
            return client.get_work_item(str(work_item_id))
        except TypeError:
            return client.get_work_item(work_item_id=str(work_item_id))
    if sequence_id is not None:
        seq = int(sequence_id)
        for item in client.list_work_items():
            if _sequence_id(item) == seq:
                return item
        try:
            return client.get_work_item(sequence_id=seq)
        except TypeError:
            pass
    raise ValueError("work_item_id or sequence_id is required")


def _markdown_to_html(markdown: str) -> str:
    escaped = html.escape(str(markdown)).replace("\n", "<br>")
    return f"<p>{escaped}</p>"


def _comment_markdown(body_markdown: str, *, prefix: bool = True) -> str:
    body = str(body_markdown or "").strip()
    if not body:
        raise ValueError("body_markdown is required")
    if prefix and not body.startswith("[Nova]"):
        return f"[Nova] {body}"
    return body


def _parse_plane_linkage_from_body(body: Optional[str]) -> dict[str, Any]:
    """Extract Plane linkage fields from a Hermes kanban task body.

    ``plane_import_to_kanban`` stores the foreign keys as simple
    ``plane_*: value`` lines in the task body. Keep this parser deliberately
    small and tolerant so imported cards created by older drafts remain usable.
    """
    fields: dict[str, Any] = {}
    for line in str(body or "").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if not key.startswith("plane_"):
            continue
        fields[key] = value.strip()
    if "plane_sequence_id" in fields:
        try:
            fields["plane_sequence_id"] = int(fields["plane_sequence_id"])
        except (TypeError, ValueError):
            fields.pop("plane_sequence_id", None)
    return fields


def _lookup_plane_link_from_kanban_task(hermes_card_id: str) -> dict[str, Any]:
    card_id = str(hermes_card_id or "").strip()
    if not card_id:
        raise ValueError("hermes_card_id is required")

    from hermes_cli import kanban_db as kb

    kb.init_db()
    conn = kb.connect()
    try:
        task = kb.get_task(conn, card_id)
    finally:
        conn.close()
    if task is None:
        raise ValueError(f"Hermes kanban task not found: {card_id}")

    linkage = _parse_plane_linkage_from_body(task.body)
    work_item_id = str(linkage.get("plane_work_item_id") or "").strip()
    sequence_id = linkage.get("plane_sequence_id")
    if not work_item_id and sequence_id is None:
        raise ValueError(f"Hermes kanban task {card_id} is not linked to a Plane work item")
    linkage["hermes_card_id"] = card_id
    return linkage


def _resolve_state_id(client: PlaneClient, value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    if hasattr(client, "resolve_state_id"):
        return client.resolve_state_id(value)
    raw = str(value).strip()
    for state in client.list_states():
        if raw == str(state.get("id") or ""):
            return raw
        if raw.casefold() == str(state.get("name") or "").casefold():
            return str(state.get("id") or "")
    raise ValueError(f"Unknown Plane state: {value}")


def _canonical_external_source(args: dict[str, Any]) -> str:
    return str(args.get("external_source") or "nova-hermes").strip() or "nova-hermes"


def _fallback_external_id(args: dict[str, Any], client: PlaneClient, external_source: str) -> str:
    # Explicit V1 fallback contract: if the caller does not provide an
    # external_id, derive one from the stable creation identity: workspace,
    # project, external source, and normalized name. Optional mutable fields
    # such as description, labels, state, and dates are deliberately excluded
    # so retries with small payload differences still deduplicate.
    name = " ".join(str(args.get("name") or "").split()).casefold()
    if not name:
        raise ValueError("name is required")
    basis = "\0".join([
        _client_workspace(client),
        _client_project_id(client),
        external_source,
        name,
    ])
    digest = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:32]
    return f"plane-create:{digest}"


def _prepare_create_idempotency(args: dict[str, Any], client: PlaneClient) -> tuple[str, str, bool]:
    external_source = _canonical_external_source(args)
    provided = args.get("external_id") is not None and str(args.get("external_id") or "").strip() != ""
    external_id = str(args.get("external_id") or "").strip()
    if not external_id:
        external_id = _fallback_external_id(args, client, external_source)
    return external_source, external_id, provided


def _find_existing_by_external_id(client: PlaneClient, external_source: str, external_id: str) -> Optional[dict[str, Any]]:
    if hasattr(client, "find_work_item_by_external_id"):
        return client.find_work_item_by_external_id(
            external_source=external_source,
            external_id=external_id,
        )
    for item in client.list_work_items():
        if (
            str(item.get("external_source") or "").strip() == external_source
            and str(item.get("external_id") or "").strip() == external_id
        ):
            return item
    return None


def _build_work_item_payload(args: dict[str, Any], client: PlaneClient, *, require_name: bool = False) -> dict[str, Any]:
    # PATCH partial contract: a field passed as None (or absent) is ignored
    # and never sent to Plane. Only explicitly-provided non-null fields are
    # forwarded. Empty list ([]) for assignees/labels is explicit "clear".
    payload: dict[str, Any] = {}
    if args.get("name"):
        payload["name"] = str(args["name"]).strip()
    elif require_name:
        raise ValueError("name is required")

    if args.get("description_html"):
        payload["description_html"] = str(args["description_html"])
    elif args.get("description_markdown"):
        payload["description_html"] = _markdown_to_html(str(args["description_markdown"]))

    for key in ("priority", "start_date", "target_date", "external_id"):
        if args.get(key) is not None:
            payload[key] = args[key]

    if "external_source" in args:
        payload["external_source"] = args.get("external_source") or "nova-hermes"
    elif require_name:
        payload["external_source"] = "nova-hermes"

    state_id = _resolve_state_id(client, args.get("state"))
    if state_id:
        # Live Plane behavior on AI_Factory requires both keys for state
        # transitions. Sending only state_id can be ignored server-side and
        # silently leave the item in its previous or default state.
        payload["state_id"] = state_id
        payload["state"] = state_id

    if args.get("assignees") is not None:
        payload["assignees"] = args.get("assignees") or []

    if args.get("labels") is not None:
        if hasattr(client, "resolve_label_ids"):
            payload["labels"] = client.resolve_label_ids(args.get("labels"))
        else:
            payload["labels"] = args.get("labels") or []

    return {k: v for k, v in payload.items() if v is not None}


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug[:80] or "work-item"


def prepare_workdir(
    sequence_id: int,
    title: str,
    base_dir: str = DEFAULT_WORKDIR_BASE,
    *,
    project_key: Optional[str] = None,
) -> str:
    key = str(project_key or "").strip().upper() or PROJECT_KEY_FALLBACK
    base = Path(base_dir).expanduser()
    workdir = base / f"{key}-{int(sequence_id)}_{_safe_slug(title)}"
    (workdir / "work").mkdir(parents=True, exist_ok=True)
    (workdir / "deliverables").mkdir(parents=True, exist_ok=True)
    readme = workdir / "README.md"
    if not readme.exists():
        readme.write_text(
            f"# {key}-{int(sequence_id)} {title}\n\n"
            "## Objectif\n\n"
            "À compléter.\n\n"
            "## Livrables\n\n"
            "Déposer les livrables dans `deliverables/`.\n",
            encoding="utf-8",
        )
    return str(workdir)


def _handle_board_snapshot(args: dict[str, Any], **kw) -> str:
    try:
        client = get_plane_client()
        project = client.get_project()
        states = client.list_states()
        items = client.list_work_items()
        verbose = bool(args.get("verbose"))
        counts = {str(state.get("name") or state.get("id") or ""): 0 for state in states}
        for item in items:
            state = _state_name(item, client) or "No state"
            counts[state] = counts.get(state, 0) + 1
        state_rows = [_compact_state(state, counts.get(str(state.get("name") or state.get("id") or ""), 0)) for state in states]
        result = {
            "project": _compact_project(project),
            "states": state_rows,
            "counts_by_state": counts,
            "total_items": len(items),
            "items": [_summarize_item(client, item, project) for item in items[: int(args.get("limit") or 20)]],
        }
        if args.get("include_items_per_state"):
            per_state_limit = int(args.get("per_state_limit") or 5)
            grouped: dict[str, list[dict[str, Any]]] = {}
            for item in items:
                state = _state_name(item, client) or "No state"
                grouped.setdefault(state, [])
                if len(grouped[state]) < per_state_limit:
                    grouped[state].append(_summarize_item(client, item, project))
            result["items_by_state"] = grouped
        if verbose:
            result.update({
                "project_payload": project,
                "states_payload": states,
                "items_payload": items,
            })
        return _ok(**result)
    except Exception as exc:
        return tool_error(str(exc))


def _handle_list_work_items(args: dict[str, Any], **kw) -> str:
    try:
        client = get_plane_client()
        project = client.get_project()
        limit = int(args.get("limit") or 50)
        verbose = bool(args.get("verbose"))
        items = _filter_items(client, client.list_work_items(), args)[:limit]
        result = {"items": [_summarize_item(client, item, project) for item in items], "count": len(items)}
        if verbose:
            result["items_payload"] = items
        return _ok(**result)
    except Exception as exc:
        return tool_error(str(exc))


def _handle_get_work_item(args: dict[str, Any], **kw) -> str:
    try:
        client = get_plane_client()
        project = client.get_project()
        item = _resolve_item(
            client,
            work_item_id=args.get("work_item_id"),
            sequence_id=args.get("sequence_id"),
        )
        result = {"item": _summarize_item(client, item, project)}
        if args.get("verbose"):
            result["payload"] = item
            result["enriched_item"] = _enrich_item(client, item, project)
        return _ok(**result)
    except Exception as exc:
        return tool_error(str(exc))


def _handle_create_work_item(args: dict[str, Any], **kw) -> str:
    try:
        client = get_plane_client()
        project = client.get_project()
        external_source, external_id, external_id_provided = _prepare_create_idempotency(args, client)
        create_args = {**args, "external_source": external_source, "external_id": external_id}
        existing = _find_existing_by_external_id(client, external_source, external_id)
        if existing:
            enriched = _enrich_item(client, existing, project)
            return _ok(
                item=enriched,
                created=None,
                already_existed=True,
                external_source=external_source,
                external_id=external_id,
                external_id_generated=not external_id_provided,
            )

        payload = _build_work_item_payload(create_args, client, require_name=True)
        try:
            item = client.create_work_item(payload)
        except PlaneAPIError as exc:
            # Plane server-side uniqueness on external_source + external_id
            # surfaces as 409 when the pre-create lookup missed the existing
            # card. Re-scan and treat the conflict as a clean idempotent hit.
            if exc.status_code == 409:
                recovered = _find_existing_by_external_id(client, external_source, external_id)
                if recovered:
                    enriched = _enrich_item(client, recovered, project)
                    return _ok(
                        item=enriched,
                        created=None,
                        already_existed=True,
                        external_source=external_source,
                        external_id=external_id,
                        external_id_generated=not external_id_provided,
                    )
            raise
        enriched = _enrich_item(client, item, project)
        return _ok(
            item=enriched,
            created=enriched,
            already_existed=False,
            external_source=external_source,
            external_id=external_id,
            external_id_generated=not external_id_provided,
        )
    except Exception as exc:
        return tool_error(str(exc))


def _handle_update_work_item(args: dict[str, Any], **kw) -> str:
    try:
        client = get_plane_client()
        work_item_id = str(args.get("work_item_id") or "").strip()
        if not work_item_id:
            raise ValueError("work_item_id is required")
        payload = _build_work_item_payload(args, client, require_name=False)
        if not payload:
            raise ValueError("at least one updatable field is required")
        item = client.update_work_item(work_item_id, payload)
        return _ok(item=item)
    except Exception as exc:
        return tool_error(str(exc))


def _handle_add_comment(args: dict[str, Any], **kw) -> str:
    try:
        client = get_plane_client()
        item = _resolve_item(
            client,
            work_item_id=args.get("work_item_id"),
            sequence_id=args.get("sequence_id"),
        )
        work_item_id = str(item.get("id") or args.get("work_item_id") or "").strip()
        if not work_item_id:
            raise ValueError("resolved Plane work item has no id")
        body_markdown = _comment_markdown(
            str(args.get("body_markdown") or ""),
            prefix=bool(args.get("prefix", True)),
        )
        comment_html = _markdown_to_html(body_markdown)
        comment = client.add_comment(work_item_id, comment_html)
        project = client.get_project()
        return _ok(
            comment=comment,
            item=_summarize_item(client, item, project),
            comment_html=comment_html,
        )
    except Exception as exc:
        return tool_error(str(exc))


def _handle_sync_progress(args: dict[str, Any], **kw) -> str:
    try:
        hermes_card_id = str(
            args.get("hermes_card_id")
            or kw.get("task_id")
            or os.getenv("HERMES_KANBAN_TASK")
            or ""
        ).strip()
        summary = str(args.get("summary") or args.get("body_markdown") or "").strip()
        if not summary:
            raise ValueError("summary is required")

        linkage = _lookup_plane_link_from_kanban_task(hermes_card_id)
        client = get_plane_client()
        item = _resolve_item(
            client,
            work_item_id=linkage.get("plane_work_item_id"),
            sequence_id=linkage.get("plane_sequence_id"),
        )
        work_item_id = str(item.get("id") or linkage.get("plane_work_item_id") or "").strip()
        if not work_item_id:
            raise ValueError("resolved Plane work item has no id")

        status = str(args.get("status") or "").strip()
        status_update = None
        if status:
            state_id = _resolve_state_id(client, status)
            if not state_id:
                raise ValueError(f"Unknown Plane state: {status}")
            payload = {"state_id": state_id, "state": state_id}
            status_update = client.update_work_item(work_item_id, payload)
            if isinstance(status_update, dict):
                item = status_update

        body_markdown = _comment_markdown(
            summary,
            prefix=bool(args.get("prefix", True)),
        )
        comment_html = _markdown_to_html(body_markdown)
        comment = client.add_comment(work_item_id, comment_html)
        project = client.get_project()
        return _ok(
            hermes_card_id=hermes_card_id,
            plane_linkage=linkage,
            item=_summarize_item(client, item, project),
            plane_url=_plane_url(client, item, project),
            comment=comment,
            comment_html=comment_html,
            status=status or None,
            status_updated=bool(status),
            status_update=status_update,
        )
    except Exception as exc:
        return tool_error(str(exc))


def _handle_check_kanban_links(args: dict[str, Any], **kw) -> str:
    try:
        hermes_card_ids = args.get("hermes_card_ids") or []
        if not isinstance(hermes_card_ids, list) or not hermes_card_ids:
            raise ValueError("hermes_card_ids is required")

        client = get_plane_client()
        project = client.get_project()
        anomalies: list[dict[str, Any]] = []

        for raw_card_id in hermes_card_ids:
            hermes_card_id = str(raw_card_id or "").strip()
            if not hermes_card_id:
                continue
            linkage = _lookup_plane_link_from_kanban_task(hermes_card_id)
            try:
                item = _resolve_item(
                    client,
                    work_item_id=linkage.get("plane_work_item_id"),
                    sequence_id=linkage.get("plane_sequence_id"),
                )
            except PlaneAPIError as exc:
                if exc.status_code != 404:
                    raise
                anomalies.append({
                    "hermes_card_id": hermes_card_id,
                    "status": "missing",
                    "plane_work_item_id": linkage.get("plane_work_item_id"),
                    "plane_sequence_id": linkage.get("plane_sequence_id"),
                    "plane_url": linkage.get("plane_url"),
                })
                continue

            if _is_cancelled_state(item, client):
                anomalies.append({
                    "hermes_card_id": hermes_card_id,
                    "status": "cancelled",
                    "plane_work_item_id": item.get("id") or linkage.get("plane_work_item_id"),
                    "plane_sequence_id": _sequence_id(item) or linkage.get("plane_sequence_id"),
                    "plane_state_id": _state_id(item, client) or linkage.get("plane_state_id"),
                    "plane_state_name": _state_name(item, client),
                    "plane_url": _plane_url(client, item, project),
                })

        return _ok(count=len(anomalies), items=anomalies)
    except Exception as exc:
        return tool_error(str(exc))


def _handle_ping(args: dict[str, Any], **kw) -> str:
    started = time.perf_counter()
    try:
        client = get_plane_client()
        user = client.get_current_user() if hasattr(client, "get_current_user") else {}
        project = client.get_project()
        latency_ms = int(round((time.perf_counter() - started) * 1000))
        return _ok(
            ok=True,
            latency_ms=latency_ms,
            user=user,
            user_email=user.get("email") or user.get("display_email") or "",
            workspace=_client_workspace(client),
            project=project,
            project_name=project.get("name") or "",
            project_identifier=_project_key(client, project),
        )
    except Exception as exc:
        return tool_error(str(exc))


def _handle_prepare_workdir(args: dict[str, Any], **kw) -> str:
    try:
        sequence_id = args.get("sequence_id")
        title = str(args.get("title") or "").strip()
        if sequence_id is None:
            raise ValueError("sequence_id is required")
        if not title:
            raise ValueError("title is required")
        project_key = str(args.get("project_key") or "").strip() or None
        if not project_key and _check_plane_requirements():
            try:
                project_key = get_plane_client().get_project_identifier() or None
            except Exception:
                project_key = None
        workdir = prepare_workdir(
            int(sequence_id),
            title,
            args.get("base_dir") or DEFAULT_WORKDIR_BASE,
            project_key=project_key,
        )
        return _ok(workdir=workdir)
    except Exception as exc:
        return tool_error(str(exc))


def _selected_items(client: PlaneClient, args: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    for wid in args.get("work_item_ids") or []:
        items.append(_resolve_item(client, work_item_id=wid))
    for seq in args.get("sequence_ids") or []:
        items.append(_resolve_item(client, sequence_id=seq))
    if not items:
        raise ValueError("work_item_ids or sequence_ids is required")
    return items


def _handle_import_to_kanban(args: dict[str, Any], **kw) -> str:
    try:
        assignee = str(args.get("assignee") or "").strip()
        if not assignee:
            raise ValueError("assignee is required")
        client = get_plane_client()
        project = client.get_project()
        key = _project_key(client, project)
        create_workdir = bool(args.get("create_workdir"))
        priority = int(args.get("priority") or 0)
        workspace_kind = str(args.get("workspace") or "scratch").strip()
        created = []

        from hermes_cli import kanban_db as kb

        kb.init_db()
        conn = kb.connect()
        try:
            for item in _selected_items(client, args):
                seq = _sequence_id(item)
                if seq is None:
                    raise ValueError(f"Plane item {item.get('id')} has no sequence_id")
                title = str(item.get("name") or item.get("title") or f"Plane {key}-{seq}").strip()
                plane_url = _plane_url(client, item, project)
                workdir = None
                task_workspace_kind = workspace_kind
                task_workspace_path = None
                if create_workdir:
                    workdir = prepare_workdir(
                        seq,
                        title,
                        args.get("workdir_base_dir") or DEFAULT_WORKDIR_BASE,
                        project_key=key,
                    )
                    task_workspace_kind = "dir"
                    task_workspace_path = workdir
                elif workspace_kind == "dir" and args.get("workspace_path"):
                    task_workspace_path = str(args["workspace_path"])

                body = "\n".join(
                    [
                        f"Imported from Plane {key}-{seq}: {title}",
                        "",
                        f"plane_workspace_slug: {client.config.workspace_slug}",
                        f"plane_project_id: {client.config.project_id}",
                        f"plane_work_item_id: {item.get('id')}",
                        f"plane_sequence_id: {seq}",
                        f"plane_url: {plane_url}",
                        f"plane_state_id: {_state_id(item, client) or ''}",
                        "",
                        f"State at import: {_state_name(item, client) or 'unknown'}",
                        f"Priority: {item.get('priority') or ''}",
                        "",
                        str(item.get("description_html") or item.get("description_stripped") or ""),
                    ]
                )
                idempotency_key = (
                    f"plane:{client.config.workspace_slug}:"
                    f"{client.config.project_id}:{item.get('id')}"
                )
                # Pre-check the idempotency key so we can surface
                # already_imported=True to the caller. kb.create_task itself
                # is already idempotent and will return the existing task id.
                existing_row = conn.execute(
                    "SELECT id FROM tasks WHERE idempotency_key = ? "
                    "AND status != 'archived' "
                    "ORDER BY created_at DESC LIMIT 1",
                    (idempotency_key,),
                ).fetchone()
                already_imported = existing_row is not None
                task_id = kb.create_task(
                    conn,
                    title=f"[Plane {key}-{seq}] {title}",
                    body=body,
                    assignee=assignee,
                    created_by=os.getenv("HERMES_PROFILE") or "plane_tool",
                    workspace_kind=task_workspace_kind,
                    workspace_path=task_workspace_path,
                    priority=priority,
                    idempotency_key=idempotency_key,
                )
                created.append(
                    {
                        "task_id": task_id,
                        "plane_work_item_id": item.get("id"),
                        "plane_sequence_id": seq,
                        "workdir": workdir,
                        "already_imported": already_imported,
                    }
                )
        finally:
            conn.close()
        return _ok(created_tasks=created)
    except Exception as exc:
        return tool_error(str(exc))


# Function-style wrappers kept for tests and direct module use.
def plane_board_snapshot_tool(include_items_per_state: bool = False, per_state_limit: int = 5, **kwargs) -> str:
    return _handle_board_snapshot({
        "include_items_per_state": include_items_per_state,
        "per_state_limit": per_state_limit,
        **kwargs,
    })


def plane_ping_tool(args: Optional[dict[str, Any]] = None) -> str:
    return _handle_ping(args or {})


def plane_list_work_items_tool(args: dict[str, Any]) -> str:
    return _handle_list_work_items(args)


def plane_get_work_item_tool(args: dict[str, Any]) -> str:
    return _handle_get_work_item(args)


def plane_create_work_item_tool(args: dict[str, Any]) -> str:
    return _handle_create_work_item(args)


def plane_update_work_item_tool(args: dict[str, Any]) -> str:
    return _handle_update_work_item(args)


def plane_add_comment_tool(args: dict[str, Any]) -> str:
    return _handle_add_comment(args)


def plane_sync_progress_tool(args: dict[str, Any]) -> str:
    return _handle_sync_progress(args)


def plane_check_kanban_links_tool(args: dict[str, Any]) -> str:
    return _handle_check_kanban_links(args)


def plane_prepare_workdir_tool(args: dict[str, Any]) -> str:
    if args.get("sequence_id") is None and args.get("readable_id"):
        match = re.search(r"(\d+)$", str(args.get("readable_id")))
        if match:
            args = {**args, "sequence_id": int(match.group(1))}
    out = json.loads(_handle_prepare_workdir(args))
    if "workdir" in out:
        out["task_dir"] = out["workdir"]
    return json.dumps(out, ensure_ascii=False)


def plane_import_to_kanban_tool(args: dict[str, Any]) -> str:
    return _handle_import_to_kanban(args)


PLANE_BOARD_SNAPSHOT_SCHEMA = {
    "name": "plane_board_snapshot",
    "description": "Read a compact snapshot of the configured Plane project. Default output: {project:{id,name,identifier}, states:[{id,name,group,count}], counts_by_state, total_items, items:[compact work items]}. Compact work item: {id, sequence_id, readable_id, name, state_name, state_id, priority, labels, assignees_names, url}. Set verbose=True to also include raw project_payload, states_payload, and items_payload.",
    "parameters": {
        "type": "object",
        "properties": {
            "include_items_per_state": {"type": "boolean", "default": False},
            "per_state_limit": {"type": "integer", "default": 5},
            "limit": {"type": "integer", "default": 20},
            "verbose": {"type": "boolean", "default": False},
        },
    },
}

PLANE_PING_SCHEMA = {
    "name": "plane_ping",
    "description": "Health check the Plane integration: verifies auth, browser-like User-Agent, network, and configured project access.",
    "parameters": {
        "type": "object",
        "properties": {},
    },
}


PLANE_LIST_WORK_ITEMS_SCHEMA = {
    "name": "plane_list_work_items",
    "description": "List Plane work items with optional client-side filters. Default output: {count, items:[{id, sequence_id, readable_id, name, state_name, state_id, priority, labels, assignees_names, url}]}. Set verbose=True to also include raw items_payload.",
    "parameters": {
        "type": "object",
        "properties": {
            "state": {"type": "string"},
            "label": {"type": "string"},
            "assignee": {"type": "string"},
            "priority": {"type": "string"},
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 50},
            "verbose": {"type": "boolean", "default": False},
        },
    },
}

PLANE_GET_WORK_ITEM_SCHEMA = {
    "name": "plane_get_work_item",
    "description": "Get a Plane work item by UUID or project sequence number. Default output: {item:{id, sequence_id, readable_id, name, state_name, state_id, priority, labels, assignees_names, url}}. Set verbose=True to also include raw payload and enriched_item.",
    "parameters": {
        "type": "object",
        "properties": {
            "work_item_id": {"type": "string"},
            "sequence_id": {"type": "integer"},
            "verbose": {"type": "boolean", "default": False},
        },
    },
}

_CREATE_UPDATE_PROPERTIES = {
    "name": {"type": "string"},
    "description_html": {"type": "string"},
    "description_markdown": {"type": "string"},
    "priority": {"type": "string"},
    "state": {"type": "string"},
    "labels": {"type": "array", "items": {"type": "string"}},
    "assignees": {"type": "array", "items": {"type": "string"}},
    "start_date": {"type": "string"},
    "target_date": {"type": "string"},
    "external_source": {"type": "string", "default": "nova-hermes"},
    "external_id": {"type": "string"},
}

PLANE_CREATE_WORK_ITEM_SCHEMA = {
    "name": "plane_create_work_item",
    "description": "Create a Plane work item explicitly requested by the user. Idempotent by external_source + external_id; if external_id is omitted, Hermes generates a stable id from workspace, project, source, and normalized name. No delete operation is exposed.",
    "parameters": {
        "type": "object",
        "properties": _CREATE_UPDATE_PROPERTIES,
        "required": ["name"],
    },
}

PLANE_UPDATE_WORK_ITEM_SCHEMA = {
    "name": "plane_update_work_item",
    "description": "Update selected fields of an existing Plane work item.",
    "parameters": {
        "type": "object",
        "properties": {"work_item_id": {"type": "string"}, **_CREATE_UPDATE_PROPERTIES},
        "required": ["work_item_id"],
    },
}

PLANE_ADD_COMMENT_SCHEMA = {
    "name": "plane_add_comment",
    "description": "Add a comment to a Plane work item by UUID or project sequence number. body_markdown is converted to simple HTML; prefix defaults to true and prepends [Nova].",
    "parameters": {
        "type": "object",
        "properties": {
            "work_item_id": {"type": "string"},
            "sequence_id": {"type": "integer"},
            "body_markdown": {"type": "string"},
            "prefix": {"type": "boolean", "default": True},
        },
        "required": ["body_markdown"],
    },
}

PLANE_SYNC_PROGRESS_SCHEMA = {
    "name": "plane_sync_progress",
    "description": "Reflect progress from an imported Hermes kanban task back to Plane: finds the linked Plane work item from the Hermes card body, posts a progress comment, and optionally updates the Plane state when status is provided. hermes_card_id defaults to the current kanban task when available.",
    "parameters": {
        "type": "object",
        "properties": {
            "hermes_card_id": {"type": "string"},
            "summary": {"type": "string"},
            "status": {"type": "string"},
            "prefix": {"type": "boolean", "default": True},
        },
        "required": ["summary"],
    },
}

PLANE_CHECK_KANBAN_LINKS_SCHEMA = {
    "name": "plane_check_kanban_links",
    "description": "Check linked Hermes kanban tasks against Plane and return only anomalies: cards whose Plane item is Cancelled or missing.",
    "parameters": {
        "type": "object",
        "properties": {
            "hermes_card_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["hermes_card_ids"],
    },
}

PLANE_IMPORT_TO_KANBAN_SCHEMA = {
    "name": "plane_import_to_kanban",
    "description": "Explicitly import one or more Plane work items into Hermes kanban execution tasks, preserving Plane IDs in the body.",
    "parameters": {
        "type": "object",
        "properties": {
            "work_item_ids": {"type": "array", "items": {"type": "string"}},
            "sequence_ids": {"type": "array", "items": {"type": "integer"}},
            "assignee": {"type": "string"},
            "workspace": {"type": "string", "default": "scratch"},
            "workspace_path": {"type": "string"},
            "priority": {"type": "integer"},
            "create_workdir": {"type": "boolean", "default": False},
            "workdir_base_dir": {"type": "string", "default": DEFAULT_WORKDIR_BASE},
        },
        "required": ["assignee"],
    },
}

PLANE_PREPARE_WORKDIR_SCHEMA = {
    "name": "plane_prepare_workdir",
    "description": "Create the canonical local AI Factory work directory for a Plane sequence id. If project_key is omitted, Hermes will resolve it from the configured Plane project when available, falling back to AIFACTORY otherwise.",
    "parameters": {
        "type": "object",
        "properties": {
            "sequence_id": {"type": "integer"},
            "title": {"type": "string"},
            "base_dir": {"type": "string", "default": DEFAULT_WORKDIR_BASE},
            "project_key": {"type": "string"},
        },
        "required": ["sequence_id", "title"],
    },
}

registry.register(
    name="plane_ping",
    toolset="plane",
    schema=PLANE_PING_SCHEMA,
    handler=_handle_ping,
    check_fn=_check_plane_requirements,
    requires_env=["PLANE_API_KEY", "PLANE_WORKSPACE", "PLANE_PROJECT_ID"],
    emoji="🛩️",
)
registry.register(
    name="plane_board_snapshot",
    toolset="plane",
    schema=PLANE_BOARD_SNAPSHOT_SCHEMA,
    handler=_handle_board_snapshot,
    check_fn=_check_plane_requirements,
    requires_env=["PLANE_API_KEY", "PLANE_WORKSPACE", "PLANE_PROJECT_ID"],
    emoji="🛩️",
)
registry.register(
    name="plane_list_work_items",
    toolset="plane",
    schema=PLANE_LIST_WORK_ITEMS_SCHEMA,
    handler=_handle_list_work_items,
    check_fn=_check_plane_requirements,
    requires_env=["PLANE_API_KEY", "PLANE_WORKSPACE", "PLANE_PROJECT_ID"],
    emoji="🛩️",
)
registry.register(
    name="plane_get_work_item",
    toolset="plane",
    schema=PLANE_GET_WORK_ITEM_SCHEMA,
    handler=_handle_get_work_item,
    check_fn=_check_plane_requirements,
    requires_env=["PLANE_API_KEY", "PLANE_WORKSPACE", "PLANE_PROJECT_ID"],
    emoji="🛩️",
)
registry.register(
    name="plane_create_work_item",
    toolset="plane",
    schema=PLANE_CREATE_WORK_ITEM_SCHEMA,
    handler=_handle_create_work_item,
    check_fn=_check_plane_requirements,
    requires_env=["PLANE_API_KEY", "PLANE_WORKSPACE", "PLANE_PROJECT_ID"],
    emoji="🛩️",
)
registry.register(
    name="plane_update_work_item",
    toolset="plane",
    schema=PLANE_UPDATE_WORK_ITEM_SCHEMA,
    handler=_handle_update_work_item,
    check_fn=_check_plane_requirements,
    requires_env=["PLANE_API_KEY", "PLANE_WORKSPACE", "PLANE_PROJECT_ID"],
    emoji="🛩️",
)
registry.register(
    name="plane_add_comment",
    toolset="plane",
    schema=PLANE_ADD_COMMENT_SCHEMA,
    handler=_handle_add_comment,
    check_fn=_check_plane_requirements,
    requires_env=["PLANE_API_KEY", "PLANE_WORKSPACE", "PLANE_PROJECT_ID"],
    emoji="🛩️",
)
registry.register(
    name="plane_sync_progress",
    toolset="plane",
    schema=PLANE_SYNC_PROGRESS_SCHEMA,
    handler=_handle_sync_progress,
    check_fn=_check_plane_requirements,
    requires_env=["PLANE_API_KEY", "PLANE_WORKSPACE", "PLANE_PROJECT_ID"],
    emoji="🛩️",
)
registry.register(
    name="plane_check_kanban_links",
    toolset="plane",
    schema=PLANE_CHECK_KANBAN_LINKS_SCHEMA,
    handler=_handle_check_kanban_links,
    check_fn=_check_plane_requirements,
    requires_env=["PLANE_API_KEY", "PLANE_WORKSPACE", "PLANE_PROJECT_ID"],
    emoji="🛩️",
)
registry.register(
    name="plane_import_to_kanban",
    toolset="plane",
    schema=PLANE_IMPORT_TO_KANBAN_SCHEMA,
    handler=_handle_import_to_kanban,
    check_fn=_check_plane_requirements,
    requires_env=["PLANE_API_KEY", "PLANE_WORKSPACE", "PLANE_PROJECT_ID"],
    emoji="🛩️",
)
registry.register(
    name="plane_prepare_workdir",
    toolset="plane",
    schema=PLANE_PREPARE_WORKDIR_SCHEMA,
    handler=_handle_prepare_workdir,
    check_fn=_check_plane_requirements,
    requires_env=["PLANE_API_KEY", "PLANE_WORKSPACE", "PLANE_PROJECT_ID"],
    emoji="🛩️",
)
