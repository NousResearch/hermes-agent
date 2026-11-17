"""Row-gated #release-updates posting (education design, dependency 5b).

``slack_post_thread`` routes here for every channel in
``slack.post_gated_channels``. Such a channel accepts **no free text**: the
caller passes a Release Assets ``release_asset_row_id`` and this module

1. fetches the row fresh and requires its parent to be the Release Assets data
   source, ``Asset type = Slack update`` and (per the ``post_release_update``
   autonomy level) ``Status = Approved``; ``Published`` / a hold status stop it;
2. follows ``prd_relation_path`` (row → Release → PRD) to the PRD(s), which
   must live in the PRD data source the agent cannot write, and requires
   every Jira key in the PRD's ``JIRA`` property to be ``Done`` (read with
   the teammate's Jira credentials via ``get_secret`` only). Keys on the row
   itself never decide — the agent drafts the row. No PRD, no key, or a
   relation Notion truncates refuses; without sign-off (veto / auto) the PRD's
   ``Release Comms Status`` must be ``In progress``;
3. posts the row's own body, rendered to escaped Slack mrkdwn
   (``tools/notion_blocks_html.render_blocks_mrkdwn``);
4. writes back ``Published URL`` + ``Status=Published`` with a FIXED body
   (``release_asset_gate.update_page_properties``) — the one Notion write the
   5a guard does not see, reachable only from here, after a post landed.

One post per row: an ``O_EXCL`` lock and a row → post ledger under
``$HERMES_HOME/autonomy/`` (a protected path). A failed writeback is recorded
and retried by the next call — never by posting again.

Autonomy (``autonomy.actions.post_release_update``): ``off`` refuses;
``signoff`` needs the row Approved; ``veto`` posts at once if the row is
already Approved, else announces in ``autonomy.announce_channel`` and posts
after the working-hours window unless a non-bot ❌, a hold status, a content
change or an expiry stops it (``agent/autonomy.run_pending``); ``auto`` posts
without the Approved status (every other check still applies) and reports.
Every level but ``off`` requires the Notion write guard to be enforcing —
otherwise the agent could set its own row to Approved.

Config (all optional except the data source)::

    release_updates:
      release_assets_data_source_id: <id>   # else zendesk.release_assets_data_source_id
      asset_type_property: Asset type
      asset_type_value: Slack update
      status_property: Status
      approved_statuses: [Approved]
      pre_approval_statuses: [Draft, In review]   # veto/auto may act from these only
      hold_statuses: [Hold]
      published_status: Published
      published_url_property: Published URL
      prd_relation_path: [Release, PRD]   # row → Release row → PRD (relation names)
      prd_data_sources: [<id>]            # else notion.write_guard.prds.data_sources
      prd_jira_property: JIRA             # the ONLY source of the gating Jira keys
      prd_status_property: Release Comms Status
      prd_active_status: In progress      # required for veto / auto
      jira_done_statuses: [Done]

Driver for due veto actions: ``slack_post_thread(run_due=true)`` or a cron
pre-run script ``python -m tools.release_update_post run-due``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

ACTION = "post_release_update"
_JIRA_KEY = re.compile(r"\b[A-Z][A-Z0-9_]+-\d+\b")
_CHANNEL = re.compile(r"^[A-Z0-9]{6,}$")
_MAX_CHARS = 12000
_MAX_RELATED = 20
_MAX_KEYS = 30

_WRITE_GUARD_MISSING = (
    "Posting a release update is refused: this profile's Notion write guard is not enforcing "
    "(notion.write_guard, plus agent_write_guard). Without it the agent could set a Release Assets row to Approved "
    "itself, so the row is not a human signal. An admin must enable the guard first.")


class _ContentChanged(RuntimeError):
    """The post differs from what the veto announcement showed."""


def _config() -> dict:
    from agent.boundary_config import read_boundary_config
    return read_boundary_config()


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)


def _audit(action: str, target: str) -> None:
    from agent.data_access_audit import record_access
    record_access(tool="slack_post_thread", action=action, target=target)


# ── settings ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Settings:
    data_source: str
    type_prop: str
    type_value: str
    status_prop: str
    approved: frozenset
    pre_approval: frozenset
    hold: frozenset
    published: str
    url_prop: str
    prd_path: tuple
    prd_data_sources: frozenset
    prd_jira_prop: str
    prd_status_prop: str
    prd_active: str
    done: frozenset


def settings(cfg: dict) -> Settings:
    r = cfg.get("release_updates") if isinstance(cfg.get("release_updates"), dict) else {}
    z = cfg.get("zendesk") if isinstance(cfg.get("zendesk"), dict) else {}

    def names(key, default):
        raw = r.get(key, default)
        if isinstance(raw, str):
            raw = [raw]
        return tuple(str(x).strip() for x in (raw or []) if str(x).strip())

    def name(key, default):
        return str(r.get(key) or default).strip()

    published = name("published_status", "Published")
    return Settings(
        data_source=str(r.get("release_assets_data_source_id")
                        or z.get("release_assets_data_source_id") or "").strip(),
        type_prop=name("asset_type_property", "Asset type"),
        type_value=name("asset_type_value", "Slack update"),
        status_prop=name("status_property", "Status"),
        # `Published` can never count as sign-off: it means already posted.
        approved=frozenset(names("approved_statuses", ["Approved"])) - {published},
        # veto / auto may act on a row not yet Approved — but only from these
        # statuses; a row a human rejected or sent back never posts.
        pre_approval=frozenset(names("pre_approval_statuses", ["Draft", "In review"])),
        hold=frozenset(names("hold_statuses", ["Hold"])),
        published=published,
        url_prop=name("published_url_property", "Published URL"),
        prd_path=names("prd_relation_path", ["Release", "PRD"]),
        prd_data_sources=_prd_sources(cfg, r),
        prd_jira_prop=name("prd_jira_property", "JIRA"),
        prd_status_prop=name("prd_status_property", "Release Comms Status"),
        prd_active=name("prd_active_status", "In progress"),
        done=frozenset(names("jira_done_statuses", ["Done"])),
    )


def _hex(value) -> str:
    h = str(value or "").replace("-", "").strip().lower()
    return h if re.fullmatch(r"[0-9a-f]{32}", h) else ""


def _prd_sources(cfg: dict, r: dict) -> frozenset:
    """The PRD data sources: ``release_updates.prd_data_sources``, else the
    write guard's ``prds.data_sources`` (the collection the agent can't edit)."""
    raw = r.get("prd_data_sources")
    if raw is None:
        notion = cfg.get("notion") if isinstance(cfg.get("notion"), dict) else {}
        guard = notion.get("write_guard") if isinstance(notion.get("write_guard"), dict) else {}
        prds = guard.get("prds") if isinstance(guard.get("prds"), dict) else {}
        raw = prds.get("data_sources")
    if isinstance(raw, str):
        raw = [raw]
    return frozenset(h for h in (_hex(x) for x in raw or []) if h)


def _row_key(row_id) -> str:
    hexid = str(row_id or "").replace("-", "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{32}", hexid):
        raise ValueError("release_asset_row_id must be the Notion id of the Release Assets row "
                         "(Slack update) to post.")
    return hexid


# ── Jira ─────────────────────────────────────────────────────────────────

def _keys_in(prop) -> list[str]:
    if not isinstance(prop, dict):
        return []
    kind = prop.get("type")
    val = prop.get(kind)
    texts: list[str] = []
    if kind in ("rich_text", "title"):
        texts = [str(t.get("plain_text") or "") for t in val or [] if isinstance(t, dict)]
    elif kind == "multi_select":
        texts = [str(o.get("name") or "") for o in val or [] if isinstance(o, dict)]
    elif kind == "select" and isinstance(val, dict):
        texts = [str(val.get("name") or "")]
    elif kind == "url" and isinstance(val, str):
        texts = [val]
    elif kind == "formula" and isinstance(val, dict):
        texts = [str(val.get("string") or "")]
    return _JIRA_KEY.findall(" ".join(texts))


def _select(prop) -> str:
    if not isinstance(prop, dict):
        return ""
    val = prop.get(prop.get("type") or "select")
    return str(val.get("name") or "").strip() if isinstance(val, dict) else ""


def _page_title(page: dict) -> str:
    for prop in (page.get("properties") or {}).values():
        if isinstance(prop, dict) and prop.get("type") == "title":
            return "".join(str(t.get("plain_text") or "") for t in prop.get("title") or []
                           if isinstance(t, dict)).strip()
    return ""


async def resolve_prds(page: dict, s: Settings) -> list[dict]:
    """The PRD page(s) the row belongs to, following ``prd_relation_path``
    (default Release Assets → Release → PRD). The Jira keys that gate a post
    come ONLY from these PRDs — a collection the agent cannot write (5a) —
    never from the row, which the agent drafts. Raises GateError."""
    from tools import release_asset_gate as gate
    if not s.prd_data_sources:
        raise gate.GateError("No PRD data source is configured (release_updates.prd_data_sources or "
                             "notion.write_guard.prds.data_sources) — cannot find the release's tickets.")
    frontier = [page]
    fetched = 0
    for hop, name in enumerate(s.prd_path):
        nxt: dict[str, dict] = {}
        for p in frontier:
            rel = (p.get("properties") or {}).get(name)
            if not isinstance(rel, dict) or rel.get("type") != "relation":
                raise gate.GateError(f"{'The row' if hop == 0 else 'A linked page'} has no {name!r} relation.")
            if rel.get("has_more"):
                raise gate.GateError(f"The {name!r} relation has more entries than Notion returned; "
                                     "cannot confirm every linked ticket is Done.")
            for item in rel.get("relation") or []:
                pid = _hex((item or {}).get("id"))
                if pid and pid not in nxt:
                    fetched += 1
                    if fetched > _MAX_RELATED:
                        raise gate.GateError("The row links too many pages to check.")
                    nxt[pid] = await gate.fetch_notion_page(pid)
        frontier = list(nxt.values())
        if not frontier:
            raise gate.GateError(f"No PRD is linked (nothing under {' → '.join(s.prd_path[:hop + 1])}); a "
                                 "release update posts only for a PRD whose tickets are Done.")
    prds = []
    for prd in frontier:
        parent = prd.get("parent") if isinstance(prd.get("parent"), dict) else {}
        if not ({_hex(parent.get("data_source_id")), _hex(parent.get("database_id"))} & s.prd_data_sources):
            raise gate.GateError("A page linked as the PRD is not in the PRD database.")
        if prd.get("in_trash") or prd.get("archived") or prd.get("is_archived"):
            raise gate.GateError("The linked PRD is in the trash.")
        prds.append(prd)
    return prds


def jira_keys_of(prds: list[dict], s: Settings) -> list[str]:
    keys = sorted({k for prd in prds for k in _keys_in((prd.get("properties") or {}).get(s.prd_jira_prop))})
    if len(keys) > _MAX_KEYS:
        from tools import release_asset_gate as gate
        raise gate.GateError(f"The PRD links more than {_MAX_KEYS} Jira tickets.")
    return keys


def _jira_creds() -> tuple[str, str, str]:
    from agent.secret_scope import UnscopedSecretError, get_secret
    try:
        return ((get_secret("JIRA_BASE_URL") or "").strip().rstrip("/"),
                (get_secret("JIRA_EMAIL") or "").strip(),
                (get_secret("JIRA_API_TOKEN") or "").strip())
    except UnscopedSecretError:
        return "", "", ""


async def _jira_status(key: str) -> str:
    """The issue's status name. Raises GateError."""
    from tools import release_asset_gate as gate
    from tools.jira_api_tool import _jira_request
    base, email, token = _jira_creds()
    if not (base and email and token):
        raise gate.GateError("Jira not configured (JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN) — "
                             "cannot confirm the linked tickets are Done.")
    try:
        resp = await _jira_request(base, email, token, f"rest/api/3/issue/{key}?fields=status")
    except Exception as e:
        raise gate.GateError(f"Jira request failed for {key}: {e}") from e
    if resp.get("status") != 200:
        raise gate.GateError(f"Jira returned HTTP {resp.get('status')} for {key}.")
    try:
        data = json.loads(resp.get("text") or "")
        return str(data["fields"]["status"]["name"])
    except (ValueError, KeyError, TypeError):
        raise gate.GateError(f"Jira returned no status for {key}.")


async def check_jira_done(keys: list[str], s: Settings) -> dict:
    from tools import release_asset_gate as gate
    if not keys:
        raise gate.GateError(
            f"The linked PRD has no Jira ticket in {s.prd_jira_prop!r} — a release update posts "
            "only once its tickets are Done.")
    statuses = {k: await _jira_status(k) for k in keys}
    open_ = {k: v for k, v in statuses.items() if v not in s.done}
    if open_:
        raise gate.GateError("Not every linked Jira ticket is Done: "
                             + ", ".join(f"{k} is {v!r}" for k, v in open_.items()) + ".")
    return statuses


# ── the row → the post ───────────────────────────────────────────────────

def _check_row(row, s: Settings, *, require_approved: bool) -> None:
    from tools import release_asset_gate as gate
    if row.asset_type != s.type_value:
        raise gate.GateError(f"The row's {s.type_prop} is {row.asset_type!r}, not {s.type_value!r}.")
    if row.status == s.published:
        raise gate.RowStopped("The row is already Published — a release update posts once.")
    if row.status in s.hold:
        raise gate.RowStopped(f"The row is on {row.status!r} — a product-team member held it.")
    if require_approved and row.status not in s.approved:
        raise gate.GateError(
            f"Sign-off required: the row's {s.status_prop} is {row.status or 'empty'!r}; it must be "
            f"{' / '.join(sorted(s.approved))} (set by a product-team member).")
    if row.status not in s.approved and row.status not in s.pre_approval:
        raise gate.RowStopped(
            f"The row's {s.status_prop} is {row.status or 'empty'!r} — not "
            f"{' / '.join(sorted(s.approved | s.pre_approval))}, so it is not posted.")


async def materialize(spec: dict, cfg: dict, *, require_approved: bool) -> dict:
    """Read the row fresh, check it and Jira, render the post. Raises GateError."""
    from tools import release_asset_gate as gate
    from tools.notion_blocks_html import render_blocks_mrkdwn
    s = settings(cfg)
    page = await gate.fetch_notion_page(spec["row_id"])
    row = gate.row_state(page, s.data_source, status_property=s.status_prop,
                         asset_type_property=s.type_prop,
                         config_key="release_updates.release_assets_data_source_id")
    _check_row(row, s, require_approved=require_approved)
    rendered = render_blocks_mrkdwn(await gate.fetch_block_tree(spec["row_id"]))
    if rendered.unsupported:
        raise gate.GateError(
            "The row's content has blocks a Slack post can't carry: "
            f"{', '.join(sorted(set(rendered.unsupported)))}. Ask the team to adjust the row.")
    if rendered.images and not spec.get("omit_images"):
        raise gate.GateError(
            f"The row has {len(rendered.images)} image(s) ({'; '.join(rendered.images)}); "
            "posting images isn't supported. Call again with omit_images=true to post the text "
            "only and report the captions, or ask the team to remove them.")
    text = rendered.text.strip()
    if not text:
        raise gate.GateError("The row has no body to post.")
    if len(text) > _MAX_CHARS:
        raise gate.GateError(f"The row's body is longer than {_MAX_CHARS} characters.")
    prds = await resolve_prds(page, s)
    if not require_approved:
        for prd in prds:
            state = _select((prd.get("properties") or {}).get(s.prd_status_prop))
            if state != s.prd_active:
                raise gate.GateError(
                    f"The PRD's {s.prd_status_prop} is {state or 'empty'!r}, not {s.prd_active!r}; "
                    "without sign-off, a release update posts only for an active release.")
    jira = await check_jira_done(jira_keys_of(prds, s), s)
    prd_info = [{"id": _hex(p.get("id")), "url": str(p.get("url") or ""), "title": _page_title(p)}
                for p in prds]
    props = page.get("properties") if isinstance(page.get("properties"), dict) else {}
    status_type = (props.get(s.status_prop) or {}).get("type") or "select"
    if status_type not in ("select", "status"):
        raise gate.GateError(f"The row's {s.status_prop} property is a {status_type}, not a select/status.")
    # The veto announcement shows the text, the PRD(s) and the ticket set;
    # execution re-derives all three and refuses (stale) if any changed.
    prd_ids = ",".join(sorted(p["id"] for p in prd_info))
    sha = hashlib.sha256("\0".join([spec["channel"], text, prd_ids, ",".join(sorted(jira))])
                         .encode("utf-8")).hexdigest()
    return {"row": row, "title": row.title or "release update", "text": text, "sha": sha,
            "jira": jira, "prds": prd_info, "status_type": status_type,
            "images_omitted": list(rendered.images)}


def _writeback_props(s: Settings, status_type: str, url: str) -> dict:
    return {s.url_prop: {"url": url}, s.status_prop: {status_type: {"name": s.published}}}


async def _writeback(key: str, entry: dict, cfg: dict) -> None:
    from agent import autonomy
    from tools import release_asset_gate as gate
    s = settings(cfg)
    try:
        await gate.update_page_properties(key, _writeback_props(s, entry["status_type"], entry["permalink"]))
    except gate.GateError as e:
        entry["published_written"] = False
        entry["writeback_error"] = str(e)
    else:
        entry["published_written"] = True
        entry.pop("writeback_error", None)
    autonomy.ledger_put(ACTION, key, entry)


def _gated_channel_denial(channel: str, cfg: dict) -> Optional[str]:
    from agent.slack_post_policy import channel_denial, is_gated
    if not is_gated(channel, cfg):
        return (f"{channel} is not a gated channel (slack.post_gated_channels); "
                "release_asset_row_id is only for gated channels.")
    return channel_denial(channel, cfg, gated_ok=True)


async def post_once(spec: dict, cfg: dict, level: str, *, require_approved: bool,
                    expected_sha: Optional[str] = None, pending_id: str = "") -> dict:
    """Post the row unless already posted. Raises GateError, _ContentChanged,
    ValueError or autonomy.AutonomyBusy."""
    from agent import autonomy
    from tools import release_asset_gate as gate
    key = spec["row_id"]
    with autonomy.exclusive(ACTION, key):
        existing = autonomy.ledger_get(ACTION, key)
        if existing and existing.get("state") == "in_flight":
            raise gate.GateError(
                f"A post of this row to {existing.get('channel')} started at {existing.get('started_at')} "
                "and its outcome is unknown (the call failed mid-way). Check the channel in Slack: an "
                "operator then removes autonomy/ledger/post_release_update/"
                f"{key}.json (if nothing was posted) — the tool never posts it a second time on its own.")
        if existing:
            if not existing.get("published_written"):
                await _writeback(key, existing, cfg)
            return {"ok": True, "already_posted": True, **existing,
                    "note": "This row was already posted; not posting again."}
        denial = _gated_channel_denial(spec["channel"], cfg)
        if denial:
            raise ValueError(denial)
        m = await materialize(spec, cfg, require_approved=require_approved)
        if expected_sha and m["sha"] != expected_sha:
            raise _ContentChanged("The row's content changed after the veto announcement.")
        if pending_id:
            autonomy.update_pending(ACTION, pending_id, status="executing")
        # In flight BEFORE the call: if anything fails after it starts (a
        # timeout after Slack accepted the post, a crash), the row stays
        # blocked until an operator checks Slack — never posted twice.
        autonomy.ledger_put(ACTION, key, {"state": "in_flight", "channel": spec["channel"],
                                          "started_at": _now().isoformat(), "level": level,
                                          "pending_id": pending_id or None})
        try:
            ts = await gate.post_announcement(spec["channel"], m["text"])
        except gate.SlackRejected:
            autonomy.ledger_delete(ACTION, key)  # Slack refused it: nothing was posted
            raise
        # Record FIRST: the post exists now, whatever happens next.
        entry = {"state": "posted", "channel": spec["channel"], "message_ts": ts,
                 "permalink": await gate.permalink(spec["channel"], ts),
                 "title": m["title"], "posted_at": _now().isoformat(), "level": level,
                 "pending_id": pending_id or None, "jira": m["jira"], "prds": m["prds"],
                 "status_type": m["status_type"], "images_omitted": m["images_omitted"],
                 "published_written": False}
        autonomy.ledger_put(ACTION, key, entry)
        await _writeback(key, entry, cfg)
    _audit(f"autonomy_{level}", f"channel={spec['channel']} ts={ts} row={key}"
           + (f" pending={pending_id}" if pending_id else ""))
    return {"ok": True, **entry, "row_url": m["row"].url}


# ── Slack text for announcements (every value escaped, links to our hosts) ──

def _esc(text) -> str:
    return str(text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _link(url, label) -> str:
    try:
        parts = urlsplit(str(url or ""))
    except ValueError:
        return _esc(label)
    host = parts.hostname or ""
    if (parts.scheme == "https" and (host in ("www.notion.so", "notion.so") or host.endswith("slack.com"))
            and not any(c in str(url) for c in "<>| \t\r\n")):
        return f"<{url}|{_esc(label)}>"
    return _esc(label)


def _chan(channel: str) -> str:
    return f"<#{channel}>" if _CHANNEL.match(channel or "") else _esc(channel)


def _quoted(text: str) -> str:
    """The FULL text that will be posted, quoted — never a truncated preview
    of it (the veto is a decision on exactly this text)."""
    return "\n".join("> " + ln for ln in text.split("\n"))


async def _say(channel: str, text: str, thread_ts: str = "") -> str:
    from tools import release_asset_gate as gate
    try:
        await gate.post_announcement(channel, text, thread_ts)
        return ""
    except gate.GateError as e:
        return str(e)


async def queue_veto(spec: dict, cfg: dict, m: dict) -> dict:
    from zoneinfo import ZoneInfo
    from agent import autonomy
    from tools import release_asset_gate as gate
    now = _now()
    try:
        plan = autonomy.plan_veto(cfg, now)
    except autonomy.AutonomyConfigError as e:
        return {"error": str(e)}
    pending_id = autonomy.new_pending_id()
    text = (f":hourglass_flowing_sand: I plan to post *{_esc(m['title'])}* to {_chan(spec['channel'])} "
            f"from {_link(m['row'].url, 'this Release Assets row')}, for PRD "
            f"{', '.join(_link(p['url'], p['title'] or p['id']) for p in m['prds'])} "
            f"(Jira {', '.join(_esc(k) for k in sorted(m['jira']))} all Done). It goes ahead after "
            f"{plan.not_before.astimezone(ZoneInfo(plan.timezone)):%a %d %b %H:%M} ({_esc(plan.timezone)}) "
            f"unless someone reacts :x: here or sets the row to Hold. "
            f"(autonomy: veto · ref {pending_id}). It will post exactly:\n{_quoted(m['text'])}")
    try:
        ts = await gate.post_announcement(plan.channel, text)
    except gate.GateError as e:
        return {"error": f"Could not announce the veto window, so nothing was queued: {e}"}
    autonomy.save_pending(ACTION, pending_id, {
        "spec": spec, "content_sha": m["sha"], "title": m["title"],
        "created_at": now.isoformat(), "not_before": plan.not_before.isoformat(),
        "expires_at": plan.expires_at.isoformat(), "announce_channel": plan.channel,
        "announcement_ts": ts,
    })
    _audit("autonomy_veto_announced", f"pending={pending_id} row={spec['row_id']}")
    return {"status": "pending", "pending_id": pending_id, "not_before": plan.not_before.isoformat(),
            "expires_at": plan.expires_at.isoformat(), "announcement_ts": ts,
            "message": ("Announced. It posts when slack_post_thread is called with run_due=true "
                        "(release-watch does this) or with this pending_id after not_before, unless "
                        "vetoed (❌ or Hold), a ticket reopens or the row changes. It expires at "
                        "expires_at.")}


async def execute_pending(pending_id: str, level: str, cfg: dict) -> dict:
    from agent import autonomy
    from tools import release_asset_gate as gate

    async def execute(signed: dict) -> dict:
        try:
            return await post_once(
                signed["spec"], cfg, level, require_approved=(level == "signoff"),
                expected_sha=None if level == "signoff" else signed["content_sha"],
                pending_id=pending_id)
        except gate.RowStopped as e:
            raise autonomy.PendingStop("vetoed", str(e).rstrip("."))
        except _ContentChanged as e:
            raise autonomy.PendingStop(
                "stale", f"{str(e).rstrip('.')} — re-queue it so the team sees the new version")
        except (gate.GateError, ValueError) as e:
            raise autonomy.PendingRetry(str(e))

    return await autonomy.run_pending(
        ACTION, pending_id, level, now=_now(), veto_reactions=gate.veto_reactions,
        say=_say, audit=_audit, execute=execute,
        done_fields=lambda r: {"message_ts": r.get("message_ts")},
        done_text=lambda r: ":white_check_mark: Posted: " + _link(r.get("permalink"), r.get("title") or "the update"),
        verb="posting it", outcome="posted")


def _level(cfg: dict) -> tuple[Optional[str], Optional[str]]:
    """(level, error) after the level / write-guard preconditions."""
    from agent import autonomy
    from tools.notion_write_guard import autonomy_precondition
    try:
        level = autonomy.resolve_level(ACTION, cfg)
    except autonomy.AutonomyConfigError as e:
        return None, str(e)
    if level == "off":
        return None, "Posting release updates is turned off for this profile (autonomy level 'off')."
    why = autonomy_precondition(cfg)
    if why:
        return None, f"{_WRITE_GUARD_MISSING} ({why})"
    return level, None


async def run_due_pending(cfg: Optional[dict] = None) -> list[dict]:
    """Execute every due veto post (or expire it). For release-watch, via
    ``slack_post_thread(run_due=true)`` or ``python -m tools.release_update_post run-due``."""
    from agent import autonomy
    cfg = _config() if cfg is None else cfg
    level, err = _level(cfg)
    if err:
        return [{"error": err}]
    return [{"pending_id": pid, **(await execute_pending(pid, level, cfg))}
            for pid in autonomy.due_pending_ids(ACTION, _now())]


def list_waiting() -> list[dict]:
    from agent import autonomy
    return [{"pending_id": r.get("pending_id"),
             "title": (r.get("signed") or {}).get("title"),
             "channel": ((r.get("signed") or {}).get("spec") or {}).get("channel"),
             "not_before": (r.get("signed") or {}).get("not_before"),
             "expires_at": (r.get("signed") or {}).get("expires_at")}
            for r in autonomy.list_pending(ACTION) if r.get("status") == "pending"]


def _earlier_veto(row_id: str) -> Optional[dict]:
    """A row already waiting returns that record; a row someone vetoed is not
    re-queued (re-announcing until nobody objects would wear vetoes down) —
    after a veto it posts only once a human sets it Approved."""
    from agent import autonomy
    for rec in autonomy.list_pending(ACTION):
        signed = rec.get("signed") if isinstance(rec.get("signed"), dict) else {}
        if (signed.get("spec") or {}).get("row_id") != row_id or not autonomy.verify_pending(rec):
            continue
        if rec.get("status") in ("pending", "executing"):
            return {"status": "pending", "pending_id": rec.get("pending_id"),
                    "not_before": signed.get("not_before"), "expires_at": signed.get("expires_at"),
                    "message": "Already announced and waiting out its veto window."}
        if rec.get("status") == "vetoed":
            return {"error": "This row was vetoed (" + str(rec.get("reason") or "") + "). It is not "
                             "re-announced; it posts once a product-team member sets it Approved."}
    return None


async def post_release_update(args: dict, cfg: dict) -> dict:
    """The gated path of ``slack_post_thread``. Returns a result dict, or
    ``{"error": ...}``."""
    from agent import autonomy
    from agent.slack_post_policy import normalise
    from tools import release_asset_gate as gate
    channel = normalise(args.get("chat_id"))
    if str(args.get("message") or "").strip():
        return {"error": f"{channel} accepts no free text: the post is the Release Assets row's own "
                         "body. Pass release_asset_row_id only."}
    if str(args.get("thread_ts") or "").strip():
        return {"error": "Release updates are top-level posts; thread_ts is not accepted for a gated channel."}
    level, err = _level(cfg)
    if err:
        return {"error": err}
    try:
        spec = {"row_id": _row_key(args.get("release_asset_row_id")), "channel": channel,
                "omit_images": args.get("omit_images") is True}
        denial = _gated_channel_denial(channel, cfg)
        if denial:
            return {"error": denial}
        if autonomy.ledger_get(ACTION, spec["row_id"]) is None and level == "veto":
            m = await materialize(spec, cfg, require_approved=False)
            if m["row"].status not in settings(cfg).approved:
                earlier = _earlier_veto(spec["row_id"])
                if earlier is not None:
                    return earlier
                return await queue_veto(spec, cfg, m)
        result = await post_once(spec, cfg, level, require_approved=(level in ("signoff", "veto")))
    except (ValueError, gate.GateError, autonomy.AutonomyBusy, autonomy.AutonomyConfigError) as e:
        return {"error": str(e)}
    if level == "auto" and not result.get("already_posted"):
        channel_home, _w, _wh = autonomy.veto_settings(cfg)
        if channel_home:
            err = await _say(channel_home, f":mega: Posted *{_esc(result.get('title'))}* to "
                             f"{_chan(channel)}: {_link(result.get('permalink'), 'view')} (autonomy: auto)")
            if err:
                result["announce_error"] = err
        else:
            result["announce_error"] = "autonomy.announce_channel is not configured"
    return result


async def handle_autonomy_ops(args: dict, cfg: dict) -> Optional[dict]:
    """``list_pending`` / ``run_due`` / ``pending_id`` — None when none given."""
    if args.get("list_pending") is True:
        return {"ok": True, "pending": list_waiting()}
    if args.get("run_due") is not True and not str(args.get("pending_id") or "").strip():
        return None
    level, err = _level(cfg)
    if err:
        return {"error": err}
    if args.get("run_due") is True:
        return {"ok": True, "results": await run_due_pending(cfg)}
    return await execute_pending(str(args.get("pending_id")).strip(), level, cfg)


if __name__ == "__main__":  # cron pre-run script: python -m tools.release_update_post run-due
    import asyncio
    import sys
    if sys.argv[1:] != ["run-due"]:
        sys.exit("usage: python -m tools.release_update_post run-due")
    print(json.dumps(asyncio.run(run_due_pending()), ensure_ascii=False, default=str))
