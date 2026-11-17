"""Row-gated #release-updates posting (education design, dependency 5b).

``slack_post_thread`` routes here for every channel in
``slack.post_gated_channels``. Such a channel accepts **no free text**: the
caller passes a Release Assets ``release_asset_row_id`` and this module

1. fetches the row fresh and requires its parent to be the Release Assets data
   source, ``Asset type = Slack update`` and (per the ``post_release_update``
   autonomy level) ``Status = Approved``; ``Published`` / a hold status stop it;
2. collects every Jira key linked from the row (and the pages its configured
   relation properties point at, two hops — e.g. row → Release → PRD) and
   requires each to be ``Done``, read with the teammate's Jira credentials
   via ``get_secret`` only; no key found, or a relation list Notion truncates,
   refuses;
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
      hold_statuses: [Hold]
      published_status: Published
      published_url_property: Published URL
      jira_key_properties: [JIRA]
      jira_relation_properties: []        # e.g. [Release, PRD]
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
    "(notion.write_guard). Without it the agent could set a Release Assets row to Approved "
    "itself, so the row is not a human signal. An admin must enable the guard first.")


class _ContentChanged(RuntimeError):
    """The post differs from what the veto announcement showed."""


def _config() -> dict:
    from hermes_cli.config import read_raw_config
    return read_raw_config()


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
    hold: frozenset
    published: str
    url_prop: str
    jira_props: tuple
    relation_props: tuple
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
        hold=frozenset(names("hold_statuses", ["Hold"])),
        published=published,
        url_prop=name("published_url_property", "Published URL"),
        jira_props=names("jira_key_properties", ["JIRA"]),
        relation_props=names("jira_relation_properties", []),
        done=frozenset(names("jira_done_statuses", ["Done"])),
    )


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


async def collect_jira_keys(page: dict, s: Settings) -> list[str]:
    """Jira keys on the row and on pages reached through ``relation_props``
    (two hops). Raises GateError on a truncated relation or too many pages."""
    from tools import release_asset_gate as gate
    keys: list[str] = []
    seen_pages: set[str] = set()
    frontier = [page]
    for hop in range(3):
        nxt: list[dict] = []
        for p in frontier:
            props = p.get("properties") if isinstance(p.get("properties"), dict) else {}
            for name in s.jira_props:
                for k in _keys_in(props.get(name)):
                    if k not in keys:
                        keys.append(k)
            if hop == 2:
                continue
            for name in s.relation_props:
                rel = props.get(name)
                if not isinstance(rel, dict) or rel.get("type") != "relation":
                    continue
                if rel.get("has_more"):
                    raise gate.GateError(
                        f"The row's {name!r} relation has more entries than Notion returned; "
                        "cannot confirm every linked Jira ticket is Done.")
                for item in rel.get("relation") or []:
                    pid = str((item or {}).get("id") or "").replace("-", "").lower()
                    if not pid or pid in seen_pages:
                        continue
                    seen_pages.add(pid)
                    if len(seen_pages) > _MAX_RELATED:
                        raise gate.GateError("The row links too many related pages to check.")
                    nxt.append(await gate.fetch_notion_page(pid))
        frontier = nxt
        if not frontier:
            break
    if len(keys) > _MAX_KEYS:
        raise gate.GateError(f"The row links more than {_MAX_KEYS} Jira tickets.")
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
            "No linked Jira ticket found on the row"
            + (f" (properties {', '.join(s.jira_props)}"
               + (f", via {', '.join(s.relation_props)}" if s.relation_props else "") + ")")
            + " — a release update posts only once its tickets are Done.")
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
    jira = await check_jira_done(await collect_jira_keys(page, s), s)
    props = page.get("properties") if isinstance(page.get("properties"), dict) else {}
    status_type = (props.get(s.status_prop) or {}).get("type") or "select"
    if status_type not in ("select", "status"):
        raise gate.GateError(f"The row's {s.status_prop} property is a {status_type}, not a select/status.")
    sha = hashlib.sha256(f"{spec['channel']}\0{text}".encode("utf-8")).hexdigest()
    return {"row": row, "title": row.title or "release update", "text": text, "sha": sha,
            "jira": jira, "status_type": status_type, "images_omitted": list(rendered.images)}


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
        ts = await gate.post_announcement(spec["channel"], m["text"])
        # Record FIRST: the post exists now, whatever happens next.
        entry = {"channel": spec["channel"], "message_ts": ts,
                 "permalink": await gate.permalink(spec["channel"], ts),
                 "title": m["title"], "posted_at": _now().isoformat(), "level": level,
                 "pending_id": pending_id or None, "jira": m["jira"],
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


def _preview(text: str) -> str:
    cut = text if len(text) <= 600 else text[:600] + "…"
    return "\n".join("> " + ln for ln in cut.split("\n"))


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
            f"from {_link(m['row'].url, 'this Release Assets row')} "
            f"(Jira {', '.join(_esc(k) for k in m['jira'])} Done). It goes ahead after "
            f"{plan.not_before.astimezone(ZoneInfo(plan.timezone)):%a %d %b %H:%M} ({_esc(plan.timezone)}) "
            f"unless someone reacts :x: here or sets the row to Hold. "
            f"(autonomy: veto · ref {pending_id})\n{_preview(m['text'])}")
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
    from tools.notion_write_guard import is_enforcing
    try:
        level = autonomy.resolve_level(ACTION, cfg)
    except autonomy.AutonomyConfigError as e:
        return None, str(e)
    if level == "off":
        return None, "Posting release updates is turned off for this profile (autonomy level 'off')."
    if not is_enforcing(cfg):
        return None, _WRITE_GUARD_MISSING
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
