"""Notion write guard — the profile-level boundary on ``notion_api`` writes.

Education teammate design, dependency 5a. ``notion_api`` takes a raw
``v1/...`` path and body, so on its own it passes any ``PATCH``/``DELETE``
through: the agent could approve its own drafts. A profile that sets
``notion.write_guard`` gets every non-read request checked here, both inside
``notion_api`` and again in the ``model_tools`` execution backstop.

The guard resolves the ACTUAL target of each write — ``pages/<id>``,
``pages/<id>/markdown``, ``blocks/<id>[/children]`` (block → parent block →
page, cached), ``comments``, ``POST pages`` with a parent — up to the row and
the data source it lives in, then applies the rules for that collection:

- **Everywhere:** no ``DELETE``; no ``archived`` / ``in_trash`` /
  ``is_archived`` key; no database / data-source / property schema changes;
  an endpoint not in the table below is refused.
- **Release Kits rows** (Releases + Release Assets): ``Status`` may only be
  ``Draft``, given explicitly on create; a row not currently ``Draft`` is
  read-only (properties AND content). Advancing past Draft is the
  ``advance_asset_status`` autonomy action — not built, so ``off`` only.
- **PRDs** (Detailed Bucket Features): read-only except
  ``Release Comms Status`` → ``In progress`` / ``Done``.
- **Ideas:** create at ``Proposed`` (explicitly); only its OWN rows
  (``created_by`` = this integration's bot user) are writable, moving along
  ``own_transitions`` (Accepted → In progress → Done); never sets a human
  status (Accepted / Rejected / Parked — the ``ideas_accept_own`` action, not
  built) or a human property (Decision note).
- **writable_pages:** its home page and descendants are free (still no trash).
- Anything else that resolves → refused; anything that doesn't resolve →
  refused.

Database ids and property names come from config — the rules are data::

    notion:
      write_guard:
        enabled: true
        release_kits: {data_sources: [<Releases>, <Release Assets>]}
        prds:  {data_sources: [<Detailed Bucket Features>]}
        ideas: {data_sources: [<Ideas>]}
        writable_pages: [<home page>]

**Contract with ``zendesk_create_draft`` / ``slack_post_thread``:** they
refuse every autonomy level but ``off`` unless ``is_enforcing(cfg)``. So
``enabled: true`` (or any block not explicitly disabled) must mean the guard
IS enforcing: an invalid or incomplete block keeps it ON and denies every
write with the validation error — never silently off.

The one write this guard does not see is the 5b writeback (``Published URL``
+ ``Status=Published`` on a Slack-update row), which
``tools/release_update_post.py`` sends itself after posting, never through
``notion_api``.

Not a tool module (registers nothing).
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional
from urllib.parse import unquote

logger = logging.getLogger(__name__)

_HEX = re.compile(r"^[0-9a-f]{32}$")
_SEGMENT = re.compile(r"^[A-Za-z0-9_-]+$")
_TRASH_KEYS = frozenset({"archived", "in_trash", "is_archived"})
_PAGE_UPDATE_KEYS = frozenset({"properties", "icon", "cover"})
_PAGE_CREATE_KEYS = frozenset({"parent", "properties", "children", "markdown", "icon", "cover"})
_MAX_WALK = 12
_BLOCK_TTL = 3600.0
_SCHEMA_TTL = 600.0
_GUARD_KEYS = frozenset({"enabled", "release_kits", "prds", "ideas", "writable_pages"})

# Autonomy actions whose rules this guard relaxes once built. Both are built
# for `off` only in v1 (agent/autonomy.py BUILT_LEVELS).
GUARDED_ACTIONS = ("advance_asset_status", "ideas_accept_own")


class GuardConfigError(ValueError):
    """``notion.write_guard`` is enabled but cannot be honoured."""


class GuardLookupError(RuntimeError):
    """A Notion lookup needed to resolve a write's target failed."""


class _Deny(Exception):
    pass


def _hex(value) -> str:
    h = str(value or "").replace("-", "").strip().lower()
    return h if _HEX.match(h) else ""


# ── config ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GuardConfig:
    error: str = ""
    kits: frozenset = frozenset()
    kit_status: str = "Status"
    kit_draft: str = "Draft"
    prds: frozenset = frozenset()
    prd_property: str = "Release Comms Status"
    prd_values: frozenset = frozenset({"In progress", "Done"})
    ideas: frozenset = frozenset()
    idea_status: str = "Status"
    idea_create: str = "Proposed"
    idea_transitions: dict = field(default_factory=dict)
    idea_human_statuses: frozenset = frozenset({"Accepted", "Rejected", "Parked"})
    idea_human_props: frozenset = frozenset({"Decision note"})
    writable: frozenset = frozenset()


def _raw_block(cfg) -> tuple[bool, object]:
    notion = cfg.get("notion") if isinstance(cfg, dict) else None
    if not isinstance(notion, dict) or "write_guard" not in notion:
        return False, None
    return True, notion.get("write_guard")


def is_enforcing(cfg) -> bool:
    """True iff this profile's Notion writes are guarded.

    Any ``notion.write_guard`` value other than absent / ``null`` / ``false``
    / a mapping with ``enabled: false`` counts — a malformed block enforces
    (and denies every write), it never switches the guard off.
    """
    present, raw = _raw_block(cfg)
    if not present or raw is None or raw is False:
        return False
    if isinstance(raw, dict):
        return raw.get("enabled") is not False
    return True


def _ids(section: str, raw) -> frozenset:
    if not isinstance(raw, (list, tuple)) or not raw:
        raise GuardConfigError(f"{section} must be a non-empty list of Notion ids")
    out = set()
    for item in raw:
        h = _hex(item)
        if not h:
            raise GuardConfigError(f"{section}: {item!r} is not a Notion id")
        out.add(h)
    return frozenset(out)


def _name(section: str, raw, default: str) -> str:
    if raw is None:
        return default
    if not isinstance(raw, str) or not raw.strip():
        raise GuardConfigError(f"{section} must be a non-empty string")
    return raw.strip()


def _names(section: str, raw, default: frozenset) -> frozenset:
    if raw is None:
        return default
    if not isinstance(raw, (list, tuple)) or not all(isinstance(x, str) and x.strip() for x in raw):
        raise GuardConfigError(f"{section} must be a list of names")
    return frozenset(x.strip() for x in raw)


def _section(guard: dict, key: str) -> dict:
    sec = guard.get(key)
    if not isinstance(sec, dict):
        raise GuardConfigError(f"notion.write_guard.{key} is required")
    return sec


def _parse(cfg: dict) -> GuardConfig:
    _present, raw = _raw_block(cfg)
    if not isinstance(raw, dict):
        raise GuardConfigError("notion.write_guard must be a mapping with enabled: true")
    if raw.get("enabled") is not True:
        raise GuardConfigError("notion.write_guard.enabled must be true or false")
    unknown = set(raw) - _GUARD_KEYS
    if unknown:
        raise GuardConfigError(f"notion.write_guard has unknown keys: {', '.join(sorted(map(str, unknown)))}")
    kits = _section(raw, "release_kits")
    prds = _section(raw, "prds")
    ideas = _section(raw, "ideas")
    p = "notion.write_guard"
    transitions_raw = ideas.get("own_transitions")
    if transitions_raw is None:
        transitions = {"Accepted": frozenset({"In progress"}), "In progress": frozenset({"Done"})}
    else:
        if not isinstance(transitions_raw, dict):
            raise GuardConfigError(f"{p}.ideas.own_transitions must map a status to a list of statuses")
        transitions = {str(k).strip(): _names(f"{p}.ideas.own_transitions.{k}", v, frozenset())
                       for k, v in transitions_raw.items()}
    writable_raw = raw.get("writable_pages")
    gc = GuardConfig(
        kits=_ids(f"{p}.release_kits.data_sources", kits.get("data_sources")),
        kit_status=_name(f"{p}.release_kits.status_property", kits.get("status_property"), "Status"),
        kit_draft=_name(f"{p}.release_kits.draft_status", kits.get("draft_status"), "Draft"),
        prds=_ids(f"{p}.prds.data_sources", prds.get("data_sources")),
        prd_property=_name(f"{p}.prds.property", prds.get("property"), "Release Comms Status"),
        prd_values=_names(f"{p}.prds.allowed_values", prds.get("allowed_values"),
                          frozenset({"In progress", "Done"})),
        ideas=_ids(f"{p}.ideas.data_sources", ideas.get("data_sources")),
        idea_status=_name(f"{p}.ideas.status_property", ideas.get("status_property"), "Status"),
        idea_create=_name(f"{p}.ideas.create_status", ideas.get("create_status"), "Proposed"),
        idea_transitions=transitions,
        idea_human_statuses=_names(f"{p}.ideas.human_statuses", ideas.get("human_statuses"),
                                   frozenset({"Accepted", "Rejected", "Parked"})),
        idea_human_props=_names(f"{p}.ideas.human_properties", ideas.get("human_properties"),
                                frozenset({"Decision note"})),
        writable=frozenset() if writable_raw in (None, []) else _ids(f"{p}.writable_pages", writable_raw),
    )
    if (gc.kits & gc.prds) or (gc.kits & gc.ideas) or (gc.prds & gc.ideas):
        raise GuardConfigError(f"{p}: release_kits, prds and ideas must name different data sources")
    if gc.idea_create in gc.idea_human_statuses:
        raise GuardConfigError(f"{p}.ideas.create_status cannot be a human-only status")
    for targets in transitions.values():
        if targets & gc.idea_human_statuses:
            raise GuardConfigError(f"{p}.ideas.own_transitions cannot move a row into a human-only status")
    if not _agent_write_guard_on(cfg):
        raise GuardConfigError(
            "notion.write_guard needs agent_write_guard enabled in the same config.yaml — "
            "otherwise the agent could edit its own boundary config with the file tools")
    from agent import autonomy
    for action in GUARDED_ACTIONS:
        try:
            level = autonomy.resolve_level(action, cfg)
        except autonomy.AutonomyConfigError as e:
            raise GuardConfigError(str(e))
        if level != "off":  # unreachable while BUILT_LEVELS has only `off`
            raise GuardConfigError(f"autonomy.actions.{action}: {level!r} is not built")
    return gc


def _agent_write_guard_on(cfg) -> bool:
    """The self-modification guard (4e) protects this config.yaml; the write
    guard is only trustworthy when the agent can't edit its own rules."""
    raw = cfg.get("agent_write_guard") if isinstance(cfg, dict) else None
    if raw is True:
        return True
    return isinstance(raw, dict) and raw.get("enabled", True) is True


_LOGGED: set[str] = set()


def load_guard(cfg) -> Optional[GuardConfig]:
    """None when the profile is unguarded; a ``GuardConfig`` otherwise, with
    ``error`` set when the block is enabled but invalid (deny every write)."""
    if not is_enforcing(cfg):
        return None
    from agent.boundary_config import is_unreadable, unreadable_message
    if is_unreadable(cfg):
        return GuardConfig(error=unreadable_message(cfg))
    try:
        return _parse(cfg)
    except GuardConfigError as e:
        msg = str(e)
        if msg not in _LOGGED:
            _LOGGED.add(msg)
            logger.error("[notion_write_guard] %s — every Notion write is denied until fixed", msg)
        return GuardConfig(error=msg)


def autonomy_precondition(cfg) -> Optional[str]:
    """Why a sign-off-gated action (zendesk_create_draft, post_release_update)
    must refuse every level but ``off``, or None. The row's Status is a human
    signal only while the agent can neither set it (this guard enforcing) nor
    rewrite its own rules (``agent_write_guard``)."""
    from agent.boundary_config import is_unreadable, unreadable_message
    if is_unreadable(cfg):
        return unreadable_message(cfg)
    if not is_enforcing(cfg):
        return "the Notion write guard is not enforcing (notion.write_guard.enabled: true)"
    if not _agent_write_guard_on(cfg):
        return "agent_write_guard is off, so the agent could edit its own boundary config"
    return None


def validate(cfg) -> list[str]:
    """Config errors for startup logging (empty when fine or unguarded)."""
    guard = load_guard(cfg)
    return [guard.error] if guard is not None and guard.error else []


# ── lookups (block parents and schemas cached; page state always fresh) ──

_BLOCK_CACHE: dict = {}
_SCHEMA_CACHE: dict = {}
_ME_CACHE: dict = {}


def clear_caches() -> None:
    _BLOCK_CACHE.clear()
    _SCHEMA_CACHE.clear()
    _ME_CACHE.clear()


def _home_key() -> str:
    try:
        from hermes_constants import get_hermes_home
        return str(get_hermes_home())
    except Exception:
        return ""


def _ntn_get(path: str) -> dict:
    from tools.notion_api_tool import _run_ntn
    ok, data = _run_ntn(path, "GET", None)
    if not ok:
        raise GuardLookupError(str(data))
    if not isinstance(data, dict):
        raise GuardLookupError(f"unexpected response for {path}")
    return data


class NotionLookup:
    """Reads the guard needs, through the same ``ntn`` credentials path as
    ``notion_api``. Raises ``GuardLookupError``."""

    def __init__(self, run: Optional[Callable[[str], dict]] = None):
        self._run = run or _ntn_get

    def _cached(self, cache: dict, ttl: float, path: str) -> dict:
        key = (_home_key(), path)
        hit = cache.get(key)
        now = time.monotonic()
        if hit and hit[0] > now:
            return hit[1]
        data = self._run(path)
        cache[key] = (now + ttl, data)
        return data

    def page(self, page_id: str) -> dict:
        return self._run(f"v1/pages/{_hex(page_id) or page_id}")

    def block(self, block_id: str) -> dict:
        return self._cached(_BLOCK_CACHE, _BLOCK_TTL, f"v1/blocks/{_hex(block_id) or block_id}")

    def data_source(self, ds_id: str) -> dict:
        return self._cached(_SCHEMA_CACHE, _SCHEMA_TTL, f"v1/data_sources/{_hex(ds_id) or ds_id}")

    def me(self) -> str:
        return str(self._cached(_ME_CACHE, _BLOCK_TTL, "v1/users/me").get("id") or "")


# ── endpoint table ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Endpoint:
    kind: str       # read | create | page_update | page_content | block_content | block_update | comment
    target: str = ""


def _endpoint(method: str, path: str) -> Optional[_Endpoint]:
    if any(c in path for c in "?#%"):
        return None
    segs = path.split("/")
    if len(segs) < 2 or segs[0] != "v1" or not all(_SEGMENT.match(s) for s in segs):
        return None
    s = segs[1:]
    n = len(s)
    ident = _hex(s[1]) if n >= 2 else ""
    if method == "POST":
        if s == ["search"] or s == ["file_uploads"]:
            return _Endpoint("read")
        if n == 3 and s[0] in ("data_sources", "databases") and ident and s[2] == "query":
            return _Endpoint("read")
        if n == 3 and s[0] == "file_uploads" and ident and s[2] in ("send", "complete"):
            return _Endpoint("read")
        if s == ["pages"]:
            return _Endpoint("create")
        if s == ["comments"]:
            return _Endpoint("comment")
    if method in ("POST", "PATCH") and n == 3 and ident:
        if s[0] == "pages" and s[2] == "markdown":
            return _Endpoint("page_content", ident)
        if s[0] == "blocks" and s[2] == "children":
            return _Endpoint("block_content", ident)
    if method == "PATCH" and n == 2 and ident:
        if s[0] == "pages":
            return _Endpoint("page_update", ident)
        if s[0] == "blocks":
            return _Endpoint("block_update", ident)
    return None


# ── target resolution ────────────────────────────────────────────────────

@dataclass
class _Target:
    kind: str            # kit | prd | idea | free
    row: Optional[dict]  # the row page for kit / prd / idea
    is_row: bool         # the requested page IS the row (not a subpage of it)


def _trashed(page: dict) -> bool:
    return any(page.get(k) for k in _TRASH_KEYS)


def _page_of_block(block_id: str, lookup) -> str:
    current = block_id
    for _ in range(_MAX_WALK * 3):
        block = lookup.block(current)
        if not isinstance(block, dict):
            raise _Deny("Notion write refused: could not resolve the block's page.")
        if _trashed(block):
            raise _Deny("Notion write refused: that block is in the trash.")
        if block.get("type") == "child_page":
            return _hex(current) or current
        parent = block.get("parent") if isinstance(block.get("parent"), dict) else {}
        if parent.get("type") == "page_id" and _hex(parent.get("page_id")):
            return _hex(parent["page_id"])
        if parent.get("type") == "block_id" and _hex(parent.get("block_id")):
            current = _hex(parent["block_id"])
            continue
        raise _Deny("Notion write refused: could not resolve which page that block belongs to.")
    raise _Deny("Notion write refused: the block is nested too deeply to resolve.")


def _classify_page(page_id: str, guard: GuardConfig, lookup) -> _Target:
    current_id = _hex(page_id)
    for depth in range(_MAX_WALK):
        page = lookup.page(current_id)
        if not isinstance(page, dict):
            raise _Deny("Notion write refused: could not read the target page.")
        if _trashed(page):
            raise _Deny("Notion write refused: the target page (or its row) is in the trash.")
        if current_id in guard.writable:
            return _Target("free", None, False)
        parent = page.get("parent") if isinstance(page.get("parent"), dict) else {}
        ds, db = _hex(parent.get("data_source_id")), _hex(parent.get("database_id"))
        if ds or db:
            for kind, ids in (("kit", guard.kits), ("prd", guard.prds), ("idea", guard.ideas)):
                if ds in ids or db in ids:
                    return _Target(kind, page, depth == 0)
            raise _Deny("Notion write refused: that page is in a database this profile may not "
                        "write (only Release Kits, Ideas and the PRDs' Release Comms Status).")
        ptype = parent.get("type")
        if ptype == "page_id" and _hex(parent.get("page_id")):
            current_id = _hex(parent["page_id"])
        elif ptype == "block_id" and _hex(parent.get("block_id")):
            current_id = _page_of_block(parent["block_id"], lookup)
        else:
            raise _Deny("Notion write refused: that page is outside the places this profile may "
                        "write (Release Kits, Ideas, its own pages).")
    raise _Deny("Notion write refused: the page is nested too deeply to resolve.")


# ── rules ────────────────────────────────────────────────────────────────

def _select(prop) -> str:
    if not isinstance(prop, dict):
        return ""
    val = prop.get(prop.get("type") or "select")
    return str(val.get("name") or "").strip() if isinstance(val, dict) else ""


def _new_value(name: str, value) -> str:
    """The option NAME a property write sets; refuses ids, null, odd shapes."""
    if not isinstance(value, dict):
        raise _Deny(f"Notion write refused: {name!r} must be set as {{'select'|'status': {{'name': ...}}}}.")
    keys = [k for k in ("status", "select") if k in value]
    if len(keys) != 1:
        raise _Deny(f"Notion write refused: {name!r} must be set by option name.")
    inner = value[keys[0]]
    if not isinstance(inner, dict) or "id" in inner or not isinstance(inner.get("name"), str):
        raise _Deny(f"Notion write refused: {name!r} must be set by option name (not id or null).")
    return inner["name"].strip()


def _normalise(props: dict, schema: dict) -> dict:
    """Map body property keys (names or ids, URL-encoded or not) to names."""
    if not isinstance(props, dict):
        raise _Deny("Notion write refused: properties must be an object.")
    by_id = {}
    for pname, meta in (schema or {}).items():
        if isinstance(meta, dict) and meta.get("id") is not None:
            pid = str(meta["id"])
            by_id[pid] = pname
            by_id[unquote(pid)] = pname
    out = {}
    for key, value in props.items():
        k = str(key)
        if k in schema:
            name = k
        elif k in by_id or unquote(k) in by_id:
            name = by_id.get(k) or by_id[unquote(k)]
        else:
            raise _Deny(f"Notion write refused: unknown property {k!r}.")
        if name in out:
            raise _Deny(f"Notion write refused: property {name!r} given twice.")
        out[name] = value
    return out


def _relation_ids(name: str, value) -> set[str]:
    if not isinstance(value, dict) or set(value) - {"relation", "type", "has_more"}:
        raise _Deny(f"Notion write refused: {name!r} must be set as {{'relation': [{{'id': ...}}]}}.")
    items = value.get("relation")
    if not isinstance(items, list):
        raise _Deny(f"Notion write refused: {name!r} must be a list of related pages.")
    out = set()
    for item in items:
        pid = _hex((item or {}).get("id")) if isinstance(item, dict) else ""
        if not pid:
            raise _Deny(f"Notion write refused: {name!r} has an entry that is not a page id.")
        out.add(pid)
    return out


def _check_relations(props: dict, schema: dict, current: Optional[dict], guard: GuardConfig,
                     lookup) -> None:
    """A two-way relation write also changes every page added or removed on
    the other side — so each of those pages must itself be writable (a Draft
    kit row, its own idea, a free page). Otherwise an edit to a Draft row
    could rewrite an Approved row's relations (and the Jira tickets a
    release update checks), a PRD, or someone else's idea."""
    for name, value in props.items():
        if (schema.get(name) or {}).get("type") != "relation":
            continue
        new = _relation_ids(name, value)
        old: set[str] = set()
        if current is not None:
            cur = current.get(name) if isinstance(current.get(name), dict) else {}
            if cur.get("has_more"):
                raise _Deny(f"Notion write refused: {name!r} has more related pages than Notion "
                            "returned, so the change can't be checked.")
            old = {_hex((i or {}).get("id")) for i in cur.get("relation") or [] if isinstance(i, dict)}
            old.discard("")
        for pid in sorted(new ^ old):
            _content_rule(_classify_page(pid, guard, lookup), guard, lookup)


def _is_own(row: dict, lookup) -> bool:
    me = lookup.me()
    creator = (row.get("created_by") or {}).get("id") if isinstance(row.get("created_by"), dict) else None
    return bool(me) and creator == me


def _content_rule(t: _Target, guard: GuardConfig, lookup) -> None:
    if t.kind == "free":
        return
    if t.kind == "kit":
        status = _select((t.row.get("properties") or {}).get(guard.kit_status))
        if status != guard.kit_draft:
            raise _Deny(f"Notion write refused: this Release Kit row is {status or 'empty'!r} — only "
                        f"{guard.kit_draft!r} rows are editable; the rest is read-only (human-only).")
        return
    if t.kind == "prd":
        raise _Deny("Notion write refused: PRD pages are read-only (only "
                    f"{guard.prd_property!r} may be set, to {' / '.join(sorted(guard.prd_values))}).")
    if t.kind == "idea":
        if not _is_own(t.row, lookup):
            raise _Deny("Notion write refused: that Ideas row is not one this teammate created — read-only.")
        return
    raise _Deny("Notion write refused: unknown target.")


def _page_update(ep: _Endpoint, body: dict, guard: GuardConfig, lookup) -> None:
    extra = set(body) - _PAGE_UPDATE_KEYS
    if extra:
        raise _Deny(f"Notion write refused: page updates may only change properties/icon/cover "
                    f"(got {', '.join(sorted(extra))}).")
    t = _classify_page(ep.target, guard, lookup)
    if t.kind == "free":
        return
    if not t.is_row:
        _content_rule(t, guard, lookup)
        return
    row_props = t.row.get("properties") if isinstance(t.row.get("properties"), dict) else {}
    props = _normalise(body.get("properties") or {}, row_props)
    if t.kind in ("kit", "idea"):
        _content_rule(t, guard, lookup)  # Draft kit row / own idea
        _check_relations(props, row_props, row_props, guard, lookup)
    if t.kind == "kit":
        if guard.kit_status in props and _new_value(guard.kit_status, props[guard.kit_status]) != guard.kit_draft:
            raise _Deny(f"Notion write refused: {guard.kit_status!r} may only be {guard.kit_draft!r} — "
                        "advancing a row is human-only (autonomy action advance_asset_status is off).")
        return
    if t.kind == "prd":
        if set(body) != {"properties"} or set(props) - {guard.prd_property}:
            _content_rule(t, guard, lookup)
        if guard.prd_property in props:
            value = _new_value(guard.prd_property, props[guard.prd_property])
            if value not in guard.prd_values:
                raise _Deny(f"Notion write refused: {guard.prd_property!r} may only be set to "
                            f"{' / '.join(sorted(guard.prd_values))}.")
        return
    if t.kind == "idea":
        _content_rule(t, guard, lookup)  # own rows only
        human = set(props) & guard.idea_human_props
        if human:
            raise _Deny(f"Notion write refused: {', '.join(sorted(human))} is written by humans only.")
        if guard.idea_status in props:
            new = _new_value(guard.idea_status, props[guard.idea_status])
            current = _select(row_props.get(guard.idea_status))
            if new in guard.idea_human_statuses:
                raise _Deny(f"Notion write refused: setting an idea to {new!r} is a human decision "
                            "(autonomy action ideas_accept_own is off).")
            if new != current and new not in guard.idea_transitions.get(current, frozenset()):
                raise _Deny(f"Notion write refused: an idea it owns may move {current or 'empty'!r} → "
                            f"{' / '.join(sorted(guard.idea_transitions.get(current, ()))) or 'nothing'}, "
                            f"not {new!r}.")
        return
    raise _Deny("Notion write refused: unknown target.")


def _create(body: dict, guard: GuardConfig, lookup) -> None:
    extra = set(body) - _PAGE_CREATE_KEYS
    if extra:
        raise _Deny(f"Notion write refused: page creation may not use {', '.join(sorted(extra))}.")
    parent = body.get("parent") if isinstance(body.get("parent"), dict) else {}
    ds = _hex(parent.get("data_source_id"))
    if ds:
        if ds in guard.prds:
            raise _Deny("Notion write refused: PRD pages are read-only; it may not create them.")
        if ds not in guard.kits and ds not in guard.ideas:
            raise _Deny("Notion write refused: it may only create rows in Release Kits and Ideas.")
        schema = lookup.data_source(ds)
        schema_props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        props = _normalise(body.get("properties") or {}, schema_props)
        _check_relations(props, schema_props, None, guard, lookup)
        if ds in guard.kits:
            if guard.kit_status not in props or _new_value(guard.kit_status, props[guard.kit_status]) != guard.kit_draft:
                raise _Deny(f"Notion write refused: a new Release Kit row must set {guard.kit_status!r} "
                            f"to {guard.kit_draft!r} explicitly.")
            return
        human = set(props) & guard.idea_human_props
        if human:
            raise _Deny(f"Notion write refused: {', '.join(sorted(human))} is written by humans only.")
        if guard.idea_status not in props or _new_value(guard.idea_status, props[guard.idea_status]) != guard.idea_create:
            raise _Deny(f"Notion write refused: a new idea must set {guard.idea_status!r} to "
                        f"{guard.idea_create!r} explicitly.")
        return
    if _hex(parent.get("database_id")):
        raise _Deny("Notion write refused: create rows with parent.data_source_id (the guard checks "
                    "data sources), not database_id.")
    if _hex(parent.get("page_id")):
        _content_rule(_classify_page(parent["page_id"], guard, lookup), guard, lookup)
        return
    if _hex(parent.get("block_id")):
        _content_rule(_classify_page(_page_of_block(parent["block_id"], lookup), guard, lookup),
                      guard, lookup)
        return
    raise _Deny("Notion write refused: a new page needs a parent the guard can check.")


def _comment(body: dict, guard: GuardConfig, lookup) -> None:
    parent = body.get("parent") if isinstance(body.get("parent"), dict) else {}
    if _hex(parent.get("page_id")):
        page = parent["page_id"]
    elif _hex(parent.get("block_id")):
        page = _page_of_block(parent["block_id"], lookup)
    else:
        raise _Deny("Notion write refused: comments need parent.page_id or parent.block_id "
                    "(replies by discussion_id can't be checked).")
    _content_rule(_classify_page(page, guard, lookup), guard, lookup)


def _dangerous_nested(body, depth: int = 0) -> Optional[str]:
    """Deletes and schema creation hidden inside a body: the markdown
    endpoint's ``allow_deleting_content`` (replace_content may trash child
    pages / databases) and ``child_database`` blocks."""
    if depth > 40:
        return "Notion write refused: the request body is nested too deeply to check."
    if isinstance(body, dict):
        if body.get("allow_deleting_content"):
            return ("Notion write refused: allow_deleting_content can trash child pages and "
                    "databases — never allowed for this profile.")
        if body.get("type") == "child_database" or "child_database" in body:
            return "Notion write refused: creating databases (child_database) is a schema change."
        for value in body.values():
            hit = _dangerous_nested(value, depth + 1)
            if hit:
                return hit
    elif isinstance(body, list):
        for value in body:
            hit = _dangerous_nested(value, depth + 1)
            if hit:
                return hit
    return None


def check_request(method: str, path: str, body, *, cfg: dict, lookup=None) -> Optional[str]:
    """Return a denial message for this ``notion_api`` request, or None."""
    method = str(method or "GET").strip().upper()
    if method == "GET":
        return None
    guard = load_guard(cfg)
    if guard is None:
        return None
    path = str(path or "").strip()
    ep = _endpoint(method, path)
    if ep is not None and ep.kind == "read":
        return None
    if guard.error:
        return (f"Notion write refused: this profile's notion.write_guard is enabled but invalid "
                f"({guard.error}). Every Notion write is blocked until an admin fixes config.yaml.")
    if method == "DELETE":
        return "Notion write refused: DELETE is never allowed for this profile (notion.write_guard)."
    if ep is None:
        return (f"Notion write refused: {method} {path} is not an endpoint the write guard allows "
                "(no schema changes, no moves, no unknown endpoints).")
    if body is None:
        body = {}
    if not isinstance(body, dict):
        return "Notion write refused: body must be a JSON object."
    if _TRASH_KEYS & set(body):
        return "Notion write refused: archiving, trashing or restoring is never allowed for this profile."
    nested = _dangerous_nested(body)
    if nested:
        return nested
    lookup = lookup or NotionLookup()
    try:
        if ep.kind == "create":
            _create(body, guard, lookup)
        elif ep.kind == "page_update":
            _page_update(ep, body, guard, lookup)
        elif ep.kind == "page_content":
            _content_rule(_classify_page(ep.target, guard, lookup), guard, lookup)
        elif ep.kind in ("block_content", "block_update"):
            page = _page_of_block(ep.target, lookup)
            _content_rule(_classify_page(page, guard, lookup), guard, lookup)
        elif ep.kind == "comment":
            _comment(body, guard, lookup)
        else:  # pragma: no cover - table and dispatch kept in step
            return "Notion write refused: unknown endpoint."
    except _Deny as d:
        return str(d)
    except GuardLookupError as e:
        return f"Notion write refused: could not resolve the write's target ({str(e)[:200]})."
    return None


def _request_parts(args) -> tuple[str, str, object]:
    args = args if isinstance(args, dict) else {}
    return (str(args.get("method") or "GET").strip().upper(),
            str(args.get("path") or "").strip(), args.get("body"))


def guard_denial(method: str, path: str, body) -> Optional[str]:
    """``check_request`` against the active profile's config.yaml. A config
    read failure denies writes (fail closed)."""
    if str(method or "GET").strip().upper() == "GET":
        return None
    try:
        from agent.boundary_config import read_boundary_config
        cfg = read_boundary_config()
    except Exception as e:
        logger.error("[notion_write_guard] config read failed: %s", e)
        return "Notion write refused: the profile's config could not be read to check the write guard."
    try:
        denial = check_request(method, path, body, cfg=cfg)
    except Exception as e:  # any bug here must not open the gate
        logger.exception("[notion_write_guard] evaluation failed")
        return f"Notion write refused: the write guard failed ({type(e).__name__})."
    if denial:
        logger.warning("[notion_write_guard] %s %s denied: %s", method, path, denial)
    return denial


def denial_for_tool_call(function_name: str, args) -> Optional[str]:
    """The ``model_tools`` execution backstop: same decision as the handler."""
    if function_name != "notion_api":
        return None
    method, path, body = _request_parts(args)
    return guard_denial(method, path, body)
