"""Profile JSON-RPC handlers — the ws twin of the dashboard's /api/profiles (desktop plugins
only have the ws door), on the same `hermes_cli.profiles` primitives. Bodies are rebound onto
server.py's globals (method_ctx.bind_module) and use them bare; module-level names are published
onto server.py, so they must not collide with its globals.
"""

import logging
import sqlite3

from hermes_cli.profiles_cache import yaml_scalar_to_json as _yaml_scalar_to_json
from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method

# ext -> mime; iteration order is the on-disk lookup order for assets.
_ASSET_EXTS = {"png": "image/png", "jpg": "image/jpeg", "webp": "image/webp"}
# ext -> [(start, end, magic bytes)]; format is sniffed, the declared mime is never trusted.
_ASSET_MAGIC = {"png": [(0, 8, b"\x89PNG\r\n\x1a\n")], "jpg": [(0, 3, b"\xff\xd8\xff")],
                "webp": [(0, 4, b"RIFF"), (8, 12, b"WEBP")]}


def _profile_handler(name: str, code: int):
    """``@method(name)`` whose body's uncaught exception becomes ``_err(rid, code, str(e))``."""
    def deco(fn):
        def handler(rid, params: dict) -> dict:
            try:
                return fn(rid, params)
            except Exception as e:
                return _err(rid, code, str(e))
        return method(name)(handler)
    return deco


def _lazy(module, name):
    """Late-bound attribute lookup; ``__import__`` builtin on purpose (rebound bodies see only
    server.py globals, not this module's imports)."""
    return getattr(__import__(module, fromlist=[name]), name)


def _pin_profile_model(profile_dir, provider, model, base_url="") -> None:
    _lazy("hermes_cli.profile_provision", "write_profile_model")(
        profile_dir, provider, model, base_url=base_url)


def _model_provider_params(params) -> tuple:
    return str(params.get("model") or "").strip(), str(params.get("provider") or "").strip()


def _try(fn, default):
    """``fn()`` or ``default`` on any exception (best-effort sections must never fail each other)."""
    try:
        return fn()
    except Exception:
        return default


def _best_effort(fn) -> bool:
    return _try(lambda: (fn(), True)[1], False)


def _hermes_home_scope(path):
    from tui_gateway.launch_profile_policy import profile_config_scope
    return profile_config_scope(path, launch_home=_launch_home())


def _resolve_profile(rid, params):
    """``(name, profile_dir, err)``; err = 4063 (name required) / 4064 (not found) response."""
    name = str(params.get("name") or "").strip()
    if not name:
        return name, None, _err(rid, 4063, "name required")
    from hermes_cli.profiles import get_profile_dir
    try:
        profile_dir = Path(get_profile_dir(name))
    except ValueError:
        return name, None, _err(rid, 4064, f"profile '{name}' not found")
    if not profile_dir.is_dir():
        return name, None, _err(rid, 4064, f"profile '{name}' not found")
    return name, profile_dir, None


def _read_profile_yaml(profile_dir) -> dict:
    """profile.yaml as a mapping; ``{}`` when missing, unreadable, unparseable, or not a mapping."""
    def load():
        import hermes_yaml as yaml
        meta_path = profile_dir / "profile.yaml"
        return (yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}) if meta_path.is_file() else {}
    loaded = _try(load, {})
    return loaded if isinstance(loaded, dict) else {}


def _clean_revisions(raw: dict) -> dict:
    """Normalise a ``_ui_meta_revisions`` map: str keys, non-bool ints clamped at 0."""
    return {str(k): max(0, int(v)) for k, v in raw.items() if isinstance(v, int) and not isinstance(v, bool)}


def _latest_message_preview(db, session_id):
    """≤80-char excerpt of the NEWEST active user/assistant message, or "" (roster semantics).
    Same query shape as ``SessionDB.latest_message_row_id``; keep them in step."""
    try:
        with db._lock:
            row = db._conn.execute(
                "SELECT content FROM messages"
                " WHERE session_id = ? AND role IN ('user', 'assistant')"
                " AND active = 1 AND content IS NOT NULL AND TRIM(content) != ''"
                " ORDER BY id DESC LIMIT 1",
                (session_id,)).fetchone()
    except Exception:
        return ""
    text = " ".join(str(row[0] or "").split()).strip() if row else ""
    return text[:80] + "..." if len(text) > 80 else text


def _resurrect_recoverable_canonical(db, profile_path, session_id):
    """Un-archive an accidentally archived canonical row (judged read-only, written via a
    short-lived writable handle); False otherwise.

    See #92687.
    """
    try:
        row = db.get_session(session_id)
        if not row or not row.get("archived"):
            return False
        tip_id = _try(lambda: db.get_compression_tip(session_id), None) or session_id
        tip = (_try(lambda: db.get_session(tip_id), None) or row) if tip_id != session_id else row
        from hermes_state import SessionDB
        from hermes_state_registry import acquire
        if (tip.get("end_reason") or "") not in SessionDB.RECOVERABLE_END_REASONS:
            return False
        wdb = acquire(Path(profile_path) / "state.db")
        try:
            return bool(wdb.unarchive_recoverable_session(session_id))
        finally:
            _best_effort(lambda: _lazy("hermes_state_registry", "release_or_close")(wdb))
    except Exception:
        return False


def _live_count_field(db, session_id) -> dict:
    """``{"live_message_count": n}`` sized like a stored-transcript read, else ``{}`` (older stores).

    The denormalized ``message_count`` also counts folded rows (orphaned compaction marks, full
    rewinds, model-only rows), which once made the roster wait for history no reader serves."""
    try:
        return {"live_message_count": db.display_message_count(str(session_id))}
    except sqlite3.Error:
        return {}


def _canonical_session_row(db, profile_path):
    """Summary of the profile's canonical "Bot Chat" row (identity is the NAME), or None.
    Lineages via ``get_compression_tip`` (NOT the resume walker's unmarked-child fallback);
    worker sources count as absent. ``id`` is the registry row, ``resolved_id`` the live tip.

    The canonical chat's identity is the NAME: the session titled exactly "Bot Chat" on this profile (core
    UNIQUE(title) makes it a registry of at most one row). Complements ``last_session``: that field answers
    "what is the newest conversation", this answers "where is the forever-chat" — so a roster row's preview
    and its click target describe the same session (hermes-agent#88200) with no client-side pointer
    involved.
    """
    try:
        row = db.get_session_by_title("Bot Chat")
        session_id = str((row or {}).get("id") or "").strip()
        if not session_id or _denied_source(row):
            return None
        # Archived = retired (absent), except accidental reaper archives: resurrect those.
        # An archived canonical row usually means the user deliberately retired it — report absent. But the
        # ws-orphan reaper / older agent cleanup can archive it by accident (#92687): resurrect those. Judge
        # recoverability READ-ONLY first so the writable open (20s write-lock patience, the very stall this
        # refactor removes from the 5s poll) is paid only in the rare accidental-archive case, then run the
        # real predicate through unarchive_recoverable_session on a short-lived writable handle.
        if row.get("archived") and not _resurrect_recoverable_canonical(db, profile_path, session_id):
            return None
        tip = _try(lambda: db.get_compression_tip(session_id), None) or session_id
        tip_row = db.get_session(tip) or row
        started = row.get("started_at") or 0
        return {
            "id": session_id, "resolved_id": tip, "root_title": row.get("title") or "",
            "title": tip_row.get("title") or "", "preview": _latest_message_preview(db, tip),
            "started_at": tip_row.get("started_at") or started,
            "last_active": tip_row.get("last_activity_at") or tip_row.get("started_at") or started,
            "message_count": tip_row.get("message_count") or 0, **_live_count_field(db, tip)}
    except Exception:
        return None


def _latest_profile_session_rows(db):
    """(newest human-facing session, newest worker session); the worker row lets rosters show a
    profile as working (workers heartbeat ``last_activity_at`` every ≤60s).

    First element mirrors session.list's deny-list (drops ``tool`` sub-agent rows and ``kanban`` dispatcher
    workers). Second element is the newest DENIED row — the freshest kanban/tool worker — so roster UIs can
    show that a profile is actively working even though worker sessions never surface in conversation lists
    (hermes-agent#90268). Workers heartbeat ``last_activity_at`` every ≤60s while running (#72016), so a
    live worker's ``last_active`` stays fresh and the client can apply its own liveness window. Best-effort:
    any failure (missing state.db, locked db, older schema) degrades to (None, None) rather than failing the
    whole profiles.list call.
    """
    try:
        human = worker = None
        # The roster polls every 5s: hydrate indexed candidates, not the entire session history.
        for s in db.list_recent_sessions_bounded(limit=20):
            title, last_active = s.get("title") or "", s.get("last_active") or s.get("started_at") or 0
            if _denied_source(s):
                if worker is None:
                    worker = {"id": s["id"], "source": (s.get("source") or "").strip().lower(),
                              "title": title, "last_active": last_active}
            elif human is None:  # rosters want "where the conversation IS": prefer the newest text
                human = {"id": s["id"], "title": title,
                         "preview": _latest_message_preview(db, s["id"]) or s.get("preview") or "",
                         "started_at": s.get("started_at") or 0, "last_active": last_active,
                         "message_count": s.get("message_count") or 0,
                         **_live_count_field(db, s["id"])}
            if human is not None and worker is not None:
                break
        return human, worker
    except Exception:
        return None, None


def _profile_session_fields(row, profile_path):
    """Attach last_session / worker_session / canonical_session to a roster row. The DB is a
    read-only attach (a writable ``SessionDB()`` waits up to 20s for the write lock + runs DDL
    and stalled the 5s roster poll); no/unreadable DB -> every field None (the readers swallow)."""
    def _read() -> dict:
        db_path = Path(profile_path) / "state.db"
        db = None
        if _try(db_path.exists, False):
            db = _try(lambda: _lazy("hermes_state", "SessionDB")(db_path=db_path, read_only=True), None)
        try:
            last, worker = _latest_profile_session_rows(db)
            # Resolved server-side on every listing so no client carries a session pointer.
            return {"last_session": last, "worker_session": worker,
                    "canonical_session": _canonical_session_row(db, profile_path)}
        finally:
            if db is not None:
                _best_effort(db.close)

    # These three are a pure function of the profile's session store, and the roster re-asks every
    # 5s per connection — so they are reused while that store has not moved (#117257).
    from tui_gateway.profile_roster_cache import cached_session_fields

    row.update(cached_session_fields(profile_path, _read))


def _profile_ui_meta_fields(row: dict, profile_dir) -> None:
    """Attach ``ui_meta`` / ``ui_meta_revisions`` / ``has_avatar`` from profile.yaml + assets.
    ``ui_meta_revisions`` is always present: it feature-detects gateway-owned CAS for a new profile."""
    def _read() -> dict:
        raw_meta = _read_profile_yaml(profile_dir)
        ui_meta, revisions = raw_meta.get("ui_meta"), raw_meta.get("_ui_meta_revisions")
        # Key order is wire-visible: ui_meta_revisions precedes ui_meta.
        fields = {"ui_meta_revisions":
                  _try(lambda: _clean_revisions(revisions), {}) if isinstance(revisions, dict) else {}}
        if isinstance(ui_meta, dict) and ui_meta:
            # YAML promotes unquoted timestamps to datetime/date; the handler's contract is JSON, so
            # coerce YAML-only scalars to their ISO string at the boundary (#92506).
            fields["ui_meta"] = json.loads(json.dumps(ui_meta, default=_yaml_scalar_to_json))
        return fields

    # Second parse of this profile.yaml in the same request (``read_profile_meta`` already read it
    # for the row's description/display name) — reused while the file has not moved (#117383). The
    # ui_meta CAS writer below keeps reading it raw and uncached: it writes the document back.
    from tui_gateway.profile_roster_cache import cached_ui_meta_fields

    row.update(cached_ui_meta_fields(profile_dir, _read))
    # Cheap existence flag so rosters skip a get_asset probe per paint.
    row["has_avatar"] = _try(lambda: any((profile_dir / "assets" / f"avatar.{e}").is_file() for e in _ASSET_EXTS), False)


@_profile_handler("profiles.list", 5061)
def _(rid, params: dict) -> dict:
    """List Hermes profiles. ``include_sessions`` (default true) adds ``last_session`` /
    ``worker_session`` / ``canonical_session`` so a roster paints previews without N calls."""
    from hermes_cli.isolated_scope import scope_to_launch_profile
    from hermes_cli.profiles import list_profiles
    include_sessions = is_truthy_value(params.get("include_sessions", True))
    out = []
    # Roster polls this every 5s: ``skill_count`` is the last known value, refreshed off-request.
    for p in scope_to_launch_profile(list_profiles(lazy_skill_count=True), lambda info: info.name):
        row = {"name": p.name, "path": str(p.path), "is_default": bool(p.is_default), "model": p.model,
               "provider": p.provider, "description": p.description or "",
               "display_name": p.display_name or "", "skill_count": p.skill_count or 0,
               "previous_names": list(p.previous_names or [])}
        if include_sessions:
            _profile_session_fields(row, p.path)
        _profile_ui_meta_fields(row, Path(str(p.path)))
        out.append(row)
    # bot_mode_protocol: this backend injects the Bot Mode teammate-messaging protocol into every
    # session, so clients must not append it to SOUL.md. install_id (same value as /api/status)
    # lets a multi-connection client prove WHICH machine answered a routed list.
    from hermes_cli.install_identity import get_install_id
    return _ok(rid, {"profiles": out, "bot_mode_protocol": True, "install_id": _try(lambda: get_install_id() or "", "")})


@method("profiles.create")
def _(rid, params: dict) -> dict:
    """Create a profile (ws twin of POST /api/profiles). Params: ``name``, ``description``,
    ``clone_from`` (omitted = fresh + bundled skills, or with ``clone_all`` a full copy of the active profile), ``clone_all``, ``clone_channels`` (opt-in: keep the
    source's bot tokens/allowlists — default strips them so two profiles never hold one bot), ``no_skills``, ``soul``,
    ``model`` + ``provider``, ``no_alias``, ``share_auth`` / ``mirror_credentials`` (both default true, the
    bot page's defaults: a bare ``create_profile()`` seeds a comment-only .env = NO provider headless).
    ``hermes profile create`` also sends ``clone_config`` (``--clone``: copy the caller's profile),
    ``sync_imports`` and ``report`` (add the receipt it prints, read here).

    On ``/api/ws`` it runs in the connection's profile (``ws_legacy_fallback.dispatch_legacy`` binds the
    ticket's), so ``--clone`` and the copied keys/model come from the profile the terminal is on. A
    refused create answers 4062 with ``data.type`` / ``data.message_parts`` so a client re-raises it."""
    from hermes_cli.config_backend import ConfigBackendUnavailable
    from hermes_cli.profile_provision import (
        CREATE_REFUSALS, ProfileCreateReceipt, ProfileCreateRequest, provision_profile, refusal,
    )
    name = str(params.get("name") or "").strip()
    if not name:  # the CLI shows its usual empty-name error, never "may have created"
        return _err(rid, 4061, "name required", data=refusal(ValueError("profile name cannot be empty"))[1])
    clone_from = str(params.get("clone_from") or "").strip() or None
    clone_all = is_truthy_value(params.get("clone_all", False))
    clone_config = is_truthy_value(params.get("clone_config", False)) or (bool(clone_from) and not clone_all)
    soul = params.get("soul")
    soul = soul if isinstance(soul, str) and soul.strip() else None
    model, provider = _model_provider_params(params)
    request = ProfileCreateRequest(
        name=name, clone_from=clone_from, clone_all=clone_all, clone_config=clone_config,
        no_skills=is_truthy_value(params.get("no_skills", False)),
        description=str(params.get("description") or "").strip() or None,
        clone_channels=is_truthy_value(params.get("clone_channels", False)), soul=soul,
        sync_imports=is_truthy_value(params.get("sync_imports", False)),
        alias=not is_truthy_value(params.get("no_alias", False)),
        mirror_credentials=is_truthy_value(params.get("mirror_credentials"), True),
        share_auth=is_truthy_value(params.get("share_auth"), True),
        model=model, provider=provider)
    try:
        result = provision_profile(request)
        receipt = ProfileCreateReceipt.of(request, result) if is_truthy_value(params.get("report", False)) else None
    except (Exception, ConfigBackendUnavailable) as e:  # health: allow BLE001 -- the request's one reply (a config outage is a SystemExit)
        # The full text (redacted) for Desktop's dialog; ``data`` carries it in the parts a client keeps.
        message, data = refusal(e)
        if isinstance(e, CREATE_REFUSALS):
            return _err(rid, 4062, message, data=data)
        # Not a refusal: it may have failed after the profile was made, so never "nothing was created".
        logging.getLogger(__name__).warning("profiles.create %r failed", name, exc_info=True)
        return _err(rid, 5062, message, data=data)
    out = {"ok": True, "name": name, "path": str(result.path), "soul_written": soul is not None,
           "model_set": result.model_set, "mirrored": result.mirrored, "warnings": result.warnings}
    if receipt is not None:
        out["receipt"] = receipt.to_wire()
    return _ok(rid, out)


@_profile_handler("profiles.describe", 5063)
def _(rid, params: dict) -> dict:
    """Editor snapshot; installed skills are enabled unless in ``skills.disabled``; ``mcp_servers``
    is ``[{name, enabled, transport}]`` (best-effort)."""
    name, profile_dir, err = _resolve_profile(rid, params)
    if err is not None:
        return err
    with _hermes_home_scope(profile_dir):
        from agent.skill_utils import iter_skill_index_files
        from hermes_cli.config import load_config
        from hermes_cli.skills_config import get_disabled_skills
        cfg = load_config() or {}
        disabled = {s.lower() for s in get_disabled_skills(cfg)}
        skills_root = profile_dir / "skills"
        installed = [
            {"name": md.parent.name, "enabled": md.parent.name.lower() not in disabled}
            for md in (iter_skill_index_files(skills_root, "SKILL.md") if skills_root.is_dir() else ())]
        toolsets_out, pinned_set = _describe_toolsets(cfg)
        soul_path = profile_dir / "SOUL.md"
        soul = _try(lambda: soul_path.read_text(encoding="utf-8", errors="replace") if soul_path.is_file() else "", "")
        mcp_cfg = cfg.get("mcp_servers")
        mcp_out = _try(lambda: [
            {"name": str(srv_name), "enabled": _mcp_entry_enabled(entry),
             "transport": str(entry.get("transport") or "http") if entry.get("url") else "stdio"}
            for srv_name in sorted(mcp_cfg.keys()) for entry in (mcp_cfg[srv_name],)
            if isinstance(entry, dict)
        ], []) if isinstance(mcp_cfg, dict) else []
        model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
        meta = _try(lambda: _lazy("hermes_cli.profiles", "read_profile_meta")(profile_dir), {})
        return _ok(rid, {
            "name": name, "description": str(meta.get("description") or ""), "soul": soul,
            "model": {"provider": str(model_cfg.get("provider") or ""),
                      "default": str(model_cfg.get("default") or "")},
            "skills": installed, "toolsets": toolsets_out,
            "toolsets_pinned": pinned_set is not None, "mcp_servers": mcp_out})


@_profile_handler("profiles.configure", 5064)
def _(rid, params: dict) -> dict:
    """Editor Save: ``name`` plus any of ``ui_meta`` (+ ``ui_meta_expected_revisions``), ``soul``,
    ``description``, ``model`` + ``provider`` (+ ``confirm_expensive_model``), ``disabled_skills``,
    ``enabled_toolsets``, ``enabled_mcp_servers``; sections are independent, ``applied`` reports each."""
    _name, profile_dir, err = _resolve_profile(rid, params)
    if err is not None:
        return err
    applied = {}
    if isinstance(params.get("ui_meta"), dict):
        _configure_ui_meta(profile_dir, params, applied)
    if isinstance(params.get("soul"), str):
        applied["soul"] = _best_effort(lambda: (profile_dir / "SOUL.md").write_text(params["soul"], encoding="utf-8"))
    if isinstance(params.get("description"), str):
        write_meta = _lazy("hermes_cli.profiles", "write_profile_meta")
        applied["description"] = _best_effort(lambda: write_meta(
            profile_dir, description=params["description"].strip(), description_auto=False))
    confirm_message = _configure_model(profile_dir, params, applied)
    if any(isinstance(params.get(k), list) for k in ("disabled_skills", "enabled_toolsets", "enabled_mcp_servers")):
        _configure_cfg_sections(profile_dir, params, applied)
    # confirm_* is the shape config.set returns, so clients reuse one confirm handler.
    return _ok(rid, {"ok": all(applied.values()) if applied else True, "applied": applied,
                     **({"confirm_required": True, "confirm_message": confirm_message}
                        if confirm_message is not None else {})})


@_profile_handler("profiles.set_asset", 5065)
def _(rid, params: dict) -> dict:
    """Store ``assets/<asset>.<ext>`` atomically. Params: ``name``, ``asset`` (``"avatar"`` only),
    ``data`` (data URL or base64; PNG/JPEG/WebP ≤2MB, format sniffed) or ``clear: true``."""
    asset = str(params.get("asset") or "avatar").strip().lower()
    if not str(params.get("name") or "").strip():
        return _err(rid, 4063, "name required")
    if asset != "avatar":
        return _err(rid, 4066, f"unknown asset '{asset}' (supported: avatar)")
    import base64
    import re
    _name, profile_dir, err = _resolve_profile(rid, params)
    if err is not None:
        return err
    assets_dir = profile_dir / "assets"
    if is_truthy_value(params.get("clear", False)):
        return _ok(rid, {"ok": True, "asset": asset, "size": 0, "removed": _unlink_asset_files(assets_dir, asset)})
    data = str(params.get("data") or "")
    if not data:
        return _err(rid, 4067, "data required (data URL or base64)")
    match = re.match(r"^data:(image/(?:png|jpeg|webp));base64,(.*)$", data, re.DOTALL)
    try:
        blob = base64.b64decode(match.group(2) if match else data, validate=True)
    except Exception:
        return _err(rid, 4068, "data is not valid base64")
    if len(blob) > 2_000_000:
        return _err(rid, 4069, f"asset too large ({len(blob)} bytes; max 2MB)")
    ext = next((e for e, magic in _ASSET_MAGIC.items() if all(blob[a:b] == m for a, b, m in magic)), None)
    if ext is None:
        return _err(rid, 4070, "unsupported image format (PNG/JPEG/WebP only)")
    assets_dir.mkdir(parents=True, exist_ok=True)
    _unlink_asset_files(assets_dir, asset)  # one canonical file per asset
    tmp = assets_dir / f"{asset}.{ext}.tmp"
    tmp.write_bytes(blob)
    tmp.replace(assets_dir / f"{asset}.{ext}")
    return _ok(rid, {"ok": True, "asset": asset, "size": len(blob)})


@_profile_handler("profiles.get_asset", 5066)
def _(rid, params: dict) -> dict:
    """Profile asset as a data URL; absent is ``found: false``, not an error."""
    asset = str(params.get("asset") or "avatar").strip().lower()
    import base64
    _name, profile_dir, err = _resolve_profile(rid, params)
    if err is not None:
        return err
    for ext, mime in _ASSET_EXTS.items():
        target = profile_dir / "assets" / f"{asset}.{ext}"
        if target.is_file():
            blob = target.read_bytes()
            return _ok(rid, {"found": True, "mime": mime, "size": len(blob),
                             "data": f"data:{mime};base64,{base64.b64encode(blob).decode('ascii')}"})
    return _ok(rid, {"found": False})


def _describe_toolsets(cfg):
    """``(toolsets, pinned_set)`` as the `hermes tools` checklist presents them (the raw registry
    leaks platform composites and reports everything enabled without a pin)."""
    from hermes_cli.tools_config import (
        _coerce_platform_toolsets_value, _get_effective_configurable_toolsets, _get_platform_tools,
        _toolset_allowed_for_platform)
    from toolsets import resolve_toolset
    pinned = _coerce_platform_toolsets_value((cfg.get("platform_toolsets") or {}).get("cli"), "cli")
    pinned_set = _clean_names(pinned) if isinstance(pinned, list) else None
    platform_enabled = _try(lambda: set(_get_platform_tools(cfg, "cli", include_default_mcp_servers=False)), set())
    default_off = _try(lambda: _lazy("hermes_cli.tools_config", "_DEFAULT_OFF_TOOLSETS"), set())
    toolsets_out = []
    for ts_name, ts_label, ts_desc in _get_effective_configurable_toolsets():
        enabled = ts_name in (pinned_set if pinned_set is not None else platform_enabled)
        # Default-off integrations are noise unless already enabled.
        if not _toolset_allowed_for_platform(ts_name, "cli") or (ts_name in default_off and not enabled):
            continue
        toolsets_out.append({"name": ts_name, "label": ts_label, "description": ts_desc or "",
                             "tool_count": _try(lambda: len(set(resolve_toolset(ts_name))), 0),
                             "enabled": enabled})
    return toolsets_out, pinned_set


def _bots_title(ui_meta: dict):
    bots = ui_meta.get("hermes-bots")
    return bots.get("title") if isinstance(bots, dict) else None


def _configure_ui_meta(profile_dir, params, applied) -> None:
    """Merge ``params["ui_meta"]`` key-wise into profile.yaml (None deletes). 64KB cap (rides
    every roster paint). ``ui_meta_expected_revisions``: per-key CAS, any mismatch rejects the
    whole write; revisions survive deletion so a stale client cannot recreate a removed key."""
    applied["ui_meta"] = False
    try:
        incoming = params["ui_meta"]
        if len(json.dumps(incoming)) > 65536:
            return
        expected = params.get("ui_meta_expected_revisions")
        if expected is not None and not isinstance(expected, dict):
            raise ValueError("ui_meta_expected_revisions must be an object")
        with _profile_ui_meta_lock:
            existing = _read_profile_yaml(profile_dir)
            raw_revisions = existing.get("_ui_meta_revisions")
            revisions = _clean_revisions(raw_revisions if isinstance(raw_revisions, dict) else {})
            conflicts = {}
            for key in incoming if isinstance(expected, dict) else ():
                wanted, actual = expected.get(key), revisions.get(key, 0)
                if not isinstance(wanted, int) or isinstance(wanted, bool) or wanted < 0 or wanted != actual:
                    conflicts[key] = {"expected": wanted, "actual": actual}
            if conflicts:
                applied["ui_meta_conflicts"] = conflicts
                applied["ui_meta_revisions"] = {key: revisions.get(key, 0) for key in incoming}
                return
            current = existing.get("ui_meta")
            current = current if isinstance(current, dict) else {}
            old_title = _bots_title(current)
            for key, value in incoming.items():
                if value is None:
                    current.pop(key, None)
                else:
                    current[key] = value
                revisions[key] = revisions.get(key, 0) + 1
            if current:
                existing["ui_meta"] = current
            else:
                existing.pop("ui_meta", None)
            existing["_ui_meta_revisions"] = revisions
            from utils import atomic_yaml_write
            atomic_yaml_write(profile_dir / "profile.yaml", existing, sort_keys=False)
            applied["ui_meta"] = True
            applied["ui_meta_revisions"] = {key: revisions[key] for key in incoming}
            if _bots_title(current) != old_title:
                # A client writing another machine's bot title lands here; the Desktop's
                # `[bot-meta win=…]` desktop.log line at the same time names the window.
                logging.getLogger(__name__).info(
                    "ui_meta hermes-bots title for profile %s: %r -> %r (revision %s)",
                    params.get("name") or profile_dir.name, old_title, _bots_title(current),
                    revisions.get("hermes-bots"))
    except Exception:
        applied["ui_meta"] = False


def _profile_group_selection_context(profile_dir, target_model):
    """The largest measured context among this profile's affected group members."""
    from agent.usage_anchor import USAGE_ANCHOR_MODEL_CONFIG_KEY, persisted_anchor_tokens
    from hermes_cli.model_selection_guards import SelectionContext
    from hermes_state import SessionDB

    path = profile_dir / "state.db"
    if not path.is_file():
        return None

    largest = None
    with SessionDB(path, read_only=True) as db:
        sessions = db.list_sessions_rich(source="bot_room", limit=-1, include_hidden=True, compact_rows=True)
        for session in sessions:
            current_model = session["model"]
            if current_model == target_model:
                continue
            sid = session["id"]
            if not db.get_session_model_config_value(sid, USAGE_ANCHOR_MODEL_CONFIG_KEY):
                continue
            tokens = persisted_anchor_tokens(db, sid, db.get_messages_as_conversation(sid))
            if tokens and (largest is None or tokens > (largest.context_tokens or 0)):
                largest = SelectionContext(context_tokens=tokens, current_model=current_model)
    return largest


def _configure_model(profile_dir, params, applied):
    """Apply a ``model`` + ``provider`` pin, or return a confirm message and write NOTHING (client
    resends with ``confirm_expensive_model``). A failing guard = no warning (as _apply_model_switch)."""
    model, provider = _model_provider_params(params)
    if not (model and provider):
        return None
    confirm_message = None
    # #95293 remainder: this is the Bots editor's model-switch path, and it used to write guarded
    # (data-policy / expensive) models silently — the ONE surface that bypassed the selection guard every
    # other switch path enforces. Same handshake contract as ``config.set model``: without
    # ``confirm_expensive_model`` a guarded pick answers ``confirm_required`` + ``confirm_message`` and
    # writes NOTHING; the client resends with ``confirm_expensive_model: true`` once the user confirms. A
    # misbehaving guard must never break the save (treated as "no warning"), matching
    # ``_apply_model_switch``.
    if not is_truthy_value(params.get("confirm_expensive_model", False)):
        warn = _lazy("hermes_cli.model_selection_guards", "combined_selection_warning")
        context = _try(lambda: _profile_group_selection_context(profile_dir, model), None)
        with _hermes_home_scope(profile_dir):
            confirm_message = _try(
                lambda: getattr(warn(model, provider=provider, selection_context=context), "message", None), None)
    if confirm_message is None:
        applied["model"] = _best_effort(lambda: _pin_profile_model(profile_dir, provider, model))
    return confirm_message


def _clean_names(values) -> set:
    return {str(v).strip() for v in values if str(v).strip()}


def _save_toolset_pin(cfg, enabled, save_config) -> None:
    """Pin ``platform_toolsets.cli``: the key ``_load_enabled_toolsets`` reads and ``hermes tools`` writes.
    An empty selection clears the pin so the platform default applies again."""
    from hermes_cli.tools_config import _save_platform_tools

    wanted = _clean_names(enabled)
    if wanted:
        _save_platform_tools(cfg, "cli", wanted)
    elif isinstance(cfg.get("platform_toolsets"), dict):
        cfg["platform_toolsets"].pop("cli", None)
    save_config(cfg)


def _mcp_entry_enabled(entry: dict) -> bool:
    """The runtime's ``enabled`` reader; a legacy ``disabled: true`` (what older editors wrote,
    migrated by config v46) still reads as off."""
    from tools.mcp_tool_common import mcp_server_enabled
    return mcp_server_enabled(entry)


def _save_mcp_toggles(cfg, enabled, launch_mcp, save_config) -> None:
    """Write the ``enabled`` flag every runtime resolver reads (``enabled_mcp_server_names``,
    coding_context, oneshot); the legacy ``disabled`` key is dropped so old configs migrate."""
    from tools.mcp_tool_common import write_mcp_server_enabled
    wanted = _clean_names(enabled)
    mcp_cfg = cfg.get("mcp_servers") if isinstance(cfg.get("mcp_servers"), dict) else {}
    for srv in wanted:
        if not isinstance(mcp_cfg.get(srv), dict) and isinstance(launch_mcp.get(srv), dict):
            mcp_cfg[srv] = dict(launch_mcp[srv])
    for srv, entry in mcp_cfg.items():
        if isinstance(entry, dict):
            write_mcp_server_enabled(entry, srv in wanted)
    if mcp_cfg:
        cfg["mcp_servers"] = mcp_cfg
    save_config(cfg)


def _configure_cfg_sections(profile_dir, params, applied) -> None:
    """Apply ``disabled_skills`` / ``enabled_toolsets`` / ``enabled_mcp_servers`` (replace
    semantics; empty toolsets clears the pin). An undefined MCP server is copied from the LAUNCH
    catalog (unknown names skipped); credentials stay in .env/auth."""
    want_mcp = isinstance(params.get("enabled_mcp_servers"), list)
    launch_mcp = {}
    if want_mcp:  # launch catalog read BEFORE the home override flips config resolution
        # RAW: a copied entry keeps its ${VAR} refs instead of the launch profile's expanded secrets.
        load_launch = _lazy("hermes_cli.config", "read_user_config_raw")
        launch_mcp = _try(lambda: (load_launch() or {}).get("mcp_servers"), {})
        launch_mcp = launch_mcp if isinstance(launch_mcp, dict) else {}
    with _hermes_home_scope(profile_dir):
        from hermes_cli.config import load_config, save_config
        cfg = load_config() or {}
        if isinstance(params.get("disabled_skills"), list):
            try:
                from hermes_cli.skills_config import save_disabled_skills
                save_disabled_skills(cfg, _clean_names(params["disabled_skills"]))
                applied["skills"] = _landed(cfg, ("skills", "disabled"))
                cfg = load_config() or {}
            except Exception:
                applied["skills"] = False
        if isinstance(params.get("enabled_toolsets"), list):
            applied["toolsets"] = _best_effort(lambda: _save_toolset_pin(cfg, params["enabled_toolsets"], save_config)) \
                and _landed(cfg, ("platform_toolsets", "cli"))
        if want_mcp:
            mcp_cfg = load_config() or {}
            applied["mcp_servers"] = _best_effort(lambda: _save_mcp_toggles(
                mcp_cfg, params["enabled_mcp_servers"], launch_mcp, save_config)) \
                and _landed(mcp_cfg, ("mcp_servers",), _mcp_toggle_state)


def _mcp_toggle_state(servers):
    """Every toggle field ``_save_mcp_toggles`` writes, per server: ``enabled`` AND the legacy
    ``disabled`` it removes. A lock that kept ``disabled: true`` leaves the server off for the
    editor (and on for the runtime) although ``enabled`` landed, so it must not read as applied."""
    if not isinstance(servers, dict):
        return servers
    return {k: (e.get("enabled"), e.get("disabled", _ABSENT)) for k, e in servers.items() if isinstance(e, dict)}


_ABSENT = "<absent>"


def _landed(cfg: dict, path: tuple, key=lambda v: v) -> bool:
    """Whether the section a bulk ``save_config`` was asked to write now reads back as written.
    Under Remote Config a bulk save omits a locked key (with a stderr note) instead of raising, so
    an editor that reported success after the save would acknowledge an edit the plane refused.
    Compared on the user layer as read back (the effective doc in remote mode), absent == absent."""
    from hermes_cli.config import read_user_config_raw

    def at(doc):
        node = doc
        for seg in path:
            if not isinstance(node, dict) or seg not in node:
                return None
            node = node[seg]
        return node

    return key(at(read_user_config_raw() or {})) == key(at(cfg))


def _unlink_asset_files(assets_dir, asset) -> int:
    """Delete every ``<asset>.<ext>`` in ``assets_dir``; returns how many existed."""
    present = [t for t in (assets_dir / f"{asset}.{ext}" for ext in _ASSET_EXTS) if t.is_file()]
    return len([t.unlink() for t in present])


def register(server) -> None:
    bind_module(globals(), server)
