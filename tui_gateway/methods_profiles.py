"""Profile JSON-RPC handlers — the ws twin of the dashboard's /api/profiles (desktop plugins
only have the ws door), on the same `hermes_cli.profiles` primitives. Bodies are rebound onto
server.py's globals (method_ctx.bind_module) and use them bare; module-level names are published
onto server.py, so they must not collide with its globals.
"""

import contextlib

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


def _pin_profile_model(profile_dir, provider, model) -> None:
    _lazy("hermes_cli.web_routers.profiles", "_write_profile_model")(profile_dir, provider, model)


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


@contextlib.contextmanager
def _hermes_home_scope(path):
    """Scope config/auth resolution to ``path`` for the block."""
    token = set_hermes_home_override(str(path))
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _resolve_profile(rid, params):
    """``(name, profile_dir, err)``; err = 4063 (name required) / 4064 (not found) response."""
    name = str(params.get("name") or "").strip()
    if not name:
        return name, None, _err(rid, 4063, "name required")
    from hermes_cli.profiles import get_profile_dir
    profile_dir = Path(get_profile_dir(name))
    if not profile_dir.is_dir():
        return name, None, _err(rid, 4064, f"profile '{name}' not found")
    return name, profile_dir, None


def _read_profile_yaml(profile_dir) -> dict:
    """profile.yaml as a mapping; ``{}`` when missing, unreadable, unparseable, or not a mapping."""
    def load():
        import yaml
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
            "message_count": tip_row.get("message_count") or 0}
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
        for s in db.list_sessions_rich(source=None, limit=20, order_by_last_active=True, compact_rows=True):
            title, last_active = s.get("title") or "", s.get("last_active") or s.get("started_at") or 0
            if _denied_source(s):
                if worker is None:
                    worker = {"id": s["id"], "source": (s.get("source") or "").strip().lower(),
                              "title": title, "last_active": last_active}
            elif human is None:  # rosters want "where the conversation IS": prefer the newest text
                human = {"id": s["id"], "title": title,
                         "preview": _latest_message_preview(db, s["id"]) or s.get("preview") or "",
                         "started_at": s.get("started_at") or 0, "last_active": last_active,
                         "message_count": s.get("message_count") or 0}
            if human is not None and worker is not None:
                break
        return human, worker
    except Exception:
        return None, None


def _profile_session_fields(row, profile_path):
    """Attach last_session / worker_session / canonical_session to a roster row. The DB is a
    read-only attach (a writable ``SessionDB()`` waits up to 20s for the write lock + runs DDL
    and stalled the 5s roster poll); no/unreadable DB -> every field None (the readers swallow)."""
    db_path = Path(profile_path) / "state.db"
    db = None
    if _try(db_path.exists, False):
        db = _try(lambda: _lazy("hermes_state", "SessionDB")(db_path=db_path, read_only=True), None)
    try:
        row["last_session"], row["worker_session"] = _latest_profile_session_rows(db)
        # Resolved server-side on every listing so no client carries a session pointer.
        row["canonical_session"] = _canonical_session_row(db, profile_path)
    finally:
        if db is not None:
            _best_effort(db.close)


def _profile_ui_meta_fields(row: dict, profile_dir) -> None:
    """Attach ``ui_meta`` / ``ui_meta_revisions`` / ``has_avatar`` from profile.yaml + assets.
    ``ui_meta_revisions`` is always present: it feature-detects gateway-owned CAS for a new profile."""
    raw_meta = _read_profile_yaml(profile_dir)
    ui_meta, revisions = raw_meta.get("ui_meta"), raw_meta.get("_ui_meta_revisions")
    # Key order is wire-visible: ui_meta_revisions precedes ui_meta.
    row["ui_meta_revisions"] = _try(lambda: _clean_revisions(revisions), {}) if isinstance(revisions, dict) else {}
    if isinstance(ui_meta, dict) and ui_meta:
        row["ui_meta"] = ui_meta
    # Cheap existence flag so rosters skip a get_asset probe per paint.
    row["has_avatar"] = _try(lambda: any((profile_dir / "assets" / f"avatar.{e}").is_file() for e in _ASSET_EXTS), False)


@_profile_handler("profiles.list", 5061)
def _(rid, params: dict) -> dict:
    """List Hermes profiles. ``include_sessions`` (default true) adds ``last_session`` /
    ``worker_session`` / ``canonical_session`` so a roster paints previews without N calls."""
    from hermes_cli.profiles import list_profiles
    include_sessions = is_truthy_value(params.get("include_sessions", True))
    out = []
    for p in list_profiles():
        row = {"name": p.name, "path": str(p.path), "is_default": bool(p.is_default), "model": p.model,
               "provider": p.provider, "description": p.description or "",
               "display_name": p.display_name or "", "skill_count": p.skill_count or 0}
        if include_sessions:
            _profile_session_fields(row, p.path)
        _profile_ui_meta_fields(row, Path(str(p.path)))
        out.append(row)
    # bot_mode_protocol: this backend injects the Bot Mode teammate-messaging protocol into every
    # session, so clients must not append it to SOUL.md.
    return _ok(rid, {"profiles": out, "bot_mode_protocol": True})


def _mirror_secret(path, launch_home, name: str, wanted) -> bool:
    """Copy the launch ``name`` file into the profile (0600) when it exists and ``wanted(src, dst)``."""
    src, dst = launch_home / name, path / name
    if not (src.is_file() and wanted(src, dst)):
        return False
    import shutil
    shutil.copy2(src, dst)
    with contextlib.suppress(OSError):
        os.chmod(str(dst), 0o600)
    return True


def _env_has_content(env_path) -> bool:
    lines = env_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return any(s and not s.startswith("#") for s in map(str.strip, lines))


def _mirror_voice_sections(path) -> bool:
    """Copy stt/tts/voice sections from the launch profile (a fresh profile has only ``model``,
    so voice fell back to defaults); True if written."""
    try:
        from hermes_cli.config import load_config_readonly, read_user_config_raw, save_config
        src_cfg = load_config_readonly() or {}
        sections = {k: src_cfg[k] for k in ("stt", "tts", "voice") if src_cfg.get(k)}
        if not sections:
            return False
        with _hermes_home_scope(path):
            # RAW file: load_config() merges DEFAULT_CONFIG (every section would look present).
            dst_cfg = read_user_config_raw() or {}
            missing = {k: v for k, v in sections.items() if k not in dst_cfg}
            if missing:
                save_config({**dst_cfg, **missing})
        return bool(missing)
    except Exception:
        return False


def _inherit_launch_model(path) -> bool:
    """Inherit launch model.provider/default when the new profile has none. Gate on the MODEL
    SECTION, not config.yaml existing: voice mirroring creates the file first."""
    # Gate on the MODEL SECTION being absent, not on config.yaml existing — earlier mirroring steps (voice
    # sections, #85755) legitimately create the file first, and a file-existence gate silently skipped
    # inheritance for every non-clone bot ("No inference provider configured" on first message, tester
    # report). Clones bring their own model section and stay untouched.
    from hermes_cli.config import load_config_readonly, read_user_config_raw
    with _hermes_home_scope(path):
        dst_model = (read_user_config_raw() or {}).get("model") or {}
    if dst_model.get("provider") and dst_model.get("default"):
        return False
    model_cfg = (load_config_readonly() or {}).get("model") or {}
    if not (model_cfg.get("provider") and model_cfg.get("default")):
        return False
    _pin_profile_model(path, str(model_cfg["provider"]), str(model_cfg["default"]))
    return True


def _mirror_launch_credentials(path, params: dict) -> dict:
    """Copy launch .env / auth.json / voice sections into a new profile (best-effort per item).
    ``share_auth`` reports ``auth: "shared"`` and skips the auth copy; ``mirror_credentials``
    false skips everything. ``model_inherited`` is filled in by the caller."""
    share_auth = is_truthy_value(params.get("share_auth", False))
    mirrored = {"env": False, "auth": "shared" if share_auth else False, "model_inherited": False,
                "voice": False}
    if not is_truthy_value(params.get("mirror_credentials", True)):
        return mirrored
    launch_home = get_hermes_home()
    # .env: only over the seeded comment-only stub (never a clone's secrets).
    mirrored["env"] = _try(lambda: _mirror_secret(path, launch_home, ".env", lambda src, dst: (
        _env_has_content(src) and not _try(lambda: _env_has_content(dst), False))), False)
    if not share_auth:  # a copy forks token state: the first refresh in either store strands the other
        mirrored["auth"] = _try(lambda: _mirror_secret(path, launch_home, "auth.json",
                                                       lambda src, dst: not dst.exists()), False)
        if mirrored["auth"]:
            # Drop single-use OAuth grants (first refresh strands every sibling); they read from the
            # root grant via the pool fallback. API keys stay.
            _best_effort(lambda: _lazy("hermes_cli.auth", "strip_cloned_single_use_oauth_grants")(path))
    mirrored["voice"] = _mirror_voice_sections(path)
    return mirrored


@method("profiles.create")
def _(rid, params: dict) -> dict:
    """Create a profile (ws twin of POST /api/profiles). Params: ``name``, ``description``,
    ``clone_from`` (omitted = fresh + bundled skills), ``clone_all``, ``no_skills``, ``soul``,
    ``model`` + ``provider``, ``share_auth``, ``mirror_credentials`` (default true: a bare
    ``create_profile()`` seeds a comment-only .env and no auth.json = NO provider headless)."""
    name = str(params.get("name") or "").strip()
    if not name:
        return _err(rid, 4061, "name required")
    try:
        from hermes_cli import profiles as profiles_mod
        clone_from = str(params.get("clone_from") or "").strip() or None
        clone_all = is_truthy_value(params.get("clone_all", False))
        path = profiles_mod.create_profile(
            name=name, clone_from=clone_from, clone_all=clone_all,
            clone_config=bool(clone_from) and not clone_all,
            no_skills=is_truthy_value(params.get("no_skills", False)),
            description=str(params.get("description") or "").strip() or None)
    except (ValueError, FileExistsError, FileNotFoundError) as e:
        return _err(rid, 4062, str(e))
    except Exception as e:
        return _err(rid, 5062, str(e))
    # CLI/REST create flow: bundled skills for fresh profiles, then the alias wrapper.
    if not clone_from:
        _best_effort(lambda: profiles_mod.seed_profile_skills(path, quiet=True))
    _best_effort(lambda: profiles_mod.check_alias_collision(name) or profiles_mod.create_wrapper_script(name))
    soul = params.get("soul")
    soul_written = isinstance(soul, str) and bool(soul.strip()) and _best_effort(
        lambda: (path / "SOUL.md").write_text(soul, encoding="utf-8"))
    mirrored = _mirror_launch_credentials(path, params)
    model, provider = _model_provider_params(params)
    model_set = False
    if model and provider:
        model_set = _best_effort(lambda: _pin_profile_model(path, provider, model))
    elif is_truthy_value(params.get("mirror_credentials", True)):
        mirrored["model_inherited"] = _try(lambda: _inherit_launch_model(path), False)
    return _ok(rid, {"ok": True, "name": name, "path": str(path), "soul_written": soul_written,
                     "model_set": model_set, "mirrored": mirrored})


def _describe_toolsets(cfg):
    """``(toolsets, pinned_set)`` as the `hermes tools` checklist presents them (the raw registry
    leaks platform composites and reports everything enabled without a pin)."""
    from hermes_cli.tools_config import (
        _get_effective_configurable_toolsets, _get_platform_tools, _toolset_allowed_for_platform)
    from toolsets import resolve_toolset
    pinned = (cfg.get("tools") if isinstance(cfg.get("tools"), dict) else {}).get("enabled_toolsets")
    pinned_set = _clean_names(pinned) if isinstance(pinned, list) else None
    platform_enabled = _try(lambda: set(_get_platform_tools(cfg, "cli", include_default_mcp_servers=False)), set())
    default_off = _try(lambda: _lazy("hermes_cli.tools_config", "_DEFAULT_OFF_TOOLSETS"), set())
    toolsets_out = []
    for ts_name, ts_label, ts_desc in _get_effective_configurable_toolsets():
        enabled = ts_name in (pinned_set if pinned_set is not None else platform_enabled)
        # Default-off integrations (+ opt-in yuanbao) are noise unless already enabled.
        if not _toolset_allowed_for_platform(ts_name, "cli") or (
                (ts_name in default_off or ts_name == "yuanbao") and not enabled):
            continue
        toolsets_out.append({"name": ts_name, "label": ts_label, "description": ts_desc or "",
                             "tool_count": _try(lambda: len(set(resolve_toolset(ts_name))), 0),
                             "enabled": enabled})
    return toolsets_out, pinned_set


@_profile_handler("profiles.describe", 5063)
def _(rid, params: dict) -> dict:
    """Editor snapshot; installed skills are enabled unless in ``skills.disabled``; ``mcp_servers``
    is ``[{name, enabled, transport}]`` (best-effort)."""
    name, profile_dir, err = _resolve_profile(rid, params)
    if err is not None:
        return err
    with _hermes_home_scope(profile_dir):
        from hermes_cli.config import load_config
        from hermes_cli.skills_config import get_disabled_skills
        cfg = load_config() or {}
        disabled = {s.lower() for s in get_disabled_skills(cfg)}
        skills_root = profile_dir / "skills"
        installed = [
            {"name": md.parent.name, "enabled": md.parent.name.lower() not in disabled}
            for md in (sorted(skills_root.rglob("SKILL.md")) if skills_root.is_dir() else ())]
        toolsets_out, pinned_set = _describe_toolsets(cfg)
        soul_path = profile_dir / "SOUL.md"
        soul = _try(lambda: soul_path.read_text(encoding="utf-8", errors="replace") if soul_path.is_file() else "", "")
        mcp_cfg = cfg.get("mcp_servers")
        mcp_out = _try(lambda: [
            {"name": str(srv_name), "enabled": not is_truthy_value(entry.get("disabled", False)),
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
    except Exception:
        applied["ui_meta"] = False


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
        confirm_message = _try(lambda: getattr(warn(model, provider=provider or None), "message", None), None)
    if confirm_message is None:
        applied["model"] = _best_effort(lambda: _pin_profile_model(profile_dir, provider, model))
    return confirm_message


def _clean_names(values) -> set:
    return {str(v).strip() for v in values if str(v).strip()}


def _save_toolset_pin(cfg, enabled, save_config) -> None:
    wanted = sorted(_clean_names(enabled))
    tools_cfg = cfg.get("tools") if isinstance(cfg.get("tools"), dict) else {}
    if wanted:
        tools_cfg["enabled_toolsets"] = wanted
    else:
        tools_cfg.pop("enabled_toolsets", None)
    cfg["tools"] = tools_cfg
    save_config(cfg)


def _save_mcp_toggles(cfg, enabled, launch_mcp, save_config) -> None:
    wanted = _clean_names(enabled)
    mcp_cfg = cfg.get("mcp_servers") if isinstance(cfg.get("mcp_servers"), dict) else {}
    for srv in wanted:
        if not isinstance(mcp_cfg.get(srv), dict) and isinstance(launch_mcp.get(srv), dict):
            mcp_cfg[srv] = dict(launch_mcp[srv])
        if isinstance(mcp_cfg.get(srv), dict):
            mcp_cfg[srv].pop("disabled", None)
    for srv, entry in mcp_cfg.items():
        if srv not in wanted and isinstance(entry, dict):
            entry["disabled"] = True
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
        load_launch = _lazy("hermes_cli.config", "load_config_readonly")
        launch_mcp = _try(lambda: (load_launch() or {}).get("mcp_servers"), {})
        launch_mcp = launch_mcp if isinstance(launch_mcp, dict) else {}
    with _hermes_home_scope(profile_dir):
        from hermes_cli.config import load_config, save_config
        cfg = load_config() or {}
        if isinstance(params.get("disabled_skills"), list):
            try:
                from hermes_cli.skills_config import save_disabled_skills
                save_disabled_skills(cfg, _clean_names(params["disabled_skills"]))
                applied["skills"] = True
                cfg = load_config() or {}
            except Exception:
                applied["skills"] = False
        if isinstance(params.get("enabled_toolsets"), list):
            applied["toolsets"] = _best_effort(lambda: _save_toolset_pin(cfg, params["enabled_toolsets"], save_config))
        if want_mcp:
            applied["mcp_servers"] = _best_effort(lambda: _save_mcp_toggles(
                load_config() or {}, params["enabled_mcp_servers"], launch_mcp, save_config))


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


def _unlink_asset_files(assets_dir, asset) -> int:
    """Delete every ``<asset>.<ext>`` in ``assets_dir``; returns how many existed."""
    present = [t for t in (assets_dir / f"{asset}.{ext}" for ext in _ASSET_EXTS) if t.is_file()]
    return len([t.unlink() for t in present])


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


def register(server) -> None:
    bind_module(globals(), server)
