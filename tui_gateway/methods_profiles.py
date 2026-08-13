"""Profile JSON-RPC handlers — the ws twin of the dashboard's /api/profiles.

Motivation: desktop plugins reach the backend exclusively through the
generic ws JSON-RPC door (`host.request`).  Profile enumeration/creation
previously lived only on the dashboard REST router, which plugins cannot
reach, so anything "one chat per agent profile"-shaped (bot rosters,
profile pickers, team panes) was impossible to build as a plugin.  These
handlers delegate to the same `hermes_cli.profiles` primitives the REST
endpoints use.

Handlers are rebound onto server.py's globals at install time — see
method_ctx.py.  They may reference server.py module globals (`_ok`,
`_err`, `is_truthy_value`, ...) that are not imported here.
"""

from .method_ctx import HandlerRegistry

_registry = HandlerRegistry()
method = _registry.method


@method("profiles.list")
def _(rid, params: dict) -> dict:
    """List Hermes profiles (name, path, model, description, skill count).

    ``include_sessions`` (default true) additionally reports each profile's
    most recent conversation as ``last_session`` so a roster UI can paint
    per-agent previews without N follow-up calls.

    NOTE: helpers must be nested — install() rebinds this handler's
    __globals__ onto server.py, so module-level names here are invisible.
    """

    def _latest_profile_session_row(profile_path):
        """Most recent human-facing session in a profile's state.db, or None.

        Mirrors session.list's deny-list (drops ``tool`` sub-agent rows and
        ``kanban`` dispatcher workers).  Best-effort: any failure (missing
        state.db, locked db, older schema) degrades to None rather than
        failing the whole profiles.list call.
        """
        try:
            from pathlib import Path

            db_path = Path(profile_path) / "state.db"
            if not db_path.exists():
                return None
            from hermes_state import SessionDB

            deny = frozenset({"kanban", "tool"})
            db = SessionDB(db_path=db_path)
            try:
                for s in db.list_sessions_rich(
                    source=None, limit=20, order_by_last_active=True, compact_rows=True
                ):
                    if (s.get("source") or "").strip().lower() in deny:
                        continue
                    return {
                        "id": s["id"],
                        "title": s.get("title") or "",
                        "preview": s.get("preview") or "",
                        "started_at": s.get("started_at") or 0,
                        "last_active": s.get("last_active") or s.get("started_at") or 0,
                        "message_count": s.get("message_count") or 0,
                    }
            finally:
                try:
                    db.close()
                except Exception:
                    pass
        except Exception:
            return None
        return None

    try:
        from hermes_cli.profiles import list_profiles

        include_sessions = is_truthy_value(params.get("include_sessions", True))
        out = []
        for p in list_profiles():
            row = {
                "name": p.name,
                "path": str(p.path),
                "is_default": bool(p.is_default),
                "model": p.model,
                "provider": p.provider,
                "description": getattr(p, "description", "") or "",
                "skill_count": getattr(p, "skill_count", 0) or 0,
            }
            if include_sessions:
                row["last_session"] = _latest_profile_session_row(p.path)
            out.append(row)
        return _ok(rid, {"profiles": out})
    except Exception as e:
        return _err(rid, 5061, str(e))


@method("profiles.create")
def _(rid, params: dict) -> dict:
    """Create a profile — the ws twin of POST /api/profiles.

    Params: ``name`` (required, lowercase slug), ``description``,
    ``clone_from`` (source profile; omitted = fresh profile with bundled
    skills), ``clone_all``, ``no_skills``, ``soul`` (SOUL.md content),
    ``model`` + ``provider`` (optional model pin, best-effort).
    """
    name = str(params.get("name") or "").strip()
    if not name:
        return _err(rid, 4061, "name required")
    try:
        from hermes_cli import profiles as profiles_mod

        clone_from = str(params.get("clone_from") or "").strip() or None
        clone_all = is_truthy_value(params.get("clone_all", False))
        path = profiles_mod.create_profile(
            name=name,
            clone_from=clone_from,
            clone_all=clone_all,
            clone_config=bool(clone_from) and not clone_all,
            no_skills=is_truthy_value(params.get("no_skills", False)),
            description=str(params.get("description") or "").strip() or None,
        )
    except (ValueError, FileExistsError, FileNotFoundError) as e:
        return _err(rid, 4062, str(e))
    except Exception as e:
        return _err(rid, 5062, str(e))

    # Mirror the CLI/REST create flow: fresh profiles get the bundled
    # skills; safe alias wrapper. Both best-effort.
    try:
        if not clone_from:
            profiles_mod.seed_profile_skills(path, quiet=True)
    except Exception:
        pass
    try:
        if not profiles_mod.check_alias_collision(name):
            profiles_mod.create_wrapper_script(name)
    except Exception:
        pass

    soul = params.get("soul")
    soul_written = False
    if isinstance(soul, str) and soul.strip():
        try:
            (path / "SOUL.md").write_text(soul, encoding="utf-8")
            soul_written = True
        except Exception:
            pass

    model = str(params.get("model") or "").strip()
    provider = str(params.get("provider") or "").strip()
    model_set = False
    if model and provider:
        try:
            from hermes_cli.web_routers.profiles import _write_profile_model

            _write_profile_model(path, provider, model)
            model_set = True
        except Exception:
            pass

    return _ok(
        rid,
        {
            "ok": True,
            "name": name,
            "path": str(path),
            "soul_written": soul_written,
            "model_set": model_set,
        },
    )


def register(server) -> None:
    _registry.install(server)
