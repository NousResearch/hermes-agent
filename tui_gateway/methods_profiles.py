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
    ``model`` + ``provider`` (optional model pin, best-effort), and
    ``mirror_credentials`` (default true) — copy the launch profile's
    ``.env`` and ``auth.json`` into the new profile, and inherit its
    model.provider/model.default when no explicit pin is given.

    Credential mirroring exists because ``create_profile()`` deliberately
    seeds a comment-only ``.env`` and never copies ``auth.json`` (OAuth
    tokens / credential pools), so a profile created headlessly from a
    plugin was born with NO inference provider — the first message failed
    with "No inference provider configured" and there is no interactive
    ``hermes setup`` in that flow to recover. A profile spawned as an
    always-available teammate must be able to think out of the box; callers
    that want an isolated/credential-free profile pass
    ``mirror_credentials: false``.
    """

    def _has_real_env_content(env_path) -> bool:
        """True when .env has any non-comment, non-blank line."""
        try:
            for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    return True
        except Exception:
            pass
        return False

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

    # Credential + provider mirroring (default ON): a headless-created
    # profile must be able to run a first turn. Copy the launch profile's
    # .env (only over the seeded comment-only stub — never clobber real
    # secrets a clone brought along) and auth.json (only when absent), then
    # inherit model.provider/model.default unless the caller pinned a model.
    mirrored = {"env": False, "auth": False, "model_inherited": False}
    if is_truthy_value(params.get("mirror_credentials", True)):
        import shutil

        from hermes_constants import get_hermes_home

        launch_home = get_hermes_home()
        try:
            src_env = launch_home / ".env"
            dst_env = path / ".env"
            if src_env.is_file() and _has_real_env_content(src_env) and not _has_real_env_content(dst_env):
                shutil.copy2(src_env, dst_env)
                try:
                    os.chmod(str(dst_env), 0o600)
                except OSError:
                    pass
                mirrored["env"] = True
        except Exception:
            pass
        try:
            src_auth = launch_home / "auth.json"
            dst_auth = path / "auth.json"
            if src_auth.is_file() and not dst_auth.exists():
                shutil.copy2(src_auth, dst_auth)
                try:
                    os.chmod(str(dst_auth), 0o600)
                except OSError:
                    pass
                mirrored["auth"] = True
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
    elif is_truthy_value(params.get("mirror_credentials", True)) and not (path / "config.yaml").exists():
        # No explicit pin and no cloned config: inherit the launch profile's
        # provider+model so the first turn resolves. Same writer as the pin.
        try:
            from hermes_cli.config import load_config_readonly
            from hermes_cli.web_routers.profiles import _write_profile_model

            cfg = load_config_readonly() or {}
            model_cfg = cfg.get("model") or {}
            inherited_provider = str(model_cfg.get("provider") or "")
            inherited_model = str(model_cfg.get("default") or "")
            if inherited_provider and inherited_model:
                _write_profile_model(path, inherited_provider, inherited_model)
                mirrored["model_inherited"] = True
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
            "mirrored": mirrored,
        },
    )


@method("profiles.describe")
def _(rid, params: dict) -> dict:
    """Full configuration snapshot of one profile, for an editor UI.

    Params: ``name`` (required). Result:
    ``{name, description, soul, model: {provider, default}, skills:
    [{name, enabled}], toolsets: [{name, description, tool_count, enabled}]}``

    Skill enablement mirrors the disabled-list model (installed = enabled
    unless in ``skills.disabled``). Toolset enablement reports the profile's
    ``tools.enabled_toolsets`` pin, or every toolset enabled when unpinned.
    All reads are scoped to the profile via the HERMES_HOME override.
    """
    name = str(params.get("name") or "").strip()
    if not name:
        return _err(rid, 4063, "name required")
    try:
        from pathlib import Path

        from hermes_cli.profiles import get_profile_dir
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override

        profile_dir = Path(get_profile_dir(name))
        if not profile_dir.is_dir():
            return _err(rid, 4064, f"profile '{name}' not found")

        token = set_hermes_home_override(str(profile_dir))
        try:
            from hermes_cli.config import load_config
            from hermes_cli.skills_config import get_disabled_skills

            cfg = load_config() or {}
            disabled = {s.lower() for s in get_disabled_skills(cfg)}

            installed = []
            skills_root = profile_dir / "skills"
            if skills_root.is_dir():
                for md in sorted(skills_root.rglob("SKILL.md")):
                    skill_name = md.parent.name
                    installed.append(
                        {"name": skill_name, "enabled": skill_name.lower() not in disabled}
                    )

            from toolsets import get_all_toolsets, get_toolset_info

            tools_cfg = cfg.get("tools") if isinstance(cfg.get("tools"), dict) else {}
            pinned = tools_cfg.get("enabled_toolsets")
            pinned_set = (
                {str(t).strip() for t in pinned if str(t).strip()}
                if isinstance(pinned, list)
                else None
            )
            toolsets_out = []
            for ts_name in sorted(get_all_toolsets().keys()):
                info = get_toolset_info(ts_name) or {}
                toolsets_out.append(
                    {
                        "name": ts_name,
                        "description": info.get("description") or "",
                        "tool_count": len(info.get("tools") or []),
                        "enabled": True if pinned_set is None else ts_name in pinned_set,
                    }
                )

            soul_path = profile_dir / "SOUL.md"
            soul = ""
            try:
                if soul_path.is_file():
                    soul = soul_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass

            model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}

            description = ""
            try:
                from hermes_cli.profiles import read_profile_meta

                description = str(read_profile_meta(profile_dir).get("description") or "")
            except Exception:
                pass

            return _ok(
                rid,
                {
                    "name": name,
                    "description": description,
                    "soul": soul,
                    "model": {
                        "provider": str(model_cfg.get("provider") or ""),
                        "default": str(model_cfg.get("default") or ""),
                    },
                    "skills": installed,
                    "toolsets": toolsets_out,
                    "toolsets_pinned": pinned_set is not None,
                },
            )
        finally:
            reset_hermes_home_override(token)
    except Exception as e:
        return _err(rid, 5063, str(e))


@method("profiles.configure")
def _(rid, params: dict) -> dict:
    """Apply configuration changes to a profile (editor Save).

    Params: ``name`` (required) plus any of:
    ``description`` (str), ``soul`` (str, full SOUL.md replacement),
    ``model`` + ``provider`` (both required together),
    ``disabled_skills`` (list[str], replace semantics),
    ``enabled_toolsets`` (list[str], replace semantics; empty list clears
    the pin so every toolset is enabled again).

    Each section is applied independently and best-effort; the result
    reports per-section success so a UI can surface partial failures.
    """
    name = str(params.get("name") or "").strip()
    if not name:
        return _err(rid, 4063, "name required")
    try:
        from pathlib import Path

        from hermes_cli.profiles import get_profile_dir
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override

        profile_dir = Path(get_profile_dir(name))
        if not profile_dir.is_dir():
            return _err(rid, 4064, f"profile '{name}' not found")

        applied = {}

        if isinstance(params.get("soul"), str):
            try:
                (profile_dir / "SOUL.md").write_text(params["soul"], encoding="utf-8")
                applied["soul"] = True
            except Exception:
                applied["soul"] = False

        if isinstance(params.get("description"), str):
            try:
                from hermes_cli.profiles import write_profile_meta

                write_profile_meta(
                    profile_dir,
                    description=params["description"].strip(),
                    description_auto=False,
                )
                applied["description"] = True
            except Exception:
                applied["description"] = False

        model = str(params.get("model") or "").strip()
        provider = str(params.get("provider") or "").strip()
        if model and provider:
            try:
                from hermes_cli.web_routers.profiles import _write_profile_model

                _write_profile_model(profile_dir, provider, model)
                applied["model"] = True
            except Exception:
                applied["model"] = False

        needs_cfg = isinstance(params.get("disabled_skills"), list) or isinstance(
            params.get("enabled_toolsets"), list
        )
        if needs_cfg:
            token = set_hermes_home_override(str(profile_dir))
            try:
                from hermes_cli.config import load_config, save_config

                cfg = load_config() or {}

                if isinstance(params.get("disabled_skills"), list):
                    try:
                        from hermes_cli.skills_config import save_disabled_skills

                        wanted = {
                            str(s).strip()
                            for s in params["disabled_skills"]
                            if str(s).strip()
                        }
                        save_disabled_skills(cfg, wanted)
                        applied["skills"] = True
                        cfg = load_config() or {}
                    except Exception:
                        applied["skills"] = False

                if isinstance(params.get("enabled_toolsets"), list):
                    try:
                        wanted = [str(t).strip() for t in params["enabled_toolsets"] if str(t).strip()]
                        tools_cfg = cfg.get("tools") if isinstance(cfg.get("tools"), dict) else {}
                        if wanted:
                            tools_cfg["enabled_toolsets"] = sorted(set(wanted))
                        else:
                            tools_cfg.pop("enabled_toolsets", None)
                        cfg["tools"] = tools_cfg
                        save_config(cfg)
                        applied["toolsets"] = True
                    except Exception:
                        applied["toolsets"] = False
            finally:
                reset_hermes_home_override(token)

        return _ok(rid, {"ok": all(applied.values()) if applied else True, "applied": applied})
    except Exception as e:
        return _err(rid, 5064, str(e))


def register(server) -> None:
    _registry.install(server)
