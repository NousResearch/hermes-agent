"""Bot Marketplace JSON-RPC handlers."""

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped


@method("bots.catalog")
@_profile_scoped
def _(rid, params: dict) -> dict:
    try:
        from hermes_cli.bot_catalog import load_bot_catalog, removed_bots

        return _ok(rid, {
            "entries": [entry.model_dump(mode="json") for entry in load_bot_catalog()],
            "removed": [entry.model_dump(mode="json") for entry in removed_bots()],
        })
    except Exception as exc:
        return _err(rid, 5070, str(exc))


@method("bots.installed")
@_profile_scoped
def _(rid, params: dict) -> dict:
    try:
        from hermes_cli.bot_setup import installed_bot_inventory

        return _ok(rid, {"bots": installed_bot_inventory()})
    except Exception as exc:
        return _err(rid, 5077, str(exc))


@method("bots.install")
@_profile_scoped
def _(rid, params: dict) -> dict:
    try:
        from hermes_cli.bot_catalog import BotCatalogError, resolve_bot_catalog_entry

        entry = resolve_bot_catalog_entry(str(params.get("catalog_name") or ""))
    except BotCatalogError as exc:
        return _err(rid, 4071, str(exc))
    except Exception as exc:
        return _err(rid, 5071, str(exc))

    try:
        from hermes_cli.bot_profile_install import BotProfileInstallSpec, install_bot_profile

        spec = BotProfileInstallSpec(
            catalog_name=entry.name,
            name=str(params.get("name") or ""),
            source_profile=str(params.get("source_profile") or ""),
            credentials=str(params.get("credentials") or "none"),
        )
        receipt = install_bot_profile(entry, spec)
        return _ok(rid, {
            "ok": True,
            "committed": receipt.committed,
            "name": receipt.name,
            "path": str(receipt.path),
            "catalog_name": receipt.catalog_name,
            "catalog_version": receipt.catalog_version,
            "source_profile": receipt.source_profile,
            "copied_credentials": list(receipt.copied_credentials),
            "oauth_setup_required": list(receipt.oauth_setup_required),
            "setup_state": receipt.setup_state,
            "setup_requirements": list(receipt.setup_requirements),
            "post_publish_warnings": list(receipt.post_publish_warnings),
        })
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        return _err(rid, 4072, str(exc))
    except Exception as exc:
        return _err(rid, 5072, str(exc))


def _runtime_readiness(params: dict) -> dict:
    response = _methods["setup.runtime_check"](None, {"profile": params.get("profile")})
    if not isinstance(response, dict):
        return {"ok": False, "error": "runtime readiness probe failed"}
    if isinstance(response.get("result"), dict):
        return response["result"]
    raw_error = response.get("error")
    error: dict = raw_error if isinstance(raw_error, dict) else {}
    return {"ok": False, "error": str(error.get("message") or "runtime readiness probe failed")}


def _bot_status_payload(params: dict) -> dict:
    from hermes_cli.bot_setup import bot_setup_status, installed_catalog_entry

    entry = installed_catalog_entry()
    payload = bot_setup_status(entry, runtime=_runtime_readiness(params))
    payload["profile"] = str(params.get("profile") or "default")
    return payload


@method("bots.status")
@_profile_scoped
def _(rid, params: dict) -> dict:
    try:
        return _ok(rid, _bot_status_payload(params))
    except (ValueError, FileNotFoundError) as exc:
        return _err(rid, 4073, str(exc))
    except Exception as exc:
        return _err(rid, 5073, str(exc))


@method("bots.setup")
@_profile_scoped
def _(rid, params: dict) -> dict:
    try:
        return _ok(rid, _bot_status_payload(params))
    except (ValueError, FileNotFoundError) as exc:
        return _err(rid, 4073, str(exc))
    except Exception as exc:
        return _err(rid, 5073, str(exc))


@method("bots.routines.list")
@_profile_scoped
def _(rid, params: dict) -> dict:
    try:
        from hermes_cli.bot_routines import list_bot_routines
        from hermes_cli.bot_setup import installed_catalog_entry

        return _ok(rid, {"routines": list_bot_routines(installed_catalog_entry())})
    except (ValueError, FileNotFoundError) as exc:
        return _err(rid, 4074, str(exc))
    except Exception as exc:
        return _err(rid, 5074, str(exc))


@method("bots.routines.activate")
@_profile_scoped
def _(rid, params: dict) -> dict:
    try:
        from hermes_cli.bot_routines import activate_bot_routine
        from hermes_cli.bot_setup import installed_catalog_entry

        entry = installed_catalog_entry()
        status = _bot_status_payload(params)
        result = activate_bot_routine(
            entry,
            str(params.get("routine_id") or ""),
            schedule=str(params.get("schedule") or ""),
            timezone_name=str(params.get("timezone") or ""),
            destination=str(params.get("destination") or ""),
            setup_ready=status["setup_state"] == "ready",
        )
        return _ok(rid, result)
    except (ValueError, FileNotFoundError, PermissionError) as exc:
        return _err(rid, 4075, str(exc))
    except Exception as exc:
        return _err(rid, 5075, str(exc))


@method("bots.routines.pause")
@_profile_scoped
def _(rid, params: dict) -> dict:
    try:
        from hermes_cli.bot_routines import pause_bot_routine
        from hermes_cli.bot_setup import installed_catalog_entry

        return _ok(rid, pause_bot_routine(installed_catalog_entry(), str(params.get("routine_id") or "")))
    except (ValueError, FileNotFoundError) as exc:
        return _err(rid, 4076, str(exc))
    except Exception as exc:
        return _err(rid, 5076, str(exc))


def register(server) -> None:
    bind_module(globals(), server, skip=("_",))
