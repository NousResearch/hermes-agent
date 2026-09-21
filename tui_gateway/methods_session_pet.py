"""Pet mascot RPC handlers: sprite payloads, gallery, generation and progress events.

Bodies are rebound onto server.py's globals at install time (method_ctx.py), so they use server
helpers (``_ok``, ``_err``, ``_emit``, ``_pet_active_selection``, ...) bare; module-level helpers
are published onto server.py the same way (tests monkeypatching ``server.X`` still intercept).
"""

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped


def _pet_display_cfg() -> dict:
    """``display.pet`` config block, ``{}`` when config is unreadable."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        display = cfg.get("display", {}) if isinstance(cfg.get("display"), dict) else {}
        return display.get("pet", {}) if isinstance(display.get("pet"), dict) else {}
    except Exception:
        return {}


def _pet_emit(event: str, payload: dict, what: str) -> None:
    """Best-effort progress emit: a transport hiccup must never abort generation."""
    try:
        _emit(event, "", payload)
    except Exception as exc:  # noqa: BLE001
        logger.debug("%s emit failed: %s", what, exc)


def _pet_gen_abort(rid, token: str, code: int, message: str) -> dict:
    """Release the cancel arm for ``token`` and return ``_err``."""
    _pet_cancel_release(token)
    return _err(rid, code, message)


def _pet_method(name: str, *, fail_open=None, slug: bool = False, scoped: bool = True):
    """``@method`` (+ ``@_profile_scoped`` unless ``scoped=False``) whose exceptions never break the surface: logged
    at debug, then ``fail_open`` (payload or ``params -> payload``) or ``_err(5031)``. ``slug``: 3rd arg (4004)."""
    def deco(fn):
        def handler(rid, params: dict) -> dict:
            try:
                if slug and not (value := _str_param(params, "slug")):
                    return _err(rid, 4004, "missing slug")
                return fn(rid, params, value) if slug else fn(rid, params)
            except Exception as exc:  # noqa: BLE001 - cosmetic surface
                logger.debug("%s failed: %s", name, exc)
                if fail_open is not None:
                    return _ok(rid, fail_open(params) if callable(fail_open) else dict(fail_open))
                return _err(rid, 5031, f"{name} failed: {exc}")
        return method(name)(_profile_scoped(handler) if scoped else handler)
    return deco


def _active_pet():
    """``(pet, scale)`` when the pet display is enabled and the pet exists, else None."""
    enabled, pet, scale = _pet_active_selection()
    return None if not enabled or pet is None or not pet.exists else (pet, scale)


# ── pet ──────────────────────────────────────────────────────────────
_PET_OFF = {"enabled": False}


@_pet_method("pet.info", fail_open=_PET_OFF)
def _(rid, params: dict) -> dict:
    """Active pet for sprite renderers: spritesheet (base64) + frame geometry + state-row taxonomy."""
    if (active := _active_pet()) is None:
        return _ok(rid, {"enabled": False})
    pet, scale = active
    payload = {"enabled": True, **_pet_sprite_payload(pet, scale=scale)}
    # Send-once for the multi-MB sheet: same revision → metadata only.
    if (known := str(params.get("knownRevision", "") or "")) and known == payload.get("spritesheetRevision"):
        # Send-once semantics for the multi-MB spritesheet (#54730): a caller that already holds the sheet
        # passes the revision it has, and an unchanged sheet comes back as metadata only
        # (spritesheetUnchanged).
        payload.pop("spritesheetBase64", None)
        payload["spritesheetUnchanged"] = True
    return _ok(rid, payload)


@_pet_method("pet.info.meta", fail_open=_PET_OFF)
def _(rid, params: dict) -> dict:
    """Cheap active-pet metadata used to avoid full payload refreshes."""
    if (active := _active_pet()) is None:
        return _ok(rid, {"enabled": False})
    pet, scale = active
    return _ok(rid, {"enabled": True, "slug": pet.slug, "displayName": pet.display_name, "scale": scale,
                     "spritesheetRevision": _pet_sheet_revision(pet.spritesheet)})


def _pet_kitty_cells(pet, pet_cfg: dict, state: str, scale: float) -> dict | None:
    """kitty payload for a TTY that speaks it (dashboard PTY falls through); only kitty is grid-safe in Ink."""
    from agent.pet import constants, render
    from agent.pet.render import PetRenderer
    configured = str(pet_cfg.get("render_mode", "auto") or "auto").lower()
    if (render.detect_terminal_graphics() if configured in ("", "auto") else configured) != "kitty":
        return None
    image_id = render.kitty_image_id(pet.slug)
    # kitty sizes from scaled pixels, so unicode_cols is moot here.
    payload = PetRenderer(str(pet.spritesheet), mode="kitty", scale=scale).kitty_payload(state, image_id=image_id)
    if not payload:
        return None
    return {"graphics": "kitty", "imageId": image_id, "color": render.kitty_color_hex(image_id),
            "cols": payload["cols"], "rows": payload["rows"], "placeholder": payload["placeholder"],
            "frames": payload["frames"], "frameMs": constants.LOOP_MS / max(1, len(payload["frames"]) or 1),
            "scale": scale}


@_pet_method("pet.cells", fail_open=_PET_OFF)
def _(rid, params: dict) -> dict:
    """Half-block cell frames (``[tr,tg,tb,ta, br,bg,bb,ba]``) for one pet ``state``; ``cols``, ``graphics``."""
    from agent.pet import constants, store
    from agent.pet.render import PetRenderer
    pet_cfg = _pet_display_cfg()
    pet = None
    if is_truthy_value(pet_cfg.get("enabled"), default=False):
        pet = store.resolve_active_pet(str(pet_cfg.get("slug", "") or ""))
    if pet is None or not pet.exists:
        return _ok(rid, {"enabled": False})
    state = str(params.get("state") or constants.PetState.IDLE.value)
    scale = float(pet_cfg.get("scale", constants.DEFAULT_SCALE) or constants.DEFAULT_SCALE)
    cols = int(params.get("cols") or 0) or constants.resolve_cols(scale, pet_cfg.get("unicode_cols", 0))
    base = {"enabled": True, "slug": pet.slug, "displayName": pet.display_name, "state": state}
    if params.get("graphics") and (kitty := _pet_kitty_cells(pet, pet_cfg, state, scale)):
        return _ok(rid, {**base, **kitty})
    renderer = PetRenderer(str(pet.spritesheet), mode="unicode", scale=scale, unicode_cols=cols)
    count = renderer.frame_count(state) or 1
    frames = [[[[*top, *bottom] for (top, bottom) in row] for row in renderer.cells(state, i, cols=cols)]
              for i in range(count)]
    return _ok(rid, {**base, "cols": cols, "frameMs": constants.LOOP_MS / max(1, count), "frames": frames,
                     "scale": scale})


@_pet_method("pet.gallery", fail_open={"enabled": False, "active": "", "pets": []})
def _(rid, params: dict) -> dict:
    """Petdex gallery + local install state (installed-only offline); ``localOnly`` skips the remote manifest."""
    local_only = bool(params.get("localOnly"))
    from agent.pet import store
    pet_cfg = _pet_display_cfg()
    installed = {p.slug: p for p in store.installed_pets()}
    gallery: list[dict] = []
    try:
        from agent.pet.manifest import fetch_manifest, prefetch
        # Local-only still warms the manifest cache in the background.
        if local_only:
            prefetch()
        for entry in [] if local_only else fetch_manifest():
            gallery.append({
                "slug": entry.slug, "displayName": entry.display_name, "installed": entry.slug in installed,
                "spritesheetUrl": entry.spritesheet_url,
                # No popularity metric; petdex's hand-picked set (by asset path) is closest.
                "curated": "/curated/" in entry.spritesheet_url,
                "generated": entry.slug in installed and installed[entry.slug].generated})
    except Exception as exc:  # noqa: BLE001 - offline: fall back to installed
        logger.debug("pet.gallery manifest fetch failed: %s", exc)
    seen = {item["slug"] for item in gallery}
    gallery.extend(
        {"slug": slug, "displayName": pet.display_name, "installed": True, "spritesheetUrl": "",
         "generated": pet.generated}
        for slug, pet in installed.items() if slug not in seen)
    return _ok(rid, {"enabled": is_truthy_value(pet_cfg.get("enabled"), default=False),
                     "active": str(pet_cfg.get("slug", "") or ""), "pets": gallery})


@_pet_method("pet.select", slug=True)
def _(rid, params: dict, slug: str) -> dict:
    """Adopt a pet: install (if needed) + activate; writes ``display.pet.*`` to config."""
    from agent.pet import store
    from agent.pet.manifest import ManifestError
    from hermes_cli.pets import _set_active
    try:
        pet = store.install_pet(slug)
    except (store.PetStoreError, ManifestError) as exc:
        return _err(rid, 5031, f"could not adopt '{slug}': {exc}")
    _set_active(slug)
    return _ok(rid, {"ok": True, "slug": slug, "displayName": pet.display_name})


@_pet_method("pet.remove", slug=True)
def _(rid, params: dict, slug: str) -> dict:
    """Uninstall a pet (delete its directory); if it was active, turn the display off."""
    from agent.pet import store
    from hermes_cli.pets import _clear_active_if
    removed = store.remove_pet(slug)
    _pet_config_followup("pet.remove", _clear_active_if, slug)
    return _ok(rid, {"ok": removed, "slug": slug})


def _pet_config_followup(what: str, fn, *args) -> None:
    """Best-effort ``hermes_cli.pets`` active-slug update after a store op that already succeeded."""
    try:
        fn(*args)
    except Exception as exc:  # noqa: BLE001
        logger.debug("%s config update failed: %s", what, exc)


def _b64(data: bytes) -> str:
    import base64
    return base64.standard_b64encode(data).decode("ascii")


@_pet_method("pet.export", slug=True)
def _(rid, params: dict, slug: str) -> dict:
    """Export an installed pet as a re-importable ``.zip`` → ``{ok, filename, zipBase64}``."""
    from agent.pet import store
    filename, data = store.export_pet(slug)
    return _ok(rid, {"ok": True, "filename": filename, "zipBase64": _b64(data)})


@_pet_method("pet.rename", slug=True)
def _(rid, params: dict, slug: str) -> dict:
    """Rename a pet's display name + realign its slug/dir; follows the active slug in config."""
    if not (name := _str_param(params, "name")):
        return _err(rid, 4004, "missing name")
    from agent.pet import store
    if not (new_slug := store.rename_pet(slug, name)):
        return _err(rid, 5031, "pet.rename failed")
    if new_slug != slug:
        from hermes_cli.pets import _rename_active_if
        _pet_config_followup("pet.rename", _rename_active_if, slug, new_slug)
    return _ok(rid, {"ok": True, "slug": new_slug, "displayName": name})


@_pet_method("pet.thumb", slug=True, fail_open=lambda params: {"ok": False, "slug": _str_param(params, "slug")})
def _(rid, params: dict, slug: str) -> dict:
    """Idle-frame PNG data URI for the picker (desktop CSP breaks CDN ``<img>``); ``url``: not-yet-installed."""
    from agent.pet import store
    if not (data := store.thumbnail_png(slug, source_url=str(params.get("url") or ""))):
        return _ok(rid, {"ok": False, "slug": slug})
    return _ok(rid, {"ok": True, "slug": slug, "dataUri": "data:image/png;base64," + _b64(data)})


@_pet_method("pet.disable")
def _(rid, params: dict) -> dict:
    """``display.pet.enabled=false`` from the desktop picker."""
    from hermes_cli.pets import _set_enabled
    _set_enabled(False)
    return _ok(rid, {"ok": True})


@_pet_method("pet.scale")
def _(rid, params: dict) -> dict:
    """Persist ``display.pet.scale`` (clamped to engine bounds) from the desktop slider."""
    from hermes_cli.pets import set_pet_scale
    scale, err = set_pet_scale(params.get("scale"))
    return _err(rid, 4004, err) if err else _ok(rid, {"ok": True, "scale": scale})


@method("pet.cancel")
def _(rid, params: dict) -> dict:
    """Stop an in-flight generate/hatch by token (idempotent; off the pool so it lands mid-generation)."""
    if token := _str_param(params, "token"):
        _pet_cancel_request(token)
    return _ok(rid, {"ok": True})


@_pet_method("pet.generate.status", scoped=False, fail_open={"available": False, "providers": []})
def _(rid, params: dict) -> dict:
    """Whether pet generation is possible: a reference-capable image backend is configured."""
    from agent.pet.generate.imagegen import GenerationError, list_sprite_providers, resolve_provider
    available, providers = True, []
    try:
        resolve_provider(require_references=True)
    except GenerationError:
        available = False
    try:
        providers = list_sprite_providers()
    except Exception as exc:  # noqa: BLE001 - picker is best-effort
        logger.debug("pet provider list failed: %s", exc)
    return _ok(rid, {"available": available, "providers": providers})


def _pet_pick_provider(params: dict, *, require_references: bool):
    """Picker-chosen ``params.provider`` resolved up front (a bad pick fails fast, not mid-fan-out)."""
    from agent.pet.generate.imagegen import resolve_provider
    name = _str_param(params, "provider")
    return resolve_provider(require_references=require_references, prefer=name) if name else None


@_pet_method("pet.generate", scoped=False)
def _(rid, params: dict) -> dict:
    """Candidate base looks for a new pet (draft step; worker pool): ``prompt`` (or a ``referenceImage``
    data URL), ``count`` (≤4), ``style``, ``provider`` → ``{ok, token, drafts:[{index, dataUri}]}``."""
    prompt = _str_param(params, "prompt")
    ref_raw = _str_param(params, "referenceImage")
    if not prompt and not ref_raw:
        return _err(rid, 4004, "missing prompt")
    count = max(1, min(4, _int_param(params, "count", 4) or 4))
    import shutil
    from agent.pet.generate import generate_base_drafts
    from agent.pet.generate.imagegen import GenerationError
    root = _pet_gen_root()
    _pet_gen_sweep(root)
    # Token up front so each draft is staged + streamed the moment it lands.
    token = uuid.uuid4().hex[:12]
    _pet_cancel_arm(token)
    stage = root / token
    stage.mkdir(parents=True, exist_ok=True)
    reference_images = None
    if ref_raw:
        try:
            reference_images = _pet_reference_images_from_data_url(ref_raw, stage)
        except ValueError as exc:
            return _pet_gen_abort(rid, token, 4004, str(exc))
    try:
        sprite = _pet_pick_provider(params, require_references=bool(reference_images))
    except GenerationError as exc:
        return _pet_gen_abort(rid, token, 5031, str(exc))
    out: list[dict] = []
    # Token-only init event so a Stop fired before the first draft can target this run.
    _pet_emit("pet.generate.progress", {"token": token, "count": count}, "pet.generate init")

    def _on_draft(index: int, src) -> None:
        dest = stage / f"draft-{index}.png"
        try:
            shutil.copyfile(src, dest)
            data_uri = _pet_png_data_uri(dest)
        except Exception as exc:  # noqa: BLE001 - skip a bad draft, keep the rest
            logger.debug("pet.generate draft %d failed: %s", index, exc)
            return
        out.append({"index": index, "dataUri": data_uri})
        _pet_emit("pet.generate.progress", {"token": token, "index": index, "dataUri": data_uri, "count": count},
                  "pet.generate progress")
    try:
        generate_base_drafts(prompt or "a pet based on the reference image", n=count,
                             style=_str_param(params, "style", "auto"), reference_images=reference_images,
                             provider=sprite, on_draft=_on_draft, is_cancelled=lambda: _pet_is_cancelled(token))
    except GenerationError as exc:
        return _pet_gen_abort(rid, token, 5031, str(exc))
    cancelled = _pet_is_cancelled(token)
    _pet_cancel_release(token)
    if cancelled or not out:
        return _err(rid, 5031, "generation cancelled" if cancelled else "generation produced no usable drafts")
    return _ok(rid, {"ok": True, "token": token, "drafts": sorted(out, key=lambda d: d["index"])})


@_pet_method("pet.hatch", scoped=False)
def _(rid, params: dict) -> dict:
    """Turn a base draft (``token`` + ``index``) into a full pet — installed but NOT active (``pet.select``
    adopts, ``pet.remove`` discards) → ``{ok, slug, displayName, warnings, pet}``."""
    token, name = _str_param(params, "token"), _str_param(params, "name")
    if not token or not name:
        return _err(rid, 4004, "missing token" if not token else "missing name")
    # Own cancel key: pet.generate may still be releasing `token`. Falls back for old clients.
    cancel_token = _str_param(params, "cancelToken") or token
    from agent.pet import store
    from agent.pet.generate import hatch_pet
    from agent.pet.generate.imagegen import GenerationError
    base = _pet_gen_root() / token / f"draft-{_int_param(params, 'index', 0)}.png"
    if not base.is_file():
        return _err(rid, 4004, "draft expired — generate again")
    try:
        sprite = _pet_pick_provider(params, require_references=True)  # rows always need reference grounding
    except GenerationError as exc:
        return _err(rid, 5031, str(exc))
    _pet_cancel_arm(cancel_token)
    slug = store.unique_slug(name)

    def _on_progress(event: str, detail: str) -> None:
        # Row progress "<state>:<done>:<total>" → "Drawing <state>… (n/total)".
        payload: dict = {"event": event, "detail": detail}
        if event == "row" and detail.count(":") == 2:
            state, done, total = detail.split(":")
            payload = {"event": "row", "state": state, "done": done, "total": total}
        _pet_emit("pet.hatch.progress", payload, "pet.hatch progress")
    try:
        result = hatch_pet(
            base_image=base, slug=slug, display_name=name, description=str(params.get("description") or ""),
            concept=str(params.get("prompt") or name), style=_str_param(params, "style", "auto"), provider=sprite,
            on_progress=_on_progress, is_cancelled=lambda: _pet_is_cancelled(cancel_token))
    except GenerationError as exc:
        return _err(rid, 5031, str(exc))
    finally:
        _pet_cancel_release(cancel_token)
    pet = store.load_pet(result.slug)
    return _ok(rid, {"ok": True, "slug": result.slug, "displayName": result.display_name,
                     "warnings": result.validation.get("warnings", []),
                     "pet": _pet_sprite_payload(pet, scale=_pet_config_scale()) if pet else {}})


def register(server) -> None:
    """Publish this module's helpers onto ``server`` (rebound to its globals) and install handlers."""
    bind_module(globals(), server, skip=("_",))
