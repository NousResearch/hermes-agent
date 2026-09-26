"""Attachment staging: image sniffing, size caps, per-session attachment dirs, path resolution.

Bodies are rebound onto server.py's globals at install time (see
method_ctx.bind_module), so they reference server.py globals bare.
"""

from __future__ import annotations

import re as _re

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()


# Match the native desktop file reader; image/browser inputs retain tighter caps.
_ATTACHMENT_MAX_BYTES = 256 * 1024 * 1024
_ATTACH_BYTES_MAX_BYTES = 25 * 1024 * 1024
_PDF_ATTACH_MAX_BYTES = 50 * 1024 * 1024
_PDF_ATTACH_MAX_PAGES = 25

# Leading magic bytes -> file extension, for filename-less uploads.
_IMAGE_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", ".png"), (b"\xff\xd8\xff", ".jpg"), (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"), (b"BM", ".bmp"))

# Context-ref values containing any of these must be quoted (desktop formatRefValue parity).
_ATTACHMENT_REF_NEEDS_QUOTING_RE = _re.compile(r"""[\s()\[\]{}<>"'`]""")
del _re  # bodies are rebound onto server globals: import inside functions only


def _b64_payload(raw: str, data_url_re: str, flags: int, *, max_bytes: int) -> bytes:
    """Strip an optional ``data:...;base64,`` wrapper and all whitespace, then strictly decode."""
    import base64 as _base64
    import re as _re
    cleaned = (raw or "").strip()
    if m := _re.match(data_url_re, cleaned, flags):
        cleaned = m.group(1)
    cleaned = _re.sub(r"\s+", "", cleaned)
    if len(cleaned) > 4 * ((max_bytes + 2) // 3):
        raise OverflowError(f"attachment too large; size limit is {max_bytes} bytes")
    payload = _base64.b64decode(cleaned, validate=True)
    if len(payload) > max_bytes:
        raise OverflowError(f"attachment too large; size limit is {max_bytes} bytes")
    return payload


def _decode_attach_base64(raw: str, *, mime_prefix: str, max_bytes: int) -> bytes | None:
    """Decode a (``data:<mime_prefix>...;base64,``-wrapped) payload; None when invalid."""
    import re as _re
    try:
        return _b64_payload(
            raw, rf"^data:{_re.escape(mime_prefix)}[a-zA-Z0-9.+-]*;base64,(.*)$", _re.DOTALL,
            max_bytes=max_bytes)
    except OverflowError:
        raise
    except Exception:
        return None


def _decode_attach_payload(
    rid, raw_b64: str, *, mime_prefix: str, max_bytes: int, label: str, empty_msg: str):
    """``(bytes, None)`` or ``(None, error)``: 4017 on bad/empty base64, 4018 over *max_bytes*."""
    try:
        data = _decode_attach_base64(raw_b64, mime_prefix=mime_prefix, max_bytes=max_bytes)
    except OverflowError as exc:
        return None, _err(rid, 4018, f"{label} too large: {exc}")
    if data is None:
        return None, _err(rid, 4017, "data is not valid base64")
    if not data:
        return None, _err(rid, 4017, empty_msg)
    return data, None


def _sniff_image_ext(img_bytes: bytes, filename: str = "") -> str:
    """Extension from the filename hint, else magic bytes (WebP: RIFF container), else ``.png``."""
    if filename and (suffix := Path(filename).suffix.lower()):
        return suffix
    head = img_bytes[:16]
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return ".webp"
    return next((ext for sig, ext in _IMAGE_MAGIC if head.startswith(sig)), ".png")


def _allowed_image_extensions() -> frozenset[str]:
    try:
        from cli import _IMAGE_EXTENSIONS
        return frozenset(_IMAGE_EXTENSIONS)
    except Exception:
        return frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"})


def _session_home_dir(session: dict, name: str) -> Path:
    """``<session home>/<name>``, anchored on the session's stored ``profile_home``: attach
    RPCs run BEFORE ``prompt.submit`` installs the profile HERMES_HOME override, while
    the sandbox mounts and the vision host-read allowlist resolve the *session profile's*
    dirs at run time — writing anywhere else means the agent can never see the file."""
    profile_home = session.get("profile_home")
    return (Path(profile_home) if profile_home else _hermes_home) / name


def _session_images_dir(session: dict) -> Path:
    return _session_home_dir(session, "images")


def _attachment_owner(session: dict, sid: str) -> tuple:
    """Capture the RPC's exact registry slot and generation before any slow I/O."""
    with _sessions_lock:
        owner = sid, session.get("profile_home") or None, session.get("profile_incarnation") or None
        _check_attachment_owner(session, owner)
        return owner


def _check_attachment_owner(session: dict, owner: tuple) -> None:
    """Called under the sessions lock; a detached/rebound record cannot publish."""
    sid, home, incarnation = owner
    if _sessions.get(sid) is not session or session.get("_closing") or session.get("_finalized"):
        raise LookupError("session not found")
    if not _session_profile_identity_matches(session, home, incarnation):
        raise FileNotFoundError("profile incarnation changed during attachment")


def _read_attachment_bytes(path: Path, max_bytes: int) -> bytes:
    # Bound the actual read, not just stat(): the source can grow after inspection.
    with path.open("rb") as source:
        payload = source.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError(f"attachment too large; size limit is {max_bytes} bytes")
    return payload


def _write_attachment_temp(root: Path, payload: bytes) -> Path:
    """Exclusive, owner-only staging; never expose a partially written attachment."""
    import io
    temp = root / f".attachment-{uuid.uuid4().hex}.tmp"
    handle = io.open(temp, "xb", opener=lambda path, flags: os.open(path, flags, 0o600))
    try:
        with handle:
            handle.write(payload)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return temp


def _publish_attachment(
    session: dict, owner: tuple, payload: bytes, filename: str, *, image_prefix: str = "",
) -> Path:
    if len(payload) > _ATTACHMENT_MAX_BYTES:
        raise ValueError(f"attachment too large; size limit is {_ATTACHMENT_MAX_BYTES} bytes")
    _, home, incarnation = owner
    # Deletion/recreation waits for the generation lease, not the global registry.
    with _profile_home_lease(home, incarnation):
        with _sessions_lock:
            _check_attachment_owner(session, owner)
            root = _session_home_dir(session, "images" if image_prefix else "attachments")
        from hermes_constants import mkdir_under_hermes_home
        from utils import publish_no_clobber
        if image_prefix and not root.parent.is_dir():
            raise FileNotFoundError(f"Profile home is missing or being deleted: {root.parent}")
        mkdir_under_hermes_home(root)
        temp = _write_attachment_temp(root, payload)
        try:
            with _sessions_lock:
                _check_attachment_owner(session, owner)
                if image_prefix:
                    counter = session.get("image_counter", 0) + 1
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{image_prefix}_{ts}_{counter}{filename}"
                filename = _sanitize_attachment_name(filename)
                target = root / filename
                stem, suffix = Path(filename).stem, Path(filename).suffix
                duplicate = 2
                while True:
                    try:
                        # No-clobber publication also fences other processes in
                        # the launch home, whose lifecycle lease is a no-op.
                        publish_no_clobber(temp, target)
                        break
                    except FileExistsError:
                        target = root / f"{stem}-{duplicate}{suffix}"
                        duplicate += 1
                if image_prefix:
                    session["image_counter"] = counter
                    session.setdefault("attached_images", []).append(str(target))
                return target
        finally:
            temp.unlink(missing_ok=True)


def _queue_attached_image(
    session: dict, img_bytes: bytes, ext: str, *, prefix: str, owner: tuple,
) -> Path:
    """Publish to the session's image queue only while its captured owner is live."""
    return _publish_attachment(session, owner, img_bytes, ext, image_prefix=prefix)


def _format_ref_value(value: str) -> str:
    """Quote a value with whitespace/brackets/quotes so the ``@file:`` ref round-trips."""
    if not value or not _ATTACHMENT_REF_NEEDS_QUOTING_RE.search(value):
        return value
    for q in ("`", '"', "'"):
        if q not in value:
            return f"{q}{value}{q}"
    return value


def _attachment_ref_path(session: dict, target: Path) -> str:
    """Workspace-relative path for an attachment, or the absolute path if outside."""
    workspace = Path(_session_cwd(session)).resolve()
    try:
        return str(target.resolve().relative_to(workspace)).replace(os.sep, "/")
    except ValueError:
        return str(target.resolve())


def _sanitize_attachment_name(name: str) -> str:
    import re as _re
    candidate = _re.sub(r"[\x00-\x1f]+", "_", Path(str(name or "").strip()).name)
    return candidate.strip().strip(".") or "attachment"


def _stage_browser_file_attachment(
    session: dict,
    staged_upload: object,
    name: str,
    *, owner: tuple,
) -> tuple[Path, bool]:
    """Copy an upload from its captured owner into this session's workspace.

    Source and destination may be different profiles on the same installation
    (an owner-routed tile can receive a foreground browser pick). Lease both
    generations; do not infer either identity from the ambient profile.
    """
    from hermes_constants import WEBAPP_ATTACHMENT_MAX_BYTES
    from hermes_cli.install_identity import get_install_id
    from hermes_cli.profile_lifecycle import profile_lifecycle_lease

    if not isinstance(staged_upload, dict):
        raise ValueError("invalid staged upload")
    install_id = get_install_id()
    if not install_id or staged_upload.get("install_id") != install_id:
        raise ValueError("Staged attachment belongs to another Hermes backend; select the file again")
    raw_home = staged_upload.get("profile_home")
    raw_path = staged_upload.get("path")
    incarnation = staged_upload.get("profile_incarnation")
    if (
        not isinstance(raw_home, str) or not raw_home
        or not isinstance(raw_path, str) or not raw_path
        or (incarnation is not None and not isinstance(incarnation, str))
    ):
        raise ValueError("invalid staged upload source")

    # Acquire both pathname locks in sorted order before checking generations.
    with profile_lifecycle_lease(raw_home, owner[1] or _hermes_home):
        with _profile_home_lease(raw_home, incarnation) as home, _profile_home_lease(owner[1], owner[2]):
            source = Path(raw_path)
            if source.is_symlink():
                raise ValueError("staged upload is no longer a regular file")
            source = source.resolve(strict=True)
            if (
                source.parent != (home / "uploads").resolve(strict=True)
                or not source.name.startswith("web-")
                or not source.is_file()
            ):
                raise ValueError("staged upload is outside its source profile")
            if source.stat().st_size > WEBAPP_ATTACHMENT_MAX_BYTES:
                raise ValueError("staged upload exceeds the browser attachment size limit")
            # Keep the existing out-of-workspace copy into attachments/, which
            # is visible to container/SSH terminal backends through cache mounts.
            return _stage_session_file_attachment(
                session, raw_path=str(source), data_url="", name=name, owner=owner,
                max_bytes=WEBAPP_ATTACHMENT_MAX_BYTES,
            )


def _stage_session_file_attachment(
    session: dict, *, raw_path: str, data_url: str, name: str, owner: tuple,
    max_bytes: int | None = None,
) -> tuple[Path, bool]:
    """Make a desktop file attachment available to the gateway agent: ``(stored_path, uploaded)``.
    Inside the workspace -> as-is; gateway-visible but outside -> copied into ``attachments/``
    (bind-mounted into container backends so ``@file:`` resolves in the sandbox); not on the
    gateway -> ``data_url`` bytes decoded into ``attachments/``."""
    max_bytes = min(max_bytes, _ATTACHMENT_MAX_BYTES) if max_bytes is not None else _ATTACHMENT_MAX_BYTES
    workspace = Path(_session_cwd(session)).resolve()
    resolved = None
    if raw_path:
        try:
            from cli import _detect_file_drop, _resolve_attachment_path, _split_path_input
        except Exception:
            _detect_file_drop = None
        if _detect_file_drop is not None:
            dropped = _detect_file_drop(raw_path)
            if dropped:
                resolved = Path(dropped["path"]).resolve()
            else:
                path_token, _remainder = _split_path_input(raw_path)
                found = _resolve_attachment_path(path_token)
                resolved = Path(found).resolve() if found is not None else None
    if resolved is not None:
        try:
            resolved.relative_to(workspace)
            with _sessions_lock:
                _check_attachment_owner(session, owner)
            return resolved, False
        except ValueError:
            payload = _read_attachment_bytes(resolved, max_bytes)
            filename = resolved.name
    else:
        if not data_url:
            raise ValueError("file not found on gateway and no data_url provided")
        # Any media type (unlike the image-specific decoder); bare base64 also accepted.
        import binascii as _binascii
        import re as _re
        try:
            payload = _b64_payload(
                data_url, r"^data:[^;,]*(?:;[^;,=]+=[^;,]+)*;base64,(.*)$", _re.DOTALL | _re.I,
                max_bytes=max_bytes)
        except (ValueError, _binascii.Error) as exc:
            raise ValueError("invalid data_url payload") from exc
        filename = _sanitize_attachment_name(name or Path(str(raw_path or "")).name)
    return _publish_attachment(session, owner, payload, filename).resolve(), True


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
