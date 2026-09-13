"""Attachment staging: image sniffing, size caps, per-session attachment dirs, path resolution.

Bodies are rebound onto server.py's globals at install time (see
method_ctx.bind_module), so they reference server.py globals bare.
"""

from __future__ import annotations

import re as _re

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()


_ATTACH_BYTES_MAX_BYTES = 25 * 1024 * 1024
_FILE_ATTACH_MAX_BYTES = 100 * 1024 * 1024
_FILE_ATTACH_CHUNK_BYTES = 1024 * 1024
_PDF_ATTACH_MAX_BYTES = 50 * 1024 * 1024
_PDF_ATTACH_MAX_PAGES = 25

# Leading magic bytes -> file extension, for filename-less uploads.
_IMAGE_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", ".png"), (b"\xff\xd8\xff", ".jpg"), (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"), (b"BM", ".bmp"))

# Context-ref values containing any of these must be quoted (desktop formatRefValue parity).
_ATTACHMENT_REF_NEEDS_QUOTING_RE = _re.compile(r"""[\s()\[\]{}<>"'`]""")
del _re  # bodies are rebound onto server globals: import inside functions only


def _b64_payload(raw: str, data_url_re: str, flags: int) -> bytes:
    """Strip an optional ``data:...;base64,`` wrapper and all whitespace, then strictly decode."""
    import base64 as _base64
    import re as _re
    cleaned = (raw or "").strip()
    if m := _re.match(data_url_re, cleaned, flags):
        cleaned = m.group(1)
    return _base64.b64decode(_re.sub(r"\s+", "", cleaned), validate=True)


def _decode_attach_base64(raw: str, *, mime_prefix: str) -> bytes | None:
    """Decode a (``data:<mime_prefix>...;base64,``-wrapped) payload; None when invalid."""
    import re as _re
    try:
        return _b64_payload(
            raw, rf"^data:{_re.escape(mime_prefix)}[a-zA-Z0-9.+-]*;base64,(.*)$", _re.DOTALL)
    except Exception:
        return None


def _decode_attach_payload(
    rid, raw_b64: str, *, mime_prefix: str, max_bytes: int, label: str, empty_msg: str):
    """``(bytes, None)`` or ``(None, error)``: 4017 on bad/empty base64, 4018 over *max_bytes*."""
    data = _decode_attach_base64(raw_b64, mime_prefix=mime_prefix)
    if data is None:
        return None, _err(rid, 4017, "data is not valid base64")
    if not data:
        return None, _err(rid, 4017, empty_msg)
    if len(data) > max_bytes:
        mb = max_bytes // (1024 * 1024)
        return None, _err(rid, 4018, f"{label} too large ({len(data)} bytes; cap is {mb} MB)")
    return data, None


def _sniff_image_ext(img_bytes: bytes, filename: str = "") -> str:
    """Extension from the filename hint, else magic bytes (WebP: RIFF container), else ``.png``."""
    if filename and (suffix := Path(filename).suffix.lower()):
        return suffix
    head = img_bytes[:16]
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return ".webp"
    return next((ext for sig, ext in _IMAGE_MAGIC if head.startswith(sig)), ".png")


def _file_attachment_image(path: Path) -> dict | None:
    """Promote only bounded, decodable rasters, not filenames, MIME hints or magic alone."""
    import stat
    from io import BytesIO
    from tools.vision_tools_image_prep import _validate_raster_image_decodable
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= _ATTACH_BYTES_MAX_BYTES:
            return None
        head = source.read(16)
        ext = next((ext for sig, ext in _IMAGE_MAGIC if head.startswith(sig)), "")
        if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
            ext = ".webp"
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".gif": "image/gif",
                ".bmp": "image/bmp", ".webp": "image/webp"}.get(ext)
        if not mime:
            return None
        data = head + source.read(_ATTACH_BYTES_MAX_BYTES + 1 - len(head))
    if len(data) > _ATTACH_BYTES_MAX_BYTES:
        return None
    # Validate the same bounded bytes, never reopen a potentially replaced path.
    # The vision decoder also caps animated frame count and aggregate pixels.
    # Pillow accepts streams as well as the decoder's annotated Path input.
    if _validate_raster_image_decodable(BytesIO(data)) is not None:  # type: ignore[arg-type]
        return None
    return {"name": path.name, "mime_type": mime}


def _validate_draft_image_paths(session: dict, paths) -> list[str]:
    """Only exact successful file.attach grants can join this deliberate prompt."""
    if not isinstance(paths, list) or len(paths) > 32 or any(not isinstance(p, str) for p in paths):
        raise ValueError("draft_image_paths must be a list of at most 32 staged image paths")
    if not paths:
        return []
    with session["history_lock"]:
        grants = set(session.get("file_attachment_paths", ()))
    root = _session_home_dir(session, "attachments").resolve()
    validated = []
    for raw in dict.fromkeys(paths):
        path = Path(raw)
        if raw not in grants or not path.is_absolute() or path.resolve() != path or not path.is_relative_to(root):
            raise ValueError("Draft image is not attached to this session")
        if _file_attachment_image(path) is None:
            raise ValueError("Draft image is unsupported or too large")
        validated.append(raw)
    return validated


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


def _queue_attached_image(session: dict, img_bytes: bytes, ext: str, *, prefix: str) -> Path:
    """Write image bytes into the session images dir and queue them for the next submit."""
    session["image_counter"] = session.get("image_counter", 0) + 1
    img_dir = _session_images_dir(session)
    img_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = img_dir / f"{prefix}_{ts}_{session['image_counter']}{ext}"
    try:
        img_path.write_bytes(img_bytes)
    except Exception:
        session["image_counter"] = max(0, session["image_counter"] - 1)
        raise
    session.setdefault("attached_images", []).append(str(img_path))
    return img_path


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


def _stage_session_file_attachment(
    session: dict, *, raw_path: str, data_url: str, name: str) -> tuple[Path, bool]:
    """Make a desktop file attachment available to the gateway agent: ``(stored_path, uploaded)``.
    Non-image workspace files stay as-is; images and gateway-visible outside files are
    copied into ``attachments/`` (bind-mounted into container backends); otherwise
    ``data_url`` bytes are decoded there."""
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
        # Copying/renaming must not launder a denied credential path into an allowed grant.
        from agent.context_references import _ensure_reference_path_allowed
        _ensure_reference_path_allowed(resolved)
        # Attach RPCs precede turn scope binding. Keep the gateway/global denies,
        # and apply the destination profile's same canonical guard before copying.
        home_token = set_hermes_home_override(_session_home(session))
        try:
            _ensure_reference_path_allowed(resolved)
        finally:
            reset_hermes_home_override(home_token)
        if resolved.is_relative_to(workspace) and _file_attachment_image(resolved) is None:
            return resolved, False
        if resolved.stat().st_size > _FILE_ATTACH_MAX_BYTES:
            raise ValueError("File is too large")
        payload = None
        filename = resolved.name
    else:
        if not data_url:
            raise ValueError("file not found on gateway and no data_url provided")
        # Refuse before regex/decoding can duplicate an unbounded JSON payload.
        if len(data_url) > ((_FILE_ATTACH_MAX_BYTES + 2) // 3) * 4 + 1024:
            raise ValueError("File is too large")
        # Any media type (unlike the image-specific decoder); bare base64 also accepted.
        import binascii as _binascii
        import re as _re
        try:
            payload = _b64_payload(
                data_url, r"^data:[^;,]*(?:;[^;,=]+=[^;,]+)*;base64,(.*)$", _re.DOTALL | _re.I)
        except (ValueError, _binascii.Error) as exc:
            raise ValueError("invalid data_url payload") from exc
        if len(payload) > _FILE_ATTACH_MAX_BYTES:
            raise ValueError("File is too large")
        filename = _sanitize_attachment_name(name or Path(str(raw_path or "")).name)
    root = _session_home_dir(session, "attachments")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    filename = _sanitize_attachment_name(filename)
    target = root / filename
    # Reserve exclusively, not exists()+write (two sockets can attach the same name).
    while True:
        try:
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            break
        except FileExistsError:
            import uuid
            target = root / f"{Path(filename).stem}-{uuid.uuid4().hex}{Path(filename).suffix}"
    try:
        with os.fdopen(fd, "wb") as out:
            if payload is not None:
                out.write(payload)
            else:
                import stat
                # Nonblocking prevents a replaced FIFO from hanging the RPC reader.
                source_fd = os.open(resolved, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
                                    | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(source_fd, "rb") as source:
                    if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                        raise ValueError("Only regular files can be attached")
                    total = 0
                    while chunk := source.read(_FILE_ATTACH_CHUNK_BYTES):
                        total += len(chunk)
                        if total > _FILE_ATTACH_MAX_BYTES:
                            raise ValueError("File is too large")
                        out.write(chunk)
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return target.resolve(), True


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
