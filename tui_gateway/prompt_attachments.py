"""Attachment staging: image sniffing, size caps, per-session attachment dirs, path resolution.

Bodies are rebound onto server.py's globals at install time (see
method_ctx.bind_module), so they reference server.py globals bare.
"""

from __future__ import annotations

import re as _re

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()


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
    """Write image bytes into the gateway's images dir and queue them.

    Mirrors what ``image.attach`` does for a local path: appends to
    ``session["attached_images"]`` so the next ``prompt.submit`` picks it up via
    the existing native-image-attach pipeline. Returns the written path.
    """
    profile_home = session.get("profile_home")
    profile_incarnation = session.get("profile_incarnation")
    # Profile mutation takes the lifecycle lease before retiring sessions, so
    # attachment writes use the same order: lifecycle → sessions → pathname.
    with _profile_home_lease(profile_home, profile_incarnation):
        with _sessions_lock:
            session["image_counter"] = session.get("image_counter", 0) + 1
            img_dir = _session_images_dir(session)
            img_path = None
            try:
                if not img_dir.parent.is_dir():
                    raise FileNotFoundError(
                        "Profile home is missing or being deleted: "
                        f"{img_dir.parent}"
                    )
                img_dir.mkdir(parents=False, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_path = img_dir / f"{prefix}_{ts}_{session['image_counter']}{ext}"
                img_path.write_bytes(img_bytes)
            except Exception:
                if img_path is not None:
                    img_path.unlink(missing_ok=True)
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


def _stage_browser_file_attachment(
    session: dict,
    staged_upload: object,
    name: str,
) -> tuple[Path, bool]:
    """Copy an upload from its captured owner into this session's workspace.

    Source and destination may be different profiles on the same installation
    (an owner-routed tile can receive a foreground browser pick). Lease both
    generations; do not infer either identity from the ambient profile.
    """
    from hermes_constants import WEBAPP_ATTACHMENT_MAX_BYTES
    from hermes_cli.install_identity import get_install_id

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

    with _profile_home_lease(raw_home, incarnation) as home:
        with _profile_home_lease(
            session.get("profile_home"), session.get("profile_incarnation")
        ):
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
                session, raw_path=str(source), data_url="", name=name
            )


def _stage_session_file_attachment(
    session: dict, *, raw_path: str, data_url: str, name: str) -> tuple[Path, bool]:
    """Make a desktop file attachment available to the gateway agent: ``(stored_path, uploaded)``.
    Inside the workspace -> as-is; gateway-visible but outside -> copied into ``attachments/``
    (bind-mounted into container backends so ``@file:`` resolves in the sandbox); not on the
    gateway -> ``data_url`` bytes decoded into ``attachments/``."""
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
            return resolved, False
        except ValueError:
            payload = resolved.read_bytes()
            filename = resolved.name
    else:
        if not data_url:
            raise ValueError("file not found on gateway and no data_url provided")
        # Any media type (unlike the image-specific decoder); bare base64 also accepted.
        import binascii as _binascii
        import re as _re
        try:
            payload = _b64_payload(
                data_url, r"^data:[^;,]*(?:;[^;,=]+=[^;,]+)*;base64,(.*)$", _re.DOTALL | _re.I)
        except (ValueError, _binascii.Error) as exc:
            raise ValueError("invalid data_url payload") from exc
        filename = _sanitize_attachment_name(name or Path(str(raw_path or "")).name)
    with _profile_home_lease(session.get("profile_home"), session.get("profile_incarnation")):
        with _sessions_lock:
            root = _session_home_dir(session, "attachments")
            from hermes_constants import profile_deletion_marker_path
            named_profile = profile_deletion_marker_path(root.parent) is not None
            if _profile_home_rejected(session.get("profile_home"), session.get("profile_incarnation")) or (named_profile and not root.parent.is_dir()):
                raise FileNotFoundError(f"Profile home is missing or being deleted: {root.parent}")
            root.mkdir(parents=not named_profile, exist_ok=True)
            filename = _sanitize_attachment_name(filename)
            target = root / filename
            if target.exists():
                stem = Path(filename).stem or "attachment"
                suffix = Path(filename).suffix
                counter = 2
                while (target := root / f"{stem}-{counter}{suffix}").exists():
                    counter += 1
            try:
                target.write_bytes(payload)
            except Exception:
                target.unlink(missing_ok=True)
                raise
            return target.resolve(), True


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
