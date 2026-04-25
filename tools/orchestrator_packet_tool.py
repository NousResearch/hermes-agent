"""orchestrator_packet_write — confined packet writer for orchestrator dispatch."""

import errno
import hashlib
import json
import logging
import os
import re
from pathlib import Path

from tools.registry import registry

logger = logging.getLogger(__name__)

_FILENAME_RE = re.compile(r'^[a-zA-Z0-9_-]{1,72}\.[a-zA-Z0-9]{1,10}$')
_PACKET_DIR_ENV = "HERMES_PACKET_DIR"
_PACKET_MAX_BYTES_ENV = "HERMES_PACKET_MAX_BYTES"
_DEFAULT_MAX_BYTES = 128 * 1024  # 128 KiB

# Profiles explicitly blocked from writing dispatch packets (exact lowercase match).
# Exact set comparison avoids brittle substring matching.
_BLOCKED_PACKET_WRITE_PROFILES = frozenset({"orcheapworker", "evidence-worker"})


def _canonical_packet_base() -> Path:
    """Return the resolved canonical base that HERMES_PACKET_DIR must reside within."""
    from hermes_constants import get_hermes_home
    return get_hermes_home().resolve()


def _packet_dir() -> Path:
    """Return the packet directory, enforcing confinement within the canonical base.

    Raises ValueError if HERMES_PACKET_DIR resolves outside the canonical base.
    """
    raw = os.environ.get(_PACKET_DIR_ENV, "")
    if raw:
        candidate = Path(raw).expanduser().resolve()
        base = _canonical_packet_base()
        try:
            candidate.relative_to(base)
        except ValueError:
            raise ValueError(
                f"HERMES_PACKET_DIR resolves to {str(candidate)!r} which is outside "
                f"the canonical base {str(base)!r}. "
                f"Set HERMES_HOME to configure an alternate deployment root."
            )
        return candidate
    return Path.home() / ".local" / "state" / "life" / "cc-packets"


def _symlinked_component(path: Path) -> str | None:
    """Return the first symlinked path component in *path*, or None.

    *path* must be the raw (user-supplied, pre-resolution) path.  Passing an
    already-resolved path defeats this check because resolution silently
    eliminates symlinks before inspection.

    Walks from *path* up to the filesystem root checking each component with
    os.lstat so that a symlinked parent directory is caught before the write
    reaches os.open.  Returns the offending component as a string, or None if
    the entire ancestor chain is symlink-free.
    """
    current = Path(os.path.abspath(path))  # anchor without resolving symlinks
    while True:
        try:
            if current.is_symlink():
                return str(current)
        except OSError:
            pass
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def _raw_packet_dir() -> Path:
    """Return HERMES_PACKET_DIR expanded but NOT resolved — for pre-resolution symlink checking.

    Unlike _packet_dir(), this function does not call .resolve(), so symlinked
    components in the path remain detectable by _symlinked_component().
    """
    raw = os.environ.get(_PACKET_DIR_ENV, "")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".local" / "state" / "life" / "cc-packets"


def _max_bytes() -> int:
    raw = os.environ.get(_PACKET_MAX_BYTES_ENV, "")
    try:
        return int(raw) if raw else _DEFAULT_MAX_BYTES
    except ValueError:
        return _DEFAULT_MAX_BYTES


def _orchestrator_profile_only() -> bool:
    """Return True only for profiles that may materialise dispatch packets."""
    profile = os.environ.get("HERMES_PROFILE", "").lower()
    return profile not in _BLOCKED_PACKET_WRITE_PROFILES


def orchestrator_packet_write(
    filename: str,
    content: str,
    overwrite: bool = False,
) -> str:
    """Write a dispatch packet to the confined cc-packets directory."""
    # Basename-only guard: reject anything containing a separator
    if (
        os.sep in filename
        or "/" in filename
        or filename.startswith("~")
        or os.path.isabs(filename)
    ):
        return json.dumps({
            "error": (
                f"filename must be a basename only (no path separators): {filename!r}"
            )
        })

    # Pattern validation
    if not _FILENAME_RE.match(filename):
        return json.dumps({
            "error": (
                f"filename {filename!r} is invalid. "
                "Must match [a-zA-Z0-9_-]{{1,72}}.<ext> "
                "(e.g. pkt_20260425_081637_1519.md). "
                "No spaces, no special characters."
            )
        })

    # Size check
    content_bytes = content.encode("utf-8")
    max_bytes = _max_bytes()
    if len(content_bytes) > max_bytes:
        return json.dumps({
            "error": (
                f"content is {len(content_bytes):,} bytes which exceeds the "
                f"size limit ({max_bytes:,} bytes). "
                "Split into smaller packets or reduce content."
            )
        })

    # Reject symlinked parents using the RAW (pre-resolution) path.
    # _packet_dir() calls .resolve() which silently eliminates symlinks, so
    # checking the resolved path would miss a symlinked component that resolves
    # inside the allowed base.  We must inspect the original path chain first.
    sym = _symlinked_component(_raw_packet_dir())
    if sym:
        return json.dumps({
            "error": (
                f"A component of the packet directory is a symlink ({sym!r}). "
                "Refusing to write to prevent directory-traversal via symlink chains."
            )
        })

    # Resolve packet directory — raises ValueError if outside canonical base
    try:
        pkt_dir = _packet_dir()
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    pkt_dir.mkdir(parents=True, exist_ok=True)
    target = pkt_dir / filename

    # sha256 computed before open so a write failure never produces a stale hash
    sha256 = hashlib.sha256(content_bytes).hexdigest()

    # Atomic open: O_NOFOLLOW rejects symlinks; O_EXCL makes the overwrite
    # check and the create atomic, eliminating the TOCTOU window.
    flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW
    if not overwrite:
        flags |= os.O_EXCL
    try:
        fd = os.open(str(target), flags, 0o600)
    except FileExistsError:
        return json.dumps({
            "error": (
                f"{filename!r} already exists in the packet directory. "
                "Pass overwrite=true to replace it."
            )
        })
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            return json.dumps({
                "error": (
                    f"{filename!r} resolves to a symlink and cannot be written to. "
                    "Remove the symlink and retry."
                )
            })
        raise

    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content_bytes)
    except Exception:
        raise

    # content is intentionally excluded from the return value
    return json.dumps({
        "filename": filename,
        "path": str(target),
        "sha256": sha256,
        "size_bytes": len(content_bytes),
    })


ORCHESTRATOR_PACKET_WRITE_SCHEMA = {
    "name": "orchestrator_packet_write",
    "description": (
        "Write a dispatch packet to the confined cc-packets directory "
        "(~/.local/state/life/cc-packets/ by default, overridable via HERMES_PACKET_DIR). "
        "Returns sha256 of the written content. "
        "Content is never echoed in the return value."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": (
                    "Basename only — no slashes or path separators. "
                    "Pattern: [a-zA-Z0-9_-]{1..72}.<ext> "
                    "(e.g. pkt_20260425_081637_1519.md)."
                ),
            },
            "content": {
                "type": "string",
                "description": (
                    "Full text content of the packet. Never echoed in the return value."
                ),
            },
            "overwrite": {
                "type": "boolean",
                "description": "If true, replace an existing packet. Default false.",
                "default": False,
            },
        },
        "required": ["filename", "content"],
    },
}


def _handle_orchestrator_packet_write(args: dict, **kw) -> str:
    return orchestrator_packet_write(
        filename=args.get("filename", ""),
        content=args.get("content", ""),
        overwrite=bool(args.get("overwrite", False)),
    )


registry.register(
    name="orchestrator_packet_write",
    toolset="orchestrator",
    schema=ORCHESTRATOR_PACKET_WRITE_SCHEMA,
    handler=_handle_orchestrator_packet_write,
    emoji="📦",
    description="Write a dispatch packet to the confined cc-packets directory.",
    check_fn=_orchestrator_profile_only,
)
