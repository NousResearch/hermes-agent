"""Shared utility functions for hermes-agent."""

import json
import logging
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Union
from urllib.parse import urlparse

import yaml

logger = logging.getLogger(__name__)


TRUTHY_STRINGS = frozenset({"1", "true", "yes", "on"})


def is_truthy_value(value: Any, default: bool = False) -> bool:
    """Coerce bool-ish values using the project's shared truthy string set."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in TRUTHY_STRINGS
    return bool(value)


def env_var_enabled(name: str, default: str = "") -> bool:
    """Return True when an environment variable is set to a truthy value."""
    return is_truthy_value(os.getenv(name, default), default=False)


def _preserve_file_mode(path: Path) -> "int | None":
    """Capture the permission bits of *path* if it exists, else ``None``."""
    try:
        return stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    except OSError:
        return None


def _restore_file_mode(path: Path, mode: "int | None") -> None:
    """Re-apply *mode* to *path* after an atomic replace.

    ``tempfile.mkstemp`` creates files with 0o600 (owner-only).  After
    ``os.replace`` swaps the temp file into place the target inherits
    those restrictive permissions, breaking Docker / NAS volume mounts
    that rely on broader permissions set by the user.  Calling this
    right after ``os.replace`` restores the original permissions.
    """
    if mode is None:
        return
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def atomic_replace(tmp_path: Union[str, Path], target: Union[str, Path]) -> str:
    """Atomically move *tmp_path* onto *target*, preserving symlinks.

    ``os.replace(tmp, target)`` atomically swaps ``tmp`` into place at
    ``target``.  When ``target`` is a symlink, the symlink itself is
    replaced with a regular file — silently detaching managed deployments
    that symlink ``config.yaml`` / ``SOUL.md`` / ``auth.json`` etc. from
    ``~/.hermes/`` to a git-tracked profile package or dotfiles repo
    (GitHub #16743).

    This helper resolves the symlink first so ``os.replace`` writes to
    the real file in-place while the symlink survives.  For non-symlink
    and non-existent paths the behavior is identical to a plain
    ``os.replace`` call.

    Returns the resolved real path used for the replace, so callers that
    need to re-apply permissions can target it instead of the symlink.
    """
    target_str = str(target)
    real_path = os.path.realpath(target_str) if os.path.islink(target_str) else target_str
    os.replace(str(tmp_path), real_path)
    return real_path


def atomic_json_write(
    path: Union[str, Path],
    data: Any,
    *,
    indent: int = 2,
    **dump_kwargs: Any,
) -> None:
    """Write JSON data to a file atomically.

    Uses temp file + fsync + os.replace to ensure the target file is never
    left in a partially-written state. If the process crashes mid-write,
    the previous version of the file remains intact.

    Args:
        path: Target file path (will be created or overwritten).
        data: JSON-serializable data to write.
        indent: JSON indentation (default 2).
        **dump_kwargs: Additional keyword args forwarded to json.dump(), such
            as default=str for non-native types.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    original_mode = _preserve_file_mode(path)

    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.stem}_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                indent=indent,
                ensure_ascii=False,
                **dump_kwargs,
            )
            f.flush()
            os.fsync(f.fileno())
        # Preserve symlinks — swap in-place on the real file (GitHub #16743).
        real_path = atomic_replace(tmp_path, path)
        _restore_file_mode(real_path, original_mode)
    except BaseException:
        # Intentionally catch BaseException so temp-file cleanup still runs for
        # KeyboardInterrupt/SystemExit before re-raising the original signal.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_yaml_write(
    path: Union[str, Path],
    data: Any,
    *,
    default_flow_style: bool = False,
    sort_keys: bool = False,
    extra_content: str | None = None,
) -> None:
    """Write YAML data to a file atomically.

    Uses temp file + fsync + os.replace to ensure the target file is never
    left in a partially-written state.  If the process crashes mid-write,
    the previous version of the file remains intact.

    Args:
        path: Target file path (will be created or overwritten).
        data: YAML-serializable data to write.
        default_flow_style: YAML flow style (default False).
        sort_keys: Whether to sort dict keys (default False).
        extra_content: Optional string to append after the YAML dump
            (e.g. commented-out sections for user reference).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    original_mode = _preserve_file_mode(path)

    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.stem}_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=default_flow_style, sort_keys=sort_keys)
            if extra_content:
                f.write(extra_content)
            f.flush()
            os.fsync(f.fileno())
        # Preserve symlinks — swap in-place on the real file (GitHub #16743).
        real_path = atomic_replace(tmp_path, path)
        _restore_file_mode(real_path, original_mode)
    except BaseException:
        # Match atomic_json_write: cleanup must also happen for process-level
        # interruptions before we re-raise them.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_roundtrip_yaml_update(
    path: Union[str, Path],
    key_path: str,
    value: Any,
) -> None:
    """Update one dotted YAML key while preserving comments and readable text.

    This is intentionally narrower than :func:`atomic_yaml_write`: it is for
    user-edited config files where comments, ordering, quoting, and Unicode
    should survive a single setting mutation.  Writes still use the same temp
    file + fsync + atomic replace pattern.
    """
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    yaml_rt = YAML(typ="rt")
    yaml_rt.preserve_quotes = True
    yaml_rt.allow_unicode = True
    yaml_rt.default_flow_style = False
    yaml_rt.indent(mapping=2, sequence=4, offset=2)

    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            config = yaml_rt.load(f) or CommentedMap()
    else:
        config = CommentedMap()

    if not isinstance(config, CommentedMap):
        config = CommentedMap(config)

    current = config
    keys = key_path.split(".")
    for key in keys[:-1]:
        next_value = current.get(key)
        if not isinstance(next_value, CommentedMap):
            next_value = CommentedMap()
            current[key] = next_value
        current = next_value
    current[keys[-1]] = value

    original_mode = _preserve_file_mode(path)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.stem}_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml_rt.dump(config, f)
            f.flush()
            os.fsync(f.fileno())
        real_path = atomic_replace(tmp_path, path)
        _restore_file_mode(real_path, original_mode)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ─── JSON Helpers ─────────────────────────────────────────────────────────────


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Parse JSON, returning *default* on any parse error.

    Replaces the ``try: json.loads(x) except (JSONDecodeError, TypeError)``
    pattern duplicated across display.py, anthropic_adapter.py,
    auxiliary_client.py, and others.
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return default


# ─── Environment Variable Helpers ─────────────────────────────────────────────


def env_int(key: str, default: int = 0) -> int:
    """Read an environment variable as an integer, with fallback."""
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except (ValueError, TypeError):
        return default


def env_bool(key: str, default: bool = False) -> bool:
    """Read an environment variable as a boolean."""
    return is_truthy_value(os.getenv(key, ""), default=default)


# ─── Proxy Helpers ────────────────────────────────────────────────────────────


_PROXY_ENV_KEYS = (
    "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
    "https_proxy", "http_proxy", "all_proxy",
)


def normalize_proxy_url(proxy_url: str | None) -> str | None:
    """Normalize proxy URLs for httpx/aiohttp compatibility.

    WSL/Clash-style environments often export SOCKS proxies as
    ``socks://127.0.0.1:PORT``. httpx rejects that alias and expects the
    explicit ``socks5://`` scheme instead.
    """
    candidate = str(proxy_url or "").strip()
    if not candidate:
        return None
    if candidate.lower().startswith("socks://"):
        return f"socks5://{candidate[len('socks://'):]}"
    return candidate


def normalize_proxy_env_vars() -> None:
    """Rewrite supported proxy env vars to canonical URL forms in-place."""
    for key in _PROXY_ENV_KEYS:
        value = os.getenv(key, "")
        normalized = normalize_proxy_url(value)
        if normalized and normalized != value:
            os.environ[key] = normalized


# ─── URL Parsing Helpers ──────────────────────────────────────────────────────


def base_url_hostname(base_url: str) -> str:
    """Return the lowercased hostname for a base URL, or ``""`` if absent.

    Use exact-hostname comparisons against known provider hosts
    (``api.openai.com``, ``api.x.ai``, ``api.anthropic.com``) instead of
    substring matches on the raw URL. Substring checks treat attacker- or
    proxy-controlled paths/hosts like ``https://api.openai.com.example/v1``
    or ``https://proxy.test/api.openai.com/v1`` as native endpoints, which
    leads to wrong api_mode / auth routing.
    """
    raw = (base_url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw else f"//{raw}")
    return (parsed.hostname or "").lower().rstrip(".")


def base_url_host_matches(base_url: str, domain: str) -> bool:
    """Return True when the base URL's hostname is ``domain`` or a subdomain.

    Safer counterpart to ``domain in base_url``, which is the substring
    false-positive class documented on ``base_url_hostname``. Accepts bare
    hosts, full URLs, and URLs with paths.

        base_url_host_matches("https://api.moonshot.ai/v1", "moonshot.ai") == True
        base_url_host_matches("https://moonshot.ai", "moonshot.ai")        == True
        base_url_host_matches("https://evil.com/moonshot.ai/v1", "moonshot.ai") == False
        base_url_host_matches("https://moonshot.ai.evil/v1", "moonshot.ai")     == False
    """
    hostname = base_url_hostname(base_url)
    if not hostname:
        return False
    domain = (domain or "").strip().lower().rstrip(".")
    if not domain:
        return False
    return hostname == domain or hostname.endswith("." + domain)
def chown_to_match_parent(path: Union[str, Path]) -> bool:
    """Chown *path* to match its parent directory's owner when running as root.

    Fixes a class of ownership bug seen with `docker exec` against containers
    where Hermes runs as a non-root user (default UID 10000): the privileged
    `docker exec` writes a state file (e.g. ``auth.json``) as ``root:root``,
    which the runtime user can no longer read.  By rewriting the file owner
    to match the parent directory immediately after the write, the runtime
    user keeps access regardless of which UID called ``hermes login``.

    No-op when:
      * Not running on POSIX (Windows ``os.chown`` does not exist).
      * Effective UID is not 0 (root) — non-root processes can only chown
        files they already own to themselves, which buys nothing.
      * The target itself is a symlink — see security note below.
      * The parent directory's owner already matches the file's owner.

    **Symlink safety (TOCTOU on auth.json).** ``os.stat`` and the default
    ``os.chown`` both follow symlinks. If an attacker replaces the target
    with a symlink to e.g. ``/etc/shadow`` between the atomic write and
    this call, a naive implementation would re-own the symlink's
    destination as root.  We defend in two layers:

      1. Use ``os.lstat`` (not ``os.stat``) on the target to read its
         on-disk metadata without following symlinks. If the target IS
         a symlink, return False without touching it — the helper only
         exists to reconcile a regular state file's ownership.
      2. Pass ``follow_symlinks=False`` to ``os.chown`` (or fall back to
         ``os.lchown`` on platforms where ``os.chown`` doesn't accept the
         keyword) so that even in the impossible-but-let's-be-paranoid
         case where a symlink slips through the lstat check, we still
         never re-own the link's destination.

    The parent-directory ``stat`` follows symlinks intentionally — root
    controls the parent, and resolving a parent-dir symlink mirrors the
    behavior of the surrounding write that just landed in the same
    directory.

    Errors are swallowed (and logged at debug) so a chown failure never
    breaks the surrounding write — the file mode is still 0o600, so the
    worst case is the pre-fix bug.

    Returns True when a chown was actually performed, False otherwise.
    """
    chown = getattr(os, "chown", None)
    geteuid = getattr(os, "geteuid", None)
    if chown is None or geteuid is None:  # Windows / non-POSIX
        return False
    try:
        if geteuid() != 0:
            return False
    except OSError:
        return False

    target = Path(path)
    try:
        # lstat: do NOT follow symlinks when inspecting the target.
        # If an attacker swapped auth.json for a symlink between the
        # atomic write and this call, st_mode tells us so.
        file_stat = os.lstat(target)
    except OSError:
        return False
    if stat.S_ISLNK(file_stat.st_mode):
        # Refuse to operate on symlinks. The caller wrote a regular file;
        # if a link is here now, something else is going on and we should
        # not chown either the link or its destination.
        logger.warning(
            "chown_to_match_parent: %s is a symlink, refusing to chown "
            "(possible TOCTOU substitution; see issue #15718).",
            target,
        )
        return False

    try:
        # Parent stat may follow symlinks: root controls the parent dir
        # and we want the same UID/GID that owns it on disk.
        parent_stat = os.stat(target.parent)
    except OSError:
        return False

    target_uid = parent_stat.st_uid
    target_gid = parent_stat.st_gid
    if file_stat.st_uid == target_uid and file_stat.st_gid == target_gid:
        return False

    try:
        # Belt-and-suspenders: even though we lstat-checked above, pass
        # follow_symlinks=False so a TOCTOU race between the lstat and
        # the chown still cannot re-own a link destination.  Fall back
        # to os.lchown when the platform's os.chown doesn't accept the
        # keyword (rare on modern Linux/macOS but possible elsewhere).
        if chown in os.supports_follow_symlinks:
            chown(target, target_uid, target_gid, follow_symlinks=False)
        elif hasattr(os, "lchown"):
            os.lchown(target, target_uid, target_gid)
        else:
            # No safe primitive available — refuse rather than risk
            # following a symlink.
            logger.debug(
                "chown_to_match_parent: no symlink-safe chown on this "
                "platform, skipping %s",
                target,
            )
            return False
    except OSError as exc:
        logger.debug("chown_to_match_parent: failed to chown %s: %s", target, exc)
        return False
    logger.warning(
        "Adjusted ownership of %s to %d:%d to match parent directory "
        "(file was written as root, but parent is owned by a non-root user). "
        "Run 'hermes login' as the runtime user to avoid this remediation.",
        target, target_uid, target_gid,
    )
    return True
