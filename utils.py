"""Shared utility functions for hermes-agent."""

import errno
import json
import logging
import os
import shutil
import stat
import tempfile
import time
from contextlib import suppress
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
    if isinstance(value, str):
        return value.strip().lower() in TRUTHY_STRINGS
    return bool(value)


def env_var_enabled(name: str, default: str = "") -> bool:
    """Return True when an environment variable is set to a truthy value."""
    return is_truthy_value(os.getenv(name, default), default=False)


def _preserve_file_mode(path: Path) -> "int | None":
    """Permission bits of *path* if it exists, else ``None``."""
    try:
        return stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    except OSError:
        return None


def _preserve_file_owner(path: Path) -> "tuple[int, int] | None":
    """Owning ``(uid, gid)`` of *path* on POSIX, else ``None``."""
    try:
        st = path.stat() if os.name == "posix" else None
    except OSError:
        return None
    return (st.st_uid, st.st_gid) if st else None


def _restore_file_metadata(path: Path, owner: "tuple[int, int] | None", mode: "int | None") -> None:
    """Best-effort re-apply of uid/gid and permission bits after an atomic replace.

    Docker/NAS installs often run some commands as root on a volume owned by the runtime user;
    ``os.replace`` swaps in the temp file's owner, so privileged callers chown it back. ``mkstemp``
    creates 0o600 files; without re-applying *mode* the target would inherit that and break
    volume mounts relying on broader permissions.
    """
    if owner is not None and hasattr(os, "chown"):
        with suppress(OSError):
            os.chown(path, owner[0], owner[1])
    if mode is not None:
        with suppress(OSError):
            os.chmod(path, mode)


def _restore_file_owner(path: Path, owner: "tuple[int, int] | None") -> None:
    _restore_file_metadata(path, owner, None)


def _restore_file_mode(path: Path, mode: "int | None") -> None:
    _restore_file_metadata(path, None, mode)


_IS_WINDOWS = os.name == "nt"
# Windows rename failures possibly caused by another handle on the target. CPython opens files
# without FILE_SHARE_DELETE, so ``os.replace`` onto an open file is denied with 5 ERROR_ACCESS_DENIED
# (what a held *target* handle actually reports — measured: a plain reader yields 5, NOT 32),
# 32 ERROR_SHARING_VIOLATION (the *source* temp file is held) or 33 ERROR_LOCK_VIOLATION (byte-range
# lock on the target). Ambiguous (a real ACL denial is also 5), so recovery is bounded and a
# still-failing write is re-raised unchanged rather than classified up front.
_WINDOWS_CONTENDED_REPLACE_ERRORS = frozenset({5, 32, 33})
# Retry budget for the atomic rename. A rename that wins here keeps the write fully atomic, so the
# budget covers a realistic hold (desktop auth-init holds auth.json >100 ms): ~200 ms recovered
# atomically, ~310 ms worst case. The cap matters as much as the count — gateway_state.json is
# rewritten every turn, so a permanently-held target pays the full budget per write. Jittered so
# concurrent writers don't retry in lockstep.
_REPLACE_RETRY_ATTEMPTS = 4
_REPLACE_RETRY_BASE_DELAY_S = 0.02
_REPLACE_RETRY_MAX_DELAY_S = 0.1
_CROSS_DEVICE_ERRNOS = (errno.EXDEV, errno.EBUSY)


def _is_contended_windows_replace_error(exc: OSError) -> bool:
    """Candidate-only: winerror 5 also covers a genuine ACL denial."""
    return _IS_WINDOWS and getattr(exc, "winerror", None) in _WINDOWS_CONTENDED_REPLACE_ERRORS


def _rewrite_in_place(tmp_str: str, real_path: str) -> None:
    """Overwrite *real_path* through the existing file — last resort for a still-held target.

    Not atomic (a smaller window than a copy, not none), so it runs only after the rename has
    genuinely failed. Writing through the target also preserves its ACL, which ``os.replace``
    does not (the temp file's inherited ACL wins there).
    """
    with open(tmp_str, "rb") as src:
        data = src.read()
    fd = os.open(real_path, os.O_WRONLY | getattr(os, "O_BINARY", 0))
    try:
        written = 0
        while written < len(data):
            written += os.write(fd, data[written:])
        os.ftruncate(fd, len(data))
        with suppress(OSError):
            os.fsync(fd)
    finally:
        os.close(fd)
    os.unlink(tmp_str)


def _copy_fallback(tmp_str: str, real_path: str) -> None:
    """Copy/fsync/unlink fallback for cross-device and bind-mount renames."""
    shutil.copyfile(tmp_str, real_path)
    with suppress(OSError):
        shutil.copystat(tmp_str, real_path)
    with suppress(OSError), open(real_path, "rb") as f:
        os.fsync(f.fileno())
    os.unlink(tmp_str)


def atomic_replace(tmp_path: Union[str, Path], target: Union[str, Path]) -> str:
    """Atomically move *tmp_path* onto *target*, preserving symlinks.

    Resolves a symlink first so ``os.replace`` writes the real file in place and the symlink
    survives. Otherwise identical to ``os.replace`` unless the rename fails with EXDEV/EBUSY
    (cross-device, bind-mount, busy file: copy/fsync/unlink immediately — these never clear on
    retry) or a Windows rename contended by another open handle (winerror 5/32/33: bounded retry,
    then in-place rewrite).
    """
    target_str = str(target)
    real_path = os.path.realpath(target_str) if os.path.islink(target_str) else target_str
    tmp_str = str(tmp_path)
    try:
        os.replace(tmp_str, real_path)
        return real_path
    except OSError as exc:
        contended = _is_contended_windows_replace_error(exc)
        if exc.errno not in _CROSS_DEVICE_ERRNOS and not contended:
            raise
        if contended:
            # Lazy: keeps ``utils`` free of a package-level dependency on ``agent``.
            from agent.retry_utils import jittered_backoff
            for attempt in range(1, _REPLACE_RETRY_ATTEMPTS + 1):
                time.sleep(jittered_backoff(attempt, base_delay=_REPLACE_RETRY_BASE_DELAY_S, max_delay=_REPLACE_RETRY_MAX_DELAY_S))
                try:
                    os.replace(tmp_str, real_path)
                    return real_path
                except OSError as retry_exc:
                    exc = retry_exc
                    if retry_exc.errno in _CROSS_DEVICE_ERRNOS:
                        contended = False  # not contention after all — stop burning the budget
                        break
                    if not _is_contended_windows_replace_error(retry_exc):
                        raise
        logger.debug("atomic_replace: %s -> %s failed with %s; falling back to %s", tmp_str, real_path,
                     getattr(exc, "winerror", None) or errno.errorcode.get(exc.errno or 0, exc.errno),
                     "in-place rewrite" if contended else "copy")
        # The rewrite re-raises its own error, so an ACL denial is reported as such, not as contention.
        (_rewrite_in_place if contended else _copy_fallback)(tmp_str, real_path)
    return real_path


def _atomic_write(path: Path, write, *, prefix: str, encoding: str = "utf-8", mode: "int | None" = None, preserve_owner: bool = True) -> None:
    """Temp file + fsync + :func:`atomic_replace`, then re-apply owner/mode.

    *write(f)* emits the payload into the open text handle. *mode* is fchmod'd onto the temp fd
    BEFORE the replace so the target never transits through mkstemp's 0600 (fchmod is Unix-only;
    the post-replace chmod is the sole path on Windows). The temp file is removed on any failure —
    ``BaseException`` on purpose, so KeyboardInterrupt / SystemExit still clean up.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    original_owner = _preserve_file_owner(path) if preserve_owner else None
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=prefix, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            if mode is not None and hasattr(os, "fchmod"):
                os.fchmod(f.fileno(), mode)
            write(f)
            f.flush()
            os.fsync(f.fileno())
        _restore_file_metadata(Path(atomic_replace(tmp_path, path)), original_owner, mode)  # symlink-preserving
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_path)
        raise


def _mode_for_write(path: Path, create_mode: "int | None", preserve: bool = True) -> "int | None":
    """Existing permission bits of *path* (when *preserve*), else *create_mode* for a new file."""
    mode = _preserve_file_mode(path) if preserve else None
    return mode if mode is not None or path.exists() else create_mode


def atomic_write_text(path: Union[str, Path], content: str, *, encoding: str = "utf-8", tmp_prefix: str = ".tmp_",
                      preserve_mode: bool = False, create_mode: "int | None" = None, mode: "int | None" = None) -> None:
    """Write *content* to *path* via temp file + fsync + atomic rename.

    The target is never left partially written on crash/interrupt. Shared by every destructive
    file rewrite (memory store, skill manager, agent importer, ...).
    """
    path = Path(path)
    _atomic_write(path, lambda f: f.write(content), prefix=tmp_prefix, encoding=encoding,
                  mode=mode if mode is not None else _mode_for_write(path, create_mode, preserve=preserve_mode),
                  preserve_owner=preserve_mode)


def open_private_append(
    path: Union[str, Path],
    *,
    encoding: str = "utf-8",
    mode: int = 0o600,
    tighten_existing: bool = False,
):
    """Append text, applying exact ``mode`` only to a newly created inode.

    Exclusive creation distinguishes a new inode from an existing one, so
    ``mode`` lands on files this call creates and an existing file keeps its
    own permissions. ``fchmod`` on the new fd applies the mode exactly rather
    than umask-narrowed, matching ``atomic_write_text(create_mode=...)``:
    managed installs need the group-write bit of ``0o660`` to survive an
    interactive user's ``0o022`` umask, or the service UID cannot append to a
    transcript the user's UID started.

    Symlinks are followed, as ``atomic_replace`` does (#16743) -- a managed
    deployment may symlink an artifact path into a profile package, including
    before its target exists. POSIX modes are advisory on Windows.
    Internal writers may opt into tightening owned regular files before append;
    explicit outputs keep their existing permissions. Linked or foreign-owned
    files are reported rather than chmodded through.
    """
    def _create(target: str, flags: int) -> int:
        fd = os.open(target, flags, mode)
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(fd, mode)
        except BaseException:
            os.close(fd)
            raise
        return fd

    def _opener(target: str, flags: int) -> int:
        try:
            return _create(target, flags | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            pass
        try:
            existing_flags = flags & ~(os.O_CREAT | os.O_EXCL)
            if not tighten_existing or os.name != "posix":
                return os.open(target, existing_flags)
            try:
                fd = os.open(target, existing_flags | os.O_NOFOLLOW)
            except OSError as exc:
                if exc.errno != errno.ELOOP:
                    raise
                logger.warning("Private artifact %s is symlinked; verify its target permissions", target)
                return os.open(target, existing_flags)
            try:
                tighten_private_fd(fd, mode, target)
            except BaseException:
                os.close(fd)
                raise
            return fd
        except FileNotFoundError:
            # A dangling symlink: O_EXCL reports EEXIST for the link itself,
            # but reopening without O_CREAT hits the missing target. Create
            # through it, as a plain open would, at the requested mode.
            return _create(target, flags | os.O_CREAT)

    return open(Path(path), "a", encoding=encoding, opener=_opener)


def tighten_private_fd(fd: int, mode: int, path: Union[str, Path]) -> None:
    """Tighten an owned artifact inode; never chmod foreign or multiply-linked files."""
    if os.name != "posix":
        return
    info = os.fstat(fd)
    if not info.st_mode & 0o777 & ~mode:
        return
    if info.st_uid != os.geteuid() or (stat.S_ISREG(info.st_mode) and info.st_nlink != 1):  # windows-footgun: ok — POSIX-only above
        logger.warning("Private artifact %s has broad permissions and cannot safely be tightened; review its ownership and links", path)
        return
    if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
        raise OSError(errno.EINVAL, "Not a regular private artifact", str(path))
    os.fchmod(fd, (stat.S_IMODE(info.st_mode) & mode) | (info.st_mode & stat.S_ISGID))


def atomic_json_write(path: Union[str, Path], data: Any, *, indent: int = 2, mode: int | None = None, **dump_kwargs: Any) -> None:
    """Write JSON to *path* atomically (temp file + fsync + replace)."""
    path = Path(path)
    _atomic_write(path, lambda f: json.dump(data, f, indent=indent, ensure_ascii=False, **dump_kwargs),
                  prefix=f".{path.stem}_", mode=mode if mode is not None else _preserve_file_mode(path))


def warn_if_credential_file_broadly_readable(path: Union[str, Path], *, label: str = "", log: logging.Logger | None = None) -> bool:
    """Warn when a credential file is group/world-readable; True when a warning was emitted.

    Hand-made secret files (or ones older Hermes wrote without an explicit mode) commonly end up
    0o644 under the default umask; call this before loading any token/credential file. No-op on
    non-POSIX (Windows ACLs don't map onto group/other bits; st_mode there is synthesized), when
    the file is missing, or when permissions are already tight.
    """
    p = Path(path)
    try:
        file_mode = p.stat().st_mode
    except OSError:
        return False
    if os.name != "posix" or not (file_mode & (stat.S_IRGRP | stat.S_IROTH)):
        return False
    (log or logger).warning("%s%s is group/world-readable (mode 0%o) and contains secrets. Run: chmod 600 %s",
                            f"{label} " if label else "", p.name, stat.S_IMODE(file_mode), p)
    return True


class IndentDumper(yaml.SafeDumper):
    """PyYAML dumper that indents list items under mapping keys (2-space).

    PyYAML emits "indentless" sequences while ruamel (:func:`atomic_roundtrip_yaml_update`)
    indents them; mixing both in one ``config.yaml`` makes stricter parsers like ``js-yaml``
    reject it, so every write path is forced to the same shape.

    Forcing ``indentless=False`` aligns the two serializers so all write paths emit byte-identical layouts
    (#31999).
    """

    def increase_indent(self, flow=False, indentless=False):  # noqa: ARG002
        return super().increase_indent(flow, False)


def atomic_yaml_write(path: Union[str, Path], data: Any, *, default_flow_style: bool = False, sort_keys: bool = False,
                      extra_content: str | None = None, create_mode: "int | None" = None) -> None:
    """Write YAML to *path* atomically (temp file + fsync + replace)."""
    path = Path(path)

    def _write(f) -> None:
        # allow_unicode=True writes emoji/kaomoji as real UTF-8. Without it PyYAML emits astral
        # chars as `\UXXXXXXXX` escapes inside `\`-continued double-quoted strings — a structure
        # stricter parsers and hand-edits routinely break into unclosed quotes, corrupting the config.
        yaml.dump(data, f, Dumper=IndentDumper, default_flow_style=default_flow_style, sort_keys=sort_keys, allow_unicode=True)
        if extra_content:
            f.write(extra_content)

    _atomic_write(path, _write, prefix=f".{path.stem}_", mode=_mode_for_write(path, create_mode))


def _roundtrip_load(path: Path):
    """``(yaml_rt, CommentedMap)``: a ruamel round-trip loader keeping quotes/Unicode with 2-space
    indents, plus *path* loaded through it (empty map when missing/blank)."""
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap

    yaml_rt = YAML(typ="rt")
    yaml_rt.preserve_quotes = True
    yaml_rt.allow_unicode = True
    yaml_rt.default_flow_style = False
    yaml_rt.indent(mapping=2, sequence=4, offset=2)
    data = yaml_rt.load(path.read_text(encoding="utf-8")) if path.exists() else None
    return yaml_rt, data if isinstance(data, CommentedMap) else CommentedMap(data or {})


def _roundtrip_dump(path: Path, yaml_rt, config) -> None:
    _atomic_write(path, lambda f: yaml_rt.dump(config, f), prefix=f".{path.stem}_", mode=_preserve_file_mode(path))


def atomic_roundtrip_yaml_update(path: Union[str, Path], key_path: str, value: Any) -> None:
    """Update one dotted YAML key while preserving comments, ordering, quoting and Unicode.

    Narrower than :func:`atomic_yaml_write` on purpose: for user-edited config files where a
    single setting mutation must not disturb the rest. Still writes via temp file + atomic replace.
    """
    from ruamel.yaml.comments import CommentedMap
    # Honor escaped dots and prefer existing literal dotted keys (model IDs like ``glm-5.3``) over
    # blind splitting — same navigation as ``hermes config set``'s ``_set_nested``; otherwise
    # /model + TUI persistence wrote ``glm-5: {'3': ...}`` phantom siblings.
    # See #91607.
    from hermes_cli.config import _greedy_literal_match, _split_key_path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    yaml_rt, config = _roundtrip_load(path)
    current = config
    keys = _split_key_path(key_path)
    i = 0
    while True:
        remaining = keys[i:]
        seg, consumed = _greedy_literal_match(dict(current), remaining) or (remaining[0], 1)
        if i + consumed == len(keys):
            current[seg] = value
            break
        next_value = current.get(seg)
        if not isinstance(next_value, CommentedMap):
            next_value = CommentedMap()
            current[seg] = next_value
        current = next_value
        i += consumed
    _roundtrip_dump(path, yaml_rt, config)


# ruamel's round-trip dumper resolves plain scalars under YAML 1.2, where only true/false/null are
# reserved — so a str like "off" or "yes" is emitted unquoted. Every other config reader here
# (PyYAML, yaml.safe_load sites) parses under YAML 1.1, where on/off/yes/no are booleans: an
# unquoted ``approvals.mode: off`` would silently round-trip back as ``False``.
_YAML11_AMBIGUOUS_WORDS = frozenset({"y", "n", "yes", "no", "true", "false", "on", "off", "null", "~"})


def atomic_roundtrip_yaml_save(path: Union[str, Path], new_state: dict) -> None:
    """Persist a full config-state dict while preserving comments and ordering.

    Comment-safe replacement for ``yaml.safe_dump(cfg, f)``: writes the whole file from
    ``new_state`` through ruamel round-trip mode so existing comments, key order, quotes and
    readable Unicode survive.
    """
    from ruamel.yaml.comments import CommentedMap
    from ruamel.yaml.scalarstring import DoubleQuotedScalarString
    from hermes_cli.config import require_readable_config_before_write

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require_readable_config_before_write(path)
    yaml_rt, existing = _roundtrip_load(path)

    def _merge(dst: CommentedMap, src: dict) -> None:
        for key, value in src.items():
            if isinstance(value, dict):
                current = dst.get(key)
                if not isinstance(current, CommentedMap):
                    current = CommentedMap()
                    dst[key] = current
                _merge(current, value)
            elif isinstance(value, str) and value.lower() in _YAML11_AMBIGUOUS_WORDS:
                dst[key] = DoubleQuotedScalarString(value)
            else:
                dst[key] = value
        # Keys missing from src are deleted: ``cfg.pop("custom_prompt")`` then save must remove
        # the key from disk ("explicit absence" semantics of the old _save_cfg pattern).
        for key in [k for k in dst if k not in src]:
            del dst[key]

    _merge(existing, new_state)
    _roundtrip_dump(path, yaml_rt, existing)


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Parse JSON, returning *default* on any parse error."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return default


# libyaml's CSafeLoader is ~8x faster than the pure-Python SafeLoader and a true drop-in for
# ``safe_load`` (same restricted tag set); startup parses config.yaml and every plugin manifest,
# so the slow path cost ~0.9 s of cold start.
_fast_yaml_loader = getattr(yaml, "CSafeLoader", None) or yaml.SafeLoader


def fast_safe_load(stream: Any) -> Any:
    """``yaml.safe_load`` (same inputs, same result) using the libyaml C loader when available."""
    return yaml.load(stream, Loader=_fast_yaml_loader)


def _env_number(key: str, default, cast):
    raw = os.getenv(key, "").strip()
    try:
        return cast(raw) if raw else default
    except (ValueError, TypeError):
        return default


def env_int(key: str, default: int = 0) -> int:
    """Read an environment variable as an integer, with fallback."""
    return _env_number(key, default, int)


def env_float(key: str, default: float = 0.0) -> float:
    """Read an environment variable as a float, with fallback."""
    return _env_number(key, default, float)


def env_bool(key: str, default: bool = False) -> bool:
    """Read an environment variable as a boolean."""
    return is_truthy_value(os.getenv(key, ""), default=default)


_PROXY_ENV_KEYS = ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy")


def normalize_proxy_url(proxy_url: str | None) -> str | None:
    """Normalize proxy URLs for httpx/aiohttp: WSL/Clash export ``socks://``, httpx needs ``socks5://``."""
    candidate = str(proxy_url or "").strip()
    if candidate.lower().startswith("socks://"):
        return f"socks5://{candidate[len('socks://'):]}"
    return candidate or None


def normalize_proxy_env_vars() -> None:
    """Rewrite supported proxy env vars to canonical URL forms in-place."""
    for key in _PROXY_ENV_KEYS:
        value = os.getenv(key, "")
        normalized = normalize_proxy_url(value)
        if normalized and normalized != value:
            os.environ[key] = normalized


def _parse_base_url(base_url: str):
    """``urlparse`` that tolerates a bare ``host[:port][/path]`` (no scheme)."""
    raw = (base_url or "").strip()
    return urlparse(raw if "://" in raw else f"//{raw}") if raw else None


def _hostname_of(parsed) -> str:
    return (parsed.hostname or "").lower().rstrip(".") if parsed else ""


def base_url_hostname(base_url: str) -> str:
    """Lowercased hostname for a base URL, or ``""`` if absent.

    Compare exact hostnames against provider hosts instead of substring-matching the raw URL:
    ``https://api.openai.com.example/v1`` or ``https://proxy.test/api.openai.com/v1`` would
    otherwise pass as native endpoints and mis-route api_mode and auth.
    """
    return _hostname_of(_parse_base_url(base_url))


def model_forces_max_completion_tokens(model: str) -> bool:
    """True for OpenAI families that reject ``max_tokens`` (HTTP 400 ``unsupported_parameter``)."""
    m = (model or "").strip().lower().rsplit("/", 1)[-1]
    return m.startswith(("gpt-4o", "gpt-4.1", "gpt-5", "o1", "o3", "o4"))


def base_url_origin(base_url: str) -> tuple[str, str, int]:
    """``(scheme, hostname, effective_port)`` for a base URL; ``("", "", 0)`` on no host/bad port.

    Origin, not just host: ``https://h`` vs ``http://h`` and two ports on one host are different
    trust boundaries, so handing a bearer secret to a new URL must compare all three — hostname
    alone would authorise an HTTPS→HTTP downgrade. Port defaults to 443/80 so ``https://h``
    equals ``https://h:443``.
    """
    parsed = _parse_base_url(base_url)
    hostname = _hostname_of(parsed)
    if not hostname:
        return ("", "", 0)
    scheme = (parsed.scheme or "").lower()
    try:
        port = parsed.port
    except ValueError:  # out-of-range or non-numeric port — not a usable origin
        return ("", "", 0)
    return (scheme, hostname, {"https": 443, "http": 80}.get(scheme, 0) if port is None else port)


def base_url_host_matches(base_url: str, domain: str) -> bool:
    """True when the base URL's hostname is ``domain`` or a subdomain.

    Safer than ``domain in base_url`` (``evil.com/moonshot.ai`` / ``moonshot.ai.evil`` must not
    match). Accepts bare hosts, full URLs, and URLs with paths.
    """
    hostname = base_url_hostname(base_url)
    domain = (domain or "").strip().lower().rstrip(".")
    return bool(hostname and domain) and (hostname == domain or hostname.endswith("." + domain))
