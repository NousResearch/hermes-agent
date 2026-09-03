"""Shared utility functions for hermes-agent."""

import errno
import json
import logging
import os
import shutil
import stat
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Union
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


def _preserve_file_owner(path: Path) -> "tuple[int, int] | None":
    """Capture the owning uid/gid of *path* if the platform supports it."""
    if os.name != "posix":
        return None
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_uid, st.st_gid


def _restore_file_owner(path: Path, owner: "tuple[int, int] | None") -> None:
    """Re-apply uid/gid after an atomic replace when permitted.

    Docker and NAS-backed installs often run some commands as root while the
    persistent volume is owned by the runtime user. ``os.replace`` swaps in the
    temp file's owner, so a root-run config write can leave ``config.yaml`` owned
    by root. Best-effort chown preserves the existing owner for privileged
    callers and is harmless for unprivileged callers that cannot chown.
    """
    if owner is None or not hasattr(os, "chown"):
        return
    try:
        os.chown(path, owner[0], owner[1])
    except OSError:
        pass


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


_IS_WINDOWS = os.name == "nt"

# Windows rename failures that can be caused by another handle on the target
# rather than by a permission problem.  ``os.replace`` onto a file that any
# other handle has open is denied because CPython opens files without
# ``FILE_SHARE_DELETE``:
#
#   5  ERROR_ACCESS_DENIED    — what a held *target* handle actually reports
#   32 ERROR_SHARING_VIOLATION — reported when the *source* temp file is held
#   33 ERROR_LOCK_VIOLATION    — byte-range lock on the target
#
# Measured on Windows 11 (build 26200, CPython 3.11): a plain reader on the
# target — in-process or cross-process — yields winerror 5, NOT 32.  Keying
# recovery on 32 alone therefore misses every real occurrence of this bug.
# These codes are ambiguous (a genuine ACL denial is also 5), which is why
# recovery is bounded and any still-failing write is re-raised unchanged
# rather than being classified up front.
_WINDOWS_CONTENDED_REPLACE_ERRORS = frozenset({5, 32, 33})

# Retry budget for the atomic rename.  A rename that wins here keeps the write
# fully atomic, so the budget is sized to cover a realistic contended hold: an
# observed desktop auth-init holds auth.json past 100 ms, while an ordinary
# status read is ~0.05 ms.  Measured on Windows 11 build 26200, this recovers
# holds up to ~200 ms atomically (~310 ms worst case).
#
# The cap matters as much as the attempt count.  gateway_state.json is
# rewritten at every turn boundary, so a permanently-held target pays the full
# budget on every write: a longer 6 x 20..400 ms budget cost ~1.3 s per write
# under a persistent reader, versus ~0.3 s here for the same atomic coverage.
# Jittered so concurrent writers don't retry in lockstep.
_REPLACE_RETRY_ATTEMPTS = 4
_REPLACE_RETRY_BASE_DELAY_S = 0.02
_REPLACE_RETRY_MAX_DELAY_S = 0.1


def _is_contended_windows_replace_error(exc: OSError) -> bool:
    """Return True for Windows rename failures a retry might clear.

    Only a *candidate* classification: ``ERROR_ACCESS_DENIED`` covers both a
    concurrent handle and a real ACL denial, and the two are not reliably
    distinguishable up front.  Probing the target with ``os.access`` does not
    work — ``os.replace`` needs delete-child rights on the *parent directory*,
    so a directory-level denial reports the target as writable.  Instead of
    guessing, both cases enter the same bounded retry and a genuine denial
    falls through to the caller with its original error.
    """
    return _IS_WINDOWS and getattr(exc, "winerror", None) in (
        _WINDOWS_CONTENDED_REPLACE_ERRORS
    )


def _rewrite_in_place(tmp_str: str, real_path: str) -> None:
    """Overwrite *real_path* with the contents of *tmp_str*, in place.

    Last-resort path for a target whose handle is still held after the retry
    budget: writing through the existing file works where renaming onto it
    does not.  Unlike ``shutil.copyfile`` this never truncates the target to
    zero first — a concurrent reader would otherwise be able to observe an
    empty ``auth.json`` / ``gateway_state.json`` mid-write (measured: a
    4-thread poller sees a 0-byte read during a plain copyfile).  A single
    ``os.write`` of the full payload followed by ``ftruncate`` keeps the
    visible content going straight from old to new.

    This is still not atomic — it is a strictly smaller window than a copy,
    not the absence of one — so it runs only after the rename has genuinely
    failed.  Writing through the target also preserves its ACL, which
    ``os.replace`` does not (the temp file's inherited ACL wins there).
    """
    with open(tmp_str, "rb") as src:
        data = src.read()
    flags = os.O_WRONLY | getattr(os, "O_BINARY", 0)
    fd = os.open(real_path, flags)
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        written = 0
        while written < len(data):
            written += os.write(fd, data[written:])
        os.ftruncate(fd, len(data))
        try:
            os.fsync(fd)
        except OSError:
            pass
    finally:
        os.close(fd)
    os.unlink(tmp_str)


def _copy_fallback(tmp_str: str, real_path: str) -> None:
    """Copy/fsync/unlink fallback for cross-device and bind-mount renames."""
    shutil.copyfile(tmp_str, real_path)
    try:
        shutil.copystat(tmp_str, real_path)
    except OSError:
        pass
    try:
        with open(real_path, "rb") as f:
            os.fsync(f.fileno())
    except OSError:
        pass
    os.unlink(tmp_str)


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
    ``os.replace`` call unless the rename fails with:

    * ``EXDEV`` / ``EBUSY`` (any platform) — cross-device, bind-mount, and
      busy-file deployments fall back to copy/fsync/unlink immediately.
      These never clear on retry.
    * A Windows rename contended by another open handle (winerror 5/32/33).
      CPython opens files without ``FILE_SHARE_DELETE``, so *any* concurrent
      reader of the target blocks the rename.  The rename is retried with
      jittered backoff first — a retry that wins keeps the write atomic —
      and only a target whose handle outlives the budget is rewritten in
      place, so the update lands instead of being silently dropped.

    A genuine Windows permission failure produces the same winerror as a
    contended one, so it is not classified up front: it exhausts the retry
    budget, fails the in-place rewrite too, and is re-raised unchanged.

    Returns the resolved real path used for the replace, so callers that
    need to re-apply permissions can target it instead of the symlink.
    """
    target_str = str(target)
    real_path = os.path.realpath(target_str) if os.path.islink(target_str) else target_str
    tmp_str = str(tmp_path)
    try:
        os.replace(tmp_str, real_path)
        return real_path
    except OSError as exc:
        contended = _is_contended_windows_replace_error(exc)
        if exc.errno not in (errno.EXDEV, errno.EBUSY) and not contended:
            raise
        if contended:
            # Lazy import: keeps ``utils`` free of a package-level dependency
            # on ``agent`` for every consumer that never hits this path.
            from agent.retry_utils import jittered_backoff

            for attempt in range(1, _REPLACE_RETRY_ATTEMPTS + 1):
                time.sleep(
                    jittered_backoff(
                        attempt,
                        base_delay=_REPLACE_RETRY_BASE_DELAY_S,
                        max_delay=_REPLACE_RETRY_MAX_DELAY_S,
                    )
                )
                try:
                    os.replace(tmp_str, real_path)
                    return real_path
                except OSError as retry_exc:
                    if retry_exc.errno in (errno.EXDEV, errno.EBUSY):
                        # Not contention after all — stop burning the budget.
                        exc = retry_exc
                        contended = False
                        break
                    if not _is_contended_windows_replace_error(retry_exc):
                        raise
                    exc = retry_exc
        logger.debug(
            "atomic_replace: %s -> %s failed with %s; falling back to %s",
            tmp_str,
            real_path,
            getattr(exc, "winerror", None)
            or errno.errorcode.get(exc.errno or 0, exc.errno),
            "in-place rewrite" if contended else "copy",
        )
        if contended:
            # Re-raises the rewrite's own error (not the rename's) when the
            # target is genuinely unwritable — an ACL denial stays an ACL
            # denial rather than being reported as contention.
            _rewrite_in_place(tmp_str, real_path)
        else:
            _copy_fallback(tmp_str, real_path)
    return real_path


def atomic_write_text(
    path: Union[str, Path],
    content: str,
    *,
    encoding: str = "utf-8",
    tmp_prefix: str = ".tmp_",
    preserve_mode: bool = False,
    create_mode: "int | None" = None,
) -> None:
    """Write *content* to *path* via temp file + fsync + atomic rename.

    Ensures the target file is never left in a partially-written state if
    the process crashes or is interrupted.  ``atomic_replace`` preserves
    symlinks and handles cross-device / busy-file fallbacks.

    Used by the memory store, skill manager, and agent importer so that
    every destructive file rewrite in the codebase shares one implementation.

    Args:
        preserve_mode: When True, carry an existing target's permission bits
            and (POSIX, best-effort) owner across the replace, like
            ``atomic_yaml_write`` does unconditionally.  ``os.replace`` swaps
            in mkstemp's 0600 temp file owned by the writing user, so without
            this a root-run rewrite of a user-owned file flips its owner and
            tightens its mode.  The mode is applied to the temp fd *before*
            the replace, so the file never transits through 0600.  Off by
            default: the historical callers (memory store, skill manager,
            cron) own their 0600-is-fine files.
        create_mode: Permission bits to apply when the target does not yet
            exist (otherwise the new file keeps mkstemp's 0600).  Never
            applied to an existing file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    original_mode = _preserve_file_mode(path) if preserve_mode else None
    original_owner = _preserve_file_owner(path) if preserve_mode else None
    effective_mode = original_mode
    if effective_mode is None and create_mode is not None and not path.exists():
        effective_mode = create_mode

    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), prefix=tmp_prefix, suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            if effective_mode is not None and hasattr(os, "fchmod"):
                # fchmod the temp fd BEFORE the replace so the target never
                # transits through mkstemp's 0600. fchmod is Unix-only; on
                # Windows the post-replace chmod below applies the mode.
                os.fchmod(handle.fileno(), effective_mode)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        real_path = atomic_replace(tmp_path, path)
        if preserve_mode:
            _restore_file_owner(Path(real_path), original_owner)
        if effective_mode is not None and not hasattr(os, "fchmod"):
            _restore_file_mode(Path(real_path), effective_mode)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_json_write(
    path: Union[str, Path],
    data: Any,
    *,
    indent: int = 2,
    mode: int | None = None,
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
        mode: Optional final permission mode. When set, the temp file is
            created and replaced with this mode, avoiding chmod-after-write
            TOCTOU exposure for secret-bearing files.
        **dump_kwargs: Additional keyword args forwarded to json.dump(), such
            as default=str for non-native types.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    original_mode = None if mode is not None else _preserve_file_mode(path)
    original_owner = _preserve_file_owner(path)

    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.stem}_",
        suffix=".tmp",
    )
    try:
        if mode is not None and hasattr(os, "fchmod"):
            # fchmod is Unix-only; Windows' os module has no fchmod. Skipping it
            # here is safe — mkstemp already created the temp file as 0o600, and
            # the post-replace os.chmod below applies the final mode durably.
            os.fchmod(fd, mode)
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
        real_path_obj = Path(real_path)
        _restore_file_owner(real_path_obj, original_owner)
        if mode is not None:
            try:
                os.chmod(real_path_obj, mode)
            except OSError:
                pass
        else:
            _restore_file_mode(real_path_obj, original_mode)
    except BaseException:
        # Intentionally catch BaseException so temp-file cleanup still runs for
        # KeyboardInterrupt/SystemExit before re-raising the original signal.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def warn_if_credential_file_broadly_readable(
    path: Union[str, Path],
    *,
    label: str = "",
    log: logging.Logger | None = None,
) -> bool:
    """Warn (once per call) when a credential file is group/world-readable.

    Secret-bearing files that users create by hand (or that older Hermes
    versions wrote without an explicit mode) commonly end up 0o644 under the
    default umask. This helper is the shared read-time check for that class:
    call it before loading any token/credential file so the owner gets a
    remediation hint in the logs.

    Returns True when a warning was emitted. No-ops (returns False) on
    platforms without POSIX permission bits semantics (best effort), when the
    file is missing, or when permissions are already tight.
    """
    p = Path(path)
    _log = log or logger
    try:
        file_mode = p.stat().st_mode
    except OSError:
        return False
    if os.name != "posix":
        # Windows ACLs don't map onto POSIX group/other bits; st_mode there
        # is synthesized and would false-positive.
        return False
    if not (file_mode & (stat.S_IRGRP | stat.S_IROTH)):
        return False
    _log.warning(
        "%s%s is group/world-readable (mode 0%o) and contains secrets. "
        "Run: chmod 600 %s",
        f"{label} " if label else "",
        p.name,
        stat.S_IMODE(file_mode),
        p,
    )
    return True


class IndentDumper(yaml.SafeDumper):
    """PyYAML dumper that indents list items under mapping keys (2-space).

    Default PyYAML emits "indentless" sequences — list items start at the
    same column as their parent mapping key.  ``ruamel.yaml`` (used by
    :func:`atomic_roundtrip_yaml_update`) emits 2-space-indented sequences.
    Mixing both styles in the same ``config.yaml`` produces a file that
    stricter parsers like ``js-yaml`` reject with ``bad indentation of a
    mapping entry``.  Forcing ``indentless=False`` aligns the two
    serializers so all write paths emit byte-identical layouts (#31999).
    """

    def increase_indent(self, flow=False, indentless=False):  # noqa: ARG002
        return super().increase_indent(flow, False)


def atomic_yaml_write(
    path: Union[str, Path],
    data: Any,
    *,
    default_flow_style: bool = False,
    sort_keys: bool = False,
    extra_content: str | None = None,
    create_mode: "int | None" = None,
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
        create_mode: Permission bits to apply when the target does not yet
            exist (a created file otherwise keeps mkstemp's 0600).  Never
            applied to an existing file, whose mode is always preserved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    original_mode = _preserve_file_mode(path)
    original_owner = _preserve_file_owner(path)
    if original_mode is None and create_mode is not None and not path.exists():
        original_mode = create_mode

    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.stem}_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            if original_mode is not None and hasattr(os, "fchmod"):
                # Apply the mode to the temp fd BEFORE the replace so the
                # target never transits through mkstemp's 0600 (the
                # post-replace _restore_file_mode below then re-applies it
                # harmlessly, and remains the sole path on Windows).
                os.fchmod(f.fileno(), original_mode)
            # allow_unicode=True writes emoji/kaomoji (e.g. personalities, skin
            # cursors) as real UTF-8 instead of fragile escape sequences. Without
            # it, PyYAML emits astral-plane chars as `\UXXXXXXXX` (8-digit) escapes
            # inside multi-line double-quoted strings wrapped with `\`
            # continuations — a structure that stricter/non-PyYAML parsers and
            # hand-edits routinely break into unclosed quotes, corrupting the whole
            # config (GitHub #51356).
            yaml.dump(
                data,
                f,
                Dumper=IndentDumper,
                default_flow_style=default_flow_style,
                sort_keys=sort_keys,
                allow_unicode=True,
            )
            if extra_content:
                f.write(extra_content)
            f.flush()
            os.fsync(f.fileno())
        # Preserve symlinks — swap in-place on the real file (GitHub #16743).
        real_path = atomic_replace(tmp_path, path)
        real_path_obj = Path(real_path)
        _restore_file_owner(real_path_obj, original_owner)
        _restore_file_mode(real_path_obj, original_mode)
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
    # Honor escaped dots and prefer existing literal dotted keys (e.g. model
    # IDs like ``glm-5.3``) over blind splitting — same navigation as
    # ``hermes config set``'s ``_set_nested`` (#91607: /model + TUI
    # persistence route through here and used to write ``glm-5: {'3': ...}``
    # phantom siblings while the runtime kept reading the literal key).
    from hermes_cli.config import _greedy_literal_match, _split_key_path

    keys = _split_key_path(key_path)
    i = 0
    while True:
        remaining = keys[i:]
        seg, consumed = remaining[0], 1
        match = _greedy_literal_match(dict(current), remaining)
        if match is not None:
            seg, consumed = match
        if i + consumed == len(keys):
            current[seg] = value
            break
        next_value = current.get(seg)
        if not isinstance(next_value, CommentedMap):
            next_value = CommentedMap()
            current[seg] = next_value
        current = next_value
        i += consumed

    original_mode = _preserve_file_mode(path)
    original_owner = _preserve_file_owner(path)
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
        real_path_obj = Path(real_path)
        _restore_file_owner(real_path_obj, original_owner)
        _restore_file_mode(real_path_obj, original_mode)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_roundtrip_yaml_mutate(path: Union[str, Path], mutate: Callable[[Any], None]) -> None:
    """Atomically mutate a round-trip YAML document without replacing siblings.

    The shared config lock covers the entire read-modify-write sequence, so a
    concurrent ``save_config`` cannot overwrite a change between parsing the
    existing document and atomically replacing it.
    """
    from hermes_cli.config import _CONFIG_LOCK

    with _CONFIG_LOCK:
        _atomic_roundtrip_yaml_mutate_unlocked(path, mutate)


def _atomic_roundtrip_yaml_mutate_unlocked(path: Union[str, Path], mutate: Callable[[Any], None]) -> None:
    """Implementation of :func:`atomic_roundtrip_yaml_mutate`; caller holds the config lock."""
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
    mutate(config)

    original_mode = _preserve_file_mode(path)
    original_owner = _preserve_file_owner(path)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.stem}_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml_rt.dump(config, f)
            f.flush()
            os.fsync(f.fileno())
        real_path = atomic_replace(tmp_path, path)
        real_path_obj = Path(real_path)
        _restore_file_owner(real_path_obj, original_owner)
        _restore_file_mode(real_path_obj, original_mode)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_roundtrip_yaml_save(
    path: Union[str, Path],
    new_state: dict,
) -> None:
    """Persist a full config-state dict while preserving comments and ordering.

    Behaves like ``atomic_yaml_write`` (writes the whole file in one shot from
    ``new_state``), but routes through ruamel.yaml round-trip mode so existing
    comments, key order, quotes, and readable Unicode survive.

    Reconciliation rules against the on-disk YAML:

    * Keys present in both are updated in-place via assignment, which keeps
      ruamel's CommentedMap anchors (and their attached comments) attached to
      their original positions.
    * Keys missing from ``new_state`` are deleted.
    * Keys added in ``new_state`` are appended at the end of their parent map.
    * Nested ``dict`` values recurse with the same rules.
    * Non-dict values (lists, scalars) are overwritten wholesale — list
      element comments are not individually preserved, matching ruamel's
      semantics.

    This is the comment-safe replacement for ``yaml.safe_dump(cfg, f)`` in
    callers that mutate a deep-loaded config dict and want to persist the
    whole thing.

    Shares the fail-closed contract ``hermes_cli.config.atomic_config_write``
    enforces for plain (non-comment-preserving) full-document writes: an
    existing-but-unreadable ``config.yaml`` (permission error, broken mount,
    transient I/O) raises rather than being silently replaced with only
    ``new_state``. Imported lazily to avoid a module-level circular import —
    ``hermes_cli.config`` itself imports from this module.
    """
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap
    from ruamel.yaml.scalarstring import DoubleQuotedScalarString

    from hermes_cli.config import require_readable_config_before_write

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require_readable_config_before_write(path)

    yaml_rt = YAML(typ="rt")
    yaml_rt.preserve_quotes = True
    yaml_rt.allow_unicode = True
    yaml_rt.default_flow_style = False
    yaml_rt.indent(mapping=2, sequence=4, offset=2)

    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            existing = yaml_rt.load(f)
        if not isinstance(existing, CommentedMap):
            existing = CommentedMap(existing or {})
    else:
        existing = CommentedMap()

    # ruamel's round-trip dumper resolves plain scalars against the YAML 1.2
    # core schema, where only true/false/null are reserved words — so a plain
    # python str like "off" or "yes" is emitted unquoted. Every other config
    # reader in this codebase (atomic_config_write's PyYAML path, yaml.safe_load
    # call sites, etc.) parses under YAML 1.1 rules, where on/off/yes/no are
    # boolean synonyms. Without forcing quotes here, a freshly written
    # `approvals.mode: off` silently round-trips back as `False` under
    # yaml.safe_load. Force-quote any new string value that YAML 1.1 would
    # otherwise misparse as bool/null.
    _YAML11_AMBIGUOUS_WORDS = {
        "y", "n", "yes", "no", "true", "false", "on", "off", "null", "~",
    }

    def _quote_if_yaml11_ambiguous(value):
        if isinstance(value, str) and value.lower() in _YAML11_AMBIGUOUS_WORDS:
            return DoubleQuotedScalarString(value)
        return value

    def _merge(dst: CommentedMap, src: dict) -> None:
        # Update / recurse into keys present in src.
        for key, value in src.items():
            if isinstance(value, dict):
                current = dst.get(key)
                if not isinstance(current, CommentedMap):
                    current = CommentedMap()
                    dst[key] = current
                _merge(current, value)
            else:
                dst[key] = _quote_if_yaml11_ambiguous(value)
        # Delete keys missing from src — preserves "explicit absence" semantics
        # of the old _save_cfg(cfg) pattern (e.g. cfg.pop("custom_prompt", None)
        # then _save_cfg must actually remove the key from disk).
        for key in [k for k in dst.keys() if k not in src]:
            del dst[key]

    _merge(existing, new_state)

    original_mode = _preserve_file_mode(path)
    original_owner = _preserve_file_owner(path)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.stem}_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml_rt.dump(existing, f)
            f.flush()
            os.fsync(f.fileno())
        real_path = atomic_replace(tmp_path, path)
        real_path_obj = Path(real_path)
        _restore_file_owner(real_path_obj, original_owner)
        _restore_file_mode(real_path_obj, original_mode)
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


# ── Fast YAML loading ────────────────────────────────────────────────────
#
# PyYAML's pure-Python SafeLoader is ~8x slower than the libyaml-backed
# ``CSafeLoader`` C extension. Startup parses config.yaml and every plugin
# manifest with the slow path, costing ~0.9s of cold-start time. The C loader
# is a true drop-in for ``safe_load`` (same restricted tag set), so prefer it
# and fall back to the pure-Python loader only when libyaml isn't compiled in.
_fast_yaml_loader = None


def _get_fast_yaml_loader():
    global _fast_yaml_loader
    if _fast_yaml_loader is None:
        _fast_yaml_loader = getattr(yaml, "CSafeLoader", None) or yaml.SafeLoader
    return _fast_yaml_loader


def fast_safe_load(stream: Any) -> Any:
    """``yaml.safe_load`` using the libyaml C loader when available.

    Accepts the same inputs as ``yaml.safe_load`` (a ``str``/``bytes`` document
    or a readable file object) and returns the same parsed structure. Falls
    back to PyYAML's pure-Python ``SafeLoader`` when ``CSafeLoader`` isn't
    available, so behavior is identical everywhere — only the speed differs.
    """
    return yaml.load(stream, Loader=_get_fast_yaml_loader())


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


def env_float(key: str, default: float = 0.0) -> float:
    """Read an environment variable as a float, with fallback."""
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
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


# ─── Model Capability Detection ──────────────────────────────────────────────


def model_forces_max_completion_tokens(model: str) -> bool:
    """Return True for model families that require ``max_completion_tokens``.

    OpenAI's newer families reject ``max_tokens`` on /v1/chat/completions with
    HTTP 400 ``unsupported_parameter`` — the caller must send
    ``max_completion_tokens`` instead. This covers:

    - ``gpt-4o`` / ``gpt-4o-mini`` / ``gpt-4o-*``
    - ``gpt-4.1`` / ``gpt-4.1-*``
    - ``gpt-5`` / ``gpt-5.x`` / ``gpt-5-*``
    - ``o1`` / ``o1-*``
    - ``o3`` / ``o3-*``
    - ``o4`` / ``o4-*``

    Handles vendor prefixes like ``openai/gpt-5.4`` by stripping to the tail.
    The URL-based check (``base_url_hostname == "api.openai.com"``) misses
    third-party OpenAI-compatible endpoints (custom OpenAI gateways,
    OpenRouter) that front these models and enforce the same parameter
    constraint, so name-based detection is required as a fallback.
    """
    m = (model or "").strip().lower()
    if not m:
        return False
    if "/" in m:
        m = m.rsplit("/", 1)[-1]
    return (
        m.startswith("gpt-4o")
        or m.startswith("gpt-4.1")
        or m.startswith("gpt-5")
        or m.startswith("o1")
        or m.startswith("o3")
        or m.startswith("o4")
    )


def base_url_origin(base_url: str) -> tuple[str, str, int]:
    """Return ``(scheme, hostname, effective_port)`` for a base URL.

    Origin, not just host. ``https://h/v1`` and ``http://h/v1`` are different
    trust boundaries, and so are two ports on the same host, so any decision
    about handing a bearer secret to a new URL has to compare all three —
    hostname equality alone would authorise an HTTPS→HTTP downgrade.

    The port is normalised to the scheme default (443/80) when absent, so
    ``https://h`` and ``https://h:443`` compare equal. Returns
    ``("", "", 0)`` when the URL yields no usable hostname or a bad port.
    """
    raw = (base_url or "").strip()
    if not raw:
        return ("", "", 0)
    parsed = urlparse(raw if "://" in raw else f"//{raw}")
    scheme = (parsed.scheme or "").lower()
    hostname = (parsed.hostname or "").lower().rstrip(".")
    if not hostname:
        return ("", "", 0)
    try:
        port = parsed.port
    except ValueError:
        # Out-of-range or non-numeric port — not a usable origin.
        return ("", "", 0)
    if port is None:
        port = {"https": 443, "http": 80}.get(scheme, 0)
    return (scheme, hostname, port)


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
