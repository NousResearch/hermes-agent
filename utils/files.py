"""Files on disk: atomic writes and replaces, temp files, directory sync, cleanup, file mode and
owner, file signatures for caches, and the JSON and YAML file helpers."""

import errno
import hashlib
import json
import logging
import os
import random
import shutil
import stat
import tempfile
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import hermes_yaml as yaml

# Named for the package, not this module: log configs and log readers know the ``utils`` logger.
logger = logging.getLogger("utils")


def file_signature(st: os.stat_result) -> tuple[int, int, int, int]:
    """Change-detection key for a stat result: ``(st_mtime_ns, st_size, st_ino, st_ctime_ns)``.

    mtime + size alone miss a replacement that preserves both (``cp -p``, ``rsync -t``, a tar
    restore, a script pinning the timestamp with ``os.utime``). The inode changes on an atomic
    replace and ctime cannot be backdated from user space, so the pair catches those writers.
    On Windows ``st_ino`` may be 0 and ``st_ctime_ns`` is the creation time — both stable across
    an in-place rewrite, so the key degrades to mtime + size there rather than misfiring.
    """
    return (st.st_mtime_ns, st.st_size, st.st_ino, st.st_ctime_ns)


# Filesystem timestamps tick coarsely (~4 ms; whole seconds on some filesystems), so a write in the
# same tick as a stat leaves the signature unchanged. A signature this fresh is "racily clean" (as in
# git's index) and must not be trusted as "nothing changed since".
RACY_MTIME_WINDOW_NS = 2_000_000_000


def mtime_is_racy(mtime_ns: int) -> bool:
    """True while ``mtime_ns`` is too recent to rule out a same-tick write after the stat."""
    return time.time_ns() - mtime_ns < RACY_MTIME_WINDOW_NS


def content_digest(data: bytes) -> str:
    """Short digest of ``data`` for cache keys (not a security primitive)."""
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def _file_digest(path: str | Path) -> str:
    """:func:`content_digest` of the file's bytes. A file that stats but cannot be read here
    (permission, removed since the stat, a directory) gets ``<ErrorName>``, which no hex digest
    matches: the key differs from any cached one, so the caller's own read runs and meets the error
    in its normal recovery, as it did when the key was stat-only and never opened the file."""
    try:
        with open(path, "rb") as f:
            return content_digest(f.read())
    except OSError as exc:
        return f"<{type(exc).__name__}>"


def racy_content_digest(path: str | Path, sig: tuple[int, int, int, int]) -> str:
    """``""`` once the :func:`file_signature` ``sig`` has settled; while it is racy, a digest of the
    file's bytes (:func:`_file_digest`, so it never raises). A cache keyed on a stat signature appends
    it, because two same-size in-place writes in one timestamp tick share mtime, ctime, size and inode
    (kernels without multigrain timestamps). Racy is judged on the later of mtime and ctime: userland
    can restore an older mtime after a write (``cp -p``, ``rsync --inplace -t``), never ctime. Read
    before the parse it guards: a write in between then costs one re-parse instead of pinning the
    older content."""
    if not mtime_is_racy(max(sig[0], sig[3])):
        return ""
    return _file_digest(path)


def file_cache_version(path: str | Path, st: os.stat_result | None = None, *,
                       always_digest: bool = False) -> tuple[Any, ...]:
    """Key for anything derived from a file's content: :func:`file_signature` plus a content digest.

    The digest is :func:`racy_content_digest` (empty once the file is 2 s old), which suits a cache
    re-validated on every read: a racily-stored entry costs one re-parse when the file settles.
    ``always_digest`` hashes on every call, for a signature stored and compared much later (a
    change detector whose false "changed" restarts something, or a "still the broken file" record),
    where the racy digest turning empty would read as a change. Raises ``OSError`` only from the
    stat (none given ``st``): a file that cannot be read gets the error's name as its digest
    (:func:`_file_digest`)."""
    sig = file_signature(Path(path).stat() if st is None else st)
    if always_digest:
        return (*sig, _file_digest(path))
    return (*sig, racy_content_digest(path, sig))


def file_has_bytes(path: str | Path, data: bytes) -> bool:
    """Whether ``path`` already holds exactly ``data``: a writer skips its atomic replace then,
    since a fresh inode reads as a change to every stat-keyed detector (gateway credential
    reconcile, config caches) and restarts what they guard."""
    try:
        return Path(path).read_bytes() == data
    except OSError:
        return False


def preserve_file_mode(path: Path) -> int | None:
    """Permission bits of *path* if it exists, else ``None``."""
    try:
        return stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    except OSError:
        return None


def preserve_file_owner(path: Path) -> tuple[int, int] | None:
    """Owning ``(uid, gid)`` of *path* on POSIX, else ``None``."""
    try:
        st = path.stat() if os.name == "posix" else None
    except OSError:
        return None
    return (st.st_uid, st.st_gid) if st else None


def _restore_file_metadata(path: Path, owner: tuple[int, int] | None, mode: int | None) -> None:
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


def default_new_file_mode() -> int | None:
    """The mode ``open(path, "w")`` gives a file it has to create (``0o666 & ~umask``); ``None``
    when the umask cannot be read or on non-POSIX hosts (Windows mode bits are synthesized).

    ``mkstemp`` always creates at 0600, so publishing a *new* non-secret file through a temp
    file would tighten it to owner-only — the Docker/NAS volume-mount hazard
    :func:`_restore_file_metadata` documents. The transient mask is 0o077: a thread that opens
    a file in the read window gets a tighter file, never a looser one.
    """
    if os.name != "posix":
        return None
    try:
        current = os.umask(0o077)
        os.umask(current)
    except OSError:
        return None
    return 0o666 & ~current


def restore_file_owner(path: Path, owner: tuple[int, int] | None) -> None:
    _restore_file_metadata(path, owner, None)


def restore_file_mode(path: Path, mode: int | None) -> None:
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


def _replace_retry_delay(attempt: int) -> float:
    """Seconds to wait before contended-rename retry *attempt* (1-based).

    Exponential from the base, capped at the max, plus up to 50% jitter. ``SystemRandom`` draws
    from the OS, so concurrent writers (forked ones included) decorrelate without touching the
    global ``random`` state.
    """
    delay = min(_REPLACE_RETRY_BASE_DELAY_S * 2 ** (attempt - 1), _REPLACE_RETRY_MAX_DELAY_S)
    return delay + random.SystemRandom().uniform(0, 0.5 * delay)


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


def atomic_replace(tmp_path: str | Path, target: str | Path) -> str:
    """Atomically move *tmp_path* onto *target*, preserving symlinks.

    Resolves a symlink first so ``os.replace`` writes the real file in place and the symlink
    survives. Otherwise identical to ``os.replace`` unless the rename fails with EXDEV/EBUSY
    (cross-device, bind-mount, busy file: copy/fsync/unlink immediately — these never clear on
    retry) or a Windows rename contended by another open handle (winerror 5/32/33: bounded retry,
    then in-place rewrite).
    """
    target_str = str(target)
    real_path = _publish_path(target_str)
    tmp_str = str(tmp_path)
    try:
        os.replace(tmp_str, real_path)
        return real_path
    except OSError as exc:
        contended = _is_contended_windows_replace_error(exc)
        if exc.errno not in _CROSS_DEVICE_ERRNOS and not contended:
            raise
        if contended:
            for attempt in range(1, _REPLACE_RETRY_ATTEMPTS + 1):
                time.sleep(_replace_retry_delay(attempt))
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


def _publish_path(target_str: str) -> str:
    """The path :func:`atomic_replace` renames onto: a symlink's real file, else the target itself."""
    return os.path.realpath(target_str) if os.path.islink(target_str) else target_str


def mkstemp_beside(target: str | Path, **kw: Any) -> tuple[int, str]:
    """``tempfile.mkstemp`` in the directory :func:`atomic_replace` will rename into.

    A temp staged next to a symlink whose target lives on another filesystem turns the publish
    rename into EXDEV, and atomic_replace's copy fallback then rewrites the file in place (torn on
    a crash). Staging beside the resolved target keeps the rename atomic. If that directory is not
    writable to us (the file itself may still be), stage beside the link instead: the save keeps
    working through the non-atomic copy fallback, exactly as before.
    """
    target_str = str(target)
    link_dir = str(Path(target_str).parent)
    stage_dir = os.path.dirname(_publish_path(target_str)) or link_dir
    try:
        return tempfile.mkstemp(dir=stage_dir, **kw)
    except PermissionError:
        if stage_dir == link_dir:
            raise
        return tempfile.mkstemp(dir=link_dir, **kw)


def fsync_directory(path: str | Path) -> None:
    """Best-effort fsync of a directory entry so a just-renamed file survives power loss.

    No-op on Windows (directories can't be opened with ``os.open``; the file fsync still applies)
    and on any OSError — durability of the directory entry is never worth failing a write that
    has already been replaced into place.
    """
    if os.name == "nt":
        return
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        with suppress(OSError):
            os.fsync(fd)
    finally:
        os.close(fd)


def unlink_files_older_than(directory: str | Path, pattern: str, max_age_seconds: float, *,
                            now: float | None = None) -> int:
    """Delete regular files in *directory* matching *pattern* whose mtime is more than
    *max_age_seconds* before *now* (default: the current time); return how many were removed.

    The one age-based cache sweep. Never raises: a missing directory, or an entry that cannot be
    stat'ed or unlinked, is skipped (the next sweep retries it).
    """
    cutoff = (time.time() if now is None else now) - max_age_seconds
    removed = 0
    with suppress(OSError):
        for path in Path(directory).glob(pattern):
            with suppress(OSError):
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
    return removed


def rmtree_readonly(path: str | Path, *, ignore_errors: bool = False) -> None:
    """``shutil.rmtree`` that can also delete read-only trees.

    ``shutil.rmtree`` stops at the first entry it cannot unlink.  Git marks
    loose object files read-only on Windows (``WinError 5``), and package
    installs (Nix store, deb/rpm) are copied ``r--r--r--`` into ``0555``
    directories on POSIX, where unlinking needs a writable *parent*.  Clear the
    write bit on the failing path and on its parent, then retry the exact
    operation that failed.  Only ``PermissionError`` is retried: every other
    failure keeps ``shutil.rmtree``'s semantics (and ``ignore_errors``).
    """

    def _on_error(func, fpath, exc_info):
        # ``onerror`` (3.11) passes ``exc_info``, ``onexc`` (3.12+) the exception.
        exc = exc_info[1] if isinstance(exc_info, tuple) else exc_info
        if not isinstance(exc, PermissionError):
            raise exc
        for candidate in (os.path.dirname(fpath), fpath):
            if candidate:
                with suppress(OSError):
                    os.chmod(candidate, os.stat(candidate).st_mode | stat.S_IWUSR | stat.S_IXUSR)
        func(fpath)

    try:
        try:
            shutil.rmtree(path, onexc=_on_error)
        except TypeError:  # ``onexc`` is 3.12+; 3.11 only knows ``onerror``
            shutil.rmtree(path, onerror=_on_error)
    except OSError:
        # ``ignore_errors`` still gets the read-only recovery; it only swallows
        # whatever is left after the retry.
        if not ignore_errors:
            raise


def _atomic_write(path: Path, write, *, prefix: str, encoding: str = "utf-8", mode: int | None = None,
                  preserve_owner: bool = True, binary: bool = False, fsync_dir: bool = False) -> None:
    """Temp file + fsync + :func:`atomic_replace`, then re-apply owner/mode.

    *write(f)* emits the payload into the open handle (text, or bytes when *binary*). The temp file
    is created by ``mkstemp`` — ``O_CREAT|O_EXCL`` at 0600 regardless of umask — so a secret is
    never readable at process umask, not even between create and chmod. *mode* is fchmod'd onto
    the temp fd BEFORE the replace so the target never transits through mkstemp's 0600 (fchmod is
    Unix-only; the post-replace chmod is the sole path on Windows). With no *mode* a NEW target
    gets what ``open(path, "w")`` would have given it (process umask) — the callers this replaced
    wrote at umask, and silently tightening every fresh cache/state file to 0600 breaks shared
    volume mounts; an existing target with no *mode* keeps mkstemp's bits, as before. *fsync_dir*
    also fsyncs the resolved target's parent so the rename itself is durable. The temp file is
    removed on any failure — ``BaseException`` on purpose, so KeyboardInterrupt / SystemExit still
    clean up.
    """
    # A profile delete leaves a tombstone beside its removed home.  Background
    # writers may retain that home in a context variable, so a plain mkdir here
    # would resurrect the profile before the write can fail.
    from hermes_constants import mkdir_under_hermes_home

    mkdir_under_hermes_home(path.parent)
    if mode is None and not path.exists():
        mode = default_new_file_mode()
    original_owner = preserve_file_owner(path) if preserve_owner else None
    fd, tmp_path = mkstemp_beside(path, prefix=prefix, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb" if binary else "w", encoding=None if binary else encoding) as f:
            if mode is not None and hasattr(os, "fchmod"):
                os.fchmod(f.fileno(), mode)
            write(f)
            f.flush()
            os.fsync(f.fileno())
        replaced = Path(atomic_replace(tmp_path, path))  # symlink-preserving actual destination
        _restore_file_metadata(replaced, original_owner, mode)
        if fsync_dir:
            fsync_directory(replaced.parent)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_path)
        raise


def _mode_for_write(path: Path, create_mode: int | None, preserve: bool = True) -> int | None:
    """Existing permission bits of *path* (when *preserve*), else *create_mode* for a new file."""
    mode = preserve_file_mode(path) if preserve else None
    return mode if mode is not None or path.exists() else create_mode


def atomic_write_text(path: str | Path, content: str, *, encoding: str = "utf-8", tmp_prefix: str = ".tmp_",
                      preserve_mode: bool = False, create_mode: int | None = None, mode: int | None = None,
                      fsync_dir: bool = False) -> None:
    """Write *content* to *path* via temp file + fsync + atomic rename.

    The target is never left partially written on crash/interrupt. Shared by every destructive
    file rewrite (memory store, skill manager, agent importer, ...). *mode* forces the final
    permission bits (secret files: ``0o600``) regardless of what exists; *create_mode* applies only
    when the target is new and *preserve_mode* carries an existing file's bits and owner across.
    """
    path = Path(path)
    _atomic_write(path, lambda f: f.write(content), prefix=tmp_prefix, encoding=encoding,
                  mode=mode if mode is not None else _mode_for_write(path, create_mode, preserve=preserve_mode),
                  preserve_owner=preserve_mode, fsync_dir=fsync_dir)


def atomic_write_bytes(path: str | Path, content: bytes, *, tmp_prefix: str = ".tmp_",
                       mode: int | None = None, fsync_dir: bool = False) -> None:
    """Bytes variant of :func:`atomic_write_text` (encrypted blobs, key material)."""
    path = Path(path)
    _atomic_write(path, lambda f: f.write(content), prefix=tmp_prefix, binary=True, preserve_owner=False,
                  mode=mode if mode is not None else preserve_file_mode(path), fsync_dir=fsync_dir)


def _dump_json(data: Any, f, *, indent: int | None, ensure_ascii: bool, dump_kwargs: dict) -> None:
    """``json.dump`` that survives surrogate-escaped strings.

    ``os.fsdecode`` of a non-UTF-8 filename/argv yields lone surrogates (``'\\udcff'``); a utf-8
    text handle rejects them with ``UnicodeEncodeError`` — a ValueError, which callers guarding
    ``except OSError`` never see. ``ensure_ascii=True`` escapes them as ``\\udcff`` and
    ``json.loads`` restores the identical str, so the retry round-trips; ``surrogateescape``
    would emit a raw 0xFF byte that the reader's utf-8 decode rejects. Serializing to a str first
    keeps the failure before any byte reaches the file, so no partial payload is left behind.
    """
    text = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii, **dump_kwargs)
    try:
        f.write(text)
    except UnicodeEncodeError:
        f.write(json.dumps(data, indent=indent, ensure_ascii=True, **dump_kwargs))


def atomic_json_write(
    path: str | Path, data: Any, *, indent: int = 2, mode: int | None = None,
    ensure_ascii: bool = False, fsync_dir: bool = False, **dump_kwargs: Any,
) -> None:
    """Write JSON to *path* atomically (temp file + fsync + replace).

    Surrogate-escaped strings (non-UTF-8 argv/paths) are always persisted: the write falls back
    to ``ensure_ascii=True`` escapes for that payload only, so normal content keeps its raw UTF-8
    bytes. ``mode=0o600`` is the private-credential form: the temp file is 0600 from creation
    (mkstemp), so the payload is never umask-readable.
    """
    path = Path(path)
    _atomic_write(path, lambda f: _dump_json(data, f, indent=indent, ensure_ascii=ensure_ascii, dump_kwargs=dump_kwargs),
                  prefix=f".{path.stem}_", mode=mode if mode is not None else preserve_file_mode(path),
                  fsync_dir=fsync_dir)


def read_json_or_empty(path: str | Path) -> dict:
    """The JSON object at *path*, or ``{}`` when the file is missing, unreadable, malformed or
    not an object. The read half of every ``read → merge → atomic_json_write`` config store
    (memory-provider ``save_config``), so a corrupt sidecar degrades to defaults instead of
    taking the provider down."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))  # utf-8-sig: a Windows-editor BOM must not wipe the config
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def warn_if_credential_file_broadly_readable(path: str | Path, *, label: str = "", log: logging.Logger | None = None) -> bool:
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


def atomic_yaml_write(path: str | Path, data: Any, *, default_flow_style: bool = False, sort_keys: bool = False,
                      extra_content: str | None = None, create_mode: int | None = None) -> None:
    """Write YAML to *path* atomically (temp file + fsync + replace)."""
    path = Path(path)

    def _write(f) -> None:
        yaml.safe_dump(data, f, default_flow_style=default_flow_style, sort_keys=sort_keys)
        if extra_content:
            f.write(extra_content)

    _atomic_write(path, _write, prefix=f".{path.stem}_", mode=_mode_for_write(path, create_mode))


def _roundtrip_load(path: Path):
    """``(yaml_rt, CommentedMap)``: a ruamel round-trip loader keeping quotes/Unicode with 2-space
    indents, plus *path* loaded through it (empty map when missing/blank)."""
    from ruamel.yaml.comments import CommentedMap

    yaml_rt = yaml.roundtrip_yaml()
    data = yaml_rt.load(path.read_text(encoding="utf-8")) if path.exists() else None
    return yaml_rt, data if isinstance(data, CommentedMap) else CommentedMap(data or {})


def _roundtrip_dump(path: Path, yaml_rt, config, *, extra_content: str | None = None) -> None:
    def _write(f) -> None:
        yaml_rt.dump(config, f)
        if extra_content:
            f.write(extra_content)

    _atomic_write(path, _write, prefix=f".{path.stem}_", mode=preserve_file_mode(path))


def atomic_roundtrip_yaml_update(path: str | Path, key_path: str, value: Any) -> None:
    """Update one dotted YAML key while preserving comments, ordering, quoting and Unicode.

    Narrower than :func:`atomic_yaml_write` on purpose: for user-edited config files where a
    single setting mutation must not disturb the rest. Still writes via temp file + atomic replace.
    ``value=None`` removes the key (a ``key: null`` leftover reads as absent everywhere but
    litters the file and diverges from whole-document writers that drop the key).
    """
    from ruamel.yaml.comments import CommentedMap
    # Honor escaped dots and prefer existing literal dotted keys (model IDs like ``glm-5.3``) over
    # blind splitting — same navigation as ``hermes config set``'s ``_set_nested``; otherwise
    # /model + TUI persistence wrote ``glm-5: {'3': ...}`` phantom siblings.
    # See #91607.
    from hermes_cli.config import _greedy_literal_match, _split_key_path

    path = Path(path)
    from hermes_constants import mkdir_under_hermes_home

    mkdir_under_hermes_home(path.parent)
    yaml_rt, config = _roundtrip_load(path)
    current = config
    keys = _split_key_path(key_path)
    i = 0
    while True:
        remaining = keys[i:]
        seg, consumed = _greedy_literal_match(dict(current), remaining) or (remaining[0], 1)
        if i + consumed == len(keys):
            if value is None:
                current.pop(seg, None)
            else:
                current[seg] = value
            break
        next_value = current.get(seg)
        if not isinstance(next_value, CommentedMap):
            if value is None:
                return  # nothing to remove under a missing/scalar parent
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


def _rt_value(value: Any) -> Any:
    """Plain Python value → ruamel node: YAML 1.1-ambiguous strings force-quoted at every depth
    (a bare ``off`` inside a list reads back as ``False`` just like one at the top level)."""
    from ruamel.yaml.comments import CommentedMap, CommentedSeq
    from ruamel.yaml.scalarstring import DoubleQuotedScalarString

    if isinstance(value, dict):
        node = CommentedMap()
        for k, v in value.items():
            node[k] = _rt_value(v)
        return node
    if isinstance(value, (list, tuple)):
        return CommentedSeq(_rt_value(v) for v in value)
    if isinstance(value, str) and value.lower() in _YAML11_AMBIGUOUS_WORDS:
        return DoubleQuotedScalarString(value)
    return value


def atomic_roundtrip_yaml_save(path: str | Path, new_state: dict, *,
                               extra_content_on_create: str | None = None) -> None:
    """Persist a full config-state dict while preserving comments and ordering.

    THE on-disk primitive for ``config.yaml`` (production callers reach it through
    ``hermes_cli.config.atomic_config_write`` or ``atomic_config_replace``): the document is
    loaded through ruamel round-trip mode and *new_state* is merged onto it, so comments, key order, quotes, blank
    lines and readable Unicode survive. Only nodes whose value actually changed are reassigned;
    an untouched scalar or list keeps its inline comments and formatting. Keys absent from
    *new_state* are deleted ("explicit absence": ``cfg.pop(k)`` + save removes ``k`` from disk).
    ``extra_content_on_create`` (commented example blocks) is appended only when the file is
    being created — re-appending it on every rewrite is how the stock boilerplate replaced
    users' own comments (#92554).
    """
    from ruamel.yaml.comments import CommentedMap, CommentedSeq
    from hermes_cli.config import require_readable_config_before_write

    path = Path(path)
    from hermes_constants import mkdir_under_hermes_home

    mkdir_under_hermes_home(path.parent)
    require_readable_config_before_write(path)
    creating = not path.exists() or not path.read_text(encoding="utf-8").strip()
    yaml_rt, existing = _roundtrip_load(path)

    def _unchanged(current: Any, value: Any) -> bool:
        # ``True == 1`` in Python; a bool↔int flip is a real change for YAML readers.
        return current == value and isinstance(current, bool) is isinstance(value, bool)

    def _merge_seq(dst: CommentedSeq, src: list) -> None:
        # Element-wise so appending/editing one entry keeps the comments on its siblings.
        for i, value in enumerate(src):
            if i < len(dst):
                _merge_item(dst, i, value)
            else:
                dst.append(_rt_value(value))
        del dst[len(src):]

    def _merge_item(dst, key, value) -> None:
        current = dst[key]
        if isinstance(value, dict) and isinstance(current, CommentedMap):
            _merge(current, value)
        elif isinstance(value, list) and isinstance(current, CommentedSeq):
            _merge_seq(current, value)
        elif not _unchanged(current, value):
            dst[key] = _rt_value(value)
        # else: unchanged — keep the existing node and the comments/quoting attached to it

    def _merge(dst: CommentedMap, src: dict) -> None:
        for key, value in src.items():
            if key in dst:
                _merge_item(dst, key, value)
            else:
                dst[key] = _rt_value(value)
        for key in [k for k in dst if k not in src]:
            del dst[key]

    _merge(existing, new_state)
    _roundtrip_dump(path, yaml_rt, existing, extra_content=extra_content_on_create if creating else None)


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Parse JSON, returning *default* on any parse error (a too-deeply nested body included)."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError, RecursionError):
        return default


def fast_safe_load(stream: Any) -> Any:
    """Use the shared safe reader (which selects ruamel's C parser when available)."""
    return yaml.safe_load(stream)


_YAML_FILE_CACHE: dict = {}
_YAML_FILE_CACHE_LOCK = threading.Lock()


def load_yaml_file_readonly(path: str | Path) -> Any:
    """``fast_safe_load`` of a file, re-parsed only when its :func:`file_cache_version` changes.

    Returns the cached object itself — callers must never mutate it. Parse errors propagate and
    are not cached; a missing file raises ``FileNotFoundError`` like ``open`` does."""
    path = Path(path)
    sig = file_cache_version(path)
    key = str(path)
    with _YAML_FILE_CACHE_LOCK:
        cached = _YAML_FILE_CACHE.get(key)
        if cached is not None and cached[0] == sig:
            return cached[1]
    with open(path, encoding="utf-8") as f:
        data = fast_safe_load(f)
    with _YAML_FILE_CACHE_LOCK:
        _YAML_FILE_CACHE[key] = (sig, data)
    return data
