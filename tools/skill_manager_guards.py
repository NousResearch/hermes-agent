"""Write/delete guards for ``skill_manage``. Every guard returns ``None`` when the
operation may proceed, else a refusal (error dict or message). Origin-owned state
(``_find_skill``, ``_skills_dir``) is reached lazily via ``tools.skill_manager_tool``
so test patches keep working."""

import contextvars as _ctxvars
import functools
import hashlib
import itertools
import logging
import os
import stat
import threading
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("tools.skill_manager_tool")


def _refusal(message: str, **extra: Any) -> dict[str, Any]:
    return {"success": False, "error": message, **extra}


def _is_background_review() -> bool:
    """True inside the autonomous curator review fork, and on any lookup failure: every guard keyed
    on this is stricter for the fork, so an unknown origin gets the fork's guards."""
    try:
        from tools.skill_provenance import is_background_review
        return bool(is_background_review())
    except Exception:
        logger.warning("write-origin lookup failed; applying background-review guards", exc_info=True)
        return True


def _is_confirmed_background_review() -> bool:
    """True only when the origin lookup succeeds and says background review. Moving a read mark vouches
    for bytes the fork will write over without re-reading, so an unknown origin must never move one
    (unlike the guards above, where unknown means stricter)."""
    try:
        from tools.skill_provenance import is_background_review
        return bool(is_background_review())
    except Exception:
        logger.warning("write-origin lookup failed; not moving background-review read marks", exc_info=True)
        return False


def _resolved_str(path: Path) -> str:
    with suppress(Exception):
        return str(path.resolve())
    return str(path)


def _file_sha256(path: Path) -> str:
    """SHA-256 hex digest of ``path``'s current bytes; raises OSError when it cannot be read or is not
    a regular file. Opened non-blocking and checked before reading: a FIFO or device swapped in for a
    skill file must not hang the guard while it holds the skill lock."""
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0))
    with open(fd, "rb", closefd=True) as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise OSError(f"{path} is not a regular file")
        return hashlib.file_digest(stream, "sha256").hexdigest()


class _BackgroundReviewReadMarks:
    """Read marks shared by copied tool contexts within one review run: resolved path -> (SHA-256 of
    the bytes the fork was served, generation). The generation is unique per update, so a
    compare-and-set tells "the update I made" from a later update that recorded the same digest."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._marks: dict[str, tuple[str, int]] = {}
        self._generations = itertools.count(1)

    def set(self, path: str, digest: str) -> None:
        with self._lock:
            self._marks[path] = (digest, next(self._generations))

    def get(self, path: str) -> Optional[str]:
        with self._lock:
            entry = self._marks.get(path)
        return None if entry is None else entry[0]

    def entry(self, path: str) -> Optional[tuple[str, int]]:
        with self._lock:
            return self._marks.get(path)

    def compare_and_set(self, path: str, expected: Optional[tuple[str, int]],
                        digest: Optional[str]) -> tuple[bool, Optional[tuple[str, int]]]:
        """If the mark is still exactly ``expected`` (same update), record ``digest`` as a new update,
        or drop the mark when ``digest`` is None. Returns (applied, the mark now)."""
        with self._lock:
            current = self._marks.get(path)
            if current != expected:
                return False, current
            if digest is None:
                self._marks.pop(path, None)
                return True, None
            self._marks[path] = new = (digest, next(self._generations))
            return True, new


_background_review_read_paths: _ctxvars.ContextVar[Optional[_BackgroundReviewReadMarks]] = (
    _ctxvars.ContextVar("background_review_read_paths", default=None))


def mark_background_review_skill_read(path: Path, digest: Optional[str] = None) -> None:
    """Record that the active background-review fork has read a skill file. The fork must not
    patch content it only inferred from the transcript: skill_view/read_file call this, and
    the write guards require the mark. ``digest`` is the SHA-256 hex of the bytes the caller
    served; without one, the file's current bytes are hashed (an unreadable file is not marked)."""
    if not _is_background_review():
        return
    if digest is None:
        try:
            digest = _file_sha256(path)
        except OSError:
            logger.debug("could not hash %s for its background-review read mark", path, exc_info=True)
            return
    if (marks := _background_review_read_paths.get()) is None:
        _background_review_read_paths.set(marks := _BackgroundReviewReadMarks())
    marks.set(_resolved_str(path), digest)


def _read_mark(path: Path) -> Optional[str]:
    marks = _background_review_read_paths.get()
    return None if marks is None else marks.get(_resolved_str(path))


def _background_review_has_read(path: Path) -> bool:
    return _read_mark(path) is not None


def _read_mark_entry(path: Path) -> Optional[tuple[str, int]]:
    """The exact current mark update for ``path`` (pass it to the refresh after a write). None
    unless the write is confirmed to be the background review's: a foreground write, even in a
    context that shares the fork's marks, never moves them (the fork has not read what it wrote),
    and neither does a write whose origin cannot be determined; the refresh then does nothing."""
    if not _is_confirmed_background_review():
        return None
    marks = _background_review_read_paths.get()
    return None if marks is None else marks.entry(_resolved_str(path))


def _refresh_background_review_read_mark(path: Path, expected: Optional[tuple[str, int]],
                                          published: Optional[bytes]) -> None:
    """After the fork's own write: what it knows is now the bytes it wrote, so a second write to
    the same file needs no re-read. ``published`` is exactly those bytes; the mark records them, never
    what is on disk afterwards: a writer outside the skill lock (an editor, a sync) can change the file
    between the atomic publish and this refresh, and its bytes must stay unseen, so the next write
    is refused as stale. ``expected`` is the mark captured (under the skill lock) BEFORE the write was
    published; only that exact update moves, so a read another worker recorded since stands. Only a
    live mark moves (a create is not a read). ``published=None`` means the fork removed the file: the
    mark is dropped without looking at the disk, so a file someone puts back in the meantime stays
    unread."""
    marks = _background_review_read_paths.get()
    if marks is None or expected is None:
        return
    key = _resolved_str(path)
    digest = hashlib.sha256(published).hexdigest() if published is not None else None
    applied, now = marks.compare_and_set(key, expected, digest)
    if applied and (journal := _batch_mark_journal.get()) is not None:
        journal[key] = now  # the update the batch's own write left


# Inside an atomic batch: the last update the batch's own operations made to each mark, so a
# rollback can withdraw exactly those. Only an operation the batch dispatched itself (it claims each one) sees the
# journal; any other skill_manage call, including one a hook makes while an operation runs, runs
# with it hidden. Parallel tool calls run in contexts copied before the batch, so never see it.
_batch_mark_journal: _ctxvars.ContextVar[Optional[dict[str, Optional[tuple[str, int]]]]] = (
    _ctxvars.ContextVar("background_review_batch_mark_journal", default=None))
_batch_op_claim: _ctxvars.ContextVar[bool] = _ctxvars.ContextVar("background_review_batch_op_claim", default=False)


def _begin_batch_mark_journal():
    return _batch_mark_journal.set({})


@contextmanager
def _batch_operation():
    """Around the batch's own dispatch of one operation: the skill_manage call it makes is the
    batch's, and journals."""
    token = _batch_op_claim.set(True)
    try:
        yield
    finally:
        _batch_op_claim.reset(token)


def _journal_owned_by_batch(fn):
    """Decorate skill_manage: a call the batch claimed keeps the batch's journal (and consumes the
    claim, so calls it makes in turn do not inherit it); every other call runs with no journal."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        owned = _batch_op_claim.get()
        claim = _batch_op_claim.set(False)
        hidden = None if owned else _batch_mark_journal.set(None)
        try:
            return fn(*args, **kwargs)
        finally:
            if hidden is not None:
                _batch_mark_journal.reset(hidden)
            _batch_op_claim.reset(claim)
    return wrapper


def _end_batch_mark_journal(token, rolled_back: bool) -> None:
    """Close the batch's journal. After a rollback the bytes the batch's writes taught the fork are
    gone, and what the fork knew before them cannot be rebuilt safely (other workers may have read
    in between), so each mark the batch's own operations left is dropped: the fork re-reads before
    writing that file again. Only a mark that is still exactly the batch's update is dropped; any
    later update stands. (A mark the batch itself dropped stays dropped: dropping is all a rollback
    ever does, so no older value can come back.)"""
    journal = _batch_mark_journal.get()
    _batch_mark_journal.reset(token)
    marks = _background_review_read_paths.get()
    if not rolled_back or not journal or marks is None:
        return
    for key, last in journal.items():
        marks.compare_and_set(key, last, None)


def _reset_background_review_read_marks() -> None:
    """Start a fresh, isolated read set for the current review context."""
    _background_review_read_paths.set(_BackgroundReviewReadMarks())


def _resolved_roots(skill_path: Path):
    """``(resolved skill_path, [(root, resolved_root), ...])`` over every resolvable skills root."""
    from agent.skill_utils import get_all_skills_dirs
    try:
        resolved = skill_path.resolve()
    except OSError:
        resolved = skill_path
    roots = []
    for root in get_all_skills_dirs():
        with suppress(OSError):
            roots.append((root, root.resolve()))
    return resolved, roots


def _containing_skills_root(skill_path: Path) -> Path:
    """Skills root (local or external_dirs) containing ``skill_path``; local dir if none match."""
    from tools import skill_manager_tool as _smt
    resolved, roots = _resolved_roots(skill_path)
    return next((root for root, r in roots if resolved.is_relative_to(r)), _smt._skills_dir())


def _is_path_redirect(path: Path) -> bool:
    """Symlink or (Windows 3.12+) junction — either lets a poisoned tree redirect rmtree outside."""
    try:
        return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())
    except OSError:
        return False


def _validate_delete_target(skill_dir: Path) -> Optional[str]:
    """Last-line guard before rmtree: even a poisoned tree must never delete (1) a path outside
    every known skills root, (2) a skills root itself, (3) a symlink/junction (rmtree follows it).

    ``_find_skill`` already restricts ``skill_dir`` to a real ``SKILL.md`` parent discovered by walking the
    skills roots, so the agent cannot inject an arbitrary path the way Kilo Code's HTTP endpoint could
    (their issue 11227: a built-in-skill sentinel resolved to the server cwd and a recursive delete wiped
    the user's entire working directory). This is the matching defense-in-depth for our agent-facing
    ``skill_manage`` delete path: even if discovery or a poisoned tree hands us a bad directory, never
    recursively delete See #11227.
    """
    if _is_path_redirect(skill_dir):
        return (f"Refusing to delete '{skill_dir}': the skill directory is a "
                f"symlink/junction. Remove the link target manually if intended.")
    try:
        skill_dir.resolve()
    except OSError as exc:
        return f"Refusing to delete '{skill_dir}': could not resolve path ({exc})."
    resolved, roots = _resolved_roots(skill_dir)
    for _root, root in roots:
        if resolved == root:
            return (f"Refusing to delete '{skill_dir}': resolves to the skills root "
                    f"itself, which would remove every installed skill.")
        if resolved.is_relative_to(root):
            return None
    return f"Refusing to delete '{skill_dir}': path does not resolve inside any known skills root."


def _is_pinned(name: str, what: str) -> Optional[bool]:
    """skill_usage pinned flag; None (logged at debug) when the record is unreadable."""
    try:
        from tools import skill_usage
        return bool(skill_usage.get_record(name).get("pinned"))
    except Exception:
        logger.debug("%s lookup failed for %s", what, name, exc_info=True)
        return None


def _pinned_guard(name: str) -> Optional[str]:
    """Refusal message if *name* is pinned or essential, else None. Pin only guards DELETION;
    patches/edits stay allowed. ESSENTIAL_SKILLS are permanently pinned (the system prompt
    references them). Best-effort: an unreadable sidecar lets the delete through."""
    try:
        from agent.skill_utils import ESSENTIAL_SKILLS
        if name in ESSENTIAL_SKILLS:
            return (
                f"Skill '{name}' is essential to Hermes (the agent's own "
                f"operating manual referenced by the system prompt) and "
                f"cannot be deleted. Patches and edits are still allowed.")
    except Exception:
        logger.debug("essential-guard lookup failed for %s", name, exc_info=True)
    if _is_pinned(name, "pinned-guard"):
        return (
            f"Skill '{name}' is pinned and cannot be deleted by skill_manage. Ask the user to "
            f"run `hermes curator unpin {name}` if they want to delete it. Patches and edits "
            f"are allowed on pinned skills; only deletion is blocked.")
    return None


def _background_review_delete_guard(name: str, skill_dir: Path) -> Optional[dict[str, Any]]:
    """Refuse autonomous deletes of anything but curator-owned sediment. Content writes are not
    gated by ownership: the review fork exists to improve every skill it learns from, and every
    write is ledgered and reversible, while an archive removes a skill the user relies on."""
    if not _is_background_review():
        return None
    refuse = "Refusing background curator delete for"
    if _is_pinned(name, "pinned skill guard"):
        return _refusal(
            f"{refuse} pinned skill '{name}': pinned skills "
            f"are off-limits to autonomous maintenance. Ask the user to run `hermes curator "
            f"unpin {name}` if they want it changed.")
    try:
        from agent.skill_utils import is_external_skill_path
        if is_external_skill_path(skill_dir):
            return _refusal(
                f"{refuse} skill '{name}': the skill lives in skills.external_dirs, which are "
                f"externally owned and read-only to autonomous curation.")
    except Exception:
        logger.debug("external skill guard lookup failed for %s", name, exc_info=True)
    try:
        from tools import skill_usage
        for predicate, label in (
            (skill_usage.is_protected_builtin, "protected built-in"),
            (skill_usage.is_hub_installed, "hub-installed"),
            (skill_usage.is_bundled, "bundled")):
            if predicate(name):
                return _refusal(f"{refuse} {label} skill '{name}'.")
        # Not curator-managed (no `created_by: "agent"`) => user-owned, never archived
        # autonomously. A MISSING record and `created_by: null` must resolve identically:
        # keying on presence let the guard's own bookkeeping flip the verdict (#67140).
        usage_rec = skill_usage.load_usage().get(name)
        if not skill_usage._is_curator_managed_record(usage_rec):
            _detail = (f"created_by={usage_rec.get('created_by')!r}" if isinstance(usage_rec, dict)
                       else "no usage record")
            return _refusal(
                f"{refuse} skill '{name}': the skill is not "
                f"curator-managed ({_detail}). Only curator-managed skills may be archived "
                f"autonomously. Run `hermes curator adopt {name}` to opt it in.")
    except Exception:
        logger.warning("owned skill guard lookup failed for %s", name, exc_info=True)
        return _refusal(
            f"{refuse} skill '{name}': agent ownership could not "
            f"be verified because the provenance record is unavailable or unreadable.")
    return None


def _background_review_read_before_write_guard(
    name: str, target: Path, action: str, file_label: str) -> Optional[dict[str, Any]]:
    """Require review forks to load the exact target before mutating it, and refuse when the
    target's bytes no longer match what was loaded (another writer changed it since). Callers
    hold the skill's mutation lock, so the check and the write are one critical section against
    every writer that takes the same lock file."""
    if not _is_background_review():
        return None
    if (expected := _read_mark(target)) is None:
        return _refusal(
            f"Refusing background curator {action} for skill '{name}': the current {file_label} "
            f"content has not been loaded in this review turn. Call skill_view(name) for SKILL.md, or "
            f"skill_view(name, file_path=...) for a supporting file, then retry the write using the "
            f"content just returned.",
            _read_before_write_required=True)
    try:
        current = _file_sha256(target)
    except OSError:
        logger.warning("could not hash %s to check it is unchanged; refusing the write", target, exc_info=True)
        current = None
    if current == expected:
        return None
    return _refusal(
        f"Refusing background curator {action} for skill '{name}': {file_label} changed since you "
        f"loaded it (or could not be read to check). Another writer updated it, and writing now would "
        f"overwrite that change. Call skill_view again for the current content and redo the change "
        f"against what it returns.",
        _stale_read=True)


def _locked_guard(name: str, action: str) -> Optional[dict[str, Any]]:
    """Refuse every mutating ``skill_manage`` action on a skill the user locked (``hermes curator
    lock`` / the Desktop lock switch), whoever asks: foreground turns, the background review and the
    curator pass alike. ``create`` is exempt (the collision check already refuses a name that resolves
    anywhere). An unreadable sidecar fails CLOSED for the autonomous passes (silently relaxing a
    lock is the failure this exists to prevent) and open for normal turns, so one corrupt file does
    not freeze every skill edit. Keyed by directory name, so the bare, ``cat/name`` and
    frontmatter-name forms all resolve to the same record."""
    if action == "create" or not name:
        return None
    try:
        from tools import skill_manager_tool as _smt
        from tools import skill_usage
        candidates = {name, Path(name).name}
        if existing := _smt._find_skill(name):
            candidates.add(existing["path"].name)
        locked = next((c for c in candidates if skill_usage.is_locked(c)), None)
    except Exception:
        logger.warning("locked skill guard lookup failed for %s", name, exc_info=True)
        if not _is_background_review():
            return None  # a corrupt sidecar must not freeze every skill edit in normal turns
        return _refusal(
            f"Refusing {action} for skill '{name}': its lock state could not be verified because the "
            f"skills/.usage.json sidecar is unreadable. Leave the skill alone and tell the user; they "
            f"can repair the sidecar or move it aside.", _locked=True)
    if locked:
        from tools.skill_lock import skill_refusal
        return _refusal(f"Refusing {action}: {skill_refusal(locked)}", _locked=True)
    return None

def _background_review_preflight(action: str, name: str) -> Optional[dict[str, Any]]:
    """Guards that run on the bare action/name before any handler, for the single-op AND the
    batch path: the content lock (every mutating action) and the ownership delete guard."""
    if (locked := _locked_guard(name, action)) is not None:
        return locked
    if action != "delete":
        return None
    from tools import skill_manager_tool as _smt
    existing = _smt._find_skill(name)
    return _background_review_delete_guard(name, existing["path"]) if existing else None


def _curator_consolidation_delete_guard(
    name: str, absorbed_into: Optional[str]) -> Optional[dict[str, Any]]:
    """Fail closed on unverified deletes during the curator consolidation pass. The fork's only
    legitimate delete is a consolidation declared via ``absorbed_into=<umbrella>`` (existence
    validated in ``_delete_skill``); the deterministic inactivity prune never calls skill_manage,
    so a bare delete here can only be the LLM pass pruning without evidence.

    A delete with no forwarding target — ``absorbed_into`` omitted (``None``) or empty (``""``) — is the
    fail-open behavior reported in #29912: the consolidation pass archived whole clusters of active skills
    with zero verified consolidations (``consolidated_this_run == 0``), leaving active automations pointing
    at names that no longer resolve. Refuse it; keep the skill active.
    """
    if not _is_background_review() or (isinstance(absorbed_into, str) and absorbed_into.strip()):
        return None
    return _refusal(
        f"Refusing background curator delete of skill '{name}': the consolidation pass may only "
        f"archive a skill it has absorbed into an umbrella. Pass absorbed_into=<umbrella> (the "
        f"umbrella must already exist) to record a verified consolidation. Pruning a skill with no "
        f"forwarding target is not permitted here — the deterministic inactivity prune handles "
        f"staleness archival separately. Keeping '{name}' active.",
        _fail_closed=True)
