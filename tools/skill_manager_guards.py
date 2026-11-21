"""Write/delete guards for ``skill_manage``. Every guard returns ``None`` when the
operation may proceed, else a refusal (error dict or message). Origin-owned state
(``_find_skill``, ``_skills_dir``) is reached lazily via ``tools.skill_manager_tool``
so test patches keep working."""

import contextvars as _ctxvars
import logging
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("tools.skill_manager_tool")


def _refusal(message: str, **extra: Any) -> dict[str, Any]:
    return {"success": False, "error": message, **extra}


# An atomic batch binds each op to the folder it checked, or for a create the folder it planned
# (``skill_manager_batch``): while the op runs, its name must still resolve to that folder, or the handler
# refuses before writing. A folder found
# again by the handler can differ when another writer, holding a different name's lock, changes which
# skill the name finds in between. ``_BOUND_NOTHING``: the name found no skill when it was checked.
_BOUND_NOTHING = object()
_batch_bound_folder: _ctxvars.ContextVar[Optional[tuple]] = _ctxvars.ContextVar(
    "skill_batch_bound_folder", default=None)


# What the running batch's own execution did, recorded at the mutation (never inferred from an op's result
# or from comparing files afterwards, which another writer can change at the same moment): each write
# sink notes its resolved folder and file just before it attempts the write, and a create notes the
# folder it made (or adopted) right after making it. That covers the batch's ops, a failed or raising
# op's partial work, and a lifecycle hook's nested skill_manage call in the same context. Another thread's
# writer is not in this context and is never recorded.
class _BatchJournal:
    def __init__(self):
        self.writes: dict = {}   # resolved folder -> {relative file written}
        self.created: dict = {}  # resolved folder a create made -> True if it adopted an empty leftover dir

    def merge_into(self, outer: "_BatchJournal") -> None:
        for folder, files in self.writes.items():
            outer.writes.setdefault(folder, set()).update(files)
        for folder, adopted in self.created.items():
            outer.created.setdefault(folder, adopted)


_batch_journal: _ctxvars.ContextVar[Optional[_BatchJournal]] = _ctxvars.ContextVar(
    "skill_batch_journal", default=None)


def _note_batch_write(skill_dir: Any, target: Any = None) -> None:
    """Record, for a running batch, that ``skill_dir`` (and ``target`` inside it) is about to change."""
    journal = _batch_journal.get()
    if journal is None:
        return
    files = journal.writes.setdefault(Path(skill_dir).resolve(), set())
    if target is not None:
        with suppress(ValueError):
            files.add(Path(target).relative_to(skill_dir).as_posix())


def _note_batch_create(skill_dir: Any, adopted: bool) -> None:
    """Record, for a running batch, that a create just made ``skill_dir`` (or adopted it, empty)."""
    journal = _batch_journal.get()
    if journal is not None:
        journal.created.setdefault(Path(skill_dir).resolve(), adopted)


def _batch_binding_error(name: str, skill_dir: Any) -> Optional[dict[str, Any]]:
    """Refusal when a batch op's name resolves to another folder than the batch checked, else None.
    Applies only to the op's own name (a hook writing another skill from inside the op is not bound)."""
    bound = _batch_bound_folder.get()
    if bound is None or bound[0] != name:
        return None
    if bound[1] is not _BOUND_NOTHING and Path(skill_dir).resolve() == bound[1]:
        return None
    return _refusal(f"'{name}' now resolves to {skill_dir}, not the folder this batch checked: the skill "
                    f"the name finds changed while the batch ran. Nothing was written.")


def _is_background_review() -> bool:
    """True inside the autonomous curator review fork, and on any lookup failure: every guard keyed
    on this is stricter for the fork, so an unknown origin gets the fork's guards."""
    try:
        from tools.skill_provenance import is_background_review
        return bool(is_background_review())
    except Exception:
        logger.warning("write-origin lookup failed; applying background-review guards", exc_info=True)
        return True


def _resolved_str(path: Path) -> str:
    with suppress(Exception):
        return str(path.resolve())
    return str(path)


class _BackgroundReviewReadMarks:
    """Read marks shared by copied tool contexts within one review run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._paths: set[str] = set()

    def add(self, path: str) -> None:
        with self._lock:
            self._paths.add(path)

    def contains(self, path: str) -> bool:
        with self._lock:
            return path in self._paths


_background_review_read_paths: _ctxvars.ContextVar[Optional[_BackgroundReviewReadMarks]] = (
    _ctxvars.ContextVar("background_review_read_paths", default=None))


def mark_background_review_skill_read(path: Path) -> None:
    """Record that the active background-review fork has read a skill file. The fork must not
    patch content it only inferred from the transcript: skill_view/read_file call this, and
    the write guards require the mark."""
    if not _is_background_review():
        return
    if (marks := _background_review_read_paths.get()) is None:
        _background_review_read_paths.set(marks := _BackgroundReviewReadMarks())
    marks.add(_resolved_str(path))


def _background_review_has_read(path: Path) -> bool:
    marks = _background_review_read_paths.get()
    return marks is not None and marks.contains(_resolved_str(path))


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
    """Require review forks to load the exact target before mutating it."""
    if not _is_background_review() or _background_review_has_read(target):
        return None
    return _refusal(
        f"Refusing background curator {action} for skill '{name}': the current {file_label} "
        f"content has not been loaded in this review turn. Call skill_view(name) for SKILL.md, or "
        f"skill_view(name, file_path=...) for a supporting file, then retry the write using the "
        f"content just returned.",
        _read_before_write_required=True)


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
