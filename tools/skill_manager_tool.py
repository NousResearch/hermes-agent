#!/usr/bin/env python3
"""Skill Manager Tool — agent-managed skill creation & editing.

Skills are the agent's procedural memory (narrow "how to do X"; MEMORY.md/USER.md are
broad, declarative). New skills land in ~/.hermes/skills/ (or ``skills.create_dir``);
existing skills (bundled, hub, user) are modified in place. Layout:
``<skills>/[category/]<skill>/SKILL.md`` + optional ``references/ templates/ scripts/ assets/``.
"""

import contextvars as _ctxvars
import json
from contextlib import suppress
import logging
import re
import shutil
import threading
import contextvars as _ctxvars
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from hermes_constants import get_hermes_home, display_hermes_home
from utils import atomic_write_text, is_truthy_value
from hermes_cli.config import cfg_get
from agent.skill_utils import (
    extract_skill_description,
    is_skill_description_truncated_for_prompt,
    parse_frontmatter as _parse_frontmatter,
    SKILL_PROMPT_DESC_LIMIT)
from tools.skill_manager_guards import (
    _background_review_preflight, _background_review_read_before_write_guard, _background_review_write_guard,
    _containing_skills_root, _curator_consolidation_delete_guard, _maybe_auto_propose_org_edit,
    _org_mirror_write_guard, _pinned_guard, _validate_delete_target, _is_background_review, _refusal as _err)
from tools.skill_manager_batch import _skill_manage_batch
from tools.skills_guard import scan_skill, should_allow_install, format_scan_report

logger = logging.getLogger(__name__)

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


_background_review_read_paths: (
    "_ctxvars.ContextVar[Optional[_BackgroundReviewReadMarks]]"
) = _ctxvars.ContextVar("background_review_read_paths", default=None)


def mark_background_review_skill_read(path: Path) -> None:
    """Record that the active background-review fork has read a skill file.

    The autonomous review fork is allowed to evolve skills, but it must not
    patch or rewrite content it has only inferred from the transcript.  The
    skill_view tool calls this after returning file content to the model; write
    paths below require the corresponding target path to be present when the
    current origin is ``background_review``.
    """
    try:
        from tools.skill_provenance import is_background_review
        if not is_background_review():
            return
    except Exception:
        return

    try:
        resolved = str(path.resolve())
    except Exception:
        resolved = str(path)
    marks = _background_review_read_paths.get()
    if marks is None:
        marks = _BackgroundReviewReadMarks()
        _background_review_read_paths.set(marks)
    marks.add(resolved)


def _background_review_has_read(path: Path) -> bool:
    try:
        resolved = str(path.resolve())
    except Exception:
        resolved = str(path)
    marks = _background_review_read_paths.get()
    return marks is not None and marks.contains(resolved)


def _reset_background_review_read_marks() -> None:
    """Start a fresh, isolated read set for the current review context."""
    _background_review_read_paths.set(_BackgroundReviewReadMarks())

# Import security scanner — external hub installs always get scanned;
# agent-created skills only get scanned when skills.guard_agent_created is on.
try:
    from tools.skills_guard import scan_skill, should_allow_install, format_scan_report
    _GUARD_AVAILABLE = True
except ImportError:
    _GUARD_AVAILABLE = False


def _guard_agent_created_enabled() -> bool:
    """skills.guard_agent_created (default False): opt-in — terminal() runs the same code ungated."""
    try:
        from hermes_cli.config import load_config
        return is_truthy_value(cfg_get(load_config(), "skills", "guard_agent_created"), default=False)
    except Exception:
        return False


def _security_scan_skill(skill_dir: Path) -> Optional[str]:
    """Post-write scan (opt-in); error string if blocked, else None. An "ask" verdict
    (dangerous findings) is surfaced as an error so the agent can retry without them."""
    if not _guard_agent_created_enabled():
        return None
    try:
        result = scan_skill(skill_dir, source="agent-created")
        allowed, reason = should_allow_install(result)
        if allowed is None:
            logger.warning("Agent-created skill blocked (dangerous findings): %s", reason)
        if allowed is not True:
            return f"Security scan blocked this skill ({reason}):\n{format_scan_report(result)}"
    except Exception as e:
        logger.warning("Security scan failed for %s: %s", skill_dir, e, exc_info=True)
    return None


# All skills live in ~/.hermes/skills/ (single source of truth)
HERMES_HOME = get_hermes_home()
SKILLS_DIR = HERMES_HOME / "skills"
_SKILLS_DIR_AT_IMPORT = SKILLS_DIR


def _skills_dir() -> Path:
    """Active profile's skills dir at call time (multi-profile runtimes rebind per session).
    An explicitly patched module-level ``SKILLS_DIR`` (tests) wins over the live HERMES_HOME.

    Long-lived multi-profile runtimes (Dashboard/TUI/Desktop backend, cron, kanban workers) import this
    module once under the launch HERMES_HOME and later bind a different profile per session (#40677).
    """
    configured = Path(SKILLS_DIR)
    return configured if configured != _SKILLS_DIR_AT_IMPORT else get_hermes_home() / "skills"


MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024


def _display_create_dir() -> str:
    """Display string for the skill-creation directory (schema/instruction text).

    Renders ``skills.create_dir`` when configured so every instruction that
    names the creation path follows the config, falling back to the
    profile-local skills dir.
    """
    try:
        from agent.skill_utils import display_skill_create_dir
        return display_skill_create_dir()
    except Exception:
        return f"{display_hermes_home()}/skills/"


def _containing_skills_root(skill_path: Path) -> Path:
    """Return the skills root directory (local or external_dirs entry) that
    contains ``skill_path``.  Falls back to the local ``SKILLS_DIR`` if no
    match is found (defensive — callers should have located the skill via
    ``_find_skill`` first).
    """
    from agent.skill_utils import get_all_skills_dirs

    try:
        resolved = skill_path.resolve()
    except OSError:
        resolved = skill_path

    for root in get_all_skills_dirs():
        try:
            resolved.relative_to(root.resolve())
            return root
        except (ValueError, OSError):
            continue
    return _skills_dir()


def _is_path_redirect(path: Path) -> bool:
    """True when ``path`` is a symlink or (on Windows) a directory junction.

    Either form lets a poisoned skills tree redirect a subsequent
    ``shutil.rmtree`` to content outside the skills root. ``is_junction``
    only exists on Python 3.12+ Windows; gate with ``hasattr``.
    """
    try:
        return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())
    except OSError:
        return False


def _validate_delete_target(skill_dir: Path) -> Optional[str]:
    """Last-line guard before ``shutil.rmtree(skill_dir)`` in ``_delete_skill``.

    ``_find_skill`` already restricts ``skill_dir`` to a real ``SKILL.md``
    parent discovered by walking the skills roots, so the agent cannot inject
    an arbitrary path the way Kilo Code's HTTP endpoint could (their issue
    #11227: a built-in-skill sentinel resolved to the server cwd and a
    recursive delete wiped the user's entire working directory). This is the
    matching defense-in-depth for our agent-facing ``skill_manage`` delete
    path: even if discovery or a poisoned tree hands us a bad directory, never
    recursively delete

      1. a path that is not strictly *inside* one of the known skills roots,
      2. a skills root itself (would wipe every installed skill), or
      3. a directory reached via a symlink / junction (``rmtree`` would follow
         it into content outside the skills tree).

    Returns an error string to refuse on, or ``None`` when the delete is safe.
    """
    from agent.skill_utils import get_all_skills_dirs

    # (3) Reject symlink/junction redirects on the skill directory itself.
    if _is_path_redirect(skill_dir):
        return (
            f"Refusing to delete '{skill_dir}': the skill directory is a "
            f"symlink/junction. Remove the link target manually if intended."
        )

    try:
        resolved = skill_dir.resolve()
    except OSError as exc:
        return f"Refusing to delete '{skill_dir}': could not resolve path ({exc})."

    roots = []
    for root in get_all_skills_dirs():
        try:
            roots.append(root.resolve())
        except OSError:
            continue

    for root in roots:
        # (2) Never rmtree a skills root itself.
        if resolved == root:
            return (
                f"Refusing to delete '{skill_dir}': resolves to the skills root "
                f"itself, which would remove every installed skill."
            )
        # (1) Must be strictly inside a known root.
        try:
            rel = resolved.relative_to(root)
        except ValueError:
            continue
        if rel.parts:  # at least one component below the root
            return None

    return (
        f"Refusing to delete '{skill_dir}': path does not resolve inside any "
        f"known skills root."
    )


def _pinned_guard(name: str) -> Optional[str]:
    """Return a refusal message if *name* is pinned or essential, else None.

    Pin protects a skill from **deletion** — both the curator's auto-archive
    passes and the agent's ``skill_manage(action="delete")`` tool call. The
    agent can still patch/edit pinned skills; pin only guards against
    irrecoverable loss, not against content evolution.

    Essential skills (``agent/skill_utils.ESSENTIAL_SKILLS``, e.g.
    ``hermes-agent``) are treated as permanently pinned: the system prompt
    always references them, so deleting one leaves a dangling instruction.

    Best-effort: if the sidecar is unreadable we let the delete through
    rather than block on a broken telemetry file.
    """
    try:
        from agent.skill_utils import ESSENTIAL_SKILLS
        if name in ESSENTIAL_SKILLS:
            return (
                f"Skill '{name}' is essential to Hermes (the agent's own "
                f"operating manual referenced by the system prompt) and "
                f"cannot be deleted. Patches and edits are still allowed."
            )
    except Exception:
        logger.debug("essential-guard lookup failed for %s", name, exc_info=True)
    try:
        from tools import skill_usage
        rec = skill_usage.get_record(name)
        if rec.get("pinned"):
            return (
                f"Skill '{name}' is pinned and cannot be deleted by "
                f"skill_manage. Ask the user to run "
                f"`hermes curator unpin {name}` if they want to delete it. "
                f"Patches and edits are allowed on pinned skills; only "
                f"deletion is blocked."
            )
    except Exception:
        logger.debug("pinned-guard lookup failed for %s", name, exc_info=True)
    return None


def _background_review_write_guard(
    name: str,
    skill_dir: Path,
    action: str,
) -> Optional[Dict[str, Any]]:
    """Refuse autonomous curator writes to externally owned skills.

    Foreground agents may still perform user-directed edits to external,
    bundled, or hub-installed skills. The background review fork is different:
    it is autonomous lifecycle maintenance, so its write surface is restricted
    to local curator-owned sediment.
    """
    try:
        from tools.skill_provenance import is_background_review
        if not is_background_review():
            return None
    except Exception:
        return None

    # Pin must be respected by autonomous maintenance. The curator already
    # skips pinned skills from every auto-transition; the background review
    # fork is the same kind of autonomous, no-user-present actor, so it must
    # not write to a pinned skill either (issue #25839). This is stricter than
    # the foreground ``_pinned_guard`` (which only blocks deletion) precisely
    # because there is no user in the loop to consent to an edit here.
    try:
        from tools import skill_usage
        if skill_usage.get_record(name).get("pinned"):
            return {
                "success": False,
                "error": (
                    f"Refusing background curator {action} for pinned skill "
                    f"'{name}': pinned skills are off-limits to autonomous "
                    "maintenance. Ask the user to run "
                    f"`hermes curator unpin {name}` if they want it changed."
                ),
            }
    except Exception:
        logger.debug("pinned skill guard lookup failed for %s", name, exc_info=True)

    try:
        from agent.skill_utils import is_external_skill_path
        if is_external_skill_path(skill_dir):
            return {
                "success": False,
                "error": (
                    f"Refusing background curator {action} for skill '{name}': "
                    "the skill lives in skills.external_dirs, which are "
                    "externally owned and read-only to autonomous curation."
                ),
            }
    except Exception:
        logger.debug("external skill guard lookup failed for %s", name, exc_info=True)

    try:
        from tools import skill_usage
        if skill_usage.is_protected_builtin(name):
            return {
                "success": False,
                "error": (
                    f"Refusing background curator {action} for protected "
                    f"built-in skill '{name}'."
                ),
            }
        if skill_usage.is_hub_installed(name):
            return {
                "success": False,
                "error": (
                    f"Refusing background curator {action} for hub-installed "
                    f"skill '{name}'."
                ),
            }
        if skill_usage.is_bundled(name):
            return {
                "success": False,
                "error": (
                    f"Refusing background curator {action} for bundled "
                    f"skill '{name}'."
                ),
            }
        # Skills that are not curator-managed are off-limits to autonomous
        # curation. This prevents the LLM consolidation pass from mutating
        # skills the user owns (manually authored, URL-installed, or created by
        # a foreground `skill_manage(create)` at the user's request), which lack
        # the `created_by: "agent"` marker.
        #
        # A MISSING record and an explicit `created_by: null` must resolve
        # IDENTICALLY (issue #67140). Keying on `isinstance(usage_rec, dict)`
        # made the policy depend on the guard's own side effect: a local skill
        # with no telemetry record passed, the successful write called
        # bump_patch() which created a `created_by: null` record, and the very
        # same write was refused from then on. "Allowed exactly once" is not a
        # policy — it is a race with our own bookkeeping. Fail closed for both
        # shapes; `hermes curator adopt <name>` is the supported way in.
        usage_data = skill_usage.load_usage()
        usage_rec = usage_data.get(name)
        if not skill_usage._is_curator_managed_record(usage_rec):
            if isinstance(usage_rec, dict):
                _detail = f"created_by={usage_rec.get('created_by')!r}"
            else:
                _detail = "no usage record"
            return {
                "success": False,
                "error": (
                    f"Refusing background curator {action} for skill "
                    f"'{name}': the skill is not curator-managed ({_detail}). "
                    "User-owned skills are off-limits to autonomous curation. "
                    f"Run `hermes curator adopt {name}` to opt it in."
                ),
            }
    except Exception:
        logger.warning("owned skill guard lookup failed for %s", name, exc_info=True)
        return {
            "success": False,
            "error": (
                f"Refusing background curator {action} for skill '{name}': "
                "agent ownership could not be verified because the provenance "
                "record is unavailable or unreadable."
            ),
        }
    return None


def _background_review_read_before_write_guard(
    name: str,
    target: Path,
    action: str,
    file_label: str,
) -> Optional[Dict[str, Any]]:
    """Require review forks to load the exact target before mutating it."""
    try:
        from tools.skill_provenance import is_background_review
        if not is_background_review():
            return None
    except Exception:
        return None

    if _background_review_has_read(target):
        return None

    return {
        "success": False,
        "error": (
            f"Refusing background curator {action} for skill '{name}': "
            f"the current {file_label} content has not been loaded in this "
            "review turn. Call skill_view(name) for SKILL.md, or "
            "skill_view(name, file_path=...) for a supporting file, then "
            "retry the write using the content just returned."
        ),
        "_read_before_write_required": True,
    }


def _background_review_preflight(action: str, name: str) -> Optional[Dict[str, Any]]:
    if action not in {"edit", "patch", "delete", "write_file", "remove_file"}:
        return None
    existing = _find_skill(name)
    if not existing:
        return None
    return _background_review_write_guard(name, existing["path"], action)


def _curator_consolidation_delete_guard(
    name: str, absorbed_into: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Fail closed on unverified deletes during the curator consolidation pass.

    The curator's forked review agent (``is_background_review()``) runs the
    LLM umbrella-building pass. Its only legitimate ``skill_manage(delete)`` is
    a *verified consolidation*: the skill's content was absorbed into an
    umbrella, declared via ``absorbed_into=<umbrella>`` where the umbrella
    exists on disk (validated separately in ``_delete_skill``).

    A delete with no forwarding target — ``absorbed_into`` omitted (``None``)
    or empty (``""``) — is the fail-open behavior reported in #29912: the
    consolidation pass archived whole clusters of active skills with zero
    verified consolidations (``consolidated_this_run == 0``), leaving active
    automations pointing at names that no longer resolve. The deterministic
    inactivity prune is the only legitimate prune path, and it archives via
    ``skill_usage.archive_skill()`` directly without ever calling
    ``skill_manage`` — so a bare prune reaching here can only be the LLM pass
    pruning without consolidation evidence. Refuse it; keep the skill active.

    Returns an error dict to abort the delete, or ``None`` when the delete is
    allowed to proceed (not the curator pass, or a declared consolidation).
    """
    try:
        from tools.skill_provenance import is_background_review
        if not is_background_review():
            return None
    except Exception:
        return None

    declared = isinstance(absorbed_into, str) and absorbed_into.strip()
    if declared:
        return None

    return {
        "success": False,
        "error": (
            f"Refusing background curator delete of skill '{name}': the "
            "consolidation pass may only archive a skill it has absorbed into "
            "an umbrella. Pass absorbed_into=<umbrella> (the umbrella must "
            "already exist) to record a verified consolidation. Pruning a "
            "skill with no forwarding target is not permitted here — the "
            "deterministic inactivity prune handles staleness archival "
            "separately. Keeping '{name}' active.".format(name=name)
        ),
        "_fail_closed": True,
    }


MAX_SKILL_CONTENT_CHARS = 100_000   # ~36k tokens at 2.75 chars/token
MAX_SKILL_FILE_BYTES = 1_048_576    # 1 MiB per supporting file
VALID_NAME_RE = re.compile(r'^[a-z0-9][a-z0-9._-]*$')  # filesystem-safe, URL-friendly
ALLOWED_SUBDIRS = {"references", "templates", "scripts", "assets"}  # for write_file/remove_file
_FRONTMATTER_END_RE = re.compile(r'\n---\s*\n')
_NAME_RULE = "Use lowercase letters, numbers, hyphens, dots, and underscores."


def _display_create_dir() -> str:
    """Skill-creation dir for schema/instruction text; follows ``skills.create_dir``."""
    try:
        from agent.skill_utils import display_skill_create_dir
        return display_skill_create_dir()
    except Exception:
        return f"{display_hermes_home()}/skills/"


# --- Validation helpers -------------------------------------------------------

def _check_identifier(value: str, label: str, invalid: str) -> Optional[str]:
    if len(value) > MAX_NAME_LENGTH:
        return f"{label} exceeds {MAX_NAME_LENGTH} characters."
    return None if VALID_NAME_RE.match(value) else invalid


def _validate_name(name: str) -> Optional[str]:
    if not name:
        return "Skill name is required."
    return _check_identifier(
        name, "Skill name", f"Invalid skill name '{name}'. {_NAME_RULE} Must start with a letter or digit.")


def _validate_category(category: Optional[str]) -> Optional[str]:
    if category is None or (isinstance(category, str) and not category.strip()):
        return None
    if not isinstance(category, str):
        return "Category must be a string."
    category = category.strip()
    invalid = (f"Invalid category '{category}'. {_NAME_RULE} "
               "Categories must be a single directory name.")
    if "/" in category or "\\" in category:
        return invalid
    return _check_identifier(category, "Category", invalid)


def _validate_frontmatter(content: str, *, new_skill: bool = False) -> Optional[str]:
    """Validate frontmatter (name + description) and a non-empty body. ``new_skill`` (create
    only) also enforces SKILL_PROMPT_DESC_LIMIT so new skills never lose routing signal to
    index truncation; edit/patch skip it so existing over-limit skills stay maintainable."""
    if not content.strip():
        return "Content cannot be empty."
    content = content.lstrip("\ufeff")  # tolerate a Windows UTF-8 BOM
    if not content.startswith("---"):
        return "SKILL.md must start with YAML frontmatter (---). See existing skills for format."
    end_match = _FRONTMATTER_END_RE.search(content[3:])
    if not end_match:
        return "SKILL.md frontmatter is not closed. Ensure you have a closing '---' line."
    try:
        parsed = yaml.safe_load(content[3:end_match.start() + 3])
    except yaml.YAMLError as e:
        return f"YAML frontmatter parse error: {e}"
    if not isinstance(parsed, dict):
        return "Frontmatter must be a YAML mapping (key: value pairs)."
    for field in ("name", "description"):
        if field not in parsed:
            return f"Frontmatter must include '{field}' field."
    desc = str(parsed["description"])
    if len(desc) > MAX_DESCRIPTION_LENGTH:
        return f"Description exceeds {MAX_DESCRIPTION_LENGTH} characters."
    if new_skill and len(desc.strip().strip("'\"")) > SKILL_PROMPT_DESC_LIMIT:
        return (
            f"Description is {len(desc.strip())} chars — new skills must fit the "
            f"{SKILL_PROMPT_DESC_LIMIT}-char system-prompt budget (one sentence, trigger first, "
            f"ends with a period). The skill index truncates longer descriptions to "
            f"{SKILL_PROMPT_DESC_LIMIT - 3} chars + '...', destroying the routing signal. "
            f"Move detail into the skill body.")
    if not content[end_match.end() + 3:].strip():
        return "SKILL.md must have content after the frontmatter (instructions, procedures, etc.)."
    return None


def _validate_content_size(content: str, label: str = "SKILL.md") -> Optional[str]:
    if len(content) > MAX_SKILL_CONTENT_CHARS:
        return (
            f"{label} content is {len(content):,} characters (limit: {MAX_SKILL_CONTENT_CHARS:,}). "
            f"Consider splitting into a smaller SKILL.md with supporting files in references/ "
            f"or templates/.")
    return None


def _description_preview(content: str) -> str:
    """First 120 chars of the frontmatter description; '' on any failure."""
    with suppress(Exception):
        fm_end = _FRONTMATTER_END_RE.search(content[3:])
        if fm_end:
            return str(yaml.safe_load(content[3:fm_end.start() + 3]).get("description", ""))[:120]
    return ""


def _resolve_skill_dir(name: str, category: str = None) -> Path:
    """Build the directory path for a new skill, optionally under a category.

    Honors ``skills.create_dir`` from config.yaml: when configured, new
    skills are created there (e.g. a shared brain/fleet directory) instead
    of the profile-local skills dir.  Falls back to the local dir when unset.
    """
    base = _skills_dir()
    try:
        from agent.skill_utils import get_skill_create_dir
        create_dir = get_skill_create_dir()
        if create_dir is not None:
            base = create_dir
    except Exception:
        logger.debug("skills.create_dir lookup failed", exc_info=True)
    if category:
        return base / category / name
    return base / name


def _find_skill(name: str) -> Optional[Dict[str, Any]]:
    """Find a skill (local skills dir, then skills.external_dirs) -> ``{"path": Path}`` | None.

    Searches the local skills dir (~/.hermes/skills/) first, then any
    external dirs configured via skills.external_dirs.  Returns
    {"path": Path} or None.

    Accepts both the bare directory name (``axolotl``) and the categorized
    relative path (``mlops/axolotl``) — the same two forms skill_view
    resolves, and the form skill_view's ambiguity hint explicitly tells
    the caller to use. The bare-name match compares the skill's own
    directory name (``parent.name``), so bare lookups keep working for
    category-nested skills.
    """
    from agent.skill_utils import get_all_skills_dirs, is_excluded_skill_path

    # Resolve the local skills root once — the categorized form matches the
    # skill dir's path RELATIVE to that root. Only computed lazily (bare-name
    # lookups never need it) and never for external dirs (relative_to raises).
    _resolved_root: Optional[Path] = None

    def _local_root() -> Path:
        nonlocal _resolved_root
        if _resolved_root is None:
            try:
                _resolved_root = _skills_dir().resolve()
            except OSError:
                logger.debug(
                    "skills dir resolve failed; categorized lookups fall back to the unresolved path",
                    exc_info=True,
                )
                _resolved_root = _skills_dir()
        return _resolved_root

    for skills_dir in get_all_skills_dirs():
        if not skills_dir.exists():
            continue
        for skill_md in skills_dir.rglob("SKILL.md"):
            if is_excluded_skill_path(skill_md):
                continue
            # Fast path first: the bare directory name. Avoids the resolve()
            # machinery entirely on the common match.
            if skill_md.parent.name == name:
                return {"path": skill_md.parent}
            # Categorized form (``category/skill-name``): compare the skill
            # dir's POSIX relative path so the lookup works on Windows too.
            if "/" in name or "\\" in name:
                try:
                    rel = skill_md.parent.resolve().relative_to(_local_root())
                except ValueError:
                    continue
                if rel.as_posix() == name:
                    return {"path": skill_md.parent}
    return None


def _maybe_auto_propose_org_edit(name: str, skill_path: Path) -> Optional[str]:
    """Submit an org-skill edit upstream when `sync.org_auto_propose` is on.

    Returns a short note for the tool result, or None when nothing happened.
    Never raises: an offline/failed submission must not fail the edit itself —
    the change is already saved locally and can be proposed later.
    """
    try:
        from agent.skill_utils import is_org_mirror_path
        from tools import skills_sync_client as ssc

        if not is_org_mirror_path(skill_path, _skills_dir()):
            return None
        if not ssc.sync_org_auto_propose():
            return (
                f"This skill is shared by your organisation. Your edit is "
                f"saved locally and will not be overwritten by org updates. "
                f"Run `hermes sync propose {name}` to share it back."
            )
        result = ssc.propose_skill(name)
        if result.get("proposal_pending"):
            return (
                f"Auto-proposed to your organisation as proposal "
                f"#{result.get('proposal_id')} (pending admin review)."
            )
        return "Auto-proposed to your organisation (merged into the shared set)."
    except Exception as e:
        logger.debug("auto-propose skipped for %s: %s", name, e)
        return (
            f"Edit saved locally. Could not submit it to your organisation "
            f"right now — run `hermes sync propose {name}` to retry."
        )


def _org_mirror_write_guard(name: str, skill_path: Path, action: str) -> Optional[Dict[str, Any]]:
    """Org-shared skills are EDITABLE IN PLACE — this only blocks deletion.

    Earlier versions refused every write to `_org/`, which broke the learning
    loop exactly where it matters most: the agent is told to patch a skill the
    moment it finds a gap, and shared skills are the ones the most people use.
    Blocking that froze org skills while personal ones kept improving, and the
    "fork it into a personal skill" alternative is not something an agent does
    mid-task — so improvements were simply lost.

    Now an edit lands in the mirror and is protected from being overwritten by
    the next org pull (see the baseline sidecar in skills_sync_client). It
    reaches the organisation when the user runs `hermes sync propose`, or
    immediately if `sync.org_auto_propose` is on.

    Deletion is still refused: the mirror is a materialized view of the org
    HEAD, so a local delete is meaningless (the next pull restores it) and
    removing a skill for the organisation is an admin action, not a local one.
    """
    if action not in {"delete", "remove_file"}:
        return None
    try:
        from agent.skill_utils import is_org_mirror_path

        if is_org_mirror_path(skill_path, _skills_dir()):
            return {
                "success": False,
                "error": (
                    f"Cannot {action} '{name}' locally: it is shared by your "
                    "organisation, so a local delete would just come back on "
                    "the next sync. Ask an org admin to remove it for "
                    "everyone. (Editing it IS allowed — your changes are kept "
                    "and can be proposed back with `hermes sync propose "
                    f"{name}`.)"
                ),
            }
    except Exception:
        logger.debug("org mirror guard lookup failed for %s", name, exc_info=True)
    return None


def _find_skill_in_other_profiles(name: str) -> List[Tuple[str, Path]]:
    """``(profile, skill_dir)`` pairs for OTHER profiles holding ``name`` (so the not-found
    error can explain a wrong-profile mistake). Fail-quiet."""
    matches: List[Tuple[str, Path]] = []
    try:
        from hermes_constants import get_default_hermes_root
        root = get_default_hermes_root()
    except Exception:
        return matches
    _active = _skills_dir()
    active_dir = _active.resolve() if _active.exists() else _active
    # Every profile's skills dir EXCEPT the active one (already searched). A candidate whose
    # path cannot be resolved is skipped (not a fatal error); is_dir() checks stay unguarded.
    candidates: List[Tuple[str, Path]] = []
    with suppress(OSError, RuntimeError):
        if (root / "skills").resolve() != active_dir:
            candidates.append(("default", root / "skills"))
    if (root / "profiles").is_dir():
        with suppress(OSError):
            for entry in (root / "profiles").iterdir():
                if not entry.is_dir():
                    continue
                try:
                    if (entry / "skills").resolve() == active_dir:
                        continue
                except (OSError, RuntimeError):
                    continue
                candidates.append((entry.name, entry / "skills"))
    for profile_name, skills_dir in candidates:
        if not skills_dir.is_dir():
            continue
        with suppress(OSError):
            hit = next((d for d in _iter_skill_dirs(skills_dir) if d.name == name), None)
            if hit is not None:
                matches.append((profile_name, hit))  # one match per profile is enough
    return matches


def _skill_not_found_error(name: str, suffix: str = "") -> str:
    """Not-found error naming other profiles that hold the skill, plus ``suffix``."""
    from agent.file_safety import _resolve_active_profile_name
    base = f"Skill '{name}' not found in active profile '{_resolve_active_profile_name()}'."
    others = _find_skill_in_other_profiles(name)
    if others:
        if len(others) == 1:
            other_profile, other_path = others[0]
            base += (
                f" A skill by that name exists in profile "
                f"'{other_profile}' ({other_path}). To edit it, switch "
                f"profiles (`hermes -p {other_profile}`) or edit the file "
                f"directly (file tools / terminal)."
            )
        else:
            names = ", ".join(f"'{p}'" for p, _ in others)
            base += (
                f" Skills by that name exist in other profiles: {names}. "
                f"Switch profiles (`hermes -p <name>`) to edit there, or "
                f"edit the files directly (file tools / terminal)."
            )
    else:
        base += " Use skills_list() to see available skills."
    return base + suffix


def _validate_file_path(file_path: str) -> Optional[str]:
    """Validate a write_file/remove_file path: under an allowed subdir, no escape."""
    from tools.path_security import has_traversal_component
    if not file_path:
        return "file_path is required."
    parts = Path(file_path).parts
    # Traversal first, so the SKILL.md exception is unreachable by a traversal-laden path.
    if has_traversal_component(file_path):
        return "Path traversal ('..') is not allowed."
    # SKILL.md lives at the skill root; accept 'SKILL.md' and '<skill>/SKILL.md'.
    if parts and parts[-1] == "SKILL.md" and len(parts) in (1, 2):
        return None
    if not parts or parts[0] not in ALLOWED_SUBDIRS:
        allowed = ", ".join(sorted(ALLOWED_SUBDIRS))
        return f"File must be under one of: {allowed}. Got: '{file_path}'"
    if len(parts) < 2:
        return f"Provide a file path, not just a directory. Example: '{parts[0]}/myfile.md'"
    return None


def _resolve_supporting_file(skill_dir: Path, file_path: str):
    """Validate ``file_path`` and resolve it inside ``skill_dir``
    -> ``(target, None)`` | ``(None, error_dict)``."""
    from tools.path_security import validate_within_dir
    target = skill_dir / (file_path or "")
    err = _validate_file_path(file_path) or validate_within_dir(target, skill_dir)
    return (None, _err(err)) if err else (target, None)


def _locate_for_write(name: str, action: str, not_found_suffix: str = "", *,
                      org_guard: bool = True):
    """Find the skill; run the org-mirror (unless ``org_guard=False``) and background-review
    write guards -> ``(skill_dir, None)`` | ``(None, error_dict)``."""
    existing = _find_skill(name)
    if not existing:
        return None, _err(_skill_not_found_error(name, not_found_suffix))
    skill_dir = existing["path"]
    guard = ((org_guard and _org_mirror_write_guard(name, skill_dir, action))
             or _background_review_write_guard(name, skill_dir, action))
    return (None, guard) if guard else (skill_dir, None)


def _guarded_write(name: str, skill_dir: Path, target: Path, action: str, label: str,
                   content: str) -> Optional[Dict[str, Any]]:
    """Read-before-write guard (existing targets only), atomic write, then the security scan;
    a blocked scan restores the original (or unlinks a new file). Error dict or None."""
    original = None
    if target.exists():
        if read_guard := _background_review_read_before_write_guard(name, target, action, label):
            return read_guard
        original = target.read_text(encoding="utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, content, preserve_mode=True, create_mode=0o644)
    scan_error = _security_scan_skill(skill_dir)
    if not scan_error:
        return None
    if original is not None:
        atomic_write_text(target, original, preserve_mode=True)
    else:
        target.unlink(missing_ok=True)
    return _err(scan_error)


def _attach_org_note(result: Dict[str, Any], name: str, skill_dir: Path) -> Dict[str, Any]:
    if org_note := _maybe_auto_propose_org_edit(name, skill_dir):
        result["org_sharing"] = org_note
        result["message"] = f"{result['message']} {org_note}"
    return result


def _add_description_prompt_preview(result: Dict[str, Any], content: str) -> Dict[str, Any]:
    fm, _ = _parse_frontmatter(content)
    if is_skill_description_truncated_for_prompt(fm):
        result["system_prompt_preview"] = (
            f"System prompt will show: \"{extract_skill_description(fm)}\" — "
            f"keep the trigger self-contained in the first "
            f"{SKILL_PROMPT_DESC_LIMIT - 3} chars."
        )


def _create_skill(name: str, content: str, category: str = None) -> Dict[str, Any]:
    """Create a new user skill with SKILL.md content."""
    # Validate name
    err = _validate_name(name)
    if err:
        return {"success": False, "error": err}

    err = _validate_category(category)
    if err:
        return {"success": False, "error": err}

    # Validate content
    err = _validate_frontmatter(content, new_skill=True)
    if err:
        return {"success": False, "error": err}

    err = _validate_content_size(content)
    if err:
        return {"success": False, "error": err}

    # Check for name collisions across all directories
    existing = _find_skill(name)
    if existing:
        return {
            "success": False,
            "error": f"A skill named '{name}' already exists at {existing['path']}."
        }

    # Create the skill directory
    skill_dir = _resolve_skill_dir(name, category)
    skill_dir.mkdir(parents=True, exist_ok=True)

    # Write instructional documents with a readable mode while preserving
    # the mode of an existing file across the atomic replacement.
    skill_md = skill_dir / "SKILL.md"
    atomic_write_text(skill_md, content, preserve_mode=True, create_mode=0o644)

    # Security scan — roll back on block
    scan_error = _security_scan_skill(skill_dir)
    if scan_error:
        shutil.rmtree(skill_dir, ignore_errors=True)
        return {"success": False, "error": scan_error}

    # Extract description from frontmatter for verbose notifications
    _desc = ""
    try:
        _fm_end = re.search(r'\n---\s*\n', content[3:])
        if _fm_end:
            _parsed = yaml.safe_load(content[3:_fm_end.start() + 3])
            _desc = str(_parsed.get("description", ""))[:120]
    except Exception:
        pass

    try:
        _display_path = str(skill_dir.relative_to(_skills_dir()))
    except ValueError:
        # Skill created under skills.create_dir — not relative to the
        # profile-local root, so show the absolute path.
        _display_path = str(skill_dir)
    result = {
        "success": True,
        "message": f"Skill '{name}' created.",
        "path": _display_path,
        "skill_md": str(skill_md),
        "_change": {"description": _desc},
    }
    if category:
        result["category"] = category
    result["hint"] = (
        "To add reference files, templates, or scripts, use "
        "skill_manage(action='write_file', name='{}', file_path='references/example.md', file_content='...')".format(name)
    )
    _add_description_prompt_preview(result, content)
    _attach_lint_findings(result, skill_md)
    return result


def _attach_lint_findings(result: Dict[str, Any], skill_md: Path) -> None:
    """Attach ADVISORY authoring findings (hard rejects already ran in _validate_frontmatter)."""
    try:
        from tools.skill_linter import lint_skill  # local import: optional path
        findings = lint_skill(skill_md)
    except Exception:
        findings = None
    if not findings:
        return
    result["lint_warnings"] = [
        {"severity": f.severity, "rule": f.rule, "message": f.message} for f in findings]
    result["lint_hint"] = (
        "The skill was created. These are advisory authoring-convention findings (not blockers) "
        "— fix them with skill_manage(action='patch') to match Hermes skill standards.")


def _clip(text: str, n: int, ellipsis: str) -> str:
    return text[:n] + (ellipsis if len(text) > n else "")


# --- Core actions -------------------------------------------------------------

def _create_skill(name: str, content: str, category: str = None) -> Dict[str, Any]:
    if err := (_validate_name(name) or _validate_category(category)
               or _validate_frontmatter(content, new_skill=True) or _validate_content_size(content)):
        return _err(err)
    if existing := _find_skill(name):
        return _err(f"A skill named '{name}' already exists at {existing['path']}.")
    skill_dir = _resolve_skill_dir(name, category)
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = skill_dir / "SKILL.md"
    atomic_write_text(skill_md, content, preserve_mode=True, create_mode=0o644)
    if scan_error := _security_scan_skill(skill_dir):
        shutil.rmtree(skill_dir, ignore_errors=True)
        return _err(scan_error)
    root = _skills_dir()  # display relative under the profile dir; absolute under skills.create_dir
    display = skill_dir.relative_to(root) if skill_dir.is_relative_to(root) else skill_dir
    result = {
        "success": True, "message": f"Skill '{name}' created.", "path": str(display),
        "skill_md": str(skill_md), "_change": {"description": _description_preview(content)},
        **({"category": category} if category else {}),
        "hint": "To add reference files, templates, or scripts, use "
                f"skill_manage(action='write_file', name='{name}', file_path='references/example.md', "
                "file_content='...')"}
    _attach_lint_findings(_add_description_prompt_preview(result, content), skill_md)
    return result


def _edit_skill(name: str, content: str) -> Dict[str, Any]:
    """Replace the SKILL.md of any existing skill (full rewrite)."""
    if err := _validate_frontmatter(content) or _validate_content_size(content):
        return _err(err)
    skill_dir, guard = _locate_for_write(name, "edit")
    # SKILL.md always exists here (_find_skill requires it), so a blocked scan restores it.
    if guard := guard or _guarded_write(name, skill_dir, skill_dir / "SKILL.md", "edit", "SKILL.md", content):
        return guard

    skill_md = existing["path"] / "SKILL.md"
    read_guard = _background_review_read_before_write_guard(
        name, skill_md, "edit", "SKILL.md"
    )
    if read_guard:
        return read_guard

    # Back up original content for rollback
    original_content = skill_md.read_text(encoding="utf-8") if skill_md.exists() else None
    atomic_write_text(skill_md, content, preserve_mode=True, create_mode=0o644)

    # Security scan — roll back on block
    scan_error = _security_scan_skill(existing["path"])
    if scan_error:
        if original_content is not None:
            atomic_write_text(skill_md, original_content, preserve_mode=True)
        return {"success": False, "error": scan_error}

    # Extract description from new content for verbose notifications
    _desc = ""
    try:
        _fm_end = re.search(r'\n---\s*\n', content[3:])
        if _fm_end:
            _parsed = yaml.safe_load(content[3:_fm_end.start() + 3])
            _desc = str(_parsed.get("description", ""))[:120]
    except Exception:
        pass

    result = {
        "success": True, "message": f"Skill '{name}' updated (full rewrite).",
        "path": str(skill_dir), "_change": {"description": _description_preview(content)}}
    return _add_description_prompt_preview(_attach_org_note(result, name, skill_dir), content)


def _patch_skill(name: str, old_string: str, new_string: str, file_path: str = None,
                 replace_all: bool = False) -> Dict[str, Any]:
    """Targeted find-and-replace in SKILL.md (default) or a supporting file; unique match unless replace_all."""
    if not old_string:
        # A bare "required" error is a dead end: the model cannot tell whether it
        # omitted the arg or supplied it wrongly, so it retries blindly and often
        # escapes to action='write_file', clobbering the whole skill file. Tell it
        # how to recover. Upstream: NousResearch/hermes-agent#33064.
        return {
            "success": False,
            "error": (
                "old_string is required for 'patch' and must be the EXACT text currently in the "
                "file. Read the target file first (read_file on the skill's SKILL.md, or the file "
                "named by file_path) and copy the snippet verbatim, then retry 'patch'. "
                "Do NOT fall back to action='write_file' — that rewrites the entire file and "
                "destroys unrelated content."
            ),
        }
    if new_string is None:
        return {"success": False, "error": "new_string is required for 'patch'. Use an empty string to delete matched text."}
    # No old_string == new_string guard here: fuzzy_find_and_replace already
    # rejects that with "old_string and new_string are identical"
    # (tools/fuzzy_match.py), and its error carries a file_preview this layer
    # cannot produce. Duplicating it here would only shadow the richer message.

    existing = _find_skill(name)
    if not existing:
        return {"success": False, "error": _skill_not_found_error(name)}

    skill_dir = existing["path"]
    org_guard = _org_mirror_write_guard(name, skill_dir, "patch")
    if org_guard:
        return org_guard
    guard = _background_review_write_guard(name, skill_dir, "patch")
    if guard:
        return guard
    target_label = file_path or "SKILL.md"
    if file_path:
        target, err = _resolve_supporting_file(skill_dir, file_path)
        if err:
            return err
    else:
        target = skill_dir / "SKILL.md"
    if not target.exists():
        return _err(f"File not found: {target.relative_to(skill_dir)}")
    if read_guard := _background_review_read_before_write_guard(name, target, "patch", target_label):
        return read_guard
    content = target.read_text(encoding="utf-8")
    # Same fuzzy engine as the file patch tool (whitespace/indent/escape normalization,
    # block anchors) so minor formatting mismatches don't fail.
    from tools.fuzzy_match import fuzzy_find_and_replace
    new_content, match_count, _strategy, match_error = fuzzy_find_and_replace(
        content, old_string, new_string, replace_all)
    if match_error:
        with suppress(Exception):
            from tools.fuzzy_match import format_no_match_hint
            err_msg += format_no_match_hint(match_error, match_count, old_string, content)
        except Exception:
            pass
        return {
            "success": False,
            "error": err_msg,
            "file_preview": preview,
        }

    # Check size limit on the result
    target_label = "SKILL.md" if not file_path else file_path
    err = _validate_content_size(new_content, label=target_label)
    if err:
        return {"success": False, "error": err}

    # If patching SKILL.md, validate frontmatter is still intact
    if not file_path:
        err = _validate_frontmatter(new_content)
        if err:
            return {
                "success": False,
                "error": f"Patch would break SKILL.md structure: {err}",
            }

    original_content = content  # for rollback
    atomic_write_text(target, new_content, preserve_mode=True, create_mode=0o644)

    # Security scan — roll back on block
    scan_error = _security_scan_skill(skill_dir)
    if scan_error:
        atomic_write_text(target, original_content, preserve_mode=True)
        return {"success": False, "error": scan_error}

    result = {
        "success": True,
        "message": f"Patched {target_label} in skill '{name}' ({match_count} replacement{'s' if match_count > 1 else ''}).",
        "_change": {"old": _clip(old_string, 200, "…"), "new": _clip(new_string, 200, "…")}}
    return _attach_org_note(result, name, skill_dir)


def _delete_skill(name: str, absorbed_into: Optional[str] = None) -> Dict[str, Any]:
    """Delete a skill. ``absorbed_into``: None = undeclared (legacy, accepted); "" = explicit prune;
    "<skill>" = absorbed into that umbrella, which must exist (so the model can't claim one)."""
    skill_dir, guard = _locate_for_write(name, "delete")
    if guard := guard or _curator_consolidation_delete_guard(name, absorbed_into):
        return guard
    if pinned_err := _pinned_guard(name):
        return _err(pinned_err)
    absorbed_target = absorbed_into.strip() if isinstance(absorbed_into, str) else ""
    if absorbed_target:
        if absorbed_target == name:
            return _err(f"absorbed_into='{absorbed_target}' cannot equal the skill being deleted.")
        if not _find_skill(absorbed_target):
            return _err(f"absorbed_into='{absorbed_target}' does not exist. "
                        f"Create or patch the umbrella skill first, then retry the delete.")
    skills_root = _containing_skills_root(skill_dir)
    if unsafe := _validate_delete_target(skill_dir):  # defense-in-depth before rmtree
        return _err(unsafe)
    # Curator consolidations must be RECOVERABLE (`hermes curator restore`): archive instead
    # of rmtree. Foreground deletes keep hard-delete semantics.
    absorbed_note = f" Content absorbed into '{absorbed_target}'." if absorbed_target else ""
    if _is_background_review():
        try:
            from tools.skill_usage import archive_skill
            ok, archive_msg = archive_skill(name)
        except Exception as e:
            return _err(f"failed to archive '{name}': {e}")
        if not ok:
            return _err(archive_msg)
        return {"success": True,
                "message": f"Skill '{name}' archived ({archive_msg}).{absorbed_note}",
                "_archived": True}
    shutil.rmtree(skill_dir)
    _rmdir_if_empty(skill_dir.parent, skills_root)  # empty category dir, never the root
    return {"success": True, "message": f"Skill '{name}' deleted.{absorbed_note}"}


def _rmdir_if_empty(parent: Path, stop: Path) -> None:
    if parent != stop and parent.exists() and not any(parent.iterdir()):
        parent.rmdir()


def _write_file(name: str, file_path: str, file_content: str) -> Dict[str, Any]:
    """Add or overwrite a supporting file within any skill directory."""
    if err := _validate_file_path(file_path):
        return _err(err)
    if not file_content and file_content != "":
        return _err("file_content is required.")
    if (content_bytes := len(file_content.encode("utf-8"))) > MAX_SKILL_FILE_BYTES:
        return _err(f"File content is {content_bytes:,} bytes (limit: {MAX_SKILL_FILE_BYTES:,} "
                    f"bytes / 1 MiB). Consider splitting into smaller files.")
    if err := _validate_content_size(file_content, label=file_path):
        return _err(err)
    skill_dir, guard = _locate_for_write(name, "write_file", " Create it first with action='create'.")
    if guard:
        return guard

    target, err = _resolve_skill_target(existing["path"], file_path)
    if err:
        return {"success": False, "error": err}
    assert target is not None
    if target.exists():
        read_guard = _background_review_read_before_write_guard(
            name, target, "write_file", file_path
        )
        if read_guard:
            return read_guard
    target.parent.mkdir(parents=True, exist_ok=True)
    # Back up for rollback
    original_content = target.read_text(encoding="utf-8") if target.exists() else None
    atomic_write_text(target, file_content, preserve_mode=True, create_mode=0o644)

    # Security scan — roll back on block
    scan_error = _security_scan_skill(existing["path"])
    if scan_error:
        if original_content is not None:
            atomic_write_text(target, original_content, preserve_mode=True)
        else:
            target.unlink(missing_ok=True)
        return {"success": False, "error": scan_error}

    result = {
        "success": True,
        "message": f"File '{file_path}' written to skill '{name}'.",
        "path": str(target),
    }
    org_note = _maybe_auto_propose_org_edit(name, existing["path"])
    if org_note:
        result["org_sharing"] = org_note
        result["message"] = f"{result['message']} {org_note}"
    return result


def _remove_file(name: str, file_path: str) -> Dict[str, Any]:
    """Remove a supporting file from any skill directory."""
    if err := _validate_file_path(file_path):
        return _err(err)
    skill_dir, guard = _locate_for_write(name, "remove_file", org_guard=False)
    if guard:
        return guard
    target, err = _resolve_supporting_file(skill_dir, file_path)
    if err:
        return err
    if not target.exists():  # list what IS there so the model can pick the right path
        available = [str(f.relative_to(skill_dir)) for subdir in ALLOWED_SUBDIRS
                     if (skill_dir / subdir).exists() for f in (skill_dir / subdir).rglob("*") if f.is_file()]
        return _err(f"File '{file_path}' not found in skill '{name}'.", available_files=available or None)
    if read_guard := _background_review_read_before_write_guard(name, target, "remove_file", file_path):
        return read_guard
    target.unlink()
    _rmdir_if_empty(target.parent, skill_dir)
    return {"success": True, "message": f"File '{file_path}' removed from skill '{name}'."}


# --- Main entry point ---------------------------------------------------------

# Set while replaying an approved staged skill write so skill_manage() does not re-gate it.
_skill_gate_bypass: "_ctxvars.ContextVar[bool]" = _ctxvars.ContextVar(
    "skill_gate_bypass", default=False)


def _run_write_gate(build_staging):
    """Shared write gate: None to proceed, else a JSON tool result (blocked/staged).
    ``build_staging(wa) -> (payload, gist)`` runs only when staging. Fails open if
    write_approval cannot be imported."""
    try:
        from tools import write_approval as wa
    except Exception:
        return None  # fail open
    decision = wa.evaluate_gate(wa.SKILLS)
    if decision.allow:
        return None
    if decision.blocked:
        return tool_error(decision.message, success=False)
    payload, gist = build_staging(wa)
    record = wa.stage_write(wa.SKILLS, payload, summary=gist, origin=wa.current_origin())
    return json.dumps({"success": True, "staged": True, "pending_id": record["id"],
                       "gist": gist, "message": decision.message}, ensure_ascii=False)


def _apply_skill_write_gate(action, name, **payload_kwargs):
    """Flat-shape gate: stage the full kwargs so approval can replay them; bypassed during replay."""
    if action not in _ACTION_HANDLERS or _skill_gate_bypass.get():
        return None
    def _staging(wa):
        payload = {"action": action, "name": name,
                   **{k: v for k, v in payload_kwargs.items() if v is not None}}
        gist_kw = {k: payload_kwargs.get(k) or ""
                   for k in ("content", "file_path", "old_string", "new_string")}
        return payload, wa.skill_gist(action, name, **gist_kw)
    return _run_write_gate(_staging)


_FLAT_OP_KEYS = ("content", "category", "file_path", "file_content", "old_string", "new_string",
                 "absorbed_into", "operations")


def _skill_manage_from(payload: Dict[str, Any], **extra) -> str:
    """Call ``skill_manage`` with the flat-shape fields (and absorbed_into/operations) of ``payload``."""
    return skill_manage(
        action=payload.get("action", ""), name=payload.get("name", ""),
        replace_all=payload.get("replace_all", False),
        **{k: payload.get(k) for k in _FLAT_OP_KEYS}, **extra)


def apply_skill_pending(payload: Dict[str, Any]) -> str:
    """Replay a staged skill write, bypassing the gate (the /skills approve handler)."""
    token = _skill_gate_bypass.set(True)
    try:
        return skill_manage(
            action=payload.get("action", ""),
            name=payload.get("name", ""),
            content=payload.get("content"),
            category=payload.get("category"),
            file_path=payload.get("file_path"),
            file_content=payload.get("file_content"),
            old_string=payload.get("old_string"),
            new_string=payload.get("new_string"),
            replace_all=payload.get("replace_all", False),
            absorbed_into=payload.get("absorbed_into"),
            operations=payload.get("operations"),
        )
    finally:
        _skill_gate_bypass.reset(token)


_BATCH_OP_ACTIONS = {"create", "patch", "write_file", "remove_file"}
_BATCH_MAX_OPS = 20


def _skill_manage_batch(
    operations,
    default_name: str = None,
    task_id: str = None,
    session_id: str = None,
) -> str:
    """Apply a sequence of operations atomically (memory-tool pattern).

    Each op carries its own ``name`` (skill) and ``action``; a single edit
    is a list of one. Every skill the batch touches is snapshotted before
    any op runs; any failure rolls ALL touched skills back to their
    pre-batch state (skills the batch created are removed).

    Rules:
    - ``delete`` only as the SOLE op of the call (its recoverable-archive
      path doesn't compose with rollback) — routed to the single-op
      handler, preserving absorbed_into/archive semantics;
    - ``create`` for a skill must precede that skill's other ops;
    - same-file clobber guard (below) rejects silently-lost work.

    ``default_name``: legacy top-level ``name`` fallback for ops that omit
    their own (staged-replay / back-compat path).
    """
    import shutil
    import tempfile

    # --- validate shape up front (no side effects before this passes) ---
    if not isinstance(operations, list) or not operations:
        return tool_error("operations must be a non-empty array.", success=False)
    if len(operations) > _BATCH_MAX_OPS:
        return tool_error(f"operations is capped at {_BATCH_MAX_OPS} ops per call.", success=False)
    # delete: sole-op only; route through the normal single-op path so the
    # gate, archive, ledger, and curator absorbed_into semantics all apply.
    if any(isinstance(op, dict) and op.get("action") == "delete" for op in operations):
        if len(operations) != 1:
            return tool_error(
                "delete must be the SOLE op in its call — it doesn't "
                "compose with other ops' rollback.",
                success=False,
            )
        op = operations[0]
        nm = op.get("name") or default_name
        if not nm:
            return tool_error("operations[0] (delete) needs a 'name'.", success=False)
        return skill_manage(
            action="delete",
            name=nm,
            absorbed_into=op.get("absorbed_into"),
            task_id=task_id,
            session_id=session_id,
        )
    names = []
    for i, op in enumerate(operations):
        if not isinstance(op, dict) or not op.get("action"):
            return tool_error(f"operations[{i}] needs an 'action'.", success=False)
        act = op["action"]
        if act not in _BATCH_OP_ACTIONS:
            return tool_error(
                f"operations[{i}]: unknown action '{act}'. "
                f"Batchable: {', '.join(sorted(_BATCH_OP_ACTIONS))}; "
                "delete must be sole.",
                success=False,
            )
        nm = op.get("name") or default_name
        if not nm:
            return tool_error(f"operations[{i}] needs a 'name' (the skill it targets).", success=False)
        names.append(nm)
        if act == "create" and nm in names[:-1]:
            return tool_error(
                f"operations[{i}]: create for '{nm}' must precede that "
                "skill's other ops.",
                success=False,
            )
        preflight = _background_review_preflight(act, nm)
        if preflight is not None:
            return json.dumps(preflight, ensure_ascii=False)

    # --- intra-batch conflict guard: sequential last-wins semantics make
    # these SILENTLY succeed while discarding earlier ops' work — always a
    # confused plan, never intentional. Rule: a DESTRUCTIVE op (write_file,
    # remove_file, full SKILL.md rewrite) on a file some earlier op in the
    # batch already touched is rejected; ADDITIVE patches are always legal,
    # so patch CHAINS (each op building on the previous text) and
    # write-then-patch both stay allowed. Paths are normalized so spelling
    # variants ('./references/x.md', 'references//x.md') can't slip past. ---
    import posixpath

    def _norm_target(op) -> str:
        fp = (op.get("file_path") or "").strip()
        if not fp:
            return "SKILL.md"
        return posixpath.normpath(fp.lstrip("/"))

    touched_files = set()  # (skill, normalized path) touched by ANY earlier op
    for i, op in enumerate(operations):
        act = op["action"]
        nm = names[i]
        # create and full-rewrite patch (content) always hit SKILL.md —
        # _edit_skill ignores file_path on the rewrite shape.
        full_rewrite = act == "patch" and bool(op.get("content"))
        target = "SKILL.md" if (act == "create" or full_rewrite) else _norm_target(op)
        key = (nm, target)
        destructive = act in ("create", "write_file", "remove_file") or full_rewrite
        if destructive and key in touched_files:
            return tool_error(
                f"operations[{i}]: {act} on '{target}' of skill '{nm}' — an "
                "earlier op in this batch already touched that file, and this "
                "op would silently discard its work. One destructive op "
                "(write_file/remove_file/full rewrite) per file per batch; "
                "put it first, or fold the change in. Patch chains are fine.",
                success=False,
            )
        touched_files.add(key)

    # --- approval gate: stage the WHOLE batch as one pending write ---
    if not _skill_gate_bypass.get():
        try:
            from tools import write_approval as wa
        except Exception:
            wa = None  # fail open, matching _apply_skill_write_gate
        if wa is not None:
            decision = wa.evaluate_gate(wa.SKILLS)
            if decision.blocked:
                return tool_error(decision.message, success=False)
            if not decision.allow:
                payload = {"action": "batch", "operations": operations}
                acts = ", ".join(op["action"] for op in operations)
                skills = ", ".join(sorted(set(names)))
                gist = f"batch({len(operations)} ops: {acts}) on {skills}"
                record = wa.stage_write(
                    wa.SKILLS, payload, summary=gist, origin=wa.current_origin()
                )
                return json.dumps(
                    {"success": True, "staged": True, "pending_id": record["id"],
                     "gist": gist, "message": decision.message},
                    ensure_ascii=False,
                )

    # --- snapshot every touched skill for rollback ---
    snap_root = Path(tempfile.mkdtemp(prefix="skill_batch_"))
    snapshots = {}  # skill name -> (pre_dir or None, snapshot_dir or None)
    for nm in dict.fromkeys(names):  # ordered unique
        pre = _find_skill(nm)
        pre_dir = Path(pre["path"]) if pre else None
        snap = None
        if pre_dir is not None and pre_dir.is_dir():
            snap = snap_root / nm
            try:
                shutil.copytree(pre_dir, snap)
            except Exception as exc:  # noqa: BLE001 — no snapshot, no atomicity
                shutil.rmtree(snap_root, ignore_errors=True)
                return tool_error(f"Could not snapshot '{nm}' for atomic batch: {exc}", success=False)
        snapshots[nm] = (pre_dir, snap)

    rollback_failed = False

    def _rollback() -> str:
        notes = []
        for nm, (pre_dir, snap) in snapshots.items():
            try:
                post = _find_skill(nm)
                post_dir = Path(post["path"]) if post else None
                if snap is not None:
                    if post_dir is not None and post_dir.is_dir():
                        # Never destroy the only other copy before the
                        # restore lands. Deleting first turned a failed
                        # copytree (disk full, locked file) into total
                        # skill loss once the finally below removed the
                        # snapshot too. Move the broken state aside, and
                        # delete it only after the snapshot is back.
                        aside = post_dir.with_name(post_dir.name + ".rollback-broken")
                        shutil.rmtree(aside, ignore_errors=True)
                        post_dir.rename(aside)
                        try:
                            shutil.copytree(snap, pre_dir)
                        except Exception:
                            # Restore failed: put the broken state back so
                            # the skill survives (half applied) rather than
                            # leaving nothing.
                            shutil.rmtree(pre_dir, ignore_errors=True)
                            aside.rename(pre_dir)
                            raise
                        shutil.rmtree(aside, ignore_errors=True)
                    else:
                        shutil.copytree(snap, pre_dir)
                elif post_dir is not None and post_dir.is_dir():
                    # Batch created this skill: remove the partial result.
                    shutil.rmtree(post_dir)
            except Exception as exc:  # noqa: BLE001
                notes.append(
                    f"ROLLBACK FAILED for '{nm}' ({exc}); snapshot preserved at '{snap}'"
                    if snap is not None
                    else f"ROLLBACK FAILED for '{nm}' ({exc})"
                )
        nonlocal rollback_failed
        rollback_failed = bool(notes)
        return "; ".join(notes) if notes else "all touched skills rolled back"

    # --- execute ops through the normal single-op path (gate bypassed:
    #     the batch already cleared/staged it above; ledger + telemetry
    #     fire per-op, which is the audit granularity we want) ---
    results = []
    token = _skill_gate_bypass.set(True)
    try:
        for i, op in enumerate(operations):
            raw = skill_manage(
                action=op["action"],
                name=names[i],
                content=op.get("content"),
                category=op.get("category"),
                file_path=op.get("file_path"),
                file_content=op.get("file_content"),
                old_string=op.get("old_string"),
                new_string=op.get("new_string"),
                replace_all=op.get("replace_all", False),
                task_id=task_id,
                session_id=session_id,
            )
            try:
                parsed = json.loads(raw)
            except Exception:  # noqa: BLE001
                parsed = {"success": False, "error": "unparseable op result"}
            if not parsed.get("success"):
                note = _rollback()
                fail = {
                    "success": False,
                    "error": (
                        f"operations[{i}] ({op['action']} on '{names[i]}') failed: "
                        f"{parsed.get('error', 'unknown error')} — batch aborted, {note}."
                    ),
                    "failed_index": i,
                    "completed_before_failure": i,
                }
                # Carry the failing op's teaching payload through (e.g.
                # patch's file_preview / fuzzy-match hints): without it the
                # model recovers blind — live A/B showed sonnet probing a
                # file with placeholder edits for 8 turns because the batch
                # path dropped the preview the flat path always returned.
                for k, v in parsed.items():
                    if k not in ("success", "error") and v is not None:
                        fail.setdefault(k, v)
                return json.dumps(fail, ensure_ascii=False)
            results.append({"name": names[i], "action": op["action"],
                            "file_path": op.get("file_path"),
                            "success": True})
    finally:
        _skill_gate_bypass.reset(token)
        if rollback_failed:
            # Keep the snapshots so the operator can still recover by
            # hand. Deleting them here is what turned one failed restore
            # into permanent skill loss.
            logger.warning(
                "skill_manage batch rollback failed, snapshots kept at %s",
                snap_root,
            )
        else:
            shutil.rmtree(snap_root, ignore_errors=True)

    return json.dumps(
        {"success": True, "operations_applied": len(results),
         "results": results},
        ensure_ascii=False,
    )


# Debounce state for the sync push hook. A burst of skill_manage writes
# (e.g. create + several write_file calls) collapses into a single push after
# a short quiet window, on a daemon timer so the agent write never blocks.
_sync_push_timer = None
_sync_push_lock = None
_SYNC_PUSH_DEBOUNCE_S = 5.0


def _maybe_debounced_sync_push(skill_name: str) -> None:
    """Debounced best-effort sync push after a skill write; never blocks the caller. Skills not
    opted into sync do nothing (no auth/network); ``maybe_push_skills`` enforces the access gate."""
    try:
        from tools.skill_usage import is_sync_enabled
        if not is_sync_enabled(skill_name):
            return
    except Exception:
        return
    from hermes_constants import hermes_home_key
    home_key = hermes_home_key()
    # Timer threads start with empty ContextVars; without the scheduling turn's context the push would
    # resolve the launch profile's home and credentials instead of the writing profile's.
    ctx = _ctxvars.copy_context()
    def _fire():
        with suppress(Exception):
            from tools.skills_sync_client import maybe_push_skills
            maybe_push_skills(message=f"sync: {skill_name}")
    with _sync_push_lock:
        pending = _sync_push_timers.get(home_key)
        if pending is not None:
            pending.cancel()  # only sets an Event; never raises
        timer = threading.Timer(_SYNC_PUSH_DEBOUNCE_S, ctx.run, args=(_fire,))
        timer.daemon = True
        _sync_push_timers[home_key] = timer
        timer.start()


def _act_patch(a):
    """Two shapes: old_string/new_string = targeted replacement (validated in _patch_skill so the
    tool and the helper give the same guidance); content alone = full rewrite (the old 'edit')."""
    if a["content"] and (a["old_string"] or a["new_string"] is not None):
        return tool_error("Pass EITHER content (full SKILL.md rewrite) OR "
                          "old_string/new_string (targeted replacement), not both.", success=False)
    if a["content"]:
        return _edit_skill(a["name"], a["content"])
    return _patch_skill(a["name"], a["old_string"], a["new_string"], a["file_path"], a["replace_all"])


# action -> handler(args dict) returning a result dict, or a tool_error JSON string for
# argument-shape errors. "edit" is a legacy alias for a full rewrite (not in the schema).
_ACTION_HANDLERS = {
    "create": lambda a: _create_skill(a["name"], a["content"], a["category"]),
    "edit": lambda a: _edit_skill(a["name"], a["content"]),
    "patch": _act_patch,
    "delete": lambda a: _delete_skill(a["name"], absorbed_into=a["absorbed_into"]),
    "write_file": lambda a: _write_file(a["name"], a["file_path"], a["file_content"]),
    "remove_file": lambda a: _remove_file(a["name"], a["file_path"])}
# action -> (arg, is_missing, error) argument-shape checks run before the handler.
_MISSING, _IS_NONE = (lambda v: not v), (lambda v: v is None)
_REQUIRED_ARGS = {
    "create": [("content", _MISSING,
                "content is required for 'create'. Provide the full SKILL.md text (frontmatter + body).")],
    "edit": [("content", _MISSING,
              "content is required for a full rewrite. Provide the full updated SKILL.md text.")],
    "write_file": [
        ("file_path", _MISSING, "file_path is required for 'write_file'. Example: 'references/api-guide.md'"),
        ("file_content", _IS_NONE, "file_content is required for 'write_file'.")],
    "remove_file": [("file_path", _MISSING, "file_path is required for 'remove_file'.")]}


def _record_success(action, name, result, *, file_path, absorbed_into, task_id,
                    session_id, ledger_before) -> None:
    """Best-effort post-mutation side effects (never break the tool): ledger, prompt-cache
    clear, curator telemetry, debounced sync push."""
    with suppress(Exception):
        from tools import skill_ledger as _ledger
        _post = _find_skill(name)
        # delete: consolidation vs prune, and whether the recoverable archive handled it
        _evidence = ({"absorbed_into": absorbed_into, "archived": bool(result.get("_archived"))}
                     if action == "delete" else {})
        _evidence.update({k: v for k, v in (("session_id", session_id), ("file_path", file_path)) if v})
        _ledger.record_mutation(
            action, name, before=ledger_before if ledger_before is not None else [],
            after_root=_post["path"] if _post else None, evidence=_evidence)
    with suppress(Exception):
        from agent.prompt_builder import clear_skills_system_prompt_cache
        clear_skills_system_prompt_cache(clear_snapshot=True)
    # Curator telemetry: only the background review fork marks a skill agent-created
    # (foreground creates belong to the user). A recoverable curator archive keeps its
    # record as STATE_ARCHIVED (`hermes curator status`/`restore`); only a hard delete forgets.
    with suppress(Exception):
        from tools.skill_usage import bump_patch, forget, record_created
        # During the curator consolidation pass, a verified consolidation must be RECOVERABLE: archival into
        # ~/.hermes/skills/.archive/ is documented as the maximum destructive action the curator may take,
        # and `hermes curator restore` promises the skill can be brought back. Route through the recoverable
        # archive primitive instead of permanent rmtree so a misjudged consolidation can be undone (#29912).
        # Foreground, user-directed deletes keep their existing hard-delete semantics.
        from tools.skill_provenance import is_background_review
        if action == "create":
            record_created(name, agent_created=is_background_review(),
                           task_id=task_id, session_id=session_id)
        elif action in {"patch", "edit", "write_file", "remove_file"}:
            bump_patch(name, action=action, task_id=task_id, session_id=session_id)
        elif action == "delete" and not result.get("_archived"):
            forget(name)
    # Only AFTER the write gate passed (staged writes returned early): never push un-reviewed content.
    with suppress(Exception):
        _maybe_debounced_sync_push(name)


def skill_manage(
    action: str,
    name: str,
    content: str = None,
    category: str = None,
    file_path: str = None,
    file_content: str = None,
    old_string: str = None,
    new_string: str = None,
    replace_all: bool = False,
    absorbed_into: str = None,
    task_id: str = None,
    session_id: str = None,
    operations=None,
) -> str:
    """
    Manage user-created skills. Dispatches to the appropriate action handler.

    ``operations``: batch shape — a list of {action, ...} dicts applied to
    ONE skill atomically (see _skill_manage_batch). When set, the flat
    single-op fields are ignored and ``action`` may be omitted/'batch'.

    Returns JSON string with results.
    """
    if operations is not None:
        return _skill_manage_batch(
            operations, default_name=name or None,
            task_id=task_id, session_id=session_id,
        )
    preflight = _background_review_preflight(action, name)
    if preflight is not None:
        return json.dumps(preflight, ensure_ascii=False)
    # Approval gate: skills are too large to review inline, so they always stage regardless
    # of origin; bypassed when replaying an approved staged write.
    args = dict(content=content, category=category, file_path=file_path, file_content=file_content,
                old_string=old_string, new_string=new_string, replace_all=replace_all,
                absorbed_into=absorbed_into)
    if (gate_result := _apply_skill_write_gate(action, name, **args)) is not None:
        return gate_result
    # Ledger pre-capture: telemetry, not a gate — failures must NEVER block the mutation. delete
    # destroys the whole package (consolidation may have re-homed support files first), so
    # complete it from the newest curator backup or a restore is hollow.
    # Audit ledger (tracker #79686 P3): capture the pre-mutation state of the skill directory so every
    # mutation — any actor — lands in the append-only JSONL ledger with before/after blobs.
    _ledger_before = None
    with suppress(Exception):
        from tools import skill_ledger as _ledger
        _pre = _find_skill(name)
        _ledger_before_dir = _pre["path"] if _pre else None
        _ledger_before = _ledger.capture_before(_ledger_before_dir)
    except Exception:
        pass

    if action == "create":
        if not content:
            return tool_error("content is required for 'create'. Provide the full SKILL.md text (frontmatter + body).", success=False)
        result = _create_skill(name, content, category)

    elif action == "edit":
        # Legacy alias for a full rewrite (kept for old transcripts/callers;
        # no longer advertised in the schema — use patch with `content`).
        if not content:
            return tool_error("content is required for a full rewrite. Provide the full updated SKILL.md text.", success=False)
        result = _edit_skill(name, content)

    elif action == "patch":
        # Two shapes: old_string/new_string = targeted replacement;
        # content (alone) = full SKILL.md rewrite (absorbs the old 'edit').
        if content and (old_string or new_string is not None):
            return tool_error(
                "Pass EITHER content (full SKILL.md rewrite) OR "
                "old_string/new_string (targeted replacement), not both.",
                success=False,
            )
        if content:
            result = _edit_skill(name, content)
        else:
            # Targeted-replacement validation lives in _patch_skill so the
            # public tool and the helper return the same actionable guidance.
            # A bare "required" error here would shadow it and leave the
            # model with nowhere to go but action='write_file'. #33064.
            result = _patch_skill(name, old_string, new_string, file_path, replace_all)

    elif action == "delete":
        result = _delete_skill(name, absorbed_into=absorbed_into)

    elif action == "write_file":
        if not file_path:
            return tool_error("file_path is required for 'write_file'. Example: 'references/api-guide.md'", success=False)
        if file_content is None:
            return tool_error("file_content is required for 'write_file'.", success=False)
        result = _write_file(name, file_path, file_content)

    elif action == "remove_file":
        if not file_path:
            return tool_error("file_path is required for 'remove_file'.", success=False)
        result = _remove_file(name, file_path)

    else:
        result = {"success": False, "error": f"Unknown action '{action}'. Use: create, edit, patch, delete, write_file, remove_file"}

    if result.get("success"):
        _record_success(
            action, name, result, file_path=file_path, absorbed_into=absorbed_into,
            task_id=task_id, session_id=session_id, ledger_before=_ledger_before)
    return json.dumps(result, ensure_ascii=False)


# --- OpenAI Function-Calling Schema -------------------------------------------

def _skill_manage_description(create_dir: str) -> str:
    return (
        "Create, update, or delete skills — your procedural memory for "
        "recurring task types. The call is an operations array (a single "
        "edit is a list of one); it applies atomically — any failure rolls "
        "every touched skill back. Ops: create (full SKILL.md; lands in "
        f"{create_dir}; must precede that skill's other "
        "ops), patch (targeted old_string/new_string fix — preferred; "
        "content alone REPLACES the whole file, read it via skill_view() "
        "first), write_file/remove_file (supporting files), delete (sole "
        "op only). Existing skills are modified wherever they live. Keep "
        "the description's first 57 chars a self-contained trigger: 'Use "
        "when <trigger>. <one-line behavior>.' Write lessons, not logs: "
        "imperative rule + why, no PR numbers/dates/incident narration, one "
        "rule per lesson, references/ named by topic (extend before adding). "
        "skill_view() shows format conventions."
    )


def _skill_manage_schema_overrides() -> dict:
    """Rebuild the create-dir hint from the ACTIVE profile at every get_definitions(): the
    multiplexed gateway serves every profile from one process, so a path baked in at import
    would name the launch profile's skills dir for everyone else (#95685)."""
    return {"description": _skill_manage_description(_display_create_dir())}


SKILL_MANAGE_SCHEMA = {
    "name": "skill_manage",
    # ONE call shape (memory-tool pattern, maintainer-directed): the call
    # IS an operations array — each op names its skill and action; a
    # single edit is a list of one. The legacy flat shape (top-level
    # action/name/content/...) is still ACCEPTED by the handler for old
    # transcripts and staged-write replay, but no longer advertised.
    "description": (
        "Create, update, or delete skills — your procedural memory for "
        "recurring task types. The call is an operations array (a single "
        "edit is a list of one); it applies atomically — any failure rolls "
        "every touched skill back. Ops: create (full SKILL.md; lands in "
        f"{_display_create_dir()}; must precede that skill's other "
        "ops), patch (targeted old_string/new_string fix — preferred; "
        "content alone REPLACES the whole file, read it via skill_view() "
        "first), write_file/remove_file (supporting files), delete (sole "
        "op only). Existing skills are modified wherever they live. Keep "
        "the description's first 57 chars a self-contained trigger: 'Use "
        "when <trigger>. <one-line behavior>.' — skill_view() shows "
        "format conventions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operations": {
                "type": "array",
                "description": "Ordered ops; each names its target skill.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": (
                                "Skill name (lowercase, hyphens/underscores, "
                                "max 64 chars); an existing skill's name "
                                "unless creating."
                            )
                        },
                        "action": {
                            "type": "string",
                            "enum": ["create", "patch", "delete", "write_file", "remove_file"]
                        },
                        "content": {
                            "type": "string",
                            "description": (
                                "Full SKILL.md text (YAML frontmatter + "
                                "markdown body) for create, or a full "
                                "rewrite on patch."
                            )
                        },
                        "category": {
                            "type": "string",
                            "description": "Optional category subdir for create (e.g. 'devops')."
                        },
                        # patch args: same fuzzy-matching semantics as the
                        # `patch` tool — teach only skill-specific facts here.
                        "old_string": {
                            "type": "string",
                            "description": "Text to find (patch; same matching semantics as the patch tool)."
                        },
                        "new_string": {
                            "type": "string",
                            "description": "Replacement (patch); empty string deletes the match."
                        },
                        "replace_all": {
                            "type": "boolean",
                            "description": "patch: replace all occurrences (default false)."
                        },
                        "file_path": {
                            "type": "string",
                            "description": (
                                "Path RELATIVE to the skill's own directory, "
                                "e.g. 'references/api.md' — no leading slash, "
                                "never absolute. write_file/remove_file: "
                                "required; first segment references/, "
                                "templates/, scripts/, or assets/. patch: "
                                "optional (default SKILL.md)."
                            )
                        },
                        "file_content": {
                            "type": "string",
                            "description": "Content for write_file."
                        }
                    },
                    "required": ["name", "action"]
                }
            },
            # NOTE: the handler also accepts the legacy flat single-op shape
            # (top-level action/name/content/old_string/new_string/
            # replace_all/category/file_path/file_content) — old transcripts
            # and staged-write replay depend on it — plus `absorbed_into` on
            # delete ops (curator-only vocabulary; the curator's prompt
            # documents it and the delete guard's error re-teaches it).
            # None are advertised.
        },
        "required": ["operations"],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="skill_manage",
    toolset="skills",
    schema=SKILL_MANAGE_SCHEMA,
    handler=lambda args, **kw: skill_manage(
        action=args.get("action", ""),
        name=args.get("name", ""),
        content=args.get("content"),
        category=args.get("category"),
        file_path=args.get("file_path"),
        file_content=args.get("file_content"),
        old_string=args.get("old_string"),
        new_string=args.get("new_string"),
        replace_all=args.get("replace_all", False),
        absorbed_into=args.get("absorbed_into"),
        operations=args.get("operations"),
        task_id=kw.get("task_id"),
        session_id=kw.get("session_id")),
    emoji="📝",
)
