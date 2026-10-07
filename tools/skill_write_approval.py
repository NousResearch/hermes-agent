"""Creation-only write policy, using the native scanner on a read-only manifest overlay."""

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
from pathlib import Path
import time



WRITE_WAIT_SECONDS = 30.0
_write_deadline: ContextVar[float | None] = ContextVar("skill_creation_write_deadline", default=None)


def _manifest_path(path: Path) -> Path:
    # unlink removes the lexical leaf; atomic_write_text follows and preserves it.
    return path.parent.resolve() / path.name


def requires_creation_approval(operations: list[dict]) -> bool:
    """Require review if any step introduces a discoverable lexical skill root."""
    from tools.skill_proposed_mutations import plan_mutations
    if any(op.get("action") == "create" for op in operations):
        return True
    return plan_mutations(operations).requires_approval


@contextmanager
def creation_transaction(args=None):
    """Serialize the whole opt-in write, including gate evaluation and approval replay."""
    from tools import write_approval as wa
    if not wa.write_approval_enabled(wa.SKILLS) or wa.skill_write_approval_mode() != "create":
        yield
        return
    from tools.skill_manager_tool import _skills_dir
    from tools.skill_usage import skill_file_lock
    # Stable across aliases, batches and processes; outside directories that a delete can remove.
    path = _skills_dir().resolve() / ".locks" / "creation-approval.lock"
    deadline = _write_deadline.get()
    if deadline is None:
        deadline = time.monotonic() + WRITE_WAIT_SECONDS
    with skill_file_lock(path, timeout=max(0, deadline - time.monotonic())):
        token = _write_deadline.set(deadline)
        try:
            with _physical_transaction(args, deadline):
                yield
        finally:
            _write_deadline.reset(token)


@contextmanager
def _physical_transaction(args, deadline):
    from tools.skill_proposed_mutations import plan_mutations
    from tools.skill_resource_fences import acquire_resources, covered, resources_held
    if args is None:  # Review owns the pending ID; replay owns its actual physical targets.
        yield
        return
    raw = args["operations"]
    operations = raw if isinstance(raw, list) else ([] if raw is not None else [args])
    operations = [{**op, "name": op.get("name") or args["name"]} for op in operations if isinstance(op, dict)]
    if resources_held():
        if not covered(plan_mutations(operations, scan_creation=False).resources):
            raise OSError("Nested skill write is outside its batch's protected physical resources.")
        yield
        return
    while True:
        plan = plan_mutations(operations, scan_creation=False)
        with acquire_resources(plan.resources, deadline):
            confirmed = plan_mutations(operations, scan_creation=False)
            if confirmed.resources <= plan.resources or covered(confirmed.resources):
                yield
                return
        if time.monotonic() >= deadline:
            raise TimeoutError("The physical skill targets kept changing while acquiring ownership.")


def remaining_write_wait():
    """Share a finite wait budget with existing per-skill locks, not just the new fence."""
    deadline = _write_deadline.get()
    return None if deadline is None else max(0, deadline - time.monotonic())


def stage_skill_write(payload, gist, message):
    """A staging acknowledgement must name a durably saved record, never a phantom ID."""
    from tools import write_approval as wa
    from tools.registry import tool_error
    record = wa.stage_write(wa.SKILLS, payload, summary=gist, origin=wa.current_origin())
    if wa.get_pending(wa.SKILLS, record["id"]) != record:
        return tool_error("Could not persist the skill approval request. Nothing was applied; retry the write.",
                          success=False, error_type="pending_write_failed", retryable=True)
    return json.dumps({"success": True, "staged": True, "pending_id": record["id"],
                       "gist": gist, "message": message}, ensure_ascii=False)


def preserve_contended_write(args, *, reason="Another writer exceeded the wait budget."):
    """Do not lose an unstarted write on timeout; approval replay keeps its original request."""
    from tools import skill_manager_tool as smt, write_approval as wa
    from tools.registry import tool_error
    if smt._skill_gate_bypass.get():
        return tool_error(reason + " The approved request remains pending; retry /skills approve.",
                          success=False, error_type="write_busy", retryable=True)
    if (error := creation_batch_preflight(args)) is not None:
        return error
    if args["operations"] is not None:
        payload = {"action": "batch", "name": args["name"], "operations": args["operations"]}
        gist = "Concurrent skill batch waiting for review"
    else:
        payload = {k: v for k, v in args.items() if v is not None and k not in ("task_id", "session_id")}
        gist = wa.skill_gist(args["action"], args["name"], content=args["content"] or "",
                             file_path=args["file_path"] or "")
    return stage_skill_write(payload, gist,
                             reason + " Nothing was applied; the request is "
                             "preserved for review with /skills pending, diff, approve or reject.")


def creation_batch_preflight(args):
    """Reject known invalid batches before waiting or staging; execution rechecks under ownership."""
    from tools.skill_target_resolution import AmbiguousSkillTarget, unique_targets_enabled
    from tools.registry import tool_error
    raw = args["operations"]
    if raw is None:
        return None
    try:
        if not unique_targets_enabled():
            return None
        from tools.skill_manager_batch import _validate_batch_ops, _BATCH_MAX_OPS
        if not isinstance(raw, list) or not raw:
            return tool_error("operations must be a non-empty array.", success=False)
        if len(raw) > _BATCH_MAX_OPS:
            return tool_error(f"operations is capped at {_BATCH_MAX_OPS} ops per call.", success=False)
        if len(raw) == 1 and isinstance(raw[0], dict) and raw[0].get("action") == "delete":
            return None  # The sole-delete path has no sibling it could clobber.
        return _validate_batch_ops(raw, args["name"] or None, tool_error)[1]
    except AmbiguousSkillTarget as exc:
        return tool_error(str(exc), success=False, error_type="ambiguous_skill_target", candidates=exc.candidates)
    except ValueError as exc:
        return tool_error(str(exc), success=False, error_type="invalid_config")
    except OSError as exc:
        # Unowned filesystem reads can race; never acknowledge an unchecked request as staged.
        return tool_error(str(exc), success=False, error_type="write_error", retryable=True)


def _revision(path: Path):
    try:
        if path.is_dir():
            stat = path.stat()
            return ("directory", stat.st_dev, stat.st_ino, stat.st_mtime_ns)
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").digest()
    except FileNotFoundError:
        return None


def replacement_revisions(args):
    """Protect queued whole-file replacements; targeted patches already match current content."""
    from tools import skill_manager_tool as smt, write_approval as wa
    if not wa.write_approval_enabled(wa.SKILLS) or wa.skill_write_approval_mode() != "create":
        return {}
    from tools.skill_proposed_mutations import plan_mutations
    raw = args["operations"]
    operations = raw if isinstance(raw, list) else ([] if raw is not None else [args])
    operations = [{**op, "name": op.get("name") or args["name"]} for op in operations if isinstance(op, dict)]
    plan = plan_mutations(operations, scan_creation=False)
    return {path: _revision(path) for path in plan.replacement_targets}


def replacements_changed(revisions):
    return any(_revision(path) != expected for path, expected in revisions.items())
