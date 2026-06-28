#!/usr/bin/env python3
"""register_artifact tool — stage a workspace file for delivery to the user.

Filesystem-only (resolve + copy): no async, no threads, and crucially NO
SSE/run-event emission. The KarinAI backend's post-run "sweep" of
``<workspace>/outputs/<product_run_id>/`` is what turns staged files into
durable, downloadable artifacts. This tool only has to place the file in that
directory under a sensible name.

Both ends of the copy are containment-checked: the SOURCE must resolve inside
the workspace, and the DESTINATION directory must too — the managed agent also
has shell access, so it could pre-plant a symlink in the outputs chain and turn
a naive copy into an arbitrary-write escape. We refuse symlinked output dirs,
re-verify the resolved dir, never overwrite an existing deliverable, and never
write through a leaf symlink.

Resolution facts (verified against the managed runtime):
  * Workspace root is ``/workspace`` (``HERMES_WRITE_SAFE_ROOT=/workspace``;
    see karinai/runtime/config.py). ``get_safe_write_root()`` returns its
    realpath.
  * The sweep dir is ``<workspace>/outputs/<PRODUCT_RUN_ID>/``. PRODUCT_RUN_ID
    is the value of the per-run HTTP header ``X-KarinAI-Run-Id`` the backend
    sends to the api_server; it is bound into per-run context as
    ``HERMES_PRODUCT_RUN_ID`` (see gateway/session_context.py and
    gateway/platforms/api_server.py:_handle_runs). It is NOT the agent's own
    internal run id (``run_<uuid>``).
  * When no product run id is bound (CLI / dev / non-managed), the tool stages
    the file under ``outputs/`` without a run-id subdir and reports that it will
    not be auto-delivered, rather than failing.
"""

import os
import shutil
from pathlib import Path

from agent.file_safety import get_safe_write_root
from gateway.session_context import get_session_env
from tools.file_tools import _resolve_path_for_task
from tools.path_security import validate_within_dir
from tools.registry import registry, tool_error, tool_result

# Fallback workspace root if HERMES_WRITE_SAFE_ROOT is unset (dev/non-managed).
_DEFAULT_WORKSPACE_ROOT = "/workspace"

# The backend's artifact sweep silently skips files over this size, so the tool
# warns up front rather than letting the agent promise an undeliverable file.
_BACKEND_MAX_FILE_BYTES = 256 * 1024 * 1024


REGISTER_ARTIFACT_SCHEMA = {
    "name": "register_artifact",
    "description": (
        "Deliver a file you produced to the user as a downloadable artifact. "
        "Copies the file at 'path' (relative to your workspace) into this run's "
        "output directory; the platform delivers it to the user as a download "
        "after the run finishes. Use this for final deliverables the user should "
        "be able to download (spreadsheets, documents, images, PDFs, archives, "
        "datasets, generated files). Do NOT just print the file path — a file the "
        "user cannot download has not been delivered. Returns immediately after "
        "staging the file."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Path to the file to deliver, relative to your workspace "
                    "(e.g. 'report.xlsx' or 'out/chart.png'). Must resolve inside "
                    "the workspace; absolute paths and '..' escapes are rejected."
                ),
            },
            "name": {
                "type": "string",
                "description": (
                    "Optional friendly filename shown to the user (e.g. "
                    "'Q3 Sales Report.xlsx'). Reduced to a safe basename. Defaults "
                    "to the source filename."
                ),
            },
            "description": {
                "type": "string",
                "description": "Optional human-readable note about the file. Informational only.",
            },
        },
        "required": ["path"],
    },
}


def _check_file_reqs():
    """Mirror the file toolset gate (terminal backend availability)."""
    from tools import check_file_requirements

    return check_file_requirements()


def _workspace_root() -> Path:
    """Absolute workspace root: realpath(HERMES_WRITE_SAFE_ROOT), else /workspace."""
    root = get_safe_write_root()
    return Path(root if root else _DEFAULT_WORKSPACE_ROOT).resolve()


def _product_run_id() -> str:
    """Backend PRODUCT run id for this run (from X-KarinAI-Run-Id), or '' if absent."""
    return (get_session_env("HERMES_PRODUCT_RUN_ID", "") or "").strip()


def _safe_basename(name: str, default: str) -> str:
    """Reduce *name* to a safe basename inside the output dir, else *default*."""
    candidate = os.path.basename((name or "").strip())
    if not candidate or candidate in {".", ".."}:
        return default
    return candidate


def _unique_dest(out_dir: Path, name: str) -> Path:
    """A non-colliding destination inside *out_dir* (``stem (2).ext`` on clash).

    Never returns an existing path or a symlink: an earlier deliverable is never
    silently overwritten (data loss) and a pre-planted leaf symlink is sidestepped
    rather than written through.
    """
    candidate = out_dir / name
    if not candidate.exists() and not candidate.is_symlink():
        return candidate
    stem, ext = os.path.splitext(name)
    for i in range(2, 1000):
        candidate = out_dir / f"{stem} ({i}){ext}"
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
    return out_dir / name


def _handle_register_artifact(args, **kw):
    task_id = kw.get("task_id") or "default"

    path = args.get("path")
    if not path or not isinstance(path, str):
        return tool_error(
            "register_artifact: missing required field 'path' (the file to deliver, "
            "relative to your workspace)."
        )

    workspace = _workspace_root()

    # Resolve with the same anchor logic the file tools use, then HARD-verify the
    # result is inside the workspace (rejects absolute paths and '..'/symlink
    # escapes via realpath containment — _resolve_path_for_task alone does not).
    src = _resolve_path_for_task(path, task_id)
    escape_err = validate_within_dir(src, workspace)
    if escape_err:
        return tool_error(
            f"register_artifact: 'path' must resolve inside your workspace. {escape_err}"
        )
    if not src.is_file():
        return tool_error(
            f"register_artifact: no file found at {path!r} (resolved to {str(src)!r})."
        )

    run_id = _product_run_id()
    outputs_root = workspace / "outputs"
    out_dir = outputs_root / run_id if run_id else outputs_root

    # Destination hardening (the agent also has shell access). Refuse a symlinked
    # output dir BEFORE creating anything — mkdir(exist_ok=True) does NOT fail on
    # a symlinked dir and would let copy2 follow it outside the workspace — then
    # re-verify the resolved dir is still contained.
    for d in (outputs_root, out_dir):
        if d.is_symlink():
            return tool_error(
                "register_artifact: refusing to deliver through a symlinked output directory."
            )
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return tool_error(f"register_artifact: failed to prepare the output directory: {exc}")
    if validate_within_dir(out_dir, workspace):
        return tool_error("register_artifact: the output directory resolves outside the workspace.")

    dest = _unique_dest(out_dir, _safe_basename(args.get("name", ""), src.name))
    if dest.is_symlink():  # belt-and-suspenders: never write through a leaf symlink
        return tool_error("register_artifact: destination name collides with a symlink.")
    dest_name = dest.name

    try:
        shutil.copy2(str(src), str(dest))
    except OSError as exc:
        return tool_error(f"register_artifact: failed to stage the file: {exc}")

    try:
        rel_dest = str(dest.relative_to(workspace))
    except ValueError:
        rel_dest = str(dest)

    # Up-front, detectable delivery caveat. Delivery itself is the backend's
    # post-run sweep, which this tool cannot observe — so it reports "staged",
    # not "delivered", and flags an oversize file the sweep would silently drop.
    oversize = False
    try:
        oversize = src.stat().st_size > _BACKEND_MAX_FILE_BYTES
    except OSError:
        pass

    if not run_id:
        # No managed run id bound (CLI/dev): the backend sweep keys on
        # outputs/<product_run_id>/, so a file in bare outputs/ is staged but
        # NOT auto-delivered. Say so plainly rather than implying delivery.
        return tool_result(
            success=True,
            staged=True,
            delivered=False,
            path=rel_dest,
            name=dest_name,
            message=(
                f"Staged {dest_name!r} to {rel_dest}, but no run id is bound "
                "(not a managed run), so it will not be auto-delivered as a "
                "download. This is expected outside the KarinAI managed runtime."
            ),
        )

    message = (
        f"Staged {dest_name!r} for delivery; the platform will deliver it to the "
        "user as a download after this run completes."
    )
    if oversize:
        message += " Warning: it exceeds the platform's ~256 MB delivery limit and may not be delivered."
    return tool_result(success=True, staged=True, path=rel_dest, name=dest_name, message=message)


registry.register(
    name="register_artifact",
    toolset="artifact",
    schema=REGISTER_ARTIFACT_SCHEMA,
    handler=_handle_register_artifact,
    check_fn=_check_file_reqs,
    emoji="📦",
)
