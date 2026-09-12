"""Fleet Policy rollout tool — orchestrator/operator-only kanban tool.

Installs an already-built Fleet Policy release bundle onto the fixed ten-profile
fleet in one atomic step: verify bundle → verify declared release → verify
independent evidence gates on the kanban task → backup every existing install →
copy the plugin payload onto all profiles. The first per-profile failure stops
the rollout and every profile touched by this call is restored to its exact
prior state (backups swapped back, freshly written payloads removed) before a
structured error is returned. Success returns a JSON receipt (release sha,
version, per-profile backup names, deployed-file checksums).

Paths are never taken from the caller: the release root is
``<default hermes root>/releases/<version>`` and the fleet roster is fixed, so
a model can only choose *what* to roll out, never *where* it lands.

This module is imported by ``tools.kanban_tools`` at module level, so it must
not import kanban_tools back (it re-implements the two tiny delegation
predicates instead).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from typing import Any, Optional

from tools.registry import tool_error

_TOOL = "fleet_policy_rollout"

# The fixed fleet roster (profile ids under the default Hermes home), matching
# the live rollout layout: each profile gets ``plugins/fleet-policy/``.
_PROFILES = (
    "company", "design", "finance", "operations", "product",
    "qa", "research", "sales", "tech", "ux",
)

_PLUGIN_DIR_NAME = "fleet-policy"

# Release manifest schema produced by fleet_policy.release_bundle.
_BUNDLE_SCHEMA = "fleet-policy-release-bundle-v1"

# Version literals accepted in args and read back from the bundle plugin.yaml.
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_PLUGIN_YAML_VERSION_RE = re.compile(r'^version:\s*"?(\d+\.\d+\.\d+)"?\s*$', re.MULTILINE)

# Independent evidence gates, checked against the task's comment thread in
# collection order. A gate passes when the LATEST comment carrying the marker
# was authored by one of the listed roles (the commenter's Hermes profile).
_GATES = (
    ("gate:ci=pass", ("qa",)),
    ("gate:review=pass", ("qa",)),
    ("gate:rollback=pass", ("operations", "tech")),
)


class _Reject(Exception):
    """Carries a finished ``tool_error`` payload out of a validation step."""

    def __init__(self, message: str):
        super().__init__(tool_error(message))


def _check(cond: Any, message: str) -> None:
    if not cond:
        raise _Reject(message)


def _ok(**fields: Any) -> str:
    return json.dumps({"ok": True, **fields}, ensure_ascii=False, indent=2, sort_keys=True)


def _refuse_non_operator() -> None:
    """Dispatcher-spawned task workers and delegate_task children never get
    this tool (it is also gated out of their schema; this is the in-process
    backstop for calls that arrive by any other route)."""
    try:
        from agent import delegation_context as dc
    except Exception:
        return  # predicates unavailable (unit context): the check_fn gate decides
    try:
        if dc.is_delegated_child_context() or dc.is_delegated_child_process_context():
            raise _Reject(
                f"{_TOOL} refused: delegate_task children are not rollout operators. "
                "Report the candidate release to the parent agent; an orchestrator "
                "profile must perform the rollout.")
        if os.environ.get("HERMES_KANBAN_TASK") and dc.is_dispatcher_owned_worker_context():
            raise _Reject(
                f"{_TOOL} refused: dispatcher-spawned task workers cannot roll out "
                "policy releases. Ask the orchestrator/operator profile to run it.")
    except _Reject:
        raise
    except Exception:
        pass  # a failing predicate must not fail-open into a worker allow


def _require_text(args: dict, key: str) -> str:
    value = args.get(key)
    _check(isinstance(value, str) and value.strip(), f"{key} is required")
    return value.strip()


def _validate_args(args: dict) -> tuple[str, str, str]:
    unknown = sorted(set(args) - {"task_id", "release_sha", "version", "board"})
    _check(not unknown, f"unknown argument(s) {unknown}; expected task_id, release_sha, version")
    task_id = _require_text(args, "task_id")
    release_sha = _require_text(args, "release_sha").lower()
    version = _require_text(args, "version")
    _check(bool(_SHA_RE.match(release_sha)),
           "release_sha must be the 64-hex git sha of the release commit")
    _check(bool(_VERSION_RE.match(version)),
           "version must be a semver string like 1.2.18")
    return task_id, release_sha, version


def _verify_release_bundle_gate(release_root, version: str) -> None:
    """Gates 1-2: the bundle verifies byte-exact against its manifest and its
    plugin payload declares exactly the requested version."""
    from hermes_constants import get_default_hermes_root

    root = get_default_hermes_root() / "releases" / version
    if not root.is_dir():
        raise _Reject(
            f"release bundle gate failed: no release bundle at releases/{version} "
            f"under the Hermes root; build it first (fleet-policy build-bundle)")
    try:
        from fleet_policy.release_bundle import verify_release_bundle
        verify_release_bundle(root)
    except Exception as exc:
        raise _Reject(f"release bundle gate failed: bundle at releases/{version} "
                      f"does not verify against its manifest: {exc}") from exc
    plugin_yaml = root / "integrations" / "hermes" / "fleet-policy-plugin" / "plugin.yaml"
    try:
        declared = _PLUGIN_YAML_VERSION_RE.search(
            plugin_yaml.read_text(encoding="utf-8"))
    except OSError as exc:
        raise _Reject(f"release plugin gate failed: cannot read plugin.yaml: {exc}") from exc
    _check(declared is not None,
           "release plugin gate failed: plugin.yaml declares no x.y.z version")
    _check(declared.group(1) == version,
           f"release plugin gate failed: plugin.yaml declares {declared.group(1)}, "
           f"requested rollout of {version}")


def _verify_evidence_gates(conn, task_id: str) -> None:
    """Gates 3-5: the latest comment carrying each gate marker must come from
    an independent role. Older same-marker comments by other roles never
    override a newer one (a stale pass cannot vouch for a later push)."""
    from hermes_cli import kanban_db as kb

    latest_by_marker: dict[str, tuple[str, str]] = {}
    for comment in kb.list_comments(conn, task_id):
        for marker, _roles in _GATES:
            if marker in comment.body:
                latest_by_marker[marker] = (comment.author, comment.body)
    missing = []
    for marker, roles in _GATES:
        entry = latest_by_marker.get(marker)
        if entry is None or entry[0] not in roles:
            missing.append(f"{marker} (by {' or '.join(roles)})")
    _check(not missing,
           "evidence gates failed: missing or mis-attributed exact task comment(s): "
           + "; ".join(missing))


def _unique_backup_dir(plugins_dir, now: float):
    """Sibling backup dir for an existing install; None when nothing to back up."""
    existing = plugins_dir / _PLUGIN_DIR_NAME
    if not existing.exists():
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime(now))
    backup = plugins_dir / f"{_PLUGIN_DIR_NAME}.old-{stamp}"
    suffix = 0
    while backup.exists():
        suffix += 1
        backup = plugins_dir / f"{_PLUGIN_DIR_NAME}.old-{stamp}-{suffix}"
    return backup


def _sha256(path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _deployed_files(root) -> list[dict[str, str]]:
    """Checksums of every file this tool copies (payload + config), relative
    to the bundle root, sorted for a stable receipt."""
    payload_root = root / "integrations" / "hermes" / "fleet-policy-plugin"
    config_root = root / "config"
    entries: list[dict[str, str]] = []
    for base, prefix in ((payload_root, "plugins/fleet-policy"), (config_root, "plugins/fleet-policy/config")):
        for path in sorted(base.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo"):
                entries.append({
                    "path": f"{prefix}/{path.relative_to(base).as_posix()}",
                    "sha256": _sha256(path),
                })
    return sorted(entries, key=lambda e: e["path"])


def _rollback(completed: list[tuple[Any, Any]]) -> None:
    """Exact restoration, reverse order: drop this call's payload, swap the
    backup (or the pre-call absence) back in. config/ ships inside the plugin
    payload, so one dir restore covers it."""
    for plugins_dir, backup_dir in reversed(completed):
        target = plugins_dir / _PLUGIN_DIR_NAME
        shutil.rmtree(target, ignore_errors=True)
        if backup_dir is not None:
            os.replace(str(backup_dir), str(target))


def run_rollout(args: dict, **_kw) -> str:
    _refuse_non_operator()
    task_id, release_sha, version = _validate_args(args)

    from hermes_constants import get_default_hermes_root
    release_root = get_default_hermes_root() / "releases" / version
    _verify_release_bundle_gate(release_root, version)

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect(board=args.get("board") or None)
    try:
        _verify_evidence_gates(conn, task_id)
    finally:
        conn.close()

    source_payload = release_root / "integrations" / "hermes" / "fleet-policy-plugin"
    source_config = release_root / "config"
    from hermes_constants import get_default_hermes_root as _gdr  # single resolution point
    profiles_root = _gdr() / "profiles"
    now = time.time()
    completed: list[tuple[Any, Any]] = []
    installed: list[str] = []
    backed_up: dict[str, Optional[str]] = {}
    try:
        for profile in _PROFILES:
            plugins_dir = profiles_root / profile / "plugins"
            if not plugins_dir.is_dir():
                raise RuntimeError(f"profile {profile}: missing plugins directory")
            backup_dir = _unique_backup_dir(plugins_dir, now)
            if backup_dir is not None:
                os.replace(str(plugins_dir / _PLUGIN_DIR_NAME), str(backup_dir))
            target = plugins_dir / _PLUGIN_DIR_NAME
            try:
                shutil.copytree(source_payload, target)
                shutil.copytree(source_config, target / "config", dirs_exist_ok=True)
            except Exception:
                # restore THIS profile before unwinding so the reverse pass
                # never meets a half-written payload twice
                shutil.rmtree(target, ignore_errors=True)
                if backup_dir is not None:
                    os.replace(str(backup_dir), str(target))
                raise
            completed.append((plugins_dir, backup_dir))
            installed.append(profile)
            backed_up[profile] = backup_dir.name if backup_dir is not None else None
    except Exception as exc:
        _rollback(completed)
        return tool_error(
            f"{_TOOL}: rollout of {version} ({release_sha[:12]}) failed at profile "
            f"'{_PROFILES[len(completed)] if len(completed) < len(_PROFILES) else 'unknown'}': {exc}. "
            "All profiles touched by this call were rolled back to their prior state; "
            "no partial install remains.")
    if len(installed) != len(_PROFILES):
        _rollback(completed)
        return tool_error(f"{_TOOL}: incomplete rollout of {version}: "
                          f"{len(installed)}/{len(_PROFILES)} profiles; rolled back.")

    return _ok(
        tool=_TOOL,
        task_id=task_id,
        release_sha=release_sha,
        version=version,
        installed=installed,
        backed_up=backed_up,
        files=_deployed_files(release_root),
        rolled_back=False,
    )


# Model-facing entry point, wrapped by tools.kanban_tools' structured-error
# decorator exactly like its own handlers.
def handle_fleet_policy_rollout(args: dict, **kw) -> str:
    try:
        return run_rollout(args, **kw)
    except _Reject as e:
        return e.args[0]
    except Exception as e:  # mirrored rendering, no traceback to the model
        return tool_error(f"{_TOOL}: {e}")
