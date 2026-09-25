"""Checkpoint store retention, orphan pruning, status and cleanup."""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

from hermes_cli.gitlock import clear_stale_tmp_packs
from utils import rmtree_readonly
from tools.checkpoint_manager import (
    _GIT_TIMEOUT, _LEGACY_PREFIX, _PRUNE_MARKER_NAME, _REFS_PREFIX, _STORE_DIRNAME,
    _dir_size_bytes, _index_path, _list_projects, _pre_v2_shadow_repos,
    _project_meta_path, _ref_name, _resolve_checkpoint_base, _run_git, _store_path,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Auto-maintenance
# ---------------------------------------------------------------------------
#
# v2 rewrite.  The sweep now operates on per-project refs inside the shared
# store rather than per-project shadow repos.  Legacy-archive dirs
# (``legacy-<ts>/``) are swept with the same retention policy.

def _delete_ref(store: Path, ref: str) -> bool:
    """Delete a ref from the store.  Returns True on success."""
    ok, _, _ = _run_git(
        ["update-ref", "-d", ref], store, str(store.parent),
        allowed_returncodes={128},
    )
    return ok


def _workdir_is_observably_gone(
    workdir: str,
    parent_dev: Optional[int] = None,
    parent_ino: Optional[int] = None,
    require_parent_identity: bool = True,
) -> bool:
    """True only when we can positively observe that ``workdir`` was removed.

    ``Path.exists()`` returns False for a deleted directory AND for one whose
    storage simply is not attached right now — an unplugged external drive, a
    network share behind a downed VPN, a bind-mount absent from this
    container, an offline Windows mapped drive. Orphan pruning deletes the
    project's entire checkpoint history, so treating that ambiguity as
    "deleted" throws away the user's restore points over a transient mount
    state, unattended, at startup.

    Require corroboration, in three steps.

    First, the parent directory must be present, so the absence of the project
    inside it is something we actually observed. When the parent is missing
    too, the volume is not there and we know nothing.

    Second, the present parent must be the directory we knew — not merely a
    directory at the same path. Unmounting swaps the directory visible at a
    mount point: while the volume is attached the path resolves to the
    mounted filesystem's root; after detach it resolves to the *underlying*
    (underlay) directory, which may carry entries of its own (a ``.keep``
    placeholder, sibling mount points). Those entries were never next to the
    project and prove nothing about the volume being attached. So the
    parent's ``(st_dev, st_ino)`` must match the identity recorded in the
    project's metadata while the project was observably live
    (``parent_dev``/``parent_ino``). A mismatch means a different directory
    is visible at that path — a detached volume, not an observed deletion.
    When no identity was ever recorded (metadata written by an older
    version) and ``require_parent_identity`` is True, stay conservative and
    do not classify as orphan. Callers that have no identity channel at all
    (the frozen pre-v2 layout) pass ``require_parent_identity=False`` to
    keep the structural checks only.

    Third, the (identity-confirmed) parent must actually carry information.
    Unmounting leaves classic static mount points (``/mnt/volume/proj``, an
    fstab entry, a container bind-mount) behind as *empty* directories, so an
    empty parent is the signature of a detached volume just as much as of a
    deleted project. Prune only when the parent holds something else (we
    observed a populated directory that does not contain the project) or is
    itself a live mount point (the volume is demonstrably attached and the
    project is demonstrably not on it).

    Genuinely abandoned projects are still reclaimed by the retention/stale
    rule, which runs off ``last_touch`` rather than a filesystem probe.
    """
    if not workdir:
        return False
    path = Path(workdir)
    try:
        if path.exists():
            return False
        parent = path.parent
        # A path whose parent is itself (a filesystem root) gives us nothing
        # to corroborate against.
        if parent == path:
            return False
        if not parent.is_dir():
            return False
        if parent_dev is not None and parent_ino is not None:
            st = parent.stat()
            if (st.st_dev, st.st_ino) != (parent_dev, parent_ino):
                # A different directory is visible at the parent's path than
                # the one the project lived in — the volume is detached (its
                # underlay showing through) or was swapped. Not a deletion.
                return False
        elif require_parent_identity:
            # No recorded identity to check against — we cannot tell the
            # project's real parent from an underlay directory exposed by an
            # unmount. Unsure never deletes; retention still reclaims.
            return False
        if _dir_has_any_entry(parent):
            return True
        # Empty parent: only evidence if that directory is a mount point, i.e.
        # the volume is attached right now and simply does not hold the
        # project. An empty plain directory is an unmounted mount point as
        # readily as an emptied project root.
        return os.path.ismount(parent)
    except OSError:
        # Probe failed (permission, I/O error) — not evidence of deletion.
        return False


def _dir_has_any_entry(directory: Path) -> bool:
    """True when ``directory`` contains at least one entry.

    Stops after the first entry rather than materializing the listing; a
    project root can hold a large tree.
    """
    with os.scandir(directory) as entries:
        for _ in entries:
            return True
    return False


def prune_checkpoints(
    retention_days: int = 7,
    delete_orphans: bool = True,
    checkpoint_base: Optional[Path] = None,
    max_total_size_mb: int = 0,
    orphan_allowlist: Optional[set] = None,
) -> Dict[str, int]:
    """Delete stale/orphan checkpoints and reclaim store space.

    A project entry is deleted when either:

    * ``delete_orphans=True`` and its ``workdir`` no longer exists on disk
      (the original project was deleted / moved); OR
    * its ``last_touch`` is older than ``retention_days`` days.

    ``orphan_allowlist``, when not ``None``, restricts orphan deletion to
    the given identities (v2 project ``_hash`` strings and/or pre-v2 shadow
    repo paths as ``str``). This lets a caller that showed the user a
    confirmation preview (built from ``store_status()``) bind the resulting
    deletion to exactly what was displayed — a project that only becomes
    orphaned *after* the preview (e.g. its workdir vanishes while the human
    is answering the prompt) is skipped rather than swept up under the
    earlier confirmation. Pass ``None`` (the default) to delete every
    currently-orphaned project, e.g. for ``--force`` or unattended callers
    that never show a preview.

    Additionally, if ``max_total_size_mb > 0`` and the store exceeds that
    after orphan/stale pruning, the oldest commit per remaining project is
    dropped until the store is under the cap.

    Legacy-archive dirs (``legacy-*``) older than ``retention_days`` are
    also deleted.

    Returns a dict with counts ``{"scanned", "deleted_orphan",
    "deleted_stale", "errors", "bytes_freed"}``.

    Never raises — maintenance must never block interactive startup.
    """
    base = checkpoint_base or _resolve_checkpoint_base()
    result = {
        "scanned": 0,
        "deleted_orphan": 0,
        "deleted_stale": 0,
        "errors": 0,
        "bytes_freed": 0,
    }
    if not base.exists():
        return result

    from tools.checkpoint_pruning import PruneError, store_lock

    try:
        with store_lock(base):
            return _prune_checkpoints(base, result, retention_days, delete_orphans, max_total_size_mb, orphan_allowlist)
    except (PruneError, OSError) as exc:
        result["errors"] += 1
        logger.warning("Checkpoint maintenance stopped: %s", exc)
        return result


def _prune_checkpoints(
    base: Path, result: Dict[str, int], retention_days: int, delete_orphans: bool,
    max_total_size_mb: int, orphan_allowlist: Optional[set],
) -> Dict[str, int]:
    size_before = _dir_size_bytes(base)

    # --- Legacy pre-v2 per-project shadow repos (kept directly under base) ---
    # Pre-v2 layout: ``base/<hash>/HEAD`` etc.  We treat these exactly as the
    # v1 pruner did so behaviour is unchanged for anyone still on that layout
    # or sitting on a mid-migration system.
    cutoff = 0.0
    if retention_days > 0:
        cutoff = time.time() - retention_days * 86400

    for child in base.iterdir():
        if not child.is_dir():
            continue
        if child.name == _STORE_DIRNAME:
            continue
        if child.name.startswith(_LEGACY_PREFIX):
            # Legacy archive: prune by dir mtime using same retention rule.
            if retention_days <= 0:
                continue
            try:
                m = child.stat().st_mtime
            except OSError:
                continue
            if m >= cutoff:
                continue
            try:
                size = _dir_size_bytes(child)
                rmtree_readonly(child)
                result["bytes_freed"] += size
                result["deleted_stale"] += 1
            except OSError as exc:
                result["errors"] += 1
                logger.warning("Failed to delete legacy archive %s: %s", child, exc)

    # Pre-v2 per-project shadow repos.  Scanned via the same helper
    # `store_status()` uses for its orphan preview, so a confirmation prompt
    # built from that preview always matches what gets deleted here.
    for repo in _pre_v2_shadow_repos(base):
        child = repo["path"]
        result["scanned"] += 1
        reason: Optional[str] = None
        if (
            delete_orphans
            and not repo["marker_unreadable"]
            and (
                repo["workdir"] is None
                # The frozen pre-v2 layout has no metadata channel to carry a
                # recorded parent identity, so only the structural checks
                # (parent present + populated / live mount point) apply here.
                or _workdir_is_observably_gone(
                    repo["workdir"], require_parent_identity=False,
                )
            )
            and (orphan_allowlist is None or str(child) in orphan_allowlist)
        ):
            reason = "orphan"
        if reason is None and retention_days > 0:
            newest = 0.0
            try:
                for p in child.rglob("*"):
                    try:
                        mt = p.stat().st_mtime
                        newest = max(newest, mt)
                    except OSError:
                        continue
            except OSError:
                pass
            if newest > 0 and newest < cutoff:
                reason = "stale"
        if reason is None:
            continue
        try:
            size = _dir_size_bytes(child)
            rmtree_readonly(child)
            result["bytes_freed"] += size
            if reason == "orphan":
                result["deleted_orphan"] += 1
            else:
                result["deleted_stale"] += 1
        except OSError as exc:
            result["errors"] += 1
            logger.warning("Failed to prune checkpoint repo %s: %s", child.name, exc)

    # --- v2 shared store: per-project ref pruning via metadata ---
    store = _store_path(base)
    if (store / "HEAD").exists():
        # A gc killed by the store timeout strands tmp_pack_* files that gc.auto=0 means git
        # itself never reclaims; sweep them even when no ref moved (a sweep is a directory
        # listing, unlike the pack-rewriting gc gated on refs below).
        clear_stale_tmp_packs(store)
        for meta in _list_projects(store):
            dir_hash = meta.get("_hash") or ""
            workdir = meta.get("workdir") or ""
            if not dir_hash:
                continue
            result["scanned"] += 1
            reason = None
            parent_dev = meta.get("workdir_parent_dev")
            parent_ino = meta.get("workdir_parent_ino")
            if not isinstance(parent_dev, int) or isinstance(parent_dev, bool):
                parent_dev = None
            if not isinstance(parent_ino, int) or isinstance(parent_ino, bool):
                parent_ino = None
            if (
                delete_orphans
                and (
                    not workdir
                    or _workdir_is_observably_gone(
                        workdir,
                        parent_dev=parent_dev,
                        parent_ino=parent_ino,
                    )
                )
                and (orphan_allowlist is None or dir_hash in orphan_allowlist)
            ):
                reason = "orphan"
            elif retention_days > 0:
                last_touch = float(meta.get("last_touch", 0) or 0)
                if last_touch > 0 and last_touch < cutoff:
                    reason = "stale"
            if reason is None:
                continue
            ref = _ref_name(dir_hash)
            if not _delete_ref(store, ref):
                result["errors"] += 1
                return result
            # Drop per-project index and metadata.
            try:
                idx = _index_path(store, dir_hash)
                if idx.exists():
                    idx.unlink()
            except OSError:
                pass
            try:
                mp = _project_meta_path(store, dir_hash)
                if mp.exists():
                    mp.unlink()
            except OSError:
                pass
            if reason == "orphan":
                result["deleted_orphan"] += 1
            else:
                result["deleted_stale"] += 1

        from tools.checkpoint_pruning import Pruner, PruneError

        pruner = Pruner(_run_git, store, str(base), _GIT_TIMEOUT, _dir_size_bytes, _REFS_PREFIX)
        try:
            if result["deleted_orphan"] + result["deleted_stale"] or pruner.gc_pending():
                pruner.reclaim()
            if not pruner.enforce_size(max_total_size_mb * 1024 * 1024):
                result["errors"] += 1
                logger.warning("Checkpoint store remains over its size cap; minimum history retained")
        except (PruneError, OSError) as exc:
            result["errors"] += 1
            logger.warning("Checkpoint pruning stopped: %s", exc)

    size_after = _dir_size_bytes(base)
    delta = size_before - size_after
    result["bytes_freed"] = max(result["bytes_freed"], delta)

    return result


def maybe_auto_prune_checkpoints(
    retention_days: int = 7,
    min_interval_hours: int = 24,
    delete_orphans: bool = True,
    checkpoint_base: Optional[Path] = None,
    max_total_size_mb: int = 0,
) -> Dict[str, object]:
    """Idempotent wrapper around ``prune_checkpoints`` for startup hooks.

    Writes ``CHECKPOINT_BASE/.last_prune`` on completion so subsequent
    calls within ``min_interval_hours`` short-circuit.

    Returns ``{"skipped": bool, "result": prune_checkpoints-dict,
    "error": optional str}``.
    """
    base = checkpoint_base or _resolve_checkpoint_base()
    out: Dict[str, object] = {"skipped": False}

    try:
        if not base.exists():
            out["result"] = {
                "scanned": 0, "deleted_orphan": 0, "deleted_stale": 0,
                "errors": 0, "bytes_freed": 0,
            }
            return out

        marker = base / _PRUNE_MARKER_NAME
        now = time.time()
        try:
            if marker.exists() and now - float(marker.read_text(encoding="utf-8-sig").strip()) < min_interval_hours * 3600:
                out["skipped"] = True
                return out
        except (OSError, ValueError):
            pass  # corrupt marker — treat as no prior run
        # Claim the interval before pruning: callers run on a periodic tick, and a prune that
        # dies mid-way must cost one skipped day, not a git gc every tick.
        try:
            marker.write_text(str(now), encoding="utf-8")
        except OSError as exc:
            logger.debug("Could not write checkpoint prune marker: %s", exc)
        result = out["result"] = prune_checkpoints(retention_days=retention_days, delete_orphans=delete_orphans,
                                                   checkpoint_base=base, max_total_size_mb=max_total_size_mb)

        total = result["deleted_orphan"] + result["deleted_stale"]
        if total > 0:
            logger.info(
                "checkpoint auto-maintenance: pruned %d entry(ies) "
                "(%d orphan, %d stale), reclaimed %.1f MB",
                total,
                result["deleted_orphan"],
                result["deleted_stale"],
                result["bytes_freed"] / (1024 * 1024),
            )
    except Exception as exc:
        logger.warning("checkpoint auto-maintenance failed: %s", exc)
        out["error"] = str(exc)

    return out


def auto_prune_from_config() -> Dict[str, object]:
    """``maybe_auto_prune_checkpoints`` driven by the ``checkpoints:`` config section — the one
    startup/housekeeping entry point for the CLI and the gateway. ``delete_orphans`` is never
    honoured unattended: a missing workdir is ambiguous (deleted vs. unmounted share); orphan
    cleanup is only via explicit ``hermes checkpoints prune``. Never raises."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config().get("checkpoints") or {}
        if not cfg.get("auto_prune", False):
            return {"skipped": True}
        return maybe_auto_prune_checkpoints(
            retention_days=int(cfg.get("retention_days", 7)),
            min_interval_hours=int(cfg.get("min_interval_hours", 24)),
            delete_orphans=False,
            max_total_size_mb=int(cfg.get("max_total_size_mb", 500)))
    except Exception as exc:
        logger.debug("checkpoint auto-maintenance skipped: %s", exc)
        return {"skipped": True, "error": str(exc)}


def checkpoint_footprint_notice() -> Optional[str]:
    """One-line notice when ``/rollback`` checkpoints are on and their store sits at or above
    ``checkpoints.max_total_size_mb``, else None. Checkpoints were on by default for a while
    (Mar–May 2026) and that ``enabled: true`` persisted into user configs; many users carry a
    GB-scale store for a feature they never invoke. The cap is a floor of one snapshot per
    project, so a big store is expected, not broken — the notice names the opt-out. Never raises."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config().get("checkpoints") or {}
        if not cfg.get("enabled", False):
            return None
        cap_mb = int(cfg.get("max_total_size_mb", 500) or 0)
        status = store_status()
        size = int(status["total_size_bytes"])
        if cap_mb <= 0 or size < cap_mb * 1024 * 1024:
            return None
        from hermes_cli.sizefmt import format_bytes
        return (f"Filesystem checkpoints (/rollback) are on: {format_bytes(size)} across "
                f"{status['project_count']} project(s), above the {cap_mb} MB cap (one snapshot per project is "
                f"always kept). Not using /rollback? `hermes config set checkpoints.enabled false` then "
                f"`hermes checkpoints clear`; or lower `checkpoints.retention_days`.")
    except Exception as exc:
        logger.debug("checkpoint footprint notice skipped: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public helpers for `hermes checkpoints` CLI
# ---------------------------------------------------------------------------

def store_status(checkpoint_base: Optional[Path] = None) -> Dict:
    """Return a summary of the shadow store.

    ``{"base": path, "store_size_bytes": N, "legacy_size_bytes": N,
       "total_size_bytes": N, "project_count": N, "projects": [...],
       "pre_v2_projects": [...], "legacy_archives": [...]}``

    ``pre_v2_projects`` covers shadow repos still on the pre-v2 per-project
    layout (``base/<hash>/HEAD``) — distinct from ``legacy_archives``, which
    are already-migrated ``legacy-<ts>/`` dirs. Callers that preview an
    orphan-deletion sweep must include both ``projects`` and
    ``pre_v2_projects``, since ``prune_checkpoints`` deletes orphans from
    both layouts.
    """
    base = checkpoint_base or _resolve_checkpoint_base()
    out: Dict = {
        "base": str(base),
        "store_size_bytes": 0,
        "legacy_size_bytes": 0,
        "total_size_bytes": 0,
        "project_count": 0,
        "projects": [],
        "pre_v2_projects": [],
        "legacy_archives": [],
    }
    if not base.exists():
        return out

    store = _store_path(base)
    if store.exists():
        out["store_size_bytes"] = _dir_size_bytes(store)
        if (store / "HEAD").exists():
            for meta in _list_projects(store):
                dir_hash = meta.get("_hash") or ""
                workdir = meta.get("workdir") or ""
                ref = _ref_name(dir_hash)
                ok, count_out, _ = _run_git(
                    ["rev-list", "--count", ref], store, str(base),
                    allowed_returncodes={128},
                )
                try:
                    commits = int(count_out) if ok else 0
                except ValueError:
                    commits = 0
                out["projects"].append({
                    "hash": dir_hash,
                    "workdir": workdir,
                    "exists": bool(workdir) and Path(workdir).exists(),
                    "created_at": meta.get("created_at"),
                    "last_touch": meta.get("last_touch"),
                    "commits": commits,
                })
    out["project_count"] = len(out["projects"])

    out["pre_v2_projects"] = [
        {
            "path": str(r["path"]),
            "workdir": r["workdir"],
            "exists": r["exists"],
        }
        for r in _pre_v2_shadow_repos(base)
    ]

    for child in base.iterdir():
        if child.is_dir() and child.name.startswith(_LEGACY_PREFIX):
            try:
                size = _dir_size_bytes(child)
            except OSError:
                size = 0
            out["legacy_size_bytes"] += size
            try:
                mt = child.stat().st_mtime
            except OSError:
                mt = 0
            out["legacy_archives"].append({
                "name": child.name,
                "size_bytes": size,
                "mtime": mt,
            })

    out["total_size_bytes"] = _dir_size_bytes(base)
    return out


def _rmtree_force(path: Path) -> None:
    rmtree_readonly(path)


def clear_all(checkpoint_base: Optional[Path] = None) -> Dict[str, int]:
    """Nuke the entire checkpoint base (store + legacy).  Irreversible.

    Returns ``{"bytes_freed": N, "deleted": bool}``.
    """
    base = checkpoint_base or _resolve_checkpoint_base()
    out = {"bytes_freed": 0, "deleted": False}
    if not base.exists():
        return out
    size = _dir_size_bytes(base)
    try:
        from tools.checkpoint_pruning import store_lock

        with store_lock(base):
            _rmtree_force(base)
        out["bytes_freed"] = size
        out["deleted"] = True
    except (OSError, RuntimeError) as exc:
        logger.warning("Could not clear checkpoint base %s: %s", base, exc)
    return out


def clear_legacy(checkpoint_base: Optional[Path] = None) -> Dict[str, int]:
    """Delete all ``legacy-*`` archive directories and report any failures."""
    base = checkpoint_base or _resolve_checkpoint_base()
    out = {"bytes_freed": 0, "deleted": 0, "errors": 0}
    if not base.exists():
        return out
    from tools.checkpoint_pruning import store_lock

    try:
        with store_lock(base):
            for child in list(base.iterdir()):
                if not child.is_dir() or not child.name.startswith(_LEGACY_PREFIX):
                    continue
                try:
                    size = _dir_size_bytes(child)
                    _rmtree_force(child)
                    out["bytes_freed"] += size
                    out["deleted"] += 1
                except OSError as exc:
                    out["errors"] += 1
                    logger.warning("Could not delete legacy archive %s: %s", child, exc)
    except (OSError, RuntimeError) as exc:
        out["errors"] += 1
        logger.warning("Could not clear legacy archives: %s", exc)
    return out
