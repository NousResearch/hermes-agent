"""Rekey profile-local checkpoint projects after a named profile directory moves.

The checkpoint store itself lives under ``HERMES_HOME/checkpoints`` and therefore moves with a
renamed profile. Project identity inside that store is different: refs, metadata and the
agent-write ledger are keyed by a hash of the absolute workdir. A profile rename changes that
absolute path for every project beneath the profile home, so those entries must follow the move.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict

from tools import checkpoint_manager as cm

logger = logging.getLogger(__name__)


def _rebase_ledger_paths(ledger: Dict, old_workdir: Path, new_workdir: Path) -> Dict:
    """Move absolute ledger keys under ``old_workdir`` to the corresponding new path."""
    rebased = {}
    for raw_path, entry in ledger.items():
        try:
            relative = Path(raw_path).relative_to(old_workdir)
        except (TypeError, ValueError):
            rebased[raw_path] = entry
        else:
            rebased[str(new_workdir / relative)] = entry
    return rebased


def _temp_json_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.profile-rename-{os.getpid()}.tmp")


def _write_json_temp(path: Path, data: Dict) -> Path:
    """Write a same-directory temporary JSON file ready for atomic ``replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temp_json_path(path)
    tmp.write_text(json.dumps(data), encoding="utf-8")
    return tmp


def migrate_profile_checkpoint_projects(old_profile_dir: Path, new_profile_dir: Path) -> Dict[str, int]:
    """Rekey live checkpoint projects whose workdirs moved with a profile rename.

    Only workdirs beneath ``old_profile_dir`` are affected. External workdirs keep the same
    absolute path even though their profile-owned checkpoint store moved, so their hash/ref stays
    valid. Target collisions fail closed rather than merging two checkpoint histories.

    The per-project git index is a rebuildable cache: after the new ref and durable metadata are
    installed, the old index is discarded and the next checkpoint seeds a fresh index from the
    migrated ref. The agent-write ledger is durable behavior state and is rebased to the moved
    absolute file paths so safe restore continues to recognize Hermes-authored writes.
    """
    old_root = cm._normalize_path(str(old_profile_dir))
    new_root = cm._normalize_path(str(new_profile_dir))
    base = new_root / "checkpoints"
    store = cm._store_path(base)
    result = {"scanned": 0, "migrated": 0, "errors": 0}
    if not cm._store_has_head(store):
        return result

    git_available = shutil.which("git") is not None
    for meta in cm._list_projects(store):
        result["scanned"] += 1
        raw_workdir = meta.get("workdir")
        old_hash = meta.get("_hash")
        if not raw_workdir or not old_hash:
            continue
        old_workdir = cm._normalize_path(str(raw_workdir))
        try:
            relative = old_workdir.relative_to(old_root)
        except ValueError:
            continue

        new_workdir = cm._normalize_path(str(new_root / relative))
        # A project that was already absent before the profile rename stays stale/orphaned; the
        # rename should not retarget its retention evidence to a path that was never moved.
        if not new_workdir.is_dir():
            continue
        if not git_available:
            result["errors"] += 1
            logger.warning(
                "Cannot migrate checkpoint project %s after profile rename: git not found",
                old_workdir,
            )
            continue

        new_hash = cm._project_hash(str(new_workdir))
        if new_hash == old_hash:  # defensive (cryptographic/path identity coincidence)
            continue

        old_ref, new_ref = cm._ref_name(old_hash), cm._ref_name(new_hash)
        old_meta = cm._project_meta_path(store, old_hash)
        new_meta = cm._project_meta_path(store, new_hash)
        old_ledger = cm._ledger_path(store, old_hash)
        new_ledger = cm._ledger_path(store, new_hash)
        old_index = cm._index_path(store, old_hash)

        old_tip = cm._ref_tip(store, str(new_workdir), old_ref)
        new_tip = cm._ref_tip(store, str(new_workdir), new_ref)
        if new_tip or new_meta.exists() or new_ledger.exists():
            result["errors"] += 1
            logger.warning(
                "Cannot migrate checkpoint project %s -> %s after profile rename: "
                "target identity %s already exists",
                old_workdir,
                new_workdir,
                new_hash,
            )
            continue

        serialized_meta = {key: value for key, value in meta.items() if key != "_hash"}
        serialized_meta["workdir"] = str(new_workdir)
        evidence = cm._volume_evidence(new_workdir)
        if evidence:
            serialized_meta.update(evidence)
        else:
            serialized_meta.pop("workdir_parent_dev", None)
            serialized_meta.pop("workdir_parent_ino", None)

        ledger = cm._read_json_dict(old_ledger) if old_ledger.exists() else None
        rebased_ledger = (
            _rebase_ledger_paths(ledger, old_workdir, new_workdir)
            if ledger is not None
            else None
        )

        meta_tmp = ledger_tmp = None
        new_ref_created = False
        installed_meta = installed_ledger = False
        try:
            meta_tmp = _write_json_temp(new_meta, serialized_meta)
            if rebased_ledger is not None:
                ledger_tmp = _write_json_temp(new_ledger, rebased_ledger)

            if old_tip:
                ok, _, err = cm._run_git(["update-ref", new_ref, old_tip], store, str(new_workdir))
                if not ok:
                    raise OSError(f"could not create target checkpoint ref: {err}")
                new_ref_created = True

            meta_tmp.replace(new_meta)
            meta_tmp = None
            installed_meta = True
            if ledger_tmp is not None:
                ledger_tmp.replace(new_ledger)
                ledger_tmp = None
                installed_ledger = True
        except Exception as exc:
            result["errors"] += 1
            logger.warning(
                "Cannot migrate checkpoint project %s -> %s after profile rename: %s",
                old_workdir,
                new_workdir,
                exc,
            )
            if meta_tmp is not None:
                cm._unlink_quiet(meta_tmp)
            if ledger_tmp is not None:
                cm._unlink_quiet(ledger_tmp)
            if installed_meta:
                cm._unlink_quiet(new_meta)
            if installed_ledger:
                cm._unlink_quiet(new_ledger)
            if new_ref_created:
                cm._delete_ref(store, new_ref)
            continue

        # Delete the source ref only after the complete target identity exists. If that destructive
        # step fails, roll the target back and leave the source metadata/ledger/index untouched so
        # the standalone migrate-identity command can retry without merging two histories.
        if old_tip and not cm._delete_ref(store, old_ref):
            result["errors"] += 1
            logger.warning("Could not delete source checkpoint ref %s after creating %s", old_ref, new_ref)
            cm._unlink_quiet(new_meta)
            cm._unlink_quiet(new_ledger)
            if new_ref_created:
                cm._delete_ref(store, new_ref)
            continue

        cm._unlink_quiet(old_meta)
        cm._unlink_quiet(old_ledger)
        cm._unlink_quiet(old_index)
        result["migrated"] += 1

    return result
