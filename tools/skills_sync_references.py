"""Reference ownership supplements the v2 origin hash; never replaces its proof."""
import json
import logging
from pathlib import Path
from typing import Optional, Set

from tools.skills_sync_optional import _is_runtime_cache, _ss
from utils import atomic_write_text

logger = logging.getLogger(__name__)


def read_reference_inventory(manifest_file: Path) -> dict:
    try:
        data = json.loads(manifest_file.with_name(".bundled_references.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {name: entry for name, entry in data.items()
            if isinstance(entry, dict) and isinstance(entry.get("hash"), str)
            and isinstance(entry.get("paths"), list)
            and all(isinstance(path, str) for path in entry["paths"])}


def write_reference_inventory(manifest_file: Path, inventory: dict) -> None:
    try:
        atomic_write_text(manifest_file.with_name(".bundled_references.json"),
                          json.dumps(inventory, sort_keys=True) + "\n", preserve_mode=True)
    except OSError:
        logger.warning("Could not save bundled reference ownership", exc_info=True)


def reference_paths(directory: Path) -> Set[Path]:
    return {p.relative_to(directory) for p in (directory / "references").rglob("*")
            if p.is_file() and not _is_runtime_cache(p, directory)}


def local_reference_additions(directory: Path, source: Path, origin_hash: str,
                             inventory: dict) -> Optional[Set[Path]]:
    """Local additions for an update-clean package, or None when it must be kept.

    A hash-bound inventory remembers shipped references even after upstream removes
    them. Old installations can prove an exact origin after excluding only paths
    absent from today's source; ambiguous old histories stay protected.
    """
    try:
        refs = directory / "references"
        if refs.is_symlink() or any(p.is_symlink() for p in refs.rglob("*")):
            return None  # copying through links could read or overwrite outside the skill
        if inventory.get("hash") == origin_hash:
            owned = {Path(p) for p in inventory["paths"]}
        else:
            if _ss()._matches_origin_hash(directory, origin_hash, excluded=set()):
                return set()  # pristine pre-inventory installs also accept upstream deletions
            owned = reference_paths(source)
        additions = reference_paths(directory) - owned
        for rel in additions:
            target = source / rel
            if target.exists() or target.is_symlink() or any(
                    (source / parent).is_file() or (source / parent).is_symlink() for parent in rel.parents):
                return None  # file/file or file/directory collision: never overwrite local content
        if not _ss()._matches_origin_hash(directory, origin_hash, excluded=additions):
            return None
        return additions
    except OSError:
        logger.debug("Could not inspect references in %s; keeping local package", directory, exc_info=True)
        return None
