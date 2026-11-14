"""Byte-preserving three-way reconciliation of bundled skill packages.

State lives outside the active skills tree. A base is always upstream content,
never the merged user copy, so subsequent upstream changes remain mergeable.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import uuid

from utils import atomic_write_text


def state_dir(home: Path, name: str) -> Path:
    directory = home / "skill-updates" / hashlib.sha256(name.encode("utf-8")).hexdigest()
    if home.resolve() not in directory.resolve().parents:
        raise ValueError("Skill update state escapes the profile")
    return directory


def read_tree(root: Path) -> dict[str, bytes]:
    from tools.skills_sync_optional import _is_runtime_cache
    from pm.filesystem import is_junction

    if root.is_symlink() or is_junction(root):
        raise ValueError(f"Skill contains a symbolic link: {root}")
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or is_junction(path):
            raise ValueError(f"Skill contains a symbolic link: {path}")
        if path.is_file() and not _is_runtime_cache(path, root):
            result[path.relative_to(root).as_posix()] = path.read_bytes()
    return result


def safe_destination(root: Path, dest: Path) -> bool:
    """Do not write through category links, even before the package exists."""
    from pm.filesystem import is_junction

    if root.resolve() not in dest.resolve().parents:
        return False
    return not any(p.is_symlink() or is_junction(p)
                   for p in [dest, *dest.parents] if p != root and root in p.parents)


def _encode(tree: dict[str, bytes]) -> dict[str, str]:
    return {path: base64.b64encode(data).decode("ascii") for path, data in tree.items()}


def _decode(tree: dict[str, str]) -> dict[str, bytes]:
    for path in tree:
        _validate_relative(path)
    return {path: base64.b64decode(data, validate=True) for path, data in tree.items()}


def _validate_relative(path: str) -> None:
    if not path or "\\" in path or ":" in path or path.startswith("/") or any(
        part in ("", ".", "..") for part in path.split("/")
    ):
        raise ValueError(f"Unsafe skill path: {path}")


def write_tree(root: Path, tree: dict[str, bytes]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for relative, data in tree.items():
        _validate_relative(relative)
        path = root / relative
        if root.resolve() not in path.resolve().parents:
            raise ValueError(f"Unsafe skill path: {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def save_base(home: Path, name: str, source: Path, origin: str) -> None:
    if load_base(home, name, origin) is not None:
        return
    directory = state_dir(home, name)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_text(directory / "base.json", json.dumps({
        "origin": origin, "files": _encode(read_tree(source)),
    }, sort_keys=True) + "\n")


def load_base(home: Path, name: str, origin: str) -> dict[str, bytes] | None:
    path = state_dir(home, name) / "base.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return _decode(data["files"]) if data["origin"] == origin else None


def _merge_bytes(ours: bytes, base: bytes, theirs: bytes) -> bytes | None:
    if b"\0" in ours + base + theirs:
        return None
    try:
        for data in (ours, base, theirs):
            data.decode("utf-8")
        with tempfile.TemporaryDirectory(prefix="skill-merge-") as temp:
            paths = [Path(temp) / name for name in ("ours", "base", "theirs")]
            for path, data in zip(paths, (ours, base, theirs)):
                path.write_bytes(data)
            merged = subprocess.run(
                ["git", "merge-file", "-p", *map(str, paths)],
                capture_output=True, timeout=30, check=False,
            )
            return merged.stdout if merged.returncode == 0 else None
    except (OSError, UnicodeDecodeError, subprocess.TimeoutExpired):
        return None


def merge_trees(base: dict[str, bytes], ours: dict[str, bytes],
                theirs: dict[str, bytes]) -> tuple[dict[str, bytes], list[str]]:
    merged, conflicts = {}, []
    for path in sorted(base.keys() | ours.keys() | theirs.keys()):
        old, local, new = base.get(path), ours.get(path), theirs.get(path)
        if local == new or new == old:
            accepted = local
        elif local == old:
            accepted = new
        elif old is not None and local is not None and new is not None:
            accepted = _merge_bytes(local, old, new)
            if accepted is None:
                conflicts.append(path)
                accepted = local
        else:
            conflicts.append(path)
            accepted = local
        if accepted is not None:
            merged[path] = accepted
    structural = [path for path in merged if any(
        parent.as_posix() in merged for parent in Path(path).parents if parent != Path(".")
    )]
    if structural:
        return ours.copy(), sorted(set(conflicts + structural))
    return merged, conflicts


def try_merge(home: Path, name: str, origin: str, source: Path, dest: Path) -> bool:
    """Apply only a complete clean merge; missing legacy bases stay protected."""
    if (state_dir(home, name) / "pending").exists():
        return False
    base = load_base(home, name, origin)
    ours, theirs = read_tree(dest), read_tree(source)
    merged, conflicts = merge_trees(base, ours, theirs) if base is not None else (ours, ["unknown-base"])
    if conflicts:
        _save_pending(home, name, origin, dest, base, ours, theirs, merged, conflicts)
        return False
    _apply_tree(home, merged, dest)
    return True


def _apply_tree(home: Path, tree: dict[str, bytes], dest: Path) -> None:
    from tools.skills_sync import _replace_skill_dir

    with tempfile.TemporaryDirectory(prefix="skill-accepted-", dir=home) as temp:
        root = Path(temp) / "merged"
        write_tree(root, tree)
        _replace_skill_dir(root, dest)


def _save_pending(home, name, origin, dest, base, ours, theirs, merged, conflicts):
    from tools.skills_sync import _skills_dir, dir_hash

    directory = state_dir(home, name)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pending-", dir=directory) as temp:
        root = Path(temp) / "payload"
        for label, tree in (("base", base or {}), ("local", ours),
                            ("upstream", theirs), ("merged", merged)):
            write_tree(root / label, tree)
        atomic_write_text(root / "pending.json", json.dumps({
            "name": name, "destination": dest.relative_to(_skills_dir()).as_posix(),
            "origin": origin, "local": _encode(ours), "upstream": _encode(theirs),
            "upstream_hash": dir_hash(root / "upstream"), "conflicts": conflicts,
        }, sort_keys=True) + "\n")
        root.rename(directory / "pending")


def resolve_skill(name: str, accept: str) -> bool:
    """Explicitly accept a reviewed pending tree, refusing intervening local edits."""
    from tools import skills_sync as ss

    if accept not in {"merged", "local", "upstream"}:
        raise ValueError("Choose merged, local or upstream")
    home = ss._hermes_home()
    pending = state_dir(home, name) / "pending"
    if not pending.exists():
        return False
    data = json.loads((pending / "pending.json").read_text(encoding="utf-8-sig"))
    _validate_relative(data["destination"])
    dest = ss._skills_dir() / data["destination"]
    if not safe_destination(ss._skills_dir(), dest):
        raise ValueError("Skill destination is linked or escapes the profile")
    manifest = ss._read_manifest()
    local = _decode(data["local"])
    if read_tree(dest) != local or manifest.get(name, "") != data["origin"]:
        raise ValueError(
            f"Local skill or baseline changed; run `hermes skills resolve {name} --refresh` "
            "to archive this review and reconcile current edits, then review again"
        )
    tree = local if accept == "local" else read_tree(pending / accept)
    if "SKILL.md" not in tree:
        raise ValueError("Resolved skill must contain SKILL.md")
    # Keeping local acknowledges upstream without rewriting the verified active files.
    if accept != "local":
        _apply_tree(home, tree, dest)
    # The upstream evidence is immutable even if the editable artifact was changed.
    with tempfile.TemporaryDirectory(prefix="skill-base-", dir=home) as temp:
        source = Path(temp) / "upstream"
        write_tree(source, _decode(data["upstream"]))
        save_base(home, name, source, data["upstream_hash"])
    manifest[name] = data["upstream_hash"]
    ss._write_manifest(manifest)
    if ss._read_manifest() != manifest:
        raise OSError(
            "Could not persist skill manifest; active content may already be accepted and "
            "pending evidence is retained. After fixing the write failure, run "
            f"`hermes skills resolve {name} --refresh` to reconcile current content, then review again."
        )
    pending.rename(pending.with_name("accepted-" + uuid.uuid4().hex))
    return True


def refresh_pending(name: str) -> None:
    """Explicitly archive stale review evidence before recomputing from current edits."""
    from tools import skills_sync as ss

    pending = state_dir(ss._hermes_home(), name) / "pending"
    if pending.exists():
        pending.rename(pending.with_name("superseded-" + uuid.uuid4().hex))
    bundled = ss._get_bundled_dir()
    source = dict(ss._discover_bundled_skills(bundled)).get(name)
    if source is None:
        raise ValueError(f"No bundled source for {name}; archived review is preserved")
    dest = ss._compute_relative_dest(source, bundled)
    if not safe_destination(ss._skills_dir(), dest) or not dest.exists():
        raise ValueError("Local skill is missing or linked; archived review is preserved")
    state = ss._SyncState(ss._read_manifest(), quiet=True)
    ss._update_existing_skill(state, name, source, dest, ss.dir_hash(source))
    ss._write_manifest(state.manifest)
