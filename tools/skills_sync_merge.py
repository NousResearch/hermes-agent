"""Byte-preserving three-way reconciliation of bundled skill packages.

State lives outside the active skills tree. A base is always upstream content,
never the merged user copy, so subsequent upstream changes remain mergeable.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import tempfile
import uuid

from utils import atomic_write_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SkillFile:
    content: bytes
    executable: int = 0


def state_dir(home: Path, name: str) -> Path:
    directory = home / "skill-updates" / hashlib.sha256(name.encode("utf-8")).hexdigest()
    if home.resolve() not in directory.resolve().parents:
        raise ValueError("Skill update state escapes the profile")
    return directory


def read_tree(root: Path) -> dict[str, SkillFile]:
    from tools.skills_sync_optional import _is_runtime_cache
    from pm.filesystem import is_junction

    if root.is_symlink() or is_junction(root):
        raise ValueError(f"Skill contains a symbolic link: {root}")
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or is_junction(path):
            raise ValueError(f"Skill contains a symbolic link: {path}")
        if path.is_file() and not _is_runtime_cache(path, root):
            executable = path.stat().st_mode & 0o111 if os.name != "nt" else 0
            result[path.relative_to(root).as_posix()] = SkillFile(path.read_bytes(), executable)
    return result


def safe_destination(root: Path, dest: Path) -> bool:
    """Do not write through category links, even before the package exists."""
    from pm.filesystem import is_junction

    if root.resolve() not in dest.resolve().parents:
        return False
    return not any(p.is_symlink() or is_junction(p)
                   for p in [dest, *dest.parents] if p != root and root in p.parents)


def _encode(tree: dict[str, SkillFile]) -> dict[str, dict]:
    return {path: {"content": base64.b64encode(file.content).decode("ascii"),
                   "executable": file.executable} for path, file in tree.items()}


def _decode(tree: dict) -> dict[str, SkillFile]:
    result = {}
    for path, file in tree.items():
        _validate_relative(path)
        executable = file["executable"]
        if type(executable) is not int or executable & ~0o111:
            raise ValueError("Invalid skill executable bits")
        result[path] = SkillFile(base64.b64decode(file["content"], validate=True), executable)
    return result


def _validate_relative(path: str) -> None:
    if not path or "\\" in path or ":" in path or path.startswith("/") or any(
        part in ("", ".", "..") for part in path.split("/")
    ):
        raise ValueError(f"Unsafe skill path: {path}")


def write_tree(root: Path, tree: dict[str, SkillFile]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for relative, file in tree.items():
        _validate_relative(relative)
        path = root / relative
        if root.resolve() not in path.resolve().parents:
            raise ValueError(f"Unsafe skill path: {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(file.content)
        if os.name != "nt" and file.executable:
            path.chmod((path.stat().st_mode & 0o666) | file.executable)


def save_base(home: Path, name: str, source: Path, origin: str) -> None:
    directory = state_dir(home, name)
    path = directory / "base.json"
    stored = _read_base(path, name)
    if stored is not None and stored[0] == origin:
        return
    directory.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({
        "origin": origin, "files": _encode(read_tree(source)),
    }, sort_keys=True) + "\n"
    if stored is None and path.exists():
        path.rename(path.with_name("invalid-base-" + uuid.uuid4().hex + ".json"))
    atomic_write_text(path, payload)


def _read_base(path: Path, name: str) -> tuple[str, dict[str, SkillFile]] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data["origin"], str) or not isinstance(data["files"], dict):
            raise ValueError("Invalid skill baseline schema")
        files = _decode(data["files"])
        if "SKILL.md" not in files:
            raise ValueError("Skill baseline is missing SKILL.md")
        return data["origin"], files
    except FileNotFoundError:
        return None
    except (OSError, KeyError, TypeError, ValueError) as exc:
        logger.warning("Cannot read bundled skill baseline for %s; treating it as unknown: %s", name, exc)
        return None


def load_base(home: Path, name: str, origin: str) -> dict[str, SkillFile] | None:
    stored = _read_base(state_dir(home, name) / "base.json", name)
    return stored[1] if stored is not None and stored[0] == origin else None


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


def _merge_file(local: SkillFile, old: SkillFile, new: SkillFile) -> SkillFile | None:
    if local.executable == new.executable or new.executable == old.executable:
        executable = local.executable
    elif local.executable == old.executable:
        executable = new.executable
    else:
        return None
    if local.content == new.content or new.content == old.content:
        content = local.content
    elif local.content == old.content:
        content = new.content
    else:
        content = _merge_bytes(local.content, old.content, new.content)
    return SkillFile(content, executable) if content is not None else None


def merge_trees(base: dict[str, SkillFile], ours: dict[str, SkillFile],
                theirs: dict[str, SkillFile]) -> tuple[dict[str, SkillFile], list[str]]:
    merged, conflicts = {}, []
    for path in sorted(base.keys() | ours.keys() | theirs.keys()):
        old, local, new = base.get(path), ours.get(path), theirs.get(path)
        if local == new or new == old:
            accepted = local
        elif local == old:
            accepted = new
        elif old is not None and local is not None and new is not None:
            accepted = _merge_file(local, old, new)
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


def _apply_tree(home: Path, tree: dict[str, SkillFile], dest: Path) -> None:
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


def archive_pending(home: Path, name: str) -> None:
    pending = state_dir(home, name) / "pending"
    if pending.exists():
        pending.rename(pending.with_name("superseded-" + uuid.uuid4().hex))


def refresh_pending(name: str) -> None:
    """Explicitly archive stale review evidence before recomputing from current edits."""
    from tools import skills_sync as ss

    archive_pending(ss._hermes_home(), name)
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
