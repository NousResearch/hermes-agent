"""Persistent, content-addressed skill snapshots for Docker.

Snapshots live under the configured Docker data root. A process exit must not
delete a directory a container still has mounted. Old snapshots are removed
only by ``cleanup_unreferenced`` after a mount check succeeds.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import re
import tempfile
import subprocess
from pathlib import Path


_SKIP_NAMES = {
    ".git", ".ssh", ".env", "secret", "secrets", "credentials",
    "credentials.json", "token", "auth.json", "node_modules",
}
_MANIFEST = "snapshot-manifest.json"


def docker_data_root() -> Path:
    """Independently configured root for Docker-owned writable state."""
    value = os.environ.get("HERMES_DOCKER_DATA_ROOT", "").strip()
    if not value:
        config = Path.home() / ".config" / "hermes-security" / "docker.json"
        try:
            value = json.loads(config.read_text(encoding="utf-8"))["data_root"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError("Docker data root is not configured") from exc
    root = Path(value).expanduser().resolve()
    if len(root.parts) >= 3 and root.parts[1] == "Volumes" and not Path("/Volumes", root.parts[2]).is_mount():
        raise RuntimeError("Docker data disk is not mounted")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _permitted(name: str) -> bool:
    folded = name.casefold()
    if folded.startswith(".") or folded in _SKIP_NAMES:
        return False
    if folded.startswith(".env") or folded.endswith(".env") or ".env." in folded:
        return False
    return True


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def _materialize(source: Path):
    """Yield ``(relative posix path, readable file)`` without leaving ``source``.

    A symlink is copied only when its target stays inside ``source``. Targets
    outside the tree, credential names, and dot directories are skipped.
    """
    root = source.resolve()

    def walk(directory: Path, relative: Path, ancestors: frozenset[Path] = frozenset()):
        real = directory.resolve()
        if real in ancestors:
            return
        ancestors = ancestors | {real}
        try:
            entries = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError:
            return
        for entry in entries:
            if not _permitted(entry.name):
                continue
            target = entry
            if entry.is_symlink():
                try:
                    resolved = entry.resolve()
                except OSError:
                    continue
                if not _inside(resolved, root):
                    continue
                if not all(_permitted(part) for part in resolved.relative_to(root).parts):
                    continue
                target = resolved
            rel = relative / entry.name
            if target.is_dir() and not target.is_symlink():
                yield from walk(target, rel, ancestors)
            elif target.is_file():
                yield rel.as_posix(), target

    yield from walk(source, Path())


def fingerprint_tree(source: Path) -> str:
    """Digest of materialized skill bytes. Directory mtimes are not included."""
    digest = hashlib.sha256()
    for rel, path in _materialize(source):
        digest.update(rel.encode("utf-8", "surrogateescape"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_files(source: Path) -> list[dict]:
    rows = []
    for rel, path in _materialize(source):
        parts = rel.split("/")
        if "scripts" in parts and rel.endswith(".py"):
            rows.append({"path": rel, "sha256": _file_sha(path)})
    return rows


def _snapshot_base(profile: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in (profile or "default"))
    if safe in {".", ".."}:
        safe = "default"
    base = docker_data_root() / "skills-snapshots" / (safe or "default")
    base.mkdir(parents=True, exist_ok=True)
    return base


def stage_skills(source: Path, profile: str, container_path: str) -> tuple[Path, str]:
    """Return ``(snapshot_dir, fingerprint)``, reusing an identical snapshot.

    Creating a snapshot does not delete older ones. The snapshot directory is
    not a ``tempfile`` and is not registered for process-exit deletion.
    """
    source = Path(source)
    if not source.is_dir():
        raise RuntimeError(f"skill source is not a directory: {source}")
    fingerprint = fingerprint_tree(source)
    base = _snapshot_base(profile)
    target = base / fingerprint
    manifest_path = base / f"{fingerprint}.manifest.json"
    if target.is_dir() and manifest_path.is_file():
        problem = snapshot_usable(target, fingerprint, manifest_path)
        if problem is None:
            return target, fingerprint
    if target.exists() or target.is_symlink():
        # Never replace a tree that an existing container may still have mounted.
        raise RuntimeError(f"existing skill snapshot is unusable; refusing replacement: {target}")
    staging = Path(tempfile.mkdtemp(prefix=f".partial-{fingerprint[:12]}-", dir=base))
    try:
        for rel, path in _materialize(source):
            destination = staging / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        if fingerprint_tree(staging) != fingerprint:
            raise RuntimeError("skill source changed while staging")
        manifest = {
            "fingerprint": fingerprint,
            "profile": profile,
            "container_path": container_path,
            "source_name": source.name,
            "files": _required_files(staging),
        }
        try:
            staging.rename(target)
        except FileExistsError:
            # A concurrent process may have published the same immutable tree.
            if snapshot_usable(target, fingerprint, manifest_path) is not None:
                raise RuntimeError("concurrent skill snapshot publication is incomplete")
        else:
            temp_manifest = base / (staging.name + ".manifest.json")
            try:
                temp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                temp_manifest.replace(manifest_path)
            finally:
                temp_manifest.unlink(missing_ok=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target, fingerprint


def snapshot_usable(source: Path, fingerprint: str, manifest_path: Path | None = None) -> str | None:
    """Return an error code when a mounted snapshot must not be reused."""
    source = Path(source)
    if "hermes-skills-safe-" in source.name or source.name.startswith("tmp"):
        return "ephemeral_source"
    if source.is_symlink() or not source.is_dir():
        return "source_missing"
    manifest_file = manifest_path or (source.parent / f"{source.name}.manifest.json")
    try:
        manifest = json.loads(Path(manifest_file).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "manifest_unreadable"
    if not isinstance(manifest, dict) or manifest.get("fingerprint") != fingerprint:
        return "fingerprint_mismatch"
    required = manifest.get("files")
    if not isinstance(required, list):
        return "required_unreadable"
    for row in required:
        if not isinstance(row, dict):
            return "required_unreadable"
        rel = str(row.get("path") or "")
        if not rel or rel.startswith("/") or ".." in rel.split("/"):
            return "required_unreadable"
        path = source / rel
        if not path.is_file():
            return "required_missing"
        try:
            readable = os.access(path, os.R_OK)
            digest = _file_sha(path)
        except OSError:
            return "required_unreadable"
        if not readable or digest != row.get("sha256"):
            return "required_mismatch"
    try:
        if fingerprint_tree(source) != fingerprint:
            return "content_mismatch"
    except (OSError, RuntimeError):
        return "content_unreadable"
    return None


def mounted_sources() -> set[str] | None:
    """Resolved bind-mount sources of Hermes containers, or ``None`` if unknown."""
    try:
        listed = subprocess.run(
            ["docker", "ps", "-aq", "--filter", "label=hermes-agent=1"],
            capture_output=True, text=True, timeout=15, stdin=subprocess.DEVNULL,
        )
        if listed.returncode != 0:
            return None
        sources: set[str] = set()
        for cid in listed.stdout.split():
            inspected = subprocess.run(
                ["docker", "inspect", "--format", "{{json .Mounts}}", cid],
                capture_output=True, text=True, timeout=15, stdin=subprocess.DEVNULL,
            )
            if inspected.returncode != 0:
                return None
            mounts = json.loads(inspected.stdout or "[]")
            if not isinstance(mounts, list):
                return None
            for mount in mounts:
                if not isinstance(mount, dict):
                    return None
                raw = str((mount or {}).get("Source") or "")
                if raw:
                    sources.add(str(Path(raw).resolve()))
        return sources
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


def safe_to_delete(path: Path) -> bool:
    """False when ``path`` is mounted or the mount check itself failed."""
    sources = mounted_sources()
    if sources is None:
        return False
    resolved = str(Path(path).resolve())
    return not any(_paths_overlap(resolved, item) for item in sources)


def _paths_overlap(left: str, right: str) -> bool:
    return left == right or left.startswith(right + os.sep) or right.startswith(left + os.sep)


def cleanup_unreferenced(profile: str, *, mounted: set[str] | None = None) -> list[str]:
    """Delete this profile's snapshots that no container mounts.

    ``mounted is None`` queries Docker. A failed query deletes nothing.
    """
    if mounted is None:
        mounted = mounted_sources()
    if mounted is None:
        return []
    base = _snapshot_base(profile)
    removed = []
    for child in list(base.iterdir()):
        if child.is_symlink() or not child.is_dir() or not re.fullmatch(r"[0-9a-f]{64}", child.name):
            continue
        resolved = str(child.resolve())
        if any(_paths_overlap(resolved, str(Path(item).resolve())) for item in mounted):
            continue
        try:
            shutil.rmtree(child)
        except OSError:
            continue
        manifest = base / f"{child.name}.manifest.json"
        manifest.unlink(missing_ok=True)
        removed.append(child.name)
    return removed
