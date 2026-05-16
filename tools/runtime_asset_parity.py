#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

IGNORED_DIR_NAMES = frozenset({"__pycache__"})
IGNORED_FILE_NAMES = frozenset({".DS_Store"})
IGNORED_SUFFIXES = frozenset({".pyc"})
IGNORED_PATTERNS = ("__pycache__/", "*.pyc", ".DS_Store")
MANIFEST_ROOT = Path(
    "/Users/preston/.local/state/hermes-operator/user-asset-install-manifests"
)
MANIFEST_SCHEMA = "hermes.user_asset_install.v2"

SECRET_VALUE_RE = re.compile(
    r"(?i)['\"]?"
    r"(api[_-]?key|token|password|passwd|private[_ -]?key|secret|credential)"
    r"['\"]?"
    r"\s*[:=]\s*['\"]?([A-Za-z0-9_./+=-]{12,})"
)
SECRET_PATH_PARTS = {
    ".env",
    "auth.json",
    "credentials.json",
    "cookies.sqlite",
    "id_rsa",
    "id_ed25519",
}
SECRET_PATH_MARKERS = (
    "private-key",
    "private_key",
    "client-secret",
    "client_secret",
)


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def git_text(repo: Path, args: list[str]) -> str | None:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def source_identity(repo: Path) -> dict[str, str | None]:
    return {
        "source_branch": git_text(repo, ["branch", "--show-current"]),
        "source_head": git_text(repo, ["rev-parse", "HEAD"]),
    }


def ignored_runtime_path(path: Path) -> bool:
    return (
        any(part in IGNORED_DIR_NAMES for part in path.parts)
        or path.name in IGNORED_FILE_NAMES
        or path.suffix in IGNORED_SUFFIXES
    )


def iter_managed_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if root.is_file():
        return [] if ignored_runtime_path(root) else [root]
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and not ignored_runtime_path(path)
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def inventory(root: Path) -> dict[str, Any]:
    if not root.exists():
        return {"exists": False}
    if root.is_file():
        return {
            "exists": True,
            "type": "file",
            "ignored": ignored_runtime_path(root),
            "sha256": None if ignored_runtime_path(root) else sha256_file(root),
            "size": root.stat().st_size,
        }
    files: dict[str, dict[str, Any]] = {}
    for path in iter_managed_files(root):
        rel = str(path.relative_to(root))
        files[rel] = {"sha256": sha256_file(path), "size": path.stat().st_size}
    return {"exists": True, "type": "dir", "files": files}


def compare_paths(source: Path, installed: Path) -> dict[str, Any]:
    source_inventory = inventory(source)
    installed_inventory = inventory(installed)
    result: dict[str, Any] = {
        "source": str(source),
        "installed": str(installed),
        "source_exists": source.exists(),
        "installed_exists": installed.exists(),
        "source_inventory": source_inventory,
        "installed_inventory": installed_inventory,
        "ignored_patterns": list(IGNORED_PATTERNS),
    }
    if not source.exists() or not installed.exists():
        result.update(
            {
                "matches_source": False,
                "missing_files": [],
                "extra_files": [],
                "changed_files": [],
            }
        )
        return result
    if source.is_file() or installed.is_file():
        matches = (
            source.is_file()
            and installed.is_file()
            and not ignored_runtime_path(source)
            and not ignored_runtime_path(installed)
            and sha256_file(source) == sha256_file(installed)
        )
        result.update(
            {
                "matches_source": matches,
                "missing_files": [],
                "extra_files": [],
                "changed_files": [] if matches else [source.name],
            }
        )
        return result

    source_files = source_inventory.get("files", {})
    installed_files = installed_inventory.get("files", {})
    missing = sorted(set(source_files) - set(installed_files))
    extra = sorted(set(installed_files) - set(source_files))
    changed = sorted(
        rel
        for rel in set(source_files) & set(installed_files)
        if source_files[rel]["sha256"] != installed_files[rel]["sha256"]
    )
    result.update(
        {
            "matches_source": not missing and not extra and not changed,
            "missing_files": missing,
            "extra_files": extra,
            "changed_files": changed,
        }
    )
    return result


def managed_files_equal(source: Path, installed: Path) -> bool:
    return bool(compare_paths(source, installed)["matches_source"])


def secret_like_findings(root: Path) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for path in iter_managed_files(root):
        rel = str(path.relative_to(root)) if root.is_dir() else path.name
        lower_parts = {part.lower() for part in path.parts}
        lower_rel = rel.lower()
        if lower_parts & SECRET_PATH_PARTS or any(
            marker in lower_rel for marker in SECRET_PATH_MARKERS
        ):
            findings.append({"path": rel, "reason": "secret-looking filename"})
            continue
        if path.suffix.lower() not in {
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".md",
            ".py",
            ".sh",
            "",
        }:
            continue
        text = path.read_text(errors="ignore")
        if "BEGIN PRIVATE KEY" in text or re.search(r"sk-[A-Za-z0-9]{20,}", text):
            findings.append({"path": rel, "reason": "secret-looking content"})
            continue
        if SECRET_VALUE_RE.search(text):
            for match in SECRET_VALUE_RE.finditer(text):
                value = match.group(2)
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", value):
                    continue
                findings.append({"path": rel, "reason": "secret-looking assignment"})
                break
    return findings


def latest_manifest(manifest_root: Path = MANIFEST_ROOT) -> dict[str, Any] | None:
    if not manifest_root.exists():
        return None
    manifests = sorted(
        path
        for path in manifest_root.glob("*.json")
        if path.name != "latest.json" and path.is_file()
    )
    if not manifests:
        latest = manifest_root / "latest.json"
        manifests = [latest] if latest.exists() else []
    if not manifests:
        return None
    path = manifests[-1]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("manifest_path", str(path))
    return payload
