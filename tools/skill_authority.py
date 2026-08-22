"""Central runtime skill authority manifest and deterministic drift checks."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from hermes_constants import get_hermes_home


CRITICAL_SKILLS = (
    "lah-workflow",
    "lah-workflow-small-model",
    "lah-repo-router",
    "mission-decomposer",
)
MANIFEST_FILENAME = ".governance_manifest.json"


def classify_skill_identifier(identifier: str, canonical_names: set[str]) -> str:
    """Classify a reference without changing resolver compatibility behavior."""
    if identifier in canonical_names:
        return "VALID_CANONICAL_NAME"
    if ":" in identifier and identifier.count(":") == 1:
        return "VALID_PLUGIN_NAMESPACE"
    if "/" in identifier:
        return "CATEGORY_PATH_USED_AS_IDENTIFIER"
    return "UNKNOWN_SKILL"


def manifest_path(runtime_root: Path | None = None) -> Path:
    root = runtime_root or (get_hermes_home() / "skills")
    return root / MANIFEST_FILENAME


def _skill_fingerprint(skill_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in skill_dir.rglob("*") if p.is_file() and not p.is_symlink()):
        relative = path.relative_to(skill_dir).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _frontmatter_name(skill_dir: Path) -> str:
    path = skill_dir / "SKILL.md"
    text = path.read_text(encoding="utf-8")
    in_frontmatter = False
    for line in text.splitlines():
        if line.strip() == "---":
            in_frontmatter = not in_frontmatter
            continue
        if in_frontmatter and line.strip().startswith("name:"):
            return line.split(":", 1)[1].strip().strip("\"'")
    return skill_dir.name


def _find_skill(runtime_root: Path, name: str) -> Path | None:
    matches = []
    for skill_md in runtime_root.rglob("SKILL.md"):
        if any(part in {".archive", ".git", "node_modules"} for part in skill_md.parts):
            continue
        try:
            if _frontmatter_name(skill_md.parent) == name:
                matches.append(skill_md.parent)
        except (OSError, UnicodeError):
            continue
    if len(matches) != 1:
        return None
    return matches[0]


def _git_sha(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def build_manifest(runtime_root: Path, declarations: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    skills: dict[str, Any] = {}
    for name, declaration in sorted(declarations.items()):
        runtime_dir = _find_skill(runtime_root, name)
        if runtime_dir is None:
            raise ValueError(f"runtime skill not uniquely discoverable: {name}")
        source_dir = Path(str(declaration["source_path"])).resolve()
        skills[name] = {
            "invocation_name": name,
            "source_repo": declaration.get("source_repo"),
            "source_path": str(source_dir),
            "source_sha": declaration.get("source_sha") or _git_sha(source_dir),
            "source_content_sha256": _skill_fingerprint(source_dir),
            "runtime_path": str(runtime_dir.relative_to(runtime_root)),
            "runtime_content_sha256": _skill_fingerprint(runtime_dir),
        }
    return {"schema_version": 1, "skills": skills}


def deploy_runtime_authority(
    runtime_root: Path,
    declarations: Mapping[str, Mapping[str, Any]],
    *,
    allow_runtime_drift: bool = False,
) -> dict[str, Any]:
    """Atomically deploy declared skill sources and write one provenance manifest.

    Existing target drift is rejected unless the caller explicitly confirms it
    was preserved and reviewed. This keeps raw manual copying out of the
    deployment contract while allowing a deliberate first convergence.
    """
    runtime_root = runtime_root.resolve()
    runtime_root.mkdir(parents=True, exist_ok=True)
    plans: list[tuple[str, Path, Path, str, str | None, Mapping[str, Any]]] = []
    for name, declaration in sorted(declarations.items()):
        source_dir = Path(str(declaration["source_path"])).resolve()
        if not (source_dir / "SKILL.md").is_file():
            raise ValueError(f"source skill missing: {name}")
        runtime_path = declaration.get("runtime_path")
        target_dir = (
            runtime_root / str(runtime_path)
            if runtime_path
            else _find_skill(runtime_root, name)
        )
        if target_dir is None:
            raise ValueError(f"runtime skill not uniquely discoverable: {name}")
        target_dir = Path(target_dir).resolve()
        if target_dir.exists():
            target_hash = _skill_fingerprint(target_dir)
            source_hash = _skill_fingerprint(source_dir)
            if target_hash != source_hash and not allow_runtime_drift:
                raise ValueError(f"unexpected runtime drift: {name}")
        else:
            source_hash = _skill_fingerprint(source_dir)
        source_sha = declaration.get("source_sha") or _git_sha(source_dir)
        plans.append((name, source_dir, target_dir, source_hash, source_sha, declaration))

    staging_root = Path(tempfile.mkdtemp(prefix=".governance-deploy-", dir=runtime_root.parent))
    manifest: dict[str, Any] = {"schema_version": 1, "skills": {}}
    try:
        for name, source_dir, target_dir, source_hash, source_sha, declaration in plans:
            relative = target_dir.relative_to(runtime_root)
            staged_dir = staging_root / "payload" / relative
            staged_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_dir, staged_dir)
            runtime_hash = _skill_fingerprint(staged_dir)
            manifest["skills"][name] = {
                "invocation_name": name,
                "source_repo": declaration.get("source_repo"),
                "source_path": str(source_dir),
                "source_sha": source_sha,
                "source_content_sha256": source_hash,
                "runtime_path": relative.as_posix(),
                "runtime_content_sha256": runtime_hash,
                "deployment_method": "tools.skill_authority.deploy_runtime_authority",
                "deployed_at": datetime.now(timezone.utc).isoformat(),
            }

        for _, _, target_dir, _, _, _ in plans:
            relative = target_dir.relative_to(runtime_root)
            staged_dir = staging_root / "payload" / relative
            if target_dir.exists():
                shutil.rmtree(target_dir)
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged_dir, target_dir)

        manifest_file = manifest_path(runtime_root)
        manifest_tmp = manifest_file.with_name(f"{manifest_file.name}.tmp")
        manifest_tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(manifest_tmp, manifest_file)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return manifest


def validate_runtime_authority(
    runtime_root: Path,
    manifest: Mapping[str, Any],
    *,
    critical: Sequence[str] = CRITICAL_SKILLS,
) -> dict[str, Any]:
    errors: list[str] = []
    entries = manifest.get("skills") if isinstance(manifest, Mapping) else None
    if manifest.get("schema_version") != 1 or not isinstance(entries, Mapping):
        return {"valid": False, "errors": ["invalid governance manifest schema"], "skills": {}}

    discovered: dict[str, list[Path]] = {}
    for skill_md in runtime_root.rglob("SKILL.md"):
        if any(part in {".archive", ".git", "node_modules"} for part in skill_md.parts):
            continue
        try:
            discovered.setdefault(_frontmatter_name(skill_md.parent), []).append(skill_md.parent)
        except (OSError, UnicodeError):
            continue
    for name, paths in discovered.items():
        if len(paths) > 1:
            errors.append(f"duplicate canonical skill name: {name}")

    results: dict[str, Any] = {}
    for name in critical:
        entry = entries.get(name)
        if not isinstance(entry, Mapping):
            errors.append(f"{name}: missing manifest entry")
            continue
        if entry.get("invocation_name") != name:
            errors.append(f"{name}: invocation identity missing or mismatched")
        if not entry.get("source_repo"):
            errors.append(f"{name}: source repository missing")
        runtime_dir = runtime_root / str(entry.get("runtime_path", ""))
        source_dir = Path(str(entry.get("source_path", "")))
        if not (runtime_dir / "SKILL.md").is_file():
            errors.append(f"{name}: runtime skill missing")
            continue
        if not (source_dir / "SKILL.md").is_file():
            errors.append(f"{name}: source skill missing")
            continue
        if _frontmatter_name(runtime_dir) != name:
            errors.append(f"{name}: runtime canonical name mismatch")
        source_hash = _skill_fingerprint(source_dir)
        runtime_hash = _skill_fingerprint(runtime_dir)
        declared_source_hash = entry.get("source_content_sha256")
        declared_runtime_hash = entry.get("runtime_content_sha256")
        if declared_source_hash and declared_source_hash != source_hash:
            errors.append(f"{name}: declared source fingerprint mismatch")
        if declared_runtime_hash and declared_runtime_hash != runtime_hash:
            errors.append(f"{name}: declared runtime fingerprint mismatch")
        declared_sha = entry.get("source_sha")
        current_sha = _git_sha(source_dir) if declared_sha else None
        if declared_sha and current_sha and declared_sha != current_sha:
            errors.append(f"{name}: source Git SHA drift")
        results[name] = {
            "source_content_sha256": source_hash,
            "runtime_content_sha256": runtime_hash,
            "match": source_hash == runtime_hash,
            "source_sha": current_sha or declared_sha,
            "invocation_name": entry.get("invocation_name"),
        }
        if source_hash != runtime_hash:
            errors.append(f"{name}: content drift between source and runtime")
    return {"valid": not errors, "errors": errors, "skills": results}


def load_runtime_authority_status(runtime_root: Path | None = None) -> dict[str, Any]:
    root = runtime_root or (get_hermes_home() / "skills")
    path = manifest_path(root)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"valid": False, "errors": [f"missing or invalid manifest: {path}"], "skills": {}}
    return validate_runtime_authority(root, manifest)
