#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import runtime_asset_parity


HERMES_ROOT = Path("/Users/preston/.hermes/hermes-agent")
BACKUP_ROOT = Path("/Users/preston/.local/state/hermes-operator/hermes-config-backups")
MANIFEST_ROOT = runtime_asset_parity.MANIFEST_ROOT

@dataclass(frozen=True)
class AssetInstall:
    label: str
    source: Path
    installed: Path
    kind: str


INSTALLS = (
    AssetInstall(
        "crypto-bot-pm skill",
        HERMES_ROOT / "skills/project-management/crypto-bot-pm",
        Path("/Users/preston/.hermes/skills/project-management/crypto-bot-pm"),
        "dir",
    ),
    AssetInstall(
        "codex-sidecar skill",
        HERMES_ROOT / "skills/development/codex-sidecar",
        Path("/Users/preston/.hermes/skills/development/codex-sidecar"),
        "dir",
    ),
    AssetInstall(
        "crypto-bot-pm plugin",
        HERMES_ROOT / "plugins/crypto-bot-pm",
        Path("/Users/preston/.hermes/plugins/crypto-bot-pm"),
        "dir",
    ),
    AssetInstall(
        "hermes-codex-audit wrapper",
        HERMES_ROOT / "wrappers/hermes-codex-audit",
        Path("/Users/preston/.local/bin/hermes-codex-audit"),
        "executable",
    ),
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def ignored_copy_names(_: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name in runtime_asset_parity.IGNORED_DIR_NAMES
        or name in runtime_asset_parity.IGNORED_FILE_NAMES
        or Path(name).suffix in runtime_asset_parity.IGNORED_SUFFIXES
    }


def copy_dir(src: Path, dest: Path) -> None:
    shutil.copytree(src, dest, ignore=ignored_copy_names)


def backup_existing(path: Path, backup_dir: Path) -> Path | None:
    if not path.exists():
        return None
    backup_path = backup_dir / path.relative_to(path.anchor)
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        copy_dir(path, backup_path)
    else:
        shutil.copy2(path, backup_path)
    return backup_path


def temp_install_path(dest: Path, timestamp: str) -> Path:
    return dest.parent / f".{dest.name}.tmp-{timestamp}-{os.getpid()}"


def prepare_temp_install(src: Path, dest: Path, kind: str, timestamp: str) -> Path:
    if not src.exists():
        raise FileNotFoundError(f"Source asset missing: {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_path = temp_install_path(dest, timestamp)
    if temp_path.exists():
        if temp_path.is_dir():
            shutil.rmtree(temp_path)
        else:
            temp_path.unlink()
    if kind == "dir":
        copy_dir(src, temp_path)
    else:
        shutil.copy2(src, temp_path)
        if kind == "executable":
            temp_path.chmod(
                temp_path.stat().st_mode
                | stat.S_IXUSR
                | stat.S_IXGRP
                | stat.S_IXOTH
            )
    if not runtime_asset_parity.managed_files_equal(src, temp_path):
        raise RuntimeError(
            f"Staged install does not match source: {src} -> {temp_path}"
        )
    return temp_path


def replace_with_temp(temp_path: Path, dest: Path) -> None:
    if dest.exists():
        if dest.is_dir():
            shutil.rmtree(dest)
        else:
            dest.unlink()
    temp_path.replace(dest)


def assert_no_secret_sources(asset: AssetInstall) -> None:
    findings = runtime_asset_parity.secret_like_findings(asset.source)
    if findings:
        paths = ", ".join(item["path"] for item in findings[:5])
        raise RuntimeError(
            f"Secret-looking source asset blocked for {asset.label}: {paths}"
        )


def install_one(
    asset: AssetInstall,
    backup_dir: Path,
    timestamp: str,
) -> dict[str, Any]:
    assert_no_secret_sources(asset)
    temp_path = prepare_temp_install(
        asset.source,
        asset.installed,
        asset.kind,
        timestamp,
    )
    backup_path = backup_existing(asset.installed, backup_dir)
    replace_with_temp(temp_path, asset.installed)
    if asset.kind == "executable":
        asset.installed.chmod(
            asset.installed.stat().st_mode
            | stat.S_IXUSR
            | stat.S_IXGRP
            | stat.S_IXOTH
        )
    comparison = runtime_asset_parity.compare_paths(asset.source, asset.installed)
    if not comparison["matches_source"]:
        raise RuntimeError(
            f"Installed asset does not match source after copy: {asset.label}"
        )
    return {
        "label": asset.label,
        "source": str(asset.source),
        "installed": str(asset.installed),
        "kind": asset.kind,
        "backup_path": str(backup_path) if backup_path else None,
        "verified": True,
        "source_inventory": comparison["source_inventory"],
        "installed_inventory": comparison["installed_inventory"],
    }


def install_assets(
    *,
    installs: tuple[AssetInstall, ...] = INSTALLS,
    hermes_root: Path = HERMES_ROOT,
    backup_root: Path = BACKUP_ROOT,
    manifest_root: Path = MANIFEST_ROOT,
) -> dict[str, Any]:
    timestamp = runtime_asset_parity.utc_timestamp()
    backup_dir = backup_root / timestamp
    backup_dir.mkdir(parents=True, exist_ok=False)
    manifest_root.mkdir(parents=True, exist_ok=True)

    assets: dict[str, dict[str, Any]] = {}
    for asset in installs:
        record = install_one(asset, backup_dir, timestamp)
        assets[asset.label] = record

    identity = runtime_asset_parity.source_identity(hermes_root)
    manifest_path = manifest_root / f"{timestamp}-user-assets.json"
    manifest = {
        "schema": runtime_asset_parity.MANIFEST_SCHEMA,
        "generated_at": utc_now(),
        "manifest_path": str(manifest_path),
        "backup_dir": str(backup_dir),
        "ignored_patterns": list(runtime_asset_parity.IGNORED_PATTERNS),
        **identity,
        "assets": assets,
        "installed": {
            record["installed"]: record["source"] for record in assets.values()
        },
        "backups": {
            record["installed"]: record["backup_path"] for record in assets.values()
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    latest_path = manifest_root / "latest.json"
    latest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=("json",), default="json")
    args = parser.parse_args()
    _ = args

    print(json.dumps(install_assets(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
