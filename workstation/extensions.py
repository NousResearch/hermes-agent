from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Optional
import urllib.request
import zipfile

from hermes_constants import get_hermes_home
from workstation.contracts import RiskLevel

_log = logging.getLogger(__name__)


_HIGH_RISK_PERMISSIONS = frozenset({
    "cookies", "webrequest", "webrequestblocking", "nativemessaging", "debugger",
    "management", "proxy", "enterprise.platformkeys",
})
_MEDIUM_RISK_PERMISSIONS = frozenset({
    "tabs", "downloads", "clipboardread", "clipboardwrite", "history", "bookmarks",
    "topSites", "sessions",
})
_LOW_RISK_PERMISSIONS = frozenset({"activetab", "storage", "contextmenus"})


def _permission_key(value: object) -> str:
    return str(value or "").strip().lower().replace("_", "")


def _is_broad_host_permission(value: object) -> bool:
    candidate = str(value or "").strip().lower()
    return candidate in {"<all_urls>", "*://*/*", "http://*/*", "https://*/*"}


def assess_extension_risk(manifest: dict[str, Any]) -> dict[str, Any]:
    """Classify manifest permissions before any extension is installed.

    This is deliberately conservative: an extension is executable third-party
    code in the authenticated Workstation profile.  Low-risk manifests are
    narrow enough for the policy engine to permit; broader host access or
    privileged browser APIs require a human approval boundary.
    """
    permissions = list(manifest.get("permissions") or [])
    host_permissions = list(manifest.get("host_permissions") or [])
    optional_permissions = list(manifest.get("optional_permissions") or [])
    optional_host_permissions = list(manifest.get("optional_host_permissions") or [])
    content_matches = [
        match
        for script in (manifest.get("content_scripts") or [])
        if isinstance(script, dict)
        for match in (script.get("matches") or [])
    ]
    all_permissions = permissions + optional_permissions
    all_hosts = host_permissions + optional_host_permissions + content_matches
    normalized = {_permission_key(item) for item in all_permissions}
    reasons: list[str] = []
    level = RiskLevel.LOW

    high = sorted(permission for permission in normalized if permission in _HIGH_RISK_PERMISSIONS)
    if high:
        level = RiskLevel.HIGH
        reasons.append(f"privileged permissions: {', '.join(high)}")
    if any(_is_broad_host_permission(host) for host in all_hosts):
        level = RiskLevel.HIGH
        reasons.append("broad host access")
    elif any(str(host).strip() for host in all_hosts):
        level = max(level, RiskLevel.MEDIUM, key=lambda item: list(RiskLevel).index(item))
        reasons.append("host-page access")

    medium = sorted(permission for permission in normalized if permission in _MEDIUM_RISK_PERMISSIONS)
    if medium and level != RiskLevel.HIGH:
        level = RiskLevel.MEDIUM
        reasons.append(f"browser-data permissions: {', '.join(medium)}")

    unknown = sorted(
        permission for permission in normalized
        if permission and permission not in _HIGH_RISK_PERMISSIONS
        and permission not in _MEDIUM_RISK_PERMISSIONS and permission not in _LOW_RISK_PERMISSIONS
    )
    if unknown and level == RiskLevel.LOW:
        level = RiskLevel.MEDIUM
        reasons.append(f"unclassified permissions: {', '.join(unknown)}")

    return {
        "risk_level": level.value,
        "reasons": reasons or ["limited active-tab/local-storage permissions"],
        "permissions": permissions,
        "host_permissions": host_permissions,
        "optional_permissions": optional_permissions,
        "optional_host_permissions": optional_host_permissions,
    }

def get_extensions_dir() -> Path:
    """Resolve directory for installed Chrome extensions."""
    d = get_hermes_home() / "workstation" / "extensions"
    d.mkdir(parents=True, exist_ok=True)
    return d


class ChromeExtensionManager:
    """Manager for discovering, downloading, unpacking and maintaining Chrome extensions."""

    CHROME_UPDATE_URL = (
        "https://clients2.google.com/service/update2/crx"
        "?response=redirect&prodversion=128.0.0.0&acceptformat=crx2,crx3&x=id%3D{ext_id}%26uc"
    )

    def __init__(self, storage_dir: Path | None = None) -> None:
        self.storage_dir = storage_dir or get_extensions_dir()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.storage_dir / "extensions.json"

    def extract_extension_id(self, input_str: str) -> str:
        """Extract a 32-character Chrome extension ID from a URL or raw ID string."""
        s = input_str.strip()
        # Direct 32-char ID
        if re.match(r"^[a-p]{32}$", s, re.IGNORECASE):
            return s.lower()

        # Match from webstore URL (e.g. /detail/.../cjpalhdlnbpafiamejdnhcphjbkeiagm)
        match = re.search(r"([a-p]{32})(?:[/?&#]|$)", s, re.IGNORECASE)
        if match:
            return match.group(1).lower()

        raise ValueError(f"Invalid Chrome Web Store extension ID or URL: '{input_str}'")

    def unpack_crx(self, crx_bytes: bytes, dest_dir: Path) -> Path:
        """Unpack CRX package by locating the underlying ZIP stream."""
        if not crx_bytes.startswith(b"Cr24"):
            raise ValueError("Not a valid CRX format (missing Cr24 magic signature)")

        # In both CRX2 and CRX3, the archive payload is a standard ZIP starting with PK\x03\x04
        zip_offset = crx_bytes.find(b"PK\x03\x04")
        if zip_offset == -1:
            raise ValueError("Failed to locate ZIP payload inside CRX archive")

        zip_data = crx_bytes[zip_offset:]
        with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
            for member in z.infolist():
                member_path = Path(member.filename)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ValueError("CRX archive contains an unsafe path")
            dest_dir.mkdir(parents=True, exist_ok=True)
            z.extractall(dest_dir)

        return dest_dir

    def download_crx(self, ext_id: str) -> bytes:
        """Download .crx bytes directly from Google Web Store update endpoint."""
        url = self.CHROME_UPDATE_URL.format(ext_id=ext_id)
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
                )
            },
        )
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            return resp.read()

    def get_manifest(self, ext_dir: Path) -> dict[str, Any]:
        """Read and parse extension manifest.json."""
        manifest_path = ext_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found in {ext_dir}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_registry(self) -> dict[str, Any]:
        if not self.registry_file.exists():
            return {"extensions": {}}
        try:
            with open(self.registry_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"extensions": {}}

    def _save_registry(self, data: dict[str, Any]) -> None:
        temporary = self.registry_file.with_suffix(".tmp")
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(temporary, self.registry_file)

    def inspect_crx(self, ext_id: str, crx_bytes: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
        """Read a CRX manifest without persisting executable extension files."""
        with tempfile.TemporaryDirectory(prefix=f"hermes-extension-{ext_id}-") as temp:
            unpacked = self.unpack_crx(crx_bytes, Path(temp) / ext_id)
            manifest = self.get_manifest(unpacked)
        if not isinstance(manifest, dict) or not str(manifest.get("version") or "").strip():
            raise ValueError("Extension manifest must be an object with a version")
        return manifest, assess_extension_risk(manifest)

    def install_from_bytes(self, ext_id: str, crx_bytes: bytes) -> dict[str, Any]:
        """Install extension from raw CRX bytes (useful for offline/tests)."""
        target_dir = self.storage_dir / ext_id
        staging_dir = Path(tempfile.mkdtemp(prefix=f".{ext_id}-", dir=self.storage_dir))
        unpacked = staging_dir / "extension"
        try:
            self.unpack_crx(crx_bytes, unpacked)
            manifest = self.get_manifest(unpacked)
            if not isinstance(manifest, dict) or not str(manifest.get("version") or "").strip():
                raise ValueError("Extension manifest must be an object with a version")
            if target_dir.exists():
                shutil.rmtree(target_dir)
            os.replace(unpacked, target_dir)
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

        name = manifest.get("name", ext_id)
        # Handle localized message __MSG_name__ fallback
        if name.startswith("__MSG_"):
            name = ext_id

        entry = {
            "id": ext_id,
            "name": name,
            "version": manifest.get("version", "1.0.0"),
            "description": manifest.get("description", ""),
            "path": str(target_dir),
            "enabled": True,
            "permissions": manifest.get("permissions", []),
            "host_permissions": manifest.get("host_permissions", []),
            "options_page": manifest.get("options_page") or manifest.get("options_ui", {}).get("page"),
            "risk": assess_extension_risk(manifest),
        }

        reg = self._load_registry()
        reg["extensions"][ext_id] = entry
        self._save_registry(reg)
        return entry

    def install_extension(self, identifier_or_url: str) -> dict[str, Any]:
        """Download, unpack and register a Chrome extension from webstore URL or ID."""
        ext_id = self.extract_extension_id(identifier_or_url)
        crx_bytes = self.download_crx(ext_id)
        return self.install_from_bytes(ext_id, crx_bytes)

    def list_installed_extensions(self) -> list[dict[str, Any]]:
        """List all registered extensions."""
        reg = self._load_registry()
        return list(reg.get("extensions", {}).values())

    def uninstall_extension(self, ext_id: str) -> bool:
        """Remove extension from disk and registry."""
        clean_id = ext_id.lower().strip()
        reg = self._load_registry()
        if clean_id not in reg.get("extensions", {}):
            return False

        del reg["extensions"][clean_id]
        self._save_registry(reg)

        target_dir = self.storage_dir / clean_id
        if target_dir.exists():
            import shutil
            shutil.rmtree(target_dir, ignore_errors=True)
        return True

    def get_options_url(self, ext_id: str) -> str | None:
        """Get the options page URL if declared in manifest."""
        clean_id = ext_id.lower().strip()
        reg = self._load_registry()
        ext = reg.get("extensions", {}).get(clean_id)
        if not ext:
            return None
        opt = ext.get("options_page")
        if opt:
            return f"chrome-extension://{clean_id}/{opt}"
        return f"chrome-extension://{clean_id}/options.html"
