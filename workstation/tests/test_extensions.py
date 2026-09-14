from __future__ import annotations

import io
import json
import zipfile
import pytest

from workstation.extensions import ChromeExtensionManager, assess_extension_risk


def create_mock_crx(manifest_data: dict) -> bytes:
    """Helper to generate a minimal valid mock CRX package."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest_data))
        z.writestr("options.html", "<html><body>Options</body></html>")

    zip_bytes = zip_buffer.getvalue()
    # CRX header: Cr24 magic + 16 dummy header bytes + ZIP payload
    crx_header = b"Cr24" + b"\x00" * 16
    return crx_header + zip_bytes


def test_extract_extension_id():
    mgr = ChromeExtensionManager()

    # Direct ID
    direct_id = "cjpalhdlnbpafiamejdnhcphjbkeiagm"
    assert mgr.extract_extension_id(direct_id) == direct_id

    # Chrome Web Store modern URL
    url1 = "https://chromewebstore.google.com/detail/ublock-origin/cjpalhdlnbpafiamejdnhcphjbkeiagm"
    assert mgr.extract_extension_id(url1) == direct_id

    # Chrome Web Store legacy URL with query params
    url2 = "https://chrome.google.com/webstore/detail/ublock-origin/cjpalhdlnbpafiamejdnhcphjbkeiagm?hl=en-US"
    assert mgr.extract_extension_id(url2) == direct_id

    # Invalid input
    with pytest.raises(ValueError, match="Invalid Chrome Web Store extension ID"):
        mgr.extract_extension_id("https://google.com/search")


def test_install_and_manage_extension(tmp_path):
    mgr = ChromeExtensionManager(storage_dir=tmp_path)

    ext_id = "cjpalhdlnbpafiamejdnhcphjbkeiagm"
    manifest = {
        "manifest_version": 3,
        "name": "uBlock Origin",
        "version": "1.58.0",
        "description": "An efficient blocker",
        "permissions": ["storage", "webRequest"],
        "options_page": "options.html",
    }

    mock_crx = create_mock_crx(manifest)
    installed = mgr.install_from_bytes(ext_id, mock_crx)

    assert installed["id"] == ext_id
    assert installed["name"] == "uBlock Origin"
    assert installed["version"] == "1.58.0"
    assert installed["enabled"] is True

    # Check persistence and listing
    all_exts = mgr.list_installed_extensions()
    assert len(all_exts) == 1
    assert all_exts[0]["id"] == ext_id

    # Options URL
    opt_url = mgr.get_options_url(ext_id)
    assert opt_url == f"chrome-extension://{ext_id}/options.html"

    # Uninstall
    assert mgr.uninstall_extension(ext_id) is True
    assert len(mgr.list_installed_extensions()) == 0


def test_extension_risk_assessment_is_conservative():
    assert assess_extension_risk({"permissions": ["activeTab", "storage"]})["risk_level"] == "low"
    assert assess_extension_risk({"permissions": ["tabs"]})["risk_level"] == "medium"
    assert assess_extension_risk({"permissions": ["cookies"]})["risk_level"] == "high"
    assert assess_extension_risk({"host_permissions": ["<all_urls>"]})["risk_level"] == "high"
    assert assess_extension_risk({"permissions": ["unknownExperimentalPermission"]})["risk_level"] == "medium"


def test_corrupt_and_unsafe_crx_never_registers_extension(tmp_path):
    mgr = ChromeExtensionManager(storage_dir=tmp_path)
    ext_id = "cjpalhdlnbpafiamejdnhcphjbkeiagm"

    with pytest.raises(ValueError, match="missing Cr24"):
        mgr.install_from_bytes(ext_id, b"not-a-crx")

    unsafe_zip = io.BytesIO()
    with zipfile.ZipFile(unsafe_zip, "w") as archive:
        archive.writestr("../outside.txt", "nope")
        archive.writestr("manifest.json", json.dumps({"name": "unsafe", "version": "1"}))
    with pytest.raises(ValueError, match="unsafe path"):
        mgr.install_from_bytes(ext_id, b"Cr24" + b"\0" * 16 + unsafe_zip.getvalue())

    assert mgr.list_installed_extensions() == []


def test_reinstall_updates_the_durable_registry_entry(tmp_path):
    mgr = ChromeExtensionManager(storage_dir=tmp_path)
    ext_id = "cjpalhdlnbpafiamejdnhcphjbkeiagm"
    mgr.install_from_bytes(ext_id, create_mock_crx({"name": "Fixture", "version": "1.0.0"}))
    updated = mgr.install_from_bytes(ext_id, create_mock_crx({"name": "Fixture", "version": "2.0.0"}))
    assert updated["version"] == "2.0.0"
    assert mgr.list_installed_extensions() == [updated]
