from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
import pytest

import workstation.extensions as extension_module
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


def test_update_rollback_preserves_last_known_good_after_candidate_verification_failure(tmp_path):
    ext_id = "cjpalhdlnbpafiamejdnhcphjbkeiagm"
    mgr = ChromeExtensionManager(storage_dir=tmp_path)
    v1 = mgr.install_from_bytes(ext_id, create_mock_crx({"name": "Fixture", "version": "1.0.0"}))

    candidate = mgr.prepare_install_from_bytes(ext_id, create_mock_crx({"name": "Fixture", "version": "2.0.0"}))
    assert candidate["version"] == "2.0.0"
    assert json.loads((tmp_path / ext_id / "manifest.json").read_text())["version"] == "2.0.0"
    mgr.rollback_update(ext_id)

    restored = ChromeExtensionManager(storage_dir=tmp_path).list_installed_extensions()
    assert restored == [v1]
    assert json.loads((tmp_path / ext_id / "manifest.json").read_text())["version"] == "1.0.0"


def test_update_replace_failure_restores_last_known_good(tmp_path, monkeypatch):
    ext_id = "cjpalhdlnbpafiamejdnhcphjbkeiagm"
    mgr = ChromeExtensionManager(storage_dir=tmp_path)
    v1 = mgr.install_from_bytes(ext_id, create_mock_crx({"name": "Fixture", "version": "1.0.0"}))
    real_replace = extension_module.os.replace
    failed = False

    def fail_candidate_promotion(source, destination):
        nonlocal failed
        if Path(source).name == "extension" and Path(destination).name == ext_id and not failed:
            failed = True
            raise OSError("simulated candidate replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(extension_module.os, "replace", fail_candidate_promotion)
    with pytest.raises(OSError, match="candidate replace"):
        mgr.install_from_bytes(ext_id, create_mock_crx({"name": "Fixture", "version": "2.0.0"}))

    assert failed is True
    assert ChromeExtensionManager(storage_dir=tmp_path).list_installed_extensions() == [v1]
    assert json.loads((tmp_path / ext_id / "manifest.json").read_text())["version"] == "1.0.0"


def test_update_staging_failure_keeps_last_known_good(tmp_path):
    ext_id = "cjpalhdlnbpafiamejdnhcphjbkeiagm"
    mgr = ChromeExtensionManager(storage_dir=tmp_path)
    v1 = mgr.install_from_bytes(ext_id, create_mock_crx({"name": "Fixture", "version": "1.0.0"}))

    with pytest.raises(ValueError, match="manifest"):
        mgr.install_from_bytes(ext_id, create_mock_crx({"name": "Invalid"}))

    assert mgr.list_installed_extensions() == [v1]
    assert json.loads((tmp_path / ext_id / "manifest.json").read_text())["version"] == "1.0.0"


def test_update_registry_interruption_recovers_last_known_good_on_restart(tmp_path, monkeypatch):
    ext_id = "cjpalhdlnbpafiamejdnhcphjbkeiagm"
    mgr = ChromeExtensionManager(storage_dir=tmp_path)
    v1 = mgr.install_from_bytes(ext_id, create_mock_crx({"name": "Fixture", "version": "1.0.0"}))
    real_save = mgr._save_registry
    save_count = 0

    def interrupt_registry_promotion(data):
        nonlocal save_count
        save_count += 1
        # The journal write succeeds; the post-promotion registry write and
        # same-process cleanup are interrupted, leaving startup recovery to
        # reconcile the durable journal and filesystem.
        if save_count in {2, 3}:
            raise OSError("simulated registry interruption")
        return real_save(data)

    monkeypatch.setattr(mgr, "_save_registry", interrupt_registry_promotion)
    with pytest.raises(OSError, match="registry interruption"):
        mgr.install_from_bytes(ext_id, create_mock_crx({"name": "Fixture", "version": "2.0.0"}))

    recovered = ChromeExtensionManager(storage_dir=tmp_path)
    assert recovered.list_installed_extensions() == [v1]
    assert json.loads((tmp_path / ext_id / "manifest.json").read_text())["version"] == "1.0.0"
    assert "_update" not in json.loads((tmp_path / "extensions.json").read_text())


def test_update_transaction_recovers_last_known_good_after_restart(tmp_path):
    ext_id = "cjpalhdlnbpafiamejdnhcphjbkeiagm"
    mgr = ChromeExtensionManager(storage_dir=tmp_path)
    v1 = mgr.install_from_bytes(ext_id, create_mock_crx({"name": "Fixture", "version": "1.0.0"}))
    mgr.prepare_install_from_bytes(ext_id, create_mock_crx({"name": "Fixture", "version": "2.0.0"}))

    recovered = ChromeExtensionManager(storage_dir=tmp_path)
    assert recovered.list_installed_extensions() == [v1]
    assert json.loads((tmp_path / ext_id / "manifest.json").read_text())["version"] == "1.0.0"
    assert "_update" not in json.loads((tmp_path / "extensions.json").read_text())


def test_verified_update_commits_and_survives_restart(tmp_path):
    ext_id = "cjpalhdlnbpafiamejdnhcphjbkeiagm"
    mgr = ChromeExtensionManager(storage_dir=tmp_path)
    mgr.install_from_bytes(ext_id, create_mock_crx({"name": "Fixture", "version": "1.0.0"}))
    v2 = mgr.prepare_install_from_bytes(ext_id, create_mock_crx({"name": "Fixture", "version": "2.0.0"}))
    mgr.commit_update(ext_id)

    assert ChromeExtensionManager(storage_dir=tmp_path).list_installed_extensions() == [v2]
    assert json.loads((tmp_path / ext_id / "manifest.json").read_text())["version"] == "2.0.0"


def test_committed_update_cleanup_interruption_never_repoints_registry_to_v1(tmp_path, monkeypatch):
    ext_id = "cjpalhdlnbpafiamejdnhcphjbkeiagm"
    mgr = ChromeExtensionManager(storage_dir=tmp_path)
    mgr.install_from_bytes(ext_id, create_mock_crx({"name": "Fixture", "version": "1.0.0"}))
    v2 = mgr.prepare_install_from_bytes(ext_id, create_mock_crx({"name": "Fixture", "version": "2.0.0"}))
    real_save = mgr._save_registry
    save_count = 0

    def fail_final_cleanup(data):
        nonlocal save_count
        save_count += 1
        if save_count == 2:
            raise OSError("simulated committed cleanup interruption")
        return real_save(data)

    monkeypatch.setattr(mgr, "_save_registry", fail_final_cleanup)
    with pytest.raises(OSError, match="committed cleanup"):
        mgr.commit_update(ext_id)

    # The caller may safely run its normal rollback handler after an exception
    # from commit cleanup.  A committed journal is terminal and must finish
    # cleanup instead of restoring v1 under a v2 directory.
    mgr.rollback_update(ext_id)
    recovered = ChromeExtensionManager(storage_dir=tmp_path)
    assert recovered.list_installed_extensions() == [v2]
    assert json.loads((tmp_path / ext_id / "manifest.json").read_text())["version"] == "2.0.0"
    assert "_update" not in json.loads((tmp_path / "extensions.json").read_text())


def test_agent_load_failure_rolls_back_real_manager_update(tmp_path, monkeypatch):
    """The Electron verification boundary must preserve the v1 install."""
    from tools import workstation_extensions as extension_tools

    ext_id = "cjpalhdlnbpafiamejdnhcphjbkeiagm"
    manager = ChromeExtensionManager(storage_dir=tmp_path)
    v1 = manager.install_from_bytes(ext_id, create_mock_crx({"name": "Fixture", "version": "1.0.0"}))
    v2_crx = create_mock_crx({"name": "Fixture", "version": "2.0.0"})

    class Journal:
        task_id = "extension-update-task"

        def __init__(self, *_args, **_kwargs):
            pass

        def record(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(extension_tools, "ChromeExtensionManager", lambda: manager)
    monkeypatch.setattr(extension_tools, "ExecutionJournal", Journal)
    monkeypatch.setattr(extension_tools, "_desktop_session", lambda: True)
    monkeypatch.setattr(manager, "download_crx", lambda _extension_id: v2_crx)
    monkeypatch.setattr(extension_tools, "_controller", lambda *_args, **_kwargs: {"loaded": False})

    result = json.loads(
        extension_tools._install(
            {"extension": ext_id},
            task_id="extension-update-task",
            session_id="extension-update-session",
        )
    )

    assert result["success"] is False
    assert result["code"] == "extension_load_failed"
    assert manager.list_installed_extensions() == [v1]
    assert json.loads((tmp_path / ext_id / "manifest.json").read_text())["version"] == "1.0.0"
    assert "_update" not in json.loads((tmp_path / "extensions.json").read_text())
