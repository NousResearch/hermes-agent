"""Publication ordering, profile identity and #90186 filesystem postconditions.

Permission and planted-symlink contracts adapted from enzo-adami's PR #90186.
"""
import json
import os
import stat
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from hermes_cli import config as config_module
from tools import approval_audit as audit


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("readonly", [False, True])
def test_stale_config_publisher_cannot_overwrite_newer_state(tmp_path, monkeypatch, initial, readonly):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(audit, "_settings", {})
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"approvals:\n  audit_log:\n    enabled: {str(initial).lower()}\n")
    paused = threading.Event()
    release_stale = threading.Event()
    real_lock = config_module._CONFIG_LOCK
    real_configure = audit.configure

    class OrderedLock:
        """Release the paused publisher when a competing loader hits its lock.

        With the fix, the competitor waits for the first publication. On the
        broken path, it publishes first and only then releases the stale one.
        Both schedules are event-driven; no sleep or negative timing assertion.
        """
        def __enter__(self):
            if not real_lock.acquire(blocking=False):
                release_stale.set()
                assert real_lock.acquire(timeout=10)
            return self

        def __exit__(self, *exc):
            real_lock.release()

    def ordered_publish(config):
        value = config["approvals"]["audit_log"]["enabled"]
        if value is initial:
            paused.set()
            assert release_stale.wait(timeout=10)
        real_configure(config)
        if value is not initial:
            release_stale.set()

    monkeypatch.setattr(config_module, "_CONFIG_LOCK", OrderedLock())
    monkeypatch.setattr(audit, "configure", ordered_publish)
    loaders = (config_module.load_config, config_module.load_config_readonly)
    with ThreadPoolExecutor(max_workers=2) as pool:
        stale = pool.submit(loaders[int(readonly)])
        try:
            assert paused.wait(timeout=10)
            config_path.write_text(f"approvals:\n  audit_log:\n    enabled: {str(not initial).lower()}\n")
            newer = pool.submit(loaders[int(not readonly)])
            assert newer.result(timeout=10)["approvals"]["audit_log"]["enabled"] is not initial
        finally:
            release_stale.set()
        assert stale.result(timeout=10)["approvals"]["audit_log"]["enabled"] is initial
    assert audit.is_enabled() is not initial
    audit.record_decision({"surface": "cli", "choice": "once"})
    assert (tmp_path / "logs" / "approvals.jsonl").exists() is not initial


@pytest.mark.parametrize("configure_alias", [False, True])
def test_profile_alias_and_resolved_path_share_enable_and_disable(tmp_path, monkeypatch, configure_alias):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    configured, observed = (alias, real) if configure_alias else (real, alias)
    monkeypatch.setattr(audit, "_settings", {})
    monkeypatch.setenv("HERMES_HOME", str(configured))
    (configured / "config.yaml").write_text("approvals:\n  audit_log:\n    enabled: true\n")
    config_module.load_config()
    monkeypatch.setenv("HERMES_HOME", str(observed))
    assert audit.is_enabled()
    audit.record_decision({"surface": "cli", "choice": "once"})
    path = real / "logs" / "approvals.jsonl"
    original = path.read_bytes()
    assert json.loads(original)["verdict"] == "approved"
    (observed / "config.yaml").write_text("approvals:\n  audit_log:\n    enabled: false\n")
    config_module.load_config_readonly()
    monkeypatch.setenv("HERMES_HOME", str(configured))
    assert not audit.is_enabled()
    audit.record_decision({"surface": "cli", "choice": "deny"})
    assert path.read_bytes() == original


def _private_file_postconditions(tmp_path, monkeypatch, existing):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(audit, "_settings", {})
    audit.configure({"approvals": {"audit_log": {"enabled": True}}})
    logs = tmp_path / "logs"
    files = (logs / "approvals.jsonl", logs / "approvals.jsonl.lock")
    if existing:
        logs.mkdir()
        for path in files:
            path.write_bytes(b"")
            path.chmod(0o666)
    audit.record_decision({"surface": "cli", "choice": "once"})
    assert json.loads(files[0].read_bytes())["verdict"] == "approved"
    for path in files:
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    if not existing:
        assert stat.S_IMODE(logs.stat().st_mode) == 0o700


@pytest.mark.linux_only
@pytest.mark.parametrize("existing", [False, True])
def test_linux_audit_files_are_private(tmp_path, monkeypatch, existing):
    _private_file_postconditions(tmp_path, monkeypatch, existing)


@pytest.mark.macos_only
@pytest.mark.parametrize("existing", [False, True])
def test_macos_audit_files_are_private(tmp_path, monkeypatch, existing):
    _private_file_postconditions(tmp_path, monkeypatch, existing)


@pytest.mark.parametrize("planted", ["approvals.jsonl", "approvals.jsonl.lock", "logs"])
def test_planted_symlink_never_touches_target(tmp_path, monkeypatch, caplog, planted):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(audit, "_settings", {})
    audit.configure({"approvals": {"audit_log": {"enabled": True}}})
    victim = tmp_path / "victim"
    logs = tmp_path / "logs"
    if planted == "logs":
        victim.mkdir()
        link = logs
        link.symlink_to(victim, target_is_directory=True)
    else:
        logs.mkdir()
        victim.write_bytes(b"")
        link = logs / planted
        link.symlink_to(victim)
    original_mode = stat.S_IMODE(victim.stat().st_mode)
    audit.record_decision({"surface": "cli", "choice": "once"})
    assert link.is_symlink()
    assert stat.S_IMODE(victim.stat().st_mode) == original_mode
    if planted == "logs":
        assert list(victim.iterdir()) == []
    else:
        assert victim.read_bytes() == b""
    assert "Approval audit write failed" in caplog.text
