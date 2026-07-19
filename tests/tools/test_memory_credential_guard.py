"""Tests for credential hardening in tools/memory_tool.py.

These reproduce the issue where prose-form credentials could be saved into
MEMORY.md / USER.md and injected into the model context, then verify the
defense-in-depth fix:

  1. Write path (add / replace / batch) rejects probable credentials and
     never echoes the secret.
  2. Legacy path: a credential already on disk is withheld from the
     system-prompt snapshot (replaced by a value-free marker) while the
     on-disk file is left untouched until the user removes it.
  3. Ordinary non-secret entries are untouched.
  4. The canary never appears in error responses, logs, or the snapshot.

All canary values are unmistakably fake.
"""

import logging

import pytest

from tools.memory_tool import MemoryStore

CANARY_PROSE = "Password for Test WebUI on this machine: CANARY_PASSWORD_7F39C1_A91D2E"
CANARY_ASSIGN = 'password = "CANARY_PASSWORD_7F39C1_A91D2E"'
CANARY_API_PROSE = "The api key is CANARY_APIKEY_ABC123DEF456GHI789"
CANARY_BEARER = "Authorization: Bearer CANARYeyJabc123def456ghi789"
CANARY_PRIVKEY = (
    "-----BEGIN RSA PRIVATE KEY-----\n"
    "CANARYMIIEogIBAAKCAQEAexamplekeyvalue1234567890\n"
    "-----END RSA PRIVATE KEY-----"
)
CANARY_CONNSTR = "postgres://admin:CANARY_Sup3rSecret@db.example.com:5432/app"


@pytest.fixture
def mem_store(tmp_path, monkeypatch):
    """A MemoryStore backed by an isolated temp HERMES_HOME/memories dir."""
    mem_dir = tmp_path / "memories"
    mem_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "tools.memory_tool.get_memory_dir",
        lambda: mem_dir,
    )
    # MemoryStore._path_for and load_from_disk use get_memory_dir via import;
    # patch the module-level reference used inside the class too.
    monkeypatch.setattr(MemoryStore, "_path_for", staticmethod(
        lambda target: mem_dir / ("USER.md" if target == "user" else "MEMORY.md")
    ))
    store = MemoryStore()
    store.load_from_disk()
    return store


def _write_legacy(mem_dir, fname, text):
    (mem_dir / fname).write_text(f"\n§\n{text}\n", encoding="utf-8")


class TestWritePathRejectsCredentials:
    def test_add_prose_password_rejected(self, mem_store):
        res = mem_store.add("memory", CANARY_PROSE)
        assert res["success"] is False
        assert "Credential" in res["error"] or "credential" in res["error"]
        assert "CANARY_PASSWORD_7F39C1_A91D2E" not in res["error"]

    def test_add_direct_assignment_rejected(self, mem_store):
        res = mem_store.add("memory", CANARY_ASSIGN)
        assert res["success"] is False
        assert "CANARY_PASSWORD_7F39C1_A91D2E" not in res["error"]

    def test_add_api_token_prose_rejected(self, mem_store):
        res = mem_store.add("memory", CANARY_API_PROSE)
        assert res["success"] is False
        assert "CANARY_APIKEY_ABC123DEF456GHI789" not in res["error"]

    def test_add_authorization_header_rejected(self, mem_store):
        res = mem_store.add("memory", CANARY_BEARER)
        assert res["success"] is False
        assert "CANARYeyJabc123def456ghi789" not in res["error"]

    def test_add_private_key_rejected(self, mem_store):
        res = mem_store.add("memory", CANARY_PRIVKEY)
        assert res["success"] is False
        assert "CANARYMIIEogIBAAKCAQEA" not in res["error"]

    def test_add_connection_string_rejected(self, mem_store):
        res = mem_store.add("memory", CANARY_CONNSTR)
        assert res["success"] is False
        assert "CANARY_Sup3rSecret" not in res["error"]

    def test_replace_with_credential_rejected(self, mem_store):
        mem_store.add("memory", "The deployment uses a webui.")
        res = mem_store.replace("memory", "The deployment uses a webui.", CANARY_PROSE)
        assert res["success"] is False
        assert "CANARY_PASSWORD_7F39C1_A91D2E" not in res["error"]

    def test_batch_with_credential_rejected(self, mem_store):
        res = mem_store.apply_batch("memory", [
            {"action": "add", "content": "A harmless note about the build."},
            {"action": "add", "content": CANARY_PROSE},
        ])
        # Whole batch must be refused; nothing committed.
        assert res["success"] is False
        assert "CANARY_PASSWORD_7F39C1_A91D2E" not in res["error"]


class TestLegacySnapshotWithholdsCredentials:
    def test_legacy_prose_password_absent_from_snapshot(self, mem_store, tmp_path):
        mem_dir = tmp_path / "memories"
        _write_legacy(mem_dir, "MEMORY.md", CANARY_PROSE)

        store = MemoryStore()
        store.load_from_disk()
        snap = store.format_for_system_prompt("memory") or ""

        # The canary must NOT reach the system prompt.
        assert "CANARY_PASSWORD_7F39C1_A91D2E" not in snap
        # A safe marker should be present instead.
        assert "[CREDENTIAL:" in snap
        # Original file on disk is preserved (user must remove + rotate).
        assert "CANARY_PASSWORD_7F39C1_A91D2E" in (
            mem_dir / "MEMORY.md"
        ).read_text(encoding="utf-8")


class TestNonSecretEntriesUnchanged:
    def test_ordinary_entry_accepted(self, mem_store):
        res = mem_store.add("memory", "User prefers concise responses.")
        assert res["success"] is True
        assert "User prefers concise responses." in (
            mem_store._path_for("memory").read_text(encoding="utf-8")
        )

    def test_security_advice_without_value_accepted(self, mem_store):
        res = mem_store.add(
            "memory",
            "Never store API keys in source control or in memory.",
        )
        assert res["success"] is True
        assert "Never store API keys in source control" in (
            mem_store._path_for("memory").read_text(encoding="utf-8")
        )

    def test_password_policy_advice_accepted(self, mem_store):
        res = mem_store.add(
            "memory",
            "User rotates passwords monthly for security hygiene.",
        )
        assert res["success"] is True


class TestNoSecretLeakInLogs:
    def test_write_rejection_does_not_log_secret(self, mem_store, tmp_path, caplog):
        with caplog.at_level(logging.WARNING):
            mem_store.add("memory", CANARY_PROSE)
        combined = caplog.text
        assert "CANARY_PASSWORD_7F39C1_A91D2E" not in combined

    def test_legacy_snapshot_does_not_log_secret(self, mem_store, tmp_path, caplog):
        mem_dir = tmp_path / "memories"
        _write_legacy(mem_dir, "MEMORY.md", CANARY_PROSE)
        with caplog.at_level(logging.WARNING):
            store = MemoryStore()
            store.load_from_disk()
        assert "CANARY_PASSWORD_7F39C1_A91D2E" not in caplog.text
