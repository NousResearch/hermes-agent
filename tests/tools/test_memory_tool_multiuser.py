"""Per-user USER.md profile scoping (memory.per_user_profiles)."""
import json

import pytest

from tools import memory_tool
from tools.memory_tool import (
    MemoryStore,
    _isolated_id_scope,
    apply_memory_pending,
    resolve_user_scope,
)


@pytest.fixture
def mem_home(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_tool, "get_hermes_home", lambda: tmp_path)
    (tmp_path / "memories").mkdir(parents=True)
    return tmp_path


def _write(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(memory_tool.ENTRY_DELIMITER.join(entries), encoding="utf-8")


class TestResolveUserScope:
    def test_flag_off_returns_legacy(self, mem_home):
        assert resolve_user_scope("12345678", {}) == ""
        assert resolve_user_scope("12345678", {"per_user_profiles": False}) == ""

    def test_no_user_id_uses_default_key(self, mem_home):
        cfg = {"per_user_profiles": True, "default_user_key": "Owner"}
        assert resolve_user_scope("", cfg) == "owner"
        assert resolve_user_scope(None, cfg) == "owner"

    def test_no_user_id_no_default_is_legacy(self, mem_home):
        assert resolve_user_scope("", {"per_user_profiles": True}) == ""

    def test_registry_maps_id_to_key(self, mem_home):
        reg = mem_home / "data" / "users.json"
        reg.parent.mkdir(parents=True)
        reg.write_text(json.dumps(
            {"users": {"111": {"key": "alice"}, "222": {"key": "bob"}}}))
        cfg = {"per_user_profiles": True}
        assert resolve_user_scope("111", cfg) == "alice"
        assert resolve_user_scope("222", cfg) == "bob"

    def test_unlisted_id_gets_own_scope(self, mem_home):
        reg = mem_home / "data" / "users.json"
        reg.parent.mkdir(parents=True)
        reg.write_text(json.dumps({"users": {"111": {"key": "alice"}}}))
        cfg = {"per_user_profiles": True}
        assert resolve_user_scope("999", cfg) == "999"

    def test_missing_registry_falls_back_to_id(self, mem_home):
        cfg = {"per_user_profiles": True}
        assert resolve_user_scope("999", cfg) == "999"
        # A raw id that sanitization would alter gets a digest suffix so it
        # can never merge with its clean twin.
        hardened = resolve_user_scope("SomeUser", cfg)
        assert hardened.startswith("someuser-")
        assert hardened != resolve_user_scope("someuser", cfg)

    def test_custom_registry_path(self, mem_home, tmp_path):
        reg = tmp_path / "custom.json"
        reg.write_text(json.dumps({"users": {"7": {"key": "carol"}}}))
        cfg = {"per_user_profiles": True, "users_registry": str(reg)}
        assert resolve_user_scope("7", cfg) == "carol"


class TestScopedStore:
    def test_legacy_paths_without_scope(self, mem_home):
        store = MemoryStore()
        assert store._path_for("user") == mem_home / "memories" / "USER.md"
        assert store._path_for("memory") == mem_home / "memories" / "MEMORY.md"

    def test_scoped_user_path_shared_memory_path(self, mem_home):
        store = MemoryStore(user_scope="alice")
        assert store._path_for("user") == (
            mem_home / "memories" / "users" / "alice" / "USER.md")
        assert store._path_for("memory") == mem_home / "memories" / "MEMORY.md"

    def test_load_reads_scoped_profile(self, mem_home):
        _write(mem_home / "memories" / "USER.md", ["legacy profile entry"])
        _write(mem_home / "memories" / "users" / "alice" / "USER.md",
               ["alice profile entry"])
        _write(mem_home / "memories" / "MEMORY.md", ["shared fact"])

        legacy = MemoryStore()
        legacy.load_from_disk()
        assert legacy.user_entries == ["legacy profile entry"]

        alice = MemoryStore(user_scope="alice")
        alice.load_from_disk()
        assert alice.user_entries == ["alice profile entry"]
        assert alice.memory_entries == ["shared fact"]

    def test_save_creates_scoped_dir(self, mem_home):
        store = MemoryStore(user_scope="bob")
        store.load_from_disk()
        store.user_entries = ["bob likes tests"]
        store.save_to_disk("user")
        assert (mem_home / "memories" / "users" / "bob" / "USER.md").exists()
        # shared file untouched
        assert not (mem_home / "memories" / "USER.md").exists()

    def test_cross_scope_isolation(self, mem_home):
        a = MemoryStore(user_scope="alice")
        a.load_from_disk()
        a.user_entries = ["alice only"]
        a.save_to_disk("user")

        b = MemoryStore(user_scope="bob")
        b.load_from_disk()
        assert b.user_entries == []

    def test_scope_is_sanitized(self, mem_home):
        store = MemoryStore(user_scope="A/B..C!")
        assert store.user_scope == "abc"

    def test_resolved_scope_survives_store_resanitization(self, mem_home):
        # Digest-hardened scopes from _isolated_id_scope are clean slugs, so
        # the store's own (lossy) re-sanitization passes them through intact.
        scope = _isolated_id_scope("Weird/Id!", "telegram")
        assert MemoryStore(user_scope=scope).user_scope == scope


class TestIsolatedIdScopeCollisions:
    def test_dropped_chars_do_not_merge_ids(self):
        assert _isolated_id_scope("a/b") != _isolated_id_scope("ab")
        assert _isolated_id_scope("ab") == "ab"

    def test_all_symbol_ids_stay_distinct_and_non_empty(self):
        s1 = _isolated_id_scope("###")
        s2 = _isolated_id_scope("@@@")
        assert s1 and s2 and s1 != s2

    def test_case_variants_stay_distinct(self):
        # Some platforms use case-sensitive ids (e.g. YouTube channel ids).
        assert _isolated_id_scope("UCabc") != _isolated_id_scope("ucabc")

    def test_deterministic(self):
        assert _isolated_id_scope("Weird/Id!") == _isolated_id_scope("Weird/Id!")


class TestResolveUserScopePlatform:
    def test_unlisted_id_is_platform_qualified(self, mem_home):
        cfg = {"per_user_profiles": True}
        tg = resolve_user_scope("123", cfg, platform="telegram")
        dc = resolve_user_scope("123", cfg, platform="discord")
        assert tg == "telegram-123"
        assert dc == "discord-123"
        assert tg != dc

    def test_registry_platform_qualified_entry_wins(self, mem_home):
        reg = mem_home / "data" / "users.json"
        reg.parent.mkdir(parents=True)
        reg.write_text(json.dumps({"users": {
            "telegram:111": {"key": "alice"},
            "111": {"key": "generic"},
        }}))
        cfg = {"per_user_profiles": True}
        assert resolve_user_scope("111", cfg, platform="telegram") == "alice"
        # Other platforms and platform-less callers fall back to the bare id
        # entry, so existing registries keep working unchanged.
        assert resolve_user_scope("111", cfg, platform="discord") == "generic"
        assert resolve_user_scope("111", cfg) == "generic"


class TestApprovalScopeRoundTrip:
    """Write-approval staging must persist the user scope and approval replay
    must recover it — no matter which store the approver happens to hold."""

    @pytest.fixture
    def gate_on(self, monkeypatch):
        from tools import write_approval as wa
        monkeypatch.setattr(wa, "write_approval_enabled", lambda subsystem: True)
        return wa

    def test_staged_single_payload_carries_scope(self, mem_home, monkeypatch, gate_on):
        staged = {}
        real_stage = gate_on.stage_write

        def capture(subsystem, payload, **kwargs):
            staged.update(payload)
            return real_stage(subsystem, payload, **kwargs)

        monkeypatch.setattr(gate_on, "stage_write", capture)
        store = MemoryStore(user_scope="alice")
        out = json.loads(memory_tool.memory_tool(
            action="add", target="user", content="likes tea", store=store))
        assert out.get("staged") is True
        assert staged["user_scope"] == "alice"

    def test_staged_batch_payload_carries_scope(self, mem_home, monkeypatch, gate_on):
        staged = {}
        real_stage = gate_on.stage_write

        def capture(subsystem, payload, **kwargs):
            staged.update(payload)
            return real_stage(subsystem, payload, **kwargs)

        monkeypatch.setattr(gate_on, "stage_write", capture)
        store = MemoryStore(user_scope="bob")
        out = json.loads(memory_tool.memory_tool(
            target="user",
            operations=[{"action": "add", "content": "batch entry"}],
            store=store))
        assert out.get("staged") is True
        assert staged["user_scope"] == "bob"

    def test_apply_recovers_scope_over_unscoped_store(self, mem_home):
        # Gateway shape: approvals are applied against a fresh UNscoped store.
        payload = {"action": "add", "target": "user",
                   "content": "alice entry", "user_scope": "alice"}
        unscoped = MemoryStore()
        unscoped.load_from_disk()
        result = apply_memory_pending(payload, unscoped)
        assert result["success"] is True
        scoped_file = mem_home / "memories" / "users" / "alice" / "USER.md"
        assert "alice entry" in scoped_file.read_text(encoding="utf-8")
        assert not (mem_home / "memories" / "USER.md").exists()

    def test_apply_does_not_hijack_approver_scope(self, mem_home):
        # CLI shape: the approver's live store is scoped to the APPROVER —
        # the staged write must still land in the STAGER's profile.
        payload = {"action": "add", "target": "user",
                   "content": "for alice", "user_scope": "alice"}
        approver = MemoryStore(user_scope="bob")
        approver.load_from_disk()
        apply_memory_pending(payload, approver)
        assert (mem_home / "memories" / "users" / "alice" / "USER.md").exists()
        assert not (mem_home / "memories" / "users" / "bob" / "USER.md").exists()

    def test_legacy_payload_without_scope_stays_legacy(self, mem_home):
        # Records staged before scopes existed carry no user_scope key and
        # keep applying to the legacy shared file, even via a scoped store.
        payload = {"action": "add", "target": "user", "content": "old style"}
        approver = MemoryStore(user_scope="bob")
        approver.load_from_disk()
        apply_memory_pending(payload, approver)
        assert (mem_home / "memories" / "USER.md").exists()
        assert not (mem_home / "memories" / "users" / "bob" / "USER.md").exists()

    def test_batch_apply_recovers_scope(self, mem_home):
        payload = {"action": "batch", "target": "user", "user_scope": "carol",
                   "operations": [{"action": "add", "content": "batch fact"}]}
        unscoped = MemoryStore()
        unscoped.load_from_disk()
        result = apply_memory_pending(payload, unscoped)
        assert result["success"] is True
        scoped_file = mem_home / "memories" / "users" / "carol" / "USER.md"
        assert "batch fact" in scoped_file.read_text(encoding="utf-8")

    def test_gateway_approve_flow_end_to_end(self, mem_home, monkeypatch, gate_on):
        # Full stage → /memory approve round trip the way the gateway runs it
        # (gateway/slash_commands.py): staged by a scoped session, applied
        # against a fresh unscoped on-disk store.
        from hermes_cli.write_approval_commands import handle_pending_subcommand
        monkeypatch.setattr(gate_on, "get_hermes_home", lambda: mem_home)

        stager = MemoryStore(user_scope="alice")
        stager.load_from_disk()
        out = json.loads(memory_tool.memory_tool(
            action="add", target="user", content="tea, not coffee",
            store=stager))
        assert out.get("staged") is True

        gateway_store = memory_tool.load_on_disk_store()
        msg = handle_pending_subcommand(
            gate_on.MEMORY, ["approve", out["pending_id"]],
            memory_store=gateway_store)
        assert isinstance(msg, str)

        scoped = mem_home / "memories" / "users" / "alice" / "USER.md"
        assert "tea, not coffee" in scoped.read_text(encoding="utf-8")
        assert not (mem_home / "memories" / "USER.md").exists()
