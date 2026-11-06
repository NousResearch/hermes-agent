"""Per-user USER.md profiles (memory.per_user_profiles).

On a multi-user gateway the shared USER.md mixes every speaker's preferences and injects
them into everyone's system prompt. With the option on, each authenticated speaker gets
memories/users/<platform>-<user_id>/USER.md; local operator sessions and identity-less
sessions keep the shared file; MEMORY.md is always shared; staged approvals keep the scope.
"""

import json
from types import SimpleNamespace

import pytest

from tools.memory_tool import (
    MemoryStore, apply_memory_pending, load_on_disk_store, memory_tool, resolve_user_scope)
from tools.memory_tool_store import USER_SCOPE_RE
from tools.skill_provenance import reset_current_write_origin, set_current_write_origin

ON = {"per_user_profiles": True}


@pytest.fixture()
def mem_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    return tmp_path


class TestResolveUserScope:
    def test_off_by_default(self):
        assert resolve_user_scope("slack", "U016U059HPY", {}) is None
        assert resolve_user_scope("slack", "U016U059HPY", None) is None
        assert resolve_user_scope("slack", "U016U059HPY", {"per_user_profiles": False}) is None

    def test_safe_id_kept_verbatim_and_platform_qualified(self):
        assert resolve_user_scope("telegram", "123456789", ON) == "telegram-123456789"
        assert resolve_user_scope("discord", "123456789", ON) == "discord-123456789"

    def test_no_identity_and_local_operator_keep_shared_profile(self):
        for user_id in (None, "", "   ", "uid:1000", "uid:0"):
            assert resolve_user_scope("slack", user_id, ON) is None

    def test_case_variants_never_collide(self):
        upper = resolve_user_scope("slack", "U016U059HPY", ON)
        lower = resolve_user_scope("slack", "u016u059hpy", ON)
        assert upper != lower
        assert upper.startswith("slack-u016u059hpy-")  # readable slug + hash of the exact id

    @pytest.mark.parametrize("user_id", [
        "../../etc", "a/b", "a\\b", ".", "..", "C:\\x", "user name", "名前", "x" * 300, "api-peer"])
    def test_every_scope_is_one_safe_component(self, user_id):
        scope = resolve_user_scope("api_server", user_id, ON)
        assert USER_SCOPE_RE.fullmatch(scope), scope
        assert "/" not in scope and "\\" not in scope and not scope.startswith(".")

    def test_missing_platform_still_scopes(self):
        assert resolve_user_scope(None, "42", ON) == "unknown-42"


class TestScopedStore:
    def test_user_profile_is_per_scope_and_memory_is_shared(self, mem_dir):
        alice = MemoryStore(user_scope="slack-alice")
        bob = MemoryStore(user_scope="slack-bob")
        for store in (alice, bob):
            store.load_from_disk()
        alice.add("user", "Alice wants short answers")
        bob.add("user", "Bob wants tables")
        alice.add("memory", "the deploy runs at 03:00")

        assert (mem_dir / "users" / "slack-alice" / "USER.md").read_text(encoding="utf-8").strip() \
            == "Alice wants short answers"
        assert "Alice" not in (mem_dir / "users" / "slack-bob" / "USER.md").read_text(encoding="utf-8")
        assert not (mem_dir / "USER.md").exists() or not (mem_dir / "USER.md").read_text(encoding="utf-8").strip()

        bob_again = MemoryStore(user_scope="slack-bob")
        bob_again.load_from_disk()
        assert bob_again.user_entries == ["Bob wants tables"]
        assert bob_again.memory_entries == ["the deploy runs at 03:00"]

    def test_new_scoped_user_does_not_inherit_shared_profile(self, mem_dir):
        operator = MemoryStore()
        operator.load_from_disk()
        operator.add("user", "operator's private preference")
        newcomer = MemoryStore(user_scope="slack-newcomer")
        newcomer.load_from_disk()
        assert newcomer.user_entries == []
        assert "operator" not in (newcomer.format_for_system_prompt("user") or "")

    def test_unscoped_store_unchanged(self, mem_dir):
        store = MemoryStore()
        store.load_from_disk()
        store.add("user", "single-user setup")
        assert (mem_dir / "USER.md").read_text(encoding="utf-8").strip() == "single-user setup"
        assert not (mem_dir / "users").exists()

    @pytest.mark.parametrize("bad", ["../x", "a/b", "", "UPPER", ".hidden", "-lead", "x" * 200])
    def test_invalid_scope_rejected(self, bad):
        with pytest.raises(ValueError):
            MemoryStore(user_scope=bad)

    def test_load_on_disk_store_honours_scope(self, mem_dir):
        MemoryStore(user_scope="slack-carol").add("user", "Carol prefers email")
        assert load_on_disk_store(user_scope="slack-carol").user_entries == ["Carol prefers email"]
        assert load_on_disk_store().user_entries == []


class TestStagedWritesKeepTheirScope:
    def _stage_remove(self, store, old_text):
        token = set_current_write_origin("background_review")
        try:
            result = json.loads(memory_tool(action="remove", target="user", old_text=old_text, store=store))
        finally:
            reset_current_write_origin(token)
        assert result["staged"] is True
        from tools.write_approval import MEMORY, get_pending
        return get_pending(MEMORY, result["pending_id"])["payload"]

    def test_approval_by_unscoped_store_lands_in_writers_profile(self, mem_dir):
        operator = load_on_disk_store()
        operator.add("user", "shared: likes detail")
        dave = load_on_disk_store(user_scope="slack-dave")
        dave.add("user", "dave: hates detail")

        payload = self._stage_remove(dave, "hates detail")
        assert payload["user_scope"] == "slack-dave"

        result = apply_memory_pending(payload, load_on_disk_store())  # e.g. /memory approve on the CLI
        assert result["success"] is True
        assert load_on_disk_store(user_scope="slack-dave").user_entries == []
        assert load_on_disk_store().user_entries == ["shared: likes detail"]

    def test_unscoped_staging_carries_no_scope(self, mem_dir):
        store = load_on_disk_store()
        store.add("user", "only profile")
        assert "user_scope" not in self._stage_remove(store, "only profile")

    @pytest.mark.parametrize("tampered", ["../../escape", "a/b", 7, "UPPER"])
    def test_tampered_scope_refused(self, mem_dir, tampered):
        store = load_on_disk_store()
        store.add("user", "keep me")
        payload = {"action": "add", "target": "user", "content": "injected", "user_scope": tampered}
        result = apply_memory_pending(payload, store)
        assert result["success"] is False
        assert load_on_disk_store().user_entries == ["keep me"]
        assert not (mem_dir.parent / "escape").exists()


class TestAgentInitWiring:
    def _agent(self, user_id):
        return SimpleNamespace(enabled_toolsets=None, disabled_toolsets=None, _user_id=user_id)

    def _init(self, agent, platform, enabled):
        from agent.agent_init import _init_memory
        cfg = {"memory": {"per_user_profiles": enabled, "provider": ""}}
        _init_memory(agent, cfg, skip_memory=True, platform=platform)
        return agent._memory_store

    def test_gateway_user_gets_scoped_store(self, mem_dir):
        agent = self._agent("U016U059HPY")
        agent.enabled_toolsets = ["memory"]
        store = self._init(agent, "slack", True)
        assert store.user_scope == resolve_user_scope("slack", "U016U059HPY", ON)

    def test_option_off_or_operator_keeps_shared(self, mem_dir):
        for user_id, enabled in (("U016U059HPY", False), ("uid:1000", True), (None, True)):
            agent = self._agent(user_id)
            agent.enabled_toolsets = ["memory"]
            assert self._init(agent, "slack", enabled).user_scope is None
