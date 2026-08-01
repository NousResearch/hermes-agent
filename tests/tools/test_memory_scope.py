"""Tests for scoped memory entries (issue #28279).

Scope markers ("[scope: telegram:123] content") restrict which sessions an
entry is injected into. Filtering is snapshot-side only: live state keeps all
entries so replace/remove work across scopes.
"""

import json
import pytest

from tools.memory_tool import (
    MemoryStore,
    memory_tool,
    parse_entry_scope,
    validate_scope,
    scope_matches,
    apply_scope_marker,
    resolve_scope,
    MEMORY_SCHEMA,
    ENTRY_DELIMITER,
)


# =========================================================================
# Scope primitives
# =========================================================================

class TestParseEntryScope:
    def test_unscoped_entry(self):
        assert parse_entry_scope("plain fact") == ("", "plain fact")

    def test_scoped_entry(self):
        scope, content = parse_entry_scope("[scope: telegram:123] DB host: 10.0.1.50")
        assert scope == "telegram:123"
        assert content == "DB host: 10.0.1.50"

    def test_wildcard_scope(self):
        scope, content = parse_entry_scope("[scope: discord:*] server rules apply")
        assert scope == "discord:*"
        assert content == "server rules apply"

    def test_profile_scope(self):
        scope, _ = parse_entry_scope("[scope: profile:work] VPN config at ~/vpn")
        assert scope == "profile:work"

    def test_scope_marker_mid_entry_is_not_a_marker(self):
        scope, content = parse_entry_scope("note about [scope: telegram:1] syntax")
        assert scope == ""
        assert content == "note about [scope: telegram:1] syntax"

    def test_multiline_entry(self):
        entry = "[scope: telegram:99] line one\nline two"
        scope, content = parse_entry_scope(entry)
        assert scope == "telegram:99"
        assert content == "line one\nline two"


class TestValidateScope:
    @pytest.mark.parametrize("scope", [
        "telegram:123456789", "discord:999", "telegram:*",
        "profile:work", "cli", "matrix:room-1.example_a",
    ])
    def test_valid(self, scope):
        assert validate_scope(scope) is None

    @pytest.mark.parametrize("scope", [
        "telegram:123]extra", "has space:1", "a:b:c:d]", "[nested]", "",
    ])
    def test_invalid(self, scope):
        assert validate_scope(scope) is not None


class TestScopeMatches:
    SESSION = ["telegram", "telegram:123", "profile:work"]

    def test_global_matches_everywhere(self):
        assert scope_matches("", self.SESSION)
        assert scope_matches("", [])

    def test_exact_chat_match(self):
        assert scope_matches("telegram:123", self.SESSION)

    def test_other_chat_no_match(self):
        assert not scope_matches("telegram:456", self.SESSION)

    def test_other_platform_no_match(self):
        assert not scope_matches("discord:123", self.SESSION)

    def test_platform_wildcard(self):
        assert scope_matches("telegram:*", self.SESSION)
        assert not scope_matches("discord:*", self.SESSION)

    def test_profile_match(self):
        assert scope_matches("profile:work", self.SESSION)
        assert not scope_matches("profile:home", self.SESSION)

    def test_none_session_scopes_disables_filtering(self):
        assert scope_matches("telegram:123", None)
        assert scope_matches("discord:456", None)


class TestApplyScopeMarker:
    def test_prepends(self):
        assert apply_scope_marker("fact", "telegram:1") == "[scope: telegram:1] fact"

    def test_no_scope_is_identity(self):
        assert apply_scope_marker("fact", None) == "fact"
        assert apply_scope_marker("fact", "") == "fact"

    def test_idempotent_same_scope(self):
        once = apply_scope_marker("fact", "telegram:1")
        assert apply_scope_marker(once, "telegram:1") == once

    def test_rescope_replaces_marker(self):
        once = apply_scope_marker("fact", "telegram:1")
        assert apply_scope_marker(once, "discord:2") == "[scope: discord:2] fact"


# =========================================================================
# Snapshot filtering
# =========================================================================

@pytest.fixture
def mem_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
    return tmp_path


def _write_entries(path, entries):
    path.write_text(ENTRY_DELIMITER.join(entries), encoding="utf-8")


class TestSnapshotScopeFiltering:
    ENTRIES = [
        "global fact",
        "[scope: telegram:123] private DM secret",
        "[scope: telegram:*] telegram-wide note",
        "[scope: discord:9] discord channel note",
        "[scope: profile:work] work-only fact",
    ]

    def _snapshot(self, store):
        return store.format_for_system_prompt("memory") or ""

    def test_matching_session_sees_scoped_entries_without_marker(self, mem_dir):
        _write_entries(mem_dir / "MEMORY.md", self.ENTRIES)
        store = MemoryStore(session_scopes=["telegram", "telegram:123", "profile:work"])
        store.load_from_disk()
        snap = self._snapshot(store)
        assert "global fact" in snap
        assert "private DM secret" in snap
        assert "telegram-wide note" in snap
        assert "work-only fact" in snap
        assert "discord channel note" not in snap
        assert "[scope:" not in snap  # markers stripped from prompt

    def test_other_chat_does_not_see_dm_secret(self, mem_dir):
        _write_entries(mem_dir / "MEMORY.md", self.ENTRIES)
        store = MemoryStore(session_scopes=["telegram", "telegram:999"])
        store.load_from_disk()
        snap = self._snapshot(store)
        assert "private DM secret" not in snap      # the issue's core scenario
        assert "telegram-wide note" in snap          # wildcard still matches
        assert "global fact" in snap

    def test_none_session_scopes_keeps_pre_scope_behaviour(self, mem_dir):
        _write_entries(mem_dir / "MEMORY.md", self.ENTRIES)
        store = MemoryStore(session_scopes=None)
        store.load_from_disk()
        snap = self._snapshot(store)
        # No filtering: every entry injected verbatim, markers intact.
        assert "private DM secret" in snap
        assert "discord channel note" in snap
        assert "[scope: telegram:123]" in snap

    def test_live_state_keeps_all_entries(self, mem_dir):
        _write_entries(mem_dir / "MEMORY.md", self.ENTRIES)
        store = MemoryStore(session_scopes=["cli"])
        store.load_from_disk()
        assert len(store.memory_entries) == len(self.ENTRIES)

    def test_scoped_entry_not_removable_from_non_matching_session(self, mem_dir):
        """Scope is an access boundary: a non-matching session can't target
        the entry, and the no-match error must not leak it either."""
        _write_entries(mem_dir / "MEMORY.md", self.ENTRIES)
        store = MemoryStore(session_scopes=["cli"])
        store.load_from_disk()
        result = json.loads(memory_tool(
            action="remove", target="memory",
            old_text="private DM secret", store=store,
        ))
        assert result["success"] is False
        # The error may echo the caller's own old_text, but the store
        # inventory it returns must not contain the scoped entry.
        assert not any("private DM secret" in e
                       for e in result.get("current_entries", []))
        # Entry untouched on disk.
        assert any("private DM secret" in e for e in store.memory_entries)

    def test_owning_session_can_remove_scoped_entry(self, mem_dir):
        _write_entries(mem_dir / "MEMORY.md", self.ENTRIES)
        store = MemoryStore(session_scopes=["telegram", "telegram:123"])
        store.load_from_disk()
        result = json.loads(memory_tool(
            action="remove", target="memory",
            old_text="private DM secret", store=store,
        ))
        assert result["success"] is True
        assert not any("private DM secret" in e for e in store.memory_entries)

    def test_user_target_also_filtered(self, mem_dir):
        _write_entries(mem_dir / "USER.md", [
            "likes short answers",
            "[scope: telegram:123] shares personal health details here",
        ])
        store = MemoryStore(session_scopes=["telegram", "telegram:999"])
        store.load_from_disk()
        snap = store.format_for_system_prompt("user") or ""
        assert "likes short answers" in snap
        assert "health details" not in snap


# =========================================================================
# Tool entry point
# =========================================================================

class TestMemoryToolScope:
    def test_add_with_scope_stores_marker(self, mem_dir):
        store = MemoryStore(session_scopes=["telegram", "telegram:123"])
        store.load_from_disk()
        result = json.loads(memory_tool(
            action="add", target="memory",
            content="DB host: 10.0.1.50", scope="telegram:123", store=store,
        ))
        assert result["success"] is True
        assert "[scope: telegram:123] DB host: 10.0.1.50" in store.memory_entries

    def test_add_with_invalid_scope_rejected(self, mem_dir):
        store = MemoryStore()
        store.load_from_disk()
        result = json.loads(memory_tool(
            action="add", target="memory",
            content="x", scope="bad scope]", store=store,
        ))
        assert result["success"] is False
        assert "Invalid scope" in result["error"]
        assert store.memory_entries == []

    def test_add_without_scope_unchanged(self, mem_dir):
        store = MemoryStore()
        store.load_from_disk()
        result = json.loads(memory_tool(
            action="add", target="memory", content="plain fact", store=store,
        ))
        assert result["success"] is True
        assert store.memory_entries == ["plain fact"]

    def test_batch_op_scope(self, mem_dir):
        store = MemoryStore()
        store.load_from_disk()
        result = json.loads(memory_tool(
            target="memory",
            operations=[
                {"action": "add", "content": "global note"},
                {"action": "add", "content": "chat secret", "scope": "telegram:42"},
            ],
            store=store,
        ))
        assert result["success"] is True
        assert "global note" in store.memory_entries
        assert "[scope: telegram:42] chat secret" in store.memory_entries

    def test_batch_invalid_scope_rejects_whole_batch(self, mem_dir):
        store = MemoryStore()
        store.load_from_disk()
        result = json.loads(memory_tool(
            target="memory",
            operations=[
                {"action": "add", "content": "ok"},
                {"action": "add", "content": "bad", "scope": "no way]"},
            ],
            store=store,
        ))
        assert result["success"] is False
        assert store.memory_entries == []

    def test_schema_documents_scope(self):
        assert "scope" in MEMORY_SCHEMA["parameters"]["properties"]
        batch_props = MEMORY_SCHEMA["parameters"]["properties"]["operations"]["items"]["properties"]
        assert "scope" in batch_props

    def test_schema_recommends_current_not_raw_ids(self):
        """The model is never shown raw chat IDs, so the schema must steer it
        to the server-resolved 'current' token (review feedback on #71224)."""
        desc = MEMORY_SCHEMA["parameters"]["properties"]["scope"]["description"]
        assert "current" in desc
        assert "server-side" in desc


# =========================================================================
# 'current' scope resolution (server-side, review feedback on #71224)
# =========================================================================

class TestResolveScope:
    def test_explicit_scope_passthrough(self):
        assert resolve_scope("telegram:99", ["cli"]) == ("telegram:99", None)

    def test_explicit_invalid_scope_errors(self):
        scope, err = resolve_scope("bad scope]", ["cli"])
        assert err is not None

    def test_current_resolves_to_chat_scope(self):
        scope, err = resolve_scope(
            "current", ["telegram", "telegram:123456789", "profile:work"])
        assert err is None
        assert scope == "telegram:123456789"

    def test_current_without_chat_id_falls_back_to_platform_wildcard(self):
        scope, err = resolve_scope("current", ["telegram", "profile:work"])
        assert err is None
        assert scope == "telegram:*"

    def test_current_with_no_identity_errors(self):
        scope, err = resolve_scope("current", None)
        assert err is not None
        scope, err = resolve_scope("current", [])
        assert err is not None

    def test_current_ignores_profile_scopes(self):
        scope, err = resolve_scope("current", ["profile:work"])
        assert err is not None


class TestCurrentScopeEndToEnd:
    def test_add_with_current_scope(self, mem_dir):
        store = MemoryStore(session_scopes=["telegram", "telegram:123"])
        store.load_from_disk()
        result = json.loads(memory_tool(
            action="add", target="memory",
            content="DB host: 10.0.1.50", scope="current", store=store,
        ))
        assert result["success"] is True
        assert "[scope: telegram:123] DB host: 10.0.1.50" in store.memory_entries

    def test_add_current_in_cli_session_errors_cleanly(self, mem_dir):
        store = MemoryStore(session_scopes=["cli"])
        store.load_from_disk()
        result = json.loads(memory_tool(
            action="add", target="memory",
            content="x", scope="current", store=store,
        ))
        # cli has no chat_id → falls back to cli:* wildcard
        assert result["success"] is True
        assert "[scope: cli:*] x" in store.memory_entries

    def test_batch_current_scope(self, mem_dir):
        store = MemoryStore(session_scopes=["discord", "discord:777"])
        store.load_from_disk()
        result = json.loads(memory_tool(
            target="memory",
            operations=[{"action": "add", "content": "secret", "scope": "current"}],
            store=store,
        ))
        assert result["success"] is True
        assert "[scope: discord:777] secret" in store.memory_entries


# =========================================================================
# Error-path leak prevention (review feedback on #71224)
# =========================================================================

class TestErrorPathsDoNotLeakScopedEntries:
    """Scope is an ACCESS BOUNDARY: no tool result reaching a non-matching
    session may contain a scoped entry's text — including every error path
    that returns current_entries or match previews."""

    SECRET = "private DM secret token XYZZY"
    ENTRIES = [
        "global fact",
        f"[scope: telegram:123] {SECRET}",
    ]

    @pytest.fixture
    def other_chat_store(self, mem_dir):
        _write_entries(mem_dir / "MEMORY.md", self.ENTRIES)
        store = MemoryStore(session_scopes=["telegram", "telegram:999"])
        store.load_from_disk()
        return store

    def _assert_no_leak(self, result):
        assert self.SECRET not in json.dumps(result, ensure_ascii=False)

    def test_missing_old_text_error(self, other_chat_store):
        result = json.loads(memory_tool(
            action="remove", target="memory", store=other_chat_store,
        ))
        assert result["success"] is False
        self._assert_no_leak(result)
        assert "global fact" in result["current_entries"]

    def test_replace_no_match_error(self, other_chat_store):
        result = json.loads(memory_tool(
            action="replace", target="memory",
            old_text="nonexistent", content="new", store=other_chat_store,
        ))
        assert result["success"] is False
        self._assert_no_leak(result)

    def test_remove_no_match_error(self, other_chat_store):
        result = json.loads(memory_tool(
            action="remove", target="memory",
            old_text="nonexistent", store=other_chat_store,
        ))
        assert result["success"] is False
        self._assert_no_leak(result)

    def test_replace_cannot_target_invisible_entry(self, other_chat_store):
        """Substring-matching an invisible entry must behave as no-match."""
        result = json.loads(memory_tool(
            action="replace", target="memory",
            old_text="XYZZY", content="stolen", store=other_chat_store,
        ))
        assert result["success"] is False
        self._assert_no_leak(result)

    def test_add_capacity_error(self, mem_dir):
        _write_entries(mem_dir / "MEMORY.md", self.ENTRIES)
        store = MemoryStore(memory_char_limit=80,
                            session_scopes=["telegram", "telegram:999"])
        store.load_from_disk()
        result = json.loads(memory_tool(
            action="add", target="memory",
            content="y" * 100, store=store,
        ))
        assert result["success"] is False
        self._assert_no_leak(result)

    def test_batch_error_inventory(self, other_chat_store):
        result = json.loads(memory_tool(
            target="memory",
            operations=[{"action": "remove", "old_text": "nonexistent"}],
            store=other_chat_store,
        ))
        assert result["success"] is False
        self._assert_no_leak(result)

    def test_batch_cannot_target_invisible_entry(self, other_chat_store):
        result = json.loads(memory_tool(
            target="memory",
            operations=[{"action": "remove", "old_text": "XYZZY"}],
            store=other_chat_store,
        ))
        assert result["success"] is False
        self._assert_no_leak(result)

    def test_owning_chat_still_sees_its_entry_in_errors(self, mem_dir):
        """The boundary must not break the owning session's consolidation flow."""
        _write_entries(mem_dir / "MEMORY.md", self.ENTRIES)
        store = MemoryStore(session_scopes=["telegram", "telegram:123"])
        store.load_from_disk()
        result = json.loads(memory_tool(
            action="remove", target="memory", store=store,
        ))
        assert result["success"] is False
        assert self.SECRET in json.dumps(result, ensure_ascii=False)
