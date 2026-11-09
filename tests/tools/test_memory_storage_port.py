"""The memory storage port as a plugin kind, and the local backend's pin.

``resolve_memory_storage(name)`` picks a backend by explicit name (no caller is wired to it yet):
``"local"`` is today's ``MEMORY.md`` / ``USER.md``; another name loads a provider through real
discovery (``$HERMES_HOME/plugins/<name>/``), and a missing one is an error, never a fallback to
the local files. The local backend serves only the local tenant and its own profile, holds the
agent level only, and keeps today's caps and paths.
"""
import textwrap

import pytest

from agent.runtime_scope import bind_profile_scope, bind_state
from hermes_memory_port import (
    MemoryLevelRefused, MemoryStorageProvider, MemoryStore, StoragePinRefused, resolve_memory_storage,
)
from hermes_memory_port_local import LocalMemoryStorageProvider
from hermes_scope import Group, Principal, Scope
from tools.memory_tool_store import ENTRY_DELIMITER

_PLUGIN = '''\
from hermes_memory_port import MemoryStorageProvider


class Probe(MemoryStorageProvider):
    @property
    def name(self):
        return "probe"

    def open(self, home=None):
        raise NotImplementedError

    def assert_isolating(self):
        raise RuntimeError("probe role bypasses row security")


def register(ctx):
    ctx.register_memory_storage(Probe())
'''


def test_local_names_the_local_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for name in ("local", ""):
        provider = resolve_memory_storage(name)
        assert isinstance(provider, LocalMemoryStorageProvider) and provider.levels == frozenset({"agent"})
        provider.assert_isolating()


def test_a_provider_that_does_not_check_isolation_cannot_be_built():
    class Unchecked(MemoryStorageProvider):
        @property
        def name(self):
            return "unchecked"

        def open(self, home=None):
            raise NotImplementedError

    with pytest.raises(TypeError, match="assert_isolating"):
        Unchecked()


def test_a_named_provider_loads_through_discovery_and_runs_its_isolation_check(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugin = tmp_path / "plugins" / "probe"
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text(textwrap.dedent(_PLUGIN))
    provider = resolve_memory_storage("probe")
    assert isinstance(provider, MemoryStorageProvider) and provider.name == "probe"
    with pytest.raises(RuntimeError, match="bypasses row security"):
        provider.assert_isolating()


def test_an_unavailable_named_provider_is_an_error_not_the_local_store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugin = tmp_path / "plugins" / "offline"
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text(textwrap.dedent(_PLUGIN).replace(
        "    def assert_isolating(self):", "    def is_available(self):\n        return False\n\n    def assert_isolating(self):"))
    with pytest.raises(LookupError, match="not available"):
        resolve_memory_storage("offline")


def test_a_missing_named_provider_is_an_error_not_the_local_store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with pytest.raises(LookupError, match="postgres"):
        resolve_memory_storage("postgres")


@pytest.fixture
def storage(tmp_path):
    opened = LocalMemoryStorageProvider().open(tmp_path)
    assert isinstance(opened.store, MemoryStore)
    return opened


def test_the_local_backend_is_todays_store(storage, tmp_path):
    """Writes through the port land in today's files, at today's paths, with today's delimiters
    and response dicts, and the frozen prompt block is the load-time snapshot."""
    with bind_profile_scope(tmp_path):
        assert storage.store.entries("memory") == []
        added = storage.store.add("memory", "prefers espresso")
        assert added["success"] is True and "usage" in added
        assert storage.store.add("user", "lives in Sydney")["success"] is True
        assert storage.store.system_prompt_block("memory") is None  # frozen before the write
        storage.store.reload()
        block = storage.store.system_prompt_block("memory")
        assert block and "prefers espresso" in block
    assert (tmp_path / "memories" / "MEMORY.md").read_text(encoding="utf-8") == "prefers espresso"
    assert (tmp_path / "memories" / "USER.md").read_text(encoding="utf-8") == "lives in Sydney"
    with bind_profile_scope(tmp_path):  # and today's store reads what the port wrote
        from tools.memory_tool import load_on_disk_store
        assert load_on_disk_store().memory_entries == ["prefers espresso"]


def test_the_local_backend_keeps_the_configured_caps(tmp_path):
    (tmp_path / "config.yaml").write_text("memory:\n  memory_char_limit: 500\n")
    opened = LocalMemoryStorageProvider().open(tmp_path)
    with bind_profile_scope(tmp_path):
        assert opened.store.char_limit("memory") == 500
        assert opened.store.add("memory", "x" * 500)["success"] is True
        refused = opened.store.add("memory", "y")
        assert refused["success"] is False and "chars" in refused["error"]
        assert opened.store.entries("memory") == ["x" * 500]


@pytest.mark.parametrize("pin", [
    # A hosted tenant naming this very path as its agent is still not this local store's.
    lambda home: Scope("acme", str(home), Principal("u"), owner=Group("support"), can_administer=False),
    lambda home: Scope("local", str(home.parent / "elsewhere"), Principal("owner")),
], ids=["another_tenant", "another_profile"])
def test_the_local_backend_refuses_any_pin_but_its_own_profile(storage, tmp_path, pin):
    with bind_profile_scope(tmp_path):
        storage.store.add("memory", "private")
    scope = pin(tmp_path)
    with bind_state(scope, hydrate_secrets=False):
        for call in (lambda: storage.store.entries("memory"),
                     lambda: storage.store.add("memory", "stolen"),
                     lambda: storage.store.reload()):
            with pytest.raises(StoragePinRefused):
                call()
    with bind_profile_scope(tmp_path):
        assert storage.store.entries("memory") == ["private"]


def test_with_no_scope_bound_the_pin_is_the_ambient_profile(storage, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert storage.store.add("memory", "ambient")["success"] is True  # a local caller with no scope
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "other"))
    with pytest.raises(StoragePinRefused):
        storage.store.entries("memory")


def test_a_bind_that_names_the_profile_without_moving_the_home_writes_nothing(storage, tmp_path, monkeypatch):
    """The store's files resolve through the ambient home, so a scope naming this profile while
    the ambient home is another profile's must refuse instead of redirecting the write."""
    from hermes_scope import LOCAL_OWNER, push_state
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "other"))
    scope = Scope("local", str(tmp_path), LOCAL_OWNER)
    with push_state(scope=scope):
        with pytest.raises(StoragePinRefused):
            storage.store.add("memory", "redirected")
    assert not (tmp_path / "memories" / "MEMORY.md").exists()
    assert not (tmp_path / "other" / "memories" / "MEMORY.md").exists()


@pytest.mark.parametrize("level", ["participant", "owner", "tenant"])
def test_a_local_store_holds_agent_level_records_only(storage, tmp_path, level):
    with bind_profile_scope(tmp_path):
        for target in ("memory", "user"):
            with pytest.raises(MemoryLevelRefused):
                storage.store.add(target, "fact", level=level)
            with pytest.raises(MemoryLevelRefused):
                storage.store.remove(target, "anything", level=level)
        assert storage.store.entries("memory") == []
        assert storage.store.entries("user") == []


def test_a_turns_scope_pins_the_store_and_its_writes_are_not_fenced(storage, tmp_path):
    """A turn writes memory under its TurnScope pin. Memory is the agent's own learning, not
    session-owned data, so no claimed lease or fence gates the write — a turn that lost its
    session lease still records what it learned."""
    from hermes_scope import Fence
    from agent.runtime_scope import bind_turn
    with bind_profile_scope(tmp_path):
        with bind_turn("s1", "t1", Fence(1, 1)):
            assert storage.store.add("memory", "learned mid-turn")["success"] is True
            assert storage.store.entries("memory") == ["learned mid-turn"]


def test_the_agent_notes_never_take_a_participant_record(storage, tmp_path):
    with bind_profile_scope(tmp_path):
        with pytest.raises(MemoryLevelRefused):
            storage.store.add("memory", "fact", level="participant")
