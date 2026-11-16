"""Delegation memory permissions follow the caller, never the routed profile."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_yaml as yaml
from agent.memory_provider import MemoryProvider
from hermes_constants import get_hermes_home
from run_agent import AIAgent
from tools.delegate_tool import delegate_task
from tools.delegate_tool_routes import TaskRoute


PROVIDER = '''\
import json
from pathlib import Path
from agent.memory_provider import MemoryProvider

class LookupProvider(MemoryProvider):
    @property
    def name(self):
        return "lookup_probe"

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self.home = Path(kwargs["hermes_home"])
        self.home.joinpath(session_id + ".identity.json").write_text(json.dumps({
            key: kwargs.get(key) for key in ("user_id", "user_id_alt")
        }))

    def get_tool_schemas(self):
        return [{"name": name, "description": name,
                 "parameters": {"type": "object", "properties": {}}}
                for name in ("lookup_search", "lookup_other", "lookup_save")]

    def read_only_tool_names(self):
        return ["lookup_search", "lookup_other"]

    def handle_tool_call(self, tool_name, args, **kwargs):
        return json.dumps({"lookup": tool_name})


def register(ctx):
    ctx.register_memory_provider(LookupProvider())
'''


def configure(home, *, memory_read=False, read_tools=None):
    home.mkdir(parents=True, exist_ok=True)
    plugin = home / "plugins" / "lookup_probe"
    plugin.mkdir(parents=True, exist_ok=True)
    (plugin / "__init__.py").write_text(PROVIDER, encoding="utf-8")
    (plugin / "plugin.yaml").write_text("name: lookup_probe\nversion: 1.0.0\n", encoding="utf-8")
    (home / "config.yaml").write_text(yaml.safe_dump({
        "memory": {"provider": "lookup_probe"},
        "delegation": {"memory_read": memory_read, "memory_read_tools": read_tools or [],
                       "max_spawn_depth": 2},
    }), encoding="utf-8")


def make_agent(**kwargs):
    return AIAgent(
        model="gpt-4.1-mini", provider="openrouter", api_key="test-key",
        base_url="http://127.0.0.1:9/v1", enabled_toolsets=["terminal", "memory", "delegation"],
        quiet_mode=True, skip_context_files=True, save_trajectories=False, **kwargs,
    )


@pytest.fixture
def memory_profiles(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home, home / "profiles" / "child"


def run_lookup_surface(agent, user_message, **kwargs):
    return {"final_response": json.dumps({
        "lookups": sorted(name for name in agent.valid_tool_names if name.startswith("lookup_")),
        "home": str(get_hermes_home()),
    }), "completed": True, "messages": []}


@pytest.mark.parametrize("parent_enabled,child_enabled", [(False, True), (True, False)])
def test_routed_child_uses_parent_memory_policy(memory_profiles, monkeypatch, parent_enabled, child_enabled):
    home, child_home = memory_profiles
    configure(home, memory_read=parent_enabled, read_tools=["lookup_search"])
    configure(child_home, memory_read=child_enabled, read_tools=["lookup_other"])
    parent = make_agent()
    monkeypatch.setattr(AIAgent, "run_conversation", run_lookup_surface)
    monkeypatch.setattr("tools.delegate_tool_routes.resolve_task_routes", lambda tasks, cfg, parent: ([TaskRoute(
        creds={"provider": "openrouter", "base_url": "http://127.0.0.1:9/v1", "api_key": "test-key",
               "api_mode": "chat_completions", "model": "gpt-4.1-mini"},
        routing_cfg={}, profile_home=child_home,
    )], None))
    try:
        result = json.loads(delegate_task(goal="Look up prior decisions", parent_agent=parent, background=False))
        assert result["results"][0]["status"] == "completed", result
        surface = json.loads(result["results"][0]["summary"])
        assert surface == {"lookups": ["lookup_search"] if parent_enabled else [], "home": str(child_home)}
    finally:
        parent.close()


@pytest.mark.parametrize("denied", ["memory", "all", "*"])
def test_explicit_memory_denial_survives_read_only_mode(memory_profiles, denied):
    home = memory_profiles[0]
    configure(home, memory_read=True)
    agent = make_agent(disabled_toolsets=[denied], memory_read_tools=["lookup_search"])
    try:
        assert "lookup_search" not in agent.valid_tool_names
    finally:
        agent.close()


def test_desktop_parent_grants_memory_across_child_platform_gates(memory_profiles, monkeypatch):
    home = memory_profiles[0]
    configure(home, memory_read=True, read_tools=["lookup_search"])
    parent = make_agent(platform="desktop")
    monkeypatch.setattr(AIAgent, "run_conversation", run_lookup_surface)
    try:
        assert "catalog" not in (parent.disabled_toolsets or ())
        assert "lookup_search" in parent.valid_tool_names
        result = json.loads(delegate_task(goal="Look up prior decisions", parent_agent=parent, background=False))
        assert result["results"][0]["status"] == "completed", result
        assert json.loads(result["results"][0]["summary"])["lookups"] == ["lookup_search"]
    finally:
        parent.close()


@pytest.mark.parametrize("child_policy", ["inherit", "narrow", "memory", "all", "*"])
def test_read_only_parent_passes_only_granted_lookups_to_grandchild(memory_profiles, monkeypatch, child_policy):
    home = memory_profiles[0]
    configure(home, memory_read=True, read_tools=["lookup_search"])
    parent = make_agent()
    surfaces = []

    def run_nested(agent, user_message, **kwargs):
        surfaces.append(set(agent.valid_tool_names))
        if len(surfaces) == 1:
            # A broader config cannot restore a lookup missing from this parent's grant.
            config = yaml.safe_load((home / "config.yaml").read_text())
            config["delegation"]["memory_read_tools"] = []
            (home / "config.yaml").write_text(yaml.safe_dump(config))
            if child_policy == "narrow":
                agent.enabled_toolsets = ["terminal", "delegation"]
            elif child_policy != "inherit":
                agent.disabled_toolsets = [child_policy]
            result = json.loads(delegate_task(goal="Look up a preference", parent_agent=agent, background=False))
            assert result["results"][0]["status"] == "completed", result
        return {"final_response": "Looked up preferences", "completed": True, "messages": []}

    monkeypatch.setattr(AIAgent, "run_conversation", run_nested)
    try:
        result = json.loads(delegate_task(goal="Coordinate a lookup", parent_agent=parent, background=False))
        assert result["results"][0]["status"] == "completed", result
        first, second = surfaces
        assert {"lookup_search", "delegate_task"} <= first
        assert {name for name in second if name.startswith("lookup_")} == (
            {"lookup_search"} if child_policy == "inherit" else set()
        )
        assert "delegate_task" not in second
    finally:
        parent.close()


def test_child_persists_memory_user_without_parent_gateway_routing(memory_profiles, monkeypatch):
    from hermes_state import SessionDB
    from openai.types.chat import ChatCompletion

    home = memory_profiles[0]
    configure(home, memory_read=True, read_tools=["lookup_search"])
    config = yaml.safe_load((home / "config.yaml").read_text())
    config["model"] = {"streaming": False, "context_length": 128000}
    (home / "config.yaml").write_text(yaml.safe_dump(config))
    db = SessionDB(db_path=home / "state.db")
    db.create_session(session_id="parent", source="telegram", user_id="alice",
                      session_key="telegram:group:room", chat_id="room", thread_id="topic")
    published = []

    def on_progress(event, *args, **payload):
        if event == "subagent.spawn_requested":
            published.append(payload["child_session_id"])

    def complete(*args, **kwargs):
        return ChatCompletion(
            id="response", created=0, model="gpt-4.1-mini", object="chat.completion",
            choices=[{"index": 0, "message": {"role": "assistant", "content": "Lookup complete"},
                      "finish_reason": "stop"}],
        )

    monkeypatch.setattr("openai.resources.chat.completions.Completions.create", complete)
    parent = make_agent(
        session_id="parent", session_db=db, tool_progress_callback=on_progress,
        user_id="alice", user_id_alt="alt-alice", user_name="Alice",
        gateway_session_key="telegram:group:room", chat_id="room", chat_name="Room",
        chat_type="group", thread_id="topic",
    )
    try:
        result = json.loads(delegate_task(goal="Look up a preference", parent_agent=parent, background=False))
        assert result["results"][0]["status"] == "completed", result
        child_id, = published
        row = db.get_session(child_id)
        assert row["source"] == "subagent"
        assert row["parent_session_id"] == parent.session_id
        assert row["message_count"] == 2
        identity = json.loads((home / f"{child_id}.identity.json").read_text())
        assert identity == {"user_id": "alice", "user_id_alt": "alt-alice"}
        assert {key: row[key] for key in ("session_key", "chat_id", "chat_type", "thread_id", "display_name")} == {
            "session_key": None, "chat_id": None, "chat_type": None, "thread_id": None, "display_name": None,
        }
        origin = json.loads(row["origin_json"])
        assert origin["user_id"] == "alice"
        assert origin["user_id_alt"] == "alt-alice"
        assert {key: origin[key] for key in ("chat_id", "thread_id", "user_name", "chat_name")} == {
            "chat_id": None, "thread_id": None, "user_name": None, "chat_name": None,
        }
        assert origin["chat_type"] != "group"
        assert db.get_session("parent")["session_key"] == "telegram:group:room"
    finally:
        parent.close()
        db.close()


class SingletonLookupProvider(MemoryProvider):
    name = "singleton_probe"

    def __init__(self):
        self.initialized_sessions = []
        self.shutdown_sessions = []
        self.session_id = None

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self.session_id = session_id
        self.initialized_sessions.append(session_id)

    def shutdown(self):
        self.shutdown_sessions.append(self.session_id)

    def get_tool_schemas(self):
        return [{"name": "singleton_search", "description": "Search memory",
                 "parameters": {"type": "object", "properties": {}}}]

    def read_only_tool_names(self):
        return ["singleton_search"]

    def handle_tool_call(self, tool_name, args, **kwargs):
        return json.dumps({"session_id": self.session_id})


def test_singleton_entry_point_is_not_reinitialized_or_closed_by_child(memory_profiles, monkeypatch, caplog):
    home = memory_profiles[0]
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(yaml.safe_dump({
        "memory": {"provider": "singleton_probe"}, "delegation": {"memory_read": True},
    }))
    provider = SingletonLookupProvider()
    entry_point = SimpleNamespace(name=provider.name, value="singleton_probe:provider", load=lambda: provider)
    monkeypatch.setattr("plugins.memory.find_provider_entry_point",
                        lambda name: entry_point if name == provider.name else None)

    def run_child(agent, user_message, **kwargs):
        return {"final_response": json.dumps(sorted(
            name for name in agent.valid_tool_names if name.startswith("singleton_")
        )), "completed": True, "messages": []}

    parent = make_agent()
    monkeypatch.setattr(AIAgent, "run_conversation", run_child)
    try:
        assert "singleton_search" in parent.valid_tool_names
        result = json.loads(delegate_task(goal="Look up prior decisions", parent_agent=parent, background=False))
        assert result["results"][0]["status"] == "completed", result
        assert (provider.initialized_sessions, provider.shutdown_sessions) == ([parent.session_id], [])
        assert json.loads(result["results"][0]["summary"]) == []
        assert json.loads(provider.handle_tool_call("singleton_search", {})) == {"session_id": parent.session_id}
        assert sum("memory_provider_instance_shared" in record.message for record in caplog.records) == 1
    finally:
        parent.close()
    assert provider.shutdown_sessions == [parent.session_id]


@pytest.mark.parametrize("configured", [
    "", " ", [""], [" "], ["lookup_search", ""], ["lookup_search", " "],
    ["lookup_search", None], ["lookup_search", 0], ["lookup_search", {}],
])
def test_blank_or_malformed_memory_allowlist_denies_child(memory_profiles, monkeypatch, configured):
    home = memory_profiles[0]
    configure(home, memory_read=True)
    config = yaml.safe_load((home / "config.yaml").read_text())
    config["delegation"]["memory_read_tools"] = configured
    (home / "config.yaml").write_text(yaml.safe_dump(config))
    parent = make_agent()
    monkeypatch.setattr(AIAgent, "run_conversation", run_lookup_surface)
    try:
        result = json.loads(delegate_task(goal="Look up prior decisions", parent_agent=parent, background=False))
        assert result["results"][0]["status"] == "completed", result
        assert json.loads(result["results"][0]["summary"])["lookups"] == []
    finally:
        parent.close()


@pytest.mark.parametrize("missing", [True, False])
def test_missing_or_empty_list_grants_provider_default(memory_profiles, monkeypatch, missing):
    home = memory_profiles[0]
    configure(home, memory_read=True)
    if missing:
        config = yaml.safe_load((home / "config.yaml").read_text())
        del config["delegation"]["memory_read_tools"]
        (home / "config.yaml").write_text(yaml.safe_dump(config))
    parent = make_agent()
    monkeypatch.setattr(AIAgent, "run_conversation", run_lookup_surface)
    try:
        result = json.loads(delegate_task(goal="Look up prior decisions", parent_agent=parent, background=False))
        assert result["results"][0]["status"] == "completed", result
        assert json.loads(result["results"][0]["summary"])["lookups"] == ["lookup_other", "lookup_search"]
    finally:
        parent.close()


def test_refresh_explicit_memory_denial_revokes_child_grant(memory_profiles, monkeypatch):
    from tools.mcp_tool_agent import refresh_agent_mcp_tools

    home = memory_profiles[0]
    configure(home, memory_read=True, read_tools=["lookup_search"])
    surfaces = []

    def run_child(agent, user_message, **kwargs):
        surfaces.append(set(agent.valid_tool_names))
        refresh_agent_mcp_tools(agent)
        surfaces.append(set(agent.valid_tool_names))
        refresh_agent_mcp_tools(agent, disabled_override=["memory"])
        surfaces.append(set(agent.valid_tool_names))
        refresh_agent_mcp_tools(agent)
        surfaces.append(set(agent.valid_tool_names))
        return {"final_response": "Checked memory policy", "completed": True, "messages": []}

    parent = make_agent()
    monkeypatch.setattr(AIAgent, "run_conversation", run_child)
    try:
        result = json.loads(delegate_task(goal="Look up prior decisions", parent_agent=parent, background=False))
        assert result["results"][0]["status"] == "completed", result
        assert ["lookup_search" in surface for surface in surfaces] == [True, True, False, False]
    finally:
        parent.close()


@pytest.mark.platforms("any")
@pytest.mark.parametrize("singleton", [True, False])
def test_hosted_memory_is_withheld_from_read_only_children(memory_profiles, monkeypatch, caplog, singleton):
    from hermes_cli.plugin_isolation import user_plugin_host

    home = memory_profiles[0]
    configure(home, memory_read=True, read_tools=["lookup_search"])
    plugin = home / "plugins" / "lookup_probe" / "__init__.py"
    plugin.write_text(PROVIDER + '''

class HostedLookupProvider(LookupProvider):
    def initialize(self, session_id, **kwargs):
        super().initialize(session_id, **kwargs)
        self.session_id = session_id
        with (self.home / "hosted-events.jsonl").open("a") as events:
            events.write(json.dumps(["initialize", session_id]) + "\\n")

    def shutdown(self):
        with (self.home / "hosted-events.jsonl").open("a") as events:
            events.write(json.dumps(["shutdown", self.session_id]) + "\\n")

singleton_provider = HostedLookupProvider()

def register(ctx):
    ctx.register_memory_provider(''' + ("singleton_provider" if singleton else "HostedLookupProvider()") + ''')
''', encoding="utf-8")
    config = yaml.safe_load((home / "config.yaml").read_text())
    config["plugins"] = {"isolation": "host"}
    (home / "config.yaml").write_text(yaml.safe_dump(config))
    events = home / "hosted-events.jsonl"
    parent = make_agent()
    monkeypatch.setattr(AIAgent, "run_conversation", run_lookup_surface)
    try:
        assert "lookup_search" in parent.valid_tool_names
        result = json.loads(delegate_task(goal="Look up prior decisions", parent_agent=parent, background=False))
        assert result["results"][0]["status"] == "completed", result
        assert json.loads(result["results"][0]["summary"])["lookups"] == []
        assert [json.loads(line) for line in events.read_text().splitlines()] == [
            ["initialize", parent.session_id],
        ]
        assert sum("memory_provider_hosted_read_only_unsupported" in record.message for record in caplog.records) == 1
    finally:
        parent.close()
        user_plugin_host().shutdown()
    assert [json.loads(line) for line in events.read_text().splitlines()] == [
        ["initialize", parent.session_id], ["shutdown", parent.session_id],
    ]
