"""Gateway ``plugins.manage`` — manifest ``config_schema`` rendered as settings fields (#46600, #87934).

Drives the real discovery + config writer against a temp HERMES_HOME: ``list`` carries each user
plugin's schema with the current values, and ``settings`` writes through the same
``plugins.entries.<id>.settings`` namespace ``ctx.get_config`` reads — never a secret into config.yaml.
"""

import pytest

from tui_gateway import server

MANIFEST = """\
name: demo-plugin
version: 1.0.0
config_schema:
  api_url: {type: str, default: "https://example.invalid", description: "Service endpoint"}
  retries: {type: int, default: 3}
  verbose: {type: bool, default: false}
  mode: {type: str, choices: [fast, careful], default: fast}
  api_key: {type: secret, description: "Token"}
"""


@pytest.fixture
def plugins_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    (home / "plugins" / "demo-plugin").mkdir(parents=True)
    (home / "plugins" / "demo-plugin" / "plugin.yaml").write_text(MANIFEST, encoding="utf-8")
    (home / "config.yaml").write_text(
        "plugins:\n  entries:\n    demo-plugin:\n      settings:\n        retries: 7\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _manage(**params):
    return server.handle_request({"id": "1", "method": "plugins.manage", "params": params})


def test_list_carries_schema_fields_with_current_values_and_no_secret_values(plugins_home, monkeypatch):
    monkeypatch.setenv("DEMO_PLUGIN_API_KEY", "shh")

    rows = _manage(action="list")["result"]["plugins"]
    row = next(r for r in rows if r["key"] == "demo-plugin")
    fields = {f["key"]: f for f in row["settings_schema"]}

    assert fields["api_url"]["type"] == "string" and fields["api_url"]["value"] == "https://example.invalid"
    assert fields["retries"]["type"] == "number" and fields["retries"]["value"] == 7  # config.yaml wins over default
    assert fields["verbose"]["type"] == "boolean" and fields["verbose"]["value"] is False
    assert fields["mode"]["type"] == "enum" and fields["mode"]["choices"] == ["fast", "careful"]
    assert fields["api_key"] == {"key": "api_key", "type": "secret", "label": "api_key", "description": "Token",
                                 "required": False, "env": "DEMO_PLUGIN_API_KEY", "has_value": True}


def test_settings_writes_the_plugin_namespace_and_refuses_secrets_and_bad_types(plugins_home):
    resp = _manage(action="settings", key="demo-plugin", values={"api_url": "https://real.invalid", "retries": 2, "mode": "careful"})

    assert resp["result"]["ok"] is True and sorted(resp["result"]["written"]) == ["api_url", "mode", "retries"]
    from hermes_cli.config import load_config_readonly
    assert load_config_readonly()["plugins"]["entries"]["demo-plugin"]["settings"] == {
        "api_url": "https://real.invalid", "retries": 2, "mode": "careful"}
    refreshed = {f["key"]: f["value"] for f in resp["result"]["plugin"]["settings_schema"] if "value" in f}
    assert refreshed["retries"] == 2 and refreshed["mode"] == "careful"

    for values in ({"api_key": "leak"}, {"retries": "two"}, {"mode": "reckless"}, {"unknown": 1}):
        assert _manage(action="settings", key="demo-plugin", values=values)["error"]["code"] == 4021
    assert "api_key" not in (plugins_home / "config.yaml").read_text(encoding="utf-8-sig")


PORTABLE_MANIFEST = """\
{
  "$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json",
  "name": "portable-demo",
  "version": "1.0.0",
  "extensions": {"com.nousresearch.hermes": {"config_schema": {
    "show_chip": {"type": "bool", "default": true},
    "api_key": {"type": "secret", "env": "PORTABLE_DEMO_KEY"}
  }}}
}
"""


def test_portable_plugin_json_schema_lists_and_saves_like_a_native_one(plugins_home, monkeypatch):
    """Agent Plugins v1 forbids unknown top-level fields, so a portable package declares its form under
    ``extensions.com.nousresearch.hermes.config_schema``; it must get the same list + save path, and a
    save tells the plugin's Desktop half (``plugin.<id>.settings.changed``) so it applies without polling."""
    sent = []
    monkeypatch.setattr(server, "_broadcast_global_event", lambda name, payload: sent.append((name, payload)))
    (plugins_home / "plugins" / "portable-demo").mkdir()
    (plugins_home / "plugins" / "portable-demo" / "plugin.json").write_text(PORTABLE_MANIFEST, encoding="utf-8")

    rows = _manage(action="list")["result"]["plugins"]
    fields = {f["key"]: f for f in next(r for r in rows if r["key"] == "portable-demo")["settings_schema"]}
    assert fields["show_chip"]["type"] == "boolean" and fields["show_chip"]["value"] is True
    assert fields["api_key"]["type"] == "secret" and fields["api_key"]["env"] == "PORTABLE_DEMO_KEY"

    resp = _manage(action="settings", key="portable-demo", values={"show_chip": False})
    assert resp["result"]["written"] == ["show_chip"]
    from hermes_cli.config import load_config_readonly
    assert load_config_readonly()["plugins"]["entries"]["portable-demo"]["settings"] == {"show_chip": False}
    assert sent == [("plugin.portable-demo.settings.changed", {"written": ["show_chip"]})]
    assert _manage(action="settings", key="portable-demo", values={"api_key": "leak"})["error"]["code"] == 4021


def test_config_source_config_section_reads_and_writes_top_level_yaml_section(plugins_home):
    """Part 2 (#144485): ``config_source: {kind: config_section, path: myengine}`` binds the schema to
    that top-level ``config.yaml`` section (preserving YAML comments and sibling keys), including field
    names like ``model`` that are reserved only under ``plugins.entries.<id>.settings``."""
    plugin_dir = plugins_home / "plugins" / "myengine"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        "name: myengine\n"
        "version: 0.2.0\n"
        "config_source:\n"
        "  kind: config_section\n"
        "  path: myengine\n"
        "config_schema:\n"
        "  model: {type: str, default: gpt-4o-mini, description: Summary model}\n"
        "  threshold: {type: float, default: 0.75}\n"
        "  api_key: {type: secret, env: MYENGINE_API_KEY}\n",
        encoding="utf-8",
    )
    (plugins_home / "config.yaml").write_text(
        "# user top comment\n"
        "myengine:\n"
        "  threshold: 0.85\n"
        "  untouched_sibling: keep-me\n",
        encoding="utf-8",
    )

    row = next(r for r in _manage(action="list")["result"]["plugins"] if r["key"] == "myengine")
    assert row["config_source"] == {"kind": "config_section", "path": "myengine", "editable": True}
    fields = {f["key"]: f for f in row["settings_schema"]}
    assert fields["model"]["value"] == "gpt-4o-mini"
    assert fields["threshold"]["value"] == 0.85
    assert fields["api_key"]["type"] == "secret" and fields["api_key"]["env"] == "MYENGINE_API_KEY"

    resp = _manage(action="settings", key="myengine", values={"model": "claude-haiku", "threshold": 0.6})
    assert resp["result"]["ok"] is True and sorted(resp["result"]["written"]) == ["model", "threshold"]

    raw_text = (plugins_home / "config.yaml").read_text(encoding="utf-8-sig")
    assert "# user top comment" in raw_text
    from hermes_cli.config import load_config_readonly
    cfg = load_config_readonly()
    assert cfg["myengine"] == {
        "model": "claude-haiku",
        "threshold": 0.6,
        "untouched_sibling": "keep-me",
    }
    assert "myengine" not in (cfg.get("plugins", {}).get("entries") or {})


def test_config_source_json_file_reads_and_writes_profile_scoped_sidecar(plugins_home):
    """Part 2 (#144485): ``config_source: {kind: json_file, path: $HERMES_HOME/mymem.json}`` expands
    against the active profile's ``get_hermes_home()`` (never ``os.environ['HERMES_HOME']``) and preserves
    existing sibling keys in the JSON sidecar."""
    import json
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    secondary = plugins_home / "profiles" / "work"
    (secondary / "plugins" / "mymem").mkdir(parents=True)
    (secondary / "plugins" / "mymem" / "plugin.yaml").write_text(
        "name: mymem\n"
        "version: 1.2.0\n"
        "config_source:\n"
        "  kind: json_file\n"
        "  path: $HERMES_HOME/mymem.json\n"
        "config_schema:\n"
        "  workspace: {type: str, default: default}\n"
        "  base_url: {type: str, default: https://api.example.invalid}\n"
        "  token: {type: secret, env: MYMEM_TOKEN}\n",
        encoding="utf-8",
    )
    (secondary / "mymem.json").write_text(
        json.dumps({"workspace": "research-lab", "internal_peer_id": "peer-42"}),
        encoding="utf-8",
    )

    token = set_hermes_home_override(secondary)
    try:
        row = next(r for r in _manage(action="list")["result"]["plugins"] if r["key"] == "mymem")
        assert row["config_source"] == {"kind": "json_file", "path": "$HERMES_HOME/mymem.json", "editable": True}
        fields = {f["key"]: f for f in row["settings_schema"]}
        assert fields["workspace"]["value"] == "research-lab"
        assert fields["base_url"]["value"] == "https://api.example.invalid"

        resp = _manage(action="settings", key="mymem", values={"workspace": "prod-lab"})
        assert resp["result"]["ok"] is True
    finally:
        reset_hermes_home_override(token)

    assert not (plugins_home / "mymem.json").exists()
    saved = json.loads((secondary / "mymem.json").read_text(encoding="utf-8-sig"))
    assert saved == {"workspace": "prod-lab", "internal_peer_id": "peer-42"}


@pytest.mark.parametrize(
    ("bad_source", "setup_corrupt"),
    [
        ({"kind": "json_file", "path": "$HERMES_HOME/../escape.json"}, False),
        ({"kind": "json_file", "path": "/etc/outside.json"}, False),
        ({"kind": "json_file", "path": "$HERMES_HOME/auth.json"}, False),
        ({"kind": "config_section", "path": "security"}, False),
        ({"kind": "toml_file", "path": "$HERMES_HOME/custom.toml"}, False),
        ({"kind": "json_file", "path": "$HERMES_HOME/corrupt.json"}, True),
    ],
)
def test_hard_rule_unresolvable_config_source_never_renders_editable_form_or_writes(
    plugins_home, bad_source, setup_corrupt
):
    """Hard rule (#144485): when ``config_source`` cannot be resolved safely, never fall back to
    ``plugins.entries.<id>.settings`` or render an editable form whose writes would be discarded."""
    import hermes_yaml as yaml

    plugin_dir = plugins_home / "plugins" / "unresolvable-plugin"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({
            "name": "unresolvable-plugin",
            "version": "1.0.0",
            "config_source": bad_source,
            "config_schema": {"endpoint": {"type": "str", "default": "https://x.invalid"}},
        }),
        encoding="utf-8",
    )
    if setup_corrupt:
        (plugins_home / "corrupt.json").write_text("{not valid json", encoding="utf-8")

    row = next(r for r in _manage(action="list")["result"]["plugins"] if r["key"] == "unresolvable-plugin")
    assert row["settings_schema"] == []
    assert row["config_source"]["editable"] is False
    assert row["config_source"]["path"] == bad_source["path"]

    resp = _manage(action="settings", key="unresolvable-plugin", values={"endpoint": "https://new.invalid"})
    assert resp["error"]["code"] == 4021
    if setup_corrupt:
        assert (plugins_home / "corrupt.json").read_text(encoding="utf-8-sig") == "{not valid json"


def test_provider_and_schemaless_plugins_surface_links_and_known_sources(plugins_home):
    """Parts 1 & 3 (#144485): provider plugins (memory, context engine, model provider) surface their
    dedicated settings link and known config file/section without executing user code."""
    # 1. Memory provider with config_schema.py
    mem_dir = plugins_home / "plugins" / "memside"
    mem_dir.mkdir(parents=True)
    (mem_dir / "plugin.yaml").write_text("name: memside\nversion: 1.0.0\n", encoding="utf-8")
    (mem_dir / "__init__.py").write_text(
        "from agent.memory_provider import MemoryProvider\n"
        "class MemSide(MemoryProvider):\n"
        "    name = 'memside'\n",
        encoding="utf-8",
    )
    (mem_dir / "config_schema.py").write_text(
        "from plugins.memory.config_schema import ProviderConfigSchema, ProviderField\n"
        "CONFIG_SCHEMA = ProviderConfigSchema(name='memside', label='MemSide', fields=(ProviderField(key='workspace', label='Workspace'),))\n",
        encoding="utf-8",
    )

    # 2. Context engine reading a top-level config.yaml section (no manifest config_schema)
    ctx_dir = plugins_home / "plugins" / "ctxengine"
    ctx_dir.mkdir(parents=True)
    (ctx_dir / "plugin.yaml").write_text("name: ctxengine\nversion: 0.4.0\n", encoding="utf-8")
    (ctx_dir / "__init__.py").write_text(
        "from agent.context_engine import ContextEngine\n"
        "from hermes_cli.config import load_config\n"
        "class Ctx(ContextEngine):\n"
        "    name = 'ctxengine'\n"
        "    def _cfg(self):\n"
        "        return load_config().get('ctxengine', {})\n",
        encoding="utf-8",
    )

    # 3. Model provider plugin
    prov_dir = plugins_home / "plugins" / "acme-llm"
    prov_dir.mkdir(parents=True)
    (prov_dir / "plugin.yaml").write_text(
        "name: acme-llm\nversion: 0.1.0\nkind: model-provider\nrequires_env:\n  - ACME_API_KEY\n",
        encoding="utf-8",
    )
    (prov_dir / "__init__.py").write_text("pass\n", encoding="utf-8")

    # 4. Pure hook plugin declaring zero settings
    hook_dir = plugins_home / "plugins" / "pure-hook"
    hook_dir.mkdir(parents=True)
    (hook_dir / "plugin.yaml").write_text("name: pure-hook\nversion: 1.0.0\n", encoding="utf-8")

    rows = {r["key"]: r for r in _manage(action="list")["result"]["plugins"]}

    assert rows["memside"]["config_source"] == {
        "kind": "json_file",
        "path": "$HERMES_HOME/memside/config.json",
        "editable": False,
        "settings_link": "memory",
    }
    assert rows["ctxengine"]["config_source"] == {
        "kind": "config_section",
        "path": "ctxengine",
        "editable": False,
        "settings_link": "context",
    }
    assert rows["acme-llm"]["config_source"] == {
        "kind": "env",
        "path": ".env (ACME_API_KEY)",
        "editable": False,
        "settings_link": "providers",
    }
    assert rows["pure-hook"]["settings_schema"] == []
    assert rows["pure-hook"].get("config_source") is None


def test_context_engine_active_status_and_managed_scope_refusals(plugins_home, monkeypatch):
    """Active ``context.engine`` plugins report ``enabled`` without a ``plugins.enabled`` entry, and
    ``config_section`` writes respect administrator-managed keys."""
    ctx_dir = plugins_home / "plugins" / "lcm"
    ctx_dir.mkdir(parents=True)
    (ctx_dir / "plugin.yaml").write_text(
        "name: lcm\n"
        "version: 0.3.0\n"
        "config_source:\n"
        "  kind: config_section\n"
        "  path: lcm\n"
        "config_schema:\n"
        "  summary_model: {type: str, default: gpt-4o-mini}\n",
        encoding="utf-8",
    )
    (plugins_home / "config.yaml").write_text("context:\n  engine: lcm\n", encoding="utf-8")

    row = next(r for r in _manage(action="list")["result"]["plugins"] if r["key"] == "lcm")
    assert row["status"] == "enabled"

    from hermes_cli import managed_scope
    monkeypatch.setattr(managed_scope, "is_key_managed", lambda dotted: dotted == "lcm.summary_model")
    denied = _manage(action="settings", key="lcm", values={"summary_model": "locked"})
    assert denied["error"]["code"] == 4021
    assert "administrator-managed" in denied["error"]["message"]


def test_memory_provider_declared_surface_falls_back_to_get_config_schema(plugins_home):
    """Part 3 (#144485): Desktop's ``getMemoryProviderConfig`` (``?surface=declared``) surfaces a memory
    provider's ``get_config_schema()`` fields when no separate ``config_schema.py`` exists."""
    import asyncio
    from hermes_cli.web_models import MemoryProviderConfigUpdate
    from hermes_cli.web_routers.memory_providers import get_memory_provider_config, update_memory_provider_config

    prov_dir = plugins_home / "plugins" / "legacymem"
    prov_dir.mkdir(parents=True)
    (prov_dir / "plugin.yaml").write_text("name: legacymem\nversion: 1.0.0\n", encoding="utf-8")
    (prov_dir / "__init__.py").write_text(
        "from agent.memory_provider import MemoryProvider\n"
        "class LegacyMem(MemoryProvider):\n"
        "    @property\n"
        "    def name(self):\n"
        "        return 'legacymem'\n"
        "    def is_available(self):\n"
        "        return True\n"
        "    def initialize(self, session_id, **kwargs):\n"
        "        pass\n"
        "    def get_tool_schemas(self):\n"
        "        return []\n"
        "    def get_config_schema(self):\n"
        "        return [\n"
        "            {'key': 'project', 'label': 'Project', 'default': 'main'},\n"
        "            {'key': 'api_key', 'label': 'API Key', 'secret': True, 'env_var': 'LEGACYMEM_API_KEY'},\n"
        "        ]\n"
        "def register(ctx):\n"
        "    ctx.register_memory_provider(LegacyMem())\n",
        encoding="utf-8",
    )

    payload = asyncio.run(get_memory_provider_config("legacymem", surface="declared"))
    by_key = {f["key"]: f for f in payload["fields"]}
    assert by_key["project"]["inline"] is True and by_key["project"]["value"] == "main"
    assert by_key["api_key"]["kind"] == "secret" and by_key["api_key"]["is_set"] is False

    res = asyncio.run(
        update_memory_provider_config(
            "legacymem",
            MemoryProviderConfigUpdate(values={"project": "alpha", "api_key": "lm-secret"}),
            surface="declared",
        )
    )
    assert res == {"ok": True}
    refreshed = asyncio.run(get_memory_provider_config("legacymem", surface="declared"))
    refreshed_by_key = {f["key"]: f for f in refreshed["fields"]}
    assert refreshed_by_key["project"]["value"] == "alpha"
    assert refreshed_by_key["api_key"]["is_set"] is True and refreshed_by_key["api_key"]["value"] == ""

