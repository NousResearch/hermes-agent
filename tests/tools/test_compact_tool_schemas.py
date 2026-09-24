"""Tests for the opt-in compact tool-schema profile.

The compact profile shrinks prose-heavy tool descriptions (and verbose
parameter descriptions) without touching JSON structure: parameter names,
types, enums, and required lists must be byte-identical between profiles so
tool-calling behaviour is unaffected.
"""

from __future__ import annotations

import json

import pytest
import yaml

from tools.registry import ToolRegistry, compact_schemas_enabled

# Tools that ship a compact variant.
COMPACT_TOOLS = {"clarify", "delegate_task", "skill_manage", "terminal", "memory"}

# Upper bound for a compact description. The target is 400 chars; terminal and
# delegate_task encode enough hard behavioural rules (tool-routing bans, the
# exit-code-masking pipe rule, self-report verification, leaf-child limits)
# that cutting them below this would drop a constraint rather than prose, so
# the assertion allows a small margin over the target.
MAX_COMPACT_DESCRIPTION_CHARS = 520

# Fraction of the telegram schema bytes the compact profile must save. The exact
# byte total tracks the live tool descriptions (upstream rewrites them), so the
# contract is proportional, not a frozen snapshot: PROSE heavy enough to matter.
MIN_TELEGRAM_BYTES_SAVED_FRACTION = 0.05


def _dummy_handler(args, **kwargs):
    return json.dumps({"ok": True})


@pytest.fixture
def cfg_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _write_cfg(home, cfg: dict) -> None:
    (home / "config.yaml").write_text(yaml.safe_dump(cfg))


def _set_compact(home, monkeypatch, enabled: bool) -> None:
    """Point config at a home whose agent.compact_tool_schemas is *enabled*."""
    _write_cfg(home, {"agent": {"compact_tool_schemas": enabled}})
    # The resolver memoizes on config mtime/size; clear any cached verdict.
    from tools import registry as registry_mod

    registry_mod._reset_compact_cache_for_tests()


def _telegram_definitions():
    """Tool definitions for the telegram toolset, as sent to the model."""
    from model_tools import get_tool_definitions

    return get_tool_definitions(enabled_toolsets=["hermes-telegram"], quiet_mode=False)


def _by_name(defs):
    return {d["function"]["name"]: d["function"] for d in defs}


def _schema_bytes(defs) -> int:
    return sum(len(json.dumps(d)) for d in defs)


def _structure(fn: dict) -> dict:
    """Everything about a schema EXCEPT prose, recursively.

    Returns a shape carrying param names, types, enums, required lists,
    defaults and limits -- the parts the model dispatches on. Descriptions
    are dropped, since those are exactly what the compact profile rewrites.
    """

    def strip(node):
        if isinstance(node, dict):
            return {
                k: strip(v)
                for k, v in sorted(node.items())
                if k != "description"
            }
        if isinstance(node, list):
            return [strip(v) for v in node]
        return node

    return {"name": fn["name"], "parameters": strip(fn.get("parameters", {}))}


# ---------------------------------------------------------------------------
# Registry-level mechanism
# ---------------------------------------------------------------------------


class TestCompactMechanism:
    def _reg(self, **register_kwargs):
        reg = ToolRegistry()
        reg.register(
            name="widget",
            toolset="core",
            schema={
                "name": "widget",
                "description": "A very long prose description " * 20,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "Long parameter prose " * 20,
                        }
                    },
                    "required": ["target"],
                },
            },
            handler=_dummy_handler,
            **register_kwargs,
        )
        return reg

    def test_compact_description_used_when_profile_on(self, cfg_home, monkeypatch):
        reg = self._reg(compact_description="Short. Does the thing.")

        _set_compact(cfg_home, monkeypatch, False)
        long_fn = reg.get_definitions({"widget"})[0]["function"]
        _set_compact(cfg_home, monkeypatch, True)
        short_fn = reg.get_definitions({"widget"})[0]["function"]

        assert short_fn["description"] == "Short. Does the thing."
        assert len(short_fn["description"]) < len(long_fn["description"])
        # Structure is untouched.
        assert _structure(short_fn) == _structure(long_fn)

    def test_compact_param_descriptions_applied(self, cfg_home, monkeypatch):
        reg = self._reg(
            compact_description="Short.",
            compact_parameter_descriptions={"target": "The target."},
        )

        _set_compact(cfg_home, monkeypatch, True)
        fn = reg.get_definitions({"widget"})[0]["function"]
        assert fn["parameters"]["properties"]["target"]["description"] == "The target."
        assert fn["parameters"]["required"] == ["target"]

    def test_tool_without_compact_variant_is_unchanged(self, cfg_home, monkeypatch):
        reg = self._reg()  # no compact_* kwargs

        _set_compact(cfg_home, monkeypatch, False)
        off = reg.get_definitions({"widget"})
        _set_compact(cfg_home, monkeypatch, True)
        on = reg.get_definitions({"widget"})

        assert off == on

    def test_compact_never_mutates_registered_schema(self, cfg_home, monkeypatch):
        reg = self._reg(
            compact_description="Short.",
            compact_parameter_descriptions={"target": "The target."},
        )
        entry = reg.get_entry("widget")
        before = json.dumps(entry.schema, sort_keys=True)

        _set_compact(cfg_home, monkeypatch, True)
        reg.get_definitions({"widget"})

        assert json.dumps(entry.schema, sort_keys=True) == before

    def test_unknown_compact_param_name_is_ignored(self, cfg_home, monkeypatch):
        """A stale compact param key must not invent a property."""
        reg = self._reg(
            compact_description="Short.",
            compact_parameter_descriptions={"nonexistent": "ignored"},
        )
        _set_compact(cfg_home, monkeypatch, True)
        fn = reg.get_definitions({"widget"})[0]["function"]
        assert "nonexistent" not in fn["parameters"]["properties"]


class TestConfigResolution:
    def test_default_is_off(self, cfg_home, monkeypatch):
        _write_cfg(cfg_home, {})
        from tools import registry as registry_mod

        registry_mod._reset_compact_cache_for_tests()
        assert compact_schemas_enabled() is False

    def test_enabled_by_config(self, cfg_home, monkeypatch):
        _set_compact(cfg_home, monkeypatch, True)
        assert compact_schemas_enabled() is True

    def test_default_config_documents_the_key(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["agent"]["compact_tool_schemas"] is False


# ---------------------------------------------------------------------------
# Real tools, real toolset
# ---------------------------------------------------------------------------


class TestTelegramToolsetProfiles:
    """Profile off must be a no-op; profile on must shrink only the five."""

    @pytest.fixture
    def profiles(self, cfg_home, monkeypatch):
        _set_compact(cfg_home, monkeypatch, False)
        off = _telegram_definitions()
        _set_compact(cfg_home, monkeypatch, True)
        on = _telegram_definitions()
        return off, on

    def test_profile_off_is_byte_identical_to_no_key_at_all(
        self, cfg_home, monkeypatch
    ):
        _write_cfg(cfg_home, {})
        from tools import registry as registry_mod

        registry_mod._reset_compact_cache_for_tests()
        absent = _telegram_definitions()

        _set_compact(cfg_home, monkeypatch, False)
        explicit_off = _telegram_definitions()

        assert json.dumps(absent) == json.dumps(explicit_off)

    def test_compact_tools_shrink(self, profiles):
        off, on = profiles
        off_fns, on_fns = _by_name(off), _by_name(on)

        shrank = set()
        for name in COMPACT_TOOLS:
            if name not in off_fns:
                continue  # check_fn gated off in this environment
            assert len(json.dumps(on_fns[name])) < len(json.dumps(off_fns[name])), (
                f"{name} did not shrink under the compact profile"
            )
            shrank.add(name)

        assert shrank, "none of the compact tools were present in the toolset"

    def test_compact_descriptions_are_within_budget(self, profiles):
        _off, on = profiles
        on_fns = _by_name(on)
        for name in COMPACT_TOOLS:
            if name not in on_fns:
                continue
            desc = on_fns[name]["description"]
            assert len(desc) <= MAX_COMPACT_DESCRIPTION_CHARS, (
                f"{name} compact description is {len(desc)} chars "
                f"(budget {MAX_COMPACT_DESCRIPTION_CHARS})"
            )

    def test_other_tools_are_untouched(self, profiles):
        off, on = profiles
        off_fns, on_fns = _by_name(off), _by_name(on)

        assert set(off_fns) == set(on_fns), "compact profile changed the tool set"

        for name, off_fn in off_fns.items():
            if name in COMPACT_TOOLS:
                continue
            assert json.dumps(on_fns[name], sort_keys=True) == json.dumps(
                off_fn, sort_keys=True
            ), f"{name} changed under the compact profile but has no compact variant"

    def test_structure_identical_for_every_tool(self, profiles):
        """Param names, types, enums and required lists never change."""
        off, on = profiles
        off_fns, on_fns = _by_name(off), _by_name(on)
        for name, off_fn in off_fns.items():
            assert _structure(on_fns[name]) == _structure(off_fn), (
                f"{name} structure changed under the compact profile"
            )

    def test_every_parameter_keeps_a_description(self, profiles):
        """Trimming prose must not leave a parameter undocumented."""
        _off, on = profiles
        for fn in _by_name(on).values():
            props = fn.get("parameters", {}).get("properties", {})
            for pname, spec in props.items():
                if not isinstance(spec, dict):
                    continue
                assert spec.get("description", "").strip(), (
                    f"{fn['name']}.{pname} lost its description"
                )

    def test_total_schema_bytes_drop_meaningfully(self, profiles):
        off, on = profiles
        full = _schema_bytes(off)
        saved = full - _schema_bytes(on)
        floor = int(full * MIN_TELEGRAM_BYTES_SAVED_FRACTION)
        assert saved >= floor, (
            f"compact profile saved only {saved} bytes across "
            f"{len(off)} telegram tools ({full} full); expected >= {floor} "
            f"({MIN_TELEGRAM_BYTES_SAVED_FRACTION:.0%})"
        )


class TestDelegateTaskDynamicCompaction:
    """delegate_task's description is config-derived; compaction must keep that."""

    def test_compact_description_still_reports_live_limits(
        self, cfg_home, monkeypatch
    ):
        _write_cfg(
            cfg_home,
            {
                "agent": {"compact_tool_schemas": True},
                "delegation": {"max_concurrent_children": 7},
            },
        )
        from tools import registry as registry_mod

        registry_mod._reset_compact_cache_for_tests()

        fns = _by_name(_telegram_definitions())
        if "delegate_task" not in fns:
            pytest.skip("delegate_task not available in this environment")
        tasks_desc = fns["delegate_task"]["parameters"]["properties"]["tasks"][
            "description"
        ]
        assert "7" in tasks_desc, (
            "compact delegate_task must still report the user's actual "
            f"max_concurrent_children; got: {tasks_desc!r}"
        )
