"""String-typed toolset lists must be coerced, not iterated per character.

A YAML-quoted string (``disabled_toolsets: '["web", "browser"]'``) or bare
scalar (``disabled_toolsets: web``) in config.yaml reaches consumers as a
Python string. Iterating it yields characters, matches no toolset, and the
denylist is silently ignored — the full tool schema set is then sent on every
model call (#61264, #61265).
"""

import logging

import model_tools
from cron.scheduler import _resolve_cron_disabled_toolsets
from toolsets import normalize_toolset_spec


class TestNormalizeToolsetSpec:
    def test_none_passthrough(self):
        assert normalize_toolset_spec(None) is None

    def test_list_passthrough(self):
        assert normalize_toolset_spec(["web", "browser"]) == ["web", "browser"]

    def test_tuple_becomes_list(self):
        assert normalize_toolset_spec(("web",)) == ["web"]

    def test_json_string_decodes(self, caplog):
        with caplog.at_level(logging.WARNING, logger="toolsets"):
            assert normalize_toolset_spec('["web", "browser"]') == ["web", "browser"]
        assert "configured as the string" in caplog.text

    def test_bare_scalar_string(self):
        assert normalize_toolset_spec("web") == ["web"]

    def test_comma_separated_string(self):
        assert normalize_toolset_spec("web, browser") == ["web", "browser"]

    def test_malformed_json_falls_back_to_comma_split(self):
        # Unbalanced bracket: json.loads fails; names are still recovered.
        assert normalize_toolset_spec('["web", "browser"') == ["web", "browser"]

    def test_quoted_names_in_comma_form(self):
        assert normalize_toolset_spec("'web', \"browser\"") == ["web", "browser"]


class TestModelToolsStringCoercion:
    def test_string_disabled_toolsets_equals_list_form(self):
        as_list = {
            t["function"]["name"]
            for t in model_tools.get_tool_definitions(
                disabled_toolsets=["terminal", "file"], quiet_mode=True
            )
        }
        as_yaml_string = {
            t["function"]["name"]
            for t in model_tools.get_tool_definitions(
                disabled_toolsets='["terminal", "file"]', quiet_mode=True
            )
        }
        assert as_yaml_string == as_list
        # And the disable actually took: unfiltered catalog is strictly larger.
        everything = {
            t["function"]["name"]
            for t in model_tools.get_tool_definitions(quiet_mode=True)
        }
        assert as_yaml_string < everything

    def test_string_enabled_toolsets_equals_list_form(self):
        as_list = {
            t["function"]["name"]
            for t in model_tools.get_tool_definitions(
                enabled_toolsets=["terminal"], quiet_mode=True
            )
        }
        as_string = {
            t["function"]["name"]
            for t in model_tools.get_tool_definitions(
                enabled_toolsets="terminal", quiet_mode=True
            )
        }
        assert as_string == as_list


class TestCronDisabledToolsetsStringCoercion:
    def test_string_config_yields_names_not_characters(self):
        cfg = {"agent": {"disabled_toolsets": '["web", "browser"]'}}
        disabled = _resolve_cron_disabled_toolsets(cfg)
        assert "web" in disabled
        assert "browser" in disabled
        # No single-character garbage from per-char iteration.
        assert not any(len(name) == 1 for name in disabled)

    def test_list_config_unchanged(self):
        cfg = {"agent": {"disabled_toolsets": ["web"]}}
        disabled = _resolve_cron_disabled_toolsets(cfg)
        assert disabled == ["cronjob", "messaging", "clarify", "web"]
