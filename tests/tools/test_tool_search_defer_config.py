"""`tools.tool_search.defer` must be a registered config key, and its semantics must hold.

Regression: ``hermes config set tools.tool_search.defer '["terminal"]'`` answered
"'tools.tool_search.defer' is not a recognized config key — it was saved anyway, but Hermes may
not read it" for a key ``ToolSearchConfig.from_raw`` reads on every assembly. The reader had no
``DEFAULT_CONFIG`` entry, so the key was invisible to ``hermes config``, unvalidated by the
value-coercion guardrail (a JSON list had to be written by hand) and unmigrated with the config
version. ``hermes_cli/AGENTS.md`` names exactly this drift mode.

The default is the curated LIST, not a string sentinel: ``_coerce_config_set_value`` returns the
user's literal verbatim when the registered default is a string, so a sentinel would store
``'["terminal"]'`` as text that the isinstance-gated reader then ignores.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from hermes_cli.config import load_config
from hermes_cli.config_defaults import DEFAULT_CONFIG
from tools.tool_search import _DEFAULT_DEFERRED_TOOLS, ToolSearchConfig, is_deferrable_tool_name


@pytest.fixture(autouse=True)
def _isolated_hermes_home(tmp_path):
    """Point HERMES_HOME at a temp dir; tests/conftest.py does the same for the suite."""
    (tmp_path / ".env").touch()
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        yield tmp_path


def _effective_defer(config_yaml: str | None):
    """The defer set a session actually gets, through the loader the consumer uses."""
    if config_yaml is not None:
        (Path(os.environ["HERMES_HOME"]) / "config.yaml").write_text(config_yaml)
    return ToolSearchConfig.from_raw(load_config()["tools"]["tool_search"]).effective_defer_tools


class TestRegisteredDefault:
    def test_registered_default_is_the_curated_runtime_set(self):
        """One source of truth: the list a user reads is the set the agent obeys."""
        assert frozenset(DEFAULT_CONFIG["tools"]["tool_search"]["defer"]) == _DEFAULT_DEFERRED_TOOLS
        assert isinstance(DEFAULT_CONFIG["tools"]["tool_search"]["defer"], list)

    def test_unset_config_defers_the_curated_set(self, _isolated_hermes_home):
        assert _effective_defer(None) == _DEFAULT_DEFERRED_TOOLS
        assert _DEFAULT_DEFERRED_TOOLS  # a sentinel default must not mean "defer nothing"

    def test_clarify_is_not_deferred_by_default(self, _isolated_hermes_home):
        """A/B verdict (288 runs): deferring clarify collapsed structured asks 18/18 -> 7/18."""
        assert is_deferrable_tool_name("clarify", _effective_defer(None)) is False


class TestUserOverride:
    def test_user_list_replaces_the_curated_set(self, _isolated_hermes_home):
        defer = _effective_defer("tools:\n  tool_search:\n    defer:\n      - terminal\n")
        assert defer == frozenset({"terminal"})
        assert is_deferrable_tool_name("terminal", defer) is True
        assert is_deferrable_tool_name("todo_list", defer) is False  # curated set replaced wholesale

    def test_empty_list_defers_no_core_tool(self, _isolated_hermes_home):
        defer = _effective_defer("tools:\n  tool_search:\n    defer: []\n")
        assert defer == frozenset()
        assert is_deferrable_tool_name("terminal", defer) is False
        assert is_deferrable_tool_name("tool_search", defer) is False  # bridge never defers

    def test_cli_set_writes_a_real_list(self, _isolated_hermes_home, capsys):
        """The documented `hermes config set` route must store a list, not its text."""
        from hermes_cli.config import set_config_value

        set_config_value("tools.tool_search.defer", '["terminal", "skills_list"]')

        assert "not a recognized config key" not in capsys.readouterr().out + capsys.readouterr().err
        written = yaml.safe_load((_isolated_hermes_home / "config.yaml").read_text())
        assert written["tools"]["tool_search"]["defer"] == ["terminal", "skills_list"]
        assert ToolSearchConfig.from_raw(
            written["tools"]["tool_search"]).effective_defer_tools == frozenset(
                {"terminal", "skills_list"})
