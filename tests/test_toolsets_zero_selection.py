"""``--toolsets`` must be able to say "zero tools", and must never silently mean the opposite.

``hermes -z '<prompt>' -t ''`` is the idiom people reach for when they want a tool-less
liveness probe or classifier. It did the opposite: ``_normalize_toolsets`` folds a falsy value
to ``None``, and ``None`` means "use the user's configured toolsets" — so the probe ran with
terminal, file writes and everything else, with nothing in the output saying so.

Two contracts, both about the relationship between what was asked for and what is sent:

* an EXPLICITLY EMPTY selection is an error, not a silent fallback to the config default;
* ``none`` is a real toolset meaning zero tools, and it survives the kanban worker's
  automatic toolset injection (which exists so a worker can always hand its card back).

Absent-vs-empty is a real distinction at the parser (``default=None``), so it is preserved
here: ``None`` still means "not specified".

Regression for #126122.
"""

import pytest

from hermes_cli.oneshot import _validate_explicit_toolsets
from model_tools import _select_tool_names
from toolsets import get_toolset, resolve_toolset, validate_toolset


class TestNoneIsARealToolset:
    """``none`` must exist and mean exactly zero tools."""

    def test_none_is_registered_and_validates(self):
        assert validate_toolset("none"), "the sentinel must pass the same validation as any name"
        assert get_toolset("none") is not None

    def test_none_resolves_to_zero_tools(self):
        # resolve_toolset is documented to return a sorted List[str], not a set.
        assert resolve_toolset("none") == [], (
            "the whole point of the sentinel is an empty tool list"
        )


class TestExplicitEmptyIsRejected:
    """``-t ''`` asked for nothing; silently granting the configured toolsets is the bug."""

    def test_empty_string_is_an_error_not_a_config_fallback(self):
        toolsets, error = _validate_explicit_toolsets("")
        assert error is not None, (
            f"an explicitly empty --toolsets silently resolved to {toolsets!r} "
            "(None means 'use the configured toolsets')"
        )
        assert toolsets is None

    @pytest.mark.parametrize("value", ["   ", ",", " , "])
    def test_whitespace_and_bare_separators_are_also_errors(self, value):
        _, error = _validate_explicit_toolsets(value)
        assert error is not None, f"{value!r} carries no toolset name and must not fall back"

    def test_absent_flag_still_means_config_default(self):
        """The absent case is unchanged — this is the behavior everything else relies on."""
        assert _validate_explicit_toolsets(None) == (None, None)


class TestNoneSentinelThroughTheOneShotPath:
    """``-t none`` must survive validation as a real, empty selection."""

    def test_none_alone_is_accepted(self):
        toolsets, error = _validate_explicit_toolsets("none")
        assert error is None
        assert toolsets == ["none"]

    def test_none_combined_with_another_toolset_is_rejected(self):
        """``none`` plus anything else is contradictory — ``none,web`` cannot mean zero tools."""
        _, error = _validate_explicit_toolsets("none,web")
        assert error is not None, "combining the zero-tool sentinel with a real toolset must fail"


class TestNoneSurvivesTheKanbanInjection:
    """A worker asked for zero tools gets zero tools.

    The dispatcher appends ``kanban`` so a worker can always hand its card back, which is
    right for a normal selection and wrong for an explicit zero-tool request: the caller
    asked for a session that cannot act at all.
    """

    def test_dispatcher_worker_gets_no_kanban_tools_when_none_requested(self, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "card-123")
        monkeypatch.setattr("model_tools._is_delegated_child_context", lambda: False)
        monkeypatch.setattr("model_tools._is_dispatcher_owned_worker", lambda: True)

        assert _select_tool_names(["none"], None, quiet_mode=True) == set(), (
            "the kanban toolset was injected into an explicit zero-tool selection"
        )

    def test_dispatcher_worker_still_gets_kanban_for_a_normal_selection(self, monkeypatch):
        """The negative case: the injection must survive for every other selection."""
        monkeypatch.setenv("HERMES_KANBAN_TASK", "card-123")
        monkeypatch.setattr("model_tools._is_delegated_child_context", lambda: False)
        monkeypatch.setattr("model_tools._is_dispatcher_owned_worker", lambda: True)

        tools = _select_tool_names(["web"], None, quiet_mode=True)
        assert tools, "a normal toolset selection must still resolve tools"
        assert any(t.startswith("kanban") for t in tools), (
            "a dispatcher-owned worker must keep the kanban lifecycle tools for a normal selection"
        )
