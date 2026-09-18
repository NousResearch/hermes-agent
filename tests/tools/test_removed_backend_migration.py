"""Removed-backend migration warnings (registry added in #100540).

A config still pointing at a backend registered in
``tools.tool_backend_helpers.REMOVED_BACKENDS`` must fail loudly and
specifically:

1. startup — ``validate_config_structure`` emits a warning naming the
   removal, instead of staying silent until the first tool call;
2. tool call — ``selection_error`` explains the backend was removed,
   instead of the generic "no registered provider has that name".

The registry ships empty on main (the Tavily removal that motivated it,
#99199, was reverted by the #99731 restore), so these tests inject a
synthetic ``legacysearch`` entry — they pin the machinery, not any
specific vendor's membership.
"""

import pytest

import tools.tool_backend_helpers as tbh
from hermes_cli.config import validate_config_structure
from tools.tool_backend_helpers import removed_backend_note, selection_error

_NOTE = "the LegacySearch backend was removed in v0.0.0 (alternatives: exa, parallel)"


@pytest.fixture
def legacy_removed(monkeypatch):
    monkeypatch.setitem(tbh.REMOVED_BACKENDS, "web", {"legacysearch": _NOTE})


class TestRemovedBackendNote:
    def test_note_lookup_normalizes_quotes_and_case(self, legacy_removed):
        assert removed_backend_note("web", "legacysearch") == _NOTE
        assert removed_backend_note("web", "'LegacySearch'") == _NOTE
        assert removed_backend_note("web", '  "LEGACYSEARCH" ') == _NOTE

    def test_unknown_names_and_sections_return_none(self, legacy_removed):
        assert removed_backend_note("web", "exa") is None
        assert removed_backend_note("web", "") is None
        assert removed_backend_note("stt", "legacysearch") is None

    def test_registry_ships_without_live_backends(self):
        # Restored/live backends must never sit in REMOVED_BACKENDS — the
        # startup warning would fire on a working provider. Guards the
        # #99731 restore against a stale tavily entry reappearing.
        from agent.web_search_registry import get_provider

        for name in tbh.REMOVED_BACKENDS.get("web", {}):
            assert get_provider(name) is None, (
                f"{name!r} is registered as removed but a live web provider "
                "with that name exists"
            )


class TestSelectionErrorRemovedBackend:
    def test_removed_backend_gets_specific_explanation(self, legacy_removed):
        msg = selection_error("web", "'legacysearch'", "no registered web search provider has that name")
        assert _NOTE in msg
        # generic failure text replaced, not appended
        assert "no registered web search provider" not in msg
        # still ends with the uniform remediation contract
        assert "Run 'hermes tools' to change it." in msg

    def test_live_backend_keeps_caller_failure_text(self, legacy_removed):
        msg = selection_error("web", "'exa'", "no registered web search provider has that name")
        assert "no registered web search provider has that name" in msg
        assert "removed" not in msg


class TestStartupWarningForRemovedWebBackend:
    @staticmethod
    def _removed_issues(config):
        return [
            i for i in validate_config_structure(config)
            if "removed" in i.message and "legacysearch" in i.message
        ]

    def test_stale_web_backend_warns_at_startup(self, legacy_removed):
        issues = self._removed_issues({"web": {"backend": "legacysearch"}})
        assert len(issues) == 1
        assert issues[0].severity == "warning"
        assert "hermes tools" in issues[0].hint

    def test_per_capability_keys_are_checked(self, legacy_removed):
        assert len(self._removed_issues({"web": {"search_backend": "legacysearch"}})) == 1
        assert len(self._removed_issues({"web": {"extract_backend": "legacysearch"}})) == 1

    def test_same_stale_value_warns_once(self, legacy_removed):
        issues = self._removed_issues(
            {"web": {"backend": "legacysearch", "search_backend": "legacysearch", "extract_backend": "legacysearch"}}
        )
        assert len(issues) == 1

    def test_healthy_backend_produces_no_removed_warning(self, legacy_removed):
        assert self._removed_issues({"web": {"backend": "exa"}}) == []
        assert self._removed_issues({"web": {}}) == []
        assert self._removed_issues({}) == []
