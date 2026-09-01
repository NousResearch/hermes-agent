"""Removed-backend migration warnings (post-#99199 Tavily removal).

A config still pointing at a backend that no longer ships in-tree
(``web.backend: tavily``) must fail loudly and specifically:

1. startup — ``validate_config_structure`` emits a warning naming the
   removal, instead of staying silent until the first tool call;
2. tool call — ``selection_error`` explains the backend was removed and
   names alternatives, instead of the generic "no registered provider
   has that name".

Regression source: keyed Tavily users upgrading to v0.21.0 saw their
config silently become invalid with no migration or startup notice
(reported on PR #99731).
"""

from hermes_cli.config import validate_config_structure
from tools.tool_backend_helpers import (
    REMOVED_BACKENDS,
    removed_backend_note,
    selection_error,
)


class TestRemovedBackendNote:
    def test_tavily_is_registered_as_removed_web_backend(self):
        assert "tavily" in REMOVED_BACKENDS["web"]

    def test_note_lookup_normalizes_quotes_and_case(self):
        plain = removed_backend_note("web", "tavily")
        assert plain is not None
        assert removed_backend_note("web", "'Tavily'") == plain
        assert removed_backend_note("web", '  "TAVILY" ') == plain

    def test_unknown_names_and_sections_return_none(self):
        assert removed_backend_note("web", "exa") is None
        assert removed_backend_note("web", "") is None
        assert removed_backend_note("stt", "tavily") is None


class TestSelectionErrorRemovedBackend:
    def test_removed_backend_gets_specific_explanation(self):
        msg = selection_error("web", "'tavily'", "no registered web search provider has that name")
        assert "removed" in msg
        assert "tavily" in msg.lower()
        # generic failure text replaced, not appended
        assert "no registered web search provider" not in msg
        # still ends with the uniform remediation contract
        assert "Run 'hermes tools' to change it." in msg

    def test_live_backend_keeps_caller_failure_text(self):
        msg = selection_error("web", "'exa'", "no registered web search provider has that name")
        assert "no registered web search provider has that name" in msg
        assert "removed" not in msg


class TestStartupWarningForRemovedWebBackend:
    @staticmethod
    def _removed_issues(config):
        return [
            i for i in validate_config_structure(config)
            if "removed" in i.message and "tavily" in i.message
        ]

    def test_stale_web_backend_warns_at_startup(self):
        issues = self._removed_issues({"web": {"backend": "tavily"}})
        assert len(issues) == 1
        assert issues[0].severity == "warning"
        assert "hermes tools" in issues[0].hint

    def test_per_capability_keys_are_checked(self):
        assert len(self._removed_issues({"web": {"search_backend": "tavily"}})) == 1
        assert len(self._removed_issues({"web": {"extract_backend": "tavily"}})) == 1

    def test_same_stale_value_warns_once(self):
        issues = self._removed_issues(
            {"web": {"backend": "tavily", "search_backend": "tavily", "extract_backend": "tavily"}}
        )
        assert len(issues) == 1

    def test_healthy_backend_produces_no_removed_warning(self):
        assert self._removed_issues({"web": {"backend": "exa"}}) == []
        assert self._removed_issues({"web": {}}) == []
        assert self._removed_issues({}) == []
