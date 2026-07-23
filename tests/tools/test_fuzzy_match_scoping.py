"""Sweeper feedback regression: guard must be scoped to candidate position."""
from __future__ import annotations

from tools.fuzzy_match import fuzzy_find_and_replace


class TestIdempotencyGuardScoping:
    """Unrelated new_string occurrences must not block fuzzy matching."""

    def test_unrelated_new_string_occurrence_does_not_block(self):
        """If new_string appears ELSEWHERE in the file but old_string's
        context doesn't overlap with that occurrence, fuzzy match should proceed."""
        content = (
            "import os\n"
            "def foo():\n"
            "    x = 1\n"
            "    return x\n"
            "def bar():\n"
            "    return 'new_value'\n"
        )
        old = "    x = 1\n    return x"
        new = "    x = 2\n    return 'new_value'"
        # new_string's "return 'new_value'" appears in bar() but that's
        # an unrelated occurrence. The patch should still apply to foo().
        result, count, strat, err = fuzzy_find_and_replace(content, old, new)
        assert err is None, f"unrelated occurrence should not block: {err}"
        assert count == 1
        assert "x = 2" in result

    def test_patch_replace_retry_error_and_unchanged_content(self):
        """Re-applying the same patch must error and leave content unchanged."""
        content = "value = 1\n"
        old = "value = 1"
        new = "value = 2"
        once, _, _, _ = fuzzy_find_and_replace(content, old, new)
        twice, count2, _, err2 = fuzzy_find_and_replace(once, old, new)
        assert err2 is not None, "re-applying must error"
        assert count2 == 0
        assert twice == once, "content must be unchanged"
