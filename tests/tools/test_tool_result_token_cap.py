"""Tests for the per-tool-result token cap in tools/tool_result_storage.py.

Background: the existing `maybe_persist_tool_result` path only kicks in for
very large outputs (default 100,000 chars ≈ 25K tokens).  A mid-size tool
result like a 56,000-char search_files dump (~14,000 tokens) slips through
and lands verbatim in conversation history.  Across a few tool calls these
add up fast and trigger the compression loop.

`cap_tool_result_tokens` is a defensive last-mile cap that runs BEFORE the
tool result is appended to the message list.  Above the soft cap (default
8,000 tokens) it slices the content down to `hard_token_target` (default
6,000 tokens) and appends a `[truncated NNNN tokens]` marker so the model
knows information was dropped.
"""

from tools.tool_result_storage import cap_tool_result_tokens


CHARS_PER_TOKEN = 4  # matches the estimator used elsewhere in the codebase


class TestCapToolResultTokens:
    def test_small_result_passes_through(self):
        content = "small result"
        out = cap_tool_result_tokens(content)
        assert out == content

    def test_result_under_soft_cap_unchanged(self):
        # 7,000 tokens × 4 chars = 28,000 chars — under the 8K-token cap
        content = "x" * (7_000 * CHARS_PER_TOKEN)
        out = cap_tool_result_tokens(content)
        assert out == content

    def test_result_just_over_soft_cap_truncated(self):
        # ~8,500 tokens = 34,000 chars — crosses the 8K-token cap
        content = "x" * (8_500 * CHARS_PER_TOKEN)
        out = cap_tool_result_tokens(content)

        # Output must be smaller than input and carry the truncation marker
        assert len(out) < len(content)
        assert "[truncated" in out and "tokens]" in out

        # Body is roughly 6,000 tokens — give the marker line a little slack
        out_tokens = len(out) // CHARS_PER_TOKEN
        assert 5_900 <= out_tokens <= 6_200

    def test_truncation_marker_reports_dropped_token_count(self):
        # 14,000 tokens → cap to 6,000 → ~8,000 dropped
        content = "x" * (14_000 * CHARS_PER_TOKEN)
        out = cap_tool_result_tokens(content)

        # Marker should be present with a non-zero "truncated NNNN tokens" count
        import re
        m = re.search(r"\[truncated\s+(\d+)\s+tokens\]", out)
        assert m, f"expected '[truncated NNNN tokens]' marker, got: {out[-200:]!r}"
        dropped = int(m.group(1))
        # ~8,000 dropped — allow some slack for marker length
        assert 7_500 <= dropped <= 8_500

    def test_non_string_content_passes_through(self):
        # Multimodal content (list of parts) must NOT be touched — they carry
        # image_url blocks that the budget enforcer handles separately.
        multimodal = [{"type": "text", "text": "x" * 200_000}]
        out = cap_tool_result_tokens(multimodal)
        assert out is multimodal

    def test_custom_caps_respected(self):
        # 3,000 tokens × 4 = 12,000 chars, soft cap 2,000 → truncate to 1,000
        content = "y" * (3_000 * CHARS_PER_TOKEN)
        out = cap_tool_result_tokens(
            content, soft_token_cap=2_000, hard_token_target=1_000,
        )
        assert "[truncated" in out
        out_tokens = len(out) // CHARS_PER_TOKEN
        assert 900 <= out_tokens <= 1_100

    def test_cap_keeps_head_of_content(self):
        # Sentinel at start must survive; sentinel at end must be dropped.
        head = "HEAD_SENTINEL_KEEP_ME"
        tail = "TAIL_SENTINEL_DROP_ME"
        body = "z" * (12_000 * CHARS_PER_TOKEN)
        content = head + body + tail
        out = cap_tool_result_tokens(content)
        assert head in out
        assert tail not in out
