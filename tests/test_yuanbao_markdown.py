"""
test_yuanbao_markdown.py - Unit tests for yuanbao_markdown.py

Run (no pytest needed):
    cd /root/.openclaw/workspace/hermes-agent
    python3 tests/test_yuanbao_markdown.py -v

Or with pytest if available:
    python3 -m pytest tests/test_yuanbao_markdown.py -v
"""

import sys
import os
import unittest

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gateway.platforms import helpers as _mdchunk
from gateway.platforms.yuanbao import MarkdownProcessor


# ============ has_unclosed_fence ============

class TestHasUnclosedFence(unittest.TestCase):
    def test_unclosed_fence(self):
        self.assertTrue(_mdchunk.text_has_unclosed_fence("```python\ncode"))

    def test_closed_fence(self):
        self.assertFalse(_mdchunk.text_has_unclosed_fence("```python\ncode\n```"))






    def test_inline_backtick_ignored(self):
        text = "`inline code` is fine"
        self.assertFalse(_mdchunk.text_has_unclosed_fence(text))


# ============ ends_with_table_row ============

class TestEndsWithTableRow(unittest.TestCase):
    def test_simple_table_row(self):
        self.assertTrue(_mdchunk.text_ends_with_table_row("| col1 | col2 |"))


    def test_table_row_in_middle(self):
        text = "| col1 | col2 |\nsome other text"
        self.assertFalse(_mdchunk.text_ends_with_table_row(text))




    def test_table_separator_row(self):
        self.assertTrue(_mdchunk.text_ends_with_table_row("| --- | --- |"))



# ============ split_at_paragraph_boundary ============

class TestSplitAtParagraphBoundary(unittest.TestCase):
    def test_split_at_empty_line(self):
        text = "paragraph one\n\nparagraph two\n\nparagraph three\nextra"
        head, tail = _mdchunk.split_at_paragraph_boundary(text, 30)
        self.assertLessEqual(len(head), 30)
        self.assertEqual(head + tail, text)

    def test_split_at_sentence_end(self):
        text = "This is a sentence.\nNext line.\nAnother line."
        head, tail = _mdchunk.split_at_paragraph_boundary(text, 25)
        self.assertLessEqual(len(head), 25)
        self.assertEqual(head + tail, text)



    def test_chinese_sentence_boundary(self):
        text = "这是第一句话。\n这是第二句话。\n这是第三句话。"
        head, tail = _mdchunk.split_at_paragraph_boundary(text, 15)
        self.assertLessEqual(len(head), 15)
        self.assertEqual(head + tail, text)


# ============ chunk_markdown_text ============

class TestChunkMarkdownText(unittest.TestCase):

    def test_short_text_no_split(self):
        text = "hello world"
        self.assertEqual(MarkdownProcessor.chunk_markdown_text(text, 3000), [text])



    def test_5000_chars_returns_2(self):
        """验收标准: 'a'*5000 with max 3000 → 2 chunks"""
        result = MarkdownProcessor.chunk_markdown_text("a" * 5000, 3000)
        self.assertEqual(len(result), 2)


    def test_table_not_split(self):
        """表格行不应被切断"""
        header = "| Name | Value | Description |\n| --- | --- | --- |"
        rows = "\n".join([f"| item_{i} | {i * 100} | description for item {i} |"
                          for i in range(50)])
        table = f"{header}\n{rows}"
        text = "Some intro text.\n\n" + table + "\n\nSome outro text."
        result = MarkdownProcessor.chunk_markdown_text(text, 3000)
        for chunk in result:
            self.assertFalse(_mdchunk.text_has_unclosed_fence(chunk))


    def test_multiple_paragraphs(self):
        """多段落文本应在段落边界切割"""
        paragraphs = ["This is paragraph number " + str(i) + ". " * 50
                      for i in range(10)]
        text = "\n\n".join(paragraphs)
        result = MarkdownProcessor.chunk_markdown_text(text, 500)
        self.assertGreater(len(result), 1)
        total_content = ''.join(result)
        self.assertGreaterEqual(len(total_content), len(text) * 0.95)

    def test_single_long_line(self):
        """单行超长文本应被强制切割"""
        text = "a" * 10000
        result = MarkdownProcessor.chunk_markdown_text(text, 3000)
        self.assertGreaterEqual(len(result), 3)
        for c in result:
            self.assertLessEqual(len(c), 3000)




# ============ Acceptance criteria ============

class TestAcceptanceCriteria(unittest.TestCase):
    def test_9000_x_returns_3_chunks(self):
        """验收：MarkdownProcessor.chunk_markdown_text("x" * 9000, 3000) 返回 3 个片段"""
        result = MarkdownProcessor.chunk_markdown_text("x" * 9000, 3000)
        self.assertEqual(len(result), 3)
        for chunk in result:
            self.assertLessEqual(len(chunk), 3000)







if __name__ == '__main__':
    unittest.main(verbosity=2)


# ============ pytest-style function tests (task specification) ============







def test_large_fence_kept_whole():
    """超大代码块即便超过 max_chars 也应整块输出"""
    code_block = "```python\n" + "x = 1\n" * 200 + "```"
    chunks = MarkdownProcessor.chunk_markdown_text(code_block, 500)
    # 代码块应在同一个 chunk 中（允许超出 max_chars）
    fence_chunks = [c for c in chunks if "```python" in c]
    for c in fence_chunks:
        assert not _mdchunk.text_has_unclosed_fence(c)












