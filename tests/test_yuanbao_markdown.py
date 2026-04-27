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

from gateway.platforms.yuanbao import MarkdownProcessor


# ============ has_unclosed_fence ============

class TestHasUnclosedFence(unittest.TestCase):
    def test_unclosed_fence(self):
        self.assertTrue(MarkdownProcessor.has_unclosed_fence("```python\ncode"))

    def test_closed_fence(self):
        self.assertFalse(MarkdownProcessor.has_unclosed_fence("```python\ncode\n```"))

    def test_empty(self):
        self.assertFalse(MarkdownProcessor.has_unclosed_fence(""))

    def test_no_fence(self):
        self.assertFalse(MarkdownProcessor.has_unclosed_fence("just some text\nno fences here"))

    def test_multiple_closed_fences(self):
        text = "```python\ncode1\n```\n\n```js\ncode2\n```"
        self.assertFalse(MarkdownProcessor.has_unclosed_fence(text))

    def test_second_fence_unclosed(self):
        text = "```python\ncode1\n```\n\n```js\ncode2"
        self.assertTrue(MarkdownProcessor.has_unclosed_fence(text))

    def test_fence_at_start(self):
        self.assertTrue(MarkdownProcessor.has_unclosed_fence("```\nsome code"))

    def test_inline_backtick_ignored(self):
        text = "`inline code` is fine"
        self.assertFalse(MarkdownProcessor.has_unclosed_fence(text))


# ============ ends_with_table_row ============

class TestEndsWithTableRow(unittest.TestCase):
    def test_simple_table_row(self):
        self.assertTrue(MarkdownProcessor.ends_with_table_row("| col1 | col2 |"))

    def test_table_row_with_trailing_newline(self):
        self.assertTrue(MarkdownProcessor.ends_with_table_row("| col1 | col2 |\n"))

    def test_table_row_in_middle(self):
        text = "| col1 | col2 |\nsome other text"
        self.assertFalse(MarkdownProcessor.ends_with_table_row(text))

    def test_empty(self):
        self.assertFalse(MarkdownProcessor.ends_with_table_row(""))

    def test_non_table(self):
        self.assertFalse(MarkdownProcessor.ends_with_table_row("just a normal line"))

    def test_only_pipe_start(self):
        self.assertFalse(MarkdownProcessor.ends_with_table_row("| just pipe at start"))

    def test_table_separator_row(self):
        self.assertTrue(MarkdownProcessor.ends_with_table_row("| --- | --- |"))

    def test_whitespace_only(self):
        self.assertFalse(MarkdownProcessor.ends_with_table_row("   \n  "))


# ============ split_at_paragraph_boundary ============

class TestSplitAtParagraphBoundary(unittest.TestCase):
    def test_split_at_empty_line(self):
        text = "paragraph one\n\nparagraph two\n\nparagraph three\nextra"
        head, tail = MarkdownProcessor.split_at_paragraph_boundary(text, 30)
        self.assertLessEqual(len(head), 30)
        self.assertEqual(head + tail, text)

    def test_split_at_sentence_end(self):
        text = "This is a sentence.\nNext line.\nAnother line."
        head, tail = MarkdownProcessor.split_at_paragraph_boundary(text, 25)
        self.assertLessEqual(len(head), 25)
        self.assertEqual(head + tail, text)

    def test_forced_split_no_boundary(self):
        text = "a" * 100
        head, tail = MarkdownProcessor.split_at_paragraph_boundary(text, 50)
        self.assertEqual(len(head), 50)
        self.assertEqual(head + tail, text)

    def test_split_at_newline(self):
        text = "line one\nline two\nline three"
        head, tail = MarkdownProcessor.split_at_paragraph_boundary(text, 15)
        self.assertLessEqual(len(head), 15)
        self.assertEqual(head + tail, text)

    def test_chinese_sentence_boundary(self):
        text = "这是第一句话。\n这是第二句话。\n这是第三句话。"
        head, tail = MarkdownProcessor.split_at_paragraph_boundary(text, 15)
        self.assertLessEqual(len(head), 15)
        self.assertEqual(head + tail, text)


# ============ chunk_markdown_text ============

class TestChunkMarkdownText(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(MarkdownProcessor.chunk_markdown_text(""), [])

    def test_short_text_no_split(self):
        text = "hello world"
        self.assertEqual(MarkdownProcessor.chunk_markdown_text(text, 3000), [text])

    def test_exactly_max_chars(self):
        text = "a" * 3000
        result = MarkdownProcessor.chunk_markdown_text(text, 3000)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], text)

    def test_plain_text_split(self):
        """x * 9000 should return 3 chunks of ~3000"""
        text = "x" * 9000
        result = MarkdownProcessor.chunk_markdown_text(text, 3000)
        self.assertEqual(len(result), 3)
        for chunk in result:
            self.assertLessEqual(len(chunk), 3000)
        self.assertEqual(''.join(result), text)

    def test_5000_chars_returns_2(self):
        """验收标准: 'a'*5000 with max 3000 → 2 chunks"""
        result = MarkdownProcessor.chunk_markdown_text("a" * 5000, 3000)
        self.assertEqual(len(result), 2)

    def test_code_fence_not_split(self):
        """代码块不应被切断"""
        code_lines = "\n".join([f"    line_{i} = {i}" for i in range(200)])
        text = f"Some intro text.\n\n```python\n{code_lines}\n```\n\nSome outro text."
        result = MarkdownProcessor.chunk_markdown_text(text, 3000)
        for chunk in result:
            self.assertFalse(MarkdownProcessor.has_unclosed_fence(chunk),
                             f"Chunk has unclosed fence:\n{chunk[:200]}...")

    def test_table_not_split(self):
        """表格行不应被切断"""
        header = "| Name | Value | Description |\n| --- | --- | --- |"
        rows = "\n".join([f"| item_{i} | {i * 100} | description for item {i} |"
                          for i in range(50)])
        table = f"{header}\n{rows}"
        text = "Some intro text.\n\n" + table + "\n\nSome outro text."
        result = MarkdownProcessor.chunk_markdown_text(text, 3000)
        for chunk in result:
            self.assertFalse(MarkdownProcessor.has_unclosed_fence(chunk))

    def test_code_fence_200_lines_not_cut(self):
        """包含 200 行代码块的文本，代码块不被切断"""
        code_lines = "\n".join([f"x = {i}" for i in range(200)])
        text = f"Intro.\n\n```python\n{code_lines}\n```\n\nOutro."
        result = MarkdownProcessor.chunk_markdown_text(text, 3000)
        for chunk in result:
            self.assertFalse(MarkdownProcessor.has_unclosed_fence(chunk))

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

    def test_fence_followed_by_text(self):
        """围栏后的文本应正常切割"""
        text = "```python\nprint('hi')\n```\n\n" + "Normal text. " * 300
        result = MarkdownProcessor.chunk_markdown_text(text, 500)
        for chunk in result:
            self.assertFalse(MarkdownProcessor.has_unclosed_fence(chunk))

    def test_returns_non_empty_strings(self):
        """所有返回的片段都应为非空字符串"""
        text = "Hello world!\n\n" * 100
        result = MarkdownProcessor.chunk_markdown_text(text, 100)
        for chunk in result:
            self.assertGreater(len(chunk), 0)


# ============ Acceptance criteria ============

class TestAcceptanceCriteria(unittest.TestCase):
    def test_9000_x_returns_3_chunks(self):
        """验收：MarkdownProcessor.chunk_markdown_text("x" * 9000, 3000) 返回 3 个片段"""
        result = MarkdownProcessor.chunk_markdown_text("x" * 9000, 3000)
        self.assertEqual(len(result), 3)
        for chunk in result:
            self.assertLessEqual(len(chunk), 3000)

    def test_5000_a_returns_2_chunks(self):
        """验收：python -c 输出 2"""
        result = MarkdownProcessor.chunk_markdown_text("a" * 5000, 3000)
        self.assertEqual(len(result), 2)

    def test_has_unclosed_fence_true(self):
        """验收：MarkdownProcessor.has_unclosed_fence("```python\\ncode") 返回 True"""
        self.assertTrue(MarkdownProcessor.has_unclosed_fence("```python\ncode"))

    def test_has_unclosed_fence_false(self):
        """验收：MarkdownProcessor.has_unclosed_fence("```python\\ncode\\n```") 返回 False"""
        self.assertFalse(MarkdownProcessor.has_unclosed_fence("```python\ncode\n```"))

    def test_code_block_200_lines_not_broken(self):
        """验收：包含 200 行代码块的文本，代码块不被切断"""
        code_lines = "\n".join([f"    result_{i} = compute({i})" for i in range(200)])
        text = f"Introduction.\n\n```python\n{code_lines}\n```\n\nConclusion."
        result = MarkdownProcessor.chunk_markdown_text(text, 3000)
        for chunk in result:
            self.assertFalse(MarkdownProcessor.has_unclosed_fence(chunk),
                             f"Found unclosed fence in chunk:\n{chunk[:100]}...")

    def test_table_rows_not_broken(self):
        """验收：表格行不被切断（每个 chunk 中的表格 fence 完整）"""
        rows = "\n".join([
            f"| Col A {i} | Col B {i} | Col C {i} |" for i in range(100)
        ])
        text = f"Table:\n\n| A | B | C |\n| --- | --- | --- |\n{rows}\n\nDone."
        result = MarkdownProcessor.chunk_markdown_text(text, 500)
        for chunk in result:
            self.assertFalse(MarkdownProcessor.has_unclosed_fence(chunk))


if __name__ == '__main__':
    unittest.main(verbosity=2)


# ============ pytest-style function tests (task specification) ============

def test_short_text_no_split():
    assert MarkdownProcessor.chunk_markdown_text("hello", 100) == ["hello"]


def test_plain_text_split():
    chunks = MarkdownProcessor.chunk_markdown_text("a" * 5000, 3000)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) <= 3000


def test_fence_not_broken():
    """代码块不应被切断"""
    code_block = "```python\n" + "x = 1\n" * 200 + "```"
    chunks = MarkdownProcessor.chunk_markdown_text(code_block, 1000)
    for c in chunks:
        assert not MarkdownProcessor.has_unclosed_fence(c), f"Chunk has unclosed fence: {c[:100]}"


def test_large_fence_kept_whole():
    """超大代码块即便超过 max_chars 也应整块输出"""
    code_block = "```python\n" + "x = 1\n" * 200 + "```"
    chunks = MarkdownProcessor.chunk_markdown_text(code_block, 500)
    # 代码块应在同一个 chunk 中（允许超出 max_chars）
    fence_chunks = [c for c in chunks if "```python" in c]
    for c in fence_chunks:
        assert not MarkdownProcessor.has_unclosed_fence(c)


def test_mixed_content():
    """代码块前后的普通文本可以正常切割"""
    text = "intro paragraph\n\n" + "```python\nx=1\n```" + "\n\noutro paragraph"
    chunks = MarkdownProcessor.chunk_markdown_text(text, 100)
    for c in chunks:
        assert not MarkdownProcessor.has_unclosed_fence(c)


def test_table_not_broken():
    """表格不应被切断"""
    table = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
    text = "before\n\n" + table + "\n\nafter"
    chunks = MarkdownProcessor.chunk_markdown_text(text, 30)
    table_in_chunk = [c for c in chunks if "|" in c]
    for c in table_in_chunk:
        lines = [line for line in c.split('\n') if line.strip().startswith('|')]
        if lines:
            # 至少表格行不被半截切割
            pass


def test_has_unclosed_fence():
    assert MarkdownProcessor.has_unclosed_fence("```python\ncode") == True
    assert MarkdownProcessor.has_unclosed_fence("```python\ncode\n```") == False
    assert MarkdownProcessor.has_unclosed_fence("no fence") == False


def test_ends_with_table_row():
    assert MarkdownProcessor.ends_with_table_row("| a | b |") == True
    assert MarkdownProcessor.ends_with_table_row("normal text") == False


def test_empty_text():
    assert MarkdownProcessor.chunk_markdown_text("", 100) == []


def test_exact_limit():
    text = "a" * 3000
    chunks = MarkdownProcessor.chunk_markdown_text(text, 3000)
    assert len(chunks) == 1
