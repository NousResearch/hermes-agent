"""Atomic targeting regressions for ordered V4A updates."""
import pytest

from tools.file_operations_common import ReadResult, WriteResult
from tools.patch_parser import apply_v4a_operations, parse_v4a_patch


class MemoryFiles:
    def __init__(self, source):
        self.source = source
        self.writes = []

    def read_file_raw(self, path):
        return ReadResult(content=self.source, total_lines=self.source.count('\n') + 1, file_size=len(self.source))

    def write_file(self, path, content, pre_content=None):
        self.writes.append(content)
        self.source = content
        return WriteResult(bytes_written=len(content), verified=True)


def apply(source, body):
    ops, error = parse_v4a_patch('*** Begin Patch\n*** Update File: f.py\n' + body + '\n*** End Patch\n')
    assert not error
    files = MemoryFiles(source)
    result = apply_v4a_operations(ops, files)
    return result, files


@pytest.mark.parametrize('gap', [0, 3000])
def test_repeated_hint_with_footer_is_atomic(gap):
    block = '# target\nvalue = old\n'
    source = block + ('# ' + 'x' * gap + '\n' if gap else '') + block + 'footer = old\n'
    result, files = apply(source, '@@ target @@\n-value = old\n+value = new\n@@\n-footer = old\n+footer = new')
    assert not result.success
    assert not files.writes
    assert files.source == source


def test_consumed_source_not_hidden_by_preexisting_replacement():
    source = 'keep = hostish_d\nhost = "a"\nhost = "b"\n'
    result, files = apply(source, '@@\n-host = "a"\n+host = "c"\n@@\n-host = "a"\n+keep = hostish_d')
    assert not result.success
    assert not files.writes
    assert files.source == source


@pytest.mark.parametrize('hint', ['second', 'old'])
def test_hint_context_may_overlap_cursor(hint):
    source = 'def second():\n    value = old\ndef third():\n    return 1\n'
    body = '@@\n def second():\n' + f'@@ {hint} @@\n def second():\n-    value = old\n+    value = new'
    result, files = apply(source, body)
    assert result.success, result.error
    assert files.source == source.replace('value = old', 'value = new')


def test_hint_selects_unique_partial_line():
    source = 'prefix AAA_UNRELATED\n' + 'x' * 3000 + '\nprefix SECRET_A_EDIT\n'
    result, files = apply(source, '@@ SECRET_A_EDIT @@\n-prefix \n+prefixZ')
    assert result.success, result.error
    assert files.source == source.replace('prefix SECRET_A_EDIT', 'prefixZSECRET_A_EDIT')
