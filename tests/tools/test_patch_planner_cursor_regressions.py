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


def test_one_hint_with_two_sites_does_not_choose_first():
    source = '# marker\nvalue = old\nvalue = old\nfooter = old\n'
    result, files = apply(source, '@@ marker @@\n-value = old\n+value = new\n@@\n-footer = old\n+footer = new')
    assert not result.success
    assert not files.writes
    assert files.source == source


def test_normalized_consumed_source_is_not_suppressed():
    source = 'keep = hostish_d\nhost  = "a"\nhost = "b"\n'
    result, files = apply(source, '@@\n-host = "a"\n+host = "c"\n@@\n-host = "a"\n+keep = hostish_d')
    assert not result.success
    assert not files.writes
    assert files.source == source


def test_ambiguous_partial_hint_fails_atomically_without_exposing_hint():
    source = 'prefix AAA_UNRELATED\n' + 'x' * 100 + '\nprefix SECRET_A_EDIT\n'
    result, files = apply(source, '@@ SECRET_A_EDIT @@\n-prefix \n+prefixZ')
    assert not result.success
    assert not files.writes
    assert files.source == source
    assert result.error and 'SECRET_A_EDIT' not in result.error


def test_partial_hint_with_footer_keeps_the_hinted_site():
    source = 'prefix AAA_UNRELATED\n' + 'x' * 3000 + '\nprefix SECRET_A_EDIT\nfooter = old\n'
    body = '@@ SECRET_A_EDIT @@\n-prefix \n+prefixZ\n@@\n-footer = old\n+footer = new'
    result, files = apply(source, body)
    assert result.success, result.error
    assert files.source == source.replace('prefix SECRET_A_EDIT', 'prefixZSECRET_A_EDIT').replace('footer = old', 'footer = new')


def test_ambiguous_partial_search_does_not_gain_order_from_footer():
    source = 'prefix AAA\nprefix BBB\nfooter = old\n'
    body = '@@\n-prefix \n+prefixZ\n@@\n-footer = old\n+footer = new'
    result, files = apply(source, body)
    assert not result.success
    assert files.source == source and not files.writes


def test_later_nonadjacent_search_does_not_override_ambiguous_hint():
    source = '# marker\nprefix old\nprefix old\nfooter = old\n'
    body = '@@ marker @@\n-prefix \n+prefixZ\n@@\n-footer = old\n+footer = new\n@@\n-prefix \n+prefixQ'
    result, files = apply(source, body)
    assert not result.success
    assert files.source == source and not files.writes


def test_hint_with_two_sites_does_not_silently_target_after_anchor():
    source = '# SECRET\nvalue = old\n# anchor\nvalue = old\n'
    body = '@@\n # anchor\n@@ SECRET @@\n-value = old\n+value = new'
    result, files = apply(source, body)
    assert not result.success
    assert files.source == source and not files.writes


def test_consumed_part_of_multiline_hunk_not_hidden_by_replacement():
    source = 'keep = hostish_d\nhost = "a"\nhost = "b"\n'
    body = '@@\n-host = "a"\n+host = "c"\n@@\n-host = "a"\n-host = "b"\n+keep = hostish_d'
    result, files = apply(source, body)
    assert not result.success
    assert files.source == source and not files.writes


def test_blank_line_hunk_after_anchor_returns_result_without_exception():
    source = 'anchor\n\nend\n'
    result, files = apply(source, '@@\n anchor\n@@\n-\n+replacement')
    assert result is not None
    assert files.source in (source, 'anchor\nreplacement\nend\n')


def test_multiline_consumed_hunk_with_same_replacement_is_not_dropped():
    source = 'foo = 1\nbar = 2\n'
    body = '@@\n-foo = 1\n+keep = hostish_d\n@@\n-foo = 1\n-bar = 2\n+keep = hostish_d'
    result, files = apply(source, body)
    assert not result.success
    assert files.source == source and not files.writes


def test_missing_hinted_site_after_prior_edit_does_not_target_distant_match():
    source = 'prefix SECRET_A_EDIT\n' + 'y' * 3000 + '\nprefix SECRET_B_EDIT\n'
    body = '@@\n-prefix SECRET_A\n+ZZSECRET_A\n@@ SECRET_A_EDIT @@\n-prefix SECRET\n+DONE'
    result, files = apply(source, body)
    assert not result.success
    assert files.source == source and not files.writes
