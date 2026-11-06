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


def test_displaced_hint_does_not_select_nearby_other_site():
    source = 'prefix SECRET_A_EDIT\n' + 'y' * 100 + '\nprefix SECRET\n'
    body = '@@\n-prefix SECRET_A\n+ZZSECRET_A\n@@ SECRET_A_EDIT @@\n-prefix SECRET\n+DONE'
    result, files = apply(source, body)
    assert not result.success
    assert files.source == source and not files.writes


def test_successive_hinted_hunks_do_not_fall_back_outside_window():
    source = '# marker\nvalue = old\n' + 'z' * 3000 + '\nvalue = old\n'
    body = '@@ marker @@\n-value = old\n+DONE\n@@ marker @@\n-value = old\n+OTHER'
    result, files = apply(source, body)
    assert not result.success
    assert files.source == source and not files.writes


def test_partial_consumption_does_not_fuzzy_write_a_different_block():
    source = 'keep = hostish_d\nfoo = 1\nbar = 2\nfoo = 1\nbar = 9\n'
    body = '@@\n-foo = 1\n+keep = hostish_d\n@@\n-foo = 1\n-bar = 2\n+keep = hostish_d'
    result, files = apply(source, body)
    assert not result.success
    assert files.source == source and not files.writes


def test_newly_written_context_may_be_used_as_hint():
    source = 'a = 1\nc = 3\n'
    body = '@@\n-a = 1\n+a = 1\n+b = 2\n@@ b = 2 @@\n b = 2\n-c = 3\n+c = 4'
    result, files = apply(source, body)
    assert result.success, result.error
    assert files.source == 'a = 1\nb = 2\nc = 4\n'


def test_empty_context_anchor_after_anchor_does_not_crash():
    from tools.patch_parser import Hunk, HunkLine, _plan_hunks

    plan = _plan_hunks('a\n\nb\n', [Hunk(lines=[HunkLine(' ', 'a')]), Hunk(lines=[HunkLine(' ', '')])])
    assert plan.errors == []


def test_blank_anchor_preserves_hunk_order():
    source = 'value = old\n\nvalue = old'
    body = '@@\n \n@@\n-value = old\n+FIRST\n@@\n-value = old\n+SECOND'
    result, files = apply(source, body)
    assert result.success, result.error
    assert files.source == 'SECOND\n\nFIRST'


def test_ordered_partial_hunks_refuse_ambiguous_substrings():
    source = 'NOTFOO\nreal = FOO\n'
    body = '@@\n-FOO\n+BAR\n@@\n-FOO\n+BAZ'
    result, files = apply(source, body)
    assert not result.success
    assert files.source == source and not files.writes


def test_hint_occurrence_outside_prior_write_still_usable():
    source = 'hostname = "edge-1"\nkeep = marker\nname = "edge-1"\n'
    body = '@@\n-hostname = "edge-1"\n+host = "marker"\n@@ marker @@\n-name = "edge-1"\n+name = "edge-2"'
    result, files = apply(source, body)
    assert result.success, result.error
    assert files.source == 'host = "marker"\nkeep = marker\nname = "edge-2"\n'


def test_long_search_containing_hint_can_be_ordered():
    line = 'line = ' + 'a' * 600
    source = line + '\nHINT\ntail = 1\n'
    body = '@@ HINT @@\n-' + line + '\n HINT\n+changed\n@@\n-tail = 1\n+tail = 2'
    result, files = apply(source, body)
    assert result.success, result.error
    assert files.source == 'HINT\nchanged\ntail = 2\n'


def test_partially_consumed_hinted_line_does_not_select_other_copy():
    source = 'prefix SECRET_A_EDIT\n' + 'y' * 100 + '\nSECRET_A_EDIT\n'
    body = '@@\n-prefix SECRET_A\n+ZZ\n@@ SECRET_A_EDIT @@\n-SECRET_A_EDIT\n+DONE'
    result, files = apply(source, body)
    assert not result.success
    assert files.source == source and not files.writes


def test_consumed_hint_does_not_target_substring_of_other_identifier():
    source = 'hostname = "edge-1"\nkeep = marker\nrename = "edge-1"\n'
    body = '@@\n-hostname = "edge-1"\n+host = "marker"\n@@ marker @@\n-name = "edge-1"\n+name = "edge-2"'
    result, files = apply(source, body)
    assert not result.success
    assert files.source == source and not files.writes


@pytest.mark.parametrize('later', [
    '@@ SECRET_A_EDIT @@\n SECRET_A_EDIT\n@@\n-near = old\n+near = new',
    '@@ SECRET_A_EDIT @@\n+INSERTED',
])
def test_damaged_hint_rejects_anchor_and_addition(later):
    source = 'prefix SECRET_A_EDIT\nnear = old\n' + 'y' * 100 + '\nSECRET_A_EDIT\nnear = old\n'
    body = '@@\n-prefix SECRET_A\n+ZZ\n' + later
    result, files = apply(source, body)
    assert not result.success
    assert files.source == source and not files.writes


def test_inserted_text_is_not_reedited_after_trimmed_eof():
    source = 'keep = 1\n\n\n\n'
    result, files = apply(source, '@@\n+X\n@@\n-X\n+Y')
    assert not result.success
    assert files.source == source and not files.writes
