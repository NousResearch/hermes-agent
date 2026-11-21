"""Executable proposed V4A targeting contract; red cases are expected until repaired."""
import pytest

from tools.file_operations_common import ReadResult, WriteResult
from tools.patch_parser import apply_v4a_operations, parse_v4a_patch


class MemoryFiles:
    def __init__(self, files):
        self.files = dict(files)
        self.writes = []

    def read_file_raw(self, path):
        text = self.files[path]
        return ReadResult(content=text, total_lines=text.count('\n') + 1, file_size=len(text.encode()))

    def write_file(self, path, content, pre_content=None):
        self.writes.append((path, content))
        self.files[path] = content
        return WriteResult(bytes_written=len(content.encode()), verified=True)


def patch(body):
    return '*** Begin Patch\n*** Update File: f.py\n' + body + '\n*** End Patch\n'


CHANGE = '@@\n-value = 1\n+CHANGED'
FOOTER = '\n@@\n-footer = old\n+footer = new'
REPEAT = CHANGE + '\n@@\n-value = 1\n+OTHER'

# Every success row specifies exact output; lower-tier decoys must not mask
# a unique higher-tier candidate. Hint-selected partial matches remain valid.
SUCCESS = [
    ('exact-decoy-before', 'value = 10\nvalue = 1\nfooter = old\n', CHANGE + FOOTER, 'value = 10\nCHANGED\nfooter = new\n'),
    ('exact-decoy-after', 'value = 1\nvalue = 10\nfooter = old\n', CHANGE + FOOTER, 'CHANGED\nvalue = 10\nfooter = new\n'),
    ('normalized-tab-decoy', 'value = 10\nvalue\t=\t1\nfooter = old\n', CHANGE + FOOTER, 'value = 10\nCHANGED\nfooter = new\n'),
    ('normalized-space-decoy', 'value = 10\nvalue  =  1\nfooter = old\n', CHANGE + FOOTER, 'value = 10\nCHANGED\nfooter = new\n'),
    ('exact-over-normalized', 'value\t=\t1\nvalue = 1\n', CHANGE, 'value\t=\t1\nCHANGED\n'),
    ('unique-partial', 'call(value = 1)\nfooter = old\n', CHANGE + FOOTER, 'call(CHANGED)\nfooter = new\n'),
    ('unique-normalized-tab', 'value\t=\t1\nfooter = old\n', CHANGE + FOOTER, 'CHANGED\nfooter = new\n'),
    ('ordered-distinct', 'a = old\nb = old\n', '@@\n-a = old\n+a = new\n@@\n-b = old\n+b = new', 'a = new\nb = new\n'),
    ('hint-context-overlaps-cursor', 'def second():\n    value = old\ndef third():\n    return 1\n', '@@\n def second():\n@@ second @@\n def second():\n-    value = old\n+    value = new', 'def second():\n    value = new\ndef third():\n    return 1\n'),
    ('hint-partial-fallback', 'prefix AAA_UNRELATED\n' + 'x' * 3000 + '\nprefix SECRET_A_EDIT\n', '@@ SECRET_A_EDIT @@\n-prefix \n+prefixZ', 'prefix AAA_UNRELATED\n' + 'x' * 3000 + '\nprefixZSECRET_A_EDIT\n'),
    ('written-context-hint', 'a = 1\nc = 3\n', '@@\n-a = 1\n+a = 1\n+b = 2\n@@ b = 2 @@\n b = 2\n-c = 3\n+c = 4', 'a = 1\nb = 2\nc = 4\n'),
    ('addition-only-eof', 'first\n', '@@\n+last', 'first\nlast\n'),
    ('empty-anchor', 'a\n\nb\n', '@@\n a\n@@\n \n@@\n-b\n+B', 'a\n\nB\n'),
]

# A refusal must preserve the original bytes, not merely report an error.
REFUSE = [
    ('repeat-consumed-prefix', 'value = 1\nvalue = 10\n', REPEAT),
    ('repeat-consumed-prefix-footer', 'value = 1\nvalue = 10\nfooter = old\n', REPEAT + FOOTER),
    ('repeated-identical-no-hint', 'value = 1\nvalue = 1\n', REPEAT),
    ('repeated-ambiguous-hint-footer', '# target\nvalue = old\n# target\nvalue = old\nfooter = old\n', '@@ target @@\n-value = old\n+value = new\n@@\n-footer = old\n+footer = new'),
    ('consumed-replacement-already-present', 'keep = hostish_d\nhost = "a"\nhost = "b"\n', '@@\n-host = "a"\n+host = "c"\n@@\n-host = "a"\n+keep = hostish_d'),
    ('consumed-multiline-replacement-present', 'foo = 1\nbar = 2\n', '@@\n-foo = 1\n+keep = hostish_d\n@@\n-foo = 1\n-bar = 2\n+keep = hostish_d'),
    ('ambiguous-partials', 'prefix AAA\nprefix BBB\nfooter = old\n', '@@\n-prefix \n+prefixZ\n@@\n-footer = old\n+footer = new'),
    ('hint-cannot-bypass-tier', 'value = 1\n' + 'x' * 3000 + '\n# marker\nvalue = 10\n', '@@ marker @@\n-value = 1\n+CHANGED'),
    ('missing-second-hunk-atomic', 'a = old\nb = old\n', '@@\n-a = old\n+a = new\n@@\n-missing = old\n+missing = new'),
    ('one-hint-two-sites', '# marker\nvalue = old\nvalue = old\nfooter = old\n', '@@ marker @@\n-value = old\n+value = new\n@@\n-footer = old\n+footer = new'),
]


@pytest.mark.parametrize('name,source,body,expected', SUCCESS, ids=[row[0] for row in SUCCESS])
def test_success(name, source, body, expected):
    ops, error = parse_v4a_patch(patch(body))
    assert not error, error
    files = MemoryFiles({'f.py': source})
    result = apply_v4a_operations(ops, files)
    assert result.success, result.error
    assert files.files['f.py'].encode() == expected.encode()
    assert files.writes == [('f.py', expected)]


@pytest.mark.parametrize('name,source,body', REFUSE, ids=[row[0] for row in REFUSE])
def test_refusal(name, source, body):
    ops, error = parse_v4a_patch(patch(body))
    assert not error, error
    files = MemoryFiles({'f.py': source})
    result = apply_v4a_operations(ops, files)
    assert not result.success, f'unexpected success: {files.files!r}'
    assert files.files['f.py'].encode() == source.encode()
    assert not files.writes


def test_cross_file_refusal_is_atomic():
    text = ('*** Begin Patch\n*** Update File: a.py\n@@\n-old\n+new\n'
            '*** Update File: b.py\n@@\n-absent\n+replacement\n*** End Patch\n')
    ops, error = parse_v4a_patch(text)
    assert not error, error
    before = {'a.py': 'old\n', 'b.py': 'other\n'}
    files = MemoryFiles(before)
    result = apply_v4a_operations(ops, files)
    assert not result.success
    assert files.files == before
    assert not files.writes
