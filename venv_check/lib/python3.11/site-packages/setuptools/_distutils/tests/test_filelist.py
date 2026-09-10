"""Tests for distutils.filelist."""

import logging
import os
import re
from distutils import debug, filelist
from distutils.errors import DistutilsTemplateError
from distutils.filelist import FileList, glob_to_re, translate_pattern

import jaraco.path
import pytest

from .compat import py39 as os_helper

MANIFEST_IN = """\
include ok
include xo
exclude xo
include foo.tmp
include buildout.cfg
global-include *.x
global-include *.txt
global-exclude *.tmp
recursive-include f *.oo
recursive-exclude global *.x
graft dir
prune dir3
"""


def make_local_path(s):
    """Converts '/' in a string to os.sep"""
    return s.replace('/', os.sep)


class TestFileList:
    def assertNoWarnings(self, caplog):
        warnings = [rec for rec in caplog.records if rec.levelno == logging.WARNING]
        assert not warnings
        caplog.clear()

    def assertWarnings(self, caplog):
        warnings = [rec for rec in caplog.records if rec.levelno == logging.WARNING]
        assert warnings
        caplog.clear()

    def test_glob_to_re(self):
        sep = os.sep
        if os.sep == '\\':
            sep = re.escape(os.sep)

        for glob, regex in (
            # simple cases
            ('foo*', r'(?s:foo[^%(sep)s]*)\Z'),
            ('foo?', r'(?s:foo[^%(sep)s])\Z'),
            ('foo??', r'(?s:foo[^%(sep)s][^%(sep)s])\Z'),
            # special cases
            (r'foo\\*', r'(?s:foo\\\\[^%(sep)s]*)\Z'),
            (r'foo\\\*', r'(?s:foo\\\\\\[^%(sep)s]*)\Z'),
            ('foo????', r'(?s:foo[^%(sep)s][^%(sep)s][^%(sep)s][^%(sep)s])\Z'),
            (r'foo\\??', r'(?s:foo\\\\[^%(sep)s][^%(sep)s])\Z'),
        ):
            regex = regex % {'sep': sep}
            assert glob_to_re(glob) == regex

    def test_process_template_line(self):
        # testing  all MANIFEST.in template patterns
        file_list = FileList()
        mlp = make_local_path

        # simulated file list
        file_list.allfiles = [
            'foo.tmp',
            'ok',
            'xo',
            'four.txt',
            'buildout.cfg',
            # filelist does not filter out VCS directories,
            # it's sdist that does
            mlp('.hg/last-message.txt'),
            mlp('global/one.txt'),
            mlp('global/two.txt'),
            mlp('global/files.x'),
            mlp('global/here.tmp'),
            mlp('f/o/f.oo'),
            mlp('dir/graft-one'),
            mlp('dir/dir2/graft2'),
            mlp('dir3/ok'),
            mlp('dir3/sub/ok.txt'),
        ]

        for line in MANIFEST_IN.split('\n'):
            if line.strip() == '':
                continue
            file_list.process_template_line(line)

        wanted = [
            'ok',
            'buildout.cfg',
            'four.txt',
            mlp('.hg/last-message.txt'),
            mlp('global/one.txt'),
            mlp('global/two.txt'),
            mlp('f/o/f.oo'),
            mlp('dir/graft-one'),
            mlp('dir/dir2/graft2'),
        ]

        assert file_list.files == wanted

    def test_debug_print(self, capsys, monkeypatch):
        file_list = FileList()
        file_list.debug_print('xxx')
        assert capsys.readouterr().out == ''

        monkeypatch.setattr(debug, 'DEBUG', True)
        file_list.debug_print('xxx')
        assert capsys.readouterr().out == 'xxx\n'

    def test_set_allfiles(self):
        file_list = FileList()
        files = ['a', 'b', 'c']
        file_list.set_allfiles(files)
        assert file_list.allfiles == files

    def test_remove_duplicates(self):
        file_list = FileList()
        file_list.files = ['a', 'b', 'a', 'g', 'c', 'g']
        # files must be sorted beforehand (sdist does it)
        file_list.sort()
        file_list.remove_duplicates()
        assert file_list.files == ['a', 'b', 'c', 'g']

    def test_translate_pattern(self):
        # not regex
        assert hasattr(translate_pattern('a', anchor=True, is_regex=False), 'search')

        # is a regex
        regex = re.compile('a')
        assert translate_pattern(regex, anchor=True, is_regex=True) == regex

        # plain string flagged as regex
        assert hasattr(translate_pattern('a', anchor=True, is_regex=True), 'search')

        # glob support
        assert translate_pattern('*.py', anchor=True, is_regex=False).search(
            'filelist.py'
        )

    def test_exclude_pattern(self):
        # return False if no match
        file_list = FileList()
        assert not file_list.exclude_pattern('*.py')

        # return True if files match
        file_list = FileList()
        file_list.files = ['a.py', 'b.py']
        assert file_list.exclude_pattern('*.py')

        # test excludes
        file_list = FileList()
        file_list.files = ['a.py', 'a.txt']
        file_list.exclude_pattern('*.py')
        assert file_list.files == ['a.txt']

    def test_include_pattern(self):
        # return False if no match
        file_list = FileList()
        file_list.set_allfiles([])
        assert not file_list.include_pattern('*.py')

        # return True if files match
        file_list = FileList()
        file_list.set_allfiles(['a.py', 'b.txt'])
        assert file_list.include_pattern('*.py')

        # test * matches all files
        file_list = FileList()
        assert file_list.allfiles is None
        file_list.set_allfiles(['a.py', 'b.txt'])
        file_list.include_pattern('*')
        assert file_list.allfiles == ['a.py', 'b.txt']

    def test_process_template(self, caplog):
        mlp = make_local_path
        # invalid lines
        file_list = FileList()
        for action in (
            'include',
            'exclude',
            'global-include',
            'global-exclude',
            'recursive-include',
            'recursive-exclude',
            'graft',
            'prune',
            'blarg',
        ):
            with pytest.raises(DistutilsTemplateError):
                file_list.process_template_line(action)

        # include
        file_list = FileList()
        file_list.set_allfiles(['a.py', 'b.txt', mlp('d/c.py')])

        file_list.process_template_line('include *.py')
        assert file_list.files == ['a.py']
        self.assertNoWarnings(caplog)

        file_list.process_template_line('include *.rb')
        assert file_list.files == ['a.py']
        self.assertWarnings(caplog)

        # exclude
        file_list = FileList()
        file_list.files = ['a.py', 'b.txt', mlp('d/c.py')]

        file_list.process_template_line('exclude *.py')
        assert file_list.files == ['b.txt', mlp('d/c.py')]
        self.assertNoWarnings(caplog)

        file_list.process_template_line('exclude *.rb')
        assert file_list.files == ['b.txt', mlp('d/c.py')]
        self.assertWarnings(caplog)

        # global-include
        file_list = FileList()
        file_list.set_allfiles(['a.py', 'b.txt', mlp('d/c.py')])

        file_list.process_template_line('global-include *.py')
        assert file_list.files == ['a.py', mlp('d/c.py')]
        self.assertNoWarnings(caplog)

        file_list.process_template_line('global-include *.rb')
        assert file_list.files == ['a.py', mlp('d/c.py')]
        self.assertWarnings(caplog)

        # global-exclude
        file_list = FileList()
        file_list.files = ['a.py', 'b.txt', mlp('d/c.py')]

        file_list.process_template_line('global-exclude *.py')
        assert file_list.files == ['b.txt']
        self.assertNoWarnings(caplog)

        file_list.process_template_line('global-exclude *.rb')
        assert file_list.files == ['b.txt']
        self.assertWarnings(caplog)

        # recursive-include
        file_list = FileList()
        file_list.set_allfiles(['a.py', mlp('d/b.py'), mlp('d/c.txt'), mlp('d/d/e.py')])

        file_list.process_template_line('recursive-include d *.py')
        assert file_list.files == [mlp('d/b.py'), mlp('d/d/e.py')]
        self.assertNoWarnings(caplog)

        file_list.process_template_line('recursive-include e *.py')
        assert file_list.files == [mlp('d/b.py'), mlp('d/d/e.py')]
        self.assertWarnings(caplog)

        # recursive-exclude
        file_list = FileList()
        file_list.files = ['a.py', mlp('d/b.py'), mlp('d/c.txt'), mlp('d/d/e.py')]

        file_list.process_template_line('recursive-exclude d *.py')
        assert file_list.files == ['a.py', mlp('d/c.txt')]
        self.assertNoWarnings(caplog)

        file_list.process_template_line('recursive-exclude e *.py')
        assert file_list.files == ['a.py', mlp('d/c.txt')]
        self.assertWarnings(caplog)

        # graft
        file_list = FileList()
        file_list.set_allfiles(['a.py', mlp('d/b.py'), mlp('d/d/e.py'), mlp('f/f.py')])

        file_list.process_template_line('graft d')
        assert file_list.files == [mlp('d/b.py'), mlp('d/d/e.py')]
        self.assertNoWarnings(caplog)

        file_list.process_template_line('graft e')
        assert file_list.files == [mlp('d/b.py'), mlp('d/d/e.py')]
        self.assertWarnings(caplog)

        # prune
        file_list = FileList()
        file_list.files = ['a.py', mlp('d/b.py'), mlp('d/d/e.py'), mlp('f/f.py')]

        file_list.process_template_line('prune d')
        assert file_list.files == ['a.py', mlp('f/f.py')]
        self.assertNoWarnings(caplog)

        file_list.process_template_line('prune e')
        assert file_list.files == ['a.py', mlp('f/f.py')]
        self.assertWarnings(caplog)


class TestFindAll:
    @os_helper.skip_unless_symlink
    def test_missing_symlink(self, temp_cwd):
        os.symlink('foo', 'bar')
        assert filelist.findall() == []

    def test_basic_discovery(self, temp_cwd):
        """
        When findall is called with no parameters or with
        '.' as the parameter, the dot should be omitted from
        the results.
        """
        jaraco.path.build({'foo': {'file1.txt': ''}, 'bar': {'file2.txt': ''}})
        file1 = os.path.join('foo', 'file1.txt')
        file2 = os.path.join('bar', 'file2.txt')
        expected = [file2, file1]
        assert sorted(filelist.findall()) == expected

    def test_non_local_discovery(self, tmp_path):
        """
        When findall is called with another path, the full
        path name should be returned.
        """
        jaraco.path.build({'file1.txt': ''}, tmp_path)
        expected = [str(tmp_path / 'file1.txt')]
        assert filelist.findall(tmp_path) == expected

    @os_helper.skip_unless_symlink
    def test_symlink_loop(self, tmp_path):
        jaraco.path.build(
            {
                'link-to-parent': jaraco.path.Symlink('.'),
                'somefile': '',
            },
            tmp_path,
        )
        files = filelist.findall(tmp_path)
        assert len(files) == 1
