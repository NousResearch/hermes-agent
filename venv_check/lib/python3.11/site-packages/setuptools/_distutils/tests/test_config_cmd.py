"""Tests for distutils.command.config."""

import os
import sys
from distutils._log import log
from distutils.command.config import config, dump_file
from distutils.tests import missing_compiler_executable, support

import more_itertools
import path
import pytest


@pytest.fixture(autouse=True)
def info_log(request, monkeypatch):
    self = request.instance
    self._logs = []
    monkeypatch.setattr(log, 'info', self._info)


@support.combine_markers
class TestConfig(support.TempdirManager):
    def _info(self, msg, *args):
        for line in msg.splitlines():
            self._logs.append(line)

    def test_dump_file(self):
        this_file = path.Path(__file__).with_suffix('.py')
        with this_file.open(encoding='utf-8') as f:
            numlines = more_itertools.ilen(f)

        dump_file(this_file, 'I am the header')
        assert len(self._logs) == numlines + 1

    @pytest.mark.skipif('platform.system() == "Windows"')
    def test_search_cpp(self):
        cmd = missing_compiler_executable(['preprocessor'])
        if cmd is not None:
            self.skipTest(f'The {cmd!r} command is not found')
        pkg_dir, dist = self.create_dist()
        cmd = config(dist)
        cmd._check_compiler()
        compiler = cmd.compiler
        if sys.platform[:3] == "aix" and "xlc" in compiler.preprocessor[0].lower():
            self.skipTest(
                'xlc: The -E option overrides the -P, -o, and -qsyntaxonly options'
            )

        # simple pattern searches
        match = cmd.search_cpp(pattern='xxx', body='/* xxx */')
        assert match == 0

        match = cmd.search_cpp(pattern='_configtest', body='/* xxx */')
        assert match == 1

    def test_finalize_options(self):
        # finalize_options does a bit of transformation
        # on options
        pkg_dir, dist = self.create_dist()
        cmd = config(dist)
        cmd.include_dirs = f'one{os.pathsep}two'
        cmd.libraries = 'one'
        cmd.library_dirs = f'three{os.pathsep}four'
        cmd.ensure_finalized()

        assert cmd.include_dirs == ['one', 'two']
        assert cmd.libraries == ['one']
        assert cmd.library_dirs == ['three', 'four']

    def test_clean(self):
        # _clean removes files
        tmp_dir = self.mkdtemp()
        f1 = os.path.join(tmp_dir, 'one')
        f2 = os.path.join(tmp_dir, 'two')

        self.write_file(f1, 'xxx')
        self.write_file(f2, 'xxx')

        for f in (f1, f2):
            assert os.path.exists(f)

        pkg_dir, dist = self.create_dist()
        cmd = config(dist)
        cmd._clean(f1, f2)

        for f in (f1, f2):
            assert not os.path.exists(f)
