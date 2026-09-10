"""Tests for distutils.cmd."""

import os
from distutils import debug
from distutils.cmd import Command
from distutils.dist import Distribution
from distutils.errors import DistutilsOptionError

import pytest


class MyCmd(Command):
    def initialize_options(self):
        pass


@pytest.fixture
def cmd(request):
    return MyCmd(Distribution())


class TestCommand:
    def test_ensure_string_list(self, cmd):
        cmd.not_string_list = ['one', 2, 'three']
        cmd.yes_string_list = ['one', 'two', 'three']
        cmd.not_string_list2 = object()
        cmd.yes_string_list2 = 'ok'
        cmd.ensure_string_list('yes_string_list')
        cmd.ensure_string_list('yes_string_list2')

        with pytest.raises(DistutilsOptionError):
            cmd.ensure_string_list('not_string_list')

        with pytest.raises(DistutilsOptionError):
            cmd.ensure_string_list('not_string_list2')

        cmd.option1 = 'ok,dok'
        cmd.ensure_string_list('option1')
        assert cmd.option1 == ['ok', 'dok']

        cmd.option2 = ['xxx', 'www']
        cmd.ensure_string_list('option2')

        cmd.option3 = ['ok', 2]
        with pytest.raises(DistutilsOptionError):
            cmd.ensure_string_list('option3')

    def test_make_file(self, cmd):
        # making sure it raises when infiles is not a string or a list/tuple
        with pytest.raises(TypeError):
            cmd.make_file(infiles=True, outfile='', func='func', args=())

        # making sure execute gets called properly
        def _execute(func, args, exec_msg, level):
            assert exec_msg == 'generating out from in'

        cmd.force = True
        cmd.execute = _execute
        cmd.make_file(infiles='in', outfile='out', func='func', args=())

    def test_dump_options(self, cmd):
        msgs = []

        def _announce(msg, level):
            msgs.append(msg)

        cmd.announce = _announce
        cmd.option1 = 1
        cmd.option2 = 1
        cmd.user_options = [('option1', '', ''), ('option2', '', '')]
        cmd.dump_options()

        wanted = ["command options for 'MyCmd':", '  option1 = 1', '  option2 = 1']
        assert msgs == wanted

    def test_ensure_string(self, cmd):
        cmd.option1 = 'ok'
        cmd.ensure_string('option1')

        cmd.option2 = None
        cmd.ensure_string('option2', 'xxx')
        assert hasattr(cmd, 'option2')

        cmd.option3 = 1
        with pytest.raises(DistutilsOptionError):
            cmd.ensure_string('option3')

    def test_ensure_filename(self, cmd):
        cmd.option1 = __file__
        cmd.ensure_filename('option1')
        cmd.option2 = 'xxx'
        with pytest.raises(DistutilsOptionError):
            cmd.ensure_filename('option2')

    def test_ensure_dirname(self, cmd):
        cmd.option1 = os.path.dirname(__file__) or os.curdir
        cmd.ensure_dirname('option1')
        cmd.option2 = 'xxx'
        with pytest.raises(DistutilsOptionError):
            cmd.ensure_dirname('option2')

    def test_debug_print(self, cmd, capsys, monkeypatch):
        cmd.debug_print('xxx')
        assert capsys.readouterr().out == ''
        monkeypatch.setattr(debug, 'DEBUG', True)
        cmd.debug_print('xxx')
        assert capsys.readouterr().out == 'xxx\n'
