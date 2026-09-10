"""Tests for distutils.spawn."""

import os
import stat
import sys
import unittest.mock as mock
from distutils.errors import DistutilsExecError
from distutils.spawn import find_executable, spawn
from distutils.tests import support

import path
import pytest
from test.support import unix_shell

from .compat import py39 as os_helper


class TestSpawn(support.TempdirManager):
    @pytest.mark.skipif("os.name not in ('nt', 'posix')")
    def test_spawn(self):
        tmpdir = self.mkdtemp()

        # creating something executable
        # through the shell that returns 1
        if sys.platform != 'win32':
            exe = os.path.join(tmpdir, 'foo.sh')
            self.write_file(exe, f'#!{unix_shell}\nexit 1')
        else:
            exe = os.path.join(tmpdir, 'foo.bat')
            self.write_file(exe, 'exit 1')

        os.chmod(exe, 0o777)
        with pytest.raises(DistutilsExecError):
            spawn([exe])

        # now something that works
        if sys.platform != 'win32':
            exe = os.path.join(tmpdir, 'foo.sh')
            self.write_file(exe, f'#!{unix_shell}\nexit 0')
        else:
            exe = os.path.join(tmpdir, 'foo.bat')
            self.write_file(exe, 'exit 0')

        os.chmod(exe, 0o777)
        spawn([exe])  # should work without any error

    def test_find_executable(self, tmp_path):
        program_path = self._make_executable(tmp_path, '.exe')
        program = program_path.name
        program_noeext = program_path.with_suffix('').name
        filename = str(program_path)
        tmp_dir = path.Path(tmp_path)

        # test path parameter
        rv = find_executable(program, path=tmp_dir)
        assert rv == filename

        if sys.platform == 'win32':
            # test without ".exe" extension
            rv = find_executable(program_noeext, path=tmp_dir)
            assert rv == filename

        # test find in the current directory
        with tmp_dir:
            rv = find_executable(program)
            assert rv == program

        # test non-existent program
        dont_exist_program = "dontexist_" + program
        rv = find_executable(dont_exist_program, path=tmp_dir)
        assert rv is None

        # PATH='': no match, except in the current directory
        with os_helper.EnvironmentVarGuard() as env:
            env['PATH'] = ''
            with (
                mock.patch(
                    'distutils.spawn.os.confstr', return_value=tmp_dir, create=True
                ),
                mock.patch('distutils.spawn.os.defpath', tmp_dir),
            ):
                rv = find_executable(program)
                assert rv is None

                # look in current directory
                with tmp_dir:
                    rv = find_executable(program)
                    assert rv == program

        # PATH=':': explicitly looks in the current directory
        with os_helper.EnvironmentVarGuard() as env:
            env['PATH'] = os.pathsep
            with (
                mock.patch('distutils.spawn.os.confstr', return_value='', create=True),
                mock.patch('distutils.spawn.os.defpath', ''),
            ):
                rv = find_executable(program)
                assert rv is None

                # look in current directory
                with tmp_dir:
                    rv = find_executable(program)
                    assert rv == program

        # missing PATH: test os.confstr("CS_PATH") and os.defpath
        with os_helper.EnvironmentVarGuard() as env:
            env.pop('PATH', None)

            # without confstr
            with (
                mock.patch(
                    'distutils.spawn.os.confstr', side_effect=ValueError, create=True
                ),
                mock.patch('distutils.spawn.os.defpath', tmp_dir),
            ):
                rv = find_executable(program)
                assert rv == filename

            # with confstr
            with (
                mock.patch(
                    'distutils.spawn.os.confstr', return_value=tmp_dir, create=True
                ),
                mock.patch('distutils.spawn.os.defpath', ''),
            ):
                rv = find_executable(program)
                assert rv == filename

    @staticmethod
    def _make_executable(tmp_path, ext):
        # Give the temporary program a suffix regardless of platform.
        # It's needed on Windows and not harmful on others.
        program = tmp_path.joinpath('program').with_suffix(ext)
        program.write_text("", encoding='utf-8')
        program.chmod(stat.S_IXUSR)
        return program

    def test_spawn_missing_exe(self):
        with pytest.raises(DistutilsExecError) as ctx:
            spawn(['does-not-exist'])
        assert "command 'does-not-exist' failed" in str(ctx.value)
