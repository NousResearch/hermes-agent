"""Tests for distutils.unixccompiler."""

import os
import sys
import unittest.mock as mock
from distutils import sysconfig
from distutils.compat import consolidate_linker_args
from distutils.errors import DistutilsPlatformError
from distutils.tests import support
from distutils.tests.compat.py39 import EnvironmentVarGuard
from distutils.util import _clear_cached_macosx_ver

import pytest

from .. import unix


@pytest.fixture(autouse=True)
def save_values(monkeypatch):
    monkeypatch.setattr(sys, 'platform', sys.platform)
    monkeypatch.setattr(sysconfig, 'get_config_var', sysconfig.get_config_var)
    monkeypatch.setattr(sysconfig, 'get_config_vars', sysconfig.get_config_vars)


@pytest.fixture(autouse=True)
def compiler_wrapper(request):
    class CompilerWrapper(unix.Compiler):
        def rpath_foo(self):
            return self.runtime_library_dir_option('/foo')

    request.instance.cc = CompilerWrapper()


class TestUnixCCompiler(support.TempdirManager):
    @pytest.mark.skipif('platform.system == "Windows"')
    def test_runtime_libdir_option(self):  # noqa: C901
        # Issue #5900; GitHub Issue #37
        #
        # Ensure RUNPATH is added to extension modules with RPATH if
        # GNU ld is used

        # darwin
        sys.platform = 'darwin'
        darwin_ver_var = 'MACOSX_DEPLOYMENT_TARGET'
        darwin_rpath_flag = '-Wl,-rpath,/foo'
        darwin_lib_flag = '-L/foo'

        # (macOS version from syscfg, macOS version from env var) -> flag
        # Version value of None generates two tests: as None and as empty string
        # Expected flag value of None means an mismatch exception is expected
        darwin_test_cases = [
            ((None, None), darwin_lib_flag),
            ((None, '11'), darwin_rpath_flag),
            (('10', None), darwin_lib_flag),
            (('10.3', None), darwin_lib_flag),
            (('10.3.1', None), darwin_lib_flag),
            (('10.5', None), darwin_rpath_flag),
            (('10.5.1', None), darwin_rpath_flag),
            (('10.3', '10.3'), darwin_lib_flag),
            (('10.3', '10.5'), darwin_rpath_flag),
            (('10.5', '10.3'), darwin_lib_flag),
            (('10.5', '11'), darwin_rpath_flag),
            (('10.4', '10'), None),
        ]

        def make_darwin_gcv(syscfg_macosx_ver):
            def gcv(var):
                if var == darwin_ver_var:
                    return syscfg_macosx_ver
                return "xxx"

            return gcv

        def do_darwin_test(syscfg_macosx_ver, env_macosx_ver, expected_flag):
            env = os.environ
            msg = f"macOS version = (sysconfig={syscfg_macosx_ver!r}, env={env_macosx_ver!r})"

            # Save
            old_gcv = sysconfig.get_config_var
            old_env_macosx_ver = env.get(darwin_ver_var)

            # Setup environment
            _clear_cached_macosx_ver()
            sysconfig.get_config_var = make_darwin_gcv(syscfg_macosx_ver)
            if env_macosx_ver is not None:
                env[darwin_ver_var] = env_macosx_ver
            elif darwin_ver_var in env:
                env.pop(darwin_ver_var)

            # Run the test
            if expected_flag is not None:
                assert self.cc.rpath_foo() == expected_flag, msg
            else:
                with pytest.raises(
                    DistutilsPlatformError, match=darwin_ver_var + r' mismatch'
                ):
                    self.cc.rpath_foo()

            # Restore
            if old_env_macosx_ver is not None:
                env[darwin_ver_var] = old_env_macosx_ver
            elif darwin_ver_var in env:
                env.pop(darwin_ver_var)
            sysconfig.get_config_var = old_gcv
            _clear_cached_macosx_ver()

        for macosx_vers, expected_flag in darwin_test_cases:
            syscfg_macosx_ver, env_macosx_ver = macosx_vers
            do_darwin_test(syscfg_macosx_ver, env_macosx_ver, expected_flag)
            # Bonus test cases with None interpreted as empty string
            if syscfg_macosx_ver is None:
                do_darwin_test("", env_macosx_ver, expected_flag)
            if env_macosx_ver is None:
                do_darwin_test(syscfg_macosx_ver, "", expected_flag)
            if syscfg_macosx_ver is None and env_macosx_ver is None:
                do_darwin_test("", "", expected_flag)

        old_gcv = sysconfig.get_config_var

        # hp-ux
        sys.platform = 'hp-ux'

        def gcv(v):
            return 'xxx'

        sysconfig.get_config_var = gcv
        assert self.cc.rpath_foo() == ['+s', '-L/foo']

        def gcv(v):
            return 'gcc'

        sysconfig.get_config_var = gcv
        assert self.cc.rpath_foo() == ['-Wl,+s', '-L/foo']

        def gcv(v):
            return 'g++'

        sysconfig.get_config_var = gcv
        assert self.cc.rpath_foo() == ['-Wl,+s', '-L/foo']

        sysconfig.get_config_var = old_gcv

        # GCC GNULD
        sys.platform = 'bar'

        def gcv(v):
            if v == 'CC':
                return 'gcc'
            elif v == 'GNULD':
                return 'yes'

        sysconfig.get_config_var = gcv
        assert self.cc.rpath_foo() == consolidate_linker_args([
            '-Wl,--enable-new-dtags',
            '-Wl,-rpath,/foo',
        ])

        def gcv(v):
            if v == 'CC':
                return 'gcc -pthread -B /bar'
            elif v == 'GNULD':
                return 'yes'

        sysconfig.get_config_var = gcv
        assert self.cc.rpath_foo() == consolidate_linker_args([
            '-Wl,--enable-new-dtags',
            '-Wl,-rpath,/foo',
        ])

        # GCC non-GNULD
        sys.platform = 'bar'

        def gcv(v):
            if v == 'CC':
                return 'gcc'
            elif v == 'GNULD':
                return 'no'

        sysconfig.get_config_var = gcv
        assert self.cc.rpath_foo() == '-Wl,-R/foo'

        # GCC GNULD with fully qualified configuration prefix
        # see #7617
        sys.platform = 'bar'

        def gcv(v):
            if v == 'CC':
                return 'x86_64-pc-linux-gnu-gcc-4.4.2'
            elif v == 'GNULD':
                return 'yes'

        sysconfig.get_config_var = gcv
        assert self.cc.rpath_foo() == consolidate_linker_args([
            '-Wl,--enable-new-dtags',
            '-Wl,-rpath,/foo',
        ])

        # non-GCC GNULD
        sys.platform = 'bar'

        def gcv(v):
            if v == 'CC':
                return 'cc'
            elif v == 'GNULD':
                return 'yes'

        sysconfig.get_config_var = gcv
        assert self.cc.rpath_foo() == consolidate_linker_args([
            '-Wl,--enable-new-dtags',
            '-Wl,-rpath,/foo',
        ])

        # non-GCC non-GNULD
        sys.platform = 'bar'

        def gcv(v):
            if v == 'CC':
                return 'cc'
            elif v == 'GNULD':
                return 'no'

        sysconfig.get_config_var = gcv
        assert self.cc.rpath_foo() == '-Wl,-R/foo'

    @pytest.mark.skipif('platform.system == "Windows"')
    def test_cc_overrides_ldshared(self):
        # Issue #18080:
        # ensure that setting CC env variable also changes default linker
        def gcv(v):
            if v == 'LDSHARED':
                return 'gcc-4.2 -bundle -undefined dynamic_lookup '
            return 'gcc-4.2'

        def gcvs(*args, _orig=sysconfig.get_config_vars):
            if args:
                return list(map(sysconfig.get_config_var, args))
            return _orig()

        sysconfig.get_config_var = gcv
        sysconfig.get_config_vars = gcvs
        with EnvironmentVarGuard() as env:
            env['CC'] = 'my_cc'
            del env['LDSHARED']
            sysconfig.customize_compiler(self.cc)
        assert self.cc.linker_so[0] == 'my_cc'

    @pytest.mark.skipif('platform.system == "Windows"')
    def test_cxx_commands_used_are_correct(self):
        def gcv(v):
            if v == 'LDSHARED':
                return 'ccache gcc-4.2 -bundle -undefined dynamic_lookup'
            elif v == 'LDCXXSHARED':
                return 'ccache g++-4.2 -bundle -undefined dynamic_lookup'
            elif v == 'CXX':
                return 'ccache g++-4.2'
            elif v == 'CC':
                return 'ccache gcc-4.2'
            return ''

        def gcvs(*args, _orig=sysconfig.get_config_vars):
            if args:
                return list(map(sysconfig.get_config_var, args))
            return _orig()  # pragma: no cover

        sysconfig.get_config_var = gcv
        sysconfig.get_config_vars = gcvs
        with (
            mock.patch.object(self.cc, 'spawn', return_value=None) as mock_spawn,
            mock.patch.object(self.cc, '_need_link', return_value=True),
            mock.patch.object(self.cc, 'mkpath', return_value=None),
            EnvironmentVarGuard() as env,
        ):
            # override environment overrides in case they're specified by CI
            del env['CXX']
            del env['LDCXXSHARED']

            sysconfig.customize_compiler(self.cc)
            assert self.cc.linker_so_cxx[0:2] == ['ccache', 'g++-4.2']
            assert self.cc.linker_exe_cxx[0:2] == ['ccache', 'g++-4.2']
            self.cc.link(None, [], 'a.out', target_lang='c++')
            call_args = mock_spawn.call_args[0][0]
            expected = ['ccache', 'g++-4.2', '-bundle', '-undefined', 'dynamic_lookup']
            assert call_args[:5] == expected

            self.cc.link_executable([], 'a.out', target_lang='c++')
            call_args = mock_spawn.call_args[0][0]
            expected = ['ccache', 'g++-4.2', '-o', self.cc.executable_filename('a.out')]
            assert call_args[:4] == expected

            env['LDCXXSHARED'] = 'wrapper g++-4.2 -bundle -undefined dynamic_lookup'
            env['CXX'] = 'wrapper g++-4.2'
            sysconfig.customize_compiler(self.cc)
            assert self.cc.linker_so_cxx[0:2] == ['wrapper', 'g++-4.2']
            assert self.cc.linker_exe_cxx[0:2] == ['wrapper', 'g++-4.2']
            self.cc.link(None, [], 'a.out', target_lang='c++')
            call_args = mock_spawn.call_args[0][0]
            expected = ['wrapper', 'g++-4.2', '-bundle', '-undefined', 'dynamic_lookup']
            assert call_args[:5] == expected

            self.cc.link_executable([], 'a.out', target_lang='c++')
            call_args = mock_spawn.call_args[0][0]
            expected = [
                'wrapper',
                'g++-4.2',
                '-o',
                self.cc.executable_filename('a.out'),
            ]
            assert call_args[:4] == expected

    @pytest.mark.skipif('platform.system == "Windows"')
    @pytest.mark.usefixtures('disable_macos_customization')
    def test_cc_overrides_ldshared_for_cxx_correctly(self):
        """
        Ensure that setting CC env variable also changes default linker
        correctly when building C++ extensions.

        pypa/distutils#126
        """

        def gcv(v):
            if v == 'LDSHARED':
                return 'gcc-4.2 -bundle -undefined dynamic_lookup '
            elif v == 'LDCXXSHARED':
                return 'g++-4.2 -bundle -undefined dynamic_lookup '
            elif v == 'CXX':
                return 'g++-4.2'
            elif v == 'CC':
                return 'gcc-4.2'
            return ''

        def gcvs(*args, _orig=sysconfig.get_config_vars):
            if args:
                return list(map(sysconfig.get_config_var, args))
            return _orig()

        sysconfig.get_config_var = gcv
        sysconfig.get_config_vars = gcvs
        with (
            mock.patch.object(self.cc, 'spawn', return_value=None) as mock_spawn,
            mock.patch.object(self.cc, '_need_link', return_value=True),
            mock.patch.object(self.cc, 'mkpath', return_value=None),
            EnvironmentVarGuard() as env,
        ):
            env['CC'] = 'ccache my_cc'
            env['CXX'] = 'my_cxx'
            del env['LDSHARED']
            sysconfig.customize_compiler(self.cc)
            assert self.cc.linker_so[0:2] == ['ccache', 'my_cc']
            self.cc.link(None, [], 'a.out', target_lang='c++')
            call_args = mock_spawn.call_args[0][0]
            expected = ['my_cxx', '-bundle', '-undefined', 'dynamic_lookup']
            assert call_args[:4] == expected

    @pytest.mark.skipif('platform.system == "Windows"')
    def test_explicit_ldshared(self):
        # Issue #18080:
        # ensure that setting CC env variable does not change
        #   explicit LDSHARED setting for linker
        def gcv(v):
            if v == 'LDSHARED':
                return 'gcc-4.2 -bundle -undefined dynamic_lookup '
            return 'gcc-4.2'

        def gcvs(*args, _orig=sysconfig.get_config_vars):
            if args:
                return list(map(sysconfig.get_config_var, args))
            return _orig()

        sysconfig.get_config_var = gcv
        sysconfig.get_config_vars = gcvs
        with EnvironmentVarGuard() as env:
            env['CC'] = 'my_cc'
            env['LDSHARED'] = 'my_ld -bundle -dynamic'
            sysconfig.customize_compiler(self.cc)
        assert self.cc.linker_so[0] == 'my_ld'

    def test_has_function(self):
        # Issue https://github.com/pypa/distutils/issues/64:
        # ensure that setting output_dir does not raise
        # FileNotFoundError: [Errno 2] No such file or directory: 'a.out'
        self.cc.output_dir = 'scratch'
        os.chdir(self.mkdtemp())
        self.cc.has_function('abort')

    def test_find_library_file(self, monkeypatch):
        compiler = unix.Compiler()
        compiler._library_root = lambda dir: dir
        monkeypatch.setattr(os.path, 'exists', lambda d: 'existing' in d)

        libname = 'libabc.dylib' if sys.platform != 'cygwin' else 'cygabc.dll'
        dirs = ('/foo/bar/missing', '/foo/bar/existing')
        assert (
            compiler.find_library_file(dirs, 'abc').replace('\\', '/')
            == f'/foo/bar/existing/{libname}'
        )
        assert (
            compiler.find_library_file(reversed(dirs), 'abc').replace('\\', '/')
            == f'/foo/bar/existing/{libname}'
        )

        monkeypatch.setattr(
            os.path,
            'exists',
            lambda d: 'existing' in d and '.a' in d and '.dll.a' not in d,
        )
        assert (
            compiler.find_library_file(dirs, 'abc').replace('\\', '/')
            == '/foo/bar/existing/libabc.a'
        )
        assert (
            compiler.find_library_file(reversed(dirs), 'abc').replace('\\', '/')
            == '/foo/bar/existing/libabc.a'
        )
