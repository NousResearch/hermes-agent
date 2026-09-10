import os
import sys
import sysconfig
import threading
import unittest.mock as mock
from distutils.errors import DistutilsPlatformError
from distutils.tests import support
from distutils.util import get_platform

import pytest

from .. import msvc

needs_winreg = pytest.mark.skipif('not hasattr(msvc, "winreg")')


class Testmsvccompiler(support.TempdirManager):
    def test_no_compiler(self, monkeypatch):
        # makes sure query_vcvarsall raises
        # a DistutilsPlatformError if the compiler
        # is not found
        def _find_vcvarsall(plat_spec):
            return None, None

        monkeypatch.setattr(msvc, '_find_vcvarsall', _find_vcvarsall)

        with pytest.raises(DistutilsPlatformError):
            msvc._get_vc_env(
                'wont find this version',
            )

    @pytest.mark.skipif(
        not sysconfig.get_platform().startswith("win"),
        reason="Only run test for non-mingw Windows platforms",
    )
    @pytest.mark.parametrize(
        "plat_name, expected",
        [
            ("win-arm64", "win-arm64"),
            ("win-amd64", "win-amd64"),
            (None, get_platform()),
        ],
    )
    def test_cross_platform_compilation_paths(self, monkeypatch, plat_name, expected):
        """
        Ensure a specified target platform is passed to _get_vcvars_spec.
        """
        compiler = msvc.Compiler()

        def _get_vcvars_spec(host_platform, platform):
            assert platform == expected

        monkeypatch.setattr(msvc, '_get_vcvars_spec', _get_vcvars_spec)
        compiler.initialize(plat_name)

    @needs_winreg
    def test_get_vc_env_unicode(self):
        test_var = 'ṰḖṤṪ┅ṼẨṜ'
        test_value = '₃⁴₅'

        # Ensure we don't early exit from _get_vc_env
        old_distutils_use_sdk = os.environ.pop('DISTUTILS_USE_SDK', None)
        os.environ[test_var] = test_value
        try:
            env = msvc._get_vc_env('x86')
            assert test_var.lower() in env
            assert test_value == env[test_var.lower()]
        finally:
            os.environ.pop(test_var)
            if old_distutils_use_sdk:
                os.environ['DISTUTILS_USE_SDK'] = old_distutils_use_sdk

    @needs_winreg
    @pytest.mark.parametrize('ver', (2015, 2017))
    def test_get_vc(self, ver):
        # This function cannot be mocked, so pass if VC is found
        # and skip otherwise.
        lookup = getattr(msvc, f'_find_vc{ver}')
        expected_version = {2015: 14, 2017: 15}[ver]
        version, path = lookup()
        if not version:
            pytest.skip(f"VS {ver} is not installed")
        assert version >= expected_version
        assert os.path.isdir(path)


class CheckThread(threading.Thread):
    exc_info = None

    def run(self):
        try:
            super().run()
        except Exception:
            self.exc_info = sys.exc_info()

    def __bool__(self):
        return not self.exc_info


class TestSpawn:
    def test_concurrent_safe(self):
        """
        Concurrent calls to spawn should have consistent results.
        """
        compiler = msvc.Compiler()
        compiler._paths = "expected"
        inner_cmd = 'import os; assert os.environ["PATH"] == "expected"'
        command = [sys.executable, '-c', inner_cmd]

        threads = [
            CheckThread(target=compiler.spawn, args=[command]) for n in range(100)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert all(threads)

    def test_concurrent_safe_fallback(self):
        """
        If CCompiler.spawn has been monkey-patched without support
        for an env, it should still execute.
        """
        from distutils import ccompiler

        compiler = msvc.Compiler()
        compiler._paths = "expected"

        def CCompiler_spawn(self, cmd):
            "A spawn without an env argument."
            assert os.environ["PATH"] == "expected"

        with mock.patch.object(ccompiler.CCompiler, 'spawn', CCompiler_spawn):
            compiler.spawn(["n/a"])

        assert os.environ.get("PATH") != "expected"
