"""Tests for distutils.command.build."""

import os
import sys
from distutils.command.build import build
from distutils.tests import support
from sysconfig import get_config_var, get_platform


class TestBuild(support.TempdirManager):
    def test_finalize_options(self):
        pkg_dir, dist = self.create_dist()
        cmd = build(dist)
        cmd.finalize_options()

        # if not specified, plat_name gets the current platform
        assert cmd.plat_name == get_platform()

        # build_purelib is build + lib
        wanted = os.path.join(cmd.build_base, 'lib')
        assert cmd.build_purelib == wanted

        # build_platlib is 'build/lib.platform-cache_tag[-pydebug]'
        # examples:
        #   build/lib.macosx-10.3-i386-cpython39
        plat_spec = f'.{cmd.plat_name}-{sys.implementation.cache_tag}'
        if get_config_var('Py_GIL_DISABLED'):
            plat_spec += 't'
        if hasattr(sys, 'gettotalrefcount'):
            assert cmd.build_platlib.endswith('-pydebug')
            plat_spec += '-pydebug'
        wanted = os.path.join(cmd.build_base, 'lib' + plat_spec)
        assert cmd.build_platlib == wanted

        # by default, build_lib = build_purelib
        assert cmd.build_lib == cmd.build_purelib

        # build_temp is build/temp.<plat>
        wanted = os.path.join(cmd.build_base, 'temp' + plat_spec)
        assert cmd.build_temp == wanted

        # build_scripts is build/scripts-x.x
        wanted = os.path.join(
            cmd.build_base, f'scripts-{sys.version_info.major}.{sys.version_info.minor}'
        )
        assert cmd.build_scripts == wanted

        # executable is os.path.normpath(sys.executable)
        assert cmd.executable == os.path.normpath(sys.executable)
