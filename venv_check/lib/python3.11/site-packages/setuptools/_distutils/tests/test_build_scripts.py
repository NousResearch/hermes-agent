"""Tests for distutils.command.build_scripts."""

import os
import textwrap
from distutils import sysconfig
from distutils.command.build_scripts import build_scripts
from distutils.core import Distribution
from distutils.tests import support

import jaraco.path


class TestBuildScripts(support.TempdirManager):
    def test_default_settings(self):
        cmd = self.get_build_scripts_cmd("/foo/bar", [])
        assert not cmd.force
        assert cmd.build_dir is None

        cmd.finalize_options()

        assert cmd.force
        assert cmd.build_dir == "/foo/bar"

    def test_build(self):
        source = self.mkdtemp()
        target = self.mkdtemp()
        expected = self.write_sample_scripts(source)

        cmd = self.get_build_scripts_cmd(
            target, [os.path.join(source, fn) for fn in expected]
        )
        cmd.finalize_options()
        cmd.run()

        built = os.listdir(target)
        for name in expected:
            assert name in built

    def get_build_scripts_cmd(self, target, scripts):
        import sys

        dist = Distribution()
        dist.scripts = scripts
        dist.command_obj["build"] = support.DummyCommand(
            build_scripts=target, force=True, executable=sys.executable
        )
        return build_scripts(dist)

    @staticmethod
    def write_sample_scripts(dir):
        spec = {
            'script1.py': textwrap.dedent("""
                #! /usr/bin/env python2.3
                # bogus script w/ Python sh-bang
                pass
                """).lstrip(),
            'script2.py': textwrap.dedent("""
                #!/usr/bin/python
                # bogus script w/ Python sh-bang
                pass
                """).lstrip(),
            'shell.sh': textwrap.dedent("""
                #!/bin/sh
                # bogus shell script w/ sh-bang
                exit 0
                """).lstrip(),
        }
        jaraco.path.build(spec, dir)
        return list(spec)

    def test_version_int(self):
        source = self.mkdtemp()
        target = self.mkdtemp()
        expected = self.write_sample_scripts(source)

        cmd = self.get_build_scripts_cmd(
            target, [os.path.join(source, fn) for fn in expected]
        )
        cmd.finalize_options()

        # https://bugs.python.org/issue4524
        #
        # On linux-g++-32 with command line `./configure --enable-ipv6
        # --with-suffix=3`, python is compiled okay but the build scripts
        # failed when writing the name of the executable
        old = sysconfig.get_config_vars().get('VERSION')
        sysconfig._config_vars['VERSION'] = 4
        try:
            cmd.run()
        finally:
            if old is not None:
                sysconfig._config_vars['VERSION'] = old

        built = os.listdir(target)
        for name in expected:
            assert name in built
