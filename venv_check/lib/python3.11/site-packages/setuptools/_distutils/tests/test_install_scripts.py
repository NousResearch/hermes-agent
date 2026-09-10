"""Tests for distutils.command.install_scripts."""

import os
from distutils.command.install_scripts import install_scripts
from distutils.core import Distribution
from distutils.tests import support

from . import test_build_scripts


class TestInstallScripts(support.TempdirManager):
    def test_default_settings(self):
        dist = Distribution()
        dist.command_obj["build"] = support.DummyCommand(build_scripts="/foo/bar")
        dist.command_obj["install"] = support.DummyCommand(
            install_scripts="/splat/funk",
            force=True,
            skip_build=True,
        )
        cmd = install_scripts(dist)
        assert not cmd.force
        assert not cmd.skip_build
        assert cmd.build_dir is None
        assert cmd.install_dir is None

        cmd.finalize_options()

        assert cmd.force
        assert cmd.skip_build
        assert cmd.build_dir == "/foo/bar"
        assert cmd.install_dir == "/splat/funk"

    def test_installation(self):
        source = self.mkdtemp()

        expected = test_build_scripts.TestBuildScripts.write_sample_scripts(source)

        target = self.mkdtemp()
        dist = Distribution()
        dist.command_obj["build"] = support.DummyCommand(build_scripts=source)
        dist.command_obj["install"] = support.DummyCommand(
            install_scripts=target,
            force=True,
            skip_build=True,
        )
        cmd = install_scripts(dist)
        cmd.finalize_options()
        cmd.run()

        installed = os.listdir(target)
        for name in expected:
            assert name in installed
