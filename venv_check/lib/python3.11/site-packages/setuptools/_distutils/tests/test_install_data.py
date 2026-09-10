"""Tests for distutils.command.install_data."""

import os
import pathlib
from distutils.command.install_data import install_data
from distutils.tests import support

import pytest


@pytest.mark.usefixtures('save_env')
class TestInstallData(
    support.TempdirManager,
):
    def test_simple_run(self):
        pkg_dir, dist = self.create_dist()
        cmd = install_data(dist)
        cmd.install_dir = inst = os.path.join(pkg_dir, 'inst')

        # data_files can contain
        #  - simple files
        #  - a Path object
        #  - a tuple with a path, and a list of file
        one = os.path.join(pkg_dir, 'one')
        self.write_file(one, 'xxx')
        inst2 = os.path.join(pkg_dir, 'inst2')
        two = os.path.join(pkg_dir, 'two')
        self.write_file(two, 'xxx')
        three = pathlib.Path(pkg_dir) / 'three'
        self.write_file(three, 'xxx')

        cmd.data_files = [one, (inst2, [two]), three]
        assert cmd.get_inputs() == [one, (inst2, [two]), three]

        # let's run the command
        cmd.ensure_finalized()
        cmd.run()

        # let's check the result
        assert len(cmd.get_outputs()) == 3
        rthree = os.path.split(one)[-1]
        assert os.path.exists(os.path.join(inst, rthree))
        rtwo = os.path.split(two)[-1]
        assert os.path.exists(os.path.join(inst2, rtwo))
        rone = os.path.split(one)[-1]
        assert os.path.exists(os.path.join(inst, rone))
        cmd.outfiles = []

        # let's try with warn_dir one
        cmd.warn_dir = True
        cmd.ensure_finalized()
        cmd.run()

        # let's check the result
        assert len(cmd.get_outputs()) == 3
        assert os.path.exists(os.path.join(inst, rthree))
        assert os.path.exists(os.path.join(inst2, rtwo))
        assert os.path.exists(os.path.join(inst, rone))
        cmd.outfiles = []

        # now using root and empty dir
        cmd.root = os.path.join(pkg_dir, 'root')
        inst5 = os.path.join(pkg_dir, 'inst5')
        four = os.path.join(cmd.install_dir, 'four')
        self.write_file(four, 'xx')
        cmd.data_files = [one, (inst2, [two]), three, ('inst5', [four]), (inst5, [])]
        cmd.ensure_finalized()
        cmd.run()

        # let's check the result
        assert len(cmd.get_outputs()) == 5
        assert os.path.exists(os.path.join(inst, rthree))
        assert os.path.exists(os.path.join(inst2, rtwo))
        assert os.path.exists(os.path.join(inst, rone))
