"""Tests for distutils.command.install_data."""

import importlib.util
import os
import sys
from distutils.command.install_lib import install_lib
from distutils.errors import DistutilsOptionError
from distutils.extension import Extension
from distutils.tests import support

import pytest


@support.combine_markers
@pytest.mark.usefixtures('save_env')
class TestInstallLib(
    support.TempdirManager,
):
    def test_finalize_options(self):
        dist = self.create_dist()[1]
        cmd = install_lib(dist)

        cmd.finalize_options()
        assert cmd.compile == 1
        assert cmd.optimize == 0

        # optimize must be 0, 1, or 2
        cmd.optimize = 'foo'
        with pytest.raises(DistutilsOptionError):
            cmd.finalize_options()
        cmd.optimize = '4'
        with pytest.raises(DistutilsOptionError):
            cmd.finalize_options()

        cmd.optimize = '2'
        cmd.finalize_options()
        assert cmd.optimize == 2

    @pytest.mark.skipif('sys.dont_write_bytecode')
    def test_byte_compile(self):
        project_dir, dist = self.create_dist()
        os.chdir(project_dir)
        cmd = install_lib(dist)
        cmd.compile = cmd.optimize = 1

        f = os.path.join(project_dir, 'foo.py')
        self.write_file(f, '# python file')
        cmd.byte_compile([f])
        pyc_file = importlib.util.cache_from_source('foo.py', optimization='')
        pyc_opt_file = importlib.util.cache_from_source(
            'foo.py', optimization=cmd.optimize
        )
        assert os.path.exists(pyc_file)
        assert os.path.exists(pyc_opt_file)

    def test_get_outputs(self):
        project_dir, dist = self.create_dist()
        os.chdir(project_dir)
        os.mkdir('spam')
        cmd = install_lib(dist)

        # setting up a dist environment
        cmd.compile = cmd.optimize = 1
        cmd.install_dir = self.mkdtemp()
        f = os.path.join(project_dir, 'spam', '__init__.py')
        self.write_file(f, '# python package')
        cmd.distribution.ext_modules = [Extension('foo', ['xxx'])]
        cmd.distribution.packages = ['spam']
        cmd.distribution.script_name = 'setup.py'

        # get_outputs should return 4 elements: spam/__init__.py and .pyc,
        # foo.import-tag-abiflags.so / foo.pyd
        outputs = cmd.get_outputs()
        assert len(outputs) == 4, outputs

    def test_get_inputs(self):
        project_dir, dist = self.create_dist()
        os.chdir(project_dir)
        os.mkdir('spam')
        cmd = install_lib(dist)

        # setting up a dist environment
        cmd.compile = cmd.optimize = 1
        cmd.install_dir = self.mkdtemp()
        f = os.path.join(project_dir, 'spam', '__init__.py')
        self.write_file(f, '# python package')
        cmd.distribution.ext_modules = [Extension('foo', ['xxx'])]
        cmd.distribution.packages = ['spam']
        cmd.distribution.script_name = 'setup.py'

        # get_inputs should return 2 elements: spam/__init__.py and
        # foo.import-tag-abiflags.so / foo.pyd
        inputs = cmd.get_inputs()
        assert len(inputs) == 2, inputs

    def test_dont_write_bytecode(self, caplog):
        # makes sure byte_compile is not used
        dist = self.create_dist()[1]
        cmd = install_lib(dist)
        cmd.compile = True
        cmd.optimize = 1

        old_dont_write_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            cmd.byte_compile([])
        finally:
            sys.dont_write_bytecode = old_dont_write_bytecode

        assert 'byte-compiling is disabled' in caplog.messages[0]
