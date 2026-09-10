"""Tests for distutils.cygwinccompiler."""

import os
import sys
from distutils import sysconfig
from distutils.tests import support

import pytest

from .. import cygwin


@pytest.fixture(autouse=True)
def stuff(request, monkeypatch, distutils_managed_tempdir):
    self = request.instance
    self.python_h = os.path.join(self.mkdtemp(), 'python.h')
    monkeypatch.setattr(sysconfig, 'get_config_h_filename', self._get_config_h_filename)
    monkeypatch.setattr(sys, 'version', sys.version)


class TestCygwinCCompiler(support.TempdirManager):
    def _get_config_h_filename(self):
        return self.python_h

    @pytest.mark.skipif('sys.platform != "cygwin"')
    @pytest.mark.skipif('not os.path.exists("/usr/lib/libbash.dll.a")')
    def test_find_library_file(self):
        from distutils.cygwinccompiler import CygwinCCompiler

        compiler = CygwinCCompiler()
        link_name = "bash"
        linkable_file = compiler.find_library_file(["/usr/lib"], link_name)
        assert linkable_file is not None
        assert os.path.exists(linkable_file)
        assert linkable_file == f"/usr/lib/lib{link_name:s}.dll.a"

    @pytest.mark.skipif('sys.platform != "cygwin"')
    def test_runtime_library_dir_option(self):
        from distutils.cygwinccompiler import CygwinCCompiler

        compiler = CygwinCCompiler()
        assert compiler.runtime_library_dir_option('/foo') == []

    def test_check_config_h(self):
        # check_config_h looks for "GCC" in sys.version first
        # returns CONFIG_H_OK if found
        sys.version = (
            '2.6.1 (r261:67515, Dec  6 2008, 16:42:21) \n[GCC '
            '4.0.1 (Apple Computer, Inc. build 5370)]'
        )

        assert cygwin.check_config_h()[0] == cygwin.CONFIG_H_OK

        # then it tries to see if it can find "__GNUC__" in pyconfig.h
        sys.version = 'something without the *CC word'

        # if the file doesn't exist it returns  CONFIG_H_UNCERTAIN
        assert cygwin.check_config_h()[0] == cygwin.CONFIG_H_UNCERTAIN

        # if it exists but does not contain __GNUC__, it returns CONFIG_H_NOTOK
        self.write_file(self.python_h, 'xxx')
        assert cygwin.check_config_h()[0] == cygwin.CONFIG_H_NOTOK

        # and CONFIG_H_OK if __GNUC__ is found
        self.write_file(self.python_h, 'xxx __GNUC__ xxx')
        assert cygwin.check_config_h()[0] == cygwin.CONFIG_H_OK

    def test_get_msvcr(self):
        assert cygwin.get_msvcr() == []

    @pytest.mark.skipif('sys.platform != "cygwin"')
    def test_dll_libraries_not_none(self):
        from distutils.cygwinccompiler import CygwinCCompiler

        compiler = CygwinCCompiler()
        assert compiler.dll_libraries is not None
