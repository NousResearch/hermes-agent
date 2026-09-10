"""Tests for distutils.dir_util."""

import os
import pathlib
import stat
import sys
import unittest.mock as mock
from distutils import dir_util, errors
from distutils.dir_util import (
    copy_tree,
    create_tree,
    ensure_relative,
    mkpath,
    remove_tree,
)
from distutils.tests import support

import jaraco.path
import path
import pytest


@pytest.fixture(autouse=True)
def stuff(request, monkeypatch, distutils_managed_tempdir):
    self = request.instance
    tmp_dir = self.mkdtemp()
    self.root_target = os.path.join(tmp_dir, 'deep')
    self.target = os.path.join(self.root_target, 'here')
    self.target2 = os.path.join(tmp_dir, 'deep2')


class TestDirUtil(support.TempdirManager):
    def test_mkpath_remove_tree_verbosity(self, caplog):
        mkpath(self.target, verbose=False)
        assert not caplog.records
        remove_tree(self.root_target, verbose=False)

        mkpath(self.target, verbose=True)
        wanted = [f'creating {self.target}']
        assert caplog.messages == wanted
        caplog.clear()

        remove_tree(self.root_target, verbose=True)
        wanted = [f"removing '{self.root_target}' (and everything under it)"]
        assert caplog.messages == wanted

    @pytest.mark.skipif("platform.system() == 'Windows'")
    def test_mkpath_with_custom_mode(self):
        # Get and set the current umask value for testing mode bits.
        umask = os.umask(0o002)
        os.umask(umask)
        mkpath(self.target, 0o700)
        assert stat.S_IMODE(os.stat(self.target).st_mode) == 0o700 & ~umask
        mkpath(self.target2, 0o555)
        assert stat.S_IMODE(os.stat(self.target2).st_mode) == 0o555 & ~umask

    def test_create_tree_verbosity(self, caplog):
        create_tree(self.root_target, ['one', 'two', 'three'], verbose=False)
        assert caplog.messages == []
        remove_tree(self.root_target, verbose=False)

        wanted = [f'creating {self.root_target}']
        create_tree(self.root_target, ['one', 'two', 'three'], verbose=True)
        assert caplog.messages == wanted

        remove_tree(self.root_target, verbose=False)

    def test_copy_tree_verbosity(self, caplog):
        mkpath(self.target, verbose=False)

        copy_tree(self.target, self.target2, verbose=False)
        assert caplog.messages == []

        remove_tree(self.root_target, verbose=False)

        mkpath(self.target, verbose=False)
        a_file = path.Path(self.target) / 'ok.txt'
        jaraco.path.build({'ok.txt': 'some content'}, self.target)

        wanted = [f'copying {a_file} -> {self.target2}']
        copy_tree(self.target, self.target2, verbose=True)
        assert caplog.messages == wanted

        remove_tree(self.root_target, verbose=False)
        remove_tree(self.target2, verbose=False)

    def test_copy_tree_skips_nfs_temp_files(self):
        mkpath(self.target, verbose=False)

        jaraco.path.build({'ok.txt': 'some content', '.nfs123abc': ''}, self.target)

        copy_tree(self.target, self.target2)
        assert os.listdir(self.target2) == ['ok.txt']

        remove_tree(self.root_target, verbose=False)
        remove_tree(self.target2, verbose=False)

    def test_ensure_relative(self):
        if os.sep == '/':
            assert ensure_relative('/home/foo') == 'home/foo'
            assert ensure_relative('some/path') == 'some/path'
        else:  # \\
            assert ensure_relative('c:\\home\\foo') == 'c:home\\foo'
            assert ensure_relative('home\\foo') == 'home\\foo'

    def test_copy_tree_exception_in_listdir(self):
        """
        An exception in listdir should raise a DistutilsFileError
        """
        with (
            mock.patch("os.listdir", side_effect=OSError()),
            pytest.raises(errors.DistutilsFileError),
        ):
            src = self.tempdirs[-1]
            dir_util.copy_tree(src, None)

    def test_mkpath_exception_uncached(self, monkeypatch, tmp_path):
        """
        Caching should not remember failed attempts.

        pypa/distutils#304
        """

        class FailPath(pathlib.Path):
            def mkdir(self, *args, **kwargs):
                raise OSError("Failed to create directory")

            if sys.version_info < (3, 12):
                _flavour = pathlib.Path()._flavour

        target = tmp_path / 'foodir'

        with pytest.raises(errors.DistutilsFileError):
            mkpath(FailPath(target))

        assert not target.exists()

        mkpath(target)
        assert target.exists()
