"""Tests for distutils.archive_util."""

import functools
import operator
import os
import pathlib
import sys
import tarfile
from distutils import archive_util
from distutils.archive_util import (
    ARCHIVE_FORMATS,
    check_archive_formats,
    make_archive,
    make_tarball,
    make_zipfile,
)
from distutils.spawn import spawn
from distutils.tests import support
from os.path import splitdrive

import path
import pytest
from test.support import patch

from .unix_compat import UID_0_SUPPORT, grp, pwd, require_uid_0, require_unix_id


def can_fs_encode(filename):
    """
    Return True if the filename can be saved in the file system.
    """
    if os.path.supports_unicode_filenames:
        return True
    try:
        filename.encode(sys.getfilesystemencoding())
    except UnicodeEncodeError:
        return False
    return True


def all_equal(values):
    return functools.reduce(operator.eq, values)


def same_drive(*paths):
    return all_equal(pathlib.Path(path).drive for path in paths)


class ArchiveUtilTestCase(support.TempdirManager):
    @pytest.mark.usefixtures('needs_zlib')
    def test_make_tarball(self, name='archive'):
        # creating something to tar
        tmpdir = self._create_files()
        self._make_tarball(tmpdir, name, '.tar.gz')
        # trying an uncompressed one
        self._make_tarball(tmpdir, name, '.tar', compress=None)

    @pytest.mark.usefixtures('needs_zlib')
    def test_make_tarball_gzip(self):
        tmpdir = self._create_files()
        self._make_tarball(tmpdir, 'archive', '.tar.gz', compress='gzip')

    def test_make_tarball_bzip2(self):
        pytest.importorskip('bz2')
        tmpdir = self._create_files()
        self._make_tarball(tmpdir, 'archive', '.tar.bz2', compress='bzip2')

    def test_make_tarball_xz(self):
        pytest.importorskip('lzma')
        tmpdir = self._create_files()
        self._make_tarball(tmpdir, 'archive', '.tar.xz', compress='xz')

    @pytest.mark.skipif("not can_fs_encode('årchiv')")
    def test_make_tarball_latin1(self):
        """
        Mirror test_make_tarball, except filename contains latin characters.
        """
        self.test_make_tarball('årchiv')  # note this isn't a real word

    @pytest.mark.skipif("not can_fs_encode('のアーカイブ')")
    def test_make_tarball_extended(self):
        """
        Mirror test_make_tarball, except filename contains extended
        characters outside the latin charset.
        """
        self.test_make_tarball('のアーカイブ')  # japanese for archive

    def _make_tarball(self, tmpdir, target_name, suffix, **kwargs):
        tmpdir2 = self.mkdtemp()
        if same_drive(tmpdir, tmpdir2):
            pytest.skip("source and target should be on same drive")

        base_name = os.path.join(tmpdir2, target_name)

        # working with relative paths to avoid tar warnings
        with path.Path(tmpdir):
            make_tarball(splitdrive(base_name)[1], 'dist', **kwargs)

        # check if the compressed tarball was created
        tarball = base_name + suffix
        assert os.path.exists(tarball)
        assert self._tarinfo(tarball) == self._created_files

    def _tarinfo(self, path):
        tar = tarfile.open(path)
        try:
            names = tar.getnames()
            names.sort()
            return names
        finally:
            tar.close()

    _zip_created_files = [
        'dist/',
        'dist/file1',
        'dist/file2',
        'dist/sub/',
        'dist/sub/file3',
        'dist/sub2/',
    ]
    _created_files = [p.rstrip('/') for p in _zip_created_files]

    def _create_files(self):
        # creating something to tar
        tmpdir = self.mkdtemp()
        dist = os.path.join(tmpdir, 'dist')
        os.mkdir(dist)
        self.write_file([dist, 'file1'], 'xxx')
        self.write_file([dist, 'file2'], 'xxx')
        os.mkdir(os.path.join(dist, 'sub'))
        self.write_file([dist, 'sub', 'file3'], 'xxx')
        os.mkdir(os.path.join(dist, 'sub2'))
        return tmpdir

    @pytest.mark.usefixtures('needs_zlib')
    @pytest.mark.skipif("not (shutil.which('tar') and shutil.which('gzip'))")
    def test_tarfile_vs_tar(self):
        tmpdir = self._create_files()
        tmpdir2 = self.mkdtemp()
        base_name = os.path.join(tmpdir2, 'archive')
        old_dir = os.getcwd()
        os.chdir(tmpdir)
        try:
            make_tarball(base_name, 'dist')
        finally:
            os.chdir(old_dir)

        # check if the compressed tarball was created
        tarball = base_name + '.tar.gz'
        assert os.path.exists(tarball)

        # now create another tarball using `tar`
        tarball2 = os.path.join(tmpdir, 'archive2.tar.gz')
        tar_cmd = ['tar', '-cf', 'archive2.tar', 'dist']
        gzip_cmd = ['gzip', '-f', '-9', 'archive2.tar']
        old_dir = os.getcwd()
        os.chdir(tmpdir)
        try:
            spawn(tar_cmd)
            spawn(gzip_cmd)
        finally:
            os.chdir(old_dir)

        assert os.path.exists(tarball2)
        # let's compare both tarballs
        assert self._tarinfo(tarball) == self._created_files
        assert self._tarinfo(tarball2) == self._created_files

        # trying an uncompressed one
        base_name = os.path.join(tmpdir2, 'archive')
        old_dir = os.getcwd()
        os.chdir(tmpdir)
        try:
            make_tarball(base_name, 'dist', compress=None)
        finally:
            os.chdir(old_dir)
        tarball = base_name + '.tar'
        assert os.path.exists(tarball)

        # now for a dry_run
        base_name = os.path.join(tmpdir2, 'archive')
        old_dir = os.getcwd()
        os.chdir(tmpdir)
        try:
            make_tarball(base_name, 'dist', compress=None, dry_run=True)
        finally:
            os.chdir(old_dir)
        tarball = base_name + '.tar'
        assert os.path.exists(tarball)

    @pytest.mark.usefixtures('needs_zlib')
    def test_make_zipfile(self):
        zipfile = pytest.importorskip('zipfile')
        # creating something to tar
        tmpdir = self._create_files()
        base_name = os.path.join(self.mkdtemp(), 'archive')
        with path.Path(tmpdir):
            make_zipfile(base_name, 'dist')

        # check if the compressed tarball was created
        tarball = base_name + '.zip'
        assert os.path.exists(tarball)
        with zipfile.ZipFile(tarball) as zf:
            assert sorted(zf.namelist()) == self._zip_created_files

    def test_make_zipfile_no_zlib(self):
        zipfile = pytest.importorskip('zipfile')
        patch(self, archive_util.zipfile, 'zlib', None)  # force zlib ImportError

        called = []
        zipfile_class = zipfile.ZipFile

        def fake_zipfile(*a, **kw):
            if kw.get('compression', None) == zipfile.ZIP_STORED:
                called.append((a, kw))
            return zipfile_class(*a, **kw)

        patch(self, archive_util.zipfile, 'ZipFile', fake_zipfile)

        # create something to tar and compress
        tmpdir = self._create_files()
        base_name = os.path.join(self.mkdtemp(), 'archive')
        with path.Path(tmpdir):
            make_zipfile(base_name, 'dist')

        tarball = base_name + '.zip'
        assert called == [((tarball, "w"), {'compression': zipfile.ZIP_STORED})]
        assert os.path.exists(tarball)
        with zipfile.ZipFile(tarball) as zf:
            assert sorted(zf.namelist()) == self._zip_created_files

    def test_check_archive_formats(self):
        assert check_archive_formats(['gztar', 'xxx', 'zip']) == 'xxx'
        assert (
            check_archive_formats(['gztar', 'bztar', 'xztar', 'ztar', 'tar', 'zip'])
            is None
        )

    def test_make_archive(self):
        tmpdir = self.mkdtemp()
        base_name = os.path.join(tmpdir, 'archive')
        with pytest.raises(ValueError):
            make_archive(base_name, 'xxx')

    def test_make_archive_cwd(self):
        current_dir = os.getcwd()

        def _breaks(*args, **kw):
            raise RuntimeError()

        ARCHIVE_FORMATS['xxx'] = (_breaks, [], 'xxx file')
        try:
            try:
                make_archive('xxx', 'xxx', root_dir=self.mkdtemp())
            except Exception:
                pass
            assert os.getcwd() == current_dir
        finally:
            ARCHIVE_FORMATS.pop('xxx')

    def test_make_archive_tar(self):
        base_dir = self._create_files()
        base_name = os.path.join(self.mkdtemp(), 'archive')
        res = make_archive(base_name, 'tar', base_dir, 'dist')
        assert os.path.exists(res)
        assert os.path.basename(res) == 'archive.tar'
        assert self._tarinfo(res) == self._created_files

    @pytest.mark.usefixtures('needs_zlib')
    def test_make_archive_gztar(self):
        base_dir = self._create_files()
        base_name = os.path.join(self.mkdtemp(), 'archive')
        res = make_archive(base_name, 'gztar', base_dir, 'dist')
        assert os.path.exists(res)
        assert os.path.basename(res) == 'archive.tar.gz'
        assert self._tarinfo(res) == self._created_files

    def test_make_archive_bztar(self):
        pytest.importorskip('bz2')
        base_dir = self._create_files()
        base_name = os.path.join(self.mkdtemp(), 'archive')
        res = make_archive(base_name, 'bztar', base_dir, 'dist')
        assert os.path.exists(res)
        assert os.path.basename(res) == 'archive.tar.bz2'
        assert self._tarinfo(res) == self._created_files

    def test_make_archive_xztar(self):
        pytest.importorskip('lzma')
        base_dir = self._create_files()
        base_name = os.path.join(self.mkdtemp(), 'archive')
        res = make_archive(base_name, 'xztar', base_dir, 'dist')
        assert os.path.exists(res)
        assert os.path.basename(res) == 'archive.tar.xz'
        assert self._tarinfo(res) == self._created_files

    def test_make_archive_owner_group(self):
        # testing make_archive with owner and group, with various combinations
        # this works even if there's not gid/uid support
        if UID_0_SUPPORT:
            group = grp.getgrgid(0)[0]
            owner = pwd.getpwuid(0)[0]
        else:
            group = owner = 'root'

        base_dir = self._create_files()
        root_dir = self.mkdtemp()
        base_name = os.path.join(self.mkdtemp(), 'archive')
        res = make_archive(
            base_name, 'zip', root_dir, base_dir, owner=owner, group=group
        )
        assert os.path.exists(res)

        res = make_archive(base_name, 'zip', root_dir, base_dir)
        assert os.path.exists(res)

        res = make_archive(
            base_name, 'tar', root_dir, base_dir, owner=owner, group=group
        )
        assert os.path.exists(res)

        res = make_archive(
            base_name, 'tar', root_dir, base_dir, owner='kjhkjhkjg', group='oihohoh'
        )
        assert os.path.exists(res)

    @pytest.mark.usefixtures('needs_zlib')
    @require_unix_id
    @require_uid_0
    def test_tarfile_root_owner(self):
        tmpdir = self._create_files()
        base_name = os.path.join(self.mkdtemp(), 'archive')
        old_dir = os.getcwd()
        os.chdir(tmpdir)
        group = grp.getgrgid(0)[0]
        owner = pwd.getpwuid(0)[0]
        try:
            archive_name = make_tarball(
                base_name, 'dist', compress=None, owner=owner, group=group
            )
        finally:
            os.chdir(old_dir)

        # check if the compressed tarball was created
        assert os.path.exists(archive_name)

        # now checks the rights
        archive = tarfile.open(archive_name)
        try:
            for member in archive.getmembers():
                assert member.uid == 0
                assert member.gid == 0
        finally:
            archive.close()
