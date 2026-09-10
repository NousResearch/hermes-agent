from __future__ import annotations

import builtins
import datetime
import inspect
import os
import plistlib
import stat
import subprocess
import sys
import tempfile
import zipfile
from unittest import mock

import pytest

import pkg_resources
from pkg_resources import DistInfoDistribution, Distribution, EggInfoDistribution

import distutils.command.install_egg_info
import distutils.dist


class EggRemover(str):
    def __call__(self):
        if self in sys.path:
            sys.path.remove(self)
        if os.path.exists(self):
            os.remove(self)


class TestZipProvider:
    finalizers: list[EggRemover] = []

    ref_time = datetime.datetime(2013, 5, 12, 13, 25, 0)
    "A reference time for a file modification"

    @classmethod
    def setup_class(cls):
        "create a zip egg and add it to sys.path"
        egg = tempfile.NamedTemporaryFile(suffix='.egg', delete=False)
        zip_egg = zipfile.ZipFile(egg, 'w')
        zip_info = zipfile.ZipInfo()
        zip_info.filename = 'mod.py'
        zip_info.date_time = cls.ref_time.timetuple()
        zip_egg.writestr(zip_info, 'x = 3\n')
        zip_info = zipfile.ZipInfo()
        zip_info.filename = 'data.dat'
        zip_info.date_time = cls.ref_time.timetuple()
        zip_egg.writestr(zip_info, 'hello, world!')
        zip_info = zipfile.ZipInfo()
        zip_info.filename = 'subdir/mod2.py'
        zip_info.date_time = cls.ref_time.timetuple()
        zip_egg.writestr(zip_info, 'x = 6\n')
        zip_info = zipfile.ZipInfo()
        zip_info.filename = 'subdir/data2.dat'
        zip_info.date_time = cls.ref_time.timetuple()
        zip_egg.writestr(zip_info, 'goodbye, world!')
        zip_egg.close()
        egg.close()

        sys.path.append(egg.name)
        subdir = os.path.join(egg.name, 'subdir')
        sys.path.append(subdir)
        cls.finalizers.append(EggRemover(subdir))
        cls.finalizers.append(EggRemover(egg.name))

    @classmethod
    def teardown_class(cls):
        for finalizer in cls.finalizers:
            finalizer()

    def test_resource_listdir(self):
        import mod  # pyright: ignore[reportMissingImports] # Temporary package for test

        zp = pkg_resources.ZipProvider(mod)

        expected_root = ['data.dat', 'mod.py', 'subdir']
        assert sorted(zp.resource_listdir('')) == expected_root

        expected_subdir = ['data2.dat', 'mod2.py']
        assert sorted(zp.resource_listdir('subdir')) == expected_subdir
        assert sorted(zp.resource_listdir('subdir/')) == expected_subdir

        assert zp.resource_listdir('nonexistent') == []
        assert zp.resource_listdir('nonexistent/') == []

        import mod2  # pyright: ignore[reportMissingImports] # Temporary package for test

        zp2 = pkg_resources.ZipProvider(mod2)

        assert sorted(zp2.resource_listdir('')) == expected_subdir

        assert zp2.resource_listdir('subdir') == []
        assert zp2.resource_listdir('subdir/') == []

    def test_resource_filename_rewrites_on_change(self):
        """
        If a previous call to get_resource_filename has saved the file, but
        the file has been subsequently mutated with different file of the
        same size and modification time, it should not be overwritten on a
        subsequent call to get_resource_filename.
        """
        import mod  # pyright: ignore[reportMissingImports] # Temporary package for test

        manager = pkg_resources.ResourceManager()
        zp = pkg_resources.ZipProvider(mod)
        filename = zp.get_resource_filename(manager, 'data.dat')
        actual = datetime.datetime.fromtimestamp(os.stat(filename).st_mtime)
        assert actual == self.ref_time
        f = open(filename, 'w', encoding="utf-8")
        f.write('hello, world?')
        f.close()
        ts = self.ref_time.timestamp()
        os.utime(filename, (ts, ts))
        filename = zp.get_resource_filename(manager, 'data.dat')
        with open(filename, encoding="utf-8") as f:
            assert f.read() == 'hello, world!'
        manager.cleanup_resources()


class TestResourceManager:
    def test_get_cache_path(self):
        mgr = pkg_resources.ResourceManager()
        path = mgr.get_cache_path('foo')
        type_ = str(type(path))
        message = "Unexpected type from get_cache_path: " + type_
        assert isinstance(path, str), message

    def test_get_cache_path_race(self, tmpdir):
        # Patch to os.path.isdir to create a race condition
        def patched_isdir(dirname, unpatched_isdir=pkg_resources.isdir):
            patched_isdir.dirnames.append(dirname)

            was_dir = unpatched_isdir(dirname)
            if not was_dir:
                os.makedirs(dirname)
            return was_dir

        patched_isdir.dirnames = []

        # Get a cache path with a "race condition"
        mgr = pkg_resources.ResourceManager()
        mgr.set_extraction_path(str(tmpdir))

        archive_name = os.sep.join(('foo', 'bar', 'baz'))
        with mock.patch.object(pkg_resources, 'isdir', new=patched_isdir):
            mgr.get_cache_path(archive_name)

        # Because this test relies on the implementation details of this
        # function, these assertions are a sentinel to ensure that the
        # test suite will not fail silently if the implementation changes.
        called_dirnames = patched_isdir.dirnames
        assert len(called_dirnames) == 2
        assert called_dirnames[0].split(os.sep)[-2:] == ['foo', 'bar']
        assert called_dirnames[1].split(os.sep)[-1:] == ['foo']

    """
    Tests to ensure that pkg_resources runs independently from setuptools.
    """

    def test_setuptools_not_imported(self):
        """
        In a separate Python environment, import pkg_resources and assert
        that action doesn't cause setuptools to be imported.
        """
        lines = (
            'import pkg_resources',
            'import sys',
            ('assert "setuptools" not in sys.modules, "setuptools was imported"'),
        )
        cmd = [sys.executable, '-c', '; '.join(lines)]
        subprocess.check_call(cmd)


def make_test_distribution(metadata_path, metadata):
    """
    Make a test Distribution object, and return it.

    :param metadata_path: the path to the metadata file that should be
        created. This should be inside a distribution directory that should
        also be created. For example, an argument value might end with
        "<project>.dist-info/METADATA".
    :param metadata: the desired contents of the metadata file, as bytes.
    """
    dist_dir = os.path.dirname(metadata_path)
    os.mkdir(dist_dir)
    with open(metadata_path, 'wb') as f:
        f.write(metadata)
    dists = list(pkg_resources.distributions_from_metadata(dist_dir))
    (dist,) = dists

    return dist


def test_get_metadata__bad_utf8(tmpdir):
    """
    Test a metadata file with bytes that can't be decoded as utf-8.
    """
    filename = 'METADATA'
    # Convert the tmpdir LocalPath object to a string before joining.
    metadata_path = os.path.join(str(tmpdir), 'foo.dist-info', filename)
    # Encode a non-ascii string with the wrong encoding (not utf-8).
    metadata = 'née'.encode('iso-8859-1')
    dist = make_test_distribution(metadata_path, metadata=metadata)

    with pytest.raises(UnicodeDecodeError) as excinfo:
        dist.get_metadata(filename)

    exc = excinfo.value
    actual = str(exc)
    expected = (
        # The error message starts with "'utf-8' codec ..." However, the
        # spelling of "utf-8" can vary (e.g. "utf8") so we don't include it
        "codec can't decode byte 0xe9 in position 1: "
        'invalid continuation byte in METADATA file at path: '
    )
    assert expected in actual, f'actual: {actual}'
    assert actual.endswith(metadata_path), f'actual: {actual}'


def make_distribution_no_version(tmpdir, basename):
    """
    Create a distribution directory with no file containing the version.
    """
    dist_dir = tmpdir / basename
    dist_dir.ensure_dir()
    # Make the directory non-empty so distributions_from_metadata()
    # will detect it and yield it.
    dist_dir.join('temp.txt').ensure()

    dists = list(pkg_resources.distributions_from_metadata(dist_dir))
    assert len(dists) == 1
    (dist,) = dists

    return dist, dist_dir


@pytest.mark.parametrize(
    ("suffix", "expected_filename", "expected_dist_type"),
    [
        ('egg-info', 'PKG-INFO', EggInfoDistribution),
        ('dist-info', 'METADATA', DistInfoDistribution),
    ],
)
@pytest.mark.xfail(
    sys.version_info[:2] == (3, 12) and sys.version_info.releaselevel != 'final',
    reason="https://github.com/python/cpython/issues/103632",
)
def test_distribution_version_missing(
    tmpdir, suffix, expected_filename, expected_dist_type
):
    """
    Test Distribution.version when the "Version" header is missing.
    """
    basename = f'foo.{suffix}'
    dist, dist_dir = make_distribution_no_version(tmpdir, basename)

    expected_text = (
        f"Missing 'Version:' header and/or {expected_filename} file at path: "
    )
    metadata_path = os.path.join(dist_dir, expected_filename)

    # Now check the exception raised when the "version" attribute is accessed.
    with pytest.raises(ValueError) as excinfo:
        dist.version

    err = str(excinfo.value)
    # Include a string expression after the assert so the full strings
    # will be visible for inspection on failure.
    assert expected_text in err, str((expected_text, err))

    # Also check the args passed to the ValueError.
    msg, dist = excinfo.value.args
    assert expected_text in msg
    # Check that the message portion contains the path.
    assert metadata_path in msg, str((metadata_path, msg))
    assert type(dist) is expected_dist_type


@pytest.mark.xfail(
    sys.version_info[:2] == (3, 12) and sys.version_info.releaselevel != 'final',
    reason="https://github.com/python/cpython/issues/103632",
)
def test_distribution_version_missing_undetected_path():
    """
    Test Distribution.version when the "Version" header is missing and
    the path can't be detected.
    """
    # Create a Distribution object with no metadata argument, which results
    # in an empty metadata provider.
    dist = Distribution('/foo')
    with pytest.raises(ValueError) as excinfo:
        dist.version

    msg, dist = excinfo.value.args
    expected = (
        "Missing 'Version:' header and/or PKG-INFO file at path: [could not detect]"
    )
    assert msg == expected


@pytest.mark.parametrize('only', [False, True])
def test_dist_info_is_not_dir(tmp_path, only):
    """Test path containing a file with dist-info extension."""
    dist_info = tmp_path / 'foobar.dist-info'
    dist_info.touch()
    assert not pkg_resources.dist_factory(str(tmp_path), str(dist_info), only)


def test_macos_vers_fallback(monkeypatch, tmp_path):
    """Regression test for pkg_resources._macos_vers"""
    orig_open = builtins.open

    # Pretend we need to use the plist file
    monkeypatch.setattr('platform.mac_ver', mock.Mock(return_value=('', (), '')))

    # Create fake content for the fake plist file
    with open(tmp_path / 'fake.plist', 'wb') as fake_file:
        plistlib.dump({"ProductVersion": "11.4"}, fake_file)

    # Pretend the fake file exists
    monkeypatch.setattr('os.path.exists', mock.Mock(return_value=True))

    def fake_open(file, *args, **kwargs):
        return orig_open(tmp_path / 'fake.plist', *args, **kwargs)

    # Ensure that the _macos_vers works correctly
    with mock.patch('builtins.open', mock.Mock(side_effect=fake_open)) as m:
        pkg_resources._macos_vers.cache_clear()
        assert pkg_resources._macos_vers() == ["11", "4"]
        pkg_resources._macos_vers.cache_clear()

    m.assert_called()


class TestDeepVersionLookupDistutils:
    @pytest.fixture
    def env(self, tmpdir):
        """
        Create a package environment, similar to a virtualenv,
        in which packages are installed.
        """

        class Environment(str):
            pass

        env = Environment(tmpdir)
        tmpdir.chmod(stat.S_IRWXU)
        subs = 'home', 'lib', 'scripts', 'data', 'egg-base'
        env.paths = dict((dirname, str(tmpdir / dirname)) for dirname in subs)
        list(map(os.mkdir, env.paths.values()))
        return env

    def create_foo_pkg(self, env, version):
        """
        Create a foo package installed (distutils-style) to env.paths['lib']
        as version.
        """
        ld = "This package has unicode metadata! ❄"
        attrs = dict(name='foo', version=version, long_description=ld)
        dist = distutils.dist.Distribution(attrs)
        iei_cmd = distutils.command.install_egg_info.install_egg_info(dist)
        iei_cmd.initialize_options()
        iei_cmd.install_dir = env.paths['lib']
        iei_cmd.finalize_options()
        iei_cmd.run()

    def test_version_resolved_from_egg_info(self, env):
        version = '1.11.0.dev0+2329eae'
        self.create_foo_pkg(env, version)

        # this requirement parsing will raise a VersionConflict unless the
        # .egg-info file is parsed (see #419 on BitBucket)
        req = pkg_resources.Requirement.parse('foo>=1.9')
        dist = pkg_resources.WorkingSet([env.paths['lib']]).find(req)
        assert dist.version == version

    @pytest.mark.parametrize(
        ("unnormalized", "normalized"),
        [
            ('foo', 'foo'),
            ('foo/', 'foo'),
            ('foo/bar', 'foo/bar'),
            ('foo/bar/', 'foo/bar'),
        ],
    )
    def test_normalize_path_trailing_sep(self, unnormalized, normalized):
        """Ensure the trailing slash is cleaned for path comparison.

        See pypa/setuptools#1519.
        """
        result_from_unnormalized = pkg_resources.normalize_path(unnormalized)
        result_from_normalized = pkg_resources.normalize_path(normalized)
        assert result_from_unnormalized == result_from_normalized

    @pytest.mark.skipif(
        os.path.normcase('A') != os.path.normcase('a'),
        reason='Testing case-insensitive filesystems.',
    )
    @pytest.mark.parametrize(
        ("unnormalized", "normalized"),
        [
            ('MiXeD/CasE', 'mixed/case'),
        ],
    )
    def test_normalize_path_normcase(self, unnormalized, normalized):
        """Ensure mixed case is normalized on case-insensitive filesystems."""
        result_from_unnormalized = pkg_resources.normalize_path(unnormalized)
        result_from_normalized = pkg_resources.normalize_path(normalized)
        assert result_from_unnormalized == result_from_normalized

    @pytest.mark.skipif(
        os.path.sep != '\\',
        reason='Testing systems using backslashes as path separators.',
    )
    @pytest.mark.parametrize(
        ("unnormalized", "expected"),
        [
            ('forward/slash', 'forward\\slash'),
            ('forward/slash/', 'forward\\slash'),
            ('backward\\slash\\', 'backward\\slash'),
        ],
    )
    def test_normalize_path_backslash_sep(self, unnormalized, expected):
        """Ensure path seps are cleaned on backslash path sep systems."""
        result = pkg_resources.normalize_path(unnormalized)
        assert result.endswith(expected)


class TestWorkdirRequire:
    def fake_site_packages(self, tmp_path, monkeypatch, dist_files):
        site_packages = tmp_path / "site-packages"
        site_packages.mkdir()
        for file, content in self.FILES.items():
            path = site_packages / file
            path.parent.mkdir(exist_ok=True, parents=True)
            path.write_text(inspect.cleandoc(content), encoding="utf-8")

        monkeypatch.setattr(sys, "path", [site_packages])
        return os.fspath(site_packages)

    FILES = {
        "pkg1_mod-1.2.3.dist-info/METADATA": """
            Metadata-Version: 2.4
            Name: pkg1.mod
            Version: 1.2.3
            """,
        "pkg2.mod-0.42.dist-info/METADATA": """
            Metadata-Version: 2.1
            Name: pkg2.mod
            Version: 0.42
            """,
        "pkg3_mod.egg-info/PKG-INFO": """
            Name: pkg3.mod
            Version: 1.2.3.4
            """,
        "pkg4.mod.egg-info/PKG-INFO": """
            Name: pkg4.mod
            Version: 0.42.1
            """,
    }

    @pytest.mark.parametrize(
        ("version", "requirement"),
        [
            ("1.2.3", "pkg1.mod>=1"),
            ("0.42", "pkg2.mod>=0.4"),
            ("1.2.3.4", "pkg3.mod<=2"),
            ("0.42.1", "pkg4.mod>0.2,<1"),
        ],
    )
    def test_require_non_normalised_name(
        self, tmp_path, monkeypatch, version, requirement
    ):
        # https://github.com/pypa/setuptools/issues/4853
        site_packages = self.fake_site_packages(tmp_path, monkeypatch, self.FILES)
        ws = pkg_resources.WorkingSet([site_packages])

        for req in [requirement, requirement.replace(".", "-")]:
            [dist] = ws.require(req)
            assert dist.version == version
            assert os.path.samefile(
                os.path.commonpath([dist.location, site_packages]), site_packages
            )
