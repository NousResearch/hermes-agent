import shutil
from pathlib import Path

import pytest

import pkg_resources

TESTS_DATA_DIR = Path(__file__).parent / 'data'


class TestFindDistributions:
    @pytest.fixture
    def target_dir(self, tmpdir):
        target_dir = tmpdir.mkdir('target')
        # place a .egg named directory in the target that is not an egg:
        target_dir.mkdir('not.an.egg')
        return target_dir

    def test_non_egg_dir_named_egg(self, target_dir):
        dists = pkg_resources.find_distributions(str(target_dir))
        assert not list(dists)

    def test_standalone_egg_directory(self, target_dir):
        shutil.copytree(
            TESTS_DATA_DIR / 'my-test-package_unpacked-egg',
            target_dir,
            dirs_exist_ok=True,
        )
        dists = pkg_resources.find_distributions(str(target_dir))
        assert [dist.project_name for dist in dists] == ['my-test-package']
        dists = pkg_resources.find_distributions(str(target_dir), only=True)
        assert not list(dists)

    def test_zipped_egg(self, target_dir):
        shutil.copytree(
            TESTS_DATA_DIR / 'my-test-package_zipped-egg',
            target_dir,
            dirs_exist_ok=True,
        )
        dists = pkg_resources.find_distributions(str(target_dir))
        assert [dist.project_name for dist in dists] == ['my-test-package']
        dists = pkg_resources.find_distributions(str(target_dir), only=True)
        assert not list(dists)

    def test_zipped_sdist_one_level_removed(self, target_dir):
        shutil.copytree(
            TESTS_DATA_DIR / 'my-test-package-zip', target_dir, dirs_exist_ok=True
        )
        dists = pkg_resources.find_distributions(
            str(target_dir / "my-test-package.zip")
        )
        assert [dist.project_name for dist in dists] == ['my-test-package']
        dists = pkg_resources.find_distributions(
            str(target_dir / "my-test-package.zip"), only=True
        )
        assert not list(dists)
