"""Tests for automatic discovery of modules"""

import os

import pytest

from setuptools.discovery import FlatLayoutModuleFinder, ModuleFinder

from .compat.py39 import os_helper
from .test_find_packages import ensure_files


class TestModuleFinder:
    def find(self, path, *args, **kwargs):
        return set(ModuleFinder.find(str(path), *args, **kwargs))

    EXAMPLES = {
        # circumstance: (files, kwargs, expected_modules)
        "simple_folder": (
            ["file.py", "other.py"],
            {},  # kwargs
            ["file", "other"],
        ),
        "exclude": (
            ["file.py", "other.py"],
            {"exclude": ["f*"]},
            ["other"],
        ),
        "include": (
            ["file.py", "fole.py", "other.py"],
            {"include": ["f*"], "exclude": ["fo*"]},
            ["file"],
        ),
        "invalid-name": (["my-file.py", "other.file.py"], {}, []),
    }

    @pytest.mark.parametrize("example", EXAMPLES.keys())
    def test_finder(self, tmp_path, example):
        files, kwargs, expected_modules = self.EXAMPLES[example]
        ensure_files(tmp_path, files)
        assert self.find(tmp_path, **kwargs) == set(expected_modules)

    @pytest.mark.skipif(not os_helper.can_symlink(), reason='Symlink support required')
    def test_symlinked_packages_are_included(self, tmp_path):
        src = "_myfiles/file.py"
        ensure_files(tmp_path, [src])
        os.symlink(tmp_path / src, tmp_path / "link.py")
        assert self.find(tmp_path) == {"link"}


class TestFlatLayoutModuleFinder:
    def find(self, path, *args, **kwargs):
        return set(FlatLayoutModuleFinder.find(str(path)))

    EXAMPLES = {
        # circumstance: (files, expected_modules)
        "hidden-files": ([".module.py"], []),
        "private-modules": (["_module.py"], []),
        "common-names": (
            ["setup.py", "conftest.py", "test.py", "tests.py", "example.py", "mod.py"],
            ["mod"],
        ),
        "tool-specific": (
            ["tasks.py", "fabfile.py", "noxfile.py", "dodo.py", "manage.py", "mod.py"],
            ["mod"],
        ),
    }

    @pytest.mark.parametrize("example", EXAMPLES.keys())
    def test_unwanted_files_not_included(self, tmp_path, example):
        files, expected_modules = self.EXAMPLES[example]
        ensure_files(tmp_path, files)
        assert self.find(tmp_path) == set(expected_modules)
