import re
from configparser import ConfigParser
from inspect import cleandoc

import jaraco.path
import pytest
import tomli_w
from path import Path

import setuptools  # noqa: F401 # force distutils.core to be patched
from setuptools.config.pyprojecttoml import (
    _ToolsTypoInMetadata,
    apply_configuration,
    expand_configuration,
    read_configuration,
    validate,
)
from setuptools.dist import Distribution
from setuptools.errors import OptionError

import distutils.core

EXAMPLE = """
[project]
name = "myproj"
keywords = ["some", "key", "words"]
dynamic = ["version", "readme"]
requires-python = ">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*"
dependencies = [
    'importlib-metadata>=0.12;python_version<"3.8"',
    'importlib-resources>=1.0;python_version<"3.7"',
    'pathlib2>=2.3.3,<3;python_version < "3.4" and sys.platform != "win32"',
]

[project.optional-dependencies]
docs = [
    "sphinx>=3",
    "sphinx-argparse>=0.2.5",
    "sphinx-rtd-theme>=0.4.3",
]
testing = [
    "pytest>=1",
    "coverage>=3,<5",
]

[project.scripts]
exec = "pkg.__main__:exec"

[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
package-dir = {"" = "src"}
zip-safe = true
platforms = ["any"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.cmdclass]
sdist = "pkg.mod.CustomSdist"

[tool.setuptools.dynamic.version]
attr = "pkg.__version__.VERSION"

[tool.setuptools.dynamic.readme]
file = ["README.md"]
content-type = "text/markdown"

[tool.setuptools.package-data]
"*" = ["*.txt"]

[tool.setuptools.data-files]
"data" = ["_files/*.txt"]

[tool.distutils.sdist]
formats = "gztar"

[tool.distutils.bdist_wheel]
universal = true
"""


def create_example(path, pkg_root):
    files = {
        "pyproject.toml": EXAMPLE,
        "README.md": "hello world",
        "_files": {
            "file.txt": "",
        },
    }
    packages = {
        "pkg": {
            "__init__.py": "",
            "mod.py": "class CustomSdist: pass",
            "__version__.py": "VERSION = (3, 10)",
            "__main__.py": "def exec(): print('hello')",
        },
    }

    assert pkg_root  # Meta-test: cannot be empty string.

    if pkg_root == ".":
        files = {**files, **packages}
        # skip other files: flat-layout will raise error for multi-package dist
    else:
        # Use this opportunity to ensure namespaces are discovered
        files[pkg_root] = {**packages, "other": {"nested": {"__init__.py": ""}}}

    jaraco.path.build(files, prefix=path)


def verify_example(config, path, pkg_root):
    pyproject = path / "pyproject.toml"
    pyproject.write_text(tomli_w.dumps(config), encoding="utf-8")
    expanded = expand_configuration(config, path)
    expanded_project = expanded["project"]
    assert read_configuration(pyproject, expand=True) == expanded
    assert expanded_project["version"] == "3.10"
    assert expanded_project["readme"]["text"] == "hello world"
    assert "packages" in expanded["tool"]["setuptools"]
    if pkg_root == ".":
        # Auto-discovery will raise error for multi-package dist
        assert set(expanded["tool"]["setuptools"]["packages"]) == {"pkg"}
    else:
        assert set(expanded["tool"]["setuptools"]["packages"]) == {
            "pkg",
            "other",
            "other.nested",
        }
    assert expanded["tool"]["setuptools"]["include-package-data"] is True
    assert "" in expanded["tool"]["setuptools"]["package-data"]
    assert "*" not in expanded["tool"]["setuptools"]["package-data"]
    assert expanded["tool"]["setuptools"]["data-files"] == [
        ("data", ["_files/file.txt"])
    ]


def test_read_configuration(tmp_path):
    create_example(tmp_path, "src")
    pyproject = tmp_path / "pyproject.toml"

    config = read_configuration(pyproject, expand=False)
    assert config["project"].get("version") is None
    assert config["project"].get("readme") is None

    verify_example(config, tmp_path, "src")


@pytest.mark.parametrize(
    ("pkg_root", "opts"),
    [
        (".", {}),
        ("src", {}),
        ("lib", {"packages": {"find": {"where": ["lib"]}}}),
    ],
)
def test_discovered_package_dir_with_attr_directive_in_config(tmp_path, pkg_root, opts):
    create_example(tmp_path, pkg_root)

    pyproject = tmp_path / "pyproject.toml"

    config = read_configuration(pyproject, expand=False)
    assert config["project"].get("version") is None
    assert config["project"].get("readme") is None
    config["tool"]["setuptools"].pop("packages", None)
    config["tool"]["setuptools"].pop("package-dir", None)

    config["tool"]["setuptools"].update(opts)
    verify_example(config, tmp_path, pkg_root)


ENTRY_POINTS = {
    "console_scripts": {"a": "mod.a:func"},
    "gui_scripts": {"b": "mod.b:func"},
    "other": {"c": "mod.c:func [extra]"},
}


class TestEntryPoints:
    def write_entry_points(self, tmp_path):
        entry_points = ConfigParser()
        entry_points.read_dict(ENTRY_POINTS)
        with open(tmp_path / "entry-points.txt", "w", encoding="utf-8") as f:
            entry_points.write(f)

    def pyproject(self, dynamic=None):
        project = {"dynamic": dynamic or ["scripts", "gui-scripts", "entry-points"]}
        tool = {"dynamic": {"entry-points": {"file": "entry-points.txt"}}}
        return {"project": project, "tool": {"setuptools": tool}}

    def test_all_listed_in_dynamic(self, tmp_path):
        self.write_entry_points(tmp_path)
        expanded = expand_configuration(self.pyproject(), tmp_path)
        expanded_project = expanded["project"]
        assert len(expanded_project["scripts"]) == 1
        assert expanded_project["scripts"]["a"] == "mod.a:func"
        assert len(expanded_project["gui-scripts"]) == 1
        assert expanded_project["gui-scripts"]["b"] == "mod.b:func"
        assert len(expanded_project["entry-points"]) == 1
        assert expanded_project["entry-points"]["other"]["c"] == "mod.c:func [extra]"

    @pytest.mark.parametrize("missing_dynamic", ("scripts", "gui-scripts"))
    def test_scripts_not_listed_in_dynamic(self, tmp_path, missing_dynamic):
        self.write_entry_points(tmp_path)
        dynamic = {"scripts", "gui-scripts", "entry-points"} - {missing_dynamic}

        msg = f"defined outside of `pyproject.toml`:.*{missing_dynamic}"
        with pytest.raises(OptionError, match=re.compile(msg, re.S)):
            expand_configuration(self.pyproject(dynamic), tmp_path)


class TestClassifiers:
    def test_dynamic(self, tmp_path):
        # Let's create a project example that has dynamic classifiers
        # coming from a txt file.
        create_example(tmp_path, "src")
        classifiers = cleandoc(
            """
            Framework :: Flask
            Programming Language :: Haskell
            """
        )
        (tmp_path / "classifiers.txt").write_text(classifiers, encoding="utf-8")

        pyproject = tmp_path / "pyproject.toml"
        config = read_configuration(pyproject, expand=False)
        dynamic = config["project"]["dynamic"]
        config["project"]["dynamic"] = list({*dynamic, "classifiers"})
        dynamic_config = config["tool"]["setuptools"]["dynamic"]
        dynamic_config["classifiers"] = {"file": "classifiers.txt"}

        # When the configuration is expanded,
        # each line of the file should be an different classifier.
        validate(config, pyproject)
        expanded = expand_configuration(config, tmp_path)

        assert set(expanded["project"]["classifiers"]) == {
            "Framework :: Flask",
            "Programming Language :: Haskell",
        }

    def test_dynamic_without_config(self, tmp_path):
        config = """
        [project]
        name = "myproj"
        version = '42'
        dynamic = ["classifiers"]
        """

        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(cleandoc(config), encoding="utf-8")
        with pytest.raises(OptionError, match="No configuration .* .classifiers."):
            read_configuration(pyproject)

    def test_dynamic_readme_from_setup_script_args(self, tmp_path):
        config = """
        [project]
        name = "myproj"
        version = '42'
        dynamic = ["readme"]
        """
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(cleandoc(config), encoding="utf-8")
        dist = Distribution(attrs={"long_description": "42"})
        # No error should occur because of missing `readme`
        dist = apply_configuration(dist, pyproject)
        assert dist.metadata.long_description == "42"

    def test_dynamic_without_file(self, tmp_path):
        config = """
        [project]
        name = "myproj"
        version = '42'
        dynamic = ["classifiers"]

        [tool.setuptools.dynamic]
        classifiers = {file = ["classifiers.txt"]}
        """

        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(cleandoc(config), encoding="utf-8")
        with pytest.warns(UserWarning, match="File .*classifiers.txt. cannot be found"):
            expanded = read_configuration(pyproject)
        assert "classifiers" not in expanded["project"]


@pytest.mark.parametrize(
    "example",
    (
        """
        [project]
        name = "myproj"
        version = "1.2"

        [my-tool.that-disrespect.pep518]
        value = 42
        """,
    ),
)
def test_ignore_unrelated_config(tmp_path, example):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(cleandoc(example), encoding="utf-8")

    # Make sure no error is raised due to 3rd party configs in pyproject.toml
    assert read_configuration(pyproject) is not None


@pytest.mark.parametrize(
    ("example", "error_msg"),
    [
        (
            """
            [project]
            name = "myproj"
            version = "1.2"
            requires = ['pywin32; platform_system=="Windows"' ]
            """,
            "configuration error: .project. must not contain ..requires.. properties",
        ),
    ],
)
def test_invalid_example(tmp_path, example, error_msg):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(cleandoc(example), encoding="utf-8")

    pattern = re.compile(f"invalid pyproject.toml.*{error_msg}.*", re.M | re.S)
    with pytest.raises(ValueError, match=pattern):
        read_configuration(pyproject)


@pytest.mark.parametrize("config", ("", "[tool.something]\nvalue = 42"))
def test_empty(tmp_path, config):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(config, encoding="utf-8")

    # Make sure no error is raised
    assert read_configuration(pyproject) == {}


@pytest.mark.parametrize("config", ("[project]\nname = 'myproj'\nversion='42'\n",))
def test_include_package_data_by_default(tmp_path, config):
    """Builds with ``pyproject.toml`` should consider ``include-package-data=True`` as
    default.
    """
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(config, encoding="utf-8")

    config = read_configuration(pyproject)
    assert config["tool"]["setuptools"]["include-package-data"] is True


def test_include_package_data_in_setuppy(tmp_path):
    """Builds with ``pyproject.toml`` should consider ``include_package_data`` set in
    ``setup.py``.

    See https://github.com/pypa/setuptools/issues/3197#issuecomment-1079023889
    """
    files = {
        "pyproject.toml": "[project]\nname = 'myproj'\nversion='42'\n",
        "setup.py": "__import__('setuptools').setup(include_package_data=False)",
    }
    jaraco.path.build(files, prefix=tmp_path)

    with Path(tmp_path):
        dist = distutils.core.run_setup("setup.py", {}, stop_after="config")

    assert dist.get_name() == "myproj"
    assert dist.get_version() == "42"
    assert dist.include_package_data is False


def test_warn_tools_typo(tmp_path):
    """Test that the common ``tools.setuptools`` typo in ``pyproject.toml`` issues a warning

    See https://github.com/pypa/setuptools/issues/4150
    """
    config = """
    [build-system]
    requires = ["setuptools"]
    build-backend = "setuptools.build_meta"

    [project]
    name = "myproj"
    version = '42'

    [tools.setuptools]
    packages = ["package"]
    """

    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(cleandoc(config), encoding="utf-8")

    with pytest.warns(_ToolsTypoInMetadata):
        read_configuration(pyproject)
