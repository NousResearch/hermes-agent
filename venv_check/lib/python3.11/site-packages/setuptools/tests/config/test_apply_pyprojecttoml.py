"""Make sure that applying the configuration from pyproject.toml is equivalent to
applying a similar configuration from setup.cfg

To run these tests offline, please have a look on ``./downloads/preload.py``
"""

from __future__ import annotations

import io
import re
import tarfile
from inspect import cleandoc
from pathlib import Path
from unittest.mock import Mock

import pytest
from ini2toml.api import LiteTranslator
from packaging.metadata import Metadata

import setuptools  # noqa: F401 # ensure monkey patch to metadata
from setuptools._static import is_static
from setuptools.command.egg_info import write_requirements
from setuptools.config import expand, pyprojecttoml, setupcfg
from setuptools.config._apply_pyprojecttoml import _MissingDynamic, _some_attrgetter
from setuptools.dist import Distribution
from setuptools.errors import InvalidConfigError, RemovedConfigError
from setuptools.warnings import InformationOnly, SetuptoolsDeprecationWarning

from .downloads import retrieve_file, urls_from_file

HERE = Path(__file__).parent
EXAMPLES_FILE = "setupcfg_examples.txt"


def makedist(path, **attrs):
    return Distribution({"src_root": path, **attrs})


def _mock_expand_patterns(patterns, *_, **__):
    """
    Allow comparing the given patterns for 2 dist objects.
    We need to strip special chars to avoid errors when validating.
    """
    return [re.sub("[^a-z0-9]+", "", p, flags=re.I) or "empty" for p in patterns]


@pytest.mark.parametrize("url", urls_from_file(HERE / EXAMPLES_FILE))
@pytest.mark.filterwarnings("ignore")
@pytest.mark.uses_network
def test_apply_pyproject_equivalent_to_setupcfg(url, monkeypatch, tmp_path):
    monkeypatch.setattr(expand, "read_attr", Mock(return_value="0.0.1"))
    monkeypatch.setattr(
        Distribution, "_expand_patterns", Mock(side_effect=_mock_expand_patterns)
    )
    setupcfg_example = retrieve_file(url)
    pyproject_example = Path(tmp_path, "pyproject.toml")
    setupcfg_text = setupcfg_example.read_text(encoding="utf-8")
    toml_config = LiteTranslator().translate(setupcfg_text, "setup.cfg")
    pyproject_example.write_text(toml_config, encoding="utf-8")

    dist_toml = pyprojecttoml.apply_configuration(makedist(tmp_path), pyproject_example)
    dist_cfg = setupcfg.apply_configuration(makedist(tmp_path), setupcfg_example)

    pkg_info_toml = core_metadata(dist_toml)
    pkg_info_cfg = core_metadata(dist_cfg)
    assert pkg_info_toml == pkg_info_cfg

    if any(getattr(d, "license_files", None) for d in (dist_toml, dist_cfg)):
        assert set(dist_toml.license_files) == set(dist_cfg.license_files)

    if any(getattr(d, "entry_points", None) for d in (dist_toml, dist_cfg)):
        print(dist_cfg.entry_points)
        ep_toml = {
            (k, *sorted(i.replace(" ", "") for i in v))
            for k, v in dist_toml.entry_points.items()
        }
        ep_cfg = {
            (k, *sorted(i.replace(" ", "") for i in v))
            for k, v in dist_cfg.entry_points.items()
        }
        assert ep_toml == ep_cfg

    if any(getattr(d, "package_data", None) for d in (dist_toml, dist_cfg)):
        pkg_data_toml = {(k, *sorted(v)) for k, v in dist_toml.package_data.items()}
        pkg_data_cfg = {(k, *sorted(v)) for k, v in dist_cfg.package_data.items()}
        assert pkg_data_toml == pkg_data_cfg

    if any(getattr(d, "data_files", None) for d in (dist_toml, dist_cfg)):
        data_files_toml = {(k, *sorted(v)) for k, v in dist_toml.data_files}
        data_files_cfg = {(k, *sorted(v)) for k, v in dist_cfg.data_files}
        assert data_files_toml == data_files_cfg

    assert set(dist_toml.install_requires) == set(dist_cfg.install_requires)
    if any(getattr(d, "extras_require", None) for d in (dist_toml, dist_cfg)):
        extra_req_toml = {(k, *sorted(v)) for k, v in dist_toml.extras_require.items()}
        extra_req_cfg = {(k, *sorted(v)) for k, v in dist_cfg.extras_require.items()}
        assert extra_req_toml == extra_req_cfg


PEP621_EXAMPLE = """\
[project]
name = "spam"
version = "2020.0.0"
description = "Lovely Spam! Wonderful Spam!"
readme = "README.rst"
requires-python = ">=3.8"
license-files = ["LICENSE.txt"]  # Updated to be PEP 639 compliant
keywords = ["egg", "bacon", "sausage", "tomatoes", "Lobster Thermidor"]
authors = [
  {email = "hi@pradyunsg.me"},
  {name = "Tzu-Ping Chung"}
]
maintainers = [
  {name = "Brett Cannon", email = "brett@python.org"},
  {name = "John X. Ãørçeč", email = "john@utf8.org"},
  {name = "Γαμα קּ 東", email = "gama@utf8.org"},
]
classifiers = [
  "Development Status :: 4 - Beta",
  "Programming Language :: Python"
]

dependencies = [
  "httpx",
  "gidgethub[httpx]>4.0.0",
  "django>2.1; os_name != 'nt'",
  "django>2.0; os_name == 'nt'"
]

[project.optional-dependencies]
test = [
  "pytest < 5.0.0",
  "pytest-cov[all]"
]

[project.urls]
homepage = "http://example.com"
documentation = "http://readthedocs.org"
repository = "http://github.com"
changelog = "http://github.com/me/spam/blob/master/CHANGELOG.md"

[project.scripts]
spam-cli = "spam:main_cli"

[project.gui-scripts]
spam-gui = "spam:main_gui"

[project.entry-points."spam.magical"]
tomatoes = "spam:main_tomatoes"
"""

PEP621_INTERNATIONAL_EMAIL_EXAMPLE = """\
[project]
name = "spam"
version = "2020.0.0"
authors = [
  {email = "hi@pradyunsg.me"},
  {name = "Tzu-Ping Chung"}
]
maintainers = [
  {name = "Степан Бандера", email = "криївка@оун-упа.укр"},
]
"""

PEP621_EXAMPLE_SCRIPT = """
def main_cli(): pass
def main_gui(): pass
def main_tomatoes(): pass
"""

PEP639_LICENSE_TEXT = """\
[project]
name = "spam"
version = "2020.0.0"
authors = [
  {email = "hi@pradyunsg.me"},
  {name = "Tzu-Ping Chung"}
]
license = {text = "MIT"}
"""

PEP639_LICENSE_EXPRESSION = """\
[project]
name = "spam"
version = "2020.0.0"
authors = [
  {email = "hi@pradyunsg.me"},
  {name = "Tzu-Ping Chung"}
]
license = "mit or apache-2.0"  # should be normalized in metadata
classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Programming Language :: Python",
]
"""


def _pep621_example_project(
    tmp_path,
    readme="README.rst",
    pyproject_text=PEP621_EXAMPLE,
):
    pyproject = tmp_path / "pyproject.toml"
    text = pyproject_text
    replacements = {'readme = "README.rst"': f'readme = "{readme}"'}
    for orig, subst in replacements.items():
        text = text.replace(orig, subst)
    pyproject.write_text(text, encoding="utf-8")

    (tmp_path / readme).write_text("hello world", encoding="utf-8")
    (tmp_path / "LICENSE.txt").write_text("--- LICENSE stub ---", encoding="utf-8")
    (tmp_path / "spam.py").write_text(PEP621_EXAMPLE_SCRIPT, encoding="utf-8")
    return pyproject


def test_pep621_example(tmp_path):
    """Make sure the example in PEP 621 works"""
    pyproject = _pep621_example_project(tmp_path)
    dist = pyprojecttoml.apply_configuration(makedist(tmp_path), pyproject)
    assert set(dist.metadata.license_files) == {"LICENSE.txt"}


@pytest.mark.parametrize(
    ("readme", "ctype"),
    [
        ("Readme.txt", "text/plain"),
        ("readme.md", "text/markdown"),
        ("text.rst", "text/x-rst"),
    ],
)
def test_readme_content_type(tmp_path, readme, ctype):
    pyproject = _pep621_example_project(tmp_path, readme)
    dist = pyprojecttoml.apply_configuration(makedist(tmp_path), pyproject)
    assert dist.metadata.long_description_content_type == ctype


def test_undefined_content_type(tmp_path):
    pyproject = _pep621_example_project(tmp_path, "README.tex")
    with pytest.raises(ValueError, match="Undefined content type for README.tex"):
        pyprojecttoml.apply_configuration(makedist(tmp_path), pyproject)


def test_no_explicit_content_type_for_missing_extension(tmp_path):
    pyproject = _pep621_example_project(tmp_path, "README")
    dist = pyprojecttoml.apply_configuration(makedist(tmp_path), pyproject)
    assert dist.metadata.long_description_content_type is None


@pytest.mark.parametrize(
    ("pyproject_text", "expected_maintainers_meta_value"),
    (
        pytest.param(
            PEP621_EXAMPLE,
            (
                'Brett Cannon <brett@python.org>, "John X. Ãørçeč" <john@utf8.org>, '
                'Γαμα קּ 東 <gama@utf8.org>'
            ),
            id='non-international-emails',
        ),
        pytest.param(
            PEP621_INTERNATIONAL_EMAIL_EXAMPLE,
            'Степан Бандера <криївка@оун-упа.укр>',
            marks=pytest.mark.xfail(
                reason="CPython's `email.headerregistry.Address` only supports "
                'RFC 5322, as of Nov 10, 2022 and latest Python 3.11.0',
                strict=True,
            ),
            id='international-email',
        ),
    ),
)
def test_utf8_maintainer_in_metadata(  # issue-3663
    expected_maintainers_meta_value,
    pyproject_text,
    tmp_path,
):
    pyproject = _pep621_example_project(
        tmp_path,
        "README",
        pyproject_text=pyproject_text,
    )
    dist = pyprojecttoml.apply_configuration(makedist(tmp_path), pyproject)
    assert dist.metadata.maintainer_email == expected_maintainers_meta_value
    pkg_file = tmp_path / "PKG-FILE"
    with open(pkg_file, "w", encoding="utf-8") as fh:
        dist.metadata.write_pkg_file(fh)
    content = pkg_file.read_text(encoding="utf-8")
    assert f"Maintainer-email: {expected_maintainers_meta_value}" in content


@pytest.mark.parametrize(
    (
        'pyproject_text',
        'license',
        'license_expression',
        'content_str',
        'not_content_str',
    ),
    (
        pytest.param(
            PEP639_LICENSE_TEXT,
            'MIT',
            None,
            'License: MIT',
            'License-Expression: ',
            id='license-text',
            marks=[
                pytest.mark.filterwarnings(
                    "ignore:.project.license. as a TOML table is deprecated",
                )
            ],
        ),
        pytest.param(
            PEP639_LICENSE_EXPRESSION,
            None,
            'MIT OR Apache-2.0',
            'License-Expression: MIT OR Apache-2.0',
            'License: ',
            id='license-expression',
        ),
    ),
)
def test_license_in_metadata(
    license,
    license_expression,
    content_str,
    not_content_str,
    pyproject_text,
    tmp_path,
):
    pyproject = _pep621_example_project(
        tmp_path,
        "README",
        pyproject_text=pyproject_text,
    )
    dist = pyprojecttoml.apply_configuration(makedist(tmp_path), pyproject)
    assert dist.metadata.license == license
    assert dist.metadata.license_expression == license_expression
    pkg_file = tmp_path / "PKG-FILE"
    with open(pkg_file, "w", encoding="utf-8") as fh:
        dist.metadata.write_pkg_file(fh)
    content = pkg_file.read_text(encoding="utf-8")
    assert "Metadata-Version: 2.4" in content
    assert content_str in content
    assert not_content_str not in content


def test_license_classifier_with_license_expression(tmp_path):
    text = PEP639_LICENSE_EXPRESSION.rsplit("\n", 2)[0]
    pyproject = _pep621_example_project(
        tmp_path,
        "README",
        f"{text}\n    \"License :: OSI Approved :: MIT License\"\n]",
    )
    msg = "License classifiers have been superseded by license expressions"
    with pytest.raises(InvalidConfigError, match=msg) as exc:
        pyprojecttoml.apply_configuration(makedist(tmp_path), pyproject)

    assert "License :: OSI Approved :: MIT License" in str(exc.value)


def test_license_classifier_without_license_expression(tmp_path):
    text = """\
    [project]
    name = "spam"
    version = "2020.0.0"
    license = {text = "mit or apache-2.0"}
    classifiers = ["License :: OSI Approved :: MIT License"]
    """
    pyproject = _pep621_example_project(tmp_path, "README", text)

    msg1 = "License classifiers are deprecated(?:.|\n)*MIT License"
    msg2 = ".project.license. as a TOML table is deprecated"
    with (
        pytest.warns(SetuptoolsDeprecationWarning, match=msg1),
        pytest.warns(SetuptoolsDeprecationWarning, match=msg2),
    ):
        dist = pyprojecttoml.apply_configuration(makedist(tmp_path), pyproject)

    # Check license classifier is still included
    assert dist.metadata.get_classifiers() == ["License :: OSI Approved :: MIT License"]


class TestLicenseFiles:
    def base_pyproject(
        self,
        tmp_path,
        additional_text="",
        license_toml='license = {file = "LICENSE.txt"}\n',
    ):
        text = PEP639_LICENSE_EXPRESSION

        # Sanity-check
        assert 'license = "mit or apache-2.0"' in text
        assert 'license-files' not in text
        assert "[tool.setuptools]" not in text

        text = re.sub(
            r"(license = .*)\n",
            license_toml,
            text,
            count=1,
        )
        assert license_toml in text  # sanity check
        text = f"{text}\n{additional_text}\n"
        pyproject = _pep621_example_project(tmp_path, "README", pyproject_text=text)
        return pyproject

    def base_pyproject_license_pep639(self, tmp_path, additional_text=""):
        return self.base_pyproject(
            tmp_path,
            additional_text=additional_text,
            license_toml='license = "licenseref-Proprietary"'
            '\nlicense-files = ["_FILE*"]\n',
        )

    def test_both_license_and_license_files_defined(self, tmp_path):
        setuptools_config = '[tool.setuptools]\nlicense-files = ["_FILE*"]'
        pyproject = self.base_pyproject(tmp_path, setuptools_config)

        (tmp_path / "_FILE.txt").touch()
        (tmp_path / "_FILE.rst").touch()

        # Would normally match the `license_files` patterns, but we want to exclude it
        # by being explicit. On the other hand, contents should be added to `license`
        license = tmp_path / "LICENSE.txt"
        license.write_text("LicenseRef-Proprietary\n", encoding="utf-8")

        msg1 = "'tool.setuptools.license-files' is deprecated in favor of 'project.license-files'"
        msg2 = ".project.license. as a TOML table is deprecated"
        with (
            pytest.warns(SetuptoolsDeprecationWarning, match=msg1),
            pytest.warns(SetuptoolsDeprecationWarning, match=msg2),
        ):
            dist = pyprojecttoml.apply_configuration(makedist(tmp_path), pyproject)
        assert set(dist.metadata.license_files) == {"_FILE.rst", "_FILE.txt"}
        assert dist.metadata.license == "LicenseRef-Proprietary\n"

    def test_both_license_and_license_files_defined_pep639(self, tmp_path):
        # Set license and license-files
        pyproject = self.base_pyproject_license_pep639(tmp_path)

        (tmp_path / "_FILE.txt").touch()
        (tmp_path / "_FILE.rst").touch()

        msg = "Normalizing.*LicenseRef"
        with pytest.warns(InformationOnly, match=msg):
            dist = pyprojecttoml.apply_configuration(makedist(tmp_path), pyproject)

        assert set(dist.metadata.license_files) == {"_FILE.rst", "_FILE.txt"}
        assert dist.metadata.license is None
        assert dist.metadata.license_expression == "LicenseRef-Proprietary"

    def test_license_files_defined_twice(self, tmp_path):
        # Set project.license-files and tools.setuptools.license-files
        setuptools_config = '[tool.setuptools]\nlicense-files = ["_FILE*"]'
        pyproject = self.base_pyproject_license_pep639(tmp_path, setuptools_config)

        msg = "'project.license-files' is defined already. Remove 'tool.setuptools.license-files'"
        with pytest.raises(InvalidConfigError, match=msg):
            pyprojecttoml.apply_configuration(makedist(tmp_path), pyproject)

    def test_default_patterns(self, tmp_path):
        setuptools_config = '[tool.setuptools]\nzip-safe = false'
        # ^ used just to trigger section validation
        pyproject = self.base_pyproject(tmp_path, setuptools_config, license_toml="")

        license_files = "LICENCE-a.html COPYING-abc.txt AUTHORS-xyz NOTICE,def".split()

        for fname in license_files:
            (tmp_path / fname).write_text(f"{fname}\n", encoding="utf-8")

        dist = pyprojecttoml.apply_configuration(makedist(tmp_path), pyproject)

        assert (tmp_path / "LICENSE.txt").exists()  # from base example
        assert set(dist.metadata.license_files) == {*license_files, "LICENSE.txt"}

    def test_missing_patterns(self, tmp_path):
        pyproject = self.base_pyproject_license_pep639(tmp_path)
        assert list(tmp_path.glob("_FILE*")) == []  # sanity check

        msg1 = "Cannot find any files for the given pattern.*"
        msg2 = "Normalizing 'licenseref-Proprietary' to 'LicenseRef-Proprietary'"
        with (
            pytest.warns(SetuptoolsDeprecationWarning, match=msg1),
            pytest.warns(InformationOnly, match=msg2),
        ):
            pyprojecttoml.apply_configuration(makedist(tmp_path), pyproject)

    def test_deprecated_file_expands_to_text(self, tmp_path):
        """Make sure the old example with ``license = {text = ...}`` works"""

        assert 'license-files = ["LICENSE.txt"]' in PEP621_EXAMPLE  # sanity check
        text = PEP621_EXAMPLE.replace(
            'license-files = ["LICENSE.txt"]',
            'license = {file = "LICENSE.txt"}',
        )
        pyproject = _pep621_example_project(tmp_path, pyproject_text=text)

        msg = ".project.license. as a TOML table is deprecated"
        with pytest.warns(SetuptoolsDeprecationWarning, match=msg):
            dist = pyprojecttoml.apply_configuration(makedist(tmp_path), pyproject)

        assert dist.metadata.license == "--- LICENSE stub ---"
        assert set(dist.metadata.license_files) == {"LICENSE.txt"}  # auto-filled


class TestPyModules:
    # https://github.com/pypa/setuptools/issues/4316

    def dist(self, name):
        toml_config = f"""
        [project]
        name = "test"
        version = "42.0"
        [tool.setuptools]
        py-modules = [{name!r}]
        """
        pyproject = Path("pyproject.toml")
        pyproject.write_text(cleandoc(toml_config), encoding="utf-8")
        return pyprojecttoml.apply_configuration(Distribution({}), pyproject)

    @pytest.mark.parametrize("module", ["pip-run", "abc-d.λ-xyz-e"])
    def test_valid_module_name(self, tmp_path, monkeypatch, module):
        monkeypatch.chdir(tmp_path)
        assert module in self.dist(module).py_modules

    @pytest.mark.parametrize("module", ["pip run", "-pip-run", "pip-run-stubs"])
    def test_invalid_module_name(self, tmp_path, monkeypatch, module):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="py-modules"):
            self.dist(module).py_modules


class TestExtModules:
    def test_pyproject_sets_attribute(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        pyproject = Path("pyproject.toml")
        toml_config = """
        [project]
        name = "test"
        version = "42.0"
        [tool.setuptools]
        ext-modules = [
          {name = "my.ext", sources = ["hello.c", "world.c"]}
        ]
        """
        pyproject.write_text(cleandoc(toml_config), encoding="utf-8")
        with pytest.warns(pyprojecttoml._ExperimentalConfiguration):
            dist = pyprojecttoml.apply_configuration(Distribution({}), pyproject)
        assert len(dist.ext_modules) == 1
        assert dist.ext_modules[0].name == "my.ext"
        assert set(dist.ext_modules[0].sources) == {"hello.c", "world.c"}


class TestDeprecatedFields:
    def test_namespace_packages(self, tmp_path):
        pyproject = tmp_path / "pyproject.toml"
        config = """
        [project]
        name = "myproj"
        version = "42"
        [tool.setuptools]
        namespace-packages = ["myproj.pkg"]
        """
        pyproject.write_text(cleandoc(config), encoding="utf-8")
        with pytest.raises(RemovedConfigError, match="namespace-packages"):
            pyprojecttoml.apply_configuration(makedist(tmp_path), pyproject)


class TestPresetField:
    def pyproject(self, tmp_path, dynamic, extra_content=""):
        content = f"[project]\nname = 'proj'\ndynamic = {dynamic!r}\n"
        if "version" not in dynamic:
            content += "version = '42'\n"
        file = tmp_path / "pyproject.toml"
        file.write_text(content + extra_content, encoding="utf-8")
        return file

    @pytest.mark.parametrize(
        ("attr", "field", "value"),
        [
            ("license_expression", "license", "MIT"),
            pytest.param(
                *("license", "license", "Not SPDX"),
                marks=[pytest.mark.filterwarnings("ignore:.*license. overwritten")],
            ),
            ("classifiers", "classifiers", ["Private :: Classifier"]),
            ("entry_points", "scripts", {"console_scripts": ["foobar=foobar:main"]}),
            ("entry_points", "gui-scripts", {"gui_scripts": ["bazquux=bazquux:main"]}),
            pytest.param(
                *("install_requires", "dependencies", ["six"]),
                marks=[
                    pytest.mark.filterwarnings("ignore:.*install_requires. overwritten")
                ],
            ),
        ],
    )
    def test_not_listed_in_dynamic(self, tmp_path, attr, field, value):
        """Setuptools cannot set a field if not listed in ``dynamic``"""
        pyproject = self.pyproject(tmp_path, [])
        dist = makedist(tmp_path, **{attr: value})
        msg = re.compile(f"defined outside of `pyproject.toml`:.*{field}", re.S)
        with pytest.warns(_MissingDynamic, match=msg):
            dist = pyprojecttoml.apply_configuration(dist, pyproject)

        dist_value = _some_attrgetter(f"metadata.{attr}", attr)(dist)
        assert not dist_value

    @pytest.mark.parametrize(
        ("attr", "field", "value"),
        [
            ("license_expression", "license", "MIT"),
            ("install_requires", "dependencies", []),
            ("extras_require", "optional-dependencies", {}),
            ("install_requires", "dependencies", ["six"]),
            ("classifiers", "classifiers", ["Private :: Classifier"]),
        ],
    )
    def test_listed_in_dynamic(self, tmp_path, attr, field, value):
        pyproject = self.pyproject(tmp_path, [field])
        dist = makedist(tmp_path, **{attr: value})
        dist = pyprojecttoml.apply_configuration(dist, pyproject)
        dist_value = _some_attrgetter(f"metadata.{attr}", attr)(dist)
        assert dist_value == value

    def test_license_files_exempt_from_dynamic(self, monkeypatch, tmp_path):
        """
        license-file is currently not considered in the context of dynamic.
        As per 2025-02-19, https://packaging.python.org/en/latest/specifications/pyproject-toml/#license-files
        allows setuptools to fill-in `license-files` the way it sees fit:

        > If the license-files key is not defined, tools can decide how to handle license files.
        > For example they can choose not to include any files or use their own
        > logic to discover the appropriate files in the distribution.

        Using license_files from setup.py to fill-in the value is in accordance
        with this rule.
        """
        monkeypatch.chdir(tmp_path)
        pyproject = self.pyproject(tmp_path, [])
        dist = makedist(tmp_path, license_files=["LIC*"])
        (tmp_path / "LIC1").write_text("42", encoding="utf-8")
        dist = pyprojecttoml.apply_configuration(dist, pyproject)
        assert dist.metadata.license_files == ["LIC1"]

    def test_warning_overwritten_dependencies(self, tmp_path):
        src = "[project]\nname='pkg'\nversion='0.1'\ndependencies=['click']\n"
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(src, encoding="utf-8")
        dist = makedist(tmp_path, install_requires=["wheel"])
        with pytest.warns(match="`install_requires` overwritten"):
            dist = pyprojecttoml.apply_configuration(dist, pyproject)
        assert "wheel" not in dist.install_requires

    def test_optional_dependencies_dont_remove_env_markers(self, tmp_path):
        """
        Internally setuptools converts dependencies with markers to "extras".
        If ``install_requires`` is given by ``setup.py``, we have to ensure that
        applying ``optional-dependencies`` does not overwrite the mandatory
        dependencies with markers (see #3204).
        """
        # If setuptools replace its internal mechanism that uses `requires.txt`
        # this test has to be rewritten to adapt accordingly
        extra = "\n[project.optional-dependencies]\nfoo = ['bar>1']\n"
        pyproject = self.pyproject(tmp_path, ["dependencies"], extra)
        install_req = ['importlib-resources (>=3.0.0) ; python_version < "3.7"']
        dist = makedist(tmp_path, install_requires=install_req)
        dist = pyprojecttoml.apply_configuration(dist, pyproject)
        assert "foo" in dist.extras_require
        egg_info = dist.get_command_obj("egg_info")
        write_requirements(egg_info, tmp_path, tmp_path / "requires.txt")
        reqs = (tmp_path / "requires.txt").read_text(encoding="utf-8")
        assert "importlib-resources" in reqs
        assert "bar" in reqs
        assert ':python_version < "3.7"' in reqs

    @pytest.mark.parametrize(
        ("field", "group"),
        [("scripts", "console_scripts"), ("gui-scripts", "gui_scripts")],
    )
    @pytest.mark.filterwarnings("error")
    def test_scripts_dont_require_dynamic_entry_points(self, tmp_path, field, group):
        # Issue 3862
        pyproject = self.pyproject(tmp_path, [field])
        dist = makedist(tmp_path, entry_points={group: ["foobar=foobar:main"]})
        dist = pyprojecttoml.apply_configuration(dist, pyproject)
        assert group in dist.entry_points


class TestMeta:
    def test_example_file_in_sdist(self, setuptools_sdist):
        """Meta test to ensure tests can run from sdist"""
        with tarfile.open(setuptools_sdist) as tar:
            assert any(name.endswith(EXAMPLES_FILE) for name in tar.getnames())


class TestInteropCommandLineParsing:
    def test_version(self, tmp_path, monkeypatch, capsys):
        # See pypa/setuptools#4047
        # This test can be removed once the CLI interface of setup.py is removed
        monkeypatch.chdir(tmp_path)
        toml_config = """
        [project]
        name = "test"
        version = "42.0"
        """
        pyproject = Path(tmp_path, "pyproject.toml")
        pyproject.write_text(cleandoc(toml_config), encoding="utf-8")
        opts = {"script_args": ["--version"]}
        dist = pyprojecttoml.apply_configuration(Distribution(opts), pyproject)
        dist.parse_command_line()  # <-- there should be no exception here.
        captured = capsys.readouterr()
        assert "42.0" in captured.out


class TestStaticConfig:
    def test_mark_static_fields(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        toml_config = """
        [project]
        name = "test"
        version = "42.0"
        dependencies = ["hello"]
        keywords = ["world"]
        classifiers = ["private :: hello world"]
        [tool.setuptools]
        obsoletes = ["abcd"]
        provides = ["abcd"]
        platforms = ["abcd"]
        """
        pyproject = Path(tmp_path, "pyproject.toml")
        pyproject.write_text(cleandoc(toml_config), encoding="utf-8")
        dist = pyprojecttoml.apply_configuration(Distribution({}), pyproject)
        assert is_static(dist.install_requires)
        assert is_static(dist.metadata.keywords)
        assert is_static(dist.metadata.classifiers)
        assert is_static(dist.metadata.obsoletes)
        assert is_static(dist.metadata.provides)
        assert is_static(dist.metadata.platforms)


# --- Auxiliary Functions ---


def core_metadata(dist) -> str:
    with io.StringIO() as buffer:
        dist.metadata.write_pkg_file(buffer)
        pkg_file_txt = buffer.getvalue()

    # Make sure core metadata is valid
    Metadata.from_email(pkg_file_txt, validate=True)  # can raise exceptions

    skip_prefixes: tuple[str, ...] = ()
    skip_lines = set()
    # ---- DIFF NORMALISATION ----
    # PEP 621 is very particular about author/maintainer metadata conversion, so skip
    skip_prefixes += ("Author:", "Author-email:", "Maintainer:", "Maintainer-email:")
    # May be redundant with Home-page
    skip_prefixes += ("Project-URL: Homepage,", "Home-page:")
    # May be missing in original (relying on default) but backfilled in the TOML
    skip_prefixes += ("Description-Content-Type:",)
    # Remove empty lines
    skip_lines.add("")

    result = []
    for line in pkg_file_txt.splitlines():
        if line.startswith(skip_prefixes) or line in skip_lines:
            continue
        result.append(line + "\n")

    return "".join(result)
