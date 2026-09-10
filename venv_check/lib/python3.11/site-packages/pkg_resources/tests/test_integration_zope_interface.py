import platform
from inspect import cleandoc

import jaraco.path
import pytest

pytestmark = pytest.mark.integration


# For the sake of simplicity this test uses fixtures defined in
# `setuptools.test.fixtures`,
# and it also exercise conditions considered deprecated...
# So if needed this test can be deleted.
@pytest.mark.skipif(
    platform.system() != "Linux",
    reason="only demonstrated to fail on Linux in #4399",
)
def test_interop_pkg_resources_iter_entry_points(tmp_path, venv):
    """
    Importing pkg_resources.iter_entry_points on console_scripts
    seems to cause trouble with zope-interface, when deprecates installation method
    is used. See #4399.
    """
    project = {
        "pkg": {
            "foo.py": cleandoc(
                """
                from pkg_resources import iter_entry_points

                def bar():
                    print("Print me if you can")
                """
            ),
            "setup.py": cleandoc(
                """
                from setuptools import setup, find_packages

                setup(
                    install_requires=["zope-interface==6.4.post2"],
                    entry_points={
                        "console_scripts": [
                            "foo=foo:bar",
                        ],
                    },
                )
                """
            ),
        }
    }
    jaraco.path.build(project, prefix=tmp_path)
    cmd = ["pip", "install", "-e", ".", "--no-use-pep517"]
    venv.run(cmd, cwd=tmp_path / "pkg")  # Needs this version of pkg_resources installed
    out = venv.run(["foo"])
    assert "Print me if you can" in out
