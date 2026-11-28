"""A search glob must narrow results, never un-ignore files.

rg treats ``--glob <pat>`` as an override that beats .gitignore, so ``file_glob='*.py'``
returned gitignored ``.py`` files and ``target='files', pattern='*'`` walked
``node_modules``/``.git`` wholesale (port of Kilo-Org/kilocode#15017).
"""

import shutil
import subprocess

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations

pytestmark = pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")


@pytest.fixture
def repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("node_modules/\ngenerated.py\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "generated.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "index.py").write_text("needle\n", encoding="utf-8")
    return tmp_path


def _names(result, root):
    paths = result.files or [m.path for m in result.matches]
    return sorted({str(p).replace(str(root) + "/", "") for p in paths})


@pytest.mark.parametrize("target,pattern,kwargs", [
    ("content", "needle", {"file_glob": "*.py", "output_mode": "files_only"}),
    ("content", "needle", {"file_glob": "**/*.py", "output_mode": "files_only"}),
    ("files", "*.py", {}),
])
def test_glob_does_not_resurface_gitignored_files(repo, target, pattern, kwargs):
    ops = ShellFileOperations(LocalEnvironment(cwd=str(repo)), cwd=str(repo))
    result = ops.search(pattern, path=str(repo), target=target, **kwargs)
    assert result.error is None
    names = _names(result, repo)
    assert "src/app.py" in names
    assert not [n for n in names if n == "generated.py" or n.startswith("node_modules")]


def test_star_file_listing_stays_out_of_ignored_and_git_dirs(repo):
    ops = ShellFileOperations(LocalEnvironment(cwd=str(repo)), cwd=str(repo))
    names = _names(ops.search("*", path=str(repo), target="files"), repo)
    assert "src/app.py" in names
    assert not [n for n in names if n.startswith(("node_modules", ".git/")) or n == "generated.py"]
