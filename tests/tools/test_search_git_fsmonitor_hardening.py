"""Fail-closed tests for git option injection via the searched repo's config (#137582).

``_filter_gitignored_paths`` shells out to git inside the repo the model picked as
the search root. Without a ``-c core.fsmonitor=false`` override, git executes the
program named by that repo's own ``.git/config`` ``core.fsmonitor`` — and
``_path_not_found_result`` passed a dash-named parent to ``ls`` as options.
"""

import json
import subprocess

from tools.file_tools import search_tool


def _init_repo(repo):
    subprocess.run(["git", "init", "-q", str(repo)], check=True)


class TestFsmonitorInjection:

    def test_check_ignore_does_not_execute_repo_fsmonitor(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        repo = tmp_path / "repo"
        (repo / "somedir").mkdir(parents=True)
        (repo / "somedir" / "f.txt").write_text("TOKEN_ALPHA = 1\n")
        marker = tmp_path / "FSMARKER"
        _init_repo(repo)
        with open(repo / ".git" / "config", "a") as cfg:
            cfg.write(f"[core]\n\tfsmonitor = touch {marker} #\n")
        r = json.loads(search_tool("somedir", target="files", path=str(repo), task_id="t-fsm"))
        # The search still works and names the directory...
        assert "somedir" in json.dumps(r)
        # ...but the repo-configured fsmonitor program never ran (fail-closed).
        assert not marker.exists()

    def test_gitignore_filtering_still_applies(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        repo = tmp_path / "repo"
        # Directory matches come from find and are filtered only by git check-ignore;
        # "generated_cache" is outside the fixed prune list, so only git drops it.
        (repo / "generated_cache").mkdir(parents=True)
        (repo / "kept_cache").mkdir()
        _init_repo(repo)
        (repo / ".gitignore").write_text("generated_cache/\n")
        r = json.loads(search_tool("*_cache", target="files", path=str(repo), task_id="t-gi"))
        found = json.dumps(r)
        assert "kept_cache" in found
        assert "generated_cache" not in found

class TestPathNotFoundHint:

    def test_missing_relative_path_hint_lists_dash_named_parent(self, tmp_path):
        # An absolute parent starts with `/` and is never an ls option; the relative
        # spelling `--width/gone` is what the `--` separator actually protects.
        from tools.environments.local import LocalEnvironment
        from tools.file_operations import ShellFileOperations

        (tmp_path / "--width").mkdir()
        (tmp_path / "--width" / "gone_sibling.txt").write_text("x\n")
        ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)), cwd=str(tmp_path))
        result = ops._path_not_found_result("--width/gone")
        assert "gone_sibling.txt" in (result.error or "")
