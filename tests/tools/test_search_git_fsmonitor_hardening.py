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
        dependency = repo / "node_modules" / "package" / ".hidden"
        dependency.mkdir(parents=True)
        (dependency / "dependency.js").write_text("HIDDEN_GI_TOKEN = true\n")
        local = repo / ".project-local"
        local.mkdir()
        (local / "settings.cfg").write_text("HIDDEN_GI_TOKEN = true\n")
        _init_repo(repo)
        (repo / ".gitignore").write_text("node_modules/\n")
        r = json.loads(search_tool("HIDDEN_GI_TOKEN", path=str(repo), task_id="t-gi"))
        warning = r.get("warning", "")
        # check-ignore still runs (now with the fsmonitor override) and still
        # prunes gitignored dependency trees from the hidden-file hint.
        assert r["total_count"] == 0
        assert "hidden or gitignored" in warning
        assert "settings.cfg" in warning
        assert "dependency.js" not in warning


class TestPathNotFoundHint:

    def test_missing_path_hint_lists_dash_named_parent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        # A parent whose name looks like an ls option: without `--` the listing
        # fails and the similar-paths hint is silently lost.
        parent = tmp_path / "--width"
        parent.mkdir()
        (parent / "gone_sibling.txt").write_text("x\n")
        r = json.loads(search_tool("TOKEN_ALPHA", path=str(parent / "gone"), task_id="t-dash"))
        assert "gone_sibling.txt" in r.get("error", "")
