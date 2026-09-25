"""Search scope guard: broad-root refusal on the rg AND grep paths.

The 2026-09-24 incident (toolkit registry entry 0002) replayed because the
broad-root guard only protected the no-rg ``find`` fallback — the fast
ripgrep path that actually reads file data was unguarded, and the grep
fallback was unguarded too. A content search rooted at $HOME (parent of
OneDrive) made ripgrep hydrate every Files-On-Demand placeholder it touched.

These tests drive the real methods through the real local terminal backend
(sibling pattern: test_search_error_guard.py), with ``tools.file_operations.
_HOME`` pointed at a fixture home so nothing here touches the operator's
real home or any real cloud folder. The "OneDrive - Personal" directory
below is an ordinary fixture directory.

Via the real handler (``_handle_search_files``) for the opt-in path; direct
method calls where a lane must be forced (a host with rg installed would
never naturally reach grep).
"""

import shutil

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations


@pytest.fixture
def guarded_tree(tmp_path, monkeypatch, real_bash):
    """Fixture HOME containing a cloud-named folder and a narrow project tree."""
    # Isolated HERMES_HOME: the rg/grep plumbing reads manifest/config from the
    # hermes home, and the repo's home_io_guard (rightly) refuses real-home I/O.
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    # pm.paths.store_root() probes <install>/../manifest.json — the REAL home — unless
    # HERMES_RUNTIME_DIR is set, which is what trips home_io_guard during LocalEnvironment
    # session bootstrap (pm.shell.bash() -> _staged_bash() -> facts_path() -> store_root()).
    # It must be set HERE and not in the outer shell: conftest captures the guarded roots
    # BEFORE sandboxing, so a shell-exported value becomes a guarded root of its own and the
    # guard then refuses I/O against the very directory the test pointed itself at.
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(runtime))
    # bash resolution must not probe the real home either: pm.shell.windows_bash_candidates()
    # tries %LOCALAPPDATA%\hermes\git\... before Git for Windows, and os.path.isfile() on that
    # is Python file I/O under the guarded root. The override is checked first and lives
    # outside the home, so bash() returns without ever stat-ing the install tree.
    monkeypatch.setenv("HERMES_GIT_BASH_PATH", real_bash)
    home = tmp_path / "home"
    cloud = home / "OneDrive - Personal" / "notes"
    cloud.mkdir(parents=True)
    (cloud / "a.txt").write_text("needle in cloud\n", encoding="utf-8")
    proj = tmp_path / "work" / "proj"
    proj.mkdir(parents=True)
    (proj / "b.txt").write_text("needle in proj\n", encoding="utf-8")
    monkeypatch.setattr("tools.file_operations._HOME", str(home))
    return {"home": home, "cloud": cloud, "proj": proj}


def _ops(root):
    return ShellFileOperations(LocalEnvironment(cwd=str(root)), cwd=str(root))


_METHODS = ["_search_with_grep"]
if shutil.which("rg"):
    _METHODS.append("_search_with_rg")


@pytest.mark.parametrize("method", _METHODS)
class TestContentSearchGuard:
    def test_home_root_refused(self, method, guarded_tree):
        res = _search(_ops(guarded_tree["home"]), method, "needle", guarded_tree["home"])
        assert res.error and "Search refused" in res.error
        assert res.error.startswith("Search refused: root")
        assert res.matches == []

    def test_cloud_named_root_refused(self, method, guarded_tree):
        res = _search(_ops(guarded_tree["home"]), method, "needle", guarded_tree["cloud"])
        assert res.error and "Search refused" in res.error
        # The refusal names the MATCHED reason, not a generic label (a root
        # inside OneDrive is not an "ancestor of" it).
        assert "OneDrive - Personal" in res.error

    def test_narrow_root_still_works(self, method, guarded_tree):
        res = _search(_ops(guarded_tree["proj"]), method, "needle", guarded_tree["proj"])
        assert res.error is None
        assert any("needle in proj" in m.content for m in res.matches)

    def test_opt_in_admits_broad_root(self, method, guarded_tree):
        res = _search(_ops(guarded_tree["home"]), method, "needle",
                      guarded_tree["cloud"], broad_root_opt_in=True)
        assert res.error is None
        assert any("needle in cloud" in m.content for m in res.matches)

    def test_generic_box_name_boundary(self, method, guarded_tree, tmp_path):
        """``box`` is boundary-anchored: the exact name matches, lookalikes don't."""
        from tools.file_operations_search import _component_is_cloud_folder
        assert _component_is_cloud_folder("box")
        assert _component_is_cloud_folder("Box Sync")
        assert _component_is_cloud_folder("iCloudDrive")
        assert _component_is_cloud_folder("OneDrive - Personal")
        assert not _component_is_cloud_folder("boxed")
        assert not _component_is_cloud_folder("mailboxes")


def _search(ops, method, pattern, path, broad_root_opt_in=False, **kw):
    fn = getattr(ops, method)
    return fn(pattern, str(path), kw.get("file_glob"), kw.get("limit", 50),
              kw.get("offset", 0), kw.get("output_mode", "content"),
              kw.get("context", 0), broad_root_opt_in=broad_root_opt_in)


class TestGrepLaneForced:
    """A host without rg reaches grep via _search_content — the lane must be
    guarded there too (forced here via _has_command, since this host has rg)."""

    def test_content_dispatch_greps_guarded(self, guarded_tree, monkeypatch):
        ops = _ops(guarded_tree["home"])
        monkeypatch.setattr(ops, "_has_command", lambda c: c == "grep")
        res = ops._search_content("needle", str(guarded_tree["cloud"]),
                                  None, 50, 0, "content", 0)
        assert res.error and "Search refused" in res.error


class TestHandlerOptIn:
    """The real handler threads broad_root_opt_in from the tool args."""

    def test_handler_opt_in_flows(self, guarded_tree, monkeypatch):
        import tools.file_tools as ft
        args = {"pattern": "needle", "target": "content",
                "path": str(guarded_tree["cloud"]), "broad_root_opt_in": True}
        out = ft._handle_search_files(args)
        assert "needle in cloud" in out

    def test_handler_default_refuses(self, guarded_tree):
        import tools.file_tools as ft
        args = {"pattern": "needle", "target": "content",
                "path": str(guarded_tree["cloud"])}
        out = ft._handle_search_files(args)
        assert "Search refused" in out


class TestFileSearchGuard:
    """rg --files reads no file data, so it cannot hydrate — the refusal is a
    bounded-traversal guard. Kept for symmetry; still refused by default."""

    def test_files_search_home_root_refused(self, guarded_tree):
        ops = _ops(guarded_tree["home"])
        res = ops._search_files("*.txt", str(guarded_tree["home"]), 50, 0)
        assert res.error and "Search refused" in res.error

    def test_files_search_opt_in_admits(self, guarded_tree):
        ops = _ops(guarded_tree["home"])
        res = ops._search_files("*.txt", str(guarded_tree["cloud"]), 50, 0,
                                broad_root_opt_in=True)
        assert res.error is None
        assert res.files and "a.txt" in res.files[0]
