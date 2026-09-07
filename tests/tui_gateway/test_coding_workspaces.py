"""Opt-in workspace preparation uses real Git and the profile Projects registry."""
import shutil
import subprocess
from pathlib import Path

import pytest

import tui_gateway.server as server


def call(method, **params):
    response = server._methods.get(method)
    assert response is not None, f"missing workspace method: {method}"
    result = response(1, params)
    assert "error" not in result, result
    return result["result"]


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def test_new_worktree_does_not_dirty_source_or_edit_project_ignore(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "-c", "user.name=Test", "-c", "user.email=test@localhost", "commit", "--allow-empty", "-m", "base")
    exclude = repo / ".git" / "info" / "exclude"
    exclude.write_bytes(b"# preserve local rules without a trailing newline")
    assert git(repo, "status", "--porcelain") == ""
    call("projects.workspace.prepare", path=str(repo), mode="worktree", requestId="clean-root")
    assert git(repo, "status", "--porcelain") == ""
    assert not (repo / ".gitignore").exists()
    assert exclude.read_bytes().startswith(b"# preserve local rules without a trailing newline\n")
    before = exclude.read_bytes()
    call("projects.workspace.prepare", path=str(repo), mode="worktree", requestId="clean-root")
    assert exclude.read_bytes() == before


def test_managed_exclusion_does_not_follow_metadata_symlink(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "-c", "user.name=Test", "-c", "user.email=test@localhost", "commit", "--allow-empty", "-m", "base")
    outside = tmp_path / "outside.txt"
    outside.write_text("keep me")
    exclude = repo / ".git" / "info" / "exclude"
    exclude.unlink()
    exclude.symlink_to(outside)
    status_before = git(repo, "status", "--porcelain")
    worktrees_before = git(repo, "worktree", "list", "--porcelain")
    branches_before = git(repo, "branch", "--list")
    result = server._methods["projects.workspace.prepare"](1, dict(path=str(repo), mode="worktree", requestId="symlink-exclude"))
    assert "error" in result
    assert outside.read_text() == "keep me"
    assert git(repo, "status", "--porcelain") == status_before
    assert git(repo, "worktree", "list", "--porcelain") == worktrees_before
    assert git(repo, "branch", "--list") == branches_before
    assert not (repo / ".worktrees").exists()


def test_inspect_is_read_only_prepare_is_idempotent_and_owned(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "-c", "user.name=Test", "-c", "user.email=test@localhost", "commit", "--allow-empty", "-m", "base")
    (repo / "dirty.txt").write_text("keep me")
    before = call("projects.list")
    inspected = call("projects.workspace.inspect", path=str(repo))
    assert inspected["branch"] == "main" and inspected["dirty"]
    assert "main" in inspected["branches"]
    assert inspected["worktrees"][0]["dirty"]
    assert inspected["worktrees"][0]["activeSessionCount"] == 0
    assert not (repo / ".worktrees").exists()
    assert call("projects.list") == before
    intent = dict(path=str(repo), mode="worktree", requestId="task-a")
    prepared = call("projects.workspace.prepare", **intent)
    repeated = call("projects.workspace.prepare", **intent)
    assert repeated == prepared
    cwd = Path(prepared["cwd"])
    assert cwd.parent == repo / ".worktrees"
    assert git(cwd, "rev-parse", "--show-toplevel") == str(cwd)
    assert git(repo, "branch", "--show-current") == "main"
    assert not (cwd / "dirty.txt").exists()
    assert (repo / "dirty.txt").read_text() == "keep me"
    assert len(call("projects.workspace.inspect", path=str(repo))["worktrees"]) == 2
    project = call("projects.get", id=prepared["projectId"])["project"]
    assert project["primary_path"] == str(repo)
    assert call("projects.list")["active_id"] == before["active_id"]
    current = call("projects.workspace.prepare", path=str(cwd), mode="current")
    assert current["cwd"] == str(repo) and current["branch"] == "main"
    existing = call("projects.workspace.prepare", path=str(repo), mode="existing", existingPath=str(cwd))
    assert existing["cwd"] == str(cwd)
    assert existing["sourcePath"] == str(repo)
    assert current["sourcePath"] == str(cwd)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    refused = server._methods["projects.workspace.prepare"](1, dict(path=str(repo), mode="existing", existingPath=str(foreign)))
    assert "error" in refused
    assert len(call("projects.workspace.inspect", path=str(repo))["worktrees"]) == 2


def test_register_subdirectory_and_linked_checkout_share_canonical_project(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "-c", "user.name=Test", "-c", "user.email=test@localhost", "commit", "--allow-empty", "-m", "base")
    subdir = repo / "subdir"
    subdir.mkdir()
    linked = tmp_path / "linked"
    git(repo, "worktree", "add", "-b", "linked", str(linked))
    first = call("projects.workspace.register", path=str(subdir))["project"]
    second = call("projects.workspace.register", path=str(linked))["project"]
    assert first["id"] == second["id"]
    assert first["primary_path"] == str(repo)
    for path in (subdir, linked):
        prepared = call("projects.workspace.prepare", path=str(path), projectId=first["id"], mode="existing", existingPath=str(linked))
        assert prepared["projectId"] == first["id"]
    before = call("projects.list")
    other = tmp_path / "other"
    other.mkdir()
    for params in ({"mode": "folder", "projectId": first["id"]}, {"mode": "worktree"}, {"mode": "invalid"}):
        refused = server._methods["projects.workspace.prepare"](1, {"path": str(other), **params})
        assert "error" in refused
        assert call("projects.list") == before


def test_git_probe_errors_never_become_non_git(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    # A real repository with unreadable-by-Git configuration, not a simulated probe.
    (repo / ".git" / "config").write_text("[broken\n")
    refused = server._methods["projects.workspace.inspect"](1, {"path": str(repo)})
    assert "error" in refused, refused
    folder = tmp_path / "folder"
    folder.mkdir()
    assert call("projects.workspace.inspect", path=str(folder))["repoRoot"] is None
    with monkeypatch.context() as env:
        env.setenv("GIT_DIR", str(tmp_path / "missing-git-dir"))
        refused = server._methods["projects.workspace.inspect"](1, {"path": str(folder)})
        assert "error" in refused, refused
    monkeypatch.setenv("PATH", str(tmp_path / "no-git"))
    refused = server._methods["projects.workspace.inspect"](1, {"path": str(folder)})
    assert "error" in refused, refused


def test_inline_workspace_references_use_original_cwd_and_expand_new_checkout(tmp_path, monkeypatch):
    from hermes_state import SessionDB
    from agent.context_references import parse_context_references, preprocess_context_references
    db = SessionDB(tmp_path / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    repo = tmp_path / "retry-app"
    repo.mkdir()
    (repo / "README.md").write_text("first\nCHECKOUT CONTENT\nthird\n")
    (repo / "folder name").mkdir()
    (repo / "folder name" / "note.txt").write_text("tracked")
    git(repo, "init", "-b", "main")
    git(repo, "add", ".")
    git(repo, "-c", "user.name=Test", "-c", "user.email=test@localhost", "commit", "-m", "base")
    from hermes_constants import get_hermes_home
    import json
    monkeypatch.setattr(server, "_hermes_home", str(get_hermes_home()))
    (get_hermes_home() / "config.yaml").write_text(json.dumps({"terminal": {"cwd": str(tmp_path)}}))
    completion = call("complete.path", word="@file:retry-app/READ", cwd=None)
    assert any(item["text"] == "@file:retry-app/README.md" for item in completion["items"])
    inspection = call("complete.path", word="")
    assert inspection["sourceCwd"] == str(tmp_path)
    assert not (repo / ".worktrees").exists()
    prepared = call("projects.workspace.prepare", path=str(repo), mode="worktree", requestId="inline-refs")
    created = call("session.create", source="desktop", cwd=prepared["cwd"], coding_workspace=prepared)
    target = Path(prepared["cwd"])
    (repo / "README.md").write_text("WRONG SOURCE COPY")
    outside = tmp_path / "other.md"
    outside.write_text("EXTERNAL CONTENT")
    text = (f'Keep retry-app/README.md prose; (see @file:retry-app/README.md:2-2), '
            f'@folder:`retry-app/folder name` @file:"{repo}/README.md":2 '
            f'@file:`retry-app/README.md` @file:other.md @file:{outside} @url:https://example.com @diff')
    params = dict(session_id=created["session_id"], paths=["retry-app/README.md"], text=text, reference_cwd=inspection["sourceCwd"])
    result = call("session.workspace.references", **params)
    assert result["paths"] == [str(target / "README.md")]
    refs = parse_context_references(result["text"])
    assert [ref.target for ref in refs[:6]] == [str(target / "README.md"), str(target / "folder name"),
                                               str(target / "README.md"), str(target / "README.md"), str(outside), str(outside)]
    assert f'(see @file:{target}/README.md:2-2),' in result["text"]
    assert result["text"].startswith('Keep retry-app/README.md prose; ')
    assert result["text"].endswith(f'@file:{outside} @url:https://example.com @diff')
    file_text = " ".join(ref.raw for ref in refs[:4])
    expanded = preprocess_context_references(file_text, cwd=target, context_length=100000)
    assert not expanded.warnings
    assert "CHECKOUT CONTENT" in expanded.message
    assert "WRONG SOURCE COPY" not in expanded.message
    (target / "README.md").unlink()
    refused = server._methods["session.workspace.references"](1, params)
    assert "error" in refused and "checkout" in refused["error"]["message"]
    assert params["text"] == text
    # The original composer may itself be in a sibling of the selected repo.
    sibling = tmp_path / "sibling"
    sibling.mkdir()
    (target / "README.md").write_text("RESTORED CHECKOUT")
    parent_relative = call("session.workspace.references", session_id=created["session_id"], paths=[],
                           text="@file:../retry-app/README.md", reference_cwd=str(sibling))
    assert parse_context_references(parent_relative["text"])[0].target == str(target / "README.md")
    db.close()


def test_completion_base_inspection_uses_exact_owner_fallback(tmp_path, monkeypatch):
    import json
    from hermes_cli.profiles import get_profile_dir
    home = tmp_path / "deployment"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(server, "_hermes_home", str(home))
    launch, alpha, ambient = (tmp_path / name for name in ("launch-cwd", "alpha-cwd", "ambient"))
    for folder in (launch, alpha, ambient):
        folder.mkdir()
        (folder / f"{folder.name}.txt").write_text(folder.name)
    (home / "config.yaml").write_text(json.dumps({"terminal": {"cwd": str(launch)}}))
    monkeypatch.chdir(ambient)
    monkeypatch.setenv("TERMINAL_CWD", str(ambient))
    for profile in ("alpha", "beta"):
        owned = get_profile_dir(profile)
        assert owned.is_relative_to(tmp_path)
        owned.mkdir(parents=True)
    (get_profile_dir("alpha") / "config.yaml").write_text(json.dumps({"terminal": {"cwd": str(alpha)}}))
    # beta intentionally has no configured cwd: preserve complete.path's launch fallback too.
    for profile, expected in (("alpha", alpha), ("beta", launch), ("default", launch)):
        base = call("complete.path", word="", profile=profile, cwd=None)["sourceCwd"]
        menu = call("complete.path", word="@file:", profile=profile, cwd=None)
        assert base == str(expected)
        assert [item["text"] for item in menu["items"]] == [f"@file:{expected.name}.txt"]
        explicit = call("complete.path", word="", profile=profile, cwd=str(ambient))
        assert explicit["sourceCwd"] == str(ambient)


def test_workspace_references_only_remap_explicit_project_paths(tmp_path):
    from tui_gateway.coding_workspaces import remap_workspace_references
    import pytest
    source, target, other = (tmp_path / name for name in ("repo", "checkout", "other"))
    for path in (source, target, other):
        path.mkdir()
    for root in (source, target):
        (root / "file with spaces.py").write_text("source")
    binding = {"sourcePath": str(source), "cwd": str(target)}
    paths = [str(source), str(source / "file with spaces.py"), str(other), str(tmp_path / "repo-other"), "relative.py"]
    original = list(paths)
    assert remap_workspace_references(binding, paths) == [str(target), str(target / "file with spaces.py"), None, None, None]
    assert paths == original
    (source / "untracked").write_text("not in checkout")
    with pytest.raises(ValueError, match="checkout"):
        remap_workspace_references(binding, [str(source / "untracked")])
    (source / "escape").symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="checkout"):
        remap_workspace_references(binding, [str(source / "escape")])
    (source / "unsafe").mkdir()
    (target / "unsafe").symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="checkout"):
        remap_workspace_references(binding, [str(source / "unsafe")])
    assert remap_workspace_references(None, paths) == [None] * len(paths)


def test_coding_create_retry_recovers_one_durable_session(tmp_path, monkeypatch):
    from hermes_state import SessionDB
    from hermes_constants import get_hermes_home
    db = SessionDB(tmp_path / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    folder = tmp_path / "folder"
    folder.mkdir()
    prepared = call("projects.workspace.prepare", path=str(folder), mode="folder", requestId="lost-reply")
    params = dict(source="desktop", cwd=str(folder), coding_workspace={**prepared, "requestId": "lost-reply"})
    first = call("session.create", **params)
    retry = call("session.create", **params)
    assert retry["stored_session_id"] == first["stored_session_id"]
    assert retry["session_id"] == first["session_id"]
    server._sessions.pop(first["session_id"])
    recovered = call("session.create", **params)
    assert recovered["stored_session_id"] == first["stored_session_id"]
    assert db._conn.execute("SELECT count(*) FROM sessions").fetchone()[0] == 1
    artifacts = get_hermes_home() / "cache" / "session-artifacts"
    assert [p.name for p in artifacts.iterdir()] == [first["stored_session_id"]]
    assert prepared["requestId"] == "lost-reply"
    changed = {**params, "coding_workspace": {**params["coding_workspace"], "projectId": "different"}}
    assert "error" in server._methods["session.create"](1, changed)
    db.close()


def test_coding_session_persists_exact_cwd_and_refuses_missing_checkout(tmp_path, monkeypatch):
    from hermes_state import SessionDB
    db = SessionDB(tmp_path / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    folder = tmp_path / "folder"
    folder.mkdir()
    prepared = call("projects.workspace.prepare", path=str(folder), mode="folder")
    created = call("session.create", source="desktop", cwd=str(folder), coding_workspace=prepared)
    stored = db.get_session(created["stored_session_id"])
    assert stored is not None and stored["cwd"] == str(folder)
    overrides = server._stored_session_runtime_overrides(stored)
    assert overrides["coding_workspace"]["cwd"] == str(folder)
    from hermes_constants import get_hermes_home
    artifacts = Path(overrides["coding_workspace"]["artifactsPath"])
    assert artifacts.is_dir() and artifacts.is_relative_to(get_hermes_home())
    assert not artifacts.is_relative_to(folder)
    from tui_gateway.coding_workspaces import workspace_instructions
    instructions = workspace_instructions(overrides["coding_workspace"])
    assert str(artifacts) in instructions and str(folder) in instructions
    assert workspace_instructions(overrides["coding_workspace"]) == instructions
    verified = call("session.workspace.verify", session_id=created["session_id"], cwd=str(folder))
    assert verified["cwd"] == str(folder) and verified["gatewayCwd"] == str(folder)
    assert "workerCwd" not in verified
    assert server._display_session_cwd(server._sessions[created["session_id"]]) == str(folder)
    assert call("session.workspace.references", session_id=created["session_id"], paths=[str(folder)]) == {"paths": [str(folder)]}
    folder.rmdir()
    # Status/resume must not silently heal a deleted chosen checkout to an ancestor.
    assert server._display_session_cwd(server._sessions[created["session_id"]]) == str(folder)
    refused = server._methods["prompt.submit"](1, {"session_id": created["session_id"], "text": "do not dispatch"})
    assert "error" in refused and "workspace" in refused["error"]["message"].lower()
    assert not server._sessions[created["session_id"]]["running"]
    folder.mkdir()
    # Simulate loss of volatile gateway state, then take the real lazy resume path.
    server._sessions.pop(created["session_id"])
    resumed = call("session.resume", session_id=created["stored_session_id"], source="desktop", lazy=True)
    session = server._sessions[resumed["session_id"]]
    assert session["coding_workspace"]["cwd"] == str(folder)
    assert call("session.workspace.verify", session_id=resumed["session_id"], cwd=str(folder))["gatewayCwd"] == str(folder)
    db.close()


def _repo(tmp_path, name="repo"):
    repo = tmp_path / name
    repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "-c", "user.name=Test", "-c", "user.email=test@localhost", "commit", "--allow-empty", "-m", "base")
    return repo


@pytest.mark.parametrize("deletion", ["before-inspection", "during-status"])
def test_missing_sibling_does_not_block_workspace_use(tmp_path, monkeypatch, deletion):
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB
    from tui_gateway import coding_workspaces

    repo = _repo(tmp_path)
    sibling = tmp_path / "unrelated"
    git(repo, "worktree", "add", "-b", "unrelated", str(sibling))
    with SessionDB(tmp_path / "state.db") as db, monkeypatch.context() as monkeypatch:
        monkeypatch.setattr(server, "_hermes_home", str(get_hermes_home()))
        monkeypatch.setattr(server, "_get_db", lambda: db)
        monkeypatch.setattr(server, "_schedule_agent_build", lambda sid: None)
        monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
        selected = tmp_path / "selected"
        git(repo, "worktree", "add", "-b", "selected", str(selected))
        prepared = call("projects.workspace.prepare", path=str(repo), mode="existing", existingPath=str(selected), requestId="existing-owner")
        created = call("session.create", source="desktop", cwd=prepared["cwd"], coding_workspace=prepared)
        (Path(prepared["cwd"]) / "dirty.txt").write_text("keep this change", encoding="utf-8")
        if deletion == "before-inspection":
            shutil.rmtree(sibling)
        else:
            original = coding_workspaces._git_output

            def delete_during_status(path, args):
                if path == str(sibling) and args == ["status", "--porcelain"] and sibling.exists():
                    shutil.rmtree(sibling)
                return original(path, args)

            monkeypatch.setattr(coding_workspaces, "_git_output", delete_during_status)

        inspected = call("projects.workspace.inspect", path=str(repo))
        assert inspected["repoRoot"] == str(repo)
        trees = {tree["path"]: tree for tree in inspected["worktrees"]}
        assert str(sibling) not in trees
        assert trees[str(repo)]["isMain"]
        assert trees[prepared["cwd"]]["dirty"]
        assert trees[prepared["cwd"]]["activeSessionCount"] == 1
        assert call("session.workspace.verify", session_id=created["session_id"], cwd=prepared["cwd"])["cwd"] == prepared["cwd"]
        existing = call("projects.workspace.prepare", path=str(repo), mode="existing", existingPath=prepared["cwd"])
        assert existing["cwd"] == prepared["cwd"]
        fresh = call("projects.workspace.prepare", path=str(repo), mode="worktree", requestId="new-owner")
        assert fresh["cwd"] != prepared["cwd"]
        assert git(Path(fresh["cwd"]), "rev-parse", "--show-toplevel") == fresh["cwd"]
        assert (Path(prepared["cwd"]) / "dirty.txt").read_text(encoding="utf-8") == "keep this change"
        registrations = git(repo, "worktree", "list", "--porcelain", "-z")
        assert f"worktree {sibling}\0" in registrations
        assert "branch refs/heads/unrelated\0" in registrations
        assert not sibling.exists()

        # Move the primary after Git enumerates it, keeping the selected linked
        # checkout connected to the real metadata. Do not invent porcelain data.
        probe_before_move = coding_workspaces._git_output
        relocated = tmp_path / "relocated"

        def move_primary_after_listing(path, args):
            output = probe_before_move(path, args)
            if args == ["worktree", "list", "--porcelain", "-z"] and repo.exists():
                repo.rename(relocated)
                (selected / ".git").write_text(
                    f"gitdir: {relocated / '.git' / 'worktrees' / selected.name}\n", encoding="utf-8")
            return output

        monkeypatch.setattr(coding_workspaces, "_git_output", move_primary_after_listing)
        inspected = call("projects.workspace.inspect", path=str(selected))
        assert inspected["repoRoot"] == str(repo)
        assert not repo.exists()
        assert [tree["path"] for tree in inspected["worktrees"]] == [str(selected)]
        assert not inspected["worktrees"][0]["isMain"]


@pytest.mark.parametrize("failure", ["missing-selected", "deleted-selected-during-status", "corrupt-selected", "corrupt-sibling"])
def test_workspace_inspection_does_not_hide_selected_or_existing_errors(tmp_path, monkeypatch, failure):
    from tui_gateway import coding_workspaces

    repo = _repo(tmp_path)
    selected = tmp_path / "selected"
    git(repo, "worktree", "add", "-b", "selected", str(selected))
    target = repo if failure == "corrupt-sibling" else selected
    if failure == "missing-selected":
        shutil.rmtree(selected)
    elif failure == "deleted-selected-during-status":
        original = coding_workspaces._git_output

        def delete_selected_during_status(path, args):
            if path == str(selected) and args == ["status", "--porcelain"] and selected.exists():
                shutil.rmtree(selected)
            return original(path, args)

        monkeypatch.setattr(coding_workspaces, "_git_output", delete_selected_during_status)
    else:
        # An existing checkout with corrupt Git metadata is not a stale folder.
        index = Path(git(target, "rev-parse", "--path-format=absolute", "--git-path", "index"))
        index.write_bytes(b"invalid index")
    response = server._methods["projects.workspace.inspect"](1, {"path": str(selected)})
    assert "error" in response, response
    message = response["error"]["message"]
    assert "index" in message if failure.startswith("corrupt") else (
        "No such file or directory" in message or "git invocation failed" in message)
    assert f"worktree {selected}\0" in git(repo, "worktree", "list", "--porcelain", "-z")


def _claims():
    import sqlite3
    from hermes_constants import get_hermes_home
    with sqlite3.connect(get_hermes_home() / "projects.db") as db:
        return {row[0]: row[1] for row in db.execute("SELECT request_id, state FROM workspace_claims")}


def test_prepare_failure_after_worktree_creation_leaves_no_ownerless_checkout(tmp_path, monkeypatch):
    from tui_gateway import coding_workspaces
    repo = _repo(tmp_path)
    worktrees_before = git(repo, "worktree", "list", "--porcelain")
    branches_before = git(repo, "branch", "--list")

    def explode(pdb, conn, path):
        raise RuntimeError("registry unavailable after the checkout exists")
    monkeypatch.setattr(coding_workspaces, "register_folder", explode)
    result = server._methods["projects.workspace.prepare"](1, dict(path=str(repo), mode="worktree", requestId="doomed"))
    assert "error" in result
    assert git(repo, "worktree", "list", "--porcelain") == worktrees_before
    assert git(repo, "branch", "--list") == branches_before
    assert "doomed" not in _claims()


def test_bound_worktree_is_never_adopted_by_a_later_draft(tmp_path, monkeypatch):
    from hermes_state import SessionDB
    db = SessionDB(tmp_path / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    repo = _repo(tmp_path)
    prepared = call("projects.workspace.prepare", path=str(repo), mode="worktree", requestId="task-one")
    assert _claims()["task-one"] == "prepared"
    call("session.create", source="desktop", cwd=prepared["cwd"], coding_workspace=prepared)
    assert _claims()["task-one"] == "bound"
    later = call("projects.workspace.prepare", path=str(repo), mode="worktree", requestId="task-two")
    assert later["cwd"] != prepared["cwd"] and later["requestId"] == "task-two"
    db.close()
