"""Invariants for the authoritative project-tree builder (tui_gateway.project_tree).

These assert structural contracts (worktree folding, kanban collapse, lane id
scheme, membership union) rather than snapshots, so routine data changes don't
break them.
"""

from __future__ import annotations

from tui_gateway import project_tree as pt

_SID = 0


def _session(cwd, *, branch="", repo_root="", **over):
    global _SID
    _SID += 1
    row = {
        "id": f"s{_SID}",
        "cwd": cwd,
        "git_branch": branch,
        "git_repo_root": repo_root,
        "started_at": 1000,
        "last_active": 1000,
        "title": None,
        "preview": None,
        "source": "cli",
    }
    row.update(over)
    return row


def _project(pid, name, folders, **over):
    row = {
        "id": pid,
        "name": name,
        "primary_path": folders[0] if folders else None,
        "archived": False,
        "folders": [{"path": p, "is_primary": i == 0} for i, p in enumerate(folders)],
    }
    row.update(over)
    return row


def _resolver(mapping):
    """Build a resolve() from {cwd: (repo_root, worktree_root)}."""

    def resolve(cwd):
        hit = mapping.get(cwd)
        if not hit:
            return None
        return {"repo_root": hit[0], "worktree_root": hit[1]}

    return resolve


def _lane_ids(project):
    return [g["id"] for repo in project["repos"] for g in repo["groups"]]


def _home(tree):
    """The Home bucket, or None when nothing was left unplaced."""
    return next((p for p in tree["projects"] if p["id"] == pt.NO_PROJECT_ID), None)


def _home_session_ids(tree):
    home = _home(tree)
    return [s["id"] for s in _sessions_of(home)] if home else []


def _sessions_of(project):
    return [s for repo in project["repos"] for g in repo["groups"] for s in g["sessions"]]


def _real_project_ids(tree):
    """Project ids excluding the Home bucket (which is always present when any
    session went unplaced, so asserting on it in every test would be noise)."""
    return [p["id"] for p in tree["projects"] if p["id"] != pt.NO_PROJECT_ID]


# ---------------------------------------------------------------------------


def test_main_checkout_groups_by_recorded_branch_with_stable_lane_ids():
    resolve = _resolver({"/repo": ("/repo", "/repo")})
    sessions = [
        _session("/repo", branch="main"),
        _session("/repo", branch="feature"),
    ]

    tree = pt.build_tree([], sessions, [], resolve, hydrate=True)
    project = next(p for p in tree["projects"] if p["id"] == "/repo")

    assert project["isAuto"] is True
    assert _lane_ids(project) == ["/repo::branch::main", "/repo::branch::feature"]
    # Trunk sorts ahead of the feature branch; both live in the main checkout.
    assert [g["label"] for repo in project["repos"] for g in repo["groups"]] == ["main", "feature"]
    assert all(g["isMain"] for repo in project["repos"] for g in repo["groups"])


def test_linked_worktrees_fold_under_their_common_repo_root():
    # The linked worktree's own toplevel is /elsewhere/wt, but its COMMON root is
    # /repo, so it must group under /repo (not as a separate project).
    resolve = _resolver(
        {
            "/repo": ("/repo", "/repo"),
            "/elsewhere/wt": ("/repo", "/elsewhere/wt"),
        }
    )
    sessions = [
        _session("/repo", branch="main"),
        _session("/elsewhere/wt", branch="feature"),
    ]

    tree = pt.build_tree([], sessions, [], resolve, hydrate=True)

    assert [p["id"] for p in tree["projects"]] == ["/repo"]
    project = tree["projects"][0]
    assert project["repos"][0]["id"] == "/repo"
    lane_ids = _lane_ids(project)
    assert "/repo::branch::main" in lane_ids
    # Linked worktree lane is keyed by the worktree path and is not main.
    linked = next(g for repo in project["repos"] for g in repo["groups"] if not g["isMain"])
    assert linked["id"] == "/elsewhere/wt"
    assert linked["path"] == "/elsewhere/wt"


def test_kanban_task_worktrees_collapse_into_one_bucket():
    resolve = _resolver(
        {
            "/repo": ("/repo", "/repo"),
            "/repo/.worktrees/t_aaaaaaaa": ("/repo", "/repo/.worktrees/t_aaaaaaaa"),
            "/repo/.worktrees/t_bbbbbbbb": ("/repo", "/repo/.worktrees/t_bbbbbbbb"),
        }
    )
    sessions = [
        _session("/repo", branch="main"),
        _session("/repo/.worktrees/t_aaaaaaaa"),
        _session("/repo/.worktrees/t_bbbbbbbb"),
    ]

    tree = pt.build_tree([], sessions, [], resolve, hydrate=True)
    project = tree["projects"][0]
    kanban = [g for repo in project["repos"] for g in repo["groups"] if g.get("isKanban")]

    assert len(kanban) == 1
    assert kanban[0]["id"] == "/repo::kanban"
    assert kanban[0]["path"] == "/repo/.worktrees"
    assert len(kanban[0]["sessions"]) == 2
    # The bucket sorts below the real main branch.
    assert _lane_ids(project)[-1] == "/repo::kanban"


def test_user_worktree_under_dotworktrees_is_its_own_lane_not_kanban():
    # A user "New worktree" lives at <repo>/.worktrees/<slug> (no t_ id), so it
    # must NOT collapse into the kanban bucket — it gets its own linked lane.
    resolve = _resolver(
        {
            "/repo": ("/repo", "/repo"),
            "/repo/.worktrees/test-gui-stuff": ("/repo", "/repo/.worktrees/test-gui-stuff"),
        }
    )
    sessions = [
        _session("/repo", branch="main"),
        _session("/repo/.worktrees/test-gui-stuff", branch="hermes/test-gui-stuff"),
    ]

    tree = pt.build_tree([], sessions, [], resolve, hydrate=True)
    project = tree["projects"][0]
    lanes = {g["id"]: g for repo in project["repos"] for g in repo["groups"]}

    assert "/repo/.worktrees/test-gui-stuff" in lanes
    assert not lanes["/repo/.worktrees/test-gui-stuff"].get("isKanban")
    assert "/repo::kanban" not in lanes


def test_unrecorded_and_recorded_main_share_one_lane():
    # Empty git_branch (historical sessions) folds into the same trunk lane as
    # sessions that recorded branch "main" — no duplicate "main".
    resolve = _resolver({"/repo": ("/repo", "/repo")})
    sessions = [_session("/repo", branch=""), _session("/repo", branch="main")]

    tree = pt.build_tree([], sessions, [], resolve, hydrate=True)
    project = tree["projects"][0]
    main_lanes = [g for repo in project["repos"] for g in repo["groups"] if g["label"] == "main"]

    assert len(main_lanes) == 1
    assert main_lanes[0]["id"] == "/repo::branch::main"
    assert len(main_lanes[0]["sessions"]) == 2


def test_persisted_repo_root_used_when_no_live_probe():
    # No resolver (remote backend): fall back to the persisted git_repo_root and
    # split the main checkout by the session's recorded branch.
    sessions = [_session("/repo/src", branch="main", repo_root="/repo")]

    tree = pt.build_tree([], sessions, [], resolve=None, hydrate=True)
    project = next(p for p in tree["projects"] if p["id"] == "/repo")

    assert _lane_ids(project) == ["/repo::branch::main"]


def test_non_git_cwd_preserves_legacy_workspace_grouping():
    # Before first-class Projects, every non-empty session cwd appeared as a
    # workspace even when it was not a git repo. Historical sessions must keep
    # that grouping instead of falling through to the flat Sessions list.
    legacy = _session("/work/notes", title="Research notes")

    tree = pt.build_tree([], [legacy], [], resolve=lambda _cwd: None, hydrate=True)

    assert [p["id"] for p in tree["projects"]] == ["/work/notes"]
    project = tree["projects"][0]
    assert project["isAuto"] is True
    assert project["label"] == "notes"
    assert project["sessionCount"] == 1
    assert _lane_ids(project) == ["/work/notes"]
    assert tree["scoped_session_ids"] == [legacy["id"]]


def test_non_git_windows_cwd_preserves_legacy_workspace_grouping():
    cwd = r"C:\Users\alice\workspace\notes"
    legacy = _session(cwd)

    tree = pt.build_tree([], [legacy], [], resolve=lambda _cwd: None, hydrate=True)

    assert [p["id"] for p in tree["projects"]] == [cwd]
    assert tree["projects"][0]["label"] == "notes"
    assert tree["scoped_session_ids"] == [legacy["id"]]


def test_equivalent_windows_cwds_collapse_into_one_auto_project():
    sessions = [
        _session("C:/work/notes"),
        _session(r"c:\WORK\notes"),
        _session("C:/work/notes/"),
    ]

    tree = pt.build_tree([], sessions, [], resolve=lambda _cwd: None, hydrate=True)

    assert len(tree["projects"]) == 1
    project = tree["projects"][0]
    assert project["id"] == "C:/work/notes"
    assert project["sessionCount"] == 3
    assert len(project["repos"]) == 1
    assert len(project["repos"][0]["groups"]) == 1
    assert len(project["repos"][0]["groups"][0]["sessions"]) == 3


def test_windows_path_identity_preserves_explicit_project_priority():
    explicit = _project("p_notes", "Notes", ["C:/Work/Notes"])
    session = _session("c:\\work\\notes\\")

    tree = pt.build_tree([explicit], [session], [], resolve=lambda _cwd: None, hydrate=True)

    assert [p["id"] for p in tree["projects"]] == ["p_notes"]
    assert tree["projects"][0]["sessionCount"] == 1
    assert tree["scoped_session_ids"] == [session["id"]]


def test_wsl_localhost_cwds_collapse_into_one_auto_project():
    # Root-relative WSL spellings (single leading backslash) are Windows paths,
    # so case/separator variants collapse instead of spawning duplicate autos.
    sessions = [
        _session(r"\wsl.localhost\Ubuntu\home\alice\proj"),
        _session("//wsl.localhost/Ubuntu/home/alice/PROJ"),
    ]

    tree = pt.build_tree([], sessions, [], resolve=lambda _cwd: None, hydrate=True)

    assert len(tree["projects"]) == 1
    assert tree["projects"][0]["sessionCount"] == 2


def test_wsl_localhost_path_cannot_bypass_explicit_project():
    explicit = _project("p_proj", "Proj", [r"\wsl.localhost\Ubuntu\home\alice\proj"])
    session = _session("//wsl.localhost/Ubuntu/home/alice/PROJ/")

    tree = pt.build_tree([explicit], [session], [], resolve=lambda _cwd: None, hydrate=True)

    assert [p["id"] for p in tree["projects"]] == ["p_proj"]
    assert tree["projects"][0]["sessionCount"] == 1
    assert tree["scoped_session_ids"] == [session["id"]]


def test_posix_path_identity_remains_case_sensitive():
    explicit = _project("p_notes", "Notes", ["/Work/Notes"])
    session = _session("/work/notes")

    tree = pt.build_tree([explicit], [session], [], resolve=lambda _cwd: None, hydrate=True)

    assert [(p["id"], p["sessionCount"]) for p in tree["projects"]] == [
        ("p_notes", 0),
        ("/work/notes", 1),
    ]


def test_explicit_project_claims_sessions_and_beats_auto():
    project = _project("p_app", "App", ["/www/app"])
    resolve = _resolver(
        {
            "/www/app": ("/www/app", "/www/app"),
            "/www/other": ("/www/other", "/www/other"),
        }
    )
    sessions = [
        _session("/www/app", branch="main"),
        _session("/www/other", branch="main"),
    ]

    tree = pt.build_tree([project], sessions, [], resolve, hydrate=True)

    explicit = next(p for p in tree["projects"] if p["id"] == "p_app")
    assert explicit["isAuto"] is False
    assert explicit["sessionCount"] == 1
    # The unowned /www/other session becomes its own auto project.
    assert any(p["id"] == "/www/other" and p["isAuto"] for p in tree["projects"])


def test_scoped_session_ids_is_union_of_placed_sessions():
    project = _project("p_app", "App", ["/www/app"])
    resolve = _resolver(
        {
            "/www/app": ("/www/app", "/www/app"),
            "/www/repo": ("/www/repo", "/www/repo"),
        }
    )
    owned = _session("/www/app", branch="main")
    auto = _session("/www/repo", branch="main")
    homeless = _session(None)  # no cwd -> the Home bucket

    tree = pt.build_tree([project], [owned, auto, homeless], [], resolve, hydrate=True)

    assert set(tree["scoped_session_ids"]) == {owned["id"], auto["id"], homeless["id"]}
    assert _home_session_ids(tree) == [homeless["id"]]


def test_overview_drops_session_rows_but_keeps_counts_and_previews():
    resolve = _resolver({"/repo": ("/repo", "/repo")})
    sessions = [_session("/repo", branch="main") for _ in range(4)]

    tree = pt.build_tree([], sessions, [], resolve, preview_limit=3, hydrate=False)
    project = tree["projects"][0]

    assert project["sessionCount"] == 4
    assert len(project["previewSessions"]) == 3
    # Lanes carry structure + counts but no rows in overview mode.
    assert all(g["sessions"] == [] for repo in project["repos"] for g in repo["groups"])
    assert project["repos"][0]["sessionCount"] == 4


def test_discovered_repo_with_no_sessions_becomes_zero_session_project():
    discovered = [{"root": "/www/fresh", "label": "fresh", "sessions": 0, "last_active": 5}]

    tree = pt.build_tree([], [], discovered, resolve=None, hydrate=False)

    fresh = next(p for p in tree["projects"] if p["id"] == "/www/fresh")
    assert fresh["isAuto"] is True
    assert fresh["sessionCount"] == 0
    assert fresh["repos"][0]["groups"] == []


def test_explicit_project_with_no_sessions_seeds_its_folders_as_repos():
    # A brand-new (or unloaded) project must still expose its declared folders as
    # repos so the entered view renders and the desktop's optimistic overlay has a
    # lane to place a freshly-created session into (otherwise it only shows after a
    # full tree refresh).
    project = _project("p_new", "New", ["/work/blank"])

    tree = pt.build_tree([project], [], [], resolve=None, hydrate=True)

    node = next(p for p in tree["projects"] if p["id"] == "p_new")
    assert node["sessionCount"] == 0
    assert [r["path"] for r in node["repos"]] == ["/work/blank"]
    assert node["repos"][0]["groups"] == []


def test_seeded_folder_repo_does_not_duplicate_a_session_derived_repo():
    # When a folder already has sessions (same git root), seeding must not add a
    # second repo for the same path.
    project = _project("p_app", "App", ["/www/app"])
    resolve = _resolver({"/www/app": ("/www/app", "/www/app")})
    sessions = [_session("/www/app", branch="main")]

    tree = pt.build_tree([project], sessions, [], resolve, hydrate=True)

    node = next(p for p in tree["projects"] if p["id"] == "p_app")
    assert [r["path"] for r in node["repos"]] == ["/www/app"]


def test_non_git_project_folder_uses_the_standard_main_lane():
    # Non-git project folders have no resolver/persisted repo root, but they are
    # still main checkouts for the desktop tree. Use the same lane id the live
    # session overlay computes, otherwise the initial snapshot shows the folder
    # name and the overlay adds a duplicate "main" lane with the same session.
    project = _project("p_notes", "Notes", ["/work/notes"])
    session = _session("/work/notes")

    tree = pt.build_tree([project], [session], [], resolve=None, hydrate=True)

    node = next(p for p in tree["projects"] if p["id"] == "p_notes")
    lanes = node["repos"][0]["groups"]

    assert [g["id"] for g in lanes] == ["/work/notes::branch::main"]
    assert [g["label"] for g in lanes] == ["main"]
    assert lanes[0]["sessions"] == [session]


def test_discovered_repo_owned_by_explicit_project_is_not_duplicated():
    project = _project("p_app", "App", ["/www/app"])
    discovered = [{"root": "/www/app", "label": "app", "sessions": 2, "last_active": 1}]

    tree = pt.build_tree([project], [], discovered, resolve=None, hydrate=False)

    assert [p["id"] for p in tree["projects"] if p["path"] == "/www/app"] == ["p_app"]


def test_nested_project_folders_pick_the_deepest_match():
    # The folder index must resolve a session to its most-specific (deepest)
    # project folder, not just any ancestor.
    outer = _project("p_outer", "Outer", ["/work"])
    inner = _project("p_inner", "Inner", ["/work/app"])
    resolve = _resolver(
        {
            "/work/app": ("/work/app", "/work/app"),
            "/work/other": ("/work/other", "/work/other"),
        }
    )

    tree = pt.build_tree(
        [outer, inner],
        [_session("/work/app", branch="main"), _session("/work/other", branch="main")],
        [],
        resolve,
        hydrate=True,
    )
    by_id = {p["id"]: p for p in tree["projects"]}

    assert by_id["p_inner"]["sessionCount"] == 1  # /work/app → deepest folder wins
    assert by_id["p_outer"]["sessionCount"] == 1  # /work/other → only the outer project


def test_junk_root_never_becomes_an_auto_project():
    # A session whose git root is HERMES_HOME (config/state) must not spawn a
    # phantom project; it lands in the Home bucket. A real repo alongside it
    # still groups normally.
    resolve = _resolver(
        {
            "/home/me/.hermes": ("/home/me/.hermes", "/home/me/.hermes"),
            "/www/app": ("/www/app", "/www/app"),
        }
    )
    junk = _session("/home/me/.hermes", branch="main")
    real = _session("/www/app", branch="main")
    is_junk = lambda root: root == "/home/me/.hermes"

    tree = pt.build_tree([], [junk, real], [], resolve, hydrate=True, is_junk_root=is_junk)

    assert _real_project_ids(tree) == ["/www/app"]
    assert _home_session_ids(tree) == [junk["id"]]
    assert real["id"] in tree["scoped_session_ids"]


def test_junk_root_is_dropped_from_the_discovered_tier():
    discovered = [{"root": "/home/me/.hermes", "label": ".hermes", "sessions": 0, "last_active": 9}]

    tree = pt.build_tree([], [], discovered, resolve=None, is_junk_root=lambda r: r == "/home/me/.hermes")

    assert tree["projects"] == []


def test_non_git_cwd_can_group_inside_a_junk_repo_subtree():
    # Repo discovery rejects the full state subtree, but a selected non-git
    # descendant may be an intentional workspace carried over from the old UI.
    workspace = _session("/home/test/.hermes/workspaces/notes")

    tree = pt.build_tree(
        [],
        [workspace],
        [],
        resolve=lambda _cwd: None,
        hydrate=True,
        is_junk_root=lambda path: path.startswith("/home/test/.hermes"),
        is_junk_cwd=lambda path: path in {"/home/test", "/home/test/.hermes"},
    )

    assert [p["id"] for p in tree["projects"]] == ["/home/test/.hermes/workspaces/notes"]
    assert tree["scoped_session_ids"] == [workspace["id"]]


def test_broad_default_non_git_cwd_stays_unscoped():
    detached = _session("/home/test/.hermes")

    tree = pt.build_tree(
        [],
        [detached],
        [],
        resolve=lambda _cwd: None,
        hydrate=True,
        is_junk_cwd=lambda path: path in {"/home/test", "/home/test/.hermes"},
    )

    assert _real_project_ids(tree) == []
    assert _home_session_ids(tree) == [detached["id"]]


def test_deleted_sibling_worktree_folds_into_parent_home_checkout():
    # A deleted <repo>-<suffix> worktree leaves its session with an unresolvable
    # cwd and no persisted root. It joins the parent's trunk lane — no dead-path
    # lane, no phantom project.
    resolve = _resolver({"/www/hermes-agent": ("/www/hermes-agent", "/www/hermes-agent")})
    sessions = [
        _session("/www/hermes-agent", branch="main"),
        _session("/www/hermes-agent-session-links"),
    ]

    tree = pt.build_tree([], sessions, [], resolve, hydrate=True)
    project = tree["projects"][0]

    assert [p["id"] for p in tree["projects"]] == ["/www/hermes-agent"]
    assert _lane_ids(project) == ["/www/hermes-agent::branch::main"]
    main = project["repos"][0]["groups"][0]
    assert main["isMain"] and main["path"] == "/www/hermes-agent"
    assert len(main["sessions"]) == 2


def test_deleted_sibling_worktree_subdir_folds_into_parent_home_checkout():
    # Same as above, but the session's cwd is a SUBDIR of the deleted worktree
    # (an agent that cd-ed into `<repo>-<suffix>/apps/desktop`). The leaf name
    # ("desktop") shares nothing with the repo, so the sibling probe has to walk
    # the ancestors — otherwise the dead path is minted as its own project.
    resolve = _resolver({"/www/hermes-agent": ("/www/hermes-agent", "/www/hermes-agent")})
    sessions = [
        _session("/www/hermes-agent", branch="main"),
        _session("/www/hermes-agent-guiperf/apps/desktop"),
    ]

    tree = pt.build_tree([], sessions, [], resolve, hydrate=True)

    assert [p["id"] for p in tree["projects"]] == ["/www/hermes-agent"]
    project = tree["projects"][0]
    assert _lane_ids(project) == ["/www/hermes-agent::branch::main"]
    assert len(project["repos"][0]["groups"][0]["sessions"]) == 2


def test_deleted_unrelated_workspace_does_not_become_a_project():
    # A deleted dir the sibling probe can't reach by name (`hermes-salvage-drafts`
    # shares no prefix with `hermes-agent`; `/tmp/scratch` was never a worktree)
    # must not be promoted to a phantom project — it can never be opened and can
    # only be dismissed by hand. Those sessions land in the Home bucket.
    resolve = _resolver({"/www/hermes-agent": ("/www/hermes-agent", "/www/hermes-agent")})
    live, salvage, scratch = (
        _session("/www/hermes-agent", branch="main"),
        _session("/www/hermes-salvage-drafts/apps/desktop"),
        _session("/tmp/scratch"),
    )
    on_disk = {"/www/hermes-agent"}

    tree = pt.build_tree([], [live, salvage, scratch], [], resolve, hydrate=True, exists=lambda p: p in on_disk)

    assert _real_project_ids(tree) == ["/www/hermes-agent"]
    assert set(_home_session_ids(tree)) == {salvage["id"], scratch["id"]}


def test_existing_non_git_workspace_still_becomes_a_project():
    # The existence guard keys on the DIRECTORY, not on git-ness: a plain folder
    # that's still on disk is a legitimate workspace and must keep its project.
    sessions = [_session("/www/notes")]

    tree = pt.build_tree([], sessions, [], lambda _cwd: None, hydrate=True, exists=lambda _p: True)

    assert [p["id"] for p in tree["projects"]] == ["/www/notes"]


def test_stale_persisted_repo_root_does_not_become_a_project():
    # A session carrying a git_repo_root whose repo has since been deleted must
    # not resurrect it as a project on the strength of the persisted value alone.
    stale = _session("/tmp/gone/sub", repo_root="/tmp/gone")

    tree = pt.build_tree([], [stale], [], lambda _cwd: None, hydrate=True, exists=lambda _p: False)

    assert _real_project_ids(tree) == []
    assert _home_session_ids(tree) == [stale["id"]]


def test_exists_defaults_to_keeping_everything():
    # Omitting `exists` (remote backends, which can't stat) preserves the old
    # behavior: guessing "gone" would wrongly hide a project on the other host.
    sessions = [_session("/remote/workspace")]

    tree = pt.build_tree([], sessions, [], lambda _cwd: None, hydrate=True)

    assert [p["id"] for p in tree["projects"]] == ["/remote/workspace"]


def test_sibling_probe_is_bounded():
    # Each miss costs a git invocation, and this runs per session, so a deeply
    # nested unresolvable cwd must not fan out into an unbounded probe storm.
    probed = []

    def resolve(cwd):
        probed.append(cwd)
        return None

    assert pt._probe_sibling_worktree("/a-b-c/d-e-f/g-h-i/j-k-l", resolve) == ""
    assert len(probed) <= pt._MAX_SIBLING_PROBES


def test_home_bucket_leads_the_tree_and_is_lossless():
    # Every session a project didn't claim belongs to Home, and Home leads the
    # list — so the grouped view shows the same set of sessions as flat Recents.
    resolve = _resolver({"/www/app": ("/www/app", "/www/app"), "/home/me": ("/home/me", "/home/me")})
    owned = _session("/www/app", branch="main")
    cwdless = _session(None)
    junked = _session("/home/me", branch="main")

    tree = pt.build_tree(
        [],
        [owned, cwdless, junked],
        [],
        resolve,
        hydrate=True,
        is_junk_root=lambda root: root == "/home/me",
    )

    assert tree["projects"][0]["id"] == pt.NO_PROJECT_ID
    assert set(_home_session_ids(tree)) == {cwdless["id"], junked["id"]}
    assert {s["id"] for p in tree["projects"] for s in _sessions_of(p)} == {
        owned["id"],
        cwdless["id"],
        junked["id"],
    }


def test_home_bucket_is_absent_when_every_session_is_placed():
    # No leftovers, no Home row — a fully-organized sidebar shows only projects.
    resolve = _resolver({"/www/app": ("/www/app", "/www/app")})

    tree = pt.build_tree([], [_session("/www/app", branch="main")], [], resolve, hydrate=True)

    assert _home(tree) is None


def test_home_bucket_carries_previews_and_drops_rows_in_overview_mode():
    sessions = [_session(None, started_at=t, last_active=t) for t in (10, 30, 20, 40)]

    tree = pt.build_tree([], sessions, [], resolve=None, preview_limit=3, hydrate=False)
    home = _home(tree)

    assert home["sessionCount"] == 4
    assert home["lastActive"] == 40
    assert [s["last_active"] for s in home["previewSessions"]] == [40, 30, 20]
    assert _sessions_of(home) == []
    assert home["isNoProject"] is True
    assert home["path"] is None


def test_colliding_repo_basenames_disambiguate_labels():
    resolve = _resolver(
        {
            "/x/proj": ("/x/proj", "/x/proj"),
            "/y/proj": ("/y/proj", "/y/proj"),
        }
    )
    sessions = [_session("/x/proj", branch="main"), _session("/y/proj", branch="main")]

    tree = pt.build_tree([], sessions, [], resolve, hydrate=True)
    labels = sorted(p["label"] for p in tree["projects"])

    assert labels == ["x/proj", "y/proj"]
