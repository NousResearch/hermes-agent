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
    homeless = _session(None)  # no cwd -> belongs to no project

    tree = pt.build_tree([project], [owned, auto, homeless], [], resolve, hydrate=True)

    assert set(tree["scoped_session_ids"]) == {owned["id"], auto["id"]}
    assert homeless["id"] not in tree["scoped_session_ids"]


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
    # phantom project; it falls through to flat Recents (unscoped). A real repo
    # alongside it still groups normally.
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

    ids = {p["id"] for p in tree["projects"]}
    assert ids == {"/www/app"}
    assert junk["id"] not in tree["scoped_session_ids"]
    assert real["id"] in tree["scoped_session_ids"]


def test_junk_root_is_dropped_from_the_discovered_tier():
    discovered = [{"root": "/home/me/.hermes", "label": ".hermes", "sessions": 0, "last_active": 9}]

    tree = pt.build_tree([], [], discovered, resolve=None, is_junk_root=lambda r: r == "/home/me/.hermes")

    assert tree["projects"] == []


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
