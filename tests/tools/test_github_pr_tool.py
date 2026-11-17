"""Unit tests for the github_pr toolsets (github_pr_read / github_pr_write).

The network seam (_github_request), the config reader (_raw_config) and the
secret lookup (get_secret) are faked. These tests pin the per-repo action
allowlist, the "refused before any request" guarantee, the default-branch and
.github/ guards, the merge refusal, and the request shapes.
"""
import json

import pytest

import tools.github_pr_tool as gh
from agent.secret_scope import UnscopedSecretError

SKILLS = "everafterai/product-skills"
MONO = "everafterai/ea-monorepo"

BASE_CFG = {
    "github_pr": {
        "repos": {
            SKILLS: {
                "actions": ["read_pr", "read_commits", "create_branch",
                            "commit_files", "open_pr"],
                "branch_prefix": "educator/",
            },
            MONO: {"actions": ["read_pr", "read_commits"]},
        }
    }
}


def _cfg(monkeypatch, cfg):
    monkeypatch.setattr(gh, "_raw_config", lambda: cfg)


def _token(monkeypatch, value="tok"):
    monkeypatch.setattr(gh, "get_secret", lambda name, default=None: value)


class FakeGitHub:
    """Records calls; answers from a {(method, path): (status, body)} map."""

    def __init__(self, routes=None):
        self.calls = []
        self.routes = routes or {}

    async def __call__(self, method, path, token, *, params=None,
                       json_body=None, accept=None, max_bytes=None):
        self.calls.append({"method": method, "path": path, "token": token,
                           "params": params, "json": json_body, "accept": accept,
                           "max_bytes": max_bytes})
        route = self.routes.get((method, path), (200, {}))
        if callable(route):
            route = route(params or {})
        status, body = route[0], route[1]
        truncated = route[2] if len(route) > 2 else False
        text = body if isinstance(body, str) else json.dumps(body)
        return {"status": status, "text": text, "truncated": truncated}


@pytest.fixture
def fake(monkeypatch):
    f = FakeGitHub()
    monkeypatch.setattr(gh, "_github_request", f)
    return f


@pytest.fixture
def configured(monkeypatch):
    _cfg(monkeypatch, BASE_CFG)
    _token(monkeypatch)


def _read(args):
    from model_tools import _run_async
    return json.loads(_run_async(gh._github_pr_read_handler(args)))


def _write(args):
    from model_tools import _run_async
    return json.loads(_run_async(gh._github_pr_write_handler(args)))


# ── Task 1: config + guards ──────────────────────────────────────────────

def test_unconfigured_repo_refused(monkeypatch):
    _cfg(monkeypatch, BASE_CFG)
    policy, err = gh._authorize("acme/x", "get_pr", gh.READ_OPS)
    assert policy is None and "not configured" in err


def test_action_not_in_allowlist_refused(monkeypatch):
    _cfg(monkeypatch, {"github_pr": {"repos": {MONO: {"actions": ["read_pr"]}}}})
    policy, err = gh._authorize(MONO, "commits", gh.READ_OPS)
    assert policy is None and "read_commits" in err


def test_repo_key_case_insensitive(monkeypatch):
    _cfg(monkeypatch, {"github_pr": {"repos": {
        "EverAfterAI/Product-Skills": {"actions": ["read_pr"]}}}})
    policy, err = gh._authorize(SKILLS, "get_pr", gh.READ_OPS)
    assert err is None and "read_pr" in policy["actions"]


def test_no_config_block_refused(monkeypatch):
    _cfg(monkeypatch, {})
    policy, err = gh._authorize(SKILLS, "get_pr", gh.READ_OPS)
    assert policy is None and "not configured" in err


def test_unknown_action_is_config_error(monkeypatch):
    _cfg(monkeypatch, {"github_pr": {"repos": {
        SKILLS: {"actions": ["read_pr", "admin"]}}}})
    policy, err = gh._authorize(SKILLS, "get_pr", gh.READ_OPS)
    assert policy is None and "admin" in err


@pytest.mark.parametrize("action", ["merge_pr", "merge", "merge_skill_pr"])
def test_merge_action_is_config_error(monkeypatch, action):
    _cfg(monkeypatch, {"github_pr": {"repos": {
        SKILLS: {"actions": ["read_pr", action]}}}})
    # A bad entry poisons the whole block — every repo, every op is refused.
    for repo in (SKILLS, MONO):
        policy, err = gh._authorize(repo, "get_pr", gh.READ_OPS)
        assert policy is None
        assert "merge" in err and "not built" in err


@pytest.mark.parametrize("level,ok", [("off", True), (False, True), ("signoff", False),
                                      ("veto", False), ("auto", False)])
def test_merge_skill_pr_autonomy_above_off_is_config_error(monkeypatch, level, ok):
    cfg = dict(BASE_CFG)
    cfg["autonomy"] = {"actions": {"merge_skill_pr": level}}
    _cfg(monkeypatch, cfg)
    policy, err = gh._authorize(SKILLS, "get_pr", gh.READ_OPS)
    if ok:
        assert err is None
    else:
        assert policy is None and "merge_skill_pr" in err


def test_malformed_repos_block_is_config_error(monkeypatch):
    _cfg(monkeypatch, {"github_pr": {"repos": ["everafterai/product-skills"]}})
    policy, err = gh._authorize(SKILLS, "get_pr", gh.READ_OPS)
    assert policy is None and err


@pytest.mark.parametrize("name", ["educator/fix-voice", "feature_1", "a.b/c-d"])
def test_branch_validation_accepts(name):
    assert gh._valid_branch(name)


@pytest.mark.parametrize("name", ["", "../x", "a/../b", "a//b", "-x", "x.lock",
                                  "refs/heads/x", "a b", "x/", "/x", "a..b",
                                  "a~1", "a^", "a:b", "HEAD", "@", "a\\b"])
def test_branch_validation_rejects(name):
    assert not gh._valid_branch(name)


def test_file_path_validation_accepts():
    assert gh._valid_file_path("plugins/release-comms/skills/x/SKILL.md")


@pytest.mark.parametrize("path", ["", "/abs", "a/../b", "..", "./a", "a//b",
                                  ".github/workflows/x.yml", ".GitHub/x",
                                  ".github", "a/.git/config", ".git/HEAD"])
def test_file_path_validation_rejects(path):
    assert not gh._valid_file_path(path)


# ── Task 2: read tool ────────────────────────────────────────────────────

def test_read_refused_before_any_request(configured, fake):
    out = _read({"op": "get_pr", "repo": "acme/other", "number": 1})
    assert "error" in out and fake.calls == []


def test_read_action_not_granted_refused_before_request(monkeypatch, fake):
    _cfg(monkeypatch, {"github_pr": {"repos": {MONO: {"actions": ["read_pr"]}}}})
    _token(monkeypatch)
    out = _read({"op": "commits", "repo": MONO})
    assert "error" in out and fake.calls == []


def test_missing_token_not_configured(monkeypatch, fake):
    _cfg(monkeypatch, BASE_CFG)
    _token(monkeypatch, None)
    out = _read({"op": "get_pr", "repo": MONO, "number": 1})
    assert "GITHUB_PR_TOKEN" in out["error"] and fake.calls == []


def test_unscoped_secret_not_configured(monkeypatch, fake):
    _cfg(monkeypatch, BASE_CFG)

    def boom(name, default=None):
        raise UnscopedSecretError("no scope")

    monkeypatch.setattr(gh, "get_secret", boom)
    out = _read({"op": "get_pr", "repo": MONO, "number": 1})
    assert "GITHUB_PR_TOKEN" in out["error"] and fake.calls == []


def test_token_read_via_get_secret_name(monkeypatch, fake):
    _cfg(monkeypatch, BASE_CFG)
    seen = []
    monkeypatch.setattr(gh, "get_secret",
                        lambda name, default=None: seen.append(name) or "tok")
    _read({"op": "get_pr", "repo": MONO, "number": 1})
    assert seen and set(seen) == {"GITHUB_PR_TOKEN"}
    assert fake.calls[0]["token"] == "tok"


@pytest.mark.parametrize("number", [0, -3, "x", None, 1.5, True])
def test_bad_number_refused(configured, fake, number):
    out = _read({"op": "get_pr", "repo": MONO, "number": number})
    assert "error" in out and fake.calls == []


def test_unknown_read_op_refused(configured, fake):
    out = _read({"op": "delete_repo", "repo": MONO})
    assert "error" in out and fake.calls == []


def test_get_pr_compacts(configured, fake):
    fake.routes[("GET", f"repos/{MONO}/pulls/7")] = (200, {
        "number": 7, "title": "Add X", "body": "Spec here", "state": "closed",
        "draft": False, "merged": True, "merged_at": "2026-11-01T00:00:00Z",
        "user": {"login": "dev"}, "head": {"ref": "feat/x", "sha": "abc"},
        "base": {"ref": "main"}, "html_url": "https://github.com/x/7",
        "changed_files": 3, "additions": 10, "deletions": 2,
        "labels": [{"name": "release"}], "_links": {"junk": 1},
    })
    out = _read({"op": "get_pr", "repo": MONO, "number": 7})
    pr = out["data"]
    assert pr["number"] == 7 and pr["body"] == "Spec here"
    assert pr["merged"] is True and pr["head"] == "feat/x" and pr["base"] == "main"
    assert pr["author"] == "dev" and pr["labels"] == ["release"]
    assert "_links" not in pr
    assert fake.calls[0]["method"] == "GET"


def test_list_prs_params(configured, fake):
    fake.routes[("GET", f"repos/{SKILLS}/pulls")] = (200, [
        {"number": 2, "title": "t", "state": "open", "user": {"login": "u"},
         "head": {"ref": "h"}, "base": {"ref": "main"}, "html_url": "x"}])
    out = _read({"op": "list_prs", "repo": SKILLS, "state": "all", "per_page": 500})
    assert out["data"][0]["number"] == 2
    assert fake.calls[0]["params"]["state"] == "all"
    assert fake.calls[0]["params"]["per_page"] == 100


def test_pr_files_truncates_patches(configured, fake):
    fake.routes[("GET", f"repos/{MONO}/pulls/7/files")] = (200, [
        {"filename": "a.py", "status": "modified", "additions": 1,
         "deletions": 0, "changes": 1, "patch": "x" * 50000}])
    out = _read({"op": "pr_files", "repo": MONO, "number": 7})
    f = out["data"][0]
    assert f["filename"] == "a.py"
    assert len(f["patch"]) <= gh._MAX_PATCH_CHARS + 100
    assert f["patch_truncated"] is True


def test_pr_diff_uses_diff_accept_and_truncates(configured, fake):
    fake.routes[("GET", f"repos/{MONO}/pulls/7")] = (200, "d" * 200_000)
    out = _read({"op": "pr_diff", "repo": MONO, "number": 7})
    assert fake.calls[0]["accept"] == "application/vnd.github.diff"
    assert out["truncated"] is True
    assert len(out["diff"]) <= gh._MAX_DIFF_CHARS


def test_pr_diff_small_not_truncated(configured, fake):
    fake.routes[("GET", f"repos/{MONO}/pulls/7")] = (200, "diff --git a b")
    out = _read({"op": "pr_diff", "repo": MONO, "number": 7})
    assert out["truncated"] is False and out["diff"] == "diff --git a b"


def test_pr_comments_merges_three_sources(configured, fake):
    fake.routes[("GET", f"repos/{SKILLS}/issues/3/comments")] = (200, [
        {"user": {"login": "pm"}, "body": "looks good", "created_at": "t1"}])
    fake.routes[("GET", f"repos/{SKILLS}/pulls/3/comments")] = (200, [
        {"user": {"login": "pm"}, "body": "nit", "created_at": "t2",
         "path": "SKILL.md", "line": 4}])
    fake.routes[("GET", f"repos/{SKILLS}/pulls/3/reviews")] = (200, [
        {"user": {"login": "lead"}, "body": "", "state": "CHANGES_REQUESTED",
         "submitted_at": "t3"}])
    out = _read({"op": "pr_comments", "repo": SKILLS, "number": 3})
    d = out["data"]
    assert d["comments"][0]["body"] == "looks good"
    assert d["review_comments"][0]["path"] == "SKILL.md"
    assert d["review_comments"][0]["line"] == 4
    assert d["reviews"][0]["state"] == "CHANGES_REQUESTED"


def test_pr_status_reads_head_sha_checks(configured, fake):
    fake.routes[("GET", f"repos/{SKILLS}/pulls/3")] = (200, {
        "number": 3, "state": "open", "merged": False, "mergeable": True,
        "draft": False, "head": {"ref": "educator/x", "sha": "deadbeef"},
        "base": {"ref": "main"}, "html_url": "u"})
    fake.routes[("GET", f"repos/{SKILLS}/commits/deadbeef/status")] = (200, {
        "state": "success", "statuses": [{"context": "ci", "state": "success"}]})
    fake.routes[("GET", f"repos/{SKILLS}/commits/deadbeef/check-runs")] = (200, {
        "check_runs": [{"name": "lint", "status": "completed",
                        "conclusion": "success"}]})
    out = _read({"op": "pr_status", "repo": SKILLS, "number": 3})
    d = out["data"]
    assert d["state"] == "open" and d["merged"] is False
    assert d["combined_status"] == "success"
    assert d["check_runs"][0]["name"] == "lint"
    paths = [c["path"] for c in fake.calls]
    assert f"repos/{SKILLS}/commits/deadbeef/status" in paths
    assert f"repos/{SKILLS}/commits/deadbeef/check-runs" in paths


def test_commits_passes_filters(configured, fake):
    fake.routes[("GET", f"repos/{MONO}/commits")] = (200, [
        {"sha": "a1", "commit": {"message": "fix: x\n\nbody",
                                 "author": {"name": "Dev", "date": "d"}},
         "author": {"login": "dev"}, "html_url": "u"}])
    out = _read({"op": "commits", "repo": MONO, "sha": "main", "path": "apps/web",
                 "since": "2026-11-01T00:00:00Z", "per_page": 1000})
    p = fake.calls[0]["params"]
    assert p["sha"] == "main" and p["path"] == "apps/web"
    assert p["since"] == "2026-11-01T00:00:00Z" and p["per_page"] == 100
    assert out["data"][0]["sha"] == "a1"
    assert out["data"][0]["message"].startswith("fix: x")


def test_http_error_shaped(configured, fake):
    fake.routes[("GET", f"repos/{MONO}/pulls/9")] = (404, {"message": "Not Found"})
    out = _read({"op": "get_pr", "repo": MONO, "number": 9})
    assert "404" in out["error"]


def test_check_fn_read(monkeypatch):
    _cfg(monkeypatch, BASE_CFG)
    _token(monkeypatch, None)
    assert gh._check_github_pr_read() is False
    _token(monkeypatch, "tok")
    assert gh._check_github_pr_read() is True
    _cfg(monkeypatch, {"github_pr": {"repos": {SKILLS: {
        "actions": ["open_pr"], "branch_prefix": "educator/"}}}})
    assert gh._check_github_pr_read() is False
    _cfg(monkeypatch, BASE_CFG)

    def boom(name, default=None):
        raise UnscopedSecretError("no scope")

    monkeypatch.setattr(gh, "get_secret", boom)
    assert gh._check_github_pr_read() is False


def test_registered_read_toolset():
    import toolsets
    from tools.registry import registry
    assert registry.get_toolset_for_tool("github_pr_read") == "github_pr"
    assert toolsets.TOOLSETS["github_pr"]["tools"] == ["github_pr_read"]


# ── Task 3: write tool ───────────────────────────────────────────────────

def _repo_route(fake, repo=SKILLS, default="main", status=200):
    fake.routes[("GET", f"repos/{repo}")] = (status, {"default_branch": default})


def test_write_refused_on_read_only_repo(configured, fake):
    out = _write({"op": "open_pr", "repo": MONO, "head": "x", "title": "t"})
    assert "error" in out and fake.calls == []


def test_write_unconfigured_repo_refused(configured, fake):
    out = _write({"op": "create_branch", "repo": "acme/x", "branch": "b"})
    assert "error" in out and fake.calls == []


@pytest.mark.parametrize("op", ["merge", "merge_pr", "update_settings",
                                "delete_branch", "push"])
def test_unknown_write_op_refused(configured, fake, op):
    out = _write({"op": op, "repo": SKILLS})
    assert "error" in out and fake.calls == []


def test_merge_op_message_says_not_available(configured, fake):
    out = _write({"op": "merge", "repo": SKILLS, "number": 1})
    assert "merg" in out["error"].lower() and "human" in out["error"].lower()


@pytest.mark.parametrize("branch", ["educator/main", "educator/Main", "educator/MAIN"])
def test_commit_to_default_branch_refused(configured, fake, branch):
    _repo_route(fake, default="educator/main")
    out = _write({"op": "commit_files", "repo": SKILLS, "branch": branch,
                  "message": "m", "files": [{"path": "a.md", "content": "x"}]})
    assert "default branch" in out["error"]
    assert [c["path"] for c in fake.calls] == [f"repos/{SKILLS}"]


def test_create_default_branch_refused(configured, fake):
    _repo_route(fake, default="educator/trunk")
    out = _write({"op": "create_branch", "repo": SKILLS, "branch": "educator/trunk"})
    assert "default branch" in out["error"]
    assert all(c["method"] == "GET" for c in fake.calls)


def test_open_pr_from_default_branch_refused(configured, fake):
    _repo_route(fake, default="educator/main")
    out = _write({"op": "open_pr", "repo": SKILLS, "head": "educator/main", "title": "t"})
    assert "default branch" in out["error"]
    assert all(c["method"] == "GET" for c in fake.calls)


def test_default_branch_lookup_failure_refuses(configured, fake):
    _repo_route(fake, status=500)
    out = _write({"op": "commit_files", "repo": SKILLS, "branch": "educator/x",
                  "message": "m", "files": [{"path": "a.md", "content": "x"}]})
    assert "error" in out
    assert all(c["method"] == "GET" for c in fake.calls)


def test_branch_prefix_enforced(monkeypatch, fake):
    _cfg(monkeypatch, {"github_pr": {"repos": {SKILLS: {
        "actions": ["create_branch", "commit_files", "open_pr"],
        "branch_prefix": "educator/"}}}})
    _token(monkeypatch)
    for args in (
        {"op": "create_branch", "repo": SKILLS, "branch": "feature/x"},
        {"op": "commit_files", "repo": SKILLS, "branch": "feature/x",
         "message": "m", "files": [{"path": "a.md", "content": "x"}]},
        {"op": "open_pr", "repo": SKILLS, "head": "feature/x", "title": "t"},
    ):
        out = _write(args)
        assert "educator/" in out["error"]
    assert fake.calls == []


def test_invalid_branch_refused_before_request(configured, fake):
    out = _write({"op": "create_branch", "repo": SKILLS, "branch": "../main"})
    assert "error" in out and fake.calls == []


def test_create_branch_flow(configured, fake):
    _repo_route(fake)
    fake.routes[("GET", f"repos/{SKILLS}/git/ref/heads/main")] = (
        200, {"object": {"sha": "base123"}})
    fake.routes[("POST", f"repos/{SKILLS}/git/refs")] = (
        201, {"ref": "refs/heads/educator/x", "object": {"sha": "base123"}})
    out = _write({"op": "create_branch", "repo": SKILLS, "branch": "educator/x"})
    assert out["ok"] is True and out["data"]["branch"] == "educator/x"
    post = fake.calls[-1]
    assert post["method"] == "POST"
    assert post["json"] == {"ref": "refs/heads/educator/x", "sha": "base123"}


def test_create_branch_from_named_branch(configured, fake):
    _repo_route(fake)
    fake.routes[("GET", f"repos/{SKILLS}/git/ref/heads/educator/base")] = (
        200, {"object": {"sha": "b2"}})
    _write({"op": "create_branch", "repo": SKILLS, "branch": "educator/x",
            "from_branch": "educator/base"})
    assert fake.calls[-1]["json"]["sha"] == "b2"


def test_commit_files_flow(configured, fake):
    _repo_route(fake)
    fake.routes[("GET", f"repos/{SKILLS}/git/ref/heads/educator/x")] = (
        200, {"object": {"sha": "head1"}})
    fake.routes[("GET", f"repos/{SKILLS}/git/commits/head1")] = (
        200, {"tree": {"sha": "tree1"}})
    fake.routes[("GET", f"repos/{SKILLS}/git/trees/tree1")] = (200, {
        "truncated": False,
        "tree": [{"path": "skills/a/SKILL.md", "mode": "100644", "type": "blob"},
                 {"path": "skills/old.md", "mode": "100644", "type": "blob"}]})
    fake.routes[("POST", f"repos/{SKILLS}/git/trees")] = (201, {"sha": "tree2"})
    fake.routes[("POST", f"repos/{SKILLS}/git/commits")] = (
        201, {"sha": "commit2", "html_url": "https://github.com/c/commit2"})
    fake.routes[("PATCH", f"repos/{SKILLS}/git/refs/heads/educator/x")] = (
        200, {"object": {"sha": "commit2"}})
    out = _write({"op": "commit_files", "repo": SKILLS, "branch": "educator/x",
                  "message": "docs: tighten voice",
                  "files": [{"path": "skills/a/SKILL.md", "content": "new"},
                            {"path": "skills/old.md", "delete": True}]})
    assert out["ok"] is True and out["data"]["commit_sha"] == "commit2"
    seq = [(c["method"], c["path"]) for c in fake.calls]
    assert seq == [
        ("GET", f"repos/{SKILLS}"),
        ("GET", f"repos/{SKILLS}/git/ref/heads/educator/x"),
        ("GET", f"repos/{SKILLS}/git/commits/head1"),
        ("GET", f"repos/{SKILLS}/git/trees/tree1"),
        ("POST", f"repos/{SKILLS}/git/trees"),
        ("POST", f"repos/{SKILLS}/git/commits"),
        ("PATCH", f"repos/{SKILLS}/git/refs/heads/educator/x"),
    ]
    assert fake.calls[3]["params"] == {"recursive": "1"}
    tree = fake.calls[4]["json"]
    assert tree["base_tree"] == "tree1"
    assert {"path": "skills/a/SKILL.md", "mode": "100644", "type": "blob",
            "content": "new"} in tree["tree"]
    assert {"path": "skills/old.md", "mode": "100644", "type": "blob",
            "sha": None} in tree["tree"]
    assert fake.calls[5]["json"] == {"message": "docs: tighten voice",
                                     "tree": "tree2", "parents": ["head1"]}
    assert fake.calls[6]["json"] == {"sha": "commit2", "force": False}


@pytest.mark.parametrize("files", [
    [],
    [{"path": ".github/workflows/x.yml", "content": "on: push"}],
    [{"path": "../x", "content": "x"}],
    [{"path": "a.md"}],
    [{"path": "a.md", "content": 3}],
    [{"path": "a.md", "content": "x", "delete": True}],
    [{"path": f"f{i}.md", "content": "x"} for i in range(51)],
    [{"path": "big.md", "content": "x" * (1_000_001)}],
    [{"path": "a.md", "content": "x"}, {"path": "a.md", "content": "y"}],
    "not a list",
])
def test_commit_rejects_bad_files(configured, fake, files):
    out = _write({"op": "commit_files", "repo": SKILLS, "branch": "educator/x",
                  "message": "m", "files": files})
    assert "error" in out and fake.calls == []


def test_commit_unresolvable_branch_head_refused(configured, fake):
    _repo_route(fake)
    fake.routes[("GET", f"repos/{SKILLS}/git/ref/heads/educator/x")] = (200, {})
    out = _write({"op": "commit_files", "repo": SKILLS, "branch": "educator/x",
                  "message": "m", "files": [{"path": "a.md", "content": "x"}]})
    assert "error" in out
    assert all(c["method"] == "GET" for c in fake.calls)
    assert len(fake.calls) == 2


def test_commit_requires_message(configured, fake):
    out = _write({"op": "commit_files", "repo": SKILLS, "branch": "educator/x",
                  "message": "  ", "files": [{"path": "a.md", "content": "x"}]})
    assert "error" in out and fake.calls == []


def test_open_pr_flow(configured, fake):
    _repo_route(fake)
    fake.routes[("POST", f"repos/{SKILLS}/pulls")] = (201, {
        "number": 12, "html_url": "https://github.com/p/12", "state": "open",
        "draft": True})
    out = _write({"op": "open_pr", "repo": SKILLS, "head": "educator/x",
                  "title": "Tighten voice", "body": "Why", "draft": True})
    assert out["data"]["number"] == 12
    post = fake.calls[-1]
    assert post["json"] == {"title": "Tighten voice", "head": "educator/x",
                            "base": "main", "body": "Why", "draft": True}


def test_open_pr_requires_title(configured, fake):
    out = _write({"op": "open_pr", "repo": SKILLS, "head": "educator/x",
                  "title": ""})
    assert "error" in out and fake.calls == []


def test_write_http_error_shaped(configured, fake):
    _repo_route(fake)
    fake.routes[("POST", f"repos/{SKILLS}/pulls")] = (
        422, {"message": "Validation Failed"})
    out = _write({"op": "open_pr", "repo": SKILLS, "head": "educator/x",
                  "title": "t"})
    assert "422" in out["error"]


def test_write_missing_token(monkeypatch, fake):
    _cfg(monkeypatch, BASE_CFG)
    _token(monkeypatch, "")
    out = _write({"op": "create_branch", "repo": SKILLS, "branch": "educator/x"})
    assert "GITHUB_PR_TOKEN" in out["error"] and fake.calls == []


def test_check_fn_write(monkeypatch):
    _cfg(monkeypatch, BASE_CFG)
    _token(monkeypatch, "tok")
    assert gh._check_github_pr_write() is True
    _cfg(monkeypatch, {"github_pr": {"repos": {MONO: {"actions": ["read_pr"]}}}})
    assert gh._check_github_pr_write() is False
    _cfg(monkeypatch, BASE_CFG)
    _token(monkeypatch, None)
    assert gh._check_github_pr_write() is False


def test_registered_write_toolset():
    import toolsets
    from tools.registry import registry
    assert registry.get_toolset_for_tool("github_pr_write") == "github_pr_write"
    assert toolsets.TOOLSETS["github_pr_write"]["tools"] == ["github_pr_write"]


def test_module_never_reads_os_environ_directly():
    """Credentials only through get_secret — os.getenv would leak across
    profiles under multiplex."""
    from pathlib import Path
    src = Path(gh.__file__).read_text(encoding="utf-8")
    assert "os.getenv" not in src and "os.environ" not in src


# ── PR #6 review: prefix, base, commit integrity, transport ──────────────

@pytest.mark.parametrize("action", ["create_branch", "commit_files", "open_pr"])
def test_write_repo_without_prefix_is_config_error(monkeypatch, fake, action):
    _cfg(monkeypatch, {"github_pr": {"repos": {
        SKILLS: {"actions": ["read_pr", action]},
        MONO: {"actions": ["read_pr"]}}}})
    _token(monkeypatch)
    # The whole block is poisoned: even reads on the other repo are refused.
    for repo in (SKILLS, MONO):
        policy, err = gh._authorize(repo, "get_pr", gh.READ_OPS)
        assert policy is None and "branch_prefix" in err
    out = _write({"op": action, "repo": SKILLS, "branch": "x", "head": "x",
                  "title": "t", "message": "m",
                  "files": [{"path": "a.md", "content": "x"}]})
    assert "branch_prefix" in out["error"] and fake.calls == []


def test_read_only_repo_needs_no_prefix(monkeypatch):
    _cfg(monkeypatch, {"github_pr": {"repos": {MONO: {"actions": ["read_pr"]}}}})
    policy, err = gh._authorize(MONO, "get_pr", gh.READ_OPS)
    assert err is None


@pytest.mark.parametrize("prefix", ["educator", "../x/", "/", "a//", "-x/", " "])
def test_bad_prefix_is_config_error(monkeypatch, prefix):
    _cfg(monkeypatch, {"github_pr": {"repos": {SKILLS: {
        "actions": ["open_pr"], "branch_prefix": prefix}}}})
    policy, err = gh._authorize(SKILLS, "open_pr", gh.WRITE_OPS)
    assert policy is None and "branch_prefix" in err


@pytest.mark.parametrize("op,field", [("create_branch", "branch"),
                                      ("commit_files", "branch"),
                                      ("open_pr", "head")])
def test_branch_outside_prefix_refused(configured, fake, op, field):
    args = {"op": op, "repo": SKILLS, "title": "t", "message": "m",
            "files": [{"path": "a.md", "content": "x"}]}
    args[field] = "feature/someone-elses"
    out = _write(args)
    assert "educator/" in out["error"] and fake.calls == []


def test_open_pr_non_default_base_refused(configured, fake):
    _repo_route(fake)
    out = _write({"op": "open_pr", "repo": SKILLS, "head": "educator/x",
                  "title": "t", "base": "release/1.0"})
    assert "default branch" in out["error"]
    assert all(c["method"] == "GET" for c in fake.calls)


def test_open_pr_base_equal_to_default_is_fine(configured, fake):
    _repo_route(fake)
    fake.routes[("POST", f"repos/{SKILLS}/pulls")] = (201, {"number": 1})
    out = _write({"op": "open_pr", "repo": SKILLS, "head": "educator/x",
                  "title": "t", "base": "Main"})
    assert out["ok"] is True
    assert fake.calls[-1]["json"]["base"] == "main"


def test_write_schema_has_no_base():
    assert "base" not in gh.GITHUB_PR_WRITE_SCHEMA["parameters"]["properties"]


def test_heads_main_lands_under_refs_heads(monkeypatch, fake):
    _cfg(monkeypatch, {"github_pr": {"repos": {SKILLS: {
        "actions": ["create_branch"], "branch_prefix": "heads/"}}}})
    _token(monkeypatch)
    _repo_route(fake)
    fake.routes[("GET", f"repos/{SKILLS}/git/ref/heads/main")] = (
        200, {"object": {"sha": "s"}})
    out = _write({"op": "create_branch", "repo": SKILLS, "branch": "heads/main"})
    assert out["ok"] is True
    assert fake.calls[-1]["json"]["ref"] == "refs/heads/heads/main"


def _commit_routes(fake, *, tree_sha="tree1", new_tree="tree2", new_sha="commit2",
                   tree_entries=None, tree_truncated=False):
    _repo_route(fake)
    fake.routes[("GET", f"repos/{SKILLS}/git/ref/heads/educator/x")] = (
        200, {"object": {"sha": "head1"}})
    fake.routes[("GET", f"repos/{SKILLS}/git/commits/head1")] = (
        200, {"tree": {"sha": tree_sha}} if tree_sha else {})
    fake.routes[("GET", f"repos/{SKILLS}/git/trees/{tree_sha}")] = (
        200, {"truncated": tree_truncated, "tree": tree_entries or []})
    fake.routes[("POST", f"repos/{SKILLS}/git/trees")] = (
        201, {"sha": new_tree} if new_tree else {})
    fake.routes[("POST", f"repos/{SKILLS}/git/commits")] = (
        201, {"sha": new_sha} if new_sha else {})


_COMMIT = {"op": "commit_files", "repo": SKILLS, "branch": "educator/x",
           "message": "m", "files": [{"path": "a.md", "content": "x"}]}


def test_missing_base_tree_refused(configured, fake):
    _commit_routes(fake, tree_sha=None)
    out = _write(dict(_COMMIT))
    assert "error" in out
    assert all(c["method"] == "GET" for c in fake.calls)
    assert fake.calls[-1]["path"] == f"repos/{SKILLS}/git/commits/head1"


def test_missing_new_tree_sha_refused(configured, fake):
    _commit_routes(fake, new_tree=None)
    out = _write(dict(_COMMIT))
    assert "error" in out
    assert fake.calls[-1]["path"] == f"repos/{SKILLS}/git/trees"
    assert not any(c["path"].endswith("/git/commits") and c["method"] == "POST"
                   for c in fake.calls)


def test_missing_commit_sha_refused(configured, fake):
    _commit_routes(fake, new_sha=None)
    out = _write(dict(_COMMIT))
    assert "error" in out
    assert (fake.calls[-1]["method"], fake.calls[-1]["path"]) == (
        "POST", f"repos/{SKILLS}/git/commits")
    assert not any(c["method"] == "PATCH" for c in fake.calls)


@pytest.mark.parametrize("mode", ["100755", "120000", "160000", "040000"])
def test_non_regular_existing_entry_refused(configured, fake, mode):
    _commit_routes(fake, tree_entries=[{"path": "a.md", "mode": mode}])
    out = _write(dict(_COMMIT))
    assert mode in out["error"]
    assert not any(c["method"] in ("POST", "PATCH") for c in fake.calls)


def test_delete_of_non_regular_entry_refused(configured, fake):
    _commit_routes(fake, tree_entries=[{"path": "bin/run", "mode": "100755"}])
    args = dict(_COMMIT, files=[{"path": "bin/run", "delete": True}])
    out = _write(args)
    assert "error" in out
    assert not any(c["method"] in ("POST", "PATCH") for c in fake.calls)


def test_truncated_tree_refused(configured, fake):
    _commit_routes(fake, tree_truncated=True)
    out = _write(dict(_COMMIT))
    assert "error" in out
    assert not any(c["method"] in ("POST", "PATCH") for c in fake.calls)


def test_new_file_allowed_when_absent_from_tree(configured, fake):
    _commit_routes(fake, tree_entries=[{"path": "other.md", "mode": "100755"}])
    out = _write(dict(_COMMIT))
    assert out["ok"] is True


def test_redirect_is_an_error_and_stops_the_write(configured, fake):
    fake.routes[("GET", f"repos/{SKILLS}")] = (307, "")
    out = _write({"op": "create_branch", "repo": SKILLS, "branch": "educator/x"})
    assert "redirect" in out["error"]
    assert len(fake.calls) == 1


def test_truncated_json_response_refused(configured, fake):
    fake.routes[("GET", f"repos/{MONO}/pulls/7")] = (200, '{"number": 7', True)
    out = _read({"op": "get_pr", "repo": MONO, "number": 7})
    assert "exceeded" in out["error"]


def test_pr_diff_asks_for_bounded_read(configured, fake):
    fake.routes[("GET", f"repos/{MONO}/pulls/7")] = (200, "d" * 10, True)
    out = _read({"op": "pr_diff", "repo": MONO, "number": 7})
    assert fake.calls[0]["max_bytes"] == gh._MAX_DIFF_BYTES
    assert out["truncated"] is True


def _paged(total):
    def route(params):
        page = params.get("page", 1)
        start = (page - 1) * 100
        n = max(0, min(100, total - start))
        return (200, [{"user": {"login": "u"}, "body": f"c{start + i}"}
                      for i in range(n)])
    return route


def test_pr_comments_pages_through(configured, fake):
    fake.routes[("GET", f"repos/{SKILLS}/issues/3/comments")] = _paged(130)
    fake.routes[("GET", f"repos/{SKILLS}/pulls/3/comments")] = (200, [])
    fake.routes[("GET", f"repos/{SKILLS}/pulls/3/reviews")] = (200, [])
    out = _read({"op": "pr_comments", "repo": SKILLS, "number": 3})
    assert len(out["data"]["comments"]) == 130
    assert out["truncated"]["comments"] is False


def test_pr_comments_caps_and_flags_truncation(configured, fake):
    fake.routes[("GET", f"repos/{SKILLS}/issues/3/comments")] = _paged(10_000)
    fake.routes[("GET", f"repos/{SKILLS}/pulls/3/comments")] = (200, [])
    fake.routes[("GET", f"repos/{SKILLS}/pulls/3/reviews")] = (200, [])
    out = _read({"op": "pr_comments", "repo": SKILLS, "number": 3})
    assert len(out["data"]["comments"]) == gh._MAX_PAGED_ITEMS
    assert out["truncated"]["comments"] is True


# The real transport: redirects off, body read is bounded.

class _FakeStream:
    def __init__(self, data):
        self.data, self.pos = data, 0

    async def read(self, n=-1):
        chunk = self.data[self.pos:self.pos + min(n, 7)]  # small chunks
        self.pos += len(chunk)
        return chunk


class _FakeResp:
    def __init__(self, status, data):
        self.status, self.content = status, _FakeStream(data)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def _fake_session(seen, status, data):
    class _Session:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def request(self, method, url, **kw):
            seen.update(kw, method=method, url=url)
            return _FakeResp(status, data)

    return _Session


def test_transport_never_follows_redirects_and_bounds_reads(monkeypatch):
    from model_tools import _run_async
    seen = {}
    monkeypatch.setattr("aiohttp.ClientSession", _fake_session(seen, 200, b"x" * 50))
    resp = _run_async(gh._github_request("GET", "repos/a/b", "tok", max_bytes=20))
    assert seen["allow_redirects"] is False
    assert seen["url"] == "https://api.github.com/repos/a/b"
    assert seen["headers"]["Authorization"] == "Bearer tok"
    assert resp["truncated"] is True and resp["text"] == "x" * 20


def test_transport_small_body_not_truncated(monkeypatch):
    from model_tools import _run_async
    seen = {}
    monkeypatch.setattr("aiohttp.ClientSession", _fake_session(seen, 307, b"moved"))
    resp = _run_async(gh._github_request("POST", "repos/a/b/pulls", "tok",
                                         json_body={"x": 1}, max_bytes=20))
    assert resp == {"status": 307, "text": "moved", "truncated": False}
    assert seen["allow_redirects"] is False
