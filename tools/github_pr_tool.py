"""github_pr — a narrow, shell-free GitHub tool over the REST API.

Built for the Product Education teammate (a secondary Hermes profile), which
must improve shared skills only by opening PRs that humans merge, and must
read the product monorepo's PRs to understand what shipped — with no
``terminal`` and without the default profile's GitHub MCP (deliberately kept
away from secondary profiles). Design: "Dependencies → 4. GitHub tool" in
docs/superpowers/specs/2026-11-16-product-education-teammate-design.md.

Two tools, two toolsets, so RBAC gates them independently and read access
never implies writes (the ``jira_api`` / ``jira_api_write`` pattern):

- ``github_pr_read`` (toolset ``github_pr``): list/get PRs, files, diff,
  comments + reviews, status + check runs, and the commit log.
- ``github_pr_write`` (toolset ``github_pr_write``): create a branch, commit
  file changes to it, open a PR. Nothing else — there is no merge, no push
  to the default branch, no settings/admin endpoint.

**The boundary is the profile's per-repo action allowlist** (``github_pr.repos``
in config.yaml, read per call so an edit applies on the next call). Every call
resolves its repo and op against it and is refused before any request when
either is missing. Unknown action names — notably any "merge" — poison the
whole block (fail closed, with a message), as does ``autonomy.actions.
merge_skill_pr`` set above ``off``: merging is not built in v1.

The token comes ONLY from ``agent.secret_scope.get_secret`` — never the
process environment, which under a multiplexed gateway may hold another
profile's value. ``UnscopedSecretError`` reads as "not configured".
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from agent.secret_scope import UnscopedSecretError, get_secret
from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

_TOKEN_ENV = "GITHUB_PR_TOKEN"
_API_BASE = "https://api.github.com"
_API_VERSION = "2022-11-28"
_TIMEOUT_SECONDS = 30

_MAX_DIFF_CHARS = 60_000
_MAX_PATCH_CHARS = 4_000
_MAX_MESSAGE_CHARS = 2_000
_MAX_FILES_PER_COMMIT = 50
_MAX_COMMIT_CONTENT_BYTES = 1_000_000
_MAX_RESPONSE_BYTES = 5_000_000   # JSON bodies; a truncated JSON body is refused
_MAX_DIFF_BYTES = 2 * _MAX_DIFF_CHARS
_MAX_PAGED_ITEMS = 500            # per kind, for pr_comments

# op → the allowlist action that unlocks it.
READ_OPS: dict[str, str] = {
    "list_prs": "read_pr",
    "get_pr": "read_pr",
    "pr_files": "read_pr",
    "pr_diff": "read_pr",
    "pr_comments": "read_pr",
    "pr_status": "read_pr",
    "commits": "read_commits",
}
WRITE_OPS: dict[str, str] = {
    "create_branch": "create_branch",
    "commit_files": "commit_files",
    "open_pr": "open_pr",
}
KNOWN_ACTIONS: frozenset[str] = frozenset(READ_OPS.values()) | frozenset(WRITE_OPS.values())
_READ_ACTIONS = frozenset(READ_OPS.values())
_WRITE_ACTIONS = frozenset(WRITE_OPS.values())

_NOT_CONFIGURED = (
    f"GitHub is not configured for this profile (set {_TOKEN_ENV} — a "
    "fine-grained token — in this profile's .env)."
)
_MERGE_REFUSAL = (
    "Merging pull requests is not available: a human reviews and merges every "
    "PR (autonomy action merge_skill_pr is off and not built in v1)."
)

_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_BRANCH_RE = re.compile(
    r"^[A-Za-z0-9_][A-Za-z0-9._-]*(?:/[A-Za-z0-9_][A-Za-z0-9._-]*)*$"
)
_PATH_SEGMENT_RE = re.compile(r"^[^\x00-\x1f\\]+$")


# ── config ────────────────────────────────────────────────────────────────

def _raw_config() -> dict:
    """The active profile's config.yaml (mtime-cached, profile-aware)."""
    from hermes_cli.config import read_raw_config_readonly
    return read_raw_config_readonly() or {}


def _autonomy_error(cfg: dict) -> str | None:
    autonomy = cfg.get("autonomy")
    actions = autonomy.get("actions") if isinstance(autonomy, dict) else None
    if not isinstance(actions, dict) or "merge_skill_pr" not in actions:
        return None
    level = actions.get("merge_skill_pr")
    # YAML 1.1 parses a bare `off` as False.
    if level is False or str(level).strip().lower() == "off":
        return None
    return (
        f"github_pr config error: autonomy.actions.merge_skill_pr is {level!r}, "
        "but merging PRs is not built — only 'off' is valid in v1."
    )


def _load_repo_policy() -> tuple[dict[str, dict], str | None]:
    """Parse ``github_pr.repos`` → ``({repo_lower: {actions, branch_prefix}}, error)``.

    Any malformed or unknown entry returns an error for the WHOLE block, so a
    typo can never silently widen or narrow what the tool does.
    """
    cfg = _raw_config()
    err = _autonomy_error(cfg)
    if err:
        return {}, err
    block = cfg.get("github_pr")
    if block is None:
        return {}, None
    if not isinstance(block, dict):
        return {}, "github_pr config error: the github_pr block must be a mapping."
    repos = block.get("repos") or {}
    if not isinstance(repos, dict):
        return {}, (
            "github_pr config error: github_pr.repos must map 'owner/name' to "
            "{actions: [...]}."
        )

    out: dict[str, dict] = {}
    for key, entry in repos.items():
        where = f"github_pr.repos.{key}"
        if not isinstance(key, str) or not _REPO_RE.match(key):
            return {}, f"github_pr config error: '{key}' is not an 'owner/name' repo."
        if not isinstance(entry, dict):
            return {}, f"github_pr config error: {where} must be a mapping with 'actions'."
        actions = entry.get("actions") or []
        if isinstance(actions, str):
            actions = [a.strip() for a in actions.split(",") if a.strip()]
        if not isinstance(actions, list) or not actions:
            return {}, f"github_pr config error: {where}.actions must be a non-empty list."
        for action in actions:
            name = str(action).strip()
            if name in KNOWN_ACTIONS:
                continue
            if "merge" in name.lower():
                return {}, (
                    f"github_pr config error: action '{name}' in {where}.actions — "
                    "merging PRs is not built (merge_skill_pr is off in v1); a "
                    "human merges. Remove it."
                )
            return {}, (
                f"github_pr config error: unknown action '{name}' in "
                f"{where}.actions. Known: {', '.join(sorted(KNOWN_ACTIONS))}."
            )
        prefix = entry.get("branch_prefix") or ""
        if not isinstance(prefix, str):
            return {}, f"github_pr config error: {where}.branch_prefix must be a string."
        prefix = prefix.strip()
        granted = frozenset(str(a).strip() for a in actions)
        writes = granted & _WRITE_ACTIONS
        # A repo that grants any write must confine writes to the bot's own
        # branches. Without a prefix a prompt-injected agent could commit onto
        # a human's open feature branch (their merge then carries it into the
        # default branch unreviewed) or create release/* or gh-pages.
        if writes and not prefix:
            return {}, (
                f"github_pr config error: {where} grants write actions "
                f"({', '.join(sorted(writes))}) but has no branch_prefix. Set one "
                "(e.g. 'educator/') so writes only touch the bot's own branches."
            )
        if prefix and (not prefix.endswith("/") or not _valid_branch(prefix[:-1])):
            return {}, (
                f"github_pr config error: {where}.branch_prefix '{prefix}' must be "
                "a branch-name prefix ending in '/', e.g. 'educator/'."
            )
        out[key.lower()] = {"actions": granted, "branch_prefix": prefix}
    return out, None


def _authorize(repo: str, op: str, ops: dict[str, str]) -> tuple[dict | None, str | None]:
    """Resolve ``(repo, op)`` against the allowlist. No network."""
    action = ops.get(op)
    if action is None:
        if "merge" in (op or "").lower():
            return None, _MERGE_REFUSAL
        return None, f"Unknown op '{op}'. Allowed: {', '.join(ops)}."
    if not isinstance(repo, str) or not _REPO_RE.match(repo):
        return None, "repo must be 'owner/name', e.g. 'everafterai/product-skills'."
    repos, err = _load_repo_policy()
    if err:
        return None, err
    policy = repos.get(repo.lower())
    if policy is None:
        configured = ", ".join(sorted(repos)) or "none"
        return None, (
            f"Repository '{repo}' is not configured for github_pr (configured: "
            f"{configured})."
        )
    if action not in policy["actions"]:
        return None, (
            f"'{op}' on {repo} needs the '{action}' action, which this profile's "
            f"github_pr.repos.{repo}.actions does not grant."
        )
    return policy, None


# ── input guards ──────────────────────────────────────────────────────────

def _valid_branch(name: Any) -> bool:
    """A plain branch name: safe characters, no traversal, no ref syntax."""
    if not isinstance(name, str) or not name or len(name) > 200:
        return False
    if not _BRANCH_RE.match(name) or ".." in name:
        return False
    if name == "HEAD" or name.lower().startswith("refs/"):
        return False
    return not any(seg.endswith((".lock", ".")) for seg in name.split("/"))


def _valid_file_path(path: Any) -> bool:
    """A repo-relative file path, never under ``.github/`` or ``.git``.

    ``.github/`` is refused outright: a workflow committed to any branch runs
    on push with the repo's secrets, which would turn "open a PR" into code
    execution in CI.
    """
    if not isinstance(path, str) or not path or len(path) > 400:
        return False
    segments = path.split("/")
    for seg in segments:
        if seg in ("", ".", "..") or not _PATH_SEGMENT_RE.match(seg):
            return False
        if seg.lower() == ".git":
            return False
    return segments[0].lower() != ".github"


def _branch_error(name: Any, field: str, prefix: str) -> str | None:
    if not _valid_branch(name):
        return f"{field} '{name}' is not a valid branch name."
    if not prefix:
        # Unreachable through config (a write-granting repo without a prefix is
        # a config error) — fail closed regardless.
        return f"{field}: no github_pr branch_prefix is configured for this repo."
    if not name.startswith(prefix):
        return f"{field} must start with '{prefix}' (github_pr branch_prefix for this repo)."
    return None


def _pr_number(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if isinstance(value, int) and value > 0:
        return value
    return None


def _per_page(value: Any, default: int = 30) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(n, 100))


def _page(value: Any) -> int | None:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n > 1 else None


# ── HTTP ──────────────────────────────────────────────────────────────────

def _github_token() -> str:
    try:
        value = get_secret(_TOKEN_ENV)
    except UnscopedSecretError:
        return ""
    return (value or "").strip()


async def _github_request(
    method: str,
    path: str,
    token: str,
    *,
    params: dict | None = None,
    json_body: dict | None = None,
    accept: str | None = None,
    max_bytes: int | None = None,
) -> dict:
    """Call ``api.github.com/<path>``. Returns ``{status, text, truncated}``.

    Redirects are never followed: a 307 from a renamed or transferred repo
    would otherwise replay a write against a repo the allowlist never named.
    The body is read up to ``max_bytes`` (default ``_MAX_RESPONSE_BYTES``).
    """
    import aiohttp
    from gateway.platforms.base import proxy_kwargs_for_aiohttp, resolve_proxy_url

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": accept or "application/vnd.github+json",
        "X-GitHub-Api-Version": _API_VERSION,
        "User-Agent": "hermes-github-pr",
    }
    clean_params = {k: v for k, v in (params or {}).items() if v is not None}
    _sess_kw, _req_kw = proxy_kwargs_for_aiohttp(resolve_proxy_url())
    timeout = aiohttp.ClientTimeout(total=_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout, **_sess_kw) as session:
        async with session.request(
            method, f"{_API_BASE}/{path}", headers=headers,
            params=clean_params or None, json=json_body,
            allow_redirects=False, **_req_kw,
        ) as resp:
            limit = max_bytes or _MAX_RESPONSE_BYTES
            body, truncated = await _read_bounded(resp.content, limit)
            return {
                "status": resp.status,
                "text": body.decode("utf-8", errors="replace"),
                "truncated": truncated,
            }


async def _read_bounded(stream: Any, limit: int) -> tuple[bytes, bool]:
    """Read at most ``limit`` bytes; report whether more was available."""
    chunks: list[bytes] = []
    total = 0
    while total <= limit:
        chunk = await stream.read(limit + 1 - total)
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    body = b"".join(chunks)
    return body[:limit], len(body) > limit


def _http_error(resp: dict) -> str | None:
    status = resp["status"]
    if 300 <= status < 400:
        return (
            f"GitHub answered with a redirect (HTTP {status}), which github_pr does "
            "not follow — the repo may have been renamed or moved. Update the "
            "github_pr.repos allowlist to its current name."
        )
    if status < 400:
        return None
    text = (resp.get("text") or "").strip()
    try:
        message = json.loads(text).get("message") or text
    except (ValueError, AttributeError):
        message = text
    return f"GitHub API error (HTTP {status}): {str(message)[:300]}"


def _json_body(resp: dict) -> tuple[Any, str | None]:
    err = _http_error(resp)
    if err:
        return None, err
    if resp.get("truncated"):
        return None, (
            f"GitHub's response exceeded {_MAX_RESPONSE_BYTES} bytes and was not "
            "read; narrow the request (per_page, page, path)."
        )
    text = (resp.get("text") or "").strip()
    return (json.loads(text) if text else None), None


async def _get_json(path: str, token: str, params: dict | None = None) -> tuple[Any, str | None]:
    return _json_body(await _github_request("GET", path, token, params=params))


async def _send_json(method: str, path: str, token: str, body: dict) -> tuple[Any, str | None]:
    return _json_body(await _github_request(method, path, token, json_body=body))


async def _get_paged(path: str, token: str) -> tuple[list, bool, str | None]:
    """GET a list endpoint page by page, up to ``_MAX_PAGED_ITEMS``.

    Returns ``(items, truncated, error)``; ``truncated`` means the cap was hit
    with more pages likely remaining.
    """
    items: list = []
    page = 1
    while True:
        batch, err = await _get_json(path, token, {"per_page": 100, "page": page})
        if err:
            return [], False, err
        batch = batch or []
        items.extend(batch)
        if len(batch) < 100:
            return items, False, None
        if len(items) >= _MAX_PAGED_ITEMS:
            return items[:_MAX_PAGED_ITEMS], True, None
        page += 1


# ── read ops ──────────────────────────────────────────────────────────────

def _login(obj: Any) -> str | None:
    return obj.get("login") if isinstance(obj, dict) else None


def _ref(obj: Any) -> str | None:
    return obj.get("ref") if isinstance(obj, dict) else None


def _pr_summary(pr: dict) -> dict:
    return {
        "number": pr.get("number"),
        "title": pr.get("title"),
        "state": pr.get("state"),
        "draft": pr.get("draft"),
        "author": _login(pr.get("user")),
        "head": _ref(pr.get("head")),
        "base": _ref(pr.get("base")),
        "merged_at": pr.get("merged_at"),
        "updated_at": pr.get("updated_at"),
        "html_url": pr.get("html_url"),
    }


def _pr_full(pr: dict) -> dict:
    out = _pr_summary(pr)
    out.update({
        "body": pr.get("body"),
        "merged": pr.get("merged"),
        "mergeable": pr.get("mergeable"),
        "head_sha": (pr.get("head") or {}).get("sha"),
        "changed_files": pr.get("changed_files"),
        "additions": pr.get("additions"),
        "deletions": pr.get("deletions"),
        "commits": pr.get("commits"),
        "labels": [lb.get("name") for lb in pr.get("labels") or [] if isinstance(lb, dict)],
    })
    return out


async def _op_list_prs(repo: str, token: str, args: dict, number: int | None) -> str:
    state = str(args.get("state") or "open").lower()
    if state not in ("open", "closed", "all"):
        return tool_error("state must be one of: open, closed, all.")
    data, err = await _get_json(f"repos/{repo}/pulls", token, {
        "state": state, "per_page": _per_page(args.get("per_page")),
        "page": _page(args.get("page")), "sort": "updated", "direction": "desc",
    })
    if err:
        return tool_error(err)
    return tool_result({"ok": True, "data": [_pr_summary(p) for p in data or []]})


async def _op_get_pr(repo: str, token: str, args: dict, number: int | None) -> str:
    data, err = await _get_json(f"repos/{repo}/pulls/{number}", token)
    if err:
        return tool_error(err)
    return tool_result({"ok": True, "data": _pr_full(data or {})})


async def _op_pr_files(repo: str, token: str, args: dict, number: int | None) -> str:
    data, err = await _get_json(f"repos/{repo}/pulls/{number}/files", token, {
        "per_page": _per_page(args.get("per_page"), default=100),
        "page": _page(args.get("page")),
    })
    if err:
        return tool_error(err)
    files = []
    for f in data or []:
        patch = f.get("patch") or ""
        files.append({
            "filename": f.get("filename"),
            "previous_filename": f.get("previous_filename"),
            "status": f.get("status"),
            "additions": f.get("additions"),
            "deletions": f.get("deletions"),
            "changes": f.get("changes"),
            "patch": patch[:_MAX_PATCH_CHARS],
            "patch_truncated": len(patch) > _MAX_PATCH_CHARS,
        })
    return tool_result({"ok": True, "data": files})


async def _op_pr_diff(repo: str, token: str, args: dict, number: int | None) -> str:
    resp = await _github_request(
        "GET", f"repos/{repo}/pulls/{number}", token,
        accept="application/vnd.github.diff", max_bytes=_MAX_DIFF_BYTES,
    )
    err = _http_error(resp)
    if err:
        return tool_error(err)
    diff = resp.get("text") or ""
    return tool_result({
        "ok": True,
        "diff": diff[:_MAX_DIFF_CHARS],
        "truncated": bool(resp.get("truncated")) or len(diff) > _MAX_DIFF_CHARS,
    })


async def _op_pr_comments(repo: str, token: str, args: dict, number: int | None) -> str:
    comments, c_trunc, err = await _get_paged(f"repos/{repo}/issues/{number}/comments", token)
    if err:
        return tool_error(err)
    review_comments, rc_trunc, err = await _get_paged(
        f"repos/{repo}/pulls/{number}/comments", token)
    if err:
        return tool_error(err)
    reviews, r_trunc, err = await _get_paged(f"repos/{repo}/pulls/{number}/reviews", token)
    if err:
        return tool_error(err)
    return tool_result({"ok": True, "truncated": {
        "comments": c_trunc, "review_comments": rc_trunc, "reviews": r_trunc,
    }, "data": {
        "comments": [
            {"author": _login(c.get("user")), "body": c.get("body"),
             "created_at": c.get("created_at")}
            for c in comments or []
        ],
        "review_comments": [
            {"author": _login(c.get("user")), "body": c.get("body"),
             "path": c.get("path"), "line": c.get("line"),
             "created_at": c.get("created_at"),
             "in_reply_to_id": c.get("in_reply_to_id")}
            for c in review_comments or []
        ],
        "reviews": [
            {"author": _login(r.get("user")), "state": r.get("state"),
             "body": r.get("body"), "submitted_at": r.get("submitted_at")}
            for r in reviews or []
        ],
    }})


async def _op_pr_status(repo: str, token: str, args: dict, number: int | None) -> str:
    pr, err = await _get_json(f"repos/{repo}/pulls/{number}", token)
    if err:
        return tool_error(err)
    pr = pr or {}
    sha = (pr.get("head") or {}).get("sha")
    out: dict[str, Any] = {
        "number": pr.get("number"), "state": pr.get("state"),
        "draft": pr.get("draft"), "merged": pr.get("merged"),
        "mergeable": pr.get("mergeable"), "head": _ref(pr.get("head")),
        "head_sha": sha, "base": _ref(pr.get("base")),
        "html_url": pr.get("html_url"),
    }
    if sha:
        status, err = await _get_json(f"repos/{repo}/commits/{sha}/status", token)
        if err:
            out["combined_status_error"] = err
        else:
            status = status or {}
            out["combined_status"] = status.get("state")
            out["statuses"] = [
                {"context": s.get("context"), "state": s.get("state"),
                 "description": s.get("description")}
                for s in status.get("statuses") or []
            ]
        # check runs need the token's Checks permission; degrade, don't fail.
        checks, err = await _get_json(f"repos/{repo}/commits/{sha}/check-runs", token,
                                      {"per_page": 100})
        if err:
            out["check_runs_error"] = err
        else:
            out["check_runs"] = [
                {"name": c.get("name"), "status": c.get("status"),
                 "conclusion": c.get("conclusion")}
                for c in (checks or {}).get("check_runs") or []
            ]
    return tool_result({"ok": True, "data": out})


async def _op_commits(repo: str, token: str, args: dict, number: int | None) -> str:
    params = {"per_page": _per_page(args.get("per_page")), "page": _page(args.get("page"))}
    for key in ("sha", "path", "since", "until"):
        value = args.get(key)
        if value not in (None, ""):
            params[key] = str(value)
    data, err = await _get_json(f"repos/{repo}/commits", token, params)
    if err:
        return tool_error(err)
    commits = []
    for c in data or []:
        commit = c.get("commit") or {}
        author = commit.get("author") or {}
        commits.append({
            "sha": c.get("sha"),
            "message": (commit.get("message") or "")[:_MAX_MESSAGE_CHARS],
            "author_name": author.get("name"),
            "author_login": _login(c.get("author")),
            "date": author.get("date"),
            "html_url": c.get("html_url"),
        })
    return tool_result({"ok": True, "data": commits})


_READ_IMPL = {
    "list_prs": _op_list_prs,
    "get_pr": _op_get_pr,
    "pr_files": _op_pr_files,
    "pr_diff": _op_pr_diff,
    "pr_comments": _op_pr_comments,
    "pr_status": _op_pr_status,
    "commits": _op_commits,
}
_NUMBERED_OPS = frozenset({"get_pr", "pr_files", "pr_diff", "pr_comments", "pr_status"})


async def _github_pr_read_handler(args: dict, **_kw) -> str:
    op = str(args.get("op") or "").strip()
    repo = str(args.get("repo") or "").strip()
    _policy, err = _authorize(repo, op, READ_OPS)
    if err:
        return tool_error(err)
    number = None
    if op in _NUMBERED_OPS:
        number = _pr_number(args.get("number"))
        if number is None:
            return tool_error("number must be a positive pull-request number.")
    token = _github_token()
    if not token:
        return tool_error(_NOT_CONFIGURED)
    try:
        return await _READ_IMPL[op](repo.lower(), token, args, number)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("[github_pr_read] %s on %s failed: %s", op, repo, e)
        return tool_error(f"GitHub request failed: {e}")


# ── write ops ─────────────────────────────────────────────────────────────

def _tree_entries(files: Any) -> tuple[list[dict] | None, str | None]:
    """Validate ``files`` and build git-tree entries. No network."""
    if not isinstance(files, list) or not files:
        return None, "files must be a non-empty list of {path, content} or {path, delete: true}."
    if len(files) > _MAX_FILES_PER_COMMIT:
        return None, f"At most {_MAX_FILES_PER_COMMIT} files per commit."
    entries: list[dict] = []
    seen: set[str] = set()
    total = 0
    for f in files:
        if not isinstance(f, dict):
            return None, "each file must be an object."
        path = f.get("path")
        if not _valid_file_path(path):
            return None, (
                f"path '{path}' is not allowed (repo-relative, no '..', nothing "
                "under .github/ or .git)."
            )
        if path in seen:
            return None, f"path '{path}' appears twice."
        seen.add(path)
        delete = f.get("delete") is True
        has_content = "content" in f and f.get("content") is not None
        if delete == has_content:
            return None, f"'{path}': give exactly one of content (a string) or delete: true."
        if delete:
            entries.append({"path": path, "mode": "100644", "type": "blob", "sha": None})
            continue
        content = f.get("content")
        if not isinstance(content, str):
            return None, f"'{path}': content must be a string."
        total += len(content.encode("utf-8"))
        if total > _MAX_COMMIT_CONTENT_BYTES:
            return None, f"Commit content exceeds {_MAX_COMMIT_CONTENT_BYTES} bytes."
        entries.append({"path": path, "mode": "100644", "type": "blob", "content": content})
    return entries, None


async def _default_branch(repo: str, token: str) -> tuple[str | None, str | None]:
    data, err = await _get_json(f"repos/{repo}", token)
    if err:
        return None, f"Could not look up {repo}'s default branch, so no write was made: {err}"
    default = (data or {}).get("default_branch")
    if not default:
        return None, f"Could not look up {repo}'s default branch, so no write was made."
    return default, None


def _is_default(branch: str, default: str) -> bool:
    return branch.casefold() == default.casefold()


def _default_refusal(repo: str, default: str) -> str:
    return (
        f"Refused: '{default}' is {repo}'s default branch. github_pr never writes "
        "to the default branch — work on your own branch and open a PR."
    )


def _validate_write(op: str, args: dict, prefix: str) -> tuple[dict | None, str | None]:
    """Op-specific validation before any request; returns normalized inputs."""
    if op == "create_branch":
        branch = args.get("branch")
        err = _branch_error(branch, "branch", prefix)
        if err:
            return None, err
        source = args.get("from_branch")
        if source not in (None, "") and not _valid_branch(source):
            return None, f"from_branch '{source}' is not a valid branch name."
        return {"branch": branch, "from_branch": source or None}, None

    if op == "commit_files":
        branch = args.get("branch")
        err = _branch_error(branch, "branch", prefix)
        if err:
            return None, err
        message = args.get("message")
        if not isinstance(message, str) or not message.strip():
            return None, "message (the commit message) is required."
        entries, err = _tree_entries(args.get("files"))
        if err:
            return None, err
        return {"branch": branch, "message": message.strip(), "entries": entries}, None

    # open_pr
    head = args.get("head")
    err = _branch_error(head, "head", prefix)
    if err:
        return None, err
    title = args.get("title")
    if not isinstance(title, str) or not title.strip():
        return None, "title is required."
    body = args.get("body")
    if body is not None and not isinstance(body, str):
        return None, "body must be a string."
    # PRs always target the default branch. `base` is not in the schema; if a
    # caller passes one anyway it must name the default branch (checked live).
    base = args.get("base")
    if base not in (None, "") and not _valid_branch(base):
        return None, f"base '{base}' is not a valid branch name."
    return {"head": head, "title": title.strip(), "body": body or "",
            "base": base or None, "draft": bool(args.get("draft"))}, None


async def _write_create_branch(repo: str, token: str, v: dict, default: str) -> str:
    if _is_default(v["branch"], default):
        return tool_error(_default_refusal(repo, default))
    source = v["from_branch"] or default
    ref, err = await _get_json(f"repos/{repo}/git/ref/heads/{source}", token)
    if err:
        return tool_error(err)
    sha = ((ref or {}).get("object") or {}).get("sha")
    if not sha:
        return tool_error(f"Could not resolve the head of '{source}'.")
    _, err = await _send_json("POST", f"repos/{repo}/git/refs", token,
                              {"ref": f"refs/heads/{v['branch']}", "sha": sha})
    if err:
        return tool_error(err)
    logger.info("[github_pr_write] created branch %s on %s from %s", v["branch"], repo, source)
    return tool_result({"ok": True, "data": {"branch": v["branch"], "from": source, "sha": sha}})


async def _mode_check(repo: str, token: str, base_tree: str, entries: list[dict]) -> str | None:
    """Refuse to touch any existing entry that is not a regular file.

    Every entry is written as mode 100644, which would silently turn an
    executable, symlink or submodule into a plain file. New paths are fine.
    """
    data, err = await _get_json(f"repos/{repo}/git/trees/{base_tree}", token,
                                {"recursive": "1"})
    if err:
        return f"Could not read the branch's file modes, so no commit was made: {err}"
    data = data or {}
    if data.get("truncated"):
        return "The branch's tree is too large to verify file modes, so no commit was made."
    modes = {e.get("path"): e.get("mode") for e in data.get("tree") or [] if isinstance(e, dict)}
    for entry in entries:
        mode = modes.get(entry["path"])
        if mode is not None and mode != "100644":
            return (
                f"Refused: '{entry['path']}' is not a regular file (mode {mode}); "
                "github_pr only writes regular files."
            )
    return None


async def _write_commit_files(repo: str, token: str, v: dict, default: str) -> str:
    branch = v["branch"]
    if _is_default(branch, default):
        return tool_error(_default_refusal(repo, default))
    ref, err = await _get_json(f"repos/{repo}/git/ref/heads/{branch}", token)
    if err:
        return tool_error(f"{err} (create the branch first with op=create_branch)")
    head_sha = ((ref or {}).get("object") or {}).get("sha")
    if not head_sha:
        return tool_error(f"Could not resolve the head of '{branch}'.")
    commit, err = await _get_json(f"repos/{repo}/git/commits/{head_sha}", token)
    if err:
        return tool_error(err)
    base_tree = ((commit or {}).get("tree") or {}).get("sha")
    # Without base_tree GitHub builds a tree of ONLY the listed files, and the
    # commit would delete everything else on the branch.
    if not base_tree:
        return tool_error(f"Could not resolve the tree of '{branch}', so no commit was made.")
    err = await _mode_check(repo, token, base_tree, v["entries"])
    if err:
        return tool_error(err)
    tree, err = await _send_json("POST", f"repos/{repo}/git/trees", token,
                                 {"base_tree": base_tree, "tree": v["entries"]})
    if err:
        return tool_error(err)
    new_tree = (tree or {}).get("sha")
    if not new_tree:
        return tool_error("GitHub returned no tree SHA, so no commit was made.")
    new_commit, err = await _send_json("POST", f"repos/{repo}/git/commits", token, {
        "message": v["message"], "tree": new_tree, "parents": [head_sha],
    })
    if err:
        return tool_error(err)
    new_sha = (new_commit or {}).get("sha")
    if not new_sha:
        return tool_error(f"GitHub returned no commit SHA, so '{branch}' was not moved.")
    _, err = await _send_json("PATCH", f"repos/{repo}/git/refs/heads/{branch}", token,
                              {"sha": new_sha, "force": False})
    if err:
        return tool_error(err)
    logger.info("[github_pr_write] committed %s to %s:%s (%d files)",
                new_sha, repo, branch, len(v["entries"]))
    return tool_result({"ok": True, "data": {
        "branch": branch, "commit_sha": new_sha,
        "html_url": (new_commit or {}).get("html_url"),
        "files_changed": len(v["entries"]),
    }})


async def _write_open_pr(repo: str, token: str, v: dict, default: str) -> str:
    if _is_default(v["head"], default):
        return tool_error(_default_refusal(repo, default))
    if v["base"] and not _is_default(v["base"], default):
        return tool_error(
            f"Refused: PRs from github_pr always target {repo}'s default branch "
            f"('{default}'), not '{v['base']}'."
        )
    pr, err = await _send_json("POST", f"repos/{repo}/pulls", token, {
        "title": v["title"], "head": v["head"], "base": default,
        "body": v["body"], "draft": v["draft"],
    })
    if err:
        return tool_error(err)
    pr = pr or {}
    logger.info("[github_pr_write] opened PR #%s on %s", pr.get("number"), repo)
    return tool_result({"ok": True, "data": {
        "number": pr.get("number"), "html_url": pr.get("html_url"),
        "state": pr.get("state"), "draft": pr.get("draft"),
    }})


_WRITE_IMPL = {
    "create_branch": _write_create_branch,
    "commit_files": _write_commit_files,
    "open_pr": _write_open_pr,
}


async def _github_pr_write_handler(args: dict, **_kw) -> str:
    op = str(args.get("op") or "").strip()
    repo = str(args.get("repo") or "").strip()
    policy, err = _authorize(repo, op, WRITE_OPS)
    if err:
        return tool_error(err)
    values, err = _validate_write(op, args, policy["branch_prefix"])
    if err:
        return tool_error(err)
    token = _github_token()
    if not token:
        return tool_error(_NOT_CONFIGURED)
    repo = repo.lower()
    try:
        # Looked up live on every write: the one thing this tool must never do
        # is move the default branch, so it is never assumed.
        default, err = await _default_branch(repo, token)
        if err:
            return tool_error(err)
        return await _WRITE_IMPL[op](repo, token, values, default)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("[github_pr_write] %s on %s failed: %s", op, repo, e)
        return tool_error(f"GitHub request failed: {e}")


# ── availability ──────────────────────────────────────────────────────────

def _available(actions: frozenset[str]) -> bool:
    if not _github_token():
        return False
    repos, err = _load_repo_policy()
    if err:
        # Stay visible so the refusal explains the config error, rather than
        # the tool silently vanishing.
        return True
    return any(policy["actions"] & actions for policy in repos.values())


def _check_github_pr_read() -> bool:
    """Visible when a token resolves and some repo grants a read action."""
    return _available(_READ_ACTIONS)


def _check_github_pr_write() -> bool:
    """Visible when a token resolves and some repo grants a write action."""
    return _available(_WRITE_ACTIONS)


# ── schemas + registration ────────────────────────────────────────────────

GITHUB_PR_READ_SCHEMA = {
    "name": "github_pr_read",
    "description": (
        "Read GitHub pull requests and commit logs (READ-ONLY) on the repos this "
        "profile is configured for. ops: list_prs (state, per_page, page); "
        "get_pr (number) — title, description, branches, merge state; pr_files "
        "(number) — changed files with patches; pr_diff (number) — the unified "
        "diff, truncated when large; pr_comments (number) — conversation "
        "comments, inline review comments and reviews; pr_status (number) — "
        "state plus CI status/check runs; commits (sha = branch or SHA, path, "
        "since, until, per_page, page) — the commit log. Use this for all "
        "GitHub reads; never shell out."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "op": {"type": "string", "enum": list(READ_OPS)},
            "repo": {"type": "string", "description": "'owner/name', e.g. 'everafterai/ea-monorepo'."},
            "number": {"type": "integer", "description": "Pull-request number (PR ops)."},
            "state": {"type": "string", "enum": ["open", "closed", "all"],
                      "description": "list_prs filter (default open)."},
            "sha": {"type": "string", "description": "commits: branch name or SHA to list from."},
            "path": {"type": "string", "description": "commits: only commits touching this path."},
            "since": {"type": "string", "description": "commits: ISO 8601 lower bound."},
            "until": {"type": "string", "description": "commits: ISO 8601 upper bound."},
            "per_page": {"type": "integer", "description": "Page size, max 100."},
            "page": {"type": "integer", "description": "Page number (1-based)."},
        },
        "required": ["op", "repo"],
    },
}

GITHUB_PR_WRITE_SCHEMA = {
    "name": "github_pr_write",
    "description": (
        "Propose a change on GitHub by pull request, on the repos this profile is "
        "allowed to write. ops: create_branch (branch, from_branch? — defaults "
        "to the default branch); commit_files (branch, message, files: "
        "[{path, content} | {path, delete: true}]) — one commit of full-file "
        "contents onto an EXISTING branch you created; open_pr (head, title, "
        "body, draft?) — always targets the default branch. Branch names must "
        "start with this repo's configured branch prefix. It never "
        "writes to the default branch, never touches .github/, and cannot "
        "merge: a human reviews and merges every PR. Read PR comments and "
        "status with github_pr_read."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "op": {"type": "string", "enum": list(WRITE_OPS)},
            "repo": {"type": "string", "description": "'owner/name', e.g. 'everafterai/product-skills'."},
            "branch": {"type": "string", "description": "create_branch / commit_files: the branch."},
            "from_branch": {"type": "string", "description": "create_branch: start point (default branch if omitted)."},
            "message": {"type": "string", "description": "commit_files: the commit message."},
            "files": {
                "type": "array",
                "description": "commit_files: full new contents per path, or delete: true.",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "delete": {"type": "boolean"},
                    },
                    "required": ["path"],
                },
            },
            "head": {"type": "string", "description": "open_pr: the branch with your commits."},
            "title": {"type": "string", "description": "open_pr: PR title."},
            "body": {"type": "string", "description": "open_pr: PR description (markdown)."},
            "draft": {"type": "boolean", "description": "open_pr: open as a draft."},
        },
        "required": ["op", "repo"],
    },
}


registry.register(
    name="github_pr_read",
    toolset="github_pr",
    schema=GITHUB_PR_READ_SCHEMA,
    handler=lambda args, **kw: _github_pr_read_handler(args, **kw),
    check_fn=_check_github_pr_read,
    requires_env=[],
    is_async=True,
    emoji="🐙",
    max_result_size_chars=80_000,
)

registry.register(
    name="github_pr_write",
    toolset="github_pr_write",
    schema=GITHUB_PR_WRITE_SCHEMA,
    handler=lambda args, **kw: _github_pr_write_handler(args, **kw),
    check_fn=_check_github_pr_write,
    requires_env=[],
    is_async=True,
    emoji="🐙",
    max_result_size_chars=16_000,
)
