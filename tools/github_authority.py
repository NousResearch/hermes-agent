"""Deterministic GitHub operation authority classification.

This module encodes the downstream Hermes Maintenance green/yellow/red
GitHub authority model as a local command classifier.  It is intentionally
conservative: only simple, parseable, read-only green operations are eligible
for prompt-free approval here.  Yellow and red operations are surfaced as
approval warnings so they cannot silently pass through the terminal guard.

The classifier does *not* execute preflight checks.  Yellow operations require
separate mechanical evidence before a future narrower implementation may allow
any of them without a human prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
import shlex
from typing import Literal

logger = logging.getLogger(__name__)

Tier = Literal["green", "yellow", "red", "unknown"]


@dataclass(frozen=True)
class GitHubAuthorityDecision:
    """Classification result for a terminal command."""

    tier: Tier
    reason: str
    command_family: str | None = None
    requires_preflight: bool = False
    evidence: tuple[str, ...] = ()


_COMPOUND_SHELL_RE = re.compile(r"(?:^|\s)(?:&&|\|\||;|\||`|\$\(|\n)")
_MUTATING_HTTP_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
_RED_REPO_EDIT_FLAGS = {
    "--visibility",
    "--default-branch",
    "--enable-issues",
    "--disable-issues",
    "--enable-wiki",
    "--disable-wiki",
    "--enable-projects",
    "--disable-projects",
    "--enable-discussions",
    "--disable-discussions",
}
_PROTECTED_BRANCHES = {"main", "master", "develop", "dev", "production", "prod", "staging"}


def classify_github_command(command: str) -> GitHubAuthorityDecision:
    """Classify a shell command under the GitHub authority policy.

    The parser deliberately only classifies single simple commands.  Compound
    shell strings are left unknown rather than partly allowlisted because a
    safe-looking first command could hide a later write/destructive action.
    """

    raw = (command or "").strip()
    if not raw:
        return GitHubAuthorityDecision("unknown", "empty command")
    if _COMPOUND_SHELL_RE.search(raw):
        return GitHubAuthorityDecision(
            "unknown",
            "compound shell command requires normal terminal guard evaluation",
        )

    try:
        argv = shlex.split(raw)
    except ValueError as exc:
        return GitHubAuthorityDecision("unknown", f"unparseable shell command: {exc}")
    if not argv:
        return GitHubAuthorityDecision("unknown", "empty command")

    exe = _basename(argv[0])
    if exe == "gh":
        return _classify_gh(argv)
    if exe == "git":
        return _classify_git(argv)
    if exe in {"curl", "wget"}:
        return _classify_http_github(argv, exe)
    return GitHubAuthorityDecision("unknown", "not a GitHub command")


def log_github_authority_decision(command: str, decision: GitHubAuthorityDecision) -> None:
    """Emit auditable evidence for classified GitHub commands."""

    if decision.tier == "unknown":
        return
    logger.info(
        "github authority decision: tier=%s family=%s preflight=%s reason=%s command=%r evidence=%s",
        decision.tier,
        decision.command_family or "unknown",
        decision.requires_preflight,
        decision.reason,
        command[:300],
        "; ".join(decision.evidence),
    )


def _basename(exe: str) -> str:
    return exe.rsplit("/", 1)[-1].lower()


def _classify_gh(argv: list[str]) -> GitHubAuthorityDecision:
    args = _strip_global_gh_flags(argv[1:])
    if not args:
        return GitHubAuthorityDecision("unknown", "gh command missing subcommand", "gh")

    top = args[0]
    sub = args[1] if len(args) > 1 else ""

    if top == "repo":
        if sub in {"delete", "archive", "unarchive", "transfer"}:
            return _red("repository deletion/archive/transfer requires Yunuen approval", "gh repo")
        if sub == "edit" and any(a in _RED_REPO_EDIT_FLAGS for a in args[2:]):
            return _red("repository settings/default branch/visibility edits are red", "gh repo")
        if sub in {"view", "list"}:
            return _green("read-only repository inspection", "gh repo")

    if top == "secret":
        return _red("secret/token/deploy-key changes are red", "gh secret")

    if top == "release":
        if sub in {"create", "upload", "delete", "edit"}:
            return _red("release publishing or mutation is red", "gh release")
        if sub in {"view", "list", "download"}:
            return _green("read-only release inspection/download", "gh release")

    if top == "pr":
        if sub in {"view", "diff", "checks", "status", "list"}:
            return _green("read-only pull request inspection", "gh pr")
        if sub in {"merge"}:
            return _yellow("PR merge requires clean downstream preflight evidence", "gh pr")
        if sub in {"close"}:
            return _yellow("PR close requires replacement-preservation evidence", "gh pr")
        if sub in {"create", "comment", "edit", "review", "ready", "reopen"}:
            return _yellow("PR write requires scoped task evidence before automation", "gh pr")

    if top == "issue":
        if sub in {"view", "list", "status"}:
            return _green("read-only issue inspection", "gh issue")
        if sub in {"create", "comment", "edit", "reopen"}:
            return _yellow("issue write requires scoped task evidence before automation", "gh issue")
        if sub in {"close"}:
            return _yellow("issue close requires replacement-preservation evidence", "gh issue")

    if top == "run":
        if sub in {"view", "list", "watch"}:
            return _green("read-only workflow run inspection", "gh run")
        if sub in {"rerun", "cancel", "delete"}:
            return _yellow("workflow run mutation requires scoped CI preflight evidence", "gh run")

    if top == "workflow":
        if sub in {"view", "list"}:
            return _green("read-only workflow inspection", "gh workflow")
        if sub in {"run", "enable", "disable"}:
            return _red("workflow dispatch/enable/disable may deploy or alter automation", "gh workflow")

    if top == "api":
        return _classify_gh_api(args[1:])

    return GitHubAuthorityDecision("unknown", "gh subcommand is not in GitHub authority allowlist", "gh")


def _strip_global_gh_flags(args: list[str]) -> list[str]:
    """Remove common gh global flags before subcommand parsing."""

    out: list[str] = []
    i = 0
    flags_with_value = {"--repo", "-R", "--hostname"}
    while i < len(args):
        arg = args[i]
        if arg in flags_with_value:
            i += 2
            continue
        if any(arg.startswith(flag + "=") for flag in flags_with_value if flag.startswith("--")):
            i += 1
            continue
        if arg in {"--help", "--version"}:
            return ["help"]
        out.append(arg)
        i += 1
    return out


def _classify_gh_api(args: list[str]) -> GitHubAuthorityDecision:
    method = "GET"
    path = ""
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in {"-X", "--method"} and i + 1 < len(args):
            method = args[i + 1].upper()
            i += 2
            continue
        if arg.startswith("--method="):
            method = arg.split("=", 1)[1].upper()
            i += 1
            continue
        if not arg.startswith("-") and not path:
            path = arg
        i += 1
    if method in _MUTATING_HTTP_METHODS:
        if _api_path_is_red(path):
            return _red("mutating GitHub API call targets red repository/security/publishing surface", "gh api")
        return _yellow("mutating GitHub API call requires scoped preflight evidence", "gh api")
    if path and ("api.github.com" in path or path.startswith("/") or path.startswith("repos/")):
        return _green("read-only GitHub API request", "gh api")
    return GitHubAuthorityDecision("unknown", "gh api path/method not recognized", "gh api")


def _classify_git(argv: list[str]) -> GitHubAuthorityDecision:
    if len(argv) < 2:
        return GitHubAuthorityDecision("unknown", "git command missing subcommand", "git")
    sub = argv[1]
    args = argv[2:]

    if sub in {"status", "diff", "log", "show", "branch", "remote"}:
        if sub == "branch" and any(a in {"-d", "-D", "--delete"} for a in args):
            return _yellow("branch deletion requires merged/owned-branch preflight evidence", "git")
        return _green("read-only git repository inspection", "git")

    if sub == "push":
        if any(_is_force_flag(a) for a in args):
            branch = _last_non_flag(args)
            if branch in _PROTECTED_BRANCHES:
                return _red("force push to protected/shared branch is red", "git push")
            if "--force-with-lease" in args or any(a.startswith("--force-with-lease=") for a in args):
                return _yellow("force-with-lease requires remote-head and branch-ownership evidence", "git push")
            return _red("plain git force push rewrites remote history", "git push")
        if any(a in {"--delete", ":main", ":master"} for a in args):
            return _yellow("remote branch deletion requires merged/owned-branch preflight evidence", "git push")
        target = _last_non_flag(args)
        if target in _PROTECTED_BRANCHES:
            return _red("push directly to protected/shared branch is red", "git push")
        return _yellow("git push requires downstream repo, scoped branch, cleanliness, and task-link evidence", "git push")

    if sub == "commit" and "--amend" in args:
        return _yellow("commit amend requires owned-branch and remote-head evidence", "git commit")

    if sub in {"reset", "clean"}:
        return _red("destructive git history/worktree operation requires explicit approval", "git")

    return GitHubAuthorityDecision("unknown", "git subcommand is not in GitHub authority allowlist", "git")


def _classify_http_github(argv: list[str], exe: str) -> GitHubAuthorityDecision:
    if exe == "wget":
        if any(a.startswith("--method=") and a.split("=", 1)[1].upper() in _MUTATING_HTTP_METHODS for a in argv[1:]):
            return _yellow("mutating GitHub HTTP request requires scoped preflight evidence", exe)
        url = next((a for a in argv[1:] if "api.github.com" in a or "github.com" in a), "")
        if url:
            return _green("read-only GitHub HTTP request", exe)
        return GitHubAuthorityDecision("unknown", "not a GitHub HTTP request", exe)

    method = "GET"
    url = ""
    data_flag = False
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg in {"-X", "--request"} and i + 1 < len(argv):
            method = argv[i + 1].upper()
            i += 2
            continue
        if arg.startswith("-X") and len(arg) > 2:
            method = arg[2:].upper()
        if arg in {"-d", "--data", "--data-raw", "--data-binary", "--form", "-F"}:
            data_flag = True
        if arg.startswith("http") and ("api.github.com" in arg or "github.com" in arg):
            url = arg
        i += 1
    if not url:
        return GitHubAuthorityDecision("unknown", "not a GitHub HTTP request", exe)
    if method in _MUTATING_HTTP_METHODS or data_flag:
        if _api_path_is_red(url):
            return _red("mutating GitHub HTTP call targets red repository/security/publishing surface", exe)
        return _yellow("mutating GitHub HTTP call requires scoped preflight evidence", exe)
    return _green("read-only GitHub HTTP request", exe)


def _api_path_is_red(path: str) -> bool:
    lowered = (path or "").lower()
    red_fragments = (
        "/actions/secrets",
        "/branches/main/protection",
        "/branches/master/protection",
        "/rulesets",
        "/releases",
        "/packages",
        "/hooks",
        "/keys",
        "/deployments",
        "/repos/",  # PATCH/DELETE repo endpoints are settings/deletion-sensitive by default.
    )
    if any(fragment in lowered for fragment in red_fragments):
        return True
    return False


def _is_force_flag(arg: str) -> bool:
    return arg in {"--force", "-f", "+"} or arg.startswith("--force=") or arg.startswith("--force-with-lease")


def _last_non_flag(args: list[str]) -> str:
    for arg in reversed(args):
        if not arg.startswith("-") and ":" not in arg:
            return arg
    return ""


def _green(reason: str, family: str) -> GitHubAuthorityDecision:
    return GitHubAuthorityDecision(
        "green",
        reason,
        family,
        requires_preflight=False,
        evidence=("simple command parsed", "read-only/safe GitHub operation"),
    )


def _yellow(reason: str, family: str) -> GitHubAuthorityDecision:
    return GitHubAuthorityDecision(
        "yellow",
        reason,
        family,
        requires_preflight=True,
        evidence=("simple command parsed", "mechanical preflight evidence required before automation"),
    )


def _red(reason: str, family: str) -> GitHubAuthorityDecision:
    return GitHubAuthorityDecision(
        "red",
        reason,
        family,
        requires_preflight=True,
        evidence=("simple command parsed", "Yunuen approval gate required"),
    )
