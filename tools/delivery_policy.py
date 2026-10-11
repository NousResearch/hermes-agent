"""Immutable runtime capability policy for delegated software-delivery roles."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import os
import re
import shlex
from typing import Any, Final, Iterator, Mapping

DELIVERY_ROLES: Final[tuple[str, ...]] = (
    "implementer",
    "reviewer",
    "merger",
    "closure_controller",
)
_EXACT_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")

ROLE_CONTRACTS: Final[dict[str, str]] = {
    "implementer": (
        "You are the IMPLEMENTER. Reconstruct the complete business outcome and production entry point, then "
        "implement and test the real production path. You may edit, test, commit, push, and open a pull request. "
        "You MUST NOT delegate your authority, independently approve or pass review on your own work, merge it, "
        "close the pull request or tracker item, or claim delivery is complete. Stop at a review-ready handoff with "
        "the exact head SHA, changed files, and exact test/CI evidence."
    ),
    "reviewer": (
        "You are the INDEPENDENT REVIEWER. Review the supplied exact head SHA against the complete acceptance "
        "ledger and verify required tests and CI. Missing evidence, a SHA mismatch, skipped relevant tests, or "
        "absent required CI is FAIL or BLOCKED, never an assumed pass. You are read-only: do not patch or write "
        "files, commit, push, approve, merge, close work, mutate trackers, delegate work, message external systems, "
        "or repair defects. Report findings and evidence only; you may not repair and then approve your own repair."
    ),
    "merger": (
        "You are the MERGER. You may perform only the merge lifecycle for the supplied exact SHA after checking "
        "the supplied independent passing-review and CI evidence. Do not implement fixes, perform the independent "
        "review, approve a different SHA, mutate unrelated tracker state, or close the task. If the head SHA changed "
        "or evidence is missing, stop BLOCKED. Merge is delivery, not closure."
    ),
    "closure_controller": (
        "You are the CLOSURE CONTROLLER. You may perform post-merge acceptance checks and close tracked work only "
        "for the supplied merged SHA and merge evidence. Do not implement, patch, commit, push, review, approve, or "
        "merge code. Missing merge or post-merge acceptance evidence is BLOCKED; never infer completion from worker "
        "self-reports."
    ),
}


@dataclass(frozen=True, slots=True)
class DeliveryPolicy:
    """Immutable role and evidence attached to one delegated child."""

    role: str
    exact_sha: str = ""
    independent_review: str = ""
    ci_evidence: str = ""
    merged_sha: str = ""
    merge_evidence: str = ""
    post_merge_acceptance: str = ""

    def __post_init__(self) -> None:
        if self.role not in DELIVERY_ROLES:
            raise ValueError(f"Invalid delivery_role {self.role!r}; expected one of {', '.join(DELIVERY_ROLES)}")
        if self.role == "merger":
            if not _EXACT_SHA_RE.fullmatch(self.exact_sha):
                raise ValueError("delivery_role 'merger' requires delivery_evidence.exact_sha as a full 40-character SHA")
            if not self.independent_review.strip() or not self.ci_evidence.strip():
                raise ValueError("delivery_role 'merger' requires independent_review and ci_evidence")
            if self.exact_sha.lower() not in self.independent_review.lower():
                raise ValueError("delivery_evidence.independent_review must identify the reviewed exact_sha")
            if self.exact_sha.lower() not in self.ci_evidence.lower():
                raise ValueError("delivery_evidence.ci_evidence must identify the tested exact_sha")
        if self.role == "closure_controller":
            if not _EXACT_SHA_RE.fullmatch(self.merged_sha):
                raise ValueError(
                    "delivery_role 'closure_controller' requires delivery_evidence.merged_sha as a full 40-character SHA"
                )
            if not self.merge_evidence.strip() or not self.post_merge_acceptance.strip():
                raise ValueError(
                    "delivery_role 'closure_controller' requires merge_evidence and post_merge_acceptance"
                )
            if self.merged_sha.lower() not in self.merge_evidence.lower():
                raise ValueError("delivery_evidence.merge_evidence must identify the merged_sha")
            if self.merged_sha.lower() not in self.post_merge_acceptance.lower():
                raise ValueError("delivery_evidence.post_merge_acceptance must identify the accepted merged_sha")


def normalize_delivery_role(value: Any) -> str | None:
    """Normalize an explicit role while preserving ordinary delegation when omitted."""

    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in DELIVERY_ROLES:
        raise ValueError(f"Invalid delivery_role {value!r}; expected one of {', '.join(DELIVERY_ROLES)}")
    return normalized


def build_delivery_policy(role: Any, evidence: Any = None) -> DeliveryPolicy | None:
    normalized = normalize_delivery_role(role)
    if normalized is None:
        return None
    values: Mapping[str, Any] = evidence if isinstance(evidence, Mapping) else {}
    return DeliveryPolicy(
        role=normalized,
        exact_sha=str(values.get("exact_sha") or "").strip(),
        independent_review=str(values.get("independent_review") or "").strip(),
        ci_evidence=str(values.get("ci_evidence") or "").strip(),
        merged_sha=str(values.get("merged_sha") or "").strip(),
        merge_evidence=str(values.get("merge_evidence") or "").strip(),
        post_merge_acceptance=str(values.get("post_merge_acceptance") or "").strip(),
    )


_CURRENT_POLICY: ContextVar[DeliveryPolicy | None] = ContextVar("hermes_delivery_policy", default=None)


@contextmanager
def delivery_role_context(policy: DeliveryPolicy | None) -> Iterator[None]:
    token = _CURRENT_POLICY.set(policy)
    try:
        yield
    finally:
        _CURRENT_POLICY.reset(token)


def current_delivery_policy() -> DeliveryPolicy | None:
    return _CURRENT_POLICY.get()


def current_delivery_role() -> str | None:
    policy = current_delivery_policy()
    return policy.role if policy else None


_READ_ONLY_TOOLS: Final[frozenset[str]] = frozenset(
    {"terminal", "read_file", "search_files", "skills_list", "skill_view"}
)
_IMPLEMENTER_DENIED_TOOLS: Final[frozenset[str]] = frozenset({"delegate_task"})
_RESTRICTED_ROLES: Final[frozenset[str]] = frozenset({"reviewer", "merger", "closure_controller"})


def effective_tool_definitions(definitions: list[dict], policy: DeliveryPolicy | None) -> list[dict]:
    """Apply the final role boundary after all composite/dynamic toolsets expand."""

    if policy is None:
        return list(definitions)
    if policy.role == "implementer":
        return [
            item for item in definitions
            if (item.get("function") or {}).get("name") not in _IMPLEMENTER_DENIED_TOOLS
        ]
    allowed = _READ_ONLY_TOOLS
    by_name = {(item.get("function") or {}).get("name"): item for item in definitions}
    from tools.registry import registry

    # Required workflow skills remain inspectable even when the parent's
    # selected toolsets omitted the skill bundle. Every other capability is
    # subtractive: assigning a delivery role must not grant terminal or file
    # access the parent did not delegate.
    for item in registry.get_definitions({"skills_list", "skill_view"}, quiet=True):
        by_name.setdefault((item.get("function") or {}).get("name"), item)
    return [by_name[name] for name in sorted(allowed) if name in by_name]


def apply_delivery_capabilities(agent: Any, policy: DeliveryPolicy | None) -> None:
    if policy is None:
        return
    agent._delivery_policy = policy
    agent.tools = effective_tool_definitions(list(getattr(agent, "tools", None) or []), policy)
    agent.valid_tool_names = {
        (item.get("function") or {}).get("name") for item in agent.tools if (item.get("function") or {}).get("name")
    }


def _action_words(tool_name: str, args: Mapping[str, Any]) -> set[str]:
    words = {part for part in re.split(r"[^a-z]+", tool_name.lower()) if part}
    action_keys = {"action", "operation", "method", "event", "state", "status", "review", "command"}

    def _walk(value: Any, action_context: bool = False) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                key_words = {part for part in re.split(r"[^a-z]+", str(key).lower()) if part}
                nested_action_context = action_context or bool(key_words & action_keys)
                if nested_action_context:
                    words.update(key_words)
                _walk(nested, nested_action_context)
        elif isinstance(value, (list, tuple, set)):
            for nested in value:
                _walk(nested, action_context)
        elif action_context:
            words.update(part for part in re.split(r"[^a-z]+", str(value).lower()) if part)

    _walk(args)
    return words


def delivery_tool_block_reason(
    tool_name: str,
    args: Mapping[str, Any] | None = None,
    policy: DeliveryPolicy | None = None,
) -> str | None:
    """Defense-in-depth for fabricated calls and action-oriented composite tools."""

    policy = policy or current_delivery_policy()
    if policy is None:
        return None
    if policy.role in _RESTRICTED_ROLES and tool_name not in _READ_ONLY_TOOLS:
        return f"{policy.role} delivery policy forbids tool {tool_name!r}"
    if tool_name == "terminal":
        terminal_args = args or {}
        if policy.role in _RESTRICTED_ROLES and terminal_args.get("background", False):
            return f"{policy.role} terminal policy forbids background processes"
        if reason := validate_delivery_terminal_command(str(terminal_args.get("command") or ""), policy):
            return reason
    if policy.role == "implementer":
        if tool_name in _IMPLEMENTER_DENIED_TOOLS:
            return "implementer delivery policy forbids delegation; another worker cannot exercise this role's authority"
        words = _action_words(tool_name, args or {})
        tracker_markers = {"issue", "ticket", "task", "tracker", "linear", "jira"}
        closure_verbs = {"delete", "remove", "complete", "completed", "done", "resolve", "resolved", "close", "closed"}
        if words & tracker_markers and words & closure_verbs:
            return "implementer delivery policy forbids tracker deletion and closure actions"
        forbidden = words & {
            "approve", "approval", "merge", "merged", "close", "closed", "complete", "completed", "done",
            "resolve", "resolved",
        }
        if "review" in words and ({"pass", "approve", "submit"} & words or tool_name.lower().endswith("review")):
            forbidden.add("review")
        if forbidden:
            return "implementer delivery policy forbids independent review approval, merge, and closure actions"
    return None


_GIT_READ: Final[frozenset[str]] = frozenset(
    {"status", "log", "diff", "show", "rev-parse", "describe", "ls-files", "ls-tree", "cat-file", "diff-tree", "diff-index", "grep", "shortlog", "name-rev"}
)
_GH_READ: Final[dict[str, frozenset[str]]] = {
    "pr": frozenset({"view", "checks", "diff", "list", "status"}),
    "run": frozenset({"view", "list", "watch"}),
    "repo": frozenset({"view", "list"}),
    "issue": frozenset({"view", "list", "status"}),
    "workflow": frozenset({"view", "list"}),
    "release": frozenset({"view", "list"}),
}
_GH_IMPLEMENTER: Final[dict[str, frozenset[str]]] = {
    **_GH_READ,
    "pr": _GH_READ["pr"] | frozenset({"create", "edit", "comment"}),
    "issue": _GH_READ["issue"] | frozenset({"comment"}),
}
_TEST_BINS: Final[frozenset[str]] = frozenset(
    {"pytest", "ruff", "mypy", "pyright", "tsc", "eslint", "golangci-lint", "cargo", "go", "npm", "pnpm", "yarn", "bun", "gradle", "gradlew", "mvn", "dotnet"}
)


def _tokens(command: str) -> list[str] | None:
    if not command.strip() or "\n" in command or "\r" in command or "`" in command or "$" in command:
        return None
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>()")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return None
    punctuation = {";", "&", "&&", "|", "||", "<", ">", "<<", ">>", "(", ")"}
    return None if not tokens or any(token in punctuation for token in tokens) else tokens


def _git_subcommand(tokens: list[str]) -> str | None:
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "-C":
            index += 2
        elif token in {"--no-pager", "--literal-pathspecs", "--no-optional-locks"} or token.startswith(("--git-dir=", "--work-tree=")):
            index += 1
        elif token.startswith("-"):
            return None
        else:
            return token
    return None


def _read_only_terminal(tokens: list[str]) -> bool:
    executable = os.path.basename(tokens[0]).lower()
    lowered = [token.lower() for token in tokens]
    if executable == "git":
        forbidden_options = {"--ext-diff", "--textconv", "--filters", "--path", "--batch", "--batch-command", "--follow-symlinks", "--output", "-o", "--open-files-in-pager", "--web"}
        return _git_subcommand(tokens) in _GIT_READ and not any(
            token in forbidden_options
            or token.startswith(("--output=", "--open-files-in-pager="))
            or original.startswith("-O")
            for original, token in zip(tokens, lowered)
        )
    if executable == "gh":
        return (
            len(tokens) >= 3
            and lowered[2] in _GH_READ.get(lowered[1], frozenset())
            and not any(token == "--web" or token.startswith("--web=") for token in lowered)
        )
    if executable in {"python", "python3"}:
        return len(tokens) >= 3 and tokens[1] == "-m" and lowered[2] in {"pytest", "ruff", "mypy", "pyright", "unittest"}
    if executable not in _TEST_BINS:
        return False
    if executable == "ruff":
        return len(tokens) >= 2 and lowered[1] == "check" and not {"--fix", "--unsafe-fixes", "--add-noqa"} & set(lowered)
    if executable == "eslint":
        return not {"--fix", "--fix-dry-run"} & set(lowered)
    if executable in {"npm", "pnpm", "yarn", "bun"}:
        operation = lowered[2] if len(lowered) >= 3 and lowered[1] == "run" else lowered[1] if len(lowered) >= 2 else ""
        return operation in {"test", "lint", "check", "build", "typecheck", "type-check"}
    if executable == "cargo":
        return len(lowered) >= 2 and lowered[1] in {"test", "check", "clippy", "build"}
    if executable == "go":
        return len(lowered) >= 2 and lowered[1] in {"test", "vet", "build"}
    if executable in {"gradle", "gradlew"}:
        tasks = [token for token in lowered[1:] if not token.startswith("-")]
        return bool(tasks) and set(tasks) <= {"test", "check", "build", "assemble", "lint"} and not any(token.startswith("-i") or token == "--init-script" for token in lowered[1:])
    if executable == "mvn":
        goals = [token for token in lowered[1:] if not token.startswith("-")]
        return bool(goals) and set(goals) <= {"test", "verify", "package", "compile", "validate", "checkstyle:check"}
    if executable == "dotnet":
        return len(lowered) >= 2 and lowered[1] in {"test", "build"}
    return True


def validate_delivery_terminal_command(command: str, policy: DeliveryPolicy | None = None) -> str | None:
    """Return a refusal reason for a role-prohibited shell command."""

    policy = policy or current_delivery_policy()
    if policy is None:
        return None
    tokens = _tokens(command)
    if policy.role == "implementer":
        normalized_command = re.sub(r"\s+", " ", command.lower())
        forbidden_patterns = (
            "gh pr merge", "gh pr review", "gh pr close", "gh issue close", "gh api", "gh alias",
            "gh extension", "git merge",
        )
        if any(pattern in normalized_command for pattern in forbidden_patterns) or re.search(
            r"\bgit\s+push\b[^;&|\n]*(?:--force(?:-with-lease)?)(?:\s|=|$)", normalized_command,
        ):
            return "implementer delivery policy forbids review, merge, closure, force-push, and unbounded GitHub commands"
        if not tokens:
            # Other shell composition is available for implementation work. Lifecycle forms were rejected above.
            return None
        executable = os.path.basename(tokens[0]).lower()
        lowered = [token.lower() for token in tokens]
        if executable != "gh" and any(os.path.basename(token).lower() == "gh" for token in tokens[1:]):
            return "implementer delivery policy forbids wrapped GitHub commands"
        if executable == "git" and _git_subcommand(tokens) in {"merge"}:
            return "implementer delivery policy forbids merging"
        if executable == "git" and _git_subcommand(tokens) == "push" and any(
            token == "--force" or token.startswith("--force=") or token == "--force-with-lease"
            or token.startswith("--force-with-lease=") for token in lowered
        ):
            return "implementer delivery policy forbids force-pushing"
        if executable == "gh":
            if len(tokens) < 3 or lowered[2] not in _GH_IMPLEMENTER.get(lowered[1], frozenset()):
                return "implementer delivery policy permits only bounded pull-request update and GitHub read commands"
        return None
    if not tokens:
        return f"{policy.role} terminal policy allows one bounded evidence/lifecycle command only"
    if _read_only_terminal(tokens):
        return None
    executable = os.path.basename(tokens[0]).lower()
    lowered = [token.lower() for token in tokens]
    if policy.role == "merger" and executable == "gh" and len(tokens) >= 3 and lowered[1:3] == ["pr", "merge"]:
        try:
            index = lowered.index("--match-head-commit")
            supplied_sha = tokens[index + 1]
        except (ValueError, IndexError):
            supplied_sha = ""
        if supplied_sha.lower() == policy.exact_sha.lower() and any(flag in lowered for flag in {"--merge", "--squash", "--rebase"}):
            return None
        return "merger policy requires gh pr merge with --match-head-commit matching the reviewed exact SHA and an explicit merge method"
    if (
        policy.role == "closure_controller"
        and executable == "gh"
        and len(tokens) >= 3
        and lowered[1:3] == ["issue", "close"]
        and policy.merged_sha.lower() in {token.lower() for token in tokens}
    ):
        return None
    return f"{policy.role} terminal policy rejects commands outside its evidence and lifecycle allowlist"
