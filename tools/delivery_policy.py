"""Capability policy for delegated software-delivery roles.

The policy is enforced twice: tools outside the role surface are removed before a
model sees them, and every dispatch is checked again immediately before the
registered handler runs.  Delivery workers never receive generic command,
execute-code, browser, or MCP capabilities.  Effects that need git, GitHub, or a
runtime are exposed only through :mod:`tools.delivery_action`, whose handler
constructs argv itself and binds remote operations to policy evidence.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional


DELIVERY_ROLES = ("implementer", "reviewer", "merger", "closure_controller")
_ROLE_SET = frozenset(DELIVERY_ROLES)
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_IMMUTABLE_IMAGE_RE = re.compile(
    r"^[A-Za-z0-9._-]+(?::[0-9]+)?(?:/[A-Za-z0-9._-]+)+@sha256:[0-9a-f]{64}$"
)

# Deliberately small.  Generic tools are not a security boundary when they can
# start subprocesses or reach arbitrary network services under the Hermes OS
# identity.  All such effects go through delivery_action.
_ROLE_TOOLS = {
    "implementer": frozenset({
        "read_file", "search_files", "write_file", "patch", "skill_view",
        "delivery_action",
    }),
    "reviewer": frozenset({"skill_view", "delivery_action"}),
    "merger": frozenset({"skill_view", "delivery_action"}),
    "closure_controller": frozenset({"skill_view", "delivery_action"}),
}

_ROLE_CONTRACT = {
    "implementer": (
        "You may edit files and use delivery_action to run sandboxed verification, stage, commit, push, and open a "
        "pull request. You may not review, approve, merge, close, or use arbitrary command/network/MCP execution."
    ),
    "reviewer": (
        "You may inspect the policy-bound exact commit and run verification only through delivery_action. The exact "
        "commit is read from git objects or a disposable read-only checkout; the live checkout is not exposed. You "
        "may report findings but may not edit, approve, merge, close, push, or invoke arbitrary commands/network/MCP."
    ),
    "merger": (
        "You may request only the policy-bound structured merge operation. The server re-queries the exact PR head, "
        "an independent approval of that head, and required CI immediately before merging. Supplied prose is never "
        "evidence and bypass flags are not accepted."
    ),
    "closure_controller": (
        "You may request only the policy-bound structured close operation. The server verifies that the merged commit "
        "is on the repository default branch and runs the server-bound post-merge acceptance recipe before closing "
        "the bound issue. The recipe argv, immutable image digest, and attestation identity cannot be selected or "
        "overridden by this worker."
    ),
}

_CURRENT_POLICY: ContextVar[Optional["DeliveryPolicy"]] = ContextVar(
    "hermes_delivery_policy", default=None
)


def normalize_delivery_role(value: Any) -> Optional[str]:
    """Accept only exact canonical role values; explicit aliases fail closed."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("delivery_role must be a string when provided")
    if value not in _ROLE_SET:
        raise ValueError(
            f"Invalid delivery_role {value!r}; expected one of {', '.join(DELIVERY_ROLES)}"
        )
    return value


@dataclass(frozen=True)
class AcceptanceRecipe:
    """Server-owned acceptance runner bound into a closure policy before spawn."""

    argv: tuple[str, ...]
    image: str
    identity: str


def _immutable_image(value: Any, field: str, *, required: bool = False) -> str:
    if value in (None, "") and not required:
        return ""
    if not isinstance(value, str) or not _IMMUTABLE_IMAGE_RE.fullmatch(value):
        raise ValueError(
            f"delegation.delivery.{field} must be an immutable image reference ending in "
            "@sha256:<64 lowercase hex>"
        )
    return value


def _trusted_runtime_config(
    role: str, value: Optional[Mapping[str, Any]],
) -> tuple[str, Optional[AcceptanceRecipe]]:
    """Validate server config and freeze it into the per-child policy."""
    config = value or {}
    if not isinstance(config, Mapping):
        raise ValueError("delegation.delivery must be a mapping")
    unknown = sorted(set(config) - {"verification_image", "acceptance"})
    if unknown:
        raise ValueError(f"delegation.delivery contains unsupported field(s): {', '.join(unknown)}")
    image = _immutable_image(config.get("verification_image"), "verification_image")
    acceptance = config.get("acceptance") or {}
    if not isinstance(acceptance, Mapping):
        raise ValueError("delegation.delivery.acceptance must be a mapping")
    unknown_acceptance = sorted(set(acceptance) - {"command", "image"})
    if unknown_acceptance:
        raise ValueError(
            "delegation.delivery.acceptance contains unsupported field(s): " + ", ".join(unknown_acceptance)
        )
    if role != "closure_controller":
        return image, None
    argv = acceptance.get("command")
    if (
        not isinstance(argv, list) or not argv
        or not all(isinstance(item, str) and item and "\x00" not in item for item in argv)
        or len(argv) > 128 or sum(len(item.encode("utf-8")) for item in argv) > 32_768
    ):
        raise ValueError("delegation.delivery.acceptance.command must be a non-empty bounded argv array")
    acceptance_image = _immutable_image(
        acceptance.get("image"), "acceptance.image", required=True,
    )
    canonical = json.dumps(
        {"argv": argv, "image": acceptance_image},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    recipe = AcceptanceRecipe(
        argv=tuple(argv), image=acceptance_image,
        identity="sha256:" + hashlib.sha256(canonical).hexdigest(),
    )
    return image, recipe


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"delivery_evidence.{field} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"delivery_evidence.{field} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"delivery_evidence.{field} must be a positive integer")
    return result


def _repository(value: Any, field: str = "repository") -> str:
    result = str(value or "").strip()
    if not _REPOSITORY_RE.fullmatch(result):
        raise ValueError(f"delivery_evidence.{field} must be an owner/repository slug")
    return result


def _sha(value: Any, field: str) -> str:
    result = str(value or "").strip().lower()
    if not _SHA_RE.fullmatch(result):
        raise ValueError(f"delivery_evidence.{field} must be a full 40-character commit SHA")
    return result


@dataclass(frozen=True)
class DeliveryPolicy:
    """Immutable role and machine-bound evidence inherited by a child agent."""

    role: Optional[str]
    repository: str = ""
    pull_request: Optional[int] = None
    issue: Optional[int] = None
    exact_sha: str = ""
    merged_sha: str = ""
    workspace: str = ""
    verification_image: str = ""
    acceptance_recipe: Optional[AcceptanceRecipe] = None

    @property
    def allowed_tools(self) -> Optional[frozenset[str]]:
        return None if self.role is None else _ROLE_TOOLS[self.role]

    @property
    def contract(self) -> str:
        if self.role is None:
            return ""
        return _ROLE_CONTRACT[self.role]

    def bound_to_workspace(self, workspace: Any) -> "DeliveryPolicy":
        if self.role is None or workspace is None:
            return self
        try:
            root = Path(workspace).expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            return self
        if not root.is_dir():
            return self
        return replace(self, workspace=str(root))


def build_delivery_policy(
    role: Any, evidence: Optional[Mapping[str, Any]] = None,
    *, trusted_runtime: Optional[Mapping[str, Any]] = None,
) -> DeliveryPolicy:
    """Validate caller evidence and bind separately sourced server runtime policy."""
    normalized = normalize_delivery_role(role)
    if normalized is None:
        if evidence:
            raise ValueError("delivery_evidence requires a delivery_role")
        return DeliveryPolicy(role=None)
    if evidence is None:
        evidence = {}
    if not isinstance(evidence, Mapping):
        raise ValueError("delivery_evidence must be an object")
    verification_image, acceptance_recipe = _trusted_runtime_config(normalized, trusted_runtime)

    # Unknown fields are rejected so old free-form evidence cannot appear to
    # authorize an operation after a schema typo or downgrade.
    allowed = {
        "implementer": frozenset(),
        "reviewer": frozenset({"repository", "pull_request", "exact_sha"}),
        "merger": frozenset({"repository", "pull_request", "exact_sha"}),
        "closure_controller": frozenset({"repository", "issue", "merged_sha"}),
    }[normalized]
    unknown = sorted(set(evidence) - allowed)
    if unknown:
        raise ValueError(
            f"delivery_evidence contains unsupported field(s) for {normalized}: {', '.join(unknown)}"
        )

    if normalized == "implementer":
        return DeliveryPolicy(role=normalized, verification_image=verification_image)

    if normalized == "reviewer":
        if not evidence:
            # /review also supports non-repository artifact/conversation review.
            # It gets no live filesystem or generic execution capability.
            return DeliveryPolicy(role=normalized, verification_image=verification_image)
        required = ("repository", "pull_request", "exact_sha")
        missing = [field for field in required if evidence.get(field) in (None, "")]
        if missing:
            raise ValueError(
                "reviewer delivery_evidence requires repository, pull_request, and exact_sha"
            )
        return DeliveryPolicy(
            role=normalized,
            repository=_repository(evidence["repository"]),
            pull_request=_positive_int(evidence["pull_request"], "pull_request"),
            exact_sha=_sha(evidence["exact_sha"], "exact_sha"),
            verification_image=verification_image,
        )

    if normalized == "merger":
        required = ("repository", "pull_request", "exact_sha")
        missing = [field for field in required if evidence.get(field) in (None, "")]
        if missing:
            raise ValueError(
                "merger delivery_evidence requires repository, pull_request, and exact_sha"
            )
        return DeliveryPolicy(
            role=normalized,
            repository=_repository(evidence["repository"]),
            pull_request=_positive_int(evidence["pull_request"], "pull_request"),
            exact_sha=_sha(evidence["exact_sha"], "exact_sha"),
            verification_image=verification_image,
        )

    required = ("repository", "issue", "merged_sha")
    missing = [field for field in required if evidence.get(field) in (None, "")]
    if missing:
        raise ValueError(
            "closure_controller delivery_evidence requires repository, issue, and merged_sha"
        )
    return DeliveryPolicy(
        role=normalized,
        repository=_repository(evidence["repository"]),
        issue=_positive_int(evidence["issue"], "issue"),
        merged_sha=_sha(evidence["merged_sha"], "merged_sha"),
        verification_image=verification_image,
        acceptance_recipe=acceptance_recipe,
    )


def current_delivery_policy() -> Optional[DeliveryPolicy]:
    return _CURRENT_POLICY.get()


def current_delivery_role() -> Optional[str]:
    policy = current_delivery_policy()
    return policy.role if policy is not None else None


@contextmanager
def delivery_role_context(policy_or_role: Any) -> Iterator[None]:
    policy = (
        policy_or_role
        if isinstance(policy_or_role, DeliveryPolicy)
        else build_delivery_policy(policy_or_role)
    )
    token = _CURRENT_POLICY.set(policy)
    try:
        yield
    finally:
        _CURRENT_POLICY.reset(token)


def delivery_tool_block_reason(
    tool_name: str,
    arguments: Optional[Mapping[str, Any]] = None,
    policy: Optional[DeliveryPolicy] = None,
) -> Optional[str]:
    """Return a dispatch-time denial for the active delivery role.

    This intentionally does not inspect command text.  Generic command and MCP
    capabilities are absent regardless of spelling, wrappers, GraphQL shape, or
    plugin-provided tool names.
    """
    policy = policy or current_delivery_policy()
    if policy is None or policy.role is None:
        return None
    allowed = policy.allowed_tools
    assert allowed is not None
    if tool_name not in allowed:
        return (
            f"Blocked by immutable delivery role '{policy.role}': tool '{tool_name}' "
            "is outside this role's capability set."
        )
    return _workspace_path_block(policy, tool_name, arguments or {})


def _workspace_path_block(
    policy: DeliveryPolicy, tool_name: str, arguments: Mapping[str, Any]
) -> Optional[str]:
    if tool_name not in {"read_file", "search_files", "write_file", "patch"}:
        return None
    if policy.role != "implementer":
        return f"Blocked by immutable delivery role '{policy.role}': live workspace access is unavailable."
    if not policy.workspace:
        return "Blocked: delivery worker has no bound workspace."

    keys = {
        "read_file": ("path",),
        "search_files": ("path",),
        "write_file": ("path",),
        "patch": ("path",),
    }[tool_name]
    # Multi-file patch format cannot be safely path-normalized by this boundary.
    # Delivery workers use single-file replace mode; staged source edits remain
    # available without exposing repository control files.
    if tool_name == "patch" and arguments.get("mode", "replace") != "replace":
        return "Blocked: delivery-role patch accepts replace mode only."

    root = Path(policy.workspace)
    for key in keys:
        raw = arguments.get(key)
        if raw in (None, ""):
            if tool_name == "search_files":
                raw = policy.workspace
            else:
                continue
        candidate = Path(str(raw)).expanduser()
        if not candidate.is_absolute():
            return (
                f"Blocked: delivery-role {tool_name}.{key} must be an absolute path "
                f"inside {policy.workspace}."
            )
        try:
            lexical = candidate.relative_to(root)
            current = root
            for part in lexical.parts:
                current /= part
                if current.is_symlink():
                    return (
                        f"Blocked: delivery-role {tool_name}.{key} traverses a symbolic link; "
                        "delivery file paths must be physically contained in the workspace."
                    )
            resolved = candidate.resolve(strict=False)
            resolved.relative_to(root)
        except (OSError, RuntimeError, ValueError):
            return f"Blocked: {tool_name}.{key} escapes the delivery workspace {policy.workspace}."
        relative = resolved.relative_to(root)
        if ".git" in relative.parts:
            return "Blocked: repository control files are not exposed to delivery file tools."
    return None


def effective_tool_definitions(
    tools: list[dict[str, Any]], policy: Optional[DeliveryPolicy]
) -> list[dict[str, Any]]:
    """Return the effective role surface and inject only the structured action.

    Delivery actions are a built-in authorization boundary, not a general
    capability inherited from the parent.  All ordinary tools remain
    intersection-only, preserving parent disables and toolset selection.
    """
    if policy is None or policy.role is None:
        return tools
    filtered = filter_delivery_tool_definitions(tools, policy)
    names = {tool.get("function", {}).get("name") for tool in filtered}
    if "delivery_action" not in names:
        from tools import delivery_action as _delivery_action  # noqa: F401
        from tools.registry import registry

        filtered.extend(registry.get_definitions({"delivery_action"}, quiet=True))
    return filtered


def validate_delivery_terminal_command(command: str, policy: Optional[DeliveryPolicy]) -> Optional[str]:
    """Compatibility API: generic terminal is categorically unavailable."""
    if policy is None or policy.role is None:
        return None
    return (
        f"Blocked by immutable delivery role '{policy.role}': generic terminal execution is unavailable; "
        "use delivery_action for a structured operation."
    )


def apply_delivery_capabilities(agent: Any, policy: DeliveryPolicy) -> None:
    """Apply the effective role surface to an agent, including restored tools."""
    if policy.role is None:
        return
    agent.tools = effective_tool_definitions(list(getattr(agent, "tools", None) or []), policy)
    agent.valid_tool_names = {
        tool.get("function", {}).get("name") for tool in agent.tools
        if tool.get("function", {}).get("name")
    }
    agent._delivery_policy = policy
    agent._delivery_role = policy.role


def filter_delivery_tool_definitions(tools: list[dict[str, Any]], role_or_policy: Any) -> list[dict[str, Any]]:
    policy = (
        role_or_policy
        if isinstance(role_or_policy, DeliveryPolicy)
        else build_delivery_policy(role_or_policy)
    )
    if policy.role is None:
        return tools
    return [
        tool for tool in tools
        if tool.get("function", {}).get("name") in policy.allowed_tools
    ]
