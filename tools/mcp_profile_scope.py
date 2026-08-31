"""Explicit namespace policy for shared MCP memory services.

The MCP server may be physically shared by several Hermes profile backends.
This module keeps profile separation at the Hermes tool-call boundary instead
of relying on a prompt convention or on a caller-supplied namespace alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

_NAMESPACE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")
_PROFILE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_MODES = {"profile", "shared", "cross_profile"}
_DEFAULT_PREFIX = "hermes/profile"
_DEFAULT_SHARED = ("hermes/shared",)

# Tools that can be safely scoped by their namespace argument. Other semantic
# memory tools are deliberately not admitted through private/shared views.
_SEARCH_TOOLS = frozenset(
    {
        "sm_search",
        "sm_search_witnessed",
        "sm_search_proof_debt",
        "sm_search_as_of",
        "sm_benchmark_trust",
    }
)
_NAMESPACE_TOOLS = frozenset({"sm_list_facts", "sm_add_fact", "sm_ingest_document"})
# These expose records or state without a namespace argument. They are reserved
# for the explicitly enabled cross_profile view.
_DIRECT_READ_TOOLS = frozenset(
    {
        "sm_get_fact",
        "sm_get_fact_neighbors",
        "sm_graph_path",
        "sm_search_conversations",
        "sm_get_search_receipt",
    }
)


@dataclass(frozen=True)
class ProfileScope:
    mode: str
    profile: str
    private_namespace: str
    shared_namespaces: tuple[str, ...]
    namespace_prefix: str

    @property
    def read_only(self) -> bool:
        return self.mode in {"shared", "cross_profile"}

    @property
    def allowed_profile_prefix(self) -> str:
        return f"{self.namespace_prefix}/"


def _valid_namespace(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("namespace must be a string")
    value = value.strip()
    if not _NAMESPACE_RE.fullmatch(value) or ".." in value.split("/"):
        raise ValueError(f"invalid namespace {value!r}")
    return value


def resolve_profile_scope(
    config: Mapping[str, Any], profile: str | None = None
) -> ProfileScope | None:
    """Resolve a validated ``profile_scope`` config block.

    A missing block means the server is intentionally unscoped and preserves
    existing MCP behavior. A present but malformed block is an error: silently
    dropping a requested isolation policy would create a false safety claim.
    """

    raw = config.get("profile_scope")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("profile_scope must be a mapping")

    mode = str(raw.get("mode") or "").strip().lower()
    if mode not in _MODES:
        raise ValueError("profile_scope.mode must be profile, shared, or cross_profile")

    if profile is None:
        from hermes_cli.profiles import get_active_profile_name

        profile = get_active_profile_name()
    active_profile = str(profile or "").strip()
    if not _PROFILE_RE.fullmatch(active_profile):
        raise ValueError("profile_scope requires a valid active Hermes profile name")

    prefix = _valid_namespace(raw.get("namespace_prefix", _DEFAULT_PREFIX))
    shared_raw = raw.get("shared_namespaces", list(_DEFAULT_SHARED))
    if not isinstance(shared_raw, (list, tuple)) or not shared_raw:
        raise ValueError("profile_scope.shared_namespaces must be a non-empty list")
    shared = tuple(dict.fromkeys(_valid_namespace(item) for item in shared_raw))

    return ProfileScope(
        mode=mode,
        profile=active_profile,
        private_namespace=f"{prefix}/{active_profile}",
        shared_namespaces=shared,
        namespace_prefix=prefix,
    )


def _namespaces_from_args(args: Mapping[str, Any]) -> list[str] | None:
    value = args.get("namespaces")
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("namespaces must be a list")
    return [_valid_namespace(item) for item in value]


def _require_namespaces(scope: ProfileScope, args: Mapping[str, Any]) -> list[str]:
    supplied = _namespaces_from_args(args)
    if scope.mode == "profile":
        allowed = {scope.private_namespace}
        requested = [scope.private_namespace] if supplied is None else supplied
        if set(requested) != allowed:
            raise ValueError(
                f"private MCP view permits only namespace {scope.private_namespace!r}"
            )
        return requested
    if scope.mode == "shared":
        requested = list(scope.shared_namespaces) if supplied is None else supplied
        if any(item not in scope.shared_namespaces for item in requested):
            raise ValueError("shared MCP view permits only configured shared namespaces")
        return requested

    # Cross-profile access must name its namespaces explicitly. This prevents a
    # seemingly harmless search from becoming an all-profiles export.
    if not supplied:
        raise ValueError("cross_profile MCP searches require explicit namespaces")
    if any(not item.startswith(scope.allowed_profile_prefix) for item in supplied):
        raise ValueError(
            f"cross_profile MCP searches permit only namespaces below {scope.allowed_profile_prefix!r}"
        )
    return supplied


def scope_tool_arguments(
    config: Mapping[str, Any],
    tool_name: str,
    arguments: Mapping[str, Any] | None,
    profile: str | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Return scoped arguments or a fail-closed error message."""

    try:
        scope = resolve_profile_scope(config, profile)
        args = dict(arguments or {})
        if scope is None:
            return args, None

        if scope.mode in {"shared", "cross_profile"}:
            allowed = _SEARCH_TOOLS | _NAMESPACE_TOOLS | _DIRECT_READ_TOOLS
            if tool_name not in allowed or tool_name in {"sm_add_fact", "sm_ingest_document"}:
                return None, f"MCP tool {tool_name!r} is not available in {scope.mode} read-only view"
        elif tool_name not in _SEARCH_TOOLS | _NAMESPACE_TOOLS:
            return None, f"MCP tool {tool_name!r} is not available in private profile view"

        if tool_name in _SEARCH_TOOLS:
            args["namespaces"] = _require_namespaces(scope, args)
        elif tool_name in _NAMESPACE_TOOLS:
            if tool_name in {"sm_add_fact", "sm_ingest_document"} and scope.read_only:
                return None, f"MCP tool {tool_name!r} is write-capable and blocked in read-only view"
            if tool_name == "sm_list_facts":
                requested = args.get("namespace")
                if scope.mode == "profile":
                    expected = scope.private_namespace
                    if requested is None:
                        args["namespace"] = expected
                    elif _valid_namespace(requested) != expected:
                        raise ValueError(f"private MCP view permits only namespace {expected!r}")
                elif scope.mode == "shared":
                    if requested is None:
                        if len(scope.shared_namespaces) != 1:
                            raise ValueError("shared MCP view requires an explicit namespace")
                        args["namespace"] = scope.shared_namespaces[0]
                    elif _valid_namespace(requested) not in scope.shared_namespaces:
                        raise ValueError("shared MCP view permits only configured shared namespaces")
                else:
                    if requested is None or not _valid_namespace(requested).startswith(scope.allowed_profile_prefix):
                        raise ValueError("cross_profile MCP list_facts requires an explicit profile namespace")
                    args["namespace"] = _valid_namespace(requested)
            else:
                requested = args.get("namespace")
                if scope.mode == "profile":
                    expected = scope.private_namespace
                    if requested is None:
                        args["namespace"] = expected
                    elif _valid_namespace(requested) != expected:
                        raise ValueError(f"private MCP view permits only namespace {expected!r}")
                else:
                    return None, f"MCP tool {tool_name!r} is not available in {scope.mode} read-only view"

        return args, None
    except ValueError as exc:
        return None, str(exc)


__all__ = ["ProfileScope", "resolve_profile_scope", "scope_tool_arguments"]
