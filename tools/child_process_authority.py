"""Typed process-edge authority contracts for Hermes child processes.

The trusted Hermes process can hold several kinds of ambient authority at once:
profile credentials, gateway/controller markers, Kanban lifecycle ownership,
session identity, and executable-loader state.  Copying ``os.environ`` and then
subtracting names at each call site makes those authorities indistinguishable.

This module is the narrow waist for child environments.  A caller selects a
typed process intent; the policy decides which profile data, credential class,
control-plane grant, stdin contract, and descendant semantics may cross that
specific edge.  Environment values never appear in receipts.

The first migration deliberately keeps the mature terminal sanitizer in
``tools.environments.local`` as the implementation owner for model-authored
terminal execution.  This module owns every non-terminal full-environment or
credential-bearing child edge.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

POLICY_VERSION = 1


class ChildProcessIntent(str, Enum):
    """Security-relevant reason for creating a child process."""

    MODEL_DRIVER = "model_driver"
    TRUSTED_HERMES_CHILD = "trusted_hermes_child"
    INTERACTIVE_HERMES_PTY = "interactive_hermes_pty"
    VAULT_CLI = "vault_cli"
    SECRET_HELPER = "secret_helper"
    CHECKPOINT_GIT = "checkpoint_git"
    CONTAINER_CONTROL = "container_control"
    CONTAINER_IMAGE_BUILD = "container_image_build"
    PROBE = "probe"


class ChildPrincipal(str, Enum):
    """Principal expected to execute inside the child."""

    MODEL_DRIVING_CLI = "model_driving_cli"
    HERMES_RUNTIME = "hermes_runtime"
    USER_CONFIGURED_HELPER = "user_configured_helper"
    EXTERNAL_CREDENTIAL_TOOL = "external_credential_tool"
    LOCAL_INFRASTRUCTURE = "local_infrastructure"


class ChildStdinPolicy(str, Enum):
    CLOSED = "closed"
    PIPE = "pipe"
    PTY = "pty"


class DescendantPolicy(str, Enum):
    NO_AUTHORITY_DELEGATION = "no_authority_delegation"
    SAME_PROFILE_RUNTIME = "same_profile_runtime"
    TOOL_OWNED = "tool_owned"


@dataclass(frozen=True)
class ChildProcessSpec:
    """Immutable authority contract for one parent -> child edge."""

    intent: ChildProcessIntent
    principal: ChildPrincipal
    stdin: ChildStdinPolicy = ChildStdinPolicy.CLOSED
    descendants: DescendantPolicy = DescendantPolicy.NO_AUTHORITY_DELEGATION
    grants: tuple[str, ...] = ()
    grant_prefixes: tuple[str, ...] = ()
    target_profile: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        normalized = tuple(_normalize_name(name) for name in self.grants)
        prefixes = tuple(_normalize_name(prefix) for prefix in self.grant_prefixes)
        if any(not name for name in normalized):
            raise ValueError("child-process grants must be non-empty environment names")
        if any(not prefix for prefix in prefixes):
            raise ValueError("child-process grant prefixes must be non-empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("duplicate child-process grant")
        if len(set(prefixes)) != len(prefixes):
            raise ValueError("duplicate child-process grant prefix")


# Operational environment required to start ordinary local programs.  This is
# deliberately a positive baseline, not a snapshot of the live Hermes process.
_SAFE_BASE_KEYS = frozenset({
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "PWD",
    "OLDPWD",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
    "PROGRAMDATA",
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "TMPDIR",
    "TMP",
    "TEMP",
    "XDG_CONFIG_HOME",
    "XDG_CACHE_HOME",
    "XDG_RUNTIME_DIR",
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "TERM",
    "COLORTERM",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "PYTHONUTF8",
})

# Network routing is a capability too.  Vault and container-image clients may
# need enterprise proxies, while local checkpoint Git and version probes do not.
_NETWORK_ROUTE_KEYS = frozenset({
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
})

# Registry-auth exception inherited from the reviewed #77027 boundary.  No
# other process-global credential is admitted to an image build.
CONTAINER_REGISTRY_AUTH_KEYS = (
    "APPTAINER_DOCKER_USERNAME",
    "APPTAINER_DOCKER_PASSWORD",
    "SINGULARITY_DOCKER_USERNAME",
    "SINGULARITY_DOCKER_PASSWORD",
    "DOCKER_USERNAME",
    "DOCKER_PASSWORD",
)

BWS_VAULT_GRANTS = ("BWS_ACCESS_TOKEN", "BWS_SERVER_URL")
OP_VAULT_GRANTS = (
    "OP_SERVICE_ACCOUNT_TOKEN",
    "OP_ACCOUNT",
    "OP_CONNECT_HOST",
    "OP_CONNECT_TOKEN",
    "OP_LOAD_DESKTOP_APP_SETTINGS",
    "OP_CACHE",
)
OP_VAULT_GRANT_PREFIXES = ("OP_SESSION_",)

# Authority that may continue only across the dispatcher-worker -> Codex ->
# hermes-tools edge.  Arbitrary nested chats and other model drivers do not get
# this prefix.
KANBAN_WORKER_GRANTS = ("HERMES_PROFILE", "HERMES_SESSION_ID")
KANBAN_WORKER_GRANT_PREFIXES = ("HERMES_KANBAN_",)


# Tool credentials that a trusted Hermes execution child may need to perform the
# same tool calls as its parent.  Gateway transport/auth authority is
# deliberately absent.
TRUSTED_AGENT_TOOL_GRANTS = (
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "GITHUB_APP_ID",
    "GITHUB_APP_PRIVATE_KEY_PATH",
    "GITHUB_APP_INSTALLATION_ID",
    "HASS_TOKEN",
    "EMAIL_PASSWORD",
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
    "DAYTONA_API_KEY",
    *BWS_VAULT_GRANTS,
    *OP_VAULT_GRANTS,
)
TRUSTED_AGENT_TOOL_GRANT_PREFIXES = (
    "AUXILIARY_",
    "_HERMES_FORCE_",
    *OP_VAULT_GRANT_PREFIXES,
)

_CONTROL_PLANE_EXACT = frozenset({
    "_HERMES_GATEWAY",
    "HERMES_DELEGATED_CHILD_CONTEXT",
    "HERMES_DASHBOARD_SESSION_TOKEN",
})
_CONTROL_PLANE_PREFIXES = (
    "HERMES_KANBAN_",
    "HERMES_SESSION_",
)

_FORWARDED_ENV_PREFIXES = ("APPTAINERENV_", "SINGULARITYENV_")
_SECRET_SUFFIXES = (
    "_API_KEY",
    "_TOKEN",
    "_SECRET",
    "_PASSWORD",
    "_PRIVATE_KEY",
    "_CLIENT_SECRET",
)

# Mutable coordinates are explicit policy too.  These are values that reviewed
# call sites need to select a child-owned home, lifecycle cadence, vault output
# mode, or container cache location.  They are not a generic escape hatch for
# arbitrary caller state.
_OVERRIDE_COORDINATES: dict[ChildProcessIntent, frozenset[str]] = {
    ChildProcessIntent.TRUSTED_HERMES_CHILD: frozenset({
        "HERMES_HOME",
        "HERMES_COMPUTE_HOST_HEARTBEAT_SECS",
    }),
    ChildProcessIntent.INTERACTIVE_HERMES_PTY: frozenset({"HERMES_HOME"}),
    ChildProcessIntent.VAULT_CLI: frozenset({"NO_COLOR"}),
    ChildProcessIntent.CONTAINER_IMAGE_BUILD: frozenset({
        "APPTAINER_TMPDIR",
        "APPTAINER_CACHEDIR",
        "SINGULARITY_TMPDIR",
        "SINGULARITY_CACHEDIR",
    }),
}
_NETWORK_OVERRIDE_INTENTS = frozenset({
    ChildProcessIntent.MODEL_DRIVER,
    ChildProcessIntent.VAULT_CLI,
    ChildProcessIntent.CONTAINER_IMAGE_BUILD,
})


def _normalize_name(name: str) -> str:
    return str(name or "").strip().upper()


def _effective_env_name(name: str) -> str:
    normalized = _normalize_name(name)
    changed = True
    while changed:
        changed = False
        for prefix in _FORWARDED_ENV_PREFIXES:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                changed = True
    return normalized


def _casefold_lookup(source: Mapping[str, str], name: str) -> str | None:
    wanted = _normalize_name(name)
    for key, value in source.items():
        if _normalize_name(key) == wanted:
            return str(value)
    return None


def _grant_allows(spec: ChildProcessSpec, name: str) -> bool:
    effective = _effective_env_name(name)
    grants = {_normalize_name(item) for item in spec.grants}
    if effective in grants:
        return True
    return any(
        effective.startswith(_normalize_name(prefix)) for prefix in spec.grant_prefixes
    )


def _is_control_plane_authority(name: str) -> bool:
    effective = _effective_env_name(name)
    if effective in _CONTROL_PLANE_EXACT:
        return True
    return any(effective.startswith(prefix) for prefix in _CONTROL_PLANE_PREFIXES)


def _is_tier1_or_internal(name: str) -> bool:
    from tools.environments.local import (
        _ALWAYS_STRIP_KEYS,
        _HERMES_PROVIDER_ENV_FORCE_PREFIX,
        _is_hermes_internal_secret,
    )

    effective = _effective_env_name(name)
    if effective.startswith(_normalize_name(_HERMES_PROVIDER_ENV_FORCE_PREFIX)):
        return True
    if effective in {_normalize_name(key) for key in _ALWAYS_STRIP_KEYS}:
        return True
    return bool(_is_hermes_internal_secret(effective))


def _provider_env_names() -> frozenset[str]:
    """Return exact model-auth names from the provider registry.

    This intentionally does not reuse the broad terminal blocklist: that set
    also contains messaging and tool credentials.  A model-driving CLI receives
    model authentication, not every integration secret in the active profile.
    """

    names: set[str] = {
        "CLAUDE_CODE_OAUTH_TOKEN",
        "COPILOT_GITHUB_TOKEN",
        # General AWS chain used by Bedrock-backed model drivers.
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_CONFIG_FILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_ROLE_ARN",
        "AWS_ROLE_SESSION_NAME",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_EC2_METADATA_DISABLED",
    }
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY

        for provider in PROVIDER_REGISTRY.values():
            names.update(_normalize_name(item) for item in provider.api_key_env_vars)
            if provider.base_url_env_var:
                names.add(_normalize_name(provider.base_url_env_var))
    except Exception:
        # Fail closed: the caller may still declare an exact grant.
        pass
    return frozenset(name for name in names if name)


def _profile_source(
    explicit: Mapping[str, str] | None,
) -> dict[str, str]:
    """Return the current profile environment without cross-profile fallback."""

    if explicit is not None:
        return {str(key): str(value) for key, value in explicit.items()}

    try:
        from agent.secret_scope import (
            _is_global_env,
            current_secret_scope,
            is_multiplex_active,
        )

        scope = current_secret_scope()
        if scope is not None:
            source = {
                str(key): str(value)
                for key, value in os.environ.items()
                if _is_global_env(str(key))
            }
            source.update({str(key): str(value) for key, value in scope.items()})
            return source
        if is_multiplex_active():
            raise RuntimeError(
                "child-process authority requested with no active profile "
                "secret scope while multiplexing is enabled"
            )
    except ImportError:
        pass

    return {str(key): str(value) for key, value in os.environ.items()}


def _copy_named(
    source: Mapping[str, str],
    names: set[str] | frozenset[str] | tuple[str, ...],
) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in names:
        value = _casefold_lookup(source, str(name))
        if value is not None:
            out[str(name)] = value
    return out


def _copy_prefixes(
    source: Mapping[str, str],
    prefixes: tuple[str, ...],
) -> dict[str, str]:
    normalized = tuple(_normalize_name(prefix) for prefix in prefixes)
    return {
        str(key): str(value)
        for key, value in source.items()
        if any(_normalize_name(key).startswith(prefix) for prefix in normalized)
    }


def _safe_base(
    source: Mapping[str, str],
    *,
    network: bool = False,
) -> dict[str, str]:
    names = set(_SAFE_BASE_KEYS)
    if network:
        names.update(_NETWORK_ROUTE_KEYS)
    return _copy_named(source, names)


def _strip_ungranted_authority(
    env: dict[str, str],
    spec: ChildProcessSpec,
    *,
    strip_provider_credentials: bool,
) -> dict[str, str]:
    provider_names = _provider_env_names()
    for key in list(env):
        effective = _effective_env_name(key)
        granted = _grant_allows(spec, effective)
        if _is_tier1_or_internal(effective) and not granted:
            env.pop(key, None)
            continue
        if _is_control_plane_authority(effective) and not granted:
            env.pop(key, None)
            continue
        if strip_provider_credentials and effective in provider_names and not granted:
            env.pop(key, None)
            continue
        if strip_provider_credentials and _secret_like(effective) and not granted:
            env.pop(key, None)
    return env


def _secret_like(name: str) -> bool:
    effective = _effective_env_name(name)
    if _is_tier1_or_internal(effective) or _is_control_plane_authority(effective):
        return True
    return effective.endswith(_SECRET_SUFFIXES)


def _apply_overrides(
    env: dict[str, str],
    overrides: Mapping[str, str] | None,
    spec: ChildProcessSpec,
) -> None:
    """Apply only names positively admitted by the typed edge policy.

    Forwarding wrappers are separate authority channels and therefore require an
    explicit spec grant.  Safe-baseline membership controls inheritance only;
    caller writes require an intent-owned routing/network coordinate,
    model-provider authority, or an exact/prefix grant.
    """

    if not overrides:
        return

    provider_names = _provider_env_names()
    allowed_names: set[str] = set()
    if spec.intent in _NETWORK_OVERRIDE_INTENTS:
        allowed_names.update(_normalize_name(name) for name in _NETWORK_ROUTE_KEYS)
    allowed_names.update(
        _normalize_name(name) for name in _OVERRIDE_COORDINATES.get(spec.intent, ())
    )
    if spec.intent is ChildProcessIntent.MODEL_DRIVER:
        allowed_names.update(provider_names)

    for key, value in overrides.items():
        name = str(key)
        normalized = _normalize_name(name)
        effective = _effective_env_name(name)
        if _grant_allows(spec, effective):
            env[name] = str(value)
            continue
        if normalized != effective:
            # A container forwarding wrapper can only carry explicitly granted
            # authority; baseline/coordinate admission is for this child only.
            continue
        if _is_tier1_or_internal(effective) or _is_control_plane_authority(effective):
            continue
        if effective in allowed_names:
            env[name] = str(value)


def build_child_process_env(
    spec: ChildProcessSpec,
    *,
    source_env: Mapping[str, str] | None = None,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Construct the environment authorized by ``spec``.

    The returned mapping is new and detached from every source mapping.
    Caller overrides are evaluated by the same policy; they cannot re-add a
    stripped gateway token, lifecycle marker, wrapper tunnel, or dynamic Hermes
    secret unless the edge declares an exact grant.
    """

    source = _profile_source(source_env)
    trusted_profile_child = spec.intent in {
        ChildProcessIntent.TRUSTED_HERMES_CHILD,
        ChildProcessIntent.INTERACTIVE_HERMES_PTY,
        ChildProcessIntent.SECRET_HELPER,
    }
    model_driver = spec.intent is ChildProcessIntent.MODEL_DRIVER

    if trusted_profile_child:
        env = dict(source)
        env = _strip_ungranted_authority(
            env,
            spec,
            strip_provider_credentials=False,
        )
    elif model_driver:
        env = _safe_base(source, network=True)
        env.update(_copy_named(source, _provider_env_names()))
        env.update(_copy_named(source, spec.grants))
        env.update(_copy_prefixes(source, spec.grant_prefixes))
        env = _strip_ungranted_authority(
            env,
            spec,
            strip_provider_credentials=False,
        )
    elif spec.intent is ChildProcessIntent.VAULT_CLI:
        env = _safe_base(source, network=True)
        env.update(_copy_named(source, spec.grants))
        env.update(_copy_prefixes(source, spec.grant_prefixes))
        env = _strip_ungranted_authority(
            env,
            spec,
            strip_provider_credentials=True,
        )
    elif spec.intent is ChildProcessIntent.CHECKPOINT_GIT:
        env = _safe_base(source, network=False)
        env = _strip_ungranted_authority(
            env,
            spec,
            strip_provider_credentials=True,
        )
    elif spec.intent is ChildProcessIntent.CONTAINER_IMAGE_BUILD:
        from tools.environments.local import build_subprocess_env

        env = build_subprocess_env(base=source)
        env.update(_copy_named(source, CONTAINER_REGISTRY_AUTH_KEYS))
        env = _strip_ungranted_authority(
            env,
            spec,
            strip_provider_credentials=True,
        )
    elif spec.intent is ChildProcessIntent.CONTAINER_CONTROL:
        from tools.environments.local import build_subprocess_env

        env = build_subprocess_env(base=source)
        env = _strip_ungranted_authority(
            env,
            spec,
            strip_provider_credentials=True,
        )
    elif spec.intent is ChildProcessIntent.PROBE:
        env = _safe_base(source, network=False)
        env = _strip_ungranted_authority(
            env,
            spec,
            strip_provider_credentials=True,
        )
    else:  # pragma: no cover - Enum exhaustiveness guard
        raise ValueError(f"unsupported child-process intent: {spec.intent}")

    # Preserve the existing context-local HERMES_HOME / subprocess HOME
    # contract without re-snapshotting os.environ. Apply caller overrides only
    # after this bridge so an explicit target-profile HERMES_HOME remains the
    # authoritative child coordinate.
    try:
        from tools.environments.local import _inject_context_hermes_home

        _inject_context_hermes_home(env)
        from hermes_constants import apply_subprocess_home_env

        apply_subprocess_home_env(env)
    except Exception:
        pass

    _apply_overrides(env, overrides, spec)
    env = _strip_ungranted_authority(
        env,
        spec,
        strip_provider_credentials=not (trusted_profile_child or model_driver),
    )

    if spec.target_profile:
        env["HERMES_PROFILE"] = spec.target_profile

    env.setdefault("PYTHONUTF8", "1")
    return env


def stdin_for_spec(spec: ChildProcessSpec) -> Any:
    """Translate the typed stdin policy to the subprocess API value."""

    if spec.stdin is ChildStdinPolicy.CLOSED:
        return subprocess.DEVNULL
    if spec.stdin is ChildStdinPolicy.PIPE:
        return subprocess.PIPE
    return None


def model_driver_spec(
    *,
    source: str,
    grants: tuple[str, ...] = (),
    grant_prefixes: tuple[str, ...] = (),
) -> ChildProcessSpec:
    return ChildProcessSpec(
        intent=ChildProcessIntent.MODEL_DRIVER,
        principal=ChildPrincipal.MODEL_DRIVING_CLI,
        stdin=ChildStdinPolicy.PIPE,
        descendants=DescendantPolicy.NO_AUTHORITY_DELEGATION,
        grants=grants,
        grant_prefixes=grant_prefixes,
        source=source,
    )


def trusted_hermes_child_spec(
    *,
    source: str,
    stdin: ChildStdinPolicy = ChildStdinPolicy.PIPE,
    grants: tuple[str, ...] = TRUSTED_AGENT_TOOL_GRANTS,
    grant_prefixes: tuple[str, ...] = TRUSTED_AGENT_TOOL_GRANT_PREFIXES,
) -> ChildProcessSpec:
    return ChildProcessSpec(
        intent=ChildProcessIntent.TRUSTED_HERMES_CHILD,
        principal=ChildPrincipal.HERMES_RUNTIME,
        stdin=stdin,
        descendants=DescendantPolicy.SAME_PROFILE_RUNTIME,
        grants=grants,
        grant_prefixes=grant_prefixes,
        source=source,
    )


def interactive_hermes_pty_spec(*, source: str) -> ChildProcessSpec:
    return ChildProcessSpec(
        intent=ChildProcessIntent.INTERACTIVE_HERMES_PTY,
        principal=ChildPrincipal.HERMES_RUNTIME,
        stdin=ChildStdinPolicy.PTY,
        descendants=DescendantPolicy.SAME_PROFILE_RUNTIME,
        grants=TRUSTED_AGENT_TOOL_GRANTS,
        grant_prefixes=TRUSTED_AGENT_TOOL_GRANT_PREFIXES,
        source=source,
    )


def bws_vault_spec(*, source: str) -> ChildProcessSpec:
    return ChildProcessSpec(
        intent=ChildProcessIntent.VAULT_CLI,
        principal=ChildPrincipal.EXTERNAL_CREDENTIAL_TOOL,
        stdin=ChildStdinPolicy.CLOSED,
        descendants=DescendantPolicy.TOOL_OWNED,
        grants=BWS_VAULT_GRANTS,
        source=source,
    )


def op_vault_spec(*, source: str) -> ChildProcessSpec:
    return ChildProcessSpec(
        intent=ChildProcessIntent.VAULT_CLI,
        principal=ChildPrincipal.EXTERNAL_CREDENTIAL_TOOL,
        stdin=ChildStdinPolicy.CLOSED,
        descendants=DescendantPolicy.TOOL_OWNED,
        grants=OP_VAULT_GRANTS,
        grant_prefixes=OP_VAULT_GRANT_PREFIXES,
        source=source,
    )


def secret_helper_spec(*, source: str) -> ChildProcessSpec:
    return ChildProcessSpec(
        intent=ChildProcessIntent.SECRET_HELPER,
        principal=ChildPrincipal.USER_CONFIGURED_HELPER,
        stdin=ChildStdinPolicy.CLOSED,
        descendants=DescendantPolicy.TOOL_OWNED,
        grants=("HERMES_SECRET_KEY", *TRUSTED_AGENT_TOOL_GRANTS),
        grant_prefixes=TRUSTED_AGENT_TOOL_GRANT_PREFIXES,
        source=source,
    )


def checkpoint_git_spec(*, source: str) -> ChildProcessSpec:
    return ChildProcessSpec(
        intent=ChildProcessIntent.CHECKPOINT_GIT,
        principal=ChildPrincipal.LOCAL_INFRASTRUCTURE,
        stdin=ChildStdinPolicy.CLOSED,
        descendants=DescendantPolicy.NO_AUTHORITY_DELEGATION,
        grants=(
            "GIT_DIR",
            "GIT_WORK_TREE",
            "GIT_INDEX_FILE",
            "GIT_CONFIG_GLOBAL",
            "GIT_CONFIG_SYSTEM",
            "GIT_CONFIG_NOSYSTEM",
        ),
        source=source,
    )


def container_control_spec(*, source: str) -> ChildProcessSpec:
    return ChildProcessSpec(
        intent=ChildProcessIntent.CONTAINER_CONTROL,
        principal=ChildPrincipal.LOCAL_INFRASTRUCTURE,
        stdin=ChildStdinPolicy.CLOSED,
        descendants=DescendantPolicy.TOOL_OWNED,
        source=source,
    )


def container_image_build_spec(*, source: str) -> ChildProcessSpec:
    return ChildProcessSpec(
        intent=ChildProcessIntent.CONTAINER_IMAGE_BUILD,
        principal=ChildPrincipal.LOCAL_INFRASTRUCTURE,
        stdin=ChildStdinPolicy.CLOSED,
        descendants=DescendantPolicy.TOOL_OWNED,
        grants=CONTAINER_REGISTRY_AUTH_KEYS,
        source=source,
    )


def probe_spec(*, source: str) -> ChildProcessSpec:
    return ChildProcessSpec(
        intent=ChildProcessIntent.PROBE,
        principal=ChildPrincipal.LOCAL_INFRASTRUCTURE,
        stdin=ChildStdinPolicy.CLOSED,
        descendants=DescendantPolicy.NO_AUTHORITY_DELEGATION,
        source=source,
    )


def _policy_hash() -> str:
    default_specs = (
        model_driver_spec(source=""),
        trusted_hermes_child_spec(source=""),
        interactive_hermes_pty_spec(source=""),
        bws_vault_spec(source=""),
        op_vault_spec(source=""),
        secret_helper_spec(source=""),
        checkpoint_git_spec(source=""),
        container_control_spec(source=""),
        container_image_build_spec(source=""),
        probe_spec(source=""),
    )
    payload = {
        "version": POLICY_VERSION,
        "contracts": [
            {
                "intent": spec.intent.value,
                "principal": spec.principal.value,
                "stdin": spec.stdin.value,
                "descendants": spec.descendants.value,
                "grants": sorted(_normalize_name(name) for name in spec.grants),
                "grant_prefixes": sorted(
                    _normalize_name(prefix) for prefix in spec.grant_prefixes
                ),
            }
            for spec in default_specs
        ],
        "safe_base": sorted(_SAFE_BASE_KEYS),
        "network": sorted(_NETWORK_ROUTE_KEYS),
        "network_override_intents": sorted(
            intent.value for intent in _NETWORK_OVERRIDE_INTENTS
        ),
        "override_coordinates": {
            intent.value: sorted(_normalize_name(name) for name in names)
            for intent, names in sorted(
                _OVERRIDE_COORDINATES.items(), key=lambda item: item[0].value
            )
        },
        "provider_env_names": sorted(_provider_env_names()),
        "control_exact": sorted(_CONTROL_PLANE_EXACT),
        "control_prefixes": sorted(_CONTROL_PLANE_PREFIXES),
        "forwarded_prefixes": sorted(_FORWARDED_ENV_PREFIXES),
        "secret_suffixes": sorted(_SECRET_SUFFIXES),
        "registry_auth": sorted(CONTAINER_REGISTRY_AUTH_KEYS),
        "kanban_grants": sorted(KANBAN_WORKER_GRANTS),
        "kanban_grant_prefixes": sorted(KANBAN_WORKER_GRANT_PREFIXES),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_spawn_receipt(
    spec: ChildProcessSpec,
    *,
    argv: list[str] | tuple[str, ...],
    env: Mapping[str, str],
) -> dict[str, Any]:
    """Return a value-free, serializable receipt for one spawn decision."""

    return {
        "policy_version": POLICY_VERSION,
        "policy_sha256": _policy_hash(),
        "intent": spec.intent.value,
        "principal": spec.principal.value,
        "stdin": spec.stdin.value,
        "descendants": spec.descendants.value,
        "target_profile": spec.target_profile or None,
        "source": spec.source or None,
        "executable": str(argv[0]) if argv else "",
        "argv_count": len(argv),
        "environment_keys": sorted(str(key) for key in env),
        "grants": sorted(_normalize_name(name) for name in spec.grants),
        "grant_prefixes": sorted(
            _normalize_name(prefix) for prefix in spec.grant_prefixes
        ),
    }
