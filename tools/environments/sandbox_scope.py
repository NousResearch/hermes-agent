"""Which sandbox an operation may address, derived from the bound scope alone.

A hosted deployment serves many tenants' agents from one process, and every agent's commands,
code, background processes and files live in that agent's own remote sandbox. The sandbox an
operation reaches is therefore decided here, from the bound :class:`hermes_scope.Scope`'s
``(tenant, agent)``, and from nothing a caller hands in: not a tool argument, not a task id, not
a profile path. Two calls for the same agent share one sandbox; a call for another agent can never
be routed to it.

The guard sits in core, not in the backend: :class:`ScopedEnvironment` wraps every environment a
hosted backend creates, checks the bound scope against the sandbox it was created for before each
operation, and hands the backend that scope with the call. A handle that names another agent (an
environment object, a long-lived process reference) is refused with
:class:`CrossAgentHandleRefused`, and so is any operation with no scope bound.

A local install (no scope, or a scope whose tenant is ``hermes_scope.LOCAL_TENANT``) is untouched:
:func:`hosted_scope` is ``None`` there and every caller keeps today's behaviour.
"""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass, field, replace
from typing import IO, Any, Callable, Optional
from urllib.parse import quote

from hermes_scope import Scope, current_scope


class CrossAgentHandleRefused(PermissionError):
    """An operation addressed a sandbox that is not the bound agent's, or ran with no hosted scope."""


class HostedSandboxRequired(RuntimeError):
    """A hosted scope asked for a backend that would run on this host (a built-in), or for a plugin
    backend that does not declare it serves hosted scopes."""


@dataclass(frozen=True)
class SandboxIdentity:
    """The one sandbox an agent owns: ``(tenant, agent)`` and nothing else."""

    tenant: str
    agent: str

    @property
    def key(self) -> str:
        """Environment-cache key. Each part is percent-quoted, so no two identities share a key."""
        return f"sandbox:{quote(self.tenant, safe='')}/{quote(self.agent, safe='')}"


@dataclass(frozen=True)
class ProcessRef:
    """A long-lived process inside a sandbox: the sandbox it was started in and its name there."""

    sandbox: str
    name: str


@dataclass(frozen=True)
class ProcessStream:
    """A long-lived process with both stdio pipes open (bytes). ``poll`` returns the exit status, or
    ``None`` while it runs."""

    ref: ProcessRef
    stdin: IO[bytes]
    stdout: IO[bytes]
    poll: Callable[[], Optional[int]] = field(compare=False)


@dataclass(frozen=True)
class CredentialRef:
    """A credential the sandbox resolves through a broker, by name. Hosted backends receive these,
    never secret values from the secret store; no broker exists yet, so core passes none."""

    name: str


def hosted_scope() -> Optional[Scope]:
    """The bound scope when it is hosted, else ``None`` (no scope, or a local install)."""
    scope = current_scope()
    return scope if scope is not None and not scope.is_local else None


def sandbox_identity(scope: Scope) -> SandboxIdentity:
    return SandboxIdentity(scope.tenant, scope.agent)


def current_sandbox_key() -> str:
    """The bound hosted agent's sandbox key, or ``""`` outside a hosted scope."""
    scope = hosted_scope()
    return sandbox_identity(scope).key if scope is not None else ""


def sandbox_visible(sandbox_key: str) -> bool:
    """Whether a record stamped with *sandbox_key* (``""`` for a local activity's) may be seen by the
    current activity: only by the agent whose sandbox it names. A local activity sees local records
    as today; a hosted agent sees only its own, and an activity with no hosted scope bound sees no
    hosted agent's (it binds that agent's scope first). Anything else is indistinguishable from a
    missing record."""
    return sandbox_key == current_sandbox_key()


def require_identity(identity: SandboxIdentity, what: str) -> Scope:
    """The bound scope, when it is hosted and names *identity*'s agent; else refuse."""
    scope = hosted_scope()
    if scope is None:
        raise CrossAgentHandleRefused(f"{what} on a hosted sandbox needs a hosted scope bound")
    if sandbox_identity(scope) != identity:
        raise CrossAgentHandleRefused(f"{what} names another agent's sandbox; refused")
    return scope


def admit_env(env: Any) -> Any:
    """*env*, when the current activity may use it: anything outside a hosted scope; inside one,
    only the bound agent's own :class:`ScopedEnvironment`. ``None`` passes through."""
    scope = hosted_scope()
    if env is None or scope is None:
        return env
    if not isinstance(env, ScopedEnvironment) or env.sandbox != sandbox_identity(scope):
        raise CrossAgentHandleRefused("a cached environment that is not this agent's sandbox was reached; refused")
    return env


def create_hosted_environment(env_type: str, provider: Any, **kwargs: Any) -> ScopedEnvironment:
    """Create the bound hosted agent's sandbox through *provider*, behind the identity guard.

    The provider receives ``scope`` and ``sandbox`` (the :class:`SandboxIdentity`), and
    ``task_id`` is the identity's key: whatever task id the caller held is not passed on."""
    scope = hosted_scope()
    if scope is None:
        raise CrossAgentHandleRefused("a hosted sandbox can only be created inside a hosted scope")
    if provider is None or getattr(provider, "serves_hosted_scopes", False) is not True:
        raise HostedSandboxRequired(
            f"terminal.backend {env_type!r} cannot serve a hosted scope: tools for a hosted agent run only in "
            "a remote sandbox backend that declares serves_hosted_scopes")
    identity = sandbox_identity(scope)
    kwargs["task_id"] = identity.key
    inner = provider.create_environment(scope=scope, sandbox=identity, **kwargs)
    return ScopedEnvironment(inner, identity)


# Operations a hosted environment receives the bound scope with, as the ``scope`` keyword, when its
# method declares that parameter (optional plugin kwargs are signature-inspected, plugins/AGENTS.md).
_SCOPED_OPERATIONS = frozenset({"execute", "fetch_file", "fetch_realpath"})


@functools.lru_cache(maxsize=256)
def _declares_scope(owner: type, name: str) -> bool:
    try:
        return "scope" in inspect.signature(getattr(owner, name)).parameters
    except (TypeError, ValueError):
        return False
# Lifecycle calls the idle reaper makes with no scope bound: allowed then, refused from another agent.
_LIFECYCLE = frozenset({"cleanup", "stop_idle"})


class ScopedEnvironment:
    """A hosted backend's environment behind the identity guard.

    Every method call checks the bound scope first. The long-lived process calls, and the operations
    in ``_SCOPED_OPERATIONS`` whose method declares ``scope``, also pass that scope to the backend, so
    it can verify the sandbox it is about to touch. Attributes (``cwd``, ``env_type``, ...) read and write through to the backend's
    object. The file tools never take a host-filesystem fast path through it."""

    uses_host_filesystem = False
    is_local = False

    def __init__(self, inner: Any, identity: SandboxIdentity) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "sandbox", identity)

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._inner, name)
        if not callable(value):
            return value
        return self._lifecycle(value) if name in _LIFECYCLE else self._guarded(name, value)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._inner, name, value)

    def _guarded(self, name: str, method: Any) -> Any:
        @functools.wraps(method)
        def call(*args: Any, **kwargs: Any) -> Any:
            scope = require_identity(self.sandbox, name)
            if name in _SCOPED_OPERATIONS and _declares_scope(type(self._inner), name):
                kwargs["scope"] = scope
            return method(*args, **kwargs)
        return call

    def _lifecycle(self, method: Any) -> Any:
        @functools.wraps(method)
        def call(*args: Any, **kwargs: Any) -> Any:
            if current_scope() is not None:
                require_identity(self.sandbox, method.__name__)
            return method(*args, **kwargs)
        return call

    def start_process(self, name: str, argv: list[str], *, child_env: Optional[dict] = None) -> ProcessStream:
        """Start *argv* in this sandbox as the long-lived process *name*, stdio open both ways."""
        scope = require_identity(self.sandbox, "start_process")
        stream = self._inner.start_process(name, argv, child_env=child_env, scope=scope)
        return replace(stream, ref=ProcessRef(self.sandbox.key, name))

    def attach_process(self, ref: ProcessRef | str) -> Optional[ProcessStream]:
        """The running process *ref* started earlier in this sandbox, or ``None``."""
        name = self._own_name(ref, "attach_process")
        stream = self._inner.attach_process(name, scope=require_identity(self.sandbox, "attach_process"))
        return None if stream is None else replace(stream, ref=ProcessRef(self.sandbox.key, name))

    def stop_process(self, ref: ProcessRef | str) -> bool:
        """Stop *ref*; ``False`` when no such process was running."""
        name = self._own_name(ref, "stop_process")
        return self._inner.stop_process(name, scope=require_identity(self.sandbox, "stop_process"))

    def _own_name(self, ref: ProcessRef | str, what: str) -> str:
        if isinstance(ref, ProcessRef):
            if ref.sandbox != self.sandbox.key:
                raise CrossAgentHandleRefused(f"{what} names a process in another agent's sandbox; refused")
            return ref.name
        return ref

    def __repr__(self) -> str:
        return f"ScopedEnvironment({self._inner!r}, {self.sandbox.key})"
