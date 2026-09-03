"""Central registry for all hermes-agent tools.

Each tool file calls ``registry.register()`` at module level to declare its
schema, handler, toolset membership, and availability check.  ``model_tools.py``
queries the registry instead of maintaining its own parallel data structures.

Import chain (circular-import safe):
    tools/registry.py  (no imports from model_tools or tool files)
           ^
    tools/*.py  (import from tools.registry at module level)
           ^
    model_tools.py  (imports tools.registry + all tool modules)
           ^
    run_agent.py, cli.py, batch_runner.py, etc.
"""

import ast
import functools
import importlib
import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

from hermes_constants import hermes_home_key

logger = logging.getLogger(__name__)

# Cap on a tool error body; only trims runaway interpolated exceptions (static msgs are ~115 chars).
_MAX_TOOL_ERROR_CHARS = 2048
_TOOL_ERROR_TRUNCATION_MARKER = "… [truncated]"
# Logs keep more of the body than the model sees, but still a bounded amount.
_MAX_LOGGED_ERROR_CHARS = 8192


def _bound_error_text(text: str) -> str:
    """Bound an error body destined for model context; logs keep a longer prefix."""
    if len(text) <= _MAX_TOOL_ERROR_CHARS:
        return text
    logger.debug(
        "tool error body truncated for context (%d chars): %s",
        len(text),
        text[:_MAX_LOGGED_ERROR_CHARS],
    )
    return text[:_MAX_TOOL_ERROR_CHARS] + _TOOL_ERROR_TRUNCATION_MARKER


def _bound_json_error_result(result: str) -> str:
    """Trim an oversized ``error`` field in a JSON string result.

    Handlers that serialize exceptions directly — ``json.dumps({"error":
    str(exc), ...})`` instead of ``tool_error()`` — bypass the cap in
    ``tool_error``. Applied at the dispatch boundary so no registered tool
    can return an unbounded error body that stacks across retries.
    """
    if len(result) <= _MAX_TOOL_ERROR_CHARS or '"error"' not in result:
        return result
    try:
        payload = json.loads(result)
    except ValueError:
        return result
    if not isinstance(payload, dict):
        return result
    error = payload.get("error")
    if not isinstance(error, str) or len(error) <= _MAX_TOOL_ERROR_CHARS:
        return result
    payload["error"] = _bound_error_text(error)
    return json.dumps(payload, ensure_ascii=False)


def _is_registry_register_call(node: ast.AST) -> bool:
    """Return True when *node* is a ``registry.register(...)`` call expression."""
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    func = node.value.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "register"
        and isinstance(func.value, ast.Name)
        and func.value.id == "registry"
    )


def _module_registers_tools(module_path: Path) -> bool:
    """Return True when the module contains a top-level ``registry.register(...)`` call.

    Only inspects module-body statements so that helper modules which happen
    to call ``registry.register()`` inside a function are not picked up.

    A cheap text prefilter avoids the ``ast.parse`` cost for files that do not
    mention both ``registry`` and ``register`` — a necessary condition for a
    top-level ``registry.register()`` call to exist.
    """
    try:
        source = module_path.read_text(encoding="utf-8")
    except OSError:
        return False
    if "registry" not in source or "register" not in source:
        return False
    try:
        tree = ast.parse(source, filename=str(module_path))
    except SyntaxError:
        return False

    return any(_is_registry_register_call(stmt) for stmt in tree.body)


def discover_builtin_tools(tools_dir: Optional[Path] = None) -> List[str]:
    """Import built-in self-registering tool modules and return their module names.

    The per-file AST scan (:func:`_module_registers_tools`) costs ~145 ms over
    ~100 files on a warm cache, so verdicts are memoized on disk keyed by
    ``(mtime_ns, size)``. A file whose mtime_ns+size match the cached entry is
    trusted without re-reading; any mismatch (or a corrupt/missing cache file)
    falls back to a fresh scan for that file. The cache write is best-effort
    and atomic, so concurrent processes can race harmlessly.
    """
    tools_path = Path(tools_dir) if tools_dir is not None else Path(__file__).resolve().parent

    cache = _load_discovery_cache()
    fresh_cache: Dict[str, list] = {}
    cache_dirty = False

    module_names: List[str] = []
    for path in sorted(tools_path.glob("*.py")):
        if path.name in {"__init__.py", "registry.py", "mcp_tool.py"}:
            continue
        abs_path = str(path.resolve())
        try:
            st = path.stat()
            stat_key = (st.st_mtime_ns, st.st_size)
        except OSError:
            continue
        cached = cache.get(abs_path)
        if (
            isinstance(cached, (list, tuple))
            and len(cached) == 3
            and (cached[0], cached[1]) == stat_key
        ):
            registers = bool(cached[2])
        else:
            registers = _module_registers_tools(path)
            cache_dirty = True
        fresh_cache[abs_path] = [stat_key[0], stat_key[1], registers]
        if registers:
            module_names.append(f"tools.{path.stem}")

    # Drop entries for files that no longer exist; rewrite only when changed.
    if cache_dirty or set(fresh_cache) != set(cache):
        _save_discovery_cache(fresh_cache)

    imported: List[str] = []
    for mod_name in module_names:
        try:
            importlib.import_module(mod_name)
            imported.append(mod_name)
        except Exception as e:
            logger.warning("Could not import tool module %s: %s", mod_name, e)
    return imported


def _discovery_cache_path() -> Optional[Path]:
    """Path of the tool-discovery verdict cache, or None if unresolvable."""
    try:
        # Deferred import keeps tools/registry.py a no-deps leaf at module
        # import time (hermes_constants itself is stdlib-only, so no cycle).
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home()) / "cache" / "tool_discovery_cache.json"
    except Exception:
        return None


def _load_discovery_cache() -> Dict[str, list]:
    """Read the discovery cache; any error → empty dict (full scan)."""
    path = _discovery_cache_path()
    if path is None:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_discovery_cache(cache: Dict[str, list]) -> None:
    """Best-effort atomic write of the discovery cache. Never raises."""
    path = _discovery_cache_path()
    if path is None:
        return
    try:
        from utils import atomic_json_write  # stdlib+yaml only; no cycle

        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(path, cache, indent=0)
    except Exception as e:
        logger.debug("Could not write tool discovery cache %s: %s", path, e)


class ToolEntry:
    """Metadata for a single registered tool."""

    __slots__ = (
        "name", "toolset", "schema", "handler", "check_fn",
        "requires_env", "is_async", "description", "emoji",
        "max_result_size_chars", "dynamic_schema_overrides",
    )

    def __init__(self, name, toolset, schema, handler, check_fn,
                 requires_env, is_async, description, emoji,
                 max_result_size_chars=None, dynamic_schema_overrides=None):
        self.name = name
        self.toolset = toolset
        self.schema = schema
        self.handler = handler
        self.check_fn = check_fn
        self.requires_env = requires_env
        self.is_async = is_async
        self.description = description
        self.emoji = emoji
        self.max_result_size_chars = max_result_size_chars
        # Optional zero-arg callable returning a dict of schema overrides
        # applied at get_definitions() time. Use for fields that depend on
        # runtime config (e.g. delegate_task's description must reflect the
        # user's current delegation.max_concurrent_children / max_spawn_depth
        # so the model isn't told the wrong limits). The callable is invoked
        # on every get_definitions() call; results are merged shallow on top
        # of the base schema before the {"type": "function", ...} wrap.
        self.dynamic_schema_overrides = dynamic_schema_overrides


class _PluginOverridePolicy:
    """Identity-bearing authorization record for one plugin generation."""

    __slots__ = ("allowed",)

    def __init__(self, allowed: bool) -> None:
        self.allowed = bool(allowed)


# ---------------------------------------------------------------------------
# check_fn TTL cache
#
# external state (Docker daemon, Modal SDK install, playwright binary
# availability). For a long-lived CLI or gateway process, calling them on
# every get_definitions() is pure waste — external state changes on human
# timescales. Cache results for ~30 s so env-var flips via ``hermes tools``
# or live credential file changes propagate within a turn or two without
# requiring any explicit invalidation.
#
#
# Transient-failure suppression (issue #21658 / #5304): these probes can flap.
# A single ``subprocess.run([docker, "version"], timeout=5)`` that times out
# under load returns False for one call, which would silently strip the entire
# terminal+file toolset from whatever agent is being built at that instant —
# most visibly a delegate_task subagent, which then reports "Tool read_file
# does not exist". To absorb such flakes WITHOUT pinning a permanently-stale
# "available" verdict, we remember the last time each check returned True and,
# when a fresh probe fails within a short grace window of that last success,
# we serve the last-good True instead of caching the failure. A failure that
# persists past the grace window is honored normally, so a backend that really
# went down stops advertising its tools.
# ---------------------------------------------------------------------------

_CHECK_FN_TTL_SECONDS = 30.0
# How long after a successful check a subsequent transient failure is treated
# as a flake (last-good True is served) rather than a real outage. Kept short
# so a genuinely-down backend is reflected within a couple of turns.
_CHECK_FN_FAILURE_GRACE_SECONDS = 60.0
_CHECK_FN_CACHE_MAX = 512
_check_fn_cache: Dict[tuple[Callable, Optional[str]], tuple[float, bool]] = {}
_check_fn_last_good: Dict[tuple[Callable, Optional[str]], float] = {}
_check_fn_cache_lock = threading.Lock()
CHECK_FN_CACHE_BYPASS = ""
_NO_CACHE_CHECK_FNS: Set[Callable] = set()


def no_cache_check_fn(fn: Callable) -> Callable:
    """Mark a local, config-backed availability check as uncached."""
    _NO_CACHE_CHECK_FNS.add(fn)
    return fn


def _prune_check_fn_caches(now: float) -> None:
    """Expire stale entries and cap profile-dimensional cache growth.

    Caller must hold ``_check_fn_cache_lock``.
    """
    for key, (timestamp, _) in list(_check_fn_cache.items()):
        if now - timestamp >= _CHECK_FN_TTL_SECONDS:
            _check_fn_cache.pop(key, None)
    for key, timestamp in list(_check_fn_last_good.items()):
        if now - timestamp >= _CHECK_FN_FAILURE_GRACE_SECONDS:
            _check_fn_last_good.pop(key, None)
    while len(_check_fn_cache) >= _CHECK_FN_CACHE_MAX:
        _check_fn_cache.pop(next(iter(_check_fn_cache)))
    while len(_check_fn_last_good) >= _CHECK_FN_CACHE_MAX:
        _check_fn_last_good.pop(next(iter(_check_fn_last_good)))


def check_fn_cache_scope() -> Optional[str]:
    """Return the active profile key when availability is profile-scoped.

    Browser-controller availability is request-bound and can change on every
    attach/detach. A fully bound browser-control request therefore bypasses both
    this check cache and model_tools' outer definition cache; the same sentinel
    is consumed by both layers. This prevents one Browser session's live tools
    from leaking into any unrelated session.

    Single-profile processes intentionally keep the historical process-wide
    cache. A multiplex gateway installs a Hermes-home override for every
    profile turn, so the canonical profile key is the stable isolation
    boundary across repeated turns for that profile.
    """
    try:
        from gateway.session_context import get_session_env

        browser_identity = (
            get_session_env("HERMES_SESSION_ID", ""),
            get_session_env("HERMES_BROWSER_CONTROL_PRINCIPAL", ""),
            get_session_env("HERMES_BROWSER_CONTROL_TRANSPORT_FAMILY", ""),
        )
        if all(str(value or "").strip() for value in browser_identity):
            return CHECK_FN_CACHE_BYPASS
    except Exception:
        pass

    try:
        from agent.secret_scope import is_multiplex_active

        if not is_multiplex_active():
            return None
        from hermes_constants import get_hermes_home_override

        override = get_hermes_home_override()
        if not override:
            return CHECK_FN_CACHE_BYPASS
        return str(Path(override).expanduser().resolve())
    except Exception:
        # Fail closed: bypass both cache layers rather than aliasing requests
        # whose multiplex profile identity could not be resolved.
        return CHECK_FN_CACHE_BYPASS


def _run_check_fn_uncached(fn: Callable, *, unresolved_scope: bool = False) -> bool:
    """Run an availability check without cache/grace handling."""
    from agent.secret_scope import UnscopedSecretError

    try:
        return bool(fn())
    except UnscopedSecretError:
        if unresolved_scope:
            # Expected fail-closed probe: with multiplexing on, boot-time
            # check_fns run before any profile secret scope exists, so
            # get_secret raises by design. The tool re-probes on the first
            # scoped turn — log without a traceback so this cannot be
            # mistaken for a crashed check_fn (#100697).
            logger.debug(
                "check_fn %s hit the multiplex fail-closed path with no "
                "profile secret scope active; dependent tools re-probe on "
                "the first scoped turn",
                getattr(fn, "__qualname__", fn),
            )
            return False
        # The scope resolved but the read still failed closed: a genuinely
        # lost scope. Keep the loud crash-style report.
        logger.warning(
            "check_fn %s raised UnscopedSecretError while the profile cache "
            "scope was resolved; dependent tools will be unavailable this turn",
            getattr(fn, "__qualname__", fn),
            exc_info=True,
        )
        return False
    except Exception:
        detail = " while profile cache scope was unresolved" if unresolved_scope else ""
        logger.warning(
            "check_fn %s raised%s; dependent tools will be unavailable this turn",
            getattr(fn, "__qualname__", fn),
            detail,
            exc_info=True,
        )
        return False


def _check_fn_cached(fn: Callable) -> bool:
    """Return bool(fn()), TTL-cached across calls."""
    now = time.monotonic()
    if fn in _NO_CACHE_CHECK_FNS:
        return _run_check_fn_uncached(fn)
    scope = check_fn_cache_scope()
    if scope == CHECK_FN_CACHE_BYPASS:
        return _run_check_fn_uncached(fn, unresolved_scope=True)
    cache_key = (fn, scope)
    with _check_fn_cache_lock:
        _prune_check_fn_caches(now)
        cached = _check_fn_cache.get(cache_key)
        if cached is not None:
            ts, value = cached
            if now - ts < _CHECK_FN_TTL_SECONDS:
                return value

    raised = False
    try:
        value = bool(fn())
    except Exception:
        value = False
        raised = True

    with _check_fn_cache_lock:
        _prune_check_fn_caches(now)
        if value:
            _check_fn_last_good[cache_key] = now
            _check_fn_cache[cache_key] = (now, True)
            return True

        last_good = _check_fn_last_good.get(cache_key)
        if last_good is not None and now - last_good < _CHECK_FN_FAILURE_GRACE_SECONDS:
            # Recent success → treat this failure as a flake. Serve last-good
            # True and do NOT cache the failure, so the next call re-probes
            # rather than pinning a stale verdict for the full TTL.
            logger.warning(
                "check_fn %s failed (%s) within %.0fs of last success; "
                "treating as transient and keeping tool(s) available",
                getattr(fn, "__qualname__", fn),
                "raised" if raised else "returned False",
                _CHECK_FN_FAILURE_GRACE_SECONDS,
            )
            return True

        # No recent success (or grace expired) — honor the failure. Log it so
        # silent tool loss in quiet mode (subagents) is diagnosable.
        logger.warning(
            "check_fn %s %s; dependent tools will be unavailable this turn",
            getattr(fn, "__qualname__", fn),
            "raised" if raised else "returned False",
        )
        _check_fn_cache[cache_key] = (now, False)
        return False


def invalidate_check_fn_cache() -> None:
    """Drop all cached ``check_fn`` results. Call after config changes that
    affect tool availability (e.g. ``hermes tools enable``)."""
    with _check_fn_cache_lock:
        _check_fn_cache.clear()
        _check_fn_last_good.clear()


def get_cached_check_fn_result(fn: Callable) -> Optional[bool]:
    """Return the current cached verdict for *fn* if its TTL is still valid.

    Unlike :func:`_check_fn_cached`, this NEVER executes the probe. It is for
    read-only surfaces (e.g. dashboard status panels) that need the last-known
    availability without triggering network / auth / SDK work inside a request
    path. Returns ``None`` when there is no fresh cached verdict.
    """
    now = time.monotonic()
    scope = check_fn_cache_scope()
    if scope == CHECK_FN_CACHE_BYPASS:
        # Unresolved profile identity bypasses the cache entirely; there is no
        # trustworthy cached verdict to report.
        return None
    with _check_fn_cache_lock:
        cached = _check_fn_cache.get((fn, scope))
        if cached is None:
            return None
        ts, value = cached
        if now - ts < _CHECK_FN_TTL_SECONDS:
            return value
        return None


class ToolRegistry:
    """Singleton registry that collects tool schemas + handlers from tool files."""

    def __init__(self):
        # Built-in and other process-global registrations.
        self._tools: Dict[str, ToolEntry] = {}
        # Plugin registrations are overlays keyed by resolved HERMES_HOME. A
        # profile sees its own overlay first and then the global built-ins.
        self._scoped_tools: Dict[str, Dict[str, ToolEntry]] = {}
        # Plugin module namespace -> operator opt-in for built-in override.
        # Authorization records are lifecycle-managed; the separate scope map
        # remains durable so delayed callbacks stay profile-confined.
        self._plugin_override_policy: Dict[
            tuple[Optional[str], str], _PluginOverridePolicy
        ] = {}
        # Scope attribution stays durable after policy removal so delayed code
        # remains confined to the profile where its module was loaded.
        self._plugin_module_scopes: Dict[str, Set[Optional[str]]] = {}
        self._toolset_checks: Dict[str, Callable] = {}
        self._toolset_aliases: Dict[str, str] = {}
        # MCP dynamic refresh can mutate the registry while other threads are
        # reading tool metadata, so keep mutations serialized and readers on
        # stable snapshots.
        self._lock = threading.RLock()
        # Monotonically-increasing generation counter. Bumped on every
        # mutation (register / deregister / register_toolset_alias / MCP
        # refresh). External callers (e.g. get_tool_definitions) can memoize
        # against it: a cache entry keyed on the generation is valid for as
        # long as the generation hasn't changed.
        self._generation: int = 0

    @staticmethod
    def current_scope_key() -> str:
        """Return the active profile's canonical registry scope."""
        return hermes_home_key()

    def _merged_tools(self, scope: Optional[str] = None) -> Dict[str, ToolEntry]:
        """Return global tools overlaid with one profile's plugin tools."""
        active_scope = scope or self.current_scope_key()
        merged = dict(self._tools)
        merged.update(self._scoped_tools.get(active_scope, {}))
        return merged

    def _snapshot_state(
        self,
        scope: Optional[str] = None,
    ) -> tuple[List[ToolEntry], Dict[str, Callable]]:
        """Return a coherent snapshot of registry entries and toolset checks."""
        with self._lock:
            entries = list(self._merged_tools(scope).values())
            checks = dict(self._toolset_checks)
            for entry in entries:
                if entry.check_fn is not None:
                    checks[entry.toolset] = entry.check_fn
            return entries, checks

    def _snapshot_entries(self) -> List[ToolEntry]:
        """Return a stable snapshot of registered tool entries."""
        return self._snapshot_state()[0]

    def _toolset_has_exposable_tools(
        self,
        toolset: str,
        entries: List[ToolEntry],
    ) -> bool:
        """Return True when at least one tool in *toolset* would be exposed.

        Mirrors :meth:`get_tool_definitions` per-tool filtering so doctor,
        banners, and other toolset-level surfaces agree with runtime exposure.
        Mixed toolsets (e.g. ``terminal`` plus desktop-only ``read_terminal``)
        must not be gated solely by the first registered ``check_fn``.
        """
        check_results: Dict[Callable, bool] = {}
        for entry in entries:
            if entry.toolset != toolset:
                continue
            if not entry.check_fn:
                return True
            if entry.check_fn not in check_results:
                check_results[entry.check_fn] = _check_fn_cached(entry.check_fn)
            if check_results[entry.check_fn]:
                return True
        return False

    def get_entry(
        self,
        name: str,
        *,
        scope: Optional[str] = None,
    ) -> Optional[ToolEntry]:
        """Return the active profile's entry by name, falling back to global."""
        with self._lock:
            return self._merged_tools(scope).get(name)

    def snapshot_registration(
        self,
        name: str,
        *,
        scope: Optional[str] = None,
    ) -> Optional[ToolEntry]:
        """Return the local slot state without following global fallback."""
        with self._lock:
            target = self._tools if scope is None else self._scoped_tools.get(scope, {})
            return target.get(name)

    def get_registered_toolset_names(self) -> List[str]:
        """Return sorted unique toolset names present in the registry."""
        return sorted({entry.toolset for entry in self._snapshot_entries()})

    def get_all_entries(self) -> List[ToolEntry]:
        """Return the active profile's merged tool entries."""
        return self._snapshot_entries()

    def get_tool_names_for_toolset(self, toolset: str) -> List[str]:
        """Return sorted tool names registered under a given toolset."""
        return sorted(
            entry.name for entry in self._snapshot_entries()
            if entry.toolset == toolset
        )

    def register_toolset_alias(self, alias: str, toolset: str) -> None:
        """Register an explicit alias for a canonical toolset name."""
        with self._lock:
            existing = self._toolset_aliases.get(alias)
            if existing and existing != toolset:
                logger.warning(
                    "Toolset alias collision: '%s' (%s) overwritten by %s",
                    alias, existing, toolset,
                )
            self._toolset_aliases[alias] = toolset
            self._generation += 1

    def get_registered_toolset_aliases(self) -> Dict[str, str]:
        """Return a snapshot of ``{alias: canonical_toolset}`` mappings."""
        with self._lock:
            return dict(self._toolset_aliases)

    def get_toolset_alias_target(self, alias: str) -> Optional[str]:
        """Return the canonical toolset name for an alias, or None."""
        with self._lock:
            return self._toolset_aliases.get(alias)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_plugin_override_policy(
        self,
        module_namespace: str,
        allowed: bool,
        *,
        scope: Optional[str] = None,
    ) -> _PluginOverridePolicy:
        """Bind a plugin module namespace to its current operator opt-in.

        The identity-bearing result lets plugin unload/reload revoke a stale
        authorization without losing durable module-to-profile attribution.
        """
        with self._lock:
            policy = _PluginOverridePolicy(allowed)
            self._plugin_override_policy[(scope, module_namespace)] = policy
            self._plugin_module_scopes.setdefault(module_namespace, set()).add(scope)
            return policy

    def snapshot_plugin_override_policy(
        self,
        module_namespace: str,
        *,
        scope: Optional[str] = None,
    ) -> Optional[_PluginOverridePolicy]:
        """Return one local authorization generation without fallback."""
        with self._lock:
            return self._plugin_override_policy.get((scope, module_namespace))

    def restore_plugin_override_policy(
        self,
        module_namespace: str,
        current: _PluginOverridePolicy,
        previous: Optional[_PluginOverridePolicy],
        *,
        scope: Optional[str] = None,
    ) -> bool:
        """CAS-restore policy state while retaining durable scope attribution."""
        with self._lock:
            key = (scope, module_namespace)
            if self._plugin_override_policy.get(key) is not current:
                return False
            if previous is None:
                self._plugin_override_policy.pop(key, None)
            else:
                self._plugin_override_policy[key] = previous
            return True

    def _plugin_override_allowed(
        self,
        scope: Optional[str],
        module_namespace: str,
    ) -> bool:
        policy = self._plugin_override_policy.get((scope, module_namespace))
        if policy is None and scope is not None:
            policy = self._plugin_override_policy.get((None, module_namespace))
        return bool(policy and policy.allowed)

    def _plugin_owner_of(self, handler: Callable) -> Optional[str]:
        """Return the plugin module namespace that defined *handler*, or None
        if it was not defined in a loaded plugin module.

        Authorization is bound to where the handler was DEFINED
        (``handler.__globals__["__name__"]``), which is fixed at definition
        time and cannot drift with the call site, thread, or timing. Lambdas
        and nested functions inherit the defining module's globals, so a
        plugin cannot launder an override through a callback. Built-in/MCP
        handlers live outside the plugin namespace and return None (unchanged
        behavior).
        """
        mod = self._callable_module(handler)
        if not mod:
            return None
        return self._plugin_namespace_of_module(mod)

    @staticmethod
    def _callable_module(handler: Callable) -> str:
        """Resolve defining module through wrappers, partials, and objects."""
        current = handler
        seen: Set[int] = set()
        while id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, functools.partial):
                current = current.func
                continue
            func = getattr(current, "__func__", None)
            if func is not None:
                current = func
                continue
            globals_dict = getattr(current, "__globals__", None)
            if isinstance(globals_dict, dict):
                module_name = globals_dict.get("__name__", "")
                if module_name:
                    return str(module_name)
            wrapped = getattr(current, "__wrapped__", None)
            if wrapped is not None:
                current = wrapped
                continue
            break
        module_name = getattr(current, "__module__", "")
        if module_name:
            return str(module_name)
        return str(getattr(type(current), "__module__", "") or "")

    def _plugin_namespace_of_module(
        self,
        module_namespace: str,
    ) -> Optional[str]:
        """Resolve a module/submodule to its durable plugin namespace."""
        with self._lock:
            matches = [
                namespace
                for namespace in self._plugin_module_scopes
                if module_namespace == namespace
                or module_namespace.startswith(f"{namespace}.")
            ]
            if matches:
                return max(matches, key=len)
        # Also gate plugin modules currently loading but not yet policy-recorded
        # (defensive: a handler defined in the plugin namespace is plugin code).
        if module_namespace.startswith("hermes_plugins."):
            return ".".join(module_namespace.split(".")[:2])
        return None

    def _plugin_scope_of(self, module_namespace: str) -> Optional[str]:
        """Return the profile scope bound to a loaded plugin module."""
        with self._lock:
            scopes = self._plugin_module_scopes.get(module_namespace)
            if not scopes:
                return None
            active_scope = self.current_scope_key()
            if active_scope in scopes:
                return active_scope
            if len(scopes) == 1:
                return next(iter(scopes))
            raise PermissionError(
                f"Plugin module {module_namespace!r} is active in multiple "
                "profiles and cannot register outside one of those scopes."
            )

    def plugin_scope_for_module(self, module_namespace: str) -> Optional[str]:
        """Public host lookup for a loaded plugin module's immutable scope."""
        owner = self._plugin_namespace_of_module(module_namespace)
        return self._plugin_scope_of(owner or module_namespace)

    def plugin_scope_for_callable(self, callback: Callable) -> Optional[str]:
        """Return the durable plugin scope for any supported callable shape."""
        module_name = self._callable_module(callback)
        return self.plugin_scope_for_module(module_name) if module_name else None

    @staticmethod
    def _caller_module() -> str:
        """Best-effort module name of whoever called the registry method that
        invoked this helper (two frames up: this helper, then the registry
        method itself, then the actual caller).

        ``deregister()`` takes only a tool name — unlike ``register()`` it has
        no handler argument to bind authorization to via ``_plugin_owner_of``.
        Frame inspection is the only way to know who is asking.
        """
        try:
            frame = sys._getframe(2)
            return frame.f_globals.get("__name__", "") or ""
        except Exception:
            return ""

    def register(
        self,
        name: str,
        toolset: str,
        schema: dict,
        handler: Callable,
        check_fn: Callable = None,
        requires_env: list = None,
        is_async: bool = False,
        description: str = "",
        emoji: str = "",
        max_result_size_chars: int | float | None = None,
        dynamic_schema_overrides: Callable = None,
        override: bool = False,
        scope: Optional[str] = None,
    ):
        """Register a tool.  Called at module-import time by each tool file.

        ``override=True`` is an explicit opt-in for plugins that intend to
        replace an existing built-in tool implementation (e.g. swap the
        default browser tool for a headed-Chrome CDP backend). Without it,
        registrations that would shadow an existing tool from a different
        toolset are rejected to prevent accidental overwrites.
        """
        handler_owner = self._plugin_owner_of(handler)
        caller_owner = self._plugin_namespace_of_module(self._caller_module())
        owner = caller_owner or handler_owner
        if scope is None and owner is not None:
            scope = self._plugin_scope_of(owner)
        with self._lock:
            target = (
                self._tools
                if scope is None
                else self._scoped_tools.setdefault(scope, {})
            )
            existing = (
                self._tools.get(name)
                if scope is None
                else self._merged_tools(scope).get(name)
            )
            shadows_global = (
                owner is not None
                and scope is not None
                and name not in target
                and name in self._tools
            )
            if shadows_global:
                if not override:
                    logger.error(
                        "Tool registration REJECTED: plugin %r attempted to "
                        "shadow global tool %r without override=True",
                        owner,
                        name,
                    )
                    return
                if not self._plugin_override_allowed(scope, owner):
                    raise PermissionError(
                        f"Plugin module {owner!r} cannot override built-in "
                        f"tool {name!r} without operator opt-in "
                        f"(allow_tool_override)."
                    )
            if existing and existing.toolset != toolset:
                if override:
                    if owner is not None and not self._plugin_override_allowed(
                        scope, owner
                    ):
                        logger.error(
                            "Tool registration REJECTED: plugin %r attempted to "
                            "override built-in tool %r (existing toolset %r) without "
                            "operator opt-in. Set "
                            "plugins.entries.<plugin_id>.allow_tool_override: true "
                            "in config.yaml to allow it.",
                            owner, name, existing.toolset,
                        )
                        raise PermissionError(
                            f"Plugin module {owner!r} cannot override built-in "
                            f"tool {name!r} without operator opt-in "
                            f"(allow_tool_override)."
                        )
                    # Explicit opt-in (or non-plugin caller): replace the tool.
                    # Logged at INFO so the override is auditable in agent.log.
                    logger.info(
                        "Tool '%s': toolset '%s' overriding existing toolset '%s' "
                        "(override=True opt-in)",
                        name,
                        toolset,
                        existing.toolset,
                    )
                else:
                    # Reject every cross-toolset shadow, including MCP-to-MCP
                    # collisions. Legitimate MCP reconnect/refresh re-registers
                    # within the same canonical toolset and remains allowed.
                    logger.error(
                        "Tool registration REJECTED: '%s' (toolset '%s') would "
                        "shadow existing tool from toolset '%s'. Pass "
                        "override=True to register() if the replacement is "
                        "intentional, or deregister the existing tool first.",
                        name,
                        toolset,
                        existing.toolset,
                    )
                    return
            target[name] = ToolEntry(
                name=name,
                toolset=toolset,
                schema=schema,
                handler=handler,
                check_fn=check_fn,
                requires_env=requires_env or [],
                is_async=is_async,
                description=description or schema.get("description", ""),
                emoji=emoji,
                max_result_size_chars=max_result_size_chars,
                dynamic_schema_overrides=dynamic_schema_overrides,
            )
            # Availability is now derived per-tool (_toolset_has_exposable_tools),
            # so this map no longer gates a toolset. It is still consumed by
            # get_toolset_requirements -> TOOLSET_REQUIREMENTS["check_fn"], which
            # banner.py reads (presence only, never called) to classify an
            # already-unavailable toolset as lazy-init vs disabled. Keep the
            # write path for that classification.
            if scope is None and check_fn and toolset not in self._toolset_checks:
                self._toolset_checks[toolset] = check_fn
            self._generation += 1

    def deregister(self, name: str, *, scope: Optional[str] = None) -> None:
        """Remove a tool from the registry.

        Also cleans up the toolset check if no other tools remain in the
        same toolset.  Used by MCP dynamic tool discovery to nuke-and-repave
        when a server sends ``notifications/tools/list_changed``.

        ``scope`` selects a profile overlay explicitly (multiplexed MCP tools
        live in the owning profile's overlay). Plugin callers keep their own
        scope and may not name another one; non-plugin callers without
        ``scope`` keep the historical process-global target.

        Gated by the same operator opt-in policy ``register(override=True)``
        enforces. Without this, a plugin could bypass that gate entirely by
        deregistering a tool it doesn't own and then calling plain
        ``register()`` over the now-empty slot — ``register()`` only runs its
        override check when an ``existing`` entry is present, so removing it
        first skips the check altogether. MCP toolsets (``mcp-*``) are exempt:
        dynamic tool discovery legitimately nukes-and-repaves its own tools on
        every refresh and has no plugin-override concept.
        """
        with self._lock:
            caller_mod = self._caller_module()
            caller_owner = self._plugin_namespace_of_module(caller_mod)
            caller_scope = (
                self._plugin_scope_of(caller_owner)
                if caller_owner is not None
                else None
            )
            if caller_owner is not None and scope is not None and scope != caller_scope:
                raise PermissionError(
                    f"Plugin module {caller_mod!r} cannot deregister tools "
                    "outside its own profile scope."
                )
            if scope is None:
                scope = caller_scope
            target = (
                self._scoped_tools.get(scope, {})
                if scope is not None
                else self._tools
            )
            entry = target.get(name)
            if entry is None and scope is not None:
                if caller_owner is not None and name in self._tools:
                    raise PermissionError(
                        f"Scoped plugin module {caller_mod!r} cannot deregister "
                        f"process-global tool {name!r}; register a scoped "
                        "override instead."
                    )
                return
            if entry is None:
                return
            if not entry.toolset.startswith("mcp-"):
                owner = self._plugin_owner_of(entry.handler)
                # Ownership check: bind to the plugin package root
                # (``hermes_plugins.{name}``), not the exact module string.
                # A handler defined in ``hermes_plugins.pkg.handlers`` is
                # still owned by the ``hermes_plugins.pkg`` package — exact
                # string equality would wrongly block root-module cleanup code
                # from removing tools registered by a submodule of the same
                # plugin (egilewski review on #55840).
                same_plugin = bool(owner and caller_owner == owner)
                if (
                    caller_owner is not None
                    and not same_plugin
                    and not self._plugin_override_allowed(
                        caller_scope, caller_owner
                    )
                ):
                    logger.error(
                        "Tool deregistration REJECTED: plugin %r attempted to "
                        "remove tool %r (toolset %r) it does not own, without "
                        "operator opt-in. Set "
                        "plugins.entries.%s.allow_tool_override: true in "
                        "config.yaml to allow it.",
                        caller_mod, name, entry.toolset, caller_mod,
                    )
                    raise PermissionError(
                        f"Plugin module {caller_mod!r} cannot deregister tool "
                        f"{name!r} (toolset {entry.toolset!r}) without operator "
                        f"opt-in (allow_tool_override)."
                    )
            del target[name]
            if scope is not None and not target:
                self._scoped_tools.pop(scope, None)
            # Drop the toolset check and aliases if this was the last tool in
            # that toolset.
            toolset_still_exists = any(
                e.toolset == entry.toolset
                for e in self._merged_tools(scope).values()
            )
            if not toolset_still_exists:
                self._toolset_checks.pop(entry.toolset, None)
                self._toolset_aliases = {
                    alias: target
                    for alias, target in self._toolset_aliases.items()
                    if target != entry.toolset
                }
            self._generation += 1
        logger.debug("Deregistered tool: %s", name)

    def restore_registration(
        self,
        name: str,
        current: ToolEntry,
        previous: Optional[ToolEntry],
        *,
        scope: Optional[str] = None,
    ) -> bool:
        """Restore a host-owned registration if it is still current.

        This is the narrow inverse used by the plugin ownership ledger.  The
        identity check is deliberate: another plugin (or another
        ``PluginManager`` in a multi-profile process) may have registered a
        newer entry under the same name, in which case unloading this entry
        must leave the newer entry untouched.
        """
        with self._lock:
            target = (
                self._tools
                if scope is None
                else self._scoped_tools.setdefault(scope, {})
            )
            if target.get(name) is not current:
                return False

            if previous is None:
                target.pop(name, None)
            else:
                target[name] = previous
            if scope is not None and not target:
                self._scoped_tools.pop(scope, None)

            # Rebuild the affected toolset checks from the surviving entries.
            # A plugin may have replaced an entry in the same toolset, so
            # simply leaving the current check_fn behind would retain stale
            # plugin state after restoration.
            affected_toolsets = {current.toolset}
            if previous is not None:
                affected_toolsets.add(previous.toolset)
            for toolset in affected_toolsets:
                surviving = [
                    entry for entry in self._merged_tools(scope).values()
                    if entry.toolset == toolset
                ]
                check_fn = next(
                    (entry.check_fn for entry in surviving if entry.check_fn),
                    None,
                )
                if scope is None:
                    if check_fn is None:
                        self._toolset_checks.pop(toolset, None)
                    else:
                        self._toolset_checks[toolset] = check_fn
                if not surviving and not any(
                    entry.toolset == toolset
                    for entries in self._scoped_tools.values()
                    for entry in entries.values()
                ):
                    self._toolset_aliases = {
                        alias: target
                        for alias, target in self._toolset_aliases.items()
                        if target != toolset
                    }
            self._generation += 1
        logger.debug("Restored tool registration: %s", name)
        return True

    # ------------------------------------------------------------------
    # Schema retrieval
    # ------------------------------------------------------------------

    def get_definitions(self, tool_names: Set[str], quiet: bool = False) -> List[dict]:
        """Return OpenAI-format tool schemas for the requested tool names.

        Only tools whose ``check_fn()`` returns True (or have no check_fn)
        are included. ``check_fn()`` results are cached for ~30 s via
        :func:`_check_fn_cached` to amortize repeat probes (check_terminal_
        requirements probes modal/docker, browser checks probe playwright,
        etc.); TTL chosen so env-var changes (``hermes tools enable foo``)
        still take effect in near-real-time without forcing a full cache
        flush on every call.
        """
        result = []
        # Per-call cache on top of the 30 s TTL — handles repeat probes of the
        # same check_fn within one definitions pass without re-reading the
        # TTL clock.
        check_results: Dict[Callable, bool] = {}
        entries_by_name = {entry.name: entry for entry in self._snapshot_entries()}
        for name in sorted(tool_names):
            entry = entries_by_name.get(name)
            if not entry:
                continue
            if entry.check_fn:
                if entry.check_fn not in check_results:
                    check_results[entry.check_fn] = _check_fn_cached(entry.check_fn)
                if not check_results[entry.check_fn]:
                    if not quiet:
                        logger.debug("Tool %s unavailable (check failed)", name)
                    continue
            # Ensure schema always has a "name" field — use entry.name as fallback
            schema_with_name = {**entry.schema, "name": entry.name}
            # Apply runtime-dynamic overrides (e.g. delegate_task description
            # depends on current delegation.max_concurrent_children /
            # max_spawn_depth). Caller side (model_tools.get_tool_definitions)
            # already keys its memo on config.yaml mtime + size, so changes
            # to delegation.* in config invalidate the cache automatically.
            if entry.dynamic_schema_overrides is not None:
                try:
                    overrides = entry.dynamic_schema_overrides()
                    if isinstance(overrides, dict):
                        schema_with_name.update(overrides)
                except Exception as exc:
                    logger.warning(
                        "dynamic_schema_overrides for tool %s raised %s; "
                        "using static schema",
                        name, exc,
                    )
            result.append({"type": "function", "function": schema_with_name})
        return result

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_handler_result(name: str, result):
        """Enforce the result shapes supported by the agent tool pipeline.

        Normal tool results are strings.  The sole structured exception is the
        multimodal envelope consumed by the agent executor.  Returning every
        other value as a string error keeps logging, hooks, budgeting, and
        persistence from receiving values they cannot safely slice or size.
        """
        if isinstance(result, str):
            return _bound_json_error_result(result)
        if (
            isinstance(result, dict)
            and result.get("_multimodal") is True
            and isinstance(result.get("content"), list)
        ):
            return result

        result_type = type(result).__name__
        logger.error(
            "Tool %s handler returned unsupported result type: %s",
            name,
            result_type,
        )
        return tool_error(
            f"Tool handler returned unsupported result type: {result_type}",
            error_type="tool_result_contract",
            tool=name,
            result_type=result_type,
        )

    def dispatch(
        self,
        name: str,
        args: dict,
        *,
        scope: Optional[str] = None,
        **kwargs,
    ) -> str | dict:
        """Execute a tool handler by name.

        * Async handlers are bridged automatically via ``_run_async()``.
        * Handler results are normalized to a string or supported multimodal
          envelope before leaving the registry.
        * All exceptions are caught and returned as ``{"error": "..."}``
          for consistent error format.
        """
        entry = self.get_entry(name, scope=scope)
        if not entry:
            return tool_error(f"Unknown tool: {name}")
        try:
            if entry.is_async:
                from model_tools import _run_async
                result = _run_async(entry.handler(args, **kwargs))
            else:
                result = entry.handler(args, **kwargs)
            return self._normalize_handler_result(name, result)
        except Exception as e:
            # exc_info already renders the exception, so keep the message copy bounded.
            logger.exception(
                "Tool %s dispatch error: %s", name, _bound_error_text(str(e))
            )
            # Route through the sanitizer so framing tokens / CDATA / fences
            # in exception strings don't reach the model as structural noise.
            # See model_tools._sanitize_tool_error for rationale.
            raw = f"Tool execution failed: {type(e).__name__}: {e}"
            try:
                from model_tools import _sanitize_tool_error
                sanitized = _sanitize_tool_error(raw)
            except Exception:
                sanitized = raw  # defensive: never let the sanitizer block error propagation
            return tool_error(sanitized)

    # ------------------------------------------------------------------
    # Query helpers  (replace redundant dicts in model_tools.py)
    # ------------------------------------------------------------------

    def get_max_result_size(self, name: str, default: int | float | None = None) -> int | float:
        """Return per-tool max result size, or *default* (or global default)."""
        entry = self.get_entry(name)
        if entry and entry.max_result_size_chars is not None:
            return entry.max_result_size_chars
        if default is not None:
            return default
        from tools.budget_config import DEFAULT_RESULT_SIZE_CHARS
        return DEFAULT_RESULT_SIZE_CHARS

    def get_all_tool_names(self) -> List[str]:
        """Return sorted list of all registered tool names."""
        return sorted(entry.name for entry in self._snapshot_entries())

    def get_schema(self, name: str) -> Optional[dict]:
        """Return a tool's raw schema dict, bypassing check_fn filtering.

        Useful for token estimation and introspection where availability
        doesn't matter — only the schema content does.
        """
        entry = self.get_entry(name)
        return entry.schema if entry else None

    def get_toolset_for_tool(self, name: str) -> Optional[str]:
        """Return the toolset a tool belongs to, or None."""
        entry = self.get_entry(name)
        return entry.toolset if entry else None

    def get_emoji(self, name: str, default: str = "⚡") -> str:
        """Return the emoji for a tool, or *default* if unset."""
        entry = self.get_entry(name)
        return (entry.emoji if entry and entry.emoji else default)

    def get_tool_to_toolset_map(self) -> Dict[str, str]:
        """Return ``{tool_name: toolset_name}`` for every registered tool."""
        return {entry.name: entry.toolset for entry in self._snapshot_entries()}

    def is_toolset_available(self, toolset: str) -> bool:
        """Check if a toolset has at least one exposable tool.

        Returns False (rather than crashing) when a per-tool check raises
        an unexpected exception (e.g. network error, missing import, bad config).
        """
        entries, _ = self._snapshot_state()
        return self._toolset_has_exposable_tools(toolset, entries)

    def check_toolset_requirements(self) -> Dict[str, bool]:
        """Return ``{toolset: available_bool}`` for every toolset."""
        entries, _ = self._snapshot_state()
        toolsets = sorted({entry.toolset for entry in entries})
        return {
            toolset: self._toolset_has_exposable_tools(toolset, entries)
            for toolset in toolsets
        }

    def get_available_toolsets(self) -> Dict[str, dict]:
        """Return toolset metadata for UI display."""
        toolsets: Dict[str, dict] = {}
        entries, _ = self._snapshot_state()
        for entry in entries:
            ts = entry.toolset
            if ts not in toolsets:
                toolsets[ts] = {
                    "available": self._toolset_has_exposable_tools(ts, entries),
                    "tools": [],
                    "description": "",
                    "requirements": [],
                }
            toolsets[ts]["tools"].append(entry.name)
            if entry.requires_env:
                for env in entry.requires_env:
                    if env not in toolsets[ts]["requirements"]:
                        toolsets[ts]["requirements"].append(env)
        return toolsets

    def get_toolset_requirements(self) -> Dict[str, dict]:
        """Build a TOOLSET_REQUIREMENTS-compatible dict for backward compat."""
        result: Dict[str, dict] = {}
        entries, toolset_checks = self._snapshot_state()
        for entry in entries:
            ts = entry.toolset
            if ts not in result:
                result[ts] = {
                    "name": ts,
                    "env_vars": [],
                    "check_fn": toolset_checks.get(ts),
                    "setup_url": None,
                    "tools": [],
                }
            if entry.name not in result[ts]["tools"]:
                result[ts]["tools"].append(entry.name)
            for env in entry.requires_env:
                if env not in result[ts]["env_vars"]:
                    result[ts]["env_vars"].append(env)
        return result

    def check_tool_availability(self, quiet: bool = False):
        """Return (available_toolsets, unavailable_info) like the old function."""
        available = []
        unavailable = []
        entries, _ = self._snapshot_state()
        for ts in sorted({entry.toolset for entry in entries}):
            ts_entries = [entry for entry in entries if entry.toolset == ts]
            if self._toolset_has_exposable_tools(ts, entries):
                available.append(ts)
            else:
                unavailable.append({
                    "name": ts,
                    "env_vars": ts_entries[0].requires_env if ts_entries else [],
                    "tools": [entry.name for entry in ts_entries],
                })
        return available, unavailable


# Module-level singleton
registry = ToolRegistry()


# ---------------------------------------------------------------------------
# Helpers for tool response serialization
# ---------------------------------------------------------------------------
# Every tool handler must return a JSON string.  These helpers eliminate the
# boilerplate ``json.dumps({"error": msg}, ensure_ascii=False)`` that appears
# hundreds of times across tool files.
#
# Usage:
#   from tools.registry import registry, tool_error, tool_result
#
#   return tool_error("something went wrong")
#   return tool_error("not found", code=404)
#   return tool_result(success=True, data=payload)
#   return tool_result(items)            # pass a dict directly


def tool_error(message, **extra) -> str:
    """Return a JSON error string for tool handlers.

    >>> tool_error("file not found")
    '{"error": "file not found"}'
    >>> tool_error("bad input", success=False)
    '{"error": "bad input", "success": false}'
    """
    # Bound the context-bound copy so a raw exception can't bloat history across retries.
    result = {"error": _bound_error_text(str(message))}
    if extra:
        result.update(extra)
    return json.dumps(result, ensure_ascii=False)


def tool_result(data=None, **kwargs) -> str:
    """Return a JSON result string for tool handlers.

    Accepts a dict positional arg *or* keyword arguments (not both):

    >>> tool_result(success=True, count=42)
    '{"success": true, "count": 42}'
    >>> tool_result({"key": "value"})
    '{"key": "value"}'
    """
    if data is not None:
        return json.dumps(data, ensure_ascii=False)
    return json.dumps(kwargs, ensure_ascii=False)
