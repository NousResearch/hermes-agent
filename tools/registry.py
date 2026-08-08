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
import importlib
import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

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


# ``None`` continues to mean a process-global registration.  A non-empty
# canonical Hermes home is an owner only for dynamic MCP registrations.  The
# sentinel lets lookups distinguish an unresolved multiplex profile from a
# normal global-only snapshot without exposing that implementation detail in
# the public API.
_PROFILE_HOME_UNSET = object()
_PROFILE_SCOPE_UNRESOLVED = object()


def _canonical_profile_home(profile_home) -> Optional[str]:
    """Return the stable absolute key for a profile home, or ``None``."""
    if profile_home is None:
        return None
    value = str(profile_home).strip()
    if not value:
        return None
    try:
        return str(Path(value).expanduser().resolve())
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _current_profile_home():
    """Resolve the current profile owner, failing closed in multiplex mode."""
    try:
        from agent.secret_scope import is_multiplex_active

        if is_multiplex_active():
            from hermes_constants import get_hermes_home_override

            override = get_hermes_home_override()
            if not override:
                return _PROFILE_SCOPE_UNRESOLVED

        from hermes_constants import get_hermes_home

        return _canonical_profile_home(get_hermes_home())
    except Exception:
        # Registry is imported very early.  If the profile context cannot be
        # resolved, expose only process-global entries rather than risking a
        # lookup against another profile's MCP handler.
        return _PROFILE_SCOPE_UNRESOLVED


def _resolve_profile_home(profile_home=_PROFILE_HOME_UNSET):
    """Resolve an explicit or context-local owner for registry operations."""
    if profile_home is not _PROFILE_HOME_UNSET and profile_home is not None:
        return _canonical_profile_home(profile_home)
    return _current_profile_home()


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
        "max_result_size_chars", "dynamic_schema_overrides", "profile_home",
    )

    def __init__(self, name, toolset, schema, handler, check_fn,
                 requires_env, is_async, description, emoji,
                 max_result_size_chars=None, dynamic_schema_overrides=None,
                 profile_home=None):
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
        # Optional provenance for dynamic tools (currently MCP). Built-in and
        # plugin registrations leave this unset; callers that own a profile
        # pass a canonical absolute path captured at registration time.
        self.profile_home = str(profile_home) if profile_home is not None else None


# ---------------------------------------------------------------------------
# check_fn TTL cache
#
# check_fn callables like tools/terminal_tool.check_terminal_requirements
# probe external state (Docker daemon, Modal SDK install, playwright binary
# availability). For a long-lived CLI or gateway process, calling them on
# every get_definitions() is pure waste — external state changes on human
# timescales. Cache results for ~30 s so env-var flips via ``hermes tools``
# or live credential file changes propagate within a turn or two without
# requiring any explicit invalidation.
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
# Monotonic timestamp of the most recent True result per check_fn.
_check_fn_last_good: Dict[tuple[Callable, Optional[str]], float] = {}
_check_fn_cache_lock = threading.Lock()
CHECK_FN_CACHE_BYPASS = ""


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

    Single-profile processes intentionally keep the historical process-wide
    cache. A multiplex gateway installs a Hermes-home override for every
    profile turn, so the canonical profile key is the stable isolation
    boundary across repeated turns for that profile.
    """
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


def _check_fn_cached(fn: Callable) -> bool:
    """Return bool(fn()), TTL-cached across calls.

    Exceptions are swallowed as False. A transient False/exception within
    ``_CHECK_FN_FAILURE_GRACE_SECONDS`` of the last True is suppressed (the
    last-good True is returned and the failure is NOT cached, so the next call
    re-probes) to keep flaky external checks (Docker daemon busy, socket
    contention, probe timeout) from silently stripping tools mid-session.
    """
    now = time.monotonic()
    scope = check_fn_cache_scope()
    if scope == CHECK_FN_CACHE_BYPASS:
        try:
            return bool(fn())
        except Exception:
            logger.warning(
                "check_fn %s raised while profile cache scope was unresolved; "
                "dependent tools will be unavailable this turn",
                getattr(fn, "__qualname__", fn),
                exc_info=True,
            )
            return False
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
        # Built-in and plugin tools retain the historical process-global map.
        # MCP entries live in a second dimension keyed by canonical profile
        # home so identical public names can coexist without replacing one
        # another's handler or schema.
        self._tools: Dict[str, ToolEntry] = {}
        self._profile_tools: Dict[str, Dict[str, ToolEntry]] = {}
        # Durable map: plugin module namespace (handler.__globals__["__name__"])
        # -> operator opt-in for built-in override. Populated at plugin load and
        # never cleared, so a plugin's override authorization is bound to the
        # code that defined the handler, independent of WHEN the register() call
        # happens (sync during load, or a delayed/threaded callback afterwards).
        self._plugin_override_policy: Dict[str, bool] = {}
        self._toolset_checks: Dict[str, Callable] = {}
        self._profile_toolset_checks: Dict[tuple[str, str], Callable] = {}
        self._toolset_aliases: Dict[str, str] = {}
        self._profile_toolset_aliases: Dict[str, Dict[str, str]] = {}
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

    def _snapshot_state(
        self, profile_home=_PROFILE_HOME_UNSET
    ) -> tuple[List[ToolEntry], Dict[str, Callable]]:
        """Return a coherent snapshot of registry entries and toolset checks."""
        with self._lock:
            entries = list(self._tools.values())
            toolset_checks = dict(self._toolset_checks)
            scope = _resolve_profile_home(profile_home)
            if scope is not _PROFILE_SCOPE_UNRESOLVED and scope is not None:
                entries.extend(self._profile_tools.get(scope, {}).values())
                for (home, toolset), check_fn in self._profile_toolset_checks.items():
                    if home == scope:
                        toolset_checks.setdefault(toolset, check_fn)
            return entries, toolset_checks

    def _snapshot_entries(self, profile_home=_PROFILE_HOME_UNSET) -> List[ToolEntry]:
        """Return a stable snapshot of registered tool entries."""
        if profile_home is _PROFILE_HOME_UNSET:
            return self._snapshot_state()[0]
        return self._snapshot_state(profile_home)[0]

    def _entry_for_name_locked(self, name: str, profile_home=_PROFILE_HOME_UNSET):
        """Return the selected profile entry, falling back to global tools."""
        scope = _resolve_profile_home(profile_home)
        if scope is not _PROFILE_SCOPE_UNRESOLVED and scope is not None:
            entry = self._profile_tools.get(scope, {}).get(name)
            if entry is not None:
                return entry
        return self._tools.get(name)

    def _profile_entry_exists_locked(self, name: str) -> bool:
        """Return whether any profile owns *name* (for global collision checks)."""
        return any(name in entries for entries in self._profile_tools.values())

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
        self, name: str, *, profile_home=_PROFILE_HOME_UNSET
    ) -> Optional[ToolEntry]:
        """Return a global or current-profile entry by public name.

        Passing ``profile_home`` explicitly is required by background MCP
        work that runs outside the profile context.  In a multiplex process,
        an omitted/implicit profile with no Hermes-home override selects only
        global entries; it never guesses an MCP owner.
        """
        with self._lock:
            return self._entry_for_name_locked(name, profile_home)

    def get_registered_toolset_names(self, *, profile_home=_PROFILE_HOME_UNSET) -> List[str]:
        """Return sorted unique toolset names present in the registry."""
        return sorted({entry.toolset for entry in self._snapshot_entries(profile_home)})

    def get_tool_names_for_toolset(
        self, toolset: str, *, profile_home=_PROFILE_HOME_UNSET
    ) -> List[str]:
        """Return sorted tool names registered under a given toolset."""
        return sorted(
            entry.name for entry in self._snapshot_entries(profile_home)
            if entry.toolset == toolset
        )

    def register_toolset_alias(
        self, alias: str, toolset: str, *, profile_home=_PROFILE_HOME_UNSET
    ) -> None:
        """Register an explicit global or profile-scoped toolset alias."""
        with self._lock:
            scope = _resolve_profile_home(profile_home)
            if scope is _PROFILE_SCOPE_UNRESOLVED:
                logger.error(
                    "Toolset alias registration REJECTED: profile scope for %r "
                    "is unresolved in multiplex mode",
                    alias,
                )
                return
            explicit_profile = (
                profile_home is not _PROFILE_HOME_UNSET
                and _canonical_profile_home(profile_home) is not None
            )
            if scope is not None:
                profile_entries = self._profile_tools.get(scope, {}).values()
                profile_owns_toolset = any(
                    entry.toolset == toolset for entry in profile_entries
                )
            else:
                profile_owns_toolset = False
            aliases = (
                self._profile_toolset_aliases.setdefault(scope, {})
                if scope is not None and (explicit_profile or profile_owns_toolset)
                else self._toolset_aliases
            )
            existing = aliases.get(alias)
            if existing and existing != toolset:
                logger.warning(
                    "Toolset alias collision: '%s' (%s) overwritten by %s",
                    alias, existing, toolset,
                )
            aliases[alias] = toolset
            self._generation += 1

    def get_registered_toolset_aliases(
        self, *, profile_home=_PROFILE_HOME_UNSET
    ) -> Dict[str, str]:
        """Return a snapshot of ``{alias: canonical_toolset}`` mappings."""
        with self._lock:
            aliases = dict(self._toolset_aliases)
            scope = _resolve_profile_home(profile_home)
            if scope is not _PROFILE_SCOPE_UNRESOLVED and scope is not None:
                aliases.update(self._profile_toolset_aliases.get(scope, {}))
            return aliases

    def get_toolset_alias_target(
        self, alias: str, *, profile_home=_PROFILE_HOME_UNSET
    ) -> Optional[str]:
        """Return the canonical toolset name for an alias, or None."""
        with self._lock:
            scope = _resolve_profile_home(profile_home)
            if scope is not _PROFILE_SCOPE_UNRESOLVED and scope is not None:
                target = self._profile_toolset_aliases.get(scope, {}).get(alias)
                if target is not None:
                    return target
            return self._toolset_aliases.get(alias)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_plugin_override_policy(self, module_namespace: str, allowed: bool) -> None:
        """Bind a plugin module namespace to its operator opt-in for built-in
        override. Called once per plugin at load time. Durable: never cleared,
        so later (even threaded/delayed) register() calls from that module are
        still gated by the same policy.
        """
        with self._lock:
            self._plugin_override_policy[module_namespace] = bool(allowed)

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
        try:
            mod = handler.__globals__.get("__name__", "")  # type: ignore[attr-defined]
        except AttributeError:
            return None
        if mod in self._plugin_override_policy:
            return mod
        # Also gate plugin modules currently loading but not yet policy-recorded
        # (defensive: a handler defined in the plugin namespace is plugin code).
        if isinstance(mod, str) and mod.startswith("hermes_plugins."):
            return mod
        return None

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
        profile_home: Optional[str] = None,
    ):
        """Register a tool.  Called at module-import time by each tool file.

        ``override=True`` is an explicit opt-in for plugins that intend to
        replace an existing built-in tool implementation (e.g. swap the
        default browser tool for a headed-Chrome CDP backend). Without it,
        registrations that would shadow an existing tool from a different
        toolset are rejected to prevent accidental overwrites.
        """
        with self._lock:
            requested_home = _canonical_profile_home(profile_home)
            # ``profile_home`` is an MCP ownership marker.  Do not let a
            # built-in/plugin registration accidentally become profile-scoped
            # merely because a caller forwarded unrelated context metadata.
            owner_home = (
                requested_home if requested_home and toolset.startswith("mcp-") else None
            )

            if owner_home is not None:
                # A dynamic MCP entry must never shadow a global built-in or
                # plugin entry.  Allowing both would make the public name
                # ambiguous for global callers and could route to the wrong
                # handler during a profile switch.
                if name in self._tools:
                    logger.error(
                        "Tool registration REJECTED: profile-scoped MCP tool %r "
                        "would shadow global toolset %r",
                        name,
                        self._tools[name].toolset,
                    )
                    return
                target_tools = self._profile_tools.setdefault(owner_home, {})
                existing = target_tools.get(name)
            else:
                # Preserve global built-in/plugin behavior and prevent a
                # global late registration from hiding an already-live MCP
                # owner under the same public name.
                if self._profile_entry_exists_locked(name):
                    logger.error(
                        "Tool registration REJECTED: global tool %r would "
                        "shadow a profile-scoped MCP entry",
                        name,
                    )
                    return
                target_tools = self._tools
                existing = target_tools.get(name)

            if owner_home is not None and existing and existing.toolset != toolset:
                # Dynamic MCP ownership is intentionally stricter than the
                # global plugin override path: two toolsets under one profile
                # and public name are always ambiguous, even when a caller
                # passes ``override=True``.
                logger.error(
                    "Tool registration REJECTED: '%s' (toolset '%s') would "
                    "shadow existing profile tool from toolset '%s'",
                    name,
                    toolset,
                    existing.toolset,
                )
                return
            if existing and existing.toolset != toolset:
                if override:
                    _owner = self._plugin_owner_of(handler)
                    if _owner is not None and not self._plugin_override_policy.get(_owner, False):
                        logger.error(
                            "Tool registration REJECTED: plugin %r attempted to "
                            "override built-in tool %r (existing toolset %r) without "
                            "operator opt-in. Set "
                            "plugins.entries.<plugin_id>.allow_tool_override: true "
                            "in config.yaml to allow it.",
                            _owner, name, existing.toolset,
                        )
                        raise PermissionError(
                            f"Plugin module {_owner!r} cannot override built-in "
                            f"tool {name!r} without operator opt-in "
                            f"(allow_tool_override)."
                        )
                    # Explicit opt-in (or non-plugin caller): replace the tool.
                    # Logged at INFO so the override is auditable in agent.log.
                    logger.info(
                        "Tool '%s': toolset '%s' overriding existing toolset '%s' "
                        "(override=True opt-in)",
                        name, toolset, existing.toolset,
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
                        name, toolset, existing.toolset,
                    )
                    return
            target_tools[name] = ToolEntry(
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
                profile_home=owner_home,
            )
            # Availability is now derived per-tool (_toolset_has_exposable_tools),
            # so this map no longer gates a toolset. It is still consumed by
            # get_toolset_requirements -> TOOLSET_REQUIREMENTS["check_fn"], which
            # banner.py reads (presence only, never called) to classify an
            # already-unavailable toolset as lazy-init vs disabled. Keep the
            # write path for that classification.
            if check_fn:
                if owner_home is not None:
                    self._profile_toolset_checks.setdefault(
                        (owner_home, toolset), check_fn
                    )
                elif toolset not in self._toolset_checks:
                    self._toolset_checks[toolset] = check_fn
            self._generation += 1

    def deregister(self, name: str, *, profile_home=_PROFILE_HOME_UNSET) -> None:
        """Remove a tool from the registry.

        Also cleans up the toolset check if no other tools remain in the
        same toolset.  Used by MCP dynamic tool discovery to nuke-and-repave
        when a server sends ``notifications/tools/list_changed``.

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
            explicit_profile = (
                profile_home is not _PROFILE_HOME_UNSET
                and _canonical_profile_home(profile_home) is not None
            )
            scope = _resolve_profile_home(profile_home)
            if explicit_profile:
                # An owner-scoped teardown may remove only that owner's MCP
                # entry.  It must never fall back to a global or other-profile
                # entry when its own slot is absent.
                owner_home = _canonical_profile_home(profile_home)
                if owner_home is None:  # narrowed by explicit_profile; defensive
                    return
                owner_tools = self._profile_tools.get(owner_home, {})
                entry = owner_tools.get(name)
                target_tools = owner_tools
                if entry is None:
                    return
            elif scope is not _PROFILE_SCOPE_UNRESOLVED and scope is not None:
                owner_home = scope
                owner_tools = self._profile_tools.get(owner_home, {})
                entry = owner_tools.get(name)
                if entry is not None:
                    target_tools = owner_tools
                else:
                    owner_home = None
                    target_tools = self._tools
                    entry = target_tools.get(name)
            else:
                owner_home = None
                target_tools = self._tools
                entry = target_tools.get(name)
            if entry is None:
                return
            if not entry.toolset.startswith("mcp-"):
                caller_mod = self._caller_module()
                owner = self._plugin_owner_of(entry.handler)
                # Ownership check: bind to the plugin package root
                # (``hermes_plugins.{name}``), not the exact module string.
                # A handler defined in ``hermes_plugins.pkg.handlers`` is
                # still owned by the ``hermes_plugins.pkg`` package — exact
                # string equality would wrongly block root-module cleanup code
                # from removing tools registered by a submodule of the same
                # plugin (egilewski review on #55840).
                caller_root = ".".join(caller_mod.split(".")[:2])
                owner_root = ".".join(owner.split(".")[:2]) if owner else ""
                same_plugin = bool(owner and caller_root == owner_root)
                if (
                    caller_mod.startswith("hermes_plugins.")
                    and not same_plugin
                    and not self._plugin_override_policy.get(caller_root, False)
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
            del target_tools[name]
            # Drop the toolset check and aliases if this was the last tool in
            # that owner/toolset.  Profile cleanup is deliberately scoped to
            # one home; another profile may still expose the same toolset.
            toolset_still_exists = any(
                e.toolset == entry.toolset for e in target_tools.values()
            )
            if not toolset_still_exists:
                if owner_home is not None:
                    self._profile_toolset_checks.pop(
                        (owner_home, entry.toolset), None
                    )
                    profile_aliases = self._profile_toolset_aliases.get(owner_home)
                    if profile_aliases is not None:
                        self._profile_toolset_aliases[owner_home] = {
                            alias: target
                            for alias, target in profile_aliases.items()
                            if target != entry.toolset
                        }
                        if not self._profile_toolset_aliases[owner_home]:
                            self._profile_toolset_aliases.pop(owner_home, None)
                    if not target_tools:
                        self._profile_tools.pop(owner_home, None)
                else:
                    self._toolset_checks.pop(entry.toolset, None)
                    self._toolset_aliases = {
                        alias: target
                        for alias, target in self._toolset_aliases.items()
                        if target != entry.toolset
                    }
            self._generation += 1
        logger.debug("Deregistered tool: %s", name)

    # ------------------------------------------------------------------
    # Schema retrieval
    # ------------------------------------------------------------------

    def get_definitions(
        self,
        tool_names: Set[str],
        quiet: bool = False,
        *,
        profile_home=_PROFILE_HOME_UNSET,
    ) -> List[dict]:
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
        entries_by_name = {
            entry.name: entry for entry in self._snapshot_entries(profile_home)
        }
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
        profile_home=_PROFILE_HOME_UNSET,
        **kwargs,
    ) -> str | dict:
        """Execute a tool handler by name.

        * Async handlers are bridged automatically via ``_run_async()``.
        * Handler results are normalized to a string or supported multimodal
          envelope before leaving the registry.
        * All exceptions are caught and returned as ``{"error": "..."}``
          for consistent error format.
        """
        entry = self.get_entry(name, profile_home=profile_home)
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

    def get_max_result_size(
        self,
        name: str,
        default: int | float | None = None,
        *,
        profile_home=_PROFILE_HOME_UNSET,
    ) -> int | float:
        """Return per-tool max result size, or *default* (or global default)."""
        entry = self.get_entry(name, profile_home=profile_home)
        if entry and entry.max_result_size_chars is not None:
            return entry.max_result_size_chars
        if default is not None:
            return default
        from tools.budget_config import DEFAULT_RESULT_SIZE_CHARS
        return DEFAULT_RESULT_SIZE_CHARS

    def get_all_tool_names(self, *, profile_home=_PROFILE_HOME_UNSET) -> List[str]:
        """Return sorted list of all registered tool names."""
        return sorted(entry.name for entry in self._snapshot_entries(profile_home))

    def get_schema(self, name: str, *, profile_home=_PROFILE_HOME_UNSET) -> Optional[dict]:
        """Return a tool's raw schema dict, bypassing check_fn filtering.

        Useful for token estimation and introspection where availability
        doesn't matter — only the schema content does.
        """
        entry = self.get_entry(name, profile_home=profile_home)
        return entry.schema if entry else None

    def get_toolset_for_tool(
        self, name: str, *, profile_home=_PROFILE_HOME_UNSET
    ) -> Optional[str]:
        """Return the toolset a tool belongs to, or None."""
        entry = self.get_entry(name, profile_home=profile_home)
        return entry.toolset if entry else None

    def get_profile_home_for_tool(
        self, name: str, *, profile_home=_PROFILE_HOME_UNSET
    ) -> Optional[str]:
        """Return the captured profile home for a dynamic tool, if any.

        Registry entries own this provenance. Reading it never consults the
        mutable Hermes-home context, so a later profile switch cannot make a
        previously registered tool appear to belong to another profile.
        """
        entry = self.get_entry(name, profile_home=profile_home)
        return entry.profile_home if entry else None

    def get_emoji(
        self, name: str, default: str = "⚡", *, profile_home=_PROFILE_HOME_UNSET
    ) -> str:
        """Return the emoji for a tool, or *default* if unset."""
        entry = self.get_entry(name, profile_home=profile_home)
        return (entry.emoji if entry and entry.emoji else default)

    def get_tool_to_toolset_map(
        self, *, profile_home=_PROFILE_HOME_UNSET
    ) -> Dict[str, str]:
        """Return ``{tool_name: toolset_name}`` for every registered tool."""
        return {
            entry.name: entry.toolset
            for entry in self._snapshot_entries(profile_home)
        }

    def is_toolset_available(
        self, toolset: str, *, profile_home=_PROFILE_HOME_UNSET
    ) -> bool:
        """Check if a toolset has at least one exposable tool.

        Returns False (rather than crashing) when a per-tool check raises
        an unexpected exception (e.g. network error, missing import, bad config).
        """
        if profile_home is _PROFILE_HOME_UNSET:
            entries, _ = self._snapshot_state()
        else:
            entries, _ = self._snapshot_state(profile_home)
        return self._toolset_has_exposable_tools(toolset, entries)

    def check_toolset_requirements(
        self, *, profile_home=_PROFILE_HOME_UNSET
    ) -> Dict[str, bool]:
        """Return ``{toolset: available_bool}`` for every toolset."""
        if profile_home is _PROFILE_HOME_UNSET:
            entries, _ = self._snapshot_state()
        else:
            entries, _ = self._snapshot_state(profile_home)
        toolsets = sorted({entry.toolset for entry in entries})
        return {
            toolset: self._toolset_has_exposable_tools(toolset, entries)
            for toolset in toolsets
        }

    def get_available_toolsets(
        self, *, profile_home=_PROFILE_HOME_UNSET
    ) -> Dict[str, dict]:
        """Return toolset metadata for UI display."""
        toolsets: Dict[str, dict] = {}
        if profile_home is _PROFILE_HOME_UNSET:
            entries, _ = self._snapshot_state()
        else:
            entries, _ = self._snapshot_state(profile_home)
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

    def get_toolset_requirements(
        self, *, profile_home=_PROFILE_HOME_UNSET
    ) -> Dict[str, dict]:
        """Build a TOOLSET_REQUIREMENTS-compatible dict for backward compat."""
        result: Dict[str, dict] = {}
        if profile_home is _PROFILE_HOME_UNSET:
            entries, toolset_checks = self._snapshot_state()
        else:
            entries, toolset_checks = self._snapshot_state(profile_home)
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

    def check_tool_availability(
        self, quiet: bool = False, *, profile_home=_PROFILE_HOME_UNSET
    ):
        """Return (available_toolsets, unavailable_info) like the old function."""
        available = []
        unavailable = []
        if profile_home is _PROFILE_HOME_UNSET:
            entries, _ = self._snapshot_state()
        else:
            entries, _ = self._snapshot_state(profile_home)
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
