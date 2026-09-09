"""Central registry for all hermes-agent tools: each tool file calls ``registry.register()``
at import to declare schema, handler, toolset membership and availability check;
``model_tools.py`` queries the registry instead of keeping parallel data structures.
Cycle-safe import chain: this module imports nothing from model_tools or tool files;
tools/*.py import it at module level; model_tools.py imports both; run_agent/cli import
model_tools."""

import ast
import functools
import importlib
import json
import logging
import sys
import threading
import time
from dataclasses import dataclass
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
        len(text), text[:_MAX_LOGGED_ERROR_CHARS])
    return text[:_MAX_TOOL_ERROR_CHARS] + _TOOL_ERROR_TRUNCATION_MARKER


def _bound_json_error_result(result: str) -> str:
    """Trim an oversized ``error`` field in a JSON string result: handlers that
    ``json.dumps({"error": str(exc)})`` directly bypass ``tool_error``'s cap, so this runs
    at the dispatch boundary to stop unbounded errors stacking across retries."""
    if len(result) <= _MAX_TOOL_ERROR_CHARS or '"error"' not in result:
        return result
    try:
        payload = json.loads(result)
    except ValueError:
        return result
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, str) or len(error) <= _MAX_TOOL_ERROR_CHARS:
        return result
    payload["error"] = _bound_error_text(error)
    return json.dumps(payload, ensure_ascii=False)


def _is_registry_register_call(node: ast.AST) -> bool:
    """True when *node* is a ``registry.register(...)`` call expression."""
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    func = node.value.func
    return (
        isinstance(func, ast.Attribute) and func.attr == "register"
        and isinstance(func.value, ast.Name) and func.value.id == "registry")


def _module_registers_tools(module_path: Path) -> bool:
    """True when the module body (or a module-level ``for``) calls ``registry.register(...)``.
    Only module-body statements count, so helpers registering inside a function are skipped;
    a text prefilter avoids ``ast.parse`` for files lacking both words."""
    try:
        source = module_path.read_text(encoding="utf-8")
        if "registry" not in source or "register" not in source:
            return False
        tree = ast.parse(source, filename=str(module_path))
    except (OSError, SyntaxError):
        return False
    # Table-driven modules register several tools from one loop, still at import time.
    return any(
        _is_registry_register_call(stmt)
        or (isinstance(stmt, ast.For) and any(_is_registry_register_call(s) for s in stmt.body))
        for stmt in tree.body)


def discover_builtin_tools(tools_dir: Optional[Path] = None) -> List[str]:
    """Import built-in self-registering tool modules and return their module names. The
    per-file AST scan costs ~145 ms over ~100 files, so verdicts are memoized on disk keyed
    by ``(mtime_ns, size)``; a mismatch or corrupt cache re-scans that file. The write is
    best-effort and atomic, so concurrent processes race harmlessly."""
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
        if isinstance(cached, (list, tuple)) and len(cached) == 3 and tuple(cached[:2]) == stat_key:
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
        # Deferred import keeps tools/registry.py a no-deps leaf at import time.
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


@dataclass(eq=False, slots=True)
class ToolEntry:
    """Metadata for one registered tool (identity semantics: restore/CAS paths compare ``is``)."""

    name: str
    toolset: str
    schema: dict
    handler: Callable
    check_fn: Optional[Callable]
    requires_env: list
    is_async: bool
    description: str
    emoji: str
    max_result_size_chars: int | float | None = None
    # Zero-arg callable whose dict is shallow-merged onto the schema at every get_definitions()
    # — for fields tracking runtime config (delegate_task's description reflects limits).
    dynamic_schema_overrides: Optional[Callable] = None


class _PluginOverridePolicy:
    """Identity-bearing authorization record for one plugin generation."""

    __slots__ = ("allowed",)

    def __init__(self, allowed: bool) -> None:
        self.allowed = bool(allowed)


_OVERRIDE_DENIED_MSG = (
    "Plugin module {owner!r} cannot override built-in tool {name!r} "
    "without operator opt-in (allow_tool_override).")


# ---- check_fn TTL cache ----------------------------------------------------
# check_fns probe external state (Docker, Modal SDK, playwright) that changes on human
# timescales, so results are cached ~30 s: env-var flips via ``hermes tools`` still land
# within a turn or two. Transient-failure suppression: a flapping probe (``docker version``
# timing out under load) would silently strip a whole toolset from the agent being built —
# most visibly a subagent reporting "Tool read_file does not exist" — so a failure within a
# short grace window of the last success serves the last-good True WITHOUT caching it; a
# failure persisting past the window is honored so a dead backend stops advertising tools.

_CHECK_FN_TTL_SECONDS = 30.0
# Grace window after a success in which a failure counts as a flake; kept short
# so a genuinely-down backend is reflected within a couple of turns.
_CHECK_FN_FAILURE_GRACE_SECONDS = 60.0
_CHECK_FN_CACHE_MAX = 512
_check_fn_cache: Dict[tuple[Callable, Optional[str]], tuple[float, bool]] = {}
_check_fn_last_good: Dict[tuple[Callable, Optional[str]], float] = {}
_check_fn_cache_lock = threading.Lock()
CHECK_FN_CACHE_BYPASS = ""
_NO_CACHE_CHECK_FNS: Set[Callable] = set()
_BROWSER_IDENTITY_KEYS = (
    "HERMES_SESSION_ID",
    "HERMES_BROWSER_CONTROL_PRINCIPAL",
    "HERMES_BROWSER_CONTROL_TRANSPORT_FAMILY")


def no_cache_check_fn(fn: Callable) -> Callable:
    """Mark a local, config-backed availability check as uncached."""
    _NO_CACHE_CHECK_FNS.add(fn)
    return fn


def _fn_label(fn: Callable) -> object:
    return getattr(fn, "__qualname__", fn)


def _prune_check_fn_caches(now: float) -> None:
    """Expire stale entries and cap profile-dimensional cache growth. Caller holds the lock."""
    for cache, ttl, stamp in (
        (_check_fn_cache, _CHECK_FN_TTL_SECONDS, lambda v: v[0]),
        (_check_fn_last_good, _CHECK_FN_FAILURE_GRACE_SECONDS, lambda v: v)):
        for key, value in list(cache.items()):
            if now - stamp(value) >= ttl:
                cache.pop(key, None)
        while len(cache) >= _CHECK_FN_CACHE_MAX:
            cache.pop(next(iter(cache)))


def check_fn_cache_scope() -> Optional[str]:
    """Return the active profile key when availability is profile-scoped. Browser-controller
    availability is request-bound (changes on every attach/detach), so a fully bound
    browser-control request bypasses this cache AND model_tools' outer definition cache (same
    sentinel) — one Browser session's live tools must not leak into another. Single-profile
    processes keep the process-wide cache; a multiplex gateway installs a Hermes-home override
    per profile turn, so the canonical profile key is the boundary."""
    try:
        from gateway.session_context import get_session_env
        if all(str(get_session_env(k, "") or "").strip() for k in _BROWSER_IDENTITY_KEYS):
            return CHECK_FN_CACHE_BYPASS
    except Exception:
        pass
    try:
        from agent.secret_scope import is_multiplex_active
        from hermes_constants import get_hermes_home_override
        if not is_multiplex_active():
            return None
        override = get_hermes_home_override()
        return str(Path(override).expanduser().resolve()) if override else CHECK_FN_CACHE_BYPASS
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
            # Expected fail-closed probe: with multiplexing on, boot-time check_fns run before
            # any profile secret scope exists, so get_secret raises by design. No traceback,
            # so it isn't mistaken for a crashed check_fn.
            logger.debug(
                # The tool re-probes on the first scoped turn — log without a traceback so this cannot be
                # mistaken for a crashed check_fn (#100697).
                "check_fn %s hit the multiplex fail-closed path with no "
                "profile secret scope active; dependent tools re-probe on the first scoped turn",
                _fn_label(fn))
        else:
            # The scope resolved but the read still failed closed: a genuinely lost scope.
            logger.warning(
                "check_fn %s raised UnscopedSecretError while the profile cache "
                "scope was resolved; dependent tools will be unavailable this turn",
                _fn_label(fn), exc_info=True)
    except Exception:
        detail = " while profile cache scope was unresolved" if unresolved_scope else ""
        logger.warning(
            "check_fn %s raised%s; dependent tools will be unavailable this turn",
            _fn_label(fn), detail, exc_info=True)
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
        _prune_check_fn_caches(now)  # leaves only entries within TTL
        cached = _check_fn_cache.get(cache_key)
        if cached is not None:
            return cached[1]
    try:
        value, outcome = bool(fn()), "returned False"
    except Exception:
        value, outcome = False, "raised"
    with _check_fn_cache_lock:
        _prune_check_fn_caches(now)
        if value:
            _check_fn_last_good[cache_key] = now
            _check_fn_cache[cache_key] = (now, True)
            return True
        last_good = _check_fn_last_good.get(cache_key)
        if last_good is not None and now - last_good < _CHECK_FN_FAILURE_GRACE_SECONDS:
            # Recent success → flake: serve last-good True, do NOT cache (next call re-probes).
            logger.warning(
                "check_fn %s failed (%s) within %.0fs of last success; "
                "treating as transient and keeping tool(s) available",
                _fn_label(fn), outcome, _CHECK_FN_FAILURE_GRACE_SECONDS)
            return True

        # No recent success (or grace expired) — honor the failure; logged so silent tool
        # loss in quiet mode (subagents) is diagnosable.
        logger.warning(
            "check_fn %s %s; dependent tools will be unavailable this turn", _fn_label(fn), outcome)
        _check_fn_cache[cache_key] = (now, False)
        return False


def _memo_check(fn: Callable, memo: Dict[Callable, bool]) -> bool:
    """Per-pass memo on top of the TTL cache: one probe per distinct check_fn."""
    if fn not in memo:
        memo[fn] = _check_fn_cached(fn)
    return memo[fn]


def invalidate_check_fn_cache() -> None:
    """Drop all cached ``check_fn`` results (after config changes like ``hermes tools enable``)."""
    with _check_fn_cache_lock:
        _check_fn_cache.clear()
        _check_fn_last_good.clear()


def get_cached_check_fn_result(fn: Callable) -> Optional[bool]:
    """Cached verdict for *fn* if its TTL is still valid, else None. NEVER runs the probe:
    for read-only surfaces (dashboard panels) that must not do network/auth/SDK work."""
    now = time.monotonic()
    scope = check_fn_cache_scope()
    if scope == CHECK_FN_CACHE_BYPASS:
        # Unresolved profile identity bypasses the cache; nothing trustworthy to report.
        return None
    with _check_fn_cache_lock:
        cached = _check_fn_cache.get((fn, scope))
    return cached[1] if cached is not None and now - cached[0] < _CHECK_FN_TTL_SECONDS else None


class ToolRegistry:
    """Singleton registry that collects tool schemas + handlers from tool files."""

    def __init__(self):
        self._tools: Dict[str, ToolEntry] = {}  # built-in / process-global registrations
        # Plugin overlays keyed by resolved HERMES_HOME; a profile sees its overlay first.
        self._scoped_tools: Dict[str, Dict[str, ToolEntry]] = {}
        # Plugin namespace -> operator opt-in for built-in override (lifecycle-managed);
        # scope attribution stays durable after policy removal so delayed callbacks
        # remain confined to the profile that loaded them.
        self._plugin_override_policy: Dict[tuple[Optional[str], str], _PluginOverridePolicy] = {}
        self._plugin_module_scopes: Dict[str, Set[Optional[str]]] = {}
        self._toolset_checks: Dict[str, Callable] = {}
        self._toolset_aliases: Dict[str, str] = {}
        # MCP refresh mutates while other threads read: serialize writes, snapshot reads.
        self._lock = threading.RLock()
        # Bumped on every mutation; get_tool_definitions memoizes against it.
        self._generation: int = 0

    @staticmethod
    def current_scope_key() -> str:
        return hermes_home_key()

    @staticmethod
    def _grouped(entries: List[ToolEntry]) -> Dict[str, List[ToolEntry]]:
        """``{toolset: entries}`` in first-appearance order."""
        groups: Dict[str, List[ToolEntry]] = {}
        for entry in entries:
            groups.setdefault(entry.toolset, []).append(entry)
        return groups

    def _slot(self, scope: Optional[str], *, create: bool = False) -> Dict[str, ToolEntry]:
        """The registration map for *scope*: global when None, else that profile's overlay."""
        if scope is None:
            return self._tools
        if create:
            return self._scoped_tools.setdefault(scope, {})
        return self._scoped_tools.get(scope, {})

    def _drop_toolset_aliases(self, toolset: str) -> None:
        self._toolset_aliases = {
            alias: target for alias, target in self._toolset_aliases.items() if target != toolset}

    def _merged_tools(self, scope: Optional[str] = None) -> Dict[str, ToolEntry]:
        """Return global tools overlaid with one profile's plugin tools."""
        return {**self._tools, **self._scoped_tools.get(scope or self.current_scope_key(), {})}

    def _toolset_entries(self, toolset: str, scope: Optional[str]) -> List[ToolEntry]:
        return self._grouped(self._merged_tools(scope).values()).get(toolset, [])

    def _snapshot_state(
        self, scope: Optional[str] = None) -> tuple[List[ToolEntry], Dict[str, Callable]]:
        """Return a coherent snapshot of registry entries and toolset checks."""
        with self._lock:
            entries = list(self._merged_tools(scope).values())
            checks = dict(self._toolset_checks)
            checks.update({e.toolset: e.check_fn for e in entries if e.check_fn is not None})
            return entries, checks

    def _snapshot_entries(self) -> List[ToolEntry]:
        return self._snapshot_state()[0]

    def _toolset_has_exposable_tools(self, toolset: str, entries: List[ToolEntry]) -> bool:
        """True when at least one tool in *toolset* would be exposed. Mirrors
        :meth:`get_definitions` per-tool filtering so doctor/banners agree with runtime:
        mixed toolsets (``terminal`` + desktop-only ``read_terminal``) must not be gated
        by the first ``check_fn``."""
        memo: Dict[Callable, bool] = {}
        members = (e for e in entries if e.toolset == toolset)
        return any(not e.check_fn or _memo_check(e.check_fn, memo) for e in members)

    def get_entry(self, name: str, *, scope: Optional[str] = None) -> Optional[ToolEntry]:
        """Active profile's entry by name, falling back to global."""
        with self._lock:
            return self._merged_tools(scope).get(name)

    def snapshot_registration(
        self, name: str, *, scope: Optional[str] = None) -> Optional[ToolEntry]:
        """Local slot state only — no global fallback."""
        with self._lock:
            return self._slot(scope).get(name)

    def get_registered_toolset_names(self) -> List[str]:
        return sorted(self._grouped(self._snapshot_entries()))

    def get_all_entries(self) -> List[ToolEntry]:
        return self._snapshot_entries()

    def get_tool_names_for_toolset(self, toolset: str) -> List[str]:
        return sorted(e.name for e in self._grouped(self._snapshot_entries()).get(toolset, []))

    def register_toolset_alias(self, alias: str, toolset: str) -> None:
        """Register an explicit alias for a canonical toolset name."""
        with self._lock:
            existing = self._toolset_aliases.get(alias)
            if existing and existing != toolset:
                logger.warning(
                    "Toolset alias collision: '%s' (%s) overwritten by %s", alias, existing, toolset
                )
            self._toolset_aliases[alias] = toolset
            self._generation += 1

    def get_registered_toolset_aliases(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._toolset_aliases)

    def get_toolset_alias_target(self, alias: str) -> Optional[str]:
        with self._lock:
            return self._toolset_aliases.get(alias)

    # ---- Registration ------------------------------------------------

    def register_plugin_override_policy(
        self, module_namespace: str, allowed: bool, *, scope: Optional[str] = None,
    ) -> _PluginOverridePolicy:
        """Bind a plugin module namespace to its current operator opt-in. The identity-bearing
        result lets unload/reload revoke a stale authorization without losing attribution."""
        with self._lock:
            policy = _PluginOverridePolicy(allowed)
            self._plugin_override_policy[(scope, module_namespace)] = policy
            self._plugin_module_scopes.setdefault(module_namespace, set()).add(scope)
            return policy

    def snapshot_plugin_override_policy(
        self, module_namespace: str, *, scope: Optional[str] = None,
    ) -> Optional[_PluginOverridePolicy]:
        """Return one local authorization generation without fallback."""
        with self._lock:
            return self._plugin_override_policy.get((scope, module_namespace))

    def restore_plugin_override_policy(
        self, module_namespace: str, current: _PluginOverridePolicy,
        previous: Optional[_PluginOverridePolicy], *, scope: Optional[str] = None) -> bool:
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

    def _plugin_override_allowed(self, scope: Optional[str], module_namespace: str) -> bool:
        policy = self._plugin_override_policy.get((scope, module_namespace))
        if policy is None and scope is not None:
            policy = self._plugin_override_policy.get((None, module_namespace))
        return bool(policy and policy.allowed)

    def _plugin_owner_of(self, handler: Callable) -> Optional[str]:
        """Plugin namespace that DEFINED *handler* (None for built-in/MCP handlers). Bound to
        ``handler.__globals__["__name__"]``, fixed at definition time so it cannot drift with
        call site/thread/timing; lambdas and nested functions inherit it, so a plugin cannot
        launder an override via a callback."""
        mod = self._callable_module(handler)
        return self._plugin_namespace_of_module(mod) if mod else None

    @staticmethod
    def _callable_module(handler: Callable) -> str:
        """Resolve defining module through wrappers, partials, and objects."""
        current = handler
        seen: Set[int] = set()
        while id(current) not in seen:
            seen.add(id(current))
            globals_dict = getattr(current, "__globals__", None)
            if isinstance(current, functools.partial):
                current = current.func
            elif getattr(current, "__func__", None) is not None:
                current = current.__func__
            elif isinstance(globals_dict, dict) and globals_dict.get("__name__", ""):
                return str(globals_dict["__name__"])
            elif getattr(current, "__wrapped__", None) is not None:
                current = current.__wrapped__
            else:
                break
        module_name = getattr(current, "__module__", "")
        return str(module_name or getattr(type(current), "__module__", "") or "")

    def _plugin_namespace_of_module(self, module_namespace: str) -> Optional[str]:
        """Resolve a module/submodule to its durable plugin namespace."""
        with self._lock:
            matches = [
                namespace for namespace in self._plugin_module_scopes
                if module_namespace == namespace or module_namespace.startswith(f"{namespace}.")]
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
                f"Plugin module {module_namespace!r} is active in multiple profiles and cannot "
                "register outside one of those scopes.")

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
        """Best-effort module name of the registry method's caller (two frames up).
        ``deregister()`` takes only a tool name — no handler for ``_plugin_owner_of`` —
        so frame inspection is the only way to know who is asking."""
        try:
            return sys._getframe(2).f_globals.get("__name__", "") or ""
        except Exception:
            return ""

    def register(
        self, name: str, toolset: str, schema: dict, handler: Callable,
        check_fn: Callable = None, requires_env: list = None, is_async: bool = False,
        description: str = "", emoji: str = "", max_result_size_chars: int | float | None = None,
        dynamic_schema_overrides: Callable = None, override: bool = False,
        scope: Optional[str] = None):
        """Register a tool (called at import time by each tool file). ``override=True`` is an
        explicit opt-in for plugins replacing a built-in implementation (e.g. a headed-Chrome
        browser backend); without it, cross-toolset shadowing is rejected."""
        handler_owner = self._plugin_owner_of(handler)
        caller_owner = self._plugin_namespace_of_module(self._caller_module())
        owner = caller_owner or handler_owner
        if scope is None and owner is not None:
            scope = self._plugin_scope_of(owner)
        with self._lock:
            target = self._slot(scope, create=True)
            existing = (self._tools if scope is None else self._merged_tools(scope)).get(name)
            plugin_override_denied = (
                owner is not None and not self._plugin_override_allowed(scope, owner))
            shadows_global = (
                owner is not None and scope is not None
                and name not in target and name in self._tools)
            if shadows_global:
                if not override:
                    logger.error(
                        "Tool registration REJECTED: plugin %r attempted to shadow global tool %r "
                        "without override=True", owner, name)
                    return
                if plugin_override_denied:
                    raise PermissionError(_OVERRIDE_DENIED_MSG.format(owner=owner, name=name))
            if existing and existing.toolset != toolset:
                if override:
                    if plugin_override_denied:
                        logger.error(
                            "Tool registration REJECTED: plugin %r attempted to override built-in "
                            "tool %r (existing toolset %r) without operator opt-in. Set "
                            "plugins.entries.<plugin_id>.allow_tool_override: true in config.yaml "
                            "to allow it.",
                            owner, name, existing.toolset)
                        raise PermissionError(_OVERRIDE_DENIED_MSG.format(owner=owner, name=name))
                    # Explicit opt-in (or non-plugin caller): INFO so the override is auditable.
                    logger.info(
                        "Tool '%s': toolset '%s' overriding existing toolset '%s' "
                        "(override=True opt-in)", name, toolset, existing.toolset)
                else:
                    # Reject every cross-toolset shadow (incl. MCP-to-MCP); same-toolset
                    # re-registration (MCP reconnect/refresh) stays allowed.
                    logger.error(
                        "Tool registration REJECTED: '%s' (toolset '%s') would shadow existing "
                        "tool from toolset '%s'. Pass override=True to register() if the "
                        "replacement is intentional, or deregister the existing tool first.",
                        name, toolset, existing.toolset)
                    return
            target[name] = ToolEntry(
                name=name, toolset=toolset, schema=schema, handler=handler, check_fn=check_fn,
                requires_env=requires_env or [], is_async=is_async,
                description=description or schema.get("description", ""), emoji=emoji,
                max_result_size_chars=max_result_size_chars,
                dynamic_schema_overrides=dynamic_schema_overrides)
            # Availability is derived per-tool (_toolset_has_exposable_tools), so this map no
            # longer gates a toolset; it still feeds get_toolset_requirements ->
            # TOOLSET_REQUIREMENTS["check_fn"], which banner.py reads (presence only,
            # never called) to classify an unavailable toolset as lazy-init vs disabled.
            if scope is None and check_fn and toolset not in self._toolset_checks:
                self._toolset_checks[toolset] = check_fn
            self._generation += 1

    def deregister(self, name: str, *, scope: Optional[str] = None) -> None:
        """Remove a tool; drops the toolset check/aliases if it was the last in its toolset.

        ``scope`` selects a profile overlay explicitly (multiplexed MCP tools live in the
        owning profile's overlay); plugin callers may not name another scope, non-plugin
        callers default to the process-global map. Gated by the same opt-in as
        ``register(override=True)``, else a plugin could deregister a tool it doesn't own
        and re-register over the empty slot (the override check only runs when an entry
        exists). ``mcp-*`` toolsets are exempt — discovery repaves its own tools per refresh."""
        with self._lock:
            caller_mod = self._caller_module()
            caller_owner = self._plugin_namespace_of_module(caller_mod)
            caller_scope = self._plugin_scope_of(caller_owner) if caller_owner is not None else None
            if caller_owner is not None and scope is not None and scope != caller_scope:
                raise PermissionError(
                    f"Plugin module {caller_mod!r} cannot deregister tools "
                    "outside its own profile scope.")
            if scope is None:
                scope = caller_scope
            target = self._slot(scope)
            entry = target.get(name)
            if entry is None:
                if scope is not None and caller_owner is not None and name in self._tools:
                    raise PermissionError(
                        f"Scoped plugin module {caller_mod!r} cannot deregister process-global "
                        f"tool {name!r}; register a scoped override instead.")
                return
            if not entry.toolset.startswith("mcp-"):
                owner = self._plugin_owner_of(entry.handler)
                # Ownership binds to the plugin package root (``hermes_plugins.{name}``), not
                # the exact module: a submodule's handler is still the package's to remove.
                # A handler defined in ``hermes_plugins.pkg.handlers`` is still owned by the
                # ``hermes_plugins.pkg`` package — exact string equality would wrongly block root-module
                # cleanup code from removing tools registered by a submodule of the same plugin (egilewski
                # review on #55840).
                same_plugin = bool(owner and caller_owner == owner)
                if (
                    caller_owner is not None
                    and not same_plugin
                    and not self._plugin_override_allowed(caller_scope, caller_owner)):
                    logger.error(
                        "Tool deregistration REJECTED: plugin %r attempted to "
                        "remove tool %r (toolset %r) it does not own, without operator opt-in. Set "
                        "plugins.entries.%s.allow_tool_override: true in config.yaml to allow it.",
                        caller_mod, name, entry.toolset, caller_mod)
                    raise PermissionError(
                        f"Plugin module {caller_mod!r} cannot deregister tool {name!r} (toolset "
                        f"{entry.toolset!r}) without operator opt-in (allow_tool_override).")
            del target[name]
            if scope is not None and not target:
                self._scoped_tools.pop(scope, None)
            if not self._toolset_entries(entry.toolset, scope):
                self._toolset_checks.pop(entry.toolset, None)
                self._drop_toolset_aliases(entry.toolset)
            self._generation += 1
        logger.debug("Deregistered tool: %s", name)

    def restore_registration(
        self, name: str, current: ToolEntry, previous: Optional[ToolEntry], *,
        scope: Optional[str] = None) -> bool:
        """Restore a host-owned registration if it is still current (plugin ownership ledger).
        The identity check is deliberate: another plugin (or ``PluginManager`` in a
        multi-profile process) may have registered a newer entry under the same name, and
        unloading this entry must leave that newer one untouched."""
        with self._lock:
            target = self._slot(scope, create=True)
            if target.get(name) is not current:
                return False
            if previous is None:
                target.pop(name, None)
            else:
                target[name] = previous
            if scope is not None and not target:
                self._scoped_tools.pop(scope, None)

            # Rebuild affected toolset checks from survivors: a plugin may have replaced an
            # entry in the same toolset, so its check_fn would otherwise linger after restore.
            affected_toolsets = {current.toolset}
            if previous is not None:
                affected_toolsets.add(previous.toolset)
            for toolset in affected_toolsets:
                surviving = self._toolset_entries(toolset, scope)
                check_fn = next((entry.check_fn for entry in surviving if entry.check_fn), None)
                if scope is None:
                    if check_fn is None:
                        self._toolset_checks.pop(toolset, None)
                    else:
                        self._toolset_checks[toolset] = check_fn
                in_overlays = (e for m in self._scoped_tools.values() for e in m.values())
                if not surviving and not any(e.toolset == toolset for e in in_overlays):
                    self._drop_toolset_aliases(toolset)
            self._generation += 1
        logger.debug("Restored tool registration: %s", name)
        return True

    # ---- Schema retrieval --------------------------------------------

    def get_definitions(self, tool_names: Set[str], quiet: bool = False) -> List[dict]:
        """OpenAI-format schemas for the requested tools whose ``check_fn`` passes (or is
        absent). Probes use the ~30 s TTL cache so ``hermes tools enable`` lands quickly."""
        result = []
        check_results: Dict[Callable, bool] = {}
        entries_by_name = {entry.name: entry for entry in self._snapshot_entries()}
        for name in sorted(tool_names):
            entry = entries_by_name.get(name)
            if not entry:
                continue
            if entry.check_fn and not _memo_check(entry.check_fn, check_results):
                if not quiet:
                    logger.debug("Tool %s unavailable (check failed)", name)
                continue
            schema_with_name = {**entry.schema, "name": entry.name}
            # Runtime-dynamic overrides (e.g. delegate_task limits); the caller's memo is
            # keyed on config.yaml mtime+size, so config changes invalidate it automatically.
            if entry.dynamic_schema_overrides is not None:
                try:
                    overrides = entry.dynamic_schema_overrides()
                except Exception as exc:
                    overrides = None
                    logger.warning(
                        "dynamic_schema_overrides for tool %s raised %s; using static schema",
                        name, exc)
                if isinstance(overrides, dict):
                    schema_with_name.update(overrides)
            result.append({"type": "function", "function": schema_with_name})
        return result

    # ---- Dispatch ----------------------------------------------------

    @staticmethod
    def _normalize_handler_result(name: str, result):
        """Results must be a string or the multimodal envelope; anything else becomes a
        string error so logging/hooks/budgeting/persistence never receive values they
        cannot slice or size."""
        if isinstance(result, str):
            return _bound_json_error_result(result)
        if (
            isinstance(result, dict)
            and result.get("_multimodal") is True
            and isinstance(result.get("content"), list)):
            return result
        result_type = type(result).__name__
        logger.error("Tool %s handler returned unsupported result type: %s", name, result_type)
        return tool_error(
            f"Tool handler returned unsupported result type: {result_type}",
            error_type="tool_result_contract", tool=name, result_type=result_type)

    def dispatch(
        self, name: str, args: dict, *, scope: Optional[str] = None, **kwargs) -> str | dict:
        """Execute a tool handler by name: async handlers bridged via ``_run_async()``,
        results normalized, every exception returned as ``{"error": ...}``."""
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
            logger.exception("Tool %s dispatch error: %s", name, _bound_error_text(str(e)))
            # Sanitize so framing tokens/CDATA/fences in exception text aren't structural noise.
            raw = f"Tool execution failed: {type(e).__name__}: {e}"
            try:
                from model_tools import _sanitize_tool_error
                sanitized = _sanitize_tool_error(raw)
            except Exception:
                sanitized = raw  # defensive: never let the sanitizer block error propagation
            return tool_error(sanitized)

    # ---- Query helpers -----------------------------------------------

    def _attr(self, name: str, attr: str):
        return getattr(self.get_entry(name), attr, None)

    def get_max_result_size(self, name: str, default: int | float | None = None) -> int | float:
        """Return per-tool max result size, or *default* (or global default)."""
        size = self._attr(name, "max_result_size_chars")
        if size is not None:
            return size
        if default is not None:
            return default
        from tools.budget_config import DEFAULT_RESULT_SIZE_CHARS
        return DEFAULT_RESULT_SIZE_CHARS

    def get_all_tool_names(self) -> List[str]:
        return sorted(entry.name for entry in self._snapshot_entries())

    def get_schema(self, name: str) -> Optional[dict]:
        """Raw schema dict, bypassing check_fn filtering (token estimates, introspection)."""
        return self._attr(name, "schema")

    def get_toolset_for_tool(self, name: str) -> Optional[str]:
        return self._attr(name, "toolset")

    def get_emoji(self, name: str, default: str = "⚡") -> str:
        """Return the emoji for a tool, or *default* if unset."""
        return self._attr(name, "emoji") or default

    def get_tool_to_toolset_map(self) -> Dict[str, str]:
        return {entry.name: entry.toolset for entry in self._snapshot_entries()}

    def is_toolset_available(self, toolset: str) -> bool:
        """True when a toolset has at least one exposable tool (never raises)."""
        return self._toolset_has_exposable_tools(toolset, self._snapshot_entries())


    def check_toolset_requirements(self) -> Dict[str, bool]:
        entries = self._snapshot_entries()
        return {
            toolset: self._toolset_has_exposable_tools(toolset, entries)
            for toolset in sorted(self._grouped(entries))}

    def get_available_toolsets(self) -> Dict[str, dict]:
        """Return toolset metadata for UI display."""
        entries = self._snapshot_entries()
        toolsets: Dict[str, dict] = {}
        for toolset, members in self._grouped(entries).items():
            toolsets[toolset] = {
                "available": self._toolset_has_exposable_tools(toolset, entries),
                "tools": [entry.name for entry in members],
                "description": "",
                "requirements": _unique_env(members)}
        return toolsets

    def get_toolset_requirements(self) -> Dict[str, dict]:
        """Build a TOOLSET_REQUIREMENTS-compatible dict for backward compat."""
        entries, toolset_checks = self._snapshot_state()
        result: Dict[str, dict] = {}
        for toolset, members in self._grouped(entries).items():
            result[toolset] = {
                "name": toolset,
                "env_vars": _unique_env(members),
                "check_fn": toolset_checks.get(toolset),
                "setup_url": None,
                "tools": [entry.name for entry in members]}
        return result

    def check_tool_availability(self, quiet: bool = False):
        """Return (available_toolsets, unavailable_info) like the old function."""
        available, unavailable = [], []
        entries = self._snapshot_entries()
        groups = self._grouped(entries)
        for ts in sorted(groups):
            if self._toolset_has_exposable_tools(ts, entries):
                available.append(ts)
            else:
                unavailable.append({
                    "name": ts, "env_vars": groups[ts][0].requires_env,
                    "tools": [entry.name for entry in groups[ts]]})
        return available, unavailable


def _unique_env(entries: List[ToolEntry]) -> list:
    """Union of ``requires_env`` across *entries*, first-seen order, no duplicates."""
    out: list = []
    for entry in entries:
        out.extend(v for v in (entry.requires_env or []) if v not in out)
    return out


# Module-level singleton
registry = ToolRegistry()


# Tool handlers must return JSON strings; these replace the ubiquitous
# ``json.dumps({"error": msg}, ensure_ascii=False)`` boilerplate.


def tool_error(message, **extra) -> str:
    """``'{"error": "<message>", **extra}'`` — the error body is bounded so a raw
    exception can't bloat history across retries."""
    return json.dumps({"error": _bound_error_text(str(message)), **extra}, ensure_ascii=False)


def tool_result(data=None, **kwargs) -> str:
    """JSON-encode a dict positional arg *or* keyword arguments (not both)."""
    return json.dumps(data if data is not None else kwargs, ensure_ascii=False)
