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
import copy
import importlib
import json
import logging
import sys
import threading
import time
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)
_MISSING = object()


# The profile-scope module is introduced by the integration slice. Keep these
# imports lazy so this independently-testable slice remains single-profile
# compatible until that file is merged, while consuming its exact public API.
_FALLBACK_PROFILE_KEY = "__hermes_launch_profile__"


def _plugin_namespace_root(module_name: str) -> str:
    """Return the plugin package root for legacy and profile-qualified modules."""
    parts = module_name.split(".")
    if len(parts) < 2:
        return module_name
    profile_component = parts[1]
    profile_hash = profile_component.removeprefix("profile_")
    if (
        len(parts) >= 3
        and profile_component.startswith("profile_")
        and len(profile_hash) == 16
        and all(char in "0123456789abcdef" for char in profile_hash)
    ):
        return ".".join(parts[:3])
    return ".".join(parts[:2])


def _capture_profile_key(profile_key=None):
    try:
        from agent.plugin_profile_scope import freeze_profile_key

        return freeze_profile_key(profile_key)
    except ImportError:
        return str(profile_key) if profile_key is not None else _FALLBACK_PROFILE_KEY


_check_fn_profile_generations: Dict[object, int] = {}


def _check_fn_profile_generation(profile_key=None) -> int:
    return _check_fn_profile_generations.get(_capture_profile_key(profile_key), 0)


def _bump_check_fn_profile_generation(profile_key=None) -> int:
    key = _capture_profile_key(profile_key)
    generation = _check_fn_profile_generations.get(key, 0) + 1
    _check_fn_profile_generations[key] = generation
    return generation


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

    def snapshot(self) -> "ToolEntry":
        """Freeze mutable metadata while preserving executable callables."""
        return ToolEntry(
            name=self.name,
            toolset=self.toolset,
            schema=copy.deepcopy(self.schema),
            handler=self.handler,
            check_fn=self.check_fn,
            requires_env=list(self.requires_env),
            is_async=self.is_async,
            description=self.description,
            emoji=self.emoji,
            max_result_size_chars=self.max_result_size_chars,
            dynamic_schema_overrides=self.dynamic_schema_overrides,
        )


class ToolProfileSnapshot:
    """Immutable-by-convention tool authority captured for one live manager."""

    __slots__ = (
        "registry", "profile_key", "tools", "override_policies",
        "toolset_checks", "toolset_aliases", "generation",
    )

    def __init__(
        self,
        *,
        registry: "ToolRegistry",
        profile_key,
        tools: Dict[str, ToolEntry],
        override_policies: Dict[str, bool],
        toolset_checks: Dict[str, Callable],
        toolset_aliases: Dict[str, str],
        generation: int,
    ) -> None:
        self.registry = registry
        self.profile_key = profile_key
        self.tools = tools
        self.override_policies = override_policies
        self.toolset_checks = toolset_checks
        self.toolset_aliases = toolset_aliases
        self.generation = generation


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
_check_fn_cache: Dict[tuple[str, int, Callable], tuple[float, bool]] = {}
# Monotonic timestamp of the most recent True result per check_fn.
_check_fn_last_good: Dict[tuple[str, Callable], float] = {}
_check_fn_cache_lock = threading.Lock()


def _check_fn_cached(fn: Callable) -> bool:
    """Return bool(fn()), TTL-cached across calls.

    Exceptions are swallowed as False. A transient False/exception within
    ``_CHECK_FN_FAILURE_GRACE_SECONDS`` of the last True is suppressed (the
    last-good True is returned and the failure is NOT cached, so the next call
    re-probes) to keep flaky external checks (Docker daemon busy, socket
    contention, probe timeout) from silently stripping tools mid-session.
    """
    now = time.monotonic()
    profile_key = _capture_profile_key()
    generation = _check_fn_profile_generation(profile_key)
    cache_key = (profile_key, generation, fn)
    last_good_key = (profile_key, fn)
    with _check_fn_cache_lock:
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
        if value:
            _check_fn_last_good[last_good_key] = now
            _check_fn_cache[cache_key] = (now, True)
            return True

        last_good = _check_fn_last_good.get(last_good_key)
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


def invalidate_check_fn_cache(profile_key=None, *, all_profiles: bool = False) -> None:
    """Drop cached ``check_fn`` results for one profile (or every profile).

    Profile-local invalidation prevents a config/reload event in one selected
    profile from flushing or reusing another profile's secret-dependent probe.
    """
    with _check_fn_cache_lock:
        if all_profiles:
            _check_fn_cache.clear()
            _check_fn_last_good.clear()
            return
        key = _capture_profile_key(profile_key)
        for cache_key in [item for item in _check_fn_cache if item[0] == key]:
            _check_fn_cache.pop(cache_key, None)
        for cache_key in [item for item in _check_fn_last_good if item[0] == key]:
            _check_fn_last_good.pop(cache_key, None)


class ToolRegistry:
    """Singleton registry that collects tool schemas + handlers from tool files."""

    def __init__(self):
        # Shared built-in base plus per-profile overlays. The base is never
        # replaced by selected-profile plugin discovery.
        self._launch_profile_key = _capture_profile_key()
        self._tools: Dict[str, ToolEntry] = {}
        # Kept for backward compatibility with callers/tests that inspect the
        # launch policy map; new plugin policy is profile-local below.
        self._plugin_override_policy: Dict[str, bool] = {}
        self._profile_tools: Dict[object, Dict[str, ToolEntry]] = {}
        self._profile_override_policies: Dict[object, Dict[str, bool]] = {}
        self._profile_toolset_checks: Dict[object, Dict[str, Callable]] = {}
        self._profile_toolset_aliases: Dict[object, Dict[str, str]] = {}
        self._profile_registry_generations: Dict[object, int] = {}
        # Per-publication compare-and-swap tokens make failed-plugin rollback
        # safe against a concurrent later writer, even when it writes the same
        # object back into the same logical slot.
        self._profile_slot_generations: Dict[object, Dict[tuple[str, str], int]] = {}
        self._next_profile_slot_generation: int = 0
        self._toolset_checks: Dict[str, Callable] = {}
        self._toolset_aliases: Dict[str, str] = {}
        # MCP dynamic refresh can mutate the registry while other threads are
        # reading tool metadata, so keep mutations serialized and readers on
        # stable snapshots.
        self._lock = threading.RLock()
        # Built-in generation. Profile-local mutations are tracked separately.
        self._base_generation: int = 0
        self._bound_profile_snapshot: ContextVar[Optional[ToolProfileSnapshot]] = (
            ContextVar(f"tool_registry_snapshot_{id(self)}", default=None)
        )

    def capture_profile_snapshot(self, profile_key=None) -> ToolProfileSnapshot:
        """Capture one profile overlay for a live plugin-manager generation."""
        key = _capture_profile_key(profile_key)
        with self._lock:
            return ToolProfileSnapshot(
                registry=self,
                profile_key=key,
                tools={
                    name: entry.snapshot()
                    for name, entry in self._profile_tools.get(key, {}).items()
                },
                override_policies=dict(self._profile_override_policies.get(key, {})),
                toolset_checks=dict(self._profile_toolset_checks.get(key, {})),
                toolset_aliases=dict(self._profile_toolset_aliases.get(key, {})),
                generation=self._profile_registry_generations.get(key, 0),
            )

    def bind_profile_snapshot(self, snapshot: ToolProfileSnapshot) -> Token:
        """Select the exact tool overlay owned by a live plugin manager."""
        if snapshot.registry is not self:
            raise ValueError("tool profile snapshot belongs to a different registry")
        return self._bound_profile_snapshot.set(snapshot)

    def reset_profile_snapshot(self, token: Token) -> None:
        self._bound_profile_snapshot.reset(token)

    def _selected_profile_snapshot(self, profile_key=None) -> Optional[ToolProfileSnapshot]:
        snapshot = self._bound_profile_snapshot.get()
        if snapshot is None:
            return None
        key = _capture_profile_key(profile_key)
        return snapshot if snapshot.profile_key == key else None

    def _snapshot_state(self, profile_key=None) -> tuple[List[ToolEntry], Dict[str, Callable]]:
        """Return a coherent built-in + profile-overlay snapshot."""
        key = _capture_profile_key(profile_key)
        with self._lock:
            snapshot = self._selected_profile_snapshot(key)
            tools = dict(self._tools)
            tools.update(
                snapshot.tools
                if snapshot is not None
                else self._profile_tools.get(key, {})
            )
            checks = dict(self._toolset_checks)
            checks.update(
                snapshot.toolset_checks
                if snapshot is not None
                else self._profile_toolset_checks.get(key, {})
            )
            return list(tools.values()), checks

    def _snapshot_entries(self) -> List[ToolEntry]:
        """Return a stable snapshot of registered tool entries."""
        return self._snapshot_state()[0]

    def generation(self, profile_key=None) -> tuple[int, int]:
        """Return a cache fingerprint for built-ins and one profile overlay."""
        key = _capture_profile_key(profile_key)
        with self._lock:
            snapshot = self._selected_profile_snapshot(key)
            return (
                self._base_generation,
                snapshot.generation
                if snapshot is not None
                else self._profile_registry_generations.get(key, 0),
            )

    @property
    def _generation(self) -> int:
        """Backward-compatible current-profile monotonic generation."""
        key = _capture_profile_key()
        with self._lock:
            snapshot = self._selected_profile_snapshot(key)
            return (
                self._base_generation
                + (
                    snapshot.generation
                    if snapshot is not None
                    else self._profile_registry_generations.get(key, 0)
                )
            )

    def _bump_profile_mutation(self, profile_key: object) -> None:
        self._profile_registry_generations[profile_key] = (
            self._profile_registry_generations.get(profile_key, 0) + 1
        )
        _bump_check_fn_profile_generation(profile_key)

    def _write_profile_slot_generation(
        self, profile_key: object, category: str, name: str
    ) -> int:
        self._next_profile_slot_generation += 1
        generation = self._next_profile_slot_generation
        self._profile_slot_generations.setdefault(profile_key, {})[
            (category, name)
        ] = generation
        return generation

    def clear_profile(self, profile_key=None) -> None:
        """Discard one overlay without touching built-ins or peer profiles."""
        key = _capture_profile_key(profile_key)
        with self._lock:
            self._profile_tools.pop(key, None)
            self._profile_override_policies.pop(key, None)
            self._profile_toolset_checks.pop(key, None)
            self._profile_toolset_aliases.pop(key, None)
            self._profile_slot_generations.pop(key, None)
            self._bump_profile_mutation(key)
        invalidate_check_fn_cache(key)

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

    def get_entry(self, name: str) -> Optional[ToolEntry]:
        """Return the current profile's overlay entry, then built-in base."""
        key = _capture_profile_key()
        with self._lock:
            snapshot = self._selected_profile_snapshot(key)
            overlay = snapshot.tools if snapshot is not None else self._profile_tools.get(key, {})
            return overlay.get(name, self._tools.get(name))

    def get_registered_toolset_names(self) -> List[str]:
        """Return sorted unique toolset names present in the registry."""
        return sorted({entry.toolset for entry in self._snapshot_entries()})

    def get_tool_names_for_toolset(self, toolset: str) -> List[str]:
        """Return sorted tool names registered under a given toolset."""
        return sorted(
            entry.name for entry in self._snapshot_entries()
            if entry.toolset == toolset
        )

    def register_toolset_alias(self, alias: str, toolset: str) -> None:
        """Register an explicit alias for the current profile."""
        key = _capture_profile_key()
        with self._lock:
            aliases = self._profile_toolset_aliases.setdefault(key, {})
            existing = aliases.get(alias)
            if existing and existing != toolset:
                logger.warning(
                    "Toolset alias collision: '%s' (%s) overwritten by %s",
                    alias, existing, toolset,
                )
            aliases[alias] = toolset
            self._bump_profile_mutation(key)

    def get_registered_toolset_aliases(self) -> Dict[str, str]:
        """Return a built-in + current-profile alias snapshot."""
        with self._lock:
            key = _capture_profile_key()
            snapshot = self._selected_profile_snapshot(key)
            aliases = dict(self._toolset_aliases)
            aliases.update(
                snapshot.toolset_aliases
                if snapshot is not None
                else self._profile_toolset_aliases.get(key, {})
            )
            return aliases

    def get_toolset_alias_target(self, alias: str) -> Optional[str]:
        """Return the current profile's canonical target, or None."""
        with self._lock:
            key = _capture_profile_key()
            snapshot = self._selected_profile_snapshot(key)
            profile_aliases = (
                snapshot.toolset_aliases
                if snapshot is not None
                else self._profile_toolset_aliases.get(key, {})
            )
            return profile_aliases.get(alias, self._toolset_aliases.get(alias))

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_plugin_override_policy(self, module_namespace: str, allowed: bool) -> None:
        """Bind a plugin package's override opt-in to the current profile."""
        key = _capture_profile_key()
        module_root = _plugin_namespace_root(module_namespace)
        with self._lock:
            policies = self._profile_override_policies.setdefault(key, {})
            previous = policies.get(module_root, _MISSING)
            policies[module_root] = bool(allowed)
            written_generation = self._write_profile_slot_generation(
                key, "override-policy", module_root
            )
            self._bump_profile_mutation(key)

            try:
                from agent.plugin_profile_scope import record_registration_undo
            except ImportError:
                record_registration_undo = None
            if record_registration_undo is not None:
                def _undo() -> None:
                    with self._lock:
                        generations = self._profile_slot_generations.get(key, {})
                        if generations.get(("override-policy", module_root)) != written_generation:
                            return
                        current = self._profile_override_policies.get(key)
                        if current is None:
                            return
                        if previous is _MISSING:
                            current.pop(module_root, None)
                        else:
                            current[module_root] = previous
                        self._write_profile_slot_generation(
                            key, "override-policy", module_root
                        )
                        self._bump_profile_mutation(key)

                record_registration_undo(key, _undo)

    def _plugin_owner_of(
        self, handler: Callable, profile_key: object | None = None
    ) -> Optional[str]:
        """Return the defining plugin package root for *handler*, if any."""
        try:
            mod = handler.__globals__.get("__name__", "")  # type: ignore[attr-defined]
        except AttributeError:
            return None
        if not isinstance(mod, str):
            return None
        return self._plugin_owner_from_module(mod, profile_key)

    def _plugin_owner_from_module(
        self, module_name: str, profile_key: object | None = None
    ) -> Optional[str]:
        """Resolve a module to its exact plugin policy owner, when applicable."""
        if not module_name.startswith("hermes_plugins."):
            return None
        key = _capture_profile_key(profile_key)
        policy_owners = self._profile_policy(key)
        matching_owners = [
            owner
            for owner in policy_owners
            if module_name == owner or module_name.startswith(f"{owner}.")
        ]
        if matching_owners:
            return max(matching_owners, key=len)
        return _plugin_namespace_root(module_name)

    def _profile_policy(self, profile_key: object) -> Dict[str, bool]:
        snapshot = self._selected_profile_snapshot(profile_key)
        return (
            snapshot.override_policies
            if snapshot is not None
            else self._profile_override_policies.get(profile_key, {})
        )

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
    ):
        """Register a tool.  Called at module-import time by each tool file.

        ``override=True`` is an explicit opt-in for plugins that intend to
        replace an existing built-in tool implementation (e.g. swap the
        default browser tool for a headed-Chrome CDP backend). Without it,
        registrations that would shadow an existing tool from a different
        toolset are rejected to prevent accidental overwrites.
        """
        profile_key = _capture_profile_key()
        with self._lock:
            owner = self._plugin_owner_of(handler, profile_key)
            is_profile_entry = (
                owner is not None or profile_key != self._launch_profile_key
            )
            overlay = self._profile_tools.setdefault(profile_key, {})
            previous_profile_entry = overlay.get(name, _MISSING)
            existing = overlay.get(name, self._tools.get(name))
            if existing and existing.toolset != toolset:
                if override:
                    if owner is not None and not self._profile_policy(profile_key).get(owner, False):
                        logger.error(
                            "Tool registration REJECTED: plugin %r attempted to "
                            "override tool %r for profile %r without operator opt-in",
                            owner, name, profile_key,
                        )
                        raise PermissionError(
                            f"Plugin module {owner!r} cannot override tool {name!r} "
                            f"for profile {profile_key!r} without operator opt-in "
                            f"(allow_tool_override)."
                        )
                    logger.info(
                        "Tool '%s': toolset '%s' overriding existing toolset '%s' "
                        "for profile %r (override=True opt-in)",
                        name, toolset, existing.toolset, profile_key,
                    )
                else:
                    logger.error(
                        "Tool registration REJECTED: '%s' (toolset '%s') would "
                        "shadow existing tool from toolset '%s' in profile %r. "
                        "Pass override=True if intentional.",
                        name, toolset, existing.toolset, profile_key,
                    )
                    return
            entry = ToolEntry(
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
            if is_profile_entry:
                checks = self._profile_toolset_checks.setdefault(profile_key, {})
                previous_check = checks.get(toolset, _MISSING)
                overlay[name] = entry
                if check_fn and toolset not in checks:
                    checks[toolset] = check_fn
                written_generation = self._write_profile_slot_generation(
                    profile_key, "tool", name
                )
                self._bump_profile_mutation(profile_key)

                try:
                    from agent.plugin_profile_scope import record_registration_undo
                except ImportError:
                    record_registration_undo = None
                if record_registration_undo is not None:
                    def _undo() -> None:
                        with self._lock:
                            generations = self._profile_slot_generations.get(
                                profile_key, {}
                            )
                            if generations.get(("tool", name)) != written_generation:
                                return
                            current_overlay = self._profile_tools.get(profile_key)
                            if current_overlay is None:
                                return
                            if previous_profile_entry is _MISSING:
                                current_overlay.pop(name, None)
                            else:
                                current_overlay[name] = previous_profile_entry
                            current_checks = self._profile_toolset_checks.get(
                                profile_key, {}
                            )
                            if previous_check is _MISSING:
                                if not any(
                                    candidate.toolset == toolset
                                    for candidate in current_overlay.values()
                                ):
                                    current_checks.pop(toolset, None)
                            else:
                                current_checks[toolset] = previous_check
                            self._write_profile_slot_generation(
                                profile_key, "tool", name
                            )
                            self._bump_profile_mutation(profile_key)

                    record_registration_undo(profile_key, _undo)
            else:
                self._tools[name] = entry
                if check_fn and toolset not in self._toolset_checks:
                    self._toolset_checks[toolset] = check_fn
                self._base_generation += 1

    def deregister(self, name: str) -> None:
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
        profile_key = _capture_profile_key()
        with self._lock:
            overlay = self._profile_tools.get(profile_key, {})
            entry = overlay.get(name)
            target_is_profile = entry is not None
            if entry is None:
                entry = self._tools.get(name)
            if entry is None:
                return
            if not entry.toolset.startswith("mcp-"):
                caller_mod = self._caller_module()
                owner = self._plugin_owner_of(entry.handler, profile_key)
                caller_root = (
                    self._plugin_owner_from_module(caller_mod, profile_key)
                    if caller_mod.startswith("hermes_plugins.")
                    else caller_mod
                )
                same_plugin = bool(owner and caller_root == owner)
                if (
                    caller_mod.startswith("hermes_plugins.")
                    and not same_plugin
                    and not self._profile_policy(profile_key).get(caller_root, False)
                ):
                    logger.error(
                        "Tool deregistration REJECTED: plugin %r attempted to "
                        "remove tool %r (toolset %r) in profile %r without opt-in",
                        caller_mod, name, entry.toolset, profile_key,
                    )
                    raise PermissionError(
                        f"Plugin module {caller_mod!r} cannot deregister tool "
                        f"{name!r} (toolset {entry.toolset!r}) without operator "
                        f"opt-in (allow_tool_override)."
                    )
            if target_is_profile:
                del overlay[name]
                remaining = list(overlay.values())
                checks = self._profile_toolset_checks.get(profile_key, {})
                aliases = self._profile_toolset_aliases.get(profile_key, {})
            else:
                del self._tools[name]
                remaining = list(self._tools.values())
                checks = self._toolset_checks
                aliases = self._toolset_aliases
            if not any(item.toolset == entry.toolset for item in remaining):
                checks.pop(entry.toolset, None)
                filtered = {
                    alias: target for alias, target in aliases.items()
                    if target != entry.toolset
                }
                if target_is_profile:
                    self._profile_toolset_aliases[profile_key] = filtered
                else:
                    self._toolset_aliases = filtered
                    # MCP aliases are registered in the active profile overlay
                    # even when their launch-time tool entry is in the base.
                    profile_aliases = self._profile_toolset_aliases.get(profile_key, {})
                    self._profile_toolset_aliases[profile_key] = {
                        alias: target for alias, target in profile_aliases.items()
                        if target != entry.toolset
                    }
            if target_is_profile:
                self._bump_profile_mutation(profile_key)
            else:
                self._base_generation += 1
        logger.debug("Deregistered tool: %s", name)

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
            return result
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

    def dispatch(self, name: str, args: dict, **kwargs) -> str | dict:
        """Execute a tool handler by name.

        * Async handlers are bridged automatically via ``_run_async()``.
        * Handler results are normalized to a string or supported multimodal
          envelope before leaving the registry.
        * All exceptions are caught and returned as ``{"error": "..."}``
          for consistent error format.
        """
        entry = self.get_entry(name)
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
            logger.exception("Tool %s dispatch error: %s", name, e)
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
    result = {"error": str(message)}
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
