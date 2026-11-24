"""Launch-profile policy for a process that hosts several profile homes (``hermes serve`` /
``hermes dashboard`` pooling, ``?profile=``, hosted rooms; the multiplexed gateway's own worker).

Two facts anchor this module:

* ``agent.secret_scope.get_secret`` fails closed ONLY while ``set_multiplex_active(True)`` holds.
  A ``serve`` backend that hosts a second profile home never flipped it, so every unscoped read
  for a secondary profile silently returned the LAUNCH profile's ``os.environ`` value. The flip
  happens here, at the moment the process first learns it hosts another profile home.
* Once multiplexing is active the launch profile is a profile too: its turns/RPCs must run under
  their own scope instead of ambient ``os.environ`` (a secondary context may have poisoned it,
  #107422). A scope rebuilt from ``<launch home>/.env`` + ``config.yaml`` alone would drop the
  launch process's legitimate env-only policy — ``TERMINAL_ENV=ssh`` or a provider key injected
  by systemd / ``op run`` has no file to rebuild it from. The process env is trusted exactly
  once: frozen at activation, before any secondary code has run, never re-read afterwards.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Dict, Iterator, Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_snapshot: Optional[dict[str, str]] = None
# This process's dotenv publication record when the snapshot froze. The launch ``.env``'s ``${VAR}``s
# resolve against the snapshot with that output peeled back: by this record, as a later reload's no
# longer matches the frozen values, applied to the snapshot as it is now, as it follows secret sources.
_snapshot_dotenv_record: dict[str, tuple[Optional[str], str, int]] = {}
# Per (profile home key, name): a secret-source write the frozen launch env holds — (source, value the
# source put in the snapshot, the snapshot's value before the source's first write or None). Kept apart
# from env_loader's ``os.environ`` write record on purpose: that record is dropped or retargeted whenever
# ``os.environ`` diverges (another writer's value, a failed fetch, a default source displaced), while the
# snapshot never sees those changes and must still lose the source's value when the source goes.
_source_writes: dict[tuple[str, str], tuple[str, str, Optional[str]]] = {}


def capture_launch_env() -> dict[str, str]:
    """Freeze the process env as the launch profile's own; the first capture wins.

    Called at activation, immediately before the first secondary home is registered as
    served — the last moment ambient env is provably the launch profile's.
    """
    global _snapshot, _snapshot_dotenv_record, _source_writes
    with _lock:
        if _snapshot is not None:
            return dict(_snapshot)
    # The first capture holds the dotenv loader's publication lock (taken before ``_lock``, the order
    # env_loader documents), so a ``.env`` reload can't land between the env copy and the record copy.
    loader = sys.modules.get("hermes_cli.env_loader")  # not imported yet: nothing published
    with loader.dotenv_publication_lock() if loader is not None else contextlib.nullcontext(), _lock:
        if _snapshot is None:
            _snapshot = dict(os.environ)
            _snapshot_dotenv_record = loader.dotenv_publication_record() if loader is not None else {}
            _source_writes = _source_writes_held_by(_snapshot)
        return dict(_snapshot)


def launch_template_env() -> dict[str, str]:
    """What the launch ``.env``'s ``${VAR}``s resolve against in the frozen launch env (see
    ``_snapshot_dotenv_record``). A name a secret source holds there (``_source_writes``, read with the
    snapshot) is the source's value and is never peeled, even one equal to what the ``.env`` published."""
    capture_launch_env()
    with _lock:
        snapshot = dict(_snapshot)
        held = {name for (_home, name), rec in _source_writes.items() if snapshot.get(name) == rec[1]}
        record = {name: rec for name, rec in _snapshot_dotenv_record.items() if name not in held}
    loader = sys.modules.get("hermes_cli.env_loader")
    return loader.dotenv_lookup_from(snapshot, record) if loader is not None else snapshot


def _source_writes_held_by(snapshot: dict[str, str]) -> dict[tuple[str, str], tuple[str, str, Optional[str]]]:
    """The secret-source writes still in ``os.environ`` at capture, which the snapshot now holds too.
    Before activation ``os.environ`` IS the launch env, so a write's ``os.environ`` prior is the
    snapshot's pre-write value."""
    loader = sys.modules.get("hermes_cli.env_loader")
    if loader is None:
        return {}
    return {(home, name): (source, value, prior)
            for home, name, source, value, prior in loader._recorded_environ_writes()
            if snapshot.get(name) == value}


def _launch_env_captured() -> bool:
    """For ``env_loader`` only: whether the launch env is frozen yet."""
    with _lock:
        return _snapshot is not None


def _launch_env_source_writes(home: str) -> dict[str, str]:
    """For ``env_loader`` only: name → source of every write the snapshot holds for *home*."""
    with _lock:
        if _snapshot is None:
            return {}
        return {name: rec[0] for (key, name), rec in _source_writes.items() if key == home}


def _track_launch_env_write(home: str, name: str, source: str, published: str, prior: Optional[str],
                            *, adopt: bool) -> None:
    """For ``env_loader`` only: *source* published *published* for *name* into ``os.environ``.

    With a record for *name*, the snapshot follows a rotation while it still holds that source's
    last value; the record keeps the ORIGINAL pre-write value. Without one, *adopt* (activation landed
    during this pass, after the publish) records the write when the snapshot froze it, with *prior*
    (the pre-activation ``os.environ`` value). Any other write never reaches the snapshot."""
    key = (home, name)
    with _lock:
        if _snapshot is None:
            return
        rec = _source_writes.get(key)
        if rec is not None:
            if rec[0] == source and _snapshot.get(name) == rec[1] and published != rec[1]:
                _snapshot[name] = published
                _source_writes[key] = (source, published, rec[2])
        elif adopt and _snapshot.get(name) == published:
            _source_writes[key] = (source, published, prior)


def _revoke_launch_env_write(home: str, name: str) -> None:
    """For ``env_loader`` only: the source behind *home*'s write of *name* is gone. While the
    snapshot still holds the value it put there, the snapshot gets back its own pre-write value or
    loses the name. It never receives any other value: a value another writer put in ``os.environ``
    after activation never enters it. A value the operator set and a source merely also supplied
    was never written (``skipped_existing``), so it is never recorded here."""
    with _lock:
        rec = _source_writes.pop((home, name), None)
        if rec is None or _snapshot is None or _snapshot.get(name) != rec[1]:
            return
        if rec[2] is None:
            _snapshot.pop(name, None)
        else:
            _snapshot[name] = rec[2]


def activate_multi_profile_hosting() -> None:
    """This process now hosts a profile home other than its launch home: freeze the launch env
    and make unscoped credential reads fail closed (``get_secret`` raises instead of borrowing)."""
    from agent.secret_scope import set_multiplex_active
    capture_launch_env()
    set_multiplex_active(True)


def _servable_profile_homes() -> set:
    """Resolved homes this host could be asked to serve: the launch home plus every profile dir
    carrying a real servability marker.

    ``named_profile_has_identity`` accepts an EMPTY ``.env``, which is all a crashed
    ``hermes profile create`` leaves behind — counting it would flip a single-profile host
    fail-closed at its next boot.
    """
    from hermes_constants import named_profile_has_servable_identity
    from hermes_cli.profiles import profiles_to_serve

    # A profile a plugin runs elsewhere counts: the host serves it again as soon as it comes back.
    homes = {Path(home).resolve() for name, home in profiles_to_serve(
                 multiplex=True, include_standalone=True, include_parked=True, include_run_elsewhere=True)
             if name == "default" or named_profile_has_servable_identity(home)}
    homes.add(Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes").resolve())
    return homes


def activate_multi_profile_hosting_eagerly() -> bool:
    """Activate at HOST startup when this machine has more than one servable profile home.

    Activation used to be lazy and one-way — it fired the first time a *request* asked for a second
    home. Everything the host had already done by then (idle-reaper ticks, cron start, adapter
    connects, MCP discovery) ran under single-profile assumptions and is never re-scoped, and the
    launch profile's env had already been mutated by then, so ``capture_launch_env`` froze a
    polluted snapshot. One ``hermes serve`` / ``hermes gateway run`` per host means the process
    knows at boot whether it can be asked for a second home: decide there, once. Call it as the LAST
    boot step: the frozen snapshot is the only source for launch keys with no ``.env`` to rebuild
    from, so every credential the boot still injects must already be in ``os.environ``.

    ``gateway.multiplex_profiles: false`` is deliberately NOT consulted: it is retired as a
    topology opt-out, and a multi-home host that skipped activation because of a stale ``false``
    would serve a second profile with the LAUNCH profile's credentials — the exact fail-open this
    guard exists to prevent.

    A genuinely single-profile host still never activates (byte-identical behaviour, ``os.environ``
    precedence preserved). An unreadable profiles directory fails CLOSED: we cannot prove the host
    is single-profile, and the lazy backstop only fires once a request has already been answered.
    Returns True when this call activated hosting.
    """
    from agent.secret_scope import is_multiplex_active
    if is_multiplex_active():
        return False
    try:
        homes = _servable_profile_homes()
    except Exception:
        logger.warning(
            "Could not enumerate this host's profile homes; activating multi-profile hosting "
            "fail-closed (unscoped credential reads will raise instead of borrowing the launch "
            "profile's)", exc_info=True)
        activate_multi_profile_hosting()
        return True
    if len(homes) < 2:
        return False
    logger.info("Multi-profile hosting activated at startup (%d servable profile homes)", len(homes))
    activate_multi_profile_hosting()
    # Route each home's records to its own logs/ now: the Desktop cron ticker used to be the only
    # caller, and it stands down when a live gateway owns cron, leaving every secondary profile's
    # model/cwd lines in the launch profile's agent.log.
    from hermes_logging import enable_profile_log_routing
    enable_profile_log_routing(sorted(homes))
    return True


def _launch_env() -> dict[str, str]:
    """The launch profile's env: frozen once multiplexing is active; the LIVE process env before
    (no secondary has run yet, so it is provably the launch profile's, and freezing it early would
    miss values the launch process still bridges at startup)."""
    from agent.secret_scope import is_multiplex_active
    return capture_launch_env() if is_multiplex_active() else dict(os.environ)


def launch_terminal_env() -> dict[str, str]:
    """The frozen launch ``TERMINAL_*`` overlay for a launch-profile turn's terminal scope.

    Production always captured at activation; a first capture here only happens when the
    multiplexer flag was set by another owner (the messaging gateway) or a harness.
    """
    return {k: v for k, v in capture_launch_env().items() if k.startswith("TERMINAL_")}


def launch_secret_scope(launch_home: str | Path, *, frozen_templates: bool = False,
                        process_home: Optional[bool] = None) -> dict[str, str]:
    """The launch profile's secret mapping: its ``.env`` + external sources over the launch env
    (systemd / ``op run`` injection survives the fail-closed flip; a secondary never sees it because
    its scope is built from its own files only). Bound for EVERY launch-profile body, multiplexing or
    not, so the body's credential source is decided once at entry: a request that entered while
    single-profile keeps resolving from this mapping after a concurrent first secondary flips
    ``get_secret`` to fail closed (``_MULTIPLEX_ACTIVE`` is read on every ``get_secret``, the
    scope decision was made at entry).

    PRECEDENCE, and it is not the process's: ``<launch home>/.env`` (plus hydrated external sources)
    WINS over the env. For a key present in both, an unscoped read returned the ambient
    ``os.environ`` value before activation and returns the ``.env`` value after — so activation is
    not purely "stricter" for the launch tenant, it also changes which of the launch profile's own
    two values it sees. Env-only keys (systemd ``Environment=``, ``op run``, Compose) are unaffected:
    nothing in the files shadows them.

    *frozen_templates* resolves the ``.env``'s ``${VAR}``s against that same launch env once multiplexing
    froze it (``launch_template_env``), rather than the live one: for a caller that builds the mapping
    again later in a body (the plugin host's launcher), which must not follow a write to the live env
    since the body bound. *process_home* is ``build_profile_secret_scope``'s, for a caller that decided
    *launch_home* is the launch profile's by another rule (the plugin host's launcher, by the key the
    process remembered for its own path, where a renamed link no longer leads).
    """
    from agent.secret_scope import _is_global_env, build_profile_secret_scope, is_multiplex_active
    from gateway.bridged_env import follow_bridged_env
    scope = {k: v for k, v in _launch_env().items() if not _is_global_env(k)}
    # The frozen env's config.yaml-bridged lists follow the live bridge, or a user removed from the
    # launch profile's config.yaml stays admitted until a restart; operator values stay frozen.
    follow_bridged_env(scope)
    templates = launch_template_env() if frozen_templates and is_multiplex_active() else None
    scope.update(build_profile_secret_scope(Path(launch_home), process_home=process_home, template_env=templates))
    return scope


def follow_launch_bridged_env() -> None:
    """After the launch profile's live allowlist reload re-bridged its lists, bring the INSTALLED launch
    mapping (built at the body's entry, before the reload) up to date, so the rest of this body reads
    the edited lists. The caller runs this only with the launch scope bound (``_owner_scope``) and its
    bridge not skipped, so the mapping is never another profile's."""
    from agent.secret_scope import build_profile_secret_scope, current_secret_scope
    from gateway.bridged_env import follow_bridged_env
    from hermes_constants import get_hermes_home

    scope = current_secret_scope()
    if isinstance(scope, dict):
        follow_bridged_env(scope)
        scope.update(build_profile_secret_scope(get_hermes_home()))  # .env / sources still win, as at build


def owner_secret_scope(home: str | Path) -> dict[str, str]:
    """THE secret mapping for a body owned by *home*: ``launch_secret_scope`` when *home* is this
    process's own (its env-only keys from docker ``-e`` / systemd ``Environment=`` / ``op run`` have
    no file to rebuild from), else the profile's own files (``build_profile_secret_scope``). Every
    per-home binding point (routed gateway turns, cron fires, kanban ticks) builds through here, so
    the launch profile keeps its env-only credentials when a single-profile gateway starts
    multiplexing in place."""
    from agent.secret_scope import _is_process_home, build_profile_secret_scope
    if _is_process_home(Path(home)):
        return launch_secret_scope(home)
    return build_profile_secret_scope(Path(home))


@contextlib.contextmanager
def launch_body() -> Iterator[None]:
    """Mark a standalone launch-profile body (a primary-adapter message) at ENTRY. Nothing is bound
    while single-profile, so its reads keep the process-env semantics they always had; if a second
    profile switches the gateway to fail-closed mid-body, an unscoped read resolves from
    ``launch_secret_scope`` (frozen env + ``.env``) instead of raising ``UnscopedSecretError``."""
    from agent.secret_scope import reset_launch_body_scope, set_launch_body_scope
    from hermes_constants import get_process_hermes_home
    home = get_process_hermes_home()
    token = set_launch_body_scope(lambda: launch_secret_scope(home))
    try:
        yield
    finally:
        reset_launch_body_scope(token)


@contextlib.contextmanager
def launch_profile_runtime_scope(launch_home: str | Path) -> Iterator[None]:
    """Bind the launch profile's own runtime scope for one body: HERMES_HOME override naming the
    launch home, ``launch_secret_scope``, and its terminal policy over the frozen launch
    ``TERMINAL_*`` overlay. For hosts whose launch-profile bodies are not RPC sessions (the
    standalone messaging gateway after a hosted room activated multiplexing, #112878).

    The home override is bound even though the launch home IS the process home: under multiplexing
    "override unset" is the fail-closed signal for an UNBOUND context (``serves_routed_profile``,
    plugin runtime bindings such as OMH's ``pre_tool_call`` gate, per-home slots keyed on the
    override), so a launch-profile turn without it was indistinguishable from no turn at all and
    every plugin hook it fired saw an unscoped process (#118538). Routed turns already bind theirs
    (``gateway/run.py::_profile_runtime_scope``); the launch profile is a tenant like any other."""
    from agent.runtime_scope import bind_state, scope_for_profile

    home = Path(launch_home)
    # Own home: secrets unstamped (no foreign-home stamp), terminal policy over the frozen overlay.
    tokens = bind_state(scope_for_profile(home), secrets=launch_secret_scope(home), stamp_home=False,
                        env_overlay=launch_terminal_env())
    try:
        yield
    finally:
        tokens.reset()


@contextlib.contextmanager
def profile_config_scope(home: str | Path, *, launch_home: str | Path | None = None) -> Iterator[None]:
    """Bind *home*'s home + secret + terminal scope for a body that reads and writes its config.

    The launch home (``launch_home``, default ``get_process_hermes_home()``) gets its launch secret
    scope with no home override or terminal scope (an enclosing secondary's are unbound for the
    body), or the full :func:`launch_profile_runtime_scope` once the process multiplexes. Any other
    home gets only its own files, stamped as foreign. No external-source hydration.
    """
    from agent.secret_scope import build_profile_secret_scope, is_multiplex_active, reset_secret_scope, set_secret_scope
    from hermes_constants import get_process_hermes_home, reset_hermes_home_override, set_hermes_home_override
    from tools.terminal_scope import install_profile_terminal_scope, reset_terminal_scope, set_terminal_scope

    home = Path(home)
    launch = Path(launch_home if launch_home is not None else get_process_hermes_home())
    with contextlib.ExitStack() as stack:
        if home.resolve() == launch.resolve():
            if is_multiplex_active():
                stack.enter_context(launch_profile_runtime_scope(launch))
            else:
                # Unbind an enclosing secondary's home and terminal, or the launch secrets would
                # pair with that profile's config. Unset resolves to the process home, which is the
                # launch home for every caller (``launch_home`` defaults to it; the RPC passes it).
                stack.callback(reset_hermes_home_override, set_hermes_home_override(None))
                stack.callback(reset_secret_scope, set_secret_scope(launch_secret_scope(launch)))
                stack.callback(reset_terminal_scope, set_terminal_scope(None))
        else:
            secrets = build_profile_secret_scope(home)
            stack.callback(reset_hermes_home_override, set_hermes_home_override(str(home)))
            stack.callback(reset_secret_scope, set_secret_scope(secrets, profile_home=str(home)))
            stack.callback(reset_terminal_scope, install_profile_terminal_scope(home))
        yield


def launch_profile_scope_if_multiplexed():
    """The launch profile's own runtime scope once this process multiplexes; ``nullcontext``
    before.

    The single seam for "no routed profile here": under the one-process-per-host ruling the launch
    profile is a tenant like any other, so a body that used to run with NO scope at all (ambient
    ``os.environ`` + the process home) must bind the launch profile explicitly — otherwise a
    secondary's context can have poisoned the ambient state, and a fail-closed ``get_secret`` raises
    on a perfectly legitimate launch-profile read. Before activation the process env IS the launch
    profile's, so binding nothing is still correct (and keeps single-profile hosts byte-identical —
    callers assert the returned object is literally a ``nullcontext``).
    """
    from agent.secret_scope import is_multiplex_active
    if not is_multiplex_active():
        return contextlib.nullcontext()
    from hermes_constants import get_process_hermes_home
    return launch_profile_runtime_scope(get_process_hermes_home())


@contextlib.asynccontextmanager
async def async_launch_profile_scope_if_multiplexed():
    """``async with`` twin of :func:`launch_profile_scope_if_multiplexed` (no I/O of its own: the
    launch scope is rebuilt from already-hydrated sources, so there is nothing to move off-loop)."""
    with launch_profile_scope_if_multiplexed():
        yield
