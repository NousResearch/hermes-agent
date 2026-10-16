"""``manage_catalog`` install targets: catalog plugins and hub skills, installed through the card.

``prepare`` resolves every id against the plugin catalog (the Plugins tab's resolver) or the skills
hub and fills the row the card draws; an id that does not resolve, or a plugin this OS cannot run,
is drawn failed with the reason. Nothing installs until the user approves a row. The install runs on
a worker under the TARGET profile's runtime scope (the calling chat's own home unless the Advanced
modal named another profile), through the same host install the Plugins tab uses, so the catalog
pin, the kill list, the security scan and the live activation of the plugin's MCP servers and skills
are the host's.
"""

from __future__ import annotations

import contextlib
import contextvars
import io
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from hermes_cli.plugin_install_phase import InstallPhase
from hermes_constants import get_hermes_home, profile_name_for_home
from tools.connectors.contract import Actor, SettleReason, TargetState
from tools.connectors.mcp import _fail, _move
from tools.connectors.operation import ConnectionOperation, IllegalTransition, Target

logger = logging.getLogger(__name__)

# The Advanced modal's own keys (CATALOG-ROW-CONTRACT.md); every other answer key is a credential.
_OPTION_KEYS = frozenset({"target_profile", "agent_half", "desktop_half", "enable", "force", "ref"})
_COMMIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")
_TIERS = frozenset({"official", "community"})


def _flag(value: Optional[str], default: bool) -> bool:
    return default if value in (None, "") else value == "1"


def _first_sentence(text: str) -> str:
    text = " ".join(str(text or "").split())
    head, dot, _rest = text.partition(". ")
    return f"{head}." if dot else text


def _display(identifier: str) -> str:
    leaf = identifier.rstrip("/").rsplit("/", 1)[-1]
    return " ".join(part.capitalize() for part in re.split(r"[-_]+", leaf) if part) or identifier


@contextlib.contextmanager
def target_scope(home: Path):
    """Bind the profile at *home*: its home, secrets and terminal policy, the way an RPC for that
    profile does. The install writes its tree, config and ``.env`` there. The home override is
    bound for the launch profile too: the calling thread may carry another profile's override, and
    an unbound launch scope would leave it in place."""
    from tui_gateway import server
    from tui_gateway.launch_profile_policy import launch_profile_runtime_scope

    launch = home.resolve() == Path(server._hermes_home).resolve()
    scope = (launch_profile_runtime_scope(server._hermes_home) if launch
             else server._session_profile_runtime_scope({"profile_home": str(home)}))
    with scope:
        yield


def _named_home(name: str) -> Path:
    """The home of the profile the Advanced modal named; raises for one that does not exist."""
    from tui_gateway import server

    return server._profile_home(name) or Path(server._hermes_home)


class HostInstaller:
    """The host side of a catalog row. Tests replace it; production reads the real catalog, hub and
    installer."""

    def plugin_entry(self, name: str) -> Any:
        from hermes_cli.plugin_catalog import get_live_catalog_entry

        return get_live_catalog_entry(name)

    def refuse(self, entry: Any) -> None:
        """Raise with the installer's own text when the catalog would refuse this entry here."""
        from hermes_cli.plugins_cmd_catalog import _refuse_unsupported_catalog_platform, raise_if_removed

        raise_if_removed(entry.name, entry.repo)
        _refuse_unsupported_catalog_platform(entry)

    def install_plugin(self, name: str, *, force: bool, enable: bool, ref: Optional[str],
                       on_step: Callable[[InstallPhase], None]) -> dict[str, Any]:
        from hermes_cli.plugins_cmd import dashboard_install_plugin

        return dashboard_install_plugin("", force=force, enable=enable, catalog_name=name, ref=ref, on_step=on_step)

    def skill_meta(self, identifier: str) -> Optional[dict[str, Any]]:
        """The first hub source that knows the identifier; metadata only, no bundle download."""
        from hermes_cli.skills_hub import _sources
        from tools.skills_hub import skills_hub_http_session

        with skills_hub_http_session():
            for source in _sources():
                try:
                    meta = source.inspect(identifier)
                except Exception:
                    continue
                if meta is not None:
                    return {"name": meta.name, "description": meta.description, "source": meta.source,
                            "identifier": meta.identifier or identifier}
        return None

    def install_skill(self, identifier: str, *, force: bool) -> dict[str, Any]:
        """Install headless; ``{name, already_installed}``. ``do_install`` reports only by printing, so
        success is read from the hub lock file and failure from its last line. A skill that is already
        installed (force off) is left as it is and reported so."""
        from rich.console import Console

        from hermes_cli.skills_hub import do_install
        from tools.skills_hub import HubLockFile

        def entry() -> Optional[dict[str, Any]]:
            return next((e for e in HubLockFile().list_installed() if e.get("identifier") == identifier), None)

        before = entry()
        if before is not None and not force:
            return {"name": str(before["name"]), "already_installed": True}
        out = io.StringIO()
        verdict = do_install(identifier, force=force, skip_confirm=True,
                             console=Console(file=out, width=200, no_color=True, highlight=False))
        after = entry()
        from tools.skills_sync_bundled_ops import bundled_skill_for_install
        if after is None and verdict is not False and (builtin := bundled_skill_for_install(identifier)):
            return {"name": builtin, "already_installed": verdict is None}  # shipped skill, no hub lock entry
        if after is None or (before is not None and after.get("updated_at") == before.get("updated_at")):
            lines = [line.strip() for line in out.getvalue().splitlines() if line.strip()]
            raise RuntimeError(lines[-1] if lines else "the skill was not installed")
        return {"name": str(after["name"]), "already_installed": False}


@dataclass
class _Work:
    done: threading.Event = field(default_factory=threading.Event)
    outcome: dict[str, Any] = field(default_factory=dict)
    error: str = ""


class _Runner:
    """Resolved catalog facts and install work for one operation's rows."""

    def __init__(self, installer: HostInstaller):
        self.installer = installer
        self.op_id: Optional[str] = None
        self.facts: dict[str, Any] = {}  # row name -> PluginCatalogEntry | skill meta; never on the wire
        self.work: dict[str, _Work] = {}
        # The Advanced values the user approved per row; Try again (env null) reuses them. Values
        # are credentials in part, so they stay here, off the target.
        self.approved_env: dict[str, dict[str, str]] = {}
        # Built on the tool call's thread: the calling chat's own home is the default install
        # target, and its profile name labels the row (None for a home that is no named profile).
        self.home: Path = get_hermes_home()
        self.label: Optional[str] = profile_name_for_home(self.home)

    # -- prepare: resolve each id and draw its row ------------------------------------------------

    def prepare(self, operation: ConnectionOperation) -> None:
        _RUNNERS[operation.op_id] = self
        self.op_id = operation.op_id
        for target in operation.targets:
            target.extra = {"display": _display(target.name), "target_profile": self.label}
            try:
                self._resolve(target)
            except Exception as exc:
                _fail(operation, target, self._detail(exc, target))

    def _resolve(self, target: Target) -> None:
        if target.kind == "plugin":
            entry = self.installer.plugin_entry(target.name)
            if entry is None:
                raise LookupError(f"'{target.name}' is not in the Hermes plugin catalog")
            self.facts[target.name] = entry
            target.extra = _plugin_row(entry, self.label)
            target.required_env = [{"name": name, "required": False, "secret": True, "default": ""}
                                   for name in entry.capabilities.requires_env]
            self.installer.refuse(entry)
            return
        meta = self.installer.skill_meta(target.name)
        if not meta:
            raise LookupError(f"'{target.name}' was not found in the skills hub")
        self.facts[target.name] = meta
        target.extra = {
            "display": str(meta.get("name") or _display(target.name)),
            "description": _first_sentence(meta.get("description") or ""),
            "tier": "official" if meta.get("source") == "official" else "community",
            "target_profile": self.label,
        }

    # -- the card's answer ------------------------------------------------------------------------

    def approve(self, operation: ConnectionOperation, target: Target, env: Optional[dict[str, str]]) -> None:
        approved = {**self.approved_env.get(target.name, {}), **(env or {})}
        actor = Actor.user if target.state == TargetState.failed else Actor.backend_watcher
        if target.name not in self.facts:  # failed at prepare: Try again resolves once more
            try:
                self._resolve(target)
            except Exception as exc:
                _fail(operation, target, self._detail(exc, target))
                return
        error = self._check_answer(target, approved)
        if error:
            _fail(operation, target, error)
            return
        self.approved_env[target.name] = approved
        # The non-secret choices go on the row, so a Try again after the operation settled repeats
        # what the user approved; the credentials are already in the target profile's .env by then.
        options = {"force": _flag(approved.get("force"), False), "enable": _flag(approved.get("enable"), True),
                   "ref": approved.get("ref") or None}
        profile = (approved.get("target_profile") or "").strip()
        if not _move(operation, target, TargetState.initiated, actor, detail="",
                     **{**target.extra, "target_profile": profile or self.label, "approved": options}):
            return
        self._spawn(operation, target, approved, options)

    def _check_answer(self, target: Target, env: dict[str, str]) -> str:
        if target.kind == "plugin" and env.get("agent_half") == "0":
            return "only the desktop half was selected; install it from Settings, Plugins"
        ref = env.get("ref")
        if ref and not _COMMIT_SHA.match(ref):
            return "the pin must be a full 40-character commit SHA"
        declared = set(target_declared_env(self.facts.get(target.name)))
        undeclared = sorted(k for k in env if k not in _OPTION_KEYS and k not in declared)
        if undeclared:
            return f"'{target.name}' does not declare {', '.join(undeclared)}"
        return ""

    def _spawn(self, operation: ConnectionOperation, target: Target, env: dict[str, str],
               options: dict[str, Any]) -> None:
        work = _Work()
        self.work[target.name] = work

        def step(phase: InstallPhase) -> None:
            # A row that left ``initiated`` keeps its own state.
            if not operation.settled and target.state == TargetState.initiated:
                operation.refresh(target.name, connect_url=None, detail="", actor=Actor.backend_watcher,
                                  phase=phase.value)

        def body() -> None:
            try:
                work.outcome = self._install(target, env, options, step)
            except Exception as exc:
                work.error = self._detail(exc, target)
            work.done.set()
            operation.wake.set()

        # A copy of the answering thread's context; the install binds the target profile itself.
        threading.Thread(target=contextvars.copy_context().run, args=(body,), daemon=True,
                         name=f"catalog-install-{target.name}").start()

    def _install(self, target: Target, env: dict[str, str], options: dict[str, Any],
                 step: Callable[[InstallPhase], None]) -> dict[str, Any]:
        named = (env.get("target_profile") or "").strip()
        with target_scope(_named_home(named) if named else self.home):
            _save_credentials({k: v for k, v in env.items() if k not in _OPTION_KEYS and v})
            if target.kind == "skill":
                identifier = str(self.facts[target.name].get("identifier") or target.name)
                step(InstallPhase.downloading)
                return self.installer.install_skill(identifier, force=options["force"])
            result = self.installer.install_plugin(target.name, force=options["force"], enable=options["enable"],
                                                   ref=options["ref"], on_step=step)
        if not result.get("ok"):
            raise RuntimeError(result.get("error") or "the install failed")
        return {"enabled": options["enable"], **result}

    # -- the watcher --------------------------------------------------------------------------------

    def observe(self, operation: ConnectionOperation) -> None:
        for target in operation.targets:
            work = self.work.get(target.name)
            if operation.settled or target.state != TargetState.initiated or work is None or not work.done.is_set():
                continue
            self.work.pop(target.name, None)
            if work.error:
                _fail(operation, target, work.error)
                continue
            extra, detail = _installed_row(target, work.outcome)
            _move(operation, target, TargetState.connected, Actor.backend_watcher, detail=detail, **extra)

    def close(self) -> None:
        if self.op_id is not None:
            _RUNNERS.pop(self.op_id, None)

    def _detail(self, exc: Any, target: Target) -> str:
        from agent.redact import redact_sensitive_text

        text = str(exc) or exc.__class__.__name__
        # Only credentials are secret: the Advanced choices (a profile name, a commit) stay readable.
        for key, value in self.approved_env.get(target.name, {}).items():
            if key not in _OPTION_KEYS and value and len(value) > 3:
                text = text.replace(value, "[REDACTED]")
        return redact_sensitive_text(text, force=True) or "error"


def target_declared_env(fact: Any) -> list[str]:
    caps = getattr(fact, "capabilities", None)
    return list(getattr(caps, "requires_env", None) or ())


def _plugin_row(entry: Any, label: Optional[str]) -> dict[str, Any]:
    from hermes_cli.plugin_catalog_presence import presence

    row: dict[str, Any] = {
        "display": getattr(entry, "title", "") or _display(entry.name),
        "description": _first_sentence(entry.description),
        "tier": entry.tier if entry.tier in _TIERS else "community",
        "repo": entry.repo,
        "sha": entry.sha,
        "requires_hermes": entry.requires_hermes or None,
        "has_desktop_half": False,
        "target_profile": label,
        "app_state": presence(entry).state,
    }
    if entry.platforms:
        row["platforms"] = list(entry.platforms)
    if entry.subdir:
        row["subdir"] = entry.subdir
    return row


def _installed_row(target: Target, outcome: dict[str, Any]) -> tuple:
    """The connected row's fields (the drawn row plus what went live, as facts the card words) and
    the same notes in one English line for the model."""
    extra = dict(target.extra)
    if target.kind == "skill":
        already = bool(outcome.get("already_installed"))
        detail = "already installed; left as it is (Advanced, force reinstall replaces it)" if already else ""
        return {**extra, "skill": outcome["name"], "tools": [], "already_installed": already}, detail
    live = (outcome.get("activation") or {}).get("live_now") or {}
    servers = live.get("mcp_servers") or []
    tools = [name for server in servers if server.get("connected") for name in server.get("tools") or ()]
    server_errors = [{"name": s["name"], "error": s.get("error") or ""} for s in servers if not s.get("connected")]
    enabled = bool(outcome.get("enabled", True))
    missing_env = list(outcome.get("missing_env") or ())
    notes = [f"MCP server {e['name']} not connected: {e['error'] or 'unknown error'}" for e in server_errors]
    if not enabled:
        notes.append("installed but not enabled")
    if missing_env:
        notes.append(f"set {', '.join(missing_env)} to finish setup")
    skills = [s["name"] for s in live.get("skills") or () if s.get("name")]
    if skills:
        extra["skill"] = skills[0]
    facts = {"tools": tools, "enabled": enabled, "missing_env": missing_env, "server_errors": server_errors}
    return {**extra, **facts}, "; ".join(notes)


def _save_credentials(env: dict[str, str]) -> None:
    from hermes_cli.config import save_env_value, validate_env_var_name_for_write

    for key, value in env.items():
        validate_env_var_name_for_write(key)
        save_env_value(key, value)


# op_id -> the runner driving it, so the card's answer (RPC thread) finds the work.
_RUNNERS: dict[str, _Runner] = {}


def owns(op_id: str) -> bool:
    return op_id in _RUNNERS


def apply_answer(operation: ConnectionOperation, raw: str) -> None:
    """The card's ``connection.respond`` for a catalog operation: skip, approve (with the Advanced
    values or null for defaults), approve again on a failed row (Try again), and Continue."""
    try:
        answer = json.loads(raw)
    except (TypeError, ValueError):
        answer = {}
    answer = answer if isinstance(answer, dict) else {}
    runner = _RUNNERS.get(operation.op_id)
    for entry in answer.get("targets") or ():
        if not isinstance(entry, dict):
            continue
        target = operation.target(str(entry.get("name") or "").strip())
        if target is None:
            continue
        status = str(entry.get("status") or "").lower()
        if status == "skipped":
            try:
                operation.transition(target.name, TargetState.skipped, Actor.user)
            except IllegalTransition:
                if not target.resolved and not operation.settled:
                    raise
        elif status == "approved" and runner is not None and target.state in (TargetState.pending, TargetState.failed):
            env = entry.get("env")
            runner.approve(operation, target, {str(k): str(v) for k, v in env.items()} if isinstance(env, dict) else None)
    if answer.get("settled_by") == SettleReason.continue_.value and not operation.all_resolved:
        operation.settle(SettleReason.continue_)


def retry(operation: ConnectionOperation, names: list[str]) -> Optional[str]:
    runner = _RUNNERS.get(operation.op_id)
    if runner is None or operation.settled:
        return "this operation has settled; its result is frozen"
    for name in names:
        target = operation.target(name)
        if target is not None and target.state == TargetState.failed:
            runner.approve(operation, target, None)
    return None


def open_runner(installer: Optional[HostInstaller] = None) -> _Runner:
    return _Runner(installer or HostInstaller())


Callback = Callable[[dict[str, Any]], Optional[str]]
