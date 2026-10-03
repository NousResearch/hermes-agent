"""``manage_catalog`` install targets: catalog plugins and hub skills, installed through the card.

``prepare`` resolves every id against the plugin catalog (the Plugins tab's resolver) or the skills
hub and fills the row the card draws; an id that does not resolve, or a plugin this OS cannot run,
is drawn failed with the reason. Nothing installs until the user approves a row. The install runs on
a worker under the TARGET profile's runtime scope (``default`` unless the Advanced modal named
another), through the same host install the Plugins tab uses, so the catalog pin, the kill list, the
security scan and the live activation of the plugin's MCP servers and skills are the host's.
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
from typing import Any, Callable, Dict, List, Optional

from tools.connectors.contract import Actor, SettleReason, TargetState
from tools.connectors.mcp import _fail, _move
from tools.connectors.operation import ConnectionOperation, IllegalTransition, Target

logger = logging.getLogger(__name__)

DEFAULT_PROFILE = "default"
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
def target_scope(profile: str):
    """Bind the named profile's home, secrets and terminal policy, the way an RPC for that profile
    does. The install writes its tree, config and ``.env`` there, never into the setup profile. The
    home override is bound for the launch profile too: the calling thread carries the setup
    profile's override, and an unbound launch scope would leave it in place."""
    from tui_gateway import server
    from tui_gateway.launch_profile_policy import launch_profile_runtime_scope

    home = server._profile_home(profile)  # None = the launch profile; raises for a missing one
    scope = (launch_profile_runtime_scope(server._hermes_home) if home is None
             else server._session_profile_runtime_scope({"profile_home": str(home)}))
    with scope:
        yield


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

    def install_plugin(self, name: str, *, force: bool, enable: bool, ref: Optional[str]) -> Dict[str, Any]:
        from hermes_cli.plugins_cmd import dashboard_install_plugin

        return dashboard_install_plugin("", force=force, enable=enable, catalog_name=name, ref=ref)

    def skill_meta(self, identifier: str) -> Optional[Dict[str, Any]]:
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

    def install_skill(self, identifier: str, *, force: bool) -> Dict[str, Any]:
        """Install headless; ``{name, already_installed}``. ``do_install`` reports only by printing, so
        success is read from the hub lock file and failure from its last line. A skill that is already
        installed (force off) is left as it is and reported so."""
        from rich.console import Console

        from hermes_cli.skills_hub import do_install
        from tools.skills_hub import HubLockFile

        def entry() -> Optional[Dict[str, Any]]:
            return next((e for e in HubLockFile().list_installed() if e.get("identifier") == identifier), None)

        before = entry()
        if before is not None and not force:
            return {"name": str(before["name"]), "already_installed": True}
        out = io.StringIO()
        do_install(identifier, force=force, skip_confirm=True,
                   console=Console(file=out, width=200, no_color=True, highlight=False))
        after = entry()
        if after is None or (before is not None and after.get("updated_at") == before.get("updated_at")):
            lines = [line.strip() for line in out.getvalue().splitlines() if line.strip()]
            raise RuntimeError(lines[-1] if lines else "the skill was not installed")
        return {"name": str(after["name"]), "already_installed": False}


@dataclass
class _Work:
    done: threading.Event = field(default_factory=threading.Event)
    outcome: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


class _Runner:
    """Resolved catalog facts and install work for one operation's rows."""

    def __init__(self, installer: HostInstaller):
        self.installer = installer
        self.op_id: Optional[str] = None
        self.facts: Dict[str, Any] = {}  # row name -> PluginCatalogEntry | skill meta; never on the wire
        self.work: Dict[str, _Work] = {}
        # The Advanced values the user approved per row; Try again (env null) reuses them. Values
        # are credentials in part, so they stay here, off the target.
        self.approved_env: Dict[str, Dict[str, str]] = {}

    # -- prepare: resolve each id and draw its row ------------------------------------------------

    def prepare(self, operation: ConnectionOperation) -> None:
        _RUNNERS[operation.op_id] = self
        self.op_id = operation.op_id
        for target in operation.targets:
            target.extra = {"display": _display(target.name), "target_profile": DEFAULT_PROFILE}
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
            target.extra = _plugin_row(entry)
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
            "target_profile": DEFAULT_PROFILE,
        }

    # -- the card's answer ------------------------------------------------------------------------

    def approve(self, operation: ConnectionOperation, target: Target, env: Optional[Dict[str, str]]) -> None:
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
        if not _move(operation, target, TargetState.initiated, actor, detail=""):
            return
        self._spawn(operation, target, approved)

    def _check_answer(self, target: Target, env: Dict[str, str]) -> str:
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

    def _spawn(self, operation: ConnectionOperation, target: Target, env: Dict[str, str]) -> None:
        work = _Work()
        self.work[target.name] = work

        def body() -> None:
            try:
                work.outcome = self._install(target, env)
            except Exception as exc:
                work.error = self._detail(exc, target)
            work.done.set()
            operation.wake.set()

        # A copy of the answering thread's context; the install binds the target profile itself.
        threading.Thread(target=contextvars.copy_context().run, args=(body,), daemon=True,
                         name=f"catalog-install-{target.name}").start()

    def _install(self, target: Target, env: Dict[str, str]) -> Dict[str, Any]:
        profile = (env.get("target_profile") or DEFAULT_PROFILE).strip()
        force = _flag(env.get("force"), False)
        credentials = {k: v for k, v in env.items() if k not in _OPTION_KEYS and v}
        with target_scope(profile), _saved_credentials(credentials):
            if target.kind == "skill":
                identifier = str(self.facts[target.name].get("identifier") or target.name)
                return {"profile": profile, **self.installer.install_skill(identifier, force=force)}
            enable = _flag(env.get("enable"), True)
            result = self.installer.install_plugin(target.name, force=force, enable=enable, ref=env.get("ref") or None)
            if not result.get("ok"):
                raise RuntimeError(result.get("error") or "the install failed")
            return {"profile": profile, "enabled": enable, **result}

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
        for value in self.approved_env.get(target.name, {}).values():
            if value and len(value) > 3:
                text = text.replace(value, "[REDACTED]")
        return redact_sensitive_text(text, force=True) or "error"


def target_declared_env(fact: Any) -> List[str]:
    caps = getattr(fact, "capabilities", None)
    return list(getattr(caps, "requires_env", None) or ())


def _plugin_row(entry: Any) -> Dict[str, Any]:
    requirements = [f"Hermes {entry.requires_hermes}"] if entry.requires_hermes else []
    requirements += [f"{name} environment variable" for name in entry.capabilities.requires_env]
    from hermes_cli.plugin_catalog_presence import presence

    row: Dict[str, Any] = {
        "display": getattr(entry, "title", "") or _display(entry.name),
        "description": _first_sentence(entry.description),
        "tier": entry.tier if entry.tier in _TIERS else "community",
        "repo": entry.repo,
        "sha": entry.sha,
        "requirements": requirements,
        "has_desktop_half": False,
        "target_profile": DEFAULT_PROFILE,
        "app_state": presence(entry).state,
    }
    if entry.platforms:
        row["platforms"] = list(entry.platforms)
    if entry.subdir:
        row["subdir"] = entry.subdir
    return row


def _installed_row(target: Target, outcome: Dict[str, Any]) -> tuple:
    """The connected row's fields (the drawn row plus what went live) and its one-line detail."""
    extra = {**target.extra, "target_profile": outcome["profile"]}
    if target.kind == "skill":
        detail = "already installed; left as it is (Advanced, force reinstall replaces it)" \
            if outcome.get("already_installed") else ""
        return {**extra, "skill": outcome["name"], "tools": []}, detail
    live = (outcome.get("activation") or {}).get("live_now") or {}
    servers = live.get("mcp_servers") or []
    tools = [name for server in servers if server.get("connected") for name in server.get("tools") or ()]
    notes = [f"MCP server {s['name']} not connected: {s.get('error') or 'unknown error'}"
             for s in servers if not s.get("connected")]
    if not outcome.get("enabled", True):
        notes.append("installed but not enabled")
    if outcome.get("missing_env"):
        notes.append(f"set {', '.join(outcome['missing_env'])} to finish setup")
    skills = [s["name"] for s in live.get("skills") or () if s.get("name")]
    if skills:
        extra["skill"] = skills[0]
    return {**extra, "tools": tools}, "; ".join(notes)


def _save_credentials(env: Dict[str, str]) -> None:
    from hermes_cli.config import save_env_value, validate_env_var_name_for_write

    for key, value in env.items():
        validate_env_var_name_for_write(key)
        save_env_value(key, value)


@contextlib.contextmanager
def _saved_credentials(credentials: Dict[str, str]):
    """Persist what the card collected for the install, and take it back out when the install fails.

    The values have to be on disk before the install runs: enabling a plugin activates it in this
    process, and its MCP servers read their keys from the profile's ``.env``. But an install fails
    for ordinary reasons — a blocked scan, a kill-listed identifier, a clone that never lands — and
    a secret for a plugin that was never installed must not be left behind: nothing here would ever
    remove it, and the card reports only the install error. The MCP card next door writes its
    secrets only after its own install succeeds (``tools/connectors/mcp.py::install``); this path
    cannot defer the write, so it undoes it, restoring whatever each key held before.
    """
    from hermes_cli.config import invalidate_env_cache, load_env

    try:
        invalidate_env_cache()  # the memo can predate another process's write
        existing = load_env() or {}
    except Exception:  # an unreadable .env must not block the install the user asked for
        existing = {}
    previous = {key: existing.get(key) for key in credentials}
    try:
        # Inside the try: a rejected variable name raises partway, and the keys written before it
        # have to come back out too.
        _save_credentials(credentials)
        yield
    except BaseException:
        _restore_credentials(previous)
        raise


def _restore_credentials(previous: Dict[str, Optional[str]]) -> None:
    """Put each key back the way the failed install found it: removed when it was absent."""
    from hermes_cli.config import remove_env_value, save_env_value

    for key, value in previous.items():
        try:
            save_env_value(key, value) if value is not None else remove_env_value(key)
        except Exception as exc:  # report it; the install error is what the card shows
            logger.warning("could not roll back %s after a failed install: %s", key, exc)


# op_id -> the runner driving it, so the card's answer (RPC thread) finds the work.
_RUNNERS: Dict[str, _Runner] = {}


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


def retry(operation: ConnectionOperation, names: List[str]) -> Optional[str]:
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


Callback = Callable[[Dict[str, Any]], Optional[str]]
