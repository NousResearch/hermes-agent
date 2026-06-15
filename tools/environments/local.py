"""Local execution environment — spawn-per-call with session snapshot."""

import logging
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from tools.environments.base import BaseEnvironment, _pipe_stdin
from hermes_cli._subprocess_compat import windows_hide_flags

_IS_WINDOWS = platform.system() == "Windows"

logger = logging.getLogger(__name__)


def _msys_to_windows_path(cwd: str) -> str:
    """Translate a Git Bash / MSYS-style POSIX path (``/c/Users/x``) to the
    native Windows form (``C:\\Users\\x``) so ``os.path.isdir`` and
    ``subprocess.Popen(..., cwd=...)`` can find it.

    No-ops on non-Windows hosts or for paths that aren't in MSYS form.
    Returns the input unchanged when no translation applies. This is
    idempotent — calling it on an already-Windows path returns it as-is.
    """
    if not _IS_WINDOWS or not cwd:
        return cwd
    # Match leading "/<single letter>/" or exactly "/<letter>" (bare drive root).
    m = re.match(r'^/([a-zA-Z])(/.*)?$', cwd)
    if not m:
        return cwd
    drive = m.group(1).upper()
    tail = (m.group(2) or "").replace('/', '\\')
    return f"{drive}:{tail or chr(92)}"  # chr(92) = backslash, avoid raw-string escape


def _resolve_safe_cwd(cwd: str) -> str:
    """Return ``cwd`` if it exists as a directory, else the nearest existing
    ancestor.  Falls back to ``tempfile.gettempdir()`` only if walking up the
    path can't find any existing directory (effectively never on a healthy
    filesystem, but cheap belt-and-braces).

    On Windows, also normalizes Git Bash / MSYS-style POSIX paths
    (``/c/Users/x``) to native Windows form before the isdir check so a
    perfectly valid ``pwd -P`` result from bash doesn't get rejected as
    "missing" (see ``_msys_to_windows_path``).

    Used by ``_run_bash`` to recover when the configured cwd is gone — most
    commonly because a previous tool call deleted its own working directory
    (issue #17558).  Without this guard, ``subprocess.Popen(..., cwd=...)``
    raises ``FileNotFoundError`` before bash starts, wedging every subsequent
    terminal call until the gateway restarts.
    """
    cwd = _msys_to_windows_path(cwd) if _IS_WINDOWS else cwd
    if cwd and os.path.isdir(cwd):
        return cwd
    parent = os.path.dirname(cwd) if cwd else ""
    while parent:
        if os.path.isdir(parent):
            return parent
        next_parent = os.path.dirname(parent)
        if next_parent == parent:
            # Reached the filesystem root and it doesn't exist either —
            # genuinely nothing to fall back to except the temp dir.
            break
        parent = next_parent
    return tempfile.gettempdir()


# Hermes-internal env vars that should NOT leak into terminal subprocesses.
_HERMES_PROVIDER_ENV_FORCE_PREFIX = "_HERMES_FORCE_"

# Hermes-managed AWS *inference* credentials for ``auth_type="aws_sdk"``
# providers (Bedrock).  Scoped DELIBERATELY NARROW: this lists only the
# Bedrock-specific bearer token, which is a Hermes inference secret exactly
# analogous to ``OPENAI_API_KEY`` — nobody drives the ``aws``/``terraform``/
# ``boto3`` toolchain off it, so stripping it from terminal/execute_code
# subprocesses costs no user capability.
#
# The GENERAL AWS credential chain (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
# AWS_SESSION_TOKEN, AWS_PROFILE, and the config/role pointers) is INTENTIONALLY
# left inheritable.  Per SECURITY.md §3.2 the local terminal is the user's
# trusted operator shell; the agent having the same general AWS access the
# user's own shell has is the intended posture, not a leak.  Hard-blocklisting
# those vars would (a) regress every user who runs aws/terraform/cdk/boto3 in
# the agent terminal — not just Bedrock users, since the registry is iterated
# unconditionally — and (b) be unrecoverable, because env_passthrough.py
# refuses to re-allow anything in this blocklist (GHSA-rhgp-j443-p4rf).  See
# issue #32314 discussion.
_AWS_SDK_CREDENTIAL_ENV_VARS = frozenset({
    "AWS_BEARER_TOKEN_BEDROCK",
})


def _build_provider_env_blocklist() -> frozenset:
    """Derive the blocklist from provider, tool, and gateway config."""
    blocked: set[str] = set()

    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
        for pconfig in PROVIDER_REGISTRY.values():
            blocked.update(pconfig.api_key_env_vars)
            if pconfig.auth_type == "aws_sdk":
                blocked.update(_AWS_SDK_CREDENTIAL_ENV_VARS)
            if pconfig.base_url_env_var:
                blocked.add(pconfig.base_url_env_var)
    except ImportError:
        pass

    try:
        from hermes_cli.config import OPTIONAL_ENV_VARS
        for name, metadata in OPTIONAL_ENV_VARS.items():
            category = metadata.get("category")
            if category in {"tool", "messaging"}:
                blocked.add(name)
            elif category == "setting" and metadata.get("password"):
                blocked.add(name)
    except ImportError:
        pass

    blocked.update({
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_API_BASE",
        "OPENAI_ORG_ID",
        "OPENAI_ORGANIZATION",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "LLM_MODEL",
        "GOOGLE_API_KEY",
        "DEEPSEEK_API_KEY",
        "MISTRAL_API_KEY",
        "GROQ_API_KEY",
        "TOGETHER_API_KEY",
        "PERPLEXITY_API_KEY",
        "COHERE_API_KEY",
        "FIREWORKS_API_KEY",
        "XAI_API_KEY",
        "HELICONE_API_KEY",
        "PARALLEL_API_KEY",
        "FIRECRAWL_API_KEY",
        "FIRECRAWL_API_URL",
        "TELEGRAM_HOME_CHANNEL",
        "TELEGRAM_HOME_CHANNEL_NAME",
        "DISCORD_HOME_CHANNEL",
        "DISCORD_HOME_CHANNEL_NAME",
        "DISCORD_REQUIRE_MENTION",
        "DISCORD_FREE_RESPONSE_CHANNELS",
        "DISCORD_AUTO_THREAD",
        "SLACK_HOME_CHANNEL",
        "SLACK_HOME_CHANNEL_NAME",
        "SLACK_ALLOWED_USERS",
        "WHATSAPP_ENABLED",
        "WHATSAPP_MODE",
        "WHATSAPP_ALLOWED_USERS",
        "SIGNAL_HTTP_URL",
        "SIGNAL_ACCOUNT",
        "SIGNAL_ALLOWED_USERS",
        "SIGNAL_GROUP_ALLOWED_USERS",
        "SIGNAL_HOME_CHANNEL",
        "SIGNAL_HOME_CHANNEL_NAME",
        "SIGNAL_IGNORE_STORIES",
        "HASS_TOKEN",
        "HASS_URL",
        "EMAIL_ADDRESS",
        "EMAIL_PASSWORD",
        "EMAIL_IMAP_HOST",
        "EMAIL_SMTP_HOST",
        "EMAIL_HOME_ADDRESS",
        "EMAIL_HOME_ADDRESS_NAME",
        "HERMES_DASHBOARD_SESSION_TOKEN",
        "GATEWAY_ALLOWED_USERS",
        "GH_TOKEN",
        "GITHUB_APP_ID",
        "GITHUB_APP_PRIVATE_KEY_PATH",
        "GITHUB_APP_INSTALLATION_ID",
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "DAYTONA_API_KEY",
        "GATEWAY_RELAY_ID",
        "GATEWAY_RELAY_SECRET",
        "GATEWAY_RELAY_DELIVERY_KEY",
    })
    return frozenset(blocked)


_HERMES_PROVIDER_ENV_BLOCKLIST = _build_provider_env_blocklist()

# Active-virtualenv markers that must NOT leak into terminal subprocesses.
# The gateway runs inside its own venv, so its process environment carries
# VIRTUAL_ENV (and possibly CONDA_PREFIX). If those leak into commands the
# agent runs against OTHER Python projects, tools like ``uv``/``poetry`` treat
# the inherited value as the active environment and build/sync that other
# project's dependencies into the Hermes venv path instead of the project's own
# ``.venv`` — silently clobbering the Hermes environment (e.g. a project pinned
# to a different Python version overwrites it and breaks the gateway). The
# Hermes venv stays reachable via PATH (its bin dir is first), so stripping
# these markers is safe and only prevents the cross-project clobber (#23473).
_ACTIVE_VENV_MARKER_VARS = ("VIRTUAL_ENV", "CONDA_PREFIX")


def _is_hermes_internal_secret(key: str) -> bool:
    """Return True for Hermes-internal secrets injected under *dynamic* names.

    ``_HERMES_PROVIDER_ENV_BLOCKLIST`` is name-based and derived from the
    provider/tool registries, but the gateway and CLI also inject secrets into
    ``os.environ`` at runtime under names no static registry knows about:

    - ``AUXILIARY_<TASK>_API_KEY`` / ``AUXILIARY_<TASK>_BASE_URL`` — per-task
      side-LLM credentials bridged from ``config.yaml[auxiliary]`` by
      ``gateway/run.py`` and ``cli.py`` (vision, web_extract, approval,
      compression, and any plugin-registered auxiliary task). These are
      separate, often higher-spend API keys plus base URLs that may point at
      private endpoints; a model-authored shell command must never see them.
    - ``GATEWAY_RELAY_*_SECRET`` / ``GATEWAY_RELAY_*_KEY`` /
      ``GATEWAY_RELAY_*_TOKEN`` — relay-auth material provisioned by the
      gateway (``GATEWAY_RELAY_SECRET``, ``GATEWAY_RELAY_DELIVERY_KEY``).
      These are Tier-1 gateway secrets, like the messaging bot tokens in
      ``_ALWAYS_STRIP_KEYS``. Non-secret ``GATEWAY_RELAY_*`` routing hints
      (``GATEWAY_RELAY_URL``, ``GATEWAY_RELAY_PLATFORMS``, …) are NOT matched
      and remain visible.

    ``code_execution_tool.py`` already catches these via substring matching on
    ``KEY`` / ``SECRET`` / ``TOKEN``; the terminal backend's narrower name-based
    blocklist did not, which is the leak this predicate closes.

    This is the single source of truth for "Hermes-internal dynamic secret"
    across every spawn path — the terminal ``_make_run_env`` /
    ``_sanitize_subprocess_env`` filters, the Docker passthrough filter, and the
    non-terminal :func:`hermes_subprocess_env` helper all call it, so the
    dynamic patterns are stripped **unconditionally** regardless of
    ``env_passthrough`` skill registration or ``inherit_credentials``. Nothing
    a model-driving CLI legitimately needs matches these patterns.
    """
    upper = key.upper()
    if upper.startswith("AUXILIARY_") and (
        upper.endswith("_API_KEY") or upper.endswith("_BASE_URL")
    ):
        return True
    if upper.startswith("GATEWAY_RELAY_") and (
        upper.endswith("_SECRET") or upper.endswith("_KEY") or upper.endswith("_TOKEN")
    ):
        return True
    return False


def _inject_context_hermes_home(env: dict) -> None:
    """Bridge the context-local Hermes home override into subprocess env."""
    try:
        from hermes_constants import get_hermes_home_override

        value = get_hermes_home_override()
        if value:
            env["HERMES_HOME"] = value
    except Exception:
        pass


def _inject_session_context_env(env: dict) -> None:
    """Bridge gateway session ContextVars into a subprocess environment dict.

    ContextVars don't propagate to child processes, so the live session vars
    (HERMES_SESSION_*) are bridged onto the child env here.

    🔴 Cross-session leak guard. The session vars also have a process-global
    os.environ mirror (written last-writer-wins as a CLI/cron fallback, never
    cleared). Under a concurrent multi-session host (the messaging gateway, ACP
    adapter, API server, TUI) that global belongs to *whichever turn wrote it
    last* — NOT necessarily this task. A subprocess spawned from a task whose
    ContextVar is _UNSET (e.g. a sibling message task that never bound, or one
    that inherited another session's context) would otherwise inherit the
    FOREIGN global and act on another session's identity.

    So once the session-context machinery is engaged in this process (any host
    has called set_session_vars), the session vars are ContextVar-authoritative:
    - ContextVar set (incl. explicitly-empty "") → that value wins, overriding
      any stale snapshot/global value.
    - ContextVar _UNSET → STRIP the var from the child env rather than inherit
      the possibly-foreign process-global.
    In a pure single-process CLI/one-shot that never engaged the session-context
    system there is no concurrency to leak across, so the inherited fallback is
    kept. See gateway/session_context.session_context_engaged and
    tests/tools/test_local_env_session_leak.py.
    """
    try:
        from gateway.session_context import (
            _UNSET,
            _VAR_MAP,
            session_context_engaged,
        )
    except Exception:
        return

    _engaged = session_context_engaged()
    for var_name, var in _VAR_MAP.items():
        value = var.get()
        if value is not _UNSET:
            # Explicitly bound (including "") — authoritative for this task.
            env[var_name] = "" if value is None else str(value)
        elif _engaged:
            # Unset for THIS task while a concurrent host is engaged: drop any
            # inherited global so a sibling session's value can't leak in.
            env.pop(var_name, None)


def _sanitize_subprocess_env(base_env: dict | None, extra_env: dict | None = None) -> dict:
    """Filter Hermes-managed secrets from a subprocess environment."""
    try:
        from tools.env_passthrough import is_env_passthrough as _is_passthrough
    except Exception:
        _is_passthrough = lambda _: False  # noqa: E731

    sanitized: dict[str, str] = {}

    for key, value in (base_env or {}).items():
        if key.startswith(_HERMES_PROVIDER_ENV_FORCE_PREFIX):
            continue
        if _is_hermes_internal_secret(key):
            continue
        if key not in _HERMES_PROVIDER_ENV_BLOCKLIST or _is_passthrough(key):
            sanitized[key] = value

    for key, value in (extra_env or {}).items():
        if key.startswith(_HERMES_PROVIDER_ENV_FORCE_PREFIX):
            real_key = key[len(_HERMES_PROVIDER_ENV_FORCE_PREFIX):]
            if _is_hermes_internal_secret(real_key):
                continue
            sanitized[real_key] = value
        elif _is_hermes_internal_secret(key):
            continue
        elif key not in _HERMES_PROVIDER_ENV_BLOCKLIST or _is_passthrough(key):
            sanitized[key] = value

    _inject_context_hermes_home(sanitized)

    from hermes_constants import apply_subprocess_home_env
    apply_subprocess_home_env(sanitized)

    # Same cross-session leak guard as _make_run_env, for the background/PTY
    # spawn path (process_registry.spawn_local builds env via this function).
    _inject_session_context_env(sanitized)

    for _marker in _ACTIVE_VENV_MARKER_VARS:
        sanitized.pop(_marker, None)

    return sanitized


# Tier-1 secrets: stripped from EVERY spawned subprocess unconditionally —
# even when the caller opts into credential inheritance for a model-driving
# CLI (claude / codex / gemini).  These are not LLM provider credentials; no
# legitimate child Hermes spawns needs them, and they are the highest-value
# secrets to keep out of a compromised dependency's reach (gateway bot tokens,
# GitHub auth, remote-compute tokens, dashboard session secret).  The set is a
# narrow subset of _HERMES_PROVIDER_ENV_BLOCKLIST; provider keys are handled by
# the conditional Tier-2 strip in hermes_subprocess_env().
_ALWAYS_STRIP_KEYS: frozenset[str] = frozenset({
    # GitHub auth
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "GITHUB_APP_ID",
    "GITHUB_APP_PRIVATE_KEY_PATH",
    "GITHUB_APP_INSTALLATION_ID",
    # Gateway / messaging bot tokens and access control
    "TELEGRAM_BOT_TOKEN",
    "DISCORD_BOT_TOKEN",
    "SLACK_BOT_TOKEN",
    "SLACK_APP_TOKEN",
    "SLACK_SIGNING_SECRET",
    "GATEWAY_ALLOWED_USERS",
    "GATEWAY_ALLOW_ALL_USERS",
    # Gateway relay auth — the ID/secret/delivery-key triplet the gateway
    # provisions and persists to the 0600 .env. Stripped unconditionally on
    # EVERY spawn surface (terminal + model-driving CLIs) so it can't drift
    # between paths: _SECRET / _DELIVERY_KEY are also matched by
    # _is_hermes_internal_secret, but _ID has no secret suffix, so it must be
    # enumerated here to stay stripped on the inherit_credentials=True path
    # (codex / copilot), which skips the Tier-2 blocklist.
    "GATEWAY_RELAY_ID",
    "GATEWAY_RELAY_SECRET",
    "GATEWAY_RELAY_DELIVERY_KEY",
    "HASS_TOKEN",
    "EMAIL_PASSWORD",
    "HERMES_DASHBOARD_SESSION_TOKEN",
    # Remote-compute / infrastructure secrets
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
    "DAYTONA_API_KEY",
})


def hermes_subprocess_env(*, inherit_credentials: bool = False) -> dict[str, str]:
    """Build a sanitized environment dict for a spawned subprocess.

    Centralized helper for the **non-terminal** spawn surface (browser,
    ACP/CLI executors, computer-use driver, dep-ensure, TUI Node host,
    detached gateway).  Use this instead of copying ``os.environ`` directly
    so strip-by-default is the uniform policy across every spawn site, with a
    single source of truth (``_HERMES_PROVIDER_ENV_BLOCKLIST``).  The terminal
    / execute_code path keeps using :func:`_sanitize_subprocess_env`, which is
    skill-aware (``env_passthrough``); this helper is for spawns that have no
    skill-passthrough concept.

    Two-tier stripping:

    * **Tier 1 (always):** ``_ALWAYS_STRIP_KEYS`` — gateway bot tokens, GitHub
      auth, and remote-compute secrets are removed regardless of
      ``inherit_credentials``.  No child Hermes spawns legitimately needs them.
    * **Tier 2 (conditional):** the rest of ``_HERMES_PROVIDER_ENV_BLOCKLIST``
      (LLM provider API keys, tool secrets) is removed unless the caller passes
      ``inherit_credentials=True``.

    Pass ``inherit_credentials=True`` **only** when the child legitimately
    needs LLM provider credentials — a user-blessed ``claude`` / ``codex`` /
    ``gemini`` CLI executor, or the TUI Node host that makes model calls.  The
    flag is grep-able for audit: ``grep -rn 'inherit_credentials=True'`` lists
    every spawn site that still receives provider credentials.

    Callers that need a *specific* non-provider secret (e.g. the browser worker
    needs ``BROWSERBASE_API_KEY`` / ``FIRECRAWL_API_KEY``) should call with
    ``inherit_credentials=False`` and copy just those keys back from
    ``os.environ`` into the returned dict.
    """
    env = os.environ.copy()

    # Tier 1 — always strip.
    for key in _ALWAYS_STRIP_KEYS:
        env.pop(key, None)
    # Internal routing hints and Hermes-internal dynamic secrets
    # (``AUXILIARY_<TASK>_API_KEY`` / ``_BASE_URL`` side-LLM credentials,
    # ``GATEWAY_RELAY_*`` relay-auth material) must never reach a child,
    # regardless of ``inherit_credentials`` — a model-driving CLI has no
    # legitimate use for them. See :func:`_is_hermes_internal_secret`.
    for key in list(env):
        if key.startswith(_HERMES_PROVIDER_ENV_FORCE_PREFIX):
            env.pop(key, None)
        elif _is_hermes_internal_secret(key):
            env.pop(key, None)

    if not inherit_credentials:
        # Tier 2 — strip provider/tool credentials unless explicitly inherited.
        for key in _HERMES_PROVIDER_ENV_BLOCKLIST:
            env.pop(key, None)

    # Windows UTF-8 safety for spawned processes (#31420).
    env.setdefault("PYTHONUTF8", "1")

    _inject_context_hermes_home(env)
    from hermes_constants import apply_subprocess_home_env
    apply_subprocess_home_env(env)

    # Active-venv markers must not clobber another project's environment.
    for _marker in _ACTIVE_VENV_MARKER_VARS:
        env.pop(_marker, None)

    # Cross-session leak guard, same as the terminal spawn paths: this helper
    # copies os.environ, whose HERMES_SESSION_* mirror is a last-writer-wins
    # global under a concurrent multi-session host. A caller that re-binds the
    # session identity explicitly (slash_worker/ACP via --session-key argv) is
    # unaffected — bound ContextVars win here — but a caller that spawns without
    # re-binding (e.g. tui_gateway cli.exec) would otherwise inherit a FOREIGN
    # session's identity. Strip _UNSET session vars when engaged so that can't
    # happen; single uniform policy across every spawn surface.
    _inject_session_context_env(env)

    return env


def _find_bash() -> str:
    """Find bash for command execution."""
    if not _IS_WINDOWS:
        return (
            shutil.which("bash")
            or ("/usr/bin/bash" if os.path.isfile("/usr/bin/bash") else None)
            or ("/bin/bash" if os.path.isfile("/bin/bash") else None)
            or os.environ.get("SHELL")
            or "/bin/sh"
        )

    custom = os.environ.get("HERMES_GIT_BASH_PATH")
    if custom and os.path.isfile(custom):
        return custom

    # Prefer our own portable Git install first — this way a broken or
    # partially-uninstalled system Git can't hijack the bash lookup.  The
    # install.ps1 installer always drops portable Git here when the user
    # didn't already have a working system Git.
    #
    # Layouts (both checked so upgrades between MinGit and PortableGit
    # installs work transparently):
    #   PortableGit: %LOCALAPPDATA%\hermes\git\bin\bash.exe   (primary)
    #   MinGit:      %LOCALAPPDATA%\hermes\git\usr\bin\bash.exe (legacy/32-bit fallback)
    _local_appdata = os.environ.get("LOCALAPPDATA", "")
    _hermes_portable_git = os.path.join(_local_appdata, "hermes", "git") if _local_appdata else ""
    if _hermes_portable_git:
        for candidate in (
            os.path.join(_hermes_portable_git, "bin", "bash.exe"),        # PortableGit (primary)
            os.path.join(_hermes_portable_git, "usr", "bin", "bash.exe"), # MinGit fallback
        ):
            if os.path.isfile(candidate):
                return candidate

    # Check known Git for Windows install locations before PATH lookup.
    # On machines with both WSL and Git for Windows, shutil.which("bash")
    # may return WSL's bash (which doesn't understand Windows paths and
    # will fail silently).  Explicit Git-for-Windows paths avoid that.
    for candidate in (
        os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Git", "bin", "bash.exe"),
        os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Git", "bin", "bash.exe"),
        os.path.join(_local_appdata, "Programs", "Git", "bin", "bash.exe"),
    ):
        if candidate and os.path.isfile(candidate):
            return candidate

    found = shutil.which("bash")
    if found:
        return found

    raise RuntimeError(
        "Git Bash not found. Hermes Agent requires Git for Windows on Windows.\n"
        "Install it from: https://git-scm.com/download/win\n"
        "Or set HERMES_GIT_BASH_PATH to your bash.exe location."
    )


# POSIX-sh-family shells that understand the ``[shell, "-lic", "set +m; …"]``
# invocation spawn_local uses. $SHELL values outside this set (fish, csh/tcsh,
# nushell, elvish, xonsh, …) would error on that syntax, so _find_shell falls
# back to bash for them rather than honouring $SHELL. (#42203)
_SPAWN_COMPATIBLE_SHELLS = frozenset({"bash", "zsh", "sh", "dash", "ksh", "mksh"})


def _find_shell() -> str:
    """Find the user's login shell for background process spawning.

    Unlike ``_find_bash`` (which always returns a bash binary for callers
    that explicitly need bash), this function prefers the user's configured
    ``$SHELL`` on POSIX so that ``spawn_local`` uses the shell the user
    actually logs in with.

    On macOS Catalina+ the default login shell is zsh, but
    ``shutil.which("bash")`` still finds the system ``/bin/bash`` (GNU bash
    3.2).  When bash 3.2 is invoked with ``-l`` (login) and stdin is
    ``/dev/null``, it sources ``~/.bash_profile`` which on many macOS setups
    contains ``exec /bin/zsh -l``.  That ``exec`` replaces bash with zsh but
    drops the ``-c`` argument, so the background command never runs — the
    subprocess exits 0 with no output and no side effects.

    Preferring ``$SHELL`` (when it is a POSIX-``sh``-family shell) avoids this
    because zsh/bash/sh/dash/ksh handle ``-lic`` correctly even with
    redirected stdin.

    Only POSIX-sh-family shells are honoured: ``spawn_local`` invokes the
    shell as ``[shell, "-lic", "set +m; <cmd>"]``, and that ``-lic`` bundle +
    ``set +m`` job-control syntax is NOT understood by fish, csh/tcsh,
    nushell, elvish, xonsh, etc.  Returning such a ``$SHELL`` would trade the
    bash-3.2 swallow for a parse error on every background command, so for any
    non-allowlisted shell we fall back to ``_find_bash`` (the prior behaviour).

    On Windows, ``$SHELL`` is typically bash (Git Bash), so behaviour is
    unchanged — we fall through to ``_find_bash``.
    """
    if not _IS_WINDOWS:
        user_shell = os.environ.get("SHELL")
        if (
            user_shell
            and os.path.isfile(user_shell)
            and os.access(user_shell, os.X_OK)
            and Path(user_shell).name in _SPAWN_COMPATIBLE_SHELLS
        ):
            return user_shell
    return _find_bash()


# Standard PATH entries for environments with minimal PATH.
_SANE_PATH = (
    "/opt/homebrew/bin:/opt/homebrew/sbin:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)

# Cached directory containing the ``hermes`` console-script.
# ``_SENTINEL`` distinguishes "not resolved yet" from a resolved ``None``.
_SENTINEL = object()
_HERMES_BIN_DIR: "str | None | object" = _SENTINEL


def _resolve_hermes_bin_dir() -> str | None:
    """Return the directory holding the ``hermes`` console-script, or None.

    The terminal tool runs in a freshly-spawned subshell whose PATH is the
    agent process's PATH plus a static set of system dirs (``_SANE_PATH``).
    When the gateway is launched by something that does NOT source the user's
    shell rc — systemd, a service manager, a desktop launcher, cron — the
    hermes install dir (``~/.local/bin``, the venv ``bin``/``Scripts``, pipx,
    nix) is absent from that PATH, so plugins shelling out to bare ``hermes``
    via the terminal tool hit ``command not found`` (exit 127) even though
    ``hermes`` works fine in the user's own interactive terminal.

    We resolve the install dir once (it never changes within a process) and
    prepend-if-missing it to the subshell PATH so bare ``hermes`` resolves
    regardless of how the gateway was started.

    Resolution order (cheap, no heavy imports):
      1. ``shutil.which("hermes")`` — normal PATH-installed shim.
      2. The directory of ``sys.argv[0]`` when it's an absolute path to a
         real ``hermes`` executable (covers nix-store / venv wrappers).
      3. The directory of ``sys.executable`` — the running interpreter's
         venv ``bin``/``Scripts`` is where its console-scripts live.
    """
    global _HERMES_BIN_DIR
    if _HERMES_BIN_DIR is not _SENTINEL:
        return _HERMES_BIN_DIR  # type: ignore[return-value]

    candidate: str | None = None

    which = shutil.which("hermes")
    if which:
        candidate = os.path.dirname(which)

    if candidate is None:
        argv0 = sys.argv[0] if sys.argv else ""
        base = os.path.basename(argv0).lower()
        if (
            os.path.isabs(argv0)
            and (base == "hermes" or base.startswith("hermes."))
            and os.path.isfile(argv0)
        ):
            candidate = os.path.dirname(argv0)

    if candidate is None:
        exe_dir = os.path.dirname(sys.executable) if sys.executable else ""
        if exe_dir:
            shim = "hermes.exe" if _IS_WINDOWS else "hermes"
            if os.path.isfile(os.path.join(exe_dir, shim)):
                candidate = exe_dir

    if candidate and not os.path.isdir(candidate):
        candidate = None

    _HERMES_BIN_DIR = candidate
    return candidate


def _prepend_hermes_bin_dir(existing_path: str) -> str:
    """Prepend the hermes install dir to ``existing_path`` if it's missing.

    Cross-platform (uses ``os.pathsep``). First-occurrence wins, so a PATH
    that already contains the dir is returned unchanged. Returns the input
    unchanged when the install dir can't be resolved.
    """
    bin_dir = _resolve_hermes_bin_dir()
    if not bin_dir:
        return existing_path
    sep = os.pathsep
    entries = [e for e in existing_path.split(sep) if e] if existing_path else []
    if bin_dir in entries:
        return existing_path
    return sep.join([bin_dir, *entries])


def _append_missing_sane_path_entries(existing_path: str) -> str:
    """Return a normalised POSIX PATH with missing sane entries appended.

    On POSIX the caller-supplied PATH is rewritten (not merely appended to):
    empty entries and duplicate entries are dropped, preserving
    first-occurrence order, then each missing ``_SANE_PATH`` entry is appended
    once at the end so existing entries keep their precedence.

    Two intentional normalisations beyond the bare "add Homebrew dirs" fix:

    - **Empty entries are stripped.** A leading/trailing/double ``:`` encodes
      an empty PATH element, which POSIX shells interpret as the current
      working directory — a mild foot-gun in a default terminal environment.
      We drop these rather than carry them through.
    - **Duplicates are collapsed** (first occurrence wins), so a caller PATH
      that already contains repeats is not propagated verbatim.

    For a well-formed PATH (no empties, no duplicates) the leading segment is
    byte-identical to the input and ordering is preserved; only the missing
    sane entries are appended. On Windows this is a no-op passthrough (the
    separator is ``;`` and the native PATH must not be touched).
    """
    if _IS_WINDOWS:
        return existing_path

    sane_entries = [entry for entry in _SANE_PATH.split(":") if entry]
    if not existing_path:
        return ":".join(sane_entries)

    # De-duplicate the caller PATH (first occurrence wins) and drop empty
    # entries before merging in the sane fallbacks.
    seen: set[str] = set()
    ordered_entries: list[str] = []
    for entry in existing_path.split(":"):
        if not entry or entry in seen:
            continue
        seen.add(entry)
        ordered_entries.append(entry)

    # _SANE_PATH is a static, duplicate-free constant, so a membership check
    # against the caller entries is sufficient — no need to track `seen` here.
    for entry in sane_entries:
        if entry not in seen:
            ordered_entries.append(entry)

    return ":".join(ordered_entries)


def _path_env_key(run_env: dict) -> str | None:
    """Return the PATH env key to update without altering Windows casing.

    Note: this is deliberately a *second* Windows guard, distinct from the
    early-return in ``_append_missing_sane_path_entries``. Its job is to pick
    the correctly-cased key (``Path`` vs ``PATH``) so completion writes back to
    the key the caller already used; the helper's guard makes that helper safe
    to call standalone (it is, e.g. in the Windows unit tests). Both are
    intentional.
    """
    if not _IS_WINDOWS:
        return "PATH"
    for key in run_env:
        if key.upper() == "PATH":
            return key
    return None


def _make_run_env(env: dict) -> dict:
    """Build a run environment with a sane PATH and provider-var stripping."""
    try:
        from tools.env_passthrough import is_env_passthrough as _is_passthrough
    except Exception:
        _is_passthrough = lambda _: False  # noqa: E731

    merged = dict(os.environ | env)
    run_env = {}
    for k, v in merged.items():
        if k.startswith(_HERMES_PROVIDER_ENV_FORCE_PREFIX):
            real_key = k[len(_HERMES_PROVIDER_ENV_FORCE_PREFIX):]
            if _is_hermes_internal_secret(real_key):
                continue
            run_env[real_key] = v
        elif _is_hermes_internal_secret(k):
            continue
        elif k not in _HERMES_PROVIDER_ENV_BLOCKLIST or _is_passthrough(k):
            run_env[k] = v
    path_key = _path_env_key(run_env)
    if path_key is not None:
        new_path = _append_missing_sane_path_entries(run_env.get(path_key, ""))
        # Ensure the hermes install dir is reachable so plugins can shell out
        # to bare ``hermes`` via the terminal tool even when the gateway was
        # launched without it on PATH (systemd, service managers, cron, etc.).
        run_env[path_key] = _prepend_hermes_bin_dir(new_path)

    _inject_context_hermes_home(run_env)

    from hermes_constants import apply_subprocess_home_env
    apply_subprocess_home_env(run_env)

    # Bridge ContextVar-based session vars into the subprocess env (with the
    # cross-session leak guard — strips _UNSET vars when a concurrent host is
    # engaged so a sibling session's os.environ mirror can't leak in).
    _inject_session_context_env(run_env)

    for _marker in _ACTIVE_VENV_MARKER_VARS:
        run_env.pop(_marker, None)

    return run_env


def _read_terminal_shell_init_config() -> tuple[list[str], bool]:
    """Return (shell_init_files, auto_source_bashrc) from config.yaml.

    Best-effort — returns sensible defaults on any failure so terminal
    execution never breaks because the config file is unreadable.
    """
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        terminal_cfg = cfg.get("terminal") or {}
        files = terminal_cfg.get("shell_init_files") or []
        if not isinstance(files, list):
            files = []
        auto_bashrc = bool(terminal_cfg.get("auto_source_bashrc", True))
        return [str(f) for f in files if f], auto_bashrc
    except Exception:
        return [], True


def _resolve_shell_init_files() -> list[str]:
    """Resolve the list of files to source before the login-shell snapshot.

    Expands ``~`` and ``${VAR}`` references and drops anything that doesn't
    exist on disk, so a missing ``~/.bashrc`` never breaks the snapshot.
    The ``auto_source_bashrc`` path runs only when the user hasn't supplied
    an explicit list — once they have, Hermes trusts them.
    """
    explicit, auto_bashrc = _read_terminal_shell_init_config()

    candidates: list[str] = []
    if explicit:
        candidates.extend(explicit)
    elif auto_bashrc and not _IS_WINDOWS:
        # Build a login-shell-ish source list so tools like n / nvm / asdf /
        # pyenv that self-install into the user's shell rc land on PATH in
        # the captured snapshot.
        #
        # ~/.profile and ~/.bash_profile run first because they have no
        # interactivity guard — installers like ``n`` and ``nvm`` append
        # their PATH export there on most distros, and a non-interactive
        # ``. ~/.profile`` picks that up.
        #
        # ~/.bashrc runs last. On Debian/Ubuntu the default bashrc starts
        # with ``case $- in *i*) ;; *) return;; esac`` and exits early
        # when sourced non-interactively, which is why sourcing bashrc
        # alone misses nvm/n PATH additions placed below that guard. We
        # still include it so users who put PATH logic in bashrc (and
        # stripped the guard, or never had one) keep working.
        candidates.extend(["~/.profile", "~/.bash_profile", "~/.bashrc"])

    resolved: list[str] = []
    for raw in candidates:
        try:
            path = os.path.expandvars(os.path.expanduser(raw))
        except Exception:
            continue
        if path and os.path.isfile(path):
            resolved.append(path)
    return resolved


def _prepend_shell_init(cmd_string: str, files: list[str]) -> str:
    """Prepend ``source <file>`` lines (guarded + silent) to a bash script.

    Each file is wrapped so a failing rc file doesn't abort the whole
    bootstrap: ``set +e`` keeps going on errors, ``2>/dev/null`` hides
    noisy prompts, and ``|| true`` neutralises the exit status.
    """
    if not files:
        return cmd_string

    prelude_parts = ["set +e"]
    for path in files:
        # shlex.quote isn't available here without an import; the files list
        # comes from os.path.expanduser output so it's a concrete absolute
        # path.  Escape single quotes defensively anyway.
        safe = path.replace("'", "'\\''")
        prelude_parts.append(f"[ -r '{safe}' ] && . '{safe}' 2>/dev/null || true")
    prelude = "\n".join(prelude_parts) + "\n"
    return prelude + cmd_string


class LocalEnvironment(BaseEnvironment):
    """Run commands directly on the host machine.

    Spawn-per-call: every execute() spawns a fresh bash process.
    Session snapshot preserves env vars across calls.
    CWD persists via file-based read after each command.
    """

    def __init__(self, cwd: str = "", timeout: int = 60, env: dict = None):
        if cwd:
            cwd = os.path.expanduser(cwd)
        super().__init__(cwd=cwd or os.getcwd(), timeout=timeout, env=env)
        self.init_session()

    def get_temp_dir(self) -> str:
        """Return a shell-safe writable temp dir for local execution.

        Termux does not provide /tmp by default, but exposes a POSIX TMPDIR.
        Prefer POSIX-style env vars when available, keep using /tmp on regular
        Unix systems, and only fall back to tempfile.gettempdir() when it also
        resolves to a POSIX path.

        Check the environment configured for this backend first so callers can
        override the temp root explicitly (for example via terminal.env or a
        custom TMPDIR), then fall back to the host process environment.

        **Windows:** hardcoded ``/tmp`` is wrong in two ways — native Python
        can't open the path, and the Windows default temp (``%TEMP%``) often
        contains spaces (``C:\\Users\\Some Name\\AppData\\Local\\Temp``) that
        break unquoted bash interpolations.  Use a dedicated cache dir under
        ``HERMES_HOME`` instead — single-word path, guaranteed to exist, same
        string resolves in both Git Bash and native Python.
        """
        if _IS_WINDOWS:
            # Derive a Windows-safe temp dir under HERMES_HOME.  Using
            # forward slashes makes the same string work unchanged in bash
            # command interpolations AND in Python ``open()`` — Windows
            # accepts forward slashes in filesystem paths, and we control
            # the path so we can guarantee no spaces.
            try:
                from hermes_constants import get_hermes_home
                cache_dir = get_hermes_home() / "cache" / "terminal"
            except Exception:
                cache_dir = Path(tempfile.gettempdir()) / "hermes_terminal"
            cache_dir.mkdir(parents=True, exist_ok=True)
            # Force forward slashes so the same string serves both contexts.
            return str(cache_dir).replace("\\", "/")

        for env_var in ("TMPDIR", "TMP", "TEMP"):
            candidate = self.env.get(env_var) or os.environ.get(env_var)
            if candidate and candidate.startswith("/"):
                return candidate.rstrip("/") or "/"

        if os.path.isdir("/tmp") and os.access("/tmp", os.W_OK | os.X_OK):
            return "/tmp"

        candidate = tempfile.gettempdir()
        if candidate.startswith("/"):
            return candidate.rstrip("/") or "/"

        return "/tmp"

    def _run_bash(self, cmd_string: str, *, login: bool = False,
                  timeout: int = 120,
                  stdin_data: str | None = None) -> subprocess.Popen:
        bash = _find_bash()
        # For login-shell invocations (used by init_session to build the
        # environment snapshot), prepend sources for the user's bashrc /
        # custom init files so tools registered outside bash_profile
        # (nvm, asdf, pyenv, …) end up on PATH in the captured snapshot.
        # Non-login invocations are already sourcing the snapshot and
        # don't need this.
        if login:
            init_files = _resolve_shell_init_files()
            if init_files:
                cmd_string = _prepend_shell_init(cmd_string, init_files)
        args = [bash, "-l", "-c", cmd_string] if login else [bash, "-c", cmd_string]
        run_env = _make_run_env(self.env)

        # Recover when the cwd has been deleted out from under us — usually by
        # a previous tool call that ran ``rm -rf`` on its own working dir
        # (issue #17558).  Popen would otherwise raise FileNotFoundError on
        # the cwd before bash starts, wedging every subsequent call until the
        # gateway restarts.
        #
        # On Windows, ``_resolve_safe_cwd`` also normalises Git Bash-style
        # POSIX paths (``/c/Users/...``) to native form so a perfectly valid
        # ``pwd -P`` result from bash isn't mistakenly treated as "missing"
        # and spammed as a warning on every command.
        safe_cwd = _resolve_safe_cwd(self.cwd)
        if safe_cwd != self.cwd:
            # MSYS → Windows translation alone shouldn't surface as a warning
            # (it's a benign normalization, not a recovery). Only warn when
            # the directory really doesn't exist on disk.
            normalized = _msys_to_windows_path(self.cwd) if _IS_WINDOWS else self.cwd
            if safe_cwd != normalized:
                logger.warning(
                    "LocalEnvironment cwd %r is missing on disk; "
                    "falling back to %r so terminal commands keep working.",
                    self.cwd,
                    safe_cwd,
                )
            self.cwd = safe_cwd

        _popen_cwd = self.cwd

        _popen_kwargs = {"creationflags": windows_hide_flags()} if _IS_WINDOWS else {}

        proc = subprocess.Popen(
            args,
            text=True,
            env=run_env,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
            start_new_session=True,
            cwd=_popen_cwd,
            **_popen_kwargs,
        )
        if not _IS_WINDOWS:
            try:
                proc._hermes_pgid = os.getpgid(proc.pid)
            except ProcessLookupError:
                pass

        if stdin_data is not None:
            _pipe_stdin(proc, stdin_data)

        return proc

    def _kill_process(self, proc):
        """Kill the entire process group (all children)."""

        def _group_alive(pgid: int) -> bool:
            try:
                # POSIX-only: _IS_WINDOWS is handled before this helper is used.
                os.killpg(pgid, 0)  # windows-footgun: ok — POSIX process-group alive probe
                return True
            except ProcessLookupError:
                return False
            except PermissionError:
                # The group exists, even if this process cannot signal it.
                return True

        def _wait_for_group_exit(pgid: int, timeout: float) -> bool:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                # Reap the wrapper promptly. A dead but unreaped group leader
                # still makes killpg(pgid, 0) report the group as alive.
                try:
                    proc.poll()
                except Exception:
                    pass
                if not _group_alive(pgid):
                    return True
                time.sleep(0.05)
            try:
                proc.poll()
            except Exception:
                pass
            return not _group_alive(pgid)

        try:
            if _IS_WINDOWS:
                try:
                    from gateway.status import terminate_pid

                    terminate_pid(proc.pid, force=True)
                except Exception:
                    proc.kill()
                try:
                    proc.wait(timeout=2.0)
                except (subprocess.TimeoutExpired, OSError):
                    pass
            else:
                try:
                    pgid = os.getpgid(proc.pid)
                except ProcessLookupError:
                    pgid = getattr(proc, "_hermes_pgid", None)
                    if pgid is None:
                        raise

                try:
                    os.killpg(pgid, signal.SIGTERM)  # windows-footgun: ok — POSIX process-group SIGTERM (guarded by _IS_WINDOWS above)
                except ProcessLookupError:
                    return

                # Wait on the process group, not just the shell wrapper. Under
                # load the wrapper can exit before grandchildren do; returning
                # at that point leaves orphaned process-group members behind.
                if _wait_for_group_exit(pgid, 1.0):
                    return

                try:
                    # POSIX-only: _IS_WINDOWS is handled by the outer branch.
                    os.killpg(pgid, signal.SIGKILL)  # windows-footgun: ok — POSIX process-group SIGKILL
                except ProcessLookupError:
                    return
                _wait_for_group_exit(pgid, 2.0)
                try:
                    proc.wait(timeout=0.2)
                except (subprocess.TimeoutExpired, OSError):
                    pass
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except Exception:
                pass

    def _update_cwd(self, result: dict):
        """Read CWD from temp file (local-only, no round-trip needed).

        Skip the assignment when the path no longer exists as a directory —
        ``pwd -P`` on a deleted cwd can leave a stale value in the marker
        file, and propagating it would re-wedge the next ``Popen``.  The
        ``_run_bash`` recovery path will resolve a safe fallback if needed.

        On Windows, the value written by Git Bash's ``pwd -P`` is in
        MSYS form (``/c/Users/x``). Translate it to native Windows form
        before validating with ``os.path.isdir`` and before storing on
        ``self.cwd``; otherwise the isdir check rejects every valid
        result and ``_run_bash`` later prints a misleading "cwd is
        missing" warning on every command.
        """
        try:
            with open(self._cwd_file, encoding="utf-8") as f:
                cwd_path = f.read().strip()
            if _IS_WINDOWS:
                cwd_path = _msys_to_windows_path(cwd_path)
            if cwd_path and os.path.isdir(cwd_path):
                self.cwd = cwd_path
        except (OSError, FileNotFoundError):
            pass

        # Still strip the marker from output so it's not visible
        self._extract_cwd_from_output(result)

    def _extract_cwd_from_output(self, result: dict):
        """Same semantics as the base class, but on Windows the value
        emitted by ``pwd -P`` inside Git Bash is in MSYS form
        (``/c/Users/x``). Normalize to native Windows form and validate
        the directory exists before assigning to ``self.cwd`` — otherwise
        ``_run_bash``'s safe-cwd recovery would warn on every subsequent
        command.

        Always defers to the base class for stripping the marker text from
        ``result["output"]`` so output formatting is identical.
        """
        # Snapshot pre-existing cwd, defer to base for parsing + marker
        # stripping, then validate / normalize whatever it assigned.
        prev_cwd = self.cwd
        super()._extract_cwd_from_output(result)
        if self.cwd != prev_cwd:
            normalized = _msys_to_windows_path(self.cwd) if _IS_WINDOWS else self.cwd
            if normalized and os.path.isdir(normalized):
                self.cwd = normalized
            else:
                # Stale / non-existent path — keep previous cwd; _run_bash
                # will resolve a safe fallback on the next call if needed.
                self.cwd = prev_cwd

    def cleanup(self):
        """Clean up temp files."""
        for f in (self._snapshot_path, self._cwd_file):
            try:
                os.unlink(f)
            except OSError:
                pass
        # Remove any orphaned atomic-write temp snapshots (snap.tmp.<bashpid>)
        # a failed/interrupted mv could have left behind (#38249).
        try:
            import glob
            for tmp in glob.glob(f"{self._snapshot_path}.tmp.*"):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        except Exception:
            pass
