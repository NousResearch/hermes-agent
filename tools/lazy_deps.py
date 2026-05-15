"""
Lazy dependency installer for opt-in Hermes Agent backends.

Many Hermes features (Mistral TTS, ElevenLabs TTS, Honcho memory, Bedrock,
Slack, Matrix, etc.) require Python packages that not every user needs. The
historical approach was to bundle them all under ``pyproject.toml`` extras
(``hermes-agent[all]``) and install them eagerly at setup time. That has
two problems:

1. **Fragility.** When one extra's transitive dependency becomes
   unavailable on PyPI (quarantined for malware, yanked, broken upload),
   the *entire* ``[all]`` resolve fails and fresh installs silently fall
   back to a stripped tier — losing 10+ unrelated extras at once.

2. **Bloat.** A user who only ever talks to one provider pulls hundreds
   of packages they will never import.

The lazy-install pattern fixes both. Backends call :func:`ensure` at the
top of their first-import path. If the deps are missing, ``ensure`` checks
the ``security.allow_lazy_installs`` config flag (default true) and runs
a venv-scoped pip install. If the user has explicitly disabled lazy
installs, ``ensure`` raises :class:`FeatureUnavailable` with a clear
remediation hint pointing at ``hermes tools`` or the manual pip command.

Security model:

* **Venv-scoped only.** Installs target ``sys.executable`` in the active
  venv. We never touch the system Python.
* **PyPI by package name only.** Specs may be ``"package>=1.0,<2"`` etc.
  We do NOT support ``--index-url`` overrides, ``git+https://``, file:
  paths, or any other input that could be hijacked by a malicious config.
* **Allowlist.** Only specs that appear in :data:`LAZY_DEPS` can be
  installed via this path. A typo in feature name doesn't get the user
  install-anything semantics.
* **Opt-out.** Setting ``security.allow_lazy_installs: false`` in
  ``config.yaml`` disables runtime installs. Users in restricted networks
  or strict security postures can pin themselves to whatever was installed
  at setup time.
* **Offline detection.** If the install fails (offline, mirror down,
  PyPI 404 / quarantine), we surface the failure as
  :class:`FeatureUnavailable` with the actual pip stderr — no silent
  retries, no caching of bad state.

Nix support:

Hermes Agent's Nix build produces a sealed, read-only Python environment
in ``/nix/store``. Standard venv-scoped pip installs fail because the
store path is immutable. When running from a Nix store path, the
installer transparently falls back to a user-local prefix at
``~/.hermes/lazy-deps/`` (or ``$HERMES_HOME/lazy-deps/``) and injects
that prefix into ``sys.path`` so the newly-installed packages are
importable in the current process.

Adding a new backend:

1. Add an entry to :data:`LAZY_DEPS` with the package specs.
2. At the top of the backend module's import path, call
   ``ensure("feature.name")`` inside a try/except that converts
   :class:`FeatureUnavailable` to a useful runtime error.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Allowlist of lazy-installable backends.
#
# Keys are dot-separated feature names ("namespace.backend"). Values are
# tuples of pip-installable specs that match the corresponding extra in
# pyproject.toml. The framework enforces that only specs from this map
# can flow into the pip install command.
# =============================================================================


LAZY_DEPS: dict[str, tuple[str, ...]] = {
    # ─── Inference providers ───────────────────────────────────────────────
    # Native Anthropic SDK — needed when provider=anthropic (not via
    # OpenRouter / aggregators which use the openai SDK).
    "provider.anthropic": ("anthropic==0.86.0",),
    # AWS Bedrock provider
    "provider.bedrock": ("boto3==1.42.89",),

    # ─── Web search backends ───────────────────────────────────────────────
    "search.exa": ("exa-py==2.10.2",),
    "search.firecrawl": ("firecrawl-py==4.17.0",),
    "search.parallel": ("parallel-web==0.4.2",),

    # ─── TTS providers ─────────────────────────────────────────────────────
    # Pinned to exact versions to match pyproject.toml's no-ranges policy
    # (see comment at top of [project.dependencies]). When bumping, update
    # both this map AND the corresponding extra in pyproject.toml.
    #
    # NOTE: tts.mistral / stt.mistral entries are intentionally absent —
    # the `mistralai` PyPI project is quarantined as of 2026-05-12 (Mini
    # Shai-Hulud worm). Re-add when PyPI restores a clean release; see
    # comment in pyproject.toml above the (removed) `mistral` extra for
    # the full restoration checklist.
    "tts.edge": ("edge-tts==7.2.7",),
    "tts.elevenlabs": ("elevenlabs==1.59.0",),

    # ─── Speech-to-text providers ──────────────────────────────────────────
    "stt.faster_whisper": (
        "faster-whisper==1.2.1",
        "sounddevice==0.5.5",
        "numpy==2.4.3",
    ),

    # ─── Image generation backends ─────────────────────────────────────────
    "image.fal": ("fal-client==0.13.1",),

    # ─── Memory providers ──────────────────────────────────────────────────
    "memory.honcho": ("honcho-ai==2.0.1",),
    "memory.hindsight": ("hindsight-client==0.6.1",),

    # ─── Messaging platforms (lazy-installable on demand) ──────────────────
    "platform.telegram": ("python-telegram-bot[webhooks]==22.6",),
    # brotlicffi gives aiohttp a working 2-arg Decompressor.process() for
    # Discord CDN's Brotli-encoded attachments. Without it, aiohttp falls
    # back to google's `Brotli` package (1-arg API), and any .txt/.md/.doc
    # uploaded to the Discord gateway fails to decode at att.read() with
    # "Can not decode content-encoding: br" — see #12511 / #15744.
    "platform.discord": ("discord.py[voice]==2.7.1", "brotlicffi==1.2.0.1"),
    "platform.slack": (
        "slack-bolt==1.27.0",
        "slack-sdk==3.40.1",
        "aiohttp==3.13.3",
    ),
    "platform.matrix": (
        "mautrix[encryption]==0.21.0",
        "Markdown==3.10.2",
        "aiosqlite==0.22.1",
        "asyncpg==0.31.0",
        "aiohttp-socks==0.11.0",
    ),
    "platform.dingtalk": (
        "dingtalk-stream==0.24.3",
        "alibabacloud-dingtalk==2.2.42",
        "qrcode==7.4.2",
    ),
    "platform.feishu": (
        "lark-oapi==1.5.3",
        "qrcode==7.4.2",
    ),

    # ─── Terminal backends ─────────────────────────────────────────────────
    "terminal.modal": ("modal==1.3.4",),
    "terminal.daytona": ("daytona==0.155.0",),
    "terminal.vercel": ("vercel==0.5.7",),

    # ─── Skills ────────────────────────────────────────────────────────────
    "skill.google_workspace": (
        "google-api-python-client==2.194.0",
        "google-auth-oauthlib==1.3.1",
        "google-auth-httplib2==0.3.1",
    ),
    "skill.youtube": ("youtube-transcript-api==1.2.4",),

    # ─── Tools ─────────────────────────────────────────────────────────────
    # ACP adapter (VS Code / Zed / JetBrains integration)
    "tool.acp": ("agent-client-protocol==0.9.0",),
    # Dashboard (`hermes dashboard`)
    "tool.dashboard": (
        "fastapi==0.133.1",
        "uvicorn[standard]==0.41.0",
    ),
}


# Conservative regex for spec validation — package name plus optional
# version range. Reject anything that looks like a URL, file path, or shell
# metacharacter.
_SAFE_SPEC = re.compile(
    r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*"        # package name
    r"(?:\[[A-Za-z0-9_,\-]+\])?"            # optional [extras]
    r"(?:[<>=!~]=?[A-Za-z0-9_.\-+,*<>=!~]+)?"  # optional version specifier
    r"$"
)


class FeatureUnavailable(RuntimeError):
    """A lazily-installable feature is missing and cannot be made available.

    Either the deps were never installed and the user has disabled lazy
    installs, or the install attempt failed.
    """

    def __init__(self, feature: str, missing: tuple[str, ...], reason: str):
        self.feature = feature
        self.missing = missing
        self.reason = reason
        super().__init__(self._format())

    def _format(self) -> str:
        spec_list = " ".join(repr(s) for s in self.missing)
        return (
            f"Feature {self.feature!r} unavailable: {self.reason}. "
            f"To enable manually: uv pip install {spec_list}  "
            f"(or: pip install {spec_list})."
        )


@dataclass(frozen=True)
class _InstallResult:
    success: bool
    stdout: str
    stderr: str


# =============================================================================
# Nix / immutable-venv detection
# =============================================================================


def _is_nix_venv() -> bool:
    """Return True if the active Python interpreter lives in a Nix store path.

    Nix builds produce sealed, read-only environments under ``/nix/store/``.
    pip/uv cannot write into them, so lazy installs must target a user-local
    prefix instead.
    """
    exe = Path(sys.executable).resolve()
    return str(exe).startswith("/nix/store/")


def _user_local_prefix() -> Path:
    """Return the user-local prefix for lazy deps on immutable systems.

    Uses ``$HERMES_HOME/lazy-deps`` when set, otherwise ``~/.hermes/lazy-deps``.
    """
    home = os.environ.get("HERMES_HOME")
    if home:
        return Path(home) / "lazy-deps"
    return Path.home() / ".hermes" / "lazy-deps"


def _nix_site_dir() -> Path | None:
    """Return the site-packages directory inside the user-local prefix.

    Uses ``sysconfig.get_path('purelib', vars={'base': prefix})`` so the
    path matches pip's ``--prefix`` layout for the current interpreter
    version.
    """
    prefix = _user_local_prefix()
    try:
        site = Path(
            sysconfig.get_path("purelib", vars={"base": str(prefix)})
        )
    except KeyError:
        # Fallback for Python builds where sysconfig doesn't expose purelib.
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        site = prefix / "lib" / f"python{version}" / "site-packages"
    return site


def _ensure_nix_site_path() -> None:
    """Inject the user-local site-packages into ``sys.path`` if not present."""
    site = _nix_site_dir()
    if site is None:
        return
    site_str = str(site)
    if site_str not in sys.path:
        sys.path.insert(0, site_str)
        logger.debug("Added Nix lazy-deps site-packages to sys.path: %s", site_str)


# =============================================================================
# Internals
# =============================================================================


def _allow_lazy_installs() -> bool:
    """Return the ``security.allow_lazy_installs`` config flag.

    Defaults to True. If config is unreadable we fail open (allow), because
    refusing to install would lock people out of their own backends; the
    decision to block is an explicit user opt-in.
    """
    if os.environ.get("HERMES_DISABLE_LAZY_INSTALLS") == "1":
        return False
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
    except Exception:
        return True
    sec = cfg.get("security") or {}
    val = sec.get("allow_lazy_installs", True)
    return bool(val)


def _spec_is_safe(spec: str) -> bool:
    """Reject pip specs that contain URLs, paths, or shell metacharacters."""
    if not spec or len(spec) > 200:
        return False
    if any(ch in spec for ch in (";", "|", "&", "`", "$", "\n", "\r", "\t", "\\")):
        return False
    if spec.startswith(("-", "/", ".")) or "://" in spec or "@" in spec:
        return False
    return bool(_SAFE_SPEC.match(spec))


def _pkg_name_from_spec(spec: str) -> str:
    """Extract the bare package name from a pip spec.

    ``"slack-bolt>=1.18.0,<2"`` → ``"slack-bolt"``
    ``"mautrix[encryption]>=0.20"`` → ``"mautrix"``
    """
    m = re.match(r"^([A-Za-z0-9_][A-Za-z0-9_.\-]*)", spec)
    return m.group(1) if m else spec


def _specifier_from_spec(spec: str) -> str:
    """Extract just the version-specifier portion of a pip spec.

    ``"honcho-ai==2.0.1"`` → ``"==2.0.1"``
    ``"mautrix[encryption]>=0.20,<1"`` → ``">=0.20,<1"``
    ``"package"`` → ``""`` (no version constraint)
    """
    # Strip the package name + optional [extras] block.
    m = re.match(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*(?:\[[A-Za-z0-9_,\-]+\])?", spec)
    if not m:
        return ""
    return spec[m.end():]


def _is_satisfied(spec: str) -> bool:
    """Check whether a spec is already satisfied in the current env.

    If the spec has a version constraint, we try to import the package and
    compare its ``__version__`` (or ``VERSION``) attribute. If the package
    has no version attribute we treat it as satisfied (present but
    unversioned).  If the spec has no constraint, importability alone is
    sufficient.
    """
    pkg = _pkg_name_from_spec(spec)
    specifier = _specifier_from_spec(spec)

    try:
        mod = __import__(pkg.replace("-", "_"))
    except Exception:
        return False

    if not specifier:
        return True

    ver = getattr(mod, "__version__", getattr(mod, "VERSION", None))
    if ver is None:
        return True

    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    try:
        return Version(ver) in SpecifierSet(specifier)
    except Exception:
        return True


def _is_present(spec: str) -> bool:
    """Import-only check — does the top-level module exist regardless of version."""
    pkg = _pkg_name_from_spec(spec)
    try:
        __import__(pkg.replace("-", "_"))
        return True
    except Exception:
        return False


def _venv_pip_install(specs: tuple[str, ...], *, timeout: int = 300) -> _InstallResult:
    """Install ``specs`` into the active venv using uv → pip → ensurepip ladder.

    Mirrors the strategy in ``hermes_cli.tools_config._pip_install`` but
    kept independent here so this module has no CLI dependency.
    """
    if not specs:
        return _InstallResult(True, "", "")

    # ── Nix / immutable-venv fallback ──────────────────────────────────────
    # Nix store paths are read-only. Install into a user-local prefix and
    # let the caller inject that prefix into sys.path.
    if _is_nix_venv():
        prefix = _user_local_prefix()
        prefix.mkdir(parents=True, exist_ok=True)
        site = _nix_site_dir()
        if site:
            site.mkdir(parents=True, exist_ok=True)

        # Tier 1: uv with --prefix
        uv_bin = shutil.which("uv")
        if uv_bin:
            try:
                r = subprocess.run(
                    [uv_bin, "pip", "install", "--prefix", str(prefix), *specs],
                    capture_output=True, text=True, timeout=timeout,
                )
                if r.returncode == 0:
                    return _InstallResult(True, r.stdout or "", r.stderr or "")
                logger.debug("uv pip install --prefix failed: %s", r.stderr)
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                logger.debug("uv --prefix invocation failed: %s", e)

        # Tier 2: pip with --prefix (bootstrap ensurepip if needed)
        pip_cmd = [sys.executable, "-m", "pip"]
        try:
            probe = subprocess.run(
                pip_cmd + ["--version"], capture_output=True, text=True, timeout=15,
            )
            if probe.returncode != 0:
                raise FileNotFoundError("pip not in venv")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            try:
                subprocess.run(
                    [sys.executable, "-m", "ensurepip", "--upgrade", "--default-pip"],
                    capture_output=True, text=True, timeout=120, check=True,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                return _InstallResult(
                    False, "",
                    f"pip not available and ensurepip failed: {e}",
                )

        try:
            r = subprocess.run(
                pip_cmd + ["install", "--prefix", str(prefix), *specs],
                capture_output=True, text=True, timeout=timeout,
            )
            return _InstallResult(r.returncode == 0, r.stdout or "", r.stderr or "")
        except subprocess.TimeoutExpired as e:
            return _InstallResult(False, "", f"pip install --prefix timed out: {e}")
        except Exception as e:
            return _InstallResult(False, "", f"pip install --prefix failed: {e}")

    # ── Standard mutable-venv path ─────────────────────────────────────────
    venv_root = Path(sys.executable).parent.parent
    uv_env = {**os.environ, "VIRTUAL_ENV": str(venv_root)}

    # Tier 1: uv (preferred — fast, doesn't need pip in the venv)
    uv_bin = shutil.which("uv")
    if uv_bin:
        try:
            r = subprocess.run(
                [uv_bin, "pip", "install", *specs],
                capture_output=True, text=True, timeout=timeout, env=uv_env,
            )
            if r.returncode == 0:
                return _InstallResult(True, r.stdout or "", r.stderr or "")
            logger.debug("uv pip install failed: %s", r.stderr)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.debug("uv invocation failed: %s", e)

    # Tier 2: python -m pip (with ensurepip bootstrap if needed)
    pip_cmd = [sys.executable, "-m", "pip"]
    try:
        probe = subprocess.run(
            pip_cmd + ["--version"],
            capture_output=True, text=True, timeout=15,
        )
        if probe.returncode != 0:
            raise FileNotFoundError("pip not in venv")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        try:
            subprocess.run(
                [sys.executable, "-m", "ensurepip", "--upgrade", "--default-pip"],
                capture_output=True, text=True, timeout=120, check=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            return _InstallResult(False, "",
                                  f"pip not available and ensurepip failed: {e}")

    try:
        r = subprocess.run(
            pip_cmd + ["install", *specs],
            capture_output=True, text=True, timeout=timeout,
        )
        return _InstallResult(r.returncode == 0, r.stdout or "", r.stderr or "")
    except subprocess.TimeoutExpired as e:
        return _InstallResult(False, "", f"pip install timed out: {e}")
    except Exception as e:
        return _InstallResult(False, "", f"pip install failed: {e}")


# =============================================================================
# Public API
# =============================================================================


def feature_specs(feature: str) -> tuple[str, ...]:
    """Return the registered specs for a feature, or raise KeyError."""
    if feature not in LAZY_DEPS:
        raise KeyError(f"Unknown lazy feature: {feature!r}")
    return LAZY_DEPS[feature]


def feature_missing(feature: str) -> tuple[str, ...]:
    """Return the subset of specs for ``feature`` not currently installed."""
    return tuple(s for s in feature_specs(feature) if not _is_satisfied(s))


def ensure(feature: str, *, prompt: bool = True) -> None:
    """Make sure all packages for ``feature`` are importable.

    If they're missing, attempts to install them in the active venv. Raises
    :class:`FeatureUnavailable` if the user has disabled lazy installs or
    if the install attempt fails.

    ``prompt``: when True (default) and stdin is a TTY, asks the user to
    confirm before installing. Non-interactive callers (gateway, cron,
    batch) get prompt=False and skip the confirmation — config flag is
    the gate in that case.
    """
    if feature not in LAZY_DEPS:
        raise KeyError(f"Unknown lazy feature: {feature!r}")

    missing = feature_missing(feature)
    if not missing:
        return

    if not _allow_lazy_installs():
        raise FeatureUnavailable(
            feature, missing,
            "lazy installs disabled (set security.allow_lazy_installs: true or unset HERMES_DISABLE_LAZY_INSTALLS)"
        )

    # Interactive confirmation
    if prompt and sys.stdin.isatty():
        specs_str = " ".join(missing)
        print(
            f"\nFeature '{feature}' requires additional packages: {specs_str}\n"
            f"Install now? [Y/n] ", end="", flush=True
        )
        try:
            answer = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer and answer not in ("y", "yes"):
            raise FeatureUnavailable(feature, missing, "user declined install")

    # Security gate: every spec must be in the allowlist regex
    bad = [s for s in missing if not _spec_is_safe(s)]
    if bad:
        raise FeatureUnavailable(
            feature, tuple(bad),
            f"unsafe pip specs rejected by allowlist: {bad}"
        )

    result = _venv_pip_install(missing)
    if not result.success:
        raise FeatureUnavailable(feature, missing, result.stderr.strip() or "unknown error")

    # On Nix, the packages landed in a user-local prefix outside sys.path.
    # Inject that prefix so the caller's subsequent import finds them.
    if _is_nix_venv():
        _ensure_nix_site_path()


def is_available(feature: str) -> bool:
    """Return True if the feature's deps are already satisfied."""
    if feature not in LAZY_DEPS:
        return False
    return not feature_missing(feature)


def feature_install_command(feature: str) -> Optional[str]:
    """Return the ``pip install`` command a user could run manually, or None."""
    if feature not in LAZY_DEPS:
        return None
    specs = LAZY_DEPS[feature]
    return "uv pip install " + " ".join(repr(s) for s in specs)


def active_features() -> list[str]:
    """Return the list of features the user has ever lazy-installed.

    A feature counts as "active" if at least one of its declared packages
    is currently installed in the venv (presence check, ignoring version).
    Features the user has never enabled stay quiet.

    Used by ``hermes update`` to figure out which lazy backends need a
    refresh pass when pins move in :data:`LAZY_DEPS`.
    """
    active = []
    for feature, specs in LAZY_DEPS.items():
        if any(_is_present(s) for s in specs):
            active.append(feature)
    return active


def refresh_active_features(*, prompt: bool = False) -> dict[str, str]:
    """Re-run ``ensure`` for every feature the user has previously activated.

    Returns a ``{feature: status}`` map where status is one of:
        ``"current"``  — pins already satisfied, no install run
        ``"refreshed"`` — pins were stale, reinstall succeeded
        ``"failed: <reason>"`` — install attempt failed; caller decides
                                  whether to surface it (we don't raise)
        ``"skipped: <reason>"`` — gated off (config flag, user decline)

    Intended for ``hermes update``. Never raises; lazy-install failures
    here must not block the rest of the update flow.
    """
    results: dict[str, str] = {}
    for feature in active_features():
        missing = feature_missing(feature)
        if not missing:
            results[feature] = "current"
            continue
        try:
            ensure(feature, prompt=prompt)
            results[feature] = "refreshed"
        except FeatureUnavailable as e:
            # Distinguish "user opted out" from "install failed" so the
            # update command can render the right message.
            if "lazy installs disabled" in str(e) or "declined" in str(e):
                results[feature] = f"skipped: {e.reason}"
            else:
                results[feature] = f"failed: {e.reason}"
        except Exception as e:
            results[feature] = f"failed: {e}"
    return results


def ensure_and_bind(
    feature: str,
    importer: Callable[[], dict[str, Any]],
    target_globals: dict,
    *,
    prompt: bool = False,
) -> bool:
    """Ensure a feature is installed, then rebind names into the caller's globals.

    Combines :func:`ensure` with a post-install import step that rebinds
    module-level names.  This eliminates the error-prone pattern of manually
    listing every global that needs updating after lazy-install.

    ``importer`` is a zero-arg callable that returns a dict of
    ``{name: value}`` for all symbols the caller needs rebound.  It is called
    only after :func:`ensure` succeeds (or if the packages are already
    installed).

    Returns True on success, False if deps couldn't be installed or imported.

    Example usage in a platform adapter::

        def check_slack_requirements() -> bool:
            if SLACK_AVAILABLE:
                return True
            def _import():
                from slack_bolt.async_app import AsyncApp
                from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
                from slack_sdk.web.async_client import AsyncWebClient
                import aiohttp
                return {
                    "AsyncApp": AsyncApp,
                    "AsyncSocketModeHandler": AsyncSocketModeHandler,
                    "AsyncWebClient": AsyncWebClient,
                    "aiohttp": aiohttp,
                    "SLACK_AVAILABLE": True,
                }
            return ensure_and_bind("platform.slack", _import, globals(), prompt=False)
    """
    try:
        ensure(feature, prompt=prompt)
    except (FeatureUnavailable, Exception):
        return False

    try:
        bindings = importer()
    except Exception:
        return False

    target_globals.update(bindings)
    return True
