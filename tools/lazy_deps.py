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

* **Venv-scoped by default.** Installs target ``sys.executable`` in the
  active venv. We never touch the system Python.
* **Durable-target mode (immutable images).** When the deployment seals the
  agent's own venv (the Docker image sets ``HERMES_DISABLE_LAZY_INSTALLS=1``
  and makes ``/opt/hermes`` read-only), setting
  ``HERMES_LAZY_INSTALL_TARGET`` redirects lazy installs to a writable
  directory on the durable data volume (e.g. ``/opt/data/lazy-packages``).
  That directory is **appended to the end of ``sys.path``** — never
  prepended, never exported via ``PYTHONPATH`` — so the agent's own
  site-packages wins every name collision. A package installed this way can
  only ADD new importable modules; it can never shadow, downgrade, or break
  a module the core already ships. The worst a bad/incompatible backend
  package can do is fail to import and report itself unavailable — the agent
  core stays healthy. This is the structural guarantee that a lazily
  installed package cannot brick Hermes, which is what made it safe to seal
  the venv in the first place. Compiled-wheel safety across image rebuilds
  is handled by an ABI/Python-version stamp on the target subdir (see
  :func:`_ensure_target_ready`).
* **PyPI by package name only.** Specs may be ``"package>=1.0,<2"`` etc.
  We do NOT support ``--index-url`` overrides, ``git+https://``, file:
  paths, or any other input that could be hijacked by a malicious config.
* **Allowlist.** Only specs that appear in :data:`LAZY_DEPS` can be
  installed via this path. A typo in feature name doesn't get the user
  install-anything semantics.
* **Opt-out.** Setting ``security.allow_lazy_installs: false`` in
  ``config.yaml`` disables runtime installs in BOTH modes. Users in
  restricted networks or strict security postures can pin themselves to
  whatever was installed at setup time.
* **Offline detection.** If the install fails (offline, mirror down,
  PyPI 404 / quarantine), we surface the failure as
  :class:`FeatureUnavailable` with the actual pip stderr — no silent
  retries, no caching of bad state.

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
import site
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
    "provider.anthropic": ("anthropic==0.87.0",),  # CVE-2026-34450, CVE-2026-34452
    # AWS Bedrock provider
    "provider.bedrock": ("boto3==1.42.89",),
    # Microsoft Foundry — Entra ID auth (managed identity, workload identity,
    # service principal, az login, VS Code, azd, PowerShell). Only loaded
    # when model.auth_mode=entra_id is selected; key-based azure-foundry
    # users never pay this import.
    "provider.azure_identity": ("azure-identity==1.25.3",),

    # ─── Web search backends ───────────────────────────────────────────────
    "search.exa": ("exa-py==2.10.2",),
    "search.firecrawl": ("firecrawl-py==4.17.0",),
    "search.parallel": ("parallel-web==0.4.2",),

    # ─── TTS providers ─────────────────────────────────────────────────────
    # Pinned to exact versions to match pyproject.toml's no-ranges policy
    # (see comment at top of [project.dependencies]). When bumping, update
    # both this map AND the corresponding extra in pyproject.toml.
    #
    # mistralai pin tracks the `mistral` extra in pyproject.toml. PyPI
    # quarantined the project 2026-05-12 (malicious 2.4.6, Mini Shai-Hulud);
    # 2.4.6 was removed and clean releases resumed (2.4.7, 2.4.8). Voxtral
    # STT + TTS share the same SDK.
    "tts.mistral": ("mistralai==2.4.8",),
    "tts.edge": ("edge-tts==7.2.7",),
    "tts.elevenlabs": ("elevenlabs==1.59.0",),

    # ─── Speech-to-text providers ──────────────────────────────────────────
    "stt.mistral": ("mistralai==2.4.8",),
    "stt.faster_whisper": (
        "faster-whisper==1.2.1",
        "sounddevice==0.5.5",
        "numpy==2.4.3",
    ),

    # ─── Image generation backends ─────────────────────────────────────────
    "image.fal": ("fal-client==0.13.1",),

    # ─── Memory providers ──────────────────────────────────────────────────
    "memory.honcho": ("honcho-ai==2.1.2",),
    "memory.hindsight": ("hindsight-client==0.6.1",),
    # supermemory + mem0 are opt-in cloud memory providers with their own
    # SDKs. On the published Docker image the agent venv is sealed
    # (HERMES_DISABLE_LAZY_INSTALLS=1) and lazy installs are redirected to the
    # durable target — so, like honcho/hindsight, these MUST go through
    # ensure() to be installable there. Without an allowlist entry + an
    # ensure() call at the import site, the SDK never installs on a hosted
    # instance and the provider silently reports itself unavailable.
    "memory.supermemory": ("supermemory==3.50.0",),
    "memory.mem0": ("mem0ai==2.0.10",),

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
        "aiohttp==3.13.4",  # CVE-2026-34513/34518/34519/34520/34525
    ),
    "platform.matrix": (
        "mautrix[encryption]==0.21.0",
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
    # WeCom callback-mode adapter — parses untrusted XML POST bodies. Pulls
    # defusedxml only; aiohttp/httpx are core dependencies of every messaging
    # adapter and ship via `platform.discord` / `platform.slack` / etc.
    "platform.wecom_callback": ("defusedxml==0.7.1",),
    # Microsoft Teams adapter — microsoft-teams-apps pulls a heavy tree
    # (microsoft-teams-api/cards/common, dependency-injector, msal). Lazy-
    # installed on demand like every other messaging platform; also exposed
    # as the `teams` extra in pyproject for packagers / explicit installs.
    "platform.teams": ("microsoft-teams-apps==2.0.13.4", "aiohttp==3.13.4"),

    # ─── Terminal backends ─────────────────────────────────────────────────
    "terminal.modal": ("modal==1.3.4",),
    "terminal.daytona": ("daytona==0.155.0",),

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
        "starlette==1.0.1",  # CVE-2026-48710 (BadHost) — keep lazy-install in sync with pyproject [web]
        "python-multipart==0.0.27",  # FastAPI UploadFile/Form for streaming uploads (NS-501)
    ),
    # Vision image-resize recovery (Pillow). Pillow is now a CORE dependency
    # (pyproject `dependencies`), so this entry is a belt-and-suspenders fallback
    # for stripped/source-build installs that somehow dropped it. The vision
    # call site uses prompt=False so it can never raise a blocking input()
    # prompt mid-session (#40490).
    "tool.vision": ("Pillow==12.2.0",),
    # Computer Use (cua-driver) — the MCP client SDK used to spawn and talk
    # to the cua-driver process over stdio. Matches the `mcp` / `computer-use`
    # extras in pyproject.toml. The one-liner installer pulls this in via
    # `[all]`; lazy-installing here covers lean / partial / broken-extra
    # installs so computer_use never dead-ends on `No module named 'mcp'`.
    "tool.computer_use": (
        "mcp==1.26.0",
        "starlette==1.0.1",  # CVE-2026-48710 — keep in sync with pyproject [computer-use]
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
# Internals
# =============================================================================


# Environment variable that redirects lazy installs away from the (sealed)
# agent venv and into a writable directory on a durable volume. Set by the
# Docker image to /opt/data/lazy-packages. This is an internal bridge var,
# not user-facing config: the user-facing knob remains
# security.allow_lazy_installs in config.yaml. When unset, lazy installs go
# into the active venv as before.
_LAZY_TARGET_ENV = "HERMES_LAZY_INSTALL_TARGET"

# Name of the stamp file written into the target dir recording the Python
# X.Y + ABI it was populated for. If a container rebuild bumps the
# interpreter, compiled wheels (.so) in the durable store would be ABI-
# incompatible; we detect the mismatch and wipe the store so packages get
# re-resolved against the new interpreter rather than importing a stale .so.
_TARGET_STAMP_NAME = ".python-abi"


def _python_abi_tag() -> str:
    """A stable token identifying the running interpreter's ABI.

    Combines the X.Y version with the EXT_SUFFIX (which encodes the ABI
    tag and platform, e.g. ``cpython-313-x86_64-linux-gnu``). Two
    interpreters that can share compiled wheels produce the same token.
    """
    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    ext = sysconfig.get_config_var("EXT_SUFFIX") or ""
    return f"{ver}:{ext}"


def _lazy_install_target() -> Optional[Path]:
    """Return the durable install-target dir, or None for venv-scoped mode.

    Returns a path only when :data:`_LAZY_TARGET_ENV` is set to a non-empty
    value. The directory is created on demand by :func:`_ensure_target_ready`.
    """
    raw = os.environ.get(_LAZY_TARGET_ENV, "").strip()
    if not raw:
        return None
    return Path(raw)


def _ensure_target_ready(target: Path) -> Optional[str]:
    """Create the target dir and validate its ABI stamp.

    If the stamp is missing it is written. If it is present but records a
    different interpreter ABI than the one now running (e.g. the container
    image was rebuilt onto a newer Python), the directory's contents are
    wiped and the stamp rewritten, so stale compiled wheels can't be
    imported against an incompatible interpreter.

    Returns ``None`` on success, or an error string if the directory can't
    be created / written (e.g. read-only mount, permission error).
    """
    want = _python_abi_tag()
    stamp = target / _TARGET_STAMP_NAME
    try:
        if target.exists():
            have = ""
            try:
                have = stamp.read_text(encoding="utf-8").strip()
            except (OSError, FileNotFoundError):
                have = ""
            if have and have != want:
                logger.info(
                    "Lazy install target %s was built for ABI %r but running "
                    "ABI is %r; wiping stale packages.",
                    target, have, want,
                )
                for child in target.iterdir():
                    if child.is_dir() and not child.is_symlink():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        try:
                            child.unlink()
                        except OSError:
                            pass
        target.mkdir(parents=True, exist_ok=True)
        stamp.write_text(want, encoding="utf-8")
    except OSError as e:
        return f"lazy install target {target} is not writable: {e}"
    return None


def _activate_target_on_syspath(target: Path) -> None:
    """Append the durable target to ``sys.path`` so its packages import.

    Appended to the END (never prepended) so the agent's own venv
    site-packages takes precedence on every name collision. Idempotent.
    Uses :func:`site.addsitedir` so ``.pth`` files (namespace packages,
    editable installs) inside the target are honoured, then enforces the
    append ordering — ``addsitedir`` would otherwise insert near the front.
    """
    target_str = str(target)
    # Snapshot existing entries so we can restore precedence afterwards.
    before = list(sys.path)
    if target_str not in before:
        site.addsitedir(target_str)
    # site.addsitedir may have inserted target (and any .pth-added dirs) at
    # the front. Move every newly-added entry to the end, preserving the
    # core venv's precedence. New entries are those not present `before`.
    new_entries = [p for p in sys.path if p not in before]
    if new_entries:
        sys.path[:] = [p for p in sys.path if p not in new_entries] + new_entries
    # importlib.metadata caches the path-based distribution finder; clear it
    # so a just-activated dir is visible to version() checks this process.
    try:
        import importlib
        importlib.invalidate_caches()
    except Exception:
        pass


def activate_durable_lazy_target() -> None:
    """Public: wire the durable lazy-install target onto ``sys.path``.

    Safe no-op when :data:`_LAZY_TARGET_ENV` is unset or the directory does
    not yet exist. Called once early in process startup (before backends
    import) so packages installed into the durable store on a previous run
    are importable on this run. Never raises.
    """
    target = _lazy_install_target()
    if target is None:
        return
    try:
        if target.exists():
            _activate_target_on_syspath(target)
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("Failed to activate durable lazy target %s: %s", target, e)


def _allow_lazy_installs() -> bool:
    """Return whether lazy installs are permitted in this environment.

    Resolution order:

    1. ``security.allow_lazy_installs: false`` in config.yaml is an absolute
       opt-out — it disables installs in BOTH venv-scoped and durable-target
       modes. This is the user-facing kill switch.
    2. ``HERMES_DISABLE_LAZY_INSTALLS=1`` seals the *agent venv* (set by the
       immutable Docker image). It blocks venv-scoped installs — UNLESS a
       durable install target is configured, in which case installs are
       redirected there (a path that structurally cannot break the sealed
       venv) and are therefore allowed.

    Defaults to True. If config is unreadable we fail open (allow), because
    refusing to install would lock people out of their own backends; the
    decision to block is an explicit user opt-in.
    """
    # (1) Config kill switch wins in every mode.
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
    except Exception:
        cfg = None
    if cfg is not None:
        sec = cfg.get("security") or {}
        if not bool(sec.get("allow_lazy_installs", True)):
            return False

    # (2) Sealed-venv env var: blocks ONLY when there is no safe durable
    # target to redirect into. With a target set, the install goes to the
    # data volume (append-only on sys.path), so the seal is preserved.
    if os.environ.get("HERMES_DISABLE_LAZY_INSTALLS") == "1":
        return _lazy_install_target() is not None

    return True


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
    """Is ``spec`` already satisfied in the current env?

    Checks both presence AND version. If the package is installed at a
    version outside the spec's range, returns False so the caller will
    upgrade/downgrade to the pinned version. This is what makes
    ``hermes update`` propagate pin bumps in :data:`LAZY_DEPS` to already-
    installed backends instead of silently leaving stale versions in place.

    If ``packaging`` is unavailable for any reason (it's a transitive of
    pip so this should never happen), we fall back to a presence-only check
    so we err on the side of "don't churn".
    """
    pkg = _pkg_name_from_spec(spec)
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:
        return False
    try:
        installed = version(pkg)
    except PackageNotFoundError:
        return False
    except Exception:
        return False

    spec_tail = _specifier_from_spec(spec)
    if not spec_tail:
        # Bare ``"package"`` — no version constraint, presence is enough.
        return True

    try:
        from packaging.specifiers import InvalidSpecifier, SpecifierSet
        from packaging.version import InvalidVersion, Version
    except ImportError:
        # packaging unavailable — fall back to "installed counts as satisfied".
        return True

    try:
        return Version(installed) in SpecifierSet(spec_tail)
    except (InvalidSpecifier, InvalidVersion, Exception):
        # Malformed spec or installed version we can't parse — don't churn.
        return True


def _is_present(spec: str) -> bool:
    """Cheap presence-only check (package name installed at any version).

    Used by :func:`active_features` to detect backends the user has
    previously activated, regardless of whether the version pin moved.
    """
    pkg = _pkg_name_from_spec(spec)
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:
        return False
    try:
        version(pkg)
        return True
    except PackageNotFoundError:
        return False
    except Exception:
        return False


def _core_constraints_file() -> Optional[Path]:
    """Write a pip constraints file pinning every package already importable
    in the core environment to its installed version.

    Passed as ``--constraint`` for durable-target installs so the resolver
    pins shared transitive deps (httpx, pydantic, aiohttp, …) to the exact
    versions the core venv already ships, instead of pulling newer copies
    into the durable store. Two payoffs:

    * The durable store stays minimal — only genuinely-new packages land
      there; shared deps resolve to "already satisfied" against core.
    * A backend that *requires* a version conflicting with core fails loudly
      at install time (resolver conflict) rather than silently installing a
      shadowed copy that can never win on sys.path anyway.

    Returns the path to a temp constraints file, or None if enumeration
    failed (in which case the caller installs without constraints — still
    safe, just less tidy).
    """
    try:
        from importlib.metadata import distributions
    except ImportError:
        return None
    try:
        import tempfile
        lines = []
        seen = set()
        for dist in distributions():
            name = dist.metadata["Name"] if dist.metadata else None
            ver = dist.version
            if not name or not ver:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"{name}=={ver}")
        if not lines:
            return None
        fd, path = tempfile.mkstemp(prefix="hermes-core-constraints-", suffix=".txt")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(lines)) + "\n")
        return Path(path)
    except Exception as e:
        logger.debug("Could not build core constraints file: %s", e)
        return None


def _venv_pip_install(specs: tuple[str, ...], *, timeout: int = 300) -> _InstallResult:
    """Install ``specs`` using the uv → pip → ensurepip ladder.

    Two modes:

    * **Venv-scoped (default).** Installs into the active venv
      (``sys.executable``). Used on normal installs.
    * **Durable-target.** When :data:`_LAZY_TARGET_ENV` is set, installs into
      that directory via ``--target`` and constrains shared deps to the
      core venv's versions (see :func:`_core_constraints_file`). The target
      is append-only on ``sys.path`` so it can never shadow core. Used by
      the immutable Docker image to keep lazy installs off the sealed venv.

    Mirrors the strategy in ``hermes_cli.tools_config._pip_install`` but
    kept independent here so this module has no CLI dependency.
    """
    if not specs:
        return _InstallResult(True, "", "")

    target = _lazy_install_target()
    constraints: Optional[Path] = None

    if target is not None:
        err = _ensure_target_ready(target)
        if err:
            return _InstallResult(False, "", err)
        constraints = _core_constraints_file()

    target_args: list[str] = []
    if target is not None:
        # --target tells both uv and pip to install into an arbitrary dir.
        target_args = ["--target", str(target)]
    constraint_args: list[str] = []
    if constraints is not None:
        constraint_args = ["--constraint", str(constraints)]

    try:
        venv_root = Path(sys.executable).parent.parent
        from tools.environments.local import hermes_subprocess_env
        uv_env = hermes_subprocess_env(inherit_credentials=False)
        uv_env["VIRTUAL_ENV"] = str(venv_root)

        # Tier 1: uv (preferred — fast, doesn't need pip in the venv)
        uv_bin = shutil.which("uv")
        if uv_bin:
            try:
                r = subprocess.run(
                    [uv_bin, "pip", "install", *target_args, *constraint_args, *specs],
                    capture_output=True, text=True, timeout=timeout, env=uv_env,
                    stdin=subprocess.DEVNULL,
                )
                if r.returncode == 0:
                    if target is not None:
                        _activate_target_on_syspath(target)
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
                stdin=subprocess.DEVNULL,
            )
            if probe.returncode != 0:
                raise FileNotFoundError("pip not in venv")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            try:
                subprocess.run(
                    [sys.executable, "-m", "ensurepip", "--upgrade", "--default-pip"],
                    capture_output=True, text=True, timeout=120, check=True,
                    stdin=subprocess.DEVNULL,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                return _InstallResult(False, "",
                                      f"pip not available and ensurepip failed: {e}")

        try:
            r = subprocess.run(
                pip_cmd + ["install", *target_args, *constraint_args, *specs],
                capture_output=True, text=True, timeout=timeout,
                stdin=subprocess.DEVNULL,
            )
            if r.returncode == 0 and target is not None:
                _activate_target_on_syspath(target)
            return _InstallResult(r.returncode == 0, r.stdout or "", r.stderr or "")
        except subprocess.TimeoutExpired as e:
            return _InstallResult(False, "", f"pip install timed out: {e}")
        except Exception as e:
            return _InstallResult(False, "", f"pip install failed: {e}")
    finally:
        if constraints is not None:
            try:
                constraints.unlink()
            except OSError:
                pass


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
        raise FeatureUnavailable(
            feature, (), f"feature {feature!r} not in LAZY_DEPS allowlist"
        )

    missing = feature_missing(feature)
    if not missing:
        return

    # Validate every spec against the allowlist + safety regex. Belt and
    # braces — the keys-in-LAZY_DEPS check above already constrains this.
    for spec in missing:
        if not _spec_is_safe(spec):
            raise FeatureUnavailable(
                feature, missing,
                f"refusing to install unsafe spec {spec!r}"
            )

    if not _allow_lazy_installs():
        raise FeatureUnavailable(
            feature, missing,
            "lazy installs disabled (security.allow_lazy_installs=false)"
        )

    # Only show the interactive confirmation when we own a TTY and
    # prompt_toolkit isn't running.  A bare input() deadlocks when a
    # prompt_toolkit app owns the terminal because keystrokes route to
    # its event loop rather than stdin, so the prompt blocks forever.
    # Under the TUI we skip the prompt and proceed — lazy installs are
    # gated by security.allow_lazy_installs, so reaching here is
    # already user opt-in.
    _pt_active = False
    if "prompt_toolkit.application.current" in sys.modules:
        try:
            from prompt_toolkit.application.current import get_app_or_none
            _app = get_app_or_none()
            _pt_active = _app is not None and getattr(_app, "is_running", False)
        except Exception:
            _pt_active = False

    if prompt and not _pt_active and sys.stdin.isatty() and sys.stdout.isatty():
        spec_list = ", ".join(missing)
        try:
            answer = input(
                f"\nFeature {feature!r} requires: {spec_list}\n"
                f"Install into the active venv now? [Y/n] "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer and answer not in {"y", "yes"}:
            raise FeatureUnavailable(
                feature, missing, "user declined install at prompt"
            )

    logger.info("Lazy-installing %s for feature %r", " ".join(missing), feature)
    result = _venv_pip_install(missing)
    if not result.success:
        # Surface the actual pip error so the user can debug PyPI-side
        # issues (404 quarantine, network down, etc.).
        snippet = (result.stderr or result.stdout or "").strip()
        if snippet:
            # Clip to a readable size — pip can dump pages of resolution traces.
            snippet = snippet[-2000:]
        raise FeatureUnavailable(
            feature, missing,
            f"pip install failed: {snippet or 'no error output'}"
        )

    # Verify post-install. importlib.metadata caches per-process, so if we
    # just installed something the cache may not see it without a refresh.
    try:
        import importlib.metadata as _md
        if hasattr(_md, "_cache_clear"):
            _md._cache_clear()  # type: ignore[attr-defined]
    except Exception:
        pass

    still_missing = feature_missing(feature)
    if still_missing:
        raise FeatureUnavailable(
            feature, still_missing,
            "install reported success but packages still not importable "
            "(may require Python restart)"
        )

    logger.info("Lazy install complete for feature %r", feature)


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
    except ImportError:
        return False

    target_globals.update(bindings)
    return True
