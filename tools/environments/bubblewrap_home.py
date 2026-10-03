"""HOME policy for the bubblewrap backend: what a sandbox may see under HOME.

The backend hides HOME behind a tmpfs and binds back what a command may
see. This module decides what that is. It holds pure functions only: each
takes explicit inputs, reads no process state and spawns nothing, so the
backend resolves the policy once at construction and the tests need no
sandbox.

The denylist names credential stores. It has two jobs: no allowlist source
may show a denied path, and a denied path below a visible entry (a token
file inside a toolchain directory) gets an overlay of its own.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Credential stores under HOME, relative to it. The file safety policy of
# Hermes (agent.file_safety) is read at call time and joins this set, so a
# path added there is hidden here with no edit to this tuple.
DENIED_HOME_PATHS: tuple[str, ...] = (
    ".ssh",
    ".gnupg",
    ".gpg",
    ".aws",
    ".azure",
    ".boto",
    ".s3cmd",
    ".kube",
    ".docker",
    ".netrc",
    ".npmrc",
    ".pypirc",
    ".pgpass",
    ".env",
    ".git-credentials",
    ".git-credential-cache",
    ".password-store",
    ".pki",
    ".vaults",
    ".cert",
    ".terraform.d",
    ".config/gcloud",
    ".config/gh",
    ".config/hub",
    ".config/glab-cli",
    ".config/op",
    ".config/rclone",
    ".config/keybase",
    ".config/msmtp",
    ".config/1Password",
    ".config/Bitwarden",
    ".config/keepassxc",
    ".config/filezilla",
    ".config/remmina",
    ".config/git/credentials",
    ".config/google-chrome",
    ".config/chromium",
    ".config/BraveSoftware",
    ".config/microsoft-edge",
    ".config/vivaldi",
    ".local/share/keyrings",
    ".local/share/kwalletd",
    ".local/share/plasma-vault",
    ".local/share/pki",
    ".cargo/credentials",
    ".cargo/credentials.toml",
    ".m2/settings.xml",
    ".gradle/gradle.properties",
    ".cache/huggingface/token",
)


def _is_within(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _policy_home_paths(home: str) -> list[str]:
    """Paths under *home* that the file safety policy denies, as the policy spells them."""
    from agent import file_safety

    found: list[str] = []
    try:
        found += sorted(file_safety.build_write_denied_paths(home))
        found += [prefix.rstrip(os.sep) for prefix in file_safety.build_write_denied_prefixes(home)]
    except Exception:
        # The module tuple still applies; a policy that cannot be read must
        # not leave the sandbox without a denylist.
        logger.warning("Could not read the file safety policy for the bubblewrap denylist", exc_info=True)
    real_home = os.path.realpath(home)
    return [p for p in found if p != home and p != real_home and (_is_within(p, home) or _is_within(p, real_home))]


def denied_home_names(home: str) -> tuple[str, ...]:
    """Denied paths under *home* as written, before symlinks are followed.

    An allowlist source is checked against these names and against
    denied_home_paths: a name that is a symlink (``~/.ssh`` linked into a
    dotfiles directory) stays denied under both spellings.
    """
    home = os.path.abspath(os.path.expanduser(home))
    names: list[str] = []
    for path in [os.path.join(home, rel) for rel in DENIED_HOME_PATHS] + _policy_home_paths(home):
        if path not in names:
            names.append(path)
    return tuple(names)


def denied_home_paths(home: str) -> tuple[str, ...]:
    """Real host paths under *home* that must stay hidden in a sandbox.

    Each name goes through os.path.realpath, so the result names the
    directory or file that holds the secret, which is also the only path
    bwrap can mount over. A path that leaves HOME through a symlink is
    dropped here (the backend hides nothing outside HOME through this
    set), and a path under another entry is dropped as covered.
    """
    home = os.path.abspath(os.path.expanduser(home))
    real_home = os.path.realpath(home)
    resolved: list[str] = []
    for name in denied_home_names(home):
        real = os.path.realpath(name)
        if real != real_home and _is_within(real, real_home) and real not in resolved:
            resolved.append(real)
    return tuple(p for p in resolved if not any(p != other and _is_within(p, other) for other in resolved))


# Directories under HOME, relative to it, that are default-deny like HOME
# itself: only an allowed child is visible. They hold one entry per
# application, and no list of credential stores keeps up with that.
DEFAULT_DENY_DIRS: tuple[str, ...] = (".config", ".local", ".local/share")

# What a sandbox sees under HOME with no configuration, relative to HOME:
# shell and git settings a command reads, toolchain directories, and the
# cache. An entry is a top-level dot name or a child of a default-deny
# directory (an allowlist unit, see allow_unit).
ALLOWED_HOME_ENTRIES: tuple[str, ...] = (
    ".bashrc",
    ".bash_profile",
    ".bash_login",
    ".bash_logout",
    ".bash_aliases",
    ".profile",
    ".zshrc",
    ".zshenv",
    ".zprofile",
    ".inputrc",
    ".gitconfig",
    ".editorconfig",
    ".tool-versions",
    ".terminfo",
    ".cache",
    ".cargo",
    ".rustup",
    ".nvm",
    ".bun",
    ".deno",
    ".gem",
    ".npm",
    ".pyenv",
    ".rbenv",
    ".sdkman",
    ".volta",
    ".asdf",
    ".m2",
    ".gradle",
    ".dotnet",
    ".pub-cache",
    ".conda",
    ".nix-profile",
    ".config/git",
    ".config/pip",
    ".config/uv",
    ".config/npm",
    ".config/pnpm",
    ".config/yarn",
    ".config/go",
    ".config/fontconfig",
    ".local/bin",
    ".local/lib",
    ".local/include",
    ".local/share/uv",
    ".local/share/pipx",
    ".local/share/pnpm",
    ".local/share/virtualenvs",
    ".local/share/man",
    ".local/share/bash-completion",
    ".local/share/fonts",
    ".local/share/mime",
)

ALLOW_KEY = "terminal.bubblewrap_home_allow"


def _relative_to_home(home: str, path: str) -> str | None:
    """*path* relative to *home* (as given or resolved), or None when it is not strictly inside."""
    path = os.path.normpath(path)
    for root in (home, os.path.realpath(home)):
        if path != root and _is_within(path, root):
            return os.path.relpath(path, root)
    return None


def allow_unit(home: str, path: str) -> str | None:
    """The smallest allowlist unit that holds *path*, relative to *home*, else None.

    A unit is a top-level dot entry of HOME, or a child of a default-deny
    directory: HOME/.nvm/versions/x/bin gives ``.nvm``,
    HOME/.local/share/pnpm/bin gives ``.local/share/pnpm``. A path outside
    HOME, under a non-dot entry (those are visible already), or naming a
    default-deny directory itself has no unit.
    """
    home = os.path.abspath(home)
    rel = _relative_to_home(home, os.path.abspath(path))
    if rel is None:
        return None
    parts = rel.split(os.sep)
    if not parts[0].startswith(".") or parts[0] in (".", ".."):
        return None
    unit = parts[0]
    rest = parts[1:]
    while unit in DEFAULT_DENY_DIRS:
        if not rest:
            return None
        unit = f"{unit}/{rest[0]}"
        rest = rest[1:]
    return unit


def _operator_unit(home: str, item: str) -> str | None:
    """The unit an allow-key item names, or None when the item is not exactly one unit."""
    text = item.strip()
    if text.startswith("~/"):
        text = text[2:]
    if not text:
        return None
    path = text if os.path.isabs(text) else os.path.join(home, text)
    unit = allow_unit(home, path)
    if unit is None or _relative_to_home(home, os.path.abspath(path)) != unit.replace("/", os.sep):
        return None
    return unit


def resolve_allowlist(
    home: str,
    path_env: str,
    allow_items: tuple[str, ...] | list[str],
    *,
    denied_names: tuple[str, ...],
    denied_paths: tuple[str, ...],
) -> tuple[str, ...]:
    """Allowlist units under *home*, relative to it, sorted.

    Three sources: ALLOWED_HOME_ENTRIES, the unit of each directory in
    *path_env* that lies under HOME (so a toolchain on PATH keeps working
    with no configuration), and *allow_items* from the operator. The
    denylist wins over all three: a unit at or under a denied path is
    dropped. A unit that only contains a denied path stays, and the backend
    hides that path below it.

    Every input is an argument. The backend calls this once at
    construction, so a command cannot widen the set by changing PATH.
    """
    home = os.path.abspath(home)
    denied = tuple(denied_names) + tuple(denied_paths)

    def is_denied(unit: str) -> bool:
        path = os.path.join(home, unit.replace("/", os.sep))
        return any(_is_within(candidate, root) for candidate in (path, os.path.realpath(path)) for root in denied)

    units = {unit for unit in ALLOWED_HOME_ENTRIES if not is_denied(unit)}
    for entry in path_env.split(os.pathsep):
        if not entry or not os.path.isabs(entry):
            continue
        unit = allow_unit(home, entry)
        if unit is not None and not is_denied(unit):
            units.add(unit)
    for item in allow_items:
        if not item.strip():
            continue
        unit = _operator_unit(home, item)
        if unit is None:
            logger.warning(
                "Ignoring %s entry %r: it must name a dot entry at the top of HOME or a child of %s",
                ALLOW_KEY, item, ", ".join(DEFAULT_DENY_DIRS),
            )
        elif is_denied(unit):
            logger.warning("Ignoring %s entry %r: it names a credential store that stays hidden", ALLOW_KEY, item)
        else:
            units.add(unit)
    return tuple(sorted(units))
