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
