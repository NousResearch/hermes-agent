"""Shared shell-startup write targets and profile-aware detection normalization."""

import logging
import os
import re

from agent.file_safety import (
    SHELL_RC_RELATIVE_PATHS,
    SHELL_RC_ZSH_FILENAMES,
    _guard_homes,
    _ssh_config_keys,
    build_shell_rc_approval_paths,
    deny_key,
)

logger = logging.getLogger("tools.approval")

# A Hermes home: the default root or a named profile under it (profiles/<name>/ has its own config.yaml/.env).
_HERMES_HOME_DIR = (
    r'(?:~\/\.hermes/|(?:\$home|\$\{home\})/\.hermes/|(?:\$hermes_home|\$\{hermes_home\})/)'
    r'(?:profiles/[^/\s"\'`]+/)?'
)
_HERMES_ENV_PATH = (
    _HERMES_HOME_DIR + r'(?:\.op)?\.env\b'
)
# ~/.hermes/config.yaml IS the security policy (approvals.mode, yolo, allowlist) and the config cache is mtime-keyed,
# so a write takes effect mid-session. Terminal-side coverage (sed -i, tee, >, cp) pairs the file_tools deny.
_HERMES_CONFIG_PATH = (
    _HERMES_HOME_DIR + r'config\.yaml\b'
)

_SSH_SENSITIVE_PATH = r'(?:~|\$home|\$\{home\})/\.ssh(?:/|$)'
_SECRET_CREDENTIAL_FILES = r'(?:~|\$home|\$\{home\})/\.' r'(?:netrc|pgpass|npmrc|pypirc)\b'

# Shell word boundaries used by exact-file targets. `#` is deliberately not
# included: a glued `#` is part of the filename, while a comment has leading
# whitespace and is already covered by `\s`. Leading boundary anchors at a path
# start so `/mnt/root/.bashrc` never folds into `/mnt~/.bashrc`.
_WRITE_TARGET_START = r'(?<![\w.~$}/\\-])'
_WRITE_TARGET_BOUNDARY = r'(?=[\s;&|<>"\']|$)'
_CREDENTIAL_FILES = (
    r'(?:~|\$home|\$\{home\})/\.'
    rf'(?:netrc|pgpass|npmrc|pypirc){_WRITE_TARGET_BOUNDARY}'
)
_SHELL_RC_RELATIVE_PATTERN = "|".join(
    re.escape(path) for path in SHELL_RC_RELATIVE_PATHS
)
_SHELL_RC_ZSH_PATTERN = "|".join(
    re.escape(filename) for filename in SHELL_RC_ZSH_FILENAMES
)
_SHELL_RC_FILES = (
    rf'(?:'
    rf'(?:~|\$home|\$\{{home\}})/(?:{_SHELL_RC_RELATIVE_PATTERN})'
    rf'|(?:\$zdotdir|\$\{{zdotdir\}})/(?:{_SHELL_RC_ZSH_PATTERN})'
    rf'|(?:\$xdg_config_home|\$\{{xdg_config_home\}})/fish/config\.fish'
    rf'|(?:\$bash_env|\$\{{bash_env\}})'
    rf'|(?:\$env|\$\{{env\}})'
    rf'){_WRITE_TARGET_BOUNDARY}'
)

def _rewrite_resolved_shell_rc_paths(command: str) -> str:
    """Fold relocated shell startup paths into the shared canonical forms."""
    try:
        paths: dict[str, str] = {}
        homes = _guard_homes()
        ssh_configs = _ssh_config_keys(homes)
        for home in homes:
            for path in build_shell_rc_approval_paths(home):
                if deny_key(path) in ssh_configs:
                    paths[path] = "~/.ssh/config"
                    continue
                relative = os.path.relpath(path, home)
                path_name = os.path.basename(path)
                if relative in SHELL_RC_RELATIVE_PATHS:
                    canonical = f"~/{relative}"
                elif path_name in SHELL_RC_ZSH_FILENAMES:
                    canonical = f"~/{path_name}"
                elif path.endswith(os.path.join("fish", "config.fish")):
                    canonical = "~/.config/fish/config.fish"
                else:
                    # Arbitrary BASH_ENV / ENV filenames use the same rc gate.
                    canonical = "~/.bashrc"
                paths.setdefault(path, canonical)
    except Exception:
        logger.debug("Could not resolve relocated shell startup paths", exc_info=True)
        return command

    for path in sorted(paths, key=lambda value: len(value), reverse=True):
        candidates: set[str] = {path, path.replace(os.sep, "/")}
        for candidate in candidates:
            command = re.sub(_WRITE_TARGET_START + re.escape(candidate) + _WRITE_TARGET_BOUNDARY,
                             lambda match: paths[path], command)
    return command


