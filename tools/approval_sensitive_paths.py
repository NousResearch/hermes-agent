"""Sensitive-path pattern fragments of the dangerous-command tables in :mod:`tools.approval_detection`
(moved from there): credential, SSH, shell-rc, Hermes config/env and system-config write targets."""

import hermes_re as re

from agent.file_safety import SHELL_RC_FILES


# Sensitive write targets, matched via ~ / $HOME / $HERMES_HOME spellings. The resolved absolute
# home is folded into these forms at detection time by _normalize_command_for_detection(), so no
# import-time path snapshot (stale once HERMES_HOME is set after import) lives in the patterns.
_SSH_SENSITIVE_PATH = r'(?:~|\$home|\$\{home\})/\.ssh(?:/|\n?\Z)'
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
_PROJECT_ENV_PATH = r'(?:(?:/|\.{1,2}/)?(?:[^\s/"\'`]+/)*\.env(?:\.[^/\s"\'`]+)?)'
_PROJECT_CONFIG_PATH = r'(?:(?:/|\.{1,2}/)?(?:[^\s/"\'`]+/)*config\.yaml)'
_SHELL_RC_FILES = rf"(?:~|\$home|\$\{{home\}})/(?:{'|'.join(re.escape(f) for f in SHELL_RC_FILES)})\b"
_CREDENTIAL_FILES = r'(?:~|\$home|\$\{home\})/\.' r'(?:netrc|pgpass|npmrc|pypirc)\b'
_HOME_PREFIX = r'(?:~|\$home|\$\{home\})/'
_SECRET_KEY_FILE = r'id_(?:rsa|ed25519|ecdsa|dsa)\b'
_SECRET_NAMED_PATH_FILE = (
    rf'{_HOME_PREFIX}\.(?:aws/credentials|git-credentials|config/gh/hosts\.yml|docker/config\.json|kube/config)\b'
    r'|/etc/(?:shadow|passwd)\b'
)
_SECRET_FILE_NAMED = rf'{_SECRET_NAMED_PATH_FILE}|\b{_SECRET_KEY_FILE}'
_SECRET_PATH_FILE = (
    rf'{_SSH_SENSITIVE_PATH}|{_HERMES_ENV_PATH}|{_CREDENTIAL_FILES}|{_PROJECT_ENV_PATH}(?![\w-])'
    rf'|{_SECRET_NAMED_PATH_FILE}'
)
_SECRET_VAR = r'\$\{?\w*(?:key|token|secret|passwd|password|credential)\w*\}?'
# macOS: /etc, /var, /tmp, /home are symlinks to /private/*, so /private/etc/sudoers would bypass a plain
# "/etc/" check. Match both forms.
_MACOS_PRIVATE_SYSTEM_PATH = r'/private/(?:etc|var|tmp|home)/'
_SYSTEM_CONFIG_PATH = rf'(?:/etc/|{_MACOS_PRIVATE_SYSTEM_PATH})'
# Two groups, for a rule that reads on past the target: a directory-like prefix takes any rest of its word, a
# named file ends at a word boundary.
_SENSITIVE_WRITE_DIR = rf'(?:{_SYSTEM_CONFIG_PATH}|/dev/sd|{_SSH_SENSITIVE_PATH})'
_SENSITIVE_WRITE_FILE = rf'(?:{_HERMES_ENV_PATH}|{_HERMES_CONFIG_PATH}|{_SHELL_RC_FILES}|{_CREDENTIAL_FILES})'
_SENSITIVE_WRITE_TARGET = rf'(?:{_SENSITIVE_WRITE_DIR}|{_SENSITIVE_WRITE_FILE})'
_USER_SENSITIVE_WRITE_TARGET = rf'(?:{_SSH_SENSITIVE_PATH}|{_SHELL_RC_FILES}|{_CREDENTIAL_FILES})'
# touch/mkdir/ln: the home-side targets only; /etc is out of their scope (#85321) and admin commands
# like `mkdir -p /etc/apt/keyrings` would be mislabelled as credential/SSH/shell-rc writes.
_HOME_SENSITIVE_WRITE_TARGET = rf'(?:{_USER_SENSITIVE_WRITE_TARGET}|{_HERMES_ENV_PATH}|{_HERMES_CONFIG_PATH})'
# Same set for touch/mkdir/ln, but a named FILE must end the shell word: `ln -s a ~/.bashrc.bak` /
# `touch ~/.bashrc.d/x` name a sibling, not the startup or credential file (`~/.ssh/` is a directory: any entry;
# a bare `~/.ssh` operand anywhere in the command names the directory itself, `~/.sshx` does not).
# A glued redirect (`touch ~/.bashrc>/dev/null`) ends the word too; the ln rule also allows trailing redirects.
_HOME_SENSITIVE_CREATE_FILE = (
    rf'(?:{_HOME_PREFIX}\.ssh|{_SHELL_RC_FILES}|{_CREDENTIAL_FILES}|{_HERMES_ENV_PATH}|{_HERMES_CONFIG_PATH})'
)
_HOME_SENSITIVE_CREATE_TARGET = rf'(?:{_SSH_SENSITIVE_PATH}|{_HOME_SENSITIVE_CREATE_FILE}' r'(?=["\'\s;&|)`<>]|$))'
# The ln rule's tail, without possessive quantifiers or a lookahead in the middle (RE2 has neither): any number
# of redirects, then the end of the command. A redirect is its operator plus `&` + target, blank + target or a
# glued target; a bare operator can only end the command (an `&` after it belongs to it, never a separator).
# Whitespace before the end may hold newlines, but a newline is not itself an end: more text may follow it.
_LN_REDIRECT = r'[<>]+(?:&\s*[^\s<>;&|)`]*|\s+[^\s<>;&|)`]*|[^\s<>;&|)`]+)'
_LN_BARE_REDIRECT_END = r'[<>]+(?:[;|)`]|\Z)'
_LN_COMMAND_END = r'\s*(?:[;&|)`]|\Z)'
_LN_TAIL = rf'(?:\s*\d*{_LN_REDIRECT})*(?:\s*\d*{_LN_BARE_REDIRECT_END}|{_LN_COMMAND_END})'
# A named file must end its shell word: a closing quote, a redirect not glued to it by fd digits, or the end.
_LN_FILE_TAIL = (rf'(?:["\']{_LN_TAIL}|(?:\s+\d*)?{_LN_REDIRECT}{_LN_TAIL}|(?:\s+\d*)?{_LN_BARE_REDIRECT_END}'
                 rf'|{_LN_COMMAND_END})')
_PROJECT_SENSITIVE_WRITE_TARGET = rf'(?:{_PROJECT_ENV_PATH}|{_PROJECT_CONFIG_PATH})'
# cp/mv/install: the sensitive path is a write target only as the LAST argument (destination), so
# `cp config.yaml backup.yaml` (config.yaml as SOURCE) stays out.
_COMMAND_TAIL = r'(?:\s*(?:&&|\|\||;).*)?$'
# `>`/`>>`/tee: the path is ALWAYS a write target regardless of what follows, so only require a
# shell word boundary (_COMMAND_TAIL let `echo x > .env extra` / `echo x > .env # note` slip past).
# `#` is deliberately NOT a boundary: a glued `#` is part of the filename (`.env#backup`).
_WRITE_TARGET_BOUNDARY = r'(?=[\s;&|<>"\']|$)'
