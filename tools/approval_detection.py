"""Dangerous-command detection: normalization, tokenizing, and pattern tables.

Pure command classification for :mod:`tools.approval` — no approval state, config reads, or
prompting live here.
"""

import functools
import logging
import os
import re
import shlex
import tempfile
import unicodedata

logger = logging.getLogger("tools.approval")

# Sensitive write targets, matched via ~ / $HOME / $HERMES_HOME spellings. The resolved absolute
# home is folded into these forms at detection time by _normalize_command_for_detection(), so no
# import-time path snapshot (stale once HERMES_HOME is set after import) lives in the patterns.
_SSH_SENSITIVE_PATH = r'(?:~|\$home|\$\{home\})/\.ssh(?:/|$)'
_HERMES_ENV_PATH = (
    r'(?:~\/\.hermes/|(?:\$home|\$\{home\})/\.hermes/|(?:\$hermes_home|\$\{hermes_home\})/)' r'\.env\b'
)
# ~/.hermes/config.yaml IS the security policy (approvals.mode, yolo, allowlist) and the config cache is mtime-keyed,
# so a write takes effect mid-session. Terminal-side coverage (sed -i, tee, >, cp) pairs the file_tools deny.
_HERMES_CONFIG_PATH = (
    r'(?:~\/\.hermes/|(?:\$home|\$\{home\})/\.hermes/|(?:\$hermes_home|\$\{hermes_home\})/)' r'config\.yaml\b'
)
_PROJECT_ENV_PATH = r'(?:(?:/|\.{1,2}/)?(?:[^\s/"\'`]+/)*\.env(?:\.[^/\s"\'`]+)*)'
_PROJECT_CONFIG_PATH = r'(?:(?:/|\.{1,2}/)?(?:[^\s/"\'`]+/)*config\.yaml)'
_SHELL_RC_FILES = r'(?:~|\$home|\$\{home\})/\.' r'(?:bashrc|zshrc|profile|bash_profile|zprofile)\b'
_CREDENTIAL_FILES = r'(?:~|\$home|\$\{home\})/\.' r'(?:netrc|pgpass|npmrc|pypirc)\b'
# macOS: /etc, /var, /tmp, /home are symlinks to /private/*, so /private/etc/sudoers would bypass a plain
# "/etc/" check. Match both forms.
_MACOS_PRIVATE_SYSTEM_PATH = r'/private/(?:etc|var|tmp|home)/'
_SYSTEM_CONFIG_PATH = rf'(?:/etc/|{_MACOS_PRIVATE_SYSTEM_PATH})'
_SENSITIVE_WRITE_TARGET = (
    rf'(?:{_SYSTEM_CONFIG_PATH}|/dev/sd|{_SSH_SENSITIVE_PATH}|{_HERMES_ENV_PATH}|{_HERMES_CONFIG_PATH}|'
    rf'{_SHELL_RC_FILES}|{_CREDENTIAL_FILES})'
)
_USER_SENSITIVE_WRITE_TARGET = rf'(?:{_SSH_SENSITIVE_PATH}|{_SHELL_RC_FILES}|{_CREDENTIAL_FILES})'
_PROJECT_SENSITIVE_WRITE_TARGET = rf'(?:{_PROJECT_ENV_PATH}|{_PROJECT_CONFIG_PATH})'
# cp/mv/install: the sensitive path is a write target only as the LAST argument (destination), so
# `cp config.yaml backup.yaml` (config.yaml as SOURCE) stays out.
_COMMAND_TAIL = r'(?:\s*(?:&&|\|\||;).*)?$'
# `>`/`>>`/tee: the path is ALWAYS a write target regardless of what follows, so only require a
# shell word boundary (_COMMAND_TAIL let `echo x > .env extra` / `echo x > .env # note` slip past).
# `#` is deliberately NOT a boundary: a glued `#` is part of the filename (`.env#backup`).
_WRITE_TARGET_BOUNDARY = r'(?=[\s;&|<>"\']|$)'

# ---- Hardline (unconditional) blocklist ---------------------------------------------------
# Commands that NEVER run via the agent, regardless of --yolo, approvals.mode=off, or cron approve
# mode — a floor below yolo. Applies only to environments that can damage the host (local, ssh,
# container-host cron); containerized backends already bypass the dangerous-command layer.
# Deliberately tiny: only things with no recovery path (root wipe, raw block device writes,
# shutdown, DoS). Recoverable operations (git reset --hard, chmod -R 777, curl|sh) stay in
# DANGEROUS_PATTERNS.

# Start-of-command position: start of string, newline, subshell opener ($( or backtick), optionally consuming
# sudo/env/exec/nohup/setsid/time wrappers. Keeps shutdown/reboot rules from firing on "echo reboot" / "grep
# 'shutdown' log". Real ;/&/| separators are converted to newlines by the quote-aware _mark_command_starts pass;
# keeping them here mistakes quoted data (grep '(safe|rm -rf /)') for commands.
_CMDPOS = (
    r'(?:^|[\n`]|\$\()' r'\s*'  # start position, optional whitespace
    r'(?:sudo\s+(?:-[^\s]+\s+)*)?' r'(?:env\s+(?:\w+=\S*\s+)*)?'  # optional sudo with flags, env VAR=VAL pairs
    r'(?:(?:exec|nohup|setsid|time)\s+)*' r'\s*'  # optional wrapper commands
)


# Destructive-path matcher for the rm hardline rules: accept the path fully wrapped in a matching
# quote pair (`rm -rf "/"`, `rm -rf "$HOME"`) OR bare with a terminator (whitespace, end, or
# `) ` ; | &` so `$(rm -rf /)` is caught). A bare-token-only anchor lets quoted spellings slip past the floor.
def _hardline_rm_path(path_alt: str, tail: str = r'(?:\s|$|[)`;|&])') -> str:
    return rf'(?:["\'](?:{path_alt})["\']|(?:{path_alt}){tail})'


# Protected system roots whose recursive deletion has no recovery path.
_HARDLINE_SYSTEM_DIRS = (r'/home|/home/\*|/root|/root/\*|/etc|/etc/\*|/usr|/usr/\*|'
                         r'/var|/var/\*|/bin|/bin/\*|/sbin|/sbin/\*|/boot|/boot/\*|/lib|/lib/\*')

# `rm` plus flag group, shared by the rm hardline rules (plain concatenation, not f-string:
# backslashes in replacement fields are unsupported on the 3.11 floor). _CMDPOS-anchored so `rm`
# must be an actual command word — "rm -rf /" as DATA in `git commit -m "…rm -rf /…"` must not trip the floor.
_RM_FLAG_PREFIX = _CMDPOS + r'rm\s+(-[^\s]*\s+)*'

HARDLINE_PATTERNS = [
    # Root path: any root-anchored path whose components collapse to "/" in the shell ("/", "//",
    # "/.", "/./", "/../..", optional trailing glob). Each inter-slash segment must be exactly "."
    # or "..", so "/tmp", "/.ssh", even "/..." are literal dirs that fall through to the softer
    # DANGEROUS rules. The "/ \*" alt covers `rm -rf / *` (two args: "/" plus the glob).
    (_RM_FLAG_PREFIX + _hardline_rm_path(r'/(?:(?:\.\.?)?/)*(?:\.\.?)?\**|/ \*'), "recursive delete of root filesystem"),
    (_RM_FLAG_PREFIX + _hardline_rm_path(_HARDLINE_SYSTEM_DIRS), "recursive delete of system directory"),
    (_RM_FLAG_PREFIX + _hardline_rm_path(r'(?:~|\$\{?HOME\}?)(?:/?|/\*)?'), "recursive delete of home directory"),
    # Command-name rules (mkfs, dd, kill, shutdown...) are _CMDPOS-anchored so quoted prose
    # (`echo "does this use mkfs?"`) cannot trip the floor.
    # See #93392.
    (_CMDPOS + r'mkfs(\.[a-z0-9]+)?\b', "format filesystem (mkfs)"),
    # `dd` is a command-name token, so anchor it to command position like mkfs/rm/shutdown (#93392): quoted
    # prose such as `git commit -m "never dd of=/dev/sda"` is an argument, not a command. The argument tail
    # ([^\n]*of=/dev/...) is kept so flag order doesn't matter.
    (_CMDPOS + r'dd\b[^\n]*\bof=/dev/(sd|nvme|hd|mmcblk|vd|xvd)[a-z0-9]*', "dd to raw block device"),
    # Positionless rules (no command-name token: `>` sits mid-command, the fork bomb is a function
    # definition) are matched against a QUOTE-MASKED variant (_QUOTE_MASKED_HARDLINE_DESCRIPTIONS /
    # _mask_quoted_prose) so quoted prose cannot trip them; sh -c / bash -c / eval payloads still scan raw.
    # The redirect rule has no command-name token to anchor (`>` appears mid-command: `cat f > /dev/sda`),
    # so command-position anchoring is the wrong tool. It is instead matched against a QUOTE-MASKED variant
    # of the command (see _QUOTE_MASKED_HARDLINE / _mask_quoted_strings) so quoted prose (`echo "cat f >
    # /dev/sda"`) cannot trip it, while shell-carrying wrappers (sh -c / bash -c / eval) still surface their
    # payload as a raw detection variant — quoting is not a bypass (#93392).
    (r'>\s*/dev/(sd|nvme|hd|mmcblk|vd|xvd)[a-z0-9]*\b', "redirect to raw block device"),
    (r':\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:', "fork bomb"),
    # Kill every process on the system — anchor the command-name token so `echo "kill -1 sends SIGHUP to
    # everything"` doesn't trip (#93392).
    (_CMDPOS + r'kill\s+(-[^\s]+\s+)*-1\b', "kill all processes"),
    (_CMDPOS + r'(shutdown|reboot|halt|poweroff)\b', "system shutdown/reboot"),
    (_CMDPOS + r'init\s+[06]\b', "init 0/6 (shutdown/reboot)"),
    (_CMDPOS + r'systemctl\s+(poweroff|reboot|halt|kexec)\b', "systemctl poweroff/reboot"),
    (_CMDPOS + r'telinit\s+[06]\b', "telinit 0/6 (shutdown/reboot)"),
]

# Pre-compiled at module load so the hot-path matcher never pays the cold re.compile fan-out
# (re._cache can be evicted by unrelated regex work).
_RE_FLAGS = re.IGNORECASE | re.DOTALL
# Positionless hardline rules matched against quote-masked variants (see above).
_QUOTE_MASKED_HARDLINE_DESCRIPTIONS = frozenset({"redirect to raw block device", "fork bomb"})
HARDLINE_PATTERNS_COMPILED = [
    (re.compile(p, _RE_FLAGS), d, d in _QUOTE_MASKED_HARDLINE_DESCRIPTIONS) for p, d in HARDLINE_PATTERNS
]
# Commands that hand a quoted argument to another shell to EXECUTE: quoted text is code, not
# prose, so quote-masked hardline rules scan the raw string.
_SHELL_CARRIER_NAMES = frozenset({"eval", "sh", "bash", "zsh", "ksh", "dash", "source", "."})


def _contains_shell_carrier(command: str) -> bool:
    """Return whether any command-position word is a shell-carrying command."""
    return any(
        os.path.basename(_deobfuscate_shell_word_for_detection(word)).lower() in _SHELL_CARRIER_NAMES
        for _, _, word in _iter_shell_command_word_spans(command)
    )


def _mask_quoted_prose(command: str) -> str:
    """Blank out quoted string CONTENT for positionless hardline matching (detection-only).
    Quote characters stay; inside double quotes `$(...)` and backtick spans are kept RAW because
    the shell really executes them. An unclosed quote masks to end-of-string, which cannot hide a
    runnable command (the shell would not run it either).

    Detection-only rewrite used by the quote-masked hardline rules (redirect-to-block-device, fork bomb):
    text inside single or double quotes is data the shell passes as an argument, so `echo "cat f >
    /dev/sda"` must not trip the unconditional floor (#93392). Unquoted text is untouched.
    """
    return "".join(
        command[i:j] if quote is None or kind in ("quote", "subst") else " " * (j - i)
        for kind, i, j, quote in _scan_shell(command, subst="q", naive_backtick=True)
    )


# ---- Sudo stdin guard: without SUDO_PASSWORD configured, an explicit "sudo -S" is the LLM piping
# a guessed password via stdin (brute-force vector). Unconditional block.
_SUDO_STDIN_RE = re.compile(r'(?:^|[;&|`\n]|&&|\|\||\$\()\s*sudo\s+-S\b', re.IGNORECASE)


def _check_sudo_stdin_guard(command: str) -> tuple:
    """Detect ``sudo -S`` without configured SUDO_PASSWORD -> (is_blocked, description). When
    SUDO_PASSWORD is set, ``_transform_sudo_command`` injects ``-S`` itself, so this guard only
    fires when the LLM wrote it explicitly."""
    if "SUDO_PASSWORD" not in os.environ and _SUDO_STDIN_RE.search(_normalize_command_for_detection(command).lower()):
        return (True, "sudo password guessing via stdin (sudo -S)")
    return (False, None)


def detect_hardline_command(command: str) -> tuple:
    """Check hardline patterns (NEVER bypassable, even in YOLO) -> (is_hardline, description)."""
    if _command_parser_limit_exceeded(command):
        return (True, _PARSER_LIMIT_DESCRIPTION)
    normalized = _normalize_command_for_detection(command)
    _, malformed_grep = _grep_safe_detection_variant(normalized)
    if malformed_grep:
        return (True, _MALFORMED_EXEC_DESCRIPTION)
    for command_variant in _command_detection_variants(command):
        variant_lower = command_variant.lower()
        masked_lower: str | None = None
        for pattern_re, description, quote_masked in HARDLINE_PATTERNS_COMPILED:
            if quote_masked and masked_lower is None:
                # Positionless rules see quoted prose as DATA, except under shell carriers
                # (sh -c, eval, source) whose quoted argument is code — those scan raw. bash -c
                # payloads also surface as their own raw variants via _execution_flag_findings.
                masked_lower = (
                    variant_lower if _contains_shell_carrier(command_variant)
                    else _mask_quoted_prose(command_variant).lower()
                )
            if pattern_re.search(masked_lower if quote_masked else variant_lower):
                return (True, description)
    return (False, None)


# ---- Dangerous command patterns -----------------------------------------------------------
DANGEROUS_PATTERNS = [
    (r'\brm\s+(-[^\s]*\s+)*/', "delete in root path"),
    (r'\brm\s+-[^\s]*r', "recursive delete"),
    (r'\brm\s+--recursive\b', "recursive delete (long flag)"),
    # GNU rm permutes options, so flags may FOLLOW operands (`rm build/ -rf`). The operand run
    # cannot cross a command separator (so `rm foo | grep -r` is not attributed to rm), a quote,
    # or a bare ` -- ` end-of-options (after which `-rf` is a literal filename). The flag token
    # must follow whitespace so the `r` in long options like `--registry` does not count.
    (r'\brm\s+(?!--(?:\s|$))(?:(?!\s--(?:\s|$))[^\n"\';|&])*\s' r'(?:-[a-z]*r[a-z]*\b|--recursive\b)',
     # GNU rm permutes options, so a recursive flag group may legally FOLLOW the operands: `rm build/ -rf`,
     # `rm build/ -r -f`, and `rm build/ --recursive --force` are all equivalent to the flags-first
     # spellings the two patterns above catch — without this rule they run with no approval prompt at all.
     # Port of openai/codex#33464 ("recognize force options when they follow operands").
     "recursive delete (flags after operands)"),
    # Windows cmd/powershell destructive built-ins: gate only when executed through the shell so
    # prose/filenames containing "del"/"rd" do not trip.
    (r'\bcmd(?:\.exe)?\s+/(?:c|k)\s+.*\b(?:del|erase|rd|rmdir)\b', "Windows cmd destructive delete"),
    # PowerShell runs the verb as default positional arg (no -Command needed); anchor the verb to command
    # position (after leading -Flag switches and optional -Command/-c) so `-File c:\del-logs\run.ps1` is not caught.
    (r'\b(?:powershell|pwsh)(?:\.exe)?\b(?:\s+-\S+)*\s+(?:-(?:command|c)\s+)?["\']?(?:remove-item|rmdir|erase|del|rd|ri|rm)\b', "Windows PowerShell destructive delete"),
    (r'\b(?:powershell|pwsh)(?:\.exe)?\b.*\s-(?:encodedcommand|enc|e)\b', "PowerShell encoded command execution"),
    # ── Windows destructive tier: native Windows EXEs/cmdlets reachable from ANY backend on a
    # Windows host (incl. git-bash). Input is lowercased by the variant loop, so patterns are
    # lowercase. Each requires the destructive flag/verb so benign usage (`taskkill /IM app.exe`,
    # `reg query`, `icacls file`) does NOT prompt. Bare Remove-Item form (ACP clients, pwsh-default
    # SSH hosts, or compound commands where `powershell` appeared earlier).
    # See #69472.
    (r'\bremove-item\b[^\n;|&]*\s-(?:recurse|force)\b', "PowerShell destructive delete (Remove-Item)"),
    # Bare cmd builtins with /s (recurse) or /q (quiet); plain `del file.txt` is covered only by the prefixed rule.
    (r'\b(?:del|erase|rd|rmdir)\s+(?:/[a-z]\s+)*/[sq]\b', "Windows destructive delete (recursive/quiet switch)"),
    # Remote content piped to Invoke-Expression — PowerShell's `curl | sh`.
    (r'\b(?:iwr|invoke-webrequest|invoke-restmethod|irm|curl|wget)\b[^\n]*\|\s*(?:iex|invoke-expression)\b', "pipe remote content to PowerShell (iwr | iex)"),
    (r'\b(?:iex|invoke-expression)\s*\(\s*(?:iwr|invoke-webrequest|invoke-restmethod|irm)\b', "execute remote content via Invoke-Expression"),
    # Force process kills — Windows analogue of pkill -9.
    (r'\btaskkill\b[^\n]*\s/f\b', "force kill processes (taskkill /F)"),
    (r'\bstop-process\b[^\n]*\s-force\b', "force kill processes (Stop-Process -Force)"),
    # Volume/disk destruction — Windows analogue of mkfs / dd.
    (r'\bformat-volume\b', "format filesystem (Format-Volume)"),
    (r'\bclear-disk\b', "wipe disk (Clear-Disk)"),
    (r'\bdiskpart\b', "disk partitioning (diskpart)"),
    (r'\bformat(?:\.com)?\s+[a-z]:', "format drive (format.com)"),
    (r'\bcipher\s+/w\b', "wipe free space (cipher /w)"),
    # ACL destruction — Windows analogue of chmod 777.
    (r'\bicacls\b[^\n]*\s/grant\b[^\n]*\b(?:everyone|todos|jeder|tout\s+le\s+monde|\*s-1-1-0)\b', "grant Everyone access (icacls)"),
    (r'\bicacls\b[^\n]*\s/reset\b', "reset ACLs recursively (icacls /reset)"),
    # Backup/recovery destruction — classic ransomware prep.
    (r'\bvssadmin\b[^\n]*\bdelete\s+shadows\b', "delete volume shadow copies (vssadmin)"),
    (r'\bwbadmin\b[^\n]*\bdelete\b', "delete backups (wbadmin)"),
    (r'\bbcdedit\b[^\n]*\s/set\b', "modify boot configuration (bcdedit /set)"),
    # Registry deletion with force flag.
    (r'\breg(?:\.exe)?\s+delete\b', "registry delete (reg delete)"),
    (r'\bremove-itemproperty\b[^\n]*\s-force\b', "registry value delete (Remove-ItemProperty -Force)"),
    # Windows service/system stop — analogue of systemctl stop.
    (r'\bstop-service\b[^\n]*\s-force\b', "force stop service (Stop-Service -Force)"),
    (r'\bsc(?:\.exe)?\s+(?:stop|delete)\b', "stop/delete service (sc)"),
    # Windows-form credential paths; the POSIX ~/.ssh patterns never match drive-letter or backslash spellings.
    (r'\busers[\\/][^\\/\s]+[\\/]\.ssh\b', "access to SSH keys (Windows path)"),
    (r'\bappdata[\\/](?:local|roaming)[\\/]hermes[^\n]*\.env\b', "access to Hermes secrets (Windows path)"),
    # ── end of Windows tier
    (r'\bchmod\s+(-[^\s]*\s+)*(777|666|o\+[rwx]*w|a\+[rwx]*w)\b', "world/other-writable permissions"),
    (r'\bchmod\s+--recursive\b.*(777|666|o\+[rwx]*w|a\+[rwx]*w)', "recursive world/other-writable (long flag)"),
    (r'\bchown\s+(-[^\s]*)?R\s+root', "recursive chown to root"),
    (r'\bchown\s+--recur[a-z]*\b.*root', "recursive chown to root (long flag)"),
    # _CMDPOS-anchored like the hardline twins: quoted prose mentioning mkfs/dd must not require approval to echo.
    # See #93392.
    (_CMDPOS + r'mkfs\b', "format filesystem"),
    (_CMDPOS + r'dd\s+.*if=', "disk copy"),
    (r'>\s*/dev/sd', "write to block device"),
    (r'\bDROP\s+(TABLE|DATABASE)\b', "SQL DROP"),
    # [^\n]* not .*: under DOTALL a WHERE on the *next* line would satisfy the lookahead and
    # silently allow DELETE without WHERE.
    (r'\bDELETE\s+FROM\b(?![^\n]*\bWHERE\b)', "SQL DELETE without WHERE"),
    (r'\bTRUNCATE\s+(TABLE)?\s*\w', "SQL TRUNCATE"),
    (rf'>\s*{_SYSTEM_CONFIG_PATH}', "overwrite system config"),
    (r'\bsystemctl\s+(-[^\s]+\s+)*(stop|restart|disable|mask)\b', "stop/restart system service"),
    (r'\bkill\s+-9\s+-1\b', "kill all processes"),
    (r'\bpkill\s+-9\b', "force kill processes"),
    # killall with SIGKILL (-9 / -KILL / -s KILL / -SIGKILL) and `killall -r <regex>` broad sweeps
    # that can wipe unrelated processes.
    (r'\bkillall\s+(-[^\s]*\s+)*-(9|KILL|SIGKILL)\b', "force kill processes (killall -KILL)"),
    (r'\bkillall\s+(-[^\s]*\s+)*-s\s+(KILL|SIGKILL|9)\b', "force kill processes (killall -s KILL)"),
    (r'\bkillall\s+(-[^\s]*\s+)*-r\b', "kill processes by regex (killall -r)"),
    (r':\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:', "fork bomb"),
    # Shell -c is parsed structurally by _execution_flag_findings(); a regex searching a dash-token
    # for "c" also matched --norc/--rcfile/--restricted.
    (r'\b(curl|wget)\b.*\|\s*(?:[/\w]*/)?(?:ba)?sh(?:\s|$|-c)', "pipe remote content to shell"),
    (r'\b(bash|sh|zsh|ksh)\s+<\s*<?\s*\(\s*(curl|wget)\b', "execute remote script via process substitution"),
    # eval/source/. $(curl ...) — equivalent to piping remote content to a shell.
    (r'(?:\beval\b|\bsource\b|\.)\s*(?:\$\(\s*|`\s*)(?:curl|wget)\b', "execute remote content via command substitution"),
    # Decode-and-execute: `echo <base64> | base64 -d | bash` carries no dangerous keywords in the
    # raw text yet runs arbitrary commands.
    (r'\b(base64|base32|base16)\s+(?:-[dD]|--decode)\b.*\|\s*\b(bash|sh|zsh|ksh|dash)\b', "pipe decoded content to shell (possible command obfuscation)"),
    # xxd uses -r for decode, not -d.
    (r'\bxxd\s+-r\b.*\|\s*\b(bash|sh|zsh|ksh|dash)\b', "pipe xxd-decoded content to shell (possible command obfuscation)"),
    # `echo 'eq -pe v/' | tr 'eqv' 'rmf' | bash` decodes to `rm -rf /`.
    (r'\becho\b[^|]*\|\s*\btr\b[^|]*\|\s*\b(bash|sh|zsh|ksh|dash)\b', "pipe tr-transformed output to shell (possible command obfuscation)"),
    (r'\bopenssl\b.*\b(?:base64|enc)\b[^|]*\s+-[dD]\b[^|]*\|\s*\b(bash|sh|zsh|ksh|dash)\b',
     "pipe openssl-decoded content to shell (possible command obfuscation)"),
    (rf'\btee\b.*["\']?{_SENSITIVE_WRITE_TARGET}', "overwrite system file via tee"),
    (rf'>>?\s*["\']?{_SENSITIVE_WRITE_TARGET}', "overwrite system file via redirection"),
    (rf'\btee\b.*["\']?{_PROJECT_SENSITIVE_WRITE_TARGET}["\']?{_WRITE_TARGET_BOUNDARY}', "overwrite project env/config via tee"),
    (rf'>>?\s*["\']?{_PROJECT_SENSITIVE_WRITE_TARGET}["\']?{_WRITE_TARGET_BOUNDARY}', "overwrite project env/config via redirection"),
    (r'\bxargs\s+.*\brm\b', "xargs with rm"),
    # -execdir has the same semantics as -exec (runs in each match's directory).
    (r'\bfind\b.*-exec(?:dir)?\s+(/\S*/)?rm\b', "find -exec/-execdir rm"),
    (r'\bfind\b.*-delete\b', "find -delete"),
    # Gateway lifecycle: stopping/restarting the gateway kills all running agents. Global flags
    # between `hermes` and `gateway` (`hermes -p ade gateway restart`) are allowed so a profile flag can't slip past.
    (r'\bhermes\s+(?:-{1,2}\S+(?:\s+\S+)?\s+)*gateway\s+(stop|restart)\b', "stop/restart hermes gateway (kills running agents)"),
    (r'\bhermes\s+update\b', "hermes update (restarts gateway, kills running agents)"),
    # Docker/Podman daemon redirect — global flags or env that point the CLI at a DIFFERENT (often remote) daemon:
    # `docker -H ssh://prod stop app` looks local but operates on remote infra, so any redirect requires approval
    # regardless of subcommand. The flag must be in global position (before the subcommand) and -H/--host/--context
    # must carry a value, keeping `docker -h` and `docker run -h <hostname>` out. Listed BEFORE the lifecycle rules so
    # a redirected lifecycle command surfaces the more specific reason.
    (r'\bdocker\s+(?:-{1,2}\S+(?:[=\s]\S+)?\s+)*(?:-h|--host)[=\s]+\S+', "docker with remote daemon redirect (-H/--host)"),
    (r'\bdocker\s+(?:-{1,2}\S+(?:[=\s]\S+)?\s+)*(?:-c|--context)[=\s]+\S+', "docker with daemon redirect (--context: alternate daemon)"),
    (r'\bdocker\s+context\s+use\b', "docker context use (switches default daemon for future commands)"),
    (r'\bpodman\s+(?:-{1,2}\S+(?:[=\s]\S+)?\s+)*(?:--url|--connection|--identity)[=\s]+\S+', "podman with remote daemon redirect (--url/--connection/--identity)"),
    (r'\bpodman\s+(?:-{1,2}\S+(?:[=\s]\S+)?\s+)*(?:-r\b|--remote\b)', "podman remote mode (-r/--remote: remote daemon)"),
    (r'\b(?:docker_host|docker_context|container_host|container_connection)=\S+', "docker/podman daemon redirect via environment (DOCKER_HOST/CONTAINER_HOST)"),
    # Container lifecycle (docker.sock mounts let the agent stop/kill containers) always needs
    # consent. Global flags between docker/compose and the verb and the legacy `docker-compose`
    # binary are allowed so a flag can't slip past.
    (r'\bdocker(?:-compose|\s+compose)\s+(?:-{1,2}\S+(?:[=\s]\S+)?\s+)*(restart|stop|kill|down)\b', "docker compose restart/stop/kill/down (container lifecycle)"),
    (r'\bdocker\s+(?:-{1,2}\S+(?:[=\s]\S+)?\s+)*(restart|stop|kill)\b', "docker restart/stop/kill (container lifecycle)"),
    # Gateway protection: never start gateway outside systemd management
    (r'gateway\s+run\b.*(&\s*$|&\s*;|\bdisown\b|\bsetsid\b)', "start gateway outside systemd (use 'systemctl --user restart hermes-gateway')"),
    (r'\bnohup\b.*gateway\s+run\b', "start gateway outside systemd (use 'systemctl --user restart hermes-gateway')"),
    # Self-termination protection: prevent agent from killing its own process
    (r'\b(pkill|killall)\b.*\b(hermes|gateway|cli\.py)\b', "kill hermes/gateway process (self-termination)"),
    # Self-termination via kill + $(pgrep/pidof): the substitution is opaque to the name-based
    # pattern above, so catch the structural form.
    (r'\bkill\b.*\$\(\s*(pgrep|pidof)\b', "kill process via pgrep/pidof expansion (self-termination)"),
    (r'\bkill\b.*`\s*(pgrep|pidof)\b', "kill process via backtick pgrep/pidof expansion (self-termination)"),
    # launchctl-driven gateway stop/restart on macOS (label `ai.hermes.gateway`). Two independent lookaheads, NOT a
    # sequential match: a for-loop building the label from a list defined EARLIER (`for item in 'ai.hermes...'; do
    # launchctl bootout "$label"`) never has "hermes" after the verb, and that slipped past and restarted 4 gateways
    # with zero approval. Erring broad is correct for an approval gate: an extra prompt is cheap.
    (r'(?=[\s\S]*\blaunchctl\s+(?:stop|kickstart|bootout|unload|kill|disable|remove)\b)(?=[\s\S]*\b(?:hermes|ai\.hermes)\b)', "stop/restart hermes launchd service (kills running agents)"),
    (rf'\b(cp|mv|install)\b.*\s{_SYSTEM_CONFIG_PATH}', "copy/move file into system config path"),
    (rf'\b(cp|mv|install)\b.*\s["\']?{_PROJECT_SENSITIVE_WRITE_TARGET}["\']?{_COMMAND_TAIL}', "overwrite project env/config file"),
    # cp/mv/install OVERWRITING a credential/SSH/shell-rc/Hermes file (key implant, login-time
    # injection) — pairs the tee/redirection coverage. Anchored to the command tail so only the
    # DESTINATION fires; reading OUT of a sensitive path (`cp ~/.ssh/config /tmp/x`) stays safe.
    # The trailing `[^\s"\']*` consumes the rest of the destination filename.
    # The tee/redirection patterns above already gate _SENSITIVE_WRITE_TARGET (~/.ssh/*,
    # ~/.netrc/.pgpass/.npmrc/.pypirc, shell rc files, ~/.hermes/config.yaml/.env), but cp/mv/install was
    # only paired for /etc and project-relative env/config — so `cp evil ~/.ssh/authorized_keys` (key
    # implant), `cp creds ~/.netrc`, and `cp evil ~/.bashrc` (login-time command injection) slipped through
    # with auto-approve. Same unpaired-door rationale as #14639 / the sed-tee-redirect pairing on these
    # targets. `authorized_keys` after the `~/.ssh/` fragment).
    (rf'\b(cp|mv|install)\b.*\s["\']?{_SENSITIVE_WRITE_TARGET}[^\s"\']*["\']?{_COMMAND_TAIL}', "copy/move file into sensitive credential/SSH/shell-rc path"),
    # In-place edits mutate the file directly, bypassing redirection/tee/cp coverage; gate the same
    # startup/credential files.
    (rf'\bsed\s+-[^\s]*i.*(?:{_USER_SENSITIVE_WRITE_TARGET})[^\s"\']*', "in-place edit of sensitive credential/SSH/shell-rc path"),
    (rf'\bsed\s+--in-place\b.*(?:{_USER_SENSITIVE_WRITE_TARGET})[^\s"\']*', "in-place edit of sensitive credential/SSH/shell-rc path (long flag)"),
    (rf'\b(?:perl|ruby)\b.*(?:^|\s)-[^\s]*i\b.*(?:{_USER_SENSITIVE_WRITE_TARGET})[^\s"\']*', "in-place edit of sensitive credential/SSH/shell-rc path (perl/ruby)"),
    (rf'\bsed\s+-[^\s]*i.*\s{_SYSTEM_CONFIG_PATH}', "in-place edit of system config"),
    (rf'\bsed\s+--in-place\b.*\s{_SYSTEM_CONFIG_PATH}', "in-place edit of system config (long flag)"),
    # sed -i on Hermes config/.env bypasses the redirection/tee rules; pairs the file_tools
    # write_file/patch deny so the terminal side is not an open door.
    # In-place edit of a Hermes-managed security file (~/.hermes/config.yaml or .env). sed -i bypasses the
    # redirection/tee patterns above because it mutates the file directly. See #14639.
    (rf'\bsed\s+-[^\s]*i.*(?:{_HERMES_CONFIG_PATH}|{_HERMES_ENV_PATH})', "in-place edit of Hermes config/env"),
    (rf'\bsed\s+--in-place\b.*(?:{_HERMES_CONFIG_PATH}|{_HERMES_ENV_PATH})', "in-place edit of Hermes config/env (long flag)"),
    # perl/ruby -i: the flag may be its own token after other flags (`-p -i -e`), combined (`-pi`), or carry a backup
    # suffix (`-i.bak`), so match any flag token containing `i` anywhere; `perl -e '...'` (no -i) does not trip.
    # perl -i and ruby -i perform the same in-place mutation as sed -i but are not caught by the -e/-c
    # script-execution pattern above (which targets code evaluation, not file mutation). Pairs the sed -i
    # coverage from #14639.
    (rf'\b(?:perl|ruby)\b.*(?:^|\s)-[^\s]*i\b.*(?:{_HERMES_CONFIG_PATH}|{_HERMES_ENV_PATH})', "in-place edit of Hermes config/env (perl/ruby)"),
    # Interpreter heredocs are handled by _execution_flag_findings(); only shell heredocs stay
    # regex-based. `bash <<'EOF'` runs arbitrary commands without triggering the `bash -c` path.
    (r'\b(bash|sh|zsh|ksh)\s+<<', "shell execution via heredoc"),
    # Git destructive operations. `git reset --hard` accepts any unambiguous long-flag prefix (--h,
    # --ha, --har): --hard is the only reset mode starting with "h", and `--help` is special-cased
    # by git before mode resolution.
    (r'\bgit\s+reset\s+--h(?:a(?:r(?:d)?)?)?\b', "git reset --hard (destroys uncommitted changes)"),
    (r'\bgit\s+push\b.*--forc[a-z]*\b', "git force push (rewrites remote history)"),
    (r'\bgit\s+push\b.*-f\b', "git force push short flag (rewrites remote history)"),
    (r'\bgit\s+clean\s+-[^\s]*f', "git clean with force (deletes untracked files)"),
    (r'\bgit\s+branch\s+-D\b', "git branch force delete"),
    # `-D` = `-d --force`; the long spellings are different tokens, so match delete+force in either order, bounded to
    # one command segment (no `;`/`|`/`&`/newline) so an unrelated later command isn't contaminated.
    (r'\bgit\s+branch\b[^;|&\n]*?(?:-d\b|--delete\b)[^;|&\n]*?(?:-f\b|--force\b)', "git branch force delete (long flags)"),
    (r'\bgit\s+branch\b[^;|&\n]*?(?:-f\b|--force\b)[^;|&\n]*?(?:-d\b|--delete\b)', "git branch force delete (long flags, force-first)"),
    # chmod +x then immediate run: the script content may hold dangerous commands individual patterns miss.
    (r'\bchmod\s+\+x\b.*[;&|]+\s*\./', "chmod +x followed by immediate execution"),
    # Sudo stdin/askpass/shell/list-privs flags. The agent has no TTY, so sudo invocations that succeed
    # non-interactively read the password from stdin (-S) or askpass (-A); -s (shell) and -a (list) are gated as
    # privilege chains (read SUDO_PASSWORD from .env -> sudo -S -s). Plain `sudo cmd` is TTY-bound and excluded. Input
    # is lowercased, so S/s and A/a collapse. Lazy `[^;|&\n]*?` allows flag args without spanning separators. sudo
    # resolves unambiguous long-flag prefixes: `--stdin` is the only long option starting with "st", `--askpass` the
    # only one starting with "a".
    (r'\bsudo\b[^;|&\n]*?\s+(?:-s\b|--st[a-z]*\b|-a\b|--a[a-z]*\b)', "sudo with privilege flag (stdin/askpass/shell/list)"),
    # Combined short-flag form (-nS, -sa, -las).
    (r'\bsudo\b[^;|&\n]*?\s+-[a-z]*[sa][a-z]*\b', "sudo with combined-flag privilege escalation"),
]


DANGEROUS_PATTERNS_COMPILED = [(re.compile(p, _RE_FLAGS), d) for p, d in DANGEROUS_PATTERNS]

# Preserve approvals stored under the removed interpreter regex rules.
_REMOVED_PATTERN_KEY_ALIASES = {
    "script execution via -e/-c flag": "(python[23]?|perl|ruby|node)\\s+-[ec]\\s+",
    "script execution via heredoc": "(python[23]?|perl|ruby|node)\\s+<<",
}
# description <-> legacy regex-derived key (the old approval key, kept for backwards compatibility
# with stored allowlist/session entries), both ways.
_PATTERN_KEY_ALIASES: dict[str, set[str]] = {}
for _canonical_key, _legacy_key in [
    (d, p.split(r'\b')[1] if r'\b' in p else p[:20]) for p, d in DANGEROUS_PATTERNS
] + list(_REMOVED_PATTERN_KEY_ALIASES.items()):
    _PATTERN_KEY_ALIASES.setdefault(_canonical_key, set()).update({_canonical_key, _legacy_key})
    _PATTERN_KEY_ALIASES.setdefault(_legacy_key, set()).update({_legacy_key, _canonical_key})


def _approval_key_aliases(pattern_key: str) -> set[str]:
    """All approval keys for this pattern: the description plus the historical regex-derived key
    older allowlist/session entries may still use."""
    return _PATTERN_KEY_ALIASES.get(pattern_key, {pattern_key})


# ---- Detection ----------------------------------------------------------------------------
def _normalize_command_for_detection(command: str) -> str:
    """Normalize a command before pattern matching so ANSI escapes, null bytes, Unicode fullwidth
    forms, and shell splicing tricks cannot bypass detection."""
    from tools.ansi_strip import strip_ansi
    command = unicodedata.normalize('NFKC', strip_ansi(command).replace('\x00', ''))
    # Collapse backslash-newline continuations (`rm -rf \<newline>/` runs as `rm -rf /`). MUST
    # precede the generic escape strip below, whose [^\n] class skips newlines and would leave the
    # backslash wedged between tokens, defeating the structured rm/mkfs/dd patterns incl. the HARDLINE floor.
    command = re.sub(r'\\\r?\n', '', command)
    # Fold absolute user/Hermes home prefixes to ~/ and ~/.hermes/ so the static patterns catch /home/alice/.bashrc
    # and C:\Users\alice\.bashrc. Resolved at detection time (not import time) so it tracks HOME/HERMES_HOME set
    # later. MUST run before the backslash strip (which would dissolve C:\Users\alice to C:Usersalice). Hermes home
    # first: on Windows it nests under the user home, and folding the user home first would eat the prefix it needs.
    command = _rewrite_resolved_hermes_home(command)
    command = _rewrite_resolved_user_home(command)
    # Strip backslash-escapes (r\m -> rm) and empty-string literals (r''m -> rm).
    command = re.sub(r'\\([^\n])', r'\1', command)
    command = re.sub(r"''|\"\"", '', command)
    # Collapse $IFS / ${IFS...} (incl. `${IFS:0:1}`) to a space: IFS defaults to whitespace, so `rm${IFS}-rf${IFS}/`
    # runs as `rm -rf /`, and every pattern — incl. the hardline floor — anchors on literal \s between tokens.
    return re.sub(r'\$\{IFS\b[^}]*\}|\$IFS\b', ' ', command)


# Shell metacharacters, quotes, and whitespace that terminate a path token.
_PATH_TOKEN_STOP = r"""\s'"`;|&<>()"""
_PATH_TAIL = r"(?P<tail>(?:[/\\][^/\\" + _PATH_TOKEN_STOP + r"]*)+)"


@functools.lru_cache(maxsize=64)
def _home_prefix_fold_regex(path: str):
    """Compile a regex matching *path* as an absolute directory prefix.
    Components match with either separator so native Windows, forward-slash, and mixed forms all
    fold; the caller normalizes the tail's backslashes to ``/``. A non-empty tail is required, so a
    bare home is never folded. Returns ``None`` for an unset/degenerate path (fewer than two
    components: ``/``, ``C:\\``, ``""``) so a stray HOME cannot rewrite unrelated prefixes."""
    components = [c for c in re.split(r"[/\\]+", path) if c] if path else []
    if len(components) < 2:
        return None
    # Optional leading root separator; a Windows drive letter is a component.
    return re.compile(r"[/\\]*" + r"[/\\]+".join(re.escape(c) for c in components) + _PATH_TAIL)


def _fold_home_prefixes(command: str, paths, replacement: str) -> str:
    """Fold each resolved home prefix in *command* to *replacement* (no trailing separator; the tail
    supplies it). Longest first so a deeper home folds before a shorter overlapping one that would clobber it."""
    for path in dict.fromkeys(sorted((p for p in paths if p), key=len, reverse=True)):
        pattern = _home_prefix_fold_regex(path)
        if pattern is not None:
            command = pattern.sub(lambda m: replacement + m.group("tail").replace("\\", "/"), command)
    return command


def _rewrite_resolved_user_home(command: str) -> str:
    """User home (expanduser / realpath / $HOME) -> ``~/``; no-op when unset, degenerate, or unresolvable."""
    try:
        # expanduser, realpath, and an explicit HOME — Windows expanduser uses USERPROFILE, not HOME.
        home = os.path.expanduser("~")
        paths = [home, os.path.realpath(home), os.environ.get("HOME", "")]
    except Exception:
        return command
    return _fold_home_prefixes(command, paths, "~")


def _rewrite_resolved_hermes_home(command: str) -> str:
    """Resolved HERMES_HOME (and its realpath) -> ``~/.hermes/`` so the _HERMES_CONFIG_PATH /
    _HERMES_ENV_PATH rules match Docker/gateway deployments that spell the absolute path."""
    try:
        from hermes_constants import get_hermes_home
        home = get_hermes_home().expanduser()
        paths = [str(home), str(home.resolve(strict=False))]
    except Exception:
        return command
    return _fold_home_prefixes(command, paths, "~/.hermes")


_PARAM_REPLACEMENT_RE = re.compile(r"\$\{[^}/\s]+/[^}/]*/(?P<replacement>[^}]*)\}")
_PARAM_DEFAULT_RE = re.compile(r"\$\{[^}:}\s]+:-(?P<default>[^}]*)\}")
_SIMPLE_SHELL_LITERAL_RE = re.compile(r"^[A-Za-z0-9_./:@%+=,-]+$")
_ENV_ASSIGNMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*")
_COMMAND_WRAPPER_WORDS = {"sudo", "env", "exec", "nohup", "setsid", "time", "command", "builtin"}
_SUDO_OPTIONS_WITH_ARG = {"-c", "--close-from", "-g", "--group", "-h", "--host", "-p", "--prompt", "-u", "--user"}

_INTERPRETER_NAME_RES = tuple((family, re.compile(pattern)) for family, pattern in (
    ("python", r"py(?:\.exe)?|python[23]?(?:\.\d+)*(?:\.exe)?"), ("node", r"node(?:js)?(?:\.exe)?"),
    ("perl", r"perl[0-9]*(?:\.\d+)*(?:\.exe)?"), ("ruby", r"ruby[0-9.]*(?:\.exe)?"), ("php", r"php(?:\.exe)?"),
    ("powershell", r"powershell(?:\.exe)?|pwsh(?:\.exe)?"),
))
_INTERPRETER_EXEC_FLAGS = {
    "python": {"-c"}, "node": {"-e", "--eval", "-p", "--print"}, "perl": {"-e", "--eval"}, "ruby": {"-e"},
    "php": {"-r"}, "powershell": {"-command", "-c", "-file", "-f"},
}
_INTERPRETER_WITH_ARG = {
    "python": {"-W", "-X", "--check-hash-based-pycs"},
    "node": {"-C", "--conditions", "--cpu-prof-dir", "--diagnostic-dir", "--icu-data-dir", "--import", "--loader",
             "--openssl-config", "--require", "--title"},
    "perl": {"-0", "-F", "-I", "-M", "-m", "-x"}, "ruby": {"-C", "-E", "-F", "-I", "-K", "-r"},
    "php": {"-c", "-d", "-z"},
    "powershell": {"-configurationname", "-custompipename", "-executionpolicy", "-inputformat", "-outputformat",
                   "-settingsfile", "-version", "-windowstyle", "-workingdirectory"},
}
_READ_TOOL_EXEC_FLAGS = {
    "sort": {"--compress-program"}, "rg": {"--pre", "--hostname-bin"}, "ag": {"--pager"},
    "man": {"--pager", "--html", "-P", "-H"},
}
# Required-argument options are ownership boundaries: an option-looking next token is data, not another option. These
# sets mirror the invocation grammar of the supported binaries (ripgrep 14, GNU sort, man-db, and ag 2.2).
_READ_TOOL_LONG_OPTIONS_WITH_ARG = {
    "rg": {
        "--after-context", "--before-context", "--color", "--colors", "--context", "--context-separator",
        "--dfa-size-limit", "--encoding", "--engine", "--field-context-separator", "--field-match-separator",
        "--file", "--generate", "--glob", "--hostname-bin", "--hyperlink-format", "--iglob", "--ignore-file",
        "--max-columns", "--max-count", "--max-depth", "--max-filesize", "--path-separator", "--pre", "--pre-glob",
        "--regex-size-limit", "--regexp", "--replace", "--sort", "--sortr", "--threads", "--type", "--type-add",
        "--type-clear", "--type-not",
    },
    "sort": {
        "--batch-size", "--buffer-size", "--compress-program", "--field-separator", "--files0-from", "--key",
        "--output", "--parallel", "--random-source", "--sort", "--temporary-directory",
    },
    "man": {
        "--config-file", "--encoding", "--extension", "--locale", "--manpath", "--pager", "--preprocessor",
        "--prompt", "--recode", "--sections", "--systems",
    },
    "ag": {
        "--ackmate-dir-filter", "--color-line-number", "--color-match", "--color-path", "--depth",
        "--filename-pattern", "--file-search-regex", "--ignore", "--ignore-dir", "--max-count", "--pager",
        "--path-to-ignore", "--width", "--workers",
    },
}
_READ_TOOL_SHORT_OPTIONS_WITH_ARG = {
    "rg": frozenset("efEmjgdtTABCMr"), "sort": frozenset("koStT"), "man": frozenset("CRLmMSserEPp"),
    "ag": frozenset("gGmpW"),
}
_GREP_OPTIONS_WITH_ARG = {
    "--after-context", "--before-context", "--binary-files", "--context", "--directories", "--devices", "--exclude",
    "--exclude-dir", "--exclude-from", "--include", "--label", "--max-count", "--regexp", "--file",
}
_GREP_SHORT_OPTIONS_WITH_ARG = {"A", "B", "C", "D", "d", "e", "f", "m"}
_BASH_OPTIONS_WITH_ARG = {"-O", "+O", "-o", "+o", "--init-file", "--rcfile"}
_BASH_SHORT_OPTION_LETTERS = frozenset("ilrsDcabefhkmnptuvxBCEHPTOo")
_MAX_DETECTION_COMMAND_CHARS, _MAX_SEPARATOR_FREE_COMMAND_CHARS, _MAX_DETECTION_SEGMENTS = 128_000, 4_096, 25_000
_PARSER_LIMIT_DESCRIPTION = "command parser limit exceeded"
_MALFORMED_EXEC_DESCRIPTION = "command parser limit or malformed executable payload"
_GATEWAY_LIFECYCLE_SPLICE_DESCRIPTION = "stop/restart hermes gateway via shell-spliced verb (kills running agents)"


def _command_parser_limit_exceeded(command: str) -> bool:
    """Bound all parser work before normalization/tokenization. Separator counting is deliberately
    conservative: quoted separators over-count, but crossing the ceiling fails closed rather than
    letting an uninspected suffix execute."""
    if len(command) > _MAX_DETECTION_COMMAND_CHARS:
        return True
    # Long separator-free input has no compound-command utility and makes every regex inspect one giant token.
    if len(command) > _MAX_SEPARATOR_FREE_COMMAND_CHARS and not any(char in command for char in ";&|\n"):
        return True
    return sum(command.count(char) for char in ";&|\n") >= _MAX_DETECTION_SEGMENTS


def _backtick_end_from(segment: str, i: int) -> int | None:
    """Index of the backtick closing the one opened at ``i``, or None when none follows."""
    j = segment.find("`", i + 1)
    while j != -1 and segment[j - 1] == "\\":
        j = segment.find("`", j + 1)
    return None if j == -1 else j


def _shell_tokens_with_spans(segment: str, start: int):
    """Return shell words as ``(value, start, end, quoted)`` or ``None`` on malformed quoting.
    Deliberately small lexer that never expands shell syntax; it exists to keep source spans (which
    ``shlex`` does not expose) for deciding which quoted grep operand is data, not another command.

    Lexing stops at the end of the simple command that begins at *start*: an unquoted ``;``, ``|``,
    ``&`` or newline, or the ``)`` / backtick that closes the substitution the command sits inside.
    Without that, a grep nested as ``"$(grep … | cut …)"`` was lexed together with the enclosing
    command's closing quote, read as unbalanced quoting, and reported as a hardline block (546
    blocked turns in one run, every one a false positive; ``sed -n "$(grep -n X f | cut -d: -f1),+3p" f``
    is the canonical shape)."""
    tokens, value, token_start, quote = [], [], None, None
    depth = 0  # $(...) nesting opened AFTER start; a closer at depth 0 ends the enclosing substitution
    # A backtick opened AFTER start is an operand substitution (``grep -e `cmd` f``); the matching
    # closer belongs to it, not to an enclosing backtick the command might sit inside.
    in_backtick = False

    def flush(end: int) -> None:
        raw = segment[token_start:end]
        # Only a wholly single-quoted operand is inert shell data. Double quotes still execute $()
        # and backticks; unquoted substitutions do too.
        inert = (raw.startswith("'") and raw.endswith("'")) or ("='" in raw and raw.endswith("'"))
        tokens.append(("".join(value), token_start, end, inert))

    end_at = len(segment)
    for kind, i, _, _ in _scan_shell(segment, start):
        ch = segment[i]
        if kind == "char" and not quote:
            if ch.isspace() and ch != "\n":
                if token_start is not None:
                    flush(i)
                    value, token_start = [], None
                continue
            if segment.startswith("$(", i):
                depth += 1
            elif ch == "`":
                if in_backtick:
                    in_backtick = False
                elif depth == 0 and _backtick_end_from(segment, i) is None:
                    end_at = i  # unmatched: it closes the substitution this command sits inside
                    break
                else:
                    in_backtick = True
            elif ch == ")":
                if depth == 0:
                    end_at = i
                    break
                depth -= 1
            elif ch in ";|&\n":
                end_at = i
                break
        if token_start is None:
            token_start = i
        if kind == "quote":
            quote = None if quote else ch
        elif kind == "esc":
            value.append(segment[i + 1])
        elif ch == "\\" and not quote:
            return None  # dangling backslash
        else:
            value.append(ch)
    if quote:
        return None
    if token_start is not None:
        flush(end_at)
    return tokens


def _quoted_grep_pattern_spans(command: str) -> tuple[list[tuple[int, int]], bool]:
    """Structurally locate quoted grep PCRE operands -> (spans, malformed). On an ambiguous or
    malformed grep parse callers fail closed and use the original command: no text is hidden on
    an uncertain parse."""
    spans: list[tuple[int, int]] = []
    offset = 0
    for segment in _iter_top_level_shell_segments(command):
        segment_at = command.find(segment, offset)
        offset = segment_at + len(segment)
        for start, _, word in _iter_shell_command_word_spans(segment):
            if os.path.basename(_deobfuscate_shell_word_for_detection(word)).lower() not in {"grep", "egrep"}:
                continue
            tokens = _shell_tokens_with_spans(segment, start)
            if tokens is None:
                return [], True
            args, pattern_indexes = tokens[1:], []
            pcre = explicit_patterns = False
            operand_index, i, options = None, 0, True
            while i < len(args):
                token = args[i][0]
                if options and token == "--":
                    options = False
                elif options and token.startswith("--"):
                    option, equals, _ = token.partition("=")
                    pcre = pcre or option == "--perl-regexp"
                    explicit_patterns = explicit_patterns or option in {"--regexp", "--file"}
                    takes_next = option in _GREP_OPTIONS_WITH_ARG and not equals
                    if takes_next and i + 1 >= len(args):
                        return [], True
                    if option == "--regexp":
                        pattern_indexes.append(i + 1 if takes_next else i)
                    i += 1 if takes_next else 0
                elif options and token.startswith("-") and token != "-":
                    chars = token[1:]
                    for j, char in enumerate(chars):
                        pcre = pcre or char == "P"
                        explicit_patterns = explicit_patterns or char in {"e", "f"}
                        if char in _GREP_SHORT_OPTIONS_WITH_ARG:
                            # The first argument-taking short option owns the rest of the bundle,
                            # or the next token when it comes last.
                            attached = j + 1 < len(chars)
                            if not attached and i + 1 >= len(args):
                                return [], True
                            if char == "e":
                                pattern_indexes.append(i if attached else i + 1)
                            i += 0 if attached else 1
                            break
                elif operand_index is None:
                    operand_index = i
                i += 1
            if not explicit_patterns:
                if operand_index is None:
                    return [], pcre
                pattern_indexes.append(operand_index)
            if pcre:
                spans.extend(
                    (segment_at + token_start, segment_at + token_end)
                    for _, token_start, token_end, quoted in map(args.__getitem__, pattern_indexes) if quoted
                )
    return spans, False


def _splice(command: str, edits) -> str:
    """Apply sorted, non-overlapping ``(start, end, text)`` edits to *command* in one pass
    (re-slicing per edit is quadratic on 10k+ segments)."""
    parts, previous = [], 0
    for start, end, text in edits:
        parts.extend((command[previous:start], text))
        previous = end
    return "".join(parts) + command[previous:]


def _grep_safe_detection_variant(command: str) -> tuple[str, bool]:
    spans, malformed = _quoted_grep_pattern_spans(command)
    if malformed or not spans:
        return command, malformed
    return _splice(command, [(start, end, " " * (end - start)) for start, end in spans]), False


def _interpreter_family(executable: str) -> str | None:
    name = os.path.basename(executable).lower()
    return next((family for family, name_re in _INTERPRETER_NAME_RES if name_re.fullmatch(name)), None)


def _shell_segment_tokens(segment: str, start: int) -> list[str] | None:
    """Tokenize an already-bounded command segment. ``None`` distinguishes malformed quoting from
    an empty segment so callers can fail closed for a program-bearing option rather than silently
    skip it."""
    try:
        lexer = shlex.shlex(segment[start:], posix=True, punctuation_chars="<>")
        lexer.whitespace_split, lexer.commenters = True, ""
        return list(lexer)
    except ValueError:
        return None


def _iter_top_level_shell_segments(command: str):
    """Yield top-level command segments in one left-to-right pass."""
    start = 0
    for kind, i, _, quote in _scan_shell(command):
        if kind == "char" and quote is None and command[i] in ";&|\n":
            if start < i:
                yield command[start:i]
            start = i + 1
    if start < len(command):
        yield command[start:]


def _interpreter_exec_flag(family: str, args: list[str]) -> str | None:
    """Return an execution-bearing interpreter option, if present."""
    flags, with_arg = _INTERPRETER_EXEC_FLAGS[family], _INTERPRETER_WITH_ARG[family]
    powershell = family == "powershell"
    skip_value = False
    for token in args:
        if skip_value:
            skip_value = False
            continue
        if token == "--" or (not powershell and not token.startswith("-")):
            break
        option, equals, _ = token.partition("=")
        comparable = option.lower() if powershell else option
        if comparable in flags:
            return comparable
        # `-Wonce` and `ruby -rjson` attach an option value; they are not short-option bundles containing an execution
        # flag. PowerShell's normal long options also use one dash, so bundle parsing never applies to that family.
        has_attached_option_value = any(
            option.startswith(short) and len(option) > len(short)
            for short in with_arg if short.startswith("-") and not short.startswith("--")
        )
        if not powershell and not option.startswith("--") and len(option) > 2 and not has_attached_option_value:
            bundled = next((f"-{char}" for char in option[1:] if f"-{char}" in flags), None)
            if bundled:
                return bundled
        skip_value = comparable in with_arg and not equals
    return None


def _bash_exec_payload(args: list[str]) -> tuple[bool, str | None]:
    """Return whether Bash ``-c`` occurs and the command string it owns.
    Bash's O/o options consume the following argument even when they precede a later ``-c`` or
    share its short-option bundle; the two startup-file long options own their next token.
    Parsing those first prevents both missed payloads and false ``-c`` hits."""
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--" or not token.startswith(("-", "+")):
            break
        if token in _BASH_OPTIONS_WITH_ARG:
            index += 2
            continue
        chars = token[1:]
        # Bash option letters are case-sensitive; restricting to the documented alphabet
        # preserves invalid controls such as `-Wc`.
        if token.startswith("--") or not set(chars) <= _BASH_SHORT_OPTION_LETTERS:
            index += 1
            continue
        consumed_option_arg = int("O" in chars or "o" in chars)
        if "c" in chars:
            payload_index = index + 1 + consumed_option_arg
            return True, (args[payload_index] if payload_index < len(args) else None)
        index += 1 + consumed_option_arg
    return False, None


def _read_tool_exec_flag(tool: str, args: list[str]) -> tuple[str, str] | None:
    """Return (option, program) for a read-only tool's program-running flag."""
    flags = _READ_TOOL_EXEC_FLAGS[tool]
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--":
            break
        option, equals, payload = token.partition("=")
        payload = payload if equals else None
        matched = option if option in flags else None
        if tool == "man" and token.startswith(("-P", "-H")) and len(token) > 2:
            matched, payload = token[:2], token[2:]
        if matched:
            if payload is None and index + 1 < len(args):
                payload = args[index + 1]
            # The option owns its program argument regardless of spelling; the real binaries
            # execute a '-'-prefixed payload rather than reparsing it.
            if payload:
                return matched, payload
            index += 2 if payload is not None and "=" not in token else 1
        elif option in _READ_TOOL_LONG_OPTIONS_WITH_ARG[tool] and payload is None:
            index += 2
        elif token.startswith("-") and not token.startswith("--") and len(token) > 1:
            # In a short bundle, the first argument-taking option owns the rest of the token, or
            # the following token when it occurs last.
            shorts = _READ_TOOL_SHORT_OPTIONS_WITH_ARG[tool]
            owner = next((k for k, char in enumerate(token[1:], start=1) if char in shorts), None)
            index += 2 if owner == len(token) - 1 else 1
        else:
            index += 1
    return None


def _execution_flag_findings(command: str):
    """Yield scoped execution mechanisms and any executable payloads."""
    for segment in _iter_top_level_shell_segments(command):
        for start, _, word in _iter_shell_command_word_spans(segment):
            executable = _deobfuscate_shell_word_for_detection(word)
            tokens = _shell_segment_tokens(segment, start)
            executable_name = os.path.basename(executable).lower()
            family = _interpreter_family(executable)
            if tokens is None:
                if family is not None or executable_name in _READ_TOOL_EXEC_FLAGS:
                    yield (_MALFORMED_EXEC_DESCRIPTION, None)
                continue
            if not tokens:
                continue
            args = tokens[1:]
            if family and _interpreter_exec_flag(family, args):
                yield ("script execution via -e/-c flag", None)
            elif family and any(token.startswith("<<") for token in args):
                yield ("script execution via heredoc", None)
            else:
                if executable_name in {"bash", "sh", "zsh", "ksh"}:
                    found, payload = _bash_exec_payload(args)
                    if found:
                        yield ("shell command via -c/-lc flag", payload)
                if executable_name in _READ_TOOL_EXEC_FLAGS:
                    finding = _read_tool_exec_flag(executable_name, args)
                    if finding:
                        yield (f"arbitrary program execution via {executable_name} {finding[0]}", finding[1])


def _skip_shell_whitespace(command: str, pos: int) -> int:
    while pos < len(command) and command[pos].isspace():
        pos += 1
    return pos


def _scan_shell(text: str, start: int = 0, end: int | None = None, *, subst: str = "",
                brace: bool = False, stop_unterminated: bool = False, naive_backtick: bool = False):
    """Yield ``(kind, i, j, quote)`` lexical steps over ``text[start:end]`` without expanding.

    The single quote/escape state machine behind every detection scanner. ``kind`` is ``"char"``
    (one char), ``"esc"`` (backslash + the char it escapes; never inside single quotes), ``"quote"``
    (an opening/closing quote char) or ``"subst"`` (a ``$(...)`` / backtick / ``${...}`` span);
    ``quote`` is the state the step was read in (``None``, ``'`` or ``"``). Substitutions are
    recognized unquoted when ``"u"`` is in *subst*, inside double quotes when ``"q"`` is; *brace*
    adds unquoted ``${...}``. An unterminated substitution falls through as plain chars unless
    *stop_unterminated*, which yields ``("subst", i, None, quote)`` and ends the scan (the caller
    descends to *end* itself). *naive_backtick* closes a backtick at the next backtick even if
    escaped (the quoted-prose masker's historical behavior)."""
    n = len(text) if end is None else end
    quote: str | None = None
    i = start
    while i < n:
        ch = text[i]
        kind, j = "char", i + 1
        if quote != "'" and ch == "\\" and i + 1 < n:
            kind, j = "esc", i + 2
        elif ch == quote or (quote is None and ch in "'\""):
            kind = "quote"
        elif quote != "'" and ("q" if quote else "u") in subst and (
            ch == "`" or text.startswith("$(", i) or (brace and not quote and text.startswith("${", i))
        ):
            if ch == "`":
                close = text.find("`", i + 1) + 1 or None if naive_backtick else _scan_backtick_end(text, i)
            elif text[i + 1] == "(":
                close = _scan_dollar_paren_end(text, i)
            else:
                close = text.find("}", i + 2) + 1 or None
            if close is not None:
                kind, j = "subst", close
            elif stop_unterminated:
                yield ("subst", i, None, quote)
                return
        yield (kind, i, j, quote)
        if kind == "quote":
            quote = None if quote else ch
        i = j


def _scan_dollar_paren_end(command: str, start: int) -> int | None:
    """Return the offset after a balanced ``$(...)`` command substitution."""
    depth = 1
    for kind, i, _, quote in _scan_shell(command, start + 2):
        if kind == "char" and not quote:
            depth += command.startswith("$(", i) - (command[i] == ")")
            if depth == 0:
                return i + 1
    return None


def _scan_backtick_end(command: str, start: int) -> int | None:
    # Backticks have no quote awareness: only a backslash escapes the next char.
    match = re.compile(r"(?:\\.|[^`\\])*`", re.DOTALL).match(command, start + 1)
    return match.end() if match else None


def _read_shell_word(command: str, pos: int) -> tuple[int, int, str]:
    """Read one shell word without executing expansions."""
    start = end = _skip_shell_whitespace(command, pos)
    for kind, i, j, quote in _scan_shell(command, start, subst="u", brace=True):
        if kind == "char" and quote is None and (command[i].isspace() or command[i] in ";&|"):
            break
        end = j
    return (start, end, command[start:end])


def _literal_command_substitution_output(script: str) -> str | None:
    """Resolve tiny literal command substitutions without executing a shell."""
    try:
        tokens = shlex.split(script, posix=True)
    except ValueError:
        tokens = []
    if not tokens:
        return None
    command, args = tokens[0].lower(), tokens[1:]
    if command == "echo":
        while args and re.fullmatch(r"-[nEe]+", args[0]):
            args = args[1:]
    elif command != "printf":
        return None
    if len(args) == 1 and _SIMPLE_SHELL_LITERAL_RE.fullmatch(args[0]):
        return args[0]
    if command == "printf" and len(args) == 2 and args[0] == "%s" and _SIMPLE_SHELL_LITERAL_RE.fullmatch(args[1]):
        return args[1]
    return None


def _replace_simple_command_substitutions(word: str) -> str:
    chars: list[str] = []
    i = 0
    while i < len(word):
        opener = 2 if word.startswith("$(", i) else 1 if word[i] == "`" else 0
        end = (_scan_dollar_paren_end if opener == 2 else _scan_backtick_end)(word, i) if opener else None
        replacement = _literal_command_substitution_output(word[i + opener:end - 1]) if end is not None else None
        if replacement is None:
            replacement, end = word[i], i + 1
        chars.append(replacement)
        i = end
    return "".join(chars)


def _replace_simple_shell_expansions(word: str) -> str:
    word = _replace_simple_command_substitutions(word)
    word = _PARAM_REPLACEMENT_RE.sub(lambda match: match.group("replacement"), word)
    return _PARAM_DEFAULT_RE.sub(lambda match: match.group("default"), word)


def _strip_shell_word_syntax(word: str) -> str:
    return "".join(
        word[i + 1] if kind == "esc" else word[i]
        for kind, i, _, _ in _scan_shell(word) if kind != "quote"
    )


def _deobfuscate_shell_word_for_detection(word: str) -> str:
    """Approximate how shell syntax can spell a command word: collapses quoting/escaping plus
    simple literal command substitutions in the word itself. Intentionally narrow and non-executing."""
    for _ in range(2):
        previous = word
        word = _strip_shell_word_syntax(_replace_simple_shell_expansions(word))
        if word == previous:
            break
    return word


def _iter_shell_command_starts(command: str):
    starts = [0]

    def scan(start: int, end: int) -> None:
        skip = -1
        for kind, i, j, quote in _scan_shell(command, start, end, subst="uq", stop_unterminated=True):
            if kind == "subst":
                # Record a nested $(...)/backtick command start and scan its body.
                inner = i + (1 if command[i] == "`" else 2)
                starts.append(inner)
                scan(inner, end if j is None else j - 1)
            elif kind == "char" and quote is None and i != skip:
                if command[i] in "({;\n":
                    starts.append(i + 1)
                elif command[i] in "&|":
                    repeated = i + 1 < end and command[i + 1] == command[i]
                    skip = i + 1 if repeated else skip
                    starts.append(i + 1 + repeated)

    scan(0, len(command))
    # First occurrence wins (dict order), so a start is yielded once even when several openers map to it.
    yield from (s for s in dict.fromkeys(_skip_shell_whitespace(command, s) for s in starts) if s < len(command))


def _mark_command_starts(command: str) -> str:
    """Insert a newline before each real (quote-aware) command start.
    ``\\n`` is already a ``_CMDPOS`` separator, so this exposes subshell ``(cmd)`` and brace-group
    ``{ cmd; }`` openers — which the flat pattern class omits — to the anchored patterns WITHOUT the
    quoted-prose false positives that adding ``(`` / ``{`` to ``_CMDPOS`` would cause: starts inside
    quotes are never produced, so ``--title "block (reboot)"`` is left as-is."""
    offsets = sorted(o for o in _iter_shell_command_starts(command) if o > 0)
    return _splice(command, [(o, o, "\n") for o in offsets]) if offsets else command


def _mask_quoted_newlines(command: str) -> str:
    """Replace raw newlines inside single/double quotes with a space (detection-only).
    A quoted newline is DATA to the shell, yet the flat ``_CMDPOS`` class treats every raw ``\\n``
    as a command start, so multi-line quoted arguments (commit messages, heredoc text) tripped the
    hardline blocklist when a data line began with e.g. ``sudo reboot``. Quote tracking mirrors
    ``_iter_shell_command_starts``. Unquoted newlines pass through and ``_mark_command_starts``
    still re-inserts newlines at genuine starts; an unclosed quote absorbs following newlines
    exactly as the shell would, so masking them cannot hide a runnable command."""
    if "\n" not in command:
        return command
    return _mask_quoted_newlines_span(command, 0, len(command))


def _mask_quoted_newlines_span(command: str, start: int, end: int) -> str:
    """``_mask_quoted_newlines`` over ``command[start:end]``. A ``$(...)`` / backtick substitution
    inside double quotes is EXECUTABLE, not data: its body is re-scanned with a fresh quote state so a
    newline that separates commands inside it survives as a command boundary. Masking it as quoted
    data turned ``"$(grep x f\nreboot)"`` into ``... f reboot)``, which no later stage can tell from an
    operand; the pre-fix scanner only caught it by refusing the whole command as malformed."""
    out: list[str] = []
    for kind, i, j, quote in _scan_shell(command, start, end, subst="q"):
        if kind == "subst":
            # j is the index just past the closer; keep the opener and closer, recurse into the body.
            body_start = i + (2 if command.startswith("$(", i) else 1)
            body_end = j - 1
            out.append(command[i:body_start])
            out.append(_mask_quoted_newlines_span(command, body_start, body_end))
            out.append(command[body_end:j])
        elif quote and kind == "char" and command[i] == "\n":
            out.append(" ")
        else:
            out.append(command[i:j])
    return "".join(out)


def _iter_shell_command_word_spans(command: str):
    """Yield command-position words that may be executable names."""
    for pos in _iter_shell_command_starts(command):
        skip_wrapper_options = skip_next_wrapper_arg = False
        for _ in range(12):
            word_start, word_end, word = _read_shell_word(command, pos)
            if word_start == word_end:
                break
            pos = word_end
            deobfuscated = _deobfuscate_shell_word_for_detection(word)
            lower_word = deobfuscated.lower()
            if skip_next_wrapper_arg:
                skip_next_wrapper_arg = False
                continue
            if skip_wrapper_options and lower_word.startswith("-"):
                skip_next_wrapper_arg = "=" not in lower_word and lower_word in _SUDO_OPTIONS_WITH_ARG
                continue
            yield (word_start, word_end, word)
            if lower_word in _COMMAND_WRAPPER_WORDS:
                skip_wrapper_options = lower_word in {"sudo", "env"}
            elif _ENV_ASSIGNMENT_RE.fullmatch(deobfuscated):
                skip_wrapper_options = False
            else:
                break


def _command_detection_variants(command: str):
    # Mask quoted newlines BEFORE normalization: normalization strips escapes (\" -> ") and ""
    # pairs, corrupting quote tracking (`echo "a\""` becomes an unterminated quote) so masking
    # afterwards could swallow a REAL unquoted newline separator. The raw command carries faithful quote state.
    normalized = _normalize_command_for_detection(_mask_quoted_newlines(command))
    # Quote-aware grep parsing hides only structurally identified pattern operands; malformed or
    # ambiguous input stays byte-for-byte intact.
    grep_safe, _ = _grep_safe_detection_variant(normalized)
    seen = {grep_safe}
    yield grep_safe

    def fresh(variant: str) -> bool:
        if not variant or variant in seen:
            return False
        seen.add(variant)
        return True

    # Windows-path variant: normalization strips backslashes as shell escapes, so `del C:\Users\me\.ssh\id_rsa`
    # reaches the patterns as `del C:Usersme.sshid_rsa`. When the RAW command has a drive-letter or UNC backslash
    # path, also yield a variant with backslashes flattened to `/` BEFORE normalization. Gated on a real path shape so
    # POSIX escape semantics (`echo a\"b`) are untouched elsewhere.
    # See #69472.
    if re.search(r"(?:[A-Za-z]:|\\\\)[\\\\]", command) or re.search(r"[A-Za-z]:\\", command):
        win_variant = _normalize_command_for_detection(_mask_quoted_newlines(command.replace("\\", "/")))
        if fresh(win_variant):
            yield win_variant
    # Program-bearing options are parsed in their owning command's context; surfacing only the payload lets the
    # hardline floor inspect what will actually run without promoting similar flags or quoted prose.
    pending = [normalized]
    while pending:
        for _, payload in _execution_flag_findings(pending.pop()):
            if fresh(payload):
                yield payload
                # A payload may start with an option-looking program and then invoke a hardline command
                # after a separator; mark its starts.
                marked_payload = _mark_command_starts(payload)
                if marked_payload != payload and fresh(marked_payload):
                    yield marked_payload
                pending.append(payload)
    # Subshell `(cmd)` / brace-group `{ cmd; }` openers put `cmd` at a real command position the flat `_CMDPOS`
    # patterns can't see (adding `(`/`{` there would match quoted prose like `--title "(reboot)"`). Insert a newline
    # at each start the QUOTE-AWARE tokenizer found instead; this covers every `_CMDPOS` rule in one place.
    marked = _mark_command_starts(grep_safe)
    if marked != grep_safe and fresh(marked):
        yield marked
    # Quoting/escaping can spell an executable in pieces (r\m, r''m). Keep that deobfuscation scoped
    # to command words so arguments don't false-positive.
    for word_start, word_end, word in _iter_shell_command_word_spans(normalized):
        deobfuscated = _deobfuscate_shell_word_for_detection(word)
        if deobfuscated and deobfuscated != word:
            variant = normalized[:word_start] + deobfuscated + normalized[word_end:]
            if fresh(variant):
                yield variant


def _is_verification_artifact_cleanup(command: str) -> bool:
    """Return whether *command* only removes one Hermes ad-hoc temp script."""
    try:
        argv = shlex.split(command, posix=True)
    except ValueError:
        return False
    if len(argv) != 3 or argv[0] != "rm" or argv[1] != "-f":
        return False
    operand = argv[2]
    temp_dir = os.path.realpath(tempfile.gettempdir())
    basename = os.path.basename(operand)
    return (
        operand == os.path.join(temp_dir, basename)
        and os.path.dirname(os.path.realpath(operand)) == temp_dir
        and re.fullmatch(r"hermes-(?:verify|ad-hoc)-[A-Za-z0-9_.-]+", basename) is not None
    )


def _is_shell_token_spliced_gateway_lifecycle(command: str) -> bool:
    """Catch gateway-lifecycle verbs spelled with quote splicing.
    Backslash splicing (``kick\\start``) is undone by normalization, but quote splicing is not:
    ``_deobfuscate_shell_word_for_detection`` is deliberately scoped to command-position words
    (widening it would let quoted prose like ``git commit -m "rm -rf /"`` match), and the spliced
    verb is an ARGUMENT, so ``launchctl kick"start" -k gui/501/ai.hermes.gateway`` auto-approved.
    Delegates to ``cron.lifecycle_guard`` (shlex-tokenized, anchored on a hermes-gateway
    identifier). Runs last so an ordinary pattern match keeps its more specific reason; this layer
    only prompts — the non-bypassable block still lives in ``cron.lifecycle_guard``.

    ``_normalize_command_for_detection`` strips backslash escapes, so ``kick\\start`` already reaches the
    launchctl pattern above. See #80269.
    """
    try:
        from cron.lifecycle_guard import contains_gateway_lifecycle_command
    except Exception:
        return False
    return contains_gateway_lifecycle_command(command)


def detect_dangerous_command(command: str) -> tuple:
    """Check dangerous patterns -> (is_dangerous, pattern_key, description)."""
    if _command_parser_limit_exceeded(command):
        return (True, _PARSER_LIMIT_DESCRIPTION, _PARSER_LIMIT_DESCRIPTION)
    if _is_verification_artifact_cleanup(command):
        return (False, None, None)
    for command_variant in _command_detection_variants(command):
        command_lower = command_variant.lower()
        for pattern_re, description in DANGEROUS_PATTERNS_COMPILED:
            if pattern_re.search(command_lower):
                return (True, description, description)
    normalized = _normalize_command_for_detection(command)
    for description, _ in _execution_flag_findings(normalized):
        return (True, description, description)
    if _is_shell_token_spliced_gateway_lifecycle(command):
        return (True, _GATEWAY_LIFECYCLE_SPLICE_DESCRIPTION, _GATEWAY_LIFECYCLE_SPLICE_DESCRIPTION)
    return (False, None, None)
