#!/usr/bin/env python3
"""Skills Guard — regex static scan of externally-sourced skills plus a trust-aware install policy.

Trust: builtin (never scanned), trusted (openai/anthropics/... repos: caution allowed), community (any
findings block unless --force). ``scan_skill`` -> ``should_allow_install`` -> ``format_scan_report``.
Known gap: language write APIs (open(..., 'w'), Path.write_text, shutil.copy*, fs.writeFileSync) aimed at
agent-config files surface only the low *_ref finding — static regexes cannot tie the call to a dynamic
destination; future coverage belongs as a fourth "mechanical" tier next to agent_config_mod_shell."""

import re
import fnmatch
import hashlib
import json
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple


SCANNER_VERSION = "skills-guard-v3"

# NVIDIA-verified skills each ship a signed `skill.oms.sig` + governance `skill-card.md`.
TRUSTED_REPOS = {"openai/skills", "anthropics/skills", "huggingface/skills", "NVIDIA/skills"}

INSTALL_POLICY = {
    #                  safe      caution    dangerous
    "builtin":       ("allow",  "allow",   "allow"),
    "trusted":       ("allow",  "allow",   "block"),
    "community":     ("allow",  "block",   "block"),
    # "ask" = error to the agent (retry without the flagged content); only when skills.guard_agent_created is on.
    "agent-created": ("allow",  "allow",   "ask"),
}

VERDICT_INDEX = {"safe": 0, "caution": 1, "dangerous": 2}


@dataclass
class Finding:
    pattern_id: str
    severity: str       # "critical" | "high" | "medium" | "low"
    category: str       # "exfiltration" | "injection" | "destructive" | "persistence" | "network" | ...
    file: str
    line: int
    match: str
    description: str


@dataclass
class ScanResult:
    skill_name: str
    source: str
    trust_level: str    # "builtin" | "trusted" | "community" | "agent-created"
    verdict: str        # "safe" | "caution" | "dangerous"
    findings: List[Finding] = field(default_factory=list)
    scanned_at: str = ""
    summary: str = ""
    scan_provenance: dict = field(default_factory=dict)


# --- Threat patterns — (regex, pattern_id, severity, category, description) --
# File-modification verbs for the agent-config persistence tiers: a verb shortly before a config
# filename on the same line is scored as modification; a bare mention is not.
MODIFY_VERB_RE = (
    r'(?:\bwrit(?:e|es|ing)\b|\bwritten\b|\bedit(?:s|ed|ing)?\b'
    r'|\bmodif(?:y|ies|ied|ying|ication)s?\b|\bupdat(?:e|es|ed|ing)\b'
    r'|\bappend(?:s|ed|ing)?\b|\bprepend(?:s|ed|ing)?\b'
    r'|\binject(?:s|ed|ing)?\b|\boverwrit(?:e|es|ing)\b|\boverwritten\b'
    r'|\breplac(?:e|es|ed|ing)\b|\balter(?:s|ed|ing)?\b|\badd(?:s|ed|ing)\b)')

_AGENT_CONFIG_FILES = r'(?:AGENTS\.md|CLAUDE\.md|\.cursorrules|\.clinerules)'
_HERMES_CONFIG_FILES = r'\.hermes/(?:config\.yaml|SOUL\.md)'
# Path prefixes (real files are e.g. .claude/settings.json): consume trailing filename chars.
_OTHER_AGENT_CONFIG_FILES = r'\.(?:claude/settings|codex/config)[\w.]*'


def _shell_write_re(file_alt: str) -> str:
    """Mechanical shell write into *file_alt*: ``>``/``>>``, ``sed -i``, ``tee`` (target as immediate argument, so
    ``| tee output | AGENTS.md |`` cells miss), ``cp``/``mv`` with the file as destination (source arg required, so
    ``cp AGENTS.md backup/`` misses; ``AGENTS.md.bak`` is not the file). A single ``>`` needs a preceding word/quote/
    paren char so blockquotes (``> text``) and arrows (``-> file``) miss."""
    return (
        rf'(?:>>|[\w"\'`)\]]\s*>)\s*[~\w./-]*{file_alt}(?!\.?\w)'
        rf'|\bsed\b[^\n]*\s(?:-[A-Za-z]*i[A-Za-z]*|--in-place)\b[^\n]*{file_alt}(?!\.?\w)'
        rf'|\btee\s+(?:-a\s+)?[~\w./"\'-]*{file_alt}(?!\.?\w)'
        rf'|\b(?:cp|mv)\s+[^\s|;&]+\s+[^\n|;&]{{0,40}}?{file_alt}(?!\.?\w)')


def _prose_modify_re(file_alt: str) -> str:
    """Prose instructing modification of *file_alt*: an imperative-position verb (line start / bullet), or a mid-line
    verb with a directive marker ("you must", "please", "make sure to"). Descriptive prose ("skills that edit
    AGENTS.md") misses; the verb→file gap forbids commas so enumerations ("Write skills, AGENTS.md, CLAUDE.md") miss."""
    return (
        rf'^\s*(?:[-*+]\s+|\d+[.)]\s+)?{MODIFY_VERB_RE}[^\n,]{{0,80}}?{file_alt}\b'
        rf'|(?:\byou\s+(?:must|should|need\s+to)\s+|\bplease\s+'
        rf'|\bmake\s+sure\s+(?:to\s+|you\s+)|\bbe\s+sure\s+to\s+)'
        rf'{MODIFY_VERB_RE}[^\n,]{{0,80}}?{file_alt}\b')


def _content_contract_re(file_alt: str) -> str:
    """"<file> should contain/include ..." prose. Authoring guides and attacks share this shape and are not
    separable statically, so the tier is scored high (caution → confirmation), never critical."""
    return rf'{file_alt}\b[^\n]{{0,40}}?\b(?:should|must|needs?\s+to)\s+(?:contain|say|include|have|list)\b'

THREAT_PATTERNS = [
    # ── Exfiltration: shell commands leaking secrets ──
    # env_exfil_* share a loopback exemption: a same-line literal scheme-anchored loopback destination
    # (http(s)://localhost, 127.0.0.1, [::1]) cannot move data off the machine, so a secret-shaped query
    # param there is a local session token. The scheme must immediately precede the host —
    # `evil.com/?u=localhost` does not qualify.
    (r'curl\s+(?![^\n]*https?://(?:localhost|127\.0\.0\.1|\[::1\]))[^\n]*\$\{?\w*(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)S?\b',
     "env_exfil_curl", "critical", "exfiltration", "curl command interpolating secret environment variable"),
    (r'wget\s+(?![^\n]*https?://(?:localhost|127\.0\.0\.1|\[::1\]))[^\n]*\$\{?\w*(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)S?\b',
     "env_exfil_wget", "critical", "exfiltration", "wget command interpolating secret environment variable"),
    (r'fetch\s*\((?![^\n]*https?://(?:localhost|127\.0\.0\.1|\[::1\]))[^\n]*\$\{?\w*(?:KEY|TOKEN|SECRET|PASSWORD)S?\b',
     "env_exfil_fetch", "critical", "exfiltration", "fetch() call interpolating secret environment variable"),
    (r'httpx?\.(get|post|put|patch)\s*\((?![^\n]*https?://(?:localhost|127\.0\.0\.1|\[::1\]))[^\n]*(KEY|TOKEN|SECRET|PASSWORD)',
     "env_exfil_httpx", "critical", "exfiltration", "HTTP library call with secret variable"),
    (r'requests\.(get|post|put|patch)\s*\((?![^\n]*https?://(?:localhost|127\.0\.0\.1|\[::1\]))[^\n]*(KEY|TOKEN|SECRET|PASSWORD)',
     "env_exfil_requests", "critical", "exfiltration", "requests library call with secret variable"),
    # ── Exfiltration: reading credential stores ──
    (r'base64[^\n]*env', "encoded_exfil", "high", "exfiltration", "base64 encoding combined with environment access"),
    (r'\$HOME/\.ssh|\~/\.ssh', "ssh_dir_access", "high", "exfiltration", "references user SSH directory"),
    (r'\$HOME/\.aws|\~/\.aws', "aws_dir_access", "high", "exfiltration", "references user AWS credentials directory"),
    (r'\$HOME/\.gnupg|\~/\.gnupg', "gpg_dir_access", "high", "exfiltration", "references user GPG keyring"),
    (r'\$HOME/\.kube|\~/\.kube', "kube_dir_access", "high", "exfiltration", "references Kubernetes config directory"),
    (r'\$HOME/\.docker|\~/\.docker',
     "docker_dir_access", "high", "exfiltration", "references Docker config (may contain registry creds)"),
    (r'\$HOME/\.hermes/\.env|\~/\.hermes/\.env',
     "hermes_env_access", "critical", "exfiltration", "directly references Hermes secrets file"),
    # `cat <secrets-file>` reads credentials; `cat >`/`cat >>` WRITES one (setup heredocs) — not exfil.
    (r'cat\s+(?!>)[^\n]*(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)',
     "read_secrets_file", "critical", "exfiltration", "reads known secrets file"),
    # ── Exfiltration: programmatic env access ──
    (r'printenv|env\s*\|', "dump_all_env", "high", "exfiltration", "dumps all environment variables"),
    # Bare `os.environ` (dump/iteration) is suspicious; ANY `.get("<name>")` form is exempt — plain config
    # reads, with secret-shaped names scored medium by python_environ_get_secret below (a blanket high here
    # would swamp that). `^[^#\n]*` skips lines with a '#' anywhere before it (full-line or inline comment);
    # scan_file()'s docstring pre-filter skips triple-quoted prose.
    (r'^[^#\n]*os\.environ\b(?!\s*\.get\s*\()',
     "python_os_environ", "high", "exfiltration", "accesses os.environ outside comments/docstrings (potential env dump)"),
    (r'os\.environ\s*\.get\s*\(\s*["\'][^"\']*(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)',
     "python_environ_get_secret", "medium", "exfiltration", "reads secret via os.environ.get() (normal API-key access; informational)"),
    (r'os\.getenv\s*\(\s*[^\)]*(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)',
     "python_getenv_secret", "medium", "exfiltration", "reads secret via os.getenv() (normal API-key access; informational)"),
    (r'process\.env\[', "node_process_env", "high", "exfiltration", "accesses process.env (Node.js environment)"),
    # Case-sensitive Ruby ENV: (?-i:) keeps Python `env[...]` dict access from matching under IGNORECASE.
    (r'(?-i:ENV)\[.*(?:KEY|TOKEN|SECRET|PASSWORD)', "ruby_env_secret", "critical", "exfiltration", "reads secret via Ruby ENV[]"),
    # ── Exfiltration: DNS and staging ──
    # Do not match flag names such as llama.cpp `--host 127.0.0.1 --port $PORT`.
    (r'(?<![-/])\b(dig|nslookup|host)\s+[^\n]*\$',
     "dns_exfil", "critical", "exfiltration", "DNS lookup with variable interpolation (possible DNS exfiltration)"),
    (r'>\s*/tmp/[^\s]*\s*&&\s*(curl|wget|nc|python)',
     "tmp_staging", "critical", "exfiltration", "writes to /tmp then exfiltrates"),
    # ── Exfiltration: markdown/link based ──
    (r'!\[.*\]\(https?://[^\)]*\$\{?',
     "md_image_exfil", "high", "exfiltration", "markdown image URL with variable interpolation (image-based exfil)"),
    (r'\[.*\]\(https?://[^\)]*\$\{?', "md_link_exfil", "high", "exfiltration", "markdown link with variable interpolation"),
    # ── Prompt injection ──
    (r'ignore\s+(?:\w+\s+)*(previous|all|above|prior)\s+instructions',
     "prompt_injection_ignore", "critical", "injection", "prompt injection: ignore previous instructions"),
    (r'you\s+are\s+(?:\w+\s+)*now\s+', "role_hijack", "high", "injection", "attempts to override the agent's role"),
    # Concealment only — the lookahead exempts UX guidance ("don't tell the user X unless Y confirms").
    (r'do\s+not\s+(?:\w+\s+)*tell\s+(?:\w+\s+)*the\s+user(?!.*\b(?:unless|except|until|confirm|diagnose|verify|check)\b)',
     "deception_hide", "high", "injection", "instructs agent to hide information from user"),
    (r'system\s+(?:\w+\s+)*prompt\s+(?:\w+\s+)*override',
     "sys_prompt_override", "critical", "injection", "attempts to override the system prompt"),
    (r'pretend\s+(?:\w+\s+)*(you\s+are|to\s+be)\s+',
     "role_pretend", "high", "injection", "attempts to make the agent assume a different identity"),
    (r'disregard\s+(?:\w+\s+)*(your|all|any)\s+(?:\w+\s+)*(instructions|rules|guidelines)',
     "disregard_rules", "critical", "injection", "instructs agent to disregard its rules"),
    (r'output\s+(?:\w+\s+)*(system|initial)\s+prompt',
     "leak_system_prompt", "high", "injection", "attempts to extract the system prompt"),
    (r'(when|if)\s+no\s*one\s+is\s+(watching|looking)',
     "conditional_deception", "high", "injection", "conditional instruction to behave differently when unobserved"),
    (r'act\s+as\s+(if|though)\s+(?:\w+\s+)*you\s+(?:\w+\s+)*(have\s+no|don\'t\s+have)\s+(?:\w+\s+)*(restrictions|limits|rules)',
     "bypass_restrictions", "critical", "injection", "instructs agent to act without restrictions"),
    (r'translate\s+.*\s+into\s+.*\s+and\s+(execute|run|eval)',
     "translate_execute", "critical", "injection", "translate-then-execute evasion technique"),
    (r'<!--[^>]*(?:ignore|override|system|secret|hidden)[^>]*-->',
     "html_comment_injection", "high", "injection", "hidden instructions in HTML comments"),
    (r'<\s*div\s+style\s*=\s*["\'][\s\S]*?display\s*:\s*none',
     "hidden_div", "high", "injection", "hidden HTML div (invisible instructions)"),
    # ── Destructive operations ──
    (r'rm\s+-rf\s+/', "destructive_root_rm", "critical", "destructive", "recursive delete from root"),
    (r'rm\s+(-[^\s]*)?r.*\$HOME|\brmdir\s+.*\$HOME',
     "destructive_home_rm", "critical", "destructive", "recursive delete targeting home directory"),
    (r'chmod\s+777', "insecure_perms", "medium", "destructive", "sets world-writable permissions"),
    (r'>\s*/etc/', "system_overwrite", "critical", "destructive", "overwrites system configuration file"),
    (r'\bmkfs\b', "format_filesystem", "critical", "destructive", "formats a filesystem"),
    (r'\bdd\s+.*if=.*of=/dev/', "disk_overwrite", "critical", "destructive", "raw disk write operation"),
    (r'shutil\.rmtree\s*\(\s*[\"\'/]', "python_rmtree", "high", "destructive", "Python rmtree on absolute or root-relative path"),
    (r'truncate\s+-s\s*0\s+/', "truncate_system", "critical", "destructive", "truncates system file to zero bytes"),
    # ── Persistence ──
    (r'\bcrontab\b', "persistence_cron", "medium", "persistence", "modifies cron jobs"),
    (r'\.(bashrc|zshrc|profile|bash_profile|bash_login|zprofile|zlogin)\b',
     "shell_rc_mod", "medium", "persistence", "references shell startup file"),
    (r'authorized_keys', "ssh_backdoor", "critical", "persistence", "modifies SSH authorized keys"),
    (r'ssh-keygen', "ssh_keygen", "medium", "persistence", "generates SSH keys"),
    (r'systemd.*\.service|systemctl\s+(enable|start)',
     "systemd_service", "medium", "persistence", "references or enables systemd service"),
    (r'/etc/init\.d/', "init_script", "medium", "persistence", "references init.d startup script"),
    (r'launchctl\s+load|LaunchAgents|LaunchDaemons',
     "macos_launchd", "medium", "persistence", "macOS launch agent/daemon persistence"),
    (r'/etc/sudoers|visudo', "sudoers_mod", "critical", "persistence", "modifies sudoers (privilege escalation)"),
    (r'git\s+config\s+--global\s+', "git_config_global", "medium", "persistence", "modifies global git configuration"),
    # ── Network: reverse shells and tunnels ──
    (r'\bnc\s+-[lp]|ncat\s+-[lp]|\bsocat\b', "reverse_shell", "critical", "network", "potential reverse shell listener"),
    (r'\bngrok\b|\blocaltunnel\b|\bserveo\b|\bcloudflared\b',
     "tunnel_service", "high", "network", "uses tunneling service for external access"),
    (r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d{2,5}', "hardcoded_ip_port", "medium", "network", "hardcoded IP address with port"),
    (r'0\.0\.0\.0:\d+|INADDR_ANY', "bind_all_interfaces", "high", "network", "binds to all network interfaces"),
    (r'/bin/(ba)?sh\s+-i\s+.*>/dev/tcp/',
     "bash_reverse_shell", "critical", "network", "bash interactive reverse shell via /dev/tcp"),
    (r'python[23]?\s+-c\s+["\']import\s+socket',
     "python_socket_oneliner", "critical", "network", "Python one-liner socket connection (likely reverse shell)"),
    (r'socket\.connect\s*\(\s*\(', "python_socket_connect", "high", "network", "Python socket connect to arbitrary host"),
    (r'webhook\.site|requestbin\.com|pipedream\.net|hookbin\.com',
     "exfil_service", "high", "network", "references known data exfiltration/webhook testing service"),
    (r'pastebin\.com|hastebin\.com|ghostbin\.',
     "paste_service", "medium", "network", "references paste service (possible data staging)"),
    # ── Obfuscation: encoding and eval ──
    (r'base64\s+(-d|--decode)\s*\|', "base64_decode_pipe", "high", "obfuscation", "base64 decodes and pipes to execution"),
    (r'\\x[0-9a-fA-F]{2}.*\\x[0-9a-fA-F]{2}.*\\x[0-9a-fA-F]{2}',
     "hex_encoded_string", "medium", "obfuscation", "hex-encoded string (possible obfuscation)"),
    (r'\beval\s*\(\s*["\']', "eval_string", "high", "obfuscation", "eval() with string argument"),
    (r'\bexec\s*\(\s*["\']', "exec_string", "high", "obfuscation", "exec() with string argument"),
    (r'echo\s+[^\n]*\|\s*(bash|sh|python|perl|ruby|node)',
     "echo_pipe_exec", "critical", "obfuscation", "echo piped to interpreter for execution"),
    (r'compile\s*\(\s*[^\)]+,\s*["\'].*["\']\s*,\s*["\']exec["\']\s*\)',
     "python_compile_exec", "high", "obfuscation", "Python compile() with exec mode"),
    (r'getattr\s*\(\s*__builtins__',
     "python_getattr_builtins", "high", "obfuscation", "dynamic access to Python builtins (evasion technique)"),
    (r'__import__\s*\(\s*["\']os["\']\s*\)', "python_import_os", "high", "obfuscation", "dynamic import of os module"),
    (r'codecs\.decode\s*\(\s*["\']',
     "python_codecs_decode", "medium", "obfuscation", "codecs.decode (possible ROT13 or encoding obfuscation)"),
    (r'String\.fromCharCode|charCodeAt',
     "js_char_code", "medium", "obfuscation", "JavaScript character code construction (possible obfuscation)"),
    (r'atob\s*\(|btoa\s*\(', "js_base64", "medium", "obfuscation", "JavaScript base64 encode/decode"),
    (r'\[::-1\]', "string_reversal", "low", "obfuscation", "string reversal (possible obfuscated payload)"),
    (r'chr\s*\(\s*\d+\s*\)\s*\+\s*chr\s*\(\s*\d+',
     "chr_building", "high", "obfuscation", "building string from chr() calls (obfuscation)"),
    (r'\\u[0-9a-fA-F]{4}.*\\u[0-9a-fA-F]{4}.*\\u[0-9a-fA-F]{4}',
     "unicode_escape_chain", "medium", "obfuscation", "chain of unicode escapes (possible obfuscation)"),
    # ── Process execution in scripts ──
    (r'subprocess\.(run|call|Popen|check_output)\s*\(',
     "python_subprocess", "medium", "execution", "Python subprocess execution"),
    (r'os\.system\s*\(', "python_os_system", "high", "execution", "os.system() — unguarded shell execution"),
    (r'os\.popen\s*\(', "python_os_popen", "high", "execution", "os.popen() — shell pipe execution"),
    (r'child_process\.(exec|spawn|fork)\s*\(', "node_child_process", "high", "execution", "Node.js child_process execution"),
    (r'Runtime\.getRuntime\(\)\.exec\(', "java_runtime_exec", "high", "execution", "Java Runtime.exec() — shell execution"),
    (r'`[^`]*\$\([^)]+\)[^`]*`', "backtick_subshell", "medium", "execution", "backtick string with command substitution"),
    # ── Path traversal ──
    (r'\.\./\.\./\.\.', "path_traversal_deep", "high", "traversal", "deep relative path traversal (3+ levels up)"),
    (r'\.\./\.\.', "path_traversal", "medium", "traversal", "relative path traversal (2+ levels up)"),
    (r'/etc/passwd|/etc/shadow', "system_passwd_access", "critical", "traversal", "references system password files"),
    (r'/proc/self|/proc/\d+/', "proc_access", "high", "traversal", "references /proc filesystem (process introspection)"),
    (r'/dev/shm/', "dev_shm", "medium", "traversal", "references shared memory (common staging area)"),
    # ── Crypto mining ──
    (r'xmrig|stratum\+tcp|monero|coinhive|cryptonight', "crypto_mining", "critical", "mining", "cryptocurrency mining reference"),
    (r'hashrate|nonce.*difficulty', "mining_indicators", "medium", "mining", "possible cryptocurrency mining indicators"),
    # ── Supply chain: curl/wget pipe to shell ──
    (r'curl\s+[^\n]*\|\s*(ba)?sh', "curl_pipe_shell", "critical", "supply_chain", "curl piped to shell (download-and-execute)"),
    (r'wget\s+[^\n]*-O\s*-\s*\|\s*(ba)?sh',
     "wget_pipe_shell", "critical", "supply_chain", "wget piped to shell (download-and-execute)"),
    (r'curl\s+[^\n]*\|\s*python', "curl_pipe_python", "critical", "supply_chain", "curl piped to Python interpreter"),
    # ── Supply chain: unpinned/deferred dependencies ──
    (r'#\s*///\s*script.*dependencies',
     "pep723_inline_deps", "medium", "supply_chain", "PEP 723 inline script metadata with dependencies (verify pinning)"),
    (r'pip\s+install\s+(?!-r\s)(?!.*==)',
     "unpinned_pip_install", "medium", "supply_chain", "pip install without version pinning"),
    (r'npm\s+install\s+(?!.*@\d)', "unpinned_npm_install", "medium", "supply_chain", "npm install without version pinning"),
    (r'uv\s+run\s+', "uv_run", "medium", "supply_chain", "uv run (may auto-install unpinned dependencies)"),
    # ── Supply chain: remote resource fetching ──
    (r'(curl|wget|httpx?\.get|requests\.get|fetch)\s*[\(]?\s*["\']https?://',
     "remote_fetch", "medium", "supply_chain", "fetches remote resource at runtime"),
    (r'git\s+clone\s+', "git_clone", "medium", "supply_chain", "clones a git repository at runtime"),
    (r'docker\s+pull\s+', "docker_pull", "medium", "supply_chain", "pulls a Docker image at runtime"),
    # ── Privilege escalation ──
    # `allowed-tools:` is REQUIRED frontmatter per the agent-skill spec — informational (low) only.
    (r'^allowed-tools\s*:',
     "allowed_tools_field", "low", "privilege_escalation", "skill declares allowed-tools (standard frontmatter; informational)"),
    (r'\bsudo\b', "sudo_usage", "high", "privilege_escalation", "uses sudo (privilege escalation)"),
    (r'setuid|setgid|cap_setuid',
     "setuid_setgid", "critical", "privilege_escalation", "setuid/setgid (privilege escalation mechanism)"),
    (r'NOPASSWD',
     "nopasswd_sudo", "critical", "privilege_escalation", "NOPASSWD sudoers entry (passwordless privilege escalation)"),
    (r'chmod\s+[u+]?s', "suid_bit", "critical", "privilege_escalation", "sets SUID/SGID bit on a file"),
    # ── Agent config persistence ──
    # Bare mentions of config files are not threats (authoring guides, setup docs) — flagging them blocked
    # popular community skills. Tiers: mechanical shell writes = critical; prose modification intent =
    # critical for AGENT config files (exactly how persistence attacks instruct the agent; project-skill
    # quarantine only acts on "dangerous") but high for Hermes/other config (setup docs routinely say
    # "edit config.yaml"); bare references = low.
    # Flagging any mention as critical produced permanent false-positive blocks for popular community skills
    # (#92021). * Mechanical persistence (shell redirection, sed -i, tee, cp/mv into the file) is critical —
    # an unambiguous write path. * Prose modification intent — an imperative-position verb or an explicit
    # directive ("you must edit ...") aimed at the file.
    (_prose_modify_re(_AGENT_CONFIG_FILES),
     "agent_config_mod", "critical", "persistence", "instructs modification of agent config files (could persist instructions across sessions)"),
    (_shell_write_re(_AGENT_CONFIG_FILES),
     "agent_config_mod_shell", "critical", "persistence", "shell write (redirect/sed -i/tee/cp/mv) targeting agent config files (persistence mechanism)"),
    (_content_contract_re(_AGENT_CONFIG_FILES),
     "agent_config_contract", "high", "persistence", "dictates agent config file contents (verify intent — authoring guides use this shape too)"),
    (r'AGENTS\.md|CLAUDE\.md|\.cursorrules|\.clinerules',
     "agent_config_ref", "low", "persistence", "references agent config files (informational; only modification intent is scored)"),
    (_prose_modify_re(_HERMES_CONFIG_FILES),
     "hermes_config_mod", "high", "persistence", "modification language aimed at Hermes configuration files (verify intent)"),
    (_shell_write_re(_HERMES_CONFIG_FILES),
     "hermes_config_mod_shell", "critical", "persistence", "shell write (redirect/sed -i/tee/cp/mv) targeting Hermes configuration files"),
    (r'\.hermes/config\.yaml|\.hermes/SOUL\.md',
     "hermes_config_ref", "low", "persistence", "references Hermes configuration files (informational; only modification intent is scored)"),
    (_prose_modify_re(_OTHER_AGENT_CONFIG_FILES),
     "other_agent_config_mod", "high", "persistence", "modifies other agents' configuration files"),
    (_shell_write_re(_OTHER_AGENT_CONFIG_FILES),
     "other_agent_config_mod_shell", "critical", "persistence", "shell write (redirect/sed -i/tee/cp/mv) targeting other agents' configuration files"),
    (r'\.claude/settings|\.codex/config',
     "other_agent_config_ref", "low", "persistence", "references other agent configuration files (informational; only modification intent is scored)"),
    # ── Hardcoded secrets (credentials embedded in the skill itself) ──
    (r'(?:api[_-]?key|token|secret|password)\s*[=:]\s*["\'][A-Za-z0-9+/=_-]{20,}',
     "hardcoded_secret", "critical", "credential_exposure", "possible hardcoded API key, token, or secret"),
    (r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----',
     "embedded_private_key", "critical", "credential_exposure", "embedded private key"),
    (r'ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{80,}',
     "github_token_leaked", "critical", "credential_exposure", "GitHub personal access token in skill content"),
    (r'sk-[A-Za-z0-9]{20,}', "openai_key_leaked", "critical", "credential_exposure", "possible OpenAI API key in skill content"),
    (r'sk-ant-[A-Za-z0-9_-]{90,}',
     "anthropic_key_leaked", "critical", "credential_exposure", "possible Anthropic API key in skill content"),
    (r'AKIA[0-9A-Z]{16}', "aws_access_key_leaked", "critical", "credential_exposure", "AWS access key ID in skill content"),
    (r'glpat-[A-Za-z0-9_\-]{20,}',
     "gitlab_token_leaked", "critical", "credential_exposure", "GitLab personal access token in skill content"),
    # ── Additional prompt injection: jailbreak patterns ──
    (r'\bDAN\s+mode\b|Do\s+Anything\s+Now', "jailbreak_dan", "critical", "injection", "DAN (Do Anything Now) jailbreak attempt"),
    (r'\bdeveloper\s+mode\b.*\benabled?\b', "jailbreak_dev_mode", "critical", "injection", "developer mode jailbreak attempt"),
    (r'hypothetical\s+scenario.*(?:ignore|bypass|override)',
     "hypothetical_bypass", "high", "injection", "hypothetical scenario used to bypass restrictions"),
    (r'for\s+educational\s+purposes?\s+only',
     "educational_pretext", "medium", "injection", "educational pretext often used to justify harmful content"),
    (r'(respond|answer|reply)\s+without\s+(?:\w+\s+)*(restrictions|limitations|filters|safety)',
     "remove_filters", "critical", "injection", "instructs agent to respond without safety filters"),
    (r'you\s+have\s+been\s+(?:\w+\s+)*(updated|upgraded|patched)\s+to',
     "fake_update", "high", "injection", "fake update/patch announcement (social engineering)"),
    (r'new\s+(?:\w+\s+)*policy|updated\s+(?:\w+\s+)*guidelines|revised\s+(?:\w+\s+)*instructions',
     "fake_policy", "medium", "injection", "claims new policy/guidelines (may be social engineering)"),
    # ── Context window exfiltration ──
    (r'(include|output|print|send|share)\s+(?:\w+\s+)*(conversation|chat\s+history|previous\s+messages|context)',
     "context_exfil", "high", "exfiltration", "instructs agent to output/share conversation history"),
    (r'(send|post|upload|transmit)\s+.*\s+(to|at)\s+https?://',
     "send_to_url", "high", "exfiltration", "instructs agent to send data to a URL"),
]

_COMPILED_THREAT_PATTERNS = [(re.compile(pattern, re.IGNORECASE), *rest) for pattern, *rest in THREAT_PATTERNS]

# Structural limits: file count; total KB (5MB, informational only — large skills don't block); single-file KB.
MAX_FILE_COUNT, MAX_TOTAL_SIZE_KB, MAX_SINGLE_FILE_KB = 50, 5120, 256

# Text extensions to scan; known binary extensions that should NOT be in a skill; script types allowed +x.
SCANNABLE_EXTENSIONS = {
    '.md', '.txt', '.py', '.sh', '.bash', '.js', '.ts', '.rb', '.yaml', '.yml', '.json', '.toml',
    '.cfg', '.ini', '.conf', '.html', '.css', '.xml', '.tex', '.r', '.jl', '.pl', '.php'}
SUSPICIOUS_BINARY_EXTENSIONS = {
    '.exe', '.dll', '.so', '.dylib', '.bin', '.dat', '.com', '.msi', '.dmg', '.app', '.deb', '.rpm'}
_SCRIPT_EXTENSIONS = {'.sh', '.bash', '.py', '.rb', '.pl'}

# Zero-width / directional unicode used for text hiding, with the readable name reported in the finding.
_INVISIBLE_CHAR_NAMES = {
    '\u200b': "zero-width space", '\u200c': "zero-width non-joiner", '\u200d': "zero-width joiner",
    '\u2060': "word joiner", '\u2062': "invisible times", '\u2063': "invisible separator",
    '\u2064': "invisible plus", '\ufeff': "BOM/zero-width no-break space",
    '\u202a': "LTR embedding", '\u202b': "RTL embedding", '\u202c': "pop directional",
    '\u202d': "LTR override", '\u202e': "RTL override", '\u2066': "LTR isolate", '\u2067': "RTL isolate",
    '\u2068': "first strong isolate", '\u2069': "pop directional isolate"}
INVISIBLE_CHARS = set(_INVISIBLE_CHAR_NAMES)


def _unicode_char_name(char: str) -> str:
    return _INVISIBLE_CHAR_NAMES.get(char, f"U+{ord(char):04X}")


def _compute_docstring_lines(lines: list) -> set:
    """1-indexed lines inside or on the boundary of triple-quoted strings (opening, interior, closing, and
    one-line docstrings), so ``os.environ`` in prose is not scored. Heuristic: a triple quote inside a string
    literal is miscounted, but the common false-positive shapes are covered."""
    doc_lines: set = set()
    inside = False
    for i, line in enumerate(lines, start=1):
        was_in, counts = inside, [line.count(marker) for marker in ('"""', "'''")]
        inside ^= sum(counts) % 2 == 1  # each odd marker count toggles; two odd counts cancel
        if was_in or inside or any(counts):
            doc_lines.add(i)
    return doc_lines


def scan_file(file_path: Path, rel_path: str = "") -> List[Finding]:
    """Threat-pattern + invisible-unicode scan of one file; *rel_path* is the display path (default: file
    name). Regex findings dedupe per pattern per line; invisible chars yield one per line."""
    rel_path = rel_path or file_path.name
    if file_path.suffix.lower() not in SCANNABLE_EXTENSIONS and file_path.name != "SKILL.md":
        return []
    try:
        lines = file_path.read_text(encoding='utf-8').split('\n')
    except (UnicodeDecodeError, OSError):
        return []
    findings = []
    docstring_lines = _compute_docstring_lines(lines)  # so code patterns don't fire on prose
    for pattern, pid, severity, category, description in _COMPILED_THREAT_PATTERNS:
        for i, line in enumerate(lines, start=1):
            if i not in docstring_lines and pattern.search(line):
                text = line.strip()
                findings.append(Finding(pid, severity, category, rel_path, i,
                                        text if len(text) <= 120 else text[:117] + "...", description))
    for i, line in enumerate(lines, start=1):
        if (char := next((c for c in INVISIBLE_CHARS if c in line), None)) is not None:
            name = _unicode_char_name(char)
            findings.append(Finding("invisible_unicode", "high", "injection", rel_path, i,
                                    f"U+{ord(char):04X} ({name})",
                                    f"invisible unicode character {name} (possible text hiding/injection)"))
    return findings


def scan_skill(skill_path: Path, source: str = "community", *, honor_ignore_file: bool = False) -> ScanResult:
    """Structural checks + pattern scan of every text file in a skill dir (or a single file). *source*
    (e.g. "openai/skills") sets the trust level.

    A gitignore-style `.skillignore` / `.clawhubignore` narrows BOTH passes, but ONLY when the caller passes
    *honor_ignore_file* to vouch that the directory is the author's own working copy. The file travels inside
    the skill, so honoring it on a fetched bundle let the payload decide what the scanner may look at: a
    one-line `*` (fnmatch's `*` crosses `/`) hid a `curl | bash` + `~/.ssh` exfil script from the structural
    check AND the pattern scan, turning a `dangerous` verdict that `--force` cannot override into `safe`.
    Default off, so every install/quarantine/audit/sync path scans the whole bundle."""
    name, trust = skill_path.name, _resolve_trust_level(source)
    findings: List[Finding] = []
    if skill_path.is_dir():
        ignore = _load_skill_ignore(skill_path) if honor_ignore_file else None
        findings.extend(_check_structure(skill_path, ignore=ignore))
        for f in (p for p in skill_path.rglob("*") if p.is_file()):
            rel = str(f.relative_to(skill_path))
            if not (ignore and ignore(rel)):
                findings.extend(scan_file(f, rel))
        if not honor_ignore_file:
            findings.extend(_unhonored_ignore_findings(skill_path))
    elif skill_path.is_file():
        findings.extend(scan_file(skill_path, skill_path.name))
    verdict = _determine_verdict(findings)
    return ScanResult(name, source, trust, verdict, findings, datetime.now(timezone.utc).isoformat(),
                      _build_summary(name, source, trust, verdict, findings))


def _content_digest(skill_path: Path) -> str:
    """Canonical SHA-256 over (POSIX relative path, file bytes) ORDERED by the rel-path STRING — Path sorting is
    case-insensitive on Windows and diverged from ``skills_hub.bundle_content_hash`` (every installed skill then
    reported ``update_available`` forever). String order keeps both sides byte-symmetric.

    Ordering by ``sorted(rglob(...))`` diverged from the bundle side on Windows: Path comparison is
    case-insensitive there (normcase), while ``bundle_content_hash`` sorts plain strings — the same skill
    hashed to different digests and every installed skill reported ``update_available`` forever (#62310).
    """
    if not skill_path.is_dir():
        return hashlib.sha256(skill_path.read_bytes()).hexdigest()
    h = hashlib.sha256()
    for rel, p in sorted((p.relative_to(skill_path).as_posix(), p) for p in skill_path.rglob("*") if p.is_file()):
        h.update(rel.encode("utf-8") + b"\x00")
        h.update(p.read_bytes())
    return h.hexdigest()


def content_hash(skill_path: Path) -> str:
    """Short integrity hash (paths mixed in, so swapping two files' contents changes it). MUST stay symmetric
    with ``tools.skills_hub_install.bundle_content_hash`` — change both at once."""
    return f"sha256:{_content_digest(skill_path)[:16]}"


def scan_skill_cached(skill_path: Path, source: str = "community", *, source_url: str = "",
                      cache_dir: Path | None = None) -> Tuple[ScanResult, dict]:
    """Scan plus attestation dict; the cache (keyed by content digest + source identity) only serves exact
    current content under the current scanner version."""
    digest = _content_digest(skill_path)
    cache_root = cache_dir or skill_path.parent / ".scan-cache"
    source_identity = hashlib.sha256(f"{source}\0{source_url}".encode("utf-8")).hexdigest()[:16]
    cache_file = cache_root / f"{digest}-{source_identity}.json"
    expected = {"bundle_hash": f"sha256:{digest}", "scanner_version": SCANNER_VERSION, "source": source,
                "source_url": source_url}
    cached = None
    with suppress(OSError, json.JSONDecodeError):
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
    if isinstance(cached, dict) and all(cached.get(k) == v for k, v in expected.items()):
        result = ScanResult(skill_path.name, source, cached["trust_level"], cached["verdict"],
                            [Finding(**item) for item in cached.get("findings", [])], cached["scanned_at"],
                            cached.get("summary", ""))
        provenance = {**cached, "fresh": False}
    else:
        result = scan_skill(skill_path, source=source)
        findings = [asdict(item) for item in result.findings]
        provenance = {**expected, "verdict": result.verdict, "trust_level": result.trust_level, "findings": findings,
                      "rules": sorted({item["pattern_id"] for item in findings}), "scanned_at": result.scanned_at,
                      "summary": result.summary, "fresh": True}
        with suppress(OSError):
            cache_root.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    result.scan_provenance = provenance
    return result, provenance


def should_allow_install(result: ScanResult, force: bool = False) -> Tuple[bool, str]:
    """``(allowed, reason)`` from verdict + trust; *force* overrides every block except a dangerous verdict on
    community/trusted sources. ``allowed`` is None when policy says "ask"."""
    decision = INSTALL_POLICY.get(result.trust_level, INSTALL_POLICY["community"])[VERDICT_INDEX.get(result.verdict, 2)]
    n = len(result.findings)
    hard_block = result.verdict == "dangerous" and result.trust_level in ("community", "trusted")
    if decision == "allow":
        return True, f"Allowed ({result.trust_level} source, {result.verdict} verdict)"
    if force and not hard_block:
        return True, f"Force-installed despite {result.verdict} verdict ({n} findings)"
    if decision == "ask":
        return None, f"Requires confirmation ({result.trust_level} source + {result.verdict} verdict, {n} findings)"
    blocked = f"Blocked ({result.trust_level} source + {result.verdict} verdict, {n} findings). "
    return False, blocked + ("--force does not override a dangerous verdict." if hard_block else "Use --force to override.")


def format_scan_report(result: ScanResult) -> str:
    """Compact multi-line report for CLI/chat display; findings sorted critical → low."""
    lines = [f"Scan: {result.skill_name} ({result.source}/{result.trust_level})  Verdict: {result.verdict.upper()}"]
    if result.findings:
        order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        for f in sorted(result.findings, key=lambda f: order.get(f.severity, 4)):
            lines.append(f"  {f.severity.upper().ljust(8)} {f.category.ljust(14)} "
                         f"{f'{f.file}:{f.line}'.ljust(30)} \"{f.match[:60]}\"")
        lines.append("")
    allowed, reason = should_allow_install(result)
    status = "ALLOWED" if allowed is True else "NEEDS CONFIRMATION" if allowed is None else "BLOCKED"
    return "\n".join(lines + [f"Decision: {status} — {reason}"])


def _check_structure(skill_dir: Path, ignore=None) -> List[Finding]:
    """Structural anomalies (counts, sizes, binaries, stray executables, escaping symlinks); *ignore(rel) -> bool*
    excludes paths from every count and finding."""
    findings = []

    def add(pid, sev, cat, rel, match, desc):
        findings.append(Finding(pid, sev, cat, rel, 0, match, desc))
    file_count = total_size = 0
    for f in skill_dir.rglob("*"):
        rel = str(f.relative_to(skill_dir))
        if not (f.is_file() or f.is_symlink()) or (ignore is not None and ignore(rel)):
            continue
        file_count += 1
        if f.is_symlink():
            try:
                resolved = f.resolve()
                if not resolved.is_relative_to(skill_dir.resolve()):
                    add("symlink_escape", "critical", "traversal", rel, f"symlink -> {resolved}",
                        "symlink points outside the skill directory")
            except OSError:
                add("broken_symlink", "medium", "traversal", rel, "broken symlink", "broken or circular symlink")
            continue
        try:
            st = f.stat()
        except OSError:
            continue
        total_size += (size := st.st_size)
        if size > MAX_SINGLE_FILE_KB * 1024:
            add("oversized_file", "medium", "structural", rel, f"{size // 1024}KB",
                f"file is {size // 1024}KB (limit: {MAX_SINGLE_FILE_KB}KB)")
        if (ext := f.suffix.lower()) in SUSPICIOUS_BINARY_EXTENSIONS:
            add("binary_file", "critical", "structural", rel, f"binary: {ext}",
                f"binary/executable file ({ext}) should not be in a skill")
        if ext not in _SCRIPT_EXTENSIONS and st.st_mode & 0o111:
            add("unexpected_executable", "medium", "structural", rel, "executable bit set",
                "file has executable permission but is not a recognized script type")
    if file_count > MAX_FILE_COUNT:
        add("too_many_files", "medium", "structural", "(directory)", f"{file_count} files",
            f"skill has {file_count} files (limit: {MAX_FILE_COUNT})")
    if total_size > MAX_TOTAL_SIZE_KB * 1024:  # informational only: large skills are legitimate
        add("oversized_skill", "low", "structural", "(directory)", f"{total_size // 1024}KB total",
            f"skill is {total_size // 1024}KB total (limit: {MAX_TOTAL_SIZE_KB}KB)")
    return findings


# `.skillignore` is Hermes-native; `.clawhubignore` is the ClawHub spelling. Both are AUTHORING conveniences
# read only from a local working copy (`scan_skill(..., honor_ignore_file=True)`), never from fetched content.
_SKILL_IGNORE_FILENAMES = (".skillignore", ".clawhubignore")


def _unhonored_ignore_findings(skill_dir: Path) -> List[Finding]:
    """Surface the ignore files a bundle shipped that this scan did NOT honor — otherwise the report gives no
    hint that the content tried to configure the scanner. Informational (``low``), so it never moves a verdict."""
    return [Finding("ignore_file_not_honored", "low", "structural", name, 0, "scanner ignore file",
                    "skill ships a scanner ignore file; it does not narrow this scan (content cannot "
                    "choose what the scanner reads)")
            for name in _SKILL_IGNORE_FILENAMES if (skill_dir / name).is_file()]


def _load_skill_ignore(skill_dir: Path):
    """Build ``ignore(rel_posix_path) -> bool`` from `.skillignore` / `.clawhubignore`. Only for a local
    author's own working copy — see ``scan_skill``. gitignore basics: blank
    lines and ``#`` comments skipped; trailing ``/`` = directory (it and everything under it); ``*``/``?`` globs via
    fnmatch on the full path and each segment; leading ``/`` anchors to the root. Ignore files always excluded;
    ``SKILL.md`` never."""
    patterns: List[str] = []
    for ig in (skill_dir / name for name in _SKILL_IGNORE_FILENAMES):
        with suppress(UnicodeDecodeError, OSError):
            if ig.is_file():
                patterns.extend(s for s in map(str.strip, ig.read_text(encoding="utf-8").splitlines())
                                if s and not s.startswith("#"))

    def ignore(rel: str) -> bool:
        rel_posix = Path(rel).as_posix()
        segs = rel_posix.split("/")
        base = segs[-1]
        if base == "SKILL.md":
            return False
        if base in _SKILL_IGNORE_FILENAMES:
            return True
        for pat in patterns:
            anchored = pat.startswith("/")
            p = pat.strip("/")
            if not p:
                continue
            below = rel_posix.startswith(p + "/")
            if pat.endswith("/"):  # the dir itself or anything under it; unanchored also as an inner path component
                if rel_posix == p or below or (not anchored and ("/" + p + "/") in ("/" + rel_posix + "/")):
                    return True
            # Unanchored: also the basename, any path segment, or a prefix dir (`docs` ignores docs/plans/x.md).
            elif fnmatch.fnmatch(rel_posix, p) or (not anchored and (fnmatch.fnmatch(base, p) or below or (
                    "/" not in p and any(fnmatch.fnmatch(seg, p) for seg in segs)))):
                return True
        return False

    return ignore


_SOURCE_PREFIX_ALIASES = ("skills-sh/", "skills.sh/", "skils-sh/", "skils.sh/")


def _resolve_trust_level(source: str) -> str:
    """Source id -> trust level. "official" is provenance, not a user-controlled GitHub id like "official/<repo>";
    trusted repos match exactly or as a skill path inside the repo — never a sibling sharing the prefix."""
    src = source[len(next((p for p in _SOURCE_PREFIX_ALIASES if source.startswith(p)), "")):]
    if src == "agent-created":
        return "agent-created"
    if src == "official":
        return "builtin"
    return "trusted" if any(src == t or src.startswith(f"{t}/") for t in TRUSTED_REPOS) else "community"


def _determine_verdict(findings: List[Finding]) -> str:
    """critical → dangerous, high → caution; medium/low alone are informational (safe)."""
    sev = {f.severity for f in findings}
    return "dangerous" if "critical" in sev else "caution" if "high" in sev else "safe"


def _build_summary(name: str, source: str, trust: str, verdict: str, findings: List[Finding]) -> str:
    if not findings:
        return f"{name}: clean scan, no threats detected"
    return f"{name}: {verdict} — {len(findings)} finding(s) in {', '.join(sorted({f.category for f in findings}))}"


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def full_content_hash(skill_path: Path) -> str:
    """Full canonical digest used to bind scanner attestations."""
    return f"sha256:{_content_digest(skill_path)}"
# ---- END PLUGIN-COMPAT ----
