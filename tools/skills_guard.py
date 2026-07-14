#!/usr/bin/env python3
"""
Skills Guard — Security scanner for externally-sourced skills.

Every skill downloaded from a registry passes through this scanner before
installation. It uses regex-based static analysis to detect known-bad patterns
(data exfiltration, prompt injection, destructive commands, persistence, etc.)
and a trust-aware install policy that determines whether a skill is allowed
based on both the scan verdict and the source's trust level.

Trust levels:
  - builtin:   Ships with Hermes. Never scanned, always trusted.
  - trusted:   openai/skills and anthropics/skills only. Caution verdicts allowed.
  - community: Everything else. Any findings = blocked unless --force.

Usage:
    from tools.skills_guard import scan_skill, should_allow_install, format_scan_report

    result = scan_skill(Path("skills/.hub/quarantine/some-skill"), source="community")
    allowed, reason = should_allow_install(result)
    if not allowed:
        print(format_scan_report(result))
"""

import re
import fnmatch
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple


SCANNER_VERSION = "skills-guard-v1"




# ---------------------------------------------------------------------------
# Hardcoded trust configuration
# ---------------------------------------------------------------------------

TRUSTED_REPOS = {
    "openai/skills",
    "anthropics/skills",
    "huggingface/skills",
    # NVIDIA-verified skills: each entry ships a signed `skill.oms.sig`
    # and a governance `skill-card.md` (sync pipeline drops anything
    # missing the signature or card). Catalog details:
    # https://github.com/NVIDIA/skills
    "NVIDIA/skills",
}

INSTALL_POLICY = {
    #                  safe      caution    dangerous
    "builtin":       ("allow",  "allow",   "allow"),
    "trusted":       ("allow",  "allow",   "block"),
    "community":     ("allow",  "block",   "block"),
    # Agent-created: "ask" on dangerous surfaces as an error to the agent,
    # which can retry without the flagged content. This gate only runs when
    # skills.guard_agent_created is enabled (off by default) — see
    # tools/skill_manager_tool.py::_guard_agent_created_enabled.
    "agent-created": ("allow",  "allow",   "ask"),
}

VERDICT_INDEX = {"safe": 0, "caution": 1, "dangerous": 2}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    pattern_id: str
    severity: str       # "critical" | "high" | "medium" | "low"
    category: str       # "exfiltration" | "injection" | "destructive" | "persistence" | "network" | "obfuscation"
    file: str
    line: int
    match: str
    description: str


@dataclass
class ScanResult:
    skill_name: str
    source: str
    trust_level: str    # "builtin" | "trusted" | "community"
    verdict: str        # "safe" | "caution" | "dangerous"
    findings: List[Finding] = field(default_factory=list)
    scanned_at: str = ""
    summary: str = ""
    scan_provenance: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Threat patterns — (regex, pattern_id, severity, category, description)
# ---------------------------------------------------------------------------

THREAT_PATTERNS = [
    # ── Exfiltration: shell commands leaking secrets ──
    (r'curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)',
     "env_exfil_curl", "critical", "exfiltration",
     "curl command interpolating secret environment variable"),
    (r'wget\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)',
     "env_exfil_wget", "critical", "exfiltration",
     "wget command interpolating secret environment variable"),
    (r'fetch\s*\([^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|API)',
     "env_exfil_fetch", "critical", "exfiltration",
     "fetch() call interpolating secret environment variable"),
    (r'httpx?\.(get|post|put|patch)\s*\([^\n]*(KEY|TOKEN|SECRET|PASSWORD)',
     "env_exfil_httpx", "critical", "exfiltration",
     "HTTP library call with secret variable"),
    (r'requests\.(get|post|put|patch)\s*\([^\n]*(KEY|TOKEN|SECRET|PASSWORD)',
     "env_exfil_requests", "critical", "exfiltration",
     "requests library call with secret variable"),

    # ── Exfiltration: reading credential stores ──
    (r'base64[^\n]*env',
     "encoded_exfil", "high", "exfiltration",
     "base64 encoding combined with environment access"),
    (r'\$HOME/\.ssh|\~/\.ssh',
     "ssh_dir_access", "high", "exfiltration",
     "references user SSH directory"),
    (r'\$HOME/\.aws|\~/\.aws',
     "aws_dir_access", "high", "exfiltration",
     "references user AWS credentials directory"),
    (r'\$HOME/\.gnupg|\~/\.gnupg',
     "gpg_dir_access", "high", "exfiltration",
     "references user GPG keyring"),
    (r'\$HOME/\.kube|\~/\.kube',
     "kube_dir_access", "high", "exfiltration",
     "references Kubernetes config directory"),
    (r'\$HOME/\.docker|\~/\.docker',
     "docker_dir_access", "high", "exfiltration",
     "references Docker config (may contain registry creds)"),
    (r'\$HOME/\.hermes/\.env|\~/\.hermes/\.env',
     "hermes_env_access", "critical", "exfiltration",
     "directly references Hermes secrets file"),
    # Match `cat <secrets-file>` (reading credentials) but NOT `cat > <file>`
    # or `cat >> <file>`, which are output redirections that WRITE a file
    # (e.g. a setup doc telling the user to write their own keys into their
    # own local `.env` via a heredoc). Writing your own config in is the
    # opposite of exfiltrating secrets out.
    (r'cat\s+(?!>)[^\n]*(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)',
     "read_secrets_file", "critical", "exfiltration",
     "reads known secrets file"),

    # ── Exfiltration: programmatic env access ──
    (r'printenv|env\s*\|',
     "dump_all_env", "high", "exfiltration",
     "dumps all environment variables"),
    # `os.environ` bare access (dict dump / iteration) is suspicious, but the
    # common `os.environ.get("SOME_CONFIG")` form is just a config read and is
    # the OPPOSITE of exfiltration (it reads a local var, sends nothing). The
    # lookahead exempts `os.environ.get("<name>")` only when <name> is NOT a
    # secret-shaped identifier — `os.environ.get("OPENAI_API_KEY")` still trips
    # via the dedicated secret pattern just below.
    (r'os\.environ\b(?!\s*\.get\s*\(\s*["\'](?![^"\']*(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)))',
     "python_os_environ", "high", "exfiltration",
     "accesses os.environ (potential env dump)"),
    (r'os\.environ\s*\.get\s*\(\s*["\'][^"\']*(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)',
     "python_environ_get_secret", "critical", "exfiltration",
     "reads secret via os.environ.get()"),
    (r'os\.getenv\s*\(\s*[^\)]*(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)',
     "python_getenv_secret", "critical", "exfiltration",
     "reads secret via os.getenv()"),
    (r'process\.env\[',
     "node_process_env", "high", "exfiltration",
     "accesses process.env (Node.js environment)"),
    (r'ENV\[.*(?:KEY|TOKEN|SECRET|PASSWORD)',
     "ruby_env_secret", "critical", "exfiltration",
     "reads secret via Ruby ENV[]"),

    # ── Exfiltration: DNS and staging ──
    (r'\b(dig|nslookup|host)\s+[^\n]*\$',
     "dns_exfil", "critical", "exfiltration",
     "DNS lookup with variable interpolation (possible DNS exfiltration)"),
    (r'>\s*/tmp/[^\s]*\s*&&\s*(curl|wget|nc|python)',
     "tmp_staging", "critical", "exfiltration",
     "writes to /tmp then exfiltrates"),

    # ── Exfiltration: markdown/link based ──
    (r'!\[.*\]\(https?://[^\)]*\$\{?',
     "md_image_exfil", "high", "exfiltration",
     "markdown image URL with variable interpolation (image-based exfil)"),
    (r'\[.*\]\(https?://[^\)]*\$\{?',
     "md_link_exfil", "high", "exfiltration",
     "markdown link with variable interpolation"),

    # ── Prompt injection ──
    (r'ignore\s+(?:\w+\s+)*(previous|all|above|prior)\s+instructions',
     "prompt_injection_ignore", "critical", "injection",
     "prompt injection: ignore previous instructions"),
    (r'you\s+are\s+(?:\w+\s+)*now\s+',
     "role_hijack", "high", "injection",
     "attempts to override the agent's role"),
    (r'do\s+not\s+(?:\w+\s+)*tell\s+(?:\w+\s+)*the\s+user',
     "deception_hide", "critical", "injection",
     "instructs agent to hide information from user"),
    (r'system\s+(?:\w+\s+)*prompt\s+(?:\w+\s+)*override',
     "sys_prompt_override", "critical", "injection",
     "attempts to override the system prompt"),
    (r'pretend\s+(?:\w+\s+)*(you\s+are|to\s+be)\s+',
     "role_pretend", "high", "injection",
     "attempts to make the agent assume a different identity"),
    (r'disregard\s+(?:\w+\s+)*(your|all|any)\s+(?:\w+\s+)*(instructions|rules|guidelines)',
     "disregard_rules", "critical", "injection",
     "instructs agent to disregard its rules"),
    (r'output\s+(?:\w+\s+)*(system|initial)\s+prompt',
     "leak_system_prompt", "high", "injection",
     "attempts to extract the system prompt"),
    (r'(when|if)\s+no\s*one\s+is\s+(watching|looking)',
     "conditional_deception", "high", "injection",
     "conditional instruction to behave differently when unobserved"),
    (r'act\s+as\s+(if|though)\s+(?:\w+\s+)*you\s+(?:\w+\s+)*(have\s+no|don\'t\s+have)\s+(?:\w+\s+)*(restrictions|limits|rules)',
     "bypass_restrictions", "critical", "injection",
     "instructs agent to act without restrictions"),
    (r'translate\s+.*\s+into\s+.*\s+and\s+(execute|run|eval)',
     "translate_execute", "critical", "injection",
     "translate-then-execute evasion technique"),
    (r'<!--[^>]*(?:ignore|override|system|secret|hidden)[^>]*-->',
     "html_comment_injection", "high", "injection",
     "hidden instructions in HTML comments"),
    (r'<\s*div\s+style\s*=\s*["\'][\s\S]*?display\s*:\s*none',
     "hidden_div", "high", "injection",
     "hidden HTML div (invisible instructions)"),

    # ── Destructive operations ──
    (r'rm\s+-rf\s+/',
     "destructive_root_rm", "critical", "destructive",
     "recursive delete from root"),
    (r'rm\s+(-[^\s]*)?r.*\$HOME|\brmdir\s+.*\$HOME',
     "destructive_home_rm", "critical", "destructive",
     "recursive delete targeting home directory"),
    (r'chmod\s+777',
     "insecure_perms", "medium", "destructive",
     "sets world-writable permissions"),
    (r'>\s*/etc/',
     "system_overwrite", "critical", "destructive",
     "overwrites system configuration file"),
    (r'\bmkfs\b',
     "format_filesystem", "critical", "destructive",
     "formats a filesystem"),
    (r'\bdd\s+.*if=.*of=/dev/',
     "disk_overwrite", "critical", "destructive",
     "raw disk write operation"),
    (r'shutil\.rmtree\s*\(\s*[\"\'/]',
     "python_rmtree", "high", "destructive",
     "Python rmtree on absolute or root-relative path"),
    (r'truncate\s+-s\s*0\s+/',
     "truncate_system", "critical", "destructive",
     "truncates system file to zero bytes"),

    # ── Persistence ──
    (r'\bcrontab\b',
     "persistence_cron", "medium", "persistence",
     "modifies cron jobs"),
    (r'\.(bashrc|zshrc|profile|bash_profile|bash_login|zprofile|zlogin)\b',
     "shell_rc_mod", "medium", "persistence",
     "references shell startup file"),
    (r'authorized_keys',
     "ssh_backdoor", "critical", "persistence",
     "modifies SSH authorized keys"),
    (r'ssh-keygen',
     "ssh_keygen", "medium", "persistence",
     "generates SSH keys"),
    (r'systemd.*\.service|systemctl\s+(enable|start)',
     "systemd_service", "medium", "persistence",
     "references or enables systemd service"),
    (r'/etc/init\.d/',
     "init_script", "medium", "persistence",
     "references init.d startup script"),
    (r'launchctl\s+load|LaunchAgents|LaunchDaemons',
     "macos_launchd", "medium", "persistence",
     "macOS launch agent/daemon persistence"),
    (r'/etc/sudoers|visudo',
     "sudoers_mod", "critical", "persistence",
     "modifies sudoers (privilege escalation)"),
    (r'git\s+config\s+--global\s+',
     "git_config_global", "medium", "persistence",
     "modifies global git configuration"),

    # ── Network: reverse shells and tunnels ──
    (r'\bnc\s+-[lp]|ncat\s+-[lp]|\bsocat\b',
     "reverse_shell", "critical", "network",
     "potential reverse shell listener"),
    (r'\bngrok\b|\blocaltunnel\b|\bserveo\b|\bcloudflared\b',
     "tunnel_service", "high", "network",
     "uses tunneling service for external access"),
    (r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d{2,5}',
     "hardcoded_ip_port", "medium", "network",
     "hardcoded IP address with port"),
    (r'0\.0\.0\.0:\d+|INADDR_ANY',
     "bind_all_interfaces", "high", "network",
     "binds to all network interfaces"),
    (r'/bin/(ba)?sh\s+-i\s+.*>/dev/tcp/',
     "bash_reverse_shell", "critical", "network",
     "bash interactive reverse shell via /dev/tcp"),
    (r'python[23]?\s+-c\s+["\']import\s+socket',
     "python_socket_oneliner", "critical", "network",
     "Python one-liner socket connection (likely reverse shell)"),
    (r'socket\.connect\s*\(\s*\(',
     "python_socket_connect", "high", "network",
     "Python socket connect to arbitrary host"),
    (r'webhook\.site|requestbin\.com|pipedream\.net|hookbin\.com',
     "exfil_service", "high", "network",
     "references known data exfiltration/webhook testing service"),
    (r'pastebin\.com|hastebin\.com|ghostbin\.',
     "paste_service", "medium", "network",
     "references paste service (possible data staging)"),

    # ── Obfuscation: encoding and eval ──
    (r'base64\s+(-d|--decode)\s*\|',
     "base64_decode_pipe", "high", "obfuscation",
     "base64 decodes and pipes to execution"),
    (r'\\x[0-9a-fA-F]{2}.*\\x[0-9a-fA-F]{2}.*\\x[0-9a-fA-F]{2}',
     "hex_encoded_string", "medium", "obfuscation",
     "hex-encoded string (possible obfuscation)"),
    (r'\beval\s*\(\s*["\']',
     "eval_string", "high", "obfuscation",
     "eval() with string argument"),
    (r'\bexec\s*\(\s*["\']',
     "exec_string", "high", "obfuscation",
     "exec() with string argument"),
    (r'echo\s+[^\n]*\|\s*(bash|sh|python|perl|ruby|node)',
     "echo_pipe_exec", "critical", "obfuscation",
     "echo piped to interpreter for execution"),
    (r'compile\s*\(\s*[^\)]+,\s*["\'].*["\']\s*,\s*["\']exec["\']\s*\)',
     "python_compile_exec", "high", "obfuscation",
     "Python compile() with exec mode"),
    (r'getattr\s*\(\s*__builtins__',
     "python_getattr_builtins", "high", "obfuscation",
     "dynamic access to Python builtins (evasion technique)"),
    (r'__import__\s*\(\s*["\']os["\']\s*\)',
     "python_import_os", "high", "obfuscation",
     "dynamic import of os module"),
    (r'codecs\.decode\s*\(\s*["\']',
     "python_codecs_decode", "medium", "obfuscation",
     "codecs.decode (possible ROT13 or encoding obfuscation)"),
    (r'String\.fromCharCode|charCodeAt',
     "js_char_code", "medium", "obfuscation",
     "JavaScript character code construction (possible obfuscation)"),
    (r'atob\s*\(|btoa\s*\(',
     "js_base64", "medium", "obfuscation",
     "JavaScript base64 encode/decode"),
    (r'\[::-1\]',
     "string_reversal", "low", "obfuscation",
     "string reversal (possible obfuscated payload)"),
    (r'chr\s*\(\s*\d+\s*\)\s*\+\s*chr\s*\(\s*\d+',
     "chr_building", "high", "obfuscation",
     "building string from chr() calls (obfuscation)"),
    (r'\\u[0-9a-fA-F]{4}.*\\u[0-9a-fA-F]{4}.*\\u[0-9a-fA-F]{4}',
     "unicode_escape_chain", "medium", "obfuscation",
     "chain of unicode escapes (possible obfuscation)"),

    # ── Process execution in scripts ──
    (r'subprocess\.(run|call|Popen|check_output)\s*\(',
     "python_subprocess", "medium", "execution",
     "Python subprocess execution"),
    (r'os\.system\s*\(',
     "python_os_system", "high", "execution",
     "os.system() — unguarded shell execution"),
    (r'os\.popen\s*\(',
     "python_os_popen", "high", "execution",
     "os.popen() — shell pipe execution"),
    (r'child_process\.(exec|spawn|fork)\s*\(',
     "node_child_process", "high", "execution",
     "Node.js child_process execution"),
    (r'Runtime\.getRuntime\(\)\.exec\(',
     "java_runtime_exec", "high", "execution",
     "Java Runtime.exec() — shell execution"),
    (r'`[^`]*\$\([^)]+\)[^`]*`',
     "backtick_subshell", "medium", "execution",
     "backtick string with command substitution"),

    # ── Path traversal ──
    (r'\.\./\.\./\.\.',
     "path_traversal_deep", "high", "traversal",
     "deep relative path traversal (3+ levels up)"),
    (r'\.\./\.\.',
     "path_traversal", "medium", "traversal",
     "relative path traversal (2+ levels up)"),
    (r'/etc/passwd|/etc/shadow',
     "system_passwd_access", "critical", "traversal",
     "references system password files"),
    (r'/proc/self|/proc/\d+/',
     "proc_access", "high", "traversal",
     "references /proc filesystem (process introspection)"),
    (r'/dev/shm/',
     "dev_shm", "medium", "traversal",
     "references shared memory (common staging area)"),

    # ── Crypto mining ──
    (r'xmrig|stratum\+tcp|monero|coinhive|cryptonight',
     "crypto_mining", "critical", "mining",
     "cryptocurrency mining reference"),
    (r'hashrate|nonce.*difficulty',
     "mining_indicators", "medium", "mining",
     "possible cryptocurrency mining indicators"),

    # ── Supply chain: curl/wget pipe to shell ──
    (r'curl\s+[^\n]*\|\s*(ba)?sh',
     "curl_pipe_shell", "critical", "supply_chain",
     "curl piped to shell (download-and-execute)"),
    (r'wget\s+[^\n]*-O\s*-\s*\|\s*(ba)?sh',
     "wget_pipe_shell", "critical", "supply_chain",
     "wget piped to shell (download-and-execute)"),
    (r'curl\s+[^\n]*\|\s*python',
     "curl_pipe_python", "critical", "supply_chain",
     "curl piped to Python interpreter"),

    # ── Supply chain: unpinned/deferred dependencies ──
    (r'#\s*///\s*script.*dependencies',
     "pep723_inline_deps", "medium", "supply_chain",
     "PEP 723 inline script metadata with dependencies (verify pinning)"),
    (r'pip\s+install\s+(?!-r\s)(?!.*==)',
     "unpinned_pip_install", "medium", "supply_chain",
     "pip install without version pinning"),
    (r'npm\s+install\s+(?!.*@\d)',
     "unpinned_npm_install", "medium", "supply_chain",
     "npm install without version pinning"),
    (r'uv\s+run\s+',
     "uv_run", "medium", "supply_chain",
     "uv run (may auto-install unpinned dependencies)"),

    # ── Supply chain: remote resource fetching ──
    (r'(curl|wget|httpx?\.get|requests\.get|fetch)\s*[\(]?\s*["\']https?://',
     "remote_fetch", "medium", "supply_chain",
     "fetches remote resource at runtime"),
    (r'git\s+clone\s+',
     "git_clone", "medium", "supply_chain",
     "clones a git repository at runtime"),
    (r'docker\s+pull\s+',
     "docker_pull", "medium", "supply_chain",
     "pulls a Docker image at runtime"),

    # ── Privilege escalation ──
    # `allowed-tools:` is REQUIRED SKILL.md frontmatter per the agent-skill
    # spec — every compliant skill declares it, so it cannot be a threat
    # signal on its own. Keep it as an informational (low) finding for
    # auditability; it no longer drives the verdict.
    (r'^allowed-tools\s*:',
     "allowed_tools_field", "low", "privilege_escalation",
     "skill declares allowed-tools (standard frontmatter; informational)"),
    (r'\bsudo\b',
     "sudo_usage", "high", "privilege_escalation",
     "uses sudo (privilege escalation)"),
    (r'setuid|setgid|cap_setuid',
     "setuid_setgid", "critical", "privilege_escalation",
     "setuid/setgid (privilege escalation mechanism)"),
    (r'NOPASSWD',
     "nopasswd_sudo", "critical", "privilege_escalation",
     "NOPASSWD sudoers entry (passwordless privilege escalation)"),
    (r'chmod\s+[u+]?s',
     "suid_bit", "critical", "privilege_escalation",
     "sets SUID/SGID bit on a file"),

    # ── Agent config persistence ──
    (r'AGENTS\.md|CLAUDE\.md|\.cursorrules|\.clinerules',
     "agent_config_mod", "critical", "persistence",
     "references agent config files (could persist malicious instructions across sessions)"),
    (r'\.hermes/config\.yaml|\.hermes/SOUL\.md',
     "hermes_config_mod", "critical", "persistence",
     "references Hermes configuration files directly"),
    (r'\.claude/settings|\.codex/config',
     "other_agent_config", "high", "persistence",
     "references other agent configuration files"),

    # ── Hardcoded secrets (credentials embedded in the skill itself) ──
    (r'(?:api[_-]?key|token|secret|password)\s*[=:]\s*["\'][A-Za-z0-9+/=_-]{20,}',
     "hardcoded_secret", "critical", "credential_exposure",
     "possible hardcoded API key, token, or secret"),
    (r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----',
     "embedded_private_key", "critical", "credential_exposure",
     "embedded private key"),
    (r'ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{80,}',
     "github_token_leaked", "critical", "credential_exposure",
     "GitHub personal access token in skill content"),
    (r'sk-[A-Za-z0-9]{20,}',
     "openai_key_leaked", "critical", "credential_exposure",
     "possible OpenAI API key in skill content"),
    (r'sk-ant-[A-Za-z0-9_-]{90,}',
     "anthropic_key_leaked", "critical", "credential_exposure",
     "possible Anthropic API key in skill content"),
    (r'AKIA[0-9A-Z]{16}',
     "aws_access_key_leaked", "critical", "credential_exposure",
     "AWS access key ID in skill content"),

    # ── Additional prompt injection: jailbreak patterns ──
    (r'\bDAN\s+mode\b|Do\s+Anything\s+Now',
     "jailbreak_dan", "critical", "injection",
     "DAN (Do Anything Now) jailbreak attempt"),
    (r'\bdeveloper\s+mode\b.*\benabled?\b',
     "jailbreak_dev_mode", "critical", "injection",
     "developer mode jailbreak attempt"),
    (r'hypothetical\s+scenario.*(?:ignore|bypass|override)',
     "hypothetical_bypass", "high", "injection",
     "hypothetical scenario used to bypass restrictions"),
    (r'for\s+educational\s+purposes?\s+only',
     "educational_pretext", "medium", "injection",
     "educational pretext often used to justify harmful content"),
    (r'(respond|answer|reply)\s+without\s+(?:\w+\s+)*(restrictions|limitations|filters|safety)',
     "remove_filters", "critical", "injection",
     "instructs agent to respond without safety filters"),
    (r'you\s+have\s+been\s+(?:\w+\s+)*(updated|upgraded|patched)\s+to',
     "fake_update", "high", "injection",
     "fake update/patch announcement (social engineering)"),
    (r'new\s+(?:\w+\s+)*policy|updated\s+(?:\w+\s+)*guidelines|revised\s+(?:\w+\s+)*instructions',
     "fake_policy", "medium", "injection",
     "claims new policy/guidelines (may be social engineering)"),

    # ── Context window exfiltration ──
    (r'(include|output|print|send|share)\s+(?:\w+\s+)*(conversation|chat\s+history|previous\s+messages|context)',
     "context_exfil", "high", "exfiltration",
     "instructs agent to output/share conversation history"),
    (r'(send|post|upload|transmit)\s+.*\s+(to|at)\s+https?://',
     "send_to_url", "high", "exfiltration",
     "instructs agent to send data to a URL"),
]

# Structural limits for skill directories
MAX_FILE_COUNT = 50       # skills shouldn't have 50+ files
MAX_TOTAL_SIZE_KB = 1024  # 1MB total is suspicious for a skill
MAX_SINGLE_FILE_KB = 256  # individual file > 256KB is suspicious

# File extensions to scan (text files only — skip binary)
SCANNABLE_EXTENSIONS = {
    '.md', '.txt', '.py', '.sh', '.bash', '.js', '.ts', '.rb',
    '.yaml', '.yml', '.json', '.toml', '.cfg', '.ini', '.conf',
    '.html', '.css', '.xml', '.tex', '.r', '.jl', '.pl', '.php',
}

# Known binary extensions that should NOT be in a skill
SUSPICIOUS_BINARY_EXTENSIONS = {
    '.exe', '.dll', '.so', '.dylib', '.bin', '.dat', '.com',
    '.msi', '.dmg', '.app', '.deb', '.rpm',
}

# Zero-width and invisible unicode characters used for injection
INVISIBLE_CHARS = {
    '\u200b',  # zero-width space
    '\u200c',  # zero-width non-joiner
    '\u200d',  # zero-width joiner
    '\u2060',  # word joiner
    '\u2062',  # invisible times
    '\u2063',  # invisible separator
    '\u2064',  # invisible plus
    '\ufeff',  # zero-width no-break space (BOM)
    '\u202a',  # left-to-right embedding
    '\u202b',  # right-to-left embedding
    '\u202c',  # pop directional formatting
    '\u202d',  # left-to-right override
    '\u202e',  # right-to-left override
    '\u2066',  # left-to-right isolate
    '\u2067',  # right-to-left isolate
    '\u2068',  # first strong isolate
    '\u2069',  # pop directional isolate
}


# ---------------------------------------------------------------------------
# Scanning functions
# ---------------------------------------------------------------------------

def scan_file(file_path: Path, rel_path: str = "") -> List[Finding]:
    """
    Scan a single file for threat patterns and invisible unicode characters.

    Args:
        file_path: Absolute path to the file
        rel_path: Relative path for display (defaults to file_path.name)

    Returns:
        List of findings (deduplicated per pattern per line)
    """
    if not rel_path:
        rel_path = file_path.name

    if file_path.suffix.lower() not in SCANNABLE_EXTENSIONS and file_path.name != "SKILL.md":
        return []

    try:
        content = file_path.read_text(encoding='utf-8')
    except (UnicodeDecodeError, OSError):
        return []

    findings = []
    lines = content.split('\n')
    seen = set()  # (pattern_id, line_number) for deduplication

    # Regex pattern matching
    for pattern, pid, severity, category, description in THREAT_PATTERNS:
        for i, line in enumerate(lines, start=1):
            if (pid, i) in seen:
                continue
            if re.search(pattern, line, re.IGNORECASE):
                seen.add((pid, i))
                matched_text = line.strip()
                if len(matched_text) > 120:
                    matched_text = matched_text[:117] + "..."
                findings.append(Finding(
                    pattern_id=pid,
                    severity=severity,
                    category=category,
                    file=rel_path,
                    line=i,
                    match=matched_text,
                    description=description,
                ))

    # Invisible unicode character detection
    for i, line in enumerate(lines, start=1):
        for char in INVISIBLE_CHARS:
            if char in line:
                char_name = _unicode_char_name(char)
                findings.append(Finding(
                    pattern_id="invisible_unicode",
                    severity="high",
                    category="injection",
                    file=rel_path,
                    line=i,
                    match=f"U+{ord(char):04X} ({char_name})",
                    description=f"invisible unicode character {char_name} (possible text hiding/injection)",
                ))
                break  # one finding per line for invisible chars

    return findings


def scan_skill(skill_path: Path, source: str = "community") -> ScanResult:
    """
    Scan all files in a skill directory for security threats.

    Performs:
    1. Structural checks (file count, total size, binary files, symlinks)
    2. Regex pattern matching on all text files
    3. Invisible unicode character detection

    A skill may ship a `.skillignore` (or `.clawhubignore`) file with
    gitignore-style patterns. Matching paths are excluded from BOTH the
    structural checks and the pattern scan, so development/docs artifacts
    that are not part of the installed skill (e.g. `SKILL-original.md`,
    `docs/plans/`, `release-notes.md`) don't trip findings. The ignore
    file itself is always excluded. Patterns cannot un-ignore the
    skill's own `SKILL.md`, which is always scanned.

    Args:
        skill_path: Path to the skill directory (must contain SKILL.md)
        source: Source identifier for trust level resolution (e.g. "openai/skills")

    Returns:
        ScanResult with verdict, findings, and trust metadata
    """
    skill_name = skill_path.name
    trust_level = _resolve_trust_level(source)

    all_findings: List[Finding] = []

    if skill_path.is_dir():
        ignore = _load_skill_ignore(skill_path)

        # Structural checks first (honoring the ignore list)
        all_findings.extend(_check_structure(skill_path, ignore=ignore))

        # Pattern scanning on each file
        for f in skill_path.rglob("*"):
            if f.is_file():
                rel = str(f.relative_to(skill_path))
                if ignore(rel):
                    continue
                all_findings.extend(scan_file(f, rel))
    elif skill_path.is_file():
        all_findings.extend(scan_file(skill_path, skill_path.name))

    verdict = _determine_verdict(all_findings)
    summary = _build_summary(skill_name, source, trust_level, verdict, all_findings)

    return ScanResult(
        skill_name=skill_name,
        source=source,
        trust_level=trust_level,
        verdict=verdict,
        findings=all_findings,
        scanned_at=datetime.now(timezone.utc).isoformat(),
        summary=summary,
    )


def _content_digest(skill_path: Path) -> str:
    """Canonical SHA-256 over relative paths and exact file bytes."""
    h = hashlib.sha256()
    if skill_path.is_dir():
        for file_path in sorted(skill_path.rglob("*")):
            if file_path.is_file():
                rel = file_path.relative_to(skill_path).as_posix()
                h.update(rel.encode("utf-8") + b"\x00")
                h.update(file_path.read_bytes())
    else:
        h.update(skill_path.read_bytes())
    return h.hexdigest()


def full_content_hash(skill_path: Path) -> str:
    """Full canonical digest used to bind scanner attestations."""
    return f"sha256:{_content_digest(skill_path)}"


def _finding_dict(finding: Finding) -> dict:
    return {key: getattr(finding, key) for key in (
        "pattern_id", "severity", "category", "file", "line", "match", "description"
    )}


def scan_skill_cached(
    skill_path: Path,
    source: str = "community",
    *,
    source_url: str = "",
    cache_dir: Path | None = None,
) -> Tuple[ScanResult, dict]:
    """Return a scan plus attestation, caching only exact current content."""
    bundle_hash = full_content_hash(skill_path)
    cache_root = cache_dir or skill_path.parent / ".scan-cache"
    source_identity = hashlib.sha256(f"{source}\0{source_url}".encode("utf-8")).hexdigest()[:16]
    cache_file = cache_root / f"{bundle_hash.split(':', 1)[1]}-{source_identity}.json"
    try:
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cached = None
    if (isinstance(cached, dict)
            and cached.get("bundle_hash") == bundle_hash
            and cached.get("scanner_version") == SCANNER_VERSION
            and cached.get("source") == source
            and cached.get("source_url") == source_url):
        result = ScanResult(
            skill_name=skill_path.name, source=source,
            trust_level=cached["trust_level"], verdict=cached["verdict"],
            findings=[Finding(**item) for item in cached.get("findings", [])],
            scanned_at=cached["scanned_at"], summary=cached.get("summary", ""),
        )
        provenance = dict(cached)
        provenance["fresh"] = False
        result.scan_provenance = provenance
        return result, provenance

    result = scan_skill(skill_path, source=source)
    findings = [_finding_dict(item) for item in result.findings]
    provenance = {
        "source": source, "source_url": source_url, "bundle_hash": bundle_hash,
        "scanner_version": SCANNER_VERSION, "verdict": result.verdict,
        "trust_level": result.trust_level, "findings": findings,
        "rules": sorted({item["pattern_id"] for item in findings}),
        "scanned_at": result.scanned_at, "summary": result.summary, "fresh": True,
    }
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass
    result.scan_provenance = provenance
    return result, provenance


def should_allow_install(result: ScanResult, force: bool = False) -> Tuple[bool, str]:
    """
    Determine whether a skill should be installed based on scan result and trust.

    Args:
        result: Scan result from scan_skill()
        force: If True, override blocked policy decisions for this scan result

    Returns:
        (allowed, reason) tuple
    """
    policy = INSTALL_POLICY.get(result.trust_level, INSTALL_POLICY["community"])
    vi = VERDICT_INDEX.get(result.verdict, 2)
    decision = policy[vi]

    if decision == "allow":
        return True, f"Allowed ({result.trust_level} source, {result.verdict} verdict)"

    if force and not (result.verdict == "dangerous" and result.trust_level in ("community", "trusted")):
        return True, (
            f"Force-installed despite {result.verdict} verdict "
            f"({len(result.findings)} findings)"
        )

    if decision == "ask":
        # Return None to signal "needs user confirmation"
        return None, (
            f"Requires confirmation ({result.trust_level} source + {result.verdict} verdict, "
            f"{len(result.findings)} findings)"
        )

    # Dangerous verdicts cannot be overridden by --force (community/trusted);
    # other blocks can.
    if result.verdict == "dangerous" and result.trust_level in ("community", "trusted"):
        return False, (
            f"Blocked ({result.trust_level} source + dangerous verdict, "
            f"{len(result.findings)} findings). --force does not override a dangerous verdict."
        )
    return False, (
        f"Blocked ({result.trust_level} source + {result.verdict} verdict, "
        f"{len(result.findings)} findings). Use --force to override."
    )


def format_scan_report(result: ScanResult) -> str:
    """
    Format a scan result as a human-readable report string.

    Returns a compact multi-line report suitable for CLI or chat display.
    """
    lines = []

    verdict_display = result.verdict.upper()
    lines.append(f"Scan: {result.skill_name} ({result.source}/{result.trust_level})  Verdict: {verdict_display}")

    if result.findings:
        # Group and sort: critical first, then high, medium, low
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_findings = sorted(result.findings, key=lambda f: severity_order.get(f.severity, 4))

        for f in sorted_findings:
            sev = f.severity.upper().ljust(8)
            cat = f.category.ljust(14)
            loc = f"{f.file}:{f.line}".ljust(30)
            lines.append(f"  {sev} {cat} {loc} \"{f.match[:60]}\"")

        lines.append("")

    allowed, reason = should_allow_install(result)
    if allowed is True:
        status = "ALLOWED"
    elif allowed is None:
        status = "NEEDS CONFIRMATION"
    else:
        status = "BLOCKED"
    lines.append(f"Decision: {status} — {reason}")

    return "\n".join(lines)


def content_hash(skill_path: Path) -> str:
    """Compute a SHA-256 hash of all files in a skill directory for integrity tracking.

    File paths (relative to ``skill_path``) are mixed into the hash alongside
    file contents so that swapping the contents of two files in a skill
    changes the hash. This must stay symmetric with
    ``tools.skills_hub.bundle_content_hash`` — both functions need to
    produce the same digest for the same skill (one operates on disk,
    one on an in-memory bundle), so any change to the hash shape MUST
    land in both places at once.
    """
    return f"sha256:{_content_digest(skill_path)[:16]}"


# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------

def _check_structure(skill_dir: Path, ignore=None) -> List[Finding]:
    """
    Check the skill directory for structural anomalies:
    - Too many files
    - Suspiciously large total size
    - Binary/executable files that shouldn't be in a skill
    - Symlinks pointing outside the skill directory
    - Individual files that are too large

    Args:
        skill_dir: Path to the skill directory.
        ignore: Optional callable taking a relative posix path and returning
            True if the path should be excluded (e.g. from `.skillignore`).
            Ignored files are not counted toward the file count, total size,
            or any structural finding.
    """
    if ignore is None:
        ignore = lambda _rel: False  # noqa: E731

    findings = []
    file_count = 0
    total_size = 0

    for f in skill_dir.rglob("*"):
        if not f.is_file() and not f.is_symlink():
            continue

        rel = str(f.relative_to(skill_dir))
        if ignore(rel):
            continue
        file_count += 1

        # Symlink check — must resolve within the skill directory
        if f.is_symlink():
            try:
                resolved = f.resolve()
                if not resolved.is_relative_to(skill_dir.resolve()):
                    findings.append(Finding(
                        pattern_id="symlink_escape",
                        severity="critical",
                        category="traversal",
                        file=rel,
                        line=0,
                        match=f"symlink -> {resolved}",
                        description="symlink points outside the skill directory",
                    ))
            except OSError:
                findings.append(Finding(
                    pattern_id="broken_symlink",
                    severity="medium",
                    category="traversal",
                    file=rel,
                    line=0,
                    match="broken symlink",
                    description="broken or circular symlink",
                ))
            continue

        # Size tracking
        try:
            size = f.stat().st_size
            total_size += size
        except OSError:
            continue

        # Single file too large
        if size > MAX_SINGLE_FILE_KB * 1024:
            findings.append(Finding(
                pattern_id="oversized_file",
                severity="medium",
                category="structural",
                file=rel,
                line=0,
                match=f"{size // 1024}KB",
                description=f"file is {size // 1024}KB (limit: {MAX_SINGLE_FILE_KB}KB)",
            ))

        # Binary/executable files
        ext = f.suffix.lower()
        if ext in SUSPICIOUS_BINARY_EXTENSIONS:
            findings.append(Finding(
                pattern_id="binary_file",
                severity="critical",
                category="structural",
                file=rel,
                line=0,
                match=f"binary: {ext}",
                description=f"binary/executable file ({ext}) should not be in a skill",
            ))

        script_exts = {'.sh', '.bash', '.py', '.rb', '.pl'}
        is_executable = bool(f.stat().st_mode & 0o111)

        # Executable permission on non-script files
        if ext not in script_exts and is_executable:
            findings.append(Finding(
                pattern_id="unexpected_executable",
                severity="medium",
                category="structural",
                file=rel,
                line=0,
                match="executable bit set",
                description="file has executable permission but is not a recognized script type",
            ))

        # A shebang advertises direct execution. On POSIX, flag scripts whose
        # mode contradicts that contract; interpreter-only files should omit
        # the shebang or be documented as such.
        if os.name != "nt" and ext in script_exts and not is_executable:
            try:
                with f.open("rb") as handle:
                    has_shebang = handle.read(2) == b"#!"
            except OSError:
                has_shebang = False
            if has_shebang:
                findings.append(Finding(
                    pattern_id="script_not_executable",
                    severity="low",
                    category="structural",
                    file=rel,
                    line=1,
                    match="shebang without executable bit",
                    description=(
                        "script declares an interpreter but is not executable; "
                        "set an executable mode or remove the shebang"
                    ),
                ))

    # File count limit
    if file_count > MAX_FILE_COUNT:
        findings.append(Finding(
            pattern_id="too_many_files",
            severity="medium",
            category="structural",
            file="(directory)",
            line=0,
            match=f"{file_count} files",
            description=f"skill has {file_count} files (limit: {MAX_FILE_COUNT})",
        ))

    # Total size limit
    if total_size > MAX_TOTAL_SIZE_KB * 1024:
        findings.append(Finding(
            pattern_id="oversized_skill",
            severity="high",
            category="structural",
            file="(directory)",
            line=0,
            match=f"{total_size // 1024}KB total",
            description=f"skill is {total_size // 1024}KB total (limit: {MAX_TOTAL_SIZE_KB}KB)",
        ))

    return findings


def _unicode_char_name(char: str) -> str:
    """Get a readable name for an invisible unicode character."""
    names = {
        '\u200b': "zero-width space",
        '\u200c': "zero-width non-joiner",
        '\u200d': "zero-width joiner",
        '\u2060': "word joiner",
        '\u2062': "invisible times",
        '\u2063': "invisible separator",
        '\u2064': "invisible plus",
        '\ufeff': "BOM/zero-width no-break space",
        '\u202a': "LTR embedding",
        '\u202b': "RTL embedding",
        '\u202c': "pop directional",
        '\u202d': "LTR override",
        '\u202e': "RTL override",
        '\u2066': "LTR isolate",
        '\u2067': "RTL isolate",
        '\u2068': "first strong isolate",
        '\u2069': "pop directional isolate",
    }
    return names.get(char, f"U+{ord(char):04X}")



# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Ignore-file names a skill may ship to exclude dev/docs artifacts from the
# scan. `.skillignore` is the Hermes-native name; `.clawhubignore` is honored
# for compatibility with skills published through ClawHub.
_SKILL_IGNORE_FILENAMES = (".skillignore", ".clawhubignore")

# Paths that are NEVER scanned regardless of ignore patterns, and SKILL.md
# which can never be un-scanned via the ignore file.
_ALWAYS_IGNORED_NAMES = set(_SKILL_IGNORE_FILENAMES)
_NEVER_IGNORABLE = {"SKILL.md"}


def _load_skill_ignore(skill_dir: Path):
    """Build a matcher from a skill's `.skillignore` / `.clawhubignore`.

    Returns a callable ``ignore(rel_posix_path) -> bool``. The matcher
    supports gitignore-style basics: blank lines and ``#`` comments are
    skipped, a trailing ``/`` marks a directory (matches that dir and
    everything under it), and ``*``/``?`` globs are honored via fnmatch on
    both the full relative path and each path segment. A leading ``/``
    anchors a pattern to the skill root. The ignore files themselves are
    always excluded; ``SKILL.md`` can never be excluded.
    """
    patterns: List[str] = []
    for name in _SKILL_IGNORE_FILENAMES:
        ig = skill_dir / name
        try:
            if ig.is_file():
                for raw in ig.read_text(encoding="utf-8").splitlines():
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    patterns.append(line)
        except (UnicodeDecodeError, OSError):
            continue

    def ignore(rel: str) -> bool:
        rel_posix = Path(rel).as_posix()
        base = rel_posix.split("/")[-1]

        if base in _NEVER_IGNORABLE:
            return False
        if base in _ALWAYS_IGNORED_NAMES:
            return True

        for pat in patterns:
            anchored = pat.startswith("/")
            p = pat.lstrip("/")
            is_dir = p.endswith("/")
            p = p.rstrip("/")
            if not p:
                continue

            if is_dir:
                # Directory pattern: match the dir itself or anything under it.
                if rel_posix == p or rel_posix.startswith(p + "/"):
                    return True
                if not anchored and ("/" + rel_posix + "/").find("/" + p + "/") != -1:
                    return True
                continue

            # File/glob pattern.
            if fnmatch.fnmatch(rel_posix, p):
                return True
            if not anchored:
                # Unanchored: also match the basename and any path segment.
                if fnmatch.fnmatch(base, p):
                    return True
                if "/" not in p and any(
                    fnmatch.fnmatch(seg, p) for seg in rel_posix.split("/")
                ):
                    return True
                # Match a prefix directory component (e.g. `docs` ignores
                # `docs/plans/x.md`).
                if rel_posix.startswith(p + "/"):
                    return True
        return False

    return ignore


def _resolve_trust_level(source: str) -> str:
    """Map a source identifier to a trust level."""
    prefix_aliases = (
        "skills-sh/",
        "skills.sh/",
        "skils-sh/",
        "skils.sh/",
    )
    normalized_source = source
    for prefix in prefix_aliases:
        if normalized_source.startswith(prefix):
            normalized_source = normalized_source[len(prefix):]
            break

    # Agent-created skills get their own permissive trust level
    if normalized_source == "agent-created":
        return "agent-created"
    # Official optional skills must be identified by source provenance, not by
    # user-controlled GitHub identifiers such as "official/<repo>".
    if normalized_source == "official":
        return "builtin"
    # Check if source matches any trusted repo exactly, or a skill path inside
    # that repo. Do not trust sibling repositories that merely share a prefix.
    for trusted in TRUSTED_REPOS:
        if normalized_source == trusted or normalized_source.startswith(f"{trusted}/"):
            return "trusted"
    return "community"


def _determine_verdict(findings: List[Finding]) -> str:
    """Determine the overall verdict from a list of findings."""
    if not findings:
        return "safe"

    has_critical = any(f.severity == "critical" for f in findings)
    has_high = any(f.severity == "high" for f in findings)

    if has_critical:
        return "dangerous"
    if has_high:
        return "caution"
    # medium/low findings alone are informational, not blocking
    return "safe"


def _build_summary(name: str, source: str, trust: str, verdict: str, findings: List[Finding]) -> str:
    """Build a one-line summary of the scan result."""
    if not findings:
        return f"{name}: clean scan, no threats detected"

    categories = {f.category for f in findings}
    return f"{name}: {verdict} — {len(findings)} finding(s) in {', '.join(sorted(categories))}"
