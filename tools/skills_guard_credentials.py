"""Skill-only credential rules of the skills scanner (``tools.skills_guard``).

A skill's scripts run in the execute_code or terminal child, which never receives a Hermes model-provider key
(the child-env scrub; GHSA-rhgp-j443-p4rf keeps a skill from declaring one back). Loading the Hermes .env defeats
that and hands the script every key of the profile, and reading a provider key means the skill depends on getting
it some other way. Both are high on a skill. Plugins run in-process and read their own keys, so ``plugin_guard``
scans without these rules. Leaf module: nothing here imports ``tools.skills_guard``."""

import re

# Case-sensitive, and ``.hermes`` must be a whole path segment, so ``.hermes-project.env``, ``myapp.hermes.env``
# or a ``hermes_home_env_path`` variable do not match. A call right after a negation ("Never call
# ``load_dotenv(HERMES_HOME/.env)``") is prose that forbids it; the negation voids only that call, not the line.
_NEGATED_CALL = "".join(f"(?<!{re.escape(neg + verb + tick)})" for neg in ("ever ", "o not ", "n't ", "void ")
                        for verb in ("", "call ", "use ", "run ") for tick in ("", "`"))
SKILL_ONLY_PATTERNS = [(re.compile(
    rf'\b(?=(?:load_dotenv|dotenv_values)\s*\(){_NEGATED_CALL}(?:load_dotenv|dotenv_values)\s*\([^\n]*'
    r'(?:HERMES_HOME|get_hermes_home|[~/\\"\']\.hermes(?:[/\\"\']|$))'),
    "hermes_env_dotenv", "high", "exfiltration", "loads the Hermes .env (every key of the profile) into the script")]
ENV_READ_IDS = frozenset({"python_getenv_secret", "python_environ_get_secret"})
# The name a getenv / environ.get / environ[...] call reads: its first argument only, so a provider name in a
# comment, a default argument or another literal on the line does not raise the read of the skill's own key.
_ENV_READ_NAME_RE = re.compile(r'(?:getenv|environ\s*\.\s*get|environ\s*\[)\s*\(?\s*["\']([A-Za-z_]\w*)["\']')


def model_provider_key_names() -> frozenset:
    """Env names that hold a Hermes model-provider key: every provider's ``api_key_env_vars`` plus the
    secret provider entries of ``OPTIONAL_ENV_VARS`` (``OPENROUTER_API_KEY`` is declared only there)."""
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.config import OPTIONAL_ENV_VARS
    names = {var.upper() for pconfig in PROVIDER_REGISTRY.values() for var in pconfig.api_key_env_vars}
    names.update(name.upper() for name, meta in OPTIONAL_ENV_VARS.items()
                 if meta.get("category") == "provider" and meta.get("password"))
    return frozenset(names)


def raise_provider_key_reads(findings: list, lines: list) -> None:
    """A secret read that names a model-provider key becomes ``provider_key_env_read`` (high); a read of
    any other secret-shaped name stays medium (reading your own API key is normal, #60709)."""
    reads = [f for f in findings if f.pattern_id in ENV_READ_IDS]
    if not reads:
        return
    provider_keys = model_provider_key_names()
    for f in reads:
        named = [n for n in _ENV_READ_NAME_RE.findall(lines[f.line - 1]) if n.upper() in provider_keys]
        if named:
            f.pattern_id, f.severity = "provider_key_env_read", "high"
            f.description = (f"reads Hermes model-provider key {named[0]}; skill scripts never receive it, "
                             "and the skill must not fetch it another way")
