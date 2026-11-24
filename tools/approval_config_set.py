"""`hermes config set|unset` approval-policy invocation pattern for :mod:`tools.approval_detection`."""

# Top-level `hermes` flags that consume a SEPARATE word (`--profile work`): the value-taking subset
# of `hermes_cli._parser.top_level_value_flag_sets`, so `hermes --profile work config set ...` does
# not read "work" as the subcommand. Any other flag is a bare token (`--json`, `--flag=value`).
_HERMES_TOP_LEVEL_VALUE_FLAGS = (
    r'(?:--profile|-p|--config|--model|-m|--provider|--reasoning|--toolsets|-t'
    r'|--resume|-r|--skills|-s|--oneshot|-z|--usage-file|--in|--cpu-profile)'
)

# `-p`/`--profile` is stripped from ANYWHERE before `--` (`_parser.scan_profile_flag`), so it can sit
# between `config` and `set` or before the key too. A `VAR=value` word may be quoted (`NOTE='a b'`).
_HERMES_PROFILE_OPTION = r'(?:(?:-p|--profile)[ \t]+[^\s]+|--profile=[^\s]+)'
_SHELL_WORD = r'''(?:[^\s'"]+|'[^']*'|"[^"]*")*'''

# `hermes config set|unset <key>` — the invocations that rewrite the agent's own approval
# policy. Anchored to the INVOCATION (program word, `config set|unset`, policy key), so a read of
# the text (`grep -r 'hermes config set' docs`) stays ungated. The engine never backtracks, so the
# option runs are plain repeats; a flag word that could also be read as a flag's value is read both ways.
_HERMES_CONFIG_SET_APPROVAL_POLICY = (
    # The program: the `hermes` binary (or the macOS `ai.hermes`), or the module form
    # `python -m hermes_cli.main` that the updater and the cron lifecycle guard invoke.
    # Optional leading `VAR=value` assignments (`HERMES_HOME=... hermes config set ...`).
    r'(?:\w+=' + _SHELL_WORD + r'[ \t]+)*'
    r'(?:(?:(?:[a-z]:)?[^\s:]*?/)?(?:ai\.)?hermes(?:\.exe)?\b'
    r'|(?:(?:[a-z]:)?[^\s:]*?/)?python[0-9.]*(?:\.exe)?\b[ \t]+(?:-[^\s]+[ \t]+)*?-m[ \t]*hermes_cli\.main\b)'
    r'(?:[ \t]+(?:' + _HERMES_TOP_LEVEL_VALUE_FLAGS + r'[ \t]+[^\s]+'
    r'|--?[\w-]+(?:=[^\s]+)?))*'  # global options before the subcommand
    r'[ \t]+config(?:[ \t]+' + _HERMES_PROFILE_OPTION + r')*'  # `-p` also parses after a subcommand
    r'[ \t]+(?:set|unset)\b[ \t]*'  # the `config set` / `config unset` subcommand
    r'(?:(?:' + _HERMES_PROFILE_OPTION + r'|--?[\w-]+)[ \t]+)*'  # `--force`, `-p work`
    r'["\']?(?:approvals(?:\.[\w.-]*\w)?|command_allowlist)\b'  # the approval-policy key, maybe quoted
)
