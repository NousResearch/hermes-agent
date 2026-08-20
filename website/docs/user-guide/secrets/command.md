# Command Helper Secret Source

Resolve credentials by running your own helper command at startup — any secret store with a CLI works: `keepassxc-cli`, `secret-tool` (GNOME Keyring), `pass`, `gpg`, Vaultwarden's CLI, or a script that cats a tmpfs env file. The helper prints `KEY=VALUE` lines on stdout; Hermes applies them through the same orchestrator as [Bitwarden](./bitwarden) and [1Password](./onepassword), so you can enable any combination of sources simultaneously.

## How it works

1. You configure a helper command in `config.yaml` (never in `.env` — the command is configuration, `.env` holds values).
2. At startup, after `.env` loads, Hermes runs the helper ONCE via `/bin/sh -c` and parses its stdout as a dotenv blob.
3. The parsed keys flow through the standard precedence ladder: `.env`/shell win unless `override_existing: true`; mapped sources beat this bulk source on contested vars; first claim wins.

```yaml
secrets:
  command:
    enabled: true
    command: "cat /run/user/1000/hermes-secrets.env"
    # or any vault CLI that dumps KEY=VALUE lines:
    # command: "pass show hermes/env"
    # command: "secret-tool lookup service hermes-env"
```

## Config

| Key | Default | What it does |
|---|---|---|
| `enabled` | `false` | Master switch. |
| `command` | `""` | Helper run via `/bin/sh -c`; must print `KEY=VALUE` lines on stdout. |
| `helper_timeout_seconds` | `3` | Hard timeout for one helper run. Deliberately tight — the helper must be fast and NON-interactive (no unlock prompts, no touch/PIN). |
| `override_existing` | `false` | Helper values overwrite `.env`/shell values. Off by default (unlike Bitwarden/1Password) since a local helper is not a central rotation authority. |

## Security model

- The helper command string is YOUR configuration — same trust level as the `.env` file you control.
- Output is hard-capped at 1 MiB; a runaway helper can't wedge startup (process group killed on timeout).
- The helper's **stderr is discarded** — vault CLI diagnostics can carry secret material, so they never reach Hermes' output. Failures log structured fields only (exit code / signal / errno), never the command string.
- Whitespace-only values are treated as "no value" — a placeholder entry never flows into an Authorization header.
- POSIX-only (needs `/bin/sh`). On Windows the source reports itself unconfigured and startup continues.

## Recipe: macOS Keychain

macOS ships `security`, so a Mac needs no extra vault software. Store each
credential as a generic-password item and have the helper print them:

```yaml
secrets:
  command:
    enabled: true
    command: "security find-generic-password -s hermes-env -w"
```

That works when one item holds a whole dotenv blob. To store credentials
individually, keep a small helper script and point `command` at it:

```bash
#!/bin/sh
# ~/.hermes/scripts/keychain-env.sh — one Keychain item per credential.
for key in ANTHROPIC_API_KEY OPENAI_API_KEY; do
  value=$(security find-generic-password -a "$USER" -s "hermes-$key" -w 2>/dev/null)
  [ -n "$value" ] && printf '%s=%s\n' "$key" "$value"
done
```

Add items with `security add-generic-password`. The `-w` flag reads the value
from a prompt rather than `argv`, so it stays out of your shell history:

```bash
security add-generic-password -a "$USER" -s hermes-ANTHROPIC_API_KEY -w
```

On Apple Silicon the login Keychain is hardware-protected and unlocked by your
login password, so this gives you at-rest encryption with no daemon to run and
nothing to install.

:::warning The SSH session boundary
**Keychain reads and writes fail from an SSH session**, even as the same user
who is logged into the GUI. Keychain items are scoped to the aqua login session:

```console
$ ssh mac 'security add-generic-password -a u -s test -w secret'
security: SecKeychainItemCreateFromContent (<default>): User interaction is not allowed.
```

So a Hermes process started over SSH — or from a LaunchDaemon outside the user
session — gets nothing, while the identical command works in `Terminal.app`.

Note the two failures look different: a *blocked* read says `User interaction is
not allowed`, while a *missing* item says `The specified item could not be
found`. If you see the former over SSH, the item is fine and you are hitting the
session boundary.

If Hermes must run outside your GUI session, use a
[LaunchAgent](https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPSystemStartup/Chapters/CreatingLaunchdJobs.html)
(runs in the login session) rather than a LaunchDaemon, or materialize the
secrets to a `0600` file during login and `cat` that file instead.
:::

Timing is not a concern here: `security` resolves an item in well under
100 ms against an unlocked Keychain, comfortably inside the 3-second budget. A
*locked* Keychain is the problem — it raises a GUI unlock prompt, which the
helper cannot answer, so it hits the timeout and yields nothing.

## Failure modes

Startup is never blocked. Errors print one line plus a `→` remediation hint:

| Symptom | Cause | Fix |
|---|---|---|
| `secrets.command.command is empty` | Enabled without a command | Set `secrets.command.command` in config.yaml |
| `helper command failed` | Non-zero exit, timeout, spawn failure | Run the helper manually in a shell to see its real error (Hermes discards its stderr on purpose) |
| `helper output was not a KEY=VALUE map` | Helper printed a bare value or garbage | Make the helper emit dotenv-shaped lines |
| Helper works in `Terminal.app`, returns nothing under Hermes (macOS) | Keychain reads are blocked outside the GUI login session | See [the SSH session boundary](#recipe-macos-keychain) |

## When to use this vs a plugin

The command source is the escape hatch for vaults without a bundled integration. If you find yourself wrapping a complex CLI dance in a long script, consider a proper [secret-source plugin](/developer-guide/secret-source-plugin) instead — plugins get caching, provenance labels, and typed config.
