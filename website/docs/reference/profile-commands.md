---
sidebar_position: 7
---

# Profile Commands Reference

This page covers all commands related to [Hermes profiles](../user-guide/profiles.md). For general CLI commands, see [CLI Commands Reference](./cli-commands.md).

## `hermes profile`

```bash
hermes profile <subcommand>
```

Top-level command for managing profiles. Running `hermes profile` without a subcommand shows help.

| Subcommand | Description |
|------------|-------------|
| `list` | List all profiles. |
| `use` | Set the active (default) profile. |
| `create` | Create a new profile. |
| `delete` | Delete a profile. |
| `show` | Show details about a profile. |
| `alias` | Regenerate the shell alias for a profile. |
| `rename` | Rename a profile. |
| `export` | Export a profile to a tar.gz archive. |
| `import` | Import a profile from a tar.gz archive. |

## `hermes profile list`

```bash
hermes profile list
```

Lists all profiles. The currently active profile is marked with `*`.

**Example:**

```bash
$ hermes profile list
  default
* work
  dev
  personal
```

No options.

## `hermes profile use`

```bash
hermes profile use <name>
```

Sets `<name>` as the active profile. All subsequent `hermes` commands (without `-p`) will use this profile.

| Argument | Description |
|----------|-------------|
| `<name>` | Profile name to activate. Use `default` to return to the base profile. |

**Example:**

```bash
hermes profile use work
hermes profile use default
```

## `hermes profile create`

```bash
hermes profile create <name> [options]
```

Creates a new profile.

| Argument / Option | Description |
|-------------------|-------------|
| `<name>` | Name for the new profile. Must be a valid directory name (alphanumeric, hyphens, underscores). |
| `--clone` | Copy `config.yaml`, `.env`, and `SOUL.md` from the current profile. |
| `--clone-all` | Copy everything (config, memories, skills, sessions, state) from the current profile. |
| `--from <profile>` | Clone from a specific profile instead of the current one. Used with `--clone` or `--clone-all`. |

**Examples:**

```bash
# Blank profile — needs full setup
hermes profile create mybot

# Clone config only from current profile
hermes profile create work --clone

# Clone everything from current profile
hermes profile create backup --clone-all

# Clone config from a specific profile
hermes profile create work2 --clone --from work
```

## `hermes profile delete`

```bash
hermes profile delete <name> [options]
```

Deletes a profile and removes its shell alias.

| Argument / Option | Description |
|-------------------|-------------|
| `<name>` | Profile to delete. |
| `--yes`, `-y` | Skip confirmation prompt. |

**Example:**

```bash
hermes profile delete mybot
hermes profile delete mybot --yes
```

:::warning
This permanently deletes the profile's entire directory including all config, memories, sessions, and skills. Cannot delete the currently active profile.
:::

## `hermes profile show`

```bash
hermes profile show [name]
```

Displays details about a profile including its home directory, configured model, active platforms, and disk usage.

| Argument | Description |
|----------|-------------|
| `[name]` | Profile to inspect. Defaults to the current active profile if omitted. |

**Example:**

```bash
$ hermes profile show work
Profile:    work
Home:       ~/.hermes/profiles/work
Model:      anthropic/claude-sonnet-4
Platforms:  telegram, discord
Skills:     12 installed
Disk:       48 MB
```

## `hermes profile alias`

```bash
hermes profile alias <name>
```

Regenerates the shell alias script at `~/.local/bin/hermes-<name>`. Useful if the alias was accidentally deleted or if you need to update it after moving your Hermes installation.

| Argument | Description |
|----------|-------------|
| `<name>` | Profile to create/update the alias for. |

**Example:**

```bash
hermes profile alias work
# Creates/updates ~/.local/bin/hermes-work
```

## `hermes profile rename`

```bash
hermes profile rename <old-name> <new-name>
```

Renames a profile. Updates the directory and shell alias.

| Argument | Description |
|----------|-------------|
| `<old-name>` | Current profile name. |
| `<new-name>` | New profile name. |

**Example:**

```bash
hermes profile rename mybot assistant
# ~/.hermes/profiles/mybot → ~/.hermes/profiles/assistant
# ~/.local/bin/hermes-mybot → ~/.local/bin/hermes-assistant
```

## `hermes profile export`

```bash
hermes profile export <name> <output-path>
```

Exports a profile as a compressed tar.gz archive.

| Argument | Description |
|----------|-------------|
| `<name>` | Profile to export. |
| `<output-path>` | Path for the output archive (e.g., `./work-backup.tar.gz`). |

**Example:**

```bash
hermes profile export work ./work-2026-03-29.tar.gz
```

## `hermes profile import`

```bash
hermes profile import <archive-path> [name]
```

Imports a profile from a tar.gz archive.

| Argument | Description |
|----------|-------------|
| `<archive-path>` | Path to the tar.gz archive to import. |
| `[name]` | Name for the imported profile. Defaults to the original profile name from the archive. |

**Example:**

```bash
hermes profile import ./work-2026-03-29.tar.gz work-restored
```

## `hermes -p` / `hermes --profile`

```bash
hermes -p <name> <command> [options]
hermes --profile <name> <command> [options]
```

Global flag to run any Hermes command under a specific profile without changing the sticky default. This overrides the active profile for the duration of the command.

| Option | Description |
|--------|-------------|
| `-p <name>`, `--profile <name>` | Profile to use for this command. |

**Examples:**

```bash
hermes -p work chat -q "Check the server status"
hermes --profile dev gateway start
hermes -p personal skills list
hermes -p work config edit
```

## `hermes completion`

```bash
hermes completion <shell>
```

Generates shell completion scripts. Includes completions for profile names and profile subcommands.

| Argument | Description |
|----------|-------------|
| `<shell>` | Shell to generate completions for: `bash`, `zsh`, or `fish`. |

**Examples:**

```bash
# Install completions
hermes completion bash >> ~/.bashrc
hermes completion zsh >> ~/.zshrc
hermes completion fish > ~/.config/fish/completions/hermes.fish

# Reload shell
source ~/.bashrc
```

After installation, tab completion works for:
- `hermes profile <TAB>` — subcommands (list, use, create, etc.)
- `hermes profile use <TAB>` — profile names
- `hermes -p <TAB>` — profile names

## See also

- [Profiles User Guide](../user-guide/profiles.md)
- [CLI Commands Reference](./cli-commands.md)
- [FAQ — Profiles section](./faq.md#profiles)
