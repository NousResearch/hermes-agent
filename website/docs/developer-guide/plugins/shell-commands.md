---
title: Plugin terminal commands
description: Export opt-in package commands during managed plugin installation.
---

# Plugin terminal commands

Native Python plugins can opt in to bare terminal commands without asking users
to create or activate a virtual environment. Declare a packaged console script
and explicitly opt it in through `plugin.yaml`:

```toml
[build-system]
requires = ["setuptools>=75,<83"]
build-backend = "setuptools.build_meta"

[project]
name = "example-doctor"
version = "1.0.0"

[project.scripts]
example-doctor = "example_doctor.cli:main"
```

```yaml
name: example-doctor
shell_commands:
  - example-doctor
```

After accepting dependency preparation and enabling the plugin:

```sh
hermes plugins install example-doctor
example-doctor setup
example-doctor status
```

This is an installer feature, not an import/install hook supplied by the plugin.
The installer validates names and entrypoints without importing plugin code.
Only explicitly listed `project.scripts` entries are exported; scripts supplied
by dependencies are not exported. Shell snippets, path-like command names,
reserved interpreter/shell/Hermes names and duplicate declarations are refused.

## Runtime and profile ownership

The launcher re-enters the stable Hermes executable through
`hermes plugins run-shell PLUGIN COMMAND -- ARGUMENTS`, selecting the current
managed package entrypoint at launch time. It does not capture a dependency
venv generation. The launcher binds the profile used for installation, including
when the caller's ambient `HERMES_HOME` names another profile.

The runtime checks that this profile still enables the plugin and still declares
that command. It does not use the union of plugins enabled in other profiles.
Operator CLI dispatch does not load the agent plugin's `register()` entrypoint,
so `plugins.isolation: host` need not be weakened to run a terminal command.
The command has the permissions of the invoking OS user, just like a standalone
package CLI; this does not grant an agent a new model-callable tool.

One shell command name has one profile owner. Installing a second profile's
command with the same name does not silently change that owner.

## Files and lifecycle

POSIX launchers live in the existing user command directory (`~/.local/bin`);
Windows uses Hermes's user `bin` directory. The directory must already be on
PATH. No shell startup files, registry PATH entries or system command directories
are modified by plugin installation. Nonstandard/external installations that do
not expose that user command directory receive an explicit error.

Each command has an owner/digest sidecar in that directory. The installer never
overwrites a foreign command, follows a command symlink, or silently shadows an
existing PATH command. A modified previously-owned command is preserved rather
than overwritten or deleted. POSIX group-writable command directories are accepted
only when that group contains no other account; world-writable directories are
refused.

Enable/install publishes the declared commands after PM admits the dependencies.
Disable/uninstall removes only unmodified launchers belonging to that plugin and
profile. Updates reconcile additions/removals; a new command is a widened surface
requiring update consent and cannot be added by an unattended update. Core launcher
publication refreshes already-owned commands after runtime updates without exporting
new names. On Windows the native executable launcher preserves the argv boundary;
there is no batch-file `%*` fallback.

Use `after-install.md` to explain the next explicit setup command. Do not claim
that a launcher is available if the installer reports a collision or export error.
Installation must not start onboarding, scan user files, install a service or start
a monitor implicitly.
