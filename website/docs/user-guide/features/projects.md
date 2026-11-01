---
sidebar_position: 13
title: "Projects (Named Workspaces)"
description: "Group repos into named, multi-folder workspaces that anchor desktop session grouping and give kanban tasks deterministic worktrees and branches"
---

# Projects (Named Workspaces)

A **project** is a named workspace that can span several folders and repositories. You
give the group one identity, and Hermes uses it in two places: the desktop sidebar groups
sessions under it, and [kanban](./kanban) tasks linked to it get a deterministic worktree
and branch instead of a random one.

Projects are stored **per profile** in `$HERMES_HOME/projects.db`. Two profiles do not
share projects, and a worker profile started by the kanban dispatcher has its own file.

:::info Not the same as context files
[Context files](./context-files) (`AGENTS.md`, `.hermes.md`, `CLAUDE.md`) inject
*instructions* into the prompt. A project records *where the work lives* and what the
group is called. A folder can belong to a project with no context file in it, and the
other way round.
:::

## Create a project

```bash
# One folder, made the active project
hermes project create "Hermes Agent" ~/Proyectos/hermes-agent-contrib --use

# Several folders: the first one is primary unless --primary says otherwise
hermes project create "Soy Juan Paulino" ~/Proyectos/soy-juanpaulino ~/Proyectos/api \
  --primary ~/Proyectos/soy-juanpaulino

# With presentation metadata and a kanban board binding
hermes project create "Client Work" ~/work/client --icon 🧰 --color "#4f46e5" --board client
```

`create` prints the generated slug and id:

```text
Created project hermes-agent (p_1841262f)
```

The slug is derived from the name (kebab-case, `[a-z0-9_-]`, max 64 chars) and is what
kanban branch names are built from. Pass `--slug` if you want to pin it up front.

## Work with the active project

```bash
hermes project list            # * marks the active project
hermes project list --all      # include archived
hermes project use hermes-agent # set active
hermes project use             # clear it
```

The active project is a profile-level default. It does **not** move any running session's
working directory — several chats can share a profile, and moving one of them is an
explicit action (see [In the desktop app](#in-the-desktop-app)).

## Manage folders

```bash
hermes project add-folder hermes-agent ~/Proyectos/hermes-agent-contrib --label main --primary
hermes project remove-folder hermes-agent ~/Proyectos/old-checkout
hermes project set-primary hermes-agent ~/Proyectos/hermes-agent-contrib
hermes project show hermes-agent
```

`set-primary` only accepts a path that is **already** a folder of the project — add it
first. `show` lists every folder, with `*` on the primary one:

```text
hermes-agent  [p_1841262f]
  name:    Hermes Agent
  primary: /Users/you/Proyectos/hermes-agent-contrib
  folders:
    * /Users/you/Proyectos/hermes-agent-contrib
```

## Rename, archive, restore

```bash
hermes project rename hermes-agent "Hermes (upstream)"   # display name only
hermes project archive hermes-agent
hermes project restore hermes-agent
```

`rename` changes the label, never the slug — existing kanban branches keep their prefix.
Archiving hides the project from `list` but keeps its folders and board binding; nothing
is deleted, and `restore` brings it back.

## In the desktop app

The Projects view groups sessions by project and is **repo-first**: on first open it scans
your machine for git repositories and caches what it found, so later opens are instant
instead of re-walking the disk. A repository discovered there can be adopted into a
project without typing a path.

Inside a chat, the agent can move the session with the `desktop_project` tool
(`create` / `switch` / `list`) — that re-anchors the current session's workspace and the
sidebar follows. The tool ships in the `project` toolset, which the desktop and TUI enable;
in a plain CLI or messaging session there is a database row but no workspace to move.

## Kanban integration

Binding a board to a project (`--board` at create time, or `bind-board`) makes every
task linked to that project run in:

- a **worktree** at `<primary-repo>/.worktrees/<task-id>`, and
- a **branch** named `<project-slug>/<task-id>[-<title-slug>]`.

That replaces the opaque `wt/<task-id>` fallback and makes parallel agent work readable in
`git branch`. See [Kanban](./kanban) for the board itself.

```bash
hermes project bind-board hermes-agent hermes          # bind
hermes project bind-board hermes-agent                 # unbind
```

## Command reference

| Subcommand | Purpose |
|---|---|
| `create NAME [FOLDERS...]` | New project. `--slug`, `--primary`, `--description`, `--icon`, `--color`, `--board`, `--use`. |
| `list` / `ls` | List projects. `--all` includes archived. |
| `show PROJECT` | Details: name, primary path, folders. |
| `add-folder PROJECT PATH` | Add a folder. `--label`, `--primary`. |
| `remove-folder PROJECT PATH` | Remove a folder. |
| `rename PROJECT NAME` | Change the display name. |
| `set-primary PROJECT PATH` | Mark an existing folder as primary. |
| `use [PROJECT]` | Set (or clear) the active project. |
| `archive` / `restore PROJECT` | Hide / bring back. |
| `bind-board PROJECT [BOARD]` | Bind or unbind a kanban board. |

`PROJECT` accepts the id, the slug or the exact name, and matching falls back to a
case-insensitive comparison — `hermes-agent`, `Hermes Agent` and `p_1841262f` all resolve
to the same project.

## Notes

- Don't edit `projects.db` by hand while Hermes is running; writes are transactional. Use
  the CLI or the tool.
- Projects don't transfer between profiles. Recreating one is the supported path.
- Deleting is not a CLI operation: archive is the reversible way to get a project out of
  the way.
