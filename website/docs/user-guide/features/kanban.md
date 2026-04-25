# Kanban

The Kanban dashboard tab lets you queue and monitor Hermes work from the web UI. Cards are native Hermes tasks: starting a card runs `hermes chat` in the selected workspace, records the task log under your Hermes home directory, and moves the card to review when the process exits.

## Start

```bash
hermes dashboard
```

Open the **Kanban** tab, enter a title, prompt, and workspace path, then click **Add**. The workspace path must point to an existing directory.

## Card Lifecycle

Cards move through five columns:

| Column | Meaning |
| --- | --- |
| Backlog | Planned work that has not started |
| Running | A Hermes process is active for the card |
| Review | The task finished, stopped, or failed and needs inspection |
| Done | Work has been accepted |
| Trash | Archived board items |

The detail panel shows the prompt, workspace, latest log output, and git diff for the card workspace. If the workspace is a git repository, the diff panel reports `git status --short` and the current unstaged patch.

## Storage

Board data is stored at:

```text
~/.hermes/kanban/board.json
```

Task logs are stored at:

```text
~/.hermes/kanban/logs/
```

These paths are profile-aware and follow the active Hermes home.

## Security

Kanban uses the same dashboard session-token protection as other sensitive dashboard endpoints. Task launch uses argument lists rather than shell interpolation, and workspace paths are resolved before execution. The dashboard still runs local Hermes processes with your account permissions, so review prompts before starting cards.
