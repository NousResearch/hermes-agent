# Filesystem Checkpoints

Hermes can automatically snapshot your working directory before making file changes, giving you a safety net to roll back if something goes wrong.

## How It Works

When enabled, Hermes takes a **one-time snapshot** at the start of each conversation turn before the first file-modifying operation (`write_file` or `patch`). This creates a point-in-time backup you can restore to at any time.

Under the hood, checkpoints use a **shadow git repository** stored at `~/.hermes/checkpoints/`. This is completely separate from your project's git — no `.git` directory is created in your project, and your own git history is never touched.

## Enabling Checkpoints

### Per-session (CLI flag)

```bash
hermes --checkpoints
```

### Permanently (config.yaml)

```yaml
# ~/.hermes/config.yaml
checkpoints:
  enabled: true
  max_snapshots: 50  # max checkpoints per directory (default: 50)
```

## Rolling Back

Use the `/rollback` slash command:

```
/rollback          # List all available checkpoints
/rollback 1        # Restore to checkpoint #1 (most recent)
/rollback 3        # Restore to checkpoint #3 (further back)
/rollback abc1234  # Restore by git commit hash
```

Example output:

```
📸 Checkpoints for /home/user/project:

  1. abc1234  2026-03-10 14:22  before write_file
  2. def5678  2026-03-10 14:15  before patch
  3. ghi9012  2026-03-10 14:08  before write_file

Use /rollback <number> to restore, e.g. /rollback 1
```

When you restore, Hermes automatically takes a **pre-rollback snapshot** first — so you can always undo your undo.

## What Gets Checkpointed

Checkpoints capture the entire working directory (the project root), excluding common large/sensitive patterns:

- `node_modules/`, `dist/`, `build/`
- `.env`, `.env.*`
- `__pycache__/`, `*.pyc`
- `.venv/`, `venv/`
- `.git/`
- `.DS_Store`, `*.log`

## Performance

Checkpoints are designed to be lightweight:

- **Once per turn** — only the first file operation triggers a snapshot, not every write
- **Skips large directories** — directories with >50,000 files are skipped automatically
- **Skips when nothing changed** — if no files were modified since the last checkpoint, no commit is created
- **Non-blocking** — if a checkpoint fails for any reason, the file operation proceeds normally

## How It Determines the Project Root

When you write to a file like `src/components/Button.tsx`, Hermes walks up the directory tree looking for project markers (`.git`, `pyproject.toml`, `package.json`, `Cargo.toml`, etc.) to find the project root. This ensures the entire project is checkpointed, not just the file's parent directory.

## Platforms

Checkpoints work on both:
- **CLI** — uses your current working directory
- **Gateway** (Telegram, Discord, etc.) — uses `MESSAGING_CWD`

The `/rollback` command is available on all platforms.

## FAQ

**Does this conflict with my project's git?**
No. Checkpoints use a completely separate shadow git repository via `GIT_DIR` environment variables. Your project's `.git/` is never touched.

**How much disk space do checkpoints use?**
Git is very efficient at storing diffs. For most projects, checkpoint data is negligible. Old checkpoints are pruned when `max_snapshots` is exceeded.

**Can I checkpoint without git installed?**
No — git must be available on your PATH. If it's not installed, checkpoints silently disable.

**Can I roll back across sessions?**
Yes! Checkpoints persist in `~/.hermes/checkpoints/` and survive across sessions. You can roll back to a checkpoint from yesterday.
