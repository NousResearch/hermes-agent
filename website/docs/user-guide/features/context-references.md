---
sidebar_position: 6
title: "Context References"
description: "Use @ references to inject files, folders, diffs, git history, and URLs into a Hermes message"
---

# Context References

Hermes can expand scoped context references directly inside your message before it reaches the model.

This lets you tell Hermes exactly what to read instead of hoping it searches the right files.

## Supported references

| Reference | Example | What Hermes injects |
|------|------|------|
| `@file:` | `@file:src/main.py` | Full text of the file |
| `@file:` with range | `@file:src/main.py:10-40` | Only the requested line range |
| `@folder:` | `@folder:src/` | A directory listing with file metadata |
| `@diff` | `@diff` | Current unstaged git diff |
| `@staged` | `@staged` | Current staged git diff |
| `@git:N` | `@git:3` | `git log -3 -p` output |
| `@url:` | `@url:https://example.com/spec` | Extracted page content |

References can appear anywhere in the message and you can mix multiple references in one turn.

```text
Review @file:agent/context_references.py and compare it with @diff
```

## How expansion works

Hermes keeps your plain-language message, removes the raw `@...` tokens from that sentence, and appends an `Attached Context` section containing the expanded content.

That means a prompt like:

```text
Review @file:src/main.py:1-20 and @diff
```

becomes something like:

```text
Review and

--- Attached Context ---

📄 @file:src/main.py:1-20 (...)
...

🧾 git diff (...)
...
```

## Token budget rules

Hermes estimates the number of tokens injected by `@` references before sending the turn to the model.

- Above 25% of the current model context window: Hermes warns that the injected context is large.
- Above 50% of the current model context window: Hermes refuses the expansion for that turn.

This prevents a few large references from silently blowing up the prompt.

## CLI behavior

The interactive CLI supports autocomplete for:

- `@file:`
- `@folder:`

`Tab` completes matching paths while keeping slash-command completion intact.

Regular path completion still works for plain path-like text such as `src/main.py`.

## Gateway behavior

Messaging platforms such as Telegram and Discord also expand `@` references in plain text messages.

Differences from the CLI:

- There is no interactive path autocomplete.
- Paths resolve from the messaging working directory (`MESSAGING_CWD`, or the configured terminal CWD).
- Git references only work when that directory is inside a git repository.

## Notes and limitations

- Hermes only matches typed reference prefixes. It does not treat email addresses like `user@example.com` or handles like `@teammate` as context references.
- Binary files are skipped with a warning.
- Missing files or invalid git commands produce warnings instead of crashing the turn.
- URL extraction depends on the configured web extraction backend.
