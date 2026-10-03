---
sidebar_position: 3.5
title: "Skill Authoring Cookbook"
description: "A complete, worked example of building, testing, and publishing a Hermes Agent skill — from empty directory to Skills Hub."
---

# Skill Authoring Cookbook

A complete, worked example of building a Hermes Agent skill end-to-end. For the full format specification, see [Creating Skills](/developer-guide/creating-skills). This cookbook walks through the decisions and the finished artifact.

We'll build **word-count**, a skill that counts words in a file or inline text. Simple enough to follow, real enough to demonstrate every section.

---

## 1. Decide: skill, tool, or plugin?

Per the [Footprint Ladder](/developer-guide/contributing#the-footprint-ladder), prefer the highest rung that solves the problem. Word counting is expressible as instructions + a helper script + the `terminal` tool — so it's a **skill**, not a tool or plugin.

Ask yourself:

- Can the agent do this with existing tools and shell commands? → skill
- Does it need API keys wired into the agent? → tool or plugin
- Does it need custom Python that must run precisely every time? → tool

---

## 2. Directory layout

```
word-count/
├── SKILL.md              # required: instructions the agent reads
├── scripts/
│   └── word_count.py     # helper script the agent invokes via `terminal`
├── tests/
│   └── test_word_count.py
├── README.md
└── LICENSE
```

Scripts go in `scripts/`, never inline in SKILL.md — the model shouldn't re-derive parsing logic on every call.

---

## 3. Write the helper script first

Stdlib-only, cross-platform, no side effects:

```python
#!/usr/bin/env python3
"""Count words in a file or inline text."""
import argparse, sys

def count_words(text):
    return len(text.split())

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", help="File path or inline text")
    p.add_argument("--file", action="store_true", help="Treat source as a file path")
    args = p.parse_args(argv)
    if args.file:
        text = open(args.source, encoding="utf-8").read()
    else:
        text = args.source
    print(count_words(text))
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

Rules of thumb:
- **stdlib only** unless the dependency is unavoidable (then document it in `## Prerequisites`)
- **explicit `encoding='utf-8'`** on every file open (Windows locale defaults are cp1252/cp936)
- **exit codes**: `0` success, non-zero on error — the agent reads them
- **reconfigure stdout** if you print emoji or non-ASCII status glyphs

---

## 4. Write SKILL.md

```markdown
---
name: word-count
description: Count words in a file or inline text.
version: 1.0.0
author: Your Name (https://github.com/yourhandle)
license: MIT
metadata:
  hermes:
    tags: [Productivity, Text]
---

# Word Count Skill

Counts words in a file or inline text using a stdlib-only helper script.

## When to Use
- The user asks for a word count on a file or pasted text.

## Prerequisites
None — stdlib only.

## How to Run
Run the bundled script through the `terminal` tool:

    python ${HERMES_SKILL_DIR}/scripts/word_count.py --file path/to/file.md

## Quick Reference
| Task | Command |
|------|---------|
| Count in file | add `--file` |
| Count inline text | pass text directly |

## Procedure
1. Run the script via `terminal` with the user's file or text.
2. Report the count in one sentence.

## Pitfalls
- Files with only whitespace return 0 — surface that rather than guessing.

## Verification
- The output is a single integer.
- Exit code is `0` on success.
```

Key rules (enforced by reviewers):
- `description` ≤ 60 characters, one sentence, ends with a period
- Reference native Hermes tools (`terminal`, `read_file`) in backticks, not shell utilities
- Use `${HERMES_SKILL_DIR}` for script paths — the agent sees the substituted absolute path
- Modern section order: title, intro, When to Use, Prerequisites, How to Run, Quick Reference, Procedure, Pitfalls, Verification

---

## 5. Test it

Offline tests, no network, no live Hermes:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import word_count

def test_counts_words():
    assert word_count.count_words("hello world foo") == 3

def test_empty_text():
    assert word_count.count_words("") == 0
```

Then exercise the skill end-to-end:

```
hermes chat --toolsets skills -q "Use the word-count skill on README.md"
```

---

## 6. Ship it

**To a GitHub repo** (recommended — users install via `hermes skills install owner/repo`):

```
git init && git add -A && git commit -m "feat: word-count skill"
gh repo create hermes-word-count --public --source . --push
```

Then submit a PR to the [awesome-hermes-agent](https://github.com/0xNyk/awesome-hermes-agent) Community Skills list for discovery.

**To the Skills Hub** (for broadly useful skills):

```
hermes skills publish skills/word-count --to github --repo yourhandle/skills
```

---

## 7. Blueprint variant (optional)

Add a `blueprint` block to make the skill a runnable automation:

```yaml
metadata:
  hermes:
    blueprint:
      schedule: "0 9 * * *"
      deliver: origin
      prompt: "Count words in notes.md and report the total."
```

Installing the skill now registers it as a **suggested cron job** — the user accepts via `/suggestions`. See [Blueprints](/developer-guide/creating-skills#blueprints-skills-that-are-also-automations).
