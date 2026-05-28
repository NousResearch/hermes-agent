---
sidebar_position: 4
title: "Skill Change Receipt Template"
description: "Compact receipt template for validating skill updates without retaining raw transcripts or secrets."
---

# Skill Change Receipt Template

Use this for skill updates when a compact receipt needs to survive after raw context is discarded.

```json
{
  "triggering_context": "class of task that caused the skill change",
  "issue_or_request": "optional issue, PR, or user request reference",
  "changed_paths": [
    "skills/category/name/SKILL.md"
  ],
  "validation_commands": [
    {
      "command": "python -m pytest tests/tools/test_skill_manager_tool.py -q -o 'addopts='",
      "result": "pass"
    }
  ],
  "retained_artifacts": [
    "path/to/compact-receipt.json"
  ],
  "discarded_artifacts": [
    "raw transcript",
    "secret-bearing logs"
  ],
  "known_limitations_or_falsifiers": [
    "Demote or update the skill if the validation command no longer reproduces."
  ]
}
```

## Example

```json
{
  "triggering_context": "receipt-validated skill lifecycle implementation",
  "issue_or_request": "leo-guinan/hermes-agent#2, #3, #5",
  "changed_paths": [
    "tools/skill_manager_tool.py",
    "tests/tools/test_skill_manager_tool.py",
    "website/docs/developer-guide/creating-skills.md",
    "skills/software-development/writing-plans/SKILL.md"
  ],
  "validation_commands": [
    {
      "command": "python -m pytest tests/tools/test_skill_manager_tool.py -q -o 'addopts='",
      "result": "pass"
    }
  ],
  "retained_artifacts": [
    "this document"
  ],
  "discarded_artifacts": [
    "raw agent transcript",
    "intermediate failing pytest output after RED step"
  ],
  "known_limitations_or_falsifiers": [
    "This validates metadata shape at skill_manager_tool frontmatter validation only; deeper curator policy belongs in a curator audit feature.",
    "If bundled skill loading uses a different parser with strict metadata rules later, add matching parser tests before locking."
  ]
}
```
