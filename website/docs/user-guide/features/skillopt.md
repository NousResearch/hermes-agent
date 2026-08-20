---
sidebar_position: 3
title: "SkillOpt Promotion Gate"
description: "Promote candidate SKILL.md edits only after validation, identity checks, and held-out score improvement"
---

# SkillOpt Promotion Gate

Hermes can promote skill edits through a conservative gate inspired by:

> Yifan Yang et al. **"SkillOpt: Executive Strategy for Self-Evolving Agent Skills."** arXiv:2605.23904, 2026. [arXiv:2605.23904](https://arxiv.org/abs/2605.23904)

This is not a full research-system clone. It is the practical production piece Hermes needs: keep candidate skill edits quarantined, validate them, and only replace the live skill when the candidate proves better on a held-out evaluator.

## What the gate enforces

`hermes skills optimize` promotes a candidate only when all of these are true:

- the current skill and candidate are valid `SKILL.md` documents;
- the candidate is a separate quarantined file, not the live `SKILL.md` itself;
- the candidate preserves the current skill's frontmatter `name:` identity;
- baseline and candidate scores are finite numbers;
- an optional validator command exits `0`;
- the candidate score strictly beats the baseline score, unless `--allow-equal` is set.

If the gate accepts, Hermes backs up the previous `SKILL.md`, atomically replaces it with the candidate, and records the promotion. If it rejects, the live skill is left untouched and the rejection is recorded for later analysis.

## CLI usage

```bash
hermes skills optimize ~/.hermes/skills/my-skill \
  --candidate /tmp/candidate-SKILL.md \
  --baseline-score 0.72 \
  --candidate-score 0.81 \
  --validator 'python tests/evaluate_skill.py'
```

The validator runs in the live skill directory and receives:

- `HERMES_SKILLOPT_SKILL` - path to the current live `SKILL.md`
- `HERMES_SKILLOPT_CANDIDATE` - path to the candidate `SKILL.md`

Useful flags:

| Flag | Purpose |
| --- | --- |
| `--allow-equal` | Permit equal scores when the evaluator is noisy. Default requires strict improvement. |
| `--dry-run` | Evaluate and record the decision without replacing the live skill. |
| `--validator '<cmd>'` | Require a project-specific command to pass before promotion. |

## Slash usage inside chat

Inside a running Hermes session:

```bash
/skills optimize ~/.hermes/skills/my-skill \
  --candidate /tmp/candidate-SKILL.md \
  --baseline-score 0.72 \
  --candidate-score 0.81
```

By default, slash-command mutations preserve the current prompt cache and activate on the next session. Pass `--now` only when you explicitly want immediate cache invalidation in the current session:

```bash
/skills optimize ~/.hermes/skills/my-skill \
  --candidate /tmp/candidate-SKILL.md \
  --baseline-score 0.72 \
  --candidate-score 0.81 \
  --now
```

## Files written

For a skill at `~/.hermes/skills/my-skill/SKILL.md`, optimization metadata lives beside the skill:

```text
~/.hermes/skills/my-skill/.skillopt/
├── backups/
│   └── SKILL.<timestamp>.<sha>.md
├── history.jsonl
└── rejected.jsonl
```

This keeps the live skill simple while preserving enough audit trail for future autonomous optimizers and human review.
