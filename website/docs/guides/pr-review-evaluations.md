---
sidebar_position: 12
title: "PR Reviewer Evaluations"
description: "Run and extend the public OSS evaluation slice for Hermes PR Reviewer without private repositories or GitHub comments."
---

# PR Reviewer Evaluations

Hermes PR Reviewer ships a small public-OSS evaluation manifest so reviewer
changes can be tested without Larry's private repositories and without posting
comments to GitHub.

The bundled corpus lives at:

```text
plugins/pr_review/evals/public_prs.json
```

Each manifest entry records:

- `pr` — public GitHub PR reference in `owner/repo#number` form.
- `category` — one of the beta coverage buckets (`small-docs`, `frontend`,
  `backend`, `security`, `browser-tooling`, `ci-failing`,
  `generated-dependency-heavy`, `large-stress`).
- `observed_head_sha` — the PR head SHA observed when the corpus was seeded.
- `observed_check_status` — check-rollup status counts observed at that SHA.
- size metadata (`changed_files`, `additions`, `deletions`) and a short
  `rationale` explaining why the case is useful.

These fields make the corpus reproducible enough for beta comparisons while
still allowing live eval runs to refresh data from GitHub when an operator opts
in.

## Fixture/unit tests vs live evals

Fixture/unit tests are the default for CI and local development:

```bash
python -m pytest tests/plugins/test_pr_review_plugin.py -q
```

They parse the bundled manifest, verify summary output, and exercise a mocked
`--dry-run --no-llm` review path. They do not require private repositories,
network access, model calls, or GitHub write permissions.

Live evals are operator-driven checks against GitHub. They may observe newer PR
metadata than the manifest and require `gh` authentication. Keep them dry-run by
default while comparing reviewer behavior:

```bash
hermes pr-review eval-manifest
hermes pr-review eval-manifest --json
hermes pr-review review fastapi/fastapi#15815 --dry-run --json
```

Only pass `--post-comment` intentionally. The PR Reviewer CLI does not post any
GitHub comment unless that flag is present.

## Adding a case

1. Choose a public PR that covers a missing behavior bucket or improves stress
   coverage.
2. Record the current head SHA and check-rollup status counts with `gh pr view`.
3. Add a compact manifest entry with a stable `id` and clear `rationale`.
4. Run the plugin tests above.

Avoid private repos, private issue references, and cases whose value depends on
secrets or proprietary CI logs. Generated/dependency-heavy and large/stress
cases are useful, but keep CI tests on fixtures so the repository test suite
stays deterministic.
