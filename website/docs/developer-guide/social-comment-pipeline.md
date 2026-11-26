# Social comment insight pipeline

`scripts/social_comment_pipeline.py` turns user-authorized social-platform comment exports into product insight archives and Hermes Kanban task packages.

## Scope and compliance

Use only data you are allowed to process:

- Official platform APIs or export files.
- Creator/admin dashboard exports.
- User-provided CSV/JSON/JSONL comment dumps.
- Webhook events from accounts, communities, or campaigns you operate.

Do **not** use this pipeline to bypass logins, CAPTCHA, anti-bot protections, rate limits, paywalls, private groups, or platform terms. The scripts deliberately start from local export files or authorized upstream systems instead of scraping protected pages.

Before running this in production:

1. Confirm the account owner or data controller authorized collection and analysis.
2. Keep platform/user identifiers only when they are necessary for follow-up; otherwise prefer hashed or redacted IDs.
3. Store archives in an access-controlled workspace because `insights.md` and task files may contain direct user quotes.
4. Review generated Kanban tasks before automatic dispatch if the source contains sensitive support, health, finance, or minor-related content.
5. Treat LLM-generated requirements as triage suggestions, not final product decisions; a product manager should approve scope and priority.

## Input formats

The pipeline accepts `.json`, `.jsonl`, and `.csv`. It recognizes common fields:

- comment text: `text`, `content`, `comment`, `body`, `message`, `评论`, `内容`
- IDs: `id`, `comment_id`, `post_id`, `video_id`, `note_id`, etc.
- metadata: `platform`, `author`, `created_at`, `url`

Minimal JSONL example:

```jsonl
{"platform":"douyin","post_id":"p1","id":"c1","text":"希望支持导出 Excel 报表，手工统计太麻烦了"}
{"platform":"xiaohongshu","post_id":"n1","id":"c2","text":"登录验证码经常失败，账号打不开"}
```

## One-shot analysis

```bash
python scripts/social_comment_pipeline.py \
  --input ~/.hermes/social-comments/inbox \
  --output ~/.hermes/social-comments/archive \
  --dry-run-kanban
```

Outputs per run:

- `normalized_comments.jsonl`: normalized/deduplicated comments.
- `insights.json`: structured insights.
- `insights.md`: readable product insight report.
- `product_manager_brief.md`: archive package for the product-manager agent.
- `agent_tasks/*.md`: task package for product manager, developer, tester, and acceptance agents.
- `kanban_tasks.json`: structured task payloads.
- `kanban_dispatch_results.json`: dry-run commands or actual Hermes Kanban create results.
- `summary.json`: run summary.

## Dispatch to Hermes Kanban

Dry run first:

```bash
python scripts/social_comment_pipeline.py \
  --input ~/.hermes/social-comments/inbox/comments.jsonl \
  --dry-run-kanban
```

Create Kanban cards:

```bash
python scripts/social_comment_pipeline.py \
  --input ~/.hermes/social-comments/inbox/comments.jsonl \
  --dispatch-kanban \
  --workspace scratch
```

Each top requirement becomes four role-specific cards:

- `product_manager`: clarify and prioritize requirement.
- `developer`: implement MVP or technical fix.
- `tester`: design and run tests.
- `acceptance`: verify against acceptance criteria.

Use `--board <slug>` if you maintain a dedicated board.

## Scheduled directory watcher

`scripts/social_comment_watch.py` scans an inbox directory, processes new files once, and stays silent when there is nothing new. This is suitable for Hermes cron `--no-agent` jobs.

Manual run:

```bash
python scripts/social_comment_watch.py \
  --input-dir ~/.hermes/social-comments/inbox \
  --output ~/.hermes/social-comments/archive \
  --dry-run-kanban
```

Cron job example:

```bash
hermes cron create "*/30 * * * *" \
  --name social-comment-watch \
  --script social_comment_watch.py \
  --no-agent
```

When ready to create tasks automatically, pass script args through your cron wrapper or run a wrapper script that adds `--dispatch-kanban`.

## Product workflow

1. Put authorized export files into `~/.hermes/social-comments/inbox/`.
2. Watcher runs by cron and archives a report under `~/.hermes/social-comments/archive/`.
3. Product manager agent reads `product_manager_brief.md` and generated task files.
4. Optional Kanban dispatch creates subagent cards for PM/developer/tester/acceptance.
5. Acceptance agent verifies against the generated criteria before release.

## Minimal handoff example

Use this example when you want a product-manager agent to review one archive before creating implementation work.

1. Run analysis without dispatch:

```bash
python scripts/social_comment_pipeline.py \
  --input ~/.hermes/social-comments/inbox/comments.jsonl \
  --output ~/.hermes/social-comments/archive \
  --dry-run-kanban
```

2. Open the newest archive directory from `summary.json`, then hand the PM brief to an agent:

```bash
ARCHIVE_DIR=$(python - <<'PY'
import json, pathlib
summaries = sorted(pathlib.Path.home().glob('.hermes/social-comments/archive/*/summary.json'))
print(summaries[-1].parent)
PY
)

hermes chat -s hermes-agent -q "Act as the product manager agent. Read $ARCHIVE_DIR/product_manager_brief.md and the files under $ARCHIVE_DIR/agent_tasks/. Approve, merge, or reject the proposed requirements. If approved, return the exact developer/tester/acceptance tasks that should be dispatched."
```

3. After PM approval, create task cards from the same input:

```bash
python scripts/social_comment_pipeline.py \
  --input ~/.hermes/social-comments/inbox/comments.jsonl \
  --output ~/.hermes/social-comments/archive \
  --dispatch-kanban \
  --workspace scratch
```

The generated role files are intentionally explicit:

- `agent_tasks/product_manager_*.md`: problem framing, priority, and evidence review.
- `agent_tasks/developer_*.md`: implementation goal and acceptance criteria.
- `agent_tasks/tester_*.md`: regression, edge-case, and evidence-based test plan.
- `agent_tasks/acceptance_*.md`: final verification checklist tied to user quotes.
