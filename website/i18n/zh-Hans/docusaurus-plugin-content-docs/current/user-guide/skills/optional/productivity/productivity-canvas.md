---
title: "Canvas — Canvas LMS 集成 — 使用 API 令牌认证获取已注册课程和作业"
sidebar_label: "Canvas"
description: "Canvas LMS 集成 — 使用 API 令牌认证获取已注册课程和作业"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Canvas

Canvas LMS 集成 — 使用 API 令牌认证获取已注册课程和作业.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/productivity/canvas` |
| Path | `optional-skills/productivity/canvas` |
| Version | `1.0.0` |
| Author | community |
| License | MIT |
| Tags | `Canvas`, `LMS`, `Education`, `Courses`, `Assignments` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Canvas LMS — Course & Assignment Access

Canvas LMS 的只读访问，用于列出课程和作业。

## Scripts

- `scripts/canvas_api.py` — Canvas API 调用的 Python CLI

## 设置

1. Log in to your Canvas instance in a browser
2. 转到 **Account → Settings** (click your profile icon, then Settings)
3. 滚动到 **已批准的集成** and click **+ New Access Token**
4. 命名令牌 (e.g., "Hermes Agent"), 设置可选过期时间, and click **生成令牌**
5. 复制令牌并添加到 `~/.hermes/.env`:

```
CANVAS_API_TOKEN=your_token_here
CANVAS_BASE_URL=https://yourschool.instructure.com
```

The base URL is whatever appears in your browser when you're logged into Canvas (no trailing slash).

## Usage

```bash
CANVAS="python $HERMES_HOME/skills/productivity/canvas/scripts/canvas_api.py"

# List all active courses
$CANVAS list_courses --enrollment-state active

# List all courses (any state)
$CANVAS list_courses

# List assignments for a specific course
$CANVAS list_assignments 12345

# List assignments ordered by due date
$CANVAS list_assignments 12345 --order-by due_at
```

## Output Format

**list_courses** returns:
```json
[{"id": 12345, "name": "Intro to CS", "course_code": "CS101", "workflow_state": "available", "start_at": "...", "end_at": "..."}]
```

**list_assignments** returns:
```json
[{"id": 67890, "name": "Homework 1", "due_at": "2025-02-15T23:59:00Z", "points_possible": 100, "submission_types": ["online_upload"], "html_url": "...", "description": "...", "course_id": 12345}]
```

Note: Assignment descriptions are truncated to 500 characters. The `html_url` field links to the full assignment page in Canvas.

## API Reference (curl)

```bash
# List courses
curl -s -H "Authorization: Bearer $CANVAS_API_TOKEN" \
  "$CANVAS_BASE_URL/api/v1/courses?enrollment_state=active&per_page=10"

# List assignments for a course
curl -s -H "Authorization: Bearer $CANVAS_API_TOKEN" \
  "$CANVAS_BASE_URL/api/v1/courses/COURSE_ID/assignments?per_page=10&order_by=due_at"
```

Canvas uses `Link` headers for pagination. The Python script handles pagination automatically.

## Rules

- This skill is **read-only** — it only fetches data, never modifies courses or assignments
- On first use, verify auth by running `$CANVAS list_courses` — if it fails with 401, guide the user through setup
- Canvas rate-limits to ~700 requests per 10 minutes; check `X-Rate-Limit-Remaining` header if hitting limits

## Troubleshooting

| Problem | Fix |
|---------|-----|
| 401 Unauthorized | Token invalid or expired — regenerate in Canvas Settings |
| 403 Forbidden | Token lacks permission for this course |
| Empty course list | Try `--enrollment-state active` or omit the flag to see all states |
| Wrong institution | Verify `CANVAS_BASE_URL` matches the URL in your browser |
| Timeout errors | Check network connectivity to your Canvas instance |
