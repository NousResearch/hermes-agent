---
sidebar_position: 10
title: "บทช่วยสอน: GitHub PR Review Agent"
description: "สร้าง AI code reviewer แบบอัตโนมัติที่เฝ้าดู repo ของคุณ รีวิว pull request และส่งฟีดแบ็กถึงคุณ — โดยที่คุณไม่ต้องทำอะไรเลย"
---

# บทช่วยสอน: สร้าง GitHub PR Review Agent

**ปัญหาคือ:** ทีมคุณเปิด PR เร็วกว่าที่จะรีวิวทัน PR นั่งรอคนมาดูเป็นวันๆ เด็กจูเนียร์ merge โค้ดที่มี bug เพราะไม่มีใครมีเวลาตรวจ แล้วคุณก็ต้องใช้เวลาช่วงเช้าไล่อ่าน diff แทนที่จะเอาเวลาไปสร้างสรรค์

**ทางแก้คือ:** AI agent ที่เฝ้าดู repo ของคุณตลอด 24 ชั่วโมง รีวิว PR ใหม่ทุกตัวเพื่อหา bug ปัญหาด้านความปลอดภัย และคุณภาพโค้ด พร้อมสรุปส่งให้คุณ — เหลือเวลาของคุณไปให้เฉพาะ PR ที่ต้องใช้ดุลพินิจของมนุษย์จริงๆ เท่านั้น

**สิ่งที่คุณจะได้สร้าง:**

```
┌───────────────────────────────────────────────────────────────────┐
│                                                                   │
│   Cron Timer  ──▶  Hermes Agent  ──▶  GitHub API  ──▶  Review     │
│   (every 2h)       + gh CLI           (PR diffs)       delivery   │
│                    + skill                             (Telegram, │
│                    + memory                            Discord,   │
│                                                        local)     │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

คู่มือนี้ใช้ **cron jobs** โพลหา PR ตามตารางเวลา — ไม่ต้องมี server หรือ public endpoint ใช้ได้แม้อยู่หลัง NAT และ firewall

:::tip อยากได้รีวิวแบบ real-time แทน?
ถ้าคุณมี public endpoint ให้ใช้ ลองดู [Automated GitHub PR Comments with Webhooks](./webhook-github-pr-review.md) — GitHub จะ push events ไปหา Hermes ทันทีเมื่อมีการเปิดหรืออัปเดต PR
:::

---

## สิ่งที่ต้องมี

- **Hermes Agent ติดตั้งแล้ว** — ดูที่ [Installation guide](/getting-started/installation)
- **Gateway กำลังทำงานอยู่** สำหรับ cron jobs:
  ```bash
  hermes gateway install   # Install as a service
  # or
  hermes gateway           # Run in foreground
  ```
- **GitHub CLI (`gh`) ติดตั้งและ authenticate แล้ว**:
  ```bash
  # Install
  brew install gh        # macOS
  sudo apt install gh    # Ubuntu/Debian

  # Authenticate
  gh auth login
  ```
- **Messaging ตั้งค่าแล้ว** (ไม่บังคับ) — [Telegram](/user-guide/messaging/telegram) หรือ [Discord](/user-guide/messaging/discord)

:::tip ไม่มี messaging? ไม่เป็นไร
ใช้ `deliver: "local"` เพื่อบันทึกรีวิวไว้ที่ `~/.hermes/cron/output/` เหมาะมากสำหรับทดสอบก่อนต่อระบบแจ้งเตือนจริง
:::

---

## Step 1: ตรวจสอบว่า Setup พร้อม

ตรวจสอบว่า Hermes เข้าถึง GitHub ได้ เริ่ม chat:

```bash
hermes
```

ทดสอบด้วยคำสั่งง่ายๆ:

```
Run: gh pr list --repo NousResearch/hermes-agent --state open --limit 3
```

คุณควรเห็นรายการ PR ที่เปิดอยู่ ถ้าใช้ได้ คุณพร้อมแล้ว

---

## Step 2: ลองรีวิวด้วยตนเอง

ยังอยู่ใน chat เดิม ขอให้ Hermes รีวิว PR จริงสักตัว:

```
Review this pull request. Read the diff, check for bugs, security issues,
and code quality. Be specific about line numbers and quote problematic code.

Run: gh pr diff 3888 --repo NousResearch/hermes-agent
```

Hermes จะ:
1. Execute `gh pr diff` เพื่อดึงการเปลี่ยนแปลงของโค้ด
2. อ่าน diff ทั้งหมดจนจบ
3. สร้างรีวิวที่มีโครงสร้าง พร้อม findings ที่ระบุจุดชัดเจน

ถ้าคุณพอใจกับคุณภาพ ก็ถึงเวลาทำให้เป็นอัตโนมัติ

---

## Step 3: สร้าง Review Skill

Skill ให้ Hermes มีแนวทางการรีวิวที่สม่ำเสมอ ซึ่งคงอยู่ข้าม session และรอบการรัน cron ถ้าไม่มี คุณภาพรีวิวจะขึ้นกับดวง

```bash
mkdir -p ~/.hermes/skills/code-review
```

สร้างไฟล์ `~/.hermes/skills/code-review/SKILL.md`:

```markdown
---
name: code-review
description: Review pull requests for bugs, security issues, and code quality
---

# Code Review Guidelines

When reviewing a pull request:

## What to Check
1. **Bugs** — Logic errors, off-by-one, null/undefined handling
2. **Security** — Injection, auth bypass, secrets in code, SSRF
3. **Performance** — N+1 queries, unbounded loops, memory leaks
4. **Style** — Naming conventions, dead code, missing error handling
5. **Tests** — Are changes tested? Do tests cover edge cases?

## Output Format
For each finding:
- **File:Line** — exact location
- **Severity** — Critical / Warning / Suggestion
- **What's wrong** — one sentence
- **Fix** — how to fix it

## Rules
- Be specific. Quote the problematic code.
- Don't flag style nitpicks unless they affect readability.
- If the PR looks good, say so. Don't invent problems.
- End with: APPROVE / REQUEST_CHANGES / COMMENT
```

ตรวจสอบว่าโหลดสำเร็จ — เริ่ม `hermes` แล้วคุณควรเห็น `code-review` ในรายการ skills ตอนสตาร์ท

---

## Step 4: สอน Convention ของทีมให้มันรู้

นี่คือจุดที่ทำให้ reviewer มีประโยชน์จริงๆ เริ่ม session แล้วสอนมาตรฐานของทีมให้ Hermes:

```
Remember: In our backend repo, we use Python with FastAPI.
All endpoints must have type annotations and Pydantic models.
We don't allow raw SQL — only SQLAlchemy ORM.
Test files go in tests/ and must use pytest fixtures.
```

```
Remember: In our frontend repo, we use TypeScript with React.
No `any` types allowed. All components must have props interfaces.
We use React Query for data fetching, never useEffect for API calls.
```

ความจำเหล่านี้คงอยู่ตลอดไป — reviewer จะบังคับใช้ convention ของคุณโดยไม่ต้องพูดซ้ำทุกครั้ง

---

## Step 5: สร้าง Cron Job แบบอัตโนมัติ

ทีนี้ต่อทุกอย่างเข้าด้วยกัน สร้าง cron job ที่รันทุก 2 ชั่วโมง:

```bash
hermes cron create "0 */2 * * *" \
  "Check for new open PRs and review them.

Repos to monitor:
- myorg/backend-api
- myorg/frontend-app

Steps:
1. Run: gh pr list --repo REPO --state open --limit 5 --json number,title,author,createdAt
2. For each PR created or updated in the last 4 hours:
   - Run: gh pr diff NUMBER --repo REPO
   - Review the diff using the code-review guidelines
3. Format output as:

## PR Reviews — today

### [repo] #[number]: [title]
**Author:** [name] | **Verdict:** APPROVE/REQUEST_CHANGES/COMMENT
[findings]

If no new PRs found, say: No new PRs to review." \
  --name "pr-review" \
  --deliver telegram \
  --skill code-review
```

ยืนยันว่าถูกจดตารางเวลาไว้แล้ว:

```bash
hermes cron list
```

### ตารางเวลาที่มีประโยชน์อื่นๆ

| Schedule | เมื่อไหร่ |
|----------|------|
| `0 */2 * * *` | ทุก 2 ชั่วโมง |
| `0 9,13,17 * * 1-5` | วันละสามรอบ เฉพาะวันทำการ |
| `0 9 * * 1` | สรุปประจำสัปดาห์เช้าวันจันทร์ |
| `30m` | ทุก 30 นาที (repo ที่ traffic หนัก) |

---

## Step 6: รันเมื่อต้องการ

ไม่อยากรอตามตารางเวลา? สั่งรันเองได้เลย:

```bash
hermes cron run pr-review
```

หรือจากภายใน chat session:

```
/cron run pr-review
```

---

## ก้าวต่อไป

### โพสต์รีวิวลง GitHub โดยตรง

แทนที่จะส่งเข้า Telegram ให้ agent comment ลง PR เลย:

เพิ่มส่วนนี้เข้าไปใน cron prompt ของคุณ:

```
After reviewing, post your review:
- For issues: gh pr review NUMBER --repo REPO --comment --body "YOUR_REVIEW"
- For critical issues: gh pr review NUMBER --repo REPO --request-changes --body "YOUR_REVIEW"
- For clean PRs: gh pr review NUMBER --repo REPO --approve --body "Looks good"
```

:::caution
ตรวจสอบว่า `gh` มี token ที่มี scope `repo` รีวิวจะถูกโพสต์ในนามของ account ที่ `gh` authenticated อยู่
:::

### แดชบอร์ด PR รายสัปดาห์

สร้างภาพรวม repo ทั้งหมดของคุณสำหรับเช้าวันจันทร์:

```bash
hermes cron create "0 9 * * 1" \
  "Generate a weekly PR dashboard:
- myorg/backend-api
- myorg/frontend-app
- myorg/infra

For each repo show:
1. Open PR count and oldest PR age
2. PRs merged this week
3. Stale PRs (older than 5 days)
4. PRs with no reviewer assigned

Format as a clean summary." \
  --name "weekly-dashboard" \
  --deliver telegram
```

### เฝ้าดูหลาย Repo

ขยายขนาดได้โดยเติม repo เข้าไปใน prompt agent ประมวลผลไล่ทีละตัว — ไม่ต้องตั้งค่าเพิ่ม

---

## การแก้ปัญหา

### "gh: command not found"
Gateway ทำงานใน environment แบบ minimal ตรวจสอบว่า `gh` อยู่ใน system PATH แล้ว restart gateway

### รีวิวกลายๆ ไม่จำเพาะ
1. เพิ่ม skill `code-review` (Step 3)
2. สอน convention ของคุณให้ Hermes ผ่าน memory (Step 4)
3. Context เรื่อง stack ของคุณยิ่งมาก รีวิวยิ่งดีขึ้น

### Cron job ไม่ทำงาน
```bash
hermes gateway status    # Is the gateway running?
hermes cron list         # Is the job enabled?
```

### ข้อจำกัด rate limit
GitHub อนุญาต 5,000 API requests/ชั่วโมงสำหรับ user ที่ authenticated การรีวิวหนึ่งครั้งใช้ประมาณ ~3-5 requests (list + diff + comments ถ้ามี) แม้รีวิวถึง 100 PR/วันก็ยังอยู่ในขีดจำกัดอย่างสบาย

---

## ขั้นถัดไป

- **[Webhook-Based PR Reviews](./webhook-github-pr-review.md)** — รับรีวิวทันทีเมื่อมีการเปิด PR (ต้องมี public endpoint)
- **[Daily Briefing Bot](/guides/daily-briefing-bot)** — ผสาน PR review เข้ากับสรุปข่าวยามเช้าของคุณ
- **[Build a Plugin](/developer-guide/plugins)** — ห่อ logic การรีวิวเป็น plugin ที่แชร์ได้
- **[Profiles](/user-guide/profiles)** — รัน reviewer profile เฉพาะทางที่มี memory และ config เป็นของตัวเอง
- **[Fallback Providers](/user-guide/features/fallback-providers)** — มั่นใจได้ว่ารีวิวยังทำงานแม้ provider ตัวใดตัวหนึ่งล่ม
