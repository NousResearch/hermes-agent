---
title: "Kanban Worker — Hermes Kanban worker를 위한 pitfalls, examples, edge cases"
sidebar_label: "Kanban Worker"
description: "Hermes Kanban worker를 위한 pitfalls, examples, edge cases"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Kanban Worker

Hermes Kanban worker를 위한 pitfalls, examples, edge cases 문서입니다. lifecycle 자체는 `KANBAN_GUIDANCE`로 모든 worker의 system prompt에 자동 주입되며(`agent/prompt_builder.py`), 이 skill은 **특정 시나리오에서 더 깊은 상세 지침이 필요할 때** 로드하는 자료입니다.

## Skill metadata

| | |
|---|---|
| Source | Bundled (기본 설치) |
| Path | `skills/devops/kanban-worker` |
| Version | `2.0.0` |
| Tags | `kanban`, `multi-agent`, `collaboration`, `workflow`, `pitfalls` |
| Related skills | [`kanban-orchestrator`](./devops-kanban-orchestrator) |

## Reference: full SKILL.md

:::info
아래는 이 skill이 트리거될 때 Hermes가 실제로 로드하는 **전체 skill 정의**입니다. 즉, skill이 활성화되었을 때 agent가 실제 instruction으로 보는 내용입니다.
:::

# Kanban Worker — Pitfalls and Examples

> 이 skill이 보이는 이유는 Hermes Kanban dispatcher가 당신을 `--skills kanban-worker`와 함께 worker로 spawn했기 때문입니다. dispatched worker마다 자동으로 로드됩니다. **lifecycle**(6단계: orient → work → heartbeat → block/complete)은 system prompt에 자동 주입되는 `KANBAN_GUIDANCE` block에도 들어 있습니다. 이 skill은 그보다 더 구체적인 심화 설명입니다: 좋은 handoff 형태, retry 진단, edge case 등.

## Workspace handling

workspace 종류에 따라 `$HERMES_KANBAN_WORKSPACE` 안에서의 행동 방식이 달라집니다.

| Kind | 의미 | 작업 방식 |
|---|---|---|
| `scratch` | 새 tmp 디렉터리, 오직 당신만 사용 | 자유롭게 read/write 가능; task가 archived되면 GC 대상 |
| `dir:<path>` | 공유되는 persistent directory | 다른 run이 당신이 쓴 내용을 읽게 됨. 장기 상태처럼 다뤄야 함. path는 항상 절대경로임 (kernel이 상대경로 거부) |
| `worktree` | 해당 경로의 Git worktree | `.git`이 없다면 먼저 main repo에서 `git worktree add <path> <branch>`를 실행한 뒤 cd하여 작업. 여기서 commit 수행 |

## Tenant isolation

`$HERMES_TENANT`가 설정되어 있으면 이 task는 특정 tenant namespace에 속합니다. persistent memory를 읽거나 쓸 때는 tenant prefix를 붙여서 context가 다른 tenant로 새지 않게 하세요.

- Good: `business-a: Acme is our biggest customer`
- Bad (leaks): `Acme is our biggest customer`

## 좋은 summary + metadata 형태

`kanban_complete(summary=..., metadata=...)` handoff는 downstream worker가 당신의 작업을 읽는 기본 채널입니다. 잘 작동하는 패턴은 다음과 같습니다.

**코딩 task:**
```python
kanban_complete(
    summary="shipped rate limiter — token bucket, keys on user_id with IP fallback, 14 tests pass",
    metadata={
        "changed_files": ["rate_limiter.py", "tests/test_rate_limiter.py"],
        "tests_run": 14,
        "tests_passed": 14,
        "decisions": ["user_id primary, IP fallback for unauthenticated requests"],
    },
)
```

**리서치 task:**
```python
kanban_complete(
    summary="3 competing libraries reviewed; vLLM wins on throughput, SGLang on latency, Tensorrt-LLM on memory efficiency",
    metadata={
        "sources_read": 12,
        "recommendation": "vLLM",
        "benchmarks": {"vllm": 1.0, "sglang": 0.87, "trtllm": 0.72},
    },
)
```

**리뷰 task:**
```python
kanban_complete(
    summary="reviewed PR #123; 2 blocking issues found (SQL injection in /search, missing CSRF on /settings)",
    metadata={
        "pr_number": 123,
        "findings": [
            {"severity": "critical", "file": "api/search.py", "line": 42, "issue": "raw SQL concat"},
            {"severity": "high", "file": "api/settings.py", "issue": "missing CSRF middleware"},
        ],
        "approved": False,
    },
)
```

`metadata`는 downstream parser(reviewer, aggregator, scheduler)가 prose를 다시 읽지 않고도 사용할 수 있는 형태로 구성하세요.

## 빨리 답을 받을 수 있는 block reason

나쁜 예: `"stuck"` — 사람은 무슨 일이 막혔는지 알 수 없습니다.

좋은 예: **어떤 결정을 내려야 하는지 한 문장으로 특정**하고, 긴 배경 설명은 comment로 남기세요.

```python
kanban_comment(
    task_id=os.environ["HERMES_KANBAN_TASK"],
    body="Full context: I have user IPs from Cloudflare headers but some users are behind NATs with thousands of peers. Keying on IP alone causes false positives.",
)
kanban_block(reason="Rate limit key choice: IP (simple, NAT-unsafe) or user_id (requires auth, skips anonymous endpoints)?")
```

block message는 dashboard / gateway notifier에 그대로 나타나는 짧은 문구이고, comment는 사람이 task를 열었을 때 읽는 깊은 배경 설명입니다.

## 보낼 가치가 있는 heartbeat

좋은 heartbeat는 진척을 이름 붙여서 말합니다. 예: `"epoch 12/50, loss 0.31"`, `"scanned 1.2M/2.4M rows"`, `"uploaded 47/120 videos"`.

나쁜 heartbeat는 `"still working"`, 빈 note, 초단위 남발입니다. 몇 분에 한 번이면 충분하고, 2분 이하 작업이면 아예 보내지 않아도 됩니다.

## Retry 시나리오

`kanban_show` 결과의 `runs: [...]`에 닫힌 run이 하나 이상 있다면 당신은 retry worker입니다. 이전 run의 `outcome` / `summary` / `error`가 무엇이 잘 안 됐는지 알려줍니다. **같은 경로를 반복하지 마세요.** 전형적인 retry 진단은 아래와 같습니다.

- `outcome: "timed_out"` — 이전 시도가 `max_runtime_seconds`에 걸렸습니다. 작업을 chunk로 나누거나 더 짧게 만들어야 할 수 있습니다.
- `outcome: "crashed"` — OOM 또는 segfault. 메모리 사용량을 줄이세요.
- `outcome: "spawn_failed"` + `error: "..."` — 대개 profile 설정 문제(credential 누락, PATH 불량). 무작정 재시도하지 말고 `kanban_block`으로 사람에게 물으세요.
- `outcome: "reclaimed"` + `summary: "task archived..."` — operator가 이전 run 도중 task를 archive했습니다. 아마 지금 실행되면 안 되는 상태일 수 있으니 status를 먼저 확인하세요.
- `outcome: "blocked"` — 이전 시도가 block 상태였고, unblock comment가 thread에 달려 있을 가능성이 큽니다.

## Do NOT

- `kanban_create` 대신 `delegate_task`를 cross-agent handoff로 쓰지 마세요. `delegate_task`는 **당신 자신의 run 내부**에서 쓰는 짧은 reasoning subtask용이고, `kanban_create`는 API loop를 넘어서 살아남는 cross-agent handoff용입니다.
- task body에 명시되지 않았다면 `$HERMES_KANBAN_WORKSPACE` 밖의 파일을 수정하지 마세요.
- follow-up task를 자기 자신에게 assign하지 마세요. 올바른 specialist에게 assign하세요.
- 실제로 끝내지 않은 task를 completed로 처리하지 마세요. 그 대신 block하세요.

## Pitfalls

**dispatch와 worker startup 사이에 task 상태가 바뀔 수 있습니다.** dispatcher가 claim한 뒤 실제 프로세스가 부팅되기 전까지 task가 blocked, reassigned, archived 되었을 수 있습니다. 항상 먼저 `kanban_show`를 호출하세요. 결과가 `blocked` 또는 `archived`라면 중단해야 합니다. 지금 실행되면 안 되는 상태입니다.

**workspace에 stale artifact가 남아 있을 수 있습니다.** 특히 `dir:`와 `worktree` workspace는 이전 run의 파일이 남아 있을 수 있습니다. comment thread를 읽으세요. 대개 왜 다시 실행되는지, 현재 workspace 상태가 어떤지를 설명하고 있습니다.

**guidance가 있는데 CLI에 의존하지 마세요.** `kanban_*` tool은 모든 terminal backend(Docker, Modal, SSH)에서 동작합니다. 반면 terminal tool 안에서 `hermes kanban <verb>`를 실행하면, containerized backend에서는 CLI가 설치돼 있지 않아 실패할 수 있습니다. 확신이 없을 때는 tool을 쓰세요.

## CLI fallback (스크립팅용)

각 tool에는 사람/스크립트를 위한 CLI 대응물이 있습니다.
- `kanban_show` ↔ `hermes kanban show <id> --json`
- `kanban_complete` ↔ `hermes kanban complete <id> --summary "..." --metadata '{...}'`
- `kanban_block` ↔ `hermes kanban block <id> "reason"`
- `kanban_create` ↔ `hermes kanban create "title" --assignee <profile> [--parent <id>]`
- 등등

agent 내부에서는 tool을 쓰고, CLI는 터미널 앞의 인간을 위한 인터페이스라고 생각하면 됩니다.
