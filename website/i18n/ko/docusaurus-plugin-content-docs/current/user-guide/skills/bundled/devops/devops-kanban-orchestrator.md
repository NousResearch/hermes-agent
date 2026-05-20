---
title: "Kanban Orchestrator"
sidebar_label: "Kanban Orchestrator"
description: "Kanban을 통해 작업을 라우팅하는 orchestrator profile을 위한 작업 분해 playbook, specialist roster 관례, anti-temptation 규칙"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Kanban Orchestrator

Kanban을 통해 작업을 라우팅하는 orchestrator profile을 위한 작업 분해 playbook, specialist roster 관례, anti-temptation 규칙입니다. "직접 하지 말고 라우팅하라"는 규칙과 기본 lifecycle은 모든 kanban worker의 system prompt에 자동 주입되며, 이 skill은 **특히 orchestrator 역할을 수행할 때** 필요한 더 깊은 운영 지침을 담고 있습니다.

## Skill metadata

| | |
|---|---|
| Source | Bundled (기본 설치) |
| Path | `skills/devops/kanban-orchestrator` |
| Version | `2.0.0` |
| Tags | `kanban`, `multi-agent`, `orchestration`, `routing` |
| Related skills | [`kanban-worker`](./devops-kanban-worker) |

## Reference: full SKILL.md

:::info
아래 내용은 이 skill이 트리거될 때 Hermes가 실제로 로드하는 **전체 skill 정의**입니다. 즉, skill이 활성화되었을 때 agent가 실제 지침으로 보는 텍스트입니다.
:::

# Kanban Orchestrator — 작업 분해 playbook

> **핵심 worker lifecycle**(여기에는 `kanban_create` fan-out 패턴과 "분해만 하고 실행은 하지 말라"는 규칙 포함)은 `KANBAN_GUIDANCE` system-prompt block을 통해 모든 kanban process에 자동 주입됩니다. 이 skill은 **작업 라우팅만을 담당하는 orchestrator profile**일 때 참고하는 심화 playbook입니다.

## 언제 보드를 써야 하는가 (vs. 그냥 직접 해버리는가)

다음 중 하나라도 해당하면 Kanban task를 만드세요.

1. **여러 specialist가 필요할 때** — research + analysis + writing은 서로 다른 3개 profile입니다.
2. **작업이 crash나 restart 이후에도 살아남아야 할 때** — 장기 작업, 반복 작업, 중요한 작업.
3. **사용자가 중간에 끼어들 수 있어야 할 때** — 어느 단계에서든 human-in-the-loop가 필요함.
4. **여러 subtask를 병렬로 돌릴 수 있을 때** — fan-out으로 속도 개선.
5. **review / iteration이 예상될 때** — reviewer profile이 drafter 출력에 대해 반복 루프를 돌 것.
6. **감사 이력이 중요할 때** — board row는 SQLite에 영구 보존됨.

이 중 **하나도 해당하지 않고**, 단순한 one-shot reasoning task라면 Kanban 대신 `delegate_task`를 쓰거나 직접 답변하면 됩니다.

## Anti-temptation 규칙

당신의 직무 설명은 "execute가 아니라 route"입니다. 이를 강제하는 규칙은 다음과 같습니다.

- **직접 일을 실행하지 마세요.** 보통 restricted toolset에는 구현용 terminal/file/code/web조차 포함되지 않습니다. "이건 내가 빨리 고치면 되겠는데"라는 생각이 들면 멈추고, 올바른 specialist에게 task를 만드세요.
- **구체적인 작업이 생기면 무조건 Kanban task를 만들고 assign하세요.** 매번 예외 없이.
- **맞는 specialist가 없다면 어떤 profile을 새로 만들지 사용자에게 물으세요.** "대충 비슷하니까 내가 해도 되겠지"로 넘어가지 마세요.
- **분해하고, 라우팅하고, 요약하는 것 — 그게 전부입니다.**

## 표준 specialist roster (관례)

사용자 환경에서 별도로 profile을 커스터마이즈하지 않았다면, 다음 profile들이 있다고 가정합니다. 실제 환경이 다르면 그에 맞게 조정하고, 확신이 없으면 물으세요.

| Profile | 하는 일 | Typical workspace |
|---|---|---|
| `researcher` | 자료를 읽고, 사실을 수집하고, findings를 정리 | `scratch` |
| `analyst` | 종합, 랭킹, 중복 제거. 여러 `researcher` 출력물을 소비 | `scratch` |
| `writer` | 사용자의 문체에 맞춰 prose 초안 작성 | `scratch` 또는 Obsidian vault의 `dir:` |
| `reviewer` | 결과를 읽고, findings를 남기고, 승인 여부를 게이트 | `scratch` |
| `backend-eng` | 서버 사이드 코드 작성 | `worktree` |
| `frontend-eng` | 클라이언트 사이드 코드 작성 | `worktree` |
| `ops` | 스크립트 실행, 서비스 관리, 배포 처리 | ops scripts repo의 `dir:` |
| `pm` | spec, acceptance criteria 작성 | `scratch` |

## 작업 분해 playbook

### Step 1 — 목표 이해하기

목표가 애매하면 clarifying question을 하세요. 잘못된 fleet를 띄우는 비용이 질문 한 번보다 훨씬 큽니다.

### Step 2 — task graph를 먼저 스케치하기

무엇이든 생성하기 전에, 먼저 사용자에게 graph를 말로 스케치해서 보여주세요. 예를 들어 "Postgres로 마이그레이션할지 분석해줘"라면:

```
T1  researcher        research: Postgres cost vs current
T2  researcher        research: Postgres performance vs current
T3  analyst           synthesize migration recommendation       parents: T1, T2
T4  writer            draft decision memo                       parents: T3
```

이 초안을 사용자에게 보여주고, 실제 task를 만들기 전에 수정할 기회를 주세요.

### Step 3 — task 생성 및 link 연결

```python
t1 = kanban_create(
    title="research: Postgres cost vs current",
    assignee="researcher",
    body="Compare estimated infrastructure costs, migration costs, and ongoing ops costs over a 3-year window. Sources: AWS/GCP pricing, team time estimates, current Postgres bills from peers.",
    tenant=os.environ.get("HERMES_TENANT"),
)["task_id"]

t2 = kanban_create(
    title="research: Postgres performance vs current",
    assignee="researcher",
    body="Compare query latency, throughput, and scaling characteristics at our expected data volume (~500GB, 10k QPS peak). Sources: benchmark papers, public case studies, pgbench results if easy.",
)["task_id"]

t3 = kanban_create(
    title="synthesize migration recommendation",
    assignee="analyst",
    body="Read the findings from T1 (cost) and T2 (performance). Produce a 1-page recommendation with explicit trade-offs and a go/no-go call.",
    parents=[t1, t2],
)["task_id"]

t4 = kanban_create(
    title="draft decision memo",
    assignee="writer",
    body="Turn the analyst's recommendation into a 2-page memo for the CTO. Match the tone of previous decision memos in the team's knowledge base.",
    parents=[t3],
)["task_id"]
```

`parents=[...]`는 promotion을 gate합니다. 자식 task는 모든 부모가 `done`이 될 때까지 `todo`에 머물다가, 이후 자동으로 `ready`로 승격됩니다. 수동 조율은 필요 없고 dispatcher와 dependency engine이 처리합니다.

### Step 4 — 자기 자신의 task 완료 처리

만약 당신 자신도 하나의 task로 spawn된 상태였다면(예: `planner` profile이 `T0: "investigate Postgres migration"`을 assign받은 경우), 자신이 만든 task graph를 요약해서 완료 처리하세요.

```python
kanban_complete(
    summary="decomposed into T1-T4: 2 researchers parallel, 1 analyst on their outputs, 1 writer on the recommendation",
    metadata={
        "task_graph": {
            "T1": {"assignee": "researcher", "parents": []},
            "T2": {"assignee": "researcher", "parents": []},
            "T3": {"assignee": "analyst", "parents": ["T1", "T2"]},
            "T4": {"assignee": "writer", "parents": ["T3"]},
        },
    },
)
```

### Step 5 — 사용자에게 보고하기

무엇을 만들었는지 평문으로 설명하세요.

> I've queued 4 tasks:
> - **T1** (researcher): cost comparison
> - **T2** (researcher): performance comparison, in parallel with T1
> - **T3** (analyst): synthesizes T1 + T2 into a recommendation
> - **T4** (writer): turns T3 into a CTO memo
>
> The dispatcher will pick up T1 and T2 now. T3 starts when both finish. You'll get a gateway ping when T4 completes. Use the dashboard or `hermes kanban tail <id>` to follow along.

## 흔한 패턴

**Fan-out + fan-in (research → synthesize):** parent 없는 `researcher` task N개와, 그것들을 모두 parent로 가진 `analyst` task 1개.

**게이트가 있는 pipeline:** `pm → backend-eng → reviewer`. 각 단계는 `parents=[previous_task]`로 연결. reviewer는 block 또는 complete를 수행하고, reviewer가 block하면 operator가 feedback과 함께 unblock해서 다시 spawn합니다.

**동일 profile queue:** 예를 들어 task 50개가 모두 `translator`에게 assign되고 dependency가 없다면, dispatcher가 이를 직렬화합니다. translator는 priority 순서대로 처리하면서 자기 memory에 경험을 축적합니다.

**Human-in-the-loop:** 어떤 task든 `kanban_block()`으로 입력 대기 상태가 될 수 있습니다. `/unblock` 이후 dispatcher가 다시 spawn합니다. comment thread가 전체 컨텍스트를 운반합니다.

## Pitfalls

**재할당 vs. 새 task 생성.** reviewer가 "needs changes"로 block했다면, reviewer task에서 이어지는 **새 task를 만들어야지**, 같은 task를 다시 엄하게 쳐다보며 재실행하면 안 됩니다. 새 task는 원래 구현자 profile에게 assign하세요.

**link 인자 순서.** `kanban_link(parent_id=..., child_id=...)` — parent가 먼저입니다. 순서를 뒤집으면 엉뚱한 task가 `todo`로 내려갈 수 있습니다.

**중간 결과에 따라 graph 모양이 달라질 수 있다면 전체 graph를 미리 만들지 마세요.** T3 구조가 T1/T2 findings에 따라 달라진다면, T3를 "synthesize findings" task로만 두고 그 task의 첫 단계에서 부모 handoff를 읽어 후속 계획을 짜게 하면 됩니다. orchestrator는 또 다른 orchestrator를 spawn할 수 있습니다.

**Tenant 상속.** env에 `HERMES_TENANT`가 설정되어 있다면, 모든 `kanban_create` 호출에 `tenant=os.environ.get("HERMES_TENANT")`를 넣어 child task도 같은 namespace에 머물게 하세요.
