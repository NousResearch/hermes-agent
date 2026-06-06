---
title: "Kanban Worker — Hermes Kanban 워커를 위한 주의사항, 예제 및 엣지 케이스"
sidebar_label: "Kanban Worker"
description: "Hermes Kanban 워커를 위한 주의사항, 예제 및 엣지 케이스"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Kanban Worker

Hermes Kanban 워커를 위한 주의사항, 예제 및 엣지 케이스입니다. 라이프사이클 자체는 모든 워커의 시스템 프롬프트(agent/prompt_builder.py의 `KANBAN_GUIDANCE`)에 자동 주입됩니다. 이 스킬은 특정 시나리오에 대한 더 깊은 세부 정보가 필요할 때 로드하는 항목입니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | Bundled (기본 설치됨) |
| 경로 | `skills/devops/kanban-worker` |
| 버전 | `2.0.0` |
| 플랫폼 | linux, macos, windows |
| 태그 | `kanban`, `multi-agent`, `collaboration`, `workflow`, `pitfalls` |
| 관련 스킬 | [`kanban-orchestrator`](/docs/user-guide/skills/bundled/devops/devops-kanban-orchestrator) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이는 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# Kanban Worker — 주의사항 및 예제

> Hermes Kanban 디스패처가 `--skills kanban-worker`를 사용하여 워커를 생성했기 때문에 이 스킬이 표시됩니다 — 디스패치된 모든 워커에 자동으로 로드됩니다. **라이프사이클** (6단계: 파악(orient) → 작업(work) → 하트비트(heartbeat) → 블록/완료(block/complete))은 시스템 프롬프트에 자동 주입되는 `KANBAN_GUIDANCE` 블록에도 존재합니다. 이 스킬은 좋은 핸드오프 형태, 재시도 진단, 엣지 케이스와 같은 더 깊은 세부 정보를 담고 있습니다.

## 작업 공간 처리

작업 공간 종류(kind)에 따라 `$HERMES_KANBAN_WORKSPACE` 내부에서 행동하는 방식이 결정됩니다:

| 종류(Kind) | 설명 | 작업 방법 |
|---|---|---|
| `scratch` | 당신만을 위한 새로운 임시 디렉토리 | 자유롭게 읽고/씁니다; 작업이 보관(archived)될 때 가비지 컬렉션(GC)됩니다. |
| `dir:<path>` | 공유 영구 디렉토리 | 다른 실행 시 당신이 쓴 내용을 읽습니다. 오랫동안 유지되는 상태처럼 다루세요. 경로는 절대 경로임이 보장됩니다 (커널이 상대 경로를 거부합니다). |
| `worktree` | 확인된 경로에 있는 Git 워크트리 | `.git`이 없는 경우, 메인 저장소에서 먼저 `git worktree add <path> ${HERMES_KANBAN_BRANCH:-wt/$HERMES_KANBAN_TASK}`를 실행한 다음, cd로 이동하여 평소처럼 작업하세요. 여기에 작업을 커밋합니다. |

## 테넌트 격리 (Tenant isolation)

`$HERMES_TENANT`가 설정된 경우, 해당 작업은 테넌트 네임스페이스에 속합니다. 영구 메모리를 읽거나 쓸 때 컨텍스트가 다른 테넌트로 유출되지 않도록 메모리 항목 앞에 테넌트를 접두사로 붙이세요:

- 좋음: `business-a: Acme is our biggest customer`
- 나쁨 (유출됨): `Acme is our biggest customer`

## 좋은 요약 및 메타데이터 형태

`kanban_complete(summary=..., metadata=...)` 핸드오프는 하위 워커가 당신이 한 일을 읽는 방법입니다. 잘 작동하는 패턴들:

**코딩 작업:**
```python
kanban_complete(
    summary="속도 제한기(rate limiter) 출시 — 토큰 버킷, IP 폴백이 있는 user_id 키 사용, 14개 테스트 통과",
    metadata={
        "changed_files": ["rate_limiter.py", "tests/test_rate_limiter.py"],
        "tests_run": 14,
        "tests_passed": 14,
        "decisions": ["user_id 기본, 인증되지 않은 요청에 대한 IP 폴백"],
    },
)
```

**사람의 리뷰가 필요한 코딩 작업 (리뷰 필수):**

대부분의 코드를 변경하는 작업은 사람이 검토하기 전까지는 진정으로 *완료*된 것이 아닙니다. 완료하는 대신, `review-required: ` 접두사가 붙은 `reason`으로 블록(block)하여 대시보드에 리뷰가 필요한 항목으로 표시되도록 하세요. `kanban_block`은 사람이 읽을 수 있는 이유만 포함하므로 구조화된 메타데이터(변경된 파일, 테스트 수, diff/PR URL)는 먼저 댓글(comment)로 남기세요 — 댓글은 지속적인 주석 채널입니다. 리뷰어는 이를 승인하고 `hermes kanban unblock <id>`를 실행하거나(이렇게 하면 후속 작업을 위해 당신이 스레드와 함께 다시 스폰됨), 다른 댓글을 통해 수정을 요청합니다.

```python
import json

kanban_comment(
    body="review-required 핸드오프:\n" + json.dumps({
        "changed_files": ["rate_limiter.py", "tests/test_rate_limiter.py"],
        "tests_run": 14,
        "tests_passed": 14,
        "diff_path": "/path/to/worktree",  # 또는 푸시된 경우 PR URL
        "decisions": ["user_id 기본, 인증되지 않은 요청에 대한 IP 폴백"],
    }, indent=2),
)
kanban_block(
    reason="review-required: 속도 제한기 출시, 14/14 테스트 통과 — 병합하기 전에 user_id/IP 폴백 선택에 대한 검토 필요",
)
```

작업이 정말로 최종적인 경우에만 `kanban_complete`를 사용하세요 — 예: 한 줄 오타 수정, 기능에 영향을 주지 않는 문서 변경, 또는 결과물이 작성 문서 자체인 연구 작업.

**연구 작업:**
```python
kanban_complete(
    summary="3개의 경쟁 라이브러리 검토; 처리량은 vLLM 승, 지연 시간은 SGLang 승, 메모리 효율성은 Tensorrt-LLM 승",
    metadata={
        "sources_read": 12,
        "recommendation": "vLLM",
        "benchmarks": {"vllm": 1.0, "sglang": 0.87, "trtllm": 0.72},
    },
)
```

**리뷰 작업:**
```python
kanban_complete(
    summary="PR #123 리뷰 완료; 2개의 차단 문제 발견 (/search에 SQL 인젝션, /settings에 CSRF 누락)",
    metadata={
        "pr_number": 123,
        "findings": [
            {"severity": "critical", "file": "api/search.py", "line": 42, "issue": "원시 SQL 연결(concat)"},
            {"severity": "high", "file": "api/settings.py", "issue": "CSRF 미들웨어 누락"},
        ],
        "approved": False,
    },
)
```

다운스트림 파서(리뷰어, 수집기, 스케줄러)가 텍스트를 다시 읽지 않고도 사용할 수 있도록 `metadata`를 구성하세요.

## 직접 생성한 카드 요청(클레임)하기

실행 중 새 칸반 작업(`kanban_create`를 통해)을 생성한 경우, `kanban_complete`의 `created_cards`에 ID를 전달하세요. 커널은 각 ID가 존재하고 사용자의 프로필에 의해 생성되었는지 확인합니다; 존재하지 않는 ID가 있으면 잘못된 내용을 나열한 오류와 함께 완료가 차단되며, 거부된 시도는 작업의 이벤트 로그에 영구적으로 기록됩니다. **성공적인 `kanban_create` 반환 값에서 캡처한 ID만 나열하세요 — 절대로 본문에서 ID를 지어내거나, 이전 실행에서 가져와 붙여넣거나, 다른 워커가 생성한 카드의 ID를 요청하지 마세요.**

```python
# 좋음 — 반환 값을 캡처한 다음 요청합니다.
c1 = kanban_create(title="SQL 인젝션 교정", assignee="security-worker")
c2 = kanban_create(title="CSRF 미들웨어 수정", assignee="web-worker")

kanban_complete(
    summary="리뷰 완료; 두 발견에 대한 교정 작업 생성.",
    metadata={"pr_number": 123, "approved": False},
    created_cards=[c1["task_id"], c2["task_id"]],
)
```

```python
# 나쁨 — 반환 값을 캡처하지 않은 ID를 요청합니다.
kanban_complete(
    summary="교정 카드 t_a1b2c3d4, t_deadbeef를 생성했습니다",  # 환각(hallucinated)
    created_cards=["t_a1b2c3d4", "t_deadbeef"],                   # → 게이트에서 거부됨
)
```

`kanban_create` 호출이 실패하면 (예외, 도구 오류), 카드가 생성되지 않은 것입니다 — 이에 대한 존재하지 않는 ID를 포함하지 마세요. 생성을 다시 시도하거나, ID를 생략하고 요약에 실패를 언급하세요. 산문 스캔 단계는 당신의 자유 형식 요약에서 확인할 수 없는 `t_<hex>` 참조도 잡아냅니다; 이는 완료를 막지는 않지만 대시보드의 작업에 권고성 경고로 표시됩니다.

## 빨리 답변을 받을 수 있는 블록(Block) 이유

나쁨: `"stuck"` — 사람이 상황을 파악할 수 없습니다.

좋음: 필요한 구체적인 결정을 명시하는 한 문장. 더 긴 맥락은 댓글로 남기세요.

```python
kanban_comment(
    task_id=os.environ["HERMES_KANBAN_TASK"],
    body="전체 맥락: Cloudflare 헤더에서 사용자 IP를 가져오지만 일부 사용자는 수천 개의 피어가 있는 NAT 뒤에 있습니다. IP만으로 키를 설정하면 오탐지가 발생합니다.",
)
kanban_block(reason="속도 제한 키 선택: IP(단단하지만 NAT에 안전하지 않음) 아니면 user_id(인증 필요, 익명 엔드포인트 건너뜀)?")
```

블록 메시지는 대시보드 / 게이트웨이 알리미에 표시되는 내용입니다. 댓글은 사람이 작업을 열었을 때 읽는 더 깊은 맥락입니다.

## 보낼 가치가 있는 하트비트

좋은 하트비트는 진행 상황을 명시합니다: `"epoch 12/50, loss 0.31"`, `"scanned 1.2M/2.4M rows"`, `"uploaded 47/120 videos"`.

나쁜 하트비트: `"still working"`, 빈 노트, 1초 미만의 간격. 최대 몇 분 간격이어야 합니다; 약 2분 미만의 작업은 하트비트를 아예 생략하세요.

## 재시도 시나리오

작업을 열고 `kanban_show`가 하나 이상의 닫힌(closed) 실행이 포함된 `runs: [...]`를 반환하면 당신은 재시도 중인 것입니다. 이전 실행의 `outcome` / `summary` / `error`는 무엇이 제대로 작동하지 않았는지 알려줍니다. 같은 경로를 반복하지 마세요. 일반적인 재시도 진단:

- `outcome: "timed_out"` — 이전 시도가 `max_runtime_seconds`에 도달했습니다. 작업을 분할하거나 단축해야 할 수 있습니다.
- `outcome: "crashed"` — 메모리 부족(OOM) 또는 세그폴트. 메모리 사용량을 줄이세요.
- `outcome: "spawn_failed"` + `error: "..."` — 일반적으로 프로필 구성 문제(자격 증명 누락, 잘못된 PATH)입니다. 맹목적으로 재시도하는 대신 `kanban_block`을 통해 사람에게 물어보세요.
- `outcome: "reclaimed"` + `summary: "task archived..."` — 이전 실행 중에 운영자가 작업을 보관했습니다; 아마도 실행되어서는 안 될 것입니다. 상태를 주의 깊게 확인하세요.
- `outcome: "blocked"` — 이전 시도가 차단되었습니다; 지금쯤이면 스레드에 차단 해제 댓글이 있을 것입니다.

## 알림 라우팅

`~/.hermes/config.yaml`에 `notification_sources`를 추가하여 게이트웨이가 여러 프로필 간 칸반 작업 알림을 수신하도록 구성할 수 있습니다.
- `notification_sources: ['*']`는 모든 프로필의 구독을 수락합니다.
- `notification_sources: ['default', 'zilor-ppt']` 또는 `"default,zilor-ppt"`는 구독을 지정된 프로필로 제한합니다.
- 키를 생략하면 기본 동작(프로필 격리)이 유지됩니다.

## 절대 하지 말아야 할 것

- `kanban_create`의 대체품으로 `delegate_task`를 호출하지 마세요. `delegate_task`는 사용자의 실행 내에서 짧은 추론 하위 작업을 위한 것이고, `kanban_create`는 단일 API 루프보다 오래 지속되는 에이전트 간 핸드오프를 위한 것입니다.
- 사람에게 질문하기 위해 `clarify`를 호출하지 마세요. 당신은 헤드리스(headless) 상태로 실행 중입니다 — 대답해 줄 실제 사용자가 없습니다. 호출이 타임아웃되고(기본값 약 120초) 작업은 입력이 필요하다는 신호 없이 `running` 상태로 조용히 머무르게 됩니다. 대신 `kanban_comment` (맥락) + `kanban_block(reason=...)` (필요한 결정)을 사용하세요 — 그러면 작업이 보드에 차단된 것으로 표시되고, 운영자가 이를 보고 댓글로 답변과 함께 차단을 해제하면 당신이 스레드와 함께 다시 스폰됩니다.
- 작업 본문에서 지시하지 않는 한 `$HERMES_KANBAN_WORKSPACE` 외부의 파일을 수정하지 마세요.
- 자신에게 할당된 후속 작업을 만들지 마세요 — 적절한 전문가에게 할당하세요.
- 실제로 끝내지 않은 작업을 완료하지 마세요. 대신 차단하세요.

## 주의 사항 (Pitfalls)

**디스패치와 시작 사이에 작업 상태가 변경될 수 있습니다.** 디스패처가 작업을 할당한 시점과 프로세스가 실제로 부팅된 시점 사이에 작업이 차단되거나, 재할당되거나, 보관되었을 수 있습니다. 항상 `kanban_show`를 먼저 실행하세요. `blocked` 또는 `archived`라고 보고되면 중지하세요 — 당신은 실행되어서는 안 됩니다.

**작업 공간에 이전 아티팩트가 남아있을 수 있습니다.** 특히 `dir:` 및 `worktree` 작업 공간에는 이전 실행의 파일이 있을 수 있습니다. 댓글 스레드를 읽으세요 — 일반적으로 다시 실행하는 이유와 작업 공간이 어떤 상태인지 설명되어 있습니다.

**지침이 있는 경우 CLI에 의존하지 마세요.** `kanban_*` 도구는 모든 터미널 백엔드(Docker, Modal, SSH)에서 작동합니다. 터미널 도구에서의 `hermes kanban <verb>`는 해당 백엔드에 CLI가 설치되어 있지 않기 때문에 컨테이너화된 백엔드에서는 실패합니다. 확신이 서지 않을 때는 도구를 사용하세요.

## CLI 대체 수단 (스크립팅용)

모든 도구에는 인간 운영자와 스크립트를 위한 CLI 버전이 있습니다:
- `kanban_show` ↔ `hermes kanban show <id> --json`
- `kanban_complete` ↔ `hermes kanban complete <id> --summary "..." --metadata '{...}'`
- `kanban_block` ↔ `hermes kanban block <id> "reason"`
- `kanban_create` ↔ `hermes kanban create "title" --assignee <profile> [--parent <id>]`
- 등.

에이전트 내부에서는 도구를 사용하세요; CLI는 터미널에 있는 사람을 위해 존재합니다.
