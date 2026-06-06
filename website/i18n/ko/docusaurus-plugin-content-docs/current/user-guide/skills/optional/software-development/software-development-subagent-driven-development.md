---
title: "Subagent Driven Development — delegate_task 하위 에이전트를 통한 계획 실행 (2단계 리뷰)"
sidebar_label: "Subagent Driven Development"
description: "delegate_task 하위 에이전트를 통한 계획 실행 (2단계 리뷰)"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Subagent Driven Development

delegate_task 하위 에이전트(subagent)를 통해 구현 계획을 실행합니다 (2단계 리뷰 과정 포함).

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/software-development/subagent-driven-development`로 설치 |
| 경로 | `optional-skills/software-development/subagent-driven-development` |
| 버전 | `1.1.0` |
| 작성자 | Hermes Agent (obra/superpowers 기반으로 수정됨) |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `delegation`, `subagent`, `implementation`, `workflow`, `parallel` |
| 관련 스킬 | [`plan`](/docs/user-guide/skills/bundled/software-development/software-development-plan), [`requesting-code-review`](/docs/user-guide/skills/bundled/software-development/software-development-requesting-code-review), [`test-driven-development`](/docs/user-guide/skills/bundled/software-development/software-development-test-driven-development) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# 하위 에이전트 주도 개발 (Subagent-Driven Development)

## 개요

체계적인 2단계(two-stage) 리뷰 과정을 갖추고, 작업마다 새로운 하위 에이전트를 파견하여 구현 계획을 실행합니다.

**핵심 원칙:** 작업별 신규 하위 에이전트 + 2단계 리뷰 (명세 확인 후 품질 검토) = 높은 품질, 빠른 이터레이션(iteration).

## 언제 사용하나요

다음과 같은 경우 이 스킬을 사용하십시오:
- 구현 계획이 있을 때 (`plan` 스킬 또는 사용자의 요구사항으로부터 도출됨)
- 작업들이 대부분 독립적일 때
- 품질과 명세(spec) 준수가 중요할 때
- 작업들 사이에 자동화된 리뷰 과정을 원할 때

**수동 실행과의 비교:**
- 작업별로 새롭게 부여되는 맥락(context) (누적된 상태로 인한 혼란 방지)
- 자동화된 리뷰 프로세스로 문제를 조기에 발견
- 모든 작업에 걸쳐 일관된 품질 검사
- 하위 에이전트가 작업을 시작하기 전 질문할 수 있음

## 프로세스

### 1. 계획 읽기 및 분석 (Read and Parse Plan)

계획 파일을 읽습니다. 모든 작업과 그에 대한 전체 텍스트 및 맥락을 초기에 추출합니다. 다음과 같이 할 일 목록(todo list)을 만듭니다:

```python
# 계획 읽기
read_file("docs/plans/feature-plan.md")

# 모든 작업을 포함한 할 일 목록 생성
todo([
    {"id": "task-1", "content": "이메일 필드가 있는 User 모델 생성", "status": "pending"},
    {"id": "task-2", "content": "비밀번호 해싱 유틸리티 추가", "status": "pending"},
    {"id": "task-3", "content": "로그인 엔드포인트 생성", "status": "pending"},
])
```

**핵심:** 계획 파일은 단 한 번만 읽으십시오. 모든 것을 추출하세요. 하위 에이전트들에게 계획 파일을 직접 읽게 만들지 말고, 하위 에이전트의 컨텍스트(context)에 작업에 대한 전체 텍스트를 직접 제공하십시오.

### 2. 작업별 워크플로 (Per-Task Workflow)

계획의 각 작업(EACH task)에 대해:

#### 1단계: 구현 담당 하위 에이전트 파견 (Dispatch Implementer Subagent)

완전한 맥락을 담아 `delegate_task`를 사용합니다:

```python
delegate_task(
    goal="작업 1 구현: email과 password_hash 필드를 갖춘 User 모델 생성",
    context="""
    계획 내 작업 내용 (TASK FROM PLAN):
    - 생성: src/models/user.py
    - email (str) 및 password_hash (str) 필드가 있는 User 클래스 추가
    - 비밀번호 해싱에 bcrypt 사용
    - 디버깅을 위한 __repr__ 포함

    TDD 준수:
    1. tests/models/test_user.py 에 실패하는 테스트 작성
    2. 실행: pytest tests/models/test_user.py -v (FAIL 확인)
    3. 최소한의 구현 코드 작성
    4. 실행: pytest tests/models/test_user.py -v (PASS 확인)
    5. 실행: pytest tests/ -q (회귀 오류가 없는지 확인)
    6. 커밋: git add -A && git commit -m "feat: add User model with password hashing"

    프로젝트 맥락 (PROJECT CONTEXT):
    - Python 3.11, src/app.py 에 위치한 Flask 앱
    - src/models/ 에 존재하는 기존 모델들
    - 테스트는 pytest를 사용하며, 프로젝트 루트에서 실행
    - bcrypt는 requirements.txt에 이미 존재함
    """,
    toolsets=['terminal', 'file']
)
```

#### 2단계: 명세 준수 리뷰어 파견 (Dispatch Spec Compliance Reviewer)

구현 담당자가 완료하면 원래 명세와 비교하여 검증합니다:

```python
delegate_task(
    goal="구현 결과가 계획의 명세(spec)와 일치하는지 리뷰",
    context="""
    원본 작업 명세 (ORIGINAL TASK SPEC):
    - User 클래스를 포함한 src/models/user.py 생성
    - 필드: email (str), password_hash (str)
    - 비밀번호 해싱에 bcrypt 사용
    - __repr__ 포함

    확인 사항 (CHECK):
    - [ ] 명세의 모든 요구사항이 구현되었는가?
    - [ ] 파일 경로가 명세와 일치하는가?
    - [ ] 함수 시그니처가 명세와 일치하는가?
    - [ ] 동작이 예상과 일치하는가?
    - [ ] 불필요하게 추가된 것은 없는가 (범위 초과 방지)?

    출력: 통과 시 PASS, 불일치할 경우 수정해야 할 구체적인 누락 항목(gaps) 목록.
    """,
    toolsets=['file']
)
```

**명세 관련 문제가 발견된 경우:** 누락된 부분을 수정하게 한 뒤 명세 리뷰를 다시 실행합니다. 명세 준수를 완벽히 통과해야만 다음으로 넘어갑니다.

#### 3단계: 코드 품질 리뷰어 파견 (Dispatch Code Quality Reviewer)

명세 준수 리뷰를 통과한 후:

```python
delegate_task(
    goal="작업 1 구현 결과의 코드 품질 리뷰",
    context="""
    리뷰할 파일 (FILES TO REVIEW):
    - src/models/user.py
    - tests/models/test_user.py

    확인 사항 (CHECK):
    - [ ] 프로젝트의 규칙과 코드 스타일을 따르는가?
    - [ ] 적절한 에러 처리가 되어있는가?
    - [ ] 변수/함수 이름이 명확한가?
    - [ ] 테스트 커버리지가 충분한가?
    - [ ] 명백한 버그나 놓친 엣지 케이스가 없는가?
    - [ ] 보안 문제는 없는가?

    출력 형식 (OUTPUT FORMAT):
    - 치명적인 문제 (Critical Issues): [진행 전 반드시 수정해야 함]
    - 중요한 문제 (Important Issues): [수정 권장]
    - 사소한 문제 (Minor Issues): [선택 사항]
    - 판정 (Verdict): APPROVED (승인) 또는 REQUEST_CHANGES (수정 요청)
    """,
    toolsets=['file']
)
```

**품질 문제가 발견된 경우:** 문제를 수정하고 다시 리뷰합니다. 승인(APPROVED)될 때만 다음으로 넘어갑니다.

#### 4단계: 완료 표시 (Mark Complete)

```python
todo([{"id": "task-1", "content": "이메일 필드가 있는 User 모델 생성", "status": "completed"}], merge=True)
```

### 3. 최종 리뷰 (Final Review)

모든 작업이 완료된 후, 최종 통합 리뷰어를 파견합니다:

```python
delegate_task(
    goal="전체 구현의 일관성과 통합 문제가 없는지 리뷰",
    context="""
    계획의 모든 작업이 완료되었습니다. 전체 구현 내역을 리뷰하십시오:
    - 모든 구성요소가 함께 잘 작동하는가?
    - 작업들 사이에 일관성이 없는 부분이 있는가?
    - 모든 테스트가 통과하는가?
    - 머지(merge)할 준비가 되었는가?
    """,
    toolsets=['terminal', 'file']
)
```

### 4. 검증 및 커밋 (Verify and Commit)

```bash
# 전체 테스트 스위트 실행
pytest tests/ -q

# 모든 변경 사항 리뷰
git diff --stat

# 필요한 경우 최종 커밋
git add -A && git commit -m "feat: complete [기능 이름] implementation"
```

## 작업 크기 단위 (Task Granularity)

**각 작업 = 2~5분의 집중된 작업 분량.**

**너무 큰 작업:**
- "사용자 인증 시스템 구현"

**적절한 크기:**
- "이메일과 비밀번호 필드가 있는 User 모델 생성"
- "비밀번호 해싱 함수 추가"
- "로그인 엔드포인트 생성"
- "JWT 토큰 생성 추가"
- "회원가입 엔드포인트 생성"

## 금지 사항 (Red Flags — Never Do These)

- 계획 없이 구현 시작하기
- 리뷰 건너뛰기 (명세 준수 또는 코드 품질 둘 중 하나라도)
- 치명적/중요한 문제를 수정하지 않은 채 다음으로 넘어가기
- 동일한 파일을 수정하는 작업들에 대해 여러 구현 담당 하위 에이전트를 동시에 파견하기
- 하위 에이전트가 직접 계획 파일을 읽게 만들기 (대신 컨텍스트에 전체 텍스트를 제공하세요)
- 배경 지식 설정 건너뛰기 (하위 에이전트는 해당 작업이 시스템 어디에 속하는지 이해해야 함)
- 하위 에이전트의 질문 무시하기 (계속 진행하게 내버려두지 말고 답변을 먼저 제공할 것)
- 명세 준수에서 "이 정도면 대충 맞지"라며 타협하기
- 리뷰 루프 건너뛰기 (리뷰어가 문제 발견 → 구현 담당자가 수정 → 다시 리뷰)
- 구현 담당자의 셀프 리뷰로 실제 리뷰를 대체하기 (둘 다 필요함)
- **명세 준수가 통과(PASS)되기 전에 코드 품질 리뷰 시작하기** (순서가 잘못됨)
- 어느 쪽 리뷰든 열려있는 문제가 남아있는데 다음 작업으로 넘어가기

## 문제 해결 (Handling Issues)

### 하위 에이전트가 질문을 하는 경우

- 명확하고 완전하게 대답하십시오.
- 필요한 경우 추가적인 맥락을 제공하십시오.
- 성급하게 구현을 서두르도록 재촉하지 마십시오.

### 리뷰어가 문제를 발견한 경우

- 구현 담당 하위 에이전트(또는 새로운 에이전트)가 이를 수정합니다.
- 리뷰어가 다시 검토합니다.
- 승인될 때까지 반복합니다.
- 재리뷰(re-review)를 건너뛰지 마십시오.

### 하위 에이전트가 작업에 실패한 경우

- 무엇이 잘못되었는지에 대한 구체적인 지시사항을 포함하여 새로운 수정 전담 하위 에이전트를 파견하십시오.
- 컨트롤러 세션(메인 에이전트)에서 수동으로 고치려 하지 마십시오 (맥락 오염의 원인이 됩니다).

## 효율성 노트 (Efficiency Notes)

**왜 작업마다 새로운 하위 에이전트를 사용하는가:**
- 누적된 상태(state)로 인해 맥락이 오염되는 것을 방지합니다.
- 각 하위 에이전트가 깨끗하고 집중된 맥락을 부여받습니다.
- 이전 작업들의 코드나 추론 과정으로 인해 헷갈리는 일이 없습니다.

**왜 2단계 리뷰를 하는가:**
- 명세 리뷰는 부족하거나 과도한 구현을 초기에 잡아냅니다.
- 품질 리뷰는 구현이 잘 구축되었는지를 보장합니다.
- 여러 작업으로 복합되기 전에 미리 문제를 잡아냅니다.

**비용 트레이드오프:**
- 하위 에이전트 호출 횟수가 늘어납니다 (작업당 구현 담당자 1 + 리뷰어 2).
- 하지만 조기에 문제를 잡아냅니다 (나중에 겹겹이 꼬인 문제를 디버깅하는 것보다 훨씬 저렴함).

## 다른 스킬과의 통합

### plan (계획) 과 함께

이 스킬은 `plan` 스킬이 생성한 계획을 **실행(EXECUTE)**합니다:
1. 사용자 요구사항 → 계획(plan) → 구현 계획
2. 구현 계획 → 하위 에이전트 주도 개발(subagent-driven-development) → 작동하는 코드

### test-driven-development (테스트 주도 개발) 와 함께

구현 담당 하위 에이전트는 TDD를 따라야 합니다:
1. 실패하는 테스트 먼저 작성
2. 최소한의 코드 구현
3. 테스트 통과 확인
4. 커밋

모든 구현 담당자의 맥락(context)에 TDD 지시사항을 포함시키십시오.

### requesting-code-review (코드 리뷰 요청) 와 함께

이 2단계 리뷰 과정 자체가 바로 코드 리뷰입니다. 최종 통합 리뷰 시에는 requesting-code-review 스킬의 리뷰 관점(dimensions)들을 사용하십시오.

### systematic-debugging (체계적 디버깅) 과 함께

하위 에이전트가 구현 중 버그를 만났다면:
1. systematic-debugging 프로세스를 따름
2. 수정 전 근본 원인(root cause)을 찾음
3. 회귀 테스트(regression test) 작성
4. 다시 구현 재개

## 워크플로 예시

```
[계획 읽기: docs/plans/auth-feature.md]
[5개의 작업이 있는 할 일 목록 생성]

--- 작업 1: User 모델 생성 ---
[구현 담당 하위 에이전트 파견]
  구현자: "이메일은 고유해야 합니까?"
  당신: "네, 이메일은 반드시 고유해야 합니다"
  구현자: 구현됨, 테스트 3/3 통과, 커밋 완료.

[명세 리뷰어 파견]
  명세 리뷰어: ✅ PASS — 모든 요구사항 충족

[품질 리뷰어 파견]
  품질 리뷰어: ✅ APPROVED — 깔끔한 코드, 좋은 테스트

[작업 1 완료 표시]

--- 작업 2: 비밀번호 해싱 ---
[구현 담당 하위 에이전트 파견]
  구현자: 질문 없음, 구현됨, 테스트 5/5 통과.

[명세 리뷰어 파견]
  명세 리뷰어: ❌ 누락됨: 비밀번호 강도 검증 (명세에 "최소 8자" 명시)

[구현자 수정]
  구현자: 검증 로직 추가, 테스트 7/7 통과.

[명세 리뷰어 다시 파견]
  명세 리뷰어: ✅ PASS

[품질 리뷰어 파견]
  품질 리뷰어: 중요: 매직 넘버 8 사용됨, 상수로 분리할 것
  구현자: MIN_PASSWORD_LENGTH 상수로 분리
  품질 리뷰어: ✅ APPROVED

[작업 2 완료 표시]

... (모든 작업에 대해 계속 진행)

[모든 작업 완료 후: 최종 통합 리뷰어 파견]
[전체 테스트 스위트 실행: 모두 통과]
[완료!]
```

## 요약 (Remember)

```
작업마다 항상 새로운 하위 에이전트
매번 2단계 리뷰 필수
명세 준수가 첫 번째 (FIRST)
코드 품질이 두 번째 (SECOND)
절대 리뷰를 건너뛰지 말 것
문제를 조기에 발견할 것
```

**품질은 우연히 만들어지지 않습니다. 체계적인 과정의 결과입니다.**

## 추가 읽을거리 (관련될 때만 로드)

오케스트레이션에서 상당량의 맥락을 사용하거나, 긴 리뷰 루프를 돌거나, 복잡한 검증 체크포인트를 설정할 때 해당 분야와 관련된 다음 참조 문서를 로드하십시오:

- **`references/context-budget-discipline.md`** — 4단계 맥락 저하 모델 (PEAK / GOOD / DEGRADING / POOR), 맥락 크기에 비례하는 읽기 깊이 규칙 및 조용한 저하의 조기 경고 징후. 실행 시 많은 맥락을 소비할 것이 확실한 경우(다단계 계획, 많은 하위 에이전트 파견, 큰 결과물) 로드하십시오.
- **`references/gates-taxonomy.md`** — 4가지 표준 관문 유형(사전 점검, 수정, 에스컬레이션, 중단)과 그에 따른 행동, 복구 및 예시. 검증 체크포인트가 있는 모든 워크플로를 설계하거나 검토할 때 이 용어를 명시적으로 사용하여 각 관문에 진입 조건, 실패 시 행동, 재개 규칙을 정의하십시오.

두 참조 문서 모두 gsd-build/get-shit-done (MIT © 2025 Lex Christopherson)에서 각색되었습니다.
