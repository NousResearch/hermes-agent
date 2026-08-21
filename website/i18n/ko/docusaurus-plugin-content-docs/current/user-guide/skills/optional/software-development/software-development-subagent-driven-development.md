---
title: "서브에이전트 기반 개발 — delegate_task 서브에이전트를 통한 계획 실행(2단계 검토)"
sidebar_label: "서브에이전트 기반 개발"
description: "delegate_task 서브에이전트를 통한 계획 실행(2단계 검토)"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# 서브에이전트 기반 개발

delegate_task 서브에이전트를 통해 계획을 실행합니다(2단계 검토).

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 선택 사항 — `hermes skills install official/software-development/subagent-driven-development`로 설치 |
| 경로 | `optional-skills/software-development/subagent-driven-development` |
| 버전 | `1.1.0` |
| 작성자 | Hermes Agent (obra/superpowers에서 조정) |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `delegation`, `subagent`, `implementation`, `workflow`, `parallel` |
| 관련 스킬 | [`plan`](/docs/user-guide/skills/bundled/software-development/software-development-plan), [`requesting-code-review`](/docs/user-guide/skills/bundled/software-development/software-development-requesting-code-review), [`test-driven-development`](/docs/user-guide/skills/bundled/software-development/software-development-test-driven-development) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성 상태일 때 에이전트가 보게 되는 지침입니다.
:::

# 서브에이전트 기반 개발

## 개요

각 작업마다 새로운 서브에이전트를 배정하고 체계적인 2단계 검토를 진행하여 구현 계획을 실행합니다.

**핵심 원칙:** 작업마다 새로운 서브에이전트를 사용하고, 사양 검토와 품질 검토의 2단계 검토를 거치면 높은 품질을 빠르게 달성할 수 있습니다.

## 사용 시점

다음과 같은 경우 이 스킬을 사용합니다.
- 구현 계획이 있을 때(`plan` 스킬 또는 사용자 요구사항으로 작성)
- 작업이 대부분 서로 독립적일 때
- 품질과 사양 준수가 중요할 때
- 작업 사이에 자동화된 검토를 원할 때

**수동 실행과 비교할 때:**
- 작업마다 새로운 컨텍스트를 사용합니다(누적된 상태로 인한 혼동 방지).
- 자동 검토 프로세스가 문제를 조기에 발견합니다.
- 작업 전반에 일관된 품질 검사를 적용합니다.
- 서브에이전트가 작업 사이에 질문할 수 있습니다.

## 프로세스

### 1. 계획 읽기 및 분석

계획 파일을 읽습니다. 모든 작업을 전체 텍스트와 컨텍스트와 함께 미리 추출합니다. 할 일 목록을 만듭니다.

```python
# Read the plan
read_file("docs/plans/feature-plan.md")

# Create todo list with all tasks
todo([
    {"id": "task-1", "content": "Create User model with email field", "status": "pending"},
    {"id": "task-2", "content": "Add password hashing utility", "status": "pending"},
    {"id": "task-3", "content": "Create login endpoint", "status": "pending"},
])
```

**중요:** 계획을 한 번만 읽습니다. 모든 내용을 추출합니다. 서브에이전트에게 계획 파일을 읽게 하지 말고, 전체 작업 텍스트를 컨텍스트에 직접 제공합니다.

### 2. 작업별 워크플로

계획의 각 작업에 대해 다음을 수행합니다.

#### 1단계: 구현 서브에이전트 배정

전체 컨텍스트와 함께 `delegate_task`를 사용합니다.

```python
delegate_task(
    goal="Implement Task 1: Create User model with email and password_hash fields",
    context="""
    TASK FROM PLAN:
    - Create: src/models/user.py
    - Add User class with email (str) and password_hash (str) fields
    - Use bcrypt for password hashing
    - Include __repr__ for debugging

    FOLLOW TDD:
    1. Write failing test in tests/models/test_user.py
    2. Run: pytest tests/models/test_user.py -v (verify FAIL)
    3. Write minimal implementation
    4. Run: pytest tests/models/test_user.py -v (verify PASS)
    5. Run: pytest tests/ -q (verify no regressions)
    6. Commit: git add -A && git commit -m "feat: add User model with password hashing"

    PROJECT CONTEXT:
    - Python 3.11, Flask app in src/app.py
    - Existing models in src/models/
    - Tests use pytest, run from project root
    - bcrypt already in requirements.txt
    """,
    toolsets=['terminal', 'file']
)
```

#### 2단계: 사양 준수 검토 서브에이전트 배정

구현 서브에이전트가 완료되면 원래 사양에 맞는지 확인합니다.

```python
delegate_task(
    goal="Review if implementation matches the spec from the plan",
    context="""
    ORIGINAL TASK SPEC:
    - Create src/models/user.py with User class
    - Fields: email (str), password_hash (str)
    - Use bcrypt for password hashing
    - Include __repr__

    CHECK:
    - [ ] All requirements from spec implemented?
    - [ ] File paths match spec?
    - [ ] Function signatures match spec?
    - [ ] Behavior matches expected?
    - [ ] Nothing extra added (no scope creep)?

    OUTPUT: PASS or list of specific spec gaps to fix.
    """,
    toolsets=['file']
)
```

**사양 문제가 발견된 경우:** 누락된 부분을 수정한 다음 사양 검토를 다시 실행합니다. 사양을 준수할 때까지 계속합니다.

#### 3단계: 코드 품질 검토 서브에이전트 배정

사양 준수 검토를 통과한 후 코드 품질을 검토합니다.

```python
delegate_task(
    goal="Review code quality for Task 1 implementation",
    context="""
    FILES TO REVIEW:
    - src/models/user.py
    - tests/models/test_user.py

    CHECK:
    - [ ] Follows project conventions and style?
    - [ ] Proper error handling?
    - [ ] Clear variable/function names?
    - [ ] Adequate test coverage?
    - [ ] No obvious bugs or missed edge cases?
    - [ ] No security issues?

    OUTPUT FORMAT:
    - Critical Issues: [must fix before proceeding]
    - Important Issues: [should fix]
    - Minor Issues: [optional]
    - Verdict: APPROVED or REQUEST_CHANGES
    """,
    toolsets=['file']
)
```

**품질 문제가 발견된 경우:** 문제를 수정하고 다시 검토합니다. 승인될 때까지 계속합니다.

#### 4단계: 완료로 표시

```python
todo([{"id": "task-1", "content": "Create User model with email field", "status": "completed"}], merge=True)
```

### 3. 최종 검토

모든 작업이 완료되면 최종 통합 검토 서브에이전트를 배정합니다.

```python
delegate_task(
    goal="Review the entire implementation for consistency and integration issues",
    context="""
    All tasks from the plan are complete. Review the full implementation:
    - Do all components work together?
    - Any inconsistencies between tasks?
    - All tests passing?
    - Ready for merge?
    """,
    toolsets=['terminal', 'file']
)
```

### 4. 검증 및 커밋

```bash
# Run full test suite
pytest tests/ -q

# Review all changes
git diff --stat

# Final commit if needed
git add -A && git commit -m "feat: complete [feature name] implementation"
```

## 작업 세분화

**각 작업은 2~5분 동안 집중해서 처리할 수 있는 크기여야 합니다.**

**너무 큰 예:**
- "사용자 인증 시스템 구현"

**적절한 크기:**
- "이메일 및 비밀번호 필드를 포함한 User 모델 생성"
- "비밀번호 해싱 유틸리티 추가"
- "로그인 엔드포인트 생성"
- "JWT 토큰 생성 추가"
- "회원가입 엔드포인트 생성"

## 위험 신호 — 절대 하지 말아야 할 일

- 계획 없이 구현을 시작하지 않습니다.
- 검토를 건너뛰지 않습니다(사양 준수 또는 코드 품질 중 어느 하나라도).
- 해결되지 않은 중요하거나 치명적인 문제를 안고 진행하지 않습니다.
- 같은 파일을 수정하는 작업에 여러 구현 서브에이전트를 배정하지 않습니다.
- 서브에이전트가 계획 파일을 읽게 하지 않습니다(전체 작업 내용을 컨텍스트에 직접 제공합니다).
- 상황 설명을 생략하지 않습니다(서브에이전트는 작업이 전체에서 어떤 위치에 있는지 알아야 합니다).
- 서브에이전트의 질문을 무시하지 않습니다(진행하게 하기 전에 답변합니다).
- "대략 맞는 수준"을 사양 준수로 받아들이지 않습니다.
- 검토 루프를 건너뛰지 않습니다(검토자가 문제를 발견하면 구현자가 수정하고 다시 검토합니다).
- 구현자 스스로의 검토로 실제 검토를 대체하지 않습니다(둘 다 필요합니다).
- 사양 준수 검토가 PASS가 되기 전에 코드 품질 검토를 시작하지 않습니다(순서가 잘못되었습니다).
- 어느 한 검토라도 미해결 문제가 있는 상태에서 다음 작업으로 넘어가지 않습니다.

## 문제 처리

### 서브에이전트가 질문하는 경우

- 명확하고 완전하게 답변합니다.
- 필요한 경우 추가 컨텍스트를 제공합니다.
- 구현을 서두르도록 재촉하지 않습니다.

### 검토자가 문제를 발견한 경우

- 구현 서브에이전트(또는 새로운 서브에이전트)가 문제를 수정합니다.
- 검토자가 다시 검토합니다.
- 승인될 때까지 반복합니다.
- 재검토를 건너뛰지 않습니다.

### 서브에이전트가 작업에 실패한 경우

- 무엇이 잘못되었는지 구체적으로 설명하여 새로운 수정 서브에이전트를 배정합니다.
- 컨트롤러 세션에서 직접 수정하려 하지 않습니다(컨텍스트 오염 방지).

## 효율성 참고 사항

**작업마다 새로운 서브에이전트를 사용하는 이유:**
- 각 서브에이전트가 깨끗하고 집중된 컨텍스트를 얻습니다.
- 이전 작업의 코드나 추론으로 인한 혼동이 없습니다.

**2단계 검토를 사용하는 이유:**
- 사양 검토가 과소 구현 또는 과도한 구현을 조기에 발견합니다.
- 품질 검토가 구현 품질을 보장합니다.
- 문제가 여러 작업에 걸쳐 누적되기 전에 발견합니다.

**비용 절충:**
- 서브에이전트 호출이 더 많이 필요합니다(작업마다 구현자 + 검토자 2명).
- 하지만 문제를 조기에 발견하므로 나중에 문제가 누적된 뒤 디버깅하는 것보다 비용이 적습니다.

## 다른 스킬과의 통합

### plan과 함께 사용

이 스킬은 `plan` 스킬로 작성된 계획을 실행합니다.
1. 사용자 요구사항 → 계획 → 구현 계획
2. 구현 계획 → 서브에이전트 기반 개발 → 작동하는 코드

### test-driven-development와 함께 사용

구현 서브에이전트는 TDD를 따라야 합니다.
1. 먼저 실패하는 테스트를 작성합니다.
2. 최소한의 코드를 구현합니다.
3. 테스트가 통과하는지 확인합니다.
4. 커밋합니다.

모든 구현자 컨텍스트에 TDD 지침을 포함합니다.

### requesting-code-review와 함께 사용

2단계 검토 프로세스 자체가 코드 검토입니다. 최종 통합 검토에는 requesting-code-review 스킬의 검토 기준을 사용합니다.

### systematic-debugging과 함께 사용

구현 중 서브에이전트가 버그를 발견하면 다음을 수행합니다.
1. systematic-debugging 프로세스를 따릅니다.
2. 수정하기 전에 근본 원인을 찾습니다.
3. 회귀 테스트를 작성합니다.
4. 구현을 재개합니다.

## 워크플로 예시

```
[Read plan: docs/plans/auth-feature.md]
[Create todo list with 5 tasks]

--- Task 1: Create User model ---
[Dispatch implementer subagent]
  Implementer: "Should email be unique?"
  You: "Yes, email must be unique"
  Implementer: Implemented, 3/3 tests passing, committed.

[Dispatch spec reviewer]
  Spec reviewer: ✅ PASS — all requirements met

[Dispatch quality reviewer]
  Quality reviewer: ✅ APPROVED — clean code, good tests

[Mark Task 1 complete]

--- Task 2: Password hashing ---
[Dispatch implementer subagent]
  Implementer: No questions, implemented, 5/5 tests passing.

[Dispatch spec reviewer]
  Spec reviewer: ❌ Missing: password strength validation (spec says "min 8 chars")

[Implementer fixes]
  Implementer: Added validation, 7/7 tests passing.

[Dispatch spec reviewer again]
  Spec reviewer: ✅ PASS

[Dispatch quality reviewer]
  Quality reviewer: Important: Magic number 8, extract to constant
  Implementer: Extracted MIN_PASSWORD_LENGTH constant
  Quality reviewer: ✅ APPROVED

[Mark Task 2 complete]

... (continue for all tasks)

[After all tasks: dispatch final integration reviewer]
[Run full test suite: all passing]
[Done!]
```

## 기억할 사항

```
Fresh subagent per task
Two-stage review every time
Spec compliance FIRST
Code quality SECOND
Never skip reviews
Catch issues early
```

**품질은 우연히 생기지 않습니다. 체계적인 프로세스의 결과입니다.**

## 추가 읽을거리(관련 있을 때 로드)

오케스트레이션에서 상당한 컨텍스트 사용량, 긴 검토 루프 또는 복잡한 검증 체크포인트가 발생하는 경우, 해당 규율에 맞는 다음 참고 자료를 로드합니다.

- **`references/context-budget-discipline.md`** — PEAK / GOOD / DEGRADING / POOR의 4단계 컨텍스트 저하 모델, 컨텍스트 창 크기에 따라 확장되는 읽기 깊이 규칙, 조용한 저하의 초기 경고 신호입니다. 여러 단계의 계획, 많은 서브에이전트, 대규모 산출물 등으로 실행이 상당한 컨텍스트를 소비할 것이 분명한 경우 로드합니다.
- **`references/gates-taxonomy.md`** — 동작, 복구, 예시와 함께 네 가지 표준 게이트 유형(사전 점검, 수정, 에스컬레이션, 중단)을 정의합니다. 검증 체크포인트가 있는 워크플로를 설계하거나 검토할 때 로드하고, 각 게이트의 진입 조건, 실패 동작, 재개 규칙이 정의되도록 이 용어를 명시적으로 사용합니다.

두 참고 자료 모두 gsd-build/get-shit-done에서 조정되었습니다(MIT © 2025 Lex Christopherson).
