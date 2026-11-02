---
title: "계획 — .hermes/plans/에 마크다운 계획 작성; 실행하지 않음"
sidebar_label: "계획"
description: ".hermes/plans/에 마크다운 계획 작성; 실행하지 않음"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 skill의 SKILL.md에서 자동 생성합니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# 계획

.hermes/plans/에 마크다운 계획을 작성하며 실행은 하지 않습니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 번들됨 (기본 설치) |
| 경로 | `skills/software-development/plan` |
| 버전 | `2.0.0` |
| 작성자 | Hermes Agent (obra/superpowers에서 글쓰기 기법을 적용) |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `planning`, `plan-mode`, `implementation`, `workflow`, `design`, `documentation` |
| 관련 스킬 | [`subagent-driven-development`](/docs/user-guide/skills/optional/software-development/software-development-subagent-driven-development), [`test-driven-development`](/docs/user-guide/skills/bundled/software-development/software-development-test-driven-development), [`requesting-code-review`](/docs/user-guide/skills/bundled/software-development/software-development-requesting-code-review) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 내용입니다.
:::

# 계획 모드

사용자가 실행 대신 계획을 원할 때 이 스킬을 사용합니다.

## 핵심 동작

이번 턴에는 계획만 수립합니다.

- 코드를 구현하지 않습니다.
- 계획 마크다운 파일을 제외하고 프로젝트 파일을 편집하지 않습니다.
- 변경을 일으키는 터미널 명령을 실행하거나, 커밋·푸시를 하거나, 외부 작업을 수행하지 않습니다.
- 필요하다면 읽기 전용 명령이나 도구로 저장소 또는 기타 컨텍스트를 검사할 수 있습니다.
- 결과물은 활성 작업 공간의 `.hermes/plans/` 아래에 저장되는 마크다운 계획입니다.

## 출력 요구사항

구체적이고 실행 가능한 마크다운 계획을 작성합니다.

관련이 있다면 다음을 포함합니다.
- 목표
- 현재 컨텍스트 / 가정
- 제안하는 접근 방식
- 단계별 계획
- 변경 가능성이 있는 파일
- 테스트 / 검증
- 위험 요소, 트레이드오프 및 미해결 질문

코드 관련 작업이라면 정확한 파일 경로, 예상 테스트 대상, 검증 단계를 포함합니다.

## 저장 위치

다음 경로에 `write_file`을 사용해 마크다운 계획을 저장합니다.
- `.hermes/plans/YYYY-MM-DD_HHMMSS-<slug>.md`

이 경로는 활성 작업 디렉터리/백엔드 작업 공간을 기준으로 하는 상대 경로로 취급합니다. Hermes 파일 도구는 백엔드를 인식하므로 이 상대 경로를 사용하면 로컬, docker, ssh, modal, daytona 백엔드에서도 계획이 작업 공간에 저장됩니다.

런타임에서 특정 대상 경로를 제공한다면 해당 정확한 경로를 사용합니다.
그렇지 않다면 `.hermes/plans/` 아래에 합리적인 타임스탬프 기반 파일 이름을 직접 만듭니다.

## 상호작용 방식

- 요청이 충분히 명확하면 바로 계획을 작성합니다.
- `/plan`에 명시적 지시가 없다면 현재 대화 컨텍스트에서 작업을 추론합니다.
- 실제로 구체화되지 않은 부분이 있다면 추측하지 말고 짧은 확인 질문을 합니다.
- 계획을 저장한 후 무엇을 계획했는지와 저장 경로를 간단히 답합니다.

---

# 계획을 잘 작성하는 방법

다음 내용은 좋은 구현 계획을 작성하는 기술, 즉 구현자가 계획 안에 넣어야 할 내용입니다.

## 개요

코드베이스에 대한 컨텍스트가 전혀 없고 취향이 의심스러운 구현자를 가정해 포괄적인 구현 계획을 작성합니다. 구현에 필요한 작업 대상 파일, 완전한 코드, 테스트 명령, 확인할 문서를 모두 기록합니다. 작업을 작은 단위로 나눕니다. DRY. YAGNI. TDD. 자주 커밋합니다.

구현자는 숙련된 개발자이지만 도구 모음과 문제 영역에 대해서는 거의 아무것도 모르며, 좋은 테스트 설계도 잘 모른다고 가정합니다.

**핵심 원칙:** 좋은 계획은 구현을 자명하게 만듭니다. 누군가 추측해야 한다면 계획은 불완전합니다.

## 전체 구현 계획이 유용한 경우

**항상 다음 작업 전에 사용합니다.**
- 여러 단계로 이루어진 기능 구현
- 복잡한 요구사항 세분화
- subagent-driven-development를 통한 하위 에이전트 위임

**다음과 같은 경우에도 생략하지 않습니다.**
- 기능이 단순해 보이는 경우 (가정이 버그를 만듭니다)
- 직접 구현할 계획인 경우 (미래의 자신에게도 지침이 필요합니다)
- 혼자 작업하는 경우 (문서화는 중요합니다)

## 작은 작업 단위의 세분화

**각 작업 = 2~5분 동안 집중해서 수행할 수 있는 작업**입니다.

각 단계는 하나의 동작입니다.
- "실패하는 테스트 작성" — 단계
- "실패하는지 확인하도록 실행" — 단계
- "테스트를 통과시키는 최소한의 코드 구현" — 단계
- "테스트를 실행하고 통과하는지 확인" — 단계
- "커밋" — 단계

**너무 큰 예:**
```markdown
### Task 1: Build authentication system
[50 lines of code across 5 files]
```

**적절한 예:**
```markdown
### Task 1: Create User model with email field
[10 lines, 1 file]

### Task 2: Add password hash field to User
[8 lines, 1 file]

### Task 3: Create password hashing utility
[15 lines, 1 file]
```

## 계획 문서 구조

### 헤더 (필수)

모든 계획은 반드시 다음으로 시작해야 합니다.

```markdown
# [Feature Name] Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** [One sentence describing what this builds]

**Architecture:** [2-3 sentences about approach]

**Tech Stack:** [Key technologies/libraries]

---
```
### 작업 구조

각 작업은 다음 형식을 따릅니다.

````markdown
### Task N: [Descriptive Name]

**Objective:** What this task accomplishes (one sentence)

**Files:**
- Create: `exact/path/to/new_file.py`
- Modify: `exact/path/to/existing.py:45-67` (line numbers if known)
- Test: `tests/path/to/test_file.py`

**Step 1: Write failing test**

```python
def test_specific_behavior():
    result = function(input)
    assert result == expected
```

**Step 2: Run test to verify failure**

Run: `pytest tests/path/test.py::test_specific_behavior -v`
Expected: FAIL — "function not defined"

**Step 3: Write minimal implementation**

```python
def function(input):
    return expected
```

**Step 4: Run test to verify pass**

Run: `pytest tests/path/test.py::test_specific_behavior -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/path/test.py src/path/file.py
git commit -m "feat: add specific feature"
```
````

## 작성 과정

### 1단계: 요구 사항 이해

다음 내용을 읽고 이해합니다.
- 기능 요구 사항
- 설계 문서 또는 사용자 설명
- 승인 기준
- 제약 사항

### 2단계: 코드베이스 탐색

Hermes 도구를 사용해 프로젝트를 파악합니다.

```python
# Understand project structure
search_files("*.py", target="files", path="src/")

# Look at similar features
search_files("similar_pattern", path="src/", file_glob="*.py")

# Check existing tests
search_files("*.py", target="files", path="tests/")

# Read key files
read_file("src/app.py")
```

### 3단계: 접근 방식 설계

다음 사항을 결정합니다.
- 아키텍처 패턴
- 파일 구성
- 필요한 의존성
- 테스트 전략

### 4단계: 작업 작성

다음 순서로 작업을 만듭니다.
1. 설정/인프라
2. 핵심 기능 (각 기능에 TDD 적용)
3. 예외 사례
4. 통합
5. 정리/문서화

### 5단계: 세부 정보 완성

각 작업에 다음 내용을 포함합니다.
- **정확한 파일 경로** (`config file`이 아니라 `src/config/settings.py`처럼 작성)
- **완전한 코드 예시** ("검증 추가"가 아니라 실제 코드를 작성)
- **예상 출력이 포함된 정확한 명령어**
- **작동을 입증하는 검증 단계**

### 6단계: 계획 검토

다음을 확인합니다.
- [ ] 작업이 순차적이고 논리적인가
- [ ] 각 작업이 작은 단위(2~5분)인가
- [ ] 파일 경로가 정확한가
- [ ] 코드 예시가 완전하고 복사해 바로 사용할 수 있는가
- [ ] 명령어가 정확한가
- [ ] 예상 출력이 있는가
- [ ] 누락된 맥락이 없는가
- [ ] DRY, YAGNI, TDD 원칙이 적용되었는가

## 원칙

### DRY (반복하지 않기)

**나쁜 예:** 여러 곳에 검증 로직을 복사해 붙여 넣습니다.
**좋은 예:** 검증 함수를 추출해 모든 곳에서 사용합니다.

### YAGNI (아직 필요하지 않은 것은 만들지 않기)

**나쁜 예:** 미래 요구 사항을 위한 "유연성"을 추가합니다.
**좋은 예:** 지금 필요한 것만 구현합니다.

```python
# Bad — YAGNI violation
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email
        self.preferences = {}  # Not needed yet!
        self.metadata = {}     # Not needed yet!

# Good — YAGNI
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email
```

### TDD (테스트 주도 개발)

코드를 생성하는 모든 작업에는 전체 TDD 주기를 포함합니다.
1. 실패하는 테스트 작성
2. 실패를 확인하기 위해 실행
3. 최소 구현 작성
4. 통과를 확인하기 위해 실행

자세한 내용은 `test-driven-development` 스킬을 참고하세요.

### 잦은 커밋

각 작업이 끝날 때마다 커밋합니다.
```bash
git add [files]
git commit -m "type: description"
```

## 흔한 실수

### 모호한 작업

**나쁜 예:** "인증 추가"
**좋은 예:** "email 및 password_hash 필드가 있는 User 모델 생성"

### 불완전한 코드

**나쁜 예:** "1단계: 검증 함수 추가"
**좋은 예:** "1단계: 검증 함수 추가"와 함께 완전한 함수 코드를 작성합니다.

### 검증 누락

**나쁜 예:** "3단계: 작동하는지 테스트"
**좋은 예:** `pytest tests/test_auth.py -v` 실행, 예상 결과: 3 passed

### 파일 경로 누락

**나쁜 예:** "모델 파일 생성"
**좋은 예:** `src/models/user.py` 생성

## 실행 인계

계획을 저장한 후 실행 방식을 제안합니다.

**"계획을 완료하고 저장했습니다. 이제 subagent-driven-development를 사용해 실행할 준비가 되었습니다. 각 작업마다 새로운 서브에이전트를 배정하고, 2단계 검토(사양 준수 후 코드 품질)를 진행하겠습니다. 계속 진행할까요?"**

실행할 때는 `subagent-driven-development` 스킬을 사용합니다.
- 각 작업마다 전체 맥락과 함께 새로운 `delegate_task`를 배정합니다.
- 각 작업 후 사양 준수 검토를 진행합니다.
- 사양 검토를 통과한 후 코드 품질 검토를 진행합니다.
- 두 검토가 모두 승인한 경우에만 진행합니다.

## 기억할 사항

```
Bite-sized tasks (2-5 min each)
Exact file paths
Complete code (copy-pasteable)
Exact commands with expected output
Verification steps
DRY, YAGNI, TDD
Frequent commits
```

**좋은 계획은 구현을 명확하게 만듭니다.**
