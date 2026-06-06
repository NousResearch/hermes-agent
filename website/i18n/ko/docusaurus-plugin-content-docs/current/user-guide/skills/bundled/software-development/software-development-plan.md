---
title: "Plan — 계획 모드: 실행 없이 실천 가능한 마크다운 계획 작성"
sidebar_label: "Plan"
description: "계획 모드: 실행 없이 실천 가능한 마크다운 계획 작성"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py에 의해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Plan

계획 모드: 실행 없이 실천 가능한 마크다운 계획을 `.hermes/plans/`에 작성. 잘게 쪼개진 작업, 정확한 경로, 완전한 코드.

## 스킬 메타데이터

| | |
|---|---|
| Source | Bundled (기본 설치됨) |
| Path | `skills/software-development/plan` |
| Version | `2.0.0` |
| Author | Hermes Agent (obra/superpowers에서 작성 기법 채택) |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `planning`, `plan-mode`, `implementation`, `workflow`, `design`, `documentation` |
| Related skills | [`subagent-driven-development`](/docs/user-guide/skills/optional/software-development/software-development-subagent-driven-development), [`test-driven-development`](/docs/user-guide/skills/bundled/software-development/software-development-test-driven-development), [`requesting-code-review`](/docs/user-guide/skills/bundled/software-development/software-development-requesting-code-review) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화될 때 에이전트가 지침으로 보는 내용입니다.
:::

# 계획 모드

사용자가 실행 대신 계획을 원할 때 이 스킬을 사용하세요.

## 핵심 동작

이번 턴에서는 오직 계획만 세웁니다.

- 코드를 구현하지 마세요.
- 계획 마크다운 파일을 제외한 프로젝트 파일을 수정하지 마세요.
- 터미널 변경 명령, commit, push 또는 외부 작업을 수행하지 마세요.
- 필요할 때 저장소나 기타 컨텍스트를 읽기 전용 명령/도구로 검사할 수 있습니다.
- 결과물은 활성 워크스페이스의 `.hermes/plans/` 아래에 저장되는 마크다운 계획입니다.

## 출력 요구 사항

구체적이고 실천 가능한 마크다운 계획을 작성하세요.

관련 있는 경우 다음을 포함하세요:
- 목표
- 현재 컨텍스트 / 가정
- 제안하는 접근 방식
- 단계별 계획
- 변경될 가능성이 있는 파일
- 테스트 / 검증
- 위험, 트레이드오프 및 미해결 질문

작업이 코드와 관련된 경우, 정확한 파일 경로, 예상되는 테스트 대상 및 검증 단계를 포함하세요.

## 저장 위치

`write_file`을 사용하여 다음 위치에 계획을 저장하세요:
- `.hermes/plans/YYYY-MM-DD_HHMMSS-<slug>.md`

이를 활성 작업 디렉토리 / 백엔드 워크스페이스에 대한 상대 경로로 취급하세요. Hermes 파일 도구는 백엔드를 인식하므로, 이 상대 경로를 사용하면 로컬, docker, ssh, modal 및 daytona 백엔드의 워크스페이스 내에 계획을 유지할 수 있습니다.

런타임에서 특정 대상 경로를 제공하는 경우 그 정확한 경로를 사용하세요.
그렇지 않다면, `.hermes/plans/` 아래에 합리적인 타임스탬프 파일명을 직접 생성하세요.

## 상호작용 스타일

- 요청이 충분히 명확하다면 직접 계획을 작성하세요.
- `/plan`에 명시적인 지침이 동반되지 않는다면, 현재 대화 컨텍스트에서 작업을 추론하세요.
- 진정으로 명세가 부족하다면 추측하는 대신 짧은 명확화 질문을 하세요.
- 계획을 저장한 후, 계획한 내용과 저장된 경로를 짧게 답장하세요.

---

# 계획을 잘 작성하는 법

이 스킬의 나머지 부분은 훌륭한 구현 계획(위의 마크다운 파일 안에 들어갈 내용)을 작성하는 기법에 관한 것입니다.

## 개요

구현자가 코드베이스에 대한 컨텍스트가 전혀 없고 안목이 의심스럽다고 가정하고 포괄적인 구현 계획을 작성하세요. 어떤 파일을 수정해야 하는지, 완전한 코드, 테스트 명령, 확인해야 할 문서, 검증 방법 등 그들에게 필요한 모든 것을 문서화하세요. 잘게 쪼개진(Bite-sized) 작업을 제공하세요. DRY, YAGNI, TDD, 빈번한 커밋을 따르세요.

구현자가 숙련된 개발자이기는 하지만 도구 세트나 문제 영역에 대해서는 거의 모른다고 가정하세요. 좋은 테스트 설계에 대해 잘 모른다고 가정하세요.

**핵심 원칙:** 좋은 계획은 구현을 명백하게 만듭니다. 누군가 추측해야 한다면 그 계획은 불완전한 것입니다.

## 전체 구현 계획이 유용할 때

**다음의 경우 항상 사용하세요:**
- 다단계 기능 구현
- 복잡한 요구 사항 분해
- 서브에이전트 주도 개발(subagent-driven-development)을 통한 서브에이전트에게 위임

**다음의 경우에도 건너뛰지 마세요:**
- 기능이 단순해 보일 때 (가정이 버그를 유발합니다)
- 직접 구현할 계획일 때 (미래의 당신에게 지침이 필요합니다)
- 혼자 작업할 때 (문서화가 중요합니다)

## 잘게 쪼개진 작업의 세분화 정도

**각 작업 = 2-5분의 집중적인 일.**

모든 단계는 하나의 행동입니다:
- "실패하는 테스트 작성" — 단계
- "실패하는지 확인하기 위해 실행" — 단계
- "테스트를 통과하기 위한 최소한의 코드 구현" — 단계
- "테스트를 실행하고 통과하는지 확인" — 단계
- "커밋" — 단계

**너무 큰 예:**
```markdown
### 작업 1: 인증 시스템 구축
[5개 파일에 걸친 50줄의 코드]
```

**적절한 크기:**
```markdown
### 작업 1: email 필드를 가진 User 모델 생성
[10줄, 1개 파일]

### 작업 2: User에 password hash 필드 추가
[8줄, 1개 파일]

### 작업 3: 비밀번호 해싱 유틸리티 생성
[15줄, 1개 파일]
```

## 계획 문서 구조

### 헤더 (필수)

모든 계획은 다음으로 시작해야 합니다:

```markdown
# [기능 이름] 구현 계획

> **Hermes의 경우:** 서브에이전트 주도 개발 스킬을 사용하여 이 계획을 작업별로 구현하세요.

**목표:** [이것이 무엇을 구축하는지 설명하는 한 문장]

**아키텍처:** [접근 방식에 대한 2-3문장]

**기술 스택:** [주요 기술/라이브러리]

---
```

### 작업 구조

각 작업은 다음 형식을 따릅니다:

````markdown
### 작업 N: [설명적인 이름]

**목표:** 이 작업이 달성하는 것 (한 문장)

**파일:**
- 생성: `exact/path/to/new_file.py`
- 수정: `exact/path/to/existing.py:45-67` (알려진 경우 줄 번호 포함)
- 테스트: `tests/path/to/test_file.py`

**1단계: 실패하는 테스트 작성**

```python
def test_specific_behavior():
    result = function(input)
    assert result == expected
```

**2단계: 테스트를 실행하여 실패 확인**

실행: `pytest tests/path/test.py::test_specific_behavior -v`
예상: 실패(FAIL) — "function not defined"

**3단계: 최소한의 구현 작성**

```python
def function(input):
    return expected
```

**4단계: 테스트를 실행하여 성공 확인**

실행: `pytest tests/path/test.py::test_specific_behavior -v`
예상: 통과(PASS)

**5단계: 커밋**

```bash
git add tests/path/test.py src/path/file.py
git commit -m "feat: add specific feature"
```
````

## 작성 프로세스

### 1단계: 요구 사항 이해

다음을 읽고 이해하세요:
- 기능 요구 사항
- 설계 문서 또는 사용자 설명
- 수락 조건
- 제약 조건

### 2단계: 코드베이스 탐색

Hermes 도구를 사용하여 프로젝트를 이해하세요:

```python
# 프로젝트 구조 파악
search_files("*.py", target="files", path="src/")

# 유사한 패턴 검색
search_files("similar_pattern", path="src/", file_glob="*.py")

# 기존 테스트 확인
search_files("*.py", target="files", path="tests/")

# 주요 파일 읽기
read_file("src/app.py")
```

### 3단계: 접근 방식 설계

다음을 결정하세요:
- 아키텍처 패턴
- 파일 구성
- 필요한 종속성
- 테스트 전략

### 4단계: 작업 작성

순서대로 작업을 생성하세요:
1. 설정/인프라
2. 핵심 기능 (각각 TDD 적용)
3. 예외 사례(Edge cases)
4. 통합
5. 정리/문서화

### 5단계: 완전한 세부 정보 추가

각 작업에 대해 다음을 포함하세요:
- **정확한 파일 경로** ("config 파일"이 아니라 `src/config/settings.py`)
- **완전한 코드 예시** ("유효성 검사 추가"가 아니라 실제 코드)
- 예상 출력이 포함된 **정확한 명령**
- 작업이 제대로 동작하는지 증명하는 **검증 단계**

### 6단계: 계획 검토

확인하세요:
- [ ] 작업들이 순차적이고 논리적인가
- [ ] 각 작업이 잘게 쪼개져 있는가 (2-5분)
- [ ] 파일 경로가 정확한가
- [ ] 코드 예시가 완전한가 (복사-붙여넣기 가능)
- [ ] 명령이 예상 출력과 함께 정확한가
- [ ] 누락된 컨텍스트는 없는가
- [ ] DRY, YAGNI, TDD 원칙이 적용되었는가

## 원칙

### DRY (Don't Repeat Yourself, 반복하지 마라)

**나쁨:** 3곳에 유효성 검사 코드를 복사-붙여넣기
**좋음:** 유효성 검사 함수를 추출하여 모든 곳에서 사용

### YAGNI (You Aren't Gonna Need It, 넌 그게 필요하지 않을 거야)

**나쁨:** 미래의 요구 사항을 대비해 "유연성" 추가
**좋음:** 현재 필요한 것만 구현

```python
# 나쁨 — YAGNI 위반
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email
        self.preferences = {}  # 아직 필요 없음!
        self.metadata = {}     # 아직 필요 없음!

# 좋음 — YAGNI
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email
```

### TDD (Test-Driven Development, 테스트 주도 개발)

코드를 생산하는 모든 작업은 전체 TDD 주기를 포함해야 합니다:
1. 실패하는 테스트 작성
2. 실행하여 실패 확인
3. 최소한의 코드 작성
4. 실행하여 통과 확인

자세한 내용은 `test-driven-development` 스킬을 참조하세요.

### 빈번한 커밋

모든 작업 후에 커밋하세요:
```bash
git add [files]
git commit -m "type: description"
```

## 흔한 실수들

### 모호한 작업

**나쁨:** "인증 추가"
**좋음:** "email 및 password_hash 필드가 있는 User 모델 생성"

### 불완전한 코드

**나쁨:** "1단계: 유효성 검사 함수 추가"
**좋음:** "1단계: 유효성 검사 함수 추가" 와 함께 전체 함수 코드를 제시

### 누락된 검증

**나쁨:** "3단계: 작동하는지 테스트"
**좋음:** "3단계: `pytest tests/test_auth.py -v` 실행, 예상: 3 passed"

### 파일 경로 누락

**나쁨:** "모델 파일 생성"
**좋음:** "생성: `src/models/user.py`"

## 실행 인계

계획을 저장한 후, 실행 접근 방식을 제안하세요:

**"계획 작성이 완료되어 저장되었습니다. 서브에이전트 주도 개발을 사용하여 실행할 준비가 되었습니다 — 각 작업당 새로운 서브에이전트를 파견하여 2단계 검토(사양 준수 확인 후 코드 품질 검토)를 수행하겠습니다. 계속 진행할까요?"**

실행할 때는 `subagent-driven-development` 스킬을 사용하세요:
- 전체 컨텍스트를 제공하여 작업당 새로운 `delegate_task` 실행
- 각 작업 후 사양 준수 검토
- 사양을 통과한 후 코드 품질 검토
- 두 검토를 모두 통과한 경우에만 진행

## 기억하세요

```
잘게 쪼개진 작업 (각 2-5분)
정확한 파일 경로
완전한 코드 (복사-붙여넣기 가능)
예상 출력이 포함된 정확한 명령
검증 단계
DRY, YAGNI, TDD
빈번한 커밋
```

**좋은 계획은 구현을 명백하게 만듭니다.**
