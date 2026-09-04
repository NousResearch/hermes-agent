---
title: "테스트 주도 개발 — TDD: RED-GREEN-REFACTOR 강제, 코드보다 테스트 먼저"
sidebar_label: "테스트 주도 개발"
description: "TDD: RED-GREEN-REFACTOR 강제, 코드보다 테스트 먼저"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# 테스트 주도 개발

TDD: RED-GREEN-REFACTOR를 강제하고 코드보다 테스트를 먼저 작성합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 기본 제공(기본 설치됨) |
| 경로 | `skills/software-development/test-driven-development` |
| 버전 | `1.1.0` |
| 작성자 | Hermes Agent (obra/superpowers에서 각색) |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `testing`, `tdd`, `development`, `quality`, `red-green-refactor` |
| 관련 스킬 | [`systematic-debugging`](/docs/user-guide/skills/bundled/software-development/software-development-systematic-debugging), [`plan`](/docs/user-guide/skills/bundled/software-development/software-development-plan), [`subagent-driven-development`](/docs/user-guide/skills/optional/software-development/software-development-subagent-driven-development) |

## 전체 SKILL.md 참고

:::info
다음은 이 스킬이 실행될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성화되면 에이전트가 지침으로 보는 내용입니다.
:::

# 테스트 주도 개발(TDD)

## 개요

먼저 테스트를 작성합니다. 테스트가 실패하는지 확인합니다. 통과하는 데 필요한 최소한의 코드를 작성합니다.

**핵심 원칙:** 테스트가 실패하는 모습을 직접 확인하지 않았다면, 올바른 대상을 테스트하고 있는지 알 수 없습니다.

**규칙의 문구를 어기는 것은 규칙의 취지를 어기는 것입니다.**

## 사용 시점

**항상:**
- 새 기능
- 버그 수정
- 리팩터링
- 동작 변경

**예외(먼저 사용자에게 물어보세요):**
- 일회성 프로토타입
- 생성된 코드
- 구성 파일

“이번 한 번만 TDD를 건너뛸까?”라고 생각했나요? 멈추세요. 그것은 합리화입니다.

## 철칙

```
NO PRODUCTION CODE WITHOUT A FAILING TEST FIRST
```

테스트보다 먼저 코드를 작성했나요? 삭제하세요. 처음부터 다시 시작합니다.

**예외 없음:**
- “참고용”으로 남겨두지 마세요.
- 테스트를 작성하면서 그것을 “적용”하지 마세요.
- 보지 마세요.
- 삭제한다는 것은 삭제하는 것입니다.

테스트에서 새로 구현하세요. 그게 전부입니다.

## Red-Green-Refactor 사이클

### RED — 실패하는 테스트 작성

기대하는 동작을 보여 주는 최소한의 테스트 하나를 작성하세요.

**좋은 테스트:**
```python
def test_retries_failed_operations_3_times():
    attempts = 0
    def operation():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise Exception('fail')
        return 'success'

    result = retry_operation(operation)

    assert result == 'success'
    assert attempts == 3
```
명확한 이름, 실제 동작 테스트, 한 가지 대상.

**나쁜 테스트:**
```python
def test_retry_works():
    mock = MagicMock()
    mock.side_effect = [Exception(), Exception(), 'success']
    result = retry_operation(mock)
    assert result == 'success'  # What about retry count? Timing?
```
모호한 이름, 목을 테스트하며 실제 코드를 테스트하지 않음, 재시도 횟수와 타이밍을 검증하지 않음.

**요구 사항:**
- 테스트 하나당 동작 하나
- 명확하고 설명적인 이름(이름에 “and”가 들어가나요? 나누세요.)
- 목이 아닌 실제 코드(정말 피할 수 없을 때만 목 사용)
- 구현이 아니라 동작을 설명하는 이름

### RED 확인 — 실패하는지 지켜보기

**필수입니다. 절대 건너뛰지 마세요.**

```bash
# Use terminal tool to run the specific test
pytest tests/test_feature.py::test_specific_behavior -v
```

다음을 확인하세요.
- 테스트가 실패함(오타로 인한 오류가 아님)
- 실패 메시지가 예상한 내용임
- 기능이 없기 때문에 실패함

**테스트가 즉시 통과하나요?** 이미 존재하는 동작을 테스트하고 있는 것입니다. 테스트를 수정하세요.

**테스트에서 오류가 발생하나요?** 올바르게 실패할 때까지 오류를 수정하고 다시 실행하세요.

### GREEN — 최소한의 코드

테스트를 통과하는 가장 단순한 코드를 작성하세요. 그 이상은 아무것도 하지 마세요.

**좋은 예:**
```python
def add(a, b):
    return a + b  # Nothing extra
```

**나쁜 예:**
```python
def add(a, b):
    result = a + b
    logging.info(f"Adding {a} + {b} = {result}")  # Extra!
    return result
```

기능을 추가하거나 다른 코드를 리팩터링하거나 테스트가 요구하는 범위를 넘어 “개선”하지 마세요.

**GREEN에서는 속임수가 허용됩니다:**
- 반환값 하드코딩
- 복사-붙여넣기
- 코드 중복
- 엣지 케이스 무시

리팩터링 단계에서 수정합니다.

### GREEN 확인 — 통과하는지 지켜보기

**필수입니다.**

```bash
# Run the specific test
pytest tests/test_feature.py::test_specific_behavior -v

# Then run ALL tests to check for regressions
pytest tests/ -q
```

다음을 확인하세요.
- 테스트가 통과함
- 다른 테스트도 통과함
- 출력이 깨끗함(오류와 경고가 없음)

**테스트가 실패하나요?** 테스트가 아니라 코드를 수정하세요.

**다른 테스트가 실패하나요?** 지금 회귀를 수정하세요.

### REFACTOR — 정리

통과한 후에만 다음을 수행하세요.
- 중복 제거
- 이름 개선
- 헬퍼 추출
- 표현식 단순화

계속 테스트를 통과하는 상태를 유지하세요. 동작을 추가하지 마세요.

**리팩터링 중 테스트가 실패하면:** 즉시 되돌리세요. 더 작은 단계로 진행하세요.

### 반복

다음 동작을 위한 다음 실패 테스트를 작성하세요. 한 번에 한 사이클씩 진행합니다.

## 수평적 슬라이스 피하기

모든 테스트를 먼저 작성한 다음 모든 구현을 작성하지 마세요. 이것은 수평적 슬라이싱입니다. RED가 “상상한 테스트 묶음 작성”이 되고 GREEN이 “그 묶음을 통과시키기”가 됩니다. 구현이 어떤 동작을 실제로 요구하는지 알려 주기 전에 테스트를 설계하므로 취약한 테스트가 만들어집니다.

대신 수직 트레이서 불릿을 사용하세요.

```text
WRONG:
  RED:   test1, test2, test3, test4
  GREEN: impl1, impl2, impl3, impl4

RIGHT:
  RED→GREEN: test1→impl1
  RED→GREEN: test2→impl2
  RED→GREEN: test3→impl3
```

트레이서 불릿은 하나의 종단 간 동작 슬라이스입니다. 경로가 작동한다는 것을 증명하고, 인터페이스에 대해 알려 주며, 방금 배운 내용을 바탕으로 다음 테스트를 작성하게 합니다.

## 순서가 중요한 이유

**“작동하는지 확인하려고 나중에 테스트를 작성하겠습니다.”**

코드 작성 후 만든 테스트는 즉시 통과합니다. 즉시 통과한다는 것은 아무것도 증명하지 않습니다.
- 잘못된 대상을 테스트할 수 있음
- 동작이 아니라 구현을 테스트할 수 있음
- 잊고 있던 엣지 케이스를 놓칠 수 있음
- 실제로 실패를 잡는 모습을 본 적이 없음

테스트 우선 방식은 테스트가 실제로 무언가를 테스트한다는 것을 증명하기 위해 테스트 실패를 직접 확인하게 합니다.

**“모든 엣지 케이스를 직접 테스트했습니다.”**

수동 테스트는 임시방편입니다. 모든 항목을 테스트했다고 생각하지만 다음과 같은 문제가 있습니다.
- 테스트한 내용의 기록이 없음
- 코드가 변경되면 다시 실행할 수 없음
- 압박을 받으면 케이스를 잊기 쉬움
- “시도했을 때 작동했다” ≠ 종합적인 검증

자동화된 테스트는 체계적입니다. 매번 같은 방식으로 실행됩니다.

**“몇 시간 작업한 것을 삭제하는 것은 낭비입니다.”**

매몰 비용 오류입니다. 시간은 이미 사용했습니다. 이제 선택지는 다음과 같습니다.
- 삭제하고 TDD로 다시 작성(높은 신뢰도)
- 그대로 두고 나중에 테스트 추가(낮은 신뢰도, 버그 발생 가능성 높음)

“낭비”는 신뢰할 수 없는 코드를 유지하는 것입니다.

**“TDD는 독단적입니다. 실용적이라면 상황에 맞춰야 합니다.”**

TDD는 실용적입니다.
- 커밋 전에 버그를 찾아냄(나중에 디버깅하는 것보다 빠름)
- 회귀 방지(테스트가 즉시 변경 사항을 감지함)
- 동작 문서화(테스트가 사용 방법을 보여 줌)
- 리팩터링 지원(자유롭게 변경하면 테스트가 문제를 감지함)

“실용적”이라는 지름길은 프로덕션에서 디버깅하는 것과 같아 더 느립니다.

**“나중에 작성한 테스트도 같은 목표를 달성합니다. 형식이 아니라 취지의 문제입니다.”**

아닙니다. 테스트 후 작성은 “이것은 무엇을 하나요?”에 답하고, 테스트 우선은 “이것은 무엇을 해야 하나요?”에 답합니다.

테스트 후 작성은 구현에 편향됩니다. 요구 사항이 아니라 만든 것을 테스트하게 됩니다. 테스트 우선은 구현 전에 엣지 케이스를 발견하게 합니다.

## 흔한 합리화

| 핑계 | 현실 |
|--------|---------|
| “테스트하기에 너무 간단합니다” | 간단한 코드도 깨집니다. 테스트에는 30초가 걸립니다. |
| “나중에 테스트하겠습니다” | 즉시 통과하는 테스트는 아무것도 증명하지 않습니다. |
| “나중에 작성한 테스트도 같은 목표를 달성합니다” | 테스트 후 = “이것은 무엇을 하나?” 테스트 우선 = “이것은 무엇을 해야 하나?” |
| “이미 수동으로 테스트했습니다” | 임시방편 ≠ 체계적 검증. 기록이 없고 다시 실행할 수 없습니다. |
| “몇 시간 작업한 것을 삭제하는 것은 낭비입니다” | 매몰 비용 오류입니다. 검증되지 않은 코드를 유지하는 것은 기술 부채입니다. |
| “참고용으로 남겨두고 테스트를 먼저 작성하겠습니다” | 그것을 적용하게 됩니다. 그것은 테스트를 나중에 작성하는 것입니다. 삭제한다는 것은 삭제하는 것입니다. |
| “먼저 탐색해야 합니다” | 괜찮습니다. 탐색 결과는 버리고 TDD로 시작하세요. |
| “테스트하기 어렵다는 것은 설계가 불명확하다는 뜻입니다” | 테스트의 신호를 따르세요. 테스트하기 어렵다면 사용하기도 어렵습니다. |
| “TDD는 속도를 늦출 것입니다” | TDD는 디버깅보다 빠릅니다. 실용적인 방법은 테스트 우선입니다. |
| “수동 테스트가 더 빠릅니다” | 수동 테스트는 엣지 케이스를 증명하지 못합니다. 변경할 때마다 다시 테스트해야 합니다. |
| “기존 코드에는 테스트가 없습니다” | 개선하고 있는 것입니다. 건드리는 코드에 테스트를 추가하세요. |

## 위험 신호 — 멈추고 처음부터 다시 시작

다음 중 하나라도 하고 있다면 코드를 삭제하고 TDD로 다시 시작하세요.

- 코드보다 먼저 테스트하지 않음
- 구현 후 테스트함
- 첫 실행에서 테스트가 즉시 통과함
- 테스트가 실패한 이유를 설명할 수 없음
- 테스트를 “나중에” 추가함
- “이번 한 번만”이라고 합리화함
- “이미 수동으로 테스트했습니다”라고 말함
- “나중에 작성한 테스트도 같은 목적을 달성합니다”라고 말함
- “참고용으로 남겨두기” 또는 “기존 것을 적용하기”
- “이미 X시간을 썼으니 삭제는 낭비입니다”라고 말함
- “TDD는 독단적이고 나는 실용적입니다”라고 말함
- “이건 다르니까…”라고 말함

**이 모든 것은 코드를 삭제하고 TDD로 다시 시작해야 한다는 뜻입니다.**

## 검증 체크리스트

작업 완료로 표시하기 전에 다음을 확인하세요.

- [ ] 모든 새 함수/메서드에 테스트가 있음
- [ ] 각 테스트가 실패하는 모습을 확인함
- [ ] 각 테스트가 예상한 이유로 실패함(기능이 없어서 실패했으며 오타 때문이 아님)
- [ ] 통과하는 데 필요한 최소한의 코드를 작성함
- [ ] 모든 테스트가 통과함
- [ ] 출력이 깨끗함(오류와 경고가 없음)
- [ ] 테스트가 실제 코드를 사용함(피할 수 없을 때만 목 사용)
- [ ] 엣지 케이스와 오류를 다룸

모든 항목을 확인할 수 없나요? TDD를 건너뛴 것입니다. 처음부터 다시 시작하세요.

## 막혔을 때

| 문제 | 해결책 |
|----------|----------|
| “어떻게 테스트해야 할지 모르겠습니다” | 원하는 API를 작성하세요. 먼저 단언문을 작성하세요. 사용자에게 물어보세요. |
| “테스트가 너무 복잡합니다” | 설계가 너무 복잡한 것입니다. 인터페이스를 단순화하세요. |
| “모든 것을 목 처리해야 합니다” | 코드가 너무 결합되어 있습니다. 의존성 주입을 사용하세요. |
| “테스트 설정이 너무 큽니다” | 헬퍼를 추출하세요. 그래도 복잡한가요? 설계를 단순화하세요. |

## Hermes Agent 통합

### 테스트 실행

각 단계에서 `terminal` 도구를 사용해 테스트를 실행하세요.

```python
# RED — verify failure
terminal("pytest tests/test_feature.py::test_name -v")

# GREEN — verify pass
terminal("pytest tests/test_feature.py::test_name -v")

# Full suite — verify no regressions
terminal("pytest tests/ -q")
```

### delegate_task와 함께 사용

구현을 위해 하위 에이전트를 디스패치할 때 목표에 TDD를 적용하도록 하세요.

```python
delegate_task(
    goal="Implement [feature] using strict TDD",
    context="""
    Follow test-driven-development skill:
    1. Write failing test FIRST
    2. Run test to verify it fails
    3. Write minimal code to pass
    4. Run test to verify it passes
    5. Refactor if needed
    6. Commit

    Project test command: pytest tests/ -q
    Project structure: [describe relevant files]
    """,
    toolsets=['terminal', 'file']
)
```

### systematic-debugging과 함께 사용

버그를 발견했나요? 이를 재현하는 실패 테스트를 작성하세요. TDD 사이클을 따르세요. 테스트가 수정 사항을 입증하고 회귀를 방지합니다.

테스트 없이 버그를 수정하지 마세요.

## 테스트 안티 패턴

- **실제 동작 대신 목의 동작을 테스트** — 목은 상호작용을 검증하는 데 사용해야 하며 테스트 대상 시스템을 대체해서는 안 됩니다.
- **구현 세부 사항 테스트** — 내부 메서드 호출이 아니라 동작/결과를 테스트하세요.
- **성공 경로만 테스트** — 엣지 케이스, 오류, 경계를 항상 테스트하세요.
- **취약한 테스트** — 테스트는 구조가 아니라 동작을 검증해야 하며 리팩터링으로 깨져서는 안 됩니다.

## 최종 규칙

```
Production code → test exists and failed first
Otherwise → not TDD
```

사용자가 명시적으로 허용하지 않는 한 예외는 없습니다.
