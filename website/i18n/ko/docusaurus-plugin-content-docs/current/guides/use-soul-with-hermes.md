---
sidebar_position: 7
title: "Hermes에서 SOUL.md 사용하기"
description: "Hermes Agent의 기본 말투를 정하고, 그 안에 무엇을 담아야 하는지, AGENTS.md 및 /personality와 어떻게 다른지 알아봅니다"
---

# Hermes에서 SOUL.md 사용하기

`SOUL.md`는 **Hermes 인스턴스의 기본 정체성**입니다. 시스템 프롬프트에서 가장 먼저 들어가는 항목으로, 에이전트가 누구인지, 어떻게 말하는지, 무엇을 피하는지 정의합니다.

대화할 때마다 Hermes가 같은 어시스턴트처럼 느껴지게 하고 싶거나 — Hermes 페르소나를 자신의 것으로 완전히 바꾸고 싶다면 — 이 파일을 사용하세요.

## SOUL.md의 용도

다음과 같은 내용에는 `SOUL.md`를 사용하세요.
- 말투
- 성격
- 소통 방식
- Hermes가 얼마나 직설적이거나 따뜻해야 하는지
- Hermes가 말투 측면에서 피해야 할 것
- 불확실성, 의견 충돌, 모호성을 Hermes가 어떻게 다뤄야 하는지

요약하면:
- `SOUL.md`는 Hermes가 누구이며 어떻게 말하는지에 관한 파일입니다.

## SOUL.md의 용도가 아닌 것

다음과 같은 용도로는 사용하지 마세요.
- 저장소별 코딩 규칙
- 파일 경로
- 명령어
- 서비스 포트
- 아키텍처 메모
- 프로젝트 작업 흐름 지침

이런 내용은 `AGENTS.md`에 둡니다.

간단한 기준은 다음과 같습니다.
- 어디에나 적용되어야 한다면 `SOUL.md`에 작성하세요.
- 한 프로젝트에만 해당한다면 `AGENTS.md`에 작성하세요.

## 저장 위치

Hermes는 이제 현재 인스턴스에 전역 SOUL 파일만 사용합니다.

```text
~/.hermes/SOUL.md
```

사용자 지정 홈 디렉터리로 Hermes를 실행하면 경로는 다음과 같아집니다.

```text
$HERMES_HOME/SOUL.md
```

## 최초 실행 시 동작

기존 `SOUL.md`가 없으면 Hermes가 시작용 `SOUL.md`를 자동으로 생성합니다.

따라서 이제 대부분의 사용자는 바로 읽고 수정할 수 있는 실제 파일에서 시작합니다.

중요:
- 이미 `SOUL.md`가 있으면 Hermes는 덮어쓰지 않습니다.
- 파일이 존재하지만 비어 있으면 Hermes는 그 파일에서 프롬프트에 추가할 내용을 넣지 않습니다.

## Hermes가 파일을 사용하는 방식

Hermes가 세션을 시작하면 `HERMES_HOME`에서 `SOUL.md`를 읽고, 프롬프트 인젝션 패턴을 검사하고, 필요하면 길이를 줄인 뒤, **에이전트 정체성**으로 사용합니다 — 시스템 프롬프트의 1번 슬롯입니다. 즉, `SOUL.md`가 내장된 기본 정체성 텍스트를 완전히 대체합니다.

`SOUL.md`가 없거나, 비어 있거나, 불러올 수 없으면 Hermes는 내장된 기본 정체성으로 대체합니다.

파일 주위에 래퍼 문구를 추가하지 않습니다. 중요한 것은 파일 내용 자체이므로, 에이전트가 생각하고 말하기를 원하는 방식으로 작성하세요.

## 처음 해 볼 수정

다른 작업은 하지 않더라도 파일을 열어 몇 줄만 바꿔 자신의 스타일로 만들어 보세요.

예를 들면 다음과 같습니다.

```markdown
You are direct, calm, and technically precise.
Prefer substance over politeness theater.
Push back clearly when an idea is weak.
Keep answers compact unless deeper detail is useful.
```

이것만으로도 Hermes가 느껴지는 방식이 눈에 띄게 달라질 수 있습니다.

## 예시 스타일

### 1. 실용적인 엔지니어

```markdown
You are a pragmatic senior engineer.
You care more about correctness and operational reality than sounding impressive.

## Style
- Be direct
- Be concise unless complexity requires depth
- Say when something is a bad idea
- Prefer practical tradeoffs over idealized abstractions

## Avoid
- Sycophancy
- Hype language
- Overexplaining obvious things
```

### 2. 연구 파트너

```markdown
You are a thoughtful research collaborator.
You are curious, honest about uncertainty, and excited by unusual ideas.

## Style
- Explore possibilities without pretending certainty
- Distinguish speculation from evidence
- Ask clarifying questions when the idea space is underspecified
- Prefer conceptual depth over shallow completeness
```

### 3. 교사 / 설명자

```markdown
You are a patient technical teacher.
You care about understanding, not performance.

## Style
- Explain clearly
- Use examples when they help
- Do not assume prior knowledge unless the user signals it
- Build from intuition to details
```

### 4. 엄격한 리뷰어

```markdown
You are a rigorous reviewer.
You are fair, but you do not soften important criticism.

## Style
- Point out weak assumptions directly
- Prioritize correctness over harmony
- Be explicit about risks and tradeoffs
- Prefer blunt clarity to vague diplomacy
```

## 좋은 SOUL.md란?

좋은 `SOUL.md`는 다음과 같습니다.
- 안정적이어야 합니다.
- 폭넓게 적용할 수 있어야 합니다.
- 목소리가 구체적이어야 합니다.
- 임시 지침으로 가득 차 있지 않아야 합니다.

좋지 않은 `SOUL.md`는 다음과 같습니다.
- 프로젝트 세부 정보로 가득합니다.
- 서로 모순됩니다.
- 모든 응답 형식을 세세하게 관리하려 합니다.
- “도움이 되게”와 “명확하게”처럼 대부분이 뻔한 기본값을 되풀이하는 일반적인 문구로 채워져 있습니다.

Hermes는 이미 도움을 주고 명확하게 답하려고 합니다. `SOUL.md`에는 뻔한 기본값을 다시 적는 대신 실제 성격과 스타일을 추가해야 합니다.

## 권장 구조

제목은 꼭 필요하지 않지만, 있으면 도움이 됩니다.

다음과 같은 간단한 구조가 잘 작동합니다.

```markdown
# Identity
Who Hermes is.

# Style
How Hermes should sound.

# Avoid
What Hermes should not do.

# Defaults
How Hermes should behave when ambiguity appears.
```

## SOUL.md와 /personality 비교

두 기능은 서로 보완적입니다.

지속적으로 유지할 기본 설정에는 `SOUL.md`를 사용하세요.
일시적인 모드 전환에는 `/personality`를 사용하세요.

예를 들면 다음과 같습니다.
- 기본 SOUL은 실용적이고 직설적입니다.
- 그런 다음 한 세션에서 `/personality teacher`를 사용합니다.
- 나중에 기본 목소리 파일을 바꾸지 않고 원래 설정으로 돌아갑니다.

## SOUL.md와 AGENTS.md 비교

이것이 가장 흔한 실수입니다.

### 다음 내용은 SOUL.md에 작성하세요
- “직접적으로 말하세요.”
- “과장된 표현을 피하세요.”
- “깊이가 도움이 되지 않는 한 짧게 답하세요.”
- “사용자가 틀렸을 때는 분명하게 반박하세요.”

### 다음 내용은 AGENTS.md에 작성하세요
- “unittest가 아니라 pytest를 사용하세요.”
- “프런트엔드는 `frontend/`에 있습니다.”
- “마이그레이션을 직접 수정하지 마세요.”
- “API는 포트 8000에서 실행됩니다.”

## 편집 방법

```bash
nano ~/.hermes/SOUL.md
```

또는

```bash
vim ~/.hermes/SOUL.md
```

그런 다음 Hermes를 재시작하거나 새 세션을 시작하세요.

## 실용적인 작업 흐름

1. 기본으로 생성된 파일부터 시작합니다.
2. 원하는 목소리와 어울리지 않는 내용은 모두 덜어냅니다.
3. 말투와 기본 동작을 분명하게 정의하는 4–8줄을 추가합니다.
4. 한동안 Hermes와 대화합니다.
5. 여전히 어색한 부분을 기준으로 조정합니다.

한 번에 완벽한 성격을 설계하려 하기보다 이렇게 반복해서 조정하는 방식이 더 효과적입니다.

## 문제 해결

### SOUL.md를 수정했지만 Hermes의 말투가 그대로입니다

다음을 확인하세요.
- `~/.hermes/SOUL.md` 또는 `$HERMES_HOME/SOUL.md`를 수정했는지
- 저장소 내 다른 `SOUL.md`를 수정한 것은 아닌지
- 파일이 비어 있지 않은지
- 수정 후 세션을 재시작했는지
- `/personality` 오버레이가 결과에 지나치게 영향을 주고 있지 않은지

### Hermes가 SOUL.md의 일부를 무시합니다

가능한 원인은 다음과 같습니다.
- 더 높은 우선순위의 지침이 해당 내용을 덮어쓰고 있습니다.
- 파일에 서로 충돌하는 지침이 포함되어 있습니다.
- 파일이 너무 길어 잘렸습니다.
- 일부 문구가 프롬프트 인젝션 내용과 유사하여 스캐너가 차단하거나 변경했을 수 있습니다.

### SOUL.md가 프로젝트에 지나치게 특화되었습니다

프로젝트 지침을 `AGENTS.md`로 옮기고 `SOUL.md`는 정체성과 스타일에 집중하도록 유지하세요.

## 관련 문서

- [성격 및 SOUL.md](/user-guide/features/personality)
- [컨텍스트 파일](/user-guide/features/context-files)
- [구성](/user-guide/configuration)
- [팁 및 모범 사례](/guides/tips)
