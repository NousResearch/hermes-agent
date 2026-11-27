---
sidebar_position: 9
title: "개성 및 SOUL.md"
description: "전역 SOUL.md, 기본 제공 개성, 사용자 지정 페르소나 정의로 Hermes Agent의 개성을 맞춤 설정합니다"
---

# 개성 및 SOUL.md

Hermes Agent의 개성은 완전히 맞춤 설정할 수 있습니다. `SOUL.md`는 **주요 정체성**으로, 시스템 프롬프트의 가장 앞에 들어가 에이전트가 누구인지 정의합니다.

- `SOUL.md` — `HERMES_HOME`에 저장되며 에이전트의 정체성 역할을 하는 지속형 페르소나 파일(시스템 프롬프트 슬롯 #1)
- 기본 제공 또는 사용자 지정 `/personality` 프리셋 — 세션 수준의 시스템 프롬프트 오버레이

Hermes가 어떤 존재인지 바꾸거나 완전히 다른 에이전트 페르소나로 교체하려면 `SOUL.md`를 편집하세요.

## 현재 SOUL.md의 작동 방식

이제 Hermes는 다음 위치에 기본 `SOUL.md`를 자동으로 생성합니다.

```text
~/.hermes/SOUL.md
```

더 정확히 말하면 현재 인스턴스의 `HERMES_HOME`을 사용하므로, 사용자 지정 홈 디렉터리로 Hermes를 실행하면 다음 경로를 사용합니다.

```text
$HERMES_HOME/SOUL.md
```

### 중요한 동작

- **SOUL.md는 에이전트의 주요 정체성입니다.** 시스템 프롬프트의 슬롯 #1을 차지하며, 하드코딩된 기본 정체성을 대체합니다.
- Hermes는 아직 존재하지 않는 경우 시작용 `SOUL.md`를 자동으로 생성합니다.
- 기존 사용자의 `SOUL.md` 파일은 절대 덮어쓰지 않습니다.
- Hermes는 `HERMES_HOME`에서만 `SOUL.md`를 불러옵니다.
- Hermes는 현재 작업 디렉터리에서 `SOUL.md`를 찾지 않습니다.
- `SOUL.md`가 존재하지만 비어 있거나 불러올 수 없는 경우, Hermes는 기본 제공 정체성으로 대체합니다.
- `SOUL.md`에 내용이 있으면 보안 검사와 잘라내기 과정을 거친 뒤 그 내용을 그대로 주입합니다.
- SOUL.md는 컨텍스트 파일 섹션에 **중복되지 않습니다** — 정체성으로 한 번만 표시됩니다.

따라서 `SOUL.md`는 단순히 내용을 추가하는 계층이 아니라, 사용자 또는 인스턴스별 정체성이 됩니다.

## 이 설계를 택한 이유

이 방식은 개성을 예측 가능하게 유지합니다.

실행한 디렉터리에 따라 Hermes가 `SOUL.md`를 불러온다면, 프로젝트를 바꿀 때 개성이 예기치 않게 달라질 수 있습니다. `HERMES_HOME`에서만 불러오면 개성이 Hermes 인스턴스 자체에 속하게 됩니다.

또한 사용자에게 다음과 같이 안내하기도 쉬워집니다.

- "`~/.hermes/SOUL.md`를 편집하면 Hermes의 기본 개성을 바꿀 수 있습니다."

## 어디에서 편집하나요?

대부분의 사용자는 다음 파일을 편집하면 됩니다.

```bash
~/.hermes/SOUL.md
```

사용자 지정 홈을 사용하는 경우:

```bash
$HERMES_HOME/SOUL.md
```

## SOUL.md에는 무엇을 작성해야 하나요?

다음과 같은 지속적인 말투 및 개성 지침에 사용하세요.
- 말투
- 커뮤니케이션 스타일
- 직접적으로 표현하는 정도
- 기본 상호작용 방식
- 문체상 피해야 할 것
- Hermes가 불확실성, 의견 충돌 또는 모호성을 다루는 방식

다음과 같은 내용에는 되도록 사용하지 마세요.
- 일회성 프로젝트 지침
- 파일 경로
- 저장소 규칙
- 임시 워크플로 세부 사항

이러한 내용은 `SOUL.md`가 아니라 `AGENTS.md`에 작성해야 합니다.

## 좋은 SOUL.md 콘텐츠

좋은 SOUL 파일은 다음과 같습니다.
- 컨텍스트가 달라져도 안정적입니다.
- 다양한 대화에 적용할 수 있을 만큼 포괄적입니다.
- 말투에 실질적인 영향을 줄 만큼 구체적입니다.
- 작업별 지침이 아니라 커뮤니케이션과 정체성에 초점을 둡니다.

### 예시

```markdown
# Personality

You are a pragmatic senior engineer with strong taste.
You optimize for truth, clarity, and usefulness over politeness theater.

## Style
- Be direct without being cold
- Prefer substance over filler
- Push back when something is a bad idea
- Admit uncertainty plainly
- Keep explanations compact unless depth is useful

## What to avoid
- Sycophancy
- Hype language
- Repeating the user's framing if it's wrong
- Overexplaining obvious things

## Technical posture
- Prefer simple systems over clever systems
- Care about operational reality, not idealized architecture
- Treat edge cases as part of the design, not cleanup
```

## Hermes가 프롬프트에 주입하는 내용

`SOUL.md`의 내용은 시스템 프롬프트의 슬롯 #1, 즉 에이전트 정체성 위치에 바로 들어갑니다. 내용을 감싸는 별도의 래퍼 문구는 추가되지 않습니다.

내용에는 다음 과정이 적용됩니다.
- 프롬프트 인젝션 검사
- 너무 큰 경우 잘라내기

파일이 비어 있거나 공백만 포함하거나 읽을 수 없는 경우 Hermes는 기본 제공 정체성("You are Hermes Agent, an intelligent AI assistant created by Nous Research...")으로 대체합니다. `skip_context_files`가 설정된 경우(예: 서브에이전트/위임 컨텍스트)에도 이 대체 동작이 적용됩니다.

## 보안 검사

다른 컨텍스트 파일과 마찬가지로 `SOUL.md`도 포함되기 전에 프롬프트 인젝션 패턴 검사를 받습니다.

따라서 이상한 메타 지침을 몰래 넣으려 하기보다, 페르소나와 말투에 집중해 작성해야 합니다.

## SOUL.md와 AGENTS.md 비교

이 구분이 가장 중요합니다.

### SOUL.md
다음 용도로 사용합니다.
- 정체성
- 말투
- 스타일
- 기본 커뮤니케이션 방식
- 개성 수준의 동작

### AGENTS.md
다음 용도로 사용합니다.
- 프로젝트 아키텍처
- 코딩 규칙
- 도구 사용 선호 사항
- 저장소별 워크플로
- 명령어, 포트, 경로, 배포 관련 참고 사항

유용한 기준은 다음과 같습니다.
- 어디에서나 나를 따라야 하는 내용이라면 `SOUL.md`에 작성합니다.
- 프로젝트에 속한 내용이라면 `AGENTS.md`에 작성합니다.

## SOUL.md와 `/personality` 비교

`SOUL.md`는 지속적으로 적용되는 기본 개성입니다.

`/personality`는 현재 시스템 프롬프트를 변경하거나 보완하는 세션 수준의 오버레이입니다.

즉:
- `SOUL.md` = 기본 말투
- `/personality` = 임시 모드 전환

예시:
- 실용적인 기본 SOUL을 유지한 다음, 튜터링 대화에서 `/personality teacher`를 사용합니다.
- 간결한 SOUL을 유지한 다음, 브레인스토밍에서 `/personality creative`를 사용합니다.

## 기본 제공 개성

Hermes에는 `/personality`로 전환할 수 있는 기본 제공 개성이 포함되어 있습니다.

| 이름 | 설명 |
|------|-------------|
| **helpful** | 친근한 범용 어시스턴트 |
| **concise** | 간결하고 핵심을 짚는 응답 |
| **technical** | 상세하고 정확한 기술 전문가 |
| **creative** | 혁신적이고 틀을 벗어난 사고 |
| **teacher** | 명확한 예시를 제시하는 인내심 있는 교육자 |
| **kawaii** | 귀여운 표현과 반짝임, 열정 ★ |
| **catgirl** | 고양이 같은 표현을 쓰는 네코짱, 냐~ |
| **pirate** | 기술에 능숙한 해적 선장 Hermes |
| **shakespeare** | 극적인 멋을 살린 셰익스피어풍 산문 |
| **surfer** | 완전히 느긋한 브로 분위기 |
| **noir** | 하드보일드 탐정식 서술 |
| **uwu** | uwu 말투를 사용하는 최대치의 귀여움 |
| **philosopher** | 모든 질문에 대한 깊은 사색 |
| **hype** | 최고조의 에너지와 열정!!! |

## 명령어로 개성 전환하기

### CLI

```text
/personality
/personality concise
/personality technical
```

### 메시징 플랫폼

```text
/personality teacher
```

이 오버레이는 편리하지만, 전역 `SOUL.md`는 오버레이가 의미 있게 변경하지 않는 한 Hermes의 지속적인 기본 개성을 계속 제공합니다.

## 설정에서 사용자 지정 개성 사용하기

기본 제공 개성은 모든 환경(CLI, 메시징 플랫폼, TUI, 데스크톱 앱)에서 항상 사용할 수 있습니다. `~/.hermes/config.yaml`의 `agent.personalities` 아래에 직접 개성을 추가하거나, 이름을 재사용해 기본 제공 개성을 덮어쓸 수 있습니다.

```yaml
agent:
  personalities:
    codereviewer: >
      You are a meticulous code reviewer. Identify bugs, security issues,
      performance concerns, and unclear design choices. Be precise and constructive.
```

그런 다음 다음 명령어로 전환합니다.

```text
/personality codereviewer
```

선택한 개성은 `display.personality`에 이름으로 저장됩니다. 개성은 `agent.system_prompt`를 절대 건드리지 않습니다. 이 필드는 직접 작성한 수동 시스템 프롬프트를 위한 것이며, 개성을 선택하지 않은 경우에만 적용됩니다.

## 기본값으로 재설정하기

활성 개성 오버레이를 취소하고 기본 동작(사용자의 `SOUL.md` 페르소나와 설정한 경우 `agent.system_prompt`)으로 돌아가려면 다음 중 하나를 사용하세요.

```text
/personality none
/personality default
/personality neutral
```

세 명령어 모두 선택 항목(`display.personality`)을 해제하며, 변경 사항은 다음 메시지부터 적용됩니다. 인수 없이 `/personality`를 실행하면 사용 가능한 프리셋과 함께 `none`도 표시되고 현재 활성화된 항목이 표시됩니다.

:::note 업그레이드 시 한 번만 재설정
이전 Hermes 버전은 환경마다 개성 상태를 일관되지 않게 저장했기 때문에, 이전에 꺼둔 개성이 다시 활성화될 수 있었습니다. 업그레이드 후 처음 실행할 때 저장된 개성 선택 항목은 한 번 `none`으로 재설정됩니다(마이그레이션 과정에서 해제된 개성이 출력됩니다). 여전히 사용하려면 `/personality <name>`으로 다시 활성화하세요. 수동으로 설정한 `agent.system_prompt` 텍스트에는 절대 영향을 주지 않습니다.
:::

## 권장 워크플로

권장하는 기본 설정은 다음과 같습니다.

1. `~/.hermes/SOUL.md`에 사려 깊은 전역 `SOUL.md`를 유지합니다.
2. 프로젝트 지침은 `AGENTS.md`에 작성합니다.
3. 일시적으로 모드를 바꾸고 싶을 때만 `/personality`를 사용합니다.

이를 통해 다음을 얻을 수 있습니다.
- 안정적인 말투
- 적절한 위치에 적용된 프로젝트별 동작
- 필요할 때 사용할 수 있는 임시 제어

## 개성이 전체 프롬프트와 상호작용하는 방식

높은 수준에서 프롬프트 스택에는 다음이 포함됩니다.
1. **SOUL.md** (에이전트 정체성 — SOUL.md를 사용할 수 없으면 기본 제공 대체 정체성)
2. 도구 인식형 동작 지침
3. 메모리/사용자 컨텍스트
4. 스킬 지침
5. 컨텍스트 파일(`AGENTS.md`, `.cursorrules`)
6. 타임스탬프
7. 플랫폼별 서식 힌트
8. `/personality`와 같은 선택적 시스템 프롬프트 오버레이

`SOUL.md`가 토대이며, 나머지 모든 요소가 그 위에 쌓입니다.

## 관련 문서

- [컨텍스트 파일](/user-guide/features/context-files)
- [구성](/user-guide/configuration)
- [팁 및 모범 사례](/guides/tips)
- [SOUL.md 가이드](/guides/use-soul-with-hermes)

## CLI 외관과 대화형 개성 비교

대화형 개성과 CLI 외관은 서로 별개입니다.

- `SOUL.md`, `agent.system_prompt`, `/personality`는 Hermes가 말하는 방식에 영향을 줍니다.
- `display.skin`, `/skin`은 Hermes가 터미널에서 보이는 방식에 영향을 줍니다.

터미널 외관은 [스킨 및 테마](./skins.md)를 참고하세요.
