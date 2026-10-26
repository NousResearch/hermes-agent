---
title: "Honcho — Hermes용 Honcho 메모리 구성 및 문제 해결"
sidebar_label: "Honcho"
description: "Hermes용 Honcho 메모리 구성 및 문제 해결"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Honcho

Hermes용 Honcho 메모리를 구성하고 문제를 해결합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/autonomous-ai-agents/honcho`로 설치 |
| 경로 | `optional-skills/autonomous-ai-agents/honcho` |
| 버전 | `2.0.0` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Honcho`, `Memory`, `Profiles`, `Observation`, `Dialectic`, `User-Modeling`, `Session-Summary` |
| 관련 스킬 | [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 지침으로 보는 내용입니다.
:::

# Hermes용 Honcho 메모리

Honcho는 AI에 적합한 세션 간 사용자 모델링을 제공합니다. 대화 전반에서 사용자가 어떤 사람인지 학습하고, 모든 Hermes 프로필에 고유한 피어 정체성을 부여하면서 사용자의 통합된 관점을 공유합니다.

## 사용 시점

- Honcho 설정(클라우드 또는 자체 호스팅)
- 메모리가 작동하지 않거나 피어가 동기화되지 않는 문제 해결
- 각 에이전트가 자체 Honcho 피어를 갖는 다중 프로필 설정 생성
- 관찰, 회상, 변증법적 추론 깊이 또는 쓰기 빈도 설정 조정
- Honcho 도구 5개의 기능과 사용 시점 이해
- 컨텍스트 예산 및 세션 요약 주입 구성

## 설정

### 클라우드(app.honcho.dev)

```bash
hermes memory setup honcho
# select "cloud", paste API key from https://app.honcho.dev
```

### 자체 호스팅

```bash
hermes memory setup honcho
# select "local", enter base URL (e.g. http://localhost:8000)
```

참조: https://docs.honcho.dev/v3/guides/integrations/hermes#running-honcho-locally-with-hermes

### 확인

```bash
hermes honcho status    # shows resolved config, connection test, peer info
```

## 아키텍처

### 기본 컨텍스트 주입

Honcho가 시스템 프롬프트에 컨텍스트를 주입할 때(`hybrid` 또는 `context` 회상 모드) 기본 컨텍스트 블록을 다음 순서로 조합합니다.

1. **세션 요약** -- 지금까지의 현재 세션에 대한 짧은 요약(모델이 대화의 연속성을 즉시 파악할 수 있도록 맨 앞에 배치)
2. **사용자 표현** -- Honcho가 누적한 사용자 모델(선호도, 사실, 패턴)
3. **AI 피어 카드** -- 이 Hermes 프로필의 AI 피어를 위한 정체성 카드

세션 요약은 매 턴 시작 시 Honcho가 자동으로 생성합니다(이전 세션이 있는 경우). 전체 기록을 다시 재생하지 않고도 모델이 자연스럽게 시작할 수 있게 합니다.

### 콜드/웜 프롬프트 선택

Honcho는 두 프롬프트 전략 중에서 자동으로 선택합니다.

| 조건 | 전략 | 동작 |
|-----------|----------|--------------|
| 이전 세션이 없거나 표현이 비어 있음 | **콜드 시작** | 간단한 소개 프롬프트를 사용하고, 요약 주입을 건너뛰며, 모델이 사용자에 대해 학습하도록 유도 |
| 기존 표현 및/또는 세션 기록이 있음 | **웜 시작** | 전체 기본 컨텍스트 주입(요약 → 표현 → 카드)을 사용하고 더 풍부한 시스템 프롬프트를 제공 |

구성할 필요가 없습니다 -- 세션 상태에 따라 자동으로 처리됩니다.

### 피어

Honcho는 대화를 **피어** 간의 상호작용으로 모델링합니다. Hermes는 세션마다 두 개의 피어를 생성합니다.

- **사용자 피어**(`peerName`): 사람을 나타냅니다. Honcho는 관찰된 메시지에서 사용자 표현을 구축합니다.
- **AI 피어**(`aiPeer`): 이 Hermes 인스턴스를 나타냅니다. 각 프로필은 고유한 AI 피어를 가지므로 에이전트가 서로 독립적인 관점을 발전시킵니다.

### 관찰

각 피어에는 Honcho가 무엇을 학습할지 제어하는 두 개의 관찰 토글이 있습니다.

| 토글 | 기능 |
|--------|-------------|
| `observeMe` | 피어 자신의 메시지를 관찰(자기 표현 구축) |
| `observeOthers` | 다른 피어의 메시지를 관찰(피어 간 이해 구축) |

기본값: 네 가지 토글이 모두 **켜짐**(양방향 전체 관찰).

`honcho.json`에서 피어별로 구성합니다.

```json
{
  "observation": {
    "user": { "observeMe": true, "observeOthers": true },
    "ai":   { "observeMe": true, "observeOthers": true }
  }
}
```

또는 다음의 간편 프리셋을 사용합니다.

| 프리셋 | 사용자 | AI | 사용 사례 |
|----------|------|----|----------|
| `"directional"` (기본값) | me:on, others:on | me:on, others:on | 다중 에이전트, 전체 메모리 |
| `"unified"` | me:on, others:off | me:off, others:on | 단일 에이전트, 사용자 전용 모델링 |

[Honcho 대시보드](https://app.honcho.dev)에서 변경한 설정은 세션 초기화 시 다시 동기화됩니다 -- 서버 측 구성이 로컬 기본값보다 우선합니다.

### 세션

Honcho 세션은 메시지와 관찰 결과가 저장될 범위를 지정합니다. 전략 옵션은 다음과 같습니다.

| 전략 | 동작 |
|----------|----------|
| `per-directory` (기본값) | 작업 디렉터리마다 하나의 세션 |
| `per-repo` | Git 저장소 루트마다 하나의 세션 |
| `per-session` | Hermes를 실행할 때마다 새 Honcho 세션 |
| `global` | 모든 디렉터리에서 하나의 세션 |

수동 재정의: `hermes honcho map my-project-name`

### 회상 모드

에이전트가 Honcho 메모리에 액세스하는 방식입니다.

| 모드 | 컨텍스트 자동 주입? | 사용 가능한 도구? | 사용 사례 |
|----------|---------------------|-----------------|----------|
| `hybrid` (기본값) | 예 | 예 | 에이전트가 도구 사용과 자동 컨텍스트 중 선택 |
| `context` | 예 | 아니요(숨김) | 최소 토큰 비용, 도구 호출 없음 |
| `tools` | 아니요 | 예 | 에이전트가 모든 메모리 액세스를 명시적으로 제어 |

## 서로 독립적인 세 가지 조절값

Honcho의 변증법적 동작은 서로 독립적인 세 가지 차원으로 제어됩니다. 각 차원은 다른 차원에 영향을 주지 않고 조정할 수 있습니다.

### 주기(언제)

변증법적 추론 및 컨텍스트 호출이 **얼마나 자주** 발생하는지 제어합니다.

| 키 | 기본값 | 설명 |
|-----|---------|-------------|
| `contextCadence` | `1` | 컨텍스트 API 호출 사이의 최소 턴 수 |
| `dialecticCadence` | `2` | 변증법적 추론 API 호출 사이의 최소 턴 수. 권장 범위 1–5 |
| `injectionFrequency` | `every-turn` | 기본 컨텍스트 주입을 위한 `every-turn` 또는 `first-turn` |

주기 값이 클수록 변증법적 LLM이 덜 자주 실행됩니다. `dialecticCadence: 2`는 엔진이 한 턴 걸러 실행된다는 의미입니다. `1`로 설정하면 매 턴 실행됩니다.

### 깊이(몇 번)

쿼리마다 Honcho가 수행하는 변증법적 추론 **라운드 수**를 제어합니다.

| 키 | 기본값 | 범위 | 설명 |
|-----|---------|-------|-------------|
| `dialecticDepth` | `1` | 1-3 | 쿼리마다 수행하는 변증법적 추론 라운드 수 |
| `dialecticDepthLevels` | -- | 배열 | 깊이별 라운드 수준 재정의(선택 사항, 아래 참조) |

`dialecticDepth: 2`는 Honcho가 변증법적 합성을 두 라운드 실행한다는 의미입니다. 첫 번째 라운드는 초기 답변을 만들고, 두 번째 라운드는 이를 다듬습니다.

`dialecticDepthLevels`를 사용하면 각 라운드의 추론 수준을 독립적으로 설정할 수 있습니다.

```json
{
  "dialecticDepth": 3,
  "dialecticDepthLevels": ["low", "medium", "high"]
}
```

`dialecticDepthLevels`가 생략되면 라운드는 `dialecticReasoningLevel`(기본값)에서 파생된 **비례 수준**을 사용합니다.

| 깊이 | 패스 수준 |
|-------|-------------|
| 1 | [base] |
| 2 | [minimal, base] |
| 3 | [minimal, base, low] |

이렇게 하면 초기 패스는 저렴하게 유지하면서 최종 합성에는 전체 깊이를 사용할 수 있습니다.

**세션 시작 시 깊이.** 세션 시작 프리웜은 첫 번째 턴 전에 백그라운드에서 구성된 `dialecticDepth` 전체를 실행합니다. 콜드 피어에서 단일 패스 프리웜은 출력이 빈약한 경우가 많습니다 -- 다중 패스 깊이는 사용자가 말하기 전에 감사/조정 주기를 실행합니다. 1번 턴은 프리웜 결과를 직접 사용하며, 프리웜 결과가 제때 도착하지 않으면 제한된 타임아웃으로 동기 호출을 수행합니다.

### 수준(얼마나 강하게)

각 변증법적 추론 라운드의 **강도**를 제어합니다.

| 키 | 기본값 | 설명 |
|-----|---------|-------------|
| `dialecticReasoningLevel` | `low` | `minimal`, `low`, `medium`, `high`, `max` |
| `dialecticDynamic` | `true` | `true`이면 모델이 `honcho_reasoning`에 `reasoning_level`을 전달해 호출마다 기본값을 재정의할 수 있음. `false` = 항상 `dialecticReasoningLevel` 사용, 모델 재정의 무시 |

수준이 높을수록 더 풍부한 합성을 제공하지만 Honcho 백엔드에서 더 많은 토큰을 사용합니다.

## 다중 프로필 설정

각 Hermes 프로필은 동일한 작업 공간(사용자 컨텍스트)을 공유하면서 고유한 Honcho AI 피어를 가집니다. 즉:

- 모든 프로필이 동일한 사용자 표현을 봅니다.
- 각 프로필은 고유한 AI 정체성과 관찰 결과를 구축합니다.
- 한 프로필이 기록한 결론은 공유 작업 공간을 통해 다른 프로필에서도 볼 수 있습니다.

### Honcho 피어가 있는 프로필 생성

```bash
hermes profile create coder --clone
# creates host block hermes.coder, AI peer "coder", inherits config from default
```

Honcho에서 `--clone`이 수행하는 작업:
1. `honcho.json`에 `hermes.coder` 호스트 블록 생성
2. `aiPeer: "coder"` 설정(프로필 이름)
3. 기본 프로필에서 `workspace`, `peerName`, `writeFrequency`, `recallMode` 등을 상속
4. 첫 메시지 전에 존재하도록 Honcho에서 피어를 즉시 생성

### 기존 프로필 백필

```bash
hermes honcho sync    # creates host blocks for all profiles that don't have one yet
```

### 프로필별 구성

호스트 블록에서 모든 설정을 재정의할 수 있습니다.

```json
{
  "hosts": {
    "hermes.coder": {
      "aiPeer": "coder",
      "recallMode": "tools",
      "dialecticDepth": 2,
      "observation": {
        "user": { "observeMe": true, "observeOthers": false },
        "ai": { "observeMe": true, "observeOthers": true }
      }
    }
  }
}
```

## 도구

에이전트에는 양방향 Honcho 도구가 5개 있습니다(`context` 회상 모드에서는 숨겨짐).

| 도구 | LLM 호출? | 비용 | 사용 시점 |
|------|-----------|------|----------|
| `honcho_profile` | 아니요 | 최소 | 대화 시작 시 빠른 사실 스냅샷 또는 이름/역할/선호도 빠른 조회 |
| `honcho_search` | 아니요 | 낮음 | 직접 추론할 특정 과거 사실 조회 — 원시 발췌, 합성 없음 |
| `honcho_context` | 아니요 | 낮음 | 전체 세션 컨텍스트 스냅샷: 요약, 표현, 카드, 최근 메시지 |
| `honcho_reasoning` | 예 | 중간–높음 | Honcho의 변증법적 엔진이 합성한 자연어 질문 |
| `honcho_conclude` | 아니요 | 최소 | 영구 사실 작성 또는 삭제; AI 자기 지식에는 `peer: "ai"` 전달 |

### `honcho_profile`

피어 카드(이름, 역할, 선호도, 의사소통 방식 등 엄선된 주요 사실)를 읽거나 업데이트합니다. 업데이트하려면 `card: [...]`를 전달하고, 읽으려면 생략합니다. LLM 호출은 없습니다.

### `honcho_search`

특정 피어에 대해 저장된 컨텍스트를 의미론적으로 검색합니다. 합성 없이 관련성순으로 정렬된 원시 발췌를 반환합니다. 기본값은 800토큰, 최대 2000토큰입니다. 합성된 답변보다 직접 추론할 특정 과거 사실이 필요할 때 유용합니다.

### `honcho_context`

Honcho에서 전체 세션 컨텍스트 스냅샷을 가져옵니다 -- 세션 요약, 피어 표현, 피어 카드, 최근 메시지입니다. LLM 호출은 없습니다. 현재 세션과 피어에 대해 Honcho가 알고 있는 모든 내용을 한 번에 확인하고 싶을 때 사용합니다.

### `honcho_reasoning`

Honcho의 변증법적 추론 엔진이 자연어 질문에 답합니다(Honcho 백엔드에서 LLM 호출). 비용은 더 높지만 품질도 높습니다. `reasoning_level`을 전달해 깊이를 제어합니다: `minimal`(빠름/저렴함) → `low` → `medium` → `high` → `max`(철저함). 생략하면 구성된 기본값(`low`)을 사용합니다. 기본 컨텍스트만으로 충분하지 않은 사용자의 패턴, 목표 또는 현재 상태에 대한 합성된 이해가 실제로 필요할 때 사용합니다.

### `honcho_conclude`

피어에 대한 영구 결론을 작성하거나 삭제합니다. 생성하려면 `conclusion: "..."`을 전달합니다. 결론을 삭제하려면 `delete_id: "..."`를 전달합니다(PII 제거용 -- Honcho는 시간이 지나며 잘못된 결론을 자동으로 수정하므로 PII에만 삭제가 필요합니다). 두 인자 중 정확히 하나를 반드시 전달해야 합니다.

### 양방향 피어 지정

5개 도구 모두 선택적 `peer` 매개변수를 받습니다.
- `peer: "user"` (기본값) — 사용자 피어에서 작동
- `peer: "ai"` — 이 프로필의 AI 피어에서 작동
- `peer: "<explicit-id>"` — 작업 공간의 모든 피어 ID

예시:
```
honcho_profile                        # read user's card
honcho_profile peer="ai"              # read AI peer's card
honcho_reasoning query="What does this user care about most?"
honcho_reasoning query="What are my interaction patterns?" peer="ai" reasoning_level="medium"
honcho_conclude conclusion="Prefers terse answers"
honcho_conclude conclusion="I tend to over-explain code" peer="ai"
honcho_conclude delete_id="abc123"    # PII removal
```

## 에이전트 사용 패턴

Honcho 메모리가 활성화된 Hermes를 위한 지침입니다.

### 대화 시작 시

```
1. honcho_profile                  → fast warmup, no LLM cost
2. If context looks thin → honcho_context  (full snapshot, still no LLM)
3. If deep synthesis needed → honcho_reasoning  (LLM call, use sparingly)
```

매 턴마다 `honcho_reasoning`을 호출하지 마세요. 자동 주입이 진행 중인 컨텍스트 새로 고침을 이미 처리합니다. 기본 컨텍스트가 제공하지 못하는 합성된 통찰이 정말 필요할 때만 추론 도구를 사용하세요.

### 사용자가 기억할 내용을 공유할 때

```
honcho_conclude conclusion="<specific, actionable fact>"
```

좋은 결론: "글로 설명하는 것보다 코드 예시를 선호함", "2026년 4월까지 Rust 비동기 프로젝트를 진행 중"
나쁜 결론: "사용자가 Rust에 대해 무언가를 말함"(너무 모호함), "사용자가 기술적인 것 같음"(이미 표현에 있음)

### 사용자가 과거 컨텍스트를 묻거나 구체적인 내용을 회상해야 할 때

```
honcho_search query="<topic>"       → fast, no LLM, good for specific facts
honcho_context                       → full snapshot with summary + messages
honcho_reasoning query="<question>"  → synthesized answer, use when search isn't enough
```

### `peer: "ai"`를 사용할 때

AI 피어 지정을 사용해 에이전트 자신의 자기 지식을 구축하고 조회합니다.
- `honcho_conclude conclusion="I tend to be verbose when explaining architecture" peer="ai"` — 자기 교정
- `honcho_reasoning query="How do I typically handle ambiguous requests?" peer="ai"` — 자기 감사
- `honcho_profile peer="ai"` — 자신의 정체성 카드 검토

### 도구를 호출하지 않을 때

`hybrid` 및 `context` 모드에서는 기본 컨텍스트(사용자 표현 + 카드 + 세션 요약)가 매 턴 전에 자동 주입됩니다. 이미 주입된 내용을 다시 가져오지 마세요. 다음 경우에만 도구를 호출하세요.
- 주입된 컨텍스트에 없는 내용이 필요할 때
- 사용자가 메모리 회상 또는 확인을 명시적으로 요청할 때
- 새로운 내용에 대한 결론을 작성할 때

### 주기 인식

도구 측의 `honcho_reasoning`은 자동 주입 변증법과 동일한 비용을 공유합니다. 명시적인 도구 호출 후에는 자동 주입 주기가 재설정되어 같은 턴에 이중으로 비용이 청구되지 않습니다.

## 구성 참고

구성 파일: `$HERMES_HOME/honcho.json`(프로필별) 또는 `~/.honcho/config.json`(전역).

### 주요 설정

| 키 | 기본값 | 설명 |
|-----|---------|-------------|
| `apiKey` | -- | API 키([여기서 발급](https://app.honcho.dev)) |
| `baseUrl` | -- | 자체 호스팅 Honcho의 기본 URL |
| `peerName` | -- | 사용자 피어 정체성 |
| `aiPeer` | 호스트 키 | AI 피어 정체성 |
| `workspace` | 호스트 키 | 공유 작업 공간 ID |
| `recallMode` | `hybrid` | `hybrid`, `context` 또는 `tools` |
| `observation` | 모두 켜짐 | 피어별 `observeMe`/`observeOthers` 불리언 |
| `writeFrequency` | `async` | `async`, `turn`, `session` 또는 정수 N |
| `sessionStrategy` | `per-directory` | `per-directory`, `per-repo`, `per-session`, `global` |
| `messageMaxChars` | `25000` | 메시지당 최대 문자 수(초과 시 청크로 분할) |

### 변증법적 설정

| 키 | 기본값 | 설명 |
|-----|---------|-------------|
| `dialecticReasoningLevel` | `low` | `minimal`, `low`, `medium`, `high`, `max` |
| `dialecticDynamic` | `true` | 쿼리 복잡도에 따라 추론을 자동으로 높임. `false` = 고정 수준 |
| `dialecticDepth` | `1` | 쿼리당 변증법적 라운드 수(1-3) |
| `dialecticDepthLevels` | -- | 라운드별 수준의 선택적 배열(예: `["low", "high"]`) |
| `dialecticMaxInputChars` | `10000` | 변증법적 쿼리 입력의 최대 문자 수 |

### 컨텍스트 예산 및 주입

| 키 | 기본값 | 설명 |
|-----|---------|-------------|
| `contextTokens` | 제한 없음 | 결합된 기본 컨텍스트 주입(요약 + 표현 + 카드)의 최대 토큰 수. 선택적 제한 — 생략하면 제한 없음, 정수로 설정하면 주입 크기 제한 |
| `injectionFrequency` | `every-turn` | `every-turn` 또는 `first-turn` |
| `contextCadence` | `1` | 컨텍스트 API 호출 사이의 최소 턴 수 |
| `dialecticCadence` | `2` | 변증법적 LLM 호출 사이의 최소 턴 수(권장 범위 1–5) |

`contextTokens` 예산은 주입 시 적용됩니다. 세션 요약 + 표현 + 카드가 예산을 초과하면 Honcho는 먼저 요약을 줄인 다음 표현을 줄이고 카드는 유지합니다. 이렇게 하면 긴 세션에서 컨텍스트가 폭증하는 것을 방지합니다.

### 메모리 컨텍스트 정리

Honcho는 프롬프트 인젝션과 잘못된 형식의 콘텐츠를 방지하기 위해 주입 전에 `memory-context` 블록을 정리합니다.

- 사용자가 작성한 결론에서 XML/HTML 태그 제거
- 공백 및 제어 문자 정규화
- `messageMaxChars`를 초과하는 개별 결론 잘라내기
- 시스템 프롬프트 구조를 깨뜨릴 수 있는 구분자 시퀀스 이스케이프

이 수정은 마크업이나 특수 문자가 포함된 원시 사용자 결론이 주입된 컨텍스트 블록을 손상시킬 수 있는 예외 상황을 해결합니다.

## 문제 해결

### "Honcho not configured"
`hermes honcho setup`을 실행하세요. `~/.hermes/config.yaml`에 `memory.provider: honcho`가 있는지 확인하세요.

### 세션 간 메모리가 유지되지 않음
`hermes honcho status`를 확인하세요 -- `saveMessages: true`이고 `writeFrequency`가 `session`이 아닌지 확인합니다(`session`은 종료 시에만 기록).

### 프로필에 자체 피어가 할당되지 않음
생성할 때 `--clone`을 사용하세요: `hermes profile create <name> --clone`. 기존 프로필의 경우: `hermes honcho sync`.

### 대시보드의 관찰 변경 사항이 반영되지 않음
관찰 구성은 각 세션 초기화 때 서버에서 동기화됩니다. Honcho UI에서 설정을 변경한 후 새 세션을 시작하세요.

### 메시지가 잘림
`messageMaxChars`(기본값 25k)를 초과하는 메시지는 `[continued]` 표시와 함께 자동으로 청크로 분할됩니다. 이 문제가 자주 발생하면 도구 결과나 스킬 콘텐츠 때문에 메시지 크기가 커지고 있는지 확인하세요.

### 컨텍스트 주입이 너무 큼
컨텍스트 예산 초과 경고가 표시되면 `contextTokens`를 낮추거나 `dialecticDepth`를 줄이세요. 예산이 부족할 때 세션 요약이 먼저 잘립니다.

### 세션 요약이 없음
세션 요약을 사용하려면 현재 Honcho 세션에 이전 턴이 하나 이상 있어야 합니다. 콜드 시작(새 세션, 기록 없음)에서는 요약이 생략되고 Honcho가 콜드 시작 프롬프트 전략을 대신 사용합니다.

## CLI 명령

| 명령 | 설명 |
|---------|-------------|
| `hermes honcho setup` | 대화형 설정 마법사(클라우드/로컬, 정체성, 관찰, 회상, 세션) |
| `hermes honcho status` | 활성 프로필의 확인된 구성, 연결 테스트, 피어 정보 표시 |
| `hermes honcho enable` | 활성 프로필에서 Honcho 활성화(필요하면 호스트 블록 생성) |
| `hermes honcho disable` | 활성 프로필에서 Honcho 비활성화 |
| `hermes honcho peer` | 피어 이름 표시 또는 업데이트(`--user <name>`, `--ai <name>`, `--reasoning <level>`) |
| `hermes honcho peers` | 모든 프로필의 피어 정체성 표시 |
| `hermes honcho mode` | 회상 모드(`hybrid`, `context`, `tools`) 표시 또는 설정 |
| `hermes honcho tokens` | 토큰 예산 표시 또는 설정(`--context <N>`, `--dialectic <N>`) |
| `hermes honcho sessions` | 알려진 디렉터리-세션 이름 매핑 나열 |
| `hermes honcho map <name>` | 현재 작업 디렉터리를 Honcho 세션 이름에 매핑 |
| `hermes honcho identity` | AI 피어 정체성 시드 설정 또는 두 피어 표현 표시 |
| `hermes honcho sync` | 아직 호스트 블록이 없는 모든 Hermes 프로필에 호스트 블록 생성 |
| `hermes honcho migrate` | OpenClaw 기본 메모리에서 Hermes + Honcho로의 단계별 마이그레이션 가이드 |
| `hermes memory setup` | 일반 메모리 제공자 선택기("honcho" 선택 시 동일한 마법사 실행) |
| `hermes memory status` | 활성 메모리 제공자 및 구성 표시 |
| `hermes memory off` | 외부 메모리 제공자 비활성화 |
