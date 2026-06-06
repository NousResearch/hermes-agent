---
title: "Honcho"
sidebar_label: "Honcho"
description: "Hermes에서 Honcho 메모리를 구성하고 사용합니다 -- 세션 간 사용자 모델링, 다중 프로필 피어(peer) 격리, 관찰 구성, 변증법적 추론, 세션 요약 등..."
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py에 의해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Honcho

Hermes에서 Honcho 메모리를 구성하고 사용합니다 -- 세션 간 사용자 모델링, 다중 프로필 피어 격리, 관찰 구성, 변증법적 추론, 세션 요약 및 컨텍스트 예산 적용. Honcho 설정, 메모리 문제 해결, Honcho 피어를 통한 프로필 관리, 또는 관찰, 호출, 변증법적 설정을 조정할 때 사용하세요.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/autonomous-ai-agents/honcho` 명령어로 설치 |
| 경로 | `optional-skills/autonomous-ai-agents/honcho` |
| 버전 | `2.0.0` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Honcho`, `Memory`, `Profiles`, `Observation`, `Dialectic`, `User-Modeling`, `Session-Summary` |
| 관련 스킬 | [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# Hermes를 위한 Honcho 메모리

Honcho는 AI 기반의 세션 간 사용자 모델링을 제공합니다. 여러 대화에 걸쳐 사용자가 누구인지 학습하고 사용자 관점을 통합하면서 각 Hermes 프로필에 고유한 피어(peer) ID를 부여합니다.

## 사용 시기

- Honcho 설정 (클라우드 또는 자체 호스팅)
- 메모리가 작동하지 않거나 피어가 동기화되지 않는 문제 해결
- 각 에이전트가 고유한 Honcho 피어를 갖는 다중 프로필 설정 생성
- 관찰(observation), 리콜(recall), 변증법적 깊이(dialectic depth), 또는 쓰기 빈도(write frequency) 설정 조정
- 5가지 Honcho 도구의 기능과 사용 시기 파악
- 컨텍스트 예산 및 세션 요약 주입 구성

## 설정

### 클라우드 (app.honcho.dev)

```bash
hermes honcho setup
# "cloud" 선택 후, https://app.honcho.dev 에서 API 키를 복사하여 붙여넣기
```

### 자체 호스팅 (Self-hosted)

```bash
hermes honcho setup
# "local" 선택 후, 기본 URL 입력 (예: http://localhost:8000)
```

참조: https://docs.honcho.dev/v3/guides/integrations/hermes#running-honcho-locally-with-hermes

### 확인

```bash
hermes honcho status    # 확인된 구성, 연결 테스트, 피어 정보 표시
```

## 아키텍처

### 기본 컨텍스트 주입

Honcho가 시스템 프롬프트에 컨텍스트를 주입할 때 (`hybrid` 또는 `context` 리콜 모드에서), 다음과 같은 순서로 기본 컨텍스트 블록을 구성합니다:

1. **세션 요약 (Session summary)** -- 현재까지의 세션에 대한 짧은 요약 (모델이 즉각적인 대화 연속성을 갖도록 맨 앞에 배치됨)
2. **사용자 표현 (User representation)** -- Honcho가 축적한 사용자에 대한 모델 (선호도, 사실, 패턴)
3. **AI 피어 카드 (AI peer card)** -- 이 Hermes 프로필의 AI 피어에 대한 신원(Identity) 카드

세션 요약은 각 턴의 시작 시 (이전 세션이 있는 경우) Honcho에 의해 자동으로 생성됩니다. 이는 전체 기록을 재생하지 않고도 모델이 대화 맥락을 바로 파악할 수 있도록 웜 스타트(warm start)를 제공합니다.

### 콜드 / 웜 프롬프트 선택

Honcho는 두 가지 프롬프트 전략 중 하나를 자동으로 선택합니다:

| 조건 | 전략 | 결과 |
|-----------|----------|--------------|
| 이전 세션이 없거나 표현(representation)이 비어 있음 | **콜드 스타트(Cold start)** | 가벼운 소개 프롬프트; 요약 주입을 건너뜀; 모델이 사용자에 대해 배우도록 권장 |
| 기존 표현 및/또는 세션 기록이 있음 | **웜 스타트(Warm start)** | 전체 기본 컨텍스트 주입 (요약 → 표현 → 카드); 더 풍부한 시스템 프롬프트 |

이 설정은 수동으로 할 필요가 없습니다 — 세션 상태에 따라 자동으로 처리됩니다.

### 피어 (Peers)

Honcho는 대화를 **피어(peers)** 간의 상호작용으로 모델링합니다. Hermes는 세션당 두 개의 피어를 생성합니다:

- **사용자 피어** (`peerName`): 인간을 나타냅니다. Honcho는 관찰된 메시지로부터 사용자 표현을 구축합니다.
- **AI 피어** (`aiPeer`): 이 Hermes 인스턴스를 나타냅니다. 각 프로필은 고유한 AI 피어를 가지므로 에이전트는 독립적인 관점을 개발합니다.

### 관찰 (Observation)

각 피어에는 Honcho가 무엇에서 학습할지 제어하는 두 가지 관찰 토글(toggle)이 있습니다:

| 토글 | 기능 |
|--------|-------------|
| `observeMe` | 피어 자신의 메시지가 관찰됨 (자기 표현 구축) |
| `observeOthers` | 다른 피어의 메시지가 관찰됨 (피어 간의 이해 구축) |

기본값: 4개의 토글 모두 **켜짐** (전체 양방향 관찰).

`honcho.json`에서 피어별로 구성합니다:

```json
{
  "observation": {
    "user": { "observeMe": true, "observeOthers": true },
    "ai":   { "observeMe": true, "observeOthers": true }
  }
}
```

또는 짧은 프리셋(preset)을 사용할 수 있습니다:

| 프리셋 | 사용자(User) | AI | 사용 사례 |
|--------|------|----|----------|
| `"directional"` (기본) | me:on, others:on | me:on, others:on | 다중 에이전트, 전체 메모리 |
| `"unified"` | me:on, others:off | me:off, others:on | 단일 에이전트, 사용자 전용 모델링 |

[Honcho 대시보드](https://app.honcho.dev)에서 변경된 설정은 세션 초기화 시 다시 동기화됩니다 — 서버 측 설정이 로컬 기본값보다 우선합니다.

### 세션 (Sessions)

Honcho 세션은 메시지와 관찰이 기록되는 범위를 결정합니다. 전략 옵션:

| 전략 | 동작 |
|----------|----------|
| `per-directory` (기본) | 작업 디렉토리당 하나의 세션 |
| `per-repo` | git 저장소 루트당 하나의 세션 |
| `per-session` | Hermes를 실행할 때마다 새로운 Honcho 세션 |
| `global` | 모든 디렉토리에서 단일 세션 |

수동 재정의: `hermes honcho map my-project-name`

### 리콜 모드 (Recall Modes)

에이전트가 Honcho 메모리에 액세스하는 방법입니다:

| 모드 | 자동 컨텍스트 주입 | 도구 사용 가능 | 사용 사례 |
|------|---------------------|-----------------|----------|
| `hybrid` (기본) | 예 | 예 | 도구를 사용할지 자동 컨텍스트를 사용할지 에이전트가 결정 |
| `context` | 예 | 아니오 (숨김) | 최소 토큰 비용, 도구 호출 없음 |
| `tools` | 아니오 | 예 | 에이전트가 모든 메모리 액세스를 명시적으로 제어 |

## 세 가지 독립적인 조절 요소 (Orthogonal Knobs)

Honcho의 변증법적 동작은 세 가지 독립적인 차원에 의해 제어됩니다. 각각은 다른 항목에 영향을 주지 않고 조정할 수 있습니다:

### 빈도 (Cadence - 언제)

변증법적 호출 및 컨텍스트 호출이 일어나는 **빈도**를 제어합니다.

| 키(Key) | 기본값 | 설명 |
|-----|---------|-------------|
| `contextCadence` | `1` | 컨텍스트 API 호출 사이의 최소 턴 수 |
| `dialecticCadence` | `2` | 변증법적 API 호출 사이의 최소 턴 수. 권장: 1~5 |
| `injectionFrequency` | `every-turn` | 기본 컨텍스트 주입에 대해 `every-turn` 또는 `first-turn` |

빈도(cadence) 값이 높을수록 변증법적 LLM이 덜 자주 실행됩니다. `dialecticCadence: 2`는 엔진이 한 턴씩 걸러서 실행된다는 뜻입니다. 이를 `1`로 설정하면 매 턴마다 실행됩니다.

### 깊이 (Depth - 얼마나 많이)

Honcho가 쿼리당 **몇 라운드의** 변증법적 추론을 수행할지 제어합니다.

| 키(Key) | 기본값 | 범위 | 설명 |
|-----|---------|-------|-------------|
| `dialecticDepth` | `1` | 1-3 | 쿼리당 변증법적 추론 라운드 수 |
| `dialecticDepthLevels` | -- | 배열(array) | 라운드별 깊이 레벨에 대한 선택적 재정의 (아래 참조) |

`dialecticDepth: 2`는 Honcho가 변증법적 통합(synthesis) 라운드를 두 번 실행한다는 것을 의미합니다. 첫 번째 라운드는 초기 답변을 생성하고, 두 번째 라운드는 이를 구체화합니다.

`dialecticDepthLevels`를 통해 각 라운드의 추론 수준을 독립적으로 설정할 수 있습니다:

```json
{
  "dialecticDepth": 3,
  "dialecticDepthLevels": ["low", "medium", "high"]
}
```

`dialecticDepthLevels`를 생략하면, 라운드는 `dialecticReasoningLevel`(기본)에서 파생된 **비례 수준(proportional levels)**을 사용합니다:

| 깊이 | 각 패스(Pass) 수준 |
|-------|-------------|
| 1 | [base] |
| 2 | [minimal, base] |
| 3 | [minimal, base, low] |

이렇게 하면 초기 패스의 비용을 저렴하게 유지하면서 최종 통합에서는 전체 깊이를 사용할 수 있습니다.

**세션 시작 시의 깊이.** 세션 시작 시 예열(prewarm)은 턴 1 이전에 백그라운드에서 설정된 전체 `dialecticDepth`를 실행합니다. 콜드(cold) 피어에서 1회 패스 예열을 실행하면 내용이 부족한 결과가 반환되는 경우가 많습니다 — 멀티 패스 깊이는 사용자가 말하기 전에 감사(audit)/조정(reconcile) 주기를 실행합니다. 턴 1은 예열 결과를 직접 소비합니다. 예열이 제때 완료되지 않으면 턴 1은 제한된 타임아웃이 있는 동기식 호출로 대체됩니다.

### 레벨 (Level - 얼마나 강하게)

각 변증법적 추론 라운드의 **강도(intensity)**를 제어합니다.

| 키(Key) | 기본값 | 설명 |
|-----|---------|-------------|
| `dialecticReasoningLevel` | `low` | `minimal`, `low`, `medium`, `high`, `max` |
| `dialecticDynamic` | `true` | `true`인 경우, 모델은 `honcho_reasoning`에 `reasoning_level`을 전달하여 통화별 기본값을 덮어쓸 수 있습니다. `false` = 항상 `dialecticReasoningLevel`을 사용하며 모델 재정의는 무시됨 |

레벨이 높을수록 풍부한 통합(synthesis)을 생성하지만 Honcho 백엔드에서 더 많은 토큰 비용이 발생합니다.

## 다중 프로필 설정

각 Hermes 프로필은 자체 Honcho AI 피어를 가져오면서 동일한 작업 공간(사용자 컨텍스트)을 공유합니다. 이것은 다음을 의미합니다:

- 모든 프로필이 동일한 사용자 표현을 봅니다.
- 각 프로필은 자신만의 AI 정체성과 관찰 내용을 구축합니다.
- 한 프로필이 작성한 결론(Conclusions)은 공유 작업 공간을 통해 다른 프로필도 볼 수 있습니다.

### Honcho 피어가 있는 프로필 만들기

```bash
hermes profile create coder --clone
# hermes.coder라는 호스트 블록과 "coder"라는 AI 피어를 생성하고, 기본 설정에서 구성을 상속받음
```

`--clone`이 Honcho에 수행하는 작업:
1. `honcho.json`에 `hermes.coder` 호스트 블록을 생성합니다.
2. `aiPeer: "coder"`로 설정합니다 (프로필 이름).
3. 기본값에서 `workspace`, `peerName`, `writeFrequency`, `recallMode` 등을 상속받습니다.
4. 첫 번째 메시지 이전에 피어가 존재할 수 있도록 사전에 피어를 생성합니다.

### 기존 프로필 일괄 적용 (Backfill)

```bash
hermes honcho sync    # 아직 호스트 블록이 없는 모든 프로필에 대해 호스트 블록을 생성합니다.
```

### 프로필별 구성

호스트 블록의 설정을 재정의(override)합니다:

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

## 도구 (Tools)

에이전트는 5개의 양방향 Honcho 도구를 가집니다 (`context` 리콜 모드에서는 숨김 처리됨):

| 도구 | LLM 호출 여부 | 비용 | 사용 시기 |
|------|-----------|------|----------|
| `honcho_profile` | 아니오 | 최소 | 대화 시작 시점의 빠른 사실 스냅샷, 또는 이름/역할/선호도의 빠른 조회를 위해 |
| `honcho_search` | 아니오 | 낮음 | 통합(synthesis) 없이 직접 추론하기 위해 과거의 구체적인 사실을 가져올 때 (원본 발췌) |
| `honcho_context` | 아니오 | 낮음 | 전체 세션 컨텍스트 스냅샷: 요약, 표현, 카드, 최근 메시지 |
| `honcho_reasoning` | 예 | 중간~높음 | Honcho의 변증법적 엔진이 자연어 질문을 종합하여 대답 |
| `honcho_conclude` | 아니오 | 최소 | 영구적인 사실을 기록하거나 삭제함; AI의 자가 인식을 위해 `peer: "ai"` 전달 |

### `honcho_profile`
피어 카드(이름, 역할, 선호도, 커뮤니케이션 스타일 등 엄선된 핵심 사실)를 읽거나 업데이트합니다. 업데이트하려면 `card: [...]`를 전달하고, 읽기만 하려면 생략합니다. LLM 호출이 없습니다.

### `honcho_search`
특정 피어에 대해 저장된 컨텍스트에 대한 의미론적 검색(Semantic search)입니다. 통합(synthesis) 없이 관련성 순으로 정렬된 원본 발췌문(raw excerpts)을 반환합니다. 기본 800 토큰, 최대 2000 토큰입니다. 종합된 답변 대신 구체적인 과거 사실을 통해 직접 추론해야 할 때 유용합니다.

### `honcho_context`
세션 요약, 피어 표현, 피어 카드, 최근 메시지 등 Honcho의 전체 세션 컨텍스트 스냅샷입니다. LLM 호출이 없습니다. 현재 세션과 피어에 대해 Honcho가 알고 있는 모든 내용을 한 번에 보고 싶을 때 사용하세요.

### `honcho_reasoning`
Honcho의 변증법적 추론 엔진이 답하는 자연어 질문(Honcho 백엔드의 LLM 호출)입니다. 비용이 높지만 품질도 높습니다. `reasoning_level`을 전달하여 깊이를 제어합니다: `minimal`(빠름/저렴) → `low` → `medium` → `high` → `max`(철저함). 생략하면 구성된 기본값(`low`)이 사용됩니다. 사용자 패턴, 목표 또는 현재 상태에 대한 종합적인(synthesized) 이해가 필요할 때 사용하세요.

### `honcho_conclude`
피어에 대한 영구적인 결론(conclusion)을 기록하거나 삭제합니다. 생성하려면 `conclusion: "..."`를 전달합니다. 결론을 제거하려면 `delete_id: "..."`를 전달합니다 (PII 제거용 — Honcho는 시간이 지남에 따라 잘못된 결론을 스스로 수정하므로 PII 문제 시에만 삭제가 필요합니다). 반드시 둘 중 하나만 전달해야 합니다.

### 양방향 피어 타겟팅

5개의 모든 도구는 선택적으로 `peer` 매개변수를 받습니다:
- `peer: "user"` (기본값) — 사용자 피어를 대상으로 작동합니다.
- `peer: "ai"` — 이 프로필의 AI 피어를 대상으로 작동합니다.
- `peer: "<explicit-id>"` — 작업 공간 내의 임의의 피어 ID.

예시:
```
honcho_profile                        # 사용자의 카드 읽기
honcho_profile peer="ai"              # AI 피어의 카드 읽기
honcho_reasoning query="What does this user care about most?"
honcho_reasoning query="What are my interaction patterns?" peer="ai" reasoning_level="medium"
honcho_conclude conclusion="Prefers terse answers"
honcho_conclude conclusion="I tend to over-explain code" peer="ai"
honcho_conclude delete_id="abc123"    # PII 제거
```

## 에이전트 사용 패턴

Honcho 메모리가 활성화된 경우 Hermes를 위한 가이드라인.

### 대화 시작 시

```
1. honcho_profile                  → 빠른 웜업, LLM 비용 없음
2. 컨텍스트가 부족해 보이면 → honcho_context  (전체 스냅샷, 여전히 LLM 비용 없음)
3. 심층적인 종합(synthesis)이 필요하면 → honcho_reasoning  (LLM 호출 발생, 아껴서 사용)
```

매 턴마다 `honcho_reasoning`을 호출하지 마십시오. 자동 주입은 이미 진행 중인 컨텍스트 새로 고침을 처리합니다. 기본 컨텍스트가 제공하지 않는 종합적인(synthesized) 인사이트가 진정으로 필요할 때만 추론 도구를 사용하세요.

### 사용자가 기억할 만한 것을 공유할 때

```
honcho_conclude conclusion="<구체적이고 실행 가능한 사실>"
```

좋은 결론의 예: "글로 된 설명보다 코드 예시를 선호함", "2026년 4월까지 Rust 비동기 프로젝트를 진행 중임"
나쁜 결론의 예: "사용자가 Rust에 대해 언급함" (너무 모호함), "사용자가 기술 전문가인 것 같음" (이미 표현에 존재함)

### 사용자가 과거 상황을 묻거나 구체적인 내용을 회상해야 할 때

```
honcho_search query="<주제>"       → 빠름, LLM 없음, 특정 사실을 찾을 때 유용함
honcho_context                       → 요약 및 메시지가 포함된 전체 스냅샷
honcho_reasoning query="<질문>"  → 종합된 답변, 검색으로 충분하지 않을 때 사용
```

### `peer: "ai"`를 사용하는 경우

AI 피어 타겟팅을 사용하여 에이전트 자체의 자가 인식을 구축하고 쿼리합니다:
- `honcho_conclude conclusion="I tend to be verbose when explaining architecture" peer="ai"` — 자체 교정
- `honcho_reasoning query="How do I typically handle ambiguous requests?" peer="ai"` — 자가 점검
- `honcho_profile peer="ai"` — 자신의 신원 카드 검토

### 도구를 호출하면 안 되는 경우

`hybrid` 및 `context` 모드에서는 매 턴 전에 기본 컨텍스트(사용자 표현 + 카드 + 세션 요약)가 자동 주입됩니다. 이미 주입된 것을 다시 가져오지 마십시오. 다음과 같은 경우에만 도구를 호출하세요:
- 주입된 컨텍스트에 없는 내용이 필요할 때
- 사용자가 명시적으로 메모리를 회상하거나 확인해달라고 요청할 때
- 새로운 내용에 대한 결론을 작성할 때

### 빈도(Cadence) 인지

도구 측면에서의 `honcho_reasoning`은 자동 주입 변증법(dialectic)과 동일한 비용을 공유합니다. 명시적인 도구 호출 이후에는 자동 주입 빈도가 재설정되어 — 같은 턴에서 비용이 이중으로 청구되는 것을 방지합니다.

## 구성(Config) 참조

구성 파일: `$HERMES_HOME/honcho.json` (프로필 로컬) 또는 `~/.honcho/config.json` (전역).

### 주요 설정

| 키(Key) | 기본값 | 설명 |
|-----|---------|-------------|
| `apiKey` | -- | API 키 ([키 발급 받기](https://app.honcho.dev)) |
| `baseUrl` | -- | 자체 호스팅 Honcho를 위한 기본 URL |
| `peerName` | -- | 사용자 피어 식별자 |
| `aiPeer` | host key | AI 피어 식별자 |
| `workspace` | host key | 공유 작업 공간 ID |
| `recallMode` | `hybrid` | `hybrid`, `context`, 또는 `tools` |
| `observation` | 모두 켜짐(all on) | 피어별 `observeMe`/`observeOthers` 불리언(booleans) |
| `writeFrequency` | `async` | `async`, `turn`, `session`, 또는 정수 N |
| `sessionStrategy` | `per-directory` | `per-directory`, `per-repo`, `per-session`, `global` |
| `messageMaxChars` | `25000` | 메시지당 최대 문자 수 (초과 시 청크 단위로 나뉨) |

### 변증법적(Dialectic) 설정

| 키(Key) | 기본값 | 설명 |
|-----|---------|-------------|
| `dialecticReasoningLevel` | `low` | `minimal`, `low`, `medium`, `high`, `max` |
| `dialecticDynamic` | `true` | 쿼리 복잡성에 따라 추론을 자동으로 올림. `false` = 고정 레벨 |
| `dialecticDepth` | `1` | 쿼리당 변증법적 라운드 수 (1-3) |
| `dialecticDepthLevels` | -- | 선택적 라운드별 레벨 배열, 예: `["low", "high"]` |
| `dialecticMaxInputChars` | `10000` | 변증법적 쿼리 입력의 최대 문자 수 |

### 컨텍스트 예산 및 주입

| 키(Key) | 기본값 | 설명 |
|-----|---------|-------------|
| `contextTokens` | 무제한(uncapped) | 결합된 기본 컨텍스트 주입(요약 + 표현 + 카드)의 최대 토큰 수. 선택 사항(Opt-in) — 제한을 해제하려면 생략하고, 주입 크기를 제한하려면 정수로 설정하세요. |
| `injectionFrequency` | `every-turn` | `every-turn` 또는 `first-turn` |
| `contextCadence` | `1` | 컨텍스트 API 호출 사이의 최소 턴 수 |
| `dialecticCadence` | `2` | 변증법적 LLM 호출 사이의 최소 턴 수 (권장 1~5) |

`contextTokens` 예산은 주입 시점에 적용됩니다. 세션 요약 + 표현 + 카드가 예산을 초과하면 Honcho는 먼저 요약을 자르고, 그다음 카드를 보존하면서 표현을 자릅니다. 이렇게 하여 긴 세션에서 컨텍스트 폭발을 방지합니다.

### 메모리 컨텍스트 소독 (Sanitization)

Honcho는 프롬프트 인젝션(prompt injection)과 잘못된 형식의 콘텐츠를 방지하기 위해 주입 전에 `memory-context` 블록을 소독(sanitize)합니다:

- 사용자가 작성한 결론에서 XML/HTML 태그 제거
- 공백 및 제어 문자 정규화
- `messageMaxChars`를 초과하는 개별 결론 자르기
- 시스템 프롬프트 구조를 깰 수 있는 구분자(delimiter) 시퀀스 이스케이프 처리

이 수정 사항은 마크업이나 특수 문자가 포함된 원본 사용자의 결론이 주입된 컨텍스트 블록을 손상시킬 수 있는 엣지 케이스를 해결합니다.

## 문제 해결 (Troubleshooting)

### "Honcho not configured" (Honcho가 구성되지 않음)
`hermes honcho setup`을 실행하세요. `~/.hermes/config.yaml`에 `memory.provider: honcho`가 있는지 확인하세요.

### 여러 세션에 걸쳐 메모리가 유지되지 않음
`hermes honcho status`를 확인하세요 -- `saveMessages: true`를 검증하고 `writeFrequency`가 `session`(종료 시에만 쓰는 옵션)이 아닌지 확인하세요.

### 프로필이 자체 피어를 가져오지 않음
생성 시 `--clone`을 사용하세요: `hermes profile create <name> --clone`. 기존 프로필의 경우: `hermes honcho sync`.

### 대시보드에서의 관찰 변경 사항이 반영되지 않음
관찰 구성은 각 세션 초기화 시 서버에서 동기화됩니다. Honcho UI에서 설정을 변경한 후 새 세션을 시작하세요.

### 메시지가 잘림
`messageMaxChars`(기본 25k)를 초과하는 메시지는 자동으로 `[continued]` 마커와 함께 분할(chunking)됩니다. 이런 현상이 자주 발생한다면 도구 결과나 스킬 콘텐츠가 메시지 크기를 부풀리고 있는지 확인하세요.

### 주입된 컨텍스트가 너무 큼
컨텍스트 예산 초과 경고가 표시되면 `contextTokens`를 줄이거나 `dialecticDepth`를 줄이세요. 예산이 부족할 때 세션 요약이 가장 먼저 잘립니다.

### 세션 요약이 없음
세션 요약에는 현재 Honcho 세션에 최소 하나의 이전 턴이 있어야 합니다. 콜드 스타트(새로운 세션, 기록 없음) 시 요약이 생략되고 Honcho는 대신 콜드 스타트 프롬프트 전략을 사용합니다.

## CLI 명령어

| 명령어 | 설명 |
|---------|-------------|
| `hermes honcho setup` | 대화형 설정 마법사 (클라우드/로컬, 식별자, 관찰, 호출, 세션) |
| `hermes honcho status` | 활성 프로필에 대해 해결된 구성, 연결 테스트, 피어 정보 표시 |
| `hermes honcho enable` | 활성 프로필에 대해 Honcho 활성화 (필요 시 호스트 블록 생성) |
| `hermes honcho disable` | 활성 프로필에 대해 Honcho 비활성화 |
| `hermes honcho peer` | 피어 이름 표시 또는 업데이트 (`--user <name>`, `--ai <name>`, `--reasoning <level>`) |
| `hermes honcho peers` | 모든 프로필에 걸친 피어 식별자 표시 |
| `hermes honcho mode` | 호출(recall) 모드 표시 또는 설정 (`hybrid`, `context`, `tools`) |
| `hermes honcho tokens` | 토큰 예산 표시 또는 설정 (`--context <N>`, `--dialectic <N>`) |
| `hermes honcho sessions` | 알려진 디렉토리-세션 이름 매핑 목록 표시 |
| `hermes honcho map <name>` | 현재 작업 디렉토리를 Honcho 세션 이름에 매핑 |
| `hermes honcho identity` | AI 피어 정체성 기반 제공(Seed) 또는 양쪽 피어 표현 표시 |
| `hermes honcho sync` | 아직 호스트 블록이 없는 모든 Hermes 프로필에 대해 호스트 블록 생성 |
| `hermes honcho migrate` | OpenClaw 네이티브 메모리에서 Hermes + Honcho로 마이그레이션하기 위한 단계별 가이드 |
| `hermes memory setup` | 일반 메모리 제공자 선택기 ("honcho"를 선택하면 동일한 마법사 실행) |
| `hermes memory status` | 활성 메모리 제공자 및 구성 표시 |
| `hermes memory off` | 외부 메모리 제공자 비활성화 |
