---
sidebar_position: 99
title: "Honcho 메모리"
description: "Honcho를 통한 AI 네이티브 영구 메모리 — 변증법적 추론, 다중 에이전트 사용자 모델링, 심층 개인화"
---

# Honcho 메모리

[Honcho](https://github.com/plastic-labs/honcho)는 Hermes에 내장된 메모리 시스템 위에 변증법적 추론과 심층 사용자 모델링을 추가하는 AI 네이티브 메모리 백엔드입니다. 단순한 키-값 저장소 대신 Honcho는 대화가 끝난 후 대화를 추론하여 사용자의 선호도, 소통 방식, 목표, 패턴 등 사용자가 어떤 사람인지에 대한 지속적인 모델을 유지합니다.

:::info Honcho는 메모리 제공자 플러그인입니다
Honcho는 [메모리 제공자](./memory-providers.md) 시스템에 통합되어 있습니다. 아래의 모든 기능은 통합 메모리 제공자 인터페이스를 통해 사용할 수 있습니다.
:::

## Honcho가 추가하는 기능

| 기능 | 내장 메모리 | Honcho |
|-----------|----------------|--------|
| 세션 간 지속성 | ✔ 파일 기반 MEMORY.md/USER.md | ✔ API를 사용하는 서버 측 저장 |
| 사용자 프로필 | ✔ 에이전트의 수동 관리 | ✔ 자동 변증법적 추론 |
| 세션 요약 | — | ✔ 세션 범위 컨텍스트 주입 |
| 다중 에이전트 격리 | — | ✔ 피어별 프로필 분리 |
| 관찰 모드 | — | ✔ 통합 또는 방향성 관찰 |
| 결론 (도출된 인사이트) | — | ✔ 패턴에 대한 서버 측 추론 |
| 기록 전체 검색 | ✔ FTS5 세션 검색 | ✔ 결론에 대한 시맨틱 검색 |

**변증법적 추론**: 각 대화 턴 후(`dialecticCadence`로 제어됨) Honcho는 대화를 분석하고 사용자의 선호도, 습관, 목표에 대한 인사이트를 도출합니다. 이러한 인사이트는 시간이 지나며 축적되어 사용자가 명시적으로 말한 내용을 넘어서는 깊은 이해를 에이전트에 제공합니다. 변증법적 추론은 다중 패스 깊이(1–3 패스)를 지원하며 콜드/웜 프롬프트를 자동으로 선택합니다. 콜드 시작 쿼리는 일반적인 사용자 사실에 집중하고 웜 쿼리는 세션 범위 컨텍스트를 우선합니다.

**세션 범위 컨텍스트**: 이제 기본 컨텍스트에는 사용자 표현 및 피어 카드와 함께 세션 요약도 포함됩니다. 이를 통해 에이전트는 현재 세션에서 이미 논의된 내용을 파악하여 반복을 줄이고 대화의 연속성을 유지할 수 있습니다.

**다중 에이전트 프로필**: 여러 Hermes 인스턴스가 같은 사용자와 대화할 때(예: 코딩 도우미와 개인 비서) Honcho는 별도의 "피어" 프로필을 유지합니다. 각 피어는 자신의 관찰 및 결론만 확인하므로 컨텍스트가 서로 오염되는 것을 방지합니다.

## 설정

```bash
hermes memory setup    # select "honcho" from the provider list
```

또는 수동으로 설정합니다.

```yaml
# ~/.hermes/config.yaml
memory:
  provider: honcho
```

```bash
echo 'HONCHO_API_KEY=***' >> ~/.hermes/.env
```

[honcho.dev](https://honcho.dev)에서 API 키를 발급받으세요.

## 아키텍처

### 2계층 컨텍스트 주입

매 턴(`hybrid` 또는 `context` 모드에서) Honcho는 시스템 프롬프트에 주입되는 두 계층의 컨텍스트를 구성합니다.

1. **기본 컨텍스트** — 세션 요약, 사용자 표현, 사용자 피어 카드, AI 자기 표현, AI 정체성 카드입니다. `contextCadence`에 따라 새로 고쳐집니다. 이는 "이 사용자는 누구인가" 계층입니다.
2. **변증법적 보충 정보** — 사용자의 현재 상태와 필요에 대한 LLM 합성 추론입니다. `dialecticCadence`에 따라 새로 고쳐집니다. 이는 "지금 중요한 것은 무엇인가" 계층입니다.

두 계층은 모두 연결된 후 `contextTokens` 예산(설정된 경우)에 맞게 잘립니다.

### 콜드/웜 프롬프트 선택

변증법적 추론은 다음 두 프롬프트 전략 중 하나를 자동으로 선택합니다.

- **콜드 시작** (아직 기본 컨텍스트 없음): 일반 쿼리 — "이 사람은 누구인가? 선호도, 목표, 업무 방식은 무엇인가?"
- **웜 세션** (기본 컨텍스트가 존재함): 세션 범위 쿼리 — "이 세션에서 지금까지 논의된 내용을 고려할 때, 이 사용자에 대해 어떤 컨텍스트가 가장 관련 있는가?"

이는 기본 컨텍스트가 채워졌는지에 따라 자동으로 결정됩니다.

### 서로 독립적인 3가지 설정 노브

비용과 깊이는 서로 독립적인 세 가지 노브로 제어됩니다.

| 노브 | 제어 대상 | 기본값 |
|------|----------|---------|
| `contextCadence` | `context()` API 호출 사이의 턴 수 (기본 계층 새로 고침) | `1` |
| `dialecticCadence` | `peer.chat()` LLM 호출 사이의 턴 수 (변증법적 계층 새로 고침) | `2` (권장 1–5) |
| `dialecticDepth` | 변증법적 호출당 `.chat()` 패스 수 (1–3) | `1` |

이들은 서로 독립적입니다. 컨텍스트를 자주 새로 고치면서 변증법적 추론은 드물게 실행하거나, 낮은 빈도로 다중 패스 변증법적 추론을 깊게 실행할 수 있습니다. 예: `contextCadence: 1, dialecticCadence: 5, dialecticDepth: 2`는 매 턴 기본 컨텍스트를 새로 고치고, 5턴마다 변증법적 추론을 실행하며, 각 변증법적 실행에서 2개의 패스를 수행합니다.

### 변증법적 깊이 (다중 패스)

`dialecticDepth`가 1보다 크면 각 변증법적 호출에서 여러 `.chat()` 패스를 실행합니다.

- **패스 0**: 콜드 또는 웜 프롬프트 (위 설명 참조)
- **패스 1**: 자체 감사 — 초기 평가의 누락을 식별하고 최근 세션의 증거를 종합합니다.
- **패스 2**: 조정 — 이전 패스 사이의 모순을 확인하고 최종 종합 결과를 만듭니다.

각 패스는 비례적인 추론 수준을 사용합니다(초기 패스는 더 가볍고, 주요 패스는 기본 수준). `dialecticDepthLevels`로 패스별 수준을 덮어쓸 수 있습니다. 예: 깊이 3 실행에 `['minimal', 'medium', 'high']`를 지정합니다.

이전 패스가 강한 신호(길고 구조화된 출력)를 반환하면 패스가 조기에 종료되므로, 깊이 3이 항상 LLM 호출 3회를 의미하지는 않습니다.

### 세션 시작 프리워밍

세션 초기화 시 Honcho는 백그라운드에서 설정된 전체 `dialecticDepth`로 변증법적 호출을 실행하고, 그 결과를 턴 1의 컨텍스트 구성에 직접 전달합니다. 콜드 피어에서 단일 패스 프리워밍을 수행하면 출력이 빈약한 경우가 많지만, 다중 패스 깊이를 사용하면 사용자가 말을 시작하기 전에 감사/조정 사이클이 실행됩니다. 턴 1까지 프리워밍 결과가 도착하지 않으면 턴 1은 제한된 타임아웃으로 동기 호출을 수행합니다.

### 쿼리 적응형 추론 수준

자동 주입되는 변증법적 추론은 쿼리 길이에 따라 `dialecticReasoningLevel`을 조정합니다. 길이가 120자 이상이면 수준을 +1, 400자 이상이면 +2 높이고, `reasoningLevelCap`(기본값 `"high"`)에서 제한합니다. `reasoningHeuristic: false`로 비활성화하면 모든 자동 호출이 `dialecticReasoningLevel`로 고정됩니다. 사용 가능한 수준: `minimal`, `low`, `medium`, `high`, `max`.

## 설정 옵션

Honcho는 전역 `~/.honcho/config.json` 또는 프로필 로컬 `$HERMES_HOME/honcho.json`에서 설정합니다. 설정 마법사가 이 과정을 대신 처리합니다.

### 인증을 사용하는 셀프 호스팅 Honcho

Hermes를 셀프 호스팅 Honcho 서버에 연결할 때 `hermes honcho setup`(및 `hermes memory setup`)은 기본 URL 다음에 **로컬 JWT / bearer 토큰**을 요청합니다. 서버의 `AUTH_JWT_SECRET`(Honcho compose 환경 변수)로 서명된 JWT를 붙여 넣으면 인증된 액세스가 활성화되고, `AUTH_USE_AUTH=false`로 실행 중인 서버라면 비워 두세요. 로컬 토큰은 호스트 블록(`honcho.json`의 `hosts.<host>.apiKey`) 아래에 저장되며 클라우드 `apiKey`와 분리됩니다. 따라서 나중에 `Cloud or local?` 프롬프트를 다시 `cloud`로 전환해도 어느 자격 증명도 잃지 않습니다.

### 전체 설정 레퍼런스

| 키 | 기본값 | 설명 |
|-----|---------|-------------|
| `contextTokens` | `null` (제한 없음) | 턴마다 자동 주입되는 컨텍스트의 토큰 예산입니다. 제한하려면 정수(예: 1200)로 설정합니다. 단어 경계에서 자릅니다. |
| `contextCadence` | `1` | `context()` API 호출(기본 계층 새로 고침) 사이의 최소 턴 수입니다. |
| `dialecticCadence` | `2` | `peer.chat()` LLM 호출(변증법적 계층 새로 고침) 사이의 최소 턴 수입니다. 1–5를 권장합니다. `tools` 모드에서는 모델 호출이 명시적으로 이루어지므로 무관합니다. |
| `dialecticDepth` | `1` | 변증법적 호출당 `.chat()` 패스 수입니다. 1–3으로 제한됩니다. |
| `dialecticDepthLevels` | `null` | 패스별 추론 수준의 선택적 배열입니다(예: `["minimal", "low", "medium"]`). 비례 기본값을 덮어씁니다. |
| `dialecticReasoningLevel` | `'low'` | 기본 추론 수준입니다: `minimal`, `low`, `medium`, `high`, `max` |
| `dialecticDynamic` | `true` | `true`이면 모델이 도구 매개변수를 통해 호출별 추론 수준을 덮어쓸 수 있습니다. |
| `dialecticMaxChars` | `600` | 시스템 프롬프트에 주입되는 변증법적 결과의 최대 문자 수입니다. |
| `recallMode` | `'hybrid'` | `hybrid` (자동 주입 + 도구), `context` (주입만), `tools` (도구만) |
| `writeFrequency` | `'async'` | 메시지를 플러시하는 시점입니다: `async` (백그라운드 스레드), `turn` (동기), `session` (종료 시 일괄 처리), 또는 정수 N |
| `saveMessages` | `true` | 메시지를 Honcho API에 저장할지 여부입니다. |
| `observationMode` | `'directional'` | `directional` (모두 켜짐) 또는 `unified` (공유 풀)입니다. 세밀한 제어를 위해 `observation` 객체로 덮어쓸 수 있습니다. |
| `messageMaxChars` | `25000` | `add_messages()`를 통해 전송하는 메시지당 최대 문자 수입니다. 초과하면 청크로 나눕니다. |
| `dialecticMaxInputChars` | `10000` | `peer.chat()`의 변증법적 쿼리 입력에 허용되는 최대 문자 수입니다. |
| `sessionStrategy` | `'per-directory'` | `per-directory`, `per-repo`, `per-session`, 또는 `global`입니다. |
| `pinUserPeer` | `false` | 게이트웨이 전용입니다. `true`이면 모든 플랫폼 사용자가 `peerName`으로 통합됩니다. |
| `userPeerAliases` | `{}` | 런타임 ID를 피어에 매핑합니다(`{"7654321": "alice"}`). 다대일 매핑입니다. |
| `runtimePeerPrefix` | `""` | 별칭이 일치하지 않을 때 알 수 없는 런타임 ID에 네임스페이스를 지정합니다(`telegram_7654321`). |

**세션 전략**은 Honcho 세션이 작업에 매핑되는 방식을 제어합니다.
- `per-session` — 각 `hermes` 실행마다 새 세션을 생성합니다. 깔끔하게 시작하고 도구를 통해 메모리를 사용합니다. 신규 사용자에게 권장합니다.
- `per-directory` — 작업 디렉터리마다 하나의 Honcho 세션을 사용합니다. 실행 간 컨텍스트가 누적됩니다.
- `per-repo` — git 저장소마다 하나의 세션을 사용합니다.
- `global` — 모든 디렉터리에서 하나의 세션을 사용합니다.

**리콜 모드**는 메모리가 대화로 유입되는 방식을 제어합니다.
- `hybrid` — 시스템 프롬프트에 컨텍스트를 자동 주입하고 도구도 사용할 수 있습니다(모델이 쿼리 시점을 결정).
- `context` — 자동 주입만 수행하며 도구는 숨깁니다.
- `tools` — 도구만 사용하고 자동 주입은 하지 않습니다. 에이전트가 `honcho_reasoning`, `honcho_search` 등을 명시적으로 호출해야 합니다.

**리콜 모드별 설정:**

| 설정 | `hybrid` | `context` | `tools` |
|---------|----------|-----------|---------|
| `writeFrequency` | 메시지 플러시 | 메시지 플러시 | 메시지 플러시 |
| `contextCadence` | 기본 컨텍스트 새로 고침 제어 | 기본 컨텍스트 새로 고침 제어 | 무관 — 주입하지 않음 |
| `dialecticCadence` | 자동 LLM 호출 제어 | 자동 LLM 호출 제어 | 무관 — 모델이 명시적으로 호출 |
| `dialecticDepth` | 호출당 다중 패스 | 호출당 다중 패스 | 무관 — 모델이 명시적으로 호출 |
| `contextTokens` | 주입량 제한 | 주입량 제한 | 무관 — 주입하지 않음 |
| `dialecticDynamic` | 모델 덮어쓰기 제어 | 해당 없음 (도구 없음) | 모델 덮어쓰기 제어 |

`tools` 모드에서는 모델이 완전히 제어합니다. 모델이 원할 때 `honcho_reasoning`을 호출하고, 선택한 `reasoning_level`로 원하는 깊이를 지정합니다. cadence 및 예산 설정은 자동 주입(`hybrid` 및 `context`)이 있는 모드에만 적용됩니다.

## 게이트웨이 ID 매핑

이 설정은 [Hermes 게이트웨이](../../developer-guide/gateway-internals.md)를 실행할 때만 의미가 있습니다. 게이트웨이는 플랫폼별 런타임 ID(Telegram UID, Discord snowflake, Slack 사용자)를 사용자가 들어오는 진입점으로 사용하기 때문입니다. CLI, TUI, 데스크톱 세션에는 런타임 ID가 없으며 항상 `peerName`으로 확인되므로, 게이트웨이 외부에서는 이 키들이 아무런 작동도 하지 않습니다.

설정 마법사는 게이트웨이 플랫폼이 연결되어 있는지 확인하며, 연결되어 있지 않으면 이 단계를 완전히 건너뜁니다. 실행되면 한 가지 질문을 합니다 — *누가 이 게이트웨이와 대화하나요?* — 그리고 그 답에서 키를 도출합니다.

| 답변 | 결과 |
|--------|--------|
| **나만** | `pinUserPeer: true` — 에이전트가 아닌 모든 게이트웨이 사용자를 사용자의 피어로 통합합니다. 고정은 모든 별칭보다 우선하므로, 사용자 측 ID마다 별도의 피어가 필요하지 않을 때만 선택하세요. 여러 에이전트가 게이트웨이에 접근하고 각각 별도의 피어가 필요하다면 고정하지 말고 `pinUserPeer: false`로 둔 뒤 `userPeerAliases`(`[e]` 편집기)로 매핑하세요. |
| **나 + 다른 사람들** (풀링) | `pinUserPeer: false` + 런타임 ID를 `peerName`에 매핑하는 `userPeerAliases` — 사용자는 공유 기록을 유지하고 다른 사람들은 각자의 피어를 사용합니다. |
| **다른 사람들만** | `pinUserPeer: false`, 선택적 `runtimePeerPrefix` — 각 사용자가 자신의 피어를 사용합니다. |

프롬프트에서 `[e]`를 선택하면 세 키를 직접 설정할 수 있습니다.

리졸버는 위에서 아래 순서로 키를 시도하며, 처음 일치하는 항목을 사용합니다: `pinUserPeer` → `userPeerAliases[id]` → `runtimePeerPrefix + id` → 원시 런타임 ID → `peerName` → 세션 키 대체값.

:::warning 고정을 해제하면 풀링된 메모리가 고아가 됩니다
`pinUserPeer`를 `true`에서 `false`로 전환해도 데이터는 마이그레이션되지 않습니다. `peerName` 아래에 축적된 메모리는 그곳에 남고, 플랫폼 사용자는 비어 있는 새 피어로 확인됩니다. 자신의 연속성을 유지하려면 런타임 ID가 `peerName`으로 다시 별칭 지정되는 **풀링** 경로를 선택하세요. 마법사는 전환을 감지하면 이 경로를 자동으로 안내합니다.
:::

:::note 사용 중단된 키
`pinPeerName`은 `pinUserPeer`의 레거시 별칭입니다 — 이전 버전과의 호환성을 위해 계속 읽지만(`pinUserPeer`와 둘 다 설정된 경우 `pinUserPeer`가 우선), 기록하지는 않습니다. 설정을 다시 실행하면 정식 키로 마이그레이션됩니다.
:::

## 관찰 (방향성 vs. 통합)

Honcho는 대화를 메시지를 주고받는 피어들로 모델링합니다. 각 피어에는 Honcho의 `SessionPeerConfig`에 1:1로 매핑되는 두 가지 관찰 토글이 있습니다.

| 토글 | 효과 |
|--------|--------|
| `observeMe` | Honcho가 이 피어의 자체 메시지로 해당 피어의 표현을 구축합니다. |
| `observeOthers` | 이 피어가 다른 피어의 메시지를 관찰합니다(피어 간 추론에 반영). |

두 피어 × 두 토글 = 네 개의 플래그입니다. `observationMode`는 미리 설정된 단축 표현입니다.

| 프리셋 | 사용자 플래그 | AI 플래그 | 의미 |
|--------|-----------|----------|-----------|
| `"directional"` (기본값) | me: on, others: on | me: on, others: on | 완전한 상호 관찰입니다. 피어 간 변증법적 추론을 활성화합니다 — "사용자가 말한 내용과 AI가 답한 내용을 바탕으로 AI가 사용자에 대해 무엇을 아는가." |
| `"unified"` | me: on, others: off | me: off, others: on | 공유 풀 의미론입니다 — AI는 사용자의 메시지만 관찰하고 사용자 피어는 자기 모델만 구축합니다. 단일 관찰자 풀입니다. |

명시적인 `observation` 블록으로 프리셋을 덮어써 피어별로 제어할 수 있습니다.

```json
"observation": {
  "user": { "observeMe": true,  "observeOthers": true },
  "ai":   { "observeMe": true,  "observeOthers": false }
}
```

일반적인 패턴:

| 의도 | 설정 |
|--------|--------|
| 완전한 관찰 (대부분의 사용자) | `"observationMode": "directional"` |
| AI가 자신의 답변으로 사용자를 다시 모델링하지 않아야 함 | `"ai": {"observeMe": true, "observeOthers": false}` |
| AI 피어가 자기 관찰로 업데이트되지 않아야 하는 강한 페르소나 | `"ai": {"observeMe": false, "observeOthers": true}` |

[Honcho 대시보드](https://app.honcho.dev)에서 설정한 서버 측 토글은 로컬 기본값보다 우선합니다 — Hermes는 세션 초기화 시 이를 다시 동기화합니다.

## 도구

Honcho가 메모리 제공자로 활성화되면 다섯 가지 도구를 사용할 수 있습니다.

| 도구 | 용도 |
|------|---------|
| `honcho_profile` | 피어 카드 읽기 또는 업데이트 — 업데이트하려면 `card`(사실 목록)를 전달하고, 읽으려면 생략합니다. |
| `honcho_search` | 컨텍스트에 대한 시맨틱 검색 — LLM 합성 없이 원시 발췌문을 반환합니다. |
| `honcho_context` | 전체 세션 컨텍스트 — 요약, 표현, 카드, 최근 메시지입니다. |
| `honcho_reasoning` | Honcho의 LLM에서 합성한 답변 — 깊이를 제어하려면 `reasoning_level`(minimal/low/medium/high/max)을 전달합니다. |
| `honcho_conclude` | 결론 생성 또는 삭제 — 생성하려면 `conclusion`, 삭제하려면 `delete_id`(PII만)를 전달합니다. |

## CLI 명령

`hermes honcho` 하위 명령은 Honcho가 활성 메모리 제공자일 때만 등록됩니다(`config.yaml`에서 `memory.provider: honcho`). 새로 설치한 경우 `hermes memory setup honcho`로 Honcho를 직접 설정하거나(`hermes memory setup`을 실행하고 목록에서 선택할 수도 있음), 다음 호출부터 `hermes honcho` 하위 명령이 나타납니다.

```bash
hermes memory setup honcho    # Configure Honcho directly (works before activation)
hermes honcho status          # Connection status, config, and key settings
hermes honcho setup           # Redirects to `hermes memory setup` (post-activation alias)
hermes honcho strategy        # Show or set session strategy (per-session/per-directory/per-repo/global)
hermes honcho peer            # Show or update peer names + dialectic reasoning level
hermes honcho mode            # Show or set recall mode (hybrid/context/tools)
hermes honcho tokens          # Show or set token budget for context and dialectic
hermes honcho identity        # Seed or show the AI peer's Honcho identity
hermes honcho sync            # Sync Honcho config to all existing profiles
hermes honcho peers           # Show peer identities across all profiles
hermes honcho sessions        # List known Honcho session mappings
hermes honcho map             # Map current directory to a Honcho session name
hermes honcho enable          # Enable Honcho for the active profile
hermes honcho disable         # Disable Honcho for the active profile
hermes honcho migrate         # Step-by-step migration guide from openclaw-honcho
```

## `hermes honcho`에서 마이그레이션

이전에 독립 실행형 `hermes honcho setup`을 사용했다면 다음과 같습니다.

1. 기존 설정(`honcho.json` 또는 `~/.honcho/config.json`)은 보존됩니다.
2. 서버 측 데이터(메모리, 결론, 사용자 프로필)는 그대로 유지됩니다.
3. 다시 활성화하려면 config.yaml에서 `memory.provider: honcho`를 설정합니다.

다시 로그인하거나 다시 설정할 필요가 없습니다. `hermes memory setup`을 실행하고 "honcho"를 선택하면 마법사가 기존 설정을 감지합니다.

## 전체 문서

전체 레퍼런스는 [메모리 제공자 — Honcho](./memory-providers.md#honcho)를 참조하세요.
