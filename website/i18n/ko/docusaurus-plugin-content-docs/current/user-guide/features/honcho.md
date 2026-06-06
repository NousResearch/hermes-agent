---
sidebar_position: 99
title: "Honcho 메모리 (Honcho Memory)"
description: "Honcho를 통한 AI 네이티브 영구 메모리 — 변증법적 추론(dialectic reasoning), 다중 에이전트 사용자 모델링, 그리고 깊은 개인화(deep personalization)"
---

# Honcho 메모리 (Honcho Memory)

[Honcho](https://github.com/plastic-labs/honcho)는 Hermes의 기본 내장 메모리 시스템 위에 변증법적 추론과 깊은 사용자 모델링을 추가하는 AI 네이티브 메모리 백엔드입니다. 단순한 키-값(key-value) 저장소 대신, Honcho는 대화가 일어난 후에 대화를 분석하여 사용자가 명시적으로 말한 것을 넘어서는 사용자의 선호도, 의사소통 스타일, 목표, 패턴 등을 추론하며 사용자가 누구인지에 대한 모델을 지속적으로 유지합니다.

:::info Honcho는 메모리 공급자 플러그인입니다
Honcho는 [메모리 공급자 (Memory Providers)](./memory-providers.md) 시스템에 통합되어 있습니다. 아래의 모든 기능은 통합된 메모리 공급자 인터페이스를 통해 사용할 수 있습니다.
:::

## Honcho가 추가하는 기능

| 기능 | 내장 메모리 | Honcho |
|-----------|----------------|--------|
| 세션 간 영구 저장 | ✔ 파일 기반 MEMORY.md/USER.md | ✔ API를 통한 서버 측 저장 |
| 사용자 프로필 | ✔ 에이전트의 수동 큐레이션 | ✔ 자동 변증법적 추론 |
| 세션 요약 | — | ✔ 세션 범위의 컨텍스트 주입 |
| 다중 에이전트 격리 | — | ✔ 피어(peer)별 프로필 분리 |
| 관찰 모드 | — | ✔ 통합(unified) 또는 방향성(directional) 관찰 |
| 결론 (도출된 인사이트) | — | ✔ 패턴에 대한 서버 측 추론 |
| 히스토리 전체 검색 | ✔ FTS5 세션 검색 | ✔ 결론에 대한 의미론적(semantic) 검색 |

**변증법적 추론 (Dialectic reasoning)**: 대화의 각 턴(turn) 이후(`dialecticCadence`에 의해 조절됨), Honcho는 대화를 분석하여 사용자의 선호도, 습관, 목표에 대한 인사이트를 도출합니다. 이러한 인사이트는 시간이 지남에 따라 축적되어, 사용자가 명시적으로 말한 것을 넘어선 깊이 있는 이해를 에이전트에게 제공합니다. 이 변증법적 기능은 멀티패스 깊이(multi-pass depth, 1~3번의 반복)를 지원하며, 자동으로 콜드(cold)/웜(warm) 프롬프트를 선택합니다 — 콜드 스타트 쿼리는 사용자에 대한 일반적인 사실에 초점을 맞추고, 웜 쿼리는 세션 범위의 컨텍스트를 우선시합니다.

**세션 범위의 컨텍스트 (Session-scoped context)**: 이제 기본 컨텍스트에는 사용자 표현(user representation) 및 피어 카드(peer card)와 함께 세션 요약이 포함됩니다. 이를 통해 에이전트는 현재 세션에서 이미 논의된 내용을 인식하게 되어 반복을 줄이고 대화의 연속성을 보장합니다.

**다중 에이전트 프로필 (Multi-agent profiles)**: 여러 Hermes 인스턴스(예: 코딩 어시스턴트 및 개인 비서)가 동일한 사용자와 대화할 때 Honcho는 별도의 "피어(peer)" 프로필을 유지합니다. 각 피어는 자신만의 관찰 내용과 결론만을 보게 되므로 컨텍스트 간의 오염을 방지할 수 있습니다.

## 설정

```bash
hermes memory setup    # 공급자 목록에서 "honcho"를 선택하세요
```

또는 수동으로 구성하세요:

```yaml
# ~/.hermes/config.yaml
memory:
  provider: honcho
```

```bash
echo 'HONCHO_API_KEY=***' >> ~/.hermes/.env
```

[honcho.dev](https://honcho.dev)에서 API 키를 받을 수 있습니다.

## 아키텍처

### 2계층 컨텍스트 주입 (Two-Layer Context Injection)

매 턴마다 (`hybrid` 또는 `context` 모드에서), Honcho는 시스템 프롬프트에 주입할 두 계층의 컨텍스트를 조립합니다:

1. **기본 컨텍스트 (Base context)** — 세션 요약, 사용자 표현, 사용자 피어 카드, AI 자아 표현(AI self-representation), 그리고 AI 아이덴티티 카드입니다. `contextCadence`에 따라 갱신됩니다. 이는 "이 사용자는 누구인가" 계층입니다.
2. **변증법적 보충 (Dialectic supplement)** — 현재 사용자의 상태와 요구에 대한 LLM 합성 추론(reasoning)입니다. `dialecticCadence`에 따라 갱신됩니다. 이는 "지금 무엇이 중요한가" 계층입니다.

두 계층 모두 연결(concatenated)되고, `contextTokens` 예산이 설정된 경우 그에 맞게 잘려 주입됩니다.

### 콜드/웜 프롬프트 선택 (Cold/Warm Prompt Selection)

변증법적 추론은 자동으로 두 가지 프롬프트 전략 중 하나를 선택합니다:

- **콜드 스타트 (Cold start)** (아직 기본 컨텍스트가 없음): 일반적인 질문 — "이 사람은 누구입니까? 선호도, 목표 및 작업 스타일은 무엇입니까?"
- **웜 세션 (Warm session)** (기본 컨텍스트가 존재함): 세션 범위 쿼리 — "지금까지 이 세션에서 논의된 내용을 고려할 때, 이 사용자에 대한 가장 관련성 있는 맥락은 무엇입니까?"

이 과정은 기본 컨텍스트가 채워졌는지 여부에 따라 자동으로 이루어집니다.

### 세 가지 독립적인 설정 조정 (Three Orthogonal Config Knobs)

비용과 깊이는 세 가지 독립적인 설정에 의해 제어됩니다:

| 설정 (Knob) | 제어 대상 | 기본값 |
|------|----------|---------|
| `contextCadence` | `context()` API 호출(기본 계층 갱신) 간의 턴 수 | `1` |
| `dialecticCadence` | `peer.chat()` LLM 호출(변증법적 계층 갱신) 간의 턴 수 | `2` (1~5 권장) |
| `dialecticDepth` | 변증법적 호출당 발생하는 `.chat()` 반복(pass) 횟수 (1–3) | `1` |

이들은 서로 독립적(orthogonal)입니다 — 잦은 컨텍스트 갱신과 드문 변증법적 추론을 조합하거나, 빈도는 낮지만 깊이 있는 멀티패스 변증법적 추론을 사용할 수 있습니다. 예시: `contextCadence: 1, dialecticCadence: 5, dialecticDepth: 2`는 매 턴마다 기본 컨텍스트를 갱신하고, 5턴마다 변증법적 추론을 실행하며, 매 변증법적 추론 실행 시 2번의 패스(pass)를 진행합니다.

### 변증법적 깊이 (멀티패스) (Dialectic Depth (Multi-Pass))

`dialecticDepth` > 1인 경우, 각 변증법적 호출은 여러 번의 `.chat()` 패스를 실행합니다:

- **패스 0**: 콜드 또는 웜 프롬프트 (위의 내용 참조)
- **패스 1**: 자가 진단 (Self-audit) — 초기 평가의 빈틈을 식별하고 최근 세션의 증거를 종합합니다.
- **패스 2**: 조정 (Reconciliation) — 이전 패스 간의 모순을 확인하고 최종 종합본을 도출합니다.

각 패스는 비례하는 추론 수준(reasoning level)을 사용합니다(초기 패스는 더 가볍게, 메인 패스는 기본 수준으로). `dialecticDepthLevels`를 사용하여 패스별 레벨을 재정의할 수 있습니다 — 예: 깊이 3 실행에 대해 `["minimal", "medium", "high"]`.

이전 패스에서 강한 신호(길고 구조화된 출력)를 반환하면 후속 패스는 조기에 종료(bail out)되므로, 깊이가 3이라고 해서 항상 3번의 LLM 호출이 발생하는 것은 아닙니다.

### 세션 시작 프리웜 (Session-Start Prewarm)

세션 초기화 시, Honcho는 백그라운드에서 설정된 최대 `dialecticDepth`로 변증법적 호출을 실행하고 그 결과를 턴 1의 컨텍스트 조립에 직접 전달합니다. 아무런 정보가 없는 콜드 피어(cold peer)에서 단일 패스(single-pass) 프리웜은 대개 빈약한 결과를 반환하지만 — 멀티패스 깊이를 설정하면 사용자가 처음 말하기도 전에 자체 검증 및 조정(audit/reconcile) 주기를 거치게 됩니다. 턴 1 시작 전까지 프리웜 결과가 도착하지 않으면, 턴 1은 정해진 시간 제한 내에서 동기식(synchronous) 호출로 폴백합니다.

### 쿼리 적응형 추론 수준 (Query-Adaptive Reasoning Level)

자동 주입되는 변증법적 추론은 쿼리 길이에 따라 `dialecticReasoningLevel`을 확장합니다: 120자 이상이면 +1 레벨, 400자 이상이면 +2 레벨이 추가되며, `reasoningLevelCap`(기본값 `"high"`)까지만 올라갑니다. 자동 호출 시마다 `dialecticReasoningLevel`을 고정하고 싶다면 `reasoningHeuristic: false`로 비활성화하세요. 사용 가능한 레벨: `minimal`, `low`, `medium`, `high`, `max`.

## 구성 옵션 (Configuration Options)

Honcho는 전역 설정인 `~/.honcho/config.json` 또는 프로필 로컬 설정인 `$HERMES_HOME/honcho.json`에서 구성됩니다. 설정 마법사(setup wizard)가 이를 대신 처리해 줍니다.

### 인증을 적용한 자체 호스팅 Honcho (Self-Hosted Honcho with Authentication)

자체 호스팅되는 Honcho 서버로 Hermes를 가리킬 때, `hermes honcho setup` (및 `hermes memory setup`)은 기본 URL을 입력받은 후 **로컬 JWT / bearer 토큰**을 요청합니다. `AUTH_USE_AUTH=false`로 실행 중인 서버라면 빈 칸으로 두면 되고, 인증된 액세스를 활성화하려면 서버의 `AUTH_JWT_SECRET` (Honcho compose 환경 변수)으로 서명된 JWT를 붙여넣으세요. 로컬 토큰은 클라우드 `apiKey`와 분리되어 호스트 블록(`honcho.json`의 `hosts.<host>.apiKey`) 아래에 저장되므로, 나중에 둘 중 어느 쪽의 자격 증명도 잃지 않고 `Cloud or local?` 프롬프트에서 `cloud`로 되돌릴 수 있습니다.

### 전체 설정 참조

| 키 | 기본값 | 설명 |
|-----|---------|-------------|
| `contextTokens` | `null` (제한 없음) | 턴당 자동 주입되는 컨텍스트의 토큰 예산. 정수(예: 1200)로 설정하여 제한 가능. 단어 경계에서 자릅니다 |
| `contextCadence` | `1` | `context()` API 호출(기본 계층 갱신) 간의 최소 턴 수 |
| `dialecticCadence` | `2` | `peer.chat()` LLM 호출(변증법적 계층) 간의 최소 턴 수. 1~5 권장. `tools` 모드에서는 모델이 명시적으로 호출하므로 의미가 없습니다 |
| `dialecticDepth` | `1` | 변증법적 호출당 발생하는 `.chat()` 반복(pass) 횟수. 1–3으로 제한됨 |
| `dialecticDepthLevels` | `null` | 패스당 추론 수준을 담은 선택적 배열 (예: `["minimal", "low", "medium"]`). 비례적 할당 기본값을 덮어씁니다 |
| `dialecticReasoningLevel` | `'low'` | 기본 추론 수준: `minimal`, `low`, `medium`, `high`, `max` |
| `dialecticDynamic` | `true` | `true`인 경우, 모델이 도구(tool) 파라미터를 통해 호출마다 추론 수준을 덮어쓸 수 있습니다 |
| `dialecticMaxChars` | `600` | 시스템 프롬프트에 주입될 변증법적 결과의 최대 문자 수 |
| `recallMode` | `'hybrid'` | `hybrid` (자동 주입 + 도구), `context` (주입만), `tools` (도구만) |
| `writeFrequency` | `'async'` | 메시지를 언제 기록(flush)할지: `async` (백그라운드 스레드), `turn` (동기식), `session` (종료 시 일괄 처리), 또는 정수 N |
| `saveMessages` | `true` | 메시지를 Honcho API에 보존할지 여부 |
| `observationMode` | `'directional'` | `directional` (모두 켜짐) 또는 `unified` (공유 풀). 세부 제어를 위해 `observation` 객체로 덮어쓸 수 있습니다 |
| `messageMaxChars` | `25000` | `add_messages()`를 통해 전송되는 메시지당 최대 문자 수. 초과 시 청크 단위로 나뉩니다 |
| `dialecticMaxInputChars` | `10000` | `peer.chat()`에 들어가는 변증법적 쿼리 입력의 최대 문자 수 |
| `sessionStrategy` | `'per-directory'` | `per-directory`, `per-repo`, `per-session`, 또는 `global` |

**세션 전략 (Session strategy)**은 Honcho 세션이 당신의 작업 공간에 어떻게 매핑되는지 제어합니다:
- `per-session` — `hermes`를 실행할 때마다 새로운 세션을 얻습니다. 초기화된 상태에서 시작하고 메모리는 도구를 통해 접근합니다. 신규 사용자에게 권장합니다.
- `per-directory` — 작업 디렉터리당 하나의 Honcho 세션을 둡니다. 실행할 때마다 컨텍스트가 축적됩니다.
- `per-repo` — git 리포지토리당 하나의 세션을 둡니다.
- `global` — 모든 디렉터리에 걸쳐 단일 세션을 유지합니다.

**회상 모드 (Recall mode)**는 대화에 메모리가 유입되는 방식을 제어합니다:
- `hybrid` — 컨텍스트가 시스템 프롬프트에 자동 주입되며(AND) 도구도 사용할 수 있습니다(모델이 언제 쿼리할지 스스로 결정합니다).
- `context` — 자동 주입만 이루어지며 도구는 숨겨집니다.
- `tools` — 도구만 제공되며 자동 주입은 없습니다. 에이전트가 명시적으로 `honcho_reasoning`, `honcho_search` 등을 호출해야 합니다.

**회상 모드별 설정 적용 여부:**

| 설정 | `hybrid` | `context` | `tools` |
|---------|----------|-----------|---------|
| `writeFrequency` | 메시지를 기록합니다 | 메시지를 기록합니다 | 메시지를 기록합니다 |
| `contextCadence` | 기본 컨텍스트 갱신 제어 | 기본 컨텍스트 갱신 제어 | 무관함 — 주입 없음 |
| `dialecticCadence` | 자동 LLM 호출 제어 | 자동 LLM 호출 제어 | 무관함 — 모델이 명시적으로 호출 |
| `dialecticDepth` | 호출당 멀티패스 수행 | 호출당 멀티패스 수행 | 무관함 — 모델이 명시적으로 호출 |
| `contextTokens` | 주입량 제한 | 주입량 제한 | 무관함 — 주입 없음 |
| `dialecticDynamic` | 모델의 덮어쓰기 허용 | 해당 없음 (도구 없음) | 모델의 덮어쓰기 허용 |

`tools` 모드에서는 모델이 완전한 통제권을 가집니다 — 원하는 시점에, 원하는 `reasoning_level`로 `honcho_reasoning`을 호출합니다. 호출 주기와 예산 설정은 자동 주입 기능이 있는 모드(`hybrid` 및 `context`)에만 적용됩니다.

## 관찰: 방향성 vs 통합 (Observation: Directional vs. Unified)

Honcho는 대화를 메시지를 교환하는 "피어(peers)" 간의 관계로 모델링합니다. 각 피어에는 Honcho의 `SessionPeerConfig`와 1:1로 매핑되는 두 가지 관찰 토글이 있습니다:

| 토글 | 효과 |
|--------|--------|
| `observeMe` | Honcho가 이 피어 자신의 메시지들을 바탕으로 이 피어에 대한 모델(representation)을 구축합니다 |
| `observeOthers` | 이 피어가 다른 피어의 메시지를 관찰합니다 (피어 간 추론의 자료로 쓰입니다) |

두 피어 × 두 토글 = 네 개의 플래그. `observationMode`는 이를 쉽게 설정하기 위한 프리셋(preset) 단축키입니다:

| 프리셋 | User 플래그 | AI 플래그 | 의미 (Semantics) |
|--------|-----------|----------|-----------|
| `"directional"` (기본값) | me: 켜짐, others: 켜짐 | me: 켜짐, others: 켜짐 | 완전한 상호 관찰. 피어 간 변증법(cross-peer dialectic)을 활성화합니다 — "사용자가 한 말과 AI가 대답한 내용을 바탕으로 AI는 사용자에 대해 무엇을 아는가." |
| `"unified"` | me: 켜짐, others: 꺼짐 | me: 꺼짐, others: 켜짐 | 공유 풀(shared-pool) 의미론. AI는 사용자의 메시지만 관찰하고, 사용자 피어는 자기 자신만 모델링합니다. 단일 관찰자 풀. |

세부 제어를 위해 `observation` 블록을 명시적으로 작성하여 프리셋을 덮어쓸 수 있습니다:

```json
"observation": {
  "user": { "observeMe": true,  "observeOthers": true },
  "ai":   { "observeMe": true,  "observeOthers": false }
}
```

일반적인 패턴:

| 목적 (Intent) | 설정 (Config) |
|--------|--------|
| 완전한 관찰 (대부분의 사용자) | `"observationMode": "directional"` |
| AI가 자신의 답변으로부터 사용자를 다시 모델링해서는 안 될 때 | `"ai": {"observeMe": true, "observeOthers": false}` |
| AI 피어가 자가 관찰을 통해 업데이트되어선 안 되는 확고한 페르소나일 때 | `"ai": {"observeMe": false, "observeOthers": true}` |

[Honcho 대시보드](https://app.honcho.dev)를 통한 서버 측 토글 설정이 로컬 기본값보다 우선합니다 — Hermes는 세션 초기화 시 서버 측 설정을 다시 동기화합니다.

## 도구 (Tools)

Honcho가 메모리 공급자로 활성화되면 다음 5가지 도구를 사용할 수 있습니다:

| 도구 | 목적 |
|------|---------|
| `honcho_profile` | 피어 카드(peer card) 읽기 또는 업데이트 — `card`(사실들의 목록)를 전달하면 업데이트, 생략하면 읽기 |
| `honcho_search` | 컨텍스트에 대한 의미론적 검색(Semantic search) — LLM 합성 없이 원본 발췌(excerpts)만 제공 |
| `honcho_context` | 전체 세션 컨텍스트 — 요약, 표현(representation), 카드, 최근 메시지 제공 |
| `honcho_reasoning` | Honcho LLM이 합성한 답변 제공 — `reasoning_level` (minimal/low/medium/high/max)을 전달하여 추론 깊이를 조절 가능 |
| `honcho_conclude` | 결론(conclusion) 생성 또는 삭제 — 생성 시 `conclusion`을 전달, 삭제 시 `delete_id` 전달 (PII 삭제용) |

## CLI 명령어

`hermes honcho` 하위 명령어는 **Honcho가 활성 메모리 공급자일 때만(`config.yaml`에 `memory.provider: honcho`) 등록됩니다.** 새로 설치하는 경우 `hermes memory setup honcho`를 통해 Honcho를 직접 설정하거나 (`hermes memory setup`을 실행하고 목록에서 선택), 다음 실행 시에 `hermes honcho` 하위 명령어가 나타납니다.

```bash
hermes memory setup honcho    # Honcho를 직접 설정합니다 (활성화 이전에도 작동)
hermes honcho status          # 연결 상태, 설정, 및 주요 키 설정 확인
hermes honcho setup           # `hermes memory setup`으로 리디렉션됩니다 (활성화 이후의 별칭)
hermes honcho strategy        # 세션 전략 표시 또는 설정 (per-session/per-directory/per-repo/global)
hermes honcho peer            # 피어 이름과 변증법적 추론 수준 표시 또는 업데이트
hermes honcho mode            # 회상(recall) 모드 표시 또는 설정 (hybrid/context/tools)
hermes honcho tokens          # 컨텍스트와 변증법적 추론의 토큰 예산 표시 또는 설정
hermes honcho identity        # AI 피어의 Honcho 아이덴티티 부여(seed) 또는 표시
hermes honcho sync            # Honcho 구성을 모든 기존 프로필에 동기화
hermes honcho peers           # 모든 프로필에 걸친 피어 아이덴티티 표시
hermes honcho sessions        # 알려진 Honcho 세션 매핑 목록 표시
hermes honcho map             # 현재 디렉터리를 Honcho 세션 이름에 매핑
hermes honcho enable          # 활성 프로필에 대해 Honcho 활성화
hermes honcho disable         # 활성 프로필에 대해 Honcho 비활성화
hermes honcho migrate         # openclaw-honcho에서 마이그레이션하기 위한 단계별 가이드
```

## `hermes honcho`에서 마이그레이션하기

이전에 단독 명령어인 `hermes honcho setup`을 사용했었던 경우:

1. 기존 구성(`honcho.json` 또는 `~/.honcho/config.json`)은 그대로 보존됩니다.
2. 서버 측 데이터(메모리, 결론, 사용자 프로필)는 손상 없이 유지됩니다.
3. 재활성화를 위해 `config.yaml`에 `memory.provider: honcho`를 설정하세요.

다시 로그인하거나 다시 설정할 필요가 없습니다. `hermes memory setup`을 실행하고 "honcho"를 선택하면 — 마법사가 기존 구성을 자동으로 감지합니다.

## 전체 문서

전체 참조를 확인하려면 [메모리 공급자 — Honcho (Memory Providers — Honcho)](./memory-providers.md#honcho) 문서를 참조하세요.
