---
sidebar_position: 4
title: "메모리 공급자 (Memory Providers)"
description: "외부 메모리 공급자 플러그인 — Honcho, OpenViking, Mem0, Hindsight, Holographic, RetainDB, ByteRover, Supermemory"
---

# 메모리 공급자 (Memory Providers)

Hermes Agent는 내장된 MEMORY.md 및 USER.md를 넘어 에이전트에게 영구적인 교차 세션 지식을 제공하는 8개의 외부 메모리 공급자 플러그인과 함께 제공됩니다. 한 번에 **하나의** 외부 공급자만 활성화할 수 있습니다 — 내장 메모리는 항상 그 옆에 활성화되어 있습니다.

## 빠른 시작 (Quick Start)

```bash
hermes memory setup      # 대화형 선택기 + 구성
hermes memory status     # 무엇이 활성화되어 있는지 확인
hermes memory off        # 외부 공급자 비활성화
```

또한 `hermes plugins` → Provider Plugins → Memory Provider를 통해 활성 메모리 공급자를 선택할 수 있습니다.

또는 `~/.hermes/config.yaml`에서 수동으로 설정할 수 있습니다.

```yaml
memory:
  provider: openviking   # 또는 honcho, mem0, hindsight, holographic, retaindb, byterover, supermemory
```

## 작동 방식 (How It Works)

메모리 공급자가 활성화되면 Hermes는 자동으로 다음을 수행합니다.

1. **공급자 컨텍스트를 시스템 프롬프트에 주입** (공급자가 알고 있는 내용)
2. 각 턴 전에 **관련 메모리를 사전 가져오기(prefetch)** (백그라운드, 비차단)
3. 각 응답 후 **대화 턴을 공급자에 동기화**
4. 세션 종료 시 **메모리 추출** (지원하는 공급자의 경우)
5. **내장 메모리 쓰기를 외부 공급자에 미러링**
6. 에이전트가 메모리를 검색, 저장 및 관리할 수 있도록 **공급자별 도구를 추가**

내장 메모리(MEMORY.md / USER.md)는 이전과 똑같이 계속 작동합니다. 외부 공급자는 부가적입니다.

## 사용 가능한 공급자 (Available Providers)

### Honcho

변증법적 추론, 세션 범위 컨텍스트 주입, 시맨틱 검색, 그리고 지속적인 결론 도출(persistent conclusions)을 갖춘 AI 네이티브 교차 세션 사용자 모델링. 기본 컨텍스트에는 이제 사용자 표현(user representation) 및 피어 카드와 함께 세션 요약이 포함되어, 에이전트가 이미 논의된 내용을 인식할 수 있게 합니다.

| | |
|---|---|
| **가장 적합한 대상** | 교차 세션 컨텍스트, 사용자-에이전트 정렬이 있는 다중 에이전트 시스템 |
| **요구 사항** | `pip install honcho-ai` + [API 키](https://app.honcho.dev) 또는 자체 호스팅 인스턴스 |
| **데이터 저장소** | Honcho Cloud 또는 자체 호스팅 |
| **비용** | Honcho 요금제(클라우드) / 무료(자체 호스팅) |

**도구 (5개):** `honcho_profile` (피어 카드 읽기/업데이트), `honcho_search` (시맨틱 검색), `honcho_context` (세션 컨텍스트 — 요약, 표현, 카드, 메시지), `honcho_reasoning` (LLM이 합성함), `honcho_conclude` (결론 도출 생성/삭제)

**아키텍처:** 두 계층 컨텍스트 주입 — 기본 계층(`contextCadence`에서 새로 고침되는 세션 요약 + 표현 + 피어 카드)에 변증법적 보충(LLM 추론, `dialecticCadence`에서 새로 고침됨)이 더해집니다. 변증법은 기본 컨텍스트가 있는지 여부에 따라 콜드 스타트 프롬프트(일반적인 사용자 사실) 대 웜 프롬프트(세션 범위 컨텍스트)를 자동으로 선택합니다.

**세 가지 직교하는 설정 조절기(Orthogonal config knobs)** 가 비용과 깊이를 독립적으로 제어합니다.

- `contextCadence` — 기본 계층이 새로 고침되는 빈도 (API 호출 빈도)
- `dialecticCadence` — 변증법 LLM이 실행되는 빈도 (LLM 호출 빈도)
- `dialecticDepth` — 변증법 호출당 `.chat()` 패스의 수 (1–3, 추론의 깊이)

**설정 마법사:**
```bash
hermes memory setup        # "honcho"를 선택 — Honcho 전용 사후 설정(post-setup)을 실행합니다.
```

레거시 `hermes honcho setup` 명령은 여전히 작동하지만(이제 `hermes memory setup`으로 리디렉션됨), Honcho가 활성 메모리 공급자로 선택된 후에만 등록됩니다.

**구성:** `$HERMES_HOME/honcho.json` (프로필 로컬) 또는 `~/.honcho/config.json` (전역). 확인(Resolution) 순서: `$HERMES_HOME/honcho.json` > `~/.hermes/honcho.json` > `~/.honcho/config.json`. [구성 참조](https://github.com/NousResearch/hermes-agent/blob/main/plugins/memory/honcho/README.md) 및 [Honcho 통합 가이드](https://docs.honcho.dev/v3/guides/integrations/hermes)를 참조하세요.

<details>
<summary>전체 구성 참조</summary>

| 키 | 기본값 | 설명 |
|-----|---------|-------------|
| `apiKey` | -- | [app.honcho.dev](https://app.honcho.dev)의 API 키 |
| `baseUrl` | -- | 자체 호스팅 Honcho를 위한 기본 URL |
| `peerName` | -- | 사용자 피어 ID |
| `aiPeer` | host key | AI 피어 ID (프로필당 하나) |
| `workspace` | host key | 공유 작업 공간 ID |
| `contextTokens` | `null` (제한 없음) | 턴당 자동 주입되는 컨텍스트에 대한 토큰 예산. 단어 경계에서 자릅니다 |
| `contextCadence` | `1` | `context()` API 호출 간의 최소 턴 수 (기본 계층 새로 고침) |
| `dialecticCadence` | `2` | `peer.chat()` LLM 호출 간의 최소 턴 수. 권장값 1–5. `hybrid`/`context` 모드에만 적용됩니다 |
| `dialecticDepth` | `1` | 변증법 호출당 `.chat()` 패스 수. 1-3으로 제한. 패스 0: 콜드/웜 프롬프트, 패스 1: 자체 감사, 패스 2: 조정 |
| `dialecticDepthLevels` | `null` | 각 패스별 추론 수준의 선택적 배열, 예: `["minimal", "low", "medium"]`. 비례 기본값을 재정의합니다 |
| `dialecticReasoningLevel` | `'low'` | 기본 추론 수준: `minimal`, `low`, `medium`, `high`, `max` |
| `dialecticDynamic` | `true` | `true`인 경우, 모델이 도구 매개변수를 통해 호출별 추론 수준을 재정의할 수 있습니다 |
| `dialecticMaxChars` | `600` | 시스템 프롬프트에 주입되는 변증법 결과의 최대 글자 수 |
| `recallMode` | `'hybrid'` | `hybrid` (자동 주입 + 도구), `context` (주입만), `tools` (도구만) |
| `writeFrequency` | `'async'` | 메시지 플러시 시기: `async` (백그라운드 스레드), `turn` (동기화), `session` (종료 시 일괄 처리), 또는 정수 N |
| `saveMessages` | `true` | Honcho API에 메시지를 유지할지 여부 |
| `observationMode` | `'directional'` | `directional` (모두 켜짐) 또는 `unified` (공유 풀). `observation` 객체로 재정의합니다 |
| `messageMaxChars` | `25000` | 메시지당 최대 글자 수 (초과 시 청크 처리됨) |
| `dialecticMaxInputChars` | `10000` | `peer.chat()`에 입력되는 변증법 쿼리의 최대 글자 수 |
| `sessionStrategy` | `'per-directory'` | `per-directory`, `per-repo`, `per-session`, `global` |

</details>

<details>
<summary>최소 honcho.json (클라우드)</summary>

```json
{
  "apiKey": "your-key-from-app.honcho.dev",
  "hosts": {
    "hermes": {
      "enabled": true,
      "aiPeer": "hermes",
      "peerName": "your-name",
      "workspace": "hermes"
    }
  }
}
```

</details>

<details>
<summary>최소 honcho.json (자체 호스팅)</summary>

```json
{
  "baseUrl": "http://localhost:8000",
  "hosts": {
    "hermes": {
      "enabled": true,
      "aiPeer": "hermes",
      "peerName": "your-name",
      "workspace": "hermes"
    }
  }
}
```

</details>

:::tip `hermes honcho`에서 마이그레이션
이전에 `hermes honcho setup`을 사용했다면 구성과 모든 서버측 데이터가 그대로 유지됩니다. 설정 마법사를 통해 다시 활성화하거나 수동으로 `memory.provider: honcho`를 설정하여 새 시스템을 통해 다시 활성화하세요.
:::

**다중 피어 설정:**

Honcho는 대화를 피어들이 메시지를 교환하는 것으로 모델링합니다 — 사용자 피어 하나와 Hermes 프로필당 AI 피어 하나, 모두가 하나의 작업 공간을 공유합니다. 작업 공간은 공유된 환경입니다: 사용자 피어는 프로필 전체에 걸쳐 전역적이며, 각 AI 피어는 고유한 정체성(identity)입니다. 모든 AI 피어는 자신의 관찰로부터 독립적인 표현(representation) / 카드를 구축하므로, 동일한 사용자에 대해 `coder` 프로필은 코드 지향적인 상태를 유지하고 `writer` 프로필은 편집적인(editorial) 상태를 유지합니다.

매핑:

| 개념 | 설명 |
|---------|-----------|
| **작업 공간 (Workspace)** | 공유 환경. 하나의 작업 공간 아래의 모든 Hermes 프로필은 동일한 사용자 정체성을 봅니다. |
| **사용자 피어 (User peer)** (`peerName`) | 사람(human)입니다. 작업 공간 내의 프로필 간에 공유됩니다. |
| **AI 피어 (AI peer)** (`aiPeer`) | Hermes 프로필당 하나. 호스트 키 `hermes` → 기본값; 그 외의 경우 `hermes.<profile>`. |
| **관찰 (Observation)** | Honcho가 누구의 메시지로부터 무엇을 모델링할지 제어하는 피어별 토글. `directional` (기본값, 모두 켜짐) 또는 `unified` (단일 관찰자 풀). |

### 새 프로필, 완전히 새로운 Honcho 피어

```bash
hermes profile create coder --clone
```

`--clone`은 `honcho.json`에 `aiPeer: "coder"`, 공유된 `workspace`, 상속된 `peerName`, `recallMode`, `writeFrequency`, `observation` 등을 가진 `hermes.coder` 호스트 블록을 생성합니다. AI 피어는 Honcho에 조기에 생성되어 첫 메시지 이전에 이미 존재하게 됩니다.

### 기존 프로필, Honcho 피어 소급 생성 (backfill)

```bash
hermes honcho sync
```

모든 Hermes 프로필을 스캔하여 호스트 블록이 없는 프로필에 대해 생성하고, 기본 `hermes` 블록에서 설정을 상속하며, 새로운 AI 피어를 조기에 생성합니다. 멱등적(Idempotent)입니다 — 이미 호스트 블록이 있는 프로필은 건너뜁니다.

### 프로필별 관찰

각 호스트 블록은 관찰 구성을 독립적으로 재정의할 수 있습니다. 예: AI 피어가 사용자를 관찰하지만 자기 모델링(self-model)은 하지 않는 코드 중심 프로필:

```json
"hermes.coder": {
  "aiPeer": "coder",
  "observation": {
    "user": { "observeMe": true, "observeOthers": true },
    "ai":   { "observeMe": false, "observeOthers": true }
  }
}
```

**관찰 토글 (피어당 한 세트):**

| 토글 | 효과 |
|--------|--------|
| `observeMe` | Honcho는 해당 피어의 자체 메시지로부터 그 피어의 표현(representation)을 구축합니다 |
| `observeOthers` | 이 피어는 다른 피어의 메시지를 관찰합니다 (피어 간 변증법에 공급) |

`observationMode`를 통한 사전 설정(Presets):

- **`"directional"`** (기본값) — 4개의 플래그 모두 켜짐. 완전한 상호 관찰; 피어 간 변증법을 활성화합니다.
- **`"unified"`** — 사용자 `observeMe: true`, AI `observeOthers: true`, 나머지는 거짓. 단일 관찰자 풀; AI는 사용자를 모델링하지만 자기 자신은 모델링하지 않으며, 사용자 피어는 자기 자신만 모델링합니다.

[Honcho 대시보드](https://app.honcho.dev)를 통해 설정된 서버 측 토글은 로컬 기본값보다 우선하며 세션 시작 시 다시 동기화됩니다.

전체 관찰 참조는 [Honcho 페이지](./honcho.md#observation-directional-vs-unified)를 확인하세요.

<details>
<summary>전체 honcho.json 예시 (다중 프로필)</summary>

```json
{
  "apiKey": "your-key",
  "workspace": "hermes",
  "peerName": "eri",
  "hosts": {
    "hermes": {
      "enabled": true,
      "aiPeer": "hermes",
      "workspace": "hermes",
      "peerName": "eri",
      "recallMode": "hybrid",
      "writeFrequency": "async",
      "sessionStrategy": "per-directory",
      "observation": {
        "user": { "observeMe": true, "observeOthers": true },
        "ai": { "observeMe": true, "observeOthers": true }
      },
      "dialecticReasoningLevel": "low",
      "dialecticDynamic": true,
      "dialecticCadence": 2,
      "dialecticDepth": 1,
      "dialecticMaxChars": 600,
      "contextCadence": 1,
      "messageMaxChars": 25000,
      "saveMessages": true
    },
    "hermes.coder": {
      "enabled": true,
      "aiPeer": "coder",
      "workspace": "hermes",
      "peerName": "eri",
      "recallMode": "tools",
      "observation": {
        "user": { "observeMe": true, "observeOthers": false },
        "ai": { "observeMe": true, "observeOthers": true }
      }
    },
    "hermes.writer": {
      "enabled": true,
      "aiPeer": "writer",
      "workspace": "hermes",
      "peerName": "eri"
    }
  },
  "sessions": {
    "/home/user/myproject": "myproject-main"
  }
}
```

</details>

[구성 참조](https://github.com/NousResearch/hermes-agent/blob/main/plugins/memory/honcho/README.md) 및 [Honcho 통합 가이드](https://docs.honcho.dev/v3/guides/integrations/hermes)를 참조하세요.


---

### OpenViking

파일 시스템 스타일의 지식 계층, 계층형 검색 및 6개 범주로의 자동 메모리 추출 기능을 갖춘 Volcengine(ByteDance)의 컨텍스트 데이터베이스.

| | |
|---|---|
| **가장 적합한 대상** | 구조화된 브라우징을 갖춘 자체 호스팅 지식 관리 |
| **요구 사항** | `pip install openviking` + 실행 중인 서버 |
| **데이터 저장소** | 자체 호스팅 (로컬 또는 클라우드) |
| **비용** | 무료 (오픈 소스, AGPL-3.0) |

**도구:** `viking_search` (시맨틱 검색), `viking_read` (계층형: 초록/개요/전체), `viking_browse` (파일 시스템 탐색), `viking_remember` (사실 저장), `viking_add_resource` (URL/문서 수집)

**설정:**
```bash
# 먼저 OpenViking 서버를 시작합니다
pip install openviking
openviking-server

# 그런 다음 Hermes를 구성합니다
hermes memory setup    # "openviking" 선택
# 또는 수동으로:
hermes config set memory.provider openviking
echo "OPENVIKING_ENDPOINT=http://localhost:1933" >> ~/.hermes/.env
```

**주요 기능:**
- 계층형 컨텍스트 로딩: L0 (~100 토큰) → L1 (~2k) → L2 (전체)
- 세션 커밋 시 자동 메모리 추출 (프로필, 기본 설정, 엔터티, 이벤트, 사례, 패턴)
- 계층적 지식 브라우징을 위한 `viking://` URI 체계

---

### Mem0

시맨틱 검색, 재순위 매기기(reranking) 및 자동 중복 제거 기능이 있는 서버 측 LLM 사실 추출.

| | |
|---|---|
| **가장 적합한 대상** | 메모리 관리에 손을 놓고 싶을 때 — Mem0이 자동으로 추출을 처리합니다. |
| **요구 사항** | `pip install mem0ai` + API 키 |
| **데이터 저장소** | Mem0 클라우드 |
| **비용** | Mem0 요금제 |

**도구:** `mem0_profile` (저장된 모든 메모리), `mem0_search` (시맨틱 검색 + 재순위 매기기), `mem0_conclude` (그대로 사실 저장)

**설정:**
```bash
hermes memory setup    # "mem0" 선택
# 또는 수동으로:
hermes config set memory.provider mem0
echo "MEM0_API_KEY=your-key" >> ~/.hermes/.env
```

**구성:** `$HERMES_HOME/mem0.json`

| 키 | 기본값 | 설명 |
|-----|---------|-------------|
| `user_id` | `hermes-user` | 사용자 식별자 |
| `agent_id` | `hermes` | 에이전트 식별자 |

---

### Hindsight

지식 그래프, 엔터티 분해(entity resolution) 및 다중 전략 검색을 갖춘 장기 메모리. `hindsight_reflect` 도구는 다른 제공자에서는 제공하지 않는 교차 메모리 합성을 제공합니다. 대화 턴 전체(도구 호출 포함)를 세션 수준 문서 추적과 함께 자동으로 유지합니다.

| | |
|---|---|
| **가장 적합한 대상** | 엔터티 관계가 있는 지식 그래프 기반의 회상(recall) |
| **요구 사항** | 클라우드: [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io)의 API 키. 로컬: LLM API 키 (OpenAI, Groq, OpenRouter 등) |
| **데이터 저장소** | Hindsight 클라우드 또는 로컬 내장 PostgreSQL |
| **비용** | Hindsight 요금제(클라우드) 또는 무료(로컬) |

**도구:** `hindsight_retain` (엔터티 추출과 함께 저장), `hindsight_recall` (다중 전략 검색), `hindsight_reflect` (교차 메모리 합성)

**설정:**
```bash
hermes memory setup    # "hindsight" 선택
# 또는 수동으로:
hermes config set memory.provider hindsight
echo "HINDSIGHT_API_KEY=your-key" >> ~/.hermes/.env
```

설정 마법사는 종속성을 자동으로 설치하며 선택한 모드(`cloud`의 경우 `hindsight-client`, `local`의 경우 `hindsight-all`)에 필요한 것만 설치합니다. `hindsight-client >= 0.4.22`가 필요합니다(오래된 경우 세션 시작 시 자동 업그레이드됨).

**로컬 모드 UI:** `hindsight-embed -p hermes ui start`

**구성:** `$HERMES_HOME/hindsight/config.json`

| 키 | 기본값 | 설명 |
|-----|---------|-------------|
| `mode` | `cloud` | `cloud` 또는 `local` |
| `bank_id` | `hermes` | 메모리 뱅크 식별자 |
| `recall_budget` | `mid` | 회상의 철저함: `low` / `mid` / `high` |
| `memory_mode` | `hybrid` | `hybrid` (컨텍스트 + 도구), `context` (자동 주입 전용), `tools` (도구 전용) |
| `auto_retain` | `true` | 대화 턴을 자동으로 유지(retain)합니다 |
| `auto_recall` | `true` | 각 턴 전에 메모리를 자동으로 회상합니다 |
| `retain_async` | `true` | 서버에서 유지(retain) 작업을 비동기적으로 처리합니다 |
| `retain_context` | `conversation between Hermes Agent and the User` | 유지된 메모리의 컨텍스트 레이블 |
| `retain_tags` | — | 유지된 메모리에 적용되는 기본 태그; 호출별 도구 태그와 병합됨 |
| `retain_source` | — | 유지된 메모리에 첨부되는 선택적 `metadata.source` |
| `retain_user_prefix` | `User` | 자동 유지되는 트랜스크립트에서 사용자 턴 앞에 사용되는 레이블 |
| `retain_assistant_prefix` | `Assistant` | 자동 유지되는 트랜스크립트에서 보조(assistant) 턴 앞에 사용되는 레이블 |
| `recall_tags` | — | 회상(recall) 시 필터링할 태그 |

전체 구성 참조는 [플러그인 README](https://github.com/NousResearch/hermes-agent/blob/main/plugins/memory/hindsight/README.md)를 참조하세요.

---

### Holographic

FTS5 전체 텍스트 검색, 신뢰도 점수 및 조합형 대수 쿼리를 위한 HRR (Holographic Reduced Representations)을 갖춘 로컬 SQLite 팩트 저장소입니다.

| | |
|---|---|
| **가장 적합한 대상** | 고급 검색 기능이 있는 로컬 전용 메모리, 외부 종속성 없음 |
| **요구 사항** | 없음 (SQLite는 항상 사용 가능). HRR 대수를 위해서는 NumPy(선택 사항). |
| **데이터 저장소** | 로컬 SQLite |
| **비용** | 무료 |

**도구:** `fact_store` (9가지 작업: 추가, 검색, 조사(probe), 관련, 추론, 모순, 업데이트, 제거, 나열), `fact_feedback` (신뢰도 점수를 학습시키는 유용함/유용하지 않음 평가)

**설정:**
```bash
hermes memory setup    # "holographic" 선택
# 또는 수동으로:
hermes config set memory.provider holographic
```

**구성:** `plugins.hermes-memory-store` 아래의 `config.yaml`

| 키 | 기본값 | 설명 |
|-----|---------|-------------|
| `db_path` | `$HERMES_HOME/memory_store.db` | SQLite 데이터베이스 경로 |
| `auto_extract` | `false` | 세션 종료 시 팩트를 자동으로 추출합니다 |
| `default_trust` | `0.5` | 기본 신뢰도 점수 (0.0–1.0) |

**고유 기능:**
- `probe` — 특정 엔터티의 대수적 회상 (사람/사물에 관한 모든 사실)
- `reason` — 여러 엔터티에 걸친 구성형 AND 쿼리
- `contradict` — 상충되는 사실에 대한 자동 감지
- 비대칭적 피드백(+0.05 유용함 / -0.10 유용하지 않음)을 사용한 신뢰도 점수

---

### RetainDB

하이브리드 검색(Vector + BM25 + Reranking), 7가지 메모리 유형 및 델타 압축 기능을 갖춘 클라우드 메모리 API입니다.

| | |
|---|---|
| **가장 적합한 대상** | 이미 RetainDB의 인프라를 사용 중인 팀 |
| **요구 사항** | RetainDB 계정 + API 키 |
| **데이터 저장소** | RetainDB 클라우드 |
| **비용** | 월 $20 |

**도구:** `retaindb_profile` (사용자 프로필), `retaindb_search` (시맨틱 검색), `retaindb_context` (작업 관련 컨텍스트), `retaindb_remember` (유형 + 중요도와 함께 저장), `retaindb_forget` (메모리 삭제)

**설정:**
```bash
hermes memory setup    # "retaindb" 선택
# 또는 수동으로:
hermes config set memory.provider retaindb
echo "RETAINDB_API_KEY=your-key" >> ~/.hermes/.env
```

---

### ByteRover

`brv` CLI를 통한 영구 메모리 — 계층적 검색(퍼지 텍스트 → LLM 기반 검색) 기능을 갖춘 계층적 지식 트리. 선택적인 클라우드 동기화를 갖춘 로컬 우선.

| | |
|---|---|
| **가장 적합한 대상** | CLI를 사용한 휴대용 로컬 우선 메모리를 원하는 개발자 |
| **요구 사항** | ByteRover CLI (`npm install -g byterover-cli` 또는 [설치 스크립트](https://byterover.dev)) |
| **데이터 저장소** | 로컬 (기본값) 또는 ByteRover 클라우드 (선택적 동기화) |
| **비용** | 무료 (로컬) 또는 ByteRover 요금제 (클라우드) |

**도구:** `brv_query` (지식 트리 검색), `brv_curate` (사실/결정/패턴 저장), `brv_status` (CLI 버전 + 트리 통계)

**설정:**
```bash
# 먼저 CLI를 설치합니다
curl -fsSL https://byterover.dev/install.sh | sh

# 그런 다음 Hermes를 구성합니다
hermes memory setup    # "byterover" 선택
# 또는 수동으로:
hermes config set memory.provider byterover
```

**주요 기능:**
- 자동 압축 전(pre-compression) 추출 (컨텍스트 압축으로 인해 폐기되기 전에 통찰력을 저장)
- `$HERMES_HOME/byterover/` (프로필 단위 범위)에 저장되는 지식 트리
- SOC2 Type II 인증 클라우드 동기화 (선택 사항)

---

### Supermemory

프로필 회상(recall), 시맨틱 검색, 명시적 메모리 도구 및 Supermemory 그래프 API를 통한 세션 종료 대화 수집(ingest) 기능을 갖춘 시맨틱 장기 메모리입니다.

| | |
|---|---|
| **가장 적합한 대상** | 사용자 프로파일링 및 세션 수준 그래프 구축 기능을 갖춘 시맨틱 회상 |
| **요구 사항** | `pip install supermemory` + [API 키](https://supermemory.ai) |
| **데이터 저장소** | Supermemory 클라우드 |
| **비용** | Supermemory 요금제 |

**도구:** `supermemory_store` (명시적 메모리 저장), `supermemory_search` (의미 유사도 검색), `supermemory_forget` (ID 또는 일치하는 쿼리로 잊어버리기), `supermemory_profile` (영구 프로필 + 최근 컨텍스트)

**설정:**
```bash
hermes memory setup    # "supermemory" 선택
# 또는 수동으로:
hermes config set memory.provider supermemory
echo 'SUPERMEMORY_API_KEY=***' >> ~/.hermes/.env
```

**구성:** `$HERMES_HOME/supermemory.json`

| 키 | 기본값 | 설명 |
|-----|---------|-------------|
| `container_tag` | `hermes` | 검색 및 쓰기에 사용되는 컨테이너 태그. 프로필 범위 태그에 대해 `{identity}` 템플릿을 지원합니다. |
| `auto_recall` | `true` | 턴이 시작되기 전에 관련 메모리 컨텍스트를 주입합니다 |
| `auto_capture` | `true` | 각 응답 후에 정리된 사용자-어시스턴트 턴을 저장합니다 |
| `max_recall_results` | `10` | 컨텍스트로 포맷할 회상된 항목의 최대 수 |
| `profile_frequency` | `50` | 첫 턴 및 N 턴마다 프로필 사실을 포함합니다 |
| `capture_mode` | `all` | 기본적으로 아주 작거나 사소한 턴을 건너뜁니다 |
| `search_mode` | `hybrid` | 검색 모드: `hybrid`, `memories` 또는 `documents` |
| `api_timeout` | `5.0` | SDK 및 수집(ingest) 요청에 대한 시간 제한 |

**환경 변수:** `SUPERMEMORY_API_KEY` (필수), `SUPERMEMORY_CONTAINER_TAG` (구성을 재정의함).

**주요 기능:**
- 자동 컨텍스트 펜싱(fencing) — 재귀적인 메모리 오염을 방지하기 위해 캡처된 턴에서 회상된 메모리를 제거합니다.
- 전체 세션 수집 — 전체 대화는 세션 경계에서 한 번 전송됩니다.
- Supermemory에서 더 풍부한 프로필 + 그래프 구축을 위한 세션 종료 대화 수집(`/v4/conversations`로 전송).
- 첫 번째 턴 및 구성 가능한 간격으로 주입되는 프로필 사실.
- **프로필 범위 컨테이너** — Hermes 프로필별로 메모리를 격리하려면 `container_tag`에 `{identity}`를 사용하세요(예: `hermes-{identity}` → `hermes-coder`).
- **다중 컨테이너 모드** — 에이전트가 이름이 지정된 컨테이너 간에 읽고/쓸 수 있도록 `custom_containers` 목록과 함께 `enable_custom_container_tags`를 활성화합니다. 자동 작업은 주 컨테이너에 머뭅니다.

<details>
<summary>다중 컨테이너 예시</summary>

```json
{
  "container_tag": "hermes",
  "enable_custom_container_tags": true,
  "custom_containers": ["project-alpha", "shared-knowledge"],
  "custom_container_instructions": "Use project-alpha for coding context."
}
```

</details>

**지원:** [Discord](https://supermemory.link/discord) · [support@supermemory.com](mailto:support@supermemory.com)

### Memori

백그라운드 완료 턴 캡처, 도구 인식(tool-aware) 턴 컨텍스트, 그리고 사실, 요약, 할당량, 가입 및 피드백에 대한 명시적인 회상 도구를 갖춘 Memori Cloud를 사용하는 구조화된 장기 메모리.

| | |
|---|---|
| **가장 적합한 대상** | 구조화된 프로젝트 및 세션 속성 지정 기능을 갖춘 에이전트 제어 회상 |
| **요구 사항** | `pip install hermes-memori` + `hermes-memori install` + [Memori API 키](https://app.memorilabs.ai/signup) |
| **데이터 저장소** | Memori 클라우드 |
| **비용** | Memori 요금제 |

**도구:** `memori_recall` (장기 메모리 검색), `memori_recall_summary` (요약된 컨텍스트), `memori_quota` (사용량/할당량), `memori_signup` (가입 이메일 요청), `memori_feedback` (통합 피드백 보내기)

**설정:**
```bash
pip install hermes-memori
hermes-memori install
hermes config set memory.provider memori
hermes memory setup
```

---

## 공급자 비교 (Provider Comparison)

| 공급자 | 저장소 | 비용 | 도구 수 | 종속성 | 고유 기능 |
|----------|---------|------|-------|-------------|----------------|
| **Honcho** | 클라우드 | 유료 | 5 | `honcho-ai` | 변증법적 사용자 모델링 + 세션 범위 컨텍스트 |
| **OpenViking** | 자체 호스팅 | 무료 | 5 | `openviking` + 서버 | 파일 시스템 계층 + 계층형 로딩 |
| **Mem0** | 클라우드 | 유료 | 3 | `mem0ai` | 서버 측 LLM 추출 |
| **Hindsight** | 클라우드/로컬 | 무료/유료 | 3 | `hindsight-client` | 지식 그래프 + reflect 합성 |
| **Holographic** | 로컬 | 무료 | 2 | 없음 | HRR 대수 + 신뢰도 점수 |
| **RetainDB** | 클라우드 | 월 $20 | 5 | `requests` | 델타 압축 |
| **ByteRover** | 로컬/클라우드 | 무료/유료 | 3 | `brv` CLI | 압축 전 추출 |
| **Supermemory** | 클라우드 | 유료 | 4 | `supermemory` | 컨텍스트 펜싱 + 세션 그래프 수집 + 다중 컨테이너 |
| **Memori** | 클라우드 | 무료/유료 | 5 | `hermes-memori` | 도구 인식 메모리 + 구조화된 회상 |

## 프로필 격리 (Profile Isolation)

각 공급자의 데이터는 [프로필(profile)](/user-guide/profiles) 단위로 분리됩니다.

- **로컬 저장소 공급자** (Holographic, ByteRover)는 프로필별로 다른 `$HERMES_HOME/` 경로를 사용합니다.
- **구성 파일 공급자** (Honcho, Mem0, Hindsight, Supermemory)는 구성을 `$HERMES_HOME/`에 저장하므로 각 프로필은 고유한 자격 증명을 갖습니다.
- **클라우드 공급자** (RetainDB)는 프로필 범위에 해당하는 프로젝트 이름을 자동 파생(auto-derive)합니다.
- **환경 변수 공급자** (OpenViking)는 각 프로필의 `.env` 파일을 통해 구성됩니다.

## 메모리 공급자 만들기 (Building a Memory Provider)

나만의 공급자를 만드는 방법은 [개발자 가이드: 메모리 공급자 플러그인(Developer Guide: Memory Provider Plugins)](/developer-guide/memory-provider-plugin)을 확인하세요.
