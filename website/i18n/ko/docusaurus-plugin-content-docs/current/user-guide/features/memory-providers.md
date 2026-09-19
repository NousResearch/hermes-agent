---
sidebar_position: 4
title: "메모리 제공자"
description: "외부 메모리 제공자 플러그인 — Honcho, OpenViking, Mem0, Hindsight, Holographic, RetainDB, ByteRover, Supermemory"
---

# 메모리 제공자

Hermes Agent에는 기본 제공되는 MEMORY.md 및 USER.md를 넘어 에이전트에 세션 간 지속적인 지식을 제공하는 8개의 외부 메모리 제공자 플러그인이 포함되어 있습니다. 외부 제공자는 한 번에 **하나만** 활성화할 수 있으며, 기본 메모리는 외부 제공자와 항상 함께 활성화됩니다.

## 빠른 시작

```bash
hermes memory setup      # interactive picker + configuration
hermes memory status     # check what's active
hermes memory off        # disable external provider
```

`hermes plugins` → Provider Plugins → Memory Provider를 통해 활성 메모리 제공자를 선택할 수도 있습니다.

또는 `~/.hermes/config.yaml`에서 직접 설정할 수 있습니다.

```yaml
memory:
  provider: openviking   # or honcho, mem0, hindsight, holographic, retaindb, byterover, supermemory
```

## 작동 방식

메모리 제공자가 활성화되면 Hermes는 다음 작업을 자동으로 수행합니다.

1. **제공자 컨텍스트를 시스템 프롬프트에 주입** (제공자가 알고 있는 내용)
2. **매 턴 전에 관련 메모리를 미리 가져옴** (백그라운드에서 논블로킹으로 수행)
3. **각 응답 후 대화 턴을 제공자와 동기화**
4. **세션 종료 시 메모리를 추출** (지원하는 제공자의 경우)
5. **기본 메모리 기록을 외부 제공자에 미러링**
6. **제공자별 도구를 추가**하여 에이전트가 메모리를 검색·저장·관리할 수 있도록 함

기본 메모리(MEMORY.md / USER.md)는 이전과 정확히 동일하게 계속 작동합니다. 외부 제공자는 여기에 기능을 추가합니다.

## 사용 가능한 제공자

### Honcho

변증법적 추론, 세션 범위 컨텍스트 주입, 시맨틱 검색, 지속적인 결론을 제공하는 AI 네이티브 세션 간 사용자 모델링입니다. 기본 컨텍스트에는 사용자 표현 및 피어 카드와 함께 세션 요약도 포함되어, 에이전트가 지금까지 논의된 내용을 파악할 수 있습니다.

| | |
|---|---|
| **적합한 용도** | 세션 간 컨텍스트와 사용자-에이전트 정렬이 필요한 멀티 에이전트 시스템 |
| **필요 사항** | `pip install honcho-ai` + [API 키](https://app.honcho.dev) 또는 자체 호스팅 인스턴스 |
| **데이터 저장** | Honcho Cloud 또는 자체 호스팅 |
| **비용** | Honcho 요금제(클라우드) / 무료(자체 호스팅) |

**도구(5개):** `honcho_profile` (피어 카드 읽기/업데이트), `honcho_search` (시맨틱 검색), `honcho_context` (세션 컨텍스트 — 요약, 표현, 카드, 메시지), `honcho_reasoning` (LLM 합성), `honcho_conclude` (결론 생성/삭제)

**아키텍처:** 2계층 컨텍스트 주입 — 기본 계층(세션 요약 + 표현 + 피어 카드, `contextCadence`에 따라 새로 고침)과 변증법적 보충 계층(LLM 추론, `dialecticCadence`에 따라 새로 고침)으로 구성됩니다. 기본 컨텍스트의 존재 여부에 따라 콜드 스타트 프롬프트(일반적인 사용자 사실)와 웜 프롬프트(세션 범위 컨텍스트)를 자동으로 선택합니다.

**비용과 깊이를 독립적으로 제어하는 세 가지 구성 옵션:**

- `contextCadence` — 기본 계층 새로 고침 빈도(API 호출 빈도)
- `dialecticCadence` — 변증법적 LLM 실행 빈도(LLM 호출 빈도)
- `dialecticDepth` — 변증법 호출당 `.chat()` 패스 수(1–3, 추론 깊이)

자동 주입되는 변증법적 계층은 쿼리 길이에 따라 추론 수준도 조정합니다(긴 쿼리일수록 깊게 추론하며 `reasoningLevelCap`에서 제한됨). [쿼리 적응형 추론 수준](./honcho.md#query-adaptive-reasoning-level)을 참고하세요.

**설정 마법사:**
```bash
hermes memory setup        # select "honcho" — runs the Honcho-specific post-setup
```

기존 `hermes honcho setup` 명령도 계속 작동하지만(이제 `hermes memory setup`으로 리디렉션됨), Honcho를 활성 메모리 제공자로 선택한 후에만 등록됩니다.

**헤드리스 / 원격 머신:** 브라우저가 없는 환경(SSH, 원격 VM)에서 클라우드 인증을 수행하려면 마법사의 인증 방식 프롬프트에서 **device**를 선택하세요. CLI에 짧은 코드와 확인 링크가 표시됩니다. 다른 머신의 브라우저에서 링크를 열어 승인하면 설정이 완료되므로 API 키를 복사해 붙여 넣을 필요가 없습니다. 사용 가능한 로컬 브라우저가 없다고 감지되면 마법사가 자동으로 이 옵션을 기본 선택합니다.

**구성:** `$HERMES_HOME/honcho.json`(프로필별) 또는 `~/.honcho/config.json`(전역). 검색 순서: `$HERMES_HOME/honcho.json` > `~/.hermes/honcho.json` > `~/.honcho/config.json`. [구성 참조](https://github.com/NousResearch/hermes-agent/blob/main/plugins/memory/honcho/README.md)와 [Honcho 통합 가이드](https://docs.honcho.dev/v3/guides/integrations/hermes)를 참고하세요.

<details>
<summary>전체 구성 참조</summary>

| 키 | 기본값 | 설명 |
|-----|---------|-------------|
| `apiKey` | -- | [app.honcho.dev](https://app.honcho.dev)의 API 키 |
| `baseUrl` | -- | 자체 호스팅 Honcho의 기본 URL |
| `peerName` | -- | 사용자 피어 ID |
| `aiPeer` | 호스트 키 | 프로필당 하나인 AI 피어 ID |
| `workspace` | 호스트 키 | 공유 워크스페이스 ID |
| `contextTokens` | `null` (제한 없음) | 턴마다 자동 주입되는 컨텍스트의 토큰 예산. 단어 경계에서 잘림 |
| `contextCadence` | `1` | `context()` API 호출 사이의 최소 턴 수(기본 계층 새로 고침) |
| `dialecticCadence` | `2` | `peer.chat()` LLM 호출 사이의 최소 턴 수. 1–5 권장. `hybrid`/`context` 모드에만 적용 |
| `dialecticDepth` | `1` | 변증법 호출당 `.chat()` 패스 수. 1–3으로 제한됨. 패스 0: 콜드/웜 프롬프트, 패스 1: 자체 감사, 패스 2: 조정 |
| `dialecticDepthLevels` | `null` | 패스별 추론 수준의 선택적 배열(예: [`"minimal"`, `"low"`, `"medium"`]). 비례 기본값을 재정의 |
| `dialecticReasoningLevel` | `'low'` | 기본 추론 수준: `minimal`, `low`, `medium`, `high`, `max` |
| `dialecticDynamic` | `true` | `true`이면 도구 매개변수로 호출별 추론 수준을 재정의할 수 있음 |
| `dialecticMaxChars` | `600` | 시스템 프롬프트에 주입되는 변증법 결과의 최대 문자 수 |
| `recallMode` | `'hybrid'` | `hybrid`(자동 주입 + 도구), `context`(주입만), `tools`(도구만) |
| `writeFrequency` | `'async'` | 메시지 플러시 시점: `async`(백그라운드 스레드), `turn`(동기), `session`(종료 시 일괄 처리) 또는 정수 N |
| `saveMessages` | `true` | 메시지를 Honcho API에 저장할지 여부 |
| `observationMode` | `'directional'` | `directional`(모두 켜짐) 또는 `unified`(공유 풀). `observation` 객체로 재정의 |
| `messageMaxChars` | `25000` | 메시지당 최대 문자 수(초과 시 청크로 분할) |
| `dialecticMaxInputChars` | `10000` | `peer.chat()`의 변증법 쿼리 입력 최대 문자 수 |
| `sessionStrategy` | `'per-directory'` | `per-directory`, `per-repo`, `per-session`, `global` |
| `pinUserPeer` | `false` | 게이트웨이 전용. `true`이면 모든 비에이전트 게이트웨이 사용자를 `peerName`으로 통합하며, 고정이 모든 별칭보다 우선 |
| `userPeerAliases` | `{}` | 런타임 ID를 피어에 매핑(`{"7654321": "alice"}`). 다대일 매핑 |
| `runtimePeerPrefix` | `""` | 일치하는 별칭이 없을 때 알 수 없는 런타임 ID에 네임스페이스를 부여(`telegram_7654321`) |

</details>

<details>
<summary>최소 honcho.json(클라우드)</summary>

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
<summary>최소 honcho.json(자체 호스팅)</summary>

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
이전에 `hermes honcho setup`을 사용했다면 구성과 서버 측 데이터는 모두 유지됩니다. 설정 마법사를 다시 실행해 재활성화하거나 `memory.provider: honcho`를 직접 설정하여 새 시스템을 통해 다시 활성화하세요.
:::

**다중 피어 설정:**

Honcho는 대화를 메시지를 주고받는 피어로 모델링합니다. 하나의 사용자 피어와 Hermes 프로필마다 하나의 AI 피어가 하나의 워크스페이스를 공유합니다. 워크스페이스는 공유 환경입니다. 사용자 피어는 모든 프로필에서 전역으로 공유되고, 각 AI 피어는 고유한 ID를 가집니다. 모든 AI 피어는 자체 관찰을 바탕으로 독립적인 표현/카드를 구축하므로, 같은 사용자에 대해 `coder` 프로필은 코드 중심으로, `writer` 프로필은 편집 중심으로 유지됩니다.

매핑은 다음과 같습니다.

| 개념 | 설명 |
|---------|-----------|
| **워크스페이스** | 공유 환경. 하나의 워크스페이스에 속한 모든 Hermes 프로필이 동일한 사용자 ID를 봅니다. |
| **사용자 피어** (`peerName`) | 사람 사용자. 워크스페이스의 모든 프로필에서 공유됩니다. |
| **AI 피어** (`aiPeer`) | Hermes 프로필마다 하나. 호스트 키 `hermes`는 기본값이며, 그 외에는 `hermes.<profile>`입니다. |
| **관찰** | Honcho가 어느 피어의 메시지에서 무엇을 모델링할지 제어하는 피어별 토글. `directional`(기본값, 네 가지 모두 켜짐) 또는 `unified`(단일 관찰자 풀)입니다. |

### 새 프로필, 새 Honcho 피어

```bash
hermes profile create coder --clone
```

`--clone`은 `honcho.json`에 `aiPeer: "coder"`, 공유 `workspace`, 상속된 `peerName`, `recallMode`, `writeFrequency`, `observation` 등을 포함하는 `hermes.coder` 호스트 블록을 생성합니다. 첫 메시지 전에 존재하도록 Honcho에서 AI 피어를 즉시 생성합니다.

### 기존 프로필, Honcho 피어 보완

```bash
hermes honcho sync
```

모든 Hermes 프로필을 검색하고 호스트 블록이 없는 프로필에 블록을 생성하며, 기본 `hermes` 블록에서 설정을 상속하고 새 AI 피어를 즉시 생성합니다. 멱등적으로 동작하므로 이미 호스트 블록이 있는 프로필은 건너뜁니다.

### 프로필별 관찰

각 호스트 블록은 관찰 구성을 독립적으로 재정의할 수 있습니다. 예를 들어 AI 피어가 사용자를 관찰하지만 스스로를 모델링하지 않는 코드 중심 프로필은 다음과 같습니다.

```json
"hermes.coder": {
  "aiPeer": "coder",
  "observation": {
    "user": { "observeMe": true, "observeOthers": true },
    "ai":   { "observeMe": false, "observeOthers": true }
  }
}
```

**관찰 토글(피어당 하나의 세트):**

| 토글 | 효과 |
|--------|--------|
| `observeMe` | 자체 메시지를 바탕으로 이 피어의 표현을 구축 |
| `observeOthers` | 다른 피어의 메시지를 관찰(피어 간 추론에 반영) |

`observationMode`를 통한 프리셋:

- **`"directional"`** (기본값) — 네 가지 플래그가 모두 켜집니다. 완전한 상호 관찰로 피어 간 변증법을 활성화합니다.
- **`"unified"`** — 사용자 `observeMe: true`, AI `observeOthers: true`, 나머지는 false입니다. 단일 관찰자 풀로, AI는 사용자를 모델링하지만 자기 자신은 모델링하지 않고 사용자 피어만 스스로 모델링합니다.

[Honcho 대시보드](https://app.honcho.dev)에서 설정한 서버 측 토글은 로컬 기본값보다 우선하며 세션 초기화 시 다시 동기화됩니다.

전체 관찰 참조는 [Honcho 페이지](./honcho.md#observation-directional-vs-unified)를 참고하세요.

### 게이트웨이 ID 매핑

위의 피어 모델은 CLI, TUI 및 데스크톱 세션에 적용되며, 모든 대화는 `peerName`으로 확인됩니다. [게이트웨이](../../developer-guide/gateway-internals.md)에는 플랫폼별 런타임 ID(Telegram UID, Discord snowflake, Slack 사용자)가 추가되며, 세 가지 키가 각 ID가 어떤 피어로 확인될지 결정합니다.

| 키 | 효과 |
|-----|--------|
| `pinUserPeer: true` | 모든 비에이전트 게이트웨이 사용자를 `peerName`으로 통합합니다. 먼저 확인되므로 모든 별칭보다 우선합니다 — 사용자 측 ID마다 별도 피어가 필요하지 않을 때만 선택하세요 |
| `userPeerAliases` | 특정 런타임 ID를 피어에 매핑합니다(`{"7654321": "alice"}`). 각자 고유한 피어를 갖는 에이전트를 포함하여 서로 다른 ID를 라우팅하는 기준입니다 |
| `runtimePeerPrefix` | 매핑되지 않은 런타임 ID에 네임스페이스를 부여(`telegram_7654321`)하여 같은 형태의 ID를 사용하는 플랫폼 간 충돌을 방지합니다 |

게이트웨이 외부에서는 이 키들이 아무런 효과가 없습니다. 연결된 게이트웨이 플랫폼이 감지된 경우에만 `hermes memory setup`에서 이 항목을 묻습니다. 확인 단계와 설정 흐름은 [Honcho 페이지](./honcho.md#gateway-identity-mapping)를 참고하세요.

<details>
<summary>전체 honcho.json 예시(다중 프로필)</summary>

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

[구성 참조](https://github.com/NousResearch/hermes-agent/blob/main/plugins/memory/honcho/README.md)와 [Honcho 통합 가이드](https://docs.honcho.dev/v3/guides/integrations/hermes)를 참고하세요.

---

### OpenViking

파일 시스템 형태의 지식 계층, 계층형 검색, 6개 범주로의 자동 메모리 추출을 제공하는 Volcengine(ByteDance)의 컨텍스트 데이터베이스입니다.

| | |
|---|---|
| **적합한 용도** | 구조화된 탐색을 지원하는 자체 호스팅 지식 관리 |
| **필요 사항** | OpenViking 초기화, 검증 및 실행 |
| **데이터 저장** | 자체 호스팅(로컬 또는 클라우드) |
| **비용** | 무료(오픈 소스, AGPL-3.0) |

**도구(6개):** `viking_search` (시맨틱 검색), `viking_read` (계층형: 초록/개요/전체), `viking_browse` (파일 시스템 탐색), `viking_remember` (사실 저장), `viking_forget` (정확한 `viking://` URI로 메모리 파일 삭제), `viking_add_resource` (URL/문서 수집)

**설정:**
```bash
# Prepare OpenViking first
openviking-server init
openviking-server doctor
openviking-server

# Then configure Hermes
hermes memory setup    # select "openviking"
# Or manually:
hermes config set memory.provider openviking
```

`hermes memory setup`은 `~/.openviking/ovcli.conf`에서 연결 값을 재사용하거나 복사할 수 있습니다. 수동 설정은 활성 프로필의 `.env` 파일을 사용합니다. 기본 프로필에서는 `~/.hermes/.env`이고, 이름이 지정된 프로필에서는 `~/.hermes/profiles/<profile>/.env`입니다.

```text
OPENVIKING_ENDPOINT=http://127.0.0.1:1933
# OPENVIKING_API_KEY=...
# OPENVIKING_ACCOUNT=default
# OPENVIKING_USER=default
# OPENVIKING_AGENT=hermes
```

OpenViking 서버 설정은 `ov.conf`(`--config`, `OPENVIKING_CONFIG_FILE` 또는 `~/.openviking/ov.conf`)에 있습니다. 클라이언트 연결 값은 `ovcli.conf`(`OPENVIKING_CLI_CONFIG_FILE` 또는 `~/.openviking/ovcli.conf`)에 있습니다.

**주요 기능:**
- 계층형 컨텍스트 로딩: L0(~100토큰) → L1(~2k) → L2(전체)
- 세션 커밋 시 자동 메모리 추출(프로필, 선호 사항, 엔터티, 이벤트, 사례, 패턴)
- 계층형 지식 탐색을 위한 `viking://` URI 스킴

`OPENVIKING_ACCOUNT` 및 `OPENVIKING_USER`는 로컬/신뢰 모드에 사용됩니다. `OPENVIKING_AGENT`는 피어 범위 메모리를 위한 OpenViking 내 Hermes의 피어 ID입니다.

---

### Mem0

시맨틱 검색, 재순위 지정, 자동 중복 제거를 지원하는 서버 측 LLM 사실 추출입니다. 세 가지 연결 모드를 제공합니다. **Platform**(Mem0 Cloud), **자체 호스팅 대시보드**(Docker로 실행하는 Mem0 서버), **OSS**(자체 LLM 및 벡터 저장소를 사용하는 프로세스 내 Mem0)입니다.

| | |
|---|---|
| **적합한 용도** | 직접 관리할 필요가 없는 메모리 관리 — Mem0가 자동으로 추출을 처리 |
| **필요 사항** | `pip install mem0ai` + API 키(플랫폼), 실행 중인 Mem0 서버(자체 호스팅 대시보드) 또는 LLM + 벡터 저장소(OSS) |
| **데이터 저장** | Mem0 Cloud(플랫폼), 자체 Mem0 서버(자체 호스팅 대시보드) 또는 프로세스 내(OSS) |
| **비용** | Mem0 요금제(플랫폼) / 무료(자체 호스팅 또는 OSS) |

**도구(4개):** `mem0_search` (시맨틱 검색; 플랫폼 모드에서 선택적 재순위 지정, 기본값은 꺼짐), `mem0_add` (사실을 원문 그대로 저장), `mem0_update` (ID로 업데이트), `mem0_delete` (ID로 삭제)

**설정(Platform):**
```bash
hermes memory setup    # select "mem0" → "Platform"
# Or manually:
hermes config set memory.provider mem0
echo "MEM0_API_KEY=your-key" >> ~/.hermes/.env
```

**설정(OSS):**
```bash
hermes memory setup    # select "mem0" → "Open Source (self-hosted)"
# Or via flags:
hermes memory setup mem0 --mode oss --oss-llm openai --oss-llm-key sk-... --oss-vector qdrant
```

파일을 쓰지 않고 미리 보기:
```bash
hermes memory setup mem0 --mode oss --oss-llm-key sk-... --dry-run
```

**설정(자체 호스팅 대시보드):** Docker로 실행하는 Mem0 서버(대시보드의 REST API)에 연결합니다.

```bash
hermes memory setup    # select "mem0" → "Self-hosted server"
# Or via flags:
hermes memory setup mem0 --mode selfhosted --host http://localhost:8888 --api-key your-admin-api-key
```

또는 다음과 같이 수동으로 설정할 수 있습니다. 환경 변수로 설정:

```bash
echo "MEM0_HOST=http://localhost:8888" >> ~/.hermes/.env
echo "MEM0_API_KEY=your-admin-api-key" >> ~/.hermes/.env
```

또는 `mem0.json`에 설정:

```json
{ "host": "http://localhost:8888", "api_key": "your-admin-api-key" }
```

플러그인은 `X-API-Key`로 인증하고 서버의 `/search`/`/memories` 경로를 사용합니다. `api_key`는 선택 사항입니다(`AUTH_DISABLED` 서버에서만 생략). `mode: oss`는 설정하지 마세요 — `host`보다 우선합니다.

**구성:** `$HERMES_HOME/mem0.json`(동작 설정). 비밀 값인 `MEM0_API_KEY`만 `~/.hermes/.env`에 둡니다.

| 키 | 기본값 | 설명 |
|-----|---------|-------------|
| `mode` | `platform` | `platform`(Mem0 Cloud) 또는 `oss`(자체 관리, 프로세스 내) |
| `host` | — | 자체 호스팅 Mem0 서버 URL(Docker 대시보드). `X-API-Key`를 사용해 HTTP로 경로를 요청하며 `mode: oss`와 함께 사용하지 않음 |
| `user_id` | `hermes-user` | 사용자 식별자 |
| `agent_id` | `hermes` | 에이전트 식별자 |
| `rerank` | `false` | 관련성을 위해 검색 결과를 재순위 지정(플랫폼 모드만 해당) |

**OSS 지원 제공자:**

| 구성 요소 | 제공자 |
|-----------|-----------|
| LLM | openai, ollama |
| 임베더 | openai, ollama |
| 벡터 저장소 | qdrant (로컬/서버), pgvector |

**모드 전환:** `hermes memory setup mem0 --mode <platform|selfhosted|oss>`를 다시 실행하거나 `mem0.json`을 직접 편집합니다.

---

### Hindsight

지식 그래프, 엔터티 확인, 다중 전략 검색을 지원하는 장기 메모리입니다. `hindsight_reflect` 도구는 다른 제공자에는 없는 메모리 간 합성을 제공합니다. 도구 호출을 포함한 전체 대화 턴을 세션 수준 문서 추적과 함께 자동으로 보존합니다.

| | |
|---|---|
| **적합한 용도** | 엔터티 관계를 활용한 지식 그래프 기반 회상 |
| **필요 사항** | 클라우드: [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io)의 API 키. 로컬: LLM API 키(OpenAI, Groq, OpenRouter 등) |
| **데이터 저장** | Hindsight Cloud 또는 로컬 임베디드 PostgreSQL |
| **비용** | Hindsight 요금제(클라우드) 또는 무료(로컬) |

**도구:** `hindsight_retain` (엔터티 추출과 함께 저장), `hindsight_recall` (다중 전략 검색), `hindsight_reflect` (메모리 간 합성)

**설정:**
```bash
hermes memory setup    # select "hindsight"
# Or manually:
hermes config set memory.provider hindsight
echo "HINDSIGHT_API_KEY=your-key" >> ~/.hermes/.env
```

설정 마법사는 종속성을 자동으로 설치하며 선택한 모드에 필요한 항목만 설치합니다(클라우드는 `hindsight-client`, 로컬은 `hindsight-all`). `hindsight-client >= 0.4.22`가 필요하며, 오래된 경우 세션 시작 시 자동으로 업그레이드됩니다.

**로컬 모드 UI:** `hindsight-embed -p hermes ui start`

**구성:** `$HERMES_HOME/hindsight/config.json`

| 키 | 기본값 | 설명 |
|---|---|---|
| `mode` | `cloud` | `cloud` 또는 `local` |
| `bank_id` | `hermes` | 메모리 뱅크 식별자 |
| `recall_budget` | `mid` | 회상 수준: `low` / `mid` / `high` |
| `memory_mode` | `hybrid` | `hybrid`(컨텍스트 + 도구), `context`(자동 주입만), `tools`(도구만) |
| `auto_retain` | `true` | 대화 턴을 자동으로 보존 |
| `auto_recall` | `true` | 매 턴 전에 메모리를 자동으로 회상 |
| `retain_async` | `true` | 서버에서 보존 작업을 비동기로 처리 |
| `retain_context` | `conversation between Hermes Agent and the User` | 보존된 메모리를 위한 컨텍스트 레이블 |
| `retain_tags` | — | 보존된 메모리에 적용되는 기본 태그. 호출별 도구 태그와 병합 |
| `retain_source` | — | 보존된 메모리에 첨부되는 선택적 `metadata.source` |
| `retain_user_prefix` | `User` | 자동 보존 대화 기록에서 사용자 턴 앞에 사용하는 레이블 |
| `retain_assistant_prefix` | `Assistant` | 자동 보존 대화 기록에서 에이전트 턴 앞에 사용하는 레이블 |
| `recall_tags` | — | 회상을 필터링할 태그 |

전체 구성 참조는 [플러그인 README](https://github.com/NousResearch/hermes-agent/blob/main/plugins/memory/hindsight/README.md)를 참고하세요.

---

### Holographic

FTS5 전체 텍스트 검색, 신뢰도 점수, 구성적 대수 쿼리를 위한 HRR(Holographic Reduced Representations)을 지원하는 로컬 SQLite 사실 저장소입니다.

| | |
|---|---|
| **적합한 용도** | 외부 종속성 없이 고급 검색을 제공하는 로컬 전용 메모리 |
| **필요 사항** | 없음(SQLite는 항상 사용 가능). HRR 대수에는 NumPy가 선택적으로 필요합니다. |
| **데이터 저장** | 로컬 SQLite |
| **비용** | 무료 |

**도구:** `fact_store`(9개 작업: 추가, 검색, 탐색, 관련 항목, 추론, 모순, 업데이트, 제거, 목록), `fact_feedback`(신뢰도 점수를 학습시키는 유용함/유용하지 않음 평가)

**설정:**
```bash
hermes memory setup    # select "holographic"
# Or manually:
hermes config set memory.provider holographic
```

**구성:** `plugins.hermes-memory-store` 아래의 `config.yaml`

| 키 | 기본값 | 설명 |
|-----|---------|-------------|
| `db_path` | `$HERMES_HOME/memory_store.db` | SQLite 데이터베이스 경로 |
| `auto_extract` | `false` | 세션 종료 시 사실을 자동으로 추출 |
| `default_trust` | `0.5` | 기본 신뢰도 점수(0.0–1.0) |

**고유 기능:**
- `probe` — 엔터티별 대수 회상(사람/사물에 관한 모든 사실)
- `reason` — 여러 엔터티에 걸친 구성적 AND 쿼리
- `contradict` — 상충하는 사실을 자동으로 감지
- 비대칭 피드백을 적용한 신뢰도 점수(+0.05 유용함 / -0.10 유용하지 않음)

---

### RetainDB

하이브리드 검색(Vector + BM25 + 재순위 지정), 7가지 메모리 유형, 델타 압축을 제공하는 클라우드 메모리 API입니다.

| | |
|---|---|
| **적합한 용도** | 이미 RetainDB 인프라를 사용 중인 팀 |
| **필요 사항** | RetainDB 계정 + API 키 |
| **데이터 저장** | RetainDB Cloud |
| **비용** | 월 $20 |

**도구(10개):** `retaindb_profile`(사용자 프로필), `retaindb_search`(시맨틱 검색), `retaindb_context`(작업 관련 컨텍스트), `retaindb_remember`(유형 + 중요도와 함께 저장), `retaindb_forget`(메모리 삭제), 파일 도구: `retaindb_upload_file`, `retaindb_list_files`, `retaindb_read_file`, `retaindb_ingest_file`, `retaindb_delete_file`

**설정:**
```bash
hermes memory setup    # select "retaindb"
# Or manually:
hermes config set memory.provider retaindb
echo "RETAINDB_API_KEY=your-key" >> ~/.hermes/.env
```

---

### ByteRover

`brv` CLI를 통한 지속적인 메모리로, 계층형 검색(퍼지 텍스트 → LLM 기반 검색)을 지원하는 계층형 지식 트리입니다. 로컬 우선으로 동작하며 선택적으로 클라우드 동기화를 사용할 수 있습니다.

| | |
|---|---|
| **적합한 용도** | CLI로 이식 가능한 로컬 우선 메모리를 원하는 개발자 |
| **필요 사항** | ByteRover CLI(`npm install -g byterover-cli` 또는 [설치 스크립트](https://byterover.dev)) |
| **데이터 저장** | 로컬(기본값) 또는 ByteRover Cloud(선택적 동기화) |
| **비용** | 무료(로컬) 또는 ByteRover 요금제(클라우드) |

**도구:** `brv_query`(지식 트리 검색), `brv_curate`(사실/결정/패턴 저장), `brv_status`(CLI 버전 + 트리 통계)

**설정:**
```bash
# Install the CLI first
curl -fsSL https://byterover.dev/install.sh | sh

# Then configure Hermes
hermes memory setup    # select "byterover"
# Or manually:
hermes config set memory.provider byterover
```

**주요 기능:**
- 사전 압축 자동 추출(컨텍스트 압축으로 삭제되기 전에 인사이트를 저장)
- `$HERMES_HOME/byterover/`에 저장되는 지식 트리(프로필 범위)
- SOC2 Type II 인증 클라우드 동기화(선택 사항)

---

### Supermemory

프로필 회상, 시맨틱 검색, 명시적 메모리 도구, Supermemory 그래프 API를 통한 세션 종료 대화 수집을 지원하는 시맨틱 장기 메모리입니다.

| | |
|---|---|
| **적합한 용도** | 사용자 프로파일링과 세션 수준 그래프 구축을 활용한 시맨틱 회상 |
| **필요 사항** | `pip install supermemory` + [클라우드 API 키](http://app.supermemory.ai/integrations?connect=hermes) 또는 [자체 호스팅 서버](https://supermemory.ai/docs/self-hosting/overview) |
| **데이터 저장** | Supermemory Cloud 또는 자체 호스팅 |
| **비용** | Supermemory 요금제(클라우드) / 무료(자체 호스팅) |

**도구:** `supermemory_store`(명시적 메모리 저장), `supermemory_search`(시맨틱 유사도 검색), `supermemory_forget`(ID 또는 최적 일치 쿼리로 잊기), `supermemory_profile`(지속 프로필 + 최근 컨텍스트)

**설정:**
```bash
hermes memory setup    # select "supermemory"
# Or manually:
hermes config set memory.provider supermemory
echo 'SUPERMEMORY_API_KEY=***' >> ~/.hermes/.env
```

자체 호스팅 설정:

```bash
npx supermemory local
```

`hermes memory setup`을 실행하기 전에 `$HERMES_HOME/supermemory.json`에서 `base_url`을 설정합니다.

```json
{
  "base_url": "http://localhost:6767"
}
```

그런 다음 `hermes memory setup`을 실행하고 로컬 서버가 출력하는 API 키를 입력합니다. 먼저 엔드포인트를 구성하면 설정 연결 프로브도 로컬에 유지됩니다.

**구성:** `$HERMES_HOME/supermemory.json`

| 키 | 기본값 | 설명 |
|-----|---------|-------------|
| `base_url` | `https://api.supermemory.ai` | 호스팅 또는 자체 호스팅 Supermemory의 API 엔드포인트. `SUPERMEMORY_BASE_URL`보다 우선 |
| `container_tag` | `hermes` | 검색과 기록에 사용하는 컨테이너 태그. 프로필 범위 태그를 위해 `{identity}` 템플릿 지원 |
| `auto_recall` | `true` | 턴 전에 관련 메모리 컨텍스트를 주입 |
| `auto_capture` | `true` | 각 응답 후 정리된 사용자-에이전트 턴을 저장 |
| `max_recall_results` | `10` | 컨텍스트로 형식화할 최대 회상 항목 수 |
| `profile_frequency` | `50` | 첫 턴과 매 N번째 턴에 프로필 사실을 포함 |
| `capture_mode` | `all` | 기본적으로 아주 짧거나 사소한 턴을 건너뜀 |
| `search_mode` | `hybrid` | 검색 모드: `hybrid`, `memories` 또는 `documents` |
| `api_timeout` | `5.0` | SDK 및 수집 요청의 시간 제한 |

**환경 변수:** `SUPERMEMORY_API_KEY`(필수), `SUPERMEMORY_BASE_URL`(`base_url`이 구성되지 않았을 때의 호환성 대체값), `SUPERMEMORY_CONTAINER_TAG`(구성을 재정의).

기본 URL 우선순위는 `supermemory.json` → `SUPERMEMORY_BASE_URL` → `https://api.supermemory.ai`입니다. SDK 작업, 설정/상태 프로브 및 대화 수집은 모두 확인된 엔드포인트를 사용합니다.

**주요 기능:**
- 자동 컨텍스트 펜싱 — 재귀적인 메모리 오염을 방지하기 위해 수집된 턴에서 회상된 메모리를 제거
- 전체 세션 수집 — 세션 경계에서 전체 대화를 한 번 전송
- 세션 종료 대화 수집(`/v4/conversations`로 전송)으로 Supermemory에서 더 풍부한 프로필 및 그래프 구축
- 엔드투엔드 자체 호스팅 라우팅 — SDK, 프로브 및 대화 수집 요청이 동일하게 구성된 엔드포인트를 사용
- 첫 턴과 구성 가능한 간격마다 프로필 사실을 주입
- **프로필 범위 컨테이너** — `container_tag`에 `{identity}`를 사용(예: `hermes-{identity}` → `hermes-coder`)하여 Hermes 프로필별로 메모리를 격리
- **다중 컨테이너 모드** — `custom_containers` 목록과 함께 `enable_custom_container_tags`를 활성화하여 에이전트가 이름이 지정된 여러 컨테이너에서 읽고 쓸 수 있도록 함. 자동 작업은 기본 컨테이너에 유지

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

Memori Cloud를 사용하는 구조화된 장기 메모리로, 백그라운드에서 완료된 턴 수집, 도구를 인식하는 턴 컨텍스트, 사실·요약·할당량·가입·피드백을 위한 명시적 회상 도구를 제공합니다.

| | |
|---|---|
| **적합한 용도** | 구조화된 프로젝트 및 세션 귀속을 활용한 에이전트 제어 회상 |
| **필요 사항** | `pip install hermes-memori` + `hermes-memori install` + [Memori API 키](https://app.memorilabs.ai/signup) |
| **데이터 저장** | Memori Cloud |
| **비용** | Memori 요금제 |

**도구:** `memori_recall`(장기 메모리 검색), `memori_recall_summary`(요약된 컨텍스트), `memori_quota`(사용량/할당량), `memori_signup`(가입 이메일 요청), `memori_feedback`(통합 피드백 전송)

**설정:**
```bash
pip install hermes-memori
hermes-memori install
hermes config set memory.provider memori
hermes memory setup
```

---

## 제공자 비교

| 제공자 | 저장소 | 비용 | 도구 | 종속성 | 고유 기능 |
|----------|---------|------|-------|-------------|----------------|
| **Honcho** | 클라우드 | 유료 | 5 | `honcho-ai` | 변증법적 사용자 모델링 + 세션 범위 컨텍스트 |
| **OpenViking** | 자체 호스팅 | 무료 | 6 | `openviking` + 서버 | 파일 시스템 계층 + 계층형 로딩 |
| **Mem0** | 클라우드/자체 호스팅 | 무료/유료 | 4 | `mem0ai` | 서버 측 LLM 추출 + 자체 호스팅/OSS 모드 |
| **Hindsight** | 클라우드/로컬 | 무료/유료 | 3 | `hindsight-client` | 지식 그래프 + reflect 합성 |
| **Holographic** | 로컬 | 무료 | 2 | 없음 | HRR 대수 + 신뢰도 점수 |
| **RetainDB** | 클라우드 | 월 $20 | 10 | `requests` | 델타 압축 |
| **ByteRover** | 로컬/클라우드 | 무료/유료 | 3 | `brv` CLI | 사전 압축 추출 |
| **Supermemory** | 클라우드/자체 호스팅 | 무료/유료 | 4 | `supermemory` | 컨텍스트 펜싱 + 세션 그래프 수집 + 다중 컨테이너 |
| **Memori** | 클라우드 | 무료/유료 | 5 | `hermes-memori` | 도구 인식 메모리 + 구조화된 회상 |

## 프로필 격리

각 제공자의 데이터는 [프로필](/user-guide/profiles)별로 격리됩니다.

- **로컬 저장소 제공자**(Holographic, ByteRover)는 프로필마다 다른 `$HERMES_HOME/` 경로를 사용합니다.
- **구성 파일 제공자**(Honcho, Mem0, Hindsight, Supermemory)는 `$HERMES_HOME/`에 구성을 저장하므로 각 프로필에 자체 자격 증명이 있습니다.
- **클라우드 제공자**(RetainDB)는 프로필 범위 프로젝트 이름을 자동으로 파생합니다.
- **환경 변수 제공자**(OpenViking)는 각 프로필의 `.env` 파일로 구성합니다.

## 메모리 제공자 만들기

직접 메모리 제공자를 만드는 방법은 [개발자 가이드: 메모리 제공자 플러그인](/developer-guide/memory-provider-plugin)을 참고하세요.
