---
title: 자격 증명 풀 (Credential Pools)
description: 제공자당 여러 API 키 또는 OAuth 토큰을 풀(pool)로 구성하여 자동 순환 및 속도 제한(rate limit) 복구를 수행합니다.
sidebar_label: 자격 증명 풀 (Credential Pools)
sidebar_position: 9
---

# 자격 증명 풀 (Credential Pools)

자격 증명 풀을 사용하면 동일한 제공자에 대해 여러 API 키 또는 OAuth 토큰을 등록할 수 있습니다. 한 키가 속도 제한(rate limit)이나 청구 할당량(billing quota)에 도달하면, Hermes는 자동으로 다음 정상 키로 순환(rotate)하여 제공자를 전환하지 않고도 세션을 유지합니다.

이것은 완전히 *다른* 제공자로 전환하는 [대체 제공자 (Fallback Providers)](./fallback-providers.md)와는 다릅니다. 자격 증명 풀은 동일한 제공자 내에서의 순환이고, 대체 제공자는 제공자 간 장애 조치(failover)입니다. 풀이 먼저 시도되며, 모든 풀 키가 소진되면 *그때* 대체 제공자가 활성화됩니다.

:::tip
자격 증명 풀은 주로 API 키 제공자(OpenRouter, Anthropic)를 위한 것입니다. 단일 [Nous Portal](/integrations/nous-portal) OAuth는 300개 이상의 모델을 포괄하므로, Portal을 사용하는 대부분의 사용자는 풀이 필요하지 않습니다.
:::

## 작동 방식 (How It Works)

```
요청(Your request)
  → 풀에서 키 선택 (round_robin / least_used / fill_first / random)
  → 제공자에게 전송
  → 429 속도 제한(rate limit)?
      → 요금제/사용량 한도 도달 (예: ChatGPT/Codex "usage limit reached")?
          → 즉시 다음 풀 키로 순환 (재시도 안 함 — 재시도해도 한도가 풀리지 않음)
      → 일반적인 / 일시적인 429?
          → 같은 키로 한 번 재시도 (일시적인 오류)
          → 두 번째 429 → 다음 풀 키로 순환
      → 모든 키 소진 → fallback_model (다른 제공자)
  → 402 청구 오류(billing error)?
      → 즉시 다음 풀 키로 순환 (24시간 대기 시간)
  → 401 인증 만료(auth expired)?
      → 먼저 토큰 갱신 시도 (OAuth)
      → 갱신 실패 → 다음 풀 키로 순환
  → 성공 → 정상적으로 계속 진행
```

## 빠른 시작 (Quick Start)

이미 `.env`에 API 키가 설정되어 있다면 Hermes는 이를 1개의 키 풀로 자동 감지합니다. 풀링의 이점을 얻으려면 키를 추가하세요:

```bash
# 두 번째 OpenRouter 키 추가
hermes auth add openrouter --api-key sk-or-v1-your-second-key

# 두 번째 Anthropic 키 추가
hermes auth add anthropic --type api-key --api-key sk-ant-api03-your-second-key

# Anthropic OAuth 자격 증명 추가 (Claude Max 요금제 + 추가 사용량 크레딧 필요)
hermes auth add anthropic --type oauth
# OAuth 로그인을 위해 브라우저가 열립니다
```

풀을 확인하세요:

```bash
hermes auth list
```

출력:
```
openrouter (2 credentials):
  #1  OPENROUTER_API_KEY   api_key env:OPENROUTER_API_KEY ←
  #2  backup-key           api_key manual

anthropic (3 credentials):
  #1  hermes_pkce          oauth   hermes_pkce ←
  #2  claude_code          oauth   claude_code
  #3  ANTHROPIC_API_KEY    api_key env:ANTHROPIC_API_KEY
```

`←` 표시는 현재 선택된 자격 증명을 나타냅니다.

## 대화형 관리 (Interactive Management)

대화형 마법사를 실행하려면 하위 명령어 없이 `hermes auth`를 실행하세요:

```bash
hermes auth
```

전체 풀 상태를 보여주고 메뉴를 제공합니다:

```
What would you like to do?
  1. Add a credential (자격 증명 추가)
  2. Remove a credential (자격 증명 제거)
  3. Reset cooldowns for a provider (제공자의 대기 시간 초기화)
  4. Set rotation strategy for a provider (제공자의 순환 전략 설정)
  5. Exit (종료)
```

API 키와 OAuth를 모두 지원하는 제공자(Anthropic, Nous, Codex)의 경우 추가 흐름에서 유형을 묻습니다:

```
anthropic supports both API keys and OAuth login.
  1. API key (paste a key from the provider dashboard)
  2. OAuth login (authenticate via browser)
Type [1/2]:
```

## CLI 명령어 (CLI Commands)

| 명령어 (Command) | 설명 (Description) |
|---------|-------------|
| `hermes auth` | 대화형 풀 관리 마법사 |
| `hermes auth list` | 모든 풀 및 자격 증명 표시 |
| `hermes auth list <provider>` | 특정 제공자의 풀 표시 |
| `hermes auth add <provider>` | 자격 증명 추가 (유형 및 키 프롬프트 표시) |
| `hermes auth add <provider> --type api-key --api-key <key>` | 비대화형으로 API 키 추가 |
| `hermes auth add <provider> --type oauth` | 브라우저 로그인을 통해 OAuth 자격 증명 추가 |
| `hermes auth remove <provider> <index>` | 1부터 시작하는 인덱스로 자격 증명 제거 |
| `hermes auth reset <provider>` | 모든 대기 시간/소진 상태 초기화 |

## 순환 전략 (Rotation Strategies)

`hermes auth` → "Set rotation strategy"를 통하거나 `config.yaml`에서 구성합니다:

```yaml
credential_pool_strategies:
  openrouter: round_robin
  anthropic: least_used
```

| 전략 (Strategy) | 동작 (Behavior) |
|----------|----------|
| `fill_first` (기본값) | 첫 번째 정상 키를 소진될 때까지 사용한 후 다음 키로 이동 |
| `round_robin` | 키를 균등하게 순환하며, 각 선택 후 다음 키로 이동 |
| `least_used` | 항상 요청 횟수가 가장 적은 키를 선택 |
| `random` | 정상 키 중 무작위 선택 |

## 오류 복구 (Error Recovery)

풀은 오류에 따라 다르게 처리합니다:

| 오류 (Error) | 동작 (Behavior) | 대기 시간 (Cooldown) |
|-------|----------|----------|
| **429 Rate Limit (속도 제한)** | 같은 키로 한 번 재시도 (일시적). 연속으로 두 번 429 발생 시 다음 키로 순환 | 1시간 |
| **402 Billing/Quota (청구/할당량)** | 즉시 다음 키로 순환 | 24시간 |
| **401 Auth Expired (인증 만료)** | 먼저 OAuth 토큰 갱신 시도. 갱신에 실패한 경우에만 순환 | — |
| **All keys exhausted (모든 키 소진)** | 구성된 경우 `fallback_model`로 대체됨 | — |

`has_retried_429` 플래그는 성공적인 API 호출마다 재설정되므로 단일 일시적 429는 순환을 트리거하지 않습니다.

## 사용자 지정 엔드포인트 풀 (Custom Endpoint Pools)

사용자 지정 OpenAI 호환 엔드포인트(Together.ai, RunPod, 로컬 서버)는 config.yaml의 `custom_providers`에 있는 엔드포인트 이름을 키로 사용하는 자체 풀을 가집니다.

`hermes model`을 통해 사용자 지정 엔드포인트를 설정하면 "Together.ai" 또는 "Local (localhost:8080)"과 같은 이름이 자동 생성됩니다. 이 이름이 풀 키가 됩니다.

```bash
# hermes model을 통해 사용자 지정 엔드포인트를 설정한 후:
hermes auth list
# 표시됨:
#   Together.ai (1 credential):
#     #1  config key    api_key config:Together.ai ←

# 동일한 엔드포인트에 대한 두 번째 키 추가:
hermes auth add Together.ai --api-key sk-together-second-key
```

사용자 지정 엔드포인트 풀은 `auth.json`의 `credential_pool` 아래에 `custom:` 접두사와 함께 저장됩니다:

```json
{
  "credential_pool": {
    "openrouter": [...],
    "custom:together.ai": [...]
  }
}
```

## 자동 감지 (Auto-Discovery)

Hermes는 여러 소스에서 자격 증명을 자동으로 감지하고 시작 시 풀에 추가합니다:

| 소스 (Source) | 예시 (Example) | 자동 추가됨? (Auto-seeded?) |
|--------|---------|-------------|
| 환경 변수 | `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY` | 예 |
| OAuth 토큰 (auth.json) | Codex 기기 코드, Nous 기기 코드 | 예 |
| Claude Code 자격 증명 | `~/.claude/.credentials.json` | 예 (Anthropic) |
| Hermes PKCE OAuth | `~/.hermes/auth.json` | 예 (Anthropic) |
| 사용자 지정 엔드포인트 설정 | config.yaml의 `model.api_key` | 예 (사용자 지정 엔드포인트) |
| 수동 입력 | `hermes auth add`를 통해 추가됨 | auth.json에 지속됨 |

자동 추가된 항목은 풀 로드마다 업데이트됩니다 — 환경 변수를 제거하면 해당 풀 항목이 자동으로 정리됩니다. 수동 항목(`hermes auth add`를 통해 추가됨)은 절대 자동으로 정리되지 않습니다.

차용한 런타임 보안 비밀(예: 환경 변수, Bitwarden/Vault/keyring/systemd 참조 및 사용자 지정 구성 값)은 `auth.json` 경계에서 참조 전용(reference-only)입니다. Hermes는 현재 실행에 대해 메모리에서 확인된(resolved) 값을 사용할 수 있지만, 소스 참조, 라벨, 상태, 요청 카운터 및 비가역 지문과 같은 메타데이터만 지속합니다. 수동 항목 및 Hermes가 소유한 OAuth/기기 코드 상태는 갱신하는 데 필요한 지속적인 토큰을 유지합니다.

## 위임 및 서브에이전트 공유 (Delegation & Subagent Sharing)

에이전트가 `delegate_task`를 통해 서브에이전트를 생성할 때, 부모의 자격 증명 풀은 자동으로 자식과 공유됩니다:

- **동일한 제공자** — 자식은 부모의 전체 풀을 수신하여 속도 제한에 따른 키 순환을 가능하게 합니다.
- **다른 제공자** — 자식은 해당 제공자의 고유 풀을 로드합니다 (구성된 경우).
- **구성된 풀 없음** — 자식은 상속된 단일 API 키로 대체됩니다.

즉, 서브에이전트는 추가 구성 없이 부모와 동일한 속도 제한 복원력을 얻습니다. 작업별 자격 증명 임대(leasing)를 통해 키를 동시에 순환할 때 자식 에이전트 간에 충돌이 발생하지 않도록 합니다.

## 스레드 안전성 (Thread Safety)

자격 증명 풀은 모든 상태 변형(`select()`, `mark_exhausted_and_rotate()`, `try_refresh_current()`, `mark_used()`)에 스레딩 잠금(threading lock)을 사용합니다. 이를 통해 게이트웨이가 여러 채팅 세션을 동시에 처리할 때 안전한 동시 액세스를 보장합니다.

## 아키텍처 (Architecture)

전체 데이터 흐름 다이어그램은 리포지토리의 [`docs/credential-pool-flow.excalidraw`](https://excalidraw.com/#json=2Ycqhqpi6f12E_3ITyiwh,c7u9jSt5BwrmiVzHGbm87g)를 참조하세요.

자격 증명 풀은 제공자 확인(resolution) 계층에 통합됩니다:

1. **`agent/credential_pool.py`** — 풀 관리자: 저장, 선택, 순환, 대기 시간
2. **`hermes_cli/auth_commands.py`** — CLI 명령어 및 대화형 마법사
3. **`hermes_cli/runtime_provider.py`** — 풀 인식 자격 증명 확인
4. **`run_agent.py`** — 오류 복구: 429/402/401 → 풀 순환 → 대체(fallback)

## 저장소 (Storage)

풀 상태는 `~/.hermes/auth.json`의 `credential_pool` 키 아래에 저장됩니다:

```json
{
  "version": 1,
  "credential_pool": {
    "openrouter": [
      {
        "id": "abc123",
        "label": "OPENROUTER_API_KEY",
        "auth_type": "api_key",
        "priority": 0,
        "source": "env:OPENROUTER_API_KEY",
        "secret_source": "bitwarden",
        "secret_fingerprint": "sha256:12ab34cd56ef7890",
        "last_status": "ok",
        "request_count": 142
      }
    ],
    "anthropic": [
      {
        "id": "manual1",
        "label": "personal-api-key",
        "auth_type": "api_key",
        "priority": 0,
        "source": "manual",
        "access_token": "sk-ant-api03-..."
      }
    ]
  }
}
```

위의 OpenRouter 항목은 외부 소스에서 차용했으므로 원시 키는 `auth.json`에 저장되지 않습니다. 수동 Anthropic 항목은 의도적으로 Hermes의 자격 증명 저장소에 추가되었으므로 토큰은 영구적으로 보관될 수 있습니다.

전략은 `config.yaml`에 저장됩니다 (`auth.json` 아님):

```yaml
credential_pool_strategies:
  openrouter: round_robin
  anthropic: least_used
```
