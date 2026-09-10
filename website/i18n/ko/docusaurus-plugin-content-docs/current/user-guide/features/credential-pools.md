---
title: 자격 증명 풀
description: 자동 순환 및 속도 제한 복구를 위해 제공자별 API 키 또는 OAuth 토큰을 여러 개 풀링합니다.
sidebar_label: 자격 증명 풀
sidebar_position: 9
---

# 자격 증명 풀

자격 증명 풀을 사용하면 동일한 제공자에 여러 API 키 또는 OAuth 토큰을 등록할 수 있습니다. 한 키가 속도 제한 또는 결제 할당량에 도달하면 Hermes가 자동으로 다음 정상 키로 순환하므로, 제공자를 전환하지 않고도 세션을 계속 유지할 수 있습니다.

이는 완전히 *다른* 제공자로 전환하는 [대체 제공자](./fallback-providers.md)와 다릅니다. 자격 증명 풀은 동일한 제공자 내에서 순환하고, 대체 제공자는 제공자 간 장애 조치를 수행합니다. 풀을 먼저 시도하며, 풀의 모든 키를 사용할 수 없게 된 *후에야* 대체 제공자가 활성화됩니다.

:::warning 키를 순환하면 프롬프트 캐시가 초기화됩니다
제공자 측 프롬프트 캐시(Anthropic, OpenAI, OpenRouter)는 요청을 보낸 계정/API 키를 기준으로 적용됩니다. 세션 중 풀이 다른 키로 순환하면 새 키에는 대화의 캐시된 접두사가 없으므로 다음 요청에서 할인되지 않은 입력 가격으로 전체 기록을 다시 읽습니다. 이후 다시 이전 키로 돌아가더라도 해당 키의 캐시 TTL이 아직 살아 있지 않다면 또다시 전체 기록을 읽습니다. 순환은 세션을 계속 실행하기 위한 기능이며 이것이 목적이지만, 긴 대화에서는 순환할 때마다 컨텍스트를 정가로 한 번 처리하는 비용이 발생합니다.
:::

:::tip
자격 증명 풀은 주로 API 키 제공자(OpenRouter, Anthropic)를 위한 기능입니다. 단일 [Nous Portal](/integrations/nous-portal) OAuth로 300개 이상의 모델을 사용할 수 있으므로, Portal을 사용하는 대부분의 사용자는 풀을 설정할 필요가 없습니다.
:::

## 작동 방식

```
Your request
  → Pick key from pool (round_robin / least_used / fill_first / random)
  → Send to provider
  → 429 rate limit?
      → Plan/usage limit reached (e.g. ChatGPT/Codex "usage limit reached")?
          → Rotate to next pool key immediately (no retry — the cap won't clear on retry)
      → Generic / transient 429?
          → Retry same key once (transient blip)
          → Second 429 → rotate to next pool key
      → All keys exhausted → fallback_model (different provider)
  → 402 billing error?
      → Immediately rotate to next pool key (1h cooldown)
  → 401 auth expired?
      → Try refreshing the token (OAuth)
      → Refresh failed → rotate to next pool key
  → Success → continue normally
```

## 빠른 시작

`.env`에 이미 API 키가 설정되어 있다면 Hermes가 이를 1키 풀로 자동 검색합니다. 풀링의 이점을 얻으려면 키를 더 추가하세요.

```bash
# Add a second OpenRouter key
hermes auth add openrouter --api-key sk-or-v1-your-second-key

# Add a second Anthropic key
hermes auth add anthropic --type api-key --api-key sk-ant-api03-your-second-key

# Add an Anthropic OAuth credential (requires Claude Max plan + extra usage credits)
hermes auth add anthropic --type oauth
# Opens browser for OAuth login
```

풀을 확인합니다.

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

`←`는 현재 선택된 자격 증명을 나타냅니다.

## 대화형 관리

하위 명령 없이 `hermes auth`를 실행하면 대화형 마법사가 열립니다.

```bash
hermes auth
```

전체 풀 상태를 표시하고 메뉴를 제공합니다.

```
What would you like to do?
  1. Add a credential
  2. Remove a credential
  3. Reset cooldowns for a provider
  4. Set rotation strategy for a provider
  5. Exit
```

API 키와 OAuth를 모두 지원하는 제공자(Anthropic, Nous, Codex)의 경우 추가 과정에서 유형을 묻습니다.

```
anthropic supports both API keys and OAuth login.
  1. API key (paste a key from the provider dashboard)
  2. OAuth login (authenticate via browser)
Type [1/2]:
```

## CLI 명령

| 명령 | 설명 |
|---------|-------------|
| `hermes auth` | 대화형 풀 관리 마법사 |
| `hermes auth list` | 모든 풀과 자격 증명 표시 |
| `hermes auth list <provider>` | 특정 제공자의 풀 표시 |
| `hermes auth add <provider>` | 자격 증명 추가(유형과 키 입력) |
| `hermes auth add <provider> --type api-key --api-key <key>` | 대화형 입력 없이 API 키 추가 |
| `hermes auth add <provider> --type oauth` | 브라우저 로그인으로 OAuth 자격 증명 추가 |
| `hermes auth remove <provider> <index>` | 1부터 시작하는 인덱스로 자격 증명 제거 |
| `hermes auth reset <provider>` | 모든 쿨다운/소진 상태 초기화 |

## 순환 전략

`hermes auth` → "Set rotation strategy"를 통해 설정하거나 `config.yaml`에서 설정합니다.

```yaml
credential_pool_strategies:
  openrouter: round_robin
  anthropic: least_used
```

| 전략 | 동작 |
|----------|----------|
| `fill_first` (기본값) | 첫 번째 정상 키가 소진될 때까지 사용한 다음 다음 키로 이동 |
| `round_robin` | 키를 균등하게 순환하며 선택할 때마다 다음 키로 전환 |
| `least_used` | 요청 횟수가 가장 적은 키를 항상 선택 |
| `random` | 정상 키 중 무작위로 선택 |

## 오류 복구

풀은 오류 유형에 따라 다르게 처리합니다.

| 오류 | 동작 | 쿨다운 |
|----------|----------|----------|
| **429 속도 제한** | 동일한 키로 한 번 재시도(일시적 오류). 연속 두 번째 429가 발생하면 다음 키로 순환 | 1시간 |
| **402 결제/할당량** | 즉시 다음 키로 순환 | 1시간 |
| **401 인증 만료** | 먼저 OAuth 토큰 새로 고침을 시도. 새로 고침에 실패할 때만 순환 | 5분 |
| **모든 키 소진** | 설정되어 있으면 `fallback_model`로 전달 | — |

제공자가 제공하는 `reset_at` 타임스탬프는 이러한 기본 쿨다운보다 우선합니다.

`has_retried_429` 플래그는 API 호출이 성공할 때마다 초기화되므로, 한 번의 일시적인 429만으로는 순환이 발생하지 않습니다.

## 사용자 지정 엔드포인트 풀

사용자 지정 OpenAI 호환 엔드포인트(Together.ai, RunPod, 로컬 서버)는 `config.yaml`의 `providers:` 딕셔너리에 있는 엔드포인트 이름을 키로 사용하여 자체 풀을 갖습니다(기존 `custom_providers` 목록도 자동으로 마이그레이션됩니다).

`hermes model`을 통해 사용자 지정 엔드포인트를 설정하면 "Together.ai" 또는 "Local (localhost:8080)" 같은 이름이 자동으로 생성됩니다. 이 이름이 풀 키가 됩니다.

```bash
# After setting up a custom endpoint via hermes model:
hermes auth list
# Shows:
#   Together.ai (1 credential):
#     #1  config key    api_key config:Together.ai ←

# Add a second key for the same endpoint:
hermes auth add Together.ai --api-key sk-together-second-key
```

사용자 지정 엔드포인트 풀은 `auth.json`의 `credential_pool` 아래에 `custom:` 접두사를 사용하여 저장됩니다.

```json
{
  "credential_pool": {
    "openrouter": [...],
    "custom:together.ai": [...]
  }
}
```

## 자동 검색

Hermes는 여러 소스에서 자격 증명을 자동으로 검색하고 시작 시 풀에 추가합니다.

| 소스 | 예시 | 자동 추가? |
|-------------|---------|-------------|
| 환경 변수 | `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY` | 예 |
| OAuth 토큰(auth.json) | Codex device code, Nous device code | 예 |
| Claude Code 자격 증명 | `~/.claude/.credentials.json` | 예(Anthropic) |
| Hermes PKCE OAuth | `~/.hermes/auth.json` | 예(Anthropic) |
| 사용자 지정 엔드포인트 설정 | `config.yaml`의 `model.api_key` | 예(사용자 지정 엔드포인트) |
| 수동 항목 | `hermes auth add`를 통해 추가 | auth.json에 유지 |

자동으로 추가된 항목은 각 풀을 불러올 때 업데이트됩니다. 환경 변수를 제거하면 해당 풀 항목도 자동으로 정리됩니다. 수동 항목(`hermes auth add`로 추가)은 자동으로 정리되지 않습니다.

대여한 런타임 비밀(예: 환경 변수, Bitwarden/Vault/keyring/systemd 참조, 사용자 지정 설정 값)은 `auth.json` 경계에서 참조 전용입니다. Hermes는 현재 실행 중 메모리에서 해석된 값을 사용할 수 있지만, 소스 참조, 레이블, 상태, 요청 카운터, 되돌릴 수 없는 지문 같은 메타데이터만 저장합니다. 수동 항목과 Hermes가 소유한 OAuth/device-code 상태는 새로 고침에 필요한 영구 토큰을 저장합니다.

## 위임 및 서브에이전트 공유

에이전트가 `delegate_task`를 통해 서브에이전트를 생성하면 부모의 자격 증명 풀이 자식에게 자동으로 공유됩니다.

- **동일한 제공자** — 자식이 부모의 전체 풀을 받아 속도 제한 발생 시 키를 순환할 수 있습니다.
- **다른 제공자** — 자식이 해당 제공자의 자체 풀을 불러옵니다(설정된 경우).
- **풀이 설정되지 않음** — 자식이 상속된 단일 API 키로 대체됩니다.

따라서 추가 설정 없이도 서브에이전트가 부모와 동일한 속도 제한 복원력을 얻습니다. 작업별 자격 증명 임대는 자식들이 동시에 키를 순환할 때 서로 충돌하지 않도록 합니다.

## 스레드 안전성

자격 증명 풀은 모든 상태 변경(`select()`, `mark_exhausted_and_rotate()`, `try_refresh_current()`, `mark_used()`)에 스레딩 잠금을 사용합니다. 따라서 게이트웨이가 여러 채팅 세션을 동시에 처리할 때도 안전하게 접근할 수 있습니다.

## 아키텍처

전체 데이터 흐름 다이어그램은 저장소의 [`docs/credential-pool-flow.excalidraw`](https://excalidraw.com/#json=2Ycqhqpi6f12E_3ITyiwh,c7u9jSt5BwrmiVzHGbm87g)를 참조하세요.

자격 증명 풀은 제공자 확인 계층에 통합됩니다.

1. **`agent/credential_pool.py`** — 풀 관리자: 저장, 선택, 순환, 쿨다운
2. **`hermes_cli/auth_commands.py`** — CLI 명령과 대화형 마법사
3. **`hermes_cli/runtime_provider.py`** — 풀을 인식하는 자격 증명 확인
4. **`run_agent.py`** — 오류 복구: 429/402/401 → 풀 순환 → 대체

## 저장소

풀 상태는 `credential_pool` 키 아래 `~/.hermes/auth.json`에 저장됩니다.

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

위의 OpenRouter 항목은 외부 소스에서 대여한 것이므로 원시 키가 `auth.json`에 저장되지 않습니다. 수동 Anthropic 항목은 Hermes의 자격 증명 저장소에 의도적으로 추가되었으므로 토큰을 저장할 수 있습니다.

전략은 `config.yaml`에 저장됩니다(`auth.json`에는 저장되지 않음).

```yaml
credential_pool_strategies:
  openrouter: round_robin
  anthropic: least_used
```
