---
title: 대체 프로바이더
description: 기본 모델을 사용할 수 없을 때 백업 LLM 프로바이더로 자동 장애 조치를 구성합니다.
sidebar_label: 대체 프로바이더
sidebar_position: 8
---

# 대체 프로바이더

Hermes Agent에는 프로바이더에 문제가 발생해도 세션을 계속 실행하는 세 가지 복원력 계층이 있습니다.

1. **[자격 증명 풀](./credential-pools.md)** — *동일한* 프로바이더의 여러 API 키를 순환합니다(가장 먼저 시도).
2. **기본 모델 대체** — 주 모델이 실패하면 자동으로 *다른* 프로바이더:model로 전환합니다.
3. **보조 작업 대체** — 비전, 압축, 웹 추출 같은 부수 작업에 대해 독립적으로 프로바이더를 확인합니다.

자격 증명 풀은 동일 프로바이더 내 순환(예: 여러 OpenRouter 키)을 처리합니다. 이 페이지에서는 프로바이더 간 대체를 설명합니다. 두 기능 모두 선택 사항이며 서로 독립적으로 작동합니다.

## 기본 모델 대체

주 LLM 프로바이더에서 속도 제한, 서버 과부하, 인증 실패, 연결 끊김 등의 오류가 발생하면 Hermes는 세션 중간에 백업 프로바이더:model 쌍으로 자동 전환하면서 대화를 잃지 않도록 할 수 있습니다.

### 구성

가장 쉬운 방법은 대화형 관리자입니다.

```bash
hermes fallback
```

`hermes fallback`은 `hermes model`의 프로바이더 선택기를 재사용합니다. 프로바이더 목록, 자격 증명 프롬프트, 검증 방식이 동일합니다. `add`, `list`(별칭 `ls`), `remove`(별칭 `rm`), `clear` 하위 명령을 사용해 체인을 관리합니다. 변경 사항은 `config.yaml`의 최상위 `fallback_providers:` 목록에 저장됩니다.

YAML을 직접 편집하려면 `~/.hermes/config.yaml`에 최상위 `fallback_providers` 목록을 추가합니다.

```yaml
fallback_providers:
  - provider: openrouter
    model: anthropic/claude-sonnet-4
```

각 항목에는 `provider`와 `model`이 모두 필요합니다. 둘 중 하나라도 없는 항목은 무시됩니다.

:::note `fallback_model`과 `fallback_providers` 비교
`fallback_providers`(복수형, 목록)는 현재 구성 형식이며 순서대로 시도할 여러 대체 프로바이더를 지원합니다. `fallback_model`(단수형)은 기존의 단일 대체 키입니다. Hermes는 하위 호환성을 위해 여전히 이 키를 적용하지만, `hermes fallback`은 현재의 `fallback_providers` 키를 작성하고 저장할 때 기존 구성을 마이그레이션합니다. 둘 다 설정된 경우 `fallback_providers`가 우선합니다.
:::

### 지원되는 프로바이더

| 프로바이더 | 값 | 요구 사항 |
|----------|-------|-------------|
| AI Gateway | `ai-gateway` | `AI_GATEWAY_API_KEY` |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` |
| Nous Portal | `nous` | `hermes setup --portal`(새 설정) 또는 `hermes auth add nous`(OAuth) |
| OpenAI Codex | `openai-codex` | `hermes model` → **ChatGPT 또는 Codex Subscription** (ChatGPT OAuth) |
| GitHub Copilot | `copilot` | `COPILOT_GITHUB_TOKEN`, `GH_TOKEN`, 또는 `GITHUB_TOKEN` |
| GitHub Copilot ACP | `copilot-acp` | 외부 프로세스(에디터 통합) |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` 또는 Claude Code 자격 증명 |
| z.ai / GLM | `zai` | `GLM_API_KEY` |
| Kimi / Moonshot | `kimi-coding` | `KIMI_API_KEY` |
| MiniMax | `minimax` | `MINIMAX_API_KEY` |
| MiniMax (중국) | `minimax-cn` | `MINIMAX_CN_API_KEY` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` |
| NVIDIA NIM | `nvidia` | `NVIDIA_API_KEY` (선택 사항: `NVIDIA_BASE_URL`) |
| GMI Cloud | `gmi` | `GMI_API_KEY` (선택 사항: `GMI_BASE_URL`) |
| Upstage Solar | `upstage` (별칭 `solar`) | `UPSTAGE_API_KEY` (선택 사항: `UPSTAGE_BASE_URL`) |
| StepFun | `stepfun` | `STEPFUN_API_KEY` (선택 사항: `STEPFUN_BASE_URL`) |
| Ollama Cloud | `ollama-cloud` | `OLLAMA_API_KEY` |
| Google AI Studio | `gemini` | `GOOGLE_API_KEY` (별칭: `GEMINI_API_KEY`) |
| xAI (Grok) | `xai` (별칭 `grok`) | `XAI_API_KEY` (선택 사항: `XAI_BASE_URL`) |
| xAI Grok OAuth (SuperGrok) | `xai-oauth` (별칭 `grok-oauth`) | `hermes model` → xAI Grok OAuth(브라우저 로그인; SuperGrok 구독) |
| AWS Bedrock | `bedrock` | 표준 boto3 인증(`AWS_REGION` + `AWS_PROFILE` 또는 `AWS_ACCESS_KEY_ID`) |
| Qwen Portal (OAuth) | `qwen-oauth` | `hermes model`(Qwen Portal OAuth; 선택 사항: `HERMES_QWEN_BASE_URL`) |
| MiniMax (OAuth) | `minimax-oauth` | `hermes model`(MiniMax 포털 OAuth) |
| OpenCode Zen | `opencode-zen` | `OPENCODE_ZEN_API_KEY` |
| OpenCode Go | `opencode-go` | `OPENCODE_GO_API_KEY` |
| Kilo Code | `kilocode` | `KILOCODE_API_KEY` |
| Xiaomi MiMo | `xiaomi` | `XIAOMI_API_KEY` |
| Arcee AI | `arcee` | `ARCEEAI_API_KEY` |
| GMI Cloud | `gmi` | `GMI_API_KEY` |
| Alibaba / DashScope | `alibaba` | `DASHSCOPE_API_KEY` |
| Alibaba Coding Plan | `alibaba-coding-plan` | `ALIBABA_CODING_PLAN_API_KEY` (`DASHSCOPE_API_KEY`로 대체 가능) |
| Kimi / Moonshot (중국) | `kimi-coding-cn` | `KIMI_CN_API_KEY` |
| StepFun | `stepfun` | `STEPFUN_API_KEY` |
| Tencent TokenHub | `tencent-tokenhub` | `TOKENHUB_API_KEY` |
| Microsoft Foundry | `azure-foundry` | `AZURE_FOUNDRY_API_KEY` + `AZURE_FOUNDRY_BASE_URL` |
| LM Studio (로컬) | `lmstudio` | `LM_API_KEY`(또는 로컬이면 없음) + `LM_BASE_URL` |
| Hugging Face | `huggingface` | `HF_TOKEN` |
| 사용자 지정 엔드포인트 | `custom` | `base_url` + `key_env`(아래 참조) |

### 사용자 지정 엔드포인트 대체

사용자 지정 OpenAI 호환 엔드포인트의 경우 `base_url`과 선택 사항인 `key_env`를 추가합니다.

```yaml
fallback_providers:
  - provider: custom
    model: my-local-model
    base_url: http://localhost:8000/v1
    key_env: MY_LOCAL_KEY            # env var name containing the API key
```

### 대체가 실행되는 경우

기본 모델이 다음과 같은 오류로 실패하면 대체가 자동으로 활성화됩니다.

- **속도 제한**(HTTP 429) — 재시도 횟수를 모두 소진한 후
- **서버 오류**(HTTP 500, 502, 503) — 재시도 횟수를 모두 소진한 후
- **인증 실패**(HTTP 401, 403) — 즉시(재시도할 의미가 없음)
- **찾을 수 없음**(HTTP 404) — 즉시
- **유효하지 않은 응답** — API가 반복적으로 잘못된 응답이나 빈 응답을 반환할 때

대체가 실행되면 Hermes는 다음을 수행합니다.

1. 대체 프로바이더의 자격 증명을 확인합니다.
2. 새 API 클라이언트를 생성합니다.
3. 모델, 프로바이더, 클라이언트를 현재 세션에서 교체합니다.
4. 재시도 카운터를 초기화하고 대화를 계속합니다.

전환은 매끄럽게 이루어집니다. 대화 기록, 도구 호출, 컨텍스트가 보존됩니다. 에이전트는 중단된 바로 그 지점에서 계속 진행하되, 다른 모델을 사용합니다.

:::warning 대체 시 프롬프트 캐시가 초기화됩니다
프롬프트 캐시는 요청을 처리하는 모델(그리고 대부분의 프로바이더에서는 계정)을 기준으로 지정됩니다. 대체가 실행되면 새 프로바이더:model에는 대화의 캐시된 접두사가 없으므로, 다음 요청은 약 75~90% 할인된 캐시 입력 토큰 요금 대신 전체 기록을 정가로 다시 읽습니다. 턴이 끝나 기본 프로바이더가 복원될 때도 마찬가지로, 기본 프로바이더의 캐시 TTL이 만료되지 않은 경우가 아니라면 첫 요청은 전체 기록을 다시 읽습니다. 이는 장애 중에도 계속 실행하기 위해 불가피한 비용이지만, 한 프로바이더에 계속 머무는 세션보다 프로바이더 사이를 오가는 긴 세션의 비용이 눈에 띄게 높아질 수 있는 이유입니다.
:::

:::info 세션 단위가 아닌 턴 단위
대체는 **턴 범위**로 적용됩니다. 새로운 사용자 메시지가 시작될 때마다 기본 모델이 복원됩니다. 기본 모델이 턴 중간에 실패하면 해당 턴에만 대체가 활성화됩니다. 다음 메시지에서는 Hermes가 기본 모델을 다시 시도합니다. 하나의 턴 안에서는 대체가 최대 한 번만 활성화됩니다. 대체 모델도 실패하면 일반 오류 처리가 이어집니다(재시도 후 오류 메시지). 이를 통해 한 턴 안에서 연쇄 장애 조치 루프가 발생하는 것을 막으면서, 매 턴 기본 모델에 새로 시도할 기회를 제공합니다.

턴별 재시도는 **초기화 인식형**입니다. 기본 자격 증명에서 아직 지나지 않은 속도 제한 초기화 시각을 보고하면(Claude Pro/Max의 5시간 블록이나 Codex 주간 제한 같은 구독 윈도우에서는 이를 시간 또는 일 단위로 보고함) Hermes는 실패할 것이 뻔한 재시도를 건너뛰고 초기화 시각이 지날 때까지 대체 모델을 유지합니다. 이렇게 하면 턴마다 의미 없는 프로바이더 전환 두 번(및 프롬프트 캐시 무효화 두 번)을 피할 수 있습니다. 초기화 시각이 지나면 다음 턴에 자동으로 기본 모델로 돌아갑니다. 초기화 시각이 없는 일시적인 429 오류는 기존 동작을 유지합니다. 짧게 쿨다운한 후 매 턴 재시도합니다.
:::

### 예시

**Anthropic 네이티브의 대체로 OpenRouter 사용:**
```yaml
model:
  provider: anthropic
  default: claude-sonnet-4-6

fallback_providers:
  - provider: openrouter
    model: anthropic/claude-sonnet-4
```

**OpenRouter의 대체로 Nous Portal 사용:**
```yaml
model:
  provider: openrouter
  default: anthropic/claude-opus-4

fallback_providers:
  - provider: nous
    model: nous-hermes-3
```

**클라우드의 대체로 로컬 모델 사용:**
```yaml
fallback_providers:
  - provider: custom
    model: llama-3.1-70b
    base_url: http://localhost:8000/v1
    key_env: LOCAL_API_KEY
```

**대체로 Codex OAuth 사용:**
```yaml
fallback_providers:
  - provider: openai-codex
    model: gpt-5.3-codex
```

### 대체가 작동하는 위치

| 컨텍스트 | 대체 지원 |
|---------|-------------------|
| CLI 세션 | ✔ |
| 메시징 게이트웨이(Telegram, Discord 등) | ✔ |
| 서브에이전트 위임 | ✔ (서브에이전트는 부모 대체 체인을 상속) |
| Cron 작업 | ✔ (Cron 에이전트는 구성된 대체 프로바이더를 상속) |
| `provider: auto`의 보조 작업 | ✔ (작업별 대체를 시도한 후 내장 보조 검색 전에 기본 대체 체인을 시도) |

:::tip
기본 대체 체인에는 환경 변수가 없습니다. `config.yaml` 또는 `hermes fallback`을 통해서만 구성합니다. 이는 의도된 동작입니다. 대체 구성은 명시적인 선택이어야 하며, 오래된 셸 내보내기가 이를 덮어써서는 안 됩니다.
:::

---

## 보조 작업 대체

Hermes는 부수 작업에 별도의 경량 모델을 사용합니다. 각 작업에는 내장 대체 시스템으로 작동하는 자체 프로바이더 확인 체인이 있습니다.

### 독립적인 프로바이더 확인이 적용되는 작업

| 작업 | 수행 내용 | 구성 키 |
|------|-------------|-----------|
| 비전 | 이미지 분석, 브라우저 스크린샷 | `auxiliary.vision` |
| 웹 추출 | 웹 페이지 요약 | `auxiliary.web_extract` |
| 압축 | 컨텍스트 압축 요약 | `auxiliary.compression` |
| Skills Hub | 스킬 검색 및 탐색 | `auxiliary.skills_hub` |
| MCP | MCP 도우미 작업 | `auxiliary.mcp` |
| 승인 | 스마트 명령 승인 분류 | `auxiliary.approval` |
| 제목 생성 | 세션 제목 요약 | `auxiliary.title_generation` |
| 트리아지 지정 | `hermes kanban specify` / 대시보드 ✨ 버튼 — 한 줄짜리 트리아지 작업을 실제 사양으로 구체화 | `auxiliary.triage_specifier` |

### 자동 감지 체인

작업의 프로바이더가 `"auto"`(기본값)로 설정되면 Hermes는 먼저 해당 보조 작업에 주 프로바이더 + 주 모델을 시도합니다. 해당 경로를 사용할 수 없거나 나중에 용량 관련 오류로 실패하면, Hermes는 이제 내장 검색 체인을 사용하기 전에 사용자가 구성한 대체 정책을 적용합니다.

```text
Main provider + main model → auxiliary.<task>.fallback_chain →
fallback_providers / fallback_model → built-in auxiliary discovery chain
```

작업별 체인이 있으면 가장 정확한 설정이므로 우선 적용됩니다. 최상위 `fallback_providers` 체인은 주 에이전트가 사용하는 것과 동일한 정책이므로, `auto`의 보조 작업에도 무료 전용 또는 동일 프로바이더 대체 규칙이 적용됩니다.

**내장 텍스트 검색 체인(압축, 웹 추출, 제목 생성 등):**

```text
OpenRouter → Nous Portal → Custom endpoint → Codex OAuth →
API-key providers (z.ai, Kimi, MiniMax, Xiaomi MiMo, Hugging Face, Anthropic) → give up
```

**내장 비전 검색 체인:**

```text
Main provider (if vision-capable) → OpenRouter → Nous Portal →
Codex OAuth → Anthropic → Custom endpoint → give up
```

이러한 내장 체인은 작업별 또는 기본 대체 정책을 선언하지 않은 사용자를 위한 편의용 대체 기능입니다.

### 보조 프로바이더 구성

각 작업은 `config.yaml`에서 독립적으로 구성할 수 있습니다.

```yaml
auxiliary:
  vision:
    provider: "auto"              # auto | openrouter | nous | codex | main | anthropic
    model: ""                     # e.g. "openai/gpt-4o"
    base_url: ""                  # direct endpoint (takes precedence over provider)
    api_key: ""                   # API key for base_url

  web_extract:
    provider: "auto"
    model: ""

  compression:
    provider: "auto"
    model: ""
    fallback_chain:              # optional, task-specific fallback policy
      - provider: openrouter
        model: inclusionai/ring-2.6-1t:free

  skills_hub:
    provider: "auto"
    model: ""

  mcp:
    provider: "auto"
    model: ""
```

위의 모든 작업은 동일한 **provider / model / base_url** 패턴을 따릅니다. 각 작업은 자체 `fallback_chain`도 선언할 수 있습니다. 생략하면 `provider: auto`는 Hermes의 내장 보조 검색 체인보다 먼저 최상위 `fallback_providers` 체인을 사용합니다.

컨텍스트 압축은 `auxiliary.compression`에서 구성합니다.

```yaml
auxiliary:
  compression:
    provider: main                                    # Same provider options as other auxiliary tasks
    model: google/gemini-3-flash-preview
    base_url: null                                    # Custom OpenAI-compatible endpoint
```

기본 대체 체인은 다음을 사용합니다.

```yaml
fallback_providers:
  - provider: openrouter
    model: anthropic/claude-sonnet-4
    # base_url: http://localhost:8000/v1             # Optional custom endpoint
```

세 가지 모두(보조 작업, 압축, 대체)는 동일한 방식으로 작동합니다. `provider`를 설정해 요청을 처리할 주체를 선택하고, `model`을 설정해 사용할 모델을 선택하며, `base_url`을 설정해 사용자 지정 엔드포인트를 지정합니다(프로바이더보다 우선).

### 보조 작업의 프로바이더 옵션

이 옵션은 `auxiliary:`, `compression:`, `fallback_providers:` 항목에만 적용됩니다. `"main"`은 최상위 `model.provider`에 유효한 값이 아닙니다. 사용자 지정 엔드포인트에는 `model:` 섹션에서 `provider: custom`을 사용합니다([AI 프로바이더](/integrations/providers) 참조).

| 프로바이더 | 설명 | 요구 사항 |
|----------|-------------|-------------|
| `"auto"` | 하나가 작동할 때까지 프로바이더를 순서대로 시도(기본값) | 하나 이상의 프로바이더 구성 |
| `"openrouter"` | OpenRouter 강제 사용 | `OPENROUTER_API_KEY` |
| `"nous"` | Nous Portal 강제 사용 | `hermes auth` |
| `"codex"` | Codex OAuth 강제 사용 | `hermes model` → ChatGPT 또는 Codex Subscription |
| `"main"` | 주 에이전트가 사용하는 프로바이더 사용(보조 작업만) | 활성 주 프로바이더 구성 |
| `"anthropic"` | Anthropic 네이티브 강제 사용 | `ANTHROPIC_API_KEY` 또는 Claude Code 자격 증명 |

### 직접 엔드포인트 재정의

보조 작업에서 `base_url`을 설정하면 프로바이더 확인을 완전히 우회하고 해당 엔드포인트로 직접 요청을 보냅니다.

```yaml
auxiliary:
  vision:
    base_url: "http://localhost:1234/v1"
    api_key: "local-key"
    model: "qwen2.5-vl"
```

`base_url`은 `provider`보다 우선합니다. Hermes는 인증에 구성된 `api_key`를 사용하며, 설정되지 않은 경우 `OPENAI_API_KEY`로 대체합니다. 사용자 지정 엔드포인트에 `OPENROUTER_API_KEY`를 재사용하지는 **않습니다**.

---

## 보조 작업 용량 오류 대체

명시적인 보조 프로바이더(예: `auxiliary.vision.provider: glm`)를 설정하면 Hermes는 이를 선호하는 선택으로 취급합니다. 하지만 프로바이더가 **용량 오류**(HTTP 402 결제 필요, HTTP 429 일일 할당량 소진, 연결 실패)로 인해 요청을 실제로 처리할 수 없으면 Hermes는 조용히 실패하는 대신 계층형 체인을 따라 대체합니다.

1. **기본 보조 프로바이더** — 사용자가 구성한 프로바이더(항상 가장 먼저 시도)
2. **`auxiliary.<task>.fallback_chain`** — 작성한 경우 작업별 재정의 목록
3. **주 에이전트 프로바이더 + 모델** — 최후의 안전망(체인을 작성하지 않은 경우에도 항상 시도)
4. **경고 후 재발생** — 모든 계층이 실패하면 Hermes는 `Auxiliary <task>: ... all fallbacks exhausted`를 WARNING 수준으로 기록하고 원래 오류를 다시 발생시킵니다.

일시적인 HTTP 429 속도 제한(`Retry-After: ...`)은 용량 문제가 아닌 요청 제약으로 처리됩니다. 명시적인 프로바이더 선택을 존중하며 대체 계층을 **실행하지 않습니다**. 일일/월간 할당량 소진, 결제 오류, 연결 실패만 명시적 프로바이더 게이트를 우회합니다.

`provider: auto`를 사용하는 사용자(명시적인 보조 프로바이더가 없는 경우)는 2~3단계 대신 기존 자동 감지 체인을 사용합니다. 첫 단계가 이미 주 에이전트 모델이므로 `auto` 사용자는 구성 없이도 동일한 결과를 얻습니다.

### 선택 사항: 작업별 대체 체인

"주 에이전트 모델 우선"과 다른 대체 순서를 원하면 `fallback_chain`을 명시적으로 구성합니다. 각 항목에는 최소한 `provider`가 필요하며 `model`, `base_url`, `api_key`는 선택 사항입니다.

```yaml
auxiliary:
  vision:
    provider: glm
    model: glm-4v-flash
    fallback_chain:
      - provider: openrouter
        model: google/gemini-3-flash-preview
      - provider: nous
        model: anthropic/claude-sonnet-4

  compression:
    provider: openrouter
    fallback_chain:
      - provider: openai
        model: gpt-4o-mini
        timeout: 240            # optional — this candidate's own deadline (seconds)
```

대체를 사용하기 위해 `fallback_chain`을 구성할 **필요는 없습니다**. 주 에이전트 안전망은 어떤 경우에도 실행됩니다. 기본값과 다른 순서를 특별히 원할 때만 사용합니다.

각 `fallback_chain` 항목은 자체 `timeout`(초)도 선언할 수 있습니다. 이 값을 지정하지 않으면 대체 후보는 작업 수준의 타임아웃을 상속합니다. 작업 수준의 타임아웃은 기본 프로바이더에 맞게 조정되어 있을 수 있습니다. 항목별 `timeout`을 선언하면 느리지만 안정적인 대체 모델(예: 대규모 컨텍스트 요약 모델)이 기본 프로바이더의 제한 시간에 걸려 중단되지 않고 실제로 필요한 예산을 사용할 수 있습니다.

### 대체를 실행하는 프로바이더 할당량 오류

Hermes는 다음 오류를 402 크레딧 소진과 동등한 용량 오류로 인식합니다(일시적인 속도 제한은 아님).

- Bedrock / LiteLLM: `Too many tokens per day`, `daily limit`, `tokens per day`
- Vertex AI / GCP: `quota exceeded`, `resource exhausted`, `RESOURCE_EXHAUSTED`
- 일반: `daily quota`, `quota_exceeded`

프로바이더가 일일 할당량 소진에 대해 다른 문구를 반환하고 Hermes가 대체를 실행하지 않는다면 버그입니다. 정확한 오류 문자열과 함께 이슈를 열어 주세요.

---

## 컨텍스트 압축 대체

컨텍스트 압축은 `auxiliary.compression` 구성 블록을 사용해 요약을 처리할 모델과 프로바이더를 제어합니다.

```yaml
auxiliary:
  compression:
    provider: "auto"                              # auto | openrouter | nous | main
    model: "google/gemini-3-flash-preview"
```

:::info 기존 구성 마이그레이션
`compression.summary_model` / `compression.summary_provider` / `compression.summary_base_url`이 포함된 기존 구성은 처음 로드할 때 `auxiliary.compression.*`으로 자동 마이그레이션됩니다(구성 버전 17).
:::

압축에 사용할 수 있는 프로바이더가 없으면 Hermes는 세션을 실패시키는 대신 요약을 생성하지 않고 대화 중간의 턴을 삭제합니다.

---

## 위임 프로바이더 재정의

`delegate_task`로 생성된 서브에이전트는 부모 에이전트의 기본 대체 체인을 상속합니다. 그래도 비용 최적화를 위해 서브에이전트에 다른 기본 프로바이더:model 쌍을 지정할 수 있습니다.

```yaml
delegation:
  provider: "openrouter"                      # override provider for all subagents
  model: "google/gemini-3-flash-preview"      # override model
  # base_url: "http://localhost:1234/v1"      # or use a direct endpoint
  # api_key: "local-key"
```

전체 구성 세부 사항은 [서브에이전트 위임](/user-guide/features/delegation)을 참조하세요.

---

## Cron 작업 프로바이더

Cron 작업은 에이전트를 생성할 때 구성된 `fallback_providers` 체인(또는 기존 `fallback_model`)을 상속합니다. Cron 작업에 다른 기본 프로바이더를 사용하려면 Cron 작업 자체에 `provider` 및 `model` 재정의를 구성합니다.

```python
cronjob(
    action="create",
    schedule="every 2h",
    prompt="Check server status",
    provider="openrouter",
    model="google/gemini-3-flash-preview"
)
```

전체 구성 세부 사항은 [예약 작업(Cron)](/user-guide/features/cron)을 참조하세요.

---

## 요약

| 기능 | 대체 메커니즘 | 구성 위치 |
|---------|-------------------|----------------|
| 주 에이전트 모델 | `config.yaml`의 `fallback_providers` — 오류 발생 시 턴별 장애 조치(매 턴 기본 모델 복원) | `fallback_providers:`(최상위 목록) |
| 보조 작업(모든 작업) — auto 사용자 | 용량 오류 발생 시 전체 자동 감지 체인(주 에이전트 모델 우선, 이후 프로바이더 체인) | `auxiliary.<task>.provider: auto` |
| 보조 작업(모든 작업) — 명시적 프로바이더 | 용량 오류에 한해 `fallback_chain`(설정된 경우) → 주 에이전트 모델 → 경고 후 발생 | `auxiliary.<task>.fallback_chain` |
| 비전 | 계층형(위 참조) + 내부 OpenRouter 재시도 | `auxiliary.vision` |
| 웹 추출 | 계층형(위 참조) + 내부 OpenRouter 재시도 | `auxiliary.web_extract` |
| 컨텍스트 압축 | 계층형(위 참조); 모든 계층을 사용할 수 없으면 요약 없음으로 저하 | `auxiliary.compression` |
| Skills Hub | 계층형(위 참조) | `auxiliary.skills_hub` |
| MCP 도우미 | 계층형(위 참조) | `auxiliary.mcp` |
| 승인 분류 | 계층형(위 참조) | `auxiliary.approval` |
| 제목 생성 | 계층형(위 참조) | `auxiliary.title_generation` |
| 트리아지 지정 | 계층형(위 참조) | `auxiliary.triage_specifier` |
| 위임 | 부모의 `fallback_providers` 체인을 상속하며, 선택적으로 프로바이더/모델을 재정의 | `delegation.provider` / `delegation.model` |
| Cron 작업 | 구성된 `fallback_providers` 체인을 상속하며, 선택적으로 작업별 프로바이더를 재정의 | 작업별 `provider` / `model` |
