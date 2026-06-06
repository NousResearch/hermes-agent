---
title: 폴백 공급자 (Fallback Providers)
description: 기본 모델을 사용할 수 없을 때 백업 LLM 공급자로 자동 페일오버되도록 구성합니다.
sidebar_label: 폴백 공급자
sidebar_position: 8
---

# 폴백 공급자 (Fallback Providers)

Hermes Agent는 공급자에게 문제가 발생했을 때 세션이 계속 실행되도록 유지하는 세 가지 복원력(resilience) 계층을 가지고 있습니다:

1. **[자격 증명 풀 (Credential pools)](./credential-pools.md)** — *동일한* 공급자에 대한 여러 API 키를 순환합니다 (가장 먼저 시도됨).
2. **기본 모델 폴백 (Primary model fallback)** — 메인 모델이 실패할 때 자동으로 *다른* 공급자:모델로 전환합니다.
3. **보조 작업 폴백 (Auxiliary task fallback)** — 비전, 압축, 웹 추출과 같은 부가적인 작업을 위한 독립적인 공급자 확인.

자격 증명 풀은 동일한 공급자 내에서의 순환(예: 여러 개의 OpenRouter 키)을 처리합니다. 이 페이지에서는 공급자 간 폴백에 대해 다룹니다. 두 기능 모두 선택 사항이며 독립적으로 작동합니다.

## 기본 모델 폴백 (Primary Model Fallback)

메인 LLM 공급자에게 속도 제한, 서버 과부하, 인증 실패, 연결 끊김 등의 에러가 발생할 때, Hermes는 대화 내용을 잃지 않고 세션 도중에 백업 공급자:모델 쌍으로 자동으로 전환할 수 있습니다.

### 구성 (Configuration)

가장 쉬운 방법은 대화형 관리자를 사용하는 것입니다:

```bash
hermes fallback
```

`hermes fallback`은 `hermes model`의 공급자 선택기를 재사용하므로 공급자 목록, 자격 증명 프롬프트, 유효성 검사가 모두 동일합니다. 체인을 관리하려면 하위 명령어인 `add`, `list` (별칭 `ls`), `remove` (별칭 `rm`), `clear`를 사용하세요. 변경 사항은 `config.yaml` 파일의 최상위 `fallback_providers:` 목록 아래에 유지됩니다.

YAML 파일을 직접 편집하려면 `~/.hermes/config.yaml`에 최상위 `fallback_providers` 목록을 추가하세요:

```yaml
fallback_providers:
  - provider: openrouter
    model: anthropic/claude-sonnet-4
```

각 항목에는 `provider`와 `model`이 모두 필요합니다. 둘 중 하나라도 누락된 항목은 무시됩니다.

:::note `fallback_model` vs `fallback_providers`
`fallback_providers` (복수형, 목록)는 현재 구성 형태이며, 순서대로 시도되는 여러 개의 폴백을 지원합니다. `fallback_model` (단수형)은 레거시 단일 폴백 키입니다 — Hermes는 하위 호환성을 위해 이 키를 계속 인식하지만, `hermes fallback`은 현재의 `fallback_providers` 키를 기록하며, 기록 시 레거시 구성을 마이그레이션합니다. 두 가지가 모두 설정된 경우 `fallback_providers`가 우선합니다.
:::

### 지원되는 공급자

| 공급자 | 값 (Value) | 요구 사항 |
|----------|-------|-------------|
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` |
| Nous Portal | `nous` | `hermes setup --portal` (새로 설치) 또는 `hermes auth add nous` (OAuth) |
| OpenAI Codex | `openai-codex` | `hermes model` (ChatGPT OAuth) |
| GitHub Copilot | `copilot` | `COPILOT_GITHUB_TOKEN`, `GH_TOKEN`, 또는 `GITHUB_TOKEN` |
| GitHub Copilot ACP | `copilot-acp` | 외부 프로세스 (에디터 통합) |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` 또는 Claude Code 자격 증명 |
| z.ai / GLM | `zai` | `GLM_API_KEY` |
| Kimi / Moonshot | `kimi-coding` | `KIMI_API_KEY` |
| MiniMax | `minimax` | `MINIMAX_API_KEY` |
| MiniMax (China) | `minimax-cn` | `MINIMAX_CN_API_KEY` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` |
| NVIDIA NIM | `nvidia` | `NVIDIA_API_KEY` (선택 사항: `NVIDIA_BASE_URL`) |
| GMI Cloud | `gmi` | `GMI_API_KEY` (선택 사항: `GMI_BASE_URL`) |
| StepFun | `stepfun` | `STEPFUN_API_KEY` (선택 사항: `STEPFUN_BASE_URL`) |
| Ollama Cloud | `ollama-cloud` | `OLLAMA_API_KEY` |
| Google Gemini (OAuth) | `google-gemini-cli` | `hermes model` (Google OAuth; 선택 사항: `HERMES_GEMINI_PROJECT_ID`) |
| Google AI Studio | `gemini` | `GOOGLE_API_KEY` (별칭: `GEMINI_API_KEY`) |
| xAI (Grok) | `xai` (별칭 `grok`) | `XAI_API_KEY` (선택 사항: `XAI_BASE_URL`) |
| xAI Grok OAuth (SuperGrok) | `xai-oauth` (별칭 `grok-oauth`) | `hermes model` → xAI Grok OAuth (브라우저 로그인; SuperGrok 구독) |
| AWS Bedrock | `bedrock` | 표준 boto3 인증 (`AWS_REGION` + `AWS_PROFILE` 또는 `AWS_ACCESS_KEY_ID`) |
| Qwen Portal (OAuth) | `qwen-oauth` | `hermes model` (Qwen Portal OAuth; 선택 사항: `HERMES_QWEN_BASE_URL`) |
| MiniMax (OAuth) | `minimax-oauth` | `hermes model` (MiniMax portal OAuth) |
| OpenCode Zen | `opencode-zen` | `OPENCODE_ZEN_API_KEY` |
| OpenCode Go | `opencode-go` | `OPENCODE_GO_API_KEY` |
| Kilo Code | `kilocode` | `KILOCODE_API_KEY` |
| Xiaomi MiMo | `xiaomi` | `XIAOMI_API_KEY` |
| Arcee AI | `arcee` | `ARCEEAI_API_KEY` |
| GMI Cloud | `gmi` | `GMI_API_KEY` |
| Alibaba / DashScope | `alibaba` | `DASHSCOPE_API_KEY` |
| Alibaba Coding Plan | `alibaba-coding-plan` | `ALIBABA_CODING_PLAN_API_KEY` (`DASHSCOPE_API_KEY`로 폴백됨) |
| Kimi / Moonshot (China) | `kimi-coding-cn` | `KIMI_CN_API_KEY` |
| StepFun | `stepfun` | `STEPFUN_API_KEY` |
| Tencent TokenHub | `tencent-tokenhub` | `TOKENHUB_API_KEY` |
| Microsoft Foundry | `azure-foundry` | `AZURE_FOUNDRY_API_KEY` + `AZURE_FOUNDRY_BASE_URL` |
| LM Studio (로컬) | `lmstudio` | `LM_API_KEY` (로컬의 경우 없음) + `LM_BASE_URL` |
| Hugging Face | `huggingface` | `HF_TOKEN` |
| 사용자 지정 엔드포인트 | `custom` | `base_url` + `key_env` (아래 참조) |

### 사용자 지정 엔드포인트 폴백 (Custom Endpoint Fallback)

사용자 지정 OpenAI 호환 엔드포인트의 경우 `base_url`과, 선택적으로 `key_env`를 추가하세요:

```yaml
fallback_providers:
  - provider: custom
    model: my-local-model
    base_url: http://localhost:8000/v1
    key_env: MY_LOCAL_KEY            # API 키를 포함하는 환경 변수 이름
```

### 폴백이 작동하는 시점

기본 모델이 다음과 같은 이유로 실패할 때 폴백이 자동으로 활성화됩니다:

- **속도 제한 (Rate limits)** (HTTP 429) — 재시도 시도가 소진된 후
- **서버 에러 (Server errors)** (HTTP 500, 502, 503) — 재시도 시도가 소진된 후
- **인증 실패 (Auth failures)** (HTTP 401, 403) — 즉시 (재시도 의미 없음)
- **찾을 수 없음 (Not found)** (HTTP 404) — 즉시
- **잘못된 응답 (Invalid responses)** — API가 형식이 잘못되었거나 빈 응답을 반복적으로 반환할 때

폴백이 트리거되면 Hermes는 다음을 수행합니다:

1. 폴백 공급자의 자격 증명을 확인합니다.
2. 새로운 API 클라이언트를 빌드합니다.
3. 그 자리에서 모델, 공급자, 클라이언트를 교체합니다.
4. 재시도 카운터를 재설정하고 대화를 계속합니다.

이 전환은 원활하게 이루어집니다 — 대화 기록, 도구 호출(tool calls), 컨텍스트가 모두 보존됩니다. 에이전트는 모델만 다를 뿐 멈춘 위치에서 정확히 대화를 이어갑니다.

:::info 세션당이 아닌 턴당 적용
폴백은 **턴(turn) 범위**입니다: 각각의 새로운 사용자 메시지는 복원된 기본 모델로 시작합니다. 턴 도중에 기본 모델이 실패하면 해당 턴에만 폴백이 활성화됩니다. 다음 메시지가 들어오면 Hermes는 다시 기본 모델을 시도합니다. 단일 턴 내에서 폴백은 최대 한 번만 활성화되며, 만약 폴백마저 실패하면 일반적인 에러 처리 절차(재시도 후 에러 메시지 출력)가 진행됩니다. 이를 통해 한 턴 안에서 계속되는 페일오버 루프를 방지하는 동시에 기본 모델에게 턴마다 새로운 기회를 줍니다.
:::

### 예시

**Anthropic 기본의 폴백으로 OpenRouter 사용:**
```yaml
model:
  provider: anthropic
  default: claude-sonnet-4-6

fallback_providers:
  - provider: openrouter
    model: anthropic/claude-sonnet-4
```

**OpenRouter의 폴백으로 Nous Portal 사용:**
```yaml
model:
  provider: openrouter
  default: anthropic/claude-opus-4

fallback_providers:
  - provider: nous
    model: nous-hermes-3
```

**클라우드의 폴백으로 로컬 모델 사용:**
```yaml
fallback_providers:
  - provider: custom
    model: llama-3.1-70b
    base_url: http://localhost:8000/v1
    key_env: LOCAL_API_KEY
```

**폴백으로 Codex OAuth 사용:**
```yaml
fallback_providers:
  - provider: openai-codex
    model: gpt-5.3-codex
```

### 폴백이 지원되는 곳

| 컨텍스트 | 폴백 지원 여부 |
|---------|-------------------|
| CLI 세션 | ✔ |
| 메시징 게이트웨이 (Telegram, Discord 등) | ✔ |
| 하위 에이전트 위임 (Subagent delegation) | ✔ (하위 에이전트는 상위의 폴백 체인을 상속받음) |
| Cron 작업 | ✔ (cron 에이전트는 구성된 폴백 공급자를 상속받음) |
| 보조 작업 (비전, 압축) | ✘ (자체 공급자 체인을 사용 — 아래 참조) |

:::tip
기본 폴백 체인을 위한 환경 변수는 없습니다 — 반드시 `config.yaml`이나 `hermes fallback`을 통해서만 구성하세요. 이것은 의도적인 설계입니다: 폴백 구성은 신중한 선택이어야 하며, 오래된 환경 변수(stale shell export)로 인해 덮어쓰여져서는 안 됩니다.
:::

---

## 보조 작업 폴백 (Auxiliary Task Fallback)

Hermes는 부가적인 작업(side tasks)을 위해 별도의 경량 모델을 사용합니다. 각 작업에는 내장된 폴백 시스템 역할을 하는 고유한 공급자 확인 체인이 있습니다.

### 독립적인 공급자 확인을 가진 작업들

| 작업 | 하는 일 | 구성 키 |
|------|-------------|-----------|
| 비전 (Vision) | 이미지 분석, 브라우저 스크린샷 | `auxiliary.vision` |
| 웹 추출 (Web Extract) | 웹 페이지 요약 | `auxiliary.web_extract` |
| 압축 (Compression) | 컨텍스트 압축 요약 | `auxiliary.compression` |
| 스킬 허브 (Skills Hub) | 스킬 검색 및 발견 | `auxiliary.skills_hub` |
| MCP | MCP 도우미 연산 | `auxiliary.mcp` |
| 승인 (Approval) | 스마트 명령어 승인 분류 | `auxiliary.approval` |
| 제목 생성 (Title Generation) | 세션 제목 요약 | `auxiliary.title_generation` |
| 트리아지 구체화기 (Triage Specifier) | `hermes kanban specify` / 대시보드 ✨ 버튼 — 한 줄짜리 분류 작업을 진짜 스펙으로 구체화 | `auxiliary.triage_specifier` |

### 자동 감지 체인 (Auto-Detection Chain)

작업의 공급자가 `"auto"`(기본값)로 설정된 경우, Hermes는 성공할 때까지 순서대로 공급자를 시도합니다:

**텍스트 작업의 경우 (압축, 웹 추출 등):**

```text
OpenRouter → Nous Portal → 사용자 지정 엔드포인트 → Codex OAuth →
API 키 공급자 (z.ai, Kimi, MiniMax, Xiaomi MiMo, Hugging Face, Anthropic) → 포기(give up)
```

**비전 작업의 경우:**

```text
기본 공급자 (비전 기능이 있는 경우) → OpenRouter → Nous Portal →
Codex OAuth → Anthropic → 사용자 지정 엔드포인트 → 포기(give up)
```

결정된 공급자가 호출 시점에 실패하면 Hermes는 내부적으로 재시도(internal retry)를 수행합니다: 공급자가 OpenRouter가 아니고 명시적인 `base_url`이 설정되지 않은 경우, 최후의 수단으로 OpenRouter로 폴백을 시도합니다.

### 보조 공급자 구성

각 작업은 `config.yaml`에서 독립적으로 구성할 수 있습니다:

```yaml
auxiliary:
  vision:
    provider: "auto"              # auto | openrouter | nous | codex | main | anthropic
    model: ""                     # 예: "openai/gpt-4o"
    base_url: ""                  # 직접 엔드포인트 (공급자보다 우선함)
    api_key: ""                   # base_url을 위한 API 키

  web_extract:
    provider: "auto"
    model: ""

  compression:
    provider: "auto"
    model: ""

  skills_hub:
    provider: "auto"
    model: ""

  mcp:
    provider: "auto"
    model: ""
```

위의 모든 작업은 동일한 **provider / model / base_url** 패턴을 따릅니다. 컨텍스트 압축은 `auxiliary.compression` 아래에서 구성됩니다:

```yaml
auxiliary:
  compression:
    provider: main                                    # 다른 보조 작업과 동일한 공급자 옵션
    model: google/gemini-3-flash-preview
    base_url: null                                    # 사용자 지정 OpenAI 호환 엔드포인트
```

그리고 기본 폴백 체인은 다음을 사용합니다:

```yaml
fallback_providers:
  - provider: openrouter
    model: anthropic/claude-sonnet-4
    # base_url: http://localhost:8000/v1             # 선택적 사용자 지정 엔드포인트
```

보조 작업, 압축, 폴백 세 가지 모두 같은 방식으로 작동합니다: `provider`를 설정하여 요청을 처리할 곳을 지정하고, `model`로 어떤 모델을 사용할지 지정하며, `base_url`을 설정하여 사용자 지정 엔드포인트를 가리킵니다(이는 `provider`를 덮어씁니다).

### 보조 작업을 위한 공급자 옵션

이 옵션들은 `auxiliary:`, `compression:`, `fallback_providers:` 항목에만 적용됩니다 — `"main"`은 최상위 `model.provider`의 유효한 값이 **아닙니다**. 사용자 지정 엔드포인트의 경우 `model:` 섹션에서 `provider: custom`을 사용하세요 ([AI 공급자(AI Providers)](/integrations/providers) 참조).

| 공급자 | 설명 | 요구 사항 |
|----------|-------------|-------------|
| `"auto"` | 작동할 때까지 순서대로 공급자를 시도합니다 (기본값) | 최소 하나 이상의 공급자가 구성되어야 함 |
| `"openrouter"` | 강제로 OpenRouter 사용 | `OPENROUTER_API_KEY` |
| `"nous"` | 강제로 Nous Portal 사용 | `hermes auth` |
| `"codex"` | 강제로 Codex OAuth 사용 | `hermes model` → Codex |
| `"main"` | 메인 에이전트가 사용하는 공급자를 사용합니다 (보조 작업 전용) | 활성 메인 공급자가 구성되어야 함 |
| `"anthropic"` | 강제로 Anthropic 네이티브 사용 | `ANTHROPIC_API_KEY` 또는 Claude Code 자격 증명 |

### 직접 엔드포인트 덮어쓰기 (Direct Endpoint Override)

어떤 보조 작업이든 `base_url`을 설정하면 공급자 확인을 완전히 건너뛰고 해당 엔드포인트로 직접 요청을 보냅니다:

```yaml
auxiliary:
  vision:
    base_url: "http://localhost:1234/v1"
    api_key: "local-key"
    model: "qwen2.5-vl"
```

`base_url`은 `provider`보다 우선합니다. Hermes는 인증에 설정된 `api_key`를 사용하며, 설정되지 않은 경우 `OPENAI_API_KEY`로 폴백합니다. 사용자 지정 엔드포인트에는 `OPENROUTER_API_KEY`를 재사용하지 **않습니다**.

---

## 보조 용량 에러 폴백 (Auxiliary Capacity-Error Fallback)

명시적인 보조 공급자를 설정하면 (예: `auxiliary.vision.provider: glm`), Hermes는 이를 우선 선택 사항으로 취급합니다 — 하지만 해당 공급자가 **용량 에러 (capacity error)** (HTTP 402 결제 필요, HTTP 429 일일 할당량 소진, 연결 실패) 때문에 요청을 전혀 처리할 수 없는 경우, Hermes는 조용히 실패하는 대신 단계적인 체인을 통해 폴백합니다:

1. **기본 보조 공급자** — 구성한 공급자 (항상 가장 먼저 시도됨)
2. **`auxiliary.<task>.fallback_chain`** — 작성한 경우, 작업별 덮어쓰기 목록
3. **메인 에이전트 공급자 + 모델** — 최후의 안전망 (체인을 작성하지 않았더라도 항상 시도됨)
4. **경고 후 에러 재발생** — 모든 계층이 실패하면, Hermes는 WARNING 레벨로 `Auxiliary <task>: ... all fallbacks exhausted`를 기록하고 원래의 에러를 다시 발생시킵니다.

일시적인 HTTP 429 속도 제한 (`Retry-After: ...`)은 용량 문제가 아닌 요청 제약으로 처리됩니다 — 이는 명시적인 공급자 선택을 존중하며 폴백 사다리를 트리거하지 **않습니다**. 오직 일일/월간 할당량 소진, 결제 에러, 연결 실패만이 명시적 공급자 제약을 우회합니다.

`provider: auto`를 사용하는 사용자(명시적인 보조 공급자 없음)의 경우, 기존의 자동 감지 체인이 2~3단계를 대신하여 실행됩니다. 첫 번째 단계가 이미 메인 에이전트 모델이므로 `auto` 사용자는 설정 없이 동일한 결과를 얻습니다.

### 선택 사항: 작업별 폴백 체인 (per-task fallback chain)

"메인 에이전트 모델을 우선"으로 하는 기본 폴백 순서와 다른 순서를 원한다면, `fallback_chain`을 명시적으로 구성하세요. 각 항목에는 최소한 `provider`가 필요하며; `model`, `base_url`, `api_key`는 선택 사항입니다.

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
```

폴백을 얻기 위해 `fallback_chain`을 구성할 필요는 **없습니다** — 메인 에이전트 안전망은 항상 작동합니다. 기본값과 다른 특별한 순서를 원할 때만 사용하세요.

### 폴백을 트리거하는 공급자 할당량 에러

Hermes는 다음과 같은 경우를 일시적인 속도 제한이 아닌 402 크레딧 소진과 동등한 용량 에러로 인식합니다:

- Bedrock / LiteLLM: `Too many tokens per day`, `daily limit`, `tokens per day`
- Vertex AI / GCP: `quota exceeded`, `resource exhausted`, `RESOURCE_EXHAUSTED`
- 일반(Generic): `daily quota`, `quota_exceeded`

만약 공급자가 일일 할당량 소진에 대해 다른 문구를 반환하여 Hermes가 폴백을 트리거하지 않는다면 버그입니다 — 정확한 에러 문자열과 함께 이슈를 열어주세요.

---

## 컨텍스트 압축 폴백 (Context Compression Fallback)

컨텍스트 압축은 `auxiliary.compression` 구성 블록을 사용하여 요약을 처리할 모델과 공급자를 제어합니다:

```yaml
auxiliary:
  compression:
    provider: "auto"                              # auto | openrouter | nous | main
    model: "google/gemini-3-flash-preview"
```

:::info 레거시 마이그레이션
`compression.summary_model` / `compression.summary_provider` / `compression.summary_base_url`이 있는 오래된 설정은 처음 로드될 때(config version 17) 자동으로 `auxiliary.compression.*`으로 마이그레이션됩니다.
:::

압축을 위한 공급자를 사용할 수 없는 경우, Hermes는 세션을 실패 처리하는 대신 요약을 생성하지 않고 중간 대화 턴들을 드롭(drop)합니다.

---

## 위임 공급자 덮어쓰기 (Delegation Provider Override)

`delegate_task`로 생성된 하위 에이전트는 상위 에이전트의 기본 폴백 체인을 상속합니다. 비용 최적화를 위해 하위 에이전트를 다른 기본 공급자:모델 쌍으로 라우팅할 수도 있습니다:

```yaml
delegation:
  provider: "openrouter"                      # 모든 하위 에이전트에 대한 공급자 재정의
  model: "google/gemini-3-flash-preview"      # 모델 재정의
  # base_url: "http://localhost:1234/v1"      # 또는 직접 엔드포인트 사용
  # api_key: "local-key"
```

전체 구성 세부 정보는 [하위 에이전트 위임 (Subagent Delegation)](/user-guide/features/delegation)을 참조하세요.

---

## Cron 작업 공급자 (Cron Job Providers)

Cron 작업은 에이전트를 생성할 때 설정된 `fallback_providers` 체인 (또는 레거시 `fallback_model`)을 상속합니다. cron 작업에 대해 다른 기본 공급자를 사용하려면 해당 cron 작업 자체에서 `provider` 및 `model`을 덮어쓰도록 구성하세요:

```python
cronjob(
    action="create",
    schedule="every 2h",
    prompt="Check server status",
    provider="openrouter",
    model="google/gemini-3-flash-preview"
)
```

전체 구성 세부 정보는 [예약된 작업 (Scheduled Tasks (Cron))](/user-guide/features/cron)을 참조하세요.

---

## 요약 (Summary)

| 기능 | 폴백 메커니즘 | 구성 위치 |
|---------|-------------------|----------------|
| 메인 에이전트 모델 | config.yaml의 `fallback_providers` — 에러 발생 시 턴 단위 페일오버 (매 턴마다 기본 모델 복원됨) | `fallback_providers:` (최상위 목록) |
| 보조 작업 (전체) — auto 사용자 | 용량 에러 발생 시 전체 자동 감지 체인(메인 에이전트 모델을 먼저 시도 후 공급자 체인) | `auxiliary.<task>.provider: auto` |
| 보조 작업 (전체) — 명시적 공급자 | 용량 에러 발생 시에만 `fallback_chain` (설정된 경우) → 메인 에이전트 모델 → 경고 및 에러 발생 | `auxiliary.<task>.fallback_chain` |
| 비전 (Vision) | 계층형 (위 참조) + 내부 OpenRouter 재시도 | `auxiliary.vision` |
| 웹 추출 (Web extraction) | 계층형 (위 참조) + 내부 OpenRouter 재시도 | `auxiliary.web_extract` |
| 컨텍스트 압축 | 계층형 (위 참조); 모든 계층을 사용할 수 없는 경우 요약 없음으로 기능 축소 | `auxiliary.compression` |
| 스킬 허브 | 계층형 (위 참조) | `auxiliary.skills_hub` |
| MCP 도우미 | 계층형 (위 참조) | `auxiliary.mcp` |
| 승인 분류 | 계층형 (위 참조) | `auxiliary.approval` |
| 제목 생성 | 계층형 (위 참조) | `auxiliary.title_generation` |
| 트리아지 구체화기 | 계층형 (위 참조) | `auxiliary.triage_specifier` |
| 위임 (Delegation) | 공급자 덮어쓰기만 가능 (자동 폴백 없음) | `delegation.provider` / `delegation.model` |
| Cron 작업 | 작업별 공급자 덮어쓰기만 가능 (자동 폴백 없음) | 작업별 `provider` / `model` |
