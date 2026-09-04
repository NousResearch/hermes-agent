---
title: "AI 제공업체"
sidebar_label: "AI 제공업체"
sidebar_position: 1
---

# AI 제공업체

이 페이지에서는 OpenRouter와 Anthropic 같은 클라우드 API부터 Ollama와 vLLM 같은 자체 호스팅 엔드포인트, 고급 라우팅 및 폴백 구성까지 Hermes Agent의 추론 제공업체를 설정하는 방법을 설명합니다. Hermes를 사용하려면 최소 하나의 제공업체를 구성해야 합니다.

## 추론 제공업체

LLM에 연결할 방법이 최소 하나 필요합니다. `hermes model`을 사용해 제공업체와 모델을 대화형으로 전환하거나 직접 구성할 수 있습니다.

| 제공업체 | 설정 |
|----------|-------|
| **Nous Portal** | `hermes model` (OAuth, 구독 기반) |
| **OpenAI Codex** | `hermes model` → **ChatGPT 또는 Codex Subscription** (ChatGPT OAuth, Codex 모델 사용) |
| **GitHub Copilot** | `hermes model` (OAuth 디바이스 코드 플로, `COPILOT_GITHUB_TOKEN`, `GH_TOKEN` 또는 `gh auth token`) |
| **GitHub Copilot ACP** | `hermes model` (로컬 `copilot --acp --stdio` 실행) |
| **Anthropic** | `hermes model` (OAuth를 통한 Claude Max + 추가 사용 크레딧; Anthropic API 키 또는 수동 setup-token도 지원 — 아래 참고) |
| **OpenRouter** | `~/.hermes/.env`의 `OPENROUTER_API_KEY` |
| **Fireworks AI** | `~/.hermes/.env`의 `FIREWORKS_API_KEY` (제공업체: `fireworks`; 별칭: `fireworks-ai`, `fw`) |
| **NovitaAI** | `~/.hermes/.env`의 `NOVITA_API_KEY` (제공업체: `novita`, 200개 이상의 모델, Model API, Agent Sandbox, GPU Cloud) |
| **AI Gateway** | `~/.hermes/.env`의 `AI_GATEWAY_API_KEY` (제공업체: `ai-gateway`) |
| **z.ai / GLM** | `~/.hermes/.env`의 `GLM_API_KEY` (제공업체: `zai`) |
| **Kimi / Moonshot** | `~/.hermes/.env`의 `KIMI_API_KEY` (제공업체: `kimi-coding`) |
| **Kimi / Moonshot (중국)** | `~/.hermes/.env`의 `KIMI_CN_API_KEY` (제공업체: `kimi-coding-cn`; 별칭: `kimi-cn`, `moonshot-cn`) |
| **Arcee AI** | `~/.hermes/.env`의 `ARCEEAI_API_KEY` (제공업체: `arcee`; 별칭: `arcee-ai`, `arceeai`) |
| **GMI Cloud** | `~/.hermes/.env`의 `GMI_API_KEY` (제공업체: `gmi`; 별칭: `gmi-cloud`, `gmicloud`) |
| **Actual Computer** | 호스팅 릴레이에는 `ACTUAL_API_KEY`, 로컬 데몬에는 `ACTUAL_BASE_URL=http://127.0.0.1:8080` — 루프백에서는 키가 필요 없음 (제공업체: `actual`; 별칭: `actual-computer`, `actualcomputer`, `aci`) |
| **MiniMax** | `~/.hermes/.env`의 `MINIMAX_API_KEY` (제공업체: `minimax`) |
| **MiniMax 중국** | `~/.hermes/.env`의 `MINIMAX_CN_API_KEY` (제공업체: `minimax-cn`) |
| **xAI (Grok) — Responses API** | `~/.hermes/.env`의 `XAI_API_KEY` (제공업체: `xai`) |
| **xAI Grok OAuth (SuperGrok)** | `hermes model` → "xAI Grok OAuth (SuperGrok / Premium+)" — 브라우저 로그인, API 키 불필요. [가이드](../guides/xai-grok-oauth.md) 참고 |
| **Qwen Cloud (Alibaba DashScope)** | `~/.hermes/.env`의 `DASHSCOPE_API_KEY` (제공업체: `alibaba`) |
| **Alibaba Cloud (Coding Plan)** | `DASHSCOPE_API_KEY` (제공업체: `alibaba-coding-plan`, 별칭: `alibaba_coding`) — 별도 과금 SKU, 다른 엔드포인트 |
| **Kilo Code** | `~/.hermes/.env`의 `KILOCODE_API_KEY` (제공업체: `kilocode`) |
| **Xiaomi MiMo** | `~/.hermes/.env`의 `XIAOMI_API_KEY` (제공업체: `xiaomi`, 별칭: `mimo`, `xiaomi-mimo`) |
| **Tencent TokenHub** | `~/.hermes/.env`의 `TOKENHUB_API_KEY` (제공업체: `tencent-tokenhub`, 별칭: `tencent`, `tokenhub`, `tencentmaas`) |
| **OpenCode Zen** | `~/.hermes/.env`의 `OPENCODE_ZEN_API_KEY` (제공업체: `opencode-zen`) |
| **OpenCode Go** | `~/.hermes/.env`의 `OPENCODE_GO_API_KEY` (제공업체: `opencode-go`) |
| **DeepSeek** | `~/.hermes/.env`의 `DEEPSEEK_API_KEY` (제공업체: `deepseek`) |
| **Hugging Face** | `~/.hermes/.env`의 `HF_TOKEN` (제공업체: `huggingface`, 별칭: `hf`) |
| **Google / Gemini** | `~/.hermes/.env`의 `GOOGLE_API_KEY` (또는 `GEMINI_API_KEY`) (제공업체: `gemini`) |
| **Google Vertex AI** | `hermes model` → "Google Vertex AI" (제공업체: `vertex`; 서비스 계정 JSON 또는 ADC를 통한 OAuth2, GCP 과금) |
| **OpenAI API (직접 연결)** | `~/.hermes/.env`의 `OPENAI_API_KEY` (제공업체: `openai-api`, 선택 사항 `OPENAI_BASE_URL`) |
| **Azure AI Foundry** | `hermes model` → "Azure AI Foundry" (제공업체: `azure-foundry`; Azure OpenAI / Foundry 엔드포인트 및 키 사용) |
| **AWS Bedrock** | `hermes model` → "AWS Bedrock" (제공업체: `bedrock`; boto3를 통한 표준 AWS 자격 증명 체인) |
| **NVIDIA Build** | `~/.hermes/.env`의 `NVIDIA_API_KEY` (build.nvidia.com의 NIM 호스팅 모델) |
| **Ollama Cloud** | `hermes model` → "Ollama Cloud" (제공업체: `ollama-cloud`; 클라우드 호스팅 Ollama API) |
| **Qwen OAuth** | `hermes model` → "Qwen OAuth" (제공업체: `qwen-oauth`; 브라우저 PKCE 로그인) |
| **MiniMax OAuth** | `hermes model` → "MiniMax (OAuth)" (제공업체: `minimax-oauth`; 브라우저 PKCE 로그인) |
| **StepFun** | `~/.hermes/.env`의 `STEPFUN_API_KEY` (제공업체: `stepfun`) |
| **LM Studio** | `hermes model` → "LM Studio" (제공업체: `lmstudio`, 선택 사항 `LM_API_KEY`) |
| **Custom Endpoint** | `hermes model` → "Custom endpoint" 선택 (`config.yaml`에 저장) |

공식 API 키 경로는 전용 [Google Gemini 가이드](/guides/google-gemini)를 참고하세요.

:::tip 모델 키 별칭
`model:` 구성 섹션에서는 모델 ID의 키 이름으로 `default:` 또는 `model:` 중 하나를 사용할 수 있습니다. `model: { default: my-model }`과 `model: { model: my-model }`은 동일하게 작동합니다.
:::


### Nous Portal

[Nous Portal](https://portal.nousresearch.com)은 Nous Research의 통합 구독 게이트웨이이며 **Hermes Agent를 실행하는 데 권장되는 방법**입니다. 한 번의 OAuth 로그인으로 300개 이상의 최첨단 에이전트 모델(Claude, GPT, Gemini, DeepSeek, Qwen, Kimi, GLM, MiniMax, Grok, ...)과 [Tool Gateway](/user-guide/features/tool-gateway)(웹 검색, 이미지 생성, TTS, 브라우저 자동화)를 이용할 수 있습니다. 별도의 제공업체 계정마다 과금되는 대신 Nous 구독으로 과금됩니다.

```bash
hermes setup --portal     # fresh install — OAuth + provider + gateway in one command
hermes model              # existing install — pick "Nous Portal" from the list
hermes portal info        # inspect login + routing at any time
```

아직 구독이 없나요? [portal.nousresearch.com/manage-subscription](https://portal.nousresearch.com/manage-subscription)에서 구독하세요.

**전체 세부 정보:** 전용 [Nous Portal 통합 페이지](/integrations/nous-portal)(구독 내용, 모델 카탈로그, 문제 해결)와 단계별 [Nous Portal로 Hermes Agent 실행 가이드](/guides/run-hermes-with-nous-portal)를 참고하세요.

**클라이언트 식별.** Hermes Agent의 모든 Portal 요청에는 `client=hermes-client-v<version>` 태그(예: `client=hermes-client-v0.13.0`)가 설치된 릴리스에 맞춰 자동으로 추가됩니다. 이 태그는 주 채팅 루프, 보조 호출, 압축 요약, 웹 추출 등 모든 Portal 경로에 전송되며 Portal 측 텔레메트리가 Hermes 트래픽을 다른 클라이언트와 구분할 수 있게 합니다. 별도 구성이 필요하지 않으며 `hermes update`를 실행하면 태그가 자동으로 업데이트됩니다.

**JWT 인증(자동).** Hermes는 Portal 요청에 범위가 지정된 `inference:invoke` JWT를 우선 사용하고, 기존 불투명 세션 키 경로를 폴백으로 사용합니다. 별도 구성이 필요하지 않습니다. 자격 증명은 OAuth 플로에서 관리되고 투명하게 교체됩니다. 취소된 refresh token은 재생 루프를 방지하기 위해 격리됩니다.


:::info Codex 참고
OpenAI Codex 제공업체는 디바이스 코드로 인증합니다(URL을 열고 코드를 입력). Hermes는 결과 자격 증명을 `~/.hermes/auth.json`의 자체 인증 저장소에 저장하며, 존재하는 경우 기존 Codex CLI 자격 증명을 `~/.codex/auth.json`에서 가져올 수 있습니다. Codex CLI를 설치할 필요는 없습니다.

토큰 새로 고침이 최종 오류(HTTP 4xx, `invalid_grant`, 취소된 grant 등)로 실패하면 Hermes는 refresh token을 만료된 것으로 표시하고 반복 재생을 중지하므로 동일한 인증 오류가 쏟아지지 않습니다. 대신 다음 요청에서 유형이 지정된 재인증 메시지를 표시합니다. `hermes auth add openai-codex`(또는 `hermes model` → **ChatGPT 또는 Codex Subscription**)를 실행해 새 디바이스 코드 로그인을 시작하세요. 다음 교환이 성공하면 격리가 해제됩니다.
:::

:::warning
Nous Portal, Codex 또는 사용자 지정 엔드포인트를 사용하더라도 일부 도구(비전, 웹 요약, MoA)는 별도의 "보조" 모델을 사용합니다. 기본값(`auxiliary.*.provider: "auto"`)에서 Hermes는 이러한 작업을 **주 채팅 모델**(`hermes model`에서 선택한 동일한 모델)로 라우팅합니다. 각 작업을 개별적으로 재정의해 더 저렴하거나 빠른 모델(예: OpenRouter의 Gemini Flash)로 라우팅할 수 있습니다 — [보조 모델](/user-guide/configuration#auxiliary-models)을 참고하세요.
:::

:::tip Nous Tool Gateway
유료 Nous Portal 구독자는 구독을 통해 라우팅되는 **[Tool Gateway](/user-guide/features/tool-gateway)**(웹 검색, 이미지 생성, TTS, 브라우저 자동화)도 이용할 수 있습니다. 추가 API 키가 필요하지 않습니다. 새로 설치할 때 `hermes setup --portal`은 한 명령으로 로그인하고 Nous를 제공업체로 설정하며 게이트웨이를 켭니다. 기존 사용자는 `hermes model` 또는 도구별 `hermes tools`에서 활성화할 수 있습니다. `hermes portal info`로 언제든 라우팅을 확인하세요.
:::

### 모델 관리를 위한 두 명령

Hermes에는 서로 다른 용도로 사용하는 **두 가지** 모델 명령이 있습니다.

| 명령 | 실행 위치 | 기능 |
|---------|-------------|--------------|
| **`hermes model`** | 세션 외부의 터미널 | 전체 설정 마법사 — 제공업체 추가, OAuth 실행, API 키 입력, 엔드포인트 구성 |
| **`/model`** | Hermes 채팅 세션 내부 | **이미 구성된** 제공업체와 모델 사이의 빠른 전환 |

아직 설정하지 않은 제공업체로 전환하려는 경우(예: OpenRouter만 구성되어 있고 Anthropic을 사용하려는 경우) `/model`이 아니라 `hermes model`이 필요합니다. 먼저 세션을 종료하고(`Ctrl+C` 또는 `/quit`) `hermes model`을 실행해 제공업체 설정을 완료한 다음 새 세션을 시작하세요.


### 구독 플랜: 플랜에서 결제하는 항목

여러 제공업체에서는 API 키 대신 **소비자 구독**(Claude Max, ChatGPT, SuperGrok / X Premium+ 등)으로 Hermes에 로그인할 수 있습니다. 해당 구독이 실제로 결제하는 항목과 결제하지 않는 항목은 제공업체마다 다르며, 결제 관련 혼란이 발생하는 가장 흔한 원인입니다. 아래 표는 요약본이며 각 제공업체의 섹션에 자세한 내용이 있습니다.

> *현재 문서화되지 않음*으로 표시된 셀은 정확히 그 의미입니다. Hermes 문서에 아직 동작이 지정되어 있지 않습니다. 추측하지 말고 제공업체의 결제 대시보드를 확인하며, 아직 답이 정해지지 않은 질문으로 취급하세요.

| 플랜 / 경로 | Hermes에서 사용할 수 있나요? | 소비되는 항목 | 소비되지 않는 항목 | 흔한 오해 |
|---|---|---|---|---|
| **Anthropic — Claude Max + OAuth** | ✅ 예 — `hermes model` → Anthropic OAuth. Max **및** 구매한 추가 사용 크레딧 필요 | Max 플랜에 추가로 구매한 **추가/초과 사용 크레딧** | Claude Code에 기본 포함된 **기본 Max 플랜 사용량** | 포함된 Max 사용량이 그대로 남아 있어도 Hermes 사용량은 모두 "추가 사용량"으로 청구됨 |
| **Anthropic — Claude Pro** | ❌ 아니요 — Pro 구독자는 OAuth 경로를 사용할 수 없음 | 없음(경로 사용 불가) | Pro 구독 | Pro라면 작동할 것 같지만 작동하지 않음. 대신 `ANTHROPIC_API_KEY` 사용(Claude 구독과 무관한 토큰당 결제) |
| **OpenAI Codex — ChatGPT 플랜 OAuth** | ✅ 예 — `hermes model` → **ChatGPT 또는 Codex Subscription** (ChatGPT OAuth 디바이스 코드 로그인, Codex 모델 사용) | *현재 문서화되지 않음* | *현재 문서화되지 않음* | 문서에는 인증 및 토큰 새로 고침만 설명되어 있으며 플랜 할당량 의미는 아직 문서화되지 않음 |
| **xAI — SuperGrok / X Premium+ OAuth** | ✅ 예 — 브라우저 OAuth, API 키 불필요 | **구독 할당량**(X Search에 대해서는 OAuth가 API 키보다 우선되고 "API 지출 대신 구독 할당량을 사용"한다고 명시되어 있음). 그 외 추론 할당량 의미: *현재 문서화되지 않음* | OAuth 자격 증명이 구성되고 우선 사용될 때의 `XAI_API_KEY` / 토큰당 API 지출 | 로그인 성공 후 `HTTP 403` — 앱 내 구독이 활성 상태여도 xAI가 OAuth API 접근을 특정 SuperGrok 등급으로 제한함 |
| **Google — Gemini 소비자 플랜 (Google AI Pro / Ultra)** | ❌ 문서화된 경로 없음 — `gemini` 제공업체는 API 키 전용(`GOOGLE_API_KEY` / `GEMINI_API_KEY`); Vertex AI는 GCP로 과금 | API 키의 **할당량**(무료 등급 또는 결제가 활성화된 Google Cloud 프로젝트) — *소비자 플랜의 소비 방식은 현재 문서화되지 않음* | *현재 문서화되지 않음* | Hermes는 사용자 턴마다 여러 모델 호출을 할 수 있어 무료 등급 키가 몇 번의 에이전트 턴만에 소진될 수 있음 |

**Anthropic.** OAuth 경로는 Anthropic 계정에 Claude Code로 연결되며 **구매한 추가 사용 크레딧이 있는 Claude Max 플랜에서만 작동**합니다. Hermes는 기본 Max 사용량을 소비하지 않고, 추가로 구매한 추가/초과 사용 크레딧만 소비합니다. Claude Pro 구독자는 이 경로를 사용할 수 없습니다. 지원되는 대안은 `ANTHROPIC_API_KEY`이며 해당 키의 조직에 표준 API 가격으로 토큰당 청구됩니다. 아래 [Anthropic (Native)](#anthropic-native)를 참고하세요.

**OpenAI Codex.** Hermes는 ChatGPT 디바이스 코드 OAuth로 인증하고 자격 증명을 `~/.hermes/auth.json`에 저장하며, 기존 Codex CLI 자격 증명을 `~/.codex/auth.json`에서 가져올 수 있습니다. 어떤 ChatGPT 플랜 등급이 대상인지와 Hermes 사용량이 플랜의 Codex 한도에 어떻게 반영되는지는 **현재 문서화되지 않았습니다**. [Nous Portal](#nous-portal)의 Codex 참고 사항은 인증 및 토큰 새로 고침 동작만 설명합니다.

**xAI (SuperGrok / X Premium+).** 브라우저 OAuth는 연결된 X 계정의 활성 SuperGrok 구독 또는 X Premium+ 구독에서 작동하며, 동일한 bearer token이 xAI에 직접 연결되는 도구(TTS, 이미지 생성, 동영상 생성, 전사, X Search)에서도 재사용됩니다. 로그인 성공 후 추론에서 `HTTP 403`이 반환되면 오래된 토큰이 아니라 xAI 측의 등급/권한 제한입니다. 해결 방법은 `XAI_API_KEY`로 전환하는 것입니다. 아래 [xAI (Grok)](#xai-grok--responses-api--prompt-caching)와 [xAI Grok OAuth 가이드](../guides/xai-grok-oauth.md)를 참고하세요.

**Google Gemini.** 현재 소비자 Gemini 구독으로 Hermes에 로그인하는 방법은 없습니다. `gemini` 제공업체는 API 키를 사용하고 [Google Vertex AI](#google-vertex-ai)는 GCP 프로젝트에 과금합니다. 에이전트 사용에는 결제가 활성화된 Google Cloud 프로젝트를 권장합니다. 무료 등급 할당량은 장시간 실행하는 에이전트 세션에 너무 작습니다. [Google Gemini 가이드](/guides/google-gemini)를 참고하세요.

:::tip 구독 하나로 다섯 개를 대신하기
제공업체별 플랜 의미를 일일이 추적하고 싶지 않다면 [Nous Portal](#nous-portal)이 한 번의 OAuth 로그인과 단일 구독으로 300개 이상의 모델을 제공합니다.
:::

### Anthropic (Native)

OpenRouter 프록시 없이 Anthropic API를 통해 Claude 모델을 직접 사용합니다. 세 가지 인증 방법을 지원합니다.

:::caution Claude Max "추가 사용" 크레딧 필요
`hermes model` → Anthropic OAuth(또는 `hermes auth add anthropic --type oauth`)로 인증하면 Hermes는 Anthropic 계정에 Claude Code로 연결합니다. **Claude Max 플랜에 가입되어 있고 추가 사용 크레딧을 구매한 경우에만 작동합니다.** Claude Code에 기본 포함된 기본 Max 플랜 사용량은 Hermes에서 소비되지 않으며, 추가로 구매한 추가/초과 사용 크레딧만 소비됩니다. Claude Pro 구독자는 이 경로를 사용할 수 없습니다.

Max와 추가 크레딧이 없다면 대신 `ANTHROPIC_API_KEY`를 사용하세요. 해당 키의 조직에 표준 API 가격으로 토큰당 청구되며 Claude 구독과는 무관합니다.
:::

```bash
# With an API key (pay-per-token)
export ANTHROPIC_API_KEY=***
hermes chat --provider anthropic --model claude-sonnet-4-6

# Preferred: authenticate through `hermes model`
# Hermes will use Claude Code's credential store directly when available
hermes model

# Manual override with a setup-token (fallback / legacy)
export ANTHROPIC_TOKEN=***  # setup-token or manual OAuth token
hermes chat --provider anthropic

# Auto-detect Claude Code credentials (if you already use Claude Code)
hermes chat --provider anthropic  # reads Claude Code credential files automatically
```

`hermes model`에서 Anthropic OAuth를 선택하면 Hermes는 토큰을 `~/.hermes/.env`에 복사하는 대신 Claude Code 자체 자격 증명 저장소를 우선 사용합니다. 이렇게 하면 새로 고칠 수 있는 Claude 자격 증명이 계속 새로 고쳐질 수 있습니다.

또는 영구적으로 설정합니다.
```yaml
model:
  provider: "anthropic"
  default: "claude-sonnet-4-6"
```

:::tip 별칭
`--provider claude`와 `--provider claude-code`도 `--provider anthropic`의 단축형으로 작동합니다.
:::

### GitHub Copilot

Hermes는 GitHub Copilot을 두 가지 모드의 일급 제공업체로 지원합니다.

**`copilot` — 직접 Copilot API**(권장). GitHub Copilot 구독을 사용해 Copilot API를 통해 GPT-5.x, Claude, Gemini 및 기타 모델에 접근합니다.

```bash
hermes chat --provider copilot --model gpt-5.4
```

**인증 옵션**(다음 순서로 확인):

1. `COPILOT_GITHUB_TOKEN` 환경 변수
2. `GH_TOKEN` 환경 변수
3. `GITHUB_TOKEN` 환경 변수
4. `gh auth token` CLI 폴백

토큰을 찾지 못하면 `hermes model`에서 **OAuth 디바이스 코드 로그인**을 제공합니다. Copilot CLI 및 opencode와 동일한 흐름입니다.

:::warning 토큰 유형
Copilot API는 일반 Personal Access Token(`ghp_*`)을 지원하지 않습니다. 지원되는 토큰 유형은 다음과 같습니다.

| 유형 | 접두사 | 발급 방법 |
|------|--------|------------|
| OAuth 토큰 | `gho_` | `hermes model` → GitHub Copilot → GitHub로 로그인 |
| 세분화된 PAT | `github_pat_` | GitHub Settings → Developer settings → Fine-grained tokens (**Copilot Requests** 권한 필요) |
| GitHub App 토큰 | `ghu_` | GitHub App 설치를 통해 |

`gh auth token`이 `ghp_*` 토큰을 반환하면 `hermes model`을 사용해 OAuth로 인증하세요.
:::

:::info Hermes의 Copilot 인증 동작
Hermes는 지원되는 GitHub 토큰(`gho_*`, `github_pat_*` 또는 `ghu_*`)을 `api.githubcopilot.com`에 직접 전송하고 Copilot 전용 헤더(`Editor-Version`, `Copilot-Integration-Id`, `Openai-Intent`, `x-initiator`)를 포함합니다.

HTTP 401이 발생하면 Hermes는 폴백하기 전에 한 번 자격 증명 복구를 수행합니다.

1. 일반 우선순위 체인(`COPILOT_GITHUB_TOKEN` → `GH_TOKEN` → `GITHUB_TOKEN` → `gh auth token`)으로 토큰을 다시 확인
2. 새 헤더로 공유 OpenAI 클라이언트를 다시 생성
3. 요청을 한 번 재시도

일부 오래된 커뮤니티 프록시는 `api.github.com/copilot_internal/v2/token` 교환 흐름을 사용합니다. 이 엔드포인트는 일부 계정 유형에서 사용할 수 없으며(404 반환), 따라서 Hermes는 직접 토큰 인증을 기본 경로로 유지하고 견고성을 위해 런타임 자격 증명 새로 고침과 재시도에 의존합니다.
:::

**API 라우팅:** GPT-5+ 모델(`gpt-5-mini` 제외)은 자동으로 Responses API를 사용합니다. 나머지 모든 모델(GPT-4o, Claude, Gemini 등)은 Chat Completions를 사용합니다. 모델은 실시간 Copilot 카탈로그에서 자동 감지됩니다.

**`copilot-acp` — Copilot ACP 에이전트 백엔드.** 로컬 Copilot CLI를 하위 프로세스로 실행합니다.

```bash
hermes chat --provider copilot-acp --model copilot-acp
# Requires the GitHub Copilot CLI in PATH and an existing `copilot login` session
```

**영구 구성:**
```yaml
model:
  provider: "copilot"
  default: "gpt-5.4"
```

| 환경 변수 | 설명 |
|---------------------|-------------|
| `COPILOT_GITHUB_TOKEN` | Copilot API용 GitHub 토큰(최우선) |
| `HERMES_COPILOT_ACP_COMMAND` | Copilot CLI 바이너리 경로 재정의(기본값: `copilot`) |
| `HERMES_COPILOT_ACP_ARGS` | ACP 인수 재정의(기본값: `--acp --stdio`) |

### 일급 API 키 제공업체

이 제공업체들은 전용 제공업체 ID로 내장 지원됩니다. API 키를 설정하고 `--provider`로 선택하세요.

```bash
# Fireworks AI
hermes chat --provider fireworks --model accounts/fireworks/models/kimi-k2p6
# Requires: FIREWORKS_API_KEY in ~/.hermes/.env

# NovitaAI Model API
hermes chat --provider novita --model moonshotai/kimi-k2.5
# Requires: NOVITA_API_KEY in ~/.hermes/.env

# z.ai / ZhipuAI GLM
hermes chat --provider zai --model glm-5
# Requires: GLM_API_KEY in ~/.hermes/.env

# Kimi / Moonshot AI (international: api.moonshot.ai)
hermes chat --provider kimi-coding --model kimi-for-coding
# Requires: KIMI_API_KEY in ~/.hermes/.env

# Kimi / Moonshot AI (China: api.moonshot.cn)
hermes chat --provider kimi-coding-cn --model kimi-k2.5
# Requires: KIMI_CN_API_KEY in ~/.hermes/.env

# MiniMax (global endpoint)
hermes chat --provider minimax --model MiniMax-M2.7
# Requires: MINIMAX_API_KEY in ~/.hermes/.env

# MiniMax (China endpoint)
hermes chat --provider minimax-cn --model MiniMax-M2.7
# Requires: MINIMAX_CN_API_KEY in ~/.hermes/.env

# Qwen Cloud / DashScope (Qwen models)
hermes chat --provider alibaba --model qwen3.5-plus
# Requires: DASHSCOPE_API_KEY in ~/.hermes/.env

# Xiaomi MiMo
hermes chat --provider xiaomi --model mimo-v2-pro
# Requires: XIAOMI_API_KEY in ~/.hermes/.env

# Tencent TokenHub (Hy3 Preview)
hermes chat --provider tencent-tokenhub --model hy3-preview
# Requires: TOKENHUB_API_KEY in ~/.hermes/.env

# Arcee AI (Trinity models)
hermes chat --provider arcee --model trinity-large-thinking
# Requires: ARCEEAI_API_KEY in ~/.hermes/.env

# GMI Cloud
# Use the exact model ID returned by GMI's /v1/models endpoint.
hermes chat --provider gmi --model zai-org/GLM-5.1-FP8
# Requires: GMI_API_KEY in ~/.hermes/.env
```

Fireworks는 `accounts/fireworks/models/kimi-k2p6` 같은 네이티브 슬래시 형식의 카탈로그 ID를 사용합니다. `hermes model`을 실행하고 **Fireworks AI**를 선택한 다음 실시간 카탈로그에서 선택하거나 다른 Fireworks 모델 ID를 입력하세요. 기본 엔드포인트는 `https://api.fireworks.ai/inference/v1`입니다. 다른 엔드포인트는 `.env`가 아니라 `config.yaml`의 `model.base_url`을 통해 구성하세요.

또는 `config.yaml`에서 제공업체를 영구적으로 설정합니다.
```yaml
model:
  provider: "gmi"
  default: "zai-org/GLM-5.1-FP8"
```

기본 URL은 `NOVITA_BASE_URL`, `GLM_BASE_URL`, `KIMI_BASE_URL`, `MINIMAX_BASE_URL`, `MINIMAX_CN_BASE_URL`, `DASHSCOPE_BASE_URL`, `XIAOMI_BASE_URL`, `GMI_BASE_URL` 또는 `TOKENHUB_BASE_URL` 환경 변수로 재정의할 수 있습니다.

:::note Z.AI 엔드포인트 자동 감지
Z.AI / GLM 제공업체를 사용할 때 Hermes는 API 키를 허용하는 엔드포인트를 찾기 위해 여러 엔드포인트(글로벌, 중국, 코딩 변형)를 자동으로 탐색합니다. `GLM_BASE_URL`을 수동으로 설정할 필요가 없습니다. 작동하는 엔드포인트가 자동으로 감지되고 캐시됩니다.
:::

### xAI (Grok) — Responses API + 프롬프트 캐싱

xAI는 Grok 4 모델에서 자동 추론 지원을 제공하기 위해 Responses API(`codex_responses` 전송)를 통해 연결됩니다. `reasoning_effort` 매개변수는 필요하지 않으며 서버가 기본적으로 추론합니다. `~/.hermes/.env`에 `XAI_API_KEY`를 설정하고 `hermes model`에서 xAI를 선택하거나 `/model grok-4-fast-reasoning`의 단축형으로 `grok`을 입력하세요.

SuperGrok 및 X Premium+ 구독자는 API 키 대신 브라우저 OAuth로 로그인할 수 있습니다. `hermes model`에서 **xAI Grok OAuth (SuperGrok / Premium+)**를 선택하거나 `hermes auth add xai-oauth`를 실행하세요. 동일한 OAuth bearer token은 xAI에 직접 연결되는 도구(TTS, 이미지 생성, 동영상 생성, 전사)에서도 자동으로 재사용됩니다. 전체 흐름은 [xAI Grok OAuth 가이드](../guides/xai-grok-oauth.md)를 참고하세요. Hermes가 원격 호스트에서 실행 중이면 필요한 `ssh -L` 터널에 대해 [SSH / 원격 호스트를 통한 OAuth](../guides/oauth-over-ssh.md)도 참고하세요.

xAI를 제공업체로 사용할 때(`x.ai`를 포함하는 모든 기본 URL) Hermes는 모든 API 요청에 `x-grok-conv-id` 헤더를 전송해 프롬프트 캐싱을 자동으로 활성화합니다. 이 헤더는 요청을 대화 세션 내 동일한 서버로 라우팅하여 xAI 인프라가 캐시된 시스템 프롬프트와 대화 기록을 재사용할 수 있게 합니다.

별도 구성이 필요하지 않습니다. xAI 엔드포인트가 감지되고 세션 ID가 있으면 캐싱이 자동으로 활성화됩니다. 이를 통해 여러 턴의 대화에서 지연 시간과 비용을 줄일 수 있습니다.

xAI는 전용 TTS 엔드포인트(`/v1/tts`)도 제공합니다. `hermes tools` → Voice & TTS에서 **xAI TTS**를 선택하거나 [Voice & TTS](../user-guide/features/tts.md#text-to-speech) 페이지에서 구성을 확인하세요.

**사용 중단된 xAI 모델 마이그레이션(2026년 5월 15일):** xAI는 2026-05-15에 `grok-4*`, `grok-3`, `grok-code-fast-1` 및 `grok-imagine-image-pro`를 사용 중단합니다. `hermes doctor`와 `hermes chat` 시작 과정은 사용 중단된 참조를 가리키는 구성을 감지하고 권장 대체 항목을 출력합니다. `hermes migrate xai`를 사용하면 구성을 한 번에 다시 쓸 수 있습니다. 기본적으로 미리 보기만 수행하며, 변경 사항을 쓰려면 `--apply`를 추가합니다(타임스탬프가 붙은 `config.yaml.bak-pre-migrate-xai-*` 백업이 자동으로 생성됨).

```bash
hermes migrate xai          # preview replacements
hermes migrate xai --apply  # rewrite ~/.hermes/config.yaml in place
```

**xAI 웹 검색 백엔드.** [웹 검색](../user-guide/features/web-search.md) 도구 세트가 활성화되면 `web.backend: xai`는 동일한 `XAI_API_KEY` / OAuth 자격 증명을 사용해 xAI 호스팅 검색 엔드포인트로 검색을 라우팅합니다. xAI가 이미 제공업체로 구성되어 있다면 추가 설정이 필요하지 않습니다.

### NovitaAI

[NovitaAI](https://novita.ai)는 빌더와 에이전트를 위한 AI 네이티브 클라우드입니다. 한 플랫폼에서 200개 이상의 모델을 제공하는 Model API, AI 에이전트를 만들고 실행하는 Agent Sandbox, 확장 가능한 컴퓨팅을 위한 GPU Cloud라는 세 가지 제품군을 제공합니다.

```bash
# Use any available model
hermes chat --provider novita --model moonshotai/kimi-k2.5
# Requires: NOVITA_API_KEY in ~/.hermes/.env

# Short alias
hermes chat --provider novita-ai --model deepseek/deepseek-v3-0324
```

또는 `config.yaml`에서 영구적으로 설정합니다.
```yaml
model:
  provider: "novita"
  default: "moonshotai/kimi-k2.5"
  base_url: "https://api.novita.ai/openai/v1"
```

[novita.ai/settings/key-management](https://novita.ai/settings/key-management)에서 API 키를 발급하세요. 기본 URL은 `NOVITA_BASE_URL`로 재정의할 수 있습니다.

### Ollama Cloud — 관리형 Ollama 모델, OAuth + API 키

[Ollama Cloud](https://ollama.com/cloud)는 GPU가 필요하지 않은 환경에서 로컬 Ollama와 동일한 오픈 웨이트 카탈로그를 호스팅합니다. `hermes model`에서 **Ollama Cloud**를 선택하고 [ollama.com/settings/keys](https://ollama.com/settings/keys)의 API 키를 붙여 넣으면 Hermes가 사용 가능한 모델을 자동으로 검색합니다.

```bash
hermes model
# → pick "Ollama Cloud"
# → paste your OLLAMA_API_KEY
# → select from discovered models (gpt-oss:120b, glm-4.6:cloud, qwen3-coder:480b-cloud, etc.)
```

또는 `config.yaml`에 직접 설정합니다.
```yaml
model:
  provider: "ollama-cloud"
  default: "gpt-oss:120b"
```

모델 카탈로그는 `ollama.com/v1/models`에서 동적으로 가져와 한 시간 동안 캐시됩니다. `model:tag` 표기(예: `qwen3-coder:480b-cloud`)는 정규화 과정에서 유지되므로 대시를 사용하지 마세요.

:::tip Ollama Cloud와 로컬 Ollama 비교
둘 다 동일한 OpenAI 호환 API를 사용합니다. Cloud는 일급 제공업체(`--provider ollama-cloud`, `OLLAMA_API_KEY`)이고, 로컬 Ollama는 사용자 지정 엔드포인트 흐름(기본 URL `http://localhost:11434/v1`, 키 없음)으로 연결합니다. 로컬에서 실행할 수 없는 대형 모델에는 Cloud를, 개인정보 보호 또는 오프라인 작업에는 로컬을 사용하세요.
:::

### AWS Bedrock

AWS Bedrock을 통해 Anthropic Claude, Amazon Nova, DeepSeek v3.2, Meta Llama 4 및 기타 모델을 사용합니다. AWS SDK(`boto3`) 자격 증명 체인을 사용하므로 API 키가 필요하지 않고 표준 AWS 인증만 있으면 됩니다.

```bash
# Simplest — named profile in ~/.aws/credentials
hermes chat --provider bedrock --model us.anthropic.claude-sonnet-4-6

# Or with explicit env vars
AWS_PROFILE=myprofile AWS_REGION=us-east-1 hermes chat --provider bedrock --model us.anthropic.claude-sonnet-4-6
```

또는 `config.yaml`에 영구적으로 설정합니다.
```yaml
model:
  provider: "bedrock"
  default: "us.anthropic.claude-sonnet-4-6"
bedrock:
  region: "us-east-1"          # or set AWS_REGION
  # profile: "myprofile"       # or set AWS_PROFILE
  # discovery: true            # auto-discover region from IAM
  # guardrail:                 # optional Bedrock Guardrails
  #   guardrail_identifier: "your-guardrail-id"
  #   guardrail_version: "DRAFT"
```

인증은 표준 boto3 체인을 사용합니다. 명시적 `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`, `~/.aws/credentials`의 `AWS_PROFILE`, EC2/ECS/Lambda의 IAM 역할, IMDS 또는 SSO를 지원합니다. 이미 AWS CLI로 인증했다면 환경 변수가 필요하지 않습니다.

Bedrock은 내부적으로 **Converse API**를 사용합니다. 요청을 Bedrock의 모델 독립적 형식으로 변환하므로 동일한 구성이 Claude, Nova, DeepSeek 및 Llama 모델에서 작동합니다. 기본값이 아닌 리전 엔드포인트를 호출할 때만 `BEDROCK_BASE_URL`을 설정하세요.

IAM 설정, 리전 선택 및 리전 간 추론을 단계별로 설명하는 [AWS Bedrock 가이드](/guides/aws-bedrock)를 참고하세요.

### Google Vertex AI

Vertex의 OpenAI 호환 엔드포인트를 통해 Google Cloud Vertex AI에서 Gemini 모델을 사용합니다. 인증은 **OAuth2**입니다. 서비스 계정 JSON 또는 Application Default Credentials(ADC)에서 단기 액세스 토큰(약 1시간)을 발급합니다. **정적 API 키는 없습니다.** Hermes가 토큰을 발급하고 자동으로 새로 고치며, 세션 중간에 발생한 `401`에도 토큰을 다시 발급합니다.

```bash
# Service account JSON (recommended for servers / gateways)
echo "VERTEX_CREDENTIALS_PATH=/path/to/service-account.json" >> ~/.hermes/.env
# or Application Default Credentials
gcloud auth application-default login

hermes model   # → "Google Vertex AI" → project → region → model
```

또는 `config.yaml`에 설정합니다(프로젝트/리전은 비밀이 아니므로 여기에 저장하고 자격 증명 경로는 `.env`에 둡니다).
```yaml
model:
  provider: "vertex"
  default: "google/gemini-3-flash-preview"   # Vertex requires the google/ prefix
vertex:
  project_id: "my-gcp-project"   # blank → use the project embedded in the credentials
  region: "global"               # required for the Gemini 3.x previews
```

`VERTEX_PROJECT_ID` / `VERTEX_REGION` 환경 변수가 `config.yaml` 값을 재정의합니다. Hermes는 처음 사용할 때 `google-auth`를 지연 설치합니다. 관리형 설치를 복구해야 한다면 `hermes setup`을 실행하세요. 전체 단계는 [Google Vertex AI 가이드](/guides/google-vertex)를, 정적 API 키를 사용하는 AI Studio 경로는 [Google Gemini 가이드](/guides/google-gemini)를 참고하세요.

### Qwen Portal (OAuth)

브라우저 기반 OAuth 로그인을 사용하는 Alibaba의 Qwen Portal입니다. `hermes model`에서 **Qwen OAuth (Portal)**를 선택하고 브라우저에서 로그인하면 Hermes가 refresh token을 저장합니다.

```bash
hermes model
# → pick "Qwen OAuth (Portal)"
# → browser opens; sign in with your Alibaba account
# → confirm — credentials are saved to ~/.hermes/auth.json

hermes chat   # uses portal.qwen.ai/v1 endpoint
```

또는 `config.yaml`을 구성합니다.
```yaml
model:
  provider: "qwen-oauth"
  default: "qwen3-coder-plus"
```

Portal 엔드포인트가 변경되는 경우에만 `HERMES_QWEN_BASE_URL`을 설정하세요(기본값: `https://portal.qwen.ai/v1`).

:::tip Qwen OAuth와 Qwen Cloud (Alibaba DashScope) 비교
`qwen-oauth`는 OAuth 로그인 방식의 소비자용 Qwen Portal을 사용하므로 개인 사용자에게 적합합니다. `alibaba` 제공업체는 `DASHSCOPE_API_KEY`를 사용하는 Qwen Cloud(Alibaba DashScope)를 사용하므로 프로그래밍 방식 또는 프로덕션 워크로드에 적합합니다. 둘 다 Qwen 계열 모델로 라우팅하지만 엔드포인트는 서로 다릅니다.
:::

### Alibaba Cloud (Coding Plan)

표준 DashScope API 접근과 별도의 가격 SKU인 Alibaba의 **Coding Plan**을 구독한 경우 Hermes는 이를 일급 제공업체 `alibaba-coding-plan`으로 노출합니다. 엔드포인트는 `https://coding-intl.dashscope.aliyuncs.com/v1`입니다. 일반 `alibaba` 제공업체와 마찬가지로 OpenAI 호환이지만 기본 URL과 과금 방식이 다릅니다.

```yaml
model:
  provider: alibaba_coding     # alias for alibaba-coding-plan
  model: qwen3-coder-plus
```

또는 CLI에서 실행합니다.

```bash
hermes chat --provider alibaba_coding --model qwen3-coder-plus
```

`alibaba_coding`은 기존 `alibaba` 항목에서 사용하는 것과 동일한 `DASHSCOPE_API_KEY`를 사용합니다. 별도 키가 필요하지 않고 다른 라우팅 대상만 사용합니다. 이 제공업체가 등록되기 전에는 `config.yaml`에서 `provider: alibaba_coding`을 설정한 사용자가 조용히 OpenRouter 라우팅으로 넘어갔습니다.

### MiniMax (OAuth)

브라우저 OAuth 로그인으로 MiniMax-M2.7을 사용하며 API 키가 필요하지 않습니다. `hermes model`에서 **MiniMax (OAuth)**를 선택하고 브라우저에서 로그인하면 Hermes가 액세스 토큰과 refresh token을 저장합니다. 내부적으로 Anthropic Messages 호환 엔드포인트(`/anthropic`)를 사용합니다.

```bash
hermes model
# → pick "MiniMax (OAuth)"
# → browser opens; sign in with your MiniMax account (global or CN region)
# → confirm — credentials are saved to ~/.hermes/auth.json

hermes chat   # uses api.minimax.io/anthropic endpoint
```

또는 `config.yaml`을 구성합니다.
```yaml
model:
  provider: "minimax-oauth"
  default: "MiniMax-M2.7"
```

지원 모델은 `MiniMax-M2.7`(주 모델)과 `MiniMax-M2.7-highspeed`(```default auxiliary model```으로 연결됨)입니다. OAuth 경로에서는 `MINIMAX_API_KEY` / `MINIMAX_BASE_URL`을 무시합니다.

:::tip MiniMax OAuth와 API 키 비교
`minimax-oauth`는 OAuth 로그인 방식의 MiniMax 소비자용 포털을 사용하므로 과금 설정이 필요하지 않습니다. `minimax` 및 `minimax-cn` 제공업체는 `MINIMAX_API_KEY` / `MINIMAX_CN_API_KEY`를 사용해 프로그래밍 방식으로 접근합니다. 전체 단계는 [MiniMax OAuth 가이드](/guides/minimax-oauth)를 참고하세요.
:::

### NVIDIA NIM

[build.nvidia.com](https://build.nvidia.com)의 Nemotron 및 기타 오픈 소스 모델(무료 API 키) 또는 로컬 NIM 엔드포인트를 사용합니다.

```bash
# Cloud (build.nvidia.com)
hermes chat --provider nvidia --model nvidia/nemotron-3-super-120b-a12b
# Requires: NVIDIA_API_KEY in ~/.hermes/.env

# Local NIM endpoint — override base URL
NVIDIA_BASE_URL=http://localhost:8000/v1 hermes chat --provider nvidia --model nvidia/nemotron-3-super-120b-a12b
```

또는 `config.yaml`에서 영구적으로 설정합니다.
```yaml
model:
  provider: "nvidia"
  default: "nvidia/nemotron-3-super-120b-a12b"
```

:::tip 로컬 NIM
온프레미스 배포(DGX Spark, 로컬 GPU)에서는 `NVIDIA_BASE_URL=http://localhost:8000/v1`을 설정하세요. NIM은 build.nvidia.com과 동일한 OpenAI 호환 chat completions API를 노출하므로 클라우드와 로컬 사이를 한 줄의 환경 변수 변경으로 전환할 수 있습니다.
:::

Hermes는 `build.nvidia.com`으로 보내는 모든 요청에 NIM 결제 출처 헤더를 자동으로 추가하므로 별도 구성이 필요하지 않습니다. 이를 통해 NVIDIA 결제 대시보드에서 올바른 출처로 사용량이 집계됩니다.

### GMI Cloud

[GMI Cloud](https://www.gmicloud.ai/)를 통한 오픈 모델 및 추론 모델입니다. OpenAI 호환 API와 API 키 인증을 사용합니다.

```bash
# GMI Cloud
hermes chat --provider gmi --model deepseek-ai/DeepSeek-V3.2
# Requires: GMI_API_KEY in ~/.hermes/.env
```

또는 `config.yaml`에서 영구적으로 설정합니다.
```yaml
model:
  provider: "gmi"
  default: "deepseek-ai/DeepSeek-V3.2"
```

기본 URL은 `GMI_BASE_URL`로 재정의할 수 있습니다(기본값: `https://api.gmi-serving.com/v1`).

### Actual Computer

[Actual Computer](https://actual.inc)를 통해 자신의 하드웨어를 비공개 추론 클러스터로 사용합니다. 두 가지 제공 방식 모두 OpenAI 호환이며(Hermes는 Responses API 전송을 사용함) 다음과 같습니다.

- **호스팅 릴레이** — `https://api.actual.inc`, 종단 간 암호화로 *자신의* 클러스터로 라우팅합니다. [actual.inc/user/keys](https://actual.inc/user/keys)에서 발급한 `ac_` 추론 키로 인증합니다.
- **로컬 데몬** — `http://127.0.0.1:8080`에서 장치 내 실행되며 완전히 오프라인입니다. API 키가 필요하지 않습니다. Hermes가 루프백 기본 URL을 감지하고 내부 플레이스홀더로 자동 인증합니다.

```bash
# Hosted relay (ACTUAL_API_KEY in ~/.hermes/.env)
hermes chat --provider actual --model <model-id-from-your-cluster>

# Local daemon (ACTUAL_BASE_URL=http://127.0.0.1:8080 in ~/.hermes/.env, no key)
hermes chat --provider actual --model <installed-model-name>
```

또는 `config.yaml`에서 영구적으로 설정합니다.
```yaml
model:
  provider: "actual"
  default: "<model-id>"
```

참고:
- 모델 ID는 클러스터의 `GET /v1/models`에서 가져옵니다. `hermes model` 또는 `curl -s https://api.actual.inc/v1/models -H "Authorization: Bearer $ACTUAL_API_KEY"`로 확인하세요.
- 호스트만 입력한 URL은 정규화됩니다. `ACTUAL_BASE_URL=http://127.0.0.1:8080`은 자동으로 `http://127.0.0.1:8080/v1`이 됩니다.
- 추론 강도는 Actual이 지원하는 범위(`none/low/medium/high/max`)로 제한됩니다. 전역 `xhigh`/`ultra` 설정으로 인해 요청이 400으로 실패하지 않습니다.
- 소형 로컬 모델에서는 Hermes의 전체 기본 도구 세트와 시스템 프롬프트가 32k 컨텍스트 창을 초과할 수 있어 llama.cpp 계열 서버에서 빈 스트림 오류가 발생합니다. 도구 세트를 제한(`-t file,web`)하거나 더 큰 컨텍스트로 모델을 로드하세요. 선택적 `actual-setup` 스킬(`hermes skills install official/devops/actual-setup`)이 설정 및 문제 해결을 자세히 다룹니다.
- 별칭: `actual-computer`, `actualcomputer`, `aci`.

### StepFun

[StepFun](https://platform.stepfun.com)의 Step 계열 모델입니다. OpenAI 호환 API와 API 키 인증을 사용합니다.

```bash
# StepFun
hermes chat --provider stepfun --model step-3.5-flash
# Requires: STEPFUN_API_KEY in ~/.hermes/.env
```

또는 `config.yaml`에서 영구적으로 설정합니다.
```yaml
model:
  provider: "stepfun"
  default: "step-3.5-flash"
```

기본 URL은 `STEPFUN_BASE_URL`로 재정의할 수 있습니다(기본값: `https://api.stepfun.com/v1`).

### Hugging Face Inference Providers

[Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers)는 통합 OpenAI 호환 엔드포인트(`router.huggingface.co/v1`)를 통해 20개 이상의 오픈 모델로 라우팅합니다. 요청은 사용 가능한 가장 빠른 백엔드(Groq, Together, SambaNova 등)로 자동 라우팅되고 자동으로 장애 조치됩니다.

```bash
# Use any available model
hermes chat --provider huggingface --model Qwen/Qwen3.5-397B-A17B
# Requires: HF_TOKEN in ~/.hermes/.env

# Short alias
hermes chat --provider hf --model deepseek-ai/DeepSeek-V3.2
```

또는 `config.yaml`에서 영구적으로 설정합니다:
```yaml
model:
  provider: "huggingface"
  default: "Qwen/Qwen3.5-397B-A17B"
```

토큰은 [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)에서 발급하세요. "Make calls to Inference Providers" 권한을 활성화해야 합니다. 무료 등급이 포함되어 있으며(월 $0.10 크레딧, 제공업체 요금에 마크업 없음) 사용할 수 있습니다.

모델 이름에 라우팅 접미사를 추가할 수 있습니다. `:fastest`(기본값), `:cheapest\` 또는 `:provider_name`을 사용하면 특정 백엔드를 강제할 수 있습니다.

기본 URL은 `HF_BASE_URL`로 재정의할 수 있습니다.

## 사용자 지정 및 자체 호스팅 LLM 제공업체

Hermes Agent는 **모든 OpenAI 호환 API 엔드포인트**와 작동합니다. 서버가 `/v1/chat/completions`를 구현한다면 Hermes를 해당 서버로 지정할 수 있습니다. 따라서 로컬 모델, GPU 추론 서버, 다중 제공업체 라우터 또는 타사 API를 사용할 수 있습니다.

### 일반 설정

사용자 지정 엔드포인트를 구성하는 방법은 세 가지입니다.

**대화형 설정(권장):**
```bash
hermes model
# Select "Custom endpoint (self-hosted / VLLM / etc.)"
# Enter: API base URL, API key, Model name
```

**수동 구성(`config.yaml`):**
```yaml
# In ~/.hermes/config.yaml
model:
  default: your-model-name
  provider: custom
  base_url: http://localhost:8000/v1
  api_key: your-key-or-leave-empty-for-local
```

:::warning 레거시 환경 변수
`.env`의 `LLM_MODEL`은 **제거되었습니다**. `config.yaml`이 모델 및 엔드포인트 구성의 단일 기준입니다. `OPENAI_BASE_URL`은 여전히 적용되지만 `openai-api` 제공업체에만 적용됩니다(직접 API 키 접근을 위한 OpenAI 엔드포인트를 재정의). 다른 제공업체와 사용자 지정 엔드포인트에는 `hermes model`을 사용하거나 `config.yaml`에서 `model.base_url`을 직접 설정하세요. `.env`에 오래된 항목이 있으면 다음 `hermes setup` 또는 구성 마이그레이션에서 자동으로 삭제됩니다.
:::

두 방법 모두 모델, 제공업체 및 기본 URL의 기준인 `config.yaml`에 저장됩니다.

### `/model`로 모델 전환

:::warning hermes model과 /model 비교
**`hermes model`**(채팅 세션 외부의 터미널에서 실행)은 **전체 제공업체 설정 마법사**입니다. 새 제공업체 추가, OAuth 플로 실행, API 키 입력, 사용자 지정 엔드포인트 구성을 여기서 수행합니다.

**`/model`**(활성 Hermes 채팅 세션 내부에서 입력)은 **이미 설정한 제공업체와 모델 사이에서만 전환**할 수 있습니다. 새 제공업체 추가, OAuth 실행 또는 API 키 입력은 할 수 없습니다. 제공업체를 하나만 구성했다면(예: OpenRouter) `/model`에는 해당 제공업체의 모델만 표시됩니다.

**새 제공업체를 추가하려면:** 세션을 종료하고(`Ctrl+C` 또는 `/quit`) `hermes model`을 실행해 새 제공업체를 설정한 다음 새 세션을 시작하세요.
:::

사용자 지정 엔드포인트를 하나 이상 구성하면 세션 중간에 모델을 전환할 수 있습니다.

```
/model custom:qwen-2.5          # Switch to a model on your custom endpoint
/model custom                    # Auto-detect the model from the endpoint
/model openrouter:claude-sonnet-4 # Switch back to a cloud provider
```

**이름이 지정된 사용자 지정 제공업체**(아래 참고)를 구성했다면 세 부분 구문을 사용합니다.

```
/model custom:local:qwen-2.5    # Use the "local" custom provider with model qwen-2.5
/model custom:work:llama3       # Use the "work" custom provider with llama3
```

제공업체를 전환하면 Hermes가 기본 URL과 제공업체를 구성에 저장하므로 재시작 후에도 변경 사항이 유지됩니다. 사용자 지정 엔드포인트에서 내장 제공업체로 전환할 때 오래된 기본 URL은 자동으로 삭제됩니다.

:::tip
`/model custom`(모델 이름 없이 단독 사용)은 엔드포인트의 `/models` API를 조회하고 로드된 모델이 정확히 하나일 때 해당 모델을 자동으로 선택합니다. 단일 모델을 실행하는 로컬 서버에 유용합니다.
:::

이하의 모든 내용은 동일한 패턴을 따릅니다. URL, 키 및 모델 이름만 변경하면 됩니다.

---

### Ollama — 로컬 모델, 설정 불필요

[Ollama](https://ollama.com/)는 한 명령으로 오픈 웨이트 모델을 로컬에서 실행합니다. 빠른 로컬 실험, 개인정보 보호가 중요한 작업, 오프라인 사용에 적합합니다. OpenAI 호환 API를 통한 도구 호출을 지원합니다.

```bash
# Install and run a model
ollama pull qwen2.5-coder:32b
ollama serve   # Starts on port 11434
```

그런 다음 Hermes를 구성합니다.

```bash
hermes model
# Select "Custom endpoint (self-hosted / VLLM / etc.)"
# Enter URL: http://localhost:11434/v1
# Skip API key (Ollama doesn't need one)
# Enter model name (e.g. qwen2.5-coder:32b)
```

또는 `config.yaml`을 직접 구성합니다.

```yaml
model:
  default: qwen2.5-coder:32b
  provider: custom
  base_url: http://localhost:11434/v1
  context_length: 64000   # See warning below
```

:::caution Ollama는 기본 컨텍스트 길이가 매우 짧습니다
Ollama는 기본적으로 모델의 전체 컨텍스트 창을 사용하지 **않습니다**. VRAM에 따라 기본값은 다음과 같습니다.

| 사용 가능한 VRAM | 기본 컨텍스트 |
|----------------|----------------|
| 24GB 미만 | **4,096 토큰** |
| 24–48GB | 32,768 토큰 |
| 48GB 이상 | 256,000 토큰 |

Hermes Agent는 도구를 사용하는 에이전트 작업에 최소 **64,000 토큰**의 컨텍스트가 필요합니다. 시스템 프롬프트, 도구 스키마 및 작업 중인 대화 상태가 안정적인 다단계 워크플로에 충분한 공간을 필요로 하므로 더 작은 창은 시작 시 거부됩니다.

**늘리는 방법**(하나 선택):

```bash
# Option 1: Set server-wide via environment variable (recommended)
OLLAMA_CONTEXT_LENGTH=64000 ollama serve

# Option 2: For systemd-managed Ollama
sudo systemctl edit ollama.service
# Add: Environment="OLLAMA_CONTEXT_LENGTH=64000"
# Then: sudo systemctl daemon-reload && sudo systemctl restart ollama

# Option 3: Bake it into a custom model (persistent per-model)
echo -e "FROM qwen2.5-coder:32b\nPARAMETER num_ctx 64000" > Modelfile
ollama create qwen2.5-coder-64k -f Modelfile
```

**OpenAI 호환 API**(`/v1/chat/completions`)를 통해 컨텍스트 길이를 설정할 수 없습니다. 서버 측 또는 Modelfile을 통해 구성해야 합니다. Hermes 같은 도구와 Ollama를 통합할 때 가장 많이 혼동하는 부분입니다.
:::

**컨텍스트가 올바르게 설정되었는지 확인합니다.**

```bash
ollama ps
# Look at the CONTEXT column — it should show your configured value
```

:::tip
`ollama list`로 사용 가능한 모델을 나열하세요. [Ollama 라이브러리](https://ollama.com/library)의 모델은 `ollama pull <model>`로 가져올 수 있습니다. Ollama는 대부분의 설정에서 GPU 오프로딩을 자동으로 처리하므로 별도 구성이 필요하지 않습니다.
:::

---

### vLLM — 고성능 GPU 추론

[vLLM](https://docs.vllm.ai/)은 프로덕션 LLM 서빙의 표준입니다. GPU 하드웨어에서 최대 처리량, 대형 모델 서빙, 연속 배칭에 적합합니다.

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --port 8000 \
  --max-model-len 65536 \
  --tensor-parallel-size 2 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

그런 다음 Hermes를 구성합니다.

```bash
hermes model
# Select "Custom endpoint (self-hosted / VLLM / etc.)"
# Enter URL: http://localhost:8000/v1
# Skip API key (or enter one if you configured vLLM with --api-key)
# Enter model name: meta-llama/Llama-3.1-70B-Instruct
```

**컨텍스트 길이:** vLLM은 기본적으로 모델의 `max_position_embeddings`를 읽습니다. 이 값이 GPU 메모리를 초과하면 오류가 발생하고 `--max-model-len`을 낮게 설정하라는 메시지가 표시됩니다. `--max-model-len auto`를 사용해 맞는 최대값을 자동으로 찾을 수도 있습니다. `--gpu-memory-utilization 0.95`(기본값 0.9)로 설정하면 VRAM에 더 많은 컨텍스트를 넣을 수 있습니다.

**도구 호출에는 명시적인 플래그가 필요합니다.**

| 플래그 | 용도 |
|------|---------|
| `--enable-auto-tool-choice` | `tool_choice: "auto"`(Hermes의 기본값)에 필요 |
| `--tool-call-parser <name>` | 모델 도구 호출 형식용 파서 |

지원되는 파서: `hermes`(Qwen 2.5, Hermes 2/3), `llama3_json`(Llama 3.x), `mistral`, `deepseek_v3`, `deepseek_v31`, `xlam`, `pythonic`. 이 플래그가 없으면 도구 호출이 작동하지 않고 모델이 도구 호출을 텍스트로 출력합니다.

**Qwen 추론 파서:** OpenAI 호환 서버가 반환하는 `reasoning`, `reasoning_content` 및 스트리밍 추론 델타 같은 구조화된 추론 메타데이터를 Hermes가 보존합니다. 이 메타데이터는 어시스턴트의 표시 답변을 대체하지 않고 추론/사고 추적 데이터로 취급됩니다. vLLM에서 제공하는 Qwen 추론 모델의 경우 최종 사용자 표시 응답이 여전히 `content`에 나타나야 합니다. 배포 환경에서 `--reasoning-parser qwen3`을 사용했을 때 `content`가 비어 있다면 해당 파서를 비활성화하거나 `extra_body`를 통해 `chat_template_kwargs.enable_thinking: false` 같은 서버 지원 요청 옵션을 전달하세요.

:::tip
vLLM은 사람이 읽기 쉬운 크기를 지원합니다. `--max-model-len 64k`(소문자 k = 1000, 대문자 K = 1024)를 사용할 수 있습니다.
:::

---

### SGLang — RadixAttention을 사용한 빠른 서빙

[SGLang](https://github.com/sgl-project/sglang)은 KV 캐시 재사용을 위한 RadixAttention을 제공하는 vLLM의 대안입니다. 다중 턴 대화(접두사 캐싱), 제한된 디코딩, 구조화된 출력에 적합합니다.

```bash
pip install "sglang[all]"
python -m sglang.launch_server \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --port 30000 \
  --context-length 65536 \
  --tp 2 \
  --tool-call-parser qwen
```

그런 다음 Hermes를 구성합니다.

```bash
hermes model
# Select "Custom endpoint (self-hosted / VLLM / etc.)"
# Enter URL: http://localhost:30000/v1
# Enter model name: meta-llama/Llama-3.1-70B-Instruct
```

**컨텍스트 길이:** SGLang은 기본적으로 모델 구성에서 값을 읽습니다. `--context-length`로 재정의하세요. 모델에 선언된 최대값을 초과해야 한다면 `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`을 설정합니다.

**도구 호출:** 모델 계열에 맞는 파서와 함께 `--tool-call-parser`를 사용합니다. `qwen`(Qwen 2.5), `llama3`, `llama4`, `deepseekv3`, `mistral`, `glm`을 사용할 수 있습니다. 이 플래그가 없으면 도구 호출이 일반 텍스트로 반환됩니다.

:::caution SGLang의 기본 최대 출력 토큰은 128개입니다
응답이 잘린 것처럼 보이면 요청에 `max_tokens`를 추가하거나 서버에서 `--default-max-tokens`를 설정하세요. 요청에 지정하지 않을 경우 SGLang의 응답당 기본값은 128토큰뿐입니다.
:::

---

### llama.cpp / llama-server — CPU 및 Metal 추론

[llama.cpp](https://github.com/ggml-org/llama.cpp)는 CPU, Apple Silicon(Metal) 및 소비자용 GPU에서 양자화된 모델을 실행합니다. 데이터센터급 GPU 없이 모델을 실행하거나, Mac을 사용하거나, 엣지에 배포할 때 적합합니다.

```bash
# Build and start llama-server
cmake -B build && cmake --build build --config Release
./build/bin/llama-server \
  --jinja -fa \
  -c 64000 \
  -ngl 99 \
  -m models/qwen2.5-coder-32b-instruct-Q4_K_M.gguf \
  --port 8080 --host 0.0.0.0
```

**컨텍스트 길이(`-c`):** 최근 빌드의 기본값은 `0`이며 GGUF 메타데이터에서 모델의 학습 컨텍스트를 읽습니다. 학습 컨텍스트가 128k 이상인 모델에서는 전체 KV 캐시를 할당하려다 OOM이 발생할 수 있습니다. Hermes에는 `-c`를 명시적으로 설정해 최소 64,000 토큰을 할당하세요. 병렬 슬롯(`-np`)을 사용하면 전체 컨텍스트가 슬롯 사이에 나뉩니다. `-c 64000 -np 4`인 경우 각 슬롯은 16k만 할당받아 활성 세션 하나에 필요한 Hermes 최소값보다 작습니다.

그런 다음 Hermes가 해당 서버를 가리키도록 구성합니다.

```bash
hermes model
# Select "Custom endpoint (self-hosted / VLLM / etc.)"
# Enter URL: http://localhost:8080/v1
# Skip API key (local servers don't need one)
# Enter model name — or leave blank to auto-detect if only one model is loaded
```

이 엔드포인트는 `config.yaml`에 저장되어 세션 간에 유지됩니다.

:::caution 도구 호출에는 `--jinja`가 필요합니다
`--jinja`가 없으면 llama-server는 `tools` 매개변수를 완전히 무시합니다. 모델은 응답 텍스트에 JSON으로 도구 호출을 작성하지만 Hermes는 이를 도구 호출로 인식하지 않습니다. 실제 검색 대신 `{"name": "web_search", ...}` 같은 원시 JSON이 메시지로 출력됩니다.

네이티브 도구 호출 지원(최상의 성능): Llama 3.x, Qwen 2.5(Coder 포함), Hermes 2/3, Mistral, DeepSeek, Functionary. 그 밖의 모델은 작동하지만 효율이 낮을 수 있는 일반 핸들러를 사용합니다. 전체 목록은 [llama.cpp 함수 호출 문서](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md)를 참고하세요.

`http://localhost:8080/props`를 확인해 도구 지원이 활성 상태인지 검증할 수 있습니다. `chat_template` 필드가 있어야 합니다.
:::

:::tip
[Hugging Face](https://huggingface.co/models?library=gguf)에서 GGUF 모델을 다운로드하세요. Q4_K_M 양자화는 품질과 메모리 사용량 사이에서 가장 균형이 좋습니다.
:::

---

### LM Studio — 로컬 모델을 사용하는 데스크톱 앱

[LM Studio](https://lmstudio.ai/)는 GUI로 로컬 모델을 실행하는 데스크톱 앱입니다. 시각적 인터페이스를 선호하는 사용자, 빠르게 모델을 테스트하는 사용자, macOS/Windows/Linux 개발자에게 적합합니다.

LM Studio 앱(Developer 탭 → Start Server)에서 서버를 시작하거나 CLI를 사용합니다.

```bash
lms server start                        # Starts on port 1234
lms load qwen2.5-coder --context-length 64000
```

그런 다음 Hermes를 구성합니다.

```bash
hermes model
# Select "LM Studio"
# Press Enter to use http://localhost:1234/v1
# Pick one of the discovered models
# If LM Studio server auth is enabled, enter LM_API_KEY when prompted
```

Hermes는 이미 로드된 LM Studio 인스턴스의 컨텍스트를 유지합니다. 기본 명시적 모드에서 로드되지 않은 모델의 경우 Hermes에 별도로 구성된 값이 없으면 `context_length`를 생략하므로 LM Studio가 자체 모델 설정을 적용할 수 있습니다. 그 후 Hermes는 로드가 끝난 뒤 LM Studio가 보고하는 컨텍스트 길이만 사용합니다.

LM Studio에서 컨텍스트 길이를 변경하려면 다음과 같이 합니다.

1. 모델 선택기 옆의 톱니바퀴 아이콘을 클릭합니다.
2. 원활한 사용을 위해 "Context Length"를 64000 이상으로 설정합니다.
3. 변경 사항을 적용하려면 모델을 다시 로드합니다.
4. 컴퓨터에서 64000을 수용할 수 없다면 더 큰 컨텍스트 길이를 지원하는 작은 모델을 사용해 보세요.

또는 CLI를 사용합니다: `lms load model-name --context-length 64000`

CLI로 모델이 적합한지 추정할 수 있습니다: `lms load model-name --context-length 64000 --estimate-only`

모델별 기본값을 영구적으로 설정하려면 My Models 탭 → 모델의 톱니바퀴 아이콘 → 컨텍스트 크기 설정으로 이동합니다.
:::

LM Studio의 Just-In-Time 로딩 / Auto-Evict 기능을 사용하고 일반 채팅 요청에서 LM Studio가 모델 로드와 제거를 관리하도록 하려면 Hermes의 명시적 사전 로드 단계를 건너뜁니다.

```bash
hermes config set model.lmstudio_load_mode jit
```

다음 명령으로 기본 명시적 사전 로드 동작으로 되돌립니다.

```bash
hermes config set model.lmstudio_load_mode explicit
```

**도구 호출:** LM Studio 0.3.6부터 지원됩니다. 네이티브 도구 호출 학습이 된 모델(Qwen 2.5, Llama 3.x, Mistral, Hermes)은 자동 감지되어 도구 배지와 함께 표시됩니다. 그 밖의 모델은 신뢰도가 낮을 수 있는 일반 폴백을 사용합니다.

---

### WSL2 네트워킹(Windows 사용자)

Hermes Agent는 Unix 환경이 필요하므로 Windows 사용자는 WSL2 내부에서 실행합니다. 모델 서버(Ollama, LM Studio 등)가 **Windows 호스트**에서 실행 중이면 네트워크 간격을 연결해야 합니다. WSL2는 자체 서브넷이 있는 가상 네트워크 어댑터를 사용하므로 WSL2 내부의 `localhost`는 Windows 호스트가 **아니라** Linux VM을 가리킵니다.

:::tip WSL2 내부에서 모두 실행하나요? 문제없습니다.
모델 서버도 WSL2 내부에서 실행한다면(vLLM, SGLang, llama-server에서 흔함) `localhost`가 예상대로 작동하며 동일한 네트워크 네임스페이스를 공유합니다. 이 섹션을 건너뛰세요.
:::

#### 옵션 1: 미러링 네트워킹 모드(권장)

**Windows 11 22H2 이상**에서 사용할 수 있습니다. 미러링 모드는 Windows와 WSL2 사이에서 `localhost`가 양방향으로 작동하게 하므로 가장 간단한 해결책입니다.

1. `%USERPROFILE%\.wslconfig`를 만들거나 편집합니다(예: `C:\Users\YourName\.wslconfig`).
   ```ini
   [wsl2]
   networkingMode=mirrored
   ```

2. PowerShell에서 WSL을 다시 시작합니다.
   ```powershell
   wsl --shutdown
   ```

3. WSL2 터미널을 다시 엽니다. 이제 `localhost`가 Windows 서비스에 연결됩니다.
   ```bash
   curl http://localhost:11434/v1/models   # Ollama on Windows — works
   ```

:::note Hyper-V 방화벽
일부 Windows 11 빌드에서는 Hyper-V 방화벽이 기본적으로 미러링 연결을 차단합니다. 미러링 모드를 활성화한 후에도 `localhost`가 작동하지 않으면 **관리자 PowerShell**에서 다음을 실행합니다.
```powershell
Set-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -DefaultInboundAction Allow
```
:::

#### 옵션 2: Windows 호스트 IP 사용(Windows 10 / 이전 빌드)

미러링 모드를 사용할 수 없다면 WSL2 내부에서 Windows 호스트 IP를 찾고 `localhost` 대신 사용합니다.

```bash
# Get the Windows host IP (the default gateway of WSL2's virtual network)
ip route show | grep -i default | awk '{ print $3 }'
# Example output: 172.29.192.1
```

Hermes 구성에서 해당 IP를 사용합니다.

```yaml
model:
  default: qwen2.5-coder:32b
  provider: custom
  base_url: http://172.29.192.1:11434/v1   # Windows host IP, not localhost
```

:::tip 동적 헬퍼
호스트 IP는 WSL2를 다시 시작할 때 변경될 수 있습니다. 셸에서 동적으로 가져올 수 있습니다.
```bash
export WSL_HOST=$(ip route show | grep -i default | awk '{ print $3 }')
echo "Windows host at: $WSL_HOST"
curl http://$WSL_HOST:11434/v1/models   # Test Ollama
```

또는 컴퓨터의 mDNS 이름을 사용합니다(WSL2에 `libnss-mdns` 필요).
```bash
sudo apt install libnss-mdns
curl http://$(hostname).local:11434/v1/models
```
:::

#### 서버 바인드 주소(NAT 모드에 필요)

\*\*옵션 2\*\*(호스트 IP를 사용하는 NAT 모드)를 사용하는 경우 Windows의 모델 서버가 `0.0.0.0`이 아닌 `127.0.0.1` 외부의 연결을 수락해야 합니다. 대부분의 서버는 기본적으로 localhost만 수신하므로 NAT 모드의 WSL2 연결은 다른 가상 서브넷에서 시작되어 거부됩니다. 미러링 모드에서는 `localhost`가 직접 매핑되므로 기본 `127.0.0.1` 바인딩으로 작동합니다.

| 서버 | 기본 바인드 | 해결 방법 |
|--------|-------------|------------|
| **Ollama** | `127.0.0.1` | Ollama를 시작하기 전에 `OLLAMA_HOST=0.0.0.0` 환경 변수를 설정합니다(Windows의 System Settings → Environment Variables 또는 Ollama 서비스 편집). |
| **LM Studio** | `127.0.0.1` | Developer 탭 → Server settings에서 **"Serve on Network"**를 활성화합니다. |
| **llama-server** | `127.0.0.1` | 시작 명령에 `--host 0.0.0.0`을 추가합니다. |
| **vLLM** | `0.0.0.0` | 기본적으로 모든 인터페이스에 이미 바인딩합니다. |
| **SGLang** | `127.0.0.1` | 시작 명령에 `--host 0.0.0.0`을 추가합니다. |

**Windows의 Ollama(자세한 내용):** Ollama는 Windows 서비스로 실행됩니다. `OLLAMA_HOST`를 설정하려면 다음 단계를 수행합니다.
1. **System Properties** → **Environment Variables**를 엽니다.
2. 새 **System variable**을 추가합니다: `OLLAMA_HOST` = `0.0.0.0`
3. Ollama 서비스를 다시 시작하거나 재부팅합니다.

#### Windows 방화벽

Windows 방화벽은 NAT 및 미러링 모드 모두에서 WSL2를 별도 네트워크로 취급합니다. 위 단계를 수행한 뒤에도 연결이 실패하면 모델 서버의 포트에 대한 방화벽 규칙을 추가합니다.

```powershell
# Run in Admin PowerShell — replace PORT with your server's port
New-NetFirewallRule -DisplayName "Allow WSL2 to Model Server" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 11434
```

일반적인 포트: Ollama `11434`, vLLM `8000`, SGLang `30000`, llama-server `8080`, LM Studio `1234`.

#### 빠른 검증

WSL2 내부에서 모델 서버에 연결할 수 있는지 테스트합니다.

```bash
# Replace URL with your server's address and port
curl http://localhost:11434/v1/models          # Mirrored mode
curl http://172.29.192.1:11434/v1/models       # NAT mode (use your actual host IP)
```

모델 목록을 반환하는 JSON 응답을 받으면 준비가 된 것입니다. Hermes 구성의 `base_url`에도 동일한 URL을 사용하세요.

---

### 로컬 모델 문제 해결

이 문제들은 Hermes에서 사용할 때 **모든** 로컬 추론 서버에 영향을 줍니다.

#### Windows 호스트에서 실행 중인 모델 서버에 WSL2가 연결할 때 "Connection refused"

Hermes를 WSL2 내부에서 실행하고 모델 서버를 Windows 호스트에서 실행하는 경우 WSL2의 기본 NAT 네트워킹 모드에서는 `http://localhost:<port>`가 작동하지 않습니다. 해결 방법은 위의 [WSL2 네트워킹](#wsl2-networking-windows-users)을 참고하세요.

#### 도구 호출이 실행되지 않고 텍스트로 표시됨

모델이 실제로 도구를 호출하는 대신 `{"name": "web_search", "arguments": {...}}` 같은 메시지를 출력합니다.

**원인:** 서버에서 도구 호출이 활성화되지 않았거나 모델이 서버의 도구 호출 구현을 통해 이를 지원하지 않습니다.

| 서버 | 해결 방법 |
|--------|-----|
| **llama.cpp** | 시작 명령에 `--jinja` 추가 |
| **vLLM** | `--enable-auto-tool-choice --tool-call-parser hermes` 추가 |
| **SGLang** | `--tool-call-parser qwen`(또는 적절한 파서) 추가 |
| **Ollama** | 기본적으로 도구 호출이 활성화됩니다. 모델이 이를 지원하는지 확인(`ollama show model-name`) |
| **LM Studio** | 0.3.6 이상으로 업데이트하고 네이티브 도구 지원 모델 사용 |

#### 모델이 컨텍스트를 잊거나 일관되지 않은 응답을 하는 것처럼 보임

**원인:** 컨텍스트 창이 너무 작습니다. 대화가 컨텍스트 한도를 초과하면 대부분의 서버가 이전 메시지를 조용히 삭제합니다. Hermes의 시스템 프롬프트와 도구 스키마만으로도 4k–8k 토큰을 사용할 수 있습니다.

**진단:**

```bash
# Check what Hermes thinks the context is
# Look at startup line: "Context limit: X tokens"

# Check your server's actual context
# Ollama: ollama ps (CONTEXT column)
# llama.cpp: curl http://localhost:8080/props | jq '.default_generation_settings.n_ctx'
# vLLM: check --max-model-len in startup args
```

**해결:** 에이전트 사용을 위해 컨텍스트를 최소 **64,000 토큰**으로 설정합니다. 구체적인 플래그는 각 서버의 위 섹션을 참고하세요.

#### 시작 시 "Context limit: 2048 tokens"

Hermes는 서버의 `/v1/models` 엔드포인트에서 컨텍스트 길이를 자동으로 감지합니다. 서버가 낮은 값을 보고하거나 전혀 보고하지 않으면 Hermes는 잘못되었을 수 있는 모델 선언 한도를 사용합니다.

**해결:** `config.yaml`에 명시적으로 설정합니다.

```yaml
model:
  default: your-model
  provider: custom
  base_url: http://localhost:11434/v1
  context_length: 64000
```

#### 응답이 문장 중간에서 잘림

**가능한 원인:**
1. **서버의 낮은 출력 한도(`max_tokens`)** — SGLang은 응답당 기본값이 128토큰입니다. 서버에서 `--default-max-tokens`를 설정하거나 `config.yaml`의 `model.max_tokens`로 Hermes를 구성하세요. 참고: `max_tokens`는 응답 길이만 제어하며 대화 기록의 길이와는 관계가 없습니다. 대화 기록의 길이는 `context_length`입니다.
2. **컨텍스트 소진** — 모델이 컨텍스트 창을 가득 채웠습니다. `model.context_length`를 늘리거나 Hermes에서 [컨텍스트 압축](/user-guide/configuration#context-compression)을 활성화하세요.

---

### LiteLLM Proxy — 다중 제공업체 게이트웨이

[LiteLLM](https://docs.litellm.ai/)은 100개 이상의 LLM 제공업체를 단일 API로 통합하는 OpenAI 호환 프록시입니다. 구성 변경 없이 제공업체를 전환하거나, 로드 밸런싱, 폴백 체인 및 예산 제어를 사용할 때 적합합니다.

```bash
# Install and start
pip install "litellm[proxy]"
litellm --model anthropic/claude-sonnet-4 --port 4000

# Or with a config file for multiple models:
litellm --config litellm_config.yaml --port 4000
```

그런 다음 `hermes model` → Custom endpoint → `http://localhost:4000/v1`로 Hermes를 구성합니다.

폴백을 포함한 `litellm_config.yaml` 예시:
```yaml
model_list:
  - model_name: "best"
    litellm_params:
      model: anthropic/claude-sonnet-4
      api_key: sk-ant-...
  - model_name: "best"
    litellm_params:
      model: openai/gpt-4o
      api_key: sk-...
router_settings:
  routing_strategy: "latency-based-routing"
```

---

### ClawRouter — 비용 최적화 라우팅

BlockRunAI의 [ClawRouter](https://github.com/BlockRunAI/ClawRouter)는 쿼리 복잡도에 따라 모델을 자동 선택하는 로컬 라우팅 프록시입니다. 14개 차원으로 요청을 분류하고 작업을 처리할 수 있는 가장 저렴한 모델로 라우팅합니다. 결제에는 USDC 암호화폐를 사용하며 API 키는 필요하지 않습니다.

```bash
# Install and start
npx @blockrun/clawrouter    # Starts on port 8402
```

그런 다음 `hermes model` → Custom endpoint → `http://localhost:8402/v1` → 모델 이름 `blockrun/auto`로 Hermes를 구성합니다.

라우팅 프로필:
| 프로필 | 전략 | 절약 |
|---------|----------|---------|
| `blockrun/auto` | 품질/비용 균형 | 74–100% |
| `blockrun/eco` | 가능한 가장 저렴하게 | 95–100% |
| `blockrun/premium` | 최고 품질 모델 | 0% |
| `blockrun/free` | 무료 모델만 | 100% |
| `blockrun/agentic` | 도구 사용에 최적화 | 다양함 |

:::note
ClawRouter는 결제를 위해 Base 또는 Solana에서 USDC가 충전된 지갑이 필요합니다. 모든 요청은 BlockRun의 백엔드 API를 통과합니다. 지갑 상태를 확인하려면 `npx @blockrun/clawrouter doctor`를 실행하세요.
:::

---

### 기타 호환 제공업체

OpenAI 호환 API가 있는 서비스라면 모두 작동합니다. 인기 있는 몇 가지 옵션은 다음과 같습니다.

| 제공업체 | 기본 URL | 참고 |
|----------|----------|-------|
| [Together AI](https://together.ai) | `https://api.together.xyz/v1` | 클라우드 호스팅 오픈 모델 |
| [Groq](https://groq.com) | `https://api.groq.com/openai/v1` | 초고속 추론 |
| [DeepSeek](https://deepseek.com) | `https://api.deepseek.com/v1` | DeepSeek 모델 |
| [Fireworks AI](https://fireworks.ai) | `https://api.fireworks.ai/inference/v1` | 빠른 오픈 모델 호스팅 |
| [GMI Cloud](https://www.gmicloud.ai/) | `https://api.gmi-serving.com/v1` | 관리형 OpenAI 호환 추론 |
| [Actual Computer](https://actual.inc) | `https://api.actual.inc/v1` | 자체 클러스터로 연결되는 비공개 릴레이; 로컬 데몬은 `http://127.0.0.1:8080/v1` |
| [Cerebras](https://cerebras.ai) | `https://api.cerebras.ai/v1` | 웨이퍼 규모 칩 추론 |
| [Mistral AI](https://mistral.ai) | `https://api.mistral.ai/v1` | Mistral 모델 |
| [OpenAI](https://openai.com) | `https://api.openai.com/v1` | 직접 OpenAI 접근 |
| [Azure OpenAI](https://azure.microsoft.com) | `https://YOUR.openai.azure.com/` | 엔터프라이즈 OpenAI |
| [LocalAI](https://localai.io) | `http://localhost:8080/v1` | 자체 호스팅, 다중 모델 |
| [Jan](https://jan.ai) | `http://localhost:1337/v1` | 로컬 모델을 사용하는 데스크톱 앱 |

다음과 같이 `hermes model` → Custom endpoint에서 구성하거나 `config.yaml`에 설정합니다.

```yaml
model:
  default: meta-llama/Llama-3.1-70B-Instruct-Turbo
  provider: custom
  base_url: https://api.together.xyz/v1
  api_key: your-together-key
```

---

### 컨텍스트 길이 감지

:::note 혼동하기 쉬운 두 설정
**`context_length`**는 **전체 컨텍스트 창**입니다. 즉 입력 토큰과 출력 토큰을 합친 총 예산입니다(예: Claude Opus 4.6의 경우 200,000). Hermes는 이 값을 사용해 기록을 언제 압축할지 결정하고 API 요청을 검증합니다.

**`model.max_tokens`**는 **출력 상한**입니다. 모델이 **단일 응답**에서 생성할 수 있는 최대 토큰 수입니다. 대화 기록이 얼마나 길어질 수 있는지와는 관련이 없습니다. 업계 표준 이름인 `max_tokens`는 혼동을 일으키는 흔한 원인입니다. Anthropic의 네이티브 API는 이후 명확성을 위해 이름을 `max_output_tokens`로 변경했습니다.

자동 감지가 컨텍스트 창 크기를 잘못 판단할 때는 `context_length`를 설정하세요.
개별 응답의 길이를 제한해야 할 때만 `model.max_tokens`를 설정하세요.
:::

Hermes는 여러 소스의 확인 절차를 사용해 모델과 제공업체에 맞는 컨텍스트 창을 감지합니다.

1. **설정 재정의** — config.yaml의 `model.context_length`(가장 높은 우선순위)
2. **사용자 지정 제공업체의 모델별 설정** — `providers.<name>.models.<id>.context_length`
3. **영구 캐시** — 이전에 감지한 값(재시작 후에도 유지)
4. **엔드포인트 `/models`** — 서버의 API를 조회(로컬/사용자 지정 엔드포인트)
5. **Anthropic `/v1/models`** — Anthropic API에서 `max_input_tokens`를 조회(API 키 사용자만 해당)
6. **OpenRouter API** — 모델 메타데이터를 실시간으로 조회
7. **Nous Portal** — Nous 모델 ID의 접미사를 OpenRouter 메타데이터와 비교
8. **[models.dev](https://models.dev)** — 100개가 넘는 제공업체의 3,800개 이상 모델에 대해 제공업체별 컨텍스트 길이를 제공하는 커뮤니티 관리 레지스트리
9. **대체 기본값** — 광범위한 모델 제품군 패턴(기본값 128K)

대부분의 설정에서는 별도 작업 없이 바로 작동합니다. 이 시스템은 제공업체를 고려하므로, 같은 모델이라도 어느 제공업체가 제공하느냐에 따라 컨텍스트 한도가 다를 수 있습니다(예: `claude-opus-4.6`은 Anthropic 직접 연결에서는 1M이지만 GitHub Copilot에서는 128K입니다).

컨텍스트 길이를 명시적으로 설정하려면 모델 설정에 `context_length`를 추가하세요.

```yaml
model:
  default: "qwen3.5:9b"
  base_url: "http://localhost:8080/v1"
  context_length: 131072  # tokens
```

사용자 지정 엔드포인트에서는 모델별로 컨텍스트 길이를 설정할 수도 있습니다.

```yaml
providers:
  my-local-llm:
    api: "http://localhost:11434/v1"
    models:
      qwen3.5:27b:
        context_length: 64000
      deepseek-r1:70b:
        context_length: 65536
```

사용자 지정 엔드포인트를 구성할 때 `hermes model`이 컨텍스트 길이를 묻습니다. 자동 감지를 사용하려면 비워 두세요.

:::tip 직접 설정해야 하는 경우
- 모델의 최대값보다 낮은 사용자 지정 `num_ctx`로 Ollama를 사용하는 경우
- 모델의 최대값보다 낮게 컨텍스트를 제한하려는 경우(VRAM을 절약하기 위해 128k 모델을 8k로 제한하는 등)
- `/v1/models`를 노출하지 않는 프록시 뒤에서 실행하는 경우
:::

---

### 이름이 지정된 사용자 지정 제공업체

여러 사용자 지정 엔드포인트(예: 로컬 개발 서버와 원격 GPU 서버)를 사용한다면 `config.yaml`의 `providers:` 딕셔너리 아래에 제공업체 이름을 키로 사용해 이름이 지정된 사용자 지정 제공업체를 정의할 수 있습니다.

```yaml
providers:
  local:
    api: http://localhost:8080/v1
    # api_key omitted — Hermes uses "no-key-required" for keyless local servers
  work:
    api: https://gpu-server.internal.corp/v1
    key_env: CORP_API_KEY
    transport: chat_completions   # set explicitly by `hermes model` → Custom Endpoint wizard; auto-detection still happens as a fallback
  anthropic-proxy:
    api: https://proxy.example.com/anthropic
    key_env: ANTHROPIC_PROXY_KEY
    transport: anthropic_messages  # for Anthropic-compatible proxies
```

각 항목은 다음을 지원합니다. `api`(엔드포인트 기본 URL — `base_url`/`url`도 별칭으로 허용), `name`(선택적 표시 이름; 기본값은 딕셔너리 키), `key_env` 또는 인라인 `api_key` 또는 `key_cmd`(아래 참조), `transport`(`chat_completions` / `anthropic_messages` / `codex_responses`), `default_model`, `models`, `context_length`, `discover_models`, `extra_body`, `extra_headers`, `ssl_ca_cert` / `ssl_verify`, 그리고 삭제하지 않고 항목을 숨기는 `enabled: false`.

#### 명령으로 생성하는 자격 증명(`key_cmd`)

엔터프라이즈 게이트웨이는 정적 API 키 대신 수명이 짧은 전달자 토큰(SSO/OIDC 브로커, 클라우드 IAM, 내부 인증 프록시)을 발급하는 경우가 많습니다. 따라서 `.env`에 복사한 토큰은 세션 중 만료되어 요청이 401을 반환하기 시작할 수 있습니다. `key_cmd`는 토큰을 *출력하는* 명령을 지정합니다. Hermes는 이 명령을 실행하고 만료 직전까지 결과를 캐시하므로 세션을 재시작하지 않아도 긴 세션을 계속 사용할 수 있습니다.

```yaml
providers:
  my-gateway:
    base_url: "https://gateway.internal.example.com/v1"
    api_mode: chat_completions
    key_cmd: "my-auth-cli print-token --profile prod"
```

토큰을 출력하는 도우미라면 무엇이든 사용할 수 있습니다. 예를 들면 `databricks auth token`, `gcloud auth print-access-token`, `az account get-access-token`, `vault read`, Claude Code 스타일의 `apiKeyHelper` 스크립트가 있습니다.

명령은 stdout에 **토큰만** 출력해야 합니다. 토큰을 그대로 출력하거나 `access_token` 필드를 포함한 JSON으로 출력할 수 있습니다(`expires_in`이 적용되며, 절대 시각인 `expiry`/`expiresOn` ISO 타임스탬프도 적용됩니다). 여러 줄 출력은 추측으로 처리하지 않고 거부합니다. 만료 정보가 없으면 제한된 시간 창에 따라 토큰을 다시 생성합니다.

우선순위는 다음과 같습니다. 명시적인 `--api-key` 플래그가 항상 우선하며, 그렇지 않으면 같은 항목의 정적 `api_key`/`key_env`보다 `key_cmd`가 우선합니다. 생성된 자격 증명은 주 에이전트 턴과 보조 작업(제목 생성, 압축, 비전, 임베딩)에 모두 적용됩니다.

`secrets.command`와 혼동하지 마세요. `secrets.command`는 **시작 시 한 번** 도우미를 실행해 프로세스 전체의 환경 변수를 채웁니다. 여러 시크릿을 반환하는 vault/keychain 도우미에는 이를 사용하고, 한 제공업체의 자격 증명을 세션 **중에** 다시 생성해야 할 때는 `key_cmd`를 사용하세요.

:::note 레거시 형식
이전 설정에서는 최상위 `custom_providers:` 목록을 사용했습니다. 이 형식도 계속 작동합니다. Hermes는 두 형식을 모두 읽으며, `hermes update`가 이를 `providers:` 딕셔너리로 자동 마이그레이션합니다(config v12). 딕셔너리 형식에서는 필드 이름이 조금 다릅니다. 레거시 `model`은 `default_model`이고 레거시 `api_mode`는 `transport`입니다.
:::

일부 OpenAI 호환 엔드포인트에는 제공업체별 요청 본문 필드가 필요합니다. 해당 사용자 지정 제공업체에 `extra_body` 맵을 추가하면 Hermes가 해당 엔드포인트의 각 chat-completions 요청에 이를 병합합니다.

```yaml
providers:
  gemma-local:
    api: http://localhost:8080/v1
    default_model: google/gemma-4-31b-it
    extra_body:
      enable_thinking: true
      reasoning_effort: high
```

서버가 문서에 명시한 형식을 사용하세요. 예를 들어 vLLM Gemma 배포와 일부 NVIDIA NIM 엔드포인트는 최상위 `extra_body` 필드가 아니라 `chat_template_kwargs` 아래의 `enable_thinking`을 요구합니다.

```yaml
extra_body:
  chat_template_kwargs:
    enable_thinking: true
```

vLLM에서 제공하는 Qwen 추론 모델의 경우, 추론 파서가 생성된 모든 텍스트를 추론 필드로 분리해 어시스턴트 `content`를 비워 두면 같은 형식을 사용해 추론을 비활성화할 수 있습니다.

```yaml
extra_body:
  chat_template_kwargs:
    enable_thinking: false
```

이제 `hermes model` → Custom Endpoint 마법사가 API 모드를 명시적으로 묻고 답변을 `config.yaml`에 저장합니다(제공업체 항목의 `transport`로 저장). 필드를 비워 두면 URL 기반 자동 감지(예: `/anthropic` 경로 → `anthropic_messages`)가 여전히 대체 절차로 실행됩니다.

**사용자 지정 제공업체 모델의 네이티브 비전.** 사용자 지정 엔드포인트가 models.dev에 없는 비전 지원 모델을 제공한다면 `model.supports_vision: true`를 설정하세요. 그러면 Hermes는 첨부된 이미지를 `vision_analyze`를 통해 사전 처리하는 대신 네이티브 방식(`image_url` 부분)으로 전달합니다. 설정은 하나면 충분하며 `agent.image_input_mode: native`도 설정할 필요가 없습니다.

```yaml
model:
  provider: custom
  base_url: http://localhost:8080/v1
  default: qwen3.6-35b-a3b
  supports_vision: true   # send images natively; otherwise vision_analyze pre-describes them
```

이 키는 이름이 지정된 제공업체의 모델(`providers.<name>.models.<id>.supports_vision`)에도 적용되며 표준 YAML 불리언(`true/false/yes/no/on/off/1/0`)을 지원합니다.

세 구문을 사용해 세션 중 제공업체를 전환할 수 있습니다.

```
/model custom:local:qwen-2.5       # Use the "local" endpoint with qwen-2.5
/model custom:work:llama3-70b      # Use the "work" endpoint with llama3-70b
/model custom:anthropic-proxy:claude-sonnet-4  # Use the proxy
```

대화형 `hermes model` 메뉴에서 이름이 지정된 사용자 지정 제공업체를 선택할 수도 있습니다.

---

### 함께 보기: Together AI, Groq, Perplexity

[기타 호환 제공업체](#other-compatible-providers)에 나열된 클라우드 제공업체는 모두 OpenAI의 REST 방언을 사용하므로 `providers:` 딕셔너리에서 같은 방식으로 연결됩니다. 다음은 작동하는 세 가지 레시피입니다. 각각 `~/.hermes/config.yaml`에 추가하고, 일치하는 API 키를 `~/.hermes/.env`에 넣으면 됩니다.

#### Together AI

Llama, MiniMax, Gemma, DeepSeek, Qwen 등 오픈 웨이트 모델을 퍼스트파티 API보다 훨씬 저렴한 가격에 호스팅합니다. 여러 모델을 운영할 때 좋은 기본 선택입니다.

```yaml
# ~/.hermes/config.yaml
providers:
  together:
    api: https://api.together.xyz/v1
    key_env: TOGETHER_API_KEY
    # transport: chat_completions  # default — no need to set

model:
  default: MiniMaxAI/MiniMax-M2.7   # or any model from together.ai/models
  provider: custom:together
```

```bash
# ~/.hermes/.env
TOGETHER_API_KEY=your-together-key
```

세션 중 모델을 전환합니다.

```
/model custom:together:meta-llama/Llama-3.3-70B-Instruct-Turbo
/model custom:together:google/gemma-4-31b-it
/model custom:together:deepseek-ai/DeepSeek-V3
```

Together의 `/v1/models` 엔드포인트가 작동하므로 `hermes model`이 사용 가능한 모델을 자동으로 검색할 수 있습니다.

#### Groq

초고속 추론을 제공합니다(Llama-3.3-70B에서 약 500 tok/s). 카탈로그는 작지만 지연 시간에 민감한 대화형 사용에 강합니다.

```yaml
# ~/.hermes/config.yaml
providers:
  groq:
    api: https://api.groq.com/openai/v1
    key_env: GROQ_API_KEY

model:
  default: llama-3.3-70b-versatile
  provider: custom:groq
```

```bash
# ~/.hermes/.env
GROQ_API_KEY=your-groq-key
```

#### Perplexity

실시간 웹 검색과 인용을 자동으로 수행하는 모델이 필요할 때 유용합니다. 사용 가능한 모델이 엄격하게 제한되므로 현재 목록은 [perplexity.ai/settings/api](https://www.perplexity.ai/settings/api)에서 확인하세요.

```yaml
# ~/.hermes/config.yaml
providers:
  perplexity:
    api: https://api.perplexity.ai
    key_env: PERPLEXITY_API_KEY

model:
  default: sonar
  provider: custom:perplexity
```

```bash
# ~/.hermes/.env
PERPLEXITY_API_KEY=your-perplexity-key
```

#### 하나의 설정에서 여러 제공업체 사용

세 레시피는 함께 사용할 수 있습니다. 모두 사용한 다음 `/model custom:<name>:<model>`로 턴마다 전환하세요.

```yaml
providers:
  together:
    api: https://api.together.xyz/v1
    key_env: TOGETHER_API_KEY
  groq:
    api: https://api.groq.com/openai/v1
    key_env: GROQ_API_KEY
  perplexity:
    api: https://api.perplexity.ai
    key_env: PERPLEXITY_API_KEY

model:
  default: MiniMaxAI/MiniMax-M2.7
  provider: custom:together      # boot to Together; switch freely after
```

:::tip 문제 해결
- CLI 검증기가 수정된 #15083 이후 `hermes doctor`를 실행하면 이 이름들에 대해 `Unknown provider` 경고가 출력되지 않아야 합니다.
- 제공업체의 `/v1/models` 엔드포인트에 연결할 수 없으면(Perplexity가 대표적) `hermes model`은 모델을 완전히 거부하는 대신 경고와 함께 저장합니다(#15136 참조).
- 이름이 지정된 제공업체를 모두 건너뛰고 `CUSTOM_BASE_URL` 환경 변수와 함께 일반 `provider: custom`을 사용하려면 #15103을 참조하세요.
:::

---

### 적절한 설정 선택

| 사용 사례 | 권장 설정 |
|----------|-------------|
| **일단 작동하면 좋겠음** | OpenRouter(기본값) 또는 Nous Portal |
| **간편하게 설정하는 로컬 모델** | Ollama |
| **프로덕션 GPU 서빙** | vLLM 또는 SGLang |
| **Mac / GPU 없음** | Ollama 또는 llama.cpp |
| **여러 제공업체 라우팅** | LiteLLM Proxy 또는 OpenRouter |
| **비용 최적화** | ClawRouter 또는 `sort: "price"`를 사용한 OpenRouter |
| **최대 개인정보 보호** | Ollama, vLLM 또는 llama.cpp(완전 로컬) |
| **엔터프라이즈 / Azure** | 사용자 지정 엔드포인트를 사용하는 Azure OpenAI |
| **중국 AI 모델** | z.ai(GLM), Kimi/Moonshot(`kimi-coding` 또는 `kimi-coding-cn`), MiniMax, Xiaomi MiMo 또는 Tencent TokenHub(일급 제공업체) |

:::tip
`hermes model`로 언제든 제공업체를 전환할 수 있으며 재시작할 필요가 없습니다. 어떤 제공업체를 사용하든 대화 기록, 메모리, 스킬은 유지됩니다.
:::

## 선택적 API 키

| 기능 | 제공업체 | 환경 변수 |
|---------|----------|--------------|
| 웹 스크래핑 | [Firecrawl](https://firecrawl.dev/) | `FIRECRAWL_API_KEY`, `FIRECRAWL_API_URL` |
| 브라우저 자동화 | [Browserbase](https://browserbase.com/) | `BROWSERBASE_API_KEY`, `BROWSERBASE_PROJECT_ID` |
| 이미지 생성 | [FAL](https://fal.ai/) | `FAL_KEY` |
| 프리미엄 TTS 음성 | [ElevenLabs](https://elevenlabs.io/) | `ELEVENLABS_API_KEY` |
| OpenAI TTS + 음성 전사 | [OpenAI](https://platform.openai.com/api-keys) | `VOICE_TOOLS_OPENAI_KEY` |
| Mistral TTS + 음성 전사 | [Mistral](https://console.mistral.ai/) | `MISTRAL_API_KEY` |
| 세션 간 사용자 모델링 | [Honcho](https://honcho.dev/) | `HONCHO_API_KEY` |
| 시맨틱 장기 메모리 | [Supermemory](https://supermemory.ai) | `SUPERMEMORY_API_KEY` |

### Firecrawl 자체 호스팅

기본적으로 Hermes는 웹 검색과 스크래핑에 [Firecrawl 클라우드 API](https://firecrawl.dev/)를 사용합니다. Firecrawl을 로컬에서 실행하려면 Hermes가 자체 호스팅 인스턴스를 가리키도록 설정할 수 있습니다. 전체 설정 방법은 Firecrawl의 [SELF_HOST.md](https://github.com/firecrawl/firecrawl/blob/main/SELF_HOST.md)를 참조하세요.

**얻는 것:** API 키가 필요 없고, 속도 제한과 페이지별 비용이 없으며, 데이터를 완전히 직접 관리할 수 있습니다.

**잃는 것:** 클라우드 버전은 고급 안티봇 우회(Cloudflare, CAPTCHA, IP 순환)를 위해 Firecrawl의 독점적인 "Fire-engine"을 사용합니다. 자체 호스팅 버전은 기본 fetch + Playwright를 사용하므로 일부 보호된 사이트가 실패할 수 있습니다. 검색에는 Google 대신 DuckDuckGo를 사용합니다.

**설정:**

1. Firecrawl Docker 스택을 복제하고 시작합니다(컨테이너 5개: API, Playwright, Redis, RabbitMQ, PostgreSQL — 약 4~8GB RAM 필요).
   ```bash
   git clone https://github.com/firecrawl/firecrawl
   cd firecrawl
   # In .env, set: USE_DB_AUTHENTICATION=false, HOST=0.0.0.0, PORT=3002
   docker compose up -d
   ```

2. Hermes가 인스턴스를 가리키도록 설정합니다(API 키는 필요하지 않음).
   ```bash
   hermes config set FIRECRAWL_API_URL http://localhost:3002
   ```

자체 호스팅 인스턴스에서 인증을 활성화했다면 `FIRECRAWL_API_KEY`와 `FIRECRAWL_API_URL`을 모두 설정할 수도 있습니다.

## OpenRouter 제공업체 라우팅

OpenRouter를 사용할 때 제공업체 간 요청 라우팅 방식을 제어할 수 있습니다. `~/.hermes/config.yaml`에 `provider_routing` 섹션을 추가하세요.

```yaml
provider_routing:
  sort: "throughput"          # "price" (default), "throughput", or "latency"
  # only: ["anthropic"]      # Only use these providers
  # ignore: ["deepinfra"]    # Skip these providers
  # order: ["anthropic", "google"]  # Try providers in this order
  # require_parameters: true  # Only use providers that support all request params
  # data_collection: "deny"   # Exclude providers that may store/train on data
```

**단축 구문:** 모델 이름 뒤에 `:nitro`를 붙이면 처리량 기준으로 정렬합니다(예: `anthropic/claude-sonnet-4:nitro`). `:floor`를 붙이면 가격 기준으로 정렬합니다.

## OpenRouter Pareto Code 라우터

OpenRouter에는 코딩 품질 기준([Artificial Analysis](https://artificialanalysis.ai/) 순위)을 충족하는 가장 저렴한 모델로 요청을 자동 라우팅하는 실험적 코딩 모델 라우터 `openrouter/pareto-code`가 있습니다. 이 모델을 선택하고 `~/.hermes/config.yaml`에서 `min_coding_score` 설정을 조정하세요.

```yaml
model:
  provider: openrouter
  model: openrouter/pareto-code

openrouter:
  min_coding_score: 0.65   # 0.0–1.0; higher = stronger (more expensive) coders. Default 0.65.
```

참고:

- `min_coding_score`는 `model.model`이 `openrouter/pareto-code`일 때만 전송됩니다. 다른 모델에서는 이 값이 적용되지 않습니다.
- OpenRouter가 사용 가능한 가장 강력한 코더를 선택하게 하려면 빈 문자열로 설정하거나 이 줄을 삭제하세요. 플러그인 블록을 생략하면 이것이 문서화된 동작입니다.
- 특정 날짜의 점수에 따른 선택은 결정적이지만, Pareto 프런티어가 이동하면 실제 선택 모델은 바뀔 수 있습니다(새 모델, 벤치마크 업데이트).
- 전체 라우터 동작은 OpenRouter의 [Pareto Router 문서](https://openrouter.ai/docs/guides/routing/routers/pareto-router)를 참조하세요.
- 주 에이전트 대신 특정 **보조 작업**(압축, 비전 등)에 Pareto Code 라우터를 사용하려면 해당 작업 아래에 `extra_body.plugins`를 설정하세요. [보조 모델 → 보조 작업을 위한 OpenRouter 라우팅 및 Pareto Code](/user-guide/configuration#openrouter-routing--pareto-code-for-auxiliary-tasks)를 참조하세요.

## 대체 제공업체

주 모델이 실패할 때(속도 제한, 서버 오류, 인증 실패) Hermes가 순서대로 시도할 백업 제공업체 체인을 설정할 수 있습니다. 표준 형식은 최상위 `fallback_providers:` 목록입니다.

```yaml
fallback_providers:
  - provider: openrouter
    model: anthropic/claude-sonnet-4
  - provider: anthropic
    model: claude-sonnet-4
    # base_url: http://localhost:8000/v1    # optional, for custom endpoints
    # api_mode: chat_completions           # optional override
```

이전 버전과의 호환성을 위해 레거시 단일 쌍 `fallback_model:` 딕셔너리도 허용됩니다.

```yaml
fallback_model:
  provider: openrouter
  model: anthropic/claude-sonnet-4
```

활성화되면 대체 제공업체가 대화 기록을 잃지 않고 세션 중 모델과 제공업체를 교체합니다. 체인은 항목별로 시도되며 세션당 한 번만 활성화됩니다.

지원되는 제공업체: `openrouter`, `nous`, `novita`, `openai-codex`, `copilot`, `copilot-acp`, `anthropic`, `gemini`, `qwen-oauth`, `huggingface`, `zai`, `kimi-coding`, `kimi-coding-cn`, `minimax`, `minimax-cn`, `minimax-oauth`, `deepseek`, `nvidia`, `xai`, `xai-oauth`, `ollama-cloud`, `bedrock`, `ai-gateway`, `azure-foundry`, `opencode-zen`, `opencode-go`, `kilocode`, `xiaomi`, `arcee`, `gmi`, `actual`, `stepfun`, `lmstudio`, `alibaba`, `alibaba-coding-plan`, `tencent-tokenhub`, `custom`.

:::tip
대체 제공업체는 `config.yaml`에서만 설정하거나 `hermes fallback`을 통해 대화형으로 설정할 수 있습니다. 언제 작동하는지, 체인이 어떻게 진행되는지, 보조 작업 및 위임과 어떻게 상호작용하는지에 대한 자세한 내용은 [대체 제공업체](/user-guide/features/fallback-providers)를 참조하세요.
:::

## 함께 보기

- [설정](/user-guide/configuration) — 일반 설정(디렉터리 구조, 설정 우선순위, 터미널 백엔드, 메모리, 압축 등)
- [환경 변수](/reference/environment-variables) — 모든 환경 변수의 전체 참조
