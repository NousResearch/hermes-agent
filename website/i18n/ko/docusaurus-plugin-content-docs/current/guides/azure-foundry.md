---
sidebar_position: 15
title: "Microsoft Foundry"
description: "Microsoft Foundry에서 Hermes Agent 사용하기 — OpenAI 스타일 및 Anthropic 스타일 엔드포인트, 전송 방식과 배포된 모델 자동 감지"
---

# Microsoft Foundry

Hermes Agent의 `azure-foundry` provider는 Microsoft Foundry(이전 명칭 Azure AI Foundry)와 Azure OpenAI를 지원합니다. 하나의 Foundry 리소스에서 서로 다른 두 가지 와이어 형식의 모델을 호스팅할 수 있습니다.

- **OpenAI 스타일** — `https://<resource>.openai.azure.com/openai/v1`와 같은 엔드포인트에서 `POST /v1/chat/completions`를 사용합니다. GPT-4.x, GPT-5.x, Llama, Mistral 및 대부분의 오픈 웨이트 모델에 사용됩니다.
- **Anthropic 스타일** — `https://<resource>.services.ai.azure.com/anthropic`와 같은 엔드포인트에서 `POST /v1/messages`를 사용합니다. Microsoft Foundry가 Anthropic Messages API 형식으로 Claude 모델을 제공할 때 사용됩니다.

설정 마법사는 엔드포인트를 조사하여 어떤 전송 방식을 사용하는지, 어떤 배포를 사용할 수 있는지, 각 모델의 컨텍스트 길이가 얼마인지 자동으로 감지합니다.

## 사전 요구 사항

- 배포가 하나 이상 있는 Microsoft Foundry 또는 Azure OpenAI 리소스
- 배포의 엔드포인트 URL
- **API 키**(Azure Portal의 "Keys and Endpoint"에서 가져옴) **또는** Microsoft Entra ID를 사용하려는 경우 Foundry 리소스에 대한 **Azure AI User** RBAC 역할(키 없이 사용하는 방식으로 Microsoft가 권장함). Microsoft의 역할 이름 변경이 진행되는 동안 일부 테넌트에서는 이 역할이 **Foundry User**로 표시될 수 있습니다.

## 빠른 시작

```bash
hermes model
# → Select "Azure Foundry"
# → Enter your endpoint URL
# → Choose Authentication:
#     1. API key
#     2. Microsoft Entra ID  (managed identity / workload identity / az login)
# → (Entra) Hermes probes DefaultAzureCredential; on success it never asks for a key
# → (API key) Enter your API key
# Hermes probes the endpoint and auto-detects transport + models
# → Pick a model from the list (or type a deployment name manually)
```

마법사는 다음 작업을 수행합니다.

1. **URL 경로 조사** — `/anthropic`으로 끝나는 URL은 Microsoft Foundry Claude 경로로 인식됩니다.
2. **`GET <base>/models` 조사** — 엔드포인트가 OpenAI 형식의 모델 목록을 반환하면 Hermes는 `chat_completions`로 전환하고 반환된 배포 ID로 선택 목록을 미리 채웁니다.
3. **Anthropic Messages 형식 조사** — `/models`를 제공하지 않지만 Anthropic Messages 형식을 허용하는 엔드포인트를 위한 대체 경로입니다.
4. **수동 입력으로 대체** — 모든 조사를 거부하는 비공개/게이트 엔드포인트도 계속 사용할 수 있습니다. API 모드를 선택하고 배포 이름을 직접 입력하면 됩니다.

선택한 모델의 컨텍스트 길이는 Hermes의 표준 메타데이터 체인(`models.dev`, provider 메타데이터 및 하드코딩된 제품군 대체값)을 통해 확인되며, 모델이 자체 컨텍스트 창의 크기를 올바르게 조정할 수 있도록 `config.yaml`에 저장됩니다.

## Microsoft Entra ID(키 없음, RBAC) — 권장

Microsoft는 프로덕션 Foundry 워크로드에 [Microsoft Entra ID를 사용한 키 없는 인증](https://learn.microsoft.com/azure/ai-foundry/foundry-models/how-to/configure-entra-id)을 권장합니다. Hermes는 **두 API 표면 모두**에서 Entra ID를 지원합니다.

- **OpenAI 스타일**(`api_mode: chat_completions` / `codex_responses`) — GPT-4/5, Llama, Mistral, DeepSeek 등
- **Anthropic 스타일**(`api_mode: anthropic_messages`) — Microsoft Foundry의 Claude 모델

Foundry의 RBAC는 리소스별로 적용되며(`Azure AI User`는 두 표면 모두에 권한을 부여하고, 일부 테넌트에서는 `Foundry User`로 표시될 수 있음), Microsoft는 두 표면에 동일한 추론 범위(`https://ai.azure.com/.default`)를 문서화합니다. 내부적으로는 다음과 같이 동작합니다.

- OpenAI 스타일은 OpenAI Python SDK의 기본 callable `api_key=` 계약을 사용하며, SDK가 요청마다 새로운 JWT를 자동으로 발급합니다.
- Anthropic 스타일은 `agent.azure_identity_adapter.build_bearer_http_client`가 설치하는 요청 이벤트 훅과 함께 `httpx.Client`를 사용합니다. Anthropic SDK는 기본적으로 callable `auth_token`을 허용하지 않기 때문입니다. 이 훅은 모든 아웃바운드 요청마다 `Authorization: Bearer <fresh-jwt>`를 다시 작성합니다. 동일한 Microsoft RBAC, 동일한 Foundry 범위이며 SDK 계약만 다릅니다.

### Entra ID를 사용하는 이유

- 교체하거나 폐기해야 할 장기 API 키가 없습니다.
- RBAC 기반 액세스 — 설정을 다시 작성하지 않고 Foundry 리소스에서 `Azure AI User`를 부여하거나 제거할 수 있습니다.
- 모든 호출자가 하나의 정적 키를 공유하는 대신 담당자별로 액세스 및 감사 로그를 분리할 수 있습니다.
- Azure VM, AKS Pod, App Service, Functions, Container Apps 및 관리 ID를 통한 Foundry Agent Service를 위한 단일 인증 표면입니다.
- CI/CD 파이프라인을 위한 워크로드 ID 및 서비스 주체 흐름을 지원합니다.

### 일회성 설정(Azure 측)

1. Azure Portal에서 Foundry 리소스를 열고 → **Access control (IAM)** → **Add → Add role assignment**로 이동합니다.
2. **Azure AI User** 역할을 선택합니다(테넌트에서 이름이 변경된 역할을 사용하는 경우 **Foundry User**).
3. 다음 대상에 역할을 할당합니다.
   - `az login`을 사용한 로컬 개발을 위한 **사용자 계정**
   - Azure 호스팅 컴퓨팅을 위한 **관리 ID 또는 워크로드 ID**(프로덕션에 권장)
   - Hermes가 호스팅된 에이전트 내부에서 실행될 때 **Foundry Agent Service 호스팅 에이전트의 에이전트 ID**
   - 워크로드 ID를 사용할 수 없는 CI/CD 파이프라인을 위한 **서비스 주체**
4. 역할이 전파될 때까지 약 5분 기다립니다.

Azure CLI equivalent:

```bash
az role assignment create \
  --assignee <principal-or-agent-identity-client-id> \
  --role "Azure AI User" \
  --scope <foundry-resource-id>
```

### 일회성 설정(Hermes 측)

```bash
hermes model
# → Select "Azure Foundry"
# → Enter your endpoint URL
# → Authentication: 2 (Microsoft Entra ID)
# → (optional) user-assigned managed identity client ID
# → (optional) Azure tenant ID
# → Hermes probes DefaultAzureCredential() and reports which inner
#    credential succeeded (e.g. AzureCliCredential, ManagedIdentityCredential)
```

마법사는 제한 시간 10초의 제한된 사전 검사를 실행합니다. 실패하면 "save anyway, validate later"를 선택할 수 있습니다. 이는 아직 런타임 자격 증명이 없지만 이후 실행 시 자격 증명이 제공될 컴퓨터(예: 관리 ID 배포용 설정을 준비하는 경우)에서 설정할 때 유용합니다.

`azure-identity`는 Hermes의 지연 설치 경로를 통해 처음 사용할 때 자동으로 설치됩니다. 미리 설치하려면 다음을 실행합니다.

```bash
pip install azure-identity
```

### `config.yaml`에 기록되는 설정

```yaml
model:
  provider: azure-foundry
  base_url: https://my-resource.openai.azure.com/openai/v1
  api_mode: chat_completions
  auth_mode: entra_id
  default: gpt-4o
  context_length: 128000
  entra:
    scope: https://ai.azure.com/.default        # only when overriding the default
```

Hermes는 `config.yaml`에서 Entra 전용 설정 하나만 관리합니다.

- **`scope`** — OAuth 리소스 범위입니다. Microsoft가 문서화한 기본 추론 범위(`https://ai.azure.com/.default`)가 기본값입니다. 리소스가 비표준 대상에 대해 프로비저닝된 경우에만 재정의합니다.

그 외 모든 항목(테넌트, 서비스 주체 보안 비밀, 페더레이션 토큰 파일, 소버린 클라우드 권한, 브로커 기본 설정)은 `azure-identity`가 표준 `AZURE_*` 환경 변수에서 직접 읽습니다. 아래의 [자격 증명 확인 순서](#credential-resolution-order)를 참조하세요. Microsoft의 SDK 참조에 설명된 대로 해당 값을 `~/.hermes/.env` 또는 배포 환경에 설정합니다.

Entra 모드에서는 `~/.hermes/.env`에 비밀이 저장되지 않습니다. `azure-identity`가 프로세스 내에서 토큰을 캐시하며(가능한 경우 OS 키체인 / `~/.IdentityService`에도 저장), Hermes는 이를 사용합니다.

### 자격 증명 확인 순서

`azure-identity`의 `DefaultAzureCredential`은 각 토큰 요청에서 다음 체인을 순서대로 확인하며, 토큰을 반환하는 첫 번째 자격 증명에서 멈춥니다.

1. **환경 자격 증명** — `AZURE_TENANT_ID` + `AZURE_CLIENT_ID` + `AZURE_CLIENT_SECRET`(또는 `AZURE_CLIENT_CERTIFICATE_PATH` / `AZURE_FEDERATED_TOKEN_FILE`).
2. **워크로드 ID** — `AZURE_FEDERATED_TOKEN_FILE`(AKS 페더레이션 토큰 / OIDC).
3. **관리 ID** — 가상 머신의 IMDS 엔드포인트(`169.254.169.254`), App Service / Functions / Container Apps의 `IDENTITY_ENDPOINT`. Foundry Agent Service 호스팅 에이전트는 호스팅 에이전트의 에이전트 ID를 사용합니다.
4. **Visual Studio Code** — Azure 계정 확장.
5. **Azure CLI** — `az login` 세션.
6. **Azure Developer CLI** — `azd auth login`.
7. **Azure PowerShell** — `Connect-AzAccount`.
8. **브로커**(Windows / WSL만 해당) — Web Account Manager.

대화형 브라우저 자격 증명은 자동화된 Hermes 실행에서 기본적으로 제외됩니다. 대신 Azure CLI, Azure Developer CLI, 관리 ID, 워크로드 ID 또는 서비스 주체 자격 증명을 사용하세요.

### 배포 패턴

**로컬 개발:**
```bash
az login
hermes model   # pick Azure Foundry → Entra ID
hermes         # uses your az login token
```

**Azure VM / Functions / App Service / Container Apps(시스템 할당 관리 ID):**
1. 컴퓨팅 리소스에서 시스템 할당 ID를 활성화합니다.
2. Foundry 리소스에서 해당 ID에 `Azure AI User`(또는 `Foundry User`)를 부여합니다.
3. config.yaml에서 `model.auth_mode: entra_id`를 설정합니다. 환경 변수가 필요하지 않습니다.

**Azure VM / Functions / App Service / Container Apps(사용자 할당 관리 ID):**
- `AZURE_CLIENT_ID`를 사용자 할당 ID의 클라이언트 ID로 설정하여 `DefaultAzureCredential`이 올바른 ID를 선택하도록 합니다.

**Foundry Agent Service 호스팅 에이전트:**
- 호스팅 에이전트를 만들고 해당 에이전트의 ID에 Foundry 리소스의 `Azure AI User`(또는 `Foundry User`)를 부여합니다. Hermes는 호스팅 에이전트 내부에서 `ManagedIdentityCredential`을 사용합니다. 역할 할당은 상위 프로젝트나 사용자가 아니라 에이전트 ID에 해야 합니다.

**AKS 워크로드 ID(AAD Pod Identity 대체):**
- Pod의 서비스 계정에 워크로드 ID 클라이언트 ID를 주석으로 추가합니다.
- Pod의 페더레이션 토큰 파일은 `AZURE_FEDERATED_TOKEN_FILE`을 통해 자동으로 감지됩니다.
- 추가 설정 변경 없이 `model.auth_mode: entra_id`가 작동합니다.

**CI의 서비스 주체:**
- 실행기 환경에서 `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`을 설정합니다.

#### 소버린 클라우드(정부, 중국)

`AZURE_AUTHORITY_HOST`를 내보냅니다(예: Azure Government의 경우 `https://login.microsoftonline.us`, Azure China의 경우 `https://login.partner.microsoftonline.cn`). `azure-identity`가 이를 직접 읽습니다.

### 상태 확인

`hermes doctor`는 `model.auth_mode: entra_id`일 때 `DefaultAzureCredential`에 대해 10초간 조사를 실행하고, 어떤 내부 자격 증명이 성공했는지 보고합니다(환경 변수가 존재하는지, 관리 ID 엔드포인트에 연결할 수 있는지 등).

`hermes auth`는 구조화된 상태 블록을 표시합니다.

```
azure-foundry (Microsoft Entra ID):
  Endpoint: https://my-resource.openai.azure.com/openai/v1
  Scope: https://ai.azure.com/.default
  Status: configured; live token probe is skipped here
```

### 제한 사항

- **Anthropic 스타일 엔드포인트는 httpx 이벤트 훅을 사용합니다.** Anthropic Python SDK는 callable `auth_token`을 기본적으로 허용하지 않습니다(≤ 0.86.0). Hermes는 사용자 지정 `httpx.Client`에 요청 이벤트 훅을 설치하여 모든 아웃바운드 요청마다 새로운 JWT를 발급하고 `Authorization: Bearer <jwt>`를 다시 작성합니다. 이는 OpenAI SDK의 기본 `Callable[[], str]` 계약과 기능적으로 동일하지만 한 단계의 간접 계층이 추가됩니다. 향후 Anthropic SDK가 일급 callable 인증 지원을 추가하면 Hermes는 투명하게 해당 방식으로 전환합니다.
- **배치 작업 및 `multiprocessing.Pool`.** Entra 토큰 provider는 프로세스 경계를 넘어 피클링할 수 없는 클로저입니다. `batch_runner.py`는 작업자 설정에서 callable을 자동으로 제거하고 각 작업자 프로세스가 `config.yaml`에서 자체 provider를 다시 구성하도록 합니다. 사용자 작업은 필요하지 않지만 각 작업자는 시작 시 체인 확인 비용을 한 번씩 부담합니다.
- **`auth.json`에 bearer JWT를 저장하지 않습니다.** Hermes는 `azure-identity`의 내부 토큰 캐시를 복제하지 않습니다. 콜드 스타트 시 첫 추론에서 자격 증명 체인을 확인합니다.

## 설정(`config.yaml`에 기록됨)

마법사를 실행한 후 다음과 같은 내용을 볼 수 있습니다.

```yaml
model:
  provider: azure-foundry
  base_url: https://my-resource.openai.azure.com/openai/v1
  api_mode: chat_completions         # or "anthropic_messages"
  default: gpt-5.4-mini              # your deployment / model name
  context_length: 400000             # auto-detected
```

그리고 `~/.hermes/.env`에는 다음이 들어갑니다.

```
AZURE_FOUNDRY_API_KEY=<your-azure-key>
```

## OpenAI 스타일 엔드포인트(GPT, Llama 등)

Azure OpenAI의 v1 GA 엔드포인트는 약간의 변경만으로 표준 `openai` Python 클라이언트를 사용할 수 있습니다.

```yaml
model:
  provider: azure-foundry
  base_url: https://my-resource.openai.azure.com/openai/v1
  api_mode: chat_completions
  default: gpt-5.4
```

중요한 동작:

- **GPT-5.x, codex 및 o-series는 Responses API로 자동 라우팅됩니다.** Microsoft Foundry는 GPT-5 / codex / o1 / o3 / o4 모델을 Responses API 전용으로 배포합니다. 이러한 모델에 `/chat/completions`를 호출하면 `400 "The requested operation is unsupported."`가 반환됩니다. Hermes는 이름으로 이 모델 제품군을 감지하고 `config.yaml`에 여전히 `api_mode: chat_completions`가 적혀 있더라도 `api_mode`를 `codex_responses`로 투명하게 업그레이드합니다. GPT-4, GPT-4o, Llama, Mistral 및 기타 배포는 `/chat/completions`를 계속 사용합니다.
- **`max_completion_tokens`이 자동으로 사용됩니다.** Azure OpenAI는(직접 OpenAI와 마찬가지로) gpt-4o, o-series 및 gpt-5.x 모델에 `max_completion_tokens`를 요구합니다. Hermes는 엔드포인트에 따라 올바른 매개변수를 전송합니다.
- **`api-version`이 필요한 v1 이전 엔드포인트.** `https://<resource>.openai.azure.com/openai?api-version=2025-04-01-preview`와 같은 레거시 기본 URL을 사용하는 경우 Hermes는 쿼리 문자열을 추출하고 모든 요청에서 `default_query`를 통해 전달합니다(OpenAI SDK는 경로를 결합할 때 그렇지 않으면 이를 삭제합니다).

## Anthropic 스타일 엔드포인트(Microsoft Foundry를 통한 Claude)

Claude 배포에는 Anthropic 스타일 경로를 사용합니다.

```yaml
model:
  provider: azure-foundry
  base_url: https://my-resource.services.ai.azure.com/anthropic
  api_mode: anthropic_messages
  default: claude-sonnet-4-6
```

중요한 동작:

- **기본 URL에서 `/v1`이 제거됩니다.** Anthropic SDK는 모든 요청 URL에 `/v1/messages`를 추가합니다. Hermes는 SDK에 URL을 전달하기 전에 끝의 `/v1`을 제거하여 `/v1` 경로가 중복되는 것을 방지합니다.
- **`api-version`은 URL에 추가하지 않고 `default_query`를 통해 전송됩니다.** Azure Anthropic은 `api-version` 쿼리 문자열을 요구합니다. 기본 URL에 이를 포함하면 `/anthropic?api-version=.../v1/messages`와 같은 잘못된 경로가 생성되어 404가 반환됩니다. Hermes는 대신 Anthropic SDK의 `default_query`를 통해 `api-version=2025-04-15`를 전달합니다.
- **`x-api-key` 대신 Bearer 인증을 사용합니다.** Azure의 Anthropic 호환 경로는 Anthropic의 기본 `x-api-key` 헤더가 아니라 `Authorization: Bearer <key>`를 요구합니다. Hermes는 기본 URL에 `azure.com`이 있는 것을 감지하고 SDK의 `auth_token` 필드를 통해 API 키를 전달하여 올바른 헤더가 업스트림에 도달하도록 합니다.
- **1M 컨텍스트 창 베타 헤더가 유지됩니다.** Azure는 여전히 `anthropic-beta: context-1m-2025-08-07` 헤더를 통해 1M 토큰 Claude 컨텍스트(Opus 4.6/4.7, Sonnet 4.6)를 제한적으로 제공합니다. Hermes는 Azure 경로에서 해당 베타 헤더를 유지합니다(일부 구독에서 거부하기 때문에 기본 Anthropic OAuth 요청에서는 제거하지만 Azure에는 필요합니다).
- **OAuth 토큰 갱신이 비활성화됩니다.** Azure 배포는 정적 API 키를 사용합니다. Azure 엔드포인트에서는 Anthropic Console에 적용되는 `~/.claude/.credentials.json` OAuth 토큰 갱신 루프를 명시적으로 건너뛰어 Claude Code OAuth 토큰이 세션 중 Azure 키를 덮어쓰지 않도록 합니다.

## 대안: `provider: anthropic` + Azure 기본 URL

이미 `provider: anthropic`을 설정했고 Claude용으로 Microsoft Foundry를 가리키기만 하려는 경우 `azure-foundry` provider를 완전히 건너뛸 수 있습니다.

```yaml
model:
  provider: anthropic
  base_url: https://my-resource.services.ai.azure.com/anthropic
  key_env: AZURE_ANTHROPIC_KEY
  default: claude-sonnet-4-6
```

`AZURE_ANTHROPIC_KEY`를 `~/.hermes/.env`에 설정합니다. Hermes는 기본 URL에서 `azure.com`을 감지하고 Claude Code OAuth 토큰 체인을 우회하여 Azure 키를 `x-api-key` 인증으로 직접 사용합니다.

`key_env`는 정식 snake_case 필드명입니다. `api_key_env`(및 camelCase인 `keyEnv` / `apiKeyEnv`)도 별칭으로 허용됩니다. `key_env`와 `AZURE_ANTHROPIC_KEY`/`ANTHROPIC_API_KEY`가 모두 설정되어 있으면 `key_env`에 지정된 환경 변수가 우선합니다.

## 모델 검색

Azure는 *배포된* 모델 배포를 나열할 수 있는 순수 API 키 엔드포인트를 제공하지 **않습니다**. 배포 열거에는 추론 API 키가 아니라 Azure AD 주체를 사용한 Azure Resource Manager 인증(`az cognitiveservices account deployment list`)이 필요합니다.

Hermes가 할 수 있는 작업:

- Azure OpenAI v1 엔드포인트(`<resource>.openai.azure.com/openai/v1`)는 리소스의 **사용 가능한** 모델 카탈로그와 함께 `GET /models`를 제공합니다. Hermes는 이 목록을 사용하여 모델 선택 목록을 미리 채웁니다.
- Microsoft Foundry `/anthropic` 경로: URL 경로를 통해 감지되며 모델 이름은 수동으로 입력합니다.
- 비공개 / 방화벽 엔드포인트: "조사할 수 없음"이라는 친절한 메시지와 함께 수동 입력을 사용합니다.

언제든 배포 이름을 직접 입력할 수 있습니다. Hermes는 반환된 목록과 대조하여 검증하지 않습니다.

## 환경 변수

| 변수 | 용도 |
|----------|---------|
| `AZURE_FOUNDRY_API_KEY` | Microsoft Foundry / Azure OpenAI의 기본 API 키(api_key 모드) |
| `AZURE_FOUNDRY_BASE_URL` | 엔드포인트 URL(`hermes model`을 통해 설정하며, 환경 변수는 대체값으로 사용됨) |
| `AZURE_ANTHROPIC_KEY` | `provider: anthropic` + Azure 기본 URL에서 사용(`ANTHROPIC_API_KEY`의 대안) |
| `AZURE_TENANT_ID` | 서비스 주체 흐름을 위한 Entra ID 테넌트 |
| `AZURE_CLIENT_ID` | Entra ID 클라이언트 ID(서비스 주체, 워크로드 ID 또는 사용자 할당 관리 ID) |
| `AZURE_CLIENT_SECRET` | 서비스 주체 보안 비밀 |
| `AZURE_CLIENT_CERTIFICATE_PATH` | 서비스 주체 인증서(보안 비밀의 대안) |
| `AZURE_FEDERATED_TOKEN_FILE` | 워크로드 ID 페더레이션 토큰 경로(AKS) |
| `AZURE_AUTHORITY_HOST` | 소버린 클라우드 권한 호스트 재정의 |
| `IDENTITY_ENDPOINT` / `MSI_ENDPOINT` | App Service, Functions 및 Container Apps용 관리 ID 엔드포인트. VM은 일반적으로 IMDS를 대신 사용합니다. |

Azure SDK는 `AZURE_*` 환경 변수를 직접 읽습니다. Hermes는 `hermes doctor` 출력에서 어떤 출처가 있는지 보고하는 경우를 제외하고는 해당 변수를 검사하지 않습니다.

## 문제 해결

**gpt-5.x 배포에서 401 Unauthorized.**
Azure는 gpt-5.x를 `/responses`가 아니라 `/chat/completions`에서 제공합니다. URL에 `openai.azure.com`이 포함되어 있으면 Hermes가 이를 자동으로 처리하지만, 응답 본문에 `Invalid API key`가 포함된 401이 표시되면 `config.yaml`의 `api_mode`가 `chat_completions`인지 확인하세요.

**`/v1/messages?api-version=.../v1/messages`에서 404.**
이는 수정 전 Azure Anthropic 설정에서 발생하던 잘못된 URL 버그입니다. Hermes를 업그레이드하세요. 이제 `api-version` 매개변수가 기본 URL에 포함되지 않고 `default_query`를 통해 전달되므로 SDK가 URL을 결합하는 과정에서 이를 손상시킬 수 없습니다.

**마법사에 "Auto-detection incomplete."가 표시됩니다.**
엔드포인트가 `/models` 조사와 Anthropic Messages 조사를 모두 거부했습니다. 방화벽 뒤에 있거나 IP 허용 목록이 있는 비공개 엔드포인트에서는 정상입니다. 수동 API 모드 선택으로 대체하고 배포 이름을 입력하세요. 모든 기능은 계속 작동하며, Hermes가 선택 목록을 미리 채우지 못할 뿐입니다.

**잘못된 전송 방식이 선택됩니다.**
`hermes model`을 다시 실행하면 마법사가 다시 조사합니다. 그래도 잘못된 모드를 선택하면 `config.yaml`을 직접 편집할 수 있습니다.

```yaml
model:
  provider: azure-foundry
  api_mode: anthropic_messages   # or chat_completions
```

**Entra ID: `auth_mode: entra_id`로 전환한 후 "credential chain exhausted" 또는 401 Unauthorized.**
- 개발자 세션을 새로 고치려면 `az login`을 실행합니다(캐시된 토큰이 만료되었을 수 있습니다).
- `Azure AI User`(또는 `Foundry User`) 역할 할당이 적용되었는지 확인합니다. `az role assignment list --assignee <user-or-identity-id>`에 Foundry 리소스에 대한 역할이 표시되어야 합니다. 역할 전파에는 최대 5분이 걸릴 수 있습니다.
- 사용자 할당 관리 ID의 경우 `AZURE_CLIENT_ID`가 컴퓨팅 리소스에 연결된 ID와 일치하는지 다시 확인합니다.
- `hermes doctor`를 실행합니다. Azure Entra 조사는 토큰 획득 성공 여부를 보고하고 해결 방법 힌트를 포함합니다.

**Entra ID: 마법사 사전 검사가 멈추거나 시간 초과됩니다.**
10초 사전 검사는 소프트 검사입니다. "Save anyway and validate later"를 선택하고 대상 환경에 배포한 후 `hermes doctor`를 실행하세요. 토큰 서비스에 연결할 수 없거나 로컬 로그인 상태가 오래된 것이 일반적인 원인입니다. CI에서는 워크로드 ID를 우선 사용하고, 서비스 주체를 사용할 때는 `AZURE_TENANT_ID`+`AZURE_CLIENT_ID`+`AZURE_CLIENT_SECRET`을 설정하거나, 로컬 개발에서는 `az login`을 실행하세요.

**Entra ID를 사용하는 Anthropic 스타일 엔드포인트에서 401.**
동일한 `Azure AI User`(또는 `Foundry User`) 역할이 Foundry 리소스에 할당되어 있는지 확인합니다(`/openai/v1` 및 `/anthropic` 경로 모두에 적용됩니다). 마법사 중에는 OpenAI 스타일 조사가 작동하지만 런타임에 `claude-*` 요청이 실패한다면, 가장 일반적인 원인은 이전 마법사 실행에서 남은 `model.entra.scope`입니다. 런타임이 기본 `https://ai.azure.com/.default` 범위로 대체하도록 `config.yaml`에서 `entra.scope` 줄을 삭제하세요.

## 관련 문서

- [환경 변수](/reference/environment-variables)
- [설정](/user-guide/configuration)
- [AWS Bedrock](/guides/aws-bedrock) — 다른 주요 클라우드 provider 통합
- [Microsoft: Foundry용 Entra ID 구성](https://learn.microsoft.com/azure/ai-foundry/foundry-models/how-to/configure-entra-id) — 키 없는 경로에 대한 업스트림 문서
