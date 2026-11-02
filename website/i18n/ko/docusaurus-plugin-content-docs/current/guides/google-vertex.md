---
sidebar_position: 15
title: "Google Vertex AI"
description: "Google Cloud Vertex AI에서 Gemini를 사용하세요 — OAuth2 서비스 계정 또는 ADC, GCP 결제 및 할당량, 정적 API 키 불필요"
---

# Google Vertex AI

Hermes Agent는 Vertex의 OpenAI 호환 엔드포인트를 통해 **Google Cloud Vertex AI의 Gemini 모델**을 지원합니다. 정적 API 키로 `generativelanguage.googleapis.com`에 연결하는 [Google AI Studio provider](/guides/google-gemini)와 달리 Vertex는 **엔터프라이즈급 속도 제한과 GCP 결제/크레딧**을 제공하며, AI Studio 키가 아니라 Google Cloud 계정에 Gemini 사용량을 연결하려는 경우 적합합니다.

:::info Vertex는 API 키가 아니라 OAuth2로 인증합니다
표준 엔드포인트에는 **정적 API 키가 없습니다**. 모든 요청에는 서비스 계정 JSON 또는 Application Default Credentials(ADC)에서 발급되는 짧은 수명의 **OAuth2 액세스 토큰**(TTL 약 1시간)이 필요합니다. Hermes가 이 토큰을 발급하고 **자동으로 갱신**하므로 토큰을 직접 붙여 넣을 필요가 없습니다. 따라서 임시 토큰을 사용자 지정 provider의 `api_key` 필드에 붙여 넣어도 세션 중간에 만료되기 때문에 작동하지 않습니다.
:::

## 사전 요구 사항

- **Vertex AI API가 활성화되어 있고 결제가 설정된 Google Cloud 프로젝트**
- 다음 중 하나의 **자격 증명**:
  - `roles/aiplatform.user` 역할이 있는 **서비스 계정 JSON** 키 파일
  - `gcloud auth application-default login`을 통한 **Application Default Credentials**(ADC) (또는 GCP VM에서 실행할 때 메타데이터 서버)
- **`google-auth`** — Vertex를 처음 선택할 때 자동으로 설치됩니다(lazy install). 설치에 실패하면 `hermes setup`을 실행하여 관리형 설치를 복구하세요.

## 빠른 시작

```bash
# Option A — service account JSON (recommended for servers / gateways)
echo "VERTEX_CREDENTIALS_PATH=/path/to/service-account.json" >> ~/.hermes/.env

# Option B — Application Default Credentials (good for local dev)
gcloud auth application-default login

# Select Vertex as your provider
hermes model
# → Choose "More providers..." → "Google Vertex AI"
# → Enter your GCP project ID (or leave blank to use the one in your credentials)
# → Choose a region (default: global)
# → Select a Gemini model

# Start chatting
hermes chat
```

## 구성

Vertex는 설정을 민감도에 따라 나눕니다.

- **자격 증명 경로**는 비밀을 가리키는 포인터이므로 `~/.hermes/.env`에 저장합니다.
- **프로젝트 ID와 리전**은 비밀이 아닌 라우팅 설정이므로 `~/.hermes/config.yaml`에 저장합니다.

`~/.hermes/.env`:

```bash
# One of these (checked in this order); omit both to use ADC:
VERTEX_CREDENTIALS_PATH=/path/to/service-account.json
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

`~/.hermes/config.yaml`:

```yaml
model:
  default: google/gemini-3-flash-preview
  provider: vertex

vertex:
  project_id: my-gcp-project   # blank → use the project embedded in the credentials
  region: global               # "global" is required for the Gemini 3.x previews
```

:::tip 환경 변수가 config.yaml보다 우선합니다
`VERTEX_PROJECT_ID`와 `VERTEX_REGION`은 `config.yaml`의 `vertex.project_id` / `vertex.region` 값을 재정의합니다. 셸별 재정의에 사용하고, 지속적으로 사용할 설정은 `config.yaml`에 보관하세요.
:::

### 인증 작동 방식

1. Hermes는 다음 순서로 자격 증명을 확인합니다: `VERTEX_CREDENTIALS_PATH` → `GOOGLE_APPLICATION_CREDENTIALS` → ADC.
2. OAuth2 액세스 토큰(`cloud-platform` 범위)을 발급하고 캐시한 뒤, 토큰 만료까지 5분 이내가 되면 갱신합니다.
3. 토큰은 Vertex 엔드포인트를 가리키는 표준 OpenAI 클라이언트에 전달됩니다.
   ```text
   https://aiplatform.googleapis.com/v1beta1/projects/{project}/locations/{region}/endpoints/openapi
   ```
   리전 위치는 대신 `{region}-aiplatform.googleapis.com` 호스트를 사용합니다.
4. 세션이 토큰 수명보다 오래 실행되어 요청이 `401`을 반환하면 Hermes가 토큰을 다시 발급하고 자동으로 재시도합니다. 장시간 실행되는 gateway에서 ADC의 갱신 토큰 자체가 만료된 경우에는 서비스 계정 JSON이 구성되어 있으면 Hermes가 이를 대신 사용합니다.

## 사용 가능한 모델

Vertex에서는 모델 ID에 `google/` vendor 접두사가 필요합니다. `hermes model` 선택기에서 다음 모델을 제공합니다.

| 모델 | ID |
|-------|----|
| Gemini 3.1 Pro Preview | `google/gemini-3.1-pro-preview` |
| Gemini 3 Pro Preview | `google/gemini-3-pro-preview` |
| Gemini 3 Flash Preview | `google/gemini-3-flash-preview` |
| Gemini 3.1 Flash Lite Preview | `google/gemini-3.1-flash-lite-preview` |
| Gemini 2.5 Pro | `google/gemini-2.5-pro` |
| Gemini 2.5 Flash | `google/gemini-2.5-flash` |

:::note Gemini 3.x에는 `global` 리전을 사용하세요
Gemini 3.x preview 모델은 `global` 엔드포인트를 통해 제공됩니다. 리전 엔드포인트(`us-central1` 등)에서는 해당 모델이 404를 반환할 수 있습니다. 리전을 특정해야 할 이유가 없다면 `region: global`로 두세요.
:::

## 세션 중 모델 전환

```text
/model google/gemini-3-pro-preview
/model google/gemini-3-flash-preview
```

`/model`은 이미 구성된 provider와 모델 사이를 전환하며, 새 자격 증명을 수집하지 않습니다. 먼저 `hermes model`로 Vertex를 구성하세요.

## 추론 / 사고

Vertex는 OpenAI 호환 인터페이스를 통해 Gemini의 사고 예산을 노출합니다. Hermes는 reasoning-effort 설정을 `extra_body.google.thinking_config`에 자동으로 매핑하므로 `reasoning_effort`는 다른 Gemini 인터페이스와 동일하게 작동합니다.

## 진단

```bash
hermes doctor
```

doctor는 Vertex 자격 증명을 서비스 계정 경로 또는 ADC로 확인할 수 있는지, provider가 구성되었는지를 보고합니다.

## 문제 해결

### "Vertex AI credentials could not be resolved"

Hermes가 서비스 계정 JSON과 작동하는 ADC를 모두 찾지 못했습니다. `~/.hermes/.env`에 `VERTEX_CREDENTIALS_PATH`를 설정하거나 `gcloud auth application-default login`을 실행하세요. 프로젝트가 자격 증명에 포함되어 있지 않다면 `config.yaml`에 `vertex.project_id`를 설정하세요.

### `google-auth`가 설치되지 않음

Vertex provider를 처음 선택할 때 Hermes가 이를 lazy-install합니다. 설치에 실패하면 `hermes setup`을 실행하여 관리형 설치를 복구하세요.

### Gemini 3.x 모델에서 404

리전 엔드포인트를 사용하고 있을 가능성이 높습니다. `config.yaml`의 `vertex:` 섹션에서 `region: global`을 설정하거나 `VERTEX_REGION`을 해제하세요.

### 403 / 권한 거부

서비스 계정(또는 ADC ID)에 프로젝트의 `roles/aiplatform.user` 역할이 필요하며, 해당 프로젝트에서 Vertex AI API가 활성화되어 있어야 합니다.

## 관련 문서

- [Google Gemini (AI Studio)](/guides/google-gemini) — GCP 없이 정적 API 키를 사용하는 Gemini
- [AWS Bedrock](/guides/aws-bedrock) — 또 다른 네이티브 클라우드 provider 통합
- [AI Providers](/integrations/providers)
- [구성](/user-guide/configuration)
