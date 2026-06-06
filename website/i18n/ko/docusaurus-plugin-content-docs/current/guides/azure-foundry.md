---
sidebar_position: 6
title: "Azure AI Foundry"
description: "Hermes Agent를 Azure OpenAI, Azure DeepSeek 또는 Azure AI Foundry 카탈로그의 다른 모델에 연결하기"
---

# Azure AI Foundry

Hermes Agent는 [Azure AI Foundry](https://ai.azure.com/) (이전의 Azure AI Studio / Azure OpenAI)에 배포된 모델에 연결할 수 있습니다. 여기에는 OpenAI 모델(GPT-4o, o1, o3-mini)과 Azure의 모델 카탈로그를 통해 배포된 타사 모델(DeepSeek R1, Llama 3, Mistral)이 모두 포함됩니다.

## 엔드포인트 유형

Azure에는 엔드포인트가 배포되는 방식에 따라 약간 다른 인증 및 라우팅 형식을 필요로 하는 두 가지 유형의 인퍼런스 모델이 있습니다. Hermes는 두 가지 유형 모두를 최고 수준의 프로바이더로 지원합니다.

### 1. Azure OpenAI (기본 제공)
`your-resource.openai.azure.com`에 배포된 OpenAI의 자체 모델(GPT-4o, o1-mini 등)용입니다. 이는 표준 Azure OpenAI 인증 및 URL 구조를 사용합니다.

### 2. Azure AI Foundry Serverless (카탈로그 모델)
`your-endpoint.models.ai.azure.com`에 배포된 비(非) OpenAI 모델(DeepSeek R1, Llama 3, Mistral)용입니다. 이것들은 모델 카탈로그에서 "서버리스 API로 배포(Deploy as serverless API)"를 사용하여 배포되며, 표준 OpenAI SDK를 사용하지만 다른 기본 URL과 헤더가 필요합니다. Hermes는 이를 `azure-foundry` 프로바이더로 부릅니다.

---

## 옵션 1: Azure OpenAI 모델 구성

`gpt-4o` 또는 `o3-mini`를 실행 중이고 엔드포인트가 `openai.azure.com`으로 끝나는 경우 이 방법을 사용하세요.

### 1단계: 배포 세부 정보 수집
Azure Portal 또는 AI Foundry에서 다음 세 가지가 필요합니다:
1. **API 키 (API Key)** (키 1 또는 키 2)
2. **엔드포인트 URL (Endpoint URL)** (예: `https://my-company-ai.openai.azure.com/`)
3. **배포 이름 (Deployment Name)** (기본 모델 이름이 아닌 배포에 지정한 이름)

### 2단계: 환경 변수 설정
`~/.hermes/.env` 파일에 자격 증명을 추가합니다:

```env
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-08-01-preview
```

*(참고: API 버전은 수시로 변경되지만, Hermes에는 하드코딩된 기본값이 포함되어 있으므로 대부분의 경우 버전을 생략할 수 있습니다. 도구 호출이 포함된 채팅 완성을 지원하는 버전을 사용하세요.)*

### 3단계: Hermes 구성
`~/.hermes/config.yaml` 파일에서 배포를 가리키도록 설정합니다:

```yaml
model:
  provider: azure
  default: your-deployment-name  # 모델 이름이 아님!
```

> **중요:** `default` 모델은 반드시 Azure에서 설정한 **배포 이름(deployment name)**과 일치해야 합니다. `gpt-4o` 모델을 배포하고 `my-gpt4o-dev`라고 이름을 지었다면, Hermes 구성에는 `my-gpt4o-dev`를 지정해야 합니다.

### 여러 배포 간 전환하기
세션 도중이나 `config.yaml`에서 언제든지 배포 간 전환이 가능합니다:

```bash
/model azure/my-gpt4o-dev
/model azure/my-o3-mini
```

---

## 옵션 2: Azure AI Foundry (DeepSeek / 카탈로그 모델)

서버리스 엔드포인트를 사용하여 모델 카탈로그에서 **DeepSeek R1** 또는 기타 모델을 배포한 경우 이 방법을 사용하세요. 엔드포인트는 `models.ai.azure.com`으로 끝납니다.

이러한 엔드포인트는 `azure` 프로바이더 구성을 따르지 않으며 대신 `azure-foundry` 프로바이더를 사용합니다.

### 1단계: 배포 세부 정보 수집
Foundry 대시보드(프로젝트 → 배포)에서 대상 배포를 클릭합니다. 다음이 필요합니다:
1. **기본 URL (Base URL)** (엔드포인트에서 대상 경로를 제외한 부분. 예: 엔드포인트가 `https://DeepSeek-R1-xyza.eastus2.models.ai.azure.com/v1/chat/completions`인 경우 기본 URL은 `https://DeepSeek-R1-xyza.eastus2.models.ai.azure.com/v1`)
2. **API 키 (API Key)**

### 2단계: Hermes 구성
표준 Azure 인스턴스와 달리, 카탈로그 배포는 종종 각각 고유한 기본 URL을 가집니다. 이를 처리하기 위해 `~/.hermes/config.yaml`의 전용 블록에서 배포를 매핑합니다:

```yaml
azure_foundry:
  models:
    # 모델에 지정하고 싶은 이름
    deepseek-r1:
      base_url: https://DeepSeek-R1-xyza.eastus2.models.ai.azure.com/v1
      key_env: AZURE_FOUNDRY_DEEPSEEK_KEY  # 사용할 환경 변수 이름
    
    # 여러 모델을 추가할 수 있습니다
    llama-3:
      base_url: https://Llama-3-abcd.eastus2.models.ai.azure.com/v1
      key_env: AZURE_FOUNDRY_LLAMA_KEY

model:
  provider: azure-foundry
  default: deepseek-r1  # 위에서 정의한 이름과 일치해야 합니다
```

### 3단계: 환경 변수 설정
`~/.hermes/.env` 파일에 구성 파일에서 지정한 키를 추가합니다:

```env
AZURE_FOUNDRY_DEEPSEEK_KEY=your_key_for_deepseek
AZURE_FOUNDRY_LLAMA_KEY=your_key_for_llama
```

### 세션 내에서 모델 전환하기
정의한 이름은 표준 모델 선택기 메뉴(`hermes model`)와 세션 내 명령어에서 사용할 수 있게 됩니다:

```bash
/model azure-foundry/deepseek-r1
/model azure-foundry/llama-3
```

---

## 보안 및 네트워킹

### VNet 접근
Azure 리소스가 가상 네트워크(VNet) 내에 보호되어 있는 경우, Hermes를 실행하는 머신은 해당 VNet 내에 있거나 VPN/ExpressRoute를 통해 연결할 수 있어야 합니다. Hermes는 요청할 때 시스템의 프록시 설정(`HTTP_PROXY`, `HTTPS_PROXY`)을 존중합니다.

### Entra ID (이전의 Azure AD) 인증
현재 Hermes는 서비스 주체(Service Principal)를 사용한 토큰 기반(Entra ID) 인증을 직접 지원하지 않습니다. 인증에는 API 키를 사용해야 합니다.

---

## 문제 해결

### "Resource not found" (404)
Azure OpenAI는 특정 URL 형식(`.../openai/deployments/{deployment_id}/chat/completions`)을 예상합니다. 404 오류가 발생한다면 일반적으로 `model.default`가 Azure의 배포 이름과 정확히 일치하지 않는 경우입니다. 모델이 `gpt-4o`이더라도 배포 이름이 `my-deployment`라면 구성을 `my-deployment`로 설정해야 합니다.

### "Invalid authentication" (401)
API 키가 올바른 리소스와 일치하는지 확인하세요. Azure AI Foundry(서버리스)와 Azure OpenAI는 서로 다른 서비스이므로 API 키를 교차해서 사용할 수 없습니다. `AZURE_OPENAI_API_KEY`와 `AZURE_FOUNDRY_*_KEY` 환경 변수를 올바르게 설정했는지 확인하세요.

### 모델이 도구(Tools)를 사용할 수 없음
Azure 카탈로그의 일부 구형 모델이나 특정 설정은 도구 호출(함수 호출)을 지원하지 않습니다. Hermes Agent에는 도구 호출이 가능한 모델이 필요합니다. 최신 버전을 사용 중인지 확인하고 모델 세부 정보에서 도구 호출이 활성화되어 있는지(또는 지원되는 모델인지) 확인하세요.
