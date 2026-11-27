---
sidebar_position: 14
title: "AWS Bedrock"
description: "Amazon Bedrock과 Hermes Agent 사용 — 네이티브 Converse API, IAM 인증, Guardrails, 리전 간 추론"
---

# AWS Bedrock

Hermes Agent는 OpenAI 호환 엔드포인트가 아닌 **Converse API**를 사용하는 네이티브 제공자로 Amazon Bedrock을 지원합니다. 이를 통해 IAM 인증, Guardrails, 리전 간 추론 프로필, 모든 파운데이션 모델 등 Bedrock 생태계 전체에 접근할 수 있습니다.

## 사전 요구 사항

- **AWS 자격 증명** — [boto3 자격 증명 체인](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html)이 지원하는 모든 소스:
  - IAM 인스턴스 역할(EC2, ECS, Lambda — 구성 불필요)
  - `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` 환경 변수
  - SSO 또는 이름이 지정된 프로필을 위한 `AWS_PROFILE`
  - 로컬 개발을 위한 `aws configure`
- **boto3** — `cd ~/.hermes/hermes-agent && uv pip install -e "[bedrock]"`로 설치
- **IAM 권한** — 최소한 다음 권한:
  - `bedrock:InvokeModel` 및 `bedrock:InvokeModelWithResponseStream`(추론용)
  - `bedrock:ListFoundationModels` 및 `bedrock:ListInferenceProfiles`(모델 검색용)

:::tip EC2 / ECS / Lambda
AWS 컴퓨팅에서 `AmazonBedrockFullAccess`가 포함된 IAM 역할을 연결하면 완료입니다. API 키나 `.env` 구성은 필요하지 않습니다 — Hermes가 인스턴스 역할을 자동으로 감지합니다.
:::

## 빠른 시작

```bash
# Install with Bedrock support
cd ~/.hermes/hermes-agent && uv pip install -e ".[bedrock]"

# Select Bedrock as your provider
hermes model
# → Choose "More providers..." → "AWS Bedrock"
# → Select your region and model

# Start chatting
hermes chat
```

## 구성

`hermes model`을 실행하면 `~/.hermes/config.yaml`에 다음 내용이 포함됩니다.

```yaml
model:
  default: us.anthropic.claude-sonnet-4-6
  provider: bedrock
  base_url: https://bedrock-runtime.us-east-2.amazonaws.com

bedrock:
  region: us-east-2
```

### 리전

다음 방법 중 하나로 AWS 리전을 설정합니다(우선순위가 높은 순서).

1. `config.yaml`의 `bedrock.region`
2. `AWS_REGION` 환경 변수
3. `AWS_DEFAULT_REGION` 환경 변수
4. 기본값: `us-east-1`

### Guardrails

모든 모델 호출에 [Amazon Bedrock Guardrails](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html)를 적용하려면:

```yaml
bedrock:
  region: us-east-2
  guardrail:
    guardrail_identifier: "abc123def456"  # From the Bedrock console
    guardrail_version: "1"                # Version number or "DRAFT"
    stream_processing_mode: "async"       # "sync" or "async"
    trace: "disabled"                     # "enabled", "disabled", or "enabled_full"
```

### 모델 검색

Hermes는 Bedrock 컨트롤 플레인을 통해 사용할 수 있는 모델을 자동으로 검색합니다. 검색을 사용자 지정할 수 있습니다.

```yaml
bedrock:
  discovery:
    enabled: true
    provider_filter: ["anthropic", "amazon"]  # Only show these providers
    refresh_interval: 3600                     # Cache for 1 hour
```

### 프롬프트 캐싱(cachePoint)

Hermes는 시스템 프롬프트, 도구 정의, 최신 메시지 뒤에 `cachePoint` 마커를 삽입하여 Bedrock **Converse API** 경로에 프롬프트 캐싱을 자동으로 적용합니다. 이를 지원하지 않는 모델에 `cachePoint` 블록을 보내면 `ValidationException`이 발생하므로, 마커는 검증된 허용 목록에 있는 모델(Anthropic Claude 및 Amazon Nova 모델 ID)에만 추가됩니다. 알 수 없는 모델은 기본적으로 캐시 마커를 사용하지 않습니다. Claude 모델은 일반적으로 자체 프롬프트 캐싱이 있는 AnthropicBedrock SDK 경로를 사용합니다 — Converse `cachePoint` 경로는 Nova와 bearer-token Claude 폴백을 처리합니다. 별도의 구성은 필요하지 않으며, 캐시 읽기/쓰기는 사용량 집계에 표시됩니다.

### 컨텍스트 윈도우 프로빙

컨텍스트 윈도우가 Hermes의 정적 테이블에 없는 모델의 경우, Hermes는 고정된 단계(~1.3M 및 ~2.2M 토큰)에서 크기를 초과한 요청을 보내고 Bedrock의 길이 검증 오류에 보고된 `maximum`을 파싱하여 실제 제한을 프로빙할 수 있습니다. 프로빙된 값은 정적 테이블과 동일한 메타데이터 캐시에 반영됩니다. 모델의 1M 윈도우가 정식 출시되기 전에 저장되어 실제보다 작게 보고하는 오래된 캐시 항목은 더 큰 알려진 값을 사용하도록 자동으로 삭제됩니다.

## 사용 가능한 모델

Bedrock 모델은 온디맨드 호출에 **추론 프로필 ID**를 사용합니다. `hermes model` 선택기에 이 모델들이 자동으로 표시되며, 권장 모델이 맨 위에 표시됩니다.

| 모델 | ID | 참고 |
|-------|-----|-------|
| Claude Sonnet 4.6 | `us.anthropic.claude-sonnet-4-6` | 권장 — 속도와 성능의 최적 균형 |
| Claude Opus 4.6 | `us.anthropic.claude-opus-4-6-v1` | 가장 뛰어난 성능 |
| Claude Haiku 4.5 | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | 가장 빠른 Claude |
| Amazon Nova Pro | `us.amazon.nova-pro-v1:0` | Amazon의 플래그십 |
| Amazon Nova Micro | `us.amazon.nova-micro-v1:0` | 가장 빠르고 저렴함 |
| DeepSeek V3.2 | `deepseek.v3.2` | 강력한 오픈 모델 |
| Llama 4 Scout 17B | `us.meta.llama4-scout-17b-instruct-v1:0` | Meta의 최신 모델 |

:::info 리전 간 추론
`us.` 접두사가 붙은 모델은 리전 간 추론 프로필을 사용하며, AWS 리전 간 더 나은 용량과 자동 장애 조치를 제공합니다. `global.` 접두사가 붙은 모델은 전 세계에서 사용 가능한 모든 리전으로 라우팅됩니다.
:::

## 세션 중 모델 전환

대화 중 `/model` 명령을 사용합니다.

```
/model us.amazon.nova-pro-v1:0
/model deepseek.v3.2
/model us.anthropic.claude-opus-4-6-v1
```

## 진단

```bash
hermes doctor
```

doctor는 다음을 확인합니다.
- AWS 자격 증명을 사용할 수 있는지(환경 변수, IAM 역할, SSO)
- `boto3`가 설치되어 있는지
- Bedrock API에 접근할 수 있는지(ListFoundationModels)
- 해당 리전에서 사용할 수 있는 모델 수

## 게이트웨이(메시징 플랫폼)

Bedrock은 모든 Hermes 게이트웨이 플랫폼(Telegram, Discord, Slack, Feishu 등)에서 작동합니다. Bedrock을 제공자로 구성한 다음 평소처럼 게이트웨이를 시작하세요.

```bash
hermes gateway setup
hermes gateway start
```

게이트웨이는 `config.yaml`을 읽고 동일한 Bedrock 제공자 구성을 사용합니다.

## 문제 해결

### "API 키를 찾을 수 없음" / "AWS 자격 증명 없음"

Hermes는 다음 순서로 자격 증명을 확인합니다.
1. `AWS_BEARER_TOKEN_BEDROCK`
2. `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`
3. `AWS_PROFILE`
4. EC2 인스턴스 메타데이터(IMDS)
5. ECS 컨테이너 자격 증명
6. Lambda 실행 역할

아무 자격 증명도 찾지 못하면 `aws configure`를 실행하거나 컴퓨팅 인스턴스에 IAM 역할을 연결하세요.

### "온디맨드 처리량으로는 모델 ID ... 호출을 지원하지 않음"

기본 파운데이션 모델 ID 대신 `us.` 또는 `global.` 접두사가 붙은 **추론 프로필 ID**를 사용하세요. 예:
- ❌ `anthropic.claude-sonnet-4-6`
- ✅ `us.anthropic.claude-sonnet-4-6`

### "ThrottlingException"

Bedrock 모델별 속도 제한에 도달했습니다. Hermes는 백오프를 적용하여 자동으로 재시도합니다. 제한을 늘리려면 [AWS Service Quotas 콘솔](https://console.aws.amazon.com/servicequotas/)에서 할당량 증가를 요청하세요.

## 원클릭 AWS 배포

CloudFormation을 사용하여 EC2에 완전히 자동화된 배포를 수행하려면:

**[sample-hermes-agent-on-aws-with-bedrock](https://github.com/JiaDe-Wu/sample-hermes-agent-on-aws-with-bedrock)** — VPC, IAM 역할, EC2 인스턴스를 생성하고 Bedrock을 자동으로 구성합니다. 한 번의 클릭으로 모든 리전에 배포하세요.
