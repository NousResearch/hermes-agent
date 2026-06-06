---
sidebar_position: 14
title: "AWS Bedrock"
description: "Amazon Bedrock과 함께 Hermes Agent 사용하기 — 네이티브 Converse API, IAM 인증, 가드레일(Guardrails) 및 교차 리전 추론(cross-region inference)"
---

# AWS Bedrock

Hermes Agent는 OpenAI 호환 엔드포인트가 아닌 **Converse API**를 사용하여 Amazon Bedrock을 네이티브 제공자로 지원합니다. 이를 통해 IAM 인증, 가드레일, 교차 리전 추론 프로필 및 모든 파운데이션 모델 등 Bedrock 생태계에 대한 전체 액세스 권한을 얻을 수 있습니다.

## 사전 요구 사항

- **AWS 자격 증명** — [boto3 자격 증명 체인(credential chain)](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html)에서 지원하는 모든 소스:
  - IAM 인스턴스 역할 (EC2, ECS, Lambda — 구성 필요 없음)
  - `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` 환경 변수
  - SSO 또는 명명된 프로필을 위한 `AWS_PROFILE`
  - 로컬 개발을 위한 `aws configure`
- **boto3** — `pip install hermes-agent[bedrock]`으로 설치
- **IAM 권한** — 최소 권한:
  - `bedrock:InvokeModel` 및 `bedrock:InvokeModelWithResponseStream` (추론용)
  - `bedrock:ListFoundationModels` 및 `bedrock:ListInferenceProfiles` (모델 검색용)

:::tip EC2 / ECS / Lambda
AWS 컴퓨팅 환경에서는 `AmazonBedrockFullAccess` 권한이 있는 IAM 역할을 연결하기만 하면 완료됩니다. API 키나 `.env` 구성이 필요 없습니다 — Hermes가 인스턴스 역할을 자동으로 감지합니다.
:::

## 빠른 시작

```bash
# Bedrock 지원이 포함된 버전 설치
pip install hermes-agent[bedrock]

# Bedrock을 제공자로 선택
hermes model
# → "More providers..." 선택 → "AWS Bedrock" 선택
# → 리전과 모델 선택

# 채팅 시작
hermes chat
```

## 구성

`hermes model`을 실행한 후, `~/.hermes/config.yaml`에는 다음이 포함됩니다:

```yaml
model:
  default: us.anthropic.claude-sonnet-4-6
  provider: bedrock
  base_url: https://bedrock-runtime.us-east-2.amazonaws.com

bedrock:
  region: us-east-2
```

### 리전 (Region)

다음 방법 중 하나로 AWS 리전을 설정하세요 (우선순위가 높은 순서대로):

1. `config.yaml`의 `bedrock.region`
2. `AWS_REGION` 환경 변수
3. `AWS_DEFAULT_REGION` 환경 변수
4. 기본값: `us-east-1`

### 가드레일 (Guardrails)

모든 모델 호출에 [Amazon Bedrock 가드레일](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html)을 적용하려면:

```yaml
bedrock:
  region: us-east-2
  guardrail:
    guardrail_identifier: "abc123def456"  # Bedrock 콘솔에서 확인 가능
    guardrail_version: "1"                # 버전 번호 또는 "DRAFT"
    stream_processing_mode: "async"       # "sync" 또는 "async"
    trace: "disabled"                     # "enabled", "disabled", 또는 "enabled_full"
```

### 모델 검색 (Model Discovery)

Hermes는 Bedrock 컨트롤 플레인을 통해 사용 가능한 모델을 자동으로 검색합니다. 검색 방식을 사용자 지정할 수 있습니다:

```yaml
bedrock:
  discovery:
    enabled: true
    provider_filter: ["anthropic", "amazon"]  # 이 제공자들만 표시
    refresh_interval: 3600                     # 1시간 동안 캐시
```

## 사용 가능한 모델

Bedrock 모델은 온디맨드 호출을 위해 **추론 프로필 ID(inference profile IDs)**를 사용합니다. `hermes model` 선택기는 이를 자동으로 표시하며, 권장 모델이 맨 위에 표시됩니다:

| 모델 | ID | 참고 |
|-------|-----|-------|
| Claude Sonnet 4.6 | `us.anthropic.claude-sonnet-4-6` | 권장 — 속도와 기능의 최적의 균형 |
| Claude Opus 4.6 | `us.anthropic.claude-opus-4-6-v1` | 가장 뛰어난 성능 |
| Claude Haiku 4.5 | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | 가장 빠른 Claude |
| Amazon Nova Pro | `us.amazon.nova-pro-v1:0` | Amazon의 플래그십 |
| Amazon Nova Micro | `us.amazon.nova-micro-v1:0` | 가장 빠르고 저렴함 |
| DeepSeek V3.2 | `deepseek.v3.2` | 강력한 오픈 모델 |
| Llama 4 Scout 17B | `us.meta.llama4-scout-17b-instruct-v1:0` | Meta의 최신 모델 |

:::info 교차 리전 추론 (Cross-Region Inference)
`us.` 접두사가 붙은 모델은 교차 리전 추론 프로필을 사용하며, 이는 AWS 리전 전체에서 더 나은 용량과 자동 장애 조치(failover)를 제공합니다. `global.` 접두사가 붙은 모델은 전 세계의 사용 가능한 모든 리전으로 라우팅됩니다.
:::

## 세션 중간에 모델 전환하기

대화 중에 `/model` 명령어를 사용하세요:

```
/model us.amazon.nova-pro-v1:0
/model deepseek.v3.2
/model us.anthropic.claude-opus-4-6-v1
```

## 진단 (Diagnostics)

```bash
hermes doctor
```

이 명령어(doctor)는 다음을 확인합니다:
- AWS 자격 증명 사용 가능 여부 (환경 변수, IAM 역할, SSO)
- `boto3` 설치 여부
- Bedrock API 도달 가능 여부 (ListFoundationModels)
- 해당 리전에서 사용 가능한 모델 수

## 게이트웨이 (메시징 플랫폼)

Bedrock은 모든 Hermes 게이트웨이 플랫폼(Telegram, Discord, Slack, Feishu 등)에서 작동합니다. Bedrock을 제공자로 구성한 다음 평소처럼 게이트웨이를 시작하세요:

```bash
hermes gateway setup
hermes gateway start
```

게이트웨이는 `config.yaml`을 읽고 동일한 Bedrock 제공자 구성을 사용합니다.

## 문제 해결 (Troubleshooting)

### "No API key found" / "No AWS credentials"

Hermes는 다음 순서로 자격 증명을 확인합니다:
1. `AWS_BEARER_TOKEN_BEDROCK`
2. `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`
3. `AWS_PROFILE`
4. EC2 인스턴스 메타데이터 (IMDS)
5. ECS 컨테이너 자격 증명
6. Lambda 실행 역할

아무것도 찾을 수 없는 경우 `aws configure`를 실행하거나 컴퓨팅 인스턴스에 IAM 역할을 연결하세요.

### "Invocation of model ID ... with on-demand throughput isn't supported"

단일 파운데이션 모델 ID 대신 **추론 프로필 ID** (`us.` 또는 `global.` 접두사 포함)를 사용하세요. 예:
- ❌ `anthropic.claude-sonnet-4-6`
- ✅ `us.anthropic.claude-sonnet-4-6`

### "ThrottlingException"

Bedrock의 모델별 속도 제한(rate limit)에 도달했습니다. Hermes는 백오프(backoff)와 함께 자동으로 재시도합니다. 제한을 늘리려면 [AWS Service Quotas 콘솔](https://console.aws.amazon.com/servicequotas/)에서 할당량 증가를 요청하세요.

## AWS 원클릭 배포

CloudFormation을 통한 EC2의 완전 자동화된 배포를 원한다면:

**[sample-hermes-agent-on-aws-with-bedrock](https://github.com/JiaDe-Wu/sample-hermes-agent-on-aws-with-bedrock)** — VPC, IAM 역할, EC2 인스턴스를 생성하고 Bedrock을 자동으로 구성합니다. 클릭 한 번으로 모든 리전에 배포할 수 있습니다.
