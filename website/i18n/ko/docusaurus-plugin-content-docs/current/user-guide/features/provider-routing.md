---
title: 제공자 라우팅
description: 비용, 속도 또는 품질을 최적화하도록 OpenRouter 또는 Nous Portal 제공자 기본 설정을 구성합니다.
sidebar_label: 제공자 라우팅
sidebar_position: 7
---

# 제공자 라우팅

[OpenRouter](https://openrouter.ai) 또는 [Nous Portal](/integrations/nous-portal)을 LLM 제공자로 사용할 때 Hermes Agent는 **제공자 라우팅**을 지원합니다. 어떤 기반 AI 제공자가 요청을 처리할지와 우선순위를 세밀하게 제어할 수 있습니다.

OpenRouter는 여러 제공자(예: Anthropic, Google, AWS Bedrock, Together AI)로 요청을 라우팅합니다. 제공자 라우팅을 사용하면 비용, 속도, 품질을 최적화하거나 특정 제공자 요구 사항을 강제할 수 있습니다.

:::tip
Nous Portal을 통해 라우팅되는 트래픽에도 동일한 제공자 기본 설정이 적용되며, Portal 구독자는 토큰 과금 제공자를 10% 할인받습니다.
:::

## 구성

`~/.hermes/config.yaml`에 `provider_routing` 섹션을 추가합니다.

```yaml
provider_routing:
  sort: "price"           # How to rank providers
  only: []                # Whitelist: only use these providers
  ignore: []              # Blacklist: never use these providers
  order: []               # Explicit provider priority order
  require_parameters: false  # Only use providers that support all parameters
  data_collection: null   # Control data collection ("allow" or "deny")
```

:::info
제공자 라우팅은 OpenRouter 또는 Nous Portal을 사용할 때만 적용됩니다. 제공자에 직접 연결하는 경우(예: Anthropic API에 직접 연결)에는 영향을 주지 않습니다.
:::

## 옵션

### `sort`

요청에 사용할 수 있는 제공자의 순위를 OpenRouter가 정하는 방식을 제어합니다.

| 값 | 설명 |
|-------|-------------|
| `"price"` | 가장 저렴한 제공자를 우선합니다 |
| `"throughput"` | 초당 토큰 수가 가장 빠른 제공자를 우선합니다 |
| `"latency"` | 첫 토큰까지 걸리는 시간이 가장 짧은 제공자를 우선합니다 |

```yaml
provider_routing:
  sort: "price"
```

### `only`

제공자 slug의 허용 목록입니다. 설정하면 **이 제공자만** 사용됩니다. 나머지는 모두 제외됩니다. 각 제공자에 대해 OpenRouter가 표시하는 소문자 slug를 사용합니다.

```yaml
provider_routing:
  only:
    - "anthropic"
    - "google"
```

### `ignore`

제공자 이름의 차단 목록입니다. 이 제공자들은 가장 저렴하거나 빠른 옵션을 제공하더라도 **절대** 사용되지 않습니다.

```yaml
provider_routing:
  ignore:
    - "together"
    - "deepinfra"
```

### `order`

명시적인 우선순위 순서입니다. 앞에 나열된 제공자를 우선합니다. 나열되지 않은 제공자는 대체 제공자로 사용됩니다.

```yaml
provider_routing:
  order:
    - "anthropic"
    - "google"
    - "amazon-bedrock"
```

### `require_parameters`

`true`이면 OpenRouter는 요청의 **모든** 매개변수(`temperature`, `top_p`, `tools` 등)를 지원하는 제공자로만 라우팅합니다. 이를 통해 매개변수가 조용히 삭제되는 일을 방지합니다.

```yaml
provider_routing:
  require_parameters: true
```

### `data_collection`

제공자가 사용자의 프롬프트를 학습에 사용할 수 있는지 제어합니다. 옵션은 `"allow"` 또는 `"deny"`입니다.

```yaml
provider_routing:
  data_collection: "deny"
```

## 실용적인 예시

### 비용 최적화

사용 가능한 제공자 중 가장 저렴한 제공자로 라우팅합니다. 대량 사용과 개발에 적합합니다.

```yaml
provider_routing:
  sort: "price"
```

### 속도 최적화

대화형 사용을 위해 지연 시간이 짧은 제공자를 우선합니다.

```yaml
provider_routing:
  sort: "latency"
```

### 처리량 최적화

초당 토큰 수가 중요한 장문 생성에 적합합니다.

```yaml
provider_routing:
  sort: "throughput"
```

### 특정 제공자로 제한

일관성을 위해 모든 요청이 특정 제공자를 거치도록 합니다.

```yaml
provider_routing:
  only:
    - "anthropic"
```

### 특정 제공자 피하기

사용하지 않을 제공자를 제외합니다(예: 데이터 개인정보 보호를 위해).

```yaml
provider_routing:
  ignore:
    - "together"
    - "lepton"
  data_collection: "deny"
```

### 대체 제공자를 포함한 선호 순서

선호하는 제공자를 먼저 시도하고, 사용할 수 없으면 다른 제공자로 대체합니다.

```yaml
provider_routing:
  order:
    - "anthropic"
    - "google"
  require_parameters: true
```

## 작동 방식

제공자 라우팅 기본 설정은 `extra_body.provider` 필드를 통해 에이전트 채팅 요청 및 반복 제한 요약 시 OpenRouter 또는 Nous Portal로 전달됩니다. (`extra_body`는 OpenAI Python SDK 인자이며, JSON 요청에서는 최상위 `provider` 객체가 됩니다.) 압축 및 제목 생성과 같은 보조 작업은 `auxiliary.<task>.extra_body`에서 독립적으로 구성됩니다.

- **CLI 모드** — `~/.hermes/config.yaml`에서 구성하며 시작 시 로드됩니다
- **게이트웨이 모드** — 동일한 구성 파일을 사용하며 게이트웨이 시작 시 로드됩니다

라우팅 구성은 `config.yaml`에서 읽어 `AIAgent` 생성 시 매개변수로 전달됩니다.

```
providers_allowed  ← from provider_routing.only
providers_ignored  ← from provider_routing.ignore
providers_order    ← from provider_routing.order
provider_sort      ← from provider_routing.sort
provider_require_parameters ← from provider_routing.require_parameters
provider_data_collection    ← from provider_routing.data_collection
```

:::tip
여러 옵션을 조합할 수 있습니다. 예를 들어 가격순으로 정렬하면서 특정 제공자를 제외하고 매개변수 지원을 요구할 수 있습니다.

```yaml
provider_routing:
  sort: "price"
  ignore: ["together"]
  require_parameters: true
  data_collection: "deny"
```
:::

## 기본 동작

`provider_routing` 섹션이 구성되지 않은 경우(기본값), 집계자는 자체 기본 라우팅 로직을 사용하며 일반적으로 비용과 가용성의 균형을 자동으로 맞춥니다.

:::tip 제공자 라우팅과 대체 모델
제공자 라우팅은 **OpenRouter 또는 Nous Portal 뒤의 하위 제공자**가 요청을 처리하는 방식을 제어합니다. 기본 모델이 실패했을 때 완전히 다른 제공자로 자동 장애 조치를 수행하려면 [대체 제공자](/user-guide/features/fallback-providers)를 참조하세요.
:::
