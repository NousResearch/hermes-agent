---
title: 제공자 라우팅 (Provider Routing)
description: 비용, 속도 또는 품질을 최적화하기 위해 OpenRouter 제공자 기본 설정을 구성합니다.
sidebar_label: 제공자 라우팅
sidebar_position: 7
---

# 제공자 라우팅 (Provider Routing)

[OpenRouter](https://openrouter.ai)를 LLM 제공자로 사용할 때, Hermes Agent는 **제공자 라우팅(provider routing)**을 지원합니다. 이는 어떤 기본 AI 제공자가 요청을 처리하고 어떻게 우선순위를 정할지에 대한 세밀한 제어를 제공합니다.

OpenRouter는 많은 제공자(예: Anthropic, Google, AWS Bedrock, Together AI)로 요청을 라우팅합니다. 제공자 라우팅을 사용하면 비용, 속도, 품질을 최적화하거나 특정 제공자 요구 사항을 강제할 수 있습니다.

:::tip
[Nous Portal](/integrations/nous-portal)을 통해 라우팅된 트래픽도 모델별 라우팅과 우선순위 구성을 따릅니다 — 그리고 Portal 구독자는 토큰 청구 제공자에 대해 10% 할인을 받습니다.
:::

## 구성

`~/.hermes/config.yaml`에 `provider_routing` 섹션을 추가하세요:

```yaml
provider_routing:
  sort: "price"           # 제공자 순위를 매기는 방법
  only: []                # 화이트리스트: 이 제공자들만 사용
  ignore: []              # 블랙리스트: 이 제공자들은 절대 사용 안 함
  order: []               # 명시적인 제공자 우선순위
  require_parameters: false  # 모든 파라미터를 지원하는 제공자만 사용
  data_collection: null   # 데이터 수집 제어 ("allow" 또는 "deny")
```

:::info
제공자 라우팅은 OpenRouter를 사용할 때만 적용됩니다. 직접적인 제공자 연결(예: Anthropic API에 직접 연결)에는 아무런 영향을 미치지 않습니다.
:::

## 옵션

### `sort`

OpenRouter가 요청에 대해 사용 가능한 제공자의 순위를 매기는 방법을 제어합니다.

| 값 | 설명 |
|-------|-------------|
| `"price"` | 가장 저렴한 제공자 우선 |
| `"throughput"` | 초당 토큰 속도가 가장 빠른 제공자 우선 |
| `"latency"` | 첫 토큰 생성까지의 시간이 가장 짧은 제공자 우선 |

```yaml
provider_routing:
  sort: "price"
```

### `only`

제공자 이름의 화이트리스트입니다. 설정하면 이 제공자들**만** 사용됩니다. 다른 모든 제공자는 제외됩니다.

```yaml
provider_routing:
  only:
    - "Anthropic"
    - "Google"
```

### `ignore`

제공자 이름의 블랙리스트입니다. 이 제공자들은 가장 저렴하거나 빠른 옵션을 제공하더라도 **절대** 사용되지 않습니다.

```yaml
provider_routing:
  ignore:
    - "Together"
    - "DeepInfra"
```

### `order`

명시적인 우선순위 순서입니다. 먼저 나열된 제공자가 선호됩니다. 나열되지 않은 제공자는 대체(fallback) 수단으로 사용됩니다.

```yaml
provider_routing:
  order:
    - "Anthropic"
    - "Google"
    - "AWS Bedrock"
```

### `require_parameters`

`true`일 때, OpenRouter는 요청의 **모든** 파라미터(`temperature`, `top_p`, `tools` 등)를 지원하는 제공자에게만 라우팅합니다. 이는 조용히 파라미터가 무시되는 것을 방지합니다.

```yaml
provider_routing:
  require_parameters: true
```

### `data_collection`

제공자가 학습을 위해 프롬프트를 사용할 수 있는지 여부를 제어합니다. 옵션은 `"allow"` 또는 `"deny"`입니다.

```yaml
provider_routing:
  data_collection: "deny"
```

## 실제 예시

### 비용 최적화

사용 가능한 가장 저렴한 제공자로 라우팅합니다. 대용량 사용 및 개발에 좋습니다:

```yaml
provider_routing:
  sort: "price"
```

### 속도 최적화

대화형 사용을 위해 지연 시간이 짧은 제공자를 우선시합니다:

```yaml
provider_routing:
  sort: "latency"
```

### 처리량 최적화

초당 토큰 속도가 중요한 긴 텍스트 생성에 가장 좋습니다:

```yaml
provider_routing:
  sort: "throughput"
```

### 특정 제공자로 제한

일관성을 위해 모든 요청이 특정 제공자를 통과하도록 보장합니다:

```yaml
provider_routing:
  only:
    - "Anthropic"
```

### 특정 제공자 피하기

사용하고 싶지 않은 제공자를 제외합니다(예: 데이터 프라이버시 목적):

```yaml
provider_routing:
  ignore:
    - "Together"
    - "Lepton"
  data_collection: "deny"
```

### 대체 수단이 있는 선호 순서

선호하는 제공자를 먼저 시도하고, 사용할 수 없는 경우 다른 제공자로 대체합니다:

```yaml
provider_routing:
  order:
    - "Anthropic"
    - "Google"
  require_parameters: true
```

## 작동 방식

제공자 라우팅 환경 설정은 모든 API 호출에서 `extra_body.provider` 필드를 통해 OpenRouter API에 전달됩니다. 이는 다음 두 가지 모드에 모두 적용됩니다:

- **CLI 모드** — `~/.hermes/config.yaml`에 구성되어 시작 시 로드됨
- **게이트웨이 모드** — 동일한 구성 파일이 게이트웨이 시작 시 로드됨

라우팅 구성은 `config.yaml`에서 읽혀 `AIAgent`를 생성할 때 매개변수로 전달됩니다:

```
providers_allowed  ← provider_routing.only에서 가져옴
providers_ignored  ← provider_routing.ignore에서 가져옴
providers_order    ← provider_routing.order에서 가져옴
provider_sort      ← provider_routing.sort에서 가져옴
provider_require_parameters ← provider_routing.require_parameters에서 가져옴
provider_data_collection    ← provider_routing.data_collection에서 가져옴
```

:::tip
여러 옵션을 결합할 수 있습니다. 예를 들어 가격순으로 정렬하되 특정 제공자를 제외하고 파라미터 지원을 요구할 수 있습니다:

```yaml
provider_routing:
  sort: "price"
  ignore: ["Together"]
  require_parameters: true
  data_collection: "deny"
```
:::

## 기본 동작

`provider_routing` 섹션이 구성되지 않은 경우(기본값), OpenRouter는 자체 기본 라우팅 로직을 사용하여 일반적으로 비용과 가용성 사이의 균형을 자동으로 맞춥니다.

:::tip 제공자 라우팅과 대체 모델 비교 (Provider Routing vs. Fallback Models)
제공자 라우팅은 OpenRouter 내의 **어떤 하위 제공자**가 요청을 처리할지 제어합니다. 기본 모델이 실패할 때 아예 완전히 다른 제공자로 자동 장애 조치(failover)를 하려면 [대체 제공자 (Fallback Providers)](/user-guide/features/fallback-providers)를 참조하세요.
:::
