---
title: Provider Routing
description: 配置 OpenRouter provider 偏好，以优化成本、速度或质量。
sidebar_label: Provider Routing
sidebar_position: 7
---

# Provider Routing

使用 [OpenRouter](https://openrouter.ai) 作为 LLM provider 时，Hermes Agent 支持 **provider routing**（提供商路由）——对哪些底层 AI provider 处理你的请求以及如何排列优先级进行精细控制。

OpenRouter 将请求路由到多个 provider（例如 Anthropic、Google、AWS Bedrock、Together AI）。Provider routing 让你可以针对成本、速度、质量进行优化，或强制指定特定 provider。

## 配置

在 `~/.hermes/config.yaml` 中添加 `provider_routing` 部分：

```yaml
provider_routing:
  sort: "price"           # 如何对 provider 排序
  only: []                # 白名单：仅使用这些 provider
  ignore: []              # 黑名单：永不使用这些 provider
  order: []               # 显式 provider 优先级顺序
  require_parameters: false  # 仅使用支持所有参数的 provider
  data_collection: null   # 控制数据收集（"allow" 或 "deny"）
```

:::info
Provider routing 仅在使用 OpenRouter 时生效。直接连接 provider（例如直接连接 Anthropic API）时无效。
:::

## 选项

### `sort`

控制 OpenRouter 如何对可用 provider 排序。

| 值 | 说明 |
|-------|-------------|
| `"price"` | 最便宜的 provider 优先 |
| `"throughput"` | 每秒 token 数最高的 provider 优先 |
| `"latency"` | 首 token 延迟最低的 provider 优先 |

```yaml
provider_routing:
  sort: "price"
```

### `only`

Provider 名称白名单。设置后，**仅**使用这些 provider，其余全部排除。

```yaml
provider_routing:
  only:
    - "Anthropic"
    - "Google"
```

### `ignore`

Provider 名称黑名单。这些 provider **永远不会**被使用，即使它们提供最低价格或最快速度。

```yaml
provider_routing:
  ignore:
    - "Together"
    - "DeepInfra"
```

### `order`

显式优先级顺序。列在前面的 provider 优先使用，未列出的 provider 作为备选。

```yaml
provider_routing:
  order:
    - "Anthropic"
    - "Google"
    - "AWS Bedrock"
```

### `require_parameters`

设为 `true` 时，OpenRouter 仅路由到支持请求中**所有**参数（如 `temperature`、`top_p`、`tools` 等）的 provider，避免参数被静默丢弃。

```yaml
provider_routing:
  require_parameters: true
```

### `data_collection`

控制 provider 是否可将你的 prompt（提示词）用于训练。可选值为 `"allow"` 或 `"deny"`。

```yaml
provider_routing:
  data_collection: "deny"
```

## 实用示例

### 优化成本

路由到最便宜的可用 provider，适合高频使用和开发场景：

```yaml
provider_routing:
  sort: "price"
```

### 优化速度

优先选择低延迟 provider，适合交互式使用：

```yaml
provider_routing:
  sort: "latency"
```

### 优化吞吐量

适合长文本生成等非常依赖 token 生成速度的场景：

```yaml
provider_routing:
  sort: "throughput"
```

### 锁定特定 Provider

确保所有请求都通过特定 provider 处理，以保证一致性：

```yaml
provider_routing:
  only:
    - "Anthropic"
```

### 排除特定 Provider

排除不希望使用的 provider（例如出于数据隐私考虑）：

```yaml
provider_routing:
  ignore:
    - "Together"
    - "Lepton"
  data_collection: "deny"
```

### 带备选的优先顺序

优先尝试首选 provider，不可用时回退到其他 provider：

```yaml
provider_routing:
  order:
    - "Anthropic"
    - "Google"
  require_parameters: true
```

## 工作原理

Provider routing 偏好通过每次 API 调用的 `extra_body.provider` 字段传递给 OpenRouter API，适用于以下两种模式：

- **CLI 模式** — 在 `~/.hermes/config.yaml` 中配置，启动时加载
- **Gateway 模式** — 同一配置文件，gateway 启动时加载

路由配置从 `config.yaml` 读取，并在创建 `AIAgent` 时作为参数传入：

```
providers_allowed  ← 来自 provider_routing.only
providers_ignored  ← 来自 provider_routing.ignore
providers_order    ← 来自 provider_routing.order
provider_sort      ← 来自 provider_routing.sort
provider_require_parameters ← 来自 provider_routing.require_parameters
provider_data_collection    ← 来自 provider_routing.data_collection
```

:::tip
可以组合使用多个选项。例如，按价格排序，同时排除某些 provider 并要求参数支持：

```yaml
provider_routing:
  sort: "price"
  ignore: ["Together"]
  require_parameters: true
  data_collection: "deny"
```
:::

## 默认行为

未配置 `provider_routing` 部分时（默认情况），OpenRouter 使用其自身的默认路由逻辑，通常会自动在成本和可用性之间取得平衡。

:::tip Provider Routing 与 Fallback Models
Provider routing 控制 OpenRouter **内部的子 provider** 如何处理你的请求。若需要在主模型失败时自动故障转移到完全不同的 provider，请参阅 [Fallback Providers](/user-guide/features/fallback-providers)。
:::