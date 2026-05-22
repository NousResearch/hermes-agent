---
title: "提供商路由"
description: "配置 OpenRouter 提供商偏好，以优化成本、速度或质量。"
sidebar_label: Provider Routing
sidebar_position: 7
---

# 提供商路由

当使用 [OpenRouter](https://openrouter.ai) 作为你的 LLM 提供商时，Hermes Agent 支持**提供商路由**——对哪个底层 AI 提供商处理你的请求以及它们如何被优先排序进行细粒度控制。

OpenRouter 将请求路由到许多提供商（例如 Anthropic、Google、AWS Bedrock、Together AI）。提供商路由让你能够针对成本、速度、质量进行优化，或强制执行特定的提供商要求。

## 配置

在你的 `~/.hermes/config.yaml` 中添加一个 `provider_routing` 节：

```yaml
provider_routing:
  sort: "price"           # 如何对提供商进行排序
  only: []                # 白名单：只使用这些提供商
  ignore: []              # 黑名单：绝不使用这些提供商
  order: []               # 显式的提供商优先级顺序
  require_parameters: false  # 只使用支持所有参数的提供商
  data_collection: null   # 控制数据收集（"allow" 或 "deny"）
```

:::info
提供商路由仅在使用 OpenRouter 时生效。对于直接提供商连接（例如直接连接到 Anthropic API）没有影响。
:::

## 选项

### `sort`

控制 OpenRouter 如何为你的请求对可用提供商进行排序。

| 值 | 描述 |
|-------|-------------|
| `"price"` | 最便宜的提供商优先 |
| `"throughput"` | 每秒生成 token 最快的优先 |
| `"latency"` | 首 token 时间最低的优先 |

```yaml
provider_routing:
  sort: "price"
```

### `only`

提供商名称白名单。设置后，**只**会使用这些提供商。所有其他提供商都被排除。

```yaml
provider_routing:
  only:
    - "Anthropic"
    - "Google"
```

### `ignore`

提供商名称黑名单。这些提供商将**永远不会**被使用，即使它们提供了最便宜或最快的选项。

```yaml
provider_routing:
  ignore:
    - "Together"
    - "DeepInfra"
```

### `order`

显式优先级顺序。列在前面的提供商优先。未列出的提供商用作后备。

```yaml
provider_routing:
  order:
    - "Anthropic"
    - "Google"
    - "AWS Bedrock"
```

### `require_parameters`

当为 `true` 时，OpenRouter 将只把请求路由到支持你请求中**所有**参数（如 `temperature`、`top_p`、`tools` 等）的提供商。这可以避免参数被静默丢弃。

```yaml
provider_routing:
  require_parameters: true
```

### `data_collection`

控制提供商是否可以将你的提示用于训练。选项为 `"allow"` 或 `"deny"`。

```yaml
provider_routing:
  data_collection: "deny"
```

## 实用示例

### 优化成本

路由到最便宜的可用提供商。适合高用量和开发场景：

```yaml
provider_routing:
  sort: "price"
```

### 优化速度

为交互式使用优先选择低延迟提供商：

```yaml
provider_routing:
  sort: "latency"
```

### 优化吞吐量

最适合长文本生成，其中每秒 token 数很重要：

```yaml
provider_routing:
  sort: "throughput"
```

### 锁定特定提供商

确保所有请求都通过特定提供商以保持一致性：

```yaml
provider_routing:
  only:
    - "Anthropic"
```

### 避开特定提供商

排除你不想使用的提供商（例如出于数据隐私考虑）：

```yaml
provider_routing:
  ignore:
    - "Together"
    - "Lepton"
  data_collection: "deny"
```

### 带后备的优先顺序

先尝试你偏好的提供商，如果不可用则回退到其他提供商：

```yaml
provider_routing:
  order:
    - "Anthropic"
    - "Google"
  require_parameters: true
```

## 工作原理

提供商路由偏好通过每次 API 调用的 `extra_body.provider` 字段传递给 OpenRouter API。这适用于：

- **CLI 模式** —— 在 `~/.hermes/config.yaml` 中配置，启动时加载
- **Gateway 模式** —— 相同的配置文件，gateway 启动时加载

路由配置从 `config.yaml` 中读取，并在创建 `AIAgent` 时作为参数传递：

```
providers_allowed  ← 来自 provider_routing.only
providers_ignored  ← 来自 provider_routing.ignore
providers_order    ← 来自 provider_routing.order
provider_sort      ← 来自 provider_routing.sort
provider_require_parameters ← 来自 provider_routing.require_parameters
provider_data_collection    ← 来自 provider_routing.data_collection
```

:::tip
你可以组合多个选项。例如，按价格排序但排除某些提供商并要求参数支持：

```yaml
provider_routing:
  sort: "price"
  ignore: ["Together"]
  require_parameters: true
  data_collection: "deny"
```
:::

## 默认行为

当未配置 `provider_routing` 节时（默认情况），OpenRouter 使用其自己的默认路由逻辑，通常会自动平衡成本和可用性。

:::tip 提供商路由 vs. 备用模型
提供商路由控制 OpenRouter **内部**的哪些子提供商处理你的请求。对于主模型失败时自动故障转移到完全不同的提供商，请参阅[备用提供商](./fallback-providers)。
:::
