---
title: 提供者路由
description: 配置 OpenRouter 提供者偏好以优化成本、速度或质量。
sidebar_label: 提供者路由
sidebar_position: 7
---

# 提供者路由

使用 [OpenRouter](https://openrouter.ai) 作为 LLM 提供者时，Hermes Agent 支持**提供者路由** — 对哪些底层 AI 提供者处理你的请求以及它们的优先级进行精细控制。

OpenRouter 将请求路由到许多提供者（例如 Anthropic、Google、AWS Bedrock、Together AI）。提供者路由允许你优化成本、速度、质量，或强制执行特定的提供者要求。

## 配置

在 `~/.hermes/config.yaml` 中添加 `provider_routing` 部分：

```yaml
provider_routing:
  sort: "price"           # 如何对提供者排名
  only: []                # 白名单：仅使用这些提供者
  ignore: []              # 黑名单：永远不使用这些提供者
  order: []               # 明确的提供者优先级顺序
  require_parameters: false  # 仅使用支持所有参数的提供者
  data_collection: null   # 控制数据收集（"allow" 或 "deny"）
```

:::info
提供者路由仅在使用 OpenRouter 时适用。对直接提供者连接（如直接连接到 Anthropic API）没有效果。
:::

## 选项

### `sort`

控制 OpenRouter 如何为你的请求排名可用提供者。

| 值 | 描述 |
|-------|-------------|
| `"price"` | 最便宜的提供者优先 |
| `"throughput"` | 每秒最快 token 数优先 |
| `"latency"` | 最低首 token 延迟优先 |

```yaml
provider_routing:
  sort: "price"
```

### `only`

提供者名称白名单。设置后，**仅**使用这些提供者。所有其他提供者被排除。

```yaml
provider_routing:
  only:
    - "Anthropic"
    - "Google"
```

### `ignore`

提供者名称黑名单。这些提供者将**永远**不被使用，即使它们提供最便宜或最快的选项。

```yaml
provider_routing:
  ignore:
    - "Together"
    - "DeepInfra"
```

### `order`

明确的优先级顺序。列在前面的提供者优先。未列出的提供者作为回退使用。

```yaml
provider_routing:
  order:
    - "Anthropic"
    - "Google"
    - "AWS Bedrock"
```

### `require_parameters`

为 `true` 时，OpenRouter 仅将请求路由到支持你请求中**所有**参数（如 `temperature`、`top_p`、`tools` 等）的提供者。这避免了静默参数丢失。

```yaml
provider_routing:
  require_parameters: true
```

### `data_collection`

控制提供者是否可以使用你的提示进行训练。选项为 `"allow"` 或 `"deny"`。

```yaml
provider_routing:
  data_collection: "deny"
```

## 实用示例

### 优化成本

路由到最便宜的可用提供者。适用于高容量使用和开发：

```yaml
provider_routing:
  sort: "price"
```

### 优化速度

优先选择低延迟提供者用于交互使用：

```yaml
provider_routing:
  sort: "latency"
```

### 优化吞吐量

最适合每秒 token 数重要的长文本生成：

```yaml
provider_routing:
  sort: "throughput"
```

### 锁定特定提供者

确保所有请求都通过特定提供者以保持一致性：

```yaml
provider_routing:
  only:
    - "Anthropic"
```

### 避免特定提供者

排除你不想使用的提供者（例如出于数据隐私考虑）：

```yaml
provider_routing:
  ignore:
    - "Together"
    - "Lepton"
  data_collection: "deny"
```

### 带回退的首选顺序

先尝试首选提供者，如果不可用则回退到其他提供者：

```yaml
provider_routing:
  order:
    - "Anthropic"
    - "Google"
  require_parameters: true
```

## 工作原理

提供者路由偏好通过每次 API 调用的 `extra_body.provider` 字段传递给 OpenRouter API。这适用于：

- **CLI 模式** — 在 `~/.hermes/config.yaml` 中配置，启动时加载
- **网关模式** — 相同的配置文件，网关启动时加载

路由配置从 `config.yaml` 读取并在创建 `AIAgent` 时作为参数传递：

```
providers_allowed  ← 来自 provider_routing.only
providers_ignored  ← 来自 provider_routing.ignore
providers_order    ← 来自 provider_routing.order
provider_sort      ← 来自 provider_routing.sort
provider_require_parameters ← 来自 provider_routing.require_parameters
provider_data_collection    ← 来自 provider_routing.data_collection
```

:::tip
你可以组合多个选项。例如按价格排序但排除某些提供者并要求参数支持：

```yaml
provider_routing:
  sort: "price"
  ignore: ["Together"]
  require_parameters: true
  data_collection: "deny"
```
:::

## 默认行为

当没有配置 `provider_routing` 部分时（默认），OpenRouter 使用自己的默认路由逻辑，通常会自动平衡成本和可用性。

:::tip 提供者路由 vs 回退模型
提供者路由控制 OpenRouter 内部的**子提供者**如何处理你的请求。当主要模型失败时自动切换到完全不同的提供者，请参见 [回退提供者](/docs/user-guide/features/fallback-providers)。
:::
