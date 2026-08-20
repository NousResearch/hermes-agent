---
sidebar_position: 7
title: "混合代理（Mixture of Agents）"
description: "创建命名的 MoA 预设，作为 Mixture of Agents Provider 下的可选模型"
---

# 混合代理（Mixture of Agents）

混合代理（Mixture of Agents，MoA）是一个虚拟模型 Provider。每个命名的 MoA 预设作为 `moa` Provider 下的可选模型出现。

当你选择一个 MoA 预设时，预设的聚合器（Aggregator）是实际模型 —— 它是编写助手响应并发出工具调用的模型。参考模型（Reference Models）先行运行，为聚合器提供分析。

当一个困难任务受益于多模型视角，但仍然需要 Hermes 正常的 Agent 循环时，可以使用 MoA：工具调用、后续迭代、中断、转录持久化，以及与其他消息相同的会话上下文。

## 选择 MoA 预设作为你的模型

你可以通过正常的模型选择器界面选择预设：

```bash
/model default --provider moa
/model review --provider moa
```

MoA 预设在**每个 Hermes 界面**上都可选，因为 MoA 是模型系统中的普通 Provider：

- **CLI / 网关 / TUI `/model`** —— `/model <preset> --provider moa`，或 `/model --provider moa` 选择默认预设。当名称完全匹配已配置的预设时，简单的 `/model <preset>` 也有效。
- **`hermes model`** 和 **Dashboard 模型选择器** —— 会出现一个 `Mixture of Agents` Provider 行，预设名称作为其模型。
- **桌面 GUI 应用** —— 模型下拉菜单显示 `MoA presets` 部分；选择一个（`MoA: <preset>`）会将活动模型切换为该预设。桌面设置面板也可以创建和编辑预设。

因此，已配置的预设会出现在你选择其他任何模型的地方。

## 斜杠命令快捷方式

`/moa` 是一次性便捷语法糖。它使用**默认** MoA 预设运行单个提示，然后恢复你之前使用的模型：

```bash
/moa design and implement a migration plan for this flaky test cluster
```

Hermes 为该轮次临时切换到默认 MoA 预设，发送提示，然后恢复你之前的模型。整个参数就是提示 —— `/moa` 不再将其解释为预设名称。

```bash
/moa
```

不带提示的 `/moa` 仅打印使用方法。

要在剩余会话中**切换**到 MoA 预设，请从模型选择器中选择它 —— MoA 预设出现在每个模型选择界面的 `Mixture of Agents` Provider 下（见上文）。`/moa` 刻意不是模型切换，因此普通提示永远不会意外更改你的模型。

## 在 Agent 循环中的工作原理

当选择 `moa` Provider 时，对于每个主模型调用，Hermes 会：

1. 按名称解析选定的预设；
2. 运行配置的参考模型，不携带工具 schema（它们只接收对话的用户/助手文本 —— 不包括 Hermes 系统提示或工具调用记录 —— 因此参考调用保持低成本并避免严格 Provider 的拒绝）；
3. 将参考输出附加为聚合器的私有上下文；
4. 使用正常的 Hermes 工具 schema 调用配置的聚合器；
5. 将聚合器响应视为真正的模型响应；
6. 如果聚合器调用了工具，Hermes 正常执行这些工具；
7. 在下一次模型迭代中，相同的 MoA 流程在更新后的对话上再次运行，包括工具结果。

由于 MoA 是通过正常模型系统选择的，它自动与 `/goal`、网关会话、TUI 会话和桌面聊天组合使用。

## 配置预设

你可以通过以下方式配置命名的 MoA 预设：

- Dashboard → 模型 → 模型设置 → Mixture of Agents
- 桌面应用 → 设置 → 模型 → Mixture of Agents
- `hermes moa configure [name]`
- `config.yaml`

配置存储了显式的 Provider/模型对，因此你可以混合 Provider 并使用同一 Provider 的多个模型：

```yaml
moa:
  default_preset: default
  presets:
    default:
      reference_models:
        - provider: openai-codex
          model: gpt-5.5
        - provider: openrouter
          model: deepseek/deepseek-v4-pro
      aggregator:
        provider: openrouter
        model: anthropic/claude-opus-4.8
      # Optional: pin sampling temperatures. When omitted (the default),
      # temperature is NOT sent and each model uses its provider default —
      # the same behavior as a single-model Hermes agent.
      # reference_temperature: 0.6
      # aggregator_temperature: 0.4
      max_tokens: 4096
      enabled: true
```

默认预设：

- 参考模型：`openai-codex:gpt-5.5`
- 参考模型：`openrouter:deepseek/deseek-v4-pro`
- 聚合器/实际模型：`openrouter:anthropic/claude-opus-4.8`

### 通过 `reference_max_tokens` 调优顾问速度

每轮次，MoA 并行运行参考模型（顾问），然后聚合器执行。顾问生成是每轮次延迟的主要来源 —— 轮次挂钟时间与顾问输出的 token 数强相关，因为轮次等待最慢的顾问完成书写。默认情况下顾问**没有上限**（`reference_max_tokens` 未设置），因此它们可能写出冗长的、论文长度的建议。

在预设上设置 `reference_max_tokens` 来限制顾问输出，提供简洁的建议。聚合器只需要每个顾问判断的要点，因此设置上限（例如 `600`）能显著减少每轮次挂钟时间，对质量影响很小。它仅限制**顾问** —— 聚合器的输出（用户可见的答案）永远不会被限制。

```yaml
moa:
  presets:
    fast:
      reference_models:
        - provider: openrouter
          model: anthropic/claude-opus-4.8
        - provider: openrouter
          model: openai/gpt-5.5
      aggregator:
        provider: openrouter
        model: anthropic/claude-opus-4.8
      reference_max_tokens: 600   # concise advice → faster turns
```

不设置（或 `0`/空）保持之前的无上限行为。

### 按槽位的推理力度

参考和聚合器槽位也可以设置 `reasoning_effort`。当你想让同一模型以不同深度贡献，或聚合器应该比顾问参考模型思考更深时使用。有效值与 Hermes 的正常推理控制一致：`none`、`minimal`、`low`、`medium`、`high`、`xhigh` 和 `max`。

```yaml
moa:
  presets:
    deep_review:
      reference_models:
        - provider: openai-codex
          model: gpt-5.6-sol
          reasoning_effort: low
        - provider: openai-codex
          model: gpt-5.6-sol
          reasoning_effort: xhigh
        - provider: xai-oauth
          model: grok-4.5
      aggregator:
        provider: openai-codex
        model: gpt-5.6-sol
        reasoning_effort: high
```

省略 `reasoning_effort` 则使用该槽位的 Provider/Hermes 默认值。

## 终端预设管理

```bash
hermes moa list
hermes moa configure              # update the default preset
hermes moa configure review       # create or update a named preset
hermes moa delete review
```

## 基准测试

在 HermesBench 上，一个双模型 MoA 预设 —— `claude-opus-4.8` 聚合 `gpt-5.5` 参考模型 —— 得分超过任一模型单独运行：

| 模型 | HermesBench 分数 |
|---|---|
| **Opus 聚合器（opus-4.8 + gpt-5.5 参考）— MoA** | **0.8202** |
| `anthropic/claude-opus-4.8` | 0.7607 |
| `openai/gpt-5.5` | 0.7412 |

MoA 配置比其最强组件（opus-4.8）高出约 6 分，证实了聚合第二种视角能在困难任务上提升质量，而不仅仅是简单平均。

## 提示缓存

MoA 的设计确保**主对话的提示缓存永远不会被破坏**。选择 MoA 预设是正常的模型选择：它不会修改过去的上下文、更换工具集或在对话中重建系统提示。你的对话历史、系统提示和工具 schema 保持字节级稳定，因此其他模型依赖的缓存前缀会与普通模型一样被精确保持。切换到或离开 MoA 预设的缓存失效成本与任何其他 `/model` 切换相同 —— 不会更多。

两种内部调用类型都正常缓存：

- **参考模型**接收对话的裁剪视图（系统提示和工具记录被剥离 —— 参见上面的循环）。因为该视图是稳定历史的稳定函数，参考模型的提示前缀在迭代间重复并正常缓存。参考是无工具的简短咨询调用。
- **聚合器**是实际模型。参考输出被附加到最新用户轮次的*末尾*作为私有指导。因为这些文本位于尾部 —— 在整个稳定前缀（系统提示 + 先前历史）之下 —— 它不会使任何缓存前缀失效：聚合器在注入点以上获得缓存命中，只有新附加的尾部是全新的。这与每个正常轮次的行为完全一致，其中每条新的用户消息也是未缓存的尾部 token。

因此 MoA 不会牺牲任一调用类型的提示缓存。它唯一的实际成本是每次迭代的额外参考调用 —— 你为多种模型视角付费，而不是为损坏的缓存付费。与 Hermes 其余部分共享的长期对话前缀完全完好。

## 注意事项

- MoA 不再列在 `hermes tools` 下；没有需要启用的 `moa` 工具集。
- 在预设上设置 `enabled: false` 会禁用该预设的参考扇出：聚合器单独执行，就像你选择它作为普通模型一样。这是在 Dashboard 和桌面设置中显示的每个预设的开关。
- 预设的聚合器不能是另一个 MoA 预设。递归 MoA 树被刻意阻止。
- 一个参考模型的凭据失败不会中止轮次。Hermes 将失败包含在参考上下文中，继续处理返回结果的模型。
- MoA 会增加模型调用次数。一次模型迭代可能涉及多次参考调用加上聚合器调用。
