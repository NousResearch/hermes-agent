---
sidebar_position: 16
title: "持久目标"
description: "设置一个持续目标，让 Hermes 跨轮次持续工作直到完成。我们对 Ralph loop 的实现。"
---

# 持久目标（`/goal`）

`/goal` 为 Hermes 设置一个跨轮次持续存在的目标。每轮结束后，一个轻量级裁判模型会检查目标是否已被助手的最新回复满足。若未满足，Hermes 会自动将一条续行 prompt（提示词）注入同一会话并继续工作——直到目标达成、你暂停或清除目标，或者轮次预算耗尽为止。

这是我们对 **Ralph loop** 的实现，直接受 Eric Traut（OpenAI）在 [Codex CLI 0.128.0 的 `/goal`](https://github.com/openai/codex) 中的启发。核心思路——跨轮次保持目标存活、不达成不停止——源自他们。此处的实现是独立的，并已适配 Hermes 的架构。

## 适用场景

当你希望 Hermes 自主迭代、无需每轮重新提示时，使用 `/goal`：

- "修复 `src/` 中的所有 lint 错误，并验证 `ruff check` 通过"
- "从仓库 Y 移植功能 X，包含测试，并让 CI 变绿"
- "调查为何会话 ID 有时在中途压缩时发生漂移，并撰写报告"
- "构建一个小型 CLI，按 EXIF 日期重命名文件，然后对 photos/ 文件夹进行测试"

只需一轮即可完成的任务不需要 `/goal`。*否则你需要说三次"继续"* 的任务，才是它的用武之地。

## 快速开始

```
/goal Fix every failing test in tests/hermes_cli/ and make sure scripts/run_tests.sh passes for that directory
```

你将看到：

1. **目标已接受** — `⊙ Goal set (20-turn budget): <your goal>`
2. **第 1 轮运行** — Hermes 开始工作，就像你发送了一条普通消息一样。
3. **裁判运行** — 轮次结束后，裁判模型判定 `done` 或 `continue`。
4. **若需要则触发循环** — 若为 `continue`，你将看到 `↻ Continuing toward goal (1/20): <judge's reason>`，Hermes 自动执行下一步。
5. **终止** — 最终你会看到 `✓ Goal achieved: <reason>` 或 `⏸ Goal paused — N/20 turns used`。

## 命令

| 命令 | 功能 |
|---|---|
| `/goal <text>` | 设置（或替换）持续目标。立即启动第一轮，无需再发送单独消息。 |
| `/goal` 或 `/goal status` | 显示当前目标、状态及已用轮次。 |
| `/goal pause` | 停止自动续行循环，但不清除目标。 |
| `/goal resume` | 恢复循环（将轮次计数器重置为零）。 |
| `/goal clear` | 完全删除目标。 |

在 CLI 及所有 gateway 平台（Telegram、Discord、Slack、Matrix、Signal、WhatsApp、SMS、iMessage、Webhook、API server 以及 Web 控制台）上行为完全一致。

## 目标进行中追加条件：`/subgoal`

目标激活期间，你可以使用 `/subgoal <text>` 追加额外的验收条件，而不会重置循环。每次调用会向目标的子目标列表添加一个编号条目；下一轮 agent 看到的**续行 prompt** 包含原始目标以及一个"用户在循环中途追加的额外条件"块，**裁判 prompt** 也会被重写，使裁判在判定时必须考虑所有子目标——只有原始目标**和**所有子目标均满足时，目标才会被标记为完成。

| 命令 | 功能 |
|---|---|
| `/subgoal <text>` | 向活跃目标追加一个新条件。需要有活跃的 `/goal`。 |
| `/subgoal`（无参数） | 显示当前编号子目标列表。 |
| `/subgoal remove <N>` | 删除第 N 个子目标（从 1 开始计数）。 |
| `/subgoal clear` | 删除所有子目标，但保留原始目标。 |

子目标与目标一起持久化存储在 `SessionDB.state_meta` 中，因此在 `/resume` 后依然有效。设置新的 `/goal <text>` 会替换目标并清空子目标列表；`/goal clear` 同样如此。

当你启动一个循环（"修复失败的测试"）后，中途发现还需要"为刚修复的 bug 添加回归测试"时，使用此功能——`/subgoal add a regression test` 可在不中断运行循环的情况下收紧成功条件。

## 行为细节

### 裁判

每轮结束后，Hermes 会调用一个辅助模型，传入：

- 持续目标文本
- agent 最新的最终回复（最后约 4 KB 文本）
- 一个系统 prompt，要求裁判以严格 JSON 格式回复：`{"done": <bool>, "reason": "<one-sentence rationale>"}`

裁判刻意保守：只有当回复**明确**确认目标已完成、最终交付物已清晰产出，或目标不可达/被阻塞时（视为 DONE 并附带阻塞原因，以免在不可能的任务上消耗预算），才会将目标标记为 `done`。

### 失败开放语义

若裁判出错（网络抖动、响应格式错误、辅助客户端不可用），Hermes 将判定视为 `continue`——损坏的裁判不会阻塞进度。**轮次预算**才是真正的兜底机制。

### 轮次预算

默认为 20 个续行轮次（`config.yaml` 中的 `goals.max_turns`）。预算耗尽时，Hermes 自动暂停并告知你如何继续：

```
⏸ Goal paused — 20/20 turns used. Use /goal resume to keep going, or /goal clear to stop.
```

`/goal resume` 将计数器重置为零，你可以按可控的块继续推进。

### 基于进展的自动续跑

一个仍在取得进展的通宵目标，不应该等待你手动输入 `/goal resume`——而且这一步很容易被遗忘。开启 `goals.auto_extend` 后，预算耗尽会变成**进展审查检查点**，而不是无条件停止：

1. **审查。** 一个窗口的最后一轮结束后，Hermes 会询问辅助模型（与 `goal_judge` 同一条路由）：回复中是否体现出*具体的近期进展*（新的产出物、命令/测试输出、已完成的子步骤）以及明确的下一步。
2. **续跑。** 有具体进展即获得**同一会话内**的又一个完整窗口（`goals.max_turns`）：目标、完成契约、子目标标准与对话上下文全部保留。你会看到：

```
↻ Goal extended — the progress review granted another 20 turns (extension #1, 40/100 turns total): 3 of 4 modules ported; tests for the last one still failing
```

3. **完成 / 卡住。** 若审查认为目标已满足，则标记为完成；若工作已经**停滞**（回复只是重复同样的状态、复述之前的工作，或报告同一个阻塞）或目标**被阻塞**，则携带审查给出的原因暂停——停滞的循环绝不会被续跑。

与每轮裁判的失败开放（fail-open 到 `continue`）不同，进展审查是**失败关闭（fail-closed）**的：辅助模型报错、返回非 JSON 或给出无法使用的判定时，目标会带着该原因暂停，而不是在没有任何证据的情况下续跑。

`goals.max_total_turns` 是**单个目标**跨窗口与续跑可消耗轮次的总体上限——累计用量会显示在续跑通知和 `/goal status` 中（例如 `⊙ Goal (active, 3/20 turns, 43 total, 2 budget extensions): …`）。`/goal resume` 不会重置它，因此循环不会失控。

```yaml
goals:
  max_turns: 20          # 窗口大小，也是每次续跑授予的轮次数
  auto_extend: false     # 可选开启：在预算边界先做进展审查，而非直接暂停
  max_total_turns: 100   # 单个目标的累计轮次总上限（0 表示不设上限）
```

该功能是**可选开启（opt-in）**的——`auto_extend: false`（默认值）时预算行为与之前完全一致；如果你的预算用于控制花费而非防止疲劳，这正是你想要的。预算边界上仍处于失败状态的质量门禁（quality gate）依旧会暂停目标（门禁有自己的重试上限，且是验收标准尚未达成的确定性证据）。

### 用户消息始终优先

目标激活期间，你发送的任何真实消息都优先于续行循环。在 CLI 上，你的消息会在队列中的续行消息之前进入 `_pending_input`；在 gateway 上，它以同样的方式通过适配器 FIFO 传递。你的轮次结束后裁判会再次运行——因此如果你的消息恰好完成了目标，裁判会捕获到并停止循环。

### 运行中安全性（gateway）

agent 正在运行时，`/goal status`、`/goal pause` 和 `/goal clear` 可以安全执行——它们只操作控制面状态，不会中断当前轮次。在运行中设置**新**目标（`/goal <new text>`）会被拒绝，并提示你先执行 `/stop`，以防旧续行与新目标产生竞争。

### 持久化

目标状态存储在 `SessionDB.state_meta` 中，以 `goal:<session_id>` 为键。这意味着 `/resume` 可以从你离开的地方继续——设置目标、合上笔记本、明天回来、执行 `/resume`，目标依然完好如初（活跃、暂停或已完成）。

### Prompt 缓存

续行 prompt 是一条以用户角色追加到历史记录中的普通消息。它**不会**修改系统 prompt、切换工具集，也不会以任何使 Hermes prompt 缓存失效的方式改动对话。运行一个 20 轮目标，在缓存层面与 20 轮普通对话的开销相同。

## 配置

在 `~/.hermes/config.yaml` 中添加：

```yaml
goals:
  # Hermes 自动暂停并要求你执行 /goal resume 之前的最大续行轮次。
  # 默认 20。若想要更紧凑的循环可降低此值；
  # 长时间重构可适当提高。
  max_turns: 20
  # 在预算边界先做进展审查而非无条件暂停（默认关闭）。
  auto_extend: false
  # 单个目标累计轮次的总体上限（0 表示不设上限）。
  max_total_turns: 100
```

### 选择裁判模型

裁判使用 `goal_judge` 辅助任务。默认情况下，它解析为你的主模型（参见[辅助模型](../configuration.md#auxiliary-models)）。若想将裁判路由到廉价快速的模型以降低成本，可添加覆盖配置：

```yaml
auxiliary:
  goal_judge:
    provider: openrouter
    model: google/gemini-3-flash-preview
```

裁判调用量小（约 200 个输出 token），每轮运行一次，因此廉价快速的模型通常是正确选择。

## 示例演练

```
You: /goal Create four files /tmp/note_{1..4}.txt, one per turn, each containing its number as text

  ⊙ Goal set (20-turn budget): Create four files /tmp/note_{1..4}.txt, one per turn, each containing its number as text

Hermes: Creating /tmp/note_1.txt now.
  💻 echo "1" > /tmp/note_1.txt   (0.1s)
  I've created /tmp/note_1.txt with the content "1". I'll continue with the remaining files on the next turn as you specified.

  ↻ Continuing toward goal (1/20): Only 1 of 4 files has been created; 3 files remain.

Hermes: [Continuing toward your standing goal]
  💻 echo "2" > /tmp/note_2.txt   (0.1s)
  Created /tmp/note_2.txt. Two more to go.

  ↻ Continuing toward goal (2/20): 2 of 4 files created; 2 remain.

Hermes: [Continuing toward your standing goal]
  💻 echo "3" > /tmp/note_3.txt   (0.1s)
  Created /tmp/note_3.txt.

  ↻ Continuing toward goal (3/20): 3 of 4 files created; 1 remains.

Hermes: [Continuing toward your standing goal]
  💻 echo "4" > /tmp/note_4.txt   (0.1s)
  All four files have been created: /tmp/note_1.txt through /tmp/note_4.txt, each containing its number.

  ✓ Goal achieved: All four files were created with the specified content, completing the goal.

You: _
```

四轮，一次 `/goal` 调用，你零次"继续"提示。

## 裁判判断有误时

没有裁判是完美的。需注意两种失败模式：

**假阴性——目标实际已完成，裁判却说继续。** 轮次预算会兜底。你会看到 `⏸ Goal paused`，可以执行 `/goal clear` 或直接发送新消息。

**假阳性——工作尚未完成，裁判却说已完成。** 你会看到 `✓ Goal achieved`，但你知道实际情况并非如此。发送后续消息继续，或更精确地重新设置目标：`/goal <更具体的文本>`。裁判的系统 prompt 刻意保守，以使假阳性比假阴性更少出现。

如果你觉得某次裁判判定不可信，`↻ Continuing toward goal` 或 `✓ Goal achieved` 行中的原因文本会告诉你裁判看到了什么。这通常足以诊断出是目标文本存在歧义，还是模型的回复有问题。

## 致谢

`/goal` 是 Hermes 对 **Ralph loop** 模式的实现。面向用户的设计——跨轮次保持目标存活、不达成不停止，以及创建/暂停/恢复/清除控制——由 OpenAI Codex 团队的 Eric Traut 在 [Codex CLI 0.128.0](https://github.com/openai/codex) 中推广并落地。我们的实现是独立的（中央 `CommandDef` 注册表、`SessionDB.state_meta` 持久化、辅助客户端裁判、gateway 侧的适配器 FIFO 续行），但这个想法源自他们。功劳归于应得之人。