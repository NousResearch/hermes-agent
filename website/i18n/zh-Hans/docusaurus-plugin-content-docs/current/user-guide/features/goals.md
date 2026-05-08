---
sidebar_position: 16
title: "持久目标"
description: "设置一个持续目标，让 Hermes 跨轮次持续工作直到完成。我们的 Ralph 循环实现。"
---

# 持久目标（`/goal`）

`/goal` 给 Hermes 一个跨轮次存活的持续目标。在每轮之后，一个轻量级评判模型检查目标是否由助手的最后响应满足。如果不是，Hermes 自动将延续提示反馈到同一会话并继续工作——直到目标达成、你暂停或清除它，或者轮次预算耗尽。

这是我们对 **Ralph 循环**的实现，直接受 [Codex CLI 0.128.0 的 `/goal`](https://github.com/openai/codex)（作者 Eric Traut，OpenAI）启发。核心思想——跨轮次保持目标活跃，在达成之前不停止——是他们的。这里的实现是独立的并适应了 Hermes 的架构。

## 何时使用

对以下任务使用 `/goal`：你希望 Hermes 自己迭代而无需每轮重新提示：

- "修复 `src/` 中的每个 lint 错误并验证 `ruff check` 通过"
- "将功能 X 从仓库 Y 移植过来，包括测试，并让 CI 变绿"
- "调查为什么会话 ID 有时在压缩中途漂移，并撰写报告"
- "构建一个小型 CLI 按 EXIF 日期重命名文件，然后用 photos/ 文件夹测试它"

代理做一轮然后停止的任务不需要 `/goal`。如果你*本来必须说'继续'三次*的任务正是 `/goal` 的亮点。

## 快速开始

```
/goal Fix every failing test in tests/hermes_cli/ and make sure scripts/run_tests.sh passes for that directory
```

你会看到：

1. **目标被接受** — `⊙ Goal set (20-turn budget): <your goal>`
2. **第一轮运行** — Hermes 像你将目标作为正常消息发送一样开始工作。
3. **评判运行** — 轮次后，评判模型决定 `done` 或 `continue`。
4. **循环根据需要触发** — 如果 `continue`，你会看到 `↻ Continuing toward goal (1/20): <judge's reason>`，Hermes 自动采取下一步。
5. **终止** — 最终你看到 `✓ Goal achieved: <reason>` 或 `⏸ Goal paused — N/20 turns used`。

## 命令

| 命令 | 做什么 |
|---|---|
| `/goal <text>` | 设置（或替换）持续目标。立即启动第一轮，所以你不需要发送单独的消息。 |
| `/goal` 或 `/goal status` | 显示当前目标、状态和已用轮次。 |
| `/goal pause` | 停止自动延续循环而不清除目标。 |
| `/goal resume` | 恢复循环（将轮次计数器重置为零）。 |
| `/goal clear` | 完全删除目标。 |

在 CLI 和每个网关平台（Telegram、Discord、Slack、Matrix、Signal、WhatsApp、SMS、iMessage、Webhook、API 服务器和 Web 仪表板）上工作方式相同。

## 行为详情

### 评判器

每轮之后，Hermes 调用辅助模型：

- 持续目标文本
- 代理最近的最终响应（最后约 4 KB 文本）
- 一个告诉评判器用严格 JSON 回复的系统提示：`{"done": <bool>, "reason": "<one-sentence rationale>"}`

评判器故意保守：只有当响应**明确**确认目标完成时，当最终交付物明确产生时，或当目标无法实现/被阻止时（视为 DONE 并带阻止原因，这样我们不会在不可能的任务上浪费预算），才将目标标记为 `done`。

### 失败开放语义

如果评判器出错（网络故障、格式错误的响应、辅助客户端不可用），Hermes 将判决视为 `continue`——损坏的评判器永远不会阻碍进度。**轮次预算**是真正的后盾。

### 轮次预算

默认是 20 个延续轮次（`config.yaml` 中的 `goals.max_turns`）。当预算用完时，Hermes 自动暂停并准确告诉你如何继续：

```
⏸ Goal paused — 20/20 turns used. Use /goal resume to keep going, or /goal clear to stop.
```

`/goal resume` 将计数器重置为零，所以你可以按可衡量的块继续。

### 用户消息总是优先

当目标激活时你发送的任何真实消息优先于延续循环。在 CLI 上你的消息进入 `_pending_input` 位于排队的延续之前；在网关上它以相同方式通过适配器 FIFO。评判器在你的轮次之后再次运行——所以如果你的消息恰好完成了目标，评判器会捕捉到它并停止。

### 中途运行安全（网关）

当代理已经在运行时，`/goal status`、`/goal pause` 和 `/goal clear` 可以安全运行——它们只触碰控制平面状态而不中断当前轮次。在中途设置**新**目标（`/goal <new text>`）会被拒绝并显示消息让你先 `/stop`，这样旧的延续就不能与新的竞速。

### 持久化

目标状态存在于 `SessionDB.state_meta` 中，以 `goal:<session_id>` 为键。这意味着 `/resume` 从你停止的地方继续——设置一个目标，关闭笔记本，明天回来，`/resume`，目标仍然完全按你离开的方式站着（活跃、暂停或完成）。

### 提示缓存

延续提示是一个追加到历史中的普通用户角色消息。它**不会**改变系统提示、交换工具集或以任何使 Hermes 提示缓存失效的方式触碰对话。运行 20 轮目标在缓存方面的成本与 20 轮正常对话相同。

## 配置

添加到 `~/.hermes/config.yaml`：

```yaml
goals:
  # 在 Hermes 自动暂停并让你 /goal resume 之前的最长延续轮次。
  # 默认 20。如果你想要更紧密的循环则降低此值；
  # 对于长时间运行的重构则提高。
  max_turns: 20
```

### 选择评判模型

评判器使用 `goal_judge` 辅助任务。默认它解析到你的主模型（请参阅[辅助模型](/docs/user-guide/configuration#auxiliary-models)）。如果你想将评判路由到便宜的快速模型以降低成本，添加一个覆盖：

```yaml
auxiliary:
  goal_judge:
    provider: openrouter
    model: google/gemini-3-flash-preview
```

评判器调用很小（约 200 个输出令牌）且每轮运行一次，所以便宜的快速模型通常是正确选择。

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

四轮，一次 `/goal` 调用，零个"继续"提示。

## 评判器出错时

没有完美的评判器。注意两种失败模式：

**假阴性——目标实际完成时评判器说继续。** 轮次预算会捕捉到这一点。你会看到 `⏸ Goal paused`，可以 `/goal clear` 或只是发送新消息。

**假阳性——还有工作时评判器说完成。** 你会看到 `✓ Goal achieved`，但你更清楚。发送后续消息继续，或更精确地重新设置目标：`/goal <more specific text>`。评判器的系统提示故意保守，使假阳性比假阴性更少见。

如果你发现评判判决不令人信服，`↻ Continuing toward goal` 或 `✓ Goal achieved` 行中的原因文本准确告诉你评判器看到了什么。这通常足以诊断是目标文本模糊还是模型的响应模糊。

## 归属

`/goal` 是 Hermes 对 **Ralph 循环**模式的实现。用户面向的设计——跨轮次保持目标活跃，在达成之前不停止，带有创建/暂停/恢复/清除控制——由 OpenAI Codex 团队的 Eric Traut 在 [Codex CLI 0.128.0](https://github.com/openai/codex) 中推广和发布。我们的实现是独立的（中央 `CommandDef` 注册表、`SessionDB.state_meta` 持久化、辅助客户端评判器、网关端的适配器 FIFO 延续），但想法是他们的。功归于应得之处。
