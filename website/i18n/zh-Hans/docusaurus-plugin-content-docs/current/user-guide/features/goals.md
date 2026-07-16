---
sidebar_position: 16
title: "持久目标"
description: "设置一个持续目标，让 Hermes 跨轮次持续工作直到完成。我们对 Ralph loop 的实现。"
---

# 持久目标（`/goal`）

`/goal` 为 Hermes 设置一个跨轮次持续存在的目标。执行任务的同一个主模型会在每轮结束前通过 `todo` 工具记录精确的结构化 `goal_outcome`。若仍需继续，Hermes 只负责机械地将续行 prompt（提示词）送回同一会话——直到主模型报告已验证完成、在穷尽所有安全方案后报告真正阻塞、你暂停或清除目标，或者轮次预算耗尽为止。

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
3. **主模型记录结果** — 通过 `todo` 工具记录 `continue`、`complete` 或 `blocked`，并附上原因和证据。
4. **若需要则触发循环** — 若精确轮次结果为 `continue`（或缺失/无效），你将看到 `↻ Continuing toward goal (1/20): <model's reason>`，Hermes 自动执行下一步。
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

目标激活期间，你可以使用 `/subgoal <text>` 追加额外的验收条件，而不会重置循环。每次调用会向目标的子目标列表添加一个编号条目；下一轮 agent 看到的**续行 prompt** 包含原始目标以及一个"用户在循环中途追加的额外条件"块。主模型在记录 `complete` 之前必须考虑所有子目标——只有原始目标**和**所有子目标均满足时，目标才会被标记为完成。

| 命令 | 功能 |
|---|---|
| `/subgoal <text>` | 向活跃目标追加一个新条件。需要有活跃的 `/goal`。 |
| `/subgoal`（无参数） | 显示当前编号子目标列表。 |
| `/subgoal remove <N>` | 删除第 N 个子目标（从 1 开始计数）。 |
| `/subgoal clear` | 删除所有子目标，但保留原始目标。 |

子目标与目标一起持久化存储在 `SessionDB.state_meta` 中，因此在 `/resume` 后依然有效。设置新的 `/goal <text>` 会替换目标并清空子目标列表；`/goal clear` 同样如此。

当你启动一个循环（"修复失败的测试"）后，中途发现还需要"为刚修复的 bug 添加回归测试"时，使用此功能——`/subgoal add a regression test` 可在不中断运行循环的情况下收紧成功条件。

## 行为细节

### 主模型编写的结果

每个目标轮次结束前，主模型通过现有 `todo` 工具记录一个结构化结果：

- `continue` — 仍有工作，Hermes 送入下一条续行 prompt。
- `complete` — 目标和所有条件已经用具体证据验证。
- `blocked` — 已穷尽所有安全可行方案，确实需要特定用户输入或外部变化。

Hermes 只接受绑定到当前模型轮次和目标 generation 的结果。运行时机械地验证并应用该记录；它不会解析回复文本、搜索关键词或调用辅助分类器替主模型判断任务是否完成。

### 失败开放语义

若精确轮次结果缺失、过期、格式错误或无效，Hermes 将其视为 `continue`。因此外部记账逻辑既不能虚构完成，也不能静默阻塞进度。**轮次预算**是真正的兜底机制。

### 轮次预算

默认为 20 个续行轮次（`config.yaml` 中的 `goals.max_turns`）。预算耗尽时，Hermes 自动暂停并告知你如何继续：

```
⏸ Goal paused — 20/20 turns used. Use /goal resume to keep going, or /goal clear to stop.
```

`/goal resume` 将计数器重置为零，你可以按可控的块继续推进。

### 用户消息始终优先

目标激活期间，你发送的任何真实消息都优先于续行循环。在 CLI 上，你的消息会在队列中的续行消息之前进入 `_pending_input`；在 gateway 上，它以同样的方式通过适配器 FIFO 传递。主模型可以在同一轮记录更新后的结构化结果。

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
```

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

## 修正结果

主模型拥有完整任务上下文，但控制权仍在你手中。若工作已完成而它仍记录 `continue`，轮次预算会暂停循环，你可以执行 `/goal clear` 或发送新消息。若它过早记录 `complete`，请发送后续消息，或设置更精确的完成条件。`↻ Continuing toward goal`、`✓ Goal achieved` 或 `⏸ Goal blocked` 行中的原因，是主模型附在结构化结果上的原始原因。

## 致谢

`/goal` 是 Hermes 对 **Ralph loop** 模式的实现。面向用户的设计——跨轮次保持目标存活、不达成不停止，以及创建/暂停/恢复/清除控制——由 OpenAI Codex 团队的 Eric Traut 在 [Codex CLI 0.128.0](https://github.com/openai/codex) 中推广并落地。我们的实现是独立的（中央 `CommandDef` 注册表、`SessionDB.state_meta` 持久化、精确轮次的主模型结果，以及 gateway 侧的适配器 FIFO 续行），但这个想法源自他们。功劳归于应得之人。
