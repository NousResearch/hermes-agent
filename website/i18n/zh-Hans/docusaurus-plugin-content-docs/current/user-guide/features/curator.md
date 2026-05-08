---
sidebar_position: 3
title: "Curator"
description: "Agent 创建的技能的后台维护——使用跟踪、过时、归档和 LLM 驱动审查"
---

# Curator

Curator 是一个后台维护程序，用于管理 **agent 创建的技能**。它跟踪每个技能被查看、使用和打补丁的频率，通过 `active → stale → archived` 状态移动长期未使用的技能，并定期生成一个简短的辅助模型审查，提出整合建议或补丁漂移修复。

它的存在是因为通过[自我改进循环](/docs/user-guide/features/skills#agent-managed-skills-skill_manage-tool)创建的技能不会永远堆积。每次 agent 解决一个新问题并保存一个技能时，该技能就会进入 `~/.hermes/skills/`。如果没有维护，你最终会得到数十个污染目录并浪费令牌的狭窄近似重复。

Curator **从不触碰**捆绑技能（随仓库一起提供）或 hub 安装的技能（来自 [agentskills.io](https://agentskills.io)）。它只审查 agent 本身创作的技能。它也**永远不会自动删除**——最坏的结果是归档到 `~/.hermes/skills/.archive/`，这是可恢复的。

跟踪 [issue #7816](https://github.com/NousResearch/hermes-agent/issues/7816)。

## 它如何运行

Curator 由非活动检查触发，而不是 cron 守护程序。在 CLI 会话开始时，以及在 gateway 的 cron-ticker 线程内的定期刻度上，Hermes 检查是否：

1. 自上次 curator 运行以来已经过了足够的时间（`interval_hours`，默认 **7 天**），以及
2. Agent 已经空闲了足够的时间（`min_idle_hours`，默认 **2 小时**）。

如果两者都为真，它会生成一个后台 `AIAgent` 分支——与内存/技能自我改进提示使用的相同模式。分支在自己的提示缓存中运行，永远不会触及活动对话。

:::info 首次运行行为
在全新安装时（或在 `hermes update` 之后 pre-curator 安装首次滴答作响时），curator **不会立即运行**。第一次观察将 `last_run_at` 种子设为"现在"，并将第一次真正的运行推迟一个完整的 `interval_hours`。这让你有完整的间隔来审查你的技能库、固定任何重要的内容，或者在 curator 接触之前完全选择退出。

如果你想在 curator 真正运行之前看到它*会*做什么，运行 `hermes curator run --dry-run`——它会产生相同的审查报告而不改变库。
:::

一次运行有两个阶段：

1. **自动转换**（确定性，无 LLM）。未使用 `stale_after_days`（30）的技能变为 `stale`；未使用 `archive_after_days`（90）的技能移动到 `~/.hermes/skills/.archive/`。
2. **LLM 审查**（单个辅助模型轮次，`max_iterations=8`）。分叉的 agent 审查 agent 创建的技能，可以使用 `skill_view` 读取任何技能，并决定每个技能是保留、打补丁（通过 `skill_manage`）、整合重叠的技能，还是通过终端工具归档。

固定的技能不受 curator 的自动转换和 agent 自己的 `skill_manage` 工具的约束。请参阅下面的[固定技能](#pinning-a-skill)。

## 配置

所有设置位于 `config.yaml` 下的 `curator:`（不是 `.env`——这不是秘密）。默认值：

```yaml
curator:
  enabled: true
  interval_hours: 168          # 7 天
  min_idle_hours: 2
  stale_after_days: 30
  archive_after_days: 90
```

要完全禁用，设置 `curator.enabled: false`。

### 使用更便宜的辅助模型运行审查

Curator 的 LLM 审查轮次是一个常规辅助任务槽——`auxiliary.curator`——与 Vision、压缩、会话搜索等并列。"Auto"意味着"使用我的主聊天模型"；覆盖槽以固定特定提供商 + 模型进行审查轮次。

**最简单的方法——`hermes model`：**

```bash
hermes model                   # → "Auxiliary models — side-task routing"
                               # → 选择 "Curator" → 选择提供商 → 选择模型
```

相同的选项器在 Web Dashboard 的 **Models** 选项卡下可用。

**直接 config.yaml（等效）：**

```yaml
auxiliary:
  curator:
    provider: openrouter
    model: google/gemini-3-flash-preview
    timeout: 600               # 慷慨——审查可能需要几分钟
```

保留 `provider: auto`（默认）会将审查轮次路由到你主聊天模型使用的任何模型，与其他每个辅助任务的行为匹配。

:::note 旧版配置
早期版本使用一次性 `curator.auxiliary.{provider,model}` 块。该路径仍然有效，但会发出弃用日志行——请迁移到上面的 `auxiliary.curator`，以便 curator 与其他每个辅助任务共享相同的管道（`hermes model`、仪表板 Models 选项卡、`base_url`、`api_key`、`timeout`、`extra_body`）。
:::

## CLI

```bash
hermes curator status         # 上次运行、计数、固定列表、LRU 前 5
hermes curator run            # 立即触发审查（阻塞直到 LLM 轮次完成）
hermes curator run --background  # 触发后忘记：在后台线程启动 LLM 轮次
hermes curator run --dry-run  # 仅预览——报告而不进行任何更改
hermes curator backup         # 手动拍摄 ~/.hermes/skills/ 的快照
hermes curator rollback       # 从最新快照恢复
hermes curator rollback --list     # 列出可用快照
hermes curator rollback --id <ts>  # 恢复特定快照
hermes curator rollback -y         # 跳过确认提示
hermes curator pause          # 停止运行直到恢复
hermes curator resume
hermes curator pin <skill>    # 永不自动转换此技能
hermes curator unpin <skill>
hermes curator restore <skill>  # 将归档的技能移回活动状态
```

## 备份和回滚

在每次真正的 curator 运行之前，Hermes 会在 `~/.hermes/skills/.curator_backups/<utc-iso>/skills.tar.gz` 拍摄 `~/.hermes/skills/` 的 tar.gz 快照。如果一次运行归档或整合了你不想触碰的内容，你可以通过一个命令撤销整个运行：

```bash
hermes curator rollback        # 恢复最新快照（带确认）
hermes curator rollback -y     # 跳过提示
hermes curator rollback --list # 查看所有带原因 + 大小的快照
```

回滚本身是可逆的：在替换技能树之前，Hermes 拍摄另一个标记为 `pre-rollback to <target-id>` 的快照，因此错误的回滚可以通过 `--id` 向前滚动到该快照来撤销。

你也可以随时使用 `hermes curator backup --reason "before-refactor"` 拍摄手动快照。`--reason` 字符串进入快照的 `manifest.json` 并在 `--list` 中显示。

快照被修剪为 `curator.backup.keep`（默认 5）以保持磁盘使用受限：

```yaml
curator:
  backup:
    enabled: true
    keep: 5
```

设置 `curator.backup.enabled: false` 以禁用自动快照。当备份禁用时，手动 `hermes curator backup` 命令仅在你首先设置 `enabled: true` 时才有效——该标志对称地控制两条路径，因此在变种运行上不可能意外跳过预运行快照。

`hermes curator status` 还列出了五个最近最少使用的技能——快速查看下一个可能变得陈旧的方法。

相同的子命令在运行会话中的 `/curator` 斜杠命令中可用（CLI 或 gateway 平台）。

## "agent 创建的"含义

如果技能的名称**不在**以下位置，则认为该技能是 agent 创建的：

- `~/.hermes/skills/.bundled_manifest`（安装时从仓库复制的技能）
- `~/.hermes/skills/.hub/lock.json`（通过 `hermes skills install` 安装的技能）

`~/.hermes/skills/` 中的其他所有内容都是 curator 的公平游戏。这包括：

- Agent 通过对话中的 `skill_manage(action="create")` 保存的技能。
- 你使用手写 `SKILL.md` 手动创建的技能。
- 你指向 Hermes 的外部技能目录中添加的技能。

:::warning 你的手写技能看起来与 agent 保存的技能相同
这里的来源是**二元的**（捆绑/hub vs. 其他一切）。Curator 无法将你依赖的私有工作流的手写技能与自我改进循环在会话中期保存的技能区分开来。两者都进入"agent 创建"桶。

在第一次真正的运行之前（默认在安装后 7 天），请花点时间：

1. 运行 `hermes curator run --dry-run` 以准确查看 curator 会提出什么。
2. 使用 `hermes curator pin <name>` 围栏你不想被触碰的任何内容。
3. 或者在 `config.yaml` 中设置 `curator.enabled: false`，如果你宁愿自己管理库。

归档始终可以通过 `hermes curator restore <name>` 恢复，但在事后追逐整合比预先固定更容易。
:::

如果你想保护特定技能永远不被触碰——例如你依赖的手写技能——使用 `hermes curator pin <name>`。请参阅下一节。

## 固定技能

固定保护技能免受删除——既保护 curator 的自动归档运行，也保护 agent 的 `skill_manage(action="delete")` 工具调用。一旦技能被固定：

- **Curator** 在自动转换（`active → stale → archived`）期间跳过它，它的 LLM 审查轮次被指示不要触碰它。
- **Agent 的 `skill_manage` 工具** 拒绝在其上进行 `delete`，指向 `hermes curator unpin <name>`。补丁和编辑仍然通过，因此 agent 可以在出现陷阱时改进固定技能的内容，而无需固定/取消固定/重新固定的舞蹈。

使用以下命令固定和取消固定：

```bash
hermes curator pin <skill>
hermes curator unpin <skill>
```

该标志存储为 `~/.hermes/skills/.usage.json` 中技能条目的 `"pinned": true`，因此它跨会话存活。

只有 **agent 创建的**技能可以被固定——捆绑和 hub 安装的技能首先不受 curator 变种的影响，如果你尝试，`hermes curator pin` 会拒绝并显示解释性消息。

如果你想要比"不删除"更强的保证——例如，在 agent 仍然读取技能时完全冻结技能的内容——直接用编辑器编辑 `~/.hermes/skills/<name>/SKILL.md`。固定可以防止工具驱动的删除，而不是你自己的文件系统访问。

## 使用遥测

Curator 在 `~/.hermes/skills/.usage.json` 维护一个辅助文件，每个技能一个条目：

```json
{
  "my-skill": {
    "use_count": 12,
    "view_count": 34,
    "last_used_at": "2026-04-24T18:12:03Z",
    "last_viewed_at": "2026-04-23T09:44:17Z",
    "patch_count": 3,
    "last_patched_at": "2026-04-20T22:01:55Z",
    "created_at": "2026-03-01T14:20:00Z",
    "state": "active",
    "pinned": false,
    "archived_at": null
  }
}
```

计数器在以下情况下递增：

- `view_count`：agent 调用 `skill_view` 查看技能时。
- `use_count`：技能加载到会话提示中时。
- `patch_count`：`skill_manage patch/edit/write_file/remove_file` 在技能上运行时。

捆绑和 hub 安装的技能被明确排除在遥测写入之外。

## 每次运行的报告

每次 curator 运行在 `~/.hermes/logs/curator/` 下写入一个带时间戳的目录：

```
~/.hermes/logs/curator/
└── 20260429-111512/
    ├── run.json      # 机器可读：完全保真度、统计、LLM 输出
    └── REPORT.md     # 人类可读摘要
```

`REPORT.md` 是快速查看给定运行做了什么的好方法——哪些技能转换了，LLM 审查者说了什么，哪些技能被打了补丁。适合审计而不必 grep `agent.log`。

## 恢复归档的技能

如果 curator 归档了你仍然想要的内容：

```bash
hermes curator restore <skill-name>
```

这将技能从 `~/.hermes/skills/.archive/` 移回活动树并将其状态重置为 `active`。如果有捆绑或 hub 安装的技能以相同名称安装（会遮蔽上游），则恢复会拒绝。

## 每个环境禁用

Curator 默认开启。要关闭它：

- **仅对一个配置文件：** 编辑 `~/.hermes/config.yaml`（或活动配置文件的配置）并设置 `curator.enabled: false`。
- **仅针对一次运行：** `hermes curator pause`——暂停跨会话保持；使用 `resume` 重新启用。

如果 `min_idle_hours` 尚未过去，curator 也拒绝运行，因此在活动开发机器上，它自然只会在安静时间段运行。

## 另请参阅

- [技能系统](/docs/user-guide/features/skills) — 技能如何工作以及创建它们的自我改进循环
- [记忆](/docs/user-guide/features/memory) — 维护长期记忆的并行后台审查
- [捆绑技能目录](/docs/reference/skills-catalog)
- [Issue #7816](https://github.com/NousResearch/hermes-agent/issues/7816) — 原始提案和设计讨论
