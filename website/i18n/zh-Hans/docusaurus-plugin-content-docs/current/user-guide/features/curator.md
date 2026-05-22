---
sidebar_position: 3
title: "Curator"
description: "智能体创建技能的背景维护 —— 使用追踪、陈旧性归档和 LLM 驱动的审查"
---

# Curator

Curator 是**智能体创建技能**的背景维护流程。它追踪每个技能被查看、使用和修补的频率，将长期未使用的技能通过 `active → stale → archived` 状态迁移，并定期生成一个简短的辅助模型审查，提议合并或修补漂移。

它的存在是为了防止通过[自我改进循环](/user-guide/features/skills#agent-managed-skills-skill_manage-tool)创建的技能无限堆积。每次智能体解决一个新问题并保存一个技能时，该技能会进入 `~/.hermes/skills/`。如果没有维护，您最终会拥有数十个狭窄的近似重复项，它们污染目录并浪费令牌。

Curator **从不触碰**捆绑技能（随仓库一起发布）或 hub 安装的技能（来自 [agentskills.io](https://agentskills.io)）。它只审查智能体自己编写的技能。它也**从不自动删除** —— 最坏的结果是将技能归档到 `~/.hermes/skills/.archive/`，这是可恢复的。

跟踪 [issue #7816](https://github.com/NousResearch/hermes-agent/issues/7816)。

## 如何运行

Curator 由不活动检查触发，而不是 cron 守护进程。在 CLI 会话启动时，以及网关的 cron-ticker 线程内的定期滴答中，Hermes 检查是否：

1. 自上次 curator 运行以来已经过了足够时间（`interval_hours`，默认**7天**），以及
2. 智能体已经空闲了足够长的时间（`min_idle_hours`，默认**2小时**）。

如果两者都为真，它会生成一个 `AIAgent` 的后台分支 —— 与记忆/技能自我改进提示使用的相同模式。该分支在自己的提示缓存中运行，从不触碰活跃对话。

:::info 首次运行行为
在全新安装上（或首次在 curator 之前的安装上运行 `hermes update` 后），curator **不会立即运行**。第一次观察将 `last_run_at` 设置为"现在"，并将第一次真正的运行推迟一个完整的 `interval_hours`。这给了您一个完整的间隔来审查您的技能库，固定任何重要的内容，或在 curator 触碰它之前完全选择退出。

如果您想在 curator 真正运行之前看看它*会*做什么，请运行 `hermes curator run --dry-run` —— 它生成相同的审查报告，但不修改库。
:::

一次运行有两个阶段：

1. **自动转换**（确定性，无 LLM）。未使用 `stale_after_days`（30天）的技能变为 `stale`；未使用 `archive_after_days`（90天）的技能被移动到 `~/.hermes/skills/.archive/`。
2. **LLM 审查**（单次辅助模型传递，`max_iterations=8`）。分支智能体调查智能体创建的技能，可以用 `skill_view` 阅读其中任何一个，并决定每个技能是保留、修补（通过 `skill_manage`）、合并重叠的技能，还是通过终端工具归档。

固定技能对 curator 的自动转换和智能体自己的 `skill_manage` 工具都不可触碰。请参阅下面的[固定技能](#pinning-a-skill)。

## 配置

所有设置都位于 `config.yaml` 的 `curator:` 下（不是 `.env` —— 这不是秘密）。默认值：

```yaml
curator:
  enabled: true
  interval_hours: 168          # 7 天
  min_idle_hours: 2
  stale_after_days: 30
  archive_after_days: 90
```

要完全禁用，请设置 `curator.enabled: false`。

### 在更便宜的辅助模型上运行审查

Curator 的 LLM 审查传递是一个常规的辅助任务槽 —— `auxiliary.curator` —— 与 Vision、Compression、Session Search 等并列。"Auto" 表示"使用我的主聊天模型"；覆盖该槽位以固定一个特定的提供商 + 模型用于审查传递。

**最简单 — `hermes model`：**

```bash
hermes model                   # → "Auxiliary models — side-task routing"
                               # → 选择 "Curator" → 选择提供商 → 选择模型
```

相同的选取器在 Web 仪表板的 **Models** 标签页中可用。

**直接 config.yaml（等效）：**

```yaml
auxiliary:
  curator:
    provider: openrouter
    model: google/gemini-3-flash-preview
    timeout: 600               # 宽裕 —— 审查可能需要几分钟
```

保持 `provider: auto`（默认）会将审查传递路由到您的主聊天模型，与每个其他辅助任务的行为匹配。

:::note 旧版配置
早期版本使用一次性的 `curator.auxiliary.{provider,model}` 块。该路径仍然有效，但会发出弃用日志行 —— 请迁移到上面的 `auxiliary.curator`，以便 curator 与每个其他辅助任务共享相同的管道（`hermes model`、仪表板 Models 标签页、`base_url`、`api_key`、`timeout`、`extra_body`）。
:::

## CLI

```bash
hermes curator status         # 上次运行、计数、固定列表、最近最少使用的前 5 个
hermes curator run            # 立即触发审查（阻塞直到 LLM 传递完成）
hermes curator run --background  # 即发即弃：在后台线程中启动 LLM 传递
hermes curator run --dry-run  # 仅预览 —— 报告而不做任何修改
hermes curator backup         # 手动快照 ~/.hermes/skills/
hermes curator rollback       # 从最新快照恢复
hermes curator rollback --list     # 列出可用快照
hermes curator rollback --id <ts>  # 恢复特定快照
hermes curator rollback -y         # 跳过确认提示
hermes curator pause          # 暂停运行直到恢复
hermes curator resume
hermes curator pin <skill>    # 永不自动转换此技能
hermes curator unpin <skill>
hermes curator restore <skill>  # 将已归档技能移回活跃状态
```

## 备份与回滚

在每次真正的 curator 传递之前，Hermes 会在 `~/.hermes/skills/.curator_backups/<utc-iso>/skills.tar.gz` 处对 `~/.hermes/skills/` 进行 tar.gz 快照。如果一次传递归档或合并了您不想触碰的内容，您可以用一个命令撤销整个运行：

```bash
hermes curator rollback        # 恢复最新快照（带确认）
hermes curator rollback -y     # 跳过提示
hermes curator rollback --list # 查看所有快照及其原因 + 大小
```

回滚本身是可逆的：在替换技能树之前，Hermes 会拍摄另一个标记为 `pre-rollback to <target-id>` 的快照，因此错误的回滚可以通过 `--id` 回滚到该快照来撤销。

您也可以随时用 `hermes curator backup --reason "before-refactor"` 拍摄手动快照。`--reason` 字符串会进入快照的 `manifest.json`，并在 `--list` 中显示。

快照被修剪为 `curator.backup.keep`（默认 5）以保持磁盘使用受限：

```yaml
curator:
  backup:
    enabled: true
    keep: 5
```

设置 `curator.backup.enabled: false` 以禁用自动快照。当备份禁用时，手动 `hermes curator backup` 命令仅在您先设置 `enabled: true` 时才有效 —— 该标志对称地控制两条路径，因此无法在修改运行时意外跳过运行前快照。

`hermes curator status` 还列出了五个最近最少使用的技能 —— 一种快速查看接下来可能变为陈旧的内容的方式。

相同的子命令在运行会话（CLI 或网关平台）中作为 `/curator` 斜杠命令可用。

## "智能体创建"的含义

如果一个技能的名称**不在**以下列表中，则被视为智能体创建：

- `~/.hermes/skills/.bundled_manifest`（安装时从仓库复制的技能），以及
- `~/.hermes/skills/.hub/lock.json`（通过 `hermes skills install` 安装的技能）。

`~/.hermes/skills/` 中的所有其他内容都是 curator 的审查对象。这包括：

- 智能体在对话期间通过 `skill_manage(action="create")` 保存的技能。
- 您用手写 `SKILL.md` 手动创建的技能。
- 通过您指向 Hermes 的外部技能目录添加的技能。

:::warning 您手写的技能与智能体保存的技能看起来一样
这里的来源是**二元的**（捆绑/hub 与其他所有内容）。Curator 无法区分您依赖的用于私人工作流的手写技能与自我改进循环在会话中途保存的技能。两者都落入"智能体创建"桶中。

在第一次真正运行之前（默认安装后 7 天），花点时间：

1. 运行 `hermes curator run --dry-run` 以准确查看 curator 会提议什么。
2. 使用 `hermes curator pin <name>` 固定任何您不想触碰的内容。
3. 或者如果您更愿意自己管理库，请在 `config.yaml` 中设置 `curator.enabled: false`。

归档总是可以通过 `hermes curator restore <name>` 恢复，但预先固定比在事后追踪合并更容易。
:::

如果您想保护特定技能永远不被触碰 —— 例如您依赖的手写技能 —— 请使用 `hermes curator pin <name>`。请参阅下一节。

## 固定技能 {#pinning-a-skill}

固定保护技能免于删除 —— 包括 curator 的自动归档传递和智能体的 `skill_manage(action="delete")` 工具调用。一旦技能被固定：

- **Curator** 在自动转换（`active → stale → archived`）期间跳过它，其 LLM 审查传递被指示不要触碰它。
- **智能体的 `skill_manage` 工具** 拒绝在其上执行 `delete`，引导用户使用 `hermes curator unpin <name>`。修补和编辑仍然通过，因此智能体可以在不需要固定/取消固定/重新固定的繁琐操作的情况下改进固定技能的内容。

使用以下命令固定和取消固定：

```bash
hermes curator pin <skill>
hermes curator unpin <skill>
```

该标志存储为技能在 `~/.hermes/skills/.usage.json` 中条目上的 `"pinned": true`，因此它在会话之间持久存在。

只有**智能体创建**的技能可以被固定 —— 捆绑和 hub 安装的技能从一开始就不受 curator 修改的影响，如果您尝试，`hermes curator pin` 会拒绝并附带解释性消息。

如果您想要比"不删除"更强的保证 —— 例如，在智能体仍然阅读技能的同时完全冻结其内容 —— 请直接用您的编辑器编辑 `~/.hermes/skills/<name>/SKILL.md`。固定保护工具驱动的删除，但不保护您自己的文件系统访问。

## 使用遥测

Curator 在 `~/.hermes/skills/.usage.json` 中维护一个附注文件，每个技能一个条目：

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

计数器在以下情况递增：

- `view_count`：智能体调用 `skill_view` 查看该技能。
- `use_count`：该技能被加载到对话的提示中。
- `patch_count`：在该技能上运行 `skill_manage patch/edit/write_file/remove_file`。

捆绑和 hub 安装的技能被显式排除在遥测写入之外。

## 每次运行报告

每次 curator 运行都会在 `~/.hermes/logs/curator/` 下写入一个带时间戳的目录：

```
~/.hermes/logs/curator/
└── 20260429-111512/
    ├── run.json      # 机器可读：完整保真度、统计、LLM 输出
    └── REPORT.md     # 人类可读的摘要
```

`REPORT.md` 是快速查看给定运行做了什么的好方法 —— 哪些技能转换了，LLM 审查者说了什么，它修补了哪些技能。适合审计，无需 grep `agent.log`。

## 恢复已归档技能

如果 curator 归档了您仍然想要的内容：

```bash
hermes curator restore <skill-name>
```

这将技能从 `~/.hermes/skills/.archive/` 移回活跃树，并将其状态重置为 `active`。如果之后安装了同名的捆绑或 hub 技能，恢复会拒绝（会遮蔽上游）。

## 按环境禁用

Curator 默认开启。要关闭它：

- **仅一个配置文件：** 编辑 `~/.hermes/config.yaml`（或活跃配置文件的配置）并设置 `curator.enabled: false`。
- **仅一次运行：** `hermes curator pause` —— 暂停在会话之间持久存在；使用 `resume` 重新启用。

如果 `min_idle_hours` 尚未经过，curator 也拒绝运行，因此在活跃的开发机器上，它自然只在安静时段运行。

## 另请参阅

- [技能系统](/user-guide/features/skills) —— 技能的一般工作原理以及创建它们的自我改进循环
- [记忆](/user-guide/features/memory) —— 维护长期记忆的并行后台审查
- [捆绑技能目录](/reference/skills-catalog)
- [Issue #7816](https://github.com/NousResearch/hermes-agent/issues/7816) —— 原始提案和设计讨论
