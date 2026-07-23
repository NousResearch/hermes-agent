---
sidebar_position: 3
title: "Curator"
description: "Agent 创建的技能的后台维护——使用跟踪、过期检测、归档及 LLM 驱动的审查"
---

# Curator

Curator 是主要针对 **agent 创建的技能**的后台维护流程。它跟踪使用情况，将长期未使用的技能经历 `active → stale → archived` 状态流转，并定期启动一个短暂的辅助模型审查，提出合并或修补漂移的建议。

它的存在是为了防止通过[自我改进循环](/user-guide/features/skills#agent-managed-skills-skill_manage-tool)创建的技能无限堆积。每次 agent 解决新问题并保存技能时，该技能都会落入 `~/.hermes/skills/`。若没有维护，最终会出现数十个范围狭窄的近似重复项，污染技能目录并浪费 token（令牌）。

默认情况下（`prune_builtins: true`），Curator 在 `archive_after_days` 天未使用后，可以归档**未使用的捆绑内置技能**（随仓库附带），与它主要管理的 agent 自创技能一并处理。通过 [agentskills.io](https://agentskills.io) 安装的 hub 技能始终不受影响。设置 `curator.prune_builtins: false` 可恢复旧的“仅 agent 自创”行为，此时捆绑技能绝不会被触碰。Curator 也**绝不自动删除**——最坏的结果是归档到 `~/.hermes/skills/.archive/`，这是可恢复的。

跟踪 [issue #7816](https://github.com/NousResearch/hermes-agent/issues/7816)。

## 运行方式

Curator 由空闲检查触发，而非 cron 守护进程。在 CLI 会话启动时，以及 gateway 的 cron-ticker 线程内的周期性 tick 中，Hermes 会检查以下条件是否同时满足：

1. 距上次 curator 运行已过去足够长的时间（`interval_hours`，默认 **7 天**），以及
2. agent 已空闲足够长的时间（`min_idle_hours`，默认 **2 小时**）。

若两个条件均满足，则会派生一个 `AIAgent` 的后台 fork——与内存/技能自我改进 nudge 使用的模式相同。该 fork 在自己的 prompt（提示词）缓存中运行，绝不触碰当前活跃的对话。

:::info 首次运行行为
在全新安装时（或 pre-curator 版本在 `hermes update` 后首次 tick 时），curator **不会立即运行**。首次观测会将 `last_run_at` 设为"当前时间"，并将第一次真正的运行推迟整整一个 `interval_hours`。这给了你一个完整的间隔时间来审查技能库、固定重要内容，或在 curator 真正触碰它之前完全退出。

如果你想在 curator 真正运行之前查看它*会*做什么，请运行 `hermes curator run --dry-run`——它会生成相同的审查报告，但不会修改技能库。
:::

一次运行分为两个阶段：

1. **自动状态转换**（确定性，无 LLM）。未使用时间超过 `stale_after_days`（30 天）的技能变为 `stale`；未使用时间超过 `archive_after_days`（90 天）的技能被移至 `~/.hermes/skills/.archive/`。
2. **LLM 合并审查**（单次辅助模型 pass，`max_iterations=8`）——**默认关闭**。当 `curator.consolidate: true` 时，派生的 agent 会审查 agent 创建的技能，并逐个决定是保留、修补（通过 `skill_manage`）、合并到类别级 umbrella，还是通过终端工具归档。捆绑内置技能只有在 `curator.prune_builtins: true` 时才会出现，并且只作为可归档候选。

:::info 合并审查是 opt-in
默认情况下，curator 只执行**修剪**：确定性的空闲检测会标记 stale 并归档长期未使用的技能。带观点的 LLM **合并** pass（umbrella 构建、合并重叠技能）默认关闭，因为它每次运行都会消耗辅助模型 token，并且会对技能库做更广泛的结构性修改。可通过 `curator.consolidate: true` 打开，或用 `hermes curator run --consolidate` 针对单次运行开启。
:::

已固定（pinned）的技能对 curator 的自动状态转换和 agent 自身的 `skill_manage` 工具均不可操作。详见下方[固定技能](#pinning-a-skill)。

## 配置

所有设置位于 `config.yaml` 的 `curator:` 下（不在 `.env` 中——这不是密钥）。默认值：

```yaml
curator:
  enabled: true
  interval_hours: 168          # 7 days
  min_idle_hours: 2
  stale_after_days: 30
  archive_after_days: 90
  consolidate: false           # LLM umbrella-building pass — opt-in (prune-only by default)
  prune_builtins: true         # 同时归档未使用的捆绑内置技能（hub 技能始终豁免）
```

若要完全禁用，设置 `curator.enabled: false`。若要保留始终开启的修剪行为，但启用 LLM 合并审查，设置 `curator.consolidate: true`。

### 在更便宜的辅助模型上运行审查

Curator 的 LLM 审查 pass 是一个常规辅助任务槽——`auxiliary.curator`——与 Vision、Compression、Session Search 等并列。"Auto" 表示"使用我的主聊天模型"；可覆盖该槽以为审查 pass 指定特定的 provider + model。

**最简单——`hermes model`：**

```bash
hermes model                   # → "Auxiliary models — side-task routing"
                               # → pick "Curator" → pick provider → pick model
```

同样的选择器也可在 Web 控制台的 **Models** 标签页中使用。

**直接编辑 config.yaml（等效）：**

```yaml
auxiliary:
  curator:
    provider: openrouter
    model: google/gemini-3-flash-preview
    timeout: 600               # generous — reviews can take several minutes
```

保持 `provider: auto`（默认值）会将审查 pass 路由到主聊天模型，与所有其他辅助任务的行为一致。

:::note 旧版配置
早期版本使用独立的 `curator.auxiliary.{provider,model}` 块。该路径仍然有效，但会输出一条弃用日志——请迁移到上方的 `auxiliary.curator`，使 curator 与其他所有辅助任务共享相同的管道（`hermes model`、控制台 Models 标签页、`base_url`、`api_key`、`timeout`、`extra_body`）。
:::

## CLI

```bash
hermes curator status         # last run, counts, pinned list, LRU top 5
hermes curator run            # trigger a run now (blocks until done). Prune-only unless curator.consolidate: true
hermes curator run --consolidate # force the LLM consolidation pass on for this run, overriding the config default
hermes curator run --background  # fire-and-forget: start the run in a background thread
hermes curator run --dry-run  # preview only — report without any mutations
hermes curator backup         # take a manual snapshot of ~/.hermes/skills/
hermes curator rollback       # restore from the newest snapshot
hermes curator rollback --list     # list available snapshots
hermes curator rollback --id <ts>  # restore a specific snapshot
hermes curator rollback -y         # skip the confirmation prompt
hermes curator pause          # stop runs until resumed
hermes curator resume
hermes curator pin <skill>    # never auto-transition this skill
hermes curator unpin <skill>
hermes curator restore <skill>  # move an archived skill back to active
hermes curator list-archived    # list skills currently in ~/.hermes/skills/.archive/
hermes curator archive <skill>  # manually archive a single skill now
hermes curator prune [--days N] # 批量归档空闲 >= N 天的 agent-created 技能；启用时也可能包含捆绑内置技能
```

## 备份与回滚

在每次真正的 curator pass 之前，Hermes 会在 `~/.hermes/skills/.curator_backups/<utc-iso>/skills.tar.gz` 处对 `~/.hermes/skills/` 进行 tar.gz 快照。如果某次 pass 归档或合并了你不希望被触碰的内容，可以用一条命令撤销整次运行：

```bash
hermes curator rollback        # restore newest snapshot (with confirmation)
hermes curator rollback -y     # skip the prompt
hermes curator rollback --list # see all snapshots with reason + size
```

回滚本身也是可逆的：在替换技能树之前，Hermes 会再次创建一个标记为 `pre-rollback to <target-id>` 的快照，因此误操作的回滚可以通过 `--id` 滚动到该快照来撤销。

你也可以随时通过 `hermes curator backup --reason "before-refactor"` 手动创建快照。`--reason` 字符串会写入快照的 `manifest.json`，并在 `--list` 中显示。

快照会被裁剪至 `curator.backup.keep`（默认 5 个）以控制磁盘占用：

```yaml
curator:
  backup:
    enabled: true
    keep: 5
```

设置 `curator.backup.enabled: false` 可禁用自动快照。手动 `hermes curator backup` 命令仅在 `enabled: true` 时才能工作——该标志对两条路径对称生效，因此不会在变更性运行中意外跳过 pre-run 快照。

`hermes curator status` 还会列出五个最近最少使用的技能——快速查看哪些技能可能即将变为 stale。

相同的子命令也可作为 `/curator` 斜杠命令在运行中的会话（CLI 或 gateway 平台）内使用。

## "agent 创建"的含义

Curator 主要管理在 `~/.hermes/skills/.usage.json` 中显式标记为
**agent-created** 的技能。技能对应的 `.usage.json` 条目具有
`"created_by": "agent"` 或 `"agent_created": true` 时，才有这个标记。

有一个仅限归档的例外：列在 `~/.hermes/skills/.bundled_manifest` 中的捆绑内置技能，在 `curator.prune_builtins: true`（默认值）时也会成为候选。它们遵循同样的空闲归档窗口，但永远不会被修补、合并或删除。

列在 `~/.hermes/skills/.hub/lock.json` 中的 hub 安装技能永远不会被 curator 触碰，不受 `curator.prune_builtins` 影响。

目前，只有**后台自我改进审查 fork** 会设置 agent-created 标记——当它在周期性审查 pass（大约每 10 个 agent turn）中创建新的 umbrella skill 时。该后台 fork 使用 `"background_review"` 写入来源（通过 `tools/skill_provenance.py`），这是 `skill_manage` 中触发 `mark_agent_created()` 的唯一路径。

前台 agent 在对话中通过 `skill_manage(action="create")` 创建的技能**不会**被标记为 agent-created——它们被视为用户指示创建，curator 会故意跳过。

:::warning 你手写的技能不会被 curator 管理
如果你手动创建了 `SKILL.md`，或让 Hermes 指向外部技能目录，该技能在 `.usage.json` 中会是 `created_by: null`（或没有该字段）。Curator 不会触碰它。前台 agent 应你请求创建的技能同样如此。

**要查看 curator 实际管理哪些技能**，运行 `hermes curator status`。如果输出 `no agent-created skills`，则当前没有 agent-created 技能，也没有可修剪的捆绑内置技能在范围内——LLM 审查 pass 会被跳过，报告会显示 `Model: (not resolved) via (not resolved)` 和 `Duration: 0s`。
:::

Agent-created 技能遵循完整生命周期：

- `active` →（30 天未使用）`stale` →（90 天未使用）`archived`
- 已固定的技能会绕过所有自动状态转换
- 归档可通过 `hermes curator restore <name>` 恢复

捆绑内置技能只有在 `curator.prune_builtins: true` 时才遵循 stale/archive 时间窗口；它们不参与修补、合并或固定。

如果你想保护某个特定非捆绑、非 hub、非外部技能名不被工具驱动删除——例如你依赖的手写技能——请使用 `hermes curator pin <name>`。详见下一节。

## 固定技能 {#pinning-a-skill}

固定（pinning）可保护技能不被删除——包括 curator 的自动归档 pass 和 agent 的 `skill_manage(action="delete")` 工具调用。技能一旦被固定：

- **Curator** 在自动状态转换（`active → stale → archived`）时跳过它，其 LLM 审查 pass 也被指示不予处理。
- **Agent 的 `skill_manage` 工具**拒绝对其执行 `delete`，并提示用户使用 `hermes curator unpin <name>`。修补和编辑仍然可以进行，因此 agent 可以在遇到问题时改进已固定技能的内容，无需反复 pin/unpin/re-pin。

使用以下命令固定和取消固定：

```bash
hermes curator pin <skill>
hermes curator unpin <skill>
```

该标志以 `"pinned": true` 的形式存储在 `~/.hermes/skills/.usage.json` 中技能对应的条目上，因此跨会话持久有效。

只有非捆绑、非 hub、非外部的技能名才能被固定——若你尝试固定捆绑或 hub 安装的技能，`hermes curator pin` 会拒绝并给出说明。Hub 安装的技能永远不受 curator 变更。捆绑内置技能仅在 `curator.prune_builtins: true`（默认值）时会被触碰，并且即使如此也只会在 `archive_after_days` 未使用后被归档——不会被修补、合并、固定或删除。设置 `curator.prune_builtins: false` 可完全豁免捆绑技能。

少量**受保护内置技能**被硬编码为永不可归档、永不可合并，不受 `curator.prune_builtins`、pin 状态或 LLM 判断影响。它们支撑关键 UX，例如 `plan` 驱动 `/plan` 斜杠命令流程；如果悄悄归档，会让斜杠命令变成没有提示的 “Unknown command” 错误。受保护内置技能会从 curator 候选列表中完全过滤掉，因此合并 pass 也看不到它们。

如果你想要比"禁止删除"更强的保证——例如在 agent 仍可读取技能的同时完全冻结其内容——请直接用编辑器编辑 `~/.hermes/skills/<name>/SKILL.md`。pin 保护的是工具驱动的删除，而非你自己的文件系统访问。

## 使用遥测

Curator 在 `~/.hermes/skills/.usage.json` 维护一个附属文件，每个技能对应一条记录：

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

- `view_count`：agent 对该技能调用 `skill_view`。
- `use_count`：技能被加载到对话的 prompt 中。
- `patch_count`：对该技能执行 `skill_manage patch/edit/write_file/remove_file`。

使用遥测会记录 agent 查看、加载或修补的任何技能——包括捆绑技能和 hub 安装的技能。但 curator 生命周期状态（`state`、`pinned`、`archived_at`、`created_by`）仍然只会写入 curator 可管理的技能。

## 每次运行的报告

每次 curator 运行都会在 `~/.hermes/logs/curator/` 下写入一个带时间戳的目录：

```
~/.hermes/logs/curator/
└── 20260429-111512/
    ├── run.json      # machine-readable: full fidelity, stats, LLM output
    └── REPORT.md     # human-readable summary
```

`REPORT.md` 是快速查看某次运行所做操作的方式——哪些技能发生了状态转换、LLM 审查者说了什么、修补了哪些技能。无需 grep `agent.log` 即可完成审计。

### 摘要中的重命名映射

如果某次运行将多个技能合并到一个总括技能下（或合并了近似重复项），运行结束时打印的用户可见摘要会包含一个明确的重命名映射，显示 curator 应用的每个 `旧名称 → 新名称` 对。这是对逐技能状态转换行的补充，因此当一批重命名落地时，你可以一眼发现，无需对比 JSON 报告。该提示也会在 `hermes curator pin` 下显示，以便你在需要时立即固定新标签。

## 恢复已归档的技能

如果 curator 归档了你仍需要的技能：

```bash
hermes curator restore <skill-name>
```

这会将技能从 `~/.hermes/skills/.archive/` 移回活跃树，并将其状态重置为 `active`。如果此后有同名的捆绑或 hub 安装技能（会遮蔽上游），则恢复操作会被拒绝。

## 按环境禁用

Curator 默认开启。若要关闭：

- **仅针对某个 profile：** 编辑 `~/.hermes/config.yaml`（或当前活跃 profile 的配置），设置 `curator.enabled: false`。
- **仅针对单次运行：** `hermes curator pause`——暂停跨会话持久有效；使用 `resume` 重新启用。

Curator 在 `min_idle_hours` 未经过时也会拒绝运行，因此在活跃的开发机器上，它自然只会在安静时段运行。

## 另请参阅

- [技能系统](/user-guide/features/skills)——技能的总体工作原理及创建技能的自我改进循环
- [内存](/user-guide/features/memory)——维护长期记忆的并行后台审查
- [捆绑技能目录](/reference/skills-catalog)
- [Issue #7816](https://github.com/NousResearch/hermes-agent/issues/7816)——原始提案与设计讨论
