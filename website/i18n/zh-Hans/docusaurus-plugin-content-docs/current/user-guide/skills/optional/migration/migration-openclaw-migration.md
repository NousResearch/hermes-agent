---
title: "Openclaw Migration — 将用户的 OpenClaw 自定义配置迁移到 Hermes Agent"
sidebar_label: "Openclaw Migration"
description: "将用户的 OpenClaw 自定义配置迁移到 Hermes Agent"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Openclaw Migration

将用户的 OpenClaw 自定义配置迁移到 Hermes Agent。从 `~/.openclaw` 导入 Hermes 兼容的记忆、`SOUL.md`、命令白名单、用户技能及所选工作区资产，并精确报告无法迁移的内容及原因。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/migration/openclaw-migration` 安装 |
| 路径 | `optional-skills/migration/openclaw-migration` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent (Nous Research) |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `Migration`, `OpenClaw`, `Hermes`, `Memory`, `Persona`, `Import` |
| 相关 skill | [`hermes-agent`](/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

# OpenClaw -> Hermes 迁移

当用户希望以最少的手动清理将其 OpenClaw 配置迁移到 Hermes Agent 时，使用此 skill。

## CLI 命令

如需快速、非交互式迁移，使用内置 CLI 命令：

```bash
hermes claw migrate              # Full interactive migration
hermes claw migrate --dry-run    # Preview what would be migrated
hermes claw migrate --preset user-data   # Migrate without secrets
hermes claw migrate --overwrite  # Overwrite existing conflicts
hermes claw migrate --source /custom/path/.openclaw  # Custom source
```

CLI 命令运行与下文所述相同的迁移脚本。当需要交互式、引导式迁移并支持 dry-run（预览）和逐项冲突解决时，请通过 agent 使用此 skill。

**首次设置：** `hermes setup` 向导会自动检测 `~/.openclaw`，并在配置开始前提供迁移选项。

## 此 skill 的功能

它使用 `scripts/openclaw_to_hermes.py` 来：

- 将 `SOUL.md` 导入 Hermes 主目录，保存为 `SOUL.md`
- 将 OpenClaw 的 `MEMORY.md` 和 `USER.md` 转换为 Hermes 记忆条目
- 将 OpenClaw 命令审批模式合并到 Hermes `command_allowlist`
- 迁移 Hermes 兼容的消息设置，例如 `TELEGRAM_ALLOWED_USERS` 和 `MESSAGING_CWD`
- 将 OpenClaw skill 复制到 `~/.hermes/skills/openclaw-imports/`
- 可选地将 OpenClaw 工作区指令文件复制到所选 Hermes 工作区
- 将兼容的工作区资产（如 `workspace/tts/`）镜像到 `~/.hermes/tts/`
- 归档没有直接 Hermes 目标的非机密文档
- 生成结构化报告，列出已迁移项、冲突项、跳过项及原因

## 路径解析

辅助脚本位于此 skill 目录下：

- `scripts/openclaw_to_hermes.py`

从 Skills Hub 安装此 skill 后，通常位于：

- `~/.hermes/skills/migration/openclaw-migration/scripts/openclaw_to_hermes.py`

请勿猜测更短的路径，如 `~/.hermes/skills/openclaw-migration/...`。

运行辅助脚本前：

1. 优先使用 `~/.hermes/skills/migration/openclaw-migration/` 下的已安装路径。
2. 如果该路径失败，检查已安装的 skill 目录，并相对于已安装的 `SKILL.md` 解析脚本路径。
3. 仅在已安装位置缺失或 skill 被手动移动时，才使用 `find` 作为备用方案。
4. 调用终端工具时，不要传入 `workdir: "~"`。请使用绝对目录（如用户主目录），或完全省略 `workdir`。

使用 `--migrate-secrets` 时，还将导入一小组 Hermes 兼容的白名单 secret，目前包括：

- `TELEGRAM_BOT_TOKEN`

## 默认工作流

1. 首先通过 dry run 进行检查。
2. 呈现简洁摘要，说明哪些内容可以迁移、哪些不能迁移、哪些将被归档。
3. 如果 `clarify` 工具可用，使用它处理用户决策，而非要求自由格式的文字回复。
4. 如果 dry run 发现已导入 skill 目录存在冲突，在执行前询问处理方式。
5. 在执行前，请用户在两种支持的迁移模式中选择一种。
6. 仅在用户希望迁移工作区指令文件时，才询问目标工作区路径。
7. 使用匹配的 preset 和标志执行迁移。
8. 汇总结果，重点说明：
   - 已迁移的内容
   - 已归档待手动审查的内容
   - 已跳过的内容及原因

## 用户交互协议

Hermes CLI 支持 `clarify` 工具进行交互式提示，但有以下限制：

- 每次只能处理一个选择
- 最多 4 个预定义选项
- 自动提供 `Other` 自由文本选项

它**不**支持在单个提示中进行真正的多选复选框操作。

每次 `clarify` 调用：

- 必须包含非空的 `question`
- 仅对真实可选提示包含 `choices`
- `choices` 限制为 2-4 个纯字符串选项
- 不得输出占位符或截断选项，如 `...`
- 不得在选项中填充或添加额外空白
- 不得在问题中包含虚假表单字段，如 `在此输入目录`、空白行或下划线 `_____`
- 对于开放式路径问题，只询问纯文本句子；用户在面板下方的普通 CLI 提示符中输入

如果 `clarify` 调用返回错误，检查错误文本，修正 payload，并使用有效的 `question` 和干净的 choices 重试一次。

当 `clarify` 可用且 dry run 揭示任何需要用户决策的情况时，**下一个动作必须是 `clarify` 工具调用**。
不得以如下普通助手消息结束对话：

- "让我来呈现选项"
- "您希望怎么做？"
- "以下是选项"

如果需要用户决策，在生成更多文字之前通过 `clarify` 收集。
如果存在多个未解决的决策，不要在它们之间插入解释性助手消息。收到一个 `clarify` 响应后，下一个动作通常应是下一个必要的 `clarify` 调用。

当 dry run 报告以下情况时，将 `workspace-agents` 视为未解决的决策：

- `kind="workspace-agents"`
- `status="skipped"`
- 原因包含 `No workspace target was provided`

在这种情况下，必须在执行前询问工作区指令问题。不得静默地将其视为跳过的决策。

由于上述限制，使用以下简化决策流程：

1. 对于 `SOUL.md` 冲突，使用 `clarify`，选项如：
   - `keep existing`
   - `overwrite with backup`
   - `review first`
2. 如果 dry run 显示一个或多个 `kind="skill"` 项的 `status="conflict"`，使用 `clarify`，选项如：
   - `keep existing skills`
   - `overwrite conflicting skills with backup`
   - `import conflicting skills under renamed folders`
3. 对于工作区指令，使用 `clarify`，选项如：
   - `skip workspace instructions`
   - `copy to a workspace path`
   - `decide later`
4. 如果用户选择复制工作区指令，追加一个开放式 `clarify` 问题，要求提供**绝对路径**。
5. 如果用户选择 `skip workspace instructions` 或 `decide later`，继续执行而不添加 `--workspace-target`。
5. 对于迁移模式，使用 `clarify`，提供以下 3 个选项：
   - `user-data only`
   - `full compatible migration`
   - `cancel`
6. `user-data only` 表示：迁移用户数据和兼容配置，但**不**导入白名单 secret。
7. `full compatible migration` 表示：迁移相同的兼容用户数据，并在存在时导入白名单 secret。
8. 如果 `clarify` 不可用，以普通文本提出相同问题，但仍将答案限制为 `user-data only`、`full compatible migration` 或 `cancel`。

执行门控：

- 当由 `No workspace target was provided` 导致的 `workspace-agents` 跳过仍未解决时，不得执行。
- 唯一有效的解决方式为：
  - 用户明确选择 `skip workspace instructions`
  - 用户明确选择 `decide later`
  - 用户在选择 `copy to a workspace path` 后提供了工作区路径
- dry run 中缺少工作区目标本身并不构成执行许可。
- 当任何必要的 `clarify` 决策仍未解决时，不得执行。

使用以下精确的 `clarify` payload 形式作为默认模式：

- `{"question":"Your existing SOUL.md conflicts with the imported one. What should I do?","choices":["keep existing","overwrite with backup","review first"]}`
- `{"question":"One or more imported OpenClaw skills already exist in Hermes. How should I handle those skill conflicts?","choices":["keep existing skills","overwrite conflicting skills with backup","import conflicting skills under renamed folders"]}`
- `{"question":"Choose migration mode: migrate only user data, or run the full compatible migration including allowlisted secrets?","choices":["user-data only","full compatible migration","cancel"]}`
- `{"question":"Do you want to copy the OpenClaw workspace instructions file into a Hermes workspace?","choices":["skip workspace instructions","copy to a workspace path","decide later"]}`
- `{"question":"Please provide an absolute path where the workspace instructions should be copied."}`

## 决策到命令的映射

将用户决策精确映射到命令标志：

- 如果用户对 `SOUL.md` 选择 `keep existing`，**不**添加 `--overwrite`。
- 如果用户选择 `overwrite with backup`，添加 `--overwrite`。
- 如果用户选择 `review first`，在执行前停止并审查相关文件。
- 如果用户选择 `keep existing skills`，添加 `--skill-conflict skip`。
- 如果用户选择 `overwrite conflicting skills with backup`，添加 `--skill-conflict overwrite`。
- 如果用户选择 `import conflicting skills under renamed folders`，添加 `--skill-conflict rename`。
- 如果用户选择 `user-data only`，使用 `--preset user-data` 执行，**不**添加 `--migrate-secrets`。
- 如果用户选择 `full compatible migration`，使用 `--preset full --migrate-secrets` 执行。
- 仅在用户明确提供绝对工作区路径时，才添加 `--workspace-target`。
- 如果用户选择 `skip workspace instructions` 或 `decide later`，不添加 `--workspace-target`。

执行前，用简洁语言重述精确的命令计划，并确保其与用户的选择一致。

## 运行后报告规则

执行后，将脚本的 JSON 输出作为事实来源。

1. 所有计数基于 `report.summary`。
2. 仅当 `status` 恰好为 `migrated` 时，才将该项列入"已成功迁移"。
3. 除非报告显示该项为 `migrated`，否则不得声称冲突已解决。
4. 除非 `kind="soul"` 的报告项 `status="migrated"`，否则不得声称 `SOUL.md` 已被覆盖。
5. 如果 `report.summary.conflict > 0`，包含冲突部分，而非静默暗示成功。
6. 如果计数与列出的项不一致，在回复前修正列表以匹配报告。
7. 在可用时包含报告中的 `output_dir` 路径，以便用户检查 `report.json`、`summary.md`、备份和归档文件。
8. 对于记忆或用户档案溢出，除非报告明确显示归档路径，否则不得声称条目已被归档。如果 `details.overflow_file` 存在，说明完整溢出列表已导出到该位置。
9. 如果 skill 以重命名文件夹导入，报告最终目标并提及 `details.renamed_from`。
10. 如果 `report.skill_conflict_mode` 存在，将其作为所选已导入 skill 冲突策略的事实来源。
11. 如果某项 `status="skipped"`，不得将其描述为已覆盖、已备份、已迁移或已解决。
12. 如果 `kind="soul"` 的 `status="skipped"` 且原因为 `Target already matches source`，说明其保持不变，不提及备份。
13. 如果重命名的已导入 skill 的 `details.backup` 为空，不得暗示现有 Hermes skill 已被重命名或备份。仅说明已导入的副本被放置在新目标位置，并将 `details.renamed_from` 作为保持原位的已有文件夹引用。

## 迁移 preset

正常使用时优先选择以下两个 preset：

- `user-data`
- `full`

`user-data` 包含：

- `soul`
- `workspace-agents`
- `memory`
- `user-profile`
- `messaging-settings`
- `command-allowlist`
- `skills`
- `tts-assets`
- `archive`

`full` 包含 `user-data` 中的所有内容，另加：

- `secret-settings`

辅助脚本仍支持类别级别的 `--include` / `--exclude`，但将其视为高级备用方案，而非默认用户体验。

## 命令

完整发现的 dry run：

```bash
python3 ~/.hermes/skills/migration/openclaw-migration/scripts/openclaw_to_hermes.py
```

使用终端工具时，优先使用绝对调用模式，例如：

```json
{"command":"python3 /home/USER/.hermes/skills/migration/openclaw-migration/scripts/openclaw_to_hermes.py","workdir":"/home/USER"}
```

使用 user-data preset 的 dry run：

```bash
python3 ~/.hermes/skills/migration/openclaw-migration/scripts/openclaw_to_hermes.py --preset user-data
```

执行 user-data 迁移：

```bash
python3 ~/.hermes/skills/migration/openclaw-migration/scripts/openclaw_to_hermes.py --execute --preset user-data --skill-conflict skip
```

执行完整兼容迁移：

```bash
python3 ~/.hermes/skills/migration/openclaw-migration/scripts/openclaw_to_hermes.py --execute --preset full --migrate-secrets --skill-conflict skip
```

包含工作区指令的执行：

```bash
python3 ~/.hermes/skills/migration/openclaw-migration/scripts/openclaw_to_hermes.py --execute --preset user-data --skill-conflict rename --workspace-target "/absolute/workspace/path"
```

默认情况下不要使用 `$PWD` 或主目录作为工作区目标。请先明确询问工作区路径。

## 重要规则

1. 除非用户明确表示立即执行，否则在写入前先运行 dry run。
2. 默认不迁移 secret。Token、认证 blob、设备凭据和原始 gateway 配置应保留在 Hermes 之外，除非用户明确要求迁移 secret。
3. 除非用户明确要求，否则不得静默覆盖非空的 Hermes 目标。辅助脚本在启用覆盖时会保留备份。
4. 始终向用户提供跳过项报告。该报告是迁移的一部分，而非可选附加内容。
5. 优先使用主 OpenClaw 工作区（`~/.openclaw/workspace/`）而非 `workspace.default/`。仅在主文件缺失时才使用默认工作区作为备用。
6. 即使在 secret 迁移模式下，也只迁移具有干净 Hermes 目标的 secret。不支持的认证 blob 仍须报告为已跳过。
7. 如果 dry run 显示大型资产复制、冲突的 `SOUL.md` 或溢出的记忆条目，在执行前单独指出这些情况。
8. 如果用户不确定，默认选择 `user-data only`。
9. 仅在用户明确提供目标工作区路径时，才包含 `workspace-agents`。
10. 将类别级别的 `--include` / `--exclude` 视为高级逃生通道，而非正常流程。
11. 如果 `clarify` 可用，不得在 dry run 摘要结尾使用含糊的"您希望怎么做？"。改用结构化的后续提示。
12. 当真实选择提示可用时，不要使用开放式 `clarify` 提示。优先使用可选选项，仅对绝对路径或文件审查请求使用自由文本。
13. dry run 后，如果仍有未解决的决策，不得在摘要后停止。立即对最高优先级的阻塞决策使用 `clarify`。
14. 后续问题的优先顺序：
    - `SOUL.md` 冲突
    - 已导入 skill 冲突
    - 迁移模式
    - 工作区指令目标
15. 不得在同一消息中承诺稍后呈现选项。通过实际调用 `clarify` 来呈现它们。
16. 在收到迁移模式答案后，明确检查 `workspace-agents` 是否仍未解决。如果是，下一个动作必须是工作区指令的 `clarify` 调用。
17. 在任何 `clarify` 答案之后，如果还有其他必要决策待处理，不要叙述刚刚决定的内容。立即提出下一个必要问题。

## 预期结果

成功运行后，用户应拥有：

- 已导入的 Hermes persona 状态
- 已填充转换后 OpenClaw 知识的 Hermes 记忆文件
- 在 `~/.hermes/skills/openclaw-imports/` 下可用的 OpenClaw skill
- 显示任何冲突、遗漏或不支持数据的迁移报告