---
title: "Debugging Hermes Tui Commands — Debug Hermes TUI slash commands: Python, gateway, Ink UI"
sidebar_label: "Debugging Hermes Tui Commands"
description: "调试 Hermes TUI slash 命令：Python、gateway、Ink UI"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# 调试 Hermes TUI 命令

调试 Hermes TUI slash（斜杠）命令：Python、gateway、Ink UI。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/software-development/debugging-hermes-tui-commands` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `debugging`, `hermes-agent`, `tui`, `slash-commands`, `typescript`, `python` |
| 相关 skill | [`python-debugpy`](/user-guide/skills/bundled/software-development/software-development-python-debugpy)、[`node-inspect-debugger`](/user-guide/skills/bundled/software-development/software-development-node-inspect-debugger)、[`systematic-debugging`](/user-guide/skills/bundled/software-development/software-development-systematic-debugging) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发该 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# 调试 Hermes TUI Slash 命令

## 概述

Hermes slash 命令跨越三个层次——Python 命令注册表、tui_gateway JSON-RPC 桥接层，以及 Ink/TypeScript 前端。当某个命令出现异常（不在自动补全中显示、在 CLI 中正常但在 TUI 中不工作、配置已持久化但 UI 未更新），问题几乎总是某一层与另一层不同步所致。

当你在 Hermes TUI 中遇到 slash 命令问题时使用本 skill，尤其是命令未出现在自动补全中、在 TUI 中无法正常工作，或需要添加/更新命令时。

## 适用场景

- slash 命令存在于代码库的某一部分，但未完全生效
- 需要同时在后端和前端添加某个命令
- 特定命令的自动补全不工作
- 命令在 CLI 和 TUI 之间行为不一致
- 命令已持久化配置，但未在 TUI 中实时生效

## 架构概览

<!-- ascii-guard-ignore -->
```
Python backend (hermes_cli/commands.py)     <- 规范的 COMMAND_REGISTRY
       │
       ▼
TUI gateway (tui_gateway/server.py)         <- slash.exec / command.dispatch
       │
       ▼
TUI frontend (ui-tui/src/app/slash/)        <- 本地处理器 + fallthrough
```
<!-- ascii-guard-ignore-end -->

命令定义必须在 Python 和 TypeScript 中保持一致注册才能正常工作。Python 的 `COMMAND_REGISTRY` 是以下内容的唯一真实来源：CLI 分发、gateway 帮助、Telegram BotCommand 菜单、Slack 子命令映射，以及发送给 Ink 的自动补全数据。

## 排查步骤

1. **检查命令是否存在于 TUI 前端：**
   ```bash
   search_files --pattern "/commandname" --file_glob "*.ts" --path ui-tui/
   search_files --pattern "/commandname" --file_glob "*.tsx" --path ui-tui/
   ```

2. **查看 TUI 命令定义：**
   ```bash
   read_file ui-tui/src/app/slash/commands/core.ts
   # 如果不在那里：
   search_files --pattern "commandname" --path ui-tui/src/app/slash/commands --target files
   ```

3. **检查命令是否存在于 Python 后端：**
   ```bash
   search_files --pattern "CommandDef" --file_glob "*.py" --path hermes_cli/
   search_files --pattern "commandname" --path hermes_cli/commands.py --context 3
   ```

4. **查看 gateway 实现：**
   ```bash
   search_files --pattern "complete.slash|slash.exec" --path tui_gateway/
   ```

## 修复：命令自动补全缺失

如果命令存在于 TUI 但未出现在自动补全中：

1. 在 `hermes_cli/commands.py` 的 `COMMAND_REGISTRY` 中添加 `CommandDef` 条目：
   ```python
   CommandDef("commandname", "Description of the command", "Session",
              cli_only=True, aliases=("alias",),
              args_hint="[arg1|arg2|arg3]",
              subcommands=("arg1", "arg2", "arg3")),
   ```

2. 谨慎选择 `cli_only` 与 gateway 可用性：
   - `cli_only=True` — 仅在交互式 CLI/TUI 中可用
   - `gateway_only=True` — 仅在消息平台中可用
   - 两者均不设置 — 所有地方均可用
   - `gateway_config_gate="display.foo"` — 在 gateway 中受配置项控制的可用性

3. 确保 `subcommands` 与 TUI 显示的预期 tab 补全选项一致。

4. 如果命令在服务端运行，在 `cli.py` 的 `HermesCLI.process_command()` 中添加处理器：
   ```python
   elif canonical == "commandname":
       self._handle_commandname(cmd_original)
   ```

5. 对于 gateway 可用的命令，在 `gateway/run.py` 中添加处理器：
   ```python
   if canonical == "commandname":
       return await self._handle_commandname(event)
   ```

## 常见问题

1. **命令在 TUI 中显示但不在自动补全中。** 命令已在 TUI 代码库中定义，但 `hermes_cli/commands.py` 的 `COMMAND_REGISTRY` 中缺失。自动补全数据由 Python 端提供。

2. **命令在自动补全中显示但不工作。** 检查 `tui_gateway/server.py` 中的命令处理器，以及 `ui-tui/src/app/createSlashHandler.ts` 中的前端处理器。如果命令在 Ink 中是纯本地命令，必须在 `app.tsx` 的内置分支中处理；否则会 fallthrough 到 `slash.exec`，必须有对应的 Python 处理器。

3. **命令在 CLI 和 TUI 之间行为不同。** 该命令可能有不同的实现。同时检查 `cli.py::process_command` 和 TUI 的本地处理器。TUI 本地处理器优先于 gateway 分发。

4. **命令已持久化配置但未实时生效。** 对于 TUI 本地命令，仅更新 `config.set` 是不够的。还需立即修改相关的 nanostore 状态（通常是 `patchUiState(...)`），并将新状态传递给所有渲染组件。示例：`/details collapsed` 必须实时更新详情可见性，而不仅仅是保存 `details_mode`；会话内全局 `/details <mode>` 可能需要单独的命令覆盖标志，以便实时命令能覆盖内置分区默认值，同时启动/配置同步保留默认展开的 thinking/tools 行为。

5. **Gateway 分发静默忽略命令。** Gateway 只分发它已知的命令。检查 `GATEWAY_KNOWN_COMMANDS`（自动从 `COMMAND_REGISTRY` 派生）是否包含规范名称。如果命令是带有 `gateway_config_gate` 的 `cli_only`，验证被门控的配置值是否为真值。

## 调试策略

当表层排查无法定位问题时：

- **Python 端挂起或行为异常：** 使用 `python-debugpy` skill 在 `_SlashWorker.exec` 或命令处理器内设置断点。在处理器入口处设置 `remote-pdb` 是最快的方式。
- **Ink 端无响应：** 使用 `node-inspect-debugger` skill 在 `app.tsx` 的 slash 分发或本地命令分支处设置断点。`npm run build` 后执行 `sb('dist/app.js', <line>)`。
- **注册表不匹配/不清楚哪一侧有问题：** 将规范的 `COMMAND_REGISTRY` 条目与 TUI 的本地命令列表并排比较。

## 注意事项

- 不要忘记在 `CommandDef` 中为命令设置适当的分类（例如 "Session"、"Configuration"、"Tools & Skills"、"Info"、"Exit"）
- 确保所有别名都正确注册在 `aliases` 元组中——无需修改其他文件，下游所有内容（Telegram 菜单、Slack 映射、自动补全、帮助）均从此派生
- 对于带子命令的命令，确保 `CommandDef` 中的 `subcommands` 元组与 TUI 代码中的内容一致
- `cli_only=True` 的命令在 gateway/消息平台中不可用——除非添加 `gateway_config_gate` 且该门控值为真
- 添加实时 UI 状态后，搜索旧 prop/helper 的所有消费者，并将新状态贯穿所有渲染路径，而不仅仅是活跃的流式路径。TUI 详情渲染至少有两条重要路径：实时的 `StreamingAssistant`/`ToolTrail` 和转录/待处理的 `MessageLine` 行。`/clean` 操作应明确检查两者。
- 测试前重新构建 TUI（`npm --prefix ui-tui run build`）——tsx watch 模式在首次启动时可能有延迟

## 验证

修复后：

1. 重新构建 TUI：
   ```bash
   cd /home/bb/hermes-agent && npm --prefix ui-tui run build
   ```

2. 运行 TUI 并测试命令：
   ```bash
   hermes --tui
   ```

3. 输入 `/` 并验证命令出现在自动补全建议中，且显示预期的描述和参数提示。

4. 执行命令并确认：
   - 预期行为已触发
   - 所有持久化配置正确更新（`read_file ~/.hermes/config.yaml`）
   - 实时 UI 状态立即反映变更（而非重启后才生效）

5. 如果命令也支持 gateway，至少在一个消息平台上测试（或运行 gateway 测试：`scripts/run_tests.sh tests/gateway/`）。