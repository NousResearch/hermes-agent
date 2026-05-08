---
title: "Debugging Hermes Tui Commands — 调试 Hermes TUI 斜杠命令：Python、网关、Ink UI"
sidebar_label: "Debugging Hermes Tui Commands"
description: "调试 Hermes TUI 斜杠命令：Python、网关、Ink UI"
---

{/* 本页面由 website/scripts/generate-skill-docs.py 从技能的 SKILL.md 自动生成。请编辑源 SKILL.md，而非此页面。 */}

# 调试 Hermes TUI 斜杠命令

调试 Hermes TUI 斜杠命令：Python、网关、Ink UI。

## 技能元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/software-development/debugging-hermes-tui-commands` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 标签 | `debugging`, `hermes-agent`, `tui`, `slash-commands`, `typescript`, `python` |
| 相关技能 | [`python-debugpy`](/docs/user-guide/skills/bundled/software-development/software-development-python-debugpy), [`node-inspect-debugger`](/docs/user-guide/skills/bundled/software-development/software-development-node-inspect-debugger), [`systematic-debugging`](/docs/user-guide/skills/bundled/software-development/software-development-systematic-debugging) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在此技能被触发时加载的完整技能定义。这是代理在技能激活时看到的指令。
:::

# 调试 Hermes TUI 斜杠命令

## 概述

Hermes 斜杠命令跨三个层级——Python 命令注册表、tui_gateway JSON-RPC 桥接和 Ink/TypeScript 前端。当命令表现异常（自动补全中缺失、CLI 中可用但 TUI 中不可用、配置已保存但 UI 不更新）时，Bug 几乎总是某一层与另一层不同步。

当你在 Hermes TUI 中遇到斜杠命令问题时，使用此技能，特别是当命令未显示在自动补全中、在 TUI 中不正常工作或需要添加/更新时。

## 使用时机

- 斜杠命令存在于代码库的一部分但不能完全工作
- 需要在后端和前端都添加命令
- 特定命令的自动补全不工作
- 命令行为在 CLI 和 TUI 之间不一致
- 命令保存了配置但未在 TUI 中实时应用

## 架构概述

<!-- ascii-guard-ignore -->
```
Python 后端 (hermes_cli/commands.py)     <- 规范的 COMMAND_REGISTRY
       │
       ▼
TUI 网关 (tui_gateway/server.py)         <- slash.exec / command.dispatch
       │
       ▼
TUI 前端 (ui-tui/src/app/slash/)        <- 本地处理器 + 透传
```
<!-- ascii-guard-ignore-end -->

命令定义必须在 Python 和 TypeScript 之间一致注册才能正常工作。Python `COMMAND_REGISTRY` 是以下内容的真实来源：CLI 分发、网关帮助、Telegram BotCommand 菜单、Slack 子命令映射以及发送到 Ink 的自动补全数据。

## 调查步骤

1. **检查命令是否存在于 TUI 前端：**
   ```bash
   search_files --pattern "/commandname" --file_glob "*.ts" --path ui-tui/
   search_files --pattern "/commandname" --file_glob "*.tsx" --path ui-tui/
   ```

2. **检查 TUI 命令定义：**
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

4. **检查网关实现：**
   ```bash
   search_files --pattern "complete.slash|slash.exec" --path tui_gateway/
   ```

## 修复：缺少命令自动补全

如果命令存在于 TUI 但不在自动补全中显示：

1. 在 `hermes_cli/commands.py` 的 `COMMAND_REGISTRY` 中添加 `CommandDef` 条目：
   ```python
   CommandDef("commandname", "Description of the command", "Session",
              cli_only=True, aliases=("alias",),
              args_hint="[arg1|arg2|arg3]",
              subcommands=("arg1", "arg2", "arg3")),
   ```

2. 仔细选择 `cli_only` 与网关可用性：
   - `cli_only=True` ——仅在交互式 CLI/TUI 中
   - `gateway_only=True` ——仅在消息平台中
   - 都不设置 ——到处可用
   - `gateway_config_gate="display.foo"` ——网关中基于配置的可用性

3. 确保 `subcommands` 与 TUI 显示的预期 Tab 补全选项匹配。

4. 如果命令在服务端运行，在 `cli.py` 的 `HermesCLI.process_command()` 中添加处理器：
   ```python
   elif canonical == "commandname":
       self._handle_commandname(cmd_original)
   ```

5. 对于网关可用的命令，在 `gateway/run.py` 中添加处理器：
   ```python
   if canonical == "commandname":
       return await self._handle_commandname(event)
   ```

## 常见问题

1. **命令显示在 TUI 中但不在自动补全中。** 命令定义在 TUI 代码库中，但缺少 `hermes_cli/commands.py` 中 `COMMAND_REGISTRY` 的条目。自动补全数据从 Python 发送。

2. **命令显示在自动补全中但不工作。** 检查 `tui_gateway/server.py` 中的命令处理器和 `ui-tui/src/app/createSlashHandler.ts` 中的前端处理器。如果命令是 Ink 本地命令，它必须在 `app.tsx` 内置分支中处理；否则会透传到 `slash.exec` 并必须有一个 Python 处理器。

3. **命令行为在 CLI 和 TUI 之间不同。** 命令可能有不同的实现。检查 `cli.py::process_command` 和 TUI 的本地处理器。本地 TUI 处理器优先于网关分发。

4. **命令保存了配置但未实时应用。** 对于 TUI 本地命令，仅更新 `config.set` 是不够的。还需立即修补相关的 nanostore 状态（通常是 `patchUiState(...)`），并通过渲染组件传递任何新状态。

5. **网关分发静默忽略命令。** 网关只分发它知道的命令。检查 `GATEWAY_KNOWN_COMMANDS`（自动从 `COMMAND_REGISTRY` 派生）是否包含规范名称。如果命令是带 `gateway_config_gate` 的 `cli_only`，验证受控配置值是否为真。

## 调试策略

当表面检查无法揭示 Bug 时：

- **Python 端挂起或异常：** 使用 `python-debugpy` 技能在 `_SlashWorker.exec` 或命令处理器中设置断点。在处理器入口设置 `remote-pdb` 是最快的方式。
- **Ink 端无反应：** 使用 `node-inspect-debugger` 技能在 `app.tsx` 的斜杠分发或本地命令分支中设置断点。`npm run build` 后使用 `sb('dist/app.js', <line>)`。
- **注册表不匹配/不清楚哪边有误：** 将规范的 `COMMAND_REGISTRY` 条目与 TUI 的本地命令列表并排比较。

## 常见陷阱

- 不要忘记在 `CommandDef` 中为命令设置适当的类别（例如 "Session"、"Configuration"、"Tools & Skills"、"Info"、"Exit"）
- 确保所有别名在 `aliases` 元组中正确注册——不需要修改其他文件，所有下游（Telegram 菜单、Slack 映射、自动补全、帮助）都由此派生
- 对于有子命令的命令，确保 `CommandDef` 中的 `subcommands` 元组与 TUI 代码中的内容匹配
- `cli_only=True` 的命令在网关/消息平台中不起作用——除非你添加了 `gateway_config_gate` 且门控值为真
- 添加实时 UI 状态后，搜索旧 prop/helper 的每个使用者并将新状态传递到所有渲染路径，而不仅是活动的流式传输路径
- 测试前重建 TUI（`npm --prefix ui-tui run build`）——tsx watch 模式在首次启动时可能滞后

## 验证

修复后：

1. 重建 TUI：
   ```bash
   cd /home/bb/hermes-agent && npm --prefix ui-tui run build
   ```

2. 运行 TUI 并测试命令：
   ```bash
   hermes --tui
   ```

3. 输入 `/` 并验证命令出现在自动补全建议中，包含预期的描述和参数提示。

4. 执行命令并确认：
   - 预期行为触发
   - 任何持久化的配置正确更新（`read_file ~/.hermes/config.yaml`）
   - 实时 UI 状态立即反映更改（而非仅在重启后）

5. 如果命令也可在网关使用，从至少一个消息平台进行测试（或运行网关测试：`scripts/run_tests.sh tests/gateway/`）。
