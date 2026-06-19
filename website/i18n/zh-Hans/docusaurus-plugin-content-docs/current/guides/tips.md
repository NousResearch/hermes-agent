---
sidebar_position: 1
title: "技巧与最佳实践"
description: "充分发挥 Hermes Agent 潜力的实用建议——prompt 技巧、CLI 快捷键、上下文文件、记忆、成本优化与安全"
---

# 技巧与最佳实践

一份实用技巧速查集，帮助你立即提升使用 Hermes Agent 的效率。每个章节针对不同方面——扫描标题，直接跳到相关内容。

---

## 获得最佳结果

### 明确说明你的需求

模糊的 prompt（提示词）只会产生模糊的结果。不要说"修复代码"，而要说"修复 `api/handlers.py` 第 47 行的 TypeError——`process_request()` 函数从 `parse_body()` 收到了 `None`。"给出的上下文越多，所需的迭代次数就越少。

### 预先提供上下文

在请求开头就给出相关细节：文件路径、错误信息、预期行为。一条精心构造的消息胜过三轮来回确认。直接粘贴错误堆栈——agent 能够解析它们。

### 使用上下文文件处理重复指令

如果你发现自己在反复输入相同的指令（"用 tab 而非空格"、"我们用 pytest"、"API 地址是 `/api/v2`"），把它们放进 `AGENTS.md` 文件。agent 每次会话都会自动读取它——设置一次，永久生效。

### 让 Agent 使用它的工具

不要试图手把手指导每一步。说"找到并修复失败的测试"，而不是"打开 `tests/test_foo.py`，看第 42 行，然后……"。agent 拥有文件搜索、终端访问和代码执行能力——让它自行探索和迭代。

### 对复杂工作流使用 Skill

在写一大段 prompt 解释如何做某件事之前，先检查是否已有对应的 skill。输入 `/skills` 浏览可用的 skill，或直接调用，例如 `/axolotl` 或 `/github-pr-workflow`。

## CLI 高级用户技巧

### 多行输入

按 **Alt+Enter**、**Ctrl+J** 或 **Shift+Enter** 可插入换行而不发送消息。`Shift+Enter` 仅在终端将其作为独立按键发送时有效（Kitty / foot / WezTerm / Ghostty 默认支持；iTerm2 / Alacritty / VS Code 终端需启用 Kitty 键盘协议）。另外两种方式在所有终端中均可使用。

### 粘贴检测

CLI 会自动检测多行粘贴。直接粘贴代码块或错误堆栈——不会将每行作为单独消息发送。粘贴内容会被缓冲后作为一条消息发送。

### 中断与重定向

按一次 **Ctrl+C** 可中断 agent 的响应过程，然后输入新消息重新引导它。在 2 秒内双击 Ctrl+C 可强制退出。当 agent 开始走错方向时，这个功能非常有用。

### 使用 `-c` 恢复会话

上次会话有遗漏？运行 `hermes -c` 可精确恢复到上次离开的位置，完整对话历史全部还原。也可以按标题恢复：`hermes -r "my research project"`。

### 剪贴板图片粘贴

按 **Ctrl+V** 可将剪贴板中的图片直接粘贴到对话中。agent 会使用视觉能力分析截图、图表、错误弹窗或 UI 原型——无需先保存为文件。

### Slash 命令自动补全

输入 `/` 后按 **Tab** 可查看所有可用命令，包括内置命令（`/compress`、`/model`、`/title`）和所有已安装的 skill。无需记忆任何内容——Tab 补全全部搞定。

:::tip
使用 `/verbose` 循环切换工具输出显示模式：**off → new → all → verbose**。"all" 模式非常适合观察 agent 的操作过程；"off" 模式在简单问答时最为简洁。
:::

## 上下文文件

### AGENTS.md：你的项目大脑

在项目根目录创建 `AGENTS.md`，写入架构决策、编码规范和项目专属指令。该文件会自动注入每次会话，让 agent 始终了解你的项目规则。

```markdown
# Project Context
- This is a FastAPI backend with SQLAlchemy ORM
- Always use async/await for database operations
- Tests go in tests/ and use pytest-asyncio
- Never commit .env files
```

### SOUL.md：自定义个性

想让 Hermes 拥有稳定的默认风格？编辑 `~/.hermes/SOUL.md`（如果使用自定义 Hermes home，则为 `$HERMES_HOME/SOUL.md`）。Hermes 现在会自动生成一个初始 SOUL 文件，并将该全局文件作为实例级个性来源。

完整说明请参阅 [在 Hermes 中使用 SOUL.md](/guides/use-soul-with-hermes)。

```markdown
# Soul
You are a senior backend engineer. Be terse and direct.
Skip explanations unless asked. Prefer one-liners over verbose solutions.
Always consider error handling and edge cases.
```

使用 `SOUL.md` 设置持久个性，使用 `AGENTS.md` 设置项目专属指令。

### .cursorrules 兼容性

已有 `.cursorrules` 或 `.cursor/rules/*.mdc` 文件？Hermes 同样会读取它们。无需重复编写编码规范——这些文件会从工作目录自动加载。

### 发现机制

Hermes 在会话启动时从当前工作目录加载顶层 `AGENTS.md`。子目录中的 `AGENTS.md` 文件在工具调用期间通过 `subdirectory_hints.py` 延迟发现，并注入工具结果——不会在启动时预先加载到系统 prompt 中。

:::tip
保持上下文文件简洁聚焦。每个字符都会消耗 token 配额，因为它们会注入到每一条消息中。
:::

## 记忆与 Skill

### 记忆 vs. Skill：各司其职

**记忆（Memory）** 用于存储事实：你的环境、偏好、项目位置，以及 agent 了解到的关于你的信息。**Skill** 用于存储流程：多步骤工作流、特定工具的操作指南和可复用的操作方案。记忆存"是什么"，skill 存"怎么做"。

### 何时创建 Skill

如果某个任务需要 5 步以上且你会重复执行，就让 agent 为它创建一个 skill。说"把你刚才做的保存为名为 `deploy-staging` 的 skill"。下次只需输入 `/deploy-staging`，agent 就会加载完整流程。

### 管理记忆容量

记忆容量是有意限制的（`MEMORY.md` 约 2,200 字符，`USER.md` 约 1,375 字符）。当记忆填满时，agent 会自动整合条目。你也可以主动说"清理你的记忆"或"替换旧的 Python 3.9 备注——我们现在用 3.12 了"。

### 让 Agent 记住内容

在一次高效的会话结束后，说"记住这些以备下次使用"，agent 会保存关键要点。也可以具体指定："保存到记忆中，我们的 CI 使用 GitHub Actions 的 `deploy.yml` 工作流。"

:::warning
记忆是一个冻结的快照——会话期间的修改不会出现在系统 prompt 中，直到下一次会话开始。agent 会立即写入磁盘，但 prompt 缓存在会话中途不会失效。
:::

## 性能与成本

### 不要破坏 Prompt 缓存

大多数 LLM 提供商会缓存系统 prompt 前缀。如果你保持系统 prompt 稳定（相同的上下文文件、相同的记忆），同一会话中的后续消息会命中**缓存**，成本显著降低。避免在会话中途切换模型或修改系统 prompt。

### 在达到限制前使用 /compress

长会话会积累大量 token。当你发现响应变慢或被截断时，运行 `/compress`。这会对对话历史进行摘要，在大幅减少 token 数量的同时保留关键上下文。使用 `/usage` 查看当前用量。

### 使用委托实现并行工作

需要同时研究三个主题？让 agent 使用 `delegate_task` 并行分配子任务。每个子 agent 独立运行，拥有各自的上下文，最终只有摘要结果返回——大幅减少主对话的 token 消耗。

### 使用 execute_code 进行批量操作

不要逐条运行终端命令，而是让 agent 编写一个脚本一次性完成所有操作。"写一个 Python 脚本把所有 `.jpeg` 文件重命名为 `.jpg` 并运行它"比逐个重命名文件更省钱、更快速。

### 选择合适的模型

使用 `/model` 在会话中途切换模型。对于复杂推理和架构决策，使用前沿模型（Claude Sonnet/Opus、GPT-4o）；对于格式化、重命名或样板代码生成等简单任务，切换到更快的模型。

:::tip
定期运行 `/usage` 查看 token 消耗情况。运行 `/insights` 可查看过去 30 天的用量模式概览。
:::

## 消息技巧

### 设置主频道

在你偏好的 Telegram 或 Discord 聊天中使用 `/sethome`，将其指定为主频道。定时任务结果和计划任务输出会发送到这里。没有主频道，agent 就没有地方发送主动消息。

### 使用 /title 整理会话

用 `/title auth-refactor` 或 `/title research-llm-quantization` 为会话命名。命名后的会话可通过 `hermes sessions list` 轻松找到，并用 `hermes -r "auth-refactor"` 恢复。未命名的会话会堆积起来，难以区分。

### DM 配对实现团队访问

不要手动收集用户 ID 来维护白名单，而是启用 DM 配对。当团队成员向 bot 发送私信时，他们会收到一次性配对码。你用 `hermes pairing approve telegram XKGH5N7P` 批准即可——简单且安全。

### 工具进度显示模式

使用 `/verbose` 控制工具活动的显示详细程度。在消息平台上，通常越简洁越好——保持"new"模式只查看新的工具调用。在 CLI 中，"all" 模式可以实时查看 agent 的所有操作。

:::tip
在消息平台上，会话会在空闲一段时间后自动重置（默认 24 小时），或每天凌晨 4 点重置。如需更长的会话时间，可在 `~/.hermes/config.yaml` 中按平台调整。
:::

## 安全

### 对不可信代码使用 Docker

在处理不可信仓库或运行陌生代码时，使用 Docker 或 Daytona 作为终端后端。在 `.env` 中设置 `TERMINAL_BACKEND=docker`。容器内的破坏性命令不会影响宿主系统。

```bash
# In your .env:
TERMINAL_BACKEND=docker
TERMINAL_DOCKER_IMAGE=hermes-sandbox:latest
```

### 避免 Windows 编码陷阱

在 Windows 上，某些默认编码（如 `cp125x`）无法表示所有 Unicode 字符，在测试或脚本中写入文件时可能导致 `UnicodeEncodeError`。

- 建议在打开文件时显式指定 UTF-8 编码：

```python
with open("results.txt", "w", encoding="utf-8") as f:
    f.write("✓ All good\n")
```

- 在 PowerShell 中，也可以将当前会话的控制台和原生命令输出切换为 UTF-8：

```powershell
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
```

这样可以让 PowerShell 和子进程统一使用 UTF-8，避免仅在 Windows 上出现的失败。

### 谨慎选择"始终允许"

当 agent 触发危险命令审批（`rm -rf`、`DROP TABLE` 等）时，你有四个选项：**once（仅此一次）**、**session（本次会话）**、**always（始终允许）**、**deny（拒绝）**。选择"always"前请仔细考虑——它会永久将该模式加入白名单。在熟悉之前，先用"session"。

### 命令审批是你的安全防线

Hermes 在执行每条命令前都会与一份精心维护的危险模式列表进行比对，包括递归删除、SQL DROP、curl 管道到 shell 等。不要在生产环境中禁用此功能——它的存在有充分的理由。

:::warning
在容器后端（Docker、Singularity、Modal、Daytona）中运行时，危险命令检查会被**跳过**，因为容器本身就是安全边界。请确保你的容器镜像已妥善加固。
:::

### 为消息 Bot 使用白名单

永远不要在拥有终端访问权限的 bot 上设置 `GATEWAY_ALLOW_ALL_USERS=true`。始终使用平台专属白名单（`TELEGRAM_ALLOWED_USERS`、`DISCORD_ALLOWED_USERS`）或 DM 配对来控制谁可以与你的 agent 交互。

```bash
# Recommended: explicit allowlists per platform
TELEGRAM_ALLOWED_USERS=123456789,987654321
DISCORD_ALLOWED_USERS=123456789012345678

# Or use cross-platform allowlist
GATEWAY_ALLOWED_USERS=123456789,987654321
```

---

*有值得收录的技巧？欢迎提交 issue 或 PR——社区贡献随时欢迎。*