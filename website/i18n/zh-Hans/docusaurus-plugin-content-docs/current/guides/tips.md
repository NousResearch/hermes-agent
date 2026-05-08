---
sidebar_position: 1
title: "技巧与最佳实践"
description: "实用建议，让你可以充分利用 Hermes 代理——提示技巧、CLI 快捷方式、上下文文件、记忆、成本优化和安全"
---

# 技巧与最佳实践

实用技巧的快速成功集合，让你立即更有效地使用 Hermes 代理。每个部分针对不同方面——扫描标题并跳转到相关内容。

---

## 获得最佳结果

### 具体说明你想要什么

模糊的提示产生模糊的结果。不要说"修复代码"，而要说"修复 `api/handlers.py` 第 47 行的 TypeError——`process_request()` 函数从 `parse_body()` 接收到 `None`。"你给的上下文越多，需要的迭代次数就越少。

### 预先提供上下文

在请求前面加上相关细节：文件路径、错误消息、预期行为。一条精心制作的消息胜过三轮澄清。直接粘贴错误追踪——代理可以解析它们。

### 对重复指令使用上下文文件

如果你发现自己重复相同的指令（"使用制表符而不是空格"、"我们使用 pytest"、"API 在 `/api/v2`"），将它们放在 `AGENTS.md` 文件中。代理在每个会话中自动读取它——设置后零努力。

### 让代理使用其工具

不要试图手把手指导每个步骤。说"找到并修复失败的测试"而不是"打开 `tests/test_foo.py`，看第 42 行，然后……"代理有文件搜索、终端访问和代码执行——让它探索和迭代。

### 对复杂工作流使用技能

在编写长提示解释如何做某事之前，检查是否已经有相关的技能。输入 `/skills` 浏览可用的技能，或直接调用一个如 `/axolotl` 或 `/github-pr-workflow`。

## CLI 高级用户技巧

### 多行输入

按 **Alt+Enter**（或 **Ctrl+J**）插入换行符而不发送。这让你在按 Enter 发送之前可以编写多行提示、粘贴代码块或结构化复杂请求。

### 粘贴检测

CLI 自动检测多行粘贴。直接粘贴代码块或错误追踪——它不会将每一行作为单独消息发送。粘贴被缓冲并作为一条消息发送。

### 中断并重定向

按 **Ctrl+C** 一次以在代理响应中途中断。然后你可以输入新消息来重定向它。在 2 秒内双按 Ctrl+C 强制退出。当代理开始走错路时，这非常宝贵。

### 使用 `-c` 恢复会话

忘记了上次会话的一些内容？运行 `hermes -c` 精确地从你停止的地方恢复，完整对话历史已恢复。你也可以按标题恢复：`hermes -r "my research project"`。

### 剪贴板图像粘贴

按 **Ctrl+V** 直接将图像从剪贴板粘贴到聊天中。代理使用视觉来分析截图、图表、错误弹窗或 UI 模型——无需先保存到文件。

### 斜杠命令自动完成

输入 `/` 并按 **Tab** 查看所有可用命令。这包括内置命令（`/compress`、`/model`、`/title`）和每个已安装的技能。你不需要记住任何东西——Tab 补全会帮你搞定。

:::tip
使用 `/verbose` 循环切换工具输出显示模式：**off → new → all → verbose**。"all" 模式非常适合观察代理做什么；"off" 对于简单的问答最干净。
:::

## 上下文文件

### AGENTS.md：你项目的脑

在项目根目录创建 `AGENTS.md`，包含架构决策、编码约定和项目特定指令。这会自动注入到每个会话中，因此代理始终了解你项目的规则。

```markdown
# Project Context
- This is a FastAPI backend with SQLAlchemy ORM
- Always use async/await for database operations
- Tests go in tests/ and use pytest-asyncio
- Never commit .env files
```

### SOUL.md：自定义个性

想让 Hermes 有稳定的默认声音？编辑 `~/.hermes/SOUL.md`（如果你使用自定义 Hermes home，则为 `$HERMES_HOME/SOUL.md`）。Hermes 现在自动生成一个起始 SOUL 并使用该全局文件作为实例范围的个性来源。

有关完整演练，请参阅[将 SOUL.md 与 Hermes 一起使用](/docs/guides/use-soul-with-hermes)。

```markdown
# Soul
You are a senior backend engineer. Be terse and direct.
Skip explanations unless asked. Prefer one-liners over verbose solutions.
Always consider error handling and edge cases.
```

使用 `SOUL.md` 获得持久个性。使用 `AGENTS.md` 获得项目特定指令。

### .cursorrules 兼容性

已经有 `.cursorrules` 或 `.cursor/rules/*.mdc` 文件？Hermes 也会读取它们。无需重复你的编码约定——它们从工作目录自动加载。

### 发现

Hermes 在会话开始时从当前工作目录加载顶级 `AGENTS.md`。子目录 `AGENTS.md` 文件通过工具调用懒发现（通过 `subdirectory_hints.py`）并注入到工具结果中——它们不会预先加载到系统提示中。

:::tip
保持上下文文件专注和简洁。每个字符都计入你的令牌预算，因为它们被注入到每条消息中。
:::

## 记忆与技能

### 记忆 vs. 技能：什么放在哪里

**记忆**用于事实：你的环境、偏好、项目位置以及代理了解你的事情。**技能**用于程序：多步骤工作流、特定工具指令和可复用配方。记忆用于"什么"，技能用于"如何"。

### 何时创建技能

如果你发现一个需要 5+ 步骤的任务并且你会再次做，让代理为它创建一个技能。说"将你刚才做的保存为名为 `deploy-staging` 的技能"。下次，只需输入 `/deploy-staging`，代理就会加载完整程序。

### 管理记忆容量

记忆有意限制在边界内（MEMORY.md 约 2,200 字符，USER.md 约 1,375 字符）。当它满时，代理会合并条目。你可以通过说"清理你的记忆"或"替换旧的 Python 3.9 笔记——我们现在用 3.12"来提供帮助。

### 让代理记住

在一个富有成效的会话后，说"为下次记住这个"，代理会保存关键要点。你也可以具体：`save to memory that our CI uses GitHub Actions with the `deploy.yml` workflow`。

:::warning
记忆是一个冻结的快照——在会话期间所做的更改在下一会话开始前不会出现在系统提示中。代理立即写入磁盘，但提示缓存在会话中不会被使失效。
:::

## 性能与成本

### 不要破坏提示缓存

大多数 LLM 提供商缓存系统提示前缀。如果你保持系统提示稳定（相同的上下文文件、相同的记忆），会话中的后续消息会获得**缓存命中**，这要便宜得多。避免在会话中途更改模型或系统提示。

### 在达到限制前使用 /compress

长会话累积令牌。当你注意到响应变慢或被截断时，运行 `/compress`。这会总结对话历史，保留关键上下文同时显著减少令牌数量。使用 `/usage` 检查你的状态。

### 委托并行工作

需要同时研究三个主题？让代理使用 `delegate_task` 进行并行子任务。每个子代理独立运行，有自己的上下文，只有最终摘要返回——大幅减少你主对话的令牌使用。

### 对批量操作使用 execute_code

不要一次运行一个终端命令，让代理写一个一次完成所有事情的脚本。"写一个 Python 脚本将所有 `.jpeg` 文件重命名为 `.jpg` 并运行它"比单独重命名文件更便宜更快。

### 选择正确的模型

使用 `/model` 在会话中切换模型。对复杂推理和架构决策使用前沿模型（Claude Sonnet/Opus、GPT-4o）。对格式化、重命名或样板生成等简单任务切换到更快的模型。

:::tip
定期运行 `/usage` 查看你的令牌消耗。运行 `/insights` 获取过去 30 天使用模式的更广泛视图。
:::

## 消息提示

### 设置主频道

在你首选的 Telegram 或 Discord 聊天中使用 `/sethome` 将其指定为主频道。Cron 作业结果和计划任务输出在这里交付。没有它，代理没有地方发送主动消息。

### 使用 /title 组织会话

用 `/title auth-refactor` 或 `/title research-llm-quantization` 为你的会话命名。命名会话通过 `hermes sessions list` 容易找到，通过 `hermes -r "auth-refactor"` 恢复。未命名会话堆积如山，变得无法区分。

### 团队访问的 DM 配对

不要手动收集用户 ID 用于允许列表，启用 DM 配对。当队友 DM 机器人时，他们获得一个一次性配对码。你用 `hermes pairing approve telegram XKGH5N7P` 批准它——简单安全。

### 工具进度显示模式

使用 `/verbose` 控制你看到多少工具活动。在消息平台上，通常越少越好——保持"new"以仅查看新工具调用。在 CLI 中，"all" 给你一个令人满意的实时视图，观察代理做的一切。

:::tip
在消息平台上，会话在空闲时间后自动重置（默认：24 小时）或每天凌晨 4 点。如果你需要更长的会话，可以在 `~/.hermes/config.yaml` 中调整每个平台。
:::

## 安全

### 对不受信任的代码使用 Docker

当处理不受信任的仓库或运行不熟悉的代码时，使用 Docker 或 Daytona 作为你的终端后端。在你的 `.env` 中设置 `TERMINAL_BACKEND=docker`。容器内的破坏性命令无法伤害你的主机系统。

```bash
# 在你的 .env 中：
TERMINAL_BACKEND=docker
TERMINAL_DOCKER_IMAGE=hermes-sandbox:latest
```

### 避免 Windows 编码陷阱

在 Windows 上，某些默认编码（如 `cp125x`）无法表示所有 Unicode 字符，这可能在测试或脚本中写入文件时导致 `UnicodeEncodeError`。

- 首选使用显式 UTF-8 编码打开文件：

```python
with open("results.txt", "w", encoding="utf-8") as f:
    f.write("✓ All good\n")
```

- 在 PowerShell 中，你也可以将当前会话切换到 UTF-8 以处理控制台和本机命令输出：

```powershell
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
```

这使 PowerShell 和子进程保持 UTF-8 并帮助避免仅限 Windows 的失败。

### 选择"始终"前要审查

当代理触发危险命令批准（`rm -rf`、`DROP TABLE` 等）时，你得到四个选项：**一次**、**会话**、**始终**、**拒绝**。在选择"始终"前仔细考虑——它会永久允许列表该模式。从"会话"开始，直到你感到舒适。

### 命令批准是你的安全网

Hermes 在执行前检查每个命令是否匹配危险模式列表。这包括递归删除、SQL drops、curl 管道到 shell 等。不要在生产中禁用它——它存在有充分的理由。

:::warning
在容器后端（Docker、Singularity、Modal、Daytona）中运行，危险命令检查被**跳过**，因为容器是安全边界。确保你的容器镜像被正确锁定。
:::

### 对消息机器人使用允许列表

永远不要在具有终端访问权限的机器人上设置 `GATEWAY_ALLOW_ALL_USERS=true`。始终使用平台特定的允许列表（`TELEGRAM_ALLOWED_USERS`、`DISCORD_ALLOWED_USERS`）或 DM 配对来控制谁可以与你的代理交互。

```bash
# 推荐：每个平台显式允许列表
TELEGRAM_ALLOWED_USERS=123456789,987654321
DISCORD_ALLOWED_USERS=123456789012345678

# 或使用跨平台允许列表
GATEWAY_ALLOWED_USERS=123456789,987654321
```

---

*有应该在本页上的技巧？打开 issue 或 PR——欢迎社区贡献。*
