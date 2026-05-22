---
sidebar_position: 1
title: "技巧与最佳实践"
description: "使用 Hermes Agent 的实用建议 - 提示词技巧、CLI 快捷方式、上下文文件、记忆、成本优化和安全性"
---

# 技巧与最佳实践

这是一份“快速见效”的实用技巧集合，能立刻提升你使用 Hermes Agent 的效率。每一节都聚焦不同方面 - 你可以扫标题，直接跳到相关内容。

---

## 如何获得更好的结果

### 明确说出你想要什么

含糊的提示会得到含糊的结果。不要只说“修一下代码”，而要说“修复 `api/handlers.py` 第 47 行的 `TypeError` —— `process_request()` 从 `parse_body()` 收到了 `None`。”你给的上下文越多，所需迭代就越少。

### 一开始就给足上下文

把相关信息前置：文件路径、错误信息、预期行为。一个写得好的消息，胜过三轮澄清。可以直接粘贴错误堆栈 - 智能体能解析它。

### 把重复说明放进上下文文件

如果你发现自己反复说同样的要求（“用 tab 不用空格”“我们用 pytest”“API 在 `/api/v2`”），把它们放进 `AGENTS.md`。智能体会在每次会话自动读取它 - 配好后几乎零成本。

### 让智能体自己用工具

不要试图手把手带它做每一步。说“找出并修复失败的测试”就够了，不必说“打开 `tests/test_foo.py`，看第 42 行，然后……”智能体有文件搜索、终端访问和代码执行能力 - 让它自己探索和迭代。

### 复杂工作流优先用技能

在写很长的提示词解释如何做某件事之前，先看看是否已经有对应技能。输入 `/skills` 可以浏览可用技能，或者直接调用一个，比如 `/axolotl` 或 `/github-pr-workflow`。

## CLI 高手技巧

### 多行输入

按 **Alt+Enter**、**Ctrl+J** 或 **Shift+Enter** 可在不发送消息的情况下插入新行。`Shift+Enter` 只有在终端把它作为独立按键发送时才可用（Kitty / foot / WezTerm / Ghostty 默认支持；iTerm2 / Alacritty / VS Code terminal 启用 Kitty keyboard protocol 后也支持）。另外两个在所有终端都可用。

### 粘贴检测

CLI 会自动识别多行粘贴。直接粘贴代码块或错误堆栈即可 - 它不会把每一行当成单独消息发送。粘贴内容会被缓冲并作为一条消息发送。

### 中断并重定向

在智能体回复过程中按 **Ctrl+C** 一次，可以中断它。然后你就能输入新消息，把它重定向到新的目标。如果在 2 秒内连按两次 **Ctrl+C**，则强制退出。这在智能体开始跑偏时特别有用。

### 用 `-c` 恢复会话

忘了上一次会话里的内容？运行 `hermes -c` 可恢复到你离开的地方，并完整保留对话历史。你也可以按标题恢复：`hermes -r "my research project"`。

### 剪贴板图片粘贴

按 **Ctrl+V** 可以直接把剪贴板中的图片粘到聊天里。智能体会用视觉能力分析截图、图表、错误弹窗或 UI 草图 - 不需要先保存成文件。

### 斜杠命令自动补全

输入 `/` 然后按 **Tab**，就能看到所有可用命令。这包括内置命令（`/compress`、`/model`、`/title`）以及每个已安装技能。你不需要背命令 - Tab 补全会帮你。

:::tip
使用 `/verbose` 可以在工具输出显示模式之间切换：**off → new → all → verbose**。`all` 模式适合观察智能体具体做了什么；`off` 则最适合纯问答场景。
:::

## 上下文文件

### AGENTS.md：你的项目大脑

在项目根目录创建一个 `AGENTS.md`，写入架构决策、编码规范和项目专属说明。它会在每次会话中自动注入，因此智能体始终知道你项目的规则。

```markdown
# 项目上下文
- 这是一个使用 SQLAlchemy ORM 的 FastAPI 后端
- 数据库操作一律使用 async/await
- 测试放在 tests/ 中，使用 pytest-asyncio
- 不要提交 .env 文件
```

### SOUL.md：自定义个性

想让 Hermes 拥有稳定的默认语气？编辑 `~/.hermes/SOUL.md`（如果你使用自定义 Hermes home，则是 `$HERMES_HOME/SOUL.md`）。Hermes 现在会自动生成一个入门版 SOUL，并把这个全局文件作为整个实例的个性来源。

完整流程请参见 [在 Hermes 中使用 SOUL.md](/guides/use-soul-with-hermes)。

```markdown
# Soul
你是一名资深后端工程师。简洁直接。
除非被要求，否则不要展开解释。优先使用一行答案而不是冗长说明。
始终考虑错误处理和边界情况。
```

SOUL.md 用来保存长期稳定的个性。AGENTS.md 用来保存项目专属说明。

### .cursorrules 兼容性

已经有 `.cursorrules` 或 `.cursor/rules/*.mdc` 文件？Hermes 也会读取它们。你不用重复维护编码规范 - 它们会从当前工作目录自动加载。

### 发现机制

Hermes 会在会话开始时加载当前工作目录下的顶层 `AGENTS.md`。子目录中的 `AGENTS.md` 会在工具调用期间延迟发现（通过 `subdirectory_hints.py`），并注入到工具结果中 - 它们不会在启动时预先加载到系统提示词里。

:::tip
上下文文件要保持聚焦、简洁。每个字符都会算进 token 预算，因为它们会注入到每一条消息里。
:::

## 记忆与技能

### 记忆和技能：什么该放哪

**Memory** 用来存事实：你的环境、偏好、项目位置，以及智能体从你这里学到的东西。**Skills** 用来存流程：多步骤工作流、工具特定说明和可复用做法。记忆回答“是什么”，技能回答“怎么做”。

### 什么时候创建技能

如果你发现某个任务需要 5 步以上，而且以后还会再做，直接让智能体把它保存成一个技能。可以说“把你刚才做的事情保存成一个叫 `deploy-staging` 的技能。”下次你只要输入 `/deploy-staging`，智能体就会加载完整流程。

### 管理记忆容量

记忆空间是有上限的（`MEMORY.md` 约 2,200 字符，`USER.md` 约 1,375 字符）。装满后智能体会自动合并条目。你可以帮它一把，说“清理一下你的记忆”或者“把旧的 Python 3.9 记忆替换掉 - 我们现在用 3.12 了”。

### 让智能体记住

一次愉快的会话结束后，可以对智能体说“记住这次的要点”，它就会把关键结论写入记忆。你也可以说得更具体：“把我们 CI 使用 GitHub Actions 和 `deploy.yml` 工作流这件事保存到记忆里。”

:::warning
记忆是冻结快照 - 会话期间做出的更改不会在当前系统提示词里立刻生效，直到下一次会话开始。智能体会立即把内容写到磁盘，但提示词缓存不会在会话中途失效。
:::

## 性能与成本

### 不要破坏提示词缓存

大多数 LLM 提供商都会缓存系统提示词前缀。如果你保持系统提示词稳定（上下文文件、记忆都不变），同一会话后续消息就能命中缓存，成本会明显更低。避免在会话中途更换模型或系统提示词。

### 接近上限前先用 /compress

长会话会不断累积 token。当你发现回复变慢或被截断时，运行 `/compress`。它会总结对话历史，保留关键上下文，同时大幅减少 token 数。用 `/usage` 查看当前占用情况。

### 用 delegate 并行处理

需要同时研究三个主题？让智能体使用 `delegate_task` 开并行子任务。每个子智能体都有自己的上下文独立运行，只有最终摘要会回来 - 可以大幅减少主会话 token 的消耗。

### 批量操作优先用 execute_code

不要一次只跑一条终端命令，直接让智能体写一个脚本把事情一次做完。“写一个 Python 脚本把所有 `.jpeg` 重命名成 `.jpg` 并运行它”比一张一张改更便宜也更快。

### 选对模型

用 `/model` 在会话中切换模型。复杂推理和架构决策用前沿模型（Claude Sonnet/Opus、GPT-4o）；格式化、重命名或模板生成这类简单任务可以切到更快的模型。

:::tip
定期运行 `/usage` 查看 token 消耗。运行 `/insights` 可以查看过去 30 天的更广泛使用趋势。
:::

## 消息平台技巧

### 设置主频道

在你常用的 Telegram 或 Discord 聊天里使用 `/sethome`，把它设为主频道。Cron 作业结果和计划任务输出会送到这里。如果不设置，智能体就没有地方主动发消息。

### 用 /title 整理会话

用 `/title auth-refactor` 或 `/title research-llm-quantization` 给会话命名。命名会话可以用 `hermes sessions list` 很容易找到，也能用 `hermes -r "auth-refactor"` 恢复。未命名会话会越积越多，最后根本分不清。

### 团队访问用 DM 配对

不要手工收集用户 ID 再写白名单，直接启用 DM 配对。队友私信机器人后会拿到一次性配对码。你用 `hermes pairing approve telegram XKGH5N7P` 审批即可 - 简单又安全。

### 工具进度显示模式

用 `/verbose` 控制你看到多少工具活动。在消息平台里，少即是多 - 保持在 `new`，只看新的工具调用；在 CLI 里，`all` 会让你实时看到智能体做的所有事情。

:::tip
在消息平台上，闲置会话会自动重置（默认 24 小时）或者每天凌晨 4 点重置。如果你需要更长会话，可以在 `~/.hermes/config.yaml` 中按平台调整。
:::

## 安全性

### 用 Docker 跑不可信代码

处理不可信仓库或运行陌生代码时，请把 Docker 或 Daytona 作为终端后端。在 `.env` 中设置 `TERMINAL_BACKEND=docker`。容器里的破坏性命令不会影响宿主机系统。

```bash
# 在你的 .env 中：
TERMINAL_BACKEND=docker
TERMINAL_DOCKER_IMAGE=hermes-sandbox:latest
```

### 避免 Windows 编码坑

在 Windows 上，某些默认编码（例如 `cp125x`）无法表示所有 Unicode 字符，这可能导致在测试或脚本写文件时出现 `UnicodeEncodeError`。

- 优先显式使用 UTF-8 编码打开文件：

```python
with open("results.txt", "w", encoding="utf-8") as f:
    f.write("✓ All good\n")
```

- 在 PowerShell 中，你也可以把当前会话切换为 UTF-8，让控制台和原生命令输出都使用 UTF-8：

```powershell
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
```

这样能让 PowerShell 和子进程都保持 UTF-8，有助于避免 Windows 专属失败。

### 在选择“always”前先复核

当智能体触发危险命令审批（`rm -rf`、`DROP TABLE` 等）时，你会看到四个选项：**once**、**session**、**always**、**deny**。选择 `always` 前一定要想清楚 - 它会永久把这个模式加入白名单。刚开始建议先选 `session`，熟悉后再说。

### 命令审批是你的安全网

Hermes 会在执行前检查每条命令是否命中危险模式列表。这包括递归删除、SQL drop、把 curl 管道到 shell 等。生产环境不要关闭它 - 它存在是有充分理由的。

:::warning
当运行在容器后端（Docker、Singularity、Modal、Daytona）时，危险命令检查会被**跳过**，因为容器本身就是安全边界。请确保你的容器镜像已经正确加固。
:::

### 给消息机器人使用白名单

永远不要在带终端访问能力的机器人上设置 `GATEWAY_ALLOW_ALL_USERS=true`。请始终使用平台级白名单（`TELEGRAM_ALLOWED_USERS`、`DISCORD_ALLOWED_USERS`）或 DM 配对，来控制谁可以与你的智能体交互。

```bash
# 推荐：每个平台都显式设置白名单
TELEGRAM_ALLOWED_USERS=123456789,987654321
DISCORD_ALLOWED_USERS=123456789012345678

# 或者使用跨平台白名单
GATEWAY_ALLOWED_USERS=123456789,987654321
```

---

*有想补进这一页的技巧？欢迎开 issue 或 PR - 社区贡献随时欢迎。*