---
sidebar_position: 4
title: "教程：团队 Telegram 助手"
description: "分步指南：设置可供整个团队使用的 Telegram 机器人，用于代码帮助、研究、系统管理等"
---

# 设置团队 Telegram 助手

本教程逐步介绍设置由 Hermes Agent 驱动的 Telegram 机器人，多个团队成员可以使用。完成后，你的团队将拥有共享的 AI 助手，他们可以发消息寻求代码、研究、系统管理等方面的帮助 — 通过用户级授权保护安全。

## 我们要构建什么

一个 Telegram 机器人，可以：

- **任何授权的团队成员** 都可以 DM 寻求帮助 — 代码审查、研究、shell 命令、调试
- **在你的服务器上运行** 拥有完整工具访问 — 终端、文件编辑、网络搜索、代码执行
- **每个用户独立会话** — 每个人获得自己的对话上下文
- **默认安全** — 只有批准的用户可以交互，有两种授权方法
- **计划任务** — 每日站会、健康检查和提醒发送到团队频道

---

## 前置条件

开始前，确保你拥有：

- **在服务器或 VPS 上安装 Hermes Agent**（不是你的笔记本 — 机器人需要持续运行）。如果尚未安装，请按照[安装指南](/docs/getting-started/installation)操作。
- **你自己的 Telegram 账户**（机器人所有者）
- **配置了 LLM 提供商** — 至少在 `~/.hermes/.env` 中设置 OpenAI、Anthropic 或其他支持提供商的 API key

:::tip
$5/月的 VPS 足够运行 gateway。Hermes 本身很轻量 — LLM API 调用才是花钱的地方，那些远程发生。
:::

---

## 步骤 1：创建 Telegram 机器人

每个 Telegram 机器人从 **@BotFather** 开始 — Telegram 官方创建机器人的机器人。

1. **打开 Telegram** 搜索 `@BotFather`，或访问 [t.me/BotFather](https://t.me/BotFather)

2. **发送 `/newbot`** — BotFather 会问你两件事：
   - **显示名称** — 用户看到的名称（例如，`Team Hermes Assistant`）
   - **用户名** — 必须以 `bot` 结尾（例如，`myteam_hermes_bot`）

3. **复制机器人 token** — BotFather 回复类似：
   ```
   Use this token to access the HTTP API:
   7123456789:AAH1bGciOiJSUzI1NiIsInR5cCI6Ikp...
   ```
   保存这个 token — 下一步需要。

4. **设置描述**（可选但推荐）：
   ```
   /setdescription
   ```
   选择你的机器人，然后输入类似：
   ```
   Team AI assistant powered by Hermes Agent. DM me for help with code, research, debugging, and more.
   ```

5. **设置机器人命令**（可选 — 给用户命令菜单）：
   ```
   /setcommands
   ```
   选择你的机器人，然后粘贴：
   ```
   new - Start a fresh conversation
   model - Show or change the AI model
   status - Show session info
   help - Show available commands
   stop - Stop the current task
   ```

:::warning
保持机器人 token 机密。拥有 token 的任何人都可以控制机器人。如果泄露，用 BotFather 中的 `/revoke` 生成新的。
:::

---

## 步骤 2：配置 Gateway

你有两个选项：交互式设置向导（推荐）或手动配置。

### 选项 A：交互式设置（推荐）

```bash
hermes gateway setup
```

这会通过方向键选择引导你完成所有步骤。选择 **Telegram**，粘贴机器人 token，提示时输入你的用户 ID。

### 选项 B：手动配置

在 `~/.hermes/.env` 中添加这些行：

```bash
# 来自 BotFather 的 Telegram 机器人 token
TELEGRAM_BOT_TOKEN=7123456789:AAH1bGciOiJSUzI1NiIsInR5cCI6Ikp...

# 你的 Telegram 用户 ID（数字）
TELEGRAM_ALLOWED_USERS=123456789
```

### 查找你的用户 ID

你的 Telegram 用户 ID 是一个数字值（不是你的用户名）。查找方法：

1. 在 Telegram 上向 [@userinfobot](https://t.me/userinfobot) 发消息
2. 它会立即回复你的数字用户 ID
3. 将该数字复制到 `TELEGRAM_ALLOWED_USERS`

:::info
Telegram 用户 ID 是像 `123456789` 这样的永久数字。它们与你可以更改的 `@username` 不同。始终使用数字 ID 进行白名单。
:::

---

## 步骤 3：启动 Gateway

### 快速测试

首先在前台运行 gateway 确保一切正常：

```bash
hermes gateway
```

你应该看到类似输出：

```
[Gateway] Starting Hermes Gateway...
[Gateway] Telegram adapter connected
[Gateway] Cron scheduler started (tick every 60s)
```

打开 Telegram，找到你的机器人，发一条消息。如果它回复，你成功了。按 `Ctrl+C` 停止。

### 生产环境：安装为服务

要获得重启后也能持续运行持久部署：

```bash
hermes gateway install
sudo hermes gateway install --system   # 仅 Linux：启动时系统服务
```

这会创建后台服务：Linux 上默认用户级 **systemd** 服务，macOS 上 **launchd** 服务，或如果你传入 `--system` 则为启动时 Linux 系统服务。

```bash
# Linux — 管理默认用户服务
hermes gateway start
hermes gateway stop
hermes gateway status

# 查看实时日志
journalctl --user -u hermes-gateway -f

# SSH 退出后保持运行
sudo loginctl enable-linger $USER

# Linux 服务器 — 显式系统服务命令
sudo hermes gateway start --system
sudo hermes gateway status --system
journalctl -u hermes-gateway -f
```

```bash
# macOS — 管理服务
hermes gateway start
hermes gateway stop
tail -f ~/.hermes/logs/gateway.log
```

:::tip macOS PATH
launchd plist 在安装时捕获你的 shell PATH，这样 gateway 子进程可以找到 Node.js 和 ffmpeg 等工具。如果之后安装新工具，重新运行 `hermes gateway install` 更新 plist。
:::

### 验证运行

```bash
hermes gateway status
```

然后在 Telegram 上向你的机器人发送测试消息。几秒内应该收到回复。

---

## 步骤 4：设置团队访问

现在给你的队友访问权限。有两种方法。

### 方法 A：静态白名单

收集每个团队成员的 Telegram 用户 ID（让他们向 [@userinfobot](https://t.me/userinfobot) 发消息）并用逗号分隔添加：

```bash
# 在 ~/.hermes/.env 中
TELEGRAM_ALLOWED_USERS=123456789,987654321,555555555
```

更改后重启 gateway：

```bash
hermes gateway stop && hermes gateway start
```

### 方法 B：DM 配对（团队推荐）

DM 配对更灵活 — 不需要预先收集用户 ID。运作方式：

1. **队友 DM 机器人** — 由于他们不在白名单，机器人回复一次性配对码：
   ```
   🔐 配对码: XKGH5N7P
   将此代码发送给机器人所有者批准。
   ```

2. **队友通过任何渠道向你发送代码**（Slack、邮件、当面）

3. **你在服务器上批准**：
   ```bash
   hermes pairing approve telegram XKGH5N7P
   ```

4. **他们就进来了** — 机器人立即开始响应他们的消息

**管理配对用户：**

```bash
# 查看所有待处理和已批准用户
hermes pairing list

# 撤销某人的访问权限
hermes pairing revoke telegram 987654321

# 清除过期的待处理代码
hermes pairing clear-pending
```

:::tip
DM 配对是团队理想选择，因为添加新用户时无需重启 gateway。批准立即生效。
:::

### 安全注意事项

- **永远不要在有终端访问的机器人上设置 `GATEWAY_ALLOW_ALL_USERS=true`** — 任何找到你机器人的人都可以在你的服务器上运行命令
- 配对码在 **1 小时后过期**，使用加密随机性
- 速率限制防止暴力攻击：每个用户每 10 分钟 1 次请求，每个平台最多 3 个待处理代码
- 5 次失败批准尝试后，平台进入 1 小时锁定
- 所有配对数据以 `chmod 0600` 权限存储

---

## 步骤 5：配置机器人

### 设置主频道

**主频道**是机器人发送 cron 作业结果和主动消息的地方。没有它，计划任务无处发送输出。

**选项 1：** 在机器人是成员的任意 Telegram 群组或聊天中使用 `/sethome` 命令。

**选项 2：** 在 `~/.hermes/.env` 中手动设置：

```bash
TELEGRAM_HOME_CHANNEL=-1001234567890
TELEGRAM_HOME_CHANNEL_NAME="Team Updates"
```

要查找频道 ID，将 [@userinfobot](https://t.me/userinfobot) 添加到群组 — 它会报告群组的聊天 ID。

### 配置工具进度显示

控制机器人使用工具时显示多少细节。在 `~/.hermes/config.yaml` 中：

```yaml
display:
  tool_progress: new    # off | new | all | verbose
```

| 模式 | 看到的内容 |
|------|-------------|
| `off` | 仅干净响应 — 无工具活动 |
| `new` | 每个新工具调用的简要状态（推荐用于消息） |
| `all` | 每个工具调用及详情 |
| `verbose` | 包括命令结果的完整工具输出 |

用户也可以在聊天中使用 `/verbose` 命令更改每个会话的设置。

### 用 SOUL.md 设置人格

通过编辑 `~/.hermes/SOUL.md` 自定义机器人沟通方式：

完整指南见 [在 Hermes 中使用 SOUL.md](/docs/guides/use-soul-with-hermes)。

```markdown
# Soul
你是一位有用的团队助手。简洁技术。
任何代码使用代码块。跳过客套 — 团队重视直接。
调试时，始终先要错误日志再猜测解决方案。
```

### 添加项目上下文

如果你的团队处理特定项目，创建上下文文件让机器人了解你的技术栈：

```markdown
<!-- ~/.hermes/AGENTS.md -->
# 团队上下文
- 我们使用 Python 3.12 + FastAPI + SQLAlchemy
- 前端是 React + TypeScript
- CI/CD 运行在 GitHub Actions
- 生产部署到 AWS ECS
- 始终建议为新代码编写测试
```

:::info
上下文文件被注入每个会话的系统 prompt。保持简洁 — 每个字符都计入 token 预算。
:::

---

## 步骤 6：设置计划任务

Gateway 运行后，你可以安排定期任务，将结果发送到团队频道。

### 每日站会摘要

在 Telegram 上向机器人发消息：

```
每个工作日早上 9 点，检查 GitHub 仓库
github.com/myorg/myproject 的：
1. 过去 24 小时打开/合并的 Pull Requests
2. 创建或关闭的 Issues
3. 主分支上的任何 CI/CD 失败
格式化为简短的站会式摘要。
```

智能体自动创建 cron 作业并将结果发送到你提问的聊天（或主频道）。

### 服务器健康检查

```
每 6 小时，用 'df -h' 检查磁盘使用，'free -h' 检查内存，
'docker ps' 检查 Docker 容器状态。报告任何异常 —
超过 80% 的分区、重启过的容器或高内存使用。
```

### 管理计划任务

```bash
# 从 CLI
hermes cron list          # 查看所有计划作业
hermes cron status        # 检查调度器是否运行

# 从 Telegram 聊天
/cron list                # 查看作业
/cron remove <job_id>     # 删除作业
```

:::warning
Cron 作业提示在完全新鲜的会话中运行，没有先前对话的记忆。确保每个提示包含智能体需要的所有上下文 — 文件路径、URL、服务器地址和清晰指令。
:::

---

## 生产技巧

### 使用 Docker 确保安全

在共享团队机器人上，使用 Docker 作为终端后端，这样智能体命令在容器中运行而非主机上：

```bash
# 在 ~/.hermes/.env 中
TERMINAL_BACKEND=docker
TERMINAL_DOCKER_IMAGE=nikolaik/python-nodejs:python3.11-nodejs20
```

或在 `~/.hermes/config.yaml` 中：

```yaml
terminal:
  backend: docker
  container_cpu: 1
  container_memory: 5120
  container_persistent: true
```

这样，即使有人让机器人运行破坏性命令，你的主机系统也受保护。

### 监控 Gateway

```bash
# 检查 gateway 是否运行
hermes gateway status

# 查看实时日志（Linux）
journalctl --user -u hermes-gateway -f

# 查看实时日志（macOS）
tail -f ~/.hermes/logs/gateway.log
```

### 保持 Hermes 更新

从 Telegram，向机器人发送 `/update` — 它会拉取最新版本并重启。或从服务器：

```bash
hermes update
hermes gateway stop && hermes gateway start
```

### 日志位置

| 内容 | 位置 |
|------|----------|
| Gateway 日志 | `journalctl --user -u hermes-gateway`（Linux）或 `~/.hermes/logs/gateway.log`（macOS） |
| Cron 作业输出 | `~/.hermes/cron/output/{job_id}/{timestamp}.md` |
| Cron 作业定义 | `~/.hermes/cron/jobs.json` |
| 配对数据 | `~/.hermes/pairing/` |
| 会话历史 | `~/.hermes/sessions/` |

---

## 深入方向

你已拥有可用的团队 Telegram 助手。以下是下一步：

- **[安全指南](/docs/user-guide/security)** — 深入了解授权、容器隔离和命令审批
- **[消息 Gateway](/docs/user-guide/messaging)** — gateway 架构、会话管理和聊天命令完整参考
- **[Telegram 设置](/docs/user-guide/messaging/telegram)** — 包括语音消息和 TTS 的平台特定详情
- **[计划任务](/docs/user-guide/features/cron)** — 使用投递选项和 cron 表达式的高级 cron 调度
- **[上下文文件](/docs/user-guide/features/context-files)** — 用于项目知识的 AGENTS.md、SOUL.md 和 .cursorrules
- **[人格](/docs/user-guide/features/personality)** — 内置人格预设和自定义人物设定
- **添加更多平台** — 相同 gateway 可以同时运行 [Discord](/docs/user-guide/messaging/discord)、[Slack](/docs/user-guide/messaging/slack) 和 [WhatsApp](/docs/user-guide/messaging/whatsapp)

---

*有问题或问题？在 GitHub 上开 issue — 欢迎贡献。*
