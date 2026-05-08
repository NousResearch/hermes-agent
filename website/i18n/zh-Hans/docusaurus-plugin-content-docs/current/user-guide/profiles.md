---
sidebar_position: 2
---

# Profiles: 运行多个 Agent

在同一台机器上运行多个独立的 Hermes agent——每个都有自己的配置、API 密钥、内存、会话、skills 和 gateway 状态。

## 什么是 profiles？

profile 是一个独立的 Hermes 主目录。每个 profile 都有自己的目录，包含自己的 `config.yaml`、`.env`、`SOUL.md`、内存、会话、skills、cron 作业和状态数据库。Profile 让您可以为不同目的运行独立的 agent——编码助手、个人 bot、研究 agent——而不会混淆 Hermes 状态。

当您创建 profile 时，它会自动成为自己的命令。创建一个名为 `coder` 的 profile，您立即拥有 `coder chat`、`coder setup`、`coder gateway start` 等。

## 快速开始

```bash
hermes profile create coder       # 创建 profile + "coder" 命令别名
coder setup                       # 配置 API 密钥和模型
coder chat                        # 开始聊天
```

就这样。`coder` 现在是一个独立的 Hermes profile，有自己的配置、内存和状态。

## 创建 profile

### 空白 profile

```bash
hermes profile create mybot
```

创建带有捆绑 skills 种子化的新鲜 profile。运行 `mybot setup` 配置 API 密钥、模型和 gateway 令牌。

### 仅克隆配置（`--clone`）

```bash
hermes profile create work --clone
```

将当前 profile 的 `config.yaml`、`.env` 和 `SOUL.md` 复制到新 profile。相同的 API 密钥和模型，但全新的会话和内存。编辑 `~/.hermes/profiles/work/.env` 使用不同的 API 密钥，或编辑 `~/.hermes/profiles/work/SOUL.md` 使用不同的人格。

### 克隆所有内容（`--clone-all`）

```bash
hermes profile create backup --clone-all
```

复制**所有内容**——配置、API 密钥、人格、所有内存、完整会话历史、skills、cron 作业、插件。完整快照。对于备份或 fork 已经具有上下文的 agent 很有用。

### 从特定 profile 克隆

```bash
hermes profile create work --clone --clone-from coder
```

:::tip Honcho 内存 + profiles
当 Honcho 启用时，`--clone` 会自动为新 profile 创建专用 AI 对等体，同时共享相同的用户工作区。每个 profile 构建自己的观察和身份。请参阅 [Honcho — 多 agent / Profiles](./features/memory-providers.md#honcho) 获取详情。
:::

## 使用 profiles

### 命令别名

每个 profile 自动在 `~/.local/bin/<name>` 获取命令别名：

```bash
coder chat                    # 与 coder agent 聊天
coder setup                   # 配置 coder 的设置
coder gateway start           # 启动 coder 的 gateway
coder doctor                  # 检查 coder 的健康状况
coder skills list             # 列出 coder 的 skills
coder config set model.default anthropic/claude-sonnet-4
```

该别名适用于每个 hermes 子命令——它只是 `hermes -p <name>` 的底层实现。

### `-p` 标志

您也可以用任何命令显式定位 profile：

```bash
hermes -p coder chat
hermes --profile=coder doctor
hermes chat -p coder -q "hello"    # 可以在任何位置工作
```

### 粘性默认值（`hermes profile use`）

```bash
hermes profile use coder
hermes chat                   # 现在指向 coder
hermes tools                  # 配置 coder 的工具
hermes profile use default    # 切换回来
```

设置默认值，使普通 `hermes` 命令针对该 profile。像 `kubectl config use-context`。

### 了解您的位置

CLI 始终显示哪个 profile 处于活动状态：

- **提示符**：`coder ❯` 而不是 `❯`
- **横幅**：启动时显示 `Profile: coder`
- **`hermes profile`**：显示当前 profile 名称、路径、模型、gateway 状态

## Profiles vs 工作区 vs 沙箱

Profiles 通常与工作区或沙箱混淆，但它们是不同的：

- **Profile** 为 Hermes 提供自己的状态目录：`config.yaml`、`.env`、`SOUL.md`、会话、内存、日志、cron 作业和 gateway 状态。
- **工作区**或**工作目录**是终端命令启动的地方。这由 `terminal.cwd` 单独控制。
- **沙箱**限制文件系统访问。Profile **不会**沙箱化 agent。

在默认的 `local` 终端后端上，agent 仍然具有与您的用户账户相同的文件系统访问权限。Profile 不会阻止它访问 profile 目录之外的文件夹。

如果您希望 profile 默认在特定项目文件夹中工作，请在那个 profile 的 `config.yaml` 中设置显式绝对 `terminal.cwd`：

```yaml
terminal:
  backend: local
  cwd: /absolute/path/to/project
```

在本地后端上使用 `cwd: "."` 意味着"Hermes 启动时的目录"，而不是"profile 目录"。

另请注意：

- `SOUL.md` 可以指导模型，但它不会强制执行工作区边界。
- 对 `SOUL.md` 的更改会在新会话中干净地生效。现有会话可能仍在使用旧的提示状态。
- 询问模型"您在哪个目录？"不是可靠的隔离测试。如果您需要可预测的工具起始目录，请显式设置 `terminal.cwd`。

## 运行 gateway

每个 profile 作为具有自己 bot 令牌的独立进程运行自己的 gateway：

```bash
coder gateway start           # 启动 coder 的 gateway
assistant gateway start       # 启动 assistant 的 gateway（独立进程）
```

### 不同的 bot 令牌

每个 profile 有自己的 `.env` 文件。在每个中配置不同的 Telegram/Discord/Slack bot 令牌：

```bash
# 编辑 coder 的令牌
nano ~/.hermes/profiles/coder/.env

# 编辑 assistant 的令牌
nano ~/.hermes/profiles/assistant/.env
```

### 安全：令牌锁定

如果两个 profile 意外使用相同的 bot 令牌，第二个 gateway 将被阻止，并显示命名冲突 profile 的明确错误。支持 Telegram、Discord、Slack、WhatsApp 和 Signal。

### 持久服务

```bash
coder gateway install         # 创建 hermes-gateway-coder systemd/launchd 服务
assistant gateway install     # 创建 hermes-gateway-assistant 服务
```

每个 profile 有自己的服务名称。它们独立运行。

## 配置 profiles

每个 profile 有自己的：

- **`config.yaml`** — 模型、provider、工具集、所有设置
- **`.env`** — API 密钥、bot 令牌
- **`SOUL.md`** — 人格和指示

```bash
coder config set model.default anthropic/claude-sonnet-4
echo "You are a focused coding assistant." > ~/.hermes/profiles/coder/SOUL.md
```

如果您希望此 profile 默认在特定项目中工作，也请设置它自己的 `terminal.cwd`：

```bash
coder config set terminal.cwd /absolute/path/to/project
```

## 更新

`hermes update` 拉取代码一次（共享）并自动将新的捆绑 skills 同步到**所有** profiles：

```bash
hermes update
# → 代码已更新（12 commits）
# → Skills 已同步：default（已是最新）、coder（+2 新）、assistant（+2 新）
```

用户修改的 skills 永远不会被覆盖。

## 管理 profiles

```bash
hermes profile list           # 显示所有 profiles 及其状态
hermes profile show coder     # 显示一个 profile 的详细信息
hermes profile rename coder dev-bot   # 重命名（更新别名 + 服务）
hermes profile export coder   # 导出到 coder.tar.gz
hermes profile import coder.tar.gz   # 从归档导入
```

## 删除 profile

```bash
hermes profile delete coder
```

这会停止 gateway，删除 systemd/launchd 服务，删除命令别名，并删除所有 profile 数据。系统会要求您输入 profile 名称以确认。

使用 `--yes` 跳过确认：`hermes profile delete coder --yes`

:::note
您无法删除默认 profile（`~/.hermes`）。要删除所有内容，请使用 `hermes uninstall`。
:::

## Tab 补全

```bash
# Bash
eval "$(hermes completion bash)"

# Zsh
eval "$(hermes completion zsh)"
```

将该行添加到您的 `~/.bashrc` 或 `~/.zshrc` 以实现持久补全。补全 `-p` 后的 profile 名称、profile 子命令和顶级命令。

## 工作原理

Profiles 使用 `HERMES_HOME` 环境变量。当您运行 `coder chat` 时，包装脚本在启动 hermes 之前设置 `HERMES_HOME=~/.hermes/profiles/coder`。由于代码库中 119+ 个文件通过 `get_hermes_home()` 解析路径，Hermes 状态会自动将作用域限定到 profile 的目录——配置、会话、内存、skills、状态数据库、gateway PID、日志和 cron 作业。

这与终端工作目录分开。工具执行从 `terminal.cwd`（或在本地后端上 `cwd: "."` 时的启动目录）开始，而不是自动从 `HERMES_HOME` 开始。

默认 profile 就是 `~/.hermes` 本身。无需迁移——现有安装的工作方式相同。
