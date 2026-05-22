---
sidebar_position: 2
---

# Profiles：在同一台机器上运行多个 agent

在同一台机器上运行多个彼此独立的 Hermes agent - 每个 agent 都有自己的配置、API Key、记忆、会话、技能和网关状态。

## 什么是 profile？

profile 就是一个独立的 Hermes home 目录。每个 profile 都有自己的 `config.yaml`、`.env`、`SOUL.md`、记忆、会话、技能、cron 任务和状态数据库。这样你就可以为不同用途运行不同的 agent - 比如一个代码助手、一个个人机器人、一个研究 agent - 而不会把 Hermes 的状态混在一起。

当你创建 profile 后，它会自动变成自己的命令。比如创建一个叫 `coder` 的 profile，你马上就能使用 `coder chat`、`coder setup`、`coder gateway start` 等命令。

## 快速开始

```bash
hermes profile create coder       # 创建 profile + "coder" 命令别名
coder setup                       # 配置 API Key 和模型
coder chat                        # 开始聊天
```

就是这样。`coder` 现在就是一个拥有自己配置、记忆和状态的独立 Hermes profile。

## 创建 profile

### 空白 profile

```bash
hermes profile create mybot
```

创建一个全新的 profile，并预置 bundled skills。运行 `mybot setup` 来配置 API Key、模型和网关 token。

### 仅克隆配置（`--clone`）

```bash
hermes profile create work --clone
```

复制当前 profile 的 `config.yaml`、`.env` 和 `SOUL.md` 到新 profile。API Key 和模型保持一致，但会话和记忆是新的。你可以编辑 `~/.hermes/profiles/work/.env` 使用不同的 API Key，或者编辑 `~/.hermes/profiles/work/SOUL.md` 使用不同的人格设定。

### 克隆全部内容（`--clone-all`）

```bash
hermes profile create backup --clone-all
```

复制**全部内容** - 配置、API Key、人格、所有记忆、完整会话历史、技能、cron 任务、插件。相当于完整快照。适合备份，或者把已有上下文的 agent 派生出一个新分支。

### 从指定 profile 克隆

```bash
hermes profile create work --clone --clone-from coder
```

:::tip Honcho memory + profiles
启用 Honcho 时，`--clone` 会自动为新 profile 创建一个专属 AI peer，同时共享同一个用户 workspace。每个 profile 会形成自己的观察和身份。详情见 [Honcho - 多 agent / Profiles](./features/memory-providers.md#honcho)。
:::

## 使用 profile

### 命令别名

每个 profile 都会自动在 `~/.local/bin/<name>` 下生成一个命令别名：

```bash
coder chat                    # 和 coder agent 聊天
coder setup                   # 配置 coder 的设置
coder gateway start           # 启动 coder 的网关
coder doctor                  # 检查 coder 的健康状态
coder skills list             # 列出 coder 的技能
coder config set model.default anthropic/claude-sonnet-4
```

这个别名对 Hermes 的所有子命令都有效 - 底层其实就是 `hermes -p <name>`。

### `-p` 参数

你也可以用任何命令显式指定 profile：

```bash
hermes -p coder chat
hermes --profile=coder doctor
hermes chat -p coder -q "hello"    # 任何位置都可以用
```

### 粘性默认值（`hermes profile use`）

```bash
hermes profile use coder
hermes chat                   # 现在会默认指向 coder
hermes tools                  # 配置 coder 的工具
hermes profile use default    # 切回默认 profile
```

这会设置一个默认值，让普通的 `hermes` 命令都指向那个 profile。类似 `kubectl config use-context`。

### 你当前在哪个 profile

CLI 会始终显示当前活跃的是哪个 profile：

- **提示符**：`coder ❯` 而不是 `❯`
- **横幅**：启动时显示 `Profile: coder`
- **`hermes profile`**：显示当前 profile 名称、路径、模型、网关状态

## Profiles、workspace 和 sandbox 的区别

profile 常常会和 workspace 或 sandbox 混淆，但它们不是一回事：

- **profile** 给 Hermes 一套自己的状态目录：`config.yaml`、`.env`、`SOUL.md`、会话、记忆、日志、cron 任务和网关状态。
- **workspace** 或 **working directory** 决定终端命令从哪里开始运行。这个由 `terminal.cwd` 单独控制。
- **sandbox** 决定文件系统访问范围。profile 并不会给 agent 加 sandbox。

在默认的 `local` 终端后端下，agent 仍然拥有和你用户账号相同的文件系统访问权限。profile 不会阻止它访问 profile 目录之外的文件夹。

如果你想让 profile 从某个指定项目目录开始，可以在该 profile 的 `config.yaml` 中设置一个绝对路径的 `terminal.cwd`：

```yaml
terminal:
  backend: local
  cwd: /absolute/path/to/project
```

在 local 后端里，`cwd: "."` 表示“Hermes 启动时所在的目录”，而不是“profile 目录”。

另外还要注意：

- `SOUL.md` 可以指导模型，但不能强制 workspace 边界。
- 对 `SOUL.md` 的修改会在新会话里干净生效。已有会话可能仍在使用旧的 prompt 状态。
- 问模型“你现在在哪个目录？”并不是可靠的隔离测试。如果你需要一个可预测的起始目录给工具用，请显式设置 `terminal.cwd`。

## 运行网关

每个 profile 都会以自己的进程运行独立网关，并拥有各自的 bot token：

```bash
coder gateway start           # 启动 coder 的网关
assistant gateway start       # 启动 assistant 的网关（独立进程）
```

### 不同的 bot token

每个 profile 都有自己的 `.env` 文件。为每个 profile 配置不同的 Telegram / Discord / Slack bot token：

```bash
# 编辑 coder 的 token
nano ~/.hermes/profiles/coder/.env

# 编辑 assistant 的 token
nano ~/.hermes/profiles/assistant/.env
```

### 安全性：token lock

如果两个 profile 不小心用了同一个 bot token，第二个网关会被阻止启动，并给出明确错误，错误里会写出冲突的 profile。Telegram、Discord、Slack、WhatsApp 和 Signal 都支持这一点。

### 持久化服务

```bash
coder gateway install         # 创建 hermes-gateway-coder systemd/launchd 服务
assistant gateway install     # 创建 hermes-gateway-assistant 服务
```

每个 profile 都有自己的服务名，它们彼此独立运行。

## 配置 profile

每个 profile 都有自己的：

- **`config.yaml`** - 模型、提供商、工具集、所有设置
- **`.env`** - API Key、bot token
- **`SOUL.md`** - 人格和指令

```bash
coder config set model.default anthropic/claude-sonnet-4
echo "You are a focused coding assistant." > ~/.hermes/profiles/coder/SOUL.md
```

如果你希望这个 profile 默认在某个项目里工作，也可以设置它自己的 `terminal.cwd`：

```bash
coder config set terminal.cwd /absolute/path/to/project
```

## 更新

`hermes update` 会拉取一次共享代码，并把新的 bundled skills 自动同步到**所有** profile：

```bash
hermes update
# → Code updated (12 commits)
# → Skills synced: default (up to date), coder (+2 new), assistant (+2 new)
```

用户自己修改过的技能永远不会被覆盖。

## 管理 profile

```bash
hermes profile list           # 显示所有 profile 及状态
hermes profile show coder     # 查看单个 profile 的详细信息
hermes profile rename coder dev-bot   # 重命名（更新别名和服务）
hermes profile export coder   # 导出为 coder.tar.gz
hermes profile import coder.tar.gz   # 从归档导入
```

## 删除 profile

```bash
hermes profile delete coder
```

这会停止网关、删除 systemd / launchd 服务、移除命令别名，并删除全部 profile 数据。系统会要求你输入 profile 名称来确认。

用 `--yes` 可以跳过确认：`hermes profile delete coder --yes`

:::note
你不能删除默认 profile（`~/.hermes`）。如果你想清空所有内容，请使用 `hermes uninstall`。
:::---
sidebar_position: 2
---

# 配置文件：在同一台机器上运行多个 agent

在同一台机器上运行多个彼此独立的 Hermes agent - 每个 agent 都有自己的配置、API Key、记忆、会话、技能和网关状态。

## 什么是配置文件？

配置文件就是一套独立的 Hermes home 目录。每个配置文件都会拥有自己的 `config.yaml`、`.env`、`SOUL.md`、记忆、会话、技能、cron 任务和状态数据库。配置文件让你可以为不同用途运行不同的 agent - 编程助手、个人机器人、研究助手 - 而不会把 Hermes 的状态混在一起。

当你创建一个配置文件时，它会自动变成一个独立命令。比如创建一个名为 `coder` 的配置文件后，你立刻就能使用 `coder chat`、`coder setup`、`coder gateway start` 等命令。

## 快速开始

```bash
hermes profile create coder       # 创建配置文件 + "coder" 命令别名
coder setup                       # 配置 API Key 和模型
coder chat                        # 开始聊天
```

就这么简单。现在 `coder` 已经成为一个独立的 Hermes 配置文件，拥有自己的配置、记忆和状态。

## 创建配置文件

### 空白配置文件

```bash
hermes profile create mybot
```

这会创建一个全新的配置文件，并预置内置技能。运行 `mybot setup` 即可配置 API Key、模型和网关 Token。

### 仅克隆配置（`--clone`）

```bash
hermes profile create work --clone
```

会把当前配置文件的 `config.yaml`、`.env` 和 `SOUL.md` 复制到新配置文件中。API Key 和模型保持一致，但会话和记忆都是新的。你可以编辑 `~/.hermes/profiles/work/.env` 使用不同的 API Key，或编辑 `~/.hermes/profiles/work/SOUL.md` 使用不同的人格。

### 克隆全部内容（`--clone-all`）

```bash
hermes profile create backup --clone-all
```

会复制**全部内容** - 配置、API Key、人格、所有记忆、完整会话历史、技能、cron 任务、插件。相当于一个完整快照。适合备份，或者基于已有上下文分叉一个 agent。

### 从指定配置文件克隆

```bash
hermes profile create work --clone --clone-from coder
```

:::tip Honcho 记忆 + 配置文件
启用 Honcho 时，`--clone` 会为新配置文件自动创建一个专属 AI peer，同时共享同一个用户工作区。每个配置文件都会构建自己的观测与身份。详见 [Honcho - 多 agent / 配置文件](./features/memory-providers.md#honcho)。
:::

## 使用配置文件

### 命令别名

每个配置文件都会在 `~/.local/bin/<name>` 下自动生成一个命令别名：

```bash
coder chat                    # 和 coder agent 聊天
coder setup                   # 配置 coder 的设置
coder gateway start           # 启动 coder 的网关
coder doctor                  # 检查 coder 健康状态
coder skills list             # 列出 coder 的技能
coder config set model.default anthropic/claude-sonnet-4
```

这个别名适用于 Hermes 的所有子命令 - 本质上只是 `hermes -p <name>` 的包装。

### `-p` 参数

你也可以在任何命令中显式指定配置文件：

```bash
hermes -p coder chat
hermes --profile=coder doctor
hermes chat -p coder -q "hello"    # 放在任意位置都可以
```

### 粘性默认值（`hermes profile use`）

```bash
hermes profile use coder
hermes chat                   # 现在会目标到 coder
hermes tools                  # 配置 coder 的工具
hermes profile use default    # 切回默认配置文件
```

这会设置一个默认值，让普通 `hermes` 命令都指向该配置文件。类似 `kubectl config use-context`。

### 如何知道当前用的是谁

CLI 会一直显示当前激活的配置文件：

- **提示符**：显示为 `coder ❯`，而不是 `❯`
- **横幅**：启动时显示 `Profile: coder`
- **`hermes profile`**：显示当前配置文件名称、路径、模型和网关状态

## 配置文件 vs 工作区 vs 沙盒

配置文件经常会和工作区或沙盒混淆，但它们其实是不同的概念：

- **配置文件** 给 Hermes 一套自己的状态目录：`config.yaml`、`.env`、`SOUL.md`、会话、记忆、日志、cron 任务和网关状态。
- **工作区** 或 **工作目录** 是终端命令的起始目录。这由 `terminal.cwd` 单独控制。
- **沙盒** 是限制文件系统访问的机制。配置文件**不会**为 agent 提供沙盒隔离。

在默认的 `local` 终端后端上，agent 仍然拥有与你用户账户相同的文件系统访问权限。配置文件并不会阻止它访问 profile 目录之外的文件夹。

如果你希望配置文件默认从某个项目目录开始，可以在该配置文件的 `config.yaml` 中设置一个显式的绝对路径 `terminal.cwd`：

```yaml
terminal:
  backend: local
  cwd: /absolute/path/to/project
```

在 local 后端上使用 `cwd: "."` 的意思是“Hermes 启动时所在的目录”，而不是“配置文件目录”。

另外还要注意：

- `SOUL.md` 可以引导模型，但它不会强制工作区边界。
- 对 `SOUL.md` 的修改会在新会话中干净生效。已有会话可能仍在使用旧的提示词状态。
- 问模型“你现在在哪个目录？”并不是可靠的隔离测试。如果你需要工具从一个可预测的起始目录运行，请显式设置 `terminal.cwd`。

## 运行网关

每个配置文件都会以独立进程运行自己的网关，并使用自己的 bot token：

```bash
coder gateway start           # 启动 coder 的网关
assistant gateway start       # 启动 assistant 的网关（独立进程）
```

### 不同的 bot token

每个配置文件都有自己的 `.env` 文件。你可以在不同配置文件中配置不同的 Telegram / Discord / Slack bot token：

```bash
# 编辑 coder 的 token
nano ~/.hermes/profiles/coder/.env

# 编辑 assistant 的 token
nano ~/.hermes/profiles/assistant/.env
```

### 安全性：token 锁

如果两个配置文件意外使用了同一个 bot token，第二个网关会被阻止，并给出清晰错误信息，标明冲突的配置文件。Telegram、Discord、Slack、WhatsApp 和 Signal 都支持这一点。

### 常驻服务

```bash
coder gateway install         # 创建 hermes-gateway-coder systemd/launchd 服务
assistant gateway install     # 创建 hermes-gateway-assistant 服务
```

每个配置文件都会有自己的服务名，它们会独立运行。

## 配置配置文件

每个配置文件都有自己的：

- **`config.yaml`** - 模型、提供商、工具集、所有设置
- **`.env`** - API Key、bot token
- **`SOUL.md`** - 人格与指令

```bash
coder config set model.default anthropic/claude-sonnet-4
echo "You are a focused coding assistant." > ~/.hermes/profiles/coder/SOUL.md
```

如果你希望这个配置文件默认在某个项目里工作，也可以单独设置它自己的 `terminal.cwd`：

```bash
coder config set terminal.cwd /absolute/path/to/project
```

## 更新

`hermes update` 会拉取代码一次（共享），并自动把新内置技能同步到**所有**配置文件：

```bash
hermes update
# → Code updated (12 commits)
# → Skills synced: default (up to date), coder (+2 new), assistant (+2 new)
```

用户自己修改过的技能绝不会被覆盖。

## 管理配置文件

```bash
hermes profile list           # 显示所有配置文件及状态
hermes profile show coder     # 查看某个配置文件的详细信息
hermes profile rename coder dev-bot   # 重命名（会更新别名和服务）
hermes profile export coder    # 导出为 coder.tar.gz
hermes profile import coder.tar.gz   # 从归档导入
```

## 删除配置文件

```bash
hermes profile delete coder
```

这会停止网关、移除 systemd/launchd 服务、删除命令别名，并删除该配置文件的全部数据。系统会要求你输入配置文件名称进行确认。

使用 `--yes` 可以跳过确认：`hermes profile delete coder --yes`

:::note
你不能删除默认配置文件（`~/.hermes`）。如果要移除所有内容，请使用 `hermes uninstall`。
:::

## Tab 补全

```bash
# Bash
eval "$(hermes completion bash)"

# Zsh
eval "$(hermes completion zsh)"
```

把这行加入 `~/.bashrc` 或 `~/.zshrc` 后，就能永久启用补全。它会在 `-p` 后补全配置文件名、补全配置文件子命令以及顶层命令。

## 它是如何工作的

配置文件使用 `HERMES_HOME` 环境变量。当你运行 `coder chat` 时，包装脚本会在启动 hermes 前设置 `HERMES_HOME=~/.hermes/profiles/coder`。由于代码库里已有 119+ 个文件通过 `get_hermes_home()` 解析路径，Hermes 状态会自动作用于该配置文件目录 - 包括配置、会话、记忆、技能、状态数据库、网关 PID、日志和 cron 任务。

这和终端工作目录是分开的。工具执行会从 `terminal.cwd` 开始（或在 local 后端 `cwd: "."` 时从启动目录开始），并不会自动从 `HERMES_HOME` 开始。

默认配置文件就是 `~/.hermes` 本身。无需迁移 - 现有安装保持完全一致。

## 作为分发共享配置文件

你在一台机器上构建好的配置文件，可以打包成一个 **git 仓库**，并在另一台机器上一键安装 - 无论是你自己的工作站、队友的笔记本，还是社区用户的环境。共享包会包含 SOUL、配置、技能、cron 任务和 MCP 连接。凭据、记忆和会话仍然按机器隔离。

```bash
# 从 git 仓库安装一个完整 agent
hermes profile install github.com/you/research-bot --alias

# 之后作者发布新版本时更新（保留你的记忆 + .env）
hermes profile update research-bot
```

完整指南请见 **[Profile Distributions: Share a Whole Agent](/user-guide/profile-distributions)** - 包含编写、发布、更新语义、安全模型和使用场景。