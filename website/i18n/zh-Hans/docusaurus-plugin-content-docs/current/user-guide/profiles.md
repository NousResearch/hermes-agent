---
sidebar_position: 2
---

# Profiles：运行多个 Agent

在同一台机器上运行多个独立的 Hermes agent——每个 agent 拥有各自的配置、API 密钥、记忆、会话、技能和 gateway 状态。

## 什么是 profile？

profile 是一个独立的 Hermes 主目录。每个 profile 拥有自己的目录，其中包含各自的 `config.yaml`、`.env`、`SOUL.md`、记忆、会话、技能、cron 任务和状态数据库。profile 让你可以为不同用途运行独立的 agent——编程助手、个人机器人、研究 agent——而不会混淆 Hermes 状态。

创建 profile 后，它会自动成为独立的命令。创建名为 `coder` 的 profile，你立即就拥有了 `coder chat`、`coder setup`、`coder gateway start` 等命令。

## 快速开始

```bash
hermes profile create coder       # 创建 profile + "coder" 命令别名
coder setup                       # 配置 API 密钥和模型
coder chat                        # 开始对话
```

就这些。`coder` 现在是拥有独立配置、记忆和状态的 Hermes profile。

## 创建 profile

### 空白 profile

```bash
hermes profile create mybot
```

创建一个预置了内置技能的全新 profile。运行 `mybot setup` 配置 API 密钥、模型和 gateway token。

如果你计划将此 profile 用作 kanban（看板）工作节点（或希望 kanban 编排器将任务路由到它），在创建时传入 `--description "<角色>"` 以便编排器了解其能力：

```bash
hermes profile create researcher --description "Reads source code and external docs, writes findings."
```

你也可以稍后通过 `hermes profile describe` 设置或自动生成描述——完整路由模型请参阅 [Kanban 指南](./features/kanban#auto-vs-manual-orchestration)。

### 仅克隆配置（`--clone`）

```bash
hermes profile create work --clone
```

将当前 profile 的 `config.yaml`、`.env`、`SOUL.md` 和 skills 复制到新 profile。API 密钥、模型和能力相同，但会话和记忆是全新的。编辑 `~/.hermes/profiles/work/.env` 可使用不同的 API 密钥，编辑 `~/.hermes/profiles/work/SOUL.md` 可设置不同的人格。

### 克隆全部内容（`--clone-all`）

```bash
hermes profile create backup --clone-all
```

复制**所有内容**——配置、API 密钥、人格、记忆、技能、cron 任务、插件。会排除每个 profile 自己的历史数据（会话历史、`state.db`、`backups/`、`state-snapshots/`、`checkpoints/`），这些数据属于源 profile 且可能达到数十 GB。若要包含历史的完整备份，请使用 `hermes profile export` 或 `hermes backup`。

### 从指定 profile 克隆

```bash
hermes profile create work --clone-from coder
```

`--clone-from <source>` 会直接选择源 profile，并隐含执行 config/skills/SOUL 克隆。若要完整复制该源 profile，请与 `--clone-all` 组合使用：

```bash
hermes profile create work-backup --clone-from coder --clone-all
```

:::tip Honcho 记忆 + profiles
启用 Honcho 后，克隆操作会自动为新 profile 创建专属 AI 对等体，同时共享同一用户工作区。每个 profile 构建各自的观察记录和身份标识。详见 [Honcho——多 agent / Profiles](./features/memory-providers.md#honcho)。
:::

## 使用 profile

### 命令别名

每个 profile 在 `~/.local/bin/<name>` 自动获得一个命令别名：

```bash
coder chat                    # 与 coder agent 对话
coder setup                   # 配置 coder 的设置
coder gateway start           # 启动 coder 的 gateway
coder doctor                  # 检查 coder 的健康状态
coder skills list             # 列出 coder 的技能
coder config set model.default anthropic/claude-sonnet-4
```

别名支持所有 hermes 子命令——底层实际上是 `hermes -p <name>`。

### `-p` 标志

你也可以通过任意命令显式指定 profile：

```bash
hermes -p coder chat
hermes --profile=coder doctor
hermes chat -p coder -q "hello"    # 可在任意位置使用
```

### 粘性默认值（`hermes profile use`）

```bash
hermes profile use coder
hermes chat                   # 现在指向 coder
hermes tools                  # 配置 coder 的工具
hermes profile use default    # 切换回默认
```

设置默认值后，普通 `hermes` 命令将指向该 profile。类似于 `kubectl config use-context`。

### 了解当前所在 profile

CLI 始终显示当前活跃的 profile：

- **提示符**：显示 `coder ❯` 而非 `❯`
- **启动横幅**：启动时显示 `Profile: coder`
- **`hermes profile`**：显示当前 profile 名称、路径、模型、gateway 状态

## Profile vs 工作区 vs 沙箱

profile 常与工作区或沙箱混淆，但它们是不同的概念：

- **profile** 为 Hermes 提供独立的状态目录：`config.yaml`、`.env`、`SOUL.md`、会话、记忆、日志、cron 任务和 gateway 状态。
- **工作区**或**工作目录**是终端命令的起始位置，由 `terminal.cwd` 单独控制。
- **沙箱**用于限制文件系统访问。profile **不**对 agent 进行沙箱隔离。

在默认的 `local` 终端后端，agent 仍拥有与你的用户账户相同的文件系统访问权限。profile 不会阻止其访问 profile 目录之外的文件夹。

如果你希望 profile 默认在特定项目文件夹中启动，请在该 profile 的 `config.yaml` 中设置绝对路径的 `terminal.cwd`：

```yaml
terminal:
  backend: local
  cwd: /absolute/path/to/project
```

在 local 后端使用 `cwd: "."` 表示"Hermes 启动时所在的目录"，而非"profile 目录"。

另请注意：

- `SOUL.md` 可以引导模型，但不能强制限定工作区边界。
- `SOUL.md` 的更改在新会话中会生效。现有会话可能仍在使用旧的 prompt（提示词）状态。
- 询问模型"你在哪个目录？"并不是可靠的隔离测试。如果你需要工具有可预测的起始目录，请显式设置 `terminal.cwd`。

## 运行 gateway

每个 profile 以独立进程运行各自的 gateway，使用各自的 bot token：

```bash
coder gateway start           # 启动 coder 的 gateway
assistant gateway start       # 启动 assistant 的 gateway（独立进程）
```

### 不同的 bot token

每个 profile 有各自的 `.env` 文件。在各文件中配置不同的 Telegram/Discord/Slack bot token：

```bash
# 编辑 coder 的 token
nano ~/.hermes/profiles/coder/.env

# 编辑 assistant 的 token
nano ~/.hermes/profiles/assistant/.env
```

### 安全性：token 锁

如果两个 profile 意外使用了相同的 bot token，第二个 gateway 将被阻止并显示明确的错误信息，指出冲突的 profile。支持 Telegram、Discord、Slack、WhatsApp 和 Signal。

### 持久化服务

```bash
coder gateway install         # 创建 hermes-gateway-coder systemd/launchd 服务
assistant gateway install     # 创建 hermes-gateway-assistant 服务
```

每个 profile 拥有独立的服务名称，各自独立运行。

:::note 在官方 Docker 镜像中
各 profile 的 gateway 由 [s6-overlay](https://github.com/just-containers/s6-overlay)（容器中的 PID 1）监管，因此 `hermes profile create <name>` 会自动在 `/run/service/gateway-<name>/` 注册 s6 服务槽。`hermes -p <name> gateway start/stop/restart` 会调度到 `s6-svc` 而非直接启动裸进程——崩溃后自动重启，`docker restart` 会保留之前运行的 gateway 集合。详见 [各 profile gateway 监管](/user-guide/docker#per-profile-gateway-supervision)。
:::

## 配置 profile

每个 profile 拥有各自的：

- **`config.yaml`** — 模型、提供商、工具集及所有设置
- **`.env`** — API 密钥、bot token
- **`SOUL.md`** — 人格与指令

```bash
coder config set model.default anthropic/claude-sonnet-4
echo "You are a focused coding assistant." > ~/.hermes/profiles/coder/SOUL.md
```

如果你希望此 profile 默认在特定项目中工作，还需设置其 `terminal.cwd`：

```bash
coder config set terminal.cwd /absolute/path/to/project
```

## 更新

`hermes update` 拉取一次代码（共享），并自动将新的内置技能同步到**所有** profile：

```bash
hermes update
# → Code updated (12 commits)
# → Skills synced: default (up to date), coder (+2 new), assistant (+2 new)
```

用户修改过的技能不会被覆盖。

## 管理 profile

```bash
hermes profile list           # 显示所有 profile 及其状态
hermes profile show coder     # 显示某个 profile 的详细信息
hermes profile rename coder dev-bot   # 重命名（同步更新别名和服务）
hermes profile export coder   # 导出为 coder.tar.gz
hermes profile import coder.tar.gz   # 从归档文件导入
```

## 删除 profile

```bash
hermes profile delete coder
```

此操作将停止 gateway、移除 systemd/launchd 服务、移除命令别名并删除所有 profile 数据。系统会要求你输入 profile 名称以确认。

使用 `--yes` 跳过确认：`hermes profile delete coder --yes`

:::note
你无法删除默认 profile（`~/.hermes`）。如需删除所有内容，请使用 `hermes uninstall`。
:::

## Tab 补全

```bash
# Bash
eval "$(hermes completion bash)"

# Zsh
eval "$(hermes completion zsh)"
```

将该行添加到 `~/.bashrc` 或 `~/.zshrc` 以启用持久补全。支持补全 `-p` 后的 profile 名称、profile 子命令及顶级命令。

## 工作原理

profile 使用 `HERMES_HOME` 环境变量。运行 `coder chat` 时，包装脚本在启动 hermes 前将 `HERMES_HOME` 设置为 `~/.hermes/profiles/coder`。由于代码库中 119+ 个文件通过 `get_hermes_home()` 解析路径，Hermes 状态会自动限定在 profile 目录范围内——包括配置、会话、记忆、技能、状态数据库、gateway PID、日志和 cron 任务。

这与终端工作目录是分开的。工具执行从 `terminal.cwd` 开始（或在 local 后端使用 `cwd: "."` 时从启动目录开始），而非自动从 `HERMES_HOME` 开始。

默认 profile 就是 `~/.hermes` 本身。无需迁移——现有安装的工作方式完全不变。

## 将 profile 作为发行版共享

你在一台机器上构建的 profile 可以打包为 **git 仓库**，并通过一条命令安装到另一台机器——你自己的工作站、团队成员的笔记本，或社区用户的环境。共享包包含 SOUL、配置、技能、cron 任务和 MCP 连接。凭据、记忆和会话保持各机器独立。

```bash
# 从 git 仓库安装完整 agent
hermes profile install github.com/you/research-bot --alias

# 当作者发布新版本时更新（保留你的记忆和 .env）
hermes profile update research-bot
```

完整指南请参阅 **[Profile 发行版：共享完整 Agent](./profile-distributions.md)**——包括编写、发布、更新语义、安全模型和使用场景。