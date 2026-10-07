---
sidebar_position: 3
title: "更新与卸载"
description: "如何将 Rabbit Agent 更新至最新版本或将其卸载"
---

# 更新与卸载

## 更新

使用单条命令更新至最新版本：

```bash
rabbit update
```

此命令会从 `main` 拉取最新代码、更新依赖项，并提示你配置自上次更新以来新增的选项。

:::tip
`rabbit update` 会自动检测新的配置选项并提示你添加。如果跳过了该提示，可手动运行 `rabbit config check` 查看缺失的选项，再运行 `rabbit config migrate` 以交互方式添加。
:::

### 更新过程

运行 `rabbit update` 时，将依次执行以下步骤：

1. **更新前快照** — Rabbit 在每个 profile 的 `state-snapshots/` 目录中保存指定的状态文件，包括配对数据、cron 任务、`config.yaml`、`.env` 和 `auth.json`。自动快速快照会跳过单个大于 1 GiB 的文件。`updates.pre_update_backup` 可选择 `quick`、`full` 或 `off`。完整归档遵循[备份排除规则](../reference/faq.md#rabbit-backup-vs-rabbit-profile-export)。恢复方法见[快照与回滚](../user-guide/checkpoints-and-rollback.md)。快速快照恢复的是状态文件，不是应用程序代码。
2. **Git pull** — 从 `main` 分支拉取最新代码并更新子模块
3. **依赖安装** — 运行 `python -m pm.cli install` 以获取新增或变更的依赖项
4. **配置迁移** — 检测自当前版本以来新增的配置选项并提示设置
5. **Gateway 自动重启** — 更新完成后刷新正在运行的 gateway，使新代码立即生效。由服务管理的 gateway（Linux 上的 systemd、macOS 上的 launchd）通过服务管理器重启；手动启动的 gateway 在 Rabbit 能将运行中的 PID 映射回某个 profile 时会自动重新启动。

### 仅预览：`rabbit update --check`

想在拉取前确认是否有更新？运行 `rabbit update --check` — 它会获取并与 `origin/main` 比较提交。不修改任何文件，不重启 gateway。适合在以"是否有更新"为条件的脚本和 cron 任务中使用。

### 完整更新前备份：`--backup`

对于高价值 profile（生产环境 gateway、团队共享安装），可选择在拉取前对 `RABBIT_HOME`（配置、认证、会话、技能、配对数据）进行完整备份：

```bash
rabbit update --backup
```

或将其设为每次运行的默认行为：

```yaml
# ~/.rabbit/config.yaml
updates:
  pre_update_backup: full
```

`updates.pre_update_backup` 有三种模式：

- `quick` 保存上述指定的状态文件。这是默认模式。
- `full` 另加一份遵循[备份排除规则](../reference/faq.md#rabbit-backup-vs-rabbit-profile-export)的 zip 归档。大型数据目录可能需要几分钟。
- `off` 禁用更新前备份。`--no-backup` 为单次运行选择此模式。

旧版布尔值仍然有效：`true` 等同于 `full`，`false` 等同于 `off`。

### Windows：另一个 `rabbit.exe` 正在运行

在 Windows 上，如果 `rabbit update` 检测到另一个 `rabbit.exe` 进程持有 venv 入口点可执行文件的句柄，它将拒绝运行 — 最常见的情况是 Rabbit Desktop 应用启动的后端进程、另一个终端中打开的 `rabbit` REPL，或正在运行的 gateway：

```
$ rabbit update
✗ Another rabbit.exe is running:
    PID 12345  rabbit.exe

  Updating now would fail to overwrite ...\venv\Scripts\rabbit.exe because
  Windows blocks REPLACE on a running executable.

  Close Rabbit Desktop, exit any open `rabbit` REPLs, and
  stop the gateway (`rabbit gateway stop`) before retrying.
  Override with `rabbit update --force` if you've already
  confirmed those processes will not write to the venv.
```

关闭列出的进程后重试。如果你确定并发进程不会造成干扰（极少见 — 通常仅在杀毒软件 shim 被误判时有用），可传入 `--force` 跳过检查。此时更新程序仍会以指数退避方式重试 `.exe` 重命名操作，对于顽固的文件锁，会通过 `MoveFileEx(MOVEFILE_DELAY_UNTIL_REBOOT)` 将替换操作安排在下次重启时执行，以确保更新能够完成。

预期输出如下：

```
$ rabbit update
Updating Rabbit Agent...
📥 Pulling latest code...
Already up to date.  (or: Updating abc1234..def5678)
📦 Updating dependencies...
✅ Dependencies updated
🔍 Checking for new config options...
✅ Config is up to date  (or: Found 2 new options — running migration...)
🔄 Restarting gateways...
✅ Gateway restarted
✅ Rabbit Agent updated successfully!
```

### 更新后建议的验证步骤

`rabbit update` 处理主要的更新流程，但快速验证可确认一切正常落地：

1. `git status --short` — 若工作树出现意外的脏状态，请在继续前检查
2. `rabbit doctor` — 检查配置、依赖项和服务健康状态
3. `rabbit --version` — 确认版本已按预期更新
4. 如果使用 gateway：`rabbit gateway status`
5. 如果 `doctor` 报告 npm audit 问题：在标记的目录中运行 `npm audit fix`

:::warning 更新后工作树出现脏状态
如果 `rabbit update` 后 `git status --short` 显示意外变更，请在继续前停下来检查。这通常意味着本地修改被重新应用到了更新后的代码之上，或依赖步骤刷新了锁文件。
:::

### 终端在更新中途断开连接

`rabbit update` 针对意外终端断开进行了保护：

- 更新会忽略 `SIGHUP`，因此关闭 SSH 会话或终端窗口不再会在安装中途终止它。`pip` 和 `git` 子进程继承此保护，因此 Python 环境不会因连接断开而处于半安装状态。
- 更新运行期间，所有输出会同步镜像到 `~/.rabbit/logs/update.log`。如果终端消失，重新连接后检查日志，确认更新是否完成以及 gateway 重启是否成功：

```bash
tail -f ~/.rabbit/logs/update.log
```

- `Ctrl-C`（SIGINT）和系统关机（SIGTERM）仍会被响应 — 这些是主动取消操作，而非意外中断。

你不再需要将 `rabbit update` 包裹在 `screen` 或 `tmux` 中来应对终端断开。

### 查看当前版本

```bash
rabbit --version
```

与 [GitHub releases 页面](https://github.com/seven0070/Rabbit-/releases) 上的最新版本进行比较。

### 从消息平台更新

你也可以直接从 Telegram、Discord、Slack、WhatsApp 或 Teams 发送以下命令进行更新：

```
/update
```

此命令会拉取最新代码、更新依赖项并重启正在运行的 gateway。Bot 在重启期间会短暂下线（通常为 5–15 秒），之后恢复服务。

### 手动更新

如果你是手动安装的（未使用快速安装脚本）：

```bash
cd /path/to/rabbit-agent
export VIRTUAL_ENV="$(pwd)/venv"

# Pull latest code
git pull origin main

# Reinstall (picks up new dependencies)
python -m pm.cli install

# Check for new config options
rabbit config check
rabbit config migrate   # Interactively add any missing options
```

### 回滚说明

如果更新引入了问题，可以回滚到之前的版本：

```bash
cd /path/to/rabbit-agent

# List recent versions
git log --oneline -10

# Roll back to a specific commit
git checkout <commit-hash>
python -m pm.cli install

# Restart the gateway if running
rabbit gateway restart
```

回滚到特定发布标签：

```bash
git checkout v0.6.0
python -m pm.cli install
```

:::warning
如果新增了配置选项，回滚可能导致配置不兼容。回滚后运行 `rabbit config check`，如果遇到错误，请从 `config.yaml` 中删除无法识别的选项。
:::

### Nix 用户注意事项

如果你通过 Nix flake 安装，更新由 Nix 包管理器负责：

```bash
# Update the flake input
nix flake update rabbit-agent

# Or rebuild with the latest
nix profile upgrade rabbit-agent
```

Nix 安装是不可变的 — 回滚由 Nix 的 generation 系统处理：

```bash
nix profile rollback
```

详情参见 [Nix 安装](./nix-setup.md)。

---

## 卸载

```bash
rabbit uninstall
```

卸载程序会提供选项，让你保留配置文件（`~/.rabbit/`）以便将来重新安装。

### 手动卸载

```bash
rm -f ~/.local/bin/rabbit
rm -rf /path/to/rabbit-agent
rm -rf ~/.rabbit            # 可选 — 如计划重新安装则保留
```

:::info
如果你将 gateway 安装为系统服务，请先停止并禁用它：
```bash
rabbit gateway stop
# Linux: systemctl --user disable rabbit-gateway
# macOS: launchctl remove ai.rabbit.gateway
```
:::