---
sidebar_position: 3
title: "更新与卸载"
description: "如何将 Hermes Agent 更新至最新版本或将其卸载"
---

# 更新与卸载

## 更新

使用单条命令更新至最新版本：

```bash
hermes update
```

此命令会从 `main` 拉取最新代码、更新依赖项，并提示你配置自上次更新以来新增的选项。

:::tip
`hermes update` 会自动检测新的配置选项并提示你添加。如果跳过了该提示，可手动运行 `hermes config check` 查看缺失的选项，再运行 `hermes config migrate` 以交互方式添加。
:::

### 更新过程

运行 `hermes update` 时，将依次执行以下步骤：

1. **配对数据快照** — 保存一份轻量级的更新前状态快照（涵盖 `~/.hermes/pairing/`、飞书评论规则及其他运行时修改的状态文件）。可通过 [快照与回滚](../user-guide/checkpoints-and-rollback.md) 中描述的快照恢复流程进行恢复，或从 Hermes 写入 `~/.hermes/` 目录旁的最新快速快照 zip 文件中提取。
2. **Git pull** — 从 `main` 分支拉取最新代码并更新子模块
3. **依赖安装** — 运行 `uv pip install -e ".[all]"` 以获取新增或变更的依赖项
4. **配置迁移** — 检测自当前版本以来新增的配置选项并提示设置
5. **Gateway 自动重启** — 更新完成后刷新正在运行的 gateway，使新代码立即生效。由服务管理的 gateway（Linux 上的 systemd、macOS 上的 launchd）通过服务管理器重启；手动启动的 gateway 在 Hermes 能将运行中的 PID 映射回某个 profile 时会自动重新启动。

### 仅预览：`hermes update --check`

想在拉取前确认是否有更新？运行 `hermes update --check` — 它会获取并与 `origin/main` 比较提交。不修改任何文件，不重启 gateway。适合在以"是否有更新"为条件的脚本和 cron 任务中使用。

### 完整更新前备份：`--backup`

对于高价值 profile（生产环境 gateway、团队共享安装），可选择在拉取前对 `HERMES_HOME`（配置、认证、会话、技能、配对数据）进行完整备份：

```bash
hermes update --backup
```

或将其设为每次运行的默认行为：

```yaml
# ~/.hermes/config.yaml
updates:
  pre_update_backup: true
```

`--backup` 在早期版本中是始终开启的行为，但在大型 home 目录上会给每次更新增加数分钟时间，因此现已改为按需启用。上述轻量级配对数据快照仍会无条件执行。

### Windows：另一个 `hermes.exe` 正在运行

在 Windows 上，如果 `hermes update` 检测到另一个 `hermes.exe` 进程持有 venv 入口点可执行文件的句柄，它将拒绝运行 — 最常见的情况是 Hermes Desktop 应用启动的后端进程、另一个终端中打开的 `hermes` REPL，或正在运行的 gateway：

```
$ hermes update
✗ Another hermes.exe is running:
    PID 12345  hermes.exe

  Updating now would fail to overwrite ...\venv\Scripts\hermes.exe because
  Windows blocks REPLACE on a running executable.

  Close Hermes Desktop, exit any open `hermes` REPLs, and
  stop the gateway (`hermes gateway stop`) before retrying.
  Override with `hermes update --force` if you've already
  confirmed those processes will not write to the venv.
```

关闭列出的进程后重试。如果你确定并发进程不会造成干扰（极少见 — 通常仅在杀毒软件 shim 被误判时有用），可传入 `--force` 跳过检查。此时更新程序仍会以指数退避方式重试 `.exe` 重命名操作，对于顽固的文件锁，会通过 `MoveFileEx(MOVEFILE_DELAY_UNTIL_REBOOT)` 将替换操作安排在下次重启时执行，以确保更新能够完成。

预期输出如下：

```
$ hermes update
Updating Hermes Agent...
📥 Pulling latest code...
Already up to date.  (or: Updating abc1234..def5678)
📦 Updating dependencies...
✅ Dependencies updated
🔍 Checking for new config options...
✅ Config is up to date  (or: Found 2 new options — running migration...)
🔄 Restarting gateways...
✅ Gateway restarted
✅ Hermes Agent updated successfully!
```

### 更新后建议的验证步骤

`hermes update` 处理主要的更新流程，但快速验证可确认一切正常落地：

1. `git status --short` — 若工作树出现意外的脏状态，请在继续前检查
2. `hermes doctor` — 检查配置、依赖项和服务健康状态
3. `hermes --version` — 确认版本已按预期更新
4. 如果使用 gateway：`hermes gateway status`
5. 如果 `doctor` 报告 npm audit 问题：在标记的目录中运行 `npm audit fix`

:::warning 更新后工作树出现脏状态
如果 `hermes update` 后 `git status --short` 显示意外变更，请在继续前停下来检查。这通常意味着本地修改被重新应用到了更新后的代码之上，或依赖步骤刷新了锁文件。
:::

### 终端在更新中途断开连接

`hermes update` 针对意外终端断开进行了保护：

- 更新会忽略 `SIGHUP`，因此关闭 SSH 会话或终端窗口不再会在安装中途终止它。`pip` 和 `git` 子进程继承此保护，因此 Python 环境不会因连接断开而处于半安装状态。
- 更新运行期间，所有输出会同步镜像到 `~/.hermes/logs/update.log`。如果终端消失，重新连接后检查日志，确认更新是否完成以及 gateway 重启是否成功：

```bash
tail -f ~/.hermes/logs/update.log
```

- `Ctrl-C`（SIGINT）和系统关机（SIGTERM）仍会被响应 — 这些是主动取消操作，而非意外中断。

你不再需要将 `hermes update` 包裹在 `screen` 或 `tmux` 中来应对终端断开。

### 查看当前版本

```bash
hermes version
```

与 [GitHub releases 页面](https://github.com/NousResearch/hermes-agent/releases) 上的最新版本进行比较。

### 从消息平台更新

你也可以直接从 Telegram、Discord、Slack、WhatsApp 或 Teams 发送以下命令进行更新：

```
/update
```

此命令会拉取最新代码、更新依赖项并重启正在运行的 gateway。Bot 在重启期间会短暂下线（通常为 5–15 秒），之后恢复服务。

### 手动更新

如果你是手动安装的（未使用快速安装脚本）：

```bash
cd /path/to/hermes-agent
export VIRTUAL_ENV="$(pwd)/venv"

# Pull latest code
git pull origin main

# Reinstall (picks up new dependencies)
uv pip install -e ".[all]"

# Check for new config options
hermes config check
hermes config migrate   # Interactively add any missing options
```

### 回滚说明

如果更新引入了问题，可以回滚到之前的版本：

```bash
cd /path/to/hermes-agent

# List recent versions
git log --oneline -10

# Roll back to a specific commit
git checkout <commit-hash>
uv pip install -e ".[all]"

# Restart the gateway if running
hermes gateway restart
```

回滚到特定发布标签：

```bash
git checkout v0.6.0
uv pip install -e ".[all]"
```

:::warning
如果新增了配置选项，回滚可能导致配置不兼容。回滚后运行 `hermes config check`，如果遇到错误，请从 `config.yaml` 中删除无法识别的选项。
:::

### Nix 用户注意事项

如果你通过 Nix flake 安装，更新由 Nix 包管理器负责：

```bash
# Update the flake input
nix flake update hermes-agent

# Or rebuild with the latest
nix profile upgrade hermes-agent
```

Nix 安装是不可变的 — 回滚由 Nix 的 generation 系统处理：

```bash
nix profile rollback
```

详情参见 [Nix 安装](./nix-setup.md)。

---

## 卸载

```bash
hermes uninstall
```

卸载程序会提供选项，让你保留配置文件（`~/.hermes/`）以便将来重新安装。

### 手动卸载

```bash
rm -f ~/.local/bin/hermes
rm -rf /path/to/hermes-agent
rm -rf ~/.hermes            # 可选 — 如计划重新安装则保留
```

:::info
如果你将 gateway 安装为系统服务，请先停止并禁用它：
```bash
hermes gateway stop
# Linux: systemctl --user disable hermes-gateway
# macOS: launchctl remove ai.hermes.gateway
```
:::