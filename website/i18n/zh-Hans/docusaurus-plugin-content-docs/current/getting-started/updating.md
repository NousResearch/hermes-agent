---
sidebar_position: 3
title: "更新和卸载"
description: "如何将 Hermes Agent 更新到最新版本或卸载它"
---

# 更新和卸载

## 更新

用一条命令更新到最新版本：

```bash
hermes update
```

这会拉取最新代码、更新依赖项，并提示你配置自上次更新以来添加的任何新选项。

:::tip
`hermes update` 自动检测新配置选项并提示你添加它们。如果你跳过了该提示，可以手动运行 `hermes config check` 查看缺失的选项，然后 `hermes config migrate` 交互式添加它们。
:::

### 更新期间会发生什么

当你运行 `hermes update` 时，会执行以下步骤：

1. **配对数据快照** — 保存轻量级更新前状态快照（涵盖 `~/.hermes/pairing/`、飞书评论规则及其他运行时修改的状态文件）。可通过 `hermes backup restore --state pre-update` 回滚。
2. **Git pull** — 从 `main` 分支拉取最新代码并更新子模块
3. **依赖安装** — 运行 `uv pip install -e ".[all]"` 以获取新的或更改的依赖项
4. **配置迁移** — 检测自你的版本以来添加的新配置选项并提示你设置
5. **网关自动重启** — 更新完成后刷新正在运行的网关，以便新代码立即生效。服务管理的网关（Linux 上的 systemd、macOS 上的 launchd）通过服务管理器重启。手动网关在 Hermes 可以将运行中的 PID 映射回配置文件时自动重新启动。

### 仅预览：`hermes update --check`

想在实际拉取之前知道你是否落后于 `origin/main`？运行 `hermes update --check` — 它会获取、在本地提交和最新远程提交之间并排打印，如果同步则退出 `0`，如果落后则退出 `1`。不修改任何文件，不重启网关。在脚本和 cron 任务中用于检查"是否有更新"。

### 完整更新前备份：`--backup`

对于高价值配置文件（生产网关、共享团队安装），你可以选择加入 `HERMES_HOME`（配置、认证、会话、技能、配对）的完整预拉取备份：

```bash
hermes update --backup
```

或将其设为每次运行的默认行为：

```yaml
# ~/.hermes/config.yaml
update:
  backup: true
```

`--backup` 在早期构建中是始终开启的行为，但它为每个更新在大型 home 上增加了分钟数，所以现在是可选的。上面的轻量级配对数据快照仍然无条件运行。

预期输出如下：

```
$ hermes update
Updating Hermes Agent...
📥 Pulling latest code...
Already up to date.  (或：Updating abc1234..def5678)
📦 Updating dependencies...
✅ Dependencies updated
🔍 Checking for new config options...
✅ Config is up to date  (或：Found 2 new options — running migration...)
🔄 Restarting gateways...
✅ Gateway restarted
✅ Hermes Agent updated successfully!
```

### 建议的更新后验证

`hermes update` 处理主要更新路径，但快速验证可以确认一切正常落地：

1. `git status --short` — 如果树意外变脏，在继续前检查
2. `hermes doctor` — 检查配置、依赖项和服务健康状况
3. `hermes --version` — 确认版本如预期更新
4. 如果使用网关：`hermes gateway status`
5. 如果 `doctor` 报告 npm audit 问题：在标记的目录中运行 `npm audit fix`

:::warning 更新后工作树变脏
如果 `hermes update` 后 `git status --short` 显示意外更改，停下来检查它们。这通常意味着本地修改被重新应用到更新的代码之上，或者依赖步骤刷新了 lockfile。
:::

### 如果你的终端在更新中途断开连接

`hermes update` 保护自己免受意外的终端丢失：

- 更新忽略 `SIGHUP`，所以关闭 SSH 会话或终端窗口不再会在安装中途杀死它。`pip` 和 `git` 子进程继承此保护，因此 Python 环境不会因断开连接而留下半安装状态。
- 所有输出在更新运行时都被镜像到 `~/.hermes/logs/update.log`。如果你的终端消失了，重新连接并检查日志以查看更新是否完成以及网关重启是否成功：

```bash
tail -f ~/.hermes/logs/update.log
```

- `Ctrl+C`（SIGINT）和系统关闭（SIGTERM）仍然被尊重 — 那些是故意的取消，不是意外。

你不再需要将 `hermes update` 包在 `screen` 或 `tmux` 中来在终端掉线时存活。

### 检查你的当前版本

```bash
hermes version
```

与 [GitHub releases 页面](https://github.com/NousResearch/hermes-agent/releases)上的最新版本比较。

### 从消息平台更新

你也可以直接从 Telegram、Discord、Slack、WhatsApp 或 Teams 更新：

```
/update
```

这会拉取最新代码、更新依赖项并重启正在运行的网关。机器人在重启期间会短暂下线（通常 5–15 秒），然后恢复。

### 手动更新

如果你手动安装（不是通过快速安装程序）：

```bash
cd /path/to/hermes-agent
export VIRTUAL_ENV="$(pwd)/venv"

# 拉取最新代码和子模块
git pull origin main
git submodule update --init --recursive

# 重新安装（获取新依赖项）
uv pip install -e ".[all]"
uv pip install -e "./tinker-atropos"

# 检查新配置选项
hermes config check
hermes config migrate   # 交互式添加任何缺失的选项
```

### 回滚说明

如果更新引入了问题，你可以回滚到之前的版本：

```bash
cd /path/to/hermes-agent

# 列出最近版本
git log --oneline -10

# 回滚到特定提交
git checkout <commit-hash>
git submodule update --init --recursive
uv pip install -e ".[all]"

# 如果网关正在运行则重启
hermes gateway restart
```

要回滚到特定发布标签：

```bash
git checkout v0.6.0
git submodule update --init --recursive
uv pip install -e ".[all]"
```

:::warning
回滚可能在添加了新选项时导致配置不兼容。回滚后运行 `hermes config check` 并在遇到错误时从 `config.yaml` 中移除任何无法识别的选项。
:::

### Nix 用户注意事项

如果你通过 Nix flake 安装，更新通过 Nix 包管理器管理：

```bash
# 更新 flake 输入
nix flake update hermes-agent

# 或用最新版本重建
nix profile upgrade hermes-agent
```

Nix 安装是不可变的 — 回滚由 Nix 的生成系统处理：

```bash
nix profile rollback
```

有关更多详情，请参阅 [Nix 设置](./nix-setup.md)。

---

## 卸载

```bash
hermes uninstall
```

卸载程序为你提供保留配置文件（`~/.hermes/`）以供将来重新安装的选项。

### 手动卸载

```bash
rm -f ~/.local/bin/hermes
rm -rf /path/to/hermes-agent
rm -rf ~/.hermes            # 可选 — 如果你计划重新安装则保留
```

:::info
如果你将网关安装为系统服务，先停止并禁用它：
```bash
hermes gateway stop
# Linux: systemctl --user disable hermes-gateway
# macOS: launchctl remove ai.hermes.gateway
```
:::
