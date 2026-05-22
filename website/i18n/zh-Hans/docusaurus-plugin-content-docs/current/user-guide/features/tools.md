---
sidebar_position: 1
title: "工具与工具集"
description: "Hermes Agent 工具概览 —— 可用工具、工具集工作原理和终端后端"
---

# 工具与工具集

工具是扩展智能体能力的函数。它们被组织成可按平台启用或禁用的逻辑**工具集**。

## 可用工具

Hermes 附带一个广泛的内置工具注册表，涵盖网络搜索、浏览器自动化、终端执行、文件编辑、内存、委派、RL 训练、消息传递、Home Assistant 等。

:::note
**Honcho 跨会话内存** 作为内存提供商插件（`plugins/memory/honcho/`）可用，而非内置工具集。请参阅 [插件](./plugins.md) 进行安装。
:::

高级类别：

| 类别 | 示例 | 描述 |
|----------|----------|-------------|
| **Web** | `web_search`, `web_extract` | 搜索网页并提取页面内容。 |
| **终端与文件** | `terminal`, `process`, `read_file`, `patch` | 执行命令和操作文件。 |
| **浏览器** | `browser_navigate`, `browser_snapshot`, `browser_vision` | 具有文本和视觉支持的交互式浏览器自动化。 |
| **媒体** | `vision_analyze`, `image_generate`, `text_to_speech` | 多模态分析和生成。 |
| **智能体编排** | `todo`, `clarify`, `execute_code`, `delegate_task` | 规划、澄清、代码执行和子智能体委派。 |
| **内存与回忆** | `memory`, `session_search` | 持久化内存和会话搜索。 |
| **自动化与交付** | `cronjob`, `send_message` | 具有创建/列表/更新/暂停/恢复/运行/删除操作的定时任务，以及出站消息传递交付。 |
| **集成** | `ha_*`, MCP 服务器工具, `rl_*` | Home Assistant、MCP、RL 训练和其他集成。 |

有关权威的代码派生注册表，请参阅 [内置工具参考](/reference/tools-reference) 和 [工具集参考](/reference/toolsets-reference)。

:::tip Nous 工具网关
付费 [Nous Portal](https://portal.nousresearch.com) 订阅者可以通过 **[工具网关](tool-gateway.md)** 使用网络搜索、图像生成、TTS 和浏览器自动化 —— 无需单独的 API 密钥。运行 `hermes model` 启用它，或使用 `hermes tools` 配置单个工具。
:::

## 使用工具集

```bash
# 使用特定工具集
hermes chat --toolsets "web,terminal"

# 查看所有可用工具
hermes tools

# 按平台配置工具（交互式）
hermes tools
```

常见工具集包括 `web`、`search`、`terminal`、`file`、`browser`、`vision`、`image_gen`、`moa`、`skills`、`tts`、`todo`、`memory`、`session_search`、`cronjob`、`code_execution`、`delegation`、`clarify`、`homeassistant`、`messaging`、`spotify`、`discord`、`discord_admin`、`debugging`、`safe` 和 `rl`。

有关完整集合（包括平台预设如 `hermes-cli`、`hermes-telegram` 和动态 MCP 工具集如 `mcp-<server>`），请参阅 [工具集参考](/reference/toolsets-reference)。

## 终端后端

终端工具可以在不同环境中执行命令：

| 后端 | 描述 | 用例 |
|---------|-------------|----------|
| `local` | 在您的机器上运行（默认） | 开发、受信任的任务 |
| `docker` | 隔离容器 | 安全性、可重现性 |
| `ssh` | 远程服务器 | 沙盒化，让智能体远离其自身代码 |
| `singularity` | HPC 容器 | 集群计算、无 root |
| `modal` | 云执行 | 无服务器、扩展 |
| `daytona` | 云沙盒工作区 | 持久化远程开发环境 |
| `vercel_sandbox` | Vercel Sandbox 云 microVM | 具有快照支持文件系统持久化的云执行 |

### 配置

```yaml
# 在 ~/.hermes/config.yaml 中
terminal:
  backend: local    # 或：docker, ssh, singularity, modal, daytona, vercel_sandbox
  cwd: "."          # 工作目录
  timeout: 180      # 命令超时（秒）
```

### Docker 后端

```yaml
terminal:
  backend: docker
  docker_image: python:3.11-slim
```

**一个持久容器，在整个进程中共享。** Hermes 在首次使用时启动一个长期运行的容器（`docker run -d ... sleep 2h`），并通过 `docker exec` 将每个终端、文件和 `execute_code` 调用路由到同一个容器。工作目录更改、已安装的包、环境调整和写入 `/workspace` 的文件都会从一个工具调用延续到下一个，跨越 `/new`、`/reset` 和 `delegate_task` 子智能体，在 Hermes 进程的整个生命周期内。容器在关闭时停止并移除。

这意味着 Docker 后端的行为类似于持久化沙盒 VM，而非每个命令一个全新容器。如果您 `pip install foo` 一次，它会在会话的其余时间内存在。如果您 `cd /workspace/project`，后续的 `ls` 调用会看到该目录。有关完整的生命周期详情和 `container_persistent` 标志（控制 `/workspace` 和 `/root` 是否在 Hermes 重启之间保留），请参阅 [配置 → Docker 后端](../configuration.md#docker-backend)。

### SSH 后端

推荐用于安全性 —— 智能体无法修改其自身代码：

```yaml
terminal:
  backend: ssh
```
```bash
# 在 ~/.hermes/.env 中设置凭证
TERMINAL_SSH_HOST=my-server.example.com
TERMINAL_SSH_USER=myuser
TERMINAL_SSH_KEY=~/.ssh/id_rsa
```

### Singularity/Apptainer

```bash
# 为并行工作器预构建 SIF
apptainer build ~/python.sif docker://python:3.11-slim

# 配置
hermes config set terminal.backend singularity
hermes config set terminal.singularity_image ~/python.sif
```

### Modal（无服务器云）

```bash
uv pip install modal
modal setup
hermes config set terminal.backend modal
```

### Vercel Sandbox

```bash
pip install 'hermes-agent[vercel]'
hermes config set terminal.backend vercel_sandbox
hermes config set terminal.vercel_runtime node24
```

使用全部三个 `VERCEL_TOKEN`、`VERCEL_PROJECT_ID` 和 `VERCEL_TEAM_ID` 进行认证。这种访问令牌设置是 Render、Railway、Docker 和类似主机上部署和正常长期运行 Hermes 进程的支持路径。支持的运行时包括 `node24`、`node22` 和 `python3.13`；Hermes 默认使用 `/vercel/sandbox` 作为远程工作区根目录。

对于一次性本地开发，Hermes 也接受短期 Vercel OIDC 令牌：

```bash
VERCEL_OIDC_TOKEN="$(vc project token <project-name>)" hermes chat
```

从链接的 Vercel 项目目录：

```bash
VERCEL_OIDC_TOKEN="$(vc project token)" hermes chat
```

当 `container_persistent: true` 时，Hermes 使用 Vercel 快照来保留同一任务跨沙盒重新创建的文件系统状态。这可以包括沙盒内的 Hermes 同步凭证、技能和缓存文件。快照不保留实时进程、PID 空间或相同的实时沙盒身份。

后台终端命令使用 Hermes 的通用非本地进程流：生成、轮询、等待、日志和终止在沙盒存活时通过正常进程工具工作，但 Hermes 在清理或重启后不提供原生 Vercel 分离进程恢复。

将 `container_disk` 留空或保持共享默认值 `51200`；Vercel Sandbox 不支持自定义磁盘大小，并且会导致诊断/后端创建失败。

### 容器资源

为所有容器后端配置 CPU、内存、磁盘和持久化：

```yaml
terminal:
  backend: docker  # 或 singularity, modal, daytona, vercel_sandbox
  container_cpu: 1              # CPU 核心（默认：1）
  container_memory: 5120        # 内存（MB）（默认：5GB）
  container_disk: 51200         # 磁盘（MB）（默认：50GB）
  container_persistent: true    # 跨会话持久化文件系统（默认：true）
```

当 `container_persistent: true` 时，已安装的包、文件和配置在会话之间保留。

### 容器安全

所有容器后端都运行安全加固：

- 只读 root 文件系统（Docker）
- 丢弃所有 Linux capabilities
- 无特权提升
- PID 限制（256 个进程）
- 完整的命名空间隔离
- 通过卷持久化工作区，而非可写 root 层

Docker 可以通过 `terminal.docker_forward_env` 接收显式环境允许列表，但转发的变量在容器内对命令可见，应视为暴露给该会话。

## 后台进程管理

启动后台进程并管理它们：

```python
terminal(command="pytest -v tests/", background=true)
# 返回：{"session_id": "proc_abc123", "pid": 12345}

# 然后使用进程工具管理：
process(action="list")       # 显示所有运行中的进程
process(action="poll", session_id="proc_abc123")   # 检查状态
process(action="wait", session_id="proc_abc123")   # 阻塞直到完成
process(action="log", session_id="proc_abc123")    # 完整输出
process(action="kill", session_id="proc_abc123")   # 终止
process(action="write", session_id="proc_abc123", data="y")  # 发送输入
```

PTY 模式（`pty=true`）启用交互式 CLI 工具，如 Codex 和 Claude Code。

## Sudo 支持

如果命令需要 sudo，系统会提示您输入密码（在会话中缓存）。或者在 `~/.hermes/.env` 中设置 `SUDO_PASSWORD`。

:::warning
在消息平台上，如果 sudo 失败，输出会包含一个提示，建议将 `SUDO_PASSWORD` 添加到 `~/.hermes/.env`。
:::
