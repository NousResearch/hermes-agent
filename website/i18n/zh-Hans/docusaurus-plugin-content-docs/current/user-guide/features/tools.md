---
sidebar_position: 1
title: "工具和工具集"
description: "Hermes 代理工具概览——可用什么、工具集如何工作以及终端后端"
---

# 工具和工具集

工具是扩展代理能力的函数。它们被组织成逻辑**工具集**，可以按平台启用或禁用。

## 可用工具

Hermes 附带广泛的内置工具注册表，涵盖网络搜索、浏览器自动化、终端执行、文件编辑、记忆、委托、强化学习训练、消息传递交付、Home Assistant 等。

:::note
**Honcho 跨会话记忆**可作为记忆提供商插件使用（`plugins/memory/honcho/`），而不是内置工具集。参见[插件](./plugins.md)了解安装。
:::

高级类别：

| 类别 | 示例 | 描述 |
|----------|----------|-------------|
| **Web** | `web_search`, `web_extract` | 搜索网络并提取页面内容。 |
| **终端和文件** | `terminal`, `process`, `read_file`, `patch` | 执行命令并操作文件。 |
| **浏览器** | `browser_navigate`, `browser_snapshot`, `browser_vision` | 带有文本和视觉支持的交互式浏览器自动化。 |
| **媒体** | `vision_analyze`, `image_generate`, `text_to_speech` | 多模态分析和生成。 |
| **代理编排** | `todo`, `clarify`, `execute_code`, `delegate_task` | 规划、澄清、代码执行和子代理委托。 |
| **记忆和回忆** | `memory`, `session_search` | 持久记忆和会话搜索。 |
| **自动化和交付** | `cronjob`, `send_message` | 具有创建/列表/更新/暂停/恢复/运行/删除操作的计划任务，加上出站消息传递交付。 |
| **集成** | `ha_*`, MCP 服务器工具, `rl_*` | Home Assistant、MCP、强化学习训练和其他集成。 |

有关权威的代码派生注册表，请参阅[内置工具参考](/docs/reference/tools-reference)和[工具集参考](/docs/reference/toolsets-reference)。

:::tip Nous 工具网关
付费 [Nous Portal](https://portal.nousresearch.com) 订阅者可以通过**[工具网关](tool-gateway.md)**使用网络搜索、图像生成、TTS 和浏览器自动化——无需单独的 API 密钥。运行 `hermes model` 启用它，或使用 `hermes tools` 配置各个工具。
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

请参阅[工具集参考](/docs/reference/toolsets-reference)获取完整集合，包括平台预设如 `hermes-cli`、`hermes-telegram` 和动态 MCP 工具集如 `mcp-<server>`。

## 终端后端

终端工具可以在不同环境中执行命令：

| 后端 | 描述 | 用例 |
|---------|-------------|----------|
| `local` | 在你的机器上运行（默认） | 开发、信任的任务 |
| `docker` | 隔离容器 | 安全、可重现性 |
| `ssh` | 远程服务器 | 沙箱、将代理远离其自己的代码 |
| `singularity` | HPC 容器 | 集群计算、无根 |
| `modal` | 云执行 | 无服务器、扩展 |
| `daytona` | 云沙箱工作区 | 持久远程开发环境 |
| `vercel_sandbox` | Vercel Sandbox 云微VM | 带快照支持文件系统持久化的云执行 |

### 配置

```yaml
# 在 ~/.hermes/config.yaml
terminal:
  backend: local    # 或: docker, ssh, singularity, modal, daytona, vercel_sandbox
  cwd: "."          # 工作目录
  timeout: 180      # 命令超时秒数
```

### Docker 后端

```yaml
terminal:
  backend: docker
  docker_image: python:3.11-slim
```

**一个持久容器，跨整个进程共享。** Hermes 在首次使用时启动一个单一长期容器（`docker run -d ... sleep 2h`），并通过 `docker exec` 将每个终端、文件和 `execute_code` 调用路由到同一个容器。工作目录更改、安装的包、环境调整和写入 `/workspace` 的文件都从一个工具调用延续到下一个，跨 `/new`、`/reset` 和 `delegate_task` 子代理，在 Hermes 进程的生命周期内。容器在关闭时停止并移除。

这意味着 Docker 后端的行为像持久沙箱 VM，而不是每个命令的新鲜容器。如果你 `pip install foo` 一次，它在会话的其余部分都存在。如果你 `cd /workspace/project`，后续的 `ls` 调用会看到那个目录。请参阅[配置 → Docker 后端](../configuration.md#docker-backend)获取完整的生命周期细节和控制 `/workspace` 和 `/root` 是否跨 Hermes 重启存活的 `container_persistent` 标志。

### SSH 后端

推荐用于安全——代理无法修改其自己的代码：

```yaml
terminal:
  backend: ssh
```
```bash
# 在 ~/.hermes/.env 中设置凭据
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

使用所有三个进行身份验证：`VERCEL_TOKEN`、`VERCEL_PROJECT_ID` 和 `VERCEL_TEAM_ID`。此访问令牌设置是部署和在 Render、Railway、Docker 及类似主机上正常运行 Hermes 进程的支持路径。支持 `node24`、`node22` 和 `python3.13` 运行时；Hermes 默认使用 `/vercel/sandbox` 作为远程工作区根目录。

对于一次性本地开发，Hermes 也接受短期 Vercel OIDC 令牌：

```bash
VERCEL_OIDC_TOKEN="$(vc project token <project-name>)" hermes chat
```

从链接的 Vercel 项目目录：

```bash
VERCEL_OIDC_TOKEN="$(vc project token)" hermes chat
```

使用 `container_persistent: true`，Hermes 使用 Vercel 快照来在相同任务的沙箱重新创建之间保留文件系统状态。这可以包括 Hermes 同步的凭据、技能和沙箱内的缓存文件。快照不保留实时进程、PID 空间或相同的实时沙箱身份。

后台终端命令使用 Hermes 的通用非本地进程流程：当沙箱存活时，生成、轮询、等待、日志和终止通过正常进程工具工作，但 Hermes 在清理或重启后不提供原生 Vercel 分离进程恢复。

将 `container_disk` 留空或使用共享默认值 `51200`；不支持自定义磁盘大小，会导致诊断/后端创建失败。

### 容器资源

为所有容器后端配置 CPU、内存、磁盘和持久化：

```yaml
terminal:
  backend: docker  # 或 singularity, modal, daytona, vercel_sandbox
  container_cpu: 1              # CPU 核心（默认：1）
  container_memory: 5120        # 内存 MB（默认：5GB）
  container_disk: 51200         # 磁盘 MB（默认：50GB）
  container_persistent: true    # 跨会话保留文件系统（默认：true）
```

当 `container_persistent: true` 时，安装的包、文件和配置跨会话保留。

### 容器安全

所有容器后端都使用安全加固运行：

- 只读根文件系统（Docker）
- 所有 Linux 能力被删除
- 无权限提升
- PID 限制（256 个进程）
- 完整命名空间隔离
- 通过卷而非可写根层的持久工作区

Docker 可以选择通过 `terminal.docker_forward_env` 接收显式环境白名单，但转发的变量对容器内的命令可见，应被视为对该会话暴露。

## 后台进程管理

启动后台进程并管理它们：

```python
terminal(command="pytest -v tests/", background=true)
# 返回: {"session_id": "proc_abc123", "pid": 12345}

# 然后用进程工具管理：
process(action="list")       # 显示所有运行中的进程
process(action="poll", session_id="proc_abc123")   # 检查状态
process(action="wait", session_id="proc_abc123")   # 阻塞直到完成
process(action="log", session_id="proc_abc123")    # 完整输出
process(action="kill", session_id="proc_abc123")   # 终止
process(action="write", session_id="proc_abc123", data="y")  # 发送输入
```

PTY 模式（`pty=true`）启用交互式 CLI 工具如 Codex 和 Claude Code。

## Sudo 支持

如果命令需要 sudo，系统会提示你输入密码（会话缓存）。或在 `~/.hermes/.env` 中设置 `SUDO_PASSWORD`。

:::warning
在消息平台上，如果 sudo 失败，输出包含一个提示，建议将 `SUDO_PASSWORD` 添加到 `~/.hermes/.env`。
:::
