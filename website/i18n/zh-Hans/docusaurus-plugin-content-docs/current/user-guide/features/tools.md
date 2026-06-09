---
sidebar_position: 1
title: "工具与工具集"
description: "Hermes Agent 工具概览——可用工具、工具集工作方式及终端后端"
---

# 工具与工具集

工具是扩展 Agent 能力的函数。它们被组织为逻辑上的**工具集**，可按平台启用或禁用。

## 可用工具

Hermes 内置了丰富的工具注册表，涵盖网页搜索、浏览器自动化、终端执行、文件编辑、记忆、委托、RL 训练、消息投递、Home Assistant 等功能。

:::note
**Honcho 跨会话记忆**作为记忆提供者插件（`plugins/memory/honcho/`）提供，而非内置工具集。安装方式请参阅 [Plugins](./plugins.md)。
:::

高层分类：

| 分类 | 示例 | 描述 |
|----------|----------|-------------|
| **Web** | `web_search`, `web_extract` | 搜索网页并提取页面内容。 |
| **X 搜索** | `x_search` | 通过 xAI 内置的 `x_search` Responses 工具搜索 X（Twitter）帖子和话题——需要 xAI 凭据（SuperGrok OAuth 或 `XAI_API_KEY`）；默认关闭，可通过 `hermes tools` → 🐦 X (Twitter) Search 启用。 |
| **终端与文件** | `terminal`, `process`, `read_file`, `patch` | 执行命令并操作文件。 |
| **浏览器** | `browser_navigate`, `browser_snapshot`, `browser_vision` | 支持文本和视觉的交互式浏览器自动化。 |
| **媒体** | `vision_analyze`, `image_generate`, `video_generate`, `video_analyze`, `text_to_speech` | 多模态分析与生成。`video_generate` 和 `video_analyze` 需手动启用（通过 `hermes tools` 或 `--toolsets` 添加 `video_gen` / `video` 工具集）。 |
| **Agent 编排** | `todo`, `clarify`, `execute_code`, `delegate_task` | 规划、澄清、代码执行及子 Agent 委托。 |
| **记忆与召回** | `memory`, `session_search` | 持久化记忆与会话搜索。 |
| **自动化与投递** | `cronjob`, `send_message` | 支持创建/列出/更新/暂停/恢复/运行/删除操作的定时任务，以及出站消息投递。 |
| **集成** | `ha_*`、MCP server 工具、`rl_*` | Home Assistant、MCP、RL 训练及其他集成。 |

如需查看由代码派生的权威注册表，请参阅 [内置工具参考](/reference/tools-reference) 和 [工具集参考](/reference/toolsets-reference)。

:::tip Nous Tool Gateway
付费 [Nous Portal](https://portal.nousresearch.com) 订阅者可通过 **[Tool Gateway](tool-gateway.md)** 使用网页搜索、图像生成、TTS 和浏览器自动化——无需单独配置 API 密钥。运行 `hermes model` 启用，或通过 `hermes tools` 配置各工具。
:::

## 使用工具集

```bash
# 使用指定工具集
hermes chat --toolsets "web,terminal"

# 查看所有可用工具
hermes tools

# 按平台交互式配置工具
hermes tools
```

常用工具集包括 `web`、`search`、`terminal`、`file`、`browser`、`vision`、`image_gen`、`moa`、`skills`、`tts`、`todo`、`memory`、`session_search`、`cronjob`、`code_execution`、`delegation`、`clarify`、`homeassistant`、`messaging`、`spotify`、`discord`、`discord_admin`、`debugging`、`safe` 和 `rl`。

完整列表（包括 `hermes-cli`、`hermes-telegram` 等平台预设以及 `mcp-<server>` 等动态 MCP 工具集）请参阅 [工具集参考](/reference/toolsets-reference)。

## 终端后端

终端工具可在不同环境中执行命令：

| 后端 | 描述 | 适用场景 |
|---------|-------------|----------|
| `local` | 在本机运行（默认） | 开发、可信任务 |
| `docker` | 隔离容器 | 安全性、可复现性 |
| `ssh` | 远程服务器 | 沙箱隔离，防止 Agent 修改自身代码 |
| `singularity` | HPC 容器 | 集群计算、无 root 权限 |
| `modal` | 云端执行 | 无服务器、弹性扩展 |
| `daytona` | 云端沙箱工作区 | 持久化远程开发环境 |

### 配置

```yaml
# 在 ~/.hermes/config.yaml 中
terminal:
  backend: local    # 或：docker, ssh, singularity, modal, daytona
  cwd: "."          # 工作目录
  timeout: 180      # 命令超时时间（秒）
```

### Docker 后端

```yaml
terminal:
  backend: docker
  docker_image: python:3.11-slim
```

**单个持久容器，在整个进程生命周期内共享。** Hermes 在首次使用时启动一个长期运行的容器（`docker run -d ... sleep 2h`），并通过 `docker exec` 将所有终端、文件及 `execute_code` 调用路由到同一容器中。工作目录变更、已安装的包、环境调整以及写入 `/workspace` 的文件，在同一 Hermes 进程的整个生命周期内，跨 `/new`、`/reset` 和 `delegate_task` 子 Agent 均会保留。容器在关闭时停止并删除。

这意味着 Docker 后端的行为类似持久化沙箱虚拟机，而非每次命令都使用全新容器。如果你执行过一次 `pip install foo`，该包在本次会话的剩余时间内均可用。如果你执行了 `cd /workspace/project`，后续的 `ls` 调用将看到该目录。完整的生命周期详情及控制 `/workspace` 和 `/root` 是否跨 Hermes 重启保留的 `container_persistent` 标志，请参阅 [配置 → Docker 后端](../configuration.md#docker-backend)。

### SSH 后端

推荐用于安全场景——Agent 无法修改自身代码：

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
# 为并行 worker 预构建 SIF
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

### 容器资源

为所有容器后端配置 CPU、内存、磁盘和持久化：

```yaml
terminal:
  backend: docker  # 或 singularity, modal, daytona
  container_cpu: 1              # CPU 核心数（默认：1）
  container_memory: 5120        # 内存（MB，默认：5GB）
  container_disk: 51200         # 磁盘（MB，默认：50GB）
  container_persistent: true    # 跨会话持久化文件系统（默认：true）
```

启用 `container_persistent: true` 后，已安装的包、文件和配置将跨会话保留。

### 容器安全

所有容器后端均启用安全加固：

- 只读根文件系统（Docker）
- 丢弃所有 Linux capabilities
- 禁止权限提升
- PID 限制（256 个进程）
- 完整命名空间隔离
- 通过卷挂载实现持久化工作区，而非可写根层

Docker 可通过 `terminal.docker_forward_env` 接受显式的环境变量白名单，但转发的变量对容器内的命令可见，应视为在该会话中已暴露。

## 后台进程管理

启动后台进程并进行管理：

```python
terminal(command="pytest -v tests/", background=true)
# 返回：{"session_id": "proc_abc123", "pid": 12345}

# 然后使用 process 工具进行管理：
process(action="list")       # 显示所有运行中的进程
process(action="poll", session_id="proc_abc123")   # 检查状态
process(action="wait", session_id="proc_abc123")   # 阻塞直到完成
process(action="log", session_id="proc_abc123")    # 完整输出
process(action="kill", session_id="proc_abc123")   # 终止进程
process(action="write", session_id="proc_abc123", data="y")  # 发送输入
```

PTY 模式（`pty=true`）可启用 Codex 和 Claude Code 等交互式 CLI 工具。

## Sudo 支持

如果命令需要 sudo，系统会提示你输入密码（在本次会话内缓存）。也可在 `~/.hermes/.env` 中设置 `SUDO_PASSWORD`。

:::warning
在消息平台上，如果 sudo 失败，输出中会提示将 `SUDO_PASSWORD` 添加到 `~/.hermes/.env`。
:::