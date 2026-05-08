---
sidebar_position: 2
title: "配置"
description: "配置 Hermes Agent — config.yaml、provider、模型、API 密钥等"
---

# 配置

所有设置都存储在 `~/.hermes/` 目录中以便于访问。

## 目录结构

```text
~/.hermes/
├── config.yaml     # 设置（模型、终端、TTS、压缩等）
├── .env            # API 密钥和密钥
├── auth.json       # OAuth provider 凭证（Nous Portal 等）
├── SOUL.md         # 主要 agent 身份（系统提示中的槽位 #1）
├── memories/       # 持久内存（MEMORY.md、USER.md）
├── skills/         # Agent 创建的 skills（通过 skill_manage 工具管理）
├── cron/           # 计划任务
├── sessions/       # Gateway 会话
└── logs/           # 日志（errors.log、gateway.log — 密钥自动编辑）
```

## 管理配置

```bash
hermes config              # 查看当前配置
hermes config edit         # 在编辑器中打开 config.yaml
hermes config set KEY VAL  # 设置特定值
hermes config check        # 检查缺失的选项（更新后）
hermes config migrate      # 交互式添加缺失的选项

# 示例：
hermes config set model anthropic/claude-opus-4
hermes config set terminal.backend docker
hermes config set OPENROUTER_API_KEY sk-or-...  # 保存到 .env
```

:::tip
`hermes config set` 命令自动将值路由到正确的文件——API 密钥保存到 `.env`，其他所有内容保存到 `config.yaml`。
:::

## 配置优先级

设置按此顺序解析（优先级从高到低）：

1. **CLI 参数** — 例如 `hermes chat --model anthropic/claude-sonnet-4`（每次调用的覆盖）
2. **`~/.hermes/config.yaml`** — 所有非密钥设置的主要配置文件
3. **`~/.hermes/.env`** — 环境变量的后备；**密钥必需**（API 密钥、令牌、密码）
4. **内置默认值** — 当没有设置其他任何内容时的硬编码安全默认值

:::info 经验法则
密钥（API 密钥、bot 令牌、密码）放在 `.env`。其他所有内容（模型、终端后端、压缩设置、内存限制、工具集）放在 `config.yaml`。当两者都设置时，`config.yaml` 在非密钥设置上优先。
:::

## 环境变量替换

您可以使用 `${VAR_NAME}` 语法在 `config.yaml` 中引用环境变量：

```yaml
auxiliary:
  vision:
    api_key: ${GOOGLE_API_KEY}
    base_url: ${CUSTOM_VISION_URL}

delegation:
  api_key: ${DELEGATION_KEY}
```

单个值中的多个引用有效：`url: "${HOST}:${PORT}"`。如果引用的变量未设置， 占位符会保持原样（`${UNDEFINED_VAR}` 保持原样）。仅支持 `${VAR}` 语法——裸 `$VAR` 不会展开。

有关 AI provider 设置（OpenRouter、Anthropic、Copilot、自定义端点、自托管 LLM、回退模型等），请参阅 [AI Providers](/docs/integrations/providers)。

### Provider 超时

您可以为 provider 设置 `providers.<id>.request_timeout_seconds` 作为 provider 范围的请求超时，加上 `providers.<id>.models.<model>.timeout_seconds` 用于模型特定覆盖。适用于主要轮次客户端在每个传输上（OpenAI 协议、原生 Anthropic、Anthropic 兼容）、回退链、凭证轮换后重建，以及（对于 OpenAI 协议）每请求超时 kwargs——因此配置的值优先于遗留的 `HERMES_API_TIMEOUT` 环境变量。

您也可以设置 `providers.<id>.stale_timeout_seconds` 用于非流式陈旧调用检测，加上 `providers.<id>.models.<model>.stale_timeout_seconds` 用于模型特定覆盖。这优先于遗留的 `HERMES_API_CALL_STALE_TIMEOUT` 环境变量。

保留这些未设置可保持遗留默认值（`HERMES_API_TIMEOUT=1800`s、`HERMES_API_CALL_STALE_TIMEOUT=300`s、原生 Anthropic 900s）。目前未为 AWS Bedrock 连接（`bedrock_converse` 和 AnthropicBedrock SDK 路径都使用 boto3 及其自己的超时配置）。请参阅 [`cli-config.yaml.example`](https://github.com/NousResearch/hermes-agent/blob/main/cli-config.yaml.example) 中的注释示例。

## 终端后端配置

Hermes 支持七种终端后端。每个都决定 agent 的 shell 命令实际执行的位置——您的本地机器、通过 SSH 的 Docker 容器、远程服务器、Modal 云沙箱（直接或通过 Nous 管理的网关）、Daytona 工作区、Vercel Sandbox 或 Singularity/Apptainer 容器。

```yaml
terminal:
  backend: local    # local | docker | ssh | modal | daytona | vercel_sandbox | singularity
  cwd: "."          # Gateway/cron 工作目录（CLI 始终使用启动目录）
  timeout: 180      # 每个命令的超时秒数
  env_passthrough: []  # 要转发到沙箱执行的 Env var 名称（terminal + execute_code）
  singularity_image: "docker://nikolaik/python-nodejs:python3.11-nodejs20"  # Singularity 后端的容器镜像
  modal_image: "nikolaik/python-nodejs:python3.11-nodejs20"                 # Modal 后端的容器镜像
  daytona_image: "nikolaik/python-nodejs:python3.11-nodejs20"               # Daytona 后端的容器镜像
```

对于 Modal、Daytona 和 Vercel Sandbox 等云沙箱，`container_persistent: true` 意味着 Hermes 会尝试跨沙箱重建保持文件系统状态。它不承诺同一个实时沙箱、PID 空间或后台进程稍后仍在运行。

### 后端概述

| 后端 | 命令运行位置 | 隔离 | 适用于 |
|---------|-------------------|-----------|----------|
| **local** | 直接在您的机器上 | 无 | 开发、个人使用 |
| **docker** | 单个持久 Docker 容器（跨会话、`/new`、子 agent 共享） | 完整（命名空间、cap-drop） | 安全沙箱、CI/CD |
| **ssh** | 通过 SSH 在远程服务器上 | 网络边界 | 远程开发、强大硬件 |
| **modal** | Modal 云沙箱 | 完整（云 VM） | 临时云计算、评估 |
| **daytona** | Daytona 工作区 | 完整（云容器） | 托管云开发环境 |
| **vercel_sandbox** | Vercel Sandbox | 完整（云 microVM） | 带快照备份文件系统持久性的云执行 |
| **singularity** | Singularity/Apptainer 容器 | 命名空间（--containall） | HPC 集群、共享机器 |

### 本地后端

默认值。命令直接在您的机器上运行，无隔离。无需特殊设置。

```yaml
terminal:
  backend: local
```

:::warning
Agent 具有与您的用户账户相同的文件系统访问权限。使用 `hermes tools` 禁用您不想要的工具，或切换到 Docker 进行沙箱化。
:::

### Docker 后端

在具有安全强化（丢弃所有功能、无权限提升、PID 限制）的 Docker 容器内运行命令。

**单个持久容器，不是每个命令一个。** Hermes 在首次使用时启动一个长期存在的容器，并通过 `docker exec` 将每个终端、文件和 `execute_code` 调用路由到同一个容器——跨会话、`/new`、`/reset` 和 `delegate_task` 子 agent——在 Hermes 进程的整个生命周期内。工作目录更改、安装的包和 `/workspace` 中的文件从一个工具调用保留到下一个，就像本地 shell 一样。容器在关闭时停止并删除。请参阅下面的**容器生命周期**了解详情。

```yaml
terminal:
  backend: docker
  docker_image: "nikolaik/python-nodejs:python3.11-nodejs20"
  docker_mount_cwd_to_workspace: false  # 将启动目录挂载到 /workspace
  docker_run_as_host_user: false   # 请参阅下面的"以主机用户身份运行容器"
  docker_forward_env:              # 要转发到容器的 Env 变量
    - "GITHUB_TOKEN"
  docker_volumes:                  # 主机目录挂载
    - "/home/user/projects:/workspace/projects"
    - "/home/user/data:/data:ro"   # :ro 表示只读

  # 资源限制
  container_cpu: 1                 # CPU 核（0 = 无限制）
  container_memory: 5120           # MB（0 = 无限制）
  container_disk: 51200            # MB（需要 XFS+pquota 上的 overlay2）
  container_persistent: true       # 跨会话保持 /workspace 和 /root
```

**要求：**已安装并运行 Docker Desktop 或 Docker Engine。Hermes 探测 `$PATH` 加上常见 macOS 安装位置（`/usr/local/bin/docker`、`/opt/homebrew/bin/docker`、Docker Desktop 应用包）。Podman 开箱即用支持：设置 `HERMES_DOCKER_BINARY=podman`（或完整路径）以在两者都安装时强制使用它。

**容器生命周期：** Hermes 为每个终端和文件工具调用重用单个长期存在的容器（`docker run -d ... sleep 2h`），跨会话、`/new`、`/reset` 和 `delegate_task` 子 agent，在 Hermes 进程的整个生命周期内。命令通过登录 shell 运行 `docker exec`，因此工作目录更改、安装的包和 `/workspace` 中的文件都从一个工具调用保留到下一个。容器在 Hermes 关闭时停止并删除（或者当空闲清理回收它时）。

通过 `delegate_task(tasks=[...])` 并行生成的子 agent 共享这一个容器——并发的 `cd`、env 变更和写入相同路径会冲突。如果子 agent 需要隔离的沙箱，它必须通过 `register_task_env_overrides()` 注册每个任务镜像覆盖，RL 和基准环境（TerminalBench2、HermesSweEnv 等）会为其每个任务 Docker 镜像自动执行此操作。

**安全强化：**
- `--cap-drop ALL`，仅添加回 `DAC_OVERRIDE`、`CHOWN`、`FOWNER`
- `--security-opt no-new-privileges`
- `--pids-limit 256`
- 大小限制的 tmpfs：/tmp (512MB)、/var/tmp (256MB)、/run (64MB)

**凭证转发：** `docker_forward_env` 中列出的 Env 变量首先从您的 shell 环境解析，然后回退到 `~/.hermes/.env`。Skills 也可以声明 `required_environment_variables`，这些会自动合并。

### SSH 后端

通过 SSH 在远程服务器上运行命令。使用 ControlMaster 进行连接重用（5 分钟空闲保活）。默认启用持久 shell——状态（cwd、env 变量）在命令之间保持。

```yaml
terminal:
  backend: ssh
  persistent_shell: true           # 保持长期存在的 bash 会话（默认：true）
```

**必需的环境变量：**

```bash
TERMINAL_SSH_HOST=my-server.example.com
TERMINAL_SSH_USER=ubuntu
```

**可选：**

| 变量 | 默认 | 描述 |
|----------|---------|-------------|
| `TERMINAL_SSH_PORT` | `22` | SSH 端口 |
| `TERMINAL_SSH_KEY` | （系统默认） | SSH 私钥路径 |
| `TERMINAL_SSH_PERSISTENT` | `true` | 启用持久 shell |

**工作原理：**使用 `BatchMode=yes` 和 `StrictHostKeyChecking=accept-new` 在初始化时连接。持久 shell 在远程主机上保持一个 `bash -l` 进程存活，通过临时文件通信。需要 `stdin_data` 或 `sudo` 的命令自动回退到一次性模式。

### Modal 后端

在 [Modal](https://modal.com) 云沙箱中运行命令。每个任务获得一个具有可配置 CPU、内存和磁盘的隔离 VM。文件系统可以跨会话快照/恢复。

```yaml
terminal:
  backend: modal
  container_cpu: 1                 # CPU 核
  container_memory: 5120           # MB（5GB）
  container_disk: 51200            # MB（50GB）
  container_persistent: true       # 快照/恢复文件系统
```

**必需：** 要么 `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET` 环境变量，要么 `~/.modal.toml` 配置文件。

**持久性：** 启用时，沙箱文件系统在清理时快照并在下次会话时恢复。快照在 `~/.hermes/modal_snapshots.json` 中跟踪。这保留文件系统状态，不保留实时进程、PID 空间或后台作业。

**凭证文件：**自动从 `~/.hermes/` 挂载（OAuth 令牌等）并在每个命令之前同步。

### Daytona 后端

在 [Daytona](https://daytona.io) 托管工作区中运行命令。支持停止/恢复以实现持久性。

```yaml
terminal:
  backend: daytona
  container_cpu: 1                 # CPU 核
  container_memory: 5120           # MB → 转换为 GiB
  container_disk: 10240            # MB → 转换为 GiB（最大 10 GiB）
  container_persistent: true       # 停止/恢复而不是删除
```

**必需：** `DAYTONA_API_KEY` 环境变量。

**持久性：** 启用时，沙箱在清理时停止（不是删除）并在下次会话时恢复。沙箱名称遵循模式 `hermes-{task_id}`。

**磁盘限制：** Daytona 强制执行 10 GiB 最大值。高于此的请求会被警告并限制。

### Vercel Sandbox 后端

在 [Vercel Sandbox](https://vercel.com/docs/vercel-sandbox) 云 microVM 中运行命令。Hermes 使用正常的终端和文件工具表面；没有特定于 Vercel 的面向模型的工具。

```yaml
terminal:
  backend: vercel_sandbox
  vercel_runtime: node24          # node24 | node22 | python3.13
  cwd: /vercel/sandbox            # 默认工作区根目录
  container_persistent: true      # 快照/恢复文件系统
  container_disk: 51200           # 仅共享默认值；不支持自定义磁盘
```

**必需安装：** 安装可选的 SDK extra：

```bash
pip install 'hermes-agent[vercel]'
```

**必需认证：** 使用所有三个 `VERCEL_TOKEN`、`VERCEL_PROJECT_ID` 和 `VERCEL_TEAM_ID` 配置访问令牌认证。这是 Render、Railway、Docker 和类似主机的部署和正常长期运行 Hermes 进程的推荐设置。

对于一次性本地开发，Hermes 也接受短期 Vercel OIDC 令牌：

```bash
VERCEL_OIDC_TOKEN="$(vc project token <project-name>)" hermes chat
```

从链接的 Vercel 项目目录，您可以省略项目名称：

```bash
VERCEL_OIDC_TOKEN="$(vc project token)" hermes chat
```

OIDC 令牌是短期令牌，不应用作文档化的部署路径。

**运行时：** `terminal.vercel_runtime` 支持 `node24`、`node22` 和 `python3.13`。如果未设置，Hermes 默认为 `node24`。

**持久性：** 当 `container_persistent: true` 时，Hermes 在清理期间快照沙箱文件系统，并为同一任务恢复后续沙箱。快照内容可以包括 Hermes 同步的凭证、skills 和复制到沙箱中的缓存文件。这仅保留文件系统状态；它不保留实时沙箱身份、PID 空间、shell 状态或运行中的后台进程。

**后台命令：** `terminal(background=true)` 使用 Hermes 的通用非本地后台进程流程。当沙箱存活时，您可以通过正常进程工具生成、轮询、等待、查看日志和终止进程。Hermes 不在清理或重启后提供原生 Vercel 分离进程恢复。

**磁盘大小：** Vercel Sandbox 目前不支持 Hermes 的 `container_disk` 资源旋钮。保留 `container_disk` 未设置或为共享默认值 `51200`；非默认值会失败诊断和后端创建，而不是静默忽略。

### Singularity/Apptainer 后端

在 [Singularity/Apptainer](https://apptainer.org) 容器中运行命令。为 Docker 不可用的 HPC 集群和共享机器设计。

```yaml
terminal:
  backend: singularity
  singularity_image: "docker://nikolaik/python-nodejs:python3.11-nodejs20"
  container_cpu: 1                 # CPU 核
  container_memory: 5120           # MB
  container_persistent: true       # 可写覆盖跨会话保持
```

**要求：** `apptainer` 或 `singularity` 二进制文件在 `$PATH` 中。

**镜像处理：** Docker URL（`docker://...`）自动转换为 SIF 文件并缓存。现有的 `.sif` 文件直接使用。

**临时目录：** 按顺序解析：`TERMINAL_SCRATCH_DIR` → `TERMINAL_SANDBOX_DIR/singularity` → `/scratch/$USER/hermes-agent`（HPC 约定）→ `~/.hermes/sandboxes/singularity`。

**隔离：** 使用 `--containall --no-home` 进行完整命名空间隔离，而不挂载主机主目录。

### 常见终端后端问题

如果终端命令立即失败或终端工具被报告为禁用：

- **Local** — 无特殊要求。开始时最安全的默认设置。
- **Docker** — 运行 `docker version` 验证 Docker 工作。如果失败，修复 Docker 或 `hermes config set terminal.backend local`。
- **SSH** — 必须设置 `TERMINAL_SSH_HOST` 和 `TERMINAL_SSH_USER`。如果任一缺失，Hermes 记录明确错误。
- **Modal** — 需要 `MODAL_TOKEN_ID` env var 或 `~/.modal.toml`。运行 `hermes doctor` 检查。
- **Daytona** — 需要 `DAYTONA_API_KEY`。Daytona SDK 处理服务器 URL 配置。
- **Singularity** — 需要 `$PATH` 中的 `apptainer` 或 `singularity`。常见于 HPC 集群。

如有疑问，将 `terminal.backend` 设置回 `local` 并首先验证命令在那里运行。

### 远程到主机文件同步在拆卸时

对于 **SSH**、**Modal** 和 **Daytona** 后端（agent 工作树所在机器与运行 Hermes 的主机不同时），Hermes 跟踪 agent 在远程沙箱中触及的文件，并在会话拆卸/沙箱清理时**将修改的文件同步回主机**，保存在 `~/.hermes/cache/remote-syncs/<session-id>/` 下。

- 触发时机：会话关闭、`/new`、`/reset`、gateway 消息超时、使用远程后端的子 agent 完成后。
- 覆盖 agent 修改的整个树，而不仅仅是它明确打开的文件。添加、编辑和删除都被捕获。
- 当您去查找时，远程沙箱可能已被拆除；本地的 `~/.hermes/cache/remote-syncs/…` 副本是 agent 更改的权威记录。
- 大型二进制输出（模型检查点、原始数据集）按大小上限——同步跳过超过 `file_sync_max_mb`（默认 `100`）的文件。如果期望更大的产物回来，请增加此值。

```yaml
terminal:
  file_sync_max_mb: 100     # 默认——每个同步最多 100 MB
  file_sync_enabled: true   # 默认——设置为 false 完全跳过同步
```

这是您从会话结束后被销毁的临时云沙箱恢复结果的方式，而无需告诉 agent 显式 `scp` 或 `modal volume put` 每个产物。

### Docker 卷挂载

使用 Docker 后端时，`docker_volumes` 让您与容器共享主机目录。每个条目使用标准 Docker `-v` 语法：`host_path:container_path[:options]`。

```yaml
terminal:
  backend: docker
  docker_volumes:
    - "/home/user/projects:/workspace/projects"   # 读写（默认）
    - "/home/user/datasets:/data:ro"              # 只读
    - "/home/user/.hermes/cache/documents:/output" # Gateway 可见的导出
```

这对于以下情况很有用：
- **向 agent 提供文件**（数据集、配置、参考代码）
- **从 agent 接收文件**（生成的代码、报告、导出）
- **共享工作区**，您和 agent 都访问相同文件

如果您使用消息传递 gateway 并希望 agent 通过 `MEDIA:/...` 发送生成的文件，请使用专用主机可见导出挂载，例如 `/home/user/.hermes/cache/documents:/output`。

- 在 Docker 中将文件写入 `/output/...`
- 在 `MEDIA:` 中发出**主机路径**，例如：
  `MEDIA:/home/user/.hermes/cache/documents/report.txt`
- 除非该确切路径在主机上也存在于 gateway 进程，否则**不要**发出 `/workspace/...` 或 `/output/...`

:::warning
YAML 重复键静默覆盖较早的。如果已有 `docker_volumes:` 块，请将新挂载合并到同一列表中，而不是稍后在文件中添加另一个 `docker_volumes:` 键。
:::

也可以通过环境变量设置：`TERMINAL_DOCKER_VOLUMES='["/host:/container"]'`（JSON 数组）。

### Docker 凭证转发

默认情况下，Docker 终端会话不继承任意主机凭证。如果您需要在容器中使用特定令牌，请将其添加到 `terminal.docker_forward_env`。

```yaml
terminal:
  backend: docker
  docker_forward_env:
    - "GITHUB_TOKEN"
    - "NPM_TOKEN"
```

Hermes 首先从当前 shell 解析每个列出的变量，然后回退到 `~/.hermes/.env`（如果通过 `hermes config set` 保存）。

:::warning
`docker_forward_env` 中列出的任何内容对容器内运行的命令可见。仅转发您愿意在终端会话中暴露的凭证。
:::

### 可选：以主机用户身份运行容器

默认情况下 Docker 容器以 `root`（UID 0）运行。在 `/workspace` 或其他绑定挂载中创建的文件最终在主机上由 root 拥有，因此会话后您必须 `sudo chown` 才能从主机编辑器编辑它们。`terminal.docker_run_as_host_user` 标志修复了这个问题：

```yaml
terminal:
  backend: docker
  docker_run_as_host_user: true   # 默认：false
```

启用时，Hermes 将 `--user $(id -u):$(id -g)` 附加到 `docker run` 命令，因此写入绑定挂载目录（`/workspace`、`/root`、`docker_volumes` 中的任何内容）的文件由您的主机用户而不是 root 拥有。权衡：容器无法 `apt install` 或写入 root 拥有的路径如 `/root/.npm`——如果需要两者，请使用其 `HOME` 由非 root 用户拥有的基础镜像（或者在镜像构建时添加所需工具）。

保留 `false`（默认）以实现向后兼容的行为。当您的工作流程主要是"编辑挂载的主机文件"且厌倦了 `sudo chown -R` 时，将其打开。

### 可选：将启动目录挂载到 `/workspace`

Docker 沙箱默认保持隔离。Hermes **不会**将当前主机工作目录传递到容器中，除非您明确选择加入。

在 `config.yaml` 中启用：

```yaml
terminal:
  backend: docker
  docker_mount_cwd_to_workspace: true
```

启用时：
- 如果您从 `~/projects/my-app` 启动 Hermes，该主机目录绑定挂载到 `/workspace`
- Docker 后端在 `/workspace` 中启动
- 文件工具和终端命令都看到相同的挂载项目

禁用时，`/workspace` 保持沙箱拥有，除非您通过 `docker_volumes` 显式挂载某些内容。

安全权衡：
- `false` 保留沙箱边界
- `true` 赋予沙箱直接访问您启动 Hermes 的目录的权限

仅在您有意让容器处理实时主机文件时才使用选择加入。

### 持久 Shell

默认情况下，每个终端命令在各自的子进程中运行——工作目录、环境变量和 shell 变量在命令之间重置。当**启用持久 shell**时，单个长期存在的 bash 进程在 `execute()` 调用之间保持存活，以便状态在命令之间保持。

这对于 **SSH 后端**最有用，它还消除了每个命令的连接开销。持久 shell **默认为 SSH 启用**，为本地后端禁用。

```yaml
terminal:
  persistent_shell: true   # 默认——为 SSH 启用持久 shell
```

要禁用：

```bash
hermes config set terminal.persistent_shell false
```

**跨命令保持的内容：**
- 工作目录（`cd /tmp` 为下一个命令保留）
- 导出的环境变量（`export FOO=bar`）
- Shell 变量（`MY_VAR=hello`）

**优先级：**

| 级别 | 变量 | 默认 |
|-------|----------|---------|
| Config | `terminal.persistent_shell` | `true` |
| SSH 覆盖 | `TERMINAL_SSH_PERSISTENT` | 遵循 config |
| Local 覆盖 | `TERMINAL_LOCAL_PERSISTENT` | `false` |

每个后端的环境变量优先级最高。如果您也想在本地后端启用持久 shell：

```bash
export TERMINAL_LOCAL_PERSISTENT=true
```

:::note
需要 `stdin_data` 或 sudo 的命令自动回退到一次性模式，因为持久 shell 的 stdin 已被 IPC 协议占用。
:::

请参阅 [代码执行](features/code-execution.md) 和 [README 的终端部分](features/tools.md) 了解每个后端的详情。

## Skill 设置

Skills 可以通过其 SKILL.md frontmatter 声明自己的配置设置。这些是非密钥值（路径、首选项、域设置），存储在 `config.yaml` 的 `skills.config` 命名空间下。

```yaml
skills:
  config:
    myplugin:
      path: ~/myplugin-data   # 示例——每个 skill 定义自己的键
```

**skill 设置工作原理：**

- `hermes config migrate` 扫描所有启用的 skills，找到未配置的设置，并提示您
- `hermes config show` 在"Skill 设置"下显示所有 skill 设置，并标注它们所属的 skill
- 当 skill 加载时，其解析的配置值会自动注入 skill 上下文

**手动设置值：**

```bash
hermes config set skills.config.myplugin.path ~/myplugin-data
```

有关在您自己的 skills 中声明配置设置的详情，请参阅 [创建 Skills — 配置设置](/docs/developer-guide/creating-skills#config-settings-configyaml)。

### Agent 创建的 skill 写入防护

当 agent 使用 `skill_manage` 创建、编辑、补丁或删除 skill 时，Hermes 可以选择扫描新/更新内容中的危险关键字模式（凭证收集、明显的提示注入、泄露指示）。扫描器**默认关闭**——合法触及 `~/.ssh/` 或提及 `$OPENAI_API_KEY` 的真实 agent 工作流程经常触发启发式方法。如果您希望扫描器在 agent 的 skill 写入落地之前提示您，请将其重新打开：

```yaml
skills:
  guard_agent_created: true   # 默认：false
```

开启时，任何标记的 `skill_manage` 写入都会作为带扫描器理由的审批提示浮出水面。接受的写入落地；拒绝的写入向 agent 返回解释性错误。

## 内存配置

```yaml
memory:
  memory_enabled: true
  user_profile_enabled: true
  memory_char_limit: 2200   # ~800 tokens
  user_char_limit: 1375     # ~500 tokens
```

## 文件读取安全

控制单个 `read_file` 调用可以返回多少内容。超过限制的读取会被拒绝，并向 agent 返回错误，告诉它使用 `offset` 和 `limit` 获取更小的范围。这防止单次读取 minified JS 包或大型数据文件淹没上下文窗口。

```yaml
file_read_max_chars: 100000  # 默认——~25-35K tokens
```

如果您使用具有大上下文窗口的模型并经常读取大文件，请增加它。为小上下文模型降低它以保持读取效率：

```yaml
# 大上下文模型（200K+）
file_read_max_chars: 200000

# 小型本地模型（16K 上下文）
file_read_max_chars: 30000
```

Agent 还会自动去重文件读取——如果同一文件区域被读取两次且文件未更改，则返回轻量级存根而不是重新发送内容。这在上下文压缩时重置，以便 agent 在内容被摘要掉后重新读取文件。

## 工具输出截断限制

三个相关的上限控制工具在 Hermes 截断之前可以返回多少原始输出：

```yaml
tool_output:
  max_bytes: 50000        # 终端输出上限（字符）
  max_lines: 2000         # read_file 分页上限
  max_line_length: 2000   # read_file 行号视图中的每行上限
```

- **`max_bytes`** — 当 `terminal` 命令产生超过此字符数的组合 stdout/stderr 时，Hermes 保留前 40% 和后 60%，并在它们之间插入 `[OUTPUT TRUNCATED]` 通知。默认 `50000`（≈12-15K tokens 跨典型 tokenizer）。
- **`max_lines`** — 单个 `read_file` 调用的 `limit` 参数的上限。高于此的请求被限制，因此单次读取无法淹没上下文窗口。默认 `2000`。
- **`max_line_length`** — 当 `read_file` 发出行号视图时应用的每行上限。长于此的行被截断为这么多字符，后跟 `... [truncated]`。默认 `2000`。

在具有大上下文窗口且每次调用可以负担更多原始输出的模型上增加限制。为小上下文模型降低它们以保持工具结果紧凑：

```yaml
# 大上下文模型（200K+）
tool_output:
  max_bytes: 150000
  max_lines: 5000

# 小型本地模型（16K 上下文）
tool_output:
  max_bytes: 20000
  max_lines: 500
```

## 全局工具集禁用

要在一个地方抑制 CLI 和每个 gateway 平台中的特定工具集，请在 `agent.disabled_toolsets` 下列出它们的名称：

```yaml
agent:
  disabled_toolsets:
    - memory       # 隐藏内存工具 + MEMORY_GUIDANCE 注入
    - web          # 任何地方都没有 web_search / web_extract
```

这适用于**在**每平台工具配置之后（`platform_toolsets` 由 `hermes tools` 写入），因此在此列出的工具集总是被移除——即使平台的保存配置仍列出它。当您想要一个单一的开关来"在所有地方关闭 X"时使用此功能，而不是编辑 `hermes tools` UI 中的 15+ 平台行。

保留列表为空或省略键是无操作。

## Git Worktree 隔离

为在同一个仓库上并行运行多个 agent 启用隔离的 git worktree：

```yaml
worktree: true    # 始终创建 worktree（与 hermes -w 相同）
# worktree: false # 默认——仅在传递 -w 标志时
```

启用时，每个 CLI 会话在 `.worktrees/` 下创建一个带有自己分支的新 worktree。Agent 可以编辑文件、提交、推送和创建 PR，而不会相互干扰。干净的 worktree 在退出时删除；脏的保留以供手动恢复。

您还可以通过仓库根目录中的 `.worktreeinclude` 列出要复制到 worktree 的 gitignored 文件：

```
# .worktreeinclude
.env
.venv/
node_modules/
```

## 上下文压缩

Hermes 自动压缩长对话以保持在模型上下文窗口内。压缩摘要器是一个单独的 LLM 调用——您可以将其指向任何 provider 或端点。

所有压缩设置都在 `config.yaml` 中（无环境变量）。

### 完整参考

```yaml
compression:
  enabled: true                                     # 切换压缩开/关
  threshold: 0.50                                   # 在此上下文限制百分比时压缩
  target_ratio: 0.20                                # 保留为最近尾部的阈值分数
  protect_last_n: 20                                # 保持未压缩的最小最近消息数
  hygiene_hard_message_limit: 400                   # Gateway 安全阀——见下文

# 摘要模型/provider 在 auxiliary 下配置：
auxiliary:
  compression:
    model: "google/gemini-3-flash-preview"          # 用于摘要的模型
    provider: "auto"                                # Provider: "auto"、"openrouter"、"nous"、"codex"、"main" 等
    base_url: null                                  # 自定义 OpenAI 兼容端点（覆盖 provider）
```

:::info 遗留配置迁移
带有 `compression.summary_model`、`compression.summary_provider` 和 `compression.summary_base_url` 的旧配置在首次加载时自动迁移到 `auxiliary.compression.*`（配置版本 17）。无需手动操作。
:::

`hygiene_hard_message_limit` 是仅 gateway 的**压缩前安全阀**。带有数千条消息的失控会话可能在正常百分比上下文阈值触发之前达到模型上下文限制；当消息计数超过此上限时，Hermes 会强制压缩，无论 token 使用情况如何。默认 `400`——在非常长的会话正常的平台上增加它，在需要更积极压缩时降低它。编辑运行中 gateway 上的此值会在下一条消息时生效（见下文）。

:::tip Gateway 热重载压缩和上下文长度
在最近的版本中，在运行中 gateway 的 `config.yaml` 中编辑 `model.context_length` 或任何 `compression.*` 键会在下一条消息时生效——无需 gateway 重启、`/reset` 或会话轮换。缓存的 agent 签名包含这些键，因此 gateway 在看到更改时透明地重建 agent。API 密钥和工具/skill 配置仍需要通常的重新加载路径。
:::

### 常见设置

**默认（自动检测）——无需配置：**
```yaml
compression:
  enabled: true
  threshold: 0.50
```
使用您的主要 provider 和主要模型。如果您希望压缩在比主要聊天模型更便宜的模型上执行，请按任务覆盖（例如 `auxiliary.compression.provider: openrouter` + `model: google/gemini-2.5-flash`）。

**强制特定 provider**（基于 OAuth 或 API 密钥）：
```yaml
auxiliary:
  compression:
    provider: nous
    model: gemini-3-flash
```
适用于任何 provider：`nous`、`openrouter`、`codex`、`anthropic`、`main` 等。

**自定义端点**（自托管、Ollama、zai、DeepSeek 等）：
```yaml
auxiliary:
  compression:
    model: glm-4.7
    base_url: https://api.z.ai/api/coding/paas/v4
```
指向自定义 OpenAI 兼容端点。使用 `OPENAI_API_KEY` 进行身份验证。

### 三个旋钮如何交互

| `auxiliary.compression.provider` | `auxiliary.compression.base_url` | 结果 |
|---------------------|---------------------|--------|
| `auto`（默认） | 未设置 | 自动检测最佳可用 provider |
| `nous` / `openrouter` 等 | 未设置 | 强制该 provider，使用其 auth |
| 任意 | 设置 | 直接使用自定义端点（provider 被忽略） |

:::warning 摘要模型上下文长度要求
摘要模型的上下文窗口**必须**至少与主要 agent 模型的相同。压缩器将对话的完整中间部分发送到摘要模型——如果该模型的上下文窗口小于主要模型的，摘要调用将失败并出现上下文长度错误。当这种情况发生时，中间轮次**在没有摘要的情况下被丢弃**，静默丢失对话上下文。如果您覆盖模型，请验证其上下文长度满足或超过主要模型的。
:::

## 上下文引擎

上下文引擎控制对话在接近模型 token 限制时的管理方式。内置的 `compressor` 引擎使用有损摘要（请参阅 [上下文压缩](/docs/developer-guide/context-compression-and-caching)）。插件引擎可以用替代策略替换它。

```yaml
context:
  engine: "compressor"    # 默认——内置有损摘要
```

要使用插件引擎（例如，用于无损上下文管理的 LCM）：

```yaml
context:
  engine: "lcm"          # 必须匹配插件的名称
```

插件引擎**永远不会自动激活**——您必须显式将 `context.engine` 设置为插件名称。可用引擎可以通过 `hermes plugins` → Provider Plugins → Context Engine 浏览和选择。

请参阅 [内存 Providers](/docs/user-guide/features/memory-providers) 了解内存插件的类似单一选择系统。

## 迭代预算压力

当 agent 处理具有许多工具调用的复杂任务时，它可能会在意识到自己快用完之前烧穿其迭代预算（默认：90 轮）。预算压力在接近限制时自动警告模型：

| 阈值 | 级别 | 模型看到的内容 |
|-----------|-------|---------------------|
| **70%** | Caution | `[BUDGET: 63/90. 27 iterations left. Start consolidating.]` |
| **90%** | Warning | `[BUDGET WARNING: 81/90. Only 9 left. Respond NOW.]` |

警告被注入到最后一个工具结果的 JSON 中（作为 `_budget_warning` 字段），而不是作为单独的消息——这保留提示缓存且不会破坏对话结构。

```yaml
agent:
  max_turns: 90                # 每次对话轮次的最大迭代次数（默认：90）
  api_max_retries: 2           # 在回退 provider 参与之前每个 provider 的重试次数（默认：2）
```

预算压力默认启用。Agent 自然地作为工具结果的一部分看到警告，鼓励它在迭代用完之前巩固其工作并交付响应。

当迭代预算完全耗尽时，CLI 向用户显示通知：`⚠ Iteration budget reached (90/90) — response may be incomplete`。如果预算在工作期间用完，agent 会在停止之前生成已完成工作的摘要。

`agent.api_max_retries` 控制 Hermes 在回退 provider 切换参与之前对瞬态错误（速率限制、连接断开、5xx）对 provider API 调用重试的次数。默认是 `2`——总共三次尝试，与 OpenAI SDK 默认值匹配。如果您已配置 [回退 provider](/docs/user-guide/features/fallback-providers) 并希望更快地故障转移，请将其降低到 `0`，以便您主要的第一个瞬态错误立即交给回退，而不是在不稳定端点上重试。

### API 超时

Hermes 对流式传输有单独的超时层，加上非流式调用的陈旧检测器。陈旧检测器仅在您将它们保留在其隐含默认值时才为本地 provider 自动调整。

| 超时 | 默认 | 本地 provider | Config / env |
|---------|---------|----------------|--------------|
| Socket 读取超时 | 120s | 自动提高到 1800s | `HERMES_STREAM_READ_TIMEOUT` |
| 陈旧流检测 | 180s | 自动禁用 | `HERMES_STREAM_STALE_TIMEOUT` |
| 陈旧非流检测 | 300s | 隐含时自动禁用 | `providers.<id>.stale_timeout_seconds` 或 `HERMES_API_CALL_STALE_TIMEOUT` |
| API 调用（非流式） | 1800s | 不变 | `providers.<id>.request_timeout_seconds` / `timeout_seconds` 或 `HERMES_API_TIMEOUT` |

**Socket 读取超时**控制 httpx 等待 provider 下一块数据的时长。本地 LLM 在生成第一个 token 之前可能需要数分钟进行大上下文预填充，因此当检测到本地端点时，Hermes 将其提高到 30 分钟。如果您显式设置 `HERMES_STREAM_READ_TIMEOUT`，则无论端点检测如何，该值始终被使用。

**陈旧流检测**终止收到 SSE keep-alive ping 但没有实际内容的连接。这为本地 provider 完全禁用，因为它们在预填充期间不发送 keep-alive ping。

**陈旧非流检测**终止长时间没有产生响应的非流式调用。默认情况下，Hermes 在本地端点上禁用此功能，以避免在长时间预填充期间出现误报。如果您显式设置 `providers.<id>.stale_timeout_seconds`、`providers.<id>.models.<model>.stale_timeout_seconds` 或 `HERMES_API_CALL_STALE_TIMEOUT`，则即使在本地端点上也会遵守该显式值。

## 上下文压力警告

与迭代预算压力不同，上下文压力跟踪对话接近**压缩阈值**的程度——即上下文压缩触发以摘要较旧消息的时间点。这有助于您和 agent 了解对话何时变长。

| 进度 | 级别 | 发生什么 |
|----------|-------|-------------|
| **≥ 60%** 到阈值 | Info | CLI 显示青色进度条；gateway 发送信息通知 |
| **≥ 85%** 到阈值 | Warning | CLI 显示粗体黄色条；gateway 警告压缩即将来临 |

在 CLI 中，上下文压力显示为工具输出反馈中的进度条：

```
  ◐ context ████████████░░░░░░░░ 62% to compaction  48k threshold (50%) · approaching compaction
```

在消息平台上，发送纯文本通知：

```
◐ Context: ████████████░░░░░░░░ 62% to compaction (threshold: 50% of window).
```

如果自动压缩被禁用，警告会告诉您上下文可能被截断而不是。

上下文压力是自动的——无需配置。它仅作为面向用户的通知触发，不会修改消息流或注入任何内容到模型上下文中。

## 凭证池策略

当您有同一 provider 的多个 API 密钥或 OAuth 令牌时，配置轮换策略：

```yaml
credential_pool_strategies:
  openrouter: round_robin    # 均匀循环使用密钥
  anthropic: least_used      # 始终选择使用最少的密钥
```

选项：`fill_first`（默认）、`round_robin`、`least_used`、`random`。请参阅 [凭证池](/docs/user-guide/features/credential-pools) 获取完整文档。

## 辅助模型

Hermes 使用"辅助"模型处理辅助任务，如图像分析、网页摘要、浏览器截图分析、会话标题生成和上下文压缩。默认（`auxiliary.*.provider: "auto"`），Hermes 将每个辅助任务路由到您的**主要聊天模型**——您在 `hermes model` 中选择的相同 provider/模型。您无需配置任何内容即可开始，但请注意，在昂贵的推理模型（Opus、MiniMax M2.7 等）上，辅助任务会增加有意义的成本。如果您希望无论主要模型如何都有廉价快速的辅助任务，请显式设置 `auxiliary.<task>.provider` 和 `auxiliary.<task>.model`（例如，为视觉和网页提取使用 OpenRouter 上的 Gemini Flash）。

:::note 为什么"auto"使用我的主要模型
较早的版本将聚合用户（OpenRouter、Nous Portal）拆分到廉价的 provider 端默认模型。这令人惊讶——为聚合订阅付费的用户会看到不同的模型处理他们的辅助流量。`auto` 现在为所有人使用主要模型，而 `config.yaml` 中的按任务覆盖仍然优先（请参阅下面的 [完整辅助配置参考](#full-auxiliary-config-reference)）。
:::

### 交互式配置辅助模型

不要手动编辑 YAML，运行 `hermes model` 并从菜单中选择 **"Configure auxiliary models"**。您将获得交互式按任务选择器：

```
$ hermes model
→ Configure auxiliary models

[ ] vision               currently: auto / main model
[ ] web_extract          currently: auto / main model
[ ] session_search       currently: openrouter / google/gemini-2.5-flash
[ ] title_generation     currently: openrouter / google/gemini-3-flash-preview
[ ] compression          currently: auto / main model
[ ] approval             currently: auto / main model
[ ] triage_specifier     currently: auto / main model
```

选择一个任务，选择 provider（OAuth 流程打开浏览器；API 密钥 provider 提示），选择模型。更改持久化到 `config.yaml` 中的 `auxiliary.<task>.*`。与主要模型选择器相同的机制——无需学习额外语法。

### 视频教程

<div style={{position: 'relative', width: '100%', aspectRatio: '16 / 9', marginBottom: '1.5rem'}}>
  <iframe
    src="https://www.youtube.com/embed/NoF-YajElIM"
    title="Hermes Agent — Auxiliary Models Tutorial"
    style={{position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', border: 0}}
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowFullScreen
  />
</div>

### 通用配置模式

Hermes 中的每个模型槽位——辅助任务、压缩、回退——使用相同的三个旋钮：

| 键 | 作用 | 默认 |
|-----|-------------|---------|
| `provider` | 使用哪个 provider 进行 auth 和路由 | `"auto"` |
| `model` | 请求哪个模型 | provider 的默认 |
| `base_url` | 自定义 OpenAI 兼容端点（覆盖 provider） | 未设置 |

当设置 `base_url` 时，Hermes 忽略 provider 并直接调用该端点（使用 `api_key` 或 `OPENAI_API_KEY` 进行 auth）。当仅设置 `provider` 时，Hermes 使用该 provider 的内置 auth 和 base URL。

辅助任务的可用 provider：`auto`、`main`，加上 [provider 注册表](/docs/reference/environment-variables) 中的任何 provider——`openrouter`、`nous`、`openai-codex`、`copilot`、`copilot-acp`、`anthropic`、`gemini`、`google-gemini-cli`、`qwen-oauth`、`zai`、`kimi-coding`、`kimi-coding-cn`、`minimax`、`minimax-cn`、`minimax-oauth`、`deepseek`、`nvidia`、`xai`、`ollama-cloud`、`alibaba`、`bedrock`、`huggingface`、`arcee`、`xiaomi`、`kilocode`、`opencode-zen`、`opencode-go`、`ai-gateway`、`azure-foundry`——或来自您的 `custom_providers` 列表的任何命名自定义 provider（例如 `provider: "beans"`）。

:::tip MiniMax OAuth
`minimax-oauth` 通过浏览器 OAuth 登录（无需 API 密钥）。运行 `hermes model` 并选择 **MiniMax (OAuth)** 进行身份验证。辅助任务自动使用 `MiniMax-M2.7-highspeed`。请参阅 [MiniMax OAuth 指南](../guides/minimax-oauth.md)。
:::

:::warning `"main"` 仅适用于辅助任务
`"main"` provider 选项表示"使用我的主要 agent 使用的任何 provider"——它仅在 `auxiliary:`、`compression:` 和 `fallback_model:` 配置内有效。它**不是**您顶级 `model.provider` 设置的有效值。如果您使用自定义 OpenAI 兼容端点，请在 `model:` 部分设置 `provider: custom`。请参阅 [AI Providers](/docs/integrations/providers) 获取所有主要模型 provider 选项。
:::

### 完整辅助配置参考

```yaml
auxiliary:
  # 图像分析（vision_analyze 工具 + 浏览器截图）
  vision:
    provider: "auto"           # "auto"、"openrouter"、"nous"、"codex"、"main" 等
    model: ""                  # 例如 "openai/gpt-4o"、"google/gemini-2.5-flash"
    base_url: ""               # 自定义 OpenAI 兼容端点（覆盖 provider）
    api_key: ""                # base_url 的 API 密钥（回退到 OPENAI_API_KEY）
    timeout: 120               # 秒——LLM API 调用超时；视觉有效载荷需要慷慨的超时
    download_timeout: 30       # 秒——图像 HTTP 下载；为慢速连接增加

  # 网页摘要 + 浏览器页面文本提取
  web_extract:
    provider: "auto"
    model: ""                  # 例如 "google/gemini-2.5-flash"
    base_url: ""
    api_key: ""
    timeout: 360               # 秒（6 分钟）——每次 LLM 摘要尝试

  # 危险命令审批分类器
  approval:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30                # 秒

  # 上下文压缩超时（与 compression.* 配置分开）
  compression:
    timeout: 120               # 秒——压缩摘要长对话，需要更多时间

  # 会话搜索——摘要过去的会话匹配
  session_search:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30
    max_concurrency: 3       # 限制并行摘要以减少请求突发 429s
    extra_body: {}           # Provider 特定的 OpenAI 兼容请求字段

  # Skills hub — skill 匹配和搜索
  skills_hub:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30

  # MCP 工具调度
  mcp:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30

  # Kanban 分诊指定符——`hermes kanban specify <id>`（或
  # 仪表板上 Triage 列卡片的 ✨ Specify 按钮）使用此
  # 槽位将一行规范展开为具体规范并将任务提升到 `todo`。
  # 廉价快速模型在这里效果很好；规范展开很短，不需要推理深度。
  triage_specifier:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 120
```

:::tip
每个辅助任务都有可配置的 `timeout`（秒）。默认值：视觉 120s、网页提取 360s、审批 30s、压缩 120s。如果您将慢速本地模型用于辅助任务，请增加这些。如果您的连接较慢或使用自托管图像服务器，视觉还有一个单独的 `download_timeout`（默认 30s）。
:::

:::info
上下文压缩有自己的 `compression:` 块用于阈值和一个 `auxiliary.compression:` 块用于模型/provider 设置——请参阅上面的 [上下文压缩](#context-compression)。回退模型使用 `fallback_model:` 块——请参阅 [回退模型](/docs/integrations/providers#fallback-model)。所有三个都遵循相同的 provider/model/base_url 模式。
:::

### 会话搜索调优

如果您将推理密集型模型用于 `auxiliary.session_search`，Hermes 现在提供两个内置控件：

- `auxiliary.session_search.max_concurrency`：限制 Hermes 同时摘要多少匹配会话
- `auxiliary.session_search.extra_body`：在摘要调用上转发 provider 特定的 OpenAI 兼容请求字段

示例：

```yaml
auxiliary:
  session_search:
    provider: "main"
    model: "glm-4.5-air"
    timeout: 60
    max_concurrency: 2
    extra_body:
      enable_thinking: false
```

当您的 provider 限制请求突发时，使用 `max_concurrency` 让 `session_search` 交换一些并行性以获得稳定性。

仅当您的 provider 记录您希望 Hermes 为该任务传递的 OpenAI 兼容请求体字段时，才使用 `extra_body`。Hermes 按原样转发对象。

:::warning
`extra_body` 仅在您的 provider 实际支持您发送的字段时有效。如果 provider 不公开原生 OpenAI 兼容推理关闭标志，Hermes 无法为其综合一个。
:::

### 更改视觉模型

要使用 GPT-4o 而不是 Gemini Flash 进行图像分析：

```yaml
auxiliary:
  vision:
    model: "openai/gpt-4o"
```

或通过环境变量（在 `~/.hermes/.env` 中）：

```bash
AUXILIARY_VISION_MODEL=openai/gpt-4o
```

### Provider 选项

这些选项适用于**辅助任务配置**（`auxiliary:`、`compression:`、`fallback_model:`），不适用于您的主要 `model.provider` 设置。

| Provider | 描述 | 要求 |
|----------|-------------|-------------|
| `"auto"` | 最佳可用（默认）。视觉尝试 OpenRouter → Nous → Codex。 | — |
| `"openrouter"` | 强制 OpenRouter——路由到任何模型（Gemini、GPT-4o、Claude 等） | `OPENROUTER_API_KEY` |
| `"nous"` | 强制 Nous Portal | `hermes auth` |
| `"codex"` | 强制 Codex OAuth（ChatGPT 账户）。支持视觉（gpt-5.3-codex）。 | `hermes model` → Codex |
| `"minimax-oauth"` | 强制 MiniMax OAuth（浏览器登录，无需 API 密钥）。辅助任务使用 MiniMax-M2.7-highspeed。 | `hermes model` → MiniMax (OAuth) |
| `"main"` | 使用您的活动自定义/主要端点。这可以来自 `OPENAI_BASE_URL` + `OPENAI_API_KEY`，或来自通过 `hermes model` / `config.yaml` 保存的自定义端点。适用于 OpenAI、本地模型或任何 OpenAI 兼容 API。**仅辅助任务——对 `model.provider` 无效。** | 自定义端点凭证 + base URL |

来自主要 provider 目录的直接 API 密钥 provider 也可以在这里工作，当您希望辅助任务绕过默认路由器时。`gmi` 在配置 `GMI_API_KEY` 后有效：

```yaml
auxiliary:
  compression:
    provider: "gmi"
    model: "anthropic/claude-opus-4.6"
```

对于 GMI 辅助路由，使用 GMI 的 `/v1/models` 端点返回的确切模型 ID。

### 常见设置

**使用直接自定义端点**（比 `provider: "main"` 更清晰，用于本地/自托管 API）：
```yaml
auxiliary:
  vision:
    base_url: "http://localhost:1234/v1"
    api_key: "local-key"
    model: "qwen2.5-vl"
```

`base_url` 优先于 `provider`，因此这是将辅助任务路由到特定端点的最明确方式。对于直接端点覆盖，Hermes 使用配置的 `api_key` 或回退到 `OPENAI_API_KEY`；它不会为该自定义端点重用 `OPENROUTER_API_KEY`。

**使用 OpenAI API 密钥进行视觉：**
```yaml
# 在 ~/.hermes/.env 中：
# OPENAI_BASE_URL=https://api.openai.com/v1
# OPENAI_API_KEY=sk-...

auxiliary:
  vision:
    provider: "main"
    model: "gpt-4o"       # 或更便宜的 "gpt-4o-mini"
```

**使用 OpenRouter 进行视觉**（路由到任何模型）：
```yaml
auxiliary:
  vision:
    provider: "openrouter"
    model: "openai/gpt-4o"      # 或 "google/gemini-2.5-flash" 等
```

**使用 Codex OAuth**（ChatGPT Pro/Plus 账户——无需 API 密钥）：
```yaml
auxiliary:
  vision:
    provider: "codex"     # 使用您的 ChatGPT OAuth 令牌
    # model 默认为 gpt-5.3-codex（支持视觉）
```

**使用 MiniMax OAuth**（浏览器登录，无需 API 密钥）：
```yaml
model:
  default: MiniMax-M2.7
  provider: minimax-oauth
  base_url: https://api.minimax.io/anthropic
```
运行 `hermes model` 并选择 **MiniMax (OAuth)** 登录并自动设置。对于中国区域，base URL 将是 `https://api.minimaxi.com/anthropic`。请参阅 [MiniMax OAuth 指南](../guides/minimax-oauth.md) 获取完整演练。

**使用本地/自托管模型：**
```yaml
auxiliary:
  vision:
    provider: "main"      # 使用您的活动自定义端点
    model: "my-local-model"
```

`provider: "main"` 使用 Hermes 用于正常聊天的任何 provider——无论是命名自定义 provider（例如 `beans`）、内置 provider 如 `openrouter`，还是遗留 `OPENAI_BASE_URL` 端点。

:::tip
如果您将 Codex OAuth 作为主要模型 provider，视觉自动工作——无需额外配置。Codex 包含在视觉自动检测链中。
:::

:::warning
**视觉需要多模态模型。** 如果您设置 `provider: "main"`，请确保您的端点支持多模态/视觉——否则图像分析将失败。
:::

### 环境变量（遗留）

辅助模型也可以通过环境变量配置。但是，`config.yaml` 是首选方法——它更易于管理并支持所有选项，包括 `base_url` 和 `api_key`。

| 设置 | 环境变量 |
|---------|---------------------|
| 视觉 provider | `AUXILIARY_VISION_PROVIDER` |
| 视觉模型 | `AUXILIARY_VISION_MODEL` |
| 视觉端点 | `AUXILIARY_VISION_BASE_URL` |
| 视觉 API 密钥 | `AUXILIARY_VISION_API_KEY` |
| 网页提取 provider | `AUXILIARY_WEB_EXTRACT_PROVIDER` |
| 网页提取模型 | `AUXILIARY_WEB_EXTRACT_MODEL` |
| 网页提取端点 | `AUXILIARY_WEB_EXTRACT_BASE_URL` |
| 网页提取 API 密钥 | `AUXILIARY_WEB_EXTRACT_API_KEY` |

压缩和回退模型设置仅限 config.yaml。

:::tip
运行 `hermes config` 查看您当前的辅助模型设置。仅当与默认值不同时才会显示覆盖。
:::

## 推理努力

控制模型在响应之前进行多少"思考"：

```yaml
agent:
  reasoning_effort: ""   # 空 = 中等（默认）。选项：none、minimal、low、medium、high、xhigh（最大）
```

未设置时（默认），推理努力默认为"中等"——对大多数任务效果良好的平衡级别。设置值会覆盖它——更高的推理努力在复杂任务上获得更好的结果，代价是更多的 token 和延迟。

您也可以在运行时使用 `/reasoning` 命令更改推理努力：

```
/reasoning           # 显示当前努力级别和显示状态
/reasoning high      # 将推理努力设置为高
/reasoning none      # 禁用推理
/reasoning show      # 在每个响应上方显示模型思考
/reasoning hide      # 隐藏模型思考
```

## 工具使用强制执行

某些模型偶尔将预期操作描述为文本而不是进行工具调用（"我会运行测试..."而不是实际调用终端）。工具使用强制执行注入系统提示指导，将模型引导回实际调用工具。

```yaml
agent:
  tool_use_enforcement: "auto"   # "auto" | true | false | ["model-substring", ...]
```

| 值 | 行为 |
|-------|----------|
| `"auto"`（默认） | 对匹配的模型启用：`gpt`、`codex`、`gemini`、`gemma`、`grok`。对其他所有模型禁用（Claude、DeepSeek、Qwen 等）。 |
| `true` | 始终启用，无论模型如何。如果您注意到当前模型描述动作而不是执行动作，这很有用。 |
| `false` | 始终禁用，无论模型如何。 |
| `["gpt", "codex", "qwen", "llama"]` | 仅当模型名称包含列出的子字符串之一时启用（不区分大小写）。 |

### 它注入什么

启用时，三层指导可能被添加到系统提示：

1. **通用工具使用强制执行**（所有匹配的模型）——指示模型立即进行工具调用而不是描述意图，保持工作直到任务完成，并且永远不会以承诺未来行动的轮次结束。

2. **OpenAI 执行纪律**（仅 GPT 和 Codex 模型）——额外的指导，解决 GPT 特定的失败模式：在部分结果上放弃工作、跳过先决条件查找、而不是使用工具进行幻觉、以及在没有验证的情况下宣布"完成"。

3. **Google 操作指导**（仅 Gemini 和 Gemma 模型）——简洁性、绝对路径、并行工具调用和验证后编辑模式。

这些对用户是透明的，仅影响系统提示。已经可靠使用工具的模型（如 Claude）不需要此指导，这就是为什么 `"auto"` 排除它们的原因。

### 何时打开它

如果您使用的模型不在默认自动列表中，并注意到它经常描述它*会*做什么而不是做它，请设置 `tool_use_enforcement: true` 或将模型子字符串添加到列表：

```yaml
agent:
  tool_use_enforcement: ["gpt", "codex", "gemini", "grok", "my-custom-model"]
```

## TTS 配置

```yaml
tts:
  provider: "edge"              # "edge" | "elevenlabs" | "openai" | "minimax" | "mistral" | "gemini" | "xai" | "neutts"
  speed: 1.0                    # 全局速度乘数（所有 provider 的后备）
  edge:
    voice: "en-US-AriaNeural"   # 322 个声音，74 种语言
    speed: 1.0                  # 速度乘数（转换为速率百分比，例如 1.5 → +50%）
  elevenlabs:
    voice_id: "pNInz6obpgDQGcFmaJgB"
    model_id: "eleven_multilingual_v2"
  openai:
    model: "gpt-4o-mini-tts"
    voice: "alloy"              # alloy、echo、fable、onyx、nova、shimmer
    speed: 1.0                  # 速度乘数（API 限制为 0.25–4.0）
    base_url: "https://api.openai.com/v1"  # 覆盖 OpenAI 兼容 TTS 端点
  minimax:
    speed: 1.0                  # 语音速度乘数
    # base_url: ""              # 可选：覆盖 OpenAI 兼容 TTS 端点
  mistral:
    model: "voxtral-mini-tts-2603"
    voice_id: "c69964a6-ab8b-4f8a-9465-ec0925096ec8"  # Paul - Neutral（默认）
  gemini:
    model: "gemini-2.5-flash-preview-tts"   # 或 gemini-2.5-pro-preview-tts
    voice: "Kore"               # 30 个预建声音：Zephyr、Puck、Kore、Enceladus 等
  xai:
    voice_id: "eve"             # xAI TTS 声音
    language: "en"              # ISO 639-1
    sample_rate: 24000
    bit_rate: 128000            # MP3 比特率
    # base_url: "https://api.x.ai/v1"
  neutts:
    ref_audio: ''
    ref_text: ''
    model: neuphonic/neutts-air-q4-gguf
    device: cpu
```

这控制 `text_to_speech` 工具和语音模式中的语音回复（CLI 中的 `/voice tts` 或消息传递 gateway）。

**速度后备层级：** provider 特定速度（例如 `tts.edge.speed`）→ 全局 `tts.speed` → `1.0` 默认。设置全局 `tts.speed` 以在所有 provider 中应用统一速度，或按 provider 覆盖以进行细粒度控制。

## 显示设置

```yaml
display:
  tool_progress: all      # off | new | all | verbose
  tool_progress_command: false  # 在消息传递 gateway 中启用 /verbose 斜杠命令
  platforms: {}           # 按平台显示覆盖（见下文）
  tool_progress_overrides: {}  # 弃用——改用 display.platforms
  interim_assistant_messages: true  # Gateway：在单独消息中发送自然的中间轮次助手更新
  skin: default           # 内置或自定义 CLI 皮肤（请参阅 user-guide/features/skins）
  personality: "kawaii"  # 遗留 cosmetic 字段仍在某些摘要中显示
  compact: false          # 紧凑输出模式（更少空白）
  resume_display: full    # full（恢复时显示上一条消息）| minimal（仅单行）
  bell_on_complete: false # Agent 完成后播放终端铃声（非常适合长时间任务）
  show_reasoning: false   # 在每个响应上方显示模型推理/思考（使用 /reasoning show|hide 切换）
  streaming: false        # 将 token 流式传输到终端（实时输出）
  show_cost: false        # 在 CLI 状态栏中显示估计的 $ 成本
  tool_preview_length: 0  # 工具调用预览的最大字符数（0 = 无限制，显示完整路径/命令）
  runtime_metadata_footer: false  # Gateway：在最终回复后附加运行时上下文页脚
  language: en            # 静态消息的 UI 语言（审批提示、某些 gateway 回复）。en | zh | ja | de | es | fr | tr | uk
```

### 静态消息的 UI 语言

`display.language` 设置翻译一小部分静态面向用户的消息——CLI 审批提示、少数 gateway 斜杠命令回复（例如 restart-drain 通知、"approval expired"、"goal cleared"）。它**不**翻译 agent 响应、日志行、工具输出、错误回溯或斜杠命令描述——那些保持英语。如果您希望 agent 本身用另一种语言回复，只需在您的提示或系统消息中告诉它。

支持的值：`en`（默认）、`zh`（简体中文）、`ja`（日语）、`de`（德语）、`es`（西班牙语）、`fr`（法语）、`tr`（土耳其语）、`uk`（乌克兰语）。未知值回退到英语。

您也可以通过 `HERMES_LANGUAGE` env var 按会话设置，这会覆盖配置值。

```yaml
display:
  language: zh   # CLI 审批提示以中文显示
```

| 模式 | 您看到的内容 |
|------|-------------|
| `off` | 静默——仅最终响应 |
| `new` | 仅当工具更改时显示工具指示器 |
| `all` | 每个工具调用带简短预览（默认） |
| `verbose` | 完整参数、结果和调试日志 |

在 CLI 中，使用 `/verbose` 循环切换这些模式。要在消息传递平台（Telegram、Discord、Slack 等）中使用 `/verbose`，请在上面 `display` 部分设置 `tool_progress_command: true`。命令将循环模式并保存到配置。

### 运行时元数据页脚（仅 gateway）

当 `display.runtime_metadata_footer: true` 时，Hermes 在每个 gateway 轮次的**最终**消息后附加一个小的运行时上下文页脚——CLI 在其状态栏中显示的相同信息（模型、会话时长、token、成本）。默认关闭；如果您的团队希望每个回复都包含来源，请按 gateway 选择加入。

```yaml
display:
  runtime_metadata_footer: true
```

附加到 Telegram/Discord/Slack 回复的示例页脚：

```
— claude-opus-4.7 · 12 tool calls · 2m 14s · $0.042
```

只有轮次的**最终**消息获得页脚；中间更新保持干净。

### 按平台进度覆盖

不同平台有不同的详细程度需求。例如，Signal 无法编辑消息，因此每个进度更新都成为单独的消息——嘈杂。使用 `display.platforms` 设置按平台模式：

```yaml
display:
  tool_progress: all          # 全局默认
  platforms:
    signal:
      tool_progress: 'off'    # 在 Signal 上静音进度
    telegram:
      tool_progress: verbose  # 在 Telegram 上详细进度
    slack:
      tool_progress: 'off'    # 在共享 Slack 工作区中安静
```

没有覆盖的平台回退到全局 `tool_progress` 值。有效的平台键：`telegram`、`discord`、`slack`、`signal`、`whatsapp`、`matrix`、`mattermost`、`email`、`sms`、`homeassistant`、`dingtalk`、`feishu`、`wecom`、`weixin`、`bluebubbles`、`qqbot`。遗留的 `display.tool_progress_overrides` 键仍为向后兼容而加载，但在首次加载时已弃用并迁移到 `display.platforms`。

`interim_assistant_messages` 仅适用于 gateway。启用时，Hermes 将已完成的中间轮次助手更新作为单独聊天消息发送。这独立于 `tool_progress`，不需要 gateway 流式传输。

## 隐私

```yaml
privacy:
  redact_pii: false  # 从 LLM 上下文中剥离 PII（仅 gateway）
```

当 `redact_pii` 为 `true` 时，gateway 在将其发送到支持的平台上的 LLM 之前，从系统提示中编辑个人身份信息：

| 字段 | 处理方式 |
|-------|-----------|
| 电话号码（WhatsApp/Signal 上的用户 ID） | 哈希为 `user_<12-char-sha256>` |
| 用户 ID | 哈希为 `user_<12-char-sha256>` |
| 聊天 ID | 数字部分哈希，保留平台前缀（`telegram:<hash>`） |
| 主频道 ID | 数字部分哈希 |
| 用户名/用户名 | **不受影响**（用户选择、公开可见） |

**平台支持：** 编辑适用于 WhatsApp、Signal 和 Telegram。Discord 和 Slack 被排除，因为它们的提及系统（`<@user_id>`）需要在 LLM 上下文中使用真实 ID。

哈希是确定性的——同一用户始终映射到相同的哈希，因此模型仍然可以区分群聊中的用户。路由和传递在内部使用原始值。

## 语音转文本（STT）

```yaml
stt:
  provider: "local"            # "local" | "groq" | "openai" | "mistral"
  local:
    model: "base"              # tiny、base、small、medium、large-v3
  openai:
    model: "whisper-1"         # whisper-1 | gpt-4o-mini-transcribe | gpt-4o-transcribe
  # model: "whisper-1"         # 遗留后备键仍被尊重
```

Provider 行为：

- `local` 使用在您的机器上运行的 `faster-whisper`。使用 `pip install faster-whisper` 单独安装。
- `groq` 使用 Groq 的 Whisper 兼容端点并读取 `GROQ_API_KEY`。
- `openai` 使用 OpenAI 语音 API 并读取 `VOICE_TOOLS_OPENAI_KEY`。

如果请求的 provider 不可用，Hermes 按此顺序自动回退：`local` → `groq` → `openai`。

Groq 和 OpenAI 模型覆盖由环境驱动：

```bash
STT_GROQ_MODEL=whisper-large-v3-turbo
STT_OPENAI_MODEL=whisper-1
GROQ_BASE_URL=https://api.groq.com/openai/v1
STT_OPENAI_BASE_URL=https://api.openai.com/v1
```

## 语音模式（CLI）

```yaml
voice:
  record_key: "ctrl+b"         # CLI 中的按键说话
  max_recording_seconds: 120    # 长时间录制的硬停止
  auto_tts: false               # /voice on 时自动启用语音回复
  beep_enabled: true            # 在 CLI 语音模式中播放录音开始/停止提示音
  silence_threshold: 200        # 语音检测的 RMS 阈值
  silence_duration: 3.0         # 自动停止前的静音秒数
```

使用 CLI 中的 `/voice on` 启用麦克风模式，`record_key` 开始/停止录制，`/voice tts` 切换语音回复。请参阅 [语音模式](/docs/user-guide/features/voice-mode) 获取端到端设置和平台特定行为。

## 流式传输

将 token 流式传输到终端或消息传递平台，因为它们到达时，而不是等待完整响应。

### CLI 流式传输

```yaml
display:
  streaming: true         # 实时将 token 流式传输到终端
  show_reasoning: true    # 也流式传输推理/思考 token（可选）
```

启用时，响应在流式框中逐 token 显示。工具调用仍被静默捕获。如果 provider 不支持流式传输，它会自动回退到正常显示。

### Gateway 流式传输（Telegram、Discord、Slack）

```yaml
streaming:
  enabled: true           # 启用渐进式消息编辑
  transport: edit         # "edit"（渐进式消息编辑）或 "off"
  edit_interval: 0.3      # 消息编辑之间的秒数
  buffer_threshold: 40    # 强制刷新编辑前的字符数
  cursor: " ▉"            # 流式传输期间显示的光标
  fresh_final_after_seconds: 60   # 预览这么旧时发送新鲜最终（Telegram）；0 = 始终就地编辑
```

启用时，bot 在第一个 token 发送消息，然后在更多 token 到达时逐步编辑。不支持消息编辑的平台（Signal、Email、Home Assistant）在首次尝试时自动检测——流式传输为该会话优雅禁用，不会产生大量消息。

对于单独的自然的中间轮次助手更新而不是渐进式 token 编辑，设置 `display.interim_assistant_messages: true`。

**溢出处理：** 如果流式文本超过平台的消息长度限制（~4096 字符），当前消息被最终确定并自动开始新消息。

**新鲜最终（Telegram）：** Telegram 的 `editMessageText` 保留原始消息时间戳，因此长时间运行的流式回复即使在完成后也会保持第一个 token 时间戳。当 `fresh_final_after_seconds > 0`（默认 `60`）时，完成的回复作为全新消息传递（尽力删除过时的预览），以便 Telegram 的可见时间戳反映完成时间。简短短预览仍就地最终确定。设置为 `0` 以始终就地编辑。

:::note
流式传输默认禁用。在 `~/.hermes/config.yaml` 中启用它以尝试流式传输 UX。
:::

## 群聊会话隔离

控制共享聊天是每个房间保持一个对话还是每个参与者保持一个对话：

```yaml
group_sessions_per_user: true  # true = 组/频道中按用户隔离，false = 每个聊天一个共享会话
```

- `true` 是默认设置和推荐设置。在 Discord 频道、Telegram 群组、Slack 频道和类似共享上下文中，当平台提供用户 ID 时，每个发送者都会获得自己的会话。
- `false` 恢复为旧的共享房间行为。如果您明确希望 Hermes 将频道视为一个协作对话，这可能很有用，但这也意味着用户共享上下文、token 成本和中断状态。
- 直接消息不受影响。Hermes 仍然像往常一样按聊天/DM ID 键控 DMs。
- 线程无论如何都与其父频道隔离；使用 `true` 时，每个参与者也在线程内获得自己的会话。

有关行为详情和示例，请参阅 [会话](/docs/user-guide/sessions) 和 [Discord 指南](/docs/user-guide/messaging/discord)。

## 未授权 DM 行为

控制 Hermes 在未知用户发送直接消息时做什么：

```yaml
unauthorized_dm_behavior: pair

whatsapp:
  unauthorized_dm_behavior: ignore
```

- `pair` 是默认值。Hermes 拒绝访问，但在 DM 中回复一次性配对代码。
- `ignore` 静默丢弃未授权的 DM。
- 平台部分覆盖全局默认值，因此您可以广泛保持配对启用，而让一个平台更安静。

## 快速命令

定义自定义命令，这些命令要么运行 shell 命令而不调用 LLM，要么将一个斜杠命令别名化为另一个。Exec 快速命令是零 token 的，从消息传递平台（Telegram、Discord 等）用于快速服务器检查或实用脚本很有用。

```yaml
quick_commands:
  status:
    type: exec
    command: systemctl status hermes-agent
  disk:
    type: exec
    command: df -h /
  update:
    type: exec
    command: cd ~/.hermes/hermes-agent && git pull && pip install -e .
  gpu:
    type: exec
    command: nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
  restart:
    type: alias
    target: /gateway restart
```

用法：在 CLI 或任何消息传递平台中输入 `/status`、`/disk`、`/update`、`/gpu` 或 `/restart`。`exec` 命令在主机上本地运行并直接返回输出——无 LLM 调用，无 token 消耗。`alias` 命令重写为配置的斜杠命令目标。

- **30 秒超时**——长时间运行的命令会被终止并返回错误消息
- **优先级**——快速命令在 skill 命令之前检查，因此您可以覆盖 skill 名称
- **自动补全**——快速命令在调度时解析，不显示在内置斜杠命令自动补全表中
- **类型**——支持的类型是 `exec` 和 `alias`；其他类型显示错误
- **无处不在**——CLI、Telegram、Discord、Slack、WhatsApp、Signal、Email、Home Assistant

仅字符串提示快捷方式不是有效的快速命令。对于可重用的提示工作流程，请创建 skill 或别名到现有斜杠命令。

## 人类延迟

在消息传递平台中模拟类似人类的响应节奏：

```yaml
human_delay:
  mode: "off"                  # off | natural | custom
  min_ms: 800                  # 最小延迟（自定义模式）
  max_ms: 2500                 # 最大延迟（自定义模式）
```

## 代码执行

配置 `execute_code` 工具：

```yaml
code_execution:
  mode: project                # project（默认）| strict
  timeout: 300                 # 最大执行时间（秒）
  max_tool_calls: 50           # 代码执行中的最大工具调用数
```

**`mode`** 控制脚本的工作目录和 Python 解释器：

- **`project`**（默认）——脚本在使用活动 virtualenv/conda env 的 python 的会话工作目录中运行。项目依赖（`pandas`、`torch`、项目包）和相对路径（`.env`、`./data.csv`）自然解析，匹配 `terminal()` 看到的内容。
- **`strict`**——脚本在使用 `sys.executable`（Hermes 自己的 python）的临时暂存目录中运行。最大可重现性，但项目依赖和相对路径不会解析。

环境清理（剥离 `*_API_KEY`、`*_TOKEN`、`*_SECRET`、`*_PASSWORD`、`*_CREDENTIAL`、`*_PASSWD`、`*_AUTH`）和工具白名单在两种模式下相同应用——切换模式不会改变安全态势。

## Web 搜索后端

`web_search`、`web_extract` 和 `web_crawl` 工具支持五种后端 provider。在 `config.yaml` 中或通过 `hermes tools` 配置后端：

```yaml
web:
  backend: firecrawl    # firecrawl | searxng | parallel | tavily | exa

  # 或使用按能力键混合 provider（例如免费搜索 + 付费提取）：
  search_backend: "searxng"
  extract_backend: "firecrawl"
```

| 后端 | Env 变量 | 搜索 | 提取 | 爬取 |
|---------|---------|--------|---------|-------|
| **Firecrawl**（默认） | `FIRECRAWL_API_KEY` | ✔ | ✔ | ✔ |
| **SearXNG** | `SEARXNG_URL` | ✔ | — | — |
| **Parallel** | `PARALLEL_API_KEY` | ✔ | ✔ | — |
| **Tavily** | `TAVILY_API_KEY` | ✔ | ✔ | ✔ |
| **Exa** | `EXA_API_KEY` | ✔ | ✔ | — |

**后端选择：** 如果未设置 `web.backend`，则从可用 API 密钥自动检测后端。如果仅设置 `SEARXNG_URL`，则使用 SearXNG。如果仅设置 `EXA_API_KEY`，则使用 Exa。如果仅设置 `TAVILY_API_KEY`，则使用 Tavily。如果仅设置 `PARALLEL_API_KEY`，则使用 Parallel。否则 Firecrawl 是默认值。

**SearXNG** 是一个免费的、自托管的、尊重隐私的元搜索引擎，查询 70+ 搜索引擎。无需 API 密钥——只需将 `SEARXNG_URL` 设置为您的实例（例如 `http://localhost:8080`）。SearXNG 仅支持搜索；`web_extract` 和 `web_crawl` 需要单独的提取 provider（设置 `web.extract_backend`）。请参阅 [Web 搜索设置指南](/docs/user-guide/features/web-search) 获取 Docker 设置说明。

**自托管 Firecrawl：** 设置 `FIRECRAWL_API_URL` 指向您自己的实例。当设置自定义 URL 时，API 密钥变为可选（在服务器上设置 `USE_DB_AUTHENTICATION=***` 以禁用 auth）。

**Parallel 搜索模式：** 设置 `PARALLEL_SEARCH_MODE` 以控制搜索行为——`fast`、`one-shot` 或 `agentic`（默认：`agentic`）。

**Exa：** 在 `~/.hermes/.env` 中设置 `EXA_API_KEY`。支持 `category` 过滤（`company`、`research paper`、`news`、`people`、`personal site`、`pdf`）以及域/日期过滤。

## 浏览器

配置浏览器自动化行为：

```yaml
browser:
  inactivity_timeout: 120        # 自动关闭空闲会话前的秒数
  command_timeout: 30             # 浏览器命令的超时秒数（截图、导航等）
  record_sessions: false         # 自动将会话记录为 WebM 视频到 ~/.hermes/browser_recordings/
  # 可选 CDP 覆盖——设置时，Hermes 附加到您自己的
  # Chrome（通过 /browser connect）而不是启动无头浏览器。
  cdp_url: ""
  # 对话监督器——当 CDP 后端附加时（Browserbase、
  # 通过 /browser connect 的本地 Chrome）控制如何处理原生 JS 对话框
  #（alert / confirm / prompt）。在 Camofox 和默认本地 agent-browser 模式下忽略。
  dialog_policy: must_respond    # must_respond | auto_dismiss | auto_accept
  dialog_timeout_s: 300          # must_respond 下的安全自动关闭（秒）
  camofox:
    managed_persistence: false   # 当 true 时，Camofox 会话在重启之间保持 cookies/登录
```

**对话框策略：**

- `must_respond`（默认）——捕获对话框，在 `browser_snapshot.pending_dialogs` 中显示，并等待 agent 调用 `browser_dialog(action=...)`。30 秒后没有响应，对话框自动关闭以防止页面的 JS 线程永远停滞。
- `auto_dismiss`——捕获，立即关闭。agent 仍然在事后 `browser_snapshot.recent_dialogs` 中看到带有 `closed_by="auto_policy"` 的对话框记录。
- `auto_accept`——捕获，立即接受。对带有激进 `beforeunload` 提示的页面有用。

请参阅 [浏览器功能页面](./features/browser.md#browser_dialog) 获取完整的对话框工作流程。

浏览器工具集支持多个 provider。请参阅 [浏览器功能页面](/docs/user-guide/features/browser) 了解 Browserbase、Browser Use 和本地 Chrome CDP 设置的详情。

## 时区

使用 IANA 时区字符串覆盖服务器本地时区。影响日志、cron 调度和系统提示时间注入中的时间戳。

```yaml
timezone: "America/New_York"   # IANA 时区（默认："" = 服务器本地时间）
```

支持的值：任何 IANA 时区标识符（例如 `America/New_York`、`Europe/London`、`Asia/Kolkata`、`UTC`）。保留为空或省略以使用服务器本地时间。

## Discord

为消息传递 gateway 配置 Discord 特定行为：

```yaml
discord:
  require_mention: true          # 在服务器频道中需要 @提及才能响应
  free_response_channels: ""     # bot 无需 @提及即可响应的逗号分隔频道 ID 列表
  auto_thread: true              # 在频道中 @提及时自动创建线程
```

- `require_mention` — 当 `true`（默认）时，bot 仅在服务器频道中被 @提及 `@BotName` 时响应。DM 始终无需提及即可工作。
- `free_response_channels` — 逗号分隔的频道 ID 列表，bot 在其中无需 @提及即可响应每条消息。
- `auto_thread` — 当 `true`（默认）时，频道中的提及自动为对话创建线程，保持频道清洁（类似于 Slack 线程）。

## 安全

预执行安全扫描和密钥编辑：

```yaml
security:
  redact_secrets: false          # 在工具输出和日志中编辑 API 密钥模式（默认关闭）
  tirith_enabled: true           # 启用 Tirith 安全扫描以检测终端命令
  tirith_path: "tirith"          # tirith 二进制文件路径（默认：$PATH 中的 "tirith"）
  tirith_timeout: 5              # tirith 扫描超时前的秒数
  tirith_fail_open: true         # 如果 tirith 不可用则允许命令执行
  website_blocklist:             # 请参阅下面的网站阻止列表部分
    enabled: false
    domains: []
    shared_files: []
```

- `redact_secrets` — 当 `true` 时，在工具输出进入对话上下文和日志之前自动检测并编辑看起来像 API 密钥、令牌和密码的模式。**默认关闭**——如果您经常在工具输出中使用真实凭证并想要安全网，请启用。显式设置为 `true` 以打开。
- `tirith_enabled` — 当 `true` 时，终端命令在执行前由 [Tirith](https://github.com/StackGuardian/tirith) 扫描以检测潜在危险操作。
- `tirith_path` — tirith 二进制文件的路径。如果 tirith 安装在非标准位置，请设置此值。
- `tirith_timeout` — 等待 tirith 扫描的最大秒数。如果扫描超时，命令继续。
- `tirith_fail_open` — 当 `true`（默认）时，如果 tirith 不可用或失败，则允许命令执行。设置为 `false` 以在 tirith 无法验证时阻止命令。

## 网站阻止列表

阻止 agent 的 Web 和浏览器工具访问特定域：

```yaml
security:
  website_blocklist:
    enabled: false               # 启用 URL 阻止（默认：false）
    domains:                     # 阻止的域模式列表
      - "*.internal.company.com"
      - "admin.example.com"
      - "*.local"
    shared_files:                # 从外部文件加载其他规则
      - "/etc/hermes/blocked-sites.txt"
```

启用时，任何匹配阻止域模式的 URL 在 Web 或浏览器工具执行之前被拒绝。这适用于 `web_search`、`web_extract`、`browser_navigate` 和任何访问 URL 的工具。

域规则支持：
- 精确域：`admin.example.com`
- 通配符子域：`*.internal.company.com`（阻止所有子域）
- TLD 通配符：`*.local`

共享文件每行包含一个域规则（空行和 `#` 注释被忽略）。缺失或不可读的文件记录警告但不禁用其他 Web 工具。

策略缓存 30 秒，因此配置更改很快生效，无需重启。

## 智能审批

控制 Hermes 如何处理潜在危险命令：

```yaml
approvals:
  mode: manual   # manual | smart | off
```

| 模式 | 行为 |
|------|----------|
| `manual`（默认） | 在执行任何标记命令之前提示用户。在 CLI 中，显示交互式审批对话框。在消息传递中，排队待处理的审批请求。 |
| `smart` | 使用辅助 LLM 评估标记命令是否真的危险。低风险命令自动批准，会话级别持久。真正危险的命令升级给用户。 |
| `off` | 跳过所有审批检查。等同于 `HERMES_YOLO_MODE=true`。**谨慎使用。** |

智能模式对于减少审批疲劳特别有用——它让 agent 在安全操作上更自主地工作，同时仍能捕获真正危险的命令。

:::warning
设置 `approvals.mode: off` 会禁用终端命令的所有安全检查。仅在受信任的沙箱环境中使用。
:::

## 检查点

在破坏性文件操作之前的自动文件系统快照。请参阅 [检查点和回滚](/docs/user-guide/checkpoints-and-rollback) 了解详情。

```yaml
checkpoints:
  enabled: true                  # 启用自动检查点（也可：hermes --checkpoints）
  max_snapshots: 50              # 每个目录保留的最大检查点数
```

## 委托

为委托工具配置子 agent 行为：

```yaml
delegation:
  # model: "google/gemini-3-flash-preview"  # 覆盖模型（空 = 继承父级）
  # provider: "openrouter"                  # 覆盖 provider（空 = 继承父级）
  # base_url: "http://localhost:1234/v1"    # 直接 OpenAI 兼容端点（优先于 provider）
  # api_key: "local-key"                    # base_url 的 API 密钥（回退到 OPENAI_API_KEY）
  max_concurrent_children: 3                # 每个批次的并行子项（floor 1，无上限）。也可通过 DELEGATION_MAX_CONCURRENT_CHILDREN env var。
  max_spawn_depth: 1                        # 委托树深度上限（1-3，限制）。1 = 扁平（默认）：父级生成不能委托的叶子。2 = 编排器子项可以生成叶子孙子。3 = 三级。
  orchestrator_enabled: true                # 全局终止开关。当 false 时，role="orchestrator" 被忽略，每个子项强制为叶子，无论 max_spawn_depth 如何。
```

**子 agent provider:model 覆盖：** 默认情况下，子 agent 继承父级 agent 的 provider 和模型。设置 `delegation.provider` 和 `delegation.model` 以将子 agent 路由到不同的 provider:model 对——例如，为范围狭窄的子任务使用廉价/快速模型，而您的主要 agent 运行昂贵的推理模型。

**直接端点覆盖：** 如果您需要明显的自定义端点路径，设置 `delegation.base_url`、`delegation.api_key` 和 `delegation.model`。那会直接向该 OpenAI 兼容端点发送子 agent，并优先于 `delegation.provider`。如果省略 `delegation.api_key`，Hermes 仅回退到 `OPENAI_API_KEY`。

委托 provider 使用与 CLI/gateway 启动相同的凭证解析。支持所有配置的 provider：`openrouter`、`nous`、`copilot`、`zai`、`kimi-coding`、`minimax`、`minimax-cn`。当设置 provider 时，系统自动解析正确的 base URL、API 密钥和 API 模式——无需手动凭证接线。

**优先级：** `config` 中的 `delegation.base_url` → `config` 中的 `delegation.provider` → 父 provider（继承）。`config` 中的 `delegation.model` → 父模型（继承）。仅设置 `model` 而不设置 `provider` 仅更改模型名称，同时保持父级的凭证（适用于在同一 provider（如 OpenRouter）内切换模型）。

**宽度和深度：** `max_concurrent_children` 限制每个批次并行运行的子 agent 数量（默认 `3`，floor 为 1，无上限）。也可以通过 `DELEGATION_MAX_CONCURRENT_CHILDREN` env var 设置。当模型提交比上限更长的 `tasks` 数组时，`delegate_task` 返回工具错误解释限制，而不是静默截断。`max_spawn_depth` 控制委托树深度（限制为 1-3）。在默认 `1` 时，委托是扁平的：子项不能生成孙子，传递 `role="orchestrator"` 静默降级为 `leaf`。提高到 `2` 以便编排器子项可以生成叶子孙子；`3` 用于三级树。Agent 通过 `role="orchestrator"` 选择加入每个调用的编排；`orchestrator_enabled: false` 强制每个子项返回叶子，无论如何。成本按乘法缩放——在 `max_spawn_depth: 3` 且 `max_concurrent_children: 3` 时，树可以达到 3×3×3 = 27 个并发叶子 agent。请参阅 [子代理委托 → 深度限制和嵌套编排](features/delegation.md#depth-limit-and-nested-orchestration) 获取使用模式。

## 澄清

配置澄清提示行为：

```yaml
clarify:
  timeout: 120                 # 等待用户澄清响应的秒数
```

## 上下文文件（SOUL.md、AGENTS.md）

Hermes 使用两种不同的上下文范围：

| 文件 | 目的 | 范围 |
|------|---------|---------|
| `SOUL.md` | **主要 agent 身份** — 定义 agent 是谁（系统提示中的槽位 #1） | `~/.hermes/SOUL.md` 或 `$HERMES_HOME/SOUL.md` |
| `.hermes.md` / `HERMES.md` | 项目特定说明（最高优先级） | 走到 git 根目录 |
| `AGENTS.md` | 项目特定说明、编码约定 | 递归目录遍历 |
| `CLAUDE.md` | Claude Code 上下文文件（也被检测） | 仅工作目录 |
| `.cursorrules` | Cursor IDE 规则（也被检测） | 仅工作目录 |
| `.cursor/rules/*.mdc` | Cursor 规则文件（也被检测） | 仅工作目录 |

- **SOUL.md** 是 agent 的主要身份。它占用系统提示中的槽位 #1，完全替换内置默认身份。编辑它以完全自定义 agent 是谁。
- 如果 SOUL.md 缺失、空或无法加载，Hermes 回退到内置默认身份。
- **项目上下文文件使用优先级系统** — 仅加载一种类型（优先匹配）：`.hermes.md` → `AGENTS.md` → `CLAUDE.md` → `.cursorrules`。SOUL.md 始终独立加载。
- **AGENTS.md** 是分层的：如果子目录也有 AGENTS.md，则全部合并。
- Hermes 如果不存在会自动种子化默认 `SOUL.md`。
- 所有加载的上下文文件上限为 20,000 字符，并进行智能截断。

另请参阅：
- [人格和 SOUL.md](/docs/user-guide/features/personality)
- [上下文文件](/docs/user-guide/features/context-files)

## 工作目录

| 上下文 | 默认 |
|---------|---------|
| **CLI（`hermes`）** | 运行命令的当前目录 |
| **消息传递 gateway** | 主目录 `~`（使用 `MESSAGING_CWD` 覆盖） |
| **Docker / Singularity / Modal / SSH** | 容器内或远程机器上的用户主目录 |

覆盖工作目录：
```bash
# 在 ~/.hermes/.env 或 ~/.hermes/config.yaml 中：
MESSAGING_CWD=/home/myuser/projects    # Gateway 会话
TERMINAL_CWD=/workspace                # 所有终端会话
```
