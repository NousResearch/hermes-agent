---
sidebar_position: 2
title: "配置"
description: "配置 Hermes Agent —— config.yaml、提供商、模型、API 密钥等"
---

# 配置

所有设置都存储在 `~/.hermes/` 目录中，便于访问。

## 目录结构

```text
~/.hermes/
├── config.yaml     # 设置（模型、终端、TTS、压缩等）
├── .env            # API 密钥和机密
├── auth.json       # OAuth 提供商凭证（Nous Portal 等）
├── SOUL.md         # 主智能体身份（系统提示中的 #1 插槽）
├── memories/       # 持久化记忆（MEMORY.md、USER.md）
├── skills/         # 智能体创建的技能（通过 skill_manage 工具管理）
├── cron/           # 定时任务
├── sessions/       # 网关会话
└── logs/           # 日志（errors.log、gateway.log —— 机密自动脱敏）
```

## 管理配置

```bash
hermes config              # 查看当前配置
hermes config edit         # 在编辑器中打开 config.yaml
hermes config set KEY VAL  # 设置特定值
hermes config check        # 检查缺失选项（更新后）
hermes config migrate      # 交互式添加缺失选项

# 示例：
hermes config set model anthropic/claude-opus-4
hermes config set terminal.backend docker
hermes config set OPENROUTER_API_KEY sk-or-...  # 保存到 .env
```

:::tip
`hermes config set` 命令会自动将值路由到正确的文件——API 密钥保存到 `.env`，其他所有内容保存到 `config.yaml`。
:::

## 配置优先级

设置按以下顺序解析（优先级从高到低）：

1. **CLI 参数** —— 例如 `hermes chat --model anthropic/claude-sonnet-4`（每次调用覆盖）
2. **`~/.hermes/config.yaml`** —— 所有非机密设置的主配置文件
3. **`~/.hermes/.env`** —— 环境变量的后备；**必需**用于机密（API 密钥、token、密码）
4. **内置默认值** —— 未设置任何内容时的硬编码安全默认值

:::info 经验法则
机密（API 密钥、机器人 token、密码）放入 `.env`。其他所有内容（模型、终端后端、压缩设置、内存限制、工具集）放入 `config.yaml`。当两者都设置时，`config.yaml` 对非机密设置优先。
:::

## 环境变量替换

您可以在 `config.yaml` 中使用 `${VAR_NAME}` 语法引用环境变量：

```yaml
auxiliary:
  vision:
    api_key: ${GOOGLE_API_KEY}
    base_url: ${CUSTOM_VISION_URL}

delegation:
  api_key: ${DELEGATION_KEY}
```

单个值中的多个引用有效：`url: "${HOST}:${PORT}"`。如果引用的变量未设置，占位符会原样保留（`${UNDEFINED_VAR}` 保持不变）。仅支持 `${VAR}` 语法——裸 `$VAR` 不会被展开。

有关 AI 提供商设置（OpenRouter、Anthropic、Copilot、自定义端点、自托管 LLM、备用模型等），请参阅 [AI 提供商](/integrations/providers)。

### 提供商超时

您可以为每个提供商设置 `providers.<id>.request_timeout_seconds` 作为全提供商请求超时，以及 `providers.<id>.models.<model>.timeout_seconds` 作为模型特定覆盖。适用于每个传输层（OpenAI-wire、原生 Anthropic、Anthropic-compatible）的主回合客户端、备用链、凭证轮换后的重建，以及（对于 OpenAI-wire）每个请求的超时关键字参数——因此配置值优先于遗留的 `HERMES_API_TIMEOUT` 环境变量。

您还可以设置 `providers.<id>.stale_timeout_seconds` 用于非流式过时调用检测器，以及 `providers.<id>.models.<model>.stale_timeout_seconds` 作为模型特定覆盖。这优先于遗留的 `HERMES_API_CALL_STALE_TIMEOUT` 环境变量。

保留这些未设置会保持遗留默认值（`HERMES_API_TIMEOUT=1800`s，`HERMES_API_CALL_STALE_TIMEOUT=300`s，原生 Anthropic 900s）。目前未为 AWS Bedrock 连接（`bedrock_converse` 和 AnthropicBedrock SDK 路径都使用 boto3 及其自己的超时配置）。请参阅 [`cli-config.yaml.example`](https://github.com/NousResearch/hermes-agent/blob/main/cli-config.yaml.example) 中的注释示例。

## 终端后端配置

Hermes 支持七种终端后端。每个后端决定智能体的 shell 命令实际在哪里执行——您的本地机器、Docker 容器、通过 SSH 的远程服务器、Modal 云沙箱（直接或通过 Nous 管理的网关）、Daytona 工作区、Vercel Sandbox，或 Singularity/Apptainer 容器。

```yaml
terminal:
  backend: local    # local | docker | ssh | modal | daytona | vercel_sandbox | singularity
  cwd: "."          # 网关/定时任务工作目录（CLI 始终使用启动目录）
  timeout: 180      # 每条命令的超时秒数
  env_passthrough: []  # 要转发到沙箱执行的环境变量名（终端 + execute_code）
  singularity_image: "docker://nikolaik/python-nodejs:python3.11-nodejs20"  # Singularity 后端的容器镜像
  modal_image: "nikolaik/python-nodejs:python3.11-nodejs20"                 # Modal 后端的容器镜像
  daytona_image: "nikolaik/python-nodejs:python3.11-nodejs20"               # Daytona 后端的容器镜像
```

对于云沙箱（如 Modal、Daytona 和 Vercel Sandbox），`container_persistent: true` 意味着 Hermes 会尝试在沙箱重建时保留文件系统状态。它不保证相同的实时沙箱、PID 空间或后台进程稍后仍在运行。

### 后端概览

| 后端 | 命令运行位置 | 隔离 | 最适合 |
|---------|-------------------|-----------|----------|
| **local** | 直接运行在您的机器上 | 无 | 开发、个人使用 |
| **docker** | 单个持久 Docker 容器（跨会话、`/new`、子智能体共享） | 完整（命名空间、cap-drop） | 安全沙箱、CI/CD |
| **ssh** | 通过 SSH 的远程服务器 | 网络边界 | 远程开发、高性能硬件 |
| **modal** | Modal 云沙箱 | 完整（云 VM） | 临时云计算、评估 |
| **daytona** | Daytona 工作区 | 完整（云容器） | 托管云开发环境 |
| **vercel_sandbox** | Vercel Sandbox | 完整（云 microVM） | 带快照文件系统持久化的云执行 |
| **singularity** | Singularity/Apptainer 容器 | 命名空间（--containall） | HPC 集群、共享机器 |

### Local 后端

默认选项。命令直接在您的机器上运行，无隔离。无需特殊设置。

```yaml
terminal:
  backend: local
```

:::warning
智能体拥有与您的用户账户相同的文件系统访问权限。使用 `hermes tools` 禁用您不需要的工具，或切换到 Docker 进行沙箱隔离。
:::

### Docker 后端 {#docker-backend}

在安全加固的 Docker 容器中运行命令（丢弃所有能力、禁止权限提升、PID 限制）。

**单个持久容器，不是每条命令一个。** Hermes 在首次使用时启动一个长期运行的容器，并通过 `docker exec` 将每个终端、文件和 `execute_code` 调用路由到同一个容器中——跨会话、`/new`、`/reset` 和 `delegate_task` 子智能体——在 Hermes 进程的生命周期内。工作目录更改、已安装的包和 `/workspace` 中的文件从一个工具调用到下一个工具调用都会保留。容器在关闭时停止并移除。详情请参阅下面的**容器生命周期**。

```yaml
terminal:
  backend: docker
  docker_image: "nikolaik/python-nodejs:python3.11-nodejs20"
  docker_mount_cwd_to_workspace: false  # 将启动目录挂载到 /workspace
  docker_run_as_host_user: false   # 请参阅下面的"以宿主机用户运行容器"
  docker_forward_env:              # 要转发到容器的环境变量
    - "GITHUB_TOKEN"
  docker_volumes:                  # 宿主机目录挂载
    - "/home/user/projects:/workspace/projects"
    - "/home/user/data:/data:ro"   # :ro 表示只读

  # 资源限制
  container_cpu: 1                 # CPU 核心数（0 = 无限制）
  container_memory: 5120           # MB（0 = 无限制）
  container_disk: 51200            # MB（需要 overlay2 on XFS+pquota）
  container_persistent: true       # 跨会话持久化 /workspace 和 /root
```

**要求：** Docker Desktop 或 Docker Engine 已安装并运行。Hermes 探测 `$PATH` 加上常见的 macOS 安装位置（`/usr/local/bin/docker`、`/opt/homebrew/bin/docker`、Docker Desktop app bundle）。Podman 开箱即用支持：当两者都安装时，设置 `HERMES_DOCKER_BINARY=podman`（或完整路径）以强制使用它。

**容器生命周期：** Hermes 复用一个长期运行的容器（`docker run -d ... sleep 2h`）用于每个终端和文件工具调用，跨会话、`/new`、`/reset` 和 `delegate_task` 子智能体，在 Hermes 进程的生命周期内。命令通过 `docker exec` 以登录 shell 运行，因此工作目录更改、已安装的包和 `/workspace` 中的文件都从一个工具调用到下一个工具调用保留。容器在 Hermes 关闭时停止并移除（或当空闲清理回收它时）。

通过 `delegate_task(tasks=[...])` 生成的并行子智能体共享这一个容器——并发的 `cd`、环境变量更改和对同一路径的写入会发生冲突。如果子智能体需要隔离的沙箱，它必须通过 `register_task_env_overrides()` 注册每个任务的镜像覆盖，RL 和基准环境（TerminalBench2、HermesSweEnv 等）会自动为其每个任务的 Docker 镜像执行此操作。

**安全加固：**
- `--cap-drop ALL`，仅添加回 `DAC_OVERRIDE`、`CHOWN`、`FOWNER`
- `--security-opt no-new-privileges`
- `--pids-limit 256`
- 大小受限的 tmpfs：`/tmp`（512MB）、`/var/tmp`（256MB）、`/run`（64MB）

**凭证转发：** `docker_forward_env` 中列出的环境变量首先从您的 shell 环境解析，然后从 `~/.hermes/.env` 解析。技能还可以声明 `required_environment_variables`，这些会自动合并。

### SSH 后端

通过 SSH 在远程服务器上运行命令。使用 ControlMaster 进行连接复用（5 分钟空闲保活）。默认启用持久 shell——状态（cwd、环境变量）跨命令保留。

```yaml
terminal:
  backend: ssh
  persistent_shell: true           # 保持长期 bash 会话（默认：true）
```

**必需的环境变量：**

```bash
TERMINAL_SSH_HOST=my-server.example.com
TERMINAL_SSH_USER=ubuntu
```

**可选：**

| 变量 | 默认值 | 描述 |
|----------|---------|-------------|
| `TERMINAL_SSH_PORT` | `22` | SSH 端口 |
| `TERMINAL_SSH_KEY` | （系统默认） | SSH 私钥路径 |
| `TERMINAL_SSH_PERSISTENT` | `true` | 启用持久 shell |

**工作原理：** 在初始化时使用 `BatchMode=yes` 和 `StrictHostKeyChecking=accept-new` 连接。持久 shell 在远程主机上保持单个 `bash -l` 进程存活，通过临时文件通信。需要 `stdin_data` 或 `sudo` 的命令自动回退到一次性模式。

### Modal 后端

在 [Modal](https://modal.com) 云沙箱中运行命令。每个任务获得一个具有可配置 CPU、内存和磁盘的隔离 VM。文件系统可以跨会话快照/恢复。

```yaml
terminal:
  backend: modal
  container_cpu: 1                 # CPU 核心数
  container_memory: 5120           # MB（5GB）
  container_disk: 51200            # MB（50GB）
  container_persistent: true       # 快照/恢复文件系统
```

**要求：** `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET` 环境变量，或 `~/.modal.toml` 配置文件。

**持久化：** 启用时，沙箱文件系统在清理时快照化，并在下次会话时恢复。快照跟踪在 `~/.hermes/modal_snapshots.json` 中。这保留文件系统状态，而非实时进程、PID 空间或后台作业。

**凭证文件：** 自动从 `~/.hermes/` 挂载（OAuth token 等），并在每次命令前同步。

### Daytona 后端

在 [Daytona](https://daytona.io) 托管工作区中运行命令。支持停止/恢复以实现持久化。

```yaml
terminal:
  backend: daytona
  container_cpu: 1                 # CPU 核心数
  container_memory: 5120           # MB → 转换为 GiB
  container_disk: 10240            # MB → 转换为 GiB（最大 10 GiB）
  container_persistent: true       # 停止/恢复而非删除
```

**要求：** `DAYTONA_API_KEY` 环境变量。

**持久化：** 启用时，沙箱在清理时停止（不删除），并在下次会话时恢复。沙箱名称遵循 `hermes-{task_id}` 模式。

**磁盘限制：** Daytona 强制执行 10 GiB 最大值。超过此值的请求会被上限并发出警告。

### Vercel Sandbox 后端

在 [Vercel Sandbox](https://vercel.com/docs/vercel-sandbox) 云 microVM 中运行命令。Hermes 使用正常的终端和文件工具接口；没有 Vercel 特定的面向模型的工具。

```yaml
terminal:
  backend: vercel_sandbox
  vercel_runtime: node24          # node24 | node22 | python3.13
  cwd: /vercel/sandbox            # 默认工作区根目录
  container_persistent: true      # 快照/恢复文件系统
  container_disk: 51200           # 仅共享默认值；不支持自定义磁盘
```

**必需安装：** 安装可选的 SDK 额外包：

```bash
pip install 'hermes-agent[vercel]'
```

**必需认证：** 使用所有三个 `VERCEL_TOKEN`、`VERCEL_PROJECT_ID` 和 `VERCEL_TEAM_ID` 配置 access-token 认证。这是部署和在 Render、Railway、Docker 等主机上运行正常长期 Hermes 进程的推荐设置。

对于一次性本地开发，Hermes 也接受短期 Vercel OIDC token：

```bash
VERCEL_OIDC_TOKEN="$(vc project token <project-name>)" hermes chat
```

从已链接的 Vercel 项目目录中，您可以省略项目名称：

```bash
VERCEL_OIDC_TOKEN="$(vc project token)" hermes chat
```

OIDC token 是短期的，不应作为文档化的部署路径使用。

**运行时：** `terminal.vercel_runtime` 支持 `node24`、`node22` 和 `python3.13`。如果未设置，Hermes 默认使用 `node24`。

**持久化：** 当 `container_persistent: true` 时，Hermes 在清理期间快照沙箱文件系统，并在以后为同一任务恢复沙箱时从该快照恢复。快照内容可以包括已复制到沙箱的 Hermes 同步凭证、技能和缓存文件。这仅保留文件系统状态；不保留实时沙箱身份、PID 空间、shell 状态或运行的后台进程。

**后台命令：** `terminal(background=true)` 使用 Hermes 的通用非本地后台进程流程。当沙箱存活时，您可以通过正常的进程工具生成、轮询、等待、查看日志和终止进程。Hermes 在清理或重启后不提供原生的 Vercel 分离进程恢复。

**磁盘大小：** Vercel Sandbox 目前不支持 Hermes 的 `container_disk` 资源旋钮。保留 `container_disk` 未设置或处于共享默认值 `51200`；非默认值会导致诊断和后端创建失败，而不是被静默忽略。

### Singularity/Apptainer 后端

在 [Singularity/Apptainer](https://apptainer.org) 容器中运行命令。专为 Docker 不可用的 HPC 集群和共享机器设计。

```yaml
terminal:
  backend: singularity
  singularity_image: "docker://nikolaik/python-nodejs:python3.11-nodejs20"
  container_cpu: 1                 # CPU 核心数
  container_memory: 5120           # MB
  container_persistent: true       # 可写覆盖层跨会话持久化
```

**要求：** `$PATH` 中有 `apptainer` 或 `singularity` 二进制文件。

**镜像处理：** Docker URL（`docker://...`）会自动转换为 SIF 文件并缓存。现有的 `.sif` 文件直接使用。

**临时目录：** 解析顺序：`TERMINAL_SCRATCH_DIR` → `TERMINAL_SANDBOX_DIR/singularity` → `/scratch/$USER/hermes-agent`（HPC 惯例） → `~/.hermes/sandboxes/singularity`。

**隔离：** 使用 `--containall --no-home` 进行完整命名空间隔离，不挂载宿主机 home 目录。

### 常见终端后端问题

如果终端命令立即失败或终端工具被报告为禁用：

- **Local** —— 无特殊要求。入门时最安全的默认选项。
- **Docker** —— 运行 `docker version` 验证 Docker 是否正常工作。如果失败，修复 Docker 或 `hermes config set terminal.backend local`。
- **SSH** —— `TERMINAL_SSH_HOST` 和 `TERMINAL_SSH_USER` 都必须设置。如果任一缺失，Hermes 会记录明确的错误。
- **Modal** —— 需要 `MODAL_TOKEN_ID` 环境变量或 `~/.modal.toml`。运行 `hermes doctor` 检查。
- **Daytona** —— 需要 `DAYTONA_API_KEY`。Daytona SDK 处理服务器 URL 配置。
- **Singularity** —— 需要 `$PATH` 中有 `apptainer` 或 `singularity`。在 HPC 集群上常见。

如有疑问，将 `terminal.backend` 改回 `local` 并先验证命令在那里可以运行。

### 拆卸时的远程到宿主机文件同步

对于 **SSH**、**Modal** 和 **Daytona** 后端（智能体工作树位于与运行 Hermes 的宿主机不同的机器上的任何地方），Hermes 跟踪智能体在远程沙箱内接触的文件，并在会话拆卸/沙箱清理时，将修改的文件**同步回宿主机**，保存在 `~/.hermes/cache/remote-syncs/<session-id>/` 下。

- 触发时机：会话关闭、`/new`、`/reset`、网关消息超时、当子智能体使用远程后端时的 `delegate_task` 子智能体完成。
- 覆盖智能体修改的整个树，而不仅仅是它显式打开的文件。添加、编辑和删除都会被捕获。
- 远程沙箱可能已被拆除，当您去查看时；本地 `~/.hermes/cache/remote-syncs/…` 副本是智能体更改的权威记录。
- 大型二进制输出（模型检查点、原始数据集）受大小限制——同步跳过超过 `file_sync_max_mb`（默认 `100`）的文件。如果预期有更大的产物，请提高该值。

```yaml
terminal:
  file_sync_max_mb: 100     # 默认——同步最大 100 MB 的单个文件
  file_sync_enabled: true   # 默认——设为 false 以完全跳过同步
```

这就是您如何在会话结束后从被销毁的临时云沙箱中恢复结果，而无需让智能体显式 `scp` 或 `modal volume put` 每个产物。

### Docker 卷挂载

使用 Docker 后端时，`docker_volumes` 允许您与容器共享宿主机目录。每个条目使用标准 Docker `-v` 语法：`host_path:container_path[:options]`。

```yaml
terminal:
  backend: docker
  docker_volumes:
    - "/home/user/projects:/workspace/projects"   # 读写（默认）
    - "/home/user/datasets:/data:ro"              # 只读
    - "/home/user/.hermes/cache/documents:/output" # 网关可见的导出
```

这适用于：
- **向智能体提供文件**（数据集、配置、参考代码）
- **从智能体接收文件**（生成的代码、报告、导出）
- **共享工作区**，您和智能体都访问相同的文件

如果您使用消息网关并希望智能体通过 `MEDIA:/...` 发送生成的文件，请使用专用的宿主机可见导出挂载，例如 `/home/user/.hermes/cache/documents:/output`。

- 在 Docker 内将文件写入 `/output/...`
- 在 `MEDIA:` 中发出**宿主机路径**，例如：
  `MEDIA:/home/user/.hermes/cache/documents/report.txt`
- **不要**发出 `/workspace/...` 或 `/output/...`，除非该确切路径也存在于网关进程的宿主机上

:::warning
YAML 重复键会静默覆盖前面的键。如果您已经有一个 `docker_volumes:` 块，将新挂载合并到同一个列表中，而不是在文件后面添加另一个 `docker_volumes:` 键。
:::

也可以通过环境变量设置：`TERMINAL_DOCKER_VOLUMES='["/host:/container"]'`（JSON 数组）。

### Docker 凭证转发

默认情况下，Docker 终端会话不会继承任意的宿主机凭证。如果您需要在容器内使用特定 token，请将其添加到 `terminal.docker_forward_env`。

```yaml
terminal:
  backend: docker
  docker_forward_env:
    - "GITHUB_TOKEN"
    - "NPM_TOKEN"
```

Hermes 首先从您当前的 shell 解析每个列出的变量，如果它是通过 `hermes config set` 保存的，则回退到 `~/.hermes/.env`。

:::warning
`docker_forward_env` 中列出的任何内容都会对容器内运行的命令可见。仅转发您愿意暴露给终端会话的凭证。
:::

### 以宿主机用户运行容器

默认情况下 Docker 容器以 `root`（UID 0）运行。在 `/workspace` 或其他绑定挂载内创建的文件在宿主机上最终归 root 所有，因此会话后您必须 `sudo chown` 它们才能在宿主机编辑器中编辑。`terminal.docker_run_as_host_user` 标志解决了这个问题：

```yaml
terminal:
  backend: docker
  docker_run_as_host_user: true   # 默认：false
```

启用后，Hermes 会在 `docker run` 命令后追加 `--user $(id -u):$(id -g)`，因此写入绑定挂载目录（`/workspace`、`/root`、`docker_volumes` 中的任何内容）的文件归您的宿主机用户所有，而非 root。权衡：容器不能再执行 `apt install` 或写入 root 拥有的路径如 `/root/.npm`——如果您需要两者兼顾，请使用其 `HOME` 归非 root 用户所有的基础镜像（或在镜像构建时添加所需工具）。

保留 `false`（默认）以保持向后兼容的行为。当您的流程主要是"编辑挂载的宿主机文件"且您厌倦了 `sudo chown -R` 时，请开启它。

### 可选：将启动目录挂载到 `/workspace`

Docker 沙箱默认保持隔离。除非您明确选择加入，Hermes 不会将您当前的宿主机工作目录传递到容器中。

在 `config.yaml` 中启用：

```yaml
terminal:
  backend: docker
  docker_mount_cwd_to_workspace: true
```

启用后：
- 如果您从 `~/projects/my-app` 启动 Hermes，该宿主机目录会绑定挂载到 `/workspace`
- Docker 后端从 `/workspace` 启动
- 文件工具和终端命令都看到相同的挂载项目

禁用时，`/workspace` 保持沙箱所有，除非您通过 `docker_volumes` 显式挂载某些内容。

安全权衡：
- `false` 保留沙箱边界
- `true` 让沙箱直接访问您启动 Hermes 的目录

仅在您有意让容器处理实时宿主机文件时使用选择加入。

### 持久 Shell

默认情况下，每个终端命令在自己的子进程中运行——工作目录、环境变量和 shell 变量在命令之间重置。启用**持久 shell** 后，单个长期运行的 bash 进程在 `execute()` 调用之间保持存活，因此状态在命令之间保留。

这对 **SSH 后端**最有用，因为它还消除了每条命令的连接开销。持久 shell **默认对 SSH 启用**，对 local 后端禁用。

```yaml
terminal:
  persistent_shell: true   # 默认——为 SSH 启用持久 shell
```

禁用：

```bash
hermes config set terminal.persistent_shell false
```

**跨命令保留的内容：**
- 工作目录（`cd /tmp` 对下一条命令保持有效）
- 导出的环境变量（`export FOO=bar`）
- Shell 变量（`MY_VAR=hello`）

**优先级：**

| 级别 | 变量 | 默认值 |
|-------|----------|---------|
| 配置 | `terminal.persistent_shell` | `true` |
| SSH 覆盖 | `TERMINAL_SSH_PERSISTENT` | 遵循配置 |
| Local 覆盖 | `TERMINAL_LOCAL_PERSISTENT` | `false` |

每个后端的环境变量优先级最高。如果您也想在 local 后端上使用持久 shell：

```bash
export TERMINAL_LOCAL_PERSISTENT=true
```

:::note
需要 `stdin_data` 或 sudo 的命令会自动回退到一次性模式，因为持久 shell 的 stdin 已被 IPC 协议占用。
:::

有关每个后端的详细信息，请参阅 [代码执行](features/code-execution.md) 和 [README 的终端部分](features/tools.md)。

## 技能设置 {#skill-settings}

技能可以通过其 SKILL.md frontmatter 声明自己的配置设置。这些是非机密值（路径、偏好、域设置），存储在 `config.yaml` 的 `skills.config` 命名空间下。

```yaml
skills:
  config:
    myplugin:
      path: ~/myplugin-data   # 示例——每个技能定义自己的键
```

**技能设置的工作原理：**

- `hermes config migrate` 扫描所有启用的技能，发现未配置的设置，并提示您
- `hermes config show` 在"技能设置"下显示所有技能设置及其所属技能
- 当技能加载时，其解析后的配置值会自动注入到技能上下文中

**手动设置值：**

```bash
hermes config set skills.config.myplugin.path ~/myplugin-data
```

有关在您自己的技能中声明配置设置的详细信息，请参阅 [创建技能 —— 配置设置](/developer-guide/creating-skills#config-settings-configyaml)。

### 智能体创建技能写入的守卫

当智能体使用 `skill_manage` 创建、编辑、修补或删除技能时，Hermes 可以可选地扫描新/更新内容中的危险关键字模式（凭证收集、明显的提示注入、外泄指令）。扫描器**默认关闭**——真正合法地接触 `~/.ssh/` 或提及 `$OPENAI_API_KEY` 的智能体工作流太频繁地触发启发式规则。如果您希望扫描器在智能体的技能写入落地前提示您，请重新开启：

```yaml
skills:
  guard_agent_created: true   # 默认：false
```

开启后，任何被标记的 `skill_manage` 写入都会以审批提示形式出现，并附带扫描器的理由。接受的写入会落地；拒绝的写入会向智能体返回解释性错误。

## 内存配置

```yaml
memory:
  memory_enabled: true
  user_profile_enabled: true
  memory_char_limit: 2200   # ~800 tokens
  user_char_limit: 1375     # ~500 tokens
```

## 文件读取安全

控制单次 `read_file` 调用可以返回多少内容。超过限制的读取会被拒绝，并提示智能体使用 `offset` 和 `limit` 获取更小的范围。这防止了单条读取压缩后的 JS 包或大型数据文件时淹没上下文窗口。

```yaml
file_read_max_chars: 100000  # 默认——~25-35K tokens
```

如果您使用大上下文窗口模型且频繁读取大文件，请提高它。为小上下文模型降低它以保持读取效率：

```yaml
# 大上下文模型（200K+）
file_read_max_chars: 200000

# 小本地模型（16K 上下文）
file_read_max_chars: 30000
```

智能体还会自动去重文件读取——如果同一文件区域被读取两次且文件未更改，会返回轻量级存根而非重新发送内容。这在上下文压缩后重置，因此智能体可以在内容被摘要后重新读取文件。

## 工具输出截断限制

三个相关的上限控制工具在 Hermes 截断之前可以返回多少原始输出：

```yaml
tool_output:
  max_bytes: 50000        # 终端输出上限（字符）
  max_lines: 2000         # read_file 分页上限
  max_line_length: 2000   # read_file 行号视图中的每行上限
```

- **`max_bytes`** —— 当 `terminal` 命令产生超过此字符数的组合 stdout/stderr 时，Hermes 保留前 40% 和后 60%，并在它们之间插入 `[OUTPUT TRUNCATED]` 通知。默认 `50000`（在典型 tokenizer 中约 12-15K tokens）。
- **`max_lines`** —— 单次 `read_file` 调用的 `limit` 参数上限。超过此值的请求会被限制，因此单次读取无法淹没上下文窗口。默认 `2000`。
- **`max_line_length`** —— 当 `read_file` 输出行号视图时应用的每行上限。超过此长度的行会被截断为此字符数，后跟 `... [truncated]`。默认 `2000`。

在大上下文窗口模型上提高限制，它们可以承受每次调用更多的原始输出。为小上下文模型降低限制以保持工具结果紧凑：

```yaml
# 大上下文模型（200K+）
tool_output:
  max_bytes: 150000
  max_lines: 5000

# 小本地模型（16K 上下文）
tool_output:
  max_bytes: 20000
  max_lines: 500
```

## 全局工具集禁用

要在一个地方抑制 CLI 和每个网关平台的特定工具集，请在 `agent.disabled_toolsets` 下列出它们的名称：

```yaml
agent:
  disabled_toolsets:
    - memory       # 隐藏内存工具 + MEMORY_GUIDANCE 注入
    - web          # 任何地方都没有 web_search / web_extract
```

这在每个平台的工具配置（由 `hermes tools` 写入的 `platform_toolsets`）之后应用，因此此处列出的工具集总是被移除——即使平台的保存配置仍然列出它。当您想要一个"在任何地方关闭 X"的单一开关，而不是在 `hermes tools` UI 中编辑 15+ 平台行时，请使用此功能。

保留列表为空，或省略该键，是无操作。

## Git 工作树隔离

为在同一仓库上并行运行多个智能体启用隔离的 git 工作树：

```yaml
worktree: true    # 始终创建工作树（等同于 hermes -w）
# worktree: false # 默认——仅在传递 -w 标志时
```

启用后，每个 CLI 会话在 `.worktrees/` 下创建一个带有自己分支的新工作树。智能体可以编辑文件、提交、推送和创建 PR，而不会相互干扰。干净的工作树在退出时移除；脏的工作树保留以供手动恢复。

您还可以通过仓库根目录下的 `.worktreeinclude` 列出要复制到工作树的 gitignored 文件：

```
# .worktreeinclude
.env
.venv/
node_modules/
```

## 上下文压缩 {#context-compression}

Hermes 自动压缩长对话以保持在模型的上下文窗口内。压缩摘要器是一个单独的 LLM 调用——您可以将其指向任何提供商或端点。

所有压缩设置都存在于 `config.yaml` 中（无环境变量）。

### 完整参考

```yaml
compression:
  enabled: true                                     # 开关压缩
  threshold: 0.50                                   # 在此上下文限制百分比处压缩
  target_ratio: 0.20                                # 保留为最近尾部的阈值比例
  protect_last_n: 20                                # 保持未压缩的最近消息最小数量
  hygiene_hard_message_limit: 400                   # 网关安全阀——见下文

# 摘要模型/提供商配置在 auxiliary 下：
auxiliary:
  compression:
    model: ""                                       # 空 = 使用主聊天模型。覆盖例如 "google/gemini-3-flash-preview" 以获得更便宜/更快的压缩。
    provider: "auto"                                # 提供商："auto"、"openrouter"、"nous"、"codex"、"main" 等。
    base_url: null                                  # 自定义 OpenAI-compatible 端点（覆盖提供商）
```

:::info 遗留配置迁移
带有 `compression.summary_model`、`compression.summary_provider` 和 `compression.summary_base_url` 的旧配置会在首次加载时自动迁移到 `auxiliary.compression.*`（配置版本 17）。无需手动操作。
:::

`hygiene_hard_message_limit` 是一个仅网关的**预压缩安全阀**。包含数千条消息的失控会话可能在正常的上下文百分比阈值触发之前达到模型上下文限制；当消息数量超过此上限时，Hermes 会强制压缩，无论 token 使用量如何。默认 `400`——对于非常长会话是正常的平台提高它，降低它以强制更积极的压缩。在运行中的网关上编辑此值会在下一条消息时生效（见下文）。

:::tip 网关热重载压缩和上下文长度
在最近版本中，在运行中的网关上编辑 `model.context_length` 或任何 `compression.*` 键会在下一条消息时生效——无需网关重启、无需 `/reset`、无需会话轮换。缓存的智能体签名包含这些键，因此网关在看到更改时会透明地重建智能体。API 密钥和工具/技能配置仍然需要通常的重载路径。
:::

### 常见设置

**默认（自动检测）—— 无需配置：**
```yaml
compression:
  enabled: true
  threshold: 0.50
```
使用您的主提供商和主模型。如果希望压缩使用比主聊天模型更便宜的模型，请按任务覆盖（例如 `auxiliary.compression.provider: openrouter` + `model: google/gemini-2.5-flash`）。

**强制特定提供商**（基于 OAuth 或 API 密钥）：
```yaml
auxiliary:
  compression:
    provider: nous
    model: gemini-3-flash
```
适用于任何提供商：`nous`、`openrouter`、`codex`、`anthropic`、`main` 等。

**自定义端点**（自托管、Ollama、zai、DeepSeek 等）：
```yaml
auxiliary:
  compression:
    model: glm-4.7
    base_url: https://api.z.ai/api/coding/paas/v4
```
指向自定义 OpenAI-compatible 端点。使用 `OPENAI_API_KEY` 进行认证。

### 三个旋钮如何交互

| `auxiliary.compression.provider` | `auxiliary.compression.base_url` | 结果 |
|---------------------|---------------------|--------|
| `auto`（默认） | 未设置 | 自动检测最佳可用提供商 |
| `nous` / `openrouter` / 等 | 未设置 | 强制该提供商，使用其认证 |
| 任何 | 已设置 | 直接使用自定义端点（忽略提供商） |

:::warning 摘要模型上下文长度要求
摘要模型**必须**具有至少与主智能体模型一样大的上下文窗口。压缩器将对话的完整中间部分发送给摘要模型——如果该模型的上下文窗口小于主模型的，摘要调用将失败并出现上下文长度错误。发生这种情况时，中间轮次会在**没有摘要的情况下被丢弃**，静默丢失对话上下文。如果您覆盖了模型，请验证其上下文长度是否达到或超过主模型的。
:::

## 上下文引擎

上下文引擎控制当接近模型的 token 限制时如何管理对话。内置的 `compressor` 引擎使用有损摘要（请参阅 [上下文压缩](/developer-guide/context-compression-and-caching)）。插件引擎可以用替代策略替换它。

```yaml
context:
  engine: "compressor"    # 默认——内置有损摘要
```

使用插件引擎（例如，LCM 用于无损上下文管理）：

```yaml
context:
  engine: "lcm"          # 必须匹配插件名称
```

插件引擎**永远不会自动激活**——您必须显式将 `context.engine` 设置为插件名称。可用的引擎可以通过 `hermes plugins` → Provider Plugins → Context Engine 浏览和选择。

有关类似的内存插件单选系统，请参阅 [内存提供商](/user-guide/features/memory-providers)。

## 迭代预算压力

当智能体处理具有多个工具调用的复杂任务时，它可能会在意识到预算不足之前就用完迭代预算（默认：90 轮）。预算压力会在接近限制时自动警告模型：

| 阈值 | 级别 | 模型看到的内容 |
|-----------|-------|---------------------|
| **70%** | 注意 | `[BUDGET: 63/90. 27 iterations left. Start consolidating.]` |
| **90%** | 警告 | `[BUDGET WARNING: 81/90. Only 9 left. Respond NOW.]` |

警告被注入到最后一个工具结果的 JSON 中（作为 `_budget_warning` 字段），而非作为单独的消息——这保留了提示缓存，不会破坏对话结构。

```yaml
agent:
  max_turns: 90                # 每轮对话的最大迭代次数（默认：90）
  api_max_retries: 3           # 备用启动前每个提供商的重试次数（默认：3）
```

预算压力默认启用。智能体自然地将警告视为工具结果的一部分，鼓励它在用完迭代之前整合工作并交付响应。

当迭代预算完全耗尽时，CLI 会向用户显示通知：`⚠ Iteration budget reached (90/90) — response may be incomplete`。如果预算在活动工作期间耗尽，智能体会生成已完成工作的摘要，然后停止。

`agent.api_max_retries` 控制 Hermes 在临时错误（速率限制、连接断开、5xx）上重试提供商 API 调用的次数，**在**备用提供商切换启动之前。默认是 `3`——总共四次尝试。如果您配置了[备用提供商](/user-guide/features/fallback-providers)并希望更快故障转移，请将其降至 `0`，这样主端点上的第一次临时错误就会立即交给备用，而不是对该不稳定端点反复重试。

### API 超时

Hermes 为流式传输设置了单独的超时层，以及非流式调用的过时检测器。过时检测器仅在您将它们保留为隐式默认值时，才会为本地提供商自动调整。

| 超时 | 默认 | 本地提供商 | 配置 / 环境 |
|---------|---------|----------------|--------------|
| Socket 读取超时 | 120s | 自动提高到 1800s | `HERMES_STREAM_READ_TIMEOUT` |
| 流式过时检测 | 180s | 自动禁用 | `HERMES_STREAM_STALE_TIMEOUT` |
| 非流式过时检测 | 300s | 保留隐式默认值时自动禁用 | `providers.<id>.stale_timeout_seconds` 或 `HERMES_API_CALL_STALE_TIMEOUT` |
| API 调用（非流式） | 1800s | 不变 | `providers.<id>.request_timeout_seconds` / `timeout_seconds` 或 `HERMES_API_TIMEOUT` |

**Socket 读取超时**控制 httpx 等待提供商下一个数据块的时间。本地 LLM 在产生第一个 token 之前可能需要数分钟进行大上下文的预填充，因此 Hermes 在检测到本地端点时会将其提高到 30 分钟。如果您显式设置 `HERMES_STREAM_READ_TIMEOUT`，则无论端点检测如何都会使用该值。

**流式过时检测**会终止接收 SSE 保活 ping 但没有实际内容的连接。这对本地提供商完全禁用，因为它们在预填充期间不发送保活 ping。

**非流式过时检测**会终止长时间没有响应的非流式调用。默认情况下，Hermes 在本地端点上禁用此功能，以避免长预填充期间的误报。如果您显式设置 `providers.<id>.stale_timeout_seconds`、`providers.<id>.models.<model>.stale_timeout_seconds` 或 `HERMES_API_CALL_STALE_TIMEOUT`，即使在本地端点上也会遵守该显式值。

## 上下文压力警告

与迭代预算压力分开，上下文压力跟踪对话距离**压缩阈值**有多近——即上下文压缩触发以摘要旧消息的时点。这有助于您和智能体了解对话何时变长。

| 进度 | 级别 | 发生什么 |
|----------|-------|-------------|
| **≥ 60%** 到阈值 | 信息 | CLI 显示青色进度条；网关发送信息性通知 |
| **≥ 85%** 到阈值 | 警告 | CLI 显示粗体黄色条；网关警告压缩即将发生 |

在 CLI 中，上下文压力以工具输出流中的进度条形式出现：

```
  ◐ context ████████████░░░░░░░░ 62% to compaction  48k threshold (50%) · approaching compaction
```

在消息平台上，会发送纯文本通知：

```
◐ Context: ████████████░░░░░░░░ 62% to compaction (threshold: 50% of window).
```

如果自动压缩被禁用，警告会告诉您上下文可能会被截断。

上下文压力是自动的——无需配置。它纯粹作为面向用户的通知触发，不会修改消息流或向模型的上下文注入任何内容。

## 凭证池策略

当您对同一提供商有多个 API 密钥或 OAuth token 时，配置轮换策略：

```yaml
credential_pool_strategies:
  openrouter: round_robin    # 均匀循环密钥
  anthropic: least_used      # 始终选择使用最少的密钥
```

选项：`fill_first`（默认）、`round_robin`、`least_used`、`random`。有关完整文档，请参阅 [凭证池](/user-guide/features/credential-pools)。

## 辅助模型

Hermes 使用"辅助"模型处理图像分析、网页摘要、浏览器截图分析、会话标题生成和上下文压缩等副任务。默认情况下（`auxiliary.*.provider: "auto"`），Hermes 将每个辅助任务路由到您的**主聊天模型**——您在 `hermes model` 中选择的同一提供商/模型。您无需配置任何内容即可开始，但要注意在昂贵的推理模型（Opus、MiniMax M2.7 等）上，辅助任务会增加显著成本。如果您希望副任务便宜且快速，无论主模型如何，请显式设置 `auxiliary.<task>.provider` 和 `auxiliary.<task>.model`（例如，OpenRouter 上的 Gemini Flash 用于 vision 和 web extract）。

:::note 为何 "auto" 使用您的主模型
早期版本将聚合器用户（OpenRouter、Nous Portal）分流到提供商端的廉价默认模型。这令人惊讶——付费订阅聚合器的用户会看到不同的模型处理其辅助流量。`auto` 现在对每个人都使用主模型，而 `config.yaml` 中的每个任务覆盖仍然优先（请参阅下面的[完整辅助配置参考](#full-auxiliary-config-reference)）。
:::

### 交互式配置辅助模型

无需手动编辑 YAML，运行 `hermes model` 并从菜单中选择 **"Configure auxiliary models"**。您将获得交互式每任务选择器：

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

选择一个任务，选择一个提供商（OAuth 流程会打开浏览器；API 密钥提供商会提示），选择一个模型。更改会持久化到 `config.yaml` 中的 `auxiliary.<task>.*`。与主模型选择器使用相同的机制——无需学习额外的语法。

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

Hermes 中的每个模型插槽——辅助任务、压缩、备用——都使用相同的三个旋钮：

| 键 | 作用 | 默认 |
|-----|-------------|---------|
| `provider` | 用于认证和路由的提供商 | `"auto"` |

<a id="full-auxiliary-config-reference"></a>

## 完整的辅助模型/任务配置参考 {#full-auxiliary-config-reference}

该章节系统说明 `auxiliary` 子配置的完整字段与推荐取值，便于按任务类型拆分模型与预算策略。

<a id="context-compression"></a>

## 上下文压缩（引用锚点）

该小节解释会话变长时的上下文压缩策略、触发条件与常见调优方法。
| `model` | 请求的模型 | 提供商的默认模型 |
| `base_url` | 自定义 OpenAI-compatible 端点（覆盖提供商） | 未设置 |

当 `base_url` 设置时，Hermes 忽略提供商并直接调用该端点（使用 `api_key` 或 `OPENAI_API_KEY` 进行认证）。当仅设置 `provider` 时，Hermes 使用该提供商的内置认证和基础 URL。

辅助任务的可用提供商：`auto`、`main`，以及[提供商注册表](/reference/environment-variables)中的任何提供商——`openrouter`、`nous`、`openai-codex`、`copilot`、`copilot-acp`、`anthropic`、`gemini`、`google-gemini-cli`、`qwen-oauth`、`zai`、`kimi-coding`、`kimi-coding-cn`、`minimax`、`minimax-cn`、`minimax-oauth`、`deepseek`、`nvidia`、`xai`、`ollama-cloud`、`alibaba`、`bedrock`、`huggingface`、`arcee`、`xiaomi`、`kilocode`、`opencode-zen`、`opencode-go`、`ai-gateway`、`azure-foundry`——或 `custom_providers` 列表中的任何命名自定义提供商（例如 `provider: "beans"`）。

:::tip MiniMax OAuth
`minimax-oauth` 通过浏览器 OAuth 登录（无需 API 密钥）。运行 `hermes model` 并选择 **MiniMax (OAuth)** 进行认证。辅助任务自动使用 `MiniMax-M2.7-highspeed`。请参阅 [MiniMax OAuth 指南](/guides/minimax-oauth)。
:::

:::warning `"main"` 仅用于辅助任务
`"main"` 提供商选项表示"使用我的主智能体使用的任何提供商"——它仅在 `auxiliary:`、`compression:` 和 `fallback_model:` 配置内有效。它不是您顶层 `model.provider` 设置的有效值。如果您使用自定义 OpenAI-compatible 端点，请在 `model:` 部分设置 `provider: custom`。有关所有主模型提供商选项，请参阅 [AI 提供商](/integrations/providers)。
:::

### 完整辅助配置参考

```yaml
auxiliary:
  # 图像分析（vision_analyze 工具 + 浏览器截图）
  vision:
    provider: "auto"           # "auto"、"openrouter"、"nous"、"codex"、"main" 等。
    model: ""                  # 例如 "openai/gpt-4o"、"google/gemini-2.5-flash"
    base_url: ""               # 自定义 OpenAI-compatible 端点（覆盖提供商）
    api_key: ""                # base_url 的 API 密钥（回退到 OPENAI_API_KEY）
    timeout: 120               # 秒——LLM API 调用超时；vision 负载需要宽裕的超时
    download_timeout: 30       # 秒——图像 HTTP 下载；为慢速连接提高

  # 网页摘要 + 浏览器页面文本提取
  web_extract:
    provider: "auto"
    model: ""                  # 例如 "google/gemini-2.5-flash"
    base_url: ""
    api_key: ""
    timeout: 360               # 秒（6分钟）——每次尝试的 LLM 摘要

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
    max_concurrency: 3       # 限制并行摘要以减少请求突发 429
    extra_body: {}           # 提供商特定的 OpenAI-compatible 请求字段

  # 技能中心——技能匹配和搜索
  skills_hub:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30

  # MCP 工具分发
  mcp:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30

  # 看板分类指定器——`hermes kanban specify <id>`（或
  # 仪表板上 Triage 列卡片的 ✨ Specify 按钮）使用此
  # 插槽将一句话扩展为具体规范并将
  # 任务提升为 `todo`。便宜快速的模型在这里效果很好；规范扩展
  # 很短，不需要推理深度。
  triage_specifier:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 120
```

:::tip
每个辅助任务都有可配置的 `timeout`（秒）。默认值：vision 120s、web_extract 360s、approval 30s、compression 120s。如果您使用慢速本地模型处理辅助任务，请提高这些值。Vision 还有单独的 `download_timeout`（默认 30s）用于 HTTP 图像下载——为慢速连接或自托管图像服务器提高它。
:::

:::info
上下文压缩有自己的 `compression:` 块用于阈值，以及一个 `auxiliary.compression:` 块用于模型/提供商设置——请参阅上面的[上下文压缩](#context-compression)。备用模型使用 `fallback_model:` 块——请参阅 [备用提供商](/integrations/providers#fallback-providers)。所有三个都遵循相同的 provider/model/base_url 模式。
:::

### 会话搜索调优

如果您为 `auxiliary.session_search` 使用重推理模型，Hermes 现在提供两个内置控制：

- `auxiliary.session_search.max_concurrency`：限制 Hermes 同时摘要的匹配会话数量
- `auxiliary.session_search.extra_body`：在摘要调用上转发提供商特定的 OpenAI-compatible 请求字段

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

当您的提供商对请求突发进行速率限制且您希望 `session_search` 以一些并行性换取稳定性时，使用 `max_concurrency`。

仅当您的提供商记录了您希望 Hermes 为该任务传递的 OpenAI-compatible 请求体字段时，使用 `extra_body`。Hermes 按原样转发对象。

:::warning
`extra_body` 仅在您的提供商实际支持您发送的字段时才有效。如果提供商没有暴露原生的 OpenAI-compatible 推理关闭标志，Hermes 无法代表它合成一个。
:::

### OpenRouter 路由与 Pareto Code 用于辅助任务

当辅助任务解析到 OpenRouter（显式或通过 `provider: "main"` 而您的主智能体在 OpenRouter 上）时，主智能体的 `provider_routing` 和 `openrouter.min_coding_score` 设置**不会传播**——按设计，每个辅助任务是独立的。要为特定辅助任务设置 OpenRouter 提供商偏好或使用 [Pareto Code 路由器](/integrations/providers#openrouter-pareto-code-router)，请通过 `extra_body` 按任务设置：

```yaml
auxiliary:
  compression:
    provider: openrouter
    model: openrouter/pareto-code         # 为此任务使用 Pareto Code 路由器
    extra_body:
      provider:                            # OpenRouter 提供商路由偏好
        order: [anthropic, google]         # 按顺序尝试这些提供商
        sort: throughput                   # 或 "price" | "latency"
        # only: [anthropic]                # 限制到特定提供商
        # ignore: [deepinfra]              # 排除特定提供商
      plugins:                             # OpenRouter Pareto Code 路由器旋钮
        - id: pareto-router
          min_coding_score: 0.5            # 0.0–1.0；越高 = 更强的编码能力
```

形状镜像 OpenRouter 在聊天补全请求体中接受的内容。Hermes 按原样转发整个 `extra_body`，因此 [openrouter.ai/docs](https://openrouter.ai/docs) 中记录的任何其他 OpenRouter 请求体字段都以相同方式工作。

### 更改 Vision 模型

要使用 GPT-4o 替代 Gemini Flash 进行图像分析：

```yaml
auxiliary:
  vision:
    model: "openai/gpt-4o"
```

或通过环境变量（在 `~/.hermes/.env` 中）：

```bash
AUXILIARY_VISION_MODEL=openai/gpt-4o
```

### 提供商选项

这些选项适用于**辅助任务配置**（`auxiliary:`、`compression:`、`fallback_model:`），不适用于您的主 `model.provider` 设置。

| 提供商 | 描述 | 要求 |
|----------|-------------|-------------|
| `"auto"` | 最佳可用（默认）。Vision 尝试 OpenRouter → Nous → Codex。 | — |
| `"openrouter"` | 强制 OpenRouter——路由到任何模型（Gemini、GPT-4o、Claude 等） | `OPENROUTER_API_KEY` |
| `"nous"` | 强制 Nous Portal | `hermes auth` |
| `"codex"` | 强制 Codex OAuth（ChatGPT 账户）。支持 vision（gpt-5.3-codex）。 | `hermes model` → Codex |
| `"minimax-oauth"` | 强制 MiniMax OAuth（浏览器登录，无需 API 密钥）。辅助任务使用 MiniMax-M2.7-highspeed。 | `hermes model` → MiniMax (OAuth) |
| `"main"` | 使用您活动的自定义/主端点。这可以来自 `OPENAI_BASE_URL` + `OPENAI_API_KEY` 或通过 `hermes model` / `config.yaml` 保存的自定义端点。适用于 OpenAI、本地模型或任何 OpenAI-compatible API。**仅用于辅助任务——对 `model.provider` 无效。** | 自定义端点凭证 + 基础 URL |

主提供商目录中的直接 API 密钥提供商在您希望副任务绕过默认路由器时也适用于此处。一旦配置了 `GMI_API_KEY`，`gmi` 就有效：

```yaml
auxiliary:
  compression:
    provider: "gmi"
    model: "anthropic/claude-opus-4.6"
```

对于 GMI 辅助路由，使用 GMI 的 `/v1/models` 端点返回的精确模型 ID。

### 常见设置

**使用直接自定义端点**（比 `provider: "main"` 更清晰的本地/自托管 API）：
```yaml
auxiliary:
  vision:
    base_url: "http://localhost:1234/v1"
    api_key: "local-key"
    model: "qwen2.5-vl"
```

`base_url` 优先于 `provider`，因此这是将辅助任务路由到特定端点的最明确方式。对于直接端点覆盖，Hermes 使用配置的 `api_key` 或回退到 `OPENAI_API_KEY`；它不会为该自定义端点重用 `OPENROUTER_API_KEY`。

**使用 OpenAI API 密钥进行 vision：**
```yaml
# 在 ~/.hermes/.env 中：
# OPENAI_BASE_URL=https://api.openai.com/v1
# OPENAI_API_KEY=sk-...

auxiliary:
  vision:
    provider: "main"
    model: "gpt-4o"       # 或 "gpt-4o-mini" 以获得更便宜的选择
```

**使用 OpenRouter 进行 vision**（路由到任何模型）：
```yaml
auxiliary:
  vision:
    provider: "openrouter"
    model: "openai/gpt-4o"      # 或 "google/gemini-2.5-flash" 等。
```

**使用 Codex OAuth**（ChatGPT Pro/Plus 账户——无需 API 密钥）：
```yaml
auxiliary:
  vision:
    provider: "codex"     # 使用您的 ChatGPT OAuth token
    # 模型默认为 gpt-5.3-codex（支持 vision）
```

**使用 MiniMax OAuth**（浏览器登录，无需 API 密钥）：
```yaml
model:
  default: MiniMax-M2.7
  provider: minimax-oauth
  base_url: https://api.minimax.io/anthropic
```
运行 `hermes model` 并选择 **MiniMax (OAuth)** 以自动登录和设置。对于中国区域，基础 URL 将是 `https://api.minimaxi.com/anthropic`。有关完整流程，请参阅 [MiniMax OAuth 指南](/guides/minimax-oauth)。

**使用本地/自托管模型：**
```yaml
auxiliary:
  vision:
    provider: "main"      # 使用您活动的自定义端点
    model: "my-local-model"
```

`provider: "main"` 使用 Hermes 用于正常聊天的任何提供商——无论是命名自定义提供商（例如 `beans`）、内置提供商如 `openrouter`，还是遗留的 `OPENAI_BASE_URL` 端点。

:::tip
如果您使用 Codex OAuth 作为您的主模型提供商，vision 会自动工作——无需额外配置。Codex 包含在 vision 的自动检测链中。
:::

:::warning
**Vision 需要多模态模型。** 如果您设置 `provider: "main"`，请确保您的端点支持多模态/vision——否则图像分析将失败。
:::

### 环境变量（遗留）

辅助模型也可以通过环境变量配置。然而，`config.yaml` 是首选方法——它更容易管理，并支持所有选项包括 `base_url` 和 `api_key`。

| 设置 | 环境变量 |
|---------|---------------------|
| Vision 提供商 | `AUXILIARY_VISION_PROVIDER` |
| Vision 模型 | `AUXILIARY_VISION_MODEL` |
| Vision 端点 | `AUXILIARY_VISION_BASE_URL` |
| Vision API 密钥 | `AUXILIARY_VISION_API_KEY` |
| Web extract 提供商 | `AUXILIARY_WEB_EXTRACT_PROVIDER` |
| Web extract 模型 | `AUXILIARY_WEB_EXTRACT_MODEL` |
| Web extract 端点 | `AUXILIARY_WEB_EXTRACT_BASE_URL` |
| Web extract API 密钥 | `AUXILIARY_WEB_EXTRACT_API_KEY` |

压缩和备用模型设置仅限 config.yaml。

:::tip
运行 `hermes config` 查看您当前的辅助模型设置。仅当覆盖与默认值不同时才会显示。
:::

## 推理力度

控制模型在响应前进行多少"思考"：

```yaml
agent:
  reasoning_effort: ""   # 空 = medium（默认）。选项：none, minimal, low, medium, high, xhigh（最大）
```

未设置时（默认），推理力度默认为 "medium"——一个适用于大多数任务的平衡级别。设置值会覆盖它——更高的推理力度在复杂任务上给出更好的结果，但代价是更多的 token 和延迟。

您还可以在运行时使用 `/reasoning` 命令更改推理力度：

```
/reasoning           # 显示当前力度级别和显示状态
/reasoning high      # 将推理力度设为 high
/reasoning none      # 禁用推理
/reasoning show      # 在每条响应上方显示模型思考
/reasoning hide      # 隐藏模型思考
```

## 工具使用强制

某些模型偶尔会将意图操作描述为文本而非进行工具调用（"我会运行测试..." 而非实际调用终端）。工具使用强制注入系统提示引导，将模型引导回实际调用工具。

```yaml
agent:
  tool_use_enforcement: "auto"   # "auto" | true | false | ["model-substring", ...]
```

| 值 | 行为 |
|-------|----------|
| `"auto"`（默认） | 对匹配以下模型的模型启用：`gpt`、`codex`、`gemini`、`gemma`、`grok`。对所有其他模型禁用（Claude、DeepSeek、Qwen 等）。 |
| `true` | 无论模型如何始终启用。如果您注意到当前模型描述操作而非执行操作，这很有用。 |
| `false` | 无论模型如何始终禁用。 |
| `["gpt", "codex", "qwen", "llama"]` | 仅当模型名称包含列出的子字符串之一时启用（不区分大小写）。 |

### 注入的内容

启用时，三层引导可能会被添加到系统提示中：

1. **通用工具使用强制**（所有匹配的模型）——指示模型立即进行工具调用而非描述意图，持续工作直到任务完成，并且永远不要以承诺未来操作结束一轮。

2. **OpenAI 执行纪律**（仅限 GPT 和 Codex 模型）——解决 GPT 特定故障模式的额外引导：在部分结果上放弃工作、跳过先决条件查找、产生幻觉而非使用工具，以及在没有验证的情况下声明"完成"。

3. **Google 操作引导**（仅限 Gemini 和 Gemma 模型）——简洁性、绝对路径、并行工具调用和编辑前验证模式。

这些对用户是透明的，仅影响系统提示。已经可靠使用工具的模型（如 Claude）不需要此引导，这就是为什么 `"auto"` 排除它们。

### 何时开启

如果您使用的模型不在默认自动列表中，并且注意到它经常描述它*会*做什么而非实际去做，请设置 `tool_use_enforcement: true` 或将模型子字符串添加到列表中：

```yaml
agent:
  tool_use_enforcement: ["gpt", "codex", "gemini", "grok", "my-custom-model"]
```

## TTS 配置

```yaml
tts:
  provider: "edge"              # "edge" | "elevenlabs" | "openai" | "minimax" | "mistral" | "gemini" | "xai" | "neutts"
  speed: 1.0                    # 全局速度倍率（所有提供商的后备）
  edge:
    voice: "en-US-AriaNeural"   # 322 种声音，74 种语言
    speed: 1.0                  # 速度倍率（转换为速率百分比，例如 1.5 → +50%）
  elevenlabs:
    voice_id: "pNInz6obpgDQGcFmaJgB"
    model_id: "eleven_multilingual_v2"
  openai:
    model: "gpt-4o-mini-tts"
    voice: "alloy"              # alloy, echo, fable, onyx, nova, shimmer
    speed: 1.0                  # 速度倍率（由 API 限制在 0.25–4.0）
    base_url: "https://api.openai.com/v1"  # 用于 OpenAI-compatible TTS 端点的覆盖
  minimax:
    speed: 1.0                  # 语音速度倍率
    # base_url: ""              # 可选：用于 OpenAI-compatible TTS 端点的覆盖
  mistral:
    model: "voxtral-mini-tts-2603"
    voice_id: "c69964a6-ab8b-4f8a-9465-ec0925096ec8"  # Paul - Neutral（默认）
  gemini:
    model: "gemini-2.5-flash-preview-tts"   # 或 gemini-2.5-pro-preview-tts
    voice: "Kore"               # 30 种预置声音：Zephyr, Puck, Kore, Enceladus 等
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

这同时控制 `text_to_speech` 工具和语音模式中的语音回复（CLI 中的 `/voice tts` 或消息网关）。

**速度回退层级：** 提供商特定速度（例如 `tts.edge.speed`）→ 全局 `tts.speed` → `1.0` 默认值。设置全局 `tts.speed` 以在所有提供商之间应用统一速度，或按提供商覆盖以进行细粒度控制。

## 显示设置 {#display-settings}

```yaml
display:
  tool_progress: all      # off | new | all | verbose
  tool_progress_command: false  # 在消息网关中启用 /verbose 斜杠命令
  platforms: {}           # 每个平台的显示覆盖（见下文）
  tool_progress_overrides: {}  # 已弃用 —— 改用 display.platforms
  interim_assistant_messages: true  # 网关：将自然的中轮次助手更新作为单独消息发送
  skin: default           # 内置或自定义 CLI 皮肤（见 user-guide/features/skins）
  personality: "kawaii"  # 遗留的 cosmetic 字段，仍在某些摘要中显示
  compact: false          # 紧凑输出模式（更少空白）
  resume_display: full    # full（恢复时显示之前的消息）| minimal（仅一行）
  bell_on_complete: false # 智能体完成时播放终端响铃（非常适合长任务）
  show_reasoning: false   # 在每条响应上方显示模型推理/思考（使用 /reasoning show|hide 切换）
  streaming: false        # 将 token 流式传输到终端（实时输出）
  show_cost: false        # 在 CLI 状态栏中显示估计的 $ 成本
  tool_preview_length: 0  # 工具调用预览的最大字符数（0 = 无限制，显示完整路径/命令）
  runtime_footer:         # 网关：将运行时上下文页脚附加到最终回复
    enabled: false
    fields: ["model", "context_pct", "cwd"]
  language: en            # 静态消息的 UI 语言（审批提示、某些网关回复）。en | zh | ja | de | es | fr | tr | uk
```

### 静态消息的 UI 语言

`display.language` 设置翻译一小部分静态面向用户的消息——CLI 审批提示、少量网关斜杠命令回复（例如重启排空通知、"审批已过期"、"目标已清除"）。它**不会**翻译智能体响应、日志行、工具输出、错误回溯或斜杠命令描述——这些保持英文。如果您希望智能体本身用另一种语言回复，只需在提示或系统消息中告诉它。

支持的值：`en`（默认）、`zh`（简体中文）、`ja`（日语）、`de`（德语）、`es`（西班牙语）、`fr`（法语）、`tr`（土耳其语）、`uk`（乌克兰语）。未知值回退到英文。

您还可以通过 `HERMES_LANGUAGE` 环境变量按会话设置此项，它会覆盖配置值。

```yaml
display:
  language: zh   # CLI 审批提示以中文显示
```

| 模式 | 您看到的内容 |
|------|-------------|
| `off` | 静默——仅最终响应 |
| `new` | 仅在工具更改时显示工具指示器 |
| `all` | 每次工具调用附带简短预览（默认） |
| `verbose` | 完整参数、结果和调试日志 |

在 CLI 中，使用 `/verbose` 循环切换这些模式。要在消息平台（Telegram、Discord、Slack 等）中使用 `/verbose`，请在上述 `display` 部分中设置 `tool_progress_command: true`。然后该命令将循环切换模式并保存到配置。

### 运行时元数据页脚（仅限网关）

当 `display.runtime_footer.enabled: true` 时，Hermes 将一个小运行时上下文页脚附加到每轮网关对话的**最终**消息——与 CLI 状态栏中显示的相同信息（模型、上下文 %、cwd、会话持续时间、token、成本）。默认关闭；如果您的团队希望每条回复都包含来源信息，请按网关选择加入。

```yaml
display:
  runtime_footer:
    enabled: true
    fields: ["model", "context_pct", "cwd"]   # 任意组合：model, context_pct, cwd, duration, tokens, cost
```

`/footer` 斜杠命令在任何会话中运行时切换此功能。

附加到 Telegram/Discord/Slack 回复的示例页脚：

```
— claude-opus-4.7 · 12 次工具调用 · 2分14秒 · $0.042
```

只有每轮的**最终**消息获得页脚；中间更新保持干净。

### 每个平台的进度覆盖

不同平台有不同的详细程度需求。例如，Signal 无法编辑消息，因此每次进度更新都会变成单独的消息——很吵。使用 `display.platforms` 设置每个平台的模式：

```yaml
display:
  tool_progress: all          # 全局默认值
  platforms:
    signal:
      tool_progress: 'off'    # 在 Signal 上静默进度
    telegram:
      tool_progress: verbose  # 在 Telegram 上显示详细进度
    slack:
      tool_progress: 'off'    # 在共享 Slack 工作区中保持安静
```

没有覆盖的平台回退到全局 `tool_progress` 值。有效的平台键：`telegram`、`discord`、`slack`、`signal`、`whatsapp`、`matrix`、`mattermost`、`email`、`sms`、`homeassistant`、`dingtalk`、`feishu`、`wecom`、`weixin`、`bluebubbles`、`qqbot`。遗留的 `display.tool_progress_overrides` 键仍然加载以保持向后兼容，但已弃用并在首次加载时迁移到 `display.platforms`。

`interim_assistant_messages` 仅限网关。启用时，Hermes 将完成的中轮次助手更新作为单独的聊天消息发送。这与 `tool_progress` 独立，且不需要网关流式传输。

## 隐私

```yaml
privacy:
  redact_pii: false  # 从 LLM 上下文中去除 PII（仅限网关）
```

当 `redact_pii` 为 `true` 时，网关在支持的平台上将系统提示中的个人身份信息去除后再发送给 LLM：

| 字段 | 处理方式 |
|-------|-----------|
| 电话号码（WhatsApp/Signal 上的用户 ID） | 哈希化为 `user_<12-char-sha256>` |
| 用户 ID | 哈希化为 `user_<12-char-sha256>` |
| 聊天 ID | 数字部分哈希化，平台前缀保留（`telegram:<hash>`） |
| Home channel ID | 数字部分哈希化 |
| 用户名 / 用户名 | **不受影响**（用户选择，公开可见） |

**平台支持：** 去除适用于 WhatsApp、Signal 和 Telegram。Discord 和 Slack 被排除，因为它们的提及系统（`<@user_id>`）需要 LLM 上下文中的真实 ID。

哈希是确定性的——同一用户始终映射到同一哈希，因此模型仍能在群聊中区分用户。路由和交付在内部使用原始值。

## 语音转文本（STT）

```yaml
stt:
  provider: "local"            # "local" | "groq" | "openai" | "mistral"
  local:
    model: "base"              # tiny, base, small, medium, large-v3
  openai:
    model: "whisper-1"         # whisper-1 | gpt-4o-mini-transcribe | gpt-4o-transcribe
  # model: "whisper-1"         # 遗留回退键仍然被尊重
```

提供商行为：

- `local` 使用在您机器上运行的 `faster-whisper`。使用 `pip install faster-whisper` 单独安装。
- `groq` 使用 Groq 的 Whisper-compatible 端点并读取 `GROQ_API_KEY`。
- `openai` 使用 OpenAI 语音 API 并读取 `VOICE_TOOLS_OPENAI_KEY`。

如果请求的提供商不可用，Hermes 按此顺序自动回退：`local` → `groq` → `openai`。

Groq 和 OpenAI 模型覆盖是环境驱动的：

```bash
STT_GROQ_MODEL=whisper-large-v3-turbo
STT_OPENAI_MODEL=whisper-1
GROQ_BASE_URL=https://api.groq.com/openai/v1
STT_OPENAI_BASE_URL=https://api.openai.com/v1
```

## 语音模式（CLI）

```yaml
voice:
  record_key: "ctrl+b"         # CLI 内的按键通话键
  max_recording_seconds: 120    # 长录音的硬停止
  auto_tts: false               # 当 /voice on 时自动启用语音回复
  beep_enabled: true            # 在 CLI 语音模式中播放录音开始/停止提示音
  silence_threshold: 200        # 语音检测的 RMS 阈值
  silence_duration: 3.0         # 自动停止前的静音秒数
```

在 CLI 中使用 `/voice on` 启用麦克风模式，`record_key` 开始/停止录音，`/voice tts` 切换语音回复。有关端到端设置和平台特定行为，请参阅 [语音模式](/user-guide/features/voice-mode)。

## 流式传输

将 token 流式传输到终端或消息平台，而不是等待完整响应。

### CLI 流式传输

```yaml
display:
  streaming: true         # 将 token 实时流式传输到终端
  show_reasoning: true    # 同时流式传输推理/思考 token（可选）
```

启用时，响应在流式框中逐 token 出现。工具调用仍被静默捕获。如果提供商不支持流式传输，它会自动回退到正常显示。

### 网关流式传输（Telegram、Discord、Slack）

```yaml
streaming:
  enabled: true           # 启用渐进式消息编辑
  transport: edit         # "edit"（渐进式消息编辑）或 "off"
  edit_interval: 0.3      # 消息编辑之间的秒数
  buffer_threshold: 40    # 强制编辑刷新前的字符数
  cursor: " ▉"            # 流式传输期间显示的光标
  fresh_final_after_seconds: 60   # 当预览如此旧时发送新的最终消息（Telegram）；0 = 始终在原地编辑
```

启用时，机器人在第一个 token 时发送消息，然后在更多 token 到达时逐步编辑它。不支持消息编辑的平台（Signal、Email、Home Assistant）在第一次尝试时自动检测——流式传输被优雅地禁用，不会淹没消息。

对于没有渐进式 token 编辑的单独自然中轮次助手更新，设置 `display.interim_assistant_messages: true`。

**溢出处理：** 如果流式文本超过平台的消息长度限制（约 4096 字符），当前消息被最终化并自动开始新消息。

**新鲜最终消息（Telegram）：** Telegram 的 `editMessageText` 保留原始消息时间戳，因此长时间运行的流式回复即使在完成后也会保留第一个 token 的时间戳。当 `fresh_final_after_seconds > 0`（默认 `60`）时，完成的回复作为全新消息交付（尽力删除过时的预览），因此 Telegram 的可见时间戳反映完成时间。短预览仍在原地最终化。设置为 `0` 以始终在原地编辑。

:::note
流式传输默认禁用。在 `~/.hermes/config.yaml` 中启用它以尝试流式传输 UX。
:::

## 群聊会话隔离

控制共享聊天是按房间保留一个对话还是按参与者保留一个对话：

```yaml
group_sessions_per_user: true  # true = 群组/频道中的按用户隔离，false = 每个聊天一个共享会话
```

- `true` 是默认和推荐设置。在 Discord 频道、Telegram 群组、Slack 频道和类似的共享上下文中，当平台提供用户 ID 时，每个发送者获得自己的会话。
- `false` 回退到旧的共享房间行为。如果您明确希望 Hermes 将频道视为一个协作对话，这可能有用，但它也意味着用户共享上下文、token 成本和中断状态。
- 直接消息不受影响。Hermes 仍然像往常一样按聊天/DM ID 索引 DM。
- 线程无论如何都与父频道隔离；使用 `true` 时，每个参与者在线程内也获得自己的会话。

有关行为细节和示例，请参阅 [会话](/user-guide/sessions) 和 [Discord 指南](/user-guide/messaging/discord)。

## 未授权 DM 行为

控制当未知用户发送直接消息时 Hermes 做什么：

```yaml
unauthorized_dm_behavior: pair

whatsapp:
  unauthorized_dm_behavior: ignore
```

- `pair` 是默认设置。Hermes 拒绝访问，但在 DM 中回复一次性配对码。
- `ignore` 静默丢弃未授权的 DM。
- 平台部分覆盖全局默认值，因此您可以广泛保持配对启用，同时让一个平台更安静。

## 快速命令 {#quick-commands}

定义自定义命令，这些命令要么在不调用 LLM 的情况下运行 shell 命令，要么将一个斜杠命令别名到另一个。Exec 快速命令是零 token 的，在消息平台（Telegram、Discord 等）中用于快速服务器检查或实用脚本很有用。

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

用法：在 CLI 或任何消息平台中输入 `/status`、`/disk`、`/update`、`/gpu` 或 `/restart`。`exec` 命令在宿主机上本地运行并直接返回输出——没有 LLM 调用，不消耗 token。`alias` 命令重写为配置的斜杠命令目标。

- **30 秒超时**——长时间运行的命令会被终止并显示错误消息
- **优先级**——快速命令在技能命令之前检查，因此您可以覆盖技能名称
- **自动补全**——快速命令在调度时解析，不会显示在内置斜杠命令自动补全表中
- **类型**——支持的类型是 `exec` 和 `alias`；其他类型显示错误
- **随处可用**——CLI、Telegram、Discord、Slack、WhatsApp、Signal、Email、Home Assistant

纯字符串提示快捷方式不是有效的快速命令。对于可重用的提示工作流，创建技能或别名到现有斜杠命令。

## 人类延迟

在消息平台中模拟人类般的响应节奏：

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
  max_tool_calls: 50           # 代码执行内的最大工具调用次数
```

**`mode`** 控制脚本的工作目录和 Python 解释器：

- **`project`**（默认）——脚本在会话的工作目录中运行，使用活动 virtualenv/conda 环境的 python。项目依赖（`pandas`、`torch`、项目包）和相对路径（`.env`、`./data.csv`）自然解析，与 `terminal()` 看到的一致。
- **`strict`**——脚本在临时暂存目录中运行，使用 `sys.executable`（Hermes 自己的 python）。最大可重现性，但项目依赖和相对路径无法解析。

环境清理（去除 `*_API_KEY`、`*_TOKEN`、`*_SECRET`、`*_PASSWORD`、`*_CREDENTIAL`、`*_PASSWD`、`*_AUTH`）和工具白名单在两种模式下相同地应用——切换模式不会改变安全态势。

## 网络搜索后端

`web_search`、`web_extract` 和 `web_crawl` 工具支持五个后端提供商。在 `config.yaml` 中或通过 `hermes tools` 配置后端：

```yaml
web:
  backend: firecrawl    # firecrawl | searxng | parallel | tavily | exa

  # 或使用每个功能的键来混合提供商（例如免费搜索 + 付费提取）：
  search_backend: "searxng"
  extract_backend: "firecrawl"
```

| 后端 | 环境变量 | 搜索 | 提取 | 爬取 |
|---------|---------|--------|---------|-------|
| **Firecrawl**（默认） | `FIRECRAWL_API_KEY` | ✔ | ✔ | ✔ |
| **SearXNG** | `SEARXNG_URL` | ✔ | — | — |
| **Parallel** | `PARALLEL_API_KEY` | ✔ | ✔ | — |
| **Tavily** | `TAVILY_API_KEY` | ✔ | ✔ | ✔ |
| **Exa** | `EXA_API_KEY` | ✔ | ✔ | — |

**后端选择：** 如果 `web.backend` 未设置，后端会从可用的 API 密钥自动检测。如果仅设置了 `SEARXNG_URL`，则使用 SearXNG。如果仅设置了 `EXA_API_KEY`，则使用 Exa。如果仅设置了 `TAVILY_API_KEY`，则使用 Tavily。如果仅设置了 `PARALLEL_API_KEY`，则使用 Parallel。否则 Firecrawl 是默认的。

**SearXNG** 是一个免费、自托管、尊重隐私的元搜索引擎，查询 70+ 搜索引擎。不需要 API 密钥——只需将 `SEARXNG_URL` 设置为您的实例（例如，`http://localhost:8080`）。SearXNG 仅支持搜索；`web_extract` 和 `web_crawl` 需要单独的提取提供商（设置 `web.extract_backend`）。有关 Docker 设置说明，请参阅 [网络搜索设置指南](/user-guide/features/web-search)。

**自托管 Firecrawl：** 将 `FIRECRAWL_API_URL` 指向您自己的实例。设置自定义 URL 时，API 密钥变为可选（在服务器上设置 `USE_DB_AUTHENTICATION=***` 以禁用认证）。

**Parallel 搜索模式：** 将 `PARALLEL_SEARCH_MODE` 设置为控制搜索行为——`fast`、`one-shot` 或 `agentic`（默认：`agentic`）。

**Exa：** 在 `~/.hermes/.env` 中设置 `EXA_API_KEY`。支持 `category` 过滤（`company`、`research paper`、`news`、`people`、`personal site`、`pdf`）以及域名/日期过滤。

## 浏览器

配置浏览器自动化行为：

```yaml
browser:
  inactivity_timeout: 120        # 自动关闭空闲会话前的秒数
  command_timeout: 30             # 浏览器命令超时（截图、导航等）
  record_sessions: false         # 自动将浏览器会话录制为 WebM 视频到 ~/.hermes/browser_recordings/
  # 可选 CDP 覆盖——设置时，Hermes 直接附加到您自己的
  # Chrome（通过 /browser connect）而非启动无头浏览器。
  cdp_url: ""
  # 对话框监督器——控制当 CDP 后端附加时（Browserbase、通过
  # /browser connect 的本地 Chrome）如何处理原生 JS 对话框（alert / confirm / prompt）。
  # 在 Camofox 和默认本地 agent-browser 模式下被忽略。
  dialog_policy: must_respond    # must_respond | auto_dismiss | auto_accept
  dialog_timeout_s: 300          # must_respond 下的安全自动关闭（秒）
  camofox:
    managed_persistence: false   # 为 true 时，Camofox 会话在重启之间持久化 cookie/登录
```

**对话框策略：**

- `must_respond`（默认）——捕获对话框，在 `browser_snapshot.pending_dialogs` 中显示，并等待智能体调用 `browser_dialog(action=...)`。在 `dialog_timeout_s` 秒内无响应后，对话框自动关闭以防止页面的 JS 线程永远停滞。
- `auto_dismiss`——捕获，立即关闭。智能体仍会在 `browser_snapshot.recent_dialogs` 中看到带有 `closed_by="auto_policy"` 的对话框记录。
- `auto_accept`——捕获，立即接受。适用于具有激进 `beforeunload` 提示的页面。

有关完整的对话框工作流，请参阅 [浏览器功能页面](./features/browser.md#browser_dialog)。

浏览器工具集支持多个提供商。有关 Browserbase、Browser Use 和本地 Chrome CDP 设置的详细信息，请参阅 [浏览器功能页面](/user-guide/features/browser)。

## 时区

使用 IANA 时区字符串覆盖服务器本地时区。影响日志中的时间戳、cron 调度和系统提示时间注入。

```yaml
timezone: "America/New_York"   # IANA 时区（默认："" = 服务器本地时间）
```

支持的值：任何 IANA 时区标识符（例如 `America/New_York`、`Europe/London`、`Asia/Kolkata`、`UTC`）。留空或省略以使用服务器本地时间。

## Discord

配置消息网关的 Discord 特定行为：

```yaml
discord:
  require_mention: true          # 在服务器频道中需要 @提及 才能响应
  free_response_channels: ""     # 机器人无需 @提及 即可响应的频道 ID 逗号分隔列表
  auto_thread: true              # 在频道中 @提及 时自动创建线程
```

- `require_mention`——当 `true`（默认）时，机器人仅在服务器频道中被 `@BotName` 提及时响应。DM 始终无需提及即可工作。
- `free_response_channels`——机器人响应每条消息而无需提及的频道 ID 逗号分隔列表。
- `auto_thread`——当 `true`（默认）时，频道中的提及自动创建对话线程，保持频道整洁（类似于 Slack 线程）。

## 安全

预执行安全扫描和机密去除：

```yaml
security:
  redact_secrets: false          # 在工具输出和日志中去除 API 密钥模式（默认关闭）
  tirith_enabled: true           # 为终端命令启用 Tirith 安全扫描
  tirith_path: "tirith"          # tirith 二进制文件路径（默认：$PATH 中的 "tirith"）
  tirith_timeout: 5              # tirith 扫描超时前等待的秒数
  tirith_fail_open: true         # 如果 tirith 不可用，允许命令执行
  website_blocklist:             # 请参阅下面的网站黑名单部分
    enabled: false
    domains: []
    shared_files: []
```

- `redact_secrets`——当 `true` 时，自动检测并去除工具输出中看起来像 API 密钥、token 和密码的模式，然后再进入对话上下文和日志。**默认关闭**——如果您通常在工具输出中使用真实凭证并想要安全网，请启用。显式设置为 `true` 以开启。
- `tirith_enabled`——当 `true` 时，终端命令在执行前由 [Tirith](https://github.com/StackGuardian/tirith) 扫描以检测潜在危险操作。
- `tirith_path`——tirith 二进制文件的路径。如果 tirith 安装在非标准位置，请设置此项。
- `tirith_timeout`——等待 tirith 扫描的最大秒数。如果扫描超时，命令会继续执行。
- `tirith_fail_open`——当 `true`（默认）时，如果 tirith 不可用或失败，允许命令执行。设置为 `false` 以在 tirith 无法验证时阻止命令。

## 网站黑名单 {#website-blocklist}

阻止智能体的 web 和浏览器工具访问特定域名：

```yaml
security:
  website_blocklist:
    enabled: false               # 启用 URL 阻止（默认：false）
    domains:                     # 阻止的域名模式列表
      - "*.internal.company.com"
      - "admin.example.com"
      - "*.local"
    shared_files:                # 从外部文件加载额外规则
      - "/etc/hermes/blocked-sites.txt"
```

启用时，任何匹配阻止域名模式的 URL 在 web 或浏览器工具执行前被拒绝。这适用于 `web_search`、`web_extract`、`browser_navigate` 和任何访问 URL 的工具。

域名规则支持：
- 精确域名：`admin.example.com`
- 通配符子域名：`*.internal.company.com`（阻止所有子域名）
- TLD 通配符：`*.local`

共享文件每行包含一个域名规则（忽略空行和 `#` 注释）。缺失或不可读的文件会记录警告但不会禁用其他 web 工具。

策略缓存 30 秒，因此配置更改无需重启即可快速生效。

## 智能审批

控制 Hermes 如何处理潜在危险命令：

```yaml
approvals:
  mode: manual   # manual | smart | off
```

| 模式 | 行为 |
|------|----------|
| `manual`（默认） | 在执行任何标记的命令之前提示用户。在 CLI 中，显示交互式审批对话框。在消息传递中，排队待处理的审批请求。 |
| `smart` | 使用辅助 LLM 评估标记的命令是否实际危险。低风险命令通过会话级持久化自动批准。真正危险的命令升级给用户。 |
| `off` | 跳过所有审批检查。等同于 `HERMES_YOLO_MODE=true`。**谨慎使用。** |

智能模式对于减少审批疲劳特别有用——它让智能体在安全操作上更自主地工作，同时仍然捕获真正破坏性的命令。

:::warning
设置 `approvals.mode: off` 会禁用终端命令的所有安全检查。仅在受信任的、已沙箱化的环境中使用。
:::

## 检查点

破坏性文件操作前的自动文件系统快照。有关详细信息，请参阅 [检查点与回滚](/user-guide/checkpoints-and-rollback)。

```yaml
checkpoints:
  enabled: false                 # 启用自动检查点（另见：hermes chat --checkpoints）。默认：false（选择加入）。
  max_snapshots: 20              # 每个目录保留的最大检查点数（默认：20）
```

## 委派

配置 delegate 工具的子智能体行为：

```yaml
delegation:
  # model: "google/gemini-3-flash-preview"  # 覆盖模型（空 = 继承父级）
  # provider: "openrouter"                  # 覆盖提供商（空 = 继承父级）
  # base_url: "http://localhost:1234/v1"    # 直接 OpenAI-compatible 端点（优先于提供商）
  # api_key: "local-key"                    # base_url 的 API 密钥（回退到 OPENAI_API_KEY）
  max_concurrent_children: 3                # 每批并行子级数（下限 1，无上限）。也可通过 DELEGATION_MAX_CONCURRENT_CHILDREN 环境变量设置。
  max_spawn_depth: 1                        # 委派树深度上限（1-3，钳制）。1 = 扁平（默认）：父级生成不能委派的叶子。2 = 编排器子级可以生成叶子孙级。3 = 三级。
  orchestrator_enabled: true                # 全局开关。为 false 时，role="orchestrator" 被忽略，每个子级无论 max_spawn_depth 如何都被强制为叶子。
```

**子智能体 provider:model 覆盖：** 默认情况下，子智能体继承父智能体的提供商和模型。设置 `delegation.provider` 和 `delegation.model` 以将子智能体路由到不同的 provider:model 对——例如，对范围狭窄的子任务使用便宜/快速的模型，而您的主智能体运行昂贵的推理模型。

**直接端点覆盖：** 如果您想要明显的自定义端点路径，请设置 `delegation.base_url`、`delegation.api_key` 和 `delegation.model`。这将子智能体直接发送到该 OpenAI-compatible 端点，并优先于 `delegation.provider`。如果省略 `delegation.api_key`，Hermes 仅回退到 `OPENAI_API_KEY`。

委派提供商使用与 CLI/网关启动相同的凭证解析。支持所有配置的提供商：`openrouter`、`nous`、`copilot`、`zai`、`kimi-coding`、`minimax`、`minimax-cn`。设置提供商时，系统会自动解析正确的基础 URL、API 密钥和 API 模式——无需手动凭证连接。

**优先级：** config 中的 `delegation.base_url` → config 中的 `delegation.provider` → 父级提供商（继承）。config 中的 `delegation.model` → 父级模型（继承）。仅设置 `model` 而不设置 `provider` 仅更改模型名称，同时保留父级的凭证（适用于在同一提供商如 OpenRouter 内切换模型）。

**宽度和深度：** `max_concurrent_children` 限制每批并行运行的子智能体数量（默认 `3`，下限为 1，无上限）。也可通过 `DELEGATION_MAX_CONCURRENT_CHILDREN` 环境变量设置。当模型提交的 `tasks` 数组长度超过上限时，`delegate_task` 返回解释限制的工具错误，而非静默截断。`max_spawn_depth` 控制委派树深度（钳制在 1-3）。在默认 `1` 时，委派是扁平的：子级不能生成孙级，传递 `role="orchestrator"` 会静默降级为 `leaf`。提高到 `2` 以便编排器子级可以生成叶子孙级；`3` 用于三级树。智能体通过 `role="orchestrator"` 按调用选择编排；`orchestrator_enabled: false` 无论如何都会强制每个子级回退到叶子。成本呈乘法增长——在 `max_spawn_depth: 3` 且 `max_concurrent_children: 3` 时，树可以达到 3×3×3 = 27 个并发叶子智能体。有关使用模式，请参阅 [子智能体委派 → 深度限制和嵌套编排](features/delegation.md#depth-limit-and-nested-orchestration)。

## 澄清

配置澄清提示行为：

```yaml
clarify:
  timeout: 120                 # 等待用户澄清响应的秒数
```

## 上下文文件（SOUL.md、AGENTS.md）

Hermes 使用两种不同的上下文范围：

| 文件 | 用途 | 范围 |
|------|---------|-------|
| `SOUL.md` | **主智能体身份**——定义智能体是谁（系统提示中的 #1 插槽） | `~/.hermes/SOUL.md` 或 `$HERMES_HOME/SOUL.md` |
| `.hermes.md` / `HERMES.md` | 项目特定指令（最高优先级） | 遍历到 git 根目录 |
| `AGENTS.md` | 项目特定指令、编码约定 | 递归目录遍历 |
| `CLAUDE.md` | Claude Code 上下文文件（也被检测） | 仅工作目录 |
| `.cursorrules` | Cursor IDE 规则（也被检测） | 仅工作目录 |
| `.cursor/rules/*.mdc` | Cursor 规则文件（也被检测） | 仅工作目录 |

- **SOUL.md** 是智能体的主身份。它占据系统提示中的 #1 插槽，完全替换内置默认身份。编辑它以完全自定义智能体是谁。
- 如果 SOUL.md 缺失、为空或无法加载，Hermes 回退到内置默认身份。
- **项目上下文文件使用优先级系统**——仅加载一种类型（首匹配胜出）：`.hermes.md` → `AGENTS.md` → `CLAUDE.md` → `.cursorrules`。SOUL.md 始终独立加载。
- **AGENTS.md** 是分层的：如果子目录也有 AGENTS.md，所有文件都会被组合。
- Hermes 自动播种默认的 `SOUL.md`（如果不存在）。
- 所有加载的上下文文件上限为 20,000 字符，并进行智能截断。

另请参阅：
- [个性与 SOUL.md](/user-guide/features/personality)
- [上下文文件](/user-guide/features/context-files)

## 工作目录

| 上下文 | 默认 |
|---------|---------|
| **CLI (`hermes`)** | 您运行命令的当前目录 |
| **消息网关** | Home 目录 `~`（使用 `MESSAGING_CWD` 覆盖） |
| **Docker / Singularity / Modal / SSH** | 容器或远程机器内的用户 home 目录 |

覆盖工作目录：
```bash
# 在 ~/.hermes/.env 或 ~/.hermes/config.yaml 中：
MESSAGING_CWD=/home/myuser/projects    # 网关会话
TERMINAL_CWD=/workspace                # 所有终端会话
```
