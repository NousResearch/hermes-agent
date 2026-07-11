---
sidebar_position: 2
title: "配置"
description: "配置 Hermes Agent — config.yaml、providers、模型、API 密钥等"
---

# 配置

所有设置均存储在 `~/.hermes/` 目录中，便于访问。

## 目录结构

```text
~/.hermes/
├── config.yaml     # 设置（模型、终端、TTS、压缩等）
├── .env            # API 密钥和机密
├── auth.json       # OAuth provider 凭据（Nous Portal 等）
├── SOUL.md         # 主要 agent 身份（系统提示词第 #1 槽位）
├── memories/       # 持久记忆（MEMORY.md、USER.md）
├── skills/         # Agent 创建的技能（通过 skill_manage 工具管理）
├── cron/           # 定时任务
├── sessions/       # Gateway 会话
└── logs/           # 日志（errors.log、gateway.log — 机密自动脱敏）
```

## 管理配置

```bash
hermes config              # 查看当前配置
hermes config edit         # 在编辑器中打开 config.yaml
hermes config set KEY VAL  # 设置特定值
hermes config check        # 检查缺失选项（更新后使用）
hermes config migrate      # 交互式添加缺失选项

# 示例：
hermes config set model anthropic/claude-opus-4
hermes config set terminal.backend docker
hermes config set OPENROUTER_API_KEY sk-or-...  # 保存到 .env
```

:::tip
`hermes config set` 命令会自动将值路由到正确的文件 —— API 密钥保存到 `.env`，其他所有内容保存到 `config.yaml`。
:::

## 配置优先级

设置按以下顺序解析（优先级从高到低）：

1. **CLI 参数** —— 例如 `hermes chat --model anthropic/claude-sonnet-4`（单次调用覆盖）
2. **`~/.hermes/config.yaml`** —— 所有非机密设置的主配置文件
3. **`~/.hermes/.env`** —— 环境变量的回退；机密（API 密钥、token、密码）**必须**放这里
4. **内置默认值** —— 未设置任何内容时的硬编码安全默认值

:::info 经验法则
机密（API 密钥、bot token、密码）放入 `.env`。其他所有内容（模型、终端后端、压缩设置、内存限制、工具集）放入 `config.yaml`。当两者都设置时，`config.yaml` 对非机密设置优先。
:::

## 环境变量替换

可以在 `config.yaml` 中使用 `${VAR_NAME}` 语法引用环境变量：

```yaml
auxiliary:
  vision:
    api_key: ${GOOGLE_API_KEY}
    base_url: ${CUSTOM_VISION_URL}

delegation:
  api_key: ${DELEGATION_KEY}
```

单个值中可以有多个引用：`url: "${HOST}:${PORT}"`。如果引用的变量未设置，占位符将保持原样（`${UNDEFINED_VAR}` 保持不变）。仅支持 `${VAR}` 语法 —— 裸 `$VAR` 不会被展开。

关于 AI provider 设置（OpenRouter、Anthropic、Copilot、自定义端点、自托管 LLM、回退模型等），请参阅 [AI Providers](/integrations/providers)。

### Provider 超时

可以为 provider 设置 `providers.<id>.request_timeout_seconds` 作为全局请求超时，以及 `providers.<id>.models.<model>.timeout_seconds` 作为特定模型的覆盖值。适用于每种传输方式（OpenAI-wire、原生 Anthropic、Anthropic 兼容）上的主轮次客户端、回退链、凭据轮换后的重建，以及（对于 OpenAI-wire）每请求超时 kwarg —— 因此配置值优先于旧版 `HERMES_API_TIMEOUT` 环境变量。

还可以设置 `providers.<id>.stale_timeout_seconds` 用于非流式陈旧调用检测器，以及 `providers.<id>.models.<model>.stale_timeout_seconds` 作为特定模型的覆盖值。此值优先于旧版 `HERMES_API_CALL_STALE_TIMEOUT` 环境变量。

不设置这些值将保持旧版默认值（`HERMES_API_TIMEOUT=1800`s、`HERMES_API_CALL_STALE_TIMEOUT=90`s、原生 Anthropic 900s）。隐式的非流式 stale 检测会在本地端点上自动禁用，并且会在超大上下文下自动放宽。目前不适用于 AWS Bedrock（`bedrock_converse` 和 AnthropicBedrock SDK 路径均使用 boto3 及其自身的超时配置）。请参阅 [`cli-config.yaml.example`](https://github.com/NousResearch/hermes-agent/blob/main/cli-config.yaml.example) 中的注释示例。

## 终端后端配置

Hermes 支持六种终端后端。每种后端决定 agent 的 shell 命令实际在哪里执行 —— 本地机器、Docker 容器、通过 SSH 的远程服务器、Modal 云沙箱（直接或通过 Nous 托管的 gateway）、Daytona 工作区，或 Singularity/Apptainer 容器。

```yaml
terminal:
  backend: local    # local | docker | ssh | modal | daytona | singularity
  cwd: "."          # Gateway/cron 工作目录（CLI 始终使用启动目录）
  timeout: 180      # 每条命令的超时时间（秒）
  env_passthrough: []  # 转发到沙箱执行的环境变量名（terminal + execute_code）
  singularity_image: "docker://nikolaik/python-nodejs:python3.11-nodejs20"  # Singularity 后端的容器镜像
  modal_image: "nikolaik/python-nodejs:python3.11-nodejs20"                 # Modal 后端的容器镜像
  daytona_image: "nikolaik/python-nodejs:python3.11-nodejs20"               # Daytona 后端的容器镜像
```

对于 Modal 和 Daytona 等云沙箱，`container_persistent: true` 表示 Hermes 将尝试在沙箱重建后保留文件系统状态。这并不保证相同的活跃沙箱、PID 空间或后台进程之后仍在运行。

### 后端概览

| 后端 | 命令运行位置 | 隔离性 | 最适合 |
|---------|-------------------|-----------|----------|
| **local** | 直接在您的机器上 | 无 | 开发、个人使用 |
| **docker** | 单个持久 Docker 容器（跨会话、`/new`、子 agent 共享） | 完全（命名空间、cap-drop） | 安全沙箱、CI/CD |
| **ssh** | 通过 SSH 的远程服务器 | 网络边界 | 远程开发、强大硬件 |
| **modal** | Modal 云沙箱 | 完全（云 VM） | 临时云计算、评估 |
| **daytona** | Daytona 工作区 | 完全（云容器） | 托管云开发环境 |
| **singularity** | Singularity/Apptainer 容器 | 命名空间（--containall） | HPC 集群、共享机器 |

### Local 后端

默认后端。命令直接在您的机器上运行，无隔离。无需特殊设置。

```yaml
terminal:
  backend: local
```

:::warning
Agent 拥有与您的用户账户相同的文件系统访问权限。使用 `hermes tools` 禁用不需要的工具，或切换到 Docker 进行沙箱隔离。
:::

### Docker 后端

在具有安全加固的 Docker 容器内运行命令（所有权限已删除、无权限提升、PID 限制）。

**单个持久容器，而非每条命令一个容器。** Hermes 在首次使用时启动一个长期运行的容器，并通过 `docker exec` 将每个终端、文件和 `execute_code` 调用路由到同一容器中 —— 跨会话、`/new`、`/reset` 和 `delegate_task` 子 agent，贯穿 Hermes 进程的整个生命周期。工作目录更改、已安装的包以及 `/workspace` 中的文件会从一次工具调用延续到下一次，就像本地 shell 一样。容器在关闭时停止并删除。详情请参阅下方的**容器生命周期**。

```yaml
terminal:
  backend: docker
  docker_image: "nikolaik/python-nodejs:python3.11-nodejs20"
  docker_mount_cwd_to_workspace: false  # 将启动目录挂载到 /workspace
  docker_run_as_host_user: false   # 参见下方"以宿主用户身份运行容器"
  docker_forward_env:              # 转发到容器的环境变量
    - "GITHUB_TOKEN"
  docker_volumes:                  # 宿主目录挂载
    - "/home/user/projects:/workspace/projects"
    - "/home/user/data:/data:ro"   # :ro 表示只读
  docker_extra_args:               # 附加到 `docker run` 的额外标志
    - "--gpus=all"
    - "--network=host"

  # 资源限制
  container_cpu: 1                 # CPU 核心数（0 = 不限制）
  container_memory: 5120           # MB（0 = 不限制）
  container_disk: 51200            # MB（需要 XFS+pquota 上的 overlay2）
  container_persistent: true       # 跨会话持久化 /workspace 和 /root
```

**`terminal.docker_extra_args`**（也可通过 `TERMINAL_DOCKER_EXTRA_ARGS='["--gpus=all"]'` 覆盖）允许传递 Hermes 未作为一级键公开的任意 `docker run` 标志 —— `--gpus`、`--network`、`--add-host`、替代 `--security-opt` 覆盖等。每个条目必须是字符串；该列表最后附加到组装好的 `docker run` 调用中，因此可以在需要时覆盖 Hermes 的默认值。请谨慎使用 —— 与沙箱加固（权限删除、`--user`、workspace 绑定挂载）冲突的标志将悄然削弱隔离性。

**要求：** 已安装并运行 Docker Desktop 或 Docker Engine。Hermes 会探测 `$PATH` 以及常见的 macOS 安装位置（`/usr/local/bin/docker`、`/opt/homebrew/bin/docker`、Docker Desktop 应用包）。开箱即用支持 Podman：设置 `HERMES_DOCKER_BINARY=podman`（或完整路径）以在两者都安装时强制使用它。

**容器生命周期：** Hermes 为每个终端和文件工具调用重用单个长期运行的容器（`docker run -d ... sleep 2h`），跨会话、`/new`、`/reset` 和 `delegate_task` 子 agent，贯穿 Hermes 进程的整个生命周期。命令通过带登录 shell 的 `docker exec` 运行，因此工作目录更改、已安装的包以及 `/workspace` 中的文件都会从一次工具调用延续到下一次。容器在 Hermes 关闭时（或空闲清理回收时）停止并删除。

通过 `delegate_task(tasks=[...])` 生成的并行子 agent 共享这一个容器 —— 并发的 `cd`、环境变量修改以及对同一路径的写入会发生冲突。如果子 agent 需要隔离的沙箱，必须通过 `register_task_env_overrides()` 注册每任务镜像覆盖，RL 和基准测试环境（TerminalBench2、HermesSweEnv 等）会自动为其每任务 Docker 镜像执行此操作。

**安全加固：**
- `--cap-drop ALL`，仅添加回 `DAC_OVERRIDE`、`CHOWN`、`FOWNER`
- `--security-opt no-new-privileges`
- `--pids-limit 256`
- `/tmp`（512MB）、`/var/tmp`（256MB）、`/run`（64MB）的大小限制 tmpfs

**凭据转发：** `docker_forward_env` 中列出的环境变量首先从您的 shell 环境解析，然后回退到 `~/.hermes/.env`。技能也可以声明 `required_environment_variables`，这些变量会自动合并。

### SSH 后端

通过 SSH 在远程服务器上运行命令。使用 ControlMaster 进行连接复用（5 分钟空闲保活）。默认启用持久 shell —— 状态（cwd、环境变量）在命令之间保持。

```yaml
terminal:
  backend: ssh
  persistent_shell: true           # 保持长期运行的 bash 会话（默认：true）
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

**工作原理：** 使用 `BatchMode=yes` 和 `StrictHostKeyChecking=accept-new` 在初始化时连接。持久 shell 在远程主机上保持单个 `bash -l` 进程存活，通过临时文件进行通信。需要 `stdin_data` 或 `sudo` 的命令会自动回退到单次模式。

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

**必需：** `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET` 环境变量，或 `~/.modal.toml` 配置文件。

**持久化：** 启用后，沙箱文件系统在清理时快照，并在下次会话时恢复。快照在 `~/.hermes/modal_snapshots.json` 中跟踪。这保留文件系统状态，而非活跃进程、PID 空间或后台任务。

**凭据文件：** 自动从 `~/.hermes/` 挂载（OAuth token 等），并在每条命令前同步。

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

**必需：** `DAYTONA_API_KEY` 环境变量。

**持久化：** 启用后，沙箱在清理时停止（而非删除），并在下次会话时恢复。沙箱名称遵循 `hermes-{task_id}` 模式。

**磁盘限制：** Daytona 强制执行 10 GiB 最大值。超过此值的请求将被截断并发出警告。

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

**镜像处理：** Docker URL（`docker://...`）自动转换为 SIF 文件并缓存。现有 `.sif` 文件直接使用。

**临时目录：** 按顺序解析：`TERMINAL_SCRATCH_DIR` → `TERMINAL_SANDBOX_DIR/singularity` → `/scratch/$USER/hermes-agent`（HPC 惯例）→ `~/.hermes/sandboxes/singularity`。

**隔离：** 使用 `--containall --no-home` 实现完全命名空间隔离，不挂载宿主 home 目录。

### 常见终端后端问题

如果终端命令立即失败或终端工具报告为已禁用：

- **Local** —— 无特殊要求。入门时最安全的默认选项。
- **Docker** —— 运行 `docker version` 验证 Docker 是否正常工作。如果失败，修复 Docker 或执行 `hermes config set terminal.backend local`。
- **SSH** —— `TERMINAL_SSH_HOST` 和 `TERMINAL_SSH_USER` 都必须设置。如果缺少任一项，Hermes 会记录清晰的错误。
- **Modal** —— 需要 `MODAL_TOKEN_ID` 环境变量或 `~/.modal.toml`。运行 `hermes doctor` 检查。
- **Daytona** —— 需要 `DAYTONA_API_KEY`。Daytona SDK 处理服务器 URL 配置。
- **Singularity** —— 需要 `$PATH` 中有 `apptainer` 或 `singularity`。HPC 集群上常见。

如有疑问，将 `terminal.backend` 设回 `local` 并首先验证命令在那里运行。

### 拆卸时远程到宿主文件同步

对于 **SSH**、**Modal** 和 **Daytona** 后端（agent 的工作树位于与运行 Hermes 的宿主不同的机器上），Hermes 跟踪 agent 在远程沙箱中触及的文件，并在会话拆卸/沙箱清理时，将修改的文件**同步回宿主**，存放在 `~/.hermes/cache/remote-syncs/<session-id>/` 下。

- 触发时机：会话关闭、`/new`、`/reset`、gateway 消息超时、子 agent 使用远程后端时 `delegate_task` 子 agent 完成。
- 覆盖 agent 修改的整个树，而不仅仅是它明确打开的文件。添加、编辑和删除都会被捕获。
- 远程沙箱可能在您查找时已被拆除；本地 `~/.hermes/cache/remote-syncs/…` 副本是 agent 更改内容的权威记录。
- 大型二进制输出（模型检查点、原始数据集）按大小限制 —— 同步跳过超过 `file_sync_max_mb`（默认 `100`）的文件。如果您期望更大的工件返回，请调高该值。

```yaml
terminal:
  file_sync_max_mb: 100     # 默认 —— 同步最大 100 MB 的文件
  file_sync_enabled: true   # 默认 —— 设为 false 可完全跳过同步
```

这是从会话结束后被销毁的临时云沙箱中恢复结果的方式，无需告诉 agent 显式地 `scp` 或 `modal volume put` 每个工件。

### Docker 卷挂载

使用 Docker 后端时，`docker_volumes` 允许您与容器共享宿主目录。每个条目使用标准 Docker `-v` 语法：`host_path:container_path[:options]`。

```yaml
terminal:
  backend: docker
  docker_volumes:
    - "/home/user/projects:/workspace/projects"   # 读写（默认）
    - "/home/user/datasets:/data:ro"              # 只读
    - "/home/user/.hermes/cache/documents:/output" # Gateway 可见的导出
```

适用于：
- **向 agent 提供文件**（数据集、配置、参考代码）
- **从 agent 接收文件**（生成的代码、报告、导出）
- **共享工作区**，您和 agent 都访问相同的文件

如果您使用消息 gateway 并希望 agent 通过 `MEDIA:/...` 发送生成的文件，建议使用专用的宿主可见导出挂载，例如 `/home/user/.hermes/cache/documents:/output`。

- 在 Docker 中将文件写入 `/output/...`
- 在 `MEDIA:` 中发出**宿主路径**，例如：`MEDIA:/home/user/.hermes/cache/documents/report.txt`
- **不要**发出 `/workspace/...` 或 `/output/...`，除非该确切路径在宿主上对 gateway 进程也存在

:::warning
YAML 重复键会静默覆盖之前的键。如果您已有 `docker_volumes:` 块，请将新挂载合并到同一列表中，而不是在文件后面再添加一个 `docker_volumes:` 键。
:::

也可以通过环境变量设置：`TERMINAL_DOCKER_VOLUMES='["/host:/container"]'`（JSON 数组）。

### Docker 凭据转发

默认情况下，Docker 终端会话不继承任意宿主凭据。如果您需要在容器内使用特定 token，请将其添加到 `terminal.docker_forward_env`。

```yaml
terminal:
  backend: docker
  docker_forward_env:
    - "GITHUB_TOKEN"
    - "NPM_TOKEN"
```

Hermes 首先从您当前的 shell 解析每个列出的变量，然后回退到通过 `hermes config set` 保存的 `~/.hermes/.env`。

:::warning
`docker_forward_env` 中列出的任何内容都会对容器内运行的命令可见。只转发您愿意暴露给终端会话的凭据。
:::

### 以宿主用户身份运行容器

默认情况下，Docker 容器以 `root`（UID 0）身份运行。在 `/workspace` 或其他绑定挂载中创建的文件在宿主上归 root 所有，因此会话结束后您必须 `sudo chown` 才能从宿主编辑器编辑它们。`terminal.docker_run_as_host_user` 标志解决了这个问题：

```yaml
terminal:
  backend: docker
  docker_run_as_host_user: true   # 默认：false
```

启用后，Hermes 将 `--user $(id -u):$(id -g)` 附加到 `docker run` 命令，使写入绑定挂载目录（`/workspace`、`/root`、`docker_volumes` 中的任何内容）的文件归您的宿主用户所有，而非 root。权衡：容器将无法再 `apt install` 或写入 `/root/.npm` 等 root 拥有的路径 —— 如果您同时需要这两者，请使用 `HOME` 归非 root 用户所有的基础镜像（或在镜像构建时添加所需工具）。

保持 `false`（默认）以获得向后兼容的行为。当您的工作流主要是"编辑挂载的宿主文件"且厌倦了 `sudo chown -R` 时，请开启此选项。

### 可选：将启动目录挂载到 `/workspace`

Docker 沙箱默认保持隔离。Hermes **不会**将您当前的宿主工作目录传入容器，除非您明确选择加入。

在 `config.yaml` 中启用：

```yaml
terminal:
  backend: docker
  docker_mount_cwd_to_workspace: true
```

启用后：
- 如果您从 `~/projects/my-app` 启动 Hermes，该宿主目录将绑定挂载到 `/workspace`
- Docker 后端从 `/workspace` 开始
- 文件工具和终端命令都能看到相同的挂载项目

禁用时，`/workspace` 保持沙箱所有，除非您通过 `docker_volumes` 显式挂载内容。

安全权衡：
- `false` 保留沙箱边界
- `true` 使沙箱直接访问您启动 Hermes 的目录

仅在您有意希望容器处理实时宿主文件时才选择加入。

### 持久 Shell

默认情况下，每条终端命令在其自己的子进程中运行 —— 工作目录、环境变量和 shell 变量在命令之间重置。启用**持久 shell** 后，单个长期运行的 bash 进程在 `execute()` 调用之间保持存活，使状态在命令之间保持。

这对 **SSH 后端**最有用，它还消除了每条命令的连接开销。持久 shell **对 SSH 默认启用**，对本地后端禁用。

```yaml
terminal:
  persistent_shell: true   # 默认 —— 为 SSH 启用持久 shell
```

禁用：

```bash
hermes config set terminal.persistent_shell false
```

**跨命令保持的内容：**
- 工作目录（`cd /tmp` 对下一条命令生效）
- 导出的环境变量（`export FOO=bar`）
- Shell 变量（`MY_VAR=hello`）

**优先级：**

| 级别 | 变量 | 默认值 |
|-------|----------|---------|
| 配置 | `terminal.persistent_shell` | `true` |
| SSH 覆盖 | `TERMINAL_SSH_PERSISTENT` | 遵循配置 |
| Local 覆盖 | `TERMINAL_LOCAL_PERSISTENT` | `false` |

每个后端的环境变量具有最高优先级。如果您也想在本地后端使用持久 shell：

```bash
export TERMINAL_LOCAL_PERSISTENT=true
```

:::note
需要 `stdin_data` 或 sudo 的命令会自动回退到单次模式，因为持久 shell 的 stdin 已被 IPC 协议占用。
:::

有关每个后端的详细信息，请参阅[代码执行](features/code-execution.md)和 [README 的终端部分](features/tools.md)。

## 技能设置

技能可以通过其 SKILL.md frontmatter 声明自己的配置设置。这些是非机密值（路径、偏好、域设置），存储在 `config.yaml` 的 `skills.config` 命名空间下。

```yaml
skills:
  config:
    myplugin:
      path: ~/myplugin-data   # 示例 —— 每个技能定义自己的键
```

**技能设置的工作原理：**

- `hermes config migrate` 扫描所有已启用的技能，找到未配置的设置，并提供提示
- `hermes config show` 在"技能设置"下显示所有技能设置及其所属技能
- 技能加载时，其解析的配置值会自动注入到技能上下文中

**手动设置值：**

```bash
hermes config set skills.config.myplugin.path ~/myplugin-data
```

有关在您自己的技能中声明配置设置的详细信息，请参阅[创建技能 — 配置设置](/developer-guide/creating-skills#config-settings-configyaml)。

### Agent 创建技能写入的守卫

当 agent 使用 `skill_manage` 创建、编辑、修补或删除技能时，Hermes 可以选择扫描新/更新的内容以查找危险关键字模式（凭据收集、明显的 prompt 注入、数据外泄指令）。扫描器**默认关闭** —— 合法触及 `~/.ssh/` 或提及 `$OPENAI_API_KEY` 的真实 agent 工作流触发启发式规则过于频繁。如果您希望扫描器在 agent 的技能写入落地前提示您，请重新开启：

```yaml
skills:
  guard_agent_created: true   # 默认：false
```

开启后，任何被标记的 `skill_manage` 写入都会以审批提示的形式出现，并附带扫描器的理由。接受的写入落地；拒绝的写入向 agent 返回解释性错误。

## 内存配置

```yaml
memory:
  memory_enabled: true
  user_profile_enabled: true
  memory_char_limit: 2200   # ~800 tokens
  user_char_limit: 1375     # ~500 tokens
```

## 文件读取安全

控制单次 `read_file` 调用可以返回多少内容。超过限制的读取将被拒绝，并向 agent 返回错误，提示使用 `offset` 和 `limit` 读取较小范围。这可以防止单次读取压缩的 JS 包或大型数据文件时淹没上下文窗口。

```yaml
file_read_max_chars: 100000  # 默认 —— ~25-35K tokens
```

如果您使用具有大上下文窗口的模型并经常读取大文件，请调高此值。对于小上下文模型，请降低以保持读取高效：

```yaml
# 大上下文模型（200K+）
file_read_max_chars: 200000

# 小型本地模型（16K 上下文）
file_read_max_chars: 30000
```

Agent 还会自动去重文件读取 —— 如果同一文件区域被读取两次且文件未更改，则返回轻量级存根而不是重新发送内容。这在上下文压缩后重置，以便 agent 在内容被摘要后可以重新读取文件。

## 工具输出截断限制

三个相关的上限控制工具在 Hermes 截断之前可以返回多少原始输出：

```yaml
tool_output:
  max_bytes: 50000        # 终端输出上限（字符）
  max_lines: 2000         # read_file 分页上限
  max_line_length: 2000   # read_file 行号视图中的每行上限
```

- **`max_bytes`** —— 当 `terminal` 命令产生超过此字符数的合并 stdout/stderr 时，Hermes 保留前 40% 和后 60%，并在中间插入 `[OUTPUT TRUNCATED]` 通知。默认 `50000`（典型分词器约 12-15K tokens）。
- **`max_lines`** —— 单次 `read_file` 调用的 `limit` 参数上限。超过此值的请求将被截断，以防单次读取淹没上下文窗口。默认 `2000`。
- **`max_line_length`** —— `read_file` 发出行号视图时应用的每行上限。超过此长度的行将被截断为此字符数，后跟 `... [truncated]`。默认 `2000`。

对于具有大上下文窗口且每次调用可以承受更多原始输出的模型，请调高限制。对于小上下文模型，请降低以保持工具结果紧凑：

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

要在 CLI 和每个 gateway 平台上统一禁用特定工具集，请在 `agent.disabled_toolsets` 下列出其名称：

```yaml
agent:
  disabled_toolsets:
    - memory       # 隐藏内存工具 + MEMORY_GUIDANCE 注入
    - web          # 任何地方都不使用 web_search / web_extract
```

这在每个平台的工具配置（由 `hermes tools` 写入的 `platform_toolsets`）**之后**应用，因此此处列出的工具集始终被删除 —— 即使平台的已保存配置仍然列出它。当您希望有一个"到处关闭 X"的单一开关而不是编辑 `hermes tools` UI 中的 15+ 个平台行时，请使用此选项。

留空列表或省略键不会产生任何效果。

## Git Worktree 隔离

启用隔离的 git worktree，以便在同一仓库上并行运行多个 agent：

```yaml
worktree: true    # 始终创建 worktree（与 hermes -w 相同）
# worktree: false # 默认 —— 仅在传递 -w 标志时
```

启用后，每个 CLI 会话在 `.worktrees/` 下创建一个带有自己分支的新 worktree。Agent 可以编辑文件、提交、推送和创建 PR，而不会相互干扰。干净的 worktree 在退出时删除；脏的 worktree 保留以供手动恢复。

您还可以通过仓库根目录中的 `.worktreeinclude` 列出要复制到 worktree 的 gitignore 文件：

```
# .worktreeinclude
.env
.venv/
node_modules/
```

## 上下文压缩

Hermes 自动压缩长对话以保持在模型的上下文窗口内。压缩摘要器是一个单独的 LLM 调用 —— 您可以将其指向任何 provider 或端点。

所有压缩设置都在 `config.yaml` 中（无环境变量）。

### 完整参考

```yaml
compression:
  enabled: true                                     # 开启/关闭压缩
  threshold: 0.50                                   # 在上下文限制的此百分比时压缩
  target_ratio: 0.20                                # 保留为最近尾部的阈值分数
  protect_last_n: 20                                # 保持未压缩的最少最近消息数
  hygiene_hard_message_limit: 5000                  # Gateway 安全阀 —— 见下文

# 摘要模型/provider 在 auxiliary: 下配置：
auxiliary:
  compression:
    model: ""                                       # 空 = 使用主聊天模型。覆盖为例如 "google/gemini-3-flash-preview" 以获得更便宜/更快的压缩。
    provider: "auto"                                # Provider："auto"、"openrouter"、"nous"、"codex"、"main" 等
    base_url: null                                  # 自定义 OpenAI 兼容端点（覆盖 provider）
```

:::info 旧版配置迁移
带有 `compression.summary_model`、`compression.summary_provider` 和 `compression.summary_base_url` 的旧版配置在首次加载时自动迁移到 `auxiliary.compression.*`（配置版本 17）。无需手动操作。
:::

`hygiene_hard_message_limit` 是仅限 gateway 的**预压缩安全阀**。它的存在是为了打破一个死循环：当超大会话的 API 调用持续断开时，gateway 永远收不到 token 使用数据，基于 token 的阈值因此无法触发，于是 transcript 持续增长、断开愈发严重。这个基于消息数的下限仅凭消息数量触发（无论 API 是否失败，消息数始终已知），强制压缩以恢复会话。默认 `5000` —— 远高于任何正常会话，包括做数千次短轮次的大上下文（1M+）模型，它们早就在 token 阈值处压缩了。对于异常平台可调得更高；要强制更积极的压缩则调低。在运行中的 gateway 上编辑此值将在下一条消息时生效（见下文）。

:::tip Gateway 热重载压缩和上下文长度
从最近的版本开始，在运行中的 gateway 上编辑 `config.yaml` 中的 `model.context_length` 或任何 `compression.*` 键将在下一条消息时生效 —— 无需 gateway 重启、`/reset` 或会话轮换。缓存的 agent 签名包含这些键，因此 gateway 在检测到更改时会透明地重建 agent。API 密钥和工具/技能配置仍需要通常的重载路径。
:::

### 常见设置

**默认（自动检测）—— 无需配置：**
```yaml
compression:
  enabled: true
  threshold: 0.50
```
使用您的主 provider 和主模型。如果您希望在比主聊天模型更便宜的模型上进行压缩，请覆盖每任务（例如 `auxiliary.compression.provider: openrouter` + `model: google/gemini-2.5-flash`）。

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
指向自定义 OpenAI 兼容端点。使用 `OPENAI_API_KEY` 进行认证。

### 三个旋钮的交互方式

| `auxiliary.compression.provider` | `auxiliary.compression.base_url` | 结果 |
|---------------------|---------------------|--------|
| `auto`（默认） | 未设置 | 自动检测最佳可用 provider |
| `nous` / `openrouter` / 等 | 未设置 | 强制使用该 provider，使用其认证 |
| 任意 | 已设置 | 直接使用自定义端点（忽略 provider） |

:::warning 摘要模型上下文长度要求
摘要模型**必须**具有至少与您的主 agent 模型一样大的上下文窗口。压缩器将对话的完整中间部分发送给摘要模型 —— 如果该模型的上下文窗口小于主模型的，摘要调用将因上下文长度错误而失败。发生这种情况时，中间轮次将**在没有摘要的情况下被丢弃**，静默丢失对话上下文。如果您覆盖模型，请验证其上下文长度满足或超过您的主模型。
:::

## 上下文引擎

上下文引擎控制在接近模型 token 限制时如何管理对话。内置的 `compressor` 引擎使用有损摘要（参见[上下文压缩](/developer-guide/context-compression-and-caching)）。插件引擎可以用替代策略替换它。

```yaml
context:
  engine: "compressor"    # 默认 —— 内置有损摘要
```

使用插件引擎（例如，用于无损上下文管理的 LCM）：

```yaml
context:
  engine: "lcm"          # 必须与插件名称匹配
```

插件引擎**永远不会自动激活** —— 您必须将 `context.engine` 显式设置为插件名称。可用引擎可以通过 `hermes plugins` → Provider Plugins → Context Engine 浏览和选择。

有关内存插件的类似单选系统，请参阅[内存 Providers](/user-guide/features/memory-providers)。

## 迭代预算

当 agent 在处理具有许多工具调用的复杂任务时，它可能会耗尽其迭代预算（默认：90 轮）。Hermes **不会**在任务中途注入压力警告 —— 早期版本会在预算达到 70%/90% 时警告模型，这会导致模型过早放弃复杂任务，该机制已于 2026 年 4 月移除。

取而代之的是，当预算真正耗尽（90/90）时，Hermes 注入一条消息要求模型收尾，并允许一次**宽限调用**以便其给出最终响应。如果该宽限调用仍未产生文本，则会要求 agent 总结已完成的工作。

```yaml
agent:
  max_turns: 90                # 每次对话轮次的最大迭代次数（默认：90）
  api_max_retries: 3           # 回退启动前每个 provider 的重试次数（默认：3）
```

当迭代预算完全耗尽时，CLI 向用户显示通知：`⚠ Iteration budget reached (90/90) — response may be incomplete`。

`agent.api_max_retries` 控制 Hermes 在回退 provider 切换启动**之前**对瞬时错误（速率限制、连接断开、5xx）重试 provider API 调用的次数。默认为 `3` —— 总共四次尝试。如果您配置了[回退 providers](/user-guide/features/fallback-providers) 并希望更快地故障转移，请将其降至 `0`，这样主 provider 上的第一个瞬时错误会立即切换到回退，而不是对不稳定的端点进行重试。

### API 超时

Hermes 对流式传输有单独的超时层，以及用于非流式调用的陈旧检测器。陈旧检测器仅在您将其保留为隐式默认值时才会自动调整本地 provider。

| 超时 | 默认值 | 本地 providers | 配置/环境变量 |
|---------|---------|----------------|--------------|
| Socket 读取超时 | 120s | 自动提升至 1800s | `HERMES_STREAM_READ_TIMEOUT` |
| 陈旧流检测 | 180s | 自动禁用 | `HERMES_STREAM_STALE_TIMEOUT` |
| 陈旧非流检测 | 300s | 保持隐式时自动禁用 | `providers.<id>.stale_timeout_seconds` 或 `HERMES_API_CALL_STALE_TIMEOUT` |
| API 调用（非流式） | 1800s | 不变 | `providers.<id>.request_timeout_seconds` / `timeout_seconds` 或 `HERMES_API_TIMEOUT` |

**Socket 读取超时**控制 httpx 等待 provider 下一个数据块的时间。本地 LLM 在大上下文上预填充可能需要几分钟才能产生第一个 token，因此当 Hermes 检测到本地端点时，会将此值提升至 30 分钟。如果您显式设置 `HERMES_STREAM_READ_TIMEOUT`，无论端点检测如何，始终使用该值。

**陈旧流检测**终止接收 SSE 保活 ping 但没有实际内容的连接。对于本地 providers，这完全禁用，因为它们在预填充期间不发送保活 ping。

**陈旧非流检测**终止长时间没有响应的非流式调用。默认情况下，Hermes 在本地端点上禁用此功能，以避免长时间预填充期间的误报。如果您显式设置 `providers.<id>.stale_timeout_seconds`、`providers.<id>.models.<model>.stale_timeout_seconds` 或 `HERMES_API_CALL_STALE_TIMEOUT`，即使在本地端点上也会遵守该显式值。

## 上下文压力警告

与迭代预算压力分开，上下文压力跟踪对话距**压缩阈值**有多近 —— 即上下文压缩触发以摘要旧消息的点。这有助于您和 agent 了解对话何时变长。

| 进度 | 级别 | 发生的事情 |
|----------|-------|-------------|
| **≥ 60%** 到阈值 | 信息 | CLI 显示青色进度条；gateway 发送信息通知 |
| **≥ 85%** 到阈值 | 警告 | CLI 显示粗体黄色进度条；gateway 警告压缩即将发生 |

在 CLI 中，上下文压力在工具输出流中显示为进度条：

```
  ◐ context ████████████░░░░░░░░ 62% to compaction  48k threshold (50%) · approaching compaction
```

在消息平台上，发送纯文本通知：

```
◐ Context: ████████████░░░░░░░░ 62% to compaction (threshold: 50% of window).
```

如果自动压缩被禁用，警告会告诉您上下文可能被截断。

上下文压力是自动的 —— 无需配置。它纯粹作为面向用户的通知触发，不修改消息流或向模型上下文注入任何内容。

## 凭据池策略

当您为同一 provider 拥有多个 API 密钥或 OAuth token 时，配置轮换策略：

```yaml
credential_pool_strategies:
  openrouter: round_robin    # 均匀循环使用密钥
  anthropic: least_used      # 始终选择使用最少的密钥
```

选项：`fill_first`（默认）、`round_robin`、`least_used`、`random`。完整文档请参阅[凭据池](/user-guide/features/credential-pools)。

## Prompt 缓存

当活跃 provider 支持时，Hermes 自动开启跨会话 prompt 缓存 —— 无需用户配置。

对于**原生 Anthropic**、**OpenRouter** 和 **Nous Portal** 上的 Claude，Hermes 在系统提示词和技能块上附加带有 1 小时 TTL（`ttl: "1h"`）的 `cache_control` 断点。在新鲜的一小时内首次发送时按完整输入费率计费；同一小时内任何会话的后续发送以折扣缓存读取费率从缓存中提取。这意味着系统提示词、加载的技能内容以及任何长上下文包含的早期部分在第一个小时内跨 `hermes` 会话和分叉子 agent 被重用。

Qwen Cloud（阿里巴巴 DashScope）上游将缓存 TTL 限制为 5 分钟，因此 Hermes 在那里使用 5 分钟断点 TTL。其他通过第三方的 Claude 路径（AWS Bedrock、Azure Foundry）回退到 provider 自己的缓存默认值。xAI Grok 使用单独的会话固定对话 ID 机制 —— 参阅 [xAI prompt 缓存](/integrations/providers#xai-grok--responses-api--prompt-caching)。

不存在禁用此功能的旋钮 —— 缓存始终开启，即使在单轮对话中也能节省费用，因为仅系统提示词就占输入 token 数的相当大比例。

## 辅助模型

Hermes 使用"辅助"模型处理图像分析、网页摘要、浏览器截图分析、会话标题生成和上下文压缩等附带任务。默认情况下（`auxiliary.*.provider: "auto"`），Hermes 将每个辅助任务路由到您的**主聊天模型** —— 与您在 `hermes model` 中选择的相同 provider/模型。您无需配置任何内容即可开始，但请注意，在昂贵的推理模型（Opus、MiniMax M2.7 等）上，辅助任务会增加显著成本。如果您希望无论主模型如何都使用便宜且快速的附带任务，请显式设置 `auxiliary.<task>.provider` 和 `auxiliary.<task>.model`（例如，在 OpenRouter 上使用 Gemini Flash 进行视觉和网页提取）。

:::note 为什么 "auto" 使用您的主模型
早期版本将聚合器用户（OpenRouter、Nous Portal）分流到便宜的 provider 端默认值。这令人惊讶 —— 付费购买聚合器订阅的用户会看到不同的模型处理其辅助流量。`auto` 现在对所有人使用主模型，`config.yaml` 中的每任务覆盖仍然优先（见下方[完整辅助配置参考](#full-auxiliary-config-reference)）。
:::

### 交互式配置辅助模型

无需手动编辑 YAML，运行 `hermes model` 并从菜单中选择**"配置辅助模型"**。您将获得交互式的每任务选择器：

```
$ hermes model
→ Configure auxiliary models

[ ] vision               currently: auto / main model
[ ] web_extract          currently: auto / main model
[ ] title_generation     currently: openrouter / google/gemini-3-flash-preview
[ ] compression          currently: auto / main model
[ ] approval             currently: auto / main model
[ ] triage_specifier     currently: auto / main model
[ ] kanban_decomposer    currently: auto / main model
[ ] profile_describer    currently: auto / main model
```

选择任务，选择 provider（OAuth 流程打开浏览器；API 密钥 provider 提示输入），选择模型。更改持久化到 `config.yaml` 中的 `auxiliary.<task>.*`。与主模型选择器相同的机制 —— 无需学习额外语法。

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

Hermes 中的每个模型槽位 —— 辅助任务、压缩、回退 —— 使用相同的三个旋钮：

| 键 | 作用 | 默认值 |
|-----|-------------|---------|
| `provider` | 用于认证和路由的 provider | `"auto"` |
| `model` | 请求的模型 | provider 的默认值 |
| `base_url` | 自定义 OpenAI 兼容端点（覆盖 provider） | 未设置 |

当设置 `base_url` 时，Hermes 忽略 provider 并直接调用该端点（使用 `api_key` 或 `OPENAI_API_KEY` 进行认证）。当仅设置 `provider` 时，Hermes 使用该 provider 的内置认证和基础 URL。

辅助任务的可用 providers：`auto`、`main`，以及[provider 注册表](/reference/environment-variables)中的任何 provider —— `openrouter`、`nous`、`openai-codex`、`copilot`、`copilot-acp`、`anthropic`、`gemini`、`qwen-oauth`、`zai`、`kimi-coding`、`kimi-coding-cn`、`minimax`、`minimax-cn`、`minimax-oauth`、`deepseek`、`nvidia`、`xai`、`xai-oauth`、`ollama-cloud`、`alibaba`、`bedrock`、`huggingface`、`arcee`、`xiaomi`、`kilocode`、`opencode-zen`、`opencode-go`、`azure-foundry` —— 或您 `custom_providers` 列表中任何命名的自定义 provider（例如 `provider: "beans"`）。

:::tip MiniMax OAuth
`minimax-oauth` 通过浏览器 OAuth 登录（无需 API 密钥）。运行 `hermes model` 并选择 **MiniMax (OAuth)** 进行认证。辅助任务自动使用 `MiniMax-M2.7-highspeed`。参阅 [MiniMax OAuth 指南](../guides/minimax-oauth.md)。
:::

:::tip xAI Grok OAuth
`xai-oauth` 通过浏览器 OAuth 为 SuperGrok 和 X Premium+ 订阅者登录（无需 API 密钥）。运行 `hermes model` 并选择 **xAI Grok OAuth (SuperGrok / Premium+)** 进行认证。相同的 OAuth token 可重用于每个直接到 xAI 的接口（聊天、辅助任务、TTS、图像生成、视频生成、转录）。参阅 [xAI Grok OAuth 指南](../guides/xai-grok-oauth.md)，如果 Hermes 在远程主机上，请参阅 [SSH/远程主机上的 OAuth](../guides/oauth-over-ssh.md)。
:::

:::warning `"main"` 仅用于辅助任务
`"main"` provider 选项表示"使用我的主 agent 使用的任何 provider" —— 它仅在 `auxiliary:`、`compression:` 和 `fallback_model:` 配置中有效。它**不是**顶级 `model.provider` 设置的有效值。如果您使用自定义 OpenAI 兼容端点，请在 `model:` 部分设置 `provider: custom`。所有主模型 provider 选项请参阅 [AI Providers](/integrations/providers)。
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
    timeout: 120               # 秒 —— LLM API 调用超时；视觉负载需要宽裕的超时
    download_timeout: 30       # 秒 —— 图像 HTTP 下载；慢速连接请增加

  # 网页摘要 + 浏览器页面文本提取
  web_extract:
    provider: "auto"
    model: ""                  # 例如 "google/gemini-2.5-flash"
    base_url: ""
    api_key: ""
    timeout: 360               # 秒（6 分钟）—— 每次尝试的 LLM 摘要

  # 危险命令审批分类器
  approval:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30                # 秒

  # 上下文压缩超时（与 compression.* 配置分开）
  compression:
    timeout: 120               # 秒 —— 压缩摘要长对话，需要更多时间
    # fallback_chain:           # 可选 —— 发生速率限制/连接故障时尝试的 provider
    #   - provider: nous
    #     model: deepseek/deepseek-chat
    #   - provider: openrouter
    #     model: google/gemini-2.5-flash
    #     base_url: ""
    #     api_key: ""

  # 技能中心 —— 技能匹配和搜索
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

  # Kanban 分类规格说明器 —— `hermes kanban specify <id>`（或
  # 仪表板上 Triage 列卡片的 ✨ Specify 按钮）使用此
  # 槽位将单行描述扩展为具体规格并将
  # 任务提升到 `todo`。便宜快速的模型在这里效果很好；规格扩展
  # 很短，不需要推理深度。
  triage_specifier:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 120
```

:::tip
每个辅助任务都有可配置的 `timeout`（秒）。默认值：vision 120s、web_extract 360s、approval 30s、compression 120s。如果您为辅助任务使用慢速本地模型，请增加这些值。Vision 还有单独的 `download_timeout`（默认 30s）用于 HTTP 图像下载 —— 对于慢速连接或自托管图像服务器，请增加此值。
:::

:::info
上下文压缩有自己的 `compression:` 块用于阈值，以及 `auxiliary.compression:` 块用于模型/provider 设置 —— 参阅上方的[上下文压缩](#context-compression)。主备用链使用顶层的 `fallback_providers:` 列表 —— 参阅[备用提供商](/integrations/providers#fallback-providers)。三者都遵循相同的 provider/model/base_url 模式。
:::

### 辅助任务的每任务回退链

每个辅助任务都可以选择性地定义一个 `fallback_chain` —— 一个 provider/model 条目列表，当主要辅助 provider 因速率限制、网络连接问题或付费限制而失败时，Hermes 会尝试使用该列表：

```yaml
auxiliary:
  compression:
    provider: openrouter
    model: openai/gpt-4o-mini
    fallback_chain:
      - provider: nous
        model: deepseek/deepseek-chat
      - provider: openrouter
        model: google/gemini-2.5-flash
```

当主要辅助 provider（`openrouter` / `openai/gpt-4o-mini`）返回速率限制、连接超时或需要付费错误时，Hermes 将依次遍历 `fallback_chain`。它会跳过 provider 与已失败 provider 相同的条目，并尝试每个剩余条目，直到有一个成功或该链耗尽。如果所有回退都失败，Hermes 会回退到主 agent 模型作为最终的安全网。

每个条目支持与任何辅助任务配置相同的三个旋钮：

| 键 | 描述 |
|-----|-------------|
| `provider` | Provider 名称（`nous`、`openrouter`、`anthropic`、`gemini`、`main` 等） |
| `model` | 该 provider 的模型名称 |
| `base_url` | （可选）自定义 OpenAI 兼容端点 |

`fallback_chain` 适用于任何辅助任务 —— `compression`、`vision`、`web_extract`、`approval`、`skills_hub`、`mcp` 等。

### OpenRouter 路由和辅助任务的 Pareto Code

当辅助任务解析到 OpenRouter（显式或通过 `provider: "main"` 而您的主 agent 在 OpenRouter 上）时，主 agent 的 `provider_routing` 和 `openrouter.min_coding_score` 设置**不会传播** —— 按设计，每个辅助任务是独立的。要为特定辅助任务设置 OpenRouter provider 偏好或使用 [Pareto Code 路由器](/integrations/providers#openrouter-pareto-code-router)，请通过 `extra_body` 按任务设置：

```yaml
auxiliary:
  compression:
    provider: openrouter
    model: openrouter/pareto-code         # 为此任务使用 Pareto Code 路由器
    extra_body:
      provider:                            # OpenRouter provider 路由偏好
        order: [anthropic, google]         # 按此顺序尝试这些 providers
        sort: throughput                   # 或 "price" | "latency"
        # only: [anthropic]                # 限制到特定 provider
        # ignore: [deepinfra]              # 排除特定 providers
      plugins:                             # OpenRouter Pareto Code 路由器旋钮
        - id: pareto-router
          min_coding_score: 0.5            # 0.0–1.0；越高 = 更强的编码能力
```

形状与 OpenRouter 在聊天补全请求体中接受的内容一致。Hermes 原样转发整个 `extra_body`，因此 [openrouter.ai/docs](https://openrouter.ai/docs) 中记录的任何其他 OpenRouter 请求体字段都以相同方式工作。

### 更改视觉模型

使用 GPT-4o 而非 Gemini Flash 进行图像分析：

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

这些选项适用于**辅助任务配置**（`auxiliary:`、`compression:`、`fallback_model:`），而非您的主 `model.provider` 设置。

| Provider | 描述 | 要求 |
|----------|-------------|-------------|
| `"auto"` | 最佳可用（默认）。Vision 尝试 OpenRouter → Nous → Codex。 | — |
| `"openrouter"` | 强制 OpenRouter —— 路由到任何模型（Gemini、GPT-4o、Claude 等） | `OPENROUTER_API_KEY` |
| `"nous"` | 强制 Nous Portal | `hermes auth` |
| `"codex"` | 强制 Codex OAuth（ChatGPT 账户）。支持视觉（gpt-5.3-codex）。 | `hermes model` → Codex |
| `"minimax-oauth"` | 强制 MiniMax OAuth（浏览器登录，无需 API 密钥）。辅助任务使用 MiniMax-M2.7-highspeed。 | `hermes model` → MiniMax (OAuth) |
| `"xai-oauth"` | 强制 xAI Grok OAuth（SuperGrok 或 X Premium+ 订阅者的浏览器登录，无需 API 密钥）。相同的 OAuth token 涵盖聊天、TTS、图像、视频和转录。 | `hermes model` → xAI Grok OAuth (SuperGrok / Premium+) |
| `"main"` | 使用您的活跃自定义/主端点。可以来自 `OPENAI_BASE_URL` + `OPENAI_API_KEY` 或通过 `hermes model` / `config.yaml` 保存的自定义端点。适用于 OpenAI、本地模型或任何 OpenAI 兼容 API。**仅限辅助任务 —— 对 `model.provider` 无效。** | 自定义端点凭据 + 基础 URL |

当您希望附带任务绕过默认路由器时，主 provider 目录中的直接 API 密钥 providers 也在这里工作。配置 `GMI_API_KEY` 后，`gmi` 有效：

```yaml
auxiliary:
  compression:
    provider: "gmi"
    model: "anthropic/claude-opus-4.6"
```

对于 GMI 辅助路由，使用 GMI 的 `/v1/models` 端点返回的确切模型 ID。

### 常见设置

**使用直接自定义端点**（比 `provider: "main"` 对本地/自托管 API 更清晰）：
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
    model: "gpt-4o"       # 或 "gpt-4o-mini" 更便宜
```

**使用 OpenRouter 进行视觉**（路由到任何模型）：
```yaml
auxiliary:
  vision:
    provider: "openrouter"
    model: "openai/gpt-4o"      # 或 "google/gemini-2.5-flash" 等
```

**使用 Codex OAuth**（ChatGPT Pro/Plus 账户 —— 无需 API 密钥）：
```yaml
auxiliary:
  vision:
    provider: "codex"     # 使用您的 ChatGPT OAuth token
    # 模型默认为 gpt-5.3-codex（支持视觉）
```

**使用 MiniMax OAuth**（浏览器登录，无需 API 密钥）：
```yaml
model:
  default: MiniMax-M2.7
  provider: minimax-oauth
  base_url: https://api.minimax.io/anthropic
```
运行 `hermes model` 并选择 **MiniMax (OAuth)** 自动登录并设置此项。对于中国区域，基础 URL 将是 `https://api.minimaxi.com/anthropic`。完整演练请参阅 [MiniMax OAuth 指南](../guides/minimax-oauth.md)。

**使用本地/自托管模型：**
```yaml
auxiliary:
  vision:
    provider: "main"      # 使用您的活跃自定义端点
    model: "my-local-model"
```

`provider: "main"` 使用 Hermes 用于普通聊天的任何 provider —— 无论是命名的自定义 provider（例如 `beans`）、内置 provider（如 `openrouter`）还是旧版 `OPENAI_BASE_URL` 端点。

:::tip
如果您使用 Codex OAuth 作为主模型 provider，视觉会自动工作 —— 无需额外配置。Codex 包含在视觉的自动检测链中。
:::

:::warning
**视觉需要多模态模型。** 如果您设置 `provider: "main"`，请确保您的端点支持多模态/视觉 —— 否则图像分析将失败。
:::

### 环境变量（旧版）

辅助模型也可以通过环境变量配置。但是，`config.yaml` 是首选方法 —— 它更易于管理，并支持所有选项，包括 `base_url` 和 `api_key`。

| 设置 | 环境变量 |
|---------|---------------------|
| Vision provider | `AUXILIARY_VISION_PROVIDER` |
| Vision 模型 | `AUXILIARY_VISION_MODEL` |
| Vision 端点 | `AUXILIARY_VISION_BASE_URL` |
| Vision API 密钥 | `AUXILIARY_VISION_API_KEY` |
| Web 提取 provider | `AUXILIARY_WEB_EXTRACT_PROVIDER` |
| Web 提取模型 | `AUXILIARY_WEB_EXTRACT_MODEL` |
| Web 提取端点 | `AUXILIARY_WEB_EXTRACT_BASE_URL` |
| Web 提取 API 密钥 | `AUXILIARY_WEB_EXTRACT_API_KEY` |

压缩和回退模型设置仅限 config.yaml。

:::tip
运行 `hermes config` 查看您当前的辅助模型设置。覆盖仅在与默认值不同时显示。
:::

## 推理努力程度

控制模型在响应前进行多少"思考"：

```yaml
agent:
  reasoning_effort: ""   # 空 = 中等。选项：none、minimal、low、medium、high、xhigh、max、ultra
```

未设置时（默认），推理努力程度默认为"medium" —— 适合大多数任务的平衡级别。设置值会覆盖它 —— 更高的推理努力程度在复杂任务上提供更好的结果，但代价是更多 token 和延迟。

您也可以在运行时使用 `/reasoning` 命令更改推理努力程度：

```
/reasoning           # 显示当前努力程度和显示状态
/reasoning high      # 将推理努力程度设为 high
/reasoning none      # 禁用推理
/reasoning show      # 在每次响应上方显示模型思考
/reasoning hide      # 隐藏模型思考
```

## 工具使用强制

某些模型偶尔会将预期操作描述为文本而不是进行工具调用（"我会运行测试..."而不是实际调用终端）。工具使用强制会注入系统提示词指导，引导模型实际调用工具。

```yaml
agent:
  tool_use_enforcement: "auto"   # "auto" | true | false | ["model-substring", ...]
```

| 值 | 行为 |
|-------|----------|
| `"auto"`（默认） | 对匹配以下模型启用：`gpt`、`codex`、`gemini`、`gemma`、`grok`。对所有其他模型禁用（Claude、DeepSeek、Qwen 等）。 |
| `true` | 始终启用，无论模型如何。如果您注意到当前模型描述操作而不是执行操作，请使用此选项。 |
| `false` | 始终禁用，无论模型如何。 |
| `["gpt", "codex", "qwen", "llama"]` | 仅当模型名称包含列出的子字符串之一时启用（不区分大小写）。 |

### 注入的内容

启用后，系统提示词中可能会添加三层指导：

1. **通用工具使用强制**（所有匹配模型）—— 指示模型立即进行工具调用而不是描述意图，持续工作直到任务完成，永远不要以未来操作的承诺结束轮次。

2. **OpenAI 执行纪律**（仅限 GPT 和 Codex 模型）—— 针对 GPT 特定失败模式的额外指导：在部分结果上放弃工作、跳过先决条件查找、幻觉而不是使用工具、在未验证的情况下宣布"完成"。

3. **Google 操作指导**（仅限 Gemini 和 Gemma 模型）—— 简洁性、绝对路径、并行工具调用和编辑前验证模式。

这些对用户透明，仅影响系统提示词。已经可靠使用工具的模型（如 Claude）不需要此指导，这就是为什么 `"auto"` 排除它们。

### 何时开启

如果您使用的模型不在默认自动列表中，并注意到它经常描述它*会*做什么而不是实际去做，请设置 `tool_use_enforcement: true` 或将模型子字符串添加到列表中：

```yaml
agent:
  tool_use_enforcement: ["gpt", "codex", "gemini", "grok", "my-custom-model"]
```

## TTS 配置

```yaml
tts:
  provider: "edge"              # "edge" | "elevenlabs" | "openai" | "minimax" | "mistral" | "gemini" | "xai" | "neutts"
  speed: 1.0                    # 全局速度倍数（所有 provider 的回退）
  edge:
    voice: "en-US-AriaNeural"   # 322 种声音，74 种语言
    speed: 1.0                  # 速度倍数（转换为速率百分比，例如 1.5 → +50%）
  elevenlabs:
    voice_id: "pNInz6obpgDQGcFmaJgB"
    model_id: "eleven_multilingual_v2"
  openai:
    model: "gpt-4o-mini-tts"
    voice: "alloy"              # alloy、echo、fable、onyx、nova、shimmer
    speed: 1.0                  # 速度倍数（API 限制为 0.25–4.0）
    base_url: "https://api.openai.com/v1"  # 覆盖 OpenAI 兼容 TTS 端点
  minimax:
    speed: 1.0                  # 语音速度倍数
    # base_url: ""              # 可选：覆盖 OpenAI 兼容 TTS 端点
  mistral:
    model: "voxtral-mini-tts-2603"
    voice_id: "c69964a6-ab8b-4f8a-9465-ec0925096ec8"  # Paul - Neutral（默认）
  gemini:
    model: "gemini-2.5-flash-preview-tts"   # 或 gemini-2.5-pro-preview-tts
    voice: "Kore"               # 30 种预置声音：Zephyr、Puck、Kore、Enceladus 等
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

这控制 `text_to_speech` 工具和语音模式中的口语回复（CLI 中的 `/voice tts` 或消息 gateway）。

**速度回退层次：** provider 特定速度（例如 `tts.edge.speed`）→ 全局 `tts.speed` → `1.0` 默认值。设置全局 `tts.speed` 以在所有 provider 上应用统一速度，或按 provider 覆盖以进行精细控制。

## 显示设置

```yaml
display:
  tool_progress: all      # off | new | all | verbose
  tool_progress_command: false  # 在消息 gateway 中启用 /verbose 斜杠命令
  platforms: {}           # 每平台显示覆盖（见下文）
  tool_progress_overrides: {}  # 已弃用 —— 改用 display.platforms
  interim_assistant_messages: true  # Gateway：将自然的轮次中 assistant 更新作为单独消息发送
  skin: default           # 内置或自定义 CLI 皮肤（参阅 user-guide/features/skins）
  personality: "kawaii"  # 旧版外观字段，仍在某些摘要中显示
  compact: false          # 紧凑输出模式（减少空白）
  resume_display: full    # full（恢复时显示之前的消息）| minimal（仅单行）
  bell_on_complete: false # 当 agent 完成时播放终端铃声（适合长任务）
  show_reasoning: false   # 在每次响应上方显示模型推理/思考（用 /reasoning show|hide 切换）
  streaming: false        # 将 token 实时流式传输到终端
  show_cost: false        # 在 CLI 状态栏中显示估计 $ 成本
  timestamps: false       # 为 true 时，在 CLI/TUI 记录中为用户和 assistant 标签添加 [HH:MM] 时间戳前缀
  tool_preview_length: 0  # 工具调用预览的最大字符数（0 = 无限制，显示完整路径/命令）
  runtime_footer:         # Gateway：在最终回复中附加运行时上下文页脚
    enabled: false
    fields: ["model", "context_pct", "cwd"]
  file_mutation_verifier: true    # 当本轮 write_file/patch 调用失败时附加建议性页脚
  language: en            # 静态消息的 UI 语言（审批提示、部分 gateway 回复）。en | zh | zh-hant | ja | de | es | fr | tr | uk | af | ko | it | ga | pt | ru | hu
```

### 文件变更验证器

当 `display.file_mutation_verifier` 为 `true`（默认）时，每当本轮中 `write_file` 或 `patch` 调用失败且从未被对同一路径的成功写入取代时，Hermes 会在 assistant 的最终响应中附加一行建议。这捕获了"批量并行补丁，一半静默失败，模型总结成功"这类过度声明，而无需您在每次编辑后手动运行 `git status`。

示例页脚：

```
⚠️ File-mutation verifier: 3 file(s) were NOT modified this turn despite any wording above that may suggest otherwise. Run `git status` or `read_file` to confirm.
  • concepts/automatic-organization.md — [patch] Could not find match for old_string
  • concepts/lora.md — [patch] Could not find match for old_string
  • concepts/rag-pipeline.md — [patch] Could not find match for old_string
```

设置 `file_mutation_verifier: false`（或 `HERMES_FILE_MUTATION_VERIFIER=0`）以禁止页脚。验证器仅在轮次结束时有真实失败未解决时触发 —— 在同一轮次内重试失败补丁并成功的模型不会为该文件触发它。

### 静态消息的 UI 语言

`display.language` 设置翻译一小组静态面向用户的消息 —— CLI 审批提示、少数 gateway 斜杠命令回复（例如重启排空通知、"审批已过期"、"目标已清除"）。它**不**翻译 agent 响应、日志行、工具输出、错误回溯或斜杠命令描述 —— 这些保持英文。如果您希望 agent 本身用另一种语言回复，只需在您的提示词或系统消息中告诉它。

支持的值：`en`（默认）、`zh`（简体中文）、`zh-hant`（繁体中文）、`ja`（日语）、`de`（德语）、`es`（西班牙语）、`fr`（法语）、`tr`（土耳其语）、`uk`（乌克兰语）、`af`（南非荷兰语）、`ko`（韩语）、`it`（意大利语）、`ga`（爱尔兰语）、`pt`（葡萄牙语）、`ru`（俄语）、`hu`（匈牙利语）。未知值回退到英文。

您也可以使用 `HERMES_LANGUAGE` 环境变量按会话设置，它会覆盖配置值。

```yaml
display:
  language: zh   # CLI 审批提示以中文显示
```

| 模式 | 您看到的内容 |
|------|-------------|
| `off` | 静默 —— 仅最终响应 |
| `new` | 仅在工具更改时显示工具指示器 |
| `all` | 每次工具调用附带简短预览（默认） |
| `verbose` | 完整参数、结果和调试日志 |

在 CLI 中，使用 `/verbose` 循环切换这些模式。要在消息平台（Telegram、Discord、Slack 等）中使用 `/verbose`，请在上方的 `display` 部分设置 `tool_progress_command: true`。该命令将循环切换模式并保存到配置。

### 运行时元数据页脚（仅限 gateway）

当 `display.runtime_footer.enabled: true` 时，Hermes 在每个 gateway 轮次的**最终**消息中附加一个小型运行时上下文页脚。目前页脚可显示模型、上下文窗口百分比和当前工作目录。默认关闭；如果您的团队希望每个回复都包含这些来源信息，请按 gateway 选择加入。

```yaml
display:
  runtime_footer:
    enabled: true
    fields: ["model", "context_pct", "cwd"]   # 支持字段：model、context_pct、cwd
```

`/footer` 斜杠命令在任何会话中运行时切换此功能。

附加到 Telegram/Discord/Slack 回复的示例页脚：

```
— claude-opus-4.7 · 12 tool calls · 2m 14s · $0.042
```

只有轮次的**最终**消息获得页脚；中间更新保持干净。

### 每平台进度覆盖

不同平台有不同的详细程度需求。例如，Signal 无法编辑消息，因此每次进度更新都会成为单独的消息 —— 很嘈杂。使用 `display.platforms` 设置每平台模式：

```yaml
display:
  tool_progress: all          # 全局默认
  platforms:
    signal:
      tool_progress: 'off'    # 在 Signal 上静默进度
    telegram:
      tool_progress: verbose  # 在 Telegram 上详细进度
    slack:
      tool_progress: 'off'    # 在共享 Slack 工作区中保持安静
```

没有覆盖的平台回退到全局 `tool_progress` 值。有效平台键：`telegram`、`discord`、`slack`、`signal`、`whatsapp`、`matrix`、`mattermost`、`email`、`sms`、`homeassistant`、`dingtalk`、`feishu`、`wecom`、`weixin`、`bluebubbles`、`qqbot`。旧版 `display.tool_progress_overrides` 键仍可加载以向后兼容，但已弃用，并在首次加载时迁移到 `display.platforms`。

`interim_assistant_messages` 仅限 gateway。启用后，Hermes 将已完成的轮次中 assistant 更新作为单独的聊天消息发送。这与 `tool_progress` 无关，不需要 gateway 流式传输。

## 隐私

```yaml
privacy:
  redact_pii: false  # 从 LLM 上下文中删除 PII（仅限 gateway）
```

当 `redact_pii` 为 `true` 时，gateway 在将系统提示词发送到受支持平台上的 LLM 之前，会从中删除个人身份信息：

| 字段 | 处理方式 |
|-------|-----------|
| 电话号码（WhatsApp/Signal 上的用户 ID） | 哈希为 `user_<12-char-sha256>` |
| 用户 ID | 哈希为 `user_<12-char-sha256>` |
| 聊天 ID | 数字部分哈希，保留平台前缀（`telegram:<hash>`） |
| 主频道 ID | 数字部分哈希 |
| 用户名/昵称 | **不受影响**（用户选择的，公开可见） |

**平台支持：** 删除适用于 WhatsApp、Signal 和 Telegram。Discord 和 Slack 被排除，因为它们的提及系统（`<@user_id>`）需要 LLM 上下文中的真实 ID。

哈希是确定性的 —— 同一用户始终映射到同一哈希，因此模型仍然可以在群聊中区分用户。路由和传递在内部使用原始值。

## 语音转文字（STT）

```yaml
stt:
  provider: "local"            # "local" | "groq" | "openai" | "mistral"
  local:
    model: "base"              # tiny、base、small、medium、large-v3
  openai:
    model: "whisper-1"         # whisper-1 | gpt-4o-mini-transcribe | gpt-4o-transcribe
  # model: "whisper-1"         # 旧版回退键仍受支持
```

Provider 行为：

- `local` 使用在您机器上运行的 `faster-whisper`。使用 `pip install faster-whisper` 单独安装。
- `groq` 使用 Groq 的 Whisper 兼容端点，读取 `GROQ_API_KEY`。
- `openai` 使用 OpenAI 语音 API，读取 `VOICE_TOOLS_OPENAI_KEY`。

如果请求的 provider 不可用，Hermes 按此顺序自动回退：`local` → `groq` → `openai`。

Groq 和 OpenAI 模型覆盖由环境变量驱动：

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
  auto_tts: false               # /voice on 时自动启用口语回复
  beep_enabled: true            # 在 CLI 语音模式中播放录音开始/停止提示音
  silence_threshold: 200        # 语音检测的 RMS 阈值
  silence_duration: 3.0         # 自动停止前的静默秒数
```

在 CLI 中使用 `/voice on` 启用麦克风模式，使用 `record_key` 开始/停止录音，使用 `/voice tts` 切换口语回复。端到端设置和平台特定行为请参阅[语音模式](/user-guide/features/voice-mode)。

## 流式传输

将 token 实时流式传输到终端或消息平台，而不是等待完整响应。

### CLI 流式传输

```yaml
display:
  streaming: true         # 实时将 token 流式传输到终端
  show_reasoning: true    # 同时流式传输推理/思考 token（可选）
```

启用后，响应在流式传输框内逐 token 出现。工具调用仍然静默捕获。如果 provider 不支持流式传输，它会自动回退到正常显示。

### Gateway 流式传输（Telegram、Discord、Slack）

```yaml
streaming:
  enabled: true           # 启用渐进式消息编辑
  transport: edit         # "edit"（渐进式消息编辑）或 "off"
  edit_interval: 0.3      # 消息编辑之间的秒数
  buffer_threshold: 40    # 强制编辑刷新前的字符数
  cursor: " ▉"            # 流式传输期间显示的光标
  fresh_final_after_seconds: 0    # 预览超过此时间时选择发送新的最终消息（Telegram）
```

启用后，bot 在第一个 token 时发送消息，然后随着更多 token 到来渐进式编辑它。不支持消息编辑的平台（Signal、Email、Home Assistant）在第一次尝试时自动检测 —— 该会话的流式传输被优雅地禁用，不会产生大量消息。

对于不带渐进式 token 编辑的独立自然轮次中 assistant 更新，请设置 `display.interim_assistant_messages: true`。

**溢出处理：** 如果流式传输的文本超过平台的消息长度限制（约 4096 字符），当前消息被最终化，新消息自动开始。

**新的最终消息（Telegram）：** Telegram 的 `editMessageText` 保留原始消息时间戳，因此长时间运行的流式回复即使在完成后也会保留第一个 token 的时间戳。设置 `fresh_final_after_seconds > 0` 可选择将旧预览作为全新的最终消息传递，并尽力删除旧预览。默认值为 `0`，始终就地最终化流式回复，避免某些客户端短暂显示重复消息再删除其中一条。

:::note
主开关 `streaming.enabled` 默认为 `false`——在你启用之前不会有任何流式传输。启用后，是否流式传输按**平台**决定：Telegram 默认带有 `display.platforms.telegram.streaming: true`（流式传输），Discord 为 `display.platforms.discord.streaming: false`（不流式传输）。因此启用流式传输后，Telegram 开箱即用地流式传输，Discord 在你修改其开关之前仍使用整条消息回复。你可以在仪表盘的 **Channels** 开关中或直接在 `~/.hermes/config.yaml` 中调整这些按平台的开关。
:::

## 群聊会话隔离

控制共享聊天是每个房间保持一个对话还是每个参与者一个对话：

```yaml
group_sessions_per_user: true  # true = 群组/频道中每用户隔离，false = 每个聊天一个共享会话
```

- `true` 是默认和推荐设置。在 Discord 频道、Telegram 群组、Slack 频道和类似共享上下文中，当平台提供用户 ID 时，每个发送者获得自己的会话。
- `false` 恢复到旧的共享房间行为。如果您明确希望 Hermes 将频道视为一个协作对话，这可能有用，但这也意味着用户共享上下文、token 成本和中断状态。
- 私信不受影响。Hermes 仍然像往常一样通过聊天/DM ID 键入 DM。
- 线程与其父频道保持隔离；使用 `true` 时，每个参与者在线程内也获得自己的会话。

有关行为详情和示例，请参阅[会话](/user-guide/sessions)和 [Discord 指南](/user-guide/messaging/discord)。

## 未授权 DM 行为

控制当未知用户发送私信时 Hermes 的行为：

```yaml
unauthorized_dm_behavior: pair

whatsapp:
  unauthorized_dm_behavior: ignore
```

- `pair` 是默认值。Hermes 拒绝访问，但在 DM 中回复一次性配对码。
- `ignore` 静默丢弃未授权的 DM。
- 平台部分覆盖全局默认值，因此您可以在广泛范围内保持配对启用，同时使一个平台更安静。

## 快速命令

定义自定义命令，这些命令要么在不调用 LLM 的情况下运行 shell 命令，要么将一个斜杠命令别名为另一个。Exec 快速命令是零 token 的，对于从消息平台（Telegram、Discord 等）进行快速服务器检查或实用脚本很有用。

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

用法：在 CLI 或任何消息平台中输入 `/status`、`/disk`、`/update`、`/gpu` 或 `/restart`。`exec` 命令在宿主本地运行并直接返回输出 —— 无 LLM 调用，不消耗 token。`alias` 命令重写为配置的斜杠命令目标。

- **30 秒超时** —— 长时间运行的命令被终止并显示错误消息
- **优先级** —— 快速命令在技能命令之前检查，因此您可以覆盖技能名称
- **自动补全** —— 快速命令在调度时解析，不显示在内置斜杠命令自动补全表中
- **类型** —— 支持的类型为 `exec` 和 `alias`；其他类型显示错误
- **到处可用** —— CLI、Telegram、Discord、Slack、WhatsApp、Signal、Email、Home Assistant

仅字符串的 prompt 快捷方式不是有效的快速命令。对于可重用的 prompt 工作流，请创建技能或别名到现有斜杠命令。

## 人类延迟

在消息平台中模拟类人响应节奏：

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
  max_tool_calls: 50           # 代码执行中的最大工具调用次数
```

**`mode`** 控制脚本的工作目录和 Python 解释器：

- **`project`**（默认）—— 脚本在会话的工作目录中以活跃 virtualenv/conda 环境的 python 运行。项目依赖（`pandas`、`torch`、项目包）和相对路径（`.env`、`./data.csv`）自然解析，与 `terminal()` 看到的一致。
- **`strict`** —— 脚本在临时暂存目录中以 `sys.executable`（Hermes 自己的 python）运行。最大可重现性，但项目依赖和相对路径不会解析。

环境清理（删除 `*_API_KEY`、`*_TOKEN`、`*_SECRET`、`*_PASSWORD`、`*_CREDENTIAL`、`*_PASSWD`、`*_AUTH`）和工具白名单在两种模式下完全相同 —— 切换模式不会改变安全态势。

## Web 搜索后端

`web_search` 和 `web_extract` 工具支持五种后端 provider。在 `config.yaml` 中或通过 `hermes tools` 配置后端：

```yaml
web:
  backend: firecrawl    # firecrawl | searxng | parallel | tavily | exa

  # 或使用每功能键混合 provider（例如免费搜索 + 付费提取）：
  search_backend: "searxng"
  extract_backend: "firecrawl"
```

| 后端 | 环境变量 | 搜索 | 提取 |
|---------|---------|--------|---------|
| **Firecrawl**（默认） | `FIRECRAWL_API_KEY` | ✔ | ✔ |
| **SearXNG** | `SEARXNG_URL` | ✔ | — |
| **Parallel** | `PARALLEL_API_KEY` | ✔ | ✔ |
| **Tavily** | `TAVILY_API_KEY` | ✔ | ✔ |
| **Exa** | `EXA_API_KEY` | ✔ | ✔ |

**后端选择：** 如果未设置 `web.backend`，后端从可用的 API 密钥自动检测。如果仅设置了 `SEARXNG_URL`，使用 SearXNG。如果仅设置了 `EXA_API_KEY`，使用 Exa。如果仅设置了 `TAVILY_API_KEY`，使用 Tavily。如果仅设置了 `PARALLEL_API_KEY`，使用 Parallel。否则 Firecrawl 是默认值。

**SearXNG** 是一个免费、自托管、尊重隐私的元搜索引擎，查询 70+ 个搜索引擎。无需 API 密钥 —— 只需将 `SEARXNG_URL` 设置为您的实例（例如 `http://localhost:8080`）。SearXNG 仅限搜索；`web_extract` 需要单独的提取 provider（设置 `web.extract_backend`）。Docker 设置说明请参阅 [Web 搜索设置指南](/user-guide/features/web-search)。

**自托管 Firecrawl：** 设置 `FIRECRAWL_API_URL` 指向您自己的实例。设置自定义 URL 后，API 密钥变为可选（在服务器上设置 `USE_DB_AUTHENTICATION=***` 以禁用认证）。

**Parallel 搜索模式：** 设置 `PARALLEL_SEARCH_MODE` 控制搜索行为 —— `fast`、`one-shot` 或 `agentic`（默认：`agentic`）。

**Exa：** 在 `~/.hermes/.env` 中设置 `EXA_API_KEY`。支持 `category` 过滤（`company`、`research paper`、`news`、`people`、`personal site`、`pdf`）和域名/日期过滤器。

## 浏览器

配置浏览器自动化行为：

```yaml
browser:
  inactivity_timeout: 120        # 自动关闭空闲会话前的秒数
  command_timeout: 30             # 浏览器命令超时（截图、导航等）（秒）
  record_sessions: false         # 自动将浏览器会话录制为 WebM 视频到 ~/.hermes/browser_recordings/
  # 可选 CDP 覆盖 —— 设置后，Hermes 直接附加到您自己的
  # Chromium 系浏览器（通过 /browser connect），而不是启动无头浏览器。
  cdp_url: ""
  # 对话框监督器 —— 控制当 CDP 后端附加时（Browserbase、本地 Chromium 系
  # 浏览器通过 /browser connect）如何处理原生 JS 对话框（alert/confirm/prompt）。
  # 在 Camofox 和默认本地 agent 浏览器模式下忽略。
  dialog_policy: must_respond    # must_respond | auto_dismiss | auto_accept
  dialog_timeout_s: 300          # must_respond 下的安全自动关闭（秒）
  camofox:
    managed_persistence: false   # 为 true 时，Camofox 会话跨重启持久化 cookie/登录
    user_id: ""                  # 可选的外部管理 Camofox userId
    session_key: ""              # Hermes 创建标签页时发送的可选会话密钥
    adopt_existing_tab: false    # 在创建新标签页之前重用此身份的现有标签页
```

**对话框策略：**

- `must_respond`（默认）—— 捕获对话框，在 `browser_snapshot.pending_dialogs` 中显示，等待 agent 调用 `browser_dialog(action=...)`。在 `dialog_timeout_s` 秒内无响应后，对话框被自动关闭以防止页面的 JS 线程永久停滞。
- `auto_dismiss` —— 捕获，立即关闭。Agent 仍然在事后的 `browser_snapshot.recent_dialogs` 中看到对话框记录，`closed_by="auto_policy"`。
- `auto_accept` —— 捕获，立即接受。适用于有激进 `beforeunload` 提示的页面。

完整对话框工作流请参阅[浏览器功能页面](./features/browser.md#browser_dialog)。

浏览器工具集支持多个 provider。有关 Browserbase、Browser Use 和本地 Chromium 系 CDP 设置的详细信息，请参阅[浏览器功能页面](/user-guide/features/browser)。

## 时区

使用 IANA 时区字符串覆盖服务器本地时区。影响日志中的时间戳、cron 调度和系统提示词时间注入。

```yaml
timezone: "America/New_York"   # IANA 时区（默认："" = 服务器本地时间）
```

支持的值：任何 IANA 时区标识符（例如 `America/New_York`、`Europe/London`、`Asia/Kolkata`、`UTC`）。留空或省略以使用服务器本地时间。

## Discord

为消息 gateway 配置 Discord 特定行为：

```yaml
discord:
  require_mention: true          # 在服务器频道中需要 @提及才能响应
  free_response_channels: ""     # 逗号分隔的频道 ID，bot 在这些频道无需 @提及即可响应
  auto_thread: true              # 在频道中 @提及时自动创建线程
```

- `require_mention` —— 为 `true`（默认）时，bot 仅在服务器频道中被 `@BotName` 提及时响应。DM 始终无需提及即可工作。
- `free_response_channels` —— 逗号分隔的频道 ID 列表，bot 在这些频道对每条消息响应，无需提及。
- `auto_thread` —— 为 `true`（默认）时，频道中的提及会自动为对话创建线程，保持频道整洁（类似 Slack 线程）。

## 安全

预执行安全扫描和机密脱敏：

```yaml
security:
  redact_secrets: false          # 在工具输出和日志中脱敏 API 密钥模式（默认关闭）
  tirith_enabled: true           # 为终端命令启用 Tirith 安全扫描
  tirith_path: "tirith"          # tirith 二进制文件路径（默认：$PATH 中的 "tirith"）
  tirith_timeout: 5              # 等待 tirith 扫描的秒数
  tirith_fail_open: true         # 如果 tirith 不可用，允许命令执行
  website_blocklist:             # 参见下方网站黑名单部分
    enabled: false
    domains: []
    shared_files: []
```

- `redact_secrets` —— 为 `true` 时，自动检测并脱敏工具输出中看起来像 API 密钥、token 和密码的模式，然后再进入对话上下文和日志。**默认关闭** —— 如果您经常在工具输出中处理真实凭据并希望有安全网，请启用。显式设置为 `true` 以开启。
- `tirith_enabled` —— 为 `true` 时，终端命令在执行前由 [Tirith](https://github.com/sheeki03/tirith) 扫描以检测潜在危险操作。
- `tirith_path` —— tirith 二进制文件的路径。如果 tirith 安装在非标准位置，请设置此项。
- `tirith_timeout` —— 等待 tirith 扫描的最大秒数。如果扫描超时，命令继续执行。
- `tirith_fail_open` —— 为 `true`（默认）时，如果 tirith 不可用或失败，允许命令执行。设置为 `false` 以在 tirith 无法验证时阻止命令。

## 网站黑名单

阻止 agent 的 web 和浏览器工具访问特定域名：

```yaml
security:
  website_blocklist:
    enabled: false               # 启用 URL 阻止（默认：false）
    domains:                     # 被阻止的域名模式列表
      - "*.internal.company.com"
      - "admin.example.com"
      - "*.local"
    shared_files:                # 从外部文件加载额外规则
      - "/etc/hermes/blocked-sites.txt"
```

启用后，任何匹配被阻止域名模式的 URL 在 web 或浏览器工具执行之前都会被拒绝。这适用于 `web_search`、`web_extract`、`browser_navigate` 以及任何访问 URL 的工具。

域名规则支持：
- 精确域名：`admin.example.com`
- 通配符子域名：`*.internal.company.com`（阻止所有子域名）
- TLD 通配符：`*.local`

共享文件每行包含一条域名规则（空行和 `#` 注释被忽略）。缺失或不可读的文件记录警告，但不禁用其他 web 工具。

策略缓存 30 秒，因此配置更改无需重启即可快速生效。

## 智能审批

控制 Hermes 如何处理潜在危险命令：

```yaml
approvals:
  mode: manual   # manual | smart | off
```

| 模式 | 行为 |
|------|----------|
| `manual`（默认） | 在执行任何被标记的命令之前提示用户。在 CLI 中显示交互式审批对话框。在消息中排队待处理的审批请求。 |
| `smart` | 使用辅助 LLM 评估被标记的命令是否真正危险。低风险命令以会话级持久性自动批准。真正有风险的命令升级给用户。 |
| `off` | 跳过所有审批检查。等同于 `HERMES_YOLO_MODE=true`。**谨慎使用。** |

智能模式对于减少审批疲劳特别有用 —— 它让 agent 在安全操作上更自主地工作，同时仍然捕获真正破坏性的命令。

:::warning
设置 `approvals.mode: off` 会禁用终端命令的所有安全检查。仅在受信任的沙箱环境中使用。
:::

## 检查点

破坏性文件操作之前的自动文件系统快照。详情请参阅[检查点与回滚](/user-guide/checkpoints-and-rollback)。

```yaml
checkpoints:
  enabled: false                 # 启用自动检查点（也可：hermes chat --checkpoints）。默认：false（选择加入）。
  max_snapshots: 20              # 每个目录保留的最大检查点数（默认：20）
```


## 委托

为委托工具配置子 agent 行为：

```yaml
delegation:
  # model: "google/gemini-3-flash-preview"  # 覆盖模型（空 = 继承父级）
  # provider: "openrouter"                  # 覆盖 provider（空 = 继承父级）
  # base_url: "http://localhost:1234/v1"    # 直接 OpenAI 兼容端点（优先于 provider）
  # api_key: "local-key"                    # base_url 的 API 密钥（回退到 OPENAI_API_KEY）
  # api_mode: ""                            # base_url 的线路协议："chat_completions"、"codex_responses" 或 "anthropic_messages"。空 = 从 URL 自动检测（例如 /anthropic 后缀 → anthropic_messages）。对启发式无法检测的非标准端点显式设置。
  max_concurrent_children: 3                # 每批并行子 agent 数（下限 1，无上限）。也可通过 DELEGATION_MAX_CONCURRENT_CHILDREN 环境变量设置。
  max_spawn_depth: 1                        # 委托树深度上限（1-3，截断）。1 = 扁平（默认）：父级生成无法委托的叶子。2 = 编排器子级可以生成叶子孙级。3 = 三级。
  orchestrator_enabled: true                # 全局终止开关。为 false 时，role="orchestrator" 被忽略，每个子级无论 max_spawn_depth 如何都被强制为叶子。
```

**子 agent provider:model 覆盖：** 默认情况下，子 agent 继承父 agent 的 provider 和模型。设置 `delegation.provider` 和 `delegation.model` 将子 agent 路由到不同的 provider:model 对 —— 例如，在您的主 agent 运行昂贵推理模型时，为范围较窄的子任务使用便宜/快速的模型。

**直接端点覆盖：** 如果您想要明显的自定义端点路径，请设置 `delegation.base_url`、`delegation.api_key` 和 `delegation.model`。这将子 agent 直接发送到该 OpenAI 兼容端点，并优先于 `delegation.provider`。如果省略 `delegation.api_key`，Hermes 仅回退到 `OPENAI_API_KEY`。

**线路协议（`api_mode`）：** Hermes 从 `delegation.base_url` 自动检测线路协议（例如以 `/anthropic` 结尾的路径 → `anthropic_messages`；Codex/原生 Anthropic/Kimi-coding 主机名保留其现有检测）。对于启发式无法分类的端点 —— 例如 Azure AI Foundry、MiniMax、Zhipu GLM 或前置 Anthropic 形状后端的 LiteLLM 代理 —— 请将 `delegation.api_mode` 显式设置为 `chat_completions`、`codex_responses` 或 `anthropic_messages` 之一。留空（默认）以保持自动检测。

委托 provider 使用与 CLI/gateway 启动相同的凭据解析。所有配置的 provider 均受支持：`openrouter`、`nous`、`copilot`、`zai`、`kimi-coding`、`minimax`、`minimax-cn`。设置 provider 时，系统自动解析正确的基础 URL、API 密钥和 API 模式 —— 无需手动凭据连接。

**优先级：** 配置中的 `delegation.base_url` → 配置中的 `delegation.provider` → 父 provider（继承）。配置中的 `delegation.model` → 父模型（继承）。仅设置 `model` 而不设置 `provider` 仅更改模型名称，同时保留父级凭据（适用于在同一 provider（如 OpenRouter）内切换模型）。

**宽度和深度：** `max_concurrent_children` 限制每批并行运行的子 agent 数量（默认 `3`，下限 1，无上限）。也可通过 `DELEGATION_MAX_CONCURRENT_CHILDREN` 环境变量设置。当模型提交的 `tasks` 数组超过上限时，`delegate_task` 返回工具错误解释限制，而不是静默截断。`max_spawn_depth` 控制委托树深度（截断到 1-3）。在默认 `1` 时，委托是扁平的：子级无法生成孙级，传递 `role="orchestrator"` 静默降级为 `leaf`。提升到 `2` 使编排器子级可以生成叶子孙级；`3` 用于三级树。Agent 通过 `role="orchestrator"` 按调用选择编排；`orchestrator_enabled: false` 强制每个子级回到叶子，无论如何。成本呈乘法增长 —— 在 `max_spawn_depth: 3` 和 `max_concurrent_children: 3` 时，树可以达到 3×3×3 = 27 个并发叶子 agent。使用模式请参阅[子 Agent 委托 → 深度限制和嵌套编排](features/delegation.md#depth-limit-and-nested-orchestration)。

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
| `SOUL.md` | **主要 agent 身份** —— 定义 agent 是谁（系统提示词第 #1 槽位） | `~/.hermes/SOUL.md` 或 `$HERMES_HOME/SOUL.md` |
| `.hermes.md` / `HERMES.md` | 项目特定指令（最高优先级） | 向上走到 git 根目录 |
| `AGENTS.md` | 项目特定指令、编码规范 | 递归目录遍历 |
| `CLAUDE.md` | Claude Code 上下文文件（也会检测） | 仅工作目录 |
| `.cursorrules` | Cursor IDE 规则（也会检测） | 仅工作目录 |
| `.cursor/rules/*.mdc` | Cursor 规则文件（也会检测） | 仅工作目录 |

- **SOUL.md** 是 agent 的主要身份。它占据系统提示词的第 #1 槽位，完全替换内置的默认身份。编辑它以完全自定义 agent 是谁。
- 如果 SOUL.md 缺失、为空或无法加载，Hermes 回退到内置默认身份。
- **项目上下文文件使用优先级系统** —— 仅加载一种类型（第一个匹配优先）：`.hermes.md` → `AGENTS.md` → `CLAUDE.md` → `.cursorrules`。SOUL.md 始终独立加载。
- **AGENTS.md** 是分层的：如果子目录也有 AGENTS.md，所有都会合并。
- 如果 `SOUL.md` 不存在，Hermes 会自动生成默认的 `SOUL.md`。
- 所有加载的上下文文件上限为 20,000 字符，并进行智能截断。

另请参阅：
- [个性与 SOUL.md](/user-guide/features/personality)
- [上下文文件](/user-guide/features/context-files)

## 工作目录

| 上下文 | 默认值 |
|---------|---------|
| **CLI（`hermes`）** | 运行命令的当前目录 |
| **消息 gateway** | 主目录 `~`（用 `MESSAGING_CWD` 覆盖） |
| **Docker / Singularity / Modal / SSH** | 容器或远程机器内用户的主目录 |

覆盖工作目录：
```bash
# 在 ~/.hermes/.env 或 ~/.hermes/config.yaml 中：
MESSAGING_CWD=/home/myuser/projects    # Gateway 会话
TERMINAL_CWD=/workspace                # 所有终端会话
```
