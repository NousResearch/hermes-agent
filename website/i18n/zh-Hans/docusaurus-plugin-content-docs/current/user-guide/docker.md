---
sidebar_position: 7
title: "Docker"
description: "在 Docker 中运行 Hermes Agent 并使用 Docker 作为终端后端"
---

# Hermes Agent — Docker

Docker 与 Hermes Agent 有两种不同的交集方式：

1. **在 Docker 中运行 Hermes** — agent 本身在容器内运行（这是本页的主要焦点）
2. **Docker 作为终端后端** — agent 在您的主机上运行，但在单个持久 Docker sandbox 容器中执行每个命令，该容器跨工具调用、`/new` 和子 agent 存活整个 Hermes 进程（请参阅 [配置 → Docker 后端](./configuration.md#docker-backend)）

本页面涵盖选项 1。容器将所有用户数据（配置、API 密钥、会话、skills、内存）存储在挂载到主机的 `/opt/data` 目录中的单个目录中。镜像本身是无状态的，可以通过拉取新版本进行升级而不会丢失任何配置。

## 快速开始

如果您是第一次运行 Hermes Agent，请在主机上创建数据目录并以交互方式启动容器以运行设置向导：

```sh
mkdir -p ~/.hermes
docker run -it --rm \
  -v ~/.hermes:/opt/data \
  nousresearch/hermes-agent setup
```

这会使您进入设置向导，它会提示您输入 API 密钥并将它们写入 `~/.hermes/.env`。您只需要执行一次。此时强烈建议设置一个聊天系统以使 gateway 正常工作。

## 在 gateway 模式下运行

配置后，在后台运行容器作为持久 gateway（Telegram、Discord、Slack、WhatsApp 等）：

```sh
docker run -d \
  --name hermes \
  --restart unless-stopped \
  -v ~/.hermes:/opt/data \
  -p 8642:8642 \
  nousresearch/hermes-agent gateway run
```

端口 8642 暴露 gateway 的 [OpenAI 兼容 API 服务器](./features/api-server.md) 和健康端点。如果您只使用聊天平台（Telegram、Discord 等），这是可选的，但如果您想要仪表板或外部工具访问 gateway，则是必需的。

注意：API 服务器受 `API_SERVER_ENABLED=true` 限制。要在容器内将暴露范围从 `127.0.0.1` 扩展到外部，同时设置 `API_SERVER_HOST=0.0.0.0` 和 `API_SERVER_KEY`（最少 8 个字符——使用 `openssl rand -hex 32` 生成）。示例：

```sh
docker run -d \
  --name hermes \
  --restart unless-stopped \
  -v ~/.hermes:/opt/data \
  -p 8642:8642 \
  -e API_SERVER_ENABLED=true \
  -e API_SERVER_HOST=0.0.0.0 \
  -e API_SERVER_KEY=your_api_key_here \
  -e API_SERVER_CORS_ORIGINS='*' \
  nousresearch/hermes-agent gateway run
```

在面向互联网的机器上打开任何端口都是安全风险。除非您了解风险，否则不应这样做。

## 运行仪表板

内置 Web 仪表板作为可选的 side-process 在与 gateway 相同的容器内运行。设置 `HERMES_DASHBOARD=1` 并同时暴露 gateway 的 `8642` 旁边的 `9119` 端口：

```sh
docker run -d \
  --name hermes \
  --restart unless-stopped \
  -v ~/.hermes:/opt/data \
  -p 8642:8642 \
  -p 9119:9119 \
  -e HERMES_DASHBOARD=1 \
  nousresearch/hermes-agent gateway run
```

入口点在 `exec`-主命令之前在后台启动 `hermes dashboard`（以非 root `hermes` 用户身份运行）。在 `docker logs` 中，仪表板输出以 `[dashboard]` 为前缀，因此很容易与 gateway 日志分开。

| 环境变量 | 描述 | 默认 |
|---------------------|-------------|---------|
| `HERMES_DASHBOARD` | 设置为 `1`（或 `true` / `yes`）以与主命令一起启动仪表板 | *（未设置 — 不启动仪表板）* |
| `HERMES_DASHBOARD_HOST` | 仪表板 HTTP 服务器的绑定地址 | `0.0.0.0` |
| `HERMES_DASHBOARD_PORT` | 仪表板 HTTP 服务器的端口 | `9119` |
| `HERMES_DASHBOARD_TUI` | 设置为 `1` 以暴露浏览器内聊天标签（通过 PTY/WebSocket 嵌入的 `hermes --tui`） | *（未设置）* |

默认的 `HERMES_DASHBOARD_HOST=0.0.0.0` 是主机通过发布端口到达仪表板所必需的；入口点在此情况下自动传递 `--insecure` 给 `hermes dashboard`。覆盖为 `127.0.0.1`如果您想将仪表板限制为仅容器内访问（例如，在 sidecar 中的反向代理后面）。

:::note
仪表板 side-process **不受监督** — 如果它崩溃了，它会保持关闭直到容器重新启动。不支持将其作为单独的容器运行：仪表板的 gateway 存活检测需要与 gateway 进程共享 PID 命名空间。
:::

## 交互式运行（CLI 聊天）

打开与运行数据目录的交互式聊天会话：

```sh
docker run -it --rm \
  -v ~/.hermes:/opt/data \
  nousresearch/hermes-agent
```

或者，如果您已经打开了运行中容器的终端（例如通过 Docker Desktop），只需运行：

```sh
/opt/hermes/.venv/bin/hermes
```

## 持久卷

`/opt/data` 卷是所有 Hermes 状态的唯一真实来源。它映射到您主机的 `~/.hermes/` 目录，包含：

| 路径 | 内容 |
|------|----------|
| `.env` | API 密钥和密钥 |
| `config.yaml` | 所有 Hermes 配置 |
| `SOUL.md` | Agent 人格/身份 |
| `sessions/` | 对话历史 |
| `memories/` | 持久内存存储 |
| `skills/` | 已安装的 skills |
| `cron/` | 计划任务定义 |
| `hooks/` | 事件钩子 |
| `logs/` | 运行时日志 |
| `skins/` | 自定义 CLI 皮肤 |

:::warning
永远不要对同一个数据目录同时运行两个 Hermes **gateway** 容器——会话文件和内存存储不是为并发写入访问设计的。
:::

## 多配置文件支持

Hermes 支持 [多配置文件](../reference/profile-commands.md) — 独立的 `~/.hermes/` 目录，让您可以从单个安装运行独立的 agent（不同的 SOUL、skills、内存、会话、凭证）。**在 Docker 下运行时，不建议使用 Hermes 内置的多配置文件功能。**

相反，推荐的模式是**每个配置文件一个容器**，每个容器将其自己的主机目录挂载为 `/opt/data`：

```sh
# 工作配置文件
docker run -d \
  --name hermes-work \
  --restart unless-stopped \
  -v ~/.hermes-work:/opt/data \
  -p 8642:8642 \
  nousresearch/hermes-agent gateway run

# 个人配置文件
docker run -d \
  --name hermes-personal \
  --restart unless-stopped \
  -v ~/.hermes-personal:/opt/data \
  -p 8643:8642 \
  nousresearch/hermes-agent gateway run
```

为什么在 Docker 中使用独立容器而非配置文件：

- **隔离** — 每个容器有自己的文件系统、进程表和资源限制。一个配置文件中的崩溃、依赖变更或失控会话不会影响另一个。
- **独立的生命周期** — 分别升级、重启、暂停或回滚每个 agent（`docker restart hermes-work` 让 `hermes-personal` 保持不变）。
- **清晰的端口和网络分离** — 每个 gateway 绑定自己的主机端口；聊天平台或 API 服务器之间没有交叉对话的风险。
- **更简单的心理模型** — 容器*就是*配置文件。备份、迁移和权限都遵循挂载的目录，没有额外的 `--profile` 标志需要记住。
- **避免并发写入风险** — 上述关于永远不要对同一个数据目录运行两个 gateway 的警告仍然适用于单个容器内的配置文件。

在 Docker Compose 中，这只是意味着为每个配置文件声明一个服务，具有不同的 `container_name`、`volumes` 和 `ports`：

```yaml
services:
  hermes-work:
    image: nousresearch/hermes-agent:latest
    container_name: hermes-work
    restart: unless-stopped
    command: gateway run
    ports:
      - "8642:8642"
    volumes:
      - ~/.hermes-work:/opt/data

  hermes-personal:
    image: nousresearch/hermes-agent:latest
    container_name: hermes-personal
    restart: unless-stopped
    command: gateway run
    ports:
      - "8643:8642"
    volumes:
      - ~/.hermes-personal:/opt/data
```

## 环境变量转发

API 密钥从容器内的 `/opt/data/.env` 读取。您也可以直接传递环境变量：

```sh
docker run -it --rm \
  -v ~/.hermes:/opt/data \
  -e ANTHROPIC_API_KEY="sk-ant-..." \
  -e OPENAI_API_KEY="sk-..." \
  nousresearch/hermes-agent
```

直接 `-e` 标志会覆盖 `.env` 中的值。这对于 CI/CD 或 secrets-manager 集成很有用，您不希望密钥留在磁盘上。

## Docker Compose 示例

对于同时包含 gateway 和仪表板的持久部署，`docker-compose.yaml` 很方便：

```yaml
services:
  hermes:
    image: nousresearch/hermes-agent:latest
    container_name: hermes
    restart: unless-stopped
    command: gateway run
    ports:
      - "8642:8642"   # gateway API
      - "9119:9119"   # 仪表板（仅在 HERMES_DASHBOARD=1 时可访问）
    volumes:
      - ~/.hermes:/opt/data
    environment:
      - HERMES_DASHBOARD=1
      # 取消注释以转发特定环境变量而不是使用 .env 文件：
      # - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      # - OPENAI_API_KEY=${OPENAI_API_KEY}
      # - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: "2.0"
```

使用 `docker compose up -d` 启动，使用 `docker compose logs -f` 查看日志。仪表板输出以 `[dashboard]` 为前缀，因此很容易从 gateway 日志中过滤。

## 资源限制

Hermes 容器需要适度资源。推荐的最低配置：

| 资源 | 最低 | 推荐 |
|----------|---------|-------------|
| 内存 | 1 GB | 2–4 GB |
| CPU | 1 核 | 2 核 |
| 磁盘（数据卷） | 500 MB | 2+ GB（随会话/skills 增长） |

浏览器自动化（Playwright/Chromium）是最耗内存的功能。如果您不需要浏览器工具，1 GB 就足够了。启用浏览器工具时，至少分配 2 GB。

在 Docker 中设置限制：

```sh
docker run -d \
  --name hermes \
  --restart unless-stopped \
  --memory=4g --cpus=2 \
  -v ~/.hermes:/opt/data \
  nousresearch/hermes-agent gateway run
```

## Dockerfile 的作用

官方镜像基于 `debian:13.4`，包括：

- 带有所有 Hermes 依赖项的 Python 3（`uv pip install -e ".[all]"`）
- Node.js + npm（用于浏览器自动化和 WhatsApp 桥接）
- Playwright 和 Chromium（`npx playwright install --with-deps chromium --only-shell`）
- ripgrep、ffmpeg、git 和 tini 作为系统工具
- **`docker-cli`** — 使容器内的 agent 能够驱动主机的 Docker 守护进程（通过绑定挂载 `/var/run/docker.sock` 选择加入）用于 `docker build`、`docker run`、容器检查等。
- **`openssh-client`** — 从容器内启用 [SSH 终端后端](/docs/user-guide/configuration#ssh-backend)。SSH 后端调用系统 `ssh` 二进制文件；没有这个，它在容器化安装中会静默失败。
- WhatsApp 桥接（`scripts/whatsapp-bridge/`）

入口点脚本（`docker/entrypoint.sh`）在首次运行时引导数据卷：
- 创建目录结构（`sessions/`、`memories/`、`skills/` 等）
- 如果不存在 `.env` 则复制 `.env.example` → `.env`
- 如果缺少则复制默认 `config.yaml`
- 如果缺少则复制默认 `SOUL.md`
- 使用基于清单的方法同步捆绑的 skills（保留用户编辑）
- 当 `HERMES_DASHBOARD=1` 时，可选地在后台启动 `hermes dashboard` 作为 side-process（请参阅 [运行仪表板](#running-the-dashboard)）
- 然后使用您传递的任何参数运行 `hermes`

:::warning
除非您在命令链中保留 `/opt/hermes/docker/entrypoint.sh`，否则不要覆盖镜像入口点。入口点将 root 权限降级为 `hermes` 用户，然后在创建 gateway 状态文件之前。默认情况下拒绝在官方镜像中以 root 身份启动 `hermes gateway run`，因为它可能会在 `/opt/data` 中留下 root 拥有的文件并破坏后续的仪表板或 gateway 启动。仅在您故意接受该风险时才设置 `HERMES_ALLOW_ROOT_GATEWAY=1`。
:::

## 升级

拉取最新镜像并重新创建容器。您的数据目录不会被触碰。

```sh
docker pull nousresearch/hermes-agent:latest
docker rm -f hermes
docker run -d \
  --name hermes \
  --restart unless-stopped \
  -v ~/.hermes:/opt/data \
  nousresearch/hermes-agent gateway run
```

或使用 Docker Compose：

```sh
docker compose pull
docker compose up -d
```

## Skills 和凭证文件

当使用 Docker 作为执行环境（不是上述方法，而是 agent 在 Docker sandbox 内运行命令时——请参阅 [配置 → Docker 后端](./configuration.md#docker-backend)），Hermes 为所有工具调用重用单个长期存在的容器，并自动将 skills 目录（`~/.hermes/skills/`）和 skills 声明的任何凭证文件作为只读卷绑定挂载到该容器中。Skill 脚本、模板和引用在 sandbox 中无需手动配置即可使用，并且因为容器在 Hermes 进程的整个生命周期内保持持久，您安装的任何依赖项或写入的文件都会保留供下次工具调用使用。

SSH 和 Modal 后端会发生相同的同步—— skills 和凭证文件在每个命令之前通过 rsync 或 Modal 挂载 API 上传。

## 连接到本地推理服务器（vLLM、Ollama 等）

当在 Docker 中运行 Hermes 而您的推理服务器（vLLM、Ollama、text-generation-inference 等）也在主机上运行或在另一个容器中时，网络需要额外注意。

### Docker Compose（推荐）

将两个服务放在同一个 Docker 网络上。这是最可靠的方法：

```yaml
services:
  vllm:
    image: vllm/vllm-openai:latest
    container_name: vllm
    command: >
      --model Qwen/Qwen2.5-7B-Instruct
      --served-model-name my-model
      --host 0.0.0.0
      --port 8000
    ports:
      - "8000:8000"
    networks:
      - hermes-net
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  hermes:
    image: nousresearch/hermes-agent:latest
    container_name: hermes
    restart: unless-stopped
    command: gateway run
    ports:
      - "8642:8642"
    volumes:
      - ~/.hermes:/opt/data
    networks:
      - hermes-net

networks:
  hermes-net:
    driver: bridge
```

然后在您的 `~/.hermes/config.yaml` 中，使用**容器名称**作为主机名：

```yaml
model:
  provider: custom
  model: my-model
  base_url: http://vllm:8000/v1
  api_key: "none"
```

:::tip 关键点
- 使用**容器名称**（`vllm`）作为主机名——不是 `localhost` 或 `127.0.0.1`，它们指的是 Hermes 容器本身。
- `model` 值必须与您传递给 vLLM 的 `--served-model-name` 匹配。
- 将 `api_key` 设置为任何非空字符串（vLLM 需要标头但默认不验证它）。
- 在 `base_url` 中**不要**包含尾部斜杠。
:::

### 独立 Docker 运行（无 Compose）

如果您的推理服务器直接在主机上运行（不在 Docker 中），在 macOS/Windows 上使用 `host.docker.internal`，或在 Linux 上使用 `--network host`：

**macOS / Windows：**

```sh
docker run -d \
  --name hermes \
  -v ~/.hermes:/opt/data \
  -p 8642:8642 \
  nousresearch/hermes-agent gateway run
```

```yaml
# config.yaml
model:
  provider: custom
  model: my-model
  base_url: http://host.docker.internal:8000/v1
  api_key: "none"
```

**Linux（主机网络）：**

```sh
docker run -d \
  --name hermes \
  --network host \
  -v ~/.hermes:/opt/data \
  nousresearch/hermes-agent gateway run
```

```yaml
# config.yaml
model:
  provider: custom
  model: my-model
  base_url: http://127.0.0.1:8000/v1
  api_key: "none"
```

:::warning 使用 `--network host` 时，`-p` 标志被忽略——所有容器端口直接暴露在主机上。
:::

### 验证连接

从 Hermes 容器内部，确认推理服务器可访问：

```sh
docker exec hermes curl -s http://vllm:8000/v1/models
```

您应该看到列出您提供模型的 JSON 响应。如果这失败了，请检查：

1. 两个容器在同一个 Docker 网络上（`docker network inspect hermes-net`）
2. 推理服务器正在监听 `0.0.0.0`，而不是 `127.0.0.1`
3. 端口号匹配

### Ollama

Ollama 以相同方式工作。如果 Ollama 在主机上运行，使用 `host.docker.internal:11434`（macOS/Windows）或 `127.0.0.1:11434`（Linux 使用 `--network host`）。如果 Ollama 在同一 Docker 网络上的自己的容器中运行：

```yaml
model:
  provider: custom
  model: llama3
  base_url: http://ollama:11434/v1
  api_key: "none"
```

## 故障排除

### 容器立即退出

检查日志：`docker logs hermes`。常见原因：
- 缺少或无效的 `.env` 文件——首先以交互方式运行以完成设置
- 如果使用暴露端口则端口冲突

### "权限被拒绝" 错误

容器的入口点通过 `gosu` 将权限降级为非 root `hermes` 用户（UID 10000）。如果您主机的 `~/.hermes/` 由不同的 UID 拥有，请设置 `HERMES_UID`/`HERMES_GID` 以匹配您的主机用户，或确保数据目录可写：

```sh
chmod -R 755 ~/.hermes
```

### 浏览器工具不工作

Playwright 需要共享内存。将 `--shm-size=1g` 添加到您的 Docker 运行命令：

```sh
docker run -d \
  --name hermes \
  --shm-size=1g \
  -v ~/.hermes:/opt/data \
  nousresearch/hermes-agent gateway run
```

### Gateway 在网络问题后不重新连接

`--restart unless-stopped` 标志处理大多数瞬时故障。如果 gateway 卡住了，重启容器：

```sh
docker restart hermes
```

### 检查容器健康状况

```sh
docker logs --tail 50 hermes          # 最近日志
docker run -it --rm nousresearch/hermes-agent:latest version     # 验证版本
docker stats hermes                    # 资源使用情况
```
