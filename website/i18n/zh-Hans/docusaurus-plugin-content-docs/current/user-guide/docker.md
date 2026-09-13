---
sidebar_position: 7
title: "Docker"
description: "在 Docker 中运行 Hermes Agent 以及将 Docker 用作终端后端"
---

# Hermes Agent — Docker

Docker 与 Hermes Agent 的交集有两种截然不同的方式：

1. **在 Docker 中运行 Hermes** — agent 本身在容器内运行（本页的主要内容）
2. **Docker 作为终端后端** — agent 在宿主机上运行，但将每条命令在单个持久化 Docker 沙箱容器中执行，该容器在工具调用、`/new` 和子 agent 之间保持存活，直至 Hermes 进程结束（参见 [配置 → Docker 后端](./configuration.md#docker-后端)）

本页介绍选项 1。容器将所有用户数据（配置、API 密钥、会话、技能、记忆）存储在从宿主机挂载于 `/opt/data` 的单个目录中。镜像本身是无状态的，可通过拉取新版本进行升级而不会丢失任何配置。

## 镜像标签和更新

| 标签 | 含义 |
|---|---|
| `stable` / `latest` | 通过完整稳定发布验收后提升的镜像 |
| `main` | 主分支开发构建 |
| `X.Y.Z` | 发布流程生成的版本化稳定镜像；精确部署使用摘要 |

镜像流程构建并测试 amd64 和 arm64。稳定发布复用已测试归档，不重新构建。
主分支推送只更新 `main`，不会提升 `stable` 或 `latest`。
应用代码位于 `/opt/hermes`，持久数据位于挂载的 `/opt/data`，两者分别管理。

## 快速开始

如果这是你第一次运行 Hermes Agent，请在宿主机上创建一个数据目录，并以交互方式启动容器以运行设置向导：

```sh
mkdir -p ~/.hermes
docker run -it --rm \
  -v ~/.hermes:/opt/data \
  nousresearch/hermes-agent setup
```

这将进入设置向导，向导会提示你输入 API 密钥并将其写入 `~/.hermes/.env`。你只需执行一次。强烈建议此时为 gateway 配置一个聊天系统。

## 以 gateway 模式运行

配置完成后，将容器作为持久化 gateway（Telegram、Discord、Slack、WhatsApp 等）在后台运行：

```sh
docker run -d \
  --name hermes \
  --restart unless-stopped \
  -v ~/.hermes:/opt/data \
  -p 8642:8642 \
  nousresearch/hermes-agent gateway run
```

端口 8642 暴露 gateway 的 [OpenAI 兼容 API 服务器](./features/api-server.md)和健康检查端点。如果你只使用聊天平台（Telegram、Discord 等），该端口是可选的；但如果你希望 dashboard 或外部工具访问 gateway，则必须开放。

注意：API 服务器需设置 `API_SERVER_ENABLED=true` 才会启用。若要在容器内将其暴露至 `127.0.0.1` 以外，还需设置 `API_SERVER_HOST=0.0.0.0` 和 `API_SERVER_KEY`（最少 8 个字符——可用 `openssl rand -hex 32` 生成）。示例：

```sh
docker run -d \
  --name hermes \
  --restart unless-stopped \
  -v ~/.hermes:/opt/data \
  -p 8642:8642 \
  -e API_SERVER_ENABLED=true \
  -e API_SERVER_HOST=0.0.0.0 \
  -e API_SERVER_KEY="$(openssl rand -hex 32)" \
  -e API_SERVER_CORS_ORIGINS='*' \
  nousresearch/hermes-agent gateway run
```

在面向互联网的机器上开放任何端口都存在安全风险。除非你了解相关风险，否则不应这样做。

## 运行 dashboard

内置 Web dashboard 在同一容器内作为受 s6-rc 监管的服务与 gateway 并行运行。设置 `HERMES_DASHBOARD=1` 即可拉起它：

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

Dashboard 由 s6 监管：若进程崩溃，`s6-supervise` 会在短暂退避后自动重启。Dashboard 的 stdout/stderr 会直接转发到 `docker logs <container>`；gateway 的主输出现在写入每个 profile 的 s6 日志文件，见下方的 per-profile 日志说明。

| 环境变量 | 描述 | 默认值 |
|---------------------|-------------|---------|
| `HERMES_DASHBOARD` | 设为 `1`（或 `true` / `yes`）以启用受监管的 dashboard 服务 | *（未设置——服务已注册但保持关闭）* |
| `HERMES_DASHBOARD_HOST` | dashboard HTTP 服务器的绑定地址 | `0.0.0.0` |
| `HERMES_DASHBOARD_PORT` | dashboard HTTP 服务器的端口 | `9119` |
| `HERMES_DASHBOARD_INSECURE` | **已弃用 / 空操作。** 以前用于绕过鉴权门控；自 2026 年 6 月的安全加固起，它不再禁用鉴权。任何非回环绑定都必须配置鉴权提供方 | *（被忽略——请改为配置提供方）* |

容器内的 dashboard 默认绑定 `0.0.0.0`，否则发布的 `-p 9119:9119` 端口将无法从宿主机访问。若你要把它限制在容器回环地址（例如 sidecar / 反向代理拓扑），请显式设置 `HERMES_DASHBOARD_HOST=127.0.0.1`。

当以下两项同时满足时，dashboard 的鉴权门控会自动启用：

1. 绑定地址为非回环地址，**且**
2. 注册了一个 `DashboardAuthProvider` 插件。

有三种内置方式可满足第二个条件：

- **用户名/密码** —— 最简单的自托管 / 局域网 / VPN 内部署方式：设置 `HERMES_DASHBOARD_BASIC_AUTH_USERNAME` + `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD`（以及用于跨重启稳定 session 的 `HERMES_DASHBOARD_BASIC_AUTH_SECRET`）。不适合直接暴露到公网上。
- **OAuth（Nous Portal）** —— 适合托管/公网部署：设置 `HERMES_DASHBOARD_OAUTH_CLIENT_ID` 后，`dashboard_auth/nous` 提供者会自动激活。
- **自托管 OIDC** —— 通过标准 OpenID Connect 接入你自己的身份提供商：设置 `HERMES_DASHBOARD_OIDC_ISSUER` + `HERMES_DASHBOARD_OIDC_CLIENT_ID` 后，`dashboard_auth/self_hosted` 提供者会激活。

无论选择哪种，调用方在访问受保护路由前都会先被重定向到登录页。完整说明见 [Web Dashboard → 鉴权](features/web-dashboard.md)。

如果未注册提供者且绑定为非回环地址，dashboard **会在启动时
失败关闭**，并给出指向缺失环境变量的具体错误信息。现在已不再
存在以无鉴权方式在公网绑定上提供 dashboard 的“逃生通道”：
`HERMES_DASHBOARD_INSECURE=1` 现在是一个已弃用的空操作（它会
打印告警并被忽略）。请改为配置鉴权提供方，或设置
`HERMES_DASHBOARD_HOST=127.0.0.1` 并通过 SSH 隧道 / Tailscale 访问。

:::warning 为什么移除了 `--insecure`
无鉴权的公网 dashboard 是 2026 年 6 月 MCP 配置持久化攻击活动的入口：互联网扫描器访问到暴露的 dashboard（以及 OpenAI API 服务器），诱导 agent 植入 SSH 密钥后门。现在每个非回环绑定都强制启用鉴权门控。对于可信局域网 / homelab 主机，内置的用户名/密码提供方（`HERMES_DASHBOARD_BASIC_AUTH_USERNAME` + `_PASSWORD`）是满足该要求的零基础设施方式。
:::

当独立的 dashboard 容器与宿主机共享 PID 与网络命名空间时（例如 `network_mode: host`，正如仓库自带的 `docker-compose.yml` 中的 `dashboard` 服务那样），**是**支持将 dashboard 作为独立容器运行的。其 gateway 存活检测需要与 gateway 进程共享 PID 命名空间，因此该限制仅适用于在隔离的 bridge 网络容器中、且未共享 PID 命名空间的 dashboard。

## 交互式运行（CLI 聊天）

对已有数据目录打开交互式聊天会话：

```sh
docker run -it --rm \
  -v ~/.hermes:/opt/data \
  nousresearch/hermes-agent
```

或者，如果你已通过 Docker Desktop 等方式在运行中的容器内打开了终端，直接运行：

```sh
/opt/hermes/.venv/bin/hermes
```

## 持久化卷

`/opt/data` 卷是所有 Hermes 状态的唯一数据来源。它映射到宿主机的 `~/.hermes/` 目录，包含：

| 路径 | 内容 |
|------|----------|
| `.env` | API 密钥和机密 |
| `config.yaml` | 所有 Hermes 配置 |
| `SOUL.md` | Agent 个性/身份 |
| `sessions/` | 对话历史 |
| `memories/` | 持久化记忆存储 |
| `skills/` | 已安装的技能 |
| `home/` | Hermes 工具子进程（`git`、`ssh`、`gh`、`npm` 及 skill CLI）的 per-profile HOME |
| `cron/` | 定时任务定义 |
| `hooks/` | 事件 hook |
| `logs/` | 运行时日志 |
| `skins/` | 自定义 CLI 皮肤 |

### 不可变安装树

在托管/发布的 Docker 镜像中，`/opt/hermes` 是安装好的应用树。它由 root 拥有，并且对运行时的 `hermes` 用户只读，因此 agent 回合、gateway 会话、dashboard 操作以及普通的 `docker exec hermes hermes ...` 命令都不能原地修改核心源码、打包的 `.venv`、`node_modules` 或 TUI bundle。

所有可变的 Hermes 状态都应位于 `/opt/data` 下：配置、`.env`、profiles、skills、memories、sessions、logs、dashboard 上传、plugins 以及其他用户管理的文件。官方镜像还会阻止在运行时向不可变的 `/opt/hermes` 树写入 `.pyc` 或执行 Hermes 的懒安装依赖流程。

如果运维人员确实需要修复或检查 `/opt/data` 之外的文件，请有意识地使用 root shell。`hermes` shim 默认会把 `docker exec hermes hermes ...` 降回运行时用户；只有在你明确需要 root 语义时，才临时设置 `HERMES_DOCKER_EXEC_AS_ROOT=1`。

某些 skill CLI 会把凭据写到 `~` 下，因此在官方 Docker 布局里要针对子进程 HOME 初始化，而不是只针对数据卷根目录。例如 [xurl skill](./skills/bundled/social-media/social-media-xurl.md) 会把 OAuth 状态存到 `~/.xurl`；在容器里这对应 `/opt/data/home/.xurl`，因此手动认证时应使用 `HOME=/opt/data/home xurl auth status` 之类的调用。

:::warning
切勿同时对同一数据目录运行两个 Hermes **gateway** 容器——会话文件和记忆存储不支持并发写入。
:::

## 多 profile 支持

Hermes 支持[多个 profile](../reference/profile-commands.md)——独立的 `~/.hermes/` 子目录，让你可以从单个安装运行独立的 agent（不同的 SOUL、skills、memory、sessions、credentials）。**在官方 Docker 镜像内，s6 监管树把每个 profile 当作一等受监管服务**，因此推荐部署方式是：**一个容器承载多个 profile**。

每个通过 `hermes profile create <name>` 创建的 profile 都会获得：

- 一个专用的 s6 服务槽位 `/run/service/gateway-<name>/`，运行时动态注册，无需重建镜像。
- 崩溃后的自动重启，由 `s6-supervise` 管理退避。
- 每个 profile 独立的轮转日志：`${HERMES_HOME}/logs/gateways/<name>/current`。
- 跨容器重启的状态持久化：启动协调器会读取该 profile 的 `gateway_state.json`，仅在上次记录状态为 `running` 时自动拉起。

容器内生命周期命令与宿主机上一致：

```sh
# 创建 profile —— 同时注册 gateway-<name> s6 槽位
docker exec hermes hermes profile create coder

# 启停/重启 —— 底层分发给 s6-svc
docker exec hermes hermes -p coder gateway start
docker exec hermes hermes -p coder gateway stop
docker exec hermes hermes -p coder gateway restart

# 状态 —— 容器内会显示 `Manager: s6 (container supervisor)`
docker exec hermes hermes -p coder gateway status
```

若第二个 profile 也要暴露 OpenAI 兼容 API server，请在**该 profile 自己的** `.env` 中设置不同的 `API_SERVER_PORT`，然后重启该 profile 的 gateway；不要把端口放进容器级 `environment:`，否则所有 profile 都会争抢同一个端口。更底层的监管细节见后文的 [Per-profile gateway 监管](#per-profile-gateway-supervision)。

## 环境变量转发

API 密钥从容器内的 `/opt/data/.env` 读取。你也可以直接传递环境变量：

```sh
docker run -it --rm \
  -v ~/.hermes:/opt/data \
  -e ANTHROPIC_API_KEY="sk-ant-..." \
  -e OPENAI_API_KEY="sk-..." \
  nousresearch/hermes-agent
```

直接传入的 `-e` 标志会覆盖 `.env` 中的值。这对于不希望将密钥写入磁盘的 CI/CD 或密钥管理器集成非常有用。

:::note 寻找 Docker 作为**终端后端**的说明？
本页介绍在 Docker 内运行 Hermes 本身。如果你希望 Hermes 在 Docker 沙箱容器内执行 agent 的 `terminal` / `execute_code` 调用（每个 Hermes 进程对应一个持久容器），那是另一个配置块——`terminal.backend: docker` 加上 `terminal.docker_image`、`terminal.docker_volumes`、`terminal.docker_forward_env`、`terminal.docker_run_as_host_user` 和 `terminal.docker_extra_args`。完整配置请参见 [配置 → Docker 后端](configuration.md#docker-后端)。
:::

## Docker Compose 示例

对于同时运行 gateway 和 dashboard 的持久化部署，使用 `docker-compose.yaml` 更为方便：

```yaml
services:
  hermes:
    image: nousresearch/hermes-agent:latest
    container_name: hermes
    restart: unless-stopped
    command: gateway run
    ports:
      - "8642:8642"   # gateway API
      - "9119:9119"   # dashboard（仅在 HERMES_DASHBOARD=1 时生效）
    volumes:
      - ~/.hermes:/opt/data
    environment:
      - HERMES_DASHBOARD=1
      # 取消注释以直接转发特定环境变量而非使用 .env 文件：
      # - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      # - OPENAI_API_KEY=${OPENAI_API_KEY}
      # - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: "2.0"
```

使用 `docker compose up -d` 启动，使用 `docker compose logs -f` 查看日志。Dashboard 的 stdout/stderr 会直接出现在这里；gateway 主日志则写入每个 profile 的 s6 日志文件，见下方的 [Per-profile gateway 监管](#per-profile-gateway-supervision)。

## 资源限制

Hermes 容器需要适量资源。推荐最低配置：

| 资源 | 最低 | 推荐 |
|----------|---------|-------------|
| 内存 | 1 GB | 2–4 GB |
| CPU | 1 核 | 2 核 |
| 磁盘（数据卷） | 500 MB | 2+ GB（随会话/技能增长） |

浏览器自动化（Playwright/Chromium）是最耗内存的功能。如果不需要浏览器工具，1 GB 即可。启用浏览器工具时，请至少分配 2 GB。

在 Docker 中设置限制：

```sh
docker run -d \
  --name hermes \
  --restart unless-stopped \
  --memory=4g --cpus=2 \
  -v ~/.hermes:/opt/data \
  nousresearch/hermes-agent gateway run
```

## Dockerfile 说明 {#what-the-dockerfile-does}

官方镜像基于 Debian 13.4，包含：

- 按提交的 `uv.lock` 同步的 Python 3.14 环境，然后无依赖地安装 Hermes 源码。
- 选定的 extras：`all`、`messaging`、`otlp`、`anthropic`、`bedrock`、`azure-identity`、`hindsight` 和 `matrix`，不是 `--all-extras`。
- 从摘要固定的 Node 镜像提供的 Node.js 26 和 npm。
- PM 固定版本的 uv、Chromium 和 headless shell，位于 `/opt/hermes/tools`。
- 系统 Git、ripgrep、FFmpeg、OpenSSH、Docker CLI 和 Chromium 所需共享库。
- 预构建的 TUI/dashboard 和 Photon sidecar 依赖，以及 s6-overlay。

Chromium 由 PM 准备，不使用 `npx playwright install`。
实际可执行路径记录在 `/etc/hermes/agent-browser-executable-path`。
`PLAYWRIGHT_BROWSERS_PATH` 指向 `/opt/hermes/tools`，不在数据卷内。

镜像通过内部 `HERMES_DISABLE_LAZY_INSTALLS` 策略关闭按需安装。
仅修改 `security.allow_lazy_installs` 不能覆盖它。新增所需依赖应烘焙进派生镜像，
或作为独立工具放在单独环境或服务中。旧 `lazy-packages` overlay 不再使用。

构建来源记录在 `/etc/hermes/image-provenance.json`，构建戳记位于 `/opt/hermes/install-stamp.json`。
没有戳记的本地构建报告未知版本，不猜测提交。`hermes update` 不修改镜像所有的代码，应用更新需替换镜像。

入口点是 `docker/entrypoint-dispatch.sh`。正常 Docker/Podman 中它占有 PID 1，转交给 s6 的 `/init`。
若平台已有 PID-1 init，则直接运行 stage2 和主包装器；命令仍可运行，但没有 s6 监管的 dashboard 或各 profile gateway。

PID-1 路径先准备数据卷和配置，重建各 profile 的 gateway 服务槽，然后运行主命令。
不要绕过这些入口步骤，否则会失去权限处理和服务监管。
主程序及受监管服务以 `hermes` 用户运行，避免在 `/opt/data` 中留下 root 所有的文件。

### Per-profile gateway 监管 {#per-profile-gateway-supervision}

在容器内，每个通过 `hermes profile create <name>` 创建的 profile 都会自动在 `/run/service/gateway-<name>/` 注册一个受 s6 监管的 gateway 服务。你在宿主机上运行的生命周期命令在此同样适用：

```sh
hermes profile create coder            # 注册 gateway-coder s6 槽
hermes -p coder gateway start          # s6-svc -u  → 受监管的 gateway
hermes -p coder gateway stop           # s6-svc -d  → 服务停止
hermes -p coder gateway restart        # s6-svc -t  → 向 supervisor 发送 SIGTERM
hermes profile delete coder            # 拆除 s6 槽
```

**相比 pre-s6 镜像的监管优势：**

- Gateway 崩溃后由 `s6-supervise` 在约 1 秒退避后自动重启。
- Dashboard 崩溃后自动重启（设置 `HERMES_DASHBOARD=1` 以启动）。
- `docker restart` 保留运行中的 gateway：cont-init 协调器读取 `$HERMES_HOME/profiles/<name>/gateway_state.json`，若上次记录状态为 `running` 则恢复该槽。已停止的 gateway 保持停止状态。
- 各 profile 的 gateway 日志持久化于 `$HERMES_HOME/logs/gateways/<profile>/current`（由 `s6-log` 轮转），协调器的操作记录在每次启动时追加到 `$HERMES_HOME/logs/container-boot.log`。

在容器内执行 `hermes status` 会显示 `Manager: s6 (container supervisor)`。使用 `/command/s6-svstat /run/service/gateway-<name>` 查看原始 supervisor 状态（注意 `/command/` 仅在监管树进程的 PATH 中；从 `docker exec` 调用时请传入绝对路径）。

## 升级

拉取最新镜像并重建容器。你的数据目录不受影响。

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

## 技能与凭据文件

当使用 Docker 作为执行环境时（不是上述方法，而是 agent 在 Docker 沙箱内运行命令——参见 [配置 → Docker 后端](./configuration.md#docker-后端)），Hermes 为所有工具调用复用单个长期运行的容器，并自动将技能目录（`~/.hermes/skills/`）和技能声明的所有凭据文件以只读卷的形式绑定挂载到该容器中。技能脚本、模板和引用在沙箱内无需手动配置即可使用，由于容器在 Hermes 进程的整个生命周期内持续存在，你安装的任何依赖或写入的文件都会在下次工具调用时保留。

SSH 和 Modal 后端也会进行相同的同步——技能和凭据文件在每次命令执行前通过 rsync 或 Modal mount API 上传。

## 在容器中安装更多工具

官方镜像预装了一套精选工具（参见 [Dockerfile 说明](#what-the-dockerfile-does)），但并非 agent 可能需要的每个工具都已预装。以下是五种推荐方式，按工作量和持久性递增排列。

### npm 或 Python 工具——使用 `npx` 或 `uvx`

对于发布到 npm 或 PyPI 的任何工具，指示 Hermes 通过 `npx`（npm）或 `uvx`（Python）运行，并将该命令记入其持久记忆。如果工具需要配置文件或凭据，指示其将这些文件放在 `/opt/data` 下（如 `/opt/data/<tool>/config.yaml`）。

依赖按需获取并在容器生命周期内缓存。写入 `/opt/data` 的配置在容器重启后仍然存在，因为它位于绑定挂载的宿主机目录上。包缓存本身在 `docker rm` 后会重建，但 `npx` 和 `uvx` 会在下次运行工具时透明地重新获取。

### 其他工具（apt 包、二进制文件）——安装并记住

对于 npm 或 PyPI 之外的工具——`apt` 包、预构建二进制文件、镜像中未包含的语言运行时——指示 Hermes 如何安装（如 `apt-get update && apt-get install -y <package>`），并告知它记住该安装命令。工具在容器剩余生命周期内持续可用，Hermes 在容器重启后下次需要该工具时会重新运行安装命令。

这种方式适合安装快速且偶尔使用的工具。对于频繁使用的工具，建议采用下一种方式。

### 持久安装——构建派生镜像

当工具必须在每次容器启动时立即可用且无需重新安装延迟时，构建一个继承自 `nousresearch/hermes-agent` 并在层中安装该工具的新镜像：

```dockerfile
FROM nousresearch/hermes-agent:latest

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends <your-package> \
    && rm -rf /var/lib/apt/lists/*
USER hermes
```

构建并替换官方镜像使用：

```sh
docker build -t my-hermes:latest .
docker run -d \
  --name hermes \
  --restart unless-stopped \
  -v ~/.hermes:/opt/data \
  -p 8642:8642 \
  my-hermes:latest gateway run
```

入口点脚本和 `/opt/data` 语义原样继承，本页其余内容仍然适用。拉取更新的上游 `nousresearch/hermes-agent` 时记得重新构建镜像。

### 复杂工具或多服务栈——运行 sidecar 容器

对于自带服务（数据库、Web 服务器、队列、无头浏览器集群）或过于庞大而不适合放在 Hermes 容器内的工具，将其作为独立容器运行在共享 Docker 网络上。Hermes 通过容器名称访问 sidecar，与访问本地推理服务器的方式相同（参见 [连接本地推理服务器](#连接本地推理服务器vllmollama-等)）。

```yaml
services:
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

  my-tool:
    image: example/my-tool:latest
    container_name: my-tool
    restart: unless-stopped
    networks:
      - hermes-net

networks:
  hermes-net:
    driver: bridge
```

在 Hermes 容器内，sidecar 可通过 `http://my-tool:<port>` 访问（或其提供的任何协议）。这种模式使每个服务的生命周期、资源限制和升级节奏保持独立，避免因单个工具的依赖而使 Hermes 镜像臃肿。

### 广泛有用的工具——提交 issue 或 pull request

如果某个工具可能对大多数 Hermes Agent 用户有用，考虑将其贡献到上游，而不是在私有派生镜像中维护。在 [hermes-agent 仓库](https://github.com/NousResearch/hermes-agent)提交 issue 或 pull request，描述该工具及其使用场景。被纳入官方镜像的工具惠及所有用户，并避免了维护下游 fork 的开销。

## 连接本地推理服务器（vLLM、Ollama 等）

在 Docker 中运行 Hermes 且推理服务器（vLLM、Ollama、text-generation-inference 等）也在宿主机或另一个容器中运行时，网络配置需要额外注意。

### Docker Compose（推荐）

将两个服务放在同一 Docker 网络上。这是最可靠的方式：

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

然后在 `~/.hermes/config.yaml` 中，使用**容器名称**作为主机名：

```yaml
model:
  provider: custom
  model: my-model
  base_url: http://vllm:8000/v1
  api_key: "none"
```

:::tip 关键点
- 使用**容器名称**（`vllm`）作为主机名——而非 `localhost` 或 `127.0.0.1`，它们指向 Hermes 容器本身。
- `model` 值必须与传给 vLLM 的 `--served-model-name` 一致。
- 将 `api_key` 设为任意非空字符串（vLLM 要求该请求头，但默认不验证其值）。
- `base_url` 末尾**不要**加斜杠。
:::

### 独立 Docker run（无 Compose）

如果推理服务器直接在宿主机上运行（不在 Docker 中），在 macOS/Windows 上使用 `host.docker.internal`，在 Linux 上使用 `--network host`：

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

**Linux（host 网络）：**

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

:::warning 使用 `--network host` 时，`-p` 标志会被忽略——所有容器端口直接暴露在宿主机上。
:::

### 验证连通性

从 Hermes 容器内部确认推理服务器可达：

```sh
docker exec hermes curl -s http://vllm:8000/v1/models
```

你应该看到列出已服务模型的 JSON 响应。如果失败，请检查：

1. 两个容器是否在同一 Docker 网络上（`docker network inspect hermes-net`）
2. 推理服务器是否监听 `0.0.0.0` 而非 `127.0.0.1`
3. 端口号是否匹配

### Ollama

Ollama 的配置方式相同。如果 Ollama 在宿主机上运行，使用 `host.docker.internal:11434`（macOS/Windows）或 `127.0.0.1:11434`（Linux 使用 `--network host`）。如果 Ollama 在同一 Docker 网络的独立容器中运行：

```yaml
model:
  provider: custom
  model: llama3
  base_url: http://ollama:11434/v1
  api_key: "none"
```

## 故障排查

### 容器立即退出

检查日志：`docker logs hermes`。常见原因：
- `.env` 文件缺失或无效——先以交互方式运行以完成设置
- 开放端口时存在端口冲突

### "Permission denied" 错误

容器的 stage2 hook 通过 `s6-setuidgid` 在每个受监管的服务内将权限降至非 root 用户 `hermes`（UID 10000）。如果宿主机的 `~/.hermes/` 由不同 UID 拥有，请设置 `HERMES_UID`/`HERMES_GID` 以匹配宿主机用户，或确保数据目录可写：

```sh
chmod -R 755 ~/.hermes
```

### 浏览器工具无法使用

Playwright 需要共享内存。在 Docker run 命令中添加 `--shm-size=1g`：

```sh
docker run -d \
  --name hermes \
  --shm-size=1g \
  -v ~/.hermes:/opt/data \
  nousresearch/hermes-agent gateway run
```

### 网络问题后 gateway 无法重连

`--restart unless-stopped` 标志可处理大多数瞬时故障。如果 gateway 卡住，重启容器：

```sh
docker restart hermes
```

### 检查容器健康状态

```sh
docker logs --tail 50 hermes          # 最近日志
docker run -it --rm nousresearch/hermes-agent:latest version     # 验证版本
docker stats hermes                    # 资源使用情况
```