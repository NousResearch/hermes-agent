---
sidebar_position: 7
title: "Docker"
description: "在 Docker 中运行 Hermes Agent，以及把 Docker 用作终端后端"
---

# Hermes Agent - Docker

Docker 与 Hermes Agent 的交互方式有两种：

1. **在 Docker 中运行 Hermes** - agent 自身运行在容器里（本页主要讲这一种）
2. **把 Docker 当作终端后端** - agent 仍运行在宿主机上，但每条命令都会在一个单独、持久的 Docker 沙盒容器里执行，这个容器会在 Hermes 进程生命周期内跨工具调用、`/new` 和子 agent 持续存在（参见 [配置 → Docker 后端](./configuration.md#docker-backend)）

本页讲的是第 1 种方式。容器会把所有用户数据（配置、API Key、会话、技能、记忆）存放在宿主机挂载的单一目录 `/opt/data` 下。镜像本身是无状态的，只要拉取新版本就可以升级，而不会丢失任何配置。

## 快速开始

如果你是第一次运行 Hermes Agent，先在宿主机上创建一个数据目录，然后以交互方式启动容器运行安装向导：

```sh
mkdir -p ~/.hermes
docker run -it --rm \
  -v ~/.hermes:/opt/data \
  nousresearch/hermes-agent setup
```

这会带你进入安装向导，提示你输入 API Key，并把它们写入 `~/.hermes/.env`。这一步只需要做一次。强烈建议此时就先配置一个聊天系统，让网关可以正常工作。

## 以网关模式运行

配置完成后，可以把容器作为持久化网关（Telegram、Discord、Slack、WhatsApp 等）后台运行：

```sh
docker run -d \
  --name hermes \
  --restart unless-stopped \
  -v ~/.hermes:/opt/data \
  -p 8642:8642 \
  nousresearch/hermes-agent gateway run
```

8642 端口暴露的是网关的 [OpenAI-compatible API server](./features/api-server.md) 和健康检查端点。如果你只使用聊天平台（Telegram、Discord 等），这个端口是可选的；但如果你想让 dashboard 或外部工具连接网关，就需要它。

注意：API server 需要 `API_SERVER_ENABLED=true` 才会开放。如果你想把它从容器内的 `127.0.0.1` 暴露出来，还要设置 `API_SERVER_HOST=0.0.0.0` 和 `API_SERVER_KEY`（至少 8 个字符 - 可以用 `openssl rand -hex 32` 生成）。示例：

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

在一台面向互联网的机器上打开任何端口都有安全风险。除非你理解这些风险，否则不要这么做。

## 运行 dashboard

内置 Web dashboard 可以作为同一容器中的可选子进程，与网关一起运行。设置 `HERMES_DASHBOARD=1`，并同时暴露网关的 `8642` 和 dashboard 的 `9119`：

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

entrypoint 会先以非 root 的 `hermes` 用户在后台启动 `hermes dashboard`，然后再 `exec` 主命令。dashboard 的输出会在 `docker logs` 中以 `[dashboard]` 为前缀，方便你和网关日志区分。

| 环境变量 | 说明 | 默认值 |
|---------------------|---------|---------|
| `HERMES_DASHBOARD` | 设置为 `1`（或 `true` / `yes`）时，在主命令旁一起启动 dashboard | *(未设置 - 不启动 dashboard)* |
| `HERMES_DASHBOARD_HOST` | dashboard HTTP 服务器的绑定地址 | `0.0.0.0` |
| `HERMES_DASHBOARD_PORT` | dashboard HTTP 服务器端口 | `9119` |
| `HERMES_DASHBOARD_TUI` | 设置为 `1` 可暴露浏览器中的 Chat 标签页（通过 PTY/WebSocket 嵌入真实的 `hermes --tui`） | *(未设置)* |

默认的 `HERMES_DASHBOARD_HOST=0.0.0.0` 是让宿主机可以通过发布端口访问 dashboard 的必要条件；当使用这种配置时，entrypoint 会自动给 `hermes dashboard` 传入 `--insecure`。如果你想只允许容器内部访问，可以把它改成 `127.0.0.1`（例如放在 sidecar 反向代理后面）。

:::note
dashboard 子进程**没有守护机制** - 如果它崩溃了，它会一直保持关闭，直到容器重启。把它拆成单独容器运行是不支持的：dashboard 的网关存活检测需要和网关进程共享 PID namespace。
:::

## 交互式运行（CLI chat）

要对某个正在运行的数据目录打开交互式聊天会话：

```sh
docker run -it --rm \
  -v ~/.hermes:/opt/data \
  nousresearch/hermes-agent
```

如果你已经在运行中的容器里打开了终端（例如通过 Docker Desktop），可以直接运行：

```sh
/opt/hermes/.venv/bin/hermes
```

## 持久化卷

`/opt/data` 卷是 Hermes 所有状态的唯一来源。它映射到宿主机的 `~/.hermes/` 目录，并包含：

| 路径 | 内容 |
|------|----------|
| `.env` | API Key 和密钥 |
| `config.yaml` | Hermes 所有配置 |
| `SOUL.md` | agent 人格 / 身份 |
| `sessions/` | 对话历史 |
| `memories/` | 持久化记忆存储 |
| `skills/` | 已安装技能 |
| `cron/` | 定时任务定义 |
| `hooks/` | 事件钩子 |
| `logs/` | 运行日志 |
| `skins/` | 自定义 CLI 皮肤 |

:::warning
不要让两个 Hermes **网关**容器同时指向同一个数据目录 - 会话文件和记忆存储并不是为并发写入设计的。
:::

## 多配置文件支持

Hermes 支持[多个配置文件](/reference/profile-commands) - 也就是多个独立的 `~/.hermes/` 目录，让你能在单个安装里运行多个互不影响的 agent（不同的 SOUL、技能、记忆、会话、凭据）。**在 Docker 下，不推荐使用 Hermes 的内建多配置文件功能。**

更推荐的模式是**一份配置文件对应一个容器**，每个容器挂载自己独立的宿主机目录到 `/opt/data`：

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

为什么在 Docker 中更推荐分容器而不是分配置文件：

- **隔离性** - 每个容器都有自己的文件系统、进程表和资源限制。一个配置文件里的崩溃、依赖变更或失控会话不会影响另一个。
- **生命周期独立** - 每个 agent 都能单独升级、重启、暂停或回滚（`docker restart hermes-work` 不会影响 `hermes-personal`）。
- **端口和网络更干净** - 每个网关都绑定自己的宿主机端口；聊天平台或 API server 不会互相串台。
- **思维模型更简单** - 容器就是配置文件。备份、迁移和权限都跟挂载目录走，不需要额外记 `--profile`。
- **避免并发写入风险** - 上面提到的“不要让两个网关共享同一个数据目录”这条，在单容器里的多个配置文件场景下也一样成立。

在 Docker Compose 里，这只是意味着给每个配置文件单独建一个 service，分别指定不同的 `container_name`、`volumes` 和 `ports`：

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

API Key 会从容器内的 `/opt/data/.env` 读取。你也可以直接传环境变量：

```sh
docker run -it --rm \
  -v ~/.hermes:/opt/data \
  -e ANTHROPIC_API_KEY="sk-ant-..." \
  -e OPENAI_API_KEY="sk-..." \
  nousresearch/hermes-agent
```

直接使用 `-e` 传入的值会覆盖 `.env` 中的值。这对于 CI/CD 或 secrets-manager 集成很有用，尤其是不想把 key 放到磁盘上的时候。

## Docker Compose 示例

如果你想持久化部署网关和 dashboard，使用 `docker-compose.yaml` 会很方便：

```yaml
services:
  hermes:
    image: nousresearch/hermes-agent:latest
    container_name: hermes
    restart: unless-stopped
    command: gateway run
    ports:
      - "8642:8642"   # 网关 API
      - "9119:9119"   # dashboard（仅在 HERMES_DASHBOARD=1 时可访问）
    volumes:
      - ~/.hermes:/opt/data
    environment:
      - HERMES_DASHBOARD=1
      # 如需转发指定环境变量而不是使用 .env，可取消注释：
      # - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      # - OPENAI_API_KEY=${OPENAI_API_KEY}
      # - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: "2.0"
```

启动时运行 `docker compose up -d`，查看日志用 `docker compose logs -f`。dashboard 输出会以 `[dashboard]` 为前缀，方便你从网关日志中筛选出来。

## 资源限制

Hermes 容器需要中等资源。建议最低配置如下：

| 资源 | 最低 | 推荐 |
|----------|---------|-------------|
| 内存 | 1 GB | 2–4 GB |
| CPU | 1 核 | 2 核 |
| 磁盘（数据卷） | 500 MB | 2+ GB（会随会话 / 技能增长） |

浏览器自动化（Playwright / Chromium）是最吃内存的功能。如果你不需要浏览器工具，1 GB 就够了。启用浏览器工具时，至少分配 2 GB。