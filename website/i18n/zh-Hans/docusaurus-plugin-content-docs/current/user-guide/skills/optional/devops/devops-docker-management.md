---
title: "Docker 管理"
sidebar_label: "Docker 管理"
description: "管理 Docker 容器、镜像、卷、网络和 Compose 栈——生命周期操作、调试、清理及 Dockerfile 优化"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Docker 管理

管理 Docker 容器、镜像、卷、网络和 Compose 栈——生命周期操作、调试、清理及 Dockerfile 优化。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选——使用 `hermes skills install official/devops/docker-management` 安装 |
| 路径 | `optional-skills/devops/docker-management` |
| 版本 | `1.0.0` |
| 作者 | sprmn24 |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `docker`, `containers`, `devops`, `infrastructure`, `compose`, `images`, `volumes`, `networks`, `debugging` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

# Docker 管理

使用标准 Docker CLI 命令管理 Docker 容器、镜像、卷、网络和 Compose 栈。除 Docker 本身外无需额外依赖。

## 适用场景

- 运行、停止、重启、删除或检查容器
- 构建、拉取、推送、标记或清理 Docker 镜像
- 使用 Docker Compose（多服务栈）
- 管理卷或网络
- 调试崩溃的容器或分析日志
- 检查 Docker 磁盘使用情况或释放空间
- 审查或优化 Dockerfile

## 前提条件

- Docker Engine 已安装并运行
- 用户已加入 `docker` 组（或使用 `sudo`）
- Docker Compose v2（现代 Docker 安装已包含）

快速检查：

```bash
docker --version && docker compose version
```

## 快速参考

| 任务 | 命令 |
|------|---------|
| 运行容器（后台） | `docker run -d --name NAME IMAGE` |
| 停止并删除 | `docker stop NAME && docker rm NAME` |
| 查看日志（跟踪） | `docker logs --tail 50 -f NAME` |
| 进入容器 Shell | `docker exec -it NAME /bin/sh` |
| 列出所有容器 | `docker ps -a` |
| 构建镜像 | `docker build -t TAG .` |
| Compose 启动 | `docker compose up -d` |
| Compose 停止 | `docker compose down` |
| 磁盘使用情况 | `docker system df` |
| 清理悬空资源 | `docker image prune && docker container prune` |

## 操作流程

### 1. 确定操作域

判断请求属于哪个领域：

- **容器生命周期** → run、stop、start、restart、rm、pause/unpause
- **容器交互** → exec、cp、logs、inspect、stats
- **镜像管理** → build、pull、push、tag、rmi、save/load
- **Docker Compose** → up、down、ps、logs、exec、build、config
- **卷与网络** → create、inspect、rm、prune、connect
- **故障排查** → 日志分析、退出码、资源问题

### 2. 容器操作

**运行新容器：**

```bash
# 后台服务，带端口映射
docker run -d --name web -p 8080:80 nginx

# 带环境变量
docker run -d -e POSTGRES_PASSWORD=secret -e POSTGRES_DB=mydb --name db postgres:16

# 带持久化数据（命名卷）
docker run -d -v pgdata:/var/lib/postgresql/data --name db postgres:16

# 开发环境（绑定挂载源码）
docker run -d -v $(pwd)/src:/app/src -p 3000:3000 --name dev my-app

# 交互式调试（退出后自动删除）
docker run -it --rm ubuntu:22.04 /bin/bash

# 带资源限制和重启策略
docker run -d --memory=512m --cpus=1.5 --restart=unless-stopped --name app my-app
```

关键参数：`-d` 后台运行，`-it` 交互式+tty，`--rm` 自动删除，`-p` 端口（宿主机:容器），`-e` 环境变量，`-v` 卷，`--name` 名称，`--restart` 重启策略。

**管理运行中的容器：**

```bash
docker ps                        # 运行中的容器
docker ps -a                     # 所有容器（包括已停止的）
docker stop NAME                 # 优雅停止
docker start NAME                # 启动已停止的容器
docker restart NAME              # 停止并重启
docker rm NAME                   # 删除已停止的容器
docker rm -f NAME                # 强制删除运行中的容器
docker container prune           # 删除所有已停止的容器
```

**与容器交互：**

```bash
docker exec -it NAME /bin/sh          # Shell 访问（如可用则使用 /bin/bash）
docker exec NAME env                   # 查看环境变量
docker exec -u root NAME apt update    # 以指定用户运行
docker logs --tail 100 -f NAME         # 跟踪最后 100 行日志
docker logs --since 2h NAME            # 最近 2 小时的日志
docker cp NAME:/path/file ./local      # 从容器复制文件
docker cp ./file NAME:/path/           # 向容器复制文件
docker inspect NAME                    # 完整容器详情（JSON）
docker stats --no-stream               # 资源使用快照
docker top NAME                        # 运行中的进程
```

### 3. 镜像管理

```bash
# 构建
docker build -t my-app:latest .
docker build -t my-app:prod -f Dockerfile.prod .
docker build --no-cache -t my-app .              # 全量重新构建
DOCKER_BUILDKIT=1 docker build -t my-app .       # 使用 BuildKit 加速

# 拉取与推送
docker pull node:20-alpine
docker login ghcr.io
docker tag my-app:latest registry/my-app:v1.0
docker push registry/my-app:v1.0

# 检查
docker images                          # 列出本地镜像
docker history IMAGE                   # 查看层信息
docker inspect IMAGE                   # 完整详情

# 清理
docker image prune                     # 删除悬空（未标记）镜像
docker image prune -a                  # 删除所有未使用镜像（谨慎！）
docker image prune -a --filter "until=168h"   # 删除 7 天前未使用的镜像
```

### 4. Docker Compose

```bash
# 启动/停止
docker compose up -d                   # 后台启动所有服务
docker compose up -d --build           # 启动前重新构建镜像
docker compose down                    # 停止并删除容器
docker compose down -v                 # 同时删除卷（会销毁数据）

# 监控
docker compose ps                      # 列出服务
docker compose logs -f api             # 跟踪指定服务的日志
docker compose logs --tail 50          # 所有服务最后 50 行日志

# 交互
docker compose exec api /bin/sh        # 进入运行中服务的 Shell
docker compose run --rm api npm test   # 一次性命令（新容器）
docker compose restart api             # 重启指定服务

# 验证
docker compose config                  # 验证并查看解析后的配置
```

**最简 compose.yml 示例：**

```yaml
services:
  api:
    build: .
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgres://user:pass@db:5432/mydb
    depends_on:
      db:
        condition: service_healthy

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: mydb
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  pgdata:
```

### 5. 卷与网络

```bash
# 卷
docker volume ls                       # 列出卷
docker volume create mydata            # 创建命名卷
docker volume inspect mydata           # 详情（挂载点等）
docker volume rm mydata                # 删除（使用中则失败）
docker volume prune                    # 删除未使用的卷

# 网络
docker network ls                      # 列出网络
docker network create mynet            # 创建桥接网络
docker network inspect mynet           # 详情（已连接的容器）
docker network connect mynet NAME      # 将容器连接到网络
docker network disconnect mynet NAME   # 断开容器连接
docker network rm mynet                # 删除网络
docker network prune                   # 删除未使用的网络
```

### 6. 磁盘使用与清理

清理前始终先进行诊断：

```bash
# 检查空间占用
docker system df                       # 摘要
docker system df -v                    # 详细分解

# 针对性清理（安全）
docker container prune                 # 已停止的容器
docker image prune                     # 悬空镜像
docker volume prune                    # 未使用的卷
docker network prune                   # 未使用的网络

# 激进清理（请先与用户确认！）
docker system prune                    # 容器 + 镜像 + 网络
docker system prune -a                 # 同时包含未使用镜像
docker system prune -a --volumes       # 全部清除——包括命名卷
```

**警告：** 未经用户确认，切勿运行 `docker system prune -a --volumes`。此命令会删除可能包含重要数据的命名卷。

## 常见问题

| 问题 | 原因 | 解决方法 |
|---------|-------|-----|
| 容器立即退出 | 主进程结束或崩溃 | 检查 `docker logs NAME`，尝试 `docker run -it --entrypoint /bin/sh IMAGE` |
| "port is already allocated" | 该端口已被其他进程占用 | 使用 `docker ps` 或 `lsof -i :PORT` 查找 |
| "no space left on device" | Docker 磁盘已满 | 执行 `docker system df` 后针对性清理 |
| 无法连接到容器 | 容器内应用绑定到 127.0.0.1 | 应用须绑定到 `0.0.0.0`，检查 `-p` 映射 |
| 卷权限被拒绝 | 宿主机与容器 UID/GID 不匹配 | 使用 `--user $(id -u):$(id -g)` 或修复权限 |
| Compose 服务间无法互通 | 网络错误或服务名称错误 | 服务使用服务名作为主机名，检查 `docker compose config` |
| 构建缓存失效 | Dockerfile 层顺序错误 | 将不常变动的层放在前面（依赖在源码之前） |
| 镜像过大 | 未使用多阶段构建，缺少 .dockerignore | 使用多阶段构建，添加 `.dockerignore` |

## 验证

每次 Docker 操作后，验证结果：

- **容器已启动？** → `docker ps`（检查状态为 "Up"）
- **日志无异常？** → `docker logs --tail 20 NAME`（无报错）
- **端口可访问？** → `curl -s http://localhost:PORT` 或 `docker port NAME`
- **镜像已构建？** → `docker images | grep TAG`
- **Compose 栈健康？** → `docker compose ps`（所有服务状态为 "running" 或 "healthy"）
- **磁盘已释放？** → `docker system df`（对比清理前后）

## Dockerfile 优化建议

审查或创建 Dockerfile 时，建议以下改进：

1. **多阶段构建** — 将构建环境与运行时分离，减小最终镜像体积
2. **层顺序** — 将依赖放在源码之前，避免变更使缓存层失效
3. **合并 RUN 命令** — 减少层数，缩小镜像体积
4. **使用 .dockerignore** — 排除 `node_modules`、`.git`、`__pycache__` 等
5. **固定基础镜像版本** — 使用 `node:20-alpine` 而非 `node:latest`
6. **以非 root 用户运行** — 添加 `USER` 指令以提升安全性
7. **使用 slim/alpine 基础镜像** — 使用 `python:3.12-slim` 而非 `python:3.12`