---
title: "Docker 管理"
sidebar_label: "Docker 管理"
description: "管理 Docker 容器、镜像、卷、网络和 Compose 栈 — 生命周期操作、调试、清理和 Dockerfile 优化"
---

{/* 此页面由 website/scripts/generate-skill-docs.py 根据 skill 的 SKILL.md 自动生成。请编辑源 SKILL.md，而非此页面。 */}

# Docker 管理

管理 Docker 容器、镜像、卷、网络和 Compose 栈 — 生命周期操作、调试、清理和 Dockerfile 优化。

## 技能元数据

| | |
|---|---|
| 来源 | 可选 — 使用 `hermes skills install official/devops/docker-management` 安装 |
| 路径 | `optional-skills/devops/docker-management` |
| 版本 | `1.0.0` |
| 作者 | sprmn24 |
| 许可证 | MIT |
| 标签 | `docker`、`containers`、`devops`、`infrastructure`、`compose`、`images`、`volumes`、`networks`、`debugging` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 加载此技能时使用的完整技能定义。这是技能激活时代理看到的指令。
:::

# Docker 管理

使用标准 Docker CLI 命令管理 Docker 容器、镜像、卷、网络和 Compose 栈。除了 Docker 本身无需额外依赖。

## 使用场景

- 运行、停止、重启、删除或检查容器
- 构建、拉取、推送、标记或清理 Docker 镜像
- 使用 Docker Compose（多服务栈）
- 管理卷或网络
- 调试崩溃的容器或分析日志
- 检查 Docker 磁盘使用情况或释放空间
- 审查或优化 Dockerfile

## 前置条件

- Docker Engine 已安装并运行
- 用户添加到 `docker` 组（或使用 `sudo`）
- Docker Compose v2（包含在现代 Docker 安装中）

快速检查：

```bash
docker --version && docker compose version
```

## 快速参考

| 任务 | 命令 |
|------|---------|
| 运行容器（后台） | `docker run -d --name NAME IMAGE` |
| 停止 + 删除 | `docker stop NAME && docker rm NAME` |
| 查看日志（跟随） | `docker logs --tail 50 -f NAME` |
| 进入容器 shell | `docker exec -it NAME /bin/sh` |
| 列出所有容器 | `docker ps -a` |
| 构建镜像 | `docker build -t TAG .` |
| Compose 启动 | `docker compose up -d` |
| Compose 停止 | `docker compose down` |
| 磁盘使用 | `docker system df` |
| 清理悬空镜像 | `docker image prune && docker container prune` |

## 步骤

### 1. 识别领域

确定请求属于哪个领域：

- **容器生命周期** → run、stop、start、restart、rm、pause/unpause
- **容器交互** → exec、cp、logs、inspect、stats
- **镜像管理** → build、pull、push、tag、rmi、save/load
- **Docker Compose** → up、down、ps、logs、exec、build、config
- **卷与网络** → create、inspect、rm、prune、connect
- **故障排除** → 日志分析、退出码、资源问题

### 2. 容器操作

**运行新容器：**

```bash
# 后台服务，带端口映射
docker run -d --name web -p 8080:80 nginx

# 带环境变量
docker run -d -e POSTGRES_PASSWORD=secret -e POSTGRES_DB=mydb --name db postgres:16

# 带持久数据（命名卷）
docker run -d -v pgdata:/var/lib/postgresql/data --name db postgres:16

# 用于开发（绑定挂载源代码）
docker run -d -v $(pwd)/src:/app/src -p 3000:3000 --name dev my-app

# 交互式调试（退出时自动删除）
docker run -it --rm ubuntu:22.04 /bin/bash

# 带资源限制和重启策略
docker run -d --memory=512m --cpus=1.5 --restart=unless-stopped --name app my-app
```

关键标志：`-d` 后台、`-it` 交互+tty、`--rm` 自动删除、`-p` 端口（主机:容器）、`-e` 环境变量、`-v` 卷、`--name` 名称、`--restart` 重启策略。

**管理运行中的容器：**

```bash
docker ps                        # 运行中的容器
docker ps -a                     # 所有容器（包括已停止的）
docker stop NAME                 # 优雅停止
docker start NAME                # 启动已停止的容器
docker restart NAME              # 停止 + 启动
docker rm NAME                   # 删除已停止的容器
docker rm -f NAME                # 强制删除运行中的容器
docker container prune           # 删除所有已停止的容器
```

**与容器交互：**

```bash
docker exec -it NAME /bin/sh          # shell 访问（如果可用则使用 /bin/bash）
docker exec NAME env                   # 查看环境变量
docker exec -u root NAME apt update    # 以特定用户运行
docker logs --tail 100 -f NAME         # 跟随最后 100 行
docker logs --since 2h NAME            # 最近 2 小时的日志
docker cp NAME:/path/file ./local      # 从容器复制文件
docker cp ./file NAME:/path/           # 复制文件到容器
docker inspect NAME                    # 完整容器详情（JSON）
docker stats --no-stream               # 资源使用快照
docker top NAME                        # 运行中的进程
```

### 3. 镜像管理

```bash
# 构建
docker build -t my-app:latest .
docker build -t my-app:prod -f Dockerfile.prod .
docker build --no-cache -t my-app .              # 干净重建
DOCKER_BUILDKIT=1 docker build -t my-app .       # 使用 BuildKit 更快

# 拉取和推送
docker pull node:20-alpine
docker login ghcr.io
docker tag my-app:latest registry/my-app:v1.0
docker push registry/my-app:v1.0

# 检查
docker images                          # 列出本地镜像
docker history IMAGE                   # 查看层
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
docker compose up -d --build           # 启动前重建镜像
docker compose down                    # 停止并删除容器
docker compose down -v                 # 也删除卷（销毁数据）

# 监控
docker compose ps                      # 列出服务
docker compose logs -f api             # 跟随特定服务日志
docker compose logs --tail 50          # 所有服务最后 50 行

# 交互
docker compose exec api /bin/sh        # 进入运行中的服务
docker compose run --rm api npm test   # 一次性命令（新容器）
docker compose restart api             # 重启特定服务

# 验证
docker compose config                  # 验证并查看解析后的配置
```

**最小 compose.yml 示例：**

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

### 5. 卷和网络

```bash
# 卷
docker volume ls                       # 列出卷
docker volume create mydata            # 创建命名卷
docker volume inspect mydata           # 详情（挂载点等）
docker volume rm mydata                # 删除（使用中失败）
docker volume prune                    # 删除未使用的卷

# 网络
docker network ls                      # 列出网络
docker network create mynet            # 创建桥接网络
docker network inspect mynet           # 详情（连接的容器）
docker network connect mynet NAME      # 将容器连接到网络
docker network disconnect mynet NAME   # 断开容器连接
docker network rm mynet                # 删除网络
docker network prune                   # 删除未使用的网络
```

### 6. 磁盘使用和清理

清理前始终从诊断开始：

```bash
# 检查空间使用情况
docker system df                       # 摘要
docker system df -v                    # 详细分类

# 针对性清理（安全）
docker container prune                 # 已停止的容器
docker image prune                     # 悬空镜像
docker volume prune                    # 未使用的卷
docker network prune                   # 未使用的网络

# 激进清理（先与用户确认！）
docker system prune                    # 容器 + 镜像 + 网络
docker system prune -a                 # 也删除未使用的镜像
docker system prune -a --volumes       # 一切 — 也包括命名卷
```

**警告：** 在未与用户确认的情况下，永远不要运行 `docker system prune -a --volumes`。这会删除可能包含重要数据的命名卷。

## 陷阱

| 问题 | 原因 | 修复 |
|---------|-------|-----|
| 容器立即退出 | 主进程结束或崩溃 | 检查 `docker logs NAME`，尝试 `docker run -it --entrypoint /bin/sh IMAGE` |
| "端口已被分配" | 另一个进程使用该端口 | `docker ps` 或 `lsof -i :PORT` 查找 |
| "设备空间不足" | Docker 磁盘已满 | `docker system df` 然后针对性清理 |
| 无法连接到容器 | 应用在容器内绑定到 127.0.0.1 | 应用必须绑定到 `0.0.0.0`，检查 `-p` 映射 |
| 卷权限被拒绝 | 主机 vs 容器 UID/GID 不匹配 | 使用 `--user $(id -u):$(id -g)` 或修复权限 |
| Compose 服务无法相互访问 | 网络或服务名称错误 | 服务使用服务名称作为主机名，检查 `docker compose config` |
| 构建缓存不工作 | Dockerfile 中层顺序错误 | 将不常更改的层放在前面（依赖项在源代码之前） |
| 镜像过大 | 无多阶段构建，无 .dockerignore | 使用多阶段构建，添加 `.dockerignore` |

## 验证

在任何 Docker 操作后验证结果：

- **容器启动了？** → `docker ps`（检查状态为"Up"）
- **日志干净？** → `docker logs --tail 20 NAME`（无错误）
- **端口可访问？** → `curl -s http://localhost:PORT` 或 `docker port NAME`
- **镜像构建了？** → `docker images | grep TAG`
- **Compose 栈健康？** → `docker compose ps`（所有服务"running"或"healthy"）
- **空间释放了？** → `docker system df`（比较之前和之后）

## Dockerfile 优化提示

审查或创建 Dockerfile 时，建议以下改进：

1. **多阶段构建** — 将构建环境与运行时分离以减小最终镜像大小
2. **层排序** — 将依赖项放在源代码之前，以便更改不会使缓存层失效
3. **合并 RUN 命令** — 更少的层，更小的镜像
4. **使用 .dockerignore** — 排除 `node_modules`、`.git`、`__pycache__` 等
5. **固定基础镜像版本** — 使用 `node:20-alpine` 而非 `node:latest`
6. **以非 root 用户运行** — 添加 `USER` 指令以提高安全性
7. **使用 slim/alpine 基础** — 使用 `python:3.12-slim` 而非 `python:3.12`
