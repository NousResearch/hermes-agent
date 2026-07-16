---
title: "Hermes S6 容器监督"
sidebar_label: "Hermes S6 容器监督"
description: "修改、调试或扩展 Hermes Agent Docker 镜像内的 s6-overlay 监督树——添加新服务、调试 profile gateway、理解架构 B 主程序模式"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Hermes S6 容器监督

修改、调试或扩展 Hermes Agent Docker 镜像内的 s6-overlay 监督树——添加新服务、调试 profile gateway、理解架构 B 主程序模式。

## 技能元数据

| | |
|---|---|
| 来源 | 可选 — 使用 `hermes skills install official/devops/hermes-s6-container-supervision` 安装 |
| 路径 | `optional-skills/devops/hermes-s6-container-supervision` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 平台 | linux |
| 标签 | `docker`、`s6`、`supervision`、`gateway`、`profiles` |
| 相关技能 | [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent)、`hermes-agent-dev` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在此技能触发时加载的完整技能定义。这是代理在技能激活时看到的指令。
:::

# Hermes s6-overlay 容器监督

## 何时使用

在以下场景加载此技能：
- 在 Hermes Docker 镜像中添加或移除静态服务（应在每次容器启动时受监督的内容，如仪表板）
- 诊断 profile gateway 为何无法启动、重启或在 `docker restart` 后存活
- 理解容器的 CMD 为何是 `/opt/hermes/docker/main-wrapper.sh` 以及前导破折号参数如何到达用户程序
- 修改 `cont-init.d` 启动脚本
- 更改 profile gateway 的渲染运行脚本

## 架构一览

```
/init                                  ← PID 1 (s6-overlay v3.2.3.0)
├── cont-init.d                        ← oneshot 设置，以 root 运行
│   ├── 01-hermes-setup                ← docker/stage2-hook.sh
│   │   ├── UID/GID 重映射
│   │   ├── chown /opt/data
│   │   ├── chown /opt/data/profiles（每次启动）
│   │   ├── seed .env / config.yaml / SOUL.md
│   │   └── skills_sync.py
│   └── 02-reconcile-profiles          ← hermes_cli.container_boot
│
├── s6-rc.d（静态服务，位于 /etc/s6-overlay/s6-rc.d/）
│   ├── main-hermes/run                ← exec sleep infinity（空操作槽位）
│   └── dashboard/run                  ← 如果 HERMES_DASHBOARD=1，运行 `hermes dashboard`
│
├── /run/service（s6-svscan 监控；tmpfs）
│   ├── gateway-coder/                 ← 运行时注册的 per-profile
│   └── ...
│
└── CMD（"主程序"）                     ← /opt/hermes/docker/main-wrapper.sh
```

## 关键文件

| 路径 | 角色 |
|------|------|
| `Dockerfile` | s6-overlay 安装 + cont-init.d 连线 |
| `docker/stage2-hook.sh` | "旧入口逻辑" — UID 重映射、chown、seed、skills 同步 |
| `docker/cont-init.d/02-reconcile-profiles` | 每次启动时调用 `hermes_cli.container_boot` |
| `docker/main-wrapper.sh` | 容器的 CMD。路由用户参数 |
| `docker/s6-rc.d/main-hermes/run` | 空操作 `sleep infinity` |
| `docker/entrypoint.sh` | 向后兼容垫片 |

## 快速操作

### 在运行容器中验证 s6 是 PID 1

```sh
docker exec <c> sh -c 'cat /proc/1/comm; readlink /proc/1/exe'
# 预期：s6-svscan 或 init
```

### 检查 profile gateway 服务

```sh
docker exec <c> /command/s6-svstat /run/service/gateway-<name>
# "up (pid …) … seconds"            → 运行中
# "down (exitcode N) … seconds, normally up, want up, …" → 崩溃循环
```

### 手动启停服务

```sh
docker exec <c> /command/s6-svc -u /run/service/gateway-<name>   # 启动
docker exec <c> /command/s6-svc -d /run/service/gateway-<name>   # 停止
docker exec <c> /command/s6-svc -t /run/service/gateway-<name>   # SIGTERM（重启）
```

### 添加新的静态服务

1. 创建 `docker/s6-rc.d/<name>/type` 写入 `longrun\n`，创建 `docker/s6-rc.d/<name>/run`。
2. 在 run 顶部通过 `s6-setuidgid hermes` 降级到 hermes。
3. 创建空 `docker/s6-rc.d/<name>/dependencies.d/base`。
4. 创建空 `docker/s6-rc.d/user/contents.d/<name>`。
5. Dockerfile 中的 `COPY docker/s6-rc.d/` 会自动拾取——无需其他更改。

## 常见陷阱

### "command not found" 通过 `docker exec`

`/command/`（s6-overlay 放置其二进制文件的地方）仅对监督树生成的进程在 PATH 上。始终使用绝对路径 `/command/s6-svstat`。

### Profile 目录所有权

cont-init 调和器以 hermes 运行。如果 profile 目录最终归 root 所有，调和器无法读取 SOUL.md 并会失败。

### 容器退出 143

检查是否有东西在调用 `s6-svscanctl -t` 或 `/run/s6/basedir/bin/halt`——两者都会导致 /init 开始第 3 阶段关闭但返回 143。
