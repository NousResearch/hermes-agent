---
title: "Minecraft Modpack 服务器 — 托管模组 Minecraft 服务器（CurseForge、Modrinth）"
sidebar_label: "Minecraft Modpack 服务器"
description: "托管模组 Minecraft 服务器（CurseForge、Modrinth）"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Minecraft Modpack 服务器

托管模组 Minecraft 服务器（CurseForge、Modrinth）。

## 技能元数据

| | |
|---|---|
| 来源 | 可选 — 使用 `hermes skills install official/gaming/minecraft-modpack-server` 安装 |
| 路径 | `optional-skills/gaming/minecraft-modpack-server` |
| 平台 | linux、macos |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在此技能触发时加载的完整技能定义。这是代理在技能激活时看到的指令。
:::

# Minecraft Modpack 服务器设置

## 何时使用

- 用户想要从服务器包 zip 设置模组 Minecraft 服务器
- 用户需要 NeoForge/Forge 服务器配置帮助
- 用户询问 Minecraft 服务器性能调优或备份

## 首先收集用户偏好

- **服务器名称 / MOTD**
- **种子** — 具体种子还是随机？
- **难度** — peaceful / easy / normal / hard？
- **游戏模式** — survival / creative / adventure？
- **在线模式** — true（Mojang 认证）还是 false（LAN/破解友好）？
- **玩家数量**
- **RAM 分配**
- **视距 / 模拟距离**
- **PvP** — 开还是关？
- **白名单**
- **备份** — 自动备份？频率？

## 步骤

### 1. 下载并检查包

```bash
mkdir -p ~/minecraft-server
cd ~/minecraft-server
wget -O serverpack.zip "<URL>"
unzip -o serverpack.zip -d server
ls server/
```

### 2. 安装 Java

- Minecraft 1.21+ → Java 21
- Minecraft 1.18-1.20 → Java 17
- Minecraft 1.16 及以下 → Java 8

### 3. 安装模组加载器

```bash
cd ~/minecraft-server/server
ATM10_INSTALL_ONLY=true bash startserver.sh
```

### 4. 接受 EULA

```bash
echo "eula=true" > ~/minecraft-server/server/eula.txt
```

### 5. 配置 server.properties

```properties
motd=\u00a7b\u00a7lServer Name
server-port=25565
online-mode=true
difficulty=hard
allow-flight=true          # 模组必须（飞行坐骑/物品）
spawn-protection=0
max-tick-time=180000
enable-command-block=true
```

### 6. 调优 JVM 参数

```
100-200 个模组：6-12GB
200-350+ 个模组：12-24GB
至少为 OS/其他任务保留 8GB
```

### 7. 开放防火墙

```bash
sudo ufw allow 25565/tcp comment "Minecraft Server"
```

### 8. 创建启动脚本

### 9. 设置自动备份

## 陷阱

- 始终为模组设置 `allow-flight=true` — 带喷气背包/飞行的模组会踢出玩家
- `max-tick-time=180000` 或更高 — 模组服务器在世界生成时经常有长时间 tick
- 首次启动很慢（大包需要几分钟）— 不要恐慌
- 如果 online-mode=false，也设置 enforce-secure-profile=false
- 包的 startserver.sh 通常有自动重启循环——制作干净的启动脚本
