---
title: "Minecraft模组包服务器 — 托管模组 Minecraft 服务器（CurseForge、Modrinth）"
sidebar_label: "Minecraft 模组包服务器"
description: "托管模组 Minecraft 服务器（CurseForge、Modrinth）"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Minecraft 模组包服务器

托管模组 Minecraft 服务器（CurseForge、Modrinth）。

## 技能元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/gaming/minecraft-modpack-server` |
| 平台 | linux, macos |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发该技能时加载的完整技能定义。这是技能激活时 Agent 所看到的指令内容。
:::

# Minecraft 模组包服务器配置

## 适用场景
- 用户希望从服务器包 zip 文件搭建模组 Minecraft 服务器
- 用户需要 NeoForge/Forge 服务器配置方面的帮助
- 用户询问 Minecraft 服务器性能调优或备份相关问题

## 首先收集用户偏好
开始配置前，向用户询问以下内容：
- **服务器名称 / MOTD** — 服务器列表中显示什么？
- **种子（Seed）** — 指定种子还是随机？
- **难度** — 和平 / 简单 / 普通 / 困难？
- **游戏模式** — 生存 / 创造 / 冒险？
- **在线模式** — true（Mojang 验证，正版账号）还是 false（局域网/离线友好）？
- **玩家数量** — 预计多少玩家同时在线？（影响内存与视距调优）
- **内存分配** — 由用户指定，还是由 Agent 根据模组数量和可用内存决定？
- **视距 / 模拟距离** — 由用户指定，还是由 Agent 根据玩家数量和硬件决定？
- **PvP** — 开启还是关闭？
- **白名单** — 开放服务器还是仅白名单？
- **备份** — 是否需要自动备份？多久一次？

若用户不在意，使用合理默认值，但务必在生成配置前先行询问。

## 步骤

### 1. 下载并检查模组包
```bash
mkdir -p ~/minecraft-server
cd ~/minecraft-server
wget -O serverpack.zip "<URL>"
unzip -o serverpack.zip -d server
ls server/
```
查找：`startserver.sh`、安装器 jar（neoforge/forge）、`user_jvm_args.txt`、`mods/` 文件夹。
检查脚本以确定：模组加载器类型、版本及所需 Java 版本。

### 2. 安装 Java
- Minecraft 1.21+ → Java 21：`sudo apt install openjdk-21-jre-headless`
- Minecraft 1.18-1.20 → Java 17：`sudo apt install openjdk-17-jre-headless`
- Minecraft 1.16 及以下 → Java 8：`sudo apt install openjdk-8-jre-headless`
- 验证：`java -version`

### 3. 安装模组加载器
大多数服务器包包含安装脚本。使用 `INSTALL_ONLY` 环境变量可仅安装而不启动：
```bash
cd ~/minecraft-server/server
ATM10_INSTALL_ONLY=true bash startserver.sh
# 或对于通用 Forge 包：
# java -jar forge-*-installer.jar --installServer
```
此步骤会下载库文件、修补服务器 jar 等。

### 4. 接受 EULA
```bash
echo "eula=true" > ~/minecraft-server/server/eula.txt
```

### 5. 配置 server.properties
模组/局域网的关键设置：
```properties
motd=\u00a7b\u00a7lServer Name \u00a7r\u00a78| \u00a7aModpack Name
server-port=25565
online-mode=true          # false 表示无 Mojang 验证的局域网
enforce-secure-profile=true  # 与 online-mode 保持一致
difficulty=hard            # 大多数模组包以困难难度为平衡基准
allow-flight=true          # 模组服务器必须开启（飞行坐骑/物品）
spawn-protection=0         # 允许所有人在出生点建造
max-tick-time=180000       # 模组服务器需要更长的 tick 超时时间
enable-command-block=true
```

性能设置（根据硬件调整）：
```properties
# 2 名玩家，高性能机器：
view-distance=16
simulation-distance=10

# 4-6 名玩家，中等配置机器：
view-distance=10
simulation-distance=6

# 8+ 名玩家或较弱硬件：
view-distance=8
simulation-distance=4
```

### 6. 调整 JVM 参数（user_jvm_args.txt）
根据玩家数量和模组数量调整内存。模组服务器的经验法则：
- 100-200 个模组：6-12GB
- 200-350+ 个模组：12-24GB
- 为操作系统/其他任务至少保留 8GB 空闲内存

```
-Xms12G
-Xmx24G
-XX:+UseG1GC
-XX:+ParallelRefProcEnabled
-XX:MaxGCPauseMillis=200
-XX:+UnlockExperimentalVMOptions
-XX:+DisableExplicitGC
-XX:+AlwaysPreTouch
-XX:G1NewSizePercent=30
-XX:G1MaxNewSizePercent=40
-XX:G1HeapRegionSize=8M
-XX:G1ReservePercent=20
-XX:G1HeapWastePercent=5
-XX:G1MixedGCCountTarget=4
-XX:InitiatingHeapOccupancyPercent=15
-XX:G1MixedGCLiveThresholdPercent=90
-XX:G1RSetUpdatingPauseTimePercent=5
-XX:SurvivorRatio=32
-XX:+PerfDisableSharedMem
-XX:MaxTenuringThreshold=1
```

### 7. 开放防火墙
```bash
sudo ufw allow 25565/tcp comment "Minecraft Server"
```
检查：`sudo ufw status | grep 25565`

### 8. 创建启动脚本
```bash
cat > ~/start-minecraft.sh << 'EOF'
#!/bin/bash
cd ~/minecraft-server/server
java @user_jvm_args.txt @libraries/net/neoforged/neoforge/<VERSION>/unix_args.txt nogui
EOF
chmod +x ~/start-minecraft.sh
```
注意：对于 Forge（非 NeoForge），参数文件路径不同。请查看 `startserver.sh` 获取确切路径。

### 9. 配置自动备份
创建备份脚本：
```bash
cat > ~/minecraft-server/backup.sh << 'SCRIPT'
#!/bin/bash
SERVER_DIR="$HOME/minecraft-server/server"
BACKUP_DIR="$HOME/minecraft-server/backups"
WORLD_DIR="$SERVER_DIR/world"
MAX_BACKUPS=24
mkdir -p "$BACKUP_DIR"
[ ! -d "$WORLD_DIR" ] && echo "[BACKUP] No world folder" && exit 0
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
BACKUP_FILE="$BACKUP_DIR/world_${TIMESTAMP}.tar.gz"
echo "[BACKUP] Starting at $(date)"
tar -czf "$BACKUP_FILE" -C "$SERVER_DIR" world
SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
echo "[BACKUP] Saved: $BACKUP_FILE ($SIZE)"
BACKUP_COUNT=$(ls -1t "$BACKUP_DIR"/world_*.tar.gz 2>/dev/null | wc -l)
if [ "$BACKUP_COUNT" -gt "$MAX_BACKUPS" ]; then
    REMOVE=$((BACKUP_COUNT - MAX_BACKUPS))
    ls -1t "$BACKUP_DIR"/world_*.tar.gz | tail -n "$REMOVE" | xargs rm -f
    echo "[BACKUP] Pruned $REMOVE old backup(s)"
fi
echo "[BACKUP] Done at $(date)"
SCRIPT
chmod +x ~/minecraft-server/backup.sh
```

添加每小时 cron 任务：
```bash
(crontab -l 2>/dev/null | grep -v "minecraft/backup.sh"; echo "0 * * * * $HOME/minecraft-server/backup.sh >> $HOME/minecraft-server/backups/backup.log 2>&1") | crontab -
```

## 常见问题
- 模组服务器**务必**设置 `allow-flight=true` — 带喷气背包/飞行功能的模组否则会踢出玩家
- `max-tick-time=180000` 或更高 — 模组服务器在世界生成期间经常出现长 tick
- 首次启动**很慢**（大型模组包需要数分钟）— 不必惊慌
- 首次启动时出现"Can't keep up!"警告属正常现象，初始区块生成完成后会恢复
- 若 `online-mode=false`，同时设置 `enforce-secure-profile=false`，否则客户端会被拒绝连接
- 模组包的 `startserver.sh` 通常包含自动重启循环 — 请另行创建不含该循环的干净启动脚本
- 删除 `world/` 文件夹可使用新种子重新生成世界
- 部分模组包使用环境变量控制行为（例如 ATM10 使用 `ATM10_JAVA`、`ATM10_RESTART`、`ATM10_INSTALL_ONLY`）

## 验证
- `pgrep -fa neoforge` 或 `pgrep -fa minecraft` 检查是否正在运行
- 查看日志：`tail -f ~/minecraft-server/server/logs/latest.log`
- 日志中出现"Done (Xs)!"表示服务器已就绪
- 测试连接：玩家在多人游戏中添加服务器 IP