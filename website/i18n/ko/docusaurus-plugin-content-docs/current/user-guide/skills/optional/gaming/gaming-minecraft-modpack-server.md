---
title: "Minecraft Modpack Server — 모드 마인크래프트 서버 호스팅 (CurseForge, Modrinth)"
sidebar_label: "Minecraft Modpack Server"
description: "모드 마인크래프트 서버 호스팅 (CurseForge, Modrinth)"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Minecraft Modpack Server

모드가 적용된 마인크래프트 서버를 호스팅합니다 (CurseForge, Modrinth).

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/gaming/minecraft-modpack-server` |
| Path | `optional-skills/gaming/minecraft-modpack-server` |
| Platforms | linux, macos |

## Reference: full SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지시 사항으로 보는 내용입니다.
:::

# Minecraft Modpack Server Setup

## When to use
- 사용자가 서버 팩 zip 파일에서 모드 마인크래프트 서버를 설정하려고 할 때
- 사용자가 NeoForge/Forge 서버 구성에 대한 도움이 필요할 때
- 사용자가 마인크래프트 서버 성능 튜닝이나 백업에 대해 질문할 때

## Gather User Preferences First
설정을 시작하기 전에 사용자에게 다음을 물어보세요:
- **서버 이름 / MOTD** — 서버 목록에 뭐라고 표시되어야 합니까?
- **시드(Seed)** — 특정 시드를 원하시나요 아니면 무작위로 할까요?
- **난이도(Difficulty)** — 평화로움 / 쉬움 / 보통 / 어려움?
- **게임모드(Gamemode)** — 서바이벌 / 크리에이티브 / 모험?
- **온라인 모드(Online mode)** — true (Mojang 인증, 정품 계정) 또는 false (LAN/비정품 허용)?
- **플레이어 수** — 몇 명의 플레이어가 예상됩니까? (RAM 및 시야 거리 튜닝에 영향을 미침)
- **RAM 할당** — 직접 정하시겠습니까, 아니면 모드 수와 사용 가능한 RAM에 따라 에이전트가 결정하도록 맡기시겠습니까?
- **시야 거리 / 시뮬레이션 거리** — 직접 정하시겠습니까, 아니면 플레이어 수와 하드웨어에 따라 에이전트가 선택하도록 맡기시겠습니까?
- **PvP** — 켜기 또는 끄기?
- **화이트리스트(Whitelist)** — 개방형 서버 또는 화이트리스트 전용?
- **백업(Backups)** — 자동 백업을 원하십니까? 얼마나 자주 할까요?

사용자가 상관하지 않는다면 합리적인 기본값을 사용하되, 구성을 생성하기 전에 항상 물어보십시오.

## Steps

### 1. Download & Inspect the Pack
```bash
mkdir -p ~/minecraft-server
cd ~/minecraft-server
wget -O serverpack.zip "<URL>"
unzip -o serverpack.zip -d server
ls server/
```
찾을 파일: `startserver.sh`, 설치기 jar 파일 (neoforge/forge), `user_jvm_args.txt`, `mods/` 폴더.
스크립트를 확인하여 모드 로더 유형, 버전 및 필요한 Java 버전을 확인합니다.

### 2. Install Java
- Minecraft 1.21 이상 → Java 21: `sudo apt install openjdk-21-jre-headless`
- Minecraft 1.18-1.20 → Java 17: `sudo apt install openjdk-17-jre-headless`
- Minecraft 1.16 이하 → Java 8: `sudo apt install openjdk-8-jre-headless`
- 확인: `java -version`

### 3. Install the Mod Loader
대부분의 서버 팩에는 설치 스크립트가 포함되어 있습니다. 시작하지 않고 설치만 하려면 INSTALL_ONLY 환경 변수를 사용하세요:
```bash
cd ~/minecraft-server/server
ATM10_INSTALL_ONLY=true bash startserver.sh
# 또는 일반적인 Forge 팩의 경우:
# java -jar forge-*-installer.jar --installServer
```
이렇게 하면 라이브러리가 다운로드되고, 서버 jar가 패치되는 등의 작업이 수행됩니다.

### 4. Accept EULA
```bash
echo "eula=true" > ~/minecraft-server/server/eula.txt
```

### 5. Configure server.properties
모드/LAN 서버를 위한 주요 설정:
```properties
motd=\u00a7b\u00a7lServer Name \u00a7r\u00a78| \u00a7aModpack Name
server-port=25565
online-mode=true          # Mojang 인증 없는 LAN의 경우 false
enforce-secure-profile=true  # online-mode와 일치시킴
difficulty=hard            # 대부분의 모드팩은 '어려움'을 기준으로 밸런스가 맞춰져 있음
allow-flight=true          # 모드 서버에 필수 (비행 탈것/아이템 허용)
spawn-protection=0         # 누구나 스폰 지점에서 건축할 수 있게 함
max-tick-time=180000       # 모드 서버는 긴 틱 타임아웃이 필요함
enable-command-block=true
```

성능 설정 (하드웨어에 맞게 조정):
```properties
# 2인 플레이어, 고성능 컴퓨터:
view-distance=16
simulation-distance=10

# 4-6인 플레이어, 중간 성능 컴퓨터:
view-distance=10
simulation-distance=6

# 8인 이상 플레이어 또는 저성능 하드웨어:
view-distance=8
simulation-distance=4
```

### 6. Tune JVM Args (user_jvm_args.txt)
플레이어 수와 모드 수에 비례하여 RAM을 할당합니다. 모드 서버의 경험 법칙:
- 100-200개 모드: 6-12GB
- 200-350+개 모드: 12-24GB
- OS 및 기타 작업을 위해 최소 8GB를 남겨두십시오.

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

### 7. Open Firewall
```bash
sudo ufw allow 25565/tcp comment "Minecraft Server"
```
확인: `sudo ufw status | grep 25565`

### 8. Create Launch Script
```bash
cat > ~/start-minecraft.sh << 'EOF'
#!/bin/bash
cd ~/minecraft-server/server
java @user_jvm_args.txt @libraries/net/neoforged/neoforge/<VERSION>/unix_args.txt nogui
EOF
chmod +x ~/start-minecraft.sh
```
참고: Forge(NeoForge가 아닌)의 경우 args 파일 경로가 다릅니다. 정확한 경로는 `startserver.sh`를 확인하십시오.

### 9. Set Up Automated Backups
백업 스크립트 만들기:
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

매시간 실행되는 cron 추가:
```bash
(crontab -l 2>/dev/null | grep -v "minecraft/backup.sh"; echo "0 * * * * $HOME/minecraft-server/backup.sh >> $HOME/minecraft-server/backups/backup.log 2>&1") | crontab -
```

## Pitfalls
- 모드 서버에서는 항상 `allow-flight=true`로 설정하세요. 그렇지 않으면 제트팩이나 비행 기능이 있는 모드를 사용할 때 플레이어가 서버에서 쫓겨납니다.
- `max-tick-time=180000` 이상 설정 — 모드 서버는 종종 월드 생성 중에 긴 틱이 발생합니다.
- 첫 시작은 **느립니다** (큰 팩의 경우 몇 분이 소요됨) — 당황하지 마세요.
- 처음 시작할 때 나타나는 "Can't keep up!" 경고는 정상적이며 초기 청크 생성이 끝나면 진정됩니다.
- `online-mode=false`인 경우 `enforce-secure-profile=false`도 설정하세요. 그렇지 않으면 클라이언트 접속이 거부됩니다.
- 모드팩의 `startserver.sh`에는 종종 자동 재시작 루프가 있습니다 — 이것이 없는 깔끔한 실행 스크립트를 새로 만드세요.
- 새 시드로 다시 생성하려면 `world/` 폴더를 삭제하세요.
- 일부 팩은 환경 변수를 사용하여 동작을 제어합니다 (예: ATM10은 ATM10_JAVA, ATM10_RESTART, ATM10_INSTALL_ONLY를 사용함).

## Verification
- 서버가 실행 중인지 확인: `pgrep -fa neoforge` 또는 `pgrep -fa minecraft`
- 로그 확인: `tail -f ~/minecraft-server/server/logs/latest.log`
- 로그에 "Done (Xs)!"가 표시되면 서버가 준비된 것입니다.
- 연결 테스트: 플레이어가 멀티플레이어(Multiplayer) 메뉴에서 서버 IP를 추가하여 테스트합니다.
