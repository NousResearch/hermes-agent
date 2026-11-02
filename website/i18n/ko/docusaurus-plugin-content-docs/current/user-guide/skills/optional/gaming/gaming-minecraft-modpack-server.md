---
title: "Minecraft 모드팩 서버 — 모드가 적용된 Minecraft 서버 호스팅 (CurseForge, Modrinth)"
sidebar_label: "Minecraft 모드팩 서버"
description: "모드가 적용된 Minecraft 서버 호스팅 (CurseForge, Modrinth)"
---

{/* 이 페이지는 skill의 SKILL.md에서 website/scripts/generate-skill-docs.py로 자동 생성됩니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Minecraft 모드팩 서버

모드가 적용된 Minecraft 서버를 호스팅합니다 (CurseForge, Modrinth).

## Skill 메타데이터

| | |
|---|---|
| 출처 | Optional — `hermes skills install official/gaming/minecraft-modpack-server`로 설치 |
| 경로 | `optional-skills/gaming/minecraft-modpack-server` |
| 버전 | `1.0.0` |
| 작성자 | Teknium (teknium1), Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos |

## 참고: 전체 SKILL.md

:::info
다음은 이 skill이 활성화될 때 Hermes가 로드하는 전체 skill 정의입니다. skill이 활성 상태일 때 에이전트가 보게 되는 지침입니다.
:::

# Minecraft 모드팩 서버 설정

## 사용 시점
- 사용자가 서버 팩 zip으로 모드가 적용된 Minecraft 서버를 설정하려는 경우
- 사용자가 NeoForge/Forge 서버 구성에 도움이 필요한 경우
- 사용자가 Minecraft 서버 성능 조정 또는 백업에 대해 묻는 경우

## 먼저 사용자 설정 수집
설정을 시작하기 전에 사용자에게 다음을 물어보세요.
- **서버 이름 / MOTD** — 서버 목록에 무엇을 표시할까요?
- **시드** — 특정 시드를 사용할까요, 아니면 무작위로 할까요?
- **난이도** — peaceful / easy / normal / hard 중 무엇으로 할까요?
- **게임 모드** — survival / creative / adventure 중 무엇으로 할까요?
- **온라인 모드** — true (Mojang 인증, 정식 계정) 또는 false (LAN/비공식 계정 허용)?
- **플레이어 수** — 몇 명의 플레이어를 예상하나요? (RAM 및 시야 거리 조정에 영향을 줍니다.)
- **RAM 할당** — 아니면 모드 수와 사용 가능한 RAM을 기준으로 에이전트가 결정할까요?
- **시야 거리 / 시뮬레이션 거리** — 아니면 플레이어 수와 하드웨어를 기준으로 에이전트가 선택할까요?
- **PvP** — 켤까요, 끌까요?
- **화이트리스트** — 공개 서버로 할까요, 화이트리스트 전용으로 할까요?
- **백업** — 자동 백업을 원하나요? 얼마나 자주 할까요?

사용자가 신경 쓰지 않는 항목에는 합리적인 기본값을 사용하되, 구성을 생성하기 전에는 항상 물어보세요.

## 단계

### 1. 팩 다운로드 및 검사
```bash
mkdir -p ~/minecraft-server
cd ~/minecraft-server
wget -O serverpack.zip "<URL>"
unzip -o serverpack.zip -d server
ls server/
```
다음 항목을 찾으세요: `startserver.sh`, 설치 프로그램 jar (neoforge/forge), `user_jvm_args.txt`, `mods/` 폴더.
스크립트를 확인하여 모드 로더 유형, 버전, 필요한 Java 버전을 파악하세요.

### 2. Java 설치
- Minecraft 1.21 이상 → Java 21: `sudo apt install openjdk-21-jre-headless`
- Minecraft 1.18-1.20 → Java 17: `sudo apt install openjdk-17-jre-headless`
- Minecraft 1.16 이하 → Java 8: `sudo apt install openjdk-8-jre-headless`
- 확인: `java -version`

### 3. 모드 로더 설치
대부분의 서버 팩에는 설치 스크립트가 포함되어 있습니다. 실행하지 않고 설치하려면 INSTALL_ONLY 환경 변수를 사용하세요.
```bash
cd ~/minecraft-server/server
ATM10_INSTALL_ONLY=true bash startserver.sh
# Or for generic Forge packs:
# java -jar forge-*-installer.jar --installServer
```
이 과정에서 라이브러리를 다운로드하고, 서버 jar에 패치를 적용하는 등의 작업이 진행됩니다.

### 4. EULA 동의
```bash
echo "eula=true" > ~/minecraft-server/server/eula.txt
```

### 5. server.properties 구성
모드/LAN에 중요한 설정:
```properties
motd=\u00a7b\u00a7lServer Name \u00a7r\u00a78| \u00a7aModpack Name
server-port=25565
online-mode=true          # false for LAN without Mojang auth
enforce-secure-profile=true  # match online-mode
difficulty=hard            # most modpacks balance around hard
allow-flight=true          # REQUIRED for modded (flying mounts/items)
spawn-protection=0         # let everyone build at spawn
max-tick-time=180000       # modded needs longer tick timeout
enable-command-block=true
```

성능 설정 (하드웨어에 맞게 조정):
```properties
# 2 players, beefy machine:
view-distance=16
simulation-distance=10

# 4-6 players, moderate machine:
view-distance=10
simulation-distance=6

# 8+ players or weaker hardware:
view-distance=8
simulation-distance=4
```

### 6. JVM 인수 조정 (user_jvm_args.txt)
플레이어 수와 모드 수에 맞게 RAM을 조정하세요. 모드 서버의 일반적인 기준:
- 모드 100-200개: 6-12GB
- 모드 200-350개 이상: 12-24GB
- OS 및 기타 작업을 위해 최소 8GB는 여유로 남겨 두세요.

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

### 7. 방화벽 열기
```bash
sudo ufw allow 25565/tcp comment "Minecraft Server"
```
다음 명령으로 확인하세요: `sudo ufw status | grep 25565`

### 8. 실행 스크립트 생성
```bash
cat > ~/start-minecraft.sh << 'EOF'
#!/bin/bash
cd ~/minecraft-server/server
java @user_jvm_args.txt @libraries/net/neoforged/neoforge/<VERSION>/unix_args.txt nogui
EOF
chmod +x ~/start-minecraft.sh
```
참고: Forge (NeoForge가 아님)의 경우 인수 파일 경로가 다릅니다. 정확한 경로는 `startserver.sh`에서 확인하세요.

### 9. 자동 백업 설정
백업 스크립트를 생성하세요.
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

매시간 실행되는 cron을 추가하세요.
```bash
(crontab -l 2>/dev/null | grep -v "minecraft/backup.sh"; echo "0 * * * * $HOME/minecraft-server/backup.sh >> $HOME/minecraft-server/backups/backup.log 2>&1") | crontab -
```

## 주의할 점
- 모드 서버에서는 항상 `allow-flight=true`로 설정하세요. 그렇지 않으면 제트팩/비행 모드가 플레이어를 강제 퇴장시킬 수 있습니다.
- `max-tick-time=180000` 이상으로 설정하세요. 모드 서버는 월드 생성 중 긴 틱이 발생하는 경우가 많습니다.
- 첫 시작은 느립니다 (대형 팩은 몇 분이 걸립니다). 당황하지 마세요.
- 첫 실행에서 "Can't keep up!" 경고가 나타나는 것은 정상이며, 초기 청크 생성 후 안정됩니다.
- online-mode=false인 경우 `enforce-secure-profile=false`도 설정하세요. 그렇지 않으면 클라이언트가 거부됩니다.
- 팩의 `startserver.sh`에는 자동 재시작 루프가 포함된 경우가 많습니다. 루프가 없는 깔끔한 실행 스크립트를 만드세요.
- 새 시드로 다시 생성하려면 world/ 폴더를 삭제하세요.
- 일부 팩에는 동작을 제어하는 환경 변수가 있습니다 (예: ATM10은 ATM10_JAVA, ATM10_RESTART, ATM10_INSTALL_ONLY 사용).

## 확인
- 실행 중인지 확인: `pgrep -fa neoforge` 또는 `pgrep -fa minecraft`
- 로그 확인: `tail -f ~/minecraft-server/server/logs/latest.log`
- 로그에서 "Done (Xs)!"를 찾으면 서버가 준비된 것입니다.
- 연결 테스트: 플레이어가 멀티플레이어에서 서버 IP를 추가합니다.
