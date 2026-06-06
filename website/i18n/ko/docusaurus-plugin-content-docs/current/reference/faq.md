---
sidebar_position: 14
title: 자주 묻는 질문 (FAQ)
description: 일반적인 문제 해결, 설정 및 오류 해결을 위한 가이드.
---

# 자주 묻는 질문 (FAQ)

## 시작 오류 (Startup Errors)

### "Could not locate `hermes-agent` command"
**증상:** `pip install hermes-agent` 실행 후, 셸에 `hermes: command not found`가 표시됩니다.

**원인:** Python의 전역/사용자 `bin` 디렉터리가 시스템 `$PATH`에 없습니다. 이 문제는 특정 리눅스 배포판 및 macOS 환경에서 흔히 발생합니다.

**해결책:**
1. 설치 프로그램이 Python 스크립트를 어디에 배치했는지 찾습니다. (보통 리눅스의 경우 `~/.local/bin`, macOS의 경우 `~/Library/Python/3.x/bin`).
2. 해당 디렉터리를 셸 프로필(예: `~/.zshrc` 또는 `~/.bashrc`)에 추가합니다.
   ```bash
   export PATH="$HOME/.local/bin:$PATH"
   ```
3. 터미널을 다시 로드합니다 (`source ~/.zshrc`).

### SQLite 잠금 오류 (SQLite Lock Error)

**증상:** `database is locked` 또는 `OperationalError: database is locked`.

**원인:** 여러 개의 Hermes 게이트웨이 인스턴스가 동일한 프로필에서 실행 중이어서 `session.db`를 놓고 충돌하고 있습니다.

**해결책:**
백그라운드 게이트웨이와 CLI를 분리하세요. 백그라운드 프로세스가 실행 중인 경우 이를 중지하세요:
```bash
hermes gateway stop --all
```
여러 개의 봇을 동시에 실행해야 하는 경우, [프로필 (Profiles)](./profile-commands.md)을 사용하여 분리된 환경(`hermes -p work gateway start`, `hermes -p personal gateway start`)에서 봇을 실행하세요.

### ModuleNotFoundError: No module named 'hermes'

**증상:** 소스 코드에서 직접 실행하거나 사용자 지정 플러그인을 가져오려고 할 때 임포트 오류가 발생합니다.

**원인:** 패키지가 활성 가상 환경(venv)에 제대로 설치되지 않았거나 이전 `hermes` 모듈(다른 프로젝트의 모듈)과 충돌합니다.

**해결책:**
가상 환경 내에서 편집 모드로 설치하세요:
```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

---

## 텔레그램 / 디스코드 연동 오류

### 봇이 응답하지 않음 (Bot does not respond)

**증상:** 플랫폼에 봇이 온라인으로 표시되지만 메시지에 답하지 않습니다.

**해결책:**
1. 게이트웨이가 실제로 실행 중인지 확인하세요:
   ```bash
   hermes gateway status
   ```
2. 콘솔에서 오류가 있는지 로그를 확인하세요:
   ```bash
   hermes logs gateway
   ```
3. 가장 흔한 원인은 **사용자 권한**입니다. `.env` 파일에 텔레그램/디스코드 사용자 ID(사용자 이름이 아님)가 올바르게 추가되었는지 확인하세요:
   ```env
   TELEGRAM_ALLOWED_USERS=123456789
   ```

### 텔레그램 409 Conflict

**증상:** `Conflict: terminated by other getUpdates request; make sure that only one bot instance is running`.

**원인:** 텔레그램 봇 토큰 하나에 두 개의 게이트웨이 인스턴스가 연결되어 있습니다. Hermes의 프로필이 여러 개이거나 두 개의 서로 다른 컴퓨터에서 실행 중일 수 있습니다.

**해결책:**
어느 컴퓨터/프로필에서 연결을 유지할지 결정하고 다른 인스턴스에서는 `hermes gateway stop`을 실행하세요. 게이트웨이는 설계상 프로필 내에서 자체적으로 토큰 잠금을 관리하지만, 여러 시스템에 걸쳐서는 이를 조율할 수 없습니다.

### 디스코드 인텐트 오류 (Discord Intents Error)

**증상:** `Privileged message content intent is missing`.

**해결책:** 
디스코드 개발자 포털(Discord Developer Portal)에서 애플리케이션의 봇 설정 페이지로 이동하여 **Message Content Intent**를 활성화해야 합니다. 이 권한이 없으면 봇이 채널 메시지를 읽을 수 없습니다.

---

## 에이전트 및 모델 동작 (Agent & Model Behavior)

### 에이전트가 무한 루프에 빠짐 (Agent is stuck in an infinite loop)

**증상:** 에이전트가 동일한 쉘 명령을 실행하거나 동일한 도구를 반복해서 사용하며 진행되지 않습니다.

**해결책:**
1. 메시징 앱에서 **/stop**을 입력하거나 CLI에서 `Ctrl+C`를 누릅니다.
2. **/undo**를 입력하여 대화 기록에서 모델의 마지막 실수를 제거합니다.
3. 에이전트에게 시도하던 방법을 멈추고 다른 접근 방식을 취하도록 새로운 지시를 내립니다.

### "Tool execution failed" / 모델 포맷 오류

**증상:** 에이전트가 도구를 호출하려고 하지만 JSON 파싱 오류 또는 인수 누락으로 실패합니다.

**원인:** 일부 모델(특히 소형 로컬 모델이나 초기 미세 조정 모델)은 복잡한 도구 스키마(Tool schema)를 사용하는 데 어려움을 겪습니다.

**해결책:**
- 더 강력한 모델로 전환하세요. (예: `anthropic/claude-3.5-sonnet` 또는 `openai/gpt-4o`). 도구를 안정적으로 사용하려면 일반적으로 최소 30B 파라미터급 모델이나 매우 엄격하게 튜닝된 8B 모델(예: Llama-3-8B-Instruct)이 필요합니다.
- `hermes model`을 통해 다른 모델을 선택하세요.

### 모델 성능이 점차 저하됨 (Model degradation over time)

**증상:** 대화가 길어질수록 에이전트가 반복적으로 말하거나 느려지거나 지시를 무시하기 시작합니다.

**원인:** 컨텍스트 윈도우가 가득 차서, 봇이 오래된 대화 기록을 과도하게 분석하고 있습니다.

**해결책:**
**/new**를 입력하세요. 이 명령어는 현재 채팅의 대화 기록은 지우지만 내장된 요약 메모리(`MEMORY.md`)는 유지하므로, 모델이 주요 세부 사항을 잊지 않은 채 깨끗한 상태에서 다시 시작할 수 있습니다.

---

## 스킬 문제 (Skills Issues)

### 스킬이 로드되지 않거나 찾을 수 없음 (Skill fails to load / Not found)

**증상:** 스킬을 설치했지만 에이전트가 이를 사용하지 않거나 도구를 사용할 수 없다고 말합니다.

**원인:** 스킬이 올바른 활성 프로필(`~/.hermes/skills/`)에 설치되지 않았거나 매니페스트 오류가 있습니다.

**해결책:**
```bash
# 스킬이 프로필에 등록되어 있는지 확인
hermes skills list

# 스킬 재설치 시도
hermes skills install official/tools/github-cli --force
```

### 스킬 종속성 누락 (Skill dependency missing)

**증상:** 봇이 "Command not found: jq" (또는 nmap, git 등)라는 오류를 반환합니다.

**해결책:**
많은 스킬들은 로컬 시스템 바이너리에 의존하는 래퍼(Wrapper) 스크립트입니다. 봇의 오류 메시지를 읽고 필요한 도구를 호스트 머신에 직접 설치해야 합니다. (예: `brew install jq` 또는 `apt-get install jq`).

---

## 일반적인 문제 해결 (General Troubleshooting)

### 봇이 "Agent is running"이라고 하며 명령어를 무시함

**증상:** `/model`, `/memory` 등을 입력했을 때 에이전트가 명령어 처리를 거부합니다.

**해결책:**
에이전트가 현재 생성 중이거나 백그라운드 스크립트를 기다리고 있을 때는 상태가 꼬이는 것을 방지하기 위해 명령어 가로채기가 잠깁니다. 에이전트가 작업을 마칠 때까지 기다리거나 **/stop**을 입력하여 진행 중인 작업을 중단시킨 후 원래 하려던 명령어를 입력하세요.

### 완전한 재설정 및 제거 (Clean Reset / Uninstall)

Hermes를 완전히 지우고 처음부터 시작하고 싶을 때:

```bash
# 1. 백그라운드 프로세스 중지
hermes gateway stop --all

# 2. 설치된 애플리케이션 및 의존성 제거
pip uninstall -y hermes-agent

# 3. Hermes 데이터 폴더 이동 (삭제 대비 백업)
mv ~/.hermes ~/.hermes.bak
```
그런 다음 설치 안내에 따라 다시 설치하세요. 기존 세션을 복원하려면 백업 폴더의 `sessions/` 디렉터리와 `.env` 파일을 새 디렉터리로 복사하세요.

### 마이그레이션 백업 롤백 (Rollback a migration backup)

**시나리오:** `hermes claw migrate`를 실행했는데 무언가 잘못되어 되돌리고 싶습니다.

**해결책:** 기본적으로 마이그레이터 도구는 실행 전 항상 `~/.hermes/` 상태를 백업합니다.

```bash
# 사용 가능한 스냅샷 나열
ls ~/.hermes/backups/

# 이전 상태 복원 (다른 이름으로 백업)
hermes import ~/.hermes/backups/pre-migration-20260424-1030.zip --force
```

### 한 컴퓨터에서 다른 컴퓨터로 Hermes 이동

**시나리오:** 새 노트북을 구입하여 동일한 메모리, 스킬, 세션 및 구성을 유지한 채 Hermes를 이동하고 싶습니다.

**해결책:**
가장 쉬운 방법은 내장된 `backup` 및 `import` 명령어를 사용하는 것입니다:

새로운 컴퓨터(Old Machine):
```bash
# 전체 상태를 내보냅니다 (.env 파일의 API 키 포함)
hermes backup -o ~/hermes-transfer.zip
```

.zip 파일을 새 머신으로 안전하게 전송합니다.

새 컴퓨터(New Machine):
```bash
# 파이썬 가상 환경(venv)을 설정하고 빈 상태의 Hermes를 설치합니다
pip install hermes-agent

# 이전 상태 가져오기
hermes import ~/hermes-transfer.zip
```

이전에 시스템 패키지(예: Node.js, `jq`, `ripgrep` 등)를 의존하는 특정 스킬을 설치했다면, 새 머신에서도 이 패키지들이 `$PATH` 환경에 설치되어 있는지 확인해야 합니다.

가져오기 명령어 대신 `rsync`를 사용할 수도 있습니다:
```bash
# 이전 머신에서 실행
rsync -av --exclude='hermes-agent' ~/.hermes/ newmachine:~/.hermes/
```

:::tip
`hermes backup`은 Hermes가 활성화되어 실행 중일 때도 일관된 스냅샷을 생성합니다. 복원된 아카이브는 `gateway.pid` 및 `cron.pid` 같은 머신에 종속된 런타임 파일은 제외합니다.
:::

### 설치 후 셸 로딩 시 Permission denied 오류

**시나리오:** Hermes 설치 프로그램 실행 후, `source ~/.zshrc`를 실행하면 Permission denied 오류가 발생합니다.

**원인:** 이는 일반적으로 `~/.zshrc`(또는 `~/.bashrc`) 파일의 권한이 잘못되었거나 설치 프로그램이 해당 파일에 정상적으로 쓰기 작업을 하지 못했을 때 발생합니다. 이는 Hermes의 고유한 문제가 아니라 셸 설정 파일의 권한 문제입니다.

**해결책:**
```bash
# 권한 확인
ls -la ~/.zshrc

# 필요한 경우 수정 (-rw-r--r-- 또는 644가 되어야 함)
chmod 644 ~/.zshrc

# 셸 다시 로드
source ~/.zshrc

# 또는 새 터미널 창을 열면 변경된 PATH를 자동으로 인식합니다.
```

만약 설치 프로그램이 PATH 라인을 추가했지만 권한이 잘못된 경우, 수동으로 추가할 수 있습니다:
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
```

### 첫 에이전트 실행 시 Error 400 발생

**시나리오:** 설정이 정상적으로 완료되었지만, 첫 번째 채팅 시도에서 HTTP 400 오류가 발생합니다.

**원인:** 일반적으로 모델 이름 불일치 문제입니다. 구성된 모델이 해당 제공자에 존재하지 않거나, 해당 API 키로 접근 권한이 없을 때 발생합니다.

**해결책:**
```bash
# 어떤 모델과 제공자가 구성되어 있는지 확인
hermes config show | head -20

# 모델 선택 마법사 다시 실행
hermes model

# 또는 확인된(정상 작동하는) 모델로 테스트
hermes chat -q "hello" --model anthropic/claude-opus-4.7
```

OpenRouter를 사용하는 경우 API 키에 크레딧(잔액)이 있는지 확인하세요. OpenRouter에서 발생하는 400 오류는 모델을 사용하기 위해 유료 플랜이 필요하거나 모델 ID에 오타가 있음을 의미하는 경우가 많습니다.

---

## 그래도 해결되지 않나요? (Still Stuck?)

여기서 다루지 않은 문제가 있다면:

1. **기존 이슈 검색:** [GitHub Issues](https://github.com/NousResearch/hermes-agent/issues)
2. **커뮤니티에 질문하기:** [Nous Research Discord](https://discord.gg/nousresearch)
3. **버그 보고서 제출:** OS, Python 버전 (`python3 --version`), Hermes 버전 (`hermes --version`) 및 전체 오류 메시지를 포함하여 제출해주세요.
