---
sidebar_position: 0
title: CLI 커맨드 (CLI Commands)
description: 모든 Hermes 터미널 커맨드 및 하위 커맨드에 대한 종합적인 참조 매뉴얼.
---

# Hermes CLI 참조 (CLI Reference)

모든 Hermes CLI 명령어, 하위 명령어(subcommands) 및 플래그에 대한 참조 안내서입니다.

`hermes` (또는 `hermes-agent`) 실행 파일은 에이전트의 주요 진입점입니다. 모든 상태(세션 데이터, 메모리, 구성)는 기본적으로 `~/.hermes/` 디렉터리에 저장됩니다.

## 전역 옵션 (Global Options)

이 옵션들은 어떤 하위 명령어와도 함께 사용할 수 있습니다.

| 옵션 | 설명 |
|--------|-------------|
| `-p`, `--profile <name>` | 명령어 실행 시 대상 프로필을 지정합니다. (기본값: 현재 사용 중인 프로필, 보통 "default") |
| `--version` | 버전 번호를 출력하고 종료합니다. |
| `--help` | 명령어 또는 하위 명령어에 대한 도움말 메시지를 표시합니다. |

## 핵심 명령어 (Core Commands)

### `hermes setup`

```bash
hermes setup [--non-interactive]
```

대화형 첫 실행 설정 마법사입니다. 이 명령어는 다음과 같은 작업을 수행합니다:
1. Nous Portal OAuth 로그인 및 하드웨어 구성(텔레메트리 옵트아웃 포함)을 처리합니다.
2. (선택적으로) Discord, Telegram 및 Slack에 대한 메시징 플랫폼 자격 증명을 수집합니다.
3. 선호하는 LLM 제공자(Anthropic, OpenAI, OpenRouter 등)를 선택하고 API 키를 입력하라는 프롬프트를 표시합니다.
4. `~/.hermes/.env` 및 `~/.hermes/config.yaml`을 생성합니다.

이 명령어는 기존 `.env`나 `config.yaml` 파일을 덮어쓰지 않고 추가하거나 수정합니다. 자동화를 위해 `--non-interactive` 플래그를 사용할 수 있지만, 관련 환경 변수를 제공해야 합니다.

### `hermes chat`

```bash
hermes chat [options]
```

터미널에서 직접 에이전트와 대화형 대화를 시작합니다. 세션은 자동으로 저장됩니다.

| 옵션 | 설명 |
|--------|-------------|
| `-q`, `--query <text>` | 단일 프롬프트를 전송하고 응답을 출력한 후, 대화형 모드로 진입하지 않고 즉시 종료합니다. 파이핑(piping) 및 셸 스크립트에 유용합니다. |
| `-m`, `--model <name>` | 이 세션에서 구성된 기본 모델을 임시로 재정의합니다 (예: `anthropic/claude-3-opus`). |
| `-i`, `--image <path/url>` | 쿼리에 이미지를 첨부합니다 (시각 지원 모델이 필요합니다). 여러 이미지를 첨부하려면 여러 번 사용할 수 있습니다. |
| `-a`, `--audio <path>` | 쿼리에 오디오 파일을 첨부합니다 (다중 모달 오디오 지원 모델이 필요합니다). |
| `-f`, `--file <path>` | 텍스트 파일 내용을 컨텍스트로 읽어옵니다. |
| `--session-id <id>` | 특정 이전 세션을 이어서 계속 진행합니다. |
| `--json` | `-q`와 결합하여 사용 시, 에이전트의 내부 사고 과정(thoughts), 도구 호출(tool calls) 및 최종 응답을 포함하는 구조화된 JSON 객체로 출력을 반환합니다. |
| `--system <prompt>` | 시스템 프롬프트(system prompt)를 사용자 지정 텍스트로 임시로 재정의합니다. |

**대화형 모드(Interactive mode) 핫키:**
- `Ctrl+D` (빈 줄에서) 입력 제출
- `Ctrl+C` 에이전트 생성 중단 (취소)
- `Ctrl+C` (대기열에서) 세션 종료
- `/help` 대화형 슬래시 커맨드 보기 (예: `/memory`, `/undo`)

### `hermes send`

```bash
hermes send <target> [-m message] [--stdin]
```

Hermes의 백엔드 아웃바운드 라우터를 통해 메시지를 전송합니다. 게이트웨이가 실행 중일 때만 작동하며, 스크립트가 연결된 메신저 채팅방이나 그룹으로 알림/경고를 푸시하는 데 사용됩니다.

| 옵션 | 설명 |
|--------|-------------|
| `<target>` | 메시지 목적지. 형식은 `플랫폼:chat_id`입니다 (예: `telegram:123456789`). `hermes gateway list-channels` 명령어를 통해 사람이 읽을 수 있는 이름(예: `#general`)을 확인할 수 있습니다. |
| `-m <text>` | 전송할 메시지 내용입니다. |
| `--stdin` | 파이프라인에서 표준 입력(stdin)을 읽어 메시지 본문으로 사용합니다. |

**예시:**
```bash
# 직접 전송
hermes send telegram:123456789 -m "백업 완료"

# 별칭(Alias)으로 전송
hermes send "#alerts" -m "서버 CPU 90%"

# 파이프(Pipe) 출력 전송
cat report.txt | hermes send "#team" --stdin
```

## `hermes gateway`

```bash
hermes gateway <subcommand>
```

메시징 플랫폼(Telegram, Discord 등)으로 라우팅하는 장기 실행(long-running) 백그라운드 프로세스를 제어합니다. 데몬이 실행 중이 아니면 봇은 메시지에 응답하지 않습니다.

하위 명령어:

| 하위 명령어 | 설명 |
|------------|-------------|
| `start` | 백그라운드 프로세스로 게이트웨이를 시작합니다. (PID는 `~/.hermes/gateway.pid`에 저장됨) |
| `stop` | 게이트웨이를 정상적으로(gracefully) 종료합니다. `--all` 플래그는 프로필과 관계없이 전역 시스템 프로세스를 스캔합니다 (업데이트 중 사용됨). |
| `restart` | 프로세스를 다시 시작합니다. |
| `status` | 백그라운드 프로세스가 현재 실행 중인지, PID는 무엇인지 확인합니다. |
| `run` | 디버깅을 위해 포어그라운드(foreground)에서 게이트웨이를 실행합니다. `Ctrl+C`로 종료합니다. |
| `list-channels` | 연결된 모든 플랫폼에서 게이트웨이가 알고 있는 채팅 ID와 사람이 읽을 수 있는 이름/별칭 목록을 표시합니다. `hermes send` 대상 지정에 유용합니다. |
| `logs` | 게이트웨이의 로그(마지막 50줄)를 확인합니다. 실시간 팔로우는 `-f` 플래그를 추가하세요. |

## `hermes cron`

```bash
hermes cron <subcommand>
```

게이트웨이 데몬 내에서 에이전트 백그라운드 작업 및 예약된 트리거를 관리합니다.

하위 명령어:

| 하위 명령어 | 설명 |
|------------|-------------|
| `list` | 모든 활성 스케줄 크론 및 백그라운드 작업을 표시합니다. (별칭: `ls`) |
| `add` | 새로운 크론 작업을 추가합니다 (아래 옵션 참조). |
| `remove <id>` | ID(예: `task_abc123`)로 작업을 취소하고 삭제합니다. (별칭: `rm`, `delete`) |
| `pause <id>` | 작업을 일시 중단하지만 삭제하지는 않습니다. |
| `resume <id>` | 일시 중단된 작업을 다시 활성화합니다. |
| `run <id>` | 스케줄을 무시하고 대기 중인 작업을 즉시 실행합니다. |

`add` 옵션:

| 옵션 | 설명 |
|--------|-------------|
| `--schedule <expr>` | 스케줄 (일반적인 크론 형식: `* * * * *` 또는 `@daily`). |
| `--prompt <text>` | 크론이 트리거될 때마다 에이전트에게 전달될 명령어입니다. |
| `--target <id>` | 출력 라우팅. `telegram:123456`, `default` (게이트웨이 구성에 정의된 홈 채널), 또는 출력 무시 시 생략. |
| `--name <name>` | 관리하기 쉬운 사람 친화적인 레이블입니다. |

**예시:**
```bash
hermes cron add --schedule "0 9 * * *" --prompt "Hacker News 요약 읽기" --target telegram:123456789
```

## `hermes backup`

```bash
hermes backup [options]
```

`HERMES_HOME` 디렉터리 전체(설정, 환경 변수, 세션, 메모리, 다운로드된 스킬)를 백업하여 타임스탬프가 포함된 압축 파일로 생성합니다. 마이그레이션이나 프로필 간 데이터 이동에 유용합니다.

| 옵션 | 설명 |
|--------|-------------|
| `-o`, `--output <path>` | 파일 생성 위치와 이름을 지정합니다 (기본값: `~/.hermes/backups/hermes-backup-YYYYMMDD.zip`). |
| `--no-sessions` | `sessions/` 디렉터리를 백업에서 제외합니다 (크기가 커질 수 있음). |
| `--state <label>` | 롤백 시스템(curator, update)에서 사용되는 자동 스냅샷을 생성할 때 사람이 읽기 쉬운 라벨을 지정합니다. |
| `restore` (하위 명령어) | 특정 시스템 상태 스냅샷 목록을 보여주거나 복원합니다 (`hermes backup restore --state pre-update`). |

**수동 백업/복원 예시:**
```bash
hermes backup -o ~/hermes-backup-20260423.zip
rsync -av --exclude='hermes-agent' ~/.hermes/ newmachine:~/.hermes/
```

:::tip
`hermes backup`은 게이트웨이가 실행 중인 동안에도 일관된 스냅샷을 생성합니다. 복원된 아카이브는 `gateway.pid` 및 `cron.pid`와 같은 로컬 머신의 실행 관련 파일을 제외합니다.
:::

## `hermes import`

```bash
hermes import <zipfile> [options]
```

이전에 생성한 Hermes 백업을 Hermes 홈 디렉터리에 복원합니다. 아카이브의 모든 파일은 Hermes 홈의 기존 파일을 덮어씁니다. `--force` 옵션을 사용하면 대상에 이미 Hermes가 설치되어 있을 때 나타나는 확인 프롬프트를 건너뜁니다.

| 옵션 | 설명 |
|--------|-------------|
| `-f`, `--force` | 기존 설치에 대한 확인 프롬프트를 건너뜁니다. |

:::warning
실행 중인 프로세스와의 충돌을 방지하려면 가져오기(import) 전에 게이트웨이를 중지하세요.
:::

### 예시
```bash
hermes import ~/hermes-backup-20260423.zip           # 기존 설정 덮어쓰기 전 확인 메시지 표시
hermes import ~/hermes-backup-20260423.zip --force   # 프롬프트 없이 덮어쓰기
```

## `hermes logs`

```bash
hermes logs [log_name] [options]
```

Hermes 로그 파일을 확인하고, 꼬리를 물어(tail) 보고, 필터링합니다. 모든 로그는 `~/.hermes/logs/` (또는 기본이 아닌 프로필의 경우 `<profile>/logs/`)에 저장됩니다.

### 로그 파일 (Log files)

| 이름 | 파일명 | 내용 |
|------|------|-----------------|
| `agent` (기본값) | `agent.log` | 모든 에이전트 활동 — API 호출, 도구 파견, 세션 수명 주기 (INFO 이상) |
| `errors` | `errors.log` | 경고 및 오류 전용 — agent.log의 필터링된 부분집합 |
| `gateway` | `gateway.log` | 메시징 게이트웨이 활동 — 플랫폼 연결, 메시지 전달, 웹훅 이벤트 |
| `gui` | `gui.log` | 대시보드 / TUI-게이트웨이 / PTY-브릿지 / 웹소켓 이벤트 |
| `desktop` | `desktop.log` | Electron 데스크탑 앱 — 부팅, 백엔드 파생 출력 및 최근 Python 트레이스백 |

### 옵션 (Options)

| 옵션 | 설명 |
|--------|-------------|
| `log_name` | 조회할 로그: `agent` (기본값), `errors`, `gateway`, 또는 파일과 크기를 보려면 `list`. |
| `-n`, `--lines <N>` | 표시할 줄 수 (기본값: 50). |
| `-f`, `--follow` | `tail -f`처럼 실시간으로 로그를 추적합니다. 멈추려면 Ctrl+C를 누르세요. |
| `--level <LEVEL>` | 표시할 최소 로그 수준: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. |
| `--session <ID>` | 세션 ID 하위 문자열을 포함하는 줄만 필터링합니다. |
| `--since <TIME>` | 상대 시간 전부터의 줄을 표시합니다: `30m`, `1h`, `2d` 등. `s`(초), `m`(분), `h`(시간), `d`(일)를 지원합니다. |
| `--component <NAME>` | 구성 요소별 필터링: `gateway`, `agent`, `tools`, `cli`, `cron`. |

### 예시

```bash
# agent.log의 마지막 50줄 보기 (기본값)
hermes logs

# 실시간으로 agent.log 추적하기
hermes logs -f

# gateway.log의 마지막 100줄 보기
hermes logs gateway -n 100

# 최근 1시간 동안의 경고 및 오류만 표시
hermes logs --level WARNING --since 1h

# 특정 세션으로 필터링
hermes logs --session abc123

# 30분 전부터 시작하여 실시간으로 errors.log 추적하기
hermes logs errors --since 30m -f

# 모든 로그 파일과 그 크기 목록 표시
hermes logs list
```

### 필터링 (Filtering)

필터는 함께 결합하여 사용할 수 있습니다. 여러 필터가 활성화된 경우, 로그 라인은 **모든** 필터를 통과해야 표시됩니다:

```bash
# 세션 "tg-12345"를 포함하는 최근 2시간 동안의 WARNING+ 줄
hermes logs --level WARNING --since 2h --session tg-12345
```

해석 가능한 타임스탬프가 없는 줄은 `--since`가 활성화되어 있을 때 포함됩니다(다중 줄 로그 항목의 연속 줄일 수 있습니다). 레벨을 감지할 수 없는 줄은 `--level`이 활성화되어 있을 때 포함됩니다.

### 로그 회전 (Log rotation)

Hermes는 Python의 `RotatingFileHandler`를 사용합니다. 오래된 로그는 자동으로 교체(회전)됩니다 — `agent.log.1`, `agent.log.2` 등을 찾으세요. `hermes logs list` 하위 명령어는 회전된 로그를 포함한 모든 로그 파일을 표시합니다.

## `hermes prompt-size`

```bash
hermes prompt-size [--platform <name>] [--json]
```

완전히 새로운 세션에 대한 고정 프롬프트 예산 — 즉, 어떠한 대화 내용이 오가기 *전*에 모든 API 호출 시마다 전송되는 양을 보고합니다. 하류 어댑터(downstream adapter)나 프록시가 모델의 컨텍스트 윈도우보다 엄격한 프롬프트 예산을 가지고 있을 때나 어떤 블록(스킬 인덱스, 메모리, 프로필 등)이 가장 많은 공간을 차지하는지 확인하고 싶을 때 유용합니다.

이 명령어는 에이전트와 동일한 방식으로 시스템 프롬프트를 조립한 다음 구성 요소를 세분화하여 보여줍니다:

- **시스템 프롬프트 합계(System prompt total)** — 조립된 전체 프롬프트 (정체성, 지침, 스킬 인덱스, 컨텍스트 파일, 메모리, 프로필, 타임스탬프).
- **스킬 인덱스(Skills index)** — `<available_skills>` 블록. 많은 스킬이 설치되어 있는 경우 단일 블록으로 가장 큰 경우가 많습니다.
- **메모리(Memory) 및 사용자 프로필(user profile)** — `MEMORY.md` / `USER.md` 스냅샷입니다.
- **프롬프트 티어(Prompt tiers)** — 캐시 효율성을 위해 Hermes가 프롬프트를 계층화하는 방식에 맞춘 안정형(stable) / 컨텍스트형(context) / 휘발성(volatile) 분류.
- **도구 스키마(Tool schemas)** — 활성화된 모든 도구의 JSON (고정 페이로드의 나머지 절반).

모두 오프라인으로 실행되며 — API 호출이 없고 자격 증명이 구성되지 않아도 작동합니다.

```bash
# CLI 플랫폼(기본값)용 사람이 읽기 쉬운 분석
hermes prompt-size

# 메시징 플랫폼의 프롬프트 모방(다른 플랫폼 힌트 사용)
hermes prompt-size --platform telegram

# 스크립트용 기계 판독 가능한 JSON 출력
hermes prompt-size --json
```

:::tip
스킬 인덱스와 도구 스키마는 활성화된 스킬 및 도구 수에 비례하여 커집니다. 프롬프트를 줄이려면 사용하지 않는 도구 세트를 비활성화하거나 (`hermes tools`), 필요 없는 스킬을 제거하세요 (`hermes skills`). 현재 디렉터리의 컨텍스트 파일(`AGENTS.md`, `.cursorrules`)도 총계에 합산됩니다.
:::

## `hermes config`

```bash
hermes config <subcommand>
```

하위 명령어:

| 하위 명령어 | 설명 |
|------------|-------------|
| `show` | 현재 설정값을 표시합니다. |
| `edit` | 에디터에서 `config.yaml`을 엽니다. |
| `set <key> <value>` | 특정 설정값을 지정합니다. |
| `path` | 설정 파일 경로를 출력합니다. |
| `env-path` | `.env` 파일 경로를 출력합니다. |
| `check` | 누락되거나 오래된 구성이 있는지 확인합니다. |
| `migrate` | 새로 도입된 옵션을 대화형으로 추가합니다. |

## `hermes pairing`

```bash
hermes pairing <list|approve|revoke|clear-pending>
```

| 하위 명령어 | 설명 |
|------------|-------------|
| `list` | 대기 중 및 승인된 사용자를 표시합니다. |
| `approve <platform> <code>` | 페어링 코드를 승인합니다. |
| `revoke <platform> <user-id>` | 사용자의 접근 권한을 철회합니다. |
| `clear-pending` | 보류 중인 페어링 코드를 지웁니다. |

## `hermes skills`

```bash
hermes skills <subcommand>
```

하위 명령어:

| 하위 명령어 | 설명 |
|------------|-------------|
| `browse` | 스킬 레지스트리를 페이징하여 탐색하는 브라우저. |
| `search` | 스킬 레지스트리를 검색합니다. |
| `install` | 스킬을 설치합니다. |
| `inspect` | 설치하기 전에 스킬을 미리 봅니다. |
| `list` | 설치된 스킬 목록을 표시합니다. |
| `check` | 설치된 허브(hub) 스킬에서 업스트림(upstream) 업데이트가 있는지 확인합니다. |
| `update` | 업스트림 변경 사항이 있는 경우 허브 스킬을 다시 설치합니다. |
| `audit` | 설치된 허브 스킬을 다시 검사(scan)합니다. |
| `uninstall` | 허브를 통해 설치된 스킬을 제거합니다. |
| `reset` | 사용자가 수정하여 `user_modified` 플래그가 지정된 번들 스킬의 매니페스트 항목을 지워 막힌 상태를 해제합니다. `--restore` 옵션을 함께 사용하면 사용자 복사본을 번들 버전으로 교체합니다. |
| `opt-out` | 활성 프로필에 번들 스킬이 자동으로 채워지는 것을 막습니다. `.no-bundled-skills` 마커를 생성하여 설치 프로그램, `hermes update`, 동기화가 이를 무시하도록 합니다. 기본적으로 안전합니다(디스크 내 아무것도 지우지 않음). `--remove`와 함께 사용 시, 이미 존재하는 번들 스킬 중 **수정되지 않은** 스킬만 삭제합니다(사용자 수정, 허브 설치, 수동 작성 스킬은 삭제되지 않음; 확인 과정 거침, `--yes`로 건너뛰기 가능). |
| `opt-in` | `opt-out`을 취소합니다. 마커를 제거하여 다음 `hermes update` 시 번들 스킬이 다시 채워집니다. `--sync` 옵션을 주면 즉시 적용합니다. |
| `publish` | 스킬을 레지스트리에 게시합니다. |
| `snapshot` | 스킬 설정을 내보내거나(export) 가져옵니다(import). |
| `tap` | 사용자 지정 스킬 출처(source)를 관리합니다. |
| `config` | 플랫폼별로 스킬 활성/비활성을 대화형으로 구성합니다. |

일반적인 예시:

```bash
hermes skills browse
hermes skills browse --source official
hermes skills search react --source skills-sh
hermes skills search https://mintlify.com/docs --source well-known
hermes skills inspect official/security/1password
hermes skills inspect skills-sh/vercel-labs/json-render/json-render-react
hermes skills install official/migration/openclaw-migration
hermes skills install skills-sh/anthropics/skills/pdf --force
hermes skills install https://sharethis.chat/SKILL.md                     # 직접 URL (단일 파일 SKILL.md)
hermes skills install https://example.com/SKILL.md --name my-skill        # 프론트매터에 이름이 없을 때 덮어쓰기
hermes skills check
hermes skills update
hermes skills config
hermes skills reset google-workspace
hermes skills reset google-workspace --restore --yes
hermes skills opt-out                  # 향후 번들 스킬 시딩 중단 (삭제 안 됨)
hermes skills opt-out --remove --yes   # 수정 안 된 번들 스킬 삭제
hermes skills opt-in --sync            # 취소: 마커 제거 및 즉시 재시딩
```

참고:
- `--force`는 서드파티/커뮤니티 스킬에 대한 비위험적(non-dangerous) 정책 차단을 무시할 수 있습니다.
- `--force`는 `dangerous` 검사 결과를 무시하지 못합니다.
- `--source skills-sh`는 공개된 `skills.sh` 디렉터리를 검색합니다.
- `--source well-known`은 `/.well-known/skills/index.json`을 노출하는 사이트에 Hermes를 지정할 수 있게 해줍니다.
- `--source browse-sh`는 200개 이상의 사이트별 브라우저 자동화 스킬을 갖춘 [browse.sh](https://browse.sh) 카탈로그를 검색합니다. 식별자는 `browse-sh/airbnb.com/search-listings-ddgioa`와 같은 형태입니다.
- `http(s)://…/*.md` 형태의 URL을 전달하면 단일 파일 `SKILL.md`를 직접 설치합니다. 프론트매터에 `name:` 항목이 없고 URL 슬러그가 유효한 식별자가 아닐 경우 대화형 터미널에서 이름을 물어봅니다. 비대화형 환경(TUI 게이트웨이 내의 `/skills install`, 플랫폼 등)에서는 `--name <x>`가 필요합니다.

## `hermes bundles`

```bash
hermes bundles <subcommand>
```

스킬 번들은 여러 스킬을 하나의 `/<bundle-name>` 슬래시 커맨드 하위로 그룹화합니다. 번들을 호출하면 참조된 모든 스킬이 단일 통합 사용자 메시지로 로드됩니다. 저장 위치: `~/.hermes/skill-bundles/<slug>.yaml`. YAML 스키마 및 동작은 [스킬 번들(Skill Bundles)](../user-guide/features/skills.md#skill-bundles)을 참조하세요.

하위 명령어:

| 하위 명령어 | 설명 |
|------------|-------------|
| `list` | 설치된 번들 목록 (하위 명령어가 없을 때 기본값) |
| `show <name>` | 특정 번들의 이름, 설명, 포함 스킬 및 파일 경로 표시 |
| `create <name>` | 새 번들을 생성합니다. `--skill <id>`를 반복적으로 넘기거나 생략하여 대화형으로 추가합니다. `--description`, `--instruction`, `--force`를 사용할 수 있습니다. |
| `delete <name>` | 번들 파일을 제거합니다. |
| `reload` | `~/.hermes/skill-bundles/`를 재검사하고 추가/제거된 번들을 보고합니다. |

예시:

```bash
hermes bundles create backend-dev \
  --skill github-code-review \
  --skill test-driven-development \
  --skill github-pr-workflow \
  -d "백엔드 기능 작업"

hermes bundles list
hermes bundles show backend-dev
hermes bundles delete backend-dev
```

채팅 세션 내에서 `/bundles`를 입력하면 설치된 번들이 표시되며, `/<bundle-name>`으로 하나를 불러올 수 있습니다.

## `hermes curator`

```bash
hermes curator <subcommand>
```

큐레이터(curator)는 에이전트가 만든 스킬을 주기적으로 검토하여 오래된 스킬을 잘라내고, 중복을 통합하며, 쓸모없는 스킬을 보관하는 보조 모델 기반 백그라운드 작업입니다. 번들 및 허브 설치 스킬은 절대 건드리지 않습니다. 보관소(Archives)는 복구 가능하며, 자동 삭제는 절대로 발생하지 않습니다.

| 하위 명령어 | 설명 |
|------------|-------------|
| `status` | 큐레이터 상태 및 스킬 통계를 보여줍니다. |
| `run` | 지금 바로 큐레이터 검토를 시작합니다 (LLM 처리가 완료될 때까지 대기). |
| `run --background` | 백그라운드 스레드에서 LLM 검토를 시작하고 즉시 반환합니다. |
| `run --dry-run` | 미리보기 전용 — 변경 없이 검토 보고서만 생성합니다. |
| `backup` | `~/.hermes/skills/`를 수동으로 tar.gz 형식으로 스냅샷 찍습니다 (큐레이터도 실제 실행 전에 자동으로 스냅샷을 찍습니다). |
| `rollback` | `~/.hermes/skills/`를 스냅샷에서 복원합니다 (기본값은 가장 최근 스냅샷). |
| `rollback --list` | 사용 가능한 스냅샷 목록을 보여줍니다. |
| `rollback --id <ts>` | 특정 ID의 스냅샷으로 복원합니다. |
| `rollback -y` | 확인 프롬프트를 건너뜁니다. |
| `pause` | 다시 시작할 때까지 큐레이터를 일시 정지합니다. |
| `resume` | 일시 정지된 큐레이터를 다시 시작합니다. |
| `pin <skill>` | 스킬을 고정(pin)하여 큐레이터가 자동으로 변경하지 못하게 합니다. |
| `unpin <skill>` | 스킬 고정을 해제합니다. |
| `restore <skill>` | 보관된 스킬을 복원합니다. |
| `archive <skill>` | 스킬을 수동으로 보관 처리합니다. |
| `prune` | 큐레이터가 보통 알아서 정리할 스킬을 수동으로 정리합니다. |
| `list-archived` | 보관된 스킬 목록을 보여줍니다 (`restore`로 복구 가능). |

새로 설치 시, 첫 예약된 검사는 하나의 전체 `interval_hours`(기본값 7일) 만큼 연기됩니다 — 게이트웨이는 `hermes update` 후 첫 틱(tick)에서 즉시 큐레이션하지 않습니다. 그 일이 발생하기 전에 `--dry-run`으로 미리보기 해볼 수 있습니다.

동작 및 설정은 [큐레이터(Curator)](../user-guide/features/curator.md)를 참고하세요.

## `hermes fallback`

```bash
hermes fallback <subcommand>
```

폴백 프로바이더(fallback provider) 체인을 관리합니다. 기본 모델이 속도 제한(rate-limit), 과부하, 또는 연결 오류로 실패할 경우 폴백 프로바이더들이 순차적으로 시도됩니다.

| 하위 명령어 | 설명 |
|------------|-------------|
| `list` (별칭: `ls`) | 현재 폴백 체인을 보여줍니다 (하위 명령어가 없을 때 기본값) |
| `add` | 제공자 + 모델을 선택(`hermes model`과 동일한 선택기 사용)하여 체인에 추가합니다. |
| `remove` (별칭: `rm`) | 체인에서 삭제할 항목을 선택합니다. |
| `clear` | 모든 폴백 항목을 제거합니다. |

[폴백 제공자(Fallback Providers)](../user-guide/features/fallback-providers.md) 참고.

## `hermes hooks`

```bash
hermes hooks <subcommand>
```

`~/.hermes/config.yaml`에 정의된 셸 스크립트 훅을 검사하고, 합성 페이로드로 테스트하며, `~/.hermes/shell-hooks-allowlist.json`에 위치한 최초 사용 동의(first-use consent) 허용 목록을 관리합니다.

| 하위 명령어 | 설명 |
|------------|-------------|
| `list` (별칭: `ls`) | 매처(matcher), 시간 제한 및 동의 상태와 함께 구성된 훅을 나열합니다. |
| `test <event>` | 합성 페이로드에 대해 `<event>`와 일치하는 모든 훅을 실행합니다. |
| `revoke` (별칭: `remove`, `rm`) | 명령어의 허용 목록 항목을 제거합니다 (다음 재시작 시 적용됨). |
| `doctor` | 구성된 각 훅 확인: 실행 비트(exec bit), 허용 목록, 수정 시간(mtime) 차이, JSON 유효성 및 합성 실행 타이밍 확인. |

이벤트 시그니처 및 페이로드 형태에 대해서는 [훅(Hooks)](../user-guide/features/hooks.md)을 참조하세요.

## `hermes memory`

```bash
hermes memory <subcommand>
```

외부 메모리 제공자(external memory provider) 플러그인을 설정하고 관리합니다. 사용 가능한 제공자: honcho, openviking, mem0, hindsight, holographic, retaindb, byterover, supermemory. 한 번에 하나의 외부 제공자만 활성화할 수 있습니다. 기본 내장 메모리(MEMORY.md/USER.md)는 항상 활성화되어 있습니다.

하위 명령어:

| 하위 명령어 | 설명 |
|------------|-------------|
| `setup` | 대화형으로 제공자를 선택하고 구성합니다. |
| `status` | 현재 메모리 제공자 구성을 보여줍니다. |
| `off` | 외부 제공자를 비활성화합니다 (내장 메모리만 사용). |

:::info 제공자별 하위 명령어
외부 메모리 제공자가 활성화되면 제공자 전용 관리를 위해 최상위 `hermes <provider>` 명령어(예: Honcho가 활성화된 경우 `hermes honcho`)를 자체 등록할 수 있습니다. 비활성화된 제공자는 해당 하위 명령어를 노출하지 않습니다. `hermes --help`를 실행하여 현재 연결된 명령어를 확인하세요.
:::

## `hermes acp`

```bash
hermes acp
```

에디터 통합을 위해 Hermes를 ACP (Agent Client Protocol) 표준 입출력(stdio) 서버로 시작합니다.

관련 진입점:

```bash
hermes-acp
python -m acp_adapter
```

먼저 지원 패키지를 설치하세요:

```bash
pip install -e '.[acp]'
```

[ACP 에디터 연동(ACP Editor Integration)](../user-guide/features/acp.md) 및 [ACP 내부구조(ACP Internals)](../developer-guide/acp-internals.md) 참조.

## `hermes mcp`

```bash
hermes mcp <subcommand>
```

MCP (Model Context Protocol) 서버 구성을 관리하고 Hermes를 MCP 서버로 실행합니다.

| 하위 명령어 | 설명 |
|------------|-------------|
| *(none)* 또는 `picker` | 대화형 카탈로그 선택기 — Nous 승인 MCP를 둘러보고 설치/활성/비활성화. |
| `catalog` | Nous 승인 MCP 목록 표시 (일반 텍스트 형식, 스크립트화 가능). |
| `install <name>` | 카탈로그 항목 설치 (예: `hermes mcp install n8n`). |
| `serve [-v\|--verbose]` | Hermes를 MCP 서버로 실행 — 다른 에이전트에게 대화를 노출합니다. |
| `add <name> [--url URL] [--command CMD] [--args ...] [--auth oauth\|header]` | 자동 도구 검색을 갖춘 사용자 지정 MCP 서버 추가. |
| `remove <name>` (별칭: `rm`) | 설정에서 MCP 서버 제거. |
| `list` (별칭: `ls`) | 구성된 MCP 서버 나열. |
| `test <name>` | MCP 서버 연결 테스트. |
| `configure <name>` (별칭: `config`) | 특정 서버의 도구 선택 켜기/끄기(Toggle). |
| `login <name>` | OAuth 기반 MCP 서버에 대한 재인증 강제. |

[MCP 설정 레퍼런스(MCP Config Reference)](./mcp-config-reference.md), [Hermes와 MCP 사용하기(Use MCP with Hermes)](../guides/use-mcp-with-hermes.md), 및 [MCP 서버 모드(MCP Server Mode)](../user-guide/features/mcp.md#running-hermes-as-an-mcp-server) 참조.

## `hermes plugins`

```bash
hermes plugins [subcommand]
```

통합 플러그인 관리 — 일반 플러그인, 메모리 프로바이더, 컨텍스트 엔진을 한 곳에서 관리합니다. 인자 없이 `hermes plugins`를 실행하면 두 개의 섹션으로 이루어진 복합 대화형 화면이 열립니다:

- **일반 플러그인(General Plugins)** — 설치된 플러그인을 활성화/비활성화할 수 있는 다중 선택 체크박스
- **제공자 플러그인(Provider Plugins)** — 메모리 제공자 및 컨텍스트 엔진에 대한 단일 선택 구성. 카테고리에서 ENTER를 눌러 라디오 선택기를 엽니다.

| 하위 명령어 | 설명 |
|------------|-------------|
| *(none)* | 복합 대화형 UI — 일반 플러그인 토글 + 제공자 플러그인 구성. |
| `install <identifier> [--force]` | Git URL 또는 `owner/repo`에서 플러그인 설치. |
| `update <name>` | 설치된 플러그인의 최신 변경 사항 가져오기(pull). |
| `remove <name>` (별칭: `rm`, `uninstall`) | 설치된 플러그인 제거. |
| `enable <name>` | 비활성화된 플러그인 다시 활성화. |
| `disable <name>` | 플러그인을 제거하지 않고 비활성화만 수행. |
| `list` (별칭: `ls`) | 활성화/비활성화 상태와 함께 설치된 플러그인 나열. |

제공자 플러그인 선택 항목은 `config.yaml`에 저장됩니다:
- `memory.provider` — 활성 메모리 제공자 (비어 있음 = 내장 메모리만 사용)
- `context.engine` — 활성 컨텍스트 엔진 (`"compressor"` = 내장 기본값)

비활성화된 일반 플러그인 목록은 `config.yaml`의 `plugins.disabled` 항목에 저장됩니다.

[플러그인(Plugins)](../user-guide/features/plugins.md) 및 [Hermes 플러그인 만들기(Build a Hermes Plugin)](../guides/build-a-hermes-plugin.md) 참조.

## `hermes tools`

```bash
hermes tools [--summary]
```

| 옵션 | 설명 |
|--------|-------------|
| `--summary` | 현재 활성화된 도구 요약을 출력하고 종료합니다. |

`--summary`가 없으면 플랫폼별 대화형 도구 설정 UI를 엽니다.

## `hermes computer-use`

```bash
hermes computer-use <subcommand>
```

하위 명령어:

| 하위 명령어 | 설명 |
|------------|-------------|
| `install` | 업스트림 cua-driver 설치 스크립트를 실행합니다 (macOS 전용). |
| `install --upgrade` | cua-driver가 PATH에 이미 있더라도 설치 스크립트를 재실행합니다. 업스트림 스크립트는 항상 최신 릴리스를 받아오기 때문에 이 기능으로 자체 업그레이드가 수행됩니다. |
| `status` | `cua-driver`가 `$PATH` 환경에 위치하는지와 설치된 버전을 출력합니다. |

`hermes computer-use install`은 `computer_use` 도구 세트에서 사용하는 [cua-driver](https://github.com/trycua/cua) 바이너리를 설치하는 안정적인 진입점(entry point)입니다. 이 명령어는 처음 컴퓨터 제어(Computer Use) 기능을 활성화할 때 `hermes tools`가 호출하는 것과 동일한 업스트림 설치 프로그램을 실행합니다. 도구 세트를 켜도 자동으로 설치가 트리거되지 않은 경우(예: 기존 복귀 사용자 설정 등) 다시 설치하기 위해 사용하는 것이 안전합니다.

`hermes update`를 실행하면 cua-driver가 PATH 환경 변수에 존재할 경우 자동으로 업스트림 설치 프로그램이 다시 실행되므로, 대부분의 사용자는 수동으로 `--upgrade`를 호출할 필요가 없습니다. 다음 Hermes 업데이트를 기다리지 않고 업스트림 측의 최신 수정 사항을 당장 반영하고자 할 때 사용하세요.

## `hermes sessions`

```bash
hermes sessions <subcommand>
```

하위 명령어:

| 하위 명령어 | 설명 |
|------------|-------------|
| `list` | 최근 세션들을 나열합니다. |
| `browse` | 검색과 재개가 가능한 대화형 세션 선택기를 실행합니다. |
| `export <output> [--session-id ID]` | 세션들을 JSONL 형식으로 내보냅니다. |
| `delete <session-id>` | 단일 세션을 삭제합니다. |
| `prune` | 오래된 세션들을 삭제합니다. |
| `stats` | 세션 저장소 통계를 보여줍니다. |
| `rename <session-id> <title>` | 세션 제목을 지정하거나 변경합니다. |

## `hermes insights`

```bash
hermes insights [--days N] [--source platform]
```

| 옵션 | 설명 |
|--------|-------------|
| `--days <n>` | 지난 `n`일간의 통계를 분석합니다 (기본값: 30일). |
| `--source <platform>` | `cli`, `telegram`, `discord` 등과 같은 특정 플랫폼/소스 단위로 필터링합니다. |

## `hermes claw`

```bash
hermes claw migrate [options]
```

OpenClaw 설정을 Hermes로 마이그레이션(이전)합니다. `~/.openclaw` (또는 사용자 지정 경로)에서 읽어 `~/.hermes`에 기록합니다. 예전(legacy) 디렉토리명(`~/.clawdbot`, `~/.moltbot`)과 설정 파일명(`clawdbot.json`, `moltbot.json`)도 자동으로 감지합니다.

| 옵션 | 설명 |
|--------|-------------|
| `--dry-run` | 실제 쓰기 작업 없이 무엇이 마이그레이션될지 미리 보여줍니다. |
| `--preset <name>` | 마이그레이션 사전 설정 모드: `full` (호환 가능한 모든 설정) 또는 `user-data` (인프라 구성 제외). 두 사전 설정 모두 비밀키(Secret)를 자동으로 가져오지 않습니다 — `--migrate-secrets`를 명시적으로 넘겨줘야 합니다. |
| `--overwrite` | 충돌이 발생할 경우 기존 Hermes 파일을 덮어씁니다 (기본값: 마이그레이션 계획에 충돌이 있는 경우 적용을 거부합니다). |
| `--migrate-secrets` | API 키를 마이그레이션 항목에 포함합니다. `--preset full` 환경에서도 이 옵션은 필수입니다. |
| `--no-backup` | 마이그레이션 수행 전 `~/.hermes/` 디렉터리의 zip 스냅샷 생성을 생략합니다 (기본적으로 적용 전 복원 가능한 아카이브가 `~/.hermes/backups/pre-migration-*.zip` 위치에 저장되며, `hermes import`로 복원할 수 있습니다). |
| `--source <path>` | 사용자 지정 OpenClaw 디렉터리 경로 (기본값: `~/.openclaw`). |
| `--workspace-target <path>` | 워크스페이스 지침 파일(`AGENTS.md`)의 대상 디렉터리. |
| `--skill-conflict <mode>` | 스킬명 충돌 처리 방식: `skip` (건너뛰기, 기본값), `overwrite` (덮어쓰기), 또는 `rename` (이름 변경). |
| `--yes` | 확인 프롬프트를 생략합니다. |

### 이전(Migrate)되는 내용

마이그레이션은 페르소나, 메모리, 스킬, 모델 제공자, 메시징 플랫폼, 에이전트 동작, 세션 정책, MCP 서버, TTS 등 30개 이상의 범주에 걸쳐 진행됩니다. 항목들은 Hermes 내 상응하는 항목으로 **직접 이전**되거나 수동 검토를 위해 **보관(archived)**됩니다.

**직접 이전(Directly imported):** SOUL.md, MEMORY.md, USER.md, AGENTS.md, 스킬(4개 소스 디렉터리), 기본 모델, 사용자 지정 제공자, MCP 서버, 메시징 플랫폼 토큰 및 허용 목록(Telegram, Discord, Slack, WhatsApp, Signal, Matrix, Mattermost), 에이전트 기본값(추론 노력도, 압축, 응답 지연, 시간대, 샌드박스), 세션 초기화 정책, 승인 규칙, TTS 구성, 브라우저 설정, 도구 설정, 실행 시간 초과(timeout), 명령 허용 목록, 게이트웨이 구성, 그리고 3가지 소스에서 수집된 API 키.

**수동 검토를 위해 보관(Archived for manual review):** Cron 작업, 플러그인, 훅(hooks)/웹훅(webhooks), 메모리 백엔드(QMD), 스킬 레지스트리 구성, UI/Identity(정체성), 로깅, 다중 에이전트 설정, 채널 바인딩, IDENTITY.md, TOOLS.md, HEARTBEAT.md, BOOTSTRAP.md.

**API 키 해결 과정(Resolution)**은 다음 순서로 3가지 소스를 점검합니다: 설정 값 → `~/.openclaw/.env` → `auth-profiles.json`. 모든 토큰 필드는 일반 문자열, 환경 변수 템플릿(`${VAR}`), 그리고 SecretRef 객체를 처리할 수 있습니다.

전체 설정 키 매핑 정보, SecretRef 처리 세부 사항 및 마이그레이션 후 확인 목록에 대해서는 **[전체 마이그레이션 가이드](../guides/migrate-from-openclaw.md)**를 참조하십시오.

### 예시

```bash
# 마이그레이션될 내용 미리보기
hermes claw migrate --dry-run

# 전체 마이그레이션 (비밀키 제외, 호환 가능한 모든 설정 포함)
hermes claw migrate --preset full

# API 키를 포함한 전체 마이그레이션
hermes claw migrate --preset full --migrate-secrets

# 사용자 데이터만 마이그레이션 (비밀키 제외) 후 충돌 발생 시 덮어쓰기
hermes claw migrate --preset user-data --overwrite

# 사용자 지정 OpenClaw 경로에서 마이그레이션 수행
hermes claw migrate --source /home/user/old-openclaw
```

## `hermes dashboard`

```bash
hermes dashboard [options]
```

웹 기반 대시보드를 실행합니다 — 설정 관리, API 키 관리 및 세션 모니터링을 위한 브라우저 전용 UI입니다. `pip install hermes-agent[web]`(FastAPI + Uvicorn) 설치가 선행되어야 합니다. 포함된 브라우저 전용 채팅 탭은 항상 사용 가능하며 `pty` 추가 기능(`pip install 'hermes-agent[web,pty]'`)과 함께 Linux, macOS 또는 WSL2와 같은 POSIX PTY 환경이 필요합니다. 전체 문서는 [웹 대시보드(Web Dashboard)](/user-guide/features/web-dashboard)를 참고하세요.

| 옵션 | 기본값 | 설명 |
|--------|---------|-------------|
| `--port` | `9119` | 웹 서버를 실행할 포트 |
| `--host` | `127.0.0.1` | 바인드(Bind) 주소 |
| `--no-open` | — | 브라우저 자동 실행 비활성화 |
| `--insecure` | off | localhost가 아닌(외부) 호스트에 대한 바인드 허용. 대시보드 접근 권한이 네트워크에 노출되므로, 신뢰할 수 있는 네트워크 통제 하에서만 사용해야 합니다. |
| `--stop` | — | 현재 실행 중인 `hermes dashboard` 프로세스를 중지하고 종료합니다. |
| `--status` | — | 현재 실행 중인 `hermes dashboard` 프로세스 목록을 확인하고 종료합니다. |

```bash
# 기본값 — http://127.0.0.1:9119 주소로 웹 브라우저를 엽니다.
hermes dashboard

# 사용자 지정 포트, 브라우저 열지 않음
hermes dashboard --port 8080 --no-open
```

### `hermes dashboard register`

현재 설치본을 Nous Portal 계정을 기반으로 한 자체 호스팅 대시보드로 등록함으로써 대시보드의 OAuth (Nous) 인증 게이트를 사용할 수 있게 합니다. 이 과정은 기존 Nous 로그인 정보를 활용하며(로그인되어 있지 않은 경우 사전에 `hermes setup`을 실행해야 함), OAuth 클라이언트를 만들고 `HERMES_DASHBOARD_OAUTH_CLIENT_ID`를 `~/.hermes/.env` 파일에 기록한 후 로그인 게이트를 켜는 방법을 안내합니다. 또한 Portal의 [`/local-dashboards`](https://portal.nousresearch.com/local-dashboards) 페이지를 통해 대시보드를 직접 등록, 이름 지정 및 해지할 수도 있습니다.

| 옵션 | 기본값 | 설명 |
|--------|---------|-------------|
| `--name` | 자동 생성 | 대시보드의 사람이 읽기 쉬운 라벨 |
| `--redirect-uri` | — | 외부(인터넷 공개) 호스트를 위한 HTTPS OAuth 리다이렉트 URI (예: `https://hermes.example.com/auth/callback`). 로컬(localhost)에서만 사용하는 경우 생략. |

```bash
hermes dashboard register
# ✓ Registered dashboard "swift_falcon"
# …writes HERMES_DASHBOARD_OAUTH_CLIENT_ID to ~/.hermes/.env
```


## `hermes profile`

```bash
hermes profile <subcommand>
```

프로필을 관리합니다 — 각기 독립적인 설정(config), 세션, 스킬 및 홈 디렉터리를 갖는 분리된 Hermes 인스턴스 여러 개를 사용할 수 있습니다.

| 하위 명령어 | 설명 |
|------------|-------------|
| `list` | 모든 프로필 목록을 보여줍니다. |
| `use <name>` | 변경되지 않는(고정된) 기본 프로필을 지정합니다. |
| `create <name> [--clone] [--clone-all] [--clone-from <source>] [--no-alias]` | 새 프로필 생성. `--clone`은 현재 활성화된 프로필의 설정(config), `.env`, `SOUL.md`를 복사. `--clone-all`은 모든 상태 복사. `--clone-from`은 소스 대상 프로필을 직접 명시. |
| `delete <name> [-y]` | 프로필 삭제. |
| `show <name>` | 프로필의 세부 정보(홈 디렉터리, 설정 등)를 표시. |
| `alias <name> [--remove] [--name NAME]` | 프로필에 빠르게 접근할 수 있도록 래퍼 스크립트(wrapper scripts) 관리. |
| `rename <old> <new>` | 프로필 이름 변경. |
| `export <name> [-o FILE]` | 프로필을 `.tar.gz` 아카이브(로컬 백업)로 내보내기. |
| `import <archive> [--name NAME]` | `.tar.gz` 아카이브(로컬 복원)에서 프로필을 가져오기. |
| `install <source> [--name N] [--alias] [--force] [-y]` | Git URL이나 로컬 디렉토리 경로로부터 프로필 배포본(distribution) 설치. |
| `update <name> [--force-config] [-y]` | 배포본 코드 최신화 (사용자 데이터인 메모리, 세션, 인증은 보존됨). |
| `info <name>` | 프로필의 배포 매니페스트 정보 표시 (버전, 요구사항, 소스). |

예시:

```bash
hermes profile list
hermes profile create work --clone
hermes profile use work
hermes profile alias work --name h-work
hermes profile export work -o work-backup.tar.gz
hermes profile import work-backup.tar.gz --name restored
hermes profile install github.com/user/my-distro --alias
hermes profile update work
hermes -p work chat -q "업무(work) 프로필에서 안녕하세요"
```

## `hermes completion`

```bash
hermes completion [bash|zsh|fish]
```

셸 자동 완성(completion) 스크립트를 표준 출력(stdout)으로 인쇄합니다. 셸 프로필(profile)에서 결과 출력을 소스(source) 파일로 가져오면 Hermes 명령, 하위 명령 및 프로필 이름을 탭(Tab) 키로 자동 완성할 수 있습니다.

예시:

```bash
# Bash
hermes completion bash >> ~/.bashrc

# Zsh
hermes completion zsh >> ~/.zshrc

# Fish
hermes completion fish > ~/.config/fish/completions/hermes.fish
```

## `hermes update`

```bash
hermes update [--gateway] [--check] [--no-backup] [--backup] [--yes]
```

최신 `hermes-agent` 코드를 가져오고(pull) venv 내 의존성 패키지를 재설치한 다음 설치 후(post-install) 훅들(MCP 서버, 스킬 동기화, 자동완성 설치)을 재실행합니다. 시스템이 가동 중(live)일 때도 실행할 수 있어 안전합니다.

**pip 설치본:** `hermes update`는 pip 기반 설치를 자동으로 감지합니다 — PyPI에서 최신 릴리스 정보를 조회한 후 `git pull` 대신 `pip install --upgrade hermes-agent` 명령어를 실행합니다. PyPI 릴리스는 `main` 브랜치의 매 커밋 단위가 아닌 태그된(태그가 지정된) 메이저/마이너 릴리스 버전을 기준으로 따라갑니다. 코드를 가져오거나 설치 또는 재시작하는 과정 없이 단지 새로운 PyPI 릴리스가 있는지 여부만 확인하려면 `--check` 옵션을 사용할 수 있습니다.

| 옵션 | 설명 |
|--------|-------------|
| `--gateway` | 메시징 `/update` 명령어 전용 내부 모드. 프롬프트 표시 및 진행률 스트리밍을 터미널 stdin이 아닌 파일 기반 IPC로 처리. 게이트웨이 재시작 플래그가 아님. |
| `--check` | 어떠한 항목의 다운로드나 재시작 없이 업데이트 가능 여부만 점검. |
| `--no-backup` | `config.yaml`에 `updates.pre_update_backup`이 활성화되어 있어도 이번 실행에서는 업데이트 전 백업 작업을 수행하지 않음. |
| `--backup` | 소스코드를 다운로드(pull)하기 전 `HERMES_HOME` 폴더(config, auth, sessions, skills, pairing data)의 상태 스냅샷을 생성. 기본값은 **off** — 과거 항상 백업을 진행하던 방식은 홈 디렉터리의 크기가 클 때마다 업데이트 시간이 늘어나는 현상을 야기함. 영구적으로 이 기능을 켜고 싶다면 `config.yaml`의 `updates.pre_update_backup: true`로 설정. |
| `--yes`, `-y` | 설정 마이그레이션이나 임시 저장된 파일(stash)의 복구 과정 같은 대화형 프롬프트를 만날 때 전부 'yes' 처리. 단, API 키 입력 단계는 건너뜀 (API 키 처리는 `hermes config migrate` 별도 명령어로 진행). |

추가 동작 특성:

- **게이트웨이 재시작(Gateway restart).** 업데이트가 성공적으로 마무리되면 Hermes는 새로운 코드가 적용되도록 실행 중인 모든 게이트웨이 프로필들을 일괄 재시작 시도합니다. 만약 단순히 최신화 없이 재시작만 원한다면 `hermes gateway restart` 명령어를 활용하세요.
- **페어링 데이터 스냅샷(Pairing data snapshot).** `--backup` 옵션이 꺼져 있더라도, `git pull` 전 `~/.hermes/pairing/` 폴더와 Feishu 댓글(comment) 룰에 한정하여 가벼운 스냅샷이 생성됩니다. 소스코드를 가져오는 중 내가 수정 중이던 파일이 덮어쓰기 될 경우 `hermes backup restore --state pre-update`로 롤백(rollback)이 가능합니다.
- **예전 `hermes.service` 안내 경고.** 이름이 바뀐 `hermes-gateway.service` 대신 옛 방식의 시스템 유닛 파일인 `hermes.service`가 감지되면, 반복적인 꺼짐-켜짐(flap-loop) 현상을 피하기 위해 한 번 시스템 마이그레이션 안내 문구를 띄워 줍니다.
- **종료 코드(Exit codes).** 성공: `0`. 소스 다운로드나 설치 및 설치 후 스크립트 실행 오류: `1`. 사용자가 작업 중인 트리(working-tree)가 변경되어 `git pull`을 가로막는 상태: `2`.

## 유지보수 명령어 (Maintenance commands)

| 명령어 | 설명 |
|---------|-------------|
| `hermes version` | 버전 정보를 출력합니다. |
| `hermes update` | 최신 코드를 가져오고 종속성을 재설치합니다. |
| `hermes postinstall` | 내부용 부트스트랩 명령어. `pip install hermes-agent`(혹은 pip 환경에서 `hermes update`) 실행 직후 한 번 호출되어, pip로 받을 수 없는 외부 종속성(Node.js 런타임, 헤드리스 브라우저, ripgrep, ffmpeg)을 설치한 뒤 프로필이 아직 구성되지 않았다면 `hermes setup`을 촉발. 여러 번 호출해도 안전한 Idempotent 성격. |
| `hermes uninstall [--full] [--yes]` | Hermes를 삭제. 선택적으로 모든 구성 및 데이터를 함께 지울 수 있습니다. |

## 함께 보기 (See also)

- [슬래시 명령어 참고 (Slash Commands Reference)](./slash-commands.md)
- [CLI 인터페이스 (CLI Interface)](../user-guide/cli.md)
- [세션 (Sessions)](../user-guide/sessions.md)
- [스킬 시스템 (Skills System)](../user-guide/features/skills.md)
- [스킨 & 테마 (Skins & Themes)](../user-guide/features/skins.md)
