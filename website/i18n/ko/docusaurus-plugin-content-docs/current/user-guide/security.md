---
sidebar_position: 8
title: "보안 (Security)"
description: "보안 모델, 위험한 명령어 승인, 사용자 권한 부여, 컨테이너 격리 및 프로덕션 배포 모범 사례"
---

# 보안 (Security)

Hermes Agent는 심층 방어(defense-in-depth) 보안 모델로 설계되었습니다. 이 페이지에서는 명령어 승인부터 컨테이너 격리, 메시징 플랫폼의 사용자 권한 부여에 이르는 모든 보안 경계를 다룹니다.

## 개요

보안 모델은 7가지 계층으로 구성됩니다:

1. **사용자 권한 부여 (User authorization)** — 누가 에이전트와 대화할 수 있는지 (허용 목록, DM 페어링)
2. **위험한 명령어 승인 (Dangerous command approval)** — 파괴적인 작업에 대한 human-in-the-loop(사람 개입) 승인
3. **컨테이너 격리 (Container isolation)** — 강화된 설정이 적용된 Docker/Singularity/Modal 샌드박싱
4. **MCP 자격 증명 필터링 (MCP credential filtering)** — MCP 하위 프로세스에 대한 환경 변수 격리
5. **컨텍스트 파일 스캐닝 (Context file scanning)** — 프로젝트 파일 내 프롬프트 인젝션(prompt injection) 감지
6. **세션 간 격리 (Cross-session isolation)** — 세션은 서로의 데이터나 상태에 접근할 수 없습니다. 크론(cron) 작업의 저장 경로는 경로 탐색(path traversal) 공격에 대비해 강화됩니다.
7. **입력 무해화 (Input sanitization)** — 셸 인젝션(shell injection)을 방지하기 위해 터미널 도구 백엔드의 작업 디렉토리 매개변수는 허용 목록(allowlist)에 대해 유효성 검사를 거칩니다.

## 위험한 명령어 승인

Hermes는 명령어를 실행하기 전에 선별된 위험 패턴 목록과 대조하여 확인합니다. 일치하는 패턴이 발견되면 사용자가 명시적으로 승인해야 합니다.

### 승인 모드

승인 시스템은 `~/.hermes/config.yaml`의 `approvals.mode`를 통해 구성할 수 있는 세 가지 모드를 지원합니다:

```yaml
approvals:
  mode: manual                    # manual | smart | off
  timeout: 60                     # 사용자 응답을 기다리는 시간 (초) (기본값: 60)
  cron_mode: deny                 # deny | approve — 크론 작업이 위험한 명령어를 만났을 때의 동작
  mcp_reload_confirm: true        # /reload-mcp가 MCP 도구 캐시를 무효화하기 전에 확인을 요청함
  destructive_slash_confirm: true # /clear, /new, /reset, /undo가 상태를 폐기하기 전에 프롬프트를 표시함
```

전체 키 설정:

| 키 | 기본값 | 제어 대상 |
|---|---|---|
| `mode` | `manual` | 위험한 셸 명령어에 대한 승인 정책 — 아래 표를 참조하세요. |
| `timeout` | `60` | Hermes가 타임아웃되기 전에 승인 응답을 기다리는 시간(초)입니다. |
| `cron_mode` | `deny` | [크론 작업](./features/cron.md)이 헤드리스 모드로 실행 중 위험한 명령어 프롬프트를 트리거할 때의 동작입니다. `deny`는 명령어를 차단합니다(에이전트는 다른 경로를 찾아야 합니다). `approve`는 크론 컨텍스트 내의 모든 것을 자동 승인합니다. |
| `mcp_reload_confirm` | `true` | true일 경우, `/reload-mcp`가 MCP 도구 세트를 다시 빌드하기 전에 확인을 요청합니다. 다시 빌드하면 프로바이더 프롬프트 캐시가 무효화되므로(도구 스키마는 시스템 프롬프트에 위치함), 다음 메시지는 전체 입력 토큰을 다시 전송합니다. **항상 승인(Always Approve)**을 클릭한 사용자는 이 키를 `false`로 변경합니다. |
| `destructive_slash_confirm` | `true` | true일 경우, 파괴적인 세션 슬래시 명령어(`/clear`, `/new`, `/reset`, `/undo`)가 대화 상태를 폐기하기 전에 확인을 요청합니다. Telegram, Discord, Slack에서는 네이티브 yes/no 버튼을 통해 라우팅되는 3가지 옵션 대화 상자(한 번 승인 / 항상 승인 / 취소)가 제공되며, 그 외의 경우 텍스트로 대체됩니다. **항상 승인(Always Approve)**을 클릭한 사용자는 이 키를 `false`로 변경합니다. TUI는 자체 모달 오버레이를 사용합니다(비활성화하려면 `HERMES_TUI_NO_CONFIRM=1`로 설정). |

| 모드 | 동작 |
|------|----------|
| **manual** (기본값) | 위험한 명령어에 대해 항상 사용자에게 승인을 요청합니다. |
| **smart** | 보조 LLM을 사용하여 위험을 평가합니다. 위험도가 낮은 명령어(예: `python -c "print('hello')"`)는 자동 승인됩니다. 진정으로 위험한 명령어는 자동 거부됩니다. 불확실한 경우는 수동 프롬프트로 에스컬레이션됩니다. |
| **off** | 모든 승인 확인을 비활성화합니다 — `--yolo` 플래그를 사용하여 실행하는 것과 동일합니다. 모든 명령어가 프롬프트 없이 실행됩니다. |

:::warning
`approvals.mode: off`로 설정하면 모든 안전 확인 프롬프트가 비활성화됩니다. 신뢰할 수 있는 환경(CI/CD, 컨테이너 등)에서만 사용하세요.
:::

### YOLO 모드

YOLO 모드는 현재 세션에 대한 **모든** 위험한 명령어 승인 프롬프트를 우회합니다. 세 가지 방법으로 활성화할 수 있습니다:

1. **CLI 플래그**: `hermes --yolo` 또는 `hermes chat --yolo`로 세션 시작
2. **슬래시 명령어**: 세션 중에 `/yolo`를 입력하여 켜기/끄기 전환
3. **환경 변수**: `HERMES_YOLO_MODE=1`로 설정

`/yolo` 명령어는 **토글(toggle)** 기능으로, 사용할 때마다 모드가 켜지거나 꺼집니다:

```
> /yolo
  ⚡ YOLO mode ON — 모든 명령어가 자동 승인됩니다. 주의해서 사용하세요.

> /yolo
  ⚠ YOLO mode OFF — 위험한 명령어는 승인이 필요합니다.
```

YOLO 모드는 CLI 및 게이트웨이 세션 모두에서 사용할 수 있습니다. 내부적으로는 모든 명령어 실행 전에 확인되는 `HERMES_YOLO_MODE` 환경 변수를 설정합니다.

YOLO가 활성화되면 승인 프롬프트가 우회되고 있다는 사실을 잊지 않도록 두 가지 지속적인 시각적 알림을 표시합니다:

- 세션 시작 시 이미 YOLO가 활성화되어 있는 경우 표시되는 빨간색 배너 줄: `⚠ YOLO mode — all approval prompts bypassed`. 기본 배너를 깔끔하게 유지하기 위해 YOLO가 꺼져 있을 때는 숨겨집니다.
- 모든 너비 계층에 걸친 상태 표시줄의 `⚠ YOLO` 프래그먼트. YOLO를 켜거나 끌 때 실시간으로 업데이트됩니다 (리치 텍스트 렌더러 및 일반 텍스트 폴백).

:::danger
YOLO 모드는 현재 세션에 대한 **모든** 위험 명령어 안전 확인을 비활성화합니다 — **단, 엄격한 차단 목록(Hardline Blocklist)은 예외입니다** (아래 참조). 생성되는 명령어를 완전히 신뢰할 수 있을 때만 사용하세요 (예: 폐기 가능한 환경에서 잘 테스트된 자동화 스크립트).
:::

파괴적인 세션 슬래시 명령어(`/clear`, `/new` / `/reset`, `/undo`, `/quit --delete` — `/exit --delete`는 별칭)의 경우, CLI는 실행 전에 확인 프롬프트를 표시합니다. [Slash Commands — 파괴적인 명령어에 대한 확인 프롬프트](../reference/slash-commands.md#confirmation-prompts-for-destructive-commands)를 참조하세요.

### 엄격한 차단 목록 (항상 켜져 있는 최소한의 안전장치)

일부 명령어는 매우 치명적이어서 (복구 불가능한 파일 시스템 삭제, 포크 폭탄, 블록 디바이스 직접 쓰기 등) Hermes는 **다음 조건과 상관없이** 실행을 거부합니다:

- `--yolo` / `/yolo` 토글 켜짐
- `approvals.mode: off`
- 헤드리스 `approve` 모드로 실행 중인 크론 작업
- 사용자가 명시적으로 "항상 허용(allow always)"을 클릭함

차단 목록은 `--yolo` 아래의 마지노선입니다. 승인 계층이 명령어를 보기 **전에** 작동하며, 무시할 수 있는 플래그는 없습니다. 현재 다루고 있는 패턴은 다음과 같습니다 (전체 목록은 아님; `tools/approval.py::UNRECOVERABLE_BLOCKLIST`와 동기화됨):

| 패턴 | 차단되는 이유 |
|---|---|
| `rm -rf /` 및 명백한 변형 | 파일 시스템 루트를 삭제합니다 |
| `rm -rf --no-preserve-root /` | 명시적인 "네, 루트를 삭제합니다" 변형 |
| `:(){ :\|:& };:` (bash 포크 폭탄) | 재부팅할 때까지 호스트를 마비시킵니다 |
| 마운트된 루트 장치에서의 `mkfs.*` | 실행 중인 시스템을 포맷합니다 |
| `dd if=/dev/zero of=/dev/sd*` | 물리적 디스크를 0으로 덮어씁니다 |
| 루트 파일 시스템 최상위 수준에서 신뢰할 수 없는 URL을 `sh`로 파이핑 | 원격 코드 실행(RCE) 공격 벡터가 너무 광범위하여 승인할 수 없습니다 |

차단 목록에 걸리면 도구 호출은 에이전트에게 설명이 포함된 오류를 반환하고 아무것도 실행되지 않습니다. 합법적인 워크플로우에 이러한 명령어 중 하나가 필요한 경우 (예: 삭제 및 재설치 파이프라인의 운영자인 경우), 에이전트 외부에서 실행하세요.

### 승인 시간 초과 (Approval Timeout)

위험한 명령어 프롬프트가 나타나면 사용자는 구성 가능한 시간 동안 응답할 수 있습니다. 타임아웃 내에 응답이 없으면 명령어는 기본적으로 **거부**됩니다(fail-closed).

`~/.hermes/config.yaml`에서 타임아웃을 구성하세요:

```yaml
approvals:
  timeout: 60  # 초 단위 (기본값: 60)
```

### 승인을 트리거하는 항목

다음 패턴은 승인 프롬프트를 트리거합니다 (`tools/approval.py`에 정의됨):

| 패턴 | 설명 |
|---------|-------------|
| `rm -r` / `rm --recursive` | 재귀적 삭제 |
| `rm ... /` | 루트 경로에서 삭제 |
| `chmod 777/666` / `o+w` / `a+w` | 모든 사용자/기타 사용자 쓰기 가능 권한 |
| `chmod --recursive` with unsafe perms | 재귀적 모든 사용자/기타 사용자 쓰기 가능 (긴 플래그) |
| `chown -R root` / `chown --recursive root` | 루트로 재귀적 소유권 변경 |
| `mkfs` | 파일 시스템 포맷 |
| `dd if=` | 디스크 복사 |
| `> /dev/sd` | 블록 장치에 쓰기 |
| `DROP TABLE/DATABASE` | SQL DROP |
| `DELETE FROM` (WHERE 절 없이) | WHERE 절 없는 SQL DELETE |
| `TRUNCATE TABLE` | SQL TRUNCATE |
| `> /etc/` | 시스템 구성 덮어쓰기 |
| `systemctl stop/restart/disable/mask` | 시스템 서비스 중지/재시작/비활성화/마스킹 |
| `kill -9 -1` | 모든 프로세스 종료 |
| `pkill -9` | 프로세스 강제 종료 |
| Fork bomb patterns | 포크 폭탄 |
| `bash -c` / `sh -c` / `zsh -c` / `ksh -c` | `-c` 플래그를 통한 셸 명령어 실행 (`-lc`와 같이 결합된 플래그 포함) |
| `python -e` / `perl -e` / `ruby -e` / `node -c` | `-e`/`-c` 플래그를 통한 스크립트 실행 |
| `curl ... \| sh` / `wget ... \| sh` | 원격 콘텐츠를 셸로 파이프 |
| `bash <(curl ...)` / `sh <(wget ...)` | 프로세스 대체를 통한 원격 스크립트 실행 |
| `tee`를 사용하여 `/etc/`, `~/.ssh/`, `~/.hermes/.env`로 쓰기 | tee를 통해 민감한 파일 덮어쓰기 |
| `>` / `>>`를 사용하여 `/etc/`, `~/.ssh/`, `~/.hermes/.env`로 쓰기 | 리디렉션을 통해 민감한 파일 덮어쓰기 |
| `xargs rm` | rm과 함께 xargs 사용 |
| `find -exec rm` / `find -delete` | 파괴적인 작업과 함께 find 사용 |
| `cp`/`mv`/`install`을 `/etc/`로 실행 | 시스템 구성으로 파일 복사/이동 |
| `/etc/`에서 `sed -i` / `sed --in-place` | 시스템 구성 제자리 수정 (in-place edit) |
| `pkill`/`killall` hermes/gateway | 자체 종료 방지 |
| `&`/`disown`/`nohup`/`setsid`와 함께 `gateway run` 실행 | 서비스 관리자 외부에서 게이트웨이를 시작하는 것을 방지 |

:::info
**컨테이너 우회**: `docker`, `singularity`, `modal`, 또는 `daytona` 백엔드에서 실행할 때, 컨테이너 자체가 보안 경계이기 때문에 위험한 명령어 검사는 **건너뜁니다(skipped)**. 컨테이너 내부의 파괴적인 명령어는 호스트에 해를 끼칠 수 없습니다.
:::

### 승인 흐름 (CLI)

대화형 CLI에서 위험한 명령어는 인라인 승인 프롬프트를 표시합니다:

```
  ⚠️  DANGEROUS COMMAND: recursive delete
      rm -rf /tmp/old-project

      [o]nce  |  [s]ession  |  [a]lways  |  [d]eny

      Choice [o/s/a/D]:
```

네 가지 옵션:

- **once** — 이 한 번의 실행만 허용합니다.
- **session** — 세션의 남은 시간 동안 이 패턴을 허용합니다.
- **always** — 영구 허용 목록에 추가합니다 (`config.yaml`에 저장됨).
- **deny** (기본값) — 명령어를 차단합니다.

### 승인 흐름 (게이트웨이/메시징)

메시징 플랫폼에서 에이전트는 위험한 명령어의 세부 정보를 채팅으로 전송하고 사용자의 응답을 기다립니다:

- 승인하려면 **yes**, **y**, **approve**, **ok**, 또는 **go**로 회신합니다.
- 거부하려면 **no**, **n**, **deny**, 또는 **cancel**로 회신합니다.

게이트웨이를 실행할 때 `HERMES_EXEC_ASK=1` 환경 변수가 자동으로 설정됩니다.

### 영구 허용 목록 (Permanent Allowlist)

"always"로 승인된 명령어는 `~/.hermes/config.yaml`에 저장됩니다:

```yaml
# 영구적으로 허용된 위험한 명령어 패턴
command_allowlist:
  - rm
  - systemctl
```

이 패턴들은 시작 시 로드되며 향후 모든 세션에서 조용히 승인됩니다.

:::tip
`hermes config edit`를 사용하여 영구 허용 목록에서 패턴을 검토하거나 제거하세요.
:::

## 사용자 권한 부여 (게이트웨이)

메시징 게이트웨이를 실행할 때 Hermes는 계층화된 권한 부여 시스템을 통해 봇과 상호작용할 수 있는 사용자를 제어합니다.

### 권한 확인 순서

`_is_user_authorized()` 메서드는 다음 순서로 확인합니다:

1. **플랫폼별 모든 사용자 허용 플래그** (예: `DISCORD_ALLOW_ALL_USERS=true`)
2. **DM 페어링 승인 목록** (페어링 코드를 통해 승인된 사용자)
3. **플랫폼별 허용 목록** (예: `TELEGRAM_ALLOWED_USERS=12345,67890`)
4. **글로벌 허용 목록** (`GATEWAY_ALLOWED_USERS=12345,67890`)
5. **글로벌 모든 사용자 허용** (`GATEWAY_ALLOW_ALL_USERS=true`)
6. **기본값: deny(거부)**

### 플랫폼 허용 목록

허용된 사용자 ID를 `~/.hermes/.env`에 쉼표로 구분된 값으로 설정합니다:

```bash
# 플랫폼별 허용 목록
TELEGRAM_ALLOWED_USERS=123456789,987654321
DISCORD_ALLOWED_USERS=111222333444555666
WHATSAPP_ALLOWED_USERS=15551234567
SLACK_ALLOWED_USERS=U01ABC123

# 크로스 플랫폼 허용 목록 (모든 플랫폼에서 확인됨)
GATEWAY_ALLOWED_USERS=123456789

# 플랫폼별 모든 사용자 허용 (주의해서 사용)
DISCORD_ALLOW_ALL_USERS=true

# 글로벌 모든 사용자 허용 (극도의 주의를 기울여 사용)
GATEWAY_ALLOW_ALL_USERS=true
```

:::warning
**허용 목록이 구성되어 있지 않고** `GATEWAY_ALLOW_ALL_USERS`가 설정되어 있지 않으면 **모든 사용자가 거부됩니다**. 게이트웨이는 시작 시 경고를 기록합니다:

```
No user allowlists configured. All unauthorized users will be denied.
Set GATEWAY_ALLOW_ALL_USERS=true in ~/.hermes/.env to allow open access,
or configure platform allowlists (e.g., TELEGRAM_ALLOWED_USERS=your_id).
```
:::

### DM 페어링 시스템

보다 유연한 권한 부여를 위해 Hermes는 코드 기반 페어링 시스템을 포함합니다. 사전에 사용자 ID를 요구하는 대신, 알 수 없는 사용자는 봇 소유자가 CLI를 통해 승인하는 일회성 페어링 코드를 받습니다.

**작동 방식:**

1. 알 수 없는 사용자가 봇에게 DM을 보냅니다.
2. 봇은 8자의 페어링 코드로 응답합니다.
3. 봇 소유자가 CLI에서 `hermes pairing approve <platform> <code>`를 실행합니다.
4. 해당 플랫폼에 대해 사용자가 영구적으로 승인됩니다.

승인되지 않은 다이렉트 메시지가 처리되는 방식을 `~/.hermes/config.yaml`에서 제어할 수 있습니다:

```yaml
unauthorized_dm_behavior: pair

whatsapp:
  unauthorized_dm_behavior: ignore
```

- `pair`가 기본값입니다. 권한이 없는 DM은 페어링 코드 응답을 받습니다.
- `ignore`는 권한이 없는 DM을 조용히 무시합니다.
- 플랫폼 섹션은 전역 기본값을 무시하므로, WhatsApp은 무음으로 유지하면서 Telegram에서는 페어링을 유지할 수 있습니다.

**보안 기능** (OWASP + NIST SP 800-63-4 지침 기반):

| 기능 | 세부 정보 |
|---------|---------|
| 코드 형식 | 32자 명확한 알파벳에서 8자 (0/O/1/I 없음) |
| 무작위성 | 암호학적 무작위 (`secrets.choice()`) |
| 코드 TTL | 1시간 후 만료 |
| 속도 제한 | 사용자당 10분마다 1회 요청 |
| 대기 제한 | 플랫폼당 최대 3개의 보류 중인 코드 |
| 잠금(Lockout) | 5회 승인 실패 → 1시간 잠금 |
| 파일 보안 | 모든 페어링 데이터 파일에 대해 `chmod 0600` 적용 |
| 로깅 | 코드는 절대로 stdout에 기록되지 않습니다. |

**페어링 CLI 명령어:**

```bash
# 보류 중이거나 승인된 사용자 나열
hermes pairing list

# 페어링 코드 승인
hermes pairing approve telegram ABC12DEF

# 사용자의 접근 권한 취소
hermes pairing revoke telegram 123456789

# 보류 중인 모든 코드 지우기
hermes pairing clear-pending
```

**저장소:** 페어링 데이터는 `~/.hermes/pairing/` 내에 플랫폼별 JSON 파일로 저장됩니다:
- `{platform}-pending.json` — 보류 중인 페어링 요청
- `{platform}-approved.json` — 승인된 사용자
- `_rate_limits.json` — 속도 제한 및 잠금 추적

## 컨테이너 격리

`docker` 터미널 백엔드를 사용할 때 Hermes는 모든 컨테이너에 엄격한 보안 강화를 적용합니다.

### Docker 보안 플래그

모든 컨테이너는 다음 플래그와 함께 실행됩니다 (`tools/environments/docker.py`에 정의됨):

```python
_SECURITY_ARGS = [
    "--cap-drop", "ALL",                          # 모든 Linux 권한(capabilities) 제거
    "--cap-add", "DAC_OVERRIDE",                  # 루트가 바인드 마운트된 디렉토리에 쓸 수 있음
    "--cap-add", "CHOWN",                         # 패키지 관리자에게 파일 소유권이 필요함
    "--cap-add", "FOWNER",                        # 패키지 관리자에게 파일 소유권이 필요함
    "--security-opt", "no-new-privileges",         # 권한 상승 차단
    "--pids-limit", "256",                         # 프로세스 수 제한
    "--tmpfs", "/tmp:rw,nosuid,size=512m",         # 크기 제한이 있는 /tmp
    "--tmpfs", "/var/tmp:rw,noexec,nosuid,size=256m",  # 실행 불가능한 /var/tmp
    "--tmpfs", "/run:rw,noexec,nosuid,size=64m",   # 실행 불가능한 /run
]
```

### 리소스 제한

컨테이너 리소스는 `~/.hermes/config.yaml`에서 구성할 수 있습니다:

```yaml
terminal:
  backend: docker
  docker_image: "nikolaik/python-nodejs:python3.11-nodejs20"
  docker_forward_env: []  # 명시적인 허용 목록만 가능; 비워두면 컨테이너에 시크릿이 들어가지 않습니다
  container_cpu: 1        # CPU 코어 수
  container_memory: 5120  # MB 단위 (기본값 5GB)
  container_disk: 51200   # MB 단위 (기본값 50GB, XFS의 overlay2 필요)
  container_persistent: true  # 세션 전반에 걸쳐 파일 시스템 유지
```

### 파일 시스템 지속성 (Filesystem Persistence)

- **지속성 모드** (`container_persistent: true`): `~/.hermes/sandboxes/docker/<task_id>/`에서 `/workspace`와 `/root`를 바인드 마운트합니다.
- **임시 모드** (`container_persistent: false`): 워크스페이스에 tmpfs를 사용합니다 — 정리 시 모든 내용이 손실됩니다.

:::tip
프로덕션 게이트웨이 배포의 경우, 호스트 시스템에서 에이전트 명령어를 격리하기 위해 `docker`, `modal` 또는 `daytona` 백엔드를 사용하세요. 이렇게 하면 위험한 명령어 승인이 전혀 필요하지 않습니다.
:::

:::warning
`terminal.docker_forward_env`에 이름을 추가하면 해당 변수들은 의도적으로 터미널 명령어를 위해 컨테이너에 주입됩니다. 이는 `GITHUB_TOKEN`과 같은 작업 관련 자격 증명에 유용하지만, 컨테이너에서 실행되는 코드가 이를 읽고 유출할 수 있음을 의미하기도 합니다.
:::

## 터미널 백엔드 보안 비교

| 백엔드 | 격리 수준 | 위험한 명령어 확인 | 적합한 용도 |
|---------|-----------|-------------------|----------|
| **local** | 없음 — 호스트에서 실행됨 | ✅ 예 | 개발, 신뢰할 수 있는 사용자 |
| **ssh** | 원격 시스템 | ✅ 예 | 별도의 서버에서 실행 |
| **docker** | 컨테이너 | ❌ 건너뜀 (컨테이너가 경계임) | 프로덕션 게이트웨이 |
| **singularity** | 컨테이너 | ❌ 건너뜀 | HPC 환경 |
| **modal** | 클라우드 샌드박스 | ❌ 건너뜀 | 확장 가능한 클라우드 격리 |
| **daytona** | 클라우드 샌드박스 | ❌ 건너뜀 | 영구적인 클라우드 워크스페이스 |

## 환경 변수 패스스루 {#environment-variable-passthrough}

`execute_code`와 `terminal`은 모두 LLM 생성 코드에 의한 자격 증명 유출을 방지하기 위해 하위 프로세스에서 민감한 환경 변수를 제거합니다. 그러나 `required_environment_variables`를 선언하는 스킬은 합법적으로 해당 변수에 대한 접근이 필요합니다.

### 작동 방식

두 가지 메커니즘을 통해 특정 변수가 샌드박스 필터를 통과할 수 있습니다:

**1. 스킬 범위 패스스루 (자동)**

스킬이 로드될 때 (`skill_view` 또는 `/skill` 명령어를 통해) `required_environment_variables`를 선언하면, 환경에 실제로 설정된 해당 변수 중 일부가 자동으로 패스스루로 등록됩니다. (여전히 설정이 필요한 상태의) 누락된 변수는 **등록되지 않습니다**.

```yaml
# 스킬의 SKILL.md frontmatter 내부
required_environment_variables:
  - name: TENOR_API_KEY
    prompt: Tenor API key
    help: https://developers.google.com/tenor 에서 키를 발급받으세요
```

이 스킬을 로드한 후, `TENOR_API_KEY`는 수동 구성 없이도 `execute_code`, `terminal` (로컬) 및 **원격 백엔드 (Docker, Modal)**로 전달됩니다.

:::info Docker & Modal
v0.5.1 이전에는 Docker의 `forward_env`가 스킬 패스스루와 별도의 시스템이었습니다. 이제 이들이 병합되었습니다. 스킬에서 선언한 환경 변수는 `docker_forward_env`에 수동으로 추가할 필요 없이 Docker 컨테이너 및 Modal 샌드박스로 자동 전달됩니다.
:::

**2. 구성 기반 패스스루 (수동)**

어떤 스킬에서도 선언하지 않은 환경 변수의 경우, `config.yaml`의 `terminal.env_passthrough`에 추가하세요:

```yaml
terminal:
  env_passthrough:
    - MY_CUSTOM_KEY
    - ANOTHER_TOKEN
```

### 자격 증명 파일 패스스루 (OAuth 토큰 등) {#credential-file-passthrough}

일부 스킬은 샌드박스에서 환경 변수뿐만 아니라 **파일**을 필요로 합니다. 예를 들어, Google Workspace는 OAuth 토큰을 활성 프로필의 `HERMES_HOME` 아래에 `google_token.json`으로 저장합니다. 스킬은 이를 frontmatter에 선언합니다:

```yaml
required_credential_files:
  - path: google_token.json
    description: Google OAuth2 token (설정 스크립트에 의해 생성됨)
  - path: google_client_secret.json
    description: Google OAuth2 client credentials
```

로드될 때 Hermes는 이러한 파일이 활성 프로필의 `HERMES_HOME`에 존재하는지 확인하고 마운트를 위해 등록합니다:

- **Docker**: 읽기 전용 바인드 마운트 (`-v host:container:ro`)
- **Modal**: 샌드박스 생성 시 마운트 + 각 명령어 실행 전 동기화 (세션 중간 OAuth 설정 처리)
- **Local**: 추가 작업 필요 없음 (파일에 이미 접근 가능함)

`config.yaml`에 수동으로 자격 증명 파일을 나열할 수도 있습니다:

```yaml
terminal:
  credential_files:
    - google_token.json
    - my_custom_oauth_token.json
```

경로는 `~/.hermes/`에 상대적입니다. 파일은 컨테이너 내부의 `/root/.hermes/`에 마운트됩니다. 이 목록은 `tools/credential_files.py`(`terminal.credential_files`)에 의해 읽힙니다. 이는 `terminal:` 블록 아래에 존재하지만, 핵심 터미널 백엔드가 아닌 자격 증명 파일 모듈에 의해 로드되므로 번들된 `DEFAULT_CONFIG` 스냅샷의 일부가 아닙니다.

### 각 샌드박스가 필터링하는 항목

| 샌드박스 | 기본 필터 | 패스스루 재정의(Override) |
|---------|---------------|---------------------|
| **execute_code** | 이름에 `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, `CREDENTIAL`, `PASSWD`, `AUTH`를 포함하는 변수 차단; 안전한 접두사(safe-prefix) 변수만 허용 | ✅ 패스스루 변수는 두 확인 과정을 모두 우회함 |
| **terminal** (로컬) | 명시적인 Hermes 인프라 변수 (프로바이더 키, 게이트웨이 토큰, 도구 API 키) 차단 | ✅ 패스스루 변수는 차단 목록을 우회함 |
| **terminal** (Docker) | 기본적으로 호스트 환경 변수 없음 | ✅ 패스스루 변수 + `docker_forward_env`가 `-e`를 통해 전달됨 |
| **terminal** (Modal) | 기본적으로 호스트 환경/파일 없음 | ✅ 자격 증명 파일 마운트됨; 환경 패스스루는 동기화를 통해 이루어짐 |
| **MCP** | 안전한 시스템 변수 + 명시적으로 구성된 `env`를 제외한 모든 항목 차단 | ❌ 패스스루의 영향을 받지 않음 (대신 MCP `env` 구성 사용) |

### 보안 고려 사항

- 패스스루는 당신이나 당신의 스킬이 명시적으로 선언한 변수에만 영향을 미칩니다 — 임의의 LLM 생성 코드에 대한 기본 보안 태세는 변경되지 않습니다.
- 자격 증명 파일은 Docker 컨테이너에 **읽기 전용**으로 마운트됩니다.
- Skills Guard는 스킬을 설치하기 전에 의심스러운 환경 접근 패턴이 있는지 스킬 내용을 검사합니다.
- 누락되거나 설정되지 않은 변수는 절대 등록되지 않습니다 (존재하지 않는 것은 유출될 수 없습니다).
- Hermes 인프라 시크릿(프로바이더 API 키, 게이트웨이 토큰)은 절대로 `env_passthrough`에 추가해서는 안 됩니다 — 별도의 전용 메커니즘이 있습니다.

## MCP 자격 증명 처리

MCP (Model Context Protocol) 서버 하위 프로세스는 우발적인 자격 증명 유출을 방지하기 위해 **필터링된 환경**을 수신합니다.

### 안전한 환경 변수

다음 변수만 호스트에서 MCP stdio 하위 프로세스로 전달됩니다:

```
PATH, HOME, USER, LANG, LC_ALL, TERM, SHELL, TMPDIR
```

추가로 모든 `XDG_*` 변수가 포함됩니다. 다른 모든 환경 변수(API 키, 토큰, 시크릿)는 **제거(stripped)**됩니다.

MCP 서버의 `env` 구성에 명시적으로 정의된 변수는 통과됩니다:

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_..."  # 이 변수만 전달됨
```

### 자격 증명 편집 (Redaction)

MCP 도구의 오류 메시지는 LLM으로 반환되기 전에 무해화됩니다. 다음 패턴은 `[REDACTED]`로 대체됩니다:

- GitHub PATs (`ghp_...`)
- OpenAI 스타일 키 (`sk-...`)
- Bearer 토큰
- `token=`, `key=`, `API_KEY=`, `password=`, `secret=` 매개변수

### 웹사이트 접근 정책

에이전트가 웹 및 브라우저 도구를 통해 접근할 수 있는 웹사이트를 제한할 수 있습니다. 이는 에이전트가 내부 서비스, 관리자 패널 또는 기타 민감한 URL에 접근하는 것을 방지하는 데 유용합니다.

```yaml
# ~/.hermes/config.yaml 내부
security:
  website_blocklist:
    enabled: true
    domains:
      - "*.internal.company.com"
      - "admin.example.com"
    shared_files:
      - "/etc/hermes/blocked-sites.txt"
```

차단된 URL이 요청되면, 도구는 도메인이 정책에 의해 차단되었음을 설명하는 오류를 반환합니다. 차단 목록은 `web_search`, `web_extract`, `browser_navigate` 및 모든 URL 지원 도구 전반에 적용됩니다.

자세한 내용은 구성 가이드의 [웹사이트 차단 목록 (Website Blocklist)](/user-guide/configuration#website-blocklist)을 참조하세요.

### SSRF 방지

모든 URL 지원 도구(웹 검색, 웹 추출, 비전, 브라우저)는 서버 사이드 요청 위조(SSRF) 공격을 방지하기 위해 가져오기 전에 URL의 유효성을 검사합니다. 차단된 주소는 다음과 같습니다:

- **사설 네트워크** (RFC 1918): `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`
- **루프백 (Loopback)**: `127.0.0.0/8`, `::1`
- **링크 로컬 (Link-local)**: `169.254.0.0/16` (클라우드 메타데이터 `169.254.169.254` 포함)
- **CGNAT / 공유 주소 공간** (RFC 6598): `100.64.0.0/10` (Tailscale, WireGuard VPNs)
- **클라우드 메타데이터 호스트 이름**: `metadata.google.internal`, `metadata.goog`
- **예약된 주소, 멀티캐스트 및 지정되지 않은 주소**

SSRF 보호는 인터넷 연결 사용을 위해 항상 활성화되어 있으며, DNS 오류는 차단된 것으로 간주됩니다(fail-closed). 리디렉션 체인은 리디렉션 기반 우회를 방지하기 위해 각 홉에서 다시 유효성이 검사됩니다.

#### 의도적으로 사설 URL 허용하기

일부 설정에서는 `home.arpa`를 RFC 1918 공간으로 리졸브하는 홈 네트워크, LAN 전용 Ollama/llama.cpp 엔드포인트, 내부 위키, 클라우드 메타데이터 디버깅 등 사설/내부 URL 접근이 합법적으로 필요합니다. 이러한 경우를 위해 전역 옵트아웃이 있습니다:

```yaml
security:
  allow_private_urls: true   # 기본값: false
```

이 옵션을 켜면 웹 도구, 브라우저, 비전 URL 가져오기 및 게이트웨이 미디어 다운로드가 더 이상 RFC 1918 / 루프백 / 링크 로컬 / CGNAT / 클라우드 메타데이터 대상을 거부하지 않습니다. **이것은 의도적인 신뢰 경계(trust boundary)입니다** — 로컬 네트워크에 대해 프롬프트 인젝션된 임의의 URL을 실행하는 에이전트의 위험을 수용할 수 있는 머신에서만 이 옵션을 활성화하세요. 외부로 노출된 게이트웨이는 이 옵션을 꺼두어야 합니다.

기본 IP가 퍼블릭이더라도 유사해 보이는 유니코드 도메인 트릭을 차단하는 호스트 하위 문자열 방어(host-substring guard)는 이 설정과 관계없이 켜져 있습니다.

### Tirith 실행 전 보안 스캐닝

Hermes는 실행 전에 콘텐츠 수준의 명령어 스캐닝을 위해 [tirith](https://github.com/sheeki03/tirith)를 통합합니다. Tirith는 패턴 매칭만으로는 놓칠 수 있는 위협을 감지합니다:

- 동형문자(Homograph) URL 스푸핑 (국제화 도메인 공격)
- 파이프를 인터프리터로 연결하는 패턴 (`curl | bash`, `wget | sh`)
- 터미널 인젝션 공격

Tirith는 첫 사용 시 GitHub 릴리스에서 SHA-256 체크섬 확인(및 cosign이 사용 가능한 경우 cosign 출처 확인)과 함께 자동으로 설치됩니다.

```yaml
# ~/.hermes/config.yaml 내부
security:
  tirith_enabled: true       # tirith 스캐닝 활성화/비활성화 (기본값: true)
  tirith_path: "tirith"      # tirith 바이너리 경로 (기본값: PATH 조회)
  tirith_timeout: 5          # 하위 프로세스 타임아웃 (초)
  tirith_fail_open: true     # tirith를 사용할 수 없을 때 실행 허용 (기본값: true)
```

`tirith_fail_open`이 `true` (기본값)인 경우, tirith가 설치되어 있지 않거나 시간 초과가 발생해도 명령어가 진행됩니다. tirith를 사용할 수 없을 때 명령어를 차단하려면 높은 보안 환경에서 이 값을 `false`로 설정하세요.

Tirith는 Linux (x86_64 / aarch64) 및 macOS (x86_64 / arm64)용 미리 빌드된 바이너리를 제공합니다. 미리 빌드된 바이너리가 없는 플랫폼(Windows 등)에서는 tirith를 조용히 건너뜁니다 — 패턴 매칭 가드는 여전히 실행되며, CLI는 "사용 불가(unavailable)" 배너를 표시하지 않습니다. Windows에서 tirith를 사용하려면 WSL에서 Hermes를 실행하세요.

Tirith의 판정은 승인 흐름과 통합됩니다: 안전한 명령어는 통과하며, 의심스럽거나 차단된 명령어는 사용자에게 tirith의 전체 결과(심각도, 제목, 설명, 더 안전한 대안)와 함께 승인 프롬프트를 트리거합니다. 사용자는 승인하거나 거부할 수 있으며, 무인 시나리오를 안전하게 유지하기 위해 기본 선택은 '거부'입니다.

### 컨텍스트 파일 인젝션 방지

컨텍스트 파일 (AGENTS.md, .cursorrules, SOUL.md)은 시스템 프롬프트에 포함되기 전에 프롬프트 인젝션 스캔을 거칩니다. 스캐너는 다음을 확인합니다:

- 이전 지침을 무시하라는 지시
- 의심스러운 키워드가 포함된 숨겨진 HTML 주석
- 시크릿 정보(`.env`, `credentials`, `.netrc`)를 읽으려는 시도
- `curl`을 통한 자격 증명 유출
- 보이지 않는 유니코드 문자 (폭이 없는 공백, 양방향 오버라이드)

차단된 파일은 경고를 표시합니다:

```
[BLOCKED: AGENTS.md contained potential prompt injection (prompt_injection). Content not loaded.]
```

## 프로덕션 배포 모범 사례

### 게이트웨이 배포 체크리스트

1. **명시적인 허용 목록 설정** — 프로덕션 환경에서는 절대로 `GATEWAY_ALLOW_ALL_USERS=true`를 사용하지 마세요.
2. **컨테이너 백엔드 사용** — config.yaml에서 `terminal.backend: docker`를 설정하세요.
3. **리소스 제한 설정** — 적절한 CPU, 메모리 및 디스크 제한을 설정하세요.
4. **시크릿의 안전한 저장** — API 키를 적절한 파일 권한이 있는 `~/.hermes/.env`에 보관하세요.
5. **DM 페어링 활성화** — 가능하면 사용자 ID를 하드코딩하는 대신 페어링 코드를 사용하세요.
6. **명령어 허용 목록 검토** — config.yaml의 `command_allowlist`를 주기적으로 감사(audit)하세요.
7. **`terminal.cwd` 설정** — 에이전트가 민감한 디렉토리에서 작동하지 않도록 하세요.
8. **루트 권한 없이 실행** — 게이트웨이를 절대 root로 실행하지 마세요.
9. **로그 모니터링** — 무단 접근 시도가 있는지 `~/.hermes/logs/`를 확인하세요.
10. **최신 상태 유지** — 보안 패치를 위해 주기적으로 `hermes update`를 실행하세요.

### API 키 보호

```bash
# .env 파일에 적절한 권한 설정
chmod 600 ~/.hermes/.env

# 여러 서비스에 대해 분리된 키 유지
# .env 파일을 버전 관리 시스템에 절대 커밋하지 마세요
```

### 네트워크 격리

보안을 극대화하려면 게이트웨이를 별도의 머신이나 VM에서 실행하세요. `config.yaml`에 `terminal.backend: ssh`를 설정한 다음, `~/.hermes/.env`의 환경 변수를 통해 호스트 세부 정보를 제공하세요:

```yaml
# ~/.hermes/config.yaml
terminal:
  backend: ssh
```

```bash
# ~/.hermes/.env
TERMINAL_SSH_HOST=agent-worker.local
TERMINAL_SSH_USER=hermes
TERMINAL_SSH_KEY=~/.ssh/hermes_agent_key
```

SSH 연결 세부 정보는 `.env`에 저장되므로(config.yaml이 아님), 프로필 내보내기와 함께 체크인되거나 공유되지 않습니다. 이는 게이트웨이의 메시징 연결을 에이전트의 명령어 실행과 분리하여 유지합니다.

## 공급망 주의보(advisory) 확인

Hermes에는 알려진 손상된 버전(예: 2026년 5월의 `mistralai 2.4.6` 중독과 같은 공급망 웜)의 선별된 카탈로그와 일치하는 활성 venv 내 Python 패키지를 표시하는 내장 주의보 스캐너가 포함되어 있습니다. 구현 코드는 `hermes_cli/security_advisories.py`에 있습니다.

실행 방식:

- **CLI 시작 배너.** 일치하는 주의보가 있을 경우, 전체 수정 방법을 볼 수 있는 `hermes doctor`에 대한 포인터와 함께 한 줄의 경고가 출력됩니다.
- **`hermes doctor`.** 버전 세부 정보와 2-4단계의 수정 지침과 함께 모든 활성 주의보를 표시합니다.
- **게이트웨이 시작.** `gateway.log`에 기록되며, 첫 번째 대화형 메시지에 짧은 운영자 배너가 표시됩니다.

각 주의보는 안정적인 ID를 가집니다. 읽고 조치를 취한 후에는 영구적으로 해제할 수 있습니다:

```bash
hermes doctor --ack <advisory-id>
```

확인(ack)은 `config.security.acked_advisories`에 지속되며 재시작 후에도 유지됩니다. 기존 주의보는 카탈로그에서 의도적으로 제거되지 **않습니다** — 주의보를 제자리에 두어 사설 미러에 여전히 캐시되어 있을 수 있는 역사적으로 중독된 버전에 대해 새로운 설치본에 경고를 계속합니다.

확인 작업 자체는 stdlib로만 구성되어 있으며 주의보당 한 번의 `importlib.metadata.version()` 조회로 실행되므로 모든 시작 시에 안전하게 실행할 수 있습니다.

### 선택적 종속성의 지연(lazy) 설치

많은 기능(Mistral TTS, ElevenLabs, Honcho 메모리, Bedrock, Slack, Matrix 등)은 모든 사용자가 필요로 하지 않는 Python 패키지에 종속됩니다. Hermes는 `hermes-agent[all]`에 따라 적극적으로 설치하는 대신 첫 사용 시에 **지연하여(lazily)** 설치합니다. 구현 코드는 `tools/lazy_deps.py`에 있습니다.

이것이 해결하는 문제(Trade-off):

- **취약성(Fragility).** 한 추가 패키지의 전이적 종속성이 PyPI에서 사용 불가능해지면 (악성 코드 의심으로 격리, 회수, 업로드 손상 등), 전체 `[all]` 해결(resolve)이 실패하고 새로 설치 시 조용히 기능이 제거된 계층으로 폴백되어 10개 이상의 관련 없는 추가 기능이 한 번에 손실됩니다. 지연 설치는 각 백엔드를 분리하여 하나의 중독된 종속성이 관련 없는 기능을 망가뜨리지 않도록 합니다.
- **블로트(Bloat).** 한 프로바이더하고만 대화하는 사용자는 절대 import하지 않을 수백 개의 패키지를 더 이상 가져오지 않습니다.

작동 방식:

1. 백엔드 모듈이 첫 가져오기 경로 상단에서 `ensure("feature.name")`을 호출합니다.
2. 종속성이 누락된 경우, `ensure`는 `config.yaml`의 `security.allow_lazy_installs` (기본값 `true`)를 확인하고 허용 목록에 있는 사양에 대해 venv 범위의 `pip install`을 실행합니다.
3. 설치에 실패하거나 사용자가 지연 설치를 비활성화한 경우, 해당 호출은 실제 pip stderr 및 `hermes tools`의 포인터와 함께 `FeatureUnavailable`를 발생시킵니다.

`tools/lazy_deps.py`에 의해 강제되는 보안 보장:

| 보장 | 의미 |
|---|---|
| Venv 범위 전용 | 시스템 Python이 아닌 활성 venv 내의 대상 `sys.executable`에 설치합니다 |
| 이름으로 PyPI에서만 | 사양은 `"package>=1.0,<2"` 구문을 허용합니다. `--index-url`, `git+https://` 또는 file: 경로는 사용할 수 없습니다 — 악의적인 `config.yaml`이 설치 대상을 리디렉션할 수 없습니다 |
| 허용 목록 (Allowlist) | 소스 트리에 있는 `LAZY_DEPS` 맵에 나타나는 사양만 이 경로를 통해 설치할 수 있습니다. 기능 이름의 오타는 '아무거나 설치' 로직을 얻지 못합니다 |
| 옵트아웃 (Opt-out) | 런타임 설치를 완전히 비활성화하려면 `security.allow_lazy_installs: false`로 설정하세요. 제한된 네트워크나 엄격한 보안 태세에 유용합니다 |
| 조용한 재시도 없음 | 오류는 `FeatureUnavailable`로 표면화됩니다 — 잘못된 상태를 캐싱하거나 재시도 스톰(retry storm)이 발생하지 않습니다 |

런타임 설치를 비활성화하려면:

```yaml
# ~/.hermes/config.yaml
security:
  allow_lazy_installs: false
```

비활성화된 경우, 선택적 종속성이 필요한 백엔드는 사용자에게 수동으로 설치를 실행하거나 (`pip install …`) `hermes tools`를 통해 다른 백엔드를 선택하라고 알려줍니다.
