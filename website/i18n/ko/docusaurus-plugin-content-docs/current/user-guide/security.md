---
sidebar_position: 8
title: "보안"
description: "보안 모델, 위험한 명령 승인, 사용자 인증, 컨테이너 격리 및 프로덕션 배포 모범 사례"
---

# 보안

Hermes Agent는 심층 방어 보안 모델을 기반으로 설계되었습니다. 이 페이지에서는 명령 승인부터 컨테이너 격리, 메시징 플랫폼의 사용자 인증까지 모든 보안 경계를 다룹니다.

## 개요

보안 모델은 다음 여덟 계층으로 구성됩니다.

1. **사용자 인증** — 에이전트와 대화할 수 있는 사람(허용 목록, DM 페어링)
2. **위험한 명령 승인** — 파괴적 작업에 대한 사람 개입
3. **파일 쓰기 안전성** — `write_file`/`patch`에 대한 거부 목록 및 선택적 쓰기 샌드박스
4. **컨테이너 격리** — 강화된 설정을 적용한 Docker/Singularity/Modal 샌드박싱
5. **MCP 자격 증명 필터링** — MCP 하위 프로세스의 환경 변수 격리
6. **컨텍스트 파일 검사** — 프로젝트 파일의 프롬프트 인젝션 탐지
7. **세션 간 격리** — 세션은 서로의 데이터나 상태에 접근할 수 없으며, cron 작업 저장 경로는 경로 순회 공격에 대비해 강화됩니다
8. **입력 정제** — 터미널 도구 백엔드의 작업 디렉터리 매개변수를 허용 목록에 따라 검증하여 셸 인젝션 방지

## 위험한 명령 승인

명령을 실행하기 전에 Hermes는 엄선된 위험 패턴 목록과 대조합니다. 일치하는 항목이 발견되면 사용자가 명시적으로 승인해야 합니다.

### 승인 모드

승인 시스템은 `~/.hermes/config.yaml`의 `approvals.mode`로 설정하는 세 가지 모드를 지원합니다.

```yaml
approvals:
  mode: smart                     # smart | manual | off
  timeout: 300                    # seconds to wait for user response (default: 300)
  cron_mode: deny                 # deny | approve — what cron jobs do when they hit a dangerous command
  mcp_reload_confirm: true        # /reload-mcp asks before invalidating the MCP tool cache
  destructive_slash_confirm: true # /clear, /new, /reset, /undo prompt before discarding state
```

전체 키 목록:

| 키 | 기본값 | 제어하는 항목 |
|---|---|---|
| `mode` | `smart` | 위험한 셸 명령에 대한 승인 정책 — 아래 표 참조. |
| `timeout` | `300` | 승인 응답을 기다리는 시간(초). |
| `cron_mode` | `deny` | [cron 작업](./features/cron.md)이 위험한 명령 승인 프롬프트를 만났을 때 헤드리스로 동작하는 방식. `deny`는 명령을 차단하고(에이전트는 다른 경로를 찾아야 함), `approve`는 cron 컨텍스트의 모든 항목을 자동 승인합니다. |
| `mcp_reload_confirm` | `true` | `true`이면 `/reload-mcp`가 MCP 도구 세트를 다시 빌드하기 전에 확인합니다. 다시 빌드하면 공급자 프롬프트 캐시가 무효화되므로(도구 스키마가 시스템 프롬프트에 포함됨) 다음 메시지에서 전체 입력 토큰이 다시 전송됩니다. **항상 승인**을 누른 사용자는 이 키가 `false`로 바뀝니다. |
| `destructive_slash_confirm` | `true` | `true`이면 파괴적인 세션 슬래시 명령(`/clear`, `/new`, `/reset`, `/undo`)이 대화 상태를 폐기하기 전에 확인합니다. 세 가지 선택 대화상자(한 번 승인 / 항상 승인 / 취소)가 Telegram, Discord, Slack에서는 네이티브 예/아니요 버튼으로, 그 외에는 텍스트 대체 방식으로 전달됩니다. **항상 승인**을 누른 사용자는 이 키가 `false`로 바뀝니다. TUI도 `/clear`, `/new`, `/reset` 모달에 이 설정을 적용하며, `HERMES_TUI_NO_CONFIRM=1`은 설정값과 관계없이 해당 모달을 강제로 건너뜁니다. |

| 모드 | 동작 |
|------|----------|
| **smart** (기본값) | 보조 LLM을 사용해 위험을 평가합니다. 위험도가 낮은 명령(예: `python -c "print('hello')"`)은 해당 명령에 한해서 자동 승인됩니다. 실제로 위험한 명령은 자동 거부됩니다. 판단이 불확실하면 수동 프롬프트로 전달됩니다. |
| **manual** | 위험한 명령마다 항상 사용자에게 승인을 요청합니다. |
| **off** | 모든 승인 검사를 끕니다 — `--yolo`로 실행하는 것과 같습니다. 모든 명령이 프롬프트 없이 실행됩니다. |

:::warning
`approvals.mode: off`를 설정하면 모든 안전 프롬프트가 비활성화됩니다. 신뢰할 수 있는 환경(CI/CD, 컨테이너 등)에서만 사용하세요.
:::

### YOLO 모드

YOLO 모드는 현재 세션에서 **모든** 위험한 명령 승인 프롬프트를 우회합니다. 다음 세 가지 방법으로 활성화할 수 있습니다.

1. **CLI 플래그**: `hermes --yolo` 또는 `hermes chat --yolo`로 세션 시작
2. **슬래시 명령**: 세션 중 `/yolo`를 입력하여 켜기/끄기 전환
3. **환경 변수**: `HERMES_YOLO_MODE=1` 설정

`/yolo` 명령은 **토글**입니다 — 사용할 때마다 모드가 켜지거나 꺼집니다.

```
> /yolo
  ⚡ YOLO mode ON — all commands auto-approved. Use with caution.

> /yolo
  ⚠ YOLO mode OFF — dangerous commands will require approval.
```

YOLO 모드는 CLI 세션과 gateway 세션 모두에서 사용할 수 있습니다. 내부적으로는 모든 명령 실행 전에 확인하는 `HERMES_YOLO_MODE` 환경 변수를 설정합니다.

YOLO가 활성화되면 Hermes는 승인 프롬프트가 우회된다는 사실을 잊기 어렵도록 지속적인 시각 알림 두 가지를 표시합니다.

- YOLO가 이미 활성화된 상태에서 세션이 시작되면 세션 시작 시 빨간 배너 줄이 표시됩니다: `⚠ YOLO mode — all approval prompts bypassed`. YOLO가 꺼져 있으면 기본 배너가 복잡해지지 않도록 숨겨집니다.
- 상태 표시줄의 모든 너비 단계에 `⚠ YOLO` 조각이 표시되며, YOLO를 켜거나 끌 때 실시간으로 갱신됩니다(서식 있는 텍스트 렌더러와 일반 텍스트 대체 방식 모두).

:::danger
YOLO 모드는 세션의 **모든** 위험한 명령 안전 검사를 비활성화합니다 — 단, 하드라인 차단 목록(아래 참조)은 예외입니다. 생성되는 명령을 완전히 신뢰할 수 있을 때(예: 일회용 환경에서 충분히 테스트한 자동화 스크립트)만 사용하세요.
:::

파괴적인 세션 슬래시 명령(`/clear`, `/new` / `/reset`, `/undo`, `/quit --delete` — `/exit --delete`는 별칭)에서는 CLI도 실행 전에 확인을 요청합니다. [슬래시 명령 — 파괴적인 명령의 확인 프롬프트](../reference/slash-commands.md#confirmation-prompts-for-destructive-commands)를 참조하세요.

### 하드라인 차단 목록(항상 적용되는 최저선)

일부 명령은 너무 치명적이어서 — 되돌릴 수 없는 파일 시스템 삭제, fork bomb, 블록 장치에 대한 직접 쓰기 등 — Hermes는 다음 상황에서도 실행을 거부합니다.

- `--yolo` / `/yolo`가 켜진 경우
- `approvals.mode: off`인 경우
- 헤드리스 `approve` 모드로 실행되는 cron 작업
- 사용자가 명시적으로 “항상 허용”을 클릭한 경우

차단 목록은 `--yolo` 아래의 최저선입니다. 승인 계층이 명령을 확인하기도 전에 작동하며, 재정의 플래그도 없습니다. 현재 적용되는 패턴(전체 목록은 아님)은 다음과 같습니다(`tools/approval.py::UNRECOVERABLE_BLOCKLIST`와 동기화됨).

| 패턴 | 하드라인인 이유 |
|---|---|
| `rm -rf /` 및 명백한 변형 | 파일 시스템 루트를 삭제함 |
| `rm -rf --no-preserve-root /` | “정말 루트를 의미한다”는 명시적 변형 |
| `:(){ :\|:& };:` (bash fork bomb) | 재부팅할 때까지 호스트를 점유함 |
| 마운트된 루트 장치에 대한 `mkfs.*` | 실행 중인 시스템을 포맷함 |
| `dd if=/dev/zero of=/dev/sd*` | 물리 디스크를 0으로 덮어씀 |
| 루트 파일 시스템 최상위에서 신뢰할 수 없는 URL을 `sh`로 파이프 | 원격 코드 실행 공격 벡터가 너무 광범위하여 승인할 수 없음 |

차단 목록에 걸리면 도구 호출은 설명이 포함된 오류를 에이전트에 반환하며 아무것도 실행되지 않습니다. 합법적인 워크플로에 이러한 명령 중 하나가 필요한 경우(예를 들어 삭제 및 재설치 파이프라인의 운영자인 경우) 에이전트 외부에서 실행하세요.

### 사용자 정의 거부 규칙(`approvals.deny`)

하드라인 차단 목록은 고정되어 있으며 코드에 포함됩니다. `approvals.deny`는 그에 대응하는 사용자가 편집 가능한 설정으로, 일치하는 터미널 명령을 무조건 차단하는 glob 패턴 목록입니다 — **`--yolo`, `/yolo`, `approvals.mode: off`를 확인하기 전에** 적용됩니다. 이를 통해 “에이전트가 모든 작업을 하되, 특정 작업만은 어떤 경우에도 하지 않도록” yolo-with-exceptions를 실행할 수 있습니다.

```yaml
approvals:
  deny:
    - "git push --force*"
    - "*curl*|*sh*"
    - "dd if=* of=/dev/*"
```

세부 사항:

- 패턴은 `fnmatch`([Python 문서](https://docs.python.org/3/library/fnmatch.html)의 glob)이며 전체 명령 텍스트와 대소문자를 구분하지 않고 비교합니다. `git push --force*`는 `git push --force origin main`과 일치하지만 `git push origin main`과는 일치하지 않습니다.
- 위험 패턴 탐지기가 사용하는 것과 동일한 정규화/난독화 해제 명령 변형을 대상으로 비교하므로, 단순한 인용 트릭(`git pu""sh --force`)으로 규칙을 우회할 수 없습니다.
- **YAML 인용:** 패턴은 항상 인용하세요. `*`로 시작하는 값을 그대로 쓰면 YAML 별칭으로 해석되어 파싱에 실패하며, `{`, `!`, `: `에도 각각 고유한 YAML 의미가 있습니다. 셸과 비슷한 내용에는 작은따옴표가 가장 안전합니다.
- 거부 규칙은 호스트에 연결되는 백엔드(local, SSH, 호스트 마운트 Docker)에 적용됩니다. 격리된 컨테이너 백엔드는 항상 그래 왔듯이 보호 계층 전체를 건너뜁니다 — 실행되는 명령이 호스트에 접근할 수 없기 때문입니다.
- 거부된 명령은 에이전트에 재시도하거나 표현을 바꾸지 말라는 BLOCKED 오류를 반환합니다. 아무것도 실행되지 않습니다.

승인 설정의 나머지 부분과 마찬가지로 변경 사항은 즉시 적용됩니다(설정 캐시는 mtime을 기준으로 함) — 세션을 다시 시작할 필요가 없습니다.

:::note 위협 모델
거부 규칙은 위험 패턴 탐지기와 동일한 위협 모델인, 정직하지만 잘못된 에이전트에 대한 안전장치입니다. 의도적으로 적대적인 프로세스에 대한 샌드박스는 아닙니다 — 이를 위해서는 격리된 백엔드(Docker, Modal) 또는 송신이 제한된 환경을 사용하세요.
:::

### 승인 시간 초과

위험한 명령 프롬프트가 나타나면 사용자는 설정 가능한 시간 안에 응답해야 합니다. 시간 초과 내에 응답이 없으면 기본적으로 명령이 **거부**됩니다(fail-closed).

`~/.hermes/config.yaml`에서 시간 초과를 설정합니다.

```yaml
approvals:
  timeout: 300  # seconds (default: 300)
```

### 승인을 발생시키는 항목

다음 패턴은 승인 프롬프트를 발생시킵니다(`tools/approval.py`에 정의됨).

| 패턴 | 설명 |
|---------|-------------|
| `rm -r` / `rm --recursive` | 재귀 삭제 |
| `rm ... /` | 루트 경로 삭제 |
| `chmod 777/666` / `o+w` / `a+w` | 모든 사용자/기타 사용자에게 쓰기 가능한 권한 |
| 안전하지 않은 권한과 함께 사용하는 `chmod --recursive` | 재귀적인 모든 사용자/기타 사용자 쓰기 가능 권한(긴 플래그) |
| `chown -R root` / `chown --recursive root` | root로 재귀적 chown |
| `mkfs` | 파일 시스템 포맷 |
| `dd if=` | 디스크 복사 |
| `> /dev/sd` | 블록 장치 쓰기 |
| `DROP TABLE/DATABASE` | SQL DROP |
| `DELETE FROM`(WHERE 없음) | WHERE 없는 SQL DELETE |
| `TRUNCATE TABLE` | SQL TRUNCATE |
| `> /etc/` | 시스템 설정 덮어쓰기 |
| `systemctl stop/restart/disable/mask` | 시스템 서비스 중지/재시작/비활성화 |
| `kill -9 -1` | 모든 프로세스 종료 |
| `pkill -9` | 프로세스 강제 종료 |
| Fork bomb 패턴 | Fork bomb |
| `bash -c` / `sh -c` / `zsh -c` / `ksh -c` | `-c` 플래그를 통한 셸 명령 실행( `-lc` 같은 결합 플래그 포함) |
| `python -e` / `perl -e` / `ruby -e` / `node -c` | `-e`/`-c` 플래그를 통한 스크립트 실행 |
| `curl ... \| sh` / `wget ... \| sh` | 원격 콘텐츠를 셸로 파이프 |
| `bash <(curl ...)` / `sh <(wget ...)` | 프로세스 치환을 통한 원격 스크립트 실행 |
| `/etc/`, `~/.ssh/`, `~/.hermes/.env`로 `tee` | tee를 통한 민감한 파일 덮어쓰기 |
| `/etc/`, `~/.ssh/`, `~/.hermes/.env`로 `>` / `>>` | 리디렉션을 통한 덮어쓰기 |
| `xargs rm` | rm과 함께 사용하는 xargs |
| `find -exec rm` / `find -delete` | 파괴적 작업을 수행하는 find |
| `/etc/`로 `cp`/`mv`/`install` | 시스템 설정으로 파일 복사/이동 |
| `/etc/`에 대한 `sed -i` / `sed --in-place` | 시스템 설정 인플레이스 편집 |
| hermes/gateway에 대한 `pkill`/`killall` | 자체 종료 방지 |
| `&`/`disown`/`nohup`/`setsid`를 포함한 `gateway run` | 서비스 관리자 외부에서 gateway 시작 방지 |
| `docker stop/kill/restart`, `docker compose down/stop/kill/restart` | 컨테이너 수명 주기(전역 플래그와 `docker-compose`도 포착) |
| `docker -H`/`--host`/`--context`, `DOCKER_HOST=`/`DOCKER_CONTEXT=` | Docker 데몬 리디렉션 — 다른(대개 원격) 데몬을 대상으로 함 |
| `docker context use` | 이후 모든 Docker 명령의 기본 데몬 전환 |
| `podman --remote`/`-r`/`--url`/`--connection`/`--identity`, `CONTAINER_HOST=` | Podman 원격 데몬 리디렉션 |

:::info
**컨테이너 우회**: `docker`, `singularity`, `modal`, `daytona` 또는 `vercel_sandbox` 백엔드에서 실행할 때는 컨테이너 자체가 보안 경계이므로 위험한 명령 검사를 **건너뜁니다**. 컨테이너 안의 파괴적 명령은 호스트에 피해를 줄 수 없습니다.
:::

### 승인 흐름(CLI)

대화형 CLI에서 위험한 명령은 인라인 승인 프롬프트를 표시합니다.

```
  ⚠️  DANGEROUS COMMAND: recursive delete
      rm -rf /tmp/old-project

      [o]nce  |  [s]ession  |  [a]lways  |  [d]eny

      Choice [o/s/a/D]:
```

네 가지 선택지는 다음과 같습니다.

- **once** — 이 한 번의 실행만 허용
- **session** — 남은 세션 동안 이 패턴 허용
- **always** — 영구 허용 목록에 추가(`config.yaml`에 저장)
- **deny** (기본값) — 명령 차단

### 승인 흐름(Gateway/메시징)

메시징 플랫폼에서 에이전트는 위험한 명령의 세부 정보를 채팅으로 보내고 사용자의 답변을 기다립니다.

- **yes**, **y**, **approve**, **ok** 또는 **go**로 답하면 승인
- **no**, **n**, **deny** 또는 **cancel**로 답하면 거부

gateway 실행 시 `HERMES_EXEC_ASK=1` 환경 변수가 자동으로 설정됩니다.

### 영구 허용 목록

“always”로 승인한 명령은 `~/.hermes/config.yaml`에 저장됩니다.

```yaml
# Permanently allowed dangerous command patterns
command_allowlist:
  - rm
  - systemctl
```

이 패턴은 시작 시 로드되며 이후 모든 세션에서 자동으로 승인됩니다.

:::tip
영구 허용 목록의 패턴을 검토하거나 제거하려면 `hermes config edit`를 사용하세요.
:::

### 승인 기록 마이닝(`hermes approvals suggest`)

세션마다 같은 프롬프트에 답하는 대신, 과거 승인 결정을 허용 목록 제안으로 마이닝할 수 있습니다.

```bash
hermes approvals suggest            # dry run — prints a numbered proposal
hermes approvals suggest --apply 1,3  # merge picks into command_allowlist
hermes approvals suggest --json     # machine-readable output
```

이 명령은 세션 데이터베이스(`~/.hermes/state.db`)에서 실제로 실행된 위험 분류 명령 — 즉 사용자가 승인한 명령 — 을 검색하고, 이를 패턴(`git push *` 또는 복합 명령의 위험 클래스 키)으로 집계한 뒤 승인 빈도순으로 정렬합니다.

```
Proposed command_allowlist additions (from approval history, last 90 days):

  1. git push *    — approved 14x
  2. docker restart/stop/kill (container lifecycle)    — approved 9x (class key)
```

안전 규칙:

- **자동으로 적용되는 것은 없습니다** — 기본 실행은 읽기 전용이며, 명시적인 `--apply N[,M...]`만 `config.yaml`에 기록합니다.
- **파괴적 클래스는 아무리 자주 승인되었어도 제안되지 않습니다**: 재귀 삭제, `sudo`, 디스크/장치 쓰기, 자격 증명 및 시스템 설정 편집, 셸로 파이프, SQL DROP/TRUNCATE, 프로세스 종료 및 모든 하드라인 클래스가 완전히 제외됩니다. `rm -rf build/`를 100번 승인해도 `rm` 항목이 생성되지 않습니다.
- 기존 `command_allowlist`에 이미 포함된 제안은 건너뜁니다.

유용한 플래그: `--days N`(기록 기간, 기본값 90), `--min-count N`(자격을 얻기 위한 최소 승인 횟수, 기본값 2), `--limit N`, `--db PATH`.

## 파일 쓰기 안전성 {#file-write-safety}

`write_file` 또는 `patch`가 디스크를 수정하기 전에 Hermes는 대상 경로를 거부 목록 및 선택적 샌드박스와 대조합니다. 차단된 쓰기는 즉시 에이전트에 오류를 반환합니다 — **승인 프롬프트가 없으며** 채팅 UI에서 우회할 방법도 없습니다. 모델은 편집이 성공했다고 주장할 수 있지만, `display.file_mutation_verifier`가 켜져 있을 때(기본값) 어시스턴트의 마무리 요약보다 [파일 변경 검증기 푸터](./configuration.md#file-mutation-verifier)를 신뢰하세요.

### 보호 경로(항상 차단)

다음 범주는 `HERMES_WRITE_SAFE_ROOT`가 설정되지 않은 경우에도 항상 거부됩니다.

| 범주 | 예시 |
|----------|----------|
| OS 자격 증명 저장소 | `~/.ssh/`(키, `authorized_keys`), `~/.aws/`, `~/.kube/`, `/etc/sudoers`, `~/.netrc` |
| Hermes 자격 증명 저장소 | HERMES_HOME(활성 프로필 및 전역 루트) 아래의 `auth.json`, `.env`, `.anthropic_oauth.json`, `mcp-tokens/`, `pairing/` |
| 프로젝트 비밀 파일 | 디스크 어디에나 있는 `.env`, `.env.local`, `.env.production`, `.envrc` |

안전 루트 내부의 민감한 경로도 여전히 차단됩니다 — `HERMES_WRITE_SAFE_ROOT`를 `$HOME`으로 지정해도 `~/.ssh/id_rsa`에 쓸 수 없습니다.

안전 루트 위반은 `Write denied: '…' is outside HERMES_WRITE_SAFE_ROOT (…)`를 반환합니다. 자격 증명 경로 차단은 `Write denied: '…' is a protected system/credential file.`을 사용합니다.

**예외 — `~/.ssh/config`는 하드 차단이 아니라 승인 대상입니다.** SSH *클라이언트 설정*에는 개인 키 자료가 없고 편집(호스트 별칭, `ProxyJump`, VS Code Remote-SSH 대상)은 일반적인 작업이므로, `write_file`/`patch`는 `~/.ssh` 쓰기에 이미 사용되는 동일한 한 번/세션/항상 승인 프롬프트로 전달됩니다 — 기존에 적용되던 일괄 거부 대신 사용합니다. 이 파일에는 명령을 실행하는 `ProxyCommand`/`Match exec` 지시어가 포함될 수 있으므로 쓰기는 결코 묵묵히 처리되지 않습니다. 비대화형 호출자(ACP 파일 브리지, 사람이 연결되지 않은 백그라운드 작업)는 fail-closed로 종료됩니다. 개인 키, `authorized_keys` 및 `~/.ssh/` 아래의 나머지 모든 항목은 계속 하드 차단됩니다.

### HERMES_WRITE_SAFE_ROOT(선택적 샌드박스)

설정하면 `write_file`과 `patch`는 나열된 디렉터리 접두사 내부의 경로만 대상으로 삼을 수 있습니다. 외부의 모든 항목은 위험한 명령 승인으로 전달되지 않고 **하드 차단**됩니다.

- [공식 Docker 이미지](https://github.com/NousResearch/hermes-agent)에서 자동 설정됨(`HERMES_WRITE_SAFE_ROOT=/opt/data`)
- Unix에서는 `:`, Windows에서는 `;`로 구분한 여러 루트 지원
- **`~/.hermes/.env`에 아무 생각 없이 추가하지 마세요.** 프로젝트 디렉터리로 설정하면 에이전트는 `~/.hermes/cron/jobs.json`, 프로필 스킬 또는 접두사 밖의 다른 Hermes 상태에 쓸 수 없습니다

워크스페이스와 Hermes 홈을 모두 허용하려면 다음과 같이 합니다.

```bash
export HERMES_WRITE_SAFE_ROOT=/path/to/project:/home/you/.hermes
```

변수를 해제하면 보호 경로 거부 목록의 적용을 받는 범위에서 제한 없는 쓰기로 돌아갑니다. 전체 참조: [HERMES_WRITE_SAFE_ROOT](../reference/environment-variables.md#hermes_write_safe_root).

### Cron 및 기타 Hermes 상태

에이전트에게 `~/.hermes/cron/jobs.json`을 직접 `patch`하도록 요청하지 마세요. `cronjob` 도구, [`hermes cron`](./features/cron.md) 또는 `/cron`을 사용하세요 — 지원되는 API를 통해 작업 저장소를 갱신합니다. 쓰기 안전성이 직접 편집을 차단하는 다른 Hermes 제어 파일에도 동일하게 적용됩니다.

:::note 심층 방어이지 하드 경계가 아님
쓰기 가드는 `write_file`과 `patch`에만 적용됩니다. `terminal` 도구는 동일한 OS 사용자로 실행되므로 셸 명령을 통해 차단된 경로를 `cat`하거나 덮어쓸 수 있습니다. 거부 목록은 우발적 피해를 줄이고 모델에 명확한 중단 신호를 제공하지만, 적대적이거나 손상된 에이전트를 샌드박싱하지는 않습니다.
:::

## 사용자 인증(Gateway)

메시징 gateway를 실행할 때 Hermes는 계층형 인증 시스템을 통해 봇과 상호 작용할 수 있는 사람을 제어합니다.

### 인증 검사 순서

`_is_user_authorized()` 메서드는 다음 순서로 확인합니다.

1. **플랫폼별 모두 허용 플래그**(예: `DISCORD_ALLOW_ALL_USERS=true`)
2. **DM 페어링 승인 목록**(페어링 코드로 승인된 사용자)
3. **플랫폼별 허용 목록**(예: `TELEGRAM_ALLOWED_USERS=12345,67890`)
4. **전역 허용 목록**(`GATEWAY_ALLOWED_USERS=12345,67890`)
5. **전역 모두 허용**(`GATEWAY_ALLOW_ALL_USERS=true`)
6. **기본값: 거부**

### 플랫폼 허용 목록

허용된 사용자 ID를 `~/.hermes/.env`에 쉼표로 구분한 값으로 설정합니다.

```bash
# Platform-specific allowlists
TELEGRAM_ALLOWED_USERS=123456789,987654321
DISCORD_ALLOWED_USERS=111222333444555666
WHATSAPP_ALLOWED_USERS=15551234567
SLACK_ALLOWED_USERS=U01ABC123

# Cross-platform allowlist (checked for all platforms)
GATEWAY_ALLOWED_USERS=123456789

# Per-platform allow-all (use with caution)
DISCORD_ALLOW_ALL_USERS=true

# Global allow-all (use with extreme caution)
GATEWAY_ALLOW_ALL_USERS=true
```

:::warning
**허용 목록이 하나도 설정되지 않았고** `GATEWAY_ALLOW_ALL_USERS`도 설정되지 않으면 **모든 사용자가 거부**됩니다. gateway는 시작 시 다음 경고를 기록합니다.

```
No user allowlists configured. All unauthorized users will be denied.
Set GATEWAY_ALLOW_ALL_USERS=true in ~/.hermes/.env to allow open access,
or configure platform allowlists (e.g., TELEGRAM_ALLOWED_USERS=your_id).
```
:::

### DM 페어링 시스템

더 유연한 인증을 위해 Hermes에는 코드 기반 페어링 시스템이 포함되어 있습니다. 사용자 ID를 미리 요구하는 대신, 알 수 없는 사용자는 일회성 페어링 코드를 받고 봇 소유자가 CLI를 통해 승인합니다.

**작동 방식:**

1. 알 수 없는 사용자가 봇에 DM을 보냅니다
2. 봇이 8자 페어링 코드로 답합니다
3. 봇 소유자가 CLI에서 `hermes pairing approve <platform> <code>`를 실행합니다
4. 해당 사용자가 그 플랫폼에서 영구적으로 승인됩니다

`~/.hermes/config.yaml`에서 인증되지 않은 DM의 처리 방식을 제어합니다.

```yaml
unauthorized_dm_behavior: pair

whatsapp:
  unauthorized_dm_behavior: ignore
```

- `pair`는 채팅 형태의 DM 플랫폼에서 기본값입니다. 인증되지 않은 DM에는 페어링 코드로 답합니다.
- `ignore`는 인증되지 않은 DM을 조용히 버립니다.
- 이메일은 관련 없는 읽지 않은 메일이 포함될 수 있으므로 `platforms.email.unauthorized_dm_behavior: pair`가 설정되지 않는 한 기본값이 `ignore`입니다.
- 플랫폼 섹션은 전역 기본값보다 우선하므로 Telegram에서는 페어링을 유지하면서 WhatsApp에서는 조용히 처리할 수 있습니다.

**보안 기능**(OWASP + NIST SP 800-63-4 지침 기반):

| 기능 | 세부 사항 |
|---------|---------|
| 코드 형식 | 32자 모호하지 않은 알파벳에서 8자(0/O/1/I 제외) |
| 무작위성 | 암호학적 방식(`secrets.choice()`) |
| 코드 TTL | 1시간 후 만료 |
| 속도 제한 | 사용자당 10분에 1회 요청 |
| 대기 한도 | 플랫폼당 최대 3개의 대기 중 코드 |
| 잠금 | 승인 실패 5회 → 1시간 잠금 |
| 파일 보안 | 모든 페어링 데이터 파일에 `chmod 0600` |
| 로깅 | 코드는 stdout에 기록되지 않음 |

**페어링 CLI 명령:**

```bash
# List pending and approved users
hermes pairing list

# Approve a pairing code
hermes pairing approve telegram ABC12DEF

# Revoke a user's access
hermes pairing revoke telegram 123456789

# Clear all pending codes
hermes pairing clear-pending
```

:::tip Docker 사용자: `hermes` 사용자로 페어링 명령을 실행하세요
공식 Docker 이미지는 `gosu`를 통해 권한이 없는 `hermes` 사용자(uid 10000)로 gateway를 실행하지만, `docker exec`는 기본적으로 root로 실행됩니다. root가 만든 승인 파일은 `0600 root:root` 모드로 기록되어 gateway가 읽을 수 없으므로 승인이 조용히 무시됩니다([#10270][i10270]).

항상 `-u hermes`를 전달하세요.

```bash
docker exec -u hermes hermes-agent hermes pairing approve telegram ABC12DEF
```

이미 root로 명령을 실행했고 사용자가 여전히 인증되지 않는다면 컨테이너를 다시 시작하세요 — 다음 시작 시 entrypoint가 소유권을 수정합니다.

[i10270]: https://github.com/NousResearch/hermes-agent/issues/10270
:::

**저장:** 페어링 데이터는 플랫폼별 JSON 파일과 함께 `~/.hermes/pairing/`에 저장됩니다.
- `{platform}-pending.json` — 대기 중인 페어링 요청
- `{platform}-approved.json` — 승인된 사용자
- `_rate_limits.json` — 속도 제한 및 잠금 추적

## 컨테이너 격리

`docker` 터미널 백엔드를 사용하면 Hermes는 모든 컨테이너에 엄격한 보안 강화를 적용합니다.

### Docker 보안 플래그

모든 컨테이너는 다음 플래그로 실행됩니다(`tools/environments/docker.py`에 정의됨).

```python
_BASE_SECURITY_ARGS = [
    "--cap-drop", "ALL",                          # Drop ALL Linux capabilities
    "--cap-add", "DAC_OVERRIDE",                  # Root can write to bind-mounted dirs
    "--cap-add", "CHOWN",                         # Package managers need file ownership
    "--cap-add", "FOWNER",                        # Package managers need file ownership
    "--security-opt", "no-new-privileges",         # Block privilege escalation
    "--pids-limit", "256",                         # Limit process count
    "--tmpfs", "/tmp:rw,nosuid,size=512m",         # Size-limited /tmp
    "--tmpfs", "/var/tmp:rw,noexec,nosuid,size=256m",  # No-exec /var/tmp
]
```

`SETUID`/`SETGID`는 기본 목록에 없습니다 — 컨테이너가 root로 시작하고 초기화/entrypoint가 권한을 내려야 할 때 조건부로 추가됩니다(s6 권한 삭제 경로). 이미 `--user`로 비root 실행 중이면 건너뜁니다. `/run` tmpfs도 기본 목록에서 분리되어 이미지별로 마운트되며, 기본적으로 `noexec`로 강화되고 `/run`에서 실행하는 s6-overlay 이미지에만 `exec`가 적용됩니다.

### 리소스 제한

컨테이너 리소스는 `~/.hermes/config.yaml`에서 설정할 수 있습니다.

```yaml
terminal:
  backend: docker
  docker_image: "nikolaik/python-nodejs:python3.11-nodejs20"
  docker_forward_env: []  # Explicit allowlist only; empty keeps secrets out of the container
  container_cpu: 1        # CPU cores
  container_memory: 5120  # MB (default 5GB)
  container_disk: 51200   # MB (default 50GB, requires overlay2 on XFS)
  container_persistent: true  # Persist filesystem across sessions
```

### 파일 시스템 지속성

- **지속 모드**(`container_persistent: true`): `~/.hermes/sandboxes/docker/<task_id>/`에서 `/workspace`와 `/root`를 바인드 마운트합니다
- **임시 모드**(`container_persistent: false`): 워크스페이스에 tmpfs를 사용하며 정리할 때 모든 것이 사라집니다

:::tip
프로덕션 gateway 배포에서는 `docker`, `modal`, `daytona` 또는 `vercel_sandbox` 백엔드를 사용하여 에이전트 명령을 호스트 시스템에서 격리하세요. 이렇게 하면 위험한 명령 승인이 전혀 필요하지 않습니다.
:::

:::warning
`terminal.docker_forward_env`에 이름을 추가하면 해당 변수가 터미널 명령을 위해 컨테이너에 의도적으로 주입됩니다. `GITHUB_TOKEN` 같은 작업별 자격 증명에는 유용하지만, 컨테이너에서 실행되는 코드가 이를 읽고 외부로 유출할 수도 있습니다.
:::

## 터미널 백엔드 보안 비교

| 백엔드 | 격리 | 위험한 명령 검사 | 적합한 용도 |
|---------|-----------|-------------------|----------|
| **local** | 없음 — 호스트에서 실행 | ✅ 수행 | 개발, 신뢰할 수 있는 사용자 |
| **ssh** | 원격 시스템 | ✅ 수행 | 별도 서버에서 실행 |
| **docker** | 컨테이너 | ❌ 건너뜀(컨테이너가 경계) | 프로덕션 gateway |
| **singularity** | 컨테이너 | ❌ 건너뜀 | HPC 환경 |
| **modal** | 클라우드 샌드박스 | ❌ 건너뜀 | 확장 가능한 클라우드 격리 |
| **daytona** | 클라우드 샌드박스 | ❌ 건너뜀 | 지속형 클라우드 워크스페이스 |
| **vercel_sandbox** | 클라우드 microVM | ❌ 건너뜀 | 스냅샷 지속성이 있는 클라우드 실행 |

## 환경 변수 전달 {#environment-variable-passthrough}

`execute_code`와 `terminal`은 모두 자격 증명 유출을 막기 위해 하위 프로세스에서 민감한 환경 변수를 제거합니다. 그러나 `required_environment_variables`를 선언하는 스킬에는 해당 변수에 대한 접근이 합법적으로 필요합니다.

### 작동 방식

두 가지 메커니즘을 통해 특정 변수가 샌드박스 필터를 통과할 수 있습니다.

**1. 스킬 범위 전달(자동)**

스킬이 로드되고(`skill_view` 또는 `/skill` 명령을 통해) `required_environment_variables`를 선언하면, 환경에 실제로 설정된 해당 변수는 자동으로 전달 대상으로 등록됩니다. 없는 변수(아직 설정이 필요한 상태)는 등록되지 않습니다.

```yaml
# In a skill's SKILL.md frontmatter
required_environment_variables:
  - name: TENOR_API_KEY
    prompt: Tenor API key
    help: Get a key from https://developers.google.com/tenor
```

이 스킬을 로드하면 `TENOR_API_KEY`는 `execute_code`, `terminal`(local), **원격 백엔드(Docker, Modal)**로 전달됩니다 — 수동 설정이 필요하지 않습니다.

:::info Docker 및 Modal
v0.5.1 이전에는 Docker의 `forward_env`가 스킬 전달과 별도의 시스템이었습니다. 이제 둘이 통합되어 스킬이 선언한 환경 변수가 `docker_forward_env`에 수동으로 추가하지 않아도 Docker 컨테이너와 Modal 샌드박스로 자동 전달됩니다.
:::

**2. 설정 기반 전달(수동)**

어떤 스킬에도 선언되지 않은 환경 변수는 `config.yaml`의 `terminal.env_passthrough`에 추가합니다.

```yaml
terminal:
  env_passthrough:
    - MY_CUSTOM_KEY
    - ANOTHER_TOKEN
```

### 자격 증명 파일 전달(OAuth 토큰 등) {#credential-file-passthrough}

일부 스킬은 샌드박스에서 환경 변수뿐 아니라 **파일**도 필요로 합니다 — 예를 들어 Google Workspace는 활성 프로필의 `HERMES_HOME` 아래에 `google_token.json`으로 OAuth 토큰을 저장합니다. 스킬은 이를 frontmatter에 선언합니다.

```yaml
required_credential_files:
  - path: google_token.json
    description: Google OAuth2 token (created by setup script)
  - path: google_client_secret.json
    description: Google OAuth2 client credentials
```

로드되면 Hermes는 활성 프로필의 `HERMES_HOME`에 이러한 파일이 존재하는지 확인하고 마운트 대상으로 등록합니다.

- **Docker**: 읽기 전용 바인드 마운트(`-v host:container:ro`)
- **Modal**: 샌드박스 생성 시 마운트 + 각 명령 전에 동기화(세션 중간의 OAuth 설정 처리)
- **Local**: 별도 작업 없음(파일에 이미 접근 가능)

`config.yaml`에 자격 증명 파일을 수동으로 나열할 수도 있습니다.

```yaml
terminal:
  credential_files:
    - google_token.json
    - my_custom_oauth_token.json
```

경로는 `~/.hermes/`를 기준으로 상대 경로입니다. 파일은 컨테이너 내부의 `/root/.hermes/`에 마운트됩니다. 이 목록은 `tools/credential_files.py`(`terminal.credential_files`)에서 읽습니다 — `terminal:` 블록 아래에 있지만 핵심 터미널 백엔드가 아니라 자격 증명 파일 모듈에서 로드되므로 번들된 `DEFAULT_CONFIG` 스냅샷에는 포함되지 않습니다.

### 각 샌드박스가 필터링하는 항목

| 샌드박스 | 기본 필터 | 전달 재정의 |
|---------|---------------|---------------------|
| **execute_code** | 이름에 `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, `CREDENTIAL`, `PASSWD`, `AUTH`가 포함된 변수를 차단하고 안전한 접두사 변수만 통과 | ✅ 전달 변수는 두 검사를 모두 우회 |
| **terminal** (local) | 명시적인 Hermes 인프라 변수(공급자 키, gateway 토큰, 도구 API 키)를 차단 | ✅ 전달 변수가 차단 목록을 우회 |
| **terminal** (Docker) | 기본적으로 호스트 환경 변수 없음 | ✅ 전달 변수 + `docker_forward_env`가 `-e`를 통해 전달 |
| **terminal** (Modal) | 기본적으로 호스트 환경/파일 없음 | ✅ 자격 증명 파일 마운트; 동기화를 통한 환경 변수 전달 |
| **MCP** | 안전한 시스템 변수 + 명시적으로 설정된 `env`를 제외한 모든 항목 차단 | ❌ 전달의 영향을 받지 않음(MCP `env` 설정을 사용) |

### 보안 고려 사항

- 전달은 사용자나 스킬이 명시적으로 선언한 변수에만 영향을 줍니다 — 임의로 생성된 LLM 코드에 대한 기본 보안 상태는 바뀌지 않습니다
- 자격 증명 파일은 Docker 컨테이너에 **읽기 전용**으로 마운트됩니다
- Skills Guard는 설치 전에 스킬 콘텐츠에서 의심스러운 환경 접근 패턴을 검사합니다
- 없거나 설정되지 않은 변수는 절대 등록되지 않습니다(존재하지 않는 것은 유출할 수 없음)
- Hermes 인프라 비밀(공급자 API 키, gateway 토큰)은 `env_passthrough`에 추가하지 않아야 합니다 — 전용 메커니즘이 있습니다

## MCP 자격 증명 처리

MCP(Model Context Protocol) 서버 하위 프로세스는 우발적인 자격 증명 유출을 막기 위해 **필터링된 환경**을 받습니다.

### 안전한 환경 변수

호스트에서 MCP stdio 하위 프로세스로 전달되는 변수는 다음뿐입니다.

```
PATH, HOME, USER, LANG, LC_ALL, TERM, SHELL, TMPDIR
```

여기에 모든 `XDG_*` 변수도 추가됩니다. 그 외 환경 변수(API 키, 토큰, 비밀)는 모두 **제거**됩니다.

MCP 서버의 `env` 설정에 명시적으로 정의된 변수는 전달됩니다.

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_..."  # Only this is passed
```

### 자격 증명 삭제

MCP 도구의 오류 메시지는 LLM에 반환되기 전에 정제됩니다. 다음 패턴은 `[REDACTED]`로 바뀝니다.

- GitHub PAT(`ghp_...`)
- OpenAI 형식 키(`sk-...`)
- Bearer 토큰
- `token=`, `key=`, `API_KEY=`, `password=`, `secret=` 매개변수

### 웹사이트 접근 정책

웹 및 브라우저 도구를 통해 에이전트가 접근할 수 있는 웹사이트를 제한할 수 있습니다. 에이전트가 내부 서비스, 관리자 패널 또는 기타 민감한 URL에 접근하지 못하게 할 때 유용합니다.

```yaml
# In ~/.hermes/config.yaml
security:
  website_blocklist:
    enabled: true
    domains:
      - "*.internal.company.com"
      - "admin.example.com"
    shared_files:
      - "/etc/hermes/blocked-sites.txt"
```

차단된 URL을 요청하면 도구가 정책상 도메인이 차단되었다는 오류를 반환합니다. 차단 목록은 `web_search`, `web_extract`, `browser_navigate` 및 URL을 사용할 수 있는 모든 도구에 적용됩니다.

자세한 내용은 설정 가이드의 [웹사이트 차단 목록](/user-guide/configuration#website-blocklist)을 참조하세요.

### SSRF 보호

모든 URL 사용 가능 도구(웹 검색, 웹 추출, 비전, 브라우저)는 Server-Side Request Forgery(SSRF) 공격을 막기 위해 가져오기 전에 URL을 검증합니다. 차단되는 주소는 다음과 같습니다.

- **사설 네트워크**(RFC 1918): `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`
- **루프백**: `127.0.0.0/8`, `::1`
- **링크 로컬**: `169.254.0.0/16`(클라우드 메타데이터 `169.254.169.254` 포함)
- **CGNAT / 공유 주소 공간**(RFC 6598): `100.64.0.0/10`(Tailscale, WireGuard VPN)
- **클라우드 메타데이터 호스트 이름**: `metadata.google.internal`, `metadata.goog`
- **예약, 멀티캐스트 및 미지정 주소**

SSRF 보호는 인터넷에 노출된 사용에서 항상 활성화되며 DNS 실패는 차단된 것으로 처리됩니다(fail-closed). 리디렉션 기반 우회를 막기 위해 리디렉션 체인의 각 홉을 다시 검증합니다.

#### 사설 URL을 의도적으로 허용하기

일부 설정에서는 사설/내부 URL에 합법적으로 접근해야 합니다 — `home.arpa`를 RFC 1918 공간으로 해석하는 홈 네트워크, LAN 전용 Ollama/llama.cpp 엔드포인트, 내부 위키, 클라우드 메타데이터 디버깅 등이 이에 해당합니다. 이런 경우에는 전역 옵트아웃을 사용할 수 있습니다.

```yaml
security:
  allow_private_urls: true   # default: false
```

활성화하면 웹 도구, 브라우저, 비전 URL 가져오기 및 gateway 미디어 다운로드는 더 이상 RFC 1918/루프백/링크 로컬/CGNAT/클라우드 메타데이터 대상을 거부하지 않습니다. **이는 의도적인 신뢰 경계입니다** — 에이전트가 프롬프트 인젝션을 통해 로컬 네트워크를 대상으로 임의 URL을 실행하는 위험을 감수할 수 있는 시스템에서만 활성화하세요. 공개 gateway는 이 설정을 꺼 두어야 합니다.

기반 IP가 공개 주소인 경우에도 유사한 Unicode 도메인 트릭을 차단하는 호스트 부분 문자열 가드는 이 설정과 관계없이 계속 활성화됩니다.

### Tirith 사전 실행 보안 검사

Hermes는 실행 전에 콘텐츠 수준의 명령 검사를 위해 [tirith](https://github.com/sheeki03/tirith)를 통합합니다. Tirith는 패턴 매칭만으로는 놓치는 위협을 탐지합니다.

- 동형 문자 URL 스푸핑(국제화 도메인 공격)
- 인터프리터로 파이프하는 패턴(`curl | bash`, `wget | sh`)
- 터미널 인젝션 공격

Tirith는 처음 사용할 때 GitHub 릴리스에서 SHA-256 체크섬 검증(및 cosign을 사용할 수 있으면 provenance 검증)과 함께 자동 설치됩니다.

```yaml
# In ~/.hermes/config.yaml
security:
  tirith_enabled: true       # Enable/disable tirith scanning (default: true)
  tirith_path: "tirith"      # Path to tirith binary (default: PATH lookup)
  tirith_timeout: 5          # Subprocess timeout in seconds
  tirith_fail_open: true     # Allow execution when tirith is unavailable (default: true)
```

`tirith_fail_open`이 `true`(기본값)이면 tirith가 설치되지 않았거나 시간 초과가 발생해도 명령이 진행됩니다. 보안 수준이 높은 환경에서는 `false`로 설정하여 tirith를 사용할 수 없을 때 명령을 차단하세요.

Tirith는 Linux(x86_64 / aarch64)와 macOS(x86_64 / arm64)에 미리 빌드된 바이너리를 제공합니다. 미리 빌드된 바이너리가 없는 플랫폼(Windows 등)에서는 tirith를 조용히 건너뜁니다 — 패턴 매칭 가드는 계속 실행되며 CLI에는 “사용할 수 없음” 배너가 표시되지 않습니다. Windows에서 tirith를 사용하려면 WSL에서 Hermes를 실행하세요.

Tirith의 판정은 승인 흐름과 통합됩니다. 안전한 명령은 통과하고, 의심스럽거나 차단된 명령은 전체 tirith 결과(심각도, 제목, 설명, 더 안전한 대안)와 함께 사용자 승인을 요청합니다. 사용자는 승인하거나 거부할 수 있으며, 무인 시나리오를 안전하게 유지하기 위해 기본 선택은 거부입니다.

### 컨텍스트 파일 인젝션 보호

컨텍스트 파일(AGENTS.md, .cursorrules, SOUL.md)은 시스템 프롬프트에 포함되기 전에 프롬프트 인젝션 검사를 받습니다. 검사 항목은 다음과 같습니다.

- 이전 지시를 무시/무시하라는 지시
- 의심스러운 키워드가 포함된 숨겨진 HTML 주석
- 비밀 읽기 시도(`.env`, `credentials`, `.netrc`)
- `curl`을 통한 자격 증명 유출
- 보이지 않는 Unicode 문자(제로 너비 공백, 양방향 재정의)

차단된 파일은 다음과 같은 경고를 표시합니다.

```
[BLOCKED: AGENTS.md contained potential prompt injection (prompt_injection). Content not loaded.]
```

## 프로덕션 배포 모범 사례

### Gateway 배포 체크리스트

1. **명시적인 허용 목록 설정** — 프로덕션에서는 `GATEWAY_ALLOW_ALL_USERS=true`를 절대 사용하지 않기
2. **컨테이너 백엔드 사용** — config.yaml에서 `terminal.backend: docker` 설정
3. **리소스 제한** — 적절한 CPU, 메모리 및 디스크 제한 설정
4. **비밀을 안전하게 저장** — 적절한 파일 권한과 함께 API 키를 `~/.hermes/.env`에 보관
5. **DM 페어링 활성화** — 가능하면 사용자 ID를 하드코딩하는 대신 페어링 코드 사용
6. **명령 허용 목록 검토** — config.yaml의 `command_allowlist`를 주기적으로 감사
7. **`terminal.cwd` 설정** — 에이전트가 민감한 디렉터리에서 작동하지 않도록 하기
8. **비root로 실행** — gateway를 root로 절대 실행하지 않기
9. **로그 모니터링** — 인증되지 않은 접근 시도를 `~/.hermes/logs/`에서 확인
10. **최신 상태 유지** — 보안 패치를 위해 `hermes update`를 정기적으로 실행

### API 키 보호

```bash
# Set proper permissions on the .env file
chmod 600 ~/.hermes/.env

# Keep separate keys for different services
# Never commit .env files to version control
```

### 네트워크 격리

최대 보안을 위해 별도 시스템이나 VM에서 gateway를 실행하세요. `config.yaml`에서 `terminal.backend: ssh`를 설정한 뒤 `~/.hermes/.env`의 환경 변수를 통해 호스트 세부 정보를 제공합니다.

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

SSH 연결 정보는 `.env`에 저장되며(`config.yaml`이 아님), 프로필 내보내기와 함께 커밋되거나 공유되지 않습니다. 이로써 gateway의 메시징 연결과 에이전트의 명령 실행이 분리됩니다.

## 공급망 권고 확인

Hermes에는 활성 venv의 Python 패키지 중 알려진 손상 버전(2026년 5월 `mistralai 2.4.6` 중독과 같은 공급망 웜)과 일치하는 항목을 선별된 카탈로그로 표시하는 내장 권고 스캐너가 포함되어 있습니다. 구현은 `hermes_cli/security_advisories.py`에 있습니다.

실행 방식:

- **CLI 시작 배너.** 권고 사항이 일치하면 전체 해결 방법을 확인할 수 있도록 `hermes doctor`를 안내하는 한 줄 경고가 출력됩니다.
- **`hermes doctor`.** 버전 세부 정보와 2~4단계 해결 지침을 포함해 모든 활성 권고를 표시합니다.
- **Gateway 시작.** `gateway.log`에 기록되며, 첫 번째 대화형 메시지에 짧은 운영자 배너가 표시됩니다.

각 권고에는 안정적인 ID가 있습니다. 읽고 조치한 뒤에는 다음 명령으로 영구적으로 숨길 수 있습니다.

```bash
hermes doctor --ack <advisory-id>
```

ack는 `config.security.acked_advisories`에 저장되며 재시작 후에도 유지됩니다. 오래된 권고는 카탈로그에서 의도적으로 제거되지 않습니다 — 그대로 두면 사설 미러에 아직 캐시되어 있을 수 있는 과거에 오염된 버전에 대해 새 설치도 경고를 받을 수 있습니다.

검사 자체는 표준 라이브러리만 사용하며 권고당 `importlib.metadata.version()` 조회 한 번으로 수행되므로 시작할 때마다 실행해도 안전합니다.

### 선택적 의존성 지연 설치

많은 기능(Mistral TTS, ElevenLabs, Honcho memory, Bedrock, Slack, Matrix 등)은 모든 사용자가 필요로 하지는 않는 Python 패키지에 의존합니다. Hermes는 `hermes-agent[all]` 아래에 미리 설치하는 대신 처음 사용할 때 이러한 패키지를 **지연 설치**합니다. 구현은 `tools/lazy_deps.py`에 있습니다.

이 방식이 해결하는 절충점:

- **취약성.** 추가 기능 하나의 전이 의존성을 PyPI에서 사용할 수 없게 되면(악성 코드로 격리, yanked, 업로드 손상) 전체 `[all]` 해결이 실패하고 새 설치가 조용히 축소된 계층으로 대체됩니다 — 관련 없는 추가 기능 10개 이상을 한 번에 잃습니다. 지연 설치는 각 백엔드를 격리하므로 오염된 의존성 하나가 관련 없는 기능을 깨뜨리지 않습니다.
- **비대함.** 한 공급자만 사용하는 사용자는 가져오지도 않을 수백 개의 패키지를 더 이상 설치하지 않습니다.

작동 방식:

1. 백엔드 모듈이 최초 import 경로의 시작 부분에서 `ensure("feature.name")`을 호출합니다.
2. 의존성이 없으면 `ensure`가 `config.yaml`의 `security.allow_lazy_installs`(기본값 `true`)를 확인하고 허용 목록에 있는 사양에 대해 venv 범위의 `pip install`을 실행합니다.
3. 설치가 실패하거나 사용자가 지연 설치를 비활성화했으면 실제 pip stderr와 `hermes tools` 안내가 포함된 `FeatureUnavailable`을 발생시킵니다.

`tools/lazy_deps.py`가 적용하는 보안 보장:

| 보장 | 의미 |
|---|---|
| venv 범위만 | 활성 venv의 `sys.executable`을 대상으로 설치 — 시스템 Python은 절대 사용하지 않음 |
| 이름으로만 PyPI | 사양은 `"package>=1.0,<2"` 문법을 허용함. `--index-url`, `git+https://` 또는 file: 경로는 허용하지 않음 — 악성 `config.yaml`이 설치를 다른 곳으로 리디렉션할 수 없음 |
| 허용 목록 | 이 경로로 설치할 수 있는 것은 트리 내부 `LAZY_DEPS` 맵에 표시된 사양뿐임. 기능 이름의 오타가 무엇이든 설치하는 의미를 갖지 않음 |
| 옵트아웃 | `security.allow_lazy_installs: false`로 설정하면 런타임 설치를 완전히 비활성화. 제한된 네트워크나 엄격한 보안 상태에 유용 |
| 자동 재시도 없음 | 실패는 `FeatureUnavailable`로 노출됨 — 잘못된 상태를 캐시하지 않고 재시도 폭풍도 없음 |

런타임 설치를 비활성화하려면 다음과 같이 합니다.

```yaml
# ~/.hermes/config.yaml
security:
  allow_lazy_installs: false
```

비활성화하면 선택적 의존성이 필요한 백엔드는 사용자에게 수동 설치(`pip install …`)를 실행하거나 `hermes tools`를 통해 다른 백엔드를 선택하라고 안내합니다.
