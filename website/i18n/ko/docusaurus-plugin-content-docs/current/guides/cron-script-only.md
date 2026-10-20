---
sidebar_position: 13
title: "스크립트 전용 Cron 작업 (LLM 없음)"
description: "LLM을 완전히 건너뛰는 기존 감시 cron 작업 — 일정에 따라 스크립트가 실행되고 stdout이 메시징 플랫폼으로 전달됩니다. 메모리 알림, 디스크 알림, CI 핑, 주기적인 상태 점검."
---

# 스크립트 전용 Cron 작업

때로는 보내고 싶은 메시지를 이미 정확히 알고 있습니다. 에이전트가 이를 추론할 필요는 없습니다. 타이머에 따라 스크립트를 실행하고 그 출력(있는 경우)을 Telegram / Discord / Slack / Signal에 전달하기만 하면 됩니다.

Hermes는 이를 **에이전트 없음 모드**라고 부릅니다. LLM을 제외한 cron 시스템입니다.

<!-- ascii-guard-ignore -->
```
   ┌──────────────────┐          ┌──────────────────┐
   │ scheduler tick   │  every   │ run script       │
   │ (every N minutes)│ ──────▶ │ (bash or python) │
   └──────────────────┘          └──────────────────┘
                                          │
                                          │ stdout
                                          ▼
                                 ┌──────────────────┐
                                 │ delivery router  │
                                 │ (telegram/disc…) │
                                 └──────────────────┘
```
<!-- ascii-guard-ignore-end -->

- **LLM 호출 없음.** 토큰 0개, 에이전트 루프 0회, 모델 비용 0원입니다.
- **스크립트가 작업입니다.** 알림을 보낼지 여부를 스크립트가 결정합니다. 출력을 내보내면 → 메시지가 전송됩니다. 아무것도 내보내지 않으면 → 조용히 실행됩니다.
- **Bash 또는 Python.** `.sh` / `.bash` 파일은 가능한 경우 `PATH`의 `bash`에서, 그렇지 않으면 `/bin/bash`에서 실행됩니다. 그 밖의 확장자는 현재 Python 인터프리터에서 실행됩니다. 경로는 `~/.hermes/scripts/` 내부로 확인되어야 합니다 (상대 경로, 절대 경로 또는 `~` 형식은 해당 디렉터리 안에 머무는 경우 사용할 수 있습니다). Cron 스크립트는 Hermes 프로세스 환경에서 프로바이더 자격 증명을 상속하지 않습니다.
- **동일한 스케줄러.** LLM 작업과 함께 `cronjob`에 존재하므로 일시 중지, 재개, 목록 조회, 로그 확인, 전달 대상 지정이 모두 동일하게 작동합니다.

## 사용 시점

다음에는 에이전트 없음 모드를 사용하세요.

- **메모리 / 디스크 / GPU 감시.** 5분마다 실행하고 임계값을 넘을 때만 알립니다.
- **CI 훅.** 배포 완료 → 커밋 SHA를 게시합니다. 빌드 실패 → 로그의 마지막 100줄을 보냅니다.
- **주기적 지표.** 간단한 API 호출 + 예쁜 형식의 출력으로 "매일 오전 9시에 Stripe 매출"을 확인합니다.
- **외부 이벤트 폴러.** API를 확인하고 상태가 변경될 때 알립니다.
- **하트비트.** 호스트가 살아 있음을 증명하기 위해 N분마다 대시보드에 핑을 보냅니다.

에이전트가 무엇을 말할지 **결정**해야 한다면 일반적인(LLM 기반) cron 작업을 사용하세요. 긴 문서를 요약하거나, 피드에서 흥미로운 항목을 고르거나, 사람이 읽기 좋은 메시지를 작성하는 경우가 이에 해당합니다. 스크립트의 stdout 자체가 이미 메시지인 경우에는 에이전트 없음 경로를 사용합니다.

## 채팅에서 생성하기

에이전트 없음 모드의 진정한 장점은 에이전트가 직접 감시 작업을 설정할 수 있다는 것입니다. 편집기도, 셸도, CLI 플래그를 기억할 필요도 없습니다. 원하는 내용을 설명하면 Hermes가 스크립트를 작성하고, 일정을 예약하고, 언제 실행될지 알려줍니다.

### 예시 대화

> **사용자:** 5분마다 RAM이 85%를 넘으면 Telegram으로 알려줘
>
> **Hermes:** *( `~/.hermes/scripts/memory-watchdog.sh`를 작성한 다음 `no_agent=true`로 `cronjob(...)`을 호출합니다.)*
>
> 설정했습니다. 5분마다 실행되며 RAM이 85%를 넘을 때만 Telegram으로 알립니다. 스크립트: `memory-watchdog.sh`. 작업 ID: `abc123`.

내부적으로 에이전트는 두 번의 도구 호출을 수행합니다.

```python
# 1. Write the check script
write_file(
    path="~/.hermes/scripts/memory-watchdog.sh",
    content='''#!/usr/bin/env bash
ram_pct=$(free | awk '/^Mem:/ {printf "%d", $3 * 100 / $2}')
if [ "$ram_pct" -ge 85 ]; then
  echo "RAM ${ram_pct}% on $(hostname)"
fi
# Empty stdout = silent tick; no message sent.
''',
)

# 2. Schedule it — no_agent=True skips the LLM on every tick
cronjob(
    action="create",
    schedule="every 5m",
    script="memory-watchdog.sh",
    no_agent=True,
    deliver="telegram",
    name="memory-watchdog",
)
```

이제부터 모든 실행은 무료입니다. 스케줄러가 스크립트를 실행하고, stdout이 비어 있지 않으면 Telegram으로 전달하며, 모델에는 절대 접근하지 않습니다.

### 에이전트가 대신 결정하는 것

"X일 때 알려줘" 또는 "N분마다 Y를 확인하고 Z이면 알려줘"와 같이 요청하면 Hermes의 `cronjob` 도구 설명은 메시지 내용이 스크립트로 완전히 결정되는 경우 `no_agent=True`를 사용하도록 안내합니다. 요청에 추론이 필요한 경우(*"새 이슈를 요약해줘"*, *"가장 흥미로운 헤드라인을 골라줘"*, *"친근한 알림을 작성해줘"*)에는 일반적인 LLM 기반 경로로 돌아갑니다.

직접 `--no-agent`를 지정할 필요가 없습니다. 원하는 동작을 설명하기만 하면 됩니다.

### 채팅에서 감시 작업 관리하기

에이전트는 작업을 생성할 때와 동일한 방식으로 일시 중지, 재개, 수정, 삭제할 수 있습니다.

> **사용자:** 오늘 밤에는 메모리 감시를 중지해줘
>
> **Hermes:** *( `cronjob(action='pause', job_id='abc123')`를 호출합니다.)*
>
> 일시 중지했습니다. "다시 켜줘"라고 말하거나 `hermes cron resume abc123`을 사용해 재개하세요.

> **사용자:** 15분마다 실행하도록 바꿔줘
>
> **Hermes:** *( `cronjob(action='update', job_id='abc123', schedule='every 15m')`를 호출합니다.)*

전체 수명 주기(생성 / 목록 조회 / 수정 / 일시 중지 / 재개 / 지금 실행 / 삭제)를 CLI 명령을 배울 필요 없이 에이전트가 사용할 수 있습니다.

## CLI에서 생성하기

셸을 선호하나요? CLI 경로에서도 세 명령으로 동일한 결과를 얻을 수 있습니다.

```bash
# 1. Write your script
cat > ~/.hermes/scripts/memory-watchdog.sh <<'EOF'
#!/usr/bin/env bash
# Alert when RAM usage is over 85%. Silent otherwise.
RAM_PCT=$(free | awk '/^Mem:/ {printf "%d", $3 * 100 / $2}')
if [ "$RAM_PCT" -ge 85 ]; then
  echo "⚠ RAM ${RAM_PCT}% on $(hostname)"
fi
# Empty stdout = silent run; no message sent.
EOF
chmod +x ~/.hermes/scripts/memory-watchdog.sh

# 2. Schedule it
hermes cron create "every 5m" \
  --no-agent \
  --script memory-watchdog.sh \
  --deliver telegram \
  --name "memory-watchdog"

# 3. Verify
hermes cron list
hermes cron run <job_id>    # fire it once to test
```

이것이 전부입니다. 프롬프트도, 스킬도, 모델도 필요하지 않습니다.

## 스크립트 출력과 전달의 관계

| 스크립트 동작 | 결과 |
|-----------------|--------|
| 종료 코드 0, stdout이 비어 있지 않음 | stdout이 그대로 전달됨 |
| 종료 코드 0, stdout이 비어 있음 | 조용한 실행 — 전달하지 않음 |
| 종료 코드 0, stdout의 마지막 줄에 `{"wakeAgent": false}`가 포함됨 | 조용한 실행 (LLM 작업과 공유하는 게이트) |
| 0이 아닌 종료 코드 | 오류 알림이 전달됨 (고장 난 감시 작업이 조용히 실패하지 않도록) |
| 스크립트 시간 초과 | 오류 알림이 전달됨 |

"비어 있으면 조용히 실행"되는 동작이 기존 감시 패턴의 핵심입니다. 스크립트는 매분 무료로 실행할 수 있지만, 실제로 주의가 필요한 경우에만 채널에 메시지가 표시됩니다.

## 스크립트 규칙

스크립트는 `~/.hermes/scripts/`에 있어야 합니다. 이는 작업 생성 시점과 실행 시점 모두에서 적용됩니다. 절대 경로, `~/` 확장, 경로 탐색 패턴(`../`)은 거부됩니다. 동일한 디렉터리는 LLM 작업에 사용되는 사전 검사 스크립트 게이트와 공유됩니다.

인터프리터는 파일 확장자로 선택합니다.

| 확장자 | 인터프리터 |
|-----------|-------------|
| `.sh`, `.bash` | `PATH`의 `bash` (대체: `/bin/bash`) |
| 그 외 | `sys.executable` (현재 Python) |

`#!/...` 셰뱅은 의도적으로 따르지 않습니다. 인터프리터 집합을 명시적이고 작게 유지하면 스케줄러가 신뢰해야 하는 범위가 줄어듭니다.

## 일정 문법

다른 모든 cron 작업과 동일합니다.

```bash
hermes cron create "every 5m"        # interval
hermes cron create "every 2h"
hermes cron create "0 9 * * *"       # standard cron: 9am daily
hermes cron create "30m"             # one-shot: run once in 30 minutes
```

전체 문법은 [cron 기능 참고 자료](/user-guide/features/cron)를 참조하세요.

## 전달 대상

`--deliver`는 게이트웨이가 알고 있는 모든 대상을 허용합니다. 다음은 일반적인 형식입니다.

```bash
--deliver telegram                       # platform home channel
--deliver telegram:-1001234567890        # specific chat
--deliver telegram:-1001234567890:17585  # specific Telegram forum topic
--deliver discord:#ops
--deliver slack:#engineering
--deliver signal:+15551234567
--deliver local                          # just save to ~/.hermes/cron/output/
```

봇 토큰 플랫폼(Telegram, Discord, Slack, Signal, SMS, WhatsApp)의 경우 스크립트 실행 시 실행 중인 게이트웨이가 필요하지 않습니다. 도구가 `~/.hermes/.env` / `~/.hermes/config.yaml`에 이미 있는 자격 증명을 사용해 각 플랫폼의 REST 엔드포인트를 직접 호출합니다.

## 편집 및 수명 주기

```bash
hermes cron list                                    # see all jobs
hermes cron pause <job_id>                          # stop firing, keep definition
hermes cron resume <job_id>
hermes cron edit <job_id> --schedule "every 10m"    # adjust cadence
hermes cron edit <job_id> --agent                   # flip to LLM mode
hermes cron edit <job_id> --no-agent --script …     # flip back
hermes cron remove <job_id>                         # delete it
```

LLM 작업에서 작동하는 모든 기능(일시 중지, 재개, 수동 실행, 전달 대상 변경)은 에이전트 없음 작업에서도 작동합니다.

## 예제: 디스크 공간 알림

```bash
cat > ~/.hermes/scripts/disk-alert.sh <<'EOF'
#!/usr/bin/env bash
# Alert when / or /home is over 90% full.
THRESHOLD=90
df -h / /home 2>/dev/null | awk -v t="$THRESHOLD" '
  NR > 1 && $5+0 >= t {
    printf "⚠ Disk %s full on %s\n", $5, $6
  }
'
EOF
chmod +x ~/.hermes/scripts/disk-alert.sh

hermes cron create "*/15 * * * *" \
  --no-agent \
  --script disk-alert.sh \
  --deliver telegram \
  --name "disk-alert"
```

두 파일 시스템이 모두 90% 미만이면 조용히 실행되고, 하나가 가득 차면 임계값을 넘은 각 파일 시스템마다 정확히 한 줄씩 실행됩니다.

## 다른 패턴과의 비교

| 접근 방식 | 실행되는 것 | 사용 시점 |
|----------|-------------|----------|
| `cronjob --no-agent` (이 페이지) | Hermes 일정에 따른 사용자 스크립트 | 추론이 필요 없는 반복 감시 / 알림 / 지표 |
| `cronjob` (기본값, LLM) | 선택적 사전 검사 스크립트가 있는 에이전트 | 데이터에 대한 추론이 필요한 메시지 콘텐츠 |
| OS cron + [웹훅 구독](/user-guide/messaging/webhooks)에 대한 `curl` | OS 일정에 따른 사용자 스크립트 | Hermes가 비정상일 수 있는 경우 (모니터링 대상 자체인 경우) |

게이트웨이가 중단되어도 반드시 실행되어야 하는 중요한 시스템 상태 감시 작업에는 Hermes 웹훅 구독(또는 외부 알림 엔드포인트)에 일반 `curl`을 사용하는 OS 수준 cron을 사용하세요. 이러한 작업은 독립적인 OS 프로세스로 실행되며 Hermes가 실행 중인지에 의존하지 않습니다. 게이트웨이 내부 스케줄러는 모니터링 대상이 외부에 있을 때 적합합니다.

## 관련 문서

- [Cron으로 무엇이든 자동화하기](/guides/automate-with-cron) — LLM 기반 cron 패턴.
- [예약 작업 (Cron) 참고 자료](/user-guide/features/cron) — 전체 일정 문법, 수명 주기, 전달 라우팅.
- [웹훅 구독](/user-guide/messaging/webhooks) — 외부 스케줄러를 위한 단방향 HTTP 진입점.
- [게이트웨이 내부 구조](/developer-guide/gateway-internals) — 전달 라우터 내부 구조.
