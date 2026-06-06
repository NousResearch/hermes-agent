---
sidebar_position: 13
title: "스크립트 전용 크론 작업 (LLM 제외)"
description: "LLM을 완전히 생략하는 전통적인 감시(watchdog) 크론 작업 — 스크립트가 일정에 따라 실행되고 표준 출력(stdout)이 메시징 플랫폼으로 전달됩니다. 메모리 경고, 디스크 경고, CI 핑(ping), 주기적 상태 확인 등에 활용하세요."
---

# 스크립트 전용 크론 작업 (Script-Only Cron Jobs)

가끔 전달하고자 하는 메시지가 이미 무엇인지 명확할 때가 있습니다. 에이전트가 이를 추론할 필요 없이, 단지 정해진 시간에 스크립트를 실행하고 그 결과물(있을 경우)을 Telegram, Discord, Slack, Signal 등에 전송하기만 하면 됩니다.

Hermes에서는 이를 **비에이전트 모드(no-agent mode)**라고 부릅니다. 이는 LLM 기능이 빠진 크론 시스템입니다.

<!-- ascii-guard-ignore -->
```
   ┌──────────────────┐          ┌──────────────────┐
   │ 스케줄러 실행      │  매번    │ 스크립트 실행      │
   │ (N분마다)          │ ──────▶ │ (bash 또는 python)│
   └──────────────────┘          └──────────────────┘
                                          │
                                          │ stdout (표준 출력)
                                          ▼
                                 ┌──────────────────┐
                                 │ 전송 라우터        │
                                 │ (telegram/disc…) │
                                 └──────────────────┘
```
<!-- ascii-guard-ignore-end -->

- **LLM 호출이 없습니다.** 토큰 소비, 에이전트 루프, 모델 비용이 전혀 들지 않습니다.
- **스크립트가 작업의 전부입니다.** 알림 전송 여부를 스크립트가 결정합니다. 출력이 나오면 메시지가 전송되고, 출력이 없으면 아무 일 없이 넘어갑니다.
- **Bash 또는 Python.** `.sh` / `.bash` 파일은 `/bin/bash` 환경에서 실행되며, 다른 확장자를 가진 파일은 현재의 Python 인터프리터로 실행됩니다. `~/.hermes/scripts/` 디렉터리에 위치한 모든 파일을 지원합니다.
- **동일한 스케줄러 사용.** LLM 작업과 마찬가지로 `cronjob` 환경에서 작동하므로, 일시 정지, 재개, 목록 조회, 로그 확인, 전송 대상 지정 등의 기능을 동일하게 사용할 수 있습니다.

## 언제 사용하나요?

비에이전트 모드는 다음의 경우에 적합합니다:

- **메모리 / 디스크 / GPU 상태 모니터링.** 5분마다 실행하여 임계값을 초과했을 때만 알림 전송.
- **CI 훅(hooks).** 배포 완료 후 커밋 SHA 게시. 빌드 실패 시 로그의 마지막 100줄 전송.
- **주기적 수치 보고.** "매일 오전 9시에 일일 Stripe 수익"을 단순한 API 호출과 예쁜 포맷으로 출력.
- **외부 이벤트 감지기(poller).** API를 확인하고 상태 변화 시 알림 전송.
- **하트비트(Heartbeats).** 서버가 살아있음을 증명하기 위해 N분마다 대시보드에 핑 전송.

에이전트가 무슨 말을 할지 **결정**해야 하는 경우 (예: 긴 문서 요약, 피드에서 흥미로운 항목 선택, 읽기 쉬운 알림 작성)에는 일반적인(LLM 기반) 크론 작업을 사용하세요. 비에이전트 모드는 스크립트의 표준 출력이 그대로 메시지가 되는 경우를 위해 마련되었습니다.

## 채팅을 통한 생성

비에이전트 모드의 진짜 장점은 에이전트가 직접 감시 스크립트를 설정해 줄 수 있다는 것입니다 — 에디터, 셸, CLI 플래그를 기억할 필요가 없습니다. 원하는 동작을 묘사하면 Hermes가 스크립트를 작성하고 스케줄을 설정한 뒤 언제 실행될지 알려줍니다.

### 예제 대화

> **당신:** 메모리가 85%를 초과하면 5분마다 텔레그램으로 알려줘.
>
> **Hermes:** *( `~/.hermes/scripts/memory-watchdog.sh` 작성 후 `no_agent=true` 옵션으로 `cronjob(...)` 호출)*
>
> 설정되었습니다. 5분마다 실행되며 RAM이 85%를 초과할 때만 텔레그램으로 알림을 보냅니다. 스크립트: `memory-watchdog.sh`. 작업 ID: `abc123`.

백그라운드에서 에이전트는 다음의 두 도구를 호출합니다:

```python
# 1. 확인 스크립트 작성
write_file(
    path="~/.hermes/scripts/memory-watchdog.sh",
    content='''#!/usr/bin/env bash
ram_pct=$(free | awk '/^Mem:/ {printf "%d", $3 * 100 / $2}')
if [ "$ram_pct" -ge 85 ]; then
  echo "RAM ${ram_pct}% on $(hostname)"
fi
# Empty stdout = silent tick; no message sent. (빈 출력이면 아무 메시지도 전송하지 않음)
''',
)

# 2. 스케줄 등록 — no_agent=True로 인해 매 실행 시 LLM을 거치지 않음
cronjob(
    action="create",
    schedule="every 5m",
    script="memory-watchdog.sh",
    no_agent=True,
    deliver="telegram",
    name="memory-watchdog",
)
```

이후부터는 스케줄러가 스크립트를 실행하고, 출력이 있을 때만 텔레그램으로 전달하며 모델을 전혀 사용하지 않으므로 무료로 실행됩니다.

### 에이전트의 자동 결정 방식

"X할 때 알려줘" 또는 "N분마다 Y를 확인하고 Z라면 내게 말해줘"와 같이 요청하면, Hermes의 `cronjob` 도구 설명서가 그 내용이 스크립트만으로 완전히 파악된다고 판단하여 `no_agent=True`를 선택합니다. 반면 요청에 추론 과정이 필요한 경우 (*"새 이슈를 요약해 줘"*, *"가장 재미있는 헤드라인을 골라줘"*, *"친절한 알림 문구를 작성해 줘"*)에는 일반적인 LLM 기반 경로를 사용합니다.

직접 `--no-agent` 옵션을 명시할 필요는 없습니다. 동작만 묘사해 주면 됩니다.

### 채팅을 통한 작업 관리

에이전트는 작업을 생성할 때처럼 일시 정지, 재개, 수정, 삭제도 할 수 있습니다:

> **당신:** 오늘 밤은 메모리 모니터링을 잠시 꺼둬.
>
> **Hermes:** *( `cronjob(action='pause', job_id='abc123')` 호출 )*
>
> 일시 정지되었습니다. "다시 켜줘"라고 말하거나 `hermes cron resume abc123`를 통해 재개할 수 있습니다.

> **당신:** 15분마다 실행되게 바꿔줘.
>
> **Hermes:** *( `cronjob(action='update', job_id='abc123', schedule='every 15m')` 호출 )*

CLI 명령어를 배우지 않더라도 에이전트를 통해 크론 작업의 전체 수명 주기 (생성 / 조회 / 업데이트 / 일시 정지 / 재개 / 즉시 실행 / 삭제)를 관리할 수 있습니다.

## CLI를 통한 생성

셸을 선호하시나요? CLI를 사용하면 세 가지 명령어로 동일한 결과를 얻을 수 있습니다:

```bash
# 1. 스크립트 작성
cat > ~/.hermes/scripts/memory-watchdog.sh <<'EOF'
#!/usr/bin/env bash
# RAM 사용량이 85%를 초과할 때만 알림 전송, 그 외에는 침묵
RAM_PCT=$(free | awk '/^Mem:/ {printf "%d", $3 * 100 / $2}')
if [ "$RAM_PCT" -ge 85 ]; then
  echo "⚠ RAM ${RAM_PCT}% on $(hostname)"
fi
# Empty stdout = silent run; no message sent. (빈 출력이면 아무 메시지도 전송하지 않음)
EOF
chmod +x ~/.hermes/scripts/memory-watchdog.sh

# 2. 스케줄 설정
hermes cron create "every 5m" \
  --no-agent \
  --script memory-watchdog.sh \
  --deliver telegram \
  --name "memory-watchdog"

# 3. 확인
hermes cron list
hermes cron run <job_id>    # 제대로 동작하는지 한 번 테스트 실행
```

이게 전부입니다. 프롬프트도, 스킬도, 모델도 필요하지 않습니다.


## 스크립트 출력과 전송 결과

| 스크립트 동작 | 결과 |
|-----------------|--------|
| 종료 코드 0, 출력 있음 | 표준 출력(stdout) 내용이 그대로 전송됨 |
| 종료 코드 0, 출력 없음 | 무음 실행 — 아무것도 전송되지 않음 |
| 종료 코드 0, 마지막 줄이 `{"wakeAgent": false}` | 무음 실행 (LLM 작업과 공유하는 조건 처리 방식) |
| 0이 아닌 종료 코드 (실패) | 오류 알림 전송 (고장난 스크립트가 말없이 실패하는 것을 막기 위함) |
| 스크립트 시간 초과 | 오류 알림 전송 |

"비어 있으면 조용히 넘긴다(silent when empty)"는 동작 방식이 전통적인 감시자 패턴의 핵심입니다. 스크립트는 매분 자유롭게 실행될 수 있지만, 채널에는 진정 주의가 필요할 때만 메시지가 도달합니다.

## 스크립트 작성 규칙

모든 스크립트는 반드시 `~/.hermes/scripts/` 디렉터리에 있어야 합니다. 이는 작업 생성 및 실행 시점 모두에 강제되며, 절대 경로, `~/` 확장, 또는 경로를 벗어나려는 시도(`../`)는 차단됩니다. 이 디렉터리는 LLM 작업에서 사용되는 실행 전 점검(pre-check) 스크립트와 동일한 폴더를 공유합니다.

파일 확장자에 따라 인터프리터가 선택됩니다:

| 확장자 | 인터프리터 |
|-----------|-------------|
| `.sh`, `.bash` | `/bin/bash` |
| 기타 확장자 | `sys.executable` (현재 Python 환경) |

의도적으로 `#!/...` 셔뱅(shebangs) 구문은 반영하지 **않습니다** — 인터프리터를 명확하고 제한적으로 유지하여 스케줄러가 신뢰할 수 있는 영역을 좁히기 위함입니다.

## 스케줄 작성 문법 (Schedule Syntax)

기존의 모든 크론 작업 문법과 동일합니다:

```bash
hermes cron create "every 5m"        # 주기(interval)
hermes cron create "every 2h"
hermes cron create "0 9 * * *"       # 표준 크론: 매일 오전 9시
hermes cron create "30m"             # 일회성(one-shot): 30분 뒤에 딱 한 번 실행
```

전체 문법은 [예약된 작업(Cron) 기능 참조](/user-guide/features/cron)를 확인하세요.

## 전송 대상 (Delivery Targets)

`--deliver` 옵션에는 게이트웨이가 지원하는 모든 목적지를 지정할 수 있습니다. 일반적으로 다음과 같이 사용합니다:

```bash
--deliver telegram                       # 해당 플랫폼 홈 채널
--deliver telegram:-1001234567890        # 특정 채팅방
--deliver telegram:-1001234567890:17585  # 특정 텔레그램 포럼 토픽
--deliver discord:#ops
--deliver slack:#engineering
--deliver signal:+15551234567
--deliver local                          # 단순히 ~/.hermes/cron/output/ 에만 저장
```

봇 토큰이 있는 플랫폼(Telegram, Discord, Slack, Signal, SMS, WhatsApp)의 경우, 스크립트가 실행되는 시점에 게이트웨이가 켜져 있지 않아도 됩니다 — 도구는 `~/.hermes/.env`나 `~/.hermes/config.yaml`에 있는 자격 증명을 사용하여 해당 플랫폼의 REST 엔드포인트를 직접 호출합니다.

## 수정 및 관리 (Editing and Lifecycle)

```bash
hermes cron list                                    # 전체 작업 목록 조회
hermes cron pause <job_id>                          # 실행 중지 (설정은 유지)
hermes cron resume <job_id>                         # 재개
hermes cron edit <job_id> --schedule "every 10m"    # 실행 주기 변경
hermes cron edit <job_id> --agent                   # LLM 모드로 전환
hermes cron edit <job_id> --no-agent --script …     # 다시 스크립트 모드로 전환
hermes cron remove <job_id>                         # 작업 삭제
```

LLM 작업에서 동작하는 모든 명령어(일시 정지, 재개, 수동 실행, 전송 대상 변경 등)는 비에이전트 작업에서도 똑같이 작동합니다.

## 실습 예제: 디스크 용량 경고

```bash
cat > ~/.hermes/scripts/disk-alert.sh <<'EOF'
#!/usr/bin/env bash
# / 또는 /home 파티션이 90% 이상 찼을 때 알림을 보냅니다.
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

두 파티션 모두 90% 미만일 때는 아무 소리도 내지 않다가, 하나라도 가득 차면 용량이 초과된 각 파티션에 대해 한 줄의 메시지를 출력(전송)합니다.

## 다른 패턴들과의 비교

| 방식 | 실행 내용 | 언제 사용하는가 |
|----------|-----------|-------------|
| `cronjob --no-agent` (이 문서) | Hermes 스케줄에 따른 사용자의 스크립트 | 추론이 필요 없는 주기적인 모니터링 / 알림 / 수치 파악 |
| `cronjob` (기본 설정, LLM 사용) | 사전 점검 스크립트(선택)가 포함된 에이전트 | 메시지 내용에 데이터를 분석한 추론 과정이 필요할 때 |
| OS 크론 + [웹훅 구독](/user-guide/messaging/webhooks)으로의 `curl` 요청 | OS 일정에 따른 사용자의 스크립트 | 모니터링 대상이 Hermes 자신일 수 있어, Hermes가 멈추더라도 실행되어야 할 때 |

*게이트웨이가 다운되어 있을 때조차* 무조건 실행되어야 하는 핵심 시스템 건강 감시자의 경우에는 OS 단의 크론과 함께, 평범한 `curl`을 이용해 Hermes의 웹훅(webhook) 구독 주소나 다른 외부 경고 엔드포인트로 전송하는 방식을 권장합니다. 그 방식은 OS에 의해 완전히 별개로 동작하며 Hermes가 켜져 있는지 여부에 의존하지 않기 때문입니다. 하지만 게이트웨이 내부에 내장된 스케줄러를 사용하는 이 방법은 외부 요소를 모니터링하기에 가장 알맞은 방식입니다.

## 관련 문서

- **[크론을 사용한 자동화 가이드](/guides/automate-with-cron)** — LLM 기반의 크론 사용 패턴
- **[예약된 작업(Cron) 기능 참조](/user-guide/features/cron)** — 전체 스케줄 문법, 작업 주기 관리, 전송 경로 설정
- **[웹훅(Webhook) 구독](/user-guide/messaging/webhooks)** — 외부 스케줄러를 위한 단순 실행(fire-and-forget) HTTP 진입점
- **[게이트웨이 내부(Internals)](/developer-guide/gateway-internals)** — 전송 라우터 동작 방식
