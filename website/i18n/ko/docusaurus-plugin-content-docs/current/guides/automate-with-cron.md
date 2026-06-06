---
sidebar_position: 11
title: "크론(Cron)으로 무엇이든 자동화하기 (Automate Anything with Cron)"
description: "Hermes 크론을 활용한 실제 자동화 패턴 — 모니터링, 보고서, 파이프라인 및 다중 스킬 워크플로우"
---

# 크론(Cron)으로 무엇이든 자동화하기

[일일 브리핑 봇 튜토리얼](/guides/daily-briefing-bot)에서는 기본 사항을 다룹니다. 이 가이드에서는 한 걸음 더 나아가 귀하의 워크플로우에 맞게 수정할 수 있는 5가지 실제 자동화 패턴을 소개합니다.

전체 기능 레퍼런스는 [Scheduled Tasks (Cron)](/user-guide/features/cron)을 참조하세요.

:::info 핵심 개념 (Key Concept)
크론 작업은 현재 채팅의 메모리가 전혀 없는 새로운 에이전트 세션에서 실행됩니다. 프롬프트는 에이전트가 알아야 할 모든 정보를 포함하여 **완전히 독립적(self-contained)**이어야 합니다.
:::

:::tip LLM이 필요 없으신가요? 제로 토큰 옵션 2가지가 있습니다.
- **반복적인 감시(Recurring watchdog)**: 스크립트가 이미 정확한 메시지(메모리 경고, 디스크 경고, 하트비트)를 생성하는 경우 [스크립트 전용 크론 작업](/guides/cron-script-only)을 사용하세요. 동일한 스케줄러를 사용하지만 LLM은 사용하지 않습니다. 채팅에서 Hermes에게 설정을 요청할 수도 있습니다 — `cronjob` 도구가 `no_agent=True`를 선택해야 할 때를 파악하고 귀하를 위해 스크립트를 작성합니다.
- **이미 실행 중인 스크립트의 일회성 실행(One-shot from a script)** (CI 단계, 커밋 후 훅, 배포 스크립트, 외부 스케줄된 모니터링): 크론 항목을 설정하지 않고 stdout이나 파일을 Telegram / Discord / Slack 등에 바로 파이핑(pipe)하려면 [`hermes send`](/guides/pipe-script-output)를 사용하세요.
:::

---

## 패턴 1: 웹사이트 변경 모니터링

URL의 변경 사항을 지켜보고 변경된 사항이 있을 때만 알림을 받습니다.

여기서는 `script` 매개변수가 비밀 무기입니다. 각 실행 전에 Python 스크립트가 실행되고, 그 stdout이 에이전트의 컨텍스트가 됩니다. 스크립트는 기계적인 작업(가져오기, diff)을 처리하고, 에이전트는 추론(이 변경 사항이 흥미로운가?)을 처리합니다.

모니터링 스크립트를 생성하세요:

```bash
mkdir -p ~/.hermes/scripts
```

```python title="~/.hermes/scripts/watch-site.py"
import hashlib, json, os, urllib.request

URL = "https://example.com/pricing"
STATE_FILE = os.path.expanduser("~/.hermes/scripts/.watch-site-state.json")

# 현재 콘텐츠 가져오기
req = urllib.request.Request(URL, headers={"User-Agent": "Hermes-Monitor/1.0"})
content = urllib.request.urlopen(req, timeout=30).read().decode()
current_hash = hashlib.sha256(content.encode()).hexdigest()

# 이전 상태 로드
prev_hash = None
if os.path.exists(STATE_FILE):
    with open(STATE_FILE) as f:
        prev_hash = json.load(f).get("hash")

# 현재 상태 저장
with open(STATE_FILE, "w") as f:
    json.dump({"hash": current_hash, "url": URL}, f)

# 에이전트를 위한 출력
if prev_hash and prev_hash != current_hash:
    print(f"변경 감지됨 (CHANGE DETECTED on) {URL}")
    print(f"이전 해시 (Previous hash): {prev_hash}")
    print(f"현재 해시 (Current hash): {current_hash}")
    print(f"\n현재 콘텐츠 (처음 2000자):\n{content[:2000]}")
else:
    print("NO_CHANGE")
```

크론 작업을 설정하세요:

```bash
/cron add "every 1h" "만약 스크립트 출력이 CHANGE DETECTED라고 하면, 페이지에서 무엇이 변경되었는지 그리고 그것이 왜 중요할 수 있는지 요약해줘. 만약 NO_CHANGE라고 하면, 단순히 [SILENT]로 응답해." --script ~/.hermes/scripts/watch-site.py --name "Pricing monitor" --deliver telegram
```

:::tip [SILENT] 트릭
에이전트의 최종 응답에 `[SILENT]`가 포함되어 있으면 전송이 억제됩니다. 이는 조용한 시간에는 알림 스팸이 발생하지 않고 실제로 어떤 일이 일어났을 때만 알림을 받는다는 것을 의미합니다.
:::

---

## 패턴 2: 주간 보고서 (Weekly Report)

여러 소스의 정보를 포맷된 요약본으로 컴파일합니다. 일주일에 한 번 실행되며 홈 채널로 전송됩니다.

```bash
/cron add "0 9 * * 1" "다음을 다루는 주간 보고서를 생성해:

1. 웹에서 지난주의 상위 5개 AI 뉴스 스토리 검색
2. GitHub에서 'machine-learning' 토픽의 트렌딩 리포지토리 검색
3. Hacker News에서 가장 많이 논의된 AI/ML 게시물 확인

각 소스에 대한 섹션이 있는 깔끔한 요약 형식으로 작성해. 링크를 포함해줘.
500단어 미만으로 유지하고 — 중요한 내용만 강조해." --name "Weekly AI digest" --deliver telegram
```

CLI에서:

```bash
hermes cron create "0 9 * * 1" \
  "상위 AI 뉴스, 트렌딩 ML GitHub 리포지토리, 가장 많이 논의된 HN 게시물을 다루는 주간 보고서를 생성해. 섹션으로 형식화하고, 링크를 포함하며, 500단어 미만으로 유지해." \
  --name "Weekly AI digest" \
  --deliver telegram
```

`0 9 * * 1`은 표준 크론 표현식입니다: 매주 월요일 오전 9:00.

---

## 패턴 3: GitHub 리포지토리 감시자 (Repository Watcher)

리포지토리에서 새 이슈, PR 또는 릴리스를 모니터링합니다.

```bash
/cron add "every 6h" "GitHub 리포지토리 NousResearch/hermes-agent를 확인해서 다음을 알려줘:
- 지난 6시간 동안 열린 새 이슈
- 지난 6시간 동안 열리거나 병합된 새 PR
- 모든 새 릴리스

터미널을 사용해 gh 명령어를 실행해:
  gh issue list --repo NousResearch/hermes-agent --state open --json number,title,author,createdAt --limit 10
  gh pr list --repo NousResearch/hermes-agent --state all --json number,title,author,createdAt,mergedAt --limit 10

지난 6시간 동안의 항목만 필터링해. 새로운 것이 없으면 [SILENT]로 응답해.
그렇지 않으면 활동에 대한 간결한 요약을 제공해." --name "Repo watcher" --deliver discord
```

:::warning 독립적인 프롬프트 (Self-Contained Prompts)
프롬프트에 정확한 `gh` 명령어가 포함되어 있는 것을 주목하세요. 크론 에이전트는 이전 실행 내역이나 귀하의 선호도에 대한 기억이 없으므로 — 모든 것을 명시해야 합니다.
:::

---

## 패턴 4: 데이터 수집 파이프라인

정기적으로 데이터를 스크랩하고, 파일에 저장하고, 시간이 지남에 따른 추세를 감지합니다. 이 패턴은 스크립트(수집용)와 에이전트(분석용)를 결합합니다.

```python title="~/.hermes/scripts/collect-prices.py"
import json, os, urllib.request
from datetime import datetime

DATA_DIR = os.path.expanduser("~/.hermes/data/prices")
os.makedirs(DATA_DIR, exist_ok=True)

# 현재 데이터 가져오기 (예: 암호화폐 가격)
url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd"
data = json.loads(urllib.request.urlopen(url, timeout=30).read())

# 히스토리 파일에 추가
entry = {"timestamp": datetime.now().isoformat(), "prices": data}
history_file = os.path.join(DATA_DIR, "history.jsonl")
with open(history_file, "a") as f:
    f.write(json.dumps(entry) + "\n")

# 분석을 위해 최근 기록 로드
lines = open(history_file).readlines()
recent = [json.loads(l) for l in lines[-24:]]  # 최근 24개 데이터 포인트

# 에이전트를 위한 출력
print(f"현재 (Current): BTC=${data['bitcoin']['usd']}, ETH=${data['ethereum']['usd']}")
print(f"수집된 데이터 포인트: 총 {len(lines)}개, 최근 {len(recent)}개 표시 중")
print(f"\n최근 기록 (Recent history):")
for r in recent[-6:]:
    print(f"  {r['timestamp']}: BTC=${r['prices']['bitcoin']['usd']}, ETH=${r['prices']['ethereum']['usd']}")
```

```bash
/cron add "every 1h" "스크립트 출력의 가격 데이터를 분석해. 다음을 보고해:
1. 현재 가격
2. 지난 6개 데이터 포인트 동안의 추세 방향 (상승/하락/보합)
3. 주목할 만한 움직임 (>5% 변화)

가격이 변동이 없고 주목할 만한 사항이 없으면 [SILENT]로 응답해.
상당한 움직임이 있으면 무슨 일이 일어났는지 설명해." \
  --script ~/.hermes/scripts/collect-prices.py \
  --name "Price tracker" \
  --deliver telegram
```

스크립트는 기계적인 수집을 수행하고, 에이전트는 추론 계층(reasoning layer)을 추가합니다.

---

## 패턴 5: 다중 스킬 워크플로우

복잡한 예약 작업을 위해 스킬들을 함께 연결합니다. 프롬프트가 실행되기 전에 스킬들이 순서대로 로드됩니다.

```bash
# arxiv 스킬을 사용하여 논문을 찾은 다음, obsidian 스킬을 사용하여 노트를 저장
/cron add "0 8 * * *" "지난 하루 동안 '언어 모델 추론(language model reasoning)'에 관한 가장 흥미로운 논문 3개를 arXiv에서 검색해. 각 논문에 대해 제목, 저자, 초록 요약, 주요 기여가 포함된 Obsidian 노트를 작성해." \
  --skill arxiv \
  --skill obsidian \
  --name "Paper digest"
```

도구에서 직접 실행:

```python
cronjob(
    action="create",
    skills=["arxiv", "obsidian"],
    prompt="'언어 모델 추론'에 관한 논문을 지난 하루 동안 arXiv에서 검색하세요. 상위 3개를 Obsidian 노트로 저장하세요.",
    schedule="0 8 * * *",
    name="Paper digest",
    deliver="local"
)
```

스킬들은 순서대로 로드됩니다 — 먼저 `arxiv`(에이전트에게 논문 검색 방법을 가르침), 그다음 `obsidian`(노트 작성 방법을 가르침). 프롬프트는 이 둘을 하나로 묶습니다.

---

## 작업 관리하기 (Managing Your Jobs)

```bash
# 모든 활성 작업 목록 표시
/cron list

# 작업을 즉시 트리거 (테스트용)
/cron run <job_id>

# 작업을 삭제하지 않고 일시 중지
/cron pause <job_id>

# 실행 중인 작업의 일정 또는 프롬프트 편집
/cron edit <job_id> --schedule "every 4h"
/cron edit <job_id> --prompt "업데이트된 작업 설명"

# 기존 작업에서 스킬을 추가하거나 제거
/cron edit <job_id> --skill arxiv --skill obsidian
/cron edit <job_id> --clear-skills

# 작업을 영구적으로 제거
/cron remove <job_id>
```

---

## 전송 대상 (Delivery Targets)

`--deliver` 플래그는 결과가 어디로 갈지 제어합니다:

| 대상 (Target) | 예시 | 사용 사례 |
|--------|---------|----------|
| `origin` | `--deliver origin` | 작업을 생성한 동일한 채팅 (기본값) |
| `local` | `--deliver local` | 로컬 파일에만 저장 |
| `telegram` | `--deliver telegram` | 귀하의 Telegram 홈 채널 |
| `discord` | `--deliver discord` | 귀하의 Discord 홈 채널 |
| `slack` | `--deliver slack` | 귀하의 Slack 홈 채널 |
| 특정 채팅 | `--deliver telegram:-1001234567890` | 특정 Telegram 그룹 |
| 스레드 | `--deliver telegram:-1001234567890:17585` | 특정 Telegram 토픽 스레드 |

---

## 팁 (Tips)

**프롬프트를 독립적으로 만드세요.** 크론 작업의 에이전트는 귀하의 대화를 기억하지 못합니다. URL, 리포지토리 이름, 형식 선호도, 전송 지침을 프롬프트에 직접 포함하세요.

**`[SILENT]`를 적극 활용하세요.** 모니터링 작업의 경우 항상 "아무 변화가 없으면 `[SILENT]`로 응답해"와 같은 지침을 포함하세요. 이는 알림 노이즈를 방지합니다.

**데이터 수집에 스크립트를 사용하세요.** `script` 매개변수를 사용하면 지루한 부분(HTTP 요청, 파일 I/O, 상태 추적)을 Python 스크립트가 처리하도록 할 수 있습니다. 에이전트는 스크립트의 stdout만 보고 추론을 적용합니다. 이는 에이전트가 직접 가져오기를 수행하게 하는 것보다 비용이 저렴하고 더 신뢰할 수 있습니다.

**`/cron run`으로 테스트하세요.** 예약된 시간이 되기를 기다리기 전에 `/cron run <job_id>`를 사용하여 즉시 실행하고 출력이 올바른지 확인하세요.

**일정 표현식.** 지원되는 형식: 상대적 지연(`30m`), 간격(`every 2h`), 표준 크론 표현식(`0 9 * * *`), ISO 타임스탬프(`2025-06-15T09:00:00`). `매일 오전 9시(daily at 9am)`와 같은 자연어는 지원되지 않습니다 — 대신 `0 9 * * *`를 사용하세요.

---

*모든 매개변수, 엣지 케이스 및 내부 작동 방식 등을 포함한 전체 크론 레퍼런스는 [Scheduled Tasks (Cron)](/user-guide/features/cron)을 참조하세요.*
