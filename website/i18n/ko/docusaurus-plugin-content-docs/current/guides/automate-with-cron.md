---
sidebar_position: 11
title: "Cron으로 무엇이든 자동화하기"
description: "Hermes cron을 활용한 실제 자동화 패턴 — 모니터링, 보고서, 파이프라인 및 여러 스킬을 사용하는 워크플로"
---

# Cron으로 무엇이든 자동화하기

[일일 브리핑 봇 튜토리얼](/guides/daily-briefing-bot)에서 기본 사항을 다룹니다. 이 가이드에서는 한 단계 더 나아가, 자신의 워크플로에 맞게 조정할 수 있는 실제 자동화 패턴 다섯 가지를 소개합니다.

전체 기능 참조는 [예약 작업(Cron)](/user-guide/features/cron)을 참고하세요.

:::info 핵심 개념
Cron 작업은 현재 대화의 메모리가 없는 새로운 에이전트 세션에서 실행됩니다. 프롬프트는 **완전히 독립적이어야 하며** — 에이전트가 알아야 할 모든 내용을 포함해야 합니다.
:::

:::tip LLM이 필요 없나요? 토큰을 전혀 사용하지 않는 두 가지 방법이 있습니다.
- **스크립트가 이미 정확한 메시지를 생성하는 반복 감시 작업**(메모리 알림, 디스크 알림, 하트비트): [스크립트 전용 cron 작업](/guides/cron-script-only)을 사용하세요. 같은 스케줄러를 사용하지만 LLM은 필요하지 않습니다. 채팅에서 Hermes에 설정을 요청할 수도 있습니다 — `cronjob` 도구가 `no_agent=True`를 사용해야 할 때를 알고 스크립트도 대신 작성합니다.
- **이미 실행 중인 스크립트에서 한 번만 실행**(CI 단계, post-commit 훅, 배포 스크립트, 외부에서 예약된 모니터): [`hermes send`](/guides/pipe-script-output)를 사용해 cron 항목을 설정하지 않고도 stdout 또는 파일을 Telegram / Discord / Slack / 기타 서비스로 바로 전달하세요.
:::

---

## 패턴 1: 웹사이트 변경 모니터

URL의 변경 사항을 감시하고 무엇인가 달라졌을 때만 알림을 받습니다.

여기서 비장의 무기는 `script` 매개변수입니다. 각 실행 전에 Python 스크립트가 실행되고, 그 stdout이 에이전트의 컨텍스트가 됩니다. 스크립트가 기계적인 작업(가져오기, diff)을 처리하고, 에이전트가 추론(이 변경이 흥미로운가? 중요한 이유는 무엇인가?)을 처리합니다.

모니터링 스크립트를 만듭니다.

```bash
mkdir -p ~/.hermes/scripts
```

```python title="~/.hermes/scripts/watch-site.py"
import hashlib, json, os, urllib.request

URL = "https://example.com/pricing"
STATE_FILE = os.path.expanduser("~/.hermes/scripts/.watch-site-state.json")

# Fetch current content
req = urllib.request.Request(URL, headers={"User-Agent": "Hermes-Monitor/1.0"})
content = urllib.request.urlopen(req, timeout=30).read().decode()
current_hash = hashlib.sha256(content.encode()).hexdigest()

# Load previous state
prev_hash = None
if os.path.exists(STATE_FILE):
    with open(STATE_FILE) as f:
        prev_hash = json.load(f).get("hash")

# Save current state
with open(STATE_FILE, "w") as f:
    json.dump({"hash": current_hash, "url": URL}, f)

# Output for the agent
if prev_hash and prev_hash != current_hash:
    print(f"CHANGE DETECTED on {URL}")
    print(f"Previous hash: {prev_hash}")
    print(f"Current hash: {current_hash}")
    print(f"\nCurrent content (first 2000 chars):\n{content[:2000]}")
else:
    print("NO_CHANGE")
```

cron 작업을 설정합니다.

```bash
/cron add "every 1h" "If the script output says CHANGE DETECTED, summarize what changed on the page and why it might matter. If it says NO_CHANGE, respond with just [SILENT]." --script ~/.hermes/scripts/watch-site.py --name "Pricing monitor" --deliver telegram
```

:::tip [SILENT] 트릭
cron 모니터링 작업에서는 변경 사항이 없을 때 에이전트가 `[SILENT]`만 응답하도록 지시하세요. Cron 전달 기능은 `[SILENT]`를 조용히 처리하라는 표식으로 취급하므로, 실제로 무언가 발생했을 때만 알림을 받게 됩니다 — 조용한 시간대에 스팸이 오지 않습니다.
:::

---

## 패턴 2: 주간 보고서

여러 소스에서 정보를 모아 형식이 지정된 요약을 만듭니다. 매주 한 번 실행되고 기본 채널로 전달됩니다.

```bash
/cron add "0 9 * * 1" "Generate a weekly report covering:

1. Search the web for the top 5 AI news stories from the past week
2. Search GitHub for trending repositories in the 'machine-learning' topic
3. Check Hacker News for the most discussed AI/ML posts

Format as a clean summary with sections for each source. Include links.
Keep it under 500 words — highlight only what matters." --name "Weekly AI digest" --deliver telegram
```

CLI에서 실행하려면 다음과 같이 합니다.

```bash
hermes cron create "0 9 * * 1" \
  "Generate a weekly report covering the top AI news, trending ML GitHub repos, and most-discussed HN posts. Format with sections, include links, keep under 500 words." \
  --name "Weekly AI digest" \
  --deliver telegram
```

`0 9 * * 1`은 표준 cron 표현식으로, 매주 월요일 오전 9시를 뜻합니다.

---

## 패턴 3: GitHub 저장소 감시기

저장소의 새 이슈, PR 또는 릴리스를 모니터링합니다.

```bash
/cron add "every 6h" "Check the GitHub repository NousResearch/hermes-agent for:
- New issues opened in the last 6 hours
- New PRs opened or merged in the last 6 hours
- Any new releases

Use the terminal to run gh commands:
  gh issue list --repo NousResearch/hermes-agent --state open --json number,title,author,createdAt --limit 10
  gh pr list --repo NousResearch/hermes-agent --state all --json number,title,author,createdAt,mergedAt --limit 10

Filter to only items from the last 6 hours. If nothing new, respond with [SILENT].
Otherwise, provide a concise summary of the activity." --name "Repo watcher" --deliver discord
```

:::warning 독립적인 프롬프트
프롬프트에 정확한 `gh` 명령이 포함되어 있는 점을 확인하세요. cron 에이전트는 이전 실행이나 사용자의 설정을 기억하지 못하므로 — 모든 내용을 구체적으로 적어야 합니다.
:::

---

## 패턴 4: 데이터 수집 파이프라인

일정한 간격으로 데이터를 수집하고 파일에 저장한 뒤, 시간에 따른 추세를 감지합니다. 이 패턴은 수집을 위한 스크립트와 분석을 위한 에이전트를 결합합니다.

```python title="~/.hermes/scripts/collect-prices.py"
import json, os, urllib.request
from datetime import datetime

DATA_DIR = os.path.expanduser("~/.hermes/data/prices")
os.makedirs(DATA_DIR, exist_ok=True)

# Fetch current data (example: crypto prices)
url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd"
data = json.loads(urllib.request.urlopen(url, timeout=30).read())

# Append to history file
entry = {"timestamp": datetime.now().isoformat(), "prices": data}
history_file = os.path.join(DATA_DIR, "history.jsonl")
with open(history_file, "a") as f:
    f.write(json.dumps(entry) + "\n")

# Load recent history for analysis
lines = open(history_file).readlines()
recent = [json.loads(l) for l in lines[-24:]]  # Last 24 data points

# Output for the agent
print(f"Current: BTC=${data['bitcoin']['usd']}, ETH=${data['ethereum']['usd']}")
print(f"Data points collected: {len(lines)} total, showing last {len(recent)}")
print(f"\nRecent history:")
for r in recent[-6:]:
    print(f"  {r['timestamp']}: BTC=${r['prices']['bitcoin']['usd']}, ETH=${r['prices']['ethereum']['usd']}")
```

```bash
/cron add "every 1h" "Analyze the price data from the script output. Report:
1. Current prices
2. Trend direction over the last 6 data points (up/down/flat)
3. Any notable movements (>5% change)

If prices are flat and nothing notable, respond with [SILENT].
If there's a significant move, explain what happened." \
  --script ~/.hermes/scripts/collect-prices.py \
  --name "Price tracker" \
  --deliver telegram
```

스크립트가 기계적인 수집을 담당하고, 에이전트가 추론 계층을 추가합니다.

---

## 패턴 5: 여러 스킬을 사용하는 워크플로

복잡한 예약 작업을 위해 스킬을 연결합니다. 프롬프트가 실행되기 전에 스킬이 순서대로 로드됩니다.

```bash
# Use the arxiv skill to find papers, then the obsidian skill to save notes
/cron add "0 8 * * *" "Search arXiv for the 3 most interesting papers on 'language model reasoning' from the past day. For each paper, create an Obsidian note with the title, authors, abstract summary, and key contribution." \
  --skill arxiv \
  --skill obsidian \
  --name "Paper digest"
```

도구에서 직접 실행하려면 다음과 같이 합니다.

```python
cronjob(
    action="create",
    skills=["arxiv", "obsidian"],
    prompt="Search arXiv for papers on 'language model reasoning' from the past day. Save the top 3 as Obsidian notes.",
    schedule="0 8 * * *",
    name="Paper digest",
    deliver="local"
)
```

스킬은 순서대로 로드됩니다 — 먼저 `arxiv`(에이전트에게 논문 검색 방법을 알려 줌), 그다음 `obsidian`(노트 작성 방법을 알려 줌)이 로드됩니다. 프롬프트가 이 둘을 연결합니다.

---

## 작업 관리

```bash
# List all active jobs
/cron list

# Trigger a job immediately (for testing)
/cron run <job_id>

# Pause a job without deleting it
/cron pause <job_id>

# Edit a running job's schedule or prompt
/cron edit <job_id> --schedule "every 4h"
/cron edit <job_id> --prompt "Updated task description"

# Add or remove skills from an existing job
/cron edit <job_id> --skill arxiv --skill obsidian
/cron edit <job_id> --clear-skills

# Remove a job permanently
/cron remove <job_id>
```

---

## 전달 대상

`--deliver` 플래그는 결과가 전달될 위치를 제어합니다.

| 대상 | 예시 | 사용 사례 |
|--------|---------|----------|
| `origin` | `--deliver origin` | 작업을 만든 동일한 채팅(기본값) |
| `local` | `--deliver local` | 로컬 파일에만 저장 |
| `telegram` | `--deliver telegram` | Telegram 기본 채널 |
| `discord` | `--deliver discord` | Discord 기본 채널 |
| `slack` | `--deliver slack` | Slack 기본 채널 |
| 특정 채팅 | `--deliver telegram:-1001234567890` | 특정 Telegram 그룹 |
| 스레드 | `--deliver telegram:-1001234567890:17585` | 특정 Telegram 주제 스레드 |

---

## 팁

**프롬프트를 독립적으로 작성하세요.** cron 작업의 에이전트는 대화를 기억하지 못합니다. URL, 저장소 이름, 형식 설정 및 전달 지침을 프롬프트에 직접 포함하세요.

**`[SILENT]`를 의도적으로 사용하세요.** 모니터링 작업에서는 "변경 사항이 없으면 `[SILENT]`만 응답하라"와 같은 지침을 포함하세요. 조용한 경우 에이전트에게 이 토큰을 설명하라고 하지 마세요 — cron은 `[SILENT]`를 전달 억제 표식으로 취급합니다.

**데이터 수집에는 스크립트를 사용하세요.** `script` 매개변수를 사용하면 Python 스크립트가 지루한 작업인 HTTP 요청, 파일 I/O, 상태 추적을 처리할 수 있습니다. 에이전트는 스크립트의 stdout만 보고 그에 기반해 추론합니다. 에이전트가 직접 가져오게 하는 것보다 저렴하고 안정적입니다.

**`/cron run`으로 테스트하세요.** 스케줄이 실행될 때까지 기다리기 전에 `/cron run <job_id>`를 사용해 즉시 실행하고 출력이 올바르게 보이는지 확인하세요.

*모든 매개변수, 예외 사례, 내부 동작을 포함한 전체 cron 참고 자료는 [예약 작업(Cron)](/user-guide/features/cron)을 확인하세요.*
