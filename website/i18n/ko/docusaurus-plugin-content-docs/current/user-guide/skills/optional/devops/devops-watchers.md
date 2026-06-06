---
title: "Watchers — 워터마크 중복 제거 기능을 통해 RSS, JSON API, GitHub를 폴링(Poll)합니다."
sidebar_label: "Watchers"
description: "워터마크 중복 제거 기능을 통해 RSS, JSON API, GitHub를 폴링(Poll)합니다."
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Watchers

워터마크 중복 제거(watermark dedup) 기능을 사용하여 RSS, JSON API, GitHub를 주기적으로 확인(Poll)합니다.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/devops/watchers` |
| Path | `optional-skills/devops/watchers` |
| Version | `1.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | linux, macos |
| Tags | `cron`, `polling`, `rss`, `github`, `http`, `automation`, `monitoring` |

## Reference: full SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지시 사항으로 보는 내용입니다.
:::

# Watchers

일정 간격으로 외부 소스를 확인하고 새로운 항목에만 반응합니다. 3개의 미리 준비된 스크립트와 공유 워터마크 헬퍼가 포함되어 있으며, 이를 크론(cron) 작업에 연결하거나(또는 터미널에서 즉석으로 실행) 사용할 수 있습니다.

## When to Use

- 사용자가 RSS/Atom 피드를 주시하고 새 항목에 대해 알림을 받고 싶을 때
- 사용자가 GitHub 리포지토리의 이슈/풀 리퀘스트(pulls)/릴리스/커밋을 주시하고 싶을 때
- 사용자가 임의의 JSON 엔드포인트를 확인(poll)하고 새 항목에 대해 알림을 받고 싶을 때
- 사용자가 "X를 위한 워처(watcher)" 또는 "X가 변경되면 알려줘"라고 요청할 때

## Mental model

워처(watcher)는 단순히 다음을 수행하는 스크립트입니다:

1. 외부 소스에서 데이터를 가져옵니다.
2. 이전에 확인한 ID의 워터마크 파일과 비교합니다.
3. 새로운 워터마크를 다시 기록합니다.
4. 새로운 항목을 stdout에 출력합니다 (변경된 내용이 없으면 아무것도 출력하지 않음).

아래의 스크립트들은 이 세 가지를 모두 처리합니다. 에이전트는 크론 작업, 웹훅 또는 대화형 채팅에서 터미널 도구를 통해 스크립트를 실행하고 새로운 내용을 보고합니다.

## Ready-made scripts

세 가지 스크립트 모두 스킬이 설치되면 `$HERMES_HOME/skills/devops/watchers/scripts/`에 위치하게 됩니다. 각 스크립트는 `--name` 인수로 지정된 키를 사용하여 `WATCHER_STATE_DIR` (기본값 `$HERMES_HOME/watcher-state/`)에서 상태 파일을 읽습니다.

| Script | What it watches | Dedup key |
|---|---|---|
| `watch_rss.py` | RSS 2.0 또는 Atom 피드 URL | `<guid>` / `<id>` |
| `watch_http_json.py` | 객체 목록을 반환하는 모든 JSON 엔드포인트 | 구성 가능한 id 필드 |
| `watch_github.py` | 리포지토리의 GitHub 이슈/풀 리퀘스트/릴리스/커밋 | `id` / `sha` |

세 스크립트 모두 다음의 특징을 갖습니다:

- 첫 번째 실행은 기준선(baseline)을 기록합니다 — 기존 피드를 절대 다시 재생하지 않습니다.
- 메모리를 제한하기 위해 워터마크는 제한된 ID 집합(최대 500개)입니다.
- 출력 형식: 항목당 `## <title>\n<url>\n\n<optional body>`
- 새로운 내용이 없을 때 빈 stdout 반환 — 호출자는 이를 조용히 넘겨야 할 상태로 취급합니다.
- 가져오기(fetch) 오류 시 0이 아닌 종료 코드를 반환합니다.

## Usage

터미널 도구에서 직접 워처를 실행하세요:

```bash
python $HERMES_HOME/skills/devops/watchers/scripts/watch_rss.py \
  --name hn --url https://news.ycombinator.com/rss --max 5
```

GitHub 리포지토리 보기 (시간당 60개의 익명 속도 제한을 피하려면 `~/.hermes/.env`에 `GITHUB_TOKEN`을 설정하세요):

```bash
python $HERMES_HOME/skills/devops/watchers/scripts/watch_github.py \
  --name hermes-issues --repo NousResearch/hermes-agent --scope issues
```

임의의 JSON API를 가져오기(Poll):

```bash
python $HERMES_HOME/skills/devops/watchers/scripts/watch_http_json.py \
  --name api --url https://api.example.com/events \
  --id-field event_id --items-path data.events
```

## Wiring into cron

다음과 같은 프롬프트로 에이전트에게 크론 작업을 예약하도록 요청하세요:

> 15분마다 `watch_rss.py --name hn --url https://news.ycombinator.com/rss`를 실행해 줘. 만약 뭔가를 출력한다면, 헤드라인을 요약해서 전달해 줘. 아무것도 출력하지 않으면 조용히 있어 줘.

에이전트는 크론 작업의 에이전트 루프 내부에서 터미널 도구를 통해 스크립트를 호출합니다. 크론의 내장 `--script` 플래그에 대한 변경은 필요하지 않습니다.

## State files

모든 워처는 `$HERMES_HOME/watcher-state/<name>.json`을 씁니다. 점검 방법:

```bash
cat $HERMES_HOME/watcher-state/hn.json
```

다시 실행 강제 (다음 실행을 첫 번째 폴링으로 취급):

```bash
rm $HERMES_HOME/watcher-state/hn.json
```

## Writing your own

세 가지 스크립트 모두 동일한 템플릿(워터마크 로드, 가져오기, 차이 비교, 저장, 발생)을 사용합니다. `scripts/_watermark.py`는 공유 헬퍼입니다. 이를 임포트하면 원자적 쓰기 + 제한된 ID 집합 + 첫 실행 기준선 설정을 무료로 얻을 수 있습니다. 상용구(boilerplate)가 얼마나 적게 필요한지 세 가지 참조 스크립트 중 아무거나 확인해 보세요.

## Common Pitfalls

1. **매 틱마다 "새로운 항목 없음" 헤더를 출력하는 경우.** 호출자는 "빈 stdout = 조용함(silent)"에 의존합니다. 빈 델타(변화 없음)에 대해 무엇이든 출력하면 채널에 스팸을 보내게 됩니다. 제공된 스크립트들은 이를 처리하지만, 커스텀 스크립트도 이를 반드시 처리해야 합니다.
2. **첫 번째 실행에서 항목을 배출할 것이라고 기대하는 경우.** 배출하지 않습니다 — 첫 번째 실행은 기준선(baseline)을 기록합니다. 초기 요약이 필요하다면 첫 실행 후 상태 파일을 삭제하거나 자신의 스크립트에 `--prime-with-latest N` 플래그를 추가하세요.
3. **무제한의 워터마크 증가.** 공유 헬퍼는 최대 500개의 ID로 제한합니다. 변동이 큰 피드의 경우 제한을 늘리고, 제한된 파일 시스템에서는 제한을 낮추세요.
4. **에이전트의 샌드박스가 쓸 수 없는 곳에 상태 디렉토리를 두는 경우.** `$HERMES_HOME/watcher-state/`는 항상 쓸 수 있습니다. Docker/Modal 백엔드는 임의의 호스트 경로를 보지 못할 수 있습니다.
