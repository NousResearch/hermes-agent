---
title: "Watchers — 워터마크 중복 제거로 RSS, JSON API, GitHub 폴링"
sidebar_label: "Watchers"
description: "워터마크 중복 제거로 RSS, JSON API, GitHub 폴링"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Watchers

워터마크 중복 제거로 RSS, JSON API, GitHub를 폴링합니다.

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 선택 사항 — `hermes skills install official/devops/watchers`로 설치 |
| 경로 | `optional-skills/devops/watchers` |
| 버전 | `1.0.0` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos |
| 태그 | `cron`, `polling`, `rss`, `github`, `http`, `automation`, `monitoring` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성화되었을 때 에이전트가 보는 지침이기도 합니다.
:::

# Watchers

외부 소스를 일정 간격으로 폴링하고 새 항목에만 반응합니다. 공유 워터마크 헬퍼와 바로 사용할 수 있는 세 가지 스크립트를 cron 작업에 연결하거나 터미널에서 임시로 실행할 수 있습니다.

## 사용 시점

- RSS/Atom 피드를 감시하고 새 항목을 알림받고 싶을 때
- GitHub 저장소의 이슈 / 풀 리퀘스트 / 릴리스 / 커밋을 감시하고 싶을 때
- 임의의 JSON 엔드포인트를 폴링하고 새 항목을 알림받고 싶을 때
- “X용 watcher를 만들어 줘” 또는 “X가 변경되면 알려 줘”라고 요청할 때

## 작동 방식

Watcher는 다음과 같은 스크립트일 뿐입니다.

1. 외부 소스에서 데이터를 가져옵니다.
2. 이전에 확인한 ID를 기록한 워터마크 파일과 비교합니다.
3. 새 워터마크를 다시 기록합니다.
4. 새 항목을 stdout에 출력합니다(변경 사항이 없으면 아무것도 출력하지 않음).

아래 스크립트가 이 세 가지 작업을 모두 처리합니다. 에이전트는 터미널 도구를 통해 cron 작업, 웹훅 또는 대화형 채팅에서 스크립트를 실행하고 새 항목을 보고합니다.

## 바로 사용할 수 있는 스크립트

스킬을 설치하면 세 스크립트가 모두 `$HERMES_HOME/skills/devops/watchers/scripts/`에 있습니다. 각 스크립트는 상태 파일을 저장할 `WATCHER_STATE_DIR`을 읽으며(기본값은 `$HERMES_HOME/watcher-state/`), 상태 파일은 `--name` 인수로 지정한 이름을 사용합니다.

| 스크립트 | 감시 대상 | 중복 제거 키 |
|---|---|---|
| `watch_rss.py` | RSS 2.0 또는 Atom 피드 URL | `<guid>` / `<id>` |
| `watch_http_json.py` | 객체 목록을 반환하는 모든 JSON 엔드포인트 | 설정 가능한 id 필드 |
| `watch_github.py` | 저장소의 GitHub 이슈 / 풀 / 릴리스 / 커밋 | `id` / `sha` |

세 스크립트 모두 다음과 같이 동작합니다.

- 첫 실행에서는 기준선을 기록하며 기존 피드를 다시 재생하지 않습니다.
- 워터마크는 메모리 사용량을 제한하기 위해 최대 500개의 ID로 제한된 집합입니다.
- 출력 형식: 항목마다 `## <title>\n<url>\n\n<optional body>`
- 새 항목이 없으면 stdout이 비어 있습니다. 호출자는 이를 무음으로 처리합니다.
- 가져오기 오류가 발생하면 0이 아닌 종료 코드로 종료합니다.

## 사용법

터미널 도구에서 watcher를 직접 실행합니다.

```bash
python $HERMES_HOME/skills/devops/watchers/scripts/watch_rss.py \
  --name hn --url https://news.ycombinator.com/rss --max 5
```

GitHub 저장소를 감시합니다(익명 요청의 시간당 60회 제한을 피하려면 `${HERMES_HOME:-~/.hermes}/.env`에 `GITHUB_TOKEN`을 설정하세요).

```bash
python $HERMES_HOME/skills/devops/watchers/scripts/watch_github.py \
  --name hermes-issues --repo NousResearch/hermes-agent --scope issues
```

임의의 JSON API를 폴링합니다.

```bash
python $HERMES_HOME/skills/devops/watchers/scripts/watch_http_json.py \
  --name api --url https://api.example.com/events \
  --id-field event_id --items-path data.events
```

## cron에 연결하기

다음과 같은 프롬프트로 cron 작업을 예약해 달라고 에이전트에 요청하세요.

> 15분마다 `watch_rss.py --name hn --url https://news.ycombinator.com/rss`를 실행하세요. 무언가 출력되면 헤드라인을 요약해 전달하세요. 아무것도 출력되지 않으면 조용히 있으세요.

에이전트는 cron 작업 내부의 에이전트 루프에서 터미널 도구로 스크립트를 실행하므로 cron에 내장된 `--script` 플래그를 변경할 필요가 없습니다.

## 상태 파일

모든 watcher는 `$HERMES_HOME/watcher-state/<name>.json`에 기록합니다. 확인하려면 다음을 실행하세요.

```bash
cat $HERMES_HOME/watcher-state/hn.json
```

강제로 다시 재생하려면(다음 실행을 첫 폴링으로 처리):

```bash
rm $HERMES_HOME/watcher-state/hn.json
```

## 직접 작성하기

세 스크립트는 모두 같은 템플릿을 사용합니다. 워터마크를 로드하고, 가져오고, 차이를 계산하고, 저장하고, 출력합니다. `scripts/_watermark.py`는 공유 헬퍼이며, 이를 가져오면 원자적 쓰기, 제한된 ID 집합, 첫 실행 기준선을 별도의 작업 없이 사용할 수 있습니다. 보일러플레이트가 얼마나 적은지는 세 가지 참고 스크립트 중 하나를 확인하세요.

## 일반적인 실수

1. **매 틱마다 “새 항목 없음” 헤더를 출력하는 것.** 호출자는 빈 stdout을 무음으로 처리합니다. 빈 차이에 무언가를 출력하면 채널에 스팸이 발생합니다. 제공된 스크립트는 이를 처리하지만 사용자 정의 스크립트도 반드시 그래야 합니다.
2. **첫 실행에서 항목이 출력될 것으로 기대하는 것.** 첫 실행은 기준선을 기록할 뿐입니다. 초기 다이제스트가 필요하면 첫 실행 후 상태 파일을 삭제하거나 자체 스크립트에 `--prime-with-latest N` 플래그를 추가하세요.
3. **워터마크를 제한 없이 늘리는 것.** 공유 헬퍼는 ID를 500개로 제한합니다. 변경이 많은 피드에서는 이 값을 높이고, 파일 시스템이 제한된 환경에서는 낮추세요.
4. **에이전트의 샌드박스가 쓸 수 없는 위치에 상태 디렉터리를 두는 것.** `$HERMES_HOME/watcher-state/`는 항상 쓸 수 있습니다. Docker/Modal 백엔드는 임의의 호스트 경로를 보지 못할 수 있습니다.
