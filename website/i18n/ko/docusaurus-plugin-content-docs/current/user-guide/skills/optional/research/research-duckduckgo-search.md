---
title: "Duckduckgo Search — DuckDuckGo를 통한 무료 웹 검색 — 텍스트, 뉴스, 이미지, 비디오"
sidebar_label: "Duckduckgo Search"
description: "DuckDuckGo를 통한 무료 웹 검색 — 텍스트, 뉴스, 이미지, 비디오"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Duckduckgo Search

DuckDuckGo를 통한 무료 웹 검색 — 텍스트, 뉴스, 이미지, 비디오. API 키가 필요하지 않습니다. 설치되어 있다면 `ddgs` CLI를 선호합니다. 현재 런타임에서 `ddgs`를 사용할 수 있는지 확인한 후에만 Python DDGS 라이브러리를 사용하세요.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/research/duckduckgo-search` |
| Path | `optional-skills/research/duckduckgo-search` |
| Version | `1.3.0` |
| Author | gamedevCloudy |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `search`, `duckduckgo`, `web-search`, `free`, `fallback` |
| Related skills | [`arxiv`](/docs/user-guide/skills/bundled/research/research-arxiv) |

## Reference: full SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지시 사항으로 보는 내용입니다.
:::

# DuckDuckGo Search

DuckDuckGo를 사용한 무료 웹 검색. **API 키가 필요하지 않습니다.**

`web_search`를 사용할 수 없거나 부적합할 때 (예: `FIRECRAWL_API_KEY`가 설정되지 않은 경우) 선호됩니다. DuckDuckGo 결과를 특별히 원할 때 독립적인 검색 경로로도 사용할 수 있습니다.

## Detection Flow

접근 방식을 선택하기 전에 실제로 사용할 수 있는 것이 무엇인지 확인하세요:

```bash
# CLI 가용성 확인
command -v ddgs >/dev/null && echo "DDGS_CLI=installed" || echo "DDGS_CLI=missing"
```

결정 트리:
1. `ddgs` CLI가 설치되어 있으면 `terminal` + `ddgs`를 선호합니다.
2. `ddgs` CLI가 없으면 `execute_code`가 `ddgs`를 가져올 수 있다고 가정하지 마십시오.
3. 사용자가 특별히 DuckDuckGo를 원할 경우 먼저 관련 환경에 `ddgs`를 설치하십시오.
4. 그렇지 않으면 내장된 웹/브라우저 도구로 폴백하십시오.

중요한 런타임 참고 사항:
- 터미널과 `execute_code`는 별도의 런타임입니다.
- 성공적인 셸 설치가 `execute_code`에서 `ddgs`를 가져올 수 있음을 보장하지는 않습니다.
- 타사 Python 패키지가 `execute_code` 내에 미리 설치되어 있다고 가정하지 마십시오.

## Installation

DuckDuckGo 검색이 특별히 필요하고 런타임에서 아직 제공하지 않을 때만 `ddgs`를 설치하세요.

```bash
# Python 패키지 + CLI 엔트리포인트
pip install ddgs

# CLI 확인
ddgs --help
```

워크플로가 Python 가져오기에 의존하는 경우, `from ddgs import DDGS`를 사용하기 전에 동일한 런타임에서 `ddgs`를 가져올 수 있는지 확인하십시오.

## Method 1: CLI Search (Preferred)

명령어가 존재할 때 `terminal`을 통해 `ddgs` 명령을 사용하십시오. 이것이 선호되는 경로인데, 왜냐하면 `execute_code` 샌드박스에 `ddgs` Python 패키지가 설치되어 있다고 가정하는 것을 피하기 때문입니다.

```bash
# 텍스트 검색
ddgs text -q "python async programming" -m 5

# 뉴스 검색
ddgs news -q "artificial intelligence" -m 5

# 이미지 검색
ddgs images -q "landscape photography" -m 10

# 비디오 검색
ddgs videos -q "python tutorial" -m 5

# 지역 필터 사용
ddgs text -q "best restaurants" -m 5 -r us-en

# 최근 결과만 (d=day, w=week, m=month, y=year)
ddgs text -q "latest AI news" -m 5 -t w

# 구문 분석을 위한 JSON 출력
ddgs text -q "fastapi tutorial" -m 5 -o json
```

### CLI Flags

| Flag | Description | Example |
|------|-------------|---------|
| `-q` | 쿼리 — **필수** | `-q "search terms"` |
| `-m` | 최대 결과 수 | `-m 5` |
| `-r` | 지역 | `-r us-en` |
| `-t` | 시간 제한 | `-t w` (week) |
| `-s` | 세이프 서치 | `-s off` |
| `-o` | 출력 형식 | `-o json` |

## Method 2: Python API (Only After Verification)

`execute_code`나 다른 Python 런타임에서 해당 런타임에 `ddgs`가 설치되어 있는지 확인한 후에만 `DDGS` 클래스를 사용하십시오. `execute_code`에 타사 패키지가 기본적으로 포함되어 있다고 가정하지 마십시오.

안전한 표현:
- "필요한 경우 패키지를 설치하거나 확인한 후 `execute_code`와 함께 `ddgs`를 사용하십시오."

피해야 할 표현:
- "`execute_code`에는 `ddgs`가 포함되어 있습니다."
- "DuckDuckGo 검색은 `execute_code`에서 기본적으로 작동합니다."

**중요:** `max_results`는 항상 **키워드 인수**로 전달해야 합니다. 위치 인수로 사용하면 모든 메서드에서 오류가 발생합니다.

### Text Search

최적 용도: 일반적인 연구, 기업 정보, 문서.

```python
from ddgs import DDGS

with DDGS() as ddgs:
    for r in ddgs.text("python async programming", max_results=5):
        print(r["title"])
        print(r["href"])
        print(r.get("body", "")[:200])
        print()
```

반환: `title`, `href`, `body`

### News Search

최적 용도: 시사, 속보, 최신 업데이트.

```python
from ddgs import DDGS

with DDGS() as ddgs:
    for r in ddgs.news("AI regulation 2026", max_results=5):
        print(r["date"], "-", r["title"])
        print(r.get("source", ""), "|", r["url"])
        print(r.get("body", "")[:200])
        print()
```

반환: `date`, `title`, `body`, `url`, `image`, `source`

### Image Search

최적 용도: 시각적 참고 자료, 제품 이미지, 다이어그램.

```python
from ddgs import DDGS

with DDGS() as ddgs:
    for r in ddgs.images("semiconductor chip", max_results=5):
        print(r["title"])
        print(r["image"])
        print(r.get("thumbnail", ""))
        print(r.get("source", ""))
        print()
```

반환: `title`, `image`, `thumbnail`, `url`, `height`, `width`, `source`

### Video Search

최적 용도: 튜토리얼, 데모, 설명 영상.

```python
from ddgs import DDGS

with DDGS() as ddgs:
    for r in ddgs.videos("FastAPI tutorial", max_results=5):
        print(r["title"])
        print(r.get("content", ""))
        print(r.get("duration", ""))
        print(r.get("provider", ""))
        print(r.get("published", ""))
        print()
```

반환: `title`, `content`, `description`, `duration`, `provider`, `published`, `statistics`, `uploader`

### Quick Reference

| Method | Use When | Key Fields |
|--------|----------|------------|
| `text()` | 일반 연구, 기업 정보 | title, href, body |
| `news()` | 시사, 업데이트 | date, title, source, body, url |
| `images()` | 시각적 자료, 다이어그램 | title, image, thumbnail, url |
| `videos()` | 튜토리얼, 데모 | title, content, duration, provider |

## Workflow: Search then Extract

DuckDuckGo는 전체 페이지 내용이 아닌 제목, URL 및 스니펫을 반환합니다. 전체 페이지 내용을 얻으려면 먼저 검색한 다음 `web_extract`, 브라우저 도구 또는 curl을 사용하여 가장 관련성 높은 URL을 추출하세요.

CLI 예제:

```bash
ddgs text -q "fastapi deployment guide" -m 3 -o json
```

해당 런타임에 `ddgs`가 설치되었는지 확인한 후의 Python 예제:

```python
from ddgs import DDGS

with DDGS() as ddgs:
    results = list(ddgs.text("fastapi deployment guide", max_results=3))
    for r in results:
        print(r["title"], "->", r["href"])
```

그런 다음 `web_extract`나 다른 콘텐츠 검색 도구로 가장 좋은 URL을 추출하세요.

## Limitations

- **속도 제한**: DuckDuckGo는 짧은 시간에 많은 요청이 있을 경우 스로틀링(throttling)할 수 있습니다. 필요하다면 검색 사이에 짧은 지연을 추가하세요.
- **콘텐츠 추출 불가**: `ddgs`는 전체 페이지 내용이 아닌 스니펫을 반환합니다. 전체 기사/페이지를 보려면 `web_extract`, 브라우저 도구 또는 curl을 사용하세요.
- **결과 품질**: 일반적으로 우수하지만 Firecrawl의 검색만큼 구성할 수 없습니다.
- **가용성**: DuckDuckGo는 일부 클라우드 IP의 요청을 차단할 수 있습니다. 검색 결과가 비어 있으면 다른 키워드를 시도하거나 몇 초 기다리세요.
- **필드 변동성**: 반환 필드는 결과나 `ddgs` 버전에 따라 다를 수 있습니다. `KeyError`를 피하려면 선택적 필드에 `.get()`을 사용하세요.
- **별도의 런타임**: 터미널에서 `ddgs` 설치에 성공했다고 해서 `execute_code`가 자동으로 이를 가져올 수 있다는 의미는 아닙니다.

## Troubleshooting

| Problem | Likely Cause | What To Do |
|---------|--------------|------------|
| `ddgs: command not found` | 셸 환경에 CLI가 설치되지 않음 | `ddgs`를 설치하거나 내장된 웹/브라우저 도구를 대신 사용하세요 |
| `ModuleNotFoundError: No module named 'ddgs'` | Python 런타임에 패키지가 설치되지 않음 | 해당 런타임이 준비될 때까지 거기에서 Python DDGS를 사용하지 마세요 |
| Search returns nothing | 일시적인 속도 제한 또는 쿼리 불량 | 몇 초 기다린 후 다시 시도하거나 쿼리를 조정하세요 |
| CLI works but `execute_code` import fails | 터미널과 `execute_code`는 별도의 런타임임 | CLI를 계속 사용하거나 Python 런타임을 따로 준비하세요 |

## Pitfalls

- **`max_results`는 키워드 전용(keyword-only)**입니다. `ddgs.text("query", 5)`는 오류를 발생시킵니다. `ddgs.text("query", max_results=5)`를 사용하세요.
- **CLI가 존재한다고 가정하지 마세요**: 사용하기 전에 `command -v ddgs`를 확인하세요.
- **`execute_code`가 `ddgs`를 가져올 수 있다고 가정하지 마세요**: 런타임이 별도로 준비되지 않으면 `from ddgs import DDGS`가 `ModuleNotFoundError`로 실패할 수 있습니다.
- **패키지 이름**: 패키지 이름은 `ddgs`입니다(이전에는 `duckduckgo-search`). `pip install ddgs`로 설치합니다.
- **`-q`와 `-m`을 혼동하지 마세요** (CLI): `-q`는 쿼리용이고, `-m`은 최대 결과 개수용입니다.
- **빈 결과**: `ddgs`가 아무것도 반환하지 않는다면 속도 제한에 걸린 것일 수 있습니다. 몇 초 기다렸다가 다시 시도하세요.

## Validated With

`ddgs==9.11.2`의 의미론에 대해 검증된 예제들입니다. 스킬 안내서는 문서화된 워크플로가 실제 런타임 동작과 일치하도록 CLI 가용성과 Python 가져오기 가용성을 별개의 문제로 취급합니다.
