---
title: "DuckDuckGo 검색 — ddgs를 통한 무료 키 없는 웹·뉴스·이미지 검색"
sidebar_label: "DuckDuckGo 검색"
description: "ddgs를 통한 무료 키 없는 웹·뉴스·이미지 검색"
---

{/* 이 페이지는 스킬의 SKILL.md에서 website/scripts/generate-skill-docs.py로 자동 생성됩니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# DuckDuckGo 검색

ddgs를 사용한 무료 키 없는 웹·뉴스·이미지 검색입니다.

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 선택 사항 — `hermes skills install official/research/duckduckgo-search`로 설치 |
| 경로 | `optional-skills/research/duckduckgo-search` |
| 버전 | `1.3.0` |
| 작성자 | gamedevCloudy |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `search`, `duckduckgo`, `web-search`, `free`, `fallback` |
| 관련 스킬 | [`arxiv`](/docs/user-guide/skills/bundled/research/research-arxiv) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 실행될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성화되었을 때 에이전트가 지침으로 확인하는 내용입니다.
:::

# DuckDuckGo 검색

DuckDuckGo를 사용한 무료 웹 검색입니다. **API 키가 필요하지 않습니다.**

`web_search`를 사용할 수 없거나 적합하지 않을 때(예: `FIRECRAWL_API_KEY`가 설정되지 않은 경우) 우선 사용합니다. DuckDuckGo 결과를 특별히 원할 때 독립적인 검색 경로로 사용할 수도 있습니다.

## 감지 흐름

접근 방식을 선택하기 전에 실제로 무엇을 사용할 수 있는지 확인하세요.

```bash
# Check CLI availability
command -v ddgs >/dev/null && echo "DDGS_CLI=installed" || echo "DDGS_CLI=missing"
```

의사 결정 트리:
1. `ddgs` CLI가 설치되어 있으면 `terminal` + `ddgs`를 우선 사용합니다.
2. `ddgs` CLI가 없으면 `execute_code`에서 `ddgs`를 import할 수 있다고 가정하지 마세요.
3. 사용자가 DuckDuckGo를 특별히 원하면 먼저 관련 환경에 `ddgs`를 설치합니다.
4. 그 외에는 기본 제공 웹/브라우저 도구로 대체합니다.

중요한 런타임 참고 사항:
- `terminal`과 `execute_code`는 서로 다른 런타임입니다.
- 셸에서 성공적으로 설치했다고 해서 `execute_code`가 `ddgs`를 import할 수 있다는 보장은 없습니다.
- 서드파티 Python 패키지가 `execute_code`에 미리 설치되어 있다고 가정하지 마세요.

## 설치

DuckDuckGo 검색이 특별히 필요하고 런타임에서 아직 제공되지 않을 때만 `ddgs`를 설치하세요.

```bash
# Python package + CLI entrypoint
pip install ddgs

# Verify CLI
ddgs --help
```

워크플로가 Python import에 의존한다면 `from ddgs import DDGS`를 사용하기 전에 동일한 런타임에서 `ddgs`를 import할 수 있는지 확인하세요.

## 방법 1: CLI 검색(권장)

가능한 경우 `terminal`을 통해 `ddgs` 명령을 사용하세요. 이 방법은 `execute_code` 샌드박스에 `ddgs` Python 패키지가 설치되어 있다고 가정하지 않으므로 권장됩니다.

```bash
# Text search
ddgs text -q "python async programming" -m 5

# News search
ddgs news -q "artificial intelligence" -m 5

# Image search
ddgs images -q "landscape photography" -m 10

# Video search
ddgs videos -q "python tutorial" -m 5

# With region filter
ddgs text -q "best restaurants" -m 5 -r us-en

# Recent results only (d=day, w=week, m=month, y=year)
ddgs text -q "latest AI news" -m 5 -t w

# JSON output for parsing
ddgs text -q "fastapi tutorial" -m 5 -o json
```

### CLI 플래그

| 플래그 | 설명 | 예시 |
|------|-------------|---------|
| `-q` | 검색어 — **필수** | `-q "search terms"` |
| `-m` | 최대 결과 수 | `-m 5` |
| `-r` | 지역 | `-r us-en` |
| `-t` | 시간 제한 | `-t w` (주) |
| `-s` | 세이프서치 | `-s off` |
| `-o` | 출력 형식 | `-o json` |

## 방법 2: Python API(확인 후에만)

`ddgs`가 설치되어 있는지 확인한 후에만 `execute_code` 또는 다른 Python 런타임에서 `DDGS` 클래스를 사용하세요. 기본적으로 `execute_code`에 서드파티 패키지가 포함된다고 가정하지 마세요.

안전한 표현:
- "필요하다면 패키지를 설치하거나 확인한 후 `ddgs`와 함께 `execute_code`를 사용하세요"

다음과 같이 말하지 마세요.
- "`execute_code`에 `ddgs`가 포함되어 있습니다"
- "DuckDuckGo 검색은 `execute_code`에서 기본적으로 작동합니다"

**중요:** `max_results`는 항상 **키워드 인수**로 전달해야 합니다. 모든 메서드에서 위치 인수로 전달하면 오류가 발생합니다.

### 텍스트 검색

적합한 용도: 일반적인 조사, 기업, 문서.

```python
from ddgs import DDGS

with DDGS() as ddgs:
    for r in ddgs.text("python async programming", max_results=5):
        print(r["title"])
        print(r["href"])
        print(r.get("body", "")[:200])
        print()
```

반환값: `title`, `href`, `body`

### 뉴스 검색

적합한 용도: 시사, 속보, 최신 업데이트.

```python
from ddgs import DDGS

with DDGS() as ddgs:
    for r in ddgs.news("AI regulation 2026", max_results=5):
        print(r["date"], "-", r["title"])
        print(r.get("source", ""), "|", r["url"])
        print(r.get("body", "")[:200])
        print()
```

반환값: `date`, `title`, `body`, `url`, `image`, `source`

### 이미지 검색

적합한 용도: 시각적 참고 자료, 제품 이미지, 다이어그램.

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

반환값: `title`, `image`, `thumbnail`, `url`, `height`, `width`, `source`

### 동영상 검색

적합한 용도: 튜토리얼, 데모, 설명 자료.

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

반환값: `title`, `content`, `description`, `duration`, `provider`, `published`, `statistics`, `uploader`

### 빠른 참조

| 메서드 | 사용 시점 | 주요 필드 |
|--------|------------|------------|
| `text()` | 일반적인 조사, 기업 | title, href, body |
| `news()` | 시사, 업데이트 | date, title, source, body, url |
| `images()` | 시각 자료, 다이어그램 | title, image, thumbnail, url |
| `videos()` | 튜토리얼, 데모 | title, content, duration, provider |

## 워크플로: 검색 후 추출

DuckDuckGo는 제목, URL, 스니펫을 반환하지만 전체 페이지 콘텐츠는 반환하지 않습니다. 전체 페이지 콘텐츠를 얻으려면 먼저 검색한 다음 `web_extract`, 브라우저 도구 또는 curl로 가장 관련성 높은 URL을 추출하세요.

CLI 예시:

```bash
ddgs text -q "fastapi deployment guide" -m 3 -o json
```

Python 예시는 해당 런타임에 `ddgs`가 설치되어 있는지 확인한 후에만 사용하세요.

```python
from ddgs import DDGS

with DDGS() as ddgs:
    results = list(ddgs.text("fastapi deployment guide", max_results=3))
    for r in results:
        print(r["title"], "->", r["href"])
```

그런 다음 `web_extract` 또는 다른 콘텐츠 검색 도구로 가장 적합한 URL을 추출하세요.

## 제한 사항

- **속도 제한**: DuckDuckGo는 짧은 시간에 많은 요청을 보내면 요청을 제한할 수 있습니다. 필요하면 검색 사이에 짧은 지연을 추가하세요.
- **콘텐츠 추출 불가**: `ddgs`는 전체 글/페이지가 아니라 스니펫을 반환합니다. 전체 글/페이지에는 `web_extract`, 브라우저 도구 또는 curl을 사용하세요.
- **결과 품질**: 대체로 양호하지만 Firecrawl 검색보다 설정 가능성이 낮습니다.
- **사용 가능성**: DuckDuckGo는 일부 클라우드 IP의 요청을 차단할 수 있습니다. 검색 결과가 비어 있으면 다른 키워드를 사용하거나 몇 초 기다리세요.
- **필드 가변성**: 결과에 포함되는 필드는 결과나 `ddgs` 버전에 따라 달라질 수 있습니다. 선택적 필드는 `.get()`을 사용해 `KeyError`를 방지하세요.
- **분리된 런타임**: 터미널에서 `ddgs` 설치에 성공했다고 해서 `execute_code`에서 자동으로 import할 수 있는 것은 아닙니다.

## 문제 해결

| 문제 | 가능한 원인 | 해결 방법 |
|------------|------------|------------|
| `ddgs: command not found` | 셸 환경에 CLI가 설치되지 않음 | `ddgs`를 설치하거나 기본 제공 웹/브라우저 도구 사용 |
| `ModuleNotFoundError: No module named 'ddgs'` | Python 런타임에 패키지가 설치되지 않음 | 해당 런타임을 준비하기 전까지 Python DDGS를 사용하지 않음 |
| 검색 결과가 없음 | 일시적인 속도 제한 또는 부적절한 검색어 | 몇 초 기다렸다가 재시도하거나 검색어 조정 |
| CLI는 작동하지만 `execute_code` import가 실패함 | 터미널과 `execute_code`가 서로 다른 런타임임 | CLI를 계속 사용하거나 Python 런타임을 별도로 준비 |

## 주의할 점

- **`max_results`는 키워드 전용**: `ddgs.text("query", 5)`는 오류를 발생시킵니다. `ddgs.text("query", max_results=5)`를 사용하세요.
- **CLI가 있다고 가정하지 마세요**: 사용하기 전에 `command -v ddgs`를 확인하세요.
- **`execute_code`에서 `ddgs`를 import할 수 있다고 가정하지 마세요**: 해당 런타임을 별도로 준비하지 않으면 `from ddgs import DDGS`가 `ModuleNotFoundError`로 실패할 수 있습니다.
- **패키지 이름**: 패키지는 `ddgs`입니다(이전 이름은 `duckduckgo-search`). `pip install ddgs`로 설치하세요.
- **`-q`와 `-m`을 혼동하지 마세요**(CLI): `-q`는 검색어, `-m`은 최대 결과 수입니다.
- **빈 결과**: `ddgs`가 아무것도 반환하지 않으면 속도 제한일 수 있습니다. 몇 초 기다렸다가 재시도하세요.

## 검증 정보

`ddgs==9.11.2`의 의미론을 기준으로 예시를 검증했습니다. 문서화된 워크플로가 실제 런타임 동작과 일치하도록 CLI 사용 가능 여부와 Python import 가능 여부를 별개의 관심사로 다룹니다.
