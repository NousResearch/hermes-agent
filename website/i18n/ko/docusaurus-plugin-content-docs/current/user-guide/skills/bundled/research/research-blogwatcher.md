---
title: "Blogwatcher — blogwatcher-cli 도구로 블로그 및 RSS/Atom 피드 모니터링"
sidebar_label: "Blogwatcher"
description: "blogwatcher-cli 도구로 블로그 및 RSS/Atom 피드 모니터링"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Blogwatcher

blogwatcher-cli 도구로 블로그와 RSS/Atom 피드를 모니터링합니다.

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 번들 포함(기본 설치) |
| 경로 | `skills/research/blogwatcher` |
| 버전 | `2.0.0` |
| 작성자 | JulienTant(Hyaxia/blogwatcher의 포크) |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `RSS`, `Blogs`, `Feed-Reader`, `Monitoring` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# Blogwatcher

`blogwatcher-cli` 도구로 블로그와 RSS/Atom 피드 업데이트를 추적합니다. 피드 자동 검색, HTML 스크래핑 대체 동작, OPML 가져오기, 읽음/읽지 않음 기사 관리를 지원합니다.

## 설치

한 가지 방법을 선택하세요:

- **Go:** `go install github.com/JulienTant/blogwatcher-cli/cmd/blogwatcher-cli@latest`
- **Docker:** `docker run --rm -v blogwatcher-cli:/data ghcr.io/julientant/blogwatcher-cli`
- **바이너리(Linux amd64):** `curl -sL https://github.com/JulienTant/blogwatcher-cli/releases/latest/download/blogwatcher-cli_linux_amd64.tar.gz | tar xz -C /usr/local/bin blogwatcher-cli`
- **바이너리(Linux arm64):** `curl -sL https://github.com/JulienTant/blogwatcher-cli/releases/latest/download/blogwatcher-cli_linux_arm64.tar.gz | tar xz -C /usr/local/bin blogwatcher-cli`
- **바이너리(macOS Apple Silicon):** `curl -sL https://github.com/JulienTant/blogwatcher-cli/releases/latest/download/blogwatcher-cli_darwin_arm64.tar.gz | tar xz -C /usr/local/bin blogwatcher-cli`
- **바이너리(macOS Intel):** `curl -sL https://github.com/JulienTant/blogwatcher-cli/releases/latest/download/blogwatcher-cli_darwin_amd64.tar.gz | tar xz -C /usr/local/bin blogwatcher-cli`

모든 릴리스: https://github.com/JulienTant/blogwatcher-cli/releases

### 영구 저장소를 사용하는 Docker

기본적으로 데이터베이스는 `~/.blogwatcher-cli/blogwatcher-cli.db`에 저장됩니다. Docker에서는 컨테이너를 다시 시작할 때 데이터가 사라집니다. 이를 유지하려면 `BLOGWATCHER_DB` 또는 볼륨 마운트를 사용하세요:

```bash
# Named volume (simplest)
docker run --rm -v blogwatcher-cli:/data -e BLOGWATCHER_DB=/data/blogwatcher-cli.db ghcr.io/julientant/blogwatcher-cli scan

# Host bind mount
docker run --rm -v /path/on/host:/data -e BLOGWATCHER_DB=/data/blogwatcher-cli.db ghcr.io/julientant/blogwatcher-cli scan
```

### 기존 blogwatcher에서 마이그레이션

`Hyaxia/blogwatcher`에서 업그레이드하는 경우 데이터베이스를 이동하세요:

```bash
mv ~/.blogwatcher/blogwatcher.db ~/.blogwatcher-cli/blogwatcher-cli.db
```

바이너리 이름이 `blogwatcher`에서 `blogwatcher-cli`로 변경되었습니다.

## 일반적인 명령

### 블로그 관리

- 블로그 추가: `blogwatcher-cli add "My Blog" https://example.com`
- 명시적인 피드와 함께 추가: `blogwatcher-cli add "My Blog" https://example.com --feed-url https://example.com/feed.xml`
- 추적 중인 블로그 나열: `blogwatcher-cli blogs`
- 블로그 제거: `blogwatcher-cli remove "My Blog" --yes`
- OPML에서 가져오기: `blogwatcher-cli import subscriptions.opml`

### 검색 및 읽기

- 모든 블로그 검색: `blogwatcher-cli scan`
- 블로그 하나 검색: `blogwatcher-cli scan "My Blog"`
- 읽지 않은 기사 나열: `blogwatcher-cli articles`
- 모든 기사 나열: `blogwatcher-cli articles --all`
- 블로그로 필터링: `blogwatcher-cli articles --blog "My Blog"`
- 카테고리로 필터링: `blogwatcher-cli articles --category "Engineering"`
- 기사 읽음 표시: `blogwatcher-cli read 1`
- 기사 읽지 않음 표시: `blogwatcher-cli unread 1`
- 모두 읽음 표시: `blogwatcher-cli read-all`
- 특정 블로그의 모든 기사를 읽음 표시: `blogwatcher-cli read-all --blog "My Blog" --yes`

## 환경 변수

모든 플래그는 `BLOGWATCHER_` 접두사가 붙은 환경 변수로 설정할 수 있습니다:

| 변수 | 설명 |
|---|---|
| `BLOGWATCHER_DB` | SQLite 데이터베이스 파일 경로 |
| `BLOGWATCHER_WORKERS` | 동시에 검색할 작업자 수(기본값: 8) |
| `BLOGWATCHER_SILENT` | 검색할 때 "scan done"만 출력 |
| `BLOGWATCHER_YES` | 확인 프롬프트 건너뛰기 |
| `BLOGWATCHER_CATEGORY` | 카테고리별 기사의 기본 필터 |

## 출력 예시

```
$ blogwatcher-cli blogs
Tracked blogs (1):

  xkcd
    URL: https://xkcd.com
    Feed: https://xkcd.com/atom.xml
    Last scanned: 2026-04-03 10:30
```

```
$ blogwatcher-cli scan
Scanning 1 blog(s)...

  xkcd
    Source: RSS | Found: 4 | New: 4

Found 4 new article(s) total!
```

```
$ blogwatcher-cli articles
Unread articles (2):

  [1] [new] Barrel - Part 13
       Blog: xkcd
       URL: https://xkcd.com/3095/
       Published: 2026-04-02
       Categories: Comics, Science

  [2] [new] Volcano Fact
       Blog: xkcd
       URL: https://xkcd.com/3094/
       Published: 2026-04-01
       Categories: Comics
```

## 참고 사항

- `--feed-url`을 제공하지 않으면 블로그 홈페이지에서 RSS/Atom 피드를 자동으로 검색합니다.
- RSS가 실패하고 `--scrape-selector`가 구성되어 있으면 HTML 스크래핑으로 대체합니다.
- RSS/Atom 피드의 카테고리를 저장하며 기사 필터링에 사용할 수 있습니다.
- Feedly, Inoreader, NewsBlur 등에서 내보낸 OPML 파일을 사용해 블로그를 일괄 가져옵니다.
- 기본 데이터베이스 위치는 `~/.blogwatcher-cli/blogwatcher-cli.db`입니다(`--db` 또는 `BLOGWATCHER_DB`로 재정의 가능).
- 모든 플래그와 옵션을 확인하려면 `blogwatcher-cli <command> --help`를 사용하세요.
