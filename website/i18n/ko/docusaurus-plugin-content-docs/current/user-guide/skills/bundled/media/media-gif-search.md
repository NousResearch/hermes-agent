---
title: "Gif Search — curl + jq를 이용해 Tenor에서 GIF 검색/다운로드"
sidebar_label: "Gif Search"
description: "curl + jq를 이용해 Tenor에서 GIF 검색/다운로드"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Gif Search

curl + jq를 이용해 Tenor에서 GIF를 검색/다운로드합니다.

## 스킬 메타데이터 (Skill metadata)

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/media/gif-search` |
| Version | `1.1.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `GIF`, `Media`, `Search`, `Tenor`, `API` |

## 참조: 전체 SKILL.md (Reference: full SKILL.md)

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지시사항으로 보는 내용입니다.
:::

# GIF Search (Tenor API)

추가 도구 없이 curl을 사용하여 Tenor API를 통해 GIF를 직접 검색하고 다운로드합니다.

## 사용 시기 (When to use)

리액션 GIF 찾기, 시각 콘텐츠 제작, 채팅에 GIF 전송 시 유용합니다.

## 설정 (Setup)

환경에 Tenor API 키를 설정합니다(`~/.hermes/.env`에 추가):

```bash
TENOR_API_KEY=your_key_here
```

https://developers.google.com/tenor/guides/quickstart 에서 무료 API 키를 받으세요 — Google Cloud Console Tenor API 키는 무료이며 넉넉한 속도 제한(rate limits)을 제공합니다.

## 전제 조건 (Prerequisites)

- `curl` 및 `jq` (macOS/Linux 표준)
- `TENOR_API_KEY` 환경 변수

## GIF 검색 (Search for GIFs)

```bash
# 검색 및 GIF URL 가져오기
curl -s "https://tenor.googleapis.com/v2/search?q=thumbs+up&limit=5&key=${TENOR_API_KEY}" | jq -r '.results[].media_formats.gif.url'

# 더 작거나 미리보기 버전 가져오기
curl -s "https://tenor.googleapis.com/v2/search?q=nice+work&limit=3&key=${TENOR_API_KEY}" | jq -r '.results[].media_formats.tinygif.url'
```

## GIF 다운로드 (Download a GIF)

```bash
# 최상위 결과 검색 및 다운로드
URL=$(curl -s "https://tenor.googleapis.com/v2/search?q=celebration&limit=1&key=${TENOR_API_KEY}" | jq -r '.results[0].media_formats.gif.url')
curl -sL "$URL" -o celebration.gif
```

## 전체 메타데이터 가져오기 (Get Full Metadata)

```bash
curl -s "https://tenor.googleapis.com/v2/search?q=cat&limit=3&key=${TENOR_API_KEY}" | jq '.results[] | {title: .title, url: .media_formats.gif.url, preview: .media_formats.tinygif.url, dimensions: .media_formats.gif.dims}'
```

## API 파라미터 (API Parameters)

| Parameter | Description |
|-----------|-------------|
| `q` | 검색 쿼리 (공백은 `+`로 URL 인코딩) |
| `limit` | 최대 결과 (1-50, 기본값 20) |
| `key` | API 키 (`$TENOR_API_KEY` 환경 변수에서) |
| `media_filter` | 필터 포맷: `gif`, `tinygif`, `mp4`, `tinymp4`, `webm` |
| `contentfilter` | 안전(Safety): `off`, `low`, `medium`, `high` |
| `locale` | 언어: `en_US`, `es`, `ko`, 등. |

## 사용 가능한 미디어 포맷 (Available Media Formats)

각 결과는 `.media_formats` 아래에 여러 포맷을 갖습니다:

| Format | Use case |
|--------|----------|
| `gif` | 최고 품질 GIF |
| `tinygif` | 작은 미리보기 GIF |
| `mp4` | 비디오 버전 (파일 크기가 더 작음) |
| `tinymp4` | 작은 미리보기 비디오 |
| `webm` | WebM 비디오 |
| `nanogif` | 초소형 썸네일 |

## 참고 (Notes)

- 쿼리 URL 인코딩: 공백은 `+`로, 특수 문자는 `%XX`로 인코딩
- 채팅에서 전송할 때 `tinygif` URL이 더 가볍습니다
- GIF URL은 마크다운에서 직접 사용할 수 있습니다: `![alt](https://github.com/NousResearch/hermes-agent/blob/main/skills/media/gif-search/url)`
