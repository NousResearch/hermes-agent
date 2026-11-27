---
title: "Gif Search — curl + jq로 Tenor GIF 검색/다운로드"
sidebar_label: "Gif Search"
description: "curl + jq로 Tenor GIF 검색/다운로드"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동 생성합니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Gif Search

curl + jq로 Tenor에서 GIF를 검색/다운로드합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 번들됨(기본 설치) |
| 경로 | `skills/media/gif-search` |
| 버전 | `1.1.0` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `GIF`, `Media`, `Search`, `Tenor`, `API` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 내용입니다.
:::

# GIF Search (Tenor API)

curl을 사용해 Tenor API에서 직접 GIF를 검색하고 다운로드합니다. 추가 도구가 필요하지 않습니다.

## 사용 시점

리액션 GIF를 찾거나, 시각적 콘텐츠를 만들거나, 채팅으로 GIF를 보낼 때 유용합니다.

## 설정

환경에 Tenor API 키를 설정합니다(`${HERMES_HOME:-~/.hermes}/.env`에 추가):

```bash
TENOR_API_KEY=your_key_here
```

https://developers.google.com/tenor/guides/quickstart 에서 무료 API 키를 발급받으세요. Google Cloud Console Tenor API 키는 무료이며 넉넉한 속도 제한을 제공합니다.

## 사전 요구 사항

- `curl` 및 `jq`(macOS/Linux에 모두 기본 설치됨)
- `TENOR_API_KEY` 환경 변수

## GIF 검색

```bash
# Search and get GIF URLs
curl -s "https://tenor.googleapis.com/v2/search?q=thumbs+up&limit=5&key=${TENOR_API_KEY}" | jq -r '.results[].media_formats.gif.url'

# Get smaller/preview versions
curl -s "https://tenor.googleapis.com/v2/search?q=nice+work&limit=3&key=${TENOR_API_KEY}" | jq -r '.results[].media_formats.tinygif.url'
```

## GIF 다운로드

```bash
# Search and download the top result
URL=$(curl -s "https://tenor.googleapis.com/v2/search?q=celebration&limit=1&key=${TENOR_API_KEY}" | jq -r '.results[0].media_formats.gif.url')
curl -sL "$URL" -o celebration.gif
```

## 전체 메타데이터 가져오기

```bash
curl -s "https://tenor.googleapis.com/v2/search?q=cat&limit=3&key=${TENOR_API_KEY}" | jq '.results[] | {title: .title, url: .media_formats.gif.url, preview: .media_formats.tinygif.url, dimensions: .media_formats.gif.dims}'
```

## API 매개변수

| 매개변수 | 설명 |
|-----------|-------------|
| `q` | 검색어(공백은 `+`로 URL 인코딩) |
| `limit` | 최대 결과 수(1-50, 기본값 20) |
| `key` | API 키(`$TENOR_API_KEY` 환경 변수에서 가져옴) |
| `media_filter` | 형식 필터: `gif`, `tinygif`, `mp4`, `tinymp4`, `webm` |
| `contentfilter` | 안전성: `off`, `low`, `medium`, `high` |
| `locale` | 언어: `en_US`, `es`, `fr` 등 |

## 사용 가능한 미디어 형식

각 결과에는 `.media_formats` 아래에 여러 형식이 있습니다.

| 형식 | 사용 사례 |
|--------|----------|
| `gif` | 최고 품질 GIF |
| `tinygif` | 작은 미리보기 GIF |
| `mp4` | 동영상 버전(더 작은 파일 크기) |
| `tinymp4` | 작은 미리보기 동영상 |
| `webm` | WebM 동영상 |
| `nanogif` | 아주 작은 썸네일 |

## 참고

- 검색어를 URL 인코딩하세요. 공백은 `+`, 특수 문자는 `%XX`로 표시합니다.
- 채팅으로 보낼 때는 `tinygif` URL이 더 가볍습니다.
- GIF URL은 마크다운에서 직접 사용할 수 있습니다: `![alt](https://github.com/NousResearch/hermes-agent/blob/main/skills/media/gif-search/url)`
