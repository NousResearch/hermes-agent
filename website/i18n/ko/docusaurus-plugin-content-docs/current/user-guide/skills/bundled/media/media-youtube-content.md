---
title: "Youtube Content — YouTube 트랜스크립트를 요약, 스레드, 블로그로"
sidebar_label: "Youtube Content"
description: "YouTube 트랜스크립트를 요약, 스레드, 블로그로"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Youtube Content

YouTube 트랜스크립트를 요약, 스레드, 블로그로 변환합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | Bundled (기본 설치) |
| 경로 | `skills/media/youtube-content` |
| 버전 | `1.0.0` |
| 작성자 | Teknium (teknium1), Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `YouTube`, `Video`, `Transcripts`, `Media` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 내용입니다.
:::

# YouTube Content 도구

## 사용 시점

사용자가 YouTube URL 또는 동영상 링크를 공유하거나, 동영상 요약을 요청하거나, 트랜스크립트를 요청하거나, YouTube 동영상의 콘텐츠를 추출하고 다른 형식으로 변환하려고 할 때 사용합니다. 트랜스크립트를 구조화된 콘텐츠(챕터, 요약, 스레드, 블로그 글)로 변환합니다.

YouTube 동영상에서 트랜스크립트를 추출하고 유용한 형식으로 변환합니다.

## 설정

도우미 스크립트를 실행하는 동일한 Hermes 관리 환경에 의존성을 설치하려면 `uv`를 사용합니다.

```bash
uv pip install youtube-transcript-api
```

## 도우미 스크립트

`SKILL_DIR`은 이 SKILL.md 파일이 포함된 디렉터리입니다. 스크립트는 표준 YouTube URL 형식, 짧은 링크(youtu.be), 쇼츠, 임베드, 라이브 링크 또는 11자로 된 원시 동영상 ID를 모두 허용합니다.

```bash
# JSON output with metadata
uv run python3 SKILL_DIR/scripts/fetch_transcript.py "https://youtube.com/watch?v=VIDEO_ID"

# Plain text (good for piping into further processing)
uv run python3 SKILL_DIR/scripts/fetch_transcript.py "URL" --text-only

# With timestamps
uv run python3 SKILL_DIR/scripts/fetch_transcript.py "URL" --timestamps

# Specific language with fallback chain
uv run python3 SKILL_DIR/scripts/fetch_transcript.py "URL" --language tr,en
```

## 출력 형식

사용자가 요청한 형식에 따라 트랜스크립트를 변환합니다.

- **Chapters**: 주제 전환을 기준으로 묶어 타임스탬프가 포함된 챕터 목록 출력
- **Summary**: 동영상 전체를 5~10개의 간결한 문장으로 개괄
- **Chapter summaries**: 각 챕터에 짧은 단락 요약을 포함
- **Thread**: Twitter/X 스레드 형식 — 게시물마다 번호를 매기고 280자 미만으로 작성
- **Blog post**: 제목, 섹션, 핵심 요점이 포함된 전체 글
- **Quotes**: 타임스탬프가 포함된 주목할 만한 인용문

### 예시 — 챕터 출력

```
00:00 Introduction — host opens with the problem statement
03:45 Background — prior work and why existing solutions fall short
12:20 Core method — walkthrough of the proposed approach
24:10 Results — benchmark comparisons and key takeaways
31:55 Q&A — audience questions on scalability and next steps
```

## 워크플로

1. **가져오기**: `--text-only --timestamps`를 사용해 `uv run python3`로 도우미 스크립트를 통해 트랜스크립트를 가져옵니다.
2. **검증**: 출력이 비어 있지 않고 예상 언어인지 확인합니다. 비어 있으면 `--language` 없이 다시 시도해 사용 가능한 트랜스크립트를 가져옵니다. 그래도 비어 있으면 동영상에서 트랜스크립트가 비활성화되었을 가능성이 있다고 사용자에게 알립니다.
3. **필요한 경우 청크로 나누기**: 트랜스크립트가 약 50K자를 초과하면 약 40K자씩 2K자가 겹치도록 나누고, 각 청크를 요약한 뒤 병합합니다.
4. **변환**: 사용자가 요청한 출력 형식으로 변환합니다. 형식을 지정하지 않았다면 기본적으로 요약을 사용합니다.
5. **확인**: 변환된 출력을 다시 읽어 일관성, 올바른 타임스탬프, 완전성을 확인한 후 제시합니다.

## 오류 처리

- **트랜스크립트 비활성화**: 사용자에게 알리고, 동영상 페이지에서 자막을 사용할 수 있는지 확인하도록 안내합니다.
- **비공개/사용할 수 없는 동영상**: 오류를 전달하고 URL을 확인하도록 요청합니다.
- **일치하는 언어 없음**: `--language` 없이 다시 시도해 사용 가능한 언어의 트랜스크립트를 가져온 다음 실제 언어를 사용자에게 알립니다.
- **의존성 누락**: `uv pip install youtube-transcript-api`를 실행하고 다시 시도합니다.
