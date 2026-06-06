---
title: "Youtube Content — YouTube 전사본(transcript)을 요약, 스레드, 블로그로 변환"
sidebar_label: "Youtube Content"
description: "YouTube 전사본(transcript)을 요약, 스레드, 블로그로 변환"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Youtube Content

YouTube 전사본을 요약, 스레드, 블로그로 변환.

## 스킬 메타데이터

| | |
|---|---|
| Source | Bundled (기본 설치됨) |
| Path | `skills/media/youtube-content` |
| Platforms | linux, macos, windows |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화될 때 에이전트가 지침으로 보는 내용입니다.
:::

# YouTube Content 도구 (YouTube Content Tool)

## 사용 시기

사용자가 YouTube URL이나 비디오 링크를 공유하거나, 비디오 요약을 요청하거나, 전사본(transcript)을 요청하거나, 임의의 YouTube 비디오에서 콘텐츠를 추출하고 형식을 변경하고자 할 때 사용합니다. 전사본을 구조화된 콘텐츠(챕터, 요약, 스레드, 블로그 게시물)로 변환합니다.

YouTube 비디오에서 전사본을 추출하여 유용한 형식으로 변환합니다.

## 설정 (Setup)

```bash
pip install youtube-transcript-api
```

## 도우미 스크립트 (Helper Script)

`SKILL_DIR`은 이 SKILL.md 파일을 포함하는 디렉토리입니다. 스크립트는 모든 표준 YouTube URL 형식, 짧은 링크(youtu.be), 쇼츠(shorts), 임베드(embeds), 라이브 링크 또는 11자리 원시 비디오 ID를 허용합니다.

```bash
# 메타데이터가 포함된 JSON 출력
python3 SKILL_DIR/scripts/fetch_transcript.py "https://youtube.com/watch?v=VIDEO_ID"

# 일반 텍스트 (추가 처리를 위한 파이프라인에 적합)
python3 SKILL_DIR/scripts/fetch_transcript.py "URL" --text-only

# 타임스탬프 포함
python3 SKILL_DIR/scripts/fetch_transcript.py "URL" --timestamps

# 폴백(fallback) 체인이 있는 특정 언어
python3 SKILL_DIR/scripts/fetch_transcript.py "URL" --language tr,en
```

## 출력 형식 (Output Formats)

전사본을 가져온 후 사용자가 요청한 내용에 따라 서식을 지정합니다:

- **챕터 (Chapters)**: 주제 전환을 기준으로 그룹화하고, 타임스탬프가 있는 챕터 목록 출력
- **요약 (Summary)**: 전체 비디오에 대한 간결한 5-10문장의 개요
- **챕터 요약 (Chapter summaries)**: 각 챕터마다 짧은 문단 형식의 요약 포함
- **스레드 (Thread)**: Twitter/X 스레드 형식 — 번호가 매겨진 게시물들로 각각 280자 미만
- **블로그 게시물 (Blog post)**: 제목, 섹션, 주요 시사점(key takeaways)이 포함된 전체 기사
- **명언 (Quotes)**: 타임스탬프가 포함된 주목할 만한 인용구

### 예시 — 챕터 출력 (Example — Chapters Output)

```
00:00 서론 — 진행자가 문제 상황을 설명하며 시작
03:45 배경 — 이전의 작업들과 기존 솔루션들이 부족한 이유
12:20 핵심 방법 — 제안된 접근 방식에 대한 설명(walkthrough)
24:10 결과 — 벤치마크 비교 및 주요 시사점
31:55 Q&A — 확장성 및 향후 계획에 대한 청중 질문
```

## 워크플로우 (Workflow)

1. `--text-only --timestamps` 옵션을 사용하여 도우미 스크립트로 전사본을 **가져옵니다 (Fetch)**.
2. **검증 (Validate)**: 출력이 비어 있지 않고 예상 언어로 되어 있는지 확인합니다. 비어 있는 경우 `--language` 없이 재시도하여 사용 가능한 전사본을 가져옵니다. 여전히 비어 있다면, 비디오의 전사본 기능이 비활성화되었을 가능성이 있다고 사용자에게 알립니다.
3. **필요시 청크 분할 (Chunk if needed)**: 전사본이 약 5만 자를 초과하는 경우 겹치는 청크(2천 자 겹침을 포함한 약 4만 자)로 분할하고 병합하기 전에 각 청크를 요약합니다.
4. 요청된 출력 형식으로 **변환합니다 (Transform)**. 사용자가 형식을 지정하지 않은 경우 기본적으로 요약을 수행합니다.
5. **확인 (Verify)**: 제시하기 전에 일관성, 올바른 타임스탬프 및 완전성을 확인하기 위해 변환된 출력을 다시 읽습니다.

## 오류 처리 (Error Handling)

- **전사본 비활성화됨**: 사용자에게 알리고, 비디오 페이지에 자막이 제공되는지 확인해보라고 제안합니다.
- **비공개/사용할 수 없는 비디오**: 오류를 전달하고 사용자가 URL을 확인하도록 요청합니다.
- **일치하는 언어 없음**: `--language` 없이 재시도하여 사용 가능한 전사본을 가져온 다음, 사용자에게 실제 언어를 알립니다.
- **종속성 누락**: `pip install youtube-transcript-api`를 실행하고 재시도합니다.
