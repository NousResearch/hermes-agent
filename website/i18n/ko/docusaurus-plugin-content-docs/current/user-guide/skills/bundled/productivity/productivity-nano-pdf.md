---
title: "Nano Pdf — 자연어 프롬프트로 기존 PDF의 텍스트 편집"
sidebar_label: "Nano Pdf"
description: "자연어 프롬프트로 기존 PDF의 텍스트 편집"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Nano Pdf

자연어 프롬프트로 기존 PDF의 텍스트를 편집합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 기본 제공(기본 설치됨) |
| 경로 | `skills/productivity/nano-pdf` |
| 버전 | `1.0.0` |
| 작성자 | community |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `PDF`, `Documents`, `Editing`, `NLP`, `Productivity` |
| 관련 스킬 | [`pdf`](/docs/user-guide/skills/bundled/productivity/productivity-pdf), [`ocr-and-documents`](/docs/user-guide/skills/bundled/productivity/productivity-ocr-and-documents) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성 상태일 때 에이전트가 보게 되는 지침입니다.
:::

# nano-pdf

자연어 지침으로 PDF를 편집합니다. 특정 페이지를 지정하고 무엇을 변경할지 설명하세요. PDF의 구조적 작업(병합, 분할, 양식, 워터마크, 생성)은 `pdf` 스킬을, 스캔본에서 텍스트를 추출하는 작업은 `ocr-and-documents`를 참조하세요.

## 사전 요구 사항

```bash
# Install with uv (recommended — already available in Hermes)
uv pip install nano-pdf

# Or with pip
pip install nano-pdf
```

## 사용법

```bash
nano-pdf edit <file.pdf> <page_number> "<instruction>"
```

## 예시

```bash
# Change a title on page 1
nano-pdf edit deck.pdf 1 "Change the title to 'Q3 Results' and fix the typo in the subtitle"

# Update a date on a specific page
nano-pdf edit report.pdf 3 "Update the date from January to February 2026"

# Fix content
nano-pdf edit contract.pdf 2 "Change the client name from 'Acme Corp' to 'Acme Industries'"
```

## 참고 사항

- 버전에 따라 페이지 번호가 0부터 시작하거나 1부터 시작할 수 있습니다. 잘못된 페이지가 편집되면 ±1을 적용해 다시 시도하세요.
- 항상 편집 후 출력 PDF를 확인하세요(`read_file`을 사용해 파일 크기를 확인하거나 파일을 여세요).
- 이 도구는 내부적으로 LLM을 사용하므로 API 키가 필요합니다(설정은 `nano-pdf --help`를 확인하세요).
- 텍스트 변경에는 잘 작동하지만, 복잡한 레이아웃 수정에는 다른 접근 방식이 필요할 수 있습니다.
