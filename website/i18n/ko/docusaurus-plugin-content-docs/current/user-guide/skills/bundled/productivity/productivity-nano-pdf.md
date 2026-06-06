---
title: "Nano Pdf — nano-pdf CLI를 통한 PDF 텍스트/오타/제목 편집 (자연어 프롬프트)"
sidebar_label: "Nano Pdf"
description: "nano-pdf CLI를 통한 PDF 텍스트/오타/제목 편집 (자연어 프롬프트)"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Nano Pdf

nano-pdf CLI를 통한 PDF 텍스트/오타/제목 편집 (자연어 프롬프트).

## 스킬 메타데이터

| | |
|---|---|
| Source | Bundled (기본 설치됨) |
| Path | `skills/productivity/nano-pdf` |
| Version | `1.0.0` |
| Author | community |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `PDF`, `Documents`, `Editing`, `NLP`, `Productivity` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화될 때 에이전트가 지침으로 보는 내용입니다.
:::

# nano-pdf

자연어 지침을 사용하여 PDF를 편집합니다. 페이지를 지정하고 무엇을 변경할지 설명하세요.

## 사전 요구 사항

```bash
# uv로 설치 (권장 — Hermes에 이미 포함됨)
uv pip install nano-pdf

# 또는 pip로 설치
pip install nano-pdf
```

## 사용법 (Usage)

```bash
nano-pdf edit <file.pdf> <page_number> "<instruction>"
```

## 예시 (Examples)

```bash
# 1페이지의 제목 변경
nano-pdf edit deck.pdf 1 "Change the title to 'Q3 Results' and fix the typo in the subtitle"

# 특정 페이지의 날짜 업데이트
nano-pdf edit report.pdf 3 "Update the date from January to February 2026"

# 콘텐츠 수정
nano-pdf edit contract.pdf 2 "Change the client name from 'Acme Corp' to 'Acme Industries'"
```

## 참고 사항 (Notes)

- 버전에 따라 페이지 번호가 0부터 시작하거나 1부터 시작할 수 있습니다 — 편집 내용이 잘못된 페이지에 적용되는 경우 ±1로 다시 시도하세요.
- 편집 후에는 항상 결과 PDF를 확인하세요 (`read_file`을 사용하여 파일 크기를 확인하거나 파일을 직접 엽니다).
- 이 도구는 백그라운드에서 LLM을 사용하므로 API 키가 필요합니다 (구성을 위해 `nano-pdf --help` 확인).
- 텍스트 변경에 잘 작동하지만 복잡한 레이아웃 수정의 경우 다른 접근 방식이 필요할 수 있습니다.
