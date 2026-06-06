---
title: "Baoyu Infographic — Infographics: 21 layouts x 21 styles (信息图, 可视化)"
sidebar_label: "Baoyu Infographic"
description: "Infographics: 21 layouts x 21 styles (信息图, 可视化)"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Baoyu Infographic

인포그래픽: 21개 레이아웃 x 21개 스타일 (信息图, 可视化).

## 스킬 메타데이터

| | |
|---|---|
| Source | 번들 (기본 설치) |
| Path | `skills/creative/baoyu-infographic` |
| Version | `1.56.1` |
| Author | 宝玉 (JimLiu) |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `infographic`, `visual-summary`, `creative`, `image-generation` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이 스킬이 활성화되었을 때 에이전트가 지시 사항으로 보는 내용입니다.
:::

# Infographic Generator

Hermes Agent의 도구 생태계에 맞춰 [baoyu-infographic](https://github.com/JimLiu/baoyu-skills)을 변형했습니다.

두 가지 차원: **레이아웃** (정보 구조) × **스타일** (시각적 미학). 어떤 레이아웃이든 어떤 스타일과도 자유롭게 조합할 수 있습니다.

## 사용 시기

사용자가 인포그래픽, 시각적 요약, 정보 그래픽을 생성하도록 요청하거나 "信息图", "可视化", "高密度信息大图"와 같은 용어를 사용할 때 이 스킬을 트리거하세요. 사용자는 콘텐츠(텍스트, 파일 경로, URL 또는 주제)를 제공하고 선택적으로 레이아웃, 스타일, 화면 비율 또는 언어를 지정합니다.

## 옵션

| 옵션 | 값 |
|--------|--------|
| Layout | 21개 옵션 (레이아웃 갤러리 참조), 기본값: bento-grid |
| Style | 21개 옵션 (스타일 갤러리 참조), 기본값: craft-handmade |
| Aspect | 이름: landscape (16:9), portrait (9:16), square (1:1). 사용자 지정: 모든 W:H 비율 (예: 3:4, 4:3, 2.35:1) |
| Language | en, ko, zh, ja 등 |

## 레이아웃 갤러리

| 레이아웃 | 적합한 용도 |
|--------|----------|
| `linear-progression` | 타임라인, 프로세스, 튜토리얼 |
| `binary-comparison` | A vs B, 전후 비교, 장단점 |
| `comparison-matrix` | 다중 요소 비교 |
| `hierarchical-layers` | 피라미드, 우선순위 수준 |
| `tree-branching` | 카테고리, 분류 체계 |
| `hub-spoke` | 관련 항목이 있는 중심 개념 |
| `structural-breakdown` | 분해도, 단면도 |
| `bento-grid` | 여러 주제, 개요 (기본값) |
| `iceberg` | 표면과 숨겨진 측면 비교 |
| `bridge` | 문제-해결 |
| `funnel` | 전환, 필터링 |
| `isometric-map` | 공간적 관계 |
| `dashboard` | 지표, KPIs |
| `periodic-table` | 분류된 모음 |
| `comic-strip` | 서사, 시퀀스 |
| `story-mountain` | 플롯 구조, 긴장감 곡선 |
| `jigsaw` | 상호 연결된 부분들 |
| `venn-diagram` | 중복되는 개념 |
| `winding-roadmap` | 여정, 마일스톤 |
| `circular-flow` | 주기, 반복되는 프로세스 |
| `dense-modules` | 고밀도 모듈, 데이터가 풍부한 가이드 |

전체 정의: `references/layouts/<layout>.md`

## 스타일 갤러리

| 스타일 | 설명 |
|-------|-------------|
| `craft-handmade` | 손그림, 종이 공예 (기본값) |
| `claymation` | 3D 클레이 인형, 스톱모션 |
| `kawaii` | 일본풍 큐티, 파스텔 |
| `storybook-watercolor` | 부드러운 수채화, 기발한 |
| `chalkboard` | 검은 칠판에 분필 |
| `cyberpunk-neon` | 네온 글로우, 미래 지향적 |
| `bold-graphic` | 코믹 스타일, 하프톤 |
| `aged-academia` | 빈티지 과학, 세피아 |
| `corporate-memphis` | 평면 벡터, 생생한 |
| `technical-schematic` | 청사진, 엔지니어링 |
| `origami` | 접은 종이, 기하학적 |
| `pixel-art` | 레트로 8비트 |
| `ui-wireframe` | 그레이스케일 인터페이스 목업 |
| `subway-map` | 대중교통 노선도 |
| `ikea-manual` | 미니멀 라인 아트 |
| `knolling` | 깔끔하게 정리된 플랫레이(flat-lay) |
| `lego-brick` | 장난감 블록 조립 |
| `pop-laboratory` | 청사진 그리드, 좌표 마커, 실험실 정밀도 |
| `morandi-journal` | 손으로 그린 낙서, 따뜻한 모란디 톤 |
| `retro-pop-grid` | 1970년대 레트로 팝아트, 스위스 그리드, 두꺼운 윤곽선 |
| `hand-drawn-edu` | 마카롱 파스텔, 손으로 그린 흔들리는 선, 졸라맨 |

전체 정의: `references/styles/<style>.md`

## 권장 조합

| 콘텐츠 유형 | 레이아웃 + 스타일 |
|--------------|----------------|
| 타임라인/역사 | `linear-progression` + `craft-handmade` |
| 단계별 | `linear-progression` + `ikea-manual` |
| A vs B | `binary-comparison` + `corporate-memphis` |
| 계층 구조 | `hierarchical-layers` + `craft-handmade` |
| 겹침/중복 | `venn-diagram` + `craft-handmade` |
| 전환 | `funnel` + `corporate-memphis` |
| 주기/순환 | `circular-flow` + `craft-handmade` |
| 기술적 | `structural-breakdown` + `technical-schematic` |
| 지표 | `dashboard` + `corporate-memphis` |
| 교육적 | `bento-grid` + `chalkboard` |
| 여정 | `winding-roadmap` + `storybook-watercolor` |
| 카테고리 | `periodic-table` + `bold-graphic` |
| 제품 가이드 | `dense-modules` + `morandi-journal` |
| 기술 가이드 | `dense-modules` + `pop-laboratory` |
| 트렌디 가이드 | `dense-modules` + `retro-pop-grid` |
| 교육 다이어그램 | `hub-spoke` + `hand-drawn-edu` |
| 프로세스 튜토리얼 | `linear-progression` + `hand-drawn-edu` |

기본값: `bento-grid` + `craft-handmade`

## 키워드 단축키

사용자 입력에 이러한 키워드가 포함된 경우, 연결된 레이아웃을 **자동으로 선택**하고 3단계에서 관련 스타일을 최상위 추천으로 제안하세요. 일치하는 키워드에 대해서는 콘텐츠 기반 레이아웃 추론을 건너뜁니다.

단축키에 **Prompt Notes(프롬프트 참고 사항)**가 있는 경우, 생성된 프롬프트(5단계)에 추가 스타일 지침으로 덧붙이세요.

| 사용자 키워드 | 레이아웃 | 권장 스타일 | 기본 비율 | 프롬프트 참고 사항 |
|--------------|--------|--------------------|----------------|--------------|
| 高密度信息大图 / high-density-info | `dense-modules` | `morandi-journal`, `pop-laboratory`, `retro-pop-grid` | portrait | — |
| 信息图 / infographic | `bento-grid` | `craft-handmade` | landscape | 미니멀리스트: 깔끔한 캔버스, 충분한 여백, 복잡한 배경 텍스처 없음. 간단한 만화 요소 및 아이콘만 사용. |

## 출력 구조

<!-- ascii-guard-ignore -->
```
infographic/{topic-slug}/
├── source-{slug}.{ext}
├── analysis.md
├── structured-content.md
├── prompts/infographic.md
└── infographic.png
```
<!-- ascii-guard-ignore-end -->

Slug: 주제에서 추출한 2~4개 단어의 kebab-case. 충돌 시: `-YYYYMMDD-HHMMSS`를 추가합니다.

## 핵심 원칙

- 소스 데이터를 충실히 보존하세요 — 요약하거나 다른 말로 바꾸지 마세요. (단, 출력에 포함하기 전에 **모든 자격 증명, API 키, 토큰 또는 비밀 정보는 제거**하세요.)
- 콘텐츠를 구조화하기 전에 학습 목표를 정의하세요.
- 시각적 소통을 위한 구조화(헤드라인, 레이블, 시각적 요소)를 수행하세요.

## 워크플로우

### 1단계: 콘텐츠 분석

**참조 로드**: 이 스킬에서 `references/analysis-framework.md`를 읽으세요.

1. 소스 콘텐츠 저장 (파일 경로 또는 붙여넣기 → `write_file`을 사용하여 `source.md`로 저장)
   - **백업 규칙**: `source.md`가 존재하는 경우 `source-backup-YYYYMMDD-HHMMSS.md`로 이름 변경
2. 분석: 주제, 데이터 유형, 복잡성, 어조, 대상 독자
3. 소스 언어 및 사용자 언어 감지
4. 사용자 입력에서 디자인 지침 추출
5. 분석 결과를 `analysis.md`에 저장
   - **백업 규칙**: `analysis.md`가 존재하는 경우 `analysis-backup-YYYYMMDD-HHMMSS.md`로 이름 변경

자세한 형식은 `references/analysis-framework.md`를 참조하세요.

### 2단계: 구조화된 콘텐츠 생성 → `structured-content.md`

콘텐츠를 인포그래픽 구조로 변환합니다:
1. 제목 및 학습 목표
2. 다음 항목을 포함한 섹션들: 핵심 개념, 콘텐츠(원문 그대로), 시각적 요소, 텍스트 레이블
3. 데이터 포인트 (모든 통계/인용구는 정확히 복사됨)
4. 사용자의 디자인 지침

**규칙**: Markdown만 사용하세요. 새로운 정보를 추가하지 마세요. 데이터를 충실히 보존하세요. 출력에서 모든 자격 증명이나 비밀 정보를 제거하세요.

자세한 형식은 `references/structured-content-template.md`를 참조하세요.

### 3단계: 조합 추천

**3.1 키워드 단축키 우선 확인**: 사용자 입력이 **키워드 단축키** 테이블의 키워드와 일치하면 관련된 레이아웃을 자동 선택하고 관련 스타일을 최상위 추천으로 우선순위를 매깁니다. 콘텐츠 기반 레이아웃 추론을 건너뜁니다.

**3.2 그렇지 않은 경우**, 다음을 기반으로 3~5개의 레이아웃×스타일 조합을 추천합니다:
- 데이터 구조 → 일치하는 레이아웃
- 콘텐츠 어조 → 일치하는 스타일
- 대상 독자 기대치
- 사용자 디자인 지침

### 4단계: 옵션 확인

`clarify` 도구를 사용하여 옵션을 사용자와 확인합니다. `clarify`는 한 번에 하나의 질문만 처리하므로 가장 중요한 질문부터 물어보세요:

**Q1 — 조합**: 근거와 함께 3개 이상의 레이아웃×스타일 조합을 제시합니다. 사용자에게 하나를 선택하도록 요청합니다.

**Q2 — 비율**: 화면 비율 기본 설정(landscape/portrait/square 또는 사용자 지정 W:H)을 묻습니다.

**Q3 — 언어** (소스 언어와 사용자 언어가 다를 경우에만): 텍스트 콘텐츠에 사용할 언어를 묻습니다.

### 5단계: 프롬프트 생성 → `prompts/infographic.md`

**백업 규칙**: `prompts/infographic.md`가 존재하는 경우 `prompts/infographic-backup-YYYYMMDD-HHMMSS.md`로 이름 변경

**참조 로드**: `references/layouts/<layout>.md`에서 선택한 레이아웃을 읽고 `references/styles/<style>.md`에서 스타일을 읽으세요.

결합:
1. `references/layouts/<layout>.md`의 레이아웃 정의
2. `references/styles/<style>.md`의 스타일 정의
3. `references/base-prompt.md`의 기본 템플릿
4. 2단계의 구조화된 콘텐츠
5. 확인된 언어의 모든 텍스트

`{{ASPECT_RATIO}}`에 대한 **화면 비율 해석**:
- 이름이 지정된 프리셋 → 비율 문자열: landscape→`16:9`, portrait→`9:16`, square→`1:1`
- 사용자 지정 W:H 비율 → 그대로 사용 (예: `3:4`, `4:3`, `2.35:1`)

조립된 프롬프트를 `write_file`을 사용하여 `prompts/infographic.md`에 저장합니다.

### 6단계: 이미지 생성

5단계에서 조립된 프롬프트와 함께 `image_generate` 도구를 사용합니다.

- 화면 비율을 image_generate의 형식에 맞게 매핑: `16:9` → `landscape`, `9:16` → `portrait`, `1:1` → `square`
- 사용자 지정 비율의 경우 가장 가까운 이름의 비율을 선택합니다.
- 실패 시 자동으로 한 번 재시도합니다.
- 결과 이미지 URL/경로를 출력 디렉토리에 저장합니다.

### 7단계: 출력 요약

보고 내용: 주제, 레이아웃, 스타일, 비율, 언어, 출력 경로, 생성된 파일.

## 참조

- `references/analysis-framework.md` — 분석 방법론
- `references/structured-content-template.md` — 콘텐츠 형식
- `references/base-prompt.md` — 프롬프트 템플릿
- `references/layouts/<layout>.md` — 21개 레이아웃 정의
- `references/styles/<style>.md` — 21개 스타일 정의

## 주의 사항 (Pitfalls)

1. **데이터 무결성이 가장 중요합니다** — 소스 통계를 요약, 의역하거나 변경하지 마세요. "73% increase"는 "significant increase"가 아닌 "73% increase"로 유지되어야 합니다.
2. **비밀 정보 제거** — 출력 파일에 포함하기 전에 항상 소스 콘텐츠에서 API 키, 토큰 또는 자격 증명이 있는지 검사하세요.
3. **섹션당 하나의 메시지** — 각 인포그래픽 섹션은 하나의 명확한 개념을 전달해야 합니다. 섹션에 내용을 너무 많이 담으면 가독성이 떨어집니다.
4. **스타일 일관성** — 참조 파일의 스타일 정의는 인포그래픽 전체에 일관되게 적용되어야 합니다. 스타일을 혼합하지 마세요.
5. **image_generate 화면 비율** — 도구는 `landscape`, `portrait`, `square`만 지원합니다. `3:4`와 같은 사용자 지정 비율은 가장 가까운 옵션(이 경우 portrait)으로 매핑해야 합니다.
