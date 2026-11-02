---
title: "Baoyu 인포그래픽 — 인포그래픽: 21가지 레이아웃 x 21가지 스타일 (정보 그래픽, 시각화)"
sidebar_label: "Baoyu 인포그래픽"
description: "인포그래픽: 21가지 레이아웃 x 21가지 스타일 (정보 그래픽, 시각화)"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Baoyu 인포그래픽

인포그래픽: 21가지 레이아웃 x 21가지 스타일 (정보 그래픽, 시각화).

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 번들됨 (기본 설치) |
| 경로 | `skills/creative/baoyu-infographic` |
| 버전 | `1.56.1` |
| 작성자 | 宝玉 (JimLiu) |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `infographic`, `visual-summary`, `creative`, `image-generation` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성 상태일 때 에이전트가 보게 되는 지침입니다.
:::

# 인포그래픽 생성기

Hermes Agent의 도구 생태계에 맞게 조정한 [baoyu-infographic](https://github.com/JimLiu/baoyu-skills)입니다.

두 가지 차원: **레이아웃**(정보 구조) × **스타일**(시각적 미학). 모든 레이아웃과 스타일을 자유롭게 조합할 수 있습니다.

## 사용 시점

사용자가 인포그래픽, 시각적 요약, 정보 그래픽을 만들어 달라고 요청하거나 "정보 그래픽", "시각화", "고밀도 정보 대형 이미지"와 같은 용어를 사용할 때 이 스킬을 실행합니다. 사용자는 콘텐츠(텍스트, 파일 경로, URL 또는 주제)를 제공하며, 레이아웃, 스타일, 화면 비율 또는 언어를 선택적으로 지정할 수 있습니다.

## 옵션

| 옵션 | 값 |
|--------|--------|
| 레이아웃 | 21가지 옵션 (레이아웃 갤러리 참조), 기본값: bento-grid |
| 스타일 | 21가지 옵션 (스타일 갤러리 참조), 기본값: craft-handmade |
| 화면 비율 | 이름 지정: landscape (16:9), portrait (9:16), square (1:1). 사용자 지정: 임의의 W:H 비율 (예: 3:4, 4:3, 2.35:1) |
| 언어 | en, zh, ja 등 |

## 레이아웃 갤러리

| 레이아웃 | 적합한 용도 |
|--------|----------|
| `linear-progression` | 타임라인, 프로세스, 튜토리얼 |
| `binary-comparison` | A 대 B, 전후, 장단점 |
| `comparison-matrix` | 다요소 비교 |
| `hierarchical-layers` | 피라미드, 우선순위 수준 |
| `tree-branching` | 카테고리, 분류 체계 |
| `hub-spoke` | 관련 항목이 있는 중심 개념 |
| `structural-breakdown` | 분해도, 단면도 |
| `bento-grid` | 여러 주제, 개요 (기본값) |
| `iceberg` | 표면과 숨겨진 측면 |
| `bridge` | 문제-해결 |
| `funnel` | 전환, 필터링 |
| `isometric-map` | 공간 관계 |
| `dashboard` | 지표, KPI |
| `periodic-table` | 분류된 모음 |
| `comic-strip` | 서사, 순서 |
| `story-mountain` | 플롯 구조, 긴장 곡선 |
| `jigsaw` | 상호 연결된 부분 |
| `venn-diagram` | 겹치는 개념 |
| `winding-roadmap` | 여정, 마일스톤 |
| `circular-flow` | 순환, 반복 프로세스 |
| `dense-modules` | 고밀도 모듈, 데이터가 풍부한 가이드 |

전체 정의: `references/layouts/<layout>.md`

## 스타일 갤러리

| 스타일 | 설명 |
|-------|-------------|
| `craft-handmade` | 손그림, 종이 공예 (기본값) |
| `claymation` | 3D 점토 인형, 스톱 모션 |
| `kawaii` | 일본식 귀여움, 파스텔 |
| `storybook-watercolor` | 부드러운 수채화, 몽환적 분위기 |
| `chalkboard` | 칠판 위 분필 |
| `cyberpunk-neon` | 네온 발광, 미래적 분위기 |
| `bold-graphic` | 만화 스타일, 하프톤 |
| `aged-academia` | 빈티지 과학, 세피아 |
| `corporate-memphis` | 평면 벡터, 선명한 색상 |
| `technical-schematic` | 청사진, 공학 |
| `origami` | 접은 종이, 기하학적 형태 |
| `pixel-art` | 복고풍 8비트 |
| `ui-wireframe` | 회색조 인터페이스 목업 |
| `subway-map` | 교통 노선도 |
| `ikea-manual` | 미니멀 선화 |
| `knolling` | 정돈된 플랫 레이 |
| `lego-brick` | 장난감 블록 구조 |
| `pop-laboratory` | 청사진 격자, 좌표 표식, 실험실 수준의 정밀함 |
| `morandi-journal` | 손그림 낙서, 따뜻한 모란디 색조 |
| `retro-pop-grid` | 1970년대 복고 팝아트, 스위스 그리드, 굵은 윤곽선 |
| `hand-drawn-edu` | 마카롱 파스텔, 손그림의 흔들림, 막대 인물 |

전체 정의: `references/styles/<style>.md`

## 권장 조합

| 콘텐츠 유형 | 레이아웃 + 스타일 |
|--------------|----------------|
| 타임라인/역사 | `linear-progression` + `craft-handmade` |
| 단계별 안내 | `linear-progression` + `ikea-manual` |
| A 대 B | `binary-comparison` + `corporate-memphis` |
| 계층 구조 | `hierarchical-layers` + `craft-handmade` |
| 중첩 | `venn-diagram` + `craft-handmade` |
| 전환 | `funnel` + `corporate-memphis` |
| 순환 | `circular-flow` + `craft-handmade` |
| 기술 | `structural-breakdown` + `technical-schematic` |
| 지표 | `dashboard` + `corporate-memphis` |
| 교육 | `bento-grid` + `chalkboard` |
| 여정 | `winding-roadmap` + `storybook-watercolor` |
| 카테고리 | `periodic-table` + `bold-graphic` |
| 제품 가이드 | `dense-modules` + `morandi-journal` |
| 기술 가이드 | `dense-modules` + `pop-laboratory` |
| 트렌디한 가이드 | `dense-modules` + `retro-pop-grid` |
| 교육용 다이어그램 | `hub-spoke` + `hand-drawn-edu` |
| 프로세스 튜토리얼 | `linear-progression` + `hand-drawn-edu` |

기본값: `bento-grid` + `craft-handmade`

## 키워드 단축키

사용자 입력에 다음 키워드가 포함되면 연결된 레이아웃을 자동으로 선택하고 3단계에서 연결된 스타일을 우선 추천합니다. 일치하는 키워드가 있으면 콘텐츠 기반 레이아웃 추론을 건너뜁니다.

단축키에 **프롬프트 참고 사항**이 있으면 생성된 프롬프트(5단계)에 추가 스타일 지침으로 덧붙입니다.

| 사용자 키워드 | 레이아웃 | 권장 스타일 | 기본 화면 비율 | 프롬프트 참고 사항 |
|--------------|----------|--------------|----------------|--------------|
| 고밀도 정보 대형 이미지 / high-density-info | `dense-modules` | `morandi-journal`, `pop-laboratory`, `retro-pop-grid` | portrait | — |
| 정보 그래픽 / infographic | `bento-grid` | `craft-handmade` | landscape | 미니멀리즘: 깔끔한 캔버스, 넉넉한 여백, 복잡한 배경 질감 없음. 단순한 만화 요소와 아이콘만 사용. |

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

슬러그: 주제에서 가져온 2~4단어의 kebab-case. 충돌 시 `-YYYYMMDD-HHMMSS`를 덧붙입니다.

## 핵심 원칙

- 원본 데이터를 충실하게 보존 — 출력에 포함하기 전에 자격 증명, API 키, 토큰 또는 비밀 정보를 **제거**합니다 (요약하거나 바꿔 쓰지 않음).
- 콘텐츠를 구조화하기 전에 학습 목표를 정의합니다.
- 시각적 전달을 위해 구조화합니다 (헤드라인, 라벨, 시각 요소).

## 워크플로

### 1단계: 콘텐츠 분석

**참고 자료 로드**: 이 스킬에서 `references/analysis-framework.md`를 읽습니다.

1. 원본 콘텐츠를 저장합니다 (파일 경로 또는 붙여넣기 → `write_file`을 사용해 `source.md`에 저장).
   - **백업 규칙**: `source.md`가 있으면 `source-backup-YYYYMMDD-HHMMSS.md`로 이름을 변경합니다.
2. 분석합니다: 주제, 데이터 유형, 복잡도, 어조, 대상 독자
3. 원본 언어와 사용자 언어를 감지합니다.
4. 사용자 입력에서 디자인 지침을 추출합니다.
5. 분석을 `analysis.md`에 저장합니다.
   - **백업 규칙**: `analysis.md`가 있으면 `analysis-backup-YYYYMMDD-HHMMSS.md`로 이름을 변경합니다.

자세한 형식은 `references/analysis-framework.md`를 참조하세요.

### 2단계: 구조화된 콘텐츠 생성 → `structured-content.md`

콘텐츠를 인포그래픽 구조로 변환합니다.
1. 제목과 학습 목표
2. 다음 항목을 포함하는 섹션: 핵심 개념, 콘텐츠 (원문 그대로), 시각 요소, 텍스트 라벨
3. 데이터 포인트 (모든 통계/인용문을 정확히 복사)
4. 사용자의 디자인 지침

**규칙**: Markdown만 사용합니다. 새로운 정보를 추가하지 않습니다. 데이터를 충실하게 보존합니다. 출력에서 자격 증명과 비밀 정보를 제거합니다.

자세한 형식은 `references/structured-content-template.md`를 참조하세요.

### 3단계: 조합 추천

**3.1 먼저 키워드 단축키 확인**: 사용자 입력이 **키워드 단축키** 표의 키워드와 일치하면 연결된 레이아웃을 자동으로 선택하고 연결된 스타일을 최우선 추천으로 지정합니다. 콘텐츠 기반 레이아웃 추론은 건너뜁니다.

**3.2 그렇지 않으면**, 다음을 기준으로 3~5개의 레이아웃×스타일 조합을 추천합니다.
- 데이터 구조 → 일치하는 레이아웃
- 콘텐츠 어조 → 일치하는 스타일
- 대상 독자의 기대
- 사용자의 디자인 지침

### 4단계: 옵션 확인

`clarify` 도구를 사용해 사용자와 옵션을 확인합니다. `clarify`는 한 번에 하나의 질문만 처리하므로 가장 중요한 질문부터 합니다.

**Q1 — 조합**: 3개 이상의 레이아웃×스타일 조합과 그 근거를 제시합니다. 하나를 선택하도록 요청합니다.

**Q2 — 화면 비율**: 화면 비율 선호도(landscape/portrait/square 또는 사용자 지정 W:H)를 묻습니다.

**Q3 — 언어** (원본 언어 ≠ 사용자 언어인 경우에만): 텍스트 콘텐츠에 사용할 언어를 묻습니다.

### 5단계: 프롬프트 생성 → `prompts/infographic.md`

**백업 규칙**: `prompts/infographic.md`가 있으면 `prompts/infographic-backup-YYYYMMDD-HHMMSS.md`로 이름을 변경합니다.

**참고 자료 로드**: 선택한 레이아웃을 `references/layouts/<layout>.md`에서, 스타일을 `references/styles/<style>.md`에서 읽습니다.

다음을 결합합니다.
1. `references/layouts/<layout>.md`의 레이아웃 정의
2. `references/styles/<style>.md`의 스타일 정의
3. `references/base-prompt.md`의 기본 템플릿
4. 2단계의 구조화된 콘텐츠
5. 확인된 언어로 작성된 모든 텍스트

`{{ASPECT_RATIO}}`의 **화면 비율 결정**:
- 이름이 지정된 프리셋 → 비율 문자열: landscape→`16:9`, portrait→`9:16`, square→`1:1`
- 사용자 지정 W:H 비율 → 그대로 사용 (예: `3:4`, `4:3`, `2.35:1`)

`write_file`을 사용해 조합된 프롬프트를 `prompts/infographic.md`에 저장합니다.

### 6단계: 이미지 생성

5단계에서 조합한 프롬프트와 함께 `image_generate` 도구를 사용합니다.

- 화면 비율을 image_generate의 형식에 매핑합니다: `16:9` → `landscape`, `9:16` → `portrait`, `1:1` → `square`
- 사용자 지정 비율은 가장 가까운 이름 지정 화면 비율을 선택합니다.
- 실패하면 한 번 자동으로 재시도합니다.
- 결과 이미지 URL/경로를 출력 디렉터리에 저장합니다.

### 7단계: 출력 요약

주제, 레이아웃, 스타일, 화면 비율, 언어, 출력 경로, 생성된 파일을 보고합니다.

## 참고 자료

- `references/analysis-framework.md` — 분석 방법론
- `references/structured-content-template.md` — 콘텐츠 형식
- `references/base-prompt.md` — 프롬프트 템플릿
- `references/layouts/<layout>.md` — 21가지 레이아웃 정의
- `references/styles/<style>.md` — 21가지 스타일 정의

## 주의 사항

1. **데이터 무결성이 가장 중요합니다** — 통계를 요약하거나 바꿔 쓰거나 변경하지 않습니다. "73% increase"는 "significant increase"가 아니라 반드시 "73% increase"로 유지해야 합니다.
2. **비밀 정보 제거** — 출력 파일에 포함하기 전에 원본 콘텐츠에서 API 키, 토큰 또는 자격 증명을 항상 검사합니다.
3. **섹션당 하나의 메시지** — 각 인포그래픽 섹션은 하나의 명확한 개념을 전달해야 합니다. 섹션을 과도하게 채우면 가독성이 떨어집니다.
4. **스타일 일관성** — 참고 자료 파일의 스타일 정의를 전체에 일관되게 적용합니다. 스타일을 섞지 않습니다.
5. **image_generate 화면 비율** — 이 도구는 `landscape`, `portrait`, `square`만 지원합니다. `3:4`와 같은 사용자 지정 비율은 가장 가까운 옵션(이 경우 portrait)으로 매핑해야 합니다.
