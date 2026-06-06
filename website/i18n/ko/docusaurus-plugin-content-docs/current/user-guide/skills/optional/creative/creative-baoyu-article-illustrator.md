---
title: "Baoyu Article Illustrator — 문서 일러스트: 유형 × 스타일 × 팔레트 일관성"
sidebar_label: "Baoyu Article Illustrator"
description: "문서 일러스트: 유형 × 스타일 × 팔레트 일관성"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Baoyu Article Illustrator

문서 일러스트: 유형 × 스타일 × 팔레트 일관성.

## 스킬 메타데이터

| | |
|---|---|
| Source | Optional — `hermes skills install official/creative/baoyu-article-illustrator`로 설치 |
| Path | `optional-skills/creative/baoyu-article-illustrator` |
| Version | `1.57.0` |
| Author | 宝玉 (JimLiu) |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `article-illustration`, `creative`, `image-generation` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화될 때 에이전트가 지침으로 보는 내용입니다.
:::

# Article Illustrator

Hermes Agent의 도구 생태계를 위해 [baoyu-article-illustrator](https://github.com/JimLiu/baoyu-skills)에서 조정되었습니다.

문서를 분석하고, 일러스트 위치를 파악하며, **유형(Type) × 스타일(Style) × 팔레트(Palette)** 일관성을 갖춘 이미지를 생성합니다.

## 사용 시기

사용자가 문서에 일러스트를 추가하거나, 콘텐츠에 이미지를 생성해 달라고 요청하거나, "为文章配图", "illustrate article", "add images"와 같은 문구를 사용할 때 이 스킬을 트리거하세요. 사용자는 문서(파일 경로 또는 붙여넣은 내용)를 제공하며 선택적으로 유형, 스타일, 팔레트 또는 밀도를 지정합니다.

## 3가지 차원

| 차원 | 제어 항목 | 예시 |
|-----------|----------|----------|
| **유형(Type)** | 정보 구조 | 인포그래픽, 씬, 순서도, 비교, 프레임워크, 타임라인 |
| **스타일(Style)** | 렌더링 방식 | 노션, 따뜻한, 미니멀, 청사진, 수채화, 우아한 |
| **팔레트(Palette)** | 색상 구성 (선택 사항) | 마카롱, 웜, 네온 — 스타일의 기본 색상을 덮어씁니다 |

자유롭게 조합하세요: `type=infographic, style=vector-illustration, palette=macaron`.

또는 프리셋을 사용하세요: `edu-visual` → 유형 + 스타일 + 팔레트를 한 번에 적용합니다. [style-presets.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-article-illustrator/references/style-presets.md) 참조.

## 유형

| 유형 | 적합한 용도 |
|------|----------|
| `infographic` | 데이터, 지표, 기술적 내용 |
| `scene` | 서사, 감정적 내용 |
| `flowchart` | 프로세스, 워크플로우 |
| `comparison` | 나란히 비교, 옵션 |
| `framework` | 모델, 아키텍처 |
| `timeline` | 역사, 진화 |

## 스타일

핵심 스타일, 전체 갤러리 및 유형 × 스타일 호환성에 대해서는 [references/styles.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-article-illustrator/references/styles.md)를 참조하세요.

## 출력 구조

<!-- ascii-guard-ignore -->
```
{output-dir}/
├── source-{slug}.{ext}    # 붙여넣은 콘텐츠 전용
├── outline.md
├── prompts/
│   └── NN-{type}-{slug}.md
└── NN-{type}-{slug}.png
```
<!-- ascii-guard-ignore-end -->

**기본 출력 디렉토리**:

| 입력 | 출력 디렉토리 | 마크다운 삽입 경로 |
|-------|------------------|----------------------|
| 문서 파일 경로 | `{article-dir}/imgs/` | `imgs/NN-{type}-{slug}.png` |
| 붙여넣은 콘텐츠 | `illustrations/{topic-slug}/` (cwd) | `illustrations/{topic-slug}/NN-{type}-{slug}.png` |

사용자가 다른 레이아웃(예: 문서와 같은 위치에 이미지, 또는 `illustrations/` 하위 디렉토리)을 요청하면 이를 존중하세요.

**Slug**: 2-4 단어, kebab-case. **충돌 시**: `-YYYYMMDD-HHMMSS` 추가.

## 핵심 원칙

- **은유가 아닌 개념 시각화** — 문서에서 은유(예: "전기톱으로 수박 자르기")를 사용하는 경우, 문자 그대로의 이미지가 아니라 근본적인 개념을 시각화하세요.
- **레이블에 문서 데이터 사용** — 일반적인 플레이스홀더가 아닌 문서의 실제 숫자, 용어 및 인용문을 사용하세요.
- **프롬프트 파일은 재현성 기록** — 이미지가 생성되기 전에 모든 일러스트는 `prompts/` 아래에 저장된 프롬프트 파일이 있어야 합니다.
- **시크릿 정보 제거** — 디스크에 무언가를 쓰기 전에 소스 콘텐츠에서 API 키, 토큰 또는 자격 증명을 스캔하세요.

## 워크플로우

```
- [ ] 1단계: 참조 이미지 감지 (제공된 경우)
- [ ] 2단계: 콘텐츠 분석
- [ ] 3단계: 설정 확인 (도구 명확화, 한 번에 한 질문씩)
- [ ] 4단계: 개요 생성
- [ ] 5단계: 프롬프트 생성
- [ ] 6단계: 이미지 생성 (image_generate)
- [ ] 7단계: 마무리
```

### 1단계: 참조 이미지 감지

사용자가 참조 이미지를 제공하는 경우 (인라인 경로, 첨부 파일 또는 URL):

1. 각 참조에 대해 경로/URL과 함께 `vision_analyze`를 호출하고 스타일, 팔레트, 구도 및 주제를 묻는 질문을 하세요. `write_file`을 통해 반환된 설명을 `{output-dir}/references/NN-ref-{slug}.md`에 기록합니다.
2. `write_file` / `read_file`을 통해 바이너리를 복사하려고 시도하지 **마세요** — 이들은 텍스트 전용입니다. 기록을 위해 로컬 복사본을 원한다면 `terminal`을 사용하세요 (`cp "$src" "{output-dir}/references/NN-ref-{slug}.{ext}"`). 스킬 자체는 바이너리를 읽을 필요가 없으며 비전 설명을 기반으로 작동합니다.
3. `image_generate`는 이미지 입력을 받지 않으므로, 비전 설명이 5단계의 프롬프트에 포함됩니다.

전체 절차: [references/workflow.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-article-illustrator/references/workflow.md#step-1-detect-reference-images).

### 2단계: 분석

| 분석 항목 | 결과 |
|----------|--------|
| 콘텐츠 유형 | 기술(Technical) / 튜토리얼(Tutorial) / 방법론(Methodology) / 서사(Narrative) |
| 목적 | 정보 전달(information) / 시각화(visualization) / 상상(imagination) |
| 핵심 주장 | 2-5가지 주요 요점 |
| 위치 | 일러스트가 가치를 더할 수 있는 곳 |

소스 (파일 경로 → `read_file`, 또는 붙여넣은 텍스트)를 읽고 `write_file`을 사용하여 분석 결과를 `{output-dir}/analysis.md`에 작성합니다.

전체 절차: [references/workflow.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-article-illustrator/references/workflow.md#step-2-analyze).

### 3단계: 설정 확인

`clarify` 도구를 사용하세요. `clarify`는 한 번에 하나의 질문만 처리하므로 가장 중요한 질문부터 먼저 하세요. 사용자 요청에 이미 답이 있는 질문은 건너뛰세요.

| 순서 | 질문 | 옵션 |
|-------|----------|---------|
| Q1 | **프리셋 또는 유형** | [추천 프리셋], [대체 프리셋], 또는 수동: infographic, scene, flowchart, comparison, framework, timeline, mixed |
| Q2 | **밀도(Density)** | minimal (1-2), balanced (3-5), per-section (추천), rich (6+) |
| Q3 | **스타일(Style)** *(Q1에서 프리셋을 선택한 경우 건너뜀)* | [추천], minimal-flat, sci-fi, hand-drawn, editorial, scene, poster |
| Q4 | **팔레트(Palette)** *(선택 사항)* | 기본값 (스타일 색상), macaron, warm, neon |
| Q5 | **언어(Language)** *(문서 언어가 모호한 경우에만)* | 문서 언어 / 사용자 언어 |

한 번에 2-3개 이상의 `clarify` 질문을 연달아 하지 마세요. 사용자가 요청에 이미 지정한 경우 완전히 건너뛰세요.

전체 절차: [references/workflow.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-article-illustrator/references/workflow.md#step-3-confirm-settings).

### 4단계: 개요 생성 → `outline.md`

`write_file`을 사용하여 Frontmatter (유형, 밀도, 스타일, 팔레트, image_count)와 일러스트당 하나의 항목이 포함된 `{output-dir}/outline.md`를 저장하세요:

```yaml
## Illustration 1
**Position**: [섹션/단락]
**Purpose**: [이유]
**Visual Content**: [무엇을 보여줄지]
**Filename**: 01-infographic-concept-name.png
```

전체 템플릿: [references/workflow.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-article-illustrator/references/workflow.md#step-4-generate-outline).

### 5단계: 프롬프트 생성

**차단 조건(BLOCKING)**: 이미지가 생성되기 전에 모든 일러스트에는 반드시 저장된 프롬프트 파일이 있어야 합니다 — 프롬프트 파일은 재현성 기록입니다.

각 일러스트에 대해:

1. [references/prompt-construction.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-article-illustrator/references/prompt-construction.md)에 따라 프롬프트 파일을 생성합니다.
2. YAML Frontmatter와 함께 `write_file`을 사용하여 `{output-dir}/prompts/NN-{type}-{slug}.md`에 저장합니다.
3. 프롬프트는 구조화된 섹션(ZONES / LABELS / COLORS / STYLE / ASPECT)이 있는 유형별 템플릿을 반드시 사용해야 합니다.
4. LABELS는 실제 숫자, 용어, 지표, 인용문 등 문서별 데이터를 반드시 포함해야 합니다.
5. 프롬프트 Frontmatter에 따라 참조(`direct`/`style`/`palette`)를 처리합니다 — `direct` 사용의 경우 프롬프트에 참조의 텍스트 설명을 삽입합니다(`image_generate`는 참조 이미지 입력을 받지 않기 때문).

### 6단계: 이미지 생성

각 프롬프트 파일에 대해:

1. `image_generate(prompt=..., aspect_ratio=...)`를 호출합니다. `image_generate`는 이미지 URL이 포함된 JSON 결과를 반환하며, 디스크에 쓰지 않고 출력 경로를 허용하지 않습니다.
2. 프롬프트의 `ASPECT`를 `image_generate`의 열거형(enum)에 매핑합니다: `16:9` → `landscape`, `9:16` → `portrait`, `1:1` → `square`. 사용자 지정 비율은 가장 가까운 이름의 비율로 변환합니다.
3. `terminal`을 통해 반환된 URL을 `{output-dir}/NN-{type}-{slug}.png`로 다운로드합니다 (예: `curl -sSL -o "{output-dir}/NN-{type}-{slug}.png" "{url}"`).
4. 생성 실패 시 1회 자동 재시도합니다.

참고: 기본 이미지 생성 백엔드는 사용자가 구성한 것(기본값: FAL FLUX 2 Klein 9B)이며 `image_generate`를 통해 에이전트가 선택할 수 없습니다. 모델 이름이 라우팅될 것을 예상하고 프롬프트에 작성하지 마세요.

### 7단계: 마무리

해당 단락 뒤에 `![description](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-article-illustrator/{relative-path}/NN-{type}-{slug}.png)`를 삽입합니다. Alt 텍스트: 문서 언어로 간결한 설명을 작성합니다.

보고:

```
문서 일러스트 완료!
문서: [경로] | 유형: [유형] | 밀도: [수준] | 스타일: [스타일] | 팔레트: [팔레트 또는 기본값]
이미지: X/N 생성됨
```

## 수정

| 작업 | 단계 |
|--------|-------|
| 편집 | 프롬프트 업데이트 → 재생성 → 참조 업데이트 |
| 추가 | 위치 → 프롬프트 → 생성 → 개요 업데이트 → 삽입 |
| 삭제 | 파일 삭제 → 참조 제거 → 개요 업데이트 |

## 참조 자료

| 파일 | 내용 |
|------|---------|
| [references/workflow.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-article-illustrator/references/workflow.md) | 상세 절차 |
| [references/usage.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-article-illustrator/references/usage.md) | 호출 예시 |
| [references/styles.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-article-illustrator/references/styles.md) | 스타일 갤러리 + 팔레트 갤러리 |
| [references/style-presets.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-article-illustrator/references/style-presets.md) | 프리셋 단축키 (유형 + 스타일 + 팔레트) |
| [references/prompt-construction.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-article-illustrator/references/prompt-construction.md) | 프롬프트 템플릿 |

## 주의 사항

1. **데이터 무결성이 가장 중요합니다** — 소스 통계를 절대 요약, 의역, 변경하지 마세요. "73% 증가"는 "73% 증가"로 유지합니다.
2. **시크릿 정보 제거** — 출력 파일에 포함시키기 전에 소스 콘텐츠에서 API 키, 토큰 또는 자격 증명을 스캔하세요.
3. **은유를 문자 그대로 시각화하지 마세요** — 근본적인 개념을 시각화하세요.
4. **프롬프트 파일은 필수입니다** — 저장된 프롬프트 파일 없이 이미지 생성을 하지 마세요. 이 파일은 나중에 백엔드를 다시 생성하거나 전환할 수 있게 해줍니다.
5. **`image_generate` 화면 비율** — 도구는 `landscape`, `portrait`, `square`를 지원합니다. 사용자 지정 비율은 가장 가까운 옵션에 매핑됩니다.
6. **`image_generate`는 로컬 파일이 아닌 URL을 반환합니다** — 문서에 로컬 이미지 경로를 삽입하기 전에 항상 `terminal`(`curl`)을 통해 다운로드하세요.
7. **에이전트에서 백엔드 선택 불가** — `image_generate`는 사용자가 구성한 모델(기본값: FAL FLUX 2 Klein 9B)을 사용합니다. 라우팅을 기대하고 프롬프트에 `"use <model> to generate this"`를 작성하지 마세요.
