---
title: "Powerpoint — 생성, 읽기, 편집"
sidebar_label: "Powerpoint"
description: "생성, 읽기, 편집"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Powerpoint

.pptx 덱, 슬라이드, 노트, 템플릿 생성, 읽기, 편집.

## 스킬 메타데이터

| | |
|---|---|
| Source | Bundled (기본 설치됨) |
| Path | `skills/productivity/powerpoint` |
| License | Proprietary. LICENSE.txt has complete terms |
| Platforms | linux, macos, windows |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화될 때 에이전트가 지침으로 보는 내용입니다.
:::

# Powerpoint 스킬 (Powerpoint Skill)

## 사용 시기

입력, 출력 또는 양쪽 모두에서 .pptx 파일이 관련된 모든 경우에 이 스킬을 사용하세요. 여기에는 슬라이드 덱, 피치 덱 또는 프레젠테이션 만들기; .pptx 파일에서 텍스트 읽기, 파싱 또는 추출하기(추출된 내용이 이메일이나 요약 등 다른 곳에 사용되는 경우 포함); 기존 프레젠테이션 편집, 수정 또는 업데이트하기; 슬라이드 파일 결합 또는 분할하기; 템플릿, 레이아웃, 발표자 노트 또는 댓글 작업이 포함됩니다. 사용자가 "덱(deck)", "슬라이드", "프레젠테이션"을 언급하거나 .pptx 파일 이름을 참조할 때마다 사용자가 콘텐츠로 무엇을 할 계획인지에 상관없이 트리거합니다. .pptx 파일을 열거나, 만들거나, 건드려야 한다면 이 스킬을 사용하세요.

## 빠른 참조 (Quick Reference)

| 작업 | 가이드 |
|------|-------|
| 콘텐츠 읽기/분석 | `python -m markitdown presentation.pptx` |
| 템플릿에서 편집 또는 생성 | [editing.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/productivity/powerpoint/editing.md) 읽기 |
| 처음부터 만들기 | [pptxgenjs.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/productivity/powerpoint/pptxgenjs.md) 읽기 |

---

## 콘텐츠 읽기 (Reading Content)

```bash
# 텍스트 추출
python -m markitdown presentation.pptx

# 시각적 개요
python scripts/thumbnail.py presentation.pptx

# 원본 XML
python scripts/office/unpack.py presentation.pptx unpacked/
```

---

## 편집 워크플로우 (Editing Workflow)

**자세한 내용은 [editing.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/productivity/powerpoint/editing.md)를 읽어보세요.**

1. `thumbnail.py`로 템플릿 분석
2. 압축 풀기(Unpack) → 슬라이드 조작 → 콘텐츠 편집 → 정리(clean) → 압축(pack)

---

## 처음부터 만들기 (Creating from Scratch)

**자세한 내용은 [pptxgenjs.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/productivity/powerpoint/pptxgenjs.md)를 읽어보세요.**

템플릿이나 참조 프레젠테이션이 없을 때 사용하세요.

---

## 디자인 아이디어 (Design Ideas)

**지루한 슬라이드를 만들지 마세요.** 흰색 배경에 단순한 글머리 기호(bullets)는 누구에게도 감동을 주지 않습니다. 각 슬라이드에 대해 이 목록의 아이디어를 고려해 보세요.

### 시작하기 전에 (Before Starting)

- **콘텐츠에 기반한 대담한 색상 팔레트 선택**: 팔레트는 '이 특정 주제'를 위해 디자인된 느낌을 주어야 합니다. 색상을 완전히 다른 프레젠테이션으로 교체해도 여전히 "잘 어울린다"면 충분히 구체적인 선택을 하지 않은 것입니다.
- **평등성보다 지배력(Dominance over equality)**: 한 색상이 우세해야 하며(시각적 비중의 60-70%), 1-2개의 보조 톤과 하나의 날카로운 악센트가 있어야 합니다. 모든 색상에 동일한 비중을 두지 마세요.
- **다크/라이트 대비**: 제목 + 결론 슬라이드는 어두운 배경, 콘텐츠는 밝은 배경("샌드위치" 구조). 또는 고급스러운 느낌을 주려면 전체를 어두운 색상으로 유지하세요.
- **시각적 모티프에 집중**: 하나의 뚜렷한 요소를 선택하고 반복하세요 — 둥근 이미지 프레임, 색상 원 안의 아이콘, 두꺼운 단면 테두리. 모든 슬라이드에 걸쳐 일관되게 사용하세요.

### 색상 팔레트 (Color Palettes)

주제와 어울리는 색상을 선택하세요 — 일반적인 파란색을 기본으로 사용하지 마세요. 다음 팔레트를 영감으로 사용하세요:

| 테마 | 기본 (Primary) | 보조 (Secondary) | 악센트 (Accent) |
|-------|---------|-----------|--------|
| **Midnight Executive** | `1E2761` (네이비) | `CADCFC` (아이스 블루) | `FFFFFF` (화이트) |
| **Forest & Moss** | `2C5F2D` (포레스트) | `97BC62` (모스) | `F5F5F5` (크림) |
| **Coral Energy** | `F96167` (코랄) | `F9E795` (골드) | `2F3C7E` (네이비) |
| **Warm Terracotta** | `B85042` (테라코타) | `E7E8D1` (샌드) | `A7BEAE` (세이지) |
| **Ocean Gradient** | `065A82` (딥 블루) | `1C7293` (틸) | `21295C` (미드나이트) |
| **Charcoal Minimal** | `36454F` (차콜) | `F2F2F2` (오프화이트) | `212121` (블랙) |
| **Teal Trust** | `028090` (틸) | `00A896` (시폼) | `02C39A` (민트) |
| **Berry & Cream** | `6D2E46` (베리) | `A26769` (더스티 로즈) | `ECE2D0` (크림) |
| **Sage Calm** | `84B59F` (세이지) | `69A297` (유칼립투스) | `50808E` (슬레이트) |
| **Cherry Bold** | `990011` (체리) | `FCF6F5` (오프화이트) | `2F3C7E` (네이비) |

### 각 슬라이드에 대하여 (For Each Slide)

**모든 슬라이드에는 시각적 요소가 필요합니다** — 이미지, 차트, 아이콘 또는 도형. 텍스트로만 구성된 슬라이드는 기억에 남지 않습니다.

**레이아웃 옵션:**
- 2단(텍스트는 왼쪽, 일러스트레이션은 오른쪽)
- 아이콘 + 텍스트 행(색상 원 안의 아이콘, 굵은 헤더, 그 아래 설명)
- 2x2 또는 2x3 그리드(한쪽에는 이미지, 다른 쪽에는 콘텐츠 블록 그리드)
- 하프 블리드 이미지(왼쪽 또는 오른쪽 전체를 채움) 위에 콘텐츠 오버레이

**데이터 표시:**
- 큰 통계 콜아웃(60-72pt의 큰 숫자와 그 아래 작은 레이블)
- 비교 칼럼(비포/애프터, 장단점, 나란히 비교 옵션)
- 타임라인 또는 프로세스 흐름(번호가 매겨진 단계, 화살표)

**시각적 디테일:**
- 섹션 헤더 옆에 있는 작은 색상 원 안의 아이콘
- 주요 통계 또는 태그라인을 위한 이탤릭체 악센트 텍스트

### 타이포그래피 (Typography)

**흥미로운 글꼴 조합을 선택하세요** — 기본값으로 Arial을 사용하지 마세요. 개성 있는 헤더 글꼴을 선택하고 깔끔한 본문 글꼴과 짝을 지어 보세요.

| 헤더 글꼴 | 본문 글꼴 |
|-------------|-----------|
| Georgia | Calibri |
| Arial Black | Arial |
| Calibri | Calibri Light |
| Cambria | Calibri |
| Trebuchet MS | Calibri |
| Impact | Arial |
| Palatino | Garamond |
| Consolas | Calibri |

| 요소 | 크기 |
|---------|------|
| 슬라이드 제목 | 36-44pt 볼드 |
| 섹션 헤더 | 20-24pt 볼드 |
| 본문 텍스트 | 14-16pt |
| 캡션 | 10-12pt 뮤트(muted) |

### 간격 (Spacing)

- 최소 0.5인치 여백
- 콘텐츠 블록 간 0.3-0.5인치 간격
- 숨 쉴 공간을 남겨두세요 — 모든 인치를 꽉 채우지 마세요

### 피해야 할 사항 (Common Mistakes)

- **동일한 레이아웃을 반복하지 마세요** — 슬라이드 전체에 걸쳐 칼럼, 카드, 콜아웃을 다양하게 사용하세요.
- **본문 텍스트를 중앙 정렬하지 마세요** — 단락과 목록은 왼쪽 정렬하세요; 제목만 중앙 정렬하세요.
- **크기 대비를 줄이지 마세요** — 14-16pt 본문과 눈에 띄게 차이 나도록 제목은 36pt 이상이어야 합니다.
- **파란색을 기본으로 사용하지 마세요** — 특정 주제를 반영하는 색상을 선택하세요.
- **간격을 무작위로 섞지 마세요** — 0.3인치 또는 0.5인치 간격을 선택하고 일관되게 사용하세요.
- **하나의 슬라이드만 스타일링하고 나머지는 평범하게 두지 마세요** — 전체적으로 헌신하거나 일관되게 단순하게 유지하세요.
- **텍스트 전용 슬라이드를 만들지 마세요** — 이미지, 아이콘, 차트 또는 시각적 요소를 추가하세요; 단순한 제목 + 글머리 기호는 피하세요.
- **텍스트 상자의 패딩을 잊지 마세요** — 선이나 도형을 텍스트 가장자리에 정렬할 때 텍스트 상자에 `margin: 0`을 설정하거나 패딩을 고려하여 도형의 위치를 조정하세요.
- **대비가 낮은 요소를 사용하지 마세요** — 아이콘과 텍스트 모두 배경에 대해 강한 대비가 필요합니다; 밝은 배경에 밝은 텍스트 또는 어두운 배경에 어두운 텍스트는 피하세요.
- **절대 제목 아래에 악센트 선을 사용하지 마세요** — 이는 AI 생성 슬라이드의 전형적인 특징입니다; 대신 여백이나 배경색을 사용하세요.

---

## QA (필수)

**문제가 있다고 가정하세요. 여러분의 임무는 그것을 찾는 것입니다.**

첫 번째 렌더링은 거의 항상 정확하지 않습니다. QA를 확인 단계가 아닌 버그 사냥으로 접근하세요. 첫 검사에서 아무 문제도 발견하지 못했다면, 충분히 열심히 찾지 않은 것입니다.

### 콘텐츠 QA (Content QA)

```bash
python -m markitdown output.pptx
```

누락된 콘텐츠, 오타, 잘못된 순서를 확인합니다.

**템플릿을 사용할 때 남은 자리 표시자 텍스트가 있는지 확인하세요:**

```bash
python -m markitdown output.pptx | grep -iE "xxxx|lorem|ipsum|this.*(page|slide).*layout"
```

grep이 결과를 반환하면, 성공을 선언하기 전에 그것들을 고치세요.

### 시각적 QA (Visual QA)

**⚠️ 하위 에이전트를 사용하세요(USE SUBAGENTS)** — 단 2-3개의 슬라이드라도요. 여러분은 코드를 보고 있었기 때문에 거기에 있는 것이 아니라 예상하는 것을 보게 될 것입니다. 하위 에이전트는 신선한 눈을 가지고 있습니다.

슬라이드를 이미지로 변환한 다음([이미지로 변환](#이미지로-변환) 참조), 이 프롬프트를 사용하세요:

```
이 슬라이드들을 시각적으로 검사하세요. 문제가 있다고 가정하고 — 찾으세요.

다음을 찾으세요:
- 겹치는 요소 (도형을 통과하는 텍스트, 단어를 통과하는 선, 겹쳐진 요소)
- 가장자리/상자 경계에서 잘리거나 넘치는 텍스트
- 한 줄 텍스트를 위해 위치한 장식선인데 제목이 두 줄로 줄바꿈된 경우
- 위에 있는 콘텐츠와 충돌하는 출처 인용이나 바닥글
- 너무 가까운 요소(< 0.3인치 간격) 또는 거의 닿아있는 카드/섹션
- 고르지 않은 간격 (한 곳에는 넓은 빈 공간, 다른 곳은 좁은 공간)
- 슬라이드 가장자리에서 여백 부족 (< 0.5인치)
- 일관되게 정렬되지 않은 칼럼이나 유사한 요소
- 대비가 낮은 텍스트 (예: 크림색 배경에 밝은 회색 텍스트)
- 대비가 낮은 아이콘 (예: 대비되는 원 없이 어두운 배경에 어두운 아이콘)
- 과도한 줄바꿈을 유발하는 너무 좁은 텍스트 상자
- 남은 자리 표시자 콘텐츠

각 슬라이드에 대해 경미하더라도 문제나 우려 사항을 나열하세요.

다음 이미지들을 읽고 분석하세요:
1. /path/to/slide-01.jpg (Expected: [간략한 설명])
2. /path/to/slide-02.jpg (Expected: [간략한 설명])

사소한 것을 포함하여 발견된 모든 문제를 보고하세요.
```

### 검증 루프 (Verification Loop)

1. 슬라이드 생성 → 이미지로 변환 → 검사
2. **발견된 문제 나열** (발견되지 않은 경우 다시 더 비판적으로 살펴보세요)
3. 문제 수정
4. **영향을 받은 슬라이드 재검증** — 하나의 수정이 종종 또 다른 문제를 유발합니다.
5. 전체 패스에서 새로운 문제가 발견되지 않을 때까지 반복합니다.

**최소한 한 번의 수정 및 검증 주기를 완료할 때까지 성공을 선언하지 마세요.**

---

## 이미지로 변환 (Converting to Images)

시각적 검사를 위해 프레젠테이션을 개별 슬라이드 이미지로 변환합니다:

```bash
python scripts/office/soffice.py --headless --convert-to pdf output.pptx
pdftoppm -jpeg -r 150 output.pdf slide
```

이 명령은 `slide-01.jpg`, `slide-02.jpg` 등을 생성합니다.

수정 후 특정 슬라이드를 다시 렌더링하려면:

```bash
pdftoppm -jpeg -r 150 -f N -l N output.pdf slide-fixed
```

---

## 의존성 (Dependencies)

- `pip install "markitdown[pptx]"` - 텍스트 추출
- `pip install Pillow` - 썸네일 그리드
- `npm install -g pptxgenjs` - 처음부터 만들기
- LibreOffice (`soffice`) - PDF 변환 (`scripts/office/soffice.py`를 통해 샌드박스 환경을 위해 자동 구성됨)
- Poppler (`pdftoppm`) - PDF를 이미지로 변환
