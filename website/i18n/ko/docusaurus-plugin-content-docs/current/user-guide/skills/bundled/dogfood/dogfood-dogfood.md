---
title: "Dogfood — 웹 앱의 탐색적 QA: 버그, 증거, 보고서 찾기"
sidebar_label: "Dogfood"
description: "웹 앱의 탐색적 QA: 버그, 증거, 보고서 찾기"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Dogfood

웹 앱의 탐색적 QA: 버그, 증거, 보고서 찾기.

## 스킬 메타데이터

| | |
|---|---|
| Source | Bundled (기본 설치됨) |
| Path | `skills/dogfood` |
| Version | `1.0.0` |
| Platforms | linux, macos, windows |
| Tags | `qa`, `testing`, `browser`, `web`, `dogfood` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화될 때 에이전트가 지침으로 보는 내용입니다.
:::

# Dogfood: 체계적인 웹 애플리케이션 QA 테스트

## 개요

이 스킬은 브라우저 툴셋을 사용하여 웹 애플리케이션의 체계적인 탐색적 QA 테스트 과정을 안내합니다. 애플리케이션을 탐색하고, 요소와 상호 작용하며, 문제의 증거를 캡처하고, 구조화된 버그 보고서를 생성하게 됩니다.

## 사전 요구 사항

- 브라우저 툴셋을 사용할 수 있어야 합니다 (`browser_navigate`, `browser_snapshot`, `browser_click`, `browser_type`, `browser_vision`, `browser_console`, `browser_scroll`, `browser_back`, `browser_press`)
- 사용자가 제공한 대상 URL 및 테스트 범위

## 입력

사용자가 제공하는 정보:
1. **대상 URL (Target URL)** — 테스트의 진입점
2. **범위 (Scope)** — 집중할 영역/기능 (또는 포괄적인 테스트를 위한 "전체 사이트")
3. **출력 디렉토리 (Output directory)** (선택 사항) — 스크린샷 및 보고서를 저장할 위치 (기본값: `./dogfood-output`)

## 워크플로우

다음 5단계의 체계적인 워크플로우를 따르세요:

### 1단계: 계획 (Plan)

1. 출력 디렉토리 구조를 생성합니다:
<!-- ascii-guard-ignore -->
   ```
   {output_dir}/
   ├── screenshots/       # 증거 스크린샷
   └── report.md          # 최종 보고서 (5단계에서 생성됨)
   ```
<!-- ascii-guard-ignore-end -->
2. 사용자 입력을 바탕으로 테스트 범위를 파악합니다.
3. 테스트할 페이지와 기능을 계획하여 대략적인 사이트맵을 구성합니다:
   - 랜딩/홈 페이지
   - 내비게이션 링크 (헤더, 푸터, 사이드바)
   - 주요 사용자 흐름 (회원가입, 로그인, 검색, 결제 등)
   - 폼 및 대화형 요소
   - 엣지 케이스 (빈 상태, 오류 페이지, 404)

### 2단계: 탐색 (Explore)

계획된 각 페이지 또는 기능에 대해 다음을 수행합니다:

1. **탐색 (Navigate)** 페이지로 이동합니다:
   ```
   browser_navigate(url="https://example.com/page")
   ```

2. DOM 구조를 파악하기 위해 **스냅샷을 찍습니다 (Take a snapshot)**:
   ```
   browser_snapshot()
   ```

3. JavaScript 오류가 있는지 **콘솔을 확인합니다 (Check the console)**:
   ```
   browser_console(clear=true)
   ```
   이 작업은 모든 탐색 후와 모든 중요한 상호 작용 후에 수행하세요. 조용한 JS 오류는 매우 가치 있는 발견입니다.

4. 페이지를 시각적으로 평가하고 대화형 요소를 식별하기 위해 **주석이 달린 스크린샷을 찍습니다 (Take an annotated screenshot)**:
   ```
   browser_vision(question="페이지 레이아웃을 설명하고, 시각적 문제, 깨진 요소 또는 접근성 문제를 식별하세요", annotate=true)
   ```
   `annotate=true` 플래그는 대화형 요소 위에 번호가 매겨진 `[N]` 레이블을 오버레이합니다. 각 `[N]`은 이후 브라우저 명령에 사용할 참조 `@eN`에 매핑됩니다.

5. 대화형 요소를 체계적으로 **테스트합니다 (Test interactive elements)**:
   - 버튼 및 링크 클릭: `browser_click(ref="@eN")`
   - 폼 작성: `browser_type(ref="@eN", text="test input")`
   - 키보드 내비게이션 테스트: `browser_press(key="Tab")`, `browser_press(key="Enter")`
   - 콘텐츠 스크롤: `browser_scroll(direction="down")`
   - 잘못된 입력으로 폼 유효성 검사 테스트
   - 빈 폼 제출 테스트

6. **각 상호 작용 후** 다음을 확인합니다:
   - 콘솔 오류: `browser_console()`
   - 시각적 변화: `browser_vision(question="상호 작용 후 무엇이 변경되었습니까?")`
   - 예상된 동작 대 실제 동작

### 3단계: 증거 수집 (Collect Evidence)

발견된 모든 문제에 대해:

1. 문제를 보여주는 **스크린샷을 찍습니다 (Take a screenshot)**:
   ```
   browser_vision(question="이 페이지에 보이는 문제를 캡처하고 설명하세요", annotate=false)
   ```
   응답에서 제공되는 `screenshot_path`를 저장하세요 — 보고서에서 참조하게 됩니다.

2. **세부 정보를 기록합니다 (Record the details)**:
   - 문제가 발생하는 URL
   - 재현 단계 (Steps to reproduce)
   - 예상된 동작 (Expected behavior)
   - 실제 동작 (Actual behavior)
   - 콘솔 오류 (있는 경우)
   - 스크린샷 경로

3. 문제 분류 체계(`references/issue-taxonomy.md` 참조)를 사용하여 **문제를 분류합니다 (Classify the issue)**:
   - 심각도 (Severity): Critical / High / Medium / Low
   - 카테고리 (Category): Functional / Visual / Accessibility / Console / UX / Content

### 4단계: 범주화 (Categorize)

1. 수집된 모든 문제를 검토합니다.
2. 중복 제거 (De-duplicate) — 다른 위치에서 발생하는 동일한 버그인 문제들을 병합합니다.
3. 각 문제에 최종 심각도와 카테고리를 할당합니다.
4. 심각도 기준으로 정렬합니다 (Critical을 먼저, 그다음 High, Medium, Low 순).
5. 요약 보고서를 위해 심각도 및 카테고리별로 문제의 수를 셉니다.

### 5단계: 보고 (Report)

`templates/dogfood-report-template.md`에 있는 템플릿을 사용하여 최종 보고서를 생성합니다.

보고서에는 다음이 포함되어야 합니다:
1. 총 문제 수, 심각도별 분류 및 테스트 범위가 포함된 **요약 보고서 (Executive summary)**
2. 다음이 포함된 **문제별 섹션 (Per-issue sections)**:
   - 문제 번호 및 제목
   - 심각도 및 카테고리 배지
   - 관찰된 URL
   - 문제 설명
   - 재현 단계
   - 예상된 동작 vs 실제 동작
   - 스크린샷 참조 (인라인 이미지에 `MEDIA:<screenshot_path>` 사용)
   - 관련된 경우 콘솔 오류
3. 모든 문제에 대한 **요약 테이블 (Summary table)**
4. **테스트 참고 사항 (Testing notes)** — 테스트된 항목, 테스트되지 않은 항목, 차단 요소(blocker)

보고서를 `{output_dir}/report.md`에 저장합니다.

## 도구 참조 (Tools Reference)

| 도구 | 목적 |
|------|---------|
| `browser_navigate` | URL로 이동 |
| `browser_snapshot` | DOM 텍스트 스냅샷(접근성 트리) 가져오기 |
| `browser_click` | 참조(`@eN`) 또는 텍스트로 요소 클릭 |
| `browser_type` | 입력 필드에 입력 |
| `browser_scroll` | 페이지를 위/아래로 스크롤 |
| `browser_back` | 브라우저 기록에서 뒤로 가기 |
| `browser_press` | 키보드 키 누르기 |
| `browser_vision` | 스크린샷 + AI 분석; 요소 레이블에는 `annotate=true` 사용 |
| `browser_console` | JS 콘솔 출력 및 오류 가져오기 |

## 팁 (Tips)

- **탐색 후 및 중요한 상호 작용 후에 항상 `browser_console()`을 확인하세요.** 조용한 JS 오류는 가장 가치 있는 발견 중 하나입니다.
- 대화형 요소의 위치에 대해 추론해야 하거나 스냅샷 참조가 불분명할 때 **`browser_vision`과 함께 `annotate=true`를 사용하세요**.
- **유효한 입력과 잘못된 입력 모두로 테스트하세요** — 폼 유효성 검사 버그는 흔히 발생합니다.
- **긴 페이지는 스크롤하세요** — 스크롤 아래 콘텐츠에 렌더링 문제가 있을 수 있습니다.
- **내비게이션 흐름을 테스트하세요** — 다단계 프로세스를 끝에서 끝까지 클릭해보세요.
- 스크린샷에 보이는 레이아웃 문제를 기록하여 **반응형 동작을 확인하세요**.
- **엣지 케이스를 잊지 마세요**: 빈 상태, 매우 긴 텍스트, 특수 문자, 빠른 클릭.
- 스크린샷을 사용자에게 보고할 때 사용자가 증거를 인라인으로 볼 수 있도록 `MEDIA:<screenshot_path>`를 포함하세요.
