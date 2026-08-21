---
title: "Dogfood — 웹 앱 탐색 QA: 버그, 증거, 보고서 찾기"
sidebar_label: "Dogfood"
description: "웹 앱 탐색 QA: 버그, 증거, 보고서"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Dogfood

웹 앱의 탐색 QA를 수행하고 버그, 증거, 보고서를 찾습니다.

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 번들됨(기본 설치) |
| 경로 | `skills/software-development/dogfood` |
| 버전 | `1.0.0` |
| 작성자 | Teknium (teknium1), Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `qa`, `testing`, `browser`, `web`, `dogfood` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성 상태일 때 에이전트가 보는 지침이기도 합니다.
:::

# Dogfood: 체계적인 웹 애플리케이션 QA 테스트

## 개요

이 스킬은 브라우저 도구 세트를 사용해 웹 애플리케이션을 체계적으로 탐색 QA하는 과정을 안내합니다. 애플리케이션을 탐색하고, 요소와 상호작용하고, 문제의 증거를 수집하고, 구조화된 버그 보고서를 작성합니다.

## 사전 요구 사항

- 브라우저 도구 세트를 사용할 수 있어야 합니다(`browser_navigate`, `browser_snapshot`, `browser_click`, `browser_type`, `browser_vision`, `browser_console`, `browser_scroll`, `browser_back`, `browser_press`).
- 테스트 대상 URL과 테스트 범위가 사용자에게서 제공되어야 합니다.

## 입력

사용자는 다음을 제공합니다.
1. **대상 URL** — 테스트 진입점
2. **범위** — 집중할 영역/기능(종합 테스트라면 “전체 사이트”)
3. **출력 디렉터리**(선택 사항) — 스크린샷과 보고서를 저장할 위치(기본값: `./dogfood-output`)

## 워크플로

다음 5단계의 체계적인 워크플로를 따릅니다.

### 1단계: 계획

1. 출력 디렉터리 구조를 만듭니다.
<!-- ascii-guard-ignore -->
   ```
   {output_dir}/
   ├── screenshots/       # Evidence screenshots
   └── report.md          # Final report (generated in Phase 5)
   ```
<!-- ascii-guard-ignore-end -->
2. 사용자의 입력을 기준으로 테스트 범위를 파악합니다.
3. 테스트할 페이지와 기능을 계획해 대략적인 사이트맵을 작성합니다.
   - 랜딩/홈 페이지
   - 탐색 링크(헤더, 푸터, 사이드바)
   - 핵심 사용자 흐름(가입, 로그인, 검색, 결제 등)
   - 양식과 상호작용 요소
   - 엣지 케이스(빈 상태, 오류 페이지, 404 등)

### 2단계: 탐색

계획에 포함된 각 페이지 또는 기능에 대해 다음을 수행합니다.

1. 페이지로 **이동**합니다.
   ```
   browser_navigate(url="https://example.com/page")
   ```

2. DOM 구조를 파악하기 위해 **스냅샷을 촬영**합니다.
   ```
   browser_snapshot()
   ```

3. JavaScript 오류를 확인합니다.
   ```
   browser_console(clear=true)
   ```
   모든 탐색 후와 중요한 상호작용 후에 이 작업을 수행합니다. 조용히 발생하는 JS 오류는 가치가 높은 발견입니다.

4. 페이지를 시각적으로 평가하고 상호작용 요소를 식별하기 위해 **주석이 포함된 스크린샷을 촬영**합니다.
   ```
   browser_vision(question="Describe the page layout, identify any visual issues, broken elements, or accessibility concerns", annotate=true)
   ```
   `annotate=true` 플래그는 상호작용 요소 위에 번호가 매겨진 `[N]` 레이블을 표시합니다. 각 `[N]`은 후속 브라우저 명령에서 참조 `@eN`에 대응합니다.

5. **상호작용 요소를 체계적으로 테스트**합니다.
   - 버튼과 링크 클릭: `browser_click(ref="@eN")`
   - 양식 입력: `browser_type(ref="@eN", text="test input")`
   - 키보드 탐색 테스트: `browser_press(key="Tab")`, `browser_press(key="Enter")`
   - 콘텐츠 스크롤: `browser_scroll(direction="down")`
   - 잘못된 입력으로 양식 유효성 검사 테스트
   - 빈 제출 테스트

6. **각 상호작용 후** 다음을 확인합니다.
   - 콘솔 오류: `browser_console()`
   - 시각적 변화: `browser_vision(question="What changed after the interaction?")`
   - 예상 동작과 실제 동작

### 3단계: 증거 수집

발견한 각 문제에 대해 다음을 수행합니다.

1. 문제를 보여주는 스크린샷을 촬영합니다.
   ```
   browser_vision(question="Capture and describe the issue visible on this page", annotate=false)
   ```
   응답의 `screenshot_path`를 저장합니다. 보고서에서 이 경로를 참조합니다.

2. 세부 정보를 기록합니다.
   - 문제가 발생한 URL
   - 재현 단계
   - 예상 동작
   - 실제 동작
   - 콘솔 오류(있는 경우)
   - 스크린샷 경로

3. 이슈 분류 체계를 사용해 문제를 분류합니다(자세한 내용은 `references/issue-taxonomy.md` 참조).
   - 심각도: Critical / High / Medium / Low
   - 범주: Functional / Visual / Accessibility / Console / UX / Content

### 4단계: 분류

1. 수집한 모든 이슈를 검토합니다.
2. 중복을 제거합니다. 서로 다른 위치에서 나타난 동일한 버그라면 합칩니다.
3. 최종 심각도와 범주를 지정합니다.
4. 심각도 순으로 정렬합니다(Critical, High, Medium, Low 순).
5. 경영진 요약을 위해 심각도별 및 범주별 이슈 수를 집계합니다.

### 5단계: 보고

`templates/dogfood-report-template.md`의 템플릿을 사용해 최종 보고서를 생성합니다.

보고서에는 다음이 포함되어야 합니다.
1. **경영진 요약** — 전체 이슈 수, 심각도별 분류, 테스트 범위
2. **이슈별 섹션** — 다음 항목 포함:
   - 이슈 번호와 제목
   - 심각도 및 범주 배지
   - 관찰된 URL
   - 문제 설명
   - 재현 단계
   - 예상 동작과 실제 동작
   - 스크린샷 참조(인라인 이미지에는 `MEDIA:<screenshot_path>` 사용)
   - 관련 콘솔 오류
3. **이슈 요약 표** — 모든 이슈
4. **테스트 메모** — 테스트한 항목, 테스트하지 않은 항목, 차단 요소

보고서를 `{output_dir}/report.md`에 저장합니다.

## 도구 참조

| 도구 | 목적 |
|------|------|
| `browser_navigate` | 페이지로 이동 |
| `browser_snapshot` | DOM 텍스트 스냅샷(접근성 트리) 가져오기 |
| `browser_click` | 참조(`@eN`) 또는 텍스트로 요소 클릭 |
| `browser_type` | 입력 요소에 텍스트 입력 |
| `browser_scroll` | 위/아래로 스크롤 |
| `browser_back` | 브라우저 기록에서 뒤로 이동 |
| `browser_press` | 키보드 키 누르기 |
| `browser_vision` | AI 분석을 포함한 스크린샷 촬영. 요소 레이블에는 `annotate=true` 사용 |
| `browser_console` | JavaScript 콘솔 출력과 오류 가져오기 |

## 팁

- **탐색 후와 중요한 상호작용 후에는 항상 `browser_console()`을 확인하세요.** 조용히 발생하는 JS 오류는 가장 가치 있는 발견 중 하나입니다.
- 스냅샷 참조가 명확하지 않거나 요소 위치를 파악해야 할 때는 `annotate=true`와 함께 `browser_vision`을 사용하세요.
- **유효한 입력과 잘못된 입력을 모두 테스트하세요.** 양식 유효성 검사 버그는 흔합니다.
- **긴 페이지를 스크롤하세요.** 화면 아래 콘텐츠에 렌더링 문제가 있을 수 있습니다.
- **탐색 흐름을 테스트하세요.** 여러 단계로 구성된 프로세스를 처음부터 끝까지 클릭해 보세요.
- **반응형 동작을 확인하세요.** 스크린샷에서 보이는 레이아웃 문제를 기록하세요.
- **엣지 케이스를 잊지 마세요.** 빈 상태, 매우 긴 텍스트, 특수 문자, 빠른 연속 클릭을 테스트하세요.
- 사용자에게 스크린샷을 보고할 때는 인라인 표시가 가능하도록 `MEDIA:<screenshot_path>`를 포함하세요.
