---
title: "Powerpoint — python-pptx로 .pptx 프레젠테이션 만들기, 읽기, 편집하기"
sidebar_label: "Powerpoint"
description: "python-pptx로 .pptx 프레젠테이션 만들기, 읽기, 편집하기"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동 생성합니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Powerpoint

python-pptx로 .pptx 프레젠테이션을 만들고, 읽고, 편집합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 번들 제공(기본 설치) |
| 경로 | `skills/productivity/powerpoint` |
| 버전 | `1.0.0` |
| 작성자 | Nous Research |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `pptx`, `powerpoint`, `프레젠테이션`, `슬라이드`, `office`, `python-pptx` |
| 관련 스킬 | [`docx`](/docs/user-guide/skills/bundled/productivity/productivity-docx), [`xlsx`](/docs/user-guide/skills/bundled/productivity/productivity-xlsx), [`pdf`](/docs/user-guide/skills/bundled/productivity/productivity-pdf) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 지침입니다.
:::

# Powerpoint 스킬

`python-pptx` 라이브러리를 사용해 PowerPoint(.pptx) 프레젠테이션을 만들고, 검사하고, 편집합니다. 네 개의 헬퍼 스크립트가 JSON 사양에 따른 프레젠테이션 생성, 구조화된 재확인, 기존 파일 편집, 템플릿 기반 브랜드 프레젠테이션 생성을 지원합니다. 모두 오프라인으로 작동하며 PowerPoint 설치가 필요하지 않습니다.

## 사용 시점

- 사용자가 슬라이드 프레젠테이션, 보고서 프레젠테이션 또는 피치 덱 제작을 요청할 때.
- 다른 사람이 공유한 `.pptx`에서 텍스트, 메모, 표, 차트 데이터 또는 이미지를 추출해야 할 때.
- 기존 프레젠테이션을 업데이트해야 할 때: 텍스트 교체, 차트 데이터 갱신, 로고 변경, 슬라이드 제거 또는 순서 변경.
- 회사의 `.pptx` 템플릿으로 브랜드 스타일에 맞는 프레젠테이션을 만들어야 할 때.
- 레거시 바이너리 파일인 .ppt에는 사용하지 마세요 — LibreOffice를 사용할 수 있다면 먼저 `soffice --convert-to pptx old.ppt`로 변환하세요.

## 사전 요구 사항

- `python-pptx`가 설치된 Python 3.10 이상
  (`pip install python-pptx`). Pillow는 이미지를 직접 확인해야 할 때만 선택적으로 필요합니다.
- 선택 사항: 시각적 검증을 위해 슬라이드를 이미지로 렌더링하는 LibreOffice(`soffice`). 없으면 그에 맞게 처리합니다 — LibreOffice가 없어도 모든 생성/읽기/편집 작업은 작동합니다.
- `terminal`로 사용 가능 여부 확인:
  `python3 -c "import pptx; print(pptx.__version__)"` 및 `which soffice`.

## 실행 방법

모든 스크립트는 `scripts/`에 있으며 `--help`를 받고 JSON을 표준 출력으로 출력하며, 실패 시 0이 아닌 종료 코드를 반환합니다. `terminal`로 실행합니다.

```bash
python3 scripts/pptx_create.py deck.json out.pptx
python3 scripts/pptx_read.py deck.pptx --outline      # full JSON outline
python3 scripts/pptx_read.py deck.pptx --notes        # speaker notes
python3 scripts/pptx_read.py deck.pptx --images ./img # export pictures
python3 scripts/pptx_edit.py deck.pptx --replace-text "Old Corp" "New Corp"
python3 scripts/pptx_edit.py deck.pptx --chart-data update.json
python3 scripts/pptx_edit.py deck.pptx --remove-slide 3 --move-slide 2 0
python3 scripts/pptx_from_template.py brand.pptx out.pptx --values vals.json
```

JSON 사양은 `write_file`로 작성하고, 생성된 JSON과 스크립트 출력을 `read_file`로 검사합니다.

## 빠른 참고

| 작업 | 명령 |
|---|---|
| 사양으로 새 프레젠테이션 만들기 | `pptx_create.py spec.json out.pptx` |
| 16:9와 4:3 | 사양에서 `"slide_size": "16:9"` 또는 `"4:3"` |
| JSON 개요 | `pptx_read.py deck.pptx --outline` |
| 이미지 내보내기 | `pptx_read.py deck.pptx --images DIR` |
| 텍스트 교체 | `pptx_edit.py deck.pptx --replace-text OLD NEW` |
| 차트 업데이트 | `pptx_edit.py deck.pptx --chart-data spec.json` |
| 그림 교체 | `pptx_edit.py deck.pptx --swap-image N NAME new.png` |
| 슬라이드 제거 | `pptx_edit.py deck.pptx --remove-slide N` |
| 슬라이드 순서 변경 | `pptx_edit.py deck.pptx --move-slide FROM TO` |
| 템플릿 채우기 | `pptx_from_template.py tpl.pptx out.pptx --values v.json` |

## 절차

### 1. 프레젠테이션 만들기

JSON 사양을 작성한 다음 `pptx_create.py`를 실행합니다(`pptx_create.py --help`에서 전체 형식 확인). 슬라이드마다 다음을 설정할 수 있습니다. `layout`(title, title_content, section, two_content, title_only, blank), `title`, `subtitle`, `bullets`(문자열 또는 `level` 0-4, `size` pt, `bold`, `italic`, `font`, `color` hex를 포함한 딕셔너리), `images`(인치 단위의 경로 + left/top/width/height), `tables`(`rows`를 리스트의 리스트로 지정), `shapes`(rectangle, rounded_rectangle, oval, diamond, right_arrow, chevron, `fill` hex + 선택적 `text` 포함), `charts`(bar, bar_h, line, pie와 `categories` + `series` 포함), `notes`(발표자 노트).

### 2. 프레젠테이션 읽기

`pptx_read.py deck.pptx --outline`은 슬라이드 크기, 레이아웃 목록, 슬라이드별 다음 정보를 반환합니다. 레이아웃 이름, 모든 도형 텍스트, 표 셀, 이미지 목록(파일명/확장자/바이트), 차트 카테고리/시리즈/값, 발표자 노트입니다.
`--images DIR`을 사용해 포함된 그림을 파일로 덤프한 다음, 내용이 궁금한 내보낸 이미지에 `vision_analyze`를 사용합니다.

### 3. 프레젠테이션 편집

`pptx_edit.py`는 한 번에 여러 작업을 결합합니다. 원본을 유지하려면 `--output`을 사용합니다. 텍스트 교체는 슬라이드 도형, 표 셀 및 노트를 검색합니다. 차트 업데이트는 슬라이드/차트 인덱스와 새 카테고리/시리즈를 지정하는 JSON 사양으로 `chart.replace_data()`를 사용합니다. 그림 교체는 그림의 관계 ID를 새 대상으로 지정하므로 위치와 크기가 유지됩니다. 슬라이드 제거는 관계와 `<p:sldId>` 항목을 삭제하고, 순서 변경은 `<p:sldIdLst>` 안에서 `<p:sldId>` 요소를 이동합니다(python-pptx에는 둘 다를 위한 공개 API가 없으므로 스크립트가 XML 수준에서 작업합니다).

### 4. 템플릿에서 만들기

`pptx_from_template.py`는 브랜드 `.pptx`를 열고 슬라이드/표/노트 전체에서 `{{token}}`을 values JSON의 값으로 교체합니다. 또한 레이아웃 이름 또는 인덱스를 사용해 템플릿 자체의 레이아웃으로 새 슬라이드를 추가할 수 있으므로 마스터의 글꼴과 색상을 상속합니다. 팁: 슬라이드가 전혀 없는 템플릿에서 시작하려면 나중에 `pptx_edit.py --remove-slide`로 기존 슬라이드를 삭제합니다.

### 5. 시각적 검증(선택 사항)

`soffice`가 있으면 슬라이드를 PNG로 렌더링하고 `vision_analyze`로 검사합니다.

```bash
soffice --headless --convert-to png --outdir ./render deck.pptx  # slide 1
soffice --headless --convert-to pdf --outdir ./render deck.pptx  # all slides
```

PNG 내보내기는 첫 번째 슬라이드만 렌더링합니다. 모든 슬라이드는 PDF로 변환한 다음(팝플러를 사용할 수 있다면 `pdftoppm -png render/deck.pdf render/slide`) PDF에서 변환합니다. `soffice`가 없으면 `pptx_read.py`의 JSON 개요를 사용합니다 — 콘텐츠와 구조는 검증하지만 시각적 요소는 검증하지 않습니다.

## 주의 사항

- **실행 분할**: PowerPoint는 맞춤법 검사 및 서식 경계에서 문단 텍스트를 여러 실행으로 분할합니다. `--replace-text`는 일치 항목이 하나의 실행 안에 있을 때 서식을 정확히 유지하지만, 일치 항목이 여러 실행에 걸쳐 있으면 첫 번째 실행의 서식만 사용해 문단을 다시 작성합니다. 중요한 슬라이드는 교체 후 확인하세요.
- **순서 변경은 XML 수준 작업**: python-pptx에는 지원되는 순서 변경 API가 없습니다. `--move-slide`는 `<p:sldIdLst>`를 직접 조작합니다. 일반적인 프레젠테이션에서는 안전하지만, 이후 프레젠테이션을 다시 읽어 확인하세요.
- **프레젠테이션 간 슬라이드 복사는 지원되지 않음** — 레이아웃, 이미지 및 관계를 깊은 복제해야 합니다. 대신 대상 프레젠테이션에서 슬라이드를 다시 만드세요.
- 차트 편집은 전체 데이터 세트를 교체하므로 단일 셀만 패치할 수 없습니다. 시리즈 추가/제거는 가능하지만 차트 **유형** 변경은 불가능합니다.
- 기본 python-pptx 템플릿은 4:3이며, 사양에서 다르게 지정하지 않으면 생성 스크립트가 16:9로 설정합니다. 사용자 지정 템플릿은 고유한 크기를 유지합니다.
- 레이아웃 인덱스는 템플릿마다 다릅니다. 브랜드 템플릿에서는 먼저 레이아웃 이름을 나열하세요: `pptx_read.py template.pptx --outline`(`layouts_available`).
- 빈 레이아웃에서는 `slide.shapes.title`이 None입니다 — 생성 스크립트는 이를 처리하지만, 임시 python-pptx 코드를 작성할 때 기억하세요.
- 사양 파일을 작성할 때는 항상 `encoding="utf-8"`을 전달하세요. `{{city}}`와 같은 토큰은 ASCII가 아닌 값으로 채워질 수 있습니다.

## 검증

1. 생성/편집 후 `pptx_read.py OUT.pptx --outline`을 실행하고 슬라이드 수, 텍스트, 표, 노트 및 차트 값이 의도와 일치하는지 확인합니다.
2. `--images DIR`을 사용한 다음 파일 크기를 확인하면 그림이 포함되었는지 확인할 수 있습니다.
3. 중요한 프레젠테이션은 `soffice`로 렌더링하고(절차 5 참조) 각 슬라이드 이미지를 `vision_analyze`로 검토합니다.
4. 번들 테스트 스위트가 전체 계약입니다:
   `python3 -m pytest tests/ -q`(`python-pptx` 및 `pytest` 필요).
