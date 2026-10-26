---
title: "Docx — Word .docx 파일 생성, 읽기, 편집 및 템플릿 작성"
sidebar_label: "Docx"
description: "Word .docx 파일 생성, 읽기, 편집 및 템플릿 작성"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동으로 생성합니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Docx

Word .docx 파일을 생성하고, 읽고, 편집하고, 템플릿을 작성합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 번들 제공(기본 설치) |
| 경로 | `skills/productivity/docx` |
| 버전 | `1.0.0` |
| 작성자 | Nous Research |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `word`, `docx`, `documents`, `office`, `templates` |
| 관련 스킬 | [`pdf`](/docs/user-guide/skills/bundled/productivity/productivity-pdf), [`xlsx`](/docs/user-guide/skills/bundled/productivity/productivity-xlsx), [`powerpoint`](/docs/user-guide/skills/bundled/productivity/productivity-powerpoint) |

## 참고: 전체 SKILL.md

:::info
다음은 Hermes가 이 스킬이 활성화될 때 로드하는 전체 스킬 정의입니다. 스킬이 활성 상태일 때 에이전트가 보는 지침입니다.
:::

# Docx 스킬

네 개의 소형 CLI를 통해 python-docx로 Microsoft Word `.docx` 파일을 생성하고, 읽고, 편집하고, 템플릿을 작성합니다. 텍스트, 스타일, 목록, 표, 이미지, 머리글/바닥글 및 `{{token}}` 템플릿을 처리합니다. 문서를 PDF로 렌더링하거나, 기존 `.doc` 바이너리를 편집하거나, 변경 내용 추적을 수락/거부하지는 않습니다(변경 내용 추적은 감지만 합니다 — 함정 참조).

## 사용 시점

- 사용자가 Word 문서(보고서, 편지, 계약서)를 생성해 달라고 요청할 때.
- `.docx`의 텍스트, 개요, 스타일 또는 포함된 이미지가 필요할 때.
- 기존 `.docx`를 변경해야 할 때: 텍스트 교체, 표 셀 편집, 문단 삽입/삭제, 스타일 적용.
- 데이터에서 채울 `{{placeholders}}`가 있는 `.docx` 템플릿을 사용할 때.
- 사용하지 않는 경우: `.doc`(레거시), `.odt`, PDF 변환 또는 WYSIWYG 레이아웃 작업.

## 사전 요구 사항

- `python-docx`가 설치된 Python 3.10 이상:
  `pip install python-docx` (import 이름은 `docx`).
- 이미지 블록의 경우 이미지 파일이 로컬에 있어야 합니다(PNG/JPEG).

## 실행 방법

모든 도우미는 이 파일 옆의 `scripts/`에 있습니다. `terminal` 도구로 실행하세요. 각 도우미는 `--help`를 지원하며 JSON을 표준 출력으로 출력합니다.

```bash
python scripts/docx_create.py spec.json out.docx
python scripts/docx_read.py out.docx --text
python scripts/docx_edit.py replace out.docx --find old --replace new
python scripts/docx_template.py tpl.docx values.json filled.docx
```

## 빠른 참고

| 작업 | 명령 |
| --- | --- |
| JSON 사양에서 생성 | `docx_create.py spec.json out.docx` |
| 전체 텍스트(본문+표+머리글/바닥글) | `docx_read.py f.docx --text` |
| 제목 개요 + 표 구조 | `docx_read.py f.docx --structure` |
| 실제 사용된 스타일 | `docx_read.py f.docx --styles` |
| 포함된 이미지 추출 | `docx_read.py f.docx --images outdir/` |
| 변경 내용/주석 감지 | `docx_read.py f.docx --revisions` |
| 찾기/바꾸기(서식 유지) | `docx_edit.py replace f.docx --find A --replace B -o out.docx` |
| 표 셀 설정 | `docx_edit.py set-cell f.docx --table 0 --row 1 --col 2 --text X` |
| 인덱스 N 앞에 문단 삽입 | `docx_edit.py insert f.docx --index N --text X --style Normal` |
| 문단 N 삭제 | `docx_edit.py delete f.docx --index N` |
| 문단 N에 스타일 적용 | `docx_edit.py style f.docx --index N --style "Heading 1"` |
| `{{tokens}}` 채우기 | `docx_template.py tpl.docx values.json out.docx --strict` |

## 절차

1. **생성.** `write_file`로 JSON 사양을 작성한 다음 `scripts/docx_create.py`를 실행합니다. 사양은 `page`(mm 단위 크기 + 여백), `header`/`footer` 문자열, `styles`(글꼴, 크기, 굵게/기울임, 16진수 `color`를 지정하는 사용자 지정 문단 스타일), 그리고 `blocks` — `heading`(1~9 수준), `paragraph`(`text` 또는 각 실행에 `bold`/`italic`/`underline`을 설정할 수 있는 `runs` 목록), `bullet_list`, `numbered_list`, `table`(`header` 행은 굵게 렌더링되며 `Table Grid` 같은 선택적 기본 제공 표 `style` 지원), `image`(`path`, 선택적 `width_mm`), `page_break` — 를 지원합니다. 전체 사양 형식은 `scripts/docx_create.py` 상단에 문서화되어 있으므로, 작성할 때 `read_file`로 읽으세요.
2. **읽기.** 정확히 하나의 모드 플래그와 함께 `scripts/docx_read.py`를 사용합니다. `--text`는 본문 문단, 모든 표 셀 텍스트, 머리글/바닥글 텍스트를 JSON으로 반환합니다. `--structure`는 제목 개요와 문단/표/섹션 수를 반환합니다. `--images DIR`은 패키지의 `word/media/` 아래 모든 파일을 복사합니다.
3. **편집.** `scripts/docx_edit.py`를 사용합니다. `replace`는 본문, 표(중첩 표 포함), 머리글 및 바닥글을 순회하며 실행 서식을 보존합니다. 머리글/바닥글을 건너뛰려면 `--body-only`를 추가하세요. 원본을 유지하려면 `-o out.docx`를 전달하고, 생략하면 제자리에서 편집합니다. `insert`/`delete`/`style`의 문단 인덱스는 `--structure`/`--text` 본문 순서를 따릅니다.
4. **템플릿.** 문서에 `{{name}}` 형식의 토큰을 넣습니다(문자, 숫자, `_`, `.`, `-` 지원; `{{ name }}`처럼 내부 공백이 있어도 허용). JSON 객체 형태의 값과 함께 `scripts/docx_template.py`를 실행합니다. 토큰이 채워지지 않은 상태로 남으면 실패하도록 `--strict`를 사용하세요. JSON 출력에는 어느 경우든 `filled` 수와 `unfilled_tokens`가 나열됩니다.
5. **검증(항상 수행).** `docx_read.py out.docx --text` 또는 `--structure`로 출력을 다시 읽고 예상한 내용이 있는지 확인합니다.

## 함정

- **실행으로 분할된 토큰.** Word는 종종 `{{name}}`을 여러 실행으로 나눕니다. 교체 도우미는 실행을 하나로 합쳐 이를 처리하며, 교체된 텍스트는 일치가 시작된 실행의 서식을 이어받습니다. 따라서 토큰 중간의 서식 변경은 평탄화됩니다.
- **변경 내용 추적.** `--revisions`는 삽입, 삭제, 서식 변경 및 주석을 감지할 뿐입니다. 텍스트 추출은 있는 그대로의 본문을 반환합니다(삽입은 포함하고 삭제는 제외하므로 대략 수락된 보기와 같습니다). 이 스킬은 변경 내용을 수락/거부하거나 주석 텍스트를 읽을 수 없습니다. 추측하지 말고 사용자에게 이를 알리세요.
- **스타일 이름은 존재해야 함.** 문서에 정의되지 않은 스타일을 적용하면 `KeyError`가 발생합니다. `Heading 1`, `List Bullet`, `List Number`, `Table Grid` 같은 기본 스타일은 기본 템플릿에 존재하지만, 사용자 지정 스타일은 먼저 생성 사양에 선언해야 합니다.
- **번호 매기기 목록은 다시 시작됨.** `List Number`는 Word의 기본 번호 매기기에 의존하므로 별도의 목록이 다시 1부터 시작하지 않고 계속 번호가 매겨질 수 있습니다. 단순한 문서에는 허용되지만, 여러 목록의 정확한 번호 매기기가 필요한 사용자에게는 경고하세요.
- **셀 쓰기는 서식을 대체함.** `set-cell`은 `cell.text = ...`를 사용하므로 해당 셀의 실행 서식을 일반 서식으로 초기화합니다.
- **인코딩.** 모든 JSON 사양/값 파일은 명시적으로 UTF-8로 읽습니다. 자체 연결 코드를 작성할 때 로캘 기본값에 의존하지 마세요.
- **XML을 압축 해제하고 sed로 수정하지 마세요.** 스크립트(또는 python-docx)를 통해 편집하세요. `document.xml`의 원시 텍스트 치환은 파일을 쉽게 손상시킵니다. `.docx` 자체에는 `patch`/`write_file`을 사용하지 말고 JSON 입력에만 사용하세요.

## 검증

- 생성/편집/템플릿 작성 후 `docx_read.py out.docx --text`를 실행하고 예상 문자열이 나타나는지(이전 문자열은 사라졌는지도) 확인합니다.
- 템플릿은 `--strict`와 함께 실행하거나 `unfilled_tokens == []`인지 확인합니다.
- 구조 확인: `--structure`에 예상한 제목 개요와 표 구조가 표시되어야 하며, `--styles`로 사용자 지정 스타일이 적용되었는지 확인합니다.
- 유효한 `.docx`는 예외 없이 `Document(path)`로 열립니다. 읽기 스크립트가 종료 코드 0으로 끝나는 것 자체가 정상 여부를 확인하는 방법입니다.
