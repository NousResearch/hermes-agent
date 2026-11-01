---
title: "Pdf — PDF 파일 생성, 읽기, 병합, 작성 및 보안 설정"
sidebar_label: "PDF"
description: "PDF 파일을 생성하고, 읽고, 병합하고, 작성하고, 보안을 설정합니다"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py에 의해 스킬의 SKILL.md에서 자동으로 생성됩니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# PDF

구조화된 사양으로 PDF를 생성하고, 텍스트/표/메타데이터를 추출하며, 페이지를 병합/분할/회전/워터마크 처리하고, AcroForm 양식 필드를 작성하고, 암호화/복호화합니다 — pypdf, reportlab, pdfplumber를 사용합니다. 스캔된 (이미지만 있는) PDF에는 텍스트 레이어가 없습니다: OCR은 여기서 명시적으로 범위에 포함되지 않습니다 — 페이지가 이미지만으로 구성된 경우 텍스트를 추출하는 척하지 말고 중단한 뒤 `ocr-and-documents` 스킬을 사용하세요.

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 번들됨 (기본 설치) |
| 경로 | `skills/productivity/pdf` |
| 버전 | `1.0.0` |
| 작성자 | Nous Research |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `pdf`, `documents`, `forms`, `reportlab`, `pypdf`, `pdfplumber` |
| 관련 스킬 | [`docx`](/docs/user-guide/skills/bundled/productivity/productivity-docx), [`xlsx`](/docs/user-guide/skills/bundled/productivity/productivity-xlsx), [`powerpoint`](/docs/user-guide/skills/bundled/productivity/productivity-powerpoint), [`ocr-and-documents`](/docs/user-guide/skills/bundled/productivity/productivity-ocr-and-documents) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보는 지침입니다.
:::

# PDF 스킬

구조화된 사양으로 PDF를 생성하고, 텍스트/표/메타데이터를 추출하며, 페이지를 병합/분할/회전/워터마크 처리하고, AcroForm 양식 필드를 작성하고, 암호화/복호화합니다 — pypdf, reportlab, pdfplumber를 사용합니다. 스캔된 (이미지만 있는) PDF에는 텍스트 레이어가 없습니다: OCR은 여기서 명시적으로 범위에 포함되지 않습니다 — 페이지가 이미지만으로 구성된 경우 텍스트를 추출하는 척하지 말고 중단한 뒤 `ocr-and-documents` 스킬을 사용하세요.

## 사용 시점

- 보고서, 청구서 또는 여러 페이지로 구성된 문서를 PDF로 생성합니다.
- PDF에서 텍스트, 표(JSON/CSV), 메타데이터 또는 양식 필드 값을 가져옵니다.
- PDF를 병합, 분할, 회전하고, 페이지 하위 집합을 추출하며, 워터마크, 책갈피 또는 압축을 적용합니다.
- AcroForm 양식을 작성하거나 평면화하고, 비밀번호로 암호화하거나 복호화합니다.
- 스캔된/이미지만 있는 PDF에는 사용하지 않습니다 (`ocr-and-documents` 사용). 픽셀 단위로 정확한 HTML-to-PDF 렌더링에도 사용하지 않습니다 (헤드리스 브라우저 사용).

## 사전 요구 사항

- pypdf, reportlab, pdfplumber가 설치된 Python 3.10+:
  `python3 -m pip install pypdf reportlab pdfplumber`
- 각 헬퍼 스크립트는 가져오기를 지연해서 확인하며, 종속성이 없으면 설치 안내를 출력합니다.

## 실행 방법

모든 헬퍼는 `scripts/`에 있으며 argparse CLI입니다 — `terminal` 도구로 실행하세요. 모든 헬퍼는 `--help`를 지원합니다. 헬퍼는 JSON을 엄격하게 UTF-8로 읽고 쓰며, JSON 결과를 stdout에 출력하고, 실패 시 0이 아닌 종료 코드로 종료합니다.

```bash
python3 scripts/pdf_create.py spec.json -o out.pdf         # build PDF from JSON spec
python3 scripts/pdf_read.py doc.pdf --text                 # per-page text (JSON)
python3 scripts/pdf_read.py doc.pdf --tables --csv-dir t/  # tables to JSON + CSV files
python3 scripts/pdf_read.py doc.pdf --meta                 # metadata, page sizes, encrypted/scanned flags
python3 scripts/pdf_read.py form.pdf --fields              # form fields: name, type, value
python3 scripts/pdf_merge.py a.pdf b.pdf -o merged.pdf [--bookmarks]
python3 scripts/pdf_split.py doc.pdf --pages 1-3,7 -o part.pdf [--rotate 90]
python3 scripts/pdf_fill_form.py form.pdf --fields-json values.json -o filled.pdf [--flatten]
python3 scripts/pdf_secure.py doc.pdf --encrypt -o enc.pdf --user-password your-password
python3 scripts/pdf_secure.py enc.pdf --decrypt -o dec.pdf --password your-password
python3 scripts/pdf_watermark.py doc.pdf --stamp mark.pdf -o stamped.pdf [--under]
```

## 빠른 참조

| 작업 | 도구 | 명령어 / API |
|---|---|---|
| 문서 생성 (제목, 표, 이미지) | reportlab platypus | `pdf_create.py spec.json -o out.pdf` |
| 페이지별 텍스트 | pdfplumber | `pdf_read.py f.pdf --text` |
| 표 → JSON/CSV | pdfplumber | `pdf_read.py f.pdf --tables` |
| 메타데이터 / 크기 / 암호화 여부 / 스캔 여부 | pypdf + pdfplumber | `pdf_read.py f.pdf --meta` |
| 병합 (+ 개요) | pypdf | `pdf_merge.py a.pdf b.pdf -o m.pdf` |
| 분할 / 추출 / 회전 | pypdf | `pdf_split.py f.pdf --pages 2-5 --rotate 90` |
| 양식 나열 / 작성 / 평면화 | pypdf | `pdf_read.py --fields`, `pdf_fill_form.py` |
| 암호화 / 복호화 (AES-256) | pypdf | `pdf_secure.py --encrypt/--decrypt` |
| 워터마크 / 스탬프 | pypdf | `pdf_watermark.py f.pdf --stamp w.pdf` |
| 콘텐츠 스트림 압축 | pypdf | `pdf_split.py f.pdf --pages 1-N --compress` |

## 절차

1. **먼저 검사합니다.** `pdf_read.py file.pdf --meta`를 실행합니다. `encrypted`를 확인하고 (true이면 먼저 `pdf_secure.py --decrypt`로 복호화) `likely_scanned_pages`를 확인합니다. 페이지가 이미지만으로 구성된 경우 `ocr-and-documents` 스킬로 넘깁니다 — 빈 텍스트를 "콘텐츠 없음"으로 보고하지 마세요.
2. **생성합니다.** `write_file`로 JSON 사양을 작성합니다 (요소: `heading`, `paragraph`, `table`, `image`, `pagebreak`; 선택적 `title`/`author` 메타데이터; 페이지 번호는 자동으로 추가됩니다). 그런 다음 `pdf_create.py`를 실행합니다. 레이아웃이 중요하다면 렌더링한 페이지 이미지에 `vision_analyze`를 사용하여 시각적으로 확인합니다.
3. **추출합니다.** `--text`는 페이지별 문자열의 JSON 목록을 반환하고, `--tables`는 페이지별 행 배열을 반환하며 CSV 파일도 출력할 수 있습니다. 결과는 `read_file`로 읽습니다. 바이너리 PDF를 직접 눈으로 확인하지 마세요.
4. **조작합니다.** `pdf_merge.py`는 파일을 연결하고 각 원본 파일에 하나의 책갈피를 추가할 수 있습니다. `pdf_split.py`는 페이지 범위(1부터 시작, 예: `1-3,5,9-`)와 90° 단위 회전, `--compress`를 처리합니다. `pdf_create.py`로 단일 페이지 스탬프 PDF를 준비한 다음 `pdf_watermark.py`로 오버레이하여 워터마크를 적용합니다.
5. **양식을 처리합니다.** 필드 목록을 확인하여 (`--fields`) 정확한 이름과 유형을 파악하고, `write_file`로 `{"FieldName": "value"}` 형식의 UTF-8 JSON을 작성합니다 (체크박스는 `true`/`false`를 허용하며, 라디오/선택 값은 필드의 내보내기 옵션과 일치해야 합니다). 그런 다음 `pdf_fill_form.py`를 실행합니다. `--fields`로 다시 읽어 값이 적용되었는지 확인합니다.
6. **보안을 설정합니다.** 서로 다른 사용자/소유자 비밀번호와 AES-256으로 암호화합니다. 알고 있는 비밀번호를 제거하려면 `--decrypt`를 사용하여 암호화되지 않은 사본을 작성합니다.
7. **검증합니다** (아래 참조) 보고하기 전에 확인하세요.

## 주의할 점

- **스캔된 PDF**: 페이지 이미지와 함께 `extract_text()`가 비어 있으면 텍스트 레이어가 없는 것입니다. `ocr-and-documents`로 전달하세요. 텍스트를 꾸며내지 마세요.
- **평면화의 한계**: `pdf_fill_form.py --flatten`은 위젯 표시를 페이지 콘텐츠로 변환하는 pypdf의 평면화 지원을 사용합니다. 일반 텍스트 필드와 체크박스에는 안정적이지만, 특수한 위젯(서식 있는 텍스트, 사용자 지정 표시 스트림, 일부 라디오 그룹)을 삭제하거나 잘못 렌더링할 수 있습니다. 평면화된 결과를 `vision_analyze`로 시각적으로 확인하세요. 완벽한 평면화가 필요하면 외부 렌더러(예: Ghostscript 또는 `pdftoppm`+재조합)를 대안으로 사용하세요.
- **NeedAppearances**: 작성 후 표시 스트림이 있어야 뷰어가 값을 렌더링합니다. 작성 스크립트는 적합한 뷰어가 값을 다시 생성하도록 AcroForm `NeedAppearances` 플래그를 설정하지만, 일부 최소 기능 뷰어는 이를 무시합니다 — 표시 정확성이 중요하면 평면화하세요.
- **비라틴 문자 양식 값**: 값은 올바르게 저장되지만 (UTF-16), 필드의 기본 글꼴에 글리프가 없으면 데이터가 왕복 처리되어도 뷰어에 빈칸으로 표시될 수 있습니다. 시각적으로만 확인하지 말고 `--fields`로 검증하세요.
- **압축에 대한 기대치**: `--compress`는 콘텐츠 스트림만 압축 해제합니다. 일반적인 절감률은 0–20%이며, 이미지가 대부분인 PDF나 이미 압축된 스트림에는 아무런 효과가 없습니다. 이미지 다운샘플링을 대신할 수 없습니다 (Ghostscript의 영역).
- **권한 플래그는 강제하지 않습니다**: 소유자 비밀번호의 권한 비트(인쇄 금지, 복사 금지)는 뷰어가 따를 수 있는 정중한 요청일 뿐이며, 모든 라이브러리(pypdf 포함)가 이를 읽고 제거할 수 있습니다. 콘텐츠를 실제로 암호화하여 보호하는 것은 사용자 비밀번호뿐입니다. 권한 플래그를 보안 기능으로 설명하지 마세요.
- **표 추출은 휴리스틱입니다**: pdfplumber는 선과 단어 정렬을 기준으로 표를 감지하므로, 테두리가 없거나 셀이 병합된 표는 `table_settings` 조정이나 수동 정리가 필요할 수 있습니다.
- **페이지 인덱싱**: 헬퍼 CLI는 1부터 시작하는 페이지를 사용하고, pypdf API는 0부터 시작합니다. 스크립트가 변환하므로 이중으로 변환하지 마세요.
- 회전은 90의 배수여야 하며, 암호화된 입력은 다른 작업을 수행하기 전에 복호화해야 합니다.

## 검증

- 생성/병합/분할 후: `pdf_read.py out.pdf --meta` — `page_count`와 회전한 경우 페이지별 `rotation`을 확인합니다.
- 추출 후: JSON이 비어 있지 않은지 확인하고 알려진 문자열이나 셀을 일부 점검합니다.
- 양식 작성 후: `pdf_read.py filled.pdf --fields`를 실행하고 값을 비교합니다 (비ASCII 문자를 포함하여 정확히 일치해야 합니다).
- 암호화 후: `--meta`에 `"encrypted": true`가 표시되고 비밀번호 없이 열기가 실패하는지 확인합니다. 복호화 후에는 텍스트 추출 결과가 원본과 일치해야 합니다.
- 시각적인 항목(워터마크, 평면화된 양식)의 경우 렌더링하고 `vision_analyze`로 검사합니다.
