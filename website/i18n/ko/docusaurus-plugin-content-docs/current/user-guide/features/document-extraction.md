---
sidebar_position: 3
title: "문서 추출"
description: "read_file이 PDF, Office 문서, 노트북을 텍스트로 변환하는 방법과 PDF가 스캔 이미지일 때의 대처 방법"
---

# 문서 추출

`read_file` 도구는 일반적인 문서 형식을 읽을 수 있는 텍스트로 자동 변환하므로, 에이전트가 소스 코드를 읽는 것과 같은 방식으로 PDF나 스프레드시트를 검사할 수 있습니다.

## 지원 형식

| 형식 | 확장자 | 변환기 | 사용 가능 여부 |
|--------|-----------|-----------|--------------|
| Jupyter 노트북 | `.ipynb` | 기본 제공 (stdlib) | 항상 |
| Word 문서 | `.docx` | 기본 제공 (stdlib) | 항상 |
| Excel 통합 문서 | `.xlsx` | 기본 제공 (stdlib) | 항상 |
| PDF | `.pdf` | 선택 사항 `anydoc` 변환기 | 최초 사용 시 자동 설치* |
| 레거시 Office | `.doc`, `.ppt`, `.xls`, `.pptx` 및 변형 | 선택 사항 `anydoc` 변환기 | 최초 사용 시 자동 설치* |
| OpenDocument | `.odt`, `.ods`, `.odp` | 선택 사항 `anydoc` 변환기 | 최초 사용 시 자동 설치* |
| 서식 있는 텍스트 / 전자책 | `.rtf`, `.epub` | 선택 사항 `anydoc` 변환기 | 최초 사용 시 자동 설치* |

\* 선택적 변환기는 `firecrawl-anydoc` 패키지이며, 설치가 허용된 경우(`config.yaml`의 `security.allow_lazy_installs`)에 지연 설치됩니다. 이 변환기가 없더라도 세 가지 stdlib 형식은 계속 작동하고, 다른 형식은 바이너리 파일 차단으로 대체됩니다.

변환 결과는 Markdown이며 `read_file`의 일반적인 `offset`/`limit` 창을 통해 페이지 단위로 제공됩니다. 도구 호출을 제한된 범위로 유지하기 위해 50MB를 초과하는 문서는 거부됩니다.

추출은 원격 터미널 백엔드(Docker, Modal, SSH)에서도 작동합니다. 파일의 바이트가 백엔드 경계를 넘어 전송된 후 호스트 측에서 변환되므로, 샌드박스 안의 문서도 로컬 문서와 동일하게 읽힙니다.

## 스캔된 PDF: 포함 범위 경고

PDF 변환은 **텍스트 레이어만** 읽습니다. 법률 문서, 재판매 패키지, 서명된 계약서, 팩스에서 흔히 볼 수 있는 스캔 이미지 페이지에는 텍스트 레이어가 없으므로 아무 내용도 없는 것으로 조용히 변환됩니다. 대표적인 신호는 본문이 비어 있는 섹션 헤더입니다.

의미 있는 비율의 페이지에서 텍스트가 나오지 않으면(문서의 20% 초과 또는 절대적으로 10페이지 이상), `read_file`은 추출 결과 앞에 경고를 추가합니다. 읽을 수 없는 각 구간에는 그 직전에 추출된 마지막 텍스트(일반적으로 섹션 구분자)가 표시되므로, 문서 전체에 OCR을 적용하지 않고 실제로 필요한 구간만 대상으로 삼을 수 있습니다.

```
[EXTRACTION COVERAGE WARNING: 198 of 311 pages in this PDF yielded no
text. ... Unreadable gaps, each labeled with the last text extracted
before it:
  pages 42-77 (36 pages) — after "Antigua Maintenance Corp Bylaws" (p41)
  pages 92-213 (122 pages) — after "... Covenants, Codes and Regulations" (p91)
  page 224 (1 page) — after "... Insurance Declaration Pages" (p223)
Decide which gaps you actually need — do NOT OCR or render everything. ...]
```

경고에는 정확한 페이지 범위와 복구 경로가 나열됩니다:

1. **몇 페이지 — 렌더링 + 비전.** 페이지를 이미지로 변환하고 비전 도구로 읽습니다:
   ```bash
   pdftoppm -jpeg -r 150 -f 92 -l 94 document.pdf /tmp/page
   ```
   그런 다음 `vision_analyze`로 각 이미지를 검사합니다. 추가 의존성은 필요하지 않습니다(감지 자체에는 poppler가 필요함).
2. **많은 페이지 — OCR.** `ocr-and-documents` 스킬이 marker-pdf를 사용한 대량 OCR을 담당합니다(90개 이상의 언어 지원, 수식과 표 처리, 설치 용량 약 3~5GB).

감지는 페이지별 텍스트 수를 세기 위해 poppler의 `pdftotext`를 사용합니다. poppler가 설치되어 있지 않아도 추출은 작동하지만, 포함 범위 검사는 조용히 건너뜁니다.

:::tip
에이전트가 경고를 자체적으로 처리하므로 누락된 페이지를 렌더링하거나 OCR할 것을 제안합니다. 추출 결과를 직접 읽는 경우, "본문이 비어 있는 헤더"를 누락된 내용이 아니라 스캔된 섹션으로 취급하세요.
:::
