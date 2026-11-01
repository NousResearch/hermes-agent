---
title: "OCR 및 문서 — PDF/스캔에서 텍스트 추출하기 (pymupdf, marker-pdf)"
sidebar_label: "OCR 및 문서"
description: "PDF/스캔에서 텍스트 추출하기 (pymupdf, marker-pdf)"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# OCR 및 문서

PDF/스캔에서 텍스트를 추출합니다(pymupdf, marker-pdf).

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 번들됨(기본 설치) |
| 경로 | `skills/productivity/ocr-and-documents` |
| 버전 | `2.3.0` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `PDF`, `Documents`, `Research`, `Arxiv`, `Text-Extraction`, `OCR` |
| 관련 스킬 | [`pdf`](/docs/user-guide/skills/bundled/productivity/productivity-pdf), [`docx`](/docs/user-guide/skills/bundled/productivity/productivity-docx), [`powerpoint`](/docs/user-guide/skills/bundled/productivity/productivity-powerpoint) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 내용입니다.
:::

# PDF 및 문서 추출

DOCX의 경우: `docx` 스킬(생성/편집)을 참조하거나 구조화된 읽기에 `python-docx`를 사용하세요.
PPTX의 경우: `powerpoint` 스킬(전체 생성/읽기/편집 지원)을 참조하세요.
PDF 조작(병합, 분할, 양식, 워터마크, 생성)의 경우: `pdf` 스킬을 참조하세요.
이 스킬은 **PDF 및 스캔한 문서에서 텍스트를 추출하는 작업**을 다룹니다.

> **`read_file`의 텍스트 추출 범위 경고에서 이어진 경우인가요?** `read_file`은 로컬 PDF를 자동으로 변환하지만 텍스트 레이어만 읽습니다. 경고 하단에는 텍스트가 없는 페이지(스캔 이미지)가 표시됩니다. 몇 페이지 정도라면 렌더링 후 비전 기능을 사용하는 것이 가장 빠릅니다. `pdftoppm -jpeg -r 150 -f N -l N file.pdf /tmp/page`를 실행한 다음 각 이미지에 `vision_analyze`를 사용하세요. 많은 페이지를 일괄 OCR 처리하려면 아래의 marker-pdf를 사용하세요(Step 2).

## 1단계: 원격 URL을 사용할 수 있나요?

문서에 URL이 있다면 **항상 먼저 `web_extract`를 시도하세요**:

```
web_extract(urls=["https://arxiv.org/pdf/2402.03300"])
web_extract(urls=["https://example.com/report.pdf"])
```

이 기능은 Firecrawl을 사용해 PDF를 Markdown으로 변환하며, 로컬 의존성이 필요하지 않습니다.

다음과 같은 경우에만 로컬 추출을 사용하세요. 파일이 로컬에 있거나, web_extract가 실패했거나, 일괄 처리가 필요한 경우입니다.

## 2단계: 로컬 추출기 선택

| 기능 | pymupdf(~25MB) | marker-pdf(~3~5GB) |
|---------|-----------------|-----------------|
| **텍스트 기반 PDF** | ✅ | ✅ |
| **스캔 PDF(OCR)** | ❌ | ✅(90개 이상의 언어) |
| **표** | ✅(기본) | ✅(높은 정확도) |
| **수식 / LaTeX** | ❌ | ✅ |
| **코드 블록** | ❌ | ✅ |
| **양식** | ❌ | ✅ |
| **머리글/바닥글 제거** | ❌ | ✅ |
| **읽기 순서 감지** | ❌ | ✅ |
| **이미지 추출** | ✅(내장 이미지) | ✅(문맥 포함) |
| **이미지 → 텍스트(OCR)** | ❌ | ✅ |
| **EPUB** | ✅ | ✅ |
| **Markdown 출력** | ✅(pymupdf4llm 사용) | ✅(네이티브, 더 높은 품질) |
| **설치 크기** | ~25MB | ~3~5GB(PyTorch + 모델) |
| **속도** | 즉시 | ~1~14초/페이지(CPU), ~0.2초/페이지(GPU) |

**결정**: OCR, 수식, 양식 또는 복잡한 레이아웃 분석이 필요하지 않다면 pymupdf를 사용하세요.

사용자에게 marker 기능이 필요하지만 시스템에 약 5GB의 여유 디스크 공간이 없는 경우:
> "이 문서에는 OCR/고급 추출(marker-pdf)이 필요하며, PyTorch와 모델에 약 5GB가 필요합니다. 시스템에 남은 공간은 [X]GB입니다. 공간을 확보하거나, web_extract를 사용할 수 있도록 URL을 제공하거나, 텍스트 기반 PDF에서는 작동하지만 스캔 문서나 수식에서는 작동하지 않는 pymupdf를 시도할 수 있습니다."

---

## pymupdf(경량)

```bash
pip install pymupdf pymupdf4llm
```

**도우미 스크립트 사용**:
```bash
python scripts/extract_pymupdf.py document.pdf              # Plain text
python scripts/extract_pymupdf.py document.pdf --markdown    # Markdown
python scripts/extract_pymupdf.py document.pdf --tables      # Tables
python scripts/extract_pymupdf.py document.pdf --images out/ # Extract images
python scripts/extract_pymupdf.py document.pdf --metadata    # Title, author, pages
python scripts/extract_pymupdf.py document.pdf --pages 0-4   # Specific pages
```

**인라인 사용**:
```bash
python3 -c "
import pymupdf
doc = pymupdf.open('document.pdf')
for page in doc:
    print(page.get_text())
"
```

---

## marker-pdf(고품질 OCR)

```bash
# Check disk space first
python scripts/extract_marker.py --check

pip install marker-pdf
```

**도우미 스크립트 사용**:
```bash
python scripts/extract_marker.py document.pdf                # Markdown
python scripts/extract_marker.py document.pdf --json         # JSON with metadata
python scripts/extract_marker.py document.pdf --output_dir out/  # Save images
python scripts/extract_marker.py scanned.pdf                 # Scanned PDF (OCR)
python scripts/extract_marker.py document.pdf --use_llm      # LLM-boosted accuracy
```

**CLI**(marker-pdf와 함께 설치됨):
```bash
marker_single document.pdf --output_dir ./output
marker /path/to/folder --workers 4    # Batch
```

---

## Arxiv 논문

```
# Abstract only (fast)
web_extract(urls=["https://arxiv.org/abs/2402.03300"])

# Full paper
web_extract(urls=["https://arxiv.org/pdf/2402.03300"])

# Search
web_search(query="arxiv GRPO reinforcement learning 2026")
```

## 분할, 병합 및 검색

pymupdf는 이러한 작업을 기본적으로 처리합니다. `execute_code` 또는 인라인 Python을 사용하세요:

```python
# Split: extract pages 1-5 to a new PDF
import pymupdf
doc = pymupdf.open("report.pdf")
new = pymupdf.open()
for i in range(5):
    new.insert_pdf(doc, from_page=i, to_page=i)
new.save("pages_1-5.pdf")
```

```python
# Merge multiple PDFs
import pymupdf
result = pymupdf.open()
for path in ["a.pdf", "b.pdf", "c.pdf"]:
    result.insert_pdf(pymupdf.open(path))
result.save("merged.pdf")
```

```python
# Search for text across all pages
import pymupdf
doc = pymupdf.open("report.pdf")
for i, page in enumerate(doc):
    results = page.search_for("revenue")
    if results:
        print(f"Page {i+1}: {len(results)} match(es)")
        print(page.get_text("text"))
```

추가 의존성은 필요하지 않습니다. pymupdf 하나로 PDF 분할, 병합, 검색 및 텍스트 추출을 모두 처리할 수 있습니다.

---

## 참고 사항

- URL에는 항상 `web_extract`를 첫 번째로 선택하세요.
- pymupdf는 안전한 기본값입니다. 즉시 사용할 수 있고, 모델이 필요하지 않으며, 어디서나 작동합니다.
- marker-pdf는 OCR, 스캔 문서, 수식 및 복잡한 레이아웃에 사용하세요. 필요한 경우에만 설치하세요.
- 두 도우미 스크립트 모두 전체 사용법을 확인할 수 있는 `--help`를 지원합니다.
- marker-pdf는 처음 사용할 때 `~/.cache/huggingface/`에 약 2.5GB의 모델을 다운로드합니다.
- Word 문서의 경우 `pip install python-docx`를 사용하세요(OCR보다 우수하며 실제 구조를 분석합니다).
- PowerPoint의 경우 `powerpoint` 스킬을 참조하세요(python-pptx 사용).
