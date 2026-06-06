---
title: "Ocr And Documents — PDF/스캔본에서 텍스트 추출 (pymupdf, marker-pdf)"
sidebar_label: "Ocr And Documents"
description: "PDF/스캔본에서 텍스트 추출 (pymupdf, marker-pdf)"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Ocr And Documents

PDF/스캔본에서 텍스트 추출 (pymupdf, marker-pdf).

## 스킬 메타데이터

| | |
|---|---|
| Source | Bundled (기본 설치됨) |
| Path | `skills/productivity/ocr-and-documents` |
| Version | `2.3.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `PDF`, `Documents`, `Research`, `Arxiv`, `Text-Extraction`, `OCR` |
| Related skills | [`powerpoint`](/docs/user-guide/skills/bundled/productivity/productivity-powerpoint) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화될 때 에이전트가 지침으로 보는 내용입니다.
:::

# PDF 및 문서 추출 (PDF & Document Extraction)

DOCX용: `python-docx` 사용 (실제 문서 구조를 구문 분석하며 OCR보다 훨씬 우수함).
PPTX용: `powerpoint` 스킬 참조 (전체 슬라이드/노트를 지원하는 `python-pptx` 사용).
이 스킬은 **PDF 및 스캔된 문서**를 다룹니다.

## 1단계: 원격 URL을 사용할 수 있습니까?

문서에 URL이 있는 경우 **항상 `web_extract`를 먼저 시도하세요**:

```
web_extract(urls=["https://arxiv.org/pdf/2402.03300"])
web_extract(urls=["https://example.com/report.pdf"])
```

이 기능은 로컬 의존성 없이 Firecrawl을 통해 PDF-to-Markdown 변환을 처리합니다.

파일이 로컬에 있거나, web_extract가 실패하거나, 일괄 처리(batch processing)가 필요한 경우에만 로컬 추출을 사용하세요.

## 2단계: 로컬 추출기 선택

| 기능 | pymupdf (~25MB) | marker-pdf (~3-5GB) |
|---------|-----------------|---------------------|
| **텍스트 기반 PDF** | ✅ | ✅ |
| **스캔된 PDF (OCR)** | ❌ | ✅ (90개 이상의 언어) |
| **표 (Tables)** | ✅ (기본) | ✅ (높은 정확도) |
| **수식 / LaTeX** | ❌ | ✅ |
| **코드 블록** | ❌ | ✅ |
| **폼 (Forms)** | ❌ | ✅ |
| **머리글/바닥글 제거** | ❌ | ✅ |
| **읽기 순서 감지** | ❌ | ✅ |
| **이미지 추출** | ✅ (임베디드) | ✅ (컨텍스트 포함) |
| **이미지 → 텍스트 (OCR)** | ❌ | ✅ |
| **EPUB** | ✅ | ✅ |
| **마크다운 출력** | ✅ (pymupdf4llm 통해) | ✅ (네이티브, 더 높은 품질) |
| **설치 용량** | ~25MB | ~3-5GB (PyTorch + 모델) |
| **속도** | 즉시 | ~1-14초/페이지 (CPU), ~0.2초/페이지 (GPU) |

**결정**: OCR, 수식, 폼 또는 복잡한 레이아웃 분석이 필요한 경우가 아니라면 pymupdf를 사용하세요.

사용자가 marker 기능이 필요하지만 시스템에 약 5GB의 여유 디스크 공간이 없는 경우:
> "이 문서는 OCR/고급 추출(marker-pdf)이 필요하며, PyTorch 및 모델을 위해 약 5GB의 공간이 필요합니다. 시스템의 여유 공간은 [X]GB입니다. 옵션: 공간을 확보하거나, web_extract를 사용할 수 있도록 URL을 제공하거나, 텍스트 기반 PDF에는 작동하지만 스캔된 문서나 수식에는 작동하지 않는 pymupdf를 시도할 수 있습니다."

---

## pymupdf (경량)

```bash
pip install pymupdf pymupdf4llm
```

**도우미 스크립트 사용 (Via helper script)**:
```bash
python scripts/extract_pymupdf.py document.pdf              # 일반 텍스트
python scripts/extract_pymupdf.py document.pdf --markdown    # 마크다운
python scripts/extract_pymupdf.py document.pdf --tables      # 표
python scripts/extract_pymupdf.py document.pdf --images out/ # 이미지 추출
python scripts/extract_pymupdf.py document.pdf --metadata    # 제목, 저자, 페이지
python scripts/extract_pymupdf.py document.pdf --pages 0-4   # 특정 페이지
```

**인라인 (Inline)**:
```bash
python3 -c "
import pymupdf
doc = pymupdf.open('document.pdf')
for page in doc:
    print(page.get_text())
"
```

---

## marker-pdf (고품질 OCR)

```bash
# 먼저 디스크 공간을 확인하세요
python scripts/extract_marker.py --check

pip install marker-pdf
```

**도우미 스크립트 사용 (Via helper script)**:
```bash
python scripts/extract_marker.py document.pdf                # 마크다운
python scripts/extract_marker.py document.pdf --json         # 메타데이터가 포함된 JSON
python scripts/extract_marker.py document.pdf --output_dir out/  # 이미지 저장
python scripts/extract_marker.py scanned.pdf                 # 스캔된 PDF (OCR)
python scripts/extract_marker.py document.pdf --use_llm      # LLM 기반 정확도 향상
```

**CLI** (marker-pdf와 함께 설치됨):
```bash
marker_single document.pdf --output_dir ./output
marker /path/to/folder --workers 4    # 일괄 처리(Batch)
```

---

## Arxiv 논문

```
# 초록만 (빠름)
web_extract(urls=["https://arxiv.org/abs/2402.03300"])

# 전체 논문
web_extract(urls=["https://arxiv.org/pdf/2402.03300"])

# 검색
web_search(query="arxiv GRPO reinforcement learning 2026")
```

## 분할, 병합 및 검색 (Split, Merge & Search)

pymupdf는 이를 기본적으로 처리합니다 — `execute_code` 또는 인라인 Python을 사용하세요:

```python
# 분할: 1-5페이지를 새 PDF로 추출
import pymupdf
doc = pymupdf.open("report.pdf")
new = pymupdf.open()
for i in range(5):
    new.insert_pdf(doc, from_page=i, to_page=i)
new.save("pages_1-5.pdf")
```

```python
# 여러 PDF 병합
import pymupdf
result = pymupdf.open()
for path in ["a.pdf", "b.pdf", "c.pdf"]:
    result.insert_pdf(pymupdf.open(path))
result.save("merged.pdf")
```

```python
# 모든 페이지에서 텍스트 검색
import pymupdf
doc = pymupdf.open("report.pdf")
for i, page in enumerate(doc):
    results = page.search_for("revenue")
    if results:
        print(f"Page {i+1}: {len(results)} match(es)")
        print(page.get_text("text"))
```

추가 의존성이 필요하지 않습니다 — pymupdf는 하나의 패키지에서 분할, 병합, 검색 및 텍스트 추출을 모두 지원합니다.

---

## 참고 사항 (Notes)

- URL의 경우 항상 `web_extract`가 최우선 선택입니다.
- pymupdf는 안전한 기본값입니다 — 즉각적이고, 모델이 없으며, 어디서나 작동합니다.
- marker-pdf는 OCR, 스캔된 문서, 수식, 복잡한 레이아웃을 위한 것입니다 — 필요할 때만 설치하세요.
- 두 도우미 스크립트 모두 전체 사용법을 보려면 `--help`를 사용할 수 있습니다.
- marker-pdf는 처음 사용할 때 `~/.cache/huggingface/`에 약 2.5GB의 모델을 다운로드합니다.
- Word 문서용: `pip install python-docx` (OCR보다 우수 — 실제 구조를 구문 분석함)
- PowerPoint용: `powerpoint` 스킬 참조 (python-pptx 사용)
