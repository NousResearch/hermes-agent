---
title: "Nemo Curator — LLM 학습 데이터 큐레이션: 중복 제거, 필터링, PII 삭제"
sidebar_label: "Nemo Curator"
description: "LLM 학습 데이터 큐레이션: 중복 제거, 필터링, PII 삭제"
---

{/* 이 페이지는 skill의 SKILL.md에서 website/scripts/generate-skill-docs.py로 자동 생성됩니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Nemo Curator

LLM 학습 데이터의 중복을 제거하고, 필터링하고, PII를 삭제합니다.

## Skill 메타데이터

| | |
|---|---|
| 소스 | 선택 사항 — `hermes skills install official/mlops/nemo-curator`로 설치 |
| 경로 | `optional-skills/mlops/nemo-curator` |
| 버전 | `1.0.1` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 종속성 | `nemo-curator`, `cudf`, `dask`, `rapids` |
| 플랫폼 | linux, macos |
| 태그 | `Data Processing`, `NeMo Curator`, `Data Curation`, `GPU Acceleration`, `Deduplication`, `Quality Filtering`, `NVIDIA`, `RAPIDS`, `PII Redaction`, `Multimodal`, `LLM Training Data` |

## 참고: 전체 SKILL.md

:::info
다음은 이 skill이 실행될 때 Hermes가 로드하는 완전한 skill 정의입니다. skill이 활성화되면 에이전트가 보게 되는 내용입니다.
:::

# NeMo Curator - GPU 가속 데이터 큐레이션

LLM을 위한 고품질 학습 데이터를 준비하는 NVIDIA의 도구 모음입니다.

## NeMo Curator를 사용해야 하는 경우

**다음과 같은 경우 NeMo Curator를 사용하세요:**
- 웹 스크랩(Common Crawl)에서 LLM 학습 데이터를 준비할 때
- 빠른 중복 제거가 필요할 때(CPU보다 16배 빠름)
- 멀티모달 데이터셋(텍스트, 이미지, 비디오, 오디오)을 큐레이션할 때
- 품질이 낮거나 유해한 콘텐츠를 필터링할 때
- GPU 클러스터 전체로 데이터 처리를 확장할 때

**성능**:
- **퍼지 중복 제거 16배 향상**(8TB RedPajama v2)
- **CPU 대안 대비 TCO 40% 절감**
- **GPU 노드 전반에서 거의 선형적인 확장**

**대신 다음 대안을 사용하세요**:
- **datatrove**: CPU 기반 오픈 소스 데이터 처리
- **dolma**: Allen AI의 데이터 도구 모음
- **Ray Data**: 범용 ML 데이터 처리(큐레이션에 초점을 두지 않음)

## 빠른 시작

### 설치

```bash
# NeMo Curator 1.x installs with uv. Extras use hyphens (PyPI-normalized):
#   text-cuda12 / text-cpu (and image/video/audio/math variants), or `all`.

# Text curation (CUDA 12)
uv pip install "nemo-curator[text-cuda12]"

# All modalities
uv pip install "nemo-curator[all]"

# CPU-only text (slower)
uv pip install "nemo-curator[text-cpu]"
```

### 기본 텍스트 큐레이션 파이프라인

> **주요 버전 재작성(1.x):** NeMo Curator는 **Ray 기반 파이프라인/스테이지 아키텍처**를 중심으로 다시 작성되었습니다. 0.x의 `DocumentDataset` + `nemo_curator.modules.*` / `ScoreFilter` / `Modify` 객체를 데이터셋에 호출하는 API는 사라졌습니다. 1.x에서는 `ProcessingStage`를 `Pipeline`으로 조합하고 executor로 실행합니다. 정확한 stage/import 인터페이스는 모달리티마다 다릅니다. 따라서 아래 예시는 **개념을 설명하기 위한 것**(0.x 방식)으로 보고, 정확한 1.x API는 예시의 import를 그대로 복사하지 말고 현재 [빠른 시작](https://github.com/NVIDIA-NeMo/Curator/blob/main/tutorials/quickstart.py)과 [텍스트 가이드](https://docs.nvidia.com/nemo/curator/latest/get-started/text)를 따르세요.

1.x 파이프라인의 형태(업스트림 빠른 시작에서 발췌):

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.core.client import RayClient

# 1. Define/compose stages (load -> filter -> dedupe -> classify -> write).
#    Each stage declares its own Resources (CPU cores, GPU memory, replicas).
pipeline = Pipeline(name="curation", stages=[...])

# 2. Run it with an executor (Ray-backed).
client = RayClient()
client.start()
pipeline.run(XennaExecutor())
client.stop()
```

이후 섹션의 0.x 방식 스니펫은 품질 필터링, 정확/퍼지/시맨틱 중복 제거, PII 삭제, 분류기 필터링이라는 **개념**을 설명합니다. 실행 가능한 1.x 코드는 모달리티 가이드에서 해당 개념에 대응하는 stage로 매핑하세요.

## 데이터 큐레이션 파이프라인

### 1단계: 품질 필터링

```python
from nemo_curator.filters import (
    WordCountFilter,
    RepeatedLinesFilter,
    UrlRatioFilter,
    NonAlphaNumericFilter
)

# Apply 30+ heuristic filters
from nemo_curator import ScoreFilter

# Word count filter
dataset = dataset.filter(WordCountFilter(min_words=50, max_words=100000))

# Remove repetitive content
dataset = dataset.filter(RepeatedLinesFilter(max_repeated_line_fraction=0.3))

# URL ratio filter
dataset = dataset.filter(UrlRatioFilter(max_url_ratio=0.2))
```

### 2단계: 중복 제거

**정확한 중복 제거**:
```python
from nemo_curator.modules import ExactDuplicates

# Remove exact duplicates
deduped = ExactDuplicates(id_field="id", text_field="text")(dataset)
```

**퍼지 중복 제거**(GPU에서 16배 빠름):
```python
from nemo_curator.modules import FuzzyDuplicates

# MinHash + LSH deduplication
fuzzy_dedup = FuzzyDuplicates(
    id_field="id",
    text_field="text",
    num_hashes=260,      # MinHash parameters
    num_buckets=20,
    hash_method="md5"
)

deduped = fuzzy_dedup(dataset)
```

**시맨틱 중복 제거**:
```python
from nemo_curator.modules import SemanticDuplicates

# Embedding-based deduplication
semantic_dedup = SemanticDuplicates(
    id_field="id",
    text_field="text",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    threshold=0.8  # Cosine similarity threshold
)

deduped = semantic_dedup(dataset)
```

### 3단계: PII 삭제

```python
from nemo_curator.modules import Modify
from nemo_curator.modifiers import PIIRedactor

# Redact personally identifiable information
pii_redactor = PIIRedactor(
    supported_entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON", "LOCATION"],
    anonymize_action="replace"  # or "redact"
)

redacted = Modify(pii_redactor)(dataset)
```

### 4단계: 분류기 필터링

```python
from nemo_curator.classifiers import QualityClassifier

# Quality classification
quality_clf = QualityClassifier(
    model_path="nvidia/quality-classifier-deberta",
    batch_size=256,
    device="cuda"
)

# Filter low-quality documents
high_quality = dataset.filter(lambda doc: quality_clf(doc["text"]) > 0.5)
```

## GPU 가속

### GPU와 CPU 성능 비교

| 작업 | CPU (16코어) | GPU (A100) | 속도 향상 |
|-----------|----------------|------------|---------|
| 퍼지 중복 제거(8TB) | 120시간 | 7.5시간 | 16배 |
| 정확한 중복 제거(1TB) | 8시간 | 0.5시간 | 16배 |
| 품질 필터링 | 2시간 | 0.2시간 | 10배 |

### 멀티 GPU 확장

```python
from nemo_curator import get_client
import dask_cuda

# Initialize GPU cluster
client = get_client(cluster_type="gpu", n_workers=8)

# Process with 8 GPUs
deduped = FuzzyDuplicates(...)(dataset)
```

## 멀티모달 큐레이션

### 이미지 큐레이션

```python
from nemo_curator.image import (
    AestheticFilter,
    NSFWFilter,
    CLIPEmbedder
)

# Aesthetic scoring
aesthetic_filter = AestheticFilter(threshold=5.0)
filtered_images = aesthetic_filter(image_dataset)

# NSFW detection
nsfw_filter = NSFWFilter(threshold=0.9)
safe_images = nsfw_filter(filtered_images)

# Generate CLIP embeddings
clip_embedder = CLIPEmbedder(model="openai/clip-vit-base-patch32")
image_embeddings = clip_embedder(safe_images)
```

### 비디오 큐레이션

```python
from nemo_curator.video import (
    SceneDetector,
    ClipExtractor,
    InternVideo2Embedder
)

# Detect scenes
scene_detector = SceneDetector(threshold=27.0)
scenes = scene_detector(video_dataset)

# Extract clips
clip_extractor = ClipExtractor(min_duration=2.0, max_duration=10.0)
clips = clip_extractor(scenes)

# Generate embeddings
video_embedder = InternVideo2Embedder()
video_embeddings = video_embedder(clips)
```

### 오디오 큐레이션

```python
from nemo_curator.audio import (
    ASRInference,
    WERFilter,
    DurationFilter
)

# ASR transcription
asr = ASRInference(model="nvidia/stt_en_fastconformer_hybrid_large_pc")
transcribed = asr(audio_dataset)

# Filter by WER (word error rate)
wer_filter = WERFilter(max_wer=0.3)
high_quality_audio = wer_filter(transcribed)

# Duration filtering
duration_filter = DurationFilter(min_duration=1.0, max_duration=30.0)
filtered_audio = duration_filter(high_quality_audio)
```

## 일반적인 패턴

### 웹 스크랩 큐레이션(Common Crawl)

```python
from nemo_curator import ScoreFilter, Modify
from nemo_curator.filters import *
from nemo_curator.modules import *
from nemo_curator.datasets import DocumentDataset

# Load Common Crawl data
dataset = DocumentDataset.read_parquet("common_crawl/*.parquet")

# Pipeline
pipeline = [
    # 1. Quality filtering
    WordCountFilter(min_words=100, max_words=50000),
    RepeatedLinesFilter(max_repeated_line_fraction=0.2),
    SymbolToWordRatioFilter(max_symbol_to_word_ratio=0.3),
    UrlRatioFilter(max_url_ratio=0.3),

    # 2. Language filtering
    LanguageIdentificationFilter(target_languages=["en"]),

    # 3. Deduplication
    ExactDuplicates(id_field="id", text_field="text"),
    FuzzyDuplicates(id_field="id", text_field="text", num_hashes=260),

    # 4. PII redaction
    PIIRedactor(),

    # 5. NSFW filtering
    NSFWClassifier(threshold=0.8)
]

# Execute
for stage in pipeline:
    dataset = stage(dataset)

# Save
dataset.to_parquet("curated_common_crawl/")
```

### 분산 처리

```python
from nemo_curator import get_client
from dask_cuda import LocalCUDACluster

# Multi-GPU cluster
cluster = LocalCUDACluster(n_workers=8)
client = get_client(cluster=cluster)

# Process large dataset
dataset = DocumentDataset.read_parquet("s3://large_dataset/*.parquet")
deduped = FuzzyDuplicates(...)(dataset)

# Cleanup
client.close()
cluster.close()
```

## 성능 벤치마크

### 퍼지 중복 제거(8TB RedPajama v2)

- **CPU(256코어)**: 120시간
- **GPU(8× A100)**: 7.5시간
- **속도 향상**: 16배

### 정확한 중복 제거(1TB)

- **CPU(64코어)**: 8시간
- **GPU(4× A100)**: 0.5시간
- **속도 향상**: 16배

### 품질 필터링(100GB)

- **CPU(32코어)**: 2시간
- **GPU(2× A100)**: 0.2시간
- **속도 향상**: 10배

## 비용 비교

**CPU 기반 큐레이션**(AWS c5.18xlarge × 10):
- 비용: 시간당 $3.60 × 10 = 시간당 $36
- 8TB 처리 시간: 120시간
- **총 비용**: $4,320

**GPU 기반 큐레이션**(AWS p4d.24xlarge × 2):
- 비용: 시간당 $32.77 × 2 = 시간당 $65.54
- 8TB 처리 시간: 7.5시간
- **총 비용**: $491.55

**절감액**: 89% 절감($3,828 절약)

## 지원되는 데이터 형식

- **입력**: Parquet, JSONL, CSV
- **출력**: Parquet(권장), JSONL
- **WebDataset**: 멀티모달용 TAR 아카이브

## 사용 사례

**프로덕션 배포**:
- NVIDIA는 Nemotron-4 학습 데이터를 준비하는 데 NeMo Curator를 사용했습니다
- 큐레이션된 오픈 소스 데이터셋: RedPajama v2, The Pile

## 참고 자료

- **[필터링 가이드](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/nemo-curator/references/filtering.md)** - 30개 이상의 품질 필터, 휴리스틱
- **[중복 제거 가이드](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/nemo-curator/references/deduplication.md)** - 정확, 퍼지, 시맨틱 방식

## 리소스

- **GitHub**: https://github.com/NVIDIA-NeMo/Curator
- **문서**: https://docs.nvidia.com/nemo/curator/latest/
- **버전**: 1.2.0(1.x는 Ray 기반 파이프라인으로 재작성되었습니다 — 0.x 스니펫을 복사하기 전에 빠른 시작을 확인하세요)
- **라이선스**: Apache 2.0
