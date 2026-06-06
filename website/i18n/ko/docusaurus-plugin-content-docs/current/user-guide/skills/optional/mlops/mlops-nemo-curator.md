---
title: "Nemo Curator — LLM 학습을 위한 GPU 가속 데이터 큐레이션"
sidebar_label: "Nemo Curator"
description: "LLM 학습을 위한 GPU 가속 데이터 큐레이션"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Nemo Curator

LLM 학습을 위한 GPU 가속 데이터 큐레이션. 텍스트/이미지/비디오/오디오를 지원합니다. 퍼지 중복 제거(16배 빠름), 품질 필터링(30개 이상의 휴리스틱), 시맨틱 중복 제거, PII 난독화, NSFW 감지 기능을 제공합니다. RAPIDS를 통해 GPU 전반에 걸쳐 확장 가능합니다. 고품질 학습 데이터셋을 준비하거나, 웹 데이터를 정리하거나, 대규모 말뭉치를 중복 제거할 때 사용하세요.

## 스킬 메타데이터

| | |
|---|---|
| Source | Optional — `hermes skills install official/mlops/nemo-curator`로 설치 |
| Path | `optional-skills/mlops/nemo-curator` |
| Version | `1.0.0` |
| Author | Orchestra Research |
| License | MIT |
| Dependencies | `nemo-curator`, `cudf`, `dask`, `rapids` |
| Platforms | linux, macos |
| Tags | `Data Processing`, `NeMo Curator`, `Data Curation`, `GPU Acceleration`, `Deduplication`, `Quality Filtering`, `NVIDIA`, `RAPIDS`, `PII Redaction`, `Multimodal`, `LLM Training Data` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되어 있을 때 에이전트가 지침으로 보는 내용입니다.
:::

# NeMo Curator - GPU 가속 데이터 큐레이션

고품질 LLM 학습 데이터를 준비하기 위한 NVIDIA의 툴킷.

## NeMo Curator를 사용해야 할 때

**다음과 같은 경우 NeMo Curator를 사용하세요:**
- 웹 스크랩(Common Crawl)에서 LLM 학습 데이터를 준비할 때
- 빠른 중복 제거가 필요할 때 (CPU보다 16배 빠름)
- 멀티모달 데이터셋(텍스트, 이미지, 비디오, 오디오)을 큐레이션할 때
- 저품질 또는 유해 콘텐츠를 필터링할 때
- GPU 클러스터 전반에 걸쳐 데이터 처리를 확장할 때

**성능**:
- **16배 빠른** 퍼지 중복 제거 (8TB RedPajama v2)
- CPU 대안 대비 **40% 낮은 TCO**
- GPU 노드 전반에 걸친 **선형에 가까운 확장성**

**대신 다른 대안을 사용해야 할 때**:
- **datatrove**: CPU 기반의 오픈 소스 데이터 처리
- **dolma**: Allen AI의 데이터 툴킷
- **Ray Data**: 일반적인 ML 데이터 처리 (큐레이션에 중점을 두지 않음)

## 빠른 시작

### 설치

```bash
# 텍스트 큐레이션 (CUDA 12)
uv pip install "nemo-curator[text_cuda12]"

# 모든 모달리티
uv pip install "nemo-curator[all_cuda12]"

# CPU 전용 (느림)
uv pip install "nemo-curator[cpu]"
```

### 기본 텍스트 큐레이션 파이프라인

```python
from nemo_curator import ScoreFilter, Modify
from nemo_curator.datasets import DocumentDataset
import pandas as pd

# 데이터 로드
df = pd.DataFrame({"text": ["Good document", "Bad doc", "Excellent text"]})
dataset = DocumentDataset(df)

# 품질 필터링
def quality_score(doc):
    return len(doc["text"].split()) > 5  # 짧은 문서 필터링

filtered = ScoreFilter(quality_score)(dataset)

# 중복 제거
from nemo_curator.modules import ExactDuplicates
deduped = ExactDuplicates()(filtered)

# 저장
deduped.to_parquet("curated_data/")
```

## 데이터 큐레이션 파이프라인

### 1단계: 품질 필터링

```python
from nemo_curator.filters import (
    WordCountFilter,
    RepeatedLinesFilter,
    UrlRatioFilter,
    NonAlphaNumericFilter
)

# 30개 이상의 휴리스틱 필터 적용
from nemo_curator import ScoreFilter

# 단어 수 필터
dataset = dataset.filter(WordCountFilter(min_words=50, max_words=100000))

# 반복적인 콘텐츠 제거
dataset = dataset.filter(RepeatedLinesFilter(max_repeated_line_fraction=0.3))

# URL 비율 필터
dataset = dataset.filter(UrlRatioFilter(max_url_ratio=0.2))
```

### 2단계: 중복 제거

**정확한 중복 제거**:
```python
from nemo_curator.modules import ExactDuplicates

# 정확한 중복 제거
deduped = ExactDuplicates(id_field="id", text_field="text")(dataset)
```

**퍼지 중복 제거** (GPU에서 16배 빠름):
```python
from nemo_curator.modules import FuzzyDuplicates

# MinHash + LSH 중복 제거
fuzzy_dedup = FuzzyDuplicates(
    id_field="id",
    text_field="text",
    num_hashes=260,      # MinHash 매개변수
    num_buckets=20,
    hash_method="md5"
)

deduped = fuzzy_dedup(dataset)
```

**시맨틱 중복 제거**:
```python
from nemo_curator.modules import SemanticDuplicates

# 임베딩 기반 중복 제거
semantic_dedup = SemanticDuplicates(
    id_field="id",
    text_field="text",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    threshold=0.8  # 코사인 유사도 임계값
)

deduped = semantic_dedup(dataset)
```

### 3단계: PII 난독화

```python
from nemo_curator.modules import Modify
from nemo_curator.modifiers import PIIRedactor

# 개인 식별 정보 난독화
pii_redactor = PIIRedactor(
    supported_entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON", "LOCATION"],
    anonymize_action="replace"  # 또는 "redact"
)

redacted = Modify(pii_redactor)(dataset)
```

### 4단계: 분류기 필터링

```python
from nemo_curator.classifiers import QualityClassifier

# 품질 분류
quality_clf = QualityClassifier(
    model_path="nvidia/quality-classifier-deberta",
    batch_size=256,
    device="cuda"
)

# 저품질 문서 필터링
high_quality = dataset.filter(lambda doc: quality_clf(doc["text"]) > 0.5)
```

## GPU 가속

### GPU 대 CPU 성능

| 작업 | CPU (16 코어) | GPU (A100) | 속도 향상 |
|-----------|----------------|------------|---------|
| 퍼지 중복 제거 (8TB) | 120 시간 | 7.5 시간 | 16배 |
| 정확한 중복 제거 (1TB) | 8 시간 | 0.5 시간 | 16배 |
| 품질 필터링 | 2 시간 | 0.2 시간 | 10배 |

### 다중 GPU 확장

```python
from nemo_curator import get_client
import dask_cuda

# GPU 클러스터 초기화
client = get_client(cluster_type="gpu", n_workers=8)

# 8개의 GPU로 처리
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

# 미적 점수화
aesthetic_filter = AestheticFilter(threshold=5.0)
filtered_images = aesthetic_filter(image_dataset)

# NSFW 감지
nsfw_filter = NSFWFilter(threshold=0.9)
safe_images = nsfw_filter(filtered_images)

# CLIP 임베딩 생성
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

# 씬 감지
scene_detector = SceneDetector(threshold=27.0)
scenes = scene_detector(video_dataset)

# 클립 추출
clip_extractor = ClipExtractor(min_duration=2.0, max_duration=10.0)
clips = clip_extractor(scenes)

# 임베딩 생성
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

# ASR 전사
asr = ASRInference(model="nvidia/stt_en_fastconformer_hybrid_large_pc")
transcribed = asr(audio_dataset)

# WER (단어 오류율)별 필터링
wer_filter = WERFilter(max_wer=0.3)
high_quality_audio = wer_filter(transcribed)

# 길이 필터링
duration_filter = DurationFilter(min_duration=1.0, max_duration=30.0)
filtered_audio = duration_filter(high_quality_audio)
```

## 일반적인 패턴

### 웹 스크랩 큐레이션 (Common Crawl)

```python
from nemo_curator import ScoreFilter, Modify
from nemo_curator.filters import *
from nemo_curator.modules import *
from nemo_curator.datasets import DocumentDataset

# Common Crawl 데이터 로드
dataset = DocumentDataset.read_parquet("common_crawl/*.parquet")

# 파이프라인
pipeline = [
    # 1. 품질 필터링
    WordCountFilter(min_words=100, max_words=50000),
    RepeatedLinesFilter(max_repeated_line_fraction=0.2),
    SymbolToWordRatioFilter(max_symbol_to_word_ratio=0.3),
    UrlRatioFilter(max_url_ratio=0.3),

    # 2. 언어 필터링
    LanguageIdentificationFilter(target_languages=["en"]),

    # 3. 중복 제거
    ExactDuplicates(id_field="id", text_field="text"),
    FuzzyDuplicates(id_field="id", text_field="text", num_hashes=260),

    # 4. PII 난독화
    PIIRedactor(),

    # 5. NSFW 필터링
    NSFWClassifier(threshold=0.8)
]

# 실행
for stage in pipeline:
    dataset = stage(dataset)

# 저장
dataset.to_parquet("curated_common_crawl/")
```

### 분산 처리

```python
from nemo_curator import get_client
from dask_cuda import LocalCUDACluster

# 다중 GPU 클러스터
cluster = LocalCUDACluster(n_workers=8)
client = get_client(cluster=cluster)

# 대규모 데이터셋 처리
dataset = DocumentDataset.read_parquet("s3://large_dataset/*.parquet")
deduped = FuzzyDuplicates(...)(dataset)

# 정리
client.close()
cluster.close()
```

## 성능 벤치마크

### 퍼지 중복 제거 (8TB RedPajama v2)

- **CPU (256 코어)**: 120 시간
- **GPU (8× A100)**: 7.5 시간
- **속도 향상**: 16배

### 정확한 중복 제거 (1TB)

- **CPU (64 코어)**: 8 시간
- **GPU (4× A100)**: 0.5 시간
- **속도 향상**: 16배

### 품질 필터링 (100GB)

- **CPU (32 코어)**: 2 시간
- **GPU (2× A100)**: 0.2 시간
- **속도 향상**: 10배

## 비용 비교

**CPU 기반 큐레이션** (AWS c5.18xlarge × 10):
- 비용: 시간당 $3.60 × 10 = 시간당 $36
- 8TB 소요 시간: 120 시간
- **총계**: $4,320

**GPU 기반 큐레이션** (AWS p4d.24xlarge × 2):
- 비용: 시간당 $32.77 × 2 = 시간당 $65.54
- 8TB 소요 시간: 7.5 시간
- **총계**: $491.55

**절감액**: 89% 감소 ($3,828 절약)

## 지원되는 데이터 형식

- **입력**: Parquet, JSONL, CSV
- **출력**: Parquet (권장), JSONL
- **WebDataset**: 멀티모달용 TAR 아카이브

## 활용 사례

**프로덕션 배포**:
- NVIDIA는 Nemotron-4 학습 데이터를 준비하기 위해 NeMo Curator를 사용했습니다
- 큐레이션된 오픈 소스 데이터셋: RedPajama v2, The Pile

## 참조

- **[필터링 가이드](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/nemo-curator/references/filtering.md)** - 30개 이상의 품질 필터, 휴리스틱
- **[중복 제거 가이드](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/nemo-curator/references/deduplication.md)** - 정확한, 퍼지, 시맨틱 방법

## 리소스

- **GitHub**: https://github.com/NVIDIA/NeMo-Curator ⭐ 500+
- **문서**: https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/
- **버전**: 0.4.0+
- **라이선스**: Apache 2.0
