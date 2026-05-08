---
title: "Nemo Curator — 用于 LLM 训练的 GPU 加速数据整理"
sidebar_label: "Nemo Curator"
description: "用于 LLM 训练的 GPU 加速数据整理"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Nemo Curator

用于 LLM 训练的 GPU 加速数据整理。支持文本/图像/视频/音频。特性包括模糊去重（快 16 倍）、质量过滤（30+ 启发式规则）、语义去重、PII 脱敏和 NSFW 检测。可通过 RAPIDS 在多 GPU 上扩展规模。适用于准备高质量训练数据集、清理网页数据或对大规模语料库进行去重。

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/mlops/nemo-curator` |
| Path | `optional-skills/mlops/nemo-curator` |
| Version | `1.0.0` |
| Author | Orchestra Research |
| License | MIT |
| Dependencies | `nemo-curator`, `cudf`, `dask`, `rapids` |
| Tags | `Data Processing`, `NeMo Curator`, `Data Curation`, `GPU Acceleration`, `Deduplication`, `Quality Filtering`, `NVIDIA`, `RAPIDS`, `PII Redaction`, `Multimodal`, `LLM 训练ing Data` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# NeMo Curator - GPU-Accelerated Data Curation

NVIDIA 的工具包，用于为 LLM 准备高质量训练数据。

## 何时使用 NeMo Curator

**Use NeMo Curator when:**
- Preparing LLM training data from web scrapes (Common Crawl)
- Need fast deduplication (16× faster than CPU)
- Curating multi-modal datasets (text, images, video, audio)
- Filtering low-quality or toxic content
- Scaling data processing across GPU cluster

**Performance**:
- **16× faster** fuzzy deduplication (8TB RedPajama v2)
- **40% lower TCO** vs CPU alternatives
- **Near-linear scaling** across GPU nodes

**替代方案**:
- **datatrove**: CPU-based, open-source data processing
- **dolma**: Allen AI's data toolkit
- **Ray Data**: General ML data processing (no curation focus)

## 快速入门

### 安装

```bash
# Text curation (CUDA 12)
uv pip install "nemo-curator[text_cuda12]"

# All modalities
uv pip install "nemo-curator[all_cuda12]"

# CPU-only (slower)
uv pip install "nemo-curator[cpu]"
```

### 基本文本整理管道

```python
from nemo_curator import ScoreFilter, Modify
from nemo_curator.datasets import DocumentDataset
import pandas as pd

# Load data
df = pd.DataFrame({"text": ["Good document", "Bad doc", "Excellent text"]})
dataset = DocumentDataset(df)

# Quality filtering
def quality_score(doc):
    return len(doc["text"].split()) > 5  # Filter short docs

filtered = ScoreFilter(quality_score)(dataset)

# Deduplication
from nemo_curator.modules import ExactDuplicates
deduped = ExactDuplicates()(filtered)

# 保存
deduped.to_parquet("curated_data/")
```

## 数据整理管道

### 阶段 1：质量过滤

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

### 阶段 2：去重

**精确去重**:
```python
from nemo_curator.modules import ExactDuplicates

# Remove exact duplicates
deduped = ExactDuplicates(id_field="id", text_field="text")(dataset)
```

**模糊去重** (16× faster on GPU):
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

**语义去重**:
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

### 阶段 3：PII 脱敏

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

### 阶段 4：分类器过滤

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

## GPU 加速

### GPU 与 CPU 性能对比

| 操作 | CPU（16 核心） | GPU（A100） | 加速比 |
|-----------|----------------|------------|---------|
| Fuzzy dedup (8TB) | 120 hours | 7.5 hours | 16× |
| Exact dedup (1TB) | 8 hours | 0.5 hours | 16× |
| Quality filtering | 2 hours | 0.2 hours | 10× |

### 多 GPU 扩展

```python
from nemo_curator import get_client
import dask_cuda

# 初始化 GPU cluster
client = get_client(cluster_type="gpu", n_workers=8)

# Process with 8 GPUs
deduped = FuzzyDuplicates(...)(dataset)
```

## 多模态整理

### 图像整理

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

### 视频整理

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

### 音频整理

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

## Common patterns

### 网页抓取整理（Common Crawl）

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

# 保存
dataset.to_parquet("curated_common_crawl/")
```

### 分布式处理

```python
from nemo_curator import get_client
from dask_cuda import LocalCUDACluster

# 多 GPU cluster
cluster = LocalCUDACluster(n_workers=8)
client = get_client(cluster=cluster)

# Process large dataset
dataset = DocumentDataset.read_parquet("s3://large_dataset/*.parquet")
deduped = FuzzyDuplicates(...)(dataset)

# Cleanup
client.close()
cluster.close()
```

## 性能基准

### 模糊去重 (8TB RedPajama v2)

- **CPU (256 cores)**: 120 hours
- **GPU (8× A100)**: 7.5 hours
- **加速比**: 16×

### 精确去重 (1TB)

- **CPU (64 cores)**: 8 hours
- **GPU (4× A100)**: 0.5 hours
- **加速比**: 16×

### Quality filtering (100GB)

- **CPU (32 cores)**: 2 hours
- **GPU (2× A100)**: 0.2 hours
- **加速比**: 10×

## 成本比较

**CPU-based curation** (AWS c5.18xlarge × 10):
- Cost: $3.60/hour × 10 = $36/hour
- Time for 8TB: 120 hours
- **Total**: $4,320

**GPU-based curation** (AWS p4d.24xlarge × 2):
- Cost: $32.77/hour × 2 = $65.54/hour
- Time for 8TB: 7.5 hours
- **Total**: $491.55

**Savings**: 89% reduction ($3,828 saved)

## 支持的数据格式

- **Input**: Parquet, JSONL, CSV
- **Output**: Parquet (recommended), JSONL
- **WebDataset**: TAR archives for multi-modal

## 用例

**生产部署**:
- NVIDIA used NeMo Curator to prepare Nemotron-4 training data
- Open-source datasets curated: RedPajama v2, The Pile

## References

- **[Filtering Guide](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/nemo-curator/references/filtering.md)** - 30+ quality filters, heuristics
- **[Deduplication Guide](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/nemo-curator/references/deduplication.md)** - Exact, fuzzy, semantic methods

## Resources

- **GitHub**: https://github.com/NVIDIA/NeMo-Curator ⭐ 500+
- **Docs**: https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/
- **Version**: 0.4.0+
- **License**: Apache 2.0
