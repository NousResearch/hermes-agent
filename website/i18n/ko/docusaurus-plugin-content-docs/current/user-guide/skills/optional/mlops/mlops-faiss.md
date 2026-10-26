---
title: "Faiss — 10억 규모에서 빠른 벡터 유사도 검색"
sidebar_label: "Faiss"
description: "10억 규모에서 빠른 벡터 유사도 검색"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Faiss

10억 규모에서 빠른 벡터 유사도 검색.

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 선택 사항 — `hermes skills install official/mlops/faiss`로 설치 |
| 경로 | `optional-skills/mlops/faiss` |
| 버전 | `1.0.0` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 종속성 | `faiss-cpu`, `faiss-gpu`, `numpy` |
| 플랫폼 | linux, macos |
| 태그 | `RAG`, `FAISS`, `Similarity Search`, `Vector Search`, `Facebook AI`, `GPU Acceleration`, `Billion-Scale`, `K-NN`, `HNSW`, `High Performance`, `Large Scale` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# FAISS - 효율적인 유사도 검색

Facebook AI의 10억 규모 벡터 유사도 검색 라이브러리입니다.

## FAISS를 사용할 때

**다음과 같은 경우 FAISS를 사용하세요:**
- 대규모 벡터 데이터셋(수백만~수십억 개)에서 빠른 유사도 검색이 필요할 때
- GPU 가속이 필요할 때
- 순수 벡터 유사도만 필요하고 메타데이터 필터링은 필요하지 않을 때
- 높은 처리량과 짧은 지연 시간이 중요할 때
- 임베딩을 오프라인/배치 처리할 때

**지표**:
- **GitHub 스타 31,700개 이상**
- Meta/Facebook AI Research
- **수십억 개 벡터 처리**
- Python 바인딩을 제공하는 **C++**

**대신 다음 대안을 사용하세요**:
- **Chroma/Pinecone**: 메타데이터 필터링이 필요할 때
- **Weaviate**: 완전한 데이터베이스 기능이 필요할 때
- **Annoy**: 더 단순하고 기능이 적은 솔루션이 필요할 때

## 빠른 시작

### 설치

```bash
# CPU only
pip install faiss-cpu

# GPU support
pip install faiss-gpu
```

### 기본 사용법

```python
import faiss
import numpy as np

# Create sample data (1000 vectors, 128 dimensions)
d = 128
nb = 1000
vectors = np.random.random((nb, d)).astype('float32')

# Create index
index = faiss.IndexFlatL2(d)  # L2 distance
index.add(vectors)             # Add vectors

# Search
k = 5  # Find 5 nearest neighbors
query = np.random.random((1, d)).astype('float32')
distances, indices = index.search(query, k)

print(f"Nearest neighbors: {indices}")
print(f"Distances: {distances}")
```

## 인덱스 유형

### 1. Flat(정확한 검색)

```python
# L2 (Euclidean) distance
index = faiss.IndexFlatL2(d)

# Inner product (cosine similarity if normalized)
index = faiss.IndexFlatIP(d)

# Slowest, most accurate
```

### 2. IVF(역파일) - 빠른 근사 검색

```python
# Create quantizer
quantizer = faiss.IndexFlatL2(d)

# IVF index with 100 clusters
nlist = 100
index = faiss.IndexIVFFlat(quantizer, d, nlist)

# Train on data
index.train(vectors)

# Add vectors
index.add(vectors)

# Search (nprobe = clusters to search)
index.nprobe = 10
distances, indices = index.search(query, k)
```

### 3. HNSW(계층적 NSW) - 최고의 품질/속도

```python
# HNSW index
M = 32  # Number of connections per layer
index = faiss.IndexHNSWFlat(d, M)

# No training needed
index.add(vectors)

# Search
distances, indices = index.search(query, k)
```

### 4. 제품 양자화 - 메모리 효율적

```python
# PQ reduces memory by 16-32×
m = 8   # Number of subquantizers
nbits = 8
index = faiss.IndexPQ(d, m, nbits)

# Train and add
index.train(vectors)
index.add(vectors)
```

## 저장 및 로드

```python
# Save index
faiss.write_index(index, "large.index")

# Load index
index = faiss.read_index("large.index")

# Continue using
distances, indices = index.search(query, k)
```

## GPU 가속

```python
# Single GPU
res = faiss.StandardGpuResources()
index_cpu = faiss.IndexFlatL2(d)
index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)  # GPU 0

# Multi-GPU
index_gpu = faiss.index_cpu_to_all_gpus(index_cpu)

# 10-100× faster than CPU
```

## LangChain 통합

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Create FAISS vector store
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())

# Save
vectorstore.save_local("faiss_index")

# Load
vectorstore = FAISS.load_local(
    "faiss_index",
    OpenAIEmbeddings(),
    allow_dangerous_deserialization=True
)

# Search
results = vectorstore.similarity_search("query", k=5)
```

## LlamaIndex 통합

```python
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

# Create FAISS index
d = 1536
faiss_index = faiss.IndexFlatL2(d)

vector_store = FaissVectorStore(faiss_index=faiss_index)
```

## 모범 사례

1. **올바른 인덱스 유형 선택** - &lt;10K에는 Flat, 10K-1M에는 IVF, 품질이 중요하면 HNSW
2. **코사인 유사도를 위해 정규화** - 정규화된 벡터와 함께 IndexFlatIP 사용
3. **대규모 데이터셋에는 GPU 사용** - 10-100배 빠름
4. **학습된 인덱스 저장** - 학습에는 많은 비용이 듦
5. **nprobe/ef_search 조정** - 속도와 정확도의 균형
6. **메모리 모니터링** - 대규모 데이터셋에는 PQ 사용
7. **쿼리 배치 처리** - GPU 활용도가 더 높음

## 성능

| 인덱스 유형 | 구축 시간 | 검색 시간 | 메모리 | 정확도 |
|------------|------------|-------------|--------|----------|
| Flat | 빠름 | 느림 | 높음 | 100% |
| IVF | 보통 | 빠름 | 보통 | 95-99% |
| HNSW | 느림 | 가장 빠름 | 높음 | 99% |
| PQ | 보통 | 빠름 | 낮음 | 90-95% |

## 리소스

- **GitHub**: https://github.com/facebookresearch/faiss ⭐ 31,700+
- **Wiki**: https://github.com/facebookresearch/faiss/wiki
- **라이선스**: MIT
