---
title: "Faiss"
sidebar_label: "Faiss"
description: "효율적인 유사도 검색 및 밀집 벡터(dense vector) 클러스터링을 위한 고성능 라이브러리"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Faiss

효율적인 유사도 검색 및 밀집 벡터(dense vector) 클러스터링을 위한 고성능 C++/Python 라이브러리. 메모리에 맞지 않는 벡터 세트를 지원합니다. 백만 규모에서 십억 규모에 이르는 벡터 데이터셋에서 확장 가능한 가장 가까운 이웃(nearest neighbor) 검색이나 k-means 클러스터링이 필요할 때 사용합니다.

## 스킬 메타데이터

| | |
|---|---|
| Source | Optional — `hermes skills install official/mlops/faiss`로 설치 |
| Path | `optional-skills/mlops/faiss` |
| Version | `1.0.0` |
| Author | Orchestra Research |
| License | MIT |
| Dependencies | `faiss-cpu`, `numpy` |
| Platforms | linux, macos, windows |
| Tags | `Vector Search`, `FAISS`, `Machine Learning`, `Similarity Search`, `Clustering`, `Nearest Neighbors` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되어 있을 때 에이전트가 지침으로 보는 내용입니다.
:::

# FAISS (Facebook AI Similarity Search)

효율적인 유사도 검색 및 밀집 벡터 클러스터링을 위한 라이브러리입니다.

## FAISS를 사용해야 할 때

**다음과 같은 경우 FAISS를 사용하세요:**
- 대규모 문서 세트, 이미지 등에서 시맨틱 검색이나 추천 시스템을 구축할 때
- 수백만 또는 수십억 개의 벡터 데이터셋에서 빠르고 확장 가능한 K-최근접 이웃(KNN) 검색이 필요할 때
- RAM에 맞지 않는 대규모 벡터 데이터셋을 관리해야 할 때
- 엄청나게 많은 데이터 포인트를 클러스터링할 때 (예: k-means)

**주요 특징:**
- 수백만 또는 수십억 개의 벡터로 확장 가능
- 양자화(quantization) 및 차원 축소(PCA)를 통해 RAM에 맞게 인덱스 압축
- C++ 및 GPU 구현(선택 사항)으로 매우 빠름

## 빠른 시작

### 설치

```bash
# CPU 전용 (일반적으로 충분함)
pip install faiss-cpu

# GPU가 있는 경우 (더 빠름)
pip install faiss-gpu
```

### 기본 인덱스 (IndexFlatL2) - 가장 간단하고 정확함

가장 정확한 방법입니다 (완전 탐색 - exact search). 데이터셋이 약 100만 개 미만의 벡터인 경우 이 방법을 사용하세요. 데이터셋이 메모리에 맞아야 합니다.

```python
import faiss
import numpy as np

# 1. 가짜 데이터 생성
d = 64                           # 벡터 차원 (예: 임베딩 모델 출력 차원)
nb = 100000                      # 데이터베이스 크기
nq = 10                          # 쿼리 개수
np.random.seed(1234)             
xb = np.random.random((nb, d)).astype('float32') # 데이터베이스 벡터
xq = np.random.random((nq, d)).astype('float32') # 쿼리 벡터

# 2. 인덱스 생성 및 벡터 추가
index = faiss.IndexFlatL2(d)   # L2(유클리드) 거리를 사용하여 인덱스 생성
index.add(xb)                  # 데이터베이스 벡터를 인덱스에 추가
print(f"Number of vectors in index: {index.ntotal}")

# 3. 검색
k = 4                          # 쿼리당 반환할 가장 가까운 이웃의 수
distances, indices = index.search(xq, k) # 검색 실행

print("가장 가까운 이웃의 인덱스 (행 = 쿼리, 열 = 이웃 순위):")
print(indices)
print("\n가장 가까운 이웃의 거리:")
print(distances)
```

## 고급 인덱스 유형

FAISS는 속도, 메모리, 정확도 사이의 절충을 제공합니다. 데이터셋이 너무 커서 메모리에 맞지 않거나 완전 탐색이 너무 느린 경우 근사(Approximate) 방법을 사용하세요.

### 1. Inverted File (IVF) - 더 빠른 검색을 위한 분할

검색 공간을 보로노이 셀(Voronoi cells)로 분할합니다. 검색 시 전체 데이터셋이 아닌 쿼리에 가장 가까운 셀들만 탐색합니다.

```python
nlist = 100 # 생성할 셀/클러스터의 수 (보통 nb를 nlist 범위로 나눈 값이 되도록 선택)
quantizer = faiss.IndexFlatL2(d)  # 셀(클러스터)의 중심을 찾기 위한 양자화기
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2) # 메트릭은 L2 또는 INNER_PRODUCT(코사인 유사도용) 지정 가능

# IVF 인덱스는 추가하기 전에 데이터 분포를 파악하기 위해 '학습'이 필요합니다.
index_ivf.train(xb) 
index_ivf.add(xb)

# 검색 파라미터 (nprobe) 조정 - 속도 대 정확도 절충
index_ivf.nprobe = 10 # 10개의 가장 가까운 셀(클러스터) 방문 (기본값은 1)
distances, indices = index_ivf.search(xq, k)
```

### 2. Product Quantization (PQ) - 메모리 절약을 위한 압축

벡터를 압축하여 메모리 크기를 대폭 줄입니다 (손실 압축). `IndexIVFPQ`는 IVF와 PQ를 결합하여 속도와 메모리를 모두 최적화한 가장 인기 있는 대규모 방식입니다.

```python
nlist = 100 # 클러스터 수
m = 8       # 원본 차원 d=64를 부분 벡터(sub-vectors)로 나누는 수 (d는 m의 배수여야 함)
bits = 8    # 각 부분 벡터를 인코딩하는 데 사용되는 비트 수 (보통 8)

quantizer = faiss.IndexFlatL2(d)  
index_pq = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits)

index_pq.train(xb)
index_pq.add(xb)
index_pq.nprobe = 10
distances, indices = index_pq.search(xq, k)
```

### 3. HNSW (Hierarchical Navigable Small World) - 빠른 탐색을 위한 그래프

매우 빠르고 매우 정확하지만 메모리를 많이 사용합니다. 속도와 정확도가 가장 중요할 때 사용하세요.

```python
M = 32 # 그래프의 각 노드가 연결되는 이웃의 수 (보통 16~64)
index_hnsw = faiss.IndexHNSWFlat(d, M)

# HNSW는 학습이 필요 없습니다.
index_hnsw.add(xb)
distances, indices = index_hnsw.search(xq, k)
```

## 모범 사례

1. **학습(Training)**: `IVF`나 `PQ`와 같은 인덱스는 `.add()` 전에 `.train()`이 필요합니다. 훈련 세트는 대표성이 있어야 하며 최소 클러스터 수(nlist)보다 커야 합니다.
2. **거리 측정(Distance Metrics)**: 
   - `faiss.METRIC_L2` (기본값): 유클리드 거리. 값이 작을수록 더 가깝습니다.
   - `faiss.METRIC_INNER_PRODUCT`: 내적(Inner Product). 코사인 유사도를 위해 벡터를 정규화한 후 내적을 수행하세요. 값이 클수록 더 가깝습니다.
3. **정규화(Normalization)**: 코사인 유사도를 원할 경우 FAISS에 전달하기 전에 NumPy/PyTorch를 사용하여 벡터를 L2 정규화(길이가 1이 되도록)하고 `IndexFlatIP`(Inner Product)를 사용하세요.

```python
# 코사인 유사도 예시:
faiss.normalize_L2(xb)
faiss.normalize_L2(xq)
index_ip = faiss.IndexFlatIP(d)
index_ip.add(xb)
distances, indices = index_ip.search(xq, k) # 거리가 클수록 (1.0에 가까울수록) 더 유사함
```

## 인덱스 저장 및 불러오기

```python
# 인덱스 저장
faiss.write_index(index_pq, "my_index.index")

# 인덱스 불러오기
loaded_index = faiss.read_index("my_index.index")
```

## 리소스

- **문서**: [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- **가이드**: [FAISS 인덱스 가이드라인](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index) (어떤 인덱스를 선택할지 결정할 때 유용함)
