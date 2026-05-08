---
title: "Faiss — Facebook 的高效稠密向量相似性搜索和聚类库"
sidebar_label: "Faiss"
description: "Facebook 的高效稠密向量相似性搜索和聚类库"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Faiss

Facebook 的高效稠密向量相似性搜索和聚类库。支持十亿级向量、GPU 加速和各种索引类型（Flat、IVF、HNSW）。适用于快速 k-NN 搜索、大规模向量检索，或需要纯相似性搜索（无元数据过滤）的场景。最适合高性能应用。

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/mlops/faiss` |
| Path | `optional-skills/mlops/faiss` |
| Version | `1.0.0` |
| Author | Orchestra Research |
| License | MIT |
| Dependencies | `faiss-cpu`, `faiss-gpu`, `numpy` |
| Tags | `RAG`, `FAISS`, `Similarity Search`, `Vector Search`, `Facebook AI`, `GPU Acceleration`, `Billion-Scale`, `K-NN`, `HNSW`, `High Performance`, `Large Scale` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# FAISS - Efficient Similarity Search

Facebook AI 的十亿级向量相似性搜索库。

## 何时使用 FAISS

**Use FAISS when:**
- Need fast similarity search on large vector datasets (millions/billions)
- GPU 加速 required
- Pure vector similarity (no metadata filtering needed)
- High throughput, low latency critical
- Offline/batch processing of embeddings

**Metrics**:
- **31,700+ GitHub stars**
- Meta/Facebook AI Research
- **Handles billions of vectors**
- **C++** with Python bindings

**替代方案**:
- **Chroma/Pinecone**: Need metadata filtering
- **Weaviate**: Need full database features
- **Annoy**: Simpler, fewer features

## 快速入门

### 安装

```bash
# CPU only
pip install faiss-cpu

# GPU support
pip install faiss-gpu
```

### Basic usage

```python
import faiss
import numpy as np

# Create sample data (1000 vectors, 128 dimensions)
d = 128
nb = 1000
vectors = np.random.random((nb, d)).astype('float32')

# 创建集合
index = faiss.IndexFlatL2(d)  # L2 distance
index.add(vectors)             # Add vectors

# Search
k = 5  # Find 5 nearest neighbors
query = np.random.random((1, d)).astype('float32')
distances, indices = index.search(query, k)

print(f"Nearest neighbors: {indices}")
print(f"Distances: {distances}")
```

## 索引类型

### 1. Flat（精确搜索）

```python
# L2 (Euclidean) distance
index = faiss.IndexFlatL2(d)

# Inner product (cosine similarity if normalized)
index = faiss.IndexFlatIP(d)

# 最慢，最准确
```

### 2. IVF (inverted file) - 快速近似

```python
# Create quantizer
quantizer = faiss.IndexFlatL2(d)

# IVF index with 100 clusters
nlist = 100
index = faiss.IndexIVFFlat(quantizer, d, nlist)

# 训练 on data
index.train(vectors)

# Add vectors
index.add(vectors)

# Search (nprobe = clusters to search)
index.nprobe = 10
distances, indices = index.search(query, k)
```

### 3. HNSW（层次化 NSW） - 最佳质量/速度

```python
# HNSW index
M = 32  # Number of connections per layer
index = faiss.IndexHNSWFlat(d, M)

# No training needed
index.add(vectors)

# Search
distances, indices = index.search(query, k)
```

### 4. 乘积量化 - 内存高效

```python
# PQ reduces memory by 16-32×
m = 8   # Number of subquantizers
nbits = 8
index = faiss.IndexPQ(d, m, nbits)

# 训练 and add
index.train(vectors)
index.add(vectors)
```

## 保存和加载

```python
# 保存索引
faiss.write_index(index, "large.index")

# 加载索引
index = faiss.read_index("large.index")

# 继续使用
distances, indices = index.search(query, k)
```

## GPU 加速

```python
# 单 GPU
res = faiss.StandardGpuResources()
index_cpu = faiss.IndexFlatL2(d)
index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)  # GPU 0

# 多 GPU
index_gpu = faiss.index_cpu_to_all_gpus(index_cpu)

# 比 CPU 快 10-100 倍
```

## LangChain 集成

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Create FAISS vector store
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())

# 保存
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

## LlamaIndex 集成

```python
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

# Create FAISS index
d = 1536
faiss_index = faiss.IndexFlatL2(d)

vector_store = FaissVectorStore(faiss_index=faiss_index)
```

## Best practices

1. **Choose right index type** - Flat for &lt;10K, IVF for 10K-1M, HNSW for quality
2. **Normalize for cosine** - Use IndexFlatIP with normalized vectors
3. **Use GPU for large datasets** - 10-100× faster
4. **保存 trained indices** - 训练ing is expensive
5. **Tune nprobe/ef_search** - Balance speed/accuracy
6. **Monitor memory** - PQ for large datasets
7. **Batch queries** - Better GPU utilization

## Performance

| 索引类型 | 构建时间 | 搜索时间 | 内存 | 精度 |
|------------|------------|-------------|--------|----------|
| Flat | Fast | Slow | High | 100% |
| IVF | Medium | Fast | Medium | 95-99% |
| HNSW | Slow | Fastest | High | 99% |
| PQ | Medium | Fast | Low | 90-95% |

## Resources

- **GitHub**: https://github.com/facebookresearch/faiss ⭐ 31,700+
- **Wiki**: https://github.com/facebookresearch/faiss/wiki
- **License**: MIT
