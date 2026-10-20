---
title: "Faiss — Facebook 用于高效相似性搜索和密集向量聚类的库"
sidebar_label: "Faiss"
description: "Facebook 用于高效相似性搜索和密集向量聚类的库"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Faiss

Facebook 用于高效相似性搜索和密集向量聚类的库。支持数十亿向量、GPU 加速以及多种索引类型（Flat、IVF、HNSW）。适用于快速 k-NN 搜索、大规模向量检索，或仅需纯相似性搜索而无需元数据的场景。最适合高性能应用。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/mlops/faiss` 安装 |
| 路径 | `optional-skills/mlops/faiss` |
| 版本 | `1.0.0` |
| 作者 | Orchestra Research |
| 许可证 | MIT |
| 依赖项 | `faiss-cpu`, `faiss-gpu`, `numpy` |
| 平台 | linux, macos |
| 标签 | `RAG`, `FAISS`, `Similarity Search`, `Vector Search`, `Facebook AI`, `GPU Acceleration`, `Billion-Scale`, `K-NN`, `HNSW`, `High Performance`, `Large Scale` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

# FAISS - 高效相似性搜索

Facebook AI 用于十亿级向量相似性搜索的库。

## 何时使用 FAISS

**在以下情况下使用 FAISS：**
- 需要对大型向量数据集（百万/十亿级）进行快速相似性搜索
- 需要 GPU 加速
- 纯向量相似性搜索（无需元数据过滤）
- 对高吞吐量、低延迟有严格要求
- 对 embedding（嵌入向量）进行离线/批量处理

**指标**：
- **GitHub 31,700+ 星**
- Meta/Facebook AI Research 出品
- **支持数十亿向量**
- **C++** 并提供 Python 绑定

**以下情况请使用替代方案**：
- **Chroma/Pinecone**：需要元数据过滤
- **Weaviate**：需要完整数据库功能
- **Annoy**：更简单，功能较少

## 快速开始

### 安装

```bash
# 仅 CPU
pip install faiss-cpu

# GPU 支持
pip install faiss-gpu
```

### 基本用法

```python
import faiss
import numpy as np

# 创建示例数据（1000 个向量，128 维）
d = 128
nb = 1000
vectors = np.random.random((nb, d)).astype('float32')

# 创建索引
index = faiss.IndexFlatL2(d)  # L2 距离
index.add(vectors)             # 添加向量

# 搜索
k = 5  # 查找 5 个最近邻
query = np.random.random((1, d)).astype('float32')
distances, indices = index.search(query, k)

print(f"Nearest neighbors: {indices}")
print(f"Distances: {distances}")
```

## 索引类型

### 1. Flat（精确搜索）

```python
# L2（欧氏）距离
index = faiss.IndexFlatL2(d)

# 内积（归一化后等同于余弦相似度）
index = faiss.IndexFlatIP(d)

# 速度最慢，精度最高
```

### 2. IVF（倒排文件）- 快速近似搜索

```python
# 创建量化器
quantizer = faiss.IndexFlatL2(d)

# 含 100 个聚类的 IVF 索引
nlist = 100
index = faiss.IndexIVFFlat(quantizer, d, nlist)

# 在数据上训练
index.train(vectors)

# 添加向量
index.add(vectors)

# 搜索（nprobe = 搜索的聚类数）
index.nprobe = 10
distances, indices = index.search(query, k)
```

### 3. HNSW（分层小世界图）- 质量/速度最佳平衡

```python
# HNSW 索引
M = 32  # 每层连接数
index = faiss.IndexHNSWFlat(d, M)

# 无需训练
index.add(vectors)

# 搜索
distances, indices = index.search(query, k)
```

### 4. 乘积量化（Product Quantization）- 内存高效

```python
# PQ 可将内存减少 16-32 倍
m = 8   # 子量化器数量
nbits = 8
index = faiss.IndexPQ(d, m, nbits)

# 训练并添加
index.train(vectors)
index.add(vectors)
```

## 保存与加载

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

# 创建 FAISS 向量存储
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())

# 保存
vectorstore.save_local("faiss_index")

# 加载
vectorstore = FAISS.load_local(
    "faiss_index",
    OpenAIEmbeddings(),
    allow_dangerous_deserialization=True
)

# 搜索
results = vectorstore.similarity_search("query", k=5)
```

## LlamaIndex 集成

```python
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

# 创建 FAISS 索引
d = 1536
faiss_index = faiss.IndexFlatL2(d)

vector_store = FaissVectorStore(faiss_index=faiss_index)
```

## 最佳实践

1. **选择合适的索引类型** — 10K 以下用 Flat，10K-1M 用 IVF，追求质量用 HNSW
2. **余弦相似度需归一化** — 对归一化向量使用 IndexFlatIP
3. **大数据集使用 GPU** — 速度提升 10-100 倍
4. **保存已训练的索引** — 训练成本较高
5. **调整 nprobe/ef_search** — 平衡速度与精度
6. **监控内存使用** — 大数据集使用 PQ
7. **批量查询** — 提升 GPU 利用率

## 性能对比

| 索引类型 | 构建时间 | 搜索时间 | 内存占用 | 精度 |
|----------|----------|----------|----------|------|
| Flat | 快 | 慢 | 高 | 100% |
| IVF | 中等 | 快 | 中等 | 95-99% |
| HNSW | 慢 | 最快 | 高 | 99% |
| PQ | 中等 | 快 | 低 | 90-95% |

## 资源

- **GitHub**：https://github.com/facebookresearch/faiss ⭐ 31,700+
- **Wiki**：https://github.com/facebookresearch/faiss/wiki
- **许可证**：MIT