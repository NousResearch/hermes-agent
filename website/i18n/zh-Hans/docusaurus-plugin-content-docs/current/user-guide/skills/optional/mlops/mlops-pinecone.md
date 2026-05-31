---
title: "Pinecone — 面向生产级 AI 应用的托管向量数据库"
sidebar_label: "Pinecone"
description: "面向生产级 AI 应用的托管向量数据库"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Pinecone

面向生产级 AI 应用的托管向量数据库。全托管、自动扩缩容，支持混合搜索（稠密 + 稀疏向量）、元数据过滤和命名空间。低延迟（&lt;100ms p95）。适用于生产级 RAG、推荐系统或大规模语义搜索。最适合 serverless（无服务器）托管基础设施。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/mlops/pinecone` 安装 |
| 路径 | `optional-skills/mlops/pinecone` |
| 版本 | `1.0.0` |
| 作者 | Orchestra Research |
| 许可证 | MIT |
| 依赖 | `pinecone-client` |
| 平台 | linux, macos, windows |
| 标签 | `RAG`, `Pinecone`, `Vector Database`, `Managed Service`, `Serverless`, `Hybrid Search`, `Production`, `Auto-Scaling`, `Low Latency`, `Recommendations` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

# Pinecone - 托管向量数据库

面向生产级 AI 应用的向量数据库。

## 何时使用 Pinecone

**适用场景：**
- 需要托管的 serverless 向量数据库
- 生产级 RAG 应用
- 需要自动扩缩容
- 对低延迟有严格要求（&lt;100ms）
- 不想自行管理基础设施
- 需要混合搜索（稠密 + 稀疏向量）

**指标**：
- 全托管 SaaS
- 自动扩缩容至数十亿向量
- **p95 延迟 &lt;100ms**
- 99.9% 正常运行时间 SLA

**改用其他方案的场景**：
- **Chroma**：自托管、开源
- **FAISS**：离线、纯相似度搜索
- **Weaviate**：自托管、功能更丰富

## 快速开始

### 安装

```bash
pip install pinecone-client
```

### 基本用法

```python
from pinecone import Pinecone, ServerlessSpec

# Initialize
pc = Pinecone(api_key="your-api-key")

# Create index
pc.create_index(
    name="my-index",
    dimension=1536,  # Must match embedding dimension
    metric="cosine",  # or "euclidean", "dotproduct"
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

# Connect to index
index = pc.Index("my-index")

# Upsert vectors
index.upsert(vectors=[
    {"id": "vec1", "values": [0.1, 0.2, ...], "metadata": {"category": "A"}},
    {"id": "vec2", "values": [0.3, 0.4, ...], "metadata": {"category": "B"}}
])

# Query
results = index.query(
    vector=[0.1, 0.2, ...],
    top_k=5,
    include_metadata=True
)

print(results["matches"])
```

## 核心操作

### 创建索引

```python
# Serverless (recommended)
pc.create_index(
    name="my-index",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",         # or "gcp", "azure"
        region="us-east-1"
    )
)

# Pod-based (for consistent performance)
from pinecone import PodSpec

pc.create_index(
    name="my-index",
    dimension=1536,
    metric="cosine",
    spec=PodSpec(
        environment="us-east1-gcp",
        pod_type="p1.x1"
    )
)
```

### 插入向量（Upsert）

```python
# Single upsert
index.upsert(vectors=[
    {
        "id": "doc1",
        "values": [0.1, 0.2, ...],  # 1536 dimensions
        "metadata": {
            "text": "Document content",
            "category": "tutorial",
            "timestamp": "2025-01-01"
        }
    }
])

# Batch upsert (recommended)
vectors = [
    {"id": f"vec{i}", "values": embedding, "metadata": metadata}
    for i, (embedding, metadata) in enumerate(zip(embeddings, metadatas))
]

index.upsert(vectors=vectors, batch_size=100)
```

### 查询向量

```python
# Basic query
results = index.query(
    vector=[0.1, 0.2, ...],
    top_k=10,
    include_metadata=True,
    include_values=False
)

# With metadata filtering
results = index.query(
    vector=[0.1, 0.2, ...],
    top_k=5,
    filter={"category": {"$eq": "tutorial"}}
)

# Namespace query
results = index.query(
    vector=[0.1, 0.2, ...],
    top_k=5,
    namespace="production"
)

# Access results
for match in results["matches"]:
    print(f"ID: {match['id']}")
    print(f"Score: {match['score']}")
    print(f"Metadata: {match['metadata']}")
```

### 元数据过滤

```python
# Exact match
filter = {"category": "tutorial"}

# Comparison
filter = {"price": {"$gte": 100}}  # $gt, $gte, $lt, $lte, $ne

# Logical operators
filter = {
    "$and": [
        {"category": "tutorial"},
        {"difficulty": {"$lte": 3}}
    ]
}  # Also: $or

# In operator
filter = {"tags": {"$in": ["python", "ml"]}}
```

## 命名空间

```python
# Partition data by namespace
index.upsert(
    vectors=[{"id": "vec1", "values": [...]}],
    namespace="user-123"
)

# Query specific namespace
results = index.query(
    vector=[...],
    namespace="user-123",
    top_k=5
)

# List namespaces
stats = index.describe_index_stats()
print(stats['namespaces'])
```

## 混合搜索（稠密 + 稀疏向量）

```python
# Upsert with sparse vectors
index.upsert(vectors=[
    {
        "id": "doc1",
        "values": [0.1, 0.2, ...],  # Dense vector
        "sparse_values": {
            "indices": [10, 45, 123],  # Token IDs
            "values": [0.5, 0.3, 0.8]   # TF-IDF scores
        },
        "metadata": {"text": "..."}
    }
])

# Hybrid query
results = index.query(
    vector=[0.1, 0.2, ...],
    sparse_vector={
        "indices": [10, 45],
        "values": [0.5, 0.3]
    },
    top_k=5,
    alpha=0.5  # 0=sparse, 1=dense, 0.5=hybrid
)
```

## LangChain 集成

```python
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# Create vector store
vectorstore = PineconeVectorStore.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(),
    index_name="my-index"
)

# Query
results = vectorstore.similarity_search("query", k=5)

# With metadata filter
results = vectorstore.similarity_search(
    "query",
    k=5,
    filter={"category": "tutorial"}
)

# As retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
```

## LlamaIndex 集成

```python
from llama_index.vector_stores.pinecone import PineconeVectorStore

# Connect to Pinecone
pc = Pinecone(api_key="your-key")
pinecone_index = pc.Index("my-index")

# Create vector store
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# Use in LlamaIndex
from llama_index.core import StorageContext, VectorStoreIndex

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
```

## 索引管理

```python
# List indices
indexes = pc.list_indexes()

# Describe index
index_info = pc.describe_index("my-index")
print(index_info)

# Get index stats
stats = index.describe_index_stats()
print(f"Total vectors: {stats['total_vector_count']}")
print(f"Namespaces: {stats['namespaces']}")

# Delete index
pc.delete_index("my-index")
```

## 删除向量

```python
# Delete by ID
index.delete(ids=["vec1", "vec2"])

# Delete by filter
index.delete(filter={"category": "old"})

# Delete all in namespace
index.delete(delete_all=True, namespace="test")

# Delete entire index
index.delete(delete_all=True)
```

## 最佳实践

1. **使用 serverless** — 自动扩缩容，成本效益高
2. **批量 upsert** — 效率更高（每批 100-200 条）
3. **添加元数据** — 启用过滤功能
4. **使用命名空间** — 按用户/租户隔离数据
5. **监控用量** — 查看 Pinecone 控制台
6. **优化过滤器** — 对频繁过滤的字段建立索引
7. **用免费套餐测试** — 1 个索引，10 万向量免费
8. **使用混合搜索** — 质量更优
9. **设置合适的维度** — 与 embedding 模型匹配
10. **定期备份** — 导出重要数据

## 性能

| 操作 | 延迟 | 备注 |
|-----------|---------|-------|
| Upsert | ~50-100ms | 每批次 |
| 查询（p50） | ~50ms | 取决于索引大小 |
| 查询（p95） | ~100ms | SLA 目标 |
| 元数据过滤 | ~+10-20ms | 额外开销 |

## 定价（截至 2025 年）

**Serverless**：
- 每百万读取单元 $0.096
- 每百万写入单元 $0.06
- 每 GB 存储/月 $0.06

**免费套餐**：
- 1 个 serverless 索引
- 10 万向量（1536 维）
- 非常适合原型开发

## 资源

- **官网**：https://www.pinecone.io
- **文档**：https://docs.pinecone.io
- **控制台**：https://app.pinecone.io
- **定价**：https://www.pinecone.io/pricing