---
title: "Pinecone — 面向生产 AI 应用的托管向量数据库"
sidebar_label: "Pinecone"
description: "面向生产 AI 应用的托管向量数据库"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Pinecone

面向生产 AI 应用的托管向量数据库。全托管、自动扩展，支持混合搜索（稠密 + 稀疏）、元数据过滤和命名空间。延迟低（p95 < 100ms）。适用于生产 RAG、推荐系统或大规模语义搜索。最适合无服务器托管基础设施。

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/mlops/pinecone` |
| Path | `optional-skills/mlops/pinecone` |
| Version | `1.0.0` |
| Author | Orchestra Research |
| License | MIT |
| Dependencies | `pinecone-client` |
| Tags | `RAG`, `Pinecone`, `Vector Database`, `Managed Service`, `无服务器`, `Hybrid Search`, `Production`, `Auto-Scaling`, `Low Latency`, `Recommendations` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Pinecone - Managed Vector Database

面向生产 AI 应用的向量数据库。

## 何时使用 Pinecone

**Use when:**
- Need managed, serverless vector database
- Production RAG applications
- Auto-scaling required
- Low latency critical (&lt;100ms)
- Don't want to manage infrastructure
- Need hybrid search (dense + sparse vectors)

**Metrics**:
- Fully managed SaaS
- Auto-scales to billions of vectors
- **p95 latency &lt;100ms**
- 99.9% uptime SLA

**替代方案**:
- **Chroma**: Self-hosted, open-source
- **FAISS**: Offline, pure similarity search
- **Weaviate**: Self-hosted with more features

## 快速入门

### 安装

```bash
pip install pinecone-client
```

### Basic usage

```python
from pinecone import Pinecone, 无服务器Spec

# 初始化
pc = Pinecone(api_key="your-api-key")

# 创建集合
pc.create_index(
    name="my-index",
    dimension=1536,  # Must match embedding dimension
    metric="cosine",  # or "euclidean", "dotproduct"
    spec=无服务器Spec(cloud="aws", region="us-east-1")
)

# Connect to index
index = pc.Index("my-index")

# 插入/更新向量
index.upsert(vectors=[
    {"id": "vec1", "values": [0.1, 0.2, ...], "metadata": {"category": "A"}},
    {"id": "vec2", "values": [0.3, 0.4, ...], "metadata": {"category": "B"}}
])

# 查询
results = index.query(
    vector=[0.1, 0.2, ...],
    top_k=5,
    include_metadata=True
)

print(results["matches"])
```

## Core operations

### 创建集合

```python
# 无服务器 (recommended)
pc.create_index(
    name="my-index",
    dimension=1536,
    metric="cosine",
    spec=无服务器Spec(
        cloud="aws",         # or "gcp", "azure"
        region="us-east-1"
    )
)

# 基于 Pod（一致的）
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

### 插入/更新向量

```python
# 单次插入/更新
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

# 批量插入/更新（推荐）
vectors = [
    {"id": f"vec{i}", "values": embedding, "metadata": metadata}
    for i, (embedding, metadata) in enumerate(zip(embeddings, metadatas))
]

index.upsert(vectors=vectors, batch_size=100)
```

### 查询向量

```python
# 基本查询
results = index.query(
    vector=[0.1, 0.2, ...],
    top_k=10,
    include_metadata=True,
    include_values=False
)

# 带元数据过滤器ing
results = index.query(
    vector=[0.1, 0.2, ...],
    top_k=5,
    filter={"category": {"$eq": "tutorial"}}
)

# 命名空间查询
results = index.query(
    vector=[0.1, 0.2, ...],
    top_k=5,
    namespace="production"
)

# 访问结果
for match in results["matches"]:
    print(f"ID: {match['id']}")
    print(f"Score: {match['score']}")
    print(f"Metadata: {match['metadata']}")
```

### Metadata filtering

```python
# 精确匹配
filter = {"category": "tutorial"}

# Comparison
filter = {"price": {"$gte": 100}}  # $gt, $gte, $lt, $lte, $ne

# 逻辑运算符
filter = {
    "$and": [
        {"category": "tutorial"},
        {"difficulty": {"$lte": 3}}
    ]
}  # Also: $or

# In operator
filter = {"tags": {"$in": ["python", "ml"]}}
```

## Namespaces

```python
# 按命名空间分区数据
index.upsert(
    vectors=[{"id": "vec1", "values": [...]}],
    namespace="user-123"
)

# 查询特定命名空间
results = index.query(
    vector=[...],
    namespace="user-123",
    top_k=5
)

# 列出命名空间
stats = index.describe_index_stats()
print(stats['namespaces'])
```

## Hybrid search (dense + sparse)

```python
# 插入带稀疏向量的数据
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

# 混合查询
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

# 创建向量存储
vectorstore = PineconeVectorStore.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(),
    index_name="my-index"
)

# 查询
results = vectorstore.similarity_search("query", k=5)

# 带元数据过滤器
results = vectorstore.similarity_search(
    "query",
    k=5,
    filter={"category": "tutorial"}
)

# 作为检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
```

## LlamaIndex 集成

```python
from llama_index.vector_stores.pinecone import PineconeVectorStore

# 连接到 Pinecone
pc = Pinecone(api_key="your-key")
pinecone_index = pc.Index("my-index")

# 创建向量存储
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# Use in LlamaIndex
from llama_index.core import 存储Context, VectorStoreIndex

storage_context = 存储Context.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
```

## Index management

```python
# 列出索引
indexes = pc.list_indexes()

# 描述索引
index_info = pc.describe_index("my-index")
print(index_info)

# 获取索引统计
stats = index.describe_index_stats()
print(f"Total vectors: {stats['total_vector_count']}")
print(f"Namespaces: {stats['namespaces']}")

# 删除索引
pc.delete_index("my-index")
```

## Delete vectors

```python
# 按 ID 删除
index.delete(ids=["vec1", "vec2"])

# 按过滤器删除
index.delete(filter={"category": "old"})

# 删除命名空间中的所有数据
index.delete(delete_all=True, namespace="test")

# 删除整个索引
index.delete(delete_all=True)
```

## Best practices

1. **使用无服务器** - Auto-scaling, cost-effective
2. **批量插入/更新** - More efficient (100-200 per batch)
3. **添加元数据** - Enable filtering
4. **使用命名空间** - Isolate data by user/tenant
5. **监控使用情况** - Check Pinecone dashboard
6. **优化过滤器** - Index frequently filtered fields
7. **使用免费层测试** - 1 index, 100K vectors free
8. **使用混合搜索** - Better quality
9. **设置适当的维度** - Match embedding model
10. **定期备份** - Export important data

## Performance

| 操作 | 延迟 | 备注 |
|-----------|---------|-------|
| 插入/更新 | ~50-100ms | Per batch |
| 查询（p50） | ~50ms | Depends on index size |
| 查询（p95） | ~100ms | SLA target |
| 元数据过滤 | ~+10-20ms | Additional overhead |

## Pricing (as of 2025)

**无服务器**:
- $0.096 per million read units
- $0.06 per million write units
- $0.06 per GB storage/month

**免费层**:
- 1 serverless index
- 100K vectors (1536 dimensions)
- Great for prototyping

## Resources

- **Website**: https://www.pinecone.io
- **Docs**: https://docs.pinecone.io
- **Console**: https://app.pinecone.io
- **Pricing**: https://www.pinecone.io/pricing
