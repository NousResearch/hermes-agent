---
title: "Qdrant Vector Search — 用于 RAG 和语义搜索的高性能向量相似度搜索引擎"
sidebar_label: "Qdrant Vector Search"
description: "用于 RAG 和语义搜索的高性能向量相似度搜索引擎"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Qdrant Vector Search

用于 RAG 和语义搜索的高性能向量相似度搜索引擎。适用于构建需要快速最近邻搜索、带过滤的混合搜索，或基于 Rust 高性能的可扩展向量存储的生产级 RAG 系统。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 使用 `hermes skills install official/mlops/qdrant` 安装 |
| 路径 | `optional-skills/mlops/qdrant` |
| 版本 | `1.0.0` |
| 作者 | Orchestra Research |
| 许可证 | MIT |
| 依赖 | `qdrant-client>=1.12.0` |
| 平台 | linux, macos, windows |
| 标签 | `RAG`, `Vector Search`, `Qdrant`, `Semantic Search`, `Embeddings`, `Similarity Search`, `HNSW`, `Production`, `Distributed` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

# Qdrant - 向量相似度搜索引擎

用 Rust 编写的高性能向量数据库，适用于生产级 RAG 和语义搜索。

## 何时使用 Qdrant

**在以下情况下使用 Qdrant：**
- 构建需要低延迟的生产级 RAG 系统
- 需要混合搜索（向量 + 元数据过滤）
- 需要通过分片/副本实现水平扩展
- 希望本地部署并完全掌控数据
- 每条记录需要多向量存储（稠密 + 稀疏）
- 构建实时推荐系统

**核心特性：**
- **Rust 驱动**：内存安全，高性能
- **丰富过滤**：在搜索时按任意 payload 字段过滤
- **多向量**：每个点支持稠密、稀疏、多稠密向量
- **量化**：标量、乘积、二值量化，节省内存
- **分布式**：Raft 共识、分片、副本
- **REST + gRPC**：两套 API 功能完全对等

**以下情况请使用替代方案：**
- **Chroma**：更简单的配置，嵌入式使用场景
- **FAISS**：追求极致原始速度，研究/批处理场景
- **Pinecone**：完全托管，零运维偏好
- **Weaviate**：偏好 GraphQL，内置向量化器

## 快速开始

### 安装

```bash
# Python 客户端
pip install qdrant-client

# Docker（推荐用于开发）
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Docker 持久化存储
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

### 基本用法

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# 连接到 Qdrant
client = QdrantClient(host="localhost", port=6333)

# 创建集合
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# 插入带 payload 的向量
client.upsert(
    collection_name="documents",
    points=[
        PointStruct(
            id=1,
            vector=[0.1, 0.2, ...],  # 384 维向量
            payload={"title": "Doc 1", "category": "tech"}
        ),
        PointStruct(
            id=2,
            vector=[0.3, 0.4, ...],
            payload={"title": "Doc 2", "category": "science"}
        )
    ]
)

# 带过滤的搜索
results = client.search(
    collection_name="documents",
    query_vector=[0.15, 0.25, ...],
    query_filter={
        "must": [{"key": "category", "match": {"value": "tech"}}]
    },
    limit=10
)

for point in results:
    print(f"ID: {point.id}, Score: {point.score}, Payload: {point.payload}")
```

## 核心概念

### Points（点）— 基本数据单元

```python
from qdrant_client.models import PointStruct

# Point = ID + 向量 + Payload
point = PointStruct(
    id=123,                              # 整数或 UUID 字符串
    vector=[0.1, 0.2, 0.3, ...],        # 稠密向量
    payload={                            # 任意 JSON 元数据
        "title": "Document title",
        "category": "tech",
        "timestamp": 1699900000,
        "tags": ["python", "ml"]
    }
)

# 批量 upsert（推荐）
client.upsert(
    collection_name="documents",
    points=[point1, point2, point3],
    wait=True  # 等待索引完成
)
```

### Collections（集合）— 向量容器

```python
from qdrant_client.models import VectorParams, Distance, HnswConfigDiff

# 使用 HNSW 配置创建集合
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(
        size=384,                        # 向量维度
        distance=Distance.COSINE         # COSINE、EUCLID、DOT、MANHATTAN
    ),
    hnsw_config=HnswConfigDiff(
        m=16,                            # 每个节点的连接数（默认 16）
        ef_construct=100,                # 构建时精度（默认 100）
        full_scan_threshold=10000        # 低于此值切换为暴力搜索
    ),
    on_disk_payload=True                 # 将 payload 存储在磁盘上
)

# 集合信息
info = client.get_collection("documents")
print(f"Points: {info.points_count}, Vectors: {info.vectors_count}")
```

### 距离度量

| 度量 | 使用场景 | 范围 |
|--------|----------|-------|
| `COSINE` | 文本 embedding、归一化向量 | 0 到 2 |
| `EUCLID` | 空间数据、图像特征 | 0 到 ∞ |
| `DOT` | 推荐系统、非归一化向量 | -∞ 到 ∞ |
| `MANHATTAN` | 稀疏特征、离散数据 | 0 到 ∞ |

## 搜索操作

### 基本搜索

```python
# 简单最近邻搜索
results = client.search(
    collection_name="documents",
    query_vector=[0.1, 0.2, ...],
    limit=10,
    with_payload=True,
    with_vectors=False  # 不返回向量（更快）
)
```

### 带过滤的搜索

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

# 复杂过滤
results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    query_filter=Filter(
        must=[
            FieldCondition(key="category", match=MatchValue(value="tech")),
            FieldCondition(key="timestamp", range=Range(gte=1699000000))
        ],
        must_not=[
            FieldCondition(key="status", match=MatchValue(value="archived"))
        ]
    ),
    limit=10
)

# 简写过滤语法
results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    query_filter={
        "must": [
            {"key": "category", "match": {"value": "tech"}},
            {"key": "price", "range": {"gte": 10, "lte": 100}}
        ]
    },
    limit=10
)
```

### 批量搜索

```python
from qdrant_client.models import SearchRequest

# 单次请求中执行多个查询
results = client.search_batch(
    collection_name="documents",
    requests=[
        SearchRequest(vector=[0.1, ...], limit=5),
        SearchRequest(vector=[0.2, ...], limit=5, filter={"must": [...]}),
        SearchRequest(vector=[0.3, ...], limit=10)
    ]
)
```

## RAG 集成

### 与 sentence-transformers 集成

```python
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# 初始化
encoder = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(host="localhost", port=6333)

# 创建集合
client.create_collection(
    collection_name="knowledge_base",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# 索引文档
documents = [
    {"id": 1, "text": "Python is a programming language", "source": "wiki"},
    {"id": 2, "text": "Machine learning uses algorithms", "source": "textbook"},
]

points = [
    PointStruct(
        id=doc["id"],
        vector=encoder.encode(doc["text"]).tolist(),
        payload={"text": doc["text"], "source": doc["source"]}
    )
    for doc in documents
]
client.upsert(collection_name="knowledge_base", points=points)

# RAG 检索
def retrieve(query: str, top_k: int = 5) -> list[dict]:
    query_vector = encoder.encode(query).tolist()
    results = client.search(
        collection_name="knowledge_base",
        query_vector=query_vector,
        limit=top_k
    )
    return [{"text": r.payload["text"], "score": r.score} for r in results]

# 在 RAG 流水线中使用
context = retrieve("What is Python?")
prompt = f"Context: {context}\n\nQuestion: What is Python?"
```

### 与 LangChain 集成

```python
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Qdrant.from_documents(documents, embeddings, url="http://localhost:6333", collection_name="docs")
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

### 与 LlamaIndex 集成

```python
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

vector_store = QdrantVectorStore(client=client, collection_name="llama_docs")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
query_engine = index.as_query_engine()
```

## 多向量支持

### 命名向量（不同 embedding 模型）

```python
from qdrant_client.models import VectorParams, Distance

# 包含多种向量类型的集合
client.create_collection(
    collection_name="hybrid_search",
    vectors_config={
        "dense": VectorParams(size=384, distance=Distance.COSINE),
        "sparse": VectorParams(size=30000, distance=Distance.DOT)
    }
)

# 插入命名向量
client.upsert(
    collection_name="hybrid_search",
    points=[
        PointStruct(
            id=1,
            vector={
                "dense": dense_embedding,
                "sparse": sparse_embedding
            },
            payload={"text": "document text"}
        )
    ]
)

# 搜索指定向量
results = client.search(
    collection_name="hybrid_search",
    query_vector=("dense", query_dense),  # 指定使用哪个向量
    limit=10
)
```

### 稀疏向量（BM25、SPLADE）

```python
from qdrant_client.models import SparseVectorParams, SparseIndexParams, SparseVector

# 包含稀疏向量的集合
client.create_collection(
    collection_name="sparse_search",
    vectors_config={},
    sparse_vectors_config={"text": SparseVectorParams(index=SparseIndexParams(on_disk=False))}
)

# 插入稀疏向量
client.upsert(
    collection_name="sparse_search",
    points=[PointStruct(id=1, vector={"text": SparseVector(indices=[1, 5, 100], values=[0.5, 0.8, 0.2])}, payload={"text": "document"})]
)
```

## 量化（内存优化）

```python
from qdrant_client.models import ScalarQuantization, ScalarQuantizationConfig, ScalarType

# 标量量化（内存减少 4 倍）
client.create_collection(
    collection_name="quantized",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    quantization_config=ScalarQuantization(
        scalar=ScalarQuantizationConfig(
            type=ScalarType.INT8,
            quantile=0.99,        # 裁剪异常值
            always_ram=True      # 将量化数据保留在 RAM 中
        )
    )
)

# 带重新评分的搜索
results = client.search(
    collection_name="quantized",
    query_vector=query,
    search_params={"quantization": {"rescore": True}},  # 对 top 结果重新评分
    limit=10
)
```

## Payload 索引

```python
from qdrant_client.models import PayloadSchemaType

# 创建 payload 索引以加速过滤
client.create_payload_index(
    collection_name="documents",
    field_name="category",
    field_schema=PayloadSchemaType.KEYWORD
)

client.create_payload_index(
    collection_name="documents",
    field_name="timestamp",
    field_schema=PayloadSchemaType.INTEGER
)

# 索引类型：KEYWORD、INTEGER、FLOAT、GEO、TEXT（全文）、BOOL
```

## 生产部署

### Qdrant Cloud

```python
from qdrant_client import QdrantClient

# 连接到 Qdrant Cloud
client = QdrantClient(
    url="https://your-cluster.cloud.qdrant.io",
    api_key="your-api-key"
)
```

### 性能调优

```python
# 优化搜索速度（更高召回率）
client.update_collection(
    collection_name="documents",
    hnsw_config=HnswConfigDiff(ef_construct=200, m=32)
)

# 优化索引速度（批量加载）
client.update_collection(
    collection_name="documents",
    optimizer_config={"indexing_threshold": 20000}
)
```

## 最佳实践

1. **批量操作** — 使用批量 upsert/search 提升效率
2. **Payload 索引** — 对过滤中使用的字段建立索引
3. **量化** — 对大型集合（>100 万向量）启用量化
4. **分片** — 对超过 1000 万向量的集合使用分片
5. **磁盘存储** — 对大型 payload 启用 `on_disk_payload`
6. **连接池** — 复用客户端实例

## 常见问题

**带过滤的搜索速度慢：**
```python
# 为过滤字段创建 payload 索引
client.create_payload_index(
    collection_name="docs",
    field_name="category",
    field_schema=PayloadSchemaType.KEYWORD
)
```

**内存不足：**
```python
# 启用量化和磁盘存储
client.create_collection(
    collection_name="large_collection",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    quantization_config=ScalarQuantization(...),
    on_disk_payload=True
)
```

**连接问题：**
```python
# 使用超时和重试
client = QdrantClient(
    host="localhost",
    port=6333,
    timeout=30,
    prefer_grpc=True  # gRPC 性能更佳
)
```

## 参考资料

- **[高级用法](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/qdrant/references/advanced-usage.md)** — 分布式模式、混合搜索、推荐系统
- **[故障排查](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/qdrant/references/troubleshooting.md)** — 常见问题、调试、性能调优

## 资源

- **GitHub**：https://github.com/qdrant/qdrant（22k+ stars）
- **文档**：https://qdrant.tech/documentation/
- **Python 客户端**：https://github.com/qdrant/qdrant-client
- **Cloud**：https://cloud.qdrant.io
- **版本**：1.12.0+
- **许可证**：Apache 2.0