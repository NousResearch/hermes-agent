---
title: "Pinecone — 프로덕션 RAG 및 검색을 위한 관리형 벡터 DB"
sidebar_label: "Pinecone"
description: "프로덕션 RAG 및 검색을 위한 관리형 벡터 DB"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Pinecone

프로덕션 AI 애플리케이션을 위한 관리형 벡터 DB입니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/mlops/pinecone`으로 설치 |
| 경로 | `optional-skills/mlops/pinecone` |
| 버전 | `1.0.1` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 종속성 | `pinecone` |
| 플랫폼 | linux, macos, windows |
| 태그 | `RAG`, `Pinecone`, `Vector Database`, `Managed Service`, `Serverless`, `Hybrid Search`, `Production`, `Auto-Scaling`, `Low Latency`, `Recommendations` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성화되면 에이전트는 이 내용을 지침으로 봅니다.
:::

# Pinecone - 관리형 벡터 데이터베이스

프로덕션 AI 애플리케이션을 위한 벡터 데이터베이스입니다.

## Pinecone을 사용할 때

**다음과 같은 경우 사용하세요:**
- 관리형 서버리스 벡터 데이터베이스가 필요한 경우
- 프로덕션 RAG 애플리케이션
- 자동 확장이 필요한 경우
- 짧은 지연 시간이 중요한 경우 (&lt;100ms)
- 인프라를 직접 관리하고 싶지 않은 경우
- 하이브리드 검색(밀집 벡터 + 희소 벡터)이 필요한 경우

**지표**:
- 완전 관리형 SaaS
- 수십억 개 벡터까지 자동 확장
- **p95 지연 시간 &lt;100ms**
- 가동 시간 SLA 99.9%

**대신 다음 대안을 사용하세요**:
- **Chroma**: 자체 호스팅, 오픈 소스
- **FAISS**: 오프라인, 순수 유사도 검색
- **Weaviate**: 더 많은 기능을 제공하는 자체 호스팅

## 빠른 시작

### 설치

```bash
pip install pinecone
```

> 참고: 이전 `pinecone-client` 패키지는 더 이상 사용되지 않습니다. `pinecone`(v5 이상, 현재 9.x)을 설치하세요. import 문은 `from pinecone import Pinecone`으로 유지됩니다.

### 기본 사용법

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

## 핵심 작업

### 인덱스 생성

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

### 벡터 업서트

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

### 벡터 쿼리

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

### 메타데이터 필터링

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

## 네임스페이스

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

## 하이브리드 검색(밀집 + 희소)

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
# NOTE: index.query() does NOT accept an `alpha` kwarg. Pinecone stores a
# single sparse-dense vector, so weighting must be applied by pre-scaling the
# query vectors before sending them. Use the hybrid_score_norm helper below
# (alpha * dense + (1 - alpha) * sparse; alpha=1 → pure dense, 0 → pure sparse).

def hybrid_score_norm(dense, sparse, alpha: float):
    """Scale dense/sparse query vectors for weighted hybrid search."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    scaled_sparse = {
        "indices": sparse["indices"],
        "values": [v * (1 - alpha) for v in sparse["values"]],
    }
    return [v * alpha for v in dense], scaled_sparse

hdense, hsparse = hybrid_score_norm(
    dense=[0.1, 0.2, ...],
    sparse={"indices": [10, 45], "values": [0.5, 0.3]},
    alpha=0.5,  # 0=sparse, 1=dense, 0.5=balanced
)

results = index.query(
    vector=hdense,
    sparse_vector=hsparse,
    top_k=5,
)
```

## LangChain 통합

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

## LlamaIndex 통합

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

## 인덱스 관리

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

## 벡터 삭제

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

## 모범 사례

1. **서버리스 사용** - 자동 확장, 비용 효율적
2. **업서트 일괄 처리** - 더 효율적(배치당 100~200개)
3. **메타데이터 추가** - 필터링 활성화
4. **네임스페이스 사용** - 사용자/테넌트별 데이터 격리
5. **사용량 모니터링** - Pinecone 대시보드 확인
6. **필터 최적화** - 자주 필터링하는 필드 인덱싱
7. **무료 티어로 테스트** - 인덱스 1개, 벡터 100K 무료
8. **하이브리드 검색 사용** - 더 나은 품질
9. **적절한 차원 설정** - 임베딩 모델에 맞춤
10. **정기적인 백업** - 중요한 데이터 내보내기

## 성능

| 작업 | 지연 시간 | 참고 |
|-----------|---------|-------|
| 업서트 | ~50-100ms | 배치당 |
| 쿼리(p50) | ~50ms | 인덱스 크기에 따라 다름 |
| 쿼리(p95) | ~100ms | SLA 목표 |
| 메타데이터 필터 | ~+10-20ms | 추가 오버헤드 |

## 요금(2025년 기준)

**서버리스**:
- 읽기 단위 100만 개당 $0.096
- 쓰기 단위 100만 개당 $0.06
- 스토리지 GB당 월 $0.06

**무료 티어**:
- 서버리스 인덱스 1개
- 벡터 100K개(1536차원)
- 프로토타이핑에 적합

## 리소스

- **웹사이트**: https://www.pinecone.io
- **문서**: https://docs.pinecone.io
- **콘솔**: https://app.pinecone.io
- **요금**: https://www.pinecone.io/pricing
