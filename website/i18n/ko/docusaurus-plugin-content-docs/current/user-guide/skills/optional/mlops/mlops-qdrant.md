---
title: "Qdrant — 프로덕션 RAG 시스템을 위한 벡터 검색 엔진"
sidebar_label: "Qdrant"
description: "프로덕션 RAG 시스템을 위한 벡터 검색 엔진"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Qdrant

프로덕션 RAG 및 시맨틱 검색을 위한 벡터 검색 엔진입니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/mlops/qdrant`로 설치 |
| 경로 | `optional-skills/mlops/qdrant` |
| 버전 | `1.0.1` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 종속성 | `qdrant-client>=1.14.0` |
| 플랫폼 | linux, macos, windows |
| 태그 | `RAG`, `Vector Search`, `Qdrant`, `Semantic Search`, `Embeddings`, `Similarity Search`, `HNSW`, `Production`, `Distributed` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되었을 때 에이전트가 확인하는 지침입니다.
:::

# Qdrant - 벡터 유사도 검색 엔진

프로덕션 RAG 및 시맨틱 검색을 위해 Rust로 작성된 고성능 벡터 데이터베이스입니다.

## Qdrant를 사용하는 경우

**다음과 같은 경우 Qdrant를 사용하세요:**
- 짧은 지연 시간이 필요한 프로덕션 RAG 시스템을 구축하는 경우
- 하이브리드 검색(벡터 + 메타데이터 필터링)이 필요한 경우
- 수평 확장과 샤딩/복제가 필요한 경우
- 데이터를 완전히 제어할 수 있는 온프레미스 배포를 원하는 경우
- 레코드당 여러 벡터(밀집 + 희소)를 저장해야 하는 경우
- 실시간 추천 시스템을 구축하는 경우

**주요 기능:**
- **Rust 기반**: 메모리 안전성과 높은 성능
- **풍부한 필터링**: 검색 중 모든 페이로드 필드로 필터링
- **여러 벡터**: 포인트당 밀집, 희소, 다중 밀집 벡터
- **양자화**: 메모리 효율을 위한 스칼라, 곱, 이진 양자화
- **분산**: Raft 합의, 샤딩, 복제
- **REST + gRPC**: 완전한 기능 동등성을 제공하는 두 API

**대신 다음 대안을 사용하세요:**
- **Chroma**: 더 간단한 설정과 임베디드 사용 사례
- **FAISS**: 최대 원시 속도, 연구/배치 처리
- **Pinecone**: 완전 관리형 및 운영 부담이 없는 환경을 선호하는 경우
- **Weaviate**: GraphQL을 선호하고 벡터라이저가 내장된 경우

## 빠른 시작

### 설치

```bash
# Python client
pip install qdrant-client

# Docker (recommended for development)
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Docker with persistent storage
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

### 기본 사용법

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Connect to Qdrant
client = QdrantClient(host="localhost", port=6333)

# Create collection
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# Insert vectors with payload
client.upsert(
    collection_name="documents",
    points=[
        PointStruct(
            id=1,
            vector=[0.1, 0.2, ...],  # 384-dim vector
            payload={"title": "Doc 1", "category": "tech"}
        ),
        PointStruct(
            id=2,
            vector=[0.3, 0.4, ...],
            payload={"title": "Doc 2", "category": "science"}
        )
    ]
)

# Search with filtering (query_points is the current API; client.search is removed in qdrant-client 1.14+)
response = client.query_points(
    collection_name="documents",
    query=[0.15, 0.25, ...],
    query_filter={
        "must": [{"key": "category", "match": {"value": "tech"}}]
    },
    limit=10
)

for point in response.points:
    print(f"ID: {point.id}, Score: {point.score}, Payload: {point.payload}")
```

## 핵심 개념

### 포인트 - 기본 데이터 단위

```python
from qdrant_client.models import PointStruct

# Point = ID + Vector(s) + Payload
point = PointStruct(
    id=123,                              # Integer or UUID string
    vector=[0.1, 0.2, 0.3, ...],        # Dense vector
    payload={                            # Arbitrary JSON metadata
        "title": "Document title",
        "category": "tech",
        "timestamp": 1699900000,
        "tags": ["python", "ml"]
    }
)

# Batch upsert (recommended)
client.upsert(
    collection_name="documents",
    points=[point1, point2, point3],
    wait=True  # Wait for indexing
)
```

### 컬렉션 - 벡터 컨테이너

```python
from qdrant_client.models import VectorParams, Distance, HnswConfigDiff

# Create with HNSW configuration
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(
        size=384,                        # Vector dimensions
        distance=Distance.COSINE         # COSINE, EUCLID, DOT, MANHATTAN
    ),
    hnsw_config=HnswConfigDiff(
        m=16,                            # Connections per node (default 16)
        ef_construct=100,                # Build-time accuracy (default 100)
        full_scan_threshold=10000        # Switch to brute force below this
    ),
    on_disk_payload=True                 # Store payload on disk
)

# Collection info
info = client.get_collection("documents")
print(f"Points: {info.points_count}, Vectors: {info.vectors_count}")
```

### 거리 메트릭

| 메트릭 | 사용 사례 | 범위 |
|--------|----------|-------|
| `COSINE` | 텍스트 임베딩, 정규화된 벡터 | 0 ~ 2 |
| `EUCLID` | 공간 데이터, 이미지 특징 | 0 ~ ∞ |
| `DOT` | 추천, 비정규화 데이터 | -∞ ~ ∞ |
| `MANHATTAN` | 희소 특징, 이산 데이터 | 0 ~ ∞ |

## 검색 작업

### 기본 검색

```python
# Simple nearest neighbor search (returns a QueryResponse; use .points)
response = client.query_points(
    collection_name="documents",
    query=[0.1, 0.2, ...],
    limit=10,
    with_payload=True,
    with_vectors=False  # Don't return vectors (faster)
)
results = response.points
```

### 필터링된 검색

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

# Complex filtering
response = client.query_points(
    collection_name="documents",
    query=query_embedding,
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
).points

# Shorthand filter syntax
response = client.query_points(
    collection_name="documents",
    query=query_embedding,
    query_filter={
        "must": [
            {"key": "category", "match": {"value": "tech"}},
            {"key": "price", "range": {"gte": 10, "lte": 100}}
        ]
    },
    limit=10
).points
```

### 배치 검색

```python
from qdrant_client.models import QueryRequest

# Multiple queries in one request (search_batch is replaced by query_batch_points)
responses = client.query_batch_points(
    collection_name="documents",
    requests=[
        QueryRequest(query=[0.1, ...], limit=5),
        QueryRequest(query=[0.2, ...], limit=5, filter={"must": [...]}),
        QueryRequest(query=[0.3, ...], limit=10)
    ]
)
# Each element is a QueryResponse; use .points
for resp in responses:
    for point in resp.points:
        print(point.id, point.score)
```

## RAG 통합

### sentence-transformers 사용

```python
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Initialize
encoder = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(host="localhost", port=6333)

# Create collection
client.create_collection(
    collection_name="knowledge_base",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# Index documents
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

# RAG retrieval
def retrieve(query: str, top_k: int = 5) -> list[dict]:
    query_vector = encoder.encode(query).tolist()
    response = client.query_points(
        collection_name="knowledge_base",
        query=query_vector,
        limit=top_k
    )
    return [{"text": r.payload["text"], "score": r.score} for r in response.points]

# Use in RAG pipeline
context = retrieve("What is Python?")
prompt = f"Context: {context}\n\nQuestion: What is Python?"
```

### LangChain 사용

```python
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Qdrant.from_documents(documents, embeddings, url="http://localhost:6333", collection_name="docs")
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

### LlamaIndex 사용

```python
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

vector_store = QdrantVectorStore(client=client, collection_name="llama_docs")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
query_engine = index.as_query_engine()
```

## 다중 벡터 지원

### 이름이 지정된 벡터(서로 다른 임베딩 모델)

```python
from qdrant_client.models import VectorParams, Distance

# Collection with multiple vector types
client.create_collection(
    collection_name="hybrid_search",
    vectors_config={
        "dense": VectorParams(size=384, distance=Distance.COSINE),
        "sparse": VectorParams(size=30000, distance=Distance.DOT)
    }
)

# Insert with named vectors
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

# Search specific named vector (pass the vector name via `using`)
response = client.query_points(
    collection_name="hybrid_search",
    query=query_dense,
    using="dense",  # Specify which named vector to search
    limit=10
)
results = response.points
```

### 희소 벡터(BM25, SPLADE)

```python
from qdrant_client.models import SparseVectorParams, SparseIndexParams, SparseVector

# Collection with sparse vectors
client.create_collection(
    collection_name="sparse_search",
    vectors_config={},
    sparse_vectors_config={"text": SparseVectorParams(index=SparseIndexParams(on_disk=False))}
)

# Insert sparse vector
client.upsert(
    collection_name="sparse_search",
    points=[PointStruct(id=1, vector={"text": SparseVector(indices=[1, 5, 100], values=[0.5, 0.8, 0.2])}, payload={"text": "document"})]
)
```

## 양자화(메모리 최적화)

```python
from qdrant_client.models import ScalarQuantization, ScalarQuantizationConfig, ScalarType

# Scalar quantization (4x memory reduction)
client.create_collection(
    collection_name="quantized",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    quantization_config=ScalarQuantization(
        scalar=ScalarQuantizationConfig(
            type=ScalarType.INT8,
            quantile=0.99,        # Clip outliers
            always_ram=True      # Keep quantized in RAM
        )
    )
)

# Search with rescoring
response = client.query_points(
    collection_name="quantized",
    query=query,
    search_params={"quantization": {"rescore": True}},  # Rescore top results
    limit=10
)
results = response.points
```

## 페이로드 인덱싱

```python
from qdrant_client.models import PayloadSchemaType

# Create payload index for faster filtering
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

# Index types: KEYWORD, INTEGER, FLOAT, GEO, TEXT (full-text), BOOL
```

## 프로덕션 배포

### Qdrant Cloud

```python
from qdrant_client import QdrantClient

# Connect to Qdrant Cloud
client = QdrantClient(
    url="https://your-cluster.cloud.qdrant.io",
    api_key="your-api-key"
)
```

### 성능 조정

```python
# Optimize for search speed (higher recall)
client.update_collection(
    collection_name="documents",
    hnsw_config=HnswConfigDiff(ef_construct=200, m=32)
)

# Optimize for indexing speed (bulk loads)
client.update_collection(
    collection_name="documents",
    optimizer_config={"indexing_threshold": 20000}
)
```

## 모범 사례

1. **배치 작업** - 효율성을 위해 배치 업서트/검색을 사용하세요
2. **페이로드 인덱싱** - 필터에 사용되는 필드를 인덱싱하세요
3. **양자화** - 대규모 컬렉션(벡터 100만 개 초과)에 활성화하세요
4. **샤딩** - 벡터 1,000만 개를 초과하는 컬렉션에 사용하세요
5. **디스크 저장소** - 대규모 페이로드에는 `on_disk_payload`를 활성화하세요
6. **연결 풀링** - 클라이언트 인스턴스를 재사용하세요

## 일반적인 문제

**필터를 사용한 검색이 느린 경우:**
```python
# Create payload index for filtered fields
client.create_payload_index(
    collection_name="docs",
    field_name="category",
    field_schema=PayloadSchemaType.KEYWORD
)
```

**메모리 부족:**
```python
# Enable quantization and on-disk storage
client.create_collection(
    collection_name="large_collection",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    quantization_config=ScalarQuantization(...),
    on_disk_payload=True
)
```

**연결 문제:**
```python
# Use timeout and retry
client = QdrantClient(
    host="localhost",
    port=6333,
    timeout=30,
    prefer_grpc=True  # gRPC for better performance
)
```

## 참조

- **[고급 사용법](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/qdrant/references/advanced-usage.md)** - 분산 모드, 하이브리드 검색, 추천
- **[문제 해결](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/qdrant/references/troubleshooting.md)** - 일반적인 문제, 디버깅, 성능 조정

## 리소스

- **GitHub**: https://github.com/qdrant/qdrant (22k+ stars)
- **문서**: https://qdrant.tech/documentation/
- **Python 클라이언트**: https://github.com/qdrant/qdrant-client
- **Cloud**: https://cloud.qdrant.io
