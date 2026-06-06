---
title: "Qdrant Vector Search — RAG 및 시맨틱 검색을 위한 고성능 벡터 유사도 검색 엔진"
sidebar_label: "Qdrant Vector Search"
description: "RAG 및 시맨틱 검색을 위한 고성능 벡터 유사도 검색 엔진"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Qdrant Vector Search

RAG 및 시맨틱 검색을 위한 고성능 벡터 유사도 검색 엔진. 빠른 최근접 이웃 검색, 필터링이 포함된 하이브리드 검색 또는 Rust 기반 성능을 갖춘 확장 가능한 벡터 스토리지가 필요한 프로덕션 RAG 시스템을 구축할 때 사용합니다.

## 스킬 메타데이터

| | |
|---|---|
| Source | Optional — `hermes skills install official/mlops/qdrant`로 설치 |
| Path | `optional-skills/mlops/qdrant` |
| Version | `1.0.0` |
| Author | Orchestra Research |
| License | MIT |
| Dependencies | `qdrant-client>=1.12.0` |
| Platforms | linux, macos, windows |
| Tags | `RAG`, `Vector Search`, `Qdrant`, `Semantic Search`, `Embeddings`, `Similarity Search`, `HNSW`, `Production`, `Distributed` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되어 있을 때 에이전트가 지침으로 보는 내용입니다.
:::

# Qdrant - 벡터 유사도 검색 엔진

프로덕션 RAG 및 시맨틱 검색을 위해 Rust로 작성된 고성능 벡터 데이터베이스.

## Qdrant를 사용해야 할 때

**다음과 같은 경우 Qdrant를 사용하세요:**
- 낮은 지연 시간을 요구하는 프로덕션 RAG 시스템을 구축할 때
- 하이브리드 검색(벡터 + 메타데이터 필터링)이 필요할 때
- 샤딩/복제를 통한 수평적 확장이 요구될 때
- 완벽한 데이터 제어가 가능한 온프레미스 배포를 원할 때
- 레코드당 다중 벡터 스토리지(dense + sparse)가 필요할 때
- 실시간 추천 시스템을 구축할 때

**주요 특징:**
- **Rust 기반**: 메모리 안전성, 고성능
- **풍부한 필터링**: 검색 중 페이로드 필드로 필터링
- **다중 벡터**: 포인트당 Dense, sparse, multi-dense 지원
- **양자화**: 메모리 효율성을 위한 스칼라, 프로덕트, 바이너리 양자화
- **분산형**: Raft 합의, 샤딩, 복제
- **REST + gRPC**: 완벽한 기능 패리티를 갖춘 두 가지 API

**대신 다른 대안을 사용해야 할 때:**
- **Chroma**: 더 간단한 설정, 임베디드 사용 사례
- **FAISS**: 최고의 원시 속도, 연구/배치 처리
- **Pinecone**: 완전 관리형, 제로 운영 선호
- **Weaviate**: GraphQL 선호, 내장 벡터라이저

## 빠른 시작

### 설치

```bash
# 파이썬 클라이언트
pip install qdrant-client

# Docker (개발용으로 권장)
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# 영구 스토리지가 있는 Docker
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

### 기본 사용법

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Qdrant에 연결
client = QdrantClient(host="localhost", port=6333)

# 컬렉션 생성
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# 페이로드와 함께 벡터 삽입
client.upsert(
    collection_name="documents",
    points=[
        PointStruct(
            id=1,
            vector=[0.1, 0.2, ...],  # 384차원 벡터
            payload={"title": "Doc 1", "category": "tech"}
        ),
        PointStruct(
            id=2,
            vector=[0.3, 0.4, ...],
            payload={"title": "Doc 2", "category": "science"}
        )
    ]
)

# 필터링 검색
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

## 핵심 개념

### 포인트 (Points) - 기본 데이터 단위

```python
from qdrant_client.models import PointStruct

# Point = ID + Vector(s) + Payload
point = PointStruct(
    id=123,                              # 정수 또는 UUID 문자열
    vector=[0.1, 0.2, 0.3, ...],        # Dense 벡터
    payload={                            # 임의의 JSON 메타데이터
        "title": "Document title",
        "category": "tech",
        "timestamp": 1699900000,
        "tags": ["python", "ml"]
    }
)

# 일괄 삽입 (권장)
client.upsert(
    collection_name="documents",
    points=[point1, point2, point3],
    wait=True  # 인덱싱 대기
)
```

### 컬렉션 (Collections) - 벡터 컨테이너

```python
from qdrant_client.models import VectorParams, Distance, HnswConfigDiff

# HNSW 구성으로 생성
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(
        size=384,                        # 벡터 차원
        distance=Distance.COSINE         # COSINE, EUCLID, DOT, MANHATTAN
    ),
    hnsw_config=HnswConfigDiff(
        m=16,                            # 노드당 연결 수 (기본값 16)
        ef_construct=100,                # 빌드 시간 정확도 (기본값 100)
        full_scan_threshold=10000        # 이 값 이하에서는 브루트 포스로 전환
    ),
    on_disk_payload=True                 # 디스크에 페이로드 저장
)

# 컬렉션 정보
info = client.get_collection("documents")
print(f"Points: {info.points_count}, Vectors: {info.vectors_count}")
```

### 거리 지표 (Distance metrics)

| 지표 | 사용 사례 | 범위 |
|--------|----------|-------|
| `COSINE` | 텍스트 임베딩, 정규화된 벡터 | 0 to 2 |
| `EUCLID` | 공간 데이터, 이미지 기능 | 0 to ∞ |
| `DOT` | 추천, 비정규화 | -∞ to ∞ |
| `MANHATTAN` | 희소 기능, 이산 데이터 | 0 to ∞ |

## 검색 작업

### 기본 검색

```python
# 단순 최근접 이웃 검색
results = client.search(
    collection_name="documents",
    query_vector=[0.1, 0.2, ...],
    limit=10,
    with_payload=True,
    with_vectors=False  # 벡터를 반환하지 않음 (더 빠름)
)
```

### 필터링된 검색

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

# 복합 필터링
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

# 단축 필터 구문
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

### 일괄 검색

```python
from qdrant_client.models import SearchRequest

# 단일 요청에 여러 쿼리
results = client.search_batch(
    collection_name="documents",
    requests=[
        SearchRequest(vector=[0.1, ...], limit=5),
        SearchRequest(vector=[0.2, ...], limit=5, filter={"must": [...]}),
        SearchRequest(vector=[0.3, ...], limit=10)
    ]
)
```

## RAG 통합

### sentence-transformers 사용

```python
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# 초기화
encoder = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(host="localhost", port=6333)

# 컬렉션 생성
client.create_collection(
    collection_name="knowledge_base",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# 문서 인덱싱
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

# RAG 검색
def retrieve(query: str, top_k: int = 5) -> list[dict]:
    query_vector = encoder.encode(query).tolist()
    results = client.search(
        collection_name="knowledge_base",
        query_vector=query_vector,
        limit=top_k
    )
    return [{"text": r.payload["text"], "score": r.score} for r in results]

# RAG 파이프라인에서 사용
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

### 명명된 벡터 (Named vectors, 다른 임베딩 모델)

```python
from qdrant_client.models import VectorParams, Distance

# 다중 벡터 유형이 있는 컬렉션
client.create_collection(
    collection_name="hybrid_search",
    vectors_config={
        "dense": VectorParams(size=384, distance=Distance.COSINE),
        "sparse": VectorParams(size=30000, distance=Distance.DOT)
    }
)

# 명명된 벡터로 삽입
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

# 특정 벡터 검색
results = client.search(
    collection_name="hybrid_search",
    query_vector=("dense", query_dense),  # 어떤 벡터인지 지정
    limit=10
)
```

### 희소 벡터 (Sparse vectors, BM25, SPLADE)

```python
from qdrant_client.models import SparseVectorParams, SparseIndexParams, SparseVector

# 희소 벡터가 있는 컬렉션
client.create_collection(
    collection_name="sparse_search",
    vectors_config={},
    sparse_vectors_config={"text": SparseVectorParams(index=SparseIndexParams(on_disk=False))}
)

# 희소 벡터 삽입
client.upsert(
    collection_name="sparse_search",
    points=[PointStruct(id=1, vector={"text": SparseVector(indices=[1, 5, 100], values=[0.5, 0.8, 0.2])}, payload={"text": "document"})]
)
```

## 양자화 (메모리 최적화)

```python
from qdrant_client.models import ScalarQuantization, ScalarQuantizationConfig, ScalarType

# 스칼라 양자화 (메모리 4배 감소)
client.create_collection(
    collection_name="quantized",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    quantization_config=ScalarQuantization(
        scalar=ScalarQuantizationConfig(
            type=ScalarType.INT8,
            quantile=0.99,        # 이상치 클리핑
            always_ram=True      # RAM에 양자화 유지
        )
    )
)

# 리스코어링 검색
results = client.search(
    collection_name="quantized",
    query_vector=query,
    search_params={"quantization": {"rescore": True}},  # 상위 결과 리스코어링
    limit=10
)
```

## 페이로드 인덱싱

```python
from qdrant_client.models import PayloadSchemaType

# 더 빠른 필터링을 위한 페이로드 인덱스 생성
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

# 인덱스 유형: KEYWORD, INTEGER, FLOAT, GEO, TEXT (full-text), BOOL
```

## 프로덕션 배포

### Qdrant Cloud

```python
from qdrant_client import QdrantClient

# Qdrant Cloud에 연결
client = QdrantClient(
    url="https://your-cluster.cloud.qdrant.io",
    api_key="your-api-key"
)
```

### 성능 조정

```python
# 검색 속도에 최적화 (높은 재현율)
client.update_collection(
    collection_name="documents",
    hnsw_config=HnswConfigDiff(ef_construct=200, m=32)
)

# 인덱싱 속도에 최적화 (대량 로드)
client.update_collection(
    collection_name="documents",
    optimizer_config={"indexing_threshold": 20000}
)
```

## 모범 사례

1. **일괄 작업** - 효율성을 위해 일괄 upsert/search 사용
2. **페이로드 인덱싱** - 필터에 사용되는 필드 인덱싱
3. **양자화** - 대규모 컬렉션(>1M 벡터)에 대해 활성화
4. **샤딩** - >10M 벡터 컬렉션에 사용
5. **온디스크 스토리지** - 큰 페이로드에 대해 `on_disk_payload` 활성화
6. **커넥션 풀링** - 클라이언트 인스턴스 재사용

## 일반적인 문제

**필터를 사용한 느린 검색:**
```python
# 필터링된 필드에 대한 페이로드 인덱스 생성
client.create_payload_index(
    collection_name="docs",
    field_name="category",
    field_schema=PayloadSchemaType.KEYWORD
)
```

**메모리 부족:**
```python
# 양자화 및 온디스크 스토리지 활성화
client.create_collection(
    collection_name="large_collection",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    quantization_config=ScalarQuantization(...),
    on_disk_payload=True
)
```

**연결 문제:**
```python
# 시간 초과 및 재시도 사용
client = QdrantClient(
    host="localhost",
    port=6333,
    timeout=30,
    prefer_grpc=True  # 더 나은 성능을 위해 gRPC 사용
)
```

## 참조

- **[고급 사용법](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/qdrant/references/advanced-usage.md)** - 분산 모드, 하이브리드 검색, 추천
- **[문제 해결](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/qdrant/references/troubleshooting.md)** - 일반적인 문제, 디버깅, 성능 조정

## 리소스

- **GitHub**: https://github.com/qdrant/qdrant (22k+ stars)
- **문서**: https://qdrant.tech/documentation/
- **파이썬 클라이언트**: https://github.com/qdrant/qdrant-client
- **클라우드**: https://cloud.qdrant.io
- **버전**: 1.12.0+
- **라이선스**: Apache 2.0
