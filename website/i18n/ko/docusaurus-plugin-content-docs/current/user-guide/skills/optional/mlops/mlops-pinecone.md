---
title: "Pinecone — 프로덕션 AI 애플리케이션을 위한 관리형 벡터 데이터베이스"
sidebar_label: "Pinecone"
description: "프로덕션 AI 애플리케이션을 위한 관리형 벡터 데이터베이스"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Pinecone

프로덕션 AI 애플리케이션을 위한 관리형 벡터 데이터베이스입니다. 완전 관리형, 자동 확장성을 지원하며 하이브리드 검색(밀집 + 희소), 메타데이터 필터링 및 네임스페이스를 제공합니다. 짧은 지연 시간(p95 &lt;100ms)을 자랑합니다. 대규모 프로덕션 RAG, 추천 시스템 또는 시맨틱 검색에 사용하세요. 서버리스 및 관리형 인프라에 가장 적합합니다.

## Skill metadata

| | |
|---|---|
| Source | Optional — `hermes skills install official/mlops/pinecone`로 설치 |
| Path | `optional-skills/mlops/pinecone` |
| Version | `1.0.0` |
| Author | Orchestra Research |
| License | MIT |
| Dependencies | `pinecone-client` |
| Platforms | linux, macos, windows |
| Tags | `RAG`, `Pinecone`, `Vector Database`, `Managed Service`, `Serverless`, `Hybrid Search`, `Production`, `Auto-Scaling`, `Low Latency`, `Recommendations` |

## Reference: full SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# Pinecone - 관리형 벡터 데이터베이스

프로덕션 AI 애플리케이션을 위한 벡터 데이터베이스입니다.

## Pinecone을 사용해야 하는 경우

**사용 시기:**
- 관리형 서버리스 벡터 데이터베이스가 필요할 때
- 프로덕션 RAG 애플리케이션
- 자동 확장이 필요할 때
- 짧은 지연 시간이 중요할 때 (&lt;100ms)
- 인프라를 직접 관리하고 싶지 않을 때
- 하이브리드 검색(밀집 + 희소 벡터)이 필요할 때

**지표**:
- 완전 관리형 SaaS
- 수십억 개의 벡터로 자동 확장
- **p95 지연 시간 &lt;100ms**
- 99.9% 가동 시간 SLA

**대신 사용할 수 있는 대안**:
- **Chroma**: 자체 호스팅, 오픈 소스
- **FAISS**: 오프라인, 순수 유사도 검색
- **Weaviate**: 더 많은 기능을 갖춘 자체 호스팅

## 빠른 시작

### 설치

```bash
pip install pinecone-client
```

### 기본 사용법

```python
from pinecone import Pinecone, ServerlessSpec

# 초기화
pc = Pinecone(api_key="your-api-key")

# 인덱스 생성
pc.create_index(
    name="my-index",
    dimension=1536,  # 임베딩 차원과 일치해야 함
    metric="cosine",  # 또는 "euclidean", "dotproduct"
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

# 인덱스 연결
index = pc.Index("my-index")

# 벡터 업서트(Upsert)
index.upsert(vectors=[
    {"id": "vec1", "values": [0.1, 0.2, ...], "metadata": {"category": "A"}},
    {"id": "vec2", "values": [0.3, 0.4, ...], "metadata": {"category": "B"}}
])

# 쿼리
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
# 서버리스 (권장)
pc.create_index(
    name="my-index",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",         # 또는 "gcp", "azure"
        region="us-east-1"
    )
)

# Pod 기반 (일관된 성능을 위해)
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
# 단일 업서트
index.upsert(vectors=[
    {
        "id": "doc1",
        "values": [0.1, 0.2, ...],  # 1536 차원
        "metadata": {
            "text": "문서 내용",
            "category": "tutorial",
            "timestamp": "2025-01-01"
        }
    }
])

# 일괄 업서트 (권장)
vectors = [
    {"id": f"vec{i}", "values": embedding, "metadata": metadata}
    for i, (embedding, metadata) in enumerate(zip(embeddings, metadatas))
]

index.upsert(vectors=vectors, batch_size=100)
```

### 벡터 쿼리

```python
# 기본 쿼리
results = index.query(
    vector=[0.1, 0.2, ...],
    top_k=10,
    include_metadata=True,
    include_values=False
)

# 메타데이터 필터링 포함
results = index.query(
    vector=[0.1, 0.2, ...],
    top_k=5,
    filter={"category": {"$eq": "tutorial"}}
)

# 네임스페이스 쿼리
results = index.query(
    vector=[0.1, 0.2, ...],
    top_k=5,
    namespace="production"
)

# 결과 접근
for match in results["matches"]:
    print(f"ID: {match['id']}")
    print(f"점수: {match['score']}")
    print(f"메타데이터: {match['metadata']}")
```

### 메타데이터 필터링

```python
# 정확히 일치
filter = {"category": "tutorial"}

# 비교
filter = {"price": {"$gte": 100}}  # $gt, $gte, $lt, $lte, $ne

# 논리 연산자
filter = {
    "$and": [
        {"category": "tutorial"},
        {"difficulty": {"$lte": 3}}
    ]
}  # 또한: $or

# In 연산자
filter = {"tags": {"$in": ["python", "ml"]}}
```

## 네임스페이스

```python
# 네임스페이스별 데이터 분할
index.upsert(
    vectors=[{"id": "vec1", "values": [...]}],
    namespace="user-123"
)

# 특정 네임스페이스 쿼리
results = index.query(
    vector=[...],
    namespace="user-123",
    top_k=5
)

# 네임스페이스 목록 조회
stats = index.describe_index_stats()
print(stats['namespaces'])
```

## 하이브리드 검색 (밀집 + 희소)

```python
# 희소 벡터를 포함하여 업서트
index.upsert(vectors=[
    {
        "id": "doc1",
        "values": [0.1, 0.2, ...],  # 밀집 벡터
        "sparse_values": {
            "indices": [10, 45, 123],  # 토큰 ID
            "values": [0.5, 0.3, 0.8]   # TF-IDF 점수
        },
        "metadata": {"text": "..."}
    }
])

# 하이브리드 쿼리
results = index.query(
    vector=[0.1, 0.2, ...],
    sparse_vector={
        "indices": [10, 45],
        "values": [0.5, 0.3]
    },
    top_k=5,
    alpha=0.5  # 0=희소, 1=밀집, 0.5=하이브리드
)
```

## LangChain 통합

```python
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# 벡터 저장소 생성
vectorstore = PineconeVectorStore.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(),
    index_name="my-index"
)

# 쿼리
results = vectorstore.similarity_search("query", k=5)

# 메타데이터 필터 포함
results = vectorstore.similarity_search(
    "query",
    k=5,
    filter={"category": "tutorial"}
)

# 리트리버로 사용
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
```

## LlamaIndex 통합

```python
from llama_index.vector_stores.pinecone import PineconeVectorStore

# Pinecone 연결
pc = Pinecone(api_key="your-key")
pinecone_index = pc.Index("my-index")

# 벡터 저장소 생성
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# LlamaIndex에서 사용
from llama_index.core import StorageContext, VectorStoreIndex

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
```

## 인덱스 관리

```python
# 인덱스 목록 조회
indexes = pc.list_indexes()

# 인덱스 정보 설명
index_info = pc.describe_index("my-index")
print(index_info)

# 인덱스 통계 가져오기
stats = index.describe_index_stats()
print(f"총 벡터 수: {stats['total_vector_count']}")
print(f"네임스페이스: {stats['namespaces']}")

# 인덱스 삭제
pc.delete_index("my-index")
```

## 벡터 삭제

```python
# ID로 삭제
index.delete(ids=["vec1", "vec2"])

# 필터로 삭제
index.delete(filter={"category": "old"})

# 네임스페이스 내 모든 항목 삭제
index.delete(delete_all=True, namespace="test")

# 전체 인덱스 삭제
index.delete(delete_all=True)
```

## 모범 사례

1. **서버리스 사용** - 자동 확장, 비용 효율성
2. **일괄 업서트** - 더 효율적 (배치당 100-200개)
3. **메타데이터 추가** - 필터링 활성화
4. **네임스페이스 사용** - 사용자/테넌트별 데이터 격리
5. **사용량 모니터링** - Pinecone 대시보드 확인
6. **필터 최적화** - 자주 필터링되는 필드 인덱싱
7. **무료 티어로 테스트** - 인덱스 1개, 벡터 10만 개 무료
8. **하이브리드 검색 사용** - 더 나은 품질
9. **적절한 차원 설정** - 임베딩 모델과 일치
10. **정기적인 백업** - 중요한 데이터 내보내기

## 성능

| 작업 | 지연 시간 | 참고 |
|-----------|---------|-------|
| 업서트 | ~50-100ms | 배치당 |
| 쿼리 (p50) | ~50ms | 인덱스 크기에 따라 다름 |
| 쿼리 (p95) | ~100ms | SLA 목표 |
| 메타데이터 필터 | ~+10-20ms | 추가 오버헤드 |

## 가격 (2025년 기준)

**서버리스**:
- 백만 읽기 단위당 $0.096
- 백만 쓰기 단위당 $0.06
- GB 스토리지/월당 $0.06

**무료 티어**:
- 서버리스 인덱스 1개
- 벡터 10만 개 (1536 차원)
- 프로토타이핑에 적합

## 리소스

- **웹사이트**: https://www.pinecone.io
- **문서**: https://docs.pinecone.io
- **콘솔**: https://app.pinecone.io
- **가격**: https://www.pinecone.io/pricing
