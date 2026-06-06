---
title: "Chroma — AI 애플리케이션을 위한 오픈소스 임베딩 데이터베이스"
sidebar_label: "Chroma"
description: "AI 애플리케이션을 위한 오픈소스 임베딩 데이터베이스"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py를 통해 스킬의 SKILL.md에서 자동으로 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 수정하세요. */}

# Chroma

AI 애플리케이션을 위한 오픈소스 임베딩 데이터베이스입니다. 임베딩과 메타데이터를 저장하고, 벡터 및 전체 텍스트 검색을 수행하며 메타데이터로 필터링할 수 있습니다. 4개의 함수로 구성된 간단한 API를 제공합니다. 노트북 환경부터 프로덕션 클러스터까지 확장이 가능합니다. 의미 기반 검색(Semantic search), RAG 애플리케이션, 또는 문서 검색에 사용하세요. 로컬 개발 및 오픈소스 프로젝트에 가장 적합합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택사항 — `hermes skills install official/mlops/chroma`로 설치 |
| 경로 | `optional-skills/mlops/chroma` |
| 버전 | `1.0.0` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 의존성 | `chromadb`, `sentence-transformers` |
| 플랫폼 | linux, macos, windows |
| 태그 | `RAG`, `Chroma`, `Vector Database`, `Embeddings`, `Semantic Search`, `Open Source`, `Self-Hosted`, `Document Retrieval`, `Metadata Filtering` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이는 스킬이 활성화되었을 때 에이전트가 지시사항으로 보는 내용입니다.
:::

# Chroma - 오픈소스 임베딩 데이터베이스

메모리를 갖춘 LLM 애플리케이션을 구축하기 위한 AI 네이티브 데이터베이스입니다.

## Chroma를 사용하는 경우

**다음의 경우에 Chroma를 사용하세요:**
- RAG (검색 증강 생성) 애플리케이션을 구축할 때
- 로컬/자체 호스팅되는 벡터 데이터베이스가 필요할 때
- 오픈소스 솔루션(Apache 2.0)을 원할 때
- 노트북 환경에서 프로토타이핑을 할 때
- 문서에 대한 의미 기반 검색(Semantic search)을 할 때
- 메타데이터와 함께 임베딩을 저장할 때

**지표(Metrics)**:
- **24,300+ GitHub stars**
- **1,900+ forks**
- **v1.3.3** (안정판, 매주 릴리즈)
- **Apache 2.0 라이선스**

**다음을 대신 사용해 보세요**:
- **Pinecone**: 관리형 클라우드 서비스, 자동 스케일링
- **FAISS**: 메타데이터 없는 순수 유사도 검색
- **Weaviate**: 프로덕션 ML 네이티브 데이터베이스
- **Qdrant**: 고성능, Rust 기반

## 빠른 시작

### 설치

```bash
# Python
pip install chromadb

# JavaScript/TypeScript
npm install chromadb @chroma-core/default-embed
```

### 기본 사용법 (Python)

```python
import chromadb

# 클라이언트 생성
client = chromadb.Client()

# 컬렉션 생성
collection = client.create_collection(name="my_collection")

# 문서 추가
collection.add(
    documents=["This is document 1", "This is document 2"],
    metadatas=[{"source": "doc1"}, {"source": "doc2"}],
    ids=["id1", "id2"]
)

# 쿼리 수행
results = collection.query(
    query_texts=["document about topic"],
    n_results=2
)

print(results)
```

## 핵심 기능

### 1. 컬렉션 생성

```python
# 기본 컬렉션
collection = client.create_collection("my_docs")

# 사용자 지정 임베딩 함수와 함께 생성
from chromadb.utils import embedding_functions

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="your-key",
    model_name="text-embedding-3-small"
)

collection = client.create_collection(
    name="my_docs",
    embedding_function=openai_ef
)

# 기존 컬렉션 가져오기
collection = client.get_collection("my_docs")

# 컬렉션 삭제
client.delete_collection("my_docs")
```

### 2. 문서 추가

```python
# 자동으로 생성된 ID와 함께 추가
collection.add(
    documents=["Doc 1", "Doc 2", "Doc 3"],
    metadatas=[
        {"source": "web", "category": "tutorial"},
        {"source": "pdf", "page": 5},
        {"source": "api", "timestamp": "2025-01-01"}
    ],
    ids=["id1", "id2", "id3"]
)

# 사용자 지정 임베딩과 함께 추가
collection.add(
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
    documents=["Doc 1", "Doc 2"],
    ids=["id1", "id2"]
)
```

### 3. 쿼리 (유사도 검색)

```python
# 기본 쿼리
results = collection.query(
    query_texts=["machine learning tutorial"],
    n_results=5
)

# 필터를 사용한 쿼리
results = collection.query(
    query_texts=["Python programming"],
    n_results=3,
    where={"source": "web"}
)

# 메타데이터 필터를 사용한 쿼리
results = collection.query(
    query_texts=["advanced topics"],
    where={
        "$and": [
            {"category": "tutorial"},
            {"difficulty": {"$gte": 3}}
        ]
    }
)

# 결과 접근
print(results["documents"])      # 일치하는 문서들의 리스트
print(results["metadatas"])      # 각 문서의 메타데이터
print(results["distances"])      # 유사도 점수
print(results["ids"])            # 문서의 ID들
```

### 4. 문서 가져오기

```python
# ID로 가져오기
docs = collection.get(
    ids=["id1", "id2"]
)

# 필터로 가져오기
docs = collection.get(
    where={"category": "tutorial"},
    limit=10
)

# 모든 문서 가져오기
docs = collection.get()
```

### 5. 문서 업데이트

```python
# 문서 내용 업데이트
collection.update(
    ids=["id1"],
    documents=["Updated content"],
    metadatas=[{"source": "updated"}]
)
```

### 6. 문서 삭제

```python
# ID로 삭제
collection.delete(ids=["id1", "id2"])

# 필터로 삭제
collection.delete(
    where={"source": "outdated"}
)
```

## 영구 스토리지 (Persistent storage)

```python
# 디스크에 저장
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.create_collection("my_docs")
collection.add(documents=["Doc 1"], ids=["id1"])

# 데이터는 자동으로 저장됨
# 나중에 같은 경로로 다시 로드
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("my_docs")
```

## 임베딩 함수

### 기본값 (Sentence Transformers)

```python
# 기본적으로 sentence-transformers를 사용합니다
collection = client.create_collection("my_docs")
# 기본 모델: all-MiniLM-L6-v2
```

### OpenAI

```python
from chromadb.utils import embedding_functions

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="your-key",
    model_name="text-embedding-3-small"
)

collection = client.create_collection(
    name="openai_docs",
    embedding_function=openai_ef
)
```

### HuggingFace

```python
huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key="your-key",
    model_name="sentence-transformers/all-mpnet-base-v2"
)

collection = client.create_collection(
    name="hf_docs",
    embedding_function=huggingface_ef
)
```

### 사용자 지정 임베딩 함수

```python
from chromadb import Documents, EmbeddingFunction, Embeddings

class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # 임베딩 생성 로직 구현
        return embeddings

my_ef = MyEmbeddingFunction()
collection = client.create_collection(
    name="custom_docs",
    embedding_function=my_ef
)
```

## 메타데이터 필터링

```python
# 정확히 일치 (Exact match)
results = collection.query(
    query_texts=["query"],
    where={"category": "tutorial"}
)

# 비교 연산자
results = collection.query(
    query_texts=["query"],
    where={"page": {"$gt": 10}}  # $gt, $gte, $lt, $lte, $ne
)

# 논리 연산자
results = collection.query(
    query_texts=["query"],
    where={
        "$and": [
            {"category": "tutorial"},
            {"difficulty": {"$lte": 3}}
        ]
    }  # 다른 예시: $or
)

# 포함 여부 (Contains)
results = collection.query(
    query_texts=["query"],
    where={"tags": {"$in": ["python", "ml"]}}
)
```

## LangChain 통합

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 문서 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(documents)

# Chroma vector store 생성
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_db"
)

# 쿼리
results = vectorstore.similarity_search("machine learning", k=3)

# retriever로 사용
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

## LlamaIndex 통합

```python
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
import chromadb

# Chroma 초기화
db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_or_create_collection("my_collection")

# vector store 생성
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 인덱스 생성
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

# 쿼리
query_engine = index.as_query_engine()
response = query_engine.query("What is machine learning?")
```

## 서버 모드

```python
# Chroma 서버 실행
# 터미널에서: chroma run --path ./chroma_db --port 8000

# 서버에 연결
import chromadb
from chromadb.config import Settings

client = chromadb.HttpClient(
    host="localhost",
    port=8000,
    settings=Settings(anonymized_telemetry=False)
)

# 정상적으로 사용
collection = client.get_or_create_collection("my_docs")
```

## 모범 사례

1. **PersistentClient 사용** - 재시작 시 데이터가 손실되지 않도록 합니다.
2. **메타데이터 추가** - 필터링과 추적을 가능하게 합니다.
3. **일괄 작업(Batch operations)** - 여러 문서를 한 번에 추가합니다.
4. **적절한 임베딩 모델 선택** - 속도와 품질 간의 균형을 유지합니다.
5. **필터 사용** - 검색 범위를 좁힙니다.
6. **고유 ID 사용** - 충돌을 방지합니다.
7. **정기적인 백업** - chroma_db 디렉토리를 복사해 둡니다.
8. **컬렉션 크기 모니터링** - 필요한 경우 스케일 업(scale up)합니다.
9. **임베딩 함수 테스트** - 품질을 보장합니다.
10. **프로덕션 환경에서는 서버 모드 사용** - 다중 사용자 환경에 더 적합합니다.

## 성능

| 작업 | 레이턴시 | 참고 |
|-----------|---------|-------|
| 100 문서 추가 | ~1-3s | 임베딩 포함 시 |
| 쿼리 (상위 10개) | ~50-200ms | 컬렉션 크기에 따라 다름 |
| 메타데이터 필터 | ~10-50ms | 적절한 인덱싱 시 빠름 |

## 리소스

- **GitHub**: https://github.com/chroma-core/chroma ⭐ 24,300+
- **문서**: https://docs.trychroma.com
- **Discord**: https://discord.gg/MMeYNTmh3x
- **버전**: 1.3.3+
- **라이선스**: Apache 2.0
