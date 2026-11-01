---
title: "Chroma — RAG 및 의미 검색을 위한 임베딩 데이터베이스"
sidebar_label: "Chroma"
description: "RAG 및 의미 검색을 위한 임베딩 데이터베이스"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동으로 생성합니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Chroma

RAG 및 의미 검색을 위한 임베딩 데이터베이스.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/mlops/chroma`로 설치 |
| 경로 | `optional-skills/mlops/chroma` |
| 버전 | `1.0.0` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 종속성 | `chromadb`, `sentence-transformers` |
| 플랫폼 | linux, macos, windows |
| 태그 | `RAG`, `Chroma`, `Vector Database`, `Embeddings`, `Semantic Search`, `Open Source`, `Self-Hosted`, `Document Retrieval`, `Metadata Filtering` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 불러오는 완전한 스킬 정의입니다. 스킬이 활성 상태일 때 에이전트가 보는 내용도 이것입니다.
:::

# Chroma - 오픈 소스 임베딩 데이터베이스

메모리를 사용해 LLM 애플리케이션을 구축하기 위한 AI 네이티브 데이터베이스입니다.

## Chroma를 사용하는 경우

**다음과 같은 경우 Chroma를 사용하세요.**
- RAG(검색 증강 생성) 애플리케이션을 구축할 때
- 로컬/자체 호스팅 벡터 데이터베이스가 필요할 때
- 오픈 소스 솔루션(Apache 2.0)을 원할 때
- 노트북에서 프로토타이핑할 때
- 문서에서 의미 검색을 수행할 때
- 메타데이터와 함께 임베딩을 저장할 때

**지표**:
- **GitHub 스타 24,300개 이상**
- **포크 1,900개 이상**
- **v1.3.3**(안정 버전, 매주 릴리스)
- **Apache 2.0 라이선스**

**대신 다음 대안을 사용하세요.**
- **Pinecone**: 관리형 클라우드, 자동 확장
- **FAISS**: 순수 유사도 검색, 메타데이터 없음
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

# Create client
client = chromadb.Client()

# Create collection
collection = client.create_collection(name="my_collection")

# Add documents
collection.add(
    documents=["This is document 1", "This is document 2"],
    metadatas=[{"source": "doc1"}, {"source": "doc2"}],
    ids=["id1", "id2"]
)

# Query
results = collection.query(
    query_texts=["document about topic"],
    n_results=2
)

print(results)
```

## 핵심 작업

### 1. 컬렉션 생성

```python
# Simple collection
collection = client.create_collection("my_docs")

# With custom embedding function
from chromadb.utils import embedding_functions

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="your-key",
    model_name="text-embedding-3-small"
)

collection = client.create_collection(
    name="my_docs",
    embedding_function=openai_ef
)

# Get existing collection
collection = client.get_collection("my_docs")

# Delete collection
client.delete_collection("my_docs")
```

### 2. 문서 추가

```python
# Add with auto-generated IDs
collection.add(
    documents=["Doc 1", "Doc 2", "Doc 3"],
    metadatas=[
        {"source": "web", "category": "tutorial"},
        {"source": "pdf", "page": 5},
        {"source": "api", "timestamp": "2025-01-01"}
    ],
    ids=["id1", "id2", "id3"]
)

# Add with custom embeddings
collection.add(
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
    documents=["Doc 1", "Doc 2"],
    ids=["id1", "id2"]
)
```

### 3. 쿼리 (유사도 검색)

```python
# Basic query
results = collection.query(
    query_texts=["machine learning tutorial"],
    n_results=5
)

# Query with filters
results = collection.query(
    query_texts=["Python programming"],
    n_results=3,
    where={"source": "web"}
)

# Query with metadata filters
results = collection.query(
    query_texts=["advanced topics"],
    where={
        "$and": [
            {"category": "tutorial"},
            {"difficulty": {"$gte": 3}}
        ]
    }
)

# Access results
print(results["documents"])      # List of matching documents
print(results["metadatas"])      # Metadata for each doc
print(results["distances"])      # Similarity scores
print(results["ids"])            # Document IDs
```

### 4. 문서 가져오기

```python
# Get by IDs
docs = collection.get(
    ids=["id1", "id2"]
)

# Get with filters
docs = collection.get(
    where={"category": "tutorial"},
    limit=10
)

# Get all documents
docs = collection.get()
```

### 5. 문서 업데이트

```python
# Update document content
collection.update(
    ids=["id1"],
    documents=["Updated content"],
    metadatas=[{"source": "updated"}]
)
```

### 6. 문서 삭제

```python
# Delete by IDs
collection.delete(ids=["id1", "id2"])

# Delete with filter
collection.delete(
    where={"source": "outdated"}
)
```

## 영구 저장소

```python
# Persist to disk
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.create_collection("my_docs")
collection.add(documents=["Doc 1"], ids=["id1"])

# Data persisted automatically
# Reload later with same path
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("my_docs")
```

## 임베딩 함수

### 기본값 (Sentence Transformers)

```python
# Uses sentence-transformers by default
collection = client.create_collection("my_docs")
# Default model: all-MiniLM-L6-v2
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
        # Your embedding logic
        return embeddings

my_ef = MyEmbeddingFunction()
collection = client.create_collection(
    name="custom_docs",
    embedding_function=my_ef
)
```

## 메타데이터 필터링

```python
# Exact match
results = collection.query(
    query_texts=["query"],
    where={"category": "tutorial"}
)

# Comparison operators
results = collection.query(
    query_texts=["query"],
    where={"page": {"$gt": 10}}  # $gt, $gte, $lt, $lte, $ne
)

# Logical operators
results = collection.query(
    query_texts=["query"],
    where={
        "$and": [
            {"category": "tutorial"},
            {"difficulty": {"$lte": 3}}
        ]
    }  # Also: $or
)

# Contains
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

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(documents)

# Create Chroma vector store
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_db"
)

# Query
results = vectorstore.similarity_search("machine learning", k=3)

# As retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

## LlamaIndex 통합

```python
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
import chromadb

# Initialize Chroma
db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_or_create_collection("my_collection")

# Create vector store
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create index
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("What is machine learning?")
```

## 서버 모드

```python
# Run Chroma server
# Terminal: chroma run --path ./chroma_db --port 8000

# Connect to server
import chromadb
from chromadb.config import Settings

client = chromadb.HttpClient(
    host="localhost",
    port=8000,
    settings=Settings(anonymized_telemetry=False)
)

# Use as normal
collection = client.get_or_create_collection("my_docs")
```

## 모범 사례

1. **영구 클라이언트 사용** - 재시작 시 데이터를 잃지 않습니다.
2. **메타데이터 추가** - 필터링과 추적이 가능해집니다.
3. **일괄 작업** - 한 번에 여러 문서를 추가합니다.
4. **적절한 임베딩 모델 선택** - 속도와 품질의 균형을 맞춥니다.
5. **필터 사용** - 검색 범위를 좁힙니다.
6. **고유 ID 사용** - 충돌을 방지합니다.
7. **정기적인 백업** - chroma_db 디렉터리를 복사합니다.
8. **컬렉션 크기 모니터링** - 필요하면 확장합니다.
9. **임베딩 함수 테스트** - 품질을 확인합니다.
10. **프로덕션에는 서버 모드 사용** - 여러 사용자에게 더 적합합니다.

## 성능

| 작업 | 지연 시간 | 참고 |
|-----------|---------|-------|
| 문서 100개 추가 | ~1-3초 | 임베딩 포함 |
| 쿼리 (상위 10개) | ~50-200ms | 컬렉션 크기에 따라 다름 |
| 메타데이터 필터 | ~10-50ms | 적절한 인덱싱 시 빠름 |

## 리소스

- **GitHub**: https://github.com/chroma-core/chroma ⭐ 24,300+
- **문서**: https://docs.trychroma.com
- **Discord**: https://discord.gg/MMeYNTmh3x
- **버전**: 1.3.3+
- **라이선스**: Apache 2.0
