---
title: "Pinecone Research — Pinecone을 활용한 에이전트 RAG 및 장기 메모리"
sidebar_label: "Pinecone Research"
description: "Pinecone을 활용한 에이전트 RAG 및 장기 메모리"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동 생성합니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Pinecone Research

Pinecone을 활용한 에이전트 RAG 및 장기 메모리입니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/research/pinecone-research`로 설치 |
| 경로 | `optional-skills/research/pinecone-research` |
| 버전 | `1.0.0` |
| 작성자 | immuhammadfurqan |
| 라이선스 | MIT |
| 종속성 | `pinecone-client`, `langchain-pinecone` |
| 플랫폼 | linux, macos, windows |
| 태그 | `RAG`, `Pinecone`, `Memory`, `Research`, `Vector Database`, `Agent`, `Retrieval` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 지침으로 보게 되는 내용입니다.
:::

# Pinecone Research — 에이전트 RAG 및 장기 메모리

Pinecone을 에이전트 대화의 검색 증강 생성(RAG) 백엔드로 사용하세요. 임베딩을 저장하고, 과거 세션에서 관련 컨텍스트를 검색하며, 장기 메모리를 구축할 수 있습니다.

## 이 스킬을 사용하는 경우

**다음과 같은 경우 사용하세요:**
- Pinecone을 벡터 저장소로 사용하는 에이전트 RAG 파이프라인 구축
- 에이전트 세션 간에 유지되는 장기 메모리가 필요한 경우
- 검색과 에이전트 도구 사용을 결합하는 경우
- 시맨틱 검색 워크플로를 연구하거나 프로토타이핑하는 경우

**다음과 같은 경우에는 대신 mlops/pinecone 스킬을 사용하세요:**
- 일반적인 Pinecone 참고 자료가 필요한 경우(인덱스 관리, CRUD, 하이브리드 검색)
- 에이전트 통합 없이 프로덕션 인프라에서 작업하는 경우

## 빠른 시작

### 설정

```bash
pip install pinecone-client langchain-pinecone langchain-openai
```

API 키를 설정하세요:
```bash
export PINECONE_API_KEY="your-api-key"
```

### 기본 RAG 파이프라인

```python
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# Initialize Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Create or connect to index
index_name = "agent-memory"
if index_name not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# Build vector store
vectorstore = PineconeVectorStore.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(),
    index_name=index_name,
)

# Retrieve relevant context
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
results = retriever.invoke("What did the agent discuss yesterday?")
```

### 네임스페이스 기반 세션 메모리

```python
# Store per-session memory
vectorstore = PineconeVectorStore(
    index=pc.Index(index_name),
    embedding=OpenAIEmbeddings(),
    namespace=f"session-{session_id}",
)

# Query across all sessions (no namespace filter)
all_memory = PineconeVectorStore(
    index=pc.Index(index_name),
    embedding=OpenAIEmbeddings(),
)
results = all_memory.similarity_search("relevant query", k=10)
```

## 모범 사례

1. **세션 또는 사용자별 네임스페이스 지정** — 멀티 테넌트 에이전트의 데이터를 격리합니다.
2. **일괄 업서트** — 효율성을 위해 배치당 벡터 100~200개를 사용합니다.
3. **메타데이터 필터링** — 세션 ID, 타임스탬프, 주제로 벡터에 태그를 지정합니다.
4. **오래된 메모리 정리** — 비용을 관리하기 위해 오래된 네임스페이스를 삭제합니다.
5. **서버리스 사용** — 자동 확장 및 사용량 기반 과금이 가능합니다.

## 리소스

- **Pinecone 문서**: https://docs.pinecone.io
- **LangChain 통합**: https://python.langchain.com/docs/integrations/vectorstores/pinecone
- **무료 티어**: 인덱스 1개, 벡터 100K개(1536차원)
