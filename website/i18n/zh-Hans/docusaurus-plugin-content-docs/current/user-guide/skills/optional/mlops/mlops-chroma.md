---
title: "Chroma — 面向 AI 应用的开源嵌入数据库"
sidebar_label: "Chroma"
description: "面向 AI 应用的开源嵌入数据库"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Chroma

面向 AI 应用的开源嵌入数据库。存储嵌入向量和元数据，执行向量和全文搜索，按元数据过滤。简洁的 4 函数 API。可从笔记本扩展到生产集群。适用于语义搜索、RAG 应用或文档检索。最适合本地开发和开源项目。

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/mlops/chroma` |
| Path | `optional-skills/mlops/chroma` |
| Version | `1.0.0` |
| Author | Orchestra Research |
| License | MIT |
| Dependencies | `chromadb`, `sentence-transformers` |
| Tags | `RAG`, `Chroma`, `Vector Database`, `Embeddings`, `Semantic Search`, `Open Source`, `Self-Hosted`, `Document Retrieval`, `Metadata Filtering` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Chroma - Open-Source Embedding Database

The AI-native database for building LLM applications with memory.

## When to use Chroma

**Use Chroma when:**
- Building RAG (retrieval-augmented generation) applications
- Need local/self-hosted vector database
- Want open-source solution (Apache 2.0)
- Prototyping in notebooks
- Semantic search over documents
- Storing embeddings with metadata

**Metrics**:
- **24,300+ GitHub stars**
- **1,900+ forks**
- **v1.3.3** (stable, weekly releases)
- **Apache 2.0 license**

**替代方案**:
- **Pinecone**: Managed cloud, auto-scaling
- **FAISS**: Pure similarity search, no metadata
- **Weaviate**: Production ML-native database
- **Qdrant**: High performance, Rust-based

## 快速入门

### 安装

```bash
# Python
pip install chromadb

# JavaScript/TypeScript
npm install chromadb @chroma-core/default-embed
```

### Basic usage (Python)

```python
import chromadb

# 创建客户端
client = chromadb.Client()

# 创建集合
collection = client.create_collection(name="my_collection")

# 添加文档
collection.add(
    documents=["This is document 1", "This is document 2"],
    metadatas=[{"source": "doc1"}, {"source": "doc2"}],
    ids=["id1", "id2"]
)

# 查询
results = collection.query(
    query_texts=["document about topic"],
    n_results=2
)

print(results)
```

## Core operations

### 1. 创建集合

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

# 获取现有集合
collection = client.get_collection("my_docs")

# 删除集合
client.delete_collection("my_docs")
```

### 2. 添加文档

```python
# 使用自动生成的 ID 添加
collection.add(
    documents=["Doc 1", "Doc 2", "Doc 3"],
    metadatas=[
        {"source": "web", "category": "tutorial"},
        {"source": "pdf", "page": 5},
        {"source": "api", "timestamp": "2025-01-01"}
    ],
    ids=["id1", "id2", "id3"]
)

# 使用自定义嵌入添加
collection.add(
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
    documents=["Doc 1", "Doc 2"],
    ids=["id1", "id2"]
)
```

### 3. 查询（相似性搜索）

```python
# 基本查询
results = collection.query(
    query_texts=["machine learning tutorial"],
    n_results=5
)

# 查询 with filters
results = collection.query(
    query_texts=["Python programming"],
    n_results=3,
    where={"source": "web"}
)

# 查询 with metadata filters
results = collection.query(
    query_texts=["advanced topics"],
    where={
        "$and": [
            {"category": "tutorial"},
            {"difficulty": {"$gte": 3}}
        ]
    }
)

# 访问结果
print(results["documents"])      # List of matching documents
print(results["metadatas"])      # Metadata for each doc
print(results["distances"])      # Similarity scores
print(results["ids"])            # Document IDs
```

### 4. 获取文档

```python
# 按 ID 获取
docs = collection.get(
    ids=["id1", "id2"]
)

# 带过滤器获取
docs = collection.get(
    where={"category": "tutorial"},
    limit=10
)

# 获取所有文档
docs = collection.get()
```

### 5. 更新文档

```python
# 更新文档内容
collection.update(
    ids=["id1"],
    documents=["Updated content"],
    metadatas=[{"source": "updated"}]
)
```

### 6. 删除文档

```python
# 按 ID 删除s
collection.delete(ids=["id1", "id2"])

# 带过滤器删除
collection.delete(
    where={"source": "outdated"}
)
```

## 持久存储

```python
# 持久化到磁盘
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.create_collection("my_docs")
collection.add(documents=["Doc 1"], ids=["id1"])

# 数据自动持久化
# 稍后使用相同路径重新加载
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("my_docs")
```

## Embedding functions

### 默认（Sentence Transformers）

```python
# 默认使用 sentence-transformers
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

### 自定义嵌入函数

```python
from chromadb import Documents, EmbeddingFunction, Embeddings

class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # 你的嵌入逻辑
        return embeddings

my_ef = MyEmbeddingFunction()
collection = client.create_collection(
    name="custom_docs",
    embedding_function=my_ef
)
```

## Metadata filtering

```python
# 精确匹配
results = collection.query(
    query_texts=["query"],
    where={"category": "tutorial"}
)

# 比较运算符
results = collection.query(
    query_texts=["query"],
    where={"page": {"$gt": 10}}  # $gt, $gte, $lt, $lte, $ne
)

# 逻辑运算符
results = collection.query(
    query_texts=["query"],
    where={
        "$and": [
            {"category": "tutorial"},
            {"difficulty": {"$lte": 3}}
        ]
    }  # Also: $or
)

# 包含
results = collection.query(
    query_texts=["query"],
    where={"tags": {"$in": ["python", "ml"]}}
)
```

## LangChain 集成

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

# 查询
results = vectorstore.similarity_search("machine learning", k=3)

# 作为检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

## LlamaIndex 集成

```python
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, 存储Context
import chromadb

# 初始化 Chroma
db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_or_create_collection("my_collection")

# 创建向量存储
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = 存储Context.from_defaults(vector_store=vector_store)

# 创建集合
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

# 查询
query_engine = index.as_query_engine()
response = query_engine.query("What is machine learning?")
```

## Server mode

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

## Best practices

1. **使用持久客户端** - Don't lose data on restart
2. **添加元数据** - Enables filtering and tracking
3. **批量操作** - Add multiple docs at once
4. **选择合适的嵌入模型** - Balance speed/quality
5. **使用过滤器** - Narrow search space
6. **唯一 ID** - Avoid collisions
7. **定期备份** - Copy chroma_db directory
8. **监控集合大小** - Scale up if needed
9. **测试嵌入函数** - Ensure quality
10. **生产环境使用服务器模式** - Better for multi-user

## Performance

| 操作 | 延迟 | 备注 |
|-----------|---------|-------|
| Add 100 docs | ~1-3s | With embedding |
| 查询 (top 10) | ~50-200ms | Depends on collection size |
| 元数据过滤 | ~10-50ms | Fast with proper indexing |

## Resources

- **GitHub**: https://github.com/chroma-core/chroma ⭐ 24,300+
- **Docs**: https://docs.trychroma.com
- **Discord**: https://discord.gg/MMeYNTmh3x
- **Version**: 1.3.3+
- **License**: Apache 2.0
