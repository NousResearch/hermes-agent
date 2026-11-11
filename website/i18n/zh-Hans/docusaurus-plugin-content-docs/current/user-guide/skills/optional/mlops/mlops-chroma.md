---
title: "Chroma — 面向 AI 应用的开源 embedding 数据库"
sidebar_label: "Chroma"
description: "面向 AI 应用的开源 embedding 数据库"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Chroma

面向 AI 应用的开源 embedding（向量嵌入）数据库。存储 embedding 与元数据，执行向量搜索和全文搜索，按元数据过滤。简洁的 4 函数 API，从 notebook 到生产集群均可扩展。适用于语义搜索、RAG 应用或文档检索。最适合本地开发和开源项目。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/mlops/chroma` 安装 |
| 路径 | `optional-skills/mlops/chroma` |
| 版本 | `1.0.0` |
| 作者 | Orchestra Research |
| 许可证 | MIT |
| 依赖 | `chromadb`, `sentence-transformers` |
| 平台 | linux, macos, windows |
| 标签 | `RAG`, `Chroma`, `Vector Database`, `Embeddings`, `Semantic Search`, `Open Source`, `Self-Hosted`, `Document Retrieval`, `Metadata Filtering` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# Chroma - 开源 Embedding 数据库

专为构建具备记忆能力的 LLM 应用而设计的 AI 原生数据库。

## 何时使用 Chroma

**适用场景：**
- 构建 RAG（检索增强生成）应用
- 需要本地/自托管向量数据库
- 希望使用开源方案（Apache 2.0）
- 在 notebook 中快速原型验证
- 对文档进行语义搜索
- 存储带元数据的 embedding

**指标**：
- **24,300+ GitHub stars**
- **1,900+ forks**
- **v1.3.3**（稳定版，每周发布）
- **Apache 2.0 许可证**

**以下场景请使用替代方案**：
- **Pinecone**：托管云服务，自动扩缩容
- **FAISS**：纯相似度搜索，不支持元数据
- **Weaviate**：面向生产的 ML 原生数据库
- **Qdrant**：高性能，基于 Rust

## 快速开始

### 安装

```bash
# Python
pip install chromadb

# JavaScript/TypeScript
npm install chromadb @chroma-core/default-embed
```

### 基本用法（Python）

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

## 核心操作

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

# Get existing collection
collection = client.get_collection("my_docs")

# Delete collection
client.delete_collection("my_docs")
```

### 2. 添加文档

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

### 3. 查询（相似度搜索）

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

### 4. 获取文档

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

### 5. 更新文档

```python
# Update document content
collection.update(
    ids=["id1"],
    documents=["Updated content"],
    metadatas=[{"source": "updated"}]
)
```

### 6. 删除文档

```python
# Delete by IDs
collection.delete(ids=["id1", "id2"])

# Delete with filter
collection.delete(
    where={"source": "outdated"}
)
```

## 持久化存储

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

## Embedding 函数

### 默认（Sentence Transformers）

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

### 自定义 embedding 函数

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

## 元数据过滤

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

# Query
results = vectorstore.similarity_search("machine learning", k=3)

# As retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

## LlamaIndex 集成

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

## 服务器模式

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

## 最佳实践

1. **使用持久化客户端** — 避免重启后数据丢失
2. **添加元数据** — 支持过滤与追踪
3. **批量操作** — 一次性添加多个文档
4. **选择合适的 embedding 模型** — 平衡速度与质量
5. **使用过滤器** — 缩小搜索范围
6. **唯一 ID** — 避免冲突
7. **定期备份** — 复制 `chroma_db` 目录
8. **监控集合大小** — 按需扩容
9. **测试 embedding 函数** — 确保质量
10. **生产环境使用服务器模式** — 更适合多用户场景

## 性能

| 操作 | 延迟 | 备注 |
|-----------|---------|-------|
| 添加 100 个文档 | ~1-3s | 含 embedding 生成 |
| 查询（top 10） | ~50-200ms | 取决于集合大小 |
| 元数据过滤 | ~10-50ms | 正确索引下速度较快 |

## 资源

- **GitHub**: https://github.com/chroma-core/chroma ⭐ 24,300+
- **文档**: https://docs.trychroma.com
- **Discord**: https://discord.gg/MMeYNTmh3x
- **版本**: 1.3.3+
- **许可证**: Apache 2.0