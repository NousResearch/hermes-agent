---
title: "Pinecone Research — Agent RAG and long-term memory with Pinecone"
sidebar_label: "Pinecone Research"
description: "Agent RAG and long-term memory with Pinecone"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Pinecone Research

Agent RAG and long-term memory with Pinecone.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/research/pinecone-research` |
| Path | `optional-skills/research/pinecone-research` |
| Version | `1.0.0` |
| Author | immuhammadfurqan |
| License | MIT |
| Dependencies | `pinecone-client`, `langchain-pinecone` |
| Platforms | linux, macos, windows |
| Tags | `RAG`, `Pinecone`, `Memory`, `Research`, `Vector Database`, `Agent`, `Retrieval` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Pinecone Research — Agent RAG & Long-Term Memory

Use Pinecone as a retrieval-augmented generation (RAG) backend for agent
conversations: persist embeddings, retrieve relevant context from past
sessions, and build long-term memory.

## When to use this skill

**Use when:**
- Building agent RAG pipelines with Pinecone as the vector store
- Need persistent long-term memory across agent sessions
- Combining retrieval with agent tool use
- Researching or prototyping semantic search workflows

**Use the mlops/pinecone skill instead when:**
- Need a general Pinecone reference (index management, CRUD, hybrid search)
- Working on production infrastructure without agent integration

## Quick start

### Setup

```bash
pip install pinecone-client langchain-pinecone langchain-openai
```

Set your API key:
```bash
export PINECONE_API_KEY="your-api-key"
```

### Basic RAG pipeline

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

### Namespace-based session memory

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

## Best practices

1. **Namespace by session or user** — isolate data for multi-tenant agents
2. **Batch upserts** — 100–200 vectors per batch for efficiency
3. **Metadata filtering** — tag vectors with session ID, timestamp, topic
4. **Prune old memory** — delete stale namespaces to control costs
5. **Use serverless** — auto-scaling, pay-per-use pricing

## Resources

- **Pinecone Docs**: https://docs.pinecone.io
- **LangChain Integration**: https://python.langchain.com/docs/integrations/vectorstores/pinecone
- **Free Tier**: 1 index, 100K vectors (1536 dimensions)
