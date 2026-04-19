---
sidebar_position: 9
title: "Optional Skills Catalog"
description: "Official optional skills shipped with hermes-agent — install via hermes skills install official/<category>/<skill>"
---

# Optional Skills Catalog

Official optional skills ship with the hermes-agent repository under `optional-skills/` but are **not active by default**. Install them explicitly:

```bash
hermes skills install official/<category>/<skill>
```

For example:

```bash
hermes skills install official/blockchain/solana
hermes skills install official/mlops/flash-attention
```

Once installed, the skill appears in the agent's skill list and can be loaded automatically when relevant tasks are detected.

To uninstall:

```bash
hermes skills uninstall <skill-name>
```

---

## Autonomous AI Agents

| Skill | Description |
|-------|-------------|
| **blackbox** | Delegate coding tasks to Blackbox AI CLI agent. Multi-model agent with built-in judge that runs tasks through multiple LLMs and picks the best result. |
| **honcho** | Configure and use Honcho memory with Hermes — cross-session user modeling, multi-profile peer isolation, observation config, and dialectic reasoning. |

## Blockchain

| Skill | Description |
|-------|-------------|
| **base** | Query Base (Ethereum L2) blockchain data with USD pricing — wallet balances, token info, transaction details, gas analysis, contract inspection, whale detection, and live network stats. No API key required. |
| **solana** | Query Solana blockchain data with USD pricing — wallet balances, token portfolios, transaction details, NFTs, whale detection, and live network stats. No API key required. |

## Communication

| Skill | Description |
|-------|-------------|
| **one-three-one-rule** | Structured communication framework for proposals and decision-making. |

## Creative

| Skill | Description |
|-------|-------------|
| **blender-mcp** | Control Blender directly from Hermes via socket connection to the blender-mcp addon. Create 3D objects, materials, animations, and run arbitrary Blender Python (bpy) code. |
| **concept-diagrams** | Generate flat, minimal light/dark-aware SVG diagrams as standalone HTML files, using a unified educational visual language (9 semantic color ramps, automatic dark mode). Best for physics setups, chemistry mechanisms, math curves, physical objects (aircraft, turbines, smartphones), floor plans, cross-sections, lifecycle/process narratives, and hub-spoke system diagrams. Ships with 15 example diagrams. |
| **meme-generation** | Generate real meme images by picking a template and overlaying text with Pillow. Produces actual `.png` meme files. |
| **touchdesigner-mcp** | Control a running TouchDesigner instance via the twozero MCP plugin — create operators, set parameters, wire connections, execute Python, build real-time audio-reactive visuals and GLSL networks. 36 native tools. |

## DevOps

| Skill | Description |
|-------|-------------|
| **cli** | Run 150+ AI apps via inference.sh CLI (infsh) — image generation, video creation, LLMs, search, 3D, and social automation. |
| **docker-management** | Manage Docker containers, images, volumes, networks, and Compose stacks — lifecycle ops, debugging, cleanup, and Dockerfile optimization. |

## Email

| Skill | Description |
|-------|-------------|
| **agentmail** | Give the agent its own dedicated email inbox via AgentMail. Send, receive, and manage email autonomously using agent-owned email addresses. |

## Health

| Skill | Description |
|-------|-------------|
| **fitness-nutrition** | Gym workout planner and nutrition tracker. Search 690+ exercises by muscle, equipment, or category via wger. Look up macros and calories for 380,000+ foods via USDA FoodData Central. Computes BMI, TDEE, one-rep max, macro splits, and body fat — pure Python, no pip installs. |
| **neuroskill-bci** | Brain-Computer Interface (BCI) integration for neuroscience research workflows. |

## MCP

| Skill | Description |
|-------|-------------|
| **fastmcp** | Build, test, inspect, install, and deploy MCP servers with FastMCP in Python. Covers wrapping APIs or databases as MCP tools, exposing resources or prompts, and deployment. |

## Migration

| Skill | Description |
|-------|-------------|
| **openclaw-migration** | Migrate a user's OpenClaw customization footprint into Hermes Agent. Imports memories, SOUL.md, command allowlists, user skills, and selected workspace assets. |

## MLOps

The largest optional category — covers the full ML pipeline from data curation to production inference.

| Skill | Description |
|-------|-------------|
| **accelerate** | Simplest distributed training API. 4 lines to add distributed support to any PyTorch script. Unified API for DeepSpeed/FSDP/Megatron/DDP. |
| **chroma** | Open-source embedding database. Store embeddings and metadata, perform vector and full-text search. Simple 4-function API for RAG and semantic search. |
| **faiss** | Facebook's library for efficient similarity search and clustering of dense vectors. Supports billions of vectors, GPU acceleration, and various index types (Flat, IVF, HNSW). |
| **flash-attention** | Optimize transformer attention with Flash Attention for 2-4x speedup and 10-20x memory reduction. Supports PyTorch SDPA, flash-attn library, H100 FP8, and sliding window. |
| **guidance** | Control LLM output with regex and grammars, guarantee valid JSON/XML/code generation, enforce structured formats, and build multi-step workflows with Guidance — Microsoft Research's constrained generation framework. |
| **hermes-atropos-environments** | Build, test, and debug Hermes Agent RL environments for Atropos training. Covers the HermesAgentBaseEnv interface, reward functions, agent loop integration, and evaluation. |
| **huggingface-tokenizers** | Fast Rust-based tokenizers for research and production. Tokenizes 1GB in under 20 seconds. Supports BPE, WordPiece, and Unigram algorithms. |
| **instructor** | Extract structured data from LLM responses with Pydantic validation, retry failed extractions automatically, and stream partial results. |
| **lambda-labs** | Reserved and on-demand GPU cloud instances for ML training and inference. SSH access, persistent filesystems, and multi-node clusters. |
| **llava** | Large Language and Vision Assistant — visual instruction tuning and image-based conversations combining CLIP vision with LLaMA language models. |
| **nemo-curator** | GPU-accelerated data curation for LLM training. Fuzzy deduplication (16x faster), quality filtering (30+ heuristics), semantic dedup, PII redaction. Scales with RAPIDS. |
| **pinecone** | Managed vector database for production AI. Auto-scaling, hybrid search (dense + sparse), metadata filtering, and low latency (under 100ms p95). |
| **pytorch-lightning** | High-level PyTorch framework with Trainer class, automatic distributed training (DDP/FSDP/DeepSpeed), callbacks, and minimal boilerplate. |
| **qdrant** | High-performance vector similarity search engine. Rust-powered with fast nearest neighbor search, hybrid search with filtering, and scalable vector storage. |
| **saelens** | Train and analyze Sparse Autoencoders (SAEs) using SAELens to decompose neural network activations into interpretable features. |
| **simpo** | Simple Preference Optimization — reference-free alternative to DPO with better performance (+6.4 pts on AlpacaEval 2.0). No reference model needed. |
| **slime** | LLM post-training with RL using Megatron+SGLang framework. Custom data generation workflows and tight Megatron-LM integration for RL scaling. |
| **tensorrt-llm** | Optimize LLM inference with NVIDIA TensorRT for maximum throughput. 10-100x faster than PyTorch on A100/H100 with quantization (FP8/INT4) and in-flight batching. |
| **torchtitan** | PyTorch-native distributed LLM pretraining with 4D parallelism (FSDP2, TP, PP, CP). Scale from 8 to 512+ GPUs with Float8 and torch.compile. |

## Productivity

| Skill | Description |
|-------|-------------|
| **canvas** | Canvas LMS integration — fetch enrolled courses and assignments using API token authentication. |
| **memento-flashcards** | Spaced repetition flashcard system for learning and knowledge retention. |
| **siyuan** | SiYuan Note API for searching, reading, creating, and managing blocks and documents in a self-hosted knowledge base. |
| **telephony** | Give Hermes phone capabilities — provision a Twilio number, send/receive SMS/MMS, make calls, and place AI-driven outbound calls through Bland.ai or Vapi. |

## Research

| Skill | Description |
|-------|-------------|
| **bioinformatics** | Gateway to 400+ bioinformatics skills from bioSkills and ClawBio. Covers genomics, transcriptomics, single-cell, variant calling, pharmacogenomics, metagenomics, and structural biology. |
| **domain-intel** | Passive domain reconnaissance using Python stdlib. Subdomain discovery, SSL certificate inspection, WHOIS lookups, DNS records, and bulk multi-domain analysis. No API keys required. |
| **duckduckgo-search** | Free web search via DuckDuckGo — text, news, images, videos. No API key needed. |
| **gitnexus-explorer** | Index a codebase with GitNexus and serve an interactive knowledge graph via web UI and Cloudflare tunnel. |
| **parallel-cli** | Vendor skill for Parallel CLI — agent-native web search, extraction, deep research, enrichment, and monitoring. |
| **qmd** | Search personal knowledge bases, notes, docs, and meeting transcripts locally using qmd — a hybrid retrieval engine with BM25, vector search, and LLM reranking. |
| **scrapling** | Web scraping with Scrapling — HTTP fetching, stealth browser automation, Cloudflare bypass, and spider crawling via CLI and Python. |

## Security

| Skill | Description |
|-------|-------------|
| **1password** | Set up and use 1Password CLI (op). Install the CLI, enable desktop app integration, sign in, and read/inject secrets for commands. |
| **oss-forensics** | Open-source software forensics — analyze packages, dependencies, and supply chain risks. |
| **sherlock** | OSINT username search across 400+ social networks. Hunt down social media accounts by username. |

---

## Contributing Optional Skills

To add a new optional skill to the repository:

1. Create a directory under `optional-skills/<category>/<skill-name>/`
2. Add a `SKILL.md` with standard frontmatter (name, description, version, author)
3. Include any supporting files in `references/`, `templates/`, or `scripts/` subdirectories
4. Submit a pull request — the skill will appear in this catalog once merged
