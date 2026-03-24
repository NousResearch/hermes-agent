---
sidebar_position: 5
title: "Bundled Skills Catalog"
description: "Catalog of bundled skills that ship with Hermes Agent"
---

# Bundled Skills Catalog

Hermes ships with a large built-in skill library copied into `~/.hermes/skills/` on install. This page catalogs the bundled skills that live in the repository under `skills/`.

## apple

Apple/macOS-specific skills — iMessage, Reminders, Notes, FindMy, and macOS automation. These skills only load on macOS systems.

| Skill | Description | Path |
|-------|-------------|------|
| `apple-notes` | Manage Apple Notes via the memo CLI on macOS (create, view, search, edit). | `apple/apple-notes` |
| `apple-reminders` | Manage Apple Reminders via remindctl CLI (list, add, complete, delete). | `apple/apple-reminders` |
| `findmy` | Track Apple devices and AirTags via FindMy.app on macOS using AppleScript and screen capture. | `apple/findmy` |
| `imessage` | Send and receive iMessages/SMS via the imsg CLI on macOS. | `apple/imessage` |

## autonomous-ai-agents

Skills for spawning and orchestrating autonomous AI coding agents and multi-agent workflows — running independent agent processes, delegating tasks, and coordinating parallel workstreams.

| Skill | Description | Path |
|-------|-------------|------|
| `claude-code` | Delegate coding tasks to Claude Code (Anthropic's CLI agent). Use for building features, refactoring, PR reviews, and iterative coding. Requires the claude CLI installed. | `autonomous-ai-agents/claude-code` |
| `codex` | Delegate coding tasks to OpenAI Codex CLI agent. Use for building features, refactoring, PR reviews, and batch issue fixing. Requires the codex CLI and a git repository. | `autonomous-ai-agents/codex` |
| `hermes-agent-spawning` | Spawn additional Hermes Agent instances as autonomous subprocesses for independent long-running tasks. Supports non-interactive one-shot mode (-q) and interactive PTY mode for multi-turn collaboration. Different from delegate_task — this runs a full separate hermes process. | `autonomous-ai-agents/hermes-agent` |
| `opencode` | Delegate coding tasks to OpenCode CLI agent for feature implementation, refactoring, PR review, and long-running autonomous sessions. Requires the opencode CLI installed and authenticated. | `autonomous-ai-agents/opencode` |

## data-science

Skills for data science workflows — interactive exploration, Jupyter notebooks, data analysis, and visualization.

| Skill | Description | Path |
|-------|-------------|------|
| `jupyter-live-kernel` | Use a live Jupyter kernel for stateful, iterative Python execution via hamelnb. Load this skill when the task involves exploration, iteration, or inspecting intermediate results. | `data-science/jupyter-live-kernel` |

## creative

Creative content generation — ASCII art, hand-drawn style diagrams, and visual design tools.

| Skill | Description | Path |
|-------|-------------|------|
| `ascii-art` | Generate ASCII art using pyfiglet (571 fonts), cowsay, boxes, toilet, image-to-ascii, remote APIs (asciified, ascii.co.uk), and LLM fallback. No API keys required. | `creative/ascii-art` |
| `ascii-video` | "Production pipeline for ASCII art video — any format. Converts video/audio/images/generative input into colored ASCII character video output (MP4, GIF, image sequence). Covers: video-to-ASCII conversion, audio-reactive music visualizers, generative ASCII art animations, hybrid… | `creative/ascii-video` |
| `excalidraw` | Create hand-drawn style diagrams using Excalidraw JSON format. Generate .excalidraw files for architecture diagrams, flowcharts, sequence diagrams, concept maps, and more. Files can be opened at excalidraw.com or uploaded for shareable links. | `creative/excalidraw` |

## dogfood

| Skill | Description | Path |
|-------|-------------|------|
| `dogfood` | Systematic exploratory QA testing of web applications — find bugs, capture evidence, and generate structured reports. | `dogfood/dogfood` |
| `hermes-agent-setup` | Help users configure Hermes Agent — CLI usage, setup wizard, model/provider selection, tools, skills, voice/STT/TTS, gateway, and troubleshooting. | `dogfood/hermes-agent-setup` |

## email

Skills for sending, receiving, searching, and managing email from the terminal.

| Skill | Description | Path |
|-------|-------------|------|
| `himalaya` | CLI to manage emails via IMAP/SMTP. Use himalaya to list, read, write, reply, forward, search, and organize emails from the terminal. Supports multiple accounts and message composition with MML (MIME Meta Language). | `email/himalaya` |

## gaming

Skills for setting up, configuring, and managing game servers, modpacks, and gaming-related infrastructure.

| Skill | Description | Path |
|-------|-------------|------|
| `minecraft-modpack-server` | Set up a modded Minecraft server from a CurseForge/Modrinth server pack zip. Covers NeoForge/Forge install, Java version, JVM tuning, firewall, LAN config, backups, and launch scripts. | `gaming/minecraft-modpack-server` |
| `pokemon-player` | Play Pokemon games autonomously via headless emulation. Starts a game server, reads structured game state from RAM, makes strategic decisions, and sends button inputs — all from the terminal. | `gaming/pokemon-player` |

## github

GitHub workflow skills for managing repositories, pull requests, code reviews, issues, and CI/CD pipelines using the gh CLI and git via terminal.

| Skill | Description | Path |
|-------|-------------|------|
| `codebase-inspection` | Inspect and analyze codebases using pygount for LOC counting, language breakdown, and code-vs-comment ratios. Use when asked to check lines of code, repo size, language composition, or codebase stats. | `github/codebase-inspection` |
| `github-auth` | Set up GitHub authentication for the agent using git (universally available) or the gh CLI. Covers HTTPS tokens, SSH keys, credential helpers, and gh auth — with a detection flow to pick the right method automatically. | `github/github-auth` |
| `github-code-review` | Review code changes by analyzing git diffs, leaving inline comments on PRs, and performing thorough pre-push review. Works with gh CLI or falls back to git + GitHub REST API via curl. | `github/github-code-review` |
| `github-issues` | Create, manage, triage, and close GitHub issues. Search existing issues, add labels, assign people, and link to PRs. Works with gh CLI or falls back to git + GitHub REST API via curl. | `github/github-issues` |
| `github-pr-workflow` | Full pull request lifecycle — create branches, commit changes, open PRs, monitor CI status, auto-fix failures, and merge. Works with gh CLI or falls back to git + GitHub REST API via curl. | `github/github-pr-workflow` |
| `github-repo-management` | Clone, create, fork, configure, and manage GitHub repositories. Manage remotes, secrets, releases, and workflows. Works with gh CLI or falls back to git + GitHub REST API via curl. | `github/github-repo-management` |

## inference-sh

Skills for AI app execution via inference.sh cloud platform.

| Skill | Description | Path |
|-------|-------------|------|
| `inference-sh-cli` | Run 150+ AI apps via inference.sh CLI (infsh) — image generation, video creation, LLMs, search, 3D, social automation. | `inference-sh/cli` |

## leisure

| Skill | Description | Path |
|-------|-------------|------|
| `find-nearby` | Find nearby places (restaurants, cafes, bars, pharmacies, etc.) using OpenStreetMap. Works with coordinates, addresses, cities, zip codes, or Telegram location pins. No API keys needed. | `leisure/find-nearby` |

## mcp

Skills for working with MCP (Model Context Protocol) servers, tools, and integrations. Includes the built-in native MCP client (configure servers in config.yaml for automatic tool discovery) and the mcporter CLI bridge for ad-hoc server interaction.

| Skill | Description | Path |
|-------|-------------|------|
| `mcporter` | Use the mcporter CLI to list, configure, auth, and call MCP servers/tools directly (HTTP or stdio), including ad-hoc servers, config edits, and CLI/type generation. | `mcp/mcporter` |
| `native-mcp` | Built-in MCP (Model Context Protocol) client that connects to external MCP servers, discovers their tools, and registers them as native Hermes Agent tools. Supports stdio and HTTP transports with automatic reconnection, security filtering, and zero-config tool injection. | `mcp/native-mcp` |

## media

Skills for working with media content — YouTube transcripts, GIF search, music generation, and audio visualization.

| Skill | Description | Path |
|-------|-------------|------|
| `gif-search` | Search and download GIFs from Tenor using curl. No dependencies beyond curl and jq. Useful for finding reaction GIFs, creating visual content, and sending GIFs in chat. | `media/gif-search` |
| `heartmula` | Set up and run HeartMuLa, the open-source music generation model family (Suno-like). Generates full songs from lyrics + tags with multilingual support. | `media/heartmula` |
| `songsee` | Generate spectrograms and audio feature visualizations (mel, chroma, MFCC, tempogram, etc.) from audio files via CLI. Useful for audio analysis, music production debugging, and visual documentation. | `media/songsee` |
| `youtube-content` | Fetch YouTube video transcripts and transform them into structured content (chapters, summaries, threads, blog posts). | `media/youtube-content` |

## mlops

General-purpose ML operations tools — model hub management, dataset operations, and workflow orchestration.

| Skill | Description | Path |
|-------|-------------|------|
| `huggingface-hub` | Hugging Face Hub CLI (hf) — search, download, and upload models and datasets, manage repos, deploy inference endpoints. | `mlops/huggingface-hub` |

## mlops/cloud

GPU cloud providers and serverless compute platforms for ML workloads.

| Skill | Description | Path |
|-------|-------------|------|
| `lambda-labs-gpu-cloud` | Reserved and on-demand GPU cloud instances for ML training and inference. Use when you need dedicated GPU instances with simple SSH access, persistent filesystems, or high-performance multi-node clusters for large-scale training. | `mlops/cloud/lambda-labs` |
| `modal-serverless-gpu` | Serverless GPU cloud platform for running ML workloads. Use when you need on-demand GPU access without infrastructure management, deploying ML models as APIs, or running batch jobs with automatic scaling. | `mlops/cloud/modal` |

## mlops/evaluation

Model evaluation benchmarks, experiment tracking, data curation, tokenizers, and interpretability tools.

| Skill | Description | Path |
|-------|-------------|------|
| `evaluating-llms-harness` | Evaluates LLMs across 60+ academic benchmarks (MMLU, HumanEval, GSM8K, TruthfulQA, HellaSwag). Use when benchmarking model quality, comparing models, reporting academic results, or tracking training progress. Industry standard used by EleutherAI, HuggingFace, and major labs. Sup… | `mlops/evaluation/lm-evaluation-harness` |
| `huggingface-tokenizers` | Fast tokenizers optimized for research and production. Rust-based implementation tokenizes 1GB in &lt;20 seconds. Supports BPE, WordPiece, and Unigram algorithms. Train custom vocabularies, track alignments, handle padding/truncation. Integrates seamlessly with transformers. Use… | `mlops/evaluation/huggingface-tokenizers` |
| `nemo-curator` | GPU-accelerated data curation for LLM training. Supports text/image/video/audio. Features fuzzy deduplication (16× faster), quality filtering (30+ heuristics), semantic deduplication, PII redaction, NSFW detection. Scales across GPUs with RAPIDS. Use for preparing high-quality t… | `mlops/evaluation/nemo-curator` |
| `sparse-autoencoder-training` | Provides guidance for training and analyzing Sparse Autoencoders (SAEs) using SAELens to decompose neural network activations into interpretable features. Use when discovering interpretable features, analyzing superposition, or studying monosemantic representations in language m… | `mlops/evaluation/saelens` |
| `weights-and-biases` | Track ML experiments with automatic logging, visualize training in real-time, optimize hyperparameters with sweeps, and manage model registry with W&B - collaborative MLOps platform | `mlops/evaluation/weights-and-biases` |

## mlops/inference

Model serving, quantization (GGUF/GPTQ), structured output, inference optimization, and model surgery tools for deploying and running LLMs.

| Skill | Description | Path |
|-------|-------------|------|
| `gguf-quantization` | GGUF format and llama.cpp quantization for efficient CPU/GPU inference. Use when deploying models on consumer hardware, Apple Silicon, or when needing flexible quantization from 2-8 bit without GPU requirements. | `mlops/inference/gguf` |
| `guidance` | Control LLM output with regex and grammars, guarantee valid JSON/XML/code generation, enforce structured formats, and build multi-step workflows with Guidance - Microsoft Research's constrained generation framework | `mlops/inference/guidance` |
| `instructor` | Extract structured data from LLM responses with Pydantic validation, retry failed extractions automatically, parse complex JSON with type safety, and stream partial results with Instructor - battle-tested structured output library | `mlops/inference/instructor` |
| `llama-cpp` | Runs LLM inference on CPU, Apple Silicon, and consumer GPUs without NVIDIA hardware. Use for edge deployment, M1/M2/M3 Macs, AMD/Intel GPUs, or when CUDA is unavailable. Supports GGUF quantization (1.5-8 bit) for reduced memory and 4-10× speedup vs PyTorch on CPU. | `mlops/inference/llama-cpp` |
| `obliteratus` | Remove refusal behaviors from open-weight LLMs using OBLITERATUS — mechanistic interpretability techniques (diff-in-means, SVD, whitened SVD, LEACE, SAE decomposition, etc.) to excise guardrails while preserving reasoning. 9 CLI methods, 28 analysis modules, 116 model presets ac… | `mlops/inference/obliteratus` |
| `outlines` | Guarantee valid JSON/XML/code structure during generation, use Pydantic models for type-safe outputs, support local models (Transformers, vLLM), and maximize inference speed with Outlines - dottxt.ai's structured generation library | `mlops/inference/outlines` |
| `serving-llms-vllm` | Serves LLMs with high throughput using vLLM's PagedAttention and continuous batching. Use when deploying production LLM APIs, optimizing inference latency/throughput, or serving models with limited GPU memory. Supports OpenAI-compatible endpoints, quantization (GPTQ/AWQ/FP8), an… | `mlops/inference/vllm` |
| `tensorrt-llm` | Optimizes LLM inference with NVIDIA TensorRT for maximum throughput and lowest latency. Use for production deployment on NVIDIA GPUs (A100/H100), when you need 10-100x faster inference than PyTorch, or for serving models with quantization (FP8/INT4), in-flight batching, and mult… | `mlops/inference/tensorrt-llm` |

## mlops/models

Specific model architectures and tools — computer vision (CLIP, SAM, Stable Diffusion), speech (Whisper), audio generation (AudioCraft), and multimodal models (LLaVA).

| Skill | Description | Path |
|-------|-------------|------|
| `audiocraft-audio-generation` | PyTorch library for audio generation including text-to-music (MusicGen) and text-to-sound (AudioGen). Use when you need to generate music from text descriptions, create sound effects, or perform melody-conditioned music generation. | `mlops/models/audiocraft` |
| `clip` | OpenAI's model connecting vision and language. Enables zero-shot image classification, image-text matching, and cross-modal retrieval. Trained on 400M image-text pairs. Use for image search, content moderation, or vision-language tasks without fine-tuning. Best for general-purpo… | `mlops/models/clip` |
| `llava` | Large Language and Vision Assistant. Enables visual instruction tuning and image-based conversations. Combines CLIP vision encoder with Vicuna/LLaMA language models. Supports multi-turn image chat, visual question answering, and instruction following. Use for vision-language cha… | `mlops/models/llava` |
| `segment-anything-model` | Foundation model for image segmentation with zero-shot transfer. Use when you need to segment any object in images using points, boxes, or masks as prompts, or automatically generate all object masks in an image. | `mlops/models/segment-anything` |
| `stable-diffusion-image-generation` | State-of-the-art text-to-image generation with Stable Diffusion models via HuggingFace Diffusers. Use when generating images from text prompts, performing image-to-image translation, inpainting, or building custom diffusion pipelines. | `mlops/models/stable-diffusion` |
| `whisper` | OpenAI's general-purpose speech recognition model. Supports 99 languages, transcription, translation to English, and language identification. Six model sizes from tiny (39M params) to large (1550M params). Use for speech-to-text, podcast transcription, or multilingual audio proc… | `mlops/models/whisper` |

## mlops/research

ML research frameworks for building and optimizing AI systems with declarative programming.

| Skill | Description | Path |
|-------|-------------|------|
| `dspy` | Build complex AI systems with declarative programming, optimize prompts automatically, create modular RAG systems and agents with DSPy - Stanford NLP's framework for systematic LM programming | `mlops/research/dspy` |

## mlops/training

Fine-tuning, RLHF/DPO/GRPO training, distributed training frameworks, and optimization tools for training LLMs and other models.

| Skill | Description | Path |
|-------|-------------|------|
| `axolotl` | Expert guidance for fine-tuning LLMs with Axolotl - YAML configs, 100+ models, LoRA/QLoRA, DPO/KTO/ORPO/GRPO, multimodal support | `mlops/training/axolotl` |
| `distributed-llm-pretraining-torchtitan` | Provides PyTorch-native distributed LLM pretraining using torchtitan with 4D parallelism (FSDP2, TP, PP, CP). Use when pretraining Llama 3.1, DeepSeek V3, or custom models at scale from 8 to 512+ GPUs with Float8, torch.compile, and distributed checkpointing. | `mlops/training/torchtitan` |
| `fine-tuning-with-trl` | Fine-tune LLMs using reinforcement learning with TRL - SFT for instruction tuning, DPO for preference alignment, PPO/GRPO for reward optimization, and reward model training. Use when need RLHF, align model with preferences, or train from human feedback. Works with HuggingFace Tr… | `mlops/training/trl-fine-tuning` |
| `grpo-rl-training` | Expert guidance for GRPO/RL fine-tuning with TRL for reasoning and task-specific model training | `mlops/training/grpo-rl-training` |
| `hermes-atropos-environments` | Build, test, and debug Hermes Agent RL environments for Atropos training. Covers the HermesAgentBaseEnv interface, reward functions, agent loop integration, evaluation with tools, wandb logging, and the three CLI modes (serve/process/evaluate). Use when creating, reviewing, or f… | `mlops/training/hermes-atropos-environments` |
| `huggingface-accelerate` | Simplest distributed training API. 4 lines to add distributed support to any PyTorch script. Unified API for DeepSpeed/FSDP/Megatron/DDP. Automatic device placement, mixed precision (FP16/BF16/FP8). Interactive config, single launch command. HuggingFace ecosystem standard. | `mlops/training/accelerate` |
| `optimizing-attention-flash` | Optimizes transformer attention with Flash Attention for 2-4x speedup and 10-20x memory reduction. Use when training/running transformers with long sequences (&gt;512 tokens), encountering GPU memory issues with attention, or need faster inference. Supports PyTorch native SDPA,… | `mlops/training/flash-attention` |
| `peft-fine-tuning` | Parameter-efficient fine-tuning for LLMs using LoRA, QLoRA, and 25+ methods. Use when fine-tuning large models (7B-70B) with limited GPU memory, when you need to train &lt;1% of parameters with minimal accuracy loss, or for multi-adapter serving. HuggingFace's official library i… | `mlops/training/peft` |
| `pytorch-fsdp` | Expert guidance for Fully Sharded Data Parallel training with PyTorch FSDP - parameter sharding, mixed precision, CPU offloading, FSDP2 | `mlops/training/pytorch-fsdp` |
| `pytorch-lightning` | High-level PyTorch framework with Trainer class, automatic distributed training (DDP/FSDP/DeepSpeed), callbacks system, and minimal boilerplate. Scales from laptop to supercomputer with same code. Use when you want clean training loops with built-in best practices. | `mlops/training/pytorch-lightning` |
| `simpo-training` | Simple Preference Optimization for LLM alignment. Reference-free alternative to DPO with better performance (+6.4 points on AlpacaEval 2.0). No reference model needed, more efficient than DPO. Use for preference alignment when want simpler, faster training than DPO/PPO. | `mlops/training/simpo` |
| `slime-rl-training` | Provides guidance for LLM post-training with RL using slime, a Megatron+SGLang framework. Use when training GLM models, implementing custom data generation workflows, or needing tight Megatron-LM integration for RL scaling. | `mlops/training/slime` |
| `unsloth` | Expert guidance for fast fine-tuning with Unsloth - 2-5x faster training, 50-80% less memory, LoRA/QLoRA optimization | `mlops/training/unsloth` |

## mlops/vector-databases

Vector similarity search and embedding databases for RAG, semantic search, and AI application backends.

| Skill | Description | Path |
|-------|-------------|------|
| `chroma` | Open-source embedding database for AI applications. Store embeddings and metadata, perform vector and full-text search, filter by metadata. Simple 4-function API. Scales from notebooks to production clusters. Use for semantic search, RAG applications, or document retrieval. Best… | `mlops/vector-databases/chroma` |
| `faiss` | Facebook's library for efficient similarity search and clustering of dense vectors. Supports billions of vectors, GPU acceleration, and various index types (Flat, IVF, HNSW). Use for fast k-NN search, large-scale vector retrieval, or when you need pure similarity search without… | `mlops/vector-databases/faiss` |
| `pinecone` | Managed vector database for production AI applications. Fully managed, auto-scaling, with hybrid search (dense + sparse), metadata filtering, and namespaces. Low latency (&lt;100ms p95). Use for production RAG, recommendation systems, or semantic search at scale. Best for server… | `mlops/vector-databases/pinecone` |
| `qdrant-vector-search` | High-performance vector similarity search engine for RAG and semantic search. Use when building production RAG systems requiring fast nearest neighbor search, hybrid search with filtering, or scalable vector storage with Rust-powered performance. | `mlops/vector-databases/qdrant` |

## note-taking

Note taking skills, to save information, assist with research, and collab on multi-session planning and information sharing.

| Skill | Description | Path |
|-------|-------------|------|
| `obsidian` | Read, search, and create notes in the Obsidian vault. | `note-taking/obsidian` |

## productivity

Skills for document creation, presentations, spreadsheets, and other productivity workflows.

| Skill | Description | Path |
|-------|-------------|------|
| `google-workspace` | Gmail, Calendar, Drive, Contacts, Sheets, and Docs integration via Python. Uses OAuth2 with automatic token refresh. No external binaries needed — runs entirely with Google's Python client libraries in the Hermes venv. | `productivity/google-workspace` |
| `linear` | Manage Linear issues, projects, and teams via the GraphQL API. Create, update, search, and organize issues. | `productivity/linear` |
| `nano-pdf` | Edit PDFs with natural-language instructions using the nano-pdf CLI. Modify text, fix typos, update titles, and make content changes to specific pages without manual editing. | `productivity/nano-pdf` |
| `notion` | Notion API for creating and managing pages, databases, and blocks via curl. Search, create, update, and query Notion workspaces directly from the terminal. | `productivity/notion` |
| `ocr-and-documents` | Extract text from PDFs and scanned documents. Use web_extract for remote URLs, pymupdf for local text-based PDFs, marker-pdf for OCR/scanned docs. For DOCX use python-docx, for PPTX see the powerpoint skill. | `productivity/ocr-and-documents` |
| `powerpoint` | "Use this skill any time a .pptx file is involved in any way — as input, output, or both. This includes: creating slide decks, pitch decks, or presentations; reading, parsing, or extracting text from any .pptx file (even if the extracted content will be used elsewhere, like in a… | `productivity/powerpoint` |

## research

Skills for academic research, paper discovery, literature review, domain reconnaissance, market data, content monitoring, and scientific knowledge retrieval.

| Skill | Description | Path |
|-------|-------------|------|
| `arxiv` | Search and retrieve academic papers from arXiv using their free REST API. No API key needed. Search by keyword, author, category, or ID. Combine with web_extract or the ocr-and-documents skill to read full paper content. | `research/arxiv` |
| `blogwatcher` | Monitor blogs and RSS/Atom feeds for updates using the blogwatcher CLI. Add blogs, scan for new articles, and track what you've read. | `research/blogwatcher` |
| `domain-intel` | Passive domain reconnaissance using Python stdlib. Subdomain discovery, SSL certificate inspection, WHOIS lookups, DNS records, domain availability checks, and bulk multi-domain analysis. No API keys required. | `research/domain-intel` |
| `duckduckgo-search` | Free web search via DuckDuckGo — text, news, images, videos. No API key needed. Use the Python DDGS library or CLI to search, then web_extract for full content. | `research/duckduckgo-search` |
| `parallel-cli` | Optional vendor skill for Parallel CLI — agent-native web search, extraction, deep research, enrichment, FindAll, and monitoring. | `research/parallel-cli` |
| `ml-paper-writing` | Write publication-ready ML/AI papers for NeurIPS, ICML, ICLR, ACL, AAAI, COLM. Use when drafting papers from research repos, structuring arguments, verifying citations, or preparing camera-ready submissions. Includes LaTeX templates, reviewer guidelines, and citation verificatio… | `research/ml-paper-writing` |
| `polymarket` | Query Polymarket prediction market data — search markets, get prices, orderbooks, and price history. Read-only via public REST APIs, no API key needed. | `research/polymarket` |

## smart-home

Skills for controlling smart home devices — lights, switches, sensors, and home automation systems.

| Skill | Description | Path |
|-------|-------------|------|
| `openhue` | Control Philips Hue lights, rooms, and scenes via the OpenHue CLI. Turn lights on/off, adjust brightness, color, color temperature, and activate scenes. | `smart-home/openhue` |

## social-media

Skills for interacting with social platforms — posting, reading, monitoring, and account operations.

| Skill | Description | Path |
|-------|-------------|------|
| `xitter` | Interact with X/Twitter via the x-cli terminal client using official X API credentials. | `social-media/xitter` |

## software-development

| Skill | Description | Path |
|-------|-------------|------|
| `code-review` | Guidelines for performing thorough code reviews with security and quality focus | `software-development/code-review` |
| `plan` | Plan mode for Hermes — inspect context, write a markdown plan into `.hermes/plans/` in the active workspace/backend working directory, and do not execute the work. | `software-development/plan` |
| `requesting-code-review` | Use when completing tasks, implementing major features, or before merging. Validates work meets requirements through systematic review process. | `software-development/requesting-code-review` |
| `subagent-driven-development` | Use when executing implementation plans with independent tasks. Dispatches fresh delegate_task per task with two-stage review (spec compliance then code quality). | `software-development/subagent-driven-development` |
| `systematic-debugging` | Use when encountering any bug, test failure, or unexpected behavior. 4-phase root cause investigation — NO fixes without understanding the problem first. | `software-development/systematic-debugging` |
| `test-driven-development` | Use when implementing any feature or bugfix, before writing implementation code. Enforces RED-GREEN-REFACTOR cycle with test-first approach. | `software-development/test-driven-development` |
| `writing-plans` | Use when you have a spec or requirements for a multi-step task. Creates comprehensive implementation plans with bite-sized tasks, exact file paths, and complete code examples. | `software-development/writing-plans` |
