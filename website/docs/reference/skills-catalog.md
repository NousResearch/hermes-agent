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
| `hermes-agent` | Complete guide to using and extending Hermes Agent — CLI usage, setup, configuration, spawning additional agents, gateway platforms, skills, voice, tools, profiles, and a concise contributor reference. Load this skill when helping users configure Hermes, troubleshoot issues, s… | `autonomous-ai-agents/hermes-agent` |
| `opencode` | Delegate coding tasks to OpenCode CLI agent for feature implementation, refactoring, PR review, and long-running autonomous sessions. Requires the opencode CLI installed and authenticated. | `autonomous-ai-agents/opencode` |

## creative

Creative content generation — ASCII art, hand-drawn diagrams, animations, music, and visual design tools.

| Skill | Description | Path |
|-------|-------------|------|
| `architecture-diagram` | Generate dark-themed SVG diagrams of software systems and cloud infrastructure as standalone HTML files with inline SVG graphics. Semantic component colors (cyan=frontend, emerald=backend, violet=database, amber=cloud/AWS, rose=security, orange=message bus), JetBrains Mono fon… | `creative/architecture-diagram` |
| `ascii-art` | Generate ASCII art using pyfiglet (571 fonts), cowsay, boxes, toilet, image-to-ascii, remote APIs (asciified, ascii.co.uk), and LLM fallback. No API keys required. | `creative/ascii-art` |
| `ascii-video` | Production pipeline for ASCII art video — any format. Converts video/audio/images/generative input into colored ASCII character video output (MP4, GIF, image sequence). Covers: video-to-ASCII conversion, audio-reactive music visualizers, generative ASCII art animations, hybrid… | `creative/ascii-video` |
| `excalidraw` | Create hand-drawn style diagrams using Excalidraw JSON format. Generate .excalidraw files for architecture diagrams, flowcharts, sequence diagrams, concept maps, and more. Files can be opened at excalidraw.com or uploaded for shareable links. | `creative/excalidraw` |
| `ideation` | Generate project ideas through creative constraints. Use when the user says 'I want to build something', 'give me a project idea', 'I'm bored', 'what should I make', 'inspire me', or any variant of 'I have tools but no direction'. Works for code, art, hardware, writing, tools,… | `creative/creative-ideation` |
| `manim-video` | Production pipeline for mathematical and technical animations using Manim Community Edition. Creates 3Blue1Brown-style explainer videos, algorithm visualizations, equation derivations, architecture diagrams, and data stories. Use when users request: animated explanations, math… | `creative/manim-video` |
| `p5js` | Production pipeline for interactive and generative visual art using p5.js. Creates browser-based sketches, generative art, data visualizations, interactive experiences, 3D scenes, audio-reactive visuals, and motion graphics — exported as HTML, PNG, GIF, MP4, or SVG. Covers: 2D… | `creative/p5js` |
| `popular-web-designs` | 54 production-quality design systems extracted from real websites. Load a template to generate HTML/CSS that matches the visual identity of sites like Stripe, Linear, Vercel, Notion, Airbnb, and more. Each template includes colors, typography, components, layout rules, and rea… | `creative/popular-web-designs` |
| `songwriting-and-ai-music` | Songwriting craft, AI music generation prompts (Suno focus), parody/adaptation techniques, phonetic tricks, and lessons learned. These are tools and ideas, not rules. Break any of them when the art calls for it. | `creative/songwriting-and-ai-music` |

## data-science

Skills for data science workflows — interactive exploration, Jupyter notebooks, data analysis, and visualization.

| Skill | Description | Path |
|-------|-------------|------|
| `jupyter-live-kernel` | Use a live Jupyter kernel for stateful, iterative Python execution via hamelnb. Load this skill when the task involves exploration, iteration, or inspecting intermediate results — data science, ML experimentation, API exploration, or building up complex code step-by-step. Uses… | `data-science/jupyter-live-kernel` |

## devops

DevOps and infrastructure automation skills.

| Skill | Description | Path |
|-------|-------------|------|
| `webhook-subscriptions` | Create and manage webhook subscriptions for event-driven agent activation. Use when the user wants external services to trigger agent runs automatically. | `devops/webhook-subscriptions` |

## dogfood

Internal dogfooding and QA skills used to test Hermes Agent itself.

| Skill | Description | Path |
|-------|-------------|------|
| `dogfood` | Systematic exploratory QA testing of web applications — find bugs, capture evidence, and generate structured reports | `dogfood` |

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

GitHub workflow skills for managing repositories, pull requests, code reviews, issues, and CI/CD pipelines.

| Skill | Description | Path |
|-------|-------------|------|
| `codebase-inspection` | Inspect and analyze codebases using pygount for LOC counting, language breakdown, and code-vs-comment ratios. Use when asked to check lines of code, repo size, language composition, or codebase stats. | `github/codebase-inspection` |
| `github-auth` | Set up GitHub authentication for the agent using git (universally available) or the gh CLI. Covers HTTPS tokens, SSH keys, credential helpers, and gh auth — with a detection flow to pick the right method automatically. | `github/github-auth` |
| `github-code-review` | Review code changes by analyzing git diffs, leaving inline comments on PRs, and performing thorough pre-push review. Works with gh CLI or falls back to git + GitHub REST API via curl. | `github/github-code-review` |
| `github-issues` | Create, manage, triage, and close GitHub issues. Search existing issues, add labels, assign people, and link to PRs. Works with gh CLI or falls back to git + GitHub REST API via curl. | `github/github-issues` |
| `github-pr-workflow` | Full pull request lifecycle — create branches, commit changes, open PRs, monitor CI status, auto-fix failures, and merge. Works with gh CLI or falls back to git + GitHub REST API via curl. | `github/github-pr-workflow` |
| `github-repo-management` | Clone, create, fork, configure, and manage GitHub repositories. Manage remotes, secrets, releases, and workflows. Works with gh CLI or falls back to git + GitHub REST API via curl. | `github/github-repo-management` |

## mcp

Skills for working with MCP (Model Context Protocol) servers, tools, and integrations.

| Skill | Description | Path |
|-------|-------------|------|
| `native-mcp` | Built-in MCP (Model Context Protocol) client that connects to external MCP servers, discovers their tools, and registers them as native Hermes Agent tools. Supports stdio and HTTP transports with automatic reconnection, security filtering, and zero-config tool injection. | `mcp/native-mcp` |

## media

Skills for working with media content — YouTube transcripts, GIF search, music generation, and audio visualization.

| Skill | Description | Path |
|-------|-------------|------|
| `gif-search` | Search and download GIFs from Tenor using curl. No dependencies beyond curl and jq. Useful for finding reaction GIFs, creating visual content, and sending GIFs in chat. | `media/gif-search` |
| `heartmula` | Set up and run HeartMuLa, the open-source music generation model family (Suno-like). Generates full songs from lyrics + tags with multilingual support. | `media/heartmula` |
| `songsee` | Generate spectrograms and audio feature visualizations (mel, chroma, MFCC, tempogram, etc.) from audio files via CLI. Useful for audio analysis, music production debugging, and visual documentation. | `media/songsee` |
| `youtube-content` | Fetch YouTube video transcripts and transform them into structured content (chapters, summaries, threads, blog posts). Use when the user shares a YouTube URL or video link, asks to summarize a video, requests a transcript, or wants to extract and reformat content from any YouT… | `media/youtube-content` |

## mlops

General-purpose ML operations tools — model hub management, dataset operations, and workflow orchestration.

| Skill | Description | Path |
|-------|-------------|------|
| `huggingface-hub` | Hugging Face Hub CLI (hf) — search, download, and upload models and datasets, manage repos, query datasets with SQL, deploy inference endpoints, manage Spaces and buckets. | `mlops/huggingface-hub` |

## mlops/evaluation

Model evaluation benchmarks, experiment tracking, and interpretability tools.

| Skill | Description | Path |
|-------|-------------|------|
| `evaluating-llms-harness` | Evaluates LLMs across 60+ academic benchmarks (MMLU, HumanEval, GSM8K, TruthfulQA, HellaSwag). Use when benchmarking model quality, comparing models, reporting academic results, or tracking training progress. Industry standard used by EleutherAI, HuggingFace, and major labs. S… | `mlops/evaluation/lm-evaluation-harness` |
| `weights-and-biases` | Track ML experiments with automatic logging, visualize training in real-time, optimize hyperparameters with sweeps, and manage model registry with W&B - collaborative MLOps platform | `mlops/evaluation/weights-and-biases` |

## mlops/inference

Model serving, quantization (GGUF/GPTQ), structured output, inference optimization, and model surgery tools for deploying and running LLMs.

| Skill | Description | Path |
|-------|-------------|------|
| `llama-cpp` | Run LLM inference with llama.cpp on CPU, Apple Silicon, AMD/Intel GPUs, or NVIDIA — plus GGUF model conversion and quantization (2–8 bit with K-quants and imatrix). Covers CLI, Python bindings, OpenAI-compatible server, and Ollama/LM Studio integration. Use for edge deployment… | `mlops/inference/llama-cpp` |
| `obliteratus` | Remove refusal behaviors from open-weight LLMs using OBLITERATUS — mechanistic interpretability techniques (diff-in-means, SVD, whitened SVD, LEACE, SAE decomposition, etc.) to excise guardrails while preserving reasoning. 9 CLI methods, 28 analysis modules, 116 model presets … | `mlops/inference/obliteratus` |
| `outlines` | Guarantee valid JSON/XML/code structure during generation, use Pydantic models for type-safe outputs, support local models (Transformers, vLLM), and maximize inference speed with Outlines - dottxt.ai's structured generation library | `mlops/inference/outlines` |
| `serving-llms-vllm` | Serves LLMs with high throughput using vLLM's PagedAttention and continuous batching. Use when deploying production LLM APIs, optimizing inference latency/throughput, or serving models with limited GPU memory. Supports OpenAI-compatible endpoints, quantization (GPTQ/AWQ/FP8), … | `mlops/inference/vllm` |

## mlops/models

Specific model architectures — image segmentation (SAM) and audio generation (AudioCraft / MusicGen). Additional model skills (CLIP, Stable Diffusion, Whisper, LLaVA) are available as optional skills.

| Skill | Description | Path |
|-------|-------------|------|
| `audiocraft-audio-generation` | PyTorch library for audio generation including text-to-music (MusicGen) and text-to-sound (AudioGen). Use when you need to generate music from text descriptions, create sound effects, or perform melody-conditioned music generation. | `mlops/models/audiocraft` |
| `segment-anything-model` | Foundation model for image segmentation with zero-shot transfer. Use when you need to segment any object in images using points, boxes, or masks as prompts, or automatically generate all object masks in an image. | `mlops/models/segment-anything` |

## mlops/research

ML research frameworks for building and optimizing AI systems with declarative programming.

| Skill | Description | Path |
|-------|-------------|------|
| `dspy` | Build complex AI systems with declarative programming, optimize prompts automatically, create modular RAG systems and agents with DSPy - Stanford NLP's framework for systematic LM programming | `mlops/research/dspy` |

## mlops/training

Fine-tuning, RLHF/DPO/GRPO training, distributed training frameworks, and optimization tools.

| Skill | Description | Path |
|-------|-------------|------|
| `axolotl` | Expert guidance for fine-tuning LLMs with Axolotl - YAML configs, 100+ models, LoRA/QLoRA, DPO/KTO/ORPO/GRPO, multimodal support | `mlops/training/axolotl` |
| `fine-tuning-with-trl` | Fine-tune LLMs using reinforcement learning with TRL - SFT for instruction tuning, DPO for preference alignment, PPO/GRPO for reward optimization, and reward model training. Use when need RLHF, align model with preferences, or train from human feedback. Works with HuggingFace … | `mlops/training/trl-fine-tuning` |
| `unsloth` | Expert guidance for fast fine-tuning with Unsloth - 2-5x faster training, 50-80% less memory, LoRA/QLoRA optimization | `mlops/training/unsloth` |

## note-taking

Note taking skills, to save information, assist with research, and collaborate on multi-session planning.

| Skill | Description | Path |
|-------|-------------|------|
| `obsidian` | Read, search, and create notes in the Obsidian vault. | `note-taking/obsidian` |

## productivity

Skills for document creation, presentations, spreadsheets, and other productivity workflows.

| Skill | Description | Path |
|-------|-------------|------|
| `google-workspace` | Gmail, Calendar, Drive, Contacts, Sheets, and Docs integration for Hermes. Uses Hermes-managed OAuth2 setup, prefers the Google Workspace CLI (`gws`) when available for broader API coverage, and falls back to the Python client libraries otherwise. | `productivity/google-workspace` |
| `linear` | Manage Linear issues, projects, and teams via the GraphQL API. Create, update, search, and organize issues. Uses API key auth (no OAuth needed). All operations via curl — no dependencies. | `productivity/linear` |
| `maps` | Location intelligence — geocode, reverse-geocode, nearby POI search (44 categories, coordinates or address via `--near`), driving/walking/cycling distance + time, turn-by-turn directions, timezone, bounding box + area, POI search in a rectangle. Uses OpenStreetMap + Overpass + OSRM. No API key needed. Telegram location-pin friendly. | `productivity/maps` |
| `nano-pdf` | Edit PDFs with natural-language instructions using the nano-pdf CLI. Modify text, fix typos, update titles, and make content changes to specific pages without manual editing. | `productivity/nano-pdf` |
| `notion` | Notion API for creating and managing pages, databases, and blocks via curl. Search, create, update, and query Notion workspaces directly from the terminal. | `productivity/notion` |
| `ocr-and-documents` | Extract text from PDFs and scanned documents. Use web_extract for remote URLs, pymupdf for local text-based PDFs, marker-pdf for OCR/scanned docs. For DOCX use python-docx, for PPTX see the powerpoint skill. | `productivity/ocr-and-documents` |
| `powerpoint` | Use this skill any time a .pptx file is involved in any way — as input, output, or both. This includes: creating slide decks, pitch decks, or presentations; reading, parsing, or extracting text from any .pptx file (even if the extracted content will be used elsewhere, like in … | `productivity/powerpoint` |

## red-teaming

Skills for LLM red-teaming, jailbreaking, and safety filter bypass research.

| Skill | Description | Path |
|-------|-------------|------|
| `godmode` | Jailbreak API-served LLMs using G0DM0D3 techniques — Parseltongue input obfuscation (33 techniques), GODMODE CLASSIC system prompt templates, ULTRAPLINIAN multi-model racing, encoding escalation, and Hermes-native prefill/system prompt integration. Use when a user wants to byp… | `red-teaming/godmode` |

## research

Skills for academic research, paper discovery, literature review, market data, content monitoring, and scientific knowledge retrieval.

| Skill | Description | Path |
|-------|-------------|------|
| `arxiv` | Search and retrieve academic papers from arXiv using their free REST API. No API key needed. Search by keyword, author, category, or ID. Combine with web_extract or the ocr-and-documents skill to read full paper content. | `research/arxiv` |
| `blogwatcher` | Monitor blogs and RSS/Atom feeds for updates using the blogwatcher-cli tool. Add blogs, scan for new articles, track read status, and filter by category. | `research/blogwatcher` |
| `llm-wiki` | Karpathy's LLM Wiki — build and maintain a persistent, interlinked markdown knowledge base. Ingest sources, query compiled knowledge, and lint for consistency. | `research/llm-wiki` |
| `polymarket` | Query Polymarket prediction market data — search markets, get prices, orderbooks, and price history. Read-only via public REST APIs, no API key needed. | `research/polymarket` |
| `research-paper-writing` | End-to-end pipeline for writing ML/AI research papers — from experiment design through analysis, drafting, revision, and submission. Covers NeurIPS, ICML, ICLR, ACL, AAAI, COLM. Integrates automated experiment monitoring, statistical analysis, iterative writing, and citation v… | `research/research-paper-writing` |

## smart-home

Skills for controlling smart home devices — lights, switches, sensors, and home automation systems.

| Skill | Description | Path |
|-------|-------------|------|
| `openhue` | Control Philips Hue lights, rooms, and scenes via the OpenHue CLI. Turn lights on/off, adjust brightness, color, color temperature, and activate scenes. | `smart-home/openhue` |

## social-media

Skills for interacting with social platforms — posting, reading, monitoring, and account operations.

| Skill | Description | Path |
|-------|-------------|------|
| `xurl` | Interact with X/Twitter via xurl, the official X API CLI. Use for posting, replying, quoting, searching, timelines, mentions, likes, reposts, bookmarks, follows, DMs, media upload, and raw v2 endpoint access. | `social-media/xurl` |

## software-development

General software-engineering skills — planning, reviewing, debugging, and test-driven development.

| Skill | Description | Path |
|-------|-------------|------|
| `plan` | Plan mode for Hermes — inspect context, write a markdown plan into the active workspace's `.hermes/plans/` directory, and do not execute the work. | `software-development/plan` |
| `requesting-code-review` | Pre-commit verification pipeline — static security scan, baseline-aware quality gates, independent reviewer subagent, and auto-fix loop. Use after code changes and before committing, pushing, or opening a PR. | `software-development/requesting-code-review` |
| `subagent-driven-development` | Use when executing implementation plans with independent tasks. Dispatches fresh delegate_task per task with two-stage review (spec compliance then code quality). | `software-development/subagent-driven-development` |
| `systematic-debugging` | Use when encountering any bug, test failure, or unexpected behavior. 4-phase root cause investigation — NO fixes without understanding the problem first. | `software-development/systematic-debugging` |
| `test-driven-development` | Use when implementing any feature or bugfix, before writing implementation code. Enforces RED-GREEN-REFACTOR cycle with test-first approach. | `software-development/test-driven-development` |
| `writing-plans` | Use when you have a spec or requirements for a multi-step task. Creates comprehensive implementation plans with bite-sized tasks, exact file paths, and complete code examples. | `software-development/writing-plans` |


---

# Optional Skills

Optional skills ship with the repository under `optional-skills/` but are **not active by default**. They cover heavier or niche use cases. Install them with:

```bash
hermes skills install official/<category>/<skill>
```

## autonomous-ai-agents

| Skill | Description | Path |
|-------|-------------|------|
| `blackbox` | Delegate coding tasks to Blackbox AI CLI agent. Multi-model agent with built-in judge that runs tasks through multiple LLMs and picks the best result. Requires the blackbox CLI and a Blackbox AI API key. | `autonomous-ai-agents/blackbox` |

## blockchain

| Skill | Description | Path |
|-------|-------------|------|
| `base` | Query Base (Ethereum L2) blockchain data with USD pricing — wallet balances, token info, transaction details, gas analysis, contract inspection, whale detection, and live network stats. Uses Base RPC + CoinGecko. No API key required. | `blockchain/base` |
| `solana` | Query Solana blockchain data with USD pricing — wallet balances, token portfolios with values, transaction details, NFTs, whale detection, and live network stats. Uses Solana RPC + CoinGecko. No API key required. | `blockchain/solana` |

## creative

| Skill | Description | Path |
|-------|-------------|------|
| `blender-mcp` | Control Blender directly from Hermes via socket connection to the blender-mcp addon. Create 3D objects, materials, animations, and run arbitrary Blender Python (bpy) code. | `creative/blender-mcp` |
| `meme-generation` | Generate real meme images by picking a template and overlaying text with Pillow. Produces actual .png meme files. | `creative/meme-generation` |
| `touchdesigner-mcp` | Control a running TouchDesigner instance via the twozero MCP plugin — create operators, set parameters, wire connections, execute Python, build real-time audio-reactive visuals and GLSL networks. 36 native tools. | `creative/touchdesigner-mcp` |

## devops

| Skill | Description | Path |
|-------|-------------|------|
| `docker-management` | Manage Docker containers, images, volumes, networks, and Compose stacks — lifecycle ops, debugging, cleanup, and Dockerfile optimization. | `devops/docker-management` |

## email

| Skill | Description | Path |
|-------|-------------|------|
| `agentmail` | Give the agent its own dedicated email inbox via AgentMail. Send, receive, and manage email autonomously using agent-owned email addresses (e.g. hermes-agent@agentmail.to). | `email/agentmail` |

## health

| Skill | Description | Path |
|-------|-------------|------|
| `neuroskill-bci` | Connect to a running NeuroSkill instance and incorporate the user's real-time cognitive and emotional state (focus, relaxation, mood, cognitive load, drowsiness, heart rate, HRV, sleep staging, and 40+ derived EXG scores) into responses. Requires a BCI wearable (Muse 2/S or OpenBCI) and the NeuroSkill desktop app. | `health/neuroskill-bci` |

## mcp

| Skill | Description | Path |
|-------|-------------|------|
| `fastmcp` | Build, test, inspect, install, and deploy MCP servers with FastMCP in Python. Use when creating a new MCP server, wrapping an API or database as MCP tools, exposing resources or prompts, or preparing a FastMCP server for HTTP deployment. | `mcp/fastmcp` |

## migration

| Skill | Description | Path |
|-------|-------------|------|
| `openclaw-migration` | Migrate a user's OpenClaw customization footprint into Hermes Agent. Imports Hermes-compatible memories, SOUL.md, command allowlists, user skills, and selected workspace assets from ~/.openclaw, then reports what could not be migrated and why. | `migration/openclaw-migration` |

## productivity

| Skill | Description | Path |
|-------|-------------|------|
| `telephony` | Give Hermes phone capabilities — provision and persist a Twilio number, send and receive SMS/MMS, make direct calls, and place AI-driven outbound calls through Bland.ai or Vapi. | `productivity/telephony` |

## research

| Skill | Description | Path |
|-------|-------------|------|
| `bioinformatics` | Gateway to 400+ bioinformatics skills from bioSkills and ClawBio. Covers genomics, transcriptomics, single-cell, variant calling, pharmacogenomics, metagenomics, structural biology, and more. | `research/bioinformatics` |
| `qmd` | Search personal knowledge bases, notes, docs, and meeting transcripts locally using qmd — a hybrid retrieval engine with BM25, vector search, and LLM reranking. Supports CLI and MCP integration. | `research/qmd` |

## security

| Skill | Description | Path |
|-------|-------------|------|
| `1password` | Set up and use 1Password CLI (op). Use when installing the CLI, enabling desktop app integration, signing in, and reading/injecting secrets for commands. | `security/1password` |
| `oss-forensics` | Supply chain investigation, evidence recovery, and forensic analysis for GitHub repositories. Covers deleted commit recovery, force-push detection, IOC extraction, multi-source evidence collection, and structured forensic reporting. | `security/oss-forensics` |
| `sherlock` | OSINT username search across 400+ social networks. Hunt down social media accounts by username. | `security/sherlock` |
