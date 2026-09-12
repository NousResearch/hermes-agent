---
sidebar_position: 9
title: "Optional Skills Catalog"
description: "Official optional skills shipped with hermes-agent — install via hermes skills install official/<category>/<skill>"
---

# Optional Skills Catalog

Optional skills ship with hermes-agent under `optional-skills/` but are **not active by default**. Install them explicitly:

```bash
hermes skills install official/<category>/<skill>
```

For example:

```bash
hermes skills install official/blockchain/solana
hermes skills install official/mlops/flash-attention
```

Each skill below links to a dedicated page with its full definition, setup, and usage.

To uninstall:

```bash
hermes skills uninstall <skill-name>
```

## autonomous-ai-agents

| Skill | Description |
|-------|-------------|
| [**antigravity-cli**](/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-antigravity-cli) | Operate the Antigravity CLI (agy): plugins, auth, sandbox. |
| [**blackbox**](/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-blackbox) | Delegate coding tasks to the Blackbox AI multi-model CLI. |
| [**dynamic-workflow**](/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-dynamic-workflow) | Plan-in-code fan-outs, adversarial verification, waves. |
| [**grok**](/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-grok) | Delegate coding to xAI Grok Build CLI (features, PRs). |
| [**honcho**](/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-honcho) | Configure and troubleshoot Honcho memory for Hermes. |
| [**openhands**](/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-openhands) | Delegate coding to OpenHands CLI (model-agnostic, LiteLLM). |

## blockchain

| Skill | Description |
|-------|-------------|
| [**evm**](/user-guide/skills/optional/blockchain/blockchain-evm) | Read-only EVM client: wallets, tokens, gas across 8 chains. |
| [**hyperliquid**](/user-guide/skills/optional/blockchain/blockchain-hyperliquid) | Hyperliquid market data, account history, trade review. |
| [**solana**](/user-guide/skills/optional/blockchain/blockchain-solana) | Query Solana wallets, tokens, txs, and NFTs in USD. |

## communication

| Skill | Description |
|-------|-------------|
| [**one-three-one-rule**](/user-guide/skills/optional/communication/communication-one-three-one-rule) | 1-3-1 decision briefs: problem, three options, one pick. |

## creative

| Skill | Description |
|-------|-------------|
| [**archify**](/user-guide/skills/optional/creative/creative-archify) | Validated interactive HTML diagrams, upstream-maintained. |
| [**ascii-art**](/user-guide/skills/optional/creative/creative-ascii-art) | ASCII art: pyfiglet, cowsay, boxes, image-to-ascii. |
| [**audiocraft-audio-generation**](/user-guide/skills/optional/creative/creative-audiocraft-audio-generation) | AudioCraft: MusicGen text-to-music, AudioGen text-to-sound. |
| [**baoyu-article-illustrator**](/user-guide/skills/optional/creative/creative-baoyu-article-illustrator) | Article illustrations: type × style × palette consistency. |
| [**baoyu-comic**](/user-guide/skills/optional/creative/creative-baoyu-comic) | Knowledge comics (知识漫画): educational, biography, tutorial. |
| [**comfyui**](/user-guide/skills/optional/creative/creative-comfyui) | Generate images, video, and audio via diffusion workflows. |
| [**concept-diagrams**](/user-guide/skills/optional/creative/creative-concept-diagrams) | Generate flat, minimal educational SVG visuals as HTML. |
| [**creative-ideation**](/user-guide/skills/optional/creative/creative-creative-ideation) | Generate ideas via named methods from creative practice. |
| [**draw-your-font**](/user-guide/skills/optional/creative/creative-draw-your-font) | Turn a handwriting photo into an installable TTF font. |
| [**excalidraw**](/user-guide/skills/optional/creative/creative-excalidraw) | Hand-drawn Excalidraw JSON diagrams (arch, flow, seq). |
| [**heartmula**](/user-guide/skills/optional/creative/creative-heartmula) | HeartMuLa: Suno-like song generation from lyrics + tags. |
| [**hyperframes**](/user-guide/skills/optional/creative/creative-hyperframes) | Render MP4/WebM videos from HTML compositions. |
| [**impeccable**](/user-guide/skills/optional/creative/creative-impeccable) | Frontend design guidance, upstream-maintained (impeccable). |
| [**kanban-video-orchestrator**](/user-guide/skills/optional/creative/creative-kanban-video-orchestrator) | Plan and run multi-agent video production pipelines. |
| [**meme-generation**](/user-guide/skills/optional/creative/creative-meme-generation) | Create meme PNGs from templates with Pillow text overlay. |
| [**pixel-art**](/user-guide/skills/optional/creative/creative-pixel-art) | Pixel art w/ era palettes (NES, Game Boy, PICO-8). |
| [**pretext**](/user-guide/skills/optional/creative/creative-pretext) | Build creative browser demos with DOM-free text layout. |
| [**simple-english**](/user-guide/skills/optional/creative/creative-simple-english) | Rewrite text to ASD-STE100 Simplified Technical English. |
| [**sketch**](/user-guide/skills/optional/creative/creative-sketch) | Throwaway HTML mockups: 2-3 design variants to compare. |
| [**social-media-content-calendar**](/user-guide/skills/optional/creative/creative-social-media-content-calendar) | Plan multi-platform social campaigns: briefs to posting. |
| [**tldraw-offline**](/user-guide/skills/optional/creative/creative-tldraw-offline) | Drive and script tldraw offline canvases with an agent. |
| [**unreal-mcp**](/user-guide/skills/optional/creative/creative-unreal-mcp) | Automate Unreal Engine editor scenes, actors, and renders. |

## data-science

| Skill | Description |
|-------|-------------|
| [**jupyter-notebook**](/user-guide/skills/optional/data-science/data-science-jupyter-notebook) | Iterative Python via live Jupyter kernel (hamelnb). |

## devops

| Skill | Description |
|-------|-------------|
| [**actual-setup**](/user-guide/skills/optional/devops/devops-actual-setup) | Set up Actual Computer (actual.inc) inference in Hermes. |
| [**docker-management**](/user-guide/skills/optional/devops/devops-docker-management) | Manage Docker containers, images, volumes, and Compose. |
| [**hermes-s6-container-supervision**](/user-guide/skills/optional/devops/devops-hermes-s6-container-supervision) | Modify or debug s6 services in the Hermes Docker image. |
| [**inference-sh-cli**](/user-guide/skills/optional/devops/devops-inference-sh-cli) | Run 150+ AI apps (image, video, LLM) via inference.sh CLI. |
| [**pinggy-tunnel**](/user-guide/skills/optional/devops/devops-pinggy-tunnel) | Zero-install localhost tunnels over SSH via Pinggy. |
| [**setup-wizard-generator**](/user-guide/skills/optional/devops/devops-setup-wizard-generator) | Generate a bash wizard guiding a human through manual setup. |
| [**watchers**](/user-guide/skills/optional/devops/devops-watchers) | Poll RSS, JSON APIs, and GitHub with watermark dedup. |

## dogfood

| Skill | Description |
|-------|-------------|
| [**adversarial-ux-test**](/user-guide/skills/optional/dogfood/dogfood-adversarial-ux-test) | Roleplay a hostile user to find and triage UX pain points. |

## email

| Skill | Description |
|-------|-------------|
| [**agentmail**](/user-guide/skills/optional/email/email-agentmail) | Use when an agent needs AgentMail CLI email inboxes. |

## finance

| Skill | Description |
|-------|-------------|
| [**3-statement-model**](/user-guide/skills/optional/finance/finance-3-statement-model) | Build integrated IS/BS/CF financial workbooks in Excel. |
| [**comps-analysis**](/user-guide/skills/optional/finance/finance-comps-analysis) | Build comparable-company valuation workbooks in Excel. |
| [**dcf-model**](/user-guide/skills/optional/finance/finance-dcf-model) | Build discounted cash flow valuation workbooks in Excel. |
| [**excel-author**](/user-guide/skills/optional/finance/finance-excel-author) | Build auditable financial workbooks headless via openpyxl. |
| [**lbo-model**](/user-guide/skills/optional/finance/finance-lbo-model) | Build leveraged buyout workbooks with IRR/MOIC in Excel. |
| [**merger-model**](/user-guide/skills/optional/finance/finance-merger-model) | Build M&A accretion/dilution workbooks in Excel. |
| [**polymarket**](/user-guide/skills/optional/finance/finance-polymarket) | Query Polymarket: markets, prices, orderbooks, history. |
| [**pptx-author**](/user-guide/skills/optional/finance/finance-pptx-author) | Build PowerPoint decks headless with python-pptx. |
| [**stocks**](/user-guide/skills/optional/finance/finance-stocks) | Stock quotes, history, search, compare, crypto via Yahoo. |

## gaming

| Skill | Description |
|-------|-------------|
| [**minecraft-modpack-server**](/user-guide/skills/optional/gaming/gaming-minecraft-modpack-server) | Host modded Minecraft servers (CurseForge, Modrinth). |
| [**pokemon-player**](/user-guide/skills/optional/gaming/gaming-pokemon-player) | Play Pokemon via headless emulator + RAM reads. |

## health

| Skill | Description |
|-------|-------------|
| [**fitness-nutrition**](/user-guide/skills/optional/health/health-fitness-nutrition) | Workout planning, macros, and body metrics via wger/USDA. |
| [**neuroskill-bci**](/user-guide/skills/optional/health/health-neuroskill-bci) | Use live BCI cognitive and mood state from NeuroSkill. |

## mcp

| Skill | Description |
|-------|-------------|
| [**fastmcp**](/user-guide/skills/optional/mcp/mcp-fastmcp) | Build, test, and deploy Python MCP servers. |
| [**mcp-oauth-remote-gateway**](/user-guide/skills/optional/mcp/mcp-mcp-oauth-remote-gateway) | Manual OAuth for remote MCP servers on headless gateways. |
| [**mcporter**](/user-guide/skills/optional/mcp/mcp-mcporter) | List, auth, and call MCP servers/tools from the terminal. |

## migration

| Skill | Description |
|-------|-------------|
| [**openclaw-migration**](/user-guide/skills/optional/migration/migration-openclaw-migration) | Import an OpenClaw setup (memories, skills) into Hermes. |

## mlops

| Skill | Description |
|-------|-------------|
| [**accelerate**](/user-guide/skills/optional/mlops/mlops-accelerate) | Run PyTorch training across GPUs with minimal changes. |
| [**axolotl**](/user-guide/skills/optional/mlops/mlops-training-axolotl) | Axolotl: YAML LLM fine-tuning (LoRA, DPO, GRPO). |
| [**chroma**](/user-guide/skills/optional/mlops/mlops-chroma) | Embedding database for RAG and semantic search. |
| [**clip**](/user-guide/skills/optional/mlops/mlops-clip) | Zero-shot image classification and image-text search. |
| [**dspy**](/user-guide/skills/optional/mlops/mlops-research-dspy) | DSPy: declarative LM programs, auto-optimize prompts, RAG. |
| [**evaluating-llms-harness**](/user-guide/skills/optional/mlops/mlops-evaluation-evaluating-llms-harness) | lm-eval-harness: benchmark LLMs (MMLU, GSM8K, etc.). |
| [**faiss**](/user-guide/skills/optional/mlops/mlops-faiss) | Fast vector similarity search at billion scale. |
| [**flash-attention**](/user-guide/skills/optional/mlops/mlops-flash-attention) | Speed up long-sequence transformer training and inference. |
| [**guidance**](/user-guide/skills/optional/mlops/mlops-guidance) | Constrain LLM output with grammars; guarantee valid JSON. |
| [**huggingface-hub**](/user-guide/skills/optional/mlops/mlops-models-huggingface-hub) | HuggingFace hf CLI: search/download/upload models, datasets. |
| [**huggingface-tokenizers**](/user-guide/skills/optional/mlops/mlops-huggingface-tokenizers) | Fast BPE/WordPiece tokenization and custom vocab training. |
| [**instructor**](/user-guide/skills/optional/mlops/mlops-instructor) | Structured LLM outputs validated with Pydantic. |
| [**lambda-labs**](/user-guide/skills/optional/mlops/mlops-lambda-labs) | On-demand GPU cloud instances for ML training. |
| [**llama-cpp**](/user-guide/skills/optional/mlops/mlops-inference-llama-cpp) | llama.cpp local GGUF inference + HF Hub model discovery. |
| [**llava**](/user-guide/skills/optional/mlops/mlops-llava) | Vision-language chat: VQA, captioning, image dialogue. |
| [**modal**](/user-guide/skills/optional/mlops/mlops-modal) | Serverless GPU cloud for ML jobs and model APIs. |
| [**nemo-curator**](/user-guide/skills/optional/mlops/mlops-nemo-curator) | Curate LLM training data: dedupe, filter, PII redaction. |
| [**obliteratus**](/user-guide/skills/optional/mlops/mlops-obliteratus) | OBLITERATUS: abliterate LLM refusals (diff-in-means). |
| [**outlines**](/user-guide/skills/optional/mlops/mlops-inference-outlines) | Outlines: structured JSON/regex/Pydantic LLM generation. |
| [**peft**](/user-guide/skills/optional/mlops/mlops-peft) | Fine-tune large LLMs with LoRA on limited GPU memory. |
| [**pinecone**](/user-guide/skills/optional/mlops/mlops-pinecone) | Managed vector DB for production RAG and search. |
| [**pytorch-fsdp**](/user-guide/skills/optional/mlops/mlops-pytorch-fsdp) | Fully sharded data-parallel training for large models. |
| [**pytorch-lightning**](/user-guide/skills/optional/mlops/mlops-pytorch-lightning) | Clean training loops with built-in distributed support. |
| [**qdrant**](/user-guide/skills/optional/mlops/mlops-qdrant) | Vector search engine for production RAG systems. |
| [**saelens**](/user-guide/skills/optional/mlops/mlops-saelens) | Train sparse autoencoders to interpret model features. |
| [**segment-anything-model**](/user-guide/skills/optional/mlops/mlops-models-segment-anything-model) | SAM: zero-shot image segmentation via points, boxes, masks. |
| [**serving-llms-vllm**](/user-guide/skills/optional/mlops/mlops-inference-serving-llms-vllm) | vLLM: high-throughput LLM serving, OpenAI API, quantization. |
| [**simpo**](/user-guide/skills/optional/mlops/mlops-simpo) | Reference-free preference alignment, simpler than DPO. |
| [**slime**](/user-guide/skills/optional/mlops/mlops-slime) | RL post-training for LLMs with Megatron and SGLang. |
| [**stable-diffusion**](/user-guide/skills/optional/mlops/mlops-stable-diffusion) | Text-to-image generation, inpainting, and img2img. |
| [**tensorrt-llm**](/user-guide/skills/optional/mlops/mlops-tensorrt-llm) | High-throughput LLM inference on NVIDIA GPUs. |
| [**torchtitan**](/user-guide/skills/optional/mlops/mlops-torchtitan) | Pretrain LLMs at scale with PyTorch 4D parallelism. |
| [**trl-fine-tuning**](/user-guide/skills/optional/mlops/mlops-training-trl-fine-tuning) | TRL: SFT, DPO, GRPO, RLOO reward modeling for LLM RLHF. |
| [**unsloth**](/user-guide/skills/optional/mlops/mlops-training-unsloth) | Unsloth: 2-5x faster LoRA/QLoRA fine-tuning, less VRAM. |
| [**weights-and-biases**](/user-guide/skills/optional/mlops/mlops-evaluation-weights-and-biases) | W&B: log ML experiments, sweeps, model registry, dashboards. |
| [**whisper**](/user-guide/skills/optional/mlops/mlops-whisper) | Transcribe and translate speech in 99 languages. |

## payments

| Skill | Description |
|-------|-------------|
| [**mpp-agent**](/user-guide/skills/optional/payments/payments-mpp-agent) | Pay HTTP 402 APIs via Machine Payments Protocol (MPP). |
| [**stripe-link-cli**](/user-guide/skills/optional/payments/payments-stripe-link-cli) | Agent payments via Stripe Link — cards, SPT, approvals. |
| [**stripe-projects**](/user-guide/skills/optional/payments/payments-stripe-projects) | Provision SaaS services + sync creds via Stripe Projects. |

## productivity

| Skill | Description |
|-------|-------------|
| [**canvas**](/user-guide/skills/optional/productivity/productivity-canvas) | Fetch Canvas LMS courses and assignments via API token. |
| [**decision-questionnaire**](/user-guide/skills/optional/productivity/productivity-decision-questionnaire) | Turn an unanswerable decision into a questionnaire doc. |
| [**here-now**](/user-guide/skills/optional/productivity/productivity-here-now) | Publish sites to &#123;slug&#125;.here.now and store files in Drives. |
| [**memento-flashcards**](/user-guide/skills/optional/productivity/productivity-memento-flashcards) | Spaced-repetition flashcards: create, review, quiz, export. |
| [**property-listings**](/user-guide/skills/optional/productivity/productivity-property-listings) | Present property and rental listings as desktop cards. |
| [**shop**](/user-guide/skills/optional/productivity/productivity-shop) | Shop catalog search, checkout, order tracking, returns. |
| [**shopify**](/user-guide/skills/optional/productivity/productivity-shopify) | Query Shopify Admin/Storefront GraphQL APIs via curl. |
| [**siyuan**](/user-guide/skills/optional/productivity/productivity-siyuan) | Query and edit a SiYuan knowledge base via its API. |
| [**telephony**](/user-guide/skills/optional/productivity/productivity-telephony) | Provision Twilio numbers, SMS/MMS, and AI outbound calls. |

## research

| Skill | Description |
|-------|-------------|
| [**bioinformatics**](/user-guide/skills/optional/research/research-bioinformatics) | Gateway to 400+ genomics and computational biology skills. |
| [**blogwatcher**](/user-guide/skills/optional/research/research-blogwatcher) | Monitor blogs and RSS/Atom feeds via blogwatcher-cli tool. |
| [**darwinian-evolver**](/user-guide/skills/optional/research/research-darwinian-evolver) | Evolve prompts/regex/SQL/code with Imbue's evolution loop. |
| [**domain-intel**](/user-guide/skills/optional/research/research-domain-intel) | Passive recon of subdomains, SSL certs, WHOIS, and DNS. |
| [**drug-discovery**](/user-guide/skills/optional/research/research-drug-discovery) | Drug discovery: ChEMBL search, drug-likeness, interactions. |
| [**duckduckgo-search**](/user-guide/skills/optional/research/research-duckduckgo-search) | Free keyless web, news, and image search via ddgs. |
| [**gitnexus-explorer**](/user-guide/skills/optional/research/research-gitnexus-explorer) | Serve an interactive codebase knowledge graph web UI. |
| [**osint-investigation**](/user-guide/skills/optional/research/research-osint-investigation) | Follow the money via public records and sanctions data. |
| [**parallel-cli**](/user-guide/skills/optional/research/research-parallel-cli) | Agent-native web search, deep research, and enrichment. |
| [**pinecone-research**](/user-guide/skills/optional/research/research-pinecone-research) | Agent RAG and long-term memory with Pinecone. |
| [**qmd**](/user-guide/skills/optional/research/research-qmd) | Hybrid local search over notes, docs, and transcripts. |
| [**research-paper-writing**](/user-guide/skills/optional/research/research-research-paper-writing) | Write ML papers for NeurIPS/ICML/ICLR: design→submit. |
| [**rss-feeds**](/user-guide/skills/optional/research/research-rss-feeds) | Read RSS, Atom, JSON feeds; discover feeds behind a page. |
| [**scrapling**](/user-guide/skills/optional/research/research-scrapling) | Scrape sites with stealth browsing and Cloudflare bypass. |
| [**searxng-search**](/user-guide/skills/optional/research/research-searxng-search) | Free keyless meta-search aggregating 70+ engines. |

## security

| Skill | Description |
|-------|-------------|
| [**1password**](/user-guide/skills/optional/security/security-1password) | Set up op CLI, sign in, and read or inject secrets. |
| [**godmode**](/user-guide/skills/optional/security/security-godmode) | Jailbreak LLMs: Parseltongue, GODMODE, ULTRAPLINIAN. |
| [**oss-forensics**](/user-guide/skills/optional/security/security-oss-forensics) | GitHub supply-chain forensics: recovery, IOCs, reporting. |
| [**sherlock**](/user-guide/skills/optional/security/security-sherlock) | Find accounts for a username across 400+ platforms. |
| [**unbroker**](/user-guide/skills/optional/security/security-unbroker) | Autonomously remove your info from data-broker sites. |
| [**web-pentest**](/user-guide/skills/optional/security/security-web-pentest) | Authorized web pentest: recon, proof-based exploits, report. |

## smart-home

| Skill | Description |
|-------|-------------|
| [**openhue**](/user-guide/skills/optional/smart-home/smart-home-openhue) | Control Philips Hue lights, scenes, rooms via OpenHue CLI. |

## social-media

| Skill | Description |
|-------|-------------|
| [**reddit-reading**](/user-guide/skills/optional/social-media/social-media-reddit-reading) | Read Reddit: subreddits, search, threads, users. No browser. |

## software-development

| Skill | Description |
|-------|-------------|
| [**ast-grep**](/user-guide/skills/optional/software-development/software-development-ast-grep) | AST-aware structural code search and rewrite via ast-grep. |
| [**code-wiki**](/user-guide/skills/optional/software-development/software-development-code-wiki) | Generate wiki docs + Mermaid diagrams for any codebase. |
| [**grill-me**](/user-guide/skills/optional/software-development/software-development-grill-me) | Adversarial plan interview before implementation. |
| [**rest-graphql-debug**](/user-guide/skills/optional/software-development/software-development-rest-graphql-debug) | Debug REST/GraphQL APIs: status codes, auth, schemas, repro. |
| [**subagent-driven-development**](/user-guide/skills/optional/software-development/software-development-subagent-driven-development) | Execute plans via delegate_task subagents (2-stage review). |

## web-development

| Skill | Description |
|-------|-------------|
| [**cloudflare-temporary-deploy**](/user-guide/skills/optional/web-development/web-development-cloudflare-temporary-deploy) | Deploy a Worker live, no account, via wrangler --temporary. |
| [**har-derived-api-client**](/user-guide/skills/optional/web-development/web-development-har-derived-api-client) | Record a site's XHR into a HAR, derive an HTTP client. |
| [**page-agent**](/user-guide/skills/optional/web-development/web-development-page-agent) | Embed an in-page natural-language GUI copilot in web apps. |
| [**publish-site**](/user-guide/skills/optional/web-development/web-development-publish-site) | Versioned site deploys to GitHub/Cloudflare/Netlify Pages. |

## yuanbao

| Skill | Description |
|-------|-------------|
| [**yuanbao**](/user-guide/skills/optional/yuanbao/yuanbao-yuanbao) | Yuanbao (元宝) groups: @mention users, query info/members. |

---

## Contributing Optional Skills

To add a new optional skill to the repository:

1. Create a directory under `optional-skills/<category>/<skill-name>/`
2. Add a `SKILL.md` with standard frontmatter (name, description, version, author)
3. Include any supporting files in `references/`, `templates/`, or `scripts/` subdirectories
4. Submit a pull request — the skill will appear in this catalog and get its own docs page once merged
