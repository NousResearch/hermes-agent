---
sidebar_position: 5
title: "Catálogo de Skills Incluídas"
description: "Catálogo de skills incluídas que acompanham o Hermes Agent"
---

# Catálogo de Skills Incluídas

O Hermes vem com uma grande biblioteca de skills embutidas, copiadas para `~/.hermes/skills/` na instalação. Cada skill abaixo tem um link para uma página dedicada com sua definição completa, configuração e uso.

O Hermes também sincroniza as skills incluídas em `hermes update`, mas o manifesto de sincronização respeita exclusões locais e edições do usuário. Se uma skill listada aqui estiver ausente da árvore `~/.hermes/skills/` do seu perfil, ela ainda acompanha o Hermes; restaure-a com `hermes skills reset <name> --restore`.

Se uma skill estiver ausente desta lista mas presente no repositório, o catálogo é regenerado por `website/scripts/generate-skill-docs.py`.

## apple

| Skill | Descrição | Caminho |
|-------|-------------|------|
| [`apple-notes`](/docs/user-guide/skills/bundled/apple/apple-apple-notes) | Gerencia o Apple Notes via CLI memo: criar, buscar, editar. | `apple/apple-notes` |
| [`apple-reminders`](/docs/user-guide/skills/bundled/apple/apple-apple-reminders) | Apple Reminders via remindctl: adicionar, listar, concluir. | `apple/apple-reminders` |
| [`findmy`](/docs/user-guide/skills/bundled/apple/apple-findmy) | Rastreia dispositivos Apple/AirTags via FindMy.app no macOS. | `apple/findmy` |
| [`imessage`](/docs/user-guide/skills/bundled/apple/apple-imessage) | Envia e recebe iMessages/SMS via o CLI imsg no macOS. | `apple/imessage` |

## autonomous-ai-agents

| Skill | Descrição | Caminho |
|-------|-------------|------|
| [`claude-code`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-claude-code) | Delega programação para o CLI do Claude Code (features, PRs). | `autonomous-ai-agents/claude-code` |
| [`codex`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-codex) | Delega programação para o CLI do OpenAI Codex (features, PRs). | `autonomous-ai-agents/codex` |
| [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) | Configura, estende ou contribui com o Hermes Agent. | `autonomous-ai-agents/hermes-agent` |
| [`opencode`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-opencode) | Delega programação para o CLI do OpenCode (features, revisão de PR). | `autonomous-ai-agents/opencode` |

## computer-use

| Skill | Descrição | Caminho |
|-------|-------------|------|
| [`computer-use`](/docs/user-guide/skills/bundled/computer-use/computer-use-computer-use) | Opera o desktop do usuário em segundo plano — clicando, digitando, rolando, arrastando — sem roubar o cursor, o foco de teclado ou trocar áreas de trabalho virtuais / Spaces. Multiplataforma: macOS, Windows, Linux. Funciona com qualquer... | `computer-use` |

## creative

| Skill | Descrição | Caminho |
|-------|-------------|------|
| [`architecture-diagram`](/docs/user-guide/skills/bundled/creative/creative-architecture-diagram) | Diagramas SVG de arquitetura/nuvem/infra em tema escuro como HTML. | `creative/architecture-diagram` |
| [`ascii-art`](/docs/user-guide/skills/bundled/creative/creative-ascii-art) | Arte ASCII: pyfiglet, cowsay, boxes, imagem-para-ascii. | `creative/ascii-art` |
| [`ascii-video`](/docs/user-guide/skills/bundled/creative/creative-ascii-video) | Vídeo ASCII: converte vídeo/áudio em MP4/GIF ASCII colorido. | `creative/ascii-video` |
| [`baoyu-infographic`](/docs/user-guide/skills/bundled/creative/creative-baoyu-infographic) | Infográficos: 21 layouts x 21 estilos (信息图, 可视化). | `creative/baoyu-infographic` |
| [`claude-design`](/docs/user-guide/skills/bundled/creative/creative-claude-design) | Cria artefatos HTML avulsos (landing page, apresentação, prototipagem). | `creative/claude-design` |
| [`comfyui`](/docs/user-guide/skills/bundled/creative/creative-comfyui) | Gera imagens, vídeo e áudio com o ComfyUI — instala, inicia, gerencia nodes/modelos, executa workflows com injeção de parâmetros. Usa o comfy-cli oficial para o ciclo de vida e a API REST/WebSocket direta para execução. | `creative/comfyui` |
| [`design-md`](/docs/user-guide/skills/bundled/creative/creative-design-md) | Cria/valida/exporta arquivos de spec de tokens DESIGN.md do Google. | `creative/design-md` |
| [`excalidraw`](/docs/user-guide/skills/bundled/creative/creative-excalidraw) | Diagramas Excalidraw em JSON com estilo desenhado à mão (arquitetura, fluxo, sequência). | `creative/excalidraw` |
| [`humanizer`](/docs/user-guide/skills/bundled/creative/creative-humanizer) | Humaniza texto: remove marcas de IA e adiciona voz real. | `creative/humanizer` |
| [`manim-video`](/docs/user-guide/skills/bundled/creative/creative-manim-video) | Animações Manim CE: vídeos de matemática/algoritmos estilo 3Blue1Brown. | `creative/manim-video` |
| [`p5js`](/docs/user-guide/skills/bundled/creative/creative-p5js) | Sketches p5.js: arte generativa, shaders, interatividade, 3D. | `creative/p5js` |
| [`popular-web-designs`](/docs/user-guide/skills/bundled/creative/creative-popular-web-designs) | 54 sistemas de design reais (Stripe, Linear, Vercel) como HTML/CSS. | `creative/popular-web-designs` |
| [`pretext`](/docs/user-guide/skills/bundled/creative/creative-pretext) | Use ao construir demos criativas de navegador com @chenglou/pretext — layout de texto sem DOM para arte ASCII, fluxo tipográfico ao redor de obstáculos, jogos de texto-como-geometria, tipografia cinética e arte generativa baseada em texto. Produz HT... de arquivo único | `creative/pretext` |
| [`sketch`](/docs/user-guide/skills/bundled/creative/creative-sketch) | Mockups HTML descartáveis: 2-3 variantes de design para comparar. | `creative/sketch` |
| [`songwriting-and-ai-music`](/docs/user-guide/skills/bundled/creative/creative-songwriting-and-ai-music) | Ofício de composição musical e prompts de música com IA Suno. | `creative/songwriting-and-ai-music` |
| [`touchdesigner-mcp`](/docs/user-guide/skills/bundled/creative/creative-touchdesigner-mcp) | Controla uma instância em execução do TouchDesigner via MCP twozero — cria operadores, define parâmetros, conecta cabos, executa Python, constrói visuais em tempo real. 36 ferramentas nativas. | `creative/touchdesigner-mcp` |

## data-science

| Skill | Descrição | Caminho |
|-------|-------------|------|
| [`jupyter-live-kernel`](/docs/user-guide/skills/bundled/data-science/data-science-jupyter-live-kernel) | Python iterativo via kernel Jupyter ativo (hamelnb). | `data-science/jupyter-live-kernel` |

## dogfood

| Skill | Descrição | Caminho |
|-------|-------------|------|
| [`dogfood`](/docs/user-guide/skills/bundled/dogfood/dogfood-dogfood) | QA exploratório de aplicações web: encontrar bugs, evidências, relatórios. | `dogfood` |

## email

| Skill | Descrição | Caminho |
|-------|-------------|------|
| [`himalaya`](/docs/user-guide/skills/bundled/email/email-himalaya) | CLI Himalaya: e-mail IMAP/SMTP pelo terminal. | `email/himalaya` |

## github

| Skill | Descrição | Caminho |
|-------|-------------|------|
| [`codebase-inspection`](/docs/user-guide/skills/bundled/github/github-codebase-inspection) | Inspeciona bases de código com pygount: LOC, linguagens, proporções. | `github/codebase-inspection` |
| [`github-auth`](/docs/user-guide/skills/bundled/github/github-github-auth) | Configuração de autenticação no GitHub: tokens HTTPS, chaves SSH, login no CLI gh. | `github/github-auth` |
| [`github-code-review`](/docs/user-guide/skills/bundled/github/github-github-code-review) | Revisa PRs: diffs, comentários inline via gh ou REST. | `github/github-code-review` |
| [`github-issues`](/docs/user-guide/skills/bundled/github/github-github-issues) | Cria, triagem, rotula, atribui issues do GitHub via gh ou REST. | `github/github-issues` |
| [`github-pr-workflow`](/docs/user-guide/skills/bundled/github/github-github-pr-workflow) | Ciclo de vida de PR no GitHub: branch, commit, abrir, CI, merge. | `github/github-pr-workflow` |
| [`github-repo-management`](/docs/user-guide/skills/bundled/github/github-github-repo-management) | Clona/cria/faz fork de repositórios; gerencia remotes, releases. | `github/github-repo-management` |

## hermes-desktop-plugins

| Skill | Descrição | Caminho |
|-------|-------------|------|
| [`hermes-desktop-plugins`](/docs/user-guide/skills/bundled/hermes-desktop-plugins/hermes-desktop-plugins-hermes-desktop-plugins) | Escreve plugins do app desktop que adicionam painéis de UI e comandos. | `hermes-desktop-plugins` |

## media

| Skill | Descrição | Caminho |
|-------|-------------|------|
| [`gif-search`](/docs/user-guide/skills/bundled/media/media-gif-search) | Busca/baixa GIFs do Tenor via curl + jq. | `media/gif-search` |
| [`heartmula`](/docs/user-guide/skills/bundled/media/media-heartmula) | HeartMuLa: geração de músicas estilo Suno a partir de letras + tags. | `media/heartmula` |
| [`songsee`](/docs/user-guide/skills/bundled/media/media-songsee) | Espectrogramas/características de áudio (mel, chroma, MFCC) via CLI. | `media/songsee` |
| [`youtube-content`](/docs/user-guide/skills/bundled/media/media-youtube-content) | Transcrições do YouTube para resumos, threads, posts de blog. | `media/youtube-content` |

## mlops

| Skill | Descrição | Caminho |
|-------|-------------|------|
| [`audiocraft-audio-generation`](/docs/user-guide/skills/bundled/mlops/mlops-models-audiocraft) | AudioCraft: MusicGen texto-para-música, AudioGen texto-para-som. | `mlops/models/audiocraft` |
| [`huggingface-hub`](/docs/user-guide/skills/bundled/mlops/mlops-huggingface-hub) | CLI hf do HuggingFace: buscar/baixar/enviar modelos, datasets. | `mlops/huggingface-hub` |
| [`llama-cpp`](/docs/user-guide/skills/bundled/mlops/mlops-inference-llama-cpp) | Inferência GGUF local com llama.cpp + descoberta de modelos no HF Hub. | `mlops/inference/llama-cpp` |
| [`evaluating-llms-harness`](/docs/user-guide/skills/bundled/mlops/mlops-evaluation-lm-evaluation-harness) | lm-eval-harness: benchmark de LLMs (MMLU, GSM8K, etc.). | `mlops/evaluation/lm-evaluation-harness` |
| [`segment-anything-model`](/docs/user-guide/skills/bundled/mlops/mlops-models-segment-anything) | SAM: segmentação de imagem zero-shot via pontos, caixas, máscaras. | `mlops/models/segment-anything` |
| [`serving-llms-vllm`](/docs/user-guide/skills/bundled/mlops/mlops-inference-vllm) | vLLM: serving de LLM de alto throughput, API OpenAI, quantização. | `mlops/inference/vllm` |
| [`weights-and-biases`](/docs/user-guide/skills/bundled/mlops/mlops-evaluation-weights-and-biases) | W&B: registra experimentos de ML, sweeps, model registry, dashboards. | `mlops/evaluation/weights-and-biases` |

## note-taking

| Skill | Descrição | Caminho |
|-------|-------------|------|
| [`obsidian`](/docs/user-guide/skills/bundled/note-taking/note-taking-obsidian) | Lê, busca, cria e edita notas no vault do Obsidian. | `note-taking/obsidian` |

## productivity

| Skill | Descrição | Caminho |
|-------|-------------|------|
| [`airtable`](/docs/user-guide/skills/bundled/productivity/productivity-airtable) | API REST do Airtable via curl. CRUD de registros, filtros, upserts. | `productivity/airtable` |
| [`docx`](/docs/user-guide/skills/bundled/productivity/productivity-docx) | Cria, lê, edita documentos e templates do Word .docx. | `productivity/docx` |
| [`google-workspace`](/docs/user-guide/skills/bundled/productivity/productivity-google-workspace) | Gmail, Calendar, Drive, Docs, Sheets via CLI gws ou Python. | `productivity/google-workspace` |
| [`maps`](/docs/user-guide/skills/bundled/productivity/productivity-maps) | Geocodificação, POIs, rotas, timezones via OpenStreetMap/OSRM. | `productivity/maps` |
| [`nano-pdf`](/docs/user-guide/skills/bundled/productivity/productivity-nano-pdf) | Edita texto/erros de digitação/títulos de PDF via CLI nano-pdf (prompts em linguagem natural). | `productivity/nano-pdf` |
| [`notion`](/docs/user-guide/skills/bundled/productivity/productivity-notion) | API do Notion + CLI ntn: páginas, bases de dados, markdown, Workers. | `productivity/notion` |
| [`ocr-and-documents`](/docs/user-guide/skills/bundled/productivity/productivity-ocr-and-documents) | Extrai texto de PDFs/digitalizações (pymupdf, marker-pdf). | `productivity/ocr-and-documents` |
| [`pdf`](/docs/user-guide/skills/bundled/productivity/productivity-pdf) | Cria, une, divide, preenche e protege arquivos PDF. | `productivity/pdf` |
| [`petdex`](/docs/user-guide/skills/bundled/productivity/productivity-petdex) | Instala e seleciona mascotes petdex animados para o Hermes. | `productivity/petdex` |
| [`powerpoint`](/docs/user-guide/skills/bundled/productivity/productivity-powerpoint) | Cria, lê, edita apresentações .pptx, slides, notas, templates. | `productivity/powerpoint` |
| [`teams-meeting-pipeline`](/docs/user-guide/skills/bundled/productivity/productivity-teams-meeting-pipeline) | Opera o pipeline de resumo de reuniões do Teams via CLI do Hermes — resumir reuniões, inspecionar o status do pipeline, reexecutar jobs, gerenciar assinaturas do Microsoft Graph. | `productivity/teams-meeting-pipeline` |
| [`xlsx`](/docs/user-guide/skills/bundled/productivity/productivity-xlsx) | Cria, lê, edita planilhas Excel .xlsx e CSVs. | `productivity/xlsx` |

## research

| Skill | Descrição | Caminho |
|-------|-------------|------|
| [`arxiv`](/docs/user-guide/skills/bundled/research/research-arxiv) | Busca artigos no arXiv por palavra-chave, autor, categoria ou ID. | `research/arxiv` |
| [`blogwatcher`](/docs/user-guide/skills/bundled/research/research-blogwatcher) | Monitora blogs e feeds RSS/Atom via a ferramenta blogwatcher-cli. | `research/blogwatcher` |
| [`llm-wiki`](/docs/user-guide/skills/bundled/research/research-llm-wiki) | LLM Wiki do Karpathy: constrói/consulta uma base de conhecimento markdown interligada. | `research/llm-wiki` |
| [`polymarket`](/docs/user-guide/skills/bundled/research/research-polymarket) | Consulta o Polymarket: mercados, preços, livros de ordens, histórico. | `research/polymarket` |
| [`research-paper-writing`](/docs/user-guide/skills/bundled/research/research-research-paper-writing) | Escreve artigos de ML para NeurIPS/ICML/ICLR: do design à submissão. | `research/research-paper-writing` |

## smart-home

| Skill | Descrição | Caminho |
|-------|-------------|------|
| [`openhue`](/docs/user-guide/skills/bundled/smart-home/smart-home-openhue) | Controla luzes, cenas e cômodos Philips Hue via CLI OpenHue. | `smart-home/openhue` |

## social-media

| Skill | Descrição | Caminho |
|-------|-------------|------|
| [`xurl`](/docs/user-guide/skills/bundled/social-media/social-media-xurl) | X/Twitter via CLI xurl: postar, buscar, DM, mídia, API v2. | `social-media/xurl` |

## software-development

| Skill | Descrição | Caminho |
|-------|-------------|------|
| [`hermes-agent-skill-authoring`](/docs/user-guide/skills/bundled/software-development/software-development-hermes-agent-skill-authoring) | Cria SKILL.md no repositório: frontmatter, validador, estrutura e princípios de qualidade de escrita. | `software-development/hermes-agent-skill-authoring` |
| [`node-inspect-debugger`](/docs/user-guide/skills/bundled/software-development/software-development-node-inspect-debugger) | Depura Node.js via --inspect + CLI do Chrome DevTools Protocol. | `software-development/node-inspect-debugger` |
| [`plan`](/docs/user-guide/skills/bundled/software-development/software-development-plan) | Modo plano: escreve um plano markdown acionável em .hermes/plans/, sem execução. Tarefas pequenas, caminhos exatos, código completo. | `software-development/plan` |
| [`python-debugpy`](/docs/user-guide/skills/bundled/software-development/software-development-python-debugpy) | Depura Python: REPL pdb + debugpy remoto (DAP). | `software-development/python-debugpy` |
| [`requesting-code-review`](/docs/user-guide/skills/bundled/software-development/software-development-requesting-code-review) | Revisão pré-commit: varredura de segurança, gates de qualidade, auto-correção. | `software-development/requesting-code-review` |
| [`simplify-code`](/docs/user-guide/skills/bundled/software-development/software-development-simplify-code) | Limpeza paralela com 3 agentes de mudanças recentes no código. | `software-development/simplify-code` |
| [`spike`](/docs/user-guide/skills/bundled/software-development/software-development-spike) | Experimentos descartáveis para validar uma ideia antes de construir. | `software-development/spike` |
| [`systematic-debugging`](/docs/user-guide/skills/bundled/software-development/software-development-systematic-debugging) | Depuração de causa raiz em 4 fases: entender bugs antes de corrigi-los. | `software-development/systematic-debugging` |
| [`test-driven-development`](/docs/user-guide/skills/bundled/software-development/software-development-test-driven-development) | TDD: impõe RED-GREEN-REFACTOR, testes antes do código. | `software-development/test-driven-development` |

## yuanbao

| Skill | Descrição | Caminho |
|-------|-------------|------|
| [`yuanbao`](/docs/user-guide/skills/bundled/yuanbao/yuanbao-yuanbao) | Grupos do Yuanbao (元宝): menciona `@` usuários, consulta informações/membros. | `yuanbao` |
