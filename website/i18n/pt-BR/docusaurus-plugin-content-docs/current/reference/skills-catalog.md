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
| [`computer-use`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-computer-use) | Controla o desktop priorizando background; escala no sinal. | `autonomous-ai-agents/computer-use` |
| [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) | Usa, configura, tema, estende e orquestra o Hermes Agent. | `autonomous-ai-agents/hermes-agent` |
| [`merge-reconciler`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-merge-reconciler) | Resolução neutra de terceiros para conflitos de merge de agentes. | `autonomous-ai-agents/merge-reconciler` |
| [`opencode`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-opencode) | Delega programação para o CLI do OpenCode (features, revisão de PR). | `autonomous-ai-agents/opencode` |


## creative

| Skill | Descrição | Caminho |
|-------|-------------|------|
| [`architecture-diagram`](/docs/user-guide/skills/bundled/creative/creative-architecture-diagram) | Diagramas SVG de arquitetura/nuvem/infra em tema escuro como HTML. | `creative/architecture-diagram` |
| [`ascii-video`](/docs/user-guide/skills/bundled/creative/creative-ascii-video) | Vídeo ASCII: converte vídeo/áudio em MP4/GIF ASCII colorido. | `creative/ascii-video` |
| [`baoyu-infographic`](/docs/user-guide/skills/bundled/creative/creative-baoyu-infographic) | Infográficos: 21 layouts x 21 estilos (信息图, 可视化). | `creative/baoyu-infographic` |
| [`claude-design`](/docs/user-guide/skills/bundled/creative/creative-claude-design) | Cria artefatos HTML avulsos (landing page, apresentação, prototipagem). | `creative/claude-design` |
| [`design-md`](/docs/user-guide/skills/bundled/creative/creative-design-md) | Cria/valida/exporta arquivos de spec de tokens DESIGN.md do Google. | `creative/design-md` |
| [`humanizer`](/docs/user-guide/skills/bundled/creative/creative-humanizer) | Humaniza texto: remove marcas de IA e adiciona voz real. | `creative/humanizer` |
| [`manim-video`](/docs/user-guide/skills/bundled/creative/creative-manim-video) | Animações Manim CE: vídeos de matemática/algoritmos estilo 3Blue1Brown. | `creative/manim-video` |
| [`p5js`](/docs/user-guide/skills/bundled/creative/creative-p5js) | Sketches p5.js: arte generativa, shaders, interatividade, 3D. | `creative/p5js` |
| [`popular-web-designs`](/docs/user-guide/skills/bundled/creative/creative-popular-web-designs) | 54 sistemas de design reais (Stripe, Linear, Vercel) como HTML/CSS. | `creative/popular-web-designs` |
| [`songwriting-and-ai-music`](/docs/user-guide/skills/bundled/creative/creative-songwriting-and-ai-music) | Ofício de composição musical e prompts de música com IA Suno. | `creative/songwriting-and-ai-music` |


## devops

| Skill | Descrição | Caminho |
|-------|-------------|------|
| [`sdlc-review`](/docs/user-guide/skills/bundled/devops/devops-sdlc-review) | Revisa handoffs Kanban e roteia resultados verificados. | `devops/sdlc-review` |


## email

| Skill | Descrição | Caminho |
|-------|-------------|------|
| [`email-inbox-triage`](/docs/user-guide/skills/bundled/email/email-email-inbox-triage) | Faz triagem de inbox: prioriza threads, rascunha respostas com segurança. | `email/email-inbox-triage` |
| [`himalaya`](/docs/user-guide/skills/bundled/email/email-himalaya) | CLI Himalaya: e-mail IMAP/SMTP pelo terminal. | `email/himalaya` |


## media

| Skill | Descrição | Caminho |
|-------|-------------|------|
| [`gif-search`](/docs/user-guide/skills/bundled/media/media-gif-search) | Busca/baixa GIFs do Tenor via curl + jq. | `media/gif-search` |
| [`songsee`](/docs/user-guide/skills/bundled/media/media-songsee) | Espectrogramas/características de áudio (mel, chroma, MFCC) via CLI. | `media/songsee` |
| [`youtube-content`](/docs/user-guide/skills/bundled/media/media-youtube-content) | Transcrições do YouTube para resumos, threads, posts de blog. | `media/youtube-content` |


## note-taking

| Skill | Descrição | Caminho |
|-------|-------------|------|
| [`obsidian`](/docs/user-guide/skills/bundled/note-taking/note-taking-obsidian) | Lê, busca, cria e edita notas no vault do Obsidian. | `note-taking/obsidian` |


## productivity

| Skill | Descrição | Caminho |
|-------|-------------|------|
| [`airtable`](/docs/user-guide/skills/bundled/productivity/productivity-airtable) | API REST do Airtable via curl. CRUD de registros, filtros, upserts. | `productivity/airtable` |
| [`box`](/docs/user-guide/skills/bundled/productivity/productivity-box) | Box gerencia arquivos na nuvem, compartilhamento, busca e metadata. | `productivity/box` |
| [`document-to-action-items`](/docs/user-guide/skills/bundled/productivity/productivity-document-to-action-items) | Extrai obrigações, prazos e tarefas citadas de documentos. | `productivity/document-to-action-items` |
| [`docx`](/docs/user-guide/skills/bundled/productivity/productivity-docx) | Cria, lê, edita e usa templates de arquivos Word .docx. | `productivity/docx` |
| [`google-workspace`](/docs/user-guide/skills/bundled/productivity/productivity-google-workspace) | Gmail, Calendar, Drive, Docs, Sheets via CLI gws ou Python. | `productivity/google-workspace` |
| [`maps`](/docs/user-guide/skills/bundled/productivity/productivity-maps) | Geocodificação, POIs, rotas, timezones via OpenStreetMap/OSRM. | `productivity/maps` |
| [`meeting-action-items`](/docs/user-guide/skills/bundled/productivity/productivity-meeting-action-items) | Transforma notas de reunião em decisões citadas, responsáveis e tickets. | `productivity/meeting-action-items` |
| [`notion`](/docs/user-guide/skills/bundled/productivity/productivity-notion) | API do Notion + CLI ntn: páginas, bases de dados, markdown, Workers. | `productivity/notion` |
| [`pdf`](/docs/user-guide/skills/bundled/productivity/productivity-pdf) | Cria, lê, une, preenche e protege arquivos PDF. | `productivity/pdf` |
| [`powerpoint`](/docs/user-guide/skills/bundled/productivity/productivity-powerpoint) | Cria, lê e edita decks .pptx com python-pptx. | `productivity/powerpoint` |
| [`product-price-monitor`](/docs/user-guide/skills/bundled/productivity/productivity-product-price-monitor) | Monitora preços de produtos, voos ou anúncios; alerta no alvo. | `productivity/product-price-monitor` |
| [`teams-meeting-pipeline`](/docs/user-guide/skills/bundled/productivity/productivity-teams-meeting-pipeline) | Resumos de reuniões do Teams, replay de jobs, assinaturas Graph. | `productivity/teams-meeting-pipeline` |
| [`weekly-review-planning`](/docs/user-guide/skills/bundled/productivity/productivity-weekly-review-planning) | Reset semanal: compromissos, trabalho parado, plano da próxima semana. | `productivity/weekly-review-planning` |
| [`xlsx`](/docs/user-guide/skills/bundled/productivity/productivity-xlsx) | Cria, lê e edita workbooks Excel .xlsx e CSVs. | `productivity/xlsx` |


## research

| Skill | Descrição | Caminho |
|-------|-------------|------|
| [`arxiv`](/docs/user-guide/skills/bundled/research/research-arxiv) | Busca artigos no arXiv por palavra-chave, autor, categoria ou ID. | `research/arxiv` |
| [`competitor-news-monitor`](/docs/user-guide/skills/bundled/research/research-competitor-news-monitor) | Monitora empresas nomeadas por notícias relevantes; digests citados. | `research/competitor-news-monitor` |
| [`grounded-citations`](/docs/user-guide/skills/bundled/research/research-grounded-citations) | Fundamenta respostas e documentos em fontes citadas e verificáveis. | `research/grounded-citations` |
| [`llm-wiki`](/docs/user-guide/skills/bundled/research/research-llm-wiki) | LLM Wiki do Karpathy: constrói/consulta uma base de conhecimento markdown interligada. | `research/llm-wiki` |


## social-media

| Skill | Descrição | Caminho |
|-------|-------------|------|
| [`xurl`](/docs/user-guide/skills/bundled/social-media/social-media-xurl) | X/Twitter via CLI xurl: postar, buscar, DM, mídia, API v2. | `social-media/xurl` |


## software-development

| Skill | Descrição | Caminho |
|-------|-------------|------|
| [`codebase-inspection`](/docs/user-guide/skills/bundled/software-development/software-development-codebase-inspection) | Inspeciona bases de código com pygount: LOC, linguagens, proporções. | `software-development/codebase-inspection` |
| [`dogfood`](/docs/user-guide/skills/bundled/software-development/software-development-dogfood) | QA exploratório de aplicações web: encontrar bugs, evidências, relatórios. | `software-development/dogfood` |
| [`github`](/docs/user-guide/skills/bundled/software-development/software-development-github) | GitHub via gh CLI: PRs, issues, reviews, repos, auth. | `software-development/github` |
| [`hermes-agent-skill-authoring`](/docs/user-guide/skills/bundled/software-development/software-development-hermes-agent-skill-authoring) | Cria SKILL.md no repositório: frontmatter, validador, estrutura e princípios de qualidade de escrita. | `software-development/hermes-agent-skill-authoring` |
| [`inspecting-hermes-desktop-dom`](/docs/user-guide/skills/bundled/software-development/software-development-inspecting-hermes-desktop-dom) | Lê o DOM/CSS live do Hermes desktop via CDP. | `software-development/inspecting-hermes-desktop-dom` |
| [`node-inspect-debugger`](/docs/user-guide/skills/bundled/software-development/software-development-node-inspect-debugger) | Depura Node.js via --inspect + CLI do Chrome DevTools Protocol. | `software-development/node-inspect-debugger` |
| [`python-debugpy`](/docs/user-guide/skills/bundled/software-development/software-development-python-debugpy) | Depura Python: REPL pdb + debugpy remoto (DAP). | `software-development/python-debugpy` |
| [`requesting-code-review`](/docs/user-guide/skills/bundled/software-development/software-development-requesting-code-review) | Revisão pré-commit: varredura de segurança, gates de qualidade, auto-correção. | `software-development/requesting-code-review` |
| [`simplify-code`](/docs/user-guide/skills/bundled/software-development/software-development-simplify-code) | Limpeza paralela com 3 agentes de mudanças recentes no código. | `software-development/simplify-code` |
| [`spike`](/docs/user-guide/skills/bundled/software-development/software-development-spike) | Experimentos descartáveis para validar uma ideia antes de construir. | `software-development/spike` |
| [`systematic-debugging`](/docs/user-guide/skills/bundled/software-development/software-development-systematic-debugging) | Depuração de causa raiz em 4 fases: entender bugs antes de corrigi-los. | `software-development/systematic-debugging` |
| [`test-driven-development`](/docs/user-guide/skills/bundled/software-development/software-development-test-driven-development) | TDD: impõe RED-GREEN-REFACTOR, testes antes do código. | `software-development/test-driven-development` |


## web

| Skill | Descrição | Caminho |
|-------|-------------|------|
| [`blocked-page-recovery`](/docs/user-guide/skills/bundled/web/web-blocked-page-recovery) | Recupera páginas bloqueadas/paywalled/WAF via snapshots de archive e fallbacks de reader. Use quando web_extract ou o browser encontrar páginas 403/429/challenge, paywalls, ou interstitials de bot-detection. | `web/blocked-page-recovery` |
