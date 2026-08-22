---
sidebar_position: 9
title: "Catálogo de Skills Opcionais"
description: "Skills opcionais oficiais que acompanham o hermes-agent — instale via hermes skills install official/<category>/<skill>"
---

# Catálogo de Skills Opcionais

Skills opcionais acompanham o hermes-agent em `optional-skills/`, mas **não estão ativas por padrão**. Instale-as explicitamente:

```bash
hermes skills install official/<category>/<skill>
```

Por exemplo:

```bash
hermes skills install official/blockchain/solana
hermes skills install official/mlops/flash-attention
```

Cada skill abaixo tem um link para uma página dedicada com sua definição completa, configuração e uso.

Para desinstalar:

```bash
hermes skills uninstall <skill-name>
```

## autonomous-ai-agents

| Skill | Descrição |
|-------|-------------|
| [**antigravity-cli**](/docs/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-antigravity-cli) | Opera o CLI do Antigravity (agy): plugins, autenticação, sandbox. |
| [**blackbox**](/docs/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-blackbox) | Delega tarefas de programação ao agente CLI Blackbox AI. Agente multi-modelo com um juiz embutido que executa tarefas em vários LLMs e escolhe o melhor resultado. Requer o CLI blackbox e uma chave de API do Blackbox AI. |
| [**grok**](/docs/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-grok) | Delega programação para o CLI xAI Grok Build (features, PRs). |
| [**honcho**](/docs/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-honcho) | Configura e depura a memória Honcho para o Hermes. |
| [**openhands**](/docs/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-openhands) | Delega programação para o CLI OpenHands (agnóstico de modelo, LiteLLM). |

## blockchain

| Skill | Descrição |
|-------|-------------|
| [**evm**](/docs/user-guide/skills/optional/blockchain/blockchain-evm) | Cliente EVM somente leitura: carteiras, tokens, gas em 8 chains. |
| [**hyperliquid**](/docs/user-guide/skills/optional/blockchain/blockchain-hyperliquid) | Dados de mercado, histórico de contas e revisão de trades da Hyperliquid. |
| [**solana**](/docs/user-guide/skills/optional/blockchain/blockchain-solana) | Consulta dados da blockchain Solana com precificação em USD — saldos de carteira, portfólios de tokens com valores, detalhes de transações, NFTs, detecção de whales e estatísticas de rede em tempo real. Usa Solana RPC + CoinGecko. Sem necessidade de chave de API. |

## communication

| Skill | Descrição |
|-------|-------------|
| [**one-three-one-rule**](/docs/user-guide/skills/optional/communication/communication-one-three-one-rule) | Briefs de decisão 1-3-1: problema, três opções, uma escolha. |

## creative

| Skill | Descrição |
|-------|-------------|
| [**audiocraft-audio-generation**](/docs/user-guide/skills/optional/creative/creative-audiocraft-audio-generation) | AudioCraft: MusicGen texto-para-música, AudioGen texto-para-som. |
| [**baoyu-article-illustrator**](/docs/user-guide/skills/optional/creative/creative-baoyu-article-illustrator) | Ilustrações de artigos: consistência de tipo × estilo × paleta. |
| [**baoyu-comic**](/docs/user-guide/skills/optional/creative/creative-baoyu-comic) | Quadrinhos de conhecimento (知识漫画): educacionais, biográficos, tutoriais. |
| [**concept-diagrams**](/docs/user-guide/skills/optional/creative/creative-concept-diagrams) | Gera visuais SVG educacionais planos e minimalistas como HTML. |
| [**creative-ideation**](/docs/user-guide/skills/optional/creative/creative-creative-ideation) | Gera ideias via métodos nomeados da prática criativa. |
| [**draw-your-font**](/docs/user-guide/skills/optional/creative/creative-draw-your-font) | Transforma uma foto de letra manuscrita em uma fonte instalável (TTF/WOFF). |
| [**heartmula**](/docs/user-guide/skills/optional/creative/creative-heartmula) | HeartMuLa: geração de músicas estilo Suno a partir de letras + tags. |
| [**hyperframes**](/docs/user-guide/skills/optional/creative/creative-hyperframes) | Renderiza vídeos MP4/WebM a partir de composições HTML. |
| [**kanban-video-orchestrator**](/docs/user-guide/skills/optional/creative/creative-kanban-video-orchestrator) | Planeja e executa pipelines de produção de vídeo multiagente. |
| [**meme-generation**](/docs/user-guide/skills/optional/creative/creative-meme-generation) | Gera imagens de meme reais escolhendo um template e sobrepondo texto com Pillow. Produz arquivos .png de meme reais. |
| [**pixel-art**](/docs/user-guide/skills/optional/creative/creative-pixel-art) | Pixel art com paletas de época (NES, Game Boy, PICO-8). |
| [**simple-english**](/docs/user-guide/skills/optional/creative/creative-simple-english) | Reescreve texto técnico para ASD-STE100 Simplified Technical English. |
| [**social-media-content-calendar**](/docs/user-guide/skills/optional/creative/creative-social-media-content-calendar) | Planeja campanhas sociais multiplataforma: do brief à publicação. |
| [**tldraw-offline**](/docs/user-guide/skills/optional/creative/creative-tldraw-offline) | Controla e cria scripts em canvases offline do tldraw com um agente. |
| [**unreal-mcp**](/docs/user-guide/skills/optional/creative/creative-unreal-mcp) | Automatiza cenas, atores e renders do editor Unreal Engine. |

## data-science

| Skill | Descrição |
|-------|-------------|
| [**jupyter-notebook**](/docs/user-guide/skills/optional/data-science/data-science-jupyter-notebook) | Python iterativo via kernel Jupyter ao vivo (hamelnb). |

## devops

| Skill | Descrição |
|-------|-------------|
| [**actual-setup**](/docs/user-guide/skills/optional/devops/devops-actual-setup) | Configura inferência Actual Computer (actual.inc) no Hermes. |
| [**docker-management**](/docs/user-guide/skills/optional/devops/devops-docker-management) | Gerencia containers, imagens, volumes e Compose do Docker. |
| [**hermes-s6-container-supervision**](/docs/user-guide/skills/optional/devops/devops-hermes-s6-container-supervision) | Modifica ou depura serviços s6 na imagem Docker do Hermes. |
| [**inference-sh-cli**](/docs/user-guide/skills/optional/devops/devops-inference-sh-cli) | Executa 150+ apps de IA (imagem, vídeo, LLM) via CLI inference.sh. |
| [**pinggy-tunnel**](/docs/user-guide/skills/optional/devops/devops-pinggy-tunnel) | Túneis localhost sem instalação via SSH usando Pinggy. |
| [**watchers**](/docs/user-guide/skills/optional/devops/devops-watchers) | Monitora RSS, APIs JSON e GitHub com deduplicação por watermark. |

## dogfood

| Skill | Descrição |
|-------|-------------|
| [**adversarial-ux-test**](/docs/user-guide/skills/optional/dogfood/dogfood-adversarial-ux-test) | Interpreta um usuário hostil para achar e triar dores de UX. |

## email

| Skill | Descrição |
|-------|-------------|
| [**agentmail**](/docs/user-guide/skills/optional/email/email-agentmail) | Dá ao agente sua própria caixa de e-mail dedicada via AgentMail. Envia, recebe e gerencia e-mail autonomamente usando endereços de e-mail próprios do agente (ex.: hermes-agent@agentmail.to). |

## finance

| Skill | Descrição |
|-------|-------------|
| [**3-statement-model**](/docs/user-guide/skills/optional/finance/finance-3-statement-model) | Constrói modelos de 3 demonstrações totalmente integrados (DRE, BP, DFC) no Excel com cronogramas de capital de giro, roll-forwards de D&A, cronograma de dívida e os plugs que fazem o caixa e os lucros retidos fecharem. Combina com excel-author. |
| [**comps-analysis**](/docs/user-guide/skills/optional/finance/finance-comps-analysis) | Constrói análise de empresas comparáveis no Excel — métricas operacionais, múltiplos de valuation, benchmarking estatístico contra grupos de pares. Combina com excel-author. Use para valuation de empresas públicas, precificação de IPO, benchmarking setorial ou detecção de outliers. |
| [**dcf-model**](/docs/user-guide/skills/optional/finance/finance-dcf-model) | Constrói modelos de valuation DCF de qualidade institucional no Excel — projeções de receita, construção de FCF, WACC, valor terminal, cenários Bear/Base/Bull, tabelas de sensibilidade 5x5. Combina com excel-author. Use para análise de equity por valor intrínseco. |
| [**excel-author**](/docs/user-guide/skills/optional/finance/finance-excel-author) | Constrói planilhas Excel auditáveis sem interface com openpyxl — convenções de células azul/preto/verde, fórmulas em vez de valores fixos, intervalos nomeados, verificações de balanço, tabelas de sensibilidade. Use para modelos financeiros, saídas de auditoria, reconciliações. |
| [**lbo-model**](/docs/user-guide/skills/optional/finance/finance-lbo-model) | Constrói modelos de leveraged buyout no Excel — fontes & usos, cronograma de dívida, cash sweep, múltiplo de saída, sensibilidade de IRR/MOIC. Combina com excel-author. Use para triagem de PE, valuation de sponsor-case ou LBO ilustrativo em um pitch. |
| [**merger-model**](/docs/user-guide/skills/optional/finance/finance-merger-model) | Constrói modelos de acréscimo/diluição (fusão) no Excel — P&L pro-forma, sinergias, mix de financiamento, impacto no EPS. Combina com excel-author. Use para pitches de M&A, materiais de board ou avaliação de negócios. |
| [**polymarket**](/docs/user-guide/skills/optional/finance/finance-polymarket) | Consulta o Polymarket: mercados, preços, livros de ordens, histórico. |
| [**pptx-author**](/docs/user-guide/skills/optional/finance/finance-pptx-author) | Constrói apresentações do PowerPoint sem interface com python-pptx. Combina com excel-author para apresentações baseadas em modelo, onde cada número remete a uma célula da planilha. Use para pitch decks, memorandos de IC, notas de resultados. |
| [**stocks**](/docs/user-guide/skills/optional/finance/finance-stocks) | Cotações de ações, histórico, busca, comparação, criptomoedas via Yahoo. |

## gaming

| Skill | Descrição |
|-------|-------------|
| [**minecraft-modpack-server**](/docs/user-guide/skills/optional/gaming/gaming-minecraft-modpack-server) | Hospeda servidores de Minecraft modificados (CurseForge, Modrinth). |
| [**pokemon-player**](/docs/user-guide/skills/optional/gaming/gaming-pokemon-player) | Joga Pokemon via emulador headless + leituras de RAM. |

## health

| Skill | Descrição |
|-------|-------------|
| [**fitness-nutrition**](/docs/user-guide/skills/optional/health/health-fitness-nutrition) | Planejamento de treino, macros e métricas corporais via wger/USDA. |
| [**neuroskill-bci**](/docs/user-guide/skills/optional/health/health-neuroskill-bci) | Usa estado cognitivo e de humor BCI ao vivo do NeuroSkill. |

## mcp

| Skill | Descrição |
|-------|-------------|
| [**fastmcp**](/docs/user-guide/skills/optional/mcp/mcp-fastmcp) | Constrói, testa e implanta servidores MCP em Python. |
| [**mcp-oauth-remote-gateway**](/docs/user-guide/skills/optional/mcp/mcp-mcp-oauth-remote-gateway) | OAuth manual para servidores MCP remotos em gateways headless. |
| [**mcporter**](/docs/user-guide/skills/optional/mcp/mcp-mcporter) | Usa o CLI mcporter para listar, configurar, autenticar e chamar servidores/ferramentas MCP diretamente (HTTP ou stdio), incluindo servidores ad-hoc, edições de configuração e geração de CLI/tipos. |

## migration

| Skill | Descrição |
|-------|-------------|
| [**openclaw-migration**](/docs/user-guide/skills/optional/migration/migration-openclaw-migration) | Importa um setup OpenClaw (memórias, skills) para o Hermes. |

## mlops

| Skill | Descrição |
|-------|-------------|
| [**accelerate**](/docs/user-guide/skills/optional/mlops/mlops-accelerate) | Treina PyTorch em múltiplas GPUs com mudanças mínimas. |
| [**axolotl**](/docs/user-guide/skills/optional/mlops/mlops-training-axolotl) | Axolotl: fine-tuning de LLM via YAML (LoRA, DPO, GRPO). |
| [**chroma**](/docs/user-guide/skills/optional/mlops/mlops-chroma) | Banco de embeddings para RAG e busca semântica. |
| [**clip**](/docs/user-guide/skills/optional/mlops/mlops-clip) | Classificação de imagens zero-shot e busca imagem-texto. |
| [**dspy**](/docs/user-guide/skills/optional/mlops/mlops-research-dspy) | DSPy: programas declarativos de LM, otimização de prompts, RAG. |
| [**faiss**](/docs/user-guide/skills/optional/mlops/mlops-faiss) | Busca rápida de similaridade vetorial em escala de bilhões. |
| [**flash-attention**](/docs/user-guide/skills/optional/mlops/mlops-flash-attention) | Acelera treino e inferência de transformers de sequência longa. |
| [**guidance**](/docs/user-guide/skills/optional/mlops/mlops-guidance) | Restringe saída de LLM com grammars; garante JSON válido. |
| [**huggingface-tokenizers**](/docs/user-guide/skills/optional/mlops/mlops-huggingface-tokenizers) | Tokenização BPE/WordPiece rápida e treino de vocabulário customizado. |
| [**instructor**](/docs/user-guide/skills/optional/mlops/mlops-instructor) | Saídas estruturadas de LLM validadas com Pydantic. |
| [**lambda-labs**](/docs/user-guide/skills/optional/mlops/mlops-lambda-labs) | Instâncias de nuvem GPU sob demanda para treino de ML. |
| [**llava**](/docs/user-guide/skills/optional/mlops/mlops-llava) | Chat visão-linguagem: VQA, legendas, diálogo com imagem. |
| [**modal**](/docs/user-guide/skills/optional/mlops/mlops-modal) | Nuvem GPU serverless para jobs de ML e APIs de modelo. |
| [**nemo-curator**](/docs/user-guide/skills/optional/mlops/mlops-nemo-curator) | Curadoria de dados de treino de LLM: dedupe, filtro, redação de PII. |
| [**obliteratus**](/docs/user-guide/skills/optional/mlops/mlops-obliteratus) | OBLITERATUS: remove recusas de LLM por abliteração (diff-in-means). |
| [**outlines**](/docs/user-guide/skills/optional/mlops/mlops-inference-outlines) | Outlines: geração estruturada de LLM em JSON/regex/Pydantic. |
| [**peft**](/docs/user-guide/skills/optional/mlops/mlops-peft) | Fine-tune de LLMs grandes com LoRA em GPU limitada. |
| [**pinecone**](/docs/user-guide/skills/optional/mlops/mlops-pinecone) | DB vetorial gerenciado para RAG e busca em produção. |
| [**pytorch-fsdp**](/docs/user-guide/skills/optional/mlops/mlops-pytorch-fsdp) | Treino fully sharded data-parallel para modelos grandes. |
| [**pytorch-lightning**](/docs/user-guide/skills/optional/mlops/mlops-pytorch-lightning) | Loops de treino limpos com suporte distribuído embutido. |
| [**qdrant**](/docs/user-guide/skills/optional/mlops/mlops-qdrant) | Motor de busca vetorial para sistemas RAG de produção. |
| [**saelens**](/docs/user-guide/skills/optional/mlops/mlops-saelens) | Treina sparse autoencoders para interpretar features do modelo. |
| [**segment-anything-model**](/docs/user-guide/skills/optional/mlops/mlops-models-segment-anything-model) | SAM: segmentação zero-shot via pontos, caixas e máscaras. |
| [**simpo**](/docs/user-guide/skills/optional/mlops/mlops-simpo) | Alinhamento de preferência sem referência, mais simples que DPO. |
| [**slime**](/docs/user-guide/skills/optional/mlops/mlops-slime) | Pós-treino RL para LLMs com Megatron e SGLang. |
| [**stable-diffusion**](/docs/user-guide/skills/optional/mlops/mlops-stable-diffusion) | Geração texto-para-imagem, inpainting e img2img. |
| [**tensorrt-llm**](/docs/user-guide/skills/optional/mlops/mlops-tensorrt-llm) | Inferência de LLM de alto throughput em GPUs NVIDIA. |
| [**torchtitan**](/docs/user-guide/skills/optional/mlops/mlops-torchtitan) | Pré-treina LLMs em escala com paralelismo 4D do PyTorch. |
| [**trl-fine-tuning**](/docs/user-guide/skills/optional/mlops/mlops-training-trl-fine-tuning) | TRL: SFT, DPO, GRPO, RLOO reward modeling para RLHF de LLM. |
| [**unsloth**](/docs/user-guide/skills/optional/mlops/mlops-training-unsloth) | Unsloth: fine-tuning LoRA/QLoRA 2-5x mais rápido, menos VRAM. |
| [**whisper**](/docs/user-guide/skills/optional/mlops/mlops-whisper) | Transcreve e traduz fala em 99 idiomas. |

## payments

| Skill | Descrição |
|-------|-------------|
| [**mpp-agent**](/docs/user-guide/skills/optional/payments/payments-mpp-agent) | Paga APIs HTTP 402 via Machine Payments Protocol (MPP). |
| [**stripe-link-cli**](/docs/user-guide/skills/optional/payments/payments-stripe-link-cli) | Pagamentos de agente via Stripe Link — cartões, SPT, aprovações. |
| [**stripe-projects**](/docs/user-guide/skills/optional/payments/payments-stripe-projects) | Provisiona serviços SaaS + sincroniza credenciais via Stripe Projects. |

## productivity

| Skill | Descrição |
|-------|-------------|
| [**canvas**](/docs/user-guide/skills/optional/productivity/productivity-canvas) | Busca cursos e tarefas do Canvas LMS via token de API. |
| [**here-now**](/docs/user-guide/skills/optional/productivity/productivity-here-now) | Publica sites em &#123;slug&#125;.here.now e armazena arquivos em Drives. |
| [**memento-flashcards**](/docs/user-guide/skills/optional/productivity/productivity-memento-flashcards) | Flashcards por repetição espaçada: criar, revisar, quiz, exportar. |
| [**shop**](/docs/user-guide/skills/optional/productivity/productivity-shop) | Busca em catálogo de loja, checkout, rastreamento de pedidos, devoluções. |
| [**shopify**](/docs/user-guide/skills/optional/productivity/productivity-shopify) | APIs GraphQL Admin & Storefront do Shopify via curl. Produtos, pedidos, clientes, estoque, metafields. |
| [**siyuan**](/docs/user-guide/skills/optional/productivity/productivity-siyuan) | API do SiYuan Note para buscar, ler, criar e gerenciar blocos e documentos em uma base de conhecimento auto-hospedada via curl. |
| [**telephony**](/docs/user-guide/skills/optional/productivity/productivity-telephony) | Dá ao Hermes capacidades telefônicas sem mudanças no core. Provisiona e persiste um número Twilio, envia e recebe SMS/MMS, faz chamadas diretas e realiza chamadas de saída orientadas por IA via Bland.ai ou Vapi. |

## research

| Skill | Descrição |
|-------|-------------|
| [**bioinformatics**](/docs/user-guide/skills/optional/research/research-bioinformatics) | Gateway para 400+ skills de genômica e biologia computacional. |
| [**darwinian-evolver**](/docs/user-guide/skills/optional/research/research-darwinian-evolver) | Evolui prompts/regex/SQL/código com o loop evolutivo da Imbue. |
| [**domain-intel**](/docs/user-guide/skills/optional/research/research-domain-intel) | Reconhecimento passivo de domínio usando a stdlib do Python. Descoberta de subdomínios, inspeção de certificado SSL, consultas WHOIS, registros DNS, verificações de disponibilidade de domínio e análise em massa multi-domínio. Sem necessidade de chaves de API. |
| [**drug-discovery**](/docs/user-guide/skills/optional/research/research-drug-discovery) | Descoberta de fármacos: busca ChEMBL, drug-likeness, interações. |
| [**duckduckgo-search**](/docs/user-guide/skills/optional/research/research-duckduckgo-search) | Busca web gratuita via DuckDuckGo — texto, notícias, imagens, vídeos. Sem necessidade de chave de API. Prefere o CLI `ddgs` quando instalado; use a biblioteca Python DDGS apenas depois de verificar que `ddgs` está disponível no runtime atual. |
| [**gitnexus-explorer**](/docs/user-guide/skills/optional/research/research-gitnexus-explorer) | Indexa uma base de código com GitNexus e serve um grafo de conhecimento interativo via UI web + túnel Cloudflare. |
| [**osint-investigation**](/docs/user-guide/skills/optional/research/research-osint-investigation) | Siga o dinheiro via registros públicos e dados de sanções. |
| [**parallel-cli**](/docs/user-guide/skills/optional/research/research-parallel-cli) | Skill opcional de fornecedor para o CLI Parallel — busca web nativa para agentes, extração, pesquisa profunda, enriquecimento, FindAll e monitoramento. Prefere saída JSON e fluxos não interativos. |
| [**pinecone-research**](/docs/user-guide/skills/optional/research/research-pinecone-research) | RAG de agente e memória de longo prazo com Pinecone. |
| [**qmd**](/docs/user-guide/skills/optional/research/research-qmd) | Busca bases de conhecimento pessoais, notas, documentos e transcrições de reuniões localmente usando qmd — um motor de recuperação híbrido com BM25, busca vetorial e reranking por LLM. Suporta integração via CLI e MCP. |
| [**scrapling**](/docs/user-guide/skills/optional/research/research-scrapling) | Web scraping com Scrapling - fetching HTTP, automação de navegador furtiva, bypass do Cloudflare e crawling spider via CLI e Python. |
| [**searxng-search**](/docs/user-guide/skills/optional/research/research-searxng-search) | Meta-busca gratuita via SearXNG — agrega resultados de mais de 70 motores de busca. Auto-hospedado ou use uma instância pública. Sem necessidade de chave de API. Recai automaticamente quando o toolset de busca web não está disponível. |

## security

| Skill | Descrição |
|-------|-------------|
| [**1password**](/docs/user-guide/skills/optional/security/security-1password) | Configura e usa o CLI do 1Password (op). Use ao instalar o CLI, ativar a integração com o app desktop, fazer login e ler/injetar segredos para comandos. |
| [**godmode**](/docs/user-guide/skills/optional/security/security-godmode) | Jailbreak de LLMs: Parseltongue, GODMODE, ULTRAPLINIAN. |
| [**oss-forensics**](/docs/user-guide/skills/optional/security/security-oss-forensics) | Forense de supply chain no GitHub: recuperação, IOCs, relatórios. |
| [**sherlock**](/docs/user-guide/skills/optional/security/security-sherlock) | Busca OSINT de nome de usuário em mais de 400 redes sociais. Rastreia contas de redes sociais por nome de usuário. |
| [**unbroker**](/docs/user-guide/skills/optional/security/security-unbroker) | Remove autonomamente suas informações de sites de corretores de dados. |
| [**web-pentest**](/docs/user-guide/skills/optional/security/security-web-pentest) | Pentest web autorizado: recon, exploits com prova, relatório. |

## software-development

| Skill | Descrição |
|-------|-------------|
| [**code-wiki**](/docs/user-guide/skills/optional/software-development/software-development-code-wiki) | Gera documentação wiki + diagramas Mermaid para qualquer base de código. |
| [**rest-graphql-debug**](/docs/user-guide/skills/optional/software-development/software-development-rest-graphql-debug) | Depura APIs REST/GraphQL: códigos de status, autenticação, schemas, reprodução. |
| [**subagent-driven-development**](/docs/user-guide/skills/optional/software-development/software-development-subagent-driven-development) | Executa planos via subagentes delegate_task (revisão em 2 etapas). |

## web-development

| Skill | Descrição |
|-------|-------------|
| [**cloudflare-temporary-deploy**](/docs/user-guide/skills/optional/web-development/web-development-cloudflare-temporary-deploy) | Implanta um Worker em produção, sem conta, via wrangler --temporary. |
| [**page-agent**](/docs/user-guide/skills/optional/web-development/web-development-page-agent) | Incorpora um copiloto GUI em linguagem natural em apps web. |

## yuanbao

| Skill | Descrição |
|-------|-------------|
| [**yuanbao**](/docs/user-guide/skills/optional/yuanbao/yuanbao-yuanbao) | Grupos Yuanbao (元宝): @mencionar usuários, consultar info/membros. |

---

## Contributing Optional Skills

To add a new optional skill to the repository:

1. Create a directory under `optional-skills/<category>/<skill-name>/`
2. Add a `SKILL.md` with standard frontmatter (name, description, version, author)
3. Include any supporting files in `references/`, `templates/`, or `scripts/` subdirectories
4. Submit a pull request — the skill will appear in this catalog and get its own docs page once merged
