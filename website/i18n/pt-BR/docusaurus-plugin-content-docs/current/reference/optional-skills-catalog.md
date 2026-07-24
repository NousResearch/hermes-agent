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
| [**honcho**](/docs/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-honcho) | Configura e usa a memória Honcho com o Hermes -- modelagem de usuário entre sessões, isolamento de pares multiperfil, configuração de observação, raciocínio dialético, resumos de sessão e aplicação de orçamento de contexto. Use ao configurar o Honcho, resolver problemas... |
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
| [**one-three-one-rule**](/docs/user-guide/skills/optional/communication/communication-one-three-one-rule) | Framework estruturado de tomada de decisão para propostas técnicas e análise de trade-offs. Quando o usuário enfrenta uma escolha entre várias abordagens (decisões de arquitetura, seleção de ferramentas, estratégias de refatoração, caminhos de migração), esta skill p... |

## creative

| Skill | Descrição |
|-------|-------------|
| [**baoyu-article-illustrator**](/docs/user-guide/skills/optional/creative/creative-baoyu-article-illustrator) | Ilustrações de artigos: consistência de tipo × estilo × paleta. |
| [**baoyu-comic**](/docs/user-guide/skills/optional/creative/creative-baoyu-comic) | Quadrinhos de conhecimento (知识漫画): educacionais, biográficos, tutoriais. |
| [**blender-mcp**](/docs/user-guide/skills/optional/creative/creative-blender-mcp) | Opera o Blender via o MCP blender do catálogo, com receitas bpy. |
| [**concept-diagrams**](/docs/user-guide/skills/optional/creative/creative-concept-diagrams) | Gera diagramas SVG planos e minimalistas, adaptados para modo claro/escuro, como arquivos HTML autônomos, usando uma linguagem visual educacional unificada com 9 rampas de cor semânticas, tipografia em sentence-case e modo escuro automático. Mais adequado para conteúdo educacional e... |
| [**creative-ideation**](/docs/user-guide/skills/optional/creative/creative-creative-ideation) | Gera ideias via métodos nomeados da prática criativa. |
| [**hyperframes**](/docs/user-guide/skills/optional/creative/creative-hyperframes) | Cria composições de vídeo baseadas em HTML, cartões de título animados, overlays sociais, vídeos com legendas de talking-head, visuais reativos a áudio e transições com shaders usando HyperFrames. HTML é a fonte da verdade para vídeo. Use quando o usuário quiser... |
| [**kanban-video-orchestrator**](/docs/user-guide/skills/optional/creative/creative-kanban-video-orchestrator) | Planeja, configura e monitora um pipeline de produção de vídeo multiagente com base no Hermes Kanban. Use quando o usuário quiser fazer QUALQUER vídeo — filme narrativo, produto/marketing, videoclipe, explicativo, arte ASCII/terminal, loop abstrato/generativo... |
| [**meme-generation**](/docs/user-guide/skills/optional/creative/creative-meme-generation) | Gera imagens de meme reais escolhendo um template e sobrepondo texto com Pillow. Produz arquivos .png de meme reais. |
| [**pixel-art**](/docs/user-guide/skills/optional/creative/creative-pixel-art) | Pixel art com paletas de época (NES, Game Boy, PICO-8). |
| [**unreal-mcp**](/docs/user-guide/skills/optional/creative/creative-unreal-mcp) | Use quando o usuário quiser fazer qualquer coisa na Unreal Engine através do servidor MCP oficial embutido no editor da Epic (entrada do catálogo: unreal-engine) — construir/iluminar/popular cenas, posicionar e transformar atores, criar Blueprints, animar com Sequence... |

## devops

| Skill | Descrição |
|-------|-------------|
| [**inference-sh-cli**](/docs/user-guide/skills/optional/devops/devops-cli) | Executa mais de 150 apps de IA via o CLI inference.sh (infsh) — geração de imagem, criação de vídeo, LLMs, busca, 3D, automação social. Usa a ferramenta terminal. Gatilhos: inference.sh, infsh, ai apps, flux, veo, geração de imagem, geração de vídeo, seedrea... |
| [**docker-management**](/docs/user-guide/skills/optional/devops/devops-docker-management) | Gerencia containers, imagens, volumes, redes e stacks Compose do Docker — operações de ciclo de vida, depuração, limpeza e otimização de Dockerfile. |
| [**hermes-s6-container-supervision**](/docs/user-guide/skills/optional/devops/devops-hermes-s6-container-supervision) | Modifica, depura ou estende a árvore de supervisão s6-overlay dentro da imagem Docker do Hermes Agent — adicionando novos serviços, depurando gateways de perfil, entendendo o padrão main-program da Arquitetura B. |
| [**pinggy-tunnel**](/docs/user-guide/skills/optional/devops/devops-pinggy-tunnel) | Túneis localhost sem instalação via SSH usando o Pinggy. |
| [**watchers**](/docs/user-guide/skills/optional/devops/devops-watchers) | Monitora RSS, APIs JSON e GitHub com deduplicação por watermark. |

## dogfood

| Skill | Descrição |
|-------|-------------|
| [**adversarial-ux-test**](/docs/user-guide/skills/optional/dogfood/dogfood-adversarial-ux-test) | Interpreta o usuário mais difícil e resistente à tecnologia para o seu produto. Navega pelo app como essa persona, encontra todo ponto de dor de UX e depois filtra as reclamações por uma camada de pragmatismo para separar problemas reais de ruído. Cria tickets acionáveis... |

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
| [**fitness-nutrition**](/docs/user-guide/skills/optional/health/health-fitness-nutrition) | Planejador de treino de ginástica e rastreador de nutrição. Busca mais de 690 exercícios por músculo, equipamento ou categoria via wger. Consulta macros e calorias de mais de 380.000 alimentos via USDA FoodData Central. Calcula IMC, TDEE, uma repetição máxima, divisão de macros e composição... |
| [**neuroskill-bci**](/docs/user-guide/skills/optional/health/health-neuroskill-bci) | Conecta a uma instância NeuroSkill em execução e incorpora o estado cognitivo e emocional em tempo real do usuário (foco, relaxamento, humor, carga cognitiva, sonolência, frequência cardíaca, HRV, estágios do sono e mais de 40 escores EXG derivados) nas respostas.... |

## mcp

| Skill | Descrição |
|-------|-------------|
| [**fastmcp**](/docs/user-guide/skills/optional/mcp/mcp-fastmcp) | Constrói, testa, inspeciona, instala e implanta servidores MCP com FastMCP em Python. Use ao criar um novo servidor MCP, encapsular uma API ou banco de dados como ferramentas MCP, expor resources ou prompts, ou preparar um servidor FastMCP para o Claude Code, Cur... |
| [**mcp-oauth-remote-gateway**](/docs/user-guide/skills/optional/mcp/mcp-mcp-oauth-remote-gateway) | OAuth manual para servidores MCP remotos em gateways headless. |
| [**mcporter**](/docs/user-guide/skills/optional/mcp/mcp-mcporter) | Usa o CLI mcporter para listar, configurar, autenticar e chamar servidores/ferramentas MCP diretamente (HTTP ou stdio), incluindo servidores ad-hoc, edições de configuração e geração de CLI/tipos. |

## migration

| Skill | Descrição |
|-------|-------------|
| [**openclaw-migration**](/docs/user-guide/skills/optional/migration/migration-openclaw-migration) | Migra a pegada de customização do OpenClaw de um usuário para o Hermes Agent. Importa memórias compatíveis com o Hermes, SOUL.md, listas de permissão de comandos, skills do usuário e ativos selecionados do workspace a partir de ~/.openclaw, depois relata exatamente o que não pôde ser migr... |

## mlops

| Skill | Descrição |
|-------|-------------|
| [**huggingface-accelerate**](/docs/user-guide/skills/optional/mlops/mlops-accelerate) | API de treinamento distribuído mais simples. 4 linhas para adicionar suporte distribuído a qualquer script PyTorch. API unificada para DeepSpeed/FSDP/Megatron/DDP. Posicionamento automático de device, precisão mista (FP16/BF16/FP8). Configuração interativa, comando de lançamento único... |
| [**axolotl**](/docs/user-guide/skills/optional/mlops/mlops-training-axolotl) | Axolotl: fine-tuning de LLM via YAML (LoRA, DPO, GRPO). |
| [**chroma**](/docs/user-guide/skills/optional/mlops/mlops-chroma) | Banco de dados de embeddings de código aberto para aplicações de IA. Armazena embeddings e metadados, realiza busca vetorial e full-text, filtra por metadados. API simples de 4 funções. Escala de notebooks a clusters de produção. Use para busca semântica, RAG... |
| [**clip**](/docs/user-guide/skills/optional/mlops/mlops-clip) | Modelo da OpenAI que conecta visão e linguagem. Permite classificação de imagens zero-shot, correspondência imagem-texto e recuperação cross-modal. Treinado em 400M pares imagem-texto. Use para busca de imagens, moderação de conteúdo ou tarefas de visão-linguagem q... |
| [**dspy**](/docs/user-guide/skills/optional/mlops/mlops-research-dspy) | DSPy: programas declarativos de LM, otimização automática de prompts, RAG. |
| [**faiss**](/docs/user-guide/skills/optional/mlops/mlops-faiss) | Biblioteca do Facebook para busca de similaridade eficiente e clustering de vetores densos. Suporta bilhões de vetores, aceleração por GPU e vários tipos de índice (Flat, IVF, HNSW). Use para busca k-NN rápida, recuperação de vetores em larga escala, ou onde... |
| [**optimizing-attention-flash**](/docs/user-guide/skills/optional/mlops/mlops-flash-attention) | Otimiza a atenção de transformers com Flash Attention para 2-4x de aceleração e 10-20x de redução de memória. Use ao treinar/executar transformers com sequências longas (>512 tokens), ao encontrar problemas de memória de GPU com atenção, ou quando precisar de inferên... mais rápida |
| [**guidance**](/docs/user-guide/skills/optional/mlops/mlops-guidance) | Controla a saída de LLM com regex e grammars, garante geração válida de JSON/XML/código, impõe formatos estruturados e constrói workflows multi-etapa com Guidance - o framework de geração restrita da Microsoft Research |
| [**huggingface-tokenizers**](/docs/user-guide/skills/optional/mlops/mlops-huggingface-tokenizers) | Tokenizadores rápidos otimizados para pesquisa e produção. Implementação em Rust tokeniza 1GB em menos de 20 segundos. Suporta algoritmos BPE, WordPiece e Unigram. Treina vocabulários customizados, rastreia alinhamentos, trata padding/truncamento. Integ... |
| [**instructor**](/docs/user-guide/skills/optional/mlops/mlops-instructor) | Extrai dados estruturados de respostas de LLM com validação Pydantic, tenta novamente extrações falhas automaticamente, faz parsing de JSON complexo com type safety e transmite resultados parciais com Instructor - biblioteca de saída estruturada testada em produção |
| [**lambda-labs-gpu-cloud**](/docs/user-guide/skills/optional/mlops/mlops-lambda-labs) | Instâncias de nuvem GPU reservadas e sob demanda para treinamento e inferência de ML. Use quando precisar de instâncias de GPU dedicadas com acesso SSH simples, sistemas de arquivos persistentes, ou clusters multi-node de alto desempenho para treinamento em larga escala. |
| [**llava**](/docs/user-guide/skills/optional/mlops/mlops-llava) | Assistente de Linguagem e Visão de Grande Escala. Permite ajuste de instrução visual e conversas baseadas em imagem. Combina o encoder de visão CLIP com modelos de linguagem Vicuna/LLaMA. Suporta chat de imagem multi-turno, resposta a perguntas visuais e instru... |
| [**modal-serverless-gpu**](/docs/user-guide/skills/optional/mlops/mlops-modal) | Plataforma de nuvem GPU serverless para executar workloads de ML. Use quando precisar de acesso a GPU sob demanda sem gerenciamento de infraestrutura, implantando modelos de ML como APIs, ou executando jobs em batch com escalonamento automático. |
| [**nemo-curator**](/docs/user-guide/skills/optional/mlops/mlops-nemo-curator) | Curadoria de dados acelerada por GPU para treinamento de LLM. Suporta texto/imagem/vídeo/áudio. Recursos: deduplicação fuzzy (16x mais rápida), filtragem de qualidade (mais de 30 heurísticas), deduplicação semântica, redação de PII, detecção de NSFW. Escala entre GPUs com... |
| [**obliteratus**](/docs/user-guide/skills/optional/mlops/mlops-obliteratus) | OBLITERATUS: remove recusas de LLM por abliteração (diff-in-means). |
| [**outlines**](/docs/user-guide/skills/optional/mlops/mlops-inference-outlines) | Outlines: geração estruturada de LLM em JSON/regex/Pydantic. |
| [**peft-fine-tuning**](/docs/user-guide/skills/optional/mlops/mlops-peft) | Fine-tuning eficiente em parâmetros para LLMs usando LoRA, QLoRA e mais de 25 métodos. Use ao fazer fine-tuning de modelos grandes (7B-70B) com memória de GPU limitada, quando precisar treinar menos de 1% dos parâmetros com perda mínima de acurácia, ou para configurações multi-adaptador... |
| [**pinecone**](/docs/user-guide/skills/optional/mlops/mlops-pinecone) | Banco de dados vetorial gerenciado para aplicações de IA em produção. Totalmente gerenciado, com auto-escalonamento, busca híbrida (densa + esparsa), filtragem por metadados e namespaces. Baixa latência (menos de 100ms p95). Use para RAG em produção, sistemas de recomendação, ou se... |
| [**pytorch-fsdp**](/docs/user-guide/skills/optional/mlops/mlops-pytorch-fsdp) | Orientação especializada para treinamento Fully Sharded Data Parallel com PyTorch FSDP - sharding de parâmetros, precisão mista, offloading de CPU, FSDP2 |
| [**pytorch-lightning**](/docs/user-guide/skills/optional/mlops/mlops-pytorch-lightning) | Framework PyTorch de alto nível com a classe Trainer, treinamento distribuído automático (DDP/FSDP/DeepSpeed), sistema de callbacks e boilerplate mínimo. Escala do laptop ao supercomputador com o mesmo código. Use quando quiser loops de treinamento limpos... |
| [**qdrant-vector-search**](/docs/user-guide/skills/optional/mlops/mlops-qdrant) | Motor de busca por similaridade vetorial de alto desempenho para RAG e busca semântica. Use ao construir sistemas RAG de produção que exigem busca rápida de vizinhos mais próximos, busca híbrida com filtragem, ou armazenamento vetorial escalável com desempenho movido a Rust... |
| [**sparse-autoencoder-training**](/docs/user-guide/skills/optional/mlops/mlops-saelens) | Fornece orientação para treinar e analisar Sparse Autoencoders (SAEs) usando SAELens para decompor ativações de redes neurais em features interpretáveis. Use ao descobrir features interpretáveis, analisar superposição ou estudar... |
| [**simpo-training**](/docs/user-guide/skills/optional/mlops/mlops-simpo) | Otimização de Preferência Simples para alinhamento de LLM. Alternativa sem modelo de referência ao DPO com melhor desempenho (+6.4 pontos no AlpacaEval 2.0). Não precisa de modelo de referência, mais eficiente que o DPO. Use para alinhamento de preferência quando quiser simpl... |
| [**slime-rl-training**](/docs/user-guide/skills/optional/mlops/mlops-slime) | Fornece orientação para pós-treinamento de LLM com RL usando slime, um framework Megatron+SGLang. Use ao treinar modelos GLM, implementar workflows customizados de geração de dados, ou precisar de integração estreita com Megatron-LM para escalonamento de RL. |
| [**stable-diffusion-image-generation**](/docs/user-guide/skills/optional/mlops/mlops-stable-diffusion) | Geração de texto-para-imagem de última geração com modelos Stable Diffusion via HuggingFace Diffusers. Use ao gerar imagens a partir de prompts de texto, realizar tradução imagem-para-imagem, inpainting, ou construir pipelines de difusão customizados. |
| [**tensorrt-llm**](/docs/user-guide/skills/optional/mlops/mlops-tensorrt-llm) | Otimiza inferência de LLM com NVIDIA TensorRT para máximo throughput e menor latência. Use para implantação em produção em GPUs NVIDIA (A100/H100), quando precisar de inferência 10-100x mais rápida que PyTorch, ou para servir modelos com quantizaçã... |
| [**distributed-llm-pretraining-torchtitan**](/docs/user-guide/skills/optional/mlops/mlops-torchtitan) | Fornece pré-treinamento distribuído de LLM nativo em PyTorch usando torchtitan com paralelismo 4D (FSDP2, TP, PP, CP). Use ao pré-treinar Llama 3.1, DeepSeek V3, ou modelos customizados em escala de 8 a mais de 512 GPUs com Float8, torch.compile, e dist... |
| [**fine-tuning-with-trl**](/docs/user-guide/skills/optional/mlops/mlops-training-trl-fine-tuning) | TRL: SFT, DPO, PPO, GRPO, reward modeling para RLHF de LLM. |
| [**unsloth**](/docs/user-guide/skills/optional/mlops/mlops-training-unsloth) | Unsloth: fine-tuning LoRA/QLoRA 2-5x mais rápido, menos VRAM. |
| [**whisper**](/docs/user-guide/skills/optional/mlops/mlops-whisper) | Modelo de reconhecimento de fala de propósito geral da OpenAI. Suporta 99 idiomas, transcrição, tradução para o inglês e identificação de idioma. Seis tamanhos de modelo, de tiny (39M parâmetros) a large (1550M parâmetros). Use para fala-para-texto, podcast... |

## payments

| Skill | Descrição |
|-------|-------------|
| [**mpp-agent**](/docs/user-guide/skills/optional/payments/payments-mpp-agent) | Paga APIs HTTP 402 via Machine Payments Protocol (MPP). |
| [**stripe-link-cli**](/docs/user-guide/skills/optional/payments/payments-stripe-link-cli) | Pagamentos de agente via Stripe Link — cartões, SPT, aprovações. |
| [**stripe-projects**](/docs/user-guide/skills/optional/payments/payments-stripe-projects) | Provisiona serviços SaaS + sincroniza credenciais via Stripe Projects. |

## productivity

| Skill | Descrição |
|-------|-------------|
| [**canvas**](/docs/user-guide/skills/optional/productivity/productivity-canvas) | Integração com o Canvas LMS — busca cursos matriculados e tarefas usando autenticação por token de API. |
| [**here.now**](/docs/user-guide/skills/optional/productivity/productivity-here-now) | Publica sites estáticos em &#123;slug&#125;.here.now e armazena arquivos privados em Drives na nuvem para handoff agente-para-agente. |
| [**memento-flashcards**](/docs/user-guide/skills/optional/productivity/productivity-memento-flashcards) | Sistema de flashcards por repetição espaçada. Cria cartões a partir de fatos ou texto, conversa com flashcards usando respostas em texto livre avaliadas pelo agente, gera quizzes a partir de transcrições do YouTube, revisa cartões pendentes com agendamento adaptativo, e exporta/impor... |
| [**shop**](/docs/user-guide/skills/optional/productivity/productivity-shop) | Busca em catálogo de loja, checkout, rastreamento de pedidos, devoluções. |
| [**shopify**](/docs/user-guide/skills/optional/productivity/productivity-shopify) | APIs GraphQL Admin & Storefront do Shopify via curl. Produtos, pedidos, clientes, estoque, metafields. |
| [**siyuan**](/docs/user-guide/skills/optional/productivity/productivity-siyuan) | API do SiYuan Note para buscar, ler, criar e gerenciar blocos e documentos em uma base de conhecimento auto-hospedada via curl. |
| [**telephony**](/docs/user-guide/skills/optional/productivity/productivity-telephony) | Dá ao Hermes capacidades telefônicas sem mudanças no core. Provisiona e persiste um número Twilio, envia e recebe SMS/MMS, faz chamadas diretas e realiza chamadas de saída orientadas por IA via Bland.ai ou Vapi. |

## research

| Skill | Descrição |
|-------|-------------|
| [**bioinformatics**](/docs/user-guide/skills/optional/research/research-bioinformatics) | Gateway para mais de 400 skills de bioinformática do bioSkills e ClawBio. Cobre genômica, transcriptômica, célula única, chamada de variantes, farmacogenômica, metagenômica, biologia estrutural, e mais. Busca material de referência específico do domínio... |
| [**darwinian-evolver**](/docs/user-guide/skills/optional/research/research-darwinian-evolver) | Evolui prompts/regex/SQL/código com o loop evolutivo da Imbue. |
| [**domain-intel**](/docs/user-guide/skills/optional/research/research-domain-intel) | Reconhecimento passivo de domínio usando a stdlib do Python. Descoberta de subdomínios, inspeção de certificado SSL, consultas WHOIS, registros DNS, verificações de disponibilidade de domínio e análise em massa multi-domínio. Sem necessidade de chaves de API. |
| [**drug-discovery**](/docs/user-guide/skills/optional/research/research-drug-discovery) | Assistente de pesquisa farmacêutica para workflows de descoberta de fármacos. Busca compostos bioativos no ChEMBL, calcula drug-likeness (Lipinski Ro5, QED, TPSA, acessibilidade sintética), consulta interações medicamentosas via OpenFDA, interpreta ADMET... |
| [**duckduckgo-search**](/docs/user-guide/skills/optional/research/research-duckduckgo-search) | Busca web gratuita via DuckDuckGo — texto, notícias, imagens, vídeos. Sem necessidade de chave de API. Prefere o CLI `ddgs` quando instalado; use a biblioteca Python DDGS apenas depois de verificar que `ddgs` está disponível no runtime atual. |
| [**gitnexus-explorer**](/docs/user-guide/skills/optional/research/research-gitnexus-explorer) | Indexa uma base de código com GitNexus e serve um grafo de conhecimento interativo via UI web + túnel Cloudflare. |
| [**osint-investigation**](/docs/user-guide/skills/optional/research/research-osint-investigation) | Framework de investigação OSINT de registros públicos — arquivamentos do SEC EDGAR, contratos do USAspending, lobby do Senado, sanções OFAC, leaks offshore do ICIJ, registros de propriedade de NYC (ACRIS), registros da OpenCorporates, registros judiciais do CourtListener, Wayback... |
| [**parallel-cli**](/docs/user-guide/skills/optional/research/research-parallel-cli) | Skill opcional de fornecedor para o CLI Parallel — busca web nativa para agentes, extração, pesquisa profunda, enriquecimento, FindAll e monitoramento. Prefere saída JSON e fluxos não interativos. |
| [**qmd**](/docs/user-guide/skills/optional/research/research-qmd) | Busca bases de conhecimento pessoais, notas, documentos e transcrições de reuniões localmente usando qmd — um motor de recuperação híbrido com BM25, busca vetorial e reranking por LLM. Suporta integração via CLI e MCP. |
| [**scrapling**](/docs/user-guide/skills/optional/research/research-scrapling) | Web scraping com Scrapling - fetching HTTP, automação de navegador furtiva, bypass do Cloudflare e crawling spider via CLI e Python. |
| [**searxng-search**](/docs/user-guide/skills/optional/research/research-searxng-search) | Meta-busca gratuita via SearXNG — agrega resultados de mais de 70 motores de busca. Auto-hospedado ou use uma instância pública. Sem necessidade de chave de API. Recai automaticamente quando o toolset de busca web não está disponível. |

## security

| Skill | Descrição |
|-------|-------------|
| [**1password**](/docs/user-guide/skills/optional/security/security-1password) | Configura e usa o CLI do 1Password (op). Use ao instalar o CLI, ativar a integração com o app desktop, fazer login e ler/injetar segredos para comandos. |
| [**godmode**](/docs/user-guide/skills/optional/security/security-godmode) | Jailbreak de LLMs: Parseltongue, GODMODE, ULTRAPLINIAN. |
| [**oss-forensics**](/docs/user-guide/skills/optional/security/security-oss-forensics) | Investigação de supply chain, recuperação de evidências e análise forense para repositórios do GitHub. Cobre recuperação de commits excluídos, detecção de force-push, extração de IOC, coleta de evidências multi-fonte, formação/validação de hipóteses, e st... |
| [**sherlock**](/docs/user-guide/skills/optional/security/security-sherlock) | Busca OSINT de nome de usuário em mais de 400 redes sociais. Rastreia contas de redes sociais por nome de usuário. |
| [**unbroker**](/docs/user-guide/skills/optional/security/security-unbroker) | Remove autonomamente suas informações de sites de corretores de dados. |
| [**web-pentest**](/docs/user-guide/skills/optional/security/security-web-pentest) | Teste de penetração autorizado de aplicações web — reconhecimento, análise de vulnerabilidades, exploração baseada em prova e relatório profissional. Adapta a metodologia "No Exploit, No Report" de Shannon com salvaguardas rígidas para escopo, autoriza... |

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
| [**page-agent**](/docs/user-guide/skills/optional/web-development/web-development-page-agent) | Incorpora o alibaba/page-agent na sua própria aplicação web — um agente de GUI dentro da página, em JavaScript puro, distribuído como uma única tag &lt;script> ou pacote npm, que permite que usuários finais do seu site operem a UI com linguagem natural ("clicar em login, preencher usuári... |

---

## Contribuindo com Skills Opcionais {#contributing-optional-skills}

Para adicionar uma nova skill opcional ao repositório:

1. Crie um diretório em `optional-skills/<category>/<skill-name>/`
2. Adicione um `SKILL.md` com o frontmatter padrão (name, description, version, author)
3. Inclua qualquer arquivo de suporte nos subdiretórios `references/`, `templates/` ou `scripts/`
4. Envie um pull request — a skill aparecerá neste catálogo e receberá sua própria página de documentação após o merge
