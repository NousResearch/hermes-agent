---
title: "Visão geral dos recursos"
sidebar_label: "Visão geral"
sidebar_position: 1
---

# Visão geral dos recursos

O Hermes Agent inclui um conjunto rico de capacidades que vão muito além do chat básico. Da memória persistente e contexto com arquivos à automação de navegador e conversas por voz, esses recursos trabalham juntos para tornar o Hermes um assistente autônomo poderoso.

:::tip Não sabe por onde começar?
`hermes setup --portal` cobre um provedor de modelo mais as quatro ferramentas do Tool Gateway (web search, image generation, TTS, browser) em um único comando. Veja o [Nous Portal](/integrations/nous-portal).
:::

## Núcleo {#core}

- **[Ferramentas e toolsets](tools.md)** — Ferramentas são funções que estendem as capacidades do agente. Elas são organizadas em toolsets lógicos que podem ser habilitados ou desabilitados por plataforma, cobrindo web search, execução de terminal, edição de arquivos, memória, delegação e muito mais.
- **[Sistema de skills](skills.md)** — Documentos de conhecimento sob demanda que o agente pode carregar quando necessário. Skills seguem um padrão de divulgação progressiva para minimizar o uso de tokens e são compatíveis com o padrão aberto [agentskills.io](https://agentskills.io/specification).
- **[Memória persistente](memory.md)** — Memória limitada e curada que persiste entre sessões. O Hermes lembra suas preferências, projetos, ambiente e o que aprendeu via `MEMORY.md` e `USER.md`.
- **[Arquivos de contexto](context-files.md)** — O Hermes descobre e carrega automaticamente arquivos de contexto do projeto (`.hermes.md`, `AGENTS.md`, `CLAUDE.md`, `SOUL.md`, `.cursorrules`) que moldam como ele se comporta no seu projeto.
- **[Referências de contexto](context-references.md)** — Digite `@` seguido de uma referência para injetar arquivos, pastas, git diffs e URLs diretamente nas suas mensagens. O Hermes expande a referência inline e anexa o conteúdo automaticamente.
- **[Checkpoints](../checkpoints-and-rollback.md)** — O Hermes cria snapshots automaticamente do seu diretório de trabalho antes de alterar arquivos, dando uma rede de segurança para reverter com `/rollback` se algo der errado.

## Automação {#automation}

- **[Tarefas agendadas (Cron)](cron.md)** — Agende tarefas para rodar automaticamente com linguagem natural ou expressões cron. Jobs podem anexar skills, entregar resultados em qualquer plataforma e suportam pausar/retomar/editar.
- **[Delegação de subagentes](delegation.md)** — A ferramenta `delegate_task` cria instâncias filhas do agente com contexto isolado, toolsets restritos e sessões de terminal próprias. Rode 3 subagentes concorrentes por padrão (configurável) para fluxos de trabalho paralelos.
- **[Execução de código](code-execution.md)** — A ferramenta `execute_code` permite que o agente escreva scripts Python que chamam ferramentas do Hermes programaticamente, colapsando fluxos multi-etapa em um único turno de LLM via execução RPC em sandbox.
- **[Event hooks](hooks.md)** — Execute código personalizado em pontos-chave do ciclo de vida. Gateway hooks tratam logging, alertas e webhooks; plugin hooks tratam interceptação de ferramentas, métricas e guardrails.
- **[Processamento em lote](batch-processing.md)** — Execute o agente Hermes em centenas ou milhares de prompts em paralelo, gerando dados de trajetória estruturados no formato ShareGPT para geração de dados de treinamento ou avaliação.

## Mídia e web {#media--web}

- **[Modo de voz](voice-mode.md)** — Interação completa por voz no CLI e plataformas de mensagens. Fale com o agente pelo microfone, ouça respostas faladas e tenha conversas ao vivo em canais de voz do Discord.
- **[Automação de navegador](browser.md)** — Automação completa de navegador com vários backends: Browserbase na nuvem, Browser Use na nuvem, Chrome/Brave/Chromium/Edge local via CDP ou Chromium local. Navegue em sites, preencha formulários e extraia informações.
- **[Visão e colagem de imagem](vision.md)** — Suporte multimodal de visão. Cole imagens da área de transferência no CLI e peça ao agente para analisar, descrever ou trabalhar com elas usando qualquer modelo com visão.
- **[Geração de imagem](image-generation.md)** — Gere imagens a partir de prompts de texto usando FAL.ai. Onze modelos suportados (FLUX 2 Klein/Pro, GPT-Image 1.5/2, Nano Banana Pro, Ideogram V3, Recraft V4 Pro, Qwen, Z-Image Turbo, Krea V2 Medium/Large); escolha um via `hermes tools`.
- **[Voz e TTS](tts.md)** — Saída text-to-speech e transcrição de mensagens de voz em todas as plataformas de mensagens, com dez opções nativas de provedor: Edge TTS (grátis), ElevenLabs, OpenAI TTS, MiniMax, Mistral Voxtral, Google Gemini, xAI, NeuTTS, KittenTTS e Piper — além de provedores de comando personalizado para qualquer CLI TTS local.

## Integrações {#integrations}

- **[Integração MCP](mcp.md)** — Conecte-se a qualquer servidor MCP via transporte stdio ou HTTP. Acesse ferramentas externas do GitHub, bancos de dados, sistemas de arquivos e APIs internas sem escrever ferramentas nativas do Hermes. Inclui filtragem de ferramentas por servidor e suporte a sampling.
- **[Roteamento de provedores](provider-routing.md)** — Controle fino sobre quais provedores de IA atendem suas requisições. Otimize por custo, velocidade ou qualidade com ordenação, whitelists, blacklists e prioridade.
- **[Provedores de fallback](fallback-providers.md)** — Failover automático para provedores LLM de backup quando seu modelo principal encontra erros, incluindo fallback independente para tarefas auxiliares como visão e compressão.
- **[Pools de credenciais](credential-pools.md)** — Distribua chamadas de API entre várias chaves do mesmo provedor. Rotação automática em rate limits ou falhas.
- **[Prompt caching](../configuration#prompt-caching)** — Cache de prefixo cross-session de 1 hora embutido para Claude no Anthropic nativo, OpenRouter e Nous Portal. Sempre ativo; nenhuma configuração necessária.
- **[Provedores de memória](memory-providers.md)** — Conecte backends de memória externos (Honcho, OpenViking, Mem0, Hindsight, Holographic, RetainDB, ByteRover, Supermemory) para modelagem de usuário cross-session e personalização além do sistema de memória embutido.
- **[Servidor API](api-server.md)** — Exponha o Hermes como um endpoint HTTP compatível com OpenAI. Conecte qualquer frontend que fale o formato OpenAI — Open WebUI, LobeChat, LibreChat e mais.
- **[Integração com IDE (ACP)](acp.md)** — Use o Hermes dentro de editores compatíveis com ACP como VS Code, Zed e JetBrains. Chat, atividade de ferramentas, diffs de arquivo e comandos de terminal renderizam dentro do seu editor.
- **[Processamento em lote](batch-processing.md)** — Execute o agente em muitos prompts ou tarefas em paralelo a partir do CLI, com saídas estruturadas e captura de trajetória adequadas para evals ou pipelines de treinamento downstream.

## Personalização {#customization}

- **[Personalidade e SOUL.md](personality.md)** — Personalidade do agente totalmente personalizável. `SOUL.md` é o arquivo de identidade principal — a primeira coisa no system prompt — e você pode trocar presets built-in ou personalizados de `/personality` por sessão.
- **[Skins e temas](skins.md)** — Personalize a apresentação visual do CLI: cores do banner, faces e verbos do spinner, rótulos da caixa de resposta, texto de branding e o prefixo de atividade de ferramentas.
- **[Plugins](plugins.md)** — Adicione ferramentas, hooks e integrações personalizadas sem modificar o core. Três tipos de plugin: plugins gerais (ferramentas/hooks), provedores de memória (conhecimento cross-session) e context engines (gerenciamento alternativo de contexto). Gerenciados via a UI interativa unificada `hermes plugins`.
