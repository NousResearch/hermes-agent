---
title: "Integrações"
sidebar_label: "Visão Geral"
sidebar_position: 0
---

# Integrações {#integrations}

O Hermes Agent se conecta a sistemas externos para inferência de IA, servidores de ferramentas, fluxos de trabalho de IDE, acesso programático e muito mais. Essas integrações ampliam o que o Hermes pode fazer e onde ele pode ser executado.

:::tip Comece por aqui
Se você só tem tempo para configurar uma integração, configure o [Nous Portal](/integrations/nous-portal) — um único login OAuth cobre mais de 300 modelos além das quatro ferramentas do Tool Gateway (busca na web, geração de imagens, TTS e automação de navegador).
:::

## Provedores de IA e Roteamento {#ai-providers--routing}

O Hermes oferece suporte a múltiplos provedores de inferência de IA prontos para uso. Use `hermes model` para configurar interativamente, ou defina-os em `config.yaml`.

- **[Provedores de IA](/integrations/providers)** — OpenRouter, Anthropic, OpenAI, Google e qualquer endpoint compatível com OpenAI. O Hermes detecta automaticamente recursos como visão, streaming e uso de ferramentas por provedor.
- **[Roteamento de Provedores](/user-guide/features/provider-routing)** — Controle refinado sobre quais provedores subjacentes atendem suas requisições do OpenRouter. Otimize para custo, velocidade ou qualidade com ordenação, listas de permissão, listas de bloqueio e ordem de prioridade explícita.
- **[Provedores de Fallback](/user-guide/features/fallback-providers)** — Failover automático para provedores de LLM de backup quando seu modelo principal encontra erros. Inclui fallback do modelo principal e fallback independente de tarefas auxiliares para visão, compressão e extração web.

## Servidores de Ferramentas (MCP) {#tool-servers-mcp}

- **[Servidores MCP](/user-guide/features/mcp)** — Conecte o Hermes a servidores de ferramentas externos via Model Context Protocol. Acesse ferramentas do GitHub, bancos de dados, sistemas de arquivos, pilhas de navegador, APIs internas e muito mais sem escrever ferramentas nativas do Hermes. Suporta transportes stdio e SSE, filtragem de ferramentas por servidor e registro de recursos/prompts com reconhecimento de capacidades.

## Backends de Busca na Web {#web-search-backends}

As ferramentas `web_search` e `web_extract` oferecem suporte a oito provedores de backend, configurados via `config.yaml` ou `hermes tools`:

| Backend | Variável de Ambiente | Busca | Extração | Crawl |
|---------|---------|--------|---------|-------|
| **Firecrawl** (padrão) | `FIRECRAWL_API_KEY` | ✔ | ✔ | ✔ |
| **SearXNG** | `SEARXNG_URL` | ✔ | — | — |
| **Brave** (nível gratuito) | `BRAVE_SEARCH_API_KEY` | ✔ | — | — |
| **DuckDuckGo** (ddgs) | _(nenhuma)_ | ✔ | — | — |
| **Tavily** | `TAVILY_API_KEY` | ✔ | ✔ | ✔ |
| **Exa** | `EXA_API_KEY` | ✔ | ✔ | — |
| **Parallel** | `PARALLEL_API_KEY` | ✔ | ✔ | — |
| **xAI** | `XAI_API_KEY` | ✔ | — | — |

Exemplo de configuração rápida:

```yaml
web:
  backend: firecrawl    # firecrawl | searxng | brave-free | ddgs | tavily | exa | parallel | xai
```

Se `web.backend` não estiver definido, o backend é detectado automaticamente a partir da chave de API disponível. O Firecrawl auto-hospedado também é suportado via `FIRECRAWL_API_URL`.

## Automação de Navegador {#browser-automation}

O Hermes inclui automação de navegador completa com múltiplas opções de backend para navegar em sites, preencher formulários e extrair informações:

- **Browserbase** — Navegadores gerenciados na nuvem com ferramentas anti-bot, resolução de CAPTCHA e proxies residenciais
- **Browser Use** — Provedor alternativo de navegador em nuvem
- **CDP local da família Chromium** — Conecte-se ao seu navegador Chrome, Brave, Chromium ou Edge em execução usando `/browser connect`
- **Chromium local** — Navegador local headless via a CLI `agent-browser`

Veja [Automação de Navegador](/user-guide/features/browser) para configuração e uso.

## Provedores de Voz e TTS {#voice--tts-providers}

Texto para fala e fala para texto em todas as plataformas de mensagens:

| Provedor | Qualidade | Custo | Chave de API |
|----------|---------|------|---------|
| **Edge TTS** (padrão) | Boa | Gratuito | Nenhuma necessária |
| **ElevenLabs** | Excelente | Pago | `ELEVENLABS_API_KEY` |
| **OpenAI TTS** | Boa | Pago | `VOICE_TOOLS_OPENAI_KEY` |
| **MiniMax** | Boa | Pago | `MINIMAX_API_KEY` |
| **xAI TTS** | Boa | Pago | `XAI_API_KEY` |
| **NeuTTS** | Boa | Gratuito | Nenhuma necessária |

A conversão de fala para texto oferece suporte a seis provedores: faster-whisper local (gratuito, executado no dispositivo), um wrapper de comando local, Groq, API OpenAI Whisper, Mistral e xAI. A transcrição de mensagens de voz funciona no Telegram, Discord, WhatsApp e outras plataformas de mensagens. Veja [Voz e TTS](/user-guide/features/tts) e [Modo de Voz](/user-guide/features/voice-mode) para mais detalhes.

## Integração com IDEs e Editores {#ide--editor-integration}

- **[Integração com IDE (ACP)](/user-guide/features/acp)** — Use o Hermes Agent dentro de editores compatíveis com ACP, como VS Code, Zed e JetBrains. O Hermes é executado como um servidor ACP, renderizando mensagens de chat, atividade de ferramentas, diffs de arquivos e comandos de terminal dentro do seu editor.

## Acesso Programático {#programmatic-access}

- **[Servidor de API](/user-guide/features/api-server)** — Exponha o Hermes como um endpoint HTTP compatível com OpenAI. Qualquer frontend que use o formato OpenAI — Open WebUI, LobeChat, LibreChat, NextChat, ChatBox — pode se conectar e usar o Hermes como backend com todo o seu conjunto de ferramentas.

## Memória e Personalização {#memory--personalization}

- **[Memória Integrada](/user-guide/features/memory)** — Memória persistente e curada por meio de arquivos `MEMORY.md` e `USER.md`. O agente mantém armazenamentos limitados de notas pessoais e dados de perfil do usuário que persistem entre sessões.
- **[Provedores de Memória](/user-guide/features/memory-providers)** — Conecte backends de memória externos para uma personalização mais profunda. Oito provedores são suportados: Honcho (raciocínio dialético), OpenViking (recuperação em camadas), Mem0 (extração em nuvem), Hindsight (grafos de conhecimento), Holographic (SQLite local), RetainDB (busca híbrida), ByteRover (baseado em CLI) e Supermemory.

## Plataformas de Mensagens {#messaging-platforms}

O Hermes é executado como um bot de gateway em mais de 27 plataformas de mensagens, todas configuradas através do mesmo subsistema `gateway`:

- **[Telegram](/user-guide/messaging/telegram)**, **[Discord](/user-guide/messaging/discord)**, **[Slack](/user-guide/messaging/slack)**, **[WhatsApp](/user-guide/messaging/whatsapp)**, **[Signal](/user-guide/messaging/signal)**, **[Matrix](/user-guide/messaging/matrix)**, **[Mattermost](/user-guide/messaging/mattermost)**, **[Email](/user-guide/messaging/email)**, **[SMS](/user-guide/messaging/sms)**, **[DingTalk](/user-guide/messaging/dingtalk)**, **[Feishu/Lark](/user-guide/messaging/feishu)**, **[WeCom](/user-guide/messaging/wecom)**, **[WeCom Callback](/user-guide/messaging/wecom-callback)**, **[Weixin](/user-guide/messaging/weixin)**, **[BlueBubbles](/user-guide/messaging/bluebubbles)**, **[QQ Bot](/user-guide/messaging/qqbot)**, **[Yuanbao](/user-guide/messaging/yuanbao)**, **[Home Assistant](/user-guide/messaging/homeassistant)**, **[Microsoft Teams](/user-guide/messaging/teams)**, **[Microsoft Teams Meetings](/user-guide/messaging/teams-meetings)**, **[Microsoft Graph Webhook](/user-guide/messaging/msgraph-webhook)**, **[Google Chat](/user-guide/messaging/google_chat)**, **[LINE](/user-guide/messaging/line)**, **[ntfy](/user-guide/messaging/ntfy)**, **[SimpleX](/user-guide/messaging/simplex)**, **[Open WebUI](/user-guide/messaging/open-webui)**, **[Webhooks](/user-guide/messaging/webhooks)**

Veja a [visão geral do Gateway de Mensagens](/user-guide/messaging) para a tabela comparativa de plataformas e o guia de configuração.

## Automação Residencial {#home-automation}

- **[Home Assistant](/user-guide/messaging/homeassistant)** — Controle dispositivos de casa inteligente por meio de quatro ferramentas dedicadas (`ha_list_entities`, `ha_get_state`, `ha_list_services`, `ha_call_service`). O conjunto de ferramentas do Home Assistant é ativado automaticamente quando `HASS_TOKEN` está configurado.

## Plugins {#plugins}

- **[Sistema de Plugins](/user-guide/features/plugins)** — Estenda o Hermes com ferramentas personalizadas, hooks de ciclo de vida e comandos de CLI sem modificar o código principal. Os plugins são descobertos em `~/.hermes/plugins/`, no `.hermes/plugins/` local do projeto e em pontos de entrada instalados via pip.
- **[Construir um Plugin](/developer-guide/plugins)** — Guia passo a passo para criar plugins do Hermes com ferramentas, hooks e comandos de CLI.

## Treinamento e Avaliação {#training--evaluation}

- **[Processamento em Lote](/user-guide/features/batch-processing)** — Execute o agente em centenas de prompts em paralelo, gerando dados de trajetória estruturados no formato ShareGPT para geração de dados de treinamento ou avaliação.
