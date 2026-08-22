---
title: "Integrações"
sidebar_label: "Visão geral"
sidebar_position: 0
---

# Integrações {#integrations}

O Hermes Agent conecta-se a sistemas externos para inferência de IA, servidores de ferramentas, fluxos de trabalho em IDE, acesso programático e muito mais. Essas integrações ampliam o que o Hermes pode fazer e onde pode rodar.

:::tip Comece aqui
Se você só tem tempo para configurar uma integração, configure o [Nous Portal](/integrations/nous-portal) — um único login OAuth cobre 300+ modelos mais as quatro ferramentas do Tool Gateway (busca web, geração de imagem, TTS e automação de browser).
:::

## Provedores de IA e roteamento {#ai-providers--routing}

O Hermes suporta vários provedores de inferência de IA prontos para uso. Use `hermes model` para configurar interativamente, ou defina em `config.yaml`.

- **[Provedores de IA](/integrations/providers)** — OpenRouter, Anthropic, OpenAI, Google e qualquer endpoint compatível com OpenAI. O Hermes detecta automaticamente capacidades como visão, streaming e uso de ferramentas por provedor.
- **[Roteamento de provedores](/user-guide/features/provider-routing)** — Controle fino sobre quais provedores subjacentes atendem suas requisições OpenRouter. Otimize por custo, velocidade ou qualidade com ordenação, whitelists, blacklists e ordem de prioridade explícita.
- **[Provedores de fallback](/user-guide/features/fallback-providers)** — Failover automático para provedores LLM de backup quando seu modelo primário encontra erros. Inclui fallback do modelo primário e fallback independente de tarefas auxiliares para visão, compressão e extração web.

## Servidores de ferramentas (MCP) {#tool-servers-mcp}

- **[Servidores MCP](/user-guide/features/mcp)** — Conecte o Hermes a servidores de ferramentas externos via Model Context Protocol. Acesse ferramentas do GitHub, bancos de dados, sistemas de arquivos, stacks de browser, APIs internas e muito mais sem escrever ferramentas nativas do Hermes. Suporta transportes stdio e SSE, filtragem de ferramentas por servidor e registro de recursos/prompts com consciência de capacidades.

## Backends de busca web {#web-search-backends}

As ferramentas `web_search` e `web_extract` suportam oito backends de provedor, configurados via `config.yaml` ou `hermes tools`:

| Backend | Env Var | Search | Extract | Crawl |
|---------|---------|--------|---------|-------|
| **Firecrawl** (default) | `FIRECRAWL_API_KEY` | ✔ | ✔ | ✔ |
| **SearXNG** | `SEARXNG_URL` | ✔ | — | — |
| **Brave** (free tier) | `BRAVE_SEARCH_API_KEY` | ✔ | — | — |
| **DuckDuckGo** (ddgs) | _(none)_ | ✔ | — | — |
| **Tavily** | `TAVILY_API_KEY` (opcional) | ✔ | ✔ | — |
| **Exa** | `EXA_API_KEY` | ✔ | ✔ | — |
| **Parallel** | `PARALLEL_API_KEY` | ✔ | ✔ | — |
| **xAI** | `XAI_API_KEY` | ✔ | — | — |

Exemplo de setup rápido:

```yaml
web:
  backend: firecrawl    # firecrawl | searxng | brave-free | ddgs | tavily | exa | parallel | xai
```

Se `web.backend` não estiver definido, o backend é detectado automaticamente a partir de qualquer chave de API disponível. Firecrawl self-hosted também é suportado via `FIRECRAWL_API_URL`. Selecionar Tavily em `hermes tools` funciona sem chave.

## Automação de browser {#browser-automation}

O Hermes inclui automação completa de browser com várias opções de backend para navegar em sites, preencher formulários e extrair informações:

- **Browserbase** — Browsers gerenciados na nuvem com ferramentas anti-bot, resolução de CAPTCHA e proxies residenciais
- **Browser Use** — Provedor alternativo de browser na nuvem
- **CDP local Chromium-family** — Conecte ao seu Chrome, Brave, Chromium ou Edge em execução usando `/browser connect`
- **Chromium local** — Browser local headless via CLI `agent-browser`

Veja [Automação de browser](/user-guide/features/browser) para setup e uso.

## Provedores de voz e TTS {#voice--tts-providers}

Text-to-speech e speech-to-text em todas as plataformas de mensagens:

| Provider | Quality | Cost | API Key |
|----------|---------|------|---------|
| **Edge TTS** (default) | Good | Free | None needed |
| **ElevenLabs** | Excellent | Paid | `ELEVENLABS_API_KEY` |
| **OpenAI TTS** | Good | Paid | `VOICE_TOOLS_OPENAI_KEY` |
| **MiniMax** | Good | Paid | `MINIMAX_API_KEY` |
| **xAI TTS** | Good | Paid | `XAI_API_KEY` |
| **NeuTTS** | Good | Free | None needed |

Speech-to-text suporta oito provedores: faster-whisper local (gratuito, roda no dispositivo), wrapper de comando local, Groq, OpenAI Whisper API, Mistral, xAI, ElevenLabs Scribe e DeepInfra. Transcrição de mensagens de voz funciona no Telegram, Discord, WhatsApp e outras plataformas de mensagens. Veja [Voz e TTS](/user-guide/features/tts) e [Modo de voz](/user-guide/features/voice-mode) para detalhes.

## Integração com IDE e editores {#ide--editor-integration}

- **[Integração com IDE (ACP)](/user-guide/features/acp)** — Use o Hermes Agent em editores compatíveis com ACP como VS Code, Zed e JetBrains. O Hermes roda como servidor ACP, renderizando mensagens de chat, atividade de ferramentas, diffs de arquivo e comandos de terminal dentro do seu editor.

## Acesso programático {#programmatic-access}

- **[API Server](/user-guide/features/api-server)** — Exponha o Hermes como endpoint HTTP compatível com OpenAI. Qualquer frontend que fale o formato OpenAI — Open WebUI, LobeChat, LibreChat, NextChat, ChatBox — pode conectar e usar o Hermes como backend com seu conjunto completo de ferramentas.

## Memória e personalização {#memory--personalization}

- **[Memória integrada](/user-guide/features/memory)** — Memória persistente e curada via arquivos `MEMORY.md` e `USER.md`. O agente mantém stores limitados de notas pessoais e dados de perfil do usuário que sobrevivem entre sessões.
- **[Provedores de memória](/user-guide/features/memory-providers)** — Conecte backends de memória externos para personalização mais profunda. Oito provedores são suportados: Honcho (raciocínio dialético), OpenViking (recuperação em camadas), Mem0 (extração na nuvem), Hindsight (grafos de conhecimento), Holographic (SQLite local), RetainDB (busca híbrida), ByteRover (baseado em CLI) e Supermemory.

## Plataformas de mensagens {#messaging-platforms}

O Hermes roda como bot de gateway em 27+ plataformas de mensagens, todas configuradas pelo mesmo subsistema `gateway`:

- **[Telegram](/user-guide/messaging/telegram)**, **[Discord](/user-guide/messaging/discord)**, **[Slack](/user-guide/messaging/slack)**, **[WhatsApp](/user-guide/messaging/whatsapp)**, **[Signal](/user-guide/messaging/signal)**, **[Matrix](/user-guide/messaging/matrix)**, **[Mattermost](/user-guide/messaging/mattermost)**, **[Email](/user-guide/messaging/email)**, **[SMS](/user-guide/messaging/sms)**, **[DingTalk](/user-guide/messaging/dingtalk)**, **[Feishu/Lark](/user-guide/messaging/feishu)**, **[WeCom](/user-guide/messaging/wecom)**, **[WeCom Callback](/user-guide/messaging/wecom-callback)**, **[Weixin](/user-guide/messaging/weixin)**, **[BlueBubbles](/user-guide/messaging/bluebubbles)**, **[Buzz](/user-guide/messaging/buzz)**, **[QQ Bot](/user-guide/messaging/qqbot)**, **[Yuanbao](/user-guide/messaging/yuanbao)**, **[Home Assistant](/user-guide/messaging/homeassistant)**, **[Microsoft Teams](/user-guide/messaging/teams)**, **[Microsoft Teams Meetings](/user-guide/messaging/teams-meetings)**, **[Microsoft Graph Webhook](/user-guide/messaging/msgraph-webhook)**, **[Google Chat](/user-guide/messaging/google_chat)**, **[LINE](/user-guide/messaging/line)**, **[ntfy](/user-guide/messaging/ntfy)**, **[SimpleX](/user-guide/messaging/simplex)**, **[Open WebUI](/user-guide/messaging/open-webui)**, **[Webhooks](/user-guide/messaging/webhooks)**

Veja a [visão geral do Messaging Gateway](/user-guide/messaging) para a tabela comparativa de plataformas e guia de setup.

### Links rápidos de conexão {#quick-connect-links}

As grandes plataformas têm uma URL canônica de "crie seu bot/app", e algumas aceitam parâmetros que abrem o formulário certo. Pule a caça no console e vá direto:

| Platform | Direct link | What it opens |
|----------|-------------|---------------|
| **Telegram** | [t.me/BotFather](https://t.me/BotFather) | Chat with BotFather — send `/newbot` to mint a bot token |
| **Discord** | [discord.com/developers/applications?new_application=true](https://discord.com/developers/applications?new_application=true) | Developer Portal with the **New Application** dialog pre-opened |
| **Slack** | [api.slack.com/apps?new_app=1](https://api.slack.com/apps?new_app=1) | The **Create New App** dialog — pick *From an app manifest* and paste the manifest `hermes slack manifest --agent-view` generates |
| **LINE** | [developers.line.biz/console](https://developers.line.biz/console/) | LINE Developers Console for creating a Messaging API channel |
| **Feishu/Lark** | [open.feishu.cn/app](https://open.feishu.cn/app) | Feishu open-platform console for creating a custom app |

A página de setup de cada plataforma explica o que fazer quando você chegar lá.

## Workspaces de colaboração {#collaboration-workspaces}

- **[Buzz](/integrations/buzz)** — Workspace humano+agente baseado em Nostr da Block. Três caminhos de integração: Buzz Desktop inicia o Hermes como runtime ACP gerenciado, a ponte relay `buzz-acp` hospeda um servidor de identidade Hermes server-side, ou a plataforma nativa de gateway entra em canais Buzz com memória/skills/aprovações/cron completos do Hermes. A página de visão geral compara os três.

## Automação residencial {#home-automation}

- **[Home Assistant](/user-guide/messaging/homeassistant)** — Controle dispositivos smart home via quatro ferramentas dedicadas (`ha_list_entities`, `ha_get_state`, `ha_list_services`, `ha_call_service`). O toolset Home Assistant ativa automaticamente quando `HASS_TOKEN` está configurado.

## Plugins {#plugins}

- **[Sistema de plugins](/user-guide/features/plugins)** — Estenda o Hermes com ferramentas customizadas, hooks de ciclo de vida e comandos CLI sem modificar o core. Plugins são descobertos em `~/.hermes/plugins/`, `.hermes/plugins/` local ao projeto e entry points pip.
- **[Construir um plugin](/developer-guide/plugins)** — Guia passo a passo para criar plugins Hermes com ferramentas, hooks e comandos CLI.

## Treinamento e avaliação {#training--evaluation}

- **[Processamento em lote](/user-guide/features/batch-processing)** — Execute o agente em centenas de prompts em paralelo, gerando dados de trajetória estruturados no formato ShareGPT para geração de dados de treinamento ou avaliação.
