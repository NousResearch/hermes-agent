---
sidebar_position: 10
title: "Modo de Voz"
description: "Conversas de voz em tempo real com Hermes Agent — CLI, Telegram, Discord (DMs, canais de texto e canais de voz)"
---

# Modo de Voz {#voice-mode}

O Hermes Agent suporta interação de voz completa na CLI e plataformas de mensagens. Fale com o agente pelo microfone, ouça respostas faladas e tenha conversas de voz ao vivo em canais de voz do Discord.

Se quiser um walkthrough prático de setup com configurações recomendadas e padrões de uso reais, veja [Usar Modo de Voz com Hermes](../../guides/use-voice-mode-with-hermes.md).

Para iniciar sessões hands-free — dizer "hey hermes" (ou qualquer frase) para abrir uma sessão de voz nova na CLI, TUI ou app desktop — veja [Wake Word](/user-guide/features/wake-word).

## Pré-requisitos {#prerequisites}

Antes de usar recursos de voz, certifique-se de ter:

1. **Hermes Agent instalado** — via script de instalação (veja [Instalação](/getting-started/installation))
2. **Um provedor LLM configurado** — rode `hermes model` ou defina as credenciais do provedor preferido em `~/.hermes/.env`
3. **Um setup base funcionando** — rode `hermes` para verificar que o agente responde a texto antes de habilitar voz

:::tip
O diretório `~/.hermes/` e o `config.yaml` padrão são criados automaticamente na primeira vez que você roda `hermes`. Você só precisa criar `~/.hermes/.env` manualmente para API keys.
:::

:::tip Nous Portal cobre ambos
Uma assinatura paga do [Nous Portal](/user-guide/features/tool-gateway) fornece o LLM (passo 2) **e** OpenAI TTS via Tool Gateway — sem OpenAI key separada. Em instalação nova, `hermes setup --portal` configura os dois de uma vez.
:::

## Visão Geral {#overview}

| Recurso | Plataforma | Descrição |
|---------|----------|-------------|
| **Voz Interativa** | CLI | Pressione Ctrl+B para gravar, o agente detecta silêncio automaticamente e responde |
| **Resposta de Voz Automática** | Telegram, Discord | O agente envia áudio falado junto com respostas de texto |
| **Canal de Voz** | Discord | Bot entra no VC, escuta usuários falando, fala respostas de volta |

## Requisitos {#requirements}

### Pacotes Python {#python-packages}

```bash
# Modo de voz CLI (microfone + reprodução de áudio)
cd ~/.hermes/hermes-agent && uv pip install -e ".[voice]"

# Mensagens Discord + Telegram (inclui discord.py[voice] para suporte a VC)
cd ~/.hermes/hermes-agent && uv pip install -e ".[messaging]"

# TTS premium (ElevenLabs)
cd ~/.hermes/hermes-agent && uv pip install -e ".[tts-premium]"

# TTS local (NeuTTS, opcional)
python -m pip install -U neutts[all]

# Tudo de uma vez
cd ~/.hermes/hermes-agent && uv pip install -e ".[all]"
```

| Extra | Pacotes | Necessário Para |
|-------|----------|-------------|
| `voice` | `sounddevice`, `numpy` | Modo de voz CLI |
| `messaging` | `discord.py[voice]`, `python-telegram-bot`, `aiohttp` | Bots Discord & Telegram |
| `tts-premium` | `elevenlabs` | Provedor TTS ElevenLabs |

Provedor TTS local opcional: instale `neutts` separadamente com `python -m pip install -U neutts[all]`. No primeiro uso baixa o modelo automaticamente.

:::info
`discord.py[voice]` instala **PyNaCl** (para criptografia de voz) e **bindings opus** automaticamente. Isso é necessário para suporte a canal de voz do Discord.
:::

### Dependências de Sistema {#system-dependencies}

```bash
# macOS
brew install portaudio ffmpeg opus
brew install espeak-ng   # for NeuTTS

# Ubuntu/Debian
sudo apt install portaudio19-dev ffmpeg libopus0
sudo apt install espeak-ng   # for NeuTTS
```

| Dependência | Propósito | Necessário Para |
|-----------|---------|-------------|
| **PortAudio** | Entrada de microfone e reprodução de áudio | Modo de voz CLI |
| **ffmpeg** | Conversão de formato de áudio (MP3 → Opus, PCM → WAV) | Todas as plataformas |
| **Opus** | Codec de voz do Discord | Canais de voz do Discord |
| **espeak-ng** | Backend phonemizer | Provedor NeuTTS local |

### API Keys {#api-keys}

Adicione em `~/.hermes/.env`:

```bash
# Speech-to-Text — provedor local NÃO precisa de key
# pip install faster-whisper          # Grátis, roda localmente, recomendado
GROQ_API_KEY=your-key                 # Groq Whisper — rápido, tier grátis (cloud)
VOICE_TOOLS_OPENAI_KEY=your-key       # OpenAI Whisper — pago (cloud)

# Text-to-Speech (opcional — Edge TTS e NeuTTS funcionam sem key)
ELEVENLABS_API_KEY=***           # ElevenLabs — qualidade premium
# VOICE_TOOLS_OPENAI_KEY acima também habilita OpenAI TTS
```

:::tip
Se `faster-whisper` estiver instalado, o modo de voz funciona com **zero API keys** para STT. O modelo (~150 MB para `base`) baixa automaticamente no primeiro uso.
:::

---

## Modo de Voz CLI {#cli-voice-mode}

O modo de voz está disponível tanto na **CLI clássica** (`hermes chat`) quanto na **TUI** (`hermes --tui`). O comportamento é idêntico — mesmos slash commands, mesma detecção de silêncio VAD, mesmo TTS streaming, mesmo filtro de alucinação. A TUI adicionalmente encaminha logs forenses de crash para `~/.hermes/logs/` para que falhas de push-to-talk em backends de áudio exóticos possam ser reportadas com stack trace completo em vez de sumir silenciosamente.

### Início Rápido {#quick-start}

Inicie a CLI e habilite o modo de voz:

```bash
hermes                # Iniciar a CLI interativa
```

Depois use estes comandos dentro da CLI:

```
/voice          Alternar modo de voz ligado/desligado
/voice on       Habilitar modo de voz
/voice off      Desabilitar modo de voz
/voice tts      Alternar saída TTS
/voice status   Mostrar estado atual
```

### Como Funciona {#how-it-works}

1. Inicie a CLI com `hermes` e habilite o modo de voz com `/voice on`
2. **Pressione Ctrl+B** — um beep toca (880Hz), gravação começa
3. **Fale** — uma barra de nível de áudio ao vivo mostra sua entrada: `● [▁▂▃▅▇▇▅▂] ❯`
4. **Pare de falar** — após 3 segundos de silêncio, a gravação para automaticamente
5. **Dois beeps** tocam (660Hz) confirmando que a gravação terminou
6. O áudio é transcrito via Whisper e enviado ao agente
7. Se TTS estiver habilitado, a resposta do agente é falada em voz alta
8. A gravação **reinicia automaticamente** — fale de novo sem pressionar nenhuma tecla

Este loop continua até você pressionar **Ctrl+B** durante a gravação (sai do modo contínuo) ou 3 gravações consecutivas detectarem nenhuma fala.

:::tip
A tecla de gravação é configurável via `voice.record_key` em `~/.hermes/config.yaml` (padrão: `ctrl+b`).
:::

### Detecção de Silêncio {#silence-detection}

Algoritmo em duas etapas detecta quando você terminou de falar:

1. **Confirmação de fala** — espera áudio acima do limiar RMS (200) por pelo menos 0,3s, tolerando quedas breves entre sílabas
2. **Detecção de fim** — uma vez confirmada a fala, dispara após 3,0 segundos de silêncio contínuo

Se nenhuma fala for detectada por 15 segundos, a gravação para automaticamente.

Tanto `silence_threshold` quanto `silence_duration` são configuráveis em `config.yaml`. Você também pode desabilitar os beeps de início/fim de gravação com `voice.beep_enabled: false`.

### Encerrando um chat de voz por voz {#ending-a-voice-chat-by-voice}

Diga **"stop"** — e nada mais — para encerrar a conversa de voz hands-free. A correspondência é deliberadamente estrita: a utterance inteira (case-insensitive, pontuação ao redor ignorada) deve ser igual a uma frase configurada, então "stop doing that and try X instead" ainda chega ao agente normalmente. Customize a lista de frases com `voice.stop_phrases` em `config.yaml` (ex.: `["stop", "goodbye hermes"]`), ou defina como `[]` para desabilitar. Um chat de voz também termina sozinho após três ciclos silenciosos consecutivos (nenhuma fala detectada).

**Digitar** uma frase de stop isolada enquanto um chat de voz está ativo funciona da mesma forma em toda superfície (CLI, TUI, desktop): a mensagem encerra o chat de voz em vez de ser enviada ao agente. Fora de um chat de voz, "stop" digitado é uma mensagem comum.

### TTS Streaming {#streaming-tts}

Com TTS habilitado, o agente fala a resposta **frase a frase** conforme gera texto — você não espera a resposta completa. Funciona com **todo provedor TTS**:

1. Bufferiza deltas de texto em frases completas (mín. 20 chars)
2. Remove formatação markdown, emoji e blocos `<think>`
3. Reproduz áudio por frase em tempo real — provedores com API PCM chunked (ElevenLabs, OpenAI) streamam áudio bruto para o menor time-to-first-word; todo outro provedor (incluindo o Edge padrão) sintetiza e reproduz cada frase ao completar

O mesmo pipeline roda na CLI clássica, TUI e app desktop. Em conversa de voz no desktop o texto da resposta é alimentado **ao vivo** num WebSocket de fala por resposta conforme o modelo gera, então a fala sobrepõe a geração — um socket e um relógio de áudio por resposta, sem gaps de conexão por frase.

### Desktop remoto: voz client-direct (caminho de menor hop) {#desktop-remote-client-direct-voice-lowest-hop-path}

Quando o Hermes Desktop está conectado a um **gateway remoto**, o áudio não precisa ser retransmitido pelo gateway. No início da sessão de voz, o desktop busca as configurações STT/TTS resolvidas do profile ativo (provider, model, language/voice e credencial) no gateway pelo canal REST autenticado (`GET /api/audio/voice-config`) e então chama os providers **diretamente**:

- **Ditado / entrada de voz:** a gravação do microfone vai direto do seu desktop para o provider STT do profile; só o *texto* resultante é enviado ao gateway como prompt.
- **Respostas faladas:** o texto da resposta já faz stream para o desktop pelo socket de chat, então o desktop sintetiza localmente com o provider TTS do profile e reproduz — o link com o gateway nunca carrega áudio.

Não há nada a configurar no client: o profile com o qual você está falando é a única fonte de verdade para providers e chaves, exatamente como se o gateway tivesse feito o trabalho. As chaves ficam só na memória do desktop durante a sessão — nunca são gravadas em disco no client.

Providers que só podem rodar no host do gateway (whisper local, TTS `edge`, command providers, plugins) fazem fallback automaticamente para o caminho de relay (`/api/audio/transcribe` e o WebSocket de fala), assim como qualquer backend antigo sem o endpoint. Para forçar o relay para todo provider, defina:

```yaml
voice:
  client_direct: false
```

Suporte wire client-direct: OpenAI (incl. áudio gerenciado pela Nous), Groq, Mistral e DeepInfra via shapes compatíveis com OpenAI, xAI Grok STT, e ElevenLabs STT + TTS. xAI configurado via OAuth permanece no relay (o bearer OAuth renova no servidor).

### Barge-in {#barge-in}

Você pode interromper o agente em QUALQUER ponto do turno — o microfone fica ativo desde o momento em que você termina de falar até a resposta ter tocado por completo (full duplex):

- **Interromper enquanto pensa** — no modo de voz contínuo, falar durante geração LLM (antes de qualquer áudio tocar) interrompe o turno em voo e sua interjeição vira a próxima mensagem, igual a digitar sobre um turno em execução.
- **Falar por cima** — falar enquanto a resposta do agente toca corta a reprodução no momento em que você começa a falar e envia o que disse. O detector calibra o noise floor contra a *sala quiet* no início do turno (nunca contra a reprodução), então bleed do alto-falante não o deixa surdo e fala normal dispara de forma confiável.
- **Digitar ou pressionar a tecla de gravação** — enviar nova mensagem ou acionar push-to-talk para a reprodução instantaneamente em toda superfície.
- **Dizer "stop"** — a frase de stop funciona nas duas fases: no meio da geração interrompe o turno E encerra o chat de voz; no meio da reprodução corta a fala e encerra o chat.

Ajuste (config.yaml): `voice.barge_in: false` desabilita; `voice.barge_in_threshold_multiplier` (padrão `3.0`) escala o gatilho de fala sobre o floor da sala quiet; `voice.barge_in_grace_seconds` (padrão `0.5`) suprime trips logo após a reprodução começar. Defina `HERMES_VOICE_DEBUG=1` para stream de diagnósticos VAD por bloco (floor calibrado, RMS, decisões de trip) para stderr para tuning ao vivo.

O agente **sabe** que foi interrompido: a próxima mensagem traz uma nota curta dizendo ao modelo que a resposta falada foi cortada, para reagir naturalmente ("rude!") ou retomar de onde parou em vez de ser oblivious.

### Filtro de Alucinação {#hallucination-filter}

Whisper às vezes gera texto fantasma de silêncio ou ruído de fundo ("Thank you for watching", "Subscribe", etc.). O agente filtra isso com um conjunto de 26 frases conhecidas de alucinação em vários idiomas, mais um padrão regex que pega variações repetitivas.

---

## Resposta de Voz no Gateway (Telegram & Discord) {#gateway-voice-reply-telegram-discord}

Se ainda não configurou seus bots de mensagens, veja os guias por plataforma:
- [Guia de Setup do Telegram](../messaging/telegram.md)
- [Guia de Setup do Discord](../messaging/discord.md)

Inicie o gateway para conectar às plataformas de mensagens:

```bash
hermes gateway        # Iniciar o gateway (conecta às plataformas configuradas)
hermes gateway setup  # Assistente interativo de setup para configuração inicial
```

### Discord: Canais vs DMs {#discord-channels-vs-dms}

O bot suporta dois modos de interação no Discord:

| Modo | Como Falar | Menção Obrigatória | Setup |
|------|------------|-----------------|-------|
| **Direct Message (DM)** | Abra o perfil do bot → "Message" | Não | Funciona imediatamente |
| **Canal de Servidor** | Digite em canal de texto onde o bot está presente | Sim (`@botname`) | Bot deve ser convidado ao servidor |

**DM (recomendado para uso pessoal):** Abra um DM com o bot e digite — sem @mention. Respostas de voz e todos os comandos funcionam igual aos canais.

**Canais de servidor:** O bot só responde quando você @mention (ex.: `@hermesbyt4 hello`). Certifique-se de selecionar o **usuário bot** no popup de menção, não a role com o mesmo nome.

:::tip
Para desabilitar a exigência de menção em canais de servidor, adicione em `~/.hermes/.env`:
```bash
DISCORD_REQUIRE_MENTION=false
```
Ou defina canais específicos como free-response (sem menção):
```bash
DISCORD_FREE_RESPONSE_CHANNELS=123456789,987654321
```
:::

### Comandos {#commands}

Estes funcionam no Telegram e Discord (DMs e canais de texto):

```
/voice          Alternar modo de voz ligado/desligado
/voice on       Respostas de voz só quando você envia mensagem de voz
/voice tts      Respostas de voz para TODAS as mensagens
/voice off      Desabilitar respostas de voz
/voice status   Mostrar configuração atual
```

### Modos {#modes}

| Modo | Comando | Comportamento |
|------|---------|----------|
| `off` | `/voice off` | Só texto (padrão) |
| `voice_only` | `/voice on` | Fala resposta só quando você envia mensagem de voz |
| `all` | `/voice tts` | Fala resposta para toda mensagem |

A configuração de modo de voz persiste entre reinícios do gateway.

### Entrega por Plataforma {#platform-delivery}

| Plataforma | Formato | Notas |
|----------|--------|-------|
| **Telegram** | Bolha de voz (Opus/OGG) | Toca inline no chat. ffmpeg converte MP3 → Opus se necessário |
| **Discord** | Bolha de voz nativa (Opus/OGG) | Toca inline como mensagem de voz de usuário. Fallback para anexo se a API de bolha falhar |

---

## Canais de Voz do Discord {#discord-voice-channels}

O recurso de voz mais imersivo: o bot entra num canal de voz do Discord, escuta usuários falando, transcreve a fala, processa pelo agente e fala a resposta de volta no canal de voz.

### Setup {#setup}

#### 1. Permissões do Bot Discord {#1-discord-bot-permissions}

Se já tem um bot Discord configurado para texto (veja [Guia de Setup do Discord](../messaging/discord.md)), precisa adicionar permissões de voz.

Vá ao [Discord Developer Portal](https://discord.com/developers/applications) → sua aplicação → **Installation** → **Default Install Settings** → **Guild Install**:

**Adicione estas permissões às permissões de texto existentes:**

| Permissão | Propósito | Obrigatório |
|-----------|---------|----------|
| **Connect** | Entrar em canais de voz | Sim |
| **Speak** | Reproduzir áudio TTS em canais de voz | Sim |
| **Use Voice Activity** | Detectar quando usuários estão falando | Recomendado |

**Integer de Permissões Atualizado:**

| Nível | Integer | O Que Inclui |
|-------|---------|----------------|
| Só texto | `309237763136` | View Channels, Send Messages, Read History, Embeds, Attachments, Threads, Reactions, Create Public Threads |
| Texto + Voz | `309240908864` | Tudo acima + Connect, Speak |

**Re-convide o bot** com a URL de permissões atualizada:

```
https://discord.com/oauth2/authorize?client_id=YOUR_APP_ID&scope=bot+applications.commands&permissions=309240908864
```

Substitua `YOUR_APP_ID` pelo Application ID do Developer Portal.

:::warning
Re-convidar o bot a um servidor onde já está atualiza permissões sem removê-lo. Você não perde dados ou configuração.
:::

#### 2. Privileged Gateway Intents {#2-privileged-gateway-intents}

No [Developer Portal](https://discord.com/developers/applications) → sua aplicação → **Bot** → **Privileged Gateway Intents**, habilite os três:

| Intent | Propósito |
|--------|---------|
| **Presence Intent** | Detectar status online/offline do usuário |
| **Server Members Intent** | Resolver usernames em `DISCORD_ALLOWED_USERS` para IDs numéricos (condicional) |
| **Message Content Intent** | Ler conteúdo de mensagens de texto em canais |

**Message Content Intent** é obrigatório. **Server Members Intent** só é necessário se sua lista `DISCORD_ALLOWED_USERS` usa usernames — se usar IDs numéricos, pode deixar OFF. Mapeamento SSRC → user_id de canal de voz vem do opcode SPEAKING do Discord no websocket de voz e **não** exige Server Members Intent.

#### 3. Codec Opus {#3-opus-codec}

A biblioteca codec Opus deve estar instalada na máquina que roda o gateway:

```bash
# macOS (Homebrew)
brew install opus

# Ubuntu/Debian
sudo apt install libopus0
```

O bot carrega automaticamente o codec de:
- **macOS:** `/opt/homebrew/lib/libopus.dylib`
- **Linux:** `libopus.so.0`

#### 4. Variáveis de Ambiente {#4-environment-variables}

```bash
# ~/.hermes/.env

# Bot Discord (já configurado para texto)
DISCORD_BOT_TOKEN=your-bot-token
DISCORD_ALLOWED_USERS=your-user-id

# STT — provedor local não precisa de key (pip install faster-whisper)
# GROQ_API_KEY=your-key            # Alternativa: cloud, rápido, tier grátis

# TTS — opcional. Edge TTS e NeuTTS não precisam de key.
# ELEVENLABS_API_KEY=***      # Qualidade premium
# VOICE_TOOLS_OPENAI_KEY=***  # OpenAI TTS / Whisper
```

### Iniciar o Gateway {#start-the-gateway}

```bash
hermes gateway        # Iniciar com configuração existente
```

O bot deve ficar online no Discord em alguns segundos.

### Comandos {#commands-1}

Use estes no canal de texto do Discord onde o bot está presente:

```
/voice join      Bot entra no seu canal de voz atual
/voice channel   Alias para /voice join
/voice leave     Bot desconecta do canal de voz
/voice status    Mostrar modo de voz e canal conectado
```

:::info
Você deve estar num canal de voz antes de rodar `/voice join`. O bot entra no mesmo VC que você.
:::

### Como Funciona {#how-it-works-1}

Quando o bot entra num canal de voz, ele:

1. **Escuta** o stream de áudio de cada usuário independentemente
2. **Detecta silêncio** — 1,5s de silêncio após pelo menos 0,5s de fala dispara processamento
3. **Transcreve** o áudio via Whisper STT (local, Groq ou OpenAI)
4. **Processa** pelo pipeline completo do agente (sessão, ferramentas, memória)
5. **Fala** a resposta de volta no canal de voz via TTS

### Integração com Canal de Texto {#text-channel-integration}

Quando o bot está num canal de voz:

- Transcrições aparecem no canal de texto: `[Voice] @user: what you said`
- Respostas do agente são enviadas como texto no canal E faladas no VC
- O canal de texto é onde `/voice join` foi emitido

### Prevenção de Eco {#echo-prevention}

O bot pausa automaticamente o listener de áudio enquanto reproduz respostas TTS, evitando ouvir e reprocessar a própria saída.

### Controle de Acesso {#access-control}

Apenas usuários listados em `DISCORD_ALLOWED_USERS` podem interagir por voz. Áudio de outros usuários é ignorado silenciosamente.

```bash
# ~/.hermes/.env
DISCORD_ALLOWED_USERS=284102345871466496
```

---

## Referência de Configuração {#configuration-reference}

### config.yaml {#configyaml}

```yaml
# Gravação de voz (CLI)
voice:
  record_key: "ctrl+b"            # Tecla para iniciar/parar gravação
  max_recording_seconds: 120       # Duração máxima de gravação
  auto_tts: false                  # Habilitar TTS automaticamente ao iniciar modo de voz
  beep_enabled: true               # Tocar beeps de início/fim de gravação
  silence_threshold: 200           # Nível RMS (0-32767) abaixo do qual conta como silêncio
  silence_duration: 3.0            # Segundos de silêncio antes de auto-stop
  stop_phrases: ["stop"]           # Dizer exatamente uma destas encerra o chat de voz; [] desabilita

# Speech-to-Text
stt:
  enabled: true                     # false para pular auto-transcrição —
                                    # o gateway ainda cacheia o arquivo de áudio e
                                    # passa seu caminho ao agente como parte da
                                    # mensagem inbound, útil para pipelines customizados
                                    # (diarization, alignment, archival, etc.)
  provider: "local"                  # "local" (grátis) | "groq" | "openai" | "mistral" | "xai"
  local:
    model: "base"                    # tiny, base, small, medium, large-v3
    language: ""                     # hint ISO-639-1 opcional; blank = usar HERMES_LOCAL_STT_LANGUAGE se definido, senão auto-detect
  groq:
    language: ""                     # hint ISO-639-1 opcional; blank = usar HERMES_LOCAL_STT_LANGUAGE se definido, senão auto-detect
  # model: "whisper-1"              # Legacy: usado quando provider não está definido

# Text-to-Speech
tts:
  provider: "edge"                 # "edge" (grátis) | "elevenlabs" | "openai" | "neutts" | "minimax" | "mistral" | "gemini" | "xai" | "kittentts" | "piper"
  edge:
    voice: "en-US-AriaNeural"      # 322 vozes, 74 idiomas
  elevenlabs:
    voice_id: "pNInz6obpgDQGcFmaJgB"    # Adam
    model_id: "eleven_multilingual_v2"
  openai:
    model: "gpt-4o-mini-tts"
    voice: "alloy"                 # alloy, echo, fable, onyx, nova, shimmer
    base_url: "https://api.openai.com/v1"  # opcional: override para endpoints self-hosted ou compatíveis OpenAI
    # A ferramenta `text_to_speech` aceita argumento opcional por chamada `instructions`
    # (tom, emoção, ritmo, sotaque, sussurro) encaminhado a `gpt-4o-mini-tts` e
    # servidores de voice-design compatíveis OpenAI
    # (ex. Qwen3-TTS-VoiceDesign via oMLX). Veja o guia voice-design da OpenAI:
    # https://platform.openai.com/docs/guides/text-to-speech
  neutts:
    ref_audio: ''
    ref_text: ''
    model: neuphonic/neutts-air-q4-gguf
    device: cpu
```

### Variáveis de Ambiente {#environment-variables}

```bash
# Provedores Speech-to-Text (local não precisa de key)
# pip install faster-whisper        # STT local grátis — sem API key
GROQ_API_KEY=...                    # Groq Whisper (rápido, tier grátis)
VOICE_TOOLS_OPENAI_KEY=...         # OpenAI Whisper (pago)

# Overrides avançados STT (opcional)
STT_GROQ_MODEL=whisper-large-v3-turbo    # Override modelo STT Groq padrão
STT_OPENAI_MODEL=whisper-1               # Override modelo STT OpenAI padrão
GROQ_BASE_URL=https://api.groq.com/openai/v1     # Endpoint Groq customizado
STT_OPENAI_BASE_URL=https://api.openai.com/v1    # Endpoint STT OpenAI customizado

# Provedores Text-to-Speech (Edge TTS e NeuTTS não precisam de key)
ELEVENLABS_API_KEY=***             # ElevenLabs (qualidade premium)
# VOICE_TOOLS_OPENAI_KEY acima também habilita OpenAI TTS

# Canal de voz Discord
DISCORD_BOT_TOKEN=...
DISCORD_ALLOWED_USERS=...
```

### Comparação de Provedores STT {#stt-provider-comparison}

| Provedor | Modelo | Velocidade | Qualidade | Custo | API Key |
|----------|-------|-------|---------|------|---------|
| **Local** | `base` | Rápido (depende de CPU/GPU) | Boa | Grátis | Não |
| **Local** | `small` | Médio | Melhor | Grátis | Não |
| **Local** | `large-v3` | Lento | Melhor | Grátis | Não |
| **Groq** | `whisper-large-v3-turbo` | Muito rápido (~0,5s) | Boa | Tier grátis | Sim |
| **Groq** | `whisper-large-v3` | Rápido (~1s) | Melhor | Tier grátis | Sim |
| **OpenAI** | `whisper-1` | Rápido (~1s) | Boa | Pago | Sim |
| **OpenAI** | `gpt-4o-transcribe` | Médio (~2s) | Melhor | Pago | Sim |
| **OpenAI** | `gpt-transcribe` | Rápido | Melhor | Pago ($0.0045/min) | Sim |
| **Mistral** | `voxtral-mini-latest` | Rápido | Boa | Pago | Sim |
| **xAI** | `grok-stt` | Rápido | Boa | Pago | Sim |

Prioridade de provedor (fallback automático): **local** > **groq** > **openai**

### Comparação de Provedores TTS {#tts-provider-comparison}

| Provedor | Qualidade | Custo | Latência | Key Obrigatória |
|----------|---------|------|---------|-------------|
| **Edge TTS** | Boa | Grátis | ~1s | Não |
| **ElevenLabs** | Excelente | Pago | ~2s | Sim |
| **OpenAI TTS** | Boa | Pago | ~1,5s | Sim |
| **NeuTTS** | Boa | Grátis | Depende de CPU/GPU | Não |

NeuTTS usa o bloco de config `tts.neutts` acima.

Para `openai`, a ferramenta `text_to_speech` aceita argumento opcional `instructions`
que desbloqueia a capacidade de voice-design do `gpt-4o-mini-tts` (tom,
emoção, ritmo, sotaque, sussurro). O mesmo campo também roteia para
servidores de voice-design compatíveis OpenAI montados via `tts.openai.base_url`
(ex. Qwen3-TTS-VoiceDesign via oMLX).

---

## Solução de Problemas {#troubleshooting}

### "No audio device found" (CLI) {#no-audio-device-found-cli}

PortAudio não está instalado:

```bash
brew install portaudio    # macOS
sudo apt install portaudio19-dev  # Ubuntu
```

Se você roda Hermes dentro de Docker num desktop Linux, o container também precisa de acesso ao socket de áudio do host. Veja as notas da [ponte de áudio Docker](/user-guide/docker#optional-linux-desktop-audio-bridge) para setup compatível PulseAudio/PipeWire.

### Bot não responde em canais de servidor Discord {#bot-doesnt-respond-in-discord-server-channels}

O bot exige @mention por padrão em canais de servidor. Certifique-se de:

1. Digitar `@` e selecionar o **usuário bot** (com #discriminator), não a **role** com o mesmo nome
2. Ou usar DMs — sem menção necessária
3. Ou definir `DISCORD_REQUIRE_MENTION=false` em `~/.hermes/.env`

### Bot entra no VC mas não me ouve {#bot-joins-vc-but-doesnt-hear-me}

- Verifique se seu Discord user ID está em `DISCORD_ALLOWED_USERS`
- Certifique-se de não estar mutado no Discord
- O bot precisa de um evento SPEAKING do Discord antes de mapear seu áudio — comece a falar alguns segundos após entrar

### Bot me ouve mas não responde {#bot-hears-me-but-doesnt-respond}

- Verifique se STT está disponível: instale `faster-whisper` (sem key) ou defina `GROQ_API_KEY` / `VOICE_TOOLS_OPENAI_KEY`
- Verifique se o modelo LLM está configurado e acessível
- Revise logs do gateway: `tail -f ~/.hermes/logs/gateway.log`

### Bot responde em texto mas não no canal de voz {#bot-responds-in-text-but-not-in-voice-channel}

- Provedor TTS pode estar falhando — verifique API key e quota
- Edge TTS (grátis, sem key) é o fallback padrão
- Verifique logs por erros TTS

### Whisper retorna texto lixo {#whisper-returns-garbage-text}

O filtro de alucinação pega a maioria dos casos automaticamente. Se ainda receber transcrições fantasma:

- Use ambiente mais silencioso
- Ajuste `silence_threshold` na config (maior = menos sensível)
- Tente outro modelo STT
