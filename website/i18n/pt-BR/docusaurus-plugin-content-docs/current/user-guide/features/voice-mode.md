---
sidebar_position: 10
title: "Modo de voz"
description: "Conversas por voz em tempo real com o Hermes Agent — CLI, Telegram, Discord (DMs, canais de texto e canais de voz)"
---

# Modo de voz

O Hermes Agent suporta interação completa por voz no CLI e plataformas de mensagens. Fale com o agente usando seu microfone, ouça respostas faladas e tenha conversas ao vivo em canais de voz do Discord.

Se você quer um walkthrough prático de setup com configurações recomendadas e padrões de uso reais, veja [Use Voice Mode with Hermes](/guides/use-voice-mode-with-hermes).

## Pré-requisitos {#prerequisites}

Antes de usar recursos de voz, certifique-se de ter:

1. **Hermes Agent instalado** — via script de instalação (veja [Installation](/getting-started/installation))
2. **Um provider LLM configurado** — rode `hermes model` ou defina suas credenciais de provider preferidas em `~/.hermes/.env`
3. **Um setup base funcionando** — rode `hermes` para verificar que o agente responde a texto antes de habilitar voz

:::tip
O diretório `~/.hermes/` e o `config.yaml` padrão são criados automaticamente na primeira vez que você roda `hermes`. Você só precisa criar `~/.hermes/.env` manualmente para chaves de API.
:::

:::tip Nous Portal cobre ambos
Uma assinatura paga do [Nous Portal](/user-guide/features/tool-gateway) fornece o LLM (passo 2) **e** OpenAI TTS via Tool Gateway — sem chave OpenAI separada. Em instalação nova, `hermes setup --portal` configura os dois de uma vez.
:::

## Visão geral {#overview}

| Recurso | Plataforma | Descrição |
|---------|----------|-------------|
| **Voz interativa** | CLI | Pressione Ctrl+B para gravar, o agente detecta silêncio automaticamente e responde |
| **Resposta automática por voz** | Telegram, Discord | O agente envia áudio falado junto com respostas em texto |
| **Canal de voz** | Discord | Bot entra no VC, escuta usuários falando, fala respostas de volta |

## Requisitos {#requirements}

### Pacotes Python {#python-packages}

```bash
# CLI voice mode (microphone + audio playback)
cd ~/.hermes/hermes-agent && uv pip install -e ".[voice]"

# Discord + Telegram messaging (includes discord.py[voice] for VC support)
cd ~/.hermes/hermes-agent && uv pip install -e ".[messaging]"

# Premium TTS (ElevenLabs)
cd ~/.hermes/hermes-agent && uv pip install -e ".[tts-premium]"

# Local TTS (NeuTTS, optional)
python -m pip install -U neutts[all]

# Everything at once
cd ~/.hermes/hermes-agent && uv pip install -e ".[all]"
```

| Extra | Pacotes | Necessário para |
|-------|---------|-----------------|
| `voice` | `sounddevice`, `numpy` | Modo de voz CLI |
| `messaging` | `discord.py[voice]`, `python-telegram-bot`, `aiohttp` | Bots Discord e Telegram |
| `tts-premium` | `elevenlabs` | Provider ElevenLabs TTS |

Provider TTS local opcional: instale `neutts` separadamente com `python -m pip install -U neutts[all]`. No primeiro uso baixa o modelo automaticamente.

:::info
`discord.py[voice]` instala **PyNaCl** (para criptografia de voz) e **opus bindings** automaticamente. Isso é necessário para suporte a canal de voz do Discord.
:::

### Dependências de sistema {#system-dependencies}

```bash
# macOS
brew install portaudio ffmpeg opus
brew install espeak-ng   # for NeuTTS

# Ubuntu/Debian
sudo apt install portaudio19-dev ffmpeg libopus0
sudo apt install espeak-ng   # for NeuTTS
```

| Dependência | Propósito | Necessário para |
|-----------|---------|-------------|
| **PortAudio** | Entrada de microfone e reprodução de áudio | Modo de voz CLI |
| **ffmpeg** | Conversão de formato de áudio (MP3 → Opus, PCM → WAV) | Todas as plataformas |
| **Opus** | Codec de voz Discord | Canais de voz Discord |
| **espeak-ng** | Backend phonemizer | Provider NeuTTS local |

### Chaves de API {#api-keys}

Adicione em `~/.hermes/.env`:

```bash
# Speech-to-Text — local provider needs NO key at all
# pip install faster-whisper          # Free, runs locally, recommended
GROQ_API_KEY=your-key                 # Groq Whisper — fast, free tier (cloud)
VOICE_TOOLS_OPENAI_KEY=your-key       # OpenAI Whisper — paid (cloud)

# Text-to-Speech (optional — Edge TTS and NeuTTS work without any key)
ELEVENLABS_API_KEY=***           # ElevenLabs — premium quality
# VOICE_TOOLS_OPENAI_KEY above also enables OpenAI TTS
```

:::tip
Se `faster-whisper` estiver instalado, o modo de voz funciona com **zero chaves de API** para STT. O modelo (~150 MB para `base`) baixa automaticamente no primeiro uso.
:::

---

## Modo de voz CLI {#cli-voice-mode}

O modo de voz está disponível tanto no **CLI clássico** (`hermes chat`) quanto no **TUI** (`hermes --tui`). O comportamento é idêntico nos dois — mesmos slash commands, mesma detecção de silêncio VAD, mesmo streaming TTS, mesmo filtro de alucinação. O TUI adicionalmente encaminha logs forenses de crash para `~/.hermes/logs/` para que falhas de push-to-talk em backends de áudio exóticos possam ser reportadas com stack trace completo em vez de desaparecer silenciosamente.

### Início rápido {#quick-start}

Inicie o CLI e habilite o modo de voz:

```bash
hermes                # Start the interactive CLI
```

Depois use estes comandos dentro do CLI:

```
/voice          Toggle voice mode on/off
/voice on       Enable voice mode
/voice off      Disable voice mode
/voice tts      Toggle TTS output
/voice status   Show current state
```

### Como funciona {#how-it-works}

1. Inicie o CLI com `hermes` e habilite o modo de voz com `/voice on`
2. **Pressione Ctrl+B** — um beep toca (880Hz), a gravação começa
3. **Fale** — uma barra de nível de áudio ao vivo mostra sua entrada: `● [▁▂▃▅▇▇▅▂] ❯`
4. **Pare de falar** — após 3 segundos de silêncio, a gravação para automaticamente
5. **Dois beeps** tocam (660Hz) confirmando que a gravação terminou
6. O áudio é transcrito via Whisper e enviado ao agente
7. Se TTS estiver habilitado, a resposta do agente é falada em voz alta
8. A gravação **reinicia automaticamente** — fale de novo sem pressionar nenhuma tecla

Esse loop continua até você pressionar **Ctrl+B** durante a gravação (sai do modo contínuo) ou 3 gravações consecutivas não detectarem fala.

:::tip
A tecla de gravação é configurável via `voice.record_key` em `~/.hermes/config.yaml` (padrão: `ctrl+b`).
:::

### Detecção de silêncio {#silence-detection}

Algoritmo em duas etapas detecta quando você terminou de falar:

1. **Confirmação de fala** — espera áudio acima do limiar RMS (200) por pelo menos 0.3s, tolerando quedas breves entre sílabas
2. **Detecção de fim** — uma vez confirmada a fala, dispara após 3.0 segundos de silêncio contínuo

Se nenhuma fala for detectada por 15 segundos, a gravação para automaticamente.

Tanto `silence_threshold` quanto `silence_duration` são configuráveis em `config.yaml`. Você também pode desabilitar os beeps de início/fim de gravação com `voice.beep_enabled: false`.

### Streaming TTS {#streaming-tts}

Quando TTS está habilitado, o agente fala sua resposta **frase a frase** conforme gera texto — você não espera a resposta completa:

1. Bufferiza deltas de texto em frases completas (mín. 20 chars)
2. Remove formatação markdown e blocos `<think>`
3. Gera e reproduz áudio por frase em tempo real

### Filtro de alucinação {#hallucination-filter}

Whisper às vezes gera texto fantasma de silêncio ou ruído de fundo ("Thank you for watching", "Subscribe", etc.). O agente filtra isso usando um conjunto de 26 frases conhecidas de alucinação em vários idiomas, mais um padrão regex que captura variações repetitivas.

---

## Resposta por voz no gateway (Telegram e Discord) {#gateway-voice-reply-telegram--discord}

Se você ainda não configurou seus bots de mensagens, veja os guias específicos de plataforma:
- [Telegram Setup Guide](../messaging/telegram.md)
- [Discord Setup Guide](../messaging/discord.md)

Inicie o gateway para conectar às suas plataformas de mensagens:

```bash
hermes gateway        # Start the gateway (connects to configured platforms)
hermes gateway setup  # Interactive setup wizard for first-time configuration
```

### Discord: canais vs DMs {#discord-channels-vs-dms}

O bot suporta dois modos de interação no Discord:

| Modo | Como falar | Menção necessária | Setup |
|------|------------|-----------------|-------|
| **Direct Message (DM)** | Abra o perfil do bot → "Message" | Não | Funciona imediatamente |
| **Canal de servidor** | Digite em um canal de texto onde o bot está presente | Sim (`@botname`) | Bot deve ser convidado ao servidor |

**DM (recomendado para uso pessoal):** Basta abrir um DM com o bot e digitar — sem @mention necessário. Respostas por voz e todos os comandos funcionam igual que em canais.

**Canais de servidor:** O bot só responde quando você @mention (ex.: `@hermesbyt4 hello`). Certifique-se de selecionar o **usuário bot** no popup de menção, não o role com o mesmo nome.

:::tip
Para desabilitar a exigência de menção em canais de servidor, adicione em `~/.hermes/.env`:
```bash
DISCORD_REQUIRE_MENTION=false
```
Ou defina canais específicos como free-response (sem menção necessária):
```bash
DISCORD_FREE_RESPONSE_CHANNELS=123456789,987654321
```
:::

### Comandos {#commands}

Estes funcionam tanto no Telegram quanto no Discord (DMs e canais de texto):

```
/voice          Toggle voice mode on/off
/voice on       Voice replies only when you send a voice message
/voice tts      Voice replies for ALL messages
/voice off      Disable voice replies
/voice status   Show current setting
```

### Modos {#modes}

| Modo | Comando | Comportamento |
|------|---------|----------|
| `off` | `/voice off` | Apenas texto (padrão) |
| `voice_only` | `/voice on` | Fala resposta apenas quando você envia mensagem de voz |
| `all` | `/voice tts` | Fala resposta para toda mensagem |

A configuração de modo de voz persiste entre reinícios do gateway.

### Entrega por plataforma {#platform-delivery}

| Plataforma | Formato | Notas |
|----------|--------|-------|
| **Telegram** | Voice bubble (Opus/OGG) | Toca inline no chat. ffmpeg converte MP3 → Opus se necessário |
| **Discord** | Voice bubble nativo (Opus/OGG) | Toca inline como mensagem de voz de usuário. Fallback para anexo de arquivo se API de voice bubble falhar |

---

## Canais de voz Discord {#discord-voice-channels}

O recurso de voz mais imersivo: o bot entra em um canal de voz Discord, escuta usuários falando, transcreve a fala, processa pelo agente e fala a resposta de volta no canal de voz.

### Setup {#setup}

#### 1. Permissões do bot Discord {#1-discord-bot-permissions}

Se você já tem um bot Discord configurado para texto (veja [Discord Setup Guide](../messaging/discord.md)), precisa adicionar permissões de voz.

Vá ao [Discord Developer Portal](https://discord.com/developers/applications) → sua aplicação → **Installation** → **Default Install Settings** → **Guild Install**:

**Adicione estas permissões às permissões de texto existentes:**

| Permissão | Propósito | Obrigatório |
|-----------|---------|----------|
| **Connect** | Entrar em canais de voz | Sim |
| **Speak** | Reproduzir áudio TTS em canais de voz | Sim |
| **Use Voice Activity** | Detectar quando usuários estão falando | Recomendado |

**Integer de permissões atualizado:**

| Nível | Integer | O que inclui |
|-------|---------|----------------|
| Só texto | `309237763136` | View Channels, Send Messages, Read History, Embeds, Attachments, Threads, Reactions, Create Public Threads |
| Texto + voz | `309240908864` | Tudo acima + Connect, Speak |

**Re-convide o bot** com a URL de permissões atualizada:

```
https://discord.com/oauth2/authorize?client_id=YOUR_APP_ID&scope=bot+applications.commands&permissions=309240908864
```

Substitua `YOUR_APP_ID` pelo Application ID do Developer Portal.

:::warning
Re-convidar o bot a um servidor onde ele já está atualizará suas permissões sem removê-lo. Você não perderá dados ou configuração.
:::

#### 2. Privileged Gateway Intents {#2-privileged-gateway-intents}

No [Developer Portal](https://discord.com/developers/applications) → sua aplicação → **Bot** → **Privileged Gateway Intents**, habilite os três:

| Intent | Propósito |
|--------|---------|
| **Presence Intent** | Detectar status online/offline do usuário |
| **Server Members Intent** | Resolver usernames em `DISCORD_ALLOWED_USERS` para IDs numéricos (condicional) |
| **Message Content Intent** | Ler conteúdo de mensagens de texto em canais |

**Message Content Intent** é obrigatório. **Server Members Intent** só é necessário se sua lista `DISCORD_ALLOWED_USERS` usa usernames — se você usa IDs numéricos de usuário, pode deixá-lo OFF. O mapeamento SSRC → user_id de canal de voz vem do opcode SPEAKING do Discord no websocket de voz e **não** requer Server Members Intent.

#### 3. Codec Opus {#3-opus-codec}

A biblioteca codec Opus deve estar instalada na máquina que roda o gateway:

```bash
# macOS (Homebrew)
brew install opus

# Ubuntu/Debian
sudo apt install libopus0
```

O bot carrega o codec automaticamente de:
- **macOS:** `/opt/homebrew/lib/libopus.dylib`
- **Linux:** `libopus.so.0`

#### 4. Variáveis de ambiente {#4-environment-variables}

```bash
# ~/.hermes/.env

# Discord bot (already configured for text)
DISCORD_BOT_TOKEN=your-bot-token
DISCORD_ALLOWED_USERS=your-user-id

# STT — local provider needs no key (pip install faster-whisper)
# GROQ_API_KEY=your-key            # Alternative: cloud-based, fast, free tier

# TTS — optional. Edge TTS and NeuTTS need no key.
# ELEVENLABS_API_KEY=***      # Premium quality
# VOICE_TOOLS_OPENAI_KEY=***  # OpenAI TTS / Whisper
```

### Inicie o gateway {#start-the-gateway}

```bash
hermes gateway        # Start with existing configuration
```

O bot deve ficar online no Discord em alguns segundos.

### Comandos {#commands-1}

Use estes no canal de texto Discord onde o bot está presente:

```
/voice join      Bot joins your current voice channel
/voice channel   Alias for /voice join
/voice leave     Bot disconnects from voice channel
/voice status    Show voice mode and connected channel
```

:::info
Você deve estar em um canal de voz antes de rodar `/voice join`. O bot entra no mesmo VC em que você está.
:::

### Como funciona {#how-it-works-1}

Quando o bot entra em um canal de voz, ele:

1. **Escuta** o stream de áudio de cada usuário independentemente
2. **Detecta silêncio** — 1.5s de silêncio após pelo menos 0.5s de fala dispara o processamento
3. **Transcreve** o áudio via Whisper STT (local, Groq ou OpenAI)
4. **Processa** pelo pipeline completo do agente (sessão, ferramentas, memória)
5. **Fala** a resposta de volta no canal de voz via TTS

### Integração com canal de texto {#text-channel-integration}

Quando o bot está em um canal de voz:

- Transcrições aparecem no canal de texto: `[Voice] @user: what you said`
- Respostas do agente são enviadas como texto no canal E faladas no VC
- O canal de texto é aquele onde `/voice join` foi emitido

### Prevenção de eco {#echo-prevention}

O bot pausa automaticamente seu listener de áudio enquanto reproduz respostas TTS, evitando ouvir e reprocessar sua própria saída.

### Controle de acesso {#access-control}

Apenas usuários listados em `DISCORD_ALLOWED_USERS` podem interagir por voz. Áudio de outros usuários é ignorado silenciosamente.

```bash
# ~/.hermes/.env
DISCORD_ALLOWED_USERS=284102345871466496
```

---

## Referência de configuração {#configuration-reference}

### config.yaml {#configyaml}

```yaml
# Voice recording (CLI)
voice:
  record_key: "ctrl+b"            # Key to start/stop recording
  max_recording_seconds: 120       # Maximum recording length
  auto_tts: false                  # Auto-enable TTS when voice mode starts
  beep_enabled: true               # Play record start/stop beeps
  silence_threshold: 200           # RMS level (0-32767) below which counts as silence
  silence_duration: 3.0            # Seconds of silence before auto-stop

# Speech-to-Text
stt:
  enabled: true                     # set to false to skip auto-transcription —
                                    # the gateway still caches the audio file and
                                    # passes its path to the agent as part of the
                                    # inbound message, useful for custom pipelines
                                    # (diarization, alignment, archival, etc.)
  provider: "local"                  # "local" (free) | "groq" | "openai" | "mistral" | "xai"
  local:
    model: "base"                    # tiny, base, small, medium, large-v3
  # model: "whisper-1"              # Legacy: used when provider is not set

# Text-to-Speech
tts:
  provider: "edge"                 # "edge" (free) | "elevenlabs" | "openai" | "neutts" | "minimax" | "mistral" | "gemini" | "xai" | "kittentts" | "piper"
  edge:
    voice: "en-US-AriaNeural"      # 322 voices, 74 languages
  elevenlabs:
    voice_id: "pNInz6obpgDQGcFmaJgB"    # Adam
    model_id: "eleven_multilingual_v2"
  openai:
    model: "gpt-4o-mini-tts"
    voice: "alloy"                 # alloy, echo, fable, onyx, nova, shimmer
    base_url: "https://api.openai.com/v1"  # optional: override for self-hosted or OpenAI-compatible endpoints
  neutts:
    ref_audio: ''
    ref_text: ''
    model: neuphonic/neutts-air-q4-gguf
    device: cpu
```

### Variáveis de ambiente {#environment-variables}

```bash
# Speech-to-Text providers (local needs no key)
# pip install faster-whisper        # Free local STT — no API key needed
GROQ_API_KEY=...                    # Groq Whisper (fast, free tier)
VOICE_TOOLS_OPENAI_KEY=...         # OpenAI Whisper (paid)

# STT advanced overrides (optional)
STT_GROQ_MODEL=whisper-large-v3-turbo    # Override default Groq STT model
STT_OPENAI_MODEL=whisper-1               # Override default OpenAI STT model
GROQ_BASE_URL=https://api.groq.com/openai/v1     # Custom Groq endpoint
STT_OPENAI_BASE_URL=https://api.openai.com/v1    # Custom OpenAI STT endpoint

# Text-to-Speech providers (Edge TTS and NeuTTS need no key)
ELEVENLABS_API_KEY=***             # ElevenLabs (premium quality)
# VOICE_TOOLS_OPENAI_KEY above also enables OpenAI TTS

# Discord voice channel
DISCORD_BOT_TOKEN=...
DISCORD_ALLOWED_USERS=...
```

### Comparação de providers STT {#stt-provider-comparison}

| Provider | Modelo | Velocidade | Qualidade | Custo | Chave de API |
|----------|--------|------------|-----------|-------|--------------|
| **Local** | `base` | Rápido (depende de CPU/GPU) | Boa | Grátis | Não |
| **Local** | `small` | Médio | Melhor | Grátis | Não |
| **Local** | `large-v3` | Lento | Melhor | Grátis | Não |
| **Groq** | `whisper-large-v3-turbo` | Muito rápido (~0.5s) | Boa | Free tier | Sim |
| **Groq** | `whisper-large-v3` | Rápido (~1s) | Melhor | Free tier | Sim |
| **OpenAI** | `whisper-1` | Rápido (~1s) | Boa | Pago | Sim |
| **OpenAI** | `gpt-4o-transcribe` | Médio (~2s) | Melhor | Pago | Sim |
| **Mistral** | `voxtral-mini-latest` | Rápido | Boa | Pago | Sim |
| **xAI** | `grok-stt` | Rápido | Boa | Pago | Sim |

Prioridade de provider (fallback automático): **local** > **groq** > **openai**

### Comparação de providers TTS {#tts-provider-comparison}

| Provider | Qualidade | Custo | Latência | Chave necessária |
|----------|-----------|-------|----------|------------------|
| **Edge TTS** | Boa | Grátis | ~1s | Não |
| **ElevenLabs** | Excelente | Pago | ~2s | Sim |
| **OpenAI TTS** | Boa | Pago | ~1.5s | Sim |
| **NeuTTS** | Boa | Grátis | Depende de CPU/GPU | Não |

NeuTTS usa o bloco de config `tts.neutts` acima.

---

## Solução de problemas {#troubleshooting}

### "No audio device found" (CLI) {#no-audio-device-found-cli}

PortAudio não está instalado:

```bash
brew install portaudio    # macOS
sudo apt install portaudio19-dev  # Ubuntu
```

Se você roda o Hermes dentro do Docker em desktop Linux, o container também precisa de acesso ao socket de áudio do host. Veja as notas da [ponte de áudio Docker](/user-guide/docker#optional-linux-desktop-audio-bridge) para setup compatível com PulseAudio/PipeWire.

### Bot não responde em canais de servidor Discord {#bot-doesnt-respond-in-discord-server-channels}

O bot exige @mention por padrão em canais de servidor. Certifique-se de:

1. Digitar `@` e selecionar o **usuário bot** (com o #discriminator), não o **role** com o mesmo nome
2. Ou usar DMs — sem menção necessária
3. Ou definir `DISCORD_REQUIRE_MENTION=false` em `~/.hermes/.env`

### Bot entra no VC mas não me ouve {#bot-joins-vc-but-doesnt-hear-me}

- Verifique se seu ID de usuário Discord está em `DISCORD_ALLOWED_USERS`
- Certifique-se de não estar mutado no Discord
- O bot precisa de um evento SPEAKING do Discord antes de mapear seu áudio — comece a falar dentro de alguns segundos após entrar

### Bot me ouve mas não responde {#bot-hears-me-but-doesnt-respond}

- Verifique se STT está disponível: instale `faster-whisper` (sem chave necessária) ou defina `GROQ_API_KEY` / `VOICE_TOOLS_OPENAI_KEY`
- Confira se o modelo LLM está configurado e acessível
- Revise logs do gateway: `tail -f ~/.hermes/logs/gateway.log`

### Bot responde em texto mas não no canal de voz {#bot-responds-in-text-but-not-in-voice-channel}

- Provider TTS pode estar falhando — verifique chave de API e cota
- Edge TTS (grátis, sem chave) é o fallback padrão
- Confira logs por erros TTS

### Whisper retorna texto lixo {#whisper-returns-garbage-text}

O filtro de alucinação captura a maioria dos casos automaticamente. Se você ainda recebe transcrições fantasma:

- Use um ambiente mais silencioso
- Ajuste `silence_threshold` na config (maior = menos sensível)
- Tente um modelo STT diferente
