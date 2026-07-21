---
sidebar_position: 9
title: "Voz e TTS"
description: "Text-to-speech e transcrição de mensagens de voz em todas as plataformas"
---

# Voz e TTS

O Hermes Agent suporta saída text-to-speech e transcrição de mensagens de voz em todas as plataformas de mensagens.

:::tip Assinantes Nous
Se você tem assinatura paga do [Nous Portal](https://portal.nousresearch.com), OpenAI TTS está disponível pelo **[Tool Gateway](tool-gateway.md)** sem chave OpenAI API separada. Instalações novas podem rodar `hermes setup --portal` para login e ligar todas as ferramentas do gateway de uma vez; instalações existentes podem escolher **Nous Subscription** só para TTS via `hermes model` ou `hermes tools`.
:::

## Síntese de fala (text-to-speech) {#text-to-speech}

Converta texto em fala com dez providers:

| Provider | Qualidade | Custo | Chave de API |
|----------|-----------|-------|--------------|
| **Edge TTS** (padrão) | Boa | Grátis | Nenhuma necessária |
| **ElevenLabs** | Excelente | Pago | `ELEVENLABS_API_KEY` |
| **OpenAI TTS** | Boa | Pago | `VOICE_TOOLS_OPENAI_KEY` |
| **MiniMax TTS** | Excelente | Pago | `MINIMAX_API_KEY` |
| **Mistral (Voxtral TTS)** | Excelente | Pago | `MISTRAL_API_KEY` |
| **Google Gemini TTS** | Excelente | Free tier | `GEMINI_API_KEY` |
| **xAI TTS** | Excelente | Pago | `XAI_API_KEY` |
| **NeuTTS** | Boa | Grátis (local) | Nenhuma necessária |
| **KittenTTS** | Boa | Grátis (local) | Nenhuma necessária |
| **Piper** | Boa | Grátis (local) | Nenhuma necessária |

### Entrega por plataforma {#platform-delivery}

| Plataforma | Entrega | Formato |
|------------|---------|---------|
| Telegram | Voice bubble (reproduz inline) | Opus `.ogg` |
| Discord | Voice bubble (Opus/OGG), fallback para anexo de arquivo | Opus/MP3 |
| WhatsApp | Anexo de arquivo de áudio | MP3 |
| CLI | Salvo em `~/.hermes/audio_cache/` | MP3 |

### Configuração {#configuration}

```yaml
# In ~/.hermes/config.yaml
tts:
  provider: "edge"              # "edge" | "elevenlabs" | "openai" | "minimax" | "mistral" | "gemini" | "xai" | "neutts" | "kittentts" | "piper"
  speed: 1.0                    # Global speed multiplier (provider-specific settings override this)
  edge:
    voice: "en-US-AriaNeural"   # 322 voices, 74 languages
    speed: 1.0                  # Converted to rate percentage (+/-%)
  elevenlabs:
    voice_id: "pNInz6obpgDQGcFmaJgB"  # Adam
    model_id: "eleven_multilingual_v2"
  openai:
    model: "gpt-4o-mini-tts"
    voice: "alloy"              # alloy, echo, fable, onyx, nova, shimmer
    base_url: "https://api.openai.com/v1"  # Override for OpenAI-compatible TTS endpoints
    speed: 1.0                  # 0.25 - 4.0
  minimax:
    model: "speech-02-hd"     # speech-02-hd (default), speech-02-turbo
    voice_id: "English_Graceful_Lady"  # See https://platform.minimax.io/faq/system-voice-id
    speed: 1                    # 0.5 - 2.0
    vol: 1                      # 0 - 10
    pitch: 0                    # -12 - 12
  mistral:
    model: "voxtral-mini-tts-2603"
    voice_id: "c69964a6-ab8b-4f8a-9465-ec0925096ec8"  # Paul - Neutral (default)
  gemini:
    model: "gemini-2.5-flash-preview-tts"  # or gemini-3.1-flash-tts-preview
    voice: "Kore"               # 30 prebuilt voices: Zephyr, Puck, Kore, Enceladus, Gacrux, etc.
    audio_tags: false           # Enable hidden Gemini 3.1 TTS audio-tag insertion
    persona_prompt_file: ""      # Optional Markdown/text file with Gemini voice direction
  xai:
    voice_id: "eve"             # or a custom voice ID — see docs below
    language: "en"              # ISO 639-1 code
    sample_rate: 24000          # 22050 / 24000 (default) / 44100 / 48000
    bit_rate: 128000            # MP3 bitrate; only applies when codec=mp3
    # base_url: "https://api.x.ai/v1"   # Override via XAI_BASE_URL env var
  neutts:
    ref_audio: ''
    ref_text: ''
    model: neuphonic/neutts-air-q4-gguf
    device: cpu
  kittentts:
    model: KittenML/kitten-tts-nano-0.8-int8   # 25MB int8; also: kitten-tts-micro-0.8 (41MB), kitten-tts-mini-0.8 (80MB)
    voice: Jasper                               # Jasper, Bella, Luna, Bruno, Rosie, Hugo, Kiki, Leo
    speed: 1.0                                  # 0.5 - 2.0
    clean_text: true                            # Expand numbers, currencies, units
  piper:
    voice: en_US-lessac-medium                  # voice name (auto-downloaded) OR absolute path to .onnx
    # voices_dir: ''                            # default: ~/.hermes/cache/piper-voices/
    # use_cuda: false                           # requires onnxruntime-gpu
    # length_scale: 1.0                         # 2.0 = twice as slow
    # noise_scale: 0.667
    # noise_w_scale: 0.8
    # volume: 1.0                               # 0.5 = half as loud
    # normalize_audio: true
```

**Controle de velocidade**: O valor global `tts.speed` aplica-se a todos os providers por padrão. Cada provider pode sobrescrevê-lo com sua própria configuração `speed` (ex.: `tts.openai.speed: 1.5`). Velocidade específica do provider tem precedência sobre o valor global. O padrão é `1.0` (velocidade normal).

### Prompts de persona Gemini {#gemini-persona-prompts}

Gemini TTS pode seguir direção de performance em linguagem natural. Defina `tts.gemini.persona_prompt_file` para um arquivo Markdown ou texto local que descreve a persona de voz. O arquivo pode incluir seções estilo Gemini como `AUDIO PROFILE`, `SCENE`, `DIRECTOR'S NOTES`, `SAMPLE CONTEXT` e `TRANSCRIPT`.

Se o arquivo contém `{transcript}` ou `{{ transcript }}`, o Hermes substitui esse placeholder pelo texto TTS ao vivo. Caso contrário, o Hermes anexa automaticamente uma seção `TRANSCRIPT` rotulada. O prompt de persona permanece local e não é mostrado na resposta do chat.

```yaml
tts:
  provider: gemini
  gemini:
    voice: Algieba
    persona_prompt_file: ~/.hermes/tts/butler-voice.md
```

### Tags de áudio Gemini {#gemini-audio-tags}

Gemini 3.1 Flash TTS suporta tags de áudio freeform entre colchetes como `[whispers]`, `[excitedly]`, `[very slow]`, `[laughs]` e outras notas expressivas de entrega. Habilite `tts.gemini.audio_tags` para o Hermes rodar um pass de reescrita oculto antes do Gemini TTS. A reescrita insere tags inline apenas no script TTS; a resposta visível do chat permanece inalterada.

```yaml
tts:
  provider: gemini
  gemini:
    model: gemini-3.1-flash-tts-preview
    audio_tags: true
```

A reescrita usa `auxiliary.tts_audio_tags` e usa por padrão seu modelo principal de chat. Sobrescreva essa tarefa auxiliar se quiser inserção de tags tratada por um modelo mais barato ou rápido.


### Limites de comprimento de entrada {#input-length-limits}

Cada provider tem um cap documentado de caracteres por requisição. O Hermes trunca texto antes de chamar o provider para que requisições nunca falhem por erro de comprimento:

| Provider | Limite padrão (chars) |
|----------|----------------------|
| Edge TTS | 5000 |
| OpenAI | 4096 |
| xAI | 15000 |
| MiniMax | 10000 |
| Mistral | 4000 |
| Google Gemini | 32000 |
| ElevenLabs | Conforme o modelo (veja abaixo) |
| NeuTTS | 2000 |
| KittenTTS | 2000 |
| Piper | 5000 |

**ElevenLabs** escolhe um cap a partir do `model_id` configurado:

| `model_id` | Limite (chars) |
|------------|----------------|
| `eleven_flash_v2_5` | 40000 |
| `eleven_flash_v2` | 30000 |
| `eleven_multilingual_v2` (padrão), `eleven_multilingual_v1`, `eleven_english_sts_v2`, `eleven_english_sts_v1` | 10000 |
| `eleven_v3`, `eleven_ttv_v3` | 5000 |
| Modelo desconhecido | Fallback para o padrão do provider (10000) |

**Override por provider** com `max_text_length:` na seção do provider na sua config TTS:

```yaml
tts:
  openai:
    max_text_length: 8192   # raise or lower the provider cap
```

Apenas inteiros positivos são honrados. Zero, negativos, não numéricos ou booleanos caem no padrão do provider, para que config quebrada não desabilite truncamento acidentalmente.

### Voice bubbles Telegram e ffmpeg {#telegram-voice-bubbles--ffmpeg}

Voice bubbles do Telegram exigem formato de áudio Opus/OGG:

- **OpenAI, ElevenLabs e Mistral** produzem Opus nativamente — sem setup extra
- **Edge TTS** (padrão) gera MP3 e precisa de **ffmpeg** para converter:
- **MiniMax TTS** gera MP3 e precisa de **ffmpeg** para converter para voice bubbles Telegram
- **Google Gemini TTS** gera PCM cru e usa **ffmpeg** para codificar Opus diretamente para voice bubbles Telegram
- **xAI TTS** gera MP3 e precisa de **ffmpeg** para converter para voice bubbles Telegram
- **NeuTTS** gera WAV e também precisa de **ffmpeg** para converter para voice bubbles Telegram
- **KittenTTS** gera WAV e também precisa de **ffmpeg** para converter para voice bubbles Telegram
- **Piper** gera WAV e também precisa de **ffmpeg** para converter para voice bubbles Telegram

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Fedora
sudo dnf install ffmpeg
```

Sem ffmpeg, áudio Edge TTS, MiniMax TTS, NeuTTS, KittenTTS e Piper são enviados como arquivos de áudio regulares (reproduzíveis, mas mostrados como player retangular em vez de voice bubble).

:::tip
Se você quer voice bubbles sem instalar ffmpeg, mude para o provider OpenAI, ElevenLabs ou Mistral.
:::

### Vozes customizadas xAI (clonagem de voz) {#xai-custom-voices-voice-cloning}

xAI suporta clonar sua voz e usá-la com TTS. Crie uma voz customizada no [xAI Console](https://console.x.ai/team/default/voice/voice-library), depois defina o `voice_id` resultante na sua config:

```yaml
tts:
  provider: xai
  xai:
    voice_id: "nlbqfwie"   # your custom voice ID
```

Veja a [documentação xAI Custom Voices](https://docs.x.ai/developers/model-capabilities/audio/custom-voices) para detalhes sobre gravação, formatos suportados e limites.

### Piper (local, 44 idiomas) {#piper-local-44-languages}

Piper é um engine TTS neural local e rápido da Open Home Foundation (mantenedores do Home Assistant). Roda inteiramente em CPU, suporta **44 idiomas** com vozes pré-treinadas e não precisa de chave de API.

**Instale via `hermes tools`** → Voice & TTS → Piper — o Hermes roda `pip install piper-tts` para você. Ou instale manualmente: `pip install piper-tts`.

**Mude para Piper:**

```yaml
tts:
  provider: piper
  piper:
    voice: en_US-lessac-medium
```

Na primeira chamada TTS para uma voz que não está em cache local, o Hermes roda `python -m piper.download_voices <name>` e baixa o modelo (~20-90MB dependendo do tier de qualidade) em `~/.hermes/cache/piper-voices/`. Chamadas subsequentes reutilizam o modelo em cache.

**Escolhendo uma voz.** O [catálogo completo de vozes](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/VOICES.md) cobre inglês, espanhol, francês, alemão, italiano, holandês, português, russo, polonês, turco, chinês, árabe, hindi e mais — cada um com tiers de qualidade `x_low` / `low` / `medium` / `high`. Ouça amostras em [rhasspy.github.io/piper-samples](https://rhasspy.github.io/piper-samples/).

**Usando voz pré-baixada.** Defina `tts.piper.voice` para um caminho absoluto terminando em `.onnx`:

```yaml
tts:
  piper:
    voice: /path/to/my-custom-voice.onnx
```

**Knobs avançados** (`tts.piper.length_scale` / `noise_scale` / `noise_w_scale` / `volume` / `normalize_audio`, `use_cuda`) correspondem 1:1 ao `SynthesisConfig` do Piper. São ignorados em versões antigas de `piper-tts`.

### Providers de comando customizado {#custom-command-providers}

Se um engine TTS que você quer não é suportado nativamente (VoxCPM, MLX-Kokoro, XTTS CLI, script de clonagem de voz, qualquer outro que exponha CLI), você pode conectá-lo como **provider type command** sem escrever Python. O Hermes escreve o texto de entrada em um arquivo UTF-8 temp, roda seu comando shell e lê o arquivo de áudio que o comando produziu.

Declare um ou mais providers em `tts.providers.<name>` e alterne entre eles com `tts.provider: <name>` — da mesma forma que alterna entre built-ins como `edge` e `openai`.

```yaml
tts:
  provider: voxcpm                 # pick any name under tts.providers
  providers:
    voxcpm:
      type: command
      command: "voxcpm --ref ~/voice.wav --text-file {input_path} --out {output_path}"
      output_format: mp3
      timeout: 180
      voice_compatible: true       # try to deliver as a Telegram voice bubble

    mlx-kokoro:
      type: command
      command: "python -m mlx_kokoro --in {input_path} --out {output_path} --voice {voice}"
      voice: af_sky
      output_format: wav

    piper-custom:                  # native Piper also supports custom .onnx via tts.piper.voice
      type: command
      command: "piper -m /path/to/custom.onnx -f {output_path} < {input_path}"
      output_format: wav
```

#### Exemplo: Doubao (Chinese seed-tts-2.0) {#example-doubao-chinese-seed-tts-20}

Para TTS chinês de alta qualidade via API bidirectional-streaming [seed-tts-2.0](https://www.volcengine.com/docs/6561/1257544) da ByteDance, instale o pacote PyPI [`doubao-speech`](https://pypi.org/project/doubao-speech/) e conecte como command provider:

```bash
pip install doubao-speech
export VOLCENGINE_APP_ID="your-app-id"
export VOLCENGINE_ACCESS_TOKEN="your-access-token"
```

```yaml
tts:
  provider: doubao
  providers:
    doubao:
      type: command
      command: "doubao-speech say --text-file {input_path} --out {output_path}"
      output_format: mp3
      max_text_length: 1024
      timeout: 30
```

Credenciais vêm do ambiente shell (`VOLCENGINE_APP_ID` / `VOLCENGINE_ACCESS_TOKEN`) ou `~/.doubao-speech/config.yaml`. Escolha uma voz adicionando `--voice zh-female-warm` (ou outro alias de `doubao-speech list-voices`) ao comando. `doubao-speech` também inclui ASR streaming — veja a [seção STT abaixo](#example-doubao--volcengine-asr) para integração Hermes. Fonte e docs completas: [github.com/Hypnus-Yuan/doubao-speech](https://github.com/Hypnus-Yuan/doubao-speech).

#### Placeholders {#placeholders}

Seu template de comando pode referenciar estes placeholders. O Hermes substitui em render time e faz shell-quote de cada valor para o contexto circundante (bare / single-quoted / double-quoted), então caminhos com espaços e outros caracteres sensíveis ao shell são seguros.

| Placeholder      | Significado                                              |
|------------------|----------------------------------------------------------|
| `{input_path}`   | Caminho do arquivo de texto UTF-8 temp que o Hermes escreveu |
| `{text_path}`    | Alias de `{input_path}`                                  |
| `{output_path}`  | Caminho onde o comando deve escrever o áudio             |
| `{format}`       | `mp3` / `wav` / `ogg` / `flac`                           |
| `{voice}`        | `tts.providers.<name>.voice`, vazio quando não definido  |
| `{model}`        | `tts.providers.<name>.model`                             |
| `{speed}`        | Multiplicador de velocidade resolvido (provider ou global) |

Use `{{` e `}}` para chaves literais.

#### Chaves opcionais {#optional-keys}

| Chave                | Padrão | Significado                                                                                                |
|----------------------|--------|------------------------------------------------------------------------------------------------------------|
| `timeout`            | `120`  | Segundos; a árvore de processos é morta ao expirar (Unix `killpg`, Windows `taskkill /T`).                 |
| `output_format`      | `mp3`  | Um de `mp3` / `wav` / `ogg` / `flac`. Inferido automaticamente da extensão de saída se o Hermes escolher um caminho. |
| `voice_compatible`   | `false`| Quando `true`, o Hermes converte saída MP3/WAV para Opus/OGG via ffmpeg para o Telegram renderizar voice bubble. |
| `max_text_length`    | `5000` | A entrada é truncada a este comprimento antes de renderizar o comando.                                     |
| `voice` / `model`    | vazio  | Repassados ao comando apenas como valores de placeholder.                                                  |

#### Notas de comportamento {#behavior-notes}

- **Nomes built-in sempre vencem.** Uma entrada `tts.providers.openai` nunca shadow o provider OpenAI nativo, então nenhuma config de usuário pode substituir silenciosamente um built-in.
- **Entrega padrão é documento.** Command providers entregam como anexos de áudio regulares em toda plataforma. Opt-in para entrega voice-bubble por provider com `voice_compatible: true`.
- **Falhas de comando surfaceam ao agente.** Exit não-zero, saída vazia ou timeout retornam erro com stderr/stdout do comando incluído para você debugar o provider pela conversa.
- **`type: command` é o padrão quando `command:` está definido.** Escrever `type: command` explicitamente é boa prática mas não obrigatório; entrada com string `command` não vazia é tratada como command provider.
- **`{input_path}` / `{text_path}` são intercambiáveis.** Use o que ler melhor no seu comando.

#### Segurança {#security}

Command-type providers rodam qualquer comando shell que você configurar, com permissões do seu usuário. O Hermes faz quote de valores placeholder e aplica o timeout configurado, mas o template de comando em si é input local confiável — trate como trataria um shell script no seu PATH.

### Providers plugin Python {#python-plugin-providers}

Para engines TTS que não podem ser expressos como um único comando shell — SDKs Python sem CLI, engines streaming, APIs de listagem de voz, auth OAuth-refreshing — registre um plugin Python via `ctx.register_tts_provider()`. O plugin **coexiste com** (não substitui) o registro de [Custom command providers](#custom-command-providers); escolha a superfície que encaixa seu engine.

#### Quando escolher qual {#when-to-pick-which}

| Seu backend tem… | Use |
|---|---|
| Um único CLI que lê texto de arquivo/stdin e escreve áudio em arquivo/stdout | **Command provider** (sem Python) |
| Dois ou três CLIs encadeados com pipes shell | **Command provider** |
| Apenas SDK Python — sem CLI | **Plugin** |
| Bytes streaming que você quer entregar em chunks (voice bubbles mid-generation) | **Plugin** (sobrescreva `stream()`) |
| API de listagem de vozes usada por `hermes setup` | **Plugin** (sobrescreva `list_voices()`) |
| Fluxo de refresh OAuth (não bearer token estático) | **Plugin** |

Built-ins sempre vencem, e command providers vencem sobre plugin de mesmo nome — então plugins são seguros para registrar contra qualquer nome não-built-in sem se preocupar em shadow sua config existente.

#### Plugin mínimo {#minimal-plugin}

Coloque isto em `~/.hermes/plugins/my-tts/`:

`plugin.yaml`:
```yaml
name: my-tts
version: 0.1.0
description: "My custom Python TTS backend"
```

`__init__.py`:
```python
from agent.tts_provider import TTSProvider


class MyTTSProvider(TTSProvider):
    @property
    def name(self) -> str:
        return "my-tts"  # what tts.provider matches against

    @property
    def display_name(self) -> str:
        return "My Custom TTS"

    def is_available(self) -> bool:
        # Return False when credentials/deps are missing — picker skips
        # this row but the dispatcher still routes here on explicit config.
        import os
        return bool(os.environ.get("MY_TTS_API_KEY"))

    def synthesize(self, text, output_path, *, voice=None, model=None,
                   speed=None, format="mp3", **extra) -> str:
        # Write audio bytes to output_path, return the path.
        # Raise on failure — the dispatcher converts exceptions to a
        # standard error envelope.
        import my_tts_sdk
        client = my_tts_sdk.Client()
        audio_bytes = client.synthesize(text=text, voice=voice or "default")
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        return output_path


def register(ctx):
    ctx.register_tts_provider(MyTTSProvider())
```

Habilite (`hermes plugins enable my-tts`), aponte `tts.provider` para ele (`tts.provider: my-tts` em `config.yaml`), e a ferramenta `text_to_speech` roteará pelo seu plugin.

#### Hooks opcionais {#optional-hooks}

Sobrescreva estes na sua classe provider para integração mais rica:

- `list_voices()` → lista de dicts `{id, display, language, gender, preview_url}` exibida em `hermes tools`.
- `list_models()` → lista de dicts `{id, display, languages, max_text_length}`.
- `get_setup_schema()` → retorna `{name, badge, tag, env_vars: [{key, prompt, url}]}` para alimentar a linha do picker em `hermes tools` / `hermes setup`. Sem isso, o plugin ainda funciona mas sua linha no picker é mínima.
- `stream(text, *, voice, model, format, **extra)` → iterador que produz bytes de áudio para entrega streaming (padrão levanta `NotImplementedError`).
- propriedade `voice_compatible` → defina `True` se sua saída for compatível com Opus e o gateway deve entregá-la como voice bubble (padrão `False` = anexo de áudio regular).

Veja `agent/tts_provider.py` para o ABC completo incluindo docstrings.

## Transcrição de mensagens de voz (STT) {#voice-message-transcription-stt}

Mensagens de voz enviadas no Telegram, Discord, WhatsApp, Slack ou Signal são transcritas automaticamente e injetadas como texto na conversa. O agente vê a transcrição como texto normal.

| Provider | Qualidade | Custo | Chave de API |
|----------|-----------|-------|--------------|
| **Local Whisper** (padrão) | Boa | Grátis | Nenhuma necessária |
| **Groq Whisper API** | Boa–Excelente | Free tier | `GROQ_API_KEY` |
| **OpenAI Whisper API** | Boa–Excelente | Pago | `VOICE_TOOLS_OPENAI_KEY` ou `OPENAI_API_KEY` |

:::info Zero config
Transcrição local funciona out of the box quando `faster-whisper` está instalado. Se indisponível, o Hermes também pode usar CLI `whisper` local de locais comuns de instalação (como `/opt/homebrew/bin`) ou comando customizado via `HERMES_LOCAL_STT_COMMAND`.
:::

### Configuração {#configuration-1}

```yaml
# In ~/.hermes/config.yaml
stt:
  provider: "local"           # "local" | "groq" | "openai" | "mistral" | "xai"
  local:
    model: "base"             # tiny, base, small, medium, large-v3
  openai:
    model: "whisper-1"        # whisper-1, gpt-4o-mini-transcribe, gpt-4o-transcribe
  mistral:
    model: "voxtral-mini-latest"  # voxtral-mini-latest, voxtral-mini-2602
  xai:
    model: "grok-stt"         # xAI Grok STT
```

### Detalhes dos providers {#provider-details}

**Local (faster-whisper)** — Roda Whisper localmente via [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Usa CPU por padrão, GPU se disponível. Tamanhos de modelo:

| Modelo | Tamanho | Velocidade | Qualidade |
|--------|---------|------------|-----------|
| `tiny` | ~75 MB | Mais rápido | Básica |
| `base` | ~150 MB | Rápido | Boa (padrão) |
| `small` | ~500 MB | Médio | Melhor |
| `medium` | ~1.5 GB | Mais lento | Ótima |
| `large-v3` | ~3 GB | Mais lento ainda | Melhor |

**Groq API** — Requer `GROQ_API_KEY`. Bom fallback cloud quando você quer opção STT hospedada grátis.

**OpenAI API** — Aceita `VOICE_TOOLS_OPENAI_KEY` primeiro e faz fallback para `OPENAI_API_KEY`. Suporta `whisper-1`, `gpt-4o-mini-transcribe` e `gpt-4o-transcribe`.

**Mistral API (Voxtral Transcribe)** — Requer `MISTRAL_API_KEY`. Usa modelos [Voxtral Transcribe](https://docs.mistral.ai/capabilities/audio/speech_to_text/) da Mistral. Suporta 13 idiomas, diarização de speaker e timestamps word-level. Instale com `cd ~/.hermes/hermes-agent && uv pip install -e ".[mistral]"`.

**xAI Grok STT** — Requer `XAI_API_KEY`. Posta em `https://api.x.ai/v1/stt` como multipart/form-data. Boa escolha se você já usa xAI para chat ou TTS e quer uma chave de API para tudo. Ordem de auto-detecção coloca após Groq — defina explicitamente `stt.provider: xai` para forçar.

**Fallback CLI local customizado** — Defina `HERMES_LOCAL_STT_COMMAND` se quiser que o Hermes chame um comando de transcrição local diretamente. O template de comando suporta placeholders `{input_path}`, `{output_dir}`, `{language}` e `{model}`. Seu comando deve escrever um transcript `.txt` em algum lugar sob `{output_dir}`.

#### Exemplo: Doubao / Volcengine ASR {#example-doubao--volcengine-asr}

Se você usa [`doubao-speech`](https://pypi.org/project/doubao-speech/) para Doubao TTS (veja [acima](#example-doubao-chinese-seed-tts-20)), o mesmo pacote trata speech-to-text via superfície STT local-command:

```bash
pip install doubao-speech
export VOLCENGINE_APP_ID="your-app-id"
export VOLCENGINE_ACCESS_TOKEN="your-access-token"
export HERMES_LOCAL_STT_COMMAND='doubao-speech transcribe {input_path} --out {output_dir}/transcript.txt'
```

```yaml
stt:
  provider: local_command
```

O Hermes escreve a mensagem de voz recebida em `{input_path}`, roda o comando e lê o arquivo `.txt` produzido sob `{output_dir}`. Idioma é auto-detectado pelo endpoint bigmodel Volcengine.

### Comportamento de fallback {#fallback-behavior}

Se seu provider configurado não estiver disponível, o Hermes faz fallback automaticamente:
- **Local faster-whisper indisponível** → Tenta CLI `whisper` local ou `HERMES_LOCAL_STT_COMMAND` antes de providers cloud
- **Chave Groq não definida** → Fallback para transcrição local, depois OpenAI
- **Chave OpenAI não definida** → Fallback para transcrição local, depois Groq
- **Chave/SDK Mistral não definida** → Pulado em auto-detect; passa para próximo provider disponível
- **Nada disponível** → Mensagens de voz passam com nota precisa ao usuário

### Providers STT de comando customizado {#stt-custom-command-providers}

Se o engine STT que você quer não é suportado nativamente (Doubao ASR, NVIDIA Parakeet, build whisper.cpp, CLI SenseVoice open-source, qualquer outro que exponha comando shell), conecte como **provider type command** sem escrever Python. O Hermes roda seu comando shell contra o arquivo de áudio e lê de volta a transcrição.

Declare um ou mais providers em `stt.providers.<name>` e alterne com `stt.provider: <name>` — mesma forma que o registro TTS de [command providers](#custom-command-providers), adaptado para direção input=audio → output=transcript.

```yaml
stt:
  provider: parakeet                # pick any name under stt.providers
  providers:
    parakeet:
      type: command
      command: "parakeet-asr --model nvidia/parakeet-tdt-0.6b-v2 --in {input_path} --out {output_path}"
      format: txt
      language: en
      timeout: 300

    whispercpp:
      type: command
      command: "whisper-cli -m ~/models/ggml-large-v3.bin -f {input_path} -otxt -of {output_dir}/transcript"
      format: txt

    sensevoice:
      type: command
      command: "sensevoice-cli {input_path} --json | tee {output_path}"
      format: json
```

Isso complementa o escape hatch legado `HERMES_LOCAL_STT_COMMAND` — essa env var ainda funciona intacta via caminho built-in `local_command`. Use `stt.providers.<name>` quando quiser **múltiplos** engines STT driven por shell, um nome que pode escolher via `stt.provider`, ou qualquer coisa que precise de `language` / `model` / `timeout` por provider.

#### Placeholders STT {#stt-placeholders}

Seu template de comando pode referenciar estes placeholders. O Hermes substitui em render time e faz shell-quote de cada valor para o contexto circundante (bare / single-quoted / double-quoted), então caminhos com espaços são seguros.

| Placeholder       | Significado                                                              |
|-------------------|--------------------------------------------------------------------------|
| `{input_path}`    | Caminho absoluto do arquivo de áudio de entrada (localização original, somente leitura) |
| `{output_path}`   | Caminho absoluto onde o comando deve escrever a transcrição              |
| `{output_dir}`    | Diretório pai de `{output_path}` (útil para ferramentas estilo whisper)  |
| `{format}`        | Formato de saída configurado: `txt` / `json` / `srt` / `vtt`             |
| `{language}`      | Código de idioma configurado (padrão `en`)                               |
| `{model}`         | `stt.providers.<name>.model`, vazio quando não definido                  |

Use `{{` e `}}` para chaves literais (útil ao embutir snippets JSON no comando).

#### Como a transcrição é lida de volta {#how-the-transcript-is-read-back}

Após seu comando sair com sucesso:

1. Se `{output_path}` existir e não estiver vazio → Hermes lê como texto UTF-8.
2. Caso contrário, se o comando escreveu em stdout → Hermes usa isso.
3. Caso contrário → erro: "Command STT provider wrote no output file and produced no stdout".

Isso permite usar o registro tanto para CLIs que escrevem arquivo (`whisper-cli`, `parakeet-asr`) quanto one-liners estilo curl que emitem transcript em stdout (`curl … | jq -r .text`).

Para `format: json` / `srt` / `vtt`, Hermes retorna o conteúdo cru do arquivo como campo `transcript`. Extrair `.text` de JSON está fora do escopo do runner — configure `format: txt`, ou post-processe JSON downstream.

#### Chaves opcionais de command-provider STT {#stt-command-provider-optional-keys}

| Chave             | Padrão | Significado                                                                                          |
|-------------------|--------|------------------------------------------------------------------------------------------------------|
| `timeout`         | `300`  | Segundos; a árvore de processos é morta ao expirar (Unix `start_new_session`, Windows `taskkill /T`). |
| `format`          | `txt`  | Um de `txt` / `json` / `srt` / `vtt`. Define a extensão de `{output_path}`.                        |
| `language`        | `en`   | Repassado a `{language}`. Padrão de `stt.language`, depois `en`.                                     |
| `model`           | vazio  | Repassado a `{model}`. O argumento `model=` de `transcribe_audio()` sobrescreve isso.               |

#### Notas de comportamento de command-provider STT {#stt-command-provider-behavior-notes}

- **Built-ins sempre vencem.** Declarar `stt.providers.openai: type: command` NÃO override o handler real OpenAI Whisper. O nome built-in é short-circuited antes do resolvedor command-provider rodar.
- **Limpeza de process-tree.** Comando rodando além do `timeout` tem toda a process tree morta, não só o wrapper shell. Pipelines ASR longos que fork subprocessos de carregamento de modelo são reaped de forma confiável.
- **Shell-quoting é automático.** Placeholders dentro de `'…'` recebem escaping safe para single-quote; dentro de `"…"` recebem escaping de `$`/`` ` ``/`"`; fora de quotes recebem `shlex.quote`. Não faça pre-quote de valores placeholder.

#### Segurança de command-provider STT {#stt-command-provider-security}

O comando shell roda sob o mesmo usuário que o Hermes com acesso completo ao filesystem — mesmo trust model que `tts.providers.<name>: type: command` e `HERMES_LOCAL_STT_COMMAND`. Declare command providers apenas de fontes em que confia.

### Providers plugin Python (STT) {#python-plugin-providers-stt}

Para engines STT que não são built-in E não podem ser expressos como comando shell (precisam SDK Python, auth OAuth-refreshing, chunks streaming, etc.), registre plugin Python via `ctx.register_transcription_provider()`. O plugin **coexiste com** os 6 providers built-in (`local`, `local_command`, `groq`, `openai`, `mistral`, `xai`) e o registro `stt.providers.<name>: type: command` — built-ins mantêm implementações nativas e sempre vencem em colisão de nome; command providers vencem sobre plugins de mesmo nome (config é mais local que instalação de plugin).

#### Quando escolher qual (STT) {#when-to-pick-which-stt}

| Backend tem…                                                 | Use                                                              |
|--------------------------------------------------------------|------------------------------------------------------------------|
| Um único comando shell que recebe arquivo de áudio e emite texto | `stt.providers.<name>: type: command` (sem Python)              |
| Só o escape hatch legado de comando único é desejado         | env var `HERMES_LOCAL_STT_COMMAND` (preservada para retrocompat) |
| SDK Python sem CLI                                           | plugin `register_transcription_provider()`                       |
| Auth OAuth-refreshing, chunks streaming, metadados de voz    | plugin `register_transcription_provider()`                       |
| Built-in já cobre (`local`, `groq`, `openai`, …)             | defina `stt.provider: <name>` — built-ins são inline             |

#### Ordem de resolução {#resolution-order}

1. **`stt.provider` é um nome built-in** → dispatch built-in. **Sempre vence.**
2. **`stt.provider` corresponde a `stt.providers.<name>` com `command:` definido** → runner command-provider (veja [STT custom command providers](#stt-custom-command-providers)). Vence sobre plugin de mesmo nome.
3. **`stt.provider` corresponde a `TranscriptionProvider` registrado por plugin** → dispatch plugin:
   - se `is_available()` do plugin retorna `False` (credenciais ou SDK faltando), a chamada surface um envelope de erro de indisponibilidade identificando o plugin — **não** a mensagem genérica "No STT provider available".
   - caso contrário `transcribe()` do plugin é chamado com `model` (do arg público `model=`, fallback para `stt.<provider>.model`) e `language` (de `stt.<provider>.language`).
4. **Sem correspondência** → erro "No STT provider available".

#### Namespace de config por provider {#per-provider-config-namespace}

Plugins leem configuração por provider de `stt.<provider>` em `config.yaml`, espelhando como built-ins leem `stt.openai.model` / `stt.mistral.model`:

```yaml
stt:
  provider: my-stt
  my-stt:
    model: whisper-large-v3
    language: ja          # forwarded as language= to transcribe()
    # any other plugin-specific keys go here; read them via your
    # own config.yaml access in __init__/is_available/transcribe
```

O dispatcher encaminha `model` e `language` desta seção; todo o resto, o plugin pode ler sozinho.

#### Plugin mínimo {#minimal-plugin-1}

Coloque isto em `~/.hermes/plugins/my-stt/`:

`plugin.yaml`:
```yaml
name: my-stt
version: 0.1.0
description: "My custom Python STT backend"
```

`__init__.py`:
```python
from agent.transcription_provider import TranscriptionProvider


class MySTTProvider(TranscriptionProvider):
    @property
    def name(self) -> str:
        return "my-stt"  # what stt.provider matches against

    @property
    def display_name(self) -> str:
        return "My Custom STT"

    def is_available(self) -> bool:
        # Return False when credentials/deps are missing — picker skips
        # this row but the dispatcher still routes here on explicit config.
        import os
        return bool(os.environ.get("MY_STT_API_KEY"))

    def transcribe(self, file_path, *, model=None, language=None, **extra):
        # Return the standard transcribe envelope:
        #   {"success": bool, "transcript": str, "provider": str, "error": str}
        # Do NOT raise — convert exceptions to the error envelope so the
        # gateway/CLI caller sees a consistent shape on failure.
        try:
            import my_stt_sdk
            client = my_stt_sdk.Client()
            text = client.transcribe(open(file_path, "rb"))
            return {
                "success": True,
                "transcript": text,
                "provider": "my-stt",
            }
        except Exception as exc:
            return {
                "success": False,
                "transcript": "",
                "error": f"my-stt failed: {exc}",
                "provider": "my-stt",
            }


def register(ctx):
    ctx.register_transcription_provider(MySTTProvider())
```

Habilite (`hermes plugins enable my-stt`), defina `stt.provider: my-stt` em `config.yaml`, e a transcrição de mensagens de voz roteará pelo seu plugin.

#### Hooks opcionais {#optional-hooks-1}

Sobrescreva estes na sua classe provider para integração mais rica:

- `list_models()` → lista de dicts `{id, display, languages, max_audio_seconds}`.
- `default_model()` → string retornada quando o usuário não sobrescreve o modelo.
- `get_setup_schema()` → retorna `{name, badge, tag, env_vars: [{key, prompt, url}]}` para alimentar linhas do picker em `hermes tools` / `hermes setup` (a categoria picker para STT ainda não foi lançada — estes metadados estão disponíveis para plugins por forward compatibility).

Veja `agent/transcription_provider.py` para o ABC completo incluindo docstrings.
