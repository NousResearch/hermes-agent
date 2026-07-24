---
sidebar_position: 8
title: "Use o Modo de Voz com o Hermes"
description: "Um guia prático para configurar e usar o modo de voz do Hermes na CLI, Telegram, Discord e canais de voz do Discord"
---

# Use o Modo de Voz com o Hermes {#use-voice-mode-with-hermes}

Este guia é o companheiro prático da [referência do recurso Modo de Voz](/user-guide/features/voice-mode).

Se a página do recurso explica o que o modo de voz pode fazer, este guia mostra como realmente usá-lo bem.

:::tip
O [Nous Portal](/integrations/nous-portal) empacota tanto o LLM quanto o TTS através de um único OAuth — o modo de voz funciona de ponta a ponta sem credenciais extras.
:::

## Para que o modo de voz é bom {#what-voice-mode-is-good-for}

O modo de voz é especialmente útil quando:
- você quer um fluxo de trabalho de CLI sem usar as mãos
- você quer respostas faladas no Telegram ou Discord
- você quer o Hermes em um canal de voz do Discord para conversas ao vivo
- você quer capturar ideias rapidamente, depurar ou ir e voltar enquanto caminha, em vez de digitar

## Escolha sua Configuração de Modo de Voz {#choose-your-voice-mode-setup}

Existem, na verdade, três experiências de voz diferentes no Hermes.

| Modo | Melhor para | Plataforma |
|---|---|---|
| Loop de microfone interativo | Uso pessoal sem as mãos enquanto programa ou pesquisa | CLI |
| Respostas em voz no chat | Respostas faladas junto com mensagens normais | Telegram, Discord |
| Bot de canal de voz ao vivo | Conversa ao vivo em grupo ou pessoal em um canal de voz | Canais de voz do Discord |

Um bom caminho é:
1. primeiro, faça o texto funcionar
2. depois, habilite respostas por voz
3. por último, mova para canais de voz do Discord se quiser a experiência completa

## Passo 1: certifique-se de que o Hermes normal funciona primeiro {#step-1-make-sure-normal-hermes-works-first}

Antes de tocar no modo de voz, verifique se:
- o Hermes inicia
- seu provedor está configurado
- o agente consegue responder prompts de texto normalmente

```bash
hermes
```

Pergunte algo simples:

```text
What tools do you have available?
```

Se isso ainda não está sólido, corrija o modo texto primeiro.

## Passo 2: instale os extras corretos {#step-2-install-the-right-extras}

### Microfone e reprodução na CLI {#cli-microphone--playback}

```bash
cd ~/.hermes/hermes-agent && uv pip install -e ".[voice]"
```

### Plataformas de mensageria {#messaging-platforms}

```bash
cd ~/.hermes/hermes-agent && uv pip install -e ".[messaging]"
```

### TTS premium ElevenLabs {#premium-elevenlabs-tts}

```bash
cd ~/.hermes/hermes-agent && uv pip install -e ".[tts-premium]"
```

### NeuTTS local (opcional) {#local-neutts-optional}

```bash
python -m pip install -U neutts[all]
```

### Tudo {#everything}

```bash
cd ~/.hermes/hermes-agent && uv pip install -e ".[all]"
```

## Passo 3: instale as dependências do sistema {#step-3-install-system-dependencies}

### macOS {#macos}

```bash
brew install portaudio ffmpeg opus
brew install espeak-ng
```

### Ubuntu / Debian {#ubuntu--debian}

```bash
sudo apt install portaudio19-dev ffmpeg libopus0
sudo apt install espeak-ng
```

Por que elas importam:
- `portaudio` → entrada de microfone / reprodução para o modo de voz na CLI
- `ffmpeg` → conversão de áudio para TTS e entrega em mensageria
- `opus` → suporte a codec de voz do Discord
- `espeak-ng` → backend de fonemização para o NeuTTS

## Passo 4: escolha os provedores de STT e TTS {#step-4-choose-stt-and-tts-providers}

O Hermes suporta pilhas de voz locais e na nuvem.

### Configuração mais fácil / mais barata {#easiest--cheapest-setup}

Use STT local e Edge TTS gratuito:
- provedor de STT: `local`
- provedor de TTS: `edge`

Esse é geralmente o melhor lugar para começar.

### Exemplo de arquivo de ambiente {#environment-file-example}

Adicione a `~/.hermes/.env`:

```bash
# Cloud STT options (local needs no key)
GROQ_API_KEY=***
VOICE_TOOLS_OPENAI_KEY=***

# Premium TTS (optional)
ELEVENLABS_API_KEY=***
```

### Recomendações de Provedor {#provider-recommendations}

#### Fala para texto {#speech-to-text}

- `local` → melhor padrão para privacidade e uso sem custo
- `groq` → transcrição em nuvem muito rápida
- `openai` → bom fallback pago

#### Texto para fala {#text-to-speech}

- `edge` → gratuito e bom o suficiente para a maioria dos usuários
- `neutts` → TTS local/no dispositivo gratuito
- `elevenlabs` → melhor qualidade
- `openai` → bom meio-termo
- `mistral` → multilíngue, Opus nativo

### Se você usar o `hermes setup` {#if-you-use-hermes-setup}

Se você escolher NeuTTS no assistente de configuração, o Hermes verifica se o `neutts` já está instalado. Se estiver faltando, o assistente diz que o NeuTTS precisa do pacote Python `neutts` e do pacote de sistema `espeak-ng`, oferece instalá-los para você, instala o `espeak-ng` com o gerenciador de pacotes da sua plataforma, e então executa:

```bash
python -m pip install -U neutts[all]
```

Se você pular essa instalação ou ela falhar, o assistente recorre ao Edge TTS.

## Passo 5: configuração recomendada {#step-5-recommended-config}

```yaml
voice:
  record_key: "ctrl+b"
  max_recording_seconds: 120
  auto_tts: false
  beep_enabled: true
  silence_threshold: 200
  silence_duration: 3.0

stt:
  provider: "local"
  local:
    model: "base"

tts:
  provider: "edge"
  edge:
    voice: "en-US-AriaNeural"
```

Este é um bom padrão conservador para a maioria das pessoas.

Se você quiser TTS local em vez disso, troque o bloco `tts` para:

```yaml
tts:
  provider: "neutts"
  neutts:
    ref_audio: ''
    ref_text: ''
    model: neuphonic/neutts-air-q4-gguf
    device: cpu
```

## Caso de Uso 1: modo de voz na CLI {#use-case-1-cli-voice-mode}

## Ative-o {#turn-it-on}

Inicie o Hermes:

```bash
hermes
```

Dentro da CLI:

```text
/voice on
```

### Fluxo de Gravação {#recording-flow}

Tecla padrão:
- `Ctrl+B`

Fluxo de trabalho:
1. pressione `Ctrl+B`
2. fale
3. espere a detecção de silêncio parar a gravação automaticamente
4. o Hermes transcreve e responde
5. se o TTS estiver ativado, ele fala a resposta
6. o loop pode reiniciar automaticamente para uso contínuo

### Comandos Úteis {#useful-commands}

```text
/voice
/voice on
/voice off
/voice tts
/voice status
```

### Bons Fluxos de Trabalho na CLI {#good-cli-workflows}

#### Depuração improvisada {#walk-up-debugging}

Diga:

```text
I keep getting a docker permission error. Help me debug it.
```

Depois continue sem usar as mãos:
- "Leia o último erro de novo"
- "Explique a causa raiz em termos mais simples"
- "Agora me dê a correção exata"

#### Pesquisa / brainstorming {#research--brainstorming}

Ótimo para:
- caminhar enquanto pensa
- ditar ideias semiformadas
- pedir ao Hermes para estruturar seus pensamentos em tempo real

#### Acessibilidade / sessões com pouca digitação {#accessibility--low-typing-sessions}

Se digitar é inconveniente, o modo de voz é uma das formas mais rápidas de permanecer no ciclo completo do Hermes.

## Ajustando o Comportamento na CLI {#tuning-cli-behavior}

### Limiar de silêncio {#silence-threshold}

Se o Hermes inicia/para de forma muito agressiva, ajuste:

```yaml
voice:
  silence_threshold: 250
```

Limiar mais alto = menos sensível.

### Duração do silêncio {#silence-duration}

Se você faz muitas pausas entre frases, aumente:

```yaml
voice:
  silence_duration: 4.0
```

### Tecla de gravação {#record-key}

Se `Ctrl+B` conflita com seus hábitos de terminal ou tmux:

```yaml
voice:
  record_key: "ctrl+space"
```

## Caso de Uso 2: respostas em voz no Telegram ou Discord {#use-case-2-voice-replies-in-telegram-or-discord}

Esse modo é mais simples do que canais de voz completos.

O Hermes permanece um bot de chat normal, mas pode falar respostas.

### Inicie o gateway {#start-the-gateway}

```bash
hermes gateway
```

### Ative as respostas por voz {#turn-on-voice-replies}

Dentro do Telegram ou Discord:

```text
/voice on
```

ou

```text
/voice tts
```

### Modos {#modes}

| Modo | Significado |
|---|---|
| `off` | apenas texto |
| `voice_only` | fala apenas quando o usuário enviou voz |
| `all` | fala toda resposta |

### Quando usar qual modo {#when-to-use-which-mode}

- `/voice on` se você quer respostas faladas apenas para mensagens originadas em voz
- `/voice tts` se você quer um assistente falado completo o tempo todo

### Bons Fluxos de Trabalho em Mensageria {#good-messaging-workflows}

#### Assistente do Telegram no seu telefone {#telegram-assistant-on-your-phone}

Use quando:
- você está longe da sua máquina
- você quer enviar notas de voz e receber respostas faladas rápidas
- você quer que o Hermes funcione como um assistente portátil de pesquisa ou operações

#### DMs do Discord com saída falada {#discord-dms-with-spoken-output}

Útil quando você quer interação privada sem o comportamento de menção em canais de servidor.

## Caso de Uso 3: canais de voz do Discord {#use-case-3-discord-voice-channels}

Este é o modo mais avançado.

O Hermes entra em um canal de voz do Discord, escuta a fala dos usuários, a transcreve, executa o pipeline normal do agente e fala as respostas de volta no canal.

## Permissões Necessárias no Discord {#required-discord-permissions}

Além da configuração normal de bot de texto, certifique-se de que o bot tenha:
- Connect
- Speak
- de preferência Use Voice Activity

Também habilite os intents privilegiados no Developer Portal:
- Presence Intent
- Server Members Intent
- Message Content Intent

## Entrar e Sair {#join-and-leave}

Em um canal de texto do Discord onde o bot está presente:

```text
/voice join
/voice leave
/voice status
```

### O que acontece quando entra {#what-happens-when-joined}

- os usuários falam no canal de voz
- o Hermes detecta os limites da fala
- as transcrições são postadas no canal de texto associado
- o Hermes responde em texto e áudio
- o canal de texto é aquele onde `/voice join` foi emitido

### Boas Práticas para Uso de Canal de Voz do Discord {#best-practices-for-discord-vc-use}

- mantenha `DISCORD_ALLOWED_USERS` restrito
- use um canal de bot/teste dedicado no início
- verifique se STT e TTS funcionam no modo de voz normal em chat de texto antes de tentar o modo de canal de voz

## Recomendações de Qualidade de Voz {#voice-quality-recommendations}

### Melhor configuração de qualidade {#best-quality-setup}

- STT: `large-v3` local ou Groq `whisper-large-v3`
- TTS: ElevenLabs

### Melhor configuração de velocidade / conveniência {#best-speed--convenience-setup}

- STT: `base` local ou Groq
- TTS: Edge

### Melhor configuração de custo zero {#best-zero-cost-setup}

- STT: local
- TTS: Edge

## Modos Comuns de Falha {#common-failure-modes}

### "Nenhum dispositivo de áudio encontrado" {#no-audio-device-found}

Instale o `portaudio`.

### "O bot entra, mas não ouve nada" {#bot-joins-but-hears-nothing}

Verifique:
- seu ID de usuário do Discord está em `DISCORD_ALLOWED_USERS`
- você não está mutado
- os intents privilegiados estão habilitados
- o bot tem permissões de Connect/Speak

### "Ele transcreve, mas não fala" {#it-transcribes-but-does-not-speak}

Verifique:
- configuração do provedor de TTS
- chave de API / cota do ElevenLabs ou OpenAI
- instalação do `ffmpeg` para caminhos de conversão do Edge

### "O Whisper produz coisas sem sentido" {#whisper-outputs-garbage}

Tente:
- ambiente mais silencioso
- `silence_threshold` mais alto
- provedor/modelo de STT diferente
- frases mais curtas e claras

### "Funciona em DMs, mas não em canais de servidor" {#it-works-in-dms-but-not-in-server-channels}

Isso frequentemente é a política de menção.

Por padrão, o bot precisa de uma `@menção` em canais de texto de servidor no Discord, a menos que configurado de outra forma.

## Configuração Sugerida para a Primeira Semana {#suggested-first-week-setup}

Se você quer o caminho mais curto para o sucesso:

1. faça o Hermes em texto funcionar
2. instale `hermes-agent[voice]`
3. use o modo de voz na CLI com STT local + Edge TTS
4. depois habilite `/voice on` no Telegram ou Discord
5. só depois disso, tente o modo de canal de voz do Discord

Essa progressão mantém a superfície de depuração pequena.

## Onde Ler a Seguir {#where-to-read-next}

- [Referência do recurso Modo de Voz](/user-guide/features/voice-mode)
- [Gateway de Mensageria](/user-guide/messaging)
- [Configuração do Discord](/user-guide/messaging/discord)
- [Configuração do Telegram](/user-guide/messaging/telegram)
- [Configuração](/user-guide/configuration)
