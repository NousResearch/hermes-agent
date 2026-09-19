---
sidebar_position: 11
title: "Palavra de Ativação"
description: "Palavra de ativação 'Hey Hermes' mãos-livres — inicie uma sessão de voz falando, ao estilo 'Hey Siri'"
---

# Palavra de Ativação ("Hey Hermes")

A palavra de ativação transforma o Hermes em um assistente mãos-livres na CLI, na TUI
e no app desktop: com uma única configuração ligada, o Hermes escuta em segundo plano
uma frase de disparo falada. Diga-a, e o Hermes inicia uma sessão nova, abre o microfone,
captura seu comando pelo [pipeline de voz](/user-guide/features/voice-mode) normal
e responde — exatamente como "Hey Siri" ou "Alexa". Use `surface` para escolher qual
superfície escuta.

A detecção roda **inteiramente no dispositivo**. O listener sempre ativo só observa
a frase de ativação; nenhum áudio sai da sua máquina até que você realmente fale um
comando para o agente.

## Como funciona {#how-it-works}

1. Com `wake_word.enabled: true` (ou depois de `/wake on`), um detector leve de hotword
   escuta no dispositivo de entrada configurado, ou no microfone padrão do processo
   quando `wake_word.input_device` não está definido.
2. Quando ouve a frase de ativação, ele se pausa (liberando o microfone), inicia uma
   nova sessão e grava uma fala usando a detecção de silêncio do modo de voz.
3. Sua fala é transcrita e enviada ao agente. Depois que ele responde, o listener
   retoma automaticamente e aguarda a próxima palavra de ativação.

Ela vem **desativada por padrão** — nada escuta até que você a ative.

No app desktop, uma conversa de voz mãos-livres pode ser encerrada simplesmente
**dizendo "stop"** (ou "never mind", "goodbye", "cancel", "that's all") — o comando
falado encerra a conversa em vez de ser enviado ao agente. Apenas um comando de parada
que ocupe a fala inteira é reconhecido, então um pedido real como "stop the docker
container" ainda passa normalmente.

## Engines {#engines}

| Engine | Custo | Chave de API | Notas |
|--------|------|---------|-------|
| **openWakeWord** (padrão) | Gratuito | Nenhuma | Modelos ONNX locais. Vem com um modelo **"hey hermes"** embutido (padrão); também suporta `hey_jarvis`, `alexa`, `hey_mycroft`, … e modelos personalizados |
| **sherpa** | Gratuito | Nenhuma | **Vocabulário aberto** — detecta QUALQUER frase digitada sem nenhum treinamento. Um modelo pequeno em inglês baixa automaticamente no primeiro uso (~13 MB) |
| **Porcupine** | Camada gratuita/paga | `PORCUPINE_ACCESS_KEY` | Engine da Picovoice; palavras-chave embutidas + arquivos `.ppn` personalizados |

Por padrão, a frase é **"hey hermes"** — um modelo para ela já vem com o Hermes, então
funciona de imediato sem nenhum treinamento. (No primeiro uso, o openWakeWord baixa
seus modelos compartilhados de extração de características — uma pequena busca única.)

Ambos são instalados sob demanda na primeira vez que você ativa a palavra de ativação
(instalações desktop feitas com `--include-desktop` os pré-instalam, então o "ouvido"
funciona instantaneamente). Para instalar antecipadamente:

```bash
cd ~/.hermes/hermes-agent && uv pip install -e ".[wake]"
```

## Início rápido {#quick-start}

```bash
# In an interactive `hermes` session:
/wake on        # start listening (installs the engine on first use)
/wake status    # show phrase, provider, and state
/wake off       # stop listening
```

No app desktop, clique no ícone de orelha no compositor.

O botão de alternância É a configuração: ligar ou desligar a palavra de ativação —
via `/wake` ou o botão de orelha do desktop — também grava `wake_word.enabled` em
`~/.hermes/config.yaml`, então sua escolha persiste entre sessões. Você também pode
alterá-la manualmente:

```yaml
wake_word:
  enabled: true
```

## Configuração {#configuration}

```yaml
wake_word:
  enabled: false
  surface: auto               # eligible surface: "auto" | "cli" | "tui" | "gui"
  input_device: null           # PortAudio input index or device-name substring; null = process default
  provider: openwakeword      # "openwakeword" (free, local) | "sherpa" (free, any phrase) | "porcupine"
  phrase: "hey hermes"        # cosmetic label only — detection is keyed by the model/keyword below
  sensitivity: 0.6            # 0.0-1.0 — higher = stricter (fewer false triggers), consistent across all engines
  confirmation_frames: 3      # openWakeWord only — consecutive over-threshold frames required to fire
  start_new_session: true     # start a fresh session on wake vs. continue the current one
  openwakeword:
    model: hey_hermes         # bundled default; OR a built-in name OR a path to a custom .onnx/.tflite
    inference_framework: ""   # "" (auto) | "onnx" | "tflite"
  porcupine:
    keyword: jarvis           # built-in keyword OR path to a custom .ppn
```

`sensitivity`, `phrase` e `start_new_session` se aplicam a ambos os engines. Os blocos
`openwakeword` e `porcupine` selecionam o modelo de detecção de fato.

`input_device` é passado diretamente para o stream PortAudio (`sounddevice`) do
listener de ativação. Use um índice numérico de dispositivo ou uma substring
inequívoca do nome do dispositivo. Essa configuração só muda a captura da palavra de
ativação; o push-to-talk do desktop continua usando o caminho de microfone do próprio
aplicativo desktop.

### Reduzindo falsos disparos por fala ambiente {#reducing-false-triggers-on-ambient-speech}

O openWakeWord pontua um quadro curto de áudio (~80ms) por vez, então um fonema
aleatório em uma conversa de fundo pode ocasionalmente elevar um único quadro acima
do limiar e disparar a palavra de ativação sem intenção. Dois ajustes controlam isso:

- **`confirmation_frames`** (padrão `3`, apenas openWakeWord) — quantos quadros
  *consecutivos* acima do limiar são necessários antes de disparar. Um "hey hermes"
  real mantém uma pontuação alta ao longo de vários quadros; um ruído ambiente eleva
  apenas um. Aumente esse valor (por exemplo, `4`–`5`) se ainda houver falsos disparos
  em um ambiente barulhento; o custo é algumas dezenas de milissegundos extras de
  latência. `1` restaura o comportamento antigo de disparo no primeiro quadro.
- **`sensitivity`** (padrão `0.6`) — o limiar de detecção, de `0.0` a `1.0`. Quanto
  mais alto, mais rígido (menos falsos disparos). Essa direção é consistente em
  **todos** os engines — para o openWakeWord é o limiar bruto de pontuação por quadro,
  para o sherpa ele mapeia para o limiar da palavra-chave, e para o Porcupine ele é
  invertido internamente para que "mais alto = mais rígido" também valha lá. O padrão
  `0.6` fica acima da linha de base permissiva de `0.5` do openWakeWord, que deixava
  passar quase-acertos como "hey hor"; aumente até `0.8` se ainda houver falsos
  disparos, ou diminua se falas reais de "hey hermes" estiverem sendo perdidas.

Os engines `sherpa` e `porcupine` decodificam a frase inteira internamente, então não
têm o problema de pico em um único quadro e ignoram `confirmation_frames` (mas ainda
respeitam `sensitivity`).

`inference_framework` escolhe o backend do openWakeWord. Deixe vazio (o padrão) para
que o Hermes escolha por plataforma: **tflite em Apple Silicon**, onnx em todo o
resto. O backend onnx do openWakeWord retorna pontuações próximas de zero em macOS
ARM64 ([openWakeWord#336](https://github.com/dscripka/openWakeWord/issues/336)), então
um listener fixado em `onnx` ali vai armar, aparecer como escutando e nunca disparar.
O backend tflite precisa de `ai-edge-litert` no macOS, que o Hermes instala sob
demanda junto com as outras dependências da palavra de ativação.

### Superfícies (CLI, TUI, GUI) {#surfaces-cli-tui-gui}

A palavra de ativação funciona nas três superfícies do Hermes, e `surface` escolhe
qual delas é dona do listener e abre a nova sessão quando ele dispara:

| `surface` | Comportamento |
|-----------|----------|
| `auto` (padrão) | Todas as superfícies locais são elegíveis; a primeira a armar fica com o listener. |
| `cli` | Apenas a CLI clássica `hermes`. |
| `tui` | Apenas `hermes --tui`. |
| `gui` | Apenas o app desktop. |

O detector é local e usa um único microfone, então apenas uma superfície escuta por
vez, mesmo quando as superfícies do Hermes rodam em processos separados. A posse é
persistente: o primeiro candidato elegível mantém o listener até parar, desconectar
ou seu processo encerrar. O Hermes não faz failover silencioso para outra superfície
aberta. Defina `surface` quando quiser fixar a posse em vez de usar o critério de
"primeiro a reivindicar vence". A TUI e a GUI desktop compartilham o mesmo backend
Python (`tui_gateway`), que roda o detector no lado do servidor e cede o microfone
para a captura de voz enquanto um comando é gravado.

## Usando uma frase diferente {#using-a-different-phrase}

"Hey Hermes" funciona de imediato — o modelo openWakeWord embutido (`model:
hey_hermes`) é o padrão. Para ativar com outra frase, o caminho mais fácil é o
engine de vocabulário aberto:

### Opção A — sherpa (qualquer frase, sem treinamento) {#option-a--sherpa-any-phrase-zero-training}

Digite a frase que você quer; ela é tokenizada em tempo de execução — "hey coder",
"computer", "wake up neo", qualquer coisa:

```yaml
wake_word:
  enabled: true
  provider: sherpa
  phrase: "hey coder"        # detection key — just type your phrase
```

O pequeno modelo KWS em inglês (~13 MB) baixa uma vez no primeiro uso. Cada perfil
pode ter sua própria frase — "hey \<perfil\>" para cada perfil que você rodar.

### Ativando um perfil específico (desktop) {#waking-a-specific-profile-desktop}

Com o engine sherpa, UM único listener pode ativar QUALQUER perfil. Todo perfil cuja
configuração tenha `wake_word.enabled: true` é registrado automaticamente; sua frase
assume `hey <nome-do-perfil>` por padrão quando não definida. Diga a frase de um
perfil e o app desktop troca de perfil em tempo real, abre uma sessão nova ali e
inicia a voz mãos-livres:

- "hey hermes" → perfil padrão
- "hey coder" → o perfil `coder`
- "hey trader" → o perfil `trader`

Defina `wake_word.profile_routing: false` no perfil do listener para não participar
disso e escutar apenas pela própria frase. A CLI e a TUI são processos de perfil
único: uma frase de ativação pertencente a outro perfil imprime o comando de troca
(`hermes -p <perfil>`) em vez de rotear.

Os nomes são reconhecidos acusticamente por seus sons de subpalavras em inglês:
frases de duas palavras com nomes distintos, de 2+ sílabas, funcionam melhor. Nomes
muito curtos, fonologia fortemente não inglesa, ou dois perfis com nomes de som
parecido vão degradar a precisão — ajuste a `sensitivity` por perfil se necessário.

### Opção B — openWakeWord (gratuito, modelo treinado) {#option-b--openwakeword-free-trained-model}

Escolha um modelo embutido (`hey_jarvis`, `alexa`, `hey_mycroft`, …), ou treine um
modelo personalizado (≈75–90 min em uma GPU gratuita/Colab) para máxima robustez,
coloque o arquivo `.onnx` em algum lugar e referencie-o:

```yaml
wake_word:
  enabled: true
  provider: openwakeword
  phrase: "computer"
  openwakeword:
    model: ~/.hermes/wakewords/computer.onnx   # or a built-in name like hey_jarvis
```

Referências para treinamento:

- [openWakeWord](https://github.com/dscripka/openWakeWord)
- [Colab de treinamento 2026](https://github.com/alfiedennen/openwakeword-colab-2026)

:::tip Escolha uma frase distintiva
Frases de ativação que não colidem com a fala cotidiana generalizam melhor. Duas
sílabas com uma palavra incomum ("hermes" se qualifica) vencem palavras comuns como
"hello" ou "stop".
:::

### Opção C — Porcupine (palavra-chave personalizada em segundos) {#option-c--porcupine-custom-keyword-in-seconds}

Crie uma palavra-chave "Hey Hermes" no [Picovoice Console](https://console.picovoice.ai/),
baixe o `.ppn` e:

```yaml
wake_word:
  enabled: true
  provider: porcupine
  phrase: "hey hermes"
  porcupine:
    keyword: ~/.hermes/wakewords/hey_hermes.ppn
```

Defina sua chave de acesso em `~/.hermes/.env`:

```bash
PORCUPINE_ACCESS_KEY=your-key-here
```

## Requisitos {#requirements}

- Um microfone funcionando e a stack de áudio `sounddevice` + `numpy` (compartilhada
  com o modo de voz).
- Um provedor de STT para transcrever o comando falado — o `faster-whisper` local
  funciona de imediato; veja [Modo de Voz](/user-guide/features/voice-mode) para a
  lista completa de provedores.
- Um provedor de TTS para falar a resposta (o `edge-tts` padrão funciona sem chave).
  O fluxo de ativação é totalmente mãos-livres, então o botão de alternância se
  recusa a armar até que tanto o STT quanto o TTS estejam prontos — `hermes tools`
  (seção Voice) configura ambos.
- As dependências do engine de ativação (instaladas automaticamente, ou
  `hermes-agent[wake]`).

`/wake status` reporta exatamente o que está faltando se o listener não iniciar.

### "Listening" mas nunca ativa (macOS) {#listening-but-never-wakes-macos}

O macOS concede acesso ao microfone por **processo**. O STT funcionando no app
desktop prova que o *renderer* tem acesso ao microfone — o listener de ativação roda
no *backend* Python, que precisa da sua própria permissão. Sem ela, o CoreAudio
entrega ao backend um stream "funcionando" que nunca produz nada além de
silêncio, então o "ouvido" mostra que está escutando, mas a frase nunca dispara. O
Hermes detecta isso (`/wake status` mostra "mic delivers only silence"; a dica no
tooltip da orelha do desktop traz a mesma informação). Correção: Ajustes do Sistema →
Privacidade e Segurança → Microfone → habilite o backend do Hermes (pode aparecer
como seu terminal, `python` ou Hermes), depois desligue e ligue a palavra de
ativação novamente.

### "Listening" mas recebe silêncio (Windows) {#listening-but-receives-silence-windows}

O push-to-talk do desktop e a captura da palavra de ativação usam caminhos de
microfone diferentes. O push-to-talk usa a captura do navegador do aplicativo
desktop, enquanto o listener de palavra de ativação abre um stream PortAudio no
backend Python. Um pode funcionar enquanto o outro seleciona uma entrada do Windows
silenciosa ou inutilizável.

`/wake status` reporta o dispositivo de entrada selecionado e a API de host de áudio
do Windows. Quando ele reportar silêncio, defina `wake_word.input_device` com o
índice numérico ou um nome inequívoco da entrada PortAudio que funciona, e depois
alterne a palavra de ativação:

```bash
hermes config set wake_word.input_device "Microphone Array"
```

Use `null` para voltar ao padrão do processo:

```bash
hermes config set wake_word.input_device null
```

## Observações e limites {#notes--limits}

- **Apenas superfícies locais.** A palavra de ativação roda na CLI, na TUI e na GUI
  desktop — onde quer que um microfone local esteja disponível. Ela não roda no
  gateway de mensagens (Telegram, Discord, …), que não tem microfone.
- **Um microfone por vez.** O detector libera o microfone enquanto um comando está
  sendo gravado e o retoma assim que o turno termina, então não disputa com a
  captura de voz.
- **Privacidade.** A detecção de hotword é local. Aumente `sensitivity` se você tiver
  falsos disparos, diminua se ela deixar de te ouvir.
