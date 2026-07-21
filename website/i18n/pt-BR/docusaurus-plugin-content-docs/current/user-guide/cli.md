---
sidebar_position: 1
title: "Interface CLI"
description: "Domine a interface de terminal do Hermes Agent — comandos, atalhos de teclado, personalidades e muito mais"
---

# Interface CLI

O CLI do Hermes Agent é uma interface de terminal completa (TUI) — não é uma UI web. Ele oferece edição multilinha, autocompletar de slash commands, histórico de conversa, interrupção com redirecionamento e saída de ferramentas em streaming. Feito para quem vive no terminal.

:::tip Primeira configuração
Um comando — `hermes setup --portal` — e você já pode rodar `hermes chat`. Veja o [Nous Portal](/integrations/nous-portal).
:::

:::tip
O Hermes também inclui uma TUI moderna com overlays modais, seleção por mouse e input não bloqueante. Inicie com `hermes --tui` — veja o guia da [TUI](tui.md).
:::

## Executando o CLI

```bash
# Start an interactive session (default)
hermes

# Single query mode (non-interactive)
hermes chat -q "Hello"

# With a specific model
hermes chat --model "anthropic/claude-sonnet-4"

# With a specific provider
hermes chat --provider nous        # Use Nous Portal
hermes chat --provider openrouter  # Force OpenRouter

# With specific toolsets
hermes chat --toolsets "web,terminal,skills"

# Start with one or more skills preloaded
hermes -s hermes-agent-dev,github-auth
hermes chat -s github-pr-workflow -q "open a draft PR"

# Resume previous sessions
hermes --continue             # Resume the most recent CLI session (-c)
hermes --resume <session_id>  # Resume a specific session by ID (-r)

# Verbose mode (debug output)
hermes chat --verbose

# Isolated git worktree (for running multiple agents in parallel)
hermes -w                         # Interactive mode in worktree
hermes -w -z "Fix issue #123"     # Single query in worktree
```

## Layout da interface

<img className="docs-terminal-figure" src="/docs/img/docs/cli-layout.svg" alt="Prévia estilizada do layout do CLI do Hermes mostrando o banner, a área de conversa e o prompt de input fixo." />
<p className="docs-figure-caption">O banner do CLI do Hermes, o fluxo de conversa e o prompt de input fixo renderizados como uma figura estável de documentação em vez de arte ASCII frágil.</p>

O banner de boas-vindas mostra de relance o seu modelo, backend de terminal, diretório de trabalho, ferramentas disponíveis e skills instaladas.

### Barra de status

Uma barra de status persistente fica acima da área de input, atualizando em tempo real:

```
 ⚕ claude-sonnet-4-20250514 │ 12.4K/200K │ [██████░░░░] 6% │ $0.06 │ 15m
```

| Elemento | Descrição |
|---------|-------------|
| Nome do modelo | Modelo atual (truncado se tiver mais de 26 caracteres) |
| Contagem de tokens | Tokens de contexto usados / janela máxima de contexto |
| Barra de contexto | Indicador visual de preenchimento com limites codificados por cor |
| Custo | Custo estimado da sessão (ou `n/a` para modelos com preço desconhecido/zero) |
| 🗜️ N | **Contagem de compressão de contexto** — quantas vezes a sessão em execução foi comprimida automaticamente. Aparece depois que a primeira compressão ocorre. |
| ▶ N | **Tarefas em background ativas** — quantos prompts `/background` ainda estão rodando na sessão atual. Aparece sempre que pelo menos uma tarefa está em andamento. |
| Duração | Tempo decorrido da sessão |
| ⚠ YOLO | **Aviso do modo YOLO** — exibido sempre que `HERMES_YOLO_MODE` está ativo (seja `hermes --yolo` na inicialização ou `/yolo` alternado no meio da sessão). Espelha o aviso da linha do banner para você não esquecer que está no modo de aprovação automática. |

A barra se adapta à largura do terminal — layout completo com ≥ 76 colunas, compacto entre 52–75, mínimo (modelo + duração, mais o badge YOLO quando ativo) abaixo de 52.

**Codificação de cores do contexto:**

| Cor | Limite | Significado |
|-------|-----------|---------|
| Verde | < 50% | Espaço de sobra |
| Amarelo | 50–80% | Ficando cheio |
| Laranja | 80–95% | Aproximando do limite |
| Vermelho | ≥ 95% | Perto do overflow — considere `/compress` |

Use `/usage` para um detalhamento completo, incluindo custos por categoria (tokens de input vs output).

No provider `openai-codex`, `/usage` também mostra resets de limite de uso acumulados na sua conta ChatGPT ("You have N resets banked - use /usage reset to activate"). `/usage reset` resgata um reset acumulado, restaurando por completo os seus limites de 5 horas e semanais. O Hermes se recusa a resgatar enquanto os seus limites não estiverem esgotados (um reset acumulado restaura a cota completa, então gastá-lo cedo é desperdício) — passe `/usage reset --force` para resgatar mesmo assim.

### Exibição ao retomar sessão

Ao retomar uma sessão anterior (`hermes -c` ou `hermes --resume <id>`), um painel "Previous Conversation" aparece entre o banner e o prompt de input, mostrando um resumo compacto do histórico de conversa. Veja [Sessões — Resumo da conversa ao retomar](sessions.md#conversation-recap-on-resume) para detalhes e configuração.

## Atalhos de teclado {#keybindings}

| Tecla | Ação |
|-----|--------|
| `Enter` | Enviar mensagem |
| `Alt+Enter`, `Ctrl+J` ou `Shift+Enter` | Nova linha (input multilinha). `Shift+Enter` exige um terminal que o distinga de `Enter` — veja abaixo. No Windows Terminal, `Alt+Enter` é capturado pelo terminal (alternar tela cheia); use `Ctrl+Enter` ou `Ctrl+J` em vez disso. |
| `Alt+V` | Colar uma imagem da área de transferência quando o terminal suportar |
| `Ctrl+V` | Colar texto e anexar imagens da área de transferência quando possível |
| `Ctrl+B` | Iniciar/parar gravação de voz quando o modo de voz estiver ativo (`voice.record_key`, padrão: `ctrl+b`) |
| `Ctrl+G` | Abrir o buffer de input atual no `$EDITOR` (vim/nvim/nano/VS Code/etc.). Salve e saia para enviar o texto editado como o próximo prompt — ideal para prompts longos com vários parágrafos. |
| `Ctrl+X Ctrl+E` | Atalho alternativo estilo Emacs para o editor externo (mesmo comportamento de `Ctrl+G`). |
| `Ctrl+C` | Interromper o agente (pressione duas vezes em até 2s para forçar a saída) |
| `Ctrl+D` | Sair |
| `Ctrl+Z` | Suspender o Hermes em background (somente Unix). Execute `fg` no shell para retomar. |
| `Tab` | Aceitar sugestão automática (texto fantasma) ou autocompletar slash commands |

**Prévia de colagem multilinha.** Quando você cola um bloco multilinha, o CLI ecoa uma prévia compacta em linha única (`[pasted: 47 lines, 1,842 chars — press Enter to send]`) em vez de despejar todo o conteúdo no scrollback. O conteúdo completo ainda é o que será enviado; isso é apenas polimento visual.

**Remoção de markdown nas respostas finais.** O CLI remove as cercas de markdown mais verbosas e os wrappers `**bold**` / `*italic*` das respostas *finais* do agente para que renderizem como prosa legível no terminal em vez de código-fonte bruto. Blocos de código e listas são preservados. Isso não afeta plataformas de messaging ou resultados de ferramentas — elas mantêm o markdown para renderização nativa.

## Slash commands

Digite `/` para ver o menu suspenso de autocompletar. O Hermes suporta um grande conjunto de slash commands de CLI, slash commands dinâmicos de skills e quick commands definidos pelo usuário.

Exemplos comuns:

| Comando | Descrição |
|---------|-------------|
| `/help` | Mostrar ajuda de comandos |
| `/model` | Mostrar ou alterar o modelo atual |
| `/tools` | Listar ferramentas disponíveis no momento |
| `/skills browse` | Navegar pelo hub de skills e skills opcionais oficiais |
| `/background <prompt>` | Executar um prompt em uma sessão em background separada |
| `/skin` | Mostrar ou alternar a skin ativa do CLI |
| `/voice on` | Ativar o modo de voz do CLI (pressione `Ctrl+B` para gravar) |
| `/voice tts` | Alternar reprodução falada das respostas do Hermes |
| `/reasoning high` | Aumentar o esforço de raciocínio |
| `/title My Session` | Nomear a sessão atual |
| `/status` | Mostrar informações da sessão — modelo/perfil/tokens/duração — seguidas de um bloco local de **Resumo da sessão** (contagens de turnos recentes, principais ferramentas usadas, arquivos tocados, último prompt do usuário + resposta do assistente). Cálculo puramente local; sem chamada ao LLM. |
| `/sessions` | Abrir um seletor interativo de sessões direto no CLI clássico (a mesma superfície que a TUI usa). Digite para filtrar, setas para navegar, Enter para retomar. |

Para as listas completas de CLI e messaging, veja a [Referência de slash commands](../reference/slash-commands.md).

Para setup, providers, ajuste de silêncio e uso de voz em messaging/Discord, veja [Modo de voz](features/voice-mode.md).

:::tip
Os comandos não diferenciam maiúsculas de minúsculas — `/HELP` funciona igual a `/help`. Skills instaladas também viram slash commands automaticamente.
:::

## Quick commands

Você pode definir comandos personalizados que executam comandos de shell instantaneamente sem invocar o LLM. Eles funcionam tanto no CLI quanto em plataformas de messaging (Telegram, Discord, etc.).

```yaml
# ~/.hermes/config.yaml
quick_commands:
  status:
    type: exec
    command: systemctl status hermes-agent
  gpu:
    type: exec
    command: nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
  restart:
    type: alias
    target: /gateway restart
```

Depois digite `/status`, `/gpu` ou `/restart` em qualquer chat. Veja o [guia de configuração](/user-guide/configuration#quick-commands) para mais exemplos.

## Pré-carregar skills na inicialização

Se você já sabe quais skills quer ativas na sessão, passe-as na inicialização:

```bash
hermes -s hermes-agent-dev,github-auth
hermes chat -s github-pr-workflow -s github-auth
```

O Hermes carrega cada skill nomeada no prompt da sessão antes do primeiro turno. A mesma flag funciona no modo interativo e no modo de consulta única.

## Slash commands de skills

Cada skill instalada em `~/.hermes/skills/` é registrada automaticamente como slash command. O nome da skill vira o comando:

```
/gif-search funny cats
/axolotl help me fine-tune Llama 3 on my dataset
/github-pr-workflow create a PR for the auth refactor

# Just the skill name loads it and lets the agent ask what you need:
/excalidraw
```

## Personalidades

Defina uma personalidade predefinida para mudar o tom do agente:

```
/personality pirate
/personality kawaii
/personality concise
```

Personalidades integradas incluem: `helpful`, `concise`, `technical`, `creative`, `teacher`, `kawaii`, `catgirl`, `pirate`, `shakespeare`, `surfer`, `noir`, `uwu`, `philosopher`, `hype`.

Você também pode definir personalidades personalizadas em `~/.hermes/config.yaml`:

```yaml
personalities:
  helpful: "You are a helpful, friendly AI assistant."
  kawaii: "You are a kawaii assistant! Use cute expressions..."
  pirate: "Arrr! Ye be talkin' to Captain Hermes..."
  # Add your own!
```

## Input multilinha

Há duas formas de inserir mensagens multilinha:

1. **`Alt+Enter`, `Ctrl+J` ou `Shift+Enter`** — insere uma nova linha
2. **Continuação com barra invertida** — termine uma linha com `\` para continuar:

```
❯ Write a function that:\
  1. Takes a list of numbers\
  2. Returns the sum
```

:::info
Colar texto multilinha é suportado — use qualquer uma das teclas de nova linha acima, ou simplesmente cole o conteúdo diretamente.
:::

### Compatibilidade com Shift+Enter

A maioria dos terminais envia a mesma sequência de bytes para `Enter` e `Shift+Enter` por padrão, então aplicações não conseguem distingui-los. O Hermes reconhece `Shift+Enter` somente quando o terminal envia uma sequência distinta via o [protocolo de teclado Kitty](https://sw.kovidgoyal.net/kitty/keyboard-protocol/) ou o modo `modifyOtherKeys` do xterm.

| Terminal | Status |
|---|---|
| Kitty, foot, WezTerm, Ghostty | `Shift+Enter` distinto habilitado por padrão |
| iTerm2 (recente), Alacritty, terminal do VS Code, Warp | Suportado depois que o protocolo Kitty é habilitado nas configurações |
| Windows Terminal Preview 1.25+ | Suportado depois que o protocolo Kitty é habilitado nas configurações |
| macOS Terminal.app, Windows Terminal estável padrão | Não suportado — `Shift+Enter` é indistinguível de `Enter` |

Onde o terminal não consegue distingui-los, `Alt+Enter` e `Ctrl+J` continuam funcionando em todo lugar. **No Windows Terminal especificamente, `Alt+Enter` é capturado pelo terminal (alterna tela cheia) e nunca chega ao Hermes — use `Ctrl+Enter` (entregue como `Ctrl+J`) ou `Ctrl+J` diretamente para uma nova linha.**

## Interrompendo o agente

Você pode interromper o agente a qualquer momento:

- **Digite uma nova mensagem + Enter** enquanto o agente está trabalhando — ele interrompe e processa as suas novas instruções
- **`Ctrl+C`** — interrompe a operação atual (pressione duas vezes em até 2s para forçar a saída)
- Comandos de terminal em andamento são encerrados imediatamente (SIGTERM, depois SIGKILL após 1s)
- Várias mensagens digitadas durante a interrupção são combinadas em um único prompt

### Modo de input ocupado

A chave de config `display.busy_input_mode` controla o que acontece quando você pressiona Enter enquanto o agente está trabalhando:

| Modo | Comportamento |
|------|----------|
| `"interrupt"` (padrão) | A sua mensagem interrompe a operação atual e é processada imediatamente |
| `"queue"` | A sua mensagem é enfileirada silenciosamente e enviada como o próximo turno depois que o agente terminar |
| `"steer"` | A sua mensagem é injetada na execução atual via `/steer`, chegando ao agente após a próxima chamada de ferramenta — sem interrupção, sem novo turno |

```yaml
# ~/.hermes/config.yaml
display:
  busy_input_mode: "steer"   # or "queue" or "interrupt" (default)
```

O modo `"queue"` é útil quando você quer preparar mensagens de follow-up sem cancelar acidentalmente trabalho em andamento. O modo `"steer"` é útil quando você quer redirecionar o agente no meio de uma tarefa sem interromper — por exemplo, "na verdade, também verifique os testes" enquanto ele ainda está editando código. Valores desconhecidos voltam para `"interrupt"`.

`"steer"` tem dois fallbacks automáticos: se o agente ainda não iniciou, ou se imagens estão anexadas, a mensagem volta para o comportamento de `"queue"` para que nada se perca.

Você também pode alterar isso dentro do CLI:

```text
/busy queue
/busy steer
/busy interrupt
/busy status
```

:::tip Dica na primeira vez
Na primeira vez que você pressiona Enter enquanto o Hermes está trabalhando, o Hermes imprime um lembrete de uma linha explicando o controle `/busy` (`"(tip) Your message interrupted the current run…"`). Ele só aparece uma vez por instalação — uma flag em `config.yaml` em `onboarding.seen.busy_input_prompt` trava isso. Apague essa chave para ver a dica de novo.
:::

### Suspender em background

Em sistemas Unix, pressione **`Ctrl+Z`** para suspender o Hermes em background — como qualquer processo de terminal. O shell imprime uma confirmação:

```
Hermes Agent has been suspended. Run `fg` to bring Hermes Agent back.
```

Digite `fg` no seu shell para retomar a sessão exatamente de onde parou. Isso não é suportado no Windows.

## Exibição de progresso de ferramentas

O CLI mostra feedback animado enquanto o agente trabalha:

**Animação de pensamento** (durante chamadas à API):
```
  ◜ (｡•́︿•̀｡) pondering... (1.2s)
  ◠ (⊙_⊙) contemplating... (2.4s)
  ✧٩(ˊᗜˋ*)و✧ got it! (3.1s)
```

**Feed de execução de ferramentas:**
```
  ┊ 💻 terminal `ls -la` (0.3s)
  ┊ 🔍 web_search (1.2s)
  ┊ 📄 web_extract (2.1s)
```

Alterne entre modos de exibição com `/verbose`: `off → new → all → verbose`. Este comando também pode ser habilitado para plataformas de messaging — veja [configuração](/user-guide/configuration#display-settings).

### Comprimento da prévia de ferramentas

A chave de config `display.tool_preview_length` controla o número máximo de caracteres exibidos nas linhas de prévia de chamadas de ferramentas (por exemplo, caminhos de arquivo, comandos de terminal). O padrão é `0`, o que significa sem limite — caminhos e comandos completos são exibidos.

```yaml
# ~/.hermes/config.yaml
display:
  tool_preview_length: 80   # Truncate tool previews to 80 chars (0 = no limit)
```

Isso é útil em terminais estreitos ou quando argumentos de ferramentas contêm caminhos de arquivo muito longos.

## Gerenciamento de sessões

### Retomando sessões

Quando você sai de uma sessão CLI, um comando de retomada é impresso:

```
Resume this session with:
  hermes --resume 20260225_143052_a1b2c3

Session:        20260225_143052_a1b2c3
Duration:       12m 34s
Messages:       28 (5 user, 18 tool calls)
```

Opções de retomada:

```bash
hermes --continue                          # Resume the most recent CLI session
hermes -c                                  # Short form
hermes -c "my project"                     # Resume a named session (latest in lineage)
hermes --resume 20260225_143052_a1b2c3     # Resume a specific session by ID
hermes --resume "refactoring auth"         # Resume by title
hermes -r 20260225_143052_a1b2c3           # Short form
```

Retomar restaura o histórico completo de conversa do SQLite. O agente vê todas as mensagens, chamadas de ferramentas e respostas anteriores — como se você nunca tivesse saído.

Use `/title My Session Name` dentro de um chat para nomear a sessão atual, ou `hermes sessions rename <id> <title>` na linha de comando. Use `hermes sessions list` para navegar pelas sessões anteriores.

### Armazenamento de sessões

As sessões CLI ficam armazenadas no banco de estado SQLite do Hermes em `~/.hermes/state.db`. O banco mantém:

- metadados da sessão (ID, título, timestamps, contadores de tokens)
- histórico de mensagens
- linhagem entre sessões comprimidas/retomadas
- índices de busca full-text usados por `session_search`

Alguns adaptadores de messaging também mantêm arquivos de transcrição por plataforma junto ao banco, mas o próprio CLI retoma a partir do armazenamento de sessões SQLite.

### Compressão de contexto

Conversas longas são resumidas automaticamente ao se aproximar dos limites de contexto:

```yaml
# In ~/.hermes/config.yaml
compression:
  enabled: true
  threshold: 0.50    # Compress at 50% of context limit by default

# Summarization model configured under auxiliary:
auxiliary:
  compression:
    model: ""  # Leave empty to use the main chat model (default). Or pin a cheap fast model, e.g. "google/gemini-3-flash-preview".
```

Quando a compressão dispara, turnos do meio são resumidos enquanto os 3 primeiros e os 20 últimos turnos são sempre preservados.

## Sessões em background {#background-sessions}

Execute um prompt em uma sessão em background separada enquanto continua usando o CLI para outro trabalho:

```
/background Analyze the logs in /var/log and summarize any errors from today
```

O Hermes confirma a tarefa imediatamente e devolve o prompt:

```
🔄 Background task #1 started: "Analyze the logs in /var/log and summarize..."
   Task ID: bg_143022_a1b2c3
```

### Como funciona

Cada prompt `/background` cria uma **sessão de agente completamente separada** em uma thread daemon:

- **Conversa isolada** — o agente em background não tem conhecimento do histórico da sua sessão atual. Ele recebe apenas o prompt que você fornece.
- **Mesma configuração** — o agente em background herda o seu modelo, provider, toolsets, configurações de raciocínio e modelo de fallback da sessão atual.
- **Não bloqueante** — a sua sessão em primeiro plano permanece totalmente interativa. Você pode conversar, executar comandos ou até iniciar mais tarefas em background.
- **Várias tarefas** — você pode executar várias tarefas em background simultaneamente. Cada uma recebe um ID numerado.

### Resultados

Quando uma tarefa em background termina, o resultado aparece como um painel no seu terminal:

```
╭─ ⚕ Hermes (background #1) ──────────────────────────────────╮
│ Found 3 errors in syslog from today:                         │
│ 1. OOM killer invoked at 03:22 — killed process nginx        │
│ 2. Disk I/O error on /dev/sda1 at 07:15                      │
│ 3. Failed SSH login attempts from 192.168.1.50 at 14:30      │
╰──────────────────────────────────────────────────────────────╯
```

Se a tarefa falhar, você verá uma notificação de erro em vez disso. Se `display.bell_on_complete` estiver habilitado na sua config, o sino do terminal toca quando a tarefa termina.

### Casos de uso

- **Pesquisa demorada** — "/background research the latest developments in quantum error correction" enquanto você trabalha no código
- **Processamento de arquivos** — "/background analyze all Python files in this repo and list any security issues" enquanto você continua uma conversa
- **Investigações paralelas** — inicie várias tarefas em background para explorar ângulos diferentes simultaneamente

:::info
Sessões em background não aparecem no histórico principal da sua conversa. São sessões independentes com o próprio ID de tarefa (por exemplo, `bg_143022_a1b2c3`).
:::

## Modo silencioso

Por padrão, o CLI roda em modo silencioso, que:
- Suprime logging verboso das ferramentas
- Habilita feedback animado estilo kawaii
- Mantém a saída limpa e amigável

Para saída de debug:
```bash
hermes chat --verbose
```
