---
sidebar_position: 1
title: "Interface CLI"
description: "Domine a interface de terminal do Hermes Agent — comandos, atalhos de teclado, personalidades e muito mais"
---

# Interface CLI

O CLI do Hermes Agent é uma interface de usuário de terminal completa (TUI) — não uma interface web. Possui edição multi-linha, autocomplete de comandos de barra, histórico de conversas, interrupção e redirecionamento, e saída de ferramentas em streaming. Construído para pessoas que vivem no terminal.

:::tip Configuração inicial
Um comando — `hermes setup --portal` — e você está pronto para `hermes chat`. Veja [Nous Portal](/integrations/nous-portal).
:::

:::tip
O Hermes também oferece um TUI moderno com sobreposições modais, seleção com mouse e entrada não bloqueante. Inicie com `hermes --tui` — veja o guia [TUI](tui.md).
:::

## Executando o CLI

```bash
# Iniciar uma sessão interativa (padrão)
hermes

# Modo de consulta única (não interativo)
hermes chat -q "Olá"

# Com um modelo específico
hermes chat --model "anthropic/claude-sonnet-4"

# Com um provider específico
hermes chat --provider nous        # Usar Nous Portal
hermes chat --provider openrouter  # Forçar OpenRouter

# Com toolsets específicos
hermes chat --toolsets "web,terminal,skills"

# Iniciar com uma ou mais skills pré-carregadas
hermes -s hermes-agent-dev,github-auth
hermes chat -s github-pr-workflow -q "abra um draft PR"

# Retomar sessões anteriores
hermes --continue             # Retomar a sessão CLI mais recente (-c)
hermes --resume <session_id>  # Retomar uma sessão específica pelo ID (-r)

# Modo verboso (saída de debug)
hermes chat --verbose

# Git worktree isolado (para executar múltiplos agentes em paralelo)
hermes -w                         # Modo interativo em worktree
hermes -w -z "Corrigir issue #123" # Consulta única em worktree
```

## Layout da Interface

<img className="docs-terminal-figure" src="/docs/img/docs/cli-layout.svg" alt="Pré-visualização estilizada do layout do Hermes CLI mostrando o banner, área de conversa e prompt de entrada fixo." />
<p className="docs-figure-caption">O banner do Hermes CLI, o fluxo de conversa e o prompt de entrada fixo renderizados como uma figura de documentação estável em vez de arte de texto frágil.</p>

O banner de boas-vindas mostra seu modelo, backend de terminal, diretório de trabalho, ferramentas disponíveis e skills instaladas de relance.

### Barra de Status

Uma barra de status persistente fica acima da área de entrada, atualizando em tempo real:

```
 ⚕ claude-sonnet-4-20250514 │ 12.4K/200K │ [██████░░░░] 6% │ $0.06 │ 15m
```

| Elemento           | Descrição                                                                        |
|--------------------|----------------------------------------------------------------------------------|
| Nome do modelo     | Modelo atual (truncado se mais longo que 26 caracteres)                          |
| Contagem de tokens | Tokens de contexto usados / janela máxima de contexto                            |
| Barra de contexto  | Indicador visual de preenchimento com limites codificados por cores              |
| Custo              | Custo estimado da sessão (ou `n/a` para modelos desconhecidos/gratuitos)         |
| 🗜️ N               | **Contagem de compressão de contexto** — quantas vezes a sessão foi auto-comprimida |
| ▶ N                | **Tarefas em segundo plano ativas** — quantos prompts `/background` ainda estão em execução |
| Duração            | Tempo decorrido da sessão                                                        |
| ⚠ YOLO             | **Aviso de modo YOLO** — mostrado quando `HERMES_YOLO_MODE` está ativo           |

A barra se adapta à largura do terminal — layout completo em ≥ 76 colunas, compacto em 52–75, mínimo (modelo + duração, mais o badge YOLO quando ativo) abaixo de 52.

**Codificação de cores do contexto:**

| Cor    | Limiar               | Significado                         |
|--------|----------------------|-------------------------------------|
| Verde  | < 50%                | Bastante espaço                     |
| Amarelo| 50–80%               | Ficando cheio                       |
| Laranja| 80–95%               | Aproximando-se do limite            |
| Vermelho| ≥ 95%               | Quase transbordando — considere `/compress` |

Use `/usage` para uma discriminação detalhada incluindo custos por categoria (tokens de entrada vs saída).

No provider `openai-codex`, `/usage` também mostra redefinições de limite de uso acumuladas em sua conta ChatGPT ("Você tem N redefinições acumuladas — use /usage reset para ativar"). `/usage reset` resgata uma redefinição acumulada, restaurando totalmente seus limites de 5 horas e semanais.

### Exibição de Retomada de Sessão

Ao retomar uma sessão anterior (`hermes -c` ou `hermes --resume <id>`), um painel "Conversa Anterior" aparece entre o banner e o prompt de entrada, mostrando um resumo compacto do histórico da conversa. Veja [Sessões — Recapitulação de Conversa ao Retomar](sessions.md#conversation-recap-on-resume) para detalhes e configuração.

## Atalhos de Teclado

| Tecla                        | Ação                                                                                             |
|------------------------------|--------------------------------------------------------------------------------------------------|
| `Enter`                      | Enviar mensagem                                                                                  |
| `Alt+Enter`, `Ctrl+J` ou `Shift+Enter` | Nova linha (entrada multi-linha). `Shift+Enter` requer um terminal que o distingue de `Enter`. |
| `Alt+V`                      | Colar uma imagem da área de transferência quando suportado pelo terminal                         |
| `Ctrl+V`                     | Colar texto e anexar oportunisticamente imagens da área de transferência                         |
| `Ctrl+B`                     | Iniciar/parar gravação de voz quando o modo de voz está ativo                                    |
| `Ctrl+G`                     | Abrir o buffer de entrada atual no `$EDITOR` (vim/nvim/nano/VS Code/etc.)                        |
| `Ctrl+X Ctrl+E`              | Atalho alternativo no estilo Emacs para o editor externo                                         |
| `Ctrl+C`                     | Interromper agente (pressione duas vezes em 2s para forçar saída)                                |
| `Ctrl+D`                     | Sair                                                                                             |
| `Ctrl+Z`                     | Suspender Hermes para segundo plano (Unix apenas)                                                |
| `Tab`                        | Aceitar sugestão automática (texto fantasma) ou autocompletar comandos de barra                   |

**Pré-visualização de colagem multi-linha.** Quando você cola um bloco multi-linha, o CLI ecoa uma pré-visualização compacta de linha única (`[colado: 47 linhas, 1.842 caracteres — pressione Enter para enviar]`) em vez de despejar toda a carga no histórico.

**Remoção de markdown em respostas finais.** O CLI remove os fences de markdown mais verbosos e wrappers `**bold**` / `*italic*` das respostas *finais* do agente para que sejam renderizadas como prosa de terminal legível em vez de código-fonte bruto.

## Comandos de Barra (/)

Digite `/` para ver o menu suspenso de autocomplete. O Hermes suporta um grande conjunto de comandos de barra do CLI, comandos de skill dinâmicos e comandos rápidos definidos pelo usuário.

Exemplos comuns:

| Comando                        | Descrição                                                                                |
|--------------------------------|------------------------------------------------------------------------------------------|
| `/help`                        | Mostrar ajuda de comandos                                                                |
| `/model`                       | Mostrar ou alterar o modelo atual                                                        |
| `/tools`                       | Listar ferramentas atualmente disponíveis                                                 |
| `/skills browse`               | Navegar pelo skills hub e skills opcionais oficiais                                      |
| `/background <prompt>`         | Executar um prompt em uma sessão separada em segundo plano                                |
| `/skin`                        | Mostrar ou trocar a skin ativa do CLI                                                     |
| `/voice on`                    | Ativar modo de voz no CLI (pressione `Ctrl+B` para gravar)                                |
| `/voice tts`                   | Alternar reprodução por voz das respostas do Hermes                                       |
| `/reasoning high`              | Aumentar o esforço de raciocínio                                                          |
| `/title Minha Sessão`          | Nomear a sessão atual                                                                     |
| `/status`                      | Mostrar informações da sessão — modelo/profile/tokens/duração                             |
| `/sessions`                    | Abrir um seletor de sessão interativo dentro do CLI clássico                              |

Para a lista completa de comandos de barra do CLI e de mensagens, veja [Referência de Comandos de Barra](../reference/slash-commands.md).

Para configuração, providers, ajuste de silêncio e uso de voz em mensagens/Discord, veja [Modo de Voz](features/voice-mode.md).

:::tip
Comandos não diferenciam maiúsculas/minúsculas — `/HELP` funciona da mesma forma que `/help`. Skills instaladas também se tornam comandos de barra automaticamente.
:::

## Comandos Rápidos

Você pode definir comandos customizados que executam comandos shell instantaneamente sem invocar o LLM. Eles funcionam tanto no CLI quanto em plataformas de mensagens (Telegram, Discord, etc.).

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

Depois digite `/status`, `/gpu` ou `/restart` em qualquer chat. Veja o [Guia de Configuração](/user-guide/configuration#quick-commands) para mais exemplos.

## Pré-carregando Skills na Inicialização

Se você já sabe quais skills deseja ativas para a sessão, passe-as no momento da inicialização:

```bash
hermes -s hermes-agent-dev,github-auth
hermes chat -s github-pr-workflow -s github-auth
```

O Hermes carrega cada skill nomeada no prompt da sessão antes do primeiro turno. A mesma flag funciona em modo interativo e modo de consulta única.

## Comandos de Barra de Skills

Toda skill instalada em `~/.hermes/skills/` é automaticamente registrada como um comando de barra. O nome da skill se torna o comando:

```
/gif-search gatos engraçados
/axolotl me ajude a ajustar Llama 3 no meu dataset
/github-pr-workflow crie um PR para a refatoração de auth

# Apenas o nome da skill carrega e deixa o agente perguntar o que você precisa:
/excalidraw
```

## Personalidades

Defina uma personalidade predefinida para alterar o tom do agente:

```
/personality pirate
/personality kawaii
/personality concise
```

Personalidades inclusas incluem: `helpful`, `concise`, `technical`, `creative`, `teacher`, `kawaii`, `catgirl`, `pirate`, `shakespeare`, `surfer`, `noir`, `uwu`, `philosopher`, `hype`.

Você também pode definir personalidades customizadas em `~/.hermes/config.yaml`:

```yaml
personalities:
  helpful: "Você é um assistente de IA útil e amigável."
  kawaii: "Você é um assistente kawaii! Use expressões fofas..."
  pirate: "Arre! Estás falando com o Capitão Hermes..."
  # Adicione a sua própria!
```

## Entrada Multi-linha

Existem duas maneiras de inserir mensagens multi-linha:

1. **`Alt+Enter`, `Ctrl+J` ou `Shift+Enter`** — insere uma nova linha
2. **Continuação com barra invertida** — termine uma linha com `\` para continuar:

```
❯ Escreva uma função que:\
  1. Recebe uma lista de números\
  2. Retorna a soma
```

:::info
Colar texto multi-linha é suportado — use qualquer uma das teclas de nova linha acima, ou simplesmente cole o conteúdo diretamente.
:::

### Compatibilidade Shift+Enter

A maioria dos terminais envia a mesma sequência de bytes para `Enter` e `Shift+Enter` por padrão, então os aplicativos não conseguem distingui-los. O Hermes reconhece `Shift+Enter` apenas quando o terminal envia uma sequência distinta através do [protocolo de teclado Kitty](https://sw.kovidgoyal.net/kitty/keyboard-protocol/) ou modo `modifyOtherKeys` do xterm.

| Terminal                   | Status                                                               |
|----------------------------|----------------------------------------------------------------------|
| Kitty, foot, WezTerm, Ghostty | `Shift+Enter` distinto ativado por padrão                           |
| iTerm2 (recente), Alacritty, terminal VS Code, Warp | Suportado após ativar o protocolo Kitty nas configurações |
| Windows Terminal Preview 1.25+ | Suportado após ativar o protocolo Kitty nas configurações          |
| macOS Terminal.app, Windows Terminal (estável) | Não suportado — `Shift+Enter` é indistinguível de `Enter` |

Onde o terminal não consegue distingui-los, `Alt+Enter` e `Ctrl+J` continuam funcionando em todos os lugares. **No Windows Terminal especificamente, `Alt+Enter` é capturado pelo terminal (alterna tela cheia) e nunca chega ao Hermes — use `Ctrl+Enter` (entregue como `Ctrl+J`) ou `Ctrl+J` diretamente para uma nova linha.**

## Interrompendo o Agente

Você pode interromper o agente a qualquer momento:

- **Digite uma nova mensagem + Enter** enquanto o agente está trabalhando — ele interrompe e processa suas novas instruções
- **`Ctrl+C`** — interrompe a operação atual (pressione duas vezes em 2s para forçar saída)
- Comandos de terminal em andamento são mortos imediatamente (SIGTERM, depois SIGKILL após 1s)
- Múltiplas mensagens digitadas durante a interrupção são combinadas em um único prompt

### Modo de Entrada Ocupado

A chave de configuração `display.busy_input_mode` controla o que acontece quando você pressiona Enter enquanto o agente está trabalhando:

| Modo         | Comportamento                                                                                     |
|--------------|---------------------------------------------------------------------------------------------------|
| `"interrupt"` (padrão) | Sua mensagem interrompe a operação atual e é processada imediatamente                              |
| `"queue"`    | Sua mensagem é silenciosamente enfileirada e enviada como o próximo turno após o agente terminar   |
| `"steer"`    | Sua mensagem é injetada na execução atual via `/steer`, chegando ao agente após a próxima chamada de ferramenta |

```yaml
# ~/.hermes/config.yaml
display:
  busy_input_mode: "steer"   # ou "queue" ou "interrupt" (padrão)
```

### Suspensão para Segundo Plano

Em sistemas Unix, pressione **`Ctrl+Z`** para suspender o Hermes para segundo plano — como qualquer processo de terminal. O shell imprime uma confirmação:

```
Hermes Agent foi suspenso. Execute `fg` para trazer o Hermes Agent de volta.
```

Digite `fg` em seu shell para retomar a sessão exatamente de onde parou.

## Exibição de Progresso de Ferramentas

O CLI mostra feedback animado enquanto o agente trabalha:

**Animação de pensamento** (durante chamadas de API):
```
  ◜ (｡•́︿•̀｡) ponderando... (1.2s)
  ◠ (⊙_⊙) contemplando... (2.4s)
  ✧٩(ˊᗜˋ*)و✧ consegui! (3.1s)
```

**Feed de execução de ferramentas:**
```
  ┊ 💻 terminal `ls -la` (0.3s)
  ┊ 🔍 web_search (1.2s)
  ┊ 📄 web_extract (2.1s)
```

Alterne entre modos de exibição com `/verbose`: `off → new → all → verbose`.

### Tamanho da Pré-visualização de Ferramentas

A chave de configuração `display.tool_preview_length` controla o número máximo de caracteres mostrados nas linhas de pré-visualização de chamadas de ferramenta (ex.: caminhos de arquivo, comandos de terminal). O padrão é `0`, que significa sem limite — caminhos e comandos completos são mostrados.

```yaml
# ~/.hermes/config.yaml
display:
  tool_preview_length: 80   # Truncar pré-visualizações de ferramentas para 80 caracteres (0 = sem limite)
```

## Gerenciamento de Sessão

### Retomando Sessões

Quando você sai de uma sessão CLI, um comando de retomada é impresso:

```
Retome esta sessão com:
  hermes --resume 20260225_143052_a1b2c3

Sessão:        20260225_143052_a1b2c3
Duração:       12m 34s
Mensagens:     28 (5 do usuário, 18 chamadas de ferramentas)
```

Opções de retomada:

```bash
hermes --continue                          # Retomar a sessão CLI mais recente
hermes -c                                  # Forma abreviada
hermes -c "meu projeto"                    # Retomar uma sessão nomeada (mais recente na linhagem)
hermes --resume 20260225_143052_a1b2c3     # Retomar uma sessão específica pelo ID
hermes --resume "refatorando auth"         # Retomar pelo título
hermes -r 20260225_143052_a1b2c3           # Forma abreviada
```

Retomar restaura o histórico completo da conversa do SQLite. O agente vê todas as mensagens anteriores, chamadas de ferramentas e respostas — como se você nunca tivesse saído.

Use `/title Nome da Minha Sessão` dentro de um chat para nomear a sessão atual, ou `hermes sessions rename <id> <título>` pela linha de comando. Use `hermes sessions list` para navegar pelas sessões passadas.

### Armazenamento de Sessão

Sessões CLI são armazenadas no banco de dados SQLite de estado do Hermes sob `~/.hermes/state.db`. O banco mantém:

- metadados da sessão (ID, título, timestamps, contadores de token)
- histórico de mensagens
- linhagem entre sessões comprimidas/retomadas
- índices de busca de texto completo usados por `session_search`

### Compressão de Contexto

Conversas longas são automaticamente resumidas quando se aproximam dos limites de contexto:

```yaml
# Em ~/.hermes/config.yaml
compression:
  enabled: true
  threshold: 0.50    # Comprimir em 50% do limite de contexto por padrão

# Modelo de sumarização configurado em auxiliary:
auxiliary:
  compression:
    model: ""  # Deixar vazio para usar o modelo de chat principal (padrão).
```

## Sessões em Segundo Plano

Execute um prompt em uma sessão separada em segundo plano enquanto continua usando o CLI para outro trabalho:

```
/background Analise os logs em /var/log e resuma quaisquer erros de hoje
```

O Hermes confirma imediatamente a tarefa e devolve o prompt:

```
🔄 Tarefa em segundo plano #1 iniciada: "Analise os logs em /var/log e resuma..."
   Task ID: bg_143022_a1b2c3
```

### Como Funciona

Cada prompt `/background` cria uma **sessão de agente completamente separada** em uma thread daemon:

- **Conversa isolada** — o agente em segundo plano não tem conhecimento do histórico da sua sessão atual. Ele recebe apenas o prompt que você forneceu.
- **Mesma configuração** — o agente em segundo plano herda seu modelo, provider, toolsets, configurações de raciocínio e modelo de fallback da sessão atual.
- **Não bloqueante** — sua sessão em primeiro plano permanece totalmente interativa.
- **Múltiplas tarefas** — você pode executar várias tarefas em segundo plano simultaneamente.

### Resultados

Quando uma tarefa em segundo plano termina, o resultado aparece como um painel em seu terminal:

```
╭─ ⚕ Hermes (segundo plano #1) ──────────────────────────────────╮
│ Encontrados 3 erros no syslog de hoje:                          │
│ 1. OOM killer invocado às 03:22 — processo nginx morto          │
│ 2. Erro de I/O de disco em /dev/sda1 às 07:15                  │
│ 3. Tentativas de login SSH falhas de 192.168.1.50 às 14:30     │
╰──────────────────────────────────────────────────────────────────╯
```

Se a tarefa falhar, você verá uma notificação de erro.

### Casos de Uso

- **Pesquisa longa** — "/background pesquise os últimos desenvolvimentos em correção quântica de erros" enquanto você trabalha em código
- **Processamento de arquivos** — "/background analise todos os arquivos Python neste repositório e liste quaisquer problemas de segurança"
- **Investigações paralelas** — inicie múltiplas tarefas em segundo plano para explorar diferentes ângulos simultaneamente

:::info
Sessões em segundo plano não aparecem em seu histórico principal de conversas. São sessões independentes com seu próprio ID de tarefa.
:::

## Modo Silencioso

Por padrão, o CLI executa em modo silencioso que:
- Suprime logs verbosos de ferramentas
- Ativa feedback animado no estilo kawaii
- Mantém a saída limpa e amigável

Para saída de debug:
```bash
hermes chat --verbose
```
