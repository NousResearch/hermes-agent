---
sidebar_position: 2
title: "TUI"
description: "Inicie a interface de terminal moderna do Hermes — amigável ao mouse, overlays ricos e entrada não bloqueante."
---

# TUI

A TUI é a interface moderna do Hermes — uma UI de terminal apoiada no mesmo runtime Python da [CLI clássica](cli.md). Mesmo agente, mesmas sessões, mesmos slash commands; uma superfície mais limpa e responsiva para interagir com eles.

É a forma recomendada de executar o Hermes de forma interativa.

## Launch {#launch}

```bash
# Iniciar a TUI
hermes --tui

# Retomar a última sessão TUI (fallback para a última sessão clássica)
hermes --tui -c
hermes --tui --continue

# Retomar uma sessão específica por ID ou título
hermes --tui -r 20260409_000000_aa11bb
hermes --tui --resume "my t0p session"

# Executar o source diretamente — pula o passo de prebuild (para contribuidores da TUI)
hermes --tui --dev
```

Você também pode habilitá-la via variável de ambiente:

```bash
export HERMES_TUI=1
hermes          # agora usa a TUI
hermes chat     # idem
```

Ou defini-la como padrão persistente em `~/.hermes/config.yaml`:

```yaml
display:
  interface: tui   # "cli" (padrão) ou "tui"
```

Com `display.interface: tui`, um `hermes` simples (e `hermes chat`) inicia a TUI. Flags explícitas sempre prevalecem — execute `hermes --cli` para voltar ao REPL clássico em uma única invocação, ou `hermes --tui` / `HERMES_TUI=1` para forçar a TUI quando o padrão em config for `cli`.

A CLI clássica continua sendo o padrão enviado. Tudo documentado em [Interface CLI](cli.md) — slash commands, quick commands, preload de skills, personalidades, entrada multilinha, interrupções — funciona na TUI de forma idêntica.

## Why the TUI {#why-the-tui}

- **Primeiro frame instantâneo** — o banner é renderizado antes do app terminar de carregar, então o terminal nunca parece congelado enquanto o Hermes inicia.
- **Entrada não bloqueante** — digite e enfileire mensagens antes da sessão estar pronta. Seu primeiro prompt é enviado no momento em que o agente ficar online.
- **Overlays ricos** — seletor de modelo, seletor de sessão, prompts de aprovação e esclarecimento renderizam como painéis modais em vez de fluxos inline.
- **Painel de sessão ao vivo** — tools e skills são preenchidos progressivamente conforme inicializam.
- **Seleção amigável ao mouse** — arraste para destacar com fundo uniforme em vez de inversão SGR. Copie com o gesto normal de copiar do seu terminal.
- **Renderização em tela alternativa** — atualizações diferenciais significam sem flicker durante streaming, sem poluição de scrollback ao sair.
- **Recursos do compositor** — colapso inline de paste para trechos longos, paste de texto com `Cmd+V` / `Ctrl+V` com fallback para imagem da área de transferência, segurança de bracketed-paste e normalização de anexos de imagem/caminho de arquivo.

As mesmas [skins](features/skins.md) e [personalidades](features/personality.md) se aplicam. Alterne no meio da sessão com `/skin ares`, `/personality pirate`, e a UI repinta ao vivo. Veja [Skins & Temas](features/skins.md) para a lista completa de chaves personalizáveis e quais se aplicam à clássica vs TUI — a TUI respeita a paleta do banner, cores da UI, glifo/cor do prompt, exibição de sessão, menu de completion, fundo de seleção, `tool_prefix` e `help_header`.

### Collapsible banner sections {#collapsible-banner-sections}

O banner de inicialização da TUI agrupa informações de runtime em quatro seções recolhíveis, cada uma renderizada com um chevron `▸` / `▾` ao lado do título da seção:

| Seção | Estado padrão |
|---------|---------------|
| Tools | Aberta |
| Skills | Recolhida |
| System Prompt | Recolhida |
| MCP Servers | Recolhida |

Clique em qualquer lugar no cabeçalho de uma seção (ou no chevron) para alternar. A lista de Tools abre por padrão porque é a seção mais consultada no início da sessão; Skills, System Prompt e MCP Servers ficam recolhidas por padrão para o banner permanecer compacto mesmo quando você instalou dezenas de skills ou conectou muitos servidores MCP. O estado é local à instância do banner, então o próximo launch reseta para os padrões.

## Requirements {#requirements}

- **Node.js** ≥ 20 — a TUI roda como subprocesso lançado pela CLI Python. `hermes doctor` verifica isso.
- **TTY** — como a CLI clássica, redirecionar stdin ou executar em ambientes não interativos faz fallback para modo single-query.

No primeiro launch o Hermes instala as dependências Node da TUI em `ui-tui/node_modules` (única vez, alguns segundos). Lançamentos subsequentes são rápidos. Se você puxar uma nova versão do Hermes, o bundle da TUI é reconstruído automaticamente quando os sources forem mais recentes que o dist.

:::tip Trabalhando em vários git worktrees?
Contribuidores que executam `hermes --tui --dev` de muitos worktrees podem compartilhar um `node_modules` em vez de instalar por checkout — veja [TUI & Desktop a partir de Worktrees](../developer-guide/worktree-ui-dev.md).
:::

### External prebuild {#external-prebuild}

Distribuições que enviam um bundle pré-construído (Nix, pacotes de sistema) podem apontar o Hermes para ele:

```bash
export HERMES_TUI_DIR=/path/to/prebuilt/ui-tui
hermes --tui
```

O diretório deve conter `dist/entry.js`.

## Keybindings {#keybindings}

Os atalhos correspondem exatamente aos da [CLI clássica](cli.md#keybindings). As únicas diferenças de comportamento:

- **Arrastar com o mouse** destaca texto com fundo de seleção uniforme.
- **`Cmd+V` / `Ctrl+V`** primeiro tenta paste de texto normal, depois faz fallback para leituras OSC52/clipboard nativas e, por fim, anexo de imagem quando a área de transferência ou o payload colado resolve para uma imagem.
- **`/terminal-setup`** instala bindings locais de terminal para VS Code / Cursor / Windsurf para melhor paridade de `Cmd+Enter` e undo/redo no macOS.
- **Autocompletion de slash** abre como painel flutuante com descrições, não um dropdown inline.
- **`Ctrl+X`** abre o seletor de sessões ao vivo. Quando uma mensagem enfileirada está destacada (enviada enquanto o agente ainda estava executando), ainda exclui essa mensagem enfileirada. **`Esc`** cancela a edição e remove o destaque sem excluir.
- **`Ctrl+G` / `Ctrl+X Ctrl+E`** — abre o buffer de entrada atual em `$EDITOR` para composição multilinha / prompt longo; save-and-exit envia o conteúdo de volta como prompt.

## Slash commands {#slash-commands}

Todos os slash commands funcionam inalterados. Alguns são de propriedade da TUI — produzem saída mais rica ou renderizam como overlays em vez de painéis inline:

| Comando | Comportamento na TUI |
|---------|--------------|
| `/help` | Overlay com comandos categorizados, navegável com setas |
| `/sessions` (alias `/switch`) | Seletor de sessões ao vivo — lista sessões TUI abertas, alterna entre elas, fecha ou inicia outra |
| `/model` | Seletor modal de modelo agrupado por provider, com dicas de custo |
| `/skin` | Preview ao vivo — mudança de tema aplica enquanto você navega |
| `/details` | Alterna detalhes verbosos de tool call (global ou por seção) |
| `/usage` | Painel rico de token / custo / contexto |
| `/agents` (alias `/tasks`) | Overlay de observabilidade — árvore de subagentes ao vivo com controles kill/pause, rollups de custo / token / arquivo por branch, histórico turn-by-turn |
| `/reload` | Relê `~/.hermes/.env` no processo TUI em execução para novas API keys terem efeito sem reiniciar |
| `/mouse [on\|off\|toggle\|wheel\|buttons\|all]` | Escolhe um preset de mouse tracking em runtime (também persiste em `display.mouse_tracking` em `config.yaml`). `wheel` (1000+1006) mantém scroll com roda sem os eventos de hover que fazem o tmux spammar "No image in clipboard" sobre a linha do prompt; `buttons` adiciona drag-to-select; `all` é o padrão com UI orientada a hover. |

Todo outro slash command (incluindo skills instaladas, quick commands e toggles de personalidade) funciona de forma idêntica à CLI clássica. Veja [Referência de Slash Commands](../reference/slash-commands.md).

## Live session switcher {#live-session-switcher}

Use o seletor de sessões ao vivo quando quiser que um terminal atue como dispatcher para várias sessões TUI. Ele lista apenas sessões atualmente vivas neste processo TUI; sessões fechadas permanecem transcripts salvos e ainda podem ser reabertas com `/resume` ou `hermes --tui --resume <id-or-title>`.

Abra com qualquer um destes:

- `Ctrl+X` a partir da TUI.
- `/sessions` ou `/switch`.
- `/sessions new` para criar uma sessão ao vivo nova imediatamente.
- Clique no contador `N live sessions` na linha de status.

<img alt="Hermes TUI Session Orchestrator with one live session and a +new row" src="/docs/img/docs/tui-session-orchestrator/session-orchestrator.png" />

<video controls muted loop playsInline src="/docs/img/docs/tui-session-orchestrator/session-orchestrator-demo.mp4" title="Hermes TUI Session Orchestrator demo" style={{maxWidth: '100%'}}></video>

Dentro do seletor:

- `↑` / `↓` movem a seleção; cliques do mouse também selecionam linhas.
- `Enter` alterna para a sessão ao vivo selecionada.
- `Ctrl+D` fecha a sessão ao vivo selecionada.
- `Ctrl+N` inicia uma sessão ao vivo em branco.
- `Ctrl+R` atualiza a lista de sessões ao vivo.
- `Esc` fecha o seletor.
- Selecione `+new`, digite um prompt e pressione `Enter` para despachar uma nova sessão ao vivo. Pressione `Tab` primeiro se quiser escolher um modelo só para essa nova sessão.

## LaTeX math rendering {#latex-math-rendering}

O pipeline markdown da TUI renderiza matemática LaTeX inline: `$E = mc^2$` e `$$\frac{a}{b}$$` renderizam como matemática formatada em Unicode em vez do source TeX bruto. Funciona para matemática inline e em bloco; sintaxe não suportada faz fallback para mostrar o TeX literal envolvido em um code span para permanecer copiável.

Isso está sempre ativo — nada para configurar. A CLI clássica mantém o TeX bruto.

## Light-terminal detection {#light-terminal-detection}

A TUI detecta automaticamente terminais claros e troca para o tema claro de acordo. A detecção funciona em três camadas:

1. Variável de ambiente `HERMES_TUI_THEME` — maior prioridade. Valores: `light`, `dark`, ou um hex de background de 6 caracteres (ex.: `ffffff`, `1a1a2e`).
2. Variável de ambiente `COLORFGBG` — a dica clássica "qual é a cor do meu background?" usada por terminais derivados de xterm.
3. Sonda de background do terminal via OSC 11 — funciona em terminais modernos (Ghostty, Warp, iTerm2, WezTerm, Kitty) que não definem `COLORFGBG`.

Se quiser o tema claro permanentemente independente do terminal:

```bash
export HERMES_TUI_THEME=light
```

## Busy indicator styles {#busy-indicator-styles}

O indicador busy da barra de status é plugável — o padrão rotaciona a paleta kawaii de faces do Hermes a cada 2,5 segundos durante o trabalho do agente. Escolha um estilo diferente via config ou o slash command `/indicator`:

```yaml
display:
  tui_status_indicator: kaomoji   # kaomoji | emoji | unicode | ascii
```

Ou na sessão: `/indicator emoji` (etc.). Os estilos vêm com larguras de glifo correspondentes para o resto da barra de status não tremer na rotação.

## Auto-resume {#auto-resume}

Por padrão, `hermes --tui` inicia uma sessão nova a cada launch. Para reanexar automaticamente à sessão TUI mais recente (útil quando seu terminal ou conexão SSH cai inesperadamente), opte por:

```bash
export HERMES_TUI_RESUME=1          # sessão TUI mais recente
# ou:
export HERMES_TUI_RESUME=<session-id>   # sessão específica
```

Desconfigure a variável ou passe `--resume <id>` explicitamente para sobrescrever por launch.

## Status line {#status-line}

A linha de status da TUI acompanha o estado do agente em tempo real:

| Status | Significado |
|--------|---------|
| `starting agent…` | Session ID está ativo; tools e skills ainda entrando online. Você pode digitar — mensagens enfileiram e enviam quando prontas. |
| `ready` | Agente ocioso, aceitando entrada. |
| `thinking…` / `running…` | Agente está raciocinando ou executando uma tool. |
| `interrupted` | Turn atual foi cancelado; pressione Enter para enviar novamente. |
| `forging session…` / `resuming…` | Handshake de conexão inicial ou `--resume`. |

As cores e thresholds da barra de status por skin são compartilhados com a CLI clássica — veja [Skins](features/skins.md) para personalização.

A linha de status também mostra:

- **Diretório de trabalho com branch git** — `~/projects/hermes-agent (docs/two-week-gap-sweep)`. O sufixo de branch atualiza quando você faz `git checkout` em um terminal lateral (cache por mtime) para a TUI refletir sua branch ativa real, não a que era no launch.
- **Tempo decorrido por prompt** — `⏱ 12s/3m 45s` enquanto o turn está executando (ao vivo), congelado em `⏲ 32s / 3m 45s` após o turn completar. O primeiro número é o tempo desde a última mensagem do usuário; o segundo é a duração total da sessão. Reseta a cada novo prompt.
- **`🗜️ N`** — número de vezes que a sessão em execução foi auto-comprimida. Aparece após a primeira compressão.
- **`▶ N`** — número de tarefas `/background` atualmente em execução nesta sessão. Aparece sempre que pelo menos uma tarefa está em andamento.
- **`⚠ YOLO`** — aviso visível sempre que o modo YOLO está ativo (`hermes --yolo`, `/yolo`, ou `HERMES_YOLO_MODE=1`). O mesmo badge também aparece no banner de inicialização para você não lançar uma sessão auto-aprovadora sem perceber.

## Configuration {#configuration}

A TUI respeita toda a config padrão do Hermes: `~/.hermes/config.yaml`, profiles, personalidades, skins, quick commands, credential pools, memory providers, habilitação de tool/skill. Não existe arquivo de config específico da TUI.

Algumas chaves ajustam especificamente a superfície da TUI:

```yaml
display:
  skin: default              # qualquer skin built-in ou customizada
  personality: helpful
  details_mode: collapsed    # hidden | collapsed | expanded — padrão global do accordion
  sections:                  # opcional: overrides por seção (qualquer subconjunto)
    thinking: expanded       # sempre aberto
    tools: expanded          # sempre aberto
    activity: collapsed      # optar de volta IN ao painel activity (oculto por padrão)
  mouse_tracking: all        # off | wheel | buttons | all (ou true/false para back-compat).
                             #   wheel   — 1000+1006 (scroll + click; sem drag, sem hover —
                             #             recomendado dentro do tmux para silenciar o spam
                             #             "No image in clipboard" na linha do prompt por eventos hover)
                             #   buttons — adiciona 1002 para seleção por drag no terminal
                             #   all     — adiciona 1003 para hover (paginação scrollbar-on-hover,
                             #             link mouseenter, etc.)
```

Toggles em runtime:

- `/details [hidden|collapsed|expanded|cycle]` — define o modo global
- `/details <section> [hidden|collapsed|expanded|reset]` — sobrescreve uma seção
  (seções: `thinking`, `tools`, `subagents`, `activity`)

**Visibilidade padrão**

A TUI vem com defaults opinativos por seção que transmitem o turn como transcript ao vivo em vez de uma parede de chevrons:

- `thinking` — **expanded**. Raciocínio transmite inline conforme o modelo emite.
- `tools` — **expanded**. Tool calls e seus resultados renderizam abertos.
- `subagents` — segue o `details_mode` global (collapsed sob chevron por padrão — fica quieto até uma delegação acontecer de fato).
- `activity` — **hidden**. Meta ambiente (dicas de gateway, nudges de paridade de terminal, notificações de background) é ruído para a maioria do uso diário. Falhas de tool ainda renderizam inline na linha da tool que falhou; erros/avisos ambientes aparecem via backstop de alerta flutuante quando todo painel está oculto.

Overrides por seção têm precedência sobre o default da seção e o `details_mode` global. Para remodelar o layout:

- `display.sections.thinking: collapsed` — colocar thinking de volta sob um chevron
- `display.sections.tools: collapsed` — colocar tool calls de volta sob um chevron
- `display.sections.activity: collapsed` — optar o painel activity de volta
- `/details <section> <mode>` em runtime

Qualquer coisa definida explicitamente em `display.sections` prevalece sobre os defaults, então configs existentes continuam funcionando inalteradas.

## Sessions {#sessions}

Sessões são compartilhadas entre a TUI e a CLI clássica — ambas escrevem no mesmo `~/.hermes/state.db`. Você pode iniciar uma sessão em uma e retomar na outra. O seletor de sessões mostra sessões de ambas as fontes, com uma tag de source.

Veja [Sessões](sessions.md) para ciclo de vida, busca, compressão e export.

## How the TUI talks to its gateway {#how-the-tui-talks-to-its-gateway}

Por padrão a TUI gera seu próprio gateway in-process, então cada instância TUI é autocontida — não há nada para configurar.

Você pode ver uma variável de ambiente `HERMES_TUI_GATEWAY_URL` referenciada no codebase ou logs. Este é um **detalhe de wiring interno do web dashboard**, não um knob de remote-attach voltado ao usuário. Quando você abre a aba "Chat" do dashboard (`hermes dashboard` → `/chat`), o web server do dashboard gera um processo filho TUI embutido e injeta `HERMES_TUI_GATEWAY_URL` para esse filho anexar ao `tui_gateway` in-process do dashboard via WebSocket loopback (`/api/ws`). O endpoint `/api/ws` existe apenas dentro do server do dashboard (`hermes_cli/web_server.py`) e está vinculado ao lifetime e auth desse processo.

Não existe um modo geral "apontar qualquer TUI para qualquer porta de gateway standalone". Em particular, o API server compatível com OpenAI (`hermes gateway` / a plataforma `api_server`) **não** serve `/api/ws` — é a superfície model-backend (`/v1/chat/completions`, `/v1/models`, …) e deliberadamente não expõe o canal de controle JSON-RPC da TUI. Definir `HERMES_TUI_GATEWAY_URL` para essa porta resultará em 404.

Se quiser múltiplas superfícies compartilhando um conjunto de sessões, use o `~/.hermes/state.db` compartilhado (veja [Sessões](sessions.md)) ou o chat embutido do web dashboard (veja [Web Dashboard](features/web-dashboard.md#chat)) — não uma URL de gateway definida manualmente.

## Reverting to the classic CLI {#reverting-to-the-classic-cli}

Executar `hermes` (sem `--tui`) permanece na CLI clássica por padrão. Para uma máquina preferir a TUI, defina `display.interface: tui` em `~/.hermes/config.yaml` (persistente) ou `HERMES_TUI=1` no profile do shell (por shell). Para voltar, defina `interface: cli` / desconfigure a env var, ou passe `hermes --cli` para uma única vez.

Se a TUI falhar ao iniciar (sem Node, bundle ausente, problema de TTY), o Hermes imprime um diagnóstico e faz fallback — em vez de deixá-lo preso.

## See also {#see-also}

- [Interface CLI](cli.md) — referência completa de slash commands e keybindings (compartilhada)
- [Sessões](sessions.md) — retomar, branch e histórico
- [Skins & Temas](features/skins.md) — tematizar banner, barra de status e overlays
- [Modo Voz](features/voice-mode.md) — funciona em ambas as interfaces
- [Configuração](configuration.md) — todas as chaves de config
