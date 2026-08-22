---
sidebar_position: 2
title: "Referência de slash commands"
description: "Referência completa de slash commands do CLI interativo e de mensagens"
---

# Referência de slash commands

O Hermes tem duas superfícies de slash commands, ambas driven por um `COMMAND_REGISTRY` central em `hermes_cli/commands.py`:

- **Slash commands do CLI interativo** — despachados por `cli.py`, com autocomplete do registry
- **Slash commands de mensagens** — despachados por `gateway/run.py`, com texto de help e menus de plataforma gerados do registry

Skills instaladas também são expostas como slash commands dinâmicos em ambas as superfícies. Isso inclui skills bundled como `/plan`, que abre o modo plan e salva planos markdown em `.hermes/plans/` relativo ao workspace/backend working directory ativo.

## Permissões e split admin/usuário {#permissions-and-adminuser-split}

Toda plataforma de mensagens que suporta allowlist por usuário (Telegram, Discord, Slack, Matrix, Mattermost, Signal, …) também suporta split de slash commands em dois níveis: **admins** recebem todo comando registrado, **usuários regulares** só recebem os nomes que você listar em `user_allowed_commands` (mais o piso sempre permitido `/help` e `/whoami`). Configure `allow_admin_from` e `user_allowed_commands` (e os equivalentes por grupo `group_allow_admin_from` / `group_user_allowed_commands`) dentro do bloco `extra:` da plataforma em `~/.hermes/gateway-config.yaml`.

Veja a documentação por plataforma para exemplos — a estrutura é idêntica entre plataformas:

- [Telegram](../user-guide/messaging/telegram.md#slash-command-access-control)
- [Discord](../user-guide/messaging/discord.md)
- [Slack](../user-guide/messaging/slack.md)
- [Matrix](../user-guide/messaging/matrix.md)
- [Mattermost](../user-guide/messaging/mattermost.md)
- [Signal](../user-guide/messaging/signal.md)

Se `allow_admin_from` estiver unset para um escopo, aquele escopo permanece em modo backward-compat irrestrito — todo usuário permitido pode rodar todo comando.

## Slash commands do CLI interativo {#interactive-cli-slash-commands}

Digite `/` no CLI para abrir o menu de autocomplete. Comandos built-in são case-insensitive.

### Sessão {#session}

| Comando | Descrição |
|---------|-------------|
| `/new [name]` (alias: `/reset`) | Inicia uma nova sessão (session ID + histórico fresh). `[name]` opcional define o título inicial da sessão — ex.: `/new my-experiment` abre sessão fresh já titulada `my-experiment` para achar depois com `/resume` ou `/sessions`. Acrescente `now`, `--yes` ou `-y` para pular o modal de confirmação — ex.: `/reset now`, `/new --yes my-experiment`. |
| `/clear` | Limpa a tela e inicia nova sessão |
| `/history` | Mostra histórico da conversa (respeita `/timestamps`) |
| `/save` | Salva a conversa atual |
| `/prompt` (alias: `/compose`) | Componha seu próximo prompt em `$EDITOR` (markdown) em vez do input inline — útil para prompts longos, multi-linha ou cuidadosamente formatados. |
| `/retry` | Repete a última mensagem (reenvia ao agente) |
| `/undo` | Remove a última troca user/assistant |
| `/title` | Define um título para a sessão atual (uso: /title My Session Name) |
| `/compress [here [N] \| focus topic]` | Comprime contexto da conversa manualmente (flush de memórias + resumo). `/compress here [N]` resume tudo exceto as N trocas mais recentes (padrão 2), mantidas verbatim — escolha seu próprio limite de compressão. Um focus topic estreita o que um resumo completo preserva. |
| `/rollback` | Lista ou restaura checkpoints de filesystem (uso: /rollback [number]) |
| `/diff [staged\|all\|session] [--stat] [path...]` | Mostra mudanças git no diretório de trabalho. Padrão: mudanças unstaged mais arquivos untracked. `staged` mostra o que está staged para commit, `all` tudo desde HEAD, e `session` o diff cumulativo de tudo que o Hermes mudou aqui (da baseline de checkpoint retida mais antiga — requer checkpoints habilitados; complementa `/rollback diff <N>`). `--stat` imprime só o resumo de arquivos alterados; argumentos de path restringem o diff. |
| `/snapshot [create\|restore <id>\|prune]` (alias: `/snap`) | Cria ou restaura snapshots de estado do Hermes config/state. `create [label]` salva snapshot, `restore <id>` reverte para ele, `prune [N]` remove snapshots antigos, ou liste todos sem args. |
| `/stop` | Mata todos os processos em background em execução |
| `/queue <prompt>` (alias: `/q`) | Enfileira prompt para o próximo turno (não interrompe a resposta atual do agente). |
| `/steer <prompt>` | Injeta nota mid-run que chega ao agente **após a próxima chamada de ferramenta** — sem interrupt, sem novo turno user. O texto é anexado ao conteúdo do último tool result quando a ferramenta atual termina, dando novo contexto ao agente sem quebrar o loop de tool-calling atual. Use para nudgar direção mid-task (ex.: "focus on the auth module" enquanto o agente roda testes). |
| `/goal <text>` | Define meta contínua em direção à qual o Hermes trabalha entre turnos — nossa versão do Ralph loop. Após cada turno um model judge auxiliar decide se a meta está pronta; se não, o Hermes auto-continua. Subcomandos: `/goal status`, `/goal pause`, `/goal resume`, `/goal clear`. Orçamento padrão 20 turnos (`goals.max_turns`); qualquer mensagem real do usuário preempta o loop de continuação, e o estado sobrevive a `/resume`. Veja [Metas persistentes](/user-guide/features/goals) para o walkthrough completo. |
| `/subgoal <text>` | Anexa critério fornecido pelo usuário à meta ativa mid-loop. O prompt de continuação expõe todos os subgoals ao agente verbatim, e o judge os inclui no veredicto DONE/CONTINUE — a meta não é marcada done até a meta original **e** todo subgoal serem atendidos. Subcomandos: `/subgoal` (listar), `/subgoal remove <N>`, `/subgoal clear`. Requer `/goal` ativo. |
| `/heartbeat every <interval> <prompt>` (alias: `/hb`) | Define prompt recorrente que reentra **nesta sessão** como turno user normal sempre que idle e o intervalo passou (mín 60s; ticks perdidos coalescem). Subcomandos: `/heartbeat status`, `/heartbeat pause`, `/heartbeat resume`, `/heartbeat clear`. Escopo de sessão e in-process — use `hermes cron` para schedules duráveis isolados. Veja [Session Heartbeats](/user-guide/features/heartbeat). |
| `/refine [focus]` | Roda a revisão de auto-melhoria de memória/skill em background **agora** em vez de esperar o trigger pós-turno automático. Texto de focus opcional direciona a revisão (ex.: `/refine save the deploy workflow as a skill`). Roda em fork em background contra snapshot da conversa — sessão live e prompt cache intactos; resultados reportados quando terminar. |
| `/moa <prompt>` | Roda um prompt pelo preset padrão de [Mixture of Agents](/user-guide/features/mixture-of-agents), depois restaura seu model atual. One-shot — não muda o model da sessão. |
| `/resume [name]` | Retoma sessão nomeada anteriormente |
| `/sessions` (alias TUI: `/switch`) | CLI clássico: navegue e retome sessões anteriores em picker interativo. TUI: abre o session switcher live para sessões TUI abertas. Use `/sessions new` no TUI para iniciar outra sessão live imediatamente. |
| `/egress [status]` | Mostra status do proxy de egress Docker — estado enabled/configured/running, fonte de credencial, mapeamentos de token, providers descobertos e próximo passo de remediação. Funciona no CLI, TUI, chat Desktop e gateway de mensagens. |
| `/redraw` | Força repaint completo da UI (recupera de drift de terminal após resize tmux, artefatos de seleção com mouse, etc.) |
| `/status` | Mostra info da sessão — model, provider, profile, session ID, diretório de trabalho, título, timestamps created/updated, totais de tokens, estado agent-running — seguido de bloco local **Session recap** (contagens recentes de turnos user/assistant, contagem de tool results, top ferramentas usadas, últimos arquivos tocados, último prompt user e última resposta assistant). O recap é computado localmente da conversa in-memory; sem chamada LLM, sem impacto em prompt-cache. |
| `/context [all]` (alias: `/ctx`) | Breakdown visual da context window. No CLI/TUI: grid de glifos 5×20 (cada célula ≈ 1% da janela do model) mais tabela estimada por categoria — system prompt, definições de ferramentas, rules, índice de skills, MCP, subagentes, memória, conversa — versus espaço livre. Em plataformas de mensagens: gauge de uso com threshold/headroom de auto-compressão, stats de compressão, throughput cumulativo e a mesma tabela de categorias em texto plain. `/context all` anexa listagens de custo por skill e por toolset (custo de índice vs custo de load de SKILL.md; tokens de schema por toolset). Read-only e computado localmente — sem chamada LLM, sem impacto em prompt-cache. |
| `/agents` (alias: `/tasks`) | Mostra agentes ativos e tarefas em execução na sessão atual. |
| `/background <prompt>` (alias: `/bg`, `/btw`) | Roda prompt em sessão background separada. O agente processa seu prompt independentemente — sua sessão atual fica livre para outro trabalho. Resultados aparecem como painel quando a tarefa termina. Veja [Sessões background do CLI](/user-guide/cli#background-sessions). |
| `/branch [name]` (alias: `/fork`) | Ramifica a sessão atual (explora caminho diferente) |
| `/worktree [new [name]\|list]` | **Só CLI.** Inspeciona ou cria git worktrees isolados mid-session (inspirado no `/worktree new` do Copilot CLI). `/worktree` nu mostra o worktree ativo; `/worktree list` lista os worktrees do repo; `/worktree new [name]` cria um worktree sob `.worktrees/` (branched a partir do tip remoto recém-fetched, honrando `worktree_sync`) e retargets as ferramentas terminal e file da sessão para ele. Árvores nomeadas usam seu nome (branch `hermes/<name>`); sem nome recebem um `hermes-<id>` aleatório. Na saída a árvore é mantida só se tiver commits unpushed — mesmo ciclo de vida que `hermes -w`. Veja [Git Worktrees](/user-guide/git-worktrees). |
| `/handoff <platform>` | **Só CLI.** Entrega a sessão atual a uma plataforma de mensagens (Telegram, Discord, Slack, WhatsApp, Signal, Matrix). O gateway pega imediatamente, cria thread fresh em plataformas que suportam threads (tópicos Telegram, threads de text channel Discord, threads ancorados em mensagem Slack), re-vincula o destino ao seu session_id CLI para o transcript role-aware completo replay, e forja turno user sintético para o agente confirmar que está trabalhando no novo lugar. Seu CLI sai limpo em sucesso com dica de `/resume`; retome localmente a qualquer momento com `/resume <title>`. Recusado mid-turn. Requer gateway rodando e home channel configurado para a plataforma alvo (`/sethome` do chat destino). Veja [Handoff cross-platform](/user-guide/sessions#cross-platform-handoff). |
| `/journey [list\|delete <id>\|edit <id>]` (aliases: `/learning`, `/memory-graph`) | Abre a timeline learning journey de skills + memórias aprendidas. Funciona no CLI clássico, como overlay TUI e no app desktop (painel Star Map). Não disponível em plataformas de mensagens. Veja [Learning Journey](/user-guide/features/memory#learning-journey-journey). |

### Configuração {#configuration}

| Comando | Descrição |
|---------|-------------|
| `/config` | Mostra configuração atual |
| `/model [model-name]` | Mostra ou muda o model atual. Suporta: `/model claude-sonnet-4`, `/model provider:model` (troca providers), `/model custom:model` (endpoint custom), `/model custom:name:model` (provider custom nomeado), `/model custom` (auto-detect do endpoint), e aliases definidos pelo usuário (`/model fav`, `/model grok` — veja [Aliases de model custom](#custom-model-aliases)). Flags: `--global` persiste a mudança em config.yaml; `--session` força só sessão; `--once` aplica só ao próximo turno; `--refresh` re-busca a lista de models do provider; `--provider <name>` troca backend (só sessão salvo `--global`). Um `/model <name>` plain é só sessão salvo `model.persist_switch_by_default: true`. **Picker interativo:** rodar `/model` sem argumentos abre o picker provider→model; na lista de models você pode **digitar para fuzzy-filter** os models (ex. digite `grok` para estreitar aos models correspondentes), Backspace para encolher o filtro, Esc para limpá-lo (ou fechar o picker). A seleção sempre resolve para um model concreto — o filtro só estreita a lista, nunca adivinha. **Nota:** `/model` só troca entre providers já configurados. Para adicionar provider novo, saia da sessão e rode `hermes model` no terminal. **Nota de custo:** trocar models mid-conversation reseta o prompt cache — a cache key inclui o model, então seu próximo turno relê a conversa inteira a preço full de input em vez da taxa cached (~75% desconto). Esperado e inevitável, mas vale saber em sessões longas. |
| `/codex-runtime [auto\|codex_app_server\|on\|off]` | Alterna o [runtime Codex app-server](../user-guide/features/codex-app-server-runtime) opcional para models OpenAI/Codex. `auto` (padrão) usa chat completions padrão do Hermes; `codex_app_server` entrega turnos a subprocesso `codex app-server` para shell nativo, apply_patch, auth de assinatura ChatGPT e plugins Codex migrados. Efetivo na próxima sessão. |
| `/personality` | Define personalidade predefinida. `/personality none` (ou `default` / `neutral`) limpa o overlay e volta ao comportamento base. |
| `/verbose` | Cicla display de progresso de ferramentas: off → new → all → verbose. Pode ser [habilitado para mensagens](#notes) via config. |
| `/focus [on\|off\|status]` | Alterna **focus view** — modo display-only de saída reduzida mostrando só seu prompt e a resposta final. Compõe com `/verbose`: ligar snap tool progress para `off` e lembra seu modo anterior; `/focus off` restaura. Cada turno termina com linha de recovery dim (`⋯ 7 tool lines hidden · /focus off to show`) e badge persistente `◉ focus` na status bar. Nada é enviado diferente ao model — detalhe é ocultado, nunca descartado. |
| `/fast [normal\|fast\|status]` | Alterna fast mode — OpenAI Priority Processing / Anthropic Fast Mode. Opções: `normal`, `fast`, `status`. |
| `/reasoning [level\|show\|hide\|full\|clamp] [--global]` | Gerencia reasoning effort e display. Níveis incluem `none` / `minimal` / `low` / `medium` / `high` / `xhigh` / `max` / `ultra`. `show` / `hide` (ou `on` / `off`) alternam display de reasoning; `full` e `clamp` ajustam como reasoning é mostrado. `--global` persiste effort em config. |
| `/skin` | Mostra ou muda skin/tema de display |
| `/export [profile] [-o out.tar.gz]` | **Só CLI.** Empacota um profile num `.tar.gz` compartilhável — skills, memória, persona, crons, plugins, settings, e (no desktop) temas e layout. Credenciais (`auth.json`, `.env`) são removidas. Padrão: o profile ativo e `<name>.tar.gz` no diretório atual. Mesmo archive que `hermes profile export`; para um share versionado e atualizável use uma [distribuição de profile](../user-guide/profile-distributions.md) em vez disso. |
| `/import <archive.tar.gz> [--name <name>]` | **Só CLI.** Instala um archive de profile como um profile novo, inferindo o nome do archive a menos que `--name` seja dado. Recusa sobrescrever um profile existente e não pode importar como `default`. Cria um wrapper de shell quando o nome está livre. Veja [Exportar e importar um arquivo de profile](../user-guide/profile-distributions.md#export-and-import-a-profile-file). |
| `/statusbar` (alias: `/sb`) | Alterna a context/model status bar on ou off |
| `/battery [on\|off\|status]` | Alterna leitura de bateria codificada por cor como primeiro elemento da status bar (off por padrão; no-op sem bateria). |
| `/voice [on\|off\|tts\|status]` | Alterna voice mode CLI e playback falado. Gravação usa `voice.record_key` (padrão: `Ctrl+B`). |
| `/yolo` | Alterna YOLO mode — pula todos os prompts de aprovação de comando perigoso. |
| `/approvals [manual\|smart\|off]` | Mostra ou define o modo persistente de aprovação de comando perigoso. |
| `/footer [on\|off\|status]` | Alterna footer de runtime-metadata em respostas finais (mostra model, context % e cwd). |
| `/busy [queue\|steer\|interrupt\|status]` | Só CLI: controla o que Enter faz enquanto o Hermes trabalha — enfileira a nova mensagem, steer mid-turn ou interrupt imediato. |
| `/indicator [kaomoji\|emoji\|unicode\|ascii]` | Só CLI: escolhe estilo do busy-indicator TUI. |
| `/timestamps [on\|off\|status]` | Só CLI: alterna timestamps `[HH:MM]` em mensagens e em `/history`. |
| `/wake [on\|off\|status]` | Só CLI: alterna listener de wake word "Hey Hermes". |

### Ferramentas e skills {#tools--skills}

| Comando | Descrição |
|---------|-------------|
| `/tools [list\|disable\|enable] [name...]` | Gerencia ferramentas: lista disponíveis, ou disable/enable ferramentas específicas para a sessão atual. Desabilitar remove do toolset do agente e dispara reset de sessão. |
| `/toolsets` | Lista toolsets disponíveis |
| `/browser [connect\|disconnect\|status]` | Gerencia conexão CDP local Chromium-family. `connect` anexa ferramentas de browser a Chrome, Brave, Chromium ou Edge rodando (padrão: `http://127.0.0.1:9222`). `disconnect` desanexa. `status` mostra conexão atual. Auto-lança browser Chromium-family suportado se nenhum debugger for detectado. |
| `/skills` | Busca, instala, inspeciona ou gerencia skills de registries online. Também superfície de review para o gate de write-approval de skills: `/skills pending`, `/skills diff <id>`, `/skills approve <id>`, `/skills reject <id>`, `/skills approval on\|off`. Veja [Gating agent skill writes](/user-guide/features/skills#gating-agent-skill-writes-skillswrite_approval). |
| `/memory [pending\|approve\|reject\|approval]` | Revisa writes de memória pendentes staged pelo gate write-approval (`memory.write_approval`) e alterna o gate. Veja [Controlling memory writes](/user-guide/features/memory#controlling-memory-writes-write_approval). |
| `/bundles` | Lista skill bundles configurados — aliases slash `/<name>` que preload várias skills de uma vez. Configure em `bundles:` em `~/.hermes/config.yaml`. Veja [Skill Bundles](/user-guide/features/skills#skill-bundles). |
| `/learn <what to learn from>` | Destila skill reutilizável de qualquer coisa que você descrever — diretório, URL, workflow que acabou de percorrer com o agente, ou notas coladas. Aberto: o agente coleta fontes com suas ferramentas e autoria um `SKILL.md` seguindo os padrões de authoring da casa. Funciona no CLI, gateway de mensagens, TUI e página Skills do dashboard. |
| `/init [notes]` | Gera ou atualiza instruções de projeto `AGENTS.md` a partir de scan do repo (port do Codex `/init`). O agente inspeciona manifests, layout e configs de toolchain com ferramentas read-only, depois escreve `AGENTS.md` conciso — ou, se existir, merge-update preservando seu conteúdo. Notas opcionais direcionam a ênfase. Funciona no CLI, gateway de mensagens e TUI. |
| `/cron` | Gerencia tarefas agendadas (list, add/create, edit, pause, resume, run, remove) |
| `/suggestions [accept\|dismiss N\|catalog\|clear]` (alias: `/suggest`) | Revisa automações sugeridas. Use `/suggestions` para listar pendentes, `/suggestions accept <id>` para criar a automação proposta, `/suggestions dismiss <id>` para rejeitar uma, `/suggestions catalog` para adicionar automações starter curadas, e `/suggestions clear` para limpar registros de sugestões resolvidas. Jobs aceitos preservam a superfície atual como origem de entrega. |
| `/blueprint [name] [slot=value ...]` (alias: `/bp`) | Configura automação a partir de template blueprint. `/blueprint` bare lista o catálogo; `/blueprint <name>` inicia fluxo guiado de preenchimento de slots no próximo turno do agente; `/blueprint <name> slot=value ...` cria o job diretamente. |
| `/curator` | Manutenção de skills em background — `status`, `run`, `pin`, `archive`. Veja [Curator](/user-guide/features/curator). |
| `/kanban <action>` | Dirija o board de colaboração multi-profile, multi-project sem sair do chat. Superfície completa `hermes kanban` disponível: `/kanban list`, `/kanban show t_abc`, `/kanban create "title" --assignee X`, `/kanban comment t_abc "text"`, `/kanban unblock t_abc`, `/kanban dispatch`, etc. Suporte multi-board incluído: `/kanban boards list`, `/kanban boards create <slug>`, `/kanban boards switch <slug>`, `/kanban --board <slug> <action>`. Veja [Slash command Kanban](/user-guide/features/kanban#kanban-slash-command). |
| `/reload-mcp` (alias: `/reload_mcp`) | Recarrega servidores MCP de config.yaml |
| `/reload-skills` (alias: `/reload_skills`) | Re-scan `~/.hermes/skills/` por skills recém-instaladas ou removidas |
| `/reload` | Recarrega variáveis `.env` na sessão rodando (pega novas API keys sem restart) |
| `/plugins` | Lista plugins instalados e status |
| `/pet [list\|<slug>]` | Alterna ou adota mascote [petdex](/user-guide/features/pets). `/pet` alterna o painel, `/pet list` mostra pets instalados, `/pet <slug>` adota um específico. |
| `/hatch <description>` (alias: `/generate-pet`) | Gera pet petdex totalmente novo a partir de descrição em texto, usando o backend de imagem configurado (OpenRouter / Nous Portal). Veja [Pets](/user-guide/features/pets). |

### Info {#info}

| Comando | Descrição |
|---------|-------------|
| `/help` | Mostra comandos disponíveis, agrupados por categoria. Comandos core aparecem por padrão com skill commands colapsados a uma contagem de uma linha; `/help skills` lista todos os skill commands, e `/help <text>` filtra comandos (e skills correspondentes) por substring. |
| `/palette` | Abre a fuzzy command palette (também **Ctrl+P**) — digite para filtrar todos os commands + skills, ↑/↓ para mover, Enter para inserir o comando selecionado no composer (nunca auto-roda), Esc para cancelar. Matching é ranqueado pelo nome do comando primeiro, então uma query curta permanece precisa. |
| `/version` | Mostra versão, build e info de ambiente do Hermes Agent. |
| `/whoami` | Mostra seu nível de acesso a slash commands (admin / user). |
| `/usage` | Mostra uso de tokens, breakdown de custo, duração da sessão e — quando disponível do provider ativo — seção **Account limits** com quota/credits/plano restantes puxados live da API do provider. |
| `/topup` | Mostra seu saldo Nous e gerencia billing no portal (substitui os antigos `/credits` e `/billing`). |
| `/subscription` (alias: `/upgrade`) | **Só CLI.** Veja seu plano Nous e mude no browser. |
| `/insights` | Mostra insights de uso e analytics (últimos 30 dias) |
| `/update` | Atualiza o Hermes Agent para a versão mais recente. |
| `/platforms` (alias: `/gateway`) | Mostra status de plataformas gateway/mensagens (visão resumo só CLI). |
| `/paste` | Anexa imagem da clipboard |
| `/copy [number]` | Copia última resposta assistant para clipboard (ou a N-ésima de trás com número). Só CLI. |
| `/image <path>` | Anexa arquivo de imagem local para seu próximo prompt. |
| `/debug` | Faz upload de debug report (info de sistema + logs) e obtém links compartilháveis. Também disponível em mensagens. |
| `/update` | Atualiza o Hermes Agent para a versão mais recente. |
| `/profile` | Mostra nome do profile ativo e home directory |

### Sair {#exit}

| Comando | Descrição |
|---------|-------------|
| `/quit` | Sai do CLI (também: `/exit`). |

### Slash commands dinâmicos do CLI {#dynamic-cli-slash-commands}

| Comando | Descrição |
|---------|-------------|
| `/<skill-name>` | Carrega qualquer skill instalada como comando on-demand. Exemplo: `/gif-search`, `/github-pr-workflow`, `/excalidraw`. |
| `/skills ...` | Busca, navega, inspeciona, instala, audita, publica e configura skills de registries e o catálogo official optional-skills. |

### Quick Commands {#quick-commands}

Quick commands definidos pelo usuário mapeiam um slash command curto para shell command ou outro slash command. Configure em `~/.hermes/config.yaml`:

```yaml
quick_commands:
  status:
    type: exec
    command: systemctl status hermes-agent
  deploy:
    type: exec
    command: scripts/deploy.sh
  inbox:
    type: alias
    target: /gmail unread
```

Depois digite `/status`, `/deploy` ou `/inbox` no CLI ou plataforma de mensagens. Quick commands resolvem no dispatch time e podem não aparecer em toda tabela autocomplete/help built-in.

Atalhos de prompt só string não são suportados como quick commands. Coloque prompts reutilizáveis longos numa skill, ou use `type: alias` apontando para slash command existente.

### Aliases de model custom {#custom-model-aliases}

Defina nomes curtos para models que você usa muito, depois acesse com `/model <alias>` no CLI ou qualquer plataforma de mensagens. Aliases funcionam igual em ambos, em switches só sessão (padrão) e `--global`.

Dois formatos de config são suportados:

**Forma completa** — fixa model, provider e opcionalmente base URL. Coloque em `~/.hermes/config.yaml`:

```yaml
model_aliases:
  fav:
    model: claude-sonnet-4.6
    provider: anthropic
  grok:
    model: grok-4
    provider: x-ai
  ollama-qwen:
    model: qwen3-coder:30b
    provider: custom
    base_url: http://localhost:11434/v1
```

**Forma curta** — `provider/model` em uma string. Defina do shell sem editar YAML:

```bash
hermes config set model.aliases.fav anthropic/claude-opus-4.6
hermes config set model.aliases.grok x-ai/grok-4
```

Depois no chat:

```
/model fav            # só sessão
/model grok --global  # também persiste mudança de current-model em config.yaml
```

Aliases do usuário têm precedência sobre nomes curtos built-in, então nomear alias `sonnet`, `kimi`, `opus`, etc. sombreia o built-in. Nomes de alias são case-insensitive.

### Resolução de alias {#alias-resolution}

Comandos suportam prefix matching: digitar `/h` resolve para `/help`, `/mod` para `/model`. Quando prefixo é ambíguo (combina com múltiplos comandos), a primeira match na ordem do registry vence. Nomes completos de comando e aliases registrados sempre têm prioridade sobre prefix matches.

## Slash commands de mensagens {#messaging-slash-commands}

> **Comandos de thread Slack (prefixo `!`):**
> O Slack em si bloqueia slash commands nativos dentro de threads de mensagem ("/queue is not supported in threads. Sorry!") e nunca os entrega ao Hermes. Dentro de thread Slack, use prefixo `!` — `!stop`, `!new`, `!status` — e o gateway despacha exatamente como a forma slash. `@Hermes !stop` e `@Hermes /stop` funcionam em threads também. Só o primeiro token é checado contra a lista de comandos conhecidos, então mensagens como `!nice work` passam ao agente inalteradas. Veja [Using commands inside threads](/user-guide/messaging/slack#using-commands-inside-threads-the-cmd-prefix) para detalhes.

O gateway de mensagens suporta os seguintes comandos built-in dentro de chats Telegram, Discord, Slack, WhatsApp, Signal, Email, Home Assistant e Teams:

| Comando | Descrição |
|---------|-------------|
| `/start` | Comando de protocolo de plataforma. Muitas plataformas (Telegram, Discord, …) enviam `/start` automaticamente na primeira vez que um usuário abre conversa com bot. O Hermes reconhece o ping silenciosamente — sem resposta do agente, sem burn de sessão — para handshakes de first-contact não desperdiçarem turno. Você também pode enviar explicitamente para confirmar que o gateway está alcançável. |
| `/new [name]` (alias: `/reset`) | Inicia nova sessão (session ID + histórico fresh). `[name]` opcional define título inicial. Acrescente `now`, `--yes` ou `-y` para pular modal de confirmação — ex.: `/reset now`, `/new --yes my-experiment`. |
| `/status` | Mostra info da sessão, seguido de bloco local **Session recap** (contagens recentes de turnos, top ferramentas usadas, arquivos tocados, último prompt + resposta). |
| `/stop` | Mata todos os processos em background e interrompe o agente em execução. |
| `/model [provider:model]` | Mostra ou muda o model. Suporta troca de provider (`/model zai:glm-5`), endpoints custom (`/model custom:model`), providers custom nomeados (`/model custom:local:qwen`), auto-detect (`/model custom`) e aliases definidos pelo usuário (`/model fav`, `/model grok` — veja [Aliases de model custom](#custom-model-aliases)). Use `--global` para persistir em config.yaml. **Nota:** `/model` só troca entre providers já configurados. Para adicionar provider novo ou configurar API keys, use `hermes model` no terminal (fora da sessão de chat). **Nota de custo:** troca mid-session reseta prompt cache (a cache key inclui o model), então a próxima mensagem relê a conversa inteira a preço full de input. |
| `/codex-runtime [auto\|codex_app_server\|on\|off]` | Alterna o [runtime Codex app-server](../user-guide/features/codex-app-server-runtime) opcional. Persiste em `model.openai_runtime` em config.yaml e evicta o agent cached para a próxima mensagem pegar o runtime novo. Efetivo na próxima sessão. |
| `/personality [name]` | Define overlay de personalidade para a sessão. `/personality none` (ou `default` / `neutral`) limpa. |
| `/fast [normal\|fast\|status]` | Alterna fast mode — OpenAI Priority Processing / Anthropic Fast Mode. |
| `/retry` | Repete a última mensagem. |
| `/undo` | Remove a última troca. |
| `/sethome` (alias: `/set-home`) | Marca o chat atual como home channel da plataforma para entregas. |
| `/compress [here [N] \| focus topic]` | Comprime contexto da conversa manualmente. `/compress here [N]` mantém as N trocas mais recentes (padrão 2) verbatim e resume o resto. Focus topic estreita o que um resumo completo preserva. |
| `/topic [off\|help\|session-id]` | **Só Telegram DM.** Gerencia modo multi-sessão topic gerenciado pelo usuário. `/topic` habilita ou mostra status; `/topic off` desabilita e limpa bindings; `/topic help` mostra uso; `/topic <session-id>` dentro de topic restaura sessão anterior. Veja [Multi-session DM mode](/user-guide/messaging/telegram#multi-session-dm-mode-topic). |
| `/title [name]` | Define ou mostra título da sessão. |
| `/resume [name]` | Retoma sessão nomeada anteriormente. |
| `/sessions [all] [search <query>]` | Lista sessões anteriores deste chat. `/sessions search <query>` filtra por match de título/id (mais recentemente ativas primeiro); `/sessions all` lista across origins (só admin). |
| `/usage` | Mostra uso de tokens, breakdown de custo estimado (input/output), estado da context window, duração da sessão e — quando disponível do provider ativo — seção **Account limits** com quota/credits restantes da API do provider. |
| `/topup` | Mostra saldo Nous e gerencia billing no portal. |
| `/whoami` | Mostra nível de acesso a slash commands (admin / user). |
| `/insights [days]` | Mostra analytics de uso. |
| `/reasoning [level\|show\|hide\|full\|clamp] [--global]` | Muda reasoning effort (níveis até `max` / `ultra`) ou alterna display de reasoning (`full` / `clamp` incluídos). `--global` persiste em config. |
| `/voice [on\|off\|tts\|join\|channel\|leave\|status]` | Controla respostas faladas no chat. `join`/`channel`/`leave` gerenciam modo voice channel Discord. |
| `/rollback [number]` | Lista ou restaura checkpoints de filesystem. |
| `/diff [staged\|all\|session] [--stat]` | Mostra mudanças git no diretório de trabalho (fenced e truncado aos limites de mensagem da plataforma). `session` mostra diff cumulativo de tudo que o Hermes mudou; `--stat` mostra só o resumo. |
| `/background <prompt>` | Roda prompt em sessão background separada. Resultados são entregues de volta ao mesmo chat quando a tarefa termina. Veja [Sessões background de mensagens](/user-guide/messaging/#background-sessions). |
| `/queue <prompt>` (alias: `/q`) | Enfileira prompt para o próximo turno sem interromper o atual. |
| `/steer <prompt>` | Injeta mensagem após a próxima chamada de ferramenta sem interromper — o model pega na próxima iteração em vez de novo turno. |
| `/goal <text>` | Define meta contínua em direção à qual o Hermes trabalha entre turnos — nossa versão do Ralph loop. Model judge checa após cada turno; se não pronto, Hermes auto-continua até estar, você pausar/limpar, ou o orçamento de turnos (padrão 20) acabar. Subcomandos: `/goal status`, `/goal pause`, `/goal resume`, `/goal clear`. Seguro mid-agent para status/pause/clear; definir meta nova requer `/stop` primeiro. Veja [Metas persistentes](/user-guide/features/goals). |
| `/subgoal <text>` | Anexa critérios à `/goal` ativa mid-loop (`/subgoal`, `/subgoal remove <N>`, `/subgoal clear`). |
| `/heartbeat every <interval> <prompt>` (alias: `/hb`) | Define prompt recorrente que reentra nesta sessão quando idle. Subcomandos: `status`, `pause`, `resume`, `clear`. No Slack use `/hermes heartbeat …`. |
| `/refine [focus]` | Roda revisão de auto-melhoria memória/skill agora, opcionalmente com instruções de focus. No Slack use `/hermes refine …`. |
| `/moa <prompt>` | Roda um prompt pelo preset padrão [Mixture of Agents](/user-guide/features/mixture-of-agents), depois restaura model da sessão. |
| `/branch [name]` (alias: `/fork`) | Ramifica sessão atual (explora caminho diferente). |
| `/agents` (alias: `/tasks`) | Mostra agentes ativos e tarefas em execução. |
| `/sessions` | Navega e retoma sessões anteriores. |
| `/context [all]` (alias: `/ctx`) | Gauge de uso da context window e breakdown por categoria (forma texto amigável a mensagens). `/context all` adiciona detalhe de custo por skill / por toolset. |
| `/egress [status]` | Mostra status do proxy de egress Docker. |
| `/init [notes]` | Gera ou atualiza `AGENTS.md` a partir de scan do repo. |
| `/learn <what to learn from>` | Destila skill reutilizável de qualquer coisa que você descrever. |
| `/bundles` | Lista skill bundles configurados (aliases `/<name>` que preload várias skills). |
| `/reload-skills` (alias: `/reload_skills`) | Re-scan `~/.hermes/skills/` por skills recém-instaladas ou removidas. |
| `/footer [on\|off\|status]` | Alterna footer de runtime-metadata em respostas finais (mostra model, context % e cwd). |
| `/curator [status\|run\|pin\|archive]` | Controles de manutenção de skills em background. |
| `/suggestions [accept\|dismiss N\|catalog\|clear]` | Revisa automações sugeridas direto no chat. `/suggestions` lista pendentes, `catalog` adiciona automações starter curadas, e `clear` poda registros de sugestões resolvidas. Sugestões aceitas mantêm este chat/thread como origem de entrega do job. |
| `/blueprint [name] [slot=value ...]` | Navega blueprints cron, inicia conversa guiada de preenchimento de slots, ou cria job blueprint diretamente. Jobs criados diretamente entregam de volta ao chat/thread atual. |
| `/memory [pending\|approve\|reject\|approval]` | Revisa writes de memória pendentes staged pelo gate write-approval (`memory.write_approval`) — aprove ou rejeite direto no chat — e alterne o gate com `/memory approval on\|off`. Veja [Controlling memory writes](/user-guide/features/memory#controlling-memory-writes-write_approval). |
| `/skills [pending\|approve\|reject\|diff\|approval]` | Revisa writes de **skill** pendentes staged pelo gate write-approval (`skills.write_approval`). Mostra gist de uma linha por write staged; `/skills diff <id>` é truncado para chat — leia diff completo no CLI ou em `~/.hermes/pending/skills/<id>.json`. Só aparece quando gate está on (ou writes staged restam); search/install continuam só CLI. |
| `/kanban <action>` | Dirija board de colaboração multi-profile, multi-project do chat — superfície de argumentos idêntica ao CLI. Bypassa running-agent guard, então `/kanban unblock t_abc`, `/kanban comment t_abc "…"`, `/kanban list --mine`, `/kanban boards switch <slug>`, etc. funcionam mid-turn. `/kanban create …` auto-inscreve o chat de origem nos eventos de terminal da nova tarefa. Veja [Slash command Kanban](/user-guide/features/kanban#kanban-slash-command). |
| `/platform <list\|pause\|resume> [name]` | Opere plataforma gateway rodando direto do chat. `/platform list` mostra todo adapter e estado (running, paused-by-breaker, manually-paused); `/platform pause <name>` para de despachar novas mensagens para aquele adapter sem descarregá-lo; `/platform resume <name>` reabilita e limpa circuit breaker tripped quando upstream está saudável. |
| `/reload-mcp` (alias: `/reload_mcp`) | Recarrega servidores MCP de config. |
| `/verbose` | Cicla display de progresso de ferramentas. **Off por padrão em mensagens** — habilite com `display.tool_progress_command: true` em `config.yaml`. |
| `/yolo` | Alterna YOLO mode — pula todos os prompts de aprovação de comando perigoso. |
| `/commands [page]` | Navega todos os comandos e skills (paginado). |
| `/approve [session\|always]` | Aprova e executa comando perigoso pendente. `session` aprova só nesta sessão; `always` adiciona à allowlist permanente. |
| `/deny` | Rejeita comando perigoso pendente. |
| `/update` | Atualiza Hermes Agent para versão mais recente. |
| `/restart` | Reinicia gateway gracefully após drenar runs ativos. Quando gateway volta online, envia confirmação ao chat/thread do solicitante. |
| `/debug` | Faz upload de debug report (info de sistema + logs) e obtém links compartilháveis. |
| `/help` | Mostra help de mensagens. |
| `/<skill-name>` | Invoca qualquer skill instalada por nome. |

## Notas {#notes}

- `/skin`, `/snapshot`, `/export`, `/import`, `/reload`, `/tools`, `/toolsets`, `/browser`, `/config`, `/cron`, `/platforms`, `/paste`, `/image`, `/statusbar`, `/battery`, `/focus`, `/plugins`, `/busy`, `/indicator`, `/wake`, `/journey`, `/redraw`, `/clear`, `/history`, `/save`, `/copy`, `/handoff`, `/prompt`, `/pet`, `/hatch`, `/timestamps`, `/subscription` e `/quit` são comandos **só CLI**.
- `/skills` é **só CLI para search/browse/install**; subcomandos de review write-approval (`pending`, `approve`, `reject`, `diff`, `approval`) também funcionam em plataformas de mensagens quando `skills.write_approval` está on. `/memory` funciona em **ambas** superfícies.
- `/verbose` é **só CLI por padrão**, mas pode ser habilitado para plataformas de mensagens com `display.tool_progress_command: true` em `config.yaml`. Quando habilitado, cicla o modo `display.tool_progress` e salva em config.
- `/focus` e `/verbose` compartilham um caminho de supressão (`display.tool_progress`), então nunca se contradizem: `/focus on` fixa tool progress em `off` e guarda seu modo em `display.focus_saved_tool_progress`; `/focus off` restaura; ciclar `/verbose` com focus on traz o modo de volta e limpa o badge focus. Focus view é só display — nunca muda histórico de conversa, system prompt ou qualquer coisa enviada ao model, então zero impacto em prompt-cache.
- `/sethome`, `/restart`, `/approve`, `/deny`, `/topic`, `/platform` e `/commands` são comandos **só mensagens**.
- `/status`, `/egress`, `/version`, `/whoami`, `/background`, `/queue`, `/steer`, `/voice`, `/reload-mcp`, `/reload-skills`, `/rollback`, `/diff`, `/debug`, `/fast`, `/approvals`, `/footer`, `/curator`, `/kanban`, `/topup`, `/suggestions`, `/blueprint`, `/learn`, `/init`, `/sessions` e `/yolo` funcionam **tanto** no CLI quanto no gateway de mensagens.
- `/voice join`, `/voice channel` e `/voice leave` só fazem sentido no Discord.
- No TUI, `/sessions` mostra sessões live no processo TUI atual. Use `/resume [name]` ou `hermes --tui --resume <id-or-title>` para transcripts salvos ou fechados.

## Prompts de confirmação para comandos destrutivos {#confirmation-prompts-for-destructive-commands}

O CLI pede confirmação antes de rodar slash commands que descartam estado de sessão não salvo. O conjunto destrutivo atual é:

| Comando | O que destrói |
|---------|------------------|
| `/clear` | Limpa a tela e inicia sessão fresh — session ID atual e histórico in-memory se vão. |
| `/new` / `/reset` | Inicia sessão fresh (novo session ID + histórico vazio). |
| `/undo` | Remove a última troca user/assistant do histórico. |
| `/exit --delete` / `/quit --delete` | Sai **e** exclui permanentemente histórico SQLite da sessão atual e transcripts on-disk. |

Para cada um o CLI abre modal de três escolhas: **Approve Once** (prosseguir desta vez), **Always Approve** (prosseguir e persistir `approvals.destructive_slash_confirm: false` para comandos destrutivos futuros rodarem sem prompt), ou **Cancel**.

**Skip inline:** acrescente `now`, `--yes` ou `-y` para bypass do modal numa invocação — ex.: `/reset now`, `/new --yes my-session`, `/clear -y`, `/undo -y`. Útil quando modal não renderiza bem no seu terminal (veja [issue #30768](https://github.com/NousResearch/hermes-agent/issues/30768) para PowerShell nativo Windows) ou ao scriptar contra o CLI.

Defina `approvals.destructive_slash_confirm: false` em `~/.hermes/config.yaml` para desabilitar prompts globalmente; volte para `true` para reabilitar. Veja [Security — Destructive slash command confirmation](../user-guide/security.md#dangerous-command-approval) para contexto.
