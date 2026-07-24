---
sidebar_position: 2
title: "Referência de Slash Commands"
description: "Referência completa dos slash commands da CLI interativa e do gateway de mensagens"
---

# Referência de Slash Commands

O Hermes tem duas superfícies de slash commands, ambas orientadas por um `COMMAND_REGISTRY` central em `hermes_cli/commands.py`:

- **Slash commands da CLI interativa** — despachados por `cli.py`, com autocompletar a partir do registro
- **Slash commands de mensagens** — despachados por `gateway/run.py`, com texto de ajuda e menus de plataforma gerados a partir do registro

Skills instaladas também são expostas como slash commands dinâmicos em ambas as superfícies. Isso inclui skills incluídas como `/plan`, que abre o modo plano e salva planos em markdown em `.hermes/plans/` relativo ao workspace/diretório de trabalho ativo do backend.

## Permissões e divisão admin/usuário {#permissions-and-adminuser-split}

Toda plataforma de mensagens que suporta uma allowlist por usuário (Telegram, Discord, Slack, Matrix, Mattermost, Signal, …) também suporta uma divisão de slash commands em dois níveis: **admins** recebem todos os comandos registrados, **usuários comuns** só recebem os nomes listados em `user_allowed_commands` (mais o piso sempre permitido `/help` e `/whoami`). Configure `allow_admin_from` e `user_allowed_commands` (e os equivalentes por grupo `group_allow_admin_from` / `group_user_allowed_commands`) dentro do bloco `extra:` da plataforma em `~/.hermes/gateway-config.yaml`.

Veja a documentação de cada plataforma para exemplos — a estrutura é idêntica entre plataformas:

- [Telegram](../user-guide/messaging/telegram.md#slash-command-access-control)
- [Discord](../user-guide/messaging/discord.md)
- [Slack](../user-guide/messaging/slack.md)
- [Matrix](../user-guide/messaging/matrix.md)
- [Mattermost](../user-guide/messaging/mattermost.md)
- [Signal](../user-guide/messaging/signal.md)

Se `allow_admin_from` não estiver definido para um escopo, esse escopo permanece no modo de compatibilidade retroativa irrestrito — todo usuário permitido pode executar todo comando.

## Slash commands da CLI interativa {#interactive-cli-slash-commands}

Digite `/` na CLI para abrir o menu de autocompletar. Comandos embutidos não diferenciam maiúsculas/minúsculas.

### Sessão

| Comando | Descrição |
|---------|-------------|
| `/new [name]` (alias: `/reset`) | Inicia uma nova sessão (novo ID de sessão + histórico). O `[name]` opcional define o título inicial da sessão — ex.: `/new my-experiment` abre uma nova sessão já intitulada `my-experiment`, facilitando encontrá-la depois com `/resume` ou `/sessions`. Acrescente `now`, `--yes` ou `-y` para pular o modal de confirmação — ex.: `/reset now`, `/new --yes my-experiment`. |
| `/clear` | Limpa a tela e inicia uma nova sessão |
| `/history` | Mostra o histórico da conversa |
| `/save` | Salva a conversa atual |
| `/prompt` (alias: `/compose`) | Compõe seu próximo prompt no `$EDITOR` (markdown) em vez da entrada inline — útil para prompts longos, multilinha ou cuidadosamente formatados. |
| `/retry` | Tenta novamente a última mensagem (reenvia ao agente) |
| `/undo` | Remove a última troca usuário/assistente |
| `/title` | Define um título para a sessão atual (uso: /title My Session Name) |
| `/compress [here [N] \| focus topic]` | Comprime manualmente o contexto da conversa (limpa memórias + resume). `/compress here [N]` resume tudo exceto as N trocas mais recentes (padrão 2), mantidas literalmente — escolha seu próprio limite de compressão. Um tópico de foco (`focus topic`) restringe o que um resumo completo preserva. |
| `/rollback` | Lista ou restaura checkpoints do sistema de arquivos (uso: /rollback [number]) |
| `/snapshot [create\|restore <id>\|prune]` (alias: `/snap`) | Cria ou restaura snapshots de estado da configuração/estado do Hermes. `create [label]` salva um snapshot, `restore <id>` reverte para ele, `prune [N]` remove snapshots antigos, ou lista todos sem argumentos. |
| `/stop` | Mata todos os processos em segundo plano em execução |
| `/queue <prompt>` (alias: `/q`) | Enfileira um prompt para o próximo turno (não interrompe a resposta atual do agente). |
| `/steer <prompt>` | Injeta uma nota durante a execução que chega ao agente **após a próxima chamada de ferramenta** — sem interrupção, sem novo turno de usuário. O texto é acrescentado ao conteúdo do último resultado de ferramenta quando a ferramenta atual termina, dando ao agente novo contexto sem quebrar o loop atual de chamada de ferramentas. Use isso para direcionar no meio de uma tarefa (ex.: "foque no módulo de auth" enquanto o agente está executando testes). |
| `/goal <text>` | Define uma meta permanente pela qual o Hermes trabalha ao longo dos turnos — nossa versão do loop Ralph. Após cada turno, um modelo juiz auxiliar decide se a meta foi cumprida; se não, o Hermes continua automaticamente. Subcomandos: `/goal status`, `/goal pause`, `/goal resume`, `/goal clear`. O orçamento padrão é de 20 turnos (`goals.max_turns`); qualquer mensagem real do usuário antecipa o loop de continuação, e o estado sobrevive a `/resume`. Veja [Metas Persistentes](/user-guide/features/goals) para o passo a passo completo. |
| `/subgoal <text>` | Acrescenta um critério fornecido pelo usuário à meta ativa no meio do loop. O prompt de continuação apresenta todas as submetas ao agente literalmente, e o juiz as considera em seu veredito DONE/CONTINUE — então a meta não é marcada como concluída até que a meta original **e** todas as submetas sejam atendidas. Subcomandos: `/subgoal` (lista), `/subgoal remove <N>`, `/subgoal clear`. Requer um `/goal` ativo. |
| `/moa <prompt>` | Executa um único prompt pelo preset padrão de [Mixture of Agents](/user-guide/features/mixture-of-agents), depois restaura seu modelo atual. Pontual — não altera o modelo da sua sessão. |
| `/resume [name]` | Retoma uma sessão previamente nomeada |
| `/sessions` (alias na TUI: `/switch`) | CLI clássica: navega e retoma sessões anteriores em um seletor interativo. TUI: abre o alternador de sessões ativas para as sessões TUI atualmente abertas. Use `/sessions new` na TUI para iniciar outra sessão ativa imediatamente. |
| `/redraw` | Força uma repintura completa da UI (recupera de drift de terminal após redimensionamento do tmux, artefatos de seleção do mouse, etc.) |
| `/status` | Mostra informações da sessão — modelo, provedor, perfil, ID da sessão, diretório de trabalho, título, timestamps de criação/atualização, totais de tokens, estado de execução do agente — seguido por um bloco local de **Resumo da sessão** (contagens recentes de turnos de usuário/assistente, contagem de resultados de ferramenta, principais ferramentas usadas, últimos arquivos tocados, o último prompt do usuário e a última resposta do assistente). O resumo é calculado localmente a partir da conversa em memória; sem chamada de LLM, sem impacto no cache de prompt. |
| `/agents` (alias: `/tasks`) | Mostra agentes ativos e tarefas em execução na sessão atual. |
| `/background <prompt>` (alias: `/bg`, `/btw`) | Executa um prompt em uma sessão separada em segundo plano. O agente processa seu prompt de forma independente — sua sessão atual fica livre para outro trabalho. Os resultados aparecem como um painel quando a tarefa termina. Veja [Sessões em Segundo Plano da CLI](/user-guide/cli#background-sessions). |
| `/branch [name]` (alias: `/fork`) | Cria um ramo da sessão atual (explora um caminho diferente) |
| `/handoff <platform>` | **Somente CLI.** Transfere a sessão atual para uma plataforma de mensagens (Telegram, Discord, Slack, WhatsApp, Signal, Matrix). O gateway assume imediatamente, cria uma nova thread em plataformas que suportam threads (tópicos do Telegram, threads de canal de texto do Discord, threads ancoradas em mensagem do Slack), religa o destino ao `session_id` da sua CLI para que a transcrição completa com reconhecimento de papéis seja reproduzida, e forja um turno de usuário sintético para que o agente confirme que está trabalhando no novo lugar. Sua CLI encerra corretamente em caso de sucesso com uma dica de `/resume`; retome localmente a qualquer momento com `/resume <title>`. Recusado no meio de um turno. Requer que o gateway esteja em execução e um canal home configurado para a plataforma de destino (`/sethome` a partir do chat de destino). Veja [Handoff Entre Plataformas](/user-guide/sessions#cross-platform-handoff). |

### Configuração

| Comando | Descrição |
|---------|-------------|
| `/config` | Mostra a configuração atual |
| `/model [model-name]` | Mostra ou altera o modelo atual. Suporta: `/model claude-sonnet-4`, `/model provider:model` (troca de provedor), `/model custom:model` (endpoint customizado), `/model custom:name:model` (provedor customizado nomeado), `/model custom` (detecção automática a partir do endpoint), e aliases definidos pelo usuário (`/model fav`, `/model grok` — veja [Aliases de modelo customizados](#custom-model-aliases)). Use `--global` para persistir a mudança no config.yaml. **Observação:** `/model` só pode alternar entre provedores já configurados. Para adicionar um novo provedor, saia da sessão e execute `hermes model` no seu terminal. **Observação de custo:** trocar de modelo no meio da conversa reinicia o cache de prompt — a chave de cache inclui o modelo, então seu próximo turno relê a conversa inteira ao preço total de entrada em vez da taxa em cache com ~75% de desconto. Esperado e inevitável, mas vale saber em sessões longas. |
| `/codex-runtime [auto\|codex_app_server\|on\|off]` | Alterna o [runtime opcional do Codex app-server](../user-guide/features/codex-app-server-runtime) para modelos OpenAI/Codex. `auto` (padrão) usa as chat completions padrão do Hermes; `codex_app_server` passa os turnos a um subprocesso `codex app-server` para shell nativo, apply_patch, autenticação por assinatura ChatGPT e plugins Codex migrados. Efetivo na próxima sessão. |
| `/personality` | Define uma personalidade predefinida |
| `/verbose` | Alterna a exibição de progresso de ferramentas: off → new → all → verbose. Pode ser [habilitado para mensagens](#notes) via configuração. |
| `/fast [normal\|fast\|status]` | Alterna o modo rápido — OpenAI Priority Processing / Anthropic Fast Mode. Opções: `normal`, `fast`, `status`. |
| `/reasoning` | Gerencia o esforço e a exibição de raciocínio (uso: /reasoning [level\|show\|hide]) |
| `/skin` | Mostra ou altera o skin/tema de exibição |
| `/statusbar` (alias: `/sb`) | Alterna a barra de status de contexto/modelo |
| `/voice [on\|off\|tts\|status]` | Alterna o modo de voz da CLI e a reprodução falada. A gravação usa `voice.record_key` (padrão: `Ctrl+B`). |
| `/yolo` | Alterna o modo YOLO — pula todos os prompts de aprovação de comandos perigosos. |
| `/footer [on\|off\|status]` | Alterna o rodapé de metadados de runtime do gateway nas respostas finais (mostra modelo, % de contexto e cwd). |
| `/busy [queue\|steer\|interrupt\|status]` | Somente CLI: controla o que pressionar Enter faz enquanto o Hermes está trabalhando — enfileirar a nova mensagem, direcionar no meio do turno, ou interromper imediatamente. |
| `/indicator [kaomoji\|emoji\|unicode\|ascii]` | Somente CLI: escolhe o estilo do indicador de ocupado da TUI. |
| `/timestamps [on\|off\|status]` | Somente CLI: alterna timestamps `[HH:MM]` nas mensagens e em `/history`. |

### Ferramentas e Skills

| Comando | Descrição |
|---------|-------------|
| `/tools [list\|disable\|enable] [name...]` | Gerencia ferramentas: lista ferramentas disponíveis, ou desativa/ativa ferramentas específicas para a sessão atual. Desativar uma ferramenta a remove do toolset do agente e dispara uma reinicialização de sessão. |
| `/toolsets` | Lista os toolsets disponíveis |
| `/browser [connect\|disconnect\|status]` | Gerencia uma conexão CDP local da família Chromium. `connect` conecta as ferramentas de navegador a uma instância em execução do Chrome, Brave, Chromium ou Edge (padrão: `http://127.0.0.1:9222`). `disconnect` desconecta. `status` mostra a conexão atual. Inicia automaticamente um navegador suportado da família Chromium se nenhum debugger for detectado. |
| `/skills` | Busca, instala, inspeciona ou gerencia skills de registros online. Também é a superfície de revisão do gate de aprovação de escrita de skills: `/skills pending`, `/skills diff <id>`, `/skills approve <id>`, `/skills reject <id>`, `/skills approval on\|off`. Veja [Restringindo escritas de skills do agente](/user-guide/features/skills#gating-agent-skill-writes-skillswrite_approval). |
| `/memory [pending\|approve\|reject\|approval]` | Revisa escritas de memória pendentes preparadas pelo gate de aprovação de escrita (`memory.write_approval`) e alterna o gate. Veja [Controlando escritas de memória](/user-guide/features/memory#controlling-memory-writes-write_approval). |
| `/bundles` | Lista os bundles de skills configurados — aliases de slash `/<name>` que pré-carregam várias skills de uma vez. Configure em `bundles:` em `~/.hermes/config.yaml`. Veja [Skill Bundles](/user-guide/features/skills#skill-bundles). |
| `/learn <what to learn from>` | Destila uma skill reutilizável a partir de qualquer coisa que você descrever — um diretório, uma URL, o fluxo de trabalho que você acabou de mostrar ao agente, ou notas coladas. Aberto: o agente reúne as fontes com suas próprias ferramentas e escreve um `SKILL.md` seguindo os padrões internos de autoria. Funciona na CLI, no gateway de mensagens, na TUI e na página de Skills do dashboard. |
| `/cron` | Gerencia tarefas agendadas (list, add/create, edit, pause, resume, run, remove) |
| `/suggestions [accept\|dismiss N\|catalog\|clear]` (alias: `/suggest`) | Revisa automações sugeridas. Use `/suggestions` para listar sugestões pendentes, `/suggestions accept <id>` para criar a automação proposta, `/suggestions dismiss <id>` para rejeitar uma, `/suggestions catalog` para adicionar automações iniciais selecionadas, e `/suggestions clear` para limpar registros de sugestões resolvidas. Jobs aceitos preservam a superfície atual como origem de entrega. |
| `/blueprint [name] [slot=value ...]` (alias: `/bp`) | Configura uma automação a partir de um template de blueprint. `/blueprint` sozinho lista o catálogo; `/blueprint <name>` inicia um fluxo guiado de preenchimento de slots no próximo turno do agente; `/blueprint <name> slot=value ...` cria o job diretamente. |
| `/curator` | Manutenção de skills em segundo plano — `status`, `run`, `pin`, `archive`. Veja [Curator](/user-guide/features/curator). |
| `/kanban <action>` | Opera o quadro de colaboração multiperfil e multiprojeto sem saber do chat. A superfície completa do `hermes kanban` está disponível: `/kanban list`, `/kanban show t_abc`, `/kanban create "title" --assignee X`, `/kanban comment t_abc "text"`, `/kanban unblock t_abc`, `/kanban dispatch`, etc. Suporte a múltiplos quadros incluído: `/kanban boards list`, `/kanban boards create <slug>`, `/kanban boards switch <slug>`, `/kanban --board <slug> <action>`. Veja [Slash command do Kanban](/user-guide/features/kanban#kanban-slash-command). |
| `/reload-mcp` (alias: `/reload_mcp`) | Recarrega os servidores MCP a partir do config.yaml |
| `/reload-skills` (alias: `/reload_skills`) | Rescaneia `~/.hermes/skills/` por skills recém-instaladas ou removidas |
| `/reload` | Recarrega as variáveis do `.env` na sessão em execução (capta novas chaves de API sem reiniciar) |
| `/plugins` | Lista os plugins instalados e seu status |
| `/pet [list\|<slug>]` | Alterna ou adota um mascote [petdex](/user-guide/features/pets). `/pet` alterna o painel, `/pet list` mostra os pets instalados, `/pet <slug>` adota um específico. |
| `/hatch <description>` (alias: `/generate-pet`) | Gera um novo pet petdex a partir de uma descrição em texto, usando o backend de imagem configurado (OpenRouter / Nous Portal). Veja [Pets](/user-guide/features/pets). |

### Informações

| Comando | Descrição |
|---------|-------------|
| `/help` | Mostra esta mensagem de ajuda |
| `/version` | Mostra a versão do Hermes Agent, build e informações de ambiente. |
| `/usage` | Mostra o uso de tokens, detalhamento de custo, duração da sessão e — quando disponível pelo provedor ativo — uma seção de **Limites da conta** com cota restante / créditos / uso do plano obtidos em tempo real da API do provedor. |
| `/credits` | Mostra seu saldo de créditos Nous e um link de handoff para recarga. |
| `/billing` | Fluxo de Gastos Remotos da CLI para a Nous — veja saldo, compre créditos e gerencie recarga automática / limites mensais. |
| `/insights` | Mostra insights de uso e análises (últimos 30 dias) |
| `/platforms` (alias: `/gateway`) | Mostra o status da plataforma de gateway/mensagens (visão resumida somente na CLI). |
| `/paste` | Anexa uma imagem da área de transferência |
| `/copy [number]` | Copia a última resposta do assistente para a área de transferência (ou a N-ésima a partir do fim com um número). Somente CLI. |
| `/image <path>` | Anexa um arquivo de imagem local para seu próximo prompt. |
| `/debug` | Envia um relatório de depuração (informações do sistema + logs) e obtenha links compartilháveis. Também disponível em mensagens. |
| `/profile` | Mostra o nome do perfil ativo e o diretório home |

### Saída

| Comando | Descrição |
|---------|-------------|
| `/quit` | Sai da CLI (também: `/exit`). |

### Slash commands dinâmicos da CLI {#dynamic-cli-slash-commands}

| Comando | Descrição |
|---------|-------------|
| `/<skill-name>` | Carrega qualquer skill instalada como um comando sob demanda. Exemplo: `/gif-search`, `/github-pr-workflow`, `/excalidraw`. |
| `/skills ...` | Busca, navega, inspeciona, instala, audita, publica e configura skills a partir de registros e do catálogo oficial de skills opcionais. |

### Comandos rápidos

Comandos rápidos definidos pelo usuário mapeiam um slash command curto para um comando de shell ou outro slash command. Configure-os em `~/.hermes/config.yaml`:

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

Então digite `/status`, `/deploy` ou `/inbox` na CLI ou em uma plataforma de mensagens. Comandos rápidos são resolvidos no momento do despacho e podem não aparecer em toda tabela embutida de autocompletar/ajuda.

Atalhos de prompt apenas-texto não são suportados como comandos rápidos. Coloque prompts reutilizáveis mais longos em uma skill, ou use `type: alias` para apontar para um slash command existente.

### Aliases de modelo customizados {#custom-model-aliases}

Defina seus próprios nomes curtos para modelos que você usa com frequência, depois acesse-os com `/model <alias>` na CLI ou em qualquer plataforma de mensagens. Os aliases funcionam de forma idêntica em ambas, com mudanças apenas de sessão (padrão) e `--global`.

Dois formatos de configuração são suportados:

**Forma completa** — fixa um modelo, provedor e opcionalmente uma URL base exatos. Coloque isso em `~/.hermes/config.yaml`:

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

**Forma curta** — `provider/model` em uma única string. Defina a partir do shell sem editar YAML:

```bash
hermes config set model.aliases.fav anthropic/claude-opus-4.6
hermes config set model.aliases.grok x-ai/grok-4
```

Depois no chat:

```
/model fav            # apenas para esta sessão
/model grok --global  # também persiste a mudança do modelo atual no config.yaml
```

Aliases do usuário têm precedência sobre nomes curtos embutidos, então nomear um alias `sonnet`, `kimi`, `opus`, etc. vai sobrescrever o embutido. Nomes de alias não diferenciam maiúsculas/minúsculas.

### Resolução de alias {#alias-resolution}

Os comandos suportam correspondência por prefixo: digitar `/h` resolve para `/help`, `/mod` resolve para `/model`. Quando um prefixo é ambíguo (corresponde a vários comandos), a primeira correspondência na ordem do registro vence. Nomes completos de comandos e aliases registrados sempre têm prioridade sobre correspondências por prefixo.

## Slash commands de mensagens {#messaging-slash-commands}

O gateway de mensagens suporta os seguintes comandos embutidos dentro de chats do Telegram, Discord, Slack, WhatsApp, Signal, Email, Home Assistant e Teams:

| Comando | Descrição |
|---------|-------------|
| `/start` | Comando de protocolo de plataforma. Muitas plataformas de chat (Telegram, Discord, …) enviam `/start` automaticamente na primeira vez que um usuário abre uma conversa com o bot. O Hermes reconhece o ping silenciosamente — sem resposta do agente, sem consumo de sessão — de modo que handshakes de primeiro contato não desperdicem um turno. Você também pode enviá-lo explicitamente para confirmar que o gateway está acessível. |
| `/new [name]` (alias: `/reset`) | Inicia uma nova sessão (novo ID de sessão + histórico). O `[name]` opcional define o título inicial da sessão. Acrescente `now`, `--yes` ou `-y` para pular o modal de confirmação — ex.: `/reset now`, `/new --yes my-experiment`. |
| `/status` | Mostra informações da sessão, seguidas por um bloco local de **Resumo da sessão** (contagens recentes de turnos, principais ferramentas usadas, arquivos tocados, último prompt + resposta). |
| `/stop` | Mata todos os processos em segundo plano em execução e interrompe o agente em execução. |
| `/model [provider:model]` | Mostra ou altera o modelo. Suporta troca de provedor (`/model zai:glm-5`), endpoints customizados (`/model custom:model`), provedores customizados nomeados (`/model custom:local:qwen`), detecção automática (`/model custom`), e aliases definidos pelo usuário (`/model fav`, `/model grok` — veja [Aliases de modelo customizados](#custom-model-aliases)). Use `--global` para persistir a mudança no config.yaml. **Observação:** `/model` só pode alternar entre provedores já configurados. Para adicionar um novo provedor ou configurar chaves de API, use `hermes model` no seu terminal (fora da sessão de chat). **Observação de custo:** uma troca de modelo no meio da sessão reinicia o cache de prompt (a chave de cache inclui o modelo), então a próxima mensagem relê toda a conversa ao preço total de entrada. |
| `/codex-runtime [auto\|codex_app_server\|on\|off]` | Alterna o [runtime opcional do Codex app-server](../user-guide/features/codex-app-server-runtime). Persiste em `model.openai_runtime` no config.yaml e descarta o agente em cache para que a próxima mensagem use o novo runtime. Efetivo na próxima sessão. |
| `/personality [name]` | Define uma sobreposição de personalidade para a sessão. |
| `/fast [normal\|fast\|status]` | Alterna o modo rápido — OpenAI Priority Processing / Anthropic Fast Mode. |
| `/retry` | Tenta novamente a última mensagem. |
| `/undo` | Remove a última troca. |
| `/sethome` (alias: `/set-home`) | Marca o chat atual como o canal home da plataforma para entregas. |
| `/compress [here [N] \| focus topic]` | Comprime manualmente o contexto da conversa. `/compress here [N]` mantém as N trocas mais recentes (padrão 2) literalmente e resume o restante. Um tópico de foco (`focus topic`) restringe o que um resumo completo preserva. |
| `/topic [off\|help\|session-id]` | **Somente DM do Telegram.** Gerencia o modo de tópico multissessão gerenciado pelo usuário. `/topic` ativa ou mostra o status; `/topic off` desativa e limpa as vinculações; `/topic help` mostra o uso; `/topic <session-id>` dentro de um tópico restaura uma sessão anterior. Veja [Modo DM multissessão](/user-guide/messaging/telegram#multi-session-dm-mode-topic). |
| `/title [name]` | Define ou mostra o título da sessão. |
| `/resume [name]` | Retoma uma sessão previamente nomeada. |
| `/usage` | Mostra o uso de tokens, detalhamento de custo estimado (entrada/saída), estado da janela de contexto, duração da sessão e — quando disponível pelo provedor ativo — uma seção de **Limites da conta** com cota restante / créditos obtidos em tempo real da API do provedor. |
| `/credits` | Mostra seu saldo de créditos Nous e um link de recarga que abre a página de faturamento do portal em um navegador. |
| `/insights [days]` | Mostra análises de uso. |
| `/reasoning [level\|show\|hide]` | Altera o esforço de raciocínio ou alterna a exibição de raciocínio. |
| `/voice [on\|off\|tts\|join\|channel\|leave\|status]` | Controla respostas faladas no chat. `join`/`channel`/`leave` gerenciam o modo de canal de voz do Discord. |
| `/rollback [number]` | Lista ou restaura checkpoints do sistema de arquivos. |
| `/background <prompt>` | Executa um prompt em uma sessão separada em segundo plano. Os resultados são entregues de volta ao mesmo chat quando a tarefa termina. Veja [Sessões em Segundo Plano de Mensagens](/user-guide/messaging/#background-sessions). |
| `/queue <prompt>` (alias: `/q`) | Enfileira um prompt para o próximo turno sem interromper o atual. |
| `/steer <prompt>` | Injeta uma mensagem após a próxima chamada de ferramenta sem interromper — o modelo a capta na próxima iteração em vez de como um novo turno. |
| `/goal <text>` | Define uma meta permanente pela qual o Hermes trabalha ao longo dos turnos — nossa versão do loop Ralph. Um modelo juiz verifica após cada turno; se não concluída, o Hermes continua automaticamente até que esteja, você pause/limpe, ou o orçamento de turnos (padrão 20) seja atingido. Subcomandos: `/goal status`, `/goal pause`, `/goal resume`, `/goal clear`. Seguro de executar no meio do agente para status/pause/clear; definir uma nova meta requer `/stop` primeiro. Veja [Metas Persistentes](/user-guide/features/goals). |
| `/footer [on\|off\|status]` | Alterna o rodapé de metadados de runtime nas respostas finais (mostra modelo, % de contexto e cwd). |
| `/curator [status\|run\|pin\|archive]` | Controles de manutenção de skills em segundo plano. |
| `/suggestions [accept\|dismiss N\|catalog\|clear]` | Revisa automações sugeridas direto no chat. `/suggestions` lista sugestões pendentes, `catalog` adiciona automações iniciais selecionadas, e `clear` limpa registros de sugestões resolvidas. Sugestões aceitas mantêm este chat/thread como origem de entrega do job. |
| `/blueprint [name] [slot=value ...]` | Navega por blueprints de cron, inicia uma conversa guiada de preenchimento de slots, ou cria um job de blueprint diretamente. Jobs criados diretamente entregam de volta ao chat/thread atual. |
| `/memory [pending\|approve\|reject\|approval]` | Revisa escritas de memória pendentes preparadas pelo gate de aprovação de escrita (`memory.write_approval`) — aprove ou rejeite direto no chat — e alterne o gate com `/memory approval on\|off`. Veja [Controlando escritas de memória](/user-guide/features/memory#controlling-memory-writes-write_approval). |
| `/skills [pending\|approve\|reject\|diff\|approval]` | Revisa escritas de **skills** pendentes preparadas pelo gate de aprovação de escrita (`skills.write_approval`). Mostra um resumo de uma linha por escrita preparada; `/skills diff <id>` é truncado para o chat — leia o diff completo na CLI ou em `~/.hermes/pending/skills/<id>.json`. Só aparece quando o gate está ativo (ou há escritas preparadas restantes); busca/instalação permanecem somente na CLI. |
| `/kanban <action>` | Opera o quadro de colaboração multiperfil e multiprojeto a partir do chat — superfície de argumentos idêntica à da CLI. Ignora a guarda de agente em execução, então `/kanban unblock t_abc`, `/kanban comment t_abc "…"`, `/kanban list --mine`, `/kanban boards switch <slug>`, etc. funcionam no meio de um turno. `/kanban create …` inscreve automaticamente o chat de origem nos eventos de terminal da nova tarefa. Veja [Slash command do Kanban](/user-guide/features/kanban#kanban-slash-command). |
| `/platform <list\|pause\|resume> [name]` | Opera uma plataforma de gateway em execução direto do chat. `/platform list` mostra todos os adaptadores e seu estado (em execução, pausado-pelo-breaker, pausado-manualmente); `/platform pause <name>` para de despachar novas mensagens para esse adaptador sem descarregá-lo; `/platform resume <name>` o reativa e limpa um circuit breaker acionado uma vez que o upstream esteja saudável. |
| `/reload-mcp` (alias: `/reload_mcp`) | Recarrega os servidores MCP a partir da configuração. |
| `/yolo` | Alterna o modo YOLO — pula todos os prompts de aprovação de comandos perigosos. |
| `/commands [page]` | Navega por todos os comandos e skills (paginado). |
| `/approve [session\|always]` | Aprova e executa um comando perigoso pendente. `session` aprova apenas para esta sessão; `always` adiciona à allowlist permanente. |
| `/deny` | Rejeita um comando perigoso pendente. |
| `/update` | Atualiza o Hermes Agent para a versão mais recente. |
| `/restart` | Reinicia o gateway de forma controlada após drenar as execuções ativas. Quando o gateway volta a ficar online, envia uma confirmação para o chat/thread do solicitante. |
| `/debug` | Envia um relatório de depuração (informações do sistema + logs) e obtenha links compartilháveis. |
| `/help` | Mostra a ajuda de mensagens. |
| `/<skill-name>` | Invoca qualquer skill instalada pelo nome. |

## Observações {#notes}

- `/skin`, `/snapshot`, `/reload`, `/tools`, `/toolsets`, `/browser`, `/config`, `/cron`, `/platforms`, `/paste`, `/image`, `/statusbar`, `/plugins`, `/busy`, `/indicator`, `/redraw`, `/clear`, `/history`, `/save`, `/copy`, `/handoff`, `/billing` e `/quit` são comandos **somente CLI**.
- `/skills` é **somente CLI para busca/navegação/instalação**; seus subcomandos de revisão do gate de aprovação de escrita (`pending`, `approve`, `reject`, `diff`, `approval`) também funcionam em plataformas de mensagens quando `skills.write_approval` está ativo. `/memory` funciona em **ambas** as superfícies.
- `/verbose` é **somente CLI por padrão**, mas pode ser habilitado para plataformas de mensagens definindo `display.tool_progress_command: true` no `config.yaml`. Quando habilitado, alterna o modo `display.tool_progress` e salva na configuração.
- `/sethome`, `/update`, `/restart`, `/approve`, `/deny`, `/topic`, `/platform` e `/commands` são comandos **somente de mensagens**.
- `/status`, `/version`, `/background`, `/queue`, `/steer`, `/voice`, `/reload-mcp`, `/reload-skills`, `/rollback`, `/debug`, `/fast`, `/footer`, `/curator`, `/kanban`, `/credits`, `/suggestions`, `/blueprint`, `/learn`, `/sessions` e `/yolo` funcionam em **ambos** a CLI e o gateway de mensagens.
- `/voice join`, `/voice channel` e `/voice leave` só fazem sentido no Discord.
- Na TUI, `/sessions` mostra sessões ativas no processo TUI atual. Use `/resume [name]` ou `hermes --tui --resume <id-or-title>` para transcrições salvas ou encerradas.

## Prompts de confirmação para comandos destrutivos {#confirmation-prompts-for-destructive-commands}

A CLI pede confirmação antes de executar slash commands que descartam estado de sessão não salvo. O conjunto destrutivo atual é:

| Comando | O que destrói |
|---------|------------------|
| `/clear` | Limpa a tela e inicia uma nova sessão — o ID de sessão atual e o histórico em memória se perdem. |
| `/new` / `/reset` | Inicia uma nova sessão (novo ID de sessão + histórico vazio). |
| `/undo` | Remove a última troca usuário/assistente do histórico. |
| `/exit --delete` / `/quit --delete` | Sai **e** exclui permanentemente o histórico SQLite e as transcrições em disco da sessão atual. |

Para cada um desses, a CLI abre um modal de três opções: **Aprovar uma vez** (prosseguir desta vez), **Aprovar sempre** (prosseguir e persistir `approvals.destructive_slash_confirm: false` para que futuros comandos destrutivos executem sem pedir confirmação), ou **Cancelar**.

**Pular inline:** acrescente `now`, `--yes` ou `-y` para contornar o modal em uma única invocação — ex.: `/reset now`, `/new --yes my-session`, `/clear -y`, `/undo -y`. Útil quando o modal não renderiza corretamente no seu terminal (veja [issue #30768](https://github.com/NousResearch/hermes-agent/issues/30768) para o PowerShell nativo do Windows) ou ao automatizar a CLI via script.

Defina `approvals.destructive_slash_confirm: false` em `~/.hermes/config.yaml` para desativar os prompts globalmente; defina de volta como `true` para reativá-los. Veja [Segurança — Confirmação de slash command destrutivo](../user-guide/security.md#dangerous-command-approval) para contexto.
