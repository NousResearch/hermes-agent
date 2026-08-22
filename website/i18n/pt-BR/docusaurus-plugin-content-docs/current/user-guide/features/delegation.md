---
sidebar_position: 7
title: "Delegação de subagentes"
description: "Crie agentes filhos isolados para fluxos de trabalho paralelos com delegate_task"
---

# Delegação de subagentes

A ferramenta `delegate_task` cria instâncias filhas de AIAgent com contexto isolado, acesso herdado a ferramentas e sessões de terminal próprias. Cada filho recebe uma conversa nova e trabalha de forma independente — apenas o resumo final entra no contexto do pai.

Chamadas de modelo de nível superior rodam em segundo plano automaticamente. O Hermes retorna um identificador imediatamente para a conversa continuar e publica o resultado depois como uma nova mensagem. Um subagente orquestrador aguarda seus próprios workers para sintetizar os resultados antes de retornar.

## Tarefa única {#single-task}

```python
delegate_task(
    goal="Debug why tests fail",
    context="Error: assertion in test_foo.py line 42"
)
```

## Lote paralelo {#parallel-batch}

Até 3 subagentes concorrentes por padrão (configurável, sem teto rígido):

```python
delegate_task(tasks=[
    {"goal": "Research topic A", "context": "Focus on recent primary sources"},
    {"goal": "Research topic B", "context": "Compare the leading explanations"},
    {"goal": "Fix the build", "context": "Project root: /home/user/project"}
])
```

## Como funciona o contexto do subagente {#how-subagent-context-works}

:::warning Crítico: subagentes não sabem nada
Subagentes começam com uma **conversa completamente nova**. Eles não têm conhecimento do histórico de conversa do pai, de chamadas de ferramentas anteriores nem de nada discutido antes da delegação. O único contexto do subagente vem dos campos `goal` e `context` que o agente pai preenche ao chamar `delegate_task`.
:::

Isso significa que o agente pai deve passar **tudo** que o subagente precisa na chamada:

```python
# BAD - subagent has no idea what "the error" is
delegate_task(goal="Fix the error")

# GOOD - subagent has all context it needs
delegate_task(
    goal="Fix the TypeError in api/handlers.py",
    context="""The file api/handlers.py has a TypeError on line 47:
    'NoneType' object has no attribute 'get'.
    The function process_request() receives a dict from parse_body(),
    but parse_body() returns None when Content-Type is missing.
    The project is at /home/user/myproject and uses Python 3.11."""
)
```

O subagente recebe um prompt de sistema focado construído a partir do seu `goal` e `context`, instruindo-o a concluir a tarefa e fornecer um resumo estruturado do que fez, do que encontrou, de quaisquer arquivos modificados e de quaisquer problemas encontrados.

## Exemplos práticos {#practical-examples}

### Pesquisa paralela {#parallel-research}

Pesquise vários tópicos simultaneamente e colete resumos:

```python
delegate_task(tasks=[
    {
        "goal": "Research the current state of WebAssembly in 2025",
        "context": "Focus on: browser support, non-browser runtimes, language support"
    },
    {
        "goal": "Research the current state of RISC-V adoption in 2025",
        "context": "Focus on: server chips, embedded systems, software ecosystem"
    },
    {
        "goal": "Research quantum computing progress in 2025",
        "context": "Focus on: error correction breakthroughs, practical applications, key players"
    }
])
```

### Revisão de código + correção {#code-review-fix}

Delegue um fluxo de revisão e correção para um contexto novo:

```python
delegate_task(
    goal="Review the authentication module for security issues and fix any found",
    context="""Project at /home/user/webapp.
    Auth module files: src/auth/login.py, src/auth/jwt.py, src/auth/middleware.py.
    The project uses Flask, PyJWT, and bcrypt.
    Focus on: SQL injection, JWT validation, password handling, session management.
    Fix any issues found and run the test suite (pytest tests/auth/)."""
)
```

### Refatoração em vários arquivos {#multi-file-refactoring}

Delegue uma refatoração grande que inundaria o contexto do pai:

```python
delegate_task(
    goal="Refactor all Python files in src/ to replace print() with proper logging",
    context="""Project at /home/user/myproject.
    Use the 'logging' module with logger = logging.getLogger(__name__).
    Replace print() calls with appropriate log levels:
    - print(f"Error: ...") -> logger.error(...)
    - print(f"Warning: ...") -> logger.warning(...)
    - print(f"Debug: ...") -> logger.debug(...)
    - Other prints -> logger.info(...)
    Don't change print() in test files or CLI output.
    Run pytest after to verify nothing broke."""
)
```

## Detalhes do modo em lote {#batch-mode-details}

Quando um agente de nível superior fornece um array `tasks`, o Hermes retorna um identificador de segundo plano, executa os subagentes em paralelo e publica um resultado consolidado depois que todos os filhos terminam. Um subagente orquestrador aguarda seu lote no turno atual para sintetizar os resultados.

- **Concorrência máxima:** 3 tarefas por padrão (configurável via `delegation.max_concurrent_children` ou a variável de ambiente `DELEGATION_MAX_CONCURRENT_CHILDREN`; mínimo 1, sem teto rígido). Lotes maiores que o limite retornam erro de ferramenta em vez de serem truncados silenciosamente.
- **Pool de threads:** usa `ThreadPoolExecutor` com o limite de concorrência configurado como máximo de workers
- **Exibição de progresso:** no modo CLI, uma visualização em árvore mostra chamadas de ferramentas de cada subagente em tempo real com linhas de conclusão por tarefa. No gateway, o progresso é agrupado e repassado ao callback de progresso do pai
- **Ordem dos resultados:** resultados são ordenados pelo índice da tarefa para corresponder à ordem de entrada, independentemente da ordem de conclusão
- **Cancelamento:** mensagens de acompanhamento não cancelam um lote em segundo plano de nível superior. `/stop` ou fechar/redefinir a sessão proprietária cancela seus filhos ativos. Filhos orquestradores síncronos ainda seguem o estado de interrupção do pai

Delegação síncrona de tarefa única a partir de um orquestrador roda diretamente, sem overhead do pool de threads.

### Conclusões duráveis em segundo plano {#durable-background-completions}

Quando uma delegação em segundo plano termina, o Hermes armazena seu evento de conclusão no `state.db` do perfil ativo antes de publicá-lo na fila normal de turno novo. Se o Hermes reiniciar após a conclusão mas antes da entrega, o evento pendente é restaurado e roteado pelas mesmas verificações de propriedade. Consumidores concorrentes usam uma reivindicação durável, então apenas o consumidor que aceita com sucesso o turno sintético confirma a entrega; tentativas falhas liberam a reivindicação para nova tentativa.

Isso não retoma a execução do filho após uma falha. Uma delegação cujo processo proprietário desaparece enquanto ainda está em execução é registrada como `unknown`, porque o Hermes não pode provar se seus efeitos colaterais externos ocorreram. Registros pendentes e entregues são limitados e locais ao perfil.

## Substituição de modelo {#model-override}

Você pode configurar um modelo diferente para subagentes via `config.yaml` — útil para delegar tarefas simples a modelos mais baratos/rápidos:

```yaml
# In ~/.hermes/config.yaml
delegation:
  model: "google/gemini-flash-2.0"    # Cheaper model for subagents
  provider: "openrouter"              # Optional: route subagents to a different provider
```

Se omitido, subagentes usam o mesmo modelo do pai.

### Estratégia de custo: planner frontier, workers baratos {#cost-strategy-frontier-planner-inexpensive-workers}

Decompor um problema em subtarefas bem especificadas exige julgamento de nível frontier; executar uma subtarefa que já vem com um goal claro, contexto completo e um contrato de output geralmente não. Enquanto isso, os filhos são onde os tokens vão — um lote paralelo de subagentes tipicamente queima a grande maioria dos tokens totais de uma execução, então o modelo worker é onde o custo realmente mora. Pinar `delegation.model` em um modelo barato enquanto sua sessão principal fica em um modelo frontier mantém a qualidade do planejamento onde importa e corta gasto onde o volume está:

```yaml
# ~/.hermes/config.yaml
model:
  default: "your-frontier-model"     # parent (planner) stays on the frontier model
delegation:
  model: "your-inexpensive-model"    # all delegate_task children run on this
  provider: "openrouter"             # optional: route children to a different provider
```

Ordem de resolução: `delegation.base_url` (endpoint direto) tem precedência, depois `delegation.provider` (bundle completo de credenciais resolvido via o sistema de providers em runtime), e quando nenhum está definido os filhos herdam o provider e as credenciais do pai; `delegation.model` aplica em todos os casos, e quando está vazio os filhos herdam o modelo do pai.

Note que o pin é global: `delegate_task` não tem parâmetro de modelo por tarefa, então todo filho em um lote roda no modelo de delegação configurado. Para subtarefas sensíveis a qualidade que precisam de um modelo mais forte, ou deixe `delegation.model` indefinido para aquela sessão ou entregue a tarefa ao [quadro kanban](kanban.md#per-task-model-override), que de fato suporta override de modelo por tarefa.

## Acesso herdado a ferramentas {#inherited-tool-access}

`delegate_task` não aceita um parâmetro `toolsets` voltado ao modelo. Cada subagente herda os toolsets habilitados do pai, para que o modelo não possa conceder a um filho capacidades que o pai não tem. Configure as ferramentas do pai antes de iniciar a conversa se o trabalho delegado precisar de capacidades adicionais.

Certas ferramentas são bloqueadas para subagentes mesmo quando o pai as possui:
- `delegate_task` — bloqueado para subagentes folha (padrão). Mantido para filhos com `role="orchestrator"`, limitado por `max_spawn_depth` — veja [Limite de profundidade e orquestração aninhada](#depth-limit-and-nested-orchestration) abaixo.
- `clarify` — subagentes não podem interagir com o usuário
- `memory` — sem gravações em memória persistente compartilhada
- `send_message` — sem efeitos colaterais entre plataformas
- `cronjob` — sem agendar mais trabalho em nome do pai

Ambos os papéis mantêm `execute_code` (chamada programática de ferramentas) para que filhos possam agrupar trabalho mecânico.

## Máximo de iterações {#max-iterations}

Cada subagente tem um limite de iterações (padrão: 50) que controla quantos turnos de chamada de ferramentas pode fazer:

```python
delegate_task(
    goal="Quick file check",
    context="Check if /etc/nginx/nginx.conf exists and print its first 10 lines",
    max_iterations=10  # Simple task, don't need many turns
)
```

## Timeout do filho {#child-timeout}

Por padrão **não há timeout de relógio** em subagentes. Filhos falham apenas pelo que estão realmente fazendo — erros de API, erros de ferramenta ou atingir o orçamento de iterações — nunca por um cronômetro de delegação. Versões anteriores tinham um limite rígido (300s, depois 600s), que continuava matando filhos legitimamente ocupados no meio da tarefa: revisões profundas de código, grandes fan-outs de pesquisa e modelos de raciocínio lentos rotineiramente precisam de mais de 10 minutos enquanto fazem progresso constante o tempo todo.

Filhos genuinamente travados ainda são detectados: o monitor de staleness de heartbeat para de atualizar a atividade do pai quando um filho não faz progresso (sem chamadas de API, sem inícios de ferramenta e sem ticks de timestamp de atividade), permitindo que o timeout de inatividade do gateway dispare em um worker realmente emperrado. Uma espera de modelo em andamento ainda conta como progresso — subagentes atualizam o relógio de atividade enquanto aguardam o provedor, então uma conclusão local lenta / com prefill longo não é tratada como parada.

Se você quiser um limite rígido mesmo assim (por exemplo, controle de custo em delegação não supervisionada acionada por cron), ative por instalação:

```yaml
delegation:
  child_timeout_seconds: 0     # default: 0 = no timeout
  # child_timeout_seconds: 1800  # opt-in hard cap (floor 30s)
```

Um valor positivo impõe um limite rígido de relógio em cada filho; `0` ou um valor negativo desativa.

Quando um limite configurado dispara, o resultado do filho traz metadados estruturados de timeout junto com a mensagem de erro para que pais e hooks distingam um kill por cronômetro de outras falhas sem analisar texto: `timeout_seconds` (o limite configurado), `timed_out_after_seconds` (relógio real) e `timeout_phase` (`before_first_llm_call` quando o filho nunca chegou à primeira requisição, `after_llm_calls` caso contrário). Os três são `null` em erros que não são timeout.

:::tip Dump de diagnóstico em timeout com zero chamadas
Com um limite rígido configurado, se um subagente expira tendo feito **zero** chamadas de API (geralmente: provedor inacessível, falha de autenticação ou rejeição de schema de ferramenta), `delegate_task` grava um diagnóstico estruturado em `~/.hermes/logs/subagent-timeout-<session>-<timestamp>.log` contendo snapshot de config do subagente, trace de resolução de credenciais, quaisquer mensagens de erro iniciais e stack traces de **todas** as threads vivas (não só a do filho) — um filho parado aguardando uma thread auxiliar aninhada é indistinguível de um provedor lento sem o quadro completo.
:::

## Detecção de parada para subagentes em segundo plano {#stall-detection-for-background-subagents}

Delegações em segundo plano (`delegate_task(background=true)`) são monitoradas por um
**monitor de parada baseado em progresso** — ativo por padrão, zero config. Diferente de um
timeout de relógio, nunca toca um filho que está progredindo, não importa
quanto tempo rode.

O monitor amostra os sinais de progresso de cada filho destacado — contagem de chamadas de API,
ferramenta atual e timestamp da última atividade (que avança a **cada token
streamado**, transição de ferramenta e limite de chamada de API, então um filho no meio de uma
resposta longa sempre conta como vivo):

1. **Filhos em progresso nunca são tocados.** Qualquer sinal avançando reinicia
   o relógio.
2. Um filho cujo progresso está completamente congelado além do limiar de staleness
   (450s ocioso, 1200s dentro de uma ferramenta — comandos de terminal e buscas web
   legitimamente lentos recebem o teto maior) é **interrompido** e
   recebe uma janela de graça de 120s. Um filho que desfaz a tempo entrega seus
   resultados parciais pelo caminho normal de conclusão.
3. Um filho que nunca retorna é finalizado à força com um evento terminal de conclusão `stalled`,
   para que a sessão proprietária ouça um desfecho em vez de
   ficar em silêncio, e o slot assíncrono libera para novo trabalho.

O evento `stalled` traz metadados estruturados espelhando os campos de timeout
do caminho síncrono: `stalled_after_quiet_seconds`, `stall_threshold_seconds`,
`stall_phase` (`idle` / `in_tool`) e `stall_grace_seconds`.

Isso fechou um modo de falha antigo em que um filho em segundo plano emperrado
deixava a sessão parecendo morta até reinício do processo. A causa raiz subjacente
(filhos pendurados na primeira chamada de API após dias de uptime do gateway)
também foi corrigida na raiz: filhos delegados agora executam suas requisições de API
OpenAI-wire inline na própria thread de conversa em vez de uma thread worker
aninhada — a camada onde o emperramento vivia. O monitor de parada permanece
como rede de segurança para qualquer outra coisa.


## Monitorar subagentes em execução (`/agents`) {#monitoring-running-subagents-agents}

A TUI inclui uma sobreposição `/agents` (alias `/tasks`) que transforma fan-out recursivo de `delegate_task` em uma superfície de auditoria de primeira classe:

- Visualização em árvore ao vivo de subagentes em execução e recém-finalizados, agrupados por pai
- Totais por ramo de custo, tokens e arquivos tocados
- Controles de kill e pause — cancele um subagente específico no meio do voo sem interromper os irmãos
- Revisão pós-hoc: percorra o histórico turno a turno de cada subagente mesmo depois que retornou ao pai

A CLI clássica apenas imprime `/agents` como um resumo em texto; a TUI é onde a sobreposição brilha. Veja [TUI — Comandos slash](/user-guide/tui#slash-commands).

Na CLI clássica e em toda plataforma de gateway (Telegram, Discord, Slack, ...),
`/agents` também lista **delegações em segundo plano com atividade ao vivo por filho**,
amostrada diretamente de cada filho em execução:

```
Background delegations: 1 running
- deleg_ab12cd34 · running · research the delegation stall monitor
  - child 1: 4 api calls · in web_search · active 12s ago
  - child 2: 7 api calls · between turns · active 3s ago
```

Uma delegação que o monitor de parada sinalizou aparece como
`stalling · no progress 450s — interrupting`, e filhos saudáveis mas quietos há tempo
mostram seu tempo de quietude para distinguir "lento" de "travado" de relance.

## Direcionar um subagente em execução {#steering-a-running-subagent}

Interromper um filho descarta o trabalho em andamento; muitas vezes você só quer redirecioná-lo.

### Do agente pai (voltado ao modelo) {#from-the-parent-agent-model-facing}

O agente pai orquestra seus próprios filhos em execução com a mesma ferramenta `delegate_task` com que os gerou — sem ferramenta de controle separada:

```json
{"action": "list"}
{"action": "steer", "subagent_id": "sa-0-1a2b3c4d", "message": "focus on pricing instead"}
{"action": "stop",  "subagent_id": "sa-0-1a2b3c4d"}
```

- **`list`** retorna os filhos vivos da conversa: `subagent_id`, goal, status, `running_seconds`, `accepting_steer`, e o caminho da transcrição ao vivo. Ids também voltam na resposta de despacho do spawn como `subagent_ids`.
- **`steer`** enfileira uma correção de rumo em um filho em execução sem pará-lo (semântica de entrega abaixo).
- **`stop`** encerra um filho cedo no próximo limite de iteração; o resultado parcial ainda reentra na conversa como uma mensagem de conclusão normal.

Ações de controle rodam sincronamente in-turn (nunca em background), são escopadas à árvore de spawn do próprio chamador — uma conversa nunca pode ver ou controlar filhos de outra sessão — e nunca consomem o cap de spawn de subagentes por turno, então `stop` continua funcionando mesmo depois que o cap é atingido.

### Do TUI / gateway (voltado à sessão) {#from-the-tui--gateway-session-facing}

`steer_subagent(subagent_id, text)` em `tools/delegate_tool.py` é o espelho do lado de redirecionamento de `interrupt_subagent()`: enfileira texto em um filho vivo pelo mesmo mecanismo que [`/steer`](/reference/slash-commands) — o texto é anexado ao último resultado de ferramenta do filho no próximo limite de iteração, a chamada de ferramenta em andamento nunca é cortada, e o filho o vê como uma mensagem de usuário fora de banda. Hosts programáticos acessam via RPC de gateway `subagent.steer` com escopo de sessão, ao lado de `subagent.interrupt`:

```json
{"method": "subagent.steer", "params": {"session_id": "owning-ui-session", "subagent_id": "sa-0-1a2b3c4d", "text": "focus on pricing instead"}}
```

Ids de subagente vêm de `delegation.status` (ou `list_active_subagents()`) — o mesmo lugar de onde `subagent.interrupt` os obtém. O gateway aceita direcionamento apenas da sessão UI/gateway ao vivo exata que gerou o filho. Identidade de sessão ausente, estrangeira, ambígua ou obsoleta/reciclada é rejeitada; saber um id global de subagente não é autoridade. Chamadores in-process diretos mantêm o contrato de helper sem escopo deliberadamente.

**Enfileirado não é entregue, mas nunca é sucesso sintético.** Uma resposta `"queued"` significa que o texto foi aceito antes do limite de conclusão do filho, não necessariamente que o filho já o viu. Aceitação e conclusão são sincronizadas: ou o filho ainda pode consumir o texto, ou seu texto exato é drenado para o resultado como `pending_steer`. Chamadas após fechamento retornam `"rejected"`. Se um filho aceitou o steer mas já havia produzido a resposta final, a entrada de conclusão que o pai recebe a retém como `missed_steer`, com uma nota anexada ao resumo:

```
[steer did not land — the subagent finished before it could be delivered: focus on pricing instead]
```

Assim o pai (ou o operador que o conduz) distingue um filho direcionado de um que terminou com as instruções antigas, e pode reemitir a orientação como acompanhamento em vez de confiar que chegou.

## Transcrições ao vivo {#live-transcripts}

Cada despacho de `delegate_task` também cria **um log append-only legível por humanos por tarefa** para você (ou o agente pai) acompanhar um subagente trabalhando em tempo real em vez de esperar o resumo consolidado:

```
<hermes_home>/cache/delegation/live/<delegation_id>/task-<n>.log
```

A resposta do despacho inclui os caminhos como `live_transcripts`, e os arquivos são pré-criados no momento do despacho, então funciona imediatamente:

```bash
tail -f ~/.hermes/cache/delegation/live/deleg_ab12cd34/task-0.log
```

Cada linha tem timestamp e mostra o texto do assistente do filho, trechos de thinking, chamadas de ferramenta (`-> tool_name({args})`), resultados de ferramenta e um marcador de status final. Um `manifest.json` no mesmo diretório descreve o lote (goals, contagem de tarefas, status por tarefa). Os logs persistem após a conclusão — também servem como registro operacional de fidelidade total junto ao resumo — e diretórios com mais de 7 dias são podados automaticamente em novos despachos. Por estarem em `cache/delegation`, também são legíveis de backends de terminal remotos (Docker/Modal/SSH).

## Limite de profundidade e orquestração aninhada {#depth-limit-and-nested-orchestration}

Por padrão, a delegação é **plana**: um pai (profundidade 0) gera filhos (profundidade 1), e esses filhos não podem delegar mais. Isso evita delegação recursiva descontrolada.

Para fluxos multiestágio (pesquisa → síntese, ou orquestração paralela sobre subproblemas), um pai pode gerar filhos **orquestradores** que *podem* delegar seus próprios workers:

```python
delegate_task(
    goal="Survey three code review approaches and recommend one",
    role="orchestrator",  # Allows this child to spawn its own workers
    context="...",
)
```

- `role="leaf"` (padrão): filho não pode delegar mais — idêntico ao comportamento de delegação plana.
- `role="orchestrator"`: filho mantém o toolset `delegation`. Limitado por `delegation.max_spawn_depth` (padrão **1** = plano, então `role="orchestrator"` é no-op nos padrões). Aumente `max_spawn_depth` para 2 para permitir que filhos orquestradores gerem netos folha; 3+ para árvores mais profundas. Não há teto superior — custo é o limite prático.
- `delegation.orchestrator_enabled: false`: interruptor global que força todo filho a `leaf` independentemente do parâmetro `role`.

**Aviso de custo:** Com `max_spawn_depth: 3` e `max_concurrent_children: 3`, a árvore pode atingir 3×3×3 = 27 agentes folha concorrentes. Cada nível extra multiplica o gasto — aumente `max_spawn_depth` intencionalmente.

## Ciclo de vida e durabilidade {#lifetime-and-durability}

:::warning Durabilidade de conclusão em segundo plano não é execução durável
Chamadas de `delegate_task` voltadas ao modelo de nível superior rodam em segundo plano automaticamente onde a sessão suporta entrega posterior. O Hermes retorna um identificador imediatamente, e o resultado reentra na conversa depois que o filho ou lote termina. Subagentes orquestradores aguardam seus workers no turno atual porque devem sintetizar esses resultados antes de retornar. Endpoints request/response sem estado caem para execução síncrona quando não podem entregar um resultado destacado depois.

- Mensagens de acompanhamento normais não cancelam filhos em segundo plano. `/stop` cancela delegações em segundo plano em execução, e fechar ou redefinir a sessão proprietária descarta seus filhos ativos.
- Fechar/redefinir sessão explicitamente interrompe os filhos em segundo plano dessa sessão. Fechar um visualizador TUI de uma sessão do gateway não mata o trabalho do gateway.
- Reinício do processo Hermes **não** retoma um filho em execução. Sua tentativa vira `unknown` porque o Hermes não pode provar quais efeitos colaterais ocorreram.
- Um filho que concluiu antes do reinício mas cujo resultado não foi entregue é restaurado e roteado de volta pelas verificações normais da sessão proprietária.
- Filhos cancelados retornam um resultado estruturado (`status="interrupted"`, `exit_reason="interrupted"`), mas como o pai também foi interrompido, esse resultado muitas vezes nunca chega a uma resposta visível ao usuário.

Para **execução durável** que deve sobreviver ao fechamento de sessão ou reinício de processo, use:

- `cronjob` (action=`create`) — agenda uma execução de agente separada; imune a interrupções do turno do pai.
- `terminal(background=True, notify_on_complete=True)` — comandos shell de longa duração que continuam rodando enquanto o agente faz outras coisas.
:::

## Propriedades principais {#key-properties}

- Cada subagente recebe **sua própria sessão de terminal** (separada do pai)
- Subagentes herdam os toolsets habilitados do pai; o modelo não pode selecioná-los ou ampliá-los por chamada
- **Delegação aninhada é opt-in** — apenas filhos com `role="orchestrator"` podem delegar mais, e só quando `max_spawn_depth` é elevado do padrão 1 (plano). Desative globalmente com `orchestrator_enabled: false`.
- Subagentes folha **não podem** chamar: `delegate_task`, `clarify`, `memory`, `send_message`, `cronjob`. Subagentes orquestradores mantêm `delegate_task` mas conservam os outros bloqueios. Ambos os papéis mantêm `execute_code` (chamada programática de ferramentas) para que filhos agrupem trabalho mecânico em vez de queimar iterações de raciocínio.
- **Cancelamento segue propriedade** — `/stop` ou fechar/redefinir a sessão proprietária cancela seus filhos em segundo plano; descendentes síncronos sob orquestradores seguem o estado de interrupção do pai
- Apenas o resumo final entra no contexto do pai, mantendo uso de tokens eficiente
- Subagentes herdam a **chave de API, configuração de provedor e pool de credenciais** do pai (permitindo rotação de chave em rate limits)

## Isolamento de worktree {#worktree-isolation}

Por padrão, subagentes compartilham o diretório de trabalho do pai — adequado para
pesquisa e trabalho pesado em leitura, mas filhos paralelos editando o mesmo repo
podem colidir. Defina `delegation.worktree_isolation: true` para dar a cada filho
seu próprio git worktree, ramificado do `HEAD` atual do repo (inspirado no
`--subagent-worktree-isolation` do Muse Code):

```yaml
delegation:
  worktree_isolation: true   # default: false
```

Com isolamento ligado:

- Cada filho inicia seu terminal em `<repo>/.worktrees/subagent-<id>` no seu
  próprio branch `hermes-subagent/subagent-<id>`, e a mensagem de goal diz para
  trabalhar e commitar ali.
- O checkout do pai permanece intacto; filhos não podem pisar nas edições uns
  dos outros.
- Quando um filho termina, sua entrada de resultado ganha um campo `worktree`
  reportando `path`, `branch`, `commits` (à frente da base) e `dirty`. O pai
  revisa ou faz merge de cada branch (`git log <branch>`, `git merge <branch>`).
- Um worktree deixado **sem commits e com árvore limpa é podado automaticamente**
  (`pruned: true`); qualquer um que retenha trabalho é mantido.
- A poda exige prova. Se um probe de inspeção git falhar — ou a finalização
  em si der erro — o worktree e o branch são mantidos e a entrada carrega
  `inspection_failed: true` mais uma `note` — `commits`/`dirty` passam a ser
  defaults, não medições, então inspecione o worktree em vez de assumir que
  o filho não produziu nada.

Escopo: opt-in, só git, e só backend de terminal local. Num diretório que não é
git, em backends docker/ssh/modal, ou se a criação do worktree falhar, a
configuração degrada silenciosamente para o comportamento atual de workspace
compartilhado — nunca um erro.

## Delegação vs execute_code {#delegation-vs-execute_code}

| Fator | delegate_task | execute_code |
|--------|--------------|-------------|
| **Raciocínio** | Loop completo de raciocínio LLM | Apenas execução de código Python |
| **Contexto** | Conversa isolada nova | Sem conversa, apenas script |
| **Acesso a ferramentas** | Todas as ferramentas não bloqueadas com raciocínio | 7 ferramentas via RPC, sem raciocínio |
| **Paralelismo** | 3 subagentes concorrentes por padrão (configurável) | Script único |
| **Melhor para** | Tarefas complexas que precisam de julgamento | Pipelines mecânicos multi-etapa |
| **Custo de tokens** | Maior (loop LLM completo) | Menor (só stdout retornado) |
| **Interação com usuário** | Nenhuma (subagentes não podem clarificar) | Nenhuma |

**Regra prática:** Use `delegate_task` quando a subtarefa exige raciocínio, julgamento ou resolução de problemas multi-etapa. Use `execute_code` quando precisar de processamento mecânico de dados ou fluxos scriptados.

## Configuração {#configuration}

```yaml
# In ~/.hermes/config.yaml
delegation:
  max_iterations: 50                        # Max turns per child (default: 50)
  # max_concurrent_children: 3              # Parallel children per batch (default: 3)
  # worktree_isolation: false               # Give each child its own git worktree (see Worktree Isolation above)
  # max_spawn_depth: 1                      # Tree depth (floor 1, no ceiling, default 1 = flat). Raise to 2 to allow orchestrator children to spawn leaves; 3+ for deeper trees.
  # orchestrator_enabled: true              # Disable to force all children to leaf role.
  model: "google/gemini-3-flash-preview"             # Optional provider/model override
  provider: "openrouter"                             # Optional built-in provider
  api_mode: anthropic_messages                       # optional; auto-detected from base_url for anthropic_messages endpoints

# Or use a direct custom endpoint instead of provider:
delegation:
  model: "qwen2.5-coder"
  base_url: "http://localhost:1234/v1"
  api_key: "local-key"
  # api_mode: "anthropic_messages"  # Optional. Wire protocol override for base_url ("chat_completions", "codex_responses", or "anthropic_messages"). Empty = auto-detect from URL (e.g. /anthropic suffix). Set explicitly for endpoints the heuristic can't classify (Azure AI Foundry, MiniMax, Zhipu GLM, LiteLLM proxies, …).
```

Quando `base_url` aponta para um endpoint compatível com Anthropic — por exemplo um caminho terminando em `/anthropic`, uma rota Claude do Azure Foundry ou um proxy MiniMax `/anthropic` — `api_mode` é auto-detectado como `anthropic_messages` para o subagente usar o formato wire correto sem você configurar nada. Defina `api_mode` explicitamente quando a detecção automática estiver errada (raro).

:::tip
O agente lida com delegação automaticamente com base na complexidade da tarefa. Você não precisa pedir explicitamente para delegar — ele fará quando fizer sentido.
:::
