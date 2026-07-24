---
sidebar_position: 7
title: "Delegação de subagentes"
description: "Crie instâncias filhas isoladas para fluxos de trabalho paralelos com delegate_task"
---

# Delegação de subagentes

A ferramenta `delegate_task` cria instâncias filhas de AIAgent com contexto isolado, acesso herdado a ferramentas e sessões de terminal próprias. Cada filho recebe uma conversa nova e trabalha de forma independente — apenas o resumo final entra no contexto do pai.

Chamadas de modelo de nível superior rodam em background automaticamente. O Hermes retorna um handle imediatamente para a conversa continuar, depois publica o resultado de volta como nova mensagem. Um subagente orquestrador espera seus próprios workers para sintetizar os resultados antes de retornar.

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
Subagentes começam com uma **conversa completamente nova**. Eles não têm conhecimento do histórico de conversa do pai, chamadas de ferramentas anteriores ou qualquer coisa discutida antes da delegação. O único contexto do subagente vem dos campos `goal` e `context` que o agente pai preenche ao chamar `delegate_task`.
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

O subagente recebe um system prompt focado construído a partir do seu goal e context, instruindo-o a completar a tarefa e fornecer um resumo estruturado do que fez, do que encontrou, arquivos modificados e problemas encontrados.

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

### Revisão de código + correção {#code-review--fix}

Delegue um fluxo revisar-e-corrigir para um contexto novo:

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

### Refatoração multi-arquivo {#multi-file-refactoring}

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

## Detalhes do modo batch {#batch-mode-details}

Quando um agente de nível superior fornece um array `tasks`, o Hermes retorna um handle de background, executa os subagentes em paralelo e publica um resultado consolidado depois que cada filho termina. Um subagente orquestrador espera seu batch no turno atual para sintetizar os resultados.

- **Concorrência máxima:** 3 tarefas por padrão (configurável via `delegation.max_concurrent_children` ou env var `DELEGATION_MAX_CONCURRENT_CHILDREN`; piso 1, sem teto rígido). Batches maiores que o limite retornam erro de ferramenta em vez de ser truncados silenciosamente.
- **Thread pool:** Usa `ThreadPoolExecutor` com o limite de concorrência configurado como max workers
- **Exibição de progresso:** No modo CLI, uma tree-view mostra chamadas de ferramentas de cada subagente em tempo real com linhas de conclusão por tarefa. No gateway, progresso é agrupado e repassado ao callback de progresso do pai
- **Ordem dos resultados:** Resultados são ordenados por índice de tarefa para corresponder à ordem de entrada independentemente da ordem de conclusão
- **Cancelamento:** Mensagens de follow-up não cancelam um batch de background de nível superior. `/stop` ou fechar/resetar a sessão proprietária cancela seus filhos ativos. Filhos síncronos de orquestrador ainda seguem o estado de interrupt do pai

Delegação síncrona de tarefa única a partir de um orquestrador roda diretamente sem overhead de thread pool.

### Conclusões duráveis em background {#durable-background-completions}

Quando uma delegação em background termina, o Hermes armazena seu evento de conclusão no
`state.db` do perfil ativo antes de publicá-lo na fila normal de fresh-turn.
Se o Hermes reiniciar após a conclusão mas antes da entrega, o evento
pendente é restaurado e roteado pelas mesmas verificações de ownership. Consumidores
concorrentes usam um claim durável, então só o consumidor que aceita com sucesso
o turno sintético confirma a entrega; tentativas falhas liberam o claim para
retry.

Isso não retoma execução filha após crash. Uma delegação cujo processo
proprietário desaparece enquanto ainda roda é registrada como `unknown`, porque
o Hermes não pode provar se seus efeitos colaterais externos aconteceram. Registros
pendentes e entregues são limitados e profile-local.

## Override de modelo {#model-override}

Você pode configurar um modelo diferente para subagentes via `config.yaml` — útil para delegar tarefas simples a modelos mais baratos/rápidos:

```yaml
# In ~/.hermes/config.yaml
delegation:
  model: "google/gemini-flash-2.0"    # Cheaper model for subagents
  provider: "openrouter"              # Optional: route subagents to a different provider
```

Se omitido, subagentes usam o mesmo modelo do pai.

## Acesso herdado a ferramentas {#inherited-tool-access}

`delegate_task` não aceita parâmetro `toolsets` voltado ao modelo. Cada subagente herda os toolsets habilitados do pai para o modelo não poder conceder a um filho capacidades que o pai não tem. Configure as ferramentas do pai antes de iniciar a conversa se o trabalho delegado precisar de capacidades adicionais.

Certas ferramentas são bloqueadas para subagentes mesmo quando o pai as tem:
- `delegation` — bloqueada para subagentes leaf (padrão). Retida para filhos `role="orchestrator"`, limitada por `max_spawn_depth` — veja [Limite de profundidade e orquestração aninhada](#depth-limit-and-nested-orchestration) abaixo.
- `clarify` — subagentes não podem interagir com o usuário
- `memory` — sem escritas na memória persistente compartilhada
- `code_execution` — filhos devem raciocinar passo a passo

## Máximo de iterações {#max-iterations}

Cada subagente tem um limite de iteração (padrão: 50) que controla quantos turnos de tool-calling pode fazer:

```python
delegate_task(
    goal="Quick file check",
    context="Check if /etc/nginx/nginx.conf exists and print its first 10 lines",
    max_iterations=10  # Simple task, don't need many turns
)
```

## Timeout do filho {#child-timeout}

Por padrão **não há timeout de wall-clock** em subagentes. Filhos falham apenas pelo que estão fazendo de fato — erros de API, erros de ferramenta ou atingir o budget de iteração — nunca por um cronômetro de delegação. Releases anteriores tinham um cap rígido (300s, depois 600s), que matava filhos legitimamente ocupados no meio da tarefa: revisões profundas de código, fan-outs grandes de pesquisa e modelos de raciocínio lentos rotineiramente precisam de mais de 10 minutos enquanto fazem progresso constante.

Filhos genuinamente travados ainda são detectados: o monitor de staleness de heartbeat para de atualizar a atividade do pai quando um filho não progride (sem chamadas API, sem início de ferramentas), deixando o timeout de inatividade do gateway disparar em um worker realmente wedged.

Se quiser um cap rígido mesmo assim (ex.: controle de custo em delegação unattended driven por cron), opt-in por instalação:

```yaml
delegation:
  child_timeout_seconds: 0     # default: 0 = no timeout
  # child_timeout_seconds: 1800  # opt-in hard cap (floor 30s)
```

Um valor positivo impõe limite rígido de wall-clock em cada filho; `0` ou negativo desabilita.

:::tip Dump de diagnóstico em timeout com zero chamadas
Com cap rígido configurado, se um subagente expira tendo feito **zero** chamadas API (geralmente: provedor inalcançável, falha de auth ou rejeição de tool-schema), `delegate_task` escreve um diagnóstico estruturado em `~/.hermes/logs/subagent-timeout-<session>-<timestamp>.log` contendo snapshot de config do subagente, trace de resolução de credenciais e mensagens de erro precoces. Muito mais fácil de root-cause do que o comportamento anterior de timeout silencioso.
:::

## Monitorando subagentes em execução (`/agents`) {#monitoring-running-subagents-agents}

A TUI inclui um overlay `/agents` (alias `/tasks`) que transforma fan-out recursivo de `delegate_task` em superfície de auditoria de primeira classe:

- Tree view ao vivo de subagentes rodando e recém-finalizados, agrupados por pai
- Rollups de custo, token e arquivos tocados por branch
- Controles kill e pause — cancele um subagente específico no meio do voo sem interromper irmãos
- Revisão post-hoc: percorra o histórico turno a turno de cada subagente mesmo depois de retornarem ao pai

O CLI clássico só imprime `/agents` como resumo em texto; a TUI é onde o overlay brilha. Veja [TUI — Slash commands](/user-guide/tui#slash-commands).

## Transcrições ao vivo {#live-transcripts}

Cada dispatch de `delegate_task` também cria um **log append-only legível por humanos por tarefa** para você (ou o agente pai) acompanhar um subagente trabalhando em tempo real em vez de esperar o resumo consolidado:

```
<hermes_home>/cache/delegation/live/<delegation_id>/task-<n>.log
```

A resposta do dispatch inclui os caminhos como `live_transcripts`, e os arquivos são pré-criados no dispatch, então funciona imediatamente:

```bash
tail -f ~/.hermes/cache/delegation/live/deleg_ab12cd34/task-0.log
```

Cada linha tem timestamp e mostra texto assistant do filho, snippets de thinking, chamadas de ferramenta (`-> tool_name({args})`), resultados de ferramentas e um marcador de status final. Um `manifest.json` no mesmo diretório descreve o batch (goals, contagem de tarefas, status por tarefa). Os logs persistem após conclusão — também servem como registro operacional full-fidelity junto ao resumo — e diretórios com mais de 7 dias são podados automaticamente em novos dispatches. Como ficam sob `cache/delegation`, também são legíveis de backends de terminal remotos (Docker/Modal/SSH).

## Limite de profundidade e orquestração aninhada {#depth-limit-and-nested-orchestration}

Por padrão, delegação é **flat**: um pai (depth 0) cria filhos (depth 1), e esses filhos não podem delegar mais. Isso previne delegação recursiva runaway.

Para fluxos multi-estágio (pesquisa → síntese, ou orquestração paralela sobre sub-problemas), um pai pode criar filhos **orchestrator** que *podem* delegar seus próprios workers:

```python
delegate_task(
    goal="Survey three code review approaches and recommend one",
    role="orchestrator",  # Allows this child to spawn its own workers
    context="...",
)
```

- `role="leaf"` (padrão): filho não pode delegar mais — idêntico ao comportamento flat-delegation.
- `role="orchestrator"`: filho retém o toolset `delegation`. Gated por `delegation.max_spawn_depth` (padrão **1** = flat, então `role="orchestrator"` é no-op nos defaults). Aumente `max_spawn_depth` para 2 para permitir filhos orchestrator criarem netos leaf; 3+ para árvores mais profundas. Sem teto superior — custo é o limite prático.
- `delegation.orchestrator_enabled: false`: kill switch global que força todo filho a `leaf` independentemente do parâmetro `role`.

**Aviso de custo:** Com `max_spawn_depth: 3` e `max_concurrent_children: 3`, a árvore pode atingir 3×3×3 = 27 agentes leaf concorrentes. Cada nível extra multiplica gasto — aumente `max_spawn_depth` intencionalmente.

## Ciclo de vida e durabilidade {#lifetime-and-durability}

:::warning Durabilidade de conclusão em background não é execução durável
Chamadas `delegate_task` voltadas ao modelo de nível superior rodam em background automaticamente onde a sessão suporta entrega posterior. O Hermes retorna um handle imediatamente, e o resultado reentra na conversa depois que o filho ou batch termina. Subagentes orquestrador esperam seus workers no turno atual porque devem sintetizar esses resultados antes de retornar. Endpoints stateless request/response caem para execução síncrona quando não podem entregar um resultado detached depois.

- Mensagens de follow-up normais não cancelam filhos em background. `/stop` cancela delegações em background rodando, e fechar ou resetar a sessão proprietária descarta seus filhos ativos.
- Fechar/resetar sessão explicitamente interrompe os filhos em background dessa sessão. Fechar um viewer TUI de uma sessão owned pelo gateway não mata o trabalho do gateway.
- Reinício do processo Hermes **não** retoma um filho rodando. Sua tentativa vira `unknown` porque o Hermes não pode provar quais efeitos colaterais aconteceram.
- Um filho que completou antes do restart mas cujo resultado não foi entregue é restaurado e roteado de volta pelas verificações normais da sessão proprietária.
- Filhos cancelados retornam resultado estruturado (`status="interrupted"`, `exit_reason="interrupted"`), mas como o pai também foi interrompido, esse resultado muitas vezes nunca chega a uma resposta visível ao usuário.

Para **execução durável** que deve sobreviver fechamento de sessão ou restart de processo, use:

- `cronjob` (action=`create`) — agenda uma execução separada do agente; imune a interrupts de turno pai.
- `terminal(background=True, notify_on_complete=True)` — comandos shell longos que continuam rodando enquanto o agente faz outras coisas.
:::

## Propriedades-chave {#key-properties}

- Cada subagente recebe **sua própria sessão de terminal** (separada do pai)
- Subagentes herdam os toolsets habilitados do pai; o modelo não pode selecioná-los ou ampliá-los por chamada
- **Delegação aninhada é opt-in** — só filhos `role="orchestrator"` podem delegar mais, e só quando `max_spawn_depth` é elevado do padrão 1 (flat). Desabilite globalmente com `orchestrator_enabled: false`.
- Subagentes leaf **não podem** chamar: `delegate_task`, `clarify`, `memory`, `execute_code`. Subagentes orchestrator retêm `delegate_task` mas ainda não podem usar as outras três.
- **Cancelamento segue ownership** — `/stop` ou fechar/resetar a sessão proprietária cancela seus filhos em background; descendentes síncronos sob orquestradores seguem o estado de interrupt do pai
- Apenas o resumo final entra no contexto do pai, mantendo uso de tokens eficiente
- Subagentes herdam **API key, configuração de provedor e credential pool** do pai (habilitando rotação de chave em rate limits)

## Delegation vs execute_code {#delegation-vs-execute_code}

| Fator | delegate_task | execute_code |
|--------|--------------|-------------|
| **Raciocínio** | Loop completo de raciocínio LLM | Apenas execução de código Python |
| **Contexto** | Conversa isolada nova | Sem conversa, só script |
| **Acesso a ferramentas** | Todas as ferramentas não bloqueadas com raciocínio | 7 ferramentas via RPC, sem raciocínio |
| **Paralelismo** | 3 subagentes concorrentes por padrão (configurável) | Script único |
| **Melhor para** | Tarefas complexas que precisam julgamento | Pipelines mecânicos multi-etapa |
| **Custo de token** | Maior (loop LLM completo) | Menor (só stdout retornado) |
| **Interação com usuário** | Nenhuma (subagentes não podem clarify) | Nenhuma |

**Regra prática:** Use `delegate_task` quando a subtarefa exige raciocínio, julgamento ou resolução multi-etapa. Use `execute_code` quando precisa de processamento mecânico de dados ou fluxos scriptados.

## Configuração {#configuration}

```yaml
# In ~/.hermes/config.yaml
delegation:
  max_iterations: 50                        # Max turns per child (default: 50)
  # max_concurrent_children: 3              # Parallel children per batch (default: 3)
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

Quando `base_url` aponta para um endpoint compatível com Anthropic — por exemplo um path terminando em `/anthropic`, uma rota Claude do Azure Foundry ou um proxy MiniMax `/anthropic` — `api_mode` é auto-detectado como `anthropic_messages` para o subagente usar o wire format certo sem você definir nada. Defina `api_mode` explicitamente quando o palpite de auto-detecção estiver errado (raro).

:::tip
O agente lida com delegação automaticamente com base na complexidade da tarefa. Você não precisa pedir explicitamente para delegar — ele fará quando fizer sentido.
:::
