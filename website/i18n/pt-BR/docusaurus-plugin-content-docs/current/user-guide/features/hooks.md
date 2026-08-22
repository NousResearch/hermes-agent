---
sidebar_position: 6
title: "Hooks de Eventos"
description: "Execute código customizado em pontos-chave do ciclo de vida — registre atividade, envie alertas, publique em webhooks"
---

# Hooks de Eventos {#event-hooks}

O Hermes tem quatro sistemas de hooks que executam código customizado em pontos-chave do ciclo de vida:

| Sistema | Registrado via | Roda em | Caso de uso |
|--------|---------------|---------|----------|
| **[Gateway hooks](#gateway-event-hooks)** | `HOOK.yaml` + `handler.py` in `~/.hermes/hooks/` | Gateway only | Logging, alerts, webhooks |
| **[Plugin hooks](#plugin-hooks)** | `ctx.register_hook()` in a [plugin](/user-guide/features/plugins) | CLI + Gateway | Tool interception, metrics, guardrails |
| **[Shell hooks](#shell-hooks)** | `hooks:` block in `~/.hermes/config.yaml` pointing at shell scripts | CLI + Gateway | Drop-in scripts for blocking, auto-formatting, context injection |
| **[Outbound webhooks](#outbound-webhooks)** | `hooks.outbound:` list in `~/.hermes/config.yaml` | CLI + Gateway | Push signed lifecycle events to external HTTP endpoints — CI, dashboards, other agents |

Erros de callback de hook são isolados e logados em vez de derrubar o agente. Hooks não são todos passivos: hooks directive/control podem mudar o fluxo, transforms podem substituir conteúdo, e um hook shell `pre_tool_call` pode bloquear ou fail closed.

## Hooks de Eventos do Gateway {#gateway-event-hooks}

Gateway hooks disparam automaticamente durante operação do gateway (Telegram, Discord, Slack, WhatsApp, Teams) sem bloquear o pipeline principal do agente.

### Criando um hook {#creating-a-hook}

Cada hook é um diretório em `~/.hermes/hooks/` contendo dois arquivos:

```text
~/.hermes/hooks/
└── my-hook/
    ├── HOOK.yaml      # Declares which events to listen for
    └── handler.py     # Python handler function
```

#### HOOK.yaml {#hookyaml}

```yaml
name: my-hook
description: Log all agent activity to a file
events:
  - agent:start
  - agent:end
  - agent:step
```

A lista `events` determina quais eventos disparam seu handler. Você pode assinar qualquer combinação de eventos, incluindo wildcards como `command:*`.

#### handler.py {#handlerpy}

```python
import json
from datetime import datetime
from pathlib import Path

LOG_FILE = Path.home() / ".hermes" / "hooks" / "my-hook" / "activity.log"

async def handle(event_type: str, context: dict):
    """Called for each subscribed event. Must be named 'handle'."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "event": event_type,
        **context,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
```

**Regras do handler:**
- Deve se chamar `handle`
- Recebe `event_type` (string) e `context` (dict)
- Pode ser `async def` ou `def` regular — ambos funcionam
- Erros são capturados e logados, nunca derrubando o agente

### Eventos disponíveis {#available-events}

| Evento | Quando dispara | Chaves de contexto |
|--------|----------------|-------------------|
| `gateway:startup` | Processo do gateway inicia | `platforms` (lista de nomes de plataformas ativas) |
| `session:start` | Nova sessão de mensagens criada | `platform`, `user_id`, `session_id`, `session_key` |
| `session:end` | Sessão encerrada (antes do reset) | `platform`, `user_id`, `session_key` |
| `session:reset` | Usuário rodou `/new` ou `/reset` | `platform`, `user_id`, `session_key` |
| `session:compress` | Compressão de contexto concluída para uma sessão | `platform`, `session_id`, `old_session_id` (vazio quando compactado in place), `in_place` (bool — `true` = transcrição compactada no mesmo id, `false` = rotacionado de `old_session_id`), `compression_count` |
| `agent:start` | Agente começa a processar uma mensagem | `platform`, `user_id`, `chat_id`, `thread_id` (id de tópico de fórum / raiz de thread; vazio fora de thread), `chat_type` (`"dm"` \| `"group"` \| `"forum"`; vazio se desconhecido), `session_id`, `message` (truncado a 500 chars) |
| `agent:step` | Cada iteração do loop de tool-calling | `platform`, `user_id`, `session_id`, `iteration`, `tool_names` |
| `agent:end` | Agente termina processamento | mesmas chaves que `agent:start`, mais `response` (truncado a 500 chars) |
| `reaction:added` | Reação emoji adicionada a mensagem que o bot vê (adaptador Slack atualmente). Requer scope `reactions:read` + subscription do evento bot `reaction_added`; o bot deve ser membro do canal. | `platform`, `reaction`, `user_id`, `item_user_id`, `item_type`, `channel_id`, `message_ts`, `team_id`, `event_ts`, `raw_event` |
| `reaction:removed` | Reação emoji removida de mensagem que o bot vê. Requer subscription do evento bot `reaction_removed`. | mesma forma que `reaction:added` |
| `command:*` | Qualquer comando slash executado | `platform`, `user_id`, `command`, `args` |

#### Correspondência com wildcard {#wildcard-matching}

Handlers registrados para `command:*` disparam para qualquer evento `command:` (`command:model`, `command:reset`, etc.). Monitore todos os comandos slash com uma única assinatura.

:::tip Respostas em thread
Um handler postando mensagem de follow-up no mesmo tópico de fórum Telegram deve incluir `message_thread_id=int(thread_id)` quando `chat_type == "forum"` e `thread_id` não estiver vazio.
:::

### Exemplos {#examples}

#### Alerta Telegram em tarefas longas {#telegram-alert-on-long-tasks}

Envie uma mensagem a si mesmo quando o agente levar mais de 10 passos:

```yaml
# ~/.hermes/hooks/long-task-alert/HOOK.yaml
name: long-task-alert
description: Alert when agent is taking many steps
events:
  - agent:step
```

```python
# ~/.hermes/hooks/long-task-alert/handler.py
import os
import httpx

THRESHOLD = 10
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_HOME_CHANNEL")

async def handle(event_type: str, context: dict):
    iteration = context.get("iteration", 0)
    if iteration == THRESHOLD and BOT_TOKEN and CHAT_ID:
        tools = ", ".join(context.get("tool_names", []))
        text = f"⚠️ Agent has been running for {iteration} steps. Last tools: {tools}"
        async with httpx.AsyncClient() as client:
            await client.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                json={"chat_id": CHAT_ID, "text": text},
            )
```

#### Logger de uso de comandos {#command-usage-logger}

Rastreie quais comandos slash são usados:

```yaml
# ~/.hermes/hooks/command-logger/HOOK.yaml
name: command-logger
description: Log slash command usage
events:
  - command:*
```

```python
# ~/.hermes/hooks/command-logger/handler.py
import json
from datetime import datetime
from pathlib import Path

LOG = Path.home() / ".hermes" / "logs" / "command_usage.jsonl"

def handle(event_type: str, context: dict):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now().isoformat(),
        "command": context.get("command"),
        "args": context.get("args"),
        "platform": context.get("platform"),
        "user": context.get("user_id"),
    }
    with open(LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
```

#### Webhook de início de sessão {#session-start-webhook}

POST para um serviço externo em novas sessões:

```yaml
# ~/.hermes/hooks/session-webhook/HOOK.yaml
name: session-webhook
description: Notify external service on new sessions
events:
  - session:start
  - session:reset
```

```python
# ~/.hermes/hooks/session-webhook/handler.py
import httpx

WEBHOOK_URL = "https://your-service.example.com/hermes-events"

async def handle(event_type: str, context: dict):
    async with httpx.AsyncClient() as client:
        await client.post(WEBHOOK_URL, json={
            "event": event_type,
            **context,
        }, timeout=5)
```

### Tutorial: BOOT.md — Executar checklist de startup em todo boot do gateway {#tutorial-bootmd--run-a-startup-checklist-on-every-gateway-boot}

Um padrão popular da comunidade: coloque um checklist Markdown em `~/.hermes/BOOT.md` e faça o agente executá-lo uma vez toda vez que o gateway iniciar. Útil para "a cada boot, verifique falhas de cron overnight e me avise no Discord se algo falhou", ou "resuma as últimas 24h de deploy.log e poste no Slack #ops".

Este tutorial mostra como construí-lo como hook definido pelo usuário. O Hermes não envia um hook BOOT.md built-in — você conecta exatamente o comportamento que quiser.

#### O que estamos construindo {#what-were-building}

1. Um arquivo em `~/.hermes/BOOT.md` com instruções de startup em linguagem natural.
2. Um gateway hook que dispara em `gateway:startup`, gera um agente one-shot com model/credenciais resolvidos do gateway e executa as instruções do BOOT.md.
3. Uma convenção `[SILENT]` para o agente optar por não enviar mensagem quando não houver nada a reportar.

#### Passo 1: Escreva seu checklist {#step-1-write-your-checklist}

Crie `~/.hermes/BOOT.md`. Escreva como se estivesse dando instruções a um assistente humano:

```markdown
# Startup Checklist

1. Run `hermes cron list` and check if any scheduled jobs failed overnight.
2. If any failed, summarize them for Discord #ops (the hook delivers your final response to its configured target).
3. Check if `/opt/app/deploy.log` has any ERROR lines from the last 24 hours. If yes, summarize them and include in the same report.
4. If nothing went wrong, reply with only `[SILENT]` so no message is sent.
```

O agente vê isso como parte do prompt, então qualquer coisa que você descrever em linguagem simples funciona — chamadas de ferramentas, comandos shell, envio de mensagens, resumo de arquivos.

#### Passo 2: Crie o hook {#step-2-create-the-hook}

```text
~/.hermes/hooks/boot-md/
├── HOOK.yaml
└── handler.py
```

**`~/.hermes/hooks/boot-md/HOOK.yaml`**

```yaml
name: boot-md
description: Run ~/.hermes/BOOT.md on gateway startup
events:
  - gateway:startup
```

**`~/.hermes/hooks/boot-md/handler.py`**

```python
"""Run ~/.hermes/BOOT.md on every gateway startup."""

import logging
import threading
from pathlib import Path

logger = logging.getLogger("hooks.boot-md")

BOOT_FILE = Path.home() / ".hermes" / "BOOT.md"


def _build_prompt(content: str) -> str:
    return (
        "You are running a startup boot checklist. Follow the instructions "
        "below exactly.\n\n"
        "---\n"
        f"{content}\n"
        "---\n\n"
        "Execute each instruction. Put any user-facing summary in your "
        "final response — the hook delivers it to the configured channel "
        "(e.g. Discord or Slack); you do not send messages yourself.\n"
        "If nothing needs attention and there is nothing to report, reply "
        "with ONLY: [SILENT]"
    )


def _run_boot_agent(content: str) -> None:
    """Spawn a one-shot agent and execute the checklist.

    Uses the gateway's resolved model and runtime credentials so this works
    against custom endpoints, aggregators, and OAuth-based providers alike.
    """
    try:
        from gateway.run import _resolve_gateway_model, _resolve_runtime_agent_kwargs
        from run_agent import AIAgent

        agent = AIAgent(
            model=_resolve_gateway_model(),
            **_resolve_runtime_agent_kwargs(),
            platform="gateway",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            max_iterations=20,
        )
        result = agent.run_conversation(_build_prompt(content))
        response = (result.get("final_response", "") or "").strip()
        if response.upper() not in {"[SILENT]", "SILENT", "NO_REPLY", "NO REPLY"}:
            logger.info("boot-md completed: %s", response[:200])
        else:
            logger.info("boot-md completed (nothing to report)")
    except Exception as e:
        logger.error("boot-md agent failed: %s", e)


async def handle(event_type: str, context: dict) -> None:
    if not BOOT_FILE.exists():
        return
    content = BOOT_FILE.read_text(encoding="utf-8").strip()
    if not content:
        return

    logger.info("Running BOOT.md (%d chars)", len(content))

    # Background thread so gateway startup isn't blocked on a full agent turn.
    thread = threading.Thread(
        target=_run_boot_agent,
        args=(content,),
        name="boot-md",
        daemon=True,
    )
    thread.start()
```

As duas linhas-chave:

- `_resolve_gateway_model()` lê o model atualmente configurado do gateway.
- `_resolve_runtime_agent_kwargs()` resolve credenciais de provider da mesma forma que um turno normal do gateway — incluindo API keys, base URLs, tokens OAuth e credential pools.

Sem elas, um `AIAgent()` bare cai nos padrões built-in e dará 401 contra qualquer endpoint não padrão.

#### Passo 3: Teste {#step-3-test-it}

Reinicie o gateway:

```bash
hermes gateway restart
```

Acompanhe os logs:

```bash
hermes logs --follow --level INFO | grep boot-md
```

Você deve ver `Running BOOT.md (N chars)` seguido de `boot-md completed: ...` (resumo do que o agente fez) ou `boot-md completed (nothing to report)` quando o agente respondeu com um token de silêncio exato como `[SILENT]`.

Exclua `~/.hermes/BOOT.md` para desabilitar o checklist — o hook permanece carregado mas pula silenciosamente quando o arquivo não existe.

#### Estendendo o padrão {#extending-the-pattern}

- **Checklists sensíveis a schedule:** use `datetime.now().weekday()` dentro das instruções do BOOT.md ("se for segunda, verifique também o log de deploy semanal"). As instruções são texto livre, então qualquer coisa que o agente possa raciocinar vale.
- **Vários checklists:** aponte o hook para outro arquivo (`STARTUP.md`, `MORNING.md`, etc.) e registre diretórios de hook separados para cada um.
- **Variante sem agente:** se não precisa de um loop completo de agente, pule `AIAgent` e faça o handler postar notificação fixa diretamente via `httpx`. Mais barato, mais rápido e sem dependência de provider.

#### Por que isso não é built-in {#why-this-isnt-a-built-in}

Uma versão anterior do Hermes enviava isso como hook built-in e spawnava silenciosamente um agente com defaults bare a cada boot do gateway. Isso surpreendia usuários com endpoints customizados e tornava a feature invisível para quem não sabia que estava rodando. Mantê-la como padrão documentado — construído por você, no seu diretório de hooks — significa que você vê exatamente o que faz e opta escrevendo os arquivos.

### Como funciona {#how-it-works}

1. No startup do gateway, `HookRegistry.discover_and_load()` escaneia `~/.hermes/hooks/`
2. Cada subdiretório com `HOOK.yaml` + `handler.py` é carregado dinamicamente
3. Handlers são registrados para seus eventos declarados
4. Em cada ponto do ciclo de vida, `hooks.emit()` dispara todos os handlers correspondentes
5. Erros em qualquer handler são capturados e logados — um hook quebrado nunca derruba o agente

:::info
Gateway hooks só disparam no **gateway** (Telegram, Discord, Slack, WhatsApp, Teams). O CLI não carrega gateway hooks. Para hooks que funcionam em todo lugar, use [plugin hooks](#plugin-hooks).
:::

## Hooks de Plugin {#plugin-hooks}

[Plugins](/user-guide/features/plugins) podem registrar hooks que disparam em sessões **CLI e gateway**. São registrados programaticamente via `ctx.register_hook()` na função `register()` do seu plugin.

Para detalhes de empacotamento e registro de plugins, veja
o [guia de Plugins](/docs/user-guide/features/plugins).

```python
def register(ctx):
    ctx.register_hook("pre_tool_call", my_tool_observer)
    ctx.register_hook("post_tool_call", my_tool_logger)
    ctx.register_hook("pre_llm_call", my_memory_callback)
    ctx.register_hook("post_llm_call", my_sync_callback)
    ctx.register_hook("on_session_start", my_init_callback)
    ctx.register_hook("on_session_end", my_cleanup_callback)
    # Kanban board lifecycle (dependency-wait blocking may fire inside its transaction):
    ctx.register_hook("kanban_task_claimed", my_claim_callback)     # dispatcher process
    ctx.register_hook("kanban_task_completed", my_done_callback)    # worker process
    ctx.register_hook("kanban_task_blocked", my_blocked_callback)   # worker process
```

**Regras gerais para todos os hooks:**

- Callbacks recebem **argumentos nomeados**. Sempre aceite `**kwargs` para compatibilidade futura.
- Exceções de callback são logadas e ignoradas; callbacks posteriores continuam.
- O catálogo abaixo é descritivo: **observers** ignoram retornos, **transforms** aceitam a primeira substituição string válida, e hooks **directive/control** consomem shapes de retorno documentados. Plugin middleware é um registry e superfície separados, não outra categoria de hook.
- Campos de correlação como `turn_id`, `api_request_id`, `task_id`, `session_id` e `api_call_count` são específicos do hook e podem estar ausentes. Trate IDs como opacos.
- Validade de nome de evento em runtime vem de `hermes_cli.plugins.VALID_HOOKS`. `hermes hooks list` lista hooks shell/outbound configurados, não todo evento disponível; `hermes hooks test <event>` reporta o set válido só quando um evento inválido é fornecido.

### Seções cache-safe de system prompt {#cache-safe-system-prompt-sections}

Plugins que precisam de orientação durável e always-on podem registrar uma seção limitada de system
prompt em vez de injetar o mesmo texto via `pre_llm_call` em
todo turno:

```python
def board_rules(session_info):
    return f"Apply the worker rules for profile {session_info['profile_name']}."

def register(ctx):
    ctx.register_system_prompt_section(
        "kanban-advanced.worker-rules",
        board_rules,                       # a string is also accepted
        position="after_memory",
        max_chars=4000,
    )
```

O contrato é deliberadamente estreito:

- IDs são identificadores globais, estáveis, de 1–128 caracteres lowercase usando só
  letras, números, `.`, `_` e `-`. IDs duplicados são rejeitados.
- `after_memory` é o único âncora de placement. Seções são ordenadas por ID,
  renderizadas após contexto de memória/profile e antes de metadados de sessão; plugins
  não podem reordenar ou substituir conteúdo core do prompt.
- Um callable recebe um mapping read-only com `session_id`, `model`,
  `provider`, `platform`, `profile_name` e `cwd`. Roda **uma vez para uma sessão
  nova**. Seus bytes renderizados são frozen na compressão e recuperados do
  system prompt full já persistido após restart/resume de processo;
  estado de plugin não é relido para uma sessão existente.
- `max_chars` é capped em 4.000 caracteres. Todas as seções plugin juntas,
  incluindo headings de audit, são capped em 8.000 caracteres e 32
  seções. Seções vazias, non-string, oversized, over-budget no agregado, ou que raise
  são skipadas com warning; construção do prompt continua.
- Toda seção aceita é nomeada no prompt e logada no início da sessão
  com seu plugin, position e character count.

Use `pre_llm_call` para contexto per-turn realmente dinâmico. Intencionalmente não
há hook de environment-hints de plugin neste contrato: mudar cwd, branch ou
outros dados de environment não deve mutar silenciosamente o prompt cached de uma sessão.
Tal hook precisa de um consumidor concreto e as mesmas semânticas frozen/resume-safe
antes de poder ser adicionado.

### Catálogo shipped de plugin-hook {#shipped-plugin-hook-catalog}

Os campos de payload abaixo são os campos exatos específicos do evento fornecidos por cada call site. Para backward compatibility, `PluginManager` também adiciona `telemetry_schema_version="hermes.observer.v1"` a todo callback de plugin-hook. Esse marker de envelope legado não significa que todos os payloads de hook compartilham um schema semântico; contratos versionados novos pertencem à família concreta de evento ou capability.

| Hook | Categoria | Timing exato e comportamento de retorno | Campos explícitos de payload | Privacidade / sensibilidade |
|---|---|---|---|---|
| [`pre_tool_call`](#pre_tool_call) | Directive/control | Uma vez antes da execução; a primeira diretiva válida `block` ou `approve` vence, e retornos `modify` são shallow-merged nos argumentos da ferramenta. | `tool_name`, `args`, `task_id`, `session_id`, `tool_call_id`, `turn_id`, `api_request_id`, `middleware_trace` | Argumentos brutos podem conter conteúdo de usuário, paths, comandos ou secrets. |
| `post_tool_call` | Observer | Após resultado blocked, error ou successful; retorno ignorado. | `tool_name`, `args`, `result`, `task_id`, `session_id`, `tool_call_id`, `turn_id`, `api_request_id`, `duration_ms`, `status`, `error_type`, `error_message`, `middleware_trace` | Texto de result/error pode conter conteúdo arbitrário de ferramenta ou usuário e secrets. |
| `transform_tool_result` | Transform | Após `post_tool_call`, antes de append na conversa; a primeira string substitui o resultado. | `tool_name`, `args`, `result`, `task_id`, `session_id`, `tool_call_id`, `turn_id`, `api_request_id`, `duration_ms`, `status`, `error_type`, `error_message` | Expõe o resultado completo bound ao model e os argumentos. |
| `transform_terminal_output` | Transform | Após captura bounded de processo foreground, antes do limit final de output; a primeira string substitui o output. | `command`, `output`, `returncode`, `task_id`, `env_type` | Command/output podem conter credentials. |
| `pre_transcription` | Transform | Disparado pelo dispatcher STT após resolução de provider e antes de qualquer backend (built-in, command-type ou plugin-registered) ser invocado; resultados dict são aplicados em ordem de registro, last-writer-wins por campo (`prompt`, `language`, `model`; `file_path` é read-only). | `file_path`, `provider`, `model`, `language`, `prompt`, `source` | O prompt final é uploaded ao provider STT configurado com o áudio — mantenha secrets fora dos retornos do hook. |
| `pre_llm_call` | Directive/control | Uma vez por turno antes do loop; todos os retornos válidos string/`{"context": ...}` são unidos e injetados na mensagem de usuário. | `session_id`, `task_id`, `turn_id`, `user_message`, `conversation_history`, `is_first_turn`, `model`, `platform`, `parent_session_id`, `sender_id` | Mensagem de usuário completa e histórico de conversa. |
| `post_llm_call` | Observer | Finalização de turno successful, non-interrupted; retorno ignorado. | `session_id`, `task_id`, `turn_id`, `user_message`, `assistant_response`, `conversation_history`, `model`, `platform` | Prompt, response e histórico completos. |
| `transform_llm_output` | Transform | Antes de `post_llm_call` e entrega final; a primeira string não vazia substitui a response. | `response_text`, `session_id`, `model`, `platform` | Texto final completo do assistant. |
| `pre_verify` | Directive/control | No gate bounded de verify de código editado; a primeira diretiva válida continue/block-stop mantém o turno indo. | `session_id`, `platform`, `model`, `coding`, `attempt`, `final_response`, `changed_paths` | Draft response e changed paths. |
| `pre_api_request` | Observer | Por tentativa de provider, imediatamente antes do request; retorno ignorado. | `task_id`, `turn_id`, `api_request_id`, `session_id`, `user_message`, `conversation_history`, `platform`, `model`, `provider`, `base_url`, `api_mode`, `api_call_count`, `retry_count`, `request_messages`, `message_count`, `tool_count`, `approx_input_tokens`, `request_char_count`, `max_tokens`, `started_at`, `middleware_trace`, `request` | Alta sensibilidade: `user_message`, `conversation_history` e `request_messages` legado são intencionalmente raw; prefira `request` sanitizado. |
| `post_api_request` | Observer | Após sucesso normalizado do provider; retorno ignorado. | `task_id`, `turn_id`, `api_request_id`, `session_id`, `platform`, `model`, `provider`, `base_url`, `api_mode`, `api_call_count`, `api_duration`, `started_at`, `ended_at`, `finish_reason`, `message_count`, `response_model`, `response`, `usage`, `assistant_message`, `assistant_content_chars`, `assistant_tool_call_count` | `response` sanitizado está disponível, mas `assistant_message` normalizado raw pode conter conteúdo de model/usuário; `usage` é data de accounting. |
| `api_request_error` | Observer | Em cada tentativa falha de provider; retorno ignorado. | `task_id`, `turn_id`, `api_request_id`, `session_id`, `platform`, `model`, `provider`, `base_url`, `api_mode`, `api_call_count`, `api_duration`, `started_at`, `ended_at`, `status_code`, `retry_count`, `max_retries`, `retryable`, `reason`, `error`, `request` | Texto de error pode conter data de provider/usuário; `request` é intended sanitizado. |
| `on_stream_start` | Observer | Despachado quando uma response LLM streaming começa; entregue fora do token path via queue limitada owned pelo host com um worker por callback; retorno ignorado. | `turn_id`, `iteration`, `session_id`, `model`, `provider`, `surface` | Só identificadores e metadados de roteamento. |
| `on_stream_delta` | Observer | Despachado por delta de texto streaming normalizado via a queue observer limitada; um callback stalled dropa só seus próprios eventos mais antigos; retorno ignorado. | `delta`, `kind` (`text` ou `reasoning`), `turn_id`, `iteration`, `session_id`, `model`, `provider`, `surface` | Texto delta é output raw do model; deltas de reasoning exigem o opt-in `plugins.stream_reasoning_deltas`. |
| `on_stream_end` | Observer | Despachado quando uma response streaming termina ou erra, após o stream fechar; retorno ignorado. | `final_text`, `finished`, `error`, `turn_id`, `iteration`, `session_id`, `model`, `provider`, `surface` | Texto de response montado completo; texto de error pode incluir data de provider. |
| `on_interim_message` | Observer | Despachado quando uma mensagem assistant mid-loop é surfaced antes da resposta final (streaming ou non-streaming); retorno ignorado. | `text`, `already_streamed`, `turn_id`, `iteration`, `session_id`, `model`, `provider`, `surface` | Texto interim completo do assistant. |
| `transform_api_error_classification` | Transform | Em cada tentativa falha de provider, no topo do classifier built-in; todos os callbacks rodam, depois o primeiro dict com `reason` válido vence (run-all-then-pick-first), e resultados válidos skipados logam warning de runtime. Só plugins Python. | `provider`, `model`, `status_code`, `error_type`, `error_code`, `error_message`, `error_body`, `error`, `approx_tokens`, `context_length`, `num_messages` | `error_message` e `error_body` podem conter data raw de provider/usuário. |
| `on_session_start` | Observer | Primeiro turno de sessão nova; retorno ignorado. | `session_id`, `model`, `platform` | Só identificadores e metadados de roteamento. |
| `on_session_end` | Observer | Canonicamente em cada finalização de turno; exits CLI/TUI têm shapes legado reduzidos adicionais. Retorno ignorado. | Canonical: `session_id`, `task_id`, `turn_id`, `completed`, `failed`, `interrupted`, `turn_exit_reason`, `model`, `platform`; paths de exit podem adicionar `reason`/`api_request_id` e omitir campos. | IDs, model/platform e outcome; payload canônico não tem body de mensagem. |
| `on_session_finalize` | Observer | Teardown CLI/TUI/gateway via `finalize_session`; shutdown ou expiry do gateway pode finalizar sem reset. Retorno ignorado. | Surface-dependent `session_id`, `platform`, opcionalmente `reason`, `old_session_id`, `new_session_id` | Identificadores de sessão e roteamento. |
| `on_session_reset` | Observer | Boundary de sessão CLI/TUI e gateway após a sessão de substituição existir; retorno ignorado. | CLI: `session_id`, `platform`, `reason`; TUI: `session_id`, `platform`; gateway: aqueles mais `reason`, `old_session_id`, `new_session_id` | Identificadores de sessão e roteamento. |
| `on_skill_lifecycle` | Observer | Após mudança autoritativa de estado de skill-usage; retorno ignorado. | `action`, `skill_name`, `provenance`, `task_id`, `session_id`, `use_count`, `reused`, `reuse_after_patch` | Expõe o nome local da skill e proveniência. |
| `subagent_start` | Observer | Filho construído e prestes a rodar; retorno ignorado. | `parent_session_id`, `parent_turn_id`, `parent_subagent_id`, `child_session_id`, `child_subagent_id`, `child_role`, `child_goal` | Goal do filho pode conter conteúdo de usuário/projeto. |
| `subagent_stop` | Observer | Exit do filho; retorno ignorado. | `parent_session_id`, `parent_turn_id`, `child_session_id`, `child_role`, `child_summary`, `child_status`, `tool_call_history`, `duration_ms` | Summary e metadados redigidos de tool-history podem revelar estrutura de projeto. |
| `pre_gateway_dispatch` | Directive/control | Mensagem inbound non-internal antes de auth/pairing/dispatch; o primeiro `skip`, `rewrite` ou `allow` válido controla o fluxo. | `event`, `gateway`, `session_store` | Objetos in-process extremamente privilegiados expõem data inbound de usuário/roteamento e handles do host. |
| `gateway_platform_event` | Observer | Após a autorização profile-scoped do gateway suceder, quando um evento nativo de plataforma suportado é normalizado na boundary do gateway (Telegram: reações, edits de mensagem; Discord: edits/deletes de mensagem, thread created/renamed); retorno ignorado. | `platform`, `event_type`, `payload` (dict específico do tipo de evento — veja os contratos por evento abaixo) | Só envelope dict plain normalizado; objetos SDK raw, handles de adapter e bot clients nunca são expostos. |
| `pre_command` | Observer | Slash command reconhecido prestes a ser despachado, antes do handler rodar, no cold-path CLI e gateway; retorno ignorado no v1 (dicts shaped directive são logados em debug). Comandos intercept do running-agent no gateway (`/stop`, `/approve` durante um run ativo) são deliberadamente excluídos — escape hatches de control-plane devem ficar fora do alcance de plugin. | `surface` (`"cli"` \| `"gateway"`), `command` (nome canônico), `alias_used`, `args_raw`, `session_key`, `platform` | `args_raw` pode conter conteúdo de usuário ou secrets digitados após o comando. |
| `pre_approval_request` | Observer | Antes de aprovação prompted ou smart; retorno ignorado. | `command`, `description`, `pattern_key`, `pattern_keys`, `session_key`, `surface`, `turn_id`, `tool_call_id` | Command pode conter secrets; preparação observer smart force-redact, mas superfícies não têm redação idêntica. |
| `post_approval_response` | Observer | Após uma decisão, timeout, ou falha de notificação gateway; retorno ignorado. | `command`, `description`, `pattern_key`, `pattern_keys`, `session_key`, `surface`, `turn_id`, `tool_call_id`, `choice`; path smart pode adicionar `decided_by` | Mesma sensibilidade de command mais metadados de decisão. |
| `kanban_task_claimed` | Observer | Após commit de claim, no processo dispatcher antes do spawn de worker; retorno ignorado. | `task_id`, `profile_name`, `board`, `assignee`, `run_id` | Identificadores de board/task/profile/assignee. |
| `kanban_task_completed` | Observer | Após completion e cleanup, geralmente no processo worker; retorno ignorado. | `task_id`, `profile_name`, `board`, `assignee`, `run_id`, `summary` | Summary pode conter conteúdo de projeto/usuário. |
| `kanban_task_blocked` | Observer | Após uma transição blocked; o path de dependency-wait dispara antes da transação sair. Retorno ignorado. | `task_id`, `profile_name`, `board`, `assignee`, `run_id`, `reason` | Reason pode conter conteúdo de projeto/usuário. |
| `on_kanban_worker_spawned` | Observer | Após `spawn_fn` retornar e o PID do worker ser persistido; roda dentro do dispatch lock, mantenha callbacks rápidos. Retorno ignorado. | `task_id`, `profile_name`, `board`, `assignee`, `run_id`, `worker_pid`, `workspace_path` | `workspace_path` é um path de filesystem e pode revelar layout de projeto ou usernames. |
| `on_kanban_worker_exited` | Observer | Tick-derived: após `detect_crashed_workers` reclaimar uma tarefa dead-PID e o reclaim commitar. Retorno ignorado. | `task_id`, `profile_name`, `board`, `assignee`, `run_id`, `worker_pid`, `exit_kind`, `exit_code`, `outcome`, `retry_status` | Só identificadores e metadados de exit. |
| `on_kanban_worker_stale_claim` | Observer | Após um claim TTL-expired ser reclaimado; extensions live-PID não disparam. Retorno ignorado. | `task_id`, `profile_name`, `board`, `assignee`, `run_id`, `worker_pid`, `heartbeat_stale`, `retry_status` | Só identificadores e metadados de claim. |
| `on_kanban_task_updated` | Observer | Após um write committed de campo de tarefa fora do lifecycle claim/complete/block (assign, overrides, editors de dashboard). Retorno ignorado. | `task_id`, `profile_name`, `board`, `assignee`, `run_id`, `changed_fields` | `changed_fields` carrega só nomes de campo, nunca valores; os valores title/body nomeados no board DB podem conter conteúdo de usuário/projeto. |
| `on_kanban_dispatch_tick` | Observer | Uma vez por tick do dispatcher, estritamente após o dispatch lock ser released; ticks idle e contended também disparam. Retorno ignorado. | `board`, `profile_name`, `dry_run`, `outcome`, `result` | `result` é o `DispatchResult` do tick e carrega task ids, assignees e workspace paths. |

---

### Hooks de output streaming {#streaming-output-hooks}

Estes hooks observer-only deixam plugins consumir output LLM streaming para telemetria, dashboards live ou pipelines TTS sem mudar a response. São entregues por queues limitadas owned pelo host com um worker background por callback registrado, então callbacks de plugin nunca rodam inline no token path. Se um callback stall, só a queue daquele callback pode encher e dropar seu evento observer pendente mais antigo; outros observers continuam recebendo eventos independentemente.

Registre-os como qualquer outro plugin hook:

```python
def on_delta(delta, kind, model, provider, **kwargs):
    if kind == "text":
        print(delta, end="", flush=True)

def register(ctx):
    ctx.register_hook("on_stream_delta", on_delta)
```

Campos comuns para os quatro hooks:

| Parâmetro | Tipo | Descrição |
|-----------|------|-------------|
| `turn_id` | `str` | Identificador opaco de turno, quando disponível |
| `iteration` | `int` | Iteração atual do API-call/tool-loop |
| `session_id` | `str` | Session id Hermes atual |
| `model` | `str` | Identificador de modelo ativo |
| `provider` | `str` | Nome do provider ativo |
| `surface` | `str` | Superfície chamadora, ex. `cli`, `discord`, `telegram` |

Campos adicionais:

| Hook | Extra fields |
|------|--------------|
| `on_stream_start` | none |
| `on_stream_delta` | `delta: str`, `kind: "text" | "reasoning"` |
| `on_stream_end` | `final_text: str`, `finished: bool`, `error: str | None` |
| `on_interim_message` | `text: str`, `already_streamed: bool` |

`on_interim_message` também pode disparar após uma response non-streaming, então registrar só esse hook não força uma call de provider para transporte streaming.

Deltas de reasoning não são expostos a plugins por padrão. Opte in explicitamente:

```yaml
plugins:
  stream_reasoning_deltas: true
```

Valores de retorno são ignorados. Para manter o stream rápido, callbacks devem enqueue o próprio trabalho e retornar rápido. Exceções são logadas e não param o stream.

---

### `pre_tool_call`

Dispara **imediatamente antes** de toda execução de ferramenta — built-in e de plugin.

**Assinatura do callback:**

```python
def my_callback(tool_name: str, args: dict, task_id: str, **kwargs):
```

| Parâmetro | Tipo | Descrição |
|-----------|------|-------------|
| `tool_name` | `str` | Name of the tool about to execute (e.g. `"terminal"`, `"web_search"`, `"read_file"`) |
| `args` | `dict` | The arguments the model passed to the tool |
| `task_id` | `str` | Session/task identifier. Empty string if not set. |

**Dispara:** Em `model_tools.py`, dentro de `handle_function_call()`, antes do handler da ferramenta rodar. Dispara uma vez por chamada de ferramenta — se o model chamar 3 ferramentas em paralelo, dispara 3 vezes.

**Valor de retorno — bloquear ou exigir aprovação:**

```python
return {"action": "block", "message": "Reason the tool call was blocked"}
# or
return {"action": "approve", "message": "Why approval is required", "rule_key": "optional:scope"}
```

A primeira diretiva válida vence (plugins Python registrados primeiro, depois shell hooks). `block` exige um `message` não vazio e encurta a ferramenta com esse texto como o erro retornado ao model. `approve` escala a call para o gate existente de aprovação humana; `message` e `rule_key` são opcionais, e denial, timeout ou erro de gate falha closed. Outros valores de retorno são ignorados, então callbacks observer-only existentes continuam funcionando inalterados.

**Valor de retorno — reescrever os argumentos da ferramenta:**

```python
return {"action": "modify", "args": {"new_string": "fixed content"}}
```

O dict `args` retornado é shallow-merged sobre os argumentos originais da ferramenta antes dela executar. Múltiplos hooks `modify` acumulam — as keys de cada hook são merged num dict acumulado construído dos args originais, então hook A mudando `path` e hook B mudando `content` ambos sobrevivem. Se dois hooks modificam a mesma key, o hook posterior vence.

Shell hooks também aceitam o formato compatível Claude Code:

```json
{"decision": "modify", "tool_input": {"new_string": "fixed content"}}
```

Ambos os formatos são normalizados internamente para `{"action": "modify", "args": {...}}`.

**Casos de uso:** Logging, trilhas de auditoria, contadores de chamadas de ferramentas, bloqueio de operações perigosas, rate limiting, enforcement de política por usuário, sanitização de argumentos, reescrita de path, injeção de parâmetros default.

**Exemplo — log de auditoria de chamadas de ferramentas:**

```python
import json, logging
from datetime import datetime

logger = logging.getLogger(__name__)

def audit_tool_call(tool_name, args, task_id, **kwargs):
    logger.info("TOOL_CALL session=%s tool=%s args=%s",
                task_id, tool_name, json.dumps(args)[:200])

def register(ctx):
    ctx.register_hook("pre_tool_call", audit_tool_call)
```

**Exemplo — aviso em ferramentas perigosas:**

```python
DANGEROUS = {"terminal", "write_file", "patch"}

def warn_dangerous(tool_name, **kwargs):
    if tool_name in DANGEROUS:
        print(f"⚠ Executing potentially dangerous tool: {tool_name}")

def register(ctx):
    ctx.register_hook("pre_tool_call", warn_dangerous)
```

---

### `post_tool_call`

Dispara **imediatamente após** toda execução de ferramenta retornar.

**Assinatura do callback:**

```python
def my_callback(tool_name: str, args: dict, result: str, task_id: str,
                duration_ms: int, **kwargs):
```

| Parâmetro | Tipo | Descrição |
|-----------|------|-------------|
| `tool_name` | `str` | Name of the tool that just executed |
| `args` | `dict` | The arguments the model passed to the tool |
| `result` | `str` | The tool's return value (always a JSON string) |
| `task_id` | `str` | Session/task identifier. Empty string if not set. |
| `duration_ms` | `int` | How long the tool's dispatch took, in milliseconds (measured with `time.monotonic()` around `registry.dispatch()`). |

**Dispara:** Em `model_tools.py`, dentro de `handle_function_call()`, após o handler da ferramenta retornar. Dispara uma vez por chamada de ferramenta. **Não** dispara se a ferramenta levantou exceção não tratada (o erro é capturado e retornado como string JSON de erro, e `post_tool_call` dispara com essa string de erro como `result`).

**Valor de retorno:** Ignorado.

**Casos de uso:** Logging de resultados de ferramentas, coleta de métricas, rastreamento de taxas de sucesso/falha, dashboards de latência, alertas de budget por ferramenta, notificações quando ferramentas específicas completam.

**Exemplo — rastrear métricas de uso de ferramentas:**

```python
from collections import Counter, defaultdict
import json

_tool_counts = Counter()
_error_counts = Counter()
_latency_ms = defaultdict(list)

def track_metrics(tool_name, result, duration_ms=0, **kwargs):
    _tool_counts[tool_name] += 1
    _latency_ms[tool_name].append(duration_ms)
    try:
        parsed = json.loads(result)
        if "error" in parsed:
            _error_counts[tool_name] += 1
    except (json.JSONDecodeError, TypeError):
        pass

def register(ctx):
    ctx.register_hook("post_tool_call", track_metrics)
```

---

### `pre_llm_call`

Dispara **uma vez por turno**, antes do loop de tool-calling começar. Todos os retornos válidos de callback são agregados em ordem de plugin e injetados na mensagem de usuário do turno atual.

**Assinatura do callback:**

```python
def my_callback(session_id: str, user_message: str, conversation_history: list,
                is_first_turn: bool, model: str, platform: str, **kwargs):
```

| Parâmetro | Tipo | Descrição |
|-----------|------|-------------|
| `session_id` | `str` | Unique identifier for the current session |
| `user_message` | `str` | The user's original message for this turn (before any skill injection) |
| `conversation_history` | `list` | Copy of the full message list (OpenAI format: `[{"role": "user", "content": "..."}]`) |
| `is_first_turn` | `bool` | `True` if this is the first turn of a new session, `False` on subsequent turns |
| `model` | `str` | The model identifier (e.g. `"anthropic/claude-sonnet-4.6"`) |
| `platform` | `str` | Where the session is running: `"cli"`, `"telegram"`, `"discord"`, etc. |

**Dispara:** Em `run_agent.py`, dentro de `run_conversation()`, após compressão de contexto mas antes do loop `while` principal. Dispara uma vez por chamada `run_conversation()` (ou seja, uma vez por turno de usuário), não uma vez por chamada API dentro do loop de ferramentas.

**Valor de retorno:** Se o callback retornar um dict com chave `"context"`, ou string não vazia simples, o texto é anexado à mensagem de usuário do turno atual. Retorne `None` para sem injeção.

```python
# Inject context
return {"context": "Recalled memories:\n- User likes Python\n- Working on hermes-agent"}

# Plain string (equivalent)
return "Recalled memories:\n- User likes Python"

# No injection
return None
```

**Onde o contexto é injetado:** Sempre na **mensagem de usuário**, nunca no system prompt. Isso preserva o prompt cache — o system prompt permanece idêntico entre turnos, então tokens em cache são reutilizados. O system prompt é território do Hermes (orientação de model, enforcement de ferramentas, personalidade, skills). Plugins contribuem contexto junto ao input do usuário.

O `content` limpo da mensagem de usuário permanece inalterado. Para estabilidade de replay e prompt-cache, o Hermes pode persistir a mensagem exact API-bound, incluindo contexto injetado por plugin, no sidecar `api_content` da row.

Quando **vários plugins** retornam contexto, suas saídas são unidas com quebras de linha duplas na ordem de descoberta de plugins (alfabética por nome de diretório).

**Casos de uso:** Recall de memória, injeção de contexto RAG, guardrails, analytics por turno.

**Exemplo — recall de memória:**

```python
import httpx

MEMORY_API = "https://your-memory-api.example.com"

def recall(session_id, user_message, is_first_turn, **kwargs):
    try:
        resp = httpx.post(f"{MEMORY_API}/recall", json={
            "session_id": session_id,
            "query": user_message,
        }, timeout=3)
        memories = resp.json().get("results", [])
        if not memories:
            return None
        text = "Recalled context:\n" + "\n".join(f"- {m['text']}" for m in memories)
        return {"context": text}
    except Exception:
        return None

def register(ctx):
    ctx.register_hook("pre_llm_call", recall)
```

**Exemplo — guardrails:**

```python
POLICY = "Never execute commands that delete files without explicit user confirmation."

def guardrails(**kwargs):
    return {"context": POLICY}

def register(ctx):
    ctx.register_hook("pre_llm_call", guardrails)
```

---

### `post_llm_call`

Dispara **uma vez por turno**, após o loop de tool-calling completar e o agente produzir resposta final. Só dispara em turnos **bem-sucedidos** — não dispara se o turno foi interrompido.

**Assinatura do callback:**

```python
def my_callback(session_id: str, user_message: str, assistant_response: str,
                conversation_history: list, model: str, platform: str, **kwargs):
```

| Parâmetro | Tipo | Descrição |
|-----------|------|-------------|
| `session_id` | `str` | Unique identifier for the current session |
| `user_message` | `str` | The user's original message for this turn |
| `assistant_response` | `str` | The agent's final text response for this turn |
| `conversation_history` | `list` | Copy of the full message list after the turn completed |
| `model` | `str` | The model identifier |
| `platform` | `str` | Where the session is running |

**Dispara:** Em `run_agent.py`, dentro de `run_conversation()`, após o loop de ferramentas sair com resposta final. Guardado por `if final_response and not interrupted` — então **não** dispara quando o usuário interrompe mid-turn ou o agente atinge o limite de iterações sem produzir resposta.

**Valor de retorno:** Ignorado.

**Casos de uso:** Sincronizar dados de conversa com sistema de memória externo, calcular métricas de qualidade de resposta, logar resumos de turno, disparar ações de follow-up.

**Exemplo — sincronizar com memória externa:**

```python
import httpx

MEMORY_API = "https://your-memory-api.example.com"

def sync_memory(session_id, user_message, assistant_response, **kwargs):
    try:
        httpx.post(f"{MEMORY_API}/store", json={
            "session_id": session_id,
            "user": user_message,
            "assistant": assistant_response,
        }, timeout=5)
    except Exception:
        pass  # best-effort

def register(ctx):
    ctx.register_hook("post_llm_call", sync_memory)
```

**Exemplo — rastrear tamanhos de resposta:**

```python
import logging
logger = logging.getLogger(__name__)

def log_response_length(session_id, assistant_response, model, **kwargs):
    logger.info("RESPONSE session=%s model=%s chars=%d",
                session_id, model, len(assistant_response or ""))

def register(ctx):
    ctx.register_hook("post_llm_call", log_response_length)
```

---

### `pre_verify`

Dispara **uma vez por turno quando o agente editou código**, logo antes de terminar (após o guard verify-on-stop built-in). Este é um gate de política usuário/plugin: um callback pode manter o agente rodando — executar check, adiar, arrumar o diff — em vez de deixá-lo parar.

A orientação de verificação enviada pelo Hermes não é um hook `pre_verify` padrão. Ela é anexada ao nudge verify-on-stop baseado em evidência quando código editado carece de evidência de verificação fresca, então não cria um segundo caminho padrão de continuação. Defina `agent.verify_guidance: false` para manter aquele nudge de evidência built-in conciso.

**Assinatura do callback:**

```python
def my_callback(session_id: str, platform: str, model: str, coding: bool,
                attempt: int, final_response: str, changed_paths: list, **kwargs):
```

| Parâmetro | Tipo | Descrição |
|-----------|------|-------------|
| `session_id` | `str` | Unique identifier for the current session |
| `platform` | `str` | Where the session is running (`"cli"`, `"telegram"`, …) |
| `model` | `str` | The model identifier |
| `coding` | `bool` | Whether the turn is in the coding posture (in a code workspace) — scope your hook on this |
| `attempt` | `int` | How many times this turn has already been nudged (0 on the first) — self-throttle on this |
| `final_response` | `str` | The answer the agent is about to deliver |
| `changed_paths` | `list` | Files the agent edited this turn (sorted, always non-empty here) |

Escopo um hook ao contexto de coding verificando `coding` e faça one-shot com `attempt` (shell hooks leem ambos de `.extra`), da mesma forma que um hook `pre_tool_call` escopa em `tool_name` — assim você pode registrar vários hooks `pre_verify`, cada um disparando só onde deve.

**Dispara:** Em `agent/conversation_loop.py`, no ponto em que o agente aceitaria resposta final, imediatamente após o check verify-on-stop — mas só quando o agente editou código neste turno e pelo menos um hook `pre_verify` está registrado.

**Valor de retorno — manter o agente rodando:**

```python
return {"action": "continue", "message": "Run the formatter on your changes, then finish."}
```

A `message` é anexada como turno de usuário sintético e o loop roda de novo. A forma Stop do Claude-Code (`{"decision": "block", "reason": "..."}`, onde bloquear o stop significa *continuar*) também é aceita. Uma diretiva sem mensagem — ou qualquer outro retorno — deixa o turno terminar.

**Limitado:** diretivas continue consecutivas em um turno são limitadas por `agent.max_verify_nudges` (padrão 3), então um hook que sempre diz continue nunca pode prender o loop. A resposta tentada fica no histórico mas não é mostrada ao usuário enquanto o agente está sendo nudged.

**Torne idempotente:** o hook re-dispara após cada nudge, então gate em `attempt` (`if attempt: return None`) — senão só nudges até o limite.

**Casos de uso:** adiar tests/lints durante iteração criativa, exigir checks verdes para certos caminhos, bloquear "done" até existir entrada de changelog, executar checklist de verificação específico do projeto.

**Exemplo — adiar checks em trabalho UI criativo, escopado + one-shot:**

```python
UI = (".tsx", ".jsx", ".css", ".scss")

def defer_ui_checks(coding, attempt, changed_paths, **kwargs):
    if attempt or not coding:
        return None  # one-shot, coding only
    if not all(p.endswith(UI) for p in changed_paths):
        return None  # only pure-UI edits
    return {
        "action": "continue",
        "message": "This is UI work — don't run tests/lints yet; ask the user to "
                   "eyeball it first, and clean the diff before any commit.",
    }

def register(ctx):
    ctx.register_hook("pre_verify", defer_ui_checks)
```

Para orientação permanente que deve moldar o nudge built-in de evidência faltante, use `agent.verify_guidance`. Para regras mais amplas de postura de coding que não precisam *gatear* verificação, prefira `agent.coding_instructions` em `config.yaml` — vai no coding brief e não custa turno extra.

---

### `transform_api_error_classification`

Dispara uma vez por chamada API falha, no topo de `agent/error_classifier.classify_api_error()`, antes do pipeline built-in. Plugins de provider o usam para donar os quirks de erro do próprio provider sem patches no core. É behavior-changing (família transform): a classificação retornada dirige retry, compressão, rotação de credencial e roteamento de fallback.

Callbacks recebem o contexto de erro parseado como kwargs — `provider` (self-scope nisto), `model`, `status_code`, `error_type`, `error_code`, `error_message`, `error_body`, `error`, `approx_tokens`, `context_length`, `num_messages`. Retorne `None` para recusar, ou um dict para reivindicar o erro:

```python
return {"reason": "model_not_found",   # required: a FailoverReason name
        "retryable": False, "should_fallback": True}  # optional recovery-hint overrides
```

Dispatch é run-all-then-pick-first: todo callback roda, falhas são isoladas, e o primeiro resultado válido em ordem de registro vence (resultados válidos-mas-losing logam warning de runtime). Dicts inválidos e reasons desconhecidos são skipados, então um plugin quebrado nunca pode quebrar classificação.

**Privacidade:** `error_message` e `error_body` podem carregar data unredacted de provider. **Só plugins Python** — registros shell são recusados no parse de config com um warning.

---

### `on_session_start`

Dispara **uma vez** quando uma sessão totalmente nova é criada. **Não** dispara na continuação de sessão (quando o usuário envia segunda mensagem em sessão existente).

**Assinatura do callback:**

```python
def my_callback(session_id: str, model: str, platform: str, **kwargs):
```

| Parâmetro | Tipo | Descrição |
|-----------|------|-------------|
| `session_id` | `str` | Unique identifier for the new session |
| `model` | `str` | The model identifier |
| `platform` | `str` | Where the session is running |

**Dispara:** Em `run_agent.py`, dentro de `run_conversation()`, durante o primeiro turno de nova sessão — especificamente após o system prompt ser construído mas antes do loop de ferramentas iniciar. O check é `if not conversation_history` (sem mensagens anteriores = nova sessão).

**Valor de retorno:** Ignorado.

**Casos de uso:** Inicializar estado escopado à sessão, aquecer caches, registrar a sessão com serviço externo, logar inícios de sessão.

**Exemplo — inicializar cache de sessão:**

```python
_session_caches = {}

def init_session(session_id, model, platform, **kwargs):
    _session_caches[session_id] = {
        "model": model,
        "platform": platform,
        "tool_calls": 0,
        "started": __import__("datetime").datetime.now().isoformat(),
    }

def register(ctx):
    ctx.register_hook("on_session_start", init_session)
```

---

### `on_session_end`

Dispara no **fim absoluto** de toda chamada `run_conversation()`, independentemente do resultado. Também dispara do handler de exit do CLI se o agente estava mid-turn quando o usuário saiu.

**Assinatura do callback:**

```python
def my_callback(session_id: str, completed: bool, interrupted: bool,
                model: str, platform: str, **kwargs):
```

| Parâmetro | Tipo | Descrição |
|-----------|------|-------------|
| `session_id` | `str` | Unique identifier for the session |
| `completed` | `bool` | `True` if the agent produced a final response, `False` otherwise |
| `interrupted` | `bool` | `True` if the turn was interrupted (user sent new message, `/stop`, or quit) |
| `model` | `str` | The model identifier |
| `platform` | `str` | Where the session is running |

**Dispara:** Em dois lugares:
1. **`run_agent.py`** — no fim de toda chamada `run_conversation()`, após todo cleanup. Sempre dispara, mesmo se o turno deu erro.
2. **`cli.py`** — no handler atexit do CLI, mas **só** se o agente estava mid-turn (`_agent_running=True`) quando o exit ocorreu. Isso captura Ctrl+C e `/exit` durante processamento. Neste caso, `completed=False` e `interrupted=True`.

**Valor de retorno:** Ignorado.

**Casos de uso:** Flush de buffers, fechar conexões, persistir estado de sessão, logar duração de sessão, cleanup de recursos inicializados em `on_session_start`.

**Exemplo — flush e cleanup:**

```python
_session_caches = {}

def cleanup_session(session_id, completed, interrupted, **kwargs):
    cache = _session_caches.pop(session_id, None)
    if cache:
        # Flush accumulated data to disk or external service
        status = "completed" if completed else ("interrupted" if interrupted else "failed")
        print(f"Session {session_id} ended: {status}, {cache['tool_calls']} tool calls")

def register(ctx):
    ctx.register_hook("on_session_end", cleanup_session)
```

**Exemplo — rastreamento de duração de sessão:**

```python
import time, logging
logger = logging.getLogger(__name__)

_start_times = {}

def on_start(session_id, **kwargs):
    _start_times[session_id] = time.time()

def on_end(session_id, completed, interrupted, **kwargs):
    start = _start_times.pop(session_id, None)
    if start:
        duration = time.time() - start
        logger.info("SESSION_DURATION session=%s seconds=%.1f completed=%s interrupted=%s",
                     session_id, duration, completed, interrupted)

def register(ctx):
    ctx.register_hook("on_session_start", on_start)
    ctx.register_hook("on_session_end", on_end)
```

---

### `on_session_finalize`

Dispara quando o CLI ou gateway **desmonta** uma sessão ativa — por exemplo, quando o usuário roda `/new`, o gateway fez GC de sessão idle, ou o CLI saiu com agente ativo. Use para flush de estado ligado ao session ID saindo. Num reset de gateway, a sessão de substituição já existe antes deste callback rodar.

**Assinatura do callback:**

```python
def my_callback(session_id: str | None, platform: str, **kwargs):
```

| Parâmetro | Tipo | Descrição |
|-----------|------|-------------|
| `session_id` | `str` or `None` | The outgoing session ID. May be `None` if no active session existed. |
| `platform` | `str` | `"cli"` or the messaging platform name (`"telegram"`, `"discord"`, etc.). |

**Dispara:** Em teardown CLI/TUI e em paths de reset, shutdown ou idle-expiry do gateway. Shutdown e expiry do gateway podem finalizar sem um `on_session_reset` correspondente.

**Valor de retorno:** Ignorado.

**Casos de uso:** Persistir métricas finais de sessão antes do session ID ser descartado, fechar recursos por sessão, emitir evento final de telemetria, drenar writes enfileirados.

---

### `on_session_reset`

Dispara numa boundary de sessão CLI ou TUI, ou quando o gateway **troca para uma nova session key** em chat ativo. Isso deixa plugins reagirem a estado de conversa limpo sem esperar o próximo `on_session_start`.

**Assinatura do callback:**

```python
def my_callback(session_id: str, platform: str, **kwargs):
```

| Parâmetro | Tipo | Descrição |
|-----------|------|-------------|
| `session_id` | `str` | The new session's ID (already rotated to the fresh value). |
| `platform` | `str` | `"cli"`, `"tui"`, or the messaging platform name. |
| `reason` | `str`, optional | Presente em paths de reset CLI e gateway. |
| `old_session_id` | `str`, optional | Session ID outgoing só gateway. |
| `new_session_id` | `str`, optional | Session ID de substituição só gateway. |

**Dispara:** CLI fornece `session_id`, `platform` e `reason`; TUI fornece `session_id` e `platform`; gateway adiciona `reason`, `old_session_id` e `new_session_id` após alocar a key de substituição. Num reset de gateway, a ordem é: criar e persistir a substituição → `on_session_finalize(old_id)` → `on_session_reset(new_id)` → `on_session_start(new_id)` no primeiro turno inbound.

**Valor de retorno:** Ignorado.

**Casos de uso:** Resetar caches por sessão keyed por `session_id`, emitir analytics "session rotated", preparar bucket de estado fresco.

---

Veja o **[guia Construir um Plugin](/developer-guide/plugins)** para o walkthrough completo incluindo schemas de ferramentas, handlers e padrões avançados de hooks.

---

### `subagent_start`

Dispara **uma vez por agente filho** após `delegate_task` ter construído o `AIAgent` filho e antes desse filho rodar. Se você delegar uma tarefa ou um batch de três, este hook dispara uma vez para cada filho.

Este hook é específico ao ciclo de vida de delegação/subagente. Não é um gate universal "antes de qualquer invocação de agente" para gateway, CLI, cron, batch, MoA ou outras execuções de agente originadas por runners.

**Assinatura do callback:**

```python
def my_callback(parent_session_id: str | None,
                parent_turn_id: str,
                parent_subagent_id: str | None,
                child_session_id: str | None,
                child_subagent_id: str,
                child_role: str,
                child_goal: str,
                **kwargs):
```

| Parâmetro | Tipo | Descrição |
|-----------|------|-------------|
| `parent_session_id` | `str \| None` | Session ID of the delegating parent agent. |
| `parent_turn_id` | `str` | Turn ID of the parent agent turn that requested delegation, if available. |
| `parent_subagent_id` | `str \| None` | Parent subagent ID when this child was spawned by another subagent; `None` for top-level parent agents. |
| `child_session_id` | `str \| None` | Session ID allocated for the child agent. |
| `child_subagent_id` | `str` | Stable subagent ID used by delegation observability and controls. |
| `child_role` | `str` | Effective child role after delegation policy is applied, for example `"leaf"` or `"orchestrator"`. |
| `child_goal` | `str` | Delegated goal/prompt that the child agent will execute. |

**Dispara:** Em `tools/delegate_tool.py`, dentro de `_build_child_agent()`, após o `AIAgent` filho ter sido construído e anotado com metadata de identidade de subagente, e antes de `_run_single_child()` rodar o filho.

**Valor de retorno:** Ignorado. Este é hook observador apenas; retornar valor não bloqueia nem muta a execução do agente filho.

**Casos de uso:** Logging de criação de subagente, mapear relações de sessão pai/filho, rastrear árvores de delegação aninhadas, emitir registros de auditoria pre-run, pré-alocar recursos de observabilidade por filho.

**Exemplo — log de criação de subagente:**

```python
import logging

logger = logging.getLogger(__name__)

def log_subagent_start(
    parent_session_id,
    parent_turn_id,
    child_session_id,
    child_subagent_id,
    child_role,
    child_goal,
    **kwargs,
):
    logger.info(
        "SUBAGENT_START parent=%s turn=%s child_session=%s child=%s role=%s goal=%r",
        parent_session_id,
        parent_turn_id,
        child_session_id,
        child_subagent_id,
        child_role,
        child_goal[:200],
    )

def register(ctx):
    ctx.register_hook("subagent_start", log_subagent_start)
```

:::info
`subagent_start` é útil para observabilidade de delegação, mas não é hook de política bloqueante. Para bloquear delegação antes de um filho ser construído, use [`pre_tool_call`](#pre_tool_call) para bloquear a chamada da ferramenta `delegate_task`.
:::

---

### `subagent_stop`

Dispara **uma vez por agente filho** após `delegate_task` terminar. Se você delegou uma tarefa ou batch de três, este hook dispara uma vez para cada filho, serializado na thread pai.

**Assinatura do callback:**

```python
def my_callback(parent_session_id: str, child_role: str | None,
                child_summary: str | None, child_status: str,
                tool_call_history: list[dict], duration_ms: int, **kwargs):
```

| Parâmetro | Tipo | Descrição |
|-----------|------|-------------|
| `parent_session_id` | `str` | Session ID of the delegating parent agent |
| `child_role` | `str \| None` | Orchestrator role tag set on the child (`None` if the feature isn't enabled) |
| `child_summary` | `str \| None` | The final response the child returned to the parent |
| `child_status` | `str` | `"completed"`, `"failed"`, `"interrupted"`, or `"error"` |
| `tool_call_history` | `list[dict]` | Ordered metadata-only tool calls: `tool_name`, bounded `tool_input`, `input_bytes`, `output_bytes`, and `status`; raw inputs and outputs are excluded |
| `duration_ms` | `int` | Wall-clock time spent running the child, in milliseconds |

**Dispara:** Em `tools/delegate_tool.py`, após `ThreadPoolExecutor.as_completed()` drenar todos os futures filhos. O disparo é marshalled para a thread pai para autores de hook não precisarem raciocinar sobre execução concorrente de callbacks.

**Valor de retorno:** Ignorado.

**Casos de uso:** Logging de atividade de orquestração, acumular durações de filhos para billing, escrever registros de auditoria pós-delegação.

**Exemplo — log de atividade de orquestrador:**

```python
import logging
logger = logging.getLogger(__name__)

def log_subagent(parent_session_id, child_role, child_status, duration_ms, **kwargs):
    logger.info(
        "SUBAGENT parent=%s role=%s status=%s duration_ms=%d",
        parent_session_id, child_role, child_status, duration_ms,
    )

def register(ctx):
    ctx.register_hook("subagent_stop", log_subagent)
```

:::info
Com delegação pesada (ex.: roles orchestrator × 5 leaves × profundidade aninhada), `subagent_stop` dispara muitas vezes por turno. Mantenha seu callback rápido; empurre trabalho caro para fila em background.
:::

---

### `pre_gateway_dispatch`

Dispara **uma vez por `MessageEvent` inbound** no gateway, após o guard de evento interno mas **antes** de auth/pairing e dispatch do agente. Este é o ponto de interceptação para políticas de fluxo de mensagens no nível gateway (janelas listen-only, handover humano, roteamento por chat, etc.) que não encaixam limpo em um único adaptador de plataforma.

**Assinatura do callback:**

```python
def my_callback(event, gateway, session_store, **kwargs):
```

| Parâmetro | Tipo | Descrição |
|-----------|------|-------------|
| `event` | `MessageEvent` | The normalized inbound message (has `.text`, `.source`, `.message_id`, `.internal`, etc.). |
| `gateway` | `GatewayRunner` | The active gateway runner, so plugins can call `gateway.adapters[platform].send(...)` for side-channel replies (owner notifications, etc.). |
| `session_store` | `SessionStore` | For silent transcript ingestion via `session_store.append_to_transcript(...)`. |

**Dispara:** Em `gateway/run.py`, dentro de `GatewayRunner._handle_message()`, imediatamente após `is_internal` ser computado. **Eventos internos pulam o hook inteiramente** (são gerados pelo sistema — conclusões de processo em background, etc. — e não devem ser gate-kept por política user-facing).

**Valor de retorno:** `None` ou dict. O primeiro dict de ação reconhecido vence; resultados restantes de plugin são ignorados. Exceções em callbacks de plugin são capturadas e logadas; o gateway sempre cai para dispatch normal em erro.

| Retorno | Efeito |
|--------|--------|
| `{"action": "skip", "reason": "..."}` | Descarta a mensagem — sem resposta do agente, sem fluxo de pareamento, sem auth. Assume-se que o plugin tratou (ex.: ingestão silenciosa na transcrição). |
| `{"action": "rewrite", "text": "new text"}` | Substitui `event.text`, depois continua dispatch normal com o evento modificado. Útil para colapsar mensagens ambientes em buffer num único prompt. |
| `{"action": "allow"}` / `None` | Dispatch normal — executa a cadeia completa auth / pairing / agent-loop. |

**Casos de uso:** Chats em grupo listen-only (só responder quando marcado; buffer mensagens ambientes no contexto); handover humano (ingestão silenciosa de mensagens de cliente enquanto dono trata o chat manualmente); rate limiting por perfil; roteamento driven por política.

**Exemplo — descartar DMs não autorizadas silenciosamente sem disparar código de pareamento:**

```python
def deny_unauthorized_dms(event, **kwargs):
    src = event.source
    if src.chat_type == "dm" and not _is_approved_user(src.user_id):
        return {"action": "skip", "reason": "unauthorized-dm"}
    return None

def register(ctx):
    ctx.register_hook("pre_gateway_dispatch", deny_unauthorized_dms)
```

**Exemplo — reescrever buffer de mensagens ambientes em prompt único ao mencionar:**

```python
_buffers = {}

def buffer_or_rewrite(event, **kwargs):
    key = (event.source.platform, event.source.chat_id)
    buf = _buffers.setdefault(key, [])
    if _bot_mentioned(event.text):
        combined = "\n".join(buf + [event.text])
        buf.clear()
        return {"action": "rewrite", "text": combined}
    buf.append(event.text)
    return {"action": "skip", "reason": "ambient-buffered"}

def register(ctx):
    ctx.register_hook("pre_gateway_dispatch", buffer_or_rewrite)
```

---

### `gateway_platform_event`

Dispara para eventos nativos de plataforma suportados só **depois** que a checagem de autorização profile-scoped do gateway suceder. O callback recebe dicts plain; objetos SDK raw, handles de adapter, bot clients e contextos de callback nunca fazem parte deste contrato estável.

Reações de mensagem Telegram foram o primeiro evento suportado; edits de mensagem, deletes e eventos de lifecycle de thread seguiram:

```python
def on_platform_event(platform, event_type, payload, **kwargs):
    if platform == "telegram" and event_type == "reaction":
        print(payload["chat_id"], payload["message_id"], payload["emojis"])
    elif event_type == "message_edited":
        print(platform, payload["chat_id"], payload["message_id"], payload["text"])

def register(ctx):
    ctx.register_hook("gateway_platform_event", on_platform_event)
```

| Parâmetro | Tipo | Descrição |
|-----------|------|-------------|
| `platform` | `str` | Stable platform id (`"telegram"`, `"discord"`). |
| `event_type` | `str` | Id de contrato local do evento (veja a tabela abaixo). |
| `payload` | `dict` | Campos específicos do tipo de evento, documentados por tipo abaixo. |

Todo payload é aditivo e específico do evento; não há um payload gateway monolítico versionado. Todos os ids são strings; campos faltando/indisponíveis são `None`, nunca guessed. Eventos malformados e eventos cuja source não pode ser autorizada são dropados (fail closed). Um rebuild transiente de Telegram Application re-registra o observer junto com os handlers core.

**Contratos de payload por evento (v1, aditivo):**

| `event_type` | Plataformas | Campos de payload |
|--------------|-----------|----------------|
| `reaction` | telegram | `emojis: list[str]`, `custom_emoji_ids: list[str]`, `chat_id: str`, `message_id: str`, `thread_id: str \| None` (updates de reação Telegram não carregam topic id, então atualmente sempre `None`). |
| `message_edited` | telegram, discord | `chat_id: str`, `message_id: str`, `thread_id: str \| None`, `text: str \| None` (texto ou caption editado, bounded; `None` para edits só-mídia ou quando uncached), `edited_at: str \| None` (ISO 8601). |
| `message_deleted` | discord | `chat_id: str`, `message_id: str`, `thread_id: str \| None`, `author_id: str \| None`. O evento de delete do Discord não identifica o deleter; a source autorizada é o autor da mensagem deletada, e deletes uncached nunca disparam. |
| `thread_created` | discord | `thread_id: str`, `parent_chat_id: str \| None`, `name: str \| None`, `owner_id: str \| None`. |
| `thread_renamed` | discord | `thread_id: str`, `parent_chat_id: str \| None`, `old_name: str \| None`, `new_name: str`. Dispara só quando o nome realmente mudou; outros updates de thread (archive, slowmode, tags) são dropados. O evento thread-update do Discord não carrega actor, então o owner da thread é a source autorizada. |

Edits progressivos da própria bot (streaming) nunca disparam `message_edited` no Discord — eventos authored pela bot são dropados no fire-site.

Este hook é observer-only: **não** adiciona acesso raw-event ou acesso a adapter. **Acesso raw a payload SDK é deliberadamente não shipped** — objetos SDK de adapter mudam de forma sem aviso e virariam superfície de API un-evolvable; onde genuinamente necessário exige sua própria capability explícita (`gateway.raw_events`) com label "no stability guarantee" e seu próprio design (tracked em #64228). Para *agir* numa plataforma (adicionar uma reação, renomear uma thread), use a facade `ctx.platform_actions` gated por capability documentada no [guia de plugins](plugins.md#platform-actions) — é gated off por padrão atrás da capability `gateway.platform_actions`. `PluginContext.dispatch_tool()` só pode chamar tools registradas no tool registry; `send_message` intencionalmente não está registrado lá (seu transporte é reservado para paths explícitos de delivery CLI, cron, kanban e MCP). Um contrato futuro de outbound-delivery deve primeiro fornecer conteúdo/handles delivered estáveis em todos os adapters; este slice não pré-registra um hook inerte `gateway_message_delivered`.

---

### `pre_approval_request`

Dispara antes de uma decisão de aprovação ser solicitada. Cobre superfícies prompted — CLI interativo, Ink TUI, plataformas gateway e clientes ACP — e decisões `approvals.mode=smart` feitas sem prompt humano (`surface="smart"`). Em smart mode, o hook roda antes do LLM auxiliar ser chamado.

Este é o lugar certo para conectar um notificador customizado — por exemplo, app de menu-bar macOS que exibe notificação allow/deny, ou log de auditoria que registra toda requisição de aprovação com contexto.

**Assinatura do callback:**

```python
def my_callback(
    command: str,
    description: str,
    pattern_key: str,
    pattern_keys: list[str],
    session_key: str,
    surface: str,
    **kwargs,
):
```

| Parâmetro | Tipo | Descrição |
|-----------|------|-------------|
| `command` | `str` | Terminal command or `execute_code` script being assessed. Smart and gateway payloads are redacted before observer dispatch. Smart observer redaction is mandatory even when `security.redact_secrets` is disabled; if redaction fails, smart hooks are skipped. |
| `description` | `str` | Human-readable reason(s) the command is flagged (combined when multiple patterns match) |
| `pattern_key` | `str` | Primary pattern key that triggered the approval (e.g. `"rm_rf"`, `"sudo"`) |
| `pattern_keys` | `list[str]` | All pattern keys that matched |
| `session_key` | `str` | Session identifier, useful for scoping notifications per-chat |
| `surface` | `str` | `"cli"` for interactive CLI/TUI prompts, `"gateway"` for async platform approvals, or `"smart"` for auxiliary-LLM auto approve/deny decisions |

**Valor de retorno:** ignorado. Hooks aqui são observadores apenas; não podem vetar ou pré-responder a aprovação. Use [`pre_tool_call`](#pre_tool_call) para bloquear ferramenta antes de chegar ao sistema de aprovação.

**Casos de uso:** Notificações desktop, alertas push, audit logging, Slack webhooks, roteamento de escalation, métricas.

**Exemplo — notificação desktop no macOS:**

```python
import subprocess

def notify_approval(command, description, session_key, **kwargs):
    title = "Hermes needs approval"
    body = f"{description}: {command[:80]}"
    subprocess.Popen([
        "osascript", "-e",
        f'display notification "{body}" with title "{title}"',
    ])

def register(ctx):
    ctx.register_hook("pre_approval_request", notify_approval)
```

---

### `post_approval_response`

Dispara após decisão de aprovação prompted ou smart, após timeout de prompt, ou quando o gateway não consegue entregar a notificação de aprovação. Falha de notificação emite `choice="notify_failed"` antes de qualquer decisão de aprovação existir.

**Assinatura do callback:**

```python
def my_callback(
    command: str,
    description: str,
    pattern_key: str,
    pattern_keys: list[str],
    session_key: str,
    surface: str,
    choice: str,
    **kwargs,
):
```

Mesmos kwargs que `pre_approval_request`, mais:

| Parâmetro | Tipo | Descrição |
|-----------|------|-------------|
| `choice` | `str` | Prompted surfaces use `"once"`, `"session"`, `"always"`, `"deny"`, `"timeout"`, or `"notify_failed"`; smart decisions use `"smart_approve"` or `"smart_deny"` |
| `decided_by` | `str` | `"aux_llm"` for smart decisions; absent on prompted surfaces |

**Valor de retorno:** ignorado.

**Casos de uso:** Fechar notificação desktop correspondente, registrar decisão final em log de auditoria, atualizar métricas, avançar rate limiter.

```python
def log_decision(command, choice, session_key, **kwargs):
    logger.info("approval %s: %s for session %s", choice, command[:60], session_key)

def register(ctx):
    ctx.register_hook("post_approval_response", log_decision)
```

---

### `pre_transcription`

Dispara dentro do dispatcher STT (`tools.transcription_tools.transcribe_audio`) **depois** que o provider foi resolvido e **antes** de qualquer backend ser invocado, seja esse backend built-in, um provider `type: command`, ou um provider registrado por plugin. Deixa um plugin dirigir o próprio request de transcrição em vez de só observar o transcript depois.

**Assinatura do callback:**

```python
def my_callback(
    file_path: str,
    provider: str,
    model: str | None,
    language: str | None,
    prompt: str | None,
    source: str | None,
    **kwargs,
) -> dict | None:
```

| Parâmetro | Tipo | Descrição |
|-----------|------|-------------|
| `file_path` | `str` | Absolute path to the audio file about to be transcribed. Read-only. |
| `provider` | `str` | Resolved STT provider (`local`, `groq`, `openai`, `mistral`, `xai`, `elevenlabs`, `deepinfra`, `local_command`, a command provider name, or a plugin provider name). |
| `model` | `str \| None` | Model resolved so far, or `None` when the backend default applies. |
| `language` | `str \| None` | Language from the provider's config section, or `None`. |
| `prompt` | `str \| None` | The static [`stt.prompt`](/user-guide/configuration#transcription-prompt-vocabulary-hints) value, or `None`. |
| `source` | `str \| None` | Caller surface label (`gateway`, `voice_mode`, …). Observability only, not used for dispatch. |

**Valor de retorno:** um `dict` com quaisquer de `"prompt"`, `"language"`, `"model"` mapeados a strings, ou `None` para deixar o request inalterado. Valores non-string, keys desconhecidas e `file_path` são ignorados (tentativas de `file_path` são logadas como warning). Results são aplicados em **ordem de registro, last-writer-wins por campo**, sobre o valor de config `stt.prompt`. Retornar `""` para `prompt` limpa o prompt configurado daquela requisição.

**Casos de uso:** Injetar uma lista de vocabulário per-user ou per-chat antes do áudio ser uploaded, forçar `language` do locale do caller, fazer downgrade de `model` para gravações longas, rotear sources ruidosas para um modelo diferente.

```python
VOCAB = "Hermes, Teknium, Nous Research, kanban"

def add_vocab(provider, prompt, source, **kwargs):
    if source != "gateway":
        return None
    return {"prompt": f"{prompt}. {VOCAB}" if prompt else VOCAB}

def register(ctx):
    ctx.register_hook("pre_transcription", add_vocab)
```

Nem todo backend aceita um prompt. `local` mapeia para `initial_prompt` do faster-whisper; `openai`, `groq`, `mistral` e `deepinfra` enviam como `prompt`; `xai`, `elevenlabs`, `local_command` e providers `type: command` logam em DEBUG e transcrevem sem ele. Veja a [tabela de suporte de provider](/user-guide/configuration#transcription-prompt-vocabulary-hints) para a matriz completa e a boundary de privacidade. Erros de hook-plumbing são fail-open: o dispatch continua com o request unmodified.

---

### `transform_tool_result`

Dispara **depois** que uma ferramenta retorna e **antes** do resultado ser anexado à conversa. Deixa um plugin reescrever a string de resultado de QUALQUER ferramenta — não só saída de terminal — antes do model ver.

**Assinatura do callback:**

```python
def my_callback(tool_name: str, args: dict, result: str, task_id: str, **kwargs) -> str | None:
```

O payload completo também inclui `session_id`, `tool_call_id`, `turn_id`, `api_request_id`, `duration_ms`, `status`, `error_type` e `error_message`. `result` é o resultado final retornado pelo tool dispatch; ele e `args` podem conter conteúdo arbitrário de usuário/ferramenta e secrets.

**Valor de retorno:** A primeira `str` substitui o resultado (incluindo uma string vazia); `None` deixa inalterado.

**Casos de uso:** Redigir PII específico da organização de saída `web_extract`, envolver respostas JSON longas de ferramentas em header de resumo, injetar hints retrieval-augmented em resultados `read_file`, reescrever relatórios de subagente `delegate_task` em schema específico do projeto.

```python
import re
SECRET = re.compile(r"sk-[A-Za-z0-9]{32,}")

def redact_secrets(tool_name, result, **kwargs):
    if SECRET.search(result):
        return SECRET.sub("[REDACTED]", result)
    return None

def register(ctx):
    ctx.register_hook("transform_tool_result", redact_secrets)
```

Aplica-se a toda ferramenta. Para reescrita só de terminal veja `transform_terminal_output` abaixo — é mais estreito, roda antes de `transform_tool_result`, e sua substituição ainda está sujeita ao limit final de output da ferramenta terminal.

---

### `transform_terminal_output`

Dispara dentro da ferramenta `terminal` depois que a captura de processo foreground já foi limitada pelo environment, e antes do limit final de output. Deixa plugins substituir o stdout/stderr capturado; a substituição ainda está sujeita ao limit final de output.

**Assinatura do callback:**

```python
def my_callback(
    command: str,
    output: str,
    returncode: int,
    task_id: str,
    env_type: str,
    **kwargs,
) -> str | None:
```

| Parâmetro | Tipo | Descrição |
|-----------|------|-------------|
| `command` | `str` | The shell command that produced the output. |
| `output` | `str` | Combined stdout/stderr after bounded process capture. |
| `returncode` | `int` | Process return code. |
| `task_id` | `str` | Effective task identifier, or an empty string. |
| `env_type` | `str` | Execution-environment type. |

**Valor de retorno:** A primeira `str` substitui o output; `None` deixa inalterado. Command e output podem conter credentials ou outros dados sensíveis.

**Casos de uso:** Injetar resumos para comandos que produzem saída massiva (`du -ah`, `find`, `tree`), marcar saída com marker específico do projeto para hooks downstream saberem como tratar, remover ruído de timing que varia entre runs e derrota prompt caching.

```python
def summarize_find(command, output, **kwargs):
    if command.startswith("find ") and len(output) > 50_000:
        lines = output.count("\n")
        head = "\n".join(output.splitlines()[:40])
        return f"{head}\n\n[summary: {lines} paths total, showing first 40]"
    return None

def register(ctx):
    ctx.register_hook("transform_terminal_output", summarize_find)
```

Combina com `transform_tool_result`, que roda depois para toda ferramenta, incluindo `terminal`.

---

### `transform_llm_output`

Dispara **uma vez por turno** após o loop de tool-calling completar e o model produzir resposta final, **antes** dessa resposta ser entregue ao usuário (CLI, gateway ou caller programático). Deixa um plugin reescrever o texto final do assistente com métodos de programação clássica — sem tokens extras de inferência gastos em texto SOUL flavor ou transform driven por skill.

**Assinatura do callback:**

```python
def my_callback(
    response_text: str,
    session_id: str,
    model: str,
    platform: str,
    **kwargs,
) -> str | None:
```

| Parâmetro | Tipo | Descrição |
|-----------|------|-------------|
| `response_text` | `str` | The assistant's final response text for this turn. |
| `session_id` | `str` | Session ID for this conversation (may be empty for one-shot runs). |
| `model` | `str` | Model name that produced the response (e.g. `anthropic/claude-sonnet-4.6`). |
| `platform` | `str` | Delivery platform (`cli`, `telegram`, `discord`, …; empty when unset). |

**Valor de retorno:** `str` não vazia para substituir o texto de resposta, `None` ou string vazia para deixar inalterado. **Primeira string não vazia vence** quando vários plugins registram. Diferente dos transforms de ferramenta e terminal, uma string vazia não é aceita como substituição.

**Casos de uso:** Aplicar transform de personalidade/vocabulário (pirate-speak, Spongebob), redigir identificadores específicos do usuário do texto final, anexar footer de assinatura específico do projeto, impor guia de estilo house sem gastar tokens em instruções SOUL.

Quando streaming CLI está habilitado, um transform append-only é impresso após o
corpo streamed. Um transform que substitui a resposta é impresso por completo após
o corpo streamed, rotulado como transformação pós-stream, para conteúdo de substituição
nunca ser perdido silenciosamente.

```python
import os, re

def spongebob(response_text, **kwargs):
    if os.environ.get("SPONGEBOB_MODE") != "on":
        return None  # pass through unchanged
    return re.sub(r"!", "!! Tartar sauce!", response_text)

def register(ctx):
    ctx.register_hook("transform_llm_output", spongebob)
```

O hook é guardado em resposta não vazia e não interrompida — não dispara em interrupções de botão stop ou turnos vazios. Exceções são logadas como warnings e não quebram execução do agente.

### Hooks observadores de API-request {#api-request-observer-hooks}

#### `pre_api_request`

Dispara para cada tentativa de provider imediatamente antes de enviá-la. É observer-only. Os campos legado `user_message`, `conversation_history` e `request_messages` são raw e intencionalmente unsanitized para compatibilidade; consumidores novos devem preferir o envelope sanitizado `request`.

#### `post_api_request`

Dispara após uma response de provider ter sido normalizada com sucesso. É observer-only. Prefira o `response` sanitizado; `assistant_message` é a mensagem normalizada raw, e `usage` contém data de accounting.

#### `api_request_error`

Dispara para uma tentativa falha de provider com timing de status/retry, um objeto `error`, e `request` sanitizado. É observer-only. Mensagens de error ainda podem conter data de provider ou usuário.

### `on_skill_lifecycle`

Dispara após uma mudança autoritativa de estado de skill-usage. É observer-only e expõe o `skill_name` local, proveniência, IDs de correlação, usage count e flags de reuse.

### Observadores de lifecycle Kanban {#kanban-lifecycle-observers}

#### `kanban_task_claimed`

Dispara após o commit de claim no processo dispatcher, imediatamente antes do spawn de worker.

#### `kanban_task_completed`

Dispara após completion e cleanup, geralmente no processo worker. Seu `summary` pode conter conteúdo de projeto ou usuário.

#### `kanban_task_blocked`

Dispara após uma transição blocked normal. O path de dependency-wait o invoca antes da write transaction sair. Seu `reason` pode conter conteúdo de projeto ou usuário.

Todos os três hooks kanban são observer-only e carregam `task_id`, `profile_name`, `board`, `assignee` e `run_id`; completed adiciona `summary`, e blocked adiciona `reason`.

### Observadores Kanban de worker-lifecycle, task-mutation e dispatch {#kanban-worker-lifecycle-task-mutation-and-dispatch-observers}

Cinco observers adicionais (RFC #58548) estendem a família kanban. Todos são observer-only, disparam após a transação relevante commitar, e short-circuit em `has_hook` — sem subscriber, o comportamento de dispatch fica inalterado. Hooks task-scoped carregam os mesmos campos comuns dos hooks acima.

- **`on_kanban_worker_spawned`** — após `spawn_fn` retornar e o PID do worker ser persistido. Adiciona `worker_pid` (pode ser `None`) e `workspace_path`. Roda dentro do dispatch lock; mantenha callbacks rápidos.
- **`on_kanban_worker_exited`** — tick-derived, quando `detect_crashed_workers` reclaima uma tarefa dead-PID. Adiciona `worker_pid`, `exit_kind`, `exit_code`, `outcome`, `retry_status`.
- **`on_kanban_worker_stale_claim`** — quando um claim TTL-expired é reclaimado; extensions live-PID não disparam. Adiciona `worker_pid`, `heartbeat_stale`, `retry_status`.
- **`on_kanban_task_updated`** — após um write committed de campo de tarefa fora do lifecycle claim/complete/block (`assign_task`, overrides de model/reasoning, editors de dashboard). Adiciona `changed_fields` — só nomes de campo, nunca valores.
- **`on_kanban_dispatch_tick`** — uma vez por tick do dispatcher, estritamente após o dispatch lock ser released, incluindo ticks idle e lock-contended. Payload: `board`, `profile_name`, `dry_run`, `outcome`, `result`.

---

## Hooks de Shell {#shell-hooks}

Declare shell-script hooks no seu `~/.hermes/config.yaml` e o Hermes os executará como subprocessos sempre que o evento plugin-hook correspondente disparar — em sessões CLI e gateway. Não é necessário escrever plugin Python.

Use shell hooks quando quiser um script drop-in de arquivo único (Bash, Python, qualquer coisa com shebang) para:

- **Bloquear ou modificar uma chamada de ferramenta** — rejeitar comandos `terminal` perigosos, impor políticas por diretório, exigir aprovação para operações destrutivas `write_file` / `patch`, ou reescrever argumentos (sanitizar paths, injetar defaults) antes da ferramenta rodar.
- **Rodar após chamada de ferramenta** — auto-formatar arquivos Python ou TypeScript que o agente acabou de escrever, logar chamadas API, disparar workflow CI.
- **Injetar contexto no próximo turno LLM** — prepender saída de `git status`, dia da semana atual ou documentos recuperados à mensagem de usuário (veja [`pre_llm_call`](#pre_llm_call)).
- **Observar eventos de ciclo de vida** — escrever linha de log quando subagente completa (`subagent_stop`) ou sessão inicia (`on_session_start`).

Shell hooks são registrados chamando `agent.shell_hooks.register_from_config(cfg)` tanto no startup do CLI (`hermes_cli/main.py`) quanto no startup do gateway (`gateway/run.py`). Compõem naturalmente com plugin hooks Python — ambos fluem pelo mesmo dispatcher.

### Comparação rápida {#comparison-at-a-glance}

| Dimension | Shell hooks | [Plugin hooks](#plugin-hooks) | [Gateway hooks](#gateway-event-hooks) |
|-----------|-------------|-------------------------------|---------------------------------------|
| Declared in | `hooks:` block in `~/.hermes/config.yaml` | `register()` in a `plugin.yaml` plugin | `HOOK.yaml` + `handler.py` directory |
| Lives under | `~/.hermes/agent-hooks/` (by convention) | `~/.hermes/plugins/<name>/` | `~/.hermes/hooks/<name>/` |
| Language | Any (Bash, Python, Go binary, …) | Python only | Python only |
| Runs in | CLI + Gateway | CLI + Gateway | Gateway only |
| Events | `VALID_HOOKS` (incl. `subagent_stop`) | `VALID_HOOKS` | Gateway lifecycle (`gateway:startup`, `agent:*`, `command:*`) |
| Can block a tool call | Yes (`pre_tool_call`) | Yes (`pre_tool_call`) | No |
| Can inject LLM context | Yes (`pre_llm_call`) | Yes (`pre_llm_call`) | No |
| Consent | First-use prompt per `(event, command)` pair | Implicit (Python plugin trust) | Implicit (dir trust) |
| Inter-process isolation | Yes (subprocess) | No (in-process) | No (in-process) |

### Schema de configuração {#configuration-schema}

```yaml
hooks:
  <event_name>:                  # Must be in VALID_HOOKS
    - matcher: "<regex>"         # Optional; used for pre/post_tool_call only
      command: "<shell command>" # Required; runs via shlex.split, shell=False
      timeout: <seconds>         # Optional; default 60, capped at 300
      fail_closed: <bool>        # Optional; default false. pre_tool_call only.
                                 # `failClosed` also accepted (Cursor/Claude Code compat)

hooks_auto_accept: false         # See "Consent model" below
```

Nomes de evento devem ser um dos [eventos de plugin hook](#plugin-hooks); typos produzem aviso "Did you mean X?" e são ignorados. Chaves desconhecidas dentro de uma entrada são ignoradas; `command` faltando é skip-with-warning. `timeout > 300` é clamped com aviso. `fail_closed: true` em evento diferente de `pre_tool_call` avisa e é ignorado (só eventos capazes de bloquear podem fail closed).

### Protocolo wire JSON {#json-wire-protocol}

Cada vez que o evento dispara, o Hermes spawna subprocesso para todo hook correspondente (matcher permitindo), envia payload JSON para **stdin** e lê **stdout** de volta como JSON.

**stdin — payload que o script recebe:**

```json
{
  "hook_event_name": "pre_tool_call",
  "tool_name":       "terminal",
  "tool_input":      {"command": "rm -rf /"},
  "session_id":      "sess_abc123",
  "cwd":             "/home/user/project",
  "extra":           {"task_id": "...", "tool_call_id": "..."}
}
```

`tool_name` e `tool_input` são `null` para eventos não-ferramenta (`pre_llm_call`, `subagent_stop`, ciclo de vida de sessão). O dict `extra` carrega todos os kwargs específicos do evento (`user_message`, `conversation_history`, `child_role`, `duration_ms`, …). Valores não serializáveis são stringificados em vez de omitidos.

**stdout — resposta opcional:**

```jsonc
// Block a pre_tool_call (both shapes accepted; normalised internally):
{"decision": "block", "reason":  "Forbidden: rm -rf"}   // Claude-Code style
{"action":   "block", "message": "Forbidden: rm -rf"}   // Hermes-canonical

// Modify a pre_tool_call — rewrite tool args before dispatch:
{"action": "modify", "args": {"new_string": "fixed content"}}         // Hermes-canonical
{"decision": "modify", "tool_input": {"new_string": "fixed content"}} // Claude-Code style

// Inject context for pre_llm_call:
{"context": "Today is Friday, 2026-04-17"}

// Keep the agent going at the verify gate (pre_verify); both shapes accepted:
{"action": "continue", "message": "Run the formatter, then finish."}
{"decision": "block",  "reason":  "Run the formatter, then finish."}

// Silent no-op — any empty / non-matching output is fine:
```

JSON malformado, exit codes não-zero e timeouts logam aviso mas nunca abortam o loop do agente.

### Exit code 2 = block (compatível Claude Code / Cursor) {#exit-code-2--block-claude-code--cursor-compatible}

Um hook `pre_tool_call` que sai com code **2** bloqueia a chamada de ferramenta mesmo quando stdout não carrega block JSON. A mensagem de block é resolvida em ordem de prioridade:

1. stdout block JSON (`reason` / `message`), when present;
2. the first 400 characters of stderr;
3. a generic `"Blocked by shell hook."` default.

Então o hook bloqueante mais simples possível é:

```bash
#!/usr/bin/env bash
echo "policy violation: rm -rf is not permitted" >&2
exit 2
```

Para eventos cujo block directive não é honrado (tudo exceto `pre_tool_call`), exit 2 é tratado como qualquer outro exit não-zero: aviso é logado e stdout ainda é parseado.

### Fail-open vs fail-closed {#fail-open-vs-fail-closed}

Por padrão shell hooks **fail open**: erro de spawn, timeout ou stdout não parseável loga aviso e a ação prossegue. Esse é o padrão certo para hooks de observabilidade — mas errado para gates de segurança. Um secret-scanner que crashou não deve permitir silenciosamente a chamada de ferramenta que deveria vetar.

Defina `fail_closed: true` (ou `failClosed: true`, grafia Cursor/Claude Code) em entrada `pre_tool_call` para inverter isso:

```yaml
hooks:
  pre_tool_call:
    - matcher: "terminal|write_file|patch"
      command: "~/.hermes/agent-hooks/secret-scan.sh"
      timeout: 10
      fail_closed: true
```

Com `fail_closed: true`, cada um destes agora **bloqueia** a chamada de ferramenta com `hook <command> failed closed: <reason>`:

| Falha | Fail-open (padrão) | `fail_closed: true` |
|---------|--------------------|--------------------|
| Comando não encontrado / não executável | aviso, prossegue | **block** |
| Timeout | aviso, prossegue | **block** |
| stdout não-JSON (ex.: stack trace) | aviso, prossegue | **block** |
| Exit limpo, JSON no-op válido (`{}`) | prossegue | prossegue |

`fail_closed` só se aplica a eventos capazes de bloquear (`pre_tool_call` hoje); defini-lo em qualquer outro evento loga aviso no parse de config e é ignorado. `hermes hooks test` reflete essas semânticas — a linha `parsed` mostra exatamente a forma de block que o dispatcher receberia.

### Exemplos práticos {#worked-examples}

#### 1. Auto-formatar arquivos Python após cada write {#1-auto-format-python-files-after-every-write}

```yaml
# ~/.hermes/config.yaml
hooks:
  post_tool_call:
    - matcher: "write_file|patch"
      command: "~/.hermes/agent-hooks/auto-format.sh"
```

```bash
#!/usr/bin/env bash
# ~/.hermes/agent-hooks/auto-format.sh
payload="$(cat -)"
path=$(echo "$payload" | jq -r '.tool_input.path // empty')
[[ "$path" == *.py ]] && command -v black >/dev/null && black "$path" 2>/dev/null
printf '{}\n'
```

A visão in-context do agente do arquivo **não** é relida automaticamente — o reformat só afeta o arquivo em disco. Chamadas subsequentes `read_file` pegam a versão formatada.

#### 2. Bloquear comandos `terminal` destrutivos {#2-block-destructive-terminal-commands}

```yaml
hooks:
  pre_tool_call:
    - matcher: "terminal"
      command: "~/.hermes/agent-hooks/block-rm-rf.sh"
      timeout: 5
```

```bash
#!/usr/bin/env bash
# ~/.hermes/agent-hooks/block-rm-rf.sh
payload="$(cat -)"
cmd=$(echo "$payload" | jq -r '.tool_input.command // empty')
if echo "$cmd" | grep -qE 'rm[[:space:]]+-rf?[[:space:]]+/'; then
  printf '{"decision": "block", "reason": "blocked: rm -rf / is not permitted"}\n'
else
  printf '{}\n'
fi
```

#### 3. Injetar `git status` em todo turno (equivalente Claude-Code `UserPromptSubmit`) {#3-inject-git-status-into-every-turn-claude-code-userpromptsubmit-equivalent}

```yaml
hooks:
  pre_llm_call:
    - command: "~/.hermes/agent-hooks/inject-cwd-context.sh"
```

```bash
#!/usr/bin/env bash
# ~/.hermes/agent-hooks/inject-cwd-context.sh
cat - >/dev/null   # discard stdin payload
if status=$(git status --porcelain 2>/dev/null) && [[ -n "$status" ]]; then
  jq --null-input --arg s "$status" \
     '{context: ("Uncommitted changes in cwd:\n" + $s)}'
else
  printf '{}\n'
fi
```

O evento `UserPromptSubmit` do Claude Code intencionalmente não é evento Hermes separado — `pre_llm_call` dispara no mesmo lugar e já suporta injeção de contexto. Use aqui.

#### 4. Logar toda conclusão de subagente {#4-log-every-subagent-completion}

```yaml
hooks:
  subagent_stop:
    - command: "~/.hermes/agent-hooks/log-orchestration.sh"
```

```bash
#!/usr/bin/env bash
# ~/.hermes/agent-hooks/log-orchestration.sh
log=~/.hermes/logs/orchestration.log
jq -c '{ts: now, parent: .session_id, extra: .extra}' < /dev/stdin >> "$log"
printf '{}\n'
```

### Modelo de consentimento {#consent-model}

Cada par único `(event, command)` pede aprovação do usuário na primeira vez que o Hermes o vê, depois persiste a decisão em `~/.hermes/shell-hooks-allowlist.json`. Execuções subsequentes (CLI ou gateway) pulam o prompt.

Três escape hatches contornam o prompt interativo — qualquer um basta:

1. `--accept-hooks` flag on the CLI (e.g. `hermes --accept-hooks chat`)
2. `HERMES_ACCEPT_HOOKS=1` environment variable
3. `hooks_auto_accept: true` in `~/.hermes/config.yaml`

Runs non-TTY (gateway, cron, CI) precisam de um destes três — senão qualquer hook recém-adicionado fica silenciosamente não registrado e loga aviso.

**Edições de script são silenciosamente confiadas.** A allowlist keya na string exata de command, não no hash do script, então editar o script em disco não invalida consentimento. `hermes hooks doctor` sinaliza drift de mtime para você ver edições e decidir se re-aprova.

#### Allowlist manual {#manual-allowlisting}

Allowlist manual é útil para deployments non-TTY ou service-account onde operador não pode responder o prompt de first-use interativamente. O arquivo allowlist é `~/.hermes/shell-hooks-allowlist.json`, e o formato esperado é array `approvals`. Cada approval registra o `event` do hook e a string exata de `command`:

```json
{
  "approvals": [
    {
      "event": "post_llm_call",
      "command": "/home/hermes/.hermes/hooks/my-hook.py"
    }
  ]
}
```

A string de command deve corresponder exatamente ao command do hook configurado. Objeto keyed por path com campo `sha256` não é o formato esperado e não aprovará o hook. Verifique entradas manuais com `hermes hooks list`.

### A CLI `hermes hooks` {#the-hermes-hooks-cli}

| Comando | O que faz |
|---------|--------------|
| `hermes hooks list` | Dump configured hooks with matcher, timeout, and consent status |
| `hermes hooks test <event> [--for-tool X] [--payload-file F]` | Fire every matching hook against a synthetic payload and print the parsed response |
| `hermes hooks revoke <command>` | Remove every allowlist entry matching `<command>` (takes effect on next restart) |
| `hermes hooks doctor` | For every configured hook: check exec bit, allowlist status, mtime drift, JSON output validity, and rough execution time |

### Segurança {#security}

Shell hooks rodam com **suas credenciais completas de usuário** — mesmo trust boundary de entrada cron ou alias shell. Trate o bloco `hooks:` em `config.yaml` como configuração privilegiada:

- Referencie só scripts que você escreveu ou revisou por completo.
- Mantenha scripts dentro de `~/.hermes/agent-hooks/` para o caminho ser fácil de auditar.
- Re-execute `hermes hooks doctor` após pull de config compartilhada para ver hooks recém-adicionados antes de registrarem.
- Se seu config.yaml é version-controlled em equipe, revise PRs que mudam a seção `hooks:` da mesma forma que revisaria config CI.

### Ordem e precedência {#ordering-and-precedence}

Tanto plugin hooks Python quanto shell hooks fluem pelo mesmo dispatcher `invoke_hook()`. Plugins Python são registrados primeiro (`discover_and_load()`), shell hooks segundo (`register_from_config()`), então decisões de block `pre_tool_call` Python têm precedência em empates. O primeiro block válido vence — o agregador retorna assim que qualquer callback produz `{"action": "block", "message": str}` com mensagem não vazia.

## Webhooks de Saída {#outbound-webhooks}

Outbound webhooks são o espelho push-side da [plataforma webhook inbound](/user-guide/messaging/webhooks): webhooks inbound acordam o Hermes quando o mundo muda; outbound webhooks avisam o mundo quando o Hermes faz algo. Configure lista de endpoints HTTP e eventos de ciclo de vida que interessam, e o Hermes POSTa payload JSON assinado a cada endpoint sempre que evento correspondente dispara — sem polling no lado receptor.

Usos típicos:

- Notificar sistema CI ou dashboard quando turno de agente termina (`on_session_end`)
- Rastrear conclusões de subagente em uma frota (`subagent_stop`)
- Alimentar atividade de ferramentas em monitoramento externo (`post_tool_call` com `matcher`)
- Acordar *outra* instância Hermes: aponte URL ao webhook inbound daquela instância

### Configuração {#configuration}

Adicione uma lista `hooks.outbound:` em `~/.hermes/config.yaml`:

```yaml
hooks:
  outbound:
    - name: ci-notify                       # optional label for logs
      url: https://ci.example.com/hermes-events
      events: [on_session_end, subagent_stop]
      secret_env: HERMES_OUTBOUND_WEBHOOK_SECRET   # env var holding the HMAC secret
      timeout: 10                           # per-attempt seconds (1–60)

    - name: tool-monitor
      url: https://metrics.example.com/hooks/hermes
      events: [post_tool_call]
      matcher: "terminal|delegate_task"     # regex, tool-scoped events only
```

Qualquer evento do conjunto plugin-hook é válido (`pre_tool_call`, `post_tool_call`, `pre_llm_call`, `post_llm_call`, `on_session_start`, `on_session_end`, `subagent_start`, `subagent_stop`, ...). Entradas malformadas avisam e são ignoradas — webhook quebrado nunca derruba o agente. Mudanças entram em vigor na próxima sessão CLI / restart do gateway.

Secrets: prefira `secret_env` (nome de variável de ambiente, tipicamente em `~/.hermes/.env`) a literal inline `secret:`, para o arquivo de config ficar livre de credenciais. Entradas sem secret são entregues unsigned (flagged como `UNSIGNED` por `hermes hooks list`).

### Formato wire {#wire-format}

Cada disparo POSTa corpo JSON com a mesma forma top-level do stdin de shell hooks, mais metadata de entrega:

```json
{
  "hook_event_name": "on_session_end",
  "tool_name": null,
  "tool_input": null,
  "session_id": "sess_abc123",
  "cwd": "/home/user/project",
  "extra": {"completed": true, "interrupted": false, "model": "...", "platform": "cli"},
  "delivery_id": "3f2c9a...",
  "timestamp": "2026-07-22T14:00:00Z"
}
```

Headers:

| Header | Valor |
|--------|-------|
| `Content-Type` | `application/json` |
| `X-Hermes-Event` | The hook event name |
| `X-Hermes-Delivery` | Unique id per delivery — same value as `delivery_id` in the body |
| `X-Hermes-Signature-256` | `sha256=<hex>` — HMAC-SHA256 of the raw body, GitHub-style; only present when a secret is configured |

Verifique a assinatura exatamente como faria com webhook GitHub:

```python
import hashlib, hmac

def verify(body: bytes, header: str, secret: str) -> bool:
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header)
```

Como `delivery_id` e `timestamp` vivem **dentro do corpo assinado**, um receptor verificado também ganha proteção contra replay de graça:

- **Dedupe** em `delivery_id` (ou header correspondente `X-Hermes-Delivery`) — lembre ids vistos recentemente e pule duplicatas. Hermes retenta entregas falhas uma vez, então o mesmo id pode chegar legitimamente duas vezes.
- **Rejeite eventos stale** checando `timestamp` contra seu relógio com janela de tolerância (5 minutos é o padrão comum). Atacante replaying request capturado não pode forjar timestamp fresco sem o secret.

### Semântica de entrega {#delivery-semantics}

- **Fire-and-forget, fora do hot path.** Eventos são serializados e enfileirados instantaneamente; uma única thread em background executa os HTTP POSTs. Endpoint lento ou morto nunca pode travar chamada de ferramenta ou turno de agente.
- **Notify-only.** Diferente de shell hooks, outbound webhooks não podem bloquear chamadas de ferramentas ou injetar contexto — o corpo de resposta é ignorado. Observam, nunca dirigem.
- **Retries limitados.** Erros de conexão e respostas 5xx são retentados uma vez com backoff; respostas 4xx não são retentadas (o receptor disse que a requisição em si está errada). Falhas são logadas e descartadas — entrega é best-effort, não garantida.
- **Redirects nunca são seguidos.** Resposta 3xx é tratada como misconfiguration e logada — seguir POST redirecionado descartaria silenciosamente o payload assinado. Aponte `url` ao endpoint final.
- **Fila limitada.** Se a fila enche (endpoint morto, event storm), novos eventos são descartados com aviso em vez de consumir memória ilimitada.
- **Sem prompt de consentimento.** Targets outbound não executam código na sua máquina — recebem dados em URL que você configurou. `HERMES_SAFE_MODE=1` ainda pula registro, igual plugins e shell hooks. Note que payloads incluem inputs de ferramentas e metadata de evento, então aponte targets só para endpoints em que confia, e prefira `https://`.

`hermes hooks list` mostra targets outbound configurados junto com shell hooks, incluindo se cada target está assinado.
