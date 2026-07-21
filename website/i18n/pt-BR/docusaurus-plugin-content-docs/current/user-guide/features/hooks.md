---
sidebar_position: 6
title: "Hooks de eventos"
description: "Execute código personalizado em pontos-chave do ciclo de vida — registre atividade, envie alertas, publique em webhooks"
---

# Hooks de eventos

O Hermes tem três sistemas de hooks que executam código personalizado em pontos-chave do ciclo de vida:

| Sistema | Registrado via | Roda em | Caso de uso |
|--------|---------------|---------|----------|
| **[Gateway hooks](#gateway-event-hooks)** | `HOOK.yaml` + `handler.py` em `~/.hermes/hooks/` | Somente gateway | Logging, alertas, webhooks |
| **[Plugin hooks](#plugin-hooks)** | `ctx.register_hook()` em um [plugin](/user-guide/features/plugins) | CLI + Gateway | Interceptação de ferramentas, métricas, guardrails |
| **[Shell hooks](#shell-hooks)** | bloco `hooks:` em `~/.hermes/config.yaml` apontando para scripts shell | CLI + Gateway | Scripts drop-in para bloqueio, auto-formatação, injeção de contexto |

Os três sistemas são non-blocking — erros em qualquer hook são capturados e registrados, nunca derrubando o agente.

## Gateway Event Hooks {#gateway-event-hooks}

Gateway hooks disparam automaticamente durante a operação do gateway (Telegram, Discord, Slack, WhatsApp, Teams) sem bloquear o pipeline principal do agente.

### Criando um hook {#creating-a-hook}

Cada hook é um diretório em `~/.hermes/hooks/` contendo dois arquivos:

```text
~/.hermes/hooks/
└── my-hook/
    ├── HOOK.yaml      # Declares which events to listen for
    └── handler.py     # Python handler function
```

#### HOOK.yaml

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
- Erros são capturados e registrados, nunca derrubando o agente

### Eventos disponíveis {#available-events}

| Evento | Quando dispara | Chaves de contexto |
|-------|---------------|--------------|
| `gateway:startup` | Processo do gateway inicia | `platforms` (lista de nomes de plataformas ativas) |
| `session:start` | Nova sessão de mensagens criada | `platform`, `user_id`, `session_id`, `session_key` |
| `session:end` | Sessão encerrada (antes do reset) | `platform`, `user_id`, `session_key` |
| `session:reset` | Usuário executou `/new` ou `/reset` | `platform`, `user_id`, `session_key` |
| `agent:start` | Agente começa a processar uma mensagem | `platform`, `user_id`, `session_id`, `message` |
| `agent:step` | Cada iteração do loop de tool-calling | `platform`, `user_id`, `session_id`, `iteration`, `tool_names` |
| `agent:end` | Agente termina o processamento | `platform`, `user_id`, `session_id`, `message`, `response` |
| `command:*` | Qualquer slash command executado | `platform`, `user_id`, `command`, `args` |

#### Correspondência wildcard {#wildcard-matching}

Handlers registrados para `command:*` disparam para qualquer evento `command:` (`command:model`, `command:reset`, etc.). Monitore todos os slash commands com uma única assinatura.

### Exemplos {#examples}

#### Alerta Telegram em tarefas longas {#telegram-alert-on-long-tasks}

Envie uma mensagem para você quando o agente levar mais de 10 passos:

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

Rastreie quais slash commands são usados:

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

### Tutorial: BOOT.md — executar checklist de startup em todo boot do gateway {#tutorial-bootmd-run-a-startup-checklist-on-every-gateway-boot}

Um padrão popular da comunidade: coloque um checklist Markdown em `~/.hermes/BOOT.md` e faça o agente executá-lo uma vez sempre que o gateway iniciar. Útil para "a cada boot, verifique falhas de cron overnight e me avise no Discord se algo falhou," ou "resuma as últimas 24h de deploy.log e poste no Slack #ops."

Este tutorial mostra como construir você mesmo como hook definido pelo usuário. O Hermes não inclui um hook BOOT.md built-in — você conecta exatamente o comportamento que quiser.

#### O que estamos construindo {#what-were-building}

1. Um arquivo em `~/.hermes/BOOT.md` com instruções de startup em linguagem natural.
2. Um gateway hook que dispara em `gateway:startup`, spawna um agente one-shot com model/credenciais resolvidos do gateway e executa as instruções do BOOT.md.
3. Uma convenção `[SILENT]` para o agente optar por não enviar mensagem quando não houver nada a reportar.

#### Passo 1: escreva seu checklist {#step-1-write-your-checklist}

Crie `~/.hermes/BOOT.md`. Escreva como se estivesse dando instruções a um assistente humano:

```markdown
# Startup Checklist

1. Run `hermes cron list` and check if any scheduled jobs failed overnight.
2. If any failed, summarize them for Discord #ops (the hook delivers your final response to its configured target).
3. Check if `/opt/app/deploy.log` has any ERROR lines from the last 24 hours. If yes, summarize them and include in the same report.
4. If nothing went wrong, reply with only `[SILENT]` so no message is sent.
```

O agente vê isso como parte do prompt, então qualquer coisa que você descrever em linguagem simples funciona — chamadas de ferramenta, comandos shell, envio de mensagens, resumir arquivos.

#### Passo 2: crie o hook {#step-2-create-the-hook}

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
- `_resolve_runtime_agent_kwargs()` resolve credenciais de provider da mesma forma que um turn normal do gateway — incluindo API keys, base URLs, tokens OAuth e credential pools.

Sem isso, um `AIAgent()` bare cai nos defaults built-in e dará 401 contra qualquer endpoint não padrão.

#### Passo 3: teste {#step-3-test-it}

Reinicie o gateway:

```bash
hermes gateway restart
```

Acompanhe os logs:

```bash
hermes logs --follow --level INFO | grep boot-md
```

Você deve ver `Running BOOT.md (N chars)` seguido de `boot-md completed: ...` (resumo do que o agente fez) ou `boot-md completed (nothing to report)` quando o agente respondeu com token de silêncio exato como `[SILENT]`.

Exclua `~/.hermes/BOOT.md` para desabilitar o checklist — o hook permanece carregado mas pula silenciosamente quando o arquivo não está lá.

#### Estendendo o padrão {#extending-the-pattern}

- **Checklists conscientes de agenda:** use `datetime.now().weekday()` nas instruções do BOOT.md ("se for segunda, também verifique o log de deploy semanal"). As instruções são texto livre, então qualquer coisa que o agente consiga raciocinar vale.
- **Vários checklists:** aponte o hook para outro arquivo (`STARTUP.md`, `MORNING.md`, etc.) e registre diretórios de hook separados para cada um.
- **Variante sem agente:** se não precisar de loop completo de agente, pule `AIAgent` e faça o handler postar notificação fixa diretamente via `httpx`. Mais barato, mais rápido e sem dependência de provider.

#### Por que isso não é built-in {#why-this-isnt-a-built-in}

Uma versão anterior do Hermes incluía isso como hook built-in e spawnava silenciosamente um agente com defaults bare a cada boot do gateway. Isso surpreendia usuários com endpoints customizados e tornava o recurso invisível para quem não sabia que estava rodando. Mantê-lo como padrão documentado — construído por você, no seu diretório de hooks — significa que você vê exatamente o que faz e opta inscrevendo os arquivos.

### Como funciona {#how-it-works}

1. Na inicialização do gateway, `HookRegistry.discover_and_load()` escaneia `~/.hermes/hooks/`
2. Cada subdiretório com `HOOK.yaml` + `handler.py` é carregado dinamicamente
3. Handlers são registrados para seus eventos declarados
4. Em cada ponto do ciclo de vida, `hooks.emit()` dispara todos os handlers correspondentes
5. Erros em qualquer handler são capturados e registrados — um hook quebrado nunca derruba o agente

:::info
Gateway hooks só disparam no **gateway** (Telegram, Discord, Slack, WhatsApp, Teams). O CLI não carrega gateway hooks. Para hooks que funcionam em todo lugar, use [plugin hooks](#plugin-hooks).
:::

## Plugin Hooks {#plugin-hooks}

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
```

**Regras gerais para todos os hooks:**

- Callbacks recebem **argumentos nomeados**. Sempre aceite `**kwargs` para compatibilidade futura — novos parâmetros podem ser adicionados em versões futuras sem quebrar seu plugin.
- Se um callback **falhar**, ele é registrado e ignorado. Outros hooks e o agente continuam normalmente. Um plugin com mau comportamento nunca pode derrubar o agente.
- Dois valores de retorno de hooks afetam o comportamento: [`pre_tool_call`](#pre_tool_call) pode **bloquear** a ferramenta, e [`pre_llm_call`](#pre_llm_call) pode **injetar contexto** na chamada LLM. Todos os outros hooks são observadores fire-and-forget.
- Callbacks observadores recebem `telemetry_schema_version` automaticamente. Quando presente, `turn_id`, `api_request_id`, `task_id`, `session_id` e `api_call_count` são campos de correlação separados. Trate `api_request_id` como identificador opaco; não analise seu formato de string.

### Referência rápida {#quick-reference}

| Hook | Dispara quando | Retorna |
|------|-----------|---------|
| [`pre_tool_call`](#pre_tool_call) | Antes de qualquer ferramenta executar | `{"action": "block", "message": str}` para vetar a chamada |
| [`post_tool_call`](#post_tool_call) | Depois que qualquer ferramenta retorna | ignorado |
| [`pre_llm_call`](#pre_llm_call) | Uma vez por turno, antes do loop de tool-calling | `{"context": str}` para prepend contexto na mensagem do usuário |
| [`post_llm_call`](#post_llm_call) | Uma vez por turno, depois do loop de tool-calling | ignorado |
| [`pre_verify`](#pre_verify) | Uma vez por turno quando o agente editou código, antes de verificar/finalizar | `{"action": "continue", "message": str}` para continuar |
| [`on_session_start`](#on_session_start) | Nova sessão criada (somente primeiro turno) | ignorado |
| [`on_session_end`](#on_session_end) | Sessão encerra | ignorado |
| [`on_session_finalize`](#on_session_finalize) | CLI/gateway desmonta sessão ativa (flush, save, stats) | ignorado |
| [`on_session_reset`](#on_session_reset) | Gateway troca por session key nova (ex.: `/new`, `/reset`) | ignorado |
| [`subagent_start`](#subagent_start) | Filho `delegate_task` foi construído e está prestes a rodar | ignorado |
| [`subagent_stop`](#subagent_stop) | Filho `delegate_task` encerrou | ignorado |
| [`pre_gateway_dispatch`](#pre_gateway_dispatch) | Gateway recebeu mensagem do usuário, antes de auth + dispatch | `{"action": "skip" \| "rewrite" \| "allow", ...}` para influenciar fluxo |
| [`pre_approval_request`](#pre_approval_request) | Decisão de aprovação solicitada, incluindo decisões auto smart-mode | ignorado |
| [`post_approval_response`](#post_approval_response) | Decisão de aprovação tomada (ou prompt expira) | ignorado |
| [`transform_tool_result`](#transform_tool_result) | Depois que ferramenta retorna, antes do resultado voltar ao modelo | `str` para substituir resultado, `None` para deixar inalterado |
| [`transform_terminal_output`](#transform_terminal_output) | Dentro da ferramenta `terminal`, antes de truncar/ANSI-strip/redact | `str` para substituir saída bruta, `None` para deixar inalterado |
| [`transform_llm_output`](#transform_llm_output) | Depois que loop de tool-calling completa, antes da resposta final ser entregue | `str` para substituir texto da resposta, `None`/vazio para deixar inalterado |

---

### `pre_tool_call`

Dispara **imediatamente antes** de toda execução de ferramenta — ferramentas built-in e de plugin.

**Assinatura do callback:**

```python
def my_callback(tool_name: str, args: dict, task_id: str, **kwargs):
```

| Parâmetro | Tipo | Descrição |
|-----------|------|-------------|
| `tool_name` | `str` | Nome da ferramenta prestes a executar (ex.: `"terminal"`, `"web_search"`, `"read_file"`) |
| `args` | `dict` | Argumentos que o modelo passou à ferramenta |
| `task_id` | `str` | Identificador de sessão/tarefa. String vazia se não definido. |

**Dispara:** Em `model_tools.py`, dentro de `handle_function_call()`, antes do handler da ferramenta rodar. Dispara uma vez por chamada — se o modelo chama 3 ferramentas em paralelo, dispara 3 vezes.

**Valor de retorno — vetar a chamada:**

```python
return {"action": "block", "message": "Reason the tool call was blocked"}
```

O agente interrompe a ferramenta com `message` como erro retornado ao modelo. A primeira diretiva de block correspondente vence (plugins Python registrados primeiro, depois shell hooks). Qualquer outro valor de retorno é ignorado, então callbacks observadores existentes continuam funcionando.

**Casos de uso:** Logging, trilhas de auditoria, contadores de chamadas de ferramenta, bloqueio de operações perigosas, rate limiting, enforcement de política por usuário.

**Exemplo — log de auditoria de chamadas de ferramenta:**

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

Dispara **imediatamente depois** que toda execução de ferramenta retorna.

**Assinatura do callback:**

```python
def my_callback(tool_name: str, args: dict, result: str, task_id: str,
                duration_ms: int, **kwargs):
```

| Parâmetro | Tipo | Descrição |
|-----------|------|-------------|
| `tool_name` | `str` | Nome da ferramenta que acabou de executar |
| `args` | `dict` | Argumentos que o modelo passou à ferramenta |
| `result` | `str` | Valor de retorno da ferramenta (sempre string JSON) |
| `task_id` | `str` | Identificador de sessão/tarefa. String vazia se não definido. |
| `duration_ms` | `int` | Quanto tempo o dispatch da ferramenta levou, em milissegundos (medido com `time.monotonic()` em torno de `registry.dispatch()`). |

**Dispara:** Em `model_tools.py`, dentro de `handle_function_call()`, depois que o handler retorna. Dispara uma vez por chamada. **Não** dispara se a ferramenta levantou exceção não tratada (o erro é capturado e retornado como JSON; `post_tool_call` dispara com essa string como `result`).

**Valor de retorno:** Ignorado.

**Casos de uso:** Logging de resultados de ferramentas, coleta de métricas, taxas de sucesso/falha, dashboards de latência, alertas de budget por ferramenta, notificações quando ferramentas específicas completam.

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

Dispara **uma vez por turno**, antes do loop de tool-calling começar. Este é o **único hook cujo valor de retorno é usado** — pode injetar contexto na mensagem do usuário do turno atual.

**Assinatura do callback:**

```python
def my_callback(session_id: str, user_message: str, conversation_history: list,
                is_first_turn: bool, model: str, platform: str, **kwargs):
```

| Parâmetro | Tipo | Descrição |
|-----------|------|-------------|
| `session_id` | `str` | Identificador único da sessão atual |
| `user_message` | `str` | Mensagem original do usuário neste turno (antes de injeção de skill) |
| `conversation_history` | `list` | Cópia da lista completa de mensagens (formato OpenAI: `[{"role": "user", "content": "..."}]`) |
| `is_first_turn` | `bool` | `True` se é o primeiro turno de sessão nova, `False` nos turnos seguintes |
| `model` | `str` | Identificador do modelo (ex.: `"anthropic/claude-sonnet-4.6"`) |
| `platform` | `str` | Onde a sessão roda: `"cli"`, `"telegram"`, `"discord"`, etc. |

**Dispara:** Em `run_agent.py`, dentro de `run_conversation()`, depois da compressão de contexto e antes do `while` principal. Dispara uma vez por `run_conversation()` (ou seja, por turno do usuário), não por chamada API dentro do loop de ferramentas.

**Valor de retorno:** Se o callback retorna dict com chave `"context"`, ou string não vazia, o texto é anexado à mensagem do usuário do turno. Retorne `None` para não injetar.

```python
# Inject context
return {"context": "Recalled memories:\n- User likes Python\n- Working on hermes-agent"}

# Plain string (equivalent)
return "Recalled memories:\n- User likes Python"

# No injection
return None
```

**Onde o contexto é injetado:** Sempre na **mensagem do usuário**, nunca no system prompt. Isso preserva o prompt cache — o system prompt permanece idêntico entre turnos, reutilizando tokens em cache. O system prompt é território do Hermes (orientação do modelo, enforcement de ferramentas, personalidade, skills). Plugins contribuem contexto junto à entrada do usuário.

Todo contexto injetado é **efêmero** — adicionado só no momento da chamada API. A mensagem original do usuário no histórico nunca é mutada, e nada é persistido no banco de sessão.

Quando **vários plugins** retornam contexto, as saídas são unidas com quebras duplas na ordem de descoberta (alfabética por nome de diretório).

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

Dispara **uma vez por turno**, depois que o loop de tool-calling completa e o agente produziu resposta final. Só dispara em turnos **bem-sucedidos** — não dispara se o turno foi interrompido.

**Assinatura do callback:**

```python
def my_callback(session_id: str, user_message: str, assistant_response: str,
                conversation_history: list, model: str, platform: str, **kwargs):
```

| Parâmetro | Tipo | Descrição |
|-----------|------|-------------|
| `session_id` | `str` | Identificador único da sessão atual |
| `user_message` | `str` | Mensagem original do usuário neste turno |
| `assistant_response` | `str` | Resposta final em texto do agente neste turno |
| `conversation_history` | `list` | Cópia da lista completa após o turno completar |
| `model` | `str` | Identificador do modelo |
| `platform` | `str` | Onde a sessão roda |

**Dispara:** Em `run_agent.py`, dentro de `run_conversation()`, depois que o loop de ferramentas sai com resposta final. Guardado por `if final_response and not interrupted` — **não** dispara se o usuário interrompe no meio do turno ou o agente atinge o limite de iterações sem resposta.

**Valor de retorno:** Ignorado.

**Casos de uso:** Sincronizar conversa com memória externa, métricas de qualidade de resposta, logging de resumos de turno, ações de follow-up.

**Exemplo — sync com memória externa:**

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

**Exemplo — rastrear tamanho de respostas:**

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

Dispara **uma vez por turno quando o agente editou código**, logo antes de finalizar (depois do guard verify-on-stop built-in). É um gate de política usuário/plugin: um callback pode manter o agente rodando — executar check, adiar, limpar diff — em vez de deixá-lo parar.

A orientação de verificação shipped do Hermes não é um hook `pre_verify` padrão. É anexada ao nudge verify-on-stop baseado em evidência quando código editado carece de evidência fresca, sem criar segundo caminho padrão de continuação. Defina `agent.verify_guidance: false` para manter esse nudge built-in enxuto.

**Assinatura do callback:**

```python
def my_callback(session_id: str, platform: str, model: str, coding: bool,
                attempt: int, final_response: str, changed_paths: list, **kwargs):
```

| Parâmetro | Tipo | Descrição |
|-----------|------|-------------|
| `session_id` | `str` | Identificador único da sessão atual |
| `platform` | `str` | Onde a sessão roda (`"cli"`, `"telegram"`, …) |
| `model` | `str` | Identificador do modelo |
| `coding` | `bool` | Se o turno está em postura de coding (workspace de código) — escopo seu hook nisso |
| `attempt` | `int` | Quantas vezes este turno já foi nudged (0 na primeira) — auto-limite nisso |
| `final_response` | `str` | Resposta que o agente está prestes a entregar |
| `changed_paths` | `list` | Arquivos editados pelo agente neste turno (ordenados, sempre não vazios aqui) |

Escopo um hook ao contexto de coding checando `coding` e torne one-shot com `attempt` (shell hooks leem ambos de `.extra`), como um hook `pre_tool_call` escopa em `tool_name` — você pode registrar vários hooks `pre_verify`, cada um disparando só onde deve.

**Dispara:** Em `agent/conversation_loop.py`, no ponto em que o agente aceitaria resposta final, logo após verify-on-stop — mas só quando editou código neste turno e pelo menos um hook `pre_verify` está registrado.

**Valor de retorno — manter o agente rodando:**

```python
return {"action": "continue", "message": "Run the formatter on your changes, then finish."}
```

A `message` é anexada como turno sintético de usuário e o loop roda de novo. O formato Stop do Claude Code (`{"decision": "block", "reason": "..."}`, onde bloquear o stop significa *continuar*) também é aceito. Diretiva sem message — ou qualquer outro retorno — deixa o turno finalizar.

**Limitado:** diretivas continue consecutivas em um turno são limitadas por `agent.max_verify_nudges` (padrão 3), então um hook que sempre diz continue nunca prende o loop. A resposta tentada fica no histórico mas não é mostrada ao usuário enquanto o agente é nudged.

**Torne idempotente:** o hook re-dispara após cada nudge, então gate em `attempt` (`if attempt: return None`) — senão só nudges até o limite.

**Casos de uso:** adiar tests/lints em iteração criativa, exigir checks verdes para certos paths, bloquear "done" até existir entrada no changelog, checklist de verificação específico do projeto.

**Exemplo — adiar checks em UI criativa, escopado + one-shot:**

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

Para orientação permanente que molda o nudge built-in de evidência faltante, use `agent.verify_guidance`. Para regras mais amplas de postura de coding que não precisam *gatear* verificação, prefira `agent.coding_instructions` em `config.yaml` — vai no coding brief sem custo de turno extra.

---

### `on_session_start`

Dispara **uma vez** quando sessão nova é criada. **Não** dispara na continuação (quando o usuário envia segunda mensagem em sessão existente).

**Assinatura do callback:**

```python
def my_callback(session_id: str, model: str, platform: str, **kwargs):
```

| Parâmetro | Tipo | Descrição |
|-----------|------|-------------|
| `session_id` | `str` | Identificador único da nova sessão |
| `model` | `str` | Identificador do modelo |
| `platform` | `str` | Onde a sessão roda |

**Dispara:** Em `run_agent.py`, dentro de `run_conversation()`, no primeiro turno de sessão nova — após o system prompt ser montado e antes do loop de ferramentas. O check é `if not conversation_history` (sem mensagens anteriores = sessão nova).

**Valor de retorno:** Ignorado.

**Casos de uso:** Inicializar estado escopado à sessão, aquecer caches, registrar sessão em serviço externo, logging de início de sessão.

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

Dispara no **final** de toda chamada `run_conversation()`, independente do resultado. Também dispara do handler de exit do CLI se o agente estava no meio do turno quando o usuário saiu.

**Assinatura do callback:**

```python
def my_callback(session_id: str, completed: bool, interrupted: bool,
                model: str, platform: str, **kwargs):
```

| Parâmetro | Tipo | Descrição |
|-----------|------|-------------|
| `session_id` | `str` | Identificador único da sessão |
| `completed` | `bool` | `True` se o agente produziu resposta final, `False` caso contrário |
| `interrupted` | `bool` | `True` se o turno foi interrompido (nova mensagem, `/stop`, ou quit) |
| `model` | `str` | Identificador do modelo |
| `platform` | `str` | Onde a sessão roda |

**Dispara:** Em dois lugares:
1. **`run_agent.py`** — no final de toda `run_conversation()`, após cleanup. Sempre dispara, mesmo se o turno deu erro.
2. **`cli.py`** — no atexit handler do CLI, mas **somente** se o agente estava mid-turn (`_agent_running=True`) no exit. Captura Ctrl+C e `/exit` durante processamento. Nesse caso, `completed=False` e `interrupted=True`.

**Valor de retorno:** Ignorado.

**Casos de uso:** Flush de buffers, fechar conexões, persistir estado de sessão, logging de duração, cleanup de recursos de `on_session_start`.

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

Dispara quando CLI ou gateway **desmonta** sessão ativa — por exemplo, `/new`, GC de sessão idle, ou quit do CLI com agente ativo. Última chance de flush de estado ligado à sessão que sai antes da identidade sumir.

**Assinatura do callback:**

```python
def my_callback(session_id: str | None, platform: str, **kwargs):
```

| Parâmetro | Tipo | Descrição |
|-----------|------|-------------|
| `session_id` | `str` ou `None` | ID da sessão que sai. Pode ser `None` se não havia sessão ativa. |
| `platform` | `str` | `"cli"` ou nome da plataforma (`"telegram"`, `"discord"`, etc.). |

**Dispara:** Em `cli.py` (em `/new` / exit do CLI) e `gateway/run.py` (quando sessão é resetada ou GC'd). Sempre em par com `on_session_reset` no gateway.

**Valor de retorno:** Ignorado.

**Casos de uso:** Persistir métricas finais antes do ID ser descartado, fechar recursos por sessão, emitir telemetria final, drenar writes enfileirados.

---

### `on_session_reset`

Dispara quando o gateway **troca por session key nova** em chat ativo — `/new`, `/reset`, `/clear`, ou adapter escolhe sessão fresca após idle. Plugins reagem ao wipe do estado sem esperar o próximo `on_session_start`.

**Assinatura do callback:**

```python
def my_callback(session_id: str, platform: str, **kwargs):
```

| Parâmetro | Tipo | Descrição |
|-----------|------|-------------|
| `session_id` | `str` | ID da nova sessão (já rotacionado para o valor fresco). |
| `platform` | `str` | Nome da plataforma de mensagens. |

**Dispara:** Em `gateway/run.py`, logo após alocar a nova session key e antes da próxima mensagem inbound. No gateway, a ordem é: `on_session_finalize(old_id)` → swap → `on_session_reset(new_id)` → `on_session_start(new_id)` no primeiro turno inbound.

**Valor de retorno:** Ignorado.

**Casos de uso:** Reset de caches por `session_id`, analytics de "session rotated", preparar bucket de estado fresco.

---

Veja o **[guia Build a Plugin](/developer-guide/plugins)** para o walkthrough completo incluindo schemas de ferramentas, handlers e padrões avançados de hooks.

---

### `subagent_start`

Dispara **uma vez por agente filho** depois que `delegate_task` construiu o `AIAgent` filho e antes de rodá-lo. Delegue uma tarefa ou batch de três — dispara uma vez por filho.

Este hook é específico do ciclo de vida delegation/subagent. Não é gate universal "antes de qualquer invocação de agente" para gateway, CLI, cron, batch, MoA ou outras execuções originadas de runners.

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
| `parent_session_id` | `str \| None` | Session ID do agente pai que delega. |
| `parent_turn_id` | `str` | Turn ID do turno pai que solicitou delegação, se disponível. |
| `parent_subagent_id` | `str \| None` | Subagent ID pai quando filho foi spawnado por outro subagent; `None` para pais top-level. |
| `child_session_id` | `str \| None` | Session ID alocado para o agente filho. |
| `child_subagent_id` | `str` | Subagent ID estável usado por observabilidade e controles de delegação. |
| `child_role` | `str` | Role efetivo do filho após política de delegação, ex.: `"leaf"` ou `"orchestrator"`. |
| `child_goal` | `str` | Goal/prompt delegado que o agente filho executará. |

**Dispara:** Em `tools/delegate_tool.py`, dentro de `_build_child_agent()`, depois que o `AIAgent` filho foi construído e anotado com metadata de identidade de subagent, e antes de `_run_single_child()` rodar o filho.

**Valor de retorno:** Ignorado. Hook observador apenas; retornar valor não bloqueia nem muta a execução do agente filho.

**Casos de uso:** Logging de criação de subagent, mapear relações pai/filho de sessão, rastrear árvores de delegação aninhadas, audit records pré-run, pré-alocar recursos de observabilidade por filho.

**Exemplo — log de criação de subagent:**

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
`subagent_start` é útil para observabilidade de delegação, mas não é hook de política bloqueante. Para bloquear delegação antes do filho ser construído, use [`pre_tool_call`](#pre_tool_call) para bloquear a chamada `delegate_task`.
:::

---

### `subagent_stop`

Dispara **uma vez por agente filho** depois que `delegate_task` termina. Uma tarefa ou batch de três — dispara uma vez por filho, serializado na thread pai.

**Assinatura do callback:**

```python
def my_callback(parent_session_id: str, child_role: str | None,
                child_summary: str | None, child_status: str,
                duration_ms: int, **kwargs):
```

| Parâmetro | Tipo | Descrição |
|-----------|------|-------------|
| `parent_session_id` | `str` | Session ID do agente pai que delega |
| `child_role` | `str \| None` | Tag de role orchestrator no filho (`None` se feature não está habilitada) |
| `child_summary` | `str \| None` | Resposta final que o filho retornou ao pai |
| `child_status` | `str` | `"completed"`, `"failed"`, `"interrupted"` ou `"error"` |
| `duration_ms` | `int` | Tempo de relógio rodando o filho, em milissegundos |

**Dispara:** Em `tools/delegate_tool.py`, depois que `ThreadPoolExecutor.as_completed()` drena todos os futures filhos. Disparo é marshalled para a thread pai para autores não precisarem raciocinar sobre callbacks concorrentes.

**Valor de retorno:** Ignorado.

**Casos de uso:** Logging de atividade de orquestração, acumular durações de filhos para billing, audit records pós-delegação.

**Exemplo — log de atividade do orchestrator:**

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
Com delegação pesada (ex.: roles orchestrator × 5 leaves × profundidade aninhada), `subagent_stop` dispara muitas vezes por turno. Mantenha callback rápido; empurre trabalho caro para fila em background.
:::

---

### `pre_gateway_dispatch`

Dispara **uma vez por `MessageEvent` inbound** no gateway, depois do guard de evento interno mas **antes** de auth/pairing e dispatch do agente. Ponto de interceptação para políticas de fluxo de mensagem no gateway (janelas listen-only, handover humano, roteamento por chat, etc.) que não cabem em um único adapter de plataforma.

**Assinatura do callback:**

```python
def my_callback(event, gateway, session_store, **kwargs):
```

| Parâmetro | Tipo | Descrição |
|-----------|------|-------------|
| `event` | `MessageEvent` | Mensagem inbound normalizada (tem `.text`, `.source`, `.message_id`, `.internal`, etc.). |
| `gateway` | `GatewayRunner` | Gateway runner ativo, para plugins chamarem `gateway.adapters[platform].send(...)` em respostas side-channel (notificações ao owner, etc.). |
| `session_store` | `SessionStore` | Para ingestão silenciosa de transcript via `session_store.append_to_transcript(...)`. |

**Dispara:** Em `gateway/run.py`, dentro de `GatewayRunner._handle_message()`, logo após `is_internal` ser calculado. **Eventos internos pulam o hook por completo** (são gerados pelo sistema — conclusões de processos em background, etc. — e não devem ser gate-kept por política user-facing).

**Valor de retorno:** `None` ou dict. O primeiro dict de action reconhecido vence; resultados restantes de plugins são ignorados. Exceções em callbacks são capturadas e logadas; o gateway sempre faz fall-through para dispatch normal em erro.

| Retorno | Efeito |
|--------|--------|
| `{"action": "skip", "reason": "..."}` | Descarta a mensagem — sem resposta do agente, sem pairing, sem auth. Assume-se que o plugin tratou (ex.: ingestão silenciosa no transcript). |
| `{"action": "rewrite", "text": "new text"}` | Substitui `event.text`, depois continua dispatch normal com evento modificado. Útil para colapsar mensagens ambient bufferizadas em um prompt. |
| `{"action": "allow"}` / `None` | Dispatch normal — roda cadeia completa auth / pairing / agent-loop. |

**Casos de uso:** Chats de grupo listen-only (responder só quando marcado; bufferizar mensagens ambient no contexto); handover humano (ingestão silenciosa enquanto owner trata manualmente); rate limiting por profile; roteamento dirigido por política.

**Exemplo — descartar DMs não autorizados silenciosamente sem pairing code:**

```python
def deny_unauthorized_dms(event, **kwargs):
    src = event.source
    if src.chat_type == "dm" and not _is_approved_user(src.user_id):
        return {"action": "skip", "reason": "unauthorized-dm"}
    return None

def register(ctx):
    ctx.register_hook("pre_gateway_dispatch", deny_unauthorized_dms)
```

**Exemplo — reescrever buffer de mensagens ambient em prompt único ao mencionar:**

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

### `pre_approval_request`

Dispara antes de decisão de aprovação ser solicitada. Cobre superfícies prompted — CLI interativo, Ink TUI, plataformas gateway e clientes ACP — e decisões `approvals.mode=smart` sem prompt humano (`surface="smart"`). Em smart mode, o hook roda antes do LLM auxiliar ser chamado.

É o lugar certo para conectar notificador customizado — por exemplo, app de menu-bar macOS com notificação allow/deny, ou audit log que registra cada pedido de aprovação com contexto.

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
| `command` | `str` | Comando terminal ou script `execute_code` sendo avaliado. Payloads smart e gateway são redacted antes do dispatch observador. Redaction smart é obrigatória mesmo com `security.redact_secrets` desabilitado; se falhar, hooks smart são pulados. |
| `description` | `str` | Motivo(s) legíveis pelos quais o comando foi flagged (combinados quando vários patterns batem) |
| `pattern_key` | `str` | Pattern key primária que disparou aprovação (ex.: `"rm_rf"`, `"sudo"`) |
| `pattern_keys` | `list[str]` | Todas as pattern keys que bateram |
| `session_key` | `str` | Identificador de sessão, útil para escopar notificações por chat |
| `surface` | `str` | `"cli"` para prompts CLI/TUI interativos, `"gateway"` para aprovações async de plataforma, ou `"smart"` para decisões auto approve/deny do LLM auxiliar |

**Valor de retorno:** ignorado. Hooks aqui são observadores; não podem vetar nem pré-responder aprovação. Use [`pre_tool_call`](#pre_tool_call) para bloquear ferramenta antes do sistema de aprovação.

**Casos de uso:** Notificações desktop, push alerts, audit logging, Slack webhooks, roteamento de escalation, métricas.

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

Dispara depois de decisão prompted ou smart (ou depois que prompt expira).

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

Mesmos kwargs de `pre_approval_request`, mais:

| Parâmetro | Tipo | Descrição |
|-----------|------|-------------|
| `choice` | `str` | Superfícies prompted usam `"once"`, `"session"`, `"always"`, `"deny"` ou `"timeout"`; decisões smart usam `"smart_approve"` ou `"smart_deny"` |
| `decided_by` | `str` | `"aux_llm"` para decisões smart; ausente em superfícies prompted |

**Valor de retorno:** ignorado.

**Casos de uso:** Fechar notificação desktop correspondente, registrar decisão final em audit log, atualizar métricas, avançar rate limiter.

```python
def log_decision(command, choice, session_key, **kwargs):
    logger.info("approval %s: %s for session %s", choice, command[:60], session_key)

def register(ctx):
    ctx.register_hook("post_approval_response", log_decision)
```

---

### `transform_tool_result`

Dispara **depois** que ferramenta retorna e **antes** do resultado ser anexado à conversa. Permite reescrever string de resultado de QUALQUER ferramenta — não só saída de terminal — antes do modelo ver.

**Assinatura do callback:**

```python
def my_callback(
    tool_name: str,
    arguments: dict,
    result: str,
    task_id: str | None,
    **kwargs,
) -> str | None:
```

| Parâmetro | Tipo | Descrição |
|-----------|------|-------------|
| `tool_name` | `str` | Ferramenta que produziu o resultado (`read_file`, `web_extract`, `delegate_task`, …). |
| `arguments` | `dict` | Argumentos com que o modelo chamou a ferramenta. |
| `result` | `str` | String bruta de resultado da ferramenta, pós-truncation e pós-ANSI-strip. |
| `task_id` | `str \| None` | Task/session ID ao rodar em ambientes RL/benchmark. |

**Valor de retorno:** `str` para substituir resultado (a string retornada é o que o modelo vê), `None` para deixar inalterado.

**Casos de uso:** Redact PII específico da organização em saída `web_extract`, envolver respostas JSON longas em header de resumo, injetar hints RAG em resultados `read_file`, reescrever reports de subagent `delegate_task` em schema do projeto.

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

Aplica a toda ferramenta. Para rewrite só de terminal veja `transform_terminal_output` abaixo — é mais estreito e roda mais cedo no pipeline (pré-truncation, pré-redaction).

---

### `transform_terminal_output`

Dispara dentro do pipeline de saída foreground da ferramenta `terminal`, **antes** da truncation padrão de 50 KB, ANSI strip e redaction de secrets. Plugins reescrevem stdout/stderr bruto antes de processamento downstream.

**Assinatura do callback:**

```python
def my_callback(
    command: str,
    output: str,
    exit_code: int,
    cwd: str,
    task_id: str | None,
    **kwargs,
) -> str | None:
```

| Parâmetro | Tipo | Descrição |
|-----------|------|-------------|
| `command` | `str` | Comando shell que produziu a saída. |
| `output` | `str` | stdout/stderr combinado bruto (pode ser muito grande — truncation ocorre depois do hook). |
| `exit_code` | `int` | Exit code do processo. |
| `cwd` | `str` | Working directory em que o comando rodou. |

**Valor de retorno:** `str` para substituir saída, `None` para deixar inalterado.

**Casos de uso:** Injetar resumos para comandos com saída massiva (`du -ah`, `find`, `tree`), marcar saída com marker do projeto para hooks downstream, remover ruído de timing que flutua entre runs e quebra prompt caching.

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

Combina bem com `transform_tool_result` (que cobre todas as outras ferramentas).

---

### `transform_llm_output`

Dispara **uma vez por turno** depois que loop de tool-calling completa e modelo produziu resposta final, **antes** de entregar ao usuário (CLI, gateway ou caller programático). Plugin reescreve texto final do assistant com programação clássica — sem tokens extras de inferência em flavor text SOUL ou transform driven por skill.

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
| `response_text` | `str` | Texto final de resposta do assistant neste turno. |
| `session_id` | `str` | Session ID desta conversa (pode ser vazio em runs one-shot). |
| `model` | `str` | Nome do modelo que produziu a resposta (ex.: `anthropic/claude-sonnet-4.6`). |
| `platform` | `str` | Plataforma de entrega (`cli`, `telegram`, `discord`, …; vazio se unset). |

**Valor de retorno:** `str` não vazia para substituir texto da resposta, `None` ou string vazia para deixar inalterado. **Primeira string não vazia vence** com vários plugins — espelhando `transform_tool_result`.

**Casos de uso:** Aplicar transform de personalidade/vocabulário (pirate-speak, Spongebob), redact identificadores do usuário no texto final, anexar footer de assinatura do projeto, impor style guide sem queimar tokens em instruções SOUL.

```python
import os, re

def spongebob(response_text, **kwargs):
    if os.environ.get("SPONGEBOB_MODE") != "on":
        return None  # pass through unchanged
    return re.sub(r"!", "!! Tartar sauce!", response_text)

def register(ctx):
    ctx.register_hook("transform_llm_output", spongebob)
```

O hook é guardado em resposta não vazia e não interrompida — não dispara em interrupts de stop ou turnos vazios. Exceções são logadas como warnings e não quebram execução do agente.

---

## Shell hooks {#shell-hooks}

Declare shell-script hooks no seu `cli-config.yaml` e o Hermes os executará como subprocessos sempre que o evento plugin-hook correspondente disparar — em sessões CLI e gateway. Não exige authoring de plugin Python.

Use shell hooks quando quiser um script drop-in de arquivo único (Bash, Python, qualquer coisa com shebang) para:

- **Bloquear uma chamada de ferramenta** — rejeitar comandos `terminal` perigosos, impor políticas por diretório, exigir aprovação para `write_file` / `patch` destrutivos.
- **Rodar após uma chamada de ferramenta** — auto-formatar arquivos Python ou TypeScript que o agente acabou de escrever, registrar chamadas API, disparar workflow CI.
- **Injetar contexto no próximo turno LLM** — prepend saída de `git status`, dia da semana atual ou documentos recuperados na mensagem do usuário (veja [`pre_llm_call`](#pre_llm_call)).
- **Observar eventos de ciclo de vida** — escrever linha de log quando um subagente completa (`subagent_stop`) ou uma sessão inicia (`on_session_start`).

Shell hooks são registrados chamando `agent.shell_hooks.register_from_config(cfg)` tanto na inicialização do CLI (`hermes_cli/main.py`) quanto do gateway (`gateway/run.py`). Compõem naturalmente com plugin hooks Python — ambos fluem pelo mesmo dispatcher.

### Comparação rápida {#comparison-at-a-glance}

| Dimensão | Shell hooks | [Plugin hooks](#plugin-hooks) | [Gateway hooks](#gateway-event-hooks) |
|-----------|-------------|-------------------------------|---------------------------------------|
| Declarado em | bloco `hooks:` em `~/.hermes/config.yaml` | `register()` em plugin `plugin.yaml` | diretório `HOOK.yaml` + `handler.py` |
| Vive em | `~/.hermes/agent-hooks/` (por convenção) | `~/.hermes/plugins/<name>/` | `~/.hermes/hooks/<name>/` |
| Linguagem | Qualquer (Bash, Python, binário Go, …) | Só Python | Só Python |
| Roda em | CLI + Gateway | CLI + Gateway | Só gateway |
| Eventos | `VALID_HOOKS` (incl. `subagent_stop`) | `VALID_HOOKS` | Ciclo de vida gateway (`gateway:startup`, `agent:*`, `command:*`) |
| Pode bloquear chamada de ferramenta | Sim (`pre_tool_call`) | Sim (`pre_tool_call`) | Não |
| Pode injetar contexto LLM | Sim (`pre_llm_call`) | Sim (`pre_llm_call`) | Não |
| Consentimento | Prompt no primeiro uso por par `(event, command)` | Implícito (confiança no plugin Python) | Implícito (confiança no diretório) |
| Isolamento inter-processo | Sim (subprocess) | Não (in-process) | Não (in-process) |

### Schema de configuração {#configuration-schema}

```yaml
hooks:
  <event_name>:                  # Must be in VALID_HOOKS
    - matcher: "<regex>"         # Optional; used for pre/post_tool_call only
      command: "<shell command>" # Required; runs via shlex.split, shell=False
      timeout: <seconds>         # Optional; default 60, capped at 300

hooks_auto_accept: false         # See "Consent model" below
```

Nomes de evento devem ser um dos [eventos de plugin hook](#plugin-hooks); typos produzem aviso "Did you mean X?" e são ignorados. Chaves desconhecidas dentro de uma entrada são ignoradas; `command` ausente é skip-with-warning. `timeout > 300` é limitado com aviso.

### Protocolo wire JSON {#json-wire-protocol}

Cada vez que o evento dispara, o Hermes cria um subprocesso para cada hook correspondente (matcher permitindo), envia payload JSON para **stdin** e lê **stdout** de volta como JSON.

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

`tool_name` e `tool_input` são `null` para eventos que não são de ferramenta (`pre_llm_call`, `subagent_stop`, ciclo de vida de sessão). O dict `extra` carrega todos os kwargs específicos do evento (`user_message`, `conversation_history`, `child_role`, `duration_ms`, …). Valores não serializáveis viram string em vez de serem omitidos.

**stdout — optional response:**

```jsonc
// Block a pre_tool_call (both shapes accepted; normalised internally):
{"decision": "block", "reason":  "Forbidden: rm -rf"}   // Claude-Code style
{"action":   "block", "message": "Forbidden: rm -rf"}   // Hermes-canonical

// Inject context for pre_llm_call:
{"context": "Today is Friday, 2026-04-17"}

// Keep the agent going at the verify gate (pre_verify); both shapes accepted:
{"action": "continue", "message": "Run the formatter, then finish."}
{"decision": "block",  "reason":  "Run the formatter, then finish."}

// Silent no-op — any empty / non-matching output is fine:
```

JSON malformado, exit codes não zero e timeouts registram aviso mas nunca abortam o loop do agente.

### Exemplos práticos {#worked-examples}

#### 1. Auto-formatar arquivos Python após cada write

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

A visão in-context do agente sobre o arquivo **não** é relida automaticamente — o reformat afeta só o arquivo no disco. Chamadas subsequentes de `read_file` pegam a versão formatada.

#### 2. Bloquear comandos `terminal` destrutivos

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

#### 3. Injetar `git status` em todo turno (equivalente Claude-Code `UserPromptSubmit`)

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

O evento `UserPromptSubmit` do Claude Code intencionalmente não é um evento Hermes separado — `pre_llm_call` dispara no mesmo lugar e já suporta injeção de contexto. Use-o aqui.

#### 4. Registrar toda conclusão de subagente

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

Cada par único `(event, command)` solicita aprovação do usuário na primeira vez que o Hermes o vê, depois persiste a decisão em `~/.hermes/shell-hooks-allowlist.json`. Execuções subsequentes (CLI ou gateway) pulam o prompt.

Três escape hatches contornam o prompt interativo — qualquer um basta:

1. Flag `--accept-hooks` no CLI (ex.: `hermes --accept-hooks chat`)
2. Variável de ambiente `HERMES_ACCEPT_HOOKS=1`
3. `hooks_auto_accept: true` em `cli-config.yaml`

Execuções non-TTY (gateway, cron, CI) precisam de um desses três — caso contrário qualquer hook recém-adicionado fica silenciosamente não registrado e registra aviso.

**Edições de script são confiadas silenciosamente.** A allowlist usa a string exata do comando, não o hash do script, então editar o script no disco não invalida consentimento. `hermes hooks doctor` sinaliza drift de mtime para você notar edições e decidir se reaprova.

#### Allowlist manual {#manual-allowlisting}

Allowlist manual é útil para deploys non-TTY ou service-account onde um operador não pode responder o prompt de primeiro uso interativamente. O arquivo allowlist é `~/.hermes/shell-hooks-allowlist.json`, e o formato esperado é um array `approvals`. Cada aprovação registra o `event` do hook e a string exata de `command`:

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

A string de comando deve corresponder exatamente ao comando do hook configurado. Objeto keyed por path com campo `sha256` não é o formato esperado e não aprovará o hook. Verifique entradas manuais com `hermes hooks list`.

### A CLI `hermes hooks` {#the-hermes-hooks-cli}

| Comando | O que faz |
|---------|--------------|
| `hermes hooks list` | Lista hooks configurados com matcher, timeout e status de consentimento |
| `hermes hooks test <event> [--for-tool X] [--payload-file F]` | Dispara todo hook correspondente contra payload sintético e imprime resposta parseada |
| `hermes hooks revoke <command>` | Remove toda entrada allowlist correspondente a `<command>` (efeito no próximo restart) |
| `hermes hooks doctor` | Para cada hook configurado: verifica exec bit, status allowlist, drift mtime, validade JSON de saída e tempo de execução aproximado |

### Segurança {#security}

Shell hooks rodam com **suas credenciais completas de usuário** — mesma fronteira de confiança que entrada cron ou alias shell. Trate o bloco `hooks:` em `config.yaml` como configuração privilegiada:

- Referencie só scripts que você escreveu ou revisou por completo.
- Mantenha scripts dentro de `~/.hermes/agent-hooks/` para o caminho ser fácil de auditar.
- Reexecute `hermes hooks doctor` depois de puxar config compartilhada para notar hooks recém-adicionados antes de registrarem.
- Se seu config.yaml é versionado em equipe, revise PRs que mudam a seção `hooks:` da mesma forma que revisaria config CI.

### Ordem e precedência {#ordering-and-precedence}

Tanto plugin hooks Python quanto shell hooks fluem pelo mesmo dispatcher `invoke_hook()`. Plugins Python são registrados primeiro (`discover_and_load()`), shell hooks segundo (`register_from_config()`), então decisões de block `pre_tool_call` Python têm precedência em empates. O primeiro block válido vence — o agregador retorna assim que qualquer callback produz `{"action": "block", "message": str}` com message não vazia.
