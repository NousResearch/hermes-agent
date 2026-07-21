---
sidebar_position: 14
title: "API Server"
description: "Exponha o hermes-agent como API compatível com OpenAI para qualquer frontend"
---

# API Server {#api-server}

O API server expõe o hermes-agent como endpoint HTTP compatível com OpenAI. Qualquer frontend que fale o formato OpenAI — Open WebUI, LobeChat, LibreChat, NextChat, ChatBox e centenas de outros — pode conectar ao hermes-agent e usá-lo como backend.

Seu agente trata requisições com seu toolset completo (terminal, operações de arquivo, web search, memória, skills) e retorna a resposta final. Com streaming, indicadores de progresso de ferramentas aparecem inline para frontends mostrarem o que o agente está fazendo.

:::tip Um backend cobre models + tools
O Hermes em si precisa de provider configurado e backends de ferramentas para o API server ser útil. Uma assinatura [Nous Portal](/user-guide/features/tool-gateway) cobre ambos — 300+ models plus web/image/TTS/browser via Tool Gateway. Rode `hermes setup --portal` uma vez antes de iniciar o API server e frontends como Open WebUI ou LobeChat recebem um backend totalmente equipado com ferramentas.
:::

## Início rápido {#quick-start}

### 1. Habilite o API server {#1-enable-the-api-server}

Adicione em `~/.hermes/.env`:

```bash
API_SERVER_ENABLED=true
API_SERVER_KEY=change-me-local-dev
# Optional: only if a browser must call Hermes directly
# API_SERVER_CORS_ORIGINS=http://localhost:3000
```

### 2. Inicie o gateway {#2-start-the-gateway}

```bash
hermes gateway
```

Você verá:

```
[API Server] API server listening on http://127.0.0.1:8642
```

### 3. Conecte um frontend {#3-connect-a-frontend}

Aponte qualquer client compatível com OpenAI para `http://localhost:8642/v1`:

```bash
# Test with curl
curl http://localhost:8642/v1/chat/completions \
  -H "Authorization: Bearer change-me-local-dev" \
  -H "Content-Type: application/json" \
  -d '{"model": "hermes-agent", "messages": [{"role": "user", "content": "Hello!"}]}'
```

Ou conecte Open WebUI, LobeChat ou outro frontend — veja o [guia de integração Open WebUI](/user-guide/messaging/open-webui) para instruções passo a passo.

## Endpoints {#endpoints}

### POST /v1/chat/completions {#post-v1chatcompletions}

Formato padrão OpenAI Chat Completions. Stateless — a conversa completa é incluída em cada requisição via array `messages`.

**Request:**
```json
{
  "model": "hermes-agent",
  "messages": [
    {"role": "system", "content": "You are a Python expert."},
    {"role": "user", "content": "Write a fibonacci function"}
  ],
  "stream": false
}
```

**Response:**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1710000000,
  "model": "hermes-agent",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Here's a fibonacci function..."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 50, "completion_tokens": 200, "total_tokens": 250}
}
```

**Input de imagem inline:** mensagens user podem enviar `content` como array de partes `text` e `image_url`. URLs remotas `http(s)` e URLs `data:image/...` são suportadas:

```json
{
  "model": "hermes-agent",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.png", "detail": "high"}}
      ]
    }
  ]
}
```

Arquivos uploaded (`file` / `input_file` / `file_id`) e URLs `data:` não-imagem retornam `400 unsupported_content_type`.

**Streaming** (`"stream": true`): Retorna Server-Sent Events (SSE) com chunks de resposta token a token. Para **Chat Completions**, o stream usa eventos padrão `chat.completion.chunk` plus evento customizado `hermes.tool.progress` do Hermes para UX de tool-start. Para **Responses**, o stream usa tipos de evento OpenAI Responses como `response.created`, `response.output_text.delta`, `response.output_item.added`, `response.output_item.done` e `response.completed`.

**Progresso de ferramentas em streams**:
- **Chat Completions**: Hermes emite `event: hermes.tool.progress` para visibilidade de tool-start sem poluir texto assistant persistido.
- **Responses**: Hermes emite output items spec-native `function_call` e `function_call_output` durante o SSE stream, para clients renderizarem UI estruturada de ferramentas em tempo real.

### POST /v1/responses {#post-v1responses}

Formato OpenAI Responses API. Suporta estado de conversa server-side via `previous_response_id` — o servidor armazena histórico completo de conversa (incluindo tool calls e results) para contexto multi-turn ser preservado sem o client gerenciá-lo.

**Request:**
```json
{
  "model": "hermes-agent",
  "input": "What files are in my project?",
  "instructions": "You are a helpful coding assistant.",
  "store": true
}
```

**Response:**
```json
{
  "id": "resp_abc123",
  "object": "response",
  "status": "completed",
  "model": "hermes-agent",
  "output": [
    {"type": "function_call", "name": "terminal", "arguments": "{\"command\": \"ls\"}", "call_id": "call_1"},
    {"type": "function_call_output", "call_id": "call_1", "output": "README.md src/ tests/"},
    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Your project has..."}]}
  ],
  "usage": {"input_tokens": 50, "output_tokens": 200, "total_tokens": 250}
}
```

**Input de imagem inline:** `input[].content` pode conter partes `input_text` e `input_image`. URLs remotas e URLs `data:image/...` são suportadas:

```json
{
  "model": "hermes-agent",
  "input": [
    {
      "role": "user",
      "content": [
        {"type": "input_text", "text": "Describe this screenshot."},
        {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0K..."}
      ]
    }
  ]
}
```

Arquivos uploaded (`input_file` / `file_id`) e URLs `data:` não-imagem retornam `400 unsupported_content_type`.

#### Multi-turn com previous_response_id {#multi-turn-with-previous_response_id}

Encadeie responses para manter contexto completo (incluindo tool calls) entre turns:

```json
{
  "input": "Now show me the README",
  "previous_response_id": "resp_abc123"
}
```

O servidor reconstrói a conversa completa da cadeia de responses armazenada — todos os tool calls e results anteriores são preservados. Requisições encadeadas também compartilham a mesma sessão, então conversas multi-turn aparecem como uma entrada no dashboard e histórico de sessão.

#### Conversas nomeadas {#named-conversations}

Use o parâmetro `conversation` em vez de rastrear response IDs:

```json
{"input": "Hello", "conversation": "my-project"}
{"input": "What's in src/?", "conversation": "my-project"}
{"input": "Run the tests", "conversation": "my-project"}
```

O servidor encadeia automaticamente para a response mais recente nessa conversa. Como o comando `/title` para sessões gateway.

### GET /v1/responses/\{id\} {#get-v1responsesid}

Recupera uma response armazenada anteriormente por ID.

### DELETE /v1/responses/\{id\} {#delete-v1responsesid}

Deleta uma response armazenada.

### GET /v1/models {#get-v1models}

Lista o agente como model disponível. O nome de model anunciado usa por padrão o nome do [profile](/user-guide/profiles) (ou `hermes-agent` para o profile default). Obrigatório para a maioria dos frontends para descoberta de model.

### GET /v1/capabilities {#get-v1capabilities}

Retorna descrição legível por máquina da superfície estável do API server para UIs externas, orchestrators e plugin bridges.

```json
{
  "object": "hermes.api_server.capabilities",
  "platform": "hermes-agent",
  "model": "hermes-agent",
  "auth": {"type": "bearer", "required": true},
  "features": {
    "chat_completions": true,
    "responses_api": true,
    "run_submission": true,
    "run_status": true,
    "run_events_sse": true,
    "run_stop": true
  }
}
```

Use este endpoint ao integrar dashboards, browser UIs ou planos de controle para descobrirem se a versão Hermes em execução suporta runs, streaming, cancelamento e continuidade de sessão sem depender de internals Python privados.

### GET /health {#get-health}

Health check. Retorna `{"status": "ok"}`. Também disponível em **GET /v1/health** para clients compatíveis com OpenAI que esperam o prefixo `/v1/`.

### GET /health/detailed {#get-healthdetailed}

Health check de readiness autenticado para monitoramento e planos de controle. Reporta
status limitado para config do profile ativo, banco de estado, model
configurado, espaço em disco, estado gateway/platform, runs API ativas, process completions
pendentes e delegations ativas. A resposta expõe status e contagens,
não valores de config, credenciais, paths, commands, payloads de fila ou erros raw.

A rota pública `/health` permanece probe de liveness barato e não roda
checks de readiness. Resultado de readiness degraded ainda usa HTTP 200; inspecione
`status` top-level e campos `readiness.checks`.

## Runs API (alternativa amigável a streaming) {#runs-api-streaming-friendly-alternative}

Além de `/v1/chat/completions` e `/v1/responses`, o servidor expõe uma **runs** API para sessões long-form onde o client quer assinar eventos de progresso em vez de gerenciar streaming.

### POST /v1/runs {#post-v1runs}

Cria uma nova run de agente. Retorna `run_id` usável para assinar eventos de progresso.

```json
{
  "run_id": "run_abc123",
  "status": "started"
}
```

Runs aceitam string `input` simples e `session_id`, `instructions`, `conversation_history` ou `previous_response_id` opcionais. Quando `session_id` é fornecido, Hermes o expõe no status da run para UIs externas correlacionarem runs com seus próprios conversation IDs.

### GET /v1/runs/\{run_id\} {#get-v1runsrun_id}

Poll do estado atual da run. Útil para dashboards que precisam de status sem manter conexão SSE aberta, ou UIs que reconectam após navegação.

```json
{
  "object": "hermes.run",
  "run_id": "run_abc123",
  "status": "completed",
  "session_id": "space-session",
  "model": "hermes-agent",
  "output": "Done.",
  "usage": {"input_tokens": 50, "output_tokens": 200, "total_tokens": 250}
}
```

Statuses são retidos brevemente após estados terminais (`completed`, `failed` ou `cancelled`) para polling e reconciliação de UI.

### GET /v1/runs/\{run_id\}/events {#get-v1runsrun_idevents}

Stream Server-Sent Events de progresso de tool-call, token deltas e eventos de lifecycle da run. Projetado para dashboards e thick clients que querem attach/detach sem perder estado.

Buffers de eventos não consumidos expiram após cinco minutos para um client detached não
crescer memória indefinidamente. Isso expira só estado de transporte: uma run que ainda
está executando permanece visível para status polling, approval, stop control e
accounting de concorrência até seu trabalho executor sair de fato. Um subscriber SSE conectado
continua drenando normalmente.

### POST /v1/runs/\{run_id\}/stop {#post-v1runsrun_idstop}

Interrompe um turn de agente em execução. O endpoint retorna imediatamente com `{"status": "stopping"}` enquanto Hermes pede ao agente ativo para parar no próximo ponto seguro de interrupção.
A run permanece tracked como `stopping` até o trabalho backed pelo executor sair, depois
assenta como `cancelled`; pedir stop nunca esconde um worker que ainda está
rodando.

### POST /v1/runs/\{run_id\}/approval {#post-v1runsrun_idapproval}

Resolve approval pendente para run aguardando decisão humana (ex.: tool call gated por política de approval). O body carrega a decisão de approval; a run retoma quando a decisão é registrada. Este endpoint é anunciado em `/v1/capabilities` como feature `run_approval` para UIs externas detectarem suporte antes de mostrar prompt de approval.

## Jobs API (trabalho agendado em background) {#jobs-api-background-scheduled-work}

O servidor expõe superfície CRUD leve de jobs para gerenciar runs de agente agendadas/background de um client remoto. Todos os endpoints são gated pela mesma auth bearer.

### GET /api/jobs {#get-apijobs}

Lista todos os jobs agendados.

### POST /api/jobs {#post-apijobs}

Cria novo job agendado. Body aceita a mesma forma que `hermes cron` — prompt, schedule, skills, provider override, delivery target.

### GET /api/jobs/\{job_id\} {#get-apijobsjob_id}

Busca definição e estado last-run de um job.

### PATCH /api/jobs/\{job_id\} {#patch-apijobsjob_id}

Atualiza campos de job existente (prompt, schedule, etc.). Partial updates são mergeados.

### DELETE /api/jobs/\{job_id\} {#delete-apijobsjob_id}

Remove um job. Também cancela qualquer run in-flight.

### POST /api/jobs/\{job_id\}/pause {#post-apijobsjob_idpause}

Pausa job sem deletar. Timestamps next-scheduled-run ficam suspensos até resume.

### POST /api/jobs/\{job_id\}/resume {#post-apijobsjob_idresume}

Retoma job previamente pausado.

### POST /api/jobs/\{job_id\}/run {#post-apijobsjob_idrun}

Dispara job para rodar imediatamente, fora do schedule.

## Sessions API (controle de sessão over REST) {#sessions-api-session-control-over-rest}

UIs externas podem gerenciar sessões Hermes over REST sem subir o dashboard. Todos os endpoints são gated por `API_SERVER_KEY` e vivem sob `/api/sessions/*`.

| Method | Path | Descrição |
|--------|------|-------------|
| `GET` | `/api/sessions` | Lista sessões (paginado — `limit`, `offset`, `source`, `include_children`) |
| `POST` | `/api/sessions` | Cria sessão vazia |
| `GET` | `/api/sessions/{id}` | Lê metadata de sessão |
| `PATCH` | `/api/sessions/{id}` | Atualiza title ou `end_reason` |
| `DELETE` | `/api/sessions/{id}` | Deleta sessão |
| `GET` | `/api/sessions/{id}/messages` | Histórico de mensagens de sessão |
| `POST` | `/api/sessions/{id}/fork` | Branch da sessão via linhagem `SessionDB` (semântica igual CLI `/branch`) |
| `POST` | `/api/sessions/{id}/chat` | Roda um turn síncrono de agente |
| `POST` | `/api/sessions/{id}/chat/stream` | Wrapper SSE sobre um turn — emite eventos `assistant.delta`, `tool.started`, `tool.completed`, `run.completed` |

`/v1/capabilities` anuncia a superfície completa via feature flags `session_*` e entradas `endpoints.session_*` para UIs externas detectarem suporte e fazer fallback com segurança. Imagens inline são suportadas em payloads `chat` e `chat/stream` (path multimodal-aware).

```bash
# fork a session and run one turn
curl -X POST http://localhost:8642/api/sessions/$ID/fork \
  -H "Authorization: Bearer $API_SERVER_KEY" \
  -d '{"title": "explore alt path"}'

# stream a turn over SSE
curl -N -X POST http://localhost:8642/api/sessions/$ID/chat/stream \
  -H "Authorization: Bearer $API_SERVER_KEY" \
  -d '{"input": "what files changed in the last hour?"}'
```

## Descoberta de skills e toolsets {#skills-and-toolsets-discovery}

`GET /v1/skills` e `GET /v1/toolsets` deixam clients externos enumerarem capacidades do agente deterministicamente over REST em vez de perguntar ao model. Ambos são read-only e gated por `API_SERVER_KEY`.

```bash
curl http://localhost:8642/v1/skills \
  -H "Authorization: Bearer $API_SERVER_KEY"
# → [{"name": "github-pr-workflow", "description": "...", "category": "..."}, ...]

curl http://localhost:8642/v1/toolsets \
  -H "Authorization: Bearer $API_SERVER_KEY"
# → [{"name": "core", "label": "...", "description": "...", "enabled": true,
#     "configured": true, "tools": ["read_file", "write_file", ...]}, ...]
```

`/v1/skills` retorna a mesma metadata que o skills hub usa internamente. `/v1/toolsets` retorna toolsets resolvidos para plataforma `api_server` com a lista concreta `tools` que cada um expande. Ambos são anunciados sob `endpoints.*` em `/v1/capabilities`.

## Escopo de memória de longo prazo (`X-Hermes-Session-Key`) {#long-term-memory-scoping-x-hermes-session-key}

Frontends multi-usuário como Open WebUI precisam de identificador estável por canal para memória de longo prazo (Honcho, etc.) que é **independente** do `X-Hermes-Session-Id` scoped ao transcript (que rota em `/new`). Passe `X-Hermes-Session-Key` em `/v1/chat/completions`, `/v1/responses` ou `/v1/runs` e Hermes o threaded through para `AIAgent(gateway_session_key=...)`, onde o memory provider Honcho o usa para derivar scope estável.

```http
POST /v1/chat/completions HTTP/1.1
Authorization: Bearer ***
X-Hermes-Session-Id: transcript-alpha
X-Hermes-Session-Key: agent:main:webui:dm:user-42
```

Regras: max 256 chars, caracteres de controle (`\r`, `\n`, `\x00`) são rejeitados, e o valor é ecoado de volta nas responses (JSON + SSE). `/v1/capabilities` anuncia suporte via `"session_key_header": "X-Hermes-Session-Key"`. Sem a key, estratégia `per-session` do Honcho produz scope diferente por `session_id` — exatamente o comportamento que Hermes tinha antes.

## Tratamento de system prompt {#system-prompt-handling}

Quando um frontend envia mensagem `system` (Chat Completions) ou campo `instructions` (Responses API), hermes-agent **empilha sobre** seu system prompt core. Seu agente mantém todas as ferramentas, memória e skills — o system prompt do frontend adiciona instruções extras.

Isso significa que você pode customizar comportamento por frontend sem perder capacidades:
- System prompt Open WebUI: "You are a Python expert. Always include type hints."
- O agente ainda tem terminal, file tools, web search, memória, etc.

## Autenticação {#authentication}

Auth bearer token via header `Authorization`:

```
Authorization: Bearer ***
```

Configure a key via env var `API_SERVER_KEY`. Se precisar que browser chame Hermes diretamente, também defina `API_SERVER_CORS_ORIGINS` como allowlist explícita.

:::warning Segurança
O API server dá acesso completo ao toolset do hermes-agent, **incluindo comandos de terminal**. `API_SERVER_KEY` é **obrigatória para todo deployment**, incluindo bind loopback default em `127.0.0.1`. Mantenha `API_SERVER_CORS_ORIGINS` estreito para controlar acesso de browser quando você explicitamente permitir callers browser.
:::

## Configuração {#configuration}

### Environment Variables {#environment-variables}

| Variable | Default | Descrição |
|----------|---------|-------------|
| `API_SERVER_ENABLED` | `false` | Habilita o API server |
| `API_SERVER_PORT` | `8642` | Porta HTTP do servidor |
| `API_SERVER_HOST` | `127.0.0.1` | Endereço de bind (localhost only por padrão) |
| `API_SERVER_KEY` | _(required)_ | Bearer token para auth |
| `API_SERVER_CORS_ORIGINS` | _(none)_ | Origens browser permitidas separadas por vírgula |
| `API_SERVER_MODEL_NAME` | _(profile name)_ | Nome de model em `/v1/models`. Default nome do profile, ou `hermes-agent` para profile default. |

### config.yaml {#configyaml}

```yaml
# Not yet supported — use environment variables.
# config.yaml support coming in a future release.
```

## Security Headers {#security-headers}

Todas as responses incluem security headers:
- `X-Content-Type-Options: nosniff` — previne MIME type sniffing
- `Referrer-Policy: no-referrer` — previne vazamento de referrer

## CORS {#cors}

O API server **não** habilita browser CORS por padrão.

Para acesso direto de browser, defina allowlist explícita:

```bash
API_SERVER_CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

Quando CORS está habilitado:
- **Preflight responses** incluem `Access-Control-Max-Age: 600` (cache 10 minutos)
- **SSE streaming responses** incluem headers CORS para browser EventSource clients funcionarem corretamente
- **`Idempotency-Key`** é header de request permitido — clients podem enviá-lo para deduplicação (responses cached por key por 5 minutos)

A maioria dos frontends documentados como Open WebUI conecta server-to-server e não precisa de CORS.

## Frontends compatíveis {#compatible-frontends}

Qualquer frontend que suporte formato OpenAI API funciona. Integrações testadas/documentadas:

| Frontend | Stars | Conexão |
|----------|-------|------------|
| [Open WebUI](/user-guide/messaging/open-webui) | 126k | Guia completo disponível |
| LobeChat | 73k | Custom provider endpoint |
| LibreChat | 34k | Custom endpoint em librechat.yaml |
| AnythingLLM | 56k | Generic OpenAI provider |
| NextChat | 87k | BASE_URL env var |
| ChatBox | 39k | API Host setting |
| Jan | 26k | Remote model config |
| HF Chat-UI | 8k | OPENAI_BASE_URL |
| big-AGI | 7k | Custom endpoint |
| OpenAI Python SDK | — | `OpenAI(base_url="http://localhost:8642/v1")` |
| curl | — | Direct HTTP requests |

## Setup multi-usuário com profiles {#multi-user-setup-with-profiles}

Para dar a múltiplos usuários sua própria instância Hermes isolada (config, memória, skills separados), use [profiles](/user-guide/profiles):

```bash
# Create a profile per user
hermes profile create alice
hermes profile create bob

# Configure each profile's API server on a different port. API_SERVER_* are env
# vars (not config.yaml keys), so write them to each profile's .env:
cat >> ~/.hermes/profiles/alice/.env <<EOF
API_SERVER_ENABLED=true
API_SERVER_PORT=8643
API_SERVER_KEY=alice-secret
EOF

cat >> ~/.hermes/profiles/bob/.env <<EOF
API_SERVER_ENABLED=true
API_SERVER_PORT=8644
API_SERVER_KEY=bob-secret
EOF

# Start each profile's gateway
hermes -p alice gateway &
hermes -p bob gateway &
```

O API server de cada profile anuncia automaticamente o nome do profile como model ID:

- `http://localhost:8643/v1/models` → model `alice`
- `http://localhost:8644/v1/models` → model `bob`

No Open WebUI, adicione cada um como conexão separada. O dropdown de model mostra `alice` e `bob` como models distintos, cada um backed por instância Hermes totalmente isolada. Veja o [guia Open WebUI](/user-guide/messaging/open-webui#multi-user-setup-with-profiles) para detalhes.

## Limitações {#limitations}

- **Response storage** — responses armazenadas (para `previous_response_id`) persistem em SQLite e sobrevivem restarts do gateway. Max 100 responses armazenadas (eviction LRU).
- **No file upload** — imagens inline são suportadas em `/v1/chat/completions` e `/v1/responses`, mas arquivos uploaded (`file`, `input_file`, `file_id`) e inputs de documento não-imagem não são suportados pela API.
- **Model field is cosmetic** — o campo `model` em requests é aceito mas o LLM model real usado é configurado server-side em config.yaml.

## Proxy Mode {#proxy-mode}

O API server também serve como backend para **gateway proxy mode**. Quando outra instância gateway Hermes está configurada com `GATEWAY_PROXY_URL` apontando para este API server, ela encaminha todas as mensagens aqui em vez de rodar seu próprio agente. Isso habilita deployments split — por exemplo, container Docker tratando Matrix E2EE que relay para agent host-side.

Veja [Matrix Proxy Mode](/user-guide/messaging/matrix#proxy-mode-e2ee-on-macos) para o guia completo de setup.
