---
sidebar_position: 14
title: "Servidor de API"
description: "Exponha o hermes-agent como API compatível com OpenAI para qualquer frontend"
---

# Servidor de API {#api-server}

O API server expõe o hermes-agent como endpoint HTTP compatível com OpenAI. Qualquer frontend que fale o formato OpenAI — Open WebUI, LobeChat, LibreChat, NextChat, ChatBox e centenas de outros — pode conectar ao hermes-agent e usá-lo como backend.

Seu agente trata requisições com seu conjunto completo de ferramentas (terminal, operações de arquivo, busca web, memória, skills) e retorna a resposta final. Com streaming, indicadores de progresso de ferramentas aparecem inline para frontends mostrarem o que o agente está fazendo.

:::tip Um backend cobre models + ferramentas
O Hermes em si precisa de um provider configurado e backends de ferramentas para o API server ser útil. Uma assinatura do [Nous Portal](/user-guide/features/tool-gateway) cobre ambos — 300+ modelos mais web/imagem/TTS/browser via Tool Gateway. Execute `hermes setup --portal` uma vez antes de iniciar o API server e frontends como Open WebUI ou LobeChat recebem um backend totalmente equipado com ferramentas.
:::

## Início rápido {#quick-start}

### 1. Habilitar o API server {#1-enable-the-api-server}

Adicione ao `~/.hermes/.env`:

```bash
API_SERVER_ENABLED=true
API_SERVER_KEY=change-me-local-dev
# Optional: only if a browser must call Hermes directly
# API_SERVER_CORS_ORIGINS=http://localhost:3000
```

### 2. Iniciar o gateway {#2-start-the-gateway}

```bash
hermes gateway
```

Você verá:

```
[API Server] API server listening on http://127.0.0.1:8642
```

### 3. Conectar um frontend {#3-connect-a-frontend}

Aponte qualquer cliente compatível com OpenAI para `http://localhost:8642/v1`:

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

**Entrada de imagem inline:** mensagens de usuário podem enviar `content` como array de partes `text` e `image_url`. URLs remotas `http(s)` e URLs `data:image/...` são suportadas:

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

Arquivos enviados (`file` / `input_file` / `file_id`) e URLs `data:` que não são imagem retornam `400 unsupported_content_type`.

**Streaming** (`"stream": true`): Retorna Server-Sent Events (SSE) com chunks de resposta token a token. Para **Chat Completions**, o stream usa eventos padrão `chat.completion.chunk` mais o evento customizado `hermes.tool.progress` do Hermes para UX de início de ferramenta. Para **Responses**, o stream usa tipos de evento OpenAI Responses como `response.created`, `response.output_text.delta`, `response.output_item.added`, `response.output_item.done` e `response.completed`.

**Progresso de ferramentas em streams**:
- **Chat Completions**: O Hermes emite `event: hermes.tool.progress` para visibilidade de início de ferramenta sem poluir texto persistido do assistente.
- **Responses**: O Hermes emite output items spec-native `function_call` e `function_call_output` durante o SSE stream, para clientes renderizarem UI estruturada de ferramentas em tempo real.

### POST /v1/responses {#post-v1responses}

Formato OpenAI Responses API. Suporta estado de conversa server-side via `previous_response_id` — o servidor armazena histórico completo da conversa (incluindo chamadas e resultados de ferramentas) para contexto multi-turno ser preservado sem o cliente gerenciá-lo.

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
    {"type": "function_call", "status": "completed", "name": "terminal", "arguments": "{\"command\": \"ls\"}", "call_id": "call_1"},
    {"type": "function_call_output", "status": "completed", "call_id": "call_1", "output": "README.md src/ tests/"},
    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Your project has..."}]}
  ],
  "usage": {"input_tokens": 50, "output_tokens": 200, "total_tokens": 250}
}
```

Chamadas de ferramentas no array `output` já foram executadas server-side pelo agente Hermes — são replayed com `"status": "completed"` para UI estruturada de ferramentas, nunca como chamadas pendentes para o cliente executar.

**Entrada de imagem inline:** `input[].content` pode conter partes `input_text` e `input_image`. URLs remotas e URLs `data:image/...` são suportadas:

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

Arquivos enviados (`input_file` / `file_id`) e URLs `data:` que não são imagem retornam `400 unsupported_content_type`.

#### Multi-turn com previous_response_id {#multi-turn-with-previous_response_id}

Encadeie responses para manter contexto completo (incluindo chamadas de ferramentas) entre turnos:

```json
{
  "input": "Now show me the README",
  "previous_response_id": "resp_abc123"
}
```

O servidor reconstrói a conversa completa da cadeia de responses armazenadas — todas as chamadas e resultados de ferramentas anteriores são preservados. Requisições encadeadas também compartilham a mesma sessão, então conversas multi-turno aparecem como uma entrada no dashboard e histórico de sessões.

#### Conversas nomeadas {#named-conversations}

Use o parâmetro `conversation` em vez de rastrear IDs de response:

```json
{"input": "Hello", "conversation": "my-project"}
{"input": "What's in src/?", "conversation": "my-project"}
{"input": "Run the tests", "conversation": "my-project"}
```

O servidor encadeia automaticamente à response mais recente daquela conversa. Como o comando `/title` para sessões de gateway.

### GET /v1/responses/\{id\} {#get-v1responsesid}

Recupera uma response armazenada anteriormente por ID.

### DELETE /v1/responses/\{id\} {#delete-v1responsesid}

Exclui uma response armazenada.

### GET /v1/models {#get-v1models}

Lista o agente como model disponível. O nome do model anunciado usa por padrão o nome do [perfil](/user-guide/profiles) (ou `hermes-agent` para o perfil padrão). Necessário para a maioria dos frontends na descoberta de models.

`/v1/models` é intencionalmente a superfície barata compatível com OpenAI. **Não**
enumera toda combinação autenticada provider/model que o Hermes pode rotear,
e não faz enriquecimento de preço ou capacidades.

### GET /api/model/options {#get-apimodeloptions}

Clientes Hermes-aware podem solicitar o mesmo inventário curado provider/model usado
pelo dashboard e TUI. Esta rota usa a autenticação bearer normal do API server e retorna linhas de provider, dicas de capacidade de model e metadata de preço que não pertencem à resposta compatível com OpenAI `/v1/models`:

```bash
curl \
  -H "Authorization: Bearer $API_SERVER_KEY" \
  "http://127.0.0.1:8642/api/model/options"
```

Esse payload é o mesmo substrato que a página Models do dashboard e o RPC TUI
`model.options` usam. Retorna providers autenticados, listas curadas de models,
preço por model e dicas de capacidade de model.

Aberturas normais são intencionalmente conservadoras para providers customizados: o Hermes sonda
apenas o endpoint custom **atualmente selecionado** para um endpoint salvo obsoleto ou offline não bloquear o picker. Um refresh explícito muda para sondagem completa
e invalida o cache de models do provider:

```bash
curl \
  -H "Authorization: Bearer $API_SERVER_KEY" \
  "http://127.0.0.1:8642/api/model/options?refresh=1"
```

Use `/v1/models` quando um cliente compatível com OpenAI só precisa de um nome de model para
enviar de volta em requisições chat/responses. Use `/api/model/options` quando uma
UI autenticada precisa da metadata mais rica específica do Hermes do picker.

### GET /v1/capabilities {#get-v1capabilities}

Retorna descrição legível por máquina da superfície estável do API server para UIs externas, orquestradores e pontes de plugin.

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

Use este endpoint ao integrar dashboards, UIs de browser ou planos de controle para descobrirem se a versão Hermes em execução suporta runs, streaming, cancelamento e continuidade de sessão sem depender de internals Python privados.

## Seleção de model por requisição {#per-request-model-selection}

Clientes autenticados podem sobrescrever a seleção padrão de model do Hermes por requisição
enviando:

- `model` — id do model alvo para este turno
- `provider` — slug do provider Hermes para resolver credenciais/runtime deste turno
- `model_options` — controles de raciocínio / service-tier escopados à requisição

Os mesmos campos de requisição são aceitos em:

- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/runs`
- `POST /api/sessions/{session_id}/chat`
- `POST /api/sessions/{session_id}/chat/stream`

A precedência é determinística:

1. Override de `/model` da sessão, se a sessão já tiver um
2. Mapeamento estático `gateway.platforms.api_server.model_routes` selecionado quando
   o `model` da requisição é um alias de rota configurado
3. `model` / `provider` diretos da requisição quando nenhum alias de rota corresponde
4. Config global do gateway / padrões de ambiente

`model_options` permanece escopado à requisição independentemente de qual model/provider vencer.
Se uma requisição envia um `provider` que conflita com um alias `model_routes`
configurado, o Hermes rejeita a requisição com `400` em vez de remixar silenciosamente credenciais
de rota com outro provider.

**Valores `model` bare nos endpoints compatíveis com OpenAI são opt-in.** Clientes
OpenAI genéricos frequentemente hardcodam nomes de model (`gpt-4o`, ...), e deployments
existentes dependem desses caírem no padrão do gateway. Em
`POST /v1/chat/completions` e `POST /v1/responses`, um valor `model` enviado
SEM `provider` é portanto ignorado a menos que você habilite:

```yaml
gateway:
  platforms:
    api_server:
      direct_model_requests: true
```

Requisições que incluem `provider` explícito — e os endpoints nativos Hermes
`/v1/runs` e session-chat — sempre honram o model solicitado
independentemente desta flag.

Exemplo:

```json
{
  "model": "MiniMax-M3",
  "provider": "minimax",
  "model_options": {
    "reasoning_effort": "high",
    "service_tier": "priority"
  },
  "messages": [
    {"role": "user", "content": "Summarize the repo status."}
  ]
}
```

### GET /health {#get-health}

Health check. Retorna `{"status": "ok"}`. Também disponível em **GET /v1/health** para clientes compatíveis com OpenAI que esperam o prefixo `/v1/`.

### GET /health/detailed {#get-healthdetailed}

Readiness check autenticado para monitoramento e planos de controle. Reporta
status limitado para config do perfil ativo, banco de estado, model configurado,
espaço em disco, estado gateway/plataforma, runs API ativos, conclusões de processo
pendentes e delegações ativas. A resposta expõe status e contagens,
não valores de config, credenciais, caminhos, comandos, payloads de fila ou erros brutos.

A rota pública `/health` permanece uma sonda barata de liveness e não executa
checks de readiness. Um resultado de readiness degradado ainda usa HTTP 200; inspecione os
campos `status` de topo e `readiness.checks`.

## Runs API (alternativa amigável a streaming) {#runs-api-streaming-friendly-alternative}

Além de `/v1/chat/completions` e `/v1/responses`, o servidor expõe uma **runs** API para sessões longas em que o cliente quer assinar eventos de progresso em vez de gerenciar streaming.

### POST /v1/runs {#post-v1runs}

Cria um novo agent run. Retorna um `run_id` que pode ser usado para assinar eventos de progresso.

```json
{
  "run_id": "run_abc123",
  "status": "started"
}
```

Runs aceitam uma string `input` simples e `session_id`, `instructions`, `conversation_history` ou `previous_response_id` opcionais. Quando `session_id` é fornecido, o Hermes o expõe no status do run para UIs externas correlacionarem runs com seus próprios IDs de conversa.

### GET /v1/runs/\{run_id\} {#get-v1runsrun_id}

Consulta o estado atual do run. Útil para dashboards que precisam de status sem manter conexão SSE aberta, ou UIs que reconectam após navegação.

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

Status são retidos brevemente após estados terminais (`completed`, `failed` ou `cancelled`) para polling e reconciliação de UI.

### GET /v1/runs/\{run_id\}/events {#get-v1runsrun_idevents}

Stream Server-Sent Events do progresso de chamadas de ferramentas do run, deltas de token e eventos de ciclo de vida. Projetado para dashboards e clientes thick que querem attach/detach sem perder estado.

Quando o agente delega trabalho a subagentes em background, o stream também carrega
eventos de ciclo de vida `subagent.start` e `subagent.complete`, para clientes
observarem resultados de delegação — incluindo timeouts e falhas — em vez do
run ficar silencioso enquanto um filho trabalha. O payload `subagent.complete` carrega
status, summary, duração, figuras de token/custo e
`child_session_id` do filho para correlação; campos de texto livre passam por
redação forçada de secrets antes de sair do processo. Eventos por ferramenta do filho
(`subagent.tool`, ticks de progresso) são intencionalmente **não** encaminhados — são
ruído de UI de alto volume; use os arquivos de transcrição live por filho para
play-by-play.

Buffers de eventos não consumidos expiram após cinco minutos para um cliente detached não
crescer memória indefinidamente. Isso expira só estado de transporte: um run que ainda
está executando permanece visível a polling de status, aprovação, controle de stop e
contabilidade de concorrência até o trabalho do executor sair de fato. Um assinante SSE
conectado continua drenando normalmente.

### POST /v1/runs/\{run_id\}/stop {#post-v1runsrun_idstop}

Interrompe um turno de agente em execução. O endpoint retorna imediatamente com `{"status": "stopping"}` enquanto o Hermes pede ao agente ativo para parar no próximo ponto seguro de interrupção.
O run permanece rastreado como `stopping` até o trabalho backed pelo executor sair, depois
assenta como `cancelled`; solicitar stop nunca esconde um worker que ainda
está rodando.

### POST /v1/runs/\{run_id\}/approval {#post-v1runsrun_idapproval}

Resolve uma aprovação pendente para um run aguardando decisão humana (por exemplo, chamada de ferramenta gated por política de aprovação). O corpo carrega a decisão de aprovação; o run retoma quando a decisão é registrada. Este endpoint é anunciado em `/v1/capabilities` como a feature `run_approval` para UIs externas detectarem suporte antes de exibir prompt de aprovação.

## Jobs API (trabalho agendado em background) {#jobs-api-background-scheduled-work}

O servidor expõe uma superfície CRUD leve de jobs para gerenciar runs de agente agendados / em background de um cliente remoto. Todos os endpoints são gated pela mesma auth bearer.

### GET /api/jobs {#get-apijobs}

Lista todos os jobs agendados.

### POST /api/jobs {#post-apijobs}

Cria um novo job agendado. O corpo aceita a mesma forma que `hermes cron` — prompt, schedule, skills, override de provider, destino de entrega.

### GET /api/jobs/\{job_id\} {#get-apijobsjob_id}

Busca definição e estado da última execução de um job.

### PATCH /api/jobs/\{job_id\} {#patch-apijobsjob_id}

Atualiza campos de um job existente (prompt, schedule, etc.). Updates parciais são mesclados.

### DELETE /api/jobs/\{job_id\} {#delete-apijobsjob_id}

Remove um job. Também cancela qualquer run em voo.

### POST /api/jobs/\{job_id\}/pause {#post-apijobsjob_idpause}

Pausa um job sem excluí-lo. Timestamps de próxima execução agendada ficam suspensos até retomar.

### POST /api/jobs/\{job_id\}/resume {#post-apijobsjob_idresume}

Retoma um job previamente pausado.

### POST /api/jobs/\{job_id\}/run {#post-apijobsjob_idrun}

Dispara o job para rodar imediatamente, fora do schedule.

## Sessions API (controle de sessão via REST) {#sessions-api-session-control-over-rest}

UIs externas podem gerenciar sessões Hermes via REST sem subir o dashboard. Todos os endpoints são gated por `API_SERVER_KEY` e vivem em `/api/sessions/*`.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/sessions` | List sessions (paginated — `limit`, `offset`, `source`, `include_children`) |
| `POST` | `/api/sessions` | Create an empty session |
| `GET` | `/api/sessions/{id}` | Read session metadata |
| `PATCH` | `/api/sessions/{id}` | Update title or `end_reason` |
| `DELETE` | `/api/sessions/{id}` | Delete a session |
| `GET` | `/api/sessions/{id}/messages` | Message history for a session |
| `POST` | `/api/sessions/{id}/fork` | Branch the session via `SessionDB` lineage (matches CLI `/branch` semantics) |
| `POST` | `/api/sessions/{id}/chat` | Run one synchronous agent turn |
| `POST` | `/api/sessions/{id}/chat/stream` | SSE wrapper over a single turn — emits `assistant.delta`, `tool.started`, `tool.completed`, `run.completed` events |

`/v1/capabilities` anuncia a superfície completa via feature flags `session_*` e entradas `endpoints.session_*` para UIs externas detectarem suporte e caírem com segurança. Imagens inline são suportadas em payloads `chat` e `chat/stream` (caminho multimodal-aware).

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

`GET /v1/skills` e `GET /v1/toolsets` deixam clientes externos enumerarem as capacidades do agente deterministicamente via REST em vez de perguntar ao model. Ambos são read-only e gated por `API_SERVER_KEY`.

```bash
curl http://localhost:8642/v1/skills \
  -H "Authorization: Bearer $API_SERVER_KEY"
# → [{"name": "github-pr-workflow", "description": "...", "category": "..."}, ...]

curl http://localhost:8642/v1/toolsets \
  -H "Authorization: Bearer $API_SERVER_KEY"
# → [{"name": "core", "label": "...", "description": "...", "enabled": true,
#     "configured": true, "tools": ["read_file", "write_file", ...]}, ...]
```

`/v1/skills` retorna a mesma metadata que o skills hub usa internamente. `/v1/toolsets` retorna toolsets resolvidos para a plataforma `api_server` com a lista concreta de `tools` que cada um expande. Ambos são anunciados em `endpoints.*` em `/v1/capabilities`.

## Escopo de memória de longo prazo (`X-Hermes-Session-Key`) {#long-term-memory-scoping-x-hermes-session-key}

Frontends multi-usuário como Open WebUI precisam de um identificador estável por canal para memória de longo prazo (Honcho, etc.) **independente** do `X-Hermes-Session-Id` escopado à transcrição (que rotaciona em `/new`). Passe `X-Hermes-Session-Key` em `/v1/chat/completions`, `/v1/responses` ou `/v1/runs` e o Hermes propaga para `AIAgent(gateway_session_key=...)`, onde o memory provider Honcho o usa para derivar um escopo estável.

```http
POST /v1/chat/completions HTTP/1.1
Authorization: Bearer ***
X-Hermes-Session-Id: transcript-alpha
X-Hermes-Session-Key: agent:main:webui:dm:user-42
```

Regras: máx. 256 chars, caracteres de controle (`\r`, `\n`, `\x00`) são rejeitados, e o valor é ecoado nas responses (JSON + SSE). `/v1/capabilities` anuncia suporte via `"session_key_header": "X-Hermes-Session-Key"`. Sem a chave, a estratégia `per-session` do Honcho produz um escopo diferente por `session_id` — exatamente o comportamento que o Hermes tinha antes.

## Tratamento de system prompt {#system-prompt-handling}

Quando um frontend envia mensagem `system` (Chat Completions) ou campo `instructions` (Responses API), o hermes-agent **empilha por cima** de seu system prompt core. Seu agente mantém todas as ferramentas, memória e skills — o system prompt do frontend adiciona instruções extras.

Isso significa que você pode customizar comportamento por frontend sem perder capacidades:
- System prompt Open WebUI: "You are a Python expert. Always include type hints."
- O agente ainda tem terminal, ferramentas de arquivo, busca web, memória, etc.

## Autenticação {#authentication}

Auth bearer token via header `Authorization`:

```
Authorization: Bearer ***
```

Configure a chave via env var `API_SERVER_KEY`. Se um browser precisa chamar o Hermes diretamente, defina também `API_SERVER_CORS_ORIGINS` como allowlist explícita.

### Roteamento multi-perfil (`/p/<profile>/…`) {#multi-profile-routing-pprofile}

Quando [roteamento multi-perfil de gateway](/user-guide/multi-profile-gateways) está
habilitado (`gateway.multiplex_profiles`), o listener compartilhado serve todo
perfil por um prefixo de URL `/p/<profile>/` — e **a autenticação está ligada
ao perfil roteado**:

- Requisições a `/p/<profile>/v1/...` devem apresentar o próprio
  `API_SERVER_KEY` daquele perfil (de `~/.hermes/profiles/<profile>/.env`). A chave do
  listener padrão é rejeitada em prefixos de perfil nomeados.
- Rotas sem prefixo e `/p/default/...` continuam usando a chave do perfil padrão.
- Um perfil nomeado sem seu próprio `API_SERVER_KEY` falha fechado — seu
  prefixo fica inalcançável até você definir um.

:::warning Breaking change (July 2026)
Antes desta correção, uma chave válida do perfil padrão era aceita em qualquer
prefixo `/p/<profile>/`. Se você dependia de uma chave compartilhada entre prefixos de perfil,
defina um `API_SERVER_KEY` distinto no `.env` de cada perfil — chaves padrão reutilizadas
em prefixos nomeados agora retornam `401`.
:::

:::warning Segurança
O API server dá acesso total ao toolset do hermes-agent, **incluindo comandos de terminal**. `API_SERVER_KEY` é **obrigatório em todo deployment**, incluindo bind loopback padrão em `127.0.0.1`. Mantenha `API_SERVER_CORS_ORIGINS` estreito para controlar acesso de browser quando permitir callers de browser explicitamente.
:::

## Configuração {#configuration}

### Variáveis de ambiente {#environment-variables}

| Variable | Default | Description |
|----------|---------|-------------|
| `API_SERVER_ENABLED` | `false` | Enable the API server |
| `API_SERVER_PORT` | `8642` | HTTP server port |
| `API_SERVER_HOST` | `127.0.0.1` | Bind address (localhost only by default) |
| `API_SERVER_KEY` | _(required)_ | Bearer token for auth |
| `API_SERVER_CORS_ORIGINS` | _(none)_ | Comma-separated allowed browser origins |
| `API_SERVER_MODEL_NAME` | _(profile name)_ | Model name on `/v1/models`. Defaults to profile name, or `hermes-agent` for default profile. |

### config.yaml {#configyaml}

As mesmas configurações podem ficar em `~/.hermes/config.yaml` sob a seção aninhada `gateway.api_server:`:

```yaml
gateway:
  api_server:
    enabled: true
    port: 8642
    host: 127.0.0.1
    key: your-secret-key
    cors_origins: http://localhost:3000
    model_name: my-hermes
    max_concurrent_runs: 10   # concurrent-run cap; 0 disables the limit
```

`port`, `key`, `host`, `cors_origins` e `model_name` são automaticamente bridged para as configurações `extra` da plataforma, então se comportam exatamente como suas contrapartes de variável de ambiente `API_SERVER_*`. Variáveis de ambiente têm precedência sobre valores de `config.yaml`. O bloco também é aceito sob `gateway.platforms.api_server:` ou seção top-level `platforms.api_server:`.

### Limite de runs concorrentes {#concurrent-run-cap}

O API server limita quantos agent runs podem executar ao mesmo tempo nos endpoints compatíveis com OpenAI e Runs. O limite é lido de `gateway.api_server.max_concurrent_runs` (padrão **10**; `0` desabilita o limite, valores negativos clampam a 0). Quando o limite é atingido, novas requisições que iniciam run são rejeitadas com **HTTP 429** `Too many concurrent runs (max N)` — clientes devem fazer backoff e retry.

## Security headers {#security-headers}

Todas as responses incluem security headers:
- `X-Content-Type-Options: nosniff` — impede MIME type sniffing
- `Referrer-Policy: no-referrer` — impede vazamento de referrer

## CORS {#cors}

O API server **não** habilita CORS de browser por padrão.

Para acesso direto de browser, defina allowlist explícita:

```bash
API_SERVER_CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

Quando CORS está habilitado:
- **Respostas preflight** incluem `Access-Control-Max-Age: 600` (cache de 10 minutos)
- **Responses de streaming SSE** incluem headers CORS para clientes EventSource de browser funcionarem corretamente
- **`Idempotency-Key`** é header de requisição permitido — clientes podem enviá-lo para deduplicação (responses são cacheadas por chave por 5 minutos)

A maioria dos frontends documentados como Open WebUI conecta server-to-server e não precisa de CORS.

## Frontends compatíveis {#compatible-frontends}

Qualquer frontend que suporte o formato API OpenAI funciona. Integrações testadas/documentadas:

| Frontend | Stars | Connection |
|----------|-------|------------|
| [Open WebUI](/user-guide/messaging/open-webui) | 126k | Full guide available |
| LobeChat | 73k | Custom provider endpoint |
| LibreChat | 34k | Custom endpoint in librechat.yaml |
| AnythingLLM | 56k | Generic OpenAI provider |
| NextChat | 87k | BASE_URL env var |
| ChatBox | 39k | API Host setting |
| Jan | 26k | Remote model config |
| HF Chat-UI | 8k | OPENAI_BASE_URL |
| big-AGI | 7k | Custom endpoint |
| OpenAI Python SDK | — | `OpenAI(base_url="http://localhost:8642/v1")` |
| curl | — | Direct HTTP requests |

## Setup multi-usuário com perfis {#multi-user-setup-with-profiles}

Para dar a vários usuários sua própria instância Hermes isolada (config, memória, skills separados), use [perfis](/user-guide/profiles):

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

O API server de cada perfil anuncia automaticamente o nome do perfil como ID de model:

- `http://localhost:8643/v1/models` → model `alice`
- `http://localhost:8644/v1/models` → model `bob`

No Open WebUI, adicione cada um como conexão separada. O dropdown de models mostra `alice` e `bob` como models distintos, cada um backed por uma instância Hermes totalmente isolada. Veja o [guia Open WebUI](/user-guide/messaging/open-webui#multi-user-setup-with-profiles) para detalhes.

## Limitações {#limitations}

- **Armazenamento de responses** — responses armazenadas (para `previous_response_id`) são persistidas em SQLite e sobrevivem a reinícios do gateway. Máx. 100 responses armazenadas (evicção LRU).
- **Sem upload de arquivo** — imagens inline são suportadas em `/v1/chat/completions` e `/v1/responses`, mas arquivos enviados (`file`, `input_file`, `file_id`) e inputs de documento que não são imagem não são suportados via API.
- **Clientes OpenAI simples ainda veem um alias** — `/v1/models` anuncia o
  alias estável Hermes (`hermes-agent` ou o nome do perfil ativo). Clientes
  mais ricos podem enviar overrides explícitos de `provider` / `model_options` nas requisições.

## Modo proxy {#proxy-mode}

O API server também serve como backend para **modo proxy de gateway**. Quando outra instância de gateway Hermes está configurada com `GATEWAY_PROXY_URL` apontando para este API server, encaminha todas as mensagens aqui em vez de rodar seu próprio agente. Isso habilita deployments split — por exemplo, um container Docker tratando E2EE Matrix que repassa para um agente no host.

Veja [Modo proxy Matrix](/user-guide/messaging/matrix#proxy-mode-e2ee-on-macos) para o guia completo de setup.
