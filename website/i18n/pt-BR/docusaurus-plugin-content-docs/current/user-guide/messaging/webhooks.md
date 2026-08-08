---
sidebar_position: 13
title: "Webhooks"
description: "Receba eventos do GitHub, GitLab e outros serviços para disparar execuções do agente Hermes"
---

# Webhooks

Receba eventos de serviços externos (GitHub, GitLab, JIRA, Stripe, etc.) e dispare execuções do agente Hermes automaticamente. O adapter webhook roda um servidor HTTP que aceita requisições POST, valida assinaturas HMAC, transforma payloads em prompts do agente e roteia respostas de volta à origem ou para outra plataforma configurada.

O agente processa o evento e pode responder postando comentários em PRs, enviando mensagens para Telegram/Discord ou registrando o resultado.

## Video Tutorial {#video-tutorial}

<div style={{position: 'relative', width: '100%', aspectRatio: '16 / 9', marginBottom: '1.5rem'}}>
  <iframe
    src="https://www.youtube.com/embed/WNYe5mD4fY8"
    title="Hermes Agent — Webhooks Tutorial"
    style={{position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', border: 0}}
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowFullScreen
  />
</div>

---

## Quick Start {#quick-start}

1. Habilite via `hermes gateway setup` ou variáveis de ambiente
2. Defina rotas em `config.yaml` **ou** crie dinamicamente com `hermes webhook subscribe`
3. Aponte seu serviço para `http://your-server:8644/webhooks/<route-name>`

---

## Setup {#setup}

Há duas formas de habilitar o adapter webhook.

### Via setup wizard

```bash
hermes gateway setup
```

Siga os prompts para habilitar webhooks, definir a porta e um secret HMAC global.

### Via environment variables

Adicione em `~/.hermes/.env`:

```bash
WEBHOOK_ENABLED=true
WEBHOOK_PORT=8644        # padrão
WEBHOOK_SECRET=your-global-secret
```

### Verify the server

Com o gateway rodando:

```bash
curl http://localhost:8644/health
```

Resposta esperada:

```json
{"status": "ok", "platform": "webhook"}
```

---

## Configuring Routes {#configuring-routes}

Rotas definem como diferentes fontes de webhook são tratadas. Cada rota é uma entrada nomeada em `platforms.webhook.extra.routes` no seu `config.yaml`.

### Route properties

| Property | Required | Description |
|----------|----------|-------------|
| `events` | No | Lista de tipos de evento a aceitar (ex.: `["pull_request"]`). Se vazio, todos os eventos são aceitos. O tipo vem de `X-GitHub-Event`, `X-GitLab-Event` ou `event_type` no payload. |
| `secret` | **Yes** | Secret HMAC para validação de assinatura. Faz fallback para o `secret` global se não definido na rota. Defina `"INSECURE_NO_AUTH"` só para testes (pula validação). |
| `prompt` | No | String template com acesso dot-notation ao payload (ex.: `{pull_request.title}`). Se omitido, o JSON completo do payload vai para o prompt. Campos do payload são não confiáveis — veja [Authenticated does not mean trusted](#authenticated-does-not-mean-trusted). |
| `filters` | No | Filtros declarativos de payload avaliados após auth/body/event filtering e antes do trabalho de agente ou entrega direta. Não correspondências retornam `{"status":"ignored","reason":"filter"}` com HTTP 200. |
| `script` | No | Script de filtro/transformação em `~/.hermes/scripts/`. O payload webhook é passado como JSON no stdin. Stdout JSON object substitui o payload antes do templating; stdout texto expõe como `script_output`; stdout vazio, `[SILENT]` ou exit code não zero ignoram o webhook. |
| `skills` | No | Lista de nomes de skills a carregar na execução do agente. |
| `deliver` | No | Onde enviar a resposta: `github_comment`, `telegram`, `discord`, `slack`, `signal`, `sms`, `whatsapp`, `matrix`, `mattermost`, `homeassistant`, `email`, `dingtalk`, `feishu`, `wecom`, `weixin`, `bluebubbles`, `qqbot`, ou `log` (padrão). |
| `deliver_extra` | No | Config adicional de entrega — chaves dependem do tipo `deliver` (ex.: `repo`, `pr_number`, `chat_id`). Valores suportam os mesmos templates `{dot.notation}` que `prompt`. |
| `deliver_only` | No | Se `true`, pula o agente — o template `prompt` renderizado vira a mensagem literal entregue. Zero custo LLM, entrega sub-segundo. Veja [Direct Delivery Mode](#direct-delivery-mode) para casos de uso. Requer `deliver` como destino real (não `log`). |

### Full example

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      port: 8644
      secret: "global-fallback-secret"
      routes:
        github-pr:
          events: ["pull_request"]
          secret: "github-webhook-secret"
          prompt: |
            Review this pull request:
            Repository: {repository.full_name}
            PR #{number}: {pull_request.title}
            Author: {pull_request.user.login}
            URL: {pull_request.html_url}
            Diff URL: {pull_request.diff_url}
            Action: {action}
          skills: ["github-code-review"]
          deliver: "github_comment"
          deliver_extra:
            repo: "{repository.full_name}"
            pr_number: "{number}"
        deploy-notify:
          events: ["push"]
          secret: "deploy-secret"
          prompt: "New push to {repository.full_name} branch {ref}: {head_commit.message}"
          filters:
            - field: "ref"
              equals: "refs/heads/main"
          deliver: "telegram"
```

### Payload Filters

Use `filters` quando um provedor envia um stream amplo de eventos mas só alguns payloads devem acordar o agente ou disparar entrega `deliver_only`. Filtros rodam após validação de assinatura, parsing de body e `events`, mas antes de renderização de prompt, idempotency, dispatch do agente ou entrega direta.

```yaml
platforms:
  webhook:
    extra:
      routes:
        todoist:
          events: ["item:updated"]
          secret: "todoist-secret"
          filters:
            - field: "payload.labels"
              contains: "hermes"
            - any:
                - field: "payload.priority"
                  equals: 4
                - field: "payload.project_id"
                  in_file: "~/.hermes/data/todoist/watchlist.json"
          prompt: "Todoist task changed: {payload.content}"
```

Operadores suportados:

- `exists: true|false`
- `missing: true`
- `equals` / `not_equals`
- `contains` para strings, listas e chaves de dict
- `in` para listas inline
- `in_file` para arrays JSON, objetos JSON (chaves são usadas) ou arquivos texto separados por newline
- `regex`
- grupos `all`, `any` e `not`

Caminhos de campo usam dot notation. `payload.foo` lê de um objeto `payload` top-level quando existe, ou do body webhook raiz para payloads planos. `event` / `event_type` correspondem ao tipo de evento resolvido, e `headers.<Name>` lê headers de requisição.

### Script Filters and Transforms

Use `script` quando filtros declarativos não bastam. Scripts devem ficar em `~/.hermes/scripts/` para o perfil ativo; caminhos relativos resolvem lá, e path traversal fora desse diretório é bloqueado. Scripts `.sh` e `.bash` rodam com bash, e outras extensões com o interpretador Python atual.

O payload da rota é enviado ao stdin como JSON:

```python
# ~/.hermes/scripts/todoist-hermes-label.py
import json
import sys

payload = json.load(sys.stdin)
labels = payload.get("payload", {}).get("labels", [])
if "hermes" not in labels:
    print("[SILENT]")
    raise SystemExit(0)

payload["body"] = payload["payload"]["content"]
print(json.dumps(payload))
```

Resultados do script:

- Stdout JSON object substitui o payload usado por `prompt` e `deliver_extra`.
- Stdout texto não-JSON é adicionado ao payload como `script_output`.
- Stdout vazio, exato `[SILENT]`, `{"__hermes_ignore__": true}`, timeout, script ausente ou exit code não zero retorna HTTP 200 com `{"status":"ignored","reason":"script"}`.

### Prompt Templates

Prompts usam dot-notation para acessar campos aninhados no payload webhook:

- `{pull_request.title}` resolve para `payload["pull_request"]["title"]`
- `{repository.full_name}` resolve para `payload["repository"]["full_name"]`
- `{__raw__}` — token especial que despeja o **payload inteiro** como JSON indentado (truncado em 4000 caracteres). Útil para alertas de monitoramento ou webhooks genéricos onde o agente precisa do contexto completo.
- Chaves ausentes ficam como string literal `{key}` (sem erro)
- Dicts e listas aninhados são serializados em JSON e truncados em 2000 caracteres

Você pode misturar `{__raw__}` com variáveis template normais:

```yaml
prompt: "PR #{pull_request.number} by {pull_request.user.login}: {__raw__}"
```

Se nenhum template `prompt` estiver configurado para uma rota, o payload inteiro é despejado como JSON indentado (truncado em 4000 caracteres).

Os mesmos templates dot-notation funcionam em valores `deliver_extra`.

### Forum Topic Delivery

Ao entregar respostas webhook para Telegram, você pode mirar um tópico de fórum específico incluindo `message_thread_id` (ou `thread_id`) em `deliver_extra`:

```yaml
webhooks:
  routes:
    alerts:
      events: ["alert"]
      prompt: "Alert: {__raw__}"
      deliver: "telegram"
      deliver_extra:
        chat_id: "-1001234567890"
        message_thread_id: "42"
```

Se `chat_id` não for fornecido em `deliver_extra`, a entrega faz fallback para o canal home configurado na plataforma alvo.

---

## GitHub PR Review (Step by Step) {#github-pr-review}

Este walkthrough configura code review automático em todo pull request.

### 1. Create the webhook in GitHub

1. Vá ao repositório → **Settings** → **Webhooks** → **Add webhook**
2. Defina **Payload URL** para `http://your-server:8644/webhooks/github-pr`
3. Defina **Content type** como `application/json`
4. Defina **Secret** para corresponder à config da rota (ex.: `github-webhook-secret`)
5. Em **Which events?**, selecione **Let me select individual events** e marque **Pull requests**
6. Clique **Add webhook**

### 2. Add the route config

Adicione a rota `github-pr` em `~/.hermes/config.yaml` como no exemplo acima.

### 3. Ensure `gh` CLI is authenticated

O tipo de entrega `github_comment` usa GitHub CLI para postar comentários:

```bash
gh auth login
```

### 4. Test it

Abra um pull request no repositório. O webhook dispara, o Hermes processa o evento e posta um comentário de review no PR.

---

## GitLab Webhook Setup {#gitlab-webhook-setup}

Webhooks GitLab funcionam de forma similar mas usam mecanismo de autenticação diferente. O GitLab envia o secret como header plain `X-Gitlab-Token` (correspondência exata de string, não HMAC).

### 1. Create the webhook in GitLab

1. Vá ao projeto → **Settings** → **Webhooks**
2. Defina a **URL** para `http://your-server:8644/webhooks/gitlab-mr`
3. Informe seu **Secret token**
4. Selecione **Merge request events** (e outros eventos desejados)
5. Clique **Add webhook**

### 2. Add the route config

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      routes:
        gitlab-mr:
          events: ["merge_request"]
          secret: "your-gitlab-secret-token"
          prompt: |
            Review this merge request:
            Project: {project.path_with_namespace}
            MR !{object_attributes.iid}: {object_attributes.title}
            Author: {object_attributes.last_commit.author.name}
            URL: {object_attributes.url}
            Action: {object_attributes.action}
          deliver: "log"
```

---

## Delivery Options {#delivery-options}

O campo `deliver` controla para onde vai a resposta do agente após processar o evento webhook.

| Deliver Type | Description |
|-------------|-------------|
| `log` | Registra a resposta na saída de log do gateway. Padrão e útil para testes. |
| `github_comment` | Posta a resposta como comentário PR/issue via CLI `gh`. Requer `deliver_extra.repo` e `deliver_extra.pr_number`. O CLI `gh` deve estar instalado e autenticado no host gateway (`gh auth login`). |
| `telegram` | Roteia a resposta para Telegram. Usa canal home, ou especifique `chat_id` em `deliver_extra`. |
| `discord` | Roteia a resposta para Discord. Usa canal home, ou especifique `chat_id` em `deliver_extra`. |
| `slack` | Roteia a resposta para Slack. Usa canal home, ou especifique `chat_id` em `deliver_extra`. |
| `signal` | Roteia a resposta para Signal. Usa canal home, ou especifique `chat_id` em `deliver_extra`. |
| `sms` | Roteia a resposta para SMS via Twilio. Usa canal home, ou especifique `chat_id` em `deliver_extra`. |
| `whatsapp` | Roteia a resposta para WhatsApp. Usa canal home, ou especifique `chat_id` em `deliver_extra`. |
| `matrix` | Roteia a resposta para Matrix. Usa canal home, ou especifique `chat_id` em `deliver_extra`. |
| `mattermost` | Roteia a resposta para Mattermost. Usa canal home, ou especifique `chat_id` em `deliver_extra`. |
| `homeassistant` | Roteia a resposta para Home Assistant. Usa canal home, ou especifique `chat_id` em `deliver_extra`. |
| `email` | Roteia a resposta para Email. Usa canal home, ou especifique `chat_id` em `deliver_extra`. |
| `dingtalk` | Roteia a resposta para DingTalk. Usa canal home, ou especifique `chat_id` em `deliver_extra`. |
| `feishu` | Roteia a resposta para Feishu/Lark. Usa canal home, ou especifique `chat_id` em `deliver_extra`. |
| `wecom` | Roteia a resposta para WeCom. Usa canal home, ou especifique `chat_id` em `deliver_extra`. |
| `weixin` | Roteia a resposta para Weixin (WeChat). Usa canal home, ou especifique `chat_id` em `deliver_extra`. |
| `bluebubbles` | Roteia a resposta para BlueBubbles (iMessage). Usa canal home, ou especifique `chat_id` em `deliver_extra`. |

Para entrega cross-platform, a plataforma alvo também deve estar habilitada e conectada no gateway. Se nenhum `chat_id` for fornecido em `deliver_extra`, a resposta vai para o canal home configurado daquela plataforma.

---

## Direct Delivery Mode {#direct-delivery-mode}

Por padrão, todo POST webhook dispara uma execução do agente — o payload vira prompt, o agente processa e a resposta do agente é entregue. Isso custa tokens LLM em todo evento.

Para casos onde você só quer **empurrar uma notificação plain** — sem raciocínio, sem loop de agente, só entregar a mensagem — defina `deliver_only: true` na rota. O template `prompt` renderizado vira o corpo literal da mensagem, e o adapter despacha direto ao destino configurado.

### When to use direct delivery

- **Push de serviço externo** — webhook Supabase/Firebase dispara em mudança de banco → notifica usuário no Telegram instantaneamente
- **Alertas de monitoramento** — webhook Datadog/Grafana → push para canal Discord
- **Pings inter-agente** — Agente A notifica usuário do Agente B que tarefa longa terminou
- **Conclusão de job em background** — Cron job termina → posta resultado no Slack

Benefícios:

- **Zero tokens LLM** — o agente nunca é invocado
- **Entrega sub-segundo** — uma única chamada adapter, sem loop de raciocínio
- **Mesma segurança do modo agente** — auth HMAC, rate limits, idempotency e limites de body-size ainda aplicam
- **Resposta síncrona** — o POST retorna `200 OK` quando a entrega succeede, ou `502` se o alvo rejeitar, para seu serviço upstream retentar de forma inteligente

### Example: Telegram push from Supabase

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      port: 8644
      secret: "global-secret"
      routes:
        antenna-matches:
          secret: "antenna-webhook-secret"
          deliver: "telegram"
          deliver_only: true
          prompt: "🎉 New match: {match.user_name} matched with you!"
          deliver_extra:
            chat_id: "{match.telegram_chat_id}"
```

Sua edge function Supabase assina o payload com HMAC-SHA256 e faz POST para `https://your-server:8644/webhooks/antenna-matches`. O adapter webhook valida a assinatura, renderiza o template do payload, entrega no Telegram e retorna `200 OK`.

### Example: Dynamic subscription via CLI

```bash
hermes webhook subscribe antenna-matches \
  --deliver telegram \
  --deliver-chat-id "123456789" \
  --deliver-only \
  --prompt "🎉 New match: {match.user_name} matched with you!" \
  --description "Antenna match notifications"
```

### Response codes

| Status | Meaning |
|--------|---------|
| `200 OK` | Entregue com sucesso. Body: `{"status": "delivered", "route": "...", "target": "...", "delivery_id": "..."}` |
| `200 OK` (status=duplicate) | ID `X-GitHub-Delivery` duplicado dentro do TTL de idempotency (1 hora). Não re-entregue. |
| `401 Unauthorized` | Assinatura HMAC inválida ou ausente. |
| `400 Bad Request` | Body JSON malformado. |
| `404 Not Found` | Nome de rota desconhecido. |
| `413 Payload Too Large` | Body excedeu `max_body_bytes`. |
| `429 Too Many Requests` | Rate limit da rota excedido. |
| `502 Bad Gateway` | Adapter alvo rejeitou a mensagem ou levantou exceção. O erro é logado server-side; o body da resposta é `Delivery failed` genérico para não vazar internals do adapter. |

### Configuration gotchas

- `deliver_only: true` requer `deliver` como destino real. `deliver: log` (ou omitir `deliver`) é rejeitado na inicialização — o adapter recusa iniciar se encontrar rota mal configurada.
- O campo `skills` é ignorado em modo entrega direta (nenhum agente roda, então nada para injetar skills).
- Renderização de template usa a mesma sintaxe `{dot.notation}` do modo agente, incluindo token `{__raw__}`.
- Idempotency usa o mesmo header `X-GitHub-Delivery` / `X-Request-ID` — retries com o mesmo ID retornam `status=duplicate` e NÃO re-entregam.

---

## Dynamic Subscriptions (CLI) {#dynamic-subscriptions}

Além de rotas estáticas em `config.yaml`, você pode criar assinaturas webhook dinamicamente com o comando CLI `hermes webhook`. Isso é especialmente útil quando o próprio agente precisa configurar triggers event-driven.

### Create a subscription

```bash
hermes webhook subscribe github-issues \
  --events "issues" \
  --prompt "New issue #{issue.number}: {issue.title}\nBy: {issue.user.login}\n\n{issue.body}" \
  --deliver telegram \
  --deliver-chat-id "-100123456789" \
  --description "Triage new GitHub issues"
```

Isso retorna a URL webhook e um secret HMAC auto-gerado. Configure seu serviço para POST nessa URL.

### List subscriptions

```bash
hermes webhook list
```

### Remove a subscription

```bash
hermes webhook remove github-issues
```

### Test a subscription

```bash
hermes webhook test github-issues
hermes webhook test github-issues --payload '{"issue": {"number": 42, "title": "Test"}}'
```

### How dynamic subscriptions work

- Assinaturas ficam em `~/.hermes/webhook_subscriptions.json`
- O adapter webhook recarrega hot este arquivo a cada requisição entrante (mtime-gated, overhead negligível)
- Rotas estáticas de `config.yaml` sempre têm precedência sobre dinâmicas com o mesmo nome
- Assinaturas dinâmicas usam o mesmo formato de rota e capacidades das estáticas (events, prompt templates, skills, delivery)
- Sem reinício do gateway — subscribe e fica live imediatamente

### Agent-driven subscriptions

O agente pode criar assinaturas via ferramenta terminal quando guiado pela skill `webhook-subscriptions`. Peça ao agente para "set up a webhook for GitHub issues" e ele executará o comando `hermes webhook subscribe` apropriado.

---

## Security {#security}

O adapter webhook inclui várias camadas de segurança:

### HMAC signature validation

O adapter valida assinaturas webhook entrantes com o método apropriado para cada fonte:

- **GitHub**: header `X-Hub-Signature-256` — digest hex HMAC-SHA256 prefixado com `sha256=`
- **GitLab**: header `X-Gitlab-Token` — correspondência plain do secret
- **Generic (V2, recommended)**: headers `X-Webhook-Signature-V2` + `X-Webhook-Timestamp` — digest hex HMAC-SHA256 de `<timestamp>.<body>`. O timestamp (Unix seconds) deve estar dentro de ±300 segundos do relógio do servidor, o que impede replay de requisições capturadas depois.
- **Generic (V1, legacy)**: header `X-Webhook-Signature` — digest hex HMAC-SHA256 raw do body apenas. Ainda aceito por retrocompatibilidade, mas sem proteção replay (requisição capturada replay indefinidamente); o gateway loga aviso de depreciação uma vez por rota. Migre senders para V2.

Se um secret estiver configurado mas nenhum header de assinatura reconhecido estiver presente, a requisição é rejeitada.

### Secret is required

Toda rota deve ter secret — definido diretamente na rota ou herdado do `secret` global. Rotas sem secret fazem o adapter falhar na inicialização com erro. Só para dev/teste, defina secret como `"INSECURE_NO_AUTH"` para pular validação.

`INSECURE_NO_AUTH` só é aceito quando o gateway está bound a host loopback (`127.0.0.1`, `localhost`, `::1`). Combinado com bind não-loopback como `0.0.0.0` ou IP LAN, o adapter recusa iniciar — evita expor acidentalmente endpoint não autenticado em interface pública.

### Rate limiting

Cada rota tem rate limit de **30 requisições por minuto** por padrão (janela fixa). Configure globalmente:

```yaml
platforms:
  webhook:
    extra:
      rate_limit: 60  # requisições por minuto
```

Requisições acima do limite recebem resposta `429 Too Many Requests`.

### Idempotency

Delivery IDs (de `X-GitHub-Delivery`, `X-Request-ID`, ou fallback timestamp) são cacheados por **1 hora**. Entregas duplicadas (ex.: retries webhook) são puladas silenciosamente com resposta `200`, evitando execuções duplicadas do agente.

### Body size limits

Payloads acima de **1 MB** são rejeitados antes de ler o body. Configure:

```yaml
platforms:
  webhook:
    extra:
      max_body_bytes: 2097152  # 2 MB
```

### Authenticated does not mean trusted

:::warning
**Validação HMAC autentica o _remetente_, não o _conteúdo_.** Assinatura válida só prova que a requisição veio de quem tem o secret da rota (ex.: GitHub). Nada diz sobre quem escreveu os _campos de negócio_ dentro do payload — títulos de PR, mensagens de commit, descrições de issue e qualquer texto upstream são de terceiros arbitrários e devem ser tratados como não confiáveis.

Este é o mesmo modelo de confiança que se aplica a tudo que o agente lê: páginas web, arquivos e saída de ferramentas são input não confiável. O Hermes não — e não pode de forma confiável — sanitizar texto não confiável com blocklist; fraseado, encoding e tradução tornam bypass trivial. **A fronteira de confiança é a superfície de capacidade do agente, não o canal de entrada.** Endureça ali:

- **Sandbox do runtime.** Rode o gateway com backend terminal Docker ou SSH (ou em VM) quando exposto à internet, para um turno sequestrado não tocar o host.
- **Escopo do toolset.** Desabilite ferramentas `terminal`, `file` e ações outbound em sessões disparadas por webhook se a rota só precisa ler e resumir. Menos capacidades significa menor blast radius se um campo de payload carregar instruções injetadas.
- **Mantenha approvals ligados** para qualquer operação destrutiva ou outbound, para instrução injetada não agir desacompanhada.
- **Template estreito.** Prefira `prompt` específico com campos nomeados (`{pull_request.title}`) sobre `{__raw__}` ou template vazio que despeja o payload inteiro, para só os campos que você pretende chegarem ao prompt.
:::

---

## Troubleshooting {#troubleshooting}

### Webhook not arriving

- Verifique se a porta está exposta e acessível da fonte webhook
- Confira regras de firewall — porta `8644` (ou sua porta configurada) deve estar aberta
- Verifique se o caminho URL corresponde: `http://your-server:8644/webhooks/<route-name>`
- Use endpoint `/health` para confirmar que o servidor está rodando

### Signature validation failing

- Garanta que o secret na config da rota corresponde exatamente ao configurado na fonte webhook
- Para GitHub, o secret é baseado em HMAC — verifique `X-Hub-Signature-256`
- Para GitLab, o secret é token plain — verifique `X-Gitlab-Token`
- Verifique logs do gateway por avisos `Invalid signature`

### Event being ignored

- Verifique se o tipo de evento está na lista `events` da rota
- Eventos GitHub usam valores como `pull_request`, `push`, `issues` (valor do header `X-GitHub-Event`)
- Eventos GitLab usam valores como `merge_request`, `push` (valor do header `X-GitLab-Event`)
- Se `events` estiver vazio ou não definido, todos os eventos são aceitos

### Agent not responding

- Rode o gateway em foreground para ver logs: `hermes gateway run`
- Verifique se o template prompt está renderizando corretamente
- Confirme que o destino de entrega está configurado e conectado

### Duplicate responses

- O cache de idempotency deve evitar isso — verifique se a fonte webhook envia header de delivery ID (`X-GitHub-Delivery` ou `X-Request-ID`)
- Delivery IDs são cacheados por 1 hora

### `gh` CLI errors (GitHub comment delivery)

- Execute `gh auth login` no host gateway
- Garanta que o usuário GitHub autenticado tem write access ao repositório
- Verifique se `gh` está instalado e no PATH

---

## Environment Variables {#environment-variables}

| Variable | Description | Default |
|----------|-------------|---------|
| `WEBHOOK_ENABLED` | Habilita o adapter plataforma webhook | `false` |
| `WEBHOOK_PORT` | Porta HTTP do servidor para receber webhooks | `8644` |
| `WEBHOOK_SECRET` | Secret HMAC global (fallback quando rotas não especificam o próprio) | _(nenhum)_ |
