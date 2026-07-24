---
sidebar_position: 11
sidebar_label: "Revisões de PR do GitHub via Webhook"
title: "Comentários Automatizados em PRs do GitHub via Webhooks"
description: "Conecte o Hermes ao GitHub para que ele busque automaticamente diffs de PRs, revise alterações de código e poste comentários — acionado por webhooks, sem prompts manuais"
---

# Comentários Automatizados em PRs do GitHub via Webhooks

Este guia mostra como conectar o Hermes Agent ao GitHub para que ele busque automaticamente o diff de um pull request, analise as alterações de código e poste um comentário — acionado por um evento de webhook, sem prompts manuais.

Quando um PR é aberto ou atualizado, o GitHub envia um POST de webhook para sua instância do Hermes. O Hermes executa o agente com um prompt que instrui a buscar o diff via a CLI `gh`, e a resposta é postada de volta na thread do PR.

:::tip Quer uma configuração mais simples, sem endpoint público?
Se você não tem uma URL pública ou só quer começar rapidamente, veja [Construa um Agente de Revisão de PRs do GitHub](./github-pr-review-agent.md) — usa tarefas agendadas (cron jobs) para consultar por PRs em um cronograma, funciona atrás de NAT e firewalls.
:::

:::info Documentação de referência
Para a referência completa da plataforma de webhooks (todas as opções de configuração, tipos de entrega, inscrições dinâmicas, modelo de segurança), veja [Webhooks](/user-guide/messaging/webhooks).
:::

:::warning Risco de injeção de prompt
Os payloads de webhook contêm dados controlados por atacantes — títulos de PR, mensagens de commit e descrições podem conter instruções maliciosas. Quando seu endpoint de webhook está exposto à internet, rode o gateway em um ambiente isolado (Docker, backend SSH). Veja a [seção de segurança](#security-notes) abaixo.
:::

---

## Pré-requisitos {#prerequisites}

- Hermes Agent instalado e em execução (`hermes gateway`)
- [`gh` CLI](https://cli.github.com/) instalado e autenticado no host do gateway (`gh auth login`)
- Uma URL publicamente acessível para sua instância do Hermes (veja [Testes locais com ngrok](#local-testing-with-ngrok) se estiver rodando localmente)
- Acesso de administrador ao repositório do GitHub (necessário para gerenciar webhooks)

---

## Passo 1 — Ative a plataforma de webhook {#step-1--enable-the-webhook-platform}

Adicione o seguinte ao seu `~/.hermes/config.yaml`:

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      port: 8644          # default; change if another service occupies this port
      rate_limit: 30      # max requests per minute per route (not a global cap)

      routes:
        github-pr-review:
          secret: "your-webhook-secret-here"   # must match the GitHub webhook secret exactly
          events:
            - pull_request

          # The agent is instructed to fetch the actual diff before reviewing.
          # {number} and {repository.full_name} are resolved from the GitHub payload.
          prompt: |
            A pull request event was received (action: {action}).

            PR #{number}: {pull_request.title}
            Author: {pull_request.user.login}
            Branch: {pull_request.head.ref} → {pull_request.base.ref}
            Description: {pull_request.body}
            URL: {pull_request.html_url}

            If the action is "closed" or "labeled", stop here and do not post a comment.

            Otherwise:
            1. Run: gh pr diff {number} --repo {repository.full_name}
            2. Review the code changes for correctness, security issues, and clarity.
            3. Write a concise, actionable review comment and post it.

          deliver: github_comment
          deliver_extra:
            repo: "{repository.full_name}"
            pr_number: "{number}"
```

**Campos principais:**

| Campo | Descrição |
|---|---|
| `secret` (nível da rota) | Segredo HMAC para esta rota. Recorre ao `extra.secret` global se omitido. |
| `events` | Lista de valores do cabeçalho `X-GitHub-Event` a aceitar. Lista vazia = aceita todos. |
| `prompt` | Modelo (template); `{field}` e `{nested.field}` são resolvidos a partir do payload do GitHub. |
| `deliver` | `github_comment` posta via `gh pr comment`. `log` apenas escreve no log do gateway. |
| `deliver_extra.repo` | Resolve para, por exemplo, `org/repo` a partir do payload. |
| `deliver_extra.pr_number` | Resolve para o número do PR a partir do payload. |

:::note O payload não contém código
O payload de webhook do GitHub inclui metadados do PR (título, descrição, nomes de branch, URLs), mas **não o diff**. O prompt acima instrui o agente a executar `gh pr diff` para buscar as alterações reais. A ferramenta `terminal` está incluída no toolset padrão `hermes-webhook`, então nenhuma configuração extra é necessária.
:::

---

## Passo 2 — Inicie o gateway {#step-2--start-the-gateway}

```bash
hermes gateway
```

Você deve ver:

```
[webhook] Listening on 0.0.0.0:8644 — routes: github-pr-review
```

Verifique se está em execução:

```bash
curl http://localhost:8644/health
# {"status": "ok", "platform": "webhook"}
```

---

## Passo 3 — Registre o webhook no GitHub {#step-3--register-the-webhook-on-github}

1. Vá até seu repositório → **Settings** → **Webhooks** → **Add webhook**
2. Preencha:
   - **Payload URL:** `https://your-public-url.example.com/webhooks/github-pr-review`
   - **Content type:** `application/json`
   - **Secret:** o mesmo valor que você definiu em `secret` na configuração da rota
   - **Which events?** → Selecione eventos individuais → marque **Pull requests**
3. Clique em **Add webhook**

O GitHub enviará imediatamente um evento `ping` para confirmar a conexão. Ele é ignorado com segurança — `ping` não está na sua lista `events` — e retorna `{"status": "ignored", "event": "ping"}`. Ele só é registrado no nível de log DEBUG, então não aparecerá no console no nível de log padrão.

---

## Passo 4 — Abra um PR de teste {#step-4--open-a-test-pr}

Crie uma branch, envie uma alteração e abra um PR. Em 30–90 segundos (dependendo do tamanho do PR e do modelo), o Hermes deve postar um comentário de revisão.

Para acompanhar o progresso do agente em tempo real:

```bash
tail -f "${HERMES_HOME:-$HOME/.hermes}/logs/gateway.log"
```

---

## Testes locais com ngrok {#local-testing-with-ngrok}

Se o Hermes estiver rodando no seu notebook, use o [ngrok](https://ngrok.com/) para expô-lo:

```bash
ngrok http 8644
```

Copie a URL `https://...ngrok-free.app` e use-a como sua Payload URL no GitHub. No plano gratuito do ngrok, a URL muda cada vez que o ngrok é reiniciado — atualize seu webhook do GitHub a cada sessão. Contas pagas do ngrok têm um domínio estático.

Você pode testar uma rota estática diretamente com `curl` — sem necessidade de conta do GitHub ou de um PR real.

:::tip Use `deliver: log` ao testar localmente
Mude `deliver: github_comment` para `deliver: log` na sua configuração enquanto testa. Caso contrário, o agente tentará postar um comentário no repositório fictício `org/repo#99` do payload de teste, o que falhará. Volte para `deliver: github_comment` quando estiver satisfeito com a saída do prompt.
:::

```bash
SECRET="your-webhook-secret-here"
BODY='{"action":"opened","number":99,"pull_request":{"title":"Test PR","body":"Adds a feature.","user":{"login":"testuser"},"head":{"ref":"feat/x"},"base":{"ref":"main"},"html_url":"https://github.com/org/repo/pull/99"},"repository":{"full_name":"org/repo"}}'
SIG=$(printf '%s' "$BODY" | openssl dgst -sha256 -hmac "$SECRET" -hex | awk '{print "sha256="$2}')

curl -s -X POST http://localhost:8644/webhooks/github-pr-review \
  -H "Content-Type: application/json" \
  -H "X-GitHub-Event: pull_request" \
  -H "X-Hub-Signature-256: $SIG" \
  -d "$BODY"
# Expected: {"status":"accepted","route":"github-pr-review","event":"pull_request","delivery_id":"..."}
```

Depois observe o agente executar:
```bash
tail -f "${HERMES_HOME:-$HOME/.hermes}/logs/gateway.log"
```

:::note
`hermes webhook test <name>` só funciona para **inscrições dinâmicas** criadas com `hermes webhook subscribe`. Ele não lê rotas do `config.yaml`.
:::

---

## Filtrando ações específicas {#filtering-to-specific-actions}

O GitHub envia eventos `pull_request` para várias ações: `opened`, `synchronize`, `reopened`, `closed`, `labeled`, etc. A lista `events` filtra pelo valor do cabeçalho `X-GitHub-Event`, e o campo `filters` no nível da rota pode restringir por campos do payload, como `action`.

O prompt no Passo 1 já trata isso instruindo o agente a parar antecipadamente para eventos `closed` e `labeled`.

:::warning O agente ainda roda e consome tokens
A instrução "pare aqui" evita uma revisão significativa, mas o agente ainda roda até o fim para todo evento `pull_request`, independentemente da ação. Prefira filtrar antes que o agente seja acionado:

```yaml
filters:
  - field: "action"
    in: ["opened", "synchronize", "reopened"]
```

Para repositórios de alto volume, você ainda pode filtrar upstream com um workflow do GitHub Actions que chama sua URL de webhook condicionalmente.
:::

> Não há sintaxe de template Jinja2 ou condicional. `{field}` e `{nested.field}` são as únicas substituições suportadas. Qualquer outra coisa é passada literalmente para o agente.

---

## Usando uma skill para um estilo de revisão consistente {#using-a-skill-for-consistent-review-style}

Carregue uma [skill do Hermes](/user-guide/features/skills) para dar ao agente uma persona de revisão consistente. Adicione `skills` à sua rota dentro de `platforms.webhook.extra.routes` no `config.yaml`:

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      routes:
        github-pr-review:
          secret: "your-webhook-secret-here"
          events: [pull_request]
          prompt: |
            A pull request event was received (action: {action}).
            PR #{number}: {pull_request.title} by {pull_request.user.login}
            URL: {pull_request.html_url}

            If the action is "closed" or "labeled", stop here and do not post a comment.

            Otherwise:
            1. Run: gh pr diff {number} --repo {repository.full_name}
            2. Review the diff using your review guidelines.
            3. Write a concise, actionable review comment and post it.
          skills:
            - review
          deliver: github_comment
          deliver_extra:
            repo: "{repository.full_name}"
            pr_number: "{number}"
```

> **Nota:** Apenas a primeira skill da lista que for encontrada é carregada. O Hermes não empilha múltiplas skills — entradas subsequentes são ignoradas.

---

## Enviando respostas para o Slack ou Discord {#sending-responses-to-slack-or-discord-instead}

Substitua os campos `deliver` e `deliver_extra` dentro da sua rota pela plataforma de destino:

```yaml
# Inside platforms.webhook.extra.routes.<route-name>:

# Slack
deliver: slack
deliver_extra:
  chat_id: "C0123456789"   # Slack channel ID (omit to use the configured home channel)

# Discord
deliver: discord
deliver_extra:
  chat_id: "987654321012345678"  # Discord channel ID (omit to use home channel)
```

A plataforma de destino também precisa estar ativada e conectada no gateway. Se `chat_id` for omitido, a resposta é enviada ao canal principal configurado dessa plataforma.

Valores válidos para `deliver`: `log` · `github_comment` · `telegram` · `discord` · `slack` · `signal` · `sms`

---

## Suporte ao GitLab {#gitlab-support}

O mesmo adaptador funciona com o GitLab. O GitLab usa `X-Gitlab-Token` para autenticação (comparação de string simples, não HMAC) — o Hermes lida com ambos automaticamente.

Para filtragem de eventos, o GitLab define `X-GitLab-Event` com valores como `Merge Request Hook`, `Push Hook`, `Pipeline Hook`. Use o valor exato do cabeçalho em `events`:

```yaml
events:
  - Merge Request Hook
```

Os campos do payload do GitLab diferem dos do GitHub — por exemplo, `{object_attributes.title}` para o título do MR e `{object_attributes.iid}` para o número do MR. A maneira mais fácil de descobrir a estrutura completa do payload é o botão **Test** do GitLab nas configurações do seu webhook, combinado com o log **Recent Deliveries**. Alternativamente, omita `prompt` da configuração da sua rota — o Hermes então passará o payload completo como JSON formatado diretamente para o agente, e a resposta do agente (visível no log do gateway com `deliver: log`) descreverá sua estrutura.

---

## Notas de segurança {#security-notes}

- **Nunca use `INSECURE_NO_AUTH`** em produção — isso desativa completamente a validação de assinatura. É apenas para desenvolvimento local.
- **Gire seu segredo de webhook** periodicamente e atualize-o tanto no GitHub (configurações do webhook) quanto no seu `config.yaml`.
- **A limitação de taxa** é de 30 req/min por rota por padrão (configurável via `extra.rate_limit`). Excedê-la retorna `429`.
- **Entregas duplicadas** (retentativas de webhook) são deduplicadas via um cache de idempotência de 1 hora. A chave do cache é `X-GitHub-Delivery`, se presente, depois `X-Request-ID`, depois um timestamp em milissegundos. Quando nenhum cabeçalho de ID de entrega está definido, as retentativas **não** são deduplicadas.
- **Injeção de prompt:** títulos de PR, descrições e mensagens de commit são controlados por atacantes. PRs maliciosos poderiam tentar manipular as ações do agente. Rode o gateway em um ambiente isolado (Docker, VM) quando exposto à internet pública.

---

## Solução de Problemas {#troubleshooting}

| Sintoma | Verificação |
|---|---|
| `401 Invalid signature` | O segredo no config.yaml não corresponde ao segredo do webhook do GitHub |
| `404 Unknown route` | O nome da rota na URL não corresponde à chave em `routes:` |
| `429 Rate limit exceeded` | Excedidas 30 req/min por rota — comum ao reenviar eventos de teste pela interface do GitHub; espere um minuto ou aumente `extra.rate_limit` |
| Nenhum comentário postado | `gh` não está instalado, não está no PATH, ou não está autenticado (`gh auth login`) |
| O agente roda, mas não há comentário | Verifique o log do gateway — se a saída do agente estava vazia ou era apenas "SKIP", a entrega ainda é tentada |
| Porta já em uso | Altere `extra.port` no config.yaml |
| O agente roda, mas revisa apenas a descrição do PR | O prompt não está incluindo a instrução `gh pr diff` — o diff não está no payload do webhook |
| Não consigo ver o evento ping | Eventos ignorados retornam `{"status":"ignored","event":"ping"}` apenas no nível de log DEBUG — verifique o log de entregas do GitHub (repositório → Settings → Webhooks → seu webhook → Recent Deliveries) |

**A aba Recent Deliveries do GitHub** (repositório → Settings → Webhooks → seu webhook) mostra os cabeçalhos exatos da requisição, o payload, o status HTTP e o corpo da resposta de cada entrega. É a forma mais rápida de diagnosticar falhas sem precisar olhar os logs do seu servidor.

---

## Referência completa de configuração {#full-config-reference}

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      port: 8644               # listen port (default: 8644)
      secret: ""               # optional global fallback secret
      rate_limit: 30           # requests per minute per route
      max_body_bytes: 1048576  # payload size limit in bytes (default: 1 MB)

      routes:
        <route-name>:
          secret: "required-per-route"
          events: []            # [] = accept all; otherwise list X-GitHub-Event values
          prompt: ""            # {field} / {nested.field} resolved from payload
          skills: []            # first matching skill is loaded (only one)
          deliver: "log"        # log | github_comment | telegram | discord | slack | signal | sms
          deliver_extra: {}     # repo + pr_number for github_comment; chat_id for others
```

---

## O Que Vem a Seguir? {#whats-next}

- **[Revisões de PR Baseadas em Cron](./github-pr-review-agent.md)** — consulte por PRs em um cronograma, sem necessidade de endpoint público
- **[Referência de Webhook](/user-guide/messaging/webhooks)** — referência completa de configuração para a plataforma de webhook
- **[Construa um Plugin](/developer-guide/plugins)** — empacote a lógica de revisão em um plugin compartilhável
- **[Perfis](/user-guide/profiles)** — rode um perfil de revisor dedicado com sua própria memória e configuração
