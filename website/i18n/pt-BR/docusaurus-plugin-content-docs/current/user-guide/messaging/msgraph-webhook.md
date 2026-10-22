---
sidebar_position: 23
title: "Microsoft Graph Webhook Listener"
description: "Receba notificações de mudança do Microsoft Graph (reuniões, calendário, chat, etc.) no Hermes"
---

# Microsoft Graph Webhook Listener {#microsoft-graph-webhook-listener}

A plataforma de gateway `msgraph_webhook` é um listener de eventos de entrada. É como o Hermes recebe **notificações de mudança** do Microsoft Graph — "uma reunião Teams terminou", "uma nova mensagem chegou neste chat", "este evento de calendário foi atualizado". Diferente da plataforma `teams` (que é um bot de chat com o qual usuários digitam) — esta é o M365 avisando o Hermes que algo aconteceu, não uma pessoa.

Hoje o consumidor principal é o pipeline de resumo de reuniões Teams: o Graph notifica quando uma reunião produz transcrição, o pipeline a busca e o Hermes publica um resumo de volta no Teams. Outros recursos Graph (`/chats/.../messages`, `/users/.../events`) usam o mesmo listener — os consumidores de pipeline chegam com seus próprios PRs.

## Pré-requisitos {#prerequisites}

- Credenciais de aplicativo Microsoft Graph — [Registre um aplicativo Microsoft Graph](/guides/microsoft-graph-app-registration)
- Uma **URL HTTPS pública** que o Microsoft Graph consiga alcançar (Graph não chama endpoints privados). Um dev tunnel funciona para testes; produção precisa de um domínio real com certificado válido.
- Um secret compartilhado forte para usar como valor `clientState`. Gere com `openssl rand -hex 32` e coloque em `~/.hermes/.env` como `MSGRAPH_WEBHOOK_CLIENT_STATE`.

## Início rápido {#quick-start}

Mínimo em `~/.hermes/config.yaml`:

```yaml
platforms:
  msgraph_webhook:
    enabled: true
    extra:
      host: 127.0.0.1
      port: 8646
      client_state: "replace-with-a-strong-secret"
      accepted_resources:
        - "communications/onlineMeetings"
```

Ou via env vars em `~/.hermes/.env` (auto-merge na inicialização):

```bash
MSGRAPH_WEBHOOK_ENABLED=true
MSGRAPH_WEBHOOK_PORT=8646
MSGRAPH_WEBHOOK_CLIENT_STATE=<generate-with-openssl-rand-hex-32>
MSGRAPH_WEBHOOK_ACCEPTED_RESOURCES=communications/onlineMeetings
```

Nota: o host de bind é lido de `extra.host` em `config.yaml` (veja o exemplo acima); não há override por env var `MSGRAPH_WEBHOOK_HOST`.

Inicie o gateway: `hermes gateway run`. O listener expõe:

- `POST /msgraph/webhook` — notificações de mudança do Graph
- `GET /msgraph/webhook?validationToken=...` — handshake de validação de subscription Graph
- `GET /health` — probe de readiness com contadores accepted/duplicate

Exponha o listener publicamente (reverse proxy, dev tunnel, ingress). Sua notification URL para subscriptions Graph é sua origem HTTPS pública seguida de `/msgraph/webhook`:

```
https://ops.example.com/msgraph/webhook
```

## Configuração {#configuration}

Todas as configurações ficam sob `platforms.msgraph_webhook.extra`:

| Setting | Default | Description |
|---------|---------|-------------|
| `host` | `0.0.0.0` | Endereço de bind do listener HTTP. Binds não-loopback exigem `allowed_source_cidrs`; loopback (`127.0.0.1` / `::1`) é o setup mais fácil para dev-tunnel / reverse-proxy. |
| `port` | `8646` | Porta de bind. |
| `webhook_path` | `/msgraph/webhook` | Caminho de URL para onde o Graph faz POST. |
| `health_path` | `/health` | Endpoint de readiness. |
| `client_state` | — | Secret compartilhado que o Graph ecoa em toda notificação. Comparado com `hmac.compare_digest` — gere com `openssl rand -hex 32`. |
| `accepted_resources` | `[]` (accept all) | Allowlist de caminhos/padrões de recurso Graph. `*` final atua como prefix match. `/` inicial é tolerado. Exemplo: `["communications/onlineMeetings", "chats/*/messages"]`. |
| `max_seen_receipts` | `5000` | Tamanho do cache dedupe para IDs de notificação. Entradas mais antigas são evictadas quando o cap é atingido. |
| `allowed_source_cidrs` | `[]` | Obrigatório para binds não-loopback. Deixe vazio apenas quando o listener está bound em loopback e fronteado por túnel local / reverse proxy. |

A maioria das configurações também tem env var equivalente (`MSGRAPH_WEBHOOK_*`) que faz merge na config na inicialização do gateway (a exceção é `host`, que é só config — veja a nota acima) — veja a [referência de variáveis de ambiente](/reference/environment-variables#microsoft-graph-teams-meetings).

## Endurecimento de segurança {#security-hardening}

### clientState é a verificação de auth principal {#clientstate-is-the-primary-auth-check}

Toda notificação Graph inclui a string `clientState` com a qual sua subscription se registrou. O listener rejeita qualquer notificação cujo `clientState` não corresponda, usando comparação timing-safe. Este é o mecanismo documentado pela Microsoft — trate o valor como secret compartilhado forte.

Se `client_state` não estiver definido, o listener se recusa a iniciar.

### Allowlist por IP de origem (deployments de produção) {#source-ip-allowlisting-production-deployments}

Para produção, restrinja o listener aos ranges de IP de origem de webhook Graph publicados pela Microsoft. A Microsoft documenta os ranges de egress sob o [Office 365 IP Address and URL Web service](https://learn.microsoft.com/en-us/microsoft-365/enterprise/urls-and-ip-address-ranges). Configure-os como:

```yaml
platforms:
  msgraph_webhook:
    enabled: true
    extra:
      host: 0.0.0.0
      client_state: "..."
      allowed_source_cidrs:
        - "52.96.0.0/14"
        - "52.104.0.0/14"
        # ...add the current Microsoft 365 "Common" + "Teams" category egress ranges
```

Ou como env var:

```bash
MSGRAPH_WEBHOOK_ALLOWED_SOURCE_CIDRS="52.96.0.0/14,52.104.0.0/14"
```

Fazer bind de um host não-loopback como `0.0.0.0`, `::`, ou um IP LAN sem `allowed_source_cidrs` é recusado na inicialização. Se você usa dev tunnel ou reverse proxy na mesma máquina, faça bind do Hermes em `127.0.0.1` ou `::1` e deixe a allowlist vazia lá. Strings CIDR inválidas logam aviso e são ignoradas. **Revise a lista de IPs Microsoft trimestralmente** — ela muda.

### Terminação HTTPS {#https-termination}

O listener fala HTTP simples. Termine TLS no seu reverse proxy (Caddy, Nginx, Cloudflare Tunnel, AWS ALB) e faça proxy ao listener pela rede local. O Graph se recusa a entregar em endpoints não-HTTPS, então não há caminho para tráfego não criptografado chegar a você do Graph em si.

### Higiene de resposta {#response-hygiene}

Em sucesso o listener retorna `202 Accepted` com corpo vazio — contadores internos ficam fora da resposta wire. Operadores podem observar contagens via `/health`, guardado pelas mesmas regras de IP de origem do caminho webhook.

Tabela de códigos de status:

| Outcome | Status |
|---------|--------|
| Notification(s) accepted or deduped | 202 |
| Validation handshake (GET with `validationToken`) | 200 (echoes the token) |
| Every item in batch failed clientState | 403 |
| Malformed JSON / missing `value` array / unknown resource | 400 |
| Source IP not in allowlist | 403 |
| Bare GET without `validationToken` | 400 |

## Solução de problemas {#troubleshooting}

| Problem | What to check |
|---------|---------------|
| Validação de subscription Graph falha | URL pública alcançável, caminho `/msgraph/webhook` corresponde, GET com `validationToken` ecoa o token verbatim como `text/plain` em 10 segundos. |
| Notificações POST mas nada é ingerido | `client_state` corresponde ao que você registrou na subscription. Reexecute `openssl rand -hex 32` e crie nova subscription se o valor divergiu. Verifique se `accepted_resources` inclui o caminho de recurso que o Graph envia. |
| Toda notificação retorna 403 | Mismatch de `clientState` (forjado, ou subscription registrada com valor diferente). Recrie a subscription com `hermes teams-pipeline subscribe --client-state "$MSGRAPH_WEBHOOK_CLIENT_STATE" ...` (incluído no PR do runtime do pipeline). |
| Listener se recusa a iniciar em `0.0.0.0` | Defina `allowed_source_cidrs` com os ranges de egress webhook Microsoft atuais, ou faça bind do Hermes em `127.0.0.1` / `::1` atrás do seu túnel ou reverse proxy. |
| Listener inicia mas `curl http://localhost:8646/health` trava | Colisão de bind de porta. Verifique `ss -tlnp \| grep 8646` e altere `port:` se necessário. |
| Requisições Graph reais da Microsoft retornam 403 | Allowlist de IP de origem estreita demais. Amplie a lista para incluir os ranges de egress Microsoft atuais. Se ainda valida o caminho do túnel, faça bind do Hermes em loopback e deixe o túnel cuidar da exposição pública. |

## Documentação relacionada {#related-docs}

- [Registre um aplicativo Microsoft Graph](/guides/microsoft-graph-app-registration) — pré-requisito de registro de app Azure
- [Variáveis de ambiente → Microsoft Graph](/reference/environment-variables#microsoft-graph-teams-meetings) — lista completa de env vars
- [Setup do bot Microsoft Teams](/user-guide/messaging/teams) — a plataforma diferente que permite usuários conversarem com o Hermes no Teams
