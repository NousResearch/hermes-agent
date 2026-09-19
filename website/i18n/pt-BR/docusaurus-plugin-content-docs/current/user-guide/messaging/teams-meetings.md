---
sidebar_position: 6
title: "Teams Meetings"
description: "Configure o pipeline de resumo de reuniões do Microsoft Teams com webhooks do Microsoft Graph"
---

# Microsoft Teams Meetings

Use o pipeline de reuniões do Teams quando quiser que o Hermes ingira eventos de reunião do Microsoft Graph, busque transcrições primeiro, faça fallback para gravações mais STT quando necessário e entregue um resumo estruturado em destinos downstream.

Pré-requisitos: veja [Microsoft Teams](./teams.md) para a configuração subjacente do bot/credenciais.

> Execute `hermes gateway setup` e escolha **Teams Meetings** para um walk-through guiado.

Esta página foca em configuração e habilitação:
- credenciais Graph
- configuração do listener de webhook
- modos de entrega Teams
- formato de config do pipeline

Para operações do dia a dia, verificações de go-live e a planilha do operador, use o guia dedicado: [Operate the Teams Meeting Pipeline](/guides/operate-teams-meeting-pipeline).

## What This Feature Does {#what-this-feature-does}

O pipeline:
1. recebe eventos de webhook do Microsoft Graph
2. resolve a reunião e prefere artefatos de transcrição primeiro
3. faz fallback para download de gravação mais STT quando não há transcrição utilizável
4. armazena estado durável de jobs e registros de sink localmente
5. pode escrever resumos em Notion, Linear e Microsoft Teams

Ações do operador ficam na CLI (o subcomando `teams-pipeline` é registrado pelo plugin `teams_pipeline` — habilite via `hermes plugins enable teams_pipeline` ou defina `plugins.enabled: [teams_pipeline]` em `config.yaml`):

```bash
hermes teams-pipeline validate
hermes teams-pipeline list
hermes teams-pipeline maintain-subscriptions
```

## Prerequisites {#prerequisites}

Antes de habilitar o pipeline de reuniões, certifique-se de ter:

- uma instalação funcional do Hermes
- a [configuração existente do bot Microsoft Teams](/user-guide/messaging/teams) se quiser entrega outbound Teams
- credenciais de aplicação Microsoft Graph com as permissões exigidas para os recursos de reunião que pretende assinar
- uma URL HTTPS pública que o Microsoft Graph possa chamar para entrega de webhook
- `ffmpeg` instalado se quiser fallback gravação-mais-STT

## Step 1: Add Microsoft Graph Credentials {#step-1-add-microsoft-graph-credentials}

Adicione credenciais app-only Graph em `~/.hermes/.env`:

```bash
MSGRAPH_TENANT_ID=<tenant-id>
MSGRAPH_CLIENT_ID=<client-id>
MSGRAPH_CLIENT_SECRET=<client-secret>
```

Essas credenciais são usadas por:
- a fundação do cliente Graph
- comandos de manutenção de assinatura
- resolução de reunião e buscas de artefatos
- entrega outbound Teams baseada em Graph quando você não fornece um token Teams dedicado

## Step 2: Enable the Graph Webhook Listener {#step-2-enable-the-graph-webhook-listener}

O listener de webhook é uma plataforma gateway chamada `msgraph_webhook`. No mínimo, habilite e defina um valor de client state:

```bash
MSGRAPH_WEBHOOK_ENABLED=true
MSGRAPH_WEBHOOK_HOST=127.0.0.1
MSGRAPH_WEBHOOK_PORT=8646
MSGRAPH_WEBHOOK_CLIENT_STATE=<random-shared-secret>
MSGRAPH_WEBHOOK_ACCEPTED_RESOURCES=communications/onlineMeetings
```

O listener expõe:
- `/msgraph/webhook` para notificações Graph
- `/health` para um health check simples

Você precisa rotear seu endpoint HTTPS público para esse listener. Por exemplo, se seu domínio público é `https://ops.example.com`, sua URL de notificação Graph seria tipicamente:

```text
https://ops.example.com/msgraph/webhook
```

## Step 3: Configure Teams Delivery and Pipeline Behavior {#step-3-configure-teams-delivery-and-pipeline-behavior}

O pipeline de reuniões lê sua config de runtime da entrada existente `teams` da plataforma. Knobs específicos do pipeline ficam em `teams.extra.meeting_pipeline`. A entrega outbound Teams permanece na superfície normal de config da plataforma Teams.

Exemplo `~/.hermes/config.yaml`:

```yaml
platforms:
  msgraph_webhook:
    enabled: true
    extra:
      host: 127.0.0.1
      port: 8646
      client_state: "replace-me"
      accepted_resources:
        - "communications/onlineMeetings"

  teams:
    enabled: true
    extra:
      client_id: "your-teams-client-id"
      client_secret: "your-teams-client-secret"
      tenant_id: "your-teams-tenant-id"

      # entrega de resumo outbound
      delivery_mode: "graph" # ou incoming_webhook
      team_id: "team-id"
      channel_id: "channel-id"
      # incoming_webhook_url: "https://..."

      meeting_pipeline:
        transcript_min_chars: 80
        transcript_required: false
        transcription_fallback: true
        ffmpeg_extract_audio: true
        notion:
          enabled: false
        linear:
          enabled: false
```

Se você fizer bind do listener em um host não-loopback como `0.0.0.0`, também deve definir `allowed_source_cidrs` para os ranges de egress de webhook da Microsoft. Binds loopback (`127.0.0.1` / `::1`) são a configuração prevista para dev-tunnel e reverse proxy local.

## Teams Delivery Modes {#teams-delivery-modes}

O pipeline suporta dois modos de entrega de resumo Teams dentro do plugin Teams existente.

### `incoming_webhook`

Use quando quiser um post webhook simples no Teams sem criação de mensagem de canal via Graph.

Config obrigatória:

```yaml
platforms:
  teams:
    enabled: true
    extra:
      delivery_mode: "incoming_webhook"
      incoming_webhook_url: "https://..."
```

### `graph`

Use quando quiser que o Hermes publique o resumo via Microsoft Graph em um chat ou canal Teams.

Destinos suportados:
- `chat_id`
- `team_id` + `channel_id`
- `team_id` + fallback `home_channel` para a plataforma Teams existente

Exemplo:

```yaml
platforms:
  teams:
    enabled: true
    extra:
      delivery_mode: "graph"
      team_id: "team-id"
      channel_id: "channel-id"
```

## Step 4: Start the Gateway {#step-4-start-the-gateway}

Inicie o Hermes normalmente após atualizar a config:

```bash
hermes gateway run
```

Ou, se você roda o Hermes em Docker, inicie o gateway da mesma forma que já faz no seu deployment.

Verifique o listener:

```bash
curl http://localhost:8646/health
```

## Step 5: Create Graph Subscriptions {#step-5-create-graph-subscriptions}

Use a CLI do plugin para criar e inspecionar assinaturas.

Exemplos:

```bash
hermes teams-pipeline subscribe \
  --resource communications/onlineMeetings/getAllTranscripts \
  --notification-url https://ops.example.com/msgraph/webhook \
  --client-state "$MSGRAPH_WEBHOOK_CLIENT_STATE"

hermes teams-pipeline subscribe \
  --resource communications/onlineMeetings/getAllRecordings \
  --notification-url https://ops.example.com/msgraph/webhook \
  --client-state "$MSGRAPH_WEBHOOK_CLIENT_STATE"
```

:::warning Graph subscriptions expire in 72 hours

O Microsoft Graph limita assinaturas de webhook a 72 horas e não as renova automaticamente. Você DEVE agendar `hermes teams-pipeline maintain-subscriptions` antes de ir para produção, ou as notificações param silenciosamente três dias após qualquer criação manual de assinatura. Veja [Automating subscription renewal](/guides/operate-teams-meeting-pipeline#automating-subscription-renewal-required-for-production) no runbook do operador — três opções (cron Hermes, timer systemd, crontab simples).

:::

Para manutenção de assinatura e fluxos operacionais do dia a dia, continue com o guia: [Operate the Teams Meeting Pipeline](/guides/operate-teams-meeting-pipeline).

## Validation {#validation}

Execute o snapshot de validação embutido:

```bash
hermes teams-pipeline validate
```

Verificações complementares úteis:

```bash
hermes teams-pipeline token-health
hermes teams-pipeline subscriptions
```

## Troubleshooting {#troubleshooting}

| Problem | What to check |
|---------|---------------|
| Validação de webhook Graph falha | Confirme que a URL pública está correta e acessível, e que o Graph está chamando exatamente o caminho `/msgraph/webhook` |
| Jobs não aparecem em `hermes teams-pipeline list` | Confirme que `msgraph_webhook` está habilitado e que assinaturas apontam para a URL de notificação correta |
| Transcript-first nunca funciona | Verifique permissões Graph para recursos de transcrição e se o artefato de transcrição existe para aquela reunião |
| Fallback de gravação falha | Confirme que `ffmpeg` está instalado e que o app Graph pode acessar artefatos de gravação |
| Entrega de resumo Teams falha | Reveja `delivery_mode`, IDs de destino e config de auth Teams |

## Related Docs {#related-docs}

- [Microsoft Teams bot setup](/user-guide/messaging/teams)
- [Operate the Teams Meeting Pipeline](/guides/operate-teams-meeting-pipeline)
