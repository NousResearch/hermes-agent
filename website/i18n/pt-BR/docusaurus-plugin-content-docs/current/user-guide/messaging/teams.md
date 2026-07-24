---
sidebar_position: 5
title: "Microsoft Teams"
description: "Configure o Hermes Agent como bot do Microsoft Teams"
---

# Microsoft Teams Setup

Conecte o Hermes Agent ao Microsoft Teams como bot. Diferente do Socket Mode do Slack, o Teams entrega mensagens chamando um **webhook HTTPS público**, então sua instância precisa de um endpoint publicamente acessível — seja um túnel de dev (desenvolvimento local) ou um domínio real (produção).

Precisa de resumos de reunião a partir de eventos Microsoft Graph em vez de conversas normais de bot? Use a página dedicada: [Teams Meetings](/user-guide/messaging/teams-meetings).

> Execute `hermes gateway setup` e escolha **Microsoft Teams** para um walk-through guiado.

## How the Bot Responds {#how-the-bot-responds}

| Context | Behavior |
|---------|----------|
| **Chat pessoal (DM)** | O bot responde a toda mensagem. Sem @mention necessária. |
| **Chat em grupo** | O bot só responde quando @mencionado. |
| **Canal** | O bot só responde quando @mencionado. |

O Teams entrega @mentions como mensagens normais com tags `<at>BotName</at>`, que o Hermes remove automaticamente antes do processamento.

---

Para instalações a partir do código ou locais, inclua o extra Teams para que o adapter embutido possa importar o SDK Microsoft Teams:

```bash
uv sync --extra teams
# ou, para instalações editáveis:
uv pip install -e ".[teams]"
```

## Step 1: Install the Teams CLI {#step-1-install-the-teams-cli}

O `@microsoft/teams.cli` automatiza o registro do bot — sem portal Azure.

```bash
npm install -g @microsoft/teams.cli@preview
teams login
```

Para verificar seu login e encontrar seu próprio object ID AAD (necessário para `TEAMS_ALLOWED_USERS`):

```bash
teams status --verbose
```

---

## Step 2: Expose the Webhook Port {#step-2-expose-the-webhook-port}

O Teams não consegue entregar mensagens em `localhost`. Para desenvolvimento local, use qualquer ferramenta de túnel para obter uma URL HTTPS pública. A porta padrão é `3978` — altere com `TEAMS_PORT` se necessário.

```bash
# devtunnel (Microsoft)
devtunnel create hermes-bot --allow-anonymous
devtunnel port create hermes-bot -p 3978 --protocol https  # substitua 3978 por TEAMS_PORT se alterado
devtunnel host hermes-bot

# ngrok
ngrok http 3978  # substitua 3978 por TEAMS_PORT se alterado

# cloudflared
cloudflared tunnel --url http://localhost:3978  # substitua 3978 por TEAMS_PORT se alterado
```

Copie a URL `https://` da saída — você vai usá-la no próximo passo. Deixe o túnel rodando enquanto desenvolve.

Para produção, aponte o endpoint do bot para o domínio público do seu servidor (veja [Production Deployment](#production-deployment)).

---

## Step 3: Create the Bot {#step-3-create-the-bot}

```bash
teams app create \
  --name "Hermes" \
  --endpoint "https://<your-tunnel-url>/api/messages"
```

A CLI exibe seu `CLIENT_ID`, `CLIENT_SECRET` e `TENANT_ID`, além de um link de instalação para o Passo 6. Salve o client secret — ele não será exibido novamente.

---

## Step 4: Configure Environment Variables {#step-4-configure-environment-variables}

Adicione em `~/.hermes/.env`:

```bash
# Obrigatório
TEAMS_CLIENT_ID=<your-client-id>
TEAMS_CLIENT_SECRET=<your-client-secret>
TEAMS_TENANT_ID=<your-tenant-id>

# Restringir acesso a usuários específicos (recomendado)
# Use object IDs AAD de `teams status --verbose`
TEAMS_ALLOWED_USERS=<your-aad-object-id>
```

---

## Step 5: Start the Gateway {#step-5-start-the-gateway}

```bash
HERMES_UID=$(id -u) HERMES_GID=$(id -g) docker compose up -d gateway
```

Isso inicia o gateway. A porta padrão de webhook é `3978` (sobrescreva com `TEAMS_PORT`). Verifique se está rodando:

```bash
curl http://localhost:3978/health   # deve retornar: ok
docker logs -f hermes
```

Procure por:
```
[teams] Webhook server listening on 0.0.0.0:3978/api/messages
```

---

## Step 6: Install the App in Teams {#step-6-install-the-app-in-teams}

```bash
teams app get <teamsAppId> --install-link
```

Abra o link impresso no navegador — ele abre diretamente no cliente Teams. Após instalar, envie uma mensagem direta ao bot — está pronto.

---

## Configuration Reference {#configuration-reference}

### Environment Variables

| Variable | Description |
|----------|-------------|
| `TEAMS_CLIENT_ID` | ID do App Azure AD (client) |
| `TEAMS_CLIENT_SECRET` | Client secret Azure AD |
| `TEAMS_TENANT_ID` | ID do tenant Azure AD |
| `TEAMS_ALLOWED_USERS` | Object IDs AAD separados por vírgula autorizados a usar o bot |
| `TEAMS_ALLOW_ALL_USERS` | Defina `true` para pular a allowlist e permitir qualquer pessoa |
| `TEAMS_HOME_CHANNEL` | ID de conversa para entrega de cron/mensagens proativas |
| `TEAMS_HOME_CHANNEL_NAME` | Nome de exibição do canal home |
| `TEAMS_PORT` | Porta do webhook (padrão: `3978`) |

### config.yaml

Alternativamente, configure via `~/.hermes/config.yaml`:

```yaml
platforms:
  teams:
    enabled: true
    extra:
      client_id: "your-client-id"
      client_secret: "your-secret"
      tenant_id: "your-tenant-id"
      port: 3978
```

---

## Features {#features}

### Interactive Approval Cards

Quando o agente precisa executar um comando potencialmente perigoso, ele envia um Adaptive Card com quatro botões em vez de pedir para você digitar `/approve`:

- **Allow Once** — aprovar este comando específico
- **Allow Session** — aprovar este padrão pelo resto da sessão
- **Always Allow** — aprovar permanentemente este padrão
- **Deny** — rejeitar o comando

Clicar em um botão resolve a aprovação inline e substitui o card pela decisão.

### Meeting Summary Delivery (Teams Meeting Pipeline)

Quando o [plugin pipeline de reuniões Teams](/user-guide/messaging/msgraph-webhook) está habilitado, este adapter também trata a entrega outbound de resumos de reunião — uma superfície de integração Teams, não duas. Após a transcrição de uma reunião ser resumida, o writer publica o resumo no destino Teams escolhido.

A entrega de resumo do pipeline é configurada na entrada `teams` da plataforma junto com a config do bot:

```yaml
platforms:
  teams:
    enabled: true
    extra:
      # config existente do bot (client_id, client_secret, tenant_id, port) ...

      # Entrega de resumo de reunião (só usada quando o plugin teams_pipeline está habilitado)
      delivery_mode: "graph"       # ou "incoming_webhook"
      # Para delivery_mode: graph — escolha UM de:
      chat_id: "19:meeting_..."    # postar em um chat Teams
      # team_id: "..."             # OU postar em um canal
      # channel_id: "..."
      # access_token: "..."        # opcional; faz fallback para credenciais MSGRAPH_*
      # Para delivery_mode: incoming_webhook:
      # incoming_webhook_url: "https://outlook.office.com/webhook/..."
```

| Mode | Use when | Trade-off |
|------|----------|-----------|
| `incoming_webhook` | Post simples "publique um resumo neste canal" com URL estática gerada pelo Teams. | Sem threading de resposta, sem reações, aparece com a identidade configurada do webhook. |
| `graph` | Posts em canal com thread ou posts em chat 1:1/grupo sob a identidade do bot via Microsoft Graph. | Requer o [registro de app Graph](/guides/microsoft-graph-app-registration) com permissões de aplicação `ChannelMessage.Send` (canal) ou `Chat.ReadWrite.All` (chat). |

Se o plugin `teams_pipeline` **não** estiver habilitado, essas configurações são inertes — só entram em ação quando o runtime do pipeline faz bind no ingress Graph webhook.

---

## Production Deployment {#production-deployment}

Para um servidor permanente, pule devtunnel e registre seu bot com o endpoint HTTPS público do servidor:

```bash
teams app create \
  --name "Hermes" \
  --endpoint "https://your-domain.com/api/messages"
```

Se você já criou o bot e só precisa atualizar o endpoint:

```bash
teams app update --id <teamsAppId> --endpoint "https://your-domain.com/api/messages"
```

Garanta que sua porta configurada (`TEAMS_PORT`, padrão `3978`) seja acessível da internet e que seu certificado TLS seja válido — o Teams rejeita certificados autoassinados.

---

## Troubleshooting {#troubleshooting}

| Problem | Solution |
|---------|----------|
| Endpoint `health` funciona mas o bot não responde | Verifique se o túnel ainda está rodando e se o endpoint de mensagens do bot corresponde à URL do túnel |
| `KeyError: 'teams'` nos logs | Reinicie o container — isso está corrigido na versão atual |
| Bot responde com erros de auth | Verifique se `TEAMS_CLIENT_ID`, `TEAMS_CLIENT_SECRET` e `TEAMS_TENANT_ID` estão corretos |
| `No inference provider configured` | Verifique se `ANTHROPIC_API_KEY` (ou outra chave de provedor) está definida em `~/.hermes/.env` |
| Bot recebe mensagens mas as ignora | Seu object ID AAD pode não estar em `TEAMS_ALLOWED_USERS`. Execute `teams status --verbose` para encontrá-lo |
| URL do túnel muda ao reiniciar | URLs devtunnel são persistentes se você usar um túnel nomeado (`devtunnel create hermes-bot`). ngrok e cloudflared geram nova URL a cada execução a menos que você tenha plano pago — atualize o endpoint do bot com `teams app update` quando mudar |
| Teams mostra "This bot is not responding" | O webhook retornou erro. Verifique `docker logs hermes` em busca de tracebacks |
| `[teams] Failed to connect` nos logs | O SDK falhou ao autenticar. Confira suas credenciais e se o tenant ID corresponde à conta usada em `teams login` |

---

## Security {#security}

:::warning
**Sempre defina `TEAMS_ALLOWED_USERS`** com os object IDs AAD de usuários autorizados. Sem isso, qualquer pessoa que encontrar ou instalar seu bot pode interagir com ele.

Trate `TEAMS_CLIENT_SECRET` como senha — rotacione periodicamente via portal Azure ou Teams CLI.
:::

- Armazene credenciais em `~/.hermes/.env` com permissões `600` (`chmod 600 ~/.hermes/.env`)
- O bot só aceita mensagens de usuários em `TEAMS_ALLOWED_USERS`; mensagens não autorizadas são descartadas silenciosamente
- Seu endpoint público (`/api/messages`) é autenticado pelo Teams Bot Framework — requisições sem JWTs válidos são rejeitadas

## Related Docs {#related-docs}

- [Teams Meetings](/user-guide/messaging/teams-meetings)
- [Operate the Teams Meeting Pipeline](/guides/operate-teams-meeting-pipeline)
