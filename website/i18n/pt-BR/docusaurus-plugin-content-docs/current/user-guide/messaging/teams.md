---
sidebar_position: 5
title: "Microsoft Teams"
description: "Configure o Hermes Agent como bot do Microsoft Teams"
---

# Configuração do Microsoft Teams

Conecte o Hermes Agent ao Microsoft Teams como bot. Diferente do Socket Mode do Slack, o Teams entrega mensagens chamando um **webhook HTTPS público**, então sua instância precisa de um endpoint publicamente acessível — seja um túnel de desenvolvimento (dev local) ou um domínio real (produção).

Precisa de resumos de reuniões a partir de eventos do Microsoft Graph em vez de conversas normais de bot? Use a página dedicada: [Reuniões do Teams](/user-guide/messaging/teams-meetings).

> Execute `hermes gateway setup` e escolha **Microsoft Teams** para um passo a passo guiado.

## Como o bot responde {#how-the-bot-responds}

| Contexto | Comportamento |
|---------|----------|
| **Chat pessoal (DM)** | O bot responde a toda mensagem. Não precisa de @mention. |
| **Chat em grupo** | O bot só responde quando @mencionado. |
| **Canal** | O bot só responde quando @mencionado. |

O Teams entrega @mentions como mensagens normais com tags `<at>BotName</at>`, que o Hermes remove automaticamente antes do processamento.

---

Para instalações a partir do código-fonte ou locais, inclua o extra Teams para o adaptador incluído poder
importar o SDK do Microsoft Teams:

```bash
uv sync --extra teams
# or, for editable installs:
uv pip install -e ".[teams]"
```

## Passo 1: Instalar a CLI do Teams {#step-1-install-the-teams-cli}

O `@microsoft/teams.cli` automatiza o registro do bot — sem portal Azure.

```bash
npm install -g @microsoft/teams.cli@preview
teams login
```

Para verificar seu login e encontrar seu próprio object ID do AAD (necessário para `TEAMS_ALLOWED_USERS`):

```bash
teams status --verbose
```

---

## Passo 2: Expor a porta do webhook {#step-2-expose-the-webhook-port}

O Teams não pode entregar mensagens em `localhost`. Para desenvolvimento local, use qualquer ferramenta de túnel para obter uma URL HTTPS pública. A porta padrão é `3978` — altere com `TEAMS_PORT` se necessário.

```bash
# devtunnel (Microsoft)
devtunnel create hermes-bot --allow-anonymous
devtunnel port create hermes-bot -p 3978 --protocol http  # replace 3978 with TEAMS_PORT if changed
devtunnel host hermes-bot

# ngrok
ngrok http 3978  # replace 3978 with TEAMS_PORT if changed

# cloudflared
cloudflared tunnel --url http://localhost:3978  # replace 3978 with TEAMS_PORT if changed
```

Copie a URL `https://` da saída — você usará no próximo passo. Deixe o túnel rodando enquanto desenvolve.

A URL pública do túnel usa HTTPS, mas o listener local de webhook do Hermes usa HTTP simples. O túnel termina TLS e encaminha HTTP para a porta `3978`; não configure a porta local do túnel como HTTPS.

Para produção, aponte o endpoint do bot ao domínio público do seu servidor (veja [Implantação em produção](#production-deployment)).

---

## Passo 3: Criar o bot {#step-3-create-the-bot}

```bash
teams app create \
  --name "Hermes" \
  --endpoint "https://<your-tunnel-url>/api/messages"
```

A CLI exibe seu `CLIENT_ID`, `CLIENT_SECRET` e `TENANT_ID`, além de um link de instalação para o Passo 6. Salve o client secret — ele não será exibido novamente.

---

## Passo 4: Configurar variáveis de ambiente {#step-4-configure-environment-variables}

Adicione em `~/.hermes/.env`:

```bash
# Required
TEAMS_CLIENT_ID=<your-client-id>
TEAMS_CLIENT_SECRET=<your-client-secret>
TEAMS_TENANT_ID=<your-tenant-id>

# Restrict access to specific users (recommended)
# Use AAD object IDs from `teams status --verbose`
TEAMS_ALLOWED_USERS=<your-aad-object-id>
```

---

## Passo 5: Iniciar o gateway {#step-5-start-the-gateway}

**Docker** (deve rodar do diretório que contém `docker-compose.yml` — em geral seu clone `hermes-agent`, não `~`):

```bash
cd /path/to/hermes-agent
HERMES_UID=$(id -u) HERMES_GID=$(id -g) docker compose up -d gateway
```

**Instalação nativa / systemd** (instalador one-liner típico `hermes` em `~/.hermes/hermes-agent`):

```bash
hermes gateway restart
# or foreground: hermes gateway run
```

O SDK do Teams é opcional; quando o Teams está habilitado, o gateway instala preguiçosamente no venv próprio do Hermes na primeira inicialização (não use `pip install` do sistema no Ubuntu 24.04 — isso encontra PEP 668 `externally-managed-environment`). Para instalar manualmente no venv do Hermes:

```bash
~/.hermes/hermes-agent/venv/bin/pip install microsoft-teams-apps aiohttp
# or from a clone of the agent: uv sync --extra teams
```

A porta padrão do webhook é `3978` (sobrescreva com `TEAMS_PORT`). Verifique se está rodando:

```bash
curl http://localhost:3978/health   # should return: ok
# Docker:
docker logs -f hermes
# Native:
hermes gateway status -l
```

Procure por:
```
[teams] Webhook server listening on * (all interfaces, IPv4+IPv6):3978/api/messages
```

---

## Passo 6: Instalar o app no Teams {#step-6-install-the-app-in-teams}

```bash
teams app get <teamsAppId> --install-link
```

Abra o link impresso no navegador — ele abre direto no cliente Teams. Depois de instalar, envie uma mensagem direta ao seu bot — está pronto.

---

## Referência de configuração {#configuration-reference}

### Variáveis de ambiente {#environment-variables}

| Variável | Descrição |
|----------|-------------|
| `TEAMS_CLIENT_ID` | ID do App (client) Azure AD |
| `TEAMS_CLIENT_SECRET` | Client secret Azure AD |
| `TEAMS_TENANT_ID` | ID do tenant Azure AD |
| `TEAMS_ALLOWED_USERS` | Object IDs AAD separados por vírgula autorizados a usar o bot |
| `TEAMS_ALLOW_ALL_USERS` | Defina `true` para pular a allowlist e permitir qualquer pessoa |
| `TEAMS_HOME_CHANNEL` | ID de conversa para entrega de mensagens cron/proativas |
| `TEAMS_HOME_CHANNEL_NAME` | Nome de exibição do canal home |
| `TEAMS_PORT` | Porta do webhook (padrão: `3978`) |

### config.yaml {#configyaml}

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

## Recursos {#features}

### Cartões de aprovação interativos {#interactive-approval-cards}

Quando o agente precisa executar um comando potencialmente perigoso, envia um Adaptive Card com quatro botões em vez de pedir para digitar `/approve`:

- **Allow Once** — aprova este comando específico
- **Allow Session** — aprova este padrão pelo resto da sessão
- **Always Allow** — aprova permanentemente este padrão
- **Deny** — rejeita o comando

Clicar em um botão resolve a aprovação inline e substitui o cartão pela decisão.

### Entrega de resumo de reunião (pipeline de reuniões Teams) {#meeting-summary-delivery-teams-meeting-pipeline}

Quando o [plugin de pipeline de reuniões Teams](/user-guide/messaging/msgraph-webhook) está habilitado, este adaptador também trata a entrega outbound de resumos de reunião — uma superfície de integração Teams, não duas. Depois que a transcrição de uma reunião é resumida, o writer publica o resumo no destino Teams escolhido.

A entrega de resumo do pipeline é configurada na entrada `teams` da plataforma junto à config do bot:

```yaml
platforms:
  teams:
    enabled: true
    extra:
      # existing bot config (client_id, client_secret, tenant_id, port) ...

      # Meeting summary delivery (only used when the teams_pipeline plugin is enabled)
      delivery_mode: "graph"       # or "incoming_webhook"
      # For delivery_mode: graph — pick ONE of:
      chat_id: "19:meeting_..."    # post into a Teams chat
      # team_id: "..."             # OR post into a channel
      # channel_id: "..."
      # access_token: "..."        # optional; falls back to MSGRAPH_* app credentials
      # For delivery_mode: incoming_webhook:
      # incoming_webhook_url: "https://outlook.office.com/webhook/..."
```

| Modo | Use quando | Trade-off |
|------|----------|-----------|
| `incoming_webhook` | "Postar um resumo neste canal" simples com URL estática gerada pelo Teams. | Sem threading de resposta, sem reações, aparece como a identidade configurada do webhook. |
| `graph` | Posts em canal com thread ou posts em chat 1:1/grupo sob a identidade do bot via Microsoft Graph. | Requer o [registro de app Graph](/guides/microsoft-graph-app-registration) com permissões de aplicação `ChannelMessage.Send` (canal) ou `Chat.ReadWrite.All` (chat). |

Se o plugin `teams_pipeline` **não** estiver habilitado, essas configurações são inertes — só entram em ação quando o runtime do pipeline se liga ao ingress webhook Graph.

---

## Implantação em produção {#production-deployment}

Para um servidor permanente, termine TLS num reverse proxy e encaminhe as requisições para o listener HTTP simples do Hermes, normalmente `http://127.0.0.1:3978`. Registre o endpoint HTTPS público do proxy com o Teams:

```bash
teams app create \
  --name "Hermes" \
  --endpoint "https://your-domain.com/api/messages"
```

Se você já criou o bot e só precisa atualizar o endpoint:

```bash
teams app update --id <teamsAppId> --endpoint "https://your-domain.com/api/messages"
```

Certifique-se de que o endpoint HTTPS público seja alcançável da internet e use um certificado TLS válido. O Teams rejeita certificados autoassinados. Mantenha o listener do Hermes atrás do proxy; a porta `3978` não serve HTTPS por si.

---

## Solução de problemas {#troubleshooting}

| Problema | Solução |
|---------|----------|
| `Can't find a suitable configuration file` do `docker compose` | Você não está no repositório que tem `docker-compose.yml`, ou está em instalação nativa — use `hermes gateway restart`, ou `cd` no clone primeiro |
| `requirements not met` / `Teams SDK missing` / `No adapter available for teams` | Reinicie o gateway para a instalação preguiçosa rodar, ou instale no **venv do Hermes**: `~/.hermes/hermes-agent/venv/bin/pip install microsoft-teams-apps aiohttp`. `pip` do sistema falha no Ubuntu 24.04 (PEP 668) e não afetaria o serviço mesmo |
| Endpoint `health` funciona mas o bot não responde | Verifique se o túnel ainda está rodando e se o endpoint de mensagens do bot corresponde à URL do túnel |
| Logs mostram `"UNKNOWN / HTTP/1.0" 400` quando o Teams envia uma mensagem | O túnel ou reverse proxy está encaminhando HTTPS para o listener HTTP simples do Hermes. Termine TLS no proxy e encaminhe HTTP para a porta `3978` |
| `KeyError: 'teams'` nos logs | Reinicie o container — isso está corrigido na versão atual |
| Bot responde com erros de auth | Verifique se `TEAMS_CLIENT_ID`, `TEAMS_CLIENT_SECRET` e `TEAMS_TENANT_ID` estão corretos |
| `No inference provider configured` | Verifique se `ANTHROPIC_API_KEY` (ou outra chave de provedor) está em `~/.hermes/.env` |
| Bot recebe mensagens mas ignora | Seu object ID AAD pode não estar em `TEAMS_ALLOWED_USERS`. Execute `teams status --verbose` para encontrá-lo |
| URL do túnel muda ao reiniciar | URLs devtunnel são persistentes se usar um túnel nomeado (`devtunnel create hermes-bot`). ngrok e cloudflared geram URL nova a cada execução salvo plano pago — atualize o endpoint do bot com `teams app update` quando mudar |
| Teams mostra "This bot is not responding" | O webhook retornou erro. Verifique `docker logs hermes` / `hermes gateway status -l` por tracebacks |
| `[teams] Failed to connect` nos logs | O SDK falhou ao autenticar. Confira credenciais e se o tenant ID corresponde à conta usada em `teams login` |

---

## Segurança {#security}

:::warning
**Sempre defina `TEAMS_ALLOWED_USERS`** com os object IDs AAD dos usuários autorizados. Sem isso, qualquer pessoa que encontrar ou instalar seu bot pode interagir com ele.

Trate `TEAMS_CLIENT_SECRET` como senha — rotacione periodicamente via portal Azure ou Teams CLI.
:::

- Armazene credenciais em `~/.hermes/.env` com permissões `600` (`chmod 600 ~/.hermes/.env`)
- O bot só aceita mensagens de usuários em `TEAMS_ALLOWED_USERS`; mensagens não autorizadas são descartadas silenciosamente
- Seu endpoint público (`/api/messages`) é autenticado pelo Teams Bot Framework — requisições sem JWTs válidos são rejeitadas

## Documentação relacionada {#related-docs}

- [Reuniões do Teams](/user-guide/messaging/teams-meetings)
- [Operar o pipeline de reuniões Teams](/guides/operate-teams-meeting-pipeline)
