---
sidebar_position: 8
sidebar_label: "SMS (Twilio)"
title: "SMS (Twilio)"
description: "Configure o Hermes Agent como chatbot SMS via Twilio"
---

# Configuração de SMS (Twilio) {#sms-setup-twilio}

O Hermes se conecta ao SMS pela API [Twilio](https://www.twilio.com/). As pessoas enviam SMS para seu número Twilio e recebem respostas de IA — a mesma experiência conversacional do Telegram ou Discord, mas por mensagens de texto padrão.

:::info Credenciais compartilhadas
O gateway SMS compartilha credenciais com a [skill de telefonia](/reference/skills-catalog) opcional. Se você já configurou Twilio para chamadas de voz ou SMS avulsos, o gateway funciona com os mesmos `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN` e `TWILIO_PHONE_NUMBER`.
:::

---

## Pré-requisitos {#prerequisites}

- **Conta Twilio** — [Cadastre-se em twilio.com](https://www.twilio.com/try-twilio) (trial gratuito disponível)
- **Um número de telefone Twilio** com capacidade SMS
- **Um servidor publicamente acessível** — o Twilio envia webhooks ao seu servidor quando SMS chegam
- **aiohttp** — `cd ~/.hermes/hermes-agent && uv pip install -e ".[sms]"`

---

## Passo 1: Obtenha suas credenciais Twilio {#step-1-get-your-twilio-credentials}

1. Acesse o [Twilio Console](https://console.twilio.com/)
2. Copie seu **Account SID** e **Auth Token** do dashboard
3. Vá em **Phone Numbers → Manage → Active Numbers** — anote seu número de telefone no formato E.164 (ex.: `+15551234567`)

---

## Passo 2: Configure o Hermes {#step-2-configure-hermes}

### Setup interativo (recomendado) {#interactive-setup-recommended}

```bash
hermes gateway setup
```

Selecione **SMS (Twilio)** na lista de plataformas. O assistente solicitará suas credenciais.

### Setup manual {#manual-setup}

Adicione em `~/.hermes/.env`:

```bash
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_PHONE_NUMBER=+15551234567

# Security: restrict to specific phone numbers (recommended)
SMS_ALLOWED_USERS=+15559876543,+15551112222

# Optional: set a home channel for cron job delivery
SMS_HOME_CHANNEL=+15559876543
```

---

## Passo 3: Configure o webhook Twilio {#step-3-configure-twilio-webhook}

O Twilio precisa saber para onde enviar mensagens recebidas. No [Twilio Console](https://console.twilio.com/):

1. Vá em **Phone Numbers → Manage → Active Numbers**
2. Clique no seu número de telefone
3. Em **Messaging → A MESSAGE COMES IN**, defina:
   - **Webhook**: `https://your-server:8080/webhooks/twilio`
   - **HTTP Method**: `POST`

:::tip Expondo seu webhook
Se você roda o Hermes localmente, use um túnel para expor o webhook:

```bash
# Using cloudflared
cloudflared tunnel --url http://localhost:8080

# Using ngrok
ngrok http 8080
```

Defina a URL pública resultante como webhook Twilio.
:::

**Defina `SMS_WEBHOOK_URL` com a mesma URL que você configurou no Twilio.** Isso é necessário para validação de assinatura Twilio — o adaptador se recusa a iniciar sem isso:

```bash
# Must match the webhook URL in your Twilio Console
SMS_WEBHOOK_URL=https://your-server:8080/webhooks/twilio
```

A porta do webhook padrão é `8080`. Sobrescreva com:

```bash
SMS_WEBHOOK_PORT=3000
```

---

## Passo 4: Inicie o gateway {#step-4-start-the-gateway}

```bash
hermes gateway
```

Você deve ver:

```
[sms] Twilio webhook server listening on 127.0.0.1:8080, from: +1555***4567
```

Se vir `Refusing to start: SMS_WEBHOOK_URL is required`, defina `SMS_WEBHOOK_URL` com a URL pública configurada no Twilio Console (veja Passo 3).

Envie SMS para seu número Twilio — o Hermes responderá via SMS.

---

## Variáveis de ambiente {#environment-variables}

| Variable | Required | Description |
|----------|----------|-------------|
| `TWILIO_ACCOUNT_SID` | Yes | Twilio Account SID (começa com `AC`) |
| `TWILIO_AUTH_TOKEN` | Yes | Twilio Auth Token (também usado para validação de assinatura de webhook) |
| `TWILIO_PHONE_NUMBER` | Yes | Seu número Twilio (formato E.164) |
| `SMS_WEBHOOK_URL` | Yes | URL pública para validação de assinatura Twilio — deve corresponder ao webhook no Twilio Console |
| `SMS_WEBHOOK_PORT` | No | Porta do listener de webhook (padrão: `8080`) |
| `SMS_WEBHOOK_HOST` | No | Endereço de bind do webhook (padrão: `127.0.0.1`) |
| `SMS_INSECURE_NO_SIGNATURE` | No | Defina `true` para desabilitar validação de assinatura (apenas dev local — **não para produção**) |
| `SMS_ALLOWED_USERS` | No | Números E.164 separados por vírgula autorizados a conversar |
| `SMS_ALLOW_ALL_USERS` | No | Defina `true` para permitir qualquer pessoa (não recomendado) |
| `SMS_HOME_CHANNEL` | No | Número de telefone para entrega de jobs cron / notificações |
| `SMS_HOME_CHANNEL_NAME` | No | Nome de exibição do canal home (padrão: `Home`) |

---

## Comportamento específico de SMS {#sms-specific-behavior}

- **Apenas texto puro** — Markdown é removido automaticamente, já que SMS o renderiza como caracteres literais
- **Limite de 1600 caracteres** — Respostas mais longas são divididas em várias mensagens em limites naturais (quebras de linha, depois espaços)
- **Prevenção de eco** — Mensagens do seu próprio número Twilio são ignoradas para evitar loops
- **Redação de número de telefone** — Números de telefone são redigidos nos logs por privacidade

---

## Segurança {#security}

### Validação de assinatura de webhook {#webhook-signature-validation}

O Hermes valida que webhooks recebidos realmente vêm do Twilio verificando o header `X-Twilio-Signature` (HMAC-SHA1). Isso impede que atacantes injetem mensagens falsas.

**`SMS_WEBHOOK_URL` é obrigatório.** Defina-o com a URL pública configurada no Twilio Console. O adaptador se recusa a iniciar sem isso.

Para desenvolvimento local sem URL pública, você pode desabilitar a validação:

```bash
# Local dev only — NOT for production
SMS_INSECURE_NO_SIGNATURE=true
```

### Allowlists de usuário {#user-allowlists}

**O gateway nega todos os usuários por padrão.** Configure uma allowlist:

```bash
# Recommended: restrict to specific phone numbers
SMS_ALLOWED_USERS=+15559876543,+15551112222

# Or allow all (NOT recommended for bots with terminal access)
SMS_ALLOW_ALL_USERS=true
```

:::warning
SMS não tem criptografia built-in. Não use SMS para operações sensíveis a menos que você entenda as implicações de segurança. Para casos sensíveis, prefira Signal ou Telegram.
:::

---

## Solução de problemas {#troubleshooting}

### Mensagens não chegam {#messages-not-arriving}

1. Verifique se a URL do webhook Twilio está correta e publicamente acessível
2. Confirme que `TWILIO_ACCOUNT_SID` e `TWILIO_AUTH_TOKEN` estão corretos
3. Verifique Twilio Console → **Monitor → Logs → Messaging** por erros de entrega
4. Certifique-se de que seu número está em `SMS_ALLOWED_USERS` (ou `SMS_ALLOW_ALL_USERS=true`)

### Respostas não são enviadas {#replies-not-sending}

1. Verifique se `TWILIO_PHONE_NUMBER` está definido corretamente (formato E.164 com `+`)
2. Confirme que sua conta Twilio tem números com capacidade SMS
3. Verifique os logs do gateway Hermes por erros da API Twilio

### Conflitos de porta do webhook {#webhook-port-conflicts}

Se a porta 8080 já estiver em uso, altere:

```bash
SMS_WEBHOOK_PORT=3001
```

Atualize a URL do webhook no Twilio Console para corresponder.
