---
sidebar_position: 17
title: "LINE"
description: "Configure o Hermes Agent como bot da LINE Messaging API"
---

# Configuração do LINE {#line-setup}

Execute o Hermes Agent como bot [LINE](https://line.me/) via a LINE Messaging API oficial. O adaptador vive como plugin de plataforma incluído em `plugins/platforms/line/` — sem edições no core, basta habilitá-lo como qualquer outra plataforma.

O LINE é o app de mensagens dominante no Japão, Taiwan e Tailândia. Se seus usuários moram lá, é assim que eles te alcançam.

> Execute `hermes gateway setup` e escolha **LINE** para um passo a passo guiado.

## Como o bot responde {#how-the-bot-responds}

| Context | Behavior |
|---------|----------|
| **1:1 chat** (`U` IDs) | Responde a toda mensagem |
| **Group chat** (`C` IDs) | Responde quando o grupo está na allowlist |
| **Multi-user room** (`R` IDs) | Responde quando a sala está na allowlist |

Texto, imagens, áudio, vídeo, arquivos, stickers e localizações recebidos são todos tratados. Texto enviado usa o **reply token gratuito primeiro** (uso único, janela ~60s) e recorre à Push API tarifada quando o token expirou.

---

## Passo 1: Crie um canal LINE Messaging API {#step-1-create-a-line-messaging-api-channel}

1. Acesse o [LINE Developers Console](https://developers.line.biz/console/).
2. Crie um Provider e, sob ele, um canal **Messaging API**.
3. Na aba **Basic settings** do canal, copie o **Channel secret**.
4. Na aba **Messaging API**, role até **Channel access token (long-lived)** e clique **Issue**. Copie o token.
5. Na aba **Messaging API**, desabilite também **Auto-reply messages** e **Greeting messages** para não conflitarem com as respostas do bot.

---

## Passo 2: Exponha a porta do webhook {#step-2-expose-the-webhook-port}

O LINE entrega webhooks por HTTPS público. A porta padrão é `8646` — sobrescreva com `LINE_PORT` se necessário.

```bash
# Cloudflare Tunnel (recommended for production — fixed hostname)
cloudflared tunnel --url http://localhost:8646

# ngrok (good for dev)
ngrok http 8646

# devtunnel
devtunnel create hermes-line --allow-anonymous
devtunnel port create hermes-line -p 8646 --protocol https
devtunnel host hermes-line
```

Copie a URL `https://...` — você a definirá como webhook URL abaixo. **Mantenha o túnel rodando** durante testes. Para produção, configure um Cloudflare named tunnel fixo para a URL do webhook não mudar ao reiniciar.

---

## Passo 3: Configure o Hermes {#step-3-configure-hermes}

Adicione em `~/.hermes/.env`:

```env
LINE_CHANNEL_ACCESS_TOKEN=YOUR_LONG_LIVED_TOKEN
LINE_CHANNEL_SECRET=YOUR_CHANNEL_SECRET

# Allowlist — at least one of these (or LINE_ALLOW_ALL_USERS=true for dev)
LINE_ALLOWED_USERS=U1234567890abcdef...           # comma-separated U-prefixed IDs
LINE_ALLOWED_GROUPS=C1234567890abcdef...          # optional group IDs
LINE_ALLOWED_ROOMS=R1234567890abcdef...           # optional room IDs

# Required for image / audio / video sends — the public HTTPS base URL
# the tunnel resolves to.  Without it, send_image/voice/video will refuse.
LINE_PUBLIC_URL=https://my-tunnel.example.com
```

Depois em `~/.hermes/config.yaml`:

```yaml
gateway:
  platforms:
    line:
      enabled: true
```

Isso basta — o scan de plugin incluído em `gateway/config.py` encontra automaticamente `plugins/platforms/line/`. Sem editar enum `Platform.LINE`, sem registro em `_create_adapter`.

---

## Passo 4: Defina a webhook URL {#step-4-set-the-webhook-url}

De volta no console LINE:

1. Abra seu canal → aba **Messaging API**.
2. Em **Webhook settings** → **Webhook URL**, cole `https://<your-tunnel>/line/webhook` (note o caminho `/line/webhook` — o adaptador escuta lá).
3. Clique **Verify**. O LINE faz ping na URL; você deve ver 200.
4. Ative **Use webhook** para **On**.

---

## Passo 5: Execute o gateway {#step-5-run-the-gateway}

```bash
hermes gateway
```

O log do agente mostra:

```
LINE: webhook listening on 0.0.0.0:8646/line/webhook (public: https://my-tunnel.example.com)
```

Adicione o bot como amigo no app LINE (escaneie o QR na aba **Messaging API** do canal) e envie uma mensagem.

---

## Respostas lentas de LLM {#slow-llm-responses}

O reply token do LINE é de uso único e expira em cerca de 60 segundos após o evento recebido. LLMs lentos não conseguem responder a tempo, o que normalmente forçaria uma chamada Push API paga.

Quando o LLM ainda está rodando após `LINE_SLOW_RESPONSE_THRESHOLD` segundos (padrão `45`), o adaptador consome o reply token original para enviar um bubble **Template Buttons**:

> 🤔 Still thinking. Tap below to fetch the answer when it's ready.
>
> [ Get answer ]

O usuário toca **Get answer** quando conveniente — esse postback entrega um reply token *novo*, que o adaptador usa para enviar a resposta em cache (ainda gratuita).

Máquina de estados: `PENDING → READY → DELIVERED`, mais `ERROR` para execuções canceladas (o PENDING órfão resolve para "Run was interrupted before completion." após `/stop` para o botão persistente não entrar em loop).

Para desabilitar o botão postback e sempre usar fallback Push:

```env
LINE_SLOW_RESPONSE_THRESHOLD=0
```

Para o fluxo postback disparar de forma confiável, suprima chatter que consumiria o reply token antes do threshold:

```yaml
# ~/.hermes/config.yaml
display:
  interim_assistant_messages: false
  platforms:
    line:
      tool_progress: off
```

---

## Entrega de cron / notificações {#cron--notification-delivery}

```env
LINE_HOME_CHANNEL=Uxxxxxxxxxxxxxxxxxxxx     # default delivery target
```

Jobs cron com `deliver: line` roteiam para `LINE_HOME_CHANNEL`. O adaptador inclui um sender Push-only standalone para jobs cron funcionarem mesmo quando o cron roda em processo separado do gateway.

---

## Referência de variáveis de ambiente {#environment-variable-reference}

| Variable | Required | Default | Description |
|---|---|---|---|
| `LINE_CHANNEL_ACCESS_TOKEN` | yes | — | Channel access token long-lived |
| `LINE_CHANNEL_SECRET` | yes | — | Channel secret (verificação webhook HMAC-SHA256) |
| `LINE_HOST` | no | `0.0.0.0` | Host de bind do webhook |
| `LINE_PORT` | no | `8646` | Porta de bind do webhook |
| `LINE_PUBLIC_URL` | for media | — | Base URL HTTPS pública; obrigatória para envio de image/voice/video |
| `LINE_ALLOWED_USERS` | one of | — | IDs de usuário separados por vírgula (prefixo U) |
| `LINE_ALLOWED_GROUPS` | one of | — | IDs de grupo separados por vírgula (prefixo C) |
| `LINE_ALLOWED_ROOMS` | one of | — | IDs de sala separados por vírgula (prefixo R) |
| `LINE_ALLOW_ALL_USERS` | dev only | `false` | Pular allowlist por completo |
| `LINE_HOME_CHANNEL` | no | — | Alvo padrão de entrega cron / notificação |
| `LINE_SLOW_RESPONSE_THRESHOLD` | no | `45` | Segundos antes do botão postback disparar (`0` = desabilitado) |
| `LINE_PENDING_TEXT` | no | "🤔 Still thinking…" | Texto do bubble mostrado junto ao botão postback |
| `LINE_BUTTON_LABEL` | no | "Get answer" | Rótulo do botão |
| `LINE_DELIVERED_TEXT` | no | "Already replied ✅" | Resposta quando um botão já entregue é tocado de novo |
| `LINE_INTERRUPTED_TEXT` | no | "Run was interrupted before completion." | Resposta quando um botão órfão de `/stop` é tocado |

---

## Solução de problemas {#troubleshooting}

**"invalid signature" na verificação do webhook.** O `Channel secret` foi copiado errado, ou seu túnel reescreveu o corpo da requisição. Verifique primeiro com `curl -i https://<tunnel>/line/webhook/health` — deve retornar `{"status":"ok","platform":"line"}`.

**Bot não recebe nada em grupos.** Verifique se `LINE_ALLOWED_GROUPS` inclui o ID de grupo `C...`. Para encontrar um group ID, envie uma mensagem de teste e faça grep em `~/.hermes/logs/gateway.log` por `LINE: rejecting unauthorized source` — o dict de source rejeitado tem os IDs.

**`send_image` falha com "LINE_PUBLIC_URL must be set".** A Messaging API do LINE não aceita uploads binários — imagens, áudio e vídeo precisam ser URLs HTTPS acessíveis. Defina `LINE_PUBLIC_URL` com o hostname público do túnel e o adaptador servirá arquivos de `/line/media/<token>/<filename>` automaticamente.

**Botão postback nunca aparece.** Ou o LLM respondeu mais rápido que `LINE_SLOW_RESPONSE_THRESHOLD`, ou outro bubble (tool-progress, streaming) consumiu o reply token primeiro. Veja o bloco de supressão em "Respostas lentas de LLM".

**"already in use by another profile".** O mesmo channel access token está vinculado a outro perfil Hermes rodando. Pare o outro gateway ou use um canal separado.

---

## Limitações {#limitations}

* **Caps de bubble e comprimento.** Cada bubble de texto LINE tem limite de 5000 caracteres. Respostas mais longas são divididas inteligentemente em ~4500 caracteres em até 5 bubbles por chamada Reply/Push, dividindo em limites naturais quando possível.
* **Sem edição nativa de mensagem.** O LINE não tem API de editar mensagem — respostas em streaming sempre enviam bubbles novos, nunca editam os anteriores.
* **Sem renderização Markdown.** Negrito (`**`), itálico (`*`), cercas de código e headings renderizam como caracteres literais. O adaptador os remove antes de enviar; URLs são preservadas (`[label](url)` vira `label (url)`).
* **Indicador de loading é só em DM.** O LINE rejeita a API chat/loading para grupos e salas, então o indicador de digitação só aparece em chats 1:1.
