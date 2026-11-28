---
sidebar_position: 15
---

# WeCom Callback (Self-Built App)

Conecte o Hermes ao WeCom (Enterprise WeChat) como um aplicativo empresarial self-built usando o modelo callback/webhook.

:::info WeCom Bot vs WeCom Callback
O Hermes suporta dois modos de integração WeCom:
- **[WeCom Bot](wecom.md)** — estilo bot, conecta via WebSocket. Configuração mais simples, funciona em chats de grupo.
- **WeCom Callback** (esta página) — app self-built, recebe callbacks XML criptografados. Aparece como app de primeira classe na barra lateral do WeCom dos usuários. Suporta roteamento multi-corp.
:::

Veja também: [WeCom Bot](./wecom.md) para a integração estilo bot.

> Execute `hermes gateway setup` e escolha **WeCom Callback** para um passo a passo guiado.

## Como funciona {#how-it-works}

1. Você registra um aplicativo self-built no WeCom Admin Console
2. O WeCom envia XML criptografado para seu endpoint HTTP de callback
3. O Hermes descriptografa a mensagem e a enfileira para o agente
4. Confirma imediatamente (silencioso — nada é exibido ao usuário)
5. O agente processa a solicitação (tipicamente 3–30 minutos)
6. A resposta é entregue proativamente via a API WeCom `message/send`

## Pré-requisitos {#prerequisites}

- Uma conta empresarial WeCom com acesso de administrador
- Pacotes Python `aiohttp` e `httpx` (incluídos na instalação padrão)
- Um servidor publicamente acessível para a URL de callback (ou um túnel como ngrok)

## Configuração {#setup}

### 1. Crie um app self-built no WeCom

1. Acesse [WeCom Admin Console](https://work.weixin.qq.com/) → **Applications** → **Create App**
2. Anote seu **Corp ID** (exibido no topo do console de administração)
3. Nas configurações do app, crie um **Corp Secret**
4. Anote o **Agent ID** na página de visão geral do app
5. Em **Receive Messages**, configure a URL de callback:
   - URL: `http://YOUR_PUBLIC_IP:8645/wecom/callback`
   - Token: Gere um token aleatório (o WeCom fornece um)
   - EncodingAESKey: Gere uma chave (o WeCom fornece uma)

### 2. Configure variáveis de ambiente

Adicione ao seu arquivo `.env`:

```bash
WECOM_CALLBACK_CORP_ID=your-corp-id
WECOM_CALLBACK_CORP_SECRET=your-corp-secret
WECOM_CALLBACK_AGENT_ID=1000002
WECOM_CALLBACK_TOKEN=your-callback-token
WECOM_CALLBACK_ENCODING_AES_KEY=your-43-char-aes-key

# Optional
WECOM_CALLBACK_HOST=0.0.0.0
WECOM_CALLBACK_PORT=8645
WECOM_CALLBACK_ALLOWED_USERS=user1,user2
```

### 3. Inicie o gateway

```bash
hermes gateway
```

(Use `hermes gateway start` somente depois que `hermes gateway install` tiver registrado o serviço systemd/launchd.)

O adaptador de callback inicia um servidor HTTP na porta configurada. O WeCom verificará a URL de callback via uma requisição GET e então começará a enviar mensagens via POST.

## Referência de configuração {#configuration-reference}

Defina estes em `config.yaml` em `platforms.wecom_callback.extra`, ou use variáveis de ambiente:

| Setting | Default | Description |
|---------|---------|-------------|
| `corp_id` | — | Corp ID empresarial WeCom (obrigatório) |
| `corp_secret` | — | Corp secret do app self-built (obrigatório) |
| `agent_id` | — | Agent ID do app self-built (obrigatório) |
| `token` | — | Token de verificação de callback (obrigatório) |
| `encoding_aes_key` | — | Chave AES de 43 caracteres para criptografia de callback (obrigatório) |
| `host` | `0.0.0.0` | Endereço de bind do servidor HTTP de callback |
| `port` | `8645` | Porta do servidor HTTP de callback |
| `path` | `/wecom/callback` | Caminho da URL do endpoint de callback |

## Roteamento multi-app {#multi-app-routing}

Para empresas executando vários apps self-built (por exemplo, em departamentos ou subsidiárias diferentes), configure a lista `apps` em `config.yaml`:

```yaml
platforms:
  wecom_callback:
    enabled: true
    extra:
      host: "0.0.0.0"
      port: 8645
      apps:
        - name: "dept-a"
          corp_id: "ww_corp_a"
          corp_secret: "secret-a"
          agent_id: "1000002"
          token: "token-a"
          encoding_aes_key: "key-a-43-chars..."
        - name: "dept-b"
          corp_id: "ww_corp_b"
          corp_secret: "secret-b"
          agent_id: "1000003"
          token: "token-b"
          encoding_aes_key: "key-b-43-chars..."
```

Usuários são escopados por `corp_id:user_id` para evitar colisões entre corps. Quando um usuário envia uma mensagem, o adaptador registra a qual app (corp) ele pertence e roteia respostas pelo access token correto do app.

## Controle de acesso {#access-control}

Restrinja quais usuários podem interagir com o app:

```bash
# Allowlist de usuários específicos
WECOM_CALLBACK_ALLOWED_USERS=zhangsan,lisi,wangwu

# Ou permitir todos os usuários
WECOM_CALLBACK_ALLOW_ALL_USERS=true
```

## Endpoints {#endpoints}

O adaptador expõe:

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/wecom/callback` | Handshake de verificação de URL (o WeCom envia isso durante a configuração) |
| POST | `/wecom/callback` | Callback de mensagem criptografada (o WeCom envia mensagens de usuário aqui) |
| GET | `/health` | Health check — retorna `{"status": "ok"}` |

## Criptografia {#encryption}

Todos os payloads de callback são criptografados com AES-CBC usando o EncodingAESKey. O adaptador trata:

- **Entrada**: Descriptografa payload XML, verifica assinatura SHA1
- **Saída**: Respostas enviadas via API proativa (não resposta de callback criptografada)

A implementação criptográfica é compatível com o SDK oficial WXBizMsgCrypt da Tencent.

## Limitações {#limitations}

- **Sem streaming** — respostas chegam como mensagens completas depois que o agente termina
- **Sem indicadores de digitação** — o modelo de callback não suporta status de digitação
- **Somente texto** — atualmente suporta mensagens de texto na entrada; entrada de imagem/arquivo/voz ainda não implementada. O agente está ciente das capacidades de mídia de saída via a dica de plataforma WeCom (imagens, documentos, vídeo, voz).
- **Latência de resposta** — sessões do agente levam 3–30 minutos; usuários veem a resposta quando o processamento termina

## Solução de problemas {#troubleshooting}

**Falha na verificação de assinatura.**
O WeCom assina cada requisição com o **Token** que você registrou no console de
administração. Uma incompatibilidade entre o token configurado no Hermes e o token que o
console de administração espera é a causa mais comum. Copie novamente o **Token** e o
**EncodingAESKey** do console de administração — é fácil truncá-los. Espaços em branco
nos valores de `~/.hermes/.env` em torno de `=` também quebram verificações de assinatura. Depois de
corrigir, reinicie `hermes gateway run`.

**URL de callback inacessível / etapa de verificação falha.**
O WeCom acessa a URL pública que você registrou. Confirme:
1. Seu proxy reverso / túnel encaminha `/wecom/callback` para a porta do gateway.
2. A URL no console de administração é HTTPS (o WeCom rejeita HTTP simples).
3. De fora da sua rede, `curl -i https://<your-domain>/wecom/callback`
   retorna algo diferente de timeout (um 4xx sem query params está ok —
   só significa que o listener está acessível).

**Porta inacessível / listener não vinculado.**
Verifique os logs de `hermes gateway run` para o host/porta vinculados. Se o adaptador vinculou em
`127.0.0.1`, você deve colocar um proxy reverso ou túnel na frente — os servidores do WeCom
não conseguem alcançar loopback. Defina `extra.host: 0.0.0.0` em `config.yaml` (mais
`allowed_source_cidrs` se expor diretamente) ou mantenha loopback e use um túnel
como Cloudflare Tunnel / nginx.
