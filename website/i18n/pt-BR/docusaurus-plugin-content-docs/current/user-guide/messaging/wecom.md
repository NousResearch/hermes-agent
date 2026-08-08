---
sidebar_position: 14
title: "WeCom (Enterprise WeChat)"
description: "Conecte o Hermes Agent ao WeCom via o gateway WebSocket AI Bot"
---

# WeCom (Enterprise WeChat)

Conecte o Hermes ao [WeCom](https://work.weixin.qq.com/) (企业微信), a plataforma de mensagens empresariais da Tencent. O adaptador usa o gateway WebSocket AI Bot do WeCom para comunicação bidirecional em tempo real — sem endpoint público ou webhook necessário.

Veja também: [WeCom Callback](./wecom-callback.md) para configuração de webhook de entrada.

## Pré-requisitos {#prerequisites}

- Uma conta organizacional WeCom
- Um AI Bot criado no WeCom Admin Console
- O Bot ID e Secret da página de credenciais do bot
- Pacotes Python: `aiohttp` e `httpx`

## Configuração {#setup}

### Passo 1: Crie um AI Bot

#### Recomendado: Scan-to-Create (um comando)

```bash
hermes gateway setup
```

Selecione **WeCom** e escaneie o QR code com o app móvel WeCom. O Hermes criará automaticamente um aplicativo bot com as permissões corretas e salvará as credenciais.

O assistente de configuração irá:
1. Exibir um QR code no seu terminal
2. Aguardar você escaneá-lo com o app móvel WeCom
3. Recuperar automaticamente o Bot ID e Secret
4. Guiá-lo pela configuração de controle de acesso

#### Alternativa: Configuração manual

Se scan-to-create não estiver disponível, o assistente recorre à entrada manual:

1. Faça login no [WeCom Admin Console](https://work.weixin.qq.com/wework_admin/frame)
2. Navegue até **Applications** → **Create Application** → **AI Bot**
3. Configure o nome e a descrição do bot
4. Copie o **Bot ID** e **Secret** da página de credenciais
5. Execute `hermes gateway setup`, selecione **WeCom** e insira as credenciais quando solicitado

:::warning
Mantenha o Bot Secret privado. Qualquer pessoa com ele pode se passar pelo seu bot.
:::

### Passo 2: Configure o Hermes

#### Opção A: Configuração interativa (recomendada)

```bash
hermes gateway setup
```

Selecione **WeCom** e siga as instruções. O assistente irá guiá-lo por:
- Credenciais do bot (via scan QR ou entrada manual)
- Configurações de controle de acesso (allowlist, modo pairing ou acesso aberto)
- Canal home para notificações

#### Opção B: Configuração manual

Adicione o seguinte a `~/.hermes/.env`:

```bash
WECOM_BOT_ID=your-bot-id
WECOM_SECRET=your-secret

# Optional: restrict access
WECOM_ALLOWED_USERS=user_id_1,user_id_2

# Optional: home channel for cron/notifications
WECOM_HOME_CHANNEL=chat_id
```

### Passo 3: Inicie o gateway

```bash
hermes gateway
```

## Recursos {#features}

- **Transporte WebSocket** — conexão persistente, sem endpoint público necessário
- **Mensagens DM e de grupo** — políticas de acesso configuráveis
- **Allowlists de remetente por grupo** — controle detalhado sobre quem pode interagir em cada grupo
- **Suporte a mídia** — upload e download de imagens, arquivos, voz e vídeo
- **Mídia criptografada AES** — descriptografia automática para anexos de entrada
- **Contexto de citação** — preserva encadeamento de respostas
- **Renderização Markdown** — respostas em rich text
- **Correlação de resposta** — respostas são correlacionadas ao contexto da mensagem de entrada
- **Reconexão automática** — backoff exponencial em quedas de conexão

:::note Streaming e indicadores de digitação
O adaptador WeCom entrega cada resposta como uma única mensagem completa — ele **não**
transmite respostas token a token e **não** exibe um indicador de
digitação. "Correlação de resposta" (abaixo) apenas encadeia uma resposta à sua solicitação de
entrada; não é streaming ao vivo.
:::

## Opções de configuração {#configuration-options}

Defina estes em `config.yaml` em `platforms.wecom.extra`:

| Key | Default | Description |
|-----|---------|-------------|
| `bot_id` | — | WeCom AI Bot ID (obrigatório) |
| `secret` | — | WeCom AI Bot Secret (obrigatório) |
| `websocket_url` | `wss://openws.work.weixin.qq.com` | URL do gateway WebSocket |
| `dm_policy` | `open` | Acesso a DM: `open`, `allowlist`, `disabled`, `pairing` |
| `group_policy` | `open` | Acesso a grupo: `open`, `allowlist`, `disabled` |
| `allow_from` | `[]` | IDs de usuário permitidos para DMs (quando dm_policy=allowlist) |
| `group_allow_from` | `[]` | IDs de grupo permitidos (quando group_policy=allowlist) |
| `groups` | `{}` | Configuração por grupo (veja abaixo) |

## Políticas de acesso {#access-policies}

### Política de DM

Controla quem pode enviar mensagens diretas ao bot:

| Value | Behavior |
|-------|----------|
| `open` | Qualquer pessoa pode enviar DM ao bot (padrão) |
| `allowlist` | Apenas IDs de usuário em `allow_from` podem enviar DM |
| `disabled` | Todas as DMs são ignoradas |
| `pairing` | Modo pairing (para configuração inicial) |

```bash
WECOM_DM_POLICY=allowlist
```

### Política de grupo

Controla em quais grupos o bot responde:

| Value | Behavior |
|-------|----------|
| `open` | Bot responde em todos os grupos (padrão) |
| `allowlist` | Bot responde apenas em IDs de grupo listados em `group_allow_from` |
| `disabled` | Todas as mensagens de grupo são ignoradas |

```bash
WECOM_GROUP_POLICY=allowlist
```

### Allowlists de remetente por grupo

Para controle detalhado, você pode restringir quais usuários podem interagir com o bot dentro de grupos específicos. Isso é configurado em `config.yaml`:

```yaml
platforms:
  wecom:
    enabled: true
    extra:
      bot_id: "your-bot-id"
      secret: "your-secret"
      group_policy: "allowlist"
      group_allow_from:
        - "group_id_1"
        - "group_id_2"
      groups:
        group_id_1:
          allow_from:
            - "user_alice"
            - "user_bob"
        group_id_2:
          allow_from:
            - "user_charlie"
        "*":
          allow_from:
            - "user_admin"
```

**Como funciona:**

1. `group_policy` e `group_allow_from` controlam se um grupo é permitido de todo.
2. Se um grupo passa na verificação de nível superior, a lista `groups.<group_id>.allow_from` (se presente) restringe ainda mais quais remetentes dentro desse grupo podem interagir com o bot.
3. Uma entrada de grupo `"*"` serve como padrão para grupos não listados explicitamente.
4. Entradas de allowlist suportam o wildcard `*` para permitir todos os usuários, e entradas são case-insensitive.
5. Entradas podem opcionalmente usar o formato de prefixo `wecom:user:` ou `wecom:group:` — o prefixo é removido automaticamente.

Se nenhum `allow_from` estiver configurado para um grupo, todos os usuários desse grupo são permitidos (assumindo que o grupo em si passa na verificação de política de nível superior).

## Suporte a mídia {#media-support}

### Entrada (recebimento)

O adaptador recebe anexos de mídia de usuários e os armazena em cache localmente para processamento pelo agente:

| Type | How it's handled |
|------|-----------------|
| **Images** | Baixadas e armazenadas em cache localmente. Suporta imagens baseadas em URL e codificadas em base64. |
| **Files** | Baixados e armazenados em cache. O nome do arquivo é preservado da mensagem original. |
| **Voice** | A transcrição de texto da mensagem de voz é extraída se disponível. |
| **Mixed messages** | Mensagens de tipo misto WeCom (texto + imagens) são analisadas e todos os componentes extraídos. |

**Mensagens citadas:** Mídia de mensagens citadas (respondidas) também é extraída, para que o agente tenha contexto sobre o que o usuário está respondendo.

### Descriptografia de mídia criptografada AES

O WeCom criptografa alguns anexos de mídia de entrada com AES-256-CBC. O adaptador trata isso automaticamente:

- Quando um item de mídia de entrada inclui um campo `aeskey`, o adaptador baixa os bytes criptografados e os descriptografa usando AES-256-CBC com padding PKCS#7.
- A chave AES é o valor decodificado em base64 do campo `aeskey` (deve ter exatamente 32 bytes).
- O IV é derivado dos primeiros 16 bytes da chave.
- Isso requer o pacote Python `cryptography` (`pip install cryptography`).

Nenhuma configuração é necessária — a descriptografia acontece transparentemente quando mídia criptografada é recebida.

### Saída (envio)

| Method | What it sends | Size limit |
|--------|--------------|------------|
| `send` | Mensagens de texto Markdown | 4000 chars |
| `send_image` / `send_image_file` | Mensagens de imagem nativas | 10 MB |
| `send_document` | Anexos de arquivo | 20 MB |
| `send_voice` | Mensagens de voz (somente formato AMR para voz nativa) | 2 MB |
| `send_video` | Mensagens de vídeo | 10 MB |

**Upload em chunks:** Arquivos são enviados em chunks de 512 KB por um protocolo de três etapas (init → chunks → finish). O adaptador trata isso automaticamente.

**Downgrade automático:** Quando a mídia excede o limite de tamanho do tipo nativo, mas está abaixo do limite absoluto de 20 MB de arquivo, é enviada automaticamente como anexo de arquivo genérico:

- Imagens > 10 MB → enviadas como arquivo
- Vídeos > 10 MB → enviados como arquivo
- Voz > 2 MB → enviada como arquivo
- Áudio não-AMR → enviado como arquivo (WeCom suporta apenas AMR para voz nativa)

Arquivos que excedem o limite absoluto de 20 MB são rejeitados com uma mensagem informativa enviada ao chat.

## Respostas em modo reply {#reply-mode-responses}

Quando o bot recebe uma mensagem via callback WeCom, o adaptador lembra o ID da requisição de entrada. Se uma resposta for enviada enquanto o contexto da requisição ainda estiver ativo, o adaptador usa o modo reply do WeCom (`aibot_respond_msg`) para correlacionar a resposta diretamente à mensagem de entrada. Isso proporciona uma experiência de conversa mais natural no cliente WeCom.

A resposta completa é entregue como uma única mensagem — o adaptador não transmite tokens incrementalmente. Se o contexto da requisição de entrada tiver expirado ou estiver indisponível, o adaptador recorre ao envio proativo de mensagens via `aibot_send_msg`.

O modo reply também funciona para mídia: mídia enviada pode ser enviada como resposta à mensagem de origem.

## Conexão e reconexão {#connection-and-reconnection}

O adaptador mantém uma conexão WebSocket persistente com o gateway WeCom em `wss://openws.work.weixin.qq.com`.

### Ciclo de vida da conexão

1. **Connect:** Abre uma conexão WebSocket e envia um frame de autenticação `aibot_subscribe` com bot_id e secret.
2. **Heartbeat:** Envia frames ping em nível de aplicativo a cada 30 segundos para manter a conexão viva.
3. **Listen:** Lê continuamente frames de entrada e despacha callbacks de mensagem.

### Comportamento de reconexão

Em perda de conexão, o adaptador usa backoff exponencial para reconectar:

| Attempt | Delay |
|---------|-------|
| 1st retry | 2 seconds |
| 2nd retry | 5 seconds |
| 3rd retry | 10 seconds |
| 4th retry | 30 seconds |
| 5th+ retry | 60 seconds |

Após cada reconexão bem-sucedida, o contador de backoff é zerado. Todos os futures de requisição pendentes falham na desconexão para que chamadores não fiquem pendurados indefinidamente.

### Deduplicação

Mensagens de entrada são deduplicadas usando IDs de mensagem com janela de 5 minutos e cache máximo de 1000 entradas. Isso evita processamento duplo de mensagens durante reconexão ou instabilidades de rede.

## Todas as variáveis de ambiente {#all-environment-variables}

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `WECOM_BOT_ID` | ✅ | — | WeCom AI Bot ID |
| `WECOM_SECRET` | ✅ | — | WeCom AI Bot Secret |
| `WECOM_ALLOWED_USERS` | — | _(empty)_ | IDs de usuário separados por vírgula para allowlist de nível gateway |
| `WECOM_HOME_CHANNEL` | — | — | ID de chat para saída de cron/notificação |
| `WECOM_WEBSOCKET_URL` | — | `wss://openws.work.weixin.qq.com` | URL do gateway WebSocket |
| `WECOM_DM_POLICY` | — | `open` | Política de acesso a DM |
| `WECOM_GROUP_POLICY` | — | `open` | Política de acesso a grupo |

## Solução de problemas {#troubleshooting}

| Problem | Fix |
|---------|-----|
| `WECOM_BOT_ID and WECOM_SECRET are required` | Defina ambas as env vars ou configure no assistente de setup |
| `WeCom startup failed: aiohttp not installed` | Instale aiohttp: `pip install aiohttp` |
| `WeCom startup failed: httpx not installed` | Instale httpx: `pip install httpx` |
| `invalid secret (errcode=40013)` | Verifique se o secret corresponde às credenciais do seu bot |
| `Timed out waiting for subscribe acknowledgement` | Verifique conectividade de rede com `openws.work.weixin.qq.com` |
| Bot doesn't respond in groups | Verifique a configuração `group_policy` e certifique-se de que o ID do grupo está em `group_allow_from` |
| Bot ignores certain users in a group | Verifique listas `allow_from` por grupo na seção de config `groups` |
| Media decryption fails | Instale `cryptography`: `pip install cryptography` |
| `cryptography is required for WeCom media decryption` | A mídia de entrada está criptografada com AES. Instale: `pip install cryptography` |
| Voice messages sent as files | WeCom suporta apenas formato AMR para voz nativa. Outros formatos são automaticamente enviados como arquivo. |
| `File too large` error | WeCom tem limite absoluto de 20 MB em todos os uploads de arquivo. Comprima ou divida o arquivo. |
| Images sent as files | Imagens > 10 MB excedem o limite de imagem nativa e são automaticamente enviadas como anexos de arquivo. |
| `Timeout sending message to WeCom` | O WebSocket pode ter desconectado. Verifique logs em busca de mensagens de reconexão. |
| `WeCom websocket closed during authentication` | Problema de rede ou credenciais incorretas. Verifique bot_id e secret. |
