---
sidebar_position: 16
title: "Yuanbao"
description: "Conecte o Hermes Agent à plataforma de mensagens empresariais Yuanbao via gateway WebSocket"
---

# Yuanbao

Conecte o Hermes ao [Yuanbao](https://yuanbao.tencent.com/), a plataforma de mensagens empresariais da Tencent. O adaptador usa um gateway WebSocket para entrega de mensagens em tempo real e suporta conversas diretas (C2C) e de grupo.

:::info
Yuanbao é uma plataforma de mensagens empresariais usada principalmente dentro da Tencent e ambientes empresariais. Usa WebSocket para comunicação em tempo real, autenticação baseada em HMAC e suporta mídia rica incluindo imagens, arquivos e mensagens de voz.
:::

## Pré-requisitos {#prerequisites}

- Uma conta Yuanbao com permissões de criação de bot
- Yuanbao APP_ID e APP_SECRET (do admin da plataforma)
- Pacotes Python: `websockets` e `httpx`
- Para suporte a mídia: `aiofiles`

Instale as dependências necessárias:

```bash
pip install websockets httpx aiofiles
```

## Configuração {#setup}

### 1. Crie um bot no Yuanbao

1. Baixe o app Yuanbao em [https://yuanbao.tencent.com/](https://yuanbao.tencent.com/)
2. No app, vá em **PAI → My Bot** e crie um novo bot
3. Depois que o bot for criado, copie o **APP_ID** e **APP_SECRET**

### 2. Execute o assistente de setup

A forma mais fácil de configurar o Yuanbao é pelo setup interativo:

```bash
hermes gateway setup
```

Selecione **Yuanbao** quando solicitado. O assistente irá:

1. Pedir seu APP_ID
2. Pedir seu APP_SECRET
3. Salvar a configuração automaticamente

:::tip
A URL WebSocket e o API Domain têm padrões sensatos embutidos. Você só precisa fornecer APP_ID e APP_SECRET para começar.
:::

### 3. Configure variáveis de ambiente

Após o setup inicial, verifique estas variáveis em `~/.hermes/.env`:

```bash
# Required
YUANBAO_APP_ID=your-app-id
YUANBAO_APP_SECRET=your-app-secret
YUANBAO_WS_URL=wss://api.yuanbao.example.com/ws
YUANBAO_API_DOMAIN=https://api.yuanbao.example.com

# Optional: bot account ID (normally obtained automatically from sign-token)
# YUANBAO_BOT_ID=your-bot-id

# Optional: internal routing environment (e.g. test/staging/production)
# YUANBAO_ROUTE_ENV=production

# Optional: home channel for cron/notifications (format: direct:<account> or group:<group_code>)
YUANBAO_HOME_CHANNEL=direct:bot_account_id
YUANBAO_HOME_CHANNEL_NAME="Bot Notifications"

# Optional: restrict access (legacy, see Access Control below for fine-grained policies)
YUANBAO_ALLOWED_USERS=user_account_1,user_account_2
```

### 4. Inicie o gateway

```bash
hermes gateway
```

O adaptador conectará ao gateway WebSocket Yuanbao, autenticará usando assinaturas HMAC e começará a processar mensagens.

## Recursos {#features}

- **Gateway WebSocket** — comunicação bidirecional em tempo real
- **Autenticação HMAC** — assinatura segura de requisições com APP_ID/APP_SECRET
- **Mensagens C2C** — conversas diretas usuário-bot
- **Mensagens de grupo** — conversas em chats de grupo
- **Suporte a mídia** — imagens, arquivos e mensagens de voz via COS (Cloud Object Storage)
- **Formatação Markdown** — mensagens são automaticamente fragmentadas para os limites de tamanho do Yuanbao
- **Deduplicação de mensagens** — evita processamento duplicado da mesma mensagem
- **Heartbeat/keep-alive** — mantém estabilidade da conexão WebSocket
- **Indicadores de digitação** — exibe status "typing…" enquanto o agente processa
- **Reconexão automática** — trata desconexões WebSocket com backoff exponencial
- **Consultas de informação de grupo** — recupera detalhes de grupo e listas de membros
- **Suporte a Sticker/Emoji** — envia stickers TIMFaceElem e emoji em conversas
- **Auto-sethome** — o primeiro usuário a enviar mensagem ao bot é automaticamente definido como dono do canal home
- **Notificação de resposta lenta** — envia mensagem de espera quando o agente demora mais que o esperado

## Opções de configuração {#configuration-options}

### Formatos de Chat ID

O Yuanbao usa identificadores prefixados dependendo do tipo de conversa:

| Chat Type | Format | Example |
|-----------|--------|---------|
| Direct message (C2C) | `direct:<account>` | `direct:user123` |
| Group message | `group:<group_code>` | `group:grp456` |

### Uploads de mídia

O adaptador Yuanbao trata automaticamente uploads de mídia via COS (Tencent Cloud Object Storage):

- **Images**: Suporta JPEG, PNG, GIF, WebP
- **Files**: Suporta todos os tipos comuns de documento
- **Voice**: Suporta WAV, MP3, OGG

URLs de mídia são automaticamente validadas e baixadas antes do upload para prevenir ataques SSRF.

## Canal home {#home-channel}

Use o comando `/sethome` em qualquer chat Yuanbao (DM ou grupo) para designá-lo como **canal home**. Tarefas agendadas (cron jobs) entregam seus resultados a esse canal.

:::tip Auto-sethome
Se nenhum canal home estiver configurado, o primeiro usuário a enviar mensagem ao bot será automaticamente definido como dono do canal home. Se o canal home atual for um chat de grupo, a primeira DM o atualizará para um canal direto.
:::

Você também pode defini-lo manualmente em `~/.hermes/.env`:

```bash
YUANBAO_HOME_CHANNEL=direct:user_account_id
# or for a group:
# YUANBAO_HOME_CHANNEL=group:group_code
YUANBAO_HOME_CHANNEL_NAME="My Bot Updates"
```

### Exemplo: Definir canal home

1. Inicie uma conversa com o bot no Yuanbao
2. Envie o comando: `/sethome`
3. O bot responde: "Home channel set to [chat_name] with ID [chat_id]. Cron jobs will deliver to this location."
4. Cron jobs e notificações futuras serão enviados a esse canal

### Exemplo: Entrega de cron job

Crie um cron job:

```bash
/cron "0 9 * * *" Check server status
```

A saída agendada será entregue ao seu canal home Yuanbao todos os dias às 9h.

## Dicas de uso {#usage-tips}

### Iniciando uma conversa

Envie qualquer mensagem ao bot no Yuanbao:

```
hello
```

O bot responde no mesmo thread de conversa.

### Comandos disponíveis

Todos os comandos padrão do Hermes funcionam no Yuanbao:

| Command | Description |
|---------|-------------|
| `/new` | Iniciar uma conversa nova |
| `/model [provider:model]` | Mostrar ou alterar o model |
| `/sethome` | Definir este chat como canal home |
| `/status` | Mostrar info da sessão |
| `/help` | Mostrar comandos disponíveis |

### Enviando arquivos

Para enviar um arquivo ao bot, simplesmente anexe-o diretamente no chat Yuanbao. O bot baixará e processará automaticamente o anexo.

Você também pode incluir uma mensagem com o anexo:

```
Please analyze this document
```

### Recebendo arquivos

Quando você pede ao bot para criar ou exportar um arquivo, ele envia o arquivo diretamente ao seu chat Yuanbao.

## Solução de problemas {#troubleshooting}

### Bot está online mas não responde a mensagens

**Causa**: Autenticação falhou durante o handshake WebSocket.

**Correção**:
1. Verifique se APP_ID e APP_SECRET estão corretos
2. Verifique se a URL WebSocket está acessível
3. Certifique-se de que a conta do bot tem permissões adequadas
4. Revise os logs do gateway: `tail -f ~/.hermes/logs/gateway.log`

### Erro "Connection refused"

**Causa**: URL WebSocket inacessível ou incorreta.

**Correção**:
1. Verifique o formato da URL WebSocket (deve começar com `wss://`)
2. Verifique conectividade de rede ao domínio da API Yuanbao
3. Confirme que o firewall permite conexões WebSocket
4. Teste a URL com: `curl -I https://[YUANBAO_API_DOMAIN]`

### Uploads de mídia falham

**Causa**: Credenciais COS inválidas ou servidor de mídia inacessível.

**Correção**:
1. Verifique se API_DOMAIN está correto
2. Verifique se permissões de upload de mídia estão habilitadas para seu bot
3. Certifique-se de que o arquivo de mídia está acessível e não está corrompido
4. Verifique configuração do bucket COS com o admin da plataforma

### Mensagens não entregues ao canal home

**Causa**: Formato do ID do canal home incorreto ou cron job ainda não disparou.

**Correção**:
1. Verifique se YUANBAO_HOME_CHANNEL está no formato correto
2. Teste com o comando `/sethome` para auto-detectar o formato correto
3. Verifique o cronograma do cron job com `/status`
4. Verifique se o bot tem permissões de envio no chat alvo

### Desconexões frequentes

**Causa**: Conexão WebSocket instável ou rede não confiável.

**Correção**:
1. Verifique logs do gateway em busca de padrões de erro
2. Aumente timeout de heartbeat nas configurações de conexão
3. Certifique-se de conexão de rede estável com a API Yuanbao
4. Considere habilitar logging verbose: `HERMES_LOG_LEVEL=debug`

## Controle de acesso {#access-control}

O Yuanbao suporta controle de acesso detalhado para conversas DM e de grupo:

```bash
# DM policy: open (default) | allowlist | disabled
YUANBAO_DM_POLICY=open
# Comma-separated user IDs allowed to DM the bot (only used when DM_POLICY=allowlist)
YUANBAO_DM_ALLOW_FROM=user_id_1,user_id_2

# Group policy: open (default) | allowlist | disabled
YUANBAO_GROUP_POLICY=open
# Comma-separated group codes allowed (only used when GROUP_POLICY=allowlist)
YUANBAO_GROUP_ALLOW_FROM=group_code_1,group_code_2
```

Estes também podem ser definidos em `config.yaml`:

```yaml
platforms:
  yuanbao:
    extra:
      dm_policy: allowlist
      dm_allow_from: "user1,user2"
      group_policy: open
      group_allow_from: ""
```

## Configuração avançada {#advanced-configuration}

### Chunking de mensagens

O Yuanbao tem tamanho máximo de mensagem. O Hermes fragmenta automaticamente respostas grandes com split consciente de Markdown (respeita code fences, tabelas e limites de parágrafo).

### Parâmetros de conexão

Os seguintes parâmetros de conexão estão embutidos no adaptador com padrões sensatos:

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| WebSocket connect timeout | 15 seconds | Tempo de espera para handshake WS |
| Heartbeat interval | 30 seconds | Frequência de ping para manter conexão viva |
| Max reconnect attempts | 100 | Número máximo de tentativas de reconexão |
| Reconnect backoff | 1s → 60s (exponential) | Tempo de espera entre tentativas de reconexão |
| Reply heartbeat interval | 2 seconds | Frequência de envio de status RUNNING |
| Send timeout | 30 seconds | Timeout para mensagens WS de saída |

:::note
Esses valores atualmente não são configuráveis via variáveis de ambiente. Estão otimizados para deployments Yuanbao típicos.
:::

### Logging verbose

Habilite logging debug para solucionar problemas de conexão:

```bash
HERMES_LOG_LEVEL=debug hermes gateway
```

## Integração com outros recursos {#integration-with-other-features}

### Cron jobs

Agende tarefas que rodam no Yuanbao:

```
/cron "0 */4 * * *" Report system health
```

Resultados são entregues ao seu canal home.

### Tarefas em background

Execute operações longas sem bloquear a conversa:

```
/background Analyze all files in the archive
```

### Mensagens cross-platform

Envie uma mensagem da CLI para o Yuanbao:

```bash
hermes chat -q "Send 'Hello from CLI' to yuanbao:group:group_code"
```

## Documentação relacionada {#related-documentation}

- [Messaging Gateway Overview](./index.md)
- [Slash Commands Reference](/reference/slash-commands)
- [Cron Jobs](/user-guide/features/cron)
- [Background Sessions](/user-guide/cli#background-sessions)
