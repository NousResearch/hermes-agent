# BlueBubbles (iMessage)

Conecte o Hermes ao Apple iMessage via [BlueBubbles](https://bluebubbles.app/) — um servidor macOS gratuito e open source que faz a ponte do iMessage para qualquer dispositivo.

## Prerequisites {#prerequisites}

- Um **Mac** (sempre ligado) rodando [BlueBubbles Server](https://bluebubbles.app/)
- Apple ID conectado ao Messages.app nesse Mac
- BlueBubbles Server v1.0.0+ (webhooks exigem essa versão)
- Conectividade de rede entre o Hermes e o servidor BlueBubbles

## Setup {#setup}

### 1. Install BlueBubbles Server

Baixe e instale em [bluebubbles.app](https://bluebubbles.app/). Complete o assistente de configuração — faça login com seu Apple ID e configure um método de conexão (rede local, Ngrok, Cloudflare ou DNS dinâmico).

### 2. Get your Server URL and Password

No BlueBubbles Server → **Settings → API**, anote:
- **Server URL** (ex.: `http://192.168.1.10:1234`)
- **Server Password**

### 3. Configure Hermes

Execute o assistente de configuração:

```bash
hermes gateway setup
```

Selecione **BlueBubbles (iMessage)** e informe a URL do servidor e a senha.

Ou defina variáveis de ambiente diretamente em `~/.hermes/.env`:

```bash
BLUEBUBBLES_SERVER_URL=http://192.168.1.10:1234
BLUEBUBBLES_PASSWORD=your-server-password
```

#### Optional: Require mentions in group chats

Por padrão, o Hermes responde a toda DM ou mensagem de grupo BlueBubbles/iMessage autorizada. Para tornar chats em grupo opt-in, habilite a exigência de menção:

```yaml
platforms:
  bluebubbles:
    enabled: true
    extra:
      require_mention: true
```

Com `require_mention: true`, DMs continuam normais, mas mensagens em grupo são ignoradas a menos que correspondam a um padrão de menção. Se você não configurar padrões personalizados, o Hermes usa padrões conservadores para variantes de `Hermes` e `@Hermes agent`.

Para um nome de agente personalizado, defina padrões regex:

```yaml
platforms:
  bluebubbles:
    extra:
      require_mention: true
      mention_patterns:
        - '(?<![\w@])@?amos\b[,:\-]?'
```

### 4. Authorize Users

Escolha uma abordagem:

**DM Pairing (recomendado):**
Quando alguém envia mensagem para seu iMessage, o Hermes envia automaticamente um código de pareamento. Aprove com:
```bash
hermes pairing approve bluebubbles <CODE>
```
Use `hermes pairing list` para ver códigos pendentes e usuários aprovados.

**Pré-autorizar usuários específicos** (em `~/.hermes/.env`):
```bash
BLUEBUBBLES_ALLOWED_USERS=user@icloud.com,+15551234567
```

**Acesso aberto** (em `~/.hermes/.env`):
```bash
BLUEBUBBLES_ALLOW_ALL_USERS=true
```

### 5. Start the Gateway

```bash
hermes gateway run
```

O Hermes vai se conectar ao seu servidor BlueBubbles, registrar um webhook e começar a escutar mensagens iMessage.

## How It Works {#how-it-works}

```
iMessage → Messages.app → BlueBubbles Server → Webhook → Hermes
Hermes → BlueBubbles REST API → Messages.app → iMessage
```

- **Entrada:** o BlueBubbles envia eventos de webhook para um listener local quando novas mensagens chegam. Sem polling — entrega instantânea.
- **Saída:** o Hermes envia mensagens via BlueBubbles REST API.
- **Mídia:** imagens, mensagens de voz, vídeos e documentos são suportados nos dois sentidos. Anexos de entrada são baixados e armazenados em cache localmente para o agente processar.

## Environment Variables {#environment-variables}

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `BLUEBUBBLES_SERVER_URL` | Yes | — | URL do servidor BlueBubbles |
| `BLUEBUBBLES_PASSWORD` | Yes | — | Senha do servidor |
| `BLUEBUBBLES_WEBHOOK_HOST` | No | `127.0.0.1` | Endereço de bind do listener de webhook |
| `BLUEBUBBLES_WEBHOOK_PORT` | No | `8645` | Porta do listener de webhook |
| `BLUEBUBBLES_WEBHOOK_PATH` | No | `/bluebubbles-webhook` | Caminho da URL do webhook |
| `BLUEBUBBLES_HOME_CHANNEL` | No | — | Telefone/e-mail para entrega de cron |
| `BLUEBUBBLES_ALLOWED_USERS` | No | — | Usuários autorizados separados por vírgula |
| `BLUEBUBBLES_ALLOW_ALL_USERS` | No | `false` | Permitir todos os usuários |
| `BLUEBUBBLES_REQUIRE_MENTION` | No | `false` | Exigir padrão de menção antes de responder em chats em grupo |
| `BLUEBUBBLES_MENTION_PATTERNS` | No | Palavras de ativação Hermes | Array JSON, separado por newline ou padrões regex separados por vírgula para correspondência de menção em grupo |

Marcar mensagens como lidas automaticamente é controlado pela chave `send_read_receipts` em `platforms.bluebubbles.extra` em `~/.hermes/config.yaml` (padrão: `true`). Não há variável de ambiente correspondente.

## Features {#features}

### Text Messaging
Envie e receba iMessages. Markdown é removido automaticamente para entrega em texto simples limpo.

### Rich Media
- **Imagens:** fotos aparecem nativamente na conversa iMessage
- **Mensagens de voz:** arquivos de áudio enviados como mensagens de voz iMessage
- **Vídeos:** anexos de vídeo
- **Documentos:** arquivos enviados como anexos iMessage

### Tapback Reactions
Reações love, like, dislike, laugh, emphasize e question. Requer o [Private API helper](https://docs.bluebubbles.app/helper-bundle/installation) do BlueBubbles.

### Typing Indicators
Mostra "digitando..." na conversa iMessage enquanto o agente processa. Requer Private API.

### Read Receipts
Marca mensagens como lidas automaticamente após o processamento. Requer Private API.

### Chat Addressing
Você pode endereçar chats por e-mail ou número de telefone — o Hermes resolve automaticamente para GUIDs de chat BlueBubbles. Não é necessário usar o formato GUID bruto.

## Private API {#private-api}

Alguns recursos exigem o [Private API helper](https://docs.bluebubbles.app/helper-bundle/installation) do BlueBubbles:
- Reações tapback
- Indicadores de digitação
- Confirmações de leitura
- Criar novos chats por endereço

Sem a Private API, mensagens de texto básicas e mídia ainda funcionam.

## Troubleshooting {#troubleshooting}

### "Cannot reach server"
- Verifique se a URL do servidor está correta e se o Mac está ligado
- Confira se o BlueBubbles Server está rodando
- Garanta conectividade de rede (firewall, port forwarding)

### Messages not arriving
- Verifique se o webhook está registrado em BlueBubbles Server → Settings → API → Webhooks
- Confirme que a URL do webhook é acessível a partir do Mac
- Verifique `hermes logs gateway` em busca de erros de webhook (ou `hermes logs -f` para acompanhar em tempo real)

### "Private API helper not connected"
- Instale o Private API helper: [docs.bluebubbles.app](https://docs.bluebubbles.app/helper-bundle/installation)
- Mensagens básicas funcionam sem ele — apenas reações, digitação e confirmações de leitura exigem isso
