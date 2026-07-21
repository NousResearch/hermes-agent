---
sidebar_position: 6
title: "Signal"
description: "Configure o Hermes Agent como bot do Signal via daemon signal-cli"
---

# Configuração do Signal {#signal-setup}

O Hermes se conecta ao Signal pelo daemon [signal-cli](https://github.com/AsamK/signal-cli) rodando em modo HTTP. O adaptador transmite mensagens em tempo real via SSE (Server-Sent Events) e envia respostas via JSON-RPC.

O Signal é o mensageiro mainstream mais focado em privacidade — criptografia ponta a ponta por padrão, protocolo open-source, coleta mínima de metadados. Isso o torna ideal para fluxos de agente sensíveis à segurança.

:::info Sem novas dependências Python
O adaptador Signal usa `httpx` (já uma dependência core do Hermes) para toda a comunicação. Nenhum pacote Python adicional é necessário. Você só precisa do signal-cli instalado externamente.
:::

---

## Pré-requisitos {#prerequisites}

- **signal-cli** — cliente Signal baseado em Java ([GitHub](https://github.com/AsamK/signal-cli))
- **Runtime Java 17+** — exigido pelo signal-cli
- **Um número de telefone** com Signal instalado (para vincular como dispositivo secundário)

### Instalando signal-cli {#installing-signal-cli}

```bash
# macOS
brew install signal-cli

# Linux (download latest release)
VERSION=$(curl -Ls -o /dev/null -w %{url_effective} \
  https://github.com/AsamK/signal-cli/releases/latest | sed 's/^.*\/v//')
curl -L -O "https://github.com/AsamK/signal-cli/releases/download/v${VERSION}/signal-cli-${VERSION}.tar.gz"
sudo tar xf "signal-cli-${VERSION}.tar.gz" -C /opt
sudo ln -sf "/opt/signal-cli-${VERSION}/bin/signal-cli" /usr/local/bin/
```

:::caution
signal-cli **não** está nos repositórios apt ou snap. A instalação Linux acima baixa diretamente das [releases do GitHub](https://github.com/AsamK/signal-cli/releases).
:::

---

## Passo 1: Vincule sua conta Signal {#step-1-link-your-signal-account}

O signal-cli funciona como um **dispositivo vinculado** — como WhatsApp Web, mas para Signal. Seu telefone permanece o dispositivo primário.

```bash
# Generate a linking URI (displays a QR code or link)
signal-cli link -n "HermesAgent"
```

1. Abra o **Signal** no seu telefone
2. Vá em **Settings → Linked Devices**
3. Toque em **Link New Device**
4. Escaneie o QR code ou insira a URI

---

## Passo 2: Inicie o daemon signal-cli {#step-2-start-the-signal-cli-daemon}

```bash
# Replace +1234567890 with your Signal phone number (E.164 format)
signal-cli --account +1234567890 daemon --http 127.0.0.1:8080
```

:::tip
Mantenha isso rodando em segundo plano. Você pode usar `systemd`, `tmux`, `screen` ou executá-lo como serviço.
:::

Verifique se está rodando:

```bash
curl http://127.0.0.1:8080/api/v1/check
# Should return: {"versions":{"signal-cli":...}}
```

---

## Passo 3: Configure o Hermes {#step-3-configure-hermes}

A forma mais fácil:

```bash
hermes gateway setup
```

Selecione **Signal** no menu de plataformas. O assistente irá:

1. Verificar se signal-cli está instalado
2. Solicitar a URL HTTP (padrão: `http://127.0.0.1:8080`)
3. Testar conectividade com o daemon
4. Pedir seu número de telefone da conta
5. Configurar usuários permitidos e políticas de acesso

### Configuração manual {#manual-configuration}

Adicione em `~/.hermes/.env`:

```bash
# Required
SIGNAL_HTTP_URL=http://127.0.0.1:8080
SIGNAL_ACCOUNT=+1234567890

# Security (recommended)
SIGNAL_ALLOWED_USERS=+1234567890,+0987654321    # Comma-separated E.164 numbers or UUIDs

# Optional
SIGNAL_GROUP_ALLOWED_USERS=groupId1,groupId2     # Enable groups (omit to disable, * for all)
SIGNAL_HOME_CHANNEL=+1234567890                  # Default delivery target for cron jobs
```

Depois inicie o gateway:

```bash
hermes gateway              # Foreground
hermes gateway install      # Install as a user service
sudo hermes gateway install --system   # Linux only: boot-time system service
```

---

## Controle de acesso {#access-control}

### Acesso por DM {#dm-access}

O acesso por DM segue o mesmo padrão de todas as outras plataformas Hermes:

1. **`SIGNAL_ALLOWED_USERS` definido** → apenas esses usuários podem enviar mensagens
2. **Sem allowlist** → usuários desconhecidos recebem um código de pareamento por DM (aprove via `hermes pairing approve signal CODE`)
3. **`SIGNAL_ALLOW_ALL_USERS=true`** → qualquer pessoa pode enviar mensagens (use com cautela)

### Acesso em grupos {#group-access}

O acesso em grupos é controlado pela variável de ambiente `SIGNAL_GROUP_ALLOWED_USERS`:

| Configuração | Comportamento |
|---------------|----------|
| Não definido (padrão) | Todas as mensagens de grupo são ignoradas. O bot responde apenas a DMs. |
| Definido com IDs de grupo | Apenas grupos listados são monitorados (ex.: `groupId1,groupId2`). |
| Definido como `*` | O bot responde em qualquer grupo do qual seja membro. |

---

## Recursos {#features}

### Anexos {#attachments}

O adaptador suporta envio e recebimento de mídia em ambas as direções.

**Entrada** (usuário → agente):

- **Imagens** — PNG, JPEG, GIF, WebP (detectadas automaticamente via magic bytes)
- **Áudio** — MP3, OGG, WAV, M4A (mensagens de voz transcritas se Whisper estiver configurado)
- **Documentos** — PDF, ZIP e outros tipos de arquivo

**Saída** (agente → usuário):

O agente pode enviar arquivos de mídia via tags `MEDIA:` nas respostas. Os seguintes métodos de entrega são suportados:

- **Imagens** — `send_multiple_images` e `send_image_file` enviam PNG, JPEG, GIF, WebP como anexos nativos do Signal
- **Voz** — `send_voice` envia arquivos de áudio (OGG, MP3, WAV, M4A, AAC) como anexos
- **Vídeo** — `send_video` envia arquivos MP4
- **Documentos** — `send_document` envia qualquer tipo de arquivo (PDF, ZIP, etc.)

Toda mídia de saída passa pela API padrão de anexos do Signal. Diferente de algumas plataformas, o Signal não distingue entre mensagens de voz e anexos de arquivo no nível do protocolo.

Limite de tamanho de anexo: **100 MB** (ambas as direções).
:::warning
**Os servidores do Signal limitam uploads de anexos por taxa**, o adaptador usa um scheduler para envio de múltiplas imagens que agrupa imagens em lotes de 32 e limita uploads conforme a política do servidor Signal.
:::

### Formatação nativa, citações de resposta e reações {#native-formatting-reply-quotes-and-reactions}

As mensagens Signal são renderizadas com **formatação nativa** em vez de caracteres markdown literais. O adaptador converte markdown (`**bold**`, `*italic*`, `` `code` ``, `~~strike~~`, `||spoiler||`, headings) em `bodyRanges` do Signal para que o texto apareça com estilo real no cliente do destinatário, em vez de `**` / `` ` `` visíveis.

**Citações de resposta.** Quando o Hermes responde a uma mensagem específica, agora publica uma resposta nativa que cita a original — o mesmo affordance de UI que usuários Signal veem ao usar "Reply". Isso é automático para respostas geradas em resposta a uma mensagem recebida.

**Reações.** O agente pode reagir a mensagens via a API padrão de reações; reações aparecem no Signal como emoji na mensagem referenciada, em vez de texto extra.

Nada disso exige configuração adicional — vem habilitado por padrão em builds recentes do signal-cli. Se sua versão do `signal-cli` for antiga demais, o Hermes faz fallback para entrega em texto simples e registra um aviso único.

### Indicadores de digitação {#typing-indicators}

O bot envia indicadores de digitação enquanto processa mensagens, atualizando a cada 8 segundos.

### Exibição de progresso de ferramentas {#tool-progress-display}

O Signal não suporta editar mensagens já enviadas. O Hermes, portanto, suprime bolhas de progresso de ferramentas do gateway no Signal, mesmo quando `/verbose` está habilitado e salva um modo diferente de `off` para a plataforma.

Você ainda pode ver atividade de ferramentas no CLI, e respostas finais no Signal podem incluir saída normal do assistente. Se precisar de progresso ao vivo por ferramenta no chat, use uma plataforma de mensagens com suporte a edição de mensagens.

### Redação de números de telefone {#phone-number-redaction}

Todos os números de telefone são automaticamente redigidos nos logs:
- `+15551234567` → `+155****4567`
- Isso se aplica tanto aos logs do gateway Hermes quanto ao sistema global de redação

### Note to Self (configuração com um único número) {#note-to-self-single-number-setup}

Se você executar signal-cli como **dispositivo secundário vinculado** no seu próprio número (em vez de um número de bot separado), pode interagir com o Hermes pelo recurso "Note to Self" do Signal.

Basta enviar uma mensagem para si mesmo no telefone — o signal-cli captura e o Hermes responde na mesma conversa.

**Como funciona:**
- Mensagens "Note to Self" chegam como envelopes `syncMessage.sentMessage`
- O adaptador detecta quando são endereçadas à própria conta do bot e as processa como mensagens recebidas regulares
- Proteção contra eco (rastreamento de sent-timestamp) evita loops infinitos — as próprias respostas do bot são filtradas automaticamente

**Nenhuma configuração extra necessária.** Funciona automaticamente enquanto `SIGNAL_ACCOUNT` corresponder ao seu número de telefone.

### Monitoramento de saúde {#health-monitoring}

O adaptador monitora a conexão SSE e reconecta automaticamente se:
- A conexão cair (com backoff exponencial: 2s → 60s)
- Nenhuma atividade for detectada por 120 segundos (faz ping no signal-cli para verificar)

---

## Solução de problemas {#troubleshooting}

| Problema | Solução |
|---------|----------|
| **"Cannot reach signal-cli"** durante a configuração | Garanta que o daemon signal-cli está rodando: `signal-cli --account +YOUR_NUMBER daemon --http 127.0.0.1:8080` |
| **Mensagens não recebidas** | Verifique se `SIGNAL_ALLOWED_USERS` inclui o número do remetente em formato E.164 (com prefixo `+`) |
| **"signal-cli not found on PATH"** | Instale signal-cli e garanta que está no PATH, ou use Docker |
| **Conexão caindo repetidamente** | Verifique logs do signal-cli. Garanta que Java 17+ está instalado. |
| **Mensagens de grupo ignoradas** | Configure `SIGNAL_GROUP_ALLOWED_USERS` com IDs de grupo específicos, ou `*` para permitir todos os grupos. |
| **Bot não responde a ninguém** | Configure `SIGNAL_ALLOWED_USERS`, use pareamento por DM, ou permita explicitamente todos os usuários via política do gateway se quiser acesso mais amplo. |
| **Mensagens duplicadas** | Garanta que apenas uma instância signal-cli está escutando no seu número de telefone |

---

## Segurança {#security}

:::warning
**Sempre configure controles de acesso.** O bot tem acesso a terminal por padrão. Sem `SIGNAL_ALLOWED_USERS` ou pareamento por DM, o gateway nega todas as mensagens recebidas como medida de segurança.
:::

- Números de telefone são redigidos em toda a saída de log
- Use pareamento por DM ou allowlists explícitas para onboarding seguro de novos usuários
- Mantenha grupos desabilitados a menos que precise especificamente de suporte a grupos, ou permita apenas grupos em que confia
- A criptografia ponta a ponta do Signal protege o conteúdo das mensagens em trânsito
- Os dados de sessão do signal-cli em `~/.local/share/signal-cli/` contêm credenciais da conta — proteja como uma senha

---

## Referência de variáveis de ambiente {#environment-variables-reference}

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SIGNAL_HTTP_URL` | Yes | — | signal-cli HTTP endpoint |
| `SIGNAL_ACCOUNT` | Yes | — | Bot phone number (E.164) |
| `SIGNAL_ALLOWED_USERS` | No | — | Comma-separated phone numbers/UUIDs |
| `SIGNAL_GROUP_ALLOWED_USERS` | No | — | Group IDs to monitor, or `*` for all (omit to disable groups) |
| `SIGNAL_ALLOW_ALL_USERS` | No | `false` | Allow any user to interact (skip allowlist) |
| `SIGNAL_HOME_CHANNEL` | No | — | Default delivery target for cron jobs |
