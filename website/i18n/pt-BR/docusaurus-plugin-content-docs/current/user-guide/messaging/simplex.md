# SimpleX Chat {#simplex-chat}

[SimpleX Chat](https://simplex.chat/) é uma plataforma de mensagens privada e descentralizada em que os usuários possuem seus contatos e grupos. Diferente de outras plataformas, o SimpleX não atribui IDs de usuário persistentes — cada contato é identificado por um ID interno opaco gerado no momento da conexão, o que o torna um dos mensageiros mais privados disponíveis.

> Execute `hermes gateway setup` e escolha **SimpleX** para um passo a passo guiado.

## Pré-requisitos {#prerequisites}

- O CLI **simplex-chat** instalado e rodando como daemon
- Pacote Python **websockets** (`pip install websockets`)

## Instale simplex-chat {#install-simplex-chat}

Baixe a release mais recente na página de [releases do simplex-chat no GitHub](https://github.com/simplex-chat/simplex-chat/releases):

```bash
# Linux / macOS binary
curl -L https://github.com/simplex-chat/simplex-chat/releases/latest/download/simplex-chat-ubuntu-22_04-x86_64 -o simplex-chat
chmod +x simplex-chat
```

O projeto SimpleX Chat não publica uma imagem Docker pré-construída para o cliente de chat; para rodá-lo sob Docker, faça build a partir do código-fonte no [repositório simplex-chat](https://github.com/simplex-chat/simplex-chat).

## Inicie o daemon {#start-the-daemon}

```bash
simplex-chat -p 5225
```

O daemon escuta WebSocket em `ws://127.0.0.1:5225` por padrão.

## Configure o Hermes {#configure-hermes}

### Pelo assistente de setup {#via-setup-wizard}

```bash
hermes gateway setup
```

Selecione **SimpleX Chat** e siga as instruções.

### Por variáveis de ambiente {#via-environment-variables}

Adicione em `~/.hermes/.env`:

```
SIMPLEX_WS_URL=ws://127.0.0.1:5225
SIMPLEX_ALLOWED_USERS=<contact-id-1>,<contact-id-2>
SIMPLEX_HOME_CHANNEL=<contact-id>
```

| Variable | Required | Description |
|---|---|---|
| `SIMPLEX_WS_URL` | Yes | URL WebSocket do daemon simplex-chat |
| `SIMPLEX_ALLOWED_USERS` | Recommended | Allowlist separada por vírgula. Cada entrada pode ser um `contactId` numérico **ou** um nome de exibição — ambas as formas funcionam. |
| `SIMPLEX_ALLOW_ALL_USERS` | Optional | Defina `true` para permitir todo contato (use com cuidado) |
| `SIMPLEX_AUTO_ACCEPT` | Optional | Aceitar automaticamente pedidos de contato recebidos (padrão: `true`) |
| `SIMPLEX_GROUP_ALLOWED` | Optional | IDs de grupo separados por vírgula em que o bot participa, ou `*` para qualquer grupo. Omita para ignorar mensagens de grupo por completo |
| `SIMPLEX_HOME_CHANNEL` | Optional | ID de contato/grupo padrão para entrega de jobs cron |
| `SIMPLEX_HOME_CHANNEL_NAME` | Optional | Rótulo legível para o canal home |
| `HERMES_SIMPLEX_TEXT_BATCH_DELAY` | Optional | Segundos de período quieto (padrão: `0.8`) usados para concatenar mensagens de texto recebidas em rajada em um evento |

## Encontre seu contact ID ou nome de exibição {#find-your-contact-id-or-display-name}

Depois de iniciar o daemon, abra uma conversa com seu contato agente. O `contactId` numérico aparece nos logs de sessão. Se preferir usar o nome de exibição mostrado na UI do SimpleX, isso também funciona — `SIMPLEX_ALLOWED_USERS` aceita qualquer uma das formas.

## Autorização {#authorization}

Por padrão **todos os contatos são negados**. Você deve:

1. Definir `SIMPLEX_ALLOWED_USERS` como uma lista separada por vírgula de `contactId`s e/ou nomes de exibição (ex.: `SIMPLEX_ALLOWED_USERS=4,alice` corresponde ao contactId 4 ou ao contato cujo nome de exibição é "alice"), ou
2. Usar **pareamento por DM** — envie qualquer mensagem ao bot e ele responderá com um código de pareamento. Insira esse código via `hermes pairing approve simplex <CODE>`.

## Chats em grupo {#group-chats}

Por padrão o adaptador ignora mensagens de grupo — um bot em um grupo processaria o tráfego de todo membro. Opte explicitamente:

```
SIMPLEX_GROUP_ALLOWED=12,34          # specific group IDs
# or
SIMPLEX_GROUP_ALLOWED=*              # any group the bot is in
```

Enderece grupos prefixando o chat ID com `group:`, ex.:
`simplex:group:12` como alvo `deliver=` de cron ou em uma chamada `hermes send`.

## Anexos {#attachments}

O adaptador suporta anexos nativos do SimpleX em ambas as direções:

- **Entrada** — imagens, notas de voz e arquivos recebidos são aceitos via
  fluxo XFTP do daemon (`rcvFileDescrReady` → `/freceive` → aguardar
  `rcvFileComplete`) e expostos como `MessageEvent.media_urls` com o
  `MessageType` apropriado (`PHOTO`, `VOICE`, `TEXT` + documento).
- **Saída** — `send_image_file`, `send_voice`, `send_document` e
  `send_video` usam a forma estruturada `/_send` com `filePath`, então
  o cliente SimpleX receptor renderiza imagens inline e reproduz notas de voz
  inline em vez de oferecê-las como downloads.

Respostas do agente também podem embutir tags `MEDIA:/path/to/file` em texto puro —
o adaptador remove a tag do corpo e envia o arquivo como nota de voz
(extensões de áudio) ou documento.

## Usando SimpleX com jobs cron {#using-simplex-with-cron-jobs}

```python
cronjob(
    action="create",
    schedule="every 1h",
    deliver="simplex",          # uses SIMPLEX_HOME_CHANNEL
    prompt="Check for alerts and summarise."
)
```

Ou direcione um contato específico via o campo `deliver:` do job cron, ou de um script shell com a [CLI `hermes send`](/guides/pipe-script-output):

```bash
hermes send simplex:<contact-id> "Done!"
```

## Notas de privacidade {#privacy-notes}

- O SimpleX nunca revela números de telefone ou endereços de email — contatos usam IDs opacos
- A conexão entre Hermes e o daemon é WebSocket local (`ws://127.0.0.1:5225`) — nenhum dado sai da sua máquina
- Mensagens são criptografadas ponta a ponta pelo protocolo SimpleX antes de chegar ao daemon

## Solução de problemas {#troubleshooting}

**"Cannot reach daemon"** — Certifique-se de que `simplex-chat -p 5225` está rodando e a porta corresponde a `SIMPLEX_WS_URL`.

**"websockets not installed"** — Execute `pip install websockets`.

**Mensagens não recebidas** — Verifique se o ID do contato está em `SIMPLEX_ALLOWED_USERS` ou aprove via pareamento por DM.
