# SimpleX Chat {#simplex-chat}

[SimpleX Chat](https://simplex.chat/) é uma plataforma de mensagens privada e descentralizada em que os usuários são donos de seus contatos e grupos. Diferente de outras plataformas, o SimpleX não atribui IDs de usuário persistentes — cada contato é identificado por um ID interno opaco gerado no momento da conexão, o que o torna um dos mensageiros mais privados disponíveis.

> Execute `hermes gateway setup` e escolha **SimpleX** para um passo a passo guiado.

## Pré-requisitos {#prerequisites}

- A CLI **simplex-chat** instalada e rodando como daemon
- Pacote Python **websockets** (`pip install websockets`)

## Instalar simplex-chat {#install-simplex-chat}

Baixe a versão mais recente na página de [releases do simplex-chat no GitHub](https://github.com/simplex-chat/simplex-chat/releases):

```bash
# Linux / macOS binary
curl -L https://github.com/simplex-chat/simplex-chat/releases/latest/download/simplex-chat-ubuntu-22_04-x86_64 -o simplex-chat
chmod +x simplex-chat
```

O projeto SimpleX Chat não publica uma imagem Docker pré-construída para o cliente de chat; para rodá-lo com Docker, compile a partir do código-fonte no [repositório simplex-chat](https://github.com/simplex-chat/simplex-chat).

## Iniciar o daemon {#start-the-daemon}

```bash
simplex-chat -p 5225
```

O daemon escuta em WebSocket em `ws://127.0.0.1:5225` por padrão.

## Configurar o Hermes {#configure-hermes}

### Via assistente de setup {#via-setup-wizard}

```bash
hermes gateway setup
```

Selecione **SimpleX Chat** e siga as instruções.

### Via variáveis de ambiente {#via-environment-variables}

Adicione ao `~/.hermes/.env`:

```
SIMPLEX_WS_URL=ws://127.0.0.1:5225
SIMPLEX_ALLOWED_USERS=<contact-id-1>,<contact-id-2>
SIMPLEX_HOME_CHANNEL=<contact-id>
```

| Variável | Obrigatório | Descrição |
|---|---|---|
| `SIMPLEX_WS_URL` | Sim | URL WebSocket do daemon simplex-chat |
| `SIMPLEX_ALLOWED_USERS` | Recomendado | Allowlist separada por vírgulas. Cada entrada pode ser `contactId` numérico **ou** nome de exibição — ambas as formas funcionam. |
| `SIMPLEX_ALLOW_ALL_USERS` | Opcional | Defina `true` para permitir todo contato (use com cuidado) |
| `SIMPLEX_AUTO_ACCEPT` | Opcional | Aceitar automaticamente pedidos de contato incoming (padrão: `true`) |
| `SIMPLEX_GROUP_ALLOWED` | Opcional | IDs de grupo separados por vírgulas em que o bot participa, ou `*` para qualquer grupo. Omita para ignorar mensagens de grupo |
| `SIMPLEX_HOME_CHANNEL` | Opcional | ID padrão de contato/grupo para entrega de cron job |
| `SIMPLEX_HOME_CHANNEL_NAME` | Opcional | Rótulo legível para o home channel |
| `HERMES_SIMPLEX_TEXT_BATCH_DELAY` | Opcional | Segundos de quiet-period (padrão: `0.8`) usados para concatenar mensagens de texto inbound rápidas em um evento |

## Encontrar seu contact ID ou nome de exibição {#find-your-contact-id-or-display-name}

Depois de iniciar o daemon, abra uma conversa com o contato do seu agente. O `contactId` numérico aparece nos logs de sessão. Se preferir usar o nome de exibição mostrado na UI do SimpleX, isso também funciona — `SIMPLEX_ALLOWED_USERS` aceita qualquer um dos dois formatos.

## Autorização {#authorization}

Por padrão **todos os contatos são negados**. Você deve:

1. Definir `SIMPLEX_ALLOWED_USERS` como uma lista separada por vírgulas de `contactId`s e/ou nomes de exibição (ex.: `SIMPLEX_ALLOWED_USERS=4,alice` corresponde ao contactId 4 ou ao contato cujo nome de exibição é "alice"), ou
2. Usar **pareamento por DM** — envie qualquer mensagem ao bot e ele responderá com um código de pareamento. Aprove com `hermes pairing approve simplex <CODE>`.

## Chats em grupo {#group-chats}

Por padrão o adaptador ignora mensagens de grupo — um bot em um grupo processaria o tráfego de todos os membros. Opte explicitamente:

```
SIMPLEX_GROUP_ALLOWED=12,34          # specific group IDs
# or
SIMPLEX_GROUP_ALLOWED=*              # any group the bot is in
```

Enderece grupos prefixando o chat ID com `group:`, por exemplo
`simplex:group:12` como destino `deliver=` de cron ou em uma chamada `hermes send`.

## Enviando com `hermes send` {#sending-with-hermes-send}

O SimpleX funciona como destino de envio standalone — o daemon deve estar rodando,
mas um gateway ativo não é necessário para texto simples:

```bash
hermes send --to simplex:alice "hello"          # DM by contact display name
hermes send --to simplex:group:12 "hello"       # group by numeric ID
hermes send --to simplex "hello"                # SIMPLEX_HOME_CHANNEL
```

Enquanto o gateway estiver rodando, o adaptador enumera seus contatos e
grupos permitidos no diretório de canais (atualizado a cada 5 minutos), então
`hermes send --list` os mostra por nome. Antes da primeira execução do gateway, a
plataforma ainda aparece em `--list` com a dica "no channels discovered yet"
— os destinos diretos como os acima funcionam de qualquer forma.

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

Respostas do agente também podem embutir tags `MEDIA:/path/to/file` em texto simples —
o adaptador remove a tag do corpo e envia o arquivo como nota de voz (extensões de áudio) ou documento.

## Usando SimpleX com cron jobs {#using-simplex-with-cron-jobs}

```python
cronjob(
    action="create",
    schedule="every 1h",
    deliver="simplex",          # uses SIMPLEX_HOME_CHANNEL
    prompt="Check for alerts and summarise."
)
```

Ou direcione um contato específico via campo `deliver:` do cron job, ou a partir de um script shell com a [CLI `hermes send`](/guides/pipe-script-output):

```bash
hermes send simplex:<contact-id> "Done!"
```

## Notas de privacidade {#privacy-notes}

- O SimpleX nunca revela números de telefone ou endereços de e-mail — contatos usam IDs opacos
- A conexão entre o Hermes e o daemon é WebSocket local (`ws://127.0.0.1:5225`) — nenhum dado sai da sua máquina
- Mensagens são criptografadas ponta a ponta pelo protocolo SimpleX antes de chegar ao daemon

## Solução de problemas {#troubleshooting}

**"Cannot reach daemon"** — Certifique-se de que `simplex-chat -p 5225` está rodando e a porta corresponde a `SIMPLEX_WS_URL`.

**"websockets not installed"** — Execute `pip install websockets`.

**Mensagens não recebidas** — Verifique se o ID do contato está em `SIMPLEX_ALLOWED_USERS` ou aprove via pareamento por DM.
