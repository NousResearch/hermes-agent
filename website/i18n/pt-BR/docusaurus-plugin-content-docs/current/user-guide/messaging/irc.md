# IRC {#irc}

O adaptador IRC conecta o Hermes a qualquer servidor IRC e retransmite mensagens entre um canal IRC (ou mensagens diretas) e o agente. Ele fala o protocolo IRC pelo `asyncio` da stdlib do Python — **sem dependências externas, sem SDK, sem daemon**. Funciona com redes públicas como [Libera.Chat](https://libera.chat/) e qualquer ircd self-hosted.

O IRC é texto puro: não há suporte a voz, imagem, arquivo, thread, reação, digitação ou streaming — as respostas são enviadas como linhas `PRIVMSG`, com mensagens longas divididas para caber no limite de linha do IRC.

> Execute `hermes gateway setup` e escolha **IRC** para um passo a passo guiado.

## Pré-requisitos {#prerequisites}

- Um servidor IRC para conectar (ex.: `irc.libera.chat`)
- Um canal para entrar (ex.: `#hermes`) — separe por vírgula para entrar em vários
- Um nickname para o bot (padrão: `hermes-bot`)
- Opcional: um nick registrado + senha NickServ se sua rede exigir identificação

## Configure o Hermes {#configure-hermes}

Você pode configurar o IRC de duas formas — variáveis de ambiente (para uma configuração rápida só com env) ou o bloco `gateway` em `~/.hermes/gateway-config.yaml`.

### Opção A — gateway-config.yaml {#option-a--gateway-configyaml}

```yaml
gateway:
  platforms:
    irc:
      enabled: true
      extra:
        server: irc.libera.chat
        port: 6697
        nickname: hermes-bot
        channel: "#hermes"
        use_tls: true
        server_password: ""       # optional server password
        nickserv_password: ""     # optional NickServ identification
        allowed_users: []         # empty = allow all, or list of nicks
        max_message_length: 450   # IRC line limit (safe default)
```

### Opção B — variáveis de ambiente {#option-b--environment-variables}

| Variable | Required | Description |
|----------|:--------:|-------------|
| `IRC_SERVER` | ✅ | Hostname do servidor IRC (ex.: `irc.libera.chat`) |
| `IRC_CHANNEL` | ✅ | Canal(is) para entrar — separe por vírgula para vários |
| `IRC_NICKNAME` | ✅ | Nickname do bot (padrão: `hermes-bot`) |
| `IRC_PORT` | — | Porta do servidor (padrão: `6697` com TLS, `6667` sem) |
| `IRC_USE_TLS` | — | Usar TLS (`true`/`false`; padrão `true` na porta 6697) |
| `IRC_SERVER_PASSWORD` | — | Senha do servidor para o comando `PASS` |
| `IRC_NICKSERV_PASSWORD` | — | Senha NickServ para IDENTIFY automático ao conectar |
| `IRC_ALLOWED_USERS` | — | Nicks separados por vírgula autorizados a falar com o bot |
| `IRC_ALLOW_ALL_USERS` | — | Permitir que qualquer pessoa no canal fale com o bot (apenas dev) |
| `IRC_HOME_CHANNEL` | — | Canal para entrega de cron / notificações (padrão: `IRC_CHANNEL`) |

## Controle de acesso {#access-control}

Por padrão, apenas nicks listados em `allowed_users` (ou `IRC_ALLOWED_USERS`) podem falar com o bot. Deixe a lista vazia **e** defina `IRC_ALLOW_ALL_USERS=true` para permitir que qualquer pessoa no canal converse com o Hermes — útil para testes, mas não recomendado em redes públicas, já que nicks IRC não são autenticados a menos que a rede exija NickServ.

Se sua rede registra nicks, defina `IRC_NICKSERV_PASSWORD` (ou `nickserv_password`) para o bot se identificar ao NickServ ao conectar e manter seu nick registrado.

## Canais vs. DMs {#channels-vs-dms}

- Mensagens em um canal que o bot entrou são tratadas como conversa de **grupo**.
- Mensagens privadas ao bot são tratadas como **mensagens diretas**.

Jobs cron e notificações são entregues ao **canal home** — `IRC_HOME_CHANNEL` se definido, caso contrário o primeiro `IRC_CHANNEL`.

## Inicie o gateway {#run-the-gateway}

```bash
hermes gateway start
```

Verifique o status com `hermes gateway status` — o estado da conexão IRC é reportado lá, inclusive para configurações só com env.

## Notas {#notes}

- Respostas longas do agente são divididas automaticamente em várias linhas `PRIVMSG` para respeitar o limite de linha do IRC (`max_message_length`, padrão 450 bytes após overhead de protocolo).
- O adaptador adquire um lock de credencial com escopo por servidor+nick, então dois perfis Hermes não disputam a mesma identidade IRC.
