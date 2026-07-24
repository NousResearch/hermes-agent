---
sidebar_position: 18
---

# Photon iMessage {#photon-imessage}

Conecte o Hermes ao **iMessage** pelo [Photon][photon], um serviço
gerenciado que cuida da alocação de linha Apple e da camada de
prevenção de abuso para você não precisar rodar seu próprio relay Mac.

O tier gratuito usa o pool compartilhado de linhas iMessage do Photon — destinatários
diferentes podem ver números de envio diferentes, mas cada conversa
permanece estável. O tier Business pago dá a cada usuário o mesmo
número dedicado; o plugin suporta ambos, e o tier gratuito é o
ponto de partida recomendado.

:::info Grátis para começar
O pool de linhas compartilhadas do Photon é gratuito. Nenhuma assinatura é necessária para enviar
seu primeiro iMessage pelo Hermes — apenas um número de telefone que possamos vincular à
sua conta.
:::

## Arquitetura {#architecture}

O Photon é um canal de **conexão persistente**, como Discord ou Slack —
**sem webhook, sem URL pública, sem signing secret para gerenciar.**

O SDK `spectrum-ts` mantém um **stream gRPC** de longa duração ao Photon para
ambas as direções. Como o SDK é só TypeScript, o Hermes o roda em um
**sidecar Node** pequeno e supervisionado, conversando com ele por loopback:

- **Entrada** — o sidecar consome o stream gRPC `app.messages` do SDK
  e encaminha cada mensagem ao adaptador Python por loopback
  `GET /inbound` (NDJSON). O adaptador deduplica e despacha ao
  agente, reconectando automaticamente se o stream cair.
- **Saída** — respostas são POSTs loopback ao sidecar, que chama
  `space.send(...)` no SDK.

O plugin Python inicia, supervisiona e encerra o sidecar
automaticamente.

## Pré-requisitos {#prerequisites}

- Uma conta Photon — cadastre-se em [app.photon.codes][app]
- **Node.js 18.17 ou mais recente** no PATH (`node --version`)
- Um número de telefone que receba iMessage (usado para vincular sua conta)

Só isso — não há URL pública ou túnel para configurar.

## Setup inicial {#first-time-setup}

Execute o assistente unificado do gateway e escolha **Photon iMessage**:

```bash
hermes gateway setup
```

…ou execute o setup Photon diretamente (o assistente chama o mesmo fluxo):

```bash
# Device-code login + project + user + sidecar deps, all in one
hermes photon setup --phone +15551234567
```

O setup, em ordem:

1. **Login por device** (`client_id=photon-cli`) — abre
   `https://app.photon.codes/` para aprovação e armazena o bearer token.
2. **Encontra ou cria** o projeto `Hermes Agent` na sua conta.
3. **Habilita Spectrum**, lê o id Spectrum do projeto e rotaciona
   o secret do projeto.
4. **Registra seu número de telefone** como usuário Spectrum — pulado se um
   usuário com esse número já existir, então reexecutar é seguro.
5. **Imprime sua linha iMessage atribuída** — o número para o qual você envia SMS para alcançar
   seu agente.
6. **Executa `npm install`** dentro do diretório sidecar do plugin.

Credenciais de runtime são gravadas em `~/.hermes/.env`
(`PHOTON_PROJECT_ID` = o id do projeto Spectrum, `PHOTON_PROJECT_SECRET`),
o mesmo lugar onde todo outro canal guarda seu token. Metadados de gerenciamento
(device token, id de projeto do dashboard) ficam em `~/.hermes/auth.json` sob
`credential_pool.photon` / `credential_pool.photon_project`.

## Autorizando usuários {#authorizing-users}

O Photon usa o mesmo modelo de autorização de todo canal Hermes.
Escolha uma abordagem:

**Pareamento por DM (padrão).** Quando um número desconhecido envia mensagem à sua linha
Photon, o Hermes responde com um código de pareamento. Aprove com:

```bash
hermes pairing approve photon <CODE>
```

Use `hermes pairing list` para ver códigos pendentes e usuários aprovados.

**Pré-autorize números específicos** (em `~/.hermes/.env`):

```bash
PHOTON_ALLOWED_USERS=+15551234567,+15559876543
```

**Acesso aberto** (apenas dev, em `~/.hermes/.env`):

```bash
PHOTON_ALLOW_ALL_USERS=true
```

Quando `PHOTON_ALLOWED_USERS` está definido, remetentes desconhecidos são ignorados
silenciosamente em vez de receberem código de pareamento (a allowlist indica que você
restringiu o acesso deliberadamente).

### Exigir menções em chats de grupo {#require-mentions-in-group-chats}

Por padrão o Hermes responde a toda DM autorizada e mensagem de grupo.
Para tornar chats de grupo opt-in, habilite gating por menção (DMs ainda
sempre funcionam):

```yaml
gateway:
  platforms:
    photon:
      enabled: true
      require_mention: true
```

Com `require_mention: true`, mensagens de grupo são ignoradas a menos que
correspondam a um padrão wake-word. Os padrões padrão correspondem a variantes de `Hermes` e
`@Hermes agent`. Para um nome de agente customizado, defina padrões regex:

```yaml
gateway:
  platforms:
    photon:
      require_mention: true
      mention_patterns:
        - '(?<![\w@])@?amos\b[,:\-]?'
```

Ambas as chaves também aceitam env vars (`PHOTON_REQUIRE_MENTION`,
`PHOTON_MENTION_PATTERNS`). Este é o mesmo modelo de gating por menção que o
canal iMessage BlueBubbles usa.

## Inicie o gateway {#start-the-gateway}

```bash
hermes gateway start
```

Você verá algo como:

```
[photon] connected — sidecar on 127.0.0.1:8789, streaming inbound over gRPC
```

Envie um iMessage para seu número atribuído e o Hermes responderá.

## Status e solução de problemas {#status--troubleshooting}

```bash
hermes photon status
```

Imprime credenciais salvas, saúde do sidecar, seu número registrado e a
linha iMessage atribuída que o Hermes usa. Quando um token Photon e projeto do dashboard
estão disponíveis, `status` atualiza linhas de número ausentes do dashboard
sem provisionar novas linhas.

```
Photon iMessage status
──────────────────────
  device token        : ✓ stored
  dashboard project   : 3c90c3cc-0d44-4b50-...
  spectrum project id : sp-...
  project secret      : ✓ stored
  my number           : +15551234567
  assigned number     : +16282679185
  node binary         : /usr/bin/node
  sidecar deps        : ✓ installed
```

Problemas comuns:

- **`sidecar deps : ✗ run hermes photon install-sidecar`** — Node está
  instalado mas `spectrum-ts` não. Execute o comando sugerido.
- **`device token : ✗ missing`** — execute `hermes photon setup` para fazer login.
- **`No iMessage line assigned yet`** — Spectrum está habilitado mas nenhuma linha
  foi provisionada; reexecute `hermes photon setup` ou verifique o
  [dashboard][app].
- **Sidecar não inicia** — confirme que `node --version` é 18.17+ e que
  `hermes photon install-sidecar` completou sem erros.

## Limites atuais {#limits-today}

- **Anexos recebidos são só metadados.** Eventos recebidos carregam
  filename + MIME type; o agente vê um marcador mas ainda não pode ler os
  bytes. O SDK expõe bytes de anexo via `content.read()`, então isso
  é um follow-up do sidecar.
- **Anexos enviados são suportados.** O Hermes envia imagens, notas de voz,
  vídeo e documentos pelos builders de conteúdo `attachment()` /
  `voice()` do spectrum-ts via o endpoint `/send-attachment`
  do sidecar. Legendas chegam como bubble iMessage separado após a
  mídia.
- **Quotas gratuitas do Photon:** 5.000 mensagens por servidor por dia,
  50 iniciações de conversa nova por linha compartilhada por dia. Aumentos
  disponíveis — email `help@photon.codes`.

## Variáveis de ambiente {#env-vars}

| Variable                  | Default            | Notes                                      |
|---------------------------|--------------------|--------------------------------------------|
| `PHOTON_PROJECT_ID`       | from `.env`        | Id do projeto Spectrum (o `projectId` do SDK); definido pelo setup |
| `PHOTON_PROJECT_SECRET`   | from `.env`        | Secret do projeto; definido pelo setup               |
| `PHOTON_SIDECAR_PORT`     | `8789`             | Porta loopback do sidecar de controle + canal de entrada |
| `PHOTON_SIDECAR_AUTOSTART`| `true`             | Se o adaptador inicia o sidecar     |
| `PHOTON_NODE_BIN`         | `which node`       | Sobrescreve o caminho do binário Node              |
| `PHOTON_HOME_CHANNEL`     | (unset)            | Id de space padrão para cron / notificações  |
| `PHOTON_HOME_CHANNEL_NAME`| (unset)            | Rótulo legível para o canal home           |
| `PHOTON_ALLOWED_USERS`    | (unset)            | Allowlist E.164 separada por vírgula            |
| `PHOTON_ALLOW_ALL_USERS`  | `false`            | Apenas dev — aceitar qualquer remetente               |
| `PHOTON_REQUIRE_MENTION`  | `false`            | Exigir wake word antes de responder em grupos |
| `PHOTON_MENTION_PATTERNS` | Hermes wake words  | Lista JSON / vírgula / regex por linha para menções em grupo |
| `PHOTON_DASHBOARD_HOST`   | `app.photon.codes` | Sobrescreve o host do dashboard / device-login |
| `PHOTON_SPECTRUM_HOST`    | `spectrum.photon.codes` | Sobrescreve o host da API Spectrum |

[photon]: https://photon.codes/
[app]: https://app.photon.codes/
