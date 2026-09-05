---
sidebar_position: 30
title: "Hermes Relay"
description: "Conecte o Hermes a plataformas de mensagens através de um connector de relay que detém as credenciais da plataforma — cadastro, capacidades, configuração e solução de problemas"
---

# Hermes Relay (Connector)

:::warning Experimental
O Relay é **experimental**. O contrato de comunicação, o esquema de autenticação e a configuração
podem mudar sem um ciclo de depreciação enquanto o sistema está sendo validado.
:::

O Hermes Relay não é, em si, uma plataforma de chat — é um **sistema de connector** que
permite que seu gateway sirva de front para uma ou mais plataformas de mensagens reais (Discord,
Telegram, Slack, WhatsApp, …) **sem manter nenhuma credencial de plataforma**. Um
serviço separado, o *connector*, é dono dos tokens de bot e dos sockets da plataforma.
Seu gateway disca **para fora** rumo ao connector através de um único WebSocket
autenticado, recebe um descritor de capacidades no handshake e então troca
eventos de mensagem normalizados (entrada) e ações (saída) por esse socket.

Propriedades principais:

- **Rede somente de saída.** O gateway nunca abre uma porta de entrada.
  Mensagens de entrada voltam pelo mesmo WebSocket que o gateway discou, então
  o relay funciona atrás de NAT e em hosts sem IP público.
- **Nenhum segredo de plataforma no gateway.** Os tokens de bot ficam no connector.
  URLs de mídia de plataforma protegidas por autenticação são re-hospedadas do lado do connector, então as
  credenciais de plataforma nunca cruzam a rede.
- **Agnóstico de plataforma.** O gateway aprende o que a plataforma servida como front consegue fazer
  (limites de tamanho de mensagem, dialeto de markdown, suporte a edição/thread/streaming, e
  o conjunto exato de operações suportadas) a partir do descritor do handshake, não
  de lógica de plataforma fixada no código.

A interface formal gateway ⇄ connector vive no repositório em
`docs/relay-connector-contract.md`.

## Quando usar o Relay {#when-to-use-relay}

O Relay é para implantações em que um serviço de connector hospedado ou compartilhado gerencia
o lado da plataforma — por exemplo, hospedagem multi-tenant em que um único bot compartilhado
serve de front para os agentes de muitos usuários, ou configurações em que você não quer tokens de bot na
máquina do gateway. Se você roda seus próprios bots diretamente, use os adapters nativos de plataforma
([Telegram](/user-guide/messaging/telegram),
[Discord](/user-guide/messaging/discord), etc.) em vez disso.

## Cadastro {#enrollment}

Um gateway auto-hospedado se autentica no connector com um segredo por gateway.
O `hermes gateway enroll` resgata um **token de cadastro de uso único**
(emitido pelo connector quando a rota do seu tenant é provisionada e entregue
junto com a config do seu gateway) por esse segredo:

```bash
hermes gateway enroll \
  --token <enrollment-token> \
  --connector-url wss://connector.example.com/relay
```

O que ele faz:

1. Resolve um token de acesso novo do Nous Portal a partir do seu login existente
   (`~/.hermes/auth.json`) — isso prova qual org Nous (tenant) você é dono. Se
   `gateway.idp.token_url` estiver configurado, o seu próprio IdP é usado em vez
   disso (o caminho air-gapped / IdP auto-hospedado, sem envolver o Nous Portal):
   com `client_id`/`client_secret` configurados, executa um grant OAuth2
   client-credentials genérico; com nenhum dos dois configurado a URL é tratada
   como um endpoint de token ambiente (GET simples cujo body da resposta é o
   token — o padrão de metadata-server, ex.: `$DOMINO_API_PROXY/access-token` da
   Domino). Configurar só uma das duas credenciais é um erro.
2. Envia (POST) o token de cadastro e um id de gateway para o endpoint
   `/relay/enroll` do connector via TLS.
3. O connector verifica o token (assinatura, uso único, correspondência de tenant),
   emite um segredo por gateway mais uma chave de entrega por tenant, e os retorna
   uma única vez.
4. Persiste as credenciais em `~/.hermes/.env`:
   `GATEWAY_RELAY_ID`, `GATEWAY_RELAY_SECRET`, `GATEWAY_RELAY_DELIVERY_KEY`
   (mais `GATEWAY_RELAY_URL` / `GATEWAY_RELAY_WAKE_URL` quando fornecidos).

Reinicie o gateway depois disso para aplicar o novo ambiente.

Flags:

| Flag | Descrição |
|------|-------------|
| `--token` | O token de cadastro de uso único. Também configurável via `GATEWAY_RELAY_ENROLL_TOKEN`. |
| `--connector-url` | URL base do connector ou de relay (`wss://…/relay` ou `https://…`). Também configurável via `GATEWAY_RELAY_URL` ou `gateway.relay_url` em `config.yaml`. |
| `--gateway-id` | Id estável para esta instância de gateway (usado para granularidade do kill-switch). Padrão: `gw-<hostname>`. |
| `--wake-url` | URL alcançável opcional que o connector cutuca (GET sem payload) para acordar este gateway quando trabalho em buffer chega enquanto ele está ocioso. Persistido como `GATEWAY_RELAY_WAKE_URL`. Sem isso, o gateway ainda drena as mensagens em buffer sempre que reconecta em seguida. |

:::note Instalações gerenciadas
O `hermes gateway enroll` recusa a rodar em instalações gerenciadas/hospedadas — nesse caso a
plataforma de hospedagem provisiona o segredo de relay diretamente no ambiente
do container.
:::

## Configuração {#configuration}

O Relay é ativado quando uma URL de relay do connector está configurada — não existe
uma feature flag separada. Implantações que não a definem não são afetadas.

| Configuração | Onde | Significado |
|---------|-------|---------|
| `GATEWAY_RELAY_URL` | env (`~/.hermes/.env`) | URL do WebSocket de relay do connector. A presença habilita a plataforma relay. |
| `gateway.relay_url` | `config.yaml` | O mesmo de acima, na forma de arquivo de config (env tem precedência). |
| `GATEWAY_RELAY_ID` | env | O id desta instância de gateway (gravado pelo `enroll`). |
| `GATEWAY_RELAY_SECRET` | env | Segredo por gateway que autentica o upgrade do WebSocket (gravado pelo `enroll`). |
| `GATEWAY_RELAY_DELIVERY_KEY` | env | Chave de entrega por tenant (gravada pelo `enroll`; mantida por compatibilidade futura). |
| `GATEWAY_RELAY_WAKE_URL` / `gateway.relay_wake_url` | env / `config.yaml` | Alvo opcional de wake-poke para gateways ociosos/suspensos. |
| `GATEWAY_RELAY_PLATFORMS` | env | Lista de plataformas que este gateway serve de front sobre uma única conexão, separadas por vírgula (ex.: `discord,telegram`). Normalmente definido pelo deployment/orchestrator. |
| `GATEWAY_RELAY_BOT_IDS` | env | Mapa JSON de identidades de bot por plataforma, ex.: `{"discord": {"botId": "…"}}`. Usado em conjunto com `GATEWAY_RELAY_PLATFORMS`. |
| `gateway.idp.token_url` | `config.yaml` | Quando definido, o cadastro/provisionamento se autentica contra o seu próprio IdP em vez do Nous Portal: OAuth2 client-credentials quando `gateway.idp.client_id`/`client_secret` também estão definidos; senão um endpoint de token ambiente (GET simples retornando o token, cru ou `{"access_token": …}`). |

## Capacidades suportadas {#supported-capabilities}

O que de fato funciona em uma conexão de relay é negociado no handshake: o
connector anuncia uma lista `supported_ops`, e o gateway só usa uma
operação que o connector anuncia explicitamente (connectors mais antigos caem de volta para um
conjunto legado `send`/`edit`/`typing`/`follow_up`). Flags de capacidade por plataforma
(streaming baseado em edição, threads, streaming de rascunho, dialeto de markdown, limite de
tamanho de mensagem) também vêm do descritor do handshake. Sujeito a essa
negociação, o relay suporta:

- **Mensagens de texto e streaming** — envios, respostas e streaming
  progressivo baseado em edição quando a plataforma servida como front suporta edição de mensagem;
  caso contrário, a saída degrada para uma mensagem por segmento.
- **Mídia, em ambas as direções** — imagens, voz, áudio, vídeo e
  documentos de saída são enviados ao connector (ou referenciados por URL pública) e
  entregues através da faixa de upload nativa de cada plataforma, com legendas.
  Anexos de entrada são localizados como arquivos para o agente; URLs de plataforma
  protegidas por autenticação são re-hospedadas do lado do connector, então as credenciais de plataforma nunca
  chegam ao gateway. Re-hospedagens de mídia têm limite de 25 MB e expiram (~1 hora).
- **Prompts interativos nativos** — aprovações de execução, confirmações e perguntas de
  clarify são renderizados com **controles nativos da plataforma** (botões do Discord,
  teclados inline do Telegram, ações do Slack Block Kit, mensagens de botão/lista do WhatsApp)
  em vez de fallbacks em texto numerado. Cliques em botões voltam como
  respostas de prompt autenticadas do usuário que de fato clicou, então os
  portões de autorização do gateway se aplicam exatamente como a uma resposta digitada. A expiração de prompts
  é aplicada do lado do gateway.
- **Ciclo de vida de reações de confirmação** — as reações de status de processamento do bot
  (👀 enquanto trabalha, ✅/❌ ao concluir) funcionam via o relay. Reações são
  best-effort: uma reação falha nunca falha um turno.
- **Ciclo de vida de threads** — criar threads de handoff e renomear threads
  (incluindo renomeações semânticas com título gerado por LLM) através das operações abstratas de plataforma
  `thread_create` / `thread_rename`, com uma proteção anti-sobrescrita para que a
  renomeação manual de um humano prevaleça. A disponibilidade depende da plataforma (ex.:
  threads do Slack não podem ser renomeadas; o WhatsApp não tem threads).
- **Indicadores de digitação** — o gateway envia digitação (e parar-de-digitar) através
  do connector enquanto processa.
- **Metadados de chat** — buscas de `get_chat_info` são repassadas ao connector
  quando anunciadas.
- **Entrega em buffer e wake** — quando o gateway fica ocioso ou se desconecta,
  o connector armazena em buffer de forma durável as mensagens de entrada e as reproduz em ordem ao
  reconectar (com confirmação de ack, sem perda ou duplicação). Se uma URL de wake estiver registrada,
  o connector a cutuca quando trabalho em buffer chega para um gateway adormecido.

O front multi-plataforma é suportado: um gateway pode servir de front para várias plataformas
(ex.: Discord *e* Telegram) sobre uma única conexão de relay, com cada
mensagem de saída marcada para a plataforma que ela tem como alvo.

## Solução de problemas {#troubleshooting}

**O cadastro falha com 401** — o connector não conseguiu verificar seu token de
identidade. Faça login novamente com `hermes auth add nous` (ou `hermes setup`) e tente de novo.

**O cadastro falha com 403** — o token de cadastro é inválido, expirou,
já foi usado, ou pertence a outro tenant. Tokens de cadastro são de
uso único; peça um novo a quem provisionou a rota do seu tenant.

**"Could not reach the connector"** — verifique a URL do connector. Você pode colar
tanto a URL de discagem `wss://…/relay` quanto a URL base `https://…`; a CLI mapeia
entre elas automaticamente.

**O `enroll` recusa a rodar** — você está em uma instalação gerenciada/hospedada, onde o
segredo de relay é provisionado pela plataforma de hospedagem. O auto-cadastro é apenas
para gateways auto-hospedados.

**A plataforma relay aparece como desabilitada depois de ter funcionado antes** — um fechamento de WebSocket
com código 4401 *depois* de um handshake bem-sucedido significa que o segredo do gateway
foi revogado (ex.: a instância foi desprovisionada). O gateway deliberadamente
para de tentar reconectar e reporta o relay como desabilitado em vez de tentar de novo. Um 4401
*antes* de qualquer handshake bem-sucedido é tratado como uma condição transitória de
ainda-não-provisionado e é tentado novamente normalmente.

**Nada mudou depois de cadastrar** — o gateway lê `GATEWAY_RELAY_*` na
inicialização. Reinicie-o (`hermes gateway restart`).

**Um recurso (botões, mídia, threads…) degrada silenciosamente para texto simples** — o
connector da sua plataforma não anunciou essa operação no seu `supported_ops`
do handshake. O gateway intencionalmente cai de volta para o comportamento em texto
em vez de enviar uma operação que o connector não sabe lidar.
