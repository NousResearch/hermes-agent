---
sidebar_position: 16
title: "Gerenciar Hermes Cloud com MCP"
description: "Conecte o Hermes Agent ao servidor MCP do Nous Portal para o agente local listar, iniciar, parar e gerenciar suas instâncias Hermes Cloud de forma conversacional"
---

# Gerenciar Hermes Cloud com MCP {#manage-hermes-cloud-with-mcp}

[Hermes Cloud](https://portal.nousresearch.com/cloud) roda instâncias hospedadas do Hermes Agent para você. Normalmente você as gerencia na página `/agents` do [Nous Portal](/integrations/nous-portal). Este guia conecta seu Hermes Agent **local** ao servidor MCP do Portal para você gerenciar essas instâncias cloud só pedindo — "list my cloud agents", "restart the stopped one", "what's it costing me" — sem sair do terminal.

É um servidor [MCP](/user-guide/features/mcp) padrão hospedado pela Nous Research, gated pelo mesmo login OAuth que você já usa no Portal. Uma vez conectado, o Hermes ganha duas tools que pode chamar em seu nome.

## O que você pode fazer com isso {#what-you-can-do-with-it}

Uma vez conectado, o modelo pode chamar isto na sua org Hermes Cloud:

| Peça… | Por baixo dos panos |
|----------|----------------|
| "List my cloud agents" | `agents` (list) |
| "What's the status of `<name>`?" | `agents` (get / status) |
| "Roughly what is this instance costing?" | `agents` (cost_estimate) |
| "Start / stop / restart `<name>`" | `agent` (start / stop / restart) |
| "Spin up a new instance called `<name>`" | `agent` (create) |
| "Destroy `<name>`" | `agent` (destroy) |
| "Update the env / image on `<name>`" | `agent` (update_env / update_image) |

Toda chamada roda contra **sua** org com sua identidade do Portal, e a membership é re-checada em cada chamada — a conexão só toca instâncias que você já controla na web UI.

## Pré-requisitos {#prerequisites}

- Uma conta no [Nous Portal](/integrations/nous-portal) com acesso ao [Hermes Cloud](https://portal.nousresearch.com/cloud) (pelo menos uma instância, ou a habilidade de criar uma).
- Suporte MCP instalado. Se você usou o script de install padrão, já está lá; caso contrário:

  ```bash
  cd ~/.hermes/hermes-agent
  uv pip install -e ".[mcp]"
  ```

Você **não** precisa de uma API key ou client secret separada — o servidor usa OAuth com PKCE, e o login é uma ida e volta no browser.

## Passo 1: adicionar o servidor {#step-1-add-the-server}

```bash
hermes mcp add --url https://portal.nousresearch.com/mcp --auth oauth hermes-cloud
```

`--auth oauth` diz ao Hermes que este é um servidor HTTP protegido por OAuth. Na primeira conexão o Hermes:

1. Descobre os endpoints OAuth do servidor automaticamente (metadata RFC 9728 / 8414).
2. Se registra como client (RFC 7591 Dynamic Client Registration) — sem secret para copiar.
3. Abre seu browser no Portal para sign-in e authorize.
4. Guarda o token resultante em `~/.hermes/mcp-tokens/` e o reutiliza (refresh é automático).

### Escolhendo uma organization {#choosing-an-organization}

Se sua conta do Portal pertence a **mais de uma organization**, o browser mostra um **org picker** durante a authorization — escolha qual org esta conexão deve gerenciar. A escolha é feita uma vez, no browser; não há nada para passar na linha de comando. Contas de uma org só pulam este passo e fazem bind automaticamente.

Se um dia você precisar apontar a conexão para outra org, remova e re-adicione o servidor (`hermes mcp remove hermes-cloud`, depois o comando `add` de novo) e escolha a outra org no browser.

## Passo 2: verificar a conexão {#step-2-verify-it-connected}

```bash
hermes mcp test hermes-cloud
```

Depois inicie (ou recarregue) uma sessão:

```bash
hermes chat
```

```text
/reload-mcp
```

Peça uma pergunta read-only para confirmar que as tools estão ao vivo:

```text
List my Hermes Cloud agents and their current status.
```

Você deve receber de volta as mesmas instâncias que vê na página `/agents` do Portal.

## Passo 3: usar {#step-3-use-it}

Perguntas read-only são sempre seguras:

```text
Which of my cloud agents is currently running, and roughly what is each one costing?
```

Ações de lifecycle mapeiam para pedidos em prosa:

```text
Restart the instance called research-bot.
```

```text
Create a new Hermes Cloud instance named scratch, then tell me when it's ready.
```

O Hermes reporta o que cada tool retornou — a lista de instâncias, o novo status, os detalhes da instância criada — para você confirmar que a ação pousou.

## Configuração {#configuration}

Depois de `hermes mcp add`, o servidor vive em `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  hermes-cloud:
    url: "https://portal.nousresearch.com/mcp"
    auth: oauth
```

Nenhuma credencial vai em `config.yaml` — o token OAuth fica separado em `~/.hermes/mcp-tokens/`, do mesmo jeito que o refresh token do Portal fica fora da sua config.

### Limitando a superfície de tools {#limiting-the-tool-surface}

O servidor expõe tools de leitura (`agents`) e de mutação (`agent`). Se você quiser a conexão **read-only** — listar e inspecionar, mas nunca start/stop/create/destroy — restrinja à tool `agents`:

```yaml
mcp_servers:
  hermes-cloud:
    url: "https://portal.nousresearch.com/mcp"
    auth: oauth
    tools:
      include: [agents]
```

Rode `/reload-mcp` depois de mudar a config. Veja [Usar MCP com o Hermes](/guides/use-mcp-with-hermes) para o modelo completo de filtering (`include`/`exclude`, `prompts`, `resources`).

## Troubleshooting {#troubleshooting}

### O browser mostra um org picker e não sei qual escolher {#the-browser-shows-an-org-picker-and-im-not-sure-which-to-choose}

Você pertence a várias organizations do Portal. Escolha a org cujas instâncias Hermes Cloud você quer gerenciar por esta conexão. Se não tiver certeza, é a org dona das instâncias que você vê na página `/agents` do Portal. Você pode reescolher depois removendo e re-adicionando o servidor.

### "invalid_client" ou "unknown client" na conexão {#invalid_client-or-unknown-client-on-connect}

O client registration guardado não casa mais com o servidor (por exemplo, você conectou a um ambiente diferente antes). Limpe o estado OAuth em cache deste servidor e re-adicione:

```bash
hermes mcp remove hermes-cloud
rm -f ~/.hermes/mcp-tokens/hermes-cloud.*
hermes mcp add --url https://portal.nousresearch.com/mcp --auth oauth hermes-cloud
```

### As tools não aparecem depois de adicionar o servidor {#the-tools-arent-showing-up-after-adding-the-server}

Recarregue MCP dentro da sessão e cheque de novo:

```text
/reload-mcp
```

```text
Tell me which MCP-backed tools are available right now.
```

Se ainda faltarem, rode `hermes mcp test hermes-cloud` para ver o erro de conexão direto.

### Pede login de novo {#it-asks-me-to-log-in-again}

Tokens OAuth fazem refresh automaticamente, mas se o Portal invalidar sua sessão (mudança de senha, revoke, expiry) a próxima chamada pede re-authorize. Rode de novo o comando `hermes mcp add` — o fluxo do browser reminta um token.

### Headless / SSH / host remoto {#headless--ssh--remote-host}

O callback OAuth do browser roda na máquina onde o Hermes está rodando. Num host remoto, faça forward da porta loopback pelo SSH — o mesmo padrão de qualquer outro login OAuth. Veja [OAuth over SSH / Remote Hosts](/guides/oauth-over-ssh).

## Veja também {#see-also}

- **[Nous Portal](/integrations/nous-portal)** — a subscription, os models e o Tool Gateway atrás do mesmo login
- **[Usar MCP com o Hermes](/guides/use-mcp-with-hermes)** — conectar e filtrar servidores MCP em geral
- **[Visão geral de MCP](/user-guide/features/mcp)** — o que é MCP e como o Hermes usa
- **[Referência de configuração MCP](/reference/mcp-config-reference)** — cada campo de `mcp_servers`, incluindo `auth: oauth`
- **[OAuth over SSH](/guides/oauth-over-ssh)** — login a partir de ambientes remotos ou só-browser
