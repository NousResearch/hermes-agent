---
sidebar_position: 8
title: "Referência de Configuração MCP"
description: "Referência das chaves de configuração MCP do Hermes Agent, semântica de filtragem e política de ferramentas utilitárias"
---

# Referência de Configuração MCP {#mcp-config-reference}

Esta página é a referência compacta que complementa a documentação principal de MCP.

Para orientação conceitual, consulte:
- [MCP (Model Context Protocol)](/user-guide/features/mcp)
- [Usar MCP com Hermes](/guides/use-mcp-with-hermes)

## Formato raiz da configuração {#root-config-shape}

```yaml
mcp_servers:
  <server_name>:
    command: "..."      # servidores stdio
    args: []
    env: {}

    # OU
    url: "..."          # servidores HTTP
    headers: {}

    # Configurações TLS opcionais para HTTP/SSE:
    ssl_verify: true                # bool ou caminho para um bundle CA (PEM)
    client_cert: "/path/to/cert.pem"  # certificado cliente mTLS (veja abaixo)
    # client_key: "/path/to/key.pem"  # opcional, quando a chave está em arquivo separado

    enabled: true
    timeout: 120
    connect_timeout: 60
    supports_parallel_tool_calls: false
    tools:
      include: []
      exclude: []
      resources: true
      prompts: true
```

## Chaves de servidor {#server-keys}

| Chave | Tipo | Aplica-se a | Significado |
|---|---|---|---|
| `command` | string | stdio | Executável a ser iniciado |
| `args` | list | stdio | Argumentos para o subprocesso |
| `env` | mapping | stdio | Ambiente passado ao subprocesso |
| `url` | string | HTTP | Endpoint MCP remoto |
| `headers` | mapping | HTTP | Cabeçalhos para requisições ao servidor remoto |
| `ssl_verify` | bool or string | HTTP | Verificação TLS. `true` (padrão) usa CAs do sistema, `false` desabilita a verificação (inseguro), ou um caminho string para um bundle CA customizado (PEM) |
| `client_cert` | string or list | HTTP | Certificado cliente mTLS. String = caminho para um arquivo PEM contendo cert + key. List `[cert, key]` = arquivos separados. List `[cert, key, password]` = chave criptografada |
| `client_key` | string | HTTP | Caminho para a chave privada do cliente, quando `client_cert` é uma string e a chave está em arquivo separado |
| `enabled` | bool | both | Ignora o servidor por completo quando false |
| `timeout` | number | both | Timeout de chamada de ferramenta em segundos (padrão: `300`) |
| `connect_timeout` | number | both | Timeout de conexão inicial em segundos (padrão: `60`) |
| `protocol` | string | both | Negociação de era do protocolo: `auto` (padrão — handshake legado `initialize` primeiro, caindo para o probe stateless `server/discover` de 2026-07-28 quando o servidor rejeita o handshake como modern-only), `stateless` (probe `server/discover` primeiro; um retry legado), ou `legacy` (só handshake, sem fallback) |
| `supports_parallel_tool_calls` | bool | both | Permite que ferramentas deste servidor executem em paralelo |
| `skip_preflight` | bool | HTTP | Ignora a sonda fail-fast de content-type para endpoints Streamable HTTP válidos cujo HEAD/GET responde com content-type não-MCP (padrão: `false`) |
| `transport` | string | HTTP | Defina como `sse` para usar o transporte SSE em vez de Streamable HTTP |
| `keepalive_interval` | number | both | Cadência de ping de liveness em segundos (padrão: `180`, mínimo 5s). Defina abaixo do TTL de sessão do servidor para servidores que fazem GC de sessões ociosas rapidamente |
| `idle_timeout_seconds` | number | stdio | Reciclagem opcional do servidor stdio após tempo ocioso (`0` desabilita). Também pode ficar sob um mapping `lifecycle:` |
| `max_lifetime_seconds` | number | stdio | Reciclagem opcional do servidor stdio após idade (`0` desabilita). Também pode ficar sob um mapping `lifecycle:` |
| `tools` | mapping | both | Filtragem e política de ferramentas utilitárias |
| `auth` | string | HTTP | Método de autenticação. Defina como `oauth` para habilitar OAuth 2.1 com PKCE |
| `sampling` | mapping | both | Política de requisições LLM iniciadas pelo servidor (veja o guia MCP) |
| `elicitation` | mapping | both | Requisições de input do usuário iniciadas pelo servidor. `enabled` (padrão `true`) e `timeout` em segundos (padrão `300`). Requisições em modo form passam pela superfície de aprovação; modo URL é recusado (veja o guia MCP) |
| `trust` | string | both | Nível de confiança: `full` (padrão) ou `untrusted`. Em um servidor `untrusted`, toda chamada de ferramenta com capacidade de escrita (qualquer ferramenta sem anotação `readOnlyHint: true`) exige aprovação do usuário pela superfície de aprovação padrão antes de executar. `readOnlyHint` é uma *dica* fornecida pelo servidor — um servidor mentiroso pode no máximo pular aprovação para ferramentas que alega serem somente leitura, nunca ganhar acesso extra — então marque como `untrusted` qualquer servidor que você não controla totalmente. Valores não reconhecidos são tratados como `untrusted` (fail-closed) |

## Referências a variáveis de ambiente {#environment-variable-references}

Valores string em qualquer lugar de uma entrada de servidor (`env`, `headers`, `args`, `url`, …) podem referenciar variáveis de ambiente com `${VAR}` ou a forma SecretRef estilo Cursor `${env:VAR}` — ambas resolvem para a mesma variável, então snippets MCP copiados de configs Cursor / Claude funcionam sem alteração:

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "${env:GITHUB_TOKEN}"   # igual a "${GITHUB_TOKEN}"
```

Os valores resolvem a partir do escopo de segredos do profile ativo (com fallback para o ambiente do processo), então coloque o segredo em `~/.hermes/.env`. Uma variável não definida mantém o placeholder literal.

### Variáveis de contexto {#context-variables}

Além de variáveis de env, as variáveis de contexto estilo Cursor também são interpoladas (nomes são sensíveis a maiúsculas/minúsculas):

| Variável | Resolve para |
|---|---|
| `${userHome}` | Diretório home do usuário atual |
| `${workspaceFolder}` | Raiz do workspace da sessão (cwd do terminal da sessão quando conhecido, senão o cwd do processo) |
| `${workspaceFolderBasename}` | O basename de `${workspaceFolder}` |
| `${pathSeparator}` / `${/}` | O separador de caminho do SO (`os.sep`) |

```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "${workspaceFolder}"]
    env:
      CACHE_DIR: "${userHome}${/}.cache${/}mcp"
```

Qualquer outra referência `${...}` cai na busca de variável de env acima.

## Chaves de política `tools` {#tools-policy-keys}

| Chave | Tipo | Significado |
|---|---|---|
| `include` | string or list | Lista branca de ferramentas MCP nativas do servidor. Entradas podem ser nomes exatos ou globs estilo fnmatch (`*_radar_*`, `get_zones_*`) |
| `exclude` | string or list | Lista negra de ferramentas MCP nativas do servidor. Mesma semântica de nome exato / glob que `include` |
| `resources` | bool-like | Habilita/desabilita `list_resources` + `read_resource` |
| `prompts` | bool-like | Habilita/desabilita `list_prompts` + `get_prompt` |

## Semântica de filtragem {#filtering-semantics}

### `include` {#include}

Se `include` estiver definido, apenas essas ferramentas MCP nativas do servidor são registradas.

```yaml
tools:
  include: [create_issue, list_issues]
```

### `exclude` {#exclude}

Se `exclude` estiver definido e `include` não, todas as ferramentas MCP nativas do servidor exceto esses nomes são registradas.

```yaml
tools:
  exclude: [delete_customer]
```

### Precedência {#precedence}

Se ambos estiverem definidos, `include` vence.

```yaml
tools:
  include: [create_issue]
  exclude: [create_issue, delete_issue]
```

Resultado:
- `create_issue` ainda é permitido
- `delete_issue` é ignorado porque `include` tem precedência

## Política de ferramentas utilitárias {#utility-tool-policy}

O Hermes pode registrar estes wrappers utilitários por servidor MCP:

Resources:
- `list_resources`
- `read_resource`

Prompts:
- `list_prompts`
- `get_prompt`

### Desabilitar resources {#disable-resources}

```yaml
tools:
  resources: false
```

### Desabilitar prompts {#disable-prompts}

```yaml
tools:
  prompts: false
```

### Registro consciente de capacidades {#capability-aware-registration}

Mesmo com `resources: true` ou `prompts: true`, o Hermes só registra essas ferramentas utilitárias se a sessão MCP expuser de fato a capacidade correspondente.

Então isto é normal:
- você habilita prompts
- mas nenhuma utilidade de prompt aparece
- porque o servidor não suporta prompts

## `enabled: false` {#enabled-false}

```yaml
mcp_servers:
  legacy:
    url: "https://mcp.legacy.internal"
    enabled: false
```

Comportamento:
- nenhuma tentativa de conexão
- nenhuma descoberta
- nenhum registro de ferramenta
- a config permanece no lugar para reutilização posterior

## Comportamento com resultado vazio {#empty-result-behavior}

Se a filtragem remover todas as ferramentas nativas do servidor e nenhuma ferramenta utilitária for registrada, o Hermes não cria um toolset MCP runtime vazio para esse servidor.

## Configs de exemplo {#example-configs}

### Allowlist segura do GitHub {#safe-github-allowlist}

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
    tools:
      include: [list_issues, create_issue, update_issue, search_code]
      resources: false
      prompts: false
```

### Blacklist do Stripe {#stripe-blacklist}

```yaml
mcp_servers:
  stripe:
    url: "https://mcp.stripe.com"
    headers:
      Authorization: "Bearer ***"
    tools:
      exclude: [delete_customer, refund_payment]
```

### Servidor de docs somente resources {#resource-only-docs-server}

```yaml
mcp_servers:
  docs:
    url: "https://mcp.docs.example.com"
    tools:
      include: []
      resources: true
      prompts: false
```

### Certificado cliente TLS (mTLS) {#tls-client-certificate-mtls}

Para servidores HTTP/SSE que exigem certificado cliente, defina `client_cert` (e opcionalmente `client_key`):

```yaml
mcp_servers:
  # Cert + key combinados em um único arquivo PEM
  internal_api:
    url: "https://mcp.internal.example.com/mcp"
    client_cert: "~/secrets/mcp-client.pem"

  # Arquivos de cert e key separados
  partner_api:
    url: "https://mcp.partner.example.com/mcp"
    client_cert: "~/secrets/client.crt"
    client_key: "~/secrets/client.key"

  # Chave criptografada com passphrase (forma de lista com 3 elementos)
  bank_api:
    url: "https://mcp.bank.example.com/mcp"
    client_cert: ["~/secrets/client.crt", "~/secrets/client.key", "my-passphrase"]

  # Bundle CA customizado (CA privada / servidor autoassinado)
  lab_api:
    url: "https://mcp.lab.local/mcp"
    ssl_verify: "~/secrets/lab-ca.pem"
    client_cert: "~/secrets/lab-client.pem"
```

Notas:
- Caminhos suportam expansão de `~`. Arquivos ausentes falham rápido na conexão com mensagem de erro escopada ao servidor.
- `ssl_verify: false` desabilita a verificação de certificado do servidor por completo. Não use isso com serviços reais.
- Funciona tanto em transportes Streamable HTTP quanto SSE.

## Recarregar config {#reloading-config}

Após alterar a config MCP, recarregue os servidores com:

```text
/reload-mcp
```

## Nomenclatura de ferramentas {#tool-naming}

Ferramentas MCP nativas do servidor tornam-se:

```text
mcp__<server>__<tool>
```

Exemplos:
- `mcp__github__create_issue`
- `mcp__filesystem__read_file`
- `mcp__my_api__query_data`

Ferramentas utilitárias seguem o mesmo padrão de prefixo:
- `mcp__<server>__list_resources`
- `mcp__<server>__read_resource`
- `mcp__<server>__list_prompts`
- `mcp__<server>__get_prompt`

O delimitador de sublinhado duplo (`mcp__…__…`) segue a convenção usada por Claude Code, Codex e OpenCode, e desambigua a fronteira servidor/ferramenta mesmo quando qualquer componente contém sublinhados.

### Sanitização de nomes {#name-sanitization}

Qualquer caractere que não seja letra, dígito ou sublinhado (hífens, pontos, espaços, etc.) em nomes de servidor e de ferramenta é substituído por sublinhado antes do registro. Isso garante que os nomes de ferramenta sejam identificadores válidos para APIs de function-calling de LLM.

Por exemplo, um servidor chamado `my-api` expondo uma ferramenta `list-items.v2` torna-se:

```text
mcp__my_api__list_items_v2
```

Tenha isso em mente ao escrever filtros `include` / `exclude` — use o **nome original** da ferramenta MCP (com hífens/pontos), não a versão sanitizada.

## Autenticação OAuth 2.1 {#oauth-21-authentication}

Para servidores HTTP que exigem OAuth, defina `auth: oauth` na entrada do servidor:

```yaml
mcp_servers:
  protected_api:
    url: "https://mcp.example.com/mcp"
    auth: oauth
```

Comportamento:
- O Hermes usa o fluxo OAuth 2.1 PKCE do MCP SDK (descoberta de metadados, identificação de cliente, troca de token e refresh)
- Na primeira conexão, uma janela do navegador abre para autorização
- Tokens são persistidos em `~/.hermes/mcp-tokens/<server>.json` e reutilizados entre sessões
- Refresh de token é automático; reautorização só ocorre quando o refresh falha
- Aplica-se apenas ao transporte HTTP/StreamableHTTP (servidores baseados em `url`)

### Identificação de cliente: CIMD e DCR {#client-identification-cimd-and-dcr}

O Hermes se identifica para authorization servers com um **Client ID Metadata Document** (CIMD), o mecanismo que a spec MCP `2026-07-28` adotou no lugar de Dynamic Client Registration. O documento é publicado em
`https://nousresearch.github.io/hermes-agent/docs/oauth/client-metadata.json`, e essa URL *é* o `client_id` — o authorization server o busca para aprender o nome, logo e redirect URIs permitidos do Hermes. Nada é registrado por install, e nada é específico do usuário.

A escolha final pertence ao authorization server: o SDK envia a URL do documento como `client_id` só quando o server anuncia `client_id_metadata_document_supported: true` nos metadados, e caso contrário registra via DCR exatamente como antes. DCR está deprecated na spec MCP mas ainda é o que quase todo server deployed usa hoje.

#### Portas de callback {#callback-ports}

O documento declara um conjunto fixo de redirect URIs de loopback, e a spec exige que o redirect URI num pedido de autorização seja um *match exato de string* contra um deles — então um fluxo CIMD não pode usar a porta alta aleatória que o Hermes normalmente escolhe. O Hermes portanto fixa o callback em uma das portas `27890`–`27894`.

Esse pin precisa ser escolhido antes das capabilities do server serem conhecidas, porque o redirect URI é fixado no início do fluxo enquanto os metadados do server só chegam no meio. Então o Hermes fixa a porta para qualquer fluxo que *possa* acabar usando CIMD, e reverte para uma porta aleatória no restante:

- Um server ao qual o Hermes já se conectou antes, cujos metadados em cache não anunciam CIMD, mantém a porta aleatória que sempre usou.
- Um server que o Hermes nunca alcançou recebe uma porta fixada nesse primeiro login, já que adivinhar é a única forma de CIMD alguma vez poder ser usado.
- Qualquer coisa que moveria o callback para outro lugar também reverte: um `oauth.client_id` pré-registrado, um `oauth.client_secret`, um `oauth.client_name` ou `oauth.token_endpoint_auth_method` customizado, um override `oauth.redirect_uri` ou `oauth.redirect_port`, um login dirigido por dashboard ou desktop, um registro de cliente existente em disco, ou todas as cinco portas sendo seguradas por outros processos.

Cada porta fixada é bound assim que escolhida e mantida até o redirect do browser chegar, então dois logins concorrentes — um segundo profile, ou outro server no mesmo processo — não podem cair no mesmo listener.

#### Quando um server rejeita o documento {#when-a-server-rejects-the-document}

Se um server busca o documento e o recusa no endpoint de *token* (`invalid_client`), o Hermes loga a rejeição, a registra sob `~/.hermes/mcp-tokens/<server>.cimd-off`, e usa DCR para aquele server daí em diante.

Um server que não consegue buscar ou validar o documento de forma alguma aborta no endpoint de *authorization* em vez disso, antes de qualquer redirect. Não há sinal que o Hermes possa observar ali, então o browser mostra um erro invalid-client e o login dá timeout depois de cinco minutos. A mensagem de timeout nomeia o documento e aponta para `cimd: false`. Rodar `hermes mcp login <server>` limpa a rejeição registrada, então um documento corrigido ganha outra chance.

#### Chaves opcionais por server {#optional-per-server-keys}

```yaml
mcp_servers:
  protected_api:
    url: "https://mcp.example.com/mcp"
    auth: oauth
    oauth:
      client_metadata_url: "https://example.com/my-cimd.json"  # self-hosted document
      cimd: false                                              # force DCR
      user_agent: "My-MCP-Client/1.0"                          # token-request User-Agent
```

`client_metadata_url` deve ser uma URL HTTPS com path (sem origin bare, sem fragment, sem userinfo, sem segmentos `.`/`..`) que retorna `200` e `Content-Type: application/json` com **sem redirect** — authorization servers são proibidos de seguir redirects ao buscá-la. O Hermes ainda fixa seu callback no mesmo range `27890`–`27894`, então um documento self-hosted deve declarar todas as dez URIs de loopback (`http://127.0.0.1:<port>/callback` e `http://localhost:<port>/callback` para cada porta), e seu `client_id` deve ser sua própria URL.

`user_agent` substitui o `User-Agent` padrão da biblioteca HTTP **só em requests ao token-endpoint** (troca de authorization-code e refresh) — alguns authorization servers e WAFs rejeitam o valor padrão `python-httpx/...` ali. Nunca se aplica a tráfego MCP ou descoberta OAuth, e nenhum outro header de token-request é configurável. Valores vazios ou null são ignorados.

## Link Add to Hermes {#add-to-hermes-link}

Vendors e docs de MCP podem oferecer um botão **"Add to Hermes"** de um clique que abre o app desktop do Hermes com uma config de servidor pré-preenchida, espelhando o scheme `cursor://anysphere.cursor-deeplink/mcp/install` do Cursor:

```text
hermes://mcp/install?name=NAME&config=BASE64
```

- `name` — o nome do servidor. Deve corresponder a `^[A-Za-z0-9._-]{1,64}$`.
- `config` — o objeto de config do servidor como **JSON encoded em base64url** (base64 padrão também é aceito). O JSON decodificado deve ser um objeto com um campo string `url` (`http://`/`https://` apenas) ou um campo string `command`, e pode carregar qualquer uma das chaves de servidor documentadas acima. Payloads acima de 32KB são rejeitados.

Exemplo (JavaScript):

```js
const config = { url: 'https://mcp.example.com/mcp' }
const link = `hermes://mcp/install?name=example&config=${btoa(JSON.stringify(config))
  .replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')}`
```

Abrir o link nunca instala nada por si: o app desktop mostra um diálogo de confirmação com o nome do servidor e a config pretty-printed completa (com um cuidado extra para servidores baseados em `command`, que rodam um processo local), e o usuário precisa confirmar explicitamente. Nomes de servidor existentes nunca são sobrescritos — o usuário é pedido para renomear ou cancelar.
