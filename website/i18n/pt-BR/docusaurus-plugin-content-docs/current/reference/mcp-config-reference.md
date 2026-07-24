---
sidebar_position: 8
title: "Referência de Configuração MCP"
description: "Referência das chaves de configuração MCP do Hermes, semântica de filtragem e política de ferramentas utilitárias"
---

# Referência de Configuração MCP

Esta página é a referência compacta que acompanha a documentação principal de MCP.

Para orientação conceitual, veja:
- [MCP (Model Context Protocol)](/user-guide/features/mcp)
- [Usar MCP com o Hermes](/guides/use-mcp-with-hermes)

## Formato da configuração raiz {#root-config-shape}

```yaml
mcp_servers:
  <server_name>:
    command: "..."      # servidores stdio
    args: []
    env: {}

    # OU
    url: "..."          # servidores HTTP
    headers: {}

    # Configurações opcionais de TLS para HTTP/SSE:
    ssl_verify: true                # bool ou caminho para um pacote de CA (PEM)
    client_cert: "/path/to/cert.pem"  # certificado de cliente mTLS (veja abaixo)
    # client_key: "/path/to/key.pem"  # opcional, quando a chave está em um arquivo separado

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

## Chaves do servidor {#server-keys}

| Chave | Tipo | Aplica-se a | Significado |
|---|---|---|---|
| `command` | string | stdio | Executável a ser iniciado |
| `args` | list | stdio | Argumentos para o subprocesso |
| `env` | mapping | stdio | Ambiente passado ao subprocesso |
| `url` | string | HTTP | Endpoint MCP remoto |
| `headers` | mapping | HTTP | Cabeçalhos para requisições ao servidor remoto |
| `ssl_verify` | bool ou string | HTTP | Verificação TLS. `true` (padrão) usa as CAs do sistema, `false` desativa a verificação (inseguro), ou uma string com o caminho para um pacote de CA personalizado (PEM) |
| `client_cert` | string ou list | HTTP | Certificado de cliente mTLS. String = caminho para um arquivo PEM contendo certificado + chave. Lista `[cert, key]` = arquivos separados. Lista `[cert, key, password]` = chave criptografada |
| `client_key` | string | HTTP | Caminho para a chave privada do cliente, quando `client_cert` é uma string e a chave está em um arquivo separado |
| `enabled` | bool | ambos | Ignora o servidor completamente quando `false` |
| `timeout` | number | ambos | Timeout de chamada de ferramenta em segundos (padrão: `300`) |
| `connect_timeout` | number | ambos | Timeout de conexão inicial em segundos (padrão: `60`) |
| `supports_parallel_tool_calls` | bool | ambos | Permite que ferramentas deste servidor sejam executadas em paralelo |
| `skip_preflight` | bool | HTTP | Ignora a verificação fail-fast de content-type para endpoints Streamable HTTP válidos cujo HEAD/GET responde com um content-type não-MCP (padrão: `false`) |
| `tools` | mapping | ambos | Política de filtragem e ferramentas utilitárias |
| `auth` | string | HTTP | Método de autenticação. Defina como `oauth` para ativar OAuth 2.1 com PKCE |
| `sampling` | mapping | ambos | Política de requisições de LLM iniciadas pelo servidor (veja o guia de MCP) |

## Chaves da política `tools` {#tools-policy-keys}

| Chave | Tipo | Significado |
|---|---|---|
| `include` | string ou list | Lista de permissão de ferramentas MCP nativas do servidor |
| `exclude` | string ou list | Lista de bloqueio de ferramentas MCP nativas do servidor |
| `resources` | tipo booleano | Ativa/desativa `list_resources` + `read_resource` |
| `prompts` | tipo booleano | Ativa/desativa `list_prompts` + `get_prompt` |

## Semântica de filtragem {#filtering-semantics}

### `include` {#include}

Se `include` estiver definido, apenas essas ferramentas MCP nativas do servidor são registradas.

```yaml
tools:
  include: [create_issue, list_issues]
```

### `exclude` {#exclude}

Se `exclude` estiver definido e `include` não, toda ferramenta MCP nativa do servidor é registrada, exceto essas.

```yaml
tools:
  exclude: [delete_customer]
```

### Precedência {#precedence}

Se ambos estiverem definidos, `include` prevalece.

```yaml
tools:
  include: [create_issue]
  exclude: [create_issue, delete_issue]
```

Resultado:
- `create_issue` ainda é permitida
- `delete_issue` é ignorada porque `include` tem precedência

## Política de ferramentas utilitárias {#utility-tool-policy}

O Hermes pode registrar estes wrappers utilitários por servidor MCP:

Resources:
- `list_resources`
- `read_resource`

Prompts:
- `list_prompts`
- `get_prompt`

### Desativar resources {#disable-resources}

```yaml
tools:
  resources: false
```

### Desativar prompts {#disable-prompts}

```yaml
tools:
  prompts: false
```

### Registro consciente de capacidades {#capability-aware-registration}

Mesmo quando `resources: true` ou `prompts: true`, o Hermes só registra essas ferramentas utilitárias se a sessão MCP realmente expuser a capacidade correspondente.

Então isto é normal:
- você ativa prompts
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
- nenhum registro de ferramentas
- a configuração permanece disponível para reutilização posterior

## Comportamento com resultado vazio {#empty-result-behavior}

Se a filtragem remover todas as ferramentas nativas do servidor e nenhuma ferramenta utilitária for registrada, o Hermes não cria um toolset MCP vazio em runtime para aquele servidor.

## Exemplos de configuração {#example-configs}

### Lista de permissão segura do GitHub {#safe-github-allowlist}

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

### Lista de bloqueio do Stripe {#stripe-blacklist}

```yaml
mcp_servers:
  stripe:
    url: "https://mcp.stripe.com"
    headers:
      Authorization: "Bearer ***"
    tools:
      exclude: [delete_customer, refund_payment]
```

### Servidor de documentos somente resources {#resource-only-docs-server}

```yaml
mcp_servers:
  docs:
    url: "https://mcp.docs.example.com"
    tools:
      include: []
      resources: true
      prompts: false
```

### Certificado de cliente TLS (mTLS) {#tls-client-certificate-mtls}

Para servidores HTTP/SSE que exigem um certificado de cliente, defina `client_cert` (e opcionalmente `client_key`):

```yaml
mcp_servers:
  # Certificado + chave combinados em um único arquivo PEM
  internal_api:
    url: "https://mcp.internal.example.com/mcp"
    client_cert: "~/secrets/mcp-client.pem"

  # Arquivos de certificado e chave separados
  partner_api:
    url: "https://mcp.partner.example.com/mcp"
    client_cert: "~/secrets/client.crt"
    client_key: "~/secrets/client.key"

  # Chave criptografada com senha (forma de lista de 3 elementos)
  bank_api:
    url: "https://mcp.bank.example.com/mcp"
    client_cert: ["~/secrets/client.crt", "~/secrets/client.key", "my-passphrase"]

  # Pacote de CA personalizado (CA privada / servidor autoassinado)
  lab_api:
    url: "https://mcp.lab.local/mcp"
    ssl_verify: "~/secrets/lab-ca.pem"
    client_cert: "~/secrets/lab-client.pem"
```

Notas:
- Caminhos suportam expansão de `~`. Arquivos ausentes falham rapidamente no momento da conexão com uma mensagem de erro específica do servidor.
- `ssl_verify: false` desativa totalmente a verificação de certificado do servidor. Não use isso com serviços reais.
- Funciona tanto em transporte Streamable HTTP quanto SSE.

## Recarregando a configuração {#reloading-config}

Depois de alterar a configuração MCP, recarregue os servidores com:

```text
/reload-mcp
```

## Nomeação de ferramentas {#tool-naming}

Ferramentas MCP nativas do servidor se tornam:

```text
mcp_<server>_<tool>
```

Exemplos:
- `mcp_github_create_issue`
- `mcp_filesystem_read_file`
- `mcp_my_api_query_data`

Ferramentas utilitárias seguem o mesmo padrão de prefixo:
- `mcp_<server>_list_resources`
- `mcp_<server>_read_resource`
- `mcp_<server>_list_prompts`
- `mcp_<server>_get_prompt`

### Sanitização de nomes {#name-sanitization}

Hifens (`-`) e pontos (`.`) tanto em nomes de servidor quanto em nomes de ferramenta são substituídos por underscores antes do registro. Isso garante que os nomes de ferramenta sejam identificadores válidos para as APIs de function-calling dos LLMs.

Por exemplo, um servidor chamado `my-api` que expõe uma ferramenta chamada `list-items.v2` se torna:

```text
mcp_my_api_list_items_v2
```

Tenha isso em mente ao escrever filtros `include` / `exclude` — use o nome **original** da ferramenta MCP (com hifens/pontos), não a versão sanitizada.

## Autenticação OAuth 2.1 {#oauth-21-authentication}

Para servidores HTTP que exigem OAuth, defina `auth: oauth` na entrada do servidor:

```yaml
mcp_servers:
  protected_api:
    url: "https://mcp.example.com/mcp"
    auth: oauth
```

Comportamento:
- O Hermes usa o fluxo OAuth 2.1 PKCE do SDK MCP (descoberta de metadados, registro dinâmico de cliente, troca e renovação de token)
- Na primeira conexão, uma janela do navegador é aberta para autorização
- Os tokens são persistidos em `~/.hermes/mcp-tokens/<server>.json` e reutilizados entre sessões
- A renovação de token é automática; a reautorização só ocorre quando a renovação falha
- Aplica-se apenas ao transporte HTTP/StreamableHTTP (servidores baseados em `url`)
