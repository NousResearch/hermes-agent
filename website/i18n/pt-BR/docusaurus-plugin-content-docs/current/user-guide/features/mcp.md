---
sidebar_position: 4
title: "MCP (Model Context Protocol)"
description: "Conecte o Hermes Agent a servidores de ferramentas externos via MCP — e controle exatamente quais ferramentas MCP o Hermes carrega"
---

# MCP (Model Context Protocol)

O MCP permite que o Hermes Agent se conecte a servidores de ferramentas externos para que o agente possa usar ferramentas que ficam fora do próprio Hermes — GitHub, bancos de dados, sistemas de arquivos, stacks de navegador, APIs internas e muito mais.

Se você já quis que o Hermes usasse uma ferramenta que já existe em outro lugar, o MCP costuma ser a forma mais limpa de fazer isso.

## O que o MCP oferece {#what-mcp-gives-you}

- Acesso a ecossistemas de ferramentas externos sem precisar escrever uma ferramenta nativa do Hermes primeiro
- Servidores stdio locais e servidores MCP HTTP remotos na mesma configuração
- Descoberta automática de ferramentas e registro na inicialização
- Wrappers utilitários para recursos e prompts MCP quando o servidor suporta
- Filtragem por servidor para expor apenas as ferramentas MCP que você realmente quer que o Hermes veja

## Início rápido {#quick-start}

1. O suporte a MCP vem com a instalação padrão — nenhum passo extra necessário.

2. Adicione um servidor MCP em `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"]
```

3. Inicie o Hermes:

```bash
hermes chat
```

4. Peça ao Hermes para usar a capacidade fornecida pelo MCP.

Por exemplo:

```text
List the files in /home/user/projects and summarize the repo structure.
```

O Hermes descobrirá as ferramentas do servidor MCP e as usará como qualquer outra ferramenta.

## Catálogo: instalação com um clique para MCPs aprovados pela Nous {#catalog-one-click-install-for-nous-approved-mcps}

O Hermes inclui um catálogo curado de servidores MCP que a equipe da Nous revisou
e incorporou. Eles ficam desabilitados por padrão — instale apenas o que você
realmente quer.

```bash
hermes mcp                # interactive picker (default)
hermes mcp catalog        # plain-text list, scriptable
hermes mcp install n8n    # install a catalog entry by name
```

O seletor mostra cada entrada com seu status atual:

```
n8n          available              Manage and inspect n8n workflows from Hermes
linear       enabled                Linear issue/project management (remote OAuth)
github       installed (disabled)   GitHub repo + PR tools
```

Pressione `Enter` em uma linha para instalar (e percorrer as credenciais necessárias),
habilitar, desabilitar ou desinstalar. As entradas do catálogo ficam em
`optional-mcps/` no repositório hermes-agent — a presença nesse diretório significa
aprovação da Nous. Não há camada de submissão da comunidade; as entradas são adicionadas
por merge de PR.

As entradas do catálogo podem exigir:

- **API key** — o Hermes solicita na instalação e grava o valor em
  `~/.hermes/.env`. Valores não secretos (base URLs) vão para o mesmo arquivo.
- **OAuth** (MCP remoto) — escrito como `auth: oauth` na sua config; o cliente MCP
  abre o navegador na primeira conexão.
- **OAuth** (provedor terceiro como Google/GitHub) — o Hermes aponta para
  `hermes auth <provider>` se você ainda não tiver autenticado.

### Seleção de ferramentas na instalação

Depois que as credenciais são configuradas, o Hermes sonda o servidor MCP para listar cada
ferramenta que ele expõe e apresenta uma checklist:

```
Select tools for 'linear' (SPACE toggle, ENTER confirm)
  [x] find_issues       Find issues matching a query
  [x] get_issue         Get a single issue
  [x] create_issue      Create a new issue
  [ ] delete_workspace  Delete a Linear workspace
  ...
```

As linhas pré-marcadas vêm de:

1. **Sua seleção anterior** se você já instalou essa entrada antes (reinstalações
   preservam o que você tinha — os padrões do manifest não sobrescrevem)
2. **O `tools.default_enabled` do manifest** se a entrada declara um (algumas
   entradas do catálogo pré-eliminam ferramentas mutáveis ou raramente úteis)
3. **Tudo** se nenhum dos casos se aplica

Confirme a checklist com ENTER. Apenas as ferramentas marcadas entram em
`mcp_servers.<name>.tools.include`. Se você selecionar tudo, nenhum filtro é
escrito (formato de config mais limpo, comportamento idêntico).

**Se a sonda falhar** (servidor inacessível, OAuth ainda não concluído,
serviço de suporte não em execução), a instalação ainda tem sucesso: o
`tools.default_enabled` do manifest é aplicado diretamente (se declarado), ou nenhum filtro é
escrito (se não). Execute `hermes mcp configure <name>` novamente quando o servidor estiver
acessível para refinar.

### Modelo de confiança

Instalar uma entrada do catálogo executa o que o manifest especifica — `git clone`,
os comandos `bootstrap` da entrada (`pip install`, `npm install`, etc.) e,
por fim, o próprio código do servidor MCP. Os manifests são controlados por revisão de PR no
repositório hermes-agent, então a Nous revisou cada entrada antes de publicar —
**mas você ainda deve ler o manifest antes de instalar**, especialmente o
repositório do campo `source:`, os comandos `install.bootstrap:` e qualquer
invocação `transport.command:`.

Os manifests ficam em
[`optional-mcps/<name>/manifest.yaml`](https://github.com/NousResearch/hermes-agent/tree/main/optional-mcps)
no GitHub. O seletor também imprime a URL `source:` do manifest na instalação
para que você possa verificar rapidamente o repositório upstream. A página MCP do
dashboard web expõe os mesmos detalhes por entrada do catálogo — transporte, tipo de auth,
URL do endpoint (HTTP) ou command + args (stdio), source/ref git da instalação e
comandos bootstrap, e notas de setup — com o `source:` renderizado como um
link clicável, para você inspecionar exatamente a que uma entrada se conecta ou o que executa
antes de clicar em Install.

### Compatibilidade de versão do manifest

Os manifests fixam um `manifest_version`. O catálogo é compatível para frente: se um
PR adiciona uma entrada com `manifest_version` mais novo do que o Hermes instalado
entende, o seletor exibirá um aviso (`⚠ '<name>' requires a newer
Hermes`) para essa entrada em vez de ocultá-la silenciosamente. Execute `hermes update`
para instalar o Hermes mais recente quando vir isso.

### Substituição `${ENV_VAR}` em runtime

Dentro de `transport.command`, `transport.args`, `transport.url`
e `headers` de uma entrada, placeholders `${VAR}` são resolvidos no momento da conexão do servidor
a partir de variáveis de ambiente (que incluem tudo em `~/.hermes/.env`).
Isso é útil quando uma entrada do catálogo quer referenciar um valor que o usuário
configurou em outro lugar — por exemplo `${HOME}/foo` ou `${MY_PROVIDER_TOKEN}`.

Note que isso é distinto de `${INSTALL_DIR}` nos manifests do catálogo, que é
substituído na instalação pelo caminho para o qual o catálogo clonou o repositório da entrada.

### Atualizando a seleção de ferramentas depois

```bash
hermes mcp configure linear
```

Reabre a mesma checklist com sua seleção atual pré-marcada. Use isso
quando quiser mais ferramentas habilitadas, ou quando o servidor adicionou novas ferramentas que
você quer optar por usar.

### Atualizando o manifest do catálogo

Os MCPs nunca são atualizados automaticamente. Execute `hermes mcp install <name>` novamente para atualizar
depois de uma atualização do Hermes se a versão do manifest mudou.

Para adicionar um MCP ao catálogo, abra um PR em
[`optional-mcps/`](https://github.com/NousResearch/hermes-agent/tree/main/optional-mcps).

## Dois tipos de servidores MCP {#two-kinds-of-mcp-servers}

### Servidores stdio

Servidores stdio rodam como subprocessos locais e se comunicam via stdin/stdout.

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
```

Use servidores stdio quando:
- o servidor está instalado localmente
- você quer acesso de baixa latência a recursos locais
- você está seguindo a documentação do servidor MCP que mostra `command`, `args` e `env`

### Servidores HTTP

Servidores MCP HTTP são endpoints remotos aos quais o Hermes se conecta diretamente.

```yaml
mcp_servers:
  remote_api:
    url: "https://mcp.example.com/mcp"
    headers:
      Authorization: "Bearer ***"
```

Use servidores HTTP quando:
- o servidor MCP está hospedado em outro lugar
- sua organização expõe endpoints MCP internos
- você não quer que o Hermes crie um subprocesso local para essa integração

### Servidores HTTP autenticados com OAuth

A maioria dos servidores MCP hospedados (Linear, Sentry, Atlassian, Asana, Figma, Stripe, …) exige OAuth 2.1 em vez de um bearer token estático. Defina `auth: oauth` e o Hermes cuida de discovery, registro dinâmico de cliente, PKCE, troca de token, refresh e step-up auth via o MCP Python SDK.

```yaml
mcp_servers:
  linear:
    url: "https://mcp.linear.app/mcp"
    auth: oauth
```

Na primeira conexão, o Hermes imprime uma URL de autorização, abre seu navegador quando possível e aguarda o callback OAuth em uma porta loopback local. Os tokens ficam em cache em `~/.hermes/mcp-tokens/<server>.json` com permissões 0o600; execuções subsequentes os reutilizam silenciosamente até o refresh falhar.

**Hosts remotos / headless.** Quando o Hermes roda em uma máquina diferente do seu navegador, o callback loopback não alcança seu laptop. Duas formas de concluir o fluxo:

- **Paste-back (sem setup):** em um terminal interativo o Hermes imprime "Or paste the redirect URL here…" junto com a URL de autorização. Abra a URL no navegador, aprove, copie a URL completa em que o navegador termina (o redirect mostrará um erro de conexão — isso é esperado), cole no prompt. Strings de query bare `?code=…&state=…` também funcionam.
- **Port forward SSH:** `ssh -N -L <port>:127.0.0.1:<port> user@host` em um terminal separado, depois deixe o fluxo de redirect seguir normalmente.
- **Callback proxied (`redirect_uri`):** quando um endpoint HTTPS público encaminha para o host (por exemplo, um Tailscale Funnel ou reverse proxy apontado para a porta de callback), defina `oauth.redirect_uri` e o redirect do navegador alcança o Hermes por conta própria — sem túnel ou paste:

```yaml
mcp_servers:
  myserver:
    url: "https://mcp.example.com/mcp"
    auth: oauth
    oauth:
      redirect_port: 8765                                # fixed port for the proxy to target
      redirect_uri: "https://oauth.example.ts.net/callback"
```

Para gateways totalmente headless (bot de mensagens, sem terminal interativo algum), a skill opcional [`mcp-oauth-remote-gateway`](../skills/optional/mcp/mcp-mcp-oauth-remote-gateway.md) guia o agente na conclusão manual do fluxo e na gravação dos tokens onde o Hermes espera.

**Armadilha — WAF rejeita redirect URIs `127.0.0.1`.** Alguns provedores colocam um WAF na frente do authorization server que retorna 403 em qualquer requisição de autorização cuja query string contenha um `127.0.0.1` literal (o AWS API Gateway do Reclaim.ai é um exemplo conhecido — toda tentativa retorna `{"message":"Forbidden"}` antes de chegar ao app OAuth). Defina `oauth.redirect_host: localhost` para usar `http://localhost:<port>/callback` em vez disso; o listener de callback continua vinculado a `127.0.0.1` de qualquer forma.

Veja [OAuth over SSH / Remote Hosts](../../guides/oauth-over-ssh.md#mcp-servers) para o walkthrough completo, incluindo servidores sem DCR (por exemplo, Slack), `client_id`/`client_secret` pré-registrados, customização de escopos e re-auth via `hermes mcp login <server>`.

**Armadilha — provedores que não suportam registro automático (Google Drive, Atlassian).** Alguns servidores rejeitam a etapa de registro dinâmico de cliente (RFC 7591) da qual o `auth: oauth` bare depende — o servidor oficial do Google Drive (`https://drivemcp.googleapis.com/mcp/v1`) retorna `400 Bad Request`, então nenhum cliente OAuth é criado e nenhum token é adquirido. O sintoma é sutil: esses servidores também servem `tools/list` *sem* auth, então `hermes mcp login` pode listar as ferramentas e parecer que funcionou, mas toda chamada real de ferramenta depois dá timeout. `hermes mcp login` agora detecta isso (verifica se um token realmente foi gravado em disco) e diz para você fornecer seu próprio cliente OAuth. Crie um no console do provedor e adicione à config:

```yaml
mcp_servers:
  googledrive:
    url: "https://drivemcp.googleapis.com/mcp/v1"
    auth: oauth
    oauth:
      client_id: "<your-oauth-client-id>"
      client_secret: "<your-oauth-client-secret>"
```

Depois execute `hermes mcp login googledrive` — com o cliente pré-registrado, o Hermes pula o registro e executa o fluxo normal de autorização no navegador.

**Armadilha — corrida de auto-reload da config.** Quando você edita `~/.hermes/config.yaml` de dentro de uma sessão Hermes em execução, o CLI recarrega automaticamente as conexões MCP com timeout de 30s. Isso não basta para um fluxo OAuth interativo. Adicione a entrada, depois execute `hermes mcp login <server>` em um terminal novo — ele aguarda os 5 minutos completos para você concluir a auth.

## mTLS / certificados de cliente {#mtls-client-certificates}

Servidores MCP HTTP remotos que exigem mutual TLS (autenticação por certificado de cliente) são suportados via `client_cert` / `client_key`. O Hermes passa o certificado resolvido ao cliente HTTP subjacente para o handshake TLS.

`client_cert` aceita três formas:

- **Um único caminho PEM combinado** — um arquivo com certificado e chave privada:

```yaml
mcp_servers:
  internal_api:
    url: "https://mcp.internal.example.com/mcp"
    client_cert: "~/.certs/mcp-client.pem"
```

- **Uma tupla 2-elementos `[cert, key]`** — certificado e chave em arquivos separados (equivalente a definir `client_cert` + `client_key`):

```yaml
mcp_servers:
  internal_api:
    url: "https://mcp.internal.example.com/mcp"
    client_cert: ["~/.certs/mcp-client.crt", "~/.certs/mcp-client.key"]
```

- **Uma tupla 3-elementos `[cert, key, password]`** — quando a chave privada é criptografada, o terceiro elemento é a passphrase da chave:

```yaml
mcp_servers:
  internal_api:
    url: "https://mcp.internal.example.com/mcp"
    client_cert: ["~/.certs/mcp-client.crt", "~/.certs/mcp-client.key", "${MCP_KEY_PASSWORD}"]
```

Você também pode manter certificado e chave totalmente separados via `client_cert` (PEM combinado) mais um `client_key` explícito. Caminhos suportam expansão de `~`; um arquivo ausente gera um erro claro, com escopo do servidor, em vez de uma falha opaca de handshake TLS.

## Referência básica de configuração {#basic-configuration-reference}

O Hermes lê a config MCP de `~/.hermes/config.yaml` em `mcp_servers`.

### Chaves comuns

| Key | Type | Meaning |
|---|---|---|
| `command` | string | Executável para um servidor MCP stdio |
| `args` | list | Argumentos para o servidor stdio |
| `env` | mapping | Variáveis de ambiente passadas ao servidor stdio |
| `url` | string | Endpoint MCP HTTP |
| `headers` | mapping | Headers HTTP para servidores remotos |
| `client_cert` | string \| list | Certificado de cliente para mTLS — um caminho PEM combinado, ou `[cert, key]` / `[cert, key, password]` |
| `client_key` | string | Caminho PEM da chave privada do cliente (quando separada de `client_cert`) |
| `timeout` | number | Timeout de chamada de ferramenta |
| `connect_timeout` | number | Timeout de conexão inicial (também limita o handshake MCP `initialize`) |
| `idle_timeout_seconds` | number | Recicla um servidor stdio após tantos segundos sem chamada de ferramenta (`0` = nunca, padrão). O servidor reinicia transparentemente na próxima chamada de ferramenta. |
| `max_lifetime_seconds` | number | Recicla um servidor stdio após essa idade total (`0` = nunca, padrão). Reinicia transparentemente no próximo uso. |
| `enabled` | bool | Se `false`, o Hermes ignora o servidor completamente |
| `supports_parallel_tool_calls` | bool | Se `true`, ferramentas desse servidor podem rodar em paralelo |
| `tools` | mapping | Filtragem de ferramentas e política de utilitários por servidor |

### Exemplo stdio mínimo

```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
```

### Reciclando servidores stdio que consomem muita memória

Servidores MCP baseados em navegador (por exemplo, `@playwright/mcp`) mantêm um Chromium
completo residente após a primeira chamada de ferramenta — centenas de MB que nunca são
liberados. Opte por reciclagem automática e o servidor é encerrado após
o limite idle/lifetime, depois reiniciado transparentemente na próxima vez que uma de
suas ferramentas for chamada (suas ferramentas permanecem registradas o tempo todo):

```yaml
mcp_servers:
  playwright:
    command: "npx"
    args: ["-y", "@playwright/mcp@latest", "--headless"]
    idle_timeout_seconds: 900     # recycle after 15 min without a tool call
    max_lifetime_seconds: 86400   # and at least once a day regardless
```

### Exemplo HTTP mínimo

```yaml
mcp_servers:
  company_api:
    url: "https://mcp.internal.example.com"
    headers:
      Authorization: "Bearer ***"
```

## Presets built-in {#built-in-presets}

Para servidores MCP conhecidos, `hermes mcp add` aceita a flag `--preset` que preenche os detalhes de transporte para você não precisar procurar command e args. O preset só fornece padrões — qualquer outra coisa (env vars, headers, filtragem) que você passar na mesma linha de comando ainda prevalece.

| Preset | What it wires up |
|---|---|
| `codex` | O servidor MCP do Codex CLI (`codex mcp-server` over stdio). Requer o CLI `codex` no PATH. |

```bash
# Add Codex CLI as an MCP server in one line
hermes mcp add codex --preset codex
```

Isso grava o equivalente a:

```yaml
mcp_servers:
  codex:
    command: "codex"
    args: ["mcp-server"]
```

Você pode escolher qualquer nome local (`hermes mcp add my-codex --preset codex` funciona); o preset só fornece os padrões de `command`/`args`.

## Como o Hermes registra ferramentas MCP {#how-hermes-registers-mcp-tools}

O Hermes prefixa ferramentas MCP para não colidirem com nomes built-in:

```text
mcp_<server_name>_<tool_name>
```

Exemplos:

| Server | MCP tool | Registered name |
|---|---|---|
| `filesystem` | `read_file` | `mcp_filesystem_read_file` |
| `github` | `create-issue` | `mcp_github_create_issue` |
| `my-api` | `query.data` | `mcp_my_api_query_data` |

Na prática, você geralmente não precisa chamar o nome prefixado manualmente — o Hermes vê a ferramenta e a escolhe durante o raciocínio normal.

## Ferramentas utilitárias MCP {#mcp-utility-tools}

Quando suportado, o Hermes também registra ferramentas utilitárias em torno de recursos e prompts MCP:

- `list_resources`
- `read_resource`
- `list_prompts`
- `get_prompt`

Elas são registradas por servidor com o mesmo padrão de prefixo, por exemplo:

- `mcp_github_list_resources`
- `mcp_github_get_prompt`

### Importante

Essas ferramentas utilitárias agora são conscientes de capacidade:
- o Hermes só registra utilitários de recursos se a sessão MCP realmente suporta operações de recursos
- o Hermes só registra utilitários de prompts se a sessão MCP realmente suporta operações de prompts

Então um servidor que expõe ferramentas invocáveis mas sem recursos/prompts não receberá esses wrappers extras.

## Filtragem por servidor {#per-server-filtering}

Você pode controlar quais ferramentas cada servidor MCP contribui ao Hermes, permitindo gerenciamento fino do seu namespace de ferramentas.

### Desabilitar um servidor completamente

```yaml
mcp_servers:
  legacy:
    url: "https://mcp.legacy.internal"
    enabled: false
```

Se `enabled: false`, o Hermes ignora o servidor completamente e nem tenta conectar.

### Whitelist de ferramentas do servidor

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
    tools:
      include: [create_issue, list_issues]
```

Apenas essas ferramentas do servidor MCP são registradas.

### Blacklist de ferramentas do servidor

```yaml
mcp_servers:
  stripe:
    url: "https://mcp.stripe.com"
    tools:
      exclude: [delete_customer]
```

Todas as ferramentas do servidor são registradas exceto as excluídas.

### Regra de precedência

Se ambos estiverem presentes:

```yaml
tools:
  include: [create_issue]
  exclude: [create_issue, delete_issue]
```

`include` prevalece.

### Filtrar ferramentas utilitárias também

Você também pode desabilitar separadamente os wrappers utilitários adicionados pelo Hermes:

```yaml
mcp_servers:
  docs:
    url: "https://mcp.docs.example.com"
    tools:
      prompts: false
      resources: false
```

Isso significa:
- `tools.resources: false` desabilita `list_resources` e `read_resource`
- `tools.prompts: false` desabilita `list_prompts` e `get_prompt`

### Exemplo completo

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
    tools:
      include: [create_issue, list_issues, search_code]
      prompts: false

  stripe:
    url: "https://mcp.stripe.com"
    headers:
      Authorization: "Bearer ***"
    tools:
      exclude: [delete_customer]
      resources: false

  legacy:
    url: "https://mcp.legacy.internal"
    enabled: false
```

## O que acontece se tudo for filtrado? {#what-happens-if-everything-is-filtered-out}

Se sua config filtrar todas as ferramentas invocáveis e desabilitar ou omitir todos os utilitários suportados, o Hermes não cria um toolset MCP runtime vazio para esse servidor.

Isso mantém a lista de ferramentas limpa.

## Comportamento em runtime {#runtime-behavior}

### Momento da descoberta

O Hermes descobre servidores MCP na inicialização e registra suas ferramentas no registro normal de ferramentas.

### Descoberta dinâmica de ferramentas

Servidores MCP podem notificar o Hermes quando suas ferramentas disponíveis mudam em runtime enviando uma notificação `notifications/tools/list_changed`. Quando o Hermes recebe essa notificação, ele busca automaticamente novamente a lista de ferramentas do servidor e atualiza o registro — sem `/reload-mcp` manual necessário.

Isso é útil para servidores MCP cujas capacidades mudam dinamicamente (por exemplo, um servidor que adiciona ferramentas quando um novo schema de banco de dados é carregado, ou remove ferramentas quando um serviço fica offline).

O refresh é protegido por lock para que notificações em rajada do mesmo servidor não causem refreshes sobrepostos. Notificações de mudança de prompts e recursos (`prompts/list_changed`, `resources/list_changed`) são recebidas mas ainda não são processadas.

### Recarregando

Se você alterar a config MCP, use:

```text
/reload-mcp
```

Isso recarrega servidores MCP da config e atualiza a lista de ferramentas disponíveis. Para mudanças de ferramentas em runtime enviadas pelo próprio servidor, veja [Dynamic Tool Discovery](#dynamic-tool-discovery) acima.

### Toolsets

Cada servidor MCP configurado também cria um toolset runtime quando contribui com pelo menos uma ferramenta registrada:

```text
mcp-<server>
```

Isso torna os servidores MCP mais fáceis de raciocinar no nível de toolset.

## Modelo de segurança {#security-model}

### Filtragem de env stdio

Para servidores stdio, o Hermes não repassa cegamente todo o ambiente do seu shell.

Apenas `env` configurado explicitamente mais uma baseline segura são repassados. Isso reduz vazamento acidental de segredos.

### Controle de exposição no nível da config

O novo suporte a filtragem também é um controle de segurança:
- desabilite ferramentas perigosas que você não quer que o modelo veja
- exponha apenas uma whitelist mínima para um servidor sensível
- desabilite wrappers de recursos/prompts quando não quiser essa superfície exposta

## Casos de uso de exemplo {#example-use-cases}

### Servidor GitHub com superfície mínima de gestão de issues

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
    tools:
      include: [list_issues, create_issue, update_issue]
      prompts: false
      resources: false
```

Use assim:

```text
Show me open issues labeled bug, then draft a new issue for the flaky MCP reconnection behavior.
```

### Servidor Stripe com ações perigosas removidas

```yaml
mcp_servers:
  stripe:
    url: "https://mcp.stripe.com"
    headers:
      Authorization: "Bearer ***"
    tools:
      exclude: [delete_customer, refund_payment]
```

Use assim:

```text
Look up the last 10 failed payments and summarize common failure reasons.
```

### Servidor filesystem para uma única raiz de projeto

```yaml
mcp_servers:
  project_fs:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/my-project"]
```

Use assim:

```text
Inspect the project root and explain the directory layout.
```

## Solução de problemas {#troubleshooting}

### Servidor MCP não conecta

Verifique:

```bash
# Verify MCP deps are installed (already included in standard install)
cd ~/.hermes/hermes-agent && uv pip install -e ".[mcp]"

node --version
npx --version
```

Depois verifique sua config e reinicie o Hermes.

### Ferramentas não aparecem

Possíveis causas:
- o servidor falhou ao conectar
- a descoberta falhou
- sua config de filtro excluiu as ferramentas
- a capacidade utilitária não existe nesse servidor
- o servidor está desabilitado com `enabled: false`

Se você está filtrando intencionalmente, isso é esperado.

### Por que utilitários de recursos ou prompts não apareceram?

Porque o Hermes agora só registra esses wrappers quando ambos são verdadeiros:
1. sua config permite
2. a sessão do servidor realmente suporta a capacidade

Isso é intencional e mantém a lista de ferramentas honesta.

## Chamadas de ferramentas em paralelo {#parallel-tool-calls}

Por padrão, ferramentas MCP rodam sequencialmente — uma de cada vez. Se seu servidor MCP expõe ferramentas seguras para rodar em paralelo (por exemplo, consultas somente leitura, chamadas de API independentes), você pode optar por execução paralela:

```yaml
mcp_servers:
  docs:
    command: "docs-server"
    supports_parallel_tool_calls: true
```

Quando `supports_parallel_tool_calls` é `true`, o Hermes pode executar várias ferramentas desse servidor ao mesmo tempo dentro de um único lote de chamadas de ferramenta, assim como faz para ferramentas built-in somente leitura (web_search, read_file, etc.).

:::caution
Habilite chamadas paralelas apenas para servidores MCP cujas ferramentas são seguras para rodar ao mesmo tempo. Se ferramentas leem e escrevem estado compartilhado, arquivos, bancos de dados ou recursos externos, revise as condições de corrida leitura/escrita antes de habilitar essa configuração.
:::

## Suporte a sampling MCP {#mcp-sampling-support}

Servidores MCP podem solicitar inferência LLM do Hermes via o protocolo `sampling/createMessage`. Isso permite que um servidor MCP peça ao Hermes para gerar texto em seu nome — útil para servidores que precisam de capacidades LLM mas não têm acesso próprio a modelos.

O sampling está **habilitado por padrão** para todos os servidores MCP (quando o MCP SDK suporta). Configure por servidor sob a chave `sampling`:

```yaml
mcp_servers:
  my_server:
    command: "my-mcp-server"
    sampling:
      enabled: true            # Enable sampling (default: true)
      model: "openai/gpt-4o"  # Override model for sampling requests (optional)
      max_tokens_cap: 4096     # Max tokens per sampling response (default: 4096)
      timeout: 30              # Timeout in seconds per request (default: 30)
      max_rpm: 10              # Rate limit: max requests per minute (default: 10)
      max_tool_rounds: 5       # Max tool-use rounds in sampling loops (default: 5)
      allowed_models: []       # Allowlist of model names the server may request (empty = any)
      log_level: "info"        # Audit log level: debug, info, or warning (default: info)
```

O handler de sampling inclui um rate limiter de janela deslizante, timeouts por requisição e limites de profundidade de loop de ferramentas para evitar uso descontrolado. Métricas (contagem de requisições, erros, tokens usados) são rastreadas por instância de servidor.

Para desabilitar sampling para um servidor específico:

```yaml
mcp_servers:
  untrusted_server:
    url: "https://mcp.example.com"
    sampling:
      enabled: false
```

## Executando o Hermes como servidor MCP {#running-hermes-as-an-mcp-server}

Além de se conectar **a** servidores MCP, o Hermes também pode **ser** um servidor MCP. Isso permite que outros agentes compatíveis com MCP (Claude Code, Cursor, Codex ou qualquer cliente MCP) usem as capacidades de mensagens do Hermes — listar conversas, ler histórico de mensagens e enviar mensagens em todas as suas plataformas conectadas.

### Quando usar

- Você quer que Claude Code, Cursor ou outro agente de coding envie e leia mensagens Telegram/Discord/Slack pelo Hermes
- Você quer um único servidor MCP que faça ponte para todas as plataformas de mensagens conectadas do Hermes de uma vez
- Você já tem um gateway Hermes em execução com plataformas conectadas

### Início rápido

```bash
hermes mcp serve
```

Isso inicia um servidor MCP stdio. O cliente MCP (não você) gerencia o ciclo de vida do processo.

### Configuração do cliente MCP

Adicione o Hermes à config do seu cliente MCP. Por exemplo, no `~/.claude/claude_desktop_config.json` do Claude Code:

```json
{
  "mcpServers": {
    "hermes": {
      "command": "hermes",
      "args": ["mcp", "serve"]
    }
  }
}
```

Ou se você instalou o Hermes em um local específico:

```json
{
  "mcpServers": {
    "hermes": {
      "command": "/home/user/.hermes/hermes-agent/venv/bin/hermes",
      "args": ["mcp", "serve"]
    }
  }
}
```

### Ferramentas disponíveis

O servidor MCP expõe 10 ferramentas, correspondendo à superfície de ponte de canais do OpenClaw mais um navegador de canais específico do Hermes:

| Tool | Description |
|------|-------------|
| `conversations_list` | List active messaging conversations. Filter by platform or search by name. |
| `conversation_get` | Get detailed info about one conversation by session key. |
| `messages_read` | Read recent message history for a conversation. |
| `attachments_fetch` | Extract non-text attachments (images, media) from a specific message. |
| `events_poll` | Poll for new conversation events since a cursor position. |
| `events_wait` | Long-poll / block until the next event arrives (near-real-time). |
| `messages_send` | Send a message through a platform (e.g. `telegram:123456`, `discord:#general`). |
| `channels_list` | List available messaging targets across all platforms. |
| `permissions_list_open` | List pending approval requests observed during this bridge session. |
| `permissions_respond` | Allow or deny a pending approval request. |

### Sistema de eventos

O servidor MCP inclui uma ponte de eventos ao vivo que faz poll no banco de sessões do Hermes por novas mensagens. Isso dá aos clientes MCP consciência quase em tempo real de conversas recebidas:

```
# Poll for new events (non-blocking)
events_poll(after_cursor=0)

# Wait for next event (blocks up to timeout)
events_wait(after_cursor=42, timeout_ms=30000)
```

Tipos de evento: `message`, `approval_requested`, `approval_resolved`

A fila de eventos é em memória e inicia quando a ponte conecta. Mensagens mais antigas estão disponíveis via `messages_read`.

### Opções

```bash
hermes mcp serve              # Normal mode
hermes mcp serve --verbose    # Debug logging on stderr
```

### Como funciona

O servidor MCP lê dados de conversa diretamente do armazenamento de sessões do Hermes (`~/.hermes/sessions/sessions.json` e o banco SQLite). Uma thread em background faz poll no banco por novas mensagens e mantém uma fila de eventos em memória. Para enviar mensagens, usa o mesmo motor interno de envio (`tools/send_message_tool.py`) que alimenta entrega de cron e o CLI `hermes send`.

O gateway NÃO precisa estar em execução para operações de leitura (listar conversas, ler histórico, fazer poll de eventos). Ele PRECISA estar em execução para operações de envio, já que os adapters de plataforma precisam de conexões ativas.

### Limites atuais

- O `hermes mcp serve` embutido expõe um servidor MCP **somente stdio** hoje. Se você precisa de um servidor MCP HTTP, rode um adapter separado — ou, muito mais comum, use o lado **cliente** MCP do Hermes, que já fala stdio e HTTP (`url` + `headers` em `mcp_servers.yaml` / `config.yaml`; veja [HTTP servers](#http-servers) acima).
- Poll de eventos em intervalos de ~200ms via poll otimizado por mtime no DB (pula trabalho quando arquivos não mudaram)
- Ainda sem protocolo de push notification `claude/channel`
- Envios somente texto (sem envio de mídia/anexos via `messages_send`)

## Documentação relacionada {#related-docs}

- [Use MCP with Hermes](/guides/use-mcp-with-hermes)
- [CLI Commands](/reference/cli-commands)
- [Slash Commands](/reference/slash-commands)
- [FAQ](/reference/faq)
