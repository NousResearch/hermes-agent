---
sidebar_position: 4
title: "MCP (Model Context Protocol)"
description: "Conecte o Hermes Agent a servidores de ferramentas externos via MCP — e controle exatamente quais ferramentas MCP o Hermes carrega"
---

# MCP (Model Context Protocol) {#mcp-model-context-protocol}

MCP permite que o Hermes Agent se conecte a servidores de ferramentas externos para o agente usar ferramentas que vivem fora do próprio Hermes — GitHub, bancos de dados, sistemas de arquivos, stacks de browser, APIs internas e mais.

Se você já quis que o Hermes usasse uma ferramenta que já existe em outro lugar, MCP costuma ser a forma mais limpa de fazer isso.

:::tip Vindo do Claude Code?
O bloco `mcpServers` no seu `~/.claude.json` mapeia para `mcp_servers` no `config.yaml` do Hermes — e `hermes import-agent claude-code` migra isso (junto com skills e instruções) automaticamente. Veja [Importar de Outros Agentes](../import-from-other-agents.md).
:::

## O que MCP oferece {#what-mcp-gives-you}

- Acesso a ecossistemas de ferramentas externos sem escrever uma ferramenta nativa Hermes primeiro
- Servidores stdio locais e servidores MCP HTTP remotos na mesma config
- Descoberta e registro automático de ferramentas na inicialização
- Wrappers utilitários para resources e prompts MCP quando suportados pelo servidor
- Filtragem por servidor para expor só as ferramentas MCP que você quer que o Hermes veja

## Início rápido {#quick-start}

1. Suporte MCP vem com a instalação padrão — nenhum passo extra.

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

4. Peça ao Hermes para usar a capacidade backed por MCP.

Por exemplo:

```text
List the files in /home/user/projects and summarize the repo structure.
```

O Hermes descobrirá as ferramentas do servidor MCP e as usará como qualquer outra ferramenta.

## Catálogo: instalação com um clique para MCPs aprovados pela Nous {#catalog-one-click-install-for-nous-approved-mcps}

O Hermes inclui um catálogo curado de servidores MCP que a equipe Nous revisou
e mergeou. Estão desabilitados por padrão — instale só o que você realmente
quer.

```bash
hermes mcp                # seletor interativo (padrão)
hermes mcp catalog        # lista texto simples, scriptável
hermes mcp install n8n    # instalar entrada do catálogo por nome
```

O seletor mostra cada entrada com seu status atual:

```
n8n          available              Manage and inspect n8n workflows from Hermes
linear       enabled                Linear issue/project management (remote OAuth)
github       installed (disabled)   GitHub repo + PR tools
```

Pressione `Enter` numa linha para instalar (e percorrer credenciais necessárias),
habilitar, desabilitar ou desinstalar. Entradas do catálogo ficam em
`optional-mcps/` no repo hermes-agent — presença nesse diretório significa
aprovação Nous. Não há tier de submissão comunitária; entradas são adicionadas por
merge de PR.

Entradas do catálogo podem exigir:

- **API key** — Hermes pergunta na instalação e grava o valor em
  `~/.hermes/.env`. Valores não secretos (base URLs) vão no mesmo arquivo.
- **OAuth** (MCP remoto) — escrito como `auth: oauth` na sua config; o cliente MCP
  abre o navegador na primeira conexão.
- **OAuth** (provedor terceiro como Google/GitHub) — Hermes aponta para
  `hermes auth <provider>` se você ainda não autenticou.

### Seleção de ferramentas na instalação {#tool-selection-at-install-time}

Após credenciais configuradas, o Hermes sonda o servidor MCP para listar toda
ferramenta exposta e apresenta um checklist:

```
Select tools for 'linear' (SPACE toggle, ENTER confirm)
  [x] find_issues       Find issues matching a query
  [x] get_issue         Get a single issue
  [x] create_issue      Create a new issue
  [ ] delete_workspace  Delete a Linear workspace
  ...
```

As linhas pré-marcadas vêm de:

1. **Sua seleção anterior** se você já instalou esta entrada antes (reinstalações
   preservam o que tinha — defaults do manifest não sobrescrevem)
2. **`tools.default_enabled` do manifest** se a entrada declara um (algumas
   entradas pré-podam ferramentas mutantes ou raramente úteis)
3. **Tudo** se nenhum dos dois se aplica

Envie o checklist com ENTER. Só ferramentas marcadas entram em
`mcp_servers.<name>.tools.include`. Se selecionar tudo, nenhum filtro é
escrito (formato de config mais limpo, comportamento idêntico).

**Se a sonda falhar** (servidor inacessível, OAuth ainda incompleto,
serviço backing não rodando), a instalação ainda tem sucesso: o
`tools.default_enabled` do manifest é aplicado diretamente (se declarado), ou nenhum filtro é
escrito (se não). Re-rode `hermes mcp configure <name>` quando o servidor estiver
acessível para refinar.

### Modelo de confiança {#trust-model}

Instalar uma entrada do catálogo executa o que o manifest especifica — `git clone`,
comandos `bootstrap` da entrada (`pip install`, `npm install`, etc.), e
por fim o código do próprio servidor MCP. Manifests são gated por revisão de PR no
repo hermes-agent, então a Nous revisou cada entrada antes de shippar —
**mas você ainda deve ler o manifest antes de instalar**, especialmente o repositório do campo
`source:`, os comandos `install.bootstrap:` e qualquer
invocação `transport.command:`.

Manifests ficam em
[`optional-mcps/<name>/manifest.yaml`](https://github.com/NousResearch/hermes-agent/tree/main/optional-mcps)
no GitHub. O seletor também imprime a URL `source:` do manifest na instalação
para você verificar rapidamente o repo upstream. A página MCP do web dashboard
mostra o mesmo detalhe por entrada do catálogo — transporte, tipo de auth, a
URL do endpoint (HTTP) ou command + args (stdio), source/ref git da instalação e
comandos bootstrap, e notas de setup — com `source:` renderizado como link
clicável, para inspecionar exatamente a que uma entrada se conecta ou o que executa
antes de clicar Install.

### Compatibilidade de versão do manifest {#manifest-version-compatibility}

Manifests fixam um `manifest_version`. O catálogo é forward-compatible: se um
PR adiciona entrada com `manifest_version` mais novo que seu Hermes instalado
entende, o seletor mostra aviso (`⚠ '<name>' requires a newer
Hermes`) para essa entrada em vez de escondê-la silenciosamente. Rode `hermes update`
para instalar o Hermes mais recente quando vir isso.

### Substituição runtime `${ENV_VAR}` {#runtime-env_var-substitution}

Dentro de `transport.command`, `transport.args`, `transport.url`,
e `headers` de uma entrada, placeholders `${VAR}` resolvem na hora de conectar ao servidor
a partir de variáveis de ambiente (que incluem tudo em `~/.hermes/.env`).
Útil quando uma entrada do catálogo quer referenciar valor que o usuário
configurou em outro lugar — ex. `${HOME}/foo` ou `${MY_PROVIDER_TOKEN}`.

Variáveis de contexto estilo Cursor também são substituídas (case-sensitive):
`${userHome}` (diretório home), `${workspaceFolder}` (raiz do workspace da sessão),
`${workspaceFolderBasename}`, e `${pathSeparator}` / `${/}`
(separador de caminho do SO). Veja a
[referência de config MCP](/docs/reference/mcp-config-reference) para detalhes.

Note que isto é distinto de `${INSTALL_DIR}` em manifests do catálogo, que é
substituído na instalação pelo caminho onde o catálogo clonou o repo da entrada.

### Atualizando seleção de ferramentas depois {#updating-tool-selection-later}

```bash
hermes mcp configure linear
```

Reabre o mesmo checklist com sua seleção atual pré-marcada. Use quando quiser
mais ferramentas habilitadas, ou quando o servidor adicionou ferramentas novas que
quer opt-in.

### Atualizando o manifest do catálogo {#updating-the-catalog-manifest}

MCPs nunca são auto-atualizados. Re-rode `hermes mcp install <name>` para refresh
após update do Hermes se a versão do manifest mudou.

Para adicionar MCP ao catálogo, abra PR contra
[`optional-mcps/`](https://github.com/NousResearch/hermes-agent/tree/main/optional-mcps).

### Metadados de sugestão (`suggest:`) {#suggestion-metadata-suggest}

Um manifest pode declarar um bloco opcional `suggest:` com listas `keywords:` e/ou
`hosts:`. Superfícies de UI (atualmente o composer do app Desktop) usam isso para
oferecer uma pílula de um clique "Add &lt;server&gt;" quando seu rascunho menciona uma das
keywords como palavra completa, ou contém um link colado cujo hostname termina
com um dos sufixos de host. É puramente consultivo — installs ainda fluem
pelos mesmos caminhos validados de catálogo/config — e a maioria das entradas
hosted remotas (Atlassian, Sentry, Notion, Stripe, Vercel, Supabase e amigos)
o declara.

GitHub deliberadamente **não** está no catálogo: o MCP hosted dele exige que cada
cliente traga seu próprio OAuth app (registro dinâmico genérico de client é
rejeitado), e as skills bundled `github/*` do Hermes dirigindo a CLI `gh` são uma
integração mais capaz. No Desktop, menções ao GitHub em vez disso oferecem a
skill `github-auth` quando `gh` ainda não está logado.

## Dois tipos de servidores MCP {#two-kinds-of-mcp-servers}

### Servidores stdio {#stdio-servers}

Servidores stdio rodam como subprocessos locais e falam por stdin/stdout.

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
- segue docs de servidor MCP que mostram `command`, `args` e `env`

### Servidores HTTP {#http-servers}

Servidores MCP HTTP são endpoints remotos aos quais o Hermes conecta diretamente.

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
- você não quer que o Hermes dispare subprocesso local para essa integração

### Servidores HTTP com autenticação OAuth {#oauth-authenticated-http-servers}

A maioria dos servidores MCP hospedados (Linear, Sentry, Atlassian, Asana, Figma, Stripe, …) exige OAuth 2.1 em vez de bearer token estático. Defina `auth: oauth` e o Hermes cuida de discovery, identificação de cliente, PKCE, troca de token, refresh e step-up auth via MCP Python SDK.

O Hermes se identifica com um [Client ID Metadata Document](../../reference/mcp-config-reference.md#client-identification-cimd-and-dcr) em servidores que suportam um, e faz fallback para Dynamic Client Registration nos que não suportam. Ambos são automáticos; não há nada a configurar.

:::tip Figma remote MCP
O endpoint hospedado da Figma (`https://mcp.figma.com/mcp`) allowlista Dynamic Client Registration por **`client_name` exato** — `"Hermes Agent"` nu 403s, enquanto `"Claude Code"` e `"Codex"` passam. O Hermes auto-define `oauth.client_name: "Claude Code"` para `mcp.figma.com` para install/login funcionar sem truque especial:

```yaml
mcp_servers:
  figma:
    url: "https://mcp.figma.com/mcp"
    auth: oauth
```

Ou: `hermes mcp install figma`, depois `hermes mcp login figma`.
:::

```yaml
mcp_servers:
  linear:
    url: "https://mcp.linear.app/mcp"
    auth: oauth
```

Na primeira conexão, o Hermes imprime URL de autorização, abre o navegador quando possível e espera o callback OAuth numa porta loopback local. Tokens são cacheados em `~/.hermes/mcp-tokens/<server>.json` com perms 0o600; execuções seguintes reutilizam silenciosamente até refresh falhar.

**Hosts remotos / headless.** Quando o Hermes roda numa máquina diferente do seu navegador, o callback loopback não alcança seu laptop. Duas formas de completar o fluxo:

- **Paste-back (sem setup):** num terminal interativo o Hermes imprime "Or paste the redirect URL here…" junto com a URL de autorização. Abra a URL no navegador, aprove, copie a URL completa onde o navegador termina (o redirect mostrará erro de conexão — esperado), cole no prompt. Query strings nuas `?code=…&state=…` também funcionam.
- **Port forward SSH:** `ssh -N -L <port>:127.0.0.1:<port> user@host` num terminal separado, depois deixe o redirect fluir normalmente.
- **Callback proxied (`redirect_uri`):** quando um endpoint HTTPS público encaminha ao host (ex. Tailscale Funnel ou reverse proxy apontado à porta de callback), defina `oauth.redirect_uri` e o redirect do navegador alcança o Hermes sozinho — sem tunnel nem paste:

```yaml
mcp_servers:
  myserver:
    url: "https://mcp.example.com/mcp"
    auth: oauth
    oauth:
      redirect_port: 8765                                # porta fixa para o proxy mirar
      redirect_uri: "https://oauth.example.ts.net/callback"
```

Para gateways totalmente headless (bot de mensagens, sem terminal interativo), a skill opcional [`mcp-oauth-remote-gateway`](../skills/optional/mcp/mcp-mcp-oauth-remote-gateway.md) guia o agente a completar o fluxo manualmente e gravar tokens onde o Hermes espera.

**Armadilha — WAF rejeita redirect URIs `127.0.0.1`.** Alguns provedores colocam WAF na frente do authorization server que 403a qualquer authorize request cuja query string contenha `127.0.0.1` literal (AWS API Gateway da Reclaim.ai é exemplo conhecido — toda tentativa retorna `{"message":"Forbidden"}` antes de chegar ao app OAuth). Defina `oauth.redirect_host: localhost` para usar `http://localhost:<port>/callback`; o listener de callback ainda binda `127.0.0.1` de qualquer forma.

Veja [OAuth over SSH / Remote Hosts](../../guides/oauth-over-ssh.md#mcp-servers) para o walkthrough completo, incluindo servidores sem DCR (ex. Slack), `client_id`/`client_secret` pré-registrados, customização de scope e re-auth via `hermes mcp login <server>`.

**Armadilha — provedores sem registro automático (Google Drive, Atlassian).** Alguns servidores rejeitam o passo de registro dinâmico de cliente (RFC 7591) em que `auth: oauth` nu confia — o servidor oficial Drive do Google (`https://drivemcp.googleapis.com/mcp/v1`) retorna `400 Bad Request`, então nenhum cliente OAuth é criado e nenhum token é adquirido. O sintoma é sutil: esses servidores também servem `tools/list` *sem* auth, então `hermes mcp login` pode listar ferramentas e parecer que funcionou, mas toda chamada real de ferramenta depois dá timeout. `hermes mcp login` agora detecta isso (verifica que um token realmente caiu no disco) e diz para você fornecer seu próprio cliente OAuth. Crie um no console do provedor e adicione à config:

```yaml
mcp_servers:
  googledrive:
    url: "https://drivemcp.googleapis.com/mcp/v1"
    auth: oauth
    oauth:
      client_id: "<your-oauth-client-id>"
      client_secret: "<your-oauth-client-secret>"
```

Depois rode `hermes mcp login googledrive` — com cliente pré-registrado, o Hermes pula registro e roda o fluxo normal de autorização no navegador.

**Armadilha — race de auto-reload de config.** Quando você edita `~/.hermes/config.yaml` de dentro de sessão Hermes rodando, a CLI auto-recarrega conexões MCP com timeout de 30s. Isso não basta para fluxo OAuth interativo. Adicione a entrada, depois rode `hermes mcp login <server>` de terminal novo — espera os 5 minutos completos para você completar auth.

## mTLS / certificados cliente {#mtls-client-certificates}

Servidores MCP HTTP remotos que exigem TLS mútuo (autenticação por certificado cliente) são suportados via `client_cert` / `client_key`. O Hermes passa o certificado resolvido ao cliente HTTP subjacente para o handshake TLS.

`client_cert` aceita três formas:

- **Um caminho PEM combinado** — um arquivo com certificado e chave privada:

```yaml
mcp_servers:
  internal_api:
    url: "https://mcp.internal.example.com/mcp"
    client_cert: "~/.certs/mcp-client.pem"
```

- **Tupla 2 `[cert, key]`** — certificado e chave em arquivos separados (equivalente a `client_cert` + `client_key`):

```yaml
mcp_servers:
  internal_api:
    url: "https://mcp.internal.example.com/mcp"
    client_cert: ["~/.certs/mcp-client.crt", "~/.certs/mcp-client.key"]
```

- **Tupla 3 `[cert, key, password]`** — quando a chave privada é criptografada, o terceiro elemento é a passphrase:

```yaml
mcp_servers:
  internal_api:
    url: "https://mcp.internal.example.com/mcp"
    client_cert: ["~/.certs/mcp-client.crt", "~/.certs/mcp-client.key", "${MCP_KEY_PASSWORD}"]
```

Você também pode manter cert e key totalmente separados via `client_cert` (PEM combinado) mais `client_key` explícito. Caminhos suportam expansão `~`; arquivo ausente levanta erro claro escopado ao servidor em vez de falha opaca de handshake TLS.

## Cabeçalho de identidade por usuário {#per-user-identity-header}

Servidores MCP HTTP/SSE remotos que chaveiam comportamento por identidade do caller (rate limits por usuário, audit trails, roteamento multi-tenant) podem receber cabeçalho de identidade em toda requisição via `identity_header`:

```yaml
mcp_servers:
  team_api:
    url: "https://mcp.team.example.com/mcp"
    identity_header:
      name: "X-User-Id"
      value_from: "static"   # "static" (padrão) ou "profile"
      value: "alice"         # obrigatório para static
```

- `value_from: static` envia o `value` literal do config.yaml.
- `value_from: profile` envia o nome do profile Hermes ativo, resolvido uma vez na conexão — útil quando vários profiles numa máquina falam com o mesmo servidor e ele precisa distingui-los.

Entrada explícita no mapping `headers` do servidor com o mesmo nome (qualquer casing) sempre vence; o identity header nunca sobrescreve sua config de header. Blocos `identity_header` inválidos geram aviso e são ignorados — nunca bloqueiam a conexão do servidor. Em servidores stdio a chave é ignorada com aviso (transportes stdio não têm headers).

## Referência básica de configuração {#basic-configuration-reference}

O Hermes lê config MCP de `~/.hermes/config.yaml` sob `mcp_servers`.

### Chaves comuns {#common-keys}

| Chave | Tipo | Significado |
|---|---|---|
| `command` | string | Executável para servidor MCP stdio |
| `args` | list | Argumentos para o servidor stdio |
| `env` | mapping | Variáveis de ambiente passadas ao servidor stdio |
| `url` | string | Endpoint MCP HTTP |
| `headers` | mapping | Headers HTTP para servidores remotos |
| `client_cert` | string \| list | Certificado cliente para mTLS — caminho PEM combinado, ou `[cert, key]` / `[cert, key, password]` |
| `client_key` | string | Caminho PEM da chave privada cliente (quando separada de `client_cert`) |
| `identity_header` | mapping | Header de identidade por usuário opcional para servidores HTTP/SSE — `{name, value_from: static\|profile, value}` |
| `timeout` | number | Timeout de chamada de ferramenta |
| `connect_timeout` | number | Timeout de conexão inicial (também limita handshake MCP `initialize`) |
| `idle_timeout_seconds` | number | Reciclar servidor stdio após tantos segundos sem chamada de ferramenta (`0` = nunca, padrão). O servidor reinicia transparentemente na próxima chamada. |
| `max_lifetime_seconds` | number | Reciclar servidor stdio após idade total (`0` = nunca, padrão). Reinicia transparentemente no próximo uso. |
| `enabled` | bool | Se `false`, Hermes ignora o servidor por completo |
| `supports_parallel_tool_calls` | bool | Se `true`, ferramentas deste servidor podem rodar em paralelo |
| `tools` | mapping | Filtragem por servidor e política utilitária |

### Exemplo stdio mínimo {#minimal-stdio-example}

```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
```

### Reciclando servidores stdio pesados em memória {#recycling-memory-heavy-stdio-servers}

Servidores MCP baseados em browser (ex. `@playwright/mcp`) mantêm Chromium
completo residente após a primeira chamada de ferramenta — centenas de MB que nunca são
liberados. Opte por reciclagem automática e o servidor é derrubado após
limite idle/lifetime, depois reiniciado transparentemente na próxima vez que uma de
suas ferramentas for chamada (suas ferramentas permanecem registradas o tempo todo):

```yaml
mcp_servers:
  playwright:
    command: "npx"
    args: ["-y", "@playwright/mcp@latest", "--headless"]
    idle_timeout_seconds: 900     # reciclar após 15 min sem chamada de ferramenta
    max_lifetime_seconds: 86400   # e pelo menos uma vez por dia de qualquer forma
```

### Exemplo HTTP mínimo {#minimal-http-example}

```yaml
mcp_servers:
  company_api:
    url: "https://mcp.internal.example.com"
    headers:
      Authorization: "Bearer ***"
```

## Presets built-in {#built-in-presets}

Para servidores MCP conhecidos, `hermes mcp add` aceita flag `--preset` que preenche detalhes de transporte para você não precisar buscar command e args. O preset só fornece defaults — qualquer outra coisa (env vars, headers, filtragem) que você passa na mesma linha de comando ainda vence.

| Preset | O que configura |
|---|---|
| `codex` | Servidor MCP do Codex CLI (`codex mcp-server` over stdio). Exige CLI `codex` no PATH. |

```bash
# Adicionar Codex CLI como servidor MCP numa linha
hermes mcp add codex --preset codex
```

Isso grava o equivalente de:

```yaml
mcp_servers:
  codex:
    command: "codex"
    args: ["mcp-server"]
```

Você pode escolher qualquer nome local (`hermes mcp add my-codex --preset codex` serve); o preset só fornece defaults de `command`/`args`.

## Como o Hermes registra ferramentas MCP {#how-hermes-registers-mcp-tools}

O Hermes prefixa ferramentas MCP para não colidir com nomes built-in:

```text
mcp_<server_name>_<tool_name>
```

Exemplos:

| Servidor | Ferramenta MCP | Nome registrado |
|---|---|---|
| `filesystem` | `read_file` | `mcp_filesystem_read_file` |
| `github` | `create-issue` | `mcp_github_create_issue` |
| `my-api` | `query.data` | `mcp_my_api_query_data` |

Na prática, você normalmente não precisa chamar o nome prefixado manualmente — o Hermes vê a ferramenta e a escolhe durante raciocínio normal.

### Sanitização de tool-result e `_meta` {#tool-result-sanitization-and-_meta}

Dois comportamentos se aplicam a todo resultado de ferramenta MCP antes do modelo vê-lo:

- **Caracteres Unicode TAG invisíveis são stripped.** Caracteres na faixa U+E0000–U+E007F renderizam como nada em terminais e UIs de chat mas são plenamente visíveis ao modelo — um canal clássico de prompt-injection smuggling para um servidor malicioso ou comprometido. O Hermes os stripa de resultados de ferramenta, conteúdo de resource e descrições de ferramenta. Sequências legítimas de emoji tag (bandeiras regionais como 🏴󠁧󠁢󠁳󠁣󠁴󠁿) são preservadas.
- **`_meta` de vendor é exposto; chaves reservadas do protocolo não.** Quando um servidor anexa um mapping `_meta` a um resultado de ferramenta (namespaces de vendor como `com.example/handoff`), o Hermes o passa ao modelo junto com o conteúdo do resultado. Chaves sob prefixos reservados do protocolo — um label `modelcontextprotocol` ou `mcp` seguido de outro label, ex. `modelcontextprotocol.io/...` ou `tools.mcp.com/...` — são dropped, casando com as regras de nome de chave do spec MCP. Se nada voltado ao modelo permanecer, o campo `_meta` é omitido por completo.

## Ferramentas utilitárias MCP {#mcp-utility-tools}

Quando suportado, o Hermes também registra ferramentas utilitárias em torno de resources e prompts MCP:

- `list_resources`
- `read_resource`
- `list_prompts`
- `get_prompt`

Estas são registradas por servidor com o mesmo padrão de prefixo, por exemplo:

- `mcp_github_list_resources`
- `mcp_github_get_prompt`

### Importante {#important}

Estas ferramentas utilitárias agora são capability-aware:
- Hermes só registra utilitários de resource se a sessão MCP suportar operações de resource de fato
- Hermes só registra utilitários de prompt se a sessão MCP suportar operações de prompt de fato

Então um servidor que expõe ferramentas callable mas sem resources/prompts não recebe esses wrappers extras.

## Filtragem por servidor {#per-server-filtering}

Você controla quais ferramentas cada servidor MCP contribui ao Hermes, permitindo gerenciamento fino do seu namespace de ferramentas.

### Desabilitar um servidor por completo {#disable-a-server-entirely}

```yaml
mcp_servers:
  legacy:
    url: "https://mcp.legacy.internal"
    enabled: false
```

Se `enabled: false`, o Hermes ignora o servidor por completo e nem tenta conexão.

### Whitelist de ferramentas do servidor {#whitelist-server-tools}

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

Só essas ferramentas MCP do servidor são registradas.

### Blacklist de ferramentas do servidor {#blacklist-server-tools}

```yaml
mcp_servers:
  stripe:
    url: "https://mcp.stripe.com"
    tools:
      exclude: [delete_customer]
```

Todas as ferramentas do servidor são registradas exceto as excluídas.

### Padrões glob {#glob-patterns}

Ambas as listas aceitam globs estilo fnmatch junto com nomes exatos — essencial para
superfícies flat enormes como API MCP da Cloudflare (`?codemode=false`, ~3.300
ferramentas) onde excluir áreas de produto endpoint a endpoint é impraticável:

```yaml
mcp_servers:
  cloudflare:
    url: "https://mcp.cloudflare.com/mcp?codemode=false"
    auth: oauth
    tools:
      exclude: ["*_radar_*", "*_accounts_dlp_*", "*_zones_web3_*"]
```

Entradas sem metacaracteres glob (`*`, `?`, `[`) correspondem exatamente — `docs`
exclui só a ferramenta chamada `docs`, nunca `docs_search`.

### Regra de precedência {#precedence-rule}

Se ambos estiverem presentes:

```yaml
tools:
  include: [create_issue]
  exclude: [create_issue, delete_issue]
```

`include` vence.

### Filtrar ferramentas utilitárias também {#filter-utility-tools-too}

Você também pode desabilitar separadamente wrappers utilitários adicionados pelo Hermes:

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

### Exemplo completo {#full-example}

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

Se sua config filtrar todas as ferramentas callable e desabilitar ou omitir todos os utilitários suportados, o Hermes não cria toolset MCP runtime vazio para esse servidor.

Isso mantém a lista de ferramentas limpa.

## Comportamento em runtime {#runtime-behavior}

### Momento da descoberta {#discovery-time}

O Hermes descobre servidores MCP na inicialização e registra suas ferramentas no registry normal de ferramentas.

### Descoberta Dinâmica de Ferramentas {#dynamic-tool-discovery}

Servidores MCP podem notificar o Hermes quando ferramentas disponíveis mudam em runtime enviando notificação `notifications/tools/list_changed`. Quando o Hermes recebe, re-busca automaticamente a lista de ferramentas do servidor e atualiza o registry — sem `/reload-mcp` manual.

Útil para servidores MCP cujas capacidades mudam dinamicamente (ex. servidor que adiciona ferramentas quando novo schema de banco carrega, ou remove quando serviço cai).

O refresh é protegido por lock para notificações rápidas do mesmo servidor não causarem refreshes sobrepostos. Notificações de mudança de prompt e resource (`prompts/list_changed`, `resources/list_changed`) são recebidas mas ainda não acionadas.

### Recarregar {#reloading}

Se mudar config MCP, use:

```text
/reload-mcp
```

Isso recarrega servidores MCP da config e refresha a lista de ferramentas disponíveis. Para mudanças de ferramenta em runtime empurradas pelo servidor, veja [Descoberta Dinâmica de Ferramentas](#dynamic-tool-discovery) acima.

### Toolsets {#toolsets}

Cada servidor MCP configurado também cria toolset runtime quando contribui com pelo menos uma ferramenta registrada:

```text
mcp-<server>
```

Isso torna servidores MCP mais fáceis de raciocinar no nível de toolset.

## Modelo de segurança {#security-model}

### Filtragem de env stdio {#stdio-env-filtering}

Para servidores stdio, o Hermes não passa cegamente todo o ambiente do seu shell.

Só `env` explicitamente configurado mais baseline seguro passam. Isso reduz vazamento acidental de segredos.

### Controle de exposição em nível de config {#config-level-exposure-control}

O novo suporte de filtragem também é controle de segurança:
- desabilitar ferramentas perigosas que você não quer que o modelo veja
- expor só whitelist mínima para servidor sensível
- desabilitar wrappers resource/prompt quando não quer essa superfície exposta

## Casos de uso de exemplo {#example-use-cases}

### Servidor GitHub com superfície mínima de gestão de issues {#github-server-with-a-minimal-issue-management-surface}

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

### Servidor Stripe com ações perigosas removidas {#stripe-server-with-dangerous-actions-removed}

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

### Servidor filesystem para uma raiz de projeto {#filesystem-server-for-a-single-project-root}

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

### Servidor MCP não conecta {#mcp-server-not-connecting}

Verifique:

```bash
# Verificar se deps MCP estão instaladas (já incluídas na instalação padrão)
cd ~/.hermes/hermes-agent && uv pip install -e ".[mcp]"

node --version
npx --version
```

Depois verifique sua config e reinicie o Hermes.

### Ferramentas não aparecem {#tools-not-appearing}

Possíveis causas:
- o servidor falhou ao conectar
- discovery falhou
- sua config de filtro excluiu as ferramentas
- a capacidade utilitária não existe nesse servidor
- o servidor está desabilitado com `enabled: false`

Se você está filtrando de propósito, isso é esperado.

### Por que utilitários de resource ou prompt não apareceram? {#why-didnt-resource-or-prompt-utilities-appear}

Porque o Hermes agora só registra esses wrappers quando ambos são verdadeiros:
1. sua config permite
2. a sessão do servidor suporta de fato a capacidade

Isso é intencional e mantém a lista de ferramentas honesta.

## Chamadas de Ferramenta em Paralelo {#parallel-tool-calls}

Por padrão, ferramentas MCP rodam sequencialmente — uma por vez. Se seu servidor MCP expõe ferramentas seguras para rodar concorrentemente (ex. consultas read-only, chamadas API independentes), você pode optar por execução paralela:

```yaml
mcp_servers:
  docs:
    command: "docs-server"
    supports_parallel_tool_calls: true
```

Quando `supports_parallel_tool_calls` é `true`, o Hermes pode executar múltiplas ferramentas desse servidor ao mesmo tempo dentro de um batch de tool-call, como faz para ferramentas built-in read-only (web_search, read_file, etc.).

:::caution
Habilite chamadas paralelas só para servidores MCP cujas ferramentas são seguras para rodar ao mesmo tempo. Se ferramentas leem e escrevem estado compartilhado, arquivos, bancos ou recursos externos, revise race conditions read/write antes de habilitar.
:::

## Suporte a MCP Sampling {#mcp-sampling-support}

Servidores MCP podem solicitar inferência LLM do Hermes via protocolo `sampling/createMessage`. Isso permite que um servidor MCP peça ao Hermes para gerar texto em seu nome — útil para servidores que precisam de capacidades LLM mas não têm acesso próprio a modelo.

Sampling está **habilitado por padrão** para todos os servidores MCP (quando o MCP SDK suporta). Configure por servidor sob a chave `sampling`:

```yaml
mcp_servers:
  my_server:
    command: "my-mcp-server"
    sampling:
      enabled: true            # Habilitar sampling (padrão: true)
      model: "openai/gpt-4o"  # Override de modelo para requisições sampling (opcional)
      max_tokens_cap: 4096     # Max tokens por resposta sampling (padrão: 4096)
      timeout: 30              # Timeout em segundos por requisição (padrão: 30)
      max_rpm: 10              # Rate limit: max requisições por minuto (padrão: 10)
      max_tool_rounds: 5       # Max rounds de tool-use em loops sampling (padrão: 5)
      allowed_models: []       # Allowlist de nomes de modelo que o servidor pode pedir (vazio = qualquer)
      log_level: "info"        # Nível de audit log: debug, info, ou warning (padrão: info)
```

O handler de sampling inclui rate limiter sliding-window, timeouts por requisição e limites de profundidade de tool-loop para prevenir uso descontrolado. Métricas (contagem de requisições, erros, tokens usados) são rastreadas por instância de servidor.

Para desabilitar sampling num servidor específico:

```yaml
mcp_servers:
  untrusted_server:
    url: "https://mcp.example.com"
    sampling:
      enabled: false
```

## Suporte a MCP Elicitation {#mcp-elicitation-support}

Servidores MCP podem pedir input estruturado do usuário no meio de tool-call via protocolo `elicitation/create` (mcp Python SDK ≥ 1.11.0). O Hermes roteia elicitations **form-mode** pela superfície de aprovação existente — prompt interativo na CLI/TUI, ou botões de aprovação em plataformas gateway como Telegram e Slack — para a requisição chegar a você onde quer que a sessão viva. Elicitations **URL-mode** (onde um servidor aponta para URL externa) são recusadas como não suportadas.

Elicitation está **habilitada por padrão** por servidor. Configure sob a chave `elicitation`:

```yaml
mcp_servers:
  my_server:
    command: "my-mcp-server"
    elicitation:
      enabled: true    # padrão: true
      timeout: 300     # segundos para esperar sua resposta (padrão: 300)
```

O timeout padrão de 5 minutos espelha o default de aprovação do gateway para usuários em superfícies async terem tempo de responder antes do servidor desistir. Métricas por servidor (requisições, aceitas, recusadas, erros) são rastreadas no handler.

## Rodando Hermes como servidor MCP {#running-hermes-as-an-mcp-server}

Além de conectar **a** servidores MCP, o Hermes também pode **ser** um servidor MCP. Isso permite que outros agentes compatíveis com MCP (Claude Code, Cursor, Codex, ou qualquer cliente MCP) usem capacidades de mensagens do Hermes — listar conversas, ler histórico de mensagens e enviar mensagens por todas as plataformas conectadas.

### Quando usar {#when-to-use-this}

- Você quer que Claude Code, Cursor ou outro agente de coding envie e leia mensagens Telegram/Discord/Slack pelo Hermes
- Você quer um único servidor MCP que faça bridge para todas as plataformas de mensagens conectadas do Hermes de uma vez
- Você já tem gateway Hermes rodando com plataformas conectadas

### Início rápido {#quick-start-1}

```bash
hermes mcp serve
```

Isso inicia um servidor MCP stdio. O cliente MCP (não você) gerencia o ciclo de vida do processo.

### Configuração do cliente MCP {#mcp-client-configuration}

Adicione Hermes à config do seu cliente MCP. Por exemplo, no `~/.claude/claude_desktop_config.json` do Claude Code:

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

Ou se instalou Hermes num local específico:

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

### Ferramentas disponíveis {#available-tools}

O servidor MCP expõe 10 ferramentas, matching a superfície channel bridge do OpenClaw mais um browser de canais específico Hermes:

| Ferramenta | Descrição |
|------|-------------|
| `conversations_list` | Listar conversas de mensagens ativas. Filtrar por plataforma ou buscar por nome. |
| `conversation_get` | Obter info detalhada de uma conversa por session key. |
| `messages_read` | Ler histórico recente de mensagens de uma conversa. |
| `attachments_fetch` | Extrair anexos não-texto (imagens, mídia) de mensagem específica. |
| `events_poll` | Poll de novos eventos de conversa desde posição cursor. |
| `events_wait` | Long-poll / bloquear até próximo evento chegar (quase tempo real). |
| `messages_send` | Enviar mensagem por plataforma (ex. `telegram:123456`, `discord:#general`). |
| `channels_list` | Listar targets de mensagens disponíveis em todas as plataformas. |
| `permissions_list_open` | Listar requisições de aprovação pendentes observadas nesta sessão bridge. |
| `permissions_respond` | Permitir ou negar requisição de aprovação pendente. |

### Sistema de eventos {#event-system}

O servidor MCP inclui bridge de eventos ao vivo que faz poll no banco de sessões do Hermes por novas mensagens. Isso dá aos clientes MCP consciência quase tempo real de conversas entrantes:

```
# Poll de novos eventos (non-blocking)
events_poll(after_cursor=0)

# Esperar próximo evento (bloqueia até timeout)
events_wait(after_cursor=42, timeout_ms=30000)
```

Tipos de evento: `message`, `approval_requested`, `approval_resolved`

A fila de eventos é in-memory e começa quando o bridge conecta. Mensagens mais antigas estão disponíveis via `messages_read`.

### Opções {#options}

```bash
hermes mcp serve              # Modo normal
hermes mcp serve --verbose    # Debug logging em stderr
```

### Como funciona {#how-it-works}

O servidor MCP lê dados de conversa diretamente do session store do Hermes — `~/.hermes/state.db` é a fonte primária, com `sessions.json` mantido só como fallback legacy. Uma thread em background faz poll no banco por novas mensagens e mantém fila de eventos in-memory. Para enviar mensagens, usa o mesmo send engine interno (`tools/send_message_tool.py`) que alimenta entrega cron e CLI `hermes send`.

O gateway NÃO precisa estar rodando para operações de leitura (listar conversas, ler histórico, poll de eventos). PRECISA estar rodando para operações de envio, já que os adapters de plataforma precisam de conexões ativas.

### Limites atuais {#current-limits}

- O `hermes mcp serve` embutido expõe servidor MCP **somente stdio** hoje. Se precisar de servidor MCP HTTP, rode adapter separado — ou, muito mais comum, use o lado **cliente** MCP do Hermes, que já fala stdio e HTTP (`url` + `headers` em `mcp_servers.yaml` / `config.yaml`; veja [Servidores HTTP](#http-servers) acima).
- Poll de eventos em intervalos ~200ms via poll DB otimizado por mtime (pula trabalho quando arquivos não mudaram)
- Ainda sem protocolo push notification `claude/channel`
- Envios só texto (sem envio de mídia/anexo por `messages_send`)

## Docs relacionados {#related-docs}

- [Usar MCP com Hermes](/guides/use-mcp-with-hermes)
- [Comandos CLI](/reference/cli-commands)
- [Slash Commands](/reference/slash-commands)
- [FAQ](/reference/faq)
