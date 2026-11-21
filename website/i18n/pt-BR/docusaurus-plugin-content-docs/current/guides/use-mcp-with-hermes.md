---
sidebar_position: 6
title: "Use o MCP com o Hermes"
description: "Um guia prático para conectar servidores MCP ao Hermes Agent, filtrar suas ferramentas e usá-los com segurança em fluxos de trabalho reais"
---

# Use o MCP com o Hermes {#use-mcp-with-hermes}

Este guia mostra como usar de fato o MCP com o Hermes Agent em fluxos de trabalho do dia a dia.

Se a página de recursos explica o que é o MCP, este guia trata de como extrair valor dele rapidamente e com segurança.

## Quando você deve usar o MCP? {#when-should-you-use-mcp}

Use o MCP quando:
- uma ferramenta já existe na forma MCP e você não quer construir uma ferramenta nativa do Hermes
- você quer que o Hermes opere em um sistema local ou remoto através de uma camada RPC limpa
- você quer controle granular de exposição por servidor
- você quer conectar o Hermes a APIs internas, bancos de dados ou sistemas corporativos sem modificar o núcleo do Hermes

Não use o MCP quando:
- uma ferramenta integrada do Hermes já resolve bem o trabalho
- o servidor expõe uma superfície de ferramentas enorme e perigosa e você não está preparado para filtrá-la
- você só precisa de uma integração muito específica e uma ferramenta nativa seria mais simples e segura

## Modelo Mental {#mental-model}

Pense no MCP como uma camada de adaptação:

- o Hermes permanece sendo o agente
- servidores MCP contribuem com ferramentas
- o Hermes descobre essas ferramentas na inicialização ou no momento de recarregamento
- o modelo pode usá-las como ferramentas normais
- você controla quanto de cada servidor fica visível

Essa última parte importa. Um bom uso do MCP não é apenas "conectar tudo". É "conectar a coisa certa, com a menor superfície útil".

## Passo 1: instale o suporte a MCP {#step-1-install-mcp-support}

Se você instalou o Hermes com o script de instalação padrão, o suporte a MCP já está incluído (o instalador executa `uv pip install -e ".[all]"`).

Se você instalou sem extras e precisa adicionar o MCP separadamente:

```bash
cd ~/.hermes/hermes-agent
uv pip install -e ".[mcp]"
```

Para servidores baseados em npm, certifique-se de que Node.js e `npx` estejam disponíveis.

Para muitos servidores MCP em Python, `uvx` é uma boa opção padrão.

## Passo 2: adicione um servidor primeiro {#step-2-add-one-server-first}

Comece com um único servidor seguro.

Exemplo: acesso ao sistema de arquivos de apenas um diretório de projeto.

```yaml
mcp_servers:
  project_fs:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/my-project"]
```

Então inicie o Hermes:

```bash
hermes chat
```

Agora peça algo concreto:

```text
Inspect this project and summarize the repo layout.
```

## Passo 3: verifique se o MCP foi carregado {#step-3-verify-mcp-loaded}

Você pode verificar o MCP de algumas formas:

- o banner/status do Hermes deve mostrar a integração MCP quando configurada
- pergunte ao Hermes quais ferramentas ele tem disponíveis
- use `/reload-mcp` após mudanças de configuração
- verifique os logs se o servidor falhar ao conectar

Um prompt de teste prático:

```text
Tell me which MCP-backed tools are available right now.
```

## Passo 4: comece a filtrar imediatamente {#step-4-start-filtering-immediately}

Não espere até mais tarde se o servidor expõe muitas ferramentas.

### Exemplo: liste na whitelist apenas o que você quer {#example-whitelist-only-what-you-want}

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
    tools:
      include: [list_issues, create_issue, search_code]
```

Esse é geralmente o melhor padrão para sistemas sensíveis.

## WSL2: conecte o Hermes no WSL ao Chrome do Windows {#wsl2-bridge-hermes-in-wsl-to-windows-chrome}

Essa é a configuração prática quando:

- o Hermes é executado dentro do WSL2
- o navegador que você quer controlar é seu Chrome normal, já conectado, no Windows
- `/browser connect` é estranho ou não confiável a partir do WSL

Nessa configuração, o Hermes **não** se conecta ao Chrome diretamente. Em vez disso:

- o Hermes é executado no WSL
- o Hermes inicia um servidor MCP local em stdio
- esse servidor MCP é iniciado através da interoperabilidade do Windows (`cmd.exe` ou `powershell.exe`)
- o servidor MCP se conecta à sua sessão viva do Chrome no Windows

Modelo mental:

```text
Hermes (WSL) -> MCP stdio bridge -> Windows Chrome
```

### Por que esse modo é útil {#why-this-mode-is-useful}

- você mantém seu perfil real do navegador Windows, cookies e logins
- o Hermes permanece em seu ambiente Unix suportado (WSL2)
- o controle do navegador é exposto como ferramentas MCP em vez de depender do transporte de navegador do núcleo do Hermes

### Servidor recomendado {#recommended-server}

Use `chrome-devtools-mcp`.

Se o seu Chrome do Windows já tem a depuração remota ativa em `chrome://inspect/#remote-debugging`, adicione-o assim a partir do WSL:

```bash
hermes mcp add chrome-devtools-win --command cmd.exe --args /c npx -y chrome-devtools-mcp@latest --autoConnect --no-usage-statistics
```

Depois de salvar o servidor:

```bash
hermes mcp test chrome-devtools-win
```

Então inicie uma nova sessão do Hermes ou execute:

```text
/reload-mcp
```

### Prompt típico {#typical-prompt}

Uma vez carregado, o Hermes pode usar diretamente as ferramentas de navegador com prefixo MCP. Por exemplo:

```text
调用 MCP 工具 mcp_chrome_devtools_win_list_pages，列出当前浏览器标签页。
```

### Quando `/browser connect` é a ferramenta errada {#when-browser-connect-is-the-wrong-tool}

Se o Hermes é executado no WSL e o Chrome é executado no Windows, `/browser connect` pode falhar mesmo que o Chrome esteja aberto e depurável.

Motivos comuns:

- o WSL não consegue alcançar o mesmo endpoint local que o Chrome expõe para ferramentas do Windows
- os fluxos mais novos de depuração ao vivo do Chrome não são iguais a um clássico `ws://localhost:9222`
- o navegador é mais fácil de conectar a partir de um auxiliar do lado do Windows, como o `chrome-devtools-mcp`

Nesses casos, mantenha `/browser connect` para configurações no mesmo ambiente e use o MCP para a ponte WSL-para-Windows do navegador.

### Armadilhas conhecidas {#known-pitfalls}

- Inicie o Hermes a partir de um caminho montado no Windows, como `/mnt/c/Users/<you>` ou `/mnt/c/workspace/...`, ao usar executáveis stdio do Windows através do MCP.
- Se você iniciar o Hermes a partir de `/root` ou `/home/...`, o Windows pode emitir um aviso de diretório atual `UNC` antes do servidor MCP iniciar.
- Se `chrome-devtools-mcp --autoConnect` expirar ao enumerar páginas, reduza abas em segundo plano/congeladas no Chrome e tente novamente.

### Exemplo: coloque ações perigosas na blacklist {#example-blacklist-dangerous-actions}

```yaml
mcp_servers:
  stripe:
    url: "https://mcp.stripe.com"
    headers:
      Authorization: "Bearer ***"
    tools:
      exclude: [delete_customer, refund_payment]
```

### Exemplo: desabilite também os wrappers utilitários {#example-disable-utility-wrappers-too}

```yaml
mcp_servers:
  docs:
    url: "https://mcp.docs.example.com"
    tools:
      prompts: false
      resources: false
```

## O que a filtragem realmente afeta? {#what-does-filtering-actually-affect}

Existem duas categorias de funcionalidade exposta pelo MCP no Hermes:

1. Ferramentas nativas do servidor MCP
- filtradas com:
  - `tools.include`
  - `tools.exclude`

2. Wrappers utilitários adicionados pelo Hermes
- filtrados com:
  - `tools.resources`
  - `tools.prompts`

### Wrappers utilitários que você pode ver {#utility-wrappers-you-may-see}

Resources (recursos):
- `list_resources`
- `read_resource`

Prompts:
- `list_prompts`
- `get_prompt`

Esses wrappers só aparecem se:
- sua configuração os permitir, e
- a sessão do servidor MCP realmente suportar essas capacidades

Então o Hermes não vai fingir que um servidor tem resources/prompts se ele não tiver.

## Padrões Comuns {#common-patterns}

### Padrão 1: assistente de projeto local {#pattern-1-local-project-assistant}

Use o MCP para um servidor de sistema de arquivos ou git local ao repositório quando você quer que o Hermes raciocine sobre um espaço de trabalho delimitado.

```yaml
mcp_servers:
  fs:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/project"]

  git:
    command: "uvx"
    args: ["mcp-server-git", "--repository", "/home/user/project"]
```

Bons prompts:

```text
Review the project structure and identify where configuration lives.
```

```text
Check the local git state and summarize what changed recently.
```

### Padrão 2: registro de trabalho nativo do repositório com Open Scaffold {#pattern-2-repo-native-work-record-with-open-scaffold}

Use o [Open Scaffold](https://github.com/graphanov/open-scaffold) quando você quiser que o Hermes leia o registro duradouro de trabalho de IA de um repositório: missão, planos, notas de evidência, pacotes de handoff e resultados de revisão/gate. O Hermes continua sendo o agente; o Open Scaffold continua sendo o registro local do repositório.

Adicione o servidor para um repositório com scaffold:

```bash
hermes mcp add open_scaffold --command npx --args -y open-scaffold@latest mcp serve --repo /absolute/path/to/repo
hermes mcp test open_scaffold
```

Depois mantenha a superfície exposta orientada à leitura. Escolha `select` no prompt do `hermes mcp add`, ou edite o `config.yaml` posteriormente:

```yaml
mcp_servers:
  open_scaffold:
    command: "npx"
    args: ["-y", "open-scaffold@latest", "mcp", "serve", "--repo", "/absolute/path/to/repo"]
    tools:
      include:
        - list_plans
        - get_plan
        - get_mission
        - list_evidence
        - get_evidence
        - get_status
        - search_plans
        - list_amendments
        - get_handoff
        - analyze_loop
        - gate_loop
      prompts: false
```

Bons prompts:

```text
Use the Open Scaffold MCP tools to compile the current handoff packet and tell me the next legal action.
```

```text
Inspect the active plans and evidence notes, then say whether this repo is ready for human review or needs another attempt.
```

Notas sobre limites:

- O Open Scaffold MCP é local-first e somente leitura por padrão.
- Suas ferramentas de escrita exigem que o servidor seja iniciado com `--allow-write`; não habilite isso até que você queira explicitamente que o Hermes altere arquivos `.osc`.
- O Open Scaffold registra e faz gate do trabalho; ele não autoriza o Hermes a fazer merge, publicar, implantar ou iniciar runtimes.
- Fixe `open-scaffold@<version>` em vez de `@latest` se você precisar de esquemas de ferramentas reproduzíveis.

### Padrão 3: assistente de triagem do GitHub {#pattern-3-github-triage-assistant}

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
    tools:
      include: [list_issues, create_issue, update_issue, search_code]
      prompts: false
      resources: false
```

Bons prompts:

```text
List open issues about MCP, cluster them by theme, and draft a high-quality issue for the most common bug.
```

```text
Search the repo for uses of _discover_and_register_server and explain how MCP tools are registered.
```

### Padrão 4: assistente de API interna {#pattern-4-internal-api-assistant}

```yaml
mcp_servers:
  internal_api:
    url: "https://mcp.internal.example.com"
    headers:
      Authorization: "Bearer ***"
    tools:
      include: [list_customers, get_customer, list_invoices]
      resources: false
      prompts: false
```

Bons prompts:

```text
Look up customer ACME Corp and summarize recent invoice activity.
```

Esse é o tipo de lugar onde uma whitelist estrita é muito melhor do que uma lista de exclusão.

### Padrão 4: servidores de documentação / conhecimento {#pattern-4-documentation--knowledge-servers}

Alguns servidores MCP expõem prompts ou resources que são mais parecidos com ativos de conhecimento compartilhado do que ações diretas.

```yaml
mcp_servers:
  docs:
    url: "https://mcp.docs.example.com"
    tools:
      prompts: true
      resources: true
```

Bons prompts:

```text
List available MCP resources from the docs server, then read the onboarding guide and summarize it.
```

```text
List prompts exposed by the docs server and tell me which ones would help with incident response.
```

## Tutorial: configuração completa com filtragem {#tutorial-end-to-end-setup-with-filtering}

Aqui está uma progressão prática.

### Fase 1: adicione o MCP do GitHub com uma whitelist rígida {#phase-1-add-github-mcp-with-a-tight-whitelist}

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
    tools:
      include: [list_issues, create_issue, search_code]
      prompts: false
      resources: false
```

Inicie o Hermes e pergunte:

```text
Search the codebase for references to MCP and summarize the main integration points.
```

### Fase 2: expanda apenas quando necessário {#phase-2-expand-only-when-needed}

Se depois você também precisar de atualizações de issues:

```yaml
tools:
  include: [list_issues, create_issue, update_issue, search_code]
```

Então recarregue:

```text
/reload-mcp
```

### Fase 3: adicione um segundo servidor com uma política diferente {#phase-3-add-a-second-server-with-different-policy}

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
    tools:
      include: [list_issues, create_issue, update_issue, search_code]
      prompts: false
      resources: false

  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/project"]
```

Agora o Hermes pode combiná-los:

```text
Inspect the local project files, then create a GitHub issue summarizing the bug you find.
```

É aí que o MCP se torna poderoso: fluxos de trabalho multissistema sem alterar o núcleo do Hermes.

## Recomendações de Uso Seguro {#safe-usage-recommendations}

### Prefira whitelists para sistemas perigosos {#prefer-allowlists-for-dangerous-systems}

Para qualquer coisa financeira, voltada ao cliente ou destrutiva:
- use `tools.include`
- comece com o menor conjunto possível

### Desabilite utilitários não usados {#disable-unused-utilities}

Se você não quer que o modelo navegue pelos resources/prompts fornecidos pelo servidor, desative-os:

```yaml
tools:
  resources: false
  prompts: false
```

### Mantenha os servidores com escopo estreito {#keep-servers-scoped-narrowly}

Exemplos:
- servidor de sistema de arquivos ancorado a um diretório de projeto, não a todo o seu diretório home
- servidor git apontando para um repositório
- servidor de API interna com exposição de ferramentas voltada à leitura por padrão

### Recarregue após mudanças de configuração {#reload-after-config-changes}

```text
/reload-mcp
```

Faça isso após alterar:
- listas de include/exclude
- flags de enabled
- alternâncias de resources/prompts
- headers de autenticação / env

## Solução de Problemas por Sintoma {#troubleshooting-by-symptom}

### "O servidor conecta, mas as ferramentas que eu esperava estão faltando" {#the-server-connects-but-the-tools-i-expected-are-missing}

Causas possíveis:
- filtrado por `tools.include`
- excluído por `tools.exclude`
- wrappers utilitários desabilitados via `resources: false` ou `prompts: false`
- o servidor não suporta de fato resources/prompts

### "O servidor está configurado, mas nada carrega" {#the-server-is-configured-but-nothing-loads}

Verifique:
- `enabled: false` não ficou na configuração
- o comando/runtime existe (`npx`, `uvx`, etc.)
- o endpoint HTTP está acessível
- o env ou headers de autenticação estão corretos

### "Por que vejo menos ferramentas do que o servidor MCP anuncia?" {#why-do-i-see-fewer-tools-than-the-mcp-server-advertises}

Porque o Hermes agora respeita sua política por servidor e o registro consciente de capacidades. Isso é esperado e geralmente desejável.

### "Como removo um servidor MCP sem excluir a configuração?" {#how-do-i-remove-an-mcp-server-without-deleting-the-config}

Use:

```yaml
enabled: false
```

Isso mantém a configuração, mas impede a conexão e o registro.

## Configurações MCP Iniciais Recomendadas {#recommended-first-mcp-setups}

Bons primeiros servidores para a maioria dos usuários:
- filesystem
- git
- GitHub
- servidores MCP de fetch / documentação
- uma API interna estreita

Primeiros servidores não muito bons:
- sistemas de negócios gigantes com muitas ações destrutivas e nenhuma filtragem
- qualquer coisa que você não entenda o suficiente para restringir

## Documentos Relacionados {#related-docs}

- [MCP (Model Context Protocol)](/user-guide/features/mcp)
- [FAQ](/reference/faq)
- [Comandos de Barra](/reference/slash-commands)
