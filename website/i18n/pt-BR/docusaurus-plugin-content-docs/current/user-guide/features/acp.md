---
sidebar_position: 11
title: "ACP Editor Integration"
description: "Use o Hermes Agent em editores compatíveis com ACP como VS Code, Zed e JetBrains"
---

# ACP Editor Integration

O Hermes Agent pode rodar como servidor ACP, permitindo que editores compatíveis com ACP conversem com o Hermes via stdio e renderizem:

- chat messages
- tool activity
- file diffs
- terminal commands
- approval prompts
- streamed thinking / response chunks

ACP é uma boa opção quando você quer que o Hermes se comporte como agente de coding nativo do editor em vez de CLI standalone ou bot de mensagens.

## What Hermes exposes in ACP mode {#what-hermes-exposes-in-acp-mode}

O Hermes roda com um toolset curado `hermes-acp` pensado para workflows de editor. Inclui:

- file tools: `read_file`, `write_file`, `patch`, `search_files`
- terminal tools: `terminal`, `process`
- web/browser tools
- memory, todo, session search
- skills
- execute_code and delegate_task
- vision

Exclui intencionalmente coisas que não encaixam na UX típica de editor, como entrega de mensagens e gerenciamento de cronjob.

## Installation {#installation}

Instale o Hermes normalmente, depois adicione o extra ACP:

```bash
pip install -e '.[acp]'
```

Isso instala a dependência `agent-client-protocol` e habilita:

- `hermes acp`
- `hermes-acp`
- `python -m acp_adapter`

Para installs via registry do Zed, o Zed lança o Hermes pela entrada oficial do ACP Registry. Essa entrada usa uma distribuição `uvx` que roda:

```bash
uvx --from 'hermes-agent[acp]==<version>' hermes-acp
```

Certifique-se de que `uv` está disponível no `PATH` antes de usar o caminho de install via registry.

## Launching the ACP server {#launching-the-acp-server}

Qualquer um dos seguintes inicia o Hermes em modo ACP:

```bash
hermes acp
```

```bash
hermes-acp
```

```bash
python -m acp_adapter
```

O Hermes registra em stderr para stdout permanecer reservado ao tráfego JSON-RPC ACP.

Para checagens não interativas:

```bash
hermes acp --version
hermes acp --check
```

### Browser tools (optional) {#browser-tools-optional}

Browser tools (`browser_navigate`, `browser_click`, etc.) dependem do pacote npm `agent-browser` e Chromium, que não fazem parte da wheel Python. Instale com:

```bash
hermes acp --setup-browser           # interactive (prompts before ~400 MB download)
hermes acp --setup-browser --yes     # accept the download non-interactively
```

Este é o comando standalone. O fluxo terminal-auth do registry Zed (`hermes acp --setup`) também oferece o bootstrap de browser como pergunta de follow-up após seleção de modelo, então a maioria dos usuários nunca precisa rodar `--setup-browser` diretamente.

O que faz:

- Instala Node.js 22 LTS em `~/.hermes/node/` se ausente
- `npm install -g agent-browser @askjo/camofox-browser` nesse prefix (sem sudo — o `--prefix` do `npm` aponta para o Node gerenciado pelo Hermes gravável pelo usuário)
- Instala Playwright Chromium, ou usa Chrome/Chromium do sistema detectado quando disponível

O bootstrap é idempotente — re-executar é rápido e pula trabalho já feito.

## Editor setup {#editor-setup}

### VS Code {#vs-code}

Instale a extensão [ACP Client](https://marketplace.visualstudio.com/items?itemName=formulahendry.acp-client).

Para conectar:

1. Abra o painel ACP Client na Activity Bar.
2. Selecione **Hermes Agent** na lista built-in de agentes.
3. Conecte e comece a conversar.

Se quiser definir o Hermes manualmente, adicione via settings do VS Code em `acp.agents`:

```json
{
  "acp.agents": {
    "Hermes Agent": {
      "command": "hermes",
      "args": ["acp"]
    }
  }
}
```

### Zed {#zed}

Zed v0.221.x e mais novo instala agentes externos pelo ACP Registry oficial.

1. Abra o Agent Panel.
2. Clique **Add Agent**, ou execute o comando `zed: acp registry`.
3. Busque **Hermes Agent**.
4. Instale e inicie uma nova thread de external-agent Hermes.

Prerequisites:

- Configure credenciais de provider do Hermes primeiro com `hermes model`, ou defina em `~/.hermes/.env` / `~/.hermes/config.yaml`.
- Instale `uv` para o launcher do registry rodar `uvx --from 'hermes-agent[acp]==<version>' hermes-acp`.

Para desenvolvimento local antes da entrada do registry estar disponível, use custom agent server nas settings do Zed:

```json
{
  "agent_servers": {
    "hermes-agent": {
      "type": "custom",
      "command": "hermes",
      "args": ["acp"]
    }
  }
}
```

### JetBrains {#jetbrains}

Use um plugin compatível com ACP e aponte para:

```text
/path/to/hermes-agent/acp_registry
```

## Registry manifest {#registry-manifest}

A cópia source dos metadados oficiais do ACP Registry do Hermes fica em:

```text
acp_registry/agent.json
acp_registry/icon.svg
```

O PR upstream do registry copia esses arquivos para o diretório top-level `hermes-agent/` em `agentclientprotocol/registry`.

A entrada do registry usa distribuição `uvx` que aponta diretamente para o release PyPI `hermes-agent`:

```text
uvx --from 'hermes-agent[acp]==<version>' hermes-acp
```

O CI do registry verifica que a versão pinada existe no PyPI, então `version` do manifest e pin `package` do uvx devem sempre bater com `pyproject.toml`. `scripts/release.py` os mantém em lockstep automaticamente.

## Configuration and credentials {#configuration-and-credentials}

Modo ACP usa a mesma configuração Hermes da CLI:

- `~/.hermes/.env`
- `~/.hermes/config.yaml`
- `~/.hermes/skills/`
- `~/.hermes/state.db`

A resolução de provider usa o resolver runtime normal do Hermes, então ACP herda o provider e credenciais configurados no momento. O Hermes também anuncia método de auth terminal (`--setup`) para clientes registry em first-run; isso abre o setup interativo de model/provider do Hermes.

## Session behavior {#session-behavior}

Sessões ACP são rastreadas pelo session manager in-memory do adapter ACP enquanto o servidor roda.

Cada sessão armazena:

- session ID
- working directory
- selected model
- current conversation history
- cancel event

O `AIAgent` subjacente ainda usa os caminhos normais de persistência/logging do Hermes, mas ACP `list/load/resume/fork` são escopados ao processo do servidor ACP em execução no momento.

## Working directory behavior {#working-directory-behavior}

Sessões ACP vinculam o cwd do editor ao task ID do Hermes para file e terminal tools rodarem relativos ao workspace do editor, não ao cwd do processo do servidor.

## Approvals {#approvals}

Comandos de terminal perigosos podem ser roteados de volta ao editor como approval prompts. Opções de approval ACP são mais simples que o fluxo CLI:

- allow once
- allow always
- deny

Em timeout ou erro, a approval bridge nega a requisição.

### Session-scoped edit auto-approval {#session-scoped-edit-auto-approval}

ACP expõe um terceiro tier entre *allow once* e *allow always*: **Allow for session**. Escolher no prompt de permissão do editor registra a approval só dentro da sessão ACP atual — todo comando correspondente subsequente nesta sessão passa sem prompt, mas uma nova sessão ACP (ou reiniciar o editor) reseta a lista e re-prompta na primeira vez.

| Option | Editor label | Scope | Persisted across restarts |
|---|---|---|---|
| `allow_once` | Allow once | This one tool call | No |
| `allow_session` | Allow for session | All matching calls in this ACP session | No — cleared when the session ends |
| `allow_always` | Allow always | All future sessions | Yes (written to the Hermes permanent allowlist) |
| `deny` | Deny | This one tool call | No |

`allow_session` é o padrão certo para workflow de editor onde você confia no agente pela duração de uma tarefa mas não quer conceder entrada de allowlist de longa duração. O trade-off de segurança é direto: quanto mais amplo o escopo, menos o editor interrompe você, e mais dano um agente mal-comportado (ou prompt injection) pode fazer antes de você notar. Comece com `allow_once` para comandos desconhecidos; promova para `allow_session` quando vir o agente rodar o mesmo padrão corretamente algumas vezes; reserve `allow_always` para comandos verdadeiramente idempotentes em que você confia para sempre (por exemplo, `git status`).

A bridge ACP mapeia essas opções sobre a semântica interna de approval do Hermes — `allow_always` grava entrada permanente de allowlist da mesma forma que a CLI, enquanto `allow_session` só afeta o cache de approval in-process da sessão ACP atual.

## Troubleshooting {#troubleshooting}

### ACP agent does not appear in the editor {#acp-agent-does-not-appear-in-the-editor}

Verifique:

- No Zed, abra o ACP Registry com `zed: acp registry` e busque **Hermes Agent**.
- Para desenvolvimento manual/local, verifique se o comando custom `agent_servers` aponta para `hermes acp`.
- Hermes está instalado e no seu PATH.
- O extra ACP está instalado (`pip install -e '.[acp]'`).
- `uv` está instalado se lançando pela entrada oficial do registry Zed.

### ACP starts but immediately errors {#acp-starts-but-immediately-errors}

Tente estas checagens:

```bash
hermes acp --version
hermes acp --check
hermes doctor
hermes status
```

### Missing credentials {#missing-credentials}

Modo ACP usa o setup de provider existente do Hermes. Configure credenciais com:

```bash
hermes model
```

ou editando `~/.hermes/.env`. Clientes registry também podem disparar o fluxo terminal auth do Hermes, que roda o mesmo setup interativo de provider/model.

### Zed registry launcher cannot find uv {#zed-registry-launcher-cannot-find-uv}

Instale `uv` pelos docs oficiais de instalação do uv, depois retente a thread Hermes Agent no Zed.

## See also {#see-also}

- [ACP Internals](../../developer-guide/acp-internals.md)
- [Provider Runtime Resolution](../../developer-guide/provider-runtime.md)
- [Tools Runtime](../../developer-guide/tools-runtime.md)
