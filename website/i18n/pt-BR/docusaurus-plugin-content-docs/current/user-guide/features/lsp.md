---
sidebar_position: 16
title: "LSP — Diagnósticos Semânticos"
description: "Language servers reais (pyright, gopls, rust-analyzer, …) integrados à verificação de lint pós-escrita usada por write_file e patch."
---

# Language Server Protocol (LSP)

O Hermes executa language servers completos — pyright, gopls, rust-analyzer,
typescript-language-server, clangd e mais ~20 — como subprocessos em segundo plano
e alimenta seus diagnósticos semânticos na verificação de lint pós-escrita
usada por `write_file` e `patch`. Quando o agente edita um
arquivo, ele vê exatamente os erros que a edição introduziu — não só
erros de sintaxe, mas **erros de tipo, nomes indefinidos, imports ausentes
e problemas semânticos em todo o projeto** que o language server detecta.

É a mesma arquitetura que agentes de código de alto nível usam. O Hermes
entrega isso autocontido: sem host de editor, sem plugins para
instalar, sem daemon separado para gerenciar.

## Quando o LSP roda {#when-lsp-runs}

O LSP é condicionado à **detecção de workspace git**. Quando o diretório de trabalho
do agente (ou o arquivo sendo editado) está dentro de um repositório git, o LSP
roda contra esse workspace. Quando nenhum dos dois está em um repo git, o LSP
permanece inativo — útil para gateways de mensagens onde o cwd é o
diretório home do usuário e não há projeto para diagnosticar.

A verificação é em camadas: verificação de sintaxe in-process primeiro (microssegundos),
depois diagnósticos LSP quando a sintaxe está limpa. Um language server
instável ou ausente nunca pode quebrar uma escrita — todo caminho de falha LSP
cai silenciosamente para o resultado só de sintaxe.

Concretamente, em cada `write_file` ou `patch` bem-sucedido:

1. O Hermes captura uma baseline dos diagnósticos atuais do arquivo.
2. Executa a escrita.
3. Reconsulta o language server, filtra diagnósticos que já
   estavam na baseline e expõe só os novos.

O agente vê saída como:

```
{
  "bytes_written": 42,
  "dirs_created": false,
  "lint": {"status": "ok", "output": ""},
  "lsp_diagnostics": "LSP diagnostics introduced by this edit:\n<diagnostics file=\"/path/to/foo.py\">\nERROR [42:5] Cannot find name 'foo' [reportUndefinedVariable] (Pyright)\nERROR [50:1] Argument of type \"str\" is not assignable to \"int\" [reportArgumentType] (Pyright)\n</diagnostics>"
}
```

O campo `lint` carrega o resultado da verificação de sintaxe (parse
in-process de microssegundos via `ast.parse`, `json.loads`, etc.); o
campo `lsp_diagnostics` carrega os diagnósticos semânticos do
language server real. Dois canais, sinais independentes — o
agente vê um arquivo sintaticamente limpo com problemas semânticos como
``lint: ok`` mais um ``lsp_diagnostics`` preenchido.

## Linguagens suportadas {#supported-languages}

| Linguagem | Servidor | Auto-instalação |
|----------|--------|--------------|
| Python | `pyright-langserver` | npm |
| TypeScript / JavaScript / JSX / TSX | `typescript-language-server` | npm |
| Vue | `@vue/language-server` | npm |
| Svelte | `svelte-language-server` | npm |
| Astro | `@astrojs/language-server` | npm |
| Go | `gopls` | `go install` |
| Rust | `rust-analyzer` | manual (rustup) |
| C / C++ | `clangd` | manual (LLVM) |
| Bash / Zsh | `bash-language-server` | npm |
| YAML | `yaml-language-server` | npm |
| Lua | `lua-language-server` | manual (GitHub releases) |
| PHP | `intelephense` | npm |
| OCaml | `ocaml-lsp` | manual (opam) |
| Dockerfile | `dockerfile-language-server-nodejs` | npm |
| Terraform | `terraform-ls` | manual |
| Dart | `dart language-server` | manual (dart sdk) |
| Haskell | `haskell-language-server` | manual (ghcup) |
| Julia | `julia` + LanguageServer.jl | manual |
| Clojure | `clojure-lsp` | manual |
| Nix | `nixd` | manual |
| Zig | `zls` | manual |
| Gleam | `gleam lsp` | manual (gleam install) |
| Elixir | `elixir-ls` | manual |
| Prisma | `prisma language-server` | manual |
| Kotlin | `kotlin-language-server` | manual |
| Java | `jdtls` | manual |
| PowerShell | `PowerShellEditorServices` (`pwsh` host) | manual (release zip) |

Para entradas "manual", instale o server pelo gerenciador de toolchain
que fizer sentido para aquela linguagem (rustup, ghcup, opam, brew,
…). O Hermes detecta automaticamente o binário no PATH ou em
`<HERMES_HOME>/lsp/bin/`.

### PowerShell {#powershell}

PowerShellEditorServices não é um binário único — é um bundle de módulo PowerShell
lançado por um host `pwsh` (PowerShell 7+) ou `powershell`.
Setup:

1. Instale [PowerShell](https://github.com/PowerShell/PowerShell) para que
   `pwsh` (ou `powershell` no Windows) esteja no PATH.
2. Baixe o zip da release mais recente em
   [PowerShellEditorServices releases](https://github.com/PowerShell/PowerShellEditorServices/releases)
   e extraia.
3. Aponte o Hermes para o bundle extraído — o diretório que contém
   `PowerShellEditorServices/Start-EditorServices.ps1`. Ou:
   - defina `lsp.servers.powershell.command: ["/path/to/bundle"]` em
     `config.yaml`, ou
   - extraia para `<HERMES_HOME>/lsp/PowerShellEditorServices`, ou
   - exporte `PSES_BUNDLE_PATH=/path/to/bundle`.

`hermes lsp status` reporta `installed` assim que `pwsh` for encontrado; se o
bundle estiver ausente você verá um aviso único nos logs com o
link de download.

Alguns servers são instalados junto com uma dependência peer que o npm
não puxa automaticamente. O caso atual é `typescript-language-server`,
que exige o SDK `typescript` importável da mesma
árvore `node_modules` — o Hermes instala ambos os pacotes juntos quando você
executa `hermes lsp install typescript` ou a auto-instalação dispara no primeiro
uso.

## CLI {#cli}

```
hermes lsp status          # service state + per-server install status
hermes lsp list            # registry, optionally --installed-only
hermes lsp install <id>    # eagerly install one server
hermes lsp install-all     # try every server with a known recipe
hermes lsp restart         # tear down running clients
hermes lsp which <id>      # print resolved binary path
```

`hermes lsp status` é o melhor ponto de partida — mostra quais
linguagens terão diagnósticos semânticos hoje e quais precisam de um
binário instalado.

## Configuração {#configuration}

Os padrões funcionam para setups típicos; nada a definir se os binários
estiverem no PATH.

```yaml
# config.yaml
lsp:
  # Master toggle. Disabling skips the entire subsystem — no servers
  # spawn, no background event loop runs.
  enabled: true

  # How long to wait for diagnostics after each write.
  wait_mode: document      # "document" or "full"
  # Max seconds to wait for the server to re-check the file after an
  # edit. Only *fresh* diagnostics (produced for the post-edit
  # content) are ever reported; if the server doesn't finish within
  # this budget, the edit reports "no LSP data" rather than stale
  # errors from before the edit. Raise this for slow servers on big
  # projects (tsserver, rust-analyzer mid-indexing).
  wait_timeout: 5.0

  # How to handle missing server binaries.
  #   auto    — install via npm/pip/go install into <HERMES_HOME>/lsp/bin
  #   manual  — only use binaries already on PATH
  install_strategy: auto

  # Per-server overrides (all optional).
  servers:
    pyright:
      disabled: false
      command: ["/abs/path/to/pyright-langserver", "--stdio"]
      env: { PYRIGHT_LOG_LEVEL: "info" }
      initialization_options:
        python:
          analysis:
            typeCheckingMode: "strict"
    typescript:
      disabled: true       # skip TS even when its extensions match
```

### Chaves por server {#per-server-keys}

* `disabled: true` — pula este server inteiramente mesmo quando suas
  extensões correspondem a um arquivo.
* `command: [bin, ...args]` — fixa um caminho de binário customizado. Ignora
  auto-install.
* `env: {KEY: value}` — variáveis de ambiente extras passadas ao processo spawnado.
* `initialization_options: {...}` — mesclado no payload LSP
  `initializationOptions` enviado no handshake
  `initialize`. Específico do server; consulte a documentação do language server.

## Locais de instalação {#installation-locations}

Quando `install_strategy: auto`, o Hermes instala binários em
`<HERMES_HOME>/lsp/bin/`. Pacotes NPM vão para
`<HERMES_HOME>/lsp/node_modules/` com symlinks bin um nível acima.
Binários Go vêm de `go install` com `GOBIN` apontando para o
diretório de staging.

Nada é instalado em `/usr/local/`, `~/.local/` ou qualquer outro
local compartilhado — o diretório de staging é totalmente do Hermes e é
removido quando você reseta o perfil.

## Características de desempenho {#performance-characteristics}

Servers LSP são **spawnados sob demanda** no primeiro uso. Editar um arquivo Python
em um projeto que nunca viu tráfego `.py` spawna pyright; o
spawn leva 1–3 segundos para a maioria dos servers (rust-analyzer pode levar 10+
em um projeto frio). Edições subsequentes no mesmo workspace reutilizam
o server em execução.

A camada LSP adiciona alguns milissegundos a escritas limpas quando nenhum
diagnóstico é emitido. Quando diagnósticos são emitidos, o orçamento de espera
é `wait_timeout` segundos — tipicamente o server responde em
dezenas de milissegundos para pyright/tsserver e alguns segundos para
rust-analyzer no meio da indexação.

Diagnósticos são **limitados por frescor**: um resultado só conta quando o
server o produziu para o conteúdo da edição atual (um push
`publishDiagnostics` no/depois da mudança, ou uma requisição pull
respondida depois). Servers lentos que ainda não re-verificaram resultam
em "no data" para aquela edição — nunca em erros de ontem sendo
re-reportados como atuais.

Servers ficam vivos durante a vida do processo Hermes. Não há
reaper de idle-timeout — o custo de reiniciar o índice do server
a cada escrita seria muito maior que manter o daemon.

## Desativar {#disabling}

Defina `lsp.enabled: false` em `config.yaml` para desativar todo o
subsistema. A verificação pós-escrita cai para a verificação de sintaxe
in-process (`ast.parse` para Python, `json.loads` para JSON, etc.), que
permanece inalterada das versões anteriores.

Para desativar uma linguagem sem desligar a camada inteira:

```yaml
lsp:
  servers:
    rust-analyzer:
      disabled: true
```

## Solução de problemas {#troubleshooting}

**`hermes lsp status` mostra um server como "missing"**

O binário não está no PATH nem em `<HERMES_HOME>/lsp/bin/`. Execute
`hermes lsp install <server_id>` para tentar auto-install, ou
instale o binário manualmente pela toolchain normal da linguagem.

**Seção `Backend warnings` em `hermes lsp status`**

Alguns servers vêm como wrappers finos em torno de um CLI externo para diagnósticos
reais — spawnam limpo e aceitam requisições, mas nunca emitem
erros quando o binário sidecar está ausente. O caso mais comum é
`bash-language-server`, que delega diagnósticos a `shellcheck`.
Quando `hermes lsp status` mostra uma seção `Backend warnings`, instale
a ferramenta nomeada pelo gerenciador de pacotes do seu SO:

```
apt install shellcheck      # Debian / Ubuntu
brew install shellcheck     # macOS
scoop install shellcheck    # Windows
```

O mesmo aviso é logado uma vez no spawn do server em
`~/.hermes/logs/agent.log`.

**Server inicia mas nunca retorna diagnósticos**

Verifique `~/.hermes/logs/agent.log` por entradas `[agent.lsp.client]` —
tanto stderr do language server quanto erros de protocolo vão
para lá. Alguns servers (rust-analyzer especialmente) precisam terminar uma
indexação em todo o projeto antes de emitir diagnósticos por arquivo; a primeira
edição após o start do server pode completar sem diagnósticos, com
edições subsequentes capturando-os.

**Server crashou**

Um server crashado é adicionado ao broken-set e não será retentado pelo
resto da sessão. Execute `hermes lsp restart` para limpar o set;
a próxima edição re-spawna.

**Editando um arquivo fora de qualquer repo git**

Por design, LSP só roda dentro de um repositório git. Se o projeto ainda não
foi inicializado, execute `git init` para habilitar diagnósticos LSP. Caso contrário o
fallback só de sintaxe in-process se aplica.
